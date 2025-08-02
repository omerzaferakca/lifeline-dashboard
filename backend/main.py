import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample, butter, filtfilt, find_peaks
import neurokit2 as nk
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ecg_analysis.log', encoding='utf-8'), # Dosyaya UTF-8 yaz
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ############################################################################
# ##                          DATABASE OPERATIONS                           ##
# ############################################################################

class DatabaseManager:
    """Database operations manager for patient and ECG file management."""
    
    def __init__(self, db_file: str = 'database.db'):
        self.db_file = db_file
        self.init_db()
    
    def get_connection(self):
        """Create and return database connection."""
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_db(self):
        """Initialize database tables."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Patients table (değişiklik yok)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS patients (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, tc TEXT UNIQUE NOT NULL,
                        age INTEGER, gender TEXT, phone TEXT, medications TEXT,
                        complaints TEXT, created_at DATETIME, updated_at DATETIME
                    )''')
                
                # --- DÜZELTME: `duration` sütunu CREATE komutuna eklendi ---
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS ekg_files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id INTEGER NOT NULL, file_name TEXT NOT NULL,
                        uploaded_at DATETIME, sampling_rate INTEGER, duration REAL,
                        ekg_data BLOB NOT NULL, analysis_results TEXT,
                        FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE
                    )''')
                
                # --- DÜZELTME: Mevcut veritabanları için `duration` sütununu ekleyen kontrol ---
                try:
                    cursor.execute("SELECT duration FROM ekg_files LIMIT 1")
                except sqlite3.OperationalError:
                    # Sütun yoksa, ekle
                    cursor.execute("ALTER TABLE ekg_files ADD COLUMN duration REAL")
                    logger.info("Mevcut 'ekg_files' tablosuna 'duration' sütunu eklendi.")
                
                # Analysis results table (değişiklik yok)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_results (
                        id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, analysis_type TEXT,
                        results TEXT NOT NULL, confidence_score REAL, analyzed_at DATETIME,
                        FOREIGN KEY (file_id) REFERENCES ekg_files (id) ON DELETE CASCADE
                    )''')
                
                conn.commit()
                logger.info("Veritabanı başarıyla başlatıldı")
                
        except Exception as e:
            logger.error(f"Veritabanı başlatma hatası: {e}")
            raise


# ############################################################################
# ##                          CNN MODEL ARCHITECTURE                        ##
# ############################################################################

class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

class ConvNormPool(nn.Module):
    """Convolutional block with normalization and pooling."""
    def __init__(self, input_size: int, hidden_size: int, kernel_size: int, dropout_rate: float = 0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv1d(input_size, hidden_size, kernel_size, padding=kernel_size//2)
        self.conv_2 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size//2)
        self.conv_3 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size//2)
        self.swish_1, self.swish_2, self.swish_3 = Swish(), Swish(), Swish()
        
        # --- DÜZELTME: İsim `model.pth` dosyasıyla eşleşecek şekilde değiştirildi ---
        # `norm_1` -> `normalization_1`
        self.normalization_1 = nn.BatchNorm1d(hidden_size)
        self.normalization_2 = nn.BatchNorm1d(hidden_size)
        self.normalization_3 = nn.BatchNorm1d(hidden_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = nn.MaxPool1d(kernel_size=2)
    
    def forward(self, x):
        conv1 = self.conv_1(x)
        # --- DÜZELTME: Doğru katman ismi kullanıldı ---
        x = self.normalization_1(conv1)
        x = self.swish_1(x); x = self.dropout(x)
        x = self.conv_2(x)
        x = self.normalization_2(x)
        x = self.swish_2(x); x = self.dropout(x)
        conv3 = self.conv_3(x)
        x = self.normalization_3(conv1 + conv3)
        x = self.swish_3(x)
        return self.pool(x)


class EnhancedCNN(nn.Module):
    """Enhanced CNN model for ECG classification."""
    
    def __init__(self, input_size: int = 1, hid_size: int = 128, kernel_size: int = 5, 
                 num_classes: int = 5, dropout_rate: float = 0.1):
        super().__init__()
        
        self.conv1 = ConvNormPool(input_size, hid_size, kernel_size, dropout_rate)
        self.conv2 = ConvNormPool(hid_size, hid_size//2, kernel_size, dropout_rate)
        self.conv3 = ConvNormPool(hid_size//2, hid_size//4, kernel_size, dropout_rate)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hid_size//4, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)


# ############################################################################
# ##                    ADVANCED ECG PROCESSOR CLASS                        ##
# ############################################################################

class ECGProcessor:
    """Advanced ECG signal processing and analysis."""
    
    def __init__(self, model_path: str = "model.pth", device: str = "cpu"):
        self.device = device
        self.model = None
        self.model_path = Path(model_path)
        self.class_names = {
            0: "Normal (N)", 
            1: "Supraventricular (S)", 
            2: "Ventricular (V)", 
            3: "Fusion (F)", 
            4: "Unknown (Q)"
        }
        self.class_descriptions = {
            0: "Normal heartbeat - regular rhythm",
            1: "Supraventricular ectopic beat - originates above ventricles",
            2: "Ventricular ectopic beat - originates in ventricles",
            3: "Fusion beat - combination of normal and ectopic",
            4: "Unknown or unclassifiable beat"
        }
        self.load_model()
    
    def load_model(self):
        """Load the trained CNN model."""
        try:
            if not self.model_path.exists():
                logger.warning(f"Model file not found: {self.model_path}")
                self.model = None
                return
            
            self.model = EnhancedCNN(num_classes=len(self.class_names), hid_size=128)
            state_dict = torch.load(self.model_path, map_location=self.device)
            
            # Handle potential key mismatches
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            self.model.load_state_dict(state_dict)
            self.model.eval().to(self.device)
            logger.info(f"Model loaded successfully: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            self.model = None
    
    def apply_filters(self, signal: np.ndarray, fs: int) -> np.ndarray:
        """Apply advanced signal filtering."""
        try:
            # Remove baseline wander using high-pass filter (0.5 Hz)
            nyquist = fs / 2
            low_freq = 0.5 / nyquist
            high_freq = min(40.0, nyquist - 1) / nyquist
            
            if low_freq >= high_freq:
                logger.warning("Invalid filter frequencies, skipping filtering")
                return signal
            
            # Bandpass filter (0.5-40 Hz for ECG)
            b, a = butter(4, [low_freq, high_freq], btype='band')
            filtered_signal = filtfilt(b, a, signal)
            
            # Notch filter for 50Hz power line interference
            if fs > 100:  # Only apply if sampling rate is high enough
                notch_freq = 50.0 / nyquist
                if notch_freq < 0.95:  # Only if frequency is valid
                    b_notch, a_notch = butter(2, [notch_freq - 0.01, notch_freq + 0.01], btype='bandstop')
                    filtered_signal = filtfilt(b_notch, a_notch, filtered_signal)
            
            return filtered_signal
            
        except Exception as e:
            logger.warning(f"Filtering failed: {e}, using original signal")
            return signal
    
    def robust_r_peak_detection(self, signal: np.ndarray, fs: int) -> List[int]:
        """Çoklu yöntem kullanarak güçlü R-peak tespiti."""
        r_peaks = []
        
        # 1. NeuroKit2 yöntemleri
        neurokit_methods = ['neurokit', 'pantompkins1985', 'hamilton2002']
        
        for method in neurokit_methods:
            try:
                _, rpeaks_info = nk.ecg_peaks(signal, sampling_rate=fs, method=method)
                peaks = rpeaks_info.get('ECG_R_Peaks', [])
                
                if len(peaks) > 2:  # En az 3 peak gerekli
                    r_peaks = list(peaks)
                    logger.info(f"R-peaks detected using {method}: {len(r_peaks)}")
                    break
                    
            except Exception as e:
                logger.warning(f"R-peak detection failed with {method}: {e}")
                continue
        
        # 2. Eğer NeuroKit başarısızsa, manuel scipy yöntemi
        if len(r_peaks) < 3:
            try:
                # Sinyali normalleştir
                normalized_signal = (signal - np.mean(signal)) / np.std(signal)
                
                # Adaptif eşik ile peak bulma
                mean_amplitude = np.mean(np.abs(normalized_signal))
                threshold = mean_amplitude * 0.6  # Daha düşük eşik
                
                # Minimum mesafe (RR interval ~ 0.4-2.0 saniye)
                min_distance = int(0.4 * fs)  # 0.4 saniye
                
                peaks, _ = find_peaks(normalized_signal, 
                                    height=threshold, 
                                    distance=min_distance)
                
                if len(peaks) > 2:
                    r_peaks = peaks.tolist()
                    logger.info(f"R-peaks detected using scipy method: {len(r_peaks)}")
                
            except Exception as e:
                logger.warning(f"Manual R-peak detection failed: {e}")
        
        # 3. Son çare: basit maksimum bulma
        if len(r_peaks) < 3:
            try:
                # Sinyali segmentlere böl ve her segmentte maksimum bul
                segment_length = int(1.5 * fs)  # 1.5 saniye segmentler
                
                for i in range(0, len(signal) - segment_length, segment_length):
                    segment = signal[i:i + segment_length]
                    local_max = np.argmax(segment) + i
                    
                    # Çok yakın peak'leri filtrele
                    if not r_peaks or (local_max - r_peaks[-1]) > int(0.4 * fs):
                        r_peaks.append(local_max)
                
                if len(r_peaks) > 2:
                    logger.info(f"R-peaks detected using segmentation method: {len(r_peaks)}")
                
            except Exception as e:
                logger.warning(f"Segmentation R-peak detection failed: {e}")
        
        # Sonuçları temizle ve doğrula
        if r_peaks:
            r_peaks = [int(p) for p in r_peaks if 0 <= p < len(signal)]
            r_peaks = sorted(set(r_peaks))  # Duplikatları kaldır ve sırala
            
            # RR intervallerini kontrol et (fizyolojik sınırlar)
            if len(r_peaks) > 1:
                rr_intervals = np.diff(r_peaks) / fs
                valid_rr = (rr_intervals > 0.3) & (rr_intervals < 3.0)  # 20-200 BPM arası
                
                if np.sum(valid_rr) < len(rr_intervals) * 0.5:  # %50'den az geçerliyse
                    logger.warning("Detected R-peaks have invalid RR intervals")
        
        return r_peaks
    
    def extract_beats(self, signal: np.ndarray, r_peaks: List[int], window: int = 187) -> np.ndarray:
        """Extract individual heartbeats around R-peaks with improved validation."""
        if not r_peaks:
            logger.warning("No R-peaks provided for beat extraction")
            return np.array([])
        
        half_window = window // 2
        beats = []
        valid_beat_count = 0
        
        logger.info(f"Extracting beats from {len(r_peaks)} R-peaks with window size {window}")
        
        for i, r_peak in enumerate(r_peaks):
            try:
                start = max(0, r_peak - half_window)
                end = min(len(signal), r_peak + half_window)
                
                # Skip if the beat window is too small
                if (end - start) < window * 0.7:  # At least 70% of expected window
                    logger.debug(f"Skipping beat {i}: insufficient data around R-peak {r_peak}")
                    continue
                
                beat = signal[start:end]
                
                # Pad or truncate to fixed window size
                if len(beat) < window:
                    # Pad with edge values
                    pad_before = (window - len(beat)) // 2
                    pad_after = window - len(beat) - pad_before
                    beat = np.pad(beat, (pad_before, pad_after), mode='edge')
                elif len(beat) > window:
                    beat = beat[:window]
                
                # Validate beat quality
                beat_std = np.std(beat)
                if beat_std < 1e-6:  # Avoid flat signals
                    logger.debug(f"Skipping beat {i}: too flat (std={beat_std})")
                    continue
                
                beats.append(beat)
                valid_beat_count += 1
                
            except Exception as e:
                logger.warning(f"Error extracting beat {i} at R-peak {r_peak}: {e}")
                continue
        
        logger.info(f"Successfully extracted {valid_beat_count} valid beats from {len(r_peaks)} R-peaks")
        return np.array(beats)
    
    def predict_beats(self, beats: np.ndarray) -> List[Dict]:
        """Predict beat classifications using the CNN model with improved error handling."""
        if self.model is None:
            logger.warning("Model not available for beat prediction")
            return []
        
        if len(beats) == 0:
            logger.warning("No beats provided for prediction")
            return []
        
        predictions = []
        successful_predictions = 0
        
        logger.info(f"Starting prediction for {len(beats)} beats")
        
        try:
            with torch.no_grad():
                for i, beat in enumerate(beats):
                    try:
                        # Normalize beat
                        beat_std = np.std(beat)
                        if beat_std < 1e-8:
                            logger.debug(f"Skipping flat beat {i}")
                            continue
                        
                        beat_norm = (beat - np.mean(beat)) / beat_std
                        
                        # Convert to tensor
                        tensor = torch.FloatTensor(beat_norm).unsqueeze(0).unsqueeze(0).to(self.device)
                        
                        # Validate tensor
                        if not torch.isfinite(tensor).all():
                            logger.debug(f"Skipping beat {i}: invalid tensor values")
                            continue
                        
                        # Get prediction
                        probs = self.model(tensor)
                        confidence, predicted_class = torch.max(probs, dim=1)
                        
                        # Get all class probabilities
                        class_probs = {
                            self.class_names[j]: float(probs[0][j].item()) 
                            for j in range(len(self.class_names))
                        }
                        
                        predictions.append({
                            "beat_id": i,
                            "predicted_class": int(predicted_class.item()),
                            "class_name": self.class_names.get(predicted_class.item(), "Unknown"),
                            "description": self.class_descriptions.get(predicted_class.item(), "Unknown"),
                            "confidence": float(confidence.item()),
                            "class_probabilities": class_probs
                        })
                        
                        successful_predictions += 1
                        
                    except Exception as e:
                        logger.warning(f"Error predicting beat {i}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Critical error in beat prediction: {e}")
        
        logger.info(f"Successfully predicted {successful_predictions} out of {len(beats)} beats")
        return predictions
    
    def calculate_advanced_clinical_metrics(self, signal: np.ndarray, r_peaks: List[int], fs: int) -> Dict:
        """Gelişmiş klinik ECG metrikleri hesapla (HRV dahil)."""
        metrics = {
            "heart_rate_bpm": None,
            "heart_rate_variability": {},  # HRV metrikleri
            "qrs_duration_ms": None,
            "pr_interval_ms": None,
            "qt_interval_ms": None,
            "rr_intervals_ms": None
        }
        
        try:
            if len(r_peaks) < 2:
                return metrics
            
            # RR intervallerini hesapla
            rr_intervals = np.diff(r_peaks) / fs  # saniye cinsinden
            rr_intervals_ms = rr_intervals * 1000  # milisaniye cinsinden
            
            # Kalp hızı hesaplama
            if len(rr_intervals) > 0:
                hr = 60.0 / np.mean(rr_intervals)
                metrics["heart_rate_bpm"] = float(hr)
                metrics["rr_intervals_ms"] = rr_intervals_ms.tolist()
                
                # ===== HRV ANALİZİ =====
                if len(rr_intervals_ms) > 1:
                    hrv_metrics = {}
                    
                    # Time-domain HRV metrics
                    # RMSSD (Root Mean Square of Successive Differences)
                    rr_diff = np.diff(rr_intervals_ms)
                    rmssd = np.sqrt(np.mean(rr_diff ** 2))
                    hrv_metrics["rmssd_ms"] = float(rmssd)
                    
                    # SDNN (Standard Deviation of NN intervals)
                    sdnn = np.std(rr_intervals_ms)
                    hrv_metrics["sdnn_ms"] = float(sdnn)
                    
                    # pNN50 (Percentage of successive RR intervals that differ by more than 50ms)
                    nn50 = np.sum(np.abs(rr_diff) > 50)
                    pnn50 = (nn50 / len(rr_diff)) * 100
                    hrv_metrics["pnn50_percent"] = float(pnn50)
                    
                    # SDSD (Standard Deviation of Successive Differences)
                    sdsd = np.std(rr_diff)
                    hrv_metrics["sdsd_ms"] = float(sdsd)
                    
                    # Triangular Index (approximate)
                    hist, _ = np.histogram(rr_intervals_ms, bins=50)
                    tri_index = len(rr_intervals_ms) / np.max(hist) if np.max(hist) > 0 else 0
                    hrv_metrics["triangular_index"] = float(tri_index)
                    
                    # Frequency-domain HRV (basit yaklaşım)
                    if len(rr_intervals_ms) > 10:
                        try:
                            # FFT tabanlı güç spektral yoğunluğu
                            freqs = np.fft.fftfreq(len(rr_intervals_ms), d=np.mean(rr_intervals))
                            power = np.abs(np.fft.fft(rr_intervals_ms - np.mean(rr_intervals_ms))) ** 2
                            
                            # VLF (0.003-0.04 Hz), LF (0.04-0.15 Hz), HF (0.15-0.4 Hz)
                            vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
                            lf_mask = (freqs >= 0.04) & (freqs < 0.15)
                            hf_mask = (freqs >= 0.15) & (freqs < 0.4)
                            
                            vlf_power = np.sum(power[vlf_mask]) if np.any(vlf_mask) else 0
                            lf_power = np.sum(power[lf_mask]) if np.any(lf_mask) else 0
                            hf_power = np.sum(power[hf_mask]) if np.any(hf_mask) else 0
                            
                            total_power = vlf_power + lf_power + hf_power
                            
                            if total_power > 0:
                                hrv_metrics["vlf_power"] = float(vlf_power)
                                hrv_metrics["lf_power"] = float(lf_power)
                                hrv_metrics["hf_power"] = float(hf_power)
                                hrv_metrics["lf_hf_ratio"] = float(lf_power / hf_power) if hf_power > 0 else 0
                                hrv_metrics["total_power"] = float(total_power)
                        
                        except Exception as e:
                            logger.warning(f"Frequency-domain HRV calculation failed: {e}")
                    
                    metrics["heart_rate_variability"] = hrv_metrics
            
            # Wave delineation for intervals
            try:
                _, waves = nk.ecg_delineate(signal, r_peaks, sampling_rate=fs, method="dwt")
                
                # QRS duration
                q_peaks = waves.get('ECG_Q_Peaks')
                s_peaks = waves.get('ECG_S_Peaks')
                if q_peaks is not None and s_peaks is not None:
                    q_peaks = np.array(q_peaks)[~np.isnan(q_peaks)]
                    s_peaks = np.array(s_peaks)[~np.isnan(s_peaks)]
                    if len(q_peaks) > 0 and len(s_peaks) > 0:
                        qrs_durations = (s_peaks[:min(len(q_peaks), len(s_peaks))] - 
                                       q_peaks[:min(len(q_peaks), len(s_peaks))]) / fs * 1000
                        metrics["qrs_duration_ms"] = float(np.mean(qrs_durations))
                
                # PR interval
                p_onsets = waves.get('ECG_P_Onsets')
                if p_onsets is not None and len(r_peaks) > 0:
                    p_onsets = np.array(p_onsets)[~np.isnan(p_onsets)]
                    if len(p_onsets) > 0:
                        pr_intervals = (np.array(r_peaks[:len(p_onsets)]) - p_onsets) / fs * 1000
                        metrics["pr_interval_ms"] = float(np.mean(pr_intervals[pr_intervals > 0]))
                
                # QT interval
                t_offsets = waves.get('ECG_T_Offsets')
                if q_peaks is not None and t_offsets is not None:
                    t_offsets = np.array(t_offsets)[~np.isnan(t_offsets)]
                    if len(t_offsets) > 0 and len(q_peaks) > 0:
                        qt_intervals = (t_offsets[:min(len(q_peaks), len(t_offsets))] - 
                                      q_peaks[:min(len(q_peaks), len(t_offsets))]) / fs * 1000
                        metrics["qt_interval_ms"] = float(np.mean(qt_intervals[qt_intervals > 0]))
                        
            except Exception as e:
                logger.warning(f"Wave delineation failed: {e}")
        
        except Exception as e:
            logger.error(f"Clinical metrics calculation error: {e}")
        
        return metrics
    
    def detect_anomalies(self, predictions: List[Dict], clinical_metrics: Dict) -> Dict:
        """Gelişmiş anomali ve aritmi tespiti."""
        anomalies = {
            "arrhythmias": [],
            "morphology_abnormalities": [],
            "conduction_abnormalities": [],
            "severity": "Normal"
        }
        
        try:
            if not predictions:
                return anomalies
            
            # Beat sınıflandırma analizi
            beat_counts = {}
            total_beats = len(predictions)
            
            for pred in predictions:
                class_name = pred['class_name']
                beat_counts[class_name] = beat_counts.get(class_name, 0) + 1
            
            beat_percentages = {k: (v/total_beats)*100 for k, v in beat_counts.items()}
            
            # Ventriküler aritmi tespiti
            v_count = beat_counts.get("Ventricular (V)", 0)
            v_percentage = beat_percentages.get("Ventricular (V)", 0)
            
            if v_count > 0:
                if v_percentage > 30:
                    anomalies["arrhythmias"].append({
                        "type": "Çok Sık Ventriküler Ektopi",
                        "severity": "Kritik",
                        "description": f"Atımların %{v_percentage:.1f}'i ventriküler kökenli",
                        "recommendation": "Acil kardiyoloji konsültasyonu gerekli"
                    })
                    anomalies["severity"] = "Kritik"
                elif v_percentage > 10:
                    anomalies["arrhythmias"].append({
                        "type": "Sık Ventriküler Ektopi",
                        "severity": "Yüksek",
                        "description": f"Atımların %{v_percentage:.1f}'i ventriküler kökenli",
                        "recommendation": "24 saatlik Holter ve kardiyoloji değerlendirmesi"
                    })
                    if anomalies["severity"] == "Normal":
                        anomalies["severity"] = "Yüksek"
                elif v_count > 3:
                    anomalies["arrhythmias"].append({
                        "type": "Ventriküler Ektopi",
                        "severity": "Orta",
                        "description": f"{v_count} ventriküler ektopik atım tespit edildi",
                        "recommendation": "Elektrolit paneli ve followup önerilir"
                    })
                    if anomalies["severity"] == "Normal":
                        anomalies["severity"] = "Orta"
            
            # Supraventriküler aritmi tespiti
            s_count = beat_counts.get("Supraventricular (S)", 0)
            s_percentage = beat_percentages.get("Supraventricular (S)", 0)
            
            if s_percentage > 25:
                anomalies["arrhythmias"].append({
                    "type": "Sık Supraventriküler Ektopi",
                    "severity": "Orta",
                    "description": f"Atımların %{s_percentage:.1f}'i supraventriküler kökenli",
                    "recommendation": "Holter takibi ve elektrolit değerlendirmesi"
                })
                if anomalies["severity"] == "Normal":
                    anomalies["severity"] = "Orta"
            
            # Fusion beat analizi
            f_count = beat_counts.get("Fusion (F)", 0)
            if f_count > 2:
                anomalies["morphology_abnormalities"].append({
                    "type": "Fusion Beat'ler",
                    "severity": "Orta",
                    "description": f"{f_count} fusion beat tespit edildi",
                    "recommendation": "İleti sistem değerlendirmesi önerilir"
                })
            
            # Klinik metrik anomalileri
            hr = clinical_metrics.get("heart_rate_bpm")
            if hr:
                if hr > 120:
                    anomalies["arrhythmias"].append({
                        "type": "Taşikardi",
                        "severity": "Yüksek" if hr > 150 else "Orta",
                        "description": f"Kalp hızı: {hr:.0f} bpm",
                        "recommendation": "Hızla kardiyoloji değerlendirmesi gerekli"
                    })
                    if hr > 150 and anomalies["severity"] != "Kritik":
                        anomalies["severity"] = "Yüksek"
                elif hr < 45:
                    anomalies["arrhythmias"].append({
                        "type": "Ciddi Bradikardi",
                        "severity": "Yüksek",
                        "description": f"Kalp hızı: {hr:.0f} bpm",
                        "recommendation": "Pacemaker değerlendirmesi gerekebilir"
                    })
                    if anomalies["severity"] != "Kritik":
                        anomalies["severity"] = "Yüksek"
                elif hr < 60:
                    anomalies["arrhythmias"].append({
                        "type": "Bradikardi",
                        "severity": "Orta",
                        "description": f"Kalp hızı: {hr:.0f} bpm",
                        "recommendation": "Klinik korelasyon ve followup önerilir"
                    })
            
            # QRS genişlik analizi
            qrs = clinical_metrics.get("qrs_duration_ms")
            if qrs and qrs > 120:
                severity = "Yüksek" if qrs > 140 else "Orta"
                anomalies["conduction_abnormalities"].append({
                    "type": "Geniş QRS Kompleksi",
                    "severity": severity,
                    "description": f"QRS süresi: {qrs:.0f} ms",
                    "recommendation": "Dal bloğu/ventriküler ileti bozukluğu değerlendirmesi"
                })
                if severity == "Yüksek" and anomalies["severity"] not in ["Kritik", "Yüksek"]:
                    anomalies["severity"] = "Yüksek"
            
            # HRV analizi (ANS değerlendirmesi)
            hrv = clinical_metrics.get("heart_rate_variability", {})
            rmssd = hrv.get("rmssd_ms")
            if rmssd is not None:
                if rmssd < 15:  # Düşük HRV
                    anomalies["arrhythmias"].append({
                        "type": "Düşük Kalp Hızı Variabilitesi",
                        "severity": "Orta",
                        "description": f"RMSSD: {rmssd:.1f} ms (normal >20 ms)",
                        "recommendation": "Otonom sinir sistemi değerlendirmesi"
                    })
                elif rmssd > 100:  # Çok yüksek HRV
                    anomalies["arrhythmias"].append({
                        "type": "Anormal Yüksek HRV",
                        "severity": "Orta",
                        "description": f"RMSSD: {rmssd:.1f} ms",
                        "recommendation": "Aritmi taraması önerilir"
                    })
            
            # PR interval analizi
            pr = clinical_metrics.get("pr_interval_ms")
            if pr:
                if pr > 200:
                    anomalies["conduction_abnormalities"].append({
                        "type": "1. Derece AV Blok",
                        "severity": "Orta",
                        "description": f"PR interval: {pr:.0f} ms",
                        "recommendation": "İleti sistemi değerlendirmesi"
                    })
                elif pr < 120:
                    anomalies["conduction_abnormalities"].append({
                        "type": "Kısa PR Interval",
                        "severity": "Orta",
                        "description": f"PR interval: {pr:.0f} ms",
                        "recommendation": "Pre-eksitasyon sendromu değerlendirmesi"
                    })
            
            # QT interval analizi
            qt = clinical_metrics.get("qt_interval_ms")
            if qt and hr:
                # QTc hesaplama (Bazett formülü)
                qtc = qt / np.sqrt((60/hr) / 60)
                if qtc > 450:  # Uzun QT
                    severity = "Yüksek" if qtc > 500 else "Orta"
                    anomalies["conduction_abnormalities"].append({
                        "type": "Uzun QT Sendromu",
                        "severity": severity,
                        "description": f"QTc: {qtc:.0f} ms",
                        "recommendation": "Kardiyoloji konsültasyonu ve ilaç gözden geçirmesi"
                    })
                    if severity == "Yüksek" and anomalies["severity"] != "Kritik":
                        anomalies["severity"] = "Yüksek"
        
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
        
        return anomalies
    
    def generate_ai_summary(self, predictions: List[Dict], clinical_metrics: Dict, anomalies: Dict) -> Dict:
        """Gelişmiş yapay zeka tabanlı klinik özet."""
        if not predictions:
            return {
                "risk_level": "Bilinmiyor",
                "findings": ["Analiz için yeterli kalp atışı bulunamadı"],
                "recommendations": ["Sinyal kalitesinin iyi olduğundan emin olun ve kaydı tekrarlayın"],
                "detailed_analysis": {}
            }
        
        # Beat dağılımı
        beat_counts = {}
        total_beats = len(predictions)
        for pred in predictions:
            class_name = pred['class_name']
            beat_counts[class_name] = beat_counts.get(class_name, 0) + 1
        
        beat_percentages = {k: (v/total_beats)*100 for k, v in beat_counts.items()}
        
        # Risk seviyesini anomalilerden al
        risk_level = anomalies.get("severity", "Normal")
        
        # Bulgular ve tavsiyeler
        findings = []
        recommendations = []
        
        # Anomali bulgularını ekle
        for arrhythmia in anomalies.get("arrhythmias", []):
            findings.append(f"🫀 {arrhythmia['type']}: {arrhythmia['description']}")
            recommendations.append(f" {arrhythmia['recommendation']}")
        
        for abnormality in anomalies.get("morphology_abnormalities", []):
            findings.append(f"📊 {abnormality['type']}: {abnormality['description']}")
            recommendations.append(f" {abnormality['recommendation']}")
        
        for conduction in anomalies.get("conduction_abnormalities", []):
            findings.append(f"⚡ {conduction['type']}: {conduction['description']}")
            recommendations.append(f" {conduction['recommendation']}")
        
        # Temel klinik bulgular
        hr = clinical_metrics.get("heart_rate_bpm")
        if hr:
            if 60 <= hr <= 100:
                findings.append(f"✅ Normal kalp hızı: {hr:.0f} bpm")
        
        # HRV değerlendirmesi
        hrv = clinical_metrics.get("heart_rate_variability", {})
        if hrv:
            rmssd = hrv.get("rmssd_ms")
            if rmssd and 20 <= rmssd <= 50:
                findings.append(f"✅ Normal kalp hızı variabilitesi (RMSSD: {rmssd:.1f} ms)")
        
        # Beat dağılım analizi
        normal_percentage = beat_percentages.get("Normal (N)", 0)
        if normal_percentage > 90:
            findings.append(f"✅ Çoğunlukla normal kalp atımları (%{normal_percentage:.1f})")
        
        # Hiçbir anormal bulgu yoksa
        if risk_level == "Normal" and not findings:
            findings.append("✅ Önemli kardiyak anormallik tespit edilmedi")
            recommendations.append("Klinik olarak belirtildikçe rutin kardiyak izlemeye devam edin")
            recommendations.append("Sağlıklı yaşam tarzını sürdürün")
        
        # Detaylı analiz raporu
        detailed_analysis = {
            "rhythm_analysis": {
                "dominant_rhythm": "Sinüs ritmi" if normal_percentage > 70 else "Düzensiz ritim",
                "beat_distribution": beat_percentages,
                "total_beats_analyzed": total_beats
            },
            "clinical_parameters": {
                "heart_rate": {
                    "value": hr,
                    "unit": "bpm",
                    "status": "Normal" if hr and 60 <= hr <= 100 else "Anormal"
                } if hr else None,
                "hrv_metrics": hrv if hrv else None,
                "intervals": {
                    "qrs_duration": clinical_metrics.get("qrs_duration_ms"),
                    "pr_interval": clinical_metrics.get("pr_interval_ms"),
                    "qt_interval": clinical_metrics.get("qt_interval_ms")
                }
            },
            "risk_assessment": {
                "overall_risk": risk_level,
                "arrhythmia_burden": len(anomalies.get("arrhythmias", [])),
                "conduction_issues": len(anomalies.get("conduction_abnormalities", [])),
                "morphology_issues": len(anomalies.get("morphology_abnormalities", []))
            }
        }
        
        return {
            "risk_level": risk_level,
            "findings": findings,
            "recommendations": list(set(recommendations)),  # Duplikatları kaldır
            "detailed_analysis": detailed_analysis,
            "anomalies": anomalies
        }
    
    def process_ecg_record(self, raw_ecg: np.ndarray, fs_in: int = 500, fs_out: int = 125) -> Dict:
        """Ana EKG işleme akışı - 500Hz giriş, 125Hz çıkış ve geliştirilmiş analiz."""
        try:
            logger.info(f"EKG analizi başladı... Örneklem: {len(raw_ecg)}, Giriş Örnekleme Hızı: {fs_in}Hz, Çıkış: {fs_out}Hz")

            # Giriş verisi kontrolü
            if raw_ecg is None or raw_ecg.size == 0:
                return {"success": False, "error": "Boş veya geçersiz EKG verisi alındı."}
            
            # Sinyalin anlamlı varyasyona sahip olup olmadığını kontrol et
            signal_std = np.std(raw_ecg)
            if signal_std < 1e-5:
                return {"success": False, "error": "EKG sinyali anlamlı bir varyasyon içermiyor (düz çizgi)."}
            
            # NaN ve inf değerleri temizle
            if not np.isfinite(raw_ecg).all():
                raw_ecg = np.nan_to_num(raw_ecg)
            
            logger.info(f"Sinyal standart sapması: {signal_std:.6f}")

            # Adım 1: Sinyal ön işleme (orijinal örnekleme hızında)
            cleaned_signal = self.apply_filters(raw_ecg, fs_in)
            logger.info("Sinyal filtreleme tamamlandı")
            
            # Adım 2: Güçlü R-peak tespiti (orijinal örnekleme hızında)
            r_peaks = self.robust_r_peak_detection(cleaned_signal, fs_in)
            logger.info(f"{len(r_peaks)} R-peak tespit edildi")
            
            # R-peak sayısı kontrolü - daha esnek yaklaşım
            if len(r_peaks) < 2:
                error_message = f"Yetersiz kalp atışı tespit edildi (Bulunan: {len(r_peaks)}). En az 2 atım gerekli."
                logger.warning(error_message)
                return {
                    "success": True,
                    "display_signal": cleaned_signal.tolist(),
                    "r_peaks": [],
                    "beat_predictions": [],
                    "clinical_metrics": {"heart_rate_bpm": None},
                    "ai_summary": {
                        "risk_level": "Belirlenemedi", 
                        "findings": [error_message], 
                        "recommendations": ["Sinyal kalitesini kontrol edin", "Daha uzun kayıt alın"]
                    }
                }

            # Adım 3: Gelişmiş klinik metrikleri hesapla (orijinal örnekleme hızında)
            clinical_metrics = self.calculate_advanced_clinical_metrics(cleaned_signal, r_peaks, fs_in)
            logger.info("Klinik metrikler hesaplandı")
            
            # Adım 4: Model için sinyali yeniden örnekle (125Hz)
            resampled_signal = cleaned_signal
            r_peaks_resampled = r_peaks
            
            if fs_in != fs_out:
                try:
                    resampled_signal = resample(cleaned_signal, int(len(cleaned_signal) * fs_out / fs_in))
                    r_peaks_resampled = [int(p * fs_out / fs_in) for p in r_peaks]
                    r_peaks_resampled = [p for p in r_peaks_resampled if 0 <= p < len(resampled_signal)]
                    logger.info(f"Sinyal {fs_in}Hz'den {fs_out}Hz'e yeniden örneklendi")
                except Exception as e:
                    logger.warning(f"Resampling failed: {e}, using original signal")
                    fs_out = fs_in  # Fallback to original sampling rate

            # Adım 5: Beat ekstraksiyon ve AI Tahminleri
            beats = self.extract_beats(resampled_signal, r_peaks_resampled, window=187)
            predictions = self.predict_beats(beats)
            logger.info(f"{len(predictions)} kalp atışı sınıflandırıldı")
            
            # Adım 6: Anomali Tespiti
            anomalies = self.detect_anomalies(predictions, clinical_metrics)
            logger.info(f"Anomali analizi tamamlandı - Risk seviyesi: {anomalies.get('severity', 'Normal')}")
            
            # Adım 7: Gelişmiş AI Özetini Oluştur
            ai_summary = self.generate_ai_summary(predictions, clinical_metrics, anomalies)
            
            # Sonucu hazırla
            result = {
                "success": True,
                "display_signal": cleaned_signal.tolist(),
                "r_peaks": r_peaks,  # Orijinal örnekleme hızındaki R-peak'ler
                "beat_predictions": predictions,
                "clinical_metrics": clinical_metrics,
                "anomalies": anomalies,
                "ai_summary": ai_summary,
                "processing_info": {
                    "input_sampling_rate": fs_in,
                    "output_sampling_rate": fs_out,
                    "signal_length_seconds": len(raw_ecg) / fs_in,
                    "beats_analyzed": len(predictions),
                    "signal_quality": "Good" if signal_std > 0.1 else "Fair"
                }
            }
            
            logger.info("EKG analizi başarıyla tamamlandı")
            return result
            
        except Exception as e:
            logger.error(f"EKG analizinde kritik hata: {e}", exc_info=True)
            return {"success": False, "error": str(e)}


# ############################################################################
# ##                           FLASK WEB API                                ##
# ############################################################################

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize components
db_manager = DatabaseManager()
ecg_processor = None

try:
    ecg_processor = ECGProcessor(model_path="model.pth")
except Exception as e:
    logger.critical(f"ECG Processor initialization failed: {e}")

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ecg_processor_available": ecg_processor is not None
    })

# Patient management endpoints
@app.route('/api/patients', methods=['GET'])
def get_patients():
    """Get all patients."""
    try:
        with db_manager.get_connection() as conn:
            patients = conn.execute(
                'SELECT * FROM patients ORDER BY name ASC'
            ).fetchall()
            return jsonify([dict(row) for row in patients])
    except Exception as e:
        logger.error(f"Error fetching patients: {e}")
        return jsonify({"error": "Failed to fetch patients"}), 500

@app.route('/api/patients', methods=['POST'])
def add_patient():
    """Add new patient."""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'tc']
        for field in required_fields:
            if field not in data or not data[field].strip():
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        with db_manager.get_connection() as conn:
            conn.execute(
                '''INSERT INTO patients 
                   (name, tc, age, gender, phone, medications, complaints) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (
                    data['name'].strip(),
                    data['tc'].strip(),
                    data.get('age'),
                    data.get('gender', '').strip(),
                    data.get('phone', '').strip(),
                    json.dumps(data.get('medications', [])),
                    data.get('complaints', '').strip()
                )
            )
            conn.commit()
            
        return jsonify({"message": "Patient added successfully"}), 201
        
    except sqlite3.IntegrityError:
        return jsonify({"error": "Patient with this TC already exists"}), 409
    except Exception as e:
        logger.error(f"Error adding patient: {e}")
        return jsonify({"error": "Failed to add patient"}), 500

@app.route('/api/patients/<int:patient_id>', methods=['GET'])
def get_patient(patient_id):
    """Get specific patient."""
    try:
        with db_manager.get_connection() as conn:
            patient = conn.execute(
                'SELECT * FROM patients WHERE id=?',
                (patient_id,)
            ).fetchone()
            
            if not patient:
                return jsonify({"error": "Patient not found"}), 404
            
            return jsonify(dict(patient))
    except Exception as e:
        logger.error(f"Error fetching patient {patient_id}: {e}")
        return jsonify({"error": "Failed to fetch patient"}), 500

@app.route('/api/patients/<int:patient_id>', methods=['PUT'])
def update_patient(patient_id):
    """Update patient information."""
    try:
        data = request.get_json()
        
        with db_manager.get_connection() as conn:
            # Check if patient exists
            patient = conn.execute('SELECT id FROM patients WHERE id=?', (patient_id,)).fetchone()
            if not patient:
                return jsonify({"error": "Patient not found"}), 404
            
            conn.execute(
                '''UPDATE patients 
                   SET name=?, tc=?, age=?, gender=?, phone=?, medications=?, complaints=?, updated_at=?
                   WHERE id=?''',
                (
                    data['name'].strip(),
                    data['tc'].strip(),
                    data.get('age'),
                    data.get('gender', '').strip(),
                    data.get('phone', '').strip(),
                    json.dumps(data.get('medications', [])),
                    data.get('complaints', '').strip(),
                    datetime.now().isoformat(),
                    patient_id
                )
            )
            conn.commit()
            
        return jsonify({"message": "Patient updated successfully"})
        
    except Exception as e:
        logger.error(f"Error updating patient {patient_id}: {e}")
        return jsonify({"error": "Failed to update patient"}), 500

@app.route('/api/patients/<int:patient_id>', methods=['DELETE'])
def delete_patient(patient_id):
    """Delete patient and all associated files."""
    try:
        with db_manager.get_connection() as conn:
            result = conn.execute('DELETE FROM patients WHERE id=?', (patient_id,))
            conn.commit()
            
            if result.rowcount == 0:
                return jsonify({"error": "Patient not found"}), 404
                
        return jsonify({"message": "Patient deleted successfully"})
        
    except Exception as e:
        logger.error(f"Error deleting patient {patient_id}: {e}")
        return jsonify({"error": "Failed to delete patient"}), 500

# File management endpoints
@app.route('/api/patients/<int:patient_id>/files', methods=['GET'])
def get_patient_files(patient_id):
    """Get all ECG files for a patient."""
    try:
        with db_manager.get_connection() as conn:
            # Check which columns exist first
            cursor = conn.cursor()
            table_info = cursor.execute("PRAGMA table_info(ekg_files)").fetchall()
            columns = [col[1] for col in table_info]
            
            # Build query based on available columns
            base_columns = ['id', 'file_name', 'uploaded_at', 'sampling_rate']
            select_columns = []
            
            for col in base_columns:
                if col in columns:
                    select_columns.append(col)
            
            # Add duration if it exists
            if 'duration' in columns:
                select_columns.append('duration')
            else:
                # Calculate duration from data length if duration column doesn't exist
                select_columns.append('LENGTH(ekg_data)/4/sampling_rate as duration')
            
            query = f'''SELECT {', '.join(select_columns)} 
                       FROM ekg_files WHERE patient_id=? ORDER BY uploaded_at DESC'''
            
            files = conn.execute(query, (patient_id,)).fetchall()
            return jsonify([dict(row) for row in files])
    except Exception as e:
        logger.error(f"Error fetching files for patient {patient_id}: {e}")
        return jsonify({"error": "Failed to fetch files"}), 500

@app.route('/api/patients/<int:patient_id>/files', methods=['POST'])
def upload_file(patient_id):
    """Bir hasta için yeni bir EKG dosyası yükler."""
    try:
        data = request.get_json()
        
        # Frontend'den gelen anahtar isimleriyle eşleştir
        file_name = data.get('name')
        ekg_data_list = data.get('data')
        sampling_rate = data.get('samplingRate', 500)
        uploaded_at = data.get('uploadedAt', datetime.now().isoformat())

        if not ekg_data_list or not file_name:
            return jsonify({"error": "Eksik alanlar: 'data' ve 'name' gerekli."}), 400
        
        ekg_data_np = np.array(ekg_data_list, dtype=np.float32)
        duration = len(ekg_data_np) / sampling_rate
        
        with db_manager.get_connection() as conn:
            conn.execute(
                '''INSERT INTO ekg_files 
                   (patient_id, file_name, uploaded_at, sampling_rate, duration, ekg_data) 
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (patient_id, file_name.strip(), uploaded_at, sampling_rate, duration, ekg_data_np.tobytes())
            )
            conn.commit()
            
        logger.info(f"Hasta {patient_id} için dosya yüklendi: {file_name}")
        return jsonify({"message": "Dosya başarıyla yüklendi"}), 201
        
    except Exception as e:
        logger.error(f"Dosya yükleme hatası (Hasta ID: {patient_id}): {e}", exc_info=True)
        return jsonify({"error": "Sunucu hatası nedeniyle dosya yüklenemedi."}), 500

@app.route('/api/files/<int:file_id>', methods=['GET'])
def get_file_info(file_id):
    """Get detailed information about an ECG file."""
    try:
        with db_manager.get_connection() as conn:
            file_info = conn.execute(
                '''SELECT f.*, p.name as patient_name 
                   FROM ekg_files f 
                   JOIN patients p ON f.patient_id = p.id 
                   WHERE f.id=?''',
                (file_id,)
            ).fetchone()
            
            if not file_info:
                return jsonify({"error": "File not found"}), 404
            
            # Get analysis results if available
            analysis = conn.execute(
                'SELECT * FROM analysis_results WHERE file_id=? ORDER BY analyzed_at DESC LIMIT 1',
                (file_id,)
            ).fetchone()
            
            result = dict(file_info)
            result['ekg_data'] = None  # Don't send raw data
            if analysis:
                result['latest_analysis'] = dict(analysis)
            
            return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error fetching file info {file_id}: {e}")
        return jsonify({"error": "Failed to fetch file information"}), 500

@app.route('/api/files/<int:file_id>', methods=['DELETE'])
def delete_file(file_id):
    """Delete an ECG file."""
    try:
        with db_manager.get_connection() as conn:
            # Check if file exists
            file_exists = conn.execute('SELECT id FROM ekg_files WHERE id=?', (file_id,)).fetchone()
            if not file_exists:
                return jsonify({"error": "File not found"}), 404
            
            conn.execute('DELETE FROM ekg_files WHERE id=?', (file_id,))
            conn.commit()
            
        logger.info(f"ECG file {file_id} deleted successfully")
        return jsonify({"message": "File deleted successfully"})
        
    except Exception as e:
        logger.error(f"Error deleting file {file_id}: {e}")
        return jsonify({"error": "Failed to delete file"}), 500

# Analysis endpoints
@app.route('/api/analyze/<int:file_id>', methods=['GET'])
def analyze_file(file_id):
    """Analyze an ECG file and return results."""
    try:
        if ecg_processor is None:
            return jsonify({"success": False, "error": "Analysis service not available"}), 503
        
        with db_manager.get_connection() as conn:
            # Get file data
            file_record = conn.execute(
                'SELECT ekg_data, sampling_rate, file_name FROM ekg_files WHERE id=?',
                (file_id,)
            ).fetchone()
            
            if not file_record:
                return jsonify({"success": False, "error": "File not found"}), 404
            
            # Process ECG data - 500Hz giriş, 125Hz çıkış kullan
            ekg_data = np.frombuffer(file_record['ekg_data'], dtype=np.float32)
            result = ecg_processor.process_ecg_record(ekg_data, fs_in=file_record['sampling_rate'], fs_out=125)
            
            # Store analysis results if successful
            if result.get('success'):
                try:
                    # Calculate overall confidence score
                    predictions = result.get('beat_predictions', [])
                    avg_confidence = np.mean([p['confidence'] for p in predictions]) if predictions else 0.0
                    
                    conn.execute(
                        '''INSERT INTO analysis_results (file_id, analysis_type, results, confidence_score)
                           VALUES (?, ?, ?, ?)''',
                        (file_id, 'ai_prediction', json.dumps(result), avg_confidence)
                    )
                    conn.commit()
                    logger.info(f"Analysis results stored for file {file_id}")
                except Exception as e:
                    logger.warning(f"Failed to store analysis results: {e}")
            
            return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error analyzing file {file_id}: {e}")
        return jsonify({"success": False, "error": "Analysis failed"}), 500

@app.route('/api/files/<int:file_id>/analysis-history', methods=['GET'])
def get_analysis_history(file_id):
    """Get analysis history for a file."""
    try:
        with db_manager.get_connection() as conn:
            analyses = conn.execute(
                '''SELECT id, analysis_type, confidence_score, analyzed_at 
                   FROM analysis_results WHERE file_id=? ORDER BY analyzed_at DESC''',
                (file_id,)
            ).fetchall()
            
            return jsonify([dict(row) for row in analyses])
            
    except Exception as e:
        logger.error(f"Error fetching analysis history for file {file_id}: {e}")
        return jsonify({"error": "Failed to fetch analysis history"}), 500

@app.route('/api/analysis/<int:analysis_id>', methods=['GET'])
def get_analysis_details(analysis_id):
    """Get detailed analysis results."""
    try:
        with db_manager.get_connection() as conn:
            analysis = conn.execute(
                'SELECT * FROM analysis_results WHERE id=?',
                (analysis_id,)
            ).fetchone()
            
            if not analysis:
                return jsonify({"error": "Analysis not found"}), 404
            
            result = dict(analysis)
            # Parse JSON results
            if result['results']:
                result['results'] = json.loads(result['results'])
            
            return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error fetching analysis {analysis_id}: {e}")
        return jsonify({"error": "Failed to fetch analysis details"}), 500

# Statistics and reporting endpoints
@app.route('/api/statistics/overview', methods=['GET'])
def get_statistics_overview():
    """Get system overview statistics."""
    try:
        with db_manager.get_connection() as conn:
            # Patient statistics
            patient_count = conn.execute('SELECT COUNT(*) as count FROM patients').fetchone()['count']
            
            # File statistics
            file_count = conn.execute('SELECT COUNT(*) as count FROM ekg_files').fetchone()['count']
            
            # Analysis statistics
            analysis_count = conn.execute('SELECT COUNT(*) as count FROM analysis_results').fetchone()['count']
            
            # Recent activity
            recent_files = conn.execute(
                '''SELECT COUNT(*) as count FROM ekg_files 
                   WHERE uploaded_at > datetime('now', '-7 days')'''
            ).fetchone()['count']
            
            recent_analyses = conn.execute(
                '''SELECT COUNT(*) as count FROM analysis_results 
                   WHERE analyzed_at > datetime('now', '-7 days')'''
            ).fetchone()['count']
            
            return jsonify({
                "total_patients": patient_count,
                "total_files": file_count,
                "total_analyses": analysis_count,
                "recent_files_week": recent_files,
                "recent_analyses_week": recent_analyses,
                "system_status": "operational",
                "ecg_processor_available": ecg_processor is not None
            })
            
    except Exception as e:
        logger.error(f"Error fetching statistics: {e}")
        return jsonify({"error": "Failed to fetch statistics"}), 500

# Cleanup function
import atexit

def cleanup_on_exit():
    """Cleanup function called on application exit."""
    logger.info("Performing cleanup before shutdown...")
    try:
        # Clear model from memory
        if ecg_processor and ecg_processor.model:
            del ecg_processor.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

atexit.register(cleanup_on_exit)

# Global exception handler
@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify({
        "error": "An unexpected error occurred",
        "timestamp": datetime.now().isoformat()
    }), 500


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Enhanced ECG Analysis System v3.1 - FIXED")
    logger.info("=" * 60)
    logger.info("BUG FIXES:")
    logger.info("  - Fixed beat extraction and classification pipeline")
    logger.info("  - Corrected resampling: 500Hz input → 125Hz output")
    logger.info("  - Improved beat validation and error handling")
    logger.info("  - Enhanced R-peak to beat conversion accuracy")
    logger.info("=" * 60)
    logger.info("Initializing components...")
    
    # Component status check
    db_status = "Connected" if db_manager else "✗ Failed"
    model_status = "Loaded" if ecg_processor and ecg_processor.model else "✗ Not Available"
    
    logger.info(f"Database: {db_status}")
    logger.info(f"AI Model: {model_status}")
    logger.info("=" * 60)
    
    if ecg_processor and ecg_processor.model:
        logger.info("Available Analysis Features:")
        logger.info("  - Robust multi-method R-peak detection")
        logger.info("  - AI-powered beat classification (FIXED)")
        logger.info("  - Advanced clinical metrics with HRV")
        logger.info("  - Comprehensive anomaly detection")
        logger.info("  - 500Hz→125Hz optimized processing")
        logger.info("  - Risk assessment and clinical recommendations")
    else:
        logger.warning("Limited functionality - AI model not available")
    
    logger.info("=" * 60)
    logger.info("Processing Pipeline (CORRECTED):")
    logger.info("  1. Input: 500Hz ECG signal")
    logger.info("  2. Filter & clean at original sampling rate")
    logger.info("  3. R-peak detection at original rate")
    logger.info("  4. Clinical metrics calculation")
    logger.info("  5. Resample to 125Hz for AI model")
    logger.info("  6. Extract beats around R-peaks")
    logger.info("  7. AI classification of individual beats")
    logger.info("  8. Anomaly detection & risk assessment")
    logger.info("=" * 60)
    logger.info("Key Improvements:")
    logger.info("  - Beat extraction with proper validation")
    logger.info("  - Tensor validation before AI prediction")
    logger.info("  - Improved error handling in beat pipeline")
    logger.info("  - Better logging for debugging")
    logger.info("  - Correct sampling rate conversion")
    logger.info("=" * 60)
    logger.info("API Endpoints Available:")
    logger.info("  - Patient Management: /api/patients")
    logger.info("  - File Management: /api/files")
    logger.info("  - ECG Analysis: /api/analyze")
    logger.info("  - Statistics: /api/statistics")
    logger.info("  - System Health: /health")
    logger.info("=" * 60)
    logger.info("Starting server on http://0.0.0.0:5001")
    logger.info("Press Ctrl+C to stop the server")
    logger.info("=" * 60)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.critical(f"Server startup failed: {e}")
    finally:
        logger.info("ECG Analysis System shutdown complete")