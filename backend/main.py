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
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS patients (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, tc TEXT UNIQUE NOT NULL,
                        age INTEGER, gender TEXT, phone TEXT, medications TEXT,
                        complaints TEXT, created_at DATETIME, updated_at DATETIME
                    )''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS ekg_files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id INTEGER NOT NULL, file_name TEXT NOT NULL,
                        uploaded_at DATETIME, sampling_rate INTEGER, duration REAL,
                        ekg_data BLOB NOT NULL, analysis_results TEXT,
                        FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE
                    )''')
                
                try:
                    cursor.execute("SELECT duration FROM ekg_files LIMIT 1")
                except sqlite3.OperationalError:
                    # Sütun yoksa, ekle
                    cursor.execute("ALTER TABLE ekg_files ADD COLUMN duration REAL")
                    logger.info("Mevcut 'ekg_files' tablosuna 'duration' sütunu eklendi.")
                
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
        

        self.normalization_1 = nn.BatchNorm1d(hidden_size)
        self.normalization_2 = nn.BatchNorm1d(hidden_size)
        self.normalization_3 = nn.BatchNorm1d(hidden_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = nn.MaxPool1d(kernel_size=2)
    
    def forward(self, x):
        conv1 = self.conv_1(x)
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
            nyquist = fs / 2
            low_freq = 0.5 / nyquist
            high_freq = min(40.0, nyquist - 1) / nyquist
            
            if low_freq >= high_freq:
                logger.warning("Invalid filter frequencies, skipping filtering")
                return signal
            
            b, a = butter(4, [low_freq, high_freq], btype='band')
            filtered_signal = filtfilt(b, a, signal)
            
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
        
        if len(r_peaks) < 3:
            try:
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


    def extract_beats(self, signal: np.ndarray, r_peaks: list, window_size: int = 187) -> np.ndarray:
        """
        Orijinal eğitim reçetesindeki segmentasyon, padding ve discarding
        kurallarını harfiyen uygular.
        """
        if len(r_peaks) < 2:
            return np.array([])
        
        beats = []
        logger.info(f"Extracting beats from {len(r_peaks)} R-peaks using training recipe...")
        
        for i in range(len(r_peaks) - 1):
            # Bir R-peak'ten bir sonrakinin başlangıcına kadar olan bölümü al
            start = r_peaks[i]
            end = r_peaks[i+1]
            
            segment = signal[start:end]
            
            # Adım 4: 187'den uzunsa ATLA (discard)
            if len(segment) > window_size:
                continue
                
            # Adım 5: 187'den kısaysa sonuna SIFIR EKLE (pad with zeroes)
            if len(segment) < window_size:
                padding = np.zeros(window_size - len(segment))
                segment = np.concatenate([segment, padding])
            
            beats.append(segment)
            
        logger.info(f"Extracted {len(beats)} valid beats (segments > 187 were discarded)")
        return np.array(beats)
    
    def predict_beats(self, beats: np.ndarray, fs_in: int = 200, fs_model: int = 125) -> List[Dict]:
        """Predict beat classifications using the CNN model with correct preprocessing pipeline."""
        if self.model is None:
            logger.warning("Model not available for beat prediction")
            return []
        
        if len(beats) == 0:
            logger.warning("No beats provided for prediction")
            return []
        
        predictions = []
        successful_predictions = 0
        
        logger.info(f"Starting prediction for {len(beats)} beats with correct preprocessing pipeline")
        logger.info(f"Resampling from {fs_in}Hz to {fs_model}Hz and applying Min-Max normalization")
        
        try:
            with torch.no_grad():
                for i, beat in enumerate(beats):
                    try:
                        # DOĞRU İŞLEM SIRASI:
                        # 1. Her bir atımı fs_in'den fs_model'e yeniden örnekle
                        if fs_in != fs_model:
                            resampled_beat = resample(beat, int(len(beat) * fs_model / fs_in))
                        else:
                            resampled_beat = beat.copy()
                        
                        # 2. Min-Max normalizasyon (0-1 arası) - test_model.py'deki gibi
                        min_val, max_val = np.min(resampled_beat), np.max(resampled_beat)
                        if (max_val - min_val) > 1e-6:
                            normalized_beat = (resampled_beat - min_val) / (max_val - min_val)
                        else:
                            normalized_beat = resampled_beat - min_val
                        
                        # 3. 187 boyutuna getir (padding/truncating)
                        final_beat = np.zeros(187)
                        if len(normalized_beat) >= 187:
                            final_beat = normalized_beat[:187]
                        else:
                            final_beat[:len(normalized_beat)] = normalized_beat
                        
                        # 4. Tensor'e dönüştür ve tahmin yap
                        tensor = torch.FloatTensor(final_beat).unsqueeze(0).unsqueeze(0).to(self.device)
                        
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
        
        logger.info(f"Successfully predicted {successful_predictions} out of {len(beats)} beats using corrected pipeline")
        return predictions
    
    def calculate_advanced_clinical_metrics(self, signal: np.ndarray, r_peaks: List[int], fs: int) -> Dict:
        """Gelişmiş klinik ECG metrikleri hesapla (HRV dahil) - DÜZELTME"""
        metrics = {
            "heart_rate_bpm": None,
            "heart_rate_variability": {},
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
            
            # FİZYOLOJİK SINIR KONTROLÜ - Anormal RR intervallerini filtrele
            # Normal RR: 300-2000 ms (30-200 BPM arası)
            valid_rr_mask = (rr_intervals_ms >= 300) & (rr_intervals_ms <= 2000)
            rr_intervals_ms_filtered = rr_intervals_ms[valid_rr_mask]
            
            if len(rr_intervals_ms_filtered) == 0:
                logger.warning("Hiçbir geçerli RR interval bulunamadı")
                return metrics
            
            # Kalp hızı hesaplama (filtrelenmiş RR'lar ile)
            hr = 60.0 / np.mean(rr_intervals_ms_filtered / 1000)  # BPM
            metrics["heart_rate_bpm"] = float(hr)
            metrics["rr_intervals_ms"] = rr_intervals_ms_filtered.tolist()
            
            if len(rr_intervals_ms_filtered) > 1:
                hrv_metrics = {}
                
                # Time-domain HRV metrics
                # RMSSD (Root Mean Square of Successive Differences)
                rr_diff = np.diff(rr_intervals_ms_filtered)

                if len(rr_diff) > 0:
                    # Ham RMSSD
                    rmssd_raw = np.sqrt(np.mean(rr_diff ** 2))
                    hrv_metrics["rmssd_ms"] = float(rmssd_raw)
                    
                    # Filtreli RMSSD (opsiyonel)
                    rr_diff_filt = rr_diff[np.abs(rr_diff) <= 500]
                    if len(rr_diff_filt) > 0:
                        rmssd_filt = np.sqrt(np.mean(rr_diff_filt ** 2))
                        hrv_metrics["rmssd_filtered_ms"] = float(rmssd_filt)
                    else:
                        hrv_metrics["rmssd_filtered_ms"] = None
                else:
                    hrv_metrics["rmssd_ms"] = None
                    hrv_metrics["rmssd_filtered_ms"] = None
                
                # SDNN (Standard Deviation of NN intervals)
                sdnn = np.std(rr_intervals_ms_filtered)
                hrv_metrics["sdnn_ms"] = float(sdnn)
                
                # pNN50 (Percentage of successive RR intervals that differ by more than 50ms)
                if len(rr_diff) > 0:
                    nn50 = np.sum(np.abs(rr_diff) > 50)
                    pnn50 = (nn50 / len(rr_diff)) * 100
                    hrv_metrics["pnn50_percent"] = float(pnn50)
                else:
                    hrv_metrics["pnn50_percent"] = 0.0
                
                # SDSD (Standard Deviation of Successive Differences)
                if len(rr_diff) > 0:
                    sdsd = np.std(rr_diff)
                    hrv_metrics["sdsd_ms"] = float(sdsd)
                else:
                    hrv_metrics["sdsd_ms"] = 0.0
                
                # Triangular Index
                if len(rr_intervals_ms_filtered) > 5:
                    hist, _ = np.histogram(rr_intervals_ms_filtered, bins=min(20, len(rr_intervals_ms_filtered)//2))
                    max_hist = np.max(hist)
                    tri_index = len(rr_intervals_ms_filtered) / max_hist if max_hist > 0 else 0
                    hrv_metrics["triangular_index"] = float(tri_index)
                else:
                    hrv_metrics["triangular_index"] = 0.0
                
                # Frequency-domain HRV (geliştirilmiş)
                if len(rr_intervals_ms_filtered) > 10:
                    try:
                        # RR serisi için interpolasyon (4Hz sampling)
                        time_stamps = np.cumsum(rr_intervals_ms_filtered / 1000)
                        interp_time = np.arange(0, time_stamps[-1], 0.25)  # 4Hz
                        
                        if len(time_stamps) > 1 and len(interp_time) > 1:
                            rr_interp = np.interp(interp_time, time_stamps[:-1], rr_intervals_ms_filtered[:-1])
                            
                            # Detrend
                            rr_detrend = rr_interp - np.mean(rr_interp)
                            
                            # FFT
                            freqs = np.fft.fftfreq(len(rr_detrend), d=0.25)
                            power = np.abs(np.fft.fft(rr_detrend)) ** 2
                            
                            # Pozitif frekanslar
                            pos_mask = freqs > 0
                            freqs_pos = freqs[pos_mask]
                            power_pos = power[pos_mask]
                            
                            # Frekans bantları
                            vlf_mask = (freqs_pos >= 0.0033) & (freqs_pos < 0.04)
                            lf_mask = (freqs_pos >= 0.04) & (freqs_pos < 0.15)
                            hf_mask = (freqs_pos >= 0.15) & (freqs_pos < 0.4)
                            
                            vlf_power = np.sum(power_pos[vlf_mask]) if np.any(vlf_mask) else 0
                            lf_power = np.sum(power_pos[lf_mask]) if np.any(lf_mask) else 0
                            hf_power = np.sum(power_pos[hf_mask]) if np.any(hf_mask) else 0
                            
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
            
            # ===== DÜZELTME: WAVE DELINEATION =====
            try:
                _, waves = nk.ecg_delineate(signal, r_peaks, sampling_rate=fs, method="dwt")
                
                # QRS duration (Q'dan S'ye)
                q_peaks = waves.get('ECG_Q_Peaks')
                s_peaks = waves.get('ECG_S_Peaks')
                
                if q_peaks is not None and s_peaks is not None:
                    q_peaks_clean = np.array([x for x in q_peaks if not np.isnan(x)], dtype=int)
                    s_peaks_clean = np.array([x for x in s_peaks if not np.isnan(x)], dtype=int)
                    
                    if len(q_peaks_clean) > 0 and len(s_peaks_clean) > 0:
                        # Eşleştirilebilir Q-S çiftlerini bul
                        min_pairs = min(len(q_peaks_clean), len(s_peaks_clean))
                        qrs_durations = []
                        
                        for i in range(min_pairs):
                            if q_peaks_clean[i] < s_peaks_clean[i]:  # Q, S'den önce olmalı
                                duration_ms = (s_peaks_clean[i] - q_peaks_clean[i]) / fs * 1000
                                if 40 <= duration_ms <= 200:  # Fizyolojik sınırlar
                                    qrs_durations.append(duration_ms)
                        
                        if qrs_durations:
                            metrics["qrs_duration_ms"] = float(np.mean(qrs_durations))
                
                # PR interval (P başlangıcından R'ye)
                p_onsets = waves.get('ECG_P_Onsets')
                if p_onsets is not None and len(r_peaks) > 0:
                    p_onsets_clean = np.array([x for x in p_onsets if not np.isnan(x)], dtype=int)
                    
                    if len(p_onsets_clean) > 0:
                        pr_intervals = []
                        
                        for p_onset in p_onsets_clean:
                            # En yakın R peak'i bul (P'den sonra gelen)
                            following_r = [r for r in r_peaks if r > p_onset]
                            if following_r:
                                closest_r = min(following_r)
                                pr_ms = (closest_r - p_onset) / fs * 1000
                                if 80 <= pr_ms <= 300:  # Fizyolojik sınırlar
                                    pr_intervals.append(pr_ms)
                        
                        if pr_intervals:
                            metrics["pr_interval_ms"] = float(np.mean(pr_intervals))
                
                t_offsets = waves.get('ECG_T_Offsets')
                if q_peaks is not None and t_offsets is not None:
                    q_peaks_clean = np.array([x for x in q_peaks if not np.isnan(x)], dtype=int)
                    t_offsets_clean = np.array([x for x in t_offsets if not np.isnan(x)], dtype=int)
                    
                    if len(q_peaks_clean) > 0 and len(t_offsets_clean) > 0:
                        qt_intervals = []
                        
                        for q_peak in q_peaks_clean:
                            # En yakın T offset'i bul (Q'dan sonra gelen)
                            following_t = [t for t in t_offsets_clean if t > q_peak]
                            if following_t:
                                closest_t = min(following_t)
                                qt_ms = (closest_t - q_peak) / fs * 1000
                                if 200 <= qt_ms <= 600:  # Fizyolojik sınırlar
                                    qt_intervals.append(qt_ms)
                        
                        if qt_intervals:
                            metrics["qt_interval_ms"] = float(np.mean(qt_intervals))
                            
            except Exception as e:
                logger.warning(f"Wave delineation failed: {e}")
        
        except Exception as e:
            logger.error(f"Clinical metrics calculation error: {e}")
        
        return metrics


    def detect_anomalies(self, predictions: List[Dict], clinical_metrics: Dict) -> Dict:
        """Gelişmiş anomali ve aritmi tespiti - DÜZELTME"""
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
            
            # ===== KALP HIZI ANOMALİLERİ =====
            hr = clinical_metrics.get("heart_rate_bpm")
            if hr:
                if hr > 120:
                    anomalies["arrhythmias"].append({
                        "type": "Taşikardi",
                        "severity": "Yüksek" if hr > 150 else "Orta",
                        "description": f"Kalp hızı: {hr:.0f} bpm (normal: 60-100 bpm)",
                        "recommendation": "Hızla kardiyoloji değerlendirmesi gerekli"
                    })
                    if hr > 150 and anomalies["severity"] != "Kritik":
                        anomalies["severity"] = "Yüksek"
                elif hr < 50:
                    severity = "Yüksek" if hr < 40 else "Orta"
                    anomalies["arrhythmias"].append({
                        "type": "Bradikardi",
                        "severity": severity,
                        "description": f"Kalp hızı: {hr:.0f} bpm (normal: 60-100 bpm)",
                        "recommendation": "Kardiyoloji konsültasyonu önerilir"
                    })
                    if severity == "Yüksek" and anomalies["severity"] != "Kritik":
                        anomalies["severity"] = "Yüksek"
            
            # ===== VENTRİKÜLER ARİTMİ TESPİTİ =====
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
            
            # ===== QRS GENİŞLİK ANALİZİ =====
            qrs = clinical_metrics.get("qrs_duration_ms")
            if qrs and qrs > 120:
                severity = "Yüksek" if qrs > 140 else "Orta"
                anomalies["conduction_abnormalities"].append({
                    "type": "Geniş QRS Kompleksi",
                    "severity": severity,
                    "description": f"QRS süresi: {qrs:.0f} ms (normal: <120 ms)",
                    "recommendation": "Dal bloğu/ventriküler ileti bozukluğu değerlendirmesi"
                })
                if severity == "Yüksek" and anomalies["severity"] not in ["Kritik", "Yüksek"]:
                    anomalies["severity"] = "Yüksek"
            
            # ===== DÜZELTME: HRV ANALİZİ =====
            hrv = clinical_metrics.get("heart_rate_variability", {})
            rmssd = hrv.get("rmssd_ms")
            if rmssd is not None:
                if rmssd < 15:  # Düşük HRV
                    anomalies["arrhythmias"].append({
                        "type": "Düşük Kalp Hızı Variabilitesi",
                        "severity": "Orta",
                        "description": f"RMSSD: {rmssd:.1f} ms (normal: 20-50 ms)",
                        "recommendation": "Otonom sinir sistemi değerlendirmesi"
                    })
                    if anomalies["severity"] == "Normal":
                        anomalies["severity"] = "Orta"
                elif rmssd > 80:  # Çok yüksek HRV (önceki 100'den düşürüldü)
                    anomalies["arrhythmias"].append({
                        "type": "Anormal Yüksek HRV",
                        "severity": "Orta",
                        "description": f"RMSSD: {rmssd:.1f} ms (normal: 20-50 ms)",
                        "recommendation": "Aritmi taraması önerilir"
                    })
                    if anomalies["severity"] == "Normal":
                        anomalies["severity"] = "Orta"
            
            # ===== PR INTERVAL ANALİZİ =====
            pr = clinical_metrics.get("pr_interval_ms")
            if pr:
                if pr > 200:
                    anomalies["conduction_abnormalities"].append({
                        "type": "1. Derece AV Blok",
                        "severity": "Orta",
                        "description": f"PR interval: {pr:.0f} ms (normal: 120-200 ms)",
                        "recommendation": "İleti sistemi değerlendirmesi"
                    })
                    if anomalies["severity"] == "Normal":
                        anomalies["severity"] = "Orta"
                elif pr < 120:
                    anomalies["conduction_abnormalities"].append({
                        "type": "Kısa PR Interval",
                        "severity": "Orta",
                        "description": f"PR interval: {pr:.0f} ms (normal: 120-200 ms)",
                        "recommendation": "Pre-eksitasyon sendromu değerlendirmesi"
                    })
                    if anomalies["severity"] == "Normal":
                        anomalies["severity"] = "Orta"
            
            # ===== DÜZELTME: QT INTERVAL ve QTc ANALİZİ =====
            qt = clinical_metrics.get("qt_interval_ms")
            if qt and hr:
                # Düzeltilmiş QTc hesaplama (Bazett formülü)
                rr_seconds = 60.0 / hr  # RR interval in seconds
                qtc = qt / np.sqrt(rr_seconds)  # Corrected QT
                
                # QTc normal değerleri: Erkek <430ms, Kadın <450ms (genel olarak <450ms)
                if qtc > 450:  # Uzun QT
                    severity = "Yüksek" if qtc > 500 else "Orta"
                    anomalies["conduction_abnormalities"].append({
                        "type": "Uzun QT Sendromu",
                        "severity": severity,
                        "description": f"QTc: {qtc:.0f} ms (normal: <450 ms), QT: {qt:.0f} ms",
                        "recommendation": "Kardiyoloji konsültasyonu ve ilaç gözden geçirmesi"
                    })
                    if severity == "Yüksek" and anomalies["severity"] != "Kritik":
                        anomalies["severity"] = "Yüksek"
                elif qt < 350:  # Kısa QT
                    anomalies["conduction_abnormalities"].append({
                        "type": "Kısa QT Sendromu",
                        "severity": "Orta",
                        "description": f"QTc: {qtc:.0f} ms (normal: >350 ms), QT: {qt:.0f} ms",
                        "recommendation": "Genetik kardiyomyopati değerlendirmesi"
                    })
                    if anomalies["severity"] == "Normal":
                        anomalies["severity"] = "Orta"
            
            # ===== SUPRAVENTRİKÜLER ARİTMİ TESPİTİ =====
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
            
            # ===== FUSION BEAT ANALİZİ =====
            f_count = beat_counts.get("Fusion (F)", 0)
            if f_count > 2:
                anomalies["morphology_abnormalities"].append({
                    "type": "Fusion Beat'ler",
                    "severity": "Orta",
                    "description": f"{f_count} fusion beat tespit edildi",
                    "recommendation": "İleti sistem değerlendirmesi önerilir"
                })
                if anomalies["severity"] == "Normal":
                    anomalies["severity"] = "Orta"
        
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
            findings.append(f"{arrhythmia['type']}: {arrhythmia['description']}")
            recommendations.append(f" {arrhythmia['recommendation']}")
        
        for abnormality in anomalies.get("morphology_abnormalities", []):
            findings.append(f"{abnormality['type']}: {abnormality['description']}")
            recommendations.append(f" {abnormality['recommendation']}")
        
        for conduction in anomalies.get("conduction_abnormalities", []):
            findings.append(f"{conduction['type']}: {conduction['description']}")
            recommendations.append(f" {conduction['recommendation']}")
        
        # Temel klinik bulgular
        hr = clinical_metrics.get("heart_rate_bpm")
        if hr:
            if 60 <= hr <= 100:
                findings.append(f"Normal kalp hızı: {hr:.0f} bpm")
        
        # HRV değerlendirmesi
        hrv = clinical_metrics.get("heart_rate_variability", {})
        if hrv:
            rmssd = hrv.get("rmssd_ms")
            if rmssd and 20 <= rmssd <= 50:
                findings.append(f"Normal kalp hızı variabilitesi (RMSSD: {rmssd:.1f} ms)")
        
        # Beat dağılım analizi
        normal_percentage = beat_percentages.get("Normal (N)", 0)
        if normal_percentage > 90:
            findings.append(f"Çoğunlukla normal kalp atımları (%{normal_percentage:.1f})")
        
        # Hiçbir anormal bulgu yoksa
        if risk_level == "Normal" and not findings:
            findings.append("Önemli kardiyak anormallik tespit edilmedi")
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
    
    def process_ecg_record(self, raw_ecg: np.ndarray, fs_in: int = 200, fs_out: int = 125) -> Dict:
        """
        Ana EKG işleme akışı.
        DÜZELTME: Eksik 'anomalies' argümanı eklendi.
        """
        try:
            logger.info(f"Analiz başladı - Uzunluk: {len(raw_ecg)}, Giriş Frekansı: {fs_in}Hz, Model Frekansı: {fs_out}Hz")
            
            # Adım 1: Sinyal Ön İşleme
            cleaned_signal = self.apply_filters(raw_ecg, fs_in)
            
            # Adım 2: R-Peak Tespiti
            r_peaks = self.robust_r_peak_detection(cleaned_signal, fs_in)
            logger.info(f"{len(r_peaks)} R-peak tespit edildi.")

            if len(r_peaks) < 2:
                # Yeterli R-peak yoksa, varsayılan bir özetle dön.
                ai_summary = self.generate_ai_summary([], {}, {}) # Boş argümanlarla çağır
                return {
                    "success": True, 
                    "display_signal": cleaned_signal.tolist(),
                    "r_peaks": [],
                    "beat_predictions": [],
                    "clinical_metrics": ai_summary['detailed_analysis']['clinical_parameters'], # Boş metrikler
                    "ai_summary": ai_summary,
                }

            # Adım 3: Klinik Metrikler
            clinical_metrics = self.calculate_advanced_clinical_metrics(cleaned_signal, r_peaks, fs_in)
            
            # Adım 4: Model için Sinyal Hazırlama
            resampled_signal = resample(cleaned_signal, int(len(cleaned_signal) * fs_out / fs_in)) if fs_in != fs_out else cleaned_signal
            r_peaks_resampled = [int(p * fs_out / fs_in) for p in r_peaks]
            r_peaks_resampled = [p for p in r_peaks_resampled if 0 <= p < len(resampled_signal)]

            # Adım 5: AI Tahminleri
            beats = self.extract_beats(resampled_signal, r_peaks_resampled)
            predictions = self.predict_beats(beats)
            

            # Adım 6: Anomali Tespiti
            # Bu fonksiyon, `generate_ai_summary` için gereken 'anomalies' objesini oluşturur.
            anomalies = self.detect_anomalies(predictions, clinical_metrics)
            
            # Adım 7: Gelişmiş AI Özetini Oluştur
            # Fonksiyonu artık doğru, 3 argümanla çağırıyoruz.
            ai_summary = self.generate_ai_summary(predictions, clinical_metrics, anomalies)
            
            # Adım 8: Sonucu hazırla
            result = {
                "success": True,
                "display_signal": cleaned_signal.tolist(),
                "r_peaks": r_peaks,
                "beat_predictions": predictions,
                "clinical_metrics": clinical_metrics,
                "anomalies": anomalies,
                "ai_summary": ai_summary,
            }
            logger.info("Analiz başarıyla tamamlandı.")
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
        sampling_rate = data.get('samplingRate', 200)
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
    logger.info("  - Corrected resampling: 1000Hz input → 125Hz output")
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
        logger.info("  - 1000Hz→125Hz optimized processing")
        logger.info("  - Risk assessment and clinical recommendations")
    else:
        logger.warning("Limited functionality - AI model not available")
    
    logger.info("=" * 60)
    logger.info("Processing Pipeline (CORRECTED):")
    logger.info("  1. Input: 1000Hz ECG signal")
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