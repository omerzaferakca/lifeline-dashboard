# main.py - GELİŞMİŞ EKG ANALİZİ VE ÖZELLİK ÇIKARIMI

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample
import logging
from typing import Dict, List
import neurokit2 as nk
from flask import Flask, request, jsonify
from flask_cors import CORS
import json

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ############################################################################
# ##                    JSON SERİALİZATİON YARDIMCI FONKSİYONLARI          ##
# ############################################################################

class NumpyEncoder(json.JSONEncoder):
    """NumPy array'lerini JSON'a çevirmek için özel encoder"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def safe_json_convert(data):
    """Veriyi JSON uyumlu hale getirir"""
    if isinstance(data, dict):
        return {key: safe_json_convert(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [safe_json_convert(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.integer, np.int64, np.int32)):
        return int(data)
    elif isinstance(data, (np.floating, np.float64, np.float32)):
        # NaN ve inf kontrolü
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, (int, float, str, bool)) and data is not None:
        # Normal Python tipleri için NaN/inf kontrolü
        if isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
            return None
        return data
    elif data is None:
        return None
    else:
        # Diğer tüm durumlar için string'e çevir
        try:
            return str(data)
        except:
            return None

# ############################################################################
# ##                          CNN MODEL MİMARİSİ                            ##
# ############################################################################

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ConvNormPool(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv1d(input_size, hidden_size, kernel_size)
        self.conv_2 = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        self.conv_3 = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        self.swish_1 = Swish()
        self.swish_2 = Swish()
        self.swish_3 = Swish()
        self.normalization_1 = nn.BatchNorm1d(hidden_size)
        self.normalization_2 = nn.BatchNorm1d(hidden_size)
        self.normalization_3 = nn.BatchNorm1d(hidden_size)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
    def forward(self, input):
        conv1 = self.conv_1(input)
        x = self.normalization_1(conv1)
        x = self.swish_1(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))
        
        x = self.conv_2(x)
        x = self.normalization_2(x)
        x = self.swish_2(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))
        
        conv3 = self.conv_3(x)
        x = self.normalization_3(conv1 + conv3)  # Residual Connection
        x = self.swish_3(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))
        
        x = self.pool(x)
        return x

class CNN(nn.Module):
    def __init__(self, input_size=1, hid_size=128, kernel_size=5, num_classes=5):
        super().__init__()
        self.conv1 = ConvNormPool(input_size, hid_size, kernel_size)
        self.conv2 = ConvNormPool(hid_size, hid_size//2, kernel_size)
        self.conv3 = ConvNormPool(hid_size//2, hid_size//4, kernel_size)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hid_size//4, num_classes)
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(-1, x.size(1) * x.size(2))
        x = F.softmax(self.fc(x), dim=1)
        return x

# ############################################################################
# ##                    GELİŞMİŞ ECG PROCESSOR SINIFI                       ##
# ############################################################################

class ECGProcessor:
    def __init__(self, model_path: str = "model.pth", device: str = "cpu"):
        self.device = device
        self.model = None
        self.model_path = model_path
        self.class_names = {
            0: "N",  # Normal
            1: "S",  # Atriyal Prematüre
            2: "V",  # Prematüre Ventriküler
            3: "F",  # Füzyon
            4: "Q"   # Paced / Diğer
        }
        self.load_model()
    
    def load_model(self):
        """Model dosyasını yükler"""
        try:
            self.model = CNN(num_classes=len(self.class_names), hid_size=128)
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model.to(self.device)
            logger.info(f"Model başarıyla yüklendi: {self.model_path}")
        except Exception as e:
            logger.error(f"Model yüklenirken hata: {e}")
            self.model = None
            raise

    def extract_beats(self, signal: np.ndarray, r_peaks: list, window: int = 187) -> np.ndarray:
        """R-peak'ler etrafında beat segmentlerini çıkarır"""
        if not r_peaks:
            return np.array([])
        
        half = window // 2
        beats = []
        
        for r in r_peaks:
            start = max(0, r - half)
            end = min(len(signal), r + half)
            beat = signal[start:end]
            
            # Beat boyutunu window boyutuna ayarla
            if len(beat) < window:
                pad_len = window - len(beat)
                beat = np.pad(beat, (pad_len // 2, pad_len - (pad_len // 2)), 'edge')
            elif len(beat) > window:
                beat = beat[:window]
                
            beats.append(beat)
        
        return np.array(beats)

    def predict_beats(self, beats: np.ndarray) -> List[Dict]:
        """Beat segmentleri için AI tahminleri yapar"""
        if self.model is None or len(beats) == 0:
            return []
        
        predictions = []
        with torch.no_grad():
            for i, beat in enumerate(beats):
                beat_tensor = torch.FloatTensor(beat).unsqueeze(0).unsqueeze(0).to(self.device)
                output_probabilities = self.model(beat_tensor)
                confidence, predicted_class_idx = torch.max(output_probabilities, dim=1)
                
                class_label = self.class_names.get(predicted_class_idx.item(), "Q")
                
                predictions.append({
                    "beat_id": int(i),
                    "predicted_class": int(predicted_class_idx.item()),
                    "class_name": str(class_label),
                    "confidence": float(round(confidence.item(), 4))
                })
        
        return predictions

    def summarize_ai_findings(self, predictions: List[Dict]) -> Dict:
        """AI tahminlerini klinik anlamlı özete dönüştürür"""
        if not predictions:
            return {
                "arrhythmia_level": "Bilinmiyor", 
                "risk_findings": ["Veri yetersiz."],
                "beat_distribution": {}
            }
        
        # Sınıf sayılarını hesapla
        counts = {label: 0 for label in self.class_names.values()}
        for pred in predictions:
            class_label = self.class_names.get(pred['predicted_class'], "Q")
            counts[class_label] += 1
        
        total_beats = len(predictions)
        
        # Aritmiya risk seviyesini belirle
        arrhythmia_level = "Düşük"
        v_ratio = counts["V"] / total_beats if total_beats > 0 else 0
        s_ratio = counts["S"] / total_beats if total_beats > 0 else 0
        
        if counts["V"] > 1 or v_ratio > 0.05:
            arrhythmia_level = "Yüksek"
        elif counts["V"] == 1 or s_ratio > 0.2:
            arrhythmia_level = "Orta"
        
        # Risk bulgularını oluştur
        risk_findings = []
        if counts["V"] > 0:
            risk_findings.append(f"{counts['V']} adet ventriküler ektopik atım (V) tespit edildi.")
        if counts["S"] > 5:
            risk_findings.append("Sık supraventriküler ektopik atımlar (S) gözlendi.")
        if counts["F"] > 0:
            risk_findings.append(f"{counts['F']} adet füzyon atımı (F) tespit edildi.")
        if counts["Q"] > 0:
            risk_findings.append(f"{counts['Q']} adet paced/diğer atım (Q) tespit edildi.")
        
        if not risk_findings:
            risk_findings.append("Belirgin bir ritim bozukluğu bulgusuna rastlanmadı.")
        
        return {
            "arrhythmia_level": str(arrhythmia_level),
            "risk_findings": risk_findings,
            "beat_distribution": {str(k): int(v) for k, v in counts.items()}
        }

    def calculate_clinical_metrics(self, ecg_signals: Dict, info: Dict) -> Dict:
        """NeuroKit2 çıktısından klinik metrikleri hesaplar"""
        try:
            # Güvenli metrik hesaplama
            def safe_mean(data_key):
                try:
                    if data_key in info and info[data_key] is not None:
                        values = info[data_key]
                        if hasattr(values, '__len__') and len(values) > 0:
                            # Liste veya array'e çevir
                            values = np.asarray(values).flatten()
                            # NaN olmayan değerleri filtrele
                            finite_values = values[np.isfinite(values)]
                            if len(finite_values) > 0:
                                return np.mean(finite_values)
                    return None
                except Exception as e:
                    logger.debug(f"Metrik hesaplama hatası {data_key}: {e}")
                    return None

            def safe_heart_rate():
                try:
                    if "ECG_Rate" in ecg_signals:
                        rate_values = ecg_signals["ECG_Rate"]
                        if hasattr(rate_values, '__len__') and len(rate_values) > 0:
                            rate_array = np.asarray(rate_values).flatten()
                            finite_rates = rate_array[np.isfinite(rate_array)]
                            if len(finite_rates) > 0:
                                return np.mean(finite_rates)
                    return None
                except Exception as e:
                    logger.debug(f"Kalp ritmi hesaplama hatası: {e}")
                    return None

            # Temel metrikleri güvenli şekilde hesapla
            heart_rate = safe_heart_rate()
            qrs_duration = safe_mean("ECG_QRS_Durations")
            pr_interval = safe_mean("ECG_PR_Intervals") 
            qt_interval = safe_mean("ECG_QT_Intervals")

            # Değerleri JSON uyumlu hale getir
            def safe_convert(value, multiplier=1):
                try:
                    if value is not None and np.isfinite(value):
                        converted = float(value * multiplier)
                        return round(converted, 1) if np.isfinite(converted) else None
                    return None
                except:
                    return None

            metrics = {
                "heart_rate": safe_convert(heart_rate),
                "qrs_duration": safe_convert(qrs_duration, 1000),  # saniye -> milisaniye
                "pr_interval": safe_convert(pr_interval, 1000),
                "qt_interval": safe_convert(qt_interval, 1000),
            }
            
            logger.info(f"Hesaplanan metrikler: {metrics}")
            return metrics
            
        except Exception as e:
            logger.warning(f"Klinik metrikleri hesaplanırken hata: {e}")
            return {
                "heart_rate": None,
                "qrs_duration": None,
                "pr_interval": None,
                "qt_interval": None
            }

    def process_ecg_record(self, raw_ecg: np.ndarray, fs_in: int = 1000, fs_out: int = 125) -> Dict:
        """
        Ham EKG sinyalini NeuroKit2 ile detaylı analiz eder ve AI tahminleri yapar
        """
        try:
            logger.info(f"Gelişmiş EKG analizi başladı... Sinyal uzunluğu: {len(raw_ecg)}")
            
            # Veri temizliği ve format kontrolü
            raw_ecg = np.asarray(raw_ecg, dtype=np.float64).flatten()
            
            # Çok büyük dosyalar için örnekleme (memory management)
            max_length = 100000  # ~100 saniye @ 1000Hz
            if len(raw_ecg) > max_length:
                logger.info(f"Büyük dosya tespit edildi ({len(raw_ecg)} sample), ilk {max_length} sample alınıyor")
                raw_ecg = raw_ecg[:max_length]
            
            # NaN ve inf değerleri temizle
            finite_mask = np.isfinite(raw_ecg)
            if not np.all(finite_mask):
                logger.warning("NaN veya inf değerler tespit edildi, interpolasyon yapılıyor")
                raw_ecg = np.interp(np.arange(len(raw_ecg)), 
                                   np.arange(len(raw_ecg))[finite_mask], 
                                   raw_ecg[finite_mask])
            
            # NeuroKit2 ile sinyal işleme ve dalga tespiti
            try:
                logger.info("NeuroKit2 ile EKG işleme başlıyor...")
                ecg_signals, info = nk.ecg_process(raw_ecg, sampling_rate=fs_in)
                logger.info("NeuroKit2 işleme tamamlandı")
            except Exception as nk_error:
                logger.error(f"NeuroKit2 işleme hatası: {nk_error}")
                # Fallback: minimal processing
                ecg_signals = {"ECG_Clean": raw_ecg, "ECG_Rate": np.full(len(raw_ecg), np.nan)}
                info = {}
            
            # Klinik metrikleri hesapla
            clinical_metrics = self.calculate_clinical_metrics(ecg_signals, info)
            
            # Model için sinyali hazırla
            clean_signal = ecg_signals.get("ECG_Clean", raw_ecg)
            clean_signal = np.asarray(clean_signal, dtype=np.float64).flatten()
            
            # Downsampling
            try:
                if len(clean_signal) > 0:
                    target_length = int(len(clean_signal) * fs_out / fs_in)
                    if target_length > 0:
                        downsampled = resample(clean_signal, target_length)
                    else:
                        downsampled = clean_signal
                else:
                    downsampled = clean_signal
            except Exception as resample_error:
                logger.warning(f"Resampling hatası: {resample_error}, orijinal sinyal kullanılıyor")
                downsampled = clean_signal
            
            # Normalizasyon
            if len(downsampled) > 0:
                signal_std = np.std(downsampled)
                if signal_std > 1e-10:  # Çok küçük standart sapma kontrolü
                    normalized_signal = (downsampled - np.mean(downsampled)) / signal_std
                else:
                    normalized_signal = downsampled - np.mean(downsampled)
            else:
                normalized_signal = downsampled
            
            # R-peak tespiti (daha güvenli)
            r_peak_indices = []
            if len(normalized_signal) > fs_out:  # En az 1 saniye sinyal gerekli
                try:
                    _, rpeaks_info = nk.ecg_peaks(normalized_signal, sampling_rate=fs_out)
                    r_peak_indices = rpeaks_info.get('ECG_R_Peaks', [])
                    if isinstance(r_peak_indices, np.ndarray):
                        r_peak_indices = r_peak_indices.tolist()
                    logger.info(f"{len(r_peak_indices)} R-peak tespit edildi")
                except Exception as peak_error:
                    logger.warning(f"R-peak tespiti başarısız: {peak_error}")
                    r_peak_indices = []
            
            # AI tahminleri
            beats = self.extract_beats(normalized_signal, r_peak_indices, window=187)
            predictions = self.predict_beats(beats)
            ai_summary = self.summarize_ai_findings(predictions)
            
            # Signal'ı büyük dosyalar için kısalt (frontend için)
            display_signal = normalized_signal
            if len(display_signal) > 10000:  # Frontend'de görüntülemek için max 10k sample
                step = len(display_signal) // 10000
                display_signal = display_signal[::step]
            
            # Sonuçları JSON uyumlu hale getir
            result = {
                "success": True,
                "processed_signal": display_signal.tolist(),
                "r_peaks": [int(peak) for peak in r_peak_indices if isinstance(peak, (int, np.integer, float, np.floating))],
                "predictions": predictions,
                "clinical_features": clinical_metrics,
                "ai_summary": ai_summary,
                "total_beats_detected": int(len(r_peak_indices)),
                "original_length": int(len(raw_ecg)),
                "processed_length": int(len(normalized_signal))
            }
            
            # Son güvenlik kontrolü
            result = safe_json_convert(result)
            
            logger.info("Gelişmiş EKG analizi başarıyla tamamlandı.")
            return result
            
        except Exception as e:
            logger.error(f"EKG analizi sırasında hata: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}

# ############################################################################
# ##                           FLASK WEB API                                ##
# ############################################################################

app = Flask(__name__)
CORS(app)

# JSON encoder'ı ayarla
app.json_encoder = NumpyEncoder

# Processor'ı başlat
try:
    processor = ECGProcessor(model_path="model.pth")
    logger.info("EKG işleyicisi başarıyla başlatıldı.")
except Exception as e:
    processor = None
    logger.critical(f"EKG işleyicisi başlatılamadı: {e}")

@app.route('/process-ecg', methods=['POST'])
def process_ecg_endpoint():
    """EKG analizi endpoint'i"""
    if processor is None:
        return jsonify({
            "success": False, 
            "error": "Model yüklenemedi, sunucu servis veremiyor."
        }), 500
    
    try:
        json_data = request.get_json()
        if not json_data or 'ecg_signal' not in json_data:
            return jsonify({
                "success": False, 
                "error": "'ecg_signal' verisi gerekli"
            }), 400
        
        ecg_signal = np.array(json_data['ecg_signal'], dtype=np.float32)
        sampling_rate = json_data.get('sampling_rate', 1000)
        
        # EKG analizi
        result = processor.process_ecg_record(ecg_signal, fs_in=sampling_rate)
        
        # JSON serializasyon için son kontrol
        result = safe_json_convert(result)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API endpoint hatası: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Sunucu sağlık kontrolü"""
    return jsonify({
        "status": "ok",
        "model_loaded": processor is not None,
        "neurokit2_available": True
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Model ve sınıf bilgileri"""
    if processor is None:
        return jsonify({"error": "Model yüklü değil"}), 500
    
    return jsonify({
        "class_names": {str(k): str(v) for k, v in processor.class_names.items()},
        "model_path": str(processor.model_path),
        "device": str(processor.device)
    })

if __name__ == '__main__':
    logger.info("Flask sunucusu http://127.0.0.1:5001 adresinde başlatılıyor...")
    app.run(debug=True, port=5001)