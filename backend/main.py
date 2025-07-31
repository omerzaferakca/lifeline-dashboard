# main.py - EN OKUNAKLI, TAM VE HATALARI DÜZELTİLMİŞ VERSİYON

# --- Gerekli Kütüphaneler ---
import sqlite3
import json
import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample
import neurokit2 as nk
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Logging Ayarları ---
# Uygulama genelinde olayları kaydetmek için bir logger yapılandırıyoruz.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ############################################################################
# ##                          VERİTABANI İŞLEMLERİ                           ##
# ############################################################################

DATABASE_FILE = 'database.db'

def get_db_connection():
    """Veritabanına bir bağlantı oluşturur ve döndürür."""
    conn = sqlite3.connect(DATABASE_FILE)
    # Sonuçları sözlük gibi erişilebilir hale getirir (örn: row['name'])
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Veritabanı şemasını (tabloları) kontrol eder ve eğer yoksa oluşturur.
    Sunucu ilk başladığında bir kez çalıştırılır.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Hastalar tablosu
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            tc TEXT UNIQUE NOT NULL,
            age INTEGER,
            gender TEXT,
            phone TEXT,
            medications TEXT, -- JSON formatında ['ilaç1', 'ilaç2']
            complaints TEXT
        )
        ''')
        
        # EKG Dosyaları tablosu
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ekg_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            file_name TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            sampling_rate INTEGER,
            ekg_data BLOB NOT NULL, -- Ham EKG verisini ikili formatta sakla
            FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Veritabanı başarıyla başlatıldı veya zaten mevcuttu.")
    except Exception as e:
        logger.critical(f"Veritabanı başlatılırken KRİTİK HATA: {e}")


# ############################################################################
# ##                          CNN MODEL MİMARİSİ                            ##
# ############################################################################
# Bu bölümde bir değişiklik yok.
class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)
class ConvNormPool(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__(); self.kernel_size = kernel_size; self.conv_1 = nn.Conv1d(input_size, hidden_size, kernel_size); self.conv_2 = nn.Conv1d(hidden_size, hidden_size, kernel_size); self.conv_3 = nn.Conv1d(hidden_size, hidden_size, kernel_size); self.swish_1, self.swish_2, self.swish_3 = Swish(), Swish(), Swish(); self.normalization_1 = nn.BatchNorm1d(hidden_size); self.normalization_2 = nn.BatchNorm1d(hidden_size); self.normalization_3 = nn.BatchNorm1d(hidden_size); self.pool = nn.MaxPool1d(kernel_size=2)
    def forward(self, input):
        conv1 = self.conv_1(input); x = self.normalization_1(conv1); x = self.swish_1(x); x = F.pad(x, (self.kernel_size - 1, 0)); x = self.conv_2(x); x = self.normalization_2(x); x = self.swish_2(x); x = F.pad(x, (self.kernel_size - 1, 0)); conv3 = self.conv_3(x); x = self.normalization_3(conv1 + conv3); x = self.swish_3(x); x = F.pad(x, (self.kernel_size - 1, 0)); return self.pool(x)
class CNN(nn.Module):
    def __init__(self, input_size=1, hid_size=128, kernel_size=5, num_classes=5):
        super().__init__(); self.conv1 = ConvNormPool(input_size, hid_size, kernel_size); self.conv2 = ConvNormPool(hid_size, hid_size//2, kernel_size); self.conv3 = ConvNormPool(hid_size//2, hid_size//4, kernel_size); self.avgpool = nn.AdaptiveAvgPool1d(1); self.fc = nn.Linear(hid_size//4, num_classes)
    def forward(self, input):
        x = self.conv1(input); x = self.conv2(x); x = self.conv3(x); x = self.avgpool(x); x = x.view(-1, x.size(1) * x.size(2)); return F.softmax(self.fc(x), dim=1)


# ############################################################################
# ##                    GELİŞMİŞ ECG PROCESSOR SINIFI                       ##
# ############################################################################

class ECGProcessor:
    def __init__(self, model_path: str = "model.pth", device: str = "cpu"):
        self.device = device
        self.model = None
        self.model_path = model_path
        self.class_names = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
        self.load_model()
    
    def load_model(self):
        try:
            self.model = CNN(num_classes=len(self.class_names), hid_size=128)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            self.model.to(self.device)
            logger.info(f"Model başarıyla yüklendi: {self.model_path}")
        except Exception as e:
            logger.error(f"Model yüklenirken hata: {e}")
            self.model = None
            raise

    def calculate_clinical_metrics(self, ecg_signals: Dict, info: Dict) -> Dict:
        """
        NeuroKit2 çıktısından klinik metrikleri güvenli bir şekilde hesaplar.
        "Ambiguous" hatasını önlemek için daha sağlam hale getirildi.
        """
        results = {
            "heart_rate": None,
            "qrs_duration": None,
            "pr_interval": None,
            "qt_interval": None
        }
        try:
            # Kalp Hızı
            hr_data = ecg_signals.get("ECG_Rate")
            if hr_data is not None:
                # np.nanmean, NaN değerleri yok sayarak ortalama alır.
                hr_mean = np.nanmean(hr_data)
                # Sonucun hala geçerli bir sayı olup olmadığını kontrol et
                if np.isfinite(hr_mean):
                    results["heart_rate"] = float(hr_mean)
            
            # Diğer metrikler (QRS, PR, QT)
            metric_keys = {
                "ECG_QRS_Durations": "qrs_duration",
                "ECG_PR_Intervals": "pr_interval",
                "ECG_QT_Intervals": "qt_interval"
            }
            
            for nk_key, result_key in metric_keys.items():
                data_array = info.get(nk_key)
                if data_array is not None:
                    # Gelen veriyi her zaman bir numpy dizisine çevir
                    values = np.array(data_array, dtype=np.float64).flatten()
                    # Sadece geçerli (sonlu) sayısal değerleri al
                    finite_values = values[np.isfinite(values)]
                    if finite_values.size > 0:
                        mean_val = np.mean(finite_values)
                        # saniyeden milisaniyeye çevir ve float yap
                        results[result_key] = float(mean_val * 1000)
        except Exception as e:
            logger.warning(f"Klinik metrikler hesaplanırken bir istisna oluştu: {e}")

        return results

    def predict_beats(self, beats: np.ndarray) -> List[Dict]:
        if self.model is None or not len(beats): return []
        predictions = []
        with torch.no_grad():
            for i, beat in enumerate(beats):
                tensor = torch.FloatTensor(beat).unsqueeze(0).unsqueeze(0).to(self.device)
                probabilities = self.model(tensor)
                confidence, class_idx = torch.max(probabilities, dim=1)
                
                # Sonuçları JSON uyumlu standart Python tiplerine çevir
                predictions.append({
                    "beat_id": int(i),
                    "predicted_class": int(class_idx.item()),
                    "class_name": self.class_names.get(class_idx.item(), "Q"),
                    "confidence": float(confidence.item())
                })
        return predictions

    def summarize_ai_findings(self, predictions: List[Dict]) -> Dict:
        if not predictions:
            return {"arrhythmia_level": "Bilinmiyor", "risk_findings": ["Analiz için yeterli kalp atışı bulunamadı."]}
        
        counts = {label: 0 for label in self.class_names.values()}
        for p in predictions:
            counts[self.class_names.get(p['predicted_class'], "Q")] += 1
            
        total_beats = len(predictions)
        v_ratio = counts["V"] / total_beats if total_beats > 0 else 0
        s_ratio = counts["S"] / total_beats if total_beats > 0 else 0
        
        level = "Düşük"
        if counts["V"] > 1 or v_ratio > 0.05:
            level = "Yüksek"
        elif counts["V"] == 1 or s_ratio > 0.2:
            level = "Orta"
            
        findings = []
        if counts["V"] > 0: findings.append(f"{counts['V']} adet ventriküler atım (V) tespit edildi.")
        if counts["S"] > 5: findings.append(f"Sık supraventriküler atım (S) gözlendi.")
        if not findings: findings.append("Belirgin bir ritim bozukluğu bulgusuna rastlanmadı.")
        
        return {"arrhythmia_level": level, "risk_findings": findings}

    def extract_beats(self, signal: np.ndarray, r_peaks: list, window: int = 187) -> np.ndarray:
        if not r_peaks: return np.array([])
        half = window // 2; beats = []
        for r in r_peaks:
            start, end = max(0, r - half), min(len(signal), r + half)
            beat = signal[start:end]
            if len(beat) < window: beat = np.pad(beat, (0, window - len(beat)), 'edge')
            elif len(beat) > window: beat = beat[:window]
            beats.append(beat)
        return np.array(beats)
        
    def process_ecg_record(self, raw_ecg: np.ndarray, fs_in: int = 1000, fs_out: int = 125) -> Dict:
        try:
            logger.info(f"EKG analizi başladı... Örneklem: {len(raw_ecg)}, Frekans: {fs_in}Hz")
            
            ecg_signals, info = nk.ecg_process(raw_ecg, sampling_rate=fs_in)
            clinical_metrics = self.calculate_clinical_metrics(ecg_signals, info)
            
            # Grafik için genliği korunmuş temiz sinyal
            display_signal = ecg_signals.get("ECG_Clean", raw_ecg)
            
            # Model için yeniden örneklenmiş ve normalize edilmiş sinyal
            downsampled = resample(display_signal, int(len(display_signal) * fs_out / fs_in))
            normalized_signal = (downsampled - np.mean(downsampled)) / (np.std(downsampled) + 1e-8)
            
            _, rpeaks_info = nk.ecg_peaks(normalized_signal, sampling_rate=fs_out)
            r_peak_indices = rpeaks_info.get('ECG_R_Peaks', [])
            
            beats = self.extract_beats(normalized_signal, r_peak_indices, window=187)
            predictions = self.predict_beats(beats)
            ai_summary = self.summarize_ai_findings(predictions)
            
            result = {
                "success": True,
                "display_signal": display_signal.tolist(),
                "r_peaks": [int(p) for p in r_peak_indices],
                "predictions": predictions,
                "clinical_features": clinical_metrics,
                "ai_summary": ai_summary,
            }
            logger.info("EKG analizi başarıyla tamamlandı.")
            return result
        except Exception as e:
            logger.error(f"EKG analizi sırasında kritik hata: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

# ############################################################################
# ##                           FLASK WEB API                                ##
# ############################################################################

app = Flask(__name__)
CORS(app)

try:
    processor = ECGProcessor(model_path="model.pth")
except Exception as e:
    processor = None
    logger.critical(f"Kritik Hata: EKG işleyicisi başlatılamadı: {e}")

# --- HASTA API ENDPOINT'LERİ ---
@app.route('/api/patients', methods=['GET'])
def get_patients():
    conn = get_db_connection()
    patients_cursor = conn.execute('SELECT * FROM patients ORDER BY name ASC').fetchall()
    conn.close()
    # Veritabanı satırlarını Python sözlüklerine çevir
    patients = [dict(row) for row in patients_cursor]
    return jsonify(patients)

@app.route('/api/patients', methods=['POST'])
def add_patient():
    data = request.get_json()
    conn = get_db_connection()
    conn.execute('INSERT INTO patients (name, tc, age, gender, phone, medications, complaints) VALUES (?, ?, ?, ?, ?, ?, ?)',
                 (data['name'], data['tc'], data['age'], data['gender'], data['phone'], json.dumps(data.get('medications', [])), data.get('complaints', '')))
    conn.commit()
    conn.close()
    return jsonify({"message": "Hasta başarıyla eklendi."}), 201

@app.route('/api/patients/<int:id>', methods=['PUT'])
def update_patient(id):
    data = request.get_json()
    conn = get_db_connection()
    conn.execute('UPDATE patients SET name=?, tc=?, age=?, gender=?, phone=?, medications=?, complaints=? WHERE id=?',
                 (data['name'], data['tc'], data['age'], data['gender'], data['phone'], json.dumps(data.get('medications', [])), data.get('complaints', ''), id))
    conn.commit()
    conn.close()
    return jsonify({"message": "Hasta bilgileri güncellendi."})

@app.route('/api/patients/<int:id>', methods=['DELETE'])
def delete_patient(id):
    conn = get_db_connection()
    # İlişkili dosyaları da sil (ON DELETE CASCADE ile otomatikleşti)
    conn.execute('DELETE FROM patients WHERE id=?', (id,))
    conn.commit()
    conn.close()
    return jsonify({"message": "Hasta ve tüm kayıtları silindi."})

# --- EKG DOSYA API ENDPOINT'LERİ ---
@app.route('/api/patients/<int:patient_id>/files', methods=['GET'])
def get_patient_files(patient_id):
    conn = get_db_connection()
    files_cursor = conn.execute('SELECT id, file_name, uploaded_at FROM ekg_files WHERE patient_id=? ORDER BY uploaded_at DESC', (patient_id,)).fetchall()
    conn.close()
    files = [dict(row) for row in files_cursor]
    return jsonify(files)

@app.route('/api/patients/<int:patient_id>/files', methods=['POST'])
def upload_file(patient_id):
    data = request.get_json()
    ekg_data_np = np.array(data['data'], dtype=np.float32)
    conn = get_db_connection()
    conn.execute('INSERT INTO ekg_files (patient_id, file_name, uploaded_at, sampling_rate, ekg_data) VALUES (?, ?, ?, ?, ?)',
                 (patient_id, data['name'], data['uploadedAt'], data.get('samplingRate', 1000), ekg_data_np.tobytes()))
    conn.commit()
    conn.close()
    return jsonify({"message": "Dosya başarıyla yüklendi."}), 201

@app.route('/api/analyze/<int:file_id>', methods=['GET'])
def analyze_file(file_id):
    if processor is None:
        return jsonify({"success": False, "error": "Analiz servisi aktif değil."}), 503
    
    conn = get_db_connection()
    file_record = conn.execute('SELECT ekg_data, sampling_rate FROM ekg_files WHERE id=?', (file_id,)).fetchone()
    conn.close()
    
    if not file_record:
        return jsonify({"success": False, "error": "Dosya bulunamadı"}), 404
    
    ekg_data = np.frombuffer(file_record['ekg_data'], dtype=np.float32)
    result = processor.process_ecg_record(ekg_data, fs_in=file_record['sampling_rate'])
    return jsonify(result)

if __name__ == '__main__':
    init_db()  # Sunucu başlamadan önce veritabanını kontrol et/oluştur
    logger.info("Flask sunucusu http://12ז.0.0.1:5001 adresinde başlatılıyor...")
    app.run(debug=True, port=5001)