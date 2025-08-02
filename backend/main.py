import sqlite3
import json
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample
import neurokit2 as nk
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Logging Ayarları ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ############################################################################
# ##                          VERİTABANI İŞLEMLERİ                           ##
# ############################################################################
DATABASE_FILE = 'database.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            tc TEXT UNIQUE NOT NULL,
            age INTEGER,
            gender TEXT,
            phone TEXT,
            medications TEXT,
            complaints TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ekg_files (
            id INTEGER PRIMARY KEY,
            patient_id INTEGER NOT NULL,
            file_name TEXT,
            uploaded_at TEXT,
            sampling_rate INTEGER,
            ekg_data BLOB,
            FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Veritabanı başarıyla başlatıldı.")


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
        self.swish_1, self.swish_2, self.swish_3 = Swish(), Swish(), Swish()
        self.normalization_1 = nn.BatchNorm1d(hidden_size)
        self.normalization_2 = nn.BatchNorm1d(hidden_size)
        self.normalization_3 = nn.BatchNorm1d(hidden_size)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, input):
        conv1 = self.conv_1(input)
        x = self.normalization_1(conv1)
        x = self.swish_1(x)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv_2(x)
        x = self.normalization_2(x)
        x = self.swish_2(x)
        x = F.pad(x, (self.kernel_size - 1, 0))
        conv3 = self.conv_3(x)
        x = self.normalization_3(conv1 + conv3)
        x = self.swish_3(x)
        x = F.pad(x, (self.kernel_size - 1, 0))
        return self.pool(x)

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
        return F.softmax(self.fc(x), dim=1)


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
            self.model.eval().to(self.device)
            logger.info(f"Model başarıyla yüklendi: {self.model_path}")
        except Exception as e:
            logger.error(f"Model yüklenirken hata: {e}")
            self.model = None
            raise

    def predict_beats(self, beats: np.ndarray) -> List[Dict]:
        if self.model is None or not len(beats):
            return []
        
        predictions = []
        with torch.no_grad():
            for i, beat in enumerate(beats):
                tensor = torch.FloatTensor(beat).unsqueeze(0).unsqueeze(0).to(self.device)
                probs = self.model(tensor)
                conf, idx = torch.max(probs, dim=1)
                predictions.append({
                    "beat_id": int(i),
                    "predicted_class": int(idx.item()),
                    "class_name": self.class_names.get(idx.item(), "Q"),
                    "confidence": float(conf.item())
                })
        return predictions

    def summarize_ai_findings(self, predictions: List[Dict]) -> Dict:
        if not predictions:
            return {
                "arrhythmia_level": "Bilinmiyor",
                "risk_findings": ["Analiz için yeterli kalp atışı bulunamadı."]
            }

        counts = {label: 0 for label in self.class_names.values()}
        for p in predictions:
            counts[self.class_names.get(p['predicted_class'], "Q")] += 1

        total = len(predictions)
        v_ratio = counts["V"] / total if total > 0 else 0
        s_ratio = counts["S"] / total if total > 0 else 0
        level = "Düşük"

        if counts["V"] > 1 or v_ratio > 0.05:
            level = "Yüksek"
        elif counts["V"] == 1 or s_ratio > 0.2:
            level = "Orta"

        findings = []
        if counts["V"] > 0:
            findings.append(f"{counts['V']} adet ventriküler atım (V) tespit edildi.")
        if counts["S"] > 5:
            findings.append(f"Sık supraventriküler atım (S) gözlendi.")
        if not findings:
            findings.append("Belirgin bir ritim bozukluğu bulgusuna rastlanmadı.")

        return {"arrhythmia_level": level, "risk_findings": findings}

    def extract_beats(self, signal: np.ndarray, r_peaks: list, window: int = 187) -> np.ndarray:
        if not r_peaks:
            return np.array([])
        
        half = window // 2
        beats = []
        for r in r_peaks:
            start, end = max(0, r - half), min(len(signal), r + half)
            beat = signal[start:end]
            if len(beat) < window:
                beat = np.pad(beat, (0, window - len(beat)), 'edge')
            elif len(beat) > window:
                beat = beat[:window]
            beats.append(beat)
        return np.array(beats)

    def _safe_wave_check(self, waves: Dict, wave_key: str) -> Optional[np.ndarray]:
        """Güvenli wave kontrolü - None ve boş array kontrolü"""
        try:
            wave_data = waves.get(wave_key)
            logger.debug(f"Wave {wave_key}: type={type(wave_data)}, value={wave_data}")
            
            # İlk None kontrolü
            if wave_data is None:
                logger.debug(f"Wave {wave_key} is None")
                return None
            
            # Liste veya array kontrolü
            if hasattr(wave_data, '__len__') and len(wave_data) == 0:
                logger.debug(f"Wave {wave_key} is empty")
                return None
            
            # NumPy array'e çevir
            wave_array = np.asarray(wave_data)
            logger.debug(f"Wave {wave_key} array shape: {wave_array.shape}, dtype: {wave_array.dtype}")
            
            # Boş array kontrolü
            if wave_array.size == 0:
                logger.debug(f"Wave {wave_key} array is empty")
                return None
            
            # Tek boyutlu array'e çevir
            wave_array = wave_array.flatten()
            
            # NaN kontrolü - tüm değerler NaN mı?
            if wave_array.size > 0 and np.all(np.isnan(wave_array)):
                logger.debug(f"Wave {wave_key} all values are NaN")
                return None
            
            # Geçerli (NaN olmayan) değerler var mı?
            valid_mask = ~np.isnan(wave_array)
            if not np.any(valid_mask):
                logger.debug(f"Wave {wave_key} no valid values")
                return None
                
            return wave_array[valid_mask]
            
        except Exception as e:
            logger.error(f"Wave {wave_key} kontrolünde hata: {e}", exc_info=True)
            return None

    def _calculate_clinical_metrics(self, ecg_cleaned: np.ndarray, r_peaks: list, fs_in: int) -> Dict:
        """Klinik metrikleri güvenli ve doğru bir akışla hesaplar."""
        
        # 1. Varsayılan (boş) sonuçları oluştur
        clinical_metrics = {
            "heart_rate": None,
            "qrs_duration": None,
            "pr_interval": None,
            "qt_interval": None
        }
        
        # 2. Yeterli R-peak yoksa, boş sonuçlarla hemen geri dön
        if len(r_peaks) < 2:
            logger.warning("Yeterli R-peak bulunamadı, klinik metrikler hesaplanamıyor.")
            return clinical_metrics

        # 3. Yeterli R-peak varsa, hesaplamalara başla
        try:
            # Kalp Hızı Hesaplama
            hr = nk.ecg_rate(r_peaks, sampling_rate=fs_in, desired_length=len(ecg_cleaned))
            hr_mean = np.nanmean(hr)
            if np.isfinite(hr_mean):
                clinical_metrics["heart_rate"] = float(hr_mean)

            # Dalga Tespiti (Delineation)
            # Bu adım gürültülü sinyallerde hata verebilir, bu yüzden try-except içine alıyoruz.
            waves = {}
            try:
                _, waves = nk.ecg_delineate(ecg_cleaned, r_peaks, sampling_rate=fs_in, method="dwt")
            except Exception as e:
                logger.warning(f"Neurokit Delineate başarısız oldu: {e}. Metrikler hesaplanamayabilir.")
                # Delineate başarısız olsa bile, en azından kalp hızı hesaplanmış olur.
                # Bu yüzden burada fonksiyondan çıkıyoruz.
                return clinical_metrics

            # 4. Dalgalar başarıyla bulunduysa, metrikleri hesapla
            
            # QRS Süresi (S_Offset - Q_Peak)
            q_peaks = waves.get("ECG_Q_Peaks")
            s_offsets = waves.get("ECG_S_Offsets")
            if q_peaks is not None and s_offsets is not None and len(q_peaks) == len(s_offsets):
                qrs_durations = (np.array(s_offsets) - np.array(q_peaks)) / fs_in * 1000  # ms
                qrs_mean = np.nanmean(qrs_durations)
                if np.isfinite(qrs_mean) and qrs_mean > 0:
                    clinical_metrics["qrs_duration"] = float(qrs_mean)

            # PR Aralığı (R_Onset - P_Onset)
            p_onsets = waves.get("ECG_P_Onsets")
            r_onsets = waves.get("ECG_R_Onsets")
            if p_onsets is not None and r_onsets is not None and len(p_onsets) == len(r_onsets):
                pr_intervals = (np.array(r_onsets) - np.array(p_onsets)) / fs_in * 1000  # ms
                pr_mean = np.nanmean(pr_intervals)
                if np.isfinite(pr_mean) and pr_mean > 0:
                    clinical_metrics["pr_interval"] = float(pr_mean)

            # QT Aralığı (T_Offset - Q_Peak)
            t_offsets = waves.get("ECG_T_Offsets")
            if q_peaks is not None and t_offsets is not None and len(q_peaks) == len(t_offsets):
                qt_intervals = (np.array(t_offsets) - np.array(q_peaks)) / fs_in * 1000  # ms
                qt_mean = np.nanmean(qt_intervals)
                if np.isfinite(qt_mean) and qt_mean > 0:
                    clinical_metrics["qt_interval"] = float(qt_mean)

        except Exception as e:
            logger.error(f"Klinik metrik hesaplamasında genel bir hata oluştu: {e}", exc_info=True)

        logger.info(f"Hesaplanan Nihai Metrikler: {clinical_metrics}")
        return clinical_metrics

    def process_ecg_record(self, raw_ecg: np.ndarray, fs_in: int = 1000, fs_out: int = 125) -> Dict:
        try:
            logger.info(f"EKG analizi başladı... Örneklem: {len(raw_ecg)}, Frekans: {fs_in}Hz")
            
            # Giriş verisi kontrolü
            if len(raw_ecg) == 0:
                return {"success": False, "error": "Boş EKG verisi"}
                
            if not np.isfinite(raw_ecg).all():
                logger.warning("EKG verisinde sonsuz/NaN değerler bulundu, temizleniyor...")
                raw_ecg = np.nan_to_num(raw_ecg, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Adım 1: Sinyali temizle
            try:
                ecg_cleaned = nk.ecg_clean(raw_ecg, sampling_rate=fs_in, method='neurokit')
            except Exception as e:
                logger.warning(f"NeuroKit temizleme hatası, basit filtreleme kullanılıyor: {e}")
                # Basit filtreleme
                ecg_cleaned = raw_ecg - np.mean(raw_ecg)
                ecg_cleaned = ecg_cleaned / (np.std(ecg_cleaned) + 1e-8)
            
            # Adım 2: R-peak'leri bul
            try:
                _, rpeaks_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs_in)
                r_peaks_raw = rpeaks_info.get('ECG_R_Peaks', [])
                
                # R-peaks'i güvenli şekilde listeye çevir
                r_peaks = []
                if hasattr(r_peaks_raw, 'tolist'):
                    r_peaks = r_peaks_raw.tolist()
                elif hasattr(r_peaks_raw, '__iter__'):
                    r_peaks = list(r_peaks_raw)
                else:
                    r_peaks = []
                    
                logger.debug(f"Bulunan R-peak sayısı: {len(r_peaks)}")
                
            except Exception as e:
                logger.error(f"R-peak bulma hatası: {e}")
                r_peaks = []

            # Adım 3: Klinik metrikleri hesapla
            clinical_metrics = self._calculate_clinical_metrics(ecg_cleaned, r_peaks, fs_in)

            # Adım 4: Model için sinyali yeniden örnekle ve normalize et
            try:
                if fs_in != fs_out:
                    downsampled_signal = resample(ecg_cleaned, int(len(ecg_cleaned) * fs_out / fs_in))
                else:
                    downsampled_signal = ecg_cleaned.copy()
                    
                normalized_signal = (downsampled_signal - np.mean(downsampled_signal)) / (np.std(downsampled_signal) + 1e-8)
            except Exception as e:
                logger.error(f"Sinyal yeniden örnekleme hatası: {e}")
                normalized_signal = ecg_cleaned
            
            # Model için R-peak'leri yeniden örneklenmiş sinyalde bul
            try:
                _, rpeaks_info_ds = nk.ecg_peaks(normalized_signal, sampling_rate=fs_out)
                r_peak_indices_ds = rpeaks_info_ds.get('ECG_R_Peaks', [])
                
                # R-peak'leri güvenli şekilde listeye çevir
                if hasattr(r_peak_indices_ds, 'tolist'):
                    r_peak_indices_ds = r_peak_indices_ds.tolist()
                elif not isinstance(r_peak_indices_ds, list):
                    r_peak_indices_ds = list(r_peak_indices_ds)
                    
            except Exception as e:
                logger.warning(f"Downsampled R-peak bulma hatası: {e}")
                # Orijinal R-peak'leri ölçekle
                r_peak_indices_ds = []
                if r_peaks and fs_in != fs_out:
                    for p in r_peaks:
                        try:
                            # Güvenli şekilde integer'a çevir ve ölçekle
                            if hasattr(p, 'item'):
                                scaled_p = int(p.item() * fs_out / fs_in)
                            else:
                                scaled_p = int(p * fs_out / fs_in)
                            
                            if 0 <= scaled_p < len(normalized_signal):
                                r_peak_indices_ds.append(scaled_p)
                        except (ValueError, TypeError) as pe:
                            logger.warning(f"R-peak ölçekleme hatası: {p}, hata: {pe}")
                            continue
                else:
                    # fs_in == fs_out durumu
                    for p in r_peaks:
                        try:
                            if hasattr(p, 'item'):
                                r_peak_indices_ds.append(int(p.item()))
                            else:
                                r_peak_indices_ds.append(int(p))
                        except (ValueError, TypeError) as pe:
                            logger.warning(f"R-peak çevirme hatası: {p}, hata: {pe}")
                            continue
            
            # Adım 5: AI Tahminleri
            predictions = []
            ai_summary = {"arrhythmia_level": "Bilinmiyor", "risk_findings": ["Model analizi yapılamadı."]}
            
            try:
                beats = self.extract_beats(normalized_signal, r_peak_indices_ds, window=187)
                predictions = self.predict_beats(beats)
                ai_summary = self.summarize_ai_findings(predictions)
            except Exception as e:
                logger.error(f"AI analizi hatası: {e}")
            
            # Adım 6: Sonucu oluştur
            # R-peaks'i güvenli şekilde integer'a çevir
            safe_r_peaks = []
            if r_peaks:
                for p in r_peaks:
                    try:
                        # NumPy array, scalar veya normal sayı olabilir
                        if hasattr(p, 'item'):  # NumPy scalar
                            safe_r_peaks.append(int(p.item()))
                        elif hasattr(p, '__len__') and len(p) == 1:  # Tek elemanlı array
                            safe_r_peaks.append(int(p[0]))
                        else:  # Normal sayı
                            safe_r_peaks.append(int(p))
                    except (ValueError, TypeError, IndexError) as e:
                        logger.warning(f"R-peak değeri çevrilemedi: {p}, hata: {e}")
                        continue
            
            result = {
                "success": True,
                "display_signal": ecg_cleaned.tolist(),
                "r_peaks": safe_r_peaks,
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
    logger.critical(f"Kritik Hata: {e}")

@app.route('/api/patients', methods=['GET'])
def get_patients():
    conn = get_db_connection()
    p_cursor = conn.execute('SELECT * FROM patients ORDER BY name ASC').fetchall()
    conn.close()
    return jsonify([dict(row) for row in p_cursor])

@app.route('/api/patients', methods=['POST'])
def add_patient():
    data = request.get_json()
    conn = get_db_connection()
    conn.execute(
        'INSERT INTO patients (name, tc, age, gender, phone, medications, complaints) VALUES (?, ?, ?, ?, ?, ?, ?)',
        (data['name'], data['tc'], data['age'], data['gender'], data['phone'], 
         json.dumps(data.get('medications', [])), data.get('complaints', ''))
    )
    conn.commit()
    conn.close()
    return jsonify({"message": "Hasta eklendi."}), 201

@app.route('/api/patients/<int:id>', methods=['PUT'])
def update_patient(id):
    data = request.get_json()
    conn = get_db_connection()
    conn.execute(
        'UPDATE patients SET name=?, tc=?, age=?, gender=?, phone=?, medications=?, complaints=? WHERE id=?',
        (data['name'], data['tc'], data['age'], data['gender'], data['phone'],
         json.dumps(data.get('medications', [])), data.get('complaints', ''), id)
    )
    conn.commit()
    conn.close()
    return jsonify({"message": "Hasta güncellendi."})

@app.route('/api/patients/<int:id>', methods=['DELETE'])
def delete_patient(id):
    conn = get_db_connection()
    conn.execute('DELETE FROM patients WHERE id=?', (id,))
    conn.commit()
    conn.close()
    return jsonify({"message": "Hasta silindi."})

@app.route('/api/patients/<int:patient_id>/files', methods=['GET'])
def get_patient_files(patient_id):
    conn = get_db_connection()
    files_cursor = conn.execute(
        'SELECT id, file_name, uploaded_at FROM ekg_files WHERE patient_id=? ORDER BY uploaded_at DESC',
        (patient_id,)
    ).fetchall()
    conn.close()
    return jsonify([dict(row) for row in files_cursor])

@app.route('/api/patients/<int:patient_id>/files', methods=['POST'])
def upload_file(patient_id):
    data = request.get_json()
    ekg_data_np = np.array(data['data'], dtype=np.float32)
    conn = get_db_connection()
    conn.execute(
        'INSERT INTO ekg_files (patient_id, file_name, uploaded_at, sampling_rate, ekg_data) VALUES (?, ?, ?, ?, ?)',
        (patient_id, data['name'], data['uploadedAt'], data.get('samplingRate', 1000), ekg_data_np.tobytes())
    )
    conn.commit()
    conn.close()
    return jsonify({"message": "Dosya yüklendi."}), 201

@app.route('/api/files/<int:file_id>', methods=['DELETE'])
def delete_file(file_id):
    """Belirli bir EKG dosyasını ID'sine göre siler."""
    try:
        conn = get_db_connection()
        # Önce dosyanın var olup olmadığını kontrol edebiliriz (isteğe bağlı)
        file_exists = conn.execute('SELECT id FROM ekg_files WHERE id = ?', (file_id,)).fetchone()
        if not file_exists:
            conn.close()
            return jsonify({"message": "Dosya bulunamadı."}), 404
        
        conn.execute('DELETE FROM ekg_files WHERE id = ?', (file_id,))
        conn.commit()
        conn.close()
        logger.info(f"Dosya ID: {file_id} başarıyla silindi.")
        return jsonify({"message": "Dosya başarıyla silindi."})
    except Exception as e:
        logger.error(f"Dosya silinirken hata oluştu (ID: {file_id}): {e}")
        # Veritabanı bağlantısı açık kaldıysa kapat
        if 'conn' in locals() and conn:
            conn.close()
        return jsonify({"message": "Dosya silinirken sunucuda bir hata oluştu."}), 500
    
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
    init_db()
    logger.info("Flask sunucusu http://127.0.0.1:5001 adresinde başlatılıyor...")
    app.run(debug=True, port=5001)