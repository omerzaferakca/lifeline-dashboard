import React, { useState, useEffect } from 'react';
import InfoBox from './InfoBox';
import ChartComponent from './ChartComponent';
import PatientModal from './PatientModal';
import { patientService } from '../firebase/patientService';
import { ekgService } from '../firebase/ekgService';
import { useAuth } from '../contexts/AuthContext';

function Dashboard() {
  const [anomalyStatus, setAnomalyStatus] = useState('NORMAL');
  const [heartRate, setHeartRate] = useState(72);
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [editingPatient, setEditingPatient] = useState(null);
  const [selectedPatientForEkg, setSelectedPatientForEkg] = useState(null);
  const [uploading, setUploading] = useState(false);
  
  const { currentUser } = useAuth();

  // Hastaları yükle
  useEffect(() => {
    if (currentUser) {
      loadPatients();
    }
  }, [currentUser]);

  const loadPatients = async () => {
    if (!currentUser) {
      console.log('loadPatients: currentUser yok');
      return;
    }
    
    console.log('loadPatients: currentUser.uid =', currentUser.uid);
    setLoading(true);
    setError('');
    
    try {
      console.log('patientService.getPatientsByDoctor çağrılıyor...');
      const patientList = await patientService.getPatientsByDoctor(currentUser.uid);
      console.log('Gelen hasta listesi:', patientList);
      setPatients(patientList);
    } catch (err) {
      console.error('Hastalar yüklenirken hata:', err);
      setError('Hastalar yüklenirken bir hata oluştu: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Hasta arama
  const handleSearch = async (term) => {
    setSearchTerm(term);
    
    if (!term.trim()) {
      loadPatients();
      return;
    }

    if (!currentUser) return;

    setLoading(true);
    try {
      const results = await patientService.searchPatients(term, currentUser.uid);
      setPatients(results);
    } catch (err) {
      console.error('Arama yapılırken hata:', err);
      setError('Arama yapılırken bir hata oluştu');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!selectedPatientForEkg) {
      alert('Lütfen önce EKG dosyası yüklenecek hastayı seçin');
      return;
    }

    setUploading(true);
    
    try {
      console.log('EKG dosyası yükleniyor:', file.name);
      
      // Dosya formatını kontrol et
      ekgService.validateEkgFile(file);
      
      // EKG verisini parse et
      const parsedData = await ekgService.parseEkgData(file);
      console.log('EKG verisi parse edildi:', parsedData);
      
      // Basit analiz yap (kalp atış hızı hesapla)
      const analysisResult = performBasicEkgAnalysis(parsedData.data);
      
      // Firebase Storage'a yükle
      const result = await ekgService.uploadEkgFile(
        file, 
        selectedPatientForEkg.id, 
        currentUser.uid,
        analysisResult
      );
      
      console.log('EKG dosyası başarıyla yüklendi:', result);
      
      // Başarı mesajı
      alert(`EKG dosyası başarıyla yüklendi!\nKalp atış hızı: ${analysisResult.heartRate} BPM\nDurum: ${analysisResult.anomalyDetected ? 'Anormal' : 'Normal'}`);
      
      // Form'u temizle
      event.target.value = '';
      setSelectedPatientForEkg(null);
      
      // İstatistikleri güncelle
      setHeartRate(analysisResult.heartRate);
      setAnomalyStatus(analysisResult.anomalyDetected ? 'ANORMAL' : 'NORMAL');
      
    } catch (error) {
      console.error('EKG dosyası yüklenirken hata:', error);
      alert('EKG dosyası yüklenirken hata oluştu: ' + error.message);
    } finally {
      setUploading(false);
    }
  };

  // Basit EKG analizi fonksiyonu
  const performBasicEkgAnalysis = (ekgData) => {
    if (!ekgData || ekgData.length === 0) {
      return {
        heartRate: 0,
        anomalyDetected: false,
        result: 'Veri yetersiz',
        confidenceScore: 0
      };
    }

    // Basit kalp atış hızı hesaplama
    // Bu gerçek bir algoritma değil, demo amaçlı
    const values = ekgData.map(point => point.value);
    const avgValue = values.reduce((sum, val) => sum + val, 0) / values.length;
    const peakThreshold = avgValue + (Math.max(...values) - avgValue) * 0.5;
    
    // Peak detection (basit)
    let peaks = 0;
    for (let i = 1; i < values.length - 1; i++) {
      if (values[i] > peakThreshold && 
          values[i] > values[i-1] && 
          values[i] > values[i+1]) {
        peaks++;
      }
    }
    
    // Kalp atış hızı tahmini (örnek 10 saniye veri varsayımı)
    const estimatedDuration = ekgData.length / 250; // 250 Hz varsayımı
    const heartRate = Math.round((peaks / estimatedDuration) * 60);
    
    // Anormallik kontrolü (basit)
    const anomalyDetected = heartRate < 60 || heartRate > 100;
    
    return {
      heartRate: Math.max(0, Math.min(200, heartRate)), // 0-200 arasında sınırla
      anomalyDetected: anomalyDetected,
      result: anomalyDetected ? 'Anormal ritim tespit edildi' : 'Normal ritim',
      confidenceScore: 0.75,
      peaksDetected: peaks,
      estimatedDuration: estimatedDuration
    };
  };

  const openModal = (patient = null) => {
    setEditingPatient(patient);
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setEditingPatient(null);
    setIsModalOpen(false);
  };

  const handlePatientSave = async () => {
    try {
      console.log('handlePatientSave başladı...');
      await loadPatients(); // Hasta listesini yenile
      console.log('loadPatients tamamlandı');
      closeModal();
    } catch (error) {
      console.error('handlePatientSave hatası:', error);
    }
  };

  const formatDate = (timestamp) => {
    if (!timestamp) return 'Bilinmiyor';
    const date = timestamp.toDate ? timestamp.toDate() : new Date(timestamp);
    return date.toLocaleDateString('tr-TR');
  };

  return (
    <main className="dashboard">
      {/* İstatistik Kartları */}
      <div className="info-cards">
        <InfoBox title="Toplam Hasta" value={patients.length} />
        <InfoBox title="Kalp Atış Hızı" value={heartRate} unit="BPM" />
        <InfoBox title="Durum" value={anomalyStatus} />
      </div>

      {/* Hasta Yönetimi Bölümü */}
      <div className="patient-management">
        <div className="section-header">
          <h2>Hasta Yönetimi</h2>
          <button 
            onClick={() => openModal()} 
            className="btn btn-primary"
          >
            Yeni Hasta Ekle
          </button>
        </div>

        {/* Arama Çubuğu */}
        <div className="search-section">
          <input
            type="text"
            placeholder="Hasta ara (ad, soyad, TC)..."
            value={searchTerm}
            onChange={(e) => handleSearch(e.target.value)}
            className="search-input"
          />
        </div>

        {/* Hasta Listesi */}
        <div className="patients-list">
          {loading ? (
            <div className="loading">Hastalar yükleniyor...</div>
          ) : error ? (
            <div className="error-message">{error}</div>
          ) : patients.length === 0 ? (
            <div className="no-patients">
              {searchTerm ? 'Arama kriterine uygun hasta bulunamadı.' : 'Henüz hasta eklenmemiş.'}
            </div>
          ) : (
            <div className="patients-grid">
              {patients.map((patient) => (
                <div key={patient.id} className="patient-card">
                  <div className="patient-info">
                    <h3>{patient.firstName} {patient.lastName}</h3>
                    <p className="patient-details">
                      <span>TC: {patient.tc}</span>
                      <span>Yaş: {patient.age}</span>
                      <span>Cinsiyet: {patient.gender}</span>
                    </p>
                    <p className="patient-contact">
                      <span>📞 {patient.phone}</span>
                      {patient.email && <span>📧 {patient.email}</span>}
                    </p>
                    <p className="patient-date">
                      Kayıt: {formatDate(patient.createdAt)}
                    </p>
                  </div>
                  <div className="patient-actions">
                    <button 
                      onClick={() => openModal(patient)}
                      className="btn btn-secondary btn-sm"
                    >
                      Düzenle
                    </button>
                    <button 
                      onClick={() => {
                        setSelectedPatientForEkg(patient);
                        document.getElementById('ecg-upload').click();
                      }}
                      className="btn btn-primary btn-sm"
                      disabled={uploading}
                    >
                      {uploading && selectedPatientForEkg?.id === patient.id ? 'Yükleniyor...' : 'EKG Yükle'}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* EKG Analiz Bölümü */}
      <div className="analysis-section">
        <h2>EKG Analizi</h2>
        <ChartComponent />
        
        <div className="upload-section">
          {selectedPatientForEkg ? (
            <div className="selected-patient-info">
              <h4>Seçili Hasta: {selectedPatientForEkg.firstName} {selectedPatientForEkg.lastName}</h4>
              <p>TC: {selectedPatientForEkg.tc}</p>
              <button 
                onClick={() => setSelectedPatientForEkg(null)}
                className="btn btn-secondary btn-sm"
              >
                Seçimi İptal Et
              </button>
            </div>
          ) : (
            <p>EKG dosyası yüklemek için hasta kartından "EKG Yükle" butonuna tıklayın</p>
          )}
          
          <label htmlFor="ecg-upload" className="upload-button">
            {uploading ? 'Yükleniyor...' : 'EKG Verisi Yükle (.csv, .txt)'}
          </label>
          <input 
            id="ecg-upload" 
            type="file" 
            accept=".csv,.txt"
            onChange={handleFileUpload} 
            style={{ display: 'none' }}
            disabled={uploading}
          />
        </div>
      </div>

      {/* Hasta Modal */}
      <PatientModal
        isOpen={isModalOpen}
        onClose={closeModal}
        onSave={handlePatientSave}
        patient={editingPatient}
      />
    </main>
  );
}

export default Dashboard;