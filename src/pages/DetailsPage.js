import React, { useState, useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, Title, Tooltip, Legend,
} from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';
import zoomPlugin from 'chartjs-plugin-zoom'; // Zoom eklentisini import et

// Chart.js ve gerekli tüm eklentileri kaydet
ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  Title, Tooltip, Legend, annotationPlugin, zoomPlugin
);

// Backend API çağrı fonksiyonu (değişiklik yok)
const processEcgDataOnBackend = async (rawEcgData, samplingRate = 1000) => {
  const response = await fetch('http://127.0.0.1:5001/process-ecg', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ecg_signal: rawEcgData, sampling_rate: samplingRate }),
  });
  if (!response.ok) {
    const errorBody = await response.json();
    throw new Error(errorBody.error || `HTTP hatası: ${response.status}`);
  }
  return await response.json();
};

// Anomali tiplerine göre renklendirme haritası
const ANOMALY_COLORS = {
  'V': 'rgba(255, 99, 132, 0.3)',  // Kırmızı: Ventricular
  'S': 'rgba(255, 159, 64, 0.3)', // Turuncu: Supraventricular
  'F': 'rgba(153, 102, 255, 0.3)',// Mor: Fusion
  'Q': 'rgba(75, 192, 192, 0.3)', // Yeşil: Unknown/Paced
};

function DetailsPage({ patients, patientId, onAddFile }) {
  const [activeFileId, setActiveFileId] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const chartRef = useRef(null); // Grafik referansı için useRef hook'u

  const patient = patientId ? patients.find(p => p.id === patientId) : null;
  const patientFiles = patient ? patient.ekgFiles || [] : [];
  const activeFile = activeFileId ? patientFiles.find(f => f.id === activeFileId) : null;

  useEffect(() => {
    // Aktif dosya değiştiğinde analizi tetikle
    if (activeFile?.data) {
      const runAnalysis = async () => {
        setIsLoading(true);
        setAnalysisResult(null);
        try {
          const result = await processEcgDataOnBackend(activeFile.data, activeFile.samplingRate);
          if (result.success) {
            setAnalysisResult(result);
          } else {
            alert(`Analiz sırasında bir hata oluştu: ${result.error}`);
          }
        } catch (error) {
          alert(`Backend ile iletişim kurulamadı: ${error.message}`);
        } finally {
          setIsLoading(false);
        }
      };
      runAnalysis();
    }
  }, [activeFile]);

  // Yeni dosya yüklendiğinde çalışacak fonksiyon
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target.result;
      const ekgData = content.split(/[\n,;]/).map(val => parseFloat(val.trim())).filter(v => !isNaN(v));
      if (ekgData.length > 100) { // Minimum veri uzunluğu
        const newFile = {
          id: `file_${Date.now()}`, name: file.name, uploadedAt: new Date().toISOString().split('T')[0],
          data: ekgData, samplingRate: 1000 // Varsayılan, gerekirse değiştirilebilir
        };
        onAddFile(patient.id, newFile);
        setActiveFileId(newFile.id);
      } else {
        alert("Dosya geçerli EKG verisi içermiyor.");
      }
    };
    reader.readAsText(file);
    event.target.value = null;
  };
  
  // Çok renkli ve dinamik annotation'ları hazırla
  const getAnnotations = () => {
    if (!analysisResult?.predictions) return {};
    const annotations = {};
    const beatWidth = 187; // 187 örneklem genişliği (yaklaşık 1.5 saniye @ 125Hz)
    
    analysisResult.predictions
      .filter(p => p.class_name !== 'N') // Sadece anormalleri göster
      .forEach((beat, index) => {
        const rPeakIndex = analysisResult.r_peaks[beat.beat_id];
        if (rPeakIndex === undefined) return;
        
        annotations[`box-${index}`] = {
          type: 'box',
          xMin: Math.max(0, rPeakIndex - beatWidth / 2),
          xMax: rPeakIndex + beatWidth / 2,
          backgroundColor: ANOMALY_COLORS[beat.class_name] || 'rgba(201, 203, 207, 0.3)', // Haritadan renk seç
          borderColor: 'transparent',
          label: { // Kutunun üzerine sınıfını yaz
             content: beat.class_name,
             display: true,
             color: '#333',
             font: { weight: 'bold' },
             position: 'start',
          }
        };
      });
    return annotations;
  };
  
  // Grafiği orijinal haline döndüren fonksiyon
  const handleResetZoom = () => {
    chartRef.current?.resetZoom();
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false, // Performansı artırmak için animasyonları kapat
    plugins: {
      legend: { display: false },
      title: { display: true, text: `EKG Sinyali: ${activeFile?.name || ''}` },
      tooltip: { enabled: false }, // Performans için kapatılabilir
      annotation: { annotations: getAnnotations() },
      zoom: { // Yakınlaştırma ve kaydırma ayarları
        pan: { enabled: true, mode: 'x', threshold: 5 },
        zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' },
      },
    },
    scales: { 
      x: { title: { display: true, text: 'Örneklem (125Hz)' } }, 
      y: { title: { display: true, text: 'Genlik (Normalize)' } } 
    }
  };

  const chartData = {
    labels: analysisResult?.processed_signal?.map((_, i) => i) || [],
    datasets: [{
      label: 'İşlenmiş EKG',
      data: analysisResult?.processed_signal || [],
      borderColor: '#007bff',
      borderWidth: 1,
      pointRadius: 0,
      tension: 0.1,
    }],
  };
  
  const clinicalFeatures = analysisResult?.clinical_features;
  const aiSummary = analysisResult?.ai_summary;

  return (
    <div className="page">
      <div className="page-header">
        <h1>{patient?.name || 'Hasta Seçilmedi'}</h1>
        <p>
          ID: {patient?.id || '--'} | Yaş: {patient?.age || '--'} | Cinsiyet: {patient?.gender || '--'}
        </p>
      </div>

      <div className="patient-card" style={{marginBottom: '1.5rem'}}>
        <h3>EKG Kayıtları ({patientFiles.length})</h3>
        <ul className="ekg-file-list">
          {patientFiles.map(file => (
            <li key={file.id} className={file.id === activeFileId ? 'active-file' : ''} onClick={() => setActiveFileId(file.id)}>
              {file.name} <span>{file.uploadedAt}</span>
            </li>
          ))}
        </ul>
        <label htmlFor="patient-file-upload" className="btn btn-secondary" style={{width: '100%', textAlign: 'center', boxSizing: 'border-box', marginTop: '1rem'}}>+ Yeni Kayıt Yükle</label>
        <input id="patient-file-upload" type="file" style={{display: 'none'}} onChange={handleFileUpload} accept=".csv,.txt,.dat"/>
      </div>
      
      <div className="chart-container-full" style={{ marginBottom: '1rem', position: 'relative' }}>
        {isLoading && <div className="loading-overlay"><span>Detaylı EKG analizi yapılıyor... Lütfen bekleyin.</span></div>}
        <div style={{height: '350px'}}>
           <Line ref={chartRef} options={chartOptions} data={chartData} />
        </div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
        <button onClick={handleResetZoom} className="btn btn-primary btn-sm">Yakınlaştırmayı Sıfırla</button>
        <div style={{ fontSize: '0.9rem', color: '#6c757d' }}>
          * Yakınlaştırmak için fare tekerleğini, kaydırmak için sürükleyin.
        </div>
      </div>


      <div className="analysis-grid-bottom">
        <div className="patient-card">
          <h3>Analiz Sonuçları</h3>
          {clinicalFeatures ? (
            <div>
              <p><strong>Kalp Hızı:</strong> {clinicalFeatures.heart_rate?.toFixed(0) || '--'} bpm</p>
              <p><strong>QRS Süresi:</strong> {clinicalFeatures.qrs_duration?.toFixed(0) || '--'} ms</p>
              <p><strong>PR Aralığı:</strong> {clinicalFeatures.pr_interval?.toFixed(0) || '--'} ms</p>
              <p><strong>QT Aralığı:</strong> {clinicalFeatures.qt_interval?.toFixed(0) || '--'} ms</p>
            </div>
          ) : !isLoading && <p>Analiz sonucu için bir kayıt seçin veya yükleyin.</p>}
        </div>

        <div className="patient-card">
          <h3>Yapay Zeka Bulguları</h3>
          {aiSummary ? (
            <div>
              <p><strong>Aritmi Riski:</strong> <span className={`status-${aiSummary.arrhythmia_level?.toLowerCase()}`}>{aiSummary.arrhythmia_level || 'Bilinmiyor'}</span></p>
              <p><strong>Potansiyel Bulgular:</strong></p>
              <ul>
                {aiSummary.risk_findings?.map((finding, i) => <li key={i}>{finding}</li>)}
              </ul>
            </div>
          ) : !isLoading && <p>Analiz sonucu bekleniyor.</p>}
        </div>

        <div className="patient-card">
          <h3>Kullanılan İlaçlar</h3>
          {patient?.medications && patient.medications.length > 0 ? (
            <ul>{patient.medications.map((med, i) => <li key={i}>{med}</li>)}</ul>
          ) : <p>Hasta kayıtlı bir ilaç kullanmıyor.</p>}
        </div>
      </div>
    </div>
  );
}

export default DetailsPage;