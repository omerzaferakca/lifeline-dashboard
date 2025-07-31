import React, { useState, useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, Title, Tooltip, Legend,
} from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';
import zoomPlugin from 'chartjs-plugin-zoom';

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  Title, Tooltip, Legend, annotationPlugin, zoomPlugin
);

const API_URL = 'http://127.0.0.1:5001/api';

const ANOMALY_COLORS = {
    'V': { background: 'rgba(255, 99, 132, 0.3)', border: 'rgba(255, 99, 132, 0.5)' },
    'S': { background: 'rgba(255, 159, 64, 0.3)', border: 'rgba(255, 159, 64, 0.5)' },
    'F': { background: 'rgba(153, 102, 255, 0.3)', border: 'rgba(153, 102, 255, 0.5)' },
    'Q': { background: 'rgba(75, 192, 192, 0.3)', border: 'rgba(75, 192, 192, 0.5)' },
};

function DetailsPage({ patient }) {
  const [files, setFiles] = useState([]);
  const [activeFileId, setActiveFileId] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const chartRef = useRef(null);

  const fetchFiles = async (patientId) => {
    if (!patientId) return;
    try {
      const response = await fetch(`${API_URL}/patients/${patientId}/files`);
      const data = await response.json();
      setFiles(data);
      // Eğer hasta yeni seçilmişse veya dosya silindikten sonra hala dosya varsa,
      // ve aktif bir dosya seçili değilse, ilkini seç.
      if (data.length > 0 && activeFileId === null) {
        handleFileSelect(data[0]);
      } else if (data.length === 0) {
        // Hiç dosya kalmadıysa her şeyi temizle
        setActiveFileId(null);
        setAnalysisResult(null);
      }
    } catch (error) { 
      console.error("Dosyalar getirilirken hata:", error); 
    }
  };

  useEffect(() => {
    if (patient) {
      setActiveFileId(null); // Hasta değiştiğinde aktif dosyayı sıfırla
      setAnalysisResult(null);
      fetchFiles(patient.id);
    }
  }, [patient]);

  const handleFileSelect = async (file) => {
    if (!file || isLoading) return;
    setActiveFileId(file.id);
    setIsLoading(true);
    setAnalysisResult(null);
    try {
      const response = await fetch(`${API_URL}/analyze/${file.id}`);
      const result = await response.json();
      if (result.success) {
        setAnalysisResult(result);
      } else {
        alert(`Analiz hatası: ${result.error}`);
      }
    } catch (error) { 
      console.error("Analiz hatası:", error); 
    } 
    finally { setIsLoading(false); }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file || !patient) return;
    const reader = new FileReader();
    reader.onload = async (e) => {
      const content = e.target.result;
      const ekgData = content.split(/[\n,;]/).map(val => parseFloat(val.trim())).filter(v => !isNaN(v));
      if (ekgData.length < 100) { alert("Geçersiz EKG verisi."); return; }
      const payload = { name: file.name, uploadedAt: new Date().toISOString().split('T')[0], data: ekgData, samplingRate: 1000 };
      try {
        const res = await fetch(`${API_URL}/patients/${patient.id}/files`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        if (res.ok) { 
          // Yüklemeden sonra dosya listesini yenile
          await fetchFiles(patient.id);
        } else { 
          alert("Dosya yüklenemedi."); 
        }
      } catch (error) { console.error("Dosya yükleme hatası:", error); }
    };
    reader.readAsText(file);
    event.target.value = null;
  };
  
  // --- YENİ DOSYA SİLME FONKSİYONU ---
  const handleFileDelete = async (fileIdToDelete, event) => {
    event.stopPropagation(); // Butona tıklamanın, <li>'nin tıklama olayını tetiklemesini engelle
    
    if (window.confirm("Bu EKG kaydını kalıcı olarak silmek istediğinizden emin misiniz?")) {
      setIsLoading(true);
      try {
        const response = await fetch(`${API_URL}/files/${fileIdToDelete}`, {
          method: 'DELETE',
        });

        if (response.ok) {
          // Eğer silinen dosya o an aktif olan dosyaysa, analiz sonucunu temizle
          if (activeFileId === fileIdToDelete) {
            setActiveFileId(null);
            setAnalysisResult(null);
          }
          // Silme işlemi başarılıysa dosya listesini yenile
          await fetchFiles(patient.id);
        } else {
          alert("Dosya silinirken bir hata oluştu.");
        }
      } catch (error) {
        console.error("Dosya silme hatası:", error);
        alert("Dosya silinirken sunucuya bağlanılamadı.");
      } finally {
        setIsLoading(false);
      }
    }
  };

  const getAnnotations = () => {
    if (!analysisResult?.predictions || !analysisResult?.r_peaks) return {};
    const annotations = {};
    const beatWidth = (analysisResult.display_signal.length / analysisResult.r_peaks.length) * 0.7;
    analysisResult.predictions
      .filter(p => p.class_name !== 'N')
      .forEach((beat, index) => {
        const rPeakIndex = analysisResult.r_peaks[beat.beat_id];
        if (rPeakIndex === undefined) return;
        const color = ANOMALY_COLORS[beat.class_name] || ANOMALY_COLORS['Q'];
        annotations[`box-${index}`] = {
          type: 'box', xMin: Math.max(0, rPeakIndex - beatWidth / 2),
          xMax: rPeakIndex + beatWidth / 2, backgroundColor: color.background,
          borderColor: color.border, borderWidth: 1,
          label: {
             content: beat.class_name, display: true, color: '#333',
             font: { weight: 'bold', size: 10 }, position: 'start', yAdjust: -5,
          }
        };
      });
    return annotations;
  };

  const chartOptions = {
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: {
      legend: { display: false }, title: { display: true, text: `EKG Sinyali: ${files.find(f => f.id === activeFileId)?.file_name || ''}` },
      zoom: { pan: { enabled: true, mode: 'x' }, zoom: { wheel: { enabled: true }, mode: 'x' } },
      annotation: { annotations: getAnnotations() },
    },
    scales: { x: { title: { display: true, text: 'Örneklem' } }, y: { title: { display: true, text: 'Genlik (mV)' } } }
  };
  
  const chartData = {
    labels: analysisResult?.display_signal?.map((_, i) => i) || [],
    datasets: [{
      label: 'Temizlenmiş EKG', data: analysisResult?.display_signal || [],
      borderColor: '#007bff', borderWidth: 1, pointRadius: 0,
    }],
  };
  
  if (!patient) {
    return (
      <div className="page" style={{ textAlign: 'center', padding: '4rem' }}>
        <h2>Lütfen Bir Hasta Seçin</h2>
        <p>Hasta detaylarını ve EKG kayıtlarını görmek için anasayfadan bir hasta seçin.</p>
      </div>
    );
  }

  const clinicalFeatures = analysisResult?.clinical_features;
  const aiSummary = analysisResult?.ai_summary;
  const patientMeds = patient.medications ? (typeof patient.medications === 'string' ? JSON.parse(patient.medications) : patient.medications) : [];

  return (
    <div className="page">
      <div className="page-header"><h1>{patient.name}</h1><p><strong>TC:</strong> {patient.tc} | <strong>Yaş:</strong> {patient.age} | <strong>Cinsiyet:</strong> {patient.gender}</p></div>
      <div className="patient-card" style={{marginBottom: '1.5rem'}}>
        <h3>EKG Kayıtları ({files.length})</h3>
        <ul className="ekg-file-list">
          {files.map(file => (
            <li key={file.id} className={file.id === activeFileId ? 'active-file' : ''} onClick={() => handleFileSelect(file)}>
              <div style={{ flexGrow: 1, cursor: 'pointer' }}>
                {file.file_name} <span>{file.uploaded_at}</span>
              </div>
              <button 
                onClick={(e) => handleFileDelete(file.id, e)}
                className="btn btn-danger btn-sm"
                style={{ marginLeft: '1rem', padding: '0.2rem 0.5rem' }}
                title="Bu kaydı sil"
              >
                Sil
              </button>
            </li>
          ))}
          {files.length === 0 && !isLoading && <p>Bu hasta için EKG kaydı bulunamadı.</p>}
        </ul>
        <label htmlFor="patient-file-upload" className="btn btn-secondary" style={{width: '100%', textAlign: 'center', marginTop: '1rem'}}>+ Yeni Kayıt Yükle</label>
        <input id="patient-file-upload" type="file" style={{display: 'none'}} onChange={handleFileUpload} accept=".csv,.txt,.dat"/>
      </div>
      <div className="chart-container-full" style={{ marginBottom: '1rem', position: 'relative' }}>
        {isLoading && <div className="loading-overlay"><span>Analiz yapılıyor...</span></div>}
        <div style={{height: '350px'}}><Line ref={chartRef} options={chartOptions} data={chartData} /></div>
      </div>
      <button onClick={() => chartRef.current?.resetZoom()} className="btn btn-primary btn-sm" style={{marginBottom: '1.5rem'}}>Yakınlaştırmayı Sıfırla</button>
      <div className="analysis-grid-bottom">
        <div className="patient-card">
          <h3>Analiz Sonuçları</h3>
          {isLoading ? <p>Hesaplanıyor...</p> : analysisResult ? (
            <div>
              <p><strong>Kalp Hızı:</strong> {clinicalFeatures?.heart_rate?.toFixed(0) || '--'} bpm</p>
              <p><strong>QRS Süresi:</strong> {clinicalFeatures?.qrs_duration?.toFixed(0) || '--'} ms</p>
              <p><strong>PR Aralığı:</strong> {clinicalFeatures?.pr_interval?.toFixed(0) || '--'} ms</p>
              <p><strong>QT Aralığı:</strong> {clinicalFeatures?.qt_interval?.toFixed(0) || '--'} ms</p>
            </div>
          ) : <p>Analiz için bir kayıt seçin.</p>}
        </div>
        <div className="patient-card">
          <h3>Yapay Zeka Bulguları</h3>
          {isLoading ? <p>Hesaplanıyor...</p> : analysisResult ? (
            <div>
              <p><strong>Aritmi Riski:</strong> {aiSummary?.arrhythmia_level || '--'}</p>
              <p><strong>Potansiyel Bulgular:</strong></p>
              <ul>{aiSummary?.risk_findings?.map((f, i) => <li key={i}>{f}</li>) || <li>Bulgu yok.</li>}</ul>
            </div>
          ) : <p>Analiz bekleniyor.</p>}
        </div>
        <div className="patient-card">
          <h3>Hasta Notları ve İlaçlar</h3>
          <p><strong>Şikayet / Notlar:</strong> {patient.complaints || "Girilmemiş"}</p>
          <p><strong>Kullanılan İlaçlar:</strong></p>
          {patientMeds.length > 0 ? <ul>{patientMeds.map((med, i) => <li key={i}>{med}</li>)}</ul> : <p>İlaç kaydı yok.</p>}
        </div>
      </div>
    </div>
  );
}

export default DetailsPage;