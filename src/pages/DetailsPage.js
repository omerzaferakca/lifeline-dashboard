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

// Tarih formatlama yardımcısı
const formatDate = (isoString) => {
    if (!isoString) return '';
    return new Date(isoString).toLocaleDateString('tr-TR', {
        day: '2-digit', month: '2-digit', year: 'numeric'
    });
};

function DetailsPage({ patient }) {
  const [files, setFiles] = useState([]);
  const [activeFile, setActiveFile] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const chartRef = useRef(null);

  const fetchFiles = async (patientId) => {
    if (!patientId) return;
    try {
      const response = await fetch(`${API_URL}/patients/${patientId}/files`);
      const data = await response.json();
      setFiles(Array.isArray(data) ? data : []);
      if (data.length > 0 && !activeFile) {
        handleFileSelect(data[0]);
      }
    } catch (error) { console.error("Dosyalar getirilirken hata:", error); }
  };

  useEffect(() => {
    if (patient) {
      setActiveFile(null);
      setAnalysisResult(null);
      fetchFiles(patient.id);
    }
  }, [patient]);

  const handleFileSelect = async (file) => {
    if (!file || isLoading) return;
    setActiveFile(file);
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
    } catch (error) { console.error("Analiz hatası:", error); } 
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
      const payload = { name: file.name, uploadedAt: new Date().toISOString(), data: ekgData, samplingRate: 1000 };
      try {
        const res = await fetch(`${API_URL}/patients/${patient.id}/files`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        if (res.ok) { 
          setActiveFile(null);
          await fetchFiles(patient.id); 
        } else { 
            const errorData = await res.json();
            alert(`Dosya yüklenemedi: ${errorData.error || 'Bilinmeyen sunucu hatası.'}`); 
        }
      } catch (error) { console.error("Dosya yükleme hatası:", error); }
    };
    reader.readAsText(file);
    event.target.value = null;
  };
  
  const handleFileDelete = async (fileIdToDelete, event) => {
    event.stopPropagation();
    if (window.confirm("Bu EKG kaydını kalıcı olarak silmek istediğinizden emin misiniz?")) {
      setIsLoading(true);
      try {
        const response = await fetch(`${API_URL}/files/${fileIdToDelete}`, { method: 'DELETE' });
        if (response.ok) {
          if (activeFile?.id === fileIdToDelete) {
            setActiveFile(null);
            setAnalysisResult(null);
          }
          await fetchFiles(patient.id);
        } else { alert("Dosya silinirken bir hata oluştu."); }
      } catch (error) { console.error("Dosya silme hatası:", error); } 
      finally { setIsLoading(false); }
    }
  };

  const getAnnotations = () => {
    if (!analysisResult?.beat_predictions || !analysisResult?.r_peaks) return {};
    const annotations = {};
    const beatWidth = (analysisResult.display_signal.length / analysisResult.r_peaks.length) * 0.7;
    analysisResult.beat_predictions
      .filter(beat => !beat.class_name.startsWith("Normal"))
      .forEach((beat, index) => {
        const rPeakIndex = analysisResult.r_peaks[beat.beat_id];
        if (rPeakIndex === undefined) return;
        const char_code = beat.class_name.charAt(beat.class_name.length - 2);
        const color = ANOMALY_COLORS[char_code] || ANOMALY_COLORS['Q'];
        annotations[`box-${index}`] = {
          type: 'box', xMin: Math.max(0, rPeakIndex - beatWidth / 2),
          xMax: rPeakIndex + beatWidth / 2, backgroundColor: color.background,
          borderColor: color.border, borderWidth: 1,
          label: {
             content: char_code, display: true, color: '#333',
             font: { weight: 'bold', size: 10 }, position: 'start', yAdjust: -5,
          }
        };
      });
    return annotations;
  };

  const chartOptions = {
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: {
      legend: { display: false },
      title: { display: true, text: `EKG Sinyali: ${activeFile?.file_name || 'Dosya Seçilmedi'}` },
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
      </div>
    );
  }

  // YENİ BACKEND VERİ YAPISINA GÜNCELLENDİ
  const clinicalMetrics = analysisResult?.clinical_metrics;
  const aiSummary = analysisResult?.ai_summary;
  const hrvMetrics = clinicalMetrics?.heart_rate_variability;
  const patientMeds = patient.medications ? (typeof patient.medications === 'string' ? JSON.parse(patient.medications) : patient.medications) : [];

  return (
    <div className="page">
      <div className="page-header"><h1>{patient.name}</h1><p><strong>TC:</strong> {patient.tc} | <strong>Yaş:</strong> {patient.age} | <strong>Cinsiyet:</strong> {patient.gender}</p></div>
      <div className="patient-card" style={{marginBottom: '1.5rem'}}>
        <h3>EKG Kayıtları ({files.length})</h3>
        <ul className="ekg-file-list">
          {files.map(file => (
            <li key={file.id} className={file.id === activeFile?.id ? 'active-file' : ''} onClick={() => handleFileSelect(file)}>
              <div style={{ flexGrow: 1, cursor: 'pointer' }}>
                {file.file_name} 
                <span>{formatDate(file.uploaded_at)}</span>
              </div>
              <button onClick={(e) => handleFileDelete(file.id, e)} className="btn btn-danger btn-sm" style={{ marginLeft: '1rem', padding: '0.2rem 0.5rem' }} title="Bu kaydı sil">Sil</button>
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
          <h3>Klinik Analiz</h3>
          {isLoading ? <p>Hesaplanıyor...</p> : analysisResult ? (
            <div>
              <p><strong>Kalp Hızı:</strong> {clinicalMetrics?.heart_rate_bpm?.toFixed(0) || '--'} bpm</p>
              <p><strong>QRS Süresi:</strong> {clinicalMetrics?.qrs_duration_ms?.toFixed(0) || '--'} ms</p>
              <p><strong>PR Aralığı:</strong> {clinicalMetrics?.pr_interval_ms?.toFixed(0) || '--'} ms</p>
              <p><strong>QT Aralığı:</strong> {clinicalMetrics?.qt_interval_ms?.toFixed(0) || '--'} ms</p>
              <p><strong>HRV (RMSSD):</strong> {hrvMetrics?.rmssd_ms?.toFixed(1) || '--'} ms</p>
            </div>
          ) : <p>Analiz için bir kayıt seçin.</p>}
        </div>
        <div className="patient-card">
          <h3>Yapay Zeka Özeti</h3>
          {isLoading ? <p>Hesaplanıyor...</p> : analysisResult ? (
            <div>
              <p><strong>Risk Seviyesi:</strong> <span className={`status-${aiSummary?.risk_level?.toLowerCase()}`}>{aiSummary?.risk_level || '--'}</span></p>
              <p><strong>Bulgular:</strong></p>
              <ul>{aiSummary?.findings?.map((f, i) => <li key={i}>{f}</li>) || <li>Bulgu yok.</li>}</ul>
            </div>
          ) : <p>Analiz bekleniyor.</p>}
        </div>
        <div className="patient-card">
          <h3>Klinik Tavsiyeler</h3>
           {isLoading ? <p>Oluşturuluyor...</p> : analysisResult ? (
            <div>
              <p><strong>Öneriler:</strong></p>
              <ul>{aiSummary?.recommendations?.map((r, i) => <li key={i}>{r}</li>) || <li>Spesifik öneri yok.</li>}</ul>
            </div>
          ) : <p>Analiz bekleniyor.</p>}
        </div>
        <div className="patient-card">
          <h3>Hasta Bilgileri</h3>
          <p><strong>Şikayet / Notlar:</strong> {patient.complaints || "Girilmemiş"}</p>
          <p><strong>Kullanılan İlaçlar:</strong></p>
          {patientMeds.length > 0 ? <ul>{patientMeds.map((med, i) => <li key={i}>{med}</li>)}</ul> : <p>İlaç kaydı yok.</p>}
        </div>
      </div>
    </div>
  );
}

export default DetailsPage;