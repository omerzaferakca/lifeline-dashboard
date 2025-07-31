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
  'V': 'rgba(255, 99, 132, 0.3)',
  'S': 'rgba(255, 159, 64, 0.3)',
  'F': 'rgba(153, 102, 255, 0.3)',
  'Q': 'rgba(75, 192, 192, 0.3)',
};

function DetailsPage({ patient }) {
  const [files, setFiles] = useState([]);
  const [activeFile, setActiveFile] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const chartRef = useRef(null);

  const fetchFiles = async () => {
    if (!patient) return;
    try {
      const response = await fetch(`${API_URL}/patients/${patient.id}/files`);
      const data = await response.json();
      setFiles(data);
      // Eğer dosya varsa ve henüz aktif dosya seçilmemişse, ilkini seç
      if (data.length > 0 && !activeFile) {
        handleFileSelect(data[0]);
      }
    } catch (error) {
      console.error("Dosyalar getirilirken hata:", error);
    }
  };

  useEffect(() => {
    // Hasta değiştiğinde dosyaları yeniden getir ve state'i sıfırla
    setFiles([]);
    setActiveFile(null);
    setAnalysisResult(null);
    fetchFiles();
  }, [patient]);

  const handleFileSelect = async (file) => {
    if (!file) return;
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
    } catch (error) {
      console.error("Analiz hatası:", error);
      alert("Analiz sırasında sunucu ile iletişim kurulamadı.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file || !patient) return;

    const reader = new FileReader();
    reader.onload = async (e) => {
      const content = e.target.result;
      const ekgData = content.split(/[\n,;]/).map(val => parseFloat(val.trim())).filter(v => !isNaN(v));
      
      if (ekgData.length < 100) {
        alert("Dosya geçerli EKG verisi içermiyor veya çok kısa.");
        return;
      }

      const newFilePayload = {
        name: file.name,
        uploadedAt: new Date().toISOString().split('T')[0],
        data: ekgData,
        samplingRate: 1000 // Varsayılan, gerekirse bu da kullanıcıdan alınabilir
      };

      try {
        const response = await fetch(`${API_URL}/patients/${patient.id}/files`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(newFilePayload),
        });
        if (response.ok) {
          await fetchFiles(); // Dosya listesini yenile
        } else {
          alert("Dosya yüklenirken bir hata oluştu.");
        }
      } catch (error) {
        console.error("Dosya yükleme hatası:", error);
        alert("Dosya yüklenirken sunucuya bağlanılamadı.");
      }
    };
    reader.readAsText(file);
    event.target.value = null;
  };
  
  const getAnnotations = () => {
    if (!analysisResult?.predictions) return {};
    const annotations = {};
    const beatWidth = 187; // 1.5 saniye @ 125Hz
    analysisResult.predictions
      .filter(p => p.class_name !== 'N')
      .forEach((beat, index) => {
        const rPeakIndex = analysisResult.r_peaks[beat.beat_id];
        if (rPeakIndex === undefined) return;
        // Önemli Düzeltme: R-peak'ler normalize sinyalde bulundu, ama biz display_signal'i gösteriyoruz.
        // İkisinin de uzunluğu aynı olmalı (downsample sonrası). Bu yüzden indeksler eşleşir.
        annotations[`box-${index}`] = {
          type: 'box',
          xMin: Math.max(0, rPeakIndex - beatWidth / 2),
          xMax: rPeakIndex + beatWidth / 2,
          backgroundColor: ANOMALY_COLORS[beat.class_name] || 'rgba(201, 203, 207, 0.3)',
          borderColor: 'transparent',
        };
      });
    return annotations;
  };

  const chartOptions = {
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: {
      legend: { display: false }, title: { display: true, text: `EKG Sinyali: ${activeFile?.file_name || ''}` },
      zoom: { pan: { enabled: true, mode: 'x' }, zoom: { wheel: { enabled: true }, mode: 'x' } },
      annotation: { annotations: getAnnotations() },
    },
    scales: { x: { title: { display: true, text: 'Örneklem' } }, y: { title: { display: true, text: 'Genlik (mV)' } } }
  };
  
  // GRAFİK VERİSİ ARTIK `display_signal`'DEN GELİYOR
  const chartData = {
    labels: analysisResult?.display_signal?.map((_, i) => i) || [],
    datasets: [{
      label: 'Temizlenmiş EKG', data: analysisResult?.display_signal || [],
      borderColor: '#007bff', borderWidth: 1, pointRadius: 0,
    }],
  };
  
  const clinicalFeatures = analysisResult?.clinical_features;
  const aiSummary = analysisResult?.ai_summary;
  const patientMeds = patient?.medications ? (typeof patient.medications === 'string' ? JSON.parse(patient.medications) : patient.medications) : [];

  if (!patient) {
    return (
      <div className="page" style={{ textAlign: 'center', padding: '4rem' }}>
        <h2>Lütfen Bir Hasta Seçin</h2>
        <p>Hasta detaylarını ve EKG kayıtlarını görmek için anasayfadan bir hasta seçin.</p>
      </div>
    );
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1>{patient.name}</h1>
        <p><strong>TC:</strong> {patient.tc} | <strong>Yaş:</strong> {patient.age} | <strong>Cinsiyet:</strong> {patient.gender}</p>
      </div>

      <div className="patient-card" style={{marginBottom: '1.5rem'}}>
        <h3>EKG Kayıtları ({files.length})</h3>
        <ul className="ekg-file-list">
          {files.map(file => <li key={file.id} className={file.id === activeFile?.id ? 'active-file' : ''} onClick={() => handleFileSelect(file)}>{file.file_name} <span>{file.uploaded_at}</span></li>)}
          {files.length === 0 && <p>Bu hasta için EKG kaydı bulunamadı.</p>}
        </ul>
        <label htmlFor="patient-file-upload" className="btn btn-secondary" style={{width: '100%', textAlign: 'center', marginTop: '1rem'}}>+ Yeni Kayıt Yükle</label>
        <input id="patient-file-upload" type="file" style={{display: 'none'}} onChange={handleFileUpload} accept=".csv,.txt,.dat"/>
      </div>
      
      <div className="chart-container-full" style={{ marginBottom: '1rem', position: 'relative' }}>
        {isLoading && <div className="loading-overlay"><span>Analiz yapılıyor...</span></div>}
        <div style={{height: '350px'}}>
          <Line ref={chartRef} options={chartOptions} data={chartData} />
        </div>
      </div>
      <button onClick={() => chartRef.current?.resetZoom()} className="btn btn-primary btn-sm" style={{marginBottom: '1.5rem'}}>Yakınlaştırmayı Sıfırla</button>

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
          ) : !isLoading && <p>Analiz için bir kayıt seçin.</p>}
        </div>
        <div className="patient-card">
          <h3>Yapay Zeka Bulguları</h3>
          {aiSummary ? (
            <div>
              <p><strong>Aritmi Riski:</strong> {aiSummary.arrhythmia_level || '--'}</p>
              <p><strong>Potansiyel Bulgular:</strong></p>
              <ul>{aiSummary.risk_findings?.map((f, i) => <li key={i}>{f}</li>)}</ul>
            </div>
          ) : !isLoading && <p>Analiz bekleniyor.</p>}
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