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

const formatDate = (isoString) => {
    if (!isoString) return '';
    return new Date(isoString).toLocaleDateString('tr-TR', { day: '2-digit', month: '2-digit', year: 'numeric' });
};

const formatDateTime = (isoString) => {
    if (!isoString) return '';
    return new Date(isoString).toLocaleString('tr-TR', { 
        day: '2-digit', 
        month: '2-digit', 
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
};

// PDF'de Türkçe karakter sorunlarına karşı basit normalize edici (fallback)
const safeText = (str) => {
  if (str == null) return '';
  return String(str)
  .replaceAll('İ','I').replaceAll('İ','I').replaceAll('ı','i')
  .replaceAll('Ğ','G').replaceAll('ğ','g')
  .replaceAll('Ş','S').replaceAll('ş','s')
  .replaceAll('Ö','O').replaceAll('ö','o')
  .replaceAll('Ü','U').replaceAll('ü','u')
  .replaceAll('Ç','C').replaceAll('ç','c');
};

function DetailsPage({ patient }) {
  const [files, setFiles] = useState([]);
  const [activeFile, setActiveFile] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isGeneratingPDF, setIsGeneratingPDF] = useState(false);
  
  const chartRefs = {
    'Lead I': useRef(null),
    'Lead II': useRef(null),
    'Lead III': useRef(null),
  };

  // ---- ANALIZ CACHE (localStorage) ----
  const CACHE_VERSION = 'v1';
  const makeCacheKey = (file) => `lifeline:analysis:${CACHE_VERSION}:${file.id}`;
  const loadCachedAnalysis = (file) => {
    try {
      const raw = localStorage.getItem(makeCacheKey(file));
      if (!raw) return null;
      const cached = JSON.parse(raw);
      // Dosya değişmişse (tarih farklı) önbelleği kullanma
      if (cached?.uploaded_at !== file.uploaded_at) return null;
      return cached.result || null;
    } catch { return null; }
  };
  const saveCachedAnalysis = (file, result) => {
    try {
      const payload = { uploaded_at: file.uploaded_at, result, cached_at: new Date().toISOString() };
      localStorage.setItem(makeCacheKey(file), JSON.stringify(payload));
    } catch {
      // Olası quota hatalarını sessizce yutuyoruz
    }
  };
  const clearCachedAnalysis = (file) => {
    try { localStorage.removeItem(makeCacheKey(file)); } catch { /* ignore */ }
  };

  const handleFileSelect = React.useCallback(async (file) => {
    if (!file || isLoading) return;
    setActiveFile(file);
    // 1) Önbelleğe bak
    const cached = loadCachedAnalysis(file);
    if (cached) {
      setAnalysisResult(cached);
      return;
    }
    // 2) Yoksa analiz et ve önbelleğe yaz
    setIsLoading(true);
    setAnalysisResult(null);
    try {
      const response = await fetch(`${API_URL}/analyze/${file.id}`);
      const result = await response.json();
      if (result.success) {
        setAnalysisResult(result);
        saveCachedAnalysis(file, result);
      } else {
        alert(`Analiz hatası: ${result.error}`);
      }
    } catch (error) { console.error("Analiz hatası:", error); } 
    finally { setIsLoading(false); }
  }, [isLoading]);

  // Kullanıcı isterse önbelleği baypas ederek yeniden analiz edebilsin
  const forceReanalyze = async () => {
    if (!activeFile || isLoading) return;
    setIsLoading(true);
    setAnalysisResult(null);
    try {
      const response = await fetch(`${API_URL}/analyze/${activeFile.id}`);
      const result = await response.json();
      if (result.success) {
        setAnalysisResult(result);
        saveCachedAnalysis(activeFile, result);
      } else {
        alert(`Analiz hatası: ${result.error}`);
      }
    } catch (error) { console.error('Yeniden analiz hatası:', error); }
    finally { setIsLoading(false); }
  };

  const fetchFiles = React.useCallback(async (patientId) => {
    if (!patientId) return;
    try {
      const response = await fetch(`${API_URL}/patients/${patientId}/files`);
      const data = await response.json();
      setFiles(Array.isArray(data) ? data : []);
      // Loop'u önlemek için activeFile kontrolü kaldırıldı
      // İlk dosyayı otomatik seçme işlemi ayrı bir useEffect'te yapılacak
    } catch (error) { console.error("Dosyalar getirilirken hata:", error); }
  }, []);

  // Hasta değiştiğinde dosyaları getir
  useEffect(() => {
    if (patient) {
      setActiveFile(null);
      setAnalysisResult(null);
      fetchFiles(patient.id);
    }
  }, [patient, fetchFiles]);

  // Dosyalar yüklendiğinde ilk dosyayı seç (sadece activeFile null ise)
  useEffect(() => {
    if (files.length > 0 && !activeFile && !isLoading) {
      handleFileSelect(files[0]);
    }
  }, [files, activeFile, isLoading, handleFileSelect]);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file || !patient) return;
    const reader = new FileReader();
    reader.onload = async (e) => {
      const content = e.target.result;
      const ekgData = content.split(/[\n,;]/).map(val => parseFloat(val.trim())).filter(v => !isNaN(v));
      if (ekgData.length < 100) { alert("Geçersiz EKG verisi."); return; }
      const payload = { name: file.name, uploadedAt: new Date().toISOString(), data: ekgData, samplingRate: 200 };
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

  const loadJsPDF = () => {
    return new Promise((resolve, reject) => {
      // Eğer jsPDF zaten yüklenmişse
      if (window.jspdf) {
        resolve(window.jspdf);
        return;
      }

      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js';
      script.onload = () => {
        if (window.jspdf) {
          resolve(window.jspdf);
        } else {
          reject(new Error('jsPDF yüklenemedi'));
        }
      };
      script.onerror = () => reject(new Error('jsPDF script yüklenemedi'));
      document.head.appendChild(script);
    });
  };

  const loadHtml2Canvas = () => {
    return new Promise((resolve, reject) => {
      // Eğer html2canvas zaten yüklenmişse
      if (window.html2canvas) {
        resolve(window.html2canvas);
        return;
      }

      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js';
      script.onload = () => {
        if (window.html2canvas) {
          resolve(window.html2canvas);
        } else {
          reject(new Error('html2canvas yüklenemedi'));
        }
      };
      script.onerror = () => reject(new Error('html2canvas script yüklenemedi'));
      document.head.appendChild(script);
    });
  };

  const generatePDFReport = async () => {
    if (!patient || !analysisResult || !activeFile || isLoading) {
      alert("PDF oluşturmak için hasta ve analiz verisi gerekli.");
      return;
    }

    // Analiz verilerinin tam olduğunu kontrol et
    if (!analysisResult.clinical_metrics && !analysisResult.ai_summary) {
      alert("Analiz henüz tamamlanmamış. Lütfen bekleyin.");
      return;
    }

    setIsGeneratingPDF(true);
    
    try {
  // Kütüphaneleri yükle
  const { jsPDF } = await loadJsPDF();
      
      const doc = new jsPDF();
      const pageWidth = doc.internal.pageSize.width;
      const pageHeight = doc.internal.pageSize.height;
      let yPosition = 20;

  // Yazı tipi ve satır yüksekliği
  doc.setFont('helvetica', 'normal');
  doc.setLineHeightFactor(1.3);

  // Başlık
      doc.setFontSize(18);
  doc.setFont('helvetica', 'bold');
  doc.text(safeText('EKG ANALIZ RAPORU'), pageWidth / 2, yPosition, { align: 'center' });
      yPosition += 15;

      // Rapor tarihi
      doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  doc.text(safeText(`${formatDateTime(new Date().toISOString())}`), pageWidth - 20, yPosition, { align: 'right' });
      yPosition += 20;

      // Hasta Bilgileri
      doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  doc.text(safeText('HASTA BILGILERI'), 20, yPosition);
      yPosition += 10;

      doc.setFontSize(11);
  doc.setFont('helvetica', 'normal');
  doc.text(safeText(`Ad Soyad: ${patient.name}`), 20, yPosition);
      yPosition += 7;
  doc.text(safeText(`TC Kimlik No: ${patient.tc}`), 20, yPosition);
      yPosition += 7;
  doc.text(safeText(`Yas: ${patient.age}`), 20, yPosition);
      yPosition += 7;
  doc.text(safeText(`Cinsiyet: ${patient.gender}`), 20, yPosition);
      yPosition += 7;
  // Şikayet uzun olabilir, satırlara böl
  const complaintLines = doc.splitTextToSize(safeText(`Sikayet/Notlar: ${patient.complaints || 'Belirtilmemis'}`), pageWidth - 40);
  complaintLines.forEach(line => { doc.text(line, 20, yPosition); yPosition += 6; });
      yPosition += 15;

      // İlaçlar
      const patientMeds = patient.medications ? (typeof patient.medications === 'string' ? JSON.parse(patient.medications) : patient.medications) : [];
      doc.text(safeText('Kullanilan Ilaclar:'), 20, yPosition);
      yPosition += 7;
      if (patientMeds.length > 0) {
        patientMeds.forEach(med => {
          const lines = doc.splitTextToSize(safeText(`• ${med}`), pageWidth - 45);
          lines.forEach(line => { doc.text(line, 25, yPosition); yPosition += 6; });
        });
      } else {
        doc.text(safeText('• Ilac kaydi bulunmamaktadir'), 25, yPosition);
        yPosition += 6;
      }
      yPosition += 10;

      // EKG Kayıt Bilgileri
      doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  doc.text(safeText('EKG KAYIT BILGILERI'), 20, yPosition);
      yPosition += 10;

      doc.setFontSize(11);
  doc.setFont('helvetica', 'normal');
  doc.text(safeText(`Dosya Adi: ${activeFile.file_name}`), 20, yPosition);
      yPosition += 7;
  doc.text(safeText(`Kayit Tarihi: ${formatDate(activeFile.uploaded_at)}`), 20, yPosition);
      yPosition += 15;

      // Klinik Analiz Sonuçları
      const clinicalMetrics = analysisResult.clinical_metrics;
      doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  doc.text(safeText('KLINIK ANALIZ SONUCLARI'), 20, yPosition);
      yPosition += 10;

      doc.setFontSize(11);
  doc.setFont('helvetica', 'normal');
  doc.text(safeText(`Kalp Hizi: ${clinicalMetrics?.heart_rate_bpm?.toFixed(0) || '--'} bpm`), 20, yPosition);
      yPosition += 7;
  doc.text(safeText(`HRV (RMSSD): ${clinicalMetrics?.heart_rate_variability?.rmssd_ms?.toFixed(1) || '--'} ms`), 20, yPosition);
      yPosition += 7;
  doc.text(safeText(`QRS Suresi: ${clinicalMetrics?.qrs_duration_ms?.toFixed(0) || '--'} ms`), 20, yPosition);
      yPosition += 7;
  doc.text(safeText(`PR Araligi: ${clinicalMetrics?.pr_interval_ms?.toFixed(0) || '--'} ms`), 20, yPosition);
      yPosition += 7;
  doc.text(safeText(`QT Araligi: ${clinicalMetrics?.qt_interval_ms?.toFixed(0) || '--'} ms`), 20, yPosition);
      yPosition += 15;

      // AI Analiz Sonuçları
      const aiSummary = analysisResult.ai_summary;
      doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  doc.text(safeText('YAPAY ZEKA ANALIZI'), 20, yPosition);
      yPosition += 10;

      doc.setFontSize(11);
  doc.setFont('helvetica', 'normal');
  doc.text(safeText(`Risk Seviyesi: ${aiSummary?.risk_level || '--'}`), 20, yPosition);
      yPosition += 10;

      doc.text(safeText('Bulgular:'), 20, yPosition);
      yPosition += 7;
      if (aiSummary?.findings && aiSummary.findings.length > 0) {
        aiSummary.findings.forEach(finding => {
          // Uzun metinleri satırlara böl
          const lines = doc.splitTextToSize(safeText(`• ${finding}`), pageWidth - 45);
          lines.forEach(line => {
            if (yPosition > pageHeight - 30) {
              doc.addPage();
              yPosition = 20;
            }
            doc.text(line, 25, yPosition);
            yPosition += 6;
          });
        });
      } else {
        doc.text(safeText('• Ozel bulgu tespit edilmemistir'), 25, yPosition);
        yPosition += 6;
      }
      yPosition += 10;

      doc.text(safeText('Klinik Oneriler:'), 20, yPosition);
      yPosition += 7;
      if (aiSummary?.recommendations && aiSummary.recommendations.length > 0) {
        aiSummary.recommendations.forEach(recommendation => {
          // Uzun metinleri satırlara böl
          const lines = doc.splitTextToSize(safeText(`• ${recommendation}`), pageWidth - 45);
          lines.forEach(line => {
            if (yPosition > pageHeight - 30) {
              doc.addPage();
              yPosition = 20;
            }
            doc.text(line, 25, yPosition);
            yPosition += 6;
          });
        });
      } else {
        doc.text(safeText('• Spesifik oneri bulunmamaktadir'), 25, yPosition);
        yPosition += 6;
      }

      // Not: Grafik PDF'den kaldırıldı (istenen değişiklik)

      // Footer
      const totalPages = doc.internal.getNumberOfPages();
      for (let i = 1; i <= totalPages; i++) {
        doc.setPage(i);
        doc.setFontSize(8);
        doc.setFont('helvetica', 'normal');
        doc.text(safeText(`Sayfa ${i} / ${totalPages}`), pageWidth / 2, pageHeight - 10, { align: 'center' });
        doc.text(safeText('Bu rapor otomatik olarak olusturulmustur.'), pageWidth / 2, pageHeight - 5, { align: 'center' });
      }

      // PDF'i indir
      const fileName = `EKG_Rapor_${patient.name.replace(/\s+/g, '_')}_${new Date().toISOString().split('T')[0]}.pdf`;
      doc.save(fileName);

    } catch (error) {
      console.error('PDF olusturma hatasi:', error);
      alert(`PDF olusturulurken bir hata olustu: ${error.message}`);
    } finally {
      setIsGeneratingPDF(false);
    }
  };

  const getAnnotations = () => {
    if (!analysisResult?.beat_predictions || !analysisResult?.r_peaks) return {};
    const annotations = {};
    const beatWidth = (analysisResult.display_signal.length / analysisResult.r_peaks.length) * 0.7;
    const MAX_ANNOTATIONS = 300; // performans için üst sınır
    const anomalousBeats = analysisResult.beat_predictions.filter(beat => !beat.class_name.startsWith("Normal"));
    anomalousBeats.slice(0, MAX_ANNOTATIONS).forEach((beat, index) => {
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

  const syncCharts = (sourceChart) => {
      const { min, max } = sourceChart.scales.x;
      // Tüm büyük grafikleri senkronize et
      Object.values(chartRefs).forEach(ref => {
        const targetChart = ref.current;
        if (targetChart && targetChart.id !== sourceChart.id) {
          targetChart.zoomScale('x', { min, max }, 'none');
        }
      });
    };

  // Y ekseni için genlik aralığını hesapla (veriye göre 10% tampon)
  const getYRange = () => {
    const data = analysisResult?.display_signal;
    if (!Array.isArray(data) || data.length === 0) return null;
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (typeof v !== 'number' || Number.isNaN(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
    if (min === max) { min -= 1; max += 1; }
    const pad = (max - min) * 0.1;
    return { min: min - pad, max: max + pad };
  };
  const yRange = getYRange();

  const chartOptions = (leadName) => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    normalized: true,
    interaction: { mode: 'nearest', intersect: false, axis: 'x' },
    elements: { point: { radius: 0 }, line: { borderWidth: 1 } },
    plugins: {
      legend: { display: false },
      title: { display: true, text: leadName },
      zoom: { 
        pan: { enabled: true, mode: 'x', onPanComplete: ({chart}) => syncCharts(chart) }, 
        zoom: { wheel: { enabled: true }, mode: 'x', onZoomComplete: ({chart}) => syncCharts(chart) } 
      },
      annotation: { annotations: getAnnotations() },
    },
    scales: { 
      x: { title: { display: false }, ticks: { autoSkip: true, maxTicksLimit: 20 }, grid: { display: true } }, 
      y: { 
        title: { display: true, text: 'Genlik (mV)' }, 
        ticks: { display: true }, 
        grid: { display: true },
        ...(yRange ? { suggestedMin: yRange.min, suggestedMax: yRange.max } : {})
      } 
    }
  });
  
  const displaySignal = analysisResult?.display_signal
    ? (Array.isArray(analysisResult.display_signal) ? analysisResult.display_signal : Array.from(analysisResult.display_signal))
    : [];

  const chartData = {
    labels: displaySignal.map((_, i) => i),
    datasets: [{
      label: 'EKG', data: displaySignal,
      borderColor: '#dc3545', borderWidth: 1, pointRadius: 0,
      spanGaps: true,
    }],
  };
  
  if (!patient) {
    return (
      <div className="page" style={{ textAlign: 'center', padding: '4rem' }}>
        <h2>Lütfen Bir Hasta Seçin</h2>
      </div>
    );
  }

  const clinicalMetrics = analysisResult?.clinical_metrics;
  const aiSummary = analysisResult?.ai_summary;
  const patientMeds = patient.medications ? (typeof patient.medications === 'string' ? JSON.parse(patient.medications) : patient.medications) : [];

  return (
    <div className="page">
      <div className="page-header">
        <h1>{patient.name}</h1>
        <p><strong>TC:</strong> {patient.tc} | <strong>Yaş:</strong> {patient.age} | <strong>Cinsiyet:</strong> {patient.gender}</p>
      </div>

      {/* --- YENİ İKİ SÜTUNLU YAPI --- */}
      <div className="details-layout">
        
        {/* SOL SÜTUN: GRAFİKLER */}
        <div className="layout-left-column">
          <div className="patient-card" style={{marginBottom: '1.5rem'}}>
            <h3>EKG Kayıtları ({files.length})</h3>
            <ul className="ekg-file-list">
              {files.map(file => (
                <li key={file.id} className={file.id === activeFile?.id ? 'active-file' : ''}>
                  <div style={{ flexGrow: 1, cursor: 'pointer' }} onClick={() => handleFileSelect(file)}>
                    {file.file_name} <span>{formatDate(file.uploaded_at)}</span>
                  </div>
                  <button onClick={(e) => handleFileDelete(file.id, e)} className="btn btn-danger btn-sm" style={{ marginLeft: '1rem' }} title="Bu kaydı sil">Sil</button>
                </li>
              ))}
              {files.length === 0 && !isLoading && <p>Bu hasta için kayıt bulunamadı.</p>}
            </ul>
            <label htmlFor="patient-file-upload" className="btn btn-secondary" style={{width: '100%', textAlign: 'center', marginTop: '1rem'}}>+ Yeni Kayıt Yükle</label>
            <input id="patient-file-upload" type="file" style={{display: 'none'}} onChange={handleFileUpload} accept=".csv,.txt,.dat"/>
          </div>

          {/* Grafik Aksiyonları (üstte) */}
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginBottom: '0.5rem' }}>
            <button onClick={forceReanalyze} className="btn btn-primary btn-sm" disabled={!activeFile || isLoading} title="Bu kayıt için analiz yap veya güncelle">
              Analiz Et
            </button>
            <button onClick={() => { Object.values(chartRefs).forEach(ref => ref.current?.resetZoom()); }} className="btn btn-secondary btn-sm">
              Yakınlaştırmayı Sıfırla
            </button>
          </div>

          {/* 3 büyük grafik */}
          {['Lead I', 'Lead II', 'Lead III'].map(leadName => (
            <div key={leadName} className="chart-container-full" style={{ position: 'relative', height: '400px', marginBottom: '1rem' }}>
              {isLoading && <div className="loading-overlay"><span>Analiz yapılıyor...</span></div>}
              {analysisResult ? (
                <Line ref={chartRefs[leadName]} options={chartOptions(leadName)} data={chartData} />
              ) : !isLoading && (
                <div className="loading-placeholder" style={{height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
                  <span>Analiz için bir kayıt seçin.</span>
                </div>
              )}
            </div>
          ))}

          <div className="chart-actions-and-legend">
            <div className="legend-container">
              <div className="legend-item">
                <span className="legend-color-box" style={{ backgroundColor: ANOMALY_COLORS['S'].background }}></span>
                Supraventriküler (S)
              </div>
              <div className="legend-item">
                <span className="legend-color-box" style={{ backgroundColor: ANOMALY_COLORS['V'].background }}></span>
                Ventriküler (V)
              </div>
              <div className="legend-item">
                <span className="legend-color-box" style={{ backgroundColor: ANOMALY_COLORS['F'].background }}></span>
                Füzyon (F)
              </div>
              <div className="legend-item">
                <span className="legend-color-box" style={{ backgroundColor: ANOMALY_COLORS['Q'].background }}></span>
                Bilinmeyen (Q)
              </div>
            </div>
          </div>
        </div>

        {/* SAĞ SÜTUN: ANALİZ SONUÇLARI */}
        <div className="layout-right-column">
            <div className="patient-card">
              <h3>Klinik Metrikler</h3>
              {isLoading ? <p>Hesaplanıyor...</p> : analysisResult ? (
                <div>
                  <p><strong>Kalp Hızı:</strong> {clinicalMetrics?.heart_rate_bpm?.toFixed(0) || '--'} bpm</p>
                  <p><strong>HRV (RMSSD):</strong> {clinicalMetrics?.heart_rate_variability?.rmssd_ms?.toFixed(1) || '--'} ms</p>
                  <p><strong>QRS Süresi:</strong> {clinicalMetrics?.qrs_duration_ms?.toFixed(0) || '--'} ms</p>
                  <p><strong>PR Aralığı:</strong> {clinicalMetrics?.pr_interval_ms?.toFixed(0) || '--'} ms</p>
                  <p><strong>QT Aralığı:</strong> {clinicalMetrics?.qt_interval_ms?.toFixed(0) || '--'} ms</p>
                </div>
              ) : <p>Analiz için bir kayıt seçin.</p>}
            </div>

            <div className="patient-card">
              <h3>Yapay Zeka Özeti</h3>
              {isLoading ? <p>Oluşturuluyor...</p> : analysisResult ? (
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

            {/* PDF RAPOR BUTONU */}
            <div className="patient-card" style={{textAlign: 'center'}}>
              <button 
                onClick={generatePDFReport}
                disabled={!analysisResult || isGeneratingPDF || isLoading || !analysisResult?.clinical_metrics}
                className="btn btn-success"
                style={{
                  width: '100%',
                  padding: '12px 20px',
                  fontSize: '16px',
                  fontWeight: 'bold',
                  backgroundColor: (analysisResult && !isGeneratingPDF && !isLoading && analysisResult?.clinical_metrics) ? 'var(--primary-dark)' : '#6c757d',
                  border: 'none',
                  borderRadius: '6px',
                  color: 'white',
                  cursor: (analysisResult && !isGeneratingPDF && !isLoading && analysisResult?.clinical_metrics) ? 'pointer' : 'not-allowed',
                  transition: 'background-color 0.3s ease'
                }}
              >
                {isGeneratingPDF ? 'Rapor Oluşturuluyor...' : 'Rapor Oluştur'}
              </button>
              {(!analysisResult || isLoading) && (
                <p style={{marginTop: '10px', fontSize: '14px', color: '#6c757d', fontStyle: 'italic'}}>
                  {isLoading ? 'Analiz yapılıyor, lütfen bekleyin...' : 'Rapor oluşturmak için önce bir EKG analizi yapın.'}
                </p>
              )}
              {(analysisResult && !analysisResult?.clinical_metrics && !isLoading) && (
                <p style={{marginTop: '10px', fontSize: '14px', color: '#e74c3c', fontStyle: 'italic'}}>
                  Analiz tamamlanmadı. Lütfen bekleyin veya dosyayı yeniden seçin.
                </p>
              )}
            </div>
        </div>

      </div>
    </div>
  );
}

export default DetailsPage;