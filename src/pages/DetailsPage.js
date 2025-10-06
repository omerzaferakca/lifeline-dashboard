import React, { useState, useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, Title, Tooltip, Legend,
} from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';
import zoomPlugin from 'chartjs-plugin-zoom';
import { hybridEkgService } from '../firebase/hybridEkgService';
import { useAuth } from '../contexts/AuthContext';
import { getPatientById } from '../firebase/patientService';

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  Title, Tooltip, Legend, annotationPlugin, zoomPlugin
);

const ANOMALY_COLORS = {
    'V': { background: 'rgba(255, 99, 132, 0.3)', border: 'rgba(255, 99, 132, 0.5)' },
    'S': { background: 'rgba(255, 159, 64, 0.3)', border: 'rgba(255, 159, 64, 0.5)' },
    'F': { background: 'rgba(153, 102, 255, 0.3)', border: 'rgba(153, 102, 255, 0.5)' },
    'Q': { background: 'rgba(75, 192, 192, 0.3)', border: 'rgba(75, 192, 192, 0.5)' },
};

const formatDate = (isoString) => {
    if (!isoString) return 'Tarih bilinmiyor';
    try {
        // Firebase Timestamp object'i olabilir
        let date;
        if (isoString && typeof isoString.toDate === 'function') {
            date = isoString.toDate();
        } else {
            date = new Date(isoString);
        }
        
        if (isNaN(date.getTime())) return 'Geçersiz tarih';
        
        return date.toLocaleDateString('tr-TR', { 
            day: '2-digit', 
            month: '2-digit', 
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch (error) {
        return 'Tarih hatası';
    }
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

function DetailsPage({ selectedPatientId, showConfirm, showNotification }) {
  const { currentUser } = useAuth();
  const [patient, setPatient] = useState(null);
  const [files, setFiles] = useState([]);
  const [activeFile, setActiveFile] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isGeneratingPDF, setIsGeneratingPDF] = useState(false);
  
  const chartRefs = {
    'Lead I': useRef(null),
    'Lead II': useRef(null),
    'Lead III': useRef(null),
  };

  const handleFileSelect = React.useCallback(async (file) => {
    if (!file || isLoading) return;
    console.log('=== EKG DOSYASI SEÇİLDİ ===');
    console.log('Dosya bilgileri:', file);
    
    setActiveFile(file);
    
    // Check if file already has analysis result with display_signal
    if (file.analysisMetadata && file.analysisMetadata.display_signal) {
      console.log('✅ Dosyada mevcut analiz sonucu ve display_signal bulundu');
      setAnalysisResult({ analysis_result: file.analysisMetadata });
      return;
    }
    
    console.log('🔄 Display signal bulunamadı, backend ile yeniden analiz başlatılıyor...');
    setIsLoading(true);
    setAnalysisResult(null);
    
    try {
      console.log('Hybrid backend ile analiz başlatılıyor:', file.storagePath);
      const result = await hybridEkgService.analyzeEkgFile(file.storagePath);
      
      if (result.success) {
        setAnalysisResult(result);
        console.log('Hybrid analiz başarılı:', result);
      } else {
        console.error('Hybrid analiz başarısız:', result);
        showNotification('Analiz Hatası', `Analiz hatası: ${result.error}`, 'error');
      }
    } catch (error) {
      console.error('Hybrid analiz hatası:', error);
      showNotification('Analiz Hatası', `Analiz hatası: ${error.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  }, [isLoading]);

  // Kullanıcı isterse yeniden analiz edebilsin
  const forceReanalyze = async () => {
    if (!activeFile || isLoading) return;
    console.log('🔄 FORCE REANALYZE - Hybrid backend ile analiz:', activeFile.originalFileName);
    
    setIsLoading(true);
    setAnalysisResult(null);
    
    try {
      const result = await hybridEkgService.analyzeEkgFile(activeFile.storagePath);
      
      if (result.success) {
        setAnalysisResult(result);
        console.log('Force reanalyze başarılı:', result);
      } else {
        console.error('Force reanalyze başarısız:', result);
        showNotification('Analiz Hatası', `Analiz hatası: ${result.error}`, 'error');
      }
    } catch (error) {
      console.error('Force reanalyze hatası:', error);
      showNotification('Analiz Hatası', `Analiz hatası: ${error.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchFiles = React.useCallback(async (patientId) => {
    if (!patientId) return;
    try {
      console.log('Loading EKG files for patient:', patientId);
      // Use hybrid service to get files from Firebase 'analyses' collection
      const files = await hybridEkgService.getPatientEkgFiles(patientId);
      console.log('Files retrieved:', files);
      setFiles(Array.isArray(files) ? files : []);
    } catch (error) { 
      console.error("Dosyalar getirilirken hata:", error);
      setFiles([]);
    }
  }, []);

  // selectedPatientId'den patient'ı fetch et
  useEffect(() => {
    const loadPatient = async () => {
      if (!selectedPatientId) {
        console.log('Loading patient: selectedPatientId is null');
        setPatient(null);
        return;
      }
      
      try {
        console.log('Loading patient:', selectedPatientId);
        const patientData = await getPatientById(selectedPatientId);
        console.log('Patient loaded:', patientData);
        setPatient(patientData);
      } catch (error) {
        console.error('Error loading patient:', error);
        setPatient(null);
      }
    };

    loadPatient();
  }, [selectedPatientId]);

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
    
    // File size validation (100MB limit)
    const maxSize = 100 * 1024 * 1024; // 100MB in bytes
    if (file.size > maxSize) {
      showNotification(
        'Dosya Boyutu Hatası', 
        `Dosya boyutu 100MB'dan büyük olamaz. Seçilen dosya: ${(file.size / (1024 * 1024)).toFixed(2)} MB`, 
        'error'
      );
      event.target.value = '';
      return;
    }
    
    try {
      setIsUploading(true);
      setUploadProgress(0);
      console.log('Starting hybrid upload with analysis...');
      
      // Use hybrid service for upload with automatic analysis
      const result = await hybridEkgService.uploadEkgFileWithAnalysis(
        file, 
        patient.id, 
        currentUser?.uid || 'unknown',
        (progress) => {
          const progressPercent = Math.round(progress * 100);
          setUploadProgress(progressPercent);
          console.log(`Upload progress: ${progressPercent}%`);
        }
      );
      
      console.log('Upload and analysis completed:', result);
      
      // Refresh file list
      setActiveFile(null);
      await fetchFiles(patient.id);
      
      showNotification('Başarılı', 'Dosya başarıyla yüklendi ve analiz edildi!', 'success');
      
    } catch (error) {
      console.error("Hybrid upload error:", error);
      showNotification('Hata', `Dosya yükleme ve analiz hatası: ${error.message}`, 'error');
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
    
    event.target.value = '';
  };
  
  const handleFileDelete = async (fileIdToDelete, event) => {
    event.stopPropagation();
    
    showConfirm(
      'Dosyayı Sil',
      'Bu EKG kaydını kalıcı olarak silmek istediğinizden emin misiniz?',
      async () => {
        setIsLoading(true);
        try {
          const fileToDelete = files.find(f => f.id === fileIdToDelete);
          await hybridEkgService.deleteEkgFile(fileIdToDelete, fileToDelete?.storagePath);
          
          // Clear active file if it was deleted
          if (activeFile?.id === fileIdToDelete) {
            setActiveFile(null);
            setAnalysisResult(null);
          }
          
          // Refresh file list
          await fetchFiles(patient.id);
          
          showNotification('Başarılı', 'Dosya başarıyla silindi!', 'success');
        } catch (error) {
          console.error('Dosya silme hatası:', error);
          showNotification('Hata', `Dosya silme hatası: ${error.message}`, 'error');
        } finally {
          setIsLoading(false);
        }
      },
      'danger'
    );
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
    if (!analysisResult?.analysis_result?.beat_predictions || !analysisResult?.analysis_result?.r_peaks || !analysisResult?.analysis_result?.display_signal) return {};
    const annotations = {};
    const beatWidth = (analysisResult.analysis_result.display_signal.length / analysisResult.analysis_result.r_peaks.length) * 0.7;
    const MAX_ANNOTATIONS = 300; // performans için üst sınır
    const anomalousBeats = analysisResult.analysis_result.beat_predictions.filter(beat => !beat.class_name.startsWith("Normal"));
    anomalousBeats.slice(0, MAX_ANNOTATIONS).forEach((beat, index) => {
        const rPeakIndex = analysisResult.analysis_result.r_peaks[beat.beat_id];
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
    const data = analysisResult?.analysis_result?.display_signal;
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
  
  console.log('=== GRAFİK RENDER (Lead I) ===');
  console.log('analysisResult var mı?', !!analysisResult);
  console.log('analysis_result mevcut mu?', !!analysisResult?.analysis_result);
  console.log('display_signal mevcut mu?', !!analysisResult?.analysis_result?.display_signal);
  console.log('displaySignal boyutu:', analysisResult?.analysis_result?.display_signal?.length || 0);
  
  const displaySignal = analysisResult?.analysis_result?.display_signal
    ? (Array.isArray(analysisResult.analysis_result.display_signal) ? analysisResult.analysis_result.display_signal : Array.from(analysisResult.analysis_result.display_signal))
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

  const clinicalMetrics = analysisResult?.analysis_result?.clinical_metrics;
  const aiSummary = analysisResult?.analysis_result?.ai_summary;
  const patientMeds = patient.medications ? (typeof patient.medications === 'string' ? JSON.parse(patient.medications) : patient.medications) : [];

  console.log('🔥 FULL analysisResult structure:', analysisResult);
  console.log('🔥 clinical_metrics:', clinicalMetrics);
  console.log('🔥 ai_summary:', aiSummary);

  return (
    <div className="page">
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <h1 style={{ margin: '0 0 0.5rem 0', color: '#1e293b' }}>EKG Detay Analizi</h1>
            {patient && (
              <div style={{ 
                fontSize: '1.2rem', 
                fontWeight: '600', 
                color: '#2563eb',
                background: 'linear-gradient(135deg, rgba(37, 99, 235, 0.1), rgba(59, 130, 246, 0.1))',
                padding: '0.75rem 1.25rem',
                borderRadius: '12px',
                border: '1px solid rgba(37, 99, 235, 0.2)',
                boxShadow: '0 2px 4px rgba(37, 99, 235, 0.1)'
              }}>
                👤 {patient.firstName} {patient.lastName}
                {patient.tcNo && (
                  <span style={{ fontSize: '1rem', color: '#64748b', marginLeft: '1.5rem', fontWeight: '400' }}>
                    TC: {patient.tcNo}
                  </span>
                )}
                {patient.age && (
                  <span style={{ fontSize: '1rem', color: '#64748b', marginLeft: '1rem', fontWeight: '400' }}>
                    Yaş: {patient.age}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
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
                    <div style={{ fontWeight: 'bold' }}>{file.originalFileName || file.fileName}</div>
                    <div style={{ fontSize: '0.85em', color: '#666' }}>
                      Yüklenme: {formatDate(file.uploadDate || file.createdAt)}
                    </div>
                  </div>
                  <button onClick={(e) => handleFileDelete(file.id, e)} className="btn btn-danger btn-sm" style={{ marginLeft: '1rem' }} title="Bu kaydı sil">Sil</button>
                </li>
              ))}
              {files.length === 0 && !isLoading && <p>Bu hasta için kayıt bulunamadı.</p>}
            </ul>
            <label htmlFor="patient-file-upload" className="btn btn-secondary" style={{width: '100%', textAlign: 'center', marginTop: '1rem'}}>
              + Yeni Kayıt Yükle
            </label>
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
              {analysisResult && analysisResult.analysis_result?.display_signal ? (
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
      
      {/* Upload Progress Modal */}
      {isUploading && (
        <div className="modal" style={{ display: 'block', backgroundColor: 'rgba(0,0,0,0.7)' }}>
          <div className="modal-content" style={{ 
            margin: '15% auto', 
            width: '450px', 
            padding: '30px', 
            backgroundColor: 'white', 
            borderRadius: '12px',
            boxShadow: '0 10px 30px rgba(0,0,0,0.3)',
            textAlign: 'center'
          }}>
            <div style={{ marginBottom: '20px' }}>
              <div style={{ 
                width: '60px', 
                height: '60px', 
                margin: '0 auto 15px', 
                borderRadius: '50%', 
                backgroundColor: '#e3f2fd',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <div style={{ 
                  width: '30px', 
                  height: '30px', 
                  border: '3px solid #2196f3',
                  borderTop: '3px solid transparent',
                  borderRadius: '50%',
                  animation: 'spin 1s linear infinite'
                }}></div>
              </div>
              <h4 style={{ color: '#333', marginBottom: '10px' }}>Dosya İşleniyor</h4>
              <p style={{ color: '#666', fontSize: '0.9em', margin: '0' }}>
                Dosya yükleniyor ve EKG analizi yapılıyor...
              </p>
            </div>
            
            <div style={{ marginBottom: '20px' }}>
              <div style={{ 
                width: '100%', 
                backgroundColor: '#f5f5f5', 
                borderRadius: '15px', 
                height: '8px',
                overflow: 'hidden'
              }}>
                <div 
                  style={{ 
                    width: `${uploadProgress}%`, 
                    height: '100%', 
                    background: 'linear-gradient(90deg, #4CAF50, #45a049)', 
                    borderRadius: '15px',
                    transition: 'width 0.3s ease'
                  }}
                ></div>
              </div>
              <p style={{ 
                marginTop: '10px', 
                fontSize: '1.1em', 
                fontWeight: '600',
                color: '#4CAF50' 
              }}>
                %{uploadProgress}
              </p>
            </div>
            
            <div style={{ 
              fontSize: '0.85em', 
              color: '#888',
              padding: '15px',
              backgroundColor: '#f9f9f9',
              borderRadius: '8px',
              border: '1px solid #eee'
            }}>
              <p style={{ margin: '0 0 5px 0' }}>
                {uploadProgress < 30 ? '⚡ Dosya Firebase Storage\'a yükleniyor' :
                 uploadProgress < 80 ? '🧠 Yapay zeka ile EKG sinyali analiz ediliyor' : 
                 '📊 Sonuçlar hazırlanıyor ve kaydediliyor'}
              </p>
              <div style={{ marginTop: '8px', fontSize: '0.8em', color: '#aaa' }}>
                {uploadProgress < 50 ? 'Bu işlem birkaç dakika sürebilir...' : 'Neredeyse tamam!'}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default DetailsPage;