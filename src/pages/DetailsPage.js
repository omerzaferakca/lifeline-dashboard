import React, { useState, useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, Title, Tooltip, Legend,
} from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';
import zoomPlugin from 'chartjs-plugin-zoom';

// Firebase services
import { getPatientById } from '../firebase/patientService';
import { getPatientEkgFiles, uploadEkgFile, deleteEkgFile } from '../firebase/ekgService';
import { useAuth } from '../contexts/AuthContext';

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

function DetailsPage({ selectedPatientId, showConfirm, showNotification }) {
  const { currentUser } = useAuth();
  const [patient, setPatient] = useState(null);
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

  // Firebase'den hasta bilgilerini getir
  useEffect(() => {
    const loadPatient = async () => {
      if (!selectedPatientId || !currentUser) {
        setPatient(null);
        setFiles([]);
        setActiveFile(null);
        return;
      }

      try {
        console.log('Loading patient:', selectedPatientId);
        const patientData = await getPatientById(selectedPatientId);
        setPatient(patientData);
      } catch (error) {
        console.error('Patient load error:', error);
        setPatient(null);
        setFiles([]);
        setActiveFile(null);
      }
    };

    loadPatient();
  }, [selectedPatientId, currentUser]);

  // Firebase'den EKG dosyalarını getir
  useEffect(() => {
    const loadEkgFiles = async () => {
      if (!patient) {
        setFiles([]);
        setActiveFile(null);
        setAnalysisResult(null);
        return;
      }

      try {
        console.log('Loading EKG files for patient:', patient.id);
        const ekgFiles = await getPatientEkgFiles(patient.id);
        setFiles(ekgFiles || []);
        
        // İlk dosyayı otomatik seç ve analiz et
        if (ekgFiles && ekgFiles.length > 0) {
          const firstFile = ekgFiles[0];
          setActiveFile(firstFile);
          
          // İlk dosyayı otomatik analiz et
          setTimeout(() => {
            handleFileSelect(firstFile);
          }, 100);
        }
      } catch (error) {
        console.error('EKG files load error:', error);
        setFiles([]);
      }
    };

    loadEkgFiles();
  }, [patient]);

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
    setIsLoading(true);
    
    try {
      console.log('EKG dosyası seçildi:', file.fileName || file.file_name);
      
      // Firebase'den EKG verisini oku ve işle
      if (file.downloadURL) {
        const response = await fetch(file.downloadURL);
        const fileText = await response.text();
        
        // EKG verisini parse et
        const ekgData = parseEkgData(fileText, file.fileName || file.file_name);
        
        if (ekgData && ekgData.length > 0) {
          console.log(`EKG verisi başarıyla parse edildi: ${ekgData.length} örnek`);
          
          // Grafik için uygun örnekleme (çok büyük veriler için)
          let displayData = ekgData;
          if (ekgData.length > 5000) {
            // Büyük dosyalar için örnekleme yap
            const step = Math.ceil(ekgData.length / 5000);
            displayData = ekgData.filter((_, index) => index % step === 0);
            console.log(`Veri örneklendi: ${ekgData.length} -> ${displayData.length} örnek`);
          }
          
          // Basit R-peak tespiti için threshold hesapla
          const maxVal = Math.max(...displayData);
          const minVal = Math.min(...displayData);
          const threshold = minVal + (maxVal - minVal) * 0.6;
          
          // R-peak benzeri noktaları bul
          const peaks = [];
          for (let i = 5; i < displayData.length - 5; i++) {
            if (displayData[i] > threshold && 
                displayData[i] > displayData[i-1] && 
                displayData[i] > displayData[i+1] &&
                displayData[i] > displayData[i-2] && 
                displayData[i] > displayData[i+2]) {
              // Son peak'ten en az 20 örnek uzakta olsun
              if (peaks.length === 0 || (i - peaks[peaks.length - 1]) > 20) {
                peaks.push(i);
              }
            }
          }
          
          // Kalp hızını hesapla
          let heartRate = null;
          if (peaks.length > 1) {
            const avgInterval = (peaks[peaks.length - 1] - peaks[0]) / (peaks.length - 1);
            const samplingRate = file.samplingRate || 250; // Varsayılan sampling rate
            const bpm = Math.round(60 / (avgInterval / samplingRate));
            if (bpm >= 30 && bpm <= 300) { // Fizyolojik sınırlar
              heartRate = bpm;
            }
          }
          
          // Analiz sonucu oluştur
          const analysisResult = {
            success: true,
            display_signal: displayData,
            r_peaks: peaks,
            sampleCount: ekgData.length,
            heartRate: heartRate,
            samplingRate: file.samplingRate || 250,
            duration: ekgData.length / (file.samplingRate || 250),
            analysisType: 'real_time_display',
            timestamp: new Date().toISOString(),
            fileName: file.fileName || file.file_name,
            uploadDate: file.uploadDate || file.uploaded_at,
            
            // Basit klinik metrikler
            clinical_metrics: {
              heart_rate_bpm: heartRate,
              peak_count: peaks.length,
              signal_quality: peaks.length > 5 ? 'İyi' : 'Düşük',
              duration_seconds: ekgData.length / (file.samplingRate || 250)
            },
            
            // Basit AI özeti
            ai_summary: {
              risk_level: heartRate ? (heartRate < 60 ? 'Düşük' : heartRate > 100 ? 'Orta' : 'Normal') : 'Bilinmiyor',
              findings: [
                heartRate ? `Tespit edilen kalp hızı: ${heartRate} bpm` : 'Kalp hızı hesaplanamadı',
                `${peaks.length} adet peak tespit edildi`,
                `Kayıt süresi: ${(ekgData.length / (file.samplingRate || 250)).toFixed(1)} saniye`
              ],
              recommendations: [
                heartRate && heartRate < 60 ? 'Bradikardi tespit edildi, kardiyoloji değerlendirmesi önerilir' :
                heartRate && heartRate > 100 ? 'Taşikardi tespit edildi, klinik değerlendirme gerekebilir' :
                'Normal kalp hızı aralığında',
                'Detaylı analiz için profesyonel değerlendirme önerilir'
              ]
            }
          };
          
          setAnalysisResult(analysisResult);
          console.log('EKG analizi tamamlandı:', {
            heartRate,
            peaks: peaks.length,
            duration: (ekgData.length / (file.samplingRate || 250)).toFixed(1) + 's'
          });
          
        } else {
          console.warn('EKG verisi parse edilemedi veya boş');
          setAnalysisResult(null);
        }
      } else {
        console.warn('Dosya indirme URL\'si bulunamadı');
        setAnalysisResult(null);
      }
    } catch (error) {
      console.error('EKG dosyası işlenirken hata:', error);
      setAnalysisResult(null);
    } finally {
      setIsLoading(false);
    }
  }, [isLoading]);

  // EKG dosya formatını parse et - GELİŞTİRİLMİŞ VERSİYON
  const parseEkgData = (fileText, fileName) => {
    try {
      console.log('EKG dosyası parse ediliyor:', fileName);
      const lines = fileText.trim().split('\n');
      const data = [];
      
      // CSV formatını kontrol et
      if (fileName.toLowerCase().endsWith('.csv')) {
        console.log('CSV formatı tespit edildi');
        
        // Header satırını kontrol et
        const headerLine = lines[0].toLowerCase();
        let ekgColumnIndex = 1; // Varsayılan olarak 2. kolon
        
        // EKG verisi hangi kolonda kontrol et
        if (headerLine.includes('ekg') || headerLine.includes('ecg')) {
          const headers = lines[0].split(',');
          ekgColumnIndex = headers.findIndex(h => 
            h.toLowerCase().includes('ekg') || 
            h.toLowerCase().includes('ecg') ||
            h.toLowerCase().includes('signal')
          );
          if (ekgColumnIndex === -1) ekgColumnIndex = 1;
        }
        
        for (let i = 1; i < lines.length; i++) { // Header'ı atla
          const values = lines[i].split(',');
          if (values.length > ekgColumnIndex) {
            const value = parseFloat(values[ekgColumnIndex]);
            if (!isNaN(value) && isFinite(value)) {
              data.push(value);
            }
          }
        }
      } 
      // Text/DAT formatını kontrol et
      else if (fileName.toLowerCase().endsWith('.txt') || fileName.toLowerCase().endsWith('.dat')) {
        console.log('TXT/DAT formatı tespit edildi');
        
        for (let line of lines) {
          const cleanLine = line.trim();
          if (cleanLine === '' || cleanLine.startsWith('#') || cleanLine.startsWith('//')) {
            continue; // Boş satırları ve yorumları atla
          }
          
          // Satırdaki tüm sayıları bul
          const values = cleanLine.split(/[\s,\t;]+/);
          for (let val of values) {
            const value = parseFloat(val);
            if (!isNaN(value) && isFinite(value)) {
              data.push(value);
            }
          }
        }
      }
      // JSON formatını kontrol et
      else if (fileName.toLowerCase().endsWith('.json')) {
        console.log('JSON formatı tespit edildi');
        try {
          const jsonData = JSON.parse(fileText);
          if (Array.isArray(jsonData)) {
            for (let item of jsonData) {
              if (typeof item === 'number' && isFinite(item)) {
                data.push(item);
              } else if (typeof item === 'object' && item.ekg !== undefined) {
                const value = parseFloat(item.ekg);
                if (!isNaN(value) && isFinite(value)) {
                  data.push(value);
                }
              }
            }
          } else if (jsonData.data && Array.isArray(jsonData.data)) {
            for (let value of jsonData.data) {
              const num = parseFloat(value);
              if (!isNaN(num) && isFinite(num)) {
                data.push(num);
              }
            }
          }
        } catch (jsonError) {
          console.warn('JSON parse hatası:', jsonError);
        }
      }
      
      console.log(`Parse tamamlandı: ${data.length} veri noktası`);
      
      // Veri kalitesi kontrolü
      if (data.length < 100) {
        console.warn('Çok az veri noktası bulundu:', data.length);
        return [];
      }
      
      // Outlier'ları temizle (veri kalitesini artırmak için)
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      const std = Math.sqrt(data.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / data.length);
      const threshold = 5 * std; // 5 sigma kuralı
      
      const cleanedData = data.filter(value => Math.abs(value - mean) <= threshold);
      
      if (cleanedData.length < data.length * 0.95) {
        console.log(`${data.length - cleanedData.length} outlier temizlendi`);
      }
      
      return cleanedData;
      
    } catch (error) {
      console.error('EKG verisi parse hatası:', error);
      return [];
    }
  };

  // Basit kalp hızı hesaplama
  const calculateHeartRate = (ekgData) => {
    if (!ekgData || ekgData.length < 1000) return null;
    
    // Basit peak detection (örnek amaçlı)
    let peaks = 0;
    const threshold = Math.max(...ekgData) * 0.6;
    
    for (let i = 1; i < ekgData.length - 1; i++) {
      if (ekgData[i] > threshold && 
          ekgData[i] > ekgData[i-1] && 
          ekgData[i] > ekgData[i+1]) {
        peaks++;
      }
    }
    
    // Dakikada atım hesabı (varsayılan 360 Hz sampling rate)
    const durationInMinutes = ekgData.length / (360 * 60);
    return Math.round(peaks / durationInMinutes);
  };

  // Kullanıcı isterse yeniden analiz edebilsin
  const forceReanalyze = async () => {
    if (!activeFile || isLoading) return;
    setIsLoading(true);
    
    try {
      console.log('Yeniden analiz yapılıyor:', activeFile.fileName);
      
      if (activeFile.downloadURL) {
        const response = await fetch(activeFile.downloadURL);
        const fileText = await response.text();
        
        // EKG verisini parse et
        const ekgData = parseEkgData(fileText, activeFile.fileName);
        
        if (ekgData && ekgData.length > 0) {
          // Yeniden analiz sonucu oluştur
          const analysisResult = {
            success: true,
            display_signal: ekgData.slice(0, 3000), // İlk 3000 örnek için grafik
            sampleCount: ekgData.length,
            heartRate: calculateHeartRate(ekgData),
            analysisType: 'reanalyzed',
            timestamp: new Date().toISOString(),
            fileName: activeFile.fileName,
            uploadDate: activeFile.uploadDate
          };
          
          setAnalysisResult(analysisResult);
          console.log('Analiz tamamlandı, kalp hızı:', analysisResult.heartRate);
        }
      }
    } catch (error) {
      console.error('Yeniden analiz hatası:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // handleFileDelete fonksiyonu Firebase ile güncellenecek

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file || !patient || !currentUser) return;
    
    setIsLoading(true);
    
    // Yükleniyor mesajı göster
    if (showNotification) {
      showNotification('Dosya yükleniyor...', 'info');
    }
    
    try {
      console.log('EKG dosyası yükleniyor:', file.name);
      
      // Firebase Storage'a yükle ve Firestore'da kaydet
      const result = await uploadEkgFile(file, patient.id, currentUser.uid);
      
      if (result.success) {
        console.log('EKG dosyası başarıyla yüklendi');
        
        // Dosya listesini yenile
        const updatedFiles = await getPatientEkgFiles(patient.id);
        setFiles(updatedFiles || []);
        
        // Yeni dosyayı aktif olarak seç
        const newFile = updatedFiles?.find(f => f.fileName === file.name);
        if (newFile) {
          setActiveFile(newFile);
        }
        
        // Başarı popup'ı göster
        if (showNotification) {
          showNotification('EKG dosyası başarıyla yüklendi!', 'success');
        }
      } else {
        // Hata popup'ı göster
        if (showNotification) {
          showNotification(`Dosya yüklenemedi: ${result.error}`, 'error');
        }
      }
    } catch (error) {
      console.error('Dosya yükleme hatası:', error);
      if (showNotification) {
        showNotification('Dosya yüklenirken bir hata oluştu.', 'error');
      }
    } finally {
      setIsLoading(false);
      event.target.value = null;
    }
  };
  
  const handleFileDelete = async (fileIdToDelete, event) => {
    event.stopPropagation();
    
    const fileToDelete = files.find(f => f.id === fileIdToDelete);
    if (!fileToDelete) return;
    
    if (showConfirm) {
      showConfirm(
        'Bu EKG kaydını kalıcı olarak silmek istediğinizden emin misiniz?',
        `Dosya: ${fileToDelete.fileName || fileToDelete.originalFileName || 'Bilinmeyen Dosya'}`,
        async () => {
          setIsLoading(true);
          
          try {
            console.log('EKG dosyası siliniyor:', fileIdToDelete);
            
            const result = await deleteEkgFile(fileIdToDelete, fileToDelete.storagePath);
            
            if (result.success) {
              // Aktif dosya siliniyorsa, seçimi temizle
              if (activeFile?.id === fileIdToDelete) {
                setActiveFile(null);
                setAnalysisResult(null);
              }
              
              // Dosya listesini yenile
              const updatedFiles = await getPatientEkgFiles(patient.id);
              setFiles(updatedFiles || []);
              
              // Başarı mesajı
              if (showNotification) {
                showNotification('EKG dosyası başarıyla silindi.', 'success');
              }
            } else {
              if (showNotification) {
                showNotification(`Dosya silinemedi: ${result.error}`, 'error');
              }
            }
          } catch (error) {
            console.error('Dosya silme hatası:', error);
            if (showNotification) {
              showNotification('Dosya silinirken bir hata oluştu.', 'error');
            }
          } finally {
            setIsLoading(false);
          }
        },
        'delete'
      );
    } else {
      // Fallback to browser alert if showConfirm is not available
      if (window.confirm("Bu EKG kaydını kalıcı olarak silmek istediğinizden emin misiniz?")) {
        if (showNotification) {
          showNotification('Dosya silme özelliği henüz geliştirme aşamasında.', 'info');
        }
      }
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
    interaction: { 
      mode: 'nearest', 
      intersect: false, 
      axis: 'x' 
    },
    elements: { 
      point: { radius: 0 }, 
      line: { borderWidth: 1.5 } 
    },
    plugins: {
      legend: { 
        display: true,
        position: 'top',
        labels: {
          usePointStyle: true,
          boxWidth: 6,
          font: { size: 12 }
        }
      },
      title: { 
        display: true, 
        text: `${leadName} - ${analysisResult?.fileName || 'EKG Sinyali'}`,
        font: { size: 14, weight: 'bold' }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        callbacks: {
          title: function(context) {
            const timeInSeconds = parseFloat(context[0].label);
            return `Zaman: ${timeInSeconds}s`;
          },
          label: function(context) {
            if (context.datasetIndex === 0) {
              return `EKG: ${context.parsed.y.toFixed(3)} mV`;
            } else if (context.datasetIndex === 1 && context.parsed.y !== null) {
              return `R-Peak: ${context.parsed.y.toFixed(3)} mV`;
            }
            return '';
          }
        }
      },
      zoom: { 
        pan: { 
          enabled: true, 
          mode: 'x', 
          onPanComplete: ({chart}) => syncCharts(chart) 
        }, 
        zoom: { 
          wheel: { enabled: true }, 
          mode: 'x', 
          onZoomComplete: ({chart}) => syncCharts(chart) 
        } 
      },
      annotation: { annotations: getAnnotations() },
    },
    scales: { 
      x: { 
        title: { 
          display: true, 
          text: 'Zaman (saniye)',
          font: { size: 12 }
        }, 
        ticks: { 
          autoSkip: true, 
          maxTicksLimit: 20,
          callback: function(value, index, values) {
            return parseFloat(value).toFixed(1) + 's';
          }
        }, 
        grid: { 
          display: true,
          color: 'rgba(0,0,0,0.1)'
        } 
      }, 
      y: { 
        title: { 
          display: true, 
          text: 'Genlik (mV)',
          font: { size: 12 }
        }, 
        ticks: { 
          display: true,
          callback: function(value) {
            return value.toFixed(2);
          }
        }, 
        grid: { 
          display: true,
          color: 'rgba(0,0,0,0.1)'
        },
        ...(yRange ? { 
          suggestedMin: yRange.min, 
          suggestedMax: yRange.max 
        } : {})
      } 
    }
  });
  
  const displaySignal = analysisResult?.display_signal
    ? (Array.isArray(analysisResult.display_signal) ? analysisResult.display_signal : Array.from(analysisResult.display_signal))
    : [];

  // R-peak'leri için dataset oluştur
  const rPeaks = analysisResult?.r_peaks || [];
  const rPeakData = displaySignal.map((value, index) => {
    return rPeaks.includes(index) ? value : null;
  });

  const chartData = {
    labels: displaySignal.map((_, i) => {
      // X ekseni için zaman damgası (saniye cinsinden)
      const samplingRate = analysisResult?.samplingRate || 250;
      return (i / samplingRate).toFixed(2);
    }),
    datasets: [
      {
        label: 'EKG Sinyali',
        data: displaySignal,
        borderColor: '#dc3545',
        backgroundColor: 'rgba(220, 53, 69, 0.1)',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false,
        spanGaps: true,
        tension: 0,
      },
      {
        label: 'R-Peaks',
        data: rPeakData,
        borderColor: '#28a745',
        backgroundColor: '#28a745',
        pointBackgroundColor: '#28a745',
        pointBorderColor: '#ffffff',
        pointBorderWidth: 2,
        pointRadius: 4,
        showLine: false,
        pointHoverRadius: 6,
      }
    ],
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
                    <strong>{file.fileName || file.originalFileName || 'Bilinmeyen Dosya'}</strong>
                    <span style={{ marginLeft: '1rem', color: 'var(--text-color-light)', fontSize: '0.9rem' }}>
                      {formatDateTime(file.uploadDate?.seconds ? new Date(file.uploadDate.seconds * 1000).toISOString() : file.uploadDate)}
                    </span>
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