// Firebase EKG Storage Service
import { 
  ref, 
  uploadBytes, 
  getDownloadURL, 
  deleteObject,
  listAll 
} from 'firebase/storage';
import { 
  collection, 
  doc, 
  addDoc, 
  getDoc, 
  getDocs, 
  updateDoc, 
  deleteDoc,
  query,
  where,
  orderBy,
  serverTimestamp
} from 'firebase/firestore';
import { storage, db } from './config';

export const ekgService = {
  // EKG dosyası yükle
  async uploadEkgFile(file, patientId, doctorId, analysisData = {}) {
    try {
      console.log('EKG dosyası yükleniyor:', file.name);
      
      // Dosya adını oluştur (timestamp + original name)
      const timestamp = Date.now();
      const fileName = `${timestamp}_${file.name}`;
      const filePath = `ekg/${doctorId}/${patientId}/${fileName}`;
      
      // Storage referansı oluştur
      const storageRef = ref(storage, filePath);
      
      // Dosyayı yükle
      console.log('Firebase Storage\'a yükleniyor...');
      const snapshot = await uploadBytes(storageRef, file);
      
      // Download URL al
      const downloadURL = await getDownloadURL(snapshot.ref);
      console.log('Dosya yüklendi, URL:', downloadURL);
      
      // Firestore'da analiz kaydı oluştur
      const analysisDoc = await addDoc(collection(db, 'analyses'), {
        patientId: patientId,
        doctorId: doctorId,
        fileName: file.name,
        originalFileName: file.name,
        storagePath: filePath,
        downloadURL: downloadURL,
        fileSize: file.size,
        fileType: file.type,
        uploadDate: serverTimestamp(),
        createdAt: serverTimestamp(),
        updatedAt: serverTimestamp(),
        // Analiz sonuçları
        analysisStatus: 'pending', // pending, processing, completed, failed
        analysisResult: analysisData.result || null,
        heartRate: analysisData.heartRate || null,
        anomalyDetected: analysisData.anomalyDetected || false,
        confidenceScore: analysisData.confidenceScore || null,
        processed: false
      });
      
      // Hasta dosyasında son EKG testini güncelle
      if (analysisData.result) {
        await this.updatePatientEkgStats(patientId, analysisData);
      }
      
      console.log('EKG analizi kaydedildi:', analysisDoc.id);
      
      return {
        success: true,
        id: analysisDoc.id,
        downloadURL: downloadURL,
        analysisData: analysisData
      };
    } catch (error) {
      console.error('EKG dosyası yüklenirken hata:', error);
      return {
        success: false,
        error: error.message || 'Dosya yükleme hatası'
      };
    }
  },

  // Hastaya ait tüm EKG analizlerini getir
  async getPatientEkgAnalyses(patientId) {
    try {
      const q = query(
        collection(db, 'analyses'),
        where('patientId', '==', patientId),
        orderBy('uploadDate', 'desc')
      );
      
      const querySnapshot = await getDocs(q);
      return querySnapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }));
    } catch (error) {
      console.error('EKG analizleri getirilirken hata:', error);
      throw error;
    }
  },

  // Doktora ait tüm EKG analizlerini getir
  async getDoctorEkgAnalyses(doctorId) {
    try {
      const q = query(
        collection(db, 'analyses'),
        where('doctorId', '==', doctorId),
        orderBy('uploadDate', 'desc')
      );
      
      const querySnapshot = await getDocs(q);
      return querySnapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }));
    } catch (error) {
      console.error('Doktor EKG analizleri getirilirken hata:', error);
      throw error;
    }
  },

  // Tek bir EKG analizini getir
  async getEkgAnalysis(analysisId) {
    try {
      const docRef = doc(db, 'analyses', analysisId);
      const docSnap = await getDoc(docRef);
      
      if (docSnap.exists()) {
        return { id: docSnap.id, ...docSnap.data() };
      } else {
        throw new Error('EKG analizi bulunamadı');
      }
    } catch (error) {
      console.error('EKG analizi getirilirken hata:', error);
      throw error;
    }
  },

  // EKG analiz sonucunu güncelle
  async updateEkgAnalysis(analysisId, analysisData) {
    try {
      const docRef = doc(db, 'analyses', analysisId);
      await updateDoc(docRef, {
        ...analysisData,
        analysisStatus: 'completed',
        processed: true,
        updatedAt: serverTimestamp()
      });
      
      console.log('EKG analizi güncellendi:', analysisId);
      return await this.getEkgAnalysis(analysisId);
    } catch (error) {
      console.error('EKG analizi güncellenirken hata:', error);
      throw error;
    }
  },

  // Desteklenen dosya formatlarını kontrol et
  validateEkgFile(file) {
    const allowedTypes = [
      'text/csv',
      'text/plain',
      'application/csv',
      '.csv',
      '.txt'
    ];
    
    const maxSize = 10 * 1024 * 1024; // 10MB
    
    if (!file) {
      throw new Error('Dosya seçilmedi');
    }
    
    if (file.size > maxSize) {
      throw new Error('Dosya boyutu 10MB\'dan küçük olmalıdır');
    }
    
    const fileExtension = file.name.split('.').pop().toLowerCase();
    const isValidType = allowedTypes.some(type => 
      file.type === type || fileExtension === type.replace('.', '')
    );
    
    if (!isValidType) {
      throw new Error('Sadece CSV ve TXT dosyaları desteklenmektedir');
    }
    
    return true;
  },

  // EKG verisini parse et (basit CSV parser)
  async parseEkgData(file) {
    try {
      const text = await file.text();
      const lines = text.split('\n').filter(line => line.trim());
      
      // İlk satır header olabilir, kontrol et
      const firstLine = lines[0];
      const hasHeader = isNaN(parseFloat(firstLine.split(',')[0]));
      
      const dataLines = hasHeader ? lines.slice(1) : lines;
      const ekgData = [];
      
      dataLines.forEach((line, index) => {
        const values = line.split(',').map(val => parseFloat(val.trim()));
        if (values.some(val => !isNaN(val))) {
          ekgData.push({
            timestamp: index,
            value: values[0] || 0,
            // Eğer ikinci sütun varsa o da eklenebilir
            ...(values[1] !== undefined && { value2: values[1] })
          });
        }
      });
      
      return {
        data: ekgData,
        totalSamples: ekgData.length,
        hasHeader: hasHeader,
        originalFileName: file.name
      };
    } catch (error) {
      console.error('EKG verisi parse edilirken hata:', error);
      throw new Error('EKG verisi okunamadı: ' + error.message);
    }
  },

  // Belirli bir hastanın EKG dosyalarını getir
  async getPatientEkgFiles(patientId) {
    try {
      console.log('Hasta EKG dosyaları getiriliyor:', patientId);
      
      const q = query(
        collection(db, 'analyses'),
        where('patientId', '==', patientId),
        orderBy('uploadDate', 'desc')
      );
      
      const querySnapshot = await getDocs(q);
      const files = [];
      
      for (const doc of querySnapshot.docs) {
        const data = doc.data();
        files.push({
          id: doc.id,
          ...data,
          uploaded_at: data.uploadedAt?.toDate?.()?.toISOString() || data.uploadedAt
        });
      }
      
      console.log(`${files.length} EKG dosyası bulundu`);
      return files;
    } catch (error) {
      console.error('EKG dosyaları getirilirken hata:', error);
      throw error;
    }
  },

  // Hasta EKG istatistiklerini güncelle (döngüsel import'u önlemek için)
  async updatePatientEkgStats(patientId, analysisData) {
    try {
      // Önce mevcut hasta verilerini al
      const docRef = doc(db, 'patients', patientId);
      const docSnap = await getDoc(docRef);
      
      if (!docSnap.exists()) {
        console.warn('Hasta bulunamadı, EKG istatistikleri güncellenemiyor:', patientId);
        return;
      }
      
      const patientData = docSnap.data();
      const currentTests = patientData.totalEkgTests || 0;
      
      await updateDoc(docRef, {
        lastEkgDate: serverTimestamp(),
        lastAnalysisResult: analysisData,
        totalEkgTests: currentTests + 1,
        updatedAt: serverTimestamp()
      });
      
      console.log('Hasta EKG istatistikleri güncellendi:', patientId);
    } catch (error) {
      console.error('Hasta EKG istatistikleri güncellenirken hata:', error);
      // Bu hata critical değil, sadece log'la
    }
  },

  // EKG dosyasını sil
  async deleteEkgFile(analysisId, storagePath) {
    try {
      console.log('EKG dosyası siliniyor:', analysisId);
      
      // Firestore'dan analiz kaydını sil
      const analysisRef = doc(db, 'analyses', analysisId);
      await deleteDoc(analysisRef);
      
      // Storage'dan dosyayı sil
      if (storagePath) {
        const storageRef = ref(storage, storagePath);
        await deleteObject(storageRef);
      }
      
      console.log('EKG dosyası başarıyla silindi');
      return { success: true };
    } catch (error) {
      console.error('EKG dosyası silinirken hata:', error);
      return {
        success: false,
        error: error.message || 'Dosya silme hatası'
      };
    }
  }
};

// Named exports for convenience
export const uploadEkgFile = ekgService.uploadEkgFile.bind(ekgService);
export const parseEkgData = ekgService.parseEkgData.bind(ekgService);
export const validateEkgFile = ekgService.validateEkgFile.bind(ekgService);
export const getPatientEkgFiles = ekgService.getPatientEkgFiles.bind(ekgService);
export const deleteEkgFile = ekgService.deleteEkgFile.bind(ekgService);
export const updatePatientEkgStats = ekgService.updatePatientEkgStats.bind(ekgService);