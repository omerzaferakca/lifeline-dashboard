// Geçici Storage Çözümü - Local + Firestore Hybrid
import { 
  collection, 
  addDoc, 
  getDoc, 
  doc 
} from 'firebase/firestore';
import { db } from './config';

export const hybridStorageService = {
  // Küçük dosyalar için Firestore (Base64)
  async uploadSmallFile(file, patientId) {
    try {
      // Dosyayı Base64'e çevir
      const reader = new FileReader();
      return new Promise((resolve, reject) => {
        reader.onload = async () => {
          const base64Data = reader.result.split(',')[1];
          
          // Firestore'a kaydet
          const docRef = await addDoc(collection(db, 'files'), {
            patientId,
            fileName: file.name,
            fileSize: file.size,
            fileType: file.type,
            data: base64Data,
            uploadDate: new Date()
          });
          
          resolve(docRef.id);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
    } catch (error) {
      console.error('Dosya yüklenirken hata:', error);
      throw error;
    }
  },

  // Dosya indirme
  async downloadFile(fileId) {
    try {
      const docRef = doc(db, 'files', fileId);
      const docSnap = await getDoc(docRef);
      
      if (docSnap.exists()) {
        const fileData = docSnap.data();
        
        // Base64'ten Blob'a çevir
        const byteCharacters = atob(fileData.data);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: fileData.fileType });
        
        return {
          blob,
          fileName: fileData.fileName,
          fileSize: fileData.fileSize
        };
      }
    } catch (error) {
      console.error('Dosya indirilirken hata:', error);
      throw error;
    }
  }
};