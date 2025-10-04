// Firebase Storage Operations
import { 
  ref, 
  uploadBytes, 
  getDownloadURL, 
  deleteObject,
  listAll 
} from 'firebase/storage';
import { storage } from './config';

export const storageService = {
  // EKG dosyası yükle
  async uploadECGFile(file, patientId, analysisId) {
    try {
      const fileName = `${patientId}_${analysisId}_${file.name}`;
      const storageRef = ref(storage, `ecg-files/${fileName}`);
      
      const snapshot = await uploadBytes(storageRef, file);
      const downloadURL = await getDownloadURL(snapshot.ref);
      
      return {
        fileName,
        downloadURL,
        fullPath: snapshot.ref.fullPath,
        size: snapshot.metadata.size
      };
    } catch (error) {
      console.error('Dosya yüklenirken hata:', error);
      throw error;
    }
  },

  // Dosya URL'i getir
  async getFileURL(filePath) {
    try {
      const storageRef = ref(storage, filePath);
      return await getDownloadURL(storageRef);
    } catch (error) {
      console.error('Dosya URL getirilirken hata:', error);
      throw error;
    }
  },

  // Dosya sil
  async deleteFile(filePath) {
    try {
      const storageRef = ref(storage, filePath);
      await deleteObject(storageRef);
    } catch (error) {
      console.error('Dosya silinirken hata:', error);
      throw error;
    }
  },

  // Hastanın dosyalarını listele
  async listPatientFiles(patientId) {
    try {
      const storageRef = ref(storage, 'ecg-files/');
      const result = await listAll(storageRef);
      
      // Hasta ID'sine göre filtrele
      const patientFiles = result.items.filter(item => 
        item.name.startsWith(patientId)
      );
      
      return Promise.all(
        patientFiles.map(async (item) => ({
          name: item.name,
          fullPath: item.fullPath,
          downloadURL: await getDownloadURL(item)
        }))
      );
    } catch (error) {
      console.error('Dosyalar listelenirken hata:', error);
      throw error;
    }
  }
};