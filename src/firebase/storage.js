// Firebase Storage Operations
import { 
  ref, 
  uploadBytes, 
  getDownloadURL, 
  deleteObject,
  listAll,
  getBytes 
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

  // Dosya içeriğini doğrudan Firebase SDK ile oku (CORS sorunu olmadan)
  async getFileContent(downloadURL) {
    try {
      console.log('Firebase SDK ile dosya okunuyor:', downloadURL);
      
      // URL'den storage reference path'ini çıkar
      const url = new URL(downloadURL);
      const pathMatch = url.pathname.match(/\/o\/(.+)\?/);
      if (!pathMatch) {
        throw new Error('Invalid storage URL format');
      }
      
      const filePath = decodeURIComponent(pathMatch[1]);
      console.log('Dosya yolu çıkarıldı:', filePath);
      
      const storageRef = ref(storage, filePath);
      const arrayBuffer = await getBytes(storageRef);
      
      // ArrayBuffer'ı string'e çevir
      const decoder = new TextDecoder('utf-8');
      const text = decoder.decode(arrayBuffer);
      
      console.log('Dosya başarıyla okundu, boyut:', text.length, 'karakter');
      return text;
    } catch (error) {
      console.error('Firebase SDK ile dosya okuma hatası:', error);
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