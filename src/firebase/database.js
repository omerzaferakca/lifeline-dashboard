// Firebase Firestore Database Operations
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
  limit
} from 'firebase/firestore';
import { db } from './config';

// Hasta işlemleri
export const patientService = {
  // Yeni hasta ekle
  async addPatient(patientData) {
    try {
      const docRef = await addDoc(collection(db, 'patients'), {
        ...patientData,
        createdAt: new Date(),
        updatedAt: new Date()
      });
      return docRef.id;
    } catch (error) {
      console.error('Hasta eklenirken hata:', error);
      throw error;
    }
  },

  // Hasta bilgilerini getir
  async getPatient(patientId) {
    try {
      const docRef = doc(db, 'patients', patientId);
      const docSnap = await getDoc(docRef);
      
      if (docSnap.exists()) {
        return { id: docSnap.id, ...docSnap.data() };
      } else {
        throw new Error('Hasta bulunamadı');
      }
    } catch (error) {
      console.error('Hasta getirilirken hata:', error);
      throw error;
    }
  },

  // Tüm hastaları getir
  async getAllPatients() {
    try {
      const querySnapshot = await getDocs(
        query(collection(db, 'patients'), orderBy('createdAt', 'desc'))
      );
      
      return querySnapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }));
    } catch (error) {
      console.error('Hastalar getirilirken hata:', error);
      throw error;
    }
  },

  // Hasta güncelle
  async updatePatient(patientId, updateData) {
    try {
      const docRef = doc(db, 'patients', patientId);
      await updateDoc(docRef, {
        ...updateData,
        updatedAt: new Date()
      });
    } catch (error) {
      console.error('Hasta güncellenirken hata:', error);
      throw error;
    }
  },

  // Hasta sil
  async deletePatient(patientId) {
    try {
      await deleteDoc(doc(db, 'patients', patientId));
    } catch (error) {
      console.error('Hasta silinirken hata:', error);
      throw error;
    }
  }
};

// EKG analiz işlemleri
export const analysisService = {
  // Yeni analiz ekle
  async addAnalysis(analysisData) {
    try {
      const docRef = await addDoc(collection(db, 'analyses'), {
        ...analysisData,
        createdAt: new Date()
      });
      return docRef.id;
    } catch (error) {
      console.error('Analiz eklenirken hata:', error);
      throw error;
    }
  },

  // Hastanın analizlerini getir
  async getPatientAnalyses(patientId) {
    try {
      const q = query(
        collection(db, 'analyses'),
        where('patientId', '==', patientId),
        orderBy('createdAt', 'desc')
      );
      
      const querySnapshot = await getDocs(q);
      return querySnapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }));
    } catch (error) {
      console.error('Analizler getirilirken hata:', error);
      throw error;
    }
  },

  // Analiz detayını getir
  async getAnalysis(analysisId) {
    try {
      const docRef = doc(db, 'analyses', analysisId);
      const docSnap = await getDoc(docRef);
      
      if (docSnap.exists()) {
        return { id: docSnap.id, ...docSnap.data() };
      } else {
        throw new Error('Analiz bulunamadı');
      }
    } catch (error) {
      console.error('Analiz getirilirken hata:', error);
      throw error;
    }
  }
};