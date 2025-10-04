// Firebase Patient Management Service
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
  limit,
  serverTimestamp
} from 'firebase/firestore';
import { db } from './config';

export const patientService = {
  // Yeni hasta ekle
  async addPatient(patientData, doctorId) {
    try {
      console.log('patientService.addPatient başladı');
      console.log('patientData:', patientData);
      console.log('doctorId:', doctorId);
      
      const docRef = await addDoc(collection(db, 'patients'), {
        // Gelen hasta verileri
        firstName: patientData.firstName || '',
        lastName: patientData.lastName || '',
        tc: patientData.tc || '',
        age: patientData.age || '',
        gender: patientData.gender || '',
        phone: patientData.phone || '',
        email: patientData.email || '',
        dateOfBirth: patientData.dateOfBirth || '',
        bloodType: patientData.bloodType || '',
        medicalHistory: patientData.medicalHistory || '',
        complaints: patientData.complaints || '',
        emergencyContact: patientData.emergencyContact || {
          name: '',
          phone: '',
          relation: ''
        },
        allergies: patientData.allergies || [],
        medications: patientData.medications || [],
        // Sistem verileri
        doctorId: doctorId,
        createdAt: serverTimestamp(),
        updatedAt: serverTimestamp(),
        isActive: true,
        // EKG ile ilgili
        lastEkgDate: null,
        totalEkgTests: 0,
        lastAnalysisResult: null
      });
      
      console.log('Hasta başarıyla eklendi:', docRef.id);
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

  // Doktora ait tüm hastaları getir
  async getPatientsByDoctor(doctorId) {
    try {
      console.log('patientService.getPatientsByDoctor başladı');
      console.log('doctorId:', doctorId);
      
      const q = query(
        collection(db, 'patients'),
        where('doctorId', '==', doctorId),
        where('isActive', '==', true)
        // orderBy kaldırıldı - index gereksinimini önlemek için
      );
      
      console.log('Firestore sorgusu oluşturuldu, docs alınıyor...');
      const querySnapshot = await getDocs(q);
      console.log('querySnapshot.docs.length:', querySnapshot.docs.length);
      
      const patients = querySnapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }));
      
      // Client-side sorting
      patients.sort((a, b) => {
        const aTime = a.createdAt?.toDate?.() || new Date(0);
        const bTime = b.createdAt?.toDate?.() || new Date(0);
        return bTime - aTime; // Newest first
      });
      
      console.log('Dönüştürülen hastalar:', patients);
      return patients;
    } catch (error) {
      console.error('Hastalar getirilirken hata:', error);
      throw error;
    }
  },

  // Hasta ara (isim, email ile)
  async searchPatients(searchTerm, doctorId) {
    try {
      const q = query(
        collection(db, 'patients'),
        where('doctorId', '==', doctorId),
        where('isActive', '==', true)
      );
      
      const querySnapshot = await getDocs(q);
      const allPatients = querySnapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }));
      
      // Client-side filtreleme (Firestore'da complex search yok)
      return allPatients.filter(patient => 
        patient.firstName.toLowerCase().includes(searchTerm.toLowerCase()) ||
        patient.lastName.toLowerCase().includes(searchTerm.toLowerCase()) ||
        patient.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
        patient.tc.includes(searchTerm)
      );
    } catch (error) {
      console.error('Hasta aranırken hata:', error);
      throw error;
    }
  },

  // Hasta güncelle
  async updatePatient(patientId, updateData) {
    try {
      const docRef = doc(db, 'patients', patientId);
      await updateDoc(docRef, {
        ...updateData,
        updatedAt: serverTimestamp()
      });
      
      console.log('Hasta başarıyla güncellendi:', patientId);
      return patientId;
    } catch (error) {
      console.error('Hasta güncellenirken hata:', error);
      throw error;
    }
  },

  // Hasta sil (soft delete)
  async deletePatient(patientId) {
    try {
      const docRef = doc(db, 'patients', patientId);
      await updateDoc(docRef, {
        isActive: false,
        deletedAt: serverTimestamp()
      });
      
      console.log('Hasta başarıyla silindi:', patientId);
    } catch (error) {
      console.error('Hasta silinirken hata:', error);
      throw error;
    }
  },

  // Son EKG testini güncelle
  async updateLastEkgTest(patientId, analysisResult) {
    try {
      // Önce mevcut hasta verilerini al
      const docRef = doc(db, 'patients', patientId);
      const docSnap = await getDoc(docRef);
      
      if (!docSnap.exists()) {
        throw new Error('Hasta bulunamadı');
      }
      
      const patientData = docSnap.data();
      const currentTests = patientData.totalEkgTests || 0;
      
      await updateDoc(docRef, {
        lastEkgDate: serverTimestamp(),
        lastAnalysisResult: analysisResult,
        totalEkgTests: currentTests + 1,
        updatedAt: serverTimestamp()
      });
    } catch (error) {
      console.error('Son EKG testi güncellenirken hata:', error);
      throw error;
    }
  }
};

// Named exports for convenience
export const addPatient = patientService.addPatient.bind(patientService);
export const getPatientById = patientService.getPatient.bind(patientService);
export const getPatientsByDoctor = patientService.getPatientsByDoctor.bind(patientService);
export const updatePatient = patientService.updatePatient.bind(patientService);
export const deletePatient = patientService.deletePatient.bind(patientService);
export const searchPatients = patientService.searchPatients.bind(patientService);
export const updateLastEkgTest = patientService.updateLastEkgTest.bind(patientService);