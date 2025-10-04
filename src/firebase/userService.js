// Firebase User Management Service
import { 
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  onAuthStateChanged,
  updateProfile
} from 'firebase/auth';
import { 
  doc, 
  setDoc, 
  getDoc, 
  collection, 
  addDoc, 
  query, 
  where, 
  getDocs,
  updateDoc,
  serverTimestamp 
} from 'firebase/firestore';
import { auth, db } from './config';

export const userService = {
  // Kullanıcı kayıt ol
  async registerUser(email, password, userData) {
    try {
      // Authentication'da kullanıcı oluştur
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      const user = userCredential.user;
      
      // Display name güncelle
      await updateProfile(user, {
        displayName: userData.displayName || userData.firstName + ' ' + userData.lastName
      });
      
      // Firestore'da kullanıcı profili oluştur
      await setDoc(doc(db, 'users', user.uid), {
        uid: user.uid,
        email: user.email,
        firstName: userData.firstName,
        lastName: userData.lastName,
        displayName: userData.displayName || userData.firstName + ' ' + userData.lastName,
        role: userData.role || 'doctor', // doctor, patient, admin
        specialization: userData.specialization || '',
        hospital: userData.hospital || '',
        phone: userData.phone || '',
        createdAt: serverTimestamp(),
        updatedAt: serverTimestamp(),
        isActive: true
      });
      
      return {
        user,
        profile: await this.getUserProfile(user.uid)
      };
    } catch (error) {
      console.error('Kullanıcı kaydı sırasında hata:', error);
      throw error;
    }
  },

  // Kullanıcı giriş yap
  async signInUser(email, password) {
    try {
      const userCredential = await signInWithEmailAndPassword(auth, email, password);
      const user = userCredential.user;
      
      // Kullanıcı profilini getir
      const profile = await this.getUserProfile(user.uid);
      
      return { user, profile };
    } catch (error) {
      console.error('Giriş yapılırken hata:', error);
      throw error;
    }
  },

  // Kullanıcı çıkış yap
  async signOutUser() {
    try {
      await signOut(auth);
    } catch (error) {
      console.error('Çıkış yapılırken hata:', error);
      throw error;
    }
  },

  // Kullanıcı profili getir
  async getUserProfile(userId) {
    try {
      const docRef = doc(db, 'users', userId);
      const docSnap = await getDoc(docRef);
      
      if (docSnap.exists()) {
        return { id: docSnap.id, ...docSnap.data() };
      } else {
        throw new Error('Kullanıcı profili bulunamadı');
      }
    } catch (error) {
      console.error('Kullanıcı profili getirilirken hata:', error);
      throw error;
    }
  },

  // Kullanıcı profili güncelle
  async updateUserProfile(userId, updateData) {
    try {
      const docRef = doc(db, 'users', userId);
      await updateDoc(docRef, {
        ...updateData,
        updatedAt: serverTimestamp()
      });
      
      return await this.getUserProfile(userId);
    } catch (error) {
      console.error('Kullanıcı profili güncellenirken hata:', error);
      throw error;
    }
  },

  // Tüm doktorları getir
  async getDoctors() {
    try {
      const q = query(
        collection(db, 'users'), 
        where('role', '==', 'doctor'),
        where('isActive', '==', true)
      );
      
      const querySnapshot = await getDocs(q);
      return querySnapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }));
    } catch (error) {
      console.error('Doktorlar getirilirken hata:', error);
      throw error;
    }
  },

  // Auth state değişikliklerini dinle
  onAuthStateChanged(callback) {
    return onAuthStateChanged(auth, async (user) => {
      if (user) {
        try {
          const profile = await this.getUserProfile(user.uid);
          callback({ user, profile });
        } catch (error) {
          console.error('Kullanıcı profili alınırken hata:', error);
          callback({ user, profile: null });
        }
      } else {
        callback({ user: null, profile: null });
      }
    });
  },

  // Mevcut kullanıcıyı getir
  getCurrentUser() {
    return auth.currentUser;
  }
};