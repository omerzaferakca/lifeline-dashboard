// Firebase Authentication Operations
import { 
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signOut,
  onAuthStateChanged,
  updateProfile
} from 'firebase/auth';
import { auth } from './config';

export const authService = {
  // Giriş yap
  async signIn(email, password) {
    try {
      const userCredential = await signInWithEmailAndPassword(auth, email, password);
      return userCredential.user;
    } catch (error) {
      console.error('Giriş yapılırken hata:', error);
      throw error;
    }
  },

  // Kayıt ol
  async signUp(email, password, displayName) {
    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      
      // Kullanıcı adını güncelle
      if (displayName) {
        await updateProfile(userCredential.user, {
          displayName: displayName
        });
      }
      
      return userCredential.user;
    } catch (error) {
      console.error('Kayıt olurken hata:', error);
      throw error;
    }
  },

  // Çıkış yap
  async signOut() {
    try {
      await signOut(auth);
    } catch (error) {
      console.error('Çıkış yapılırken hata:', error);
      throw error;
    }
  },

  // Kullanıcı durumunu dinle
  onAuthStateChanged(callback) {
    return onAuthStateChanged(auth, callback);
  },

  // Mevcut kullanıcıyı getir
  getCurrentUser() {
    return auth.currentUser;
  }
};