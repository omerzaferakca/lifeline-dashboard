// Firebase Configuration Template
// Bu dosyayı config.js olarak kopyalayın ve kendi Firebase config değerlerinizi girin

import { initializeApp } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';
import { getStorage } from 'firebase/storage';
import { getAuth } from 'firebase/auth';

// Firebase config - LifeLine Dashboard
// Firebase Console'dan alınan değerleri buraya girin
const firebaseConfig = {
  apiKey: "YOUR_API_KEY_HERE",
  authDomain: "your-project.firebaseapp.com",
  projectId: "your-project-id",
  storageBucket: "your-project.firebasestorage.app",
  messagingSenderId: "your-sender-id",
  appId: "your-app-id",
  measurementId: "your-measurement-id"
};

// Firebase'i başlat
const app = initializeApp(firebaseConfig);

// Firebase servisleri
export const db = getFirestore(app);
export const storage = getStorage(app);
export const auth = getAuth(app);

export default app;