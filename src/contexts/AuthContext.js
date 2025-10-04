// Auth Context for React
import React, { createContext, useContext, useEffect, useState } from 'react';
import { userService } from '../firebase/userService';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [currentUser, setCurrentUser] = useState(null);
  const [userProfile, setUserProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Kullanıcı kayıt ol
  const register = async (email, password, userData) => {
    try {
      setError(null);
      const result = await userService.registerUser(email, password, userData);
      return result;
    } catch (error) {
      setError(error.message);
      throw error;
    }
  };

  // Kullanıcı giriş yap
  const login = async (email, password) => {
    try {
      setError(null);
      const result = await userService.signInUser(email, password);
      return result;
    } catch (error) {
      setError(error.message);
      throw error;
    }
  };

  // Kullanıcı çıkış yap
  const logout = async () => {
    try {
      setError(null);
      await userService.signOutUser();
    } catch (error) {
      setError(error.message);
      throw error;
    }
  };

  // Profil güncelle
  const updateProfile = async (updateData) => {
    try {
      setError(null);
      if (currentUser) {
        const updatedProfile = await userService.updateUserProfile(currentUser.uid, updateData);
        setUserProfile(updatedProfile);
        return updatedProfile;
      }
    } catch (error) {
      setError(error.message);
      throw error;
    }
  };

  // Auth state değişikliklerini dinle
  useEffect(() => {
    const unsubscribe = userService.onAuthStateChanged(({ user, profile }) => {
      setCurrentUser(user);
      setUserProfile(profile);
      setLoading(false);
    });

    return unsubscribe;
  }, []);

  const value = {
    currentUser,
    userProfile,
    loading,
    error,
    register,
    login,
    logout,
    updateProfile,
    setError
  };

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
};