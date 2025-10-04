import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';

// Components ve Pages import'ları
import Sidebar from './components/Sidebar';
import HomePage from './pages/HomePage';
import DetailsPage from './pages/DetailsPage';
import AboutPage from './pages/AboutPage';
import LoginPage from './pages/LoginPage';
import PatientModal from './components/PatientModal';
import ConfirmModal from './components/ConfirmModal';
import NotificationModal from './components/NotificationModal';

// Contexts
import { AuthProvider, useAuth } from './contexts/AuthContext';

// Firebase services
import { deletePatient } from './firebase/patientService';

// Ana App Layout Component'i
const AppLayout = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [selectedPatientId, setSelectedPatientId] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [patientToEdit, setPatientToEdit] = useState(null);
  const [refreshPatients, setRefreshPatients] = useState(0); // Hasta listesini yenilemek için
  
  // Confirm modal states
  const [confirmModal, setConfirmModal] = useState({
    isOpen: false,
    title: '',
    message: '',
    onConfirm: null,
    type: 'danger'
  });
  
  // Notification modal states
  const [notification, setNotification] = useState({
    isOpen: false,
    title: '',
    message: '',
    type: 'success'
  });

  const showNotification = (title, message, type = 'success') => {
    setNotification({
      isOpen: true,
      title,
      message,
      type
    });
  };

  const closeNotification = () => {
    setNotification(prev => ({ ...prev, isOpen: false }));
  };

  const showConfirm = (title, message, onConfirm, type = 'danger') => {
    setConfirmModal({
      isOpen: true,
      title,
      message,
      onConfirm,
      type
    });
  };

  const closeConfirm = () => {
    setConfirmModal(prev => ({ ...prev, isOpen: false }));
  };

  const handleOpenModal = (patient = null) => {
    setPatientToEdit(patient);
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    setPatientToEdit(null);
  };

  const handlePatientSaved = () => {
    // Hasta kaydedildiğinde listeyi yenile
    setRefreshPatients(prev => prev + 1);
    showNotification(
      'Başarılı!',
      'Hasta bilgileri başarıyla kaydedildi.',
      'success'
    );
  };

  const handleDeletePatient = (patientId) => {
    showConfirm(
      'Hastayı Sil',
      'Bu hastayı ve tüm kayıtlarını kalıcı olarak silmek istediğinizden emin misiniz? Bu işlem geri alınamaz.',
      async () => {
        try {
          await deletePatient(patientId);
          // Eğer silinen hasta seçiliyse, seçimi kaldır
          if (selectedPatientId === patientId) {
            setSelectedPatientId(null);
          }
          // HomePage'i yenile
          setRefreshPatients(prev => prev + 1);
          showNotification(
            'Başarılı!',
            'Hasta ve tüm kayıtları başarıyla silindi.',
            'success'
          );
        } catch (error) {
          console.error('Hasta silinirken hata:', error);
          showNotification(
            'Hata!',
            'Hasta silinirken bir hata oluştu. Lütfen tekrar deneyin.',
            'error'
          );
        }
      },
      'danger'
    );
  };

  return (
    <div className="app-container">
      <Sidebar 
        onHomeClick={() => setSelectedPatientId(null)} 
        isOpen={isSidebarOpen}
        setIsOpen={setIsSidebarOpen}
      />
      
      <main className={`main-content ${isSidebarOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
        <Routes>
          <Route
            path="/"
            element={
              <HomePage
                onSelectPatient={(patient) => setSelectedPatientId(patient.id)}
                onAddPatient={() => handleOpenModal()}
                onEditPatient={handleOpenModal}
                onDeletePatient={handleDeletePatient}
                selectedPatientId={selectedPatientId}
                refreshTrigger={refreshPatients}
              />
            }
          />
          <Route
            path="/detaylar"
            element={
              <DetailsPage 
                selectedPatientId={selectedPatientId}
                showConfirm={showConfirm}
                showNotification={showNotification}
              />
            }
          />
          <Route path="/hakkinda" element={<AboutPage />} />
        </Routes>
      </main>
      
      <PatientModal
        isOpen={isModalOpen}
        onClose={handleCloseModal}
        onSave={handlePatientSaved}
        patient={patientToEdit}
        showConfirm={showConfirm}
        showNotification={showNotification}
      />
      
      <ConfirmModal
        isOpen={confirmModal.isOpen}
        onClose={closeConfirm}
        onConfirm={confirmModal.onConfirm}
        title={confirmModal.title}
        message={confirmModal.message}
        type={confirmModal.type}
        confirmText="Evet, Sil"
        cancelText="İptal"
      />
      
      <NotificationModal
        isOpen={notification.isOpen}
        onClose={closeNotification}
        title={notification.title}
        message={notification.message}
        type={notification.type}
        autoClose={4000}
      />
    </div>
  );
};

// Auth ile korumalı ana uygulama
const ProtectedApp = () => {
  const { currentUser, loading } = useAuth();

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="spinner"></div>
        <p>Yükleniyor...</p>
      </div>
    );
  }

  if (!currentUser) {
    return <LoginPage />;
  }

  return <AppLayout />;
};

const App = () => (
  <AuthProvider>
    <Router>
      <ProtectedApp />
    </Router>
  </AuthProvider>
);

export default App;