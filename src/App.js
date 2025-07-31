import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';

import Sidebar from './components/Sidebar';
import PatientModal from './components/PatientModal';
import HomePage from './pages/HomePage';
import DetailsPage from './pages/DetailsPage';
import AboutPage from './pages/AboutPage';

const API_URL = 'http://127.0.0.1:5001/api';

const AppLayout = () => {
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [patientToEdit, setPatientToEdit] = useState(null);

  // Hastaları backend'den getiren fonksiyon
  const fetchPatients = async () => {
    try {
      const response = await fetch(`${API_URL}/patients`);
      const data = await response.json();
      setPatients(data);
    } catch (error) {
      console.error("Hastalar getirilirken hata oluştu:", error);
      alert("Sunucu ile bağlantı kurulamadı. Lütfen backend'in çalıştığından emin olun.");
    }
  };

  // Bileşen ilk yüklendiğinde hastaları getir
  useEffect(() => {
    fetchPatients();
  }, []);

  const handleOpenModal = (patient = null) => {
    setPatientToEdit(patient);
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    setPatientToEdit(null);
  };

  const handleSavePatient = async (formData) => {
    const isEditing = !!formData.id;
    const url = isEditing ? `${API_URL}/patients/${formData.id}` : `${API_URL}/patients`;
    const method = isEditing ? 'PUT' : 'POST';

    try {
      const response = await fetch(url, {
        method: method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (response.ok) {
        await fetchPatients(); // Listeyi yenile
        handleCloseModal();
      } else {
        alert("Hasta kaydedilirken bir hata oluştu.");
      }
    } catch (error) {
      console.error("Hasta kaydetme hatası:", error);
      alert("Sunucuya bağlanılamadı.");
    }
  };

  const handleDeletePatient = async (patientId) => {
    if (window.confirm("Bu hastayı ve tüm EKG kayıtlarını kalıcı olarak silmek istediğinizden emin misiniz?")) {
      try {
        const response = await fetch(`${API_URL}/patients/${patientId}`, { method: 'DELETE' });
        if (response.ok) {
          await fetchPatients(); // Listeyi yenile
          if (selectedPatient?.id === patientId) {
            setSelectedPatient(null);
          }
        } else {
          alert("Hasta silinirken bir hata oluştu.");
        }
      } catch (error) {
        console.error("Hasta silme hatası:", error);
        alert("Sunucuya bağlanılamadı.");
      }
    }
  };

  const handleSelectPatient = (patient) => {
    setSelectedPatient(patient);
  };

  return (
    <div className="app-container">
      <Sidebar onHomeClick={() => setSelectedPatient(null)} />
      <main className="main-content">
        <Routes>
          <Route
            path="/"
            element={
              <HomePage
                patients={patients}
                onSelectPatient={handleSelectPatient}
                onAddPatient={() => handleOpenModal()}
                onEditPatient={handleOpenModal}
                onDeletePatient={handleDeletePatient}
              />
            }
          />
          <Route
            path="/detaylar"
            element={<DetailsPage patient={selectedPatient} />}
          />
          <Route path="/hakkinda" element={<AboutPage />} />
        </Routes>
      </main>
      <PatientModal
        isOpen={isModalOpen}
        onClose={handleCloseModal}
        onSave={handleSavePatient}
        patient={patientToEdit}
      />
    </div>
  );
};

const App = () => (
  <Router>
    <AppLayout />
  </Router>
);

export default App;