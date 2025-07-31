// src/App.js

import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import './App.css';

import Sidebar from './components/Sidebar';
import PatientModal from './components/PatientModal';
import HomePage from './pages/HomePage';
import DetailsPage from './pages/DetailsPage';
import AboutPage from './pages/AboutPage';

const API_URL = 'http://127.0.0.1:5001/api';

const AppLayout = () => {
  const [patients, setPatients] = useState([]);
  // --- DEĞİŞİKLİK: Sadece ID'yi tutmak daha güvenilirdir ---
  const [selectedPatientId, setSelectedPatientId] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [patientToEdit, setPatientToEdit] = useState(null);
  const navigate = useNavigate();

  const fetchPatients = async () => {
    try {
      const response = await fetch(`${API_URL}/patients`);
      const data = await response.json();
      setPatients(data);
    } catch (error) {
      console.error("Hastalar getirilirken hata oluştu:", error);
    }
  };

  useEffect(() => {
    fetchPatients();
  }, []);
  
  const handleSelectPatient = (patient) => {
    setSelectedPatientId(patient.id);
    navigate('/detaylar');
  };

  const handleOpenModal = (patient = null) => {
    setPatientToEdit(patient);
    setIsModalOpen(true);
  };
  const handleCloseModal = () => { setIsModalOpen(false); setPatientToEdit(null); };

  const handleSavePatient = async (formData) => {
    const isEditing = !!formData.id;
    const url = isEditing ? `${API_URL}/patients/${formData.id}` : `${API_URL}/patients`;
    const method = isEditing ? 'PUT' : 'POST';
    try {
      const response = await fetch(url, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(formData) });
      if (response.ok) { await fetchPatients(); handleCloseModal(); } 
      else { alert("Hasta kaydedilirken hata."); }
    } catch (error) { console.error("Hasta kaydetme hatası:", error); }
  };

  const handleDeletePatient = async (patientId) => {
    if (window.confirm("Bu hastayı ve tüm kayıtlarını kalıcı olarak silmek istediğinizden emin misiniz?")) {
      try {
        const response = await fetch(`${API_URL}/patients/${patientId}`, { method: 'DELETE' });
        if (response.ok) {
          await fetchPatients();
          if (selectedPatientId === patientId) { setSelectedPatientId(null); }
        } else { alert("Hasta silinirken hata."); }
      } catch (error) { console.error("Hasta silme hatası:", error); }
    }
  };

  return (
    <div className="app-container">
      <Sidebar onHomeClick={() => setSelectedPatientId(null)} />
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
            // --- DEĞİŞİKLİK: DetailsPage'e tam hasta objesini ve ID'yi gönder ---
            element={<DetailsPage patient={patients.find(p => p.id === selectedPatientId)} />}
          />
          <Route path="/hakkinda" element={<AboutPage />} />
        </Routes>
      </main>
      <PatientModal isOpen={isModalOpen} onClose={handleCloseModal} onSave={handleSavePatient} patient={patientToEdit} />
    </div>
  );
};

const App = () => (<Router><AppLayout /></Router>);
export default App;