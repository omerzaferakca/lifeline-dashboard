import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import './App.css';

import Sidebar from './components/Sidebar';
import PatientModal from './components/PatientModal';
import HomePage from './pages/HomePage';
import DetailsPage from './pages/DetailsPage';
import AboutPage from './pages/AboutPage';
import mockPatientsData from './data/mockPatients';

const AppLayout = () => {
  const [patients, setPatients] = useState(mockPatientsData);
  const [selectedPatientId, setSelectedPatientId] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [patientToEdit, setPatientToEdit] = useState(null); 
  const navigate = useNavigate();

  const handleOpenModal = (patient = null) => {
    setPatientToEdit(patient);
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    setPatientToEdit(null);
  };

  const handleSavePatient = (formData) => {
    if (formData.id) {
      setPatients(prev => prev.map(p => p.id === formData.id ? { ...p, ...formData } : p));
    } else {
      const newPatient = { 
          ...formData, 
          id: Date.now(), 
          diagnosis: "Yeni Kayıt",
          medications: [],
          ekgFiles: []
      };
      setPatients(prev => [...prev, newPatient]);
    }
    handleCloseModal();
  };

  const handleDeletePatient = (patientId) => {
    if (window.confirm("Bu hastayı kalıcı olarak silmek istediğinizden emin misiniz?")) {
        setPatients(prev => prev.filter(p => p.id !== patientId));
        if (selectedPatientId === patientId) {
            setSelectedPatientId(null);
            navigate('/'); // Silindikten sonra anasayfaya yönlendir
        }
    }
  };

  // SADECE YENİ DOSYAYI HASTA VERİSİNE EKLER. ANALİZ DETAILS SAYFASINDA YAPILIR.
  const handleAddFileToPatient = (patientId, newFile) => {
    setPatients(prevPatients =>
      prevPatients.map(p => {
        if (p.id === patientId) {
          // Önceki dosyalardaki analiz sonuçlarını korumak için, sadece yeni dosyayı ekle
          const updatedFiles = [...(p.ekgFiles || []), newFile];
          return { ...p, ekgFiles: updatedFiles };
        }
        return p;
      })
    );
    alert(`'${newFile.name}' dosyası başarıyla yüklendi ve analize hazır.`);
  };

  const resetSelection = () => setSelectedPatientId(null);

  return (
    <div className="app-container">
      <Sidebar onHomeClick={resetSelection} />
      <main className="main-content">
        <Routes>
          <Route path="/" element={
            <HomePage 
              patients={patients} 
              onSelectPatient={setSelectedPatientId} 
              onAddPatient={() => handleOpenModal()}
              onEditPatient={handleOpenModal}
              onDeletePatient={handleDeletePatient}
            />} 
          />
          <Route path="/detaylar" element={
            <DetailsPage 
              patients={patients} 
              patientId={selectedPatientId}
              onAddFile={handleAddFileToPatient}
            />}
          />
          <Route path="/hakkinda" element={<AboutPage />} />
        </Routes>
      </main>
      <PatientModal isOpen={isModalOpen} onClose={handleCloseModal} onSave={handleSavePatient} patient={patientToEdit} />
    </div>
  );
};

const App = () => (
  <Router>
    <AppLayout />
  </Router>
);

export default App;