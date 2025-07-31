import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

function HomePage({ patients, onSelectPatient, onAddPatient, onEditPatient, onDeletePatient }) {
  const [searchTerm, setSearchTerm] = useState('');
  const navigate = useNavigate();

  const filteredPatients = patients.filter(patient =>
    patient.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    patient.tc.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleSelectAndNavigate = (patientId) => {
    onSelectPatient(patientId);
    navigate('/detaylar');
  };

  return (
    <div className="page">
      <div className="page-header">
        <h1>Hasta Yönetimi</h1>
        <p>Toplam {patients.length} hasta kaydı bulunmaktadır.</p>
      </div>

      <div className="patient-controls">
        <button className="btn btn-primary" onClick={onAddPatient}>
          + Yeni Hasta Ekle
        </button>
      </div>

      <input
        type="text"
        className="search-bar"
        placeholder="🔍 Hasta adı veya TC ile ara..."
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
      />

      <div className="patient-list">
        {filteredPatients.map(patient => (
          <div key={patient.id} className="patient-list-item">
            <div className="patient-list-info" onClick={() => handleSelectAndNavigate(patient.id)}>
              <h3 className="patient-name">{patient.name}</h3>
              <p><strong>TC:</strong> {patient.tc} | <strong>Yaş:</strong> {patient.age} | <strong>Cinsiyet:</strong> {patient.gender === 'E' ? 'Erkek' : 'Kadın'}</p>
              <p><strong>Genel Durum:</strong> {patient.diagnosis || 'Belirtilmemiş'}</p>
            </div>
            
            <div className="patient-list-actions">
              <button className="btn btn-primary btn-sm" onClick={() => onEditPatient(patient)}>Düzenle</button>
              <button className="btn btn-danger btn-sm" onClick={() => onDeletePatient(patient.id)}>Sil</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default HomePage;