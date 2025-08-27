import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

function HomePage({ patients, onSelectPatient, onAddPatient, onEditPatient, onDeletePatient }) {
  const [searchTerm, setSearchTerm] = useState('');
  const navigate = useNavigate();

  const handleSelectAndNavigate = (patient) => {
    onSelectPatient(patient);
    navigate('/detaylar');
  };

  return (
    <div className="page">
      <div className="page-header">
        <h1>Hasta Yönetimi</h1>
        <p>Kayıtlı {patients.length} hasta bulunmaktadır.</p>
      </div>

      <div className="controls-container">
        <input
          type="text"
          className="search-bar"
          placeholder="Hasta adı veya kimlik numarası ile ara..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
        <button className="btn btn-primary" onClick={onAddPatient}>
          + Yeni Hasta Ekle
        </button>
      </div>

      <div className="patient-list">
        {patients
          .filter(p => p.name.toLowerCase().includes(searchTerm.toLowerCase()) || p.tc.includes(searchTerm))
          .map(patient => (
            // --- YENİ: Tüm kart artık tıklanabilir bir Link ---
            <div 
              key={patient.id} 
              className="patient-list-item" 
              onClick={() => handleSelectAndNavigate(patient)}
            >
              <div className="patient-list-header">
                <h3 className="patient-name">{patient.name}</h3>
                <div className="patient-list-actions">
                  <button className="btn btn-secondary btn-sm" onClick={(e) => { e.stopPropagation(); onEditPatient(patient); }}>Düzenle</button>
                  <button className="btn btn-danger btn-sm" onClick={(e) => { e.stopPropagation(); onDeletePatient(patient.id); }}>Sil</button>
                </div>
              </div>
              <div className="patient-list-details">
                <p><strong>TC:</strong> {patient.tc} | <strong>Yaş:</strong> {patient.age} | <strong>Cinsiyet:</strong> {patient.gender}</p>
              </div>
            </div>
          ))}
      </div>
    </div>
  );
}

export default HomePage;