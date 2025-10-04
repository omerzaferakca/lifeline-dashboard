import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getPatientsByDoctor } from '../firebase/patientService';
import { useAuth } from '../contexts/AuthContext';

function HomePage({ onSelectPatient, onAddPatient, onEditPatient, onDeletePatient, selectedPatientId, refreshTrigger }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const navigate = useNavigate();
  const { currentUser } = useAuth();

  // Hastaları Firebase'den yükle
  useEffect(() => {
    const loadPatients = async () => {
      if (!currentUser) {
        setPatients([]);
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        setError('');
        console.log('HomePage: Hastalar yükleniyor...');
        const patientList = await getPatientsByDoctor(currentUser.uid);
        console.log('HomePage: Hasta listesi:', patientList);
        setPatients(patientList || []);
      } catch (err) {
        console.error('HomePage: Hastalar yüklenirken hata:', err);
        setError('Hastalar yüklenirken bir hata oluştu');
        setPatients([]);
      } finally {
        setLoading(false);
      }
    };

    loadPatients();
  }, [currentUser, refreshTrigger]); // refreshTrigger değiştiğinde listeyi yenile

  const handleSelectAndNavigate = (patient) => {
    onSelectPatient(patient);
    navigate('/detaylar');
  };

  if (loading) {
    return (
      <div className="page">
        <div className="loading-screen">
          <div className="spinner"></div>
          <p>Hastalar yükleniyor...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page">
        <div className="error-screen">
          <p style={{ color: 'red' }}>{error}</p>
          <button onClick={() => window.location.reload()}>Yeniden Dene</button>
        </div>
      </div>
    );
  }

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
          .filter(p => {
            const fullName = `${p.firstName} ${p.lastName}`.toLowerCase();
            const search = searchTerm.toLowerCase();
            return fullName.includes(search) || (p.tc && p.tc.includes(searchTerm));
          })
          .map(patient => (
            <div 
              key={patient.id} 
              className="patient-list-item" 
              onClick={() => handleSelectAndNavigate(patient)}
            >
              <div className="patient-list-header">
                <h3 className="patient-name">{patient.firstName} {patient.lastName}</h3>
                <div className="patient-list-actions">
                  <button 
                    className="btn btn-secondary btn-sm" 
                    onClick={(e) => { 
                      e.stopPropagation(); 
                      onEditPatient(patient); 
                    }}
                  >
                    Düzenle
                  </button>
                  <button 
                    className="btn btn-danger btn-sm" 
                    onClick={(e) => { 
                      e.stopPropagation(); 
                      onDeletePatient(patient.id); 
                    }}
                  >
                    Sil
                  </button>
                </div>
              </div>
              <div className="patient-list-details">
                <p>
                  <strong>TC:</strong> {patient.tc || 'Belirtilmemiş'} | 
                  <strong>Yaş:</strong> {patient.age || 'Belirtilmemiş'} | 
                  <strong>Cinsiyet:</strong> {patient.gender || 'Belirtilmemiş'}
                </p>
                {patient.phone && (
                  <p><strong>Telefon:</strong> {patient.phone}</p>
                )}
                {patient.email && (
                  <p><strong>E-mail:</strong> {patient.email}</p>
                )}
              </div>
            </div>
          ))}
        
        {patients.length === 0 && (
          <div className="empty-state">
            <p>Henüz hasta kaydı bulunmamaktadır.</p>
            <button className="btn btn-primary" onClick={onAddPatient}>
              İlk Hastayı Ekle
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default HomePage;