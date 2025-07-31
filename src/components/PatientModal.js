import React, { useState, useEffect } from 'react';

// Varsayılan boş form durumu
const initialState = {
  name: '',
  tc: '',
  age: '',
  gender: '',
  phone: '',
  medications: [], // Artık bir dizi olarak yönetilecek
  complaints: '' // Yeni alan
};

function PatientModal({ isOpen, onClose, onSave, patient }) {
  const [formData, setFormData] = useState(initialState);
  const [medsInput, setMedsInput] = useState(''); // İlaçlar için geçici input

  useEffect(() => {
    if (isOpen) {
      if (patient) {
        // Düzenleme modu: Formu hasta verisiyle doldur
        // Veritabanından gelen medications JSON string ise parse et
        const patientMeds = typeof patient.medications === 'string'
          ? JSON.parse(patient.medications)
          : patient.medications || [];
        
        setFormData({ ...initialState, ...patient, medications: patientMeds });
        setMedsInput(patientMeds.join(', '));
      } else {
        // Ekleme modu: Formu temizle
        setFormData(initialState);
        setMedsInput('');
      }
    }
  }, [isOpen, patient]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevData => ({
      ...prevData,
      [name]: value
    }));
  };

  const handleMedsChange = (e) => {
    setMedsInput(e.target.value);
    // Virgülle ayrılmış metni diziye çevir
    const medsArray = e.target.value.split(',').map(med => med.trim()).filter(Boolean);
    setFormData(prevData => ({
      ...prevData,
      medications: medsArray
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSave(formData);
  };

  if (!isOpen) return null;

  return (
    <div className="modal show">
      <div className="modal-content">
        <span className="close" onClick={onClose}>×</span>
        <form onSubmit={handleSubmit}>
          <h2>{patient ? 'Hasta Bilgilerini Düzenle' : 'Yeni Hasta Ekle'}</h2>
          
          <div className="form-group">
            <label htmlFor="name">Ad Soyad:</label>
            <input type="text" id="name" name="name" value={formData.name} onChange={handleChange} required />
          </div>
          <div className="form-group">
            <label htmlFor="tc">TC Kimlik No:</label>
            <input type="text" id="tc" name="tc" value={formData.tc} onChange={handleChange} required />
          </div>
          <div className="form-group">
            <label htmlFor="age">Yaş:</label>
            <input type="number" id="age" name="age" value={formData.age} onChange={handleChange} required />
          </div>
          <div className="form-group">
            <label htmlFor="gender">Cinsiyet:</label>
            <select id="gender" name="gender" value={formData.gender} onChange={handleChange} required>
              <option value="">Seçin</option>
              <option value="Erkek">Erkek</option>
              <option value="Kadın">Kadın</option>
            </select>
          </div>
          <div className="form-group">
            <label htmlFor="phone">Telefon:</label>
            <input type="tel" id="phone" name="phone" value={formData.phone} onChange={handleChange} />
          </div>
          <div className="form-group">
            <label htmlFor="medications">Kullanılan İlaçlar (Virgülle ayırın):</label>
            <input type="text" id="medications" name="medications" value={medsInput} onChange={handleMedsChange} />
          </div>
          <div className="form-group">
            <label htmlFor="complaints">Şikayet / Notlar:</label>
            <textarea id="complaints" name="complaints" value={formData.complaints} onChange={handleChange} rows="3"></textarea>
          </div>
          
          <button type="submit" className="btn btn-primary">
            {patient ? 'Güncelle' : 'Kaydet'}
          </button>
        </form>
      </div>
    </div>
  );
}

export default PatientModal;