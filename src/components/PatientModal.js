import React, { useState, useEffect } from 'react';

const initialState = { name: '', tc: '', age: '', gender: '', phone: '', complaints: '' };

function PatientModal({ isOpen, onClose, onSave, patient }) {
  const [formData, setFormData] = useState(initialState);
  const [medications, setMedications] = useState([]);
  const [medInput, setMedInput] = useState('');
  const [errors, setErrors] = useState({});

  useEffect(() => {
    if (isOpen) {
      if (patient) {
        const patientMeds = typeof patient.medications === 'string' 
          ? JSON.parse(patient.medications) : (patient.medications || []);
        setFormData({ ...initialState, ...patient });
        setMedications(patientMeds);
      } else {
        setFormData(initialState);
        setMedications([]);
      }
      setErrors({});
    }
  }, [isOpen, patient]);

  // --- GÜNCELLENMİŞ DOĞRULAMA FONKSİYONU ---
  const validateField = (name, value) => {
    switch (name) {
      case 'tc':
        return /^\d{11}$/.test(value) ? '' : 'TC Kimlik No 11 haneli bir sayı olmalıdır.';
      case 'phone':
        // Boş bırakılabilir, ama doluysa 5 ile başlamalı ve 10 haneli olmalı
        return !value || /^5\d{9}$/.test(value.replace(/\D/g, '')) ? '' : 'Telefon 5xxxxxxxxx formatında 10 haneli olmalıdır.';
      case 'age':
        // Yaşın 0-120 arasında olmasını kontrol et
        const ageNum = parseInt(value, 10);
        return value && !isNaN(ageNum) && ageNum >= 0 && ageNum <= 120 ? '' : 'Yaş 0 ile 120 arasında olmalıdır.';
      default:
        return '';
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setErrors(prev => ({ ...prev, [name]: validateField(name, value) }));
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const addMedication = () => {
    const trimmedMed = medInput.trim();
    if (trimmedMed && !medications.includes(trimmedMed)) {
      setMedications([...medications, trimmedMed]);
      setMedInput('');
    }
  };
  const removeMedication = (medToRemove) => {
    setMedications(medications.filter(med => med !== medToRemove));
  };
  
  const handleSubmit = (e) => {
    e.preventDefault();
    const finalErrors = {};
    Object.keys(formData).forEach(key => {
      // Sadece formda görünen alanları doğrula
      if (['tc', 'phone', 'age'].includes(key)) {
        const error = validateField(key, formData[key]);
        if (error) finalErrors[key] = error;
      }
    });

    if (Object.keys(finalErrors).length > 0) {
        setErrors(finalErrors);
        alert("Lütfen formdaki hataları düzeltin.");
        return;
    }
    onSave({ ...formData, medications });
  };

  if (!isOpen) return null;

  return (
    <div className={`modal ${isOpen ? 'show' : ''}`}>
      <div className="modal-content">
        <span className="close" onClick={onClose} title="Kapat">×</span>
        <form onSubmit={handleSubmit}>
          <h2>{patient ? 'Hasta Bilgilerini Düzenle' : 'Yeni Hasta Ekle'}</h2>
          
          <div className="form-group">
            <label htmlFor="name">Ad Soyad:</label>
            <input type="text" id="name" name="name" value={formData.name} onChange={handleChange} required />
          </div>
          <div className="form-group">
            <label htmlFor="tc">TC Kimlik No:</label>
            <input type="text" id="tc" name="tc" value={formData.tc} onChange={handleChange} required maxLength="11" />
            {errors.tc && <small className="error-text">{errors.tc}</small>}
          </div>
          <div className="form-group">
            <label htmlFor="age">Yaş:</label>
            <input type="number" id="age" name="age" value={formData.age} onChange={handleChange} required min="0" max="120" />
            {errors.age && <small className="error-text">{errors.age}</small>}
          </div>
          <div className="form-group">
            <label htmlFor="gender">Cinsiyet:</label>
            <select id="gender" name="gender" value={formData.gender} onChange={handleChange} required>
              <option value="">Seçin</option><option value="Erkek">Erkek</option><option value="Kadın">Kadın</option>
            </select>
          </div>
          <div className="form-group">
            <label htmlFor="phone">Telefon (İsteğe Bağlı):</label>
            <input type="tel" id="phone" name="phone" value={formData.phone || ''} onChange={handleChange} placeholder="5xxxxxxxxx" maxLength="10"/>
            {errors.phone && <small className="error-text">{errors.phone}</small>}
          </div>
          <div className="form-group">
            <label htmlFor="medications">Kullanılan İlaçlar:</label>
            <div className="medication-input-group">
              <input 
                type="text" id="medications" value={medInput} 
                onChange={(e) => setMedInput(e.target.value)} 
                onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); addMedication(); } }}
                placeholder="İlaç ekleyip 'Ekle'ye basın..."
              />
              <button type="button" onClick={addMedication} className="btn btn-secondary btn-sm">Ekle</button>
            </div>
            <ul className="medication-list">
              {medications.map((med, index) => (
                <li key={index}>
                  {med} <span onClick={() => removeMedication(med)} title="Bu ilacı kaldır">×</span>
                </li>
              ))}
            </ul>
          </div>
          <div className="form-group">
            <label htmlFor="complaints">Şikayet / Notlar:</label>
            <textarea id="complaints" name="complaints" value={formData.complaints || ''} onChange={handleChange} rows="3"></textarea>
          </div>
          
          <button type="submit" className="btn btn-primary" style={{width: '100%', padding: '0.8rem'}}>
            {patient ? 'Bilgileri Güncelle' : 'Hastayı Kaydet'}
          </button>
        </form>
      </div>
    </div>
  );
}

export default PatientModal;