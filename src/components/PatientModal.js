import React, { useState, useEffect } from 'react';

// Varsayılan boş form durumu
const initialState = {
  name: '',
  tc: '',
  age: '',
  gender: '',
  phone: ''
};

function PatientModal({ isOpen, onClose, onSave, patient }) {
  const [formData, setFormData] = useState(initialState);

  // Modal açıldığında veya düzenlenecek hasta değiştiğinde, formu doldur/sıfırla
  useEffect(() => {
    if (isOpen) {
      if (patient) {
        // Düzenleme modu: Formu hasta verisiyle doldur
        setFormData(patient);
      } else {
        // Ekleme modu: Formu temizle
        setFormData(initialState);
      }
    }
  }, [isOpen, patient]);

  // Formdaki inputlar değiştikçe state'i güncelle
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevData => ({
      ...prevData,
      [name]: value
    }));
  };

  // Form gönderildiğinde
  const handleSubmit = (e) => {
    e.preventDefault(); // Sayfanın yeniden yüklenmesini engelle
    onSave(formData); // Veriyi App.js'e gönder
  };

  // Eğer modal açık değilse, hiçbir şey render etme (boşluk bile bırakma)
  if (!isOpen) {
    return null;
  }

  return (
    // 'modal show' class'ı ile modal'ı görünür yapıyoruz
    <div className="modal show"> 
      <div className="modal-content">
        <span className="close" onClick={onClose}>×</span>
        
        <form onSubmit={handleSubmit}>
          {/* Başlık, moda göre değişir */}
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
              <option value="E">Erkek</option>
              <option value="K">Kadın</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="phone">Telefon:</label>
            <input type="tel" id="phone" name="phone" value={formData.phone} onChange={handleChange} />
          </div>
          
          <button type="submit" className="btn btn-primary">
            {/* Buton metni, moda göre değişir */}
            {patient ? 'Güncelle' : 'Kaydet'}
          </button>
        </form>
      </div>
    </div>
  );
}

export default PatientModal;