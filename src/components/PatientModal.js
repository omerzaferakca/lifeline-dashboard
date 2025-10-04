import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { patientService, deletePatient } from '../firebase/patientService';

const initialState = { 
  firstName: '', 
  lastName: '', 
  tc: '', 
  age: '', 
  gender: '', 
  phone: '', 
  email: '',
  dateOfBirth: '',
  bloodType: '',
  allergies: [],
  medications: [],
  medicalHistory: '',
  complaints: '',
  emergencyContact: {
    name: '',
    phone: '',
    relationship: ''
  }
};

function PatientModal({ isOpen, onClose, onSave, patient, showNotification, showConfirm }) {
  const [formData, setFormData] = useState(initialState);
  const [medications, setMedications] = useState([]);
  const [allergies, setAllergies] = useState([]);
  const [medInput, setMedInput] = useState('');
  const [allergyInput, setAllergyInput] = useState('');
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  
  const { currentUser } = useAuth();

  useEffect(() => {
    if (isOpen) {
      if (patient) {
        // Firebase'den gelen hasta verilerini forma doldur
        setFormData({
          ...initialState,
          firstName: patient.firstName || '',
          lastName: patient.lastName || '',
          tc: patient.tc || '',
          age: patient.age || '',
          gender: patient.gender || '',
          phone: patient.phone || '',
          email: patient.email || '',
          dateOfBirth: patient.dateOfBirth || '',
          bloodType: patient.bloodType || '',
          medicalHistory: patient.medicalHistory || '',
          complaints: patient.complaints || '',
          emergencyContact: patient.emergencyContact || initialState.emergencyContact
        });
        setMedications(patient.medications || []);
        setAllergies(patient.allergies || []);
      } else {
        setFormData(initialState);
        setMedications([]);
        setAllergies([]);
      }
      setErrors({});
      setMedInput('');
      setAllergyInput('');
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

  const addAllergy = () => {
    const trimmedAllergy = allergyInput.trim();
    if (trimmedAllergy && !allergies.includes(trimmedAllergy)) {
      setAllergies([...allergies, trimmedAllergy]);
      setAllergyInput('');
    }
  };

  const removeAllergy = (allergyToRemove) => {
    setAllergies(allergies.filter(allergy => allergy !== allergyToRemove));
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const finalErrors = {};
    Object.keys(formData).forEach(key => {
      // Zorunlu alanları kontrol et
      if (['firstName', 'lastName', 'tc', 'age', 'gender', 'phone'].includes(key)) {
        if (!formData[key] || !formData[key].toString().trim()) {
          finalErrors[key] = `${key === 'firstName' ? 'Ad' : 
                                key === 'lastName' ? 'Soyad' : 
                                key === 'tc' ? 'TC Kimlik No' : 
                                key === 'age' ? 'Yaş' : 
                                key === 'gender' ? 'Cinsiyet' : 
                                'Telefon'} zorunludur.`;
        }
      }
      
      // Özel doğrulamalar
      if (['tc', 'phone', 'age'].includes(key)) {
        const error = validateField(key, formData[key]);
        if (error) finalErrors[key] = error;
      }
    });

    // Email doğrulaması (zorunlu değil ama girilmişse doğru format olmalı)
    if (formData.email && formData.email.trim() && !/\S+@\S+\.\S+/.test(formData.email)) {
      finalErrors.email = 'Geçerli bir email adresi giriniz';
    }

    if (Object.keys(finalErrors).length > 0) {
      setErrors(finalErrors);
      alert("Lütfen formdaki hataları düzeltin.");
      return;
    }

    if (!currentUser) {
      setErrors({ submit: 'Kullanıcı oturumu bulunamadı' });
      return;
    }

    setLoading(true);
    
    try {
      console.log('PatientModal: Hasta kaydediliyor...');
      console.log('currentUser:', currentUser);
      console.log('formData:', formData);
      
      const patientData = {
        ...formData,
        medications,
        allergies
        // createdAt ve updatedAt patientService'de serverTimestamp() ile ayarlanacak
        // doctorId de patientService'de ayarlanacak
      };

      console.log('patientData:', patientData);

      if (patient?.id) {
        // Hasta güncelleme
        console.log('Hasta güncelleniyor, ID:', patient.id);
        await patientService.updatePatient(patient.id, patientData);
      } else {
        // Yeni hasta ekleme - sadece patientData gönder, doctorId servis içinde ayarlanacak
        console.log('Yeni hasta ekleniyor, doctorId:', currentUser.uid);
        const result = await patientService.addPatient(patientData, currentUser.uid);
        console.log('addPatient result:', result);
      }

      console.log('Hasta işlemi başarılı, onSave çağrılıyor...');
      if (onSave) {
        await onSave();
      }
      onClose();
    } catch (error) {
      console.error('=== HATA DETAYI ===');
      console.error('Hata mesajı:', error.message);
      console.error('Hata stack:', error.stack);
      console.error('currentUser:', currentUser);
      console.error('patient:', patient);
      console.error('formData:', formData);
      console.error('===================');
      setErrors({ submit: 'Hasta kaydedilirken bir hata oluştu: ' + error.message });
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = () => {
    if (!patient?.id) return;
    
    if (showConfirm) {
      showConfirm(
        'Hastayı Sil',
        'Bu hastayı ve tüm kayıtlarını kalıcı olarak silmek istediğinizden emin misiniz? Bu işlem geri alınamaz.',
        async () => {
          setLoading(true);
          try {
            await deletePatient(patient.id);
            if (showNotification) {
              showNotification('Başarılı!', 'Hasta başarıyla silindi.', 'success');
            }
            if (onSave) {
              await onSave();
            }
            onClose();
          } catch (error) {
            console.error('Hasta silinirken hata:', error);
            if (showNotification) {
              showNotification('Hata!', 'Hasta silinirken bir hata oluştu.', 'error');
            } else {
              setErrors({ submit: 'Hasta silinirken bir hata oluştu: ' + error.message });
            }
          } finally {
            setLoading(false);
          }
        },
        'danger'
      );
    } else {
      // Fallback to window.confirm if showConfirm is not provided
      if (window.confirm('Bu hastayı ve tüm kayıtlarını kalıcı olarak silmek istediğinizden emin misiniz?')) {
        const performDelete = async () => {
          setLoading(true);
          try {
            await deletePatient(patient.id);
            alert('Hasta başarıyla silindi.');
            if (onSave) {
              await onSave();
            }
            onClose();
          } catch (error) {
            console.error('Hasta silinirken hata:', error);
            setErrors({ submit: 'Hasta silinirken bir hata oluştu: ' + error.message });
          } finally {
            setLoading(false);
          }
        };
        performDelete();
      }
    }
  };

  if (!isOpen) return null;

  return (
    <div className={`modal ${isOpen ? 'show' : ''}`}>
      <div className="modal-content">
        <span className="close" onClick={onClose} title="Kapat">×</span>
        <form onSubmit={handleSubmit}>
          <h2>{patient ? 'Hasta Bilgilerini Düzenle' : 'Yeni Hasta Ekle'}</h2>
          
          {errors.submit && <div className="error-message">{errors.submit}</div>}
          
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="firstName">Ad:</label>
              <input 
                type="text" 
                id="firstName" 
                name="firstName" 
                value={formData.firstName} 
                onChange={handleChange} 
                required 
              />
              {errors.firstName && <small className="error-text">{errors.firstName}</small>}
            </div>
            <div className="form-group">
              <label htmlFor="lastName">Soyad:</label>
              <input 
                type="text" 
                id="lastName" 
                name="lastName" 
                value={formData.lastName} 
                onChange={handleChange} 
                required 
              />
              {errors.lastName && <small className="error-text">{errors.lastName}</small>}
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="tc">TC Kimlik No:</label>
              <input 
                type="text" 
                id="tc" 
                name="tc" 
                value={formData.tc} 
                onChange={handleChange} 
                required 
                maxLength="11" 
                placeholder="xxxxxxxxxxx"
              />
              {errors.tc && <small className="error-text">{errors.tc}</small>}
            </div>
            <div className="form-group">
              <label htmlFor="age">Yaş:</label>
              <input 
                type="number" 
                id="age" 
                name="age" 
                value={formData.age} 
                onChange={handleChange} 
                required 
                min="0" 
                max="120" 
              />
              {errors.age && <small className="error-text">{errors.age}</small>}
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="gender">Cinsiyet:</label>
              <select 
                id="gender" 
                name="gender" 
                value={formData.gender} 
                onChange={handleChange} 
                required
              >
                <option value="">Seçin</option>
                <option value="Erkek">Erkek</option>
                <option value="Kadın">Kadın</option>
              </select>
              {errors.gender && <small className="error-text">{errors.gender}</small>}
            </div>
            <div className="form-group">
              <label htmlFor="bloodType">Kan Grubu:</label>
              <select 
                id="bloodType" 
                name="bloodType" 
                value={formData.bloodType} 
                onChange={handleChange}
              >
                <option value="">Bilinmiyor</option>
                <option value="A+">A+</option>
                <option value="A-">A-</option>
                <option value="B+">B+</option>
                <option value="B-">B-</option>
                <option value="AB+">AB+</option>
                <option value="AB-">AB-</option>
                <option value="O+">O+</option>
                <option value="O-">O-</option>
              </select>
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="phone">Telefon:</label>
              <input 
                type="tel" 
                id="phone" 
                name="phone" 
                value={formData.phone} 
                onChange={handleChange} 
                required
                placeholder="5xxxxxxxxx" 
                maxLength="10"
              />
              {errors.phone && <small className="error-text">{errors.phone}</small>}
            </div>
            <div className="form-group">
              <label htmlFor="email">Email (İsteğe Bağlı):</label>
              <input 
                type="email" 
                id="email" 
                name="email" 
                value={formData.email} 
                onChange={handleChange}
                placeholder="ornek@email.com"
              />
              {errors.email && <small className="error-text">{errors.email}</small>}
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="dateOfBirth">Doğum Tarihi:</label>
            <input 
              type="date" 
              id="dateOfBirth" 
              name="dateOfBirth" 
              value={formData.dateOfBirth} 
              onChange={handleChange}
            />
          </div>

          <div className="form-group">
            <label htmlFor="emergencyContactName">Acil Durum İletişim - Ad Soyad:</label>
            <input 
              type="text" 
              id="emergencyContactName" 
              name="emergencyContactName" 
              value={formData.emergencyContact.name} 
              onChange={(e) => setFormData(prev => ({
                ...prev,
                emergencyContact: { ...prev.emergencyContact, name: e.target.value }
              }))}
              placeholder="Yakının adı soyadı"
            />
          </div>

          <div className="form-group">
            <label htmlFor="emergencyContactPhone">Acil Durum İletişim - Telefon:</label>
            <input 
              type="tel" 
              id="emergencyContactPhone" 
              name="emergencyContactPhone" 
              value={formData.emergencyContact.phone} 
              onChange={(e) => setFormData(prev => ({
                ...prev,
                emergencyContact: { ...prev.emergencyContact, phone: e.target.value }
              }))}
              placeholder="5xxxxxxxxx"
              maxLength="10"
            />
          </div>

          <div className="form-group">
            <label htmlFor="emergencyContactRelation">Acil Durum İletişim - Yakınlık:</label>
            <input 
              type="text" 
              id="emergencyContactRelation" 
              name="emergencyContactRelation" 
              value={formData.emergencyContact.relation} 
              onChange={(e) => setFormData(prev => ({
                ...prev,
                emergencyContact: { ...prev.emergencyContact, relation: e.target.value }
              }))}
              placeholder="Anne, Baba, Eş, vb."
            />
          </div>

          <div className="form-group">
            <label htmlFor="medicalHistory">Tıbbi Geçmiş:</label>
            <textarea 
              id="medicalHistory" 
              name="medicalHistory" 
              value={formData.medicalHistory} 
              onChange={handleChange} 
              rows="3"
              placeholder="Geçmiş hastalıklar, ameliyatlar, vb."
            ></textarea>
          </div>

          <div className="form-group">
            <label htmlFor="complaints">Şikayet / Notlar:</label>
            <textarea 
              id="complaints" 
              name="complaints" 
              value={formData.complaints} 
              onChange={handleChange} 
              rows="3"
              placeholder="Hasta şikayetleri ve notlar"
            ></textarea>
          </div>

          <div className="form-group">
            <label htmlFor="medications">Kullanılan İlaçlar:</label>
            <div className="medication-input-group">
              <input 
                type="text" 
                id="medications" 
                value={medInput} 
                onChange={(e) => setMedInput(e.target.value)} 
                onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); addMedication(); } }}
                placeholder="İlaç adını yazıp 'Ekle'ye basın..."
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
            <label htmlFor="allergies">Alerjiler:</label>
            <div className="medication-input-group">
              <input 
                type="text" 
                id="allergies" 
                value={allergyInput} 
                onChange={(e) => setAllergyInput(e.target.value)} 
                onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); addAllergy(); } }}
                placeholder="Alerji adını yazıp 'Ekle'ye basın..."
              />
              <button type="button" onClick={addAllergy} className="btn btn-secondary btn-sm">Ekle</button>
            </div>
            <ul className="medication-list">
              {allergies.map((allergy, index) => (
                <li key={index}>
                  {allergy} <span onClick={() => removeAllergy(allergy)} title="Bu alerjiyi kaldır">×</span>
                </li>
              ))}
            </ul>
          </div>
          
          <div className="form-actions-vertical">
            {patient && (
              <button 
                type="button" 
                onClick={handleDelete} 
                className="btn btn-danger btn-full"
                disabled={loading}
              >
                {loading ? 'Siliniyor...' : 'Hastayı Sil'}
              </button>
            )}
            <button 
              type="submit" 
              className="btn btn-primary btn-full"
              disabled={loading}
            >
              {loading ? 'Kaydediliyor...' : 
               patient ? 'Bilgileri Güncelle' : 'Hastayı Kaydet'}
            </button>
            <button 
              type="button" 
              onClick={onClose} 
              className="btn btn-secondary btn-full"
              disabled={loading}
            >
              İptal
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default PatientModal;