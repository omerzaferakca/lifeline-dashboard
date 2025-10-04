import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { FaHeartbeat, FaExclamationTriangle } from 'react-icons/fa';

const LoginPage = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    firstName: '',
    lastName: '',
    specialization: '',
    hospital: '',
    phone: ''
  });
  const [loading, setLoading] = useState(false);
  const { login, register, error, setError } = useAuth();

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      if (isLogin) {
        await login(formData.email, formData.password);
      } else {
        await register(formData.email, formData.password, {
          firstName: formData.firstName,
          lastName: formData.lastName,
          specialization: formData.specialization,
          hospital: formData.hospital,
          phone: formData.phone,
          role: 'doctor'
        });
      }
    } catch (error) {
      console.error('Auth error:', error);
    } finally {
      setLoading(false);
    }
  };

  const toggleMode = () => {
    setIsLogin(!isLogin);
    setError(null);
    setFormData({
      email: '',
      password: '',
      firstName: '',
      lastName: '',
      specialization: '',
      hospital: '',
      phone: ''
    });
  };

  return (
    <div className="login-container">
      <div className="login-card">
        {/* Logo/Header */}
        <div className="login-header">
          <div className="login-logo">
            <div className="login-logo-icon">
              <FaHeartbeat />
            </div>
          </div>
          <h1 className="login-title">LifeLine</h1>
          <p className="login-subtitle">EKG Analysis Dashboard</p>
        </div>

        {/* Tab Toggle */}
        <div className="login-tabs">
          <button
            type="button"
            onClick={() => setIsLogin(true)}
            className={`login-tab ${isLogin ? 'active' : ''}`}
          >
            Giriş Yap
          </button>
          <button
            type="button"
            onClick={() => setIsLogin(false)}
            className={`login-tab ${!isLogin ? 'active' : ''}`}
          >
            Kayıt Ol
          </button>
        </div>

        {/* Error Message */}
        {error && (
          <div className="login-error">
            <FaExclamationTriangle />
            {error}
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="login-form">
          {/* Email */}
          <div className="login-form-group">
            <label className="login-label">Email</label>
            <input
              type="email"
              name="email"
              value={formData.email}
              onChange={handleInputChange}
              required
              className="login-input"
              placeholder="ornek@email.com"
            />
          </div>

          {/* Password */}
          <div className="login-form-group">
            <label className="login-label">Şifre</label>
            <input
              type="password"
              name="password"
              value={formData.password}
              onChange={handleInputChange}
              required
              minLength="6"
              className="login-input"
              placeholder="••••••••"
            />
          </div>

          {/* Register Fields */}
          {!isLogin && (
            <>
              <div className="login-form-row">
                <div className="login-form-group">
                  <label className="login-label">Ad</label>
                  <input
                    type="text"
                    name="firstName"
                    value={formData.firstName}
                    onChange={handleInputChange}
                    required={!isLogin}
                    className="login-input"
                    placeholder="Adınız"
                  />
                </div>
                <div className="login-form-group">
                  <label className="login-label">Soyad</label>
                  <input
                    type="text"
                    name="lastName"
                    value={formData.lastName}
                    onChange={handleInputChange}
                    required={!isLogin}
                    className="login-input"
                    placeholder="Soyadınız"
                  />
                </div>
              </div>

              <div className="login-form-group">
                <label className="login-label">Uzmanlık</label>
                <input
                  type="text"
                  name="specialization"
                  value={formData.specialization}
                  onChange={handleInputChange}
                  className="login-input"
                  placeholder="Kardiyoloji, İç Hastalıkları, vb."
                />
              </div>

              <div className="login-form-group">
                <label className="login-label">Hastane</label>
                <input
                  type="text"
                  name="hospital"
                  value={formData.hospital}
                  onChange={handleInputChange}
                  className="login-input"
                  placeholder="Hastane adı"
                />
              </div>

              <div className="login-form-group">
                <label className="login-label">Telefon</label>
                <input
                  type="tel"
                  name="phone"
                  value={formData.phone}
                  onChange={handleInputChange}
                  className="login-input"
                  placeholder="+90 555 123 45 67"
                />
              </div>
            </>
          )}

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading}
            className="login-submit"
          >
            <div className="login-submit-content">
              {loading && <div className="login-spinner"></div>}
              {loading ? 'İşleniyor...' : (isLogin ? 'Giriş Yap' : 'Kayıt Ol')}
            </div>
          </button>
        </form>

        {/* Toggle Link */}
        <div className="login-toggle">
          <button
            type="button"
            onClick={toggleMode}
            className="login-toggle-button"
          >
            {isLogin 
              ? 'Hesabınız yok mu? Kayıt olun' 
              : 'Zaten hesabınız var mı? Giriş yapın'
            }
          </button>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;