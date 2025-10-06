import React, { useState } from 'react';
import { NavLink, Link } from 'react-router-dom';
import logo from '../assets/logo.png'; 
import { useAuth } from '../contexts/AuthContext';

// react-icons kütüphanesinden ikonları import ediyoruz
import { 
  FaHome, 
  FaInfoCircle, 
  FaBars, 
  FaTimes,
  FaHeartbeat,
  FaSignOutAlt,
  FaUser,
  FaEdit
} from 'react-icons/fa';

// --- YENİ: Detaylar sayfası için MdOutlineMonitorHeart ikonu import edildi ---
import { MdOutlineMonitorHeart } from "react-icons/md";


function Sidebar({ onHomeClick, isOpen, setIsOpen }) {
  const { userProfile, logout } = useAuth();
  const [showProfileEdit, setShowProfileEdit] = useState(false);

  const handleLogout = async () => {
    try {
      await logout();
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  const handleProfileEdit = () => {
    setShowProfileEdit(true);
    // TODO: Implement profile edit modal
    alert('Profil düzenleme özelliği yakında eklenecek!');
  };

  return (
    <aside className={`sidebar ${isOpen ? '' : 'closed'}`} style={{
      background: 'linear-gradient(180deg, #1e40af 0%, #1d4ed8 50%, #2563eb 100%)',
      padding: 0
    }}>
      
      <div className="sidebar-toggle" onClick={() => setIsOpen(!isOpen)}>
        {isOpen ? <FaTimes /> : <FaBars />}
      </div>

      <div style={{ 
        display: 'flex', 
        flexDirection: 'column', 
        height: '100%',
        padding: '1.5rem 0 0 0'
      }}>
        <div>
          <Link to="/" className="sidebar-logo-link" onClick={onHomeClick}>
            <div className="sidebar-logo">
              <img src={logo} alt="LifeLine Logosu" className="logo-image" />
              <FaHeartbeat className="icon-logo" />
            </div>
          </Link>

          <nav>
            <ul className="sidebar-nav">
              <li>
                <NavLink to="/" end onClick={onHomeClick}>
                  <FaHome className="nav-icon" />
                  <span className="nav-text">Anasayfa</span>
                </NavLink>
              </li>
              <li>
                <NavLink to="/detaylar">
                  {/* --- DEĞİŞİKLİK: İkon güncellendi --- */}
                  <MdOutlineMonitorHeart className="nav-icon" />
                  <span className="nav-text">Detaylar</span>
                </NavLink>
              </li>
              <li>
                <NavLink to="/hakkinda">
                  <FaInfoCircle className="nav-icon" />
                  <span className="nav-text">Hakkında</span>
                </NavLink>
              </li>
            </ul>
          </nav>
        </div>

        {/* User Info and Logout - Fixed at bottom of sidebar */}
        {userProfile && (
          <div className={`sidebar-footer ${isOpen ? 'expanded' : 'collapsed'}`} style={{
            marginTop: 'auto',
            padding: isOpen ? '1rem' : '0.5rem',
            borderTop: '1px solid rgba(255, 255, 255, 0.1)',
            background: 'rgba(30, 64, 175, 0.8)',
            borderRadius: isOpen ? '0 0 15px 15px' : 'none',
            margin: isOpen ? 'auto 0.5rem 0.5rem 0.5rem' : 'auto 0 0 0'
          }}>
            {isOpen ? (
              // Expanded view
              <>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  marginBottom: '0.75rem',
                  color: 'white'
                }}>
                  <div style={{
                    background: 'rgba(255, 255, 255, 0.15)',
                    borderRadius: '50%',
                    padding: '0.75rem',
                    marginRight: '0.75rem',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    border: '2px solid rgba(255, 255, 255, 0.2)'
                  }}>
                    <FaUser style={{ fontSize: '1.1rem', color: 'white' }} />
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{
                      fontWeight: '600',
                      fontSize: '0.95rem',
                      marginBottom: '0.25rem',
                      color: 'white',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis'
                    }}>
                      {userProfile.displayName || `${userProfile.firstName} ${userProfile.lastName}`}
                    </div>
                    <div style={{
                      fontSize: '0.8rem',
                      color: 'rgba(255, 255, 255, 0.8)',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis'
                    }}>
                      {userProfile.specialization}
                    </div>
                  </div>
                  <button
                    onClick={handleProfileEdit}
                    style={{
                      background: 'rgba(255, 255, 255, 0.1)',
                      border: '1px solid rgba(255, 255, 255, 0.2)',
                      color: 'white',
                      padding: '0.5rem',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}
                    title="Profili Düzenle"
                    onMouseEnter={(e) => {
                      e.target.style.background = 'rgba(255, 255, 255, 0.2)';
                      e.target.style.transform = 'scale(1.05)';
                    }}
                    onMouseLeave={(e) => {
                      e.target.style.background = 'rgba(255, 255, 255, 0.1)';
                      e.target.style.transform = 'scale(1)';
                    }}
                  >
                    <FaEdit />
                  </button>
                </div>
                
                <button
                  onClick={handleLogout}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '0.5rem',
                    padding: '0.75rem',
                    background: 'rgba(220, 53, 69, 0.8)',
                    border: '1px solid rgba(220, 53, 69, 0.3)',
                    borderRadius: '8px',
                    color: 'white',
                    fontSize: '0.875rem',
                    fontWeight: '600',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    width: '100%'
                  }}
                  onMouseEnter={(e) => {
                    e.target.style.background = 'rgba(220, 53, 69, 1)';
                    e.target.style.transform = 'translateY(-1px)';
                    e.target.style.boxShadow = '0 4px 12px rgba(220, 53, 69, 0.3)';
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.background = 'rgba(220, 53, 69, 0.8)';
                    e.target.style.transform = 'translateY(0)';
                    e.target.style.boxShadow = 'none';
                  }}
                >
                  <FaSignOutAlt />
                  <span>Çıkış Yap</span>
                </button>
              </>
            ) : (
              // Collapsed view
              <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '0.75rem'
              }}>
                <button
                  onClick={handleProfileEdit}
                  style={{
                    background: 'rgba(255, 255, 255, 0.15)',
                    border: '2px solid rgba(255, 255, 255, 0.2)',
                    color: 'white',
                    padding: '0.75rem',
                    borderRadius: '50%',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}
                  title={userProfile.displayName || `${userProfile.firstName} ${userProfile.lastName}`}
                  onMouseEnter={(e) => {
                    e.target.style.background = 'rgba(255, 255, 255, 0.25)';
                    e.target.style.transform = 'scale(1.1)';
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.background = 'rgba(255, 255, 255, 0.15)';
                    e.target.style.transform = 'scale(1)';
                  }}
                >
                  <FaUser />
                </button>
                <button
                  onClick={handleLogout}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: '0.75rem',
                    background: 'rgba(220, 53, 69, 0.8)',
                    border: '1px solid rgba(220, 53, 69, 0.3)',
                    borderRadius: '50%',
                    color: 'white',
                    fontSize: '0.875rem',
                    fontWeight: '600',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease'
                  }}
                  title="Çıkış Yap"
                  onMouseEnter={(e) => {
                    e.target.style.background = 'rgba(220, 53, 69, 1)';
                    e.target.style.transform = 'translateY(-1px)';
                    e.target.style.boxShadow = '0 4px 12px rgba(220, 53, 69, 0.3)';
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.background = 'rgba(220, 53, 69, 0.8)';
                    e.target.style.transform = 'translateY(0)';
                    e.target.style.boxShadow = 'none';
                  }}
                >
                  <FaSignOutAlt />
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </aside>
  );
}

export default Sidebar;