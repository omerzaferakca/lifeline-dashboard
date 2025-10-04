import React from 'react';
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
  FaUser
} from 'react-icons/fa';

// --- YENİ: Detaylar sayfası için MdOutlineMonitorHeart ikonu import edildi ---
import { MdOutlineMonitorHeart } from "react-icons/md";


function Sidebar({ onHomeClick, isOpen, setIsOpen }) {
  const { userProfile, logout } = useAuth();

  const handleLogout = async () => {
    try {
      await logout();
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  return (
    <aside className={`sidebar ${isOpen ? '' : 'closed'}`}>
      
      <div className="sidebar-toggle" onClick={() => setIsOpen(!isOpen)}>
        {isOpen ? <FaTimes /> : <FaBars />}
      </div>

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

      {/* User Info and Logout */}
      {isOpen && userProfile && (
        <div className="sidebar-footer" style={{ 
          marginTop: 'auto', 
          padding: '1rem', 
          borderTop: '1px solid #e2e8f0',
          background: 'rgba(255, 255, 255, 0.05)'
        }}>
          <div className="user-info" style={{ 
            display: 'flex', 
            alignItems: 'center', 
            marginBottom: '0.5rem',
            color: '#fff'
          }}>
            <FaUser style={{ marginRight: '0.5rem', fontSize: '1rem' }} />
            <div style={{ fontSize: '0.9rem' }}>
              <div style={{ fontWeight: '500' }}>
                {userProfile.displayName || `${userProfile.firstName} ${userProfile.lastName}`}
              </div>
              <div style={{ fontSize: '0.75rem', opacity: '0.8' }}>
                {userProfile.specialization}
              </div>
            </div>
          </div>
          
          <button
            onClick={handleLogout}
            style={{
              display: 'flex',
              alignItems: 'center',
              width: '100%',
              padding: '0.5rem',
              background: 'rgba(255, 255, 255, 0.1)',
              border: 'none',
              borderRadius: '0.375rem',
              color: '#fff',
              fontSize: '0.875rem',
              cursor: 'pointer',
              transition: 'background-color 0.2s'
            }}
            onMouseEnter={(e) => e.target.style.backgroundColor = 'rgba(255, 255, 255, 0.2)'}
            onMouseLeave={(e) => e.target.style.backgroundColor = 'rgba(255, 255, 255, 0.1)'}
          >
            <FaSignOutAlt style={{ marginRight: '0.5rem' }} />
            Çıkış Yap
          </button>
        </div>
      )}
    </aside>
  );
}

export default Sidebar;