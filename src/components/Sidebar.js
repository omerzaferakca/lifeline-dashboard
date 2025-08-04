import React from 'react';
import { NavLink, Link } from 'react-router-dom';
import logo from '../assets/logo.png'; 

// react-icons kütüphanesinden ikonları import ediyoruz
import { 
  FaHome, 
  FaInfoCircle, 
  FaBars, 
  FaTimes,
  FaHeartbeat
} from 'react-icons/fa';

// --- YENİ: Detaylar sayfası için MdOutlineMonitorHeart ikonu import edildi ---
import { MdOutlineMonitorHeart } from "react-icons/md";


function Sidebar({ onHomeClick, isOpen, setIsOpen }) {
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
    </aside>
  );
}

export default Sidebar;