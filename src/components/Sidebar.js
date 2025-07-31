import React from 'react';
import { NavLink } from 'react-router-dom';

function Sidebar({ onHomeClick }) {
  return (
    <aside className="sidebar">
      <div>
        <div className="sidebar-logo"><h1>LifeLine</h1></div>
        <nav>
          <ul className="sidebar-nav">
            <li><NavLink to="/" end onClick={onHomeClick}>Anasayfa</NavLink></li>
            <li><NavLink to="/detaylar">Detaylar</NavLink></li>
            <li><NavLink to="/hakkinda">Hakkında</NavLink></li>
          </ul>
        </nav>
      </div>
      {/* Dosya yükleme bölümü buradan kaldırıldı */}
    </aside>
  );
}

export default Sidebar;