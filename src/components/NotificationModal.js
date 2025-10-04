import React, { useEffect } from 'react';

function NotificationModal({ 
  isOpen, 
  onClose, 
  title = "Bildirim", 
  message = "",
  type = "success", // success, error, warning, info
  autoClose = 3000 // 3 saniye sonra otomatik kapan
}) {
  useEffect(() => {
    if (isOpen && autoClose > 0) {
      const timer = setTimeout(() => {
        onClose();
      }, autoClose);
      
      return () => clearTimeout(timer);
    }
  }, [isOpen, autoClose, onClose]);

  if (!isOpen) return null;

  const getIconByType = () => {
    switch (type) {
      case 'success':
        return (
          <div className="notification-icon success">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none">
              <path d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
        );
      case 'error':
        return (
          <div className="notification-icon error">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none">
              <path d="M9.75 9.75l4.5 4.5m0-4.5l-4.5 4.5M21 12a9 9 0 11-18 0 9 9 0 0118 0z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
        );
      case 'warning':
        return (
          <div className="notification-icon warning">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none">
              <path d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
        );
      default:
        return (
          <div className="notification-icon info">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none">
              <path d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
        );
    }
  };

  return (
    <div className="notification-modal-overlay">
      <div className={`notification-modal ${type}`}>
        <button className="notification-close" onClick={onClose}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <path d="M6 18L18 6M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
        <div className="notification-content">
          {getIconByType()}
          <div className="notification-text">
            <h4 className="notification-title">{title}</h4>
            <p className="notification-message">{message}</p>
          </div>
        </div>
        <div className="notification-progress">
          <div className="notification-progress-bar" style={{animationDuration: `${autoClose}ms`}}></div>
        </div>
      </div>
    </div>
  );
}

export default NotificationModal;