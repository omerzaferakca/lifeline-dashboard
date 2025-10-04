import React from 'react';

function ConfirmModal({ 
  isOpen, 
  onClose, 
  onConfirm, 
  title = "Onay", 
  message = "Bu işlemi gerçekleştirmek istediğinizden emin misiniz?",
  confirmText = "Evet",
  cancelText = "Hayır",
  type = "danger" // danger, warning, info, success
}) {
  if (!isOpen) return null;

  const handleConfirm = () => {
    onConfirm();
    onClose();
  };

  const getIconByType = () => {
    switch (type) {
      case 'danger':
        return (
          <div className="confirm-icon danger">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
              <path d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
        );
      case 'success':
        return (
          <div className="confirm-icon success">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
              <path d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
        );
      case 'warning':
        return (
          <div className="confirm-icon warning">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
              <path d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
        );
      default:
        return (
          <div className="confirm-icon info">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
              <path d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
        );
    }
  };

  return (
    <div className="confirm-modal-overlay">
      <div className="confirm-modal">
        <div className="confirm-modal-content">
          {getIconByType()}
          <h3 className="confirm-modal-title">{title}</h3>
          <p className="confirm-modal-message">{message}</p>
        </div>
        <div className="confirm-modal-actions">
          <button 
            type="button" 
            className="btn btn-secondary btn-full"
            onClick={onClose}
          >
            {cancelText}
          </button>
          <button 
            type="button" 
            className={`btn ${type === 'danger' ? 'btn-danger' : 'btn-primary'} btn-full`}
            onClick={handleConfirm}
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
}

export default ConfirmModal;