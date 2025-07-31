import React from 'react';

function InfoBox({ title, value, unit }) {
  return (
    <div className="info-box">
      <h3 className="info-title">{title}</h3>
      <p className="info-value">
        {value} <span className="info-unit">{unit}</span>
      </p>
    </div>
  );
}

export default InfoBox;