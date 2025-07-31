import React, { useState } from 'react';
import InfoBox from './InfoBox';
import ChartComponent from './ChartComponent';

function Dashboard() {
  const [anomalyStatus, setAnomalyStatus] = useState('NORMAL');
  const [heartRate, setHeartRate] = useState(72);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      console.log('Yüklenen dosya:', file.name);
      // Dosya işleme mantığı buraya eklenecek
    }
  };

  return (
    <main className="dashboard">
      <div className="info-cards">
        <InfoBox title="Kalp Atış Hızı" value={heartRate} unit="BPM" />
        <InfoBox title="Durum" value={anomalyStatus} />
      </div>

      <ChartComponent />
      
      <div className="upload-section">
        <label htmlFor="ecg-upload" className="upload-button">
          Veri Yükle
        </label>
        <input 
          id="ecg-upload" 
          type="file" 
          accept=".csv,.txt"
          onChange={handleFileUpload} 
          style={{ display: 'none' }} 
        />
      </div>
    </main>
  );
}

export default Dashboard;