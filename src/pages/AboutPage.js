import React from 'react';

function AboutPage() {
  return (
    <div className="page">
      <div className="page-header">
        <h1>LifeLine Hakkında</h1>
        <p>Modern Medikal Veri Yönetim Sistemi</p>
      </div>
      <div>
        <p>
          LifeLine, EKG (Elektrokardiyogram) verilerini işlemek, analiz etmek ve
          görselleştirmek için geliştirilmiş bir web tabanlı platformdur.
        </p>
        <br />
        <h3>Temel Özellikler:</h3>
        <ul>
          <li>
            <strong>Hasta Yönetimi:</strong> Hasta kayıtlarını kolayca ekleyin,
            arayın ve yönetin.
          </li>
          <li>
            <strong>EKG Veri Analizi:</strong> Yüklenen ham EKG verilerini işleyerek
            anomalileri tespit edin.
          </li>
          <li>
            <strong>Yapay Zeka Desteği:</strong> Makine öğrenmesi modelleri ile tanı
            destek sistemi sağlar. (Geliştirme Aşamasında)
          </li>
          <li>
            <strong>Güvenli ve Esnek:</strong> Gelecekte bir backend altyapısına
            kolayca entegre edilebilecek şekilde tasarlanmıştır.
          </li>
        </ul>
        <br />
        <p>
          Bu proje, React.js kullanılarak geliştirilmiştir ve medikal
          profesyonellerin hasta verilerine hızlı ve etkin bir şekilde
          ulaşmasını hedeflemektedir.
        </p>
      </div>
    </div>
  );
}

export default AboutPage;