import React from 'react';

function AboutPage() {
  return (
    <div className="page">
      <div className="page-header">
        <h1>Hakkında</h1>
      </div>
      <div className="page-content">
        <h2>Vizyonumuz</h2>
        <p>
          Medikal verilerin dijital ortamda güvenli, hızlı
          ve doğru bir şekilde yönetilmesini sağlayarak sağlık profesyonellerinin karar
          süreçlerini desteklemek ve hasta bakım kalitesini artırmaktır. Teknolojiyi insan
          sağlığı için dönüştürücü bir güç olarak kullanmayı hedefliyoruz.
        </p>

        <h2>Misyonumuz</h2>
        <p>
          EKG gibi kritik biyomedikal verileri analiz eden, kullanıcı dostu ve
          ölçeklenebilir bir platform sunmaktır. Hasta yönetimi, veri analizi ve yapay zeka
          destekli tanı sistemlerini bir araya getirerek sağlık çalışanlarına pratik çözümler
          üretmek temel amacımızdır. Açık, esnek ve entegre edilebilir bir altyapı ile sürekli
          gelişen sağlık teknolojilerine uyum sağlıyoruz.
        </p>
      </div>
    </div>
  );
}

export default AboutPage;
