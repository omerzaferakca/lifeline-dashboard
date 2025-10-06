# LifeLine Dashboard

https://gevherpositive-lifeline.web.app/

EKG analiz sistemi için React tabanlı web dashboard'u.

## Kurulum

1. Projeyi klonlayın:
```bash
git clone https://github.com/omerzaferakca/lifeline-dashboard.git
cd lifeline-dashboard
```

2. Bağımlılıkları yükleyin:
```bash
npm install
```

3. Firebase konfigürasyonunu ayarlayın:
```bash
cp src/firebase/config.example.js src/firebase/config.js
```
Ardından `src/firebase/config.js` dosyasını editleyip kendi Firebase config değerlerinizi girin.

4. Geliştirme sunucusunu başlatın:
```bash
npm start
```

## Deployment

```bash
npm run build
firebase deploy --only hosting
```

## Özellikler

- ✅ EKG dosyası yükleme ve analiz
- ✅ Hasta yönetimi 
- ✅ Gerçek zamanlı analiz sonuçları
- ✅ Grafik görselleştirme
- ✅ Kullanıcı profil yönetimi
- ✅ Responsive tasarım

## Güvenlik

⚠️ **Önemli:** `src/firebase/config.js` dosyası API anahtarları içerir ve GitHub'a yüklenmemelidir. Benim yaptığımı yapmayın xd.
