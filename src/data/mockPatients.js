// Her hasta için farklı görünen, ön işlenmiş EKG verisi simülasyonu
const ekgDataAhmet = Array.from({ length: 188 }, (_, i) => (Math.sin(i / 10) * 0.5 + Math.sin(i / 5)) * 0.5);
const ekgDataFatma = Array.from({ length: 188 }, (_, i) => (Math.sin(i / 6) * 0.6 + Math.sin(i / 3)) * 0.5);
const ekgDataMehmet = Array.from({ length: 188 }, (_, i) => (Math.sin(i / 15) * 0.4 + (Math.random() - 0.5) * 0.1));

const mockPatients = [
    {
        id: 1,
        name: "Ahmet Yılmaz",
        tc: "12345678901",
        age: 45,
        gender: "E",
        phone: "0532-123-4567",
        // HASTA SEVİYESİNDE İLAÇ LİSTESİ
        medications: ["Aspirin 100mg", "Metoprolol 50mg"],
        ekgFiles: [
            {
                id: 'file1',
                name: 'ilk_kayit.txt',
                uploadedAt: '2025-06-28',
                data: ekgDataAhmet,
                // BU DOSYAYA ÖZEL ANALİZ VERİLERİ
                analysis: {
                    heartRate: 72,
                    qrsDuration: 98,
                    prInterval: 155,
                    qtInterval: 390
                },
                aiFindings: {
                    arrhythmia: 'Düşük',
                    heartAttackRisk: 'Düşük'
                }
            }
        ]
    },
    {
        id: 2,
        name: "Fatma Demir",
        tc: "98765432109",
        age: 62,
        gender: "K",
        phone: "0533-987-6543",
        medications: ["Lisinopril 10mg", "Atorvastatin 20mg", "Amlodipin 5mg"],
        ekgFiles: [
            {
                id: 'file2',
                name: 'tasikardi_ornek.csv',
                uploadedAt: '2025-06-29',
                data: ekgDataFatma,
                analysis: {
                    heartRate: 105,
                    qrsDuration: 110,
                    prInterval: 140,
                    qtInterval: 350
                },
                aiFindings: {
                    arrhythmia: 'Orta',
                    heartAttackRisk: 'Düşük'
                }
            }
        ]
    },
    {
        id: 3,
        name: "Mehmet Kaya",
        tc: "11223344556",
        age: 38,
        gender: "E",
        phone: "0534-555-1234",
        medications: [], // İlaç kullanmıyor
        ekgFiles: [
            {
                id: 'file3',
                name: 'duzensiz_ritim.dat',
                uploadedAt: '2025-07-01',
                data: ekgDataMehmet,
                analysis: {
                    heartRate: 65,
                    qrsDuration: 85,
                    prInterval: 190,
                    qtInterval: 410
                },
                aiFindings: {
                    arrhythmia: 'Yüksek',
                    heartAttackRisk: 'Çok Düşük'
                }
            }
        ]
    }
];

export default mockPatients;