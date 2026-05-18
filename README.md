# SART Reconstruction Parameter Optimization with ANN

Bu proje, Bilgisayarlı Tomografi (CT) rekonstrüksiyon algoritması SART'ın (Simultaneous Algebraic Reconstruction Technique) parametre optimizasyonunu Yapay Sinir Ağları (ANN) kullanarak gerçekleştirmektedir.

## Proje Özeti

SART algoritmasının iki kritik parametresi olan **iterasyon sayısı** ve **relaxation faktörü**, görüntü kalitesi (PSNR, SSIM) ve hesaplama süresi üzerinde doğrudan etkilidir. Bu projede:

1. 3 farklı CT cihazı geometrisi (GE, Hologic, Siemens) için 390 SART kombinasyonu çalıştırıldı
2. Her kombinasyon için PSNR, SSIM ve süre metrikleri hesaplandı
3. İki farklı ANN modeli eğitildi (Scikit-learn ve PyTorch)
4. Kalite-süre dengesi optimize eden optimal parametreler bulundu

## Temel Bulgular

| Cihaz | Projeksiyon | En iyi denge | Süre kazancı |
|-------|-------------|--------------|--------------|
| GE | 9 | iter=65, relax=1.0 | — |
| Hologic | 15 | iter=95, relax=1.0 | — |
| Siemens | 25 | iter=16, relax=1.0 | **5x daha hızlı, %97 kalite** |

## Model Performansı

| Metrik | Scikit-learn | PyTorch (SARTNet) |
|--------|-------------|-------------------|
| PSNR R² | 0.965 | **0.981** |
| SSIM R² | 0.986 | **0.989** |
| Time R² | 0.936 | 0.941 |

## Dosya Yapısı

```
ct_ann/
├── sart_results.csv       # 390 kombinasyonluk dataset
├── evaluate_sart.py       # Metrik hesaplama (PSNR, SSIM, süre)
├── ann_model.py           # Scikit-learn MLPRegressor implementasyonu
├── ann_pytorch.py         # PyTorch SARTNet implementasyonu
├── requirements.txt       # Gerekli kütüphaneler
└── results/               # Grafik çıktıları
```

## Kurulum

```bash
pip install -r requirements.txt
```

PyTorch için (CPU):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Kullanım

### Metrikleri hesapla
```bash
python evaluate_sart.py
```
> **Not:** `evaluate_sart.py` çalıştırmak için `data/` klasöründe `phantom.npy` ve SART rekonstrüksiyon dosyaları (`.npy`) gereklidir. Bu dosyalar boyutları nedeniyle repoya dahil edilmemiştir. `ann_model.py` ve `ann_pytorch.py` için yalnızca `sart_results.csv` yeterlidir.

### Scikit-learn modeli
```bash
python ann_model.py
```

### PyTorch modeli (SARTNet)
```bash
python ann_pytorch.py
```

## Dataset

Dataset, [dbt-toolbox](https://github.com/dbt-toolbox) C++ kütüphanesi kullanılarak oluşturulmuştur.

**Parametreler:**
- **İterasyon:** 1, 2, 5, 10, 15, 20, 30, 50, 75, 100
- **Relaxation:** 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00
- **Scan presetleri:** ge_scan (9 proj), hologic_scan (15 proj), siemens_scan (25 proj)

## Model Mimarisi

### SARTNet (PyTorch)
```
Giriş (5) → Linear(128) → BatchNorm → ReLU → Dropout(0.2)
          → Linear(64)  → BatchNorm → ReLU → Dropout(0.2)
          → Linear(32)  → ReLU
          → Çıkış (2): PSNR + SSIM
```

### Optimizasyon Skoru
```
score = 0.45 × PSNR_norm + 0.45 × SSIM_norm − 0.10 × time_norm
```

## Gereksinimler

- Python 3.12+
- numpy, pandas, scikit-learn, matplotlib
- torch (CPU yeterli)
- scikit-image
