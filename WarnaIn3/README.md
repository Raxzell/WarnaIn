# WarnaIn v3 — Multi-Paradigm Deep Learning Colorization Benchmark
### (CNN ResNet-18 U-Net vs GAN Pix2Pix vs Vision Transformer Swin-UNet)

Folder **`WarnaIn3/`** adalah paket *refactoring* mandiri (terisolasi penuh sehingga tidak mengganggu kode versi sebelumnya di root `d:\WarnaIn\`) untuk memenuhi tuntutan pengujian komparatif **LO3 (Analisis Perilaku Model)** dan **LO4 (Evaluasi Limitasi Pendekatan Deep Learning)**.

---

## 1. Struktur Folder `WarnaIn3/`

```text
WarnaIn3/
├── app/
│   ├── app.py                                      # Streamlit UI Koran Klasik v3 (3 Paradigma + Head-to-Head 4 Kolom)
│   └── components.py                               # Komponen UI (Before-After Slider, Kartu Insight LO3/LO4, CIE Lab Inspector)
├── src/
│   ├── model.py                                    # Factory 3 Arsitektur: ResNetUNetColorizer, Pix2PixColorizer, TransformerColorizer
│   ├── evaluate.py                                 # Skrip Benchmark Otomatis 500 Citra COCO -> CSV & Tabel Markdown Bab 4
│   ├── dataset.py                                  # Dataset Loader CIE L*a*b* [-1, 1] (otomatis baca ../data tanpa duplikasi)
│   └── utils.py                                    # Konversi CIE L*a*b* [-1, 1], PSNR, SSIM, MAE (a*b*)
├── notebooks/
│   └── WarnaIn_v3_Training_Benchmark.ipynb         # Notebook Training & Evaluasi Interaktif (Bisa di Lokal / Google Colab GPU)
├── reports/
│   ├── benchmark_summary.csv                       # Hasil kuantitatif (PSNR, SSIM, MAE, Latency)
│   └── benchmark_table_bab4.md                     # Tabel Markdown siap copy-paste ke Bab 4 Laporan Akademik
├── checkpoints/
│   └── README.md                                   # Tempat penyimpanan bobot hasil training (.pth)
└── streamlit_app.py                                # Entry point cepat untuk menjalankan aplikasi Streamlit v3
```

---

## 2. Cara Menjalankan Aplikasi Web Streamlit v3

Dari terminal di direktori utama `d:\WarnaIn`:

```powershell
streamlit run WarnaIn3/app/app.py
```
*(Atau `streamlit run WarnaIn3/streamlit_app.py`)*

### Fitur Utama di UI v3:
1. **Dropdown 3 Paradigma Resmi (Sidebar Studio Controls)**:
   * `CNN: ResNet-18 U-Net (Pixel Regression)`
   * `GAN: Pix2Pix cGAN (Adversarial Learning)`
   * `Transformer: Swin-UNet (Self-Attention)`
2. **Tab 2 — Mode "Head-to-Head 3 Model Comparison" (4 Kolom Side-by-Side)**:
   * **Kolom 1**: Input Grayscale ($L^* \in [-1, 1]$)
   * **Kolom 2**: Output CNN ResNet-18 U-Net + Metrik (`PSNR`, `SSIM`, `MAE a*b*`, `Latency ms`)
   * **Kolom 3**: Output GAN Pix2Pix + Metrik (`PSNR`, `SSIM`, `MAE a*b*`, `Latency ms`)
   * **Kolom 4**: Output Transformer Swin-UNet + Metrik (`PSNR`, `SSIM`, `MAE a*b*`, `Latency ms`)
   * Dilengkapi **Kartu Analisis LO3 & LO4** di bagian bawah yang mengulas perilaku dan limitasi tiap paradigma.
3. **Tab 3 — CIE L\*a\*b\* Space Inspector**:
   * Dekomposisi kanal $L^*$, $a^*$, $b^*$ serta plot sebaran gamut kromatisitas 2D.

---

## 3. Cara Melatih Model Menggunakan Jupyter Notebook (`.ipynb`)

Buka berkas:
```text
WarnaIn3/notebooks/WarnaIn_v3_Training_Benchmark.ipynb
```
di **VS Code** atau unggah ke **Google Colab / Kaggle (GPU T4/P100)**:
* Sel-sel training sudah dipisahkan per model (`train_cnn_resnet_unet`, `train_pix2pix_gan`, `train_transformer_swin`).
* Bobot terbaik otomatis disimpan ke `WarnaIn3/checkpoints/{resnet_unet,pix2pix,transformer}_best.pth`.

---

## 4. Cara Menjalankan Skrip Evaluasi Komparatif (`evaluate.py`)

Untuk menjalankan benchmark pada seluruh 500 citra Test Set COCO 2017:
```powershell
python WarnaIn3/src/evaluate.py --max_samples 500 --img_size 224
```
Hasil evaluasi akan langsung tercetak dalam format tabel Markdown di terminal serta disimpan otomatis ke:
* `WarnaIn3/reports/benchmark_summary.csv`
* `WarnaIn3/reports/benchmark_table_bab4.md`

---

## 5. Cara Memindahkan `WarnaIn3` ke Folder Utama (Jika Nanti Ingin Menggantikan Versi Lama)

Jika nantinya Anda ingin menjadikan versi 3 sebagai versi utama di root `d:\WarnaIn\`, cukup salin folder `app/`, `src/`, `notebooks/`, dan `reports/` dari dalam `WarnaIn3/` ke `d:\WarnaIn\`. Semua path sudah dirancang otomatis kompatibel baik saat berada di dalam subfolder `WarnaIn3/` maupun saat dipindahkan ke root!
