# Tabel Komparasi Kuantitatif 3 Paradigma Deep Learning (Bab 4 — Implementation & Results)

**Dataset Evaluasi:** COCO 2017 Test Split (500 Citra) | **Resolusi Tensor Input:** `(B, 1, 224, 224)` dalam rentang `[-1, 1]`

| Keluarga Model | Paradigma Pembelajaran | PSNR (dB) ↑ | SSIM ↑ | MAE ($a^*b^*$) ↓ | Waktu Inferensi (ms/frame) ↓ | Analisis Perilaku & Limitasi (LO3 & LO4) |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **CNN: ResNet-18 U-Net** | Pixel Regression (Smooth L1 + MSE) | 24.23 ± 4.12 | 0.9229 ± 0.0633 | 8.01 | 115.51 ms | Struktur tepi tajam, warna konservatif/aman (regression-to-the-mean) |
| **GAN: Pix2Pix cGAN** | Adversarial Learning (U-Net + PatchGAN 70x70) | 22.76 ± 3.86 | 0.9034 ± 0.0726 | 9.15 | 115.36 ms | Warna vibrant & berani, kontras kromatik tinggi, risiko artefak minor |
| **Transformer: Swin-UNet** | Windowed Multi-Head Self-Attention (W-MSA) | 23.92 ± 3.94 | 0.9182 ± 0.0646 | 8.23 | 180.94 ms | Konsistensi semantik global terbaik & koherensi warna jarak jauh |
