# Checkpoints Directory

File bobot model yang telah dilatih (`best_model.pth`) akan disimpan otomatis di folder ini ketika menjalankan script training:

```bash
python -m src.train --epochs 30 --batch_size 16 --device cuda
```

Struktur isi checkpoint:
- `epoch`: Epoch terbaik saat validasi loss terendah
- `model_state_dict`: Bobot neural network ResNet U-Net
- `optimizer_state_dict`: State optimizer AdamW
- `val_loss`: Nilai validation loss terendah
- `val_psnr`: PSNR rata-rata pada validation set
- `val_ssim`: SSIM rata-rata pada validation set

Aplikasi Streamlit (`app/app.py`) akan mendeteksi `checkpoints/best_model.pth` secara otomatis saat aplikasi dijalankan.
