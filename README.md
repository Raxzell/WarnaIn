# 🎨 WarnaIn: AI Image Colorization Web Studio

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg?style=flat&logo=streamlit)](https://streamlit.io)
[![Color Space](https://img.shields.io/badge/Color%20Space-CIE%20L*a*b*-4F46E5.svg?style=flat)]()
[![Dataset](https://img.shields.io/badge/Dataset-COCO%202017-blue.svg?style=flat)](https://cocodataset.org)

**WarnaIn** is an end-to-end Deep Learning and Computer Vision application designed to automatically transform black-and-white (grayscale), historical, and desaturated photos into natural, vibrant full-color images using **PyTorch** and an editorial broadsheet-styled **Streamlit** web studio.

---

## 🌟 Key Highlights

* **CIE $L^*a^*b^*$ Color Space Pipeline:** Decouples luminance ($L^*$) from chrominance ($a^*, b^*$), ensuring that 100% of photographic edges, textures, and fine structural details are preserved with zero contour distortion or blur.
* **6-Category Semantic Colorization System:** Supports `general` (all-domain) alongside 5 specialized semantic categories (`animals/`, `architecture/`, `landscapes/`, `portraits/`, and `objects/`) mapped from official COCO 2017 instance and caption annotations, coupled with aspect-ratio-preserving inference and category-aware CIE $L^*a^*b^*$ chrominance calibration.
* **Aspect-Ratio Aware Before–After Comparison Slider:** Smooth touch-and-drag split canvas powered by CSS `clip-path` with halftone print overlay and custom rectangular editorial badges.
* **Saturation Degradation Experimentation:** Real-time color restoration testing across user-controlled degradation levels (0% pure grayscale to 100% original saturation) with live quantitative evaluation:
  * **PSNR (Peak Signal-to-Noise Ratio):** Measures reconstruction fidelity in decibels (dB).
  * **SSIM (Structural Similarity Index):** Assesses visual structural preservation ($0 \to 1.0$).
  * **MAE (Mean Absolute Error):** Quantifies absolute per-pixel deviation.
* **Deep Learning CIE Lab Inspector:** Real-time channel deconstruction showing $L^*$ (Luminance), $a^*$ (Green–Red axis), $b^*$ (Blue–Yellow axis), alongside a 2D Chromaticity Gamut scatter plot.
* **Multiple AI Model Engines:**
  * **Ensemble Pro (Recommended):** Blends U-Net skip-connection boundary adherence with a 313-bin semantic chrominance prior.
  * **CIE Lab U-Net (Skip Connections):** Symmetric encoder-decoder with dilated bottleneck for smooth, natural gradients.
  * **Semantic Chrominance Prior (Vivid):** Gamut classification network for bold, vivid colors.
  * **Custom ResNet-18 U-Net:** Trainable architecture for custom domain datasets (COCO 2017).
* **Editorial Broadsheet Interface:** Bespoke vintage newspaper aesthetic inspired by *Old Standard TT* and *Plus Jakarta Sans* typography, double-ruled navigation, and tactile two-column editorial layouts.

---

## 🏛️ Architecture & Color Space Methodology

```text
Input RGB Image
      │
      ▼
[RGB to CIE Lab Conversion]
      │
      ├── Luminance Channel L* (1 channel) ─────────┐ (Preserved 100% Lossless)
      │   Range: [0 .. 100]                         │
      │        │                                    ▼
      │        ▼                           ┌──────────────────┐
      │  [Deep Neural Network] ──────────> │   U-Net Decoder  │
      │  (Encoder + Skip Connections)      │  (Upsampling &   │
      │                                    │ Skip Concat)     │
      │                                    └────────┬─────────┘
      │                                             │
      │                                             ▼
      │                                    Predicted a* & b* Channels (2 channels)
      │                                    Range: [-110 .. +110]
      │                                             │
      └─────────────────────────────────────────────┴───────┐
                                                            ▼
                                               [CIE Lab to RGB Reconstruction]
                                                            │
                                                            ▼
                                                   Final Colorized Image
```

### Why CIE Lab Instead of RGB?
Standard RGB color spaces entangle illumination brightness with color hues across all three channels. In contrast, **CIE $L^*a^*b^*$** mimics human visual perception:
1. **$L^*$ Channel (Luminance):** Contains 100% of photographic lighting, contrast, and edge details.
2. **$a^*$ & $b^*$ Channels (Chrominance):**
   * $a^* \in [-128, +127]$: Green $\longleftrightarrow$ Red spectrum.
   * $b^* \in [-128, +127]$: Blue $\longleftrightarrow$ Yellow spectrum.
3. **Core Advantage:** The neural network only needs to solve an under-constrained $1 \to 2$ channel mapping. Because the input grayscale $L^*$ channel is retained unmodified, the output image can **never become blurry, lose focus, or lose structural sharpness**.

---

## 📂 Repository Structure

```text
WarnaIn/
├── data/
│   ├── annotations/          # COCO 2017 instances & captions JSON annotations
│   ├── raw/                  # Raw COCO 2017 validation dataset (5,000 images)
│   └── splits/               # Categorized splits: train/ (80%), val/ (10%), test/ (10%)
│       └── {train,val,test}/{animals,architecture,landscapes,portraits,objects}/
├── src/
│   ├── __init__.py
│   ├── dataset.py            # PyTorch Dataset, CIE Lab transforms, augmentations, and split builder
│   ├── model.py              # SIGGRAPH17 U-Net, ECCV16 Prior, Ensemble, & ResNet-18 U-Net models
│   ├── train.py              # Training pipeline: Smooth L1 + MSE loss, early stopping, checkpointing
│   ├── evaluate.py           # Quantitative test-set benchmarking (PSNR, SSIM, MAE)
│   └── utils.py              # Float32 CIE Lab <-> RGB conversions, metrics, degradation utilities
├── app/
│   ├── __init__.py
│   ├── app.py                # Main Streamlit web application
│   └── components.py         # Aspect-ratio comparison slider, metric strips, CIE Lab inspector
├── checkpoints/              # Checkpoint directory (best_model.pth)
├── notebooks/
│   └── WarnaIn_Pipeline.ipynb# End-to-end Jupyter / Google Colab pipeline
├── assets/                   # Sample test images, HTML design spec, and benchmark examples
├── .streamlit/
│   └── config.toml           # Editorial broadsheet theme configuration
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites & Environment Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/Raxzell/WarnaIn.git
cd WarnaIn

# Optional: Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch the Web Studio
Launch the Streamlit studio locally:
```bash
streamlit run app/app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your web browser. The app runs immediately with pre-loaded samples or your own uploaded photos.

---

## 📊 Dataset & Model Training (Optional)

The baseline dataset is the **COCO 2017 Validation Set** (5,000 images, ~800 MB) paired with **COCO 2017 Train/Val Annotations**:
1. **Download and Extract:**
   * Download [COCO 2017 Val Images](http://images.cocodataset.org/zips/val2017.zip) and extract into `data/raw/val2017`.
   * Download [COCO 2017 Train/Val Annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) and extract `instances_val2017.json` and `captions_val2017.json` into `data/annotations/`.
2. **Split & Categorize Dataset:**
   Automatically classify all 5,000 images into `animals/`, `architecture/`, `landscapes/`, `portraits/`, and `objects/` across 80% Train, 10% Validation, and 10% Test splits:
   ```bash
   python -m src.dataset --raw_dir data/raw/val2017 --splits_dir data/splits --annotations_dir data/annotations
   ```
3. **Train Model (General or Category-Specific):**
   Train the ResNet U-Net model on all categories (`general`) or a specific semantic category (`animals`, `architecture`, `landscapes`, `portraits`, `objects`):
   ```bash
   # General training (all 4,000 train images -> checkpoints/best_model.pth)
   python -m src.train --category general --epochs 25 --batch_size 16 --device cuda

   # Category-specific training (e.g., animals -> checkpoints/best_model_animals.pth)
   python -m src.train --category animals --epochs 25 --batch_size 16 --device cuda
   ```
4. **Evaluate Test Set:**
   Benchmark quantitative performance across all test images or per category:
   ```bash
   python -m src.evaluate --model_path checkpoints/best_model.pth --test_dir data/splits/test --category general
   ```

---

## 📈 Quantitative Benchmark (COCO Test Set)

| Evaluation Metric | WarnaIn Score (500 Test Images) | Target Quality Standard |
| :--- | :--- | :--- |
| **SSIM (Structural Similarity)** | **0.9158 ± 0.066** | $> 0.85$ (Near-identical structural fidelity) |
| **PSNR (Peak Signal-to-Noise)** | **23.10 ± 5.12 dB** | $> 20.0 \text{ dB}$ (Stable chromatic reconstruction) |
| **MAE (Mean Absolute Error)** | **13.91 ± 8.64** | Lower is better (Low chromatic drift) |

---

## 📜 Citations & References

* Zhang, R., Isola, P., & Efros, A. A. (2016). *Colorful Image Colorization*. In European Conference on Computer Vision (ECCV).
* Zhang, R., Zhu, J. Y., Isola, P., Geng, X., Lin, A. S., Yu, T., & Efros, A. A. (2017). *Real-Time User-Guided Image Colorization with Learned Deep Priors*. ACM Transactions on Graphics (SIGGRAPH).
* Lin, T. Y., et al. (2014). *Microsoft COCO: Common Objects in Context*. In ECCV.

---

## 📄 License
This project is licensed under the MIT License.
