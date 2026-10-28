"""
WarnaIn — Photo Colorization Studio (Streamlit)
Implements the editorial broadsheet design from assets/warnain_design.html
powered by a PyTorch CIE L*a*b* Deep Learning inference engine.
"""

import os
import sys
from datetime import datetime

# Ensure project root and app directory are always in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

APP_DIR = os.path.abspath(os.path.dirname(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import io
import time
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import torch

from src.dataset import VALID_CATEGORIES, CATEGORY_SUBFOLDERS
from src.model import build_colorizer
from src.utils import (
    rgb_to_lab,
    lab_to_rgb,
    image_to_tensor,
    apply_category_chroma_prior,
    degrade_saturation,
    enhance_luminance_contrast,
    calculate_psnr,
    calculate_ssim,
    calculate_mae,
    simulate_colorization_preview,
)
try:
    from app.components import (
        render_before_after_slider,
        render_metric_cards,
        render_exhibition_gallery,
        render_cielab_inspector,
        render_chroma_scatter,
    )
except ImportError:
    from components import (
        render_before_after_slider,
        render_metric_cards,
        render_exhibition_gallery,
        render_cielab_inspector,
        render_chroma_scatter,
    )


# ==========================================
# 1. Page Configuration & Editorial CSS
# ==========================================
st.set_page_config(
    page_title="WarnaIn — Photo Colorization Studio",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Old+Standard+TT:ital,wght@0,400;0,700;1,400&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

      :root {
        --paper: #E8E7E2;
        --sheet: #F2F1ED;
        --ink: #000000;
        --soft: #4A4A47;
        --rule: #000000;
        --hair: #B9B8B2;
        --serif: 'Old Standard TT', Georgia, serif;
        --sans: 'Plus Jakarta Sans', system-ui, sans-serif;
      }

      /* Hide Streamlit Chrome & Heading Anchor Links */
      header[data-testid="stHeader"],
      [data-testid="stToolbar"],
      [data-testid="stDecoration"],
      [data-testid="collapsedControl"],
      [data-testid="stHeaderActionElements"],
      section[data-testid="stSidebar"],
      #MainMenu,
      footer {
        display: none !important;
      }

      /* Outer Paper & Inner Sheet (.page) */
      .stApp {
        background-color: var(--paper) !important;
        color: var(--ink) !important;
        font-family: var(--sans) !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
      }

      .block-container {
        max-width: 1180px !important;
        margin: 0 auto !important;
        padding: 24px 28px 48px !important;
        background-color: var(--sheet) !important;
        min-height: 100vh !important;
        border-left: 1px solid var(--hair) !important;
        border-right: 1px solid var(--hair) !important;
      }

      /* Dateline & Masthead */
      .dateline {
        display: flex;
        justify-content: flex-start;
        gap: 12px;
        flex-wrap: wrap;
        font-size: 13px;
        color: var(--soft);
        padding-bottom: 8px;
        font-family: var(--sans);
      }
      .masthead {
        text-align: center;
        border-top: 4px solid var(--rule);
        border-bottom: 1px solid var(--rule);
        padding: 10px 0 6px;
      }
      .masthead h1 {
        font-family: var(--serif) !important;
        font-weight: 700 !important;
        font-size: clamp(3.2rem, 10vw, 6.5rem) !important;
        line-height: 1 !important;
        letter-spacing: -0.02em !important;
        color: var(--ink) !important;
        margin: 0 !important;
        padding: 0 !important;
      }
      .masthead p {
        font-family: var(--serif) !important;
        font-style: italic !important;
        font-size: 1.05rem !important;
        color: var(--soft) !important;
        padding-bottom: 4px !important;
        margin: 0 !important;
      }

      /* Double-ruled Navigation Bar (st.tabs) */
      .stTabs [data-baseweb="tab-list"] {
        display: flex !important;
        justify-content: center !important;
        gap: 0 !important;
        border-bottom: 4px double var(--rule) !important;
        margin-bottom: 26px !important;
        background: transparent !important;
        padding: 0 !important;
        border-radius: 0 !important;
      }
      .stTabs [data-baseweb="tab"] {
        padding: 10px 22px !important;
        font-family: var(--sans) !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        color: var(--soft) !important;
        border-left: 1px solid var(--hair) !important;
        border-radius: 0 !important;
        background: transparent !important;
        height: auto !important;
        margin: 0 !important;
      }
      .stTabs [data-baseweb="tab"]:first-child {
        border-left: 0 !important;
      }
      .stTabs [aria-selected="true"] {
        color: var(--ink) !important;
        background: var(--paper) !important;
        box-shadow: none !important;
      }
      .stTabs [data-baseweb="tab-highlight"],
      .stTabs [data-baseweb="tab-border"] {
        display: none !important;
      }

      /* Left Aside Column Border */
      div[data-testid="stTabPanel"] > div > div > div > div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="column"]:first-child {
        border-right: 1px solid var(--rule);
        padding-right: 24px !important;
      }
      div[data-testid="stTabPanel"] > div > div > div > div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="column"]:nth-child(2) {
        padding-left: 16px !important;
      }

      /* Editorial Section Headings */
      .aside-h2 {
        font-family: var(--serif) !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        color: var(--ink) !important;
        border-bottom: 1px solid var(--rule);
        padding-bottom: 4px;
        margin: 0 0 12px 0 !important;
      }
      .aside-h2-sub {
        font-family: var(--serif) !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: var(--ink) !important;
        border-bottom: 1px solid var(--rule);
        padding-bottom: 4px;
        margin: 22px 0 12px 0 !important;
      }
      .serif-sub {
        font-family: var(--serif) !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: var(--ink) !important;
        border-bottom: 1px solid var(--rule);
        padding-bottom: 4px;
        margin: 0 0 10px 0 !important;
      }
      .plate-cap {
        font-family: var(--serif);
        font-style: italic;
        font-size: 0.92rem;
        color: var(--soft);
        margin-top: 6px;
      }

      /* Lead Headline & Deck */
      .lead-h2 {
        font-family: var(--serif) !important;
        font-size: clamp(2rem, 4.4vw, 3.2rem) !important;
        line-height: 1.08 !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        color: var(--ink) !important;
        margin: 0 !important;
      }
      .lead-deck {
        font-family: var(--serif) !important;
        font-style: italic !important;
        font-size: 1.15rem !important;
        color: var(--soft) !important;
        margin: 8px 0 16px 0 !important;
        max-width: 64ch !important;
      }

      /* Stats Strip (.stats) */
      .stats {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        border-top: 1px solid var(--rule);
        border-bottom: 1px solid var(--rule);
        margin: 18px 0 22px 0;
      }
      .stat {
        padding: 12px 14px;
        border-left: 1px solid var(--hair);
      }
      .stat:first-child {
        border-left: 0;
        padding-left: 0;
      }
      .stat b {
        display: block;
        font-family: var(--serif);
        font-size: 2rem;
        line-height: 1.1;
        color: var(--ink);
      }
      .stat span {
        font-size: 12.5px;
        color: var(--soft);
        font-family: var(--sans);
      }

      /* Two-Column Editorial Text (.cols) */
      .cols {
        display: grid;
        grid-template-columns: 1fr 1fr;
        column-gap: 0;
        margin-top: 6px;
      }
      .col {
        padding-right: 24px;
      }
      .col + .col {
        padding: 0 0 0 24px;
        border-left: 1px solid var(--hair);
      }
      .col h2 {
        font-family: var(--serif) !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        border-bottom: 1px solid var(--rule);
        padding-bottom: 4px;
        margin-bottom: 14px !important;
        color: var(--ink) !important;
      }
      .col p, .col li {
        font-family: var(--serif);
        font-size: 1.02rem;
        line-height: 1.7;
        max-width: 62ch;
        color: var(--ink);
      }
      .col p + p {
        margin-top: 10px;
      }
      .col p.dropcap:first-letter {
        font-size: 3.4rem;
        float: left;
        line-height: 0.85;
        padding: 4px 8px 0 0;
        font-weight: 700;
        font-family: var(--serif);
      }
      .col ul {
        padding-left: 18px;
      }

      /* Editorial Exhibition Plate Gallery (.exhibition-section) */
      .exhibition-section {
        margin-top: 38px;
        border-top: 4px double var(--rule);
        padding-top: 18px;
      }
      .exhibition-header {
        text-align: center;
        border-bottom: 1px solid var(--rule);
        padding-bottom: 12px;
        margin-bottom: 20px;
      }
      .exhibition-kicker {
        display: block;
        font-family: var(--sans);
        font-size: 11.5px;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--soft);
        margin-bottom: 4px;
      }
      .exhibition-h2 {
        font-family: var(--serif) !important;
        font-size: clamp(1.65rem, 3.2vw, 2.35rem) !important;
        font-weight: 700 !important;
        color: var(--ink) !important;
        line-height: 1.12 !important;
        margin: 0 !important;
      }
      .exhibition-deck {
        font-family: var(--serif) !important;
        font-style: italic !important;
        font-size: 1.02rem !important;
        color: var(--soft) !important;
        margin: 6px auto 0 !important;
        max-width: 66ch;
      }
      .exhibition-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0;
      }
      .ex-card {
        padding: 4px 18px 8px 0;
      }
      .ex-card-divider {
        padding: 4px 18px 8px 18px;
        border-left: 1px solid var(--hair);
      }
      .ex-card-divider:nth-child(3n) {
        padding-right: 0;
      }
      .ex-card-row2 {
        margin-top: 16px;
        padding-top: 16px;
        border-top: 1px solid var(--hair);
      }
      .ex-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid var(--hair);
        padding-bottom: 4px;
        margin-bottom: 6px;
      }
      .ex-kicker {
        font-family: var(--sans);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: var(--ink);
      }
      .ex-cat {
        font-family: var(--serif);
        font-style: italic;
        font-size: 12.5px;
        color: var(--soft);
      }
      .ex-title {
        font-family: var(--serif) !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: var(--ink) !important;
        margin: 0 0 10px 0 !important;
      }
      .ex-pair {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
      }
      .ex-frame {
        position: relative;
        border: 1px solid var(--ink);
        background: var(--paper);
        aspect-ratio: 4 / 3;
        overflow: hidden;
      }
      .ex-frame img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
      }
      .ex-bw img {
        filter: grayscale(1) contrast(1.05);
      }
      .ex-bw:after {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle, rgba(0,0,0,.14) 0.6px, transparent 1px) 0 0 / 4px 4px;
        mix-blend-mode: multiply;
        pointer-events: none;
      }
      .ex-tag {
        position: absolute;
        bottom: 6px;
        left: 6px;
        padding: 1px 7px;
        background: var(--sheet);
        border: 1px solid var(--ink);
        font-family: var(--sans);
        font-size: 10.5px;
        font-weight: 600;
        color: var(--ink);
        z-index: 2;
      }
      .ex-cap {
        font-family: var(--serif) !important;
        font-style: italic !important;
        font-size: 0.9rem !important;
        line-height: 1.45 !important;
        color: var(--soft) !important;
        margin-top: 8px !important;
        margin-bottom: 0 !important;
      }
      @media (max-width: 900px) {
        .exhibition-grid {
          grid-template-columns: 1fr;
          gap: 18px;
        }
        .ex-card, .ex-card-divider {
          padding: 0 0 14px 0;
          border-left: 0;
          border-bottom: 1px solid var(--hair);
        }
      }

      /* Footer (.foot) */
      .foot {
        margin-top: 34px;
        border-top: 4px double var(--rule);
        padding-top: 8px;
        font-size: 12.5px;
        color: var(--soft);
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
        font-family: var(--sans);
      }

      /* Form Widgets Styling to match warnain_design.html */
      [data-testid="stFileUploaderDropzone"] {
        border: 1.5px dashed var(--ink) !important;
        background-color: var(--paper) !important;
        border-radius: 0 !important;
        padding: 16px 12px !important;
      }
      [data-testid="stFileUploaderDropzone"]:hover {
        background-color: #DEDDD7 !important;
      }
      div[data-baseweb="select"] > div {
        border: 1px solid var(--ink) !important;
        background-color: var(--sheet) !important;
        border-radius: 0 !important;
        font-size: 13.5px !important;
      }
      .stDownloadButton > button,
      .stButton > button[kind="primary"] {
        width: 100% !important;
        padding: 11px 14px !important;
        background-color: var(--ink) !important;
        color: var(--sheet) !important;
        font: 600 14px var(--sans) !important;
        border: 0 !important;
        border-radius: 0 !important;
        cursor: pointer !important;
      }
      .stDownloadButton > button:hover,
      .stButton > button[kind="primary"]:hover {
        background-color: #333333 !important;
        color: var(--sheet) !important;
      }
      .stButton > button[kind="secondary"] {
        width: 100% !important;
        padding: 8px 12px !important;
        background-color: var(--paper) !important;
        color: var(--ink) !important;
        font: 600 13px var(--sans) !important;
        border: 1px solid var(--ink) !important;
        border-radius: 0 !important;
      }
      .stSlider label p,
      .stSelectbox label p,
      .stCheckbox label p {
        font-family: var(--sans) !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
        color: var(--ink) !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 2. Cached Model & Inference Pipeline
# ==========================================
@st.cache_resource(show_spinner=False)
def get_colorizer_model(engine_key: str, checkpoint_path: str, device: str):
    try:
        model = build_colorizer(checkpoint_path=checkpoint_path, device=device, engine=engine_key)
        return model, True
    except Exception as e:
        print(f"[WARNING] Failed loading engine {engine_key}: {e}")
        return None, False


def colorize_image(
    model,
    rgb_or_gray: np.ndarray,
    target_size: tuple,
    vibrancy: float,
    warmth_deg: float,
    apply_clahe: bool,
    device: str,
    category: str = "general",
) -> tuple:
    """
    Full-resolution CIE Lab colorization with aspect-ratio preservation
    and semantic category chrominance calibration:
    - Extracts full-resolution L* channel from input.
    - Predicts a* and b* preserving aspect ratio around target_size.
    - Upsamples predicted a* and b* to original image dimensions, applies
      category chrominance prior, and merges with full-res L*.
    """
    start_time = time.time()
    working_img = enhance_luminance_contrast(rgb_or_gray) if apply_clahe else rgb_or_gray
    orig_h, orig_w = working_img.shape[:2]
    warmth = warmth_deg / 100.0  # -30..+30 -> -0.30..+0.30

    if model is not None:
        L_orig, _ = rgb_to_lab(working_img)
        L_tensor, _, _ = image_to_tensor(
            working_img,
            target_size=target_size,
            preserve_aspect=True,
        )
        L_tensor = L_tensor.to(device)

        with torch.no_grad():
            pred_ab = model(L_tensor)

        pred_ab_np = pred_ab.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        ab_full = cv2.resize(pred_ab_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        ab_calibrated = apply_category_chroma_prior(L_orig, ab_full, category=category)

        a_chan = ab_calibrated[:, :, 0] * vibrancy + (warmth * 0.12)
        b_chan = ab_calibrated[:, :, 1] * vibrancy + (warmth * 0.14)
        ab_adjusted = np.clip(np.stack([a_chan, b_chan], axis=2), -1.0, 1.0).astype(np.float32)

        final_rgb = lab_to_rgb(L_orig, ab_adjusted)
        latency = (time.time() - start_time) * 1000.0
        return final_rgb, latency
    else:
        colorized = simulate_colorization_preview(working_img, warmth=warmth, vibrancy=vibrancy)
        latency = (time.time() - start_time) * 1000.0
        return colorized, latency


@st.cache_data(show_spinner=False)
def build_exhibition_plates(sample_dir_path: str, checkpoint_path: str, device_str: str) -> list:
    """
    Precomputes and caches the 6 archival exhibition plates so the bottom exhibition gallery
    renders instantaneously.
    """
    specs = [
        {
            "plate_no": "PLATE I",
            "title": "Archival Studio Portrait",
            "filename": "vintage_portrait.jpg",
            "category_key": "portraits",
            "category_label": "Human Portraits",
            "caption": "Restored with the Human Portraits chrominance prior, preserving natural skin-tone warmth and subtle shadow gradation.",
        },
        {
            "plate_no": "PLATE II",
            "title": "Classical Stone Facade",
            "filename": "historic_building.jpg",
            "category_key": "architecture",
            "category_label": "Architecture & Landmarks",
            "caption": "Calibrated with the Architecture prior, separating warm masonry and timber midtones from crisp cerulean sky.",
        },
        {
            "plate_no": "PLATE III",
            "title": "Alpine & Coastal Vista",
            "filename": "nature_landscape.jpg",
            "category_key": "landscapes",
            "category_label": "Nature & Landscapes",
            "caption": "Synthesized under the Landscapes prior, enhancing organic foliage greens and atmospheric horizon depth.",
        },
        {
            "plate_no": "PLATE IV",
            "title": "Schoolchildren in Sukabumi",
            "filename": "vintage_bw_13_murid_sekolah_sukabumi.jpg",
            "category_key": "portraits",
            "category_label": "Human Portraits",
            "caption": "Archival group portrait from Sukabumi restored with balanced facial warmth and natural colonial-era attire tones.",
        },
        {
            "plate_no": "PLATE V",
            "title": "Pasar Senen, Old Batavia",
            "filename": "vintage_bw_01_pasar_senen_batavia.jpg",
            "category_key": "architecture",
            "category_label": "Architecture & Landmarks",
            "caption": "Historic Pasar Senen streetscape in Batavia, recovering warm terracotta roofing, timber market stalls, and tropical daylight.",
        },
        {
            "plate_no": "PLATE VI",
            "title": "Rice Terraces & Volcano Backdrop",
            "filename": "vintage_bw_12_sawah_latar_gunung_berapi.jpg",
            "category_key": "landscapes",
            "category_label": "Nature & Landscapes",
            "caption": "Javanese paddy fields beneath a volcanic peak, calibrated for lush emerald terraces and distant mountain haze.",
        },
    ]

    plates_out = []
    for sp in specs:
        fpath = os.path.join(sample_dir_path, sp["filename"])
        if not os.path.exists(fpath):
            continue
        rgb_src = np.array(Image.open(fpath).convert("RGB"))
        gray_2d = cv2.cvtColor(rgb_src, cv2.COLOR_RGB2GRAY)
        bw_rgb = cv2.cvtColor(gray_2d, cv2.COLOR_GRAY2RGB)

        base_name, _ = os.path.splitext(sp["filename"])
        col_path = os.path.join(sample_dir_path, f"{base_name}_colorized.jpg")
        if os.path.exists(col_path):
            col_rgb = np.array(Image.open(col_path).convert("RGB"))
        else:
            col_rgb = rgb_src
        plates_out.append({
            "plate_no": sp["plate_no"],
            "title": sp["title"],
            "category": sp["category_label"],
            "before_rgb": bw_rgb,
            "after_rgb": col_rgb,
            "caption": sp["caption"],
        })
    return plates_out


# ==========================================
# 3. Dateline & Masthead Header
# ==========================================
has_cuda = torch.cuda.is_available()
device_opt = "cuda" if has_cuda else "cpu"
checkpoint_default = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")
current_date_str = datetime.now().strftime("%A, %B %d, %Y")

st.markdown(
    f"""
    <div class="dateline">
      <span>{current_date_str}</span>
    </div>
    <header class="masthead">
      <h1>WarnaIn</h1>
      <p>A deep learning broadsheet for monochrome &amp; archival photo colorization</p>
    </header>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 4. Navigation Bar (Double-Ruled Broadsheet Tabs)
# ==========================================
tab_studio, tab_exp, tab_inspect = st.tabs([
    "Colorization Studio",
    "Degradation & Metrics Lab",
    "CIE L*a*b* Space Inspector",
])

# Discover Sample & Categorized COCO Dataset Paths
sample_dir = os.path.join(PROJECT_ROOT, "assets", "sample_images")
splits_test_dir = os.path.join(PROJECT_ROOT, "data", "splits", "test")
raw_val_dir = os.path.join(PROJECT_ROOT, "data", "raw", "val2017")
coco_test_dir = splits_test_dir if os.path.exists(splits_test_dir) else raw_val_dir


def get_coco_images_by_category(base_dir: str, category: str = "general") -> list:
    if not os.path.exists(base_dir):
        return []
    valid_exts = (".jpg", ".jpeg", ".png")
    if category != "general":
        cat_dir = os.path.join(base_dir, category)
        if os.path.isdir(cat_dir):
            return sorted([
                os.path.join(cat_dir, f)
                for f in os.listdir(cat_dir)
                if f.lower().endswith(valid_exts)
            ])
    collected = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(valid_exts):
                collected.append(os.path.join(root, f))
    return sorted(collected)


coco_images_all = get_coco_images_by_category(coco_test_dir, "general")

CATEGORY_LABELS = {
    "General (All Domains)": "general",
    "Animals & Wildlife": "animals",
    "Architecture & Landmarks": "architecture",
    "Nature & Landscapes": "landscapes",
    "Human Portraits": "portraits",
    "Everyday Objects & Products": "objects",
}
CATEGORY_DISPLAY_NAMES = {v: k for k, v in CATEGORY_LABELS.items()}


# ==========================================
# TAB 1: Colorization Studio (Exact 270px Aside + Main Lead)
# ==========================================
with tab_studio:
    col_aside, col_lead = st.columns([270, 850], gap="medium")

    with col_aside:
        st.markdown('<h2 class="aside-h2">Upload Photograph</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag and drop a photograph here, or browse files (JPG, PNG, or WEBP)",
            type=["png", "jpg", "jpeg", "webp"],
            key="main_uploader",
            label_visibility="collapsed",
        )

        chosen_cat_label = st.selectbox(
            "Subject Category",
            options=list(CATEGORY_LABELS.keys()),
            index=0,
            help="Select 'General (All Domains)' for unconstrained inference, or choose a specialized semantic domain to calibrate CIE L*a*b* chrominance priors.",
        )
        selected_category = CATEGORY_LABELS[chosen_cat_label]
        coco_images = get_coco_images_by_category(coco_test_dir, selected_category)
        if not coco_images:
            coco_images = coco_images_all

        st.markdown(
            '<div style="margin-top:6px;margin-bottom:8px;font-size:13px;color:var(--soft)">No photograph at hand? Try an archival plate or the COCO benchmark set.</div>',
            unsafe_allow_html=True,
        )

        fallback_options = {
            "Sample: Nature & Landscape": os.path.join(sample_dir, "nature_landscape.jpg"),
            "Sample: Historic Architecture": os.path.join(sample_dir, "historic_building.jpg"),
            "Sample: Vintage Portrait": os.path.join(sample_dir, "vintage_portrait.jpg"),
            "Archive: Schoolchildren in Sukabumi": os.path.join(sample_dir, "vintage_bw_13_murid_sekolah_sukabumi.jpg"),
            "Archive: Pasar Senen, Old Batavia": os.path.join(sample_dir, "vintage_bw_01_pasar_senen_batavia.jpg"),
            "Archive: Rice Terraces & Volcano": os.path.join(sample_dir, "vintage_bw_12_sawah_latar_gunung_berapi.jpg"),
        }
        if coco_images:
            cat_tag = "All" if selected_category == "general" else selected_category.capitalize()
            fallback_options[f"COCO 2017 Dataset ({cat_tag} · {len(coco_images)} Plates)"] = "COCO_RANDOM"

        selected_fallback = st.selectbox(
            "Sample Collection",
            options=list(fallback_options.keys()),
            label_visibility="collapsed",
        )

        if fallback_options[selected_fallback] == "COCO_RANDOM":
            if "coco_idx" not in st.session_state:
                st.session_state.coco_idx = 18
            if st.button("Draw Another COCO Plate", type="secondary"):
                st.session_state.coco_idx = int(np.random.randint(0, len(coco_images)))

        st.markdown('<h2 class="aside-h2-sub">Studio Controls</h2>', unsafe_allow_html=True)

        engine_map = {
            "Ensemble Pro (U-Net + Semantic Prior)": "ensemble",
            "CIE Lab U-Net": "siggraph17",
            "Semantic Chrominance Prior": "eccv16",
        }
        chosen_engine_label = st.selectbox("Model Architecture", options=list(engine_map.keys()), index=0)
        engine_key = engine_map[chosen_engine_label]

        vibrancy = st.slider(
            "Chroma Vibrancy",
            min_value=0.60,
            max_value=1.80,
            value=1.05,
            step=0.05,
        )
        warmth_deg = st.slider(
            "Color Temperature",
            min_value=-30,
            max_value=30,
            value=0,
            step=2,
            help="0 = neutral, negative = cooler, positive = warmer",
        )
        apply_clahe = st.checkbox("Enhance Luminance Contrast (CLAHE)", value=False)

        res_step = st.select_slider(
            "Processing Resolution",
            options=["256 px", "384 px", "512 px"],
            value="256 px",
        )
        res_lookup = {"256 px": (256, 256), "384 px": (384, 384), "512 px": (512, 512)}
        target_res = res_lookup[res_step]

        # Resolve active image
        active_rgb = None
        caption_text = "Archival demonstration plate. Upload your own photograph in the left column to inspect live restoration."
        if uploaded_file is not None:
            active_rgb = np.array(Image.open(uploaded_file).convert("RGB"))
            caption_text = (
                f"Uploaded plate: {uploaded_file.name} ({active_rgb.shape[1]}×{active_rgb.shape[0]} px) "
                f"· Category: {chosen_cat_label}."
            )
        elif fallback_options[selected_fallback] == "COCO_RANDOM" and coco_images:
            c_path = coco_images[st.session_state.coco_idx % len(coco_images)]
            active_rgb = np.array(Image.open(c_path).convert("RGB"))
            rel_cat = os.path.basename(os.path.dirname(c_path))
            caption_text = (
                f"COCO 2017 benchmark plate: {os.path.basename(c_path)} "
                f"(Split folder: {rel_cat} · Mode: {chosen_cat_label})."
            )
        else:
            f_path = fallback_options[selected_fallback]
            if os.path.exists(f_path):
                active_rgb = np.array(Image.open(f_path).convert("RGB"))
                caption_text = f"{selected_fallback} · Category: {chosen_cat_label}."

        # Resolve category-specific checkpoint if trained, fallback to default checkpoint
        cat_ckpt = os.path.join(PROJECT_ROOT, "checkpoints", f"best_model_{selected_category}.pth")
        active_ckpt = cat_ckpt if (selected_category != "general" and os.path.exists(cat_ckpt)) else checkpoint_default

        # Run model inference
        model, _ = get_colorizer_model(engine_key, active_ckpt, device_opt)
        if active_rgb is not None:
            gray_2d = cv2.cvtColor(active_rgb, cv2.COLOR_RGB2GRAY)
            input_gray_rgb = cv2.cvtColor(gray_2d, cv2.COLOR_GRAY2RGB)

            colorized_rgb, latency_ms = colorize_image(
                model=model,
                rgb_or_gray=input_gray_rgb,
                target_size=target_res,
                vibrancy=vibrancy,
                warmth_deg=warmth_deg,
                apply_clahe=apply_clahe,
                device=device_opt,
                category=selected_category,
            )

            st.session_state["last_colorized_rgb"] = colorized_rgb
            st.session_state["last_active_rgb"] = active_rgb

            buf = io.BytesIO()
            Image.fromarray(colorized_rgb).save(buf, format="PNG")
            st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
            st.download_button(
                label="Download Colorized Plate (PNG)",
                data=buf.getvalue(),
                file_name=f"WarnaIn_{int(time.time())}.png",
                mime="image/png",
                use_container_width=True,
            )

    with col_lead:
        st.markdown('<h2 class="lead-h2">Vintage Frames, Reimagined Hue</h2>', unsafe_allow_html=True)
        st.markdown(
            '<p class="lead-deck">Slide the dividing rule to compare the monochrome original against the neural reconstruction. Fine edge geometry remains untouched—only the chromatic channels are synthesized.</p>',
            unsafe_allow_html=True,
        )

        if active_rgb is not None:
            render_before_after_slider(
                img_before=input_gray_rgb,
                img_after=colorized_rgb,
                label_before="Monochrome",
                label_after="WarnaIn Plate",
                caption=caption_text,
                max_height=490,
            )

            psnr_val = calculate_psnr(active_rgb, colorized_rgb)
            ssim_val = calculate_ssim(active_rgb, colorized_rgb)
            mae_val = calculate_mae(active_rgb, colorized_rgb)
            render_metric_cards(psnr=psnr_val, ssim=ssim_val, mae=mae_val, latency_ms=latency_ms)

        st.markdown(
            """
            <div class="cols">
              <section class="col">
                <h2>Why CIE L*a*b*?</h2>
                <p class="dropcap">Standard RGB entangles luminance and color across all three channels. CIE L*a*b* decouples them cleanly, preserving the full monochrome structure while the neural network predicts only chromaticity.</p>
                <p>The L* channel retains original edges, film grain, and shadow depth, whereas the a* and b* axes encode the green–red and blue–yellow chrominance planes.</p>
              </section>
              <section class="col">
                <h2>How the Pipeline Works</h2>
                <ul>
                  <li>Full-resolution L* luminance is extracted directly from the source photograph.</li>
                  <li>The deep U-Net predicts a* and b* chrominance at the selected processing scale.</li>
                  <li>Predicted chrominance is upsampled, calibrated by semantic category priors, and fused with the original L* channel.</li>
                </ul>
              </section>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Bottom Editorial Exhibition Plate Gallery
    exhibition_plates = build_exhibition_plates(sample_dir, checkpoint_default, device_opt)
    render_exhibition_gallery(exhibition_plates)


# ==========================================
# TAB 2: Degradation & Metrics Lab
# ==========================================
with tab_exp:
    col_exp_aside, col_exp_main = st.columns([270, 850], gap="medium")

    with col_exp_aside:
        st.markdown('<h2 class="aside-h2">Degradation Test</h2>', unsafe_allow_html=True)
        exp_file = st.file_uploader(
            "Upload Ground-Truth Color Photograph",
            type=["png", "jpg", "jpeg", "webp"],
            key="exp_uploader",
            label_visibility="collapsed",
        )
        st.markdown(
            '<div style="margin-top:6px;margin-bottom:14px;font-size:13px;color:var(--soft)">Defaults to the active Studio photograph when no external file is uploaded.</div>',
            unsafe_allow_html=True,
        )

        sat_pct = st.slider(
            "Input Saturation (%)",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            help="0% = Pure monochrome, 25% = Faded archival print, 100% = Original color.",
        )

        exp_view_mode = st.selectbox(
            "Comparison Layout",
            options=[
                "Interactive Split Comparison",
                "Triptych Plates (Ground Truth · Degraded · Restored)",
            ],
        )

    with col_exp_main:
        st.markdown('<h2 class="lead-h2">Degradation &amp; Quantitative Metrics</h2>', unsafe_allow_html=True)
        st.markdown(
            '<p class="lead-deck">Simulate archival pigment fading over time and evaluate neural color recovery against the ground-truth photograph via PSNR, SSIM, and MAE.</p>',
            unsafe_allow_html=True,
        )

        gt_rgb = (
            np.array(Image.open(exp_file).convert("RGB"))
            if exp_file is not None
            else st.session_state.get("last_active_rgb", active_rgb)
        )

        if gt_rgb is not None:
            degraded_rgb = degrade_saturation(gt_rgb, saturation_factor=sat_pct / 100.0)
            restored_rgb, exp_lat = colorize_image(
                model=model,
                rgb_or_gray=degraded_rgb,
                target_size=target_res,
                vibrancy=vibrancy,
                warmth_deg=warmth_deg,
                apply_clahe=apply_clahe,
                device=device_opt,
                category=selected_category,
            )

            if exp_view_mode == "Interactive Split Comparison":
                render_before_after_slider(
                    img_before=degraded_rgb,
                    img_after=restored_rgb,
                    label_before=f"Saturation {sat_pct}%",
                    label_after="WarnaIn Restoration",
                    caption=f"Comparison between the degraded input ({sat_pct}% saturation) and the neural restoration.",
                    max_height=480,
                )
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown('<h2 class="serif-sub">1. Ground Truth</h2>', unsafe_allow_html=True)
                    st.image(gt_rgb, use_container_width=True)
                with c2:
                    st.markdown(f'<h2 class="serif-sub">2. Degraded ({sat_pct}%)</h2>', unsafe_allow_html=True)
                    st.image(degraded_rgb, use_container_width=True)
                with c3:
                    st.markdown('<h2 class="serif-sub">3. WarnaIn Plate</h2>', unsafe_allow_html=True)
                    st.image(restored_rgb, use_container_width=True)

            render_metric_cards(
                psnr=calculate_psnr(gt_rgb, restored_rgb),
                ssim=calculate_ssim(gt_rgb, restored_rgb),
                mae=calculate_mae(gt_rgb, restored_rgb),
                latency_ms=exp_lat,
            )


# ==========================================
# TAB 3: CIE L*a*b* Space Inspector
# ==========================================
with tab_inspect:
    st.markdown('<h2 class="lead-h2">CIE L*a*b* Color Space Inspector</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="lead-deck">Deconstruct the colorized plate into its perceptual lightness channel (L*) and two opponent chromatic axes: green–red (a*) and blue–yellow (b*).</p>',
        unsafe_allow_html=True,
    )

    inspected_rgb = st.session_state.get("last_colorized_rgb", None)
    if inspected_rgb is not None:
        render_cielab_inspector(inspected_rgb)
        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

        c_scat, c_notes = st.columns([1.1, 1], gap="large")
        with c_scat:
            render_chroma_scatter(inspected_rgb)
        with c_notes:
            st.markdown(
                """
                <div class="col" style="padding-right:0;">
                  <h2>Network Architecture</h2>
                  <p class="dropcap">Our architecture separates structural preservation from semantic color inference. Because the L* lightness channel bypasses neural modification, the output never suffers from spatial blurring or loss of photographic grain.</p>
                  <p>Multi-scale <em>skip connections</em> across the U-Net encoder and decoder prevent chrominance bleeding across object boundaries.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==========================================
# 5. Double-Ruled Broadsheet Footer
# ==========================================
st.markdown(
    """
    <div class="foot">
      <span>WarnaIn — Deep Learning Photo Colorization</span>
    </div>
    """,
    unsafe_allow_html=True,
)
