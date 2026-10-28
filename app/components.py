"""
WarnaIn - Editorial Broadsheet UI Components for Streamlit
Implements the visual design from assets/warnain_design.html in Editorial English:
- Broadsheet Before-After Comparison (.cmp, halftone before, sharp rectangular tags, ⇆ handle, serif figcaption)
- Editorial Stats Strip (.stats with Old Standard TT 2rem numerals & hairline dividers)
- CIE Lab Technical Plates & Chromaticity Plot matching --sheet (#F2F1ED) and --ink (#000)
- Editorial Exhibition Plate Gallery showcasing side-by-side archival restorations.
"""

import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components


def image_to_base64(image_input, max_side: int = 0) -> str:
    """Converts numpy array or PIL Image to base64 JPEG string."""
    if isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(np.clip(image_input, 0, 255).astype(np.uint8))
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    else:
        raise ValueError("Unsupported image type")

    if max_side and max(pil_img.size) > max_side:
        w, h = pil_img.size
        scale = max_side / float(max(w, h))
        pil_img = pil_img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG", quality=92)
    return base64.b64encode(buffered.getvalue()).decode()


def render_before_after_slider(
    img_before: np.ndarray,
    img_after: np.ndarray,
    label_before: str = "Monochrome",
    label_after: str = "WarnaIn Plate",
    caption: str = "Drag the dividing rule to compare the monochrome source against the WarnaIn colorization.",
    max_height: int = 500,
):
    """
    Renders the exact editorial comparison figure (.cmp + figcaption) from warnain_design.html.
    """
    h, w = img_after.shape[:2]
    aspect_ratio = w / max(1, h)
    # Match the ~820px lead column width in warnain_design.html
    cmp_height = int(min(max_height, max(320, 780 / aspect_ratio)))

    b64_before = image_to_base64(img_before)
    b64_after = image_to_base64(img_after)

    html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Old+Standard+TT:ital,wght@0,400;0,700;1,400&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
      :root {{
        --paper: #E8E7E2;
        --sheet: #F2F1ED;
        --ink: #000000;
        --soft: #4A4A47;
        --rule: #000000;
        --hair: #B9B8B2;
        --serif: 'Old Standard TT', Georgia, serif;
        --sans: 'Plus Jakarta Sans', system-ui, sans-serif;
      }}
      * {{
        box-sizing: border-box;
        margin: 0;
        padding: 0;
      }}
      body {{
        background: var(--sheet);
        color: var(--ink);
        font-family: var(--sans);
        overflow: hidden;
      }}
      figure {{
        margin: 0;
        width: 100%;
      }}
      .cmp {{
        position: relative;
        width: 100%;
        height: {cmp_height}px;
        overflow: hidden;
        border: 1px solid var(--ink);
        background: #D8D7D2;
        user-select: none;
        -webkit-user-select: none;
      }}
      .cmp .layer {{
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
      }}
      .cmp .layer img {{
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
      }}
      .cmp .before img {{
        filter: grayscale(1) contrast(1.04);
      }}
      .cmp .before:after {{
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle, rgba(0,0,0,.16) 0.6px, transparent 1px) 0 0 / 4px 4px;
        mix-blend-mode: multiply;
        pointer-events: none;
      }}
      .cmp .after {{
        clip-path: inset(0 0 0 50%);
      }}
      .cmp .tag {{
        position: absolute;
        bottom: 10px;
        padding: 3px 10px;
        background: var(--sheet);
        border: 1px solid var(--ink);
        font-size: 12.5px;
        font-weight: 600;
        color: var(--ink);
        z-index: 3;
        pointer-events: none;
      }}
      .tag.l {{ left: 10px; }}
      .tag.r {{ right: 10px; }}
      .bar {{
        position: absolute;
        top: 0;
        bottom: 0;
        left: 50%;
        width: 2px;
        background: var(--sheet);
        box-shadow: 0 0 0 1px var(--ink);
        z-index: 2;
        pointer-events: none;
      }}
      .bar:after {{
        content: "⇆";
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background: var(--sheet);
        border: 1.5px solid var(--ink);
        display: grid;
        place-items: center;
        font-size: 17px;
        color: var(--ink);
      }}
      .cmp input[type="range"] {{
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        opacity: 0;
        cursor: ew-resize;
        z-index: 4;
        margin: 0;
      }}
      figcaption {{
        font-family: var(--serif);
        font-style: italic;
        font-size: 0.95rem;
        color: var(--soft);
        padding: 6px 0 10px 0;
        border-bottom: 1px solid var(--hair);
      }}
    </style>
    </head>
    <body>
      <figure>
        <div class="cmp" id="cmp">
          <div class="layer before">
            <img src="data:image/jpeg;base64,{b64_before}" alt="Monochrome" />
          </div>
          <div class="layer after" id="after">
            <img src="data:image/jpeg;base64,{b64_after}" alt="WarnaIn Plate" />
          </div>
          <span class="tag l">{label_before}</span>
          <span class="tag r">{label_after}</span>
          <div class="bar" id="bar"></div>
          <input type="range" min="0" max="100" step="0.2" value="50" id="pos" aria-label="Slide to compare before and after">
        </div>
        <figcaption>{caption}</figcaption>
      </figure>

      <script>
        const pos = document.getElementById('pos');
        const after = document.getElementById('after');
        const bar = document.getElementById('bar');
        pos.addEventListener('input', () => {{
          after.style.clipPath = 'inset(0 0 0 ' + pos.value + '%)';
          bar.style.left = pos.value + '%';
        }});
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=cmp_height + 42)


def _fmt_num(val: float, decimals: int = 1) -> str:
    """Formats float with English decimal point."""
    return f"{val:.{decimals}f}"


def render_metric_cards(psnr: float, ssim: float, mae: float, latency_ms: float = 0.0):
    """
    Renders the editorial .stats strip from warnain_design.html.
    """
    psnr_str = _fmt_num(psnr, 1)
    ssim_str = _fmt_num(ssim, 2)
    mae_str = _fmt_num(mae, 1)
    lat_str = f"{int(round(latency_ms))}"

    st.markdown(
        f"""
        <div class="stats" aria-label="Quantitative Metrics">
          <div class="stat"><b>{psnr_str}</b><span>PSNR (dB)</span></div>
          <div class="stat"><b>{ssim_str}</b><span>SSIM</span></div>
          <div class="stat"><b>{mae_str}</b><span>MAE</span></div>
          <div class="stat"><b>{lat_str}</b><span>Inference Time (ms)</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_exhibition_gallery(plates: list):
    """
    Renders an Editorial Exhibition Plate gallery at the bottom of the Studio page.
    Each item in `plates` is a dict:
      {
        "plate_no": "PLATE I",
        "title": "Vintage Portrait",
        "category": "Human Portraits",
        "before_rgb": np.ndarray,
        "after_rgb": np.ndarray,
        "caption": str,
      }
    """
    if not plates:
        return

    cards_html = []
    for idx, item in enumerate(plates):
        b64_bw = image_to_base64(item["before_rgb"], max_side=420)
        b64_col = image_to_base64(item["after_rgb"], max_side=420)
        col_cls = "ex-card" if (idx % 3 == 0) else "ex-card ex-card-divider"
        row_cls = " ex-card-row2" if idx >= 3 else ""
        border_cls = f"{col_cls}{row_cls}"
        card_str = (
            f'<article class="{border_cls}">'
            f'<div class="ex-meta">'
            f'<span class="ex-kicker">{item["plate_no"]}</span>'
            f'<span class="ex-cat">{item["category"]}</span>'
            f'</div>'
            f'<h3 class="ex-title">{item["title"]}</h3>'
            f'<div class="ex-pair">'
            f'<div class="ex-frame ex-bw">'
            f'<img src="data:image/jpeg;base64,{b64_bw}" alt="{item["title"]} Monochrome" />'
            f'<span class="ex-tag">Monochrome</span>'
            f'</div>'
            f'<div class="ex-frame ex-col">'
            f'<img src="data:image/jpeg;base64,{b64_col}" alt="{item["title"]} Colorized" />'
            f'<span class="ex-tag">Colorized</span>'
            f'</div>'
            f'</div>'
            f'<p class="ex-cap">{item["caption"]}</p>'
            f'</article>'
        )
        cards_html.append(card_str)

    joined_cards = "".join(cards_html)
    gallery_html = (
        '<section class="exhibition-section">'
        '<div class="exhibition-header">'
        '<span class="exhibition-kicker">ARCHIVAL EXHIBITION SUPPLEMENT</span>'
        '<h2 class="exhibition-h2">Selected Monochrome &amp; Restored Plates</h2>'
        '<p class="exhibition-deck">Side-by-side comparison of archival sample photographs restored through domain-calibrated CIE L*a*b* neural inference.</p>'
        '</div>'
        f'<div class="exhibition-grid">{joined_cards}</div>'
        '</section>'
    )
    st.markdown(gallery_html, unsafe_allow_html=True)


def render_cielab_inspector(colorized_rgb: np.ndarray):
    """
    Renders the 3 CIE Lab channels (L*, a*, b*) styled as editorial technical plates.
    """
    lab = cv2.cvtColor(colorized_rgb, cv2.COLOR_RGB2LAB)
    L_channel = lab[:, :, 0]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<h2 class="serif-sub">1. L* Channel (Luminance)</h2>', unsafe_allow_html=True)
        st.image(L_channel, use_container_width=True, clamp=True)
        st.markdown(
            '<p class="plate-cap">Monochrome lightness intensity (0–100) preserving 100% of original edge geometry.</p>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown('<h2 class="serif-sub">2. a* Channel (Green–Red)</h2>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 4 * L_channel.shape[0] / max(1, L_channel.shape[1])), dpi=110)
        fig.patch.set_facecolor("#F2F1ED")
        ax.imshow(a_channel, cmap="RdYlGn_r", vmin=45, vmax=210)
        ax.axis("off")
        fig.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown(
            '<p class="plate-cap">Chrominance map along the a* axis: lower values shift toward green, higher toward crimson.</p>',
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown('<h2 class="serif-sub">3. b* Channel (Blue–Yellow)</h2>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 4 * L_channel.shape[0] / max(1, L_channel.shape[1])), dpi=110)
        fig.patch.set_facecolor("#F2F1ED")
        ax.imshow(b_channel, cmap="coolwarm", vmin=45, vmax=210)
        ax.axis("off")
        fig.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown(
            '<p class="plate-cap">Chrominance map along the b* axis: lower values shift toward cerulean blue, higher toward warm gold.</p>',
            unsafe_allow_html=True,
        )


def render_chroma_scatter(colorized_rgb: np.ndarray):
    """
    Plots the 2D CIE a* vs b* chromaticity distribution in matching broadsheet style.
    """
    lab = cv2.cvtColor(colorized_rgb, cv2.COLOR_RGB2LAB)
    h, w, _ = lab.shape
    step = max(1, int(np.sqrt((h * w) / 4500)))
    a_sub = lab[::step, ::step, 1].flatten().astype(np.float32) - 128.0
    b_sub = lab[::step, ::step, 2].flatten().astype(np.float32) - 128.0
    rgb_sub = colorized_rgb[::step, ::step, :].reshape(-1, 3) / 255.0

    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=120)
    fig.patch.set_facecolor("#F2F1ED")
    ax.set_facecolor("#E8E7E2")

    ax.scatter(a_sub, b_sub, c=rgb_sub, s=10, alpha=0.7, edgecolors="none")
    ax.axhline(0, color="#4A4A47", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="#4A4A47", linestyle="--", linewidth=0.8)

    ax.set_xlim(-85, 85)
    ax.set_ylim(-85, 85)
    ax.set_xlabel("a* Axis (Green ←  → Red)", color="#000000", fontsize=9, fontweight="bold")
    ax.set_ylabel("b* Axis (Blue ←  → Yellow)", color="#000000", fontsize=9, fontweight="bold")
    ax.set_title("CIE a*b* Chromaticity Gamut Distribution", color="#000000", fontsize=10, fontweight="bold", pad=10)
    ax.tick_params(colors="#4A4A47", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
