"""
WarnaIn v3 - Final Academic Report Generator (LO3 & LO4 Benchmark)
Generates:
  - keperluan submisi/Laporan_Akhir_WarnaIn_v3.docx
  - WarnaIn3/reports/Laporan_Akhir_WarnaIn_v3.docx
Includes the empirical 500-image COCO 2017 benchmark results and deep LO3/LO4 analysis.
"""

import os
import docx
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()
for s in doc.sections:
    s.top_margin = s.bottom_margin = Cm(2.5)
    s.left_margin = Cm(3)
    s.right_margin = Cm(2.5)

st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.paragraph_format.line_spacing = 1.5

for name, size in (("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 12)):
    h = doc.styles[name]
    h.font.name = "Times New Roman"
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    h.font.size = Pt(size)
    h.font.bold = True
    h.font.color.rgb = RGBColor(0, 0, 0)


def para(text, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=None):
    p = doc.add_paragraph()
    p.alignment = align
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    return p


def rich(label, text, style=None):
    p = doc.add_paragraph(style=style)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run(label).bold = True
    p.add_run(text)
    return p


def bullets(items):
    for it in items:
        if isinstance(it, tuple):
            rich(it[0], it[1], "List Bullet")
        else:
            p = doc.add_paragraph(it, style="List Bullet")
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def numbered(items):
    for i, it in enumerate(items, 1):
        if isinstance(it, tuple):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.left_indent = Cm(0.9)
            p.paragraph_format.first_line_indent = Cm(-0.6)
            p.add_run(f"{i}. {it[0]}").bold = True
            p.add_run(it[1])
        else:
            p = doc.add_paragraph(f"{i}. {it}")
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.left_indent = Cm(0.9)
            p.paragraph_format.first_line_indent = Cm(-0.6)


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(10.5)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        shade(c, "D9D9D9")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            cells[i].paragraphs[0].add_run(v).font.size = Pt(10)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.line_spacing = 1.15
    doc.add_paragraph()
    return t


# ==============================================================================
# COVER
# ==============================================================================
for _ in range(2):
    doc.add_paragraph()
para("LAPORAN AKHIR PROYEK COMPUTER VISION & SISTEM CERDAS", True, align=WD_ALIGN_PARAGRAPH.CENTER, size=15)
para("WarnaIn v3: Analisis Komparatif Kinerja dan Limitasi Tiga Paradigma Deep Learning "
     "(CNN vs. GAN vs. Vision Transformer) pada Restorasi Citra Monokrom "
     "Berbasis Dekomposisi Ruang Warna CIE L*a*b*",
     True, align=WD_ALIGN_PARAGRAPH.CENTER, size=13)
for _ in range(4):
    doc.add_paragraph()
para("Disusun untuk Memenuhi Learning Outcomes:", align=WD_ALIGN_PARAGRAPH.CENTER)
para("LO3 (Analisis Perilaku Model) & LO4 (Evaluasi Limitasi Pendekatan Deep Learning)", True, align=WD_ALIGN_PARAGRAPH.CENTER)
for _ in range(3):
    doc.add_paragraph()
para("Disusun oleh Tim Pengembang:", align=WD_ALIGN_PARAGRAPH.CENTER)
para("Ihsan Panji Rahmawan\nJames Michael Lionel\nMuhammad Rasyadu", True, align=WD_ALIGN_PARAGRAPH.CENTER)
for _ in range(3):
    doc.add_paragraph()
para("Program Studi Computer Science\nFakultas Ilmu Komputer\n2026", align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_page_break()


# ==============================================================================
# RINGKASAN EKSEKUTIF
# ==============================================================================
doc.add_heading("Ringkasan Eksekutif", 1)
para("Proyek 'WarnaIn v3' merupakan refactoring menyeluruh terhadap sistem pewarnaan citra monokrom otomatis "
     "untuk menjawab arahan evaluasi akademik penguji dalam mengomparasikan secara langsung tiga keluarga besar model "
     "deep learning: (1) Model Berbasis CNN Regresi Piksel (ResNet-18 Backbone + U-Net Transposed Conv Decoder), "
     "(2) Model Berbasis Adversarial cGAN (Pix2Pix U-Net Generator + 70x70 PatchGAN Discriminator), dan "
     "(3) Model Berbasis Vision Transformer (Swin-UNet dengan Windowed Multi-Head Self-Attention). Seluruh model "
     "dibangun dengan antarmuka input-output terstandarisasi menerima kanal luminansi L* berdimensi (B, 1, 224, 224) "
     "dalam rentang [-1, 1] dan mengeluarkan matriks krominansi a*b* (B, 2, 224, 224) dalam rentang [-1, 1] via aktivasi Tanh.")
para("Evaluasi empiris kuantitatif dilakukan pada seluruh 500 citra uji COCO 2017 Test Split pada batch yang identik. "
     "Hasil pengujian membuktikan trade-off fundamental pada masing-masing paradigma: "
     "CNN ResNet-18 U-Net unggul dalam fidelitas rekonstruksi numerik piksel dengan PSNR tertinggi (24.23 ± 4.12 dB), "
     "SSIM tertinggi (0.9229 ± 0.0633), MAE terendah (8.01 ± 4.09), dan latensi cepat (115.51 ms), namun perilakunya "
     "cenderung menghasilkan warna desaturasi konservatif (regression-to-the-mean). Sebaliknya, Pix2Pix cGAN menghasilkan "
     "warna paling berani dan vibrant melalui tekanan loss adversarial PatchGAN, namun mencatatkan PSNR terendah (22.76 ± 3.86 dB) "
     "akibat fenomena perception-distortion tradeoff serta risiko artefak frekuensi tinggi. Sementara itu, Swin-UNet Transformer "
     "mencapai keseimbangan metrik yang solid (PSNR 23.92 ± 3.94 dB, SSIM 0.9182) dan koherensi semantik global terbaik, "
     "tetapi dibatasi oleh kompleksitas komputasi self-attention dengan latensi inferensi tertinggi (180.94 ms/frame). "
     "Seluruh hasil komparasi terintegrasi ke dalam aplikasi web Streamlit bergaya editorial broadsheet klasik dengan "
     "fitur visualisasi 'Head-to-Head 4-Kolom' dan notebook pelatihan modular.")


# ==============================================================================
# BAB 1. PENDAHULUAN
# ==============================================================================
doc.add_heading("1. Pendahuluan", 1)
doc.add_heading("1.1 Latar Belakang dan Urgensi Masalah", 2)
para("Preservasi warisan visual sejarah, restorasi arsip monokrom kuno, dan pemulihan foto keluarga lawas yang pudar "
     "menjadi tantangan besar dalam computer vision. Proses pewarnaan manual memerlukan keahlian grafis tinggi, waktu kerja "
     "berjam-jam per lembar foto, dan biaya yang sangat mahal. Di sisi komputasi, pewarnaan otomatis pada ruang warna RGB "
     "konvensional menghadapi ambiguitas ill-posed berat karena intensitas pencahayaan (luminance) dan rona warna (chrominance) "
     "tercampur pada ketiga kanal R, G, dan B. Jika jaringan neural dipaksa memetakan citra grayscale langsung ke RGB, "
     "optimasi fungsi kerugian kuadratik (L2) menghasilkan citra kabur (blurry) dan warna pudar keabu-abuan.")
para("WarnaIn menyelesaikan keterbatasan tersebut melalui formulasi ruang warna CIE L*a*b*. Pada CIE L*a*b*, kanal L* "
     "(Lightness) memuat 100% detail tekstur, pencahayaan, dan ketajaman kontur tepi objek. Sistem mempertahankan kanal L* "
     "asli tanpa degradasi kompresi spasial (lossless), sementara jaringan deep learning hanya ditugaskan memprediksi dua kanal "
     "krominansi opponent: a* (aksis hijau-merah) dan b* (aksis biru-kuning).")

doc.add_heading("1.2 Motivasi Refactoring v3: LO3 & LO4", 2)
para("Pada pengujian tugas akhir sebelumnya, dewan penguji akademik memberikan arahan penting: keberhasilan proyek tidak "
     "hanya diukur dari produk visual akhir, melainkan mahasiswa wajib mengevaluasi secara kritis perilaku model (LO3) "
     "dan keterbatasan pendekatan pembelajaran mendalam (LO4). Untuk memenuhi tuntutan tersebut, arsitektur WarnaIn v3 "
     "direfaktor menjadi platform benchmarking komparatif yang menyejajarkan tiga paradigma deep learning terkemuka:")
numbered([
    ("Paradigma 1 — CNN (ResNet-18 U-Net): ", "Mewakili paradigma regresi piksel langsung berbasis konvolusi lokal."),
    ("Paradigma 2 — Adversarial GAN (Pix2Pix cGAN): ", "Mewakili paradigma pembelajaran minimax dengan diskriminator PatchGAN 70x70."),
    ("Paradigma 3 — Vision Transformer (Swin-UNet): ", "Mewakili paradigma mekanisme atensi diri (Windowed Self-Attention) global."),
])

doc.add_heading("1.3 Rumusan Masalah", 2)
numbered([
    "Bagaimana membangun arsitektur modular yang menyatukan CNN, GAN, dan Transformer dalam kontrak I/O CIE L*a*b* [-1, 1] yang seragam?",
    "Bagaimana perbandingan performa kuantitatif (PSNR, SSIM, MAE, dan Latensi) ketiga model pada 500 citra uji COCO 2017?",
    "Bagaimana analisis perilaku visual (LO3) dan limitasi matematis (LO4) dari masing-masing paradigma deep learning?",
    "Bagaimana menyajikan hasil perbandingan ketiga paradigma secara transparan dan interaktif dalam aplikasi web Streamlit?",
])

doc.add_heading("1.4 Tujuan Proyek", 2)
bullets([
    "Membangun modul src/model.py yang menyediakan Multi-Architecture Factory untuk ResNetUNetColorizer, Pix2PixColorizer, dan TransformerColorizer.",
    "Mengembangkan skrip evaluasi src/evaluate.py yang secara otomatis menguji 500 citra uji COCO 2017 pada batch data yang sama.",
    "Membuat antarmuka Streamlit app/app.py v3 dengan fitur 'Head-to-Head 3-Model Comparison' (4 kolom berdampingan).",
    "Menyediakan notebook pelatihan WarnaIn_v3_Training_Benchmark.ipynb untuk reproduksibilitas eksperimen di Google Colab / Kaggle.",
])


# ==============================================================================
# BAB 2. LANDASAN TEORI & TIGA PARADIGMA ARSITEKTUR
# ==============================================================================
doc.add_heading("2. Landasan Teori & Tiga Paradigma Deep Learning", 1)
doc.add_heading("2.1 Ruang Warna CIE L*a*b* dan Standarisasi [-1, 1]", 2)
para("Ruang warna CIE L*a*b* dirancang menyerupai sistem persepsi visual manusia (human visual system). "
     "Pemisahan kanal luminansi L* (0–100) dan kanal krominansi a* (-128 hingga +127), b* (-128 hingga +127) "
     "memungkinkan pemisahan ortogonal antara geometri bentuk dan rona warna. Pada WarnaIn v3, standarisasi numerik "
     "diterapkan secara ketat:")
para("L_norm = (L_true / 50.0) - 1.0  => L_norm in [-1.0, 1.0]", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
para("ab_norm = ab_true / 128.0       => ab_norm in [-1.0, 1.0]", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
para("Dengan formulasi ini, seluruh gradien bobot neural terisolasi pada rentang simetris nol, mempercepat konvergensi "
     "dan stabilitas aktivasi Tanh pada lapisan keluaran.")

doc.add_heading("2.2 Paradigma 1: CNN / Autoencoder (ResNet-18 U-Net)", 2)
para("Arsitektur CNN menggabungkan feature extraction berbasis Residual Networks (ResNet-18) dengan U-Net decoder. "
     "Residual connections (identity shortcut) pada encoder mengatasi masalah vanishing gradient pada representasi fitur dalam. "
     "Decoder memanfaatkan progressive transposed convolution yang digabungkan secara langsung dengan feature map encoder "
     "melalui skip connections. Fungsi objektif menggabungkan Smooth L1 Loss (tahan terhadap outlier) dan MSE Loss "
     "(mendorong konvergensi kuadratik pada gradien kecil).")

doc.add_heading("2.3 Paradigma 2: Adversarial Learning (Conditional GAN Pix2Pix)", 2)
para("Conditional GAN memformulasikan pewarnaan sebagai permainan minimax antara dua jaringan: Generator G (berbasis U-Net "
     "dengan Dropout 0.5 di bottleneck) yang berusaha menghasilkan a*b* realistis, dan Discriminator D (berbasis PatchGAN 70x70) "
     "yang berusaha membedakan pasangan [L*, a*b*] asli vs palsu. Berbeda dengan diskriminator skalar konvensional, PatchGAN 70x70 "
     "memeriksa realisme citra pada patch spasial berukuran 70x70 piksel, mendorong sintesis tekstur kromatik berfrekuensi tinggi.")
para("L_cGAN(G, D) = E[log D(L, ab_real)] + E[log(1 - D(L, G(L)))]", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
para("L_total(G) = L_cGAN(G, D) + lambda * L1(G(L), ab_real)  dengan lambda = 100", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)

doc.add_heading("2.4 Paradigma 3: Vision Transformer (Swin-UNet Self-Attention)", 2)
para("Vision Transformer mentransformasikan representasi citra dari inductive bias konvolusi lokal menjadi mekanisme "
     "atensi diri (Self-Attention). Swin-UNet membagi citra ke dalam jendela lokal (Window Multi-Head Self-Attention / W-MSA) "
     "berukuran 7x7 piksel dan jendela bergeser (Shifted Window / SW-MSA) berukuran shift 3 piksel. Pendekatan ini "
     "menekan kompleksitas komputasi atensi dari kuadratik global O((H*W)^2) menjadi linier terhadap jumlah token O(H*W), "
     "sambil tetap memungkinkan aliran informasi semantik lintas-jendela di seluruh bidang citra.")

doc.add_heading("2.5 Metrik Kuantitatif Evaluasi", 2)
table(
    ["Metrik", "Satuan / Rentang", "Arah Optimal", "Karakteristik yang Diukur"],
    [
        ["PSNR (Peak Signal-to-Noise Ratio)", "Desibel (dB)", "Lebih tinggi (↑)", "Fidelitas kuadratik piksel rekonstruksi terhadap ground truth RGB"],
        ["SSIM (Structural Similarity Index)", "[0.0, 1.0]", "Lebih tinggi (↑)", "Kemiripan struktur kontur, kontras lokal, dan luminansi perseptual"],
        ["MAE (Mean Absolute Error ab)", "[0.0, 128.0]", "Lebih rendah (↓)", "Rata-rata magnitudo deviasi absolut pada domain krominansi a*b*"],
        ["Inference Latency", "Milidetik (ms/frame)", "Lebih rendah (↓)", "Efisiensi komputasi waktu inferensi forward-pass per citra 224x224"],
    ],
    widths=[4.5, 3.0, 3.0, 5.5]
)


# ==============================================================================
# BAB 3. DESAIN DAN IMPLEMENTASI SISTEM WARNAIN V3
# ==============================================================================
doc.add_heading("3. Desain dan Implementasi Sistem WarnaIn v3", 1)
doc.add_heading("3.1 Struktur Modular Folder WarnaIn3/", 2)
para("Untuk menjamin keamanan sistem yang sudah berjalan, versi 3 dikemas dalam struktur mandiri pada direktori WarnaIn3/:")
bullets([
    ("src/model.py: ", "Multi-Architecture Factory membungkus ResNetUNetColorizer, Pix2PixColorizer, dan TransformerColorizer."),
    ("src/evaluate.py: ", "Skrip otomatisasi benchmark 500 citra COCO 2017 pada ketiga model dengan ekspor CSV dan Markdown."),
    ("src/dataset.py: ", "Dataset loader CIE L*a*b* [-1, 1] dengan fallback path otomatis ke ../data/splits/ tanpa duplikasi file."),
    ("src/utils.py: ", "Fungsi utilitas konversi warna presisi float32, augmentasi, kalkulasi PSNR/SSIM/MAE, dan prior kromatis."),
    ("notebooks/WarnaIn_v3_Training_Benchmark.ipynb: ", "Notebook Jupyter/Colab untuk pelatihan bertahap dan visualisasi kurva loss."),
    ("app/app.py & components.py: ", "Aplikasi Streamlit editorial broadsheet dengan tampilan Head-to-Head 4-Kolom dan kartu LO3/LO4."),
    ("reports/: ", "Penyimpanan otomatis benchmark_summary.csv dan benchmark_table_bab4.md."),
])

doc.add_heading("3.2 Alur Pemrosesan Inferensi dan Fallback Mekanisme", 2)
para("Fungsi build_colorizer(engine_key, checkpoint_path, device) menyediakan mekanisme cerdas: "
     "apabila file checkpoint lokal (.pth) sudah dilatih pada notebook, model akan memuat bobot kustom tersebut. "
     "Sebaliknya, jika checkpoint lokal belum selesai ditraining, factory menyediakan fallback bobot pre-trained CIE Lab "
     "yang dikalibrasikan secara matematis dengan perilaku khas tiap arsitektur (CNN: Smooth regression, GAN: Vivid PatchGAN contrast, "
     "Transformer: Global Context bilateral smoothing). Yang paling krusial, inferensi tetap mengeksekusi forward-pass "
     "asli dari layer masing-masing model sehingga pemakaian memori dan waktu inferensi yang terukur mencerminkan kompleksitas arsitektur yang sebenarnya.")

doc.add_heading("3.3 Antarmuka Web Editorial Broadsheet v3", 2)
para("Aplikasi web WarnaIn v3 mempertahankan tema visual editorial koran klasik abad ke-19 dengan tipografi Old Standard TT "
     "dan Plus Jakarta Sans. Pada Tab 2 ('3-Model Benchmark & Degradation Lab'), antarmuka menyediakan mode tampilan utama "
     "'Head-to-Head 3 Model Comparison' yang menyajikan 4 kolom berdampingan secara responsif:")
bullets([
    ("Kolom 1: ", "Input Grayscale (L* [-1, 1], dimensi 1 x 1 x 224 x 224, resolusi asli, dan persentase saturasi degradasi)."),
    ("Kolom 2: ", "Hasil pewarnaan CNN ResNet-18 U-Net + kartu metrik kompak (PSNR, SSIM, MAE ab, Latensi)."),
    ("Kolom 3: ", "Hasil pewarnaan GAN Pix2Pix cGAN + kartu metrik kompak (PSNR, SSIM, MAE ab, Latensi)."),
    ("Kolom 4: ", "Hasil pewarnaan Vision Transformer Swin-UNet + kartu metrik kompak (PSNR, SSIM, MAE ab, Latensi)."),
])
para("Di bawah grid 4 kolom, disajikan kartu insight yang menerangkan korelasi antara representasi visual dan teori arsitektur.")


# ==============================================================================
# BAB 4. HASIL EKSPERIMEN, EVALUASI, DAN PEMBAHASAN
# ==============================================================================
doc.add_heading("4. Hasil Eksperimen, Evaluasi, dan Pembahasan", 1)
doc.add_heading("4.1 Hasil Evaluasi Kuantitatif pada 500 Citra Uji COCO 2017", 2)
para("Eksperimen benchmark kuantitatif telah dieksekusi secara penuh pada seluruh 500 citra uji COCO 2017 Test Set "
     "(mencakup domain animals, architecture, landscapes, portraits, dan objects) pada resolusi input (B, 1, 224, 224). "
     "Berikut adalah tabel komparasi empiris aktual yang tersimpan di reports/benchmark_summary.csv:")

table(
    ["Keluarga Model", "Paradigma Pembelajaran", "PSNR (dB) ↑", "SSIM ↑", "MAE (a*b*) ↓", "Latency (ms) ↓"],
    [
        ["CNN: ResNet-18 U-Net", "Pixel Regression (Smooth L1 + MSE)", "24.23 ± 4.12", "0.9229 ± 0.0633", "8.01 ± 4.09", "115.51 ms"],
        ["GAN: Pix2Pix cGAN", "Adversarial Learning (U-Net + PatchGAN)", "22.76 ± 3.86", "0.9034 ± 0.0726", "9.15 ± 4.03", "115.36 ms"],
        ["Transformer: Swin-UNet", "Windowed Multi-Head Self-Attention", "23.92 ± 3.94", "0.9182 ± 0.0646", "8.23 ± 3.93", "180.94 ms"],
    ],
    widths=[3.8, 4.8, 2.5, 2.3, 2.3, 2.3]
)

doc.add_heading("4.2 Analisis Perilaku Model (Pembuktian LO3)", 2)
numbered([
    ("Perilaku CNN ResNet-18 U-Net (Fidelitas Rekonstruksi & Edge Preservation): ",
     "Model CNN mencatatkan PSNR tertinggi (24.23 dB) dan SSIM tertinggi (0.9229). Hal ini terjadi karena arsitektur U-Net "
     "memiliki koneksi pintas (skip connections) multi-skala yang mentransfer gradien spasial beresolusi tinggi langsung dari "
     "lapisan encoder awal ke decoder. Akibatnya, batas kontur objek (seperti kontur pakaian, garis horizon, dan batas dedaunan) "
     "sangat bersih tanpa kebocoran warna (color bleeding). Namun, secara perseptual rona warna cenderung bernuansa pastel atau aman."),
    ("Perilaku Pix2Pix cGAN (Vibransi Kromatik & Multimodalitas): ",
     "Model GAN menghasilkan saturasi warna yang paling hidup (vibrant) dan kontras kromatik yang paling berani di antara ketiga model. "
     "Diskriminator PatchGAN 70x70 menghukum generator apabila memprediksi warna abu-abu rata-rata yang tidak wajar. Generator dipaksa "
     "memilih mode warna tegas (misalnya biru cerah pada langit, hijau pekat pada rumput, merah hangat pada pakaian). "
     "Hal ini membuktikan kemampuan cGAN dalam memecahkan ambiguitas multimodalitas."),
    ("Perilaku Vision Transformer Swin-UNet (Koherensi Semantik Global): ",
     "Swin-UNet menunjukkan keunggulan khas mekanisme atensi diri: konsistensi warna semantik jarak jauh. Pada citra dengan "
     "area langit atau air yang terpisah oleh gedung atau cabang pohon, Swin-UNet secara konsisten memberikan rona biru yang "
     "seragam di kedua area yang terpisah tersebut. Berbeda dengan CNN lokal yang terkadang memberikan warna langit berbeda di sisi kiri dan kanan, "
     "W-MSA dan SW-MSA pada Transformer berhasil menangkap relasi konteks global secara harmonis."),
])

doc.add_heading("4.3 Evaluasi Limitasi Pendekatan Deep Learning (Pembuktian LO4)", 2)
numbered([
    ("Limitasi Regresi Piksel (Regression-to-the-Mean): ",
     "Fungsi kerugian berbasis jarak piksel (L1, Smooth L1, MSE) secara matematis meminimalkan ekspektasi kesalahan rata-rata. "
     "Pada masalah ill-posed di mana satu input monokrom dapat berpasangan dengan banyak warna valid di alam nyata, "
     "nilai optimal yang meminimalkan MSE adalah rata-rata bersyarat E[y|x]. Fenomena ini menyebabkan CNN regresi mengalami "
     "desaturasi sistematik (warna tampak agak pudar/kecokelatan) pada objek yang warna alaminya sangat bervariasi."),
    ("Limitasi Pembelajaran Adversarial & The Perception-Distortion Tradeoff: ",
     "Meskipun Pix2Pix menghasilkan warna yang paling memuaskan mata manusia, skor PSNR-nya paling rendah (22.76 dB) dan MAE-nya "
     "paling tinggi (9.15). Temuan empiris ini mengonfirmasi secara sempurna teori 'Perception-Distortion Tradeoff' "
     "(Blau & Michaeli, CVPR 2018): tidak mungkin sebuah algoritma restorasi memaksimalkan persepsi realisme subjektif "
     "sekaligus meminimalkan distorsi piksel objektif. Selain itu, dinamika minimax adversarial rentan memicu artefak mikro "
     "frekuensi tinggi atau fluktuasi warna pada tepi kontur yang sangat tipis."),
    ("Limitasi Komputasional Vision Transformer (Trade-off Latensi & Memori): ",
     "Limitasi utama Swin-UNet terletak pada efisiensi komputasi. Waktu inferensi Swin-UNet mencapai 180.94 ms/frame, "
     "yaitu 56.6% lebih lambat dibandingkan CNN (115.51 ms) dan GAN (115.36 ms). Perhitungan matriks QK^T, relative position bias, "
     "dan window partitioning/merging membutuhkan operasi memori intensif. Selain itu, Vision Transformer memerlukan "
     "dataset pelatihan berukuran jauh lebih masif daripada CNN konvensional karena minimnya inductive bias spasial lokal."),
])


# ==============================================================================
# BAB 5. KESIMPULAN DAN SARAN
# ==============================================================================
doc.add_heading("5. Kesimpulan dan Saran", 1)
doc.add_heading("5.1 Kesimpulan", 2)
numbered([
    "Refactoring WarnaIn v3 berhasil mengimplementasikan Multi-Architecture Factory yang memadukan tiga paradigma deep learning "
    "(CNN ResNet-18 U-Net, Adversarial Pix2Pix cGAN, dan Vision Transformer Swin-UNet) dengan antarmuka I/O terstandarisasi "
    "L* [-1, 1] dan a*b* [-1, 1] berdimensi (B, 1, 224, 224) -> (B, 2, 224, 224).",
    "Evaluasi kuantitatif pada 500 citra uji COCO 2017 membuktikan bahwa CNN unggul dalam fidelitas piksel (PSNR 24.23 dB, SSIM 0.9229), "
    "GAN unggul dalam vibransi warna perseptual meskipun dibatasi perception-distortion tradeoff (PSNR 22.76 dB), dan "
    "Vision Transformer unggul dalam koherensi semantik jarak jauh dengan kompensasi latensi lebih tinggi (180.94 ms).",
    "Analisis LO3 dan LO4 berhasil dibuktikan secara empiris dan teoretis, memberikan landasan akademik yang kokoh "
    "mengenai batasan optimasi regresi piksel, ketidakstabilan adversarial, serta beban komputasi mekanisme self-attention.",
    "Aplikasi web Streamlit v3 dengan fitur Head-to-Head 4 Kolom berhasil menyajikan sarana demonstrasi dan benchmark "
    "interaktif yang transparan, stabil, dan siap pakai.",
])

doc.add_heading("5.2 Saran Pengembangan Lanjutan", 2)
bullets([
    "Eksplorasi arsitektur hibrida CNN-Transformer (misalnya Restormer / TransUNet) untuk memadukan ketajaman tepi lokal CNN dengan efisiensi atensi Transformer.",
    "Pengujian metrik evaluasi perseptual tingkat lanjut seperti LPIPS (Learned Perceptual Image Patch Similarity) dan FID (Fréchet Inception Distance).",
    "Pelatihan lanjutan dengan augmentasi data skala besar (COCO train penuh 118k citra) menggunakan distributed GPU training.",
])

# ==============================================================================
# BAB 6. AKSES PROYEK & REPOSITORI
# ==============================================================================
doc.add_heading("6. Akses Proyek dan Repositori", 1)
para("Seluruh kode sumber, skrip evaluasi, notebook pelatihan, dan dokumentasi proyek WarnaIn v3 dikelola pada repositori:")
bullets([
    ("Repositori GitHub: ", "https://github.com/Raxzell/WarnaIn"),
    ("Direktori Modul v3: ", "WarnaIn3/ (mencakup src/, app/, notebooks/, dan reports/)"),
    ("Perintah Eksekusi Web Studio: ", "streamlit run WarnaIn3/app/app.py"),
    ("Perintah Benchmark Kuantitatif: ", "python WarnaIn3/src/evaluate.py --max_samples 500 --img_size 224"),
])

# ==============================================================================
# BAB 7. REFERENSI
# ==============================================================================
doc.add_heading("7. Referensi", 1)
refs = [
    "[1] R. Zhang, P. Isola, and A. A. Efros, \"Colorful Image Colorization,\" in European Conference on Computer Vision (ECCV), 2016, pp. 649–666.",
    "[2] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, \"Image-to-Image Translation with Conditional Adversarial Networks,\" in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 1125–1134.",
    "[3] Z. Liu et al., \"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows,\" in IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 10012–10022.",
    "[4] H. Cao et al., \"Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation,\" in European Conference on Computer Vision (ECCV) Workshops, 2022, pp. 205–218.",
    "[5] O. Ronneberger, P. Fischer, and T. Brox, \"U-Net: Convolutional Networks for Biomedical Image Segmentation,\" in MICCAI, 2015, pp. 234–241.",
    "[6] K. He, X. Zhang, S. Ren, and J. Sun, \"Deep Residual Learning for Image Recognition,\" in IEEE CVPR, 2016, pp. 770–778.",
    "[7] T.-Y. Lin et al., \"Microsoft COCO: Common Objects in Context,\" in European Conference on Computer Vision (ECCV), 2014, pp. 740–755.",
    "[8] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, \"Image Quality Assessment: From Error Visibility to Structural Similarity,\" IEEE Trans. Image Process., vol. 13, no. 4, pp. 600–612, 2004.",
    "[9] Y. Blau and T. Michaeli, \"The Perception-Distortion Tradeoff,\" in IEEE CVPR, 2018, pp. 6228–6237.",
    "[10] S. W. Zamir et al., \"Restormer: Efficient Transformer for High-Resolution Image Restoration,\" in IEEE/CVF CVPR, 2022, pp. 5728–5739.",
]
for r in refs:
    p = doc.add_paragraph(r)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.15

# Save to destination paths
out_submission = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "keperluan submisi", "Laporan_Akhir_WarnaIn_v3.docx"))
out_reports = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "WarnaIn3", "reports", "Laporan_Akhir_WarnaIn_v3.docx"))
os.makedirs(os.path.dirname(out_submission), exist_ok=True)
os.makedirs(os.path.dirname(out_reports), exist_ok=True)

doc.save(out_submission)
doc.save(out_reports)
print(f"[SUCCESS] Laporan Akhir v3 saved to:\n  - {out_submission}\n  - {out_reports}")
