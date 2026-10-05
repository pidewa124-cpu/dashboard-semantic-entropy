import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- konfigurasi
BASE = Path(__file__).parent / 'data'
LABELS_FILE = BASE / 'labels.json'
METRICS_FILE = BASE / 'eval_metrics.json'

# Palet Okabe-Ito (aman untuk buta warna; TIDAK memakai merah-hijau bersamaan)
BLUE, ORANGE, SKY, NAVY, GREY = '#0072B2', '#E69F00', '#56B4E9', '#0B2545', '#6B7A90'

st.set_page_config(page_title='Deteksi Halusinasi - Semantic Entropy',
                   page_icon='🧬', layout='wide')

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'Inter',sans-serif;}
.stApp {background:#F4F7FB;}
.block-container {padding-top:1.6rem; padding-bottom:1rem; max-width:1400px;}
header[data-testid="stHeader"] {background:transparent;}
h1 {color:#0B2545; font-weight:800; font-size:2rem !important; margin-bottom:.2rem;}
.sub {color:#6B7A90; font-size:.9rem; margin-bottom:1rem;}
/* sidebar */
[data-testid="stSidebar"] {background:#FFFFFF; border-right:1px solid #E6ECF3;}
.logo {display:flex; gap:.6rem; align-items:center; margin:.4rem 0 1.6rem 0;}
.logo-i {background:linear-gradient(135deg,#0072B2,#56B4E9); color:#fff; font-weight:800;
  width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center;}
.logo-t {color:#0B2545; font-weight:700; line-height:1.15; font-size:.95rem;}
[data-testid="stSidebar"] [role="radiogroup"] {gap:.25rem;}
[data-testid="stSidebar"] [role="radiogroup"] label {padding:.6rem .8rem; border-radius:10px;
  border-left:4px solid transparent; width:100%; cursor:pointer;}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {display:none;}
[data-testid="stSidebar"] [role="radiogroup"] label p {color:#6B7A90; font-weight:500; font-size:.95rem;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#EAF4FB; border-left-color:#0072B2;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {color:#0B2545; font-weight:700;}
/* kartu */
[data-testid="stVerticalBlockBorderWrapper"] {background:#fff; border:none !important;
  border-radius:16px; box-shadow:0 2px 12px rgba(11,37,69,.06); padding:.4rem .6rem;}
.kpi {background:#fff; border-radius:16px; padding:1rem 1.2rem; box-shadow:0 2px 12px rgba(11,37,69,.06);}
.kpi-l {color:#6B7A90; font-size:.8rem; font-weight:600; text-transform:uppercase; letter-spacing:.04em;}
.kpi-v {color:#0B2545; font-size:2.1rem; font-weight:800; line-height:1.25;}
.kpi-s {color:#6B7A90; font-size:.8rem;}
.ct {color:#0B2545; font-weight:700; font-size:1.05rem; margin-bottom:.15rem;}
.cs {color:#6B7A90; font-size:.8rem; margin-bottom:.5rem;}
/* bar */
.bar-h {display:flex; justify-content:space-between; color:#0B2545; font-size:.85rem; margin-top:.55rem;}
.bar {position:relative; height:10px; background:#E9EEF5; border-radius:6px; margin-top:.2rem;}
.bar > div:first-child {height:100%; border-radius:6px;}
.mk {position:absolute; top:-4px; width:2px; height:18px; background:#0B2545;}
.mk-l {color:#6B7A90; font-size:.7rem;}
/* confusion matrix */
table.cm {width:100%; border-collapse:separate; border-spacing:4px; font-size:.8rem; color:#0B2545;}
table.cm td {text-align:center; border-radius:8px; padding:.55rem .2rem; font-weight:700; font-size:1.05rem;}
table.cm th {font-weight:500; color:#6B7A90; font-size:.72rem;}
/* badge */
.badge {display:inline-block; padding:.25rem .7rem; border-radius:8px; font-weight:600; font-size:.85rem;}
.b-o {background:#FDF1D6; color:#8A5A00;}
.b-b {background:#DDEEF8; color:#004C75;}
/* tab */
.stTabs [data-baseweb="tab"] p {font-weight:600; color:#6B7A90;}
.stTabs [aria-selected="true"] p {color:#0B2545;}
.stTabs [data-baseweb="tab-highlight"] {background:#0072B2;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- data
@st.cache_data
def load_data():
    with open(LABELS_FILE, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    with open(METRICS_FILE, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    rows = [{
        'question_id': qid,
        'question': v['question'],
        'best_answer': v['best_answer'],
        'reference_answer': v['reference_answer'],
        'semantic_entropy': max(float(v['semantic_entropy']), 0.0),  # buang noise numerik -1e-16
        'is_correct': bool(v['is_correct']),
        'predicted_hallucination': bool(v['predicted_hallucination']),
    } for qid, v in labels.items()]
    return pd.DataFrame(rows), metrics


if not (LABELS_FILE.exists() and METRICS_FILE.exists()):
    st.error('File data tidak ditemukan. Pastikan data/labels.json dan data/eval_metrics.json ada di repo.')
    st.stop()

df, metrics = load_data()
auroc = metrics['semantic_entropy']['AUROC']
threshold = metrics['optimal_threshold']['semantic_entropy_cutoff']

# Metrik dihitung langsung dari labels.json (positif = halusinasi = jawaban salah)
h_true = ~df['is_correct']
h_pred = df['predicted_hallucination']
TP = int((h_true & h_pred).sum())
FN = int((h_true & ~h_pred).sum())
FP = int((~h_true & h_pred).sum())
TN = int((~h_true & ~h_pred).sum())
N = len(df)
acc = (TP + TN) / N
prec = TP / (TP + FP) if TP + FP else 0
rec = TP / (TP + FN) if TP + FN else 0
f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0


# ---------------------------------------------------------------- komponen UI
def kpi(label, value, sub, color=BLUE):
    st.markdown(
        f'<div class="kpi" style="border-top:4px solid {color}"><div class="kpi-l">{label}</div>'
        f'<div class="kpi-v">{value}</div><div class="kpi-s">{sub}</div></div>',
        unsafe_allow_html=True)


def bar(label, frac, right, color=BLUE, marker=None):
    pct = max(0, min(1, frac)) * 100
    mk = f'<div class="mk" style="left:{marker * 100}%"></div>' if marker is not None else ''
    st.markdown(
        f'<div class="bar-h"><span>{label}</span><b>{right}</b></div>'
        f'<div class="bar"><div style="width:{pct}%;background:{color}"></div>{mk}</div>',
        unsafe_allow_html=True)


def card_title(title, sub=''):
    st.markdown(f'<div class="ct">{title}</div><div class="cs">{sub}</div>', unsafe_allow_html=True)


def confusion_html():
    vmax = max(TP, FN, FP, TN)

    def cell(v):
        a = 0.12 + 0.78 * v / vmax
        txt = '#fff' if a > 0.5 else NAVY
        return f'<td style="background:rgba(0,114,178,{a:.2f});color:{txt}">{v}</td>'

    return (
        '<table class="cm"><tr><th></th><th>Diprediksi<br>halusinasi</th><th>Diprediksi<br>tidak</th></tr>'
        f'<tr><th>Aktual salah</th>{cell(TP)}{cell(FN)}</tr>'
        f'<tr><th>Aktual benar</th>{cell(FP)}{cell(TN)}</tr></table>')


def se_hist():
    fig, ax = plt.subplots(figsize=(8, 3.5))
    bins = np.linspace(0, max(df['semantic_entropy'].max(), threshold), 21)
    for ok, color, name in [(True, BLUE, 'Jawaban benar'), (False, ORANGE, 'Jawaban salah')]:
        s = df[df['is_correct'] == ok]['semantic_entropy']
        ax.hist(s, bins=bins, color=color, alpha=.85, label=f'{name} (n={len(s)})',
                histtype='stepfilled' if ok else 'step', linewidth=2.2, edgecolor=color)
    ax.axvline(threshold, color=NAVY, linestyle='--', linewidth=1.2)
    ax.text(threshold + .03, ax.get_ylim()[1] * .92, f'Threshold {threshold:.2f}',
            color=NAVY, fontsize=9, fontweight='bold')
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#C9D3DF')
    ax.tick_params(colors=GREY, length=0, labelsize=9)
    ax.set_xlabel('Semantic Entropy (nats)', color=GREY, fontsize=9)
    ax.set_ylabel('Jumlah pertanyaan', color=GREY, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=NAVY)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- sidebar nav
NAV = {'▦  Dashboard': 'dash', '☰  Jelajah Pertanyaan': 'explore', 'ⓘ  Tentang Metode': 'about'}
with st.sidebar:
    st.markdown('<div class="logo"><div class="logo-i">SE</div>'
                '<div class="logo-t">Halusinasi<br>ChatGPT</div></div>', unsafe_allow_html=True)
    page = NAV[st.radio('Navigasi', list(NAV), label_visibility='collapsed')]
    st.markdown('<div class="cs" style="margin-top:2rem">Deteksi berbasis Semantic Entropy<br>'
                'pada metadata statistik berbahasa Indonesia</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- halaman
def page_dash():
    st.markdown('<h1>Dashboard</h1><div class="sub">Deteksi halusinasi respons ChatGPT pada '
                'StatMetaQA menggunakan Semantic Entropy</div>', unsafe_allow_html=True)

    # Baris 1 - KPI (kiri-atas = paling penting, pola Z)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi('Jawaban benar (aktual)', f'{1-(~df["is_correct"]).mean():.1%}',
            f'{810-int((~df["is_correct"]).sum())} dari {N} pertanyaan', ORANGE)
    with c2:
        kpi('AUROC', f'{auroc["mean"]:.3f}',
            f'CI 95%: {auroc["low"]:.3f}–{auroc["high"]:.3f} · acak = 0,5')
    with c3:
        kpi('F1-score', f'{f1:.3f}', f'Precision {prec:.2f} · Recall {rec:.2f}')
    with c4:
        kpi('Threshold optimal', f'{threshold:.3f}', 'SE di atas nilai ini → diduga halusinasi', SKY)

    st.write('')
    left, right = st.columns([3, 2])

    # Baris 2 kiri - distribusi
    with left:
        with st.container(border=True):
            card_title('Jawaban salah cenderung punya Semantic Entropy lebih tinggi',
                       'Distribusi SE per pertanyaan, dipisah berdasarkan kebenaran jawaban')
            fig = se_hist()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # Baris 2 kanan - kinerja + confusion matrix
    with right:
        with st.container(border=True):
            card_title('Kinerja deteksi pada threshold optimal', f'Positif = halusinasi · N = {N}')
            bar('AUROC (garis = acak 0,5)', auroc['mean'], f'{auroc["mean"]:.3f}', BLUE, marker=.5)
            bar('Akurasi', acc, f'{acc:.1%}')
            bar('Precision', prec, f'{prec:.1%}')
            bar('Recall', rec, f'{rec:.1%}')
            st.write('')
            st.markdown(confusion_html(), unsafe_allow_html=True)


def render_explorer(sub, key):
    st.markdown(f'<div class="cs">{len(sub)} pertanyaan · diurutkan dari SE tertinggi</div>',
                unsafe_allow_html=True)
    st.dataframe(
        sub[['question', 'semantic_entropy', 'is_correct', 'predicted_hallucination']]
        .sort_values('semantic_entropy', ascending=False)
        .rename(columns={'question': 'Pertanyaan', 'semantic_entropy': 'Semantic Entropy',
                         'is_correct': 'Jawaban benar', 'predicted_hallucination': 'Diduga halusinasi'}),
        width='stretch', height=280, hide_index=True)

    if sub.empty:
        st.write('Tidak ada data sesuai filter.')
        return
    opts = sub['question_id'] + ' — ' + sub['question'].str.slice(0, 70)
    sel = st.selectbox('Pilih pertanyaan untuk detail', opts, key=f'sel_{key}')
    row = df[df['question_id'] == sel.split(' — ')[0]].iloc[0]

    with st.container(border=True):
        d1, d2 = st.columns([3, 2])
        with d1:
            st.markdown(f"**Pertanyaan:** {row['question']}")
            st.markdown(f"**Jawaban ChatGPT:** {row['best_answer']}")
            st.markdown(f"**Jawaban referensi:** {row['reference_answer']}")
        with d2:
            bar('Semantic Entropy', row['semantic_entropy'] / max(df['semantic_entropy'].max(), 1e-9),
                f"{row['semantic_entropy']:.3f}", ORANGE if row['predicted_hallucination'] else BLUE,
                marker=threshold / max(df['semantic_entropy'].max(), 1e-9))
            st.markdown(f'<div class="mk-l">Garis vertikal = threshold {threshold:.3f}</div>',
                        unsafe_allow_html=True)
            st.write('')
            if row['predicted_hallucination']:
                st.markdown('<span class="badge b-o">⚠ Diduga halusinasi (SE > threshold)</span>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge b-b">✓ Tidak terindikasi (SE ≤ threshold)</span>',
                            unsafe_allow_html=True)
            st.write('')
            st.markdown(f'**Label aktual:** {"Jawaban benar" if row["is_correct"] else "Jawaban salah"}')


def page_explore():
    st.markdown('<h1>Jelajah Pertanyaan</h1><div class="sub">Telusuri hasil deteksi per pertanyaan</div>',
                unsafe_allow_html=True)
    search = st.text_input('Cari pertanyaan', placeholder='mis. Sakernas, PDRB, SSU…')
    base = df[df['question'].str.contains(search, case=False, na=False)] if search else df
    t1, t2, t3 = st.tabs(['Semua', 'Terdeteksi Halusinasi', 'Jawaban Benar'])
    with t1:
        render_explorer(base, 'all')
    with t2:
        render_explorer(base[base['predicted_hallucination']], 'hal')
    with t3:
        render_explorer(base[base['is_correct']], 'ok')


def page_about():
    st.markdown('<h1>Tentang Metode</h1>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            """
**Semantic Entropy** mengukur ketidakpastian pada level *makna*, bukan urutan token.
Model diminta menjawab pertanyaan yang sama beberapa kali; jawaban dikelompokkan berdasarkan
kesetaraan makna (bidirectional entailment), lalu entropi dihitung atas distribusi klaster tersebut.
SE tinggi → jawaban tidak konsisten secara makna → indikasi konfabulasi.

**Cara membaca dashboard**
- Positif = halusinasi (jawaban salah). Prediksi halusinasi jika SE > threshold optimal (Youden's J pada kurva ROC).
- AUROC 0,5 setara tebakan acak; semakin mendekati 1 semakin baik.
- Palet warna aman buta warna: biru = benar/tidak terindikasi, oranye = salah/diduga halusinasi.

**Referensi**
- Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). Detecting hallucinations in large language models
  using semantic entropy. *Nature, 630*, 625–630. https://doi.org/10.1038/s41586-024-07421-0
- Kode: https://github.com/jlko/semantic_uncertainty
- Data: StatMetaQA — https://github.com/wawatsmart/StatMetaQA
""")


{'dash': page_dash, 'explore': page_explore, 'about': page_about}[page]()
