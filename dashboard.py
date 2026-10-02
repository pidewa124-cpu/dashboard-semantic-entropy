import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------ konfigurasi
BASE = Path(__file__).parent / 'data'
LABELS_FILE = BASE / 'labels.json'
METRICS_FILE = BASE / 'eval_metrics.json'

# Palet Okabe-Ito (aman buta warna; tanpa kombinasi merah-hijau)
BLUE, ORANGE, SKY, PINK, NAVY, GREY = '#0072B2', '#E69F00', '#56B4E9', '#CC79A7', '#0B2545', '#6B7A90'

st.set_page_config(page_title='Deteksi Halusinasi - Semantic Entropy', page_icon='🧬', layout='wide')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'Inter',sans-serif;}
.stApp {background:#F4F7FB;}
.block-container {padding-top:1.6rem; padding-bottom:1rem; max-width:1400px;}
header[data-testid="stHeader"] {background:transparent;}
h1 {color:#0B2545; font-weight:800; font-size:2rem !important; margin-bottom:.1rem;}
.sub {color:#6B7A90; font-size:.9rem; margin-bottom:1rem;}
[data-testid="stSidebar"] {background:#fff; border-right:1px solid #E6ECF3;}
.logo {display:flex; gap:.6rem; align-items:center; margin:.4rem 0 1.4rem 0;}
.logo-i {background:linear-gradient(135deg,#0072B2,#56B4E9); color:#fff; font-weight:800; width:38px; height:38px;
  border-radius:10px; display:flex; align-items:center; justify-content:center;}
.logo-t {color:#0B2545; font-weight:700; line-height:1.15; font-size:.95rem;}
[data-testid="stSidebar"] [role="radiogroup"] {gap:.25rem;}
[data-testid="stSidebar"] [role="radiogroup"] label {padding:.6rem .8rem; border-radius:10px; border-left:4px solid transparent; width:100%; cursor:pointer;}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {display:none;}
[data-testid="stSidebar"] [role="radiogroup"] label p {color:#6B7A90; font-weight:500; font-size:.95rem;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#EAF4FB; border-left-color:#0072B2;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {color:#0B2545; font-weight:700;}
[data-testid="stVerticalBlockBorderWrapper"] {background:#fff; border:none !important; border-radius:16px; box-shadow:0 2px 12px rgba(11,37,69,.06); padding:.4rem .6rem;}
.kpi {background:#fff; border-radius:16px; padding:1rem 1.2rem; box-shadow:0 2px 12px rgba(11,37,69,.06); height:100%;}
.kpi-l {color:#6B7A90; font-size:.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.04em;}
.kpi-v {color:#0B2545; font-size:2rem; font-weight:800; line-height:1.25;}
.kpi-s {color:#6B7A90; font-size:.8rem;}
.ct {color:#0B2545; font-weight:700; font-size:1.05rem; margin-bottom:.15rem;}
.cs {color:#6B7A90; font-size:.8rem; margin-bottom:.5rem;}
.idea {background:linear-gradient(135deg,#0B2545,#0072B2); color:#fff; border-radius:16px; padding:1.1rem 1.4rem; margin-bottom:1rem;}
.idea small {opacity:.75; font-weight:600; text-transform:uppercase; letter-spacing:.06em; font-size:.72rem;}
.idea div {font-size:1.15rem; font-weight:600; line-height:1.4; margin-top:.2rem;}
.bar-h {display:flex; justify-content:space-between; color:#0B2545; font-size:.85rem; margin-top:.55rem;}
.bar {position:relative; height:10px; background:#E9EEF5; border-radius:6px; margin-top:.2rem;}
.bar > div:first-child {height:100%; border-radius:6px;}
.mk {position:absolute; top:-4px; width:2px; height:18px; background:#0B2545;}
.mk-l {color:#6B7A90; font-size:.72rem;}
table.cm {width:100%; border-collapse:separate; border-spacing:4px; font-size:.8rem; color:#0B2545;}
table.cm td {text-align:center; border-radius:8px; padding:.55rem .2rem; font-weight:700; font-size:1.05rem;}
table.cm th {font-weight:500; color:#6B7A90; font-size:.72rem;}
.badge {display:inline-block; padding:.25rem .7rem; border-radius:8px; font-weight:600; font-size:.82rem;}
.b-o {background:#FDF1D6; color:#8A5A00;} .b-b {background:#DDEEF8; color:#004C75;}
.flow {display:flex; gap:.5rem; align-items:stretch; flex-wrap:wrap; margin-bottom:1rem;}
.flow .st {flex:1; min-width:180px; background:#fff; border-radius:14px; padding:.9rem 1rem; box-shadow:0 2px 12px rgba(11,37,69,.06); color:#0B2545; font-size:.88rem; font-weight:600;}
.flow .st b {background:#0072B2; color:#fff; border-radius:50%; width:26px; height:26px; display:inline-flex; align-items:center; justify-content:center; margin-bottom:.4rem;}
.flow .st span {display:block; color:#6B7A90; font-weight:400; font-size:.78rem; margin-top:.2rem;}
.flow .ar {align-self:center; color:#6B7A90; font-size:1.4rem;}
.seg {display:flex; height:22px; border-radius:6px; overflow:hidden; gap:2px; margin:.4rem 0;}
.seg div {height:100%;}
.stTabs [data-baseweb="tab"] p {font-weight:600; color:#6B7A90;}
.stTabs [aria-selected="true"] p {color:#0B2545;}
.stTabs [data-baseweb="tab-highlight"] {background:#0072B2;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_data
def load_data():
    with open(LABELS_FILE, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    with open(METRICS_FILE, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    rows = [{
        'question_id': qid, 'question': v['question'], 'best_answer': v['best_answer'],
        'reference_answer': v['reference_answer'],
        'semantic_entropy': max(float(v['semantic_entropy']), 0.0),  # buang noise numerik ~1e-16
        'is_correct': bool(v['is_correct']),
    } for qid, v in labels.items()]
    return pd.DataFrame(rows), metrics


if not (LABELS_FILE.exists() and METRICS_FILE.exists()):
    st.error('File data tidak ditemukan. Pastikan data/labels.json dan data/eval_metrics.json ada di repo.')
    st.stop()

df, metrics = load_data()
auroc = metrics['semantic_entropy']['AUROC']
T0 = float(metrics['optimal_threshold']['semantic_entropy_cutoff'])
N = len(df)
SE_MAX = float(df['semantic_entropy'].max())
ACC0 = float(df['is_correct'].mean())
WRONG = ~df['is_correct']


def conf(t):
    """Positif = halusinasi (jawaban salah). Diprediksi halusinasi jika SE > t."""
    pred = df['semantic_entropy'] > t
    TP, FN = int((WRONG & pred).sum()), int((WRONG & ~pred).sum())
    FP, TN = int((~WRONG & pred).sum()), int((~WRONG & ~pred).sum())
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(TP=TP, FN=FN, FP=FP, TN=TN, prec=prec, rec=rec, f1=f1, acc=(TP + TN) / N, pred=pred)


def entropy(sizes):
    p = np.array(sizes, dtype=float) / sum(sizes)
    return float(-(p * np.log(p)).sum())


# ------------------------------------------------------------------ komponen UI
def kpi(label, value, sub, color=BLUE):
    st.markdown(f'<div class="kpi" style="border-top:4px solid {color}"><div class="kpi-l">{label}</div>'
                f'<div class="kpi-v">{value}</div><div class="kpi-s">{sub}</div></div>', unsafe_allow_html=True)


def bar(label, frac, right, color=BLUE, marker=None):
    mk = f'<div class="mk" style="left:{marker * 100}%"></div>' if marker is not None else ''
    st.markdown(f'<div class="bar-h"><span>{label}</span><b>{right}</b></div>'
                f'<div class="bar"><div style="width:{max(0, min(1, frac)) * 100}%;background:{color}"></div>{mk}</div>',
                unsafe_allow_html=True)


def title(t, s=''):
    st.markdown(f'<div class="ct">{t}</div><div class="cs">{s}</div>', unsafe_allow_html=True)


def idea(label, text):
    st.markdown(f'<div class="idea"><small>{label}</small><div>{text}</div></div>', unsafe_allow_html=True)


def header(h, sub):
    st.markdown(f'<h1>{h}</h1><div class="sub">{sub}</div>', unsafe_allow_html=True)


def badge(flagged):
    return ('<span class="badge b-o">⚠ Ditandai: perlu verifikasi</span>' if flagged
            else '<span class="badge b-b">✓ Lolos: ditampilkan langsung</span>')


def confusion_html(c):
    vmax = max(c['TP'], c['FN'], c['FP'], c['TN'])

    def cell(v):
        a = 0.12 + 0.78 * v / vmax
        return f'<td style="background:rgba(0,114,178,{a:.2f});color:{"#fff" if a > .5 else NAVY}">{v}</td>'
    return ('<table class="cm"><tr><th></th><th>Diprediksi<br>halusinasi</th><th>Diprediksi<br>tidak</th></tr>'
            f'<tr><th>Aktual salah</th>{cell(c["TP"])}{cell(c["FN"])}</tr>'
            f'<tr><th>Aktual benar</th>{cell(c["FP"])}{cell(c["TN"])}</tr></table>')


def style(ax, xl, yl):
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#C9D3DF')
    ax.tick_params(colors=GREY, length=0, labelsize=9)
    ax.set_xlabel(xl, color=GREY, fontsize=9)
    ax.set_ylabel(yl, color=GREY, fontsize=9)
    ax.figure.patch.set_alpha(0)
    ax.patch.set_alpha(0)


def show(fig):
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def fig_hist(t):
    fig, ax = plt.subplots(figsize=(7, 3.3))
    bins = np.linspace(0, SE_MAX, 21)
    for ok, col, name in [(True, BLUE, 'Jawaban benar'), (False, ORANGE, 'Jawaban salah')]:
        s = df[df['is_correct'] == ok]['semantic_entropy']
        ax.hist(s, bins=bins, color=col, alpha=.85, label=f'{name} (n={len(s)})',
                histtype='stepfilled' if ok else 'step', linewidth=2.2, edgecolor=col)
    ax.axvline(t, color=NAVY, ls='--', lw=1.2)
    ax.text(t + .03, ax.get_ylim()[1] * .9, f'Threshold {t:.2f}', color=NAVY, fontsize=9, fontweight='bold')
    ax.legend(frameon=False, fontsize=9, labelcolor=NAVY)
    style(ax, 'Semantic Entropy', 'Jumlah pertanyaan')
    return fig


def fig_roc():
    s, y = df['semantic_entropy'].values, WRONG.values
    ts = np.concatenate([[s.max() + 1], np.unique(s)[::-1], [-1]])
    tpr = [(s[y] > t).sum() / y.sum() for t in ts]
    fpr = [(s[~y] > t).sum() / (~y).sum() for t in ts]
    c = conf(T0)
    fig, ax = plt.subplots(figsize=(4.2, 3.3))
    ax.plot(fpr, tpr, color=BLUE, lw=2.5)
    ax.plot([0, 1], [0, 1], color=GREY, ls=':', lw=1.2)
    ax.text(.55, .42, 'Tebakan acak', color=GREY, fontsize=8, rotation=32)
    ax.scatter([1 - c['TN'] / (c['TN'] + c['FP'])], [c['rec']], color=ORANGE, s=60, zorder=5)
    ax.text(.32, c['rec'] - .12, f'Threshold {T0:.2f}\nTPR {c["rec"]:.2f}', color=NAVY, fontsize=8, fontweight='bold')
    style(ax, 'False positive rate', 'True positive rate')
    return fig


def fig_retention(t):
    o = df.sort_values('semantic_entropy', kind='stable')
    acc_c = o['is_correct'].expanding().mean().values * 100
    frac = np.arange(1, N + 1) / N * 100
    kept = df['semantic_entropy'] <= t
    f, a = kept.mean() * 100, (df[kept]['is_correct'].mean() * 100 if kept.any() else 0)
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    ax.plot(frac, acc_c, color=BLUE, lw=2.5)
    ax.axhline(ACC0 * 100, color=GREY, ls=':', lw=1.2)
    ax.text(2, ACC0 * 100 - 5, f'Tanpa filter: {ACC0:.0%}', color=GREY, fontsize=9)
    ax.scatter([f], [a], color=ORANGE, s=70, zorder=5)
    ax.annotate(f'{a:.0f}% benar\npada {f:.0f}% terjawab', (f, a), xytext=(f - 30, a + 4),
                color=NAVY, fontsize=9, fontweight='bold')
    ax.set_ylim(0, 105)
    style(ax, '% pertanyaan yang dijawab (SE terendah dulu)', 'Akurasi jawaban terjawab (%)')
    return fig


def example(label, row):
    with st.container(border=True):
        st.markdown(f'<div class="cs" style="margin:0">{label} · SE = {row["semantic_entropy"]:.2f}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'**{row["question"]}**')
        st.markdown(f'ChatGPT: {row["best_answer"][:200]}')
        st.markdown(f'<span style="color:#6B7A90">Referensi: {str(row["reference_answer"])[:140]}</span>',
                    unsafe_allow_html=True)


# ------------------------------------------------------------------ sidebar (nav 4 langkah)
NAV = {'1 · Metode': 'metode', '2 · Praktik': 'praktik', '3 · Evaluasi': 'evaluasi', '4 · Data': 'data'}
with st.sidebar:
    st.markdown('<div class="logo"><div class="logo-i">SE</div><div class="logo-t">Halusinasi<br>ChatGPT</div></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="cs">Ikuti alur dari atas ke bawah</div>', unsafe_allow_html=True)
    page = NAV[st.radio('Navigasi', list(NAV), label_visibility='collapsed')]


# ------------------------------------------------------------------ 1. METODE
def page_metode():
    header('1 · Metode', 'Bagaimana Semantic Entropy mendeteksi halusinasi?')
    idea('Ide utama',
         'Jika ChatGPT ditanya hal yang sama berulang kali dan jawabannya berbeda makna, '
         'besar kemungkinan ia sedang menebak (berhalusinasi).')
    st.markdown('<div class="flow">'
                '<div class="st"><b>1</b>Ajukan pertanyaan yang sama berulang<span>mis. 10 jawaban dengan suhu sampling tinggi</span></div><div class="ar">→</div>'
                '<div class="st"><b>2</b>Kelompokkan jawaban yang bermakna sama<span>entailment dua arah (bukan kemiripan kata)</span></div><div class="ar">→</div>'
                '<div class="st"><b>3</b>Hitung entropi antar klaster<span>SE = −Σ p(c) · ln p(c)</span></div><div class="ar">→</div>'
                '<div class="st"><b>4</b>Bandingkan dengan threshold<span>SE tinggi → tandai untuk diverifikasi</span></div>'
                '</div>', unsafe_allow_html=True)

    with st.container(border=True):
        title('Semakin terpecah maknanya, semakin tinggi SE',
              f'Ilustrasi 10 jawaban · garis batas = threshold optimal {T0:.2f}')
        cols = st.columns(4)
        scen = [('Semua sepakat', [10]), ('Satu menyimpang', [9, 1]),
                ('Terbelah 3 makna', [5, 3, 2]), ('Semua berbeda', [1] * 10)]
        pal = [BLUE, SKY, ORANGE, PINK, GREY]
        for col, (name, sizes) in zip(cols, scen):
            se = entropy(sizes)
            segs = ''.join(f'<div style="flex:{n};background:{pal[i % 5]}"></div>' for i, n in enumerate(sizes))
            with col:
                st.markdown(f'**{name}**<div class="seg">{segs}</div>'
                            f'<div class="cs">{len(sizes)} klaster makna · SE = <b>{se:.2f}</b></div>'
                            f'{badge(se > T0)}', unsafe_allow_html=True)

    st.write('')
    st.markdown(f'<div class="cs">Pada data Anda: ChatGPT hanya benar pada <b>{ACC0:.0%}</b> dari {N} pertanyaan '
                'metadata statistik. Kontras dua pertanyaan nyata berikut:</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        example('SE terendah (model konsisten)', df.sort_values('semantic_entropy').iloc[0])
    with b:
        example('SE tertinggi (model ragu-ragu)', df.sort_values('semantic_entropy').iloc[-1])
    st.markdown('<div class="cs">Selanjutnya: <b>2 · Praktik</b> — bagaimana SE dipakai untuk mengambil keputusan.</div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------------ 2. PRAKTIK
def page_praktik():
    header('2 · Praktik', 'Dari angka SE menjadi keputusan: jawab langsung atau verifikasi manusia?')
    t = st.slider('Threshold SE (bawaan = titik optimal dari kurva ROC)', 0.0, round(SE_MAX, 2), T0, 0.01,
                  format='%.3f')
    c = conf(t)
    kept = df['semantic_entropy'] <= t
    acc_kept = df[kept]['is_correct'].mean() if kept.any() else 0.0
    idea('Pesan praktis',
         f'Dengan threshold {t:.2f}, jawaban yang lolos {acc_kept:.0%} benar (dari {ACC0:.0%} tanpa filter), '
         f'dengan mengorbankan {(1 - kept.mean()):.0%} pertanyaan untuk diverifikasi manusia.')
    k = st.columns(4)
    with k[0]:
        kpi('Langsung dijawab', f'{kept.mean():.0%}', f'{int(kept.sum())} dari {N} pertanyaan')
    with k[1]:
        kpi('Akurasi yang lolos', f'{acc_kept:.0%}', f'naik dari {ACC0:.0%} tanpa filter')
    with k[2]:
        kpi('Halusinasi tertangkap', f'{c["rec"]:.0%}', f'{c["TP"]} dari {c["TP"] + c["FN"]} jawaban salah', ORANGE)
    with k[3]:
        kpi('Alarm palsu', f'{c["FP"]}', 'ditandai padahal jawabannya benar', SKY)
    st.write('')
    l, r = st.columns([3, 2])
    with l:
        with st.container(border=True):
            title('Semakin ketat filter, semakin akurat jawaban yang lolos',
                  'Titik oranye = posisi threshold saat ini')
            show(fig_retention(t))
    with r:
        st.markdown('<div class="cs">Contoh kasus pada threshold ini</div>', unsafe_allow_html=True)
        pred = c['pred']
        for lab, m in [('✔ Tertangkap: salah & ditandai', WRONG & pred),
                       ('✖ Alarm palsu: benar tapi ditandai', ~WRONG & pred),
                       ('✖ Terlewat: salah tapi model "yakin"', WRONG & ~pred)]:
            sub = df[m].sort_values('semantic_entropy', ascending=lab.startswith('✔') or lab.startswith('✖ Alarm'))
            if lab.startswith('✖ Terlewat'):
                sub = sub.sort_values('semantic_entropy')
            elif lab.startswith('✔') or lab.startswith('✖ Alarm'):
                sub = sub.sort_values('semantic_entropy', ascending=False)
            if len(sub):
                example(lab, sub.iloc[0])


# ------------------------------------------------------------------ 3. EVALUASI
def page_evaluasi():
    header('3 · Evaluasi', 'Seberapa andal Semantic Entropy memisahkan jawaban salah dari benar?')
    c = conf(T0)
    lolos_salah = int((WRONG & (df['semantic_entropy'] < 0.01)).sum())
    idea('Temuan',
         f'SE cukup baik memisahkan jawaban salah dan benar (AUROC {auroc["mean"]:.2f}), '
         f'tetapi {lolos_salah} jawaban salah punya SE ≈ 0: model konsisten namun keliru, sehingga tidak terdeteksi.')
    k = st.columns(4)
    with k[0]:
        kpi('AUROC', f'{auroc["mean"]:.3f}', f'CI 95%: {auroc["low"]:.3f}–{auroc["high"]:.3f} · acak = 0,5')
    with k[1]:
        kpi('Precision', f'{c["prec"]:.1%}', 'dari yang ditandai, benar-benar salah', ORANGE)
    with k[2]:
        kpi('Recall', f'{c["rec"]:.1%}', 'dari jawaban salah, berhasil ditangkap', ORANGE)
    with k[3]:
        kpi('F1-score', f'{c["f1"]:.3f}', f'Akurasi klasifikasi {c["acc"]:.1%}', SKY)
    st.write('')
    l, r = st.columns([3, 2])
    with l:
        with st.container(border=True):
            title('Jawaban salah cenderung punya SE lebih tinggi', 'Distribusi SE per pertanyaan')
            show(fig_hist(T0))
    with r:
        with st.container(border=True):
            title('Kurva ROC', f'Threshold optimal = {T0:.3f}')
            show(fig_roc())
    l2, r2 = st.columns([2, 3])
    with l2:
        with st.container(border=True):
            title('Confusion matrix', 'Positif = halusinasi (jawaban salah)')
            st.markdown(confusion_html(c), unsafe_allow_html=True)
    with r2:
        with st.container(border=True):
            title('Cara membaca & keterbatasan')
            st.markdown(
                f'- **Precision {c["prec"]:.0%}**: sekitar {1 - c["prec"]:.0%} tanda halusinasi adalah alarm palsu ({c["FP"]} kasus).\n'
                f'- **Recall {c["rec"]:.0%}**: {c["FN"]} jawaban salah lolos tanpa tanda.\n'
                f'- SE mengukur *ketidakkonsistenan*, bukan kebenaran: jawaban salah yang konsisten tidak terdeteksi.\n'
                f'- Threshold dipilih dari data yang sama sehingga bisa optimistis; validasi pada data terpisah disarankan.')


# ------------------------------------------------------------------ 4. DATA
def render_explorer(sub, key):
    st.markdown(f'<div class="cs">{len(sub)} pertanyaan · diurutkan dari SE tertinggi</div>', unsafe_allow_html=True)
    if sub.empty:
        st.write('Tidak ada data sesuai filter.')
        return
    st.dataframe(sub[['question', 'semantic_entropy', 'is_correct']].sort_values('semantic_entropy', ascending=False)
                 .rename(columns={'question': 'Pertanyaan', 'semantic_entropy': 'Semantic Entropy',
                                  'is_correct': 'Jawaban benar'}),
                 width='stretch', height=260, hide_index=True)
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
            flagged = row['semantic_entropy'] > T0
            bar('Semantic Entropy', row['semantic_entropy'] / SE_MAX, f"{row['semantic_entropy']:.3f}",
                ORANGE if flagged else BLUE, marker=T0 / SE_MAX)
            st.markdown(f'<div class="mk-l">Garis vertikal = threshold {T0:.3f}</div>', unsafe_allow_html=True)
            st.write('')
            st.markdown(badge(flagged), unsafe_allow_html=True)
            st.markdown(f'<br>**Label aktual:** {"Jawaban benar" if row["is_correct"] else "Jawaban salah"}',
                        unsafe_allow_html=True)


def page_data():
    header('4 · Data', 'Telusuri hasil deteksi per pertanyaan')
    search = st.text_input('Cari pertanyaan', placeholder='mis. Sakernas, PDRB, SSU…')
    base = df[df['question'].str.contains(search, case=False, na=False)] if search else df
    pred = base['semantic_entropy'] > T0
    t1, t2, t3 = st.tabs(['Semua', 'Ditandai halusinasi', 'Jawaban benar'])
    with t1:
        render_explorer(base, 'all')
    with t2:
        render_explorer(base[pred], 'hal')
    with t3:
        render_explorer(base[base['is_correct']], 'ok')
    st.markdown('<div class="cs" style="margin-top:1rem">Rujukan: Farquhar et al. (2024), <i>Nature</i> — '
                'doi:10.1038/s41586-024-07421-0 · Kode: github.com/jlko/semantic_uncertainty · '
                'Data: github.com/wawatsmart/StatMetaQA</div>', unsafe_allow_html=True)


{'metode': page_metode, 'praktik': page_praktik, 'evaluasi': page_evaluasi, 'data': page_data}[page]()
