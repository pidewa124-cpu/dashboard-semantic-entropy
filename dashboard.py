import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Path relatif terhadap repo (folder data/ berisi output Notebook 4)
BASE = Path(__file__).parent / 'data'
LABELS_FILE = BASE / 'labels.json'
METRICS_FILE = BASE / 'eval_metrics.json'

st.set_page_config(page_title='Deteksi Halusinasi ChatGPT - Semantic Entropy', layout='wide')


@st.cache_data
def load_data():
    with open(LABELS_FILE, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    with open(METRICS_FILE, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    rows = []
    for qid, v in labels.items():
        rows.append({
            'question_id': qid,
            'question': v['question'],
            'best_answer': v['best_answer'],
            'reference_answer': v['reference_answer'],
            'semantic_entropy': v['semantic_entropy'],
            'is_correct': bool(v['is_correct']),
            'predicted_hallucination': bool(v['predicted_hallucination']),
        })
    return pd.DataFrame(rows), metrics


if not (LABELS_FILE.exists() and METRICS_FILE.exists()):
    st.error('File data tidak ditemukan. Pastikan data/labels.json dan data/eval_metrics.json ada di repo.')
    st.stop()

df, metrics = load_data()
se_metrics = metrics['semantic_entropy']
threshold = metrics['optimal_threshold']['semantic_entropy_cutoff']

st.title('Deteksi Halusinasi Respon ChatGPT — Semantic Entropy')
st.caption('Metadata Statistik Berbahasa Indonesia')

# --- Ringkasan metrik ---
c1, c2, c3, c4 = st.columns(4)
c1.metric('Total Pertanyaan', len(df))
c2.metric('Akurasi best_answer', f"{df['is_correct'].mean():.2%}")
c3.metric('AUROC (Semantic Entropy)', f"{se_metrics['AUROC']['mean']:.3f}")
c4.metric('Threshold Optimal (SE)', f"{threshold:.3f}")

st.divider()

# --- Sidebar: filter ---
st.sidebar.header('Filter')
filter_opt = st.sidebar.radio(
    'Tampilkan', ['Semua', 'Terdeteksi Halusinasi', 'Jawaban Benar'])
search = st.sidebar.text_input('Cari pertanyaan')

filtered = df.copy()
if filter_opt == 'Terdeteksi Halusinasi':
    filtered = filtered[filtered['predicted_hallucination']]
elif filter_opt == 'Jawaban Benar':
    filtered = filtered[filtered['is_correct']]
if search:
    filtered = filtered[filtered['question'].str.contains(search, case=False, na=False)]

# --- Distribusi SE ---
st.subheader('Distribusi Semantic Entropy')
fig, ax = plt.subplots(figsize=(8, 3))
for label, color in [(True, 'tab:green'), (False, 'tab:red')]:
    subset = df[df['is_correct'] == label]['semantic_entropy']
    ax.hist(subset, bins=20, alpha=0.6, color=color,
            label=f"{'Benar' if label else 'Salah'} (n={len(subset)})")
ax.axvline(threshold, color='black', linestyle='--', label=f'Threshold={threshold:.2f}')
ax.set_xlabel('Semantic Entropy')
ax.set_ylabel('Jumlah Pertanyaan')
ax.legend()
st.pyplot(fig)
plt.close(fig)

st.divider()

# --- Tabel ---
st.subheader(f'Daftar Pertanyaan ({len(filtered)} dari {len(df)})')
st.dataframe(
    filtered[['question', 'semantic_entropy', 'is_correct', 'predicted_hallucination']]
    .sort_values('semantic_entropy', ascending=False),
    width='stretch',
)

# --- Detail per pertanyaan ---
st.subheader('Detail Pertanyaan')
options = filtered['question_id'] + ' — ' + filtered['question'].str.slice(0, 60)
if len(options) > 0:
    selected = st.selectbox('Pilih pertanyaan', options)
    sel_qid = selected.split(' — ')[0]
    row = df[df['question_id'] == sel_qid].iloc[0]

    st.write(f"**Pertanyaan:** {row['question']}")
    st.write(f"**Jawaban ChatGPT (best_answer):** {row['best_answer']}")
    st.write(f"**Jawaban Referensi:** {row['reference_answer']}")
    st.write(f"**Semantic Entropy:** {row['semantic_entropy']:.4f}")

    if row['predicted_hallucination']:
        st.error('⚠️ Terindikasi Halusinasi (SE di atas threshold)')
    else:
        st.success('✅ Tidak terindikasi halusinasi (SE di bawah threshold)')

    if row['is_correct']:
        st.info('Label aktual: Jawaban Benar')
    else:
        st.warning('Label aktual: Jawaban Salah')
else:
    st.write('Tidak ada data sesuai filter.')
