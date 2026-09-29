import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ============ KONFIGURASI (EDIT BAGIAN INI) ============
JUDUL = "Deteksi Halusinasi Respon ChatGPT pada Metadata Statistik Berbahasa Indonesia"
SUBJUDUL = "Menggunakan Semantic Entropy"
NAMA = "Nama Mahasiswa"
PEMBIMBING = "Nama Dosen Pembimbing"
INSTITUSI = "Program Studi / Universitas"
MODEL_LLM = "ChatGPT (isi versi model, mis. gpt-4o-mini)"
JUMLAH_SAMPEL_SAMPEL = "M = 10"  # jumlah jawaban sampel per pertanyaan (sesuaikan)
METODE_LABEL = "Isi metode pelabelan is_correct (manual / LLM-judge / lainnya)"
# =======================================================

BASE = Path(__file__).parent / "data"
LABELS_FILE = BASE / "labels.json"
METRICS_FILE = BASE / "eval_metrics.json"

st.set_page_config(page_title="Deteksi Halusinasi - Semantic Entropy",
                   layout="wide", initial_sidebar_state="collapsed")

# Label navigasi (urutan = urutan bagian; id anchor = sec-0, sec-1, ...)
NAV = ["Ringkasan", "Latar Belakang", "Rumusan & Tujuan", "Data", "Metodologi",
       "Hasil Utama", "Distribusi SE", "ROC & Rejection", "Kasus", "Analisis",
       "Kesimpulan", "Referensi"]

# ---------- CSS halaman ----------
st.markdown(
    """
<style>
[data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {display:none;}
.block-container {padding-top: 4rem; padding-left: 250px; max-width: 1250px;}
@media (max-width: 900px) {.block-container {padding-left: 1rem; padding-top: 7rem;}}
[id^="sec-"] {scroll-margin-top: 80px;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Navigasi kotak (disuntik ke halaman utama, dengan scroll-spy) ----------
NAV_JS = """
<script>
(function () {
  const labels = __LABELS__;
  const P = window.parent, D = P.document;
  ['hal-nav', 'hal-nav-style'].forEach(id => { const o = D.getElementById(id); if (o) o.remove(); });

  const st = D.createElement('style');
  st.id = 'hal-nav-style';
  st.textContent = `
    #hal-nav{position:fixed;top:72px;left:16px;width:210px;bottom:16px;z-index:1000002;
      display:flex;flex-direction:column;gap:6px;padding:10px;background:#ffffff;
      border:1px solid #d5dbe6;border-radius:12px;overflow-y:auto;
      box-shadow:0 2px 8px rgba(0,0,0,.06);}
    #hal-nav button{border:none;border-radius:8px;padding:10px 12px;text-align:left;
      font:600 13px/1.2 system-ui,sans-serif;cursor:pointer;background:#e6ebf4;color:#1f3a63;
      transition:background .15s,color .15s;}
    #hal-nav button:hover{background:#cfd8e8;}
    #hal-nav button.active{background:#1f3a63;color:#ffffff;}
    @media (max-width:900px){
      #hal-nav{top:0;left:0;right:0;bottom:auto;width:auto;flex-direction:row;border-radius:0;
        overflow-x:auto;overflow-y:hidden;white-space:nowrap;}
      #hal-nav button{flex:0 0 auto;}
    }
  `;
  D.head.appendChild(st);

  const nav = D.createElement('div');
  nav.id = 'hal-nav';
  labels.forEach((t, i) => {
    const b = D.createElement('button');
    b.textContent = t;
    b.dataset.i = i;
    b.onclick = () => {
      const el = D.getElementById('sec-' + i);
      if (el) el.scrollIntoView({behavior: 'smooth', block: 'start'});
      setActive(i);
    };
    nav.appendChild(b);
  });
  D.body.appendChild(nav);

  function setActive(i) {
    nav.querySelectorAll('button').forEach(b => b.classList.toggle('active', +b.dataset.i === i));
    const a = nav.querySelector('button.active');
    if (a) a.scrollIntoView({inline: 'nearest', block: 'nearest'});
  }

  function spy() {
    let cur = 0;
    for (let i = 0; i < labels.length; i++) {
      const el = D.getElementById('sec-' + i);
      if (el && el.getBoundingClientRect().top <= 140) cur = i;
    }
    setActive(cur);
  }
  D.addEventListener('scroll', spy, true);
  setTimeout(spy, 300);
})();
</script>
"""
components.html(NAV_JS.replace("__LABELS__", json.dumps(NAV)), height=0)


# ---------- Data ----------
@st.cache_data
def load_data():
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        labels = json.load(f)
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    rows = []
    for qid, v in labels.items():
        se = max(float(v["semantic_entropy"]), 0.0)  # buang noise floating point
        correct = bool(v["is_correct"])
        pred = bool(v["predicted_hallucination"])
        halu = not correct  # kelas positif = jawaban salah
        kasus = ("TP" if pred else "FN") if halu else ("FP" if pred else "TN")
        rows.append({
            "question_id": qid, "question": v["question"],
            "best_answer": v["best_answer"], "reference_answer": v["reference_answer"],
            "semantic_entropy": se, "is_correct": correct,
            "predicted_hallucination": pred, "kasus": kasus,
        })
    return pd.DataFrame(rows), metrics


def roc_points(y_true, score):
    order = np.argsort(-score)
    y = np.asarray(y_true)[order]
    tpr = np.r_[0, np.cumsum(y) / y.sum()]
    fpr = np.r_[0, np.cumsum(1 - y) / (1 - y).sum()]
    return fpr, tpr


if not (LABELS_FILE.exists() and METRICS_FILE.exists()):
    st.error("File data tidak ditemukan. Pastikan data/labels.json dan data/eval_metrics.json ada.")
    st.stop()

df, metrics = load_data()
se_m = metrics["semantic_entropy"]
opt = metrics["optimal_threshold"]
thr = opt["semantic_entropy_cutoff"]
cm = opt["confusion_matrix"]
n = len(df)
acc_chatgpt = df["is_correct"].mean()
n_benar = int(df["is_correct"].sum())
n_salah = n - n_benar

# =============== 0. RINGKASAN ===============
st.title(JUDUL, anchor="sec-0")
st.subheader(SUBJUDUL)
st.caption(f"{NAMA} · {INSTITUSI} · Pembimbing: {PEMBIMBING}")
if n != sum(cm.values()):
    st.warning(f"labels.json berisi {n} baris, sedangkan confusion matrix di eval_metrics.json "
               f"berjumlah {sum(cm.values())}. Pastikan keduanya berasal dari run yang sama.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah pertanyaan", n)
c2.metric("Akurasi ChatGPT", f"{acc_chatgpt:.1%}")
c3.metric("AUROC Semantic Entropy", f"{se_m['AUROC']['mean']:.3f}")
c4.metric("F1 pada threshold optimal", f"{opt['f1_score']:.3f}")
st.info("Pertanyaan penelitian: seberapa baik ketidakpastian semantik (semantic entropy) "
        "dapat menandai jawaban ChatGPT yang salah pada metadata statistik BPS berbahasa Indonesia?")
st.divider()

# =============== 1. LATAR BELAKANG ===============
st.header("1. Latar Belakang", anchor="sec-1")
st.markdown(
    """
- **Halusinasi LLM** adalah jawaban yang fasih dan meyakinkan tetapi salah. Pada data resmi
  (statistik), satu angka atau definisi yang keliru dapat menyesatkan analisis dan kebijakan.
- **Metadata statistik** (definisi indikator, sumber data, metode survei) menuntut jawaban yang
  presisi, misalnya tahun pelaksanaan, jumlah sampel, atau kode klasifikasi.
- Metode deteksi halusinasi umumnya diuji pada bahasa Inggris. **Bukti untuk bahasa Indonesia
  dan domain statistik resmi masih terbatas.**
- **Semantic entropy** (Farquhar et al., 2024) mengukur ketidakpastian pada level *makna*, bukan
  urutan kata, sehingga cocok mendeteksi *confabulation* (jawaban salah yang berubah-ubah).
    """
)
ex = df[(~df["is_correct"]) & (df["semantic_entropy"] > 1.5)].head(1)
if len(ex):
    r = ex.iloc[0]
    st.markdown("**Contoh nyata dari data (jawaban salah dengan SE tinggi):**")
    st.write(f"**Pertanyaan:** {r['question']}")
    st.write(f"**Jawaban ChatGPT:** {r['best_answer']}")
    st.write(f"**Referensi:** {r['reference_answer']}")
    st.write(f"**Semantic entropy:** {r['semantic_entropy']:.3f}")
st.divider()

# =============== 2. RUMUSAN & TUJUAN ===============
st.header("2. Rumusan Masalah & Tujuan", anchor="sec-2")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Rumusan masalah")
    st.markdown(
        """
1. Seberapa baik semantic entropy membedakan jawaban benar dan salah ChatGPT pada metadata statistik berbahasa Indonesia?
2. Pada threshold berapa semantic entropy paling optimal menandai halusinasi?
3. Pada jenis kasus apa semantic entropy gagal mendeteksi halusinasi?
        """
    )
with c2:
    st.subheader("Tujuan")
    st.markdown(
        """
1. Mengukur kinerja semantic entropy (AUROC, AURAC) sebagai detektor halusinasi.
2. Menentukan threshold optimal dan mengevaluasinya (confusion matrix, precision, recall, F1).
3. Menganalisis kasus deteksi berhasil dan gagal.
        """
    )
st.caption("Ruang lingkup: hanya metode semantic entropy, tanpa pembanding (P(True), embedding, dll).")
st.divider()

# =============== 3. DATA ===============
st.header("3. Data", anchor="sec-3")
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        f"""
- **Sumber:** StatMetaQA (tanya-jawab metadata statistik BPS berbahasa Indonesia).
- **Sampel penelitian:** {n} pertanyaan.
- **Model:** {MODEL_LLM}
- **Pelabelan `is_correct`:** {METODE_LABEL}
        """
    )
    st.metric("Jawaban benar", f"{n_benar} ({n_benar / n:.1%})")
    st.metric("Jawaban salah (kelas positif / halusinasi)", f"{n_salah} ({n_salah / n:.1%})")
with c2:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["Benar", "Salah"], [n_benar, n_salah], color=["tab:green", "tab:red"])
    for i, v in enumerate([n_benar, n_salah]):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_ylabel("Jumlah pertanyaan")
    ax.set_title("Distribusi label jawaban ChatGPT")
    st.pyplot(fig)
    plt.close(fig)
st.divider()

# =============== 4. METODOLOGI ===============
st.header("4. Metodologi", anchor="sec-4")
steps = [
    ("1. Pertanyaan", "Setiap pertanyaan dikirim ke ChatGPT dalam bahasa Indonesia."),
    ("2. Jawaban utama", "Satu jawaban dengan suhu rendah sebagai `best_answer` yang dinilai benar/salah."),
    ("3. Jawaban sampel", f"Beberapa jawaban dengan suhu tinggi ({JUMLAH_SAMPEL_SAMPEL}) untuk melihat sebaran makna."),
    ("4. Klaster makna", "Jawaban dikelompokkan jika saling *entail* dua arah (setara secara makna)."),
    ("5. Semantic entropy", "Entropi distribusi klaster: SE tinggi = jawaban terpecah ke banyak makna."),
    ("6. Threshold", "Titik potong dipilih dengan kriteria Youden (TPR − FPR maksimum)."),
    ("7. Evaluasi", "SE > threshold = terindikasi halusinasi; dibandingkan dengan label benar/salah."),
]
idx = st.select_slider("Geser untuk melihat tahapan", options=list(range(len(steps))),
                       format_func=lambda i: steps[i][0])
st.success(f"**{steps[idx][0]}**: {steps[idx][1]}")
st.markdown("**Rumus:** " + r"$SE(x) = -\sum_{c} p(c \mid x)\,\log p(c \mid x)$"
            + ", dengan $c$ = klaster makna dan $p(c \mid x)$ = proporsi jawaban sampel pada klaster tersebut.")
st.caption("Implementasi mengacu pada github.com/jlko/semantic_uncertainty. Sesuaikan detail model NLI dan parameter dengan yang Anda pakai.")
st.divider()

# =============== 5. HASIL UTAMA ===============
st.header("5. Hasil Utama", anchor="sec-5")
c1, c2, c3, c4 = st.columns(4)
c1.metric("AUROC", f"{se_m['AUROC']['mean']:.3f}",
          f"CI {se_m['AUROC']['low']:.3f}-{se_m['AUROC']['high']:.3f}", delta_color="off")
c2.metric("AURAC", f"{se_m['AURAC']['mean']:.3f}",
          f"CI {se_m['AURAC']['low']:.3f}-{se_m['AURAC']['high']:.3f}", delta_color="off")
c3.metric("Threshold optimal (SE)", f"{thr:.3f}")
c4.metric("Akurasi deteksi", f"{opt['accuracy']:.1%}")
c1, c2, c3 = st.columns(3)
c1.metric("Precision", f"{opt['precision']:.3f}")
c2.metric("Recall", f"{opt['recall']:.3f}")
c3.metric("F1", f"{opt['f1_score']:.3f}")

st.subheader("Confusion matrix")
mat = np.array([[cm["TP"], cm["FN"]], [cm["FP"], cm["TN"]]])
fig, ax = plt.subplots(figsize=(4.5, 3.5))
ax.imshow(mat, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=14,
                color="white" if mat[i, j] > mat.max() / 2 else "black")
ax.set_xticks([0, 1], ["Diprediksi halusinasi", "Diprediksi normal"])
ax.set_yticks([0, 1], ["Aktual salah", "Aktual benar"])
st.pyplot(fig)
plt.close(fig)
st.caption("TP: salah & terdeteksi · FN: salah tapi lolos · FP: benar tapi ditandai · TN: benar & lolos.")
st.info(f"Interpretasi: AUROC {se_m['AUROC']['mean']:.2f} berarti SE punya daya pembeda sedang. "
        f"Dari {cm['TP'] + cm['FN']} jawaban salah, {cm['TP']} tertangkap dan {cm['FN']} lolos.")
st.divider()

# =============== 6. DISTRIBUSI SE ===============
st.header("6. Distribusi Semantic Entropy", anchor="sec-6")
fig, ax = plt.subplots(figsize=(8, 3.5))
for label, color in [(True, "tab:green"), (False, "tab:red")]:
    s = df[df["is_correct"] == label]["semantic_entropy"]
    ax.hist(s, bins=20, alpha=0.6, color=color, label=f"{'Benar' if label else 'Salah'} (n={len(s)})")
ax.axvline(thr, color="black", linestyle="--", label=f"Threshold = {thr:.2f}")
ax.set_xlabel("Semantic Entropy")
ax.set_ylabel("Jumlah pertanyaan")
ax.legend()
st.pyplot(fig)
plt.close(fig)
c1, c2 = st.columns(2)
c1.metric("Rata-rata SE (jawaban benar)", f"{df[df['is_correct']]['semantic_entropy'].mean():.3f}")
c2.metric("Rata-rata SE (jawaban salah)", f"{df[~df['is_correct']]['semantic_entropy'].mean():.3f}")
st.caption("Tumpang tindih kedua sebaran menjelaskan mengapa AUROC tidak mendekati 1.")
st.divider()

# =============== 7. ROC & REJECTION ===============
st.header("7. Kurva ROC & Rejection Accuracy", anchor="sec-7")
c1, c2 = st.columns(2)
with c1:
    y = (~df["is_correct"]).astype(int).values
    fpr, tpr = roc_points(y, df["semantic_entropy"].values)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label="Semantic Entropy")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Acak")
    ax.scatter([opt["fpr"]], [opt["tpr"]], color="red", zorder=3, label="Threshold optimal")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Kurva ROC")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
with c2:
    keys = ["accuracy_at_0.8", "accuracy_at_0.9", "accuracy_at_0.95", "accuracy_at_1.0"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(["80%", "90%", "95%", "100%"], [se_m[k] for k in keys], marker="o")
    ax.set_xlabel("Proporsi pertanyaan yang dijawab (paling pasti dulu)")
    ax.set_ylabel("Akurasi ChatGPT")
    ax.set_title("Akurasi setelah membuang pertanyaan tidak pasti")
    st.pyplot(fig)
    plt.close(fig)
st.caption(f"Jika 20% pertanyaan paling tidak pasti dibuang, akurasi naik dari {se_m['accuracy_at_1.0']:.1%} "
           f"menjadi {se_m['accuracy_at_0.8']:.1%}.")
st.divider()

# =============== 8. PENJELAJAH KASUS ===============
st.header("8. Penjelajah Kasus", anchor="sec-8")
ket = {"TP": "TP: salah & terdeteksi", "FP": "FP: benar tapi ditandai",
       "FN": "FN: salah tapi lolos", "TN": "TN: benar & lolos"}
c1, c2 = st.columns([2, 1])
pilih = c1.multiselect("Jenis kasus", list(ket), default=["FN", "FP"], format_func=lambda k: ket[k])
cari = c2.text_input("Cari pertanyaan")
f = df[df["kasus"].isin(pilih)]
if cari:
    f = f[f["question"].str.contains(cari, case=False, na=False)]
st.write(f"**Daftar pertanyaan ({len(f)} dari {n})**")
st.dataframe(
    f[["question", "semantic_entropy", "is_correct", "kasus"]].sort_values("semantic_entropy", ascending=False),
    width="stretch", height=300,
)
if len(f):
    opts = f["question_id"] + " — " + f["question"].str.slice(0, 60)
    sel = st.selectbox("Pilih pertanyaan", opts)
    r = df[df["question_id"] == sel.split(" — ")[0]].iloc[0]
    st.write(f"**Pertanyaan:** {r['question']}")
    st.write(f"**Jawaban ChatGPT:** {r['best_answer']}")
    st.write(f"**Jawaban referensi:** {r['reference_answer']}")
    st.write(f"**Semantic entropy:** {r['semantic_entropy']:.4f} (threshold {thr:.3f})")
    st.info(ket[r["kasus"]])
st.divider()

# =============== 9. ANALISIS & KETERBATASAN ===============
st.header("9. Analisis & Keterbatasan", anchor="sec-9")
fn = df[df["kasus"] == "FN"]
yakin = fn[fn["semantic_entropy"] < 0.05]
c1, c2, c3 = st.columns(3)
c1.metric("False negative (salah tapi lolos)", len(fn))
c2.metric("Salah dengan SE ≈ 0 (yakin tapi salah)", len(yakin))
c3.metric("False positive (benar tapi ditandai)", int((df["kasus"] == "FP").sum()))
st.markdown(
    """
**Temuan**
- SE efektif untuk jawaban yang berubah-ubah antar sampel (confabulation), tetapi tidak untuk
  kesalahan yang **konsisten**: model selalu menjawab hal yang sama, sehingga SE ≈ 0 walau salah.
- Pertanyaan berjawaban angka atau tahun spesifik cenderung SE tinggi karena jawaban sampel bervariasi.

**Keterbatasan**
- Hanya satu model LLM dan satu domain (metadata statistik BPS).
- Label kebenaran bergantung pada metode pelabelan dan sebagian jawaban bersifat parsial.
- Tanpa metode pembanding, sehingga tidak dapat disimpulkan SE lebih baik dari metode lain.
- Ukuran sampel terbatas; interval kepercayaan AUROC cukup lebar.
    """
)
st.markdown("**Contoh yakin tapi salah:**")
st.dataframe(yakin[["question", "best_answer", "reference_answer", "semantic_entropy"]].head(5), width="stretch")
st.divider()

# =============== 10. KESIMPULAN & SARAN ===============
st.header("10. Kesimpulan & Saran", anchor="sec-10")
st.markdown(
    f"""
**Kesimpulan**
1. Semantic entropy memiliki daya pembeda **sedang** (AUROC {se_m['AUROC']['mean']:.3f}, AURAC {se_m['AURAC']['mean']:.3f}).
2. Threshold optimal SE = **{thr:.3f}**, menghasilkan precision {opt['precision']:.3f}, recall {opt['recall']:.3f}, F1 {opt['f1_score']:.3f}.
3. Kegagalan utama terjadi pada kesalahan konsisten (SE rendah tetapi jawaban salah).

**Saran**
- Bandingkan dengan metode lain (P(True), embedding, self-consistency) pada data yang sama.
- Gunakan model NLI khusus bahasa Indonesia untuk klasterisasi makna.
- Tambah jumlah sampel, model LLM, dan jenis pertanyaan; validasi label oleh lebih dari satu penilai.
- Gabungkan SE dengan retrieval (RAG) pada dokumen metadata BPS untuk menekan kesalahan konsisten.
    """
)
st.divider()

# =============== 11. REFERENSI ===============
st.header("11. Referensi", anchor="sec-11")
st.markdown(
    """
1. Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). Detecting hallucinations in large language models using semantic entropy. *Nature*, 630, 625–630. https://www.nature.com/articles/s41586-024-07421-0
2. Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. *ICLR 2023*. https://arxiv.org/abs/2302.09664
3. Ji, Z. et al. (2023). Survey of Hallucination in Natural Language Generation. *ACM Computing Surveys*, 55(12).
4. Kode semantic entropy: https://github.com/jlko/semantic_uncertainty
5. Dataset StatMetaQA: https://github.com/wawatsmart/StatMetaQA
    """
)
st.caption("Tambahkan referensi lain yang Anda kutip di naskah skripsi.")
