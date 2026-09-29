import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ============ KONFIGURASI (EDIT BAGIAN INI) ============
JUDUL = "Deteksi Halusinasi Respon ChatGPT pada Metadata Statistik Berbahasa Indonesia"
SUBJUDUL = "Menggunakan Semantic Entropy"
NAMA = "Nama Mahasiswa"
PEMBIMBING = "Nama Dosen Pembimbing"
INSTITUSI = "Program Studi / Universitas"
MODEL_LLM = "ChatGPT (isi versi model, mis. gpt-4o-mini)"
JUMLAH_SAMPEL_SAMPEL = "M = 10"          # jumlah jawaban sampel per pertanyaan (sesuaikan)
METODE_LABEL = "Isi metode pelabelan is_correct (manual / LLM-judge / lainnya)"
# =======================================================

BASE = Path(__file__).parent / "data"
LABELS_FILE = BASE / "labels.json"
METRICS_FILE = BASE / "eval_metrics.json"

st.set_page_config(page_title="Deteksi Halusinasi - Semantic Entropy", layout="wide")


@st.cache_data
def load_data():
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        labels = json.load(f)
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    rows = []
    for qid, v in labels.items():
        se = max(float(v["semantic_entropy"]), 0.0)  # buang noise floating point (-2e-16)
        correct = bool(v["is_correct"])
        pred = bool(v["predicted_hallucination"])
        halu = not correct  # kelas positif = jawaban salah (halusinasi)
        if halu and pred:
            kasus = "TP"
        elif (not halu) and pred:
            kasus = "FP"
        elif halu and (not pred):
            kasus = "FN"
        else:
            kasus = "TN"
        rows.append({
            "question_id": qid,
            "question": v["question"],
            "best_answer": v["best_answer"],
            "reference_answer": v["reference_answer"],
            "semantic_entropy": se,
            "is_correct": correct,
            "predicted_hallucination": pred,
            "kasus": kasus,
        })
    return pd.DataFrame(rows), metrics


def roc_points(y_true, score):
    """ROC manual (tanpa sklearn). y_true=1 untuk halusinasi."""
    order = np.argsort(-score)
    y = np.asarray(y_true)[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = np.r_[0, tps / y.sum()]
    fpr = np.r_[0, fps / (1 - y).sum()]
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
n_salah = int((~df["is_correct"]).sum())

# ---------- Navigasi ----------
SECTIONS = [
    "0. Ringkasan",
    "1. Latar Belakang",
    "2. Rumusan Masalah & Tujuan",
    "3. Data",
    "4. Metodologi",
    "5. Hasil Utama",
    "6. Distribusi Semantic Entropy",
    "7. Kurva ROC & Rejection",
    "8. Penjelajah Kasus",
    "9. Analisis & Keterbatasan",
    "10. Kesimpulan & Saran",
    "11. Referensi",
]
st.sidebar.title("Alur Penelitian")
sec = st.sidebar.radio("Bagian", SECTIONS, label_visibility="collapsed")
st.sidebar.caption(f"{NAMA}\n\nPembimbing: {PEMBIMBING}")

if n != sum(cm.values()):
    st.sidebar.warning(
        f"labels.json berisi {n} baris, sedangkan confusion matrix di eval_metrics.json "
        f"berjumlah {sum(cm.values())}. Pastikan keduanya berasal dari run yang sama."
    )

# ---------- 0 ----------
if sec == SECTIONS[0]:
    st.title(JUDUL)
    st.subheader(SUBJUDUL)
    st.caption(f"{NAMA} · {INSTITUSI} · Pembimbing: {PEMBIMBING}")
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah pertanyaan", n)
    c2.metric("Akurasi ChatGPT", f"{acc_chatgpt:.1%}")
    c3.metric("AUROC Semantic Entropy", f"{se_m['AUROC']['mean']:.3f}")
    c4.metric("F1 pada threshold optimal", f"{opt['f1_score']:.3f}")
    st.info(
        "Pertanyaan penelitian: seberapa baik ketidakpastian semantik (semantic entropy) "
        "dapat menandai jawaban ChatGPT yang salah pada metadata statistik BPS berbahasa Indonesia?"
    )

# ---------- 1 ----------
elif sec == SECTIONS[1]:
    st.header("1. Latar Belakang")
    st.markdown(
        """
- **Halusinasi LLM** adalah jawaban yang fasih dan meyakinkan tetapi salah. Pada data resmi
  (statistik), satu angka atau definisi yang keliru dapat menyesatkan analisis dan kebijakan.
- **Metadata statistik** (definisi indikator, sumber data, metode survei) menuntut jawaban yang
  presisi, misalnya tahun pelaksanaan, jumlah sampel, atau kode klasifikasi.
- Metode deteksi halusinasi yang ada umumnya diuji pada bahasa Inggris. **Bukti untuk bahasa
  Indonesia dan domain statistik resmi masih terbatas.**
- **Semantic entropy** (Farquhar et al., 2024) mengukur ketidakpastian pada level *makna*, bukan
  urutan kata, sehingga cocok mendeteksi *confabulation* (jawaban salah yang berubah-ubah).
        """
    )
    st.markdown("**Contoh nyata dari data (jawaban salah dengan SE tinggi):**")
    ex = df[(~df["is_correct"]) & (df["semantic_entropy"] > 1.5)].head(1)
    if len(ex):
        r = ex.iloc[0]
        st.write(f"**Pertanyaan:** {r['question']}")
        st.write(f"**Jawaban ChatGPT:** {r['best_answer']}")
        st.write(f"**Referensi:** {r['reference_answer']}")
        st.write(f"**Semantic entropy:** {r['semantic_entropy']:.3f}")

# ---------- 2 ----------
elif sec == SECTIONS[2]:
    st.header("2. Rumusan Masalah & Tujuan")
    st.subheader("Rumusan masalah")
    st.markdown(
        """
1. Seberapa baik semantic entropy membedakan jawaban benar dan salah ChatGPT pada metadata statistik berbahasa Indonesia?
2. Pada nilai ambang (threshold) berapa semantic entropy paling optimal menandai halusinasi?
3. Pada jenis kasus apa semantic entropy gagal mendeteksi halusinasi?
        """
    )
    st.subheader("Tujuan")
    st.markdown(
        """
1. Mengukur kinerja semantic entropy (AUROC, AURAC) sebagai detektor halusinasi.
2. Menentukan threshold optimal dan mengevaluasinya (confusion matrix, precision, recall, F1).
3. Menganalisis kasus deteksi berhasil dan gagal.
        """
    )
    st.caption("Ruang lingkup: hanya metode semantic entropy, tanpa pembanding (P(True), embedding, dll).")

# ---------- 3 ----------
elif sec == SECTIONS[3]:
    st.header("3. Data")
    st.markdown(
        f"""
- **Sumber:** StatMetaQA (pasangan tanya-jawab metadata statistik BPS, berbahasa Indonesia).
- **Sampel penelitian:** {n} pertanyaan.
- **Model:** {MODEL_LLM}
- **Pelabelan `is_correct`:** {METODE_LABEL}
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(4, 3))
        vals = [int(df["is_correct"].sum()), n_salah]
        ax.bar(["Benar", "Salah"], vals, color=["tab:green", "tab:red"])
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha="center", va="bottom")
        ax.set_ylabel("Jumlah pertanyaan")
        ax.set_title("Distribusi label jawaban ChatGPT")
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        st.metric("Jawaban benar", f"{vals[0]} ({vals[0]/n:.1%})")
        st.metric("Jawaban salah (kelas positif / halusinasi)", f"{vals[1]} ({vals[1]/n:.1%})")
        st.caption("Kelas positif = jawaban salah. Kelas ini dideteksi oleh SE tinggi.")

# ---------- 4 ----------
elif sec == SECTIONS[4]:
    st.header("4. Metodologi")
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
    st.markdown(
        "**Rumus:** " + r"$SE(x) = -\sum_{c} p(c \mid x)\,\log p(c \mid x)$"
        + ", dengan $c$ = klaster makna dan $p(c \mid x)$ = proporsi jawaban sampel pada klaster tersebut."
    )
    st.caption("Implementasi mengacu pada github.com/jlko/semantic_uncertainty. Sesuaikan detail model NLI dan parameter dengan yang Anda pakai.")

# ---------- 5 ----------
elif sec == SECTIONS[5]:
    st.header("5. Hasil Utama")
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
            ax.text(j, i, mat[i, j], ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1], ["Diprediksi halusinasi", "Diprediksi normal"])
    ax.set_yticks([0, 1], ["Aktual salah", "Aktual benar"])
    st.pyplot(fig)
    plt.close(fig)
    st.caption("TP: salah & terdeteksi · FN: salah tapi lolos · FP: benar tapi ditandai · TN: benar & lolos.")
    st.info(
        f"Interpretasi: AUROC {se_m['AUROC']['mean']:.2f} berarti SE punya daya pembeda sedang. "
        f"Dari {cm['TP'] + cm['FN']} jawaban salah, {cm['TP']} tertangkap dan {cm['FN']} lolos."
    )

# ---------- 6 ----------
elif sec == SECTIONS[6]:
    st.header("6. Distribusi Semantic Entropy")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for label, color in [(True, "tab:green"), (False, "tab:red")]:
        s = df[df["is_correct"] == label]["semantic_entropy"]
        ax.hist(s, bins=20, alpha=0.6, color=color,
                label=f"{'Benar' if label else 'Salah'} (n={len(s)})")
    ax.axvline(thr, color="black", linestyle="--", label=f"Threshold = {thr:.2f}")
    ax.set_xlabel("Semantic Entropy")
    ax.set_ylabel("Jumlah pertanyaan")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    m_benar = df[df["is_correct"]]["semantic_entropy"].mean()
    m_salah = df[~df["is_correct"]]["semantic_entropy"].mean()
    c1, c2 = st.columns(2)
    c1.metric("Rata-rata SE (jawaban benar)", f"{m_benar:.3f}")
    c2.metric("Rata-rata SE (jawaban salah)", f"{m_salah:.3f}")
    st.caption("Tumpang tindih kedua sebaran menjelaskan mengapa AUROC tidak mendekati 1.")

# ---------- 7 ----------
elif sec == SECTIONS[7]:
    st.header("7. Kurva ROC & Rejection Accuracy")
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
        xs = ["80%", "90%", "95%", "100%"]
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(xs, [se_m[k] for k in keys], marker="o")
        ax.set_xlabel("Proporsi pertanyaan yang dijawab (paling pasti dulu)")
        ax.set_ylabel("Akurasi ChatGPT")
        ax.set_title("Akurasi setelah membuang pertanyaan tidak pasti")
        st.pyplot(fig)
        plt.close(fig)
    st.caption(
        f"Jika 20% pertanyaan paling tidak pasti dibuang, akurasi naik dari {se_m['accuracy_at_1.0']:.1%} "
        f"menjadi {se_m['accuracy_at_0.8']:.1%}."
    )

# ---------- 8 ----------
elif sec == SECTIONS[8]:
    st.header("8. Penjelajah Kasus")
    ket = {"TP": "TP: salah & terdeteksi", "FP": "FP: benar tapi ditandai",
           "FN": "FN: salah tapi lolos", "TN": "TN: benar & lolos"}
    pilih = st.multiselect("Jenis kasus", list(ket), default=["FN", "FP"], format_func=lambda k: ket[k])
    cari = st.text_input("Cari pertanyaan")
    f = df[df["kasus"].isin(pilih)]
    if cari:
        f = f[f["question"].str.contains(cari, case=False, na=False)]
    st.subheader(f"Daftar pertanyaan ({len(f)} dari {n})")
    st.dataframe(
        f[["question", "semantic_entropy", "is_correct", "kasus"]].sort_values("semantic_entropy", ascending=False),
        width="stretch",
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

# ---------- 9 ----------
elif sec == SECTIONS[9]:
    st.header("9. Analisis & Keterbatasan")
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

# ---------- 10 ----------
elif sec == SECTIONS[10]:
    st.header("10. Kesimpulan & Saran")
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

# ---------- 11 ----------
elif sec == SECTIONS[11]:
    st.header("11. Referensi")
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
