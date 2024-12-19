import streamlit as st
import tensorflow as tf
import numpy as np
from pathlib import Path
import joblib
import subprocess
import base64

# Konfigurasi halaman utama
st.set_page_config(page_title="Klasifikasi Teks Sentimen", page_icon="📝", layout="centered")

# Menambahkan gambar latar belakang menggunakan CSS
image_path = Path(__file__).parent / "model/UAP.jpg"  # Ganti dengan path gambar Anda
if image_path.is_file():
    with open(image_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            height: 100vh;
            color: black;  /* Mengubah warna teks menjadi hitam agar lebih terlihat */
            font-weight: bold;  /* Membuat semua teks menjadi tebal */
        }}
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown p {{
            color: black;  /* Pastikan semua elemen teks berwarna hitam */
            font-weight: bold;  /* Membuat semua elemen teks menjadi tebal */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("🚨 Gambar latar belakang tidak ditemukan. Pastikan file `UAP.jpg` tersedia di folder `model`.")

# Judul aplikasi
st.title("📝 Klasifikasi Teks Sentimen📝  *Selamat Datang di Aplikasi Klasifikasi Sentimen Berbasis AI!*")


# Menampilkan gambar pendukung
image_path = Path(__file__).parent / "model/saham.jpg"
if image_path.is_file():
    st.image(str(image_path), width=400)
else:
    st.warning("🚨 Gambar tidak ditemukan. Pastikan file `saham.jpg` tersedia di folder `model`.")

# Input teks dari pengguna
text = st.text_area(
    "✍️ **Silakan masukkan teks yang ingin Anda analisis:**",
    placeholder="Contoh: Produk ini luar biasa! Kualitasnya sangat memuaskan. 😊",
    height=150
)

# Fungsi prediksi sentimen
def prediction():
    tokenizer = joblib.load(Path(__file__).parent / "model/tokenizer.joblib")
    model = tf.keras.models.load_model(Path(__file__).parent / "model/model_sentiment.h5")
    sequences = tokenizer.texts_to_sequences([text])
    pad_seq = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100, padding='post')
    result = (model.predict(pad_seq) > 0.5).astype("int32")
    return result[0][0]

# Tombol untuk menganalisis sentimen
if st.button("🔍 **Analisis Sentimen**"):
    st.subheader("📊 **Hasil Analisis Sentimen**")
    classes = ["❌ Negatif", "✅ Positif"]

    # Efek loading dengan animasi
    with st.spinner("⏳ **AI sedang menganalisis teks Anda...**"):
        progress = st.progress(0)
        for percent_complete in range(1, 101):
            progress.progress(percent_complete / 100)
        result = prediction()

    # Menampilkan hasil dengan warna dinamis
    if result == 0:
        st.markdown(f"<h3 style='color: red; font-weight: bold;'>Hasil: {classes[result]}</h3>", unsafe_allow_html=True)
        st.snow()  # Efek salju untuk negatif
    else:
        st.markdown(f"<h3 style='color: green; font-weight: bold;'>Hasil: {classes[result]}</h3>", unsafe_allow_html=True)
        st.balloons()  # Efek balon untuk positif

    # Rincian tambahan dengan kotak latar belakang hitam
    with st.expander("🔎 **Lihat Detail Analisis**"):
        st.markdown(
            f"""
            <div style="background-color: black; padding: 20px; border-radius: 10px;">
                <p style="color: white; font-weight: bold;">**Teks Anda:** {text}</p>
                <p style="color: white; font-weight: bold;">**Klasifikasi Sentimen:** {classes[result]}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Tombol kembali ke halaman utama dengan warna merah
st.markdown(
    """
    🏠 Klik tombol di bawah untuk kembali ke halaman utama!
    """,
    unsafe_allow_html=True
)
if st.button("🔙 **Kembali ke Halaman Utama**"):
    st.markdown(
        f"""
        <style>
        .stButton>button {{
            background-color: red;
            color: white;
        }}
        .stButton>button:hover {{
            color: red;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.info("**Aplikasi siap untuk analisis teks berikutnya! 🚀**")
    subprocess.run(["streamlit", "run", "app.py"])

# Mengubah warna tombol "Analisis Sentimen" menjadi merah
st.markdown(
    """
    <style>
    .stButton>button {
        color: red;
    }
    </style>
    """,
    unsafe_allow_html=True
)
