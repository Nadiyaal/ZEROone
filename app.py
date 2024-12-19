import streamlit as st
from pathlib import Path
import base64
import subprocess

# Menambahkan gambar latar belakang menggunakan CSS
image_path = Path(__file__).parent / "model/UAP.jpg"  # Ganti dengan path gambar Anda
if image_path.is_file():
    # Mengonversi gambar ke base64
    with open(image_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode()

    # Menambahkan CSS untuk gambar latar belakang
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            height: 100vh;
        }}
        
        /* Mengubah semua teks menjadi hitam dan tebal */
        .main-header, .section-header, .stTextInput, p, .card {{
            color: #000000;  /* Warna hitam */
            font-weight: bold;  /* Menebalkan teks */
        }}

        .main-header {{
            font-size: 45px;
            background-color: rgba(76, 175, 80, 0.8); /* Transparansi */
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            color: #000000;  /* Warna hitam untuk teks 'Selamat Datang' */
        }}

        .section-header {{
            font-size: 30px;
            color: #ffffff;  /* Warna putih untuk subjudul */
        }}

        .card {{
            padding: 20px;
            margin: 20px auto;
            background-color: rgba(255, 87, 34, 0.9); /* Transparansi */
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            max-width: 600px;
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .card:hover {{
            transform: translateY(-10px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        }}

        .button {{
            background-color: #ffffff;
            color: #ff0000;  /* Warna merah untuk teks tombol */
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
            cursor: pointer;
            text-decoration: none;
            transition: background-color 0.3s, color 0.3s;
        }}

        .button:hover {{
            background-color: #e64a19;
            color: #ffffff;
        }}

        /* Styling untuk input nama */
        .stTextInput {{
            width: 100%;
            max-width: 400px;
            margin: 10px auto;
            display: block;
        }}

        /* Warna hitam untuk teks tertentu */
        .black-text {{
            color: #000000;
            font-weight: bold;
        }}

        /* Warna merah untuk tombol */
        .red-button {{
            color: #ff0000;  /* Warna merah */
        }}

        /* Button dengan ukuran yang lebih kecil */
        .stButton button {{
            width: 200px;  /* Menetapkan lebar tombol */
            margin: 10px auto;  /* Memusatkan tombol */
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.warning("🚨 Gambar latar belakang tidak ditemukan. Pastikan file `UAP.jpg` tersedia di folder `model`.")

# Header utama
st.markdown('<div class="main-header">✨ Selamat Datang di Dashboard AI Kreatif! ✨</div>', unsafe_allow_html=True)

# Input nama pengguna
nama = st.text_input("📝 Masukkan Nama Anda:", placeholder="Contoh: John Doe")

# Menampilkan sambutan personal jika nama dimasukkan
if nama:
    st.markdown(f"## 👋 <span class='black-text'>Selamat Datang, {nama}!</span> 🚀", unsafe_allow_html=True)
    st.markdown(
        """
        Selamat datang di platform AI kreatif kami. Di sini Anda dapat melakukan berbagai jenis klasifikasi teks 
        seperti analisis sentimen, deteksi topik, dan lainnya. 🧠💬
        """
    )

    # Card untuk klasifikasi teks
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-header">📜 Klasifikasi Teks</div>
        <p>Analisis sentimen dan konteks teks untuk memahami makna yang tersembunyi.</p>
        """, 
        unsafe_allow_html=True
    )

    # Tombol untuk memulai klasifikasi teks
    if st.button("Mulai Klasifikasi Teks", key="text_classification", help="Klik untuk mulai klasifikasi teks!"):
        subprocess.run(["streamlit", "run", "UAP.py"])

    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("## 🚨 <span class='black-text'>Silakan masukkan nama Anda untuk melanjutkan.</span>", unsafe_allow_html=True)
