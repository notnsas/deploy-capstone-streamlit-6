import streamlit as st
import pandas as pd
import time
import utils  # Module custom (Otak pemrosesan)
import visualizer  # Module custom (Visualisasi grafik)

# ==========================================
# 1. KONFIGURASI HALAMAN & TEMA
# ==========================================
st.set_page_config(
    page_title="Spotify Sentiment Intel",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS (Agar tampilan mirip Spotify: Dark & Neon Green) ---
st.markdown(
    """
<style>
    /* Mengatur Warna Utama */
    :root {
        --primary-color: #1DB954;
        --bg-color: #121212;
        --secondary-bg: #191414;
        --text-color: #FFFFFF;
    }
    
    /* Background App */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
    }
    
    /* Judul Besar */
    h1, h2, h3 {
        color: var(--text-color) !important;
        font-family: 'Helvetica', sans-serif;
    }
    
    /* Styling Metric Box (Kotak Angka) */
    div[data-testid="stMetric"] {
        background-color: var(--secondary-bg);
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricLabel"] {
        color: #b3b3b3;
    }
    div[data-testid="stMetricValue"] {
        color: var(--primary-color);
        font-weight: bold;
    }

    /* Custom Button Style */
    div.stButton > button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 20px;
        font-weight: bold;
        border: none;
        padding: 10px 24px;
    }
    div.stButton > button:hover {
        background-color: #1ed760; /* Lebih terang saat hover */
        border: 1px solid white;
    }

    /* Aspek Card Styling */
    .aspect-card-pos {
        background-color: #0d2e18;
        border-left: 5px solid #1DB954;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .aspect-card-neg {
        background-color: #3b0d10;
        border-left: 5px solid #ff4d4d;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .trigger-text {
        font-size: 0.85em;
        color: #b3b3b3;
        font-style: italic;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 2. INISIALISASI MODEL (CACHING)
# ==========================================
# Kita load model di sini agar user melihat loading spinner saat pertama buka


@st.cache_resource
def initialize_ai_engine():
    """Wrapper untuk memuat model dari utils"""
    return utils.load_all_models()


# Menampilkan Spinner Loading saat awal buka aplikasi
if "models_loaded" not in st.session_state:
    with st.spinner("🤖 Sedang Memanaskan Mesin AI (Loading Models)..."):
        dict_model = initialize_ai_engine()
        models_en = dict_model["en"]
        models_id = dict_model["id"]

        if models_en is None or models_id is None:
            st.error(
                "❌ Gagal memuat model. Pastikan folder 'models/' lengkap sesuai struktur."
            )
            st.stop()

        st.session_state["models_en"] = models_en
        st.session_state["models_id"] = models_id
        # print("loaded models_en: ")
        # print(models_en)
        st.session_state["models_loaded"] = True
    st.toast("✅ Sistem AI Siap Digunakan!", icon="🚀")
else:
    # Ambil dari cache session jika sudah ada
    models_en = st.session_state["models_en"]
    models_id = st.session_state["models_id"]


# ==========================================
# 3. SIDEBAR NAVIGASI
# ==========================================
with st.sidebar:
    # --- LOGO SECTION ---
    # Menggunakan URL Logo Spotify Official (Transparan)
    logo_url = "https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_RGB_Green.png"

    try:
        # Coba load logo lokal dulu jika ada
        st.image("assets/logo.png", width=200)
    except:
        # Jika tidak ada file lokal, gunakan URL online
        st.image(logo_url, width=200)

    st.markdown("---")
    st.header("🎛️ Main Menu")

    # Navigasi menggunakan Radio Button yang cantik
    menu = st.radio(
        "Pilih Mode Analisis:",
        ["🏠 Beranda", "📝 Analisis Teks (Single)", "📂 Analisis File (Batch)"],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### ℹ️ Tentang Sistem")
    st.info(
        """
        Sistem ini menggunakan arsitektur **Hybrid**:
        - **IndoBERT / BERT** (High Accuracy)
        - **ABSA Engine** (Granular)
        
        Dibuat untuk **Capstone Project Data Science**.
        """
    )

# ==========================================
# 4. HALAMAN UTAMA: BERANDA
# ==========================================
if menu == "🏠 Beranda":
    st.title("Welcome to Spotify Review Intelligence 👋")

    st.markdown(
        """
    Platform analisis sentimen tingkat lanjut yang dirancang untuk membedah ribuan ulasan pengguna Spotify 
    menjadi wawasan bisnis yang dapat ditindaklanjuti (*Actionable Insights*).
    """
    )

    # Showcase Fitur (Kolom)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🌐 Dual Language")
        st.write(
            "Mendeteksi otomatis Bahasa Indonesia & Inggris dengan preprocessing cerdas."
        )
    with c2:
        st.markdown("### 🔍 Aspect-Based")
        st.write(
            "Tidak hanya positif/negatif, tapi mendeteksi **Audio**, **Harga**, **Iklan**, & **Bug**."
        )
    with c3:
        st.markdown("### 📊 Interactive Dataviz")
        st.write(
            "Visualisasi WordCloud dan Chart interaktif untuk pengambilan keputusan cepat."
        )

    st.divider()
    st.markdown("##### 🚀 Cara Memulai:")
    st.markdown("1. Buka menu **Analisis Teks** untuk menguji satu kalimat review.")
    st.markdown(
        "2. Buka menu **Analisis File** untuk mengupload CSV berisi ribuan data ulasan."
    )

# ==========================================
# 5. HALAMAN KEDUA: ANALISIS TEKS TUNGGAL
# ==========================================
elif menu == "📝 Analisis Teks (Single)":
    st.title("📝 Granular Text Analysis")
    st.markdown(
        "Masukkan satu ulasan untuk melihat bagaimana AI membedah sentimen dan aspeknya."
    )

    # Input Area
    with st.container():
        input_text = st.text_area(
            "Masukkan Ulasan User:",
            height=150,
            placeholder="Contoh: Aplikasinya bagus, lagunya lengkap. Tapi sayang harga premium makin mahal dan iklannya kebanyakan...",
        )

        col_btn, col_opt = st.columns([1, 4])
        with col_btn:
            analyze_btn = st.button(
                "🔍 Analisis Sekarang", type="primary", use_container_width=True
            )

    # Hasil Analisis
    if analyze_btn and input_text:
        start_time = time.time()

        # Panggil Fungsi Utils (Logic Backend)
        print("model_en 235")
        print(models_en)
        global_sentiment, confidence, aspect_results, lang = (
            utils.analyze_single_review_complete(input_text, (models_en, models_id))
        )

        end_time = time.time()

        st.divider()
        st.markdown("### 🎯 Hasil Analisis AI")

        # Metric Utama
        m1, m2, m3 = st.columns(3)
        with m1:
            emoji = "😄" if global_sentiment == "Positive" else "😡"
            st.metric("Sentimen Global", f"{emoji} {global_sentiment}")
        with m2:
            st.metric("Keyakinan (Confidence)", f"{confidence:.1%}")
        with m3:
            flag = "🇮🇩" if lang == "id" else "🇺🇸"
            lang_name = "Indonesia" if lang == "id" else "Inggris"
            st.metric("Bahasa Terdeteksi", f"{flag} {lang_name}")

        # Tampilan Aspek Granular (Cards)
        st.subheader("🔍 Breakdown Per Aspek")

        if aspect_results:
            col_left, col_right = st.columns(2)

            # Membagi kartu ke 2 kolom agar rapi
            items = list(aspect_results.items())
            mid = (len(items) + 1) // 2

            with col_left:
                for aspect_name, data in items[:mid]:
                    # Tentukan warna kartu berdasarkan sentimen aspek
                    css_class = (
                        "aspect-card-pos"
                        if data["label"] == "Positive"
                        else "aspect-card-neg"
                    )
                    score_fmt = f"{data['score']:.1%}"
                    trigger = (
                        f"Kata Pemicu: '{data['trigger']}'"
                        if data["trigger"]
                        else "Tidak ada keyword spesifik"
                    )

                    html_card = f"""
                    <div class="{css_class}">
                        <h4 style="margin:0; color:white;">{aspect_name}</h4>
                        <div style="display:flex; justify-content:space-between; margin-top:5px;">
                            <span style="font-weight:bold;">{data['label'].upper()}</span>
                            <span>{score_fmt}</span>
                        </div>
                        <div class="trigger-text">{trigger}</div>
                    </div>
                    """
                    st.markdown(html_card, unsafe_allow_html=True)

            with col_right:
                for aspect_name, data in items[mid:]:
                    # Logic yang sama untuk kolom kanan
                    css_class = (
                        "aspect-card-pos"
                        if data["label"] == "Positive"
                        else "aspect-card-neg"
                    )
                    score_fmt = f"{data['score']:.1%}"
                    trigger = (
                        f"Kata Pemicu: '{data['trigger']}'"
                        if data["trigger"]
                        else "Tidak ada keyword spesifik"
                    )

                    html_card = f"""
                    <div class="{css_class}">
                        <h4 style="margin:0; color:white;">{aspect_name}</h4>
                        <div style="display:flex; justify-content:space-between; margin-top:5px;">
                            <span style="font-weight:bold;">{data['label'].upper()}</span>
                            <span>{score_fmt}</span>
                        </div>
                        <div class="trigger-text">{trigger}</div>
                    </div>
                    """
                    st.markdown(html_card, unsafe_allow_html=True)
        else:
            st.info(
                "ℹ️ Tidak ditemukan aspek spesifik pada ulasan ini. (Dikategorikan General)"
            )

        st.caption(f"⏱️ Waktu Pemrosesan: {end_time - start_time:.4f} detik")

# ==========================================
# 6. HALAMAN KETIGA: ANALISIS BATCH (FILE)
# ==========================================
elif menu == "📂 Analisis File (Batch)":
    st.title("📂 Batch Sentiment Processing")
    st.markdown(
        "Unggah file (CSV/Excel) ulasan aplikasi untuk analisis massal otomatis."
    )

    # File Uploader
    uploaded_file = st.file_uploader(
        "Drop file di sini (Pastikan ada kolom 'content' atau 'review')",
        type=["csv", "xlsx"],
    )

    if uploaded_file:
        # Load Data dengan Caching agar tidak reload saat klik
        df = utils.load_uploaded_file(uploaded_file)

        if df is not None:
            # Otomatis cari kolom teks
            text_col = utils.find_text_column(df)

            if text_col:
                st.success(
                    f"✅ File berhasil dimuat! Ditemukan **{len(df)}** baris data."
                )
                st.info(f"Kolom teks yang akan dianalisis: `{text_col}`")

                # Tombol Eksekusi
                if st.button("⚡ Jalankan Analisis AI (Batch)", type="primary"):

                    # Progress Bar Container
                    progress_text = "Memproses ulasan dengan Artificial Intelligence..."
                    my_bar = st.progress(0, text=progress_text)

                    # Placeholder untuk logs real-time
                    log_placeholder = st.empty()

                    # PROSES BACKEND (Di utils)
                    # Kita proses dalam batch kecil agar bar progress jalan halus
                    results = []

                    # Batasan untuk Demo Capstone agar tidak menunggu berjam-jam jika data ribuan
                    # (Bisa dihapus jika deploy di server kuat)
                    MAX_PROCESS = 500
                    df_to_process = df.head(MAX_PROCESS)

                    total_items = len(df_to_process)

                    for idx, row in df_to_process.iterrows():
                        text = str(row[text_col])

                        # Analisis
                        gl_lbl, gl_conf, aspects, lang = (
                            utils.analyze_single_review_complete(
                                text, (models_en, models_id)
                            )
                        )

                        # Susun Data untuk Report
                        res_row = {
                            "Original Text": text,
                            "Language": lang,
                            "Global Sentiment": gl_lbl,
                            "Confidence": gl_conf,
                            "Aspects JSON": str(
                                aspects
                            ),  # Disimpan sebagai string untuk CSV
                        }

                        # Tambahkan kolom dinamis per aspek untuk kemudahan Excel
                        for asp, detail in aspects.items():
                            res_row[f"{asp}_Sentiment"] = detail["label"]

                        results.append(res_row)

                        # Update Progress
                        percent = int(((idx + 1) / total_items) * 100)
                        my_bar.progress(
                            percent,
                            text=f"Sedang memproses {idx+1}/{total_items} data...",
                        )

                    # Selesai
                    my_bar.empty()
                    df_result = pd.DataFrame(results)

                    # Simpan hasil ke session state agar tidak hilang saat refresh visualisasi
                    st.session_state["batch_result"] = df_result

            else:
                st.error(
                    "❌ Tidak dapat menemukan kolom teks (seperti 'content', 'review', 'text'). Mohon rename kolom CSV Anda."
                )
        else:
            st.error("Format file tidak didukung.")

    # --- TAMPILAN DASHBOARD HASIL (Jika data sudah ada di session) ---
    if "batch_result" in st.session_state:
        df_res = st.session_state["batch_result"]

        st.divider()
        st.subheader("📊 Laporan & Dashboard")

        # Tabs agar rapi
        tab_sum, tab_viz, tab_data = st.tabs(
            ["📈 Summary & KPIs", "📊 Visualisasi Mendalam", "📥 Data Detail"]
        )

        with tab_sum:
            # Memanggil Visualizer untuk menampilkan KPI Card
            visualizer.display_kpi_metrics(df_res)

            col_don, col_bar = st.columns(2)
            with col_don:
                visualizer.plot_sentiment_donut(df_res)
            with col_bar:
                # Perlu memproses aspek JSON string kembali ke dict untuk plotting
                visualizer.plot_aspect_bar_chart(df_res)

            # 3. VISUALISASI BARU DI SINI (Di bawah KPI/Grafik summary)
            st.subheader("🗣️ Apa yang Paling Sering Dibahas?")
            st.caption(
                "Grafik ini menunjukkan kata kunci spesifik (trigger) yang muncul dalam ulasan, dibagi berdasarkan sentimennya."
            )

            # Panggil fungsi baru
            visualizer.plot_trigger_sentiment_chart(df_res)

        with tab_viz:
            st.write("#### ☁️ Wordcloud Analisis")
            lang_choice = st.selectbox(
                "Pilih Bahasa untuk Wordcloud:", df_res["Language"].unique()
            )

            # Filter teks berdasarkan bahasa & sentimen
            subset_df = df_res[df_res["Language"] == lang_choice]

            wc_col1, wc_col2 = st.columns(2)
            with wc_col1:
                st.write("**Top Words di Ulasan Positif:**")
                visualizer.generate_wordcloud(subset_df, "Positive")
            with wc_col2:
                st.write("**Top Words di Ulasan Negatif:**")
                visualizer.generate_wordcloud(subset_df, "Negative")

        with tab_data:
            st.write("#### 📄 Data Hasil Analisis")
            st.dataframe(df_res)

            # Tombol Download
            # Utils function untuk convert df ke CSV/Excel byte stream
            col_d1, col_d2 = st.columns([1, 4])
            with col_d1:
                st.download_button(
                    label="⬇️ Download CSV",
                    data=utils.convert_df_to_csv(df_res),
                    file_name=f"absa_result_{int(time.time())}.csv",
                    mime="text/csv",
                )

# Footer Profesional
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 0.8em;">
        &copy; 2025 Capstone Project - Advanced Sentiment Analytics.<br>
        Powered by IndoBERT & BERT.
    </div>
    """,
    unsafe_allow_html=True,
)
