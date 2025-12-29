import streamlit as st
import pandas as pd
import time
import copy
import utils  # Module custom (Otak pemrosesan)
import visualizer  # Module custom (Visualisasi grafik)
import setting  # Module custom (Import global variabel)

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
    .aspect-card-neutral {
        background-color: #2b2b2b; /* Abu-abu gelap */
        border-left: 5px solid #666666; /* Border abu-abu */
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        opacity: 0.7; /* Sedikit transparan agar terlihat inaktif */
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
# 2. INISIALISASI MODEL & VARIABLE (CACHING)
# ==========================================
# Kita load model di sini agar user melihat loading spinner saat pertama buka

if "ASPECT_KEYWORDS" not in st.session_state:
    st.session_state["ASPECT_KEYWORDS"] = copy.deepcopy(setting.ASPECT_KEYWORDS)


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
        [
            "🏠 Beranda",
            "📝 Analisis Teks (Single)",
            "📂 Analisis File (Batch)",
            "📚 Dokumentasi & Panduan",
        ],
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
        global_sentiment, confidence, aspect_results, lang = (
            utils.analyze_single_review_complete(
                st.session_state["ASPECT_KEYWORDS"], input_text, (models_en, models_id)
            )
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

        # ---------------------------------------------------------
        # LOGIC BARU: Menggabungkan Aspek Terdeteksi & Tidak Terdeteksi
        # ---------------------------------------------------------

        # 1. Ambil semua kemungkinan aspek untuk bahasa tersebut dari variable global
        all_possible_aspects = list(st.session_state["ASPECT_KEYWORDS"][lang].keys())

        # 2. Siapkan list untuk display
        display_list = []

        # Masukkan yang TERDETEKSI dulu (agar muncul paling atas)
        for aspect, data in aspect_results.items():
            display_list.append({"name": aspect, "data": data, "status": "active"})

        # Masukkan yang TIDAK TERDETEKSI (sisanya)
        for aspect in all_possible_aspects:
            if aspect not in aspect_results:
                display_list.append(
                    {"name": aspect, "data": None, "status": "inactive"}
                )

        # ---------------------------------------------------------
        # RENDERING KARTU
        # ---------------------------------------------------------

        col_left, col_right = st.columns(2)

        # Bagi list menjadi dua untuk kolom kiri dan kanan
        mid = (len(display_list) + 1) // 2
        left_items = display_list[:mid]
        right_items = display_list[mid:]

        # Fungsi helper untuk render HTML card agar tidak duplikasi kode
        def render_card(item):
            aspect_name = item["name"]

            if item["status"] == "active":
                # Logic untuk aspek yang TERDETEKSI (Warna Warni)
                data = item["data"]
                if data["label"] == "Positive":
                    css_class = "aspect-card-pos"
                else:
                    css_class = "aspect-card-neg"

                label_text = data["label"].upper()
                score_fmt = f"{data['score']:.1%}"
                trigger_text = (
                    f"Kata Pemicu: '{data['trigger']}'"
                    if data["trigger"]
                    else "Trigger implisit"
                )
                text_color = "white"

            else:
                # Logic untuk aspek yang TIDAK TERDETEKSI (Abu-abu)
                css_class = "aspect-card-neutral"
                label_text = "NOT DETECTED"
                score_fmt = "-"
                trigger_text = "Tidak ditemukan dalam teks"
                text_color = "#888"  # Text agak gelap

            return f"""
            <div class="{css_class}">
                <h4 style="margin:0; color:{text_color};">{aspect_name}</h4>
                <div style="display:flex; justify-content:space-between; margin-top:5px;">
                    <span style="font-weight:bold; color:{text_color};">{label_text}</span>
                    <span style="color:{text_color};">{score_fmt}</span>
                </div>
                <div class="trigger-text">{trigger_text}</div>
            </div>
            """

        with col_left:
            for item in left_items:
                st.markdown(render_card(item), unsafe_allow_html=True)

        with col_right:
            for item in right_items:
                st.markdown(render_card(item), unsafe_allow_html=True)

        st.caption(f"⏱️ Waktu Pemrosesan: {end_time - start_time:.4f} detik")

    # ==========================================
    # BAGIAN: MANAJEMEN ASPEK (UI IMPROVED)
    # ==========================================
    st.divider()
    st.subheader("⚙️ Konfigurasi Aspek & Keyword")

    with st.expander("🛠️ Buka Panel Manajemen Aspek", expanded=False):

        # 1. Pilih Bahasa (Horizontal agar hemat tempat)
        st.write("Pilih Bahasa Sasaran:")
        col_lang, _ = st.columns([1, 2])
        with col_lang:
            lang_choice = st.radio(
                "Bahasa",
                ["🇮🇩 Indonesia", "🇺🇸 English"],
                horizontal=True,
                label_visibility="collapsed",
            )

        # Mapping bahasa ke kode 'id' atau 'en'
        target_lang = "id" if "Indonesia" in lang_choice else "en"

        # Ambil data dari Session State (agar update real-time)
        current_aspects = st.session_state["ASPECT_KEYWORDS"][target_lang]

        # 2. Gunakan Tabs untuk memisahkan mode (UX lebih bersih)
        tab_add_kw, tab_new_cat = st.tabs(
            ["➕ Tambah Keyword", "🆕 Buat Kategori Baru"]
        )

        # --- TAB 1: Tambah Keyword ke Kategori Ada ---
        with tab_add_kw:
            st.caption(
                "Menambahkan kata pemicu (trigger) baru ke kategori yang sudah ada."
            )

            c1, c2 = st.columns(2)
            with c1:
                # Selectbox lebih rapi daripada Radio list panjang
                selected_cat = st.selectbox(
                    "Pilih Kategori:", list(current_aspects.keys())
                )

            with c2:
                new_keyword = st.text_input(
                    "Keyword Baru:", placeholder="Misal: lelet, lemot"
                )

            # Tampilkan keyword yang sudah ada (Preview)
            existing_kws = ", ".join([f"`{k}`" for k in current_aspects[selected_cat]])
            st.info(f"📂 **Keyword saat ini di '{selected_cat}':**\n\n {existing_kws}")

            if st.button("Simpan Keyword", type="primary"):
                if new_keyword:
                    if new_keyword not in current_aspects[selected_cat]:
                        # Update Session State
                        st.session_state["ASPECT_KEYWORDS"][target_lang][
                            selected_cat
                        ].append(new_keyword)

                        st.toast(
                            f"✅ Berhasil menambahkan '{new_keyword}' ke {selected_cat}!",
                            icon="💾",
                        )
                        time.sleep(1)  # Beri waktu baca toast
                        st.rerun()
                    else:
                        st.warning("⚠️ Keyword tersebut sudah ada.")
                else:
                    st.error("❌ Keyword tidak boleh kosong.")

        # --- TAB 2: Buat Kategori Baru ---
        with tab_new_cat:
            st.caption(
                "Membuat aspek penilaian baru (Misal: 'UI/UX', 'Customer Service')."
            )

            c_cat, c_kw = st.columns(2)
            with c_cat:
                new_cat_name = st.text_input(
                    "Nama Kategori Baru:", placeholder="Misal: Design"
                )
            with c_kw:
                first_kw_input = st.text_input(
                    "Keyword Pertama:", placeholder="Misal: tampilan, warna"
                )

            if st.button("Buat Kategori", type="primary"):
                if new_cat_name and first_kw_input:
                    if new_cat_name not in current_aspects:
                        # Update Session State
                        st.session_state["ASPECT_KEYWORDS"][target_lang][
                            new_cat_name
                        ] = [first_kw_input]

                        st.toast(
                            f"✅ Kategori '{new_cat_name}' berhasil dibuat!", icon="✨"
                        )
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.warning("⚠️ Kategori tersebut sudah ada.")
                else:
                    st.error("❌ Nama Kategori dan Keyword pertama harus diisi.")


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
                                st.session_state["ASPECT_KEYWORDS"],
                                text,
                                (models_en, models_id),
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


# ==========================================
# 7. HALAMAN KEEMPAT: DOKUMENTASI & PANDUAN
# ==========================================
elif menu == "📚 Dokumentasi & Panduan":
    st.title("📚 Dokumentasi & Referensi Sistem")
    st.markdown(
        "Panduan lengkap penggunaan aplikasi dan daftar kata kunci (keywords) yang digunakan oleh AI."
    )

    # Gunakan Tabs untuk memisahkan Panduan dan Daftar Aspek
    tab_guide, tab_dict = st.tabs(["🚀 Cara Penggunaan", "📖 Kamus Aspek (Live)"])

    # --- TAB 1: PANDUAN PENGGUNAAN ---
    with tab_guide:
        st.header("Panduan Penggunaan Aplikasi")

        with st.expander("📝 Cara Melakukan Analisis Teks (Single)", expanded=True):
            st.markdown(
                """
            1. Pergi ke menu **Analisis Teks (Single)** di sidebar.
            2. Masukkan kalimat ulasan/review pada kolom teks yang tersedia.
            3. Klik tombol **🔍 Analisis Sekarang**.
            4. Sistem akan menampilkan:
               - **Sentimen Global:** Apakah ulasan tersebut secara umum Positif atau Negatif.
               - **Deteksi Aspek:** AI akan memecah kalimat dan mendeteksi aspek spesifik (misal: Audio, Harga, Iklan).
            5. Anda juga bisa **menambahkan keyword baru** di bagian bawah halaman hasil analisis jika AI melewatkan sesuatu.
            """
            )

        with st.expander("📂 Cara Melakukan Analisis File (Batch)", expanded=True):
            st.markdown(
                """
            1. Siapkan file data dalam format **CSV** atau **Excel (.xlsx)**.
            2. Pastikan file memiliki kolom teks (misal: `content`, `review`, `text`, atau `ulasan`).
            3. Pergi ke menu **Analisis File (Batch)**.
            4. Upload file Anda ke area yang disediakan.
            5. Klik tombol **⚡ Jalankan Analisis AI**.
            6. Tunggu proses selesai (Progress bar akan berjalan).
            7. Lihat **Dashboard Visualisasi** atau download hasil lengkapnya via tombol **Download CSV**.
            """
            )

    # --- TAB 2: KAMUS ASPEK (DINAMIS DARI SESSION STATE) ---
    with tab_dict:
        st.header("📖 Daftar Keyword & Kategori Aspek")
        st.info(
            """
        Daftar ini diambil secara **Real-Time** dari memori aplikasi. 
        Jika Anda menambahkan keyword baru melalui menu 'Analisis Teks', keyword tersebut akan langsung muncul di sini.
        """
        )

        # Pilihan Bahasa untuk melihat kamus
        lang_choice_doc = st.radio(
            "Pilih Bahasa Kamus:", ["🇮🇩 Indonesia", "🇺🇸 English"], horizontal=True
        )

        # Tentukan target key session state
        target_lang_doc = "id" if "Indonesia" in lang_choice_doc else "en"

        # Cek apakah session state tersedia
        if "ASPECT_KEYWORDS" in st.session_state:
            # Ambil data langsung dari Session State (Bukan dari file setting.py mentah)
            current_aspects_doc = st.session_state["ASPECT_KEYWORDS"][target_lang_doc]

            # Loop setiap kategori
            for category, keywords in current_aspects_doc.items():
                with st.expander(
                    f"📂 Kategori: **{category}** ({len(keywords)} keywords)"
                ):
                    # Tampilkan keywords dalam bentuk tags/code agar rapi
                    # Mengurutkan keyword agar mudah dibaca
                    sorted_kws = sorted(keywords)

                    # Tampilkan
                    st.markdown("Kata kunci pemicu (Triggers):")
                    content_html = ""
                    for kw in sorted_kws:
                        content_html += f"<code style='color: #1DB954; background-color: #191414; border: 1px solid #333; margin: 2px; padding: 2px 6px; border-radius: 4px; display: inline-block;'>{kw}</code> "

                    st.markdown(content_html, unsafe_allow_html=True)
        else:
            st.error(
                "⚠️ Data aspek belum dimuat ke dalam sistem. Silakan muat ulang aplikasi."
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
