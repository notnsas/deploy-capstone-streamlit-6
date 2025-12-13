import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import torch
import nltk
from io import BytesIO
from langdetect import detect, LangDetectException

# Library NLP & Deep Learning
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ==========================================
# 1. SETUP ENVIRONMENT & RESOURCE LOADING
# ==========================================

# Definisi Device (GPU/CPU) untuk PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download NLTK Resources secara senyap jika belum ada
try:
    nltk.data.find("corpora/stopwords")
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)

# Inisialisasi Sastrawi (Hanya sekali agar cepat)
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# ==========================================
# 2. KAMUS & LINGUISTIC ASSETS
# ==========================================

# A. Kamus Aspek (Advanced with Triggers)

ASPECT_KEYWORDS = {
    "en": {
        "Audio Quality": [
            # Technical Terms
            "audio",
            "sound",
            "voice",
            "bass",
            "treble",
            "mid",
            "frequency",
            "equalizer",
            "eq",
            "dolby",
            "hifi",
            "stereo",
            "lossless",
            "bitrate",
            "kbps",
            "soundstage",
            "crisp",
            "volume",
            "normalize",
            "crossfade",
            "mono",
            "surround",
            "spatial",
            # Positive Nuance
            "crystal clear",
            "eargasm",
            "immersive",
            "HD",
            "punchy",
            # Negative Symptoms (The "Symptom-based Detection")
            "distortion",
            "noise",
            "static",
            "muffled",
            "robotic",
            "tinny",
            "echo",
            "glitchy sound",
            "crackling",
            "hissing",
            "cuts out",
            "popping",
            "clipping",
            "skip",
            "stutter",
            "quiet",
            "loud",
            "flat",
            "low quality",
            "muds",
            "leaking",
        ],
        "Content & Library": [
            # Music Items
            "song",
            "track",
            "music",
            "playlist",
            "album",
            "artist",
            "band",
            "discography",
            "library",
            "catalog",
            "genre",
            "collection",
            "release",
            "single",
            "cover",
            "lyrics",
            "karaoke",
            "sing along",
            "verse",
            "chorus",
            # Non-Music Content
            "podcast",
            "episode",
            "audiobook",
            "host",
            "story",
            "content",
            "talk show",
            # Discovery & Trends
            "recommendation",
            "suggestion",
            "discover weekly",
            "release radar",
            "mix",
            "top 50",
            "viral",
            "k-pop",
            "indie",
            "hiphop",
            "jazz",
            "lofi",
            "ost",
            # Availability Issues
            "missing",
            "greyed out",
            "unavailable",
            "removed",
            "licensing",
            "region lock",
        ],
        "App Performance & Stability": [
            # Crashes & Critical
            "crash",
            "force close",
            "closes itself",
            "shut down",
            "freeze",
            "froze",
            "black screen",
            "white screen",
            "boot",
            "stuck",
            "unresponsive",
            "hang",
            # Bugs & Glitches
            "glitch",
            "bug",
            "error",
            "failure",
            "broken",
            "issue",
            "problem",
            # Performance/Speed
            "slow",
            "lag",
            "laggy",
            "latency",
            "delay",
            "sluggish",
            "bloated",
            "heavy",
            # Connectivity & Data
            "loading",
            "buffer",
            "buffering",
            "waiting",
            "offline mode",
            "connection",
            "internet",
            "wifi",
            "signal",
            "data usage",
            "no connection",
            "network",
            # Resource Impact
            "battery",
            "drain",
            "heat",
            "app",
            "overheat",
            "storage",
            "space",
            "cache",
            "ram",
        ],
        "User Interface (UI) & UX": [
            # Visual Elements
            "interface",
            "ui",
            "design",
            "layout",
            "look",
            "theme",
            "color",
            "icon",
            "dark mode",
            "light mode",
            "button",
            "font",
            "text",
            "size",
            # Navigation & Usability
            "navigation",
            "menu",
            "screen",
            "display",
            "cluttered",
            "confusing",
            "clean",
            "modern",
            "user friendly",
            "intuitive",
            "aesthetic",
            "scroll",
            "swipe",
            "gesture",
            "sidebar",
            "tab",
            "library view",
        ],
        "Features & Functionality": [
            # Specific Features
            "search",
            "queue",
            "share",
            "connect",
            "devices",
            "remote",
            "car play",
            "android auto",
            "widget",
            "notification",
            "sleep timer",
            "lyrics sync",
            "canvas",
            "video",
            "story",
            "wrap",
            "wrapped",
            # Account & Login
            "login",
            "sign in",
            "logout",
            "account",
            "password",
            "profile",
            "download",
            "save",
            "install",
            "update",
        ],
        "Price, Premium & Ads": [
            # Costs & Plans
            "price",
            "cost",
            "expensive",
            "cheap",
            "affordable",
            "fee",
            "charge",
            "billing",
            "payment",
            "money",
            "worth",
            "value",
            "greedy",
            "rip off",
            # Premium Features
            "premium",
            "subscription",
            "plan",
            "student",
            "family",
            "duo",
            "trial",
            "renew",
            "cancel",
            "refund",
            "upgrade",
            # Advertisements
            "ad",
            "ads",
            "advertisement",
            "commercial",
            "sponsor",
            "promotion",
            "interrupt",
            "unskippable",
            "video ads",
            "30 seconds",
            "spam",
        ],
        "Algorithm & Shuffle": [
            # Randomness
            "shuffle",
            "random",
            "repeat",
            "loop",
            "smart shuffle",
            "enhanced shuffle",
            "mix",
            "radio",
            "pattern",
            "algorithm",
            "playing same song",
            # Quality of Suggestion
            "irrelevant",
            "repetitive",
            "bad suggestion",
            "same artists",
            "boring",
        ],
    },
    "id": {
        "Audio Quality": [
            # Keyword Formal
            "suara",
            "audio",
            "bunyi",
            "vokal",
            "instrumen",
            "kualitas",
            "volume",
            "bass",
            "bas",
            "treble",
            "equalizer",
            "eq",
            "dolby",
            "stereo",
            "speaker",
            # Kata Sifat & Gejala (Symptom-based)
            "jernih",
            "bening",
            # "bagus",
            "nendang",
            "mantap",
            "halus",
            "detail",
            "cempreng",
            "pecah",
            "sember",
            "mendem",
            "kresek",
            "kresek-kresek",
            "robot",
            "berisik",
            "kecil banget",
            "gede",
            "tenggelam",
            "hancur",
            "putus",
            "putus-putus",
            "macet-macet",
            "ngadat",
            "grebeg",
            "bocora",
            "mono",
            "low quality",
            "ga ada suara",
            "ilang suaranya",
            "bisor",
        ],
        "Content & Library": [
            # Musik & Item
            "lagu",
            "musik",
            "track",
            "trek",
            "playlist",
            "album",
            "artis",
            "band",
            "penyanyi",
            "katalog",
            "rilisan",
            "single",
            "cover",
            "mp3",
            "genre",
            "dangdut",
            "koplo",
            "pop",
            "jazz",
            "rock",
            "indie",
            "kpop",
            # Non-Musik
            "podcast",
            "cerita",
            "dongeng",
            "horror",
            "episode",
            "konten",
            "lirik",
            "teks",
            "karaoke",
            "terjemahan",
            # Koleksi User
            "koleksi",
            "pustaka",
            "liked songs",
            "disukai",
            "downloadan",
            "unduhan",
            # Masalah Ketersediaan
            "lagu hilang",
            "dihapus",
            "ga ada",
            "ga lengkap",
            "judul",
            "cari lagu",
        ],
        "App Performance & Stability": [
            # Crash & Hang (Bahasa PlayStore Liar)
            "crash",
            "force close",
            "fc",
            "mental",
            "keluar sendiri",
            "keluar terus",
            "keluar mulu",
            "kelempar",
            "tutup sendiri",
            "stopped",
            "stuck",
            "macet",
            "hang",
            "ngelag",
            "freeze",
            "layar hitam",
            "blank",
            "bapuk",
            # Masalah Kecepatan/Koneksi
            "lemot",
            "lambat",
            "lelet",
            "lola",
            "berat",
            "ngadat",
            "muter",
            "muter-muter",
            "loading",
            "buffer",
            "buffering",
            "lama banget",
            "lama",
            "koneksi",
            "sinyal",
            "offline",
            "internet",
            "jaringan",
            "server",
            "down",
            # Masalah Hardware/Sistem
            "baterai",
            "aplikasi",
            "batrai",
            "batre",
            "boros",
            "panas",
            "cepet panas",
            "overheat",
            "memori",
            "ram",
            "penuh",
            "cache",
            "data",
            "kuota",
            "makan tempat",
            "sedot",
            "ngabisin",
            "hp kentang",
            "rusak",
        ],
        "User Interface (UI) & UX": [
            # Visual
            "tampilan",
            "ui",
            "desain",
            "interface",
            "layout",
            "tema",
            "warna",
            "menu",
            "tombol",
            "huruf",
            "tulisan",
            "font",
            "ikon",
            "gambar",
            "background",
            "latar",
            "gelap",
            "terang",
            "dark mode",
            # Navigasi & Rasa
            "bingung",
            "pusing",
            "ribet",
            "susah",
            "gampang",
            "mudah",
            "simpel",
            "bersih",
            "rapi",
            "berantakan",
            "aneh",
            "jelek",
            "keren",
            "elegan",
            "geser",
            "pencet",
            "klik",
            "navigasi",
        ],
        "Features & Functionality": [
            # Fitur Spesifik
            "fitur",
            "lirik",
            "lyrics",
            "download",
            "unduh",
            "simpan",
            "cari",
            "search",
            "share",
            "bagi",
            "connect",
            "jam",
            "blend",
            "widget",
            "notif",
            "pemberitahuan",
            "update",
            "pembaruan",
            "versi",
            # Login & Akun
            "login",
            "masuk",
            "daftar",
            "logout",
            "keluar akun",
            "akun",
            "email",
            "password",
            "sandi",
            "verifikasi",
            "otp",
        ],
        "Price, Premium & Ads": [
            # Iklan (Sangat Sering Dikeluhkan)
            "iklan",
            "ads",
            "ad",
            "promosi",
            "pariwara",
            "sponsor",
            "commercial",
            "banyak iklan",
            "iklan mulu",
            "kebanyakan iklan",
            "nonton iklan",
            "gangguan",
            "kepotong",
            "skip",
            # Harga
            "harga",
            "biaya",
            "tarif",
            "bayar",
            "tagihan",
            "uang",
            "duid",
            "mahal",
            "kemahalan",
            "murah",
            "terjangkau",
            "murmer",
            "naik harga",
            "matre",
            "maruk",
            "kuras dompet",
            "sedot pulsa",
            # Langganan & Premium
            "premium",
            "langganan",
            "berbayar",
            "gratisan",
            "vip",
            "paket",
            "family",
            "student",
            "pelajar",
            "trial",
            "percobaan",
            "metode",
            "gopay",
            "dana",
            "ovo",
            "shopeepay",
            "pulsa",
        ],
        "Algorithm & Shuffle": [
            # Acak Lagu
            "shuffle",
            "acak",
            "random",
            "urutan",
            "ngulang",
            "itu-itu aja",
            "muter ulang",
            "campur",
            "smart shuffle",
            # Rekomendasi
            "rekomendasi",
            "saran",
            "mix",
            "radio",
            "algoritma",
            "pilihan",
            "sesuai selera",
            "ga nyambung",
            "maksa",
        ],
    },
}

# B. Kamus Slang Indonesia (Normalisasi)
SLANG_MAP = {
    "yg": "yang",
    "bgt": "banget",
    "dg": "dengan",
    "dgn": "dengan",
    "gak": "tidak",
    "ga": "tidak",
    "gx": "tidak",
    "nggak": "tidak",
    "tak": "tidak",
    "klo": "kalau",
    "kalo": "kalau",
    "kl": "kalau",
    "jgn": "jangan",
    "krn": "karena",
    "krna": "karena",
    "blm": "belum",
    "sdh": "sudah",
    "udh": "sudah",
    "dh": "sudah",
    "aja": "saja",
    "aj": "saja",
    "bkn": "bukan",
    "km": "kamu",
    "aq": "aku",
    "ak": "aku",
    "dr": "dari",
    "utk": "untuk",
    "bwt": "untuk",
    "gw": "gue",
    "lu": "lo",
    "gmn": "bagaimana",
    "gimana": "bagaimana",
    "bgs": "bagus",
    "keren": "bagus",
    "best": "bagus",
    "good": "bagus",
    "jelek": "buruk",
    "bad": "buruk",
    "lagi": "sedang",
    "lg": "sedang",
    "sy": "saya",
    "sm": "sama",
    "dengerin": "dengar",
    "didengerin": "didengar",
    "dengarkan": "dengar",
    "mendengarkan": "dengar",
    "puter": "putar",
    "muter": "putar",
    "diputer": "diputar",
    "aplikasi": "aplikasi",
    "apk": "aplikasi",
    "app": "aplikasi",
    "apps": "aplikasi",
    "donlot": "unduh",
    "download": "unduh",
    "dunlut": "unduh",
    "login": "masuk",
    "log in": "masuk",
    "sign in": "masuk",
    "logout": "keluar",
    "keluar sendiri": "crash",
    "mental": "crash",
    "force close": "crash",
    "fc": "crash",
    "ngelag": "lambat",
    "lag": "lambat",
    "lemot": "lambat",
    "lelet": "lambat",
    "iklan": "iklan",
    "ads": "iklan",
    "kebanyakan": "banyak",
    "dikit": "sedikit",
    "bs": "bisa",
    "gabisa": "tidak bisa",
    "tau": "tahu",
    "liat": "lihat",
    "pas": "saat",
    "mw": "mau",
    "pingin": "mau",
    "woy": "",
    "anjir": "",
    "anj": "",
    "njir": "",  # Menghapus swear words noise
}

# C. Whitelist Negasi (Agar tidak dihapus Stopwords)
NEGATION_WORDS = {
    # Basic Negation
    "no",
    "not",
    "none",
    "neither",
    "never",
    "nobody",
    "nothing",
    "nowhere",
    # Contractions
    "can't",
    "cannot",
    "don't",
    "doesn't",
    "didn't",
    "won't",
    "wouldn't",
    "shouldn't",
    "couldn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "hasn't",
    "haven't",
    # Contrast Conjunctions
    "but",
    "however",
    "although",
    "despite",
    "except",
    "yet",
    "whereas",
    "unfortunately",
    # Emotive Negative
    "hate",
    "bad",
    "worst",
    "awful",
    "terrible",
    "sucks",
    "useless",
    "disappointed",
    # Basic Negation
    "no",
    "not",
    "none",
    "neither",
    "never",
    "nobody",
    "nothing",
    "nowhere",
    # Contractions
    "can't",
    "cannot",
    "don't",
    "doesn't",
    "didn't",
    "won't",
    "wouldn't",
    "shouldn't",
    "couldn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "hasn't",
    "haven't",
    # Contrast Conjunctions
    "but",
    "however",
    "although",
    "despite",
    "except",
    "yet",
    "whereas",
    "unfortunately",
    # Emotive Negative
    "hate",
    "bad",
    "worst",
    "awful",
    "terrible",
    "sucks",
    "useless",
    "disappointed",
}


# ==========================================
# 3. CORE LOGIC: PREPROCESSING
# ==========================================


def reduce_repeating_chars(text, max_repeat=2):
    pattern = r"(.)\1{" + str(max_repeat) + r",}"
    return re.sub(pattern, r"\1" * max_repeat, text)


def normalize_slang_id(tokens):
    """Mapping list token berdasarkan kamus slang."""
    return [SLANG_MAP.get(word, word) for word in tokens]


def fix_ui_nya(text):
    """
    Stemming kata ui, karena ui tidak ada di KBBI jadi tidak bisa
    di pakai disastrawi.
    """
    return text.replace("uinya", "ui nya")


def build_keyword_set(ASPECT_KEYWORDS, lang):
    """
    Stemming kata seperti ui, fitur, dll; karena ui, fitur, dll tidak ada di KBBI jadi tidak bisa
    di pakai disastrawi.
    """
    keywords = set()
    for aspect in ASPECT_KEYWORDS[lang].values():
        for k in aspect:
            keywords.add(k.lower())
    return keywords


def normalize_by_prefix(token, keywords):
    """
    Normalisasi dengan prefix, jadi huruf setelah base bakal dihapus
    """
    for kw in keywords:
        if token.startswith(kw) and token != kw:
            return kw
    return token


def normalize_text(text, keywords):
    """
    Normalisasi kata dengan fungsi normalise_by_prefix()
    """
    tokens = text.lower().split()
    tokens = [normalize_by_prefix(t, keywords) for t in tokens]
    return " ".join(tokens)


def clean_text_advanced(text, lang="en", use_stemming=True):
    """Membersihkan teks dengan standar NLP Professional."""
    # Membuat keyword id untuk stemming kata tidak diKBBI
    KEYWORDS_ID = build_keyword_set(ASPECT_KEYWORDS, "id")
    KEYWORDS_EN = build_keyword_set(ASPECT_KEYWORDS, "en")

    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = str(text).lower()
    print(f"text lower case : {text}")

    # 2. Hapus URL & Mention/Hashtag
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#\w+", "", text)
    print(f"text hashtag : {text}")

    # 3. Hapus Angka (Kecuali yang nempel sama huruf seperti 4g, mp3 biar konteks jalan)
    # Opsional, di sini kita hapus angka murni saja
    text = re.sub(r"\b\d+\b", "", text)
    print(f"text hapus angka : {text}")

    # 4. Handle Tanda Baca untuk Segmentasi (Keep . , ! ? tapi kasih spasi)
    # Tujuannya agar tokenisasi nanti memisahkan "bagus." menjadi "bagus" dan "."
    text = re.sub(r"([.,!?])", r" \1 ", text)
    print(f"text tanda baca : {text}")

    # 5. Hapus karakter simbol aneh (keep alpha-numeric & punctuation)
    text = re.sub(r"[^a-z0-9\s.,!?]", " ", text)
    print(f"text simbol : {text}")

    # 6. Reduksi karakter berulang (Baangeeet -> banget)
    text = reduce_repeating_chars(text)
    print(f"text repeating char : {text}")

    # 7. Normalisasi Spasi
    text = re.sub(r"\s+", " ", text).strip()
    print(f"normalisasi spasi : {text}")

    # 8. Fix kata yg ga di KBBI
    print(f"Temp text sebelum fix uinya : {text}")
    # text = fix_ui_nya(text)  # Stemming kata ui
    text = normalize_text(text, KEYWORDS_ID)
    text = normalize_text(text, KEYWORDS_EN)

    print(f"Temp text setelah fix uinya : {text}")

    # 9. Tokenisasi
    tokens = text.split()

    # 10. Handling per Bahasa
    if lang == "id":
        # Normalisasi Slang
        tokens = [SLANG_MAP.get(t, t) for t in tokens]

        # Stemming Sastrawi (Optional: Bisa dimatikan jika terlalu lambat untuk batch besar)
        # Kita limit hanya stem kalimat < 30 kata agar responsif di Streamlit
        if use_stemming and len(tokens) < 30:
            try:
                # Re-join dulu karena Sastrawi lebih cepat proses string
                temp_text = " ".join(tokens)
                temp_text = stemmer.stem(temp_text)

                tokens = temp_text.split()
            except:
                pass

    # 11. Stopword Removal (Hati-hati dengan Negasi)
    if lang == "id":
        stops = set(stopwords.words("indonesian")) - NEGATION_WORDS
    else:
        stops = set(stopwords.words("english")) - NEGATION_WORDS

    tokens = [t for t in tokens if t not in stops]

    return " ".join(tokens)


# ==========================================
# 4. MODEL MANAGEMENT (CACHING SYSTEM)
# ==========================================


@st.cache_resource(show_spinner=False)
def load_all_models():
    """
    Memuat semua model AI ke RAM. Menggunakan Cache Streamlit
    agar tidak loading ulang setiap ada interaksi user.
    """
    try:
        # Load English Models
        path_en = "Hamusssss12/spotify-absa-english"
        tok_bert_en = AutoTokenizer.from_pretrained(path_en)
        mod_bert_en = AutoModelForSequenceClassification.from_pretrained(path_en)

        # Load Indonesian Models
        path_id = "Hamusssss12/spotify-absa-indonesian"
        tok_bert_id = AutoTokenizer.from_pretrained(path_id)
        mod_bert_id = AutoModelForSequenceClassification.from_pretrained(path_id)
        # Note: LSTM Models kita keep untuk keperluan advanced development/comparison jika perlu
        # Tapi untuk deployment utama, kita pakai Transformer (BERT) karena akurasi lebih tinggi.

        return {"en": (mod_bert_en, tok_bert_en), "id": (mod_bert_id, tok_bert_id)}

    except Exception as e:
        st.error(f"⚠️ Error Critical: Gagal memuat model AI. Pesan Error: {str(e)}")
        st.info("Pastikan folder 'models' berisi hasil ekstrak ZIP yang benar.")
        return None, None


# ==========================================
# 5. INFERENCE ENGINE (OTAK PREDIKSI)
# ==========================================


def detect_language(text):
    """Mendeteksi bahasa input (ID/EN) secara otomatis."""
    try:
        # Deteksi cepat
        lang = detect(text)
        return "id" if lang == "id" or lang == "in" else "en"
    except:
        # Fallback manual check: Cari kata 'yang', 'dan'
        if any(w in text.lower() for w in ["yang", "dan", "di", "aku"]):
            return "id"
        return "en"


def get_bert_prob(text, model, tokenizer):
    """Mengembalikan skor probabilitas POSITIVE (0.0 - 1.0)."""
    # Pindahkan ke CPU untuk deployment (kecuali server ada GPU)
    # Ini aman untuk Streamlit Cloud/Lokal Laptop biasa
    model.to("cpu")

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    return probs[1]  # Probabilitas kelas 1 (Positive)


def get_smart_aspects(segment, lang):
    """
    Mendeteksi aspek + Mengembalikan kata pemicunya.
    Output: [('Audio', 'suara'), ('Price', 'mahal')]
    """
    detected = []
    text_lower = segment.lower()

    # Ambil kamus sesuai bahasa
    vocab = ASPECT_KEYWORDS.get(lang, ASPECT_KEYWORDS["en"])

    for aspect, keywords in vocab.items():
        for key in keywords:
            # Gunakan regex word boundary agar akurat ('ads' not in 'loads')
            pattern = r"\b" + re.escape(key) + r"\b"
            match = re.search(pattern, text_lower)
            if match:
                detected.append((aspect, key))  # Simpan Nama Aspek & Kata Pemicu
                break  # Cukup 1 trigger per aspek per segmen

    return detected


def analyze_single_review_complete(text, models_tuple):
    """
    PIPELINE UTAMA ABSA END-TO-END
    Menerima teks -> Cleaning -> Split Segmen -> Deteksi Aspek -> Scoring BERT.
    """
    # 1. Identifikasi Bahasa & Model
    models_en, models_id = models_tuple
    if not models_en or not models_id:
        return "Error", 0.0, {}, "en"

    lang = detect_language(text)

    # Load pasangan model & tokenizer yang tepat
    if lang == "id":
        model, tokenizer = models_id
    else:
        model, tokenizer = models_en

    # 2. Preprocessing & Segmentasi Kalimat
    # Kita pisah kalimat jika ada tanda baca atau kata hubung kontras
    if lang == "id":
        delimiters = (
            r"("
            r"\.|!|\?|;|,\s|"
            r"\btapi\b|\btp\b|\btetapi\b|\bnamun\b|\bmelainkan\b|\bakan tetapi\b|"
            r"\bpadahal\b|\bsedangkan\b|\bsebaliknya\b|\bjustru\b|"
            r"\bwalaupun\b|\bwalau\b|\bmeskipun\b|\bmeski\b|\bkendati\b|\bbiarpun\b|"
            r"\bcuma\b|\bcman\b|\bcma\b|\bcm\b|\bhanya\b|\bhanya saja\b|"
            r"\bsayang\b|\bsayangnya\b|\bsyg\b|\bdisayangkan\b|"
            r"\bkecuali\b|\bselain itu\b"
            r")"
        )
    else:
        delimiters = (
            r"("
            r"\.|!|\?|;|,\s|"
            r"\bbut\b|\bhowever\b|\byet\b|\bnevertheless\b|\bnonetheless\b|"
            r"\balthough\b|\bthough\b|\beven though\b|\balbeit\b|"
            r"\bdespite\b|\bin spite of\b|\bregardless\b|"
            r"\bwhile\b|\bwhereas\b|\bon the other hand\b|"
            r"\bexcept\b|\bexception\b|\bunless\b|\bbarring\b|"
            r"\bunfortunately\b|\bsadly\b|\bregrettably\b|\bpity\b"
            r")"
        )

    raw_segments = re.split(delimiters, text.lower())
    segments = [s.strip() for s in raw_segments if len(s.split()) >= 2]
    if not segments:
        segments = [text]  # Fallback jika kalimat pendek

    aspect_sentiment_store = {}

    # 3. Loop Analisis per Segmen
    for seg in segments:
        print(f"seg : {seg}")
        seg_clean = clean_text_advanced(seg, lang, use_stemming=True)
        print(f"seg_clean : {seg_clean}")
        # A. Deteksi Aspek & Trigger
        found_aspects = get_smart_aspects(seg_clean, lang)
        print(f"found_aspects : {found_aspects}")
        if found_aspects:
            # B. Hitung Sentimen Segmen ini
            # Preprocess khusus model (pake stemming jika perlu)
            if not seg_clean:
                seg_clean = seg
            pos_prob = get_bert_prob(seg, model, tokenizer)

            # Simpan hasil
            for aspect_name, trigger_word in found_aspects:
                if aspect_name not in aspect_sentiment_store:
                    aspect_sentiment_store[aspect_name] = []

                aspect_sentiment_store[aspect_name].append(
                    {"prob": pos_prob, "trigger": trigger_word}
                )

    # 4. Aggregasi Hasil Aspek (Average & Logic)
    final_aspects_output = {}

    if aspect_sentiment_store:
        for asp, data_list in aspect_sentiment_store.items():
            # Rata-rata probabilitas jika aspek muncul beberapa kali
            avg_prob = np.mean([d["prob"] for d in data_list])

            # Ambil trigger word yang pertama ditemukan (representatif)
            triggers = list(set([d["trigger"] for d in data_list]))
            trigger_str = ", ".join(triggers)

            # Penentuan Label (Threshold 0.5)
            # Area abu-abu: 0.45 - 0.55 -> Neutral (Optional)
            # if 0.45 <= avg_prob <= 0.55:
            #     label = "Neutral/Mixed"
            #     score = 1.0 - (abs(0.5 - avg_prob) * 2)  # Normalisasi confidencenya
            if avg_prob > 0.5 and lang == "en":
                label = "Positive"
                score = avg_prob
            elif avg_prob < 0.5 and lang == "en":
                label = "Negative"
                score = 1.0 - avg_prob
            elif (
                avg_prob > 0.5 and lang == "id"
            ):  # Yang bahasa indonesia terbalik sentimenya
                label = "Negative"
                score = avg_prob
            elif avg_prob < 0.5 and lang == "id":
                label = "Positive"
                score = 1.0 - avg_prob
            final_aspects_output[asp] = {
                "label": label,
                "score": score,
                "trigger": trigger_str,
            }

    # 5. Global Sentiment Prediction (Text Utuh)
    clean_global = clean_text_advanced(text, lang, use_stemming=True)
    global_prob = get_bert_prob(clean_global, model, tokenizer)
    global_label = "Positive" if global_prob > 0.5 else "Negative"
    global_conf = global_prob if global_label == "Positive" else 1.0 - global_prob

    return global_label, global_conf, final_aspects_output, lang


# ==========================================
# 6. FILE HANDLER UTILITIES
# ==========================================


def load_uploaded_file(uploaded_file):
    """Membaca file CSV/Excel ke DataFrame"""
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            print(f"excel : {df}")
        return df
    except Exception as e:
        return None


def find_text_column(df):
    """Mencari kolom teks secara otomatis"""
    print(f"df : {df}")
    candidates = [
        "content",
        "review",
        "text",
        "ulasan",
        "komentar",
        "feedback",
        "reviewText",
    ]
    for col in df.columns:
        list_lower = [c.lower() for c in candidates]
        if col.lower() in [c.lower() for c in candidates]:
            return col
    # Jika tidak ketemu, cari kolom objek pertama yang panjang
    for col in df.select_dtypes(include=["object"]):
        return col
    return None


def convert_df_to_csv(df):
    """Mengubah DF ke CSV string untuk download button"""
    return df.to_csv(index=False).encode("utf-8")
