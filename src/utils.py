import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import setting
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
# 2. CORE LOGIC: PREPROCESSING
# ==========================================


def reduce_repeating_chars(text, max_repeat=2):
    pattern = r"(.)\1{" + str(max_repeat) + r",}"
    return re.sub(pattern, r"\1" * max_repeat, text)


def normalize_slang_id(tokens):
    """Mapping list token berdasarkan kamus slang."""
    return [setting.SLANG_MAP.get(word, word) for word in tokens]


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
    norm_token = None
    for kw in keywords:
        cond = token.startswith(kw) and token != kw and norm_token == None

        if token.startswith(kw) and token != kw and norm_token == None:
            print(f"before kw : {kw}")
            print(f"norm_token : {norm_token}, {cond}")
            print(f"\nterpisah cond 1\n")
            norm_token = kw
            print(f"after norm_token : {norm_token}")
        elif norm_token:
            if token.startswith(kw) and token != kw and (len(kw) > len(norm_token)):
                print(f"before kw : {kw}")
                print(f"norm_token : {norm_token}, {cond}")
                print("\n\nbener terjadi cond 2")
                norm_token = kw
                print(f"after norm_token : {norm_token}")
    if norm_token:
        return norm_token
    return token


def normalize_text(text, keywords):
    """
    Normalisasi kata dengan fungsi normalise_by_prefix()
    """
    tokens = text.lower().split()
    tokens = [normalize_by_prefix(t, keywords) for t in tokens]
    return " ".join(tokens)


def clean_text_advanced(ASPECT_KEYWORDS, text, lang="en", use_stemming=True):
    """Membersihkan teks dengan standar NLP Professional."""
    # Membuat keyword id untuk stemming kata tidak diKBBI
    KEYWORDS_ID = build_keyword_set(ASPECT_KEYWORDS, "id")
    KEYWORDS_EN = build_keyword_set(ASPECT_KEYWORDS, "en")
    KEYWORDS = KEYWORDS_ID.union(KEYWORDS_EN)

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
    text = normalize_text(text, KEYWORDS)

    print(f"Temp text setelah fix uinya : {text}")

    # 9. Tokenisasi
    tokens = text.split()

    # 10. Handling per Bahasa
    if lang == "id":
        # Normalisasi Slang
        tokens = [setting.SLANG_MAP.get(t, t) for t in tokens]

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
        stops = set(stopwords.words("indonesian")) - setting.NEGATION_WORDS
    else:
        stops = set(stopwords.words("english")) - setting.NEGATION_WORDS

    tokens = [t for t in tokens if t not in stops]
    print(" ".join(tokens))
    return " ".join(tokens)


# ==========================================
# 3. MODEL MANAGEMENT (CACHING SYSTEM)
# ==========================================


@st.cache_resource(show_spinner=False)
def load_all_models():
    """
    Memuat semua model AI ke RAM. Menggunakan Cache Streamlit
    agar tidak loading ulang setiap ada interaksi user.
    """
    try:
        # Load English Models
        path_en = "Hamusssss12/spotify-absa-english-v2"
        tok_bert_en = AutoTokenizer.from_pretrained(path_en)
        mod_bert_en = AutoModelForSequenceClassification.from_pretrained(path_en)

        # Load Indonesian Models
        path_id = "Hamusssss12/spotify-absa-indonesian-v2"
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
# 4. INFERENCE ENGINE (OTAK PREDIKSI)
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


def get_bert_prob(text, model, tokenizer, lang):
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

    if lang == "en":
        return probs[1]  # Probabilitas kelas 1 (Positive)
    elif lang == "id":
        return probs[0]  # Probabilitas kelas 0 (Positive)


def get_smart_aspects(ASPECT_KEYWORDS, segment, lang):
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


def analyze_single_review_complete(ASPECT_KEYWORDS, text, models_tuple):
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
        seg_clean = clean_text_advanced(ASPECT_KEYWORDS, seg, lang, use_stemming=True)
        print(f"seg_clean : {seg_clean}")
        # A. Deteksi Aspek & Trigger
        found_aspects = get_smart_aspects(ASPECT_KEYWORDS, seg_clean, lang)
        print(f"found_aspects : {found_aspects}")
        if found_aspects:
            # B. Hitung Sentimen Segmen ini
            # Preprocess khusus model (pake stemming jika perlu)
            if not seg_clean:
                seg_clean = seg
            pos_prob = get_bert_prob(seg, model, tokenizer, lang)

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
            if avg_prob > 0.5:
                label = "Positive"
                score = avg_prob
            elif avg_prob < 0.5:
                label = "Negative"
                score = 1.0 - avg_prob

            final_aspects_output[asp] = {
                "label": label,
                "score": score,
                "trigger": trigger_str,
            }

    # 5. Global Sentiment Prediction (Text Utuh)
    clean_global = clean_text_advanced(ASPECT_KEYWORDS, text, lang, use_stemming=True)
    global_prob = get_bert_prob(clean_global, model, tokenizer, lang)

    global_label = "Positive" if global_prob > 0.5 else "Negative"
    global_conf = global_prob if global_label == "Positive" else 1.0 - global_prob

    return global_label, global_conf, final_aspects_output, lang


# ==========================================
# 5. FILE HANDLER UTILITIES
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
