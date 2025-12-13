import streamlit as st
from collections import Counter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import ast
import re

# ==========================================
# 🎨 COLOR PALETTE (SPOTIFY THEME)
# ==========================================
COLOR_POS = "#1DB954"  # Spotify Green
COLOR_NEG = "#E22134"  # Red
COLOR_NEU = "#B3B3B3"  # Grey
COLOR_BG = "rgba(0,0,0,0)"  # Transparent


# ==========================================
# 1. KPI METRICS (KARTU STATISTIK)
# ==========================================
def display_kpi_metrics(df):
    """Menampilkan total ulasan dan persentase sentimen"""
    if df.empty:
        return

    total_reviews = len(df)
    pos_count = len(df[df["Global Sentiment"] == "Positive"])
    neg_count = len(df[df["Global Sentiment"] == "Negative"])

    pos_pct = (pos_count / total_reviews) * 100
    neg_pct = (neg_count / total_reviews) * 100

    # Layout 3 Kolom
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Total Data", f"{total_reviews:,}", "Ulasan")

    with c2:
        st.metric(
            "Sentimen Positif",
            f"{pos_count:,}",
            f"{pos_pct:.1f}%",
            delta_color="normal",
        )

    with c3:
        st.metric(
            "Sentimen Negatif",
            f"{neg_count:,}",
            f"-{neg_pct:.1f}%",
            delta_color="inverse",
        )

    st.markdown("---")


# ==========================================
# 2. SENTIMENT DONUT CHART
# ==========================================
def plot_sentiment_donut(df):
    """Pie Chart bolong tengah (Donut) untuk Global Sentiment"""
    counts = df["Global Sentiment"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]

    fig = px.pie(
        counts,
        values="Count",
        names="Sentiment",
        hole=0.6,
        color="Sentiment",
        color_discrete_map={"Positive": COLOR_POS, "Negative": COLOR_NEG},
        title="<b>Proporsi Sentimen Global</b>",
    )

    # Styling agar menyatu dengan Dark Mode
    fig.update_layout(
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_BG,
        font=dict(color="white", size=14),
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
    )
    # Menambahkan Text di tengah Donut
    fig.add_annotation(
        text="Sentiment", showarrow=False, font_size=20, font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 3. ASPECT STACKED BAR CHART (STYLED)
# ==========================================
def plot_aspect_bar_chart(df):
    """
    Visualisasi Aspek dengan styling HTML pada label text.
    Count = Bold Putih, Persen = Kuning Emas.
    """
    aspect_data = []
    aspect_cols = [c for c in df.columns if "_Sentiment" in c]

    if not aspect_cols:
        st.warning("Belum ada data aspek yang diproses.")
        return

    for col in aspect_cols:
        aspect_name = col.replace("_Sentiment", "")
        counts = df[col].value_counts()
        if "Positive" in counts:
            aspect_data.append(
                {
                    "Aspect": aspect_name,
                    "Sentiment": "Positive",
                    "Count": counts["Positive"],
                }
            )
        if "Negative" in counts:
            aspect_data.append(
                {
                    "Aspect": aspect_name,
                    "Sentiment": "Negative",
                    "Count": counts["Negative"],
                }
            )

    if not aspect_data:
        st.info("Tidak ada aspek spesifik terdeteksi.")
        return

    df_aspects = pd.DataFrame(aspect_data)

    # Hitung Persentase
    total_per_aspect = df_aspects.groupby("Aspect")["Count"].transform("sum")
    df_aspects["Pct"] = (df_aspects["Count"] / total_per_aspect * 100).round(1)

    # --- STYLING LABEL DENGAN HTML ---
    # <b>{Count}</b> : Angka tebal
    # <span style='...'>...</span> : Ubah warna & ukuran persen
    df_aspects["Label"] = df_aspects.apply(
        lambda x: f"<b>{x['Count']}</b> <span style='color:#FFFFFF; font-weight:normal; font-size:0.9em'>({x['Pct']}%)</span>",
        axis=1,
    )

    fig = px.bar(
        df_aspects,
        x="Count",
        y="Aspect",
        color="Sentiment",
        orientation="h",
        title="<b>Analisis Sentimen per Aspek</b>",
        color_discrete_map={"Positive": COLOR_POS, "Negative": COLOR_NEG},
        text="Label",  # Masukkan kolom label HTML
        template="plotly_dark",
    )

    fig.update_layout(
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_BG,
        font=dict(color="white", size=14),
        xaxis_title="Jumlah Ulasan",
        yaxis_title="",
        yaxis={"categoryorder": "total ascending"},
        barmode="stack",
    )

    # Update Traces agar HTML terbaca
    fig.update_traces(
        textposition="inside",
        insidetextanchor="middle",
        texttemplate="%{text}",  # PENTING: Memaksa Plotly render HTML
        hovertemplate="<b>%{y}</b><br>Sentimen: %{data.name}<br>Jumlah: %{x}<br>Persentase: %{customdata[0]}%<extra></extra>",
        customdata=df_aspects[["Pct"]],  # Kirim data persen ke tooltip
    )

    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 4. WORDCLOUD GENERATOR
# ==========================================
def generate_wordcloud(df, sentiment_filter):
    """Membuat WordCloud dari ulasan berdasarkan filter sentimen"""

    # Filter Data
    subset = df[df["Global Sentiment"] == sentiment_filter]

    if subset.empty:
        st.caption("Tidak ada data untuk kategori ini.")
        return
    print("subset")
    print(subset.columns)
    text_combined = " ".join(subset["Original Text"].astype(str).tolist())

    # Setup Warna (Hijau untuk Positif, Merah Api untuk Negatif)
    colormap = "Greens" if sentiment_filter == "Positive" else "Reds"

    wc = WordCloud(
        width=800,
        height=400,
        background_color="#121212",  # Dark Background
        colormap=colormap,
        max_words=100,
        contour_color="white",
        contour_width=1,
    ).generate(text_combined)

    # Tampilkan menggunakan Matplotlib di Streamlit
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#121212")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig)


# ==========================================
# 5. TRIGGER SENTIMENT CHART
# ==========================================
def plot_trigger_sentiment_chart(df):
    """
    Visualisasi Trigger Words dengan styling HTML yang lebih cantik.
    """
    if "Aspects JSON" not in df.columns:
        st.warning("Data aspek detail tidak ditemukan.")
        return

    trigger_data = []

    def clean_json_str(s):
        return re.sub(r"np\.float32\(([^)]+)\)", r"\1", str(s))

    for _, row in df.iterrows():
        try:
            json_str = clean_json_str(row["Aspects JSON"])
            if pd.isna(json_str) or json_str == "{}":
                continue
            aspect_data = ast.literal_eval(json_str)
            for details in aspect_data.values():
                trigger_str = details.get("trigger", "")
                label = details.get("label", "Negative")
                if trigger_str:
                    words = [w.strip() for w in trigger_str.split(",")]
                    for w in words:
                        if w:
                            trigger_data.append({"Keyword": w, "Sentiment": label})
        except Exception:
            continue

    if not trigger_data:
        st.info("Belum ada kata kunci spesifik.")
        return

    df_trig = pd.DataFrame(trigger_data)
    df_counts = (
        df_trig.groupby(["Keyword", "Sentiment"]).size().reset_index(name="Count")
    )

    # Sorting & Filtering Top 25
    df_total = df_counts.groupby("Keyword")["Count"].sum().reset_index(name="Total")
    top_keywords = df_total.nlargest(25, "Total")["Keyword"].tolist()
    df_final = df_counts[df_counts["Keyword"].isin(top_keywords)].copy()

    # Hitung Persentase
    total_per_keyword = df_final.groupby("Keyword")["Count"].transform("sum")
    df_final["Pct"] = (df_final["Count"] / total_per_keyword * 100).round(1)

    # --- STYLING LABEL ---
    # Count: Putih Tebal
    # Persen: Kuning Emas (#FFD700), Font agak kecil
    df_final["Label"] = df_final.apply(
        lambda x: f"<b>{x['Count']}</b> <span style='color:#FFFFFF; font-size:0.85em'>({x['Pct']}%)</span>",
        axis=1,
    )

    dynamic_height = 400 + (len(top_keywords) * 35)

    fig = px.bar(
        df_final,
        x="Count",
        y="Keyword",
        color="Sentiment",
        orientation="h",
        title="<b>Frekuensi Kata Pemicu per Sentimen</b>",
        color_discrete_map={"Positive": COLOR_POS, "Negative": COLOR_NEG},
        text="Label",
        template="plotly_dark",
        height=dynamic_height,
    )

    fig.update_layout(
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_BG,
        font=dict(color="white", size=14),
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Jumlah Kemunculan",
        yaxis_title="",
        barmode="stack",
        margin=dict(l=150, r=50, t=80, b=50),
        legend=dict(orientation="h", y=1.02, x=0, title_text=""),
    )

    # Update Traces untuk render HTML dan Tooltip Bagus
    fig.update_traces(
        textposition="inside",
        insidetextanchor="middle",
        texttemplate="%{text}",  # Render HTML
        hovertemplate="<b>%{y}</b><br>Sentimen: %{data.name}<br>Jumlah: %{x}<br>Persentase: %{customdata[0]}%<extra></extra>",
        customdata=df_final[["Pct"]],
    )

    st.plotly_chart(fig, use_container_width=True)
