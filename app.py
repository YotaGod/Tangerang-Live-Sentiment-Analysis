import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import io
import json
import numpy as np
import keras
import re
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemover import StopWordRemover
from Sastrawi.Dictionary.ArrayDictionary import ArrayDictionary
from nltk.corpus import stopwords

# ===========================================================================
# PAGE CONFIG
# ===========================================================================

st.set_page_config(
    page_title="SentiAI — Analisis Sentimen",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================================================================
# GLOBAL CSS
# ===========================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.main { background: #0d0d14; }
section[data-testid="stSidebar"] > div:first-child { background: #10101a; border-right: 1px solid rgba(99,102,241,.25); }
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: #c4c4d4 !important; }
::-webkit-scrollbar { width:6px; } ::-webkit-scrollbar-track { background:#0d0d14; }
::-webkit-scrollbar-thumb { background:#3b3b5e; border-radius:3px; }

.hero {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4c1d95 100%);
    border: 1px solid rgba(129,140,248,.3);
    border-radius: 20px;
    padding: 44px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(79,70,229,.25);
}
.hero::before {
    content: '';
    position: absolute; top: -40px; right: -40px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(167,139,250,.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 { margin:0 0 10px; font-size:2.4em; font-weight:800; color:#fff; letter-spacing:-.5px; }
.hero p  { margin:0; font-size:1.05em; color:#c4b5fd; line-height:1.7; }
.hero .badge {
    display:inline-block; background:rgba(167,139,250,.2); border:1px solid rgba(167,139,250,.4);
    color:#c4b5fd; padding:4px 12px; border-radius:20px; font-size:.8em; font-weight:600;
    margin-bottom:14px; letter-spacing:.5px;
}
.sec-title {
    font-size:1.3em; font-weight:700; color:#e0e0f0;
    border-left:4px solid #6366f1; padding-left:14px;
    margin:28px 0 18px;
}
.res-positive {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981; border-radius:16px; padding:28px;
    color:#fff; margin:20px 0;
    box-shadow: 0 8px 32px rgba(16,185,129,.2);
}
.res-negative {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444; border-radius:16px; padding:28px;
    color:#fff; margin:20px 0;
    box-shadow: 0 8px 32px rgba(239,68,68,.2);
}
.res-positive h2, .res-negative h2 { margin:0 0 10px; font-size:2em; font-weight:800; }
.res-positive p,  .res-negative p  { margin:0; font-size:1em; opacity:.9; line-height:1.7; }
.feat-card {
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.08);
    border-radius:14px; padding:22px; height:100%;
    transition: border-color .3s, transform .3s;
}
.feat-card:hover { border-color:#6366f1; transform:translateY(-3px); }
.feat-card h4 { margin:0 0 10px; font-size:1.05em; font-weight:700; }
.feat-card p  { margin:0; color:#9ca3af; font-size:.9em; line-height:1.6; }
.empty-state {
    text-align:center; padding:64px 32px;
    background: rgba(255,255,255,.03);
    border:2px dashed rgba(99,102,241,.3);
    border-radius:20px; margin:20px 0;
}
.empty-state h3 { color:#e0e0f0; margin:0 0 12px; font-size:1.6em; }
.empty-state p  { color:#6b7280; font-size:1em; margin:0; }
[data-testid="stMetric"] {
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.08);
    border-radius:14px; padding:16px !important;
}
[data-testid="stMetricLabel"] { color:#9ca3af !important; font-size:.85em !important; font-weight:600 !important; }
[data-testid="stMetricValue"] { color:#a5b4fc !important; font-size:1.9em !important; font-weight:800 !important; }
.stButton > button { border-radius:10px; font-weight:600; transition: all .25s ease; }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border:none !important; color:#fff !important;
    box-shadow: 0 4px 15px rgba(99,102,241,.35);
}
.stButton > button[kind="primary"]:hover { transform:translateY(-2px); box-shadow: 0 8px 20px rgba(99,102,241,.5); }
.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,.05) !important;
    border:1px solid rgba(255,255,255,.15) !important;
    color:#c4c4d4 !important;
}
.stButton > button[kind="secondary"]:hover { border-color:#6366f1 !important; background: rgba(99,102,241,.1) !important; }
.stTextArea textarea {
    background: rgba(255,255,255,.05) !important;
    border:1px solid rgba(99,102,241,.35) !important;
    border-radius:12px !important; color:#e0e0f0 !important; font-size:.97em;
}
.stTextArea textarea:focus { border-color:#818cf8 !important; box-shadow:0 0 0 3px rgba(99,102,241,.2) !important; }
.stAlert { border-radius:12px !important; }
.badge-active   { display:inline-block; background:#064e3b; color:#6ee7b7; border:1px solid #10b981; padding:5px 14px; border-radius:20px; font-size:.85em; font-weight:700; }
.badge-inactive { display:inline-block; background:#450a0a; color:#fca5a5; border:1px solid #ef4444; padding:5px 14px; border-radius:20px; font-size:.85em; font-weight:700; }
.sidebar-logo { text-align:center; padding:24px 0 20px; border-bottom:1px solid rgba(99,102,241,.25); margin-bottom:20px; }
.sidebar-logo h2 { margin:0; color:#818cf8; font-size:1.6em; font-weight:800; letter-spacing:-1px; }
.sidebar-logo p  { margin:6px 0 0; color:#6b7280; font-size:.85em; }
.app-footer {
    text-align:center; padding:28px; margin-top:48px;
    background: rgba(255,255,255,.03);
    border:1px solid rgba(255,255,255,.07);
    border-radius:16px; color:#6b7280;
}
.app-footer strong { color:#818cf8; }
[data-testid="stExpander"] { border:1px solid rgba(255,255,255,.08) !important; border-radius:12px !important; }
</style>
""", unsafe_allow_html=True)

# ===========================================================================
# NLTK SETUP
# ===========================================================================

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# ===========================================================================
# PREPROCESSING FUNCTIONS
# ===========================================================================

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[-+]?[0-9]+', '', text)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@st.cache_resource
def load_slang_dictionary():
    try:
        slang_df = pd.read_csv('slang.csv', sep=';', header=None, names=['slang', 'formal'], skiprows=1)
        slang_df['slang'] = slang_df['slang'].astype(str).str.strip().str.lower()
        slang_df['formal'] = slang_df['formal'].astype(str).str.strip().str.lower()
        return dict(zip(slang_df['slang'], slang_df['formal']))
    except:
        return {}

def normalize_slang(text, slang_dict):
    if not text: return ""
    return ' '.join([slang_dict.get(w, w) for w in text.split()])

@st.cache_resource
def create_stopword_remover():
    factory = StopWordRemoverFactory()
    sw_list = factory.get_stop_words()
    extra = ['aja','saja','banget','kok','lagi','punya','terus','nih','loh',
             'deh','dong','gitu','gini','udah','tapi','yg','nya','sih','pun','lah','anjir']
    for w in extra:
        if w not in sw_list: sw_list.append(w)
    for w in stopwords.words("english"):
        if w not in sw_list: sw_list.append(w)
    for neg in ['tidak','gak','nggak','kurang','bukan','jangan','belum']:
        if neg in sw_list: sw_list.remove(neg)
    return StopWordRemover(ArrayDictionary(sw_list))

def remove_stopwords(text, remover):
    return remover.remove(text) if text else ""

def handle_negation(text):
    neg_words = ['tidak','gak','nggak','kurang','bukan','jangan','belom','belum']
    words = text.split(); result = []; i = 0
    while i < len(words):
        if words[i] in neg_words and i + 1 < len(words):
            result.append(f"{words[i]}_{words[i+1]}"); i += 2
        else:
            result.append(words[i]); i += 1
    return " ".join(result)

def load_stemmer():
    if 'stemmer_instance' not in st.session_state:
        factory = StemmerFactory()
        st.session_state['stemmer_instance'] = factory.create_stemmer()
    return st.session_state['stemmer_instance']

def stem_text(text, stemmer):
    if not text: return ""
    result = []
    for word in text.split():
        if "_" in word:
            parts = word.split("_")
            result.append("_".join([stemmer.stem(p) for p in parts]))
        else:
            result.append(stemmer.stem(word))
    return " ".join(result)

# ===========================================================================
# ASSET LOADING
# ===========================================================================

def load_assets():
    try:
        from tf_keras.preprocessing.text import tokenizer_from_json
        from tf_keras.preprocessing.sequence import pad_sequences
        with open('tokenizer (14).json', 'r', encoding='utf-8') as f:
            data = f.read()
        if data.startswith('"') and data.endswith('"'):
            data = json.loads(data)
        tokenizer = tokenizer_from_json(data)
        model = keras.models.load_model('model_tangerang_live_biner.keras', compile=False)
        slang_dict = load_slang_dictionary()
        stopword_remover = create_stopword_remover()
        stemmer = load_stemmer()
        return model, tokenizer, slang_dict, stopword_remover, stemmer
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None, None, None, None

model, tokenizer, slang_dict, stopword_remover, stemmer = load_assets()

# ===========================================================================
# SIDEBAR
# ===========================================================================

with st.sidebar:
    st.markdown("""
    <div class='sidebar-logo'>
        <h2>🧠 SentiAI</h2>
        <p>Sentiment Analysis Engine</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🧭 Navigasi")
    page = st.radio("Menu:", [
        "📝 Analisis Teks",
        "📊 Dashboard",
        "📥 Scraper Play Store"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("#### ⚙️ Status Sistem")
    if model:
        st.markdown('<span class="badge-active">✅ Model Aktif</span>', unsafe_allow_html=True)
        st.caption("Bi-LSTM · Binary · Max 50 kata")
    else:
        st.markdown('<span class="badge-inactive">❌ Model Tidak Dimuat</span>', unsafe_allow_html=True)
        st.error("Periksa file model di folder web/", icon="🚨")

    st.markdown("---")
    st.markdown("#### 🏷️ Kelas Sentimen")
    st.markdown("""
    <div style='background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.3);border-radius:10px;padding:12px 14px;margin:6px 0;'>
        <strong style='color:#6ee7b7;'>😊 Positif</strong><br>
        <small style='color:#9ca3af;'>Kepuasan, pujian, rekomendasi</small>
    </div>
    <div style='background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);border-radius:10px;padding:12px 14px;margin:6px 0;'>
        <strong style='color:#fca5a5;'>😞 Negatif</strong><br>
        <small style='color:#9ca3af;'>Keluhan, kritik, kekecewaan</small>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("💡 Contoh Ulasan"):
        st.markdown("**Positif:**")
        st.code("Aplikasi sangat membantu dan mudah digunakan!", language='text')
        st.markdown("**Negatif:**")
        st.code("Sering error dan force close, sangat mengecewakan.", language='text')

    st.markdown("""
    <div style='text-align:center;margin-top:30px;color:#4b5563;font-size:.8em;'>
        Powered by <strong style='color:#818cf8;'>Bi-LSTM + FastText</strong><br>
        Skripsi 2026
    </div>
    """, unsafe_allow_html=True)

# ===========================================================================
# PAGE: SINGLE ANALYSIS
# ===========================================================================

def show_single_analysis():
    st.markdown("""
    <div class='hero'>
        <div class='badge'>🧠 NATURAL LANGUAGE PROCESSING</div>
        <h1>Analisis Sentimen Ulasan</h1>
        <p>Masukkan teks ulasan dalam Bahasa Indonesia. Model Bi-LSTM dengan FastText Embedding
        akan menganalisis sentimen secara otomatis dalam hitungan detik.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sec-title'>✍️ Masukkan Ulasan</div>", unsafe_allow_html=True)
    user_input = st.text_area(
        label="Teks ulasan",
        placeholder="Ketik ulasan di sini… Contoh: Aplikasi ini sangat membantu, fiturnya lengkap dan responsif.",
        height=160,
        label_visibility="collapsed"
    )
    if len(user_input) > 0:
        st.caption(f"📝 {len(user_input)} karakter · {len(user_input.split())} kata")

    col_btn1, col_btn2 = st.columns([5, 1])
    with col_btn1:
        analyze_btn = st.button("🔍 Analisis Sekarang", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🗑️ Reset", use_container_width=True):
            st.session_state.clear(); st.rerun()

    st.markdown("---")

    if analyze_btn and user_input and model:
        with st.spinner('🔄 Memproses ulasan...'):
            try:
                from tf_keras.preprocessing.sequence import pad_sequences

                text_cleaned = clean_text(user_input)
                text_norm    = normalize_slang(text_cleaned, slang_dict)
                text_neg     = handle_negation(text_norm)
                text_stopped = remove_stopwords(text_neg, stopword_remover)
                text_final   = stem_text(text_stopped, stemmer)

                sequences  = tokenizer.texts_to_sequences([text_final])
                padded     = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')
                prediction = model.predict(padded, verbose=0)[0]

                if len(prediction.shape) == 0 or prediction.shape[0] == 1:
                    prob_pos = float(prediction) if len(prediction.shape) == 0 else float(prediction[0])
                    prob_neg = 1 - prob_pos
                else:
                    prob_neg = float(prediction[0])
                    prob_pos = float(prediction[1])

                confidence  = max(prob_pos, prob_neg) * 100
                final_label = "Positif" if prob_pos > prob_neg else "Negatif"
                class_idx   = 1 if prob_pos > prob_neg else 0

                if final_label == "Positif":
                    st.markdown("""
                    <div class='res-positive'>
                        <h2>😊 Sentimen Positif 🎉</h2>
                        <p>Ulasan ini mengandung sentimen <strong>positif</strong>. Model mendeteksi ekspresi kepuasan, pujian, atau rekomendasi.</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='res-negative'>
                        <h2>😞 Sentimen Negatif 💭</h2>
                        <p>Ulasan ini mengandung sentimen <strong>negatif</strong>. Model mendeteksi keluhan, kritik, atau ekspresi kekecewaan.</p>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<div class='sec-title'>📈 Detail Analisis</div>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("🎯 Tingkat Keyakinan", f"{confidence:.1f}%")
                qual = "⚠️ Perlu Review" if confidence < 70 else ("✅ Cukup Baik" if confidence < 85 else "🌟 Sangat Yakin")
                m2.metric("📊 Kualitas Prediksi", qual)
                m3.metric("🔤 Panjang Ulasan", f"{len(user_input.split())} kata")

                st.markdown("---")

                col_chart, col_bar = st.columns(2)
                with col_chart:
                    st.markdown("#### 🥧 Distribusi Probabilitas")
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['Negatif', 'Positif'],
                        values=[prob_neg * 100, prob_pos * 100],
                        hole=0.5,
                        marker_colors=['#ef4444', '#10b981'],
                        textfont_size=13
                    )])
                    fig_pie.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#c4c4d4', legend=dict(bgcolor='rgba(0,0,0,0)'),
                        margin=dict(t=10, b=10, l=10, r=10), height=280
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col_bar:
                    st.markdown("#### 📊 Perbandingan Probabilitas")
                    fig_bar = go.Figure(data=[
                        go.Bar(name='Negatif', x=['Negatif'], y=[prob_neg * 100],
                               marker_color='#ef4444', text=[f'{prob_neg*100:.1f}%'], textposition='outside'),
                        go.Bar(name='Positif', x=['Positif'], y=[prob_pos * 100],
                               marker_color='#10b981', text=[f'{prob_pos*100:.1f}%'], textposition='outside'),
                    ])
                    fig_bar.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#c4c4d4', showlegend=False, yaxis_range=[0, 115],
                        margin=dict(t=10, b=10, l=10, r=10), height=280
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                with st.expander("🔬 Detail Proses Preprocessing"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**📥 Teks Asli:**"); st.code(user_input, language='text')
                        st.markdown("**1️⃣ Cleaning:**"); st.caption("Hapus URL, angka, emoji, karakter khusus")
                        st.code(text_cleaned, language='text')
                    with c2:
                        st.markdown("**2️⃣ Normalisasi Slang:**")
                        st.code(text_norm if text_norm != text_cleaned else "*(tidak ada perubahan)*", language='text')
                        st.markdown("**3️⃣ Handling Negasi:**")
                        st.code(text_neg if text_neg != text_norm else "*(tidak ada perubahan)*", language='text')
                        st.markdown("**4️⃣ Stopword + Stemming:**")
                        st.code(text_final if text_final != text_neg else "*(tidak ada perubahan)*", language='text')

                st.markdown("---")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("🔄 Analisis Ulasan Lain", type="secondary", use_container_width=True):
                        st.rerun()
                with c2:
                    result_text = f"Ulasan: {user_input}\nSentimen: {final_label}\nKeyakinan: {confidence:.1f}%"
                    st.download_button("📥 Unduh Hasil", result_text,
                                       file_name="hasil_analisis.txt", mime="text/plain",
                                       use_container_width=True)
            except Exception as e:
                st.error(f"Terjadi kesalahan: {type(e).__name__}")
                st.exception(e)

    elif analyze_btn and not model:
        st.error("❌ Model belum dimuat. Periksa file model dan restart aplikasi.")

    elif not user_input:
        st.markdown("""
        <div class='empty-state'>
            <h3>👋 Selamat Datang di SentiAI!</h3>
            <p>Tulis ulasan di atas lalu klik <strong>Analisis Sekarang</strong> untuk memulai.<br>
            Model akan mengidentifikasi sentimen positif atau negatif secara otomatis.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='sec-title'>✨ Keunggulan Sistem</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        feats = [
            ("🧠", "#818cf8", "Bi-LSTM + FastText",
             "Arsitektur Bidirectional LSTM dengan pre-trained FastText Bahasa Indonesia untuk akurasi tinggi."),
            ("🗣️", "#34d399", "Memahami Bahasa Indonesia",
             "Dilengkapi normalisasi slang, handling negasi, dan stopword removal untuk Bahasa Indonesia."),
            ("⚡", "#f59e0b", "Analisis Instan",
             "Hasil prediksi dalam hitungan detik disertai visualisasi probabilitas yang mudah dipahami."),
        ]
        for col, (icon, color, title, desc) in zip([c1, c2, c3], feats):
            col.markdown(f"""
            <div class='feat-card'>
                <h4 style='color:{color};'>{icon} {title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)


# ===========================================================================
# PAGE: DASHBOARD
# ===========================================================================

@st.cache_data(show_spinner=False)
def predict_batch(texts):
    from tf_keras.preprocessing.sequence import pad_sequences
    progress_bar = st.progress(0)
    status_text  = st.empty()
    cleaned = []
    total = len(texts)
    for i, text in enumerate(texts):
        if i % 100 == 0 or i == total - 1:
            progress_bar.progress((i + 1) / total)
            status_text.text(f"Memproses {i+1}/{total} teks...")
        t = clean_text(str(text))
        t = normalize_slang(t, slang_dict)
        t = handle_negation(t)
        t = remove_stopwords(t, stopword_remover)
        t = stem_text(t, stemmer)
        cleaned.append(t)
    status_text.text("Tokenisasi dan padding...")
    sequences = tokenizer.texts_to_sequences(cleaned)
    padded    = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')
    status_text.text("Memprediksi sentimen dalam batch...")
    predictions = model.predict(padded, batch_size=128, verbose=0)
    labels = []
    for pred in predictions:
        if len(pred.shape) == 0 or pred.shape[0] == 1:
            p = float(pred) if len(pred.shape) == 0 else float(pred[0])
            labels.append("Positif" if p > 0.5 else "Negatif")
        else:
            labels.append("Positif" if pred[1] > pred[0] else "Negatif")
    progress_bar.empty(); status_text.empty()
    return labels


def show_dashboard():
    st.markdown("""
    <div class='hero'>
        <div class='badge'>📊 DASHBOARD ANALITIK</div>
        <h1>Dashboard Analisis Sentimen</h1>
        <p>Visualisasi dan analisis interaktif dari dataset ulasan pengguna.
        Gunakan data CSV yang tersedia atau unggah dataset Anda sendiri.</p>
    </div>
    """, unsafe_allow_html=True)

    data_source = st.radio(
        "Sumber Data:",
        ["📂 Gunakan Data Tersedia", "📤 Unggah File CSV Baru"],
        horizontal=True
    )

    df = None
    if "Gunakan Data Tersedia" in data_source:
        try:
            df = pd.read_csv('hasil_review_Tangerang_Live.csv')
            st.success(f"✅ Berhasil memuat data — {len(df):,} baris")
        except:
            st.error("❌ File CSV tidak ditemukan. Pastikan file ada di folder web/")
    else:
        uploaded = st.file_uploader(
            "Upload CSV (kolom ulasan: 'Review Teks', 'review', atau 'text')",
            type=['csv']
        )
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.success(f"✅ File berhasil dimuat — {len(df):,} baris")
            except:
                try:
                    uploaded.seek(0)
                    df = pd.read_csv(uploaded, sep=';')
                    st.success(f"✅ File berhasil dimuat — {len(df):,} baris")
                except Exception as e2:
                    st.error(f"❌ Gagal membaca CSV: {e2}")

    if df is None:
        return

    text_col = None
    for c in ['Review Teks', 'review', 'text', 'content']:
        if c in df.columns:
            text_col = c; break

    if text_col is None:
        st.error("❌ Kolom teks ulasan tidak ditemukan ('Review Teks', 'review', atau 'text').")
        return

    st.markdown("---")

    if 'Sentimen' not in df.columns:
        st.info("🔄 Memulai prediksi batch... Proses ini mungkin memakan waktu untuk dataset besar.")
        df['Sentimen'] = predict_batch(df[text_col].tolist())
        st.success("✅ Prediksi selesai!")
    else:
        st.success("✅ Data sudah memiliki label Sentimen.")

    st.markdown("<div class='sec-title'>📈 Ringkasan Metrik</div>", unsafe_allow_html=True)
    total_reviews = len(df)
    total_pos = len(df[df['Sentimen'] == 'Positif'])
    total_neg = len(df[df['Sentimen'] == 'Negatif'])
    pct_pos   = total_pos / total_reviews * 100 if total_reviews else 0

    avg_rating = "N/A"
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        if not df['rating'].dropna().empty:
            avg_rating = f"{df['rating'].mean():.1f} ⭐"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📋 Total Ulasan",     f"{total_reviews:,}")
    m2.metric("😊 Sentimen Positif", f"{total_pos:,}")
    m3.metric("😞 Sentimen Negatif", f"{total_neg:,}")
    m4.metric("⭐ Rata-rata Rating", avg_rating)

    st.markdown("<div class='sec-title'>📊 Visualisasi Data</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Distribusi Sentimen")
        fig_pie = px.pie(
            df, names='Sentimen', color='Sentimen',
            color_discrete_map={'Positif': '#10b981', 'Negatif': '#ef4444'},
            hole=0.45
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#c4c4d4', legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        if 'rating' in df.columns and not df['rating'].dropna().empty:
            st.markdown("#### Distribusi Rating per Sentimen")
            fig_bar = px.histogram(
                df, x='rating', color='Sentimen', barmode='group',
                color_discrete_map={'Positif': '#10b981', 'Negatif': '#ef4444'}
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#c4c4d4', legend=dict(bgcolor='rgba(0,0,0,0)'),
                margin=dict(t=10, b=10)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.markdown("#### Proporsi Sentimen")
            fig_prop = go.Figure(data=[
                go.Bar(name='Positif', x=['Positif'], y=[total_pos],
                       marker_color='#10b981', text=[f'{pct_pos:.1f}%'], textposition='outside'),
                go.Bar(name='Negatif', x=['Negatif'], y=[total_neg],
                       marker_color='#ef4444', text=[f'{100-pct_pos:.1f}%'], textposition='outside'),
            ])
            fig_prop.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#c4c4d4', showlegend=False, margin=dict(t=10, b=10)
            )
            st.plotly_chart(fig_prop, use_container_width=True)

    for date_col in ['date', 'tanggal', 'at']:
        if date_col in df.columns:
            try:
                df['_date'] = pd.to_datetime(df[date_col], format='mixed', errors='coerce').dt.date
                trend = df.dropna(subset=['_date']).groupby(['_date', 'Sentimen']).size().reset_index(name='count')
                st.markdown("#### Tren Sentimen Seiring Waktu")
                fig_line = px.line(
                    trend, x='_date', y='count', color='Sentimen',
                    color_discrete_map={'Positif': '#10b981', 'Negatif': '#ef4444'}
                )
                fig_line.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#c4c4d4', legend=dict(bgcolor='rgba(0,0,0,0)'),
                    margin=dict(t=10, b=10)
                )
                st.plotly_chart(fig_line, use_container_width=True)
            except:
                pass
            break

    st.markdown("<div class='sec-title'>☁️ Word Cloud</div>", unsafe_allow_html=True)
    wc1, wc2 = st.columns(2)

    def make_wordcloud(texts, title, cmap):
        combined = " ".join(texts)
        if not combined.strip(): return None
        wc = WordCloud(width=800, height=400, background_color='#10101a', colormap=cmap).generate(combined)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
        ax.set_title(title, color='white', pad=16, fontsize=16, fontweight='bold')
        fig.patch.set_facecolor('#10101a')
        return fig

    with wc1:
        pos_texts = df[df['Sentimen'] == 'Positif'][text_col].dropna().astype(str).tolist()
        if pos_texts:
            fig = make_wordcloud(pos_texts, '😊 Kata Ulasan Positif', 'Greens')
            if fig: st.pyplot(fig)
    with wc2:
        neg_texts = df[df['Sentimen'] == 'Negatif'][text_col].dropna().astype(str).tolist()
        if neg_texts:
            fig = make_wordcloud(neg_texts, '😞 Kata Ulasan Negatif', 'Reds')
            if fig: st.pyplot(fig)

    st.markdown("<div class='sec-title'>🗂️ Tabel Data</div>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "💾 Download Hasil Analisis (CSV)", csv,
        file_name="hasil_analisis_sentimen.csv", mime="text/csv",
        type="primary", use_container_width=True
    )


# ===========================================================================
# PAGE: SCRAPER
# ===========================================================================

def show_scraper():
    st.markdown("""
    <div class='hero'>
        <div class='badge'>📥 SCRAPER</div>
        <h1>Scraper Google Play Store</h1>
        <p>Tarik ulasan pengguna langsung dari Google Play Store.
        Masukkan URL atau App ID aplikasi yang ingin dianalisis.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sec-title'>⚙️ Pengaturan Scraping</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        app_id_input = st.text_input(
            "URL / App ID Google Play",
            placeholder="https://play.google.com/store/apps/details?id=com.example.app"
        )
        sort_by = st.selectbox("Urutan", ["Terbaru", "Paling Relevan"])
    with c2:
        count        = st.slider("Jumlah Ulasan", 100, 10000, 1000, 100)
        filter_score = st.selectbox("Filter Rating", ["Semua","Bintang 1","Bintang 2","Bintang 3","Bintang 4","Bintang 5"])

    if st.button("🚀 Mulai Scraping", type="primary", use_container_width=True):
        if not app_id_input.strip():
            st.warning("Masukkan URL atau App ID terlebih dahulu!"); return
        try:
            from urllib.parse import urlparse, parse_qs
            from google_play_scraper import Sort, reviews
            app_id = app_id_input.strip()
            if "id=" in app_id:
                qs = parse_qs(urlparse(app_id).query)
                if 'id' in qs: app_id = qs['id'][0]
            sort_enum    = Sort.NEWEST if sort_by == "Terbaru" else Sort.MOST_RELEVANT
            score_filter = None if filter_score == "Semua" else int(filter_score.split()[1])
            with st.spinner(f"Menarik {count} ulasan untuk '{app_id}'..."):
                result, _ = reviews(app_id, lang='id', country='id',
                                    sort=sort_enum, count=count,
                                    filter_score_with=score_filter)
            if result:
                st.success(f"✅ Berhasil menarik {len(result)} ulasan!")
                df_r = pd.DataFrame(result).rename(columns={'content': 'Review Teks', 'score': 'rating', 'at': 'date'})
                df_r['date'] = pd.to_datetime(df_r['date'])
                keep = [c for c in ['reviewId','userName','Review Teks','rating','thumbsUpCount','date'] if c in df_r.columns]
                df_r = df_r[keep]
                st.markdown("<div class='sec-title'>📋 Pratinjau Hasil</div>", unsafe_allow_html=True)
                st.dataframe(df_r.head(100), use_container_width=True)
                csv = df_r.to_csv(index=False).encode('utf-8')
                st.download_button(
                    f"💾 Download {len(result)} Ulasan (CSV)", csv,
                    file_name=f"{app_id}_reviews.csv", mime="text/csv",
                    type="primary", use_container_width=True
                )
            else:
                st.warning("Tidak ada ulasan ditemukan.")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")


# ===========================================================================
# ROUTING
# ===========================================================================

if   page == "📝 Analisis Teks":       show_single_analysis()
elif page == "📊 Dashboard":           show_dashboard()
elif page == "📥 Scraper Play Store":  show_scraper()
else:                                  show_single_analysis()

# ===========================================================================
# FOOTER
# ===========================================================================

st.markdown("---")
st.markdown("""
<div class='app-footer'>
    <p>
        🧠 <strong>SentiAI</strong> — Analisis Sentimen Ulasan Pengguna Berbasis <strong>Bidirectional LSTM + FastText</strong><br>
        <small>© 2026 · Skripsi Ilmu Komputer · Semua hak dilindungi</small>
    </p>
</div>
""", unsafe_allow_html=True)
