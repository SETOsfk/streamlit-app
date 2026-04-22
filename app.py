import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc, roc_auc_score)
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════
# SAYFA AYARLARI
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Veri Madenciliği Master Paneli",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════
# PROFESYONEL CSS
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main .block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1400px; }

    .hero-band {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px; padding: 28px 36px; margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }
    .hero-band h1 { color: #e2e8f0; font-size: 1.9rem; font-weight: 700; margin: 0; }
    .hero-band p  { color: #94a3b8; font-size: 0.88rem; margin: 6px 0 0 0; }
    .hero-badge {
        display: inline-block; background: rgba(99,102,241,0.25);
        color: #a5b4fc; border: 1px solid rgba(99,102,241,0.4);
        border-radius: 999px; padding: 3px 12px; font-size: 0.78rem;
        font-weight: 600; margin-right: 6px; margin-top: 10px;
    }

    .kpi-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #334155; border-radius: 14px;
        padding: 20px 24px; text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2); transition: transform .2s;
    }
    .kpi-card:hover { transform: translateY(-2px); }
    .kpi-icon  { font-size: 1.8rem; margin-bottom: 4px; }
    .kpi-value { font-size: 1.75rem; font-weight: 700; color: #e2e8f0; line-height: 1.1; }
    .kpi-label { font-size: 0.78rem; color: #64748b; font-weight: 500; margin-top: 4px;
                 text-transform: uppercase; letter-spacing: .05em; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: #0f172a;
        border-radius: 12px; padding: 6px; flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; padding: 8px 14px;
        font-weight: 600; font-size: 0.82rem;
        color: #94a3b8 !important; background: transparent !important;
        border: none !important; white-space: nowrap;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: #fff !important; box-shadow: 0 2px 8px rgba(99,102,241,.4);
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }

    .section-header {
        display: flex; align-items: center; gap: 10px;
        background: linear-gradient(90deg, #1e293b, transparent);
        border-left: 4px solid #6366f1; border-radius: 0 8px 8px 0;
        padding: 10px 18px; margin: 20px 0 14px 0;
    }
    .section-header span { font-size: 1.05rem; font-weight: 700; color: #e2e8f0; }

    .result-pass {
        background: rgba(16,185,129,.12); border: 1px solid rgba(16,185,129,.35);
        border-left: 4px solid #10b981; border-radius: 10px;
        padding: 14px 18px; margin: 10px 0; color: #6ee7b7;
    }
    .result-fail {
        background: rgba(245,158,11,.1); border: 1px solid rgba(245,158,11,.3);
        border-left: 4px solid #f59e0b; border-radius: 10px;
        padding: 14px 18px; margin: 10px 0; color: #fcd34d;
    }
    .stat-pill {
        display: inline-block; background: #1e293b; border: 1px solid #334155;
        border-radius: 6px; padding: 3px 10px; font-size: 0.82rem;
        color: #cbd5e1; margin: 2px 3px;
    }
    .stat-pill b { color: #a5b4fc; }

    .streamlit-expanderHeader {
        background: #0f172a !important; border: 1px solid #1e293b !important;
        border-radius: 8px !important; color: #a5b4fc !important;
        font-weight: 600 !important; font-size: 0.83rem !important;
    }
    .streamlit-expanderContent {
        background: #0d1117 !important; border: 1px solid #1e293b !important;
        border-top: none !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] * { color: #cbd5e1; }

    .info-banner {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        border: 1px solid #4338ca; border-radius: 12px;
        padding: 16px 20px; margin: 12px 0;
        color: #c7d2fe; font-size: 0.88rem; line-height: 1.6;
    }
    .info-banner b { color: #a5b4fc; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# VERİ YÜKLEME
# ═══════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("Consumer_Shopping_Trends_2026.csv")
    try:
        kur = yf.Ticker("INRUSD=X").history(period="1d")['Close'].iloc[-1]
    except Exception:
        kur = 0.012
    for col in ['monthly_income', 'avg_online_spend', 'avg_store_spend']:
        df[col] = df[col] * kur
    df = df.dropna()
    df = df[(df['age'] > 0) & (df['monthly_income'] >= 0)]
    num_cols = df.select_dtypes(include=[np.number]).columns
    z = pd.DataFrame(np.abs(stats.zscore(df[num_cols])), columns=num_cols)
    mask = (z > 3).any(axis=1)
    return df[~mask].copy(), df[mask].copy(), kur

df, df_outliers, guncel_kur = load_data()

# ═══════════════════════════════════════════════════════════════════════
# SABİTLER & YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════════════
NUM_COLS = df.select_dtypes(include=[np.number]).columns.tolist()
CAT_COLS = df.select_dtypes(include=['object']).columns.tolist()

LABELS = {
    'age': 'Yaş', 'monthly_income': 'Aylık Gelir ($)', 'daily_internet_hours': 'Günlük İnternet (sa)',
    'smartphone_usage_years': 'Akıllı Telefon Yılı', 'social_media_hours': 'Sosyal Medya (sa)',
    'online_payment_trust_score': 'Online Ödeme Güveni', 'tech_savvy_score': 'Teknoloji Skoru',
    'monthly_online_orders': 'Aylık Online Sipariş', 'monthly_store_visits': 'Mağaza Ziyareti/Ay',
    'avg_online_spend': 'Ort. Online Harcama ($)', 'avg_store_spend': 'Ort. Mağaza Harcama ($)',
    'discount_sensitivity': 'İndirim Hassasiyeti', 'return_frequency': 'İade Sıklığı',
    'avg_delivery_days': 'Ort. Teslimat Süresi', 'delivery_fee_sensitivity': 'Kargo Ücreti Hss.',
    'free_return_importance': 'Ücretsiz İade Önemi', 'product_availability_online': 'Online Ürün Erişimi',
    'impulse_buying_score': 'Anlık Alım Skoru', 'need_touch_feel_score': 'Dokunma İhtiyacı',
    'brand_loyalty_score': 'Marka Sadakati', 'environmental_awareness': 'Çevre Bilinci',
    'time_pressure_level': 'Zaman Baskısı', 'gender': 'Cinsiyet',
    'city_tier': 'Şehir Seviyesi', 'shopping_preference': 'Alışveriş Tercihi'
}

T = "plotly_dark"
CS = px.colors.qualitative.Vivid


def lbl(c): return LABELS.get(c, c)


def normality_test(data, alpha=0.05):
    s = data.dropna()
    if len(s) < 3: return False, 1.0
    if len(s) > 5000: s = s.sample(5000, random_state=42)
    _, p = stats.shapiro(s)
    return p > alpha, p


def cohen_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    pooled = np.sqrt(((n1-1)*g1.var() + (n2-1)*g2.var()) / (n1+n2-2))
    return (g1.mean()-g2.mean()) / pooled if pooled != 0 else 0.0


def d_label(d):
    d = abs(d)
    if d < 0.2: return "ihmal edilebilir"
    if d < 0.5: return "küçük"
    if d < 0.8: return "orta"
    return "büyük"


def test_box(passed, text):
    cls = "result-pass" if passed else "result-fail"
    icon = "✅" if passed else "⚠️"
    st.markdown(f'<div class="{cls}">{icon} {text}</div>', unsafe_allow_html=True)


def pills(*items):
    """Render stat pills. Dollar signs are escaped to prevent LaTeX rendering."""
    def safe(v): return str(v).replace("$", "&#36;").replace("<", "&lt;").replace(">", "&gt;")
    html = " ".join(f'<span class="stat-pill">{k}: <b>{safe(v)}</b></span>' for k, v in items)
    st.markdown(html, unsafe_allow_html=True)


def sec(title):
    st.markdown(f'<div class="section-header"><span>{title}</span></div>', unsafe_allow_html=True)


def banner(text):
    st.markdown(f'<div class="info-banner">{text}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# YAN MENÜ
# ═══════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## 💎 Veri Madenciliği\n**Kontrol Paneli**")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Filtreler")

f_c = st.sidebar.multiselect("Cinsiyet",         df['gender'].unique(),            default=df['gender'].unique())
f_s = st.sidebar.multiselect("Şehir Seviyesi",   df['city_tier'].unique(),         default=df['city_tier'].unique())
f_t = st.sidebar.multiselect("Alışveriş Tercihi",df['shopping_preference'].unique(),default=df['shopping_preference'].unique())
f_y = st.sidebar.slider("Yaş Aralığı", int(df['age'].min()), int(df['age'].max()),
                          (int(df['age'].min()), int(df['age'].max())))

df_f = df[
    (df['gender'].isin(f_c)) & (df['city_tier'].isin(f_s)) &
    (df['shopping_preference'].isin(f_t)) &
    (df['age'] >= f_y[0]) & (df['age'] <= f_y[1])
]

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style='font-size:.82rem; color:#64748b; line-height:1.9'>
📊 Filtrelenmiş: <b style='color:#a5b4fc'>{len(df_f):,}</b> satır<br>
🗑️ Aykırı değer: <b style='color:#f87171'>{len(df_outliers):,}</b> satır silindi<br>
💱 1 INR = <b style='color:#34d399'>{guncel_kur:.5f}</b> USD
</div>
""", unsafe_allow_html=True)

if len(df_f) == 0:
    st.warning("⚠️ Filtrelere uygun veri yok — filtreleri gevşetin.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════
# HERO & KPIs
# ═══════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero-band">
    <h1>💎 Veri Madenciliği Master Paneli</h1>
    <p>Consumer Shopping Trends 2026 · {len(df_f):,} gözlem · {len(df.columns)} değişken</p>
    <span class="hero-badge">Ön İşleme</span>
    <span class="hero-badge">Görselleştirme</span>
    <span class="hero-badge">Hipotez Testleri</span>
    <span class="hero-badge">Sınıflandırma</span>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
for col_, icon_, val_, lbl_ in [
    (k1, "🧑‍🤝‍🧑", f"{df_f['age'].mean():.1f}",                        "ORTALAMA YAŞ"),
    (k2, "💰",  f"&#36;{df_f['monthly_income'].mean():,.0f}",            "ORT. AYLIK GELİR"),
    (k3, "🛒",  f"&#36;{df_f['avg_online_spend'].sum():,.0f}",           "TOPLAM ONLİNE HARC."),
    (k4, "💻",  f"{df_f['tech_savvy_score'].mean():.1f}",                "ORT. TEKNOLOJİ SKORU"),
]:
    with col_:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon_}</div>
            <div class="kpi-value">{val_}</div>
            <div class="kpi-label">{lbl_}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# SEKMELER
# ═══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📖 Veri Ön İşleme",
    "📊 Görselleştirme",
    "🧪 Parametrik Testler",
    "🎲 Ki-Kare Testi",
    "📐 Non-Parametrik Testler",
    "🎯 Otomatik Test Arayüzü",
    "📦 İnteraktif Boxplot",
    "🤖 Sınıflandırma Modelleri"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — VERİ ÖN İŞLEME
# ══════════════════════════════════════════════════════════════
with tab1:

    sec("1 · Veri Sözlüğü")
    rows = [{"Sütun": c, "Türkçe Adı": lbl(c),
             "Tip": "Kategorik 🏷️" if c in CAT_COLS else "Sayısal 🔢",
             "Benzersiz": df_f[c].nunique(), "Eksik": df_f[c].isnull().sum()}
            for c in df_f.columns]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("💻 Veri Keşif Kodu"):
        st.code("""
import pandas as pd
df = pd.read_csv("Consumer_Shopping_Trends_2026.csv")
print(df.info())
print(df.describe())
print(df.nunique())
print(df.isnull().sum())
        """, language='python')

    st.markdown("---")
    sec("2 · Tanımlayıcı İstatistikler")
    num_df = df_f.select_dtypes(include=[np.number])
    desc = num_df.describe().T.copy()
    desc['CV (%)']   = (desc['std'] / desc['mean'] * 100).round(2)
    desc['IQR']      = (desc['75%'] - desc['25%']).round(2)
    desc['Çarpıklık'] = num_df.skew().round(3)
    desc['Basıklık']  = num_df.kurtosis().round(3)
    st.dataframe(
        desc.style.format("{:.3f}").background_gradient(cmap='Blues', subset=['CV (%)']),
        use_container_width=True
    )

    with st.expander("💻 Tanımlayıcı İstatistik Kodu"):
        st.code("""
num_df = df.select_dtypes(include=['number'])
desc = num_df.describe().T
desc['CV (%)']    = (desc['std'] / desc['mean'] * 100).round(2)  # Varyasyon katsayısı
desc['IQR']       = desc['75%'] - desc['25%']                     # Çeyreklik aralığı
desc['Çarpıklık'] = num_df.skew()                                 # Skewness
desc['Basıklık']  = num_df.kurtosis()                             # Kurtosis
print(desc)
        """, language='python')

    st.markdown("---")
    sec("3 · Aykırı Değer Analizi — Z-Skoru & IQR")

    c1, c2 = st.columns(2)
    with c1:
        comb = pd.concat([df.assign(**{'Durum':'Temiz'}),
                          df_outliers.assign(**{'Durum':'Aykırı (silindi)'})])
        fig = px.scatter(comb, x='monthly_income', y='avg_online_spend', color='Durum',
                         color_discrete_map={'Temiz':'#34d399','Aykırı (silindi)':'#f87171'},
                         title="Z-Skoru Aykırı Değer Dağılımı", opacity=.55, template=T)
        fig.update_layout(legend=dict(orientation='h', y=1.08))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(f"**{len(df_outliers)} satır** Z-Skoru > 3 eşiğiyle çıkarıldı.")
        iqr_rows = []
        for col in NUM_COLS[:8]:
            Q1, Q3 = df_f[col].quantile(.25), df_f[col].quantile(.75)
            IQR = Q3 - Q1
            n = int(((df_f[col] < Q1-1.5*IQR) | (df_f[col] > Q3+1.5*IQR)).sum())
            iqr_rows.append({"Sütun": lbl(col), "Alt Sınır": f"{Q1-1.5*IQR:.1f}",
                             "Üst Sınır": f"{Q3+1.5*IQR:.1f}", "IQR Aykırı": n})
        st.dataframe(pd.DataFrame(iqr_rows), use_container_width=True, hide_index=True)

    with st.expander("💻 Z-Skoru & IQR Kodu"):
        st.code("""
from scipy import stats
import numpy as np

# Yöntem 1: Z-Skoru
num_cols = df.select_dtypes(include=[np.number]).columns
z = np.abs(stats.zscore(df[num_cols]))
df_clean = df[(z < 3).all(axis=1)]

# Yöntem 2: IQR
Q1 = df['monthly_income'].quantile(0.25)
Q3 = df['monthly_income'].quantile(0.75)
IQR = Q3 - Q1
df_iqr = df[df['monthly_income'].between(Q1 - 1.5*IQR, Q3 + 1.5*IQR)]
        """, language='python')

    st.markdown("---")
    sec("4 · Normalizasyon & Standardizasyon")

    nc = st.selectbox("Sütun seçin:", NUM_COLS, key='nc')
    sd = df_f[nc].dropna()
    mm = MinMaxScaler().fit_transform(sd.values.reshape(-1,1)).flatten()
    ss = StandardScaler().fit_transform(sd.values.reshape(-1,1)).flatten()

    for col_w, data, title, color in [
        (st.columns(3)[0], sd, f"Orijinal: {lbl(nc)}", "#60a5fa"),
        (st.columns(3)[1], mm, "Min-Max [0, 1]",         "#34d399"),
        (st.columns(3)[2], ss, "Z-Skoru Std.",            "#f472b6"),
    ]:
        pass

    ca, cb, cc = st.columns(3)
    for col_w, data, title, color in [
        (ca, sd, f"Orijinal: {lbl(nc)}", "#60a5fa"),
        (cb, mm, "Min-Max [0, 1]",       "#34d399"),
        (cc, ss, "Z-Skoru Std.",          "#f472b6"),
    ]:
        with col_w:
            fig = px.histogram(pd.Series(data), nbins=40, title=title,
                               color_discrete_sequence=[color], template=T)
            fig.update_layout(showlegend=False, margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("💻 Normalizasyon Kodu"):
        st.code("""
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Min-Max: [0, 1] aralığına taşır
df['normalized'] = MinMaxScaler().fit_transform(df[['monthly_income']])

# Z-Skoru: ortalama=0, std=1
df['standardized'] = StandardScaler().fit_transform(df[['monthly_income']])
        """, language='python')


# ══════════════════════════════════════════════════════════════
# TAB 2 — GÖRSELLEŞTİRME
# ══════════════════════════════════════════════════════════════
with tab2:

    sec("1 · Kategorik Değişken Dağılımları")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.pie(df_f, names='gender', title="Cinsiyet", hole=.45,
                     color_discrete_sequence=CS, template=T)
        fig.update_traces(textposition='outside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df_f, x='city_tier', color='city_tier', title="Şehir Seviyesi",
                           color_discrete_sequence=CS, template=T)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        fig = px.pie(df_f, names='shopping_preference', title="Alışveriş Tercihi", hole=.45,
                     color_discrete_sequence=px.colors.qualitative.Pastel, template=T)
        fig.update_traces(textposition='outside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("💻 Kategorik Dağılım Kodu"):
        st.code("""
import plotly.express as px

fig = px.pie(df, names='gender', hole=0.4)
fig.show()

fig2 = px.histogram(df, x='city_tier', color='city_tier')
fig2.show()
        """, language='python')

    st.markdown("---")
    sec("2 · Boxplot & Violin Plot")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(df_f, x='city_tier', y='avg_store_spend', color='city_tier',
                     title="Şehir Seviyesi × Mağaza Harcaması",
                     color_discrete_sequence=CS, template=T)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.violin(df_f, x='shopping_preference', y='tech_savvy_score',
                        color='shopping_preference', box=True, points='outliers',
                        title="Alışveriş Tercihi × Teknoloji Skoru",
                        color_discrete_sequence=CS, template=T)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("💻 Boxplot & Violin Kodu"):
        st.code("""
# Boxplot: Q1 / Medyan / Q3 + uç değerler
fig = px.box(df, x='city_tier', y='avg_store_spend', color='city_tier')
fig.show()

# Violin: Boxplot + KDE yoğunluk dağılımı
fig = px.violin(df, x='shopping_preference', y='tech_savvy_score',
                color='shopping_preference', box=True, points='outliers')
fig.show()
        """, language='python')

    st.markdown("---")
    sec("3 · Scatter Plot (Saçılım + OLS Trend)")

    c1, c2 = st.columns(2)
    with c1: xs = st.selectbox("X:", NUM_COLS, index=NUM_COLS.index('monthly_income'), key='sx')
    with c2: ys = st.selectbox("Y:", NUM_COLS, index=NUM_COLS.index('avg_online_spend'), key='sy')

    fig = px.scatter(df_f, x=xs, y=ys, color='shopping_preference',
                     trendline='ols', opacity=.55,
                     title=f"{lbl(xs)} vs {lbl(ys)}",
                     color_discrete_sequence=CS, template=T)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("💻 Scatter Plot Kodu"):
        st.code("""
fig = px.scatter(df, x='monthly_income', y='avg_online_spend',
                 color='shopping_preference', trendline='ols', opacity=0.55)
fig.show()
        """, language='python')

    st.markdown("---")
    sec("4 · Spearman Korelasyon Matrisi")

    sel_c = st.multiselect("Sütun seçin:", NUM_COLS,
                            default=['age','monthly_income','avg_online_spend',
                                     'avg_store_spend','tech_savvy_score','social_media_hours'])
    if len(sel_c) >= 2:
        corr = df_f[sel_c].corr(method='spearman')
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        aspect="auto", title="Spearman Korelasyon Matrisi", template=T)
        st.plotly_chart(fig, use_container_width=True)

        pairs = [(corr.columns[i], corr.columns[j], corr.iloc[i,j])
                 for i in range(len(corr)) for j in range(i+1, len(corr))]
        top5 = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:5]
        st.markdown("**🔗 En Güçlü 5 Korelasyon:**")
        for v1, v2, r in top5:
            st.markdown(f"- `{lbl(v1)}` × `{lbl(v2)}` → **r = {r:.3f}** "
                        f"({'↗ Pozitif' if r > 0 else '↘ Negatif'})")

    with st.expander("💻 Spearman Korelasyon Kodu"):
        st.code("""
corr = df[cols].corr(method='spearman')
fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r')
fig.show()
        """, language='python')


# ══════════════════════════════════════════════════════════════
# TAB 3 — PARAMETRİK TESTLER
# ══════════════════════════════════════════════════════════════
with tab3:
    banner("Parametrik testler verilerin <b>normal dağıldığını</b> varsayar. "
           "Her test öncesi <b>Shapiro-Wilk</b> normallik testi gösterilmektedir.")

    # 1. Tek Örneklem T-Testi
    sec("1 · Tek Örneklem T-Testi")
    os_col = st.selectbox("Test sütunu:", NUM_COLS, index=0, key='os_col')
    pop_mu = st.number_input("H₀ değeri (μ₀):",
                              value=float(df_f[os_col].mean().round(1)), key='pop_mu')
    d_os = df_f[os_col].dropna()
    is_n, sp = normality_test(d_os)
    t_s, p_s = stats.ttest_1samp(d_os, pop_mu)

    c1, c2 = st.columns([2,1])
    with c1:
        fig = px.histogram(d_os, nbins=40, title=f"Dağılım: {lbl(os_col)}",
                           color_discrete_sequence=['#60a5fa'], template=T)
        fig.add_vline(x=pop_mu,      line_dash="dash", line_color="#f87171",
                      annotation_text=f"H₀={pop_mu:.1f}", annotation_font_color="#f87171")
        fig.add_vline(x=d_os.mean(), line_dash="dash", line_color="#34d399",
                      annotation_text=f"x̄={d_os.mean():.1f}", annotation_font_color="#34d399")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        pills(("t", f"{t_s:.4f}"), ("p", f"{p_s:.6f}"),
              ("Shapiro p", f"{sp:.4f}"), ("n", str(len(d_os))),
              ("Normallik", "✅ Normal" if is_n else "❌ Normal Değil"))
        test_box(p_s < .05,
                 f"H₀ Reddedildi (p={p_s:.4f}): Ortalama {pop_mu:.1f} değerinden anlamlı şekilde farklı."
                 if p_s < .05 else
                 f"H₀ Reddedilemez (p={p_s:.4f}): Anlamlı fark yok.")

    with st.expander("💻 Tek Örneklem T-Testi Kodu"):
        st.code("""
from scipy import stats

# H₀: μ = belirli bir değer
t, p = stats.ttest_1samp(df['age'], popmean=40)
if p < 0.05:
    print("H₀ reddedildi — ortalama 40'tan farklı.")
        """, language='python')

    st.markdown("---")
    sec("2 · Bağımsız İki Örneklem T-Testi")

    erkek = df_f[df_f['gender'] == 'Male']['avg_online_spend']
    kadin = df_f[df_f['gender'] == 'Female']['avg_online_spend']

    if len(erkek) > 1 and len(kadin) > 1:
        t_i, p_i = stats.ttest_ind(erkek, kadin)
        d_v = cohen_d(erkek, kadin)

        c1, c2 = st.columns([2,1])
        with c1:
            fig = go.Figure([
                go.Histogram(x=erkek, name='Erkek', opacity=.7,
                             marker_color='#60a5fa', bingroup=1),
                go.Histogram(x=kadin, name='Kadın', opacity=.7,
                             marker_color='#f472b6', bingroup=1),
            ])
            fig.update_layout(barmode='overlay', template=T,
                               title="Cinsiyete Göre Online Harcama")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            pills(("t", f"{t_i:.4f}"), ("p", f"{p_i:.6f}"),
                  ("Cohen's d", f"{d_v:.3f}"), ("Etki", d_label(d_v)),
                  ("Erkek Ort.", f"${erkek.mean():,.0f}"),
                  ("Kadın Ort.", f"${kadin.mean():,.0f}"))
            test_box(p_i < .05,
                     f"H₀ Reddedildi (p={p_i:.4f}): Cinsiyet farkı anlamlı, etki = {d_label(d_v)}."
                     if p_i < .05 else
                     f"H₀ Reddedilemez (p={p_i:.4f}): Anlamlı fark yok.")

    with st.expander("💻 Bağımsız T-Testi Kodu"):
        st.code("""
from scipy import stats
import numpy as np

erkek = df[df['gender']=='Male']['avg_online_spend']
kadin = df[df['gender']=='Female']['avg_online_spend']
t, p  = stats.ttest_ind(erkek, kadin)

# Cohen's d: etki büyüklüğü
def cohen_d(g1, g2):
    pooled = np.sqrt(((len(g1)-1)*g1.var() + (len(g2)-1)*g2.var()) / (len(g1)+len(g2)-2))
    return (g1.mean()-g2.mean()) / pooled

print(f"t={t:.4f}, p={p:.4f}, d={cohen_d(erkek,kadin):.3f}")
        """, language='python')

    st.markdown("---")
    sec("3 · Eşleştirilmiş T-Testi")

    online_s = df_f['avg_online_spend']
    store_s  = df_f['avg_store_spend']
    t_p, p_p = stats.ttest_rel(online_s, store_s)
    diff_p   = online_s - store_s

    c1, c2 = st.columns([2,1])
    with c1:
        fig = go.Figure([
            go.Box(y=online_s, name='Online', marker_color='#34d399'),
            go.Box(y=store_s,  name='Mağaza', marker_color='#f59e0b'),
        ])
        fig.update_layout(template=T, title="Online vs Mağaza Harcama")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        pills(("t", f"{t_p:.4f}"), ("p", f"{p_p:.6f}"),
              ("Fark Ort.", f"${diff_p.mean():,.0f}"),
              ("Online Med.", f"${online_s.median():,.0f}"),
              ("Mağaza Med.", f"${store_s.median():,.0f}"))
        test_box(p_p < .05,
                 f"H₀ Reddedildi (p={p_p:.4f}): Online ve mağaza harcaması anlamlı şekilde farklı."
                 if p_p < .05 else
                 f"H₀ Reddedilemez (p={p_p:.4f}): Anlamlı fark yok.")

    with st.expander("💻 Eşleştirilmiş T-Testi Kodu"):
        st.code("""
from scipy import stats

t, p = stats.ttest_rel(df['avg_online_spend'], df['avg_store_spend'])
diff = df['avg_online_spend'] - df['avg_store_spend']
print(f"t={t:.4f}, p={p:.4f}, Fark Ort.={diff.mean():.2f}")
        """, language='python')

    st.markdown("---")
    sec("4 · Tek Yönlü ANOVA")

    t1 = df_f[df_f['city_tier']=='Tier 1']['avg_store_spend']
    t2 = df_f[df_f['city_tier']=='Tier 2']['avg_store_spend']
    t3 = df_f[df_f['city_tier']=='Tier 3']['avg_store_spend']

    if all(len(g) > 1 for g in [t1, t2, t3]):
        f_s, p_a = stats.f_oneway(t1, t2, t3)

        c1, c2 = st.columns([2,1])
        with c1:
            adf = pd.concat([
                t1.rename('v').to_frame().assign(Grup='Tier 1'),
                t2.rename('v').to_frame().assign(Grup='Tier 2'),
                t3.rename('v').to_frame().assign(Grup='Tier 3'),
            ])
            fig = px.box(adf, x='Grup', y='v', color='Grup',
                         title="ANOVA: Şehir Seviyesi × Mağaza Harcaması",
                         color_discrete_sequence=CS, template=T)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            pills(("F", f"{f_s:.4f}"), ("p", f"{p_a:.6f}"),
                  ("Tier 1 Ort.", f"${t1.mean():,.0f}"),
                  ("Tier 2 Ort.", f"${t2.mean():,.0f}"),
                  ("Tier 3 Ort.", f"${t3.mean():,.0f}"))
            test_box(p_a < .05,
                     f"H₀ Reddedildi (p={p_a:.4f}): Şehir seviyesi mağaza harcamasını etkiliyor."
                     if p_a < .05 else
                     f"H₀ Reddedilemez (p={p_a:.4f}): Gruplar benzer.")

    with st.expander("💻 ANOVA Kodu"):
        st.code("""
from scipy import stats

t1 = df[df['city_tier']=='Tier 1']['avg_store_spend']
t2 = df[df['city_tier']=='Tier 2']['avg_store_spend']
t3 = df[df['city_tier']=='Tier 3']['avg_store_spend']

f, p = stats.f_oneway(t1, t2, t3)
if p < 0.05:
    print("En az bir grubun ortalaması anlamlı şekilde farklı.")
        """, language='python')


# ══════════════════════════════════════════════════════════════
# TAB 4 — Kİ-KARE
# ══════════════════════════════════════════════════════════════
with tab4:
    banner("<b>Ki-Kare Bağımsızlık Testi:</b> İki kategorik değişkenin birbirinden bağımsız "
           "olup olmadığını test eder. H₀: Değişkenler bağımsızdır. "
           "<b>Cramér's V</b> etki büyüklüğünü ölçer (0 = ilişki yok, 1 = tam ilişki).")

    def chi_panel(col_w, var1, var2, cmap, title):
        with col_w:
            sec(title)
            ct = pd.crosstab(df_f[var1], df_f[var2])
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            cv = np.sqrt(chi2 / (len(df_f) * (min(ct.shape)-1)))
            fig = px.imshow(ct, text_auto=True, color_continuous_scale=cmap,
                            title=f"Çapraz Tablo: {lbl(var1)} × {lbl(var2)}", template=T)
            st.plotly_chart(fig, use_container_width=True)
            pills(("χ²", f"{chi2:.4f}"), ("p", f"{p:.6f}"),
                  ("df", str(dof)), ("Cramér's V", f"{cv:.3f}"))
            test_box(p < .05,
                     f"Bağımlılık var (p={p:.4f}): {lbl(var1)} ile {lbl(var2)} arasında anlamlı ilişki."
                     if p < .05 else
                     f"Bağımsız (p={p:.4f}): İki değişken arasında anlamlı ilişki yok.")

    c1, c2 = st.columns(2)
    chi_panel(c1, 'gender',    'shopping_preference', 'Blues',   "Test 1 · Cinsiyet × Alışveriş Tercihi")
    chi_panel(c2, 'city_tier', 'shopping_preference', 'Oranges', "Test 2 · Şehir Seviyesi × Alışveriş Tercihi")

    st.markdown("---")
    sec("Test 3 · Kullanıcı Seçimli Ki-Kare")
    cc1, cc2 = st.columns(2)
    with cc1: cv1 = st.selectbox("1. Değişken:", CAT_COLS, index=0, key='cv1')
    with cc2: cv2 = st.selectbox("2. Değişken:", CAT_COLS, index=2, key='cv2')
    if cv1 != cv2:
        chi_panel(st.container(), cv1, cv2, 'Purples', f"{lbl(cv1)} × {lbl(cv2)}")

    with st.expander("💻 Ki-Kare Testi Kodu"):
        st.code("""
import pandas as pd
from scipy import stats
import numpy as np

ct = pd.crosstab(df['gender'], df['shopping_preference'])
chi2, p, dof, expected = stats.chi2_contingency(ct)

# Cramér's V (etki büyüklüğü)
cramers_v = np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1)))
print(f"chi2={chi2:.4f}, p={p:.4f}, Cramér's V={cramers_v:.3f}")
        """, language='python')


# ══════════════════════════════════════════════════════════════
# TAB 5 — NON-PARAMETRİK TESTLER
# ══════════════════════════════════════════════════════════════
with tab5:
    banner("Veriler <b>normal dağılmadığında</b> ya da ölçek <b>sıralı (ordinal)</b> olduğunda "
           "parametrik olmayan testler tercih edilir. Bu testler daha az varsayım gerektirir ve "
           "<b>aykırı değerlere karşı dayanıklıdır</b>.")

    sec("1 · Wilcoxon İşaretli Sıra Testi  —  Eşleştirilmiş T-Testi'nin alternatifi")
    w_s, w_p = stats.wilcoxon(df_f['avg_online_spend'].values,
                               df_f['avg_store_spend'].values)
    diff_w = df_f['avg_online_spend'].values - df_f['avg_store_spend'].values

    c1, c2 = st.columns([2,1])
    with c1:
        fig = px.histogram(pd.Series(diff_w), nbins=50,
                           title="Fark Dağılımı: Online − Mağaza",
                           color_discrete_sequence=['#a78bfa'], template=T)
        fig.add_vline(x=0, line_dash="dash", line_color="#f87171",
                      annotation_text="Fark=0", annotation_font_color="#f87171")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        pills(("W", f"{w_s:,.0f}"), ("p", f"{w_p:.6f}"),
              ("Medyan Fark", f"${np.median(diff_w):,.0f}"))
        test_box(w_p < .05,
                 f"H₀ Reddedildi (p={w_p:.4f}): İki ölçüm anlamlı şekilde farklı."
                 if w_p < .05 else
                 f"H₀ Reddedilemez (p={w_p:.4f}): Anlamlı fark yok.")

    with st.expander("💻 Wilcoxon Testi Kodu"):
        st.code("""
from scipy import stats

w, p = stats.wilcoxon(df['avg_online_spend'], df['avg_store_spend'])
print(f"W={w:.0f}, p={p:.6f}")
        """, language='python')

    st.markdown("---")
    sec("2 · Mann-Whitney U Testi  —  Bağımsız T-Testi'nin alternatifi")

    e_tech = df_f[df_f['gender']=='Male']['tech_savvy_score']
    k_tech = df_f[df_f['gender']=='Female']['tech_savvy_score']

    if len(e_tech) > 1 and len(k_tech) > 1:
        u_s, u_p = stats.mannwhitneyu(e_tech, k_tech, alternative='two-sided')

        c1, c2 = st.columns([2,1])
        with c1:
            fig = go.Figure([
                go.Violin(y=e_tech, name='Erkek', box_visible=True, meanline_visible=True,
                          fillcolor='#60a5fa', opacity=.65),
                go.Violin(y=k_tech, name='Kadın', box_visible=True, meanline_visible=True,
                          fillcolor='#f472b6', opacity=.65),
            ])
            fig.update_layout(template=T, title="Cinsiyete Göre Teknoloji Skoru")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            pills(("U", f"{u_s:,.0f}"), ("p", f"{u_p:.6f}"),
                  ("Erkek Med.", f"{e_tech.median():.1f}"),
                  ("Kadın Med.", f"{k_tech.median():.1f}"))
            test_box(u_p < .05,
                     f"H₀ Reddedildi (p={u_p:.4f}): Cinsiyetler arası teknoloji skoru farklı."
                     if u_p < .05 else
                     f"H₀ Reddedilemez (p={u_p:.4f}): Dağılımlar benzer.")

    with st.expander("💻 Mann-Whitney U Kodu"):
        st.code("""
from scipy import stats

erkek = df[df['gender']=='Male']['tech_savvy_score']
kadin = df[df['gender']=='Female']['tech_savvy_score']

u, p = stats.mannwhitneyu(erkek, kadin, alternative='two-sided')
print(f"U={u:.0f}, p={p:.6f}")
        """, language='python')

    st.markdown("---")
    sec("3 · Kruskal-Wallis Testi  —  ANOVA'nın alternatifi")

    kw_grps = [g['avg_online_spend'].values for _, g in df_f.groupby('city_tier')]
    if all(len(g) > 1 for g in kw_grps):
        h_s, kw_p = stats.kruskal(*kw_grps)

        c1, c2 = st.columns([2,1])
        with c1:
            fig = px.box(df_f, x='city_tier', y='avg_online_spend', color='city_tier',
                         title="Kruskal-Wallis: Şehir Seviyesi × Online Harcama",
                         color_discrete_sequence=CS, template=T)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            pills(("H", f"{h_s:.4f}"), ("p", f"{kw_p:.6f}"))
            for name, g in df_f.groupby('city_tier'):
                pills((f"{name} Med.", f"${g['avg_online_spend'].median():,.0f}"))
            test_box(kw_p < .05,
                     f"H₀ Reddedildi (p={kw_p:.4f}): Gruplar arası anlamlı fark var."
                     if kw_p < .05 else
                     f"H₀ Reddedilemez (p={kw_p:.4f}): Gruplar benzer.")

    with st.expander("💻 Kruskal-Wallis Kodu"):
        st.code("""
from scipy import stats

grps = [g['avg_online_spend'].values for _, g in df.groupby('city_tier')]
h, p = stats.kruskal(*grps)
print(f"H={h:.4f}, p={p:.6f}")
        """, language='python')


# ══════════════════════════════════════════════════════════════
# TAB 6 — OTOMATİK TEST ARAYÜZÜ
# ══════════════════════════════════════════════════════════════
with tab6:
    banner("İki değişken seçin. Sistem değişken tipini ve <b>Shapiro-Wilk normallik testini</b> "
           "otomatik çalıştırarak uygun istatistiksel testi belirleyip uygular ve sonucu yorumlar.<br><br>"
           "<b>Karar mantığı:</b> "
           "Sayısal × Sayısal → Pearson / Spearman &nbsp;|&nbsp; "
           "Sayısal × Kategorik (2 grup) → T-Test / Mann-Whitney &nbsp;|&nbsp; "
           "Sayısal × Kategorik (3+ grup) → ANOVA / Kruskal-Wallis &nbsp;|&nbsp; "
           "Kategorik × Kategorik → Ki-Kare")

    all_cols = NUM_COLS + CAT_COLS
    cs1, cs2 = st.columns(2)
    with cs1: v1 = st.selectbox("🔵 Birinci değişken:", all_cols, index=0, key='auto1')
    with cs2: v2 = st.selectbox("🔴 İkinci değişken:", all_cols, index=len(NUM_COLS), key='auto2')

    if v1 == v2:
        st.warning("Lütfen iki farklı değişken seçin.")
    else:
        v1n = v1 in NUM_COLS
        v2n = v2 in NUM_COLS
        st.markdown("---")

        # ── Sayısal × Sayısal ──
        if v1n and v2n:
            d1 = df_f[v1].dropna(); d2 = df_f[v2].dropna()
            n = min(len(d1), len(d2)); d1, d2 = d1.iloc[:n], d2.iloc[:n]
            n1, sp1 = normality_test(d1); n2, sp2 = normality_test(d2)
            if n1 and n2:
                method = "Pearson"; r, rp = stats.pearsonr(d1, d2)
            else:
                method = "Spearman"; r, rp = stats.spearmanr(d1, d2)

            sec(f"Korelasyon Analizi: {lbl(v1)} × {lbl(v2)}")
            ca, cb = st.columns([2,1])
            with ca:
                fig = px.scatter(df_f, x=v1, y=v2, trendline='ols', opacity=.5,
                                 color='shopping_preference',
                                 title=f"{method}: {lbl(v1)} vs {lbl(v2)}",
                                 color_discrete_sequence=CS, template=T)
                st.plotly_chart(fig, use_container_width=True)
            with cb:
                st.markdown(f"**Seçilen Test:** {method} Korelasyonu")
                pills((f"Normallik {v1[:10]}", "✅" if n1 else "❌"),
                      (f"Normallik {v2[:10]}", "✅" if n2 else "❌"),
                      ("r", f"{r:.4f}"), ("p", f"{rp:.6f}"))
                r_abs = abs(r)
                guc = "güçlü" if r_abs>=.7 else "orta" if r_abs>=.4 else "zayıf" if r_abs>=.2 else "çok zayıf"
                yon = "pozitif ↗" if r > 0 else "negatif ↘"
                test_box(rp < .05,
                         f"Anlamlı {guc} {yon} korelasyon (r={r:.3f}, p={rp:.4f})."
                         if rp < .05 else
                         f"Anlamlı korelasyon bulunamadı (p={rp:.4f}).")

            with st.expander("💻 Otomatik Korelasyon Kodu"):
                st.code(f"""
from scipy import stats

d1, d2 = df['{v1}'], df['{v2}']
_, p1  = stats.shapiro(d1.sample(min(5000, len(d1))))
_, p2  = stats.shapiro(d2.sample(min(5000, len(d2))))

if p1 > 0.05 and p2 > 0.05:
    r, p = stats.pearsonr(d1, d2)    # Her ikisi de normal → Pearson
else:
    r, p = stats.spearmanr(d1, d2)   # En az biri normal değil → Spearman
                """, language='python')

        # ── Sayısal × Kategorik ──
        elif v1n != v2n:
            num_v = v1 if v1n else v2
            cat_v = v2 if v1n else v1
            grps  = {k: pd.Series(v).dropna() for k, v in
                     df_f.groupby(cat_v)[num_v].apply(list).items()}
            n_grp = len(grps)
            all_n = all(normality_test(g)[0] for g in grps.values() if len(g) > 2)

            if n_grp == 2:
                g1, g2 = list(grps.values())
                if all_n:
                    tname = "Bağımsız T-Testi"; st_v, p_v = stats.ttest_ind(g1, g2)
                else:
                    tname = "Mann-Whitney U";   st_v, p_v = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            else:
                arrs = [g.values for g in grps.values() if len(g) > 1]
                if all_n:
                    tname = "ANOVA";          st_v, p_v = stats.f_oneway(*arrs)
                else:
                    tname = "Kruskal-Wallis"; st_v, p_v = stats.kruskal(*arrs)

            sec(f"{tname}: {lbl(num_v)} ~ {lbl(cat_v)}")
            ca, cb = st.columns([2,1])
            with ca:
                fig = px.box(df_f, x=cat_v, y=num_v, color=cat_v,
                             title=f"{tname}: {lbl(cat_v)} → {lbl(num_v)}",
                             color_discrete_sequence=CS, template=T)
                st.plotly_chart(fig, use_container_width=True)
            with cb:
                st.markdown(f"**Seçilen Test:** {tname}")
                for gn, g in grps.items():
                    is_n, sp = normality_test(g)
                    pills((f"{gn} Normallik", "✅" if is_n else "❌"),
                          (f"{gn} Med.", f"{g.median():.2f}"))
                pills(("İstatistik", f"{st_v:.4f}"), ("p", f"{p_v:.6f}"))
                test_box(p_v < .05,
                         f"H₀ Reddedildi (p={p_v:.4f}): Gruplar arası anlamlı fark var."
                         if p_v < .05 else
                         f"H₀ Reddedilemez (p={p_v:.4f}): Gruplar benzer.")

            with st.expander("💻 Otomatik Test Seçim Kodu"):
                st.code(f"""
from scipy import stats
import pandas as pd

groups = df.groupby('{cat_v}')['{num_v}'].apply(list).to_dict()
all_n  = all(stats.shapiro(pd.Series(v))[1] > 0.05 for v in groups.values())

if len(groups) == 2:
    g1, g2 = [pd.Series(v) for v in groups.values()]
    stat, p = stats.ttest_ind(g1, g2) if all_n else stats.mannwhitneyu(g1, g2, alternative='two-sided')
else:
    arrs = [pd.Series(v).values for v in groups.values()]
    stat, p = stats.f_oneway(*arrs) if all_n else stats.kruskal(*arrs)
                """, language='python')

        # ── Kategorik × Kategorik ──
        else:
            sec(f"Ki-Kare: {lbl(v1)} × {lbl(v2)}")
            ct = pd.crosstab(df_f[v1], df_f[v2])
            chi2, p_chi, dof_chi, _ = stats.chi2_contingency(ct)
            cv = np.sqrt(chi2 / (len(df_f) * (min(ct.shape)-1)))
            ca, cb = st.columns([2,1])
            with ca:
                fig = px.imshow(ct, text_auto=True, color_continuous_scale='Viridis',
                                title=f"Çapraz Tablo: {lbl(v1)} × {lbl(v2)}", template=T)
                st.plotly_chart(fig, use_container_width=True)
            with cb:
                pills(("χ²", f"{chi2:.4f}"), ("p", f"{p_chi:.6f}"),
                      ("df", str(dof_chi)), ("Cramér's V", f"{cv:.3f}"))
                test_box(p_chi < .05,
                         f"Bağımlılık var (p={p_chi:.4f}): İki değişken anlamlı şekilde ilişkili."
                         if p_chi < .05 else
                         f"Bağımsız (p={p_chi:.4f}): Anlamlı ilişki yok.")

            with st.expander("💻 Ki-Kare Kodu"):
                st.code(f"""
ct = pd.crosstab(df['{v1}'], df['{v2}'])
chi2, p, dof, _ = stats.chi2_contingency(ct)
                """, language='python')


# ══════════════════════════════════════════════════════════════
# TAB 7 — İNTERAKTİF BOXPLOT
# ══════════════════════════════════════════════════════════════
with tab7:
    sec("İnteraktif Boxplot / Violin / Strip Oluşturucu")

    c1, c2, c3 = st.columns(3)
    with c1: bp_y = st.selectbox("Y — Sayısal:", NUM_COLS,
                                  index=NUM_COLS.index('avg_online_spend'), key='bpy')
    with c2: bp_x = st.selectbox("X — Gruplama:", ['—'] + CAT_COLS + NUM_COLS,
                                  index=1, key='bpx')
    with c3: bp_c = st.selectbox("Renk:", ['—'] + CAT_COLS, index=0, key='bpc')

    bp_type = st.radio("Grafik türü:", ['Boxplot','Violin','Strip','Hepsi'],
                        horizontal=True, key='bpt')

    xv = None if bp_x == '—' else bp_x
    cv_val = None if bp_c == '—' else bp_c
    pargs = dict(x=xv, y=bp_y, color=cv_val, color_discrete_sequence=CS, template=T)

    if bp_type in ('Boxplot', 'Hepsi'):
        st.plotly_chart(
            px.box(df_f, **pargs,
                   title=f"Boxplot: {lbl(bp_y)}" + (f" × {lbl(bp_x)}" if xv else "")),
            use_container_width=True)

    if bp_type in ('Violin', 'Hepsi'):
        st.plotly_chart(
            px.violin(df_f, **pargs, box=True,
                      title=f"Violin: {lbl(bp_y)}" + (f" × {lbl(bp_x)}" if xv else "")),
            use_container_width=True)

    if bp_type in ('Strip', 'Hepsi'):
        st.plotly_chart(
            px.strip(df_f, **pargs,
                     title=f"Strip: {lbl(bp_y)}" + (f" × {lbl(bp_x)}" if xv else "")),
            use_container_width=True)

    st.markdown("---")
    sec("Özet İstatistikler")
    if xv and xv in CAT_COLS:
        summ = df_f.groupby(xv)[bp_y].agg(N='count', Ort='mean', Medyan='median',
                                            Std='std', Min='min', Max='max').round(2)
        st.dataframe(summ, use_container_width=True)
    else:
        s = df_f[bp_y]
        pills(("N", str(len(s))), ("Ort.", f"{s.mean():.2f}"), ("Med.", f"{s.median():.2f}"),
              ("Std.", f"{s.std():.2f}"), ("Min", f"{s.min():.2f}"), ("Max", f"{s.max():.2f}"))

    with st.expander("💻 Boxplot Kodu"):
        st.code(f"""
import plotly.express as px

# Boxplot
fig = px.box(df, x={repr(xv)}, y='{bp_y}', color={repr(cv_val)})
fig.show()

# Violin
fig = px.violin(df, x={repr(xv)}, y='{bp_y}', color={repr(cv_val)}, box=True)
fig.show()
        """, language='python')


# ══════════════════════════════════════════════════════════════
# TAB 8 — SINIFLANDIRMA MODELLERİ
# ══════════════════════════════════════════════════════════════
with tab8:
    banner("Hedef değişken: <b>shopping_preference</b> (Online / Store / Hybrid). "
           "Karşılaştırılan modeller: <b>Karar Ağacı</b> (Entropy & Gini), <b>KNN</b>, "
           "<b>Naive Bayes</b>, <b>Lojistik Regresyon</b>. "
           "Metrikler: Accuracy · Precision · Recall · F1 · AUC · Confusion Matrix · Cross-Validation.")

    sec("1 · Veri Hazırlığı")
    feat_cols = [c for c in NUM_COLS if c in df_f.columns]
    X_raw = df_f[feat_cols].copy()
    y_raw = df_f['shopping_preference'].copy()
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    cls_names = le.classes_.tolist()
    X_sc = pd.DataFrame(MinMaxScaler().fit_transform(X_raw), columns=feat_cols, index=X_raw.index)

    c1, c2 = st.columns(2)
    with c1: test_sz = st.slider("Test oranı:", .1, .4, .3, .05, key='tsz')
    with c2: rs_val  = int(st.number_input("Random state:", value=42, step=1, key='rsv'))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y_enc, test_size=test_sz, random_state=rs_val, stratify=y_enc)

    cd1, cd2 = st.columns(2)
    with cd1:
        fig = px.histogram(y_raw, x=y_raw.values, color=y_raw.values,
                           title="Hedef Sınıf Dağılımı",
                           color_discrete_sequence=CS, template=T)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with cd2:
        pills(("Eğitim", str(len(X_tr))), ("Test", str(len(X_te))),
              ("Özellik", str(len(feat_cols))), ("Sınıf", str(len(cls_names))))
        st.dataframe(pd.DataFrame({"Özellik": [lbl(c) for c in feat_cols],
                                   "Ort (norm.)": X_sc.mean().round(4).values}).head(10),
                     use_container_width=True, hide_index=True, height=230)

    with st.expander("💻 Veri Hazırlığı Kodu"):
        st.code("""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

X = df.select_dtypes(include=['number'])
y = LabelEncoder().fit_transform(df['shopping_preference'])
X_scaled = MinMaxScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
        """, language='python')

    st.markdown("---")
    sec("2 · Model Eğitimi & Karşılaştırma")

    # ─── FIX: parametreler underscore'suz → Streamlit bunları cache key'e dahil eder ───
    @st.cache_data
    def train_models(Xtr, Xte, ytr, yte, rs):
        mdls = {
            'Karar Ağacı (Entropy)': DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=rs),
            'Karar Ağacı (Gini)':    DecisionTreeClassifier(criterion='gini',    max_depth=5, random_state=rs),
            'KNN (k=5)':              KNeighborsClassifier(n_neighbors=5),
            'KNN (k=3)':              KNeighborsClassifier(n_neighbors=3),
            'Naive Bayes':            GaussianNB(),
            'Lojistik Regresyon':     LogisticRegression(max_iter=1000, random_state=rs),
        }
        out = {}
        X_all = np.vstack([Xtr, Xte])
        y_all = np.concatenate([ytr, yte])
        for name, mdl in mdls.items():
            mdl.fit(Xtr, ytr)
            yp   = mdl.predict(Xte)
            yprb = mdl.predict_proba(Xte) if hasattr(mdl, 'predict_proba') else None
            cv   = cross_val_score(mdl, X_all, y_all, cv=5, scoring='accuracy')
            out[name] = {
                'model': mdl, 'y_pred': yp, 'y_proba': yprb,
                'y_te': yte,                      # test etiketlerini sonuçla birlikte sakla
                'accuracy':  accuracy_score(yte, yp),
                'precision': precision_score(yte, yp, average='weighted', zero_division=0),
                'recall':    recall_score(   yte, yp, average='weighted', zero_division=0),
                'f1':        f1_score(       yte, yp, average='weighted', zero_division=0),
                'cm':        confusion_matrix(yte, yp),
                'cv_mean': cv.mean(), 'cv_std': cv.std(),
            }
        return out

    results = train_models(X_tr.values, X_te.values, y_tr, y_te, rs_val)

    comp_df = pd.DataFrame([{
        'Model': n, 'Accuracy': r['accuracy'], 'Precision': r['precision'],
        'Recall': r['recall'], 'F1-Score': r['f1'],
        'CV Ort. ± Std': f"{r['cv_mean']:.4f} ± {r['cv_std']:.4f}"
    } for n, r in results.items()])

    best_name = comp_df.loc[comp_df['F1-Score'].idxmax(), 'Model']
    st.dataframe(
        comp_df.style
               .format({'Accuracy':'{:.4f}','Precision':'{:.4f}','Recall':'{:.4f}','F1-Score':'{:.4f}'})
               .highlight_max(subset=['Accuracy','Precision','Recall','F1-Score'],
                              color='rgba(99,102,241,.3)'),
        use_container_width=True, hide_index=True
    )
    st.success(f"🏆 En iyi F1 skoru: **{best_name}** — {results[best_name]['f1']:.4f}")

    fig_comp = go.Figure()
    bar_colors = ['#60a5fa','#34d399','#f59e0b','#f472b6']
    for i, metric in enumerate(['accuracy','precision','recall','f1']):
        fig_comp.add_trace(go.Bar(name=metric.capitalize(),
                                   x=list(results.keys()),
                                   y=[r[metric] for r in results.values()],
                                   marker_color=bar_colors[i], opacity=.85))
    fig_comp.update_layout(barmode='group', template=T,
                            title="Model Performans Karşılaştırması",
                            yaxis_title="Skor", xaxis_tickangle=-15)
    st.plotly_chart(fig_comp, use_container_width=True)

    with st.expander("💻 Model Eğitimi Kodu"):
        st.code("""
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

models = {
    'Karar Ağacı': DecisionTreeClassifier(criterion='entropy', max_depth=5),
    'KNN':          KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes':  GaussianNB(),
    'Loj. Regres.': LogisticRegression(max_iter=1000),
}
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    print(f"{name}: F1={f1_score(y_test, y_pred, average='weighted'):.4f}")
        """, language='python')

    st.markdown("---")
    sec("3 · Confusion Matrix")

    sel_cm = st.selectbox("Model:", list(results.keys()), key='sel_cm')
    r_cm   = results[sel_cm]
    # y_te olarak cache'den gelen etiketleri kullan — slider değişikliğinde tutarlılık sağlar
    y_te_cm = r_cm['y_te']

    c1, c2 = st.columns(2)
    with c1:
        cm_df = pd.DataFrame(r_cm['cm'], index=cls_names, columns=cls_names)
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale='Purples',
                        title=f"Confusion Matrix — {sel_cm}",
                        labels={'x':'Tahmin','y':'Gerçek'}, template=T)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        rep = classification_report(y_te_cm, r_cm['y_pred'],
                                    target_names=cls_names, output_dict=True, zero_division=0)
        rep_df = (pd.DataFrame(rep).T
                    .loc[cls_names, ['precision','recall','f1-score','support']]
                    .rename(columns={'precision':'Precision','recall':'Recall',
                                     'f1-score':'F1','support':'N'}))
        st.dataframe(
            rep_df.style.format({'Precision':'{:.3f}','Recall':'{:.3f}','F1':'{:.3f}','N':'{:.0f}'}),
            use_container_width=True
        )

    with st.expander("💻 Confusion Matrix Kodu"):
        st.code("""
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px
import pandas as pd

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
fig   = px.imshow(cm_df, text_auto=True, color_continuous_scale='Purples')
fig.show()

print(classification_report(y_test, y_pred, target_names=class_names))
        """, language='python')

    st.markdown("---")
    sec("4 · ROC Eğrisi & AUC")

    sel_roc = st.selectbox("Model:", list(results.keys()), key='sel_roc')
    r_roc   = results[sel_roc]
    y_te_roc = r_roc['y_te']    # cache ile tutarlı etiketler

    if r_roc['y_proba'] is not None:
        fig_roc = go.Figure()
        roc_clrs = ['#f87171','#60a5fa','#34d399','#f59e0b','#a78bfa']
        for i, cls in enumerate(cls_names):
            y_bin = (y_te_roc == i).astype(int)
            if y_bin.sum() == 0: continue
            fpr, tpr, _ = roc_curve(y_bin, r_roc['y_proba'][:, i])
            auc_v = auc(fpr, tpr)
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr,
                                          name=f"{cls} (AUC={auc_v:.3f})",
                                          line=dict(color=roc_clrs[i % len(roc_clrs)], width=2)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Rastgele (AUC=0.50)",
                                      line=dict(dash='dash', color='#475569', width=1)))
        fig_roc.update_layout(template=T, title=f"ROC Eğrisi — {sel_roc}",
                               xaxis_title="False Positive Rate",
                               yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)
        try:
            oa = roc_auc_score(y_te_roc, r_roc['y_proba'], multi_class='ovr', average='weighted')
            pills(("Ağırlıklı OVR AUC", f"{oa:.4f}"))
        except Exception:
            pass

    with st.expander("💻 ROC / AUC Kodu"):
        st.code("""
from sklearn.metrics import roc_curve, auc, roc_auc_score
import plotly.graph_objects as go

y_proba = model.predict_proba(X_test)
fig = go.Figure()
for i, name in enumerate(class_names):
    y_bin = (y_test == i).astype(int)
    fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} AUC={auc(fpr,tpr):.3f}"))

oa = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
print(f"Ağırlıklı AUC = {oa:.4f}")
        """, language='python')

    st.markdown("---")
    sec("5 · 5-Fold Cross-Validation")

    cv_df = pd.DataFrame([{'Model': n, 'CV Ort.': r['cv_mean'], 'CV Std': r['cv_std']}
                           for n, r in results.items()])
    fig_cv = px.bar(cv_df, x='Model', y='CV Ort.', error_y='CV Std', color='Model',
                    title="5-Fold Cross-Validation Sonuçları",
                    color_discrete_sequence=CS, template=T)
    fig_cv.update_layout(showlegend=False, xaxis_tickangle=-15)
    st.plotly_chart(fig_cv, use_container_width=True)

    with st.expander("💻 Cross-Validation Kodu"):
        st.code("""
from sklearn.model_selection import cross_val_score

cv = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Ortalama: {cv.mean():.4f} +/- {cv.std():.4f}")
print(f"Her katlama: {cv}")
        """, language='python')

    st.markdown("---")
    sec("6 · Karar Ağacı: Özellik Önemi & Entropy vs Gini")

    dt_mdl = results['Karar Ağacı (Entropy)']['model']
    cf1, cf2 = st.columns(2)

    with cf1:
        fi = pd.DataFrame({'Özellik': [lbl(c) for c in feat_cols],
                           'Önem': dt_mdl.feature_importances_}
                          ).sort_values('Önem', ascending=True).tail(12)
        fig = px.bar(fi, x='Önem', y='Özellik', orientation='h',
                     title="Özellik Önemi — Entropy Karar Ağacı",
                     color='Önem', color_continuous_scale='Viridis', template=T)
        fig.update_layout(yaxis_tickfont_size=10)
        st.plotly_chart(fig, use_container_width=True)

    with cf2:
        e_r, g_r = results['Karar Ağacı (Entropy)'], results['Karar Ağacı (Gini)']
        eg_df = pd.DataFrame({'Metrik': ['Accuracy','Precision','Recall','F1'],
                               'Entropy': [e_r['accuracy'], e_r['precision'], e_r['recall'], e_r['f1']],
                               'Gini':    [g_r['accuracy'], g_r['precision'], g_r['recall'], g_r['f1']]})
        fig = go.Figure([
            go.Bar(name='Entropy', x=eg_df['Metrik'], y=eg_df['Entropy'],
                   marker_color='#6366f1', opacity=.85),
            go.Bar(name='Gini',    x=eg_df['Metrik'], y=eg_df['Gini'],
                   marker_color='#f59e0b', opacity=.85),
        ])
        fig.update_layout(barmode='group', template=T, title="Entropy vs Gini Performansı")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("💻 Entropy & Gini Kodu"):
        st.code("""
from sklearn.tree import DecisionTreeClassifier

# Entropy: Bilgi Kazancı (Information Gain) kullanır
# Gini:    Gini Safsızlığı (Gini Impurity) kullanır

dt_ent = DecisionTreeClassifier(criterion='entropy', max_depth=5)
dt_gin = DecisionTreeClassifier(criterion='gini',    max_depth=5)

# Entropy(S) = -sum(p_i * log2(p_i))
# Gini(S)    = 1 - sum(p_i^2)
# IG(S,A)    = Entropy(S) - sum(|Sv|/|S| * Entropy(Sv))
        """, language='python')

    st.markdown("---")
    sec("7 · KNN: k Optimizasyonu")

    # ─── FIX: underscore'suz parametreler → proper cache key ───
    @st.cache_data
    def knn_search(Xtr, Xte, ytr, yte):
        ks, accs = [], []
        for k in range(1, 21):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(Xtr, ytr)
            ks.append(k); accs.append(accuracy_score(yte, knn.predict(Xte)))
        return ks, accs

    ks, accs_k = knn_search(X_tr.values, X_te.values, y_tr, y_te)
    best_k = ks[int(np.argmax(accs_k))]

    fig_knn = px.line(x=ks, y=accs_k, markers=True,
                      title="KNN: k Değerine Göre Test Accuracy",
                      labels={'x':'k (Komşu Sayısı)','y':'Accuracy'},
                      color_discrete_sequence=['#60a5fa'], template=T)
    fig_knn.add_vline(x=best_k, line_dash="dash", line_color="#f87171",
                      annotation_text=f"En iyi k={best_k}",
                      annotation_font_color="#f87171")
    st.plotly_chart(fig_knn, use_container_width=True)
    pills(("En iyi k", str(best_k)), ("Max Accuracy", f"{max(accs_k):.4f}"))

    with st.expander("💻 KNN k Optimizasyonu Kodu"):
        st.code("""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

k_scores = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    k_scores.append(accuracy_score(y_test, knn.predict(X_test)))

best_k = k_scores.index(max(k_scores)) + 1
print(f"En iyi k={best_k}, Accuracy={max(k_scores):.4f}")
        """, language='python')

# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<br>
<div style='text-align:center; padding:20px; color:#334155; font-size:.82rem;
            border-top:1px solid #1e293b;'>
    💎 <b style='color:#6366f1'>Veri Madenciliği Master Paneli</b>
    &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; Plotly &nbsp;·&nbsp; Scikit-learn &nbsp;·&nbsp; SciPy
</div>
""", unsafe_allow_html=True)