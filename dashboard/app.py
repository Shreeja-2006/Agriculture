import streamlit as st
import pandas as pd
import numpy as np
import joblib


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AgriIntelligence",
    page_icon="🌾",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #f4f0e7;
            --surface: #fffdf9;
            --panel: #ffffff;
            --soft: #f3f7f0;
            --primary: #163d2d;
            --primary-2: #285a42;
            --sage: #a9c1a2;
            --sage-2: #dfeadf;
            --muted: #5a625c;
            --text: #1d2c27;
            --line: rgba(22, 61, 45, 0.12);
            --shadow: 0 18px 40px rgba(24, 42, 33, 0.08);
            --shadow-soft: 0 10px 20px rgba(24, 42, 33, 0.05);
            --green: #2a6c4d;
            --amber: #c98a2c;
            --red: #ae433f;
            --ink: #0f1d18;
        }

        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            font-family: 'Inter', sans-serif;
            background: var(--bg);
            color: var(--text);
        }

        .stApp {
            background: linear-gradient(180deg, #f8f4ed 0%, #f4f8f3 100%);
        }

        .main .block-container{
            padding-top: 0.55rem;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] {
            display: none;
        }

        [data-testid="stSidebarCollapsedControl"] {
            display: none;
        }

        h1, h2, h3, h4, h5 {
            letter-spacing: -0.04em;
            color: var(--ink);
            margin-top: 0;
        }

        .topbar {
            position: sticky;
            top: 0;
            z-index: 10;
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(18px);
            border-bottom: 1px solid rgba(22, 61, 45, 0.08);
            margin: -1rem -1.5rem 1.4rem -1.5rem;
            padding: 0.85rem 1.6rem;
        }

        .nav-shell {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            max-width: 1500px;
            margin: 0 auto;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            min-width: 180px;
        }

        .brand-mark {
            width: 42px;
            height: 42px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, var(--primary) 0%, var(--green) 100%);
            color: white;
            font-size: 1.2rem;
            box-shadow: var(--shadow-soft);
        }

        .brand-name {
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            color: var(--ink);
        }

        .nav-links {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 0.55rem;
        }

        .nav-link {
            font-size: 0.82rem;
            color: var(--muted);
            font-weight: 600;
            text-decoration: none;
            padding: 0.6rem 0.8rem;
            border-radius: 10px;
            transition: all 0.2s ease;
        }

        .nav-link:hover {
            background: rgba(42, 108, 77, 0.06);
            color: var(--primary);
        }

        .nav-button {
            background: linear-gradient(135deg, var(--primary) 0%, var(--green) 100%);
            border: none;
            color: white;
            padding: 0.8rem 1.2rem;
            border-radius: 12px;
            font-weight: 700;
            font-size: 0.8rem;
            box-shadow: var(--shadow-soft);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .nav-button:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 26px rgba(22, 61, 45, 0.18);
        }

        .hero-shell {
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 1.5rem;
            align-items: center;
            background: linear-gradient(135deg, rgba(255,255,255,0.65), rgba(238,245,239,0.9));
            border: 1px solid var(--line);
            border-radius: 28px;
            padding: 2rem;
            box-shadow: var(--shadow);
            margin-bottom: 1.6rem;
        }

        .eyebrow {
            display: inline-block;
            padding: 0.5rem 0.75rem;
            background: rgba(42, 108, 77, 0.08);
            color: var(--primary);
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }

        .hero-title {
            font-size: clamp(2.5rem, 5vw, 4.2rem);
            font-weight: 800;
            line-height: 0.98;
            letter-spacing: -0.06em;
            margin: 0;
            color: var(--primary);
        }

        .hero-subtitle {
            font-size: 1.15rem;
            color: var(--muted);
            line-height: 1.7;
            max-width: 620px;
            margin-top: 1rem;
        }

        .hero-actions {
            display: flex;
            gap: 0.9rem;
            margin-top: 1.4rem;
            flex-wrap: wrap;
        }

        .primary-btn,
        .secondary-btn {
            border-radius: 12px;
            border: none;
            padding: 0.9rem 1.3rem;
            font-weight: 700;
            font-size: 0.9rem;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .primary-btn {
            background: linear-gradient(135deg, var(--primary) 0%, var(--green) 100%);
            color: white;
            box-shadow: 0 14px 24px rgba(22, 61, 45, 0.18);
        }

        .secondary-btn {
            background: rgba(255,255,255,0.74);
            color: var(--primary);
            border: 1px solid rgba(22, 61, 45, 0.08);
        }

        .primary-btn:hover,
        .secondary-btn:hover {
            transform: translateY(-1px);
        }

        .hero-visual {
            position: relative;
            height: 360px;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(34, 70, 55, 0.98), rgba(120, 156, 104, 0.84));
            overflow: hidden;
            box-shadow: var(--shadow);
            border: 1px solid rgba(255,255,255,0.15);
        }

        .hero-visual::before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.01));
        }

        .visual-panel {
            position: absolute;
            right: 1rem;
            bottom: 1rem;
            left: 1rem;
            background: rgba(255,255,255,0.09);
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 20px;
            padding: 1rem;
            backdrop-filter: blur(8px);
        }

        .visual-bar {
            height: 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.18);
            overflow: hidden;
            margin-bottom: 0.7rem;
        }

        .visual-bar > span {
            display: block;
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #f8f7f2 0%, #cfe3d3 100%);
        }

        .visual-caption {
            color: rgba(255,255,255,0.86);
            font-weight: 600;
            font-size: 0.82rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem;
            margin: 1.4rem 0 1.8rem;
        }

        .hero-stat {
            background: rgba(255,255,255,0.78);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1.2rem 1.1rem;
            box-shadow: var(--shadow-soft);
        }

        .hero-stat .label {
            color: var(--muted);
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
        }

        .hero-stat .value {
            font-size: clamp(1.9rem, 3vw, 2.5rem);
            font-weight: 800;
            color: var(--primary);
            margin-top: 0.35rem;
            letter-spacing: -0.05em;
        }

        .section-heading {
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }

        .section-heading h2 {
            font-size: clamp(1.9rem, 3vw, 2.5rem);
            color: var(--primary);
            letter-spacing: -0.05em;
            margin: 0;
        }

        .subheading {
            color: var(--muted);
            font-size: 1rem;
            margin-top: 0.5rem;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1.2rem;
            margin-top: 1.2rem;
        }

        .feature-card {
            background: rgba(255,255,255,0.82);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1.5rem;
            box-shadow: var(--shadow-soft);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .feature-card:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow);
        }

        .feature-icon {
            width: 52px;
            height: 52px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 14px;
            background: linear-gradient(135deg, rgba(42,108,77,0.1), rgba(169,193,162,0.3));
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }

        .feature-card h3 {
            font-size: 1.25rem;
            margin-bottom: 0.4rem;
            color: var(--ink);
        }

        .feature-card p {
            color: var(--muted);
            line-height: 1.7;
            margin: 0;
        }

        .premium-panel {
            background: rgba(255,255,255,0.85);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1.3rem;
            box-shadow: var(--shadow-soft);
            margin-top: 1.4rem;
        }

        .kpi-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem;
            margin: 1rem 0 1.4rem;
        }

        .mini-kpi {
            background: rgba(255,255,255,0.84);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: var(--shadow-soft);
        }

        .mini-kpi .label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            font-weight: 700;
        }

        .mini-kpi .number {
            font-size: 1.5rem;
            font-weight: 800;
            color: var(--primary);
            letter-spacing: -0.04em;
            margin-top: 0.35rem;
        }

        .risk-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            padding: 0.42rem 0.8rem;
            font-size: 0.75rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .risk-low {
            background: rgba(42,108,77,0.12);
            color: var(--green);
        }

        .risk-moderate {
            background: rgba(201,138,44,0.12);
            color: var(--amber);
        }

        .risk-high {
            background: rgba(174,67,63,0.12);
            color: var(--red);
        }

        .card-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
            color: var(--ink);
        }

        .compare-card {
            background: rgba(255,255,255,0.82);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: var(--shadow-soft);
        }

        .result-panel {
            background: linear-gradient(135deg, rgba(42,108,77,0.09), rgba(169,193,162,0.16));
            border: 1px solid rgba(42,108,77,0.15);
            border-radius: 24px;
            padding: 1.4rem 1.3rem;
            box-shadow: var(--shadow-soft);
            margin-top: 1rem;
        }

        .result-panel .tag {
            color: var(--primary);
            font-size: 0.74rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 800;
        }

        .result-panel .big {
            font-size: clamp(2.1rem, 4vw, 3rem);
            color: var(--primary);
            font-weight: 800;
            letter-spacing: -0.05em;
            margin-top: 0.2rem;
        }

        .result-panel .detail {
            color: var(--muted);
            margin-top: 0.45rem;
            line-height: 1.6;
        }

        .footer {
            margin-top: 2rem;
            padding-top: 1.4rem;
            border-top: 1px solid var(--line);
            color: var(--muted);
            display: grid;
            grid-template-columns: 1.2fr 1fr 1fr 1fr;
            gap: 1.2rem;
        }

        .footer-brand {
            font-size: 1.15rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            color: var(--ink);
        }

        .footer h4 {
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.85rem;
        }

        .footer a {
            color: var(--muted);
            text-decoration: none;
            display: block;
            margin-bottom: 0.35rem;
        }

        .footer-note {
            margin-top: 0.8rem;
            color: var(--muted);
            font-size: 0.8rem;
        }

        [data-testid="stHorizontalBlock"] .stButton > button {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            padding: 0.62rem 0.48rem;
            font-size: 0.78rem;
            min-height: 44px;
        }

        @media (max-width: 1100px) {
            .hero-shell, .feature-grid, .footer {
                grid-template-columns: 1fr;
            }
            .stats-grid, .kpi-row {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 720px) {
            .nav-shell {
                flex-wrap: wrap;
                justify-content: center;
            }
            .brand {
                min-width: 0;
            }
            .stats-grid, .kpi-row {
                grid-template-columns: 1fr;
            }
            .hero-shell {
                padding: 1.2rem;
            }
            .hero-title {
                font-size: 2.5rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    master = pd.read_csv("data/processed/agri_master.csv")
    recommendations = pd.read_csv("data/processed/crop_recommendations.csv")
    weather_risk = pd.read_csv("data/processed/weather_risk.csv")
    return master, recommendations, weather_risk


@st.cache_resource
def load_yield_model():
    return joblib.load("models/yield_prediction_model.pkl")


@st.cache_resource
def load_price_model():
    return joblib.load("models/price_prediction_model.pkl")


df, recommendations, weather_risk = load_data()
yield_model = load_yield_model()
price_model = load_price_model()


def format_currency(value):
    return f"₹{value:,.0f}"


def format_yield(value):
    return f"{value:.2f} t/ha"


def format_price_per_quintal(value):
    return f"₹{value:,.0f} / Quintal"


# --------------------------------------------------
# NAVIGATION STATE
# --------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "Home"


nav_items = {
    "Home": "Home",
    "Crops": "Crop Analytics",
    "Weather": "Weather",
    "Market": "Market",
    "Recommend": "Recommendations",
    "Yield": "Yield Prediction",
    "Prices": "Price Prediction",
}


# --------------------------------------------------
# TOP NAVIGATION
# --------------------------------------------------

page = st.session_state.page

header_cols = st.columns([1.55, 5.65, 0.95])

with header_cols[0]:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.7rem; min-height:48px; margin-top:0.15rem;">
            <div style="width:40px; height:40px; border-radius:12px; display:flex; align-items:center; justify-content:center; background:linear-gradient(135deg, #163d2d 0%, #2a6c4d 100%); color:white; font-size:1.1rem;">🌾</div>
            <div style="font-size:1.45rem; font-weight:800; letter-spacing:-0.05em; color:#163d2d; line-height:1;">AgriIntelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_cols[1]:
    nav_buttons = st.columns(len(nav_items))
    for idx, (label, key) in enumerate(nav_items.items()):
        with nav_buttons[idx]:
            if st.button(label, key=f"nav_{label}", use_container_width=True):
                st.session_state.page = key
                page = key

with header_cols[2]:
    st.markdown(
        """
        <div style="display:flex; justify-content:flex-end; align-items:center; min-height:48px;">
            <div style="background:linear-gradient(135deg, #163d2d 0%, #2a6c4d 100%); color:white; border-radius:12px; padding:0.8rem 1rem; font-weight:700; font-size:0.9rem; line-height:1.1; box-shadow:0 10px 20px rgba(22,61,45,0.12); text-align:center;">
                Explore<br>Insights
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

page = st.session_state.page


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "Home":
    st.markdown(
        """
        <div class="hero-shell">
            <div>
                <div class="eyebrow">AgriTech Platform</div>
                <h1 class="hero-title">AgriIntelligence</h1>
                <div class="hero-subtitle">Data-Driven Intelligence<br>for Smarter Agriculture</div>
                <p class="hero-subtitle" style="margin-top: 1rem; max-width: 620px;">
                    Understand crop performance, anticipate market movements, analyze weather risks,
                    and make data-driven agricultural decisions.
                </p>
                <div class="hero-actions">
                    <button class="primary-btn">Explore Analytics</button>
                    <button class="secondary-btn">Get Crop Recommendation</button>
                </div>
            </div>
            <div class="hero-visual">
                <div class="visual-panel">
                    <div class="visual-caption">Field performance</div>
                    <div class="visual-bar"><span style="width: 78%"></span></div>
                    <div class="visual-bar"><span style="width: 64%"></span></div>
                    <div class="visual-bar"><span style="width: 86%"></span></div>
                    <div class="visual-caption" style="margin-top:1rem;">Regional insight index: 81%</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stats = [
        ("Crops", df["Crop"].nunique()),
        ("States", df["State"].nunique()),
        ("Districts", df["District"].nunique()),
        ("Years of Data", df["Year"].nunique()),
    ]

    st.markdown('<div class="stats-grid">', unsafe_allow_html=True)
    for label, value in stats:
        st.markdown(
            f"""
            <div class="hero-stat">
                <div class="label">{label}</div>
                <div class="value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-heading">
            <h2>One Platform. Three Data Signals.</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    feature_cols = st.columns(3)
    feature_cards = [
        ("🌾", "Crop Intelligence", "Analyze yield, production and crop performance with a regional lens."),
        ("🌧️", "Weather Intelligence", "Understand rainfall, temperature, humidity and weather risk before planning."),
        ("💹", "Market Intelligence", "Track agricultural prices and market trends to uncover opportunity."),
    ]

    for idx, (icon, title, desc) in enumerate(feature_cards):
        with feature_cols[idx]:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <h3>{title}</h3>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="section-heading">
            <h2>Data Snapshot</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(df.head(15), use_container_width=True)


# --------------------------------------------------
# CROP ANALYTICS
# --------------------------------------------------

elif page == "Crop Analytics":
    st.markdown(
        """
        <div class="section-heading">
            <h2>Crop Intelligence</h2>
            <div class="subheading">Understand how crops perform across regions and time.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    state_col, district_col, year_col = st.columns(3)
    with state_col:
        selected_state = st.selectbox("State", sorted(df["State"].unique()))
    with district_col:
        districts = sorted(df[df["State"] == selected_state]["District"].unique())
        selected_district = st.selectbox("District", districts)
    with year_col:
        selected_year = st.selectbox("Year", sorted(df["Year"].unique(), reverse=True))

    filtered = df[
        (df["State"] == selected_state)
        & (df["District"] == selected_district)
        & (df["Year"] == selected_year)
    ].copy()

    if filtered.empty:
        st.warning("No records are available for the selected state, district, and year.")
    else:
        crop_kpi = st.columns(4)
        with crop_kpi[0]:
            st.metric("Average Yield", format_yield(filtered["Yield_Tonnes_Per_Hectare"].mean()))
        with crop_kpi[1]:
            st.metric("Production", f"{filtered['Production_Tonnes'].sum():,.0f} t")
        with crop_kpi[2]:
            st.metric("Average Market Price", format_currency(filtered["Avg_Modal_Price"].mean()))
        with crop_kpi[3]:
            top_crop = filtered.groupby("Crop")["Yield_Tonnes_Per_Hectare"].mean().idxmax()
            st.metric("Top Performing Crop", top_crop)

        crop_yield = (
            filtered.groupby("Crop")["Yield_Tonnes_Per_Hectare"].mean().sort_values(ascending=False)
        )
        st.markdown('<div class="premium-panel"><div class="card-title">Crop Yield Overview</div>', unsafe_allow_html=True)
        st.bar_chart(crop_yield)
        st.markdown('</div>', unsafe_allow_html=True)

        summary = (
            filtered.groupby("Crop")
            .agg(
                Average_Yield=("Yield_Tonnes_Per_Hectare", "mean"),
                Average_Production=("Production_Tonnes", "mean"),
                Average_Market_Price=("Avg_Modal_Price", "mean")
            )
            .reset_index()
        )
        st.markdown('<div class="premium-panel"><div class="card-title">Production and Yield Insights</div>', unsafe_allow_html=True)
        st.dataframe(summary.sort_values("Average_Yield", ascending=False).round(2), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------
# WEATHER ANALYTICS
# --------------------------------------------------

elif page == "Weather":
    st.markdown(
        """
        <div class="section-heading">
            <h2>Weather Intelligence</h2>
            <div class="subheading">Understand environmental conditions before making agricultural decisions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_state = st.selectbox("State", sorted(weather_risk["State"].unique()))
    weather_filtered = weather_risk[weather_risk["State"] == selected_state].copy()

    if weather_filtered.empty:
        st.warning("No weather records are available for the selected state.")
    else:
        rainfall_avg = weather_filtered["Annual_Rainfall_mm"].mean()
        temp_avg = weather_filtered["Avg_Temperature_C"].mean()
        humidity_avg = weather_filtered["Avg_Humidity_Percent"].mean()

        weather_kpis = st.columns(3)
        with weather_kpis[0]:
            st.metric("Rainfall", f"{rainfall_avg:,.0f} mm")
        with weather_kpis[1]:
            st.metric("Temperature", f"{temp_avg:.1f} °C")
        with weather_kpis[2]:
            st.metric("Humidity", f"{humidity_avg:.1f}%")

        st.markdown('<div class="premium-panel"><div class="card-title">Weather Risk</div>', unsafe_allow_html=True)
        risk_counts = weather_filtered["Weather_Risk"].value_counts()
        for risk in ["Low", "Moderate", "High"]:
            count = int(risk_counts.get(risk, 0))
            if risk == "Low":
                css_class = "risk-low"
            elif risk == "Moderate":
                css_class = "risk-moderate"
            else:
                css_class = "risk-high"
            st.markdown(
                f"<div style='margin-bottom:0.8rem;'><span class='risk-pill {css_class}'>{risk}</span> <span style='margin-left:0.75rem; color: var(--muted); font-weight:600;'>{count} records</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        rainfall = weather_filtered.groupby("Year")["Annual_Rainfall_mm"].mean().sort_index()
        st.markdown('<div class="premium-panel"><div class="card-title">Rainfall Trend</div>', unsafe_allow_html=True)
        st.line_chart(rainfall)
        st.markdown('</div>', unsafe_allow_html=True)

        risk_summary = weather_filtered["Weather_Risk"].mode().iloc[0] if not weather_filtered["Weather_Risk"].empty else "Unknown"
        st.markdown(
            f"""
            <div class="result-panel">
                <div class="tag">Environmental Insight</div>
                <div class="big" style="font-size:1.5rem; margin-top:0.5rem;">{selected_state}</div>
                <div class="detail">The selected region shows a predominant <strong>{risk_summary}</strong> risk profile, with an average rainfall of <strong>{rainfall_avg:,.0f} mm</strong>, average temperature of <strong>{temp_avg:.1f} °C</strong>, and humidity of <strong>{humidity_avg:.1f}%</strong>.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# MARKET ANALYTICS
# --------------------------------------------------

elif page == "Market":
    st.markdown(
        """
        <div class="section-heading">
            <h2>Market Intelligence</h2>
            <div class="subheading">Track agricultural price movements and identify market trends.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_crop = st.selectbox("Crop", sorted(df["Crop"].unique()))
    market_filtered = df[df["Crop"] == selected_crop].copy()

    if market_filtered.empty:
        st.warning("No market data are available for this crop.")
    else:
        kpis = st.columns(4)
        with kpis[0]:
            st.metric("Current / Average Price", format_price_per_quintal(market_filtered["Avg_Modal_Price"].mean()))
        with kpis[1]:
            st.metric("Highest Price", format_price_per_quintal(market_filtered["Avg_Modal_Price"].max()))
        with kpis[2]:
            st.metric("Lowest Price", format_price_per_quintal(market_filtered["Avg_Modal_Price"].min()))
        with kpis[3]:
            st.metric("Price Trend", f"{market_filtered['Year'].nunique()} yrs")

        st.markdown('<div class="premium-panel"><div class="card-title">Price Trend</div>', unsafe_allow_html=True)
        price_trend = market_filtered.groupby("Year")["Avg_Modal_Price"].mean().sort_index()
        st.line_chart(price_trend)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="premium-panel"><div class="card-title">State Comparison</div>', unsafe_allow_html=True)
        state_prices = market_filtered.groupby("State")["Avg_Modal_Price"].mean().sort_values(ascending=False)
        st.bar_chart(state_prices)
        st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------
# CROP RECOMMENDATION
# --------------------------------------------------

elif page == "Recommendations":
    st.markdown(
        """
        <div class="section-heading">
            <h2>Find the Right Crop to Grow</h2>
            <div class="subheading">Balance yield potential, market opportunity and weather suitability.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    state_col, district_col, year_col = st.columns(3)
    with state_col:
        selected_state = st.selectbox("State", sorted(df["State"].unique()))
    with district_col:
        selected_district = st.selectbox("District", sorted(df[df["State"] == selected_state]["District"].unique()))
    with year_col:
        selected_year = st.selectbox("Year", sorted(df["Year"].unique(), reverse=True))

    location_data = df[
        (df["State"] == selected_state)
        & (df["District"] == selected_district)
        & (df["Year"] == selected_year)
    ].copy()

    if location_data.empty:
        st.warning("No recommendation data is available for the selected location and year.")
    else:
        def normalize(series):
            minimum = series.min()
            maximum = series.max()
            if maximum == minimum:
                return pd.Series([0.5] * len(series), index=series.index)
            return (series - minimum) / (maximum - minimum)

        risk_data = weather_risk[
            (weather_risk["State"] == selected_state)
            & (weather_risk["District"] == selected_district)
            & (weather_risk["Year"] == selected_year)
        ]

        risk_score = risk_data.iloc[0]["Risk_Score"] if not risk_data.empty else 0
        weather_category = risk_data.iloc[0]["Weather_Risk"] if not risk_data.empty else "Unknown"

        results = location_data[["Crop", "Yield_Tonnes_Per_Hectare", "Avg_Modal_Price", "Annual_Rainfall_mm", "Avg_Temperature_C"]].copy()
        results["Yield_Score"] = normalize(results["Yield_Tonnes_Per_Hectare"])
        results["Price_Score"] = normalize(results["Avg_Modal_Price"])
        results["Risk_Score"] = 1 - (risk_score / 5)
        results["Risk_Score"] = results["Risk_Score"].clip(0, 1)
        results["Recommendation_Score"] = (results["Yield_Score"] * 0.50) + (results["Price_Score"] * 0.30) + (results["Risk_Score"] * 0.20)
        results["Recommendation_Percentage"] = results["Recommendation_Score"] * 100
        results = results.sort_values("Recommendation_Score", ascending=False).reset_index(drop=True)
        results["Rank"] = results.index + 1
        best = results.iloc[0]

        st.markdown(
            f"""
            <div class="result-panel">
                <div class="tag">Recommended Crop</div>
                <div class="big">{best['Crop']}</div>
                <div class="detail">{best['Yield_Tonnes_Per_Hectare']:.2f} tonnes / hectare · {format_currency(best['Avg_Modal_Price'])} / Quintal · Weather Risk: {weather_category}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Expected Yield", format_yield(best["Yield_Tonnes_Per_Hectare"]))
        with metric_cols[1]:
            st.metric("Average Market Price", format_currency(best["Avg_Modal_Price"]))
        with metric_cols[2]:
            st.metric("Weather Risk", weather_category)
        with metric_cols[3]:
            st.metric("Recommendation Score", f"{best['Recommendation_Percentage']:.1f}%")

        st.markdown(
            """
            <div class="section-heading">
                <h2>Why this crop?</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(
            f"**{best['Crop']}** ranks first because it balances the strongest yield performance, favorable market price, and weather suitability in **{selected_district}, {selected_state}** during **{selected_year}**."
        )

        display_results = results[["Rank", "Crop", "Yield_Tonnes_Per_Hectare", "Avg_Modal_Price", "Risk_Score", "Recommendation_Percentage"]].copy()
        display_results.columns = ["Rank", "Crop", "Yield (t/ha)", "Market Price (₹)", "Weather Score", "Recommendation Score (%)"]

        st.markdown('<div class="premium-panel"><div class="card-title">Compare Crops</div>', unsafe_allow_html=True)
        st.dataframe(display_results.round(2), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="premium-panel"><div class="card-title">Recommendation Score</div>', unsafe_allow_html=True)
        st.bar_chart(results.set_index("Crop")["Recommendation_Percentage"])
        st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------
# YIELD PREDICTION
# --------------------------------------------------

elif page == "Yield Prediction":
    st.markdown(
        """
        <div class="section-heading">
            <h2>Predict Crop Yield</h2>
            <div class="subheading">Estimate expected yield using agricultural and environmental conditions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("yield_form"):
        st.markdown('<div class="premium-panel">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            selected_state = st.selectbox("State", sorted(df["State"].unique()))
            selected_district = st.selectbox("District", sorted(df[df["State"] == selected_state]["District"].unique()))
            selected_crop = st.selectbox("Crop", sorted(df["Crop"].unique()))
            year = st.number_input("Year", min_value=2020, max_value=2035, value=2026)
            area = st.number_input("Area", min_value=100.0, max_value=100000.0, value=10000.0)
        with col2:
            rainfall = st.number_input("Rainfall", min_value=0.0, max_value=3000.0, value=800.0)
            temperature = st.number_input("Temperature", min_value=0.0, max_value=50.0, value=27.0)
            humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, value=65.0)
            market_price = st.number_input("Market Price", min_value=0.0, max_value=50000.0, value=3000.0)
        submitted = st.form_submit_button("Predict Yield", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        input_data = pd.DataFrame({
            "State": [selected_state],
            "District": [selected_district],
            "Crop": [selected_crop],
            "Year": [year],
            "Area_Hectares": [area],
            "Annual_Rainfall_mm": [rainfall],
            "Avg_Temperature_C": [temperature],
            "Avg_Humidity_Percent": [humidity],
            "Avg_Modal_Price": [market_price]
        })

        input_data["Rainfall_per_Hectare"] = input_data["Annual_Rainfall_mm"] / input_data["Area_Hectares"]
        input_data["Price_Yield_Ratio"] = input_data["Avg_Modal_Price"] / 1.0
        input_data["Years_From_2020"] = input_data["Year"] - 2020
        input_encoded = pd.get_dummies(input_data, columns=["State", "District", "Crop"], drop_first=True)
        model_columns = yield_model.feature_names_in_
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

        prediction = max(0, yield_model.predict(input_encoded)[0])

        st.markdown(
            f"""
            <div class="result-panel">
                <div class="tag">Predicted Yield</div>
                <div class="big">{prediction:.2f}</div>
                <div class="detail">tonnes / hectare</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(f"Estimated yield for {selected_crop} in {selected_district}, {selected_state} under the current input conditions.")


# --------------------------------------------------
# PRICE PREDICTION
# --------------------------------------------------

elif page == "Price Prediction":
    st.markdown(
        """
        <div class="section-heading">
            <h2>Predict Market Price</h2>
            <div class="subheading">Estimate future market prices using historical market patterns.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("price_form"):
        st.markdown('<div class="premium-panel">', unsafe_allow_html=True)
        crop_col, year_col, month_col = st.columns(3)
        with crop_col:
            selected_crop = st.selectbox("Crop", sorted(df["Crop"].unique()))
        with year_col:
            selected_year = st.number_input("Year", min_value=2020, max_value=2035, value=2026)
        with month_col:
            selected_month = st.selectbox("Month", list(range(1, 13)), index=0)

        previous_price = st.number_input("Previous Modal Price", min_value=0.0, max_value=50000.0, value=3000.0)
        submitted = st.form_submit_button("Predict Market Price", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        input_data = pd.DataFrame({
            "Year": [selected_year],
            "Month": [selected_month],
            "Previous_Modal_Price": [previous_price]
        })
        prediction = max(0, price_model.predict(input_data)[0])
        price_change = prediction - previous_price

        st.markdown(
            f"""
            <div class="result-panel">
                <div class="tag">Predicted Market Price</div>
                <div class="big">{format_currency(prediction)}</div>
                <div class="detail">per Quintal</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        result_cols = st.columns(3)
        with result_cols[0]:
            st.metric("Previous Price", format_currency(previous_price))
        with result_cols[1]:
            st.metric("Predicted Price", format_currency(prediction))
        with result_cols[2]:
            movement = "Expected upward movement" if price_change >= 0 else "Expected downward movement"
            st.metric("Price Change", f"{movement}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        <div>
            <div class="footer-brand">AgriIntelligence</div>
            <div>Data-Driven Intelligence for Smarter Agriculture</div>
        </div>
        <div>
            <h4>About</h4>
            <a href="#">Overview</a>
            <a href="#">Platform</a>
            <a href="#">Mission</a>
        </div>
        <div>
            <h4>Analytics</h4>
            <a href="#">Crop</a>
            <a href="#">Weather</a>
            <a href="#">Market</a>
        </div>
        <div>
            <h4>Predictions</h4>
            <a href="#">Yield</a>
            <a href="#">Prices</a>
            <a href="#">Recommendations</a>
        </div>
    </div>
    <div class="footer-note">Engineering Capstone Project 2026–2027</div>
    """,
    unsafe_allow_html=True,
)