# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import altair as alt

# ------------------------------------------------
# Page Config
# ------------------------------------------------
st.set_page_config(page_title="PrithviSense", layout="wide")

# ------------------------------------------------
# UI Styling (100% Streamlit Cloud compatible)
# ------------------------------------------------
st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #d9f8ed, #c3e9ff, #e7ffd9);
    background-size: 300% 300%;
    animation: gradientMove 14s ease infinite;
    font-family: 'Segoe UI', sans-serif;
}
@keyframes gradientMove { 
    0% { background-position: 0% 0%; } 
    50% { background-position: 100% 100%; } 
    100% { background-position: 0% 0%; }
}

[data-testid="stHeader"] { background: transparent; }

/* Center Text */
.center-text { text-align:center; }

/* Bigger Project Title */
h1.center-text {
    font-size: 64px !important;
    font-weight: 700 !important;
    letter-spacing: 1px;
    color: #134b39;
    text-shadow: 0px 3px 10px rgba(0,0,0,0.18);
}


/* Tagline */
p.tagline {
    font-size:20px;
    font-style:italic;
    color:#225b44;
    margin-top:-12px;
}

/* Card Styling */
.card {
    background: rgba(255,255,255,0.78);
    padding: 24px;
    border-radius: 14px;
    box-shadow: 0px 6px 22px rgba(0,0,0,0.12);
    backdrop-filter: blur(14px);
    margin-bottom: 22px;
    transition: transform 0.28s ease, box-shadow 0.28s ease;
}
.card:hover {
    transform: translateY(-6px);
    box-shadow: 0px 10px 30px rgba(0,0,0,0.18);
}

/* Buttons */
button[kind="primary"] {
    background-color: #1b7057 !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 600 !important;
}
button[kind="primary"]:hover {
    background-color: #145843 !important;
    box-shadow: 0 0 14px rgba(27,112,87,0.6) !important;
    transform: translateY(-2px);
}

/* Table Hover */
tbody tr:hover {
    background: #eafff5 !important;
    cursor: pointer;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Data Load
# ------------------------------------------------
DATA_FILE = "thermal_data.csv"
MODEL_FILE = "thermal_model.joblib"

ZONES = [
    "Main Parking Lot","Academic Block A","Academic Block B",
    "Boys Hostel 1","Boys Hostel 2","Girls Hostel",
    "Sports Stadium","Central Library","Green Quad","Food Court"
]

def load_data():
    try:
        return pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    except:
        # Generate demo data
        now = pd.Timestamp.now().floor("H")
        rows=[]
        for z in ZONES:
            for h in range(48):
                ts = now - pd.Timedelta(hours=h)
                temp = round(28 + np.sin(h/4)*4 + np.random.randn()*0.5, 1)
                uv = round(max(0, np.sin(h/4)*7 + np.random.randn()*0.3), 1)
                rows.append([ts, z, temp, uv])
        return pd.DataFrame(rows, columns=["timestamp","zone","temp","uv"])

df = load_data()

def load_model():
    try: return joblib.load(MODEL_FILE)
    except: return None

model = load_model()

def status(temp):
    if temp > 40: return "🔥 Hotspot"
    if temp > 36: return "🌤 Medium"
    return "✅ Safe"

latest = df.sort_values("timestamp").groupby("zone").last().reset_index()
latest["status"] = latest["temp"].apply(status)

# ------------------------------------------------
# Header
# ------------------------------------------------
st.markdown("<h1 class='center-text'>PrithviSense</h1>", unsafe_allow_html=True)
st.markdown("<p class='center-text tagline'>Smart Thermal Awareness for Sustainable Campuses</p>", unsafe_allow_html=True)
st.write("")

# ------------------------------------------------
# 1) Current Status
# ------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("1) Current Campus Heat & UV Status")
st.dataframe(latest[["zone","temp","uv","status"]], use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------
# 2) Trends
# ------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("2) Temperature Trend by Zone")

zone = st.selectbox("Choose Zone", ZONES)
hist = df[df.zone == zone].sort_values("timestamp").tail(48)

chart = alt.Chart(hist).mark_line(color="#1b7057", strokeWidth=2).encode(
    x="timestamp:T", y="temp:Q",
    tooltip=["timestamp:T","temp"]
).properties(height=260)

st.altair_chart(chart, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------
# 3) Forecast
# ------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("3) Forecast (Model Prediction)")

c1, c2, c3 = st.columns(3)
zone_in = c1.selectbox("Zone to Forecast", ZONES)
date_in = c2.date_input("Date", datetime.now().date())
time_in = c3.time_input("Time", datetime.now().time())

if st.button("Predict"):
    if model:
        ts = pd.to_datetime(f"{date_in} {time_in}")
        X = pd.DataFrame([{
            "hour": ts.hour, "dayofweek": ts.dayofweek, "month": ts.month,
            **{f"zone_{z}": int(z==zone_in) for z in ZONES}
        }])
        p_temp, p_uv = model.predict(X)[0]
        p_temp, p_uv = round(p_temp,1), round(p_uv,1)
    else:
        base = latest[latest.zone==zone_in].iloc[0]
        p_temp = round(base.temp + np.random.randn(),1)
        p_uv = round(base.uv + np.random.randn()*0.4,1)

    st.success(f"🌡 Temperature: **{p_temp}°C**  |  ☀ UV Index: **{p_uv}**")

    if p_uv <= 2.5:
        st.success("✅ UV Safe — Outdoor activities are okay.")
    else:
        st.error("⚠ UV High — Limit sunlight, stay hydrated, seek shade.")

st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------
# 4) ROI Calculator
# ------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("4) ROI Calculator")

colA, colB, colC = st.columns(3)
cost = colA.number_input("Installation Cost (₹)", value=20000)
saving = colB.number_input("Yearly Savings (₹)", value=5000)
life = colC.number_input("Lifetime (years)", value=5)

total = saving * life
roi = ((total - cost) / cost) * 100 if cost > 0 else 0

st.info(f"💰 Total Savings: **₹{total:,}**")
st.success(f"📈 ROI: **{roi:.1f}%**")
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.markdown("<p class='center-text' style='color:#445;'>Made with ❤️ for Hackathons</p>", unsafe_allow_html=True)

