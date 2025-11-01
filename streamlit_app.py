# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import altair as alt

# --------------------------
# Page Settings
# --------------------------
st.set_page_config(page_title="PrithviSense", layout="wide")

# --------------------------
# Updated Background & UI Styling
# --------------------------
st.markdown("""
<style>
:root{
  --sea1: #e9f5f1;
  --sea2: #d2efe6;
  --card-bg: rgba(255,255,255,0.82);
}

/* Page Background */
body {
    background: linear-gradient(135deg, var(--sea1), var(--sea2)) fixed !important;
    background-size: cover;
    margin: 0;
}

/* Blur behind content */
.reportview-container .main .block-container {
    backdrop-filter: blur(10px) brightness(1.08);
    padding-top: 12px;
}

/* Card styling */
.card {
    background: var(--card-bg);
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.07);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    margin-bottom: 18px;
}
.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 26px rgba(0,0,0,0.14);
}

/* Center Text Utility */
.center-text { text-align: center; }

/* Dataframe: Center Align Columns */
.dataframe td, .dataframe th {
    text-align: center !important;
    vertical-align: middle !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Data Setup
# --------------------------
DATA_FILE = "thermal_data.csv"
MODEL_FILE = "thermal_model.joblib"

ZONES = [
    "Main Parking Lot", "Academic Block A", "Academic Block B",
    "Boys Hostel 1", "Boys Hostel 2", "Girls Hostel",
    "Sports Stadium", "Central Library", "Green Quad", "Food Court"
]

def load_data():
    try:
        df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
        return df
    except:
        now = pd.Timestamp.now().floor("H")
        rows = []
        for z in ZONES:
            for h in range(48):
                ts = now - pd.Timedelta(hours=h)
                temp = round(28 + np.sin(h/3)*5 + np.random.randn()*0.7, 1)
                uv = round(max(0, np.sin(h/4)*8 + np.random.randn()*0.4), 1)
                rows.append([ts, z, temp, uv])
        return pd.DataFrame(rows, columns=["timestamp","zone","temp","uv"])

df = load_data()

def load_model():
    try:
        return joblib.load(MODEL_FILE)
    except:
        return None

model = load_model()

# --------------------------
# Status Logic
# --------------------------
def status_label(temp):
    if temp > 40: return "Hotspot"
    if temp > 36: return "Medium"
    return "Safe"

latest = df.sort_values("timestamp").groupby("zone").last().reset_index()
latest["status"] = latest["temp"].apply(status_label)

# --------------------------
# Header
# --------------------------
st.markdown("<h1 class='center-text' style='font-size:48px;'>PrithviSense</h1>", unsafe_allow_html=True)
st.markdown("<p class='center-text' style='font-size:20px; font-style:italic;'>Smart Thermal Awareness for Sustainable Campuses</p>", unsafe_allow_html=True)

# --------------------------
# Section 1: Current Table (Centered)
# --------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("1) Current Campus Heat & UV Status")

st.dataframe(
    latest[["zone","temp","uv","status"]].style.set_properties(**{
        'text-align': 'center'
    }).set_table_styles([
        dict(selector='th', props=[('text-align', 'center')])
    ])
)
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# Section 2: Trends
# --------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("2) Temperature Trend by Zone")
selected_zone = st.selectbox("Select Zone", ZONES)
hist = df[df.zone == selected_zone].sort_values("timestamp").tail(48)
chart = alt.Chart(hist).mark_line().encode(
    x="timestamp:T",
    y="temp:Q",
    tooltip=["timestamp:T","temp"]
).properties(height=260)
st.altair_chart(chart, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# Section 3: Forecast
# --------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("3) Forecast Heat & UV Conditions")

col1, col2, col3 = st.columns(3)
zone_in = col1.selectbox("Zone", ZONES)
date_in = col2.date_input("Date", datetime.now().date())
time_in = col3.time_input("Time", (datetime.now()+timedelta(hours=1)).time())

if st.button("Predict"):
    if model:
        ts = pd.to_datetime(f"{date_in} {time_in}")
        X = pd.DataFrame([{
            "hour": ts.hour, "dayofweek": ts.dayofweek, "month": ts.month,
            **{f"zone_{z}": int(z==zone_in) for z in ZONES}
        }])
        pred = model.predict(X)[0]
        p_temp, p_uv = round(pred[0],1), round(pred[1],1)
    else:
        p_temp = round(latest[latest.zone==zone_in].temp.values[0] + np.random.randn(),1)
        p_uv = round(latest[latest.zone==zone_in].uv.values[0] + np.random.randn()*0.4,1)

    st.success(f"🌡 Temperature: **{p_temp}°C**   |   ☀ UV Index: **{p_uv}**")

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# Section 4: ROI
# --------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("4) ROI Calculator for Heat Mitigation Actions")

c1, c2, c3 = st.columns(3)
cost = c1.number_input("Installation Cost (₹)", value=20000)
saving = c2.number_input("Yearly Savings (₹)", value=5000)
life = c3.number_input("Expected Lifetime (years)", value=5)

total = saving * life
roi = ((total - cost) / cost) * 100 if cost > 0 else 0

st.info(f"Total Savings Over Lifetime: **₹{total:,}**")
st.success(f"Estimated ROI: **{roi:.1f}%**")
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# Footer
# --------------------------
st.markdown("<p class='center-text' style='color:#444;'>Made with ❤️ for Hackathons</p>", unsafe_allow_html=True)
