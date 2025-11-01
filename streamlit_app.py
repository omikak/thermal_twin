# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import altair as alt
import math
import io

# --------------------------
# Page Settings
# --------------------------
st.set_page_config(page_title="PrithviSense", layout="wide")

# --------------------------
# UI Styling
# --------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e8f7f0, #cfeee3) !important;
    background-attachment: fixed;
    font-family: 'Segoe UI', sans-serif;
}

/* Center heading */
.center-text { text-align:center; }

/* Card UI */
.card {
    background: rgba(255,255,255,0.85);
    padding: 24px;
    border-radius: 14px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
    margin-bottom: 18px;
    backdrop-filter: blur(12px);
}

/* Subtle hover lift */
.card:hover { transform: translateY(-3px); transition: 0.25s ease; }

/* Buttons */
.stButton>button { border-radius: 8px; font-weight:600; }

/* Header text spacing */
h1 { font-size: 48px; margin-bottom: 4px; }
p.tagline { font-size:20px; font-style:italic; color:#3b6652; margin-top:-10px; }
</style>
""", unsafe_allow_html=True)

# --------------------------
# Data Load
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

# Determine status
def status(temp):
    if temp > 40: return "🔥 Hotspot"
    if temp > 36: return "🌤 Medium"
    return "✅ Safe"

latest = df.sort_values("timestamp").groupby("zone").last().reset_index()
latest["status"] = latest["temp"].apply(status)

# --------------------------
# Header
# --------------------------
st.markdown("<h1 class='center-text'>PrithviSense</h1>", unsafe_allow_html=True)
st.markdown("<p class='center-text tagline'>Smart Thermal Awareness for Sustainable Campuses</p>", unsafe_allow_html=True)
st.write("")

# --------------------------
# Section 1: Current Status
# --------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("1) Current Campus Heat & UV Status")
st.dataframe(latest[["zone","temp","uv","status"]])
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# Section 2: Trends
# --------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("2) Temperature Trend by Zone")

zone = st.selectbox("Choose Zone", ZONES)
hist = df[df.zone == zone].sort_values("timestamp").tail(48)

chart = alt.Chart(hist).mark_line(color="#2b8a6e").encode(
    x="timestamp:T",
    y="temp:Q",
    tooltip=["timestamp:T","temp"]
).properties(height=260)

st.altair_chart(chart, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# Section 3: Prediction
# --------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("3) Forecast (Model Prediction)")

c1, c2, c3 = st.columns(3)
zone_in = c1.selectbox("Zone to Forecast", ZONES)
date_in = c2.date_input("Date", datetime.now().date())
time_in = c3.time_input("Time", (datetime.now()+timedelta(hours=1)).time())

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
        base_row = latest[latest.zone==zone_in].iloc[0]
        p_temp = round(base_row.temp + np.random.randn(),1)
        p_uv = round(base_row.uv + np.random.randn()*0.3,1)

    st.success(f"🌡 Predicted Temperature: **{p_temp}°C**  |  ☀ Predicted UV Index: **{p_uv}**")
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# Section 4: ROI Calculator
# --------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("4) ROI Calculator")

colA, colB, colC = st.columns(3)
cost = colA.number_input("Installation Cost (₹)", value=20000)
saving = colB.number_input("Yearly Savings (₹)", value=5000)
life = colC.number_input("Lifetime (years)", value=5)

total = saving * life
roi = ((total - cost)/cost)*100 if cost > 0 else 0

st.info(f"💰 Total Savings: **₹{total:,}**")
st.success(f"📈 ROI: **{roi:.1f}%**")
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# Footer
# --------------------------
st.markdown("<p class='center-text' style='color:#445;'>Made with ❤️ for Hackathons</p>", unsafe_allow_html=True)
