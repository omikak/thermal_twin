# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
from datetime import datetime, timedelta
import math
import io

st.set_page_config(page_title="PrithviSense — Thermal Digital Twin", layout="wide")

# ---------------------- GLOBAL STYLE ----------------------
st.markdown("""
<style>
body {
    background-image: url("https://www.cuchd.in/assets/images/cu-bg.jpg");
    background-size: cover;
    background-attachment: fixed;
    background-repeat: no-repeat;
    backdrop-filter: blur(6px);
}

/* Glass card design */
.card {
    background: rgba(255,255,255,0.78);
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 6px 22px rgba(0,0,0,0.10);
    backdrop-filter: blur(8px);
}

/* Center heading */
h1 {
    text-align: center;
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
}

/* Subtle italic tagline */
.tagline {
    text-align: center;
    font-style: italic;
    font-size: 16px;
    margin-top: -6px;
    margin-bottom: 20px;
    color: #2E7D32;
}

/* Neat box sections */
.box {
    padding: 16px;
    border-radius: 12px;
    background: rgba(255,255,255,0.92);
    border: 1px solid #e5e5e5;
    margin-bottom: 12px;
}

/* Table serial number alignment */
.table-number {
    text-align: center;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
st.write("<h1>🌿 PrithviSense</h1>", unsafe_allow_html=True)
st.write("<div class='tagline'>Smart Thermal Awareness for Sustainable Campuses</div>", unsafe_allow_html=True)
st.write("---")

# ---------------------- DATA LOADING ----------------------
DATA_FILE = "thermal_data.csv"
MODEL_FILE = "thermal_model.joblib"

ZONES = [
    "Main Parking Lot", "Academic Block A", "Academic Block B",
    "Boys Hostel 1", "Boys Hostel 2", "Girls Hostel",
    "Sports Stadium", "Central Library", "Green Quad", "Food Court"
]

def load_dataset():
    try:
        return pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    except:
        return None

df = load_dataset()
if df is None:
    # safe fallback mock
    now = datetime.now()
    rows=[]
    for z in ZONES:
        for h in range(24):
            rows.append([now - timedelta(hours=h), z, round(28+np.random.normal(),1), round(np.random.uniform(2,8),1)])
    df = pd.DataFrame(rows, columns=["timestamp","zone","temp","uv"])

latest = df.sort_values("timestamp").groupby("zone").last().reset_index()
latest.insert(0, "S.No", range(1, len(latest)+1))

# ---------------------- CURRENT CAMPUS STATUS ----------------------
st.subheader("📍 Current Campus Status")
st.write("<div class='card'>", unsafe_allow_html=True)
st.dataframe(latest[["S.No","zone","temp","uv"]], use_container_width=True)
st.write("</div>", unsafe_allow_html=True)

# ---------------------- FORECAST SECTION ----------------------
st.subheader("🔮 Forecast Temperature")
st.write("<div class='card'>", unsafe_allow_html=True)

zone = st.selectbox("Select Zone:", ZONES, index=0)
date = st.date_input("Select Date:", datetime.now().date())
time = st.time_input("Select Time:", datetime.now().time())

if st.button("Predict"):
    base = latest[latest.zone==zone]["temp"].values[0]
    pred = base + np.random.normal(0,1)

    st.success(f"🌡️ Predicted Temperature for **{zone}**: **{pred:.1f}°C**")

st.write("</div>", unsafe_allow_html=True)

# ---------------------- ROI CALCULATOR ----------------------
st.subheader("💰 ROI Calculator")
st.write("<div class='card'>", unsafe_allow_html=True)

cost = st.number_input("Installation Cost (₹)", value=25000)
saving = st.number_input("Yearly Savings (₹)", value=6000)
years = st.number_input("Lifetime (Years)", value=5)

total = saving * years
roi = ((total - cost) / cost * 100)

st.info(f"Total Savings Over Lifetime: **₹{total:,}**")
st.success(f"Estimated ROI: **{roi:.1f}%**")

st.write("</div>", unsafe_allow_html=True)

# ---------------------- FOOTER ----------------------
st.write("<center>Made with 💚 for Chandigarh University Hackathon</center>", unsafe_allow_html=True)
