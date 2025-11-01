# streamlit_app.py (UI Clean + Proper Responsive Layout)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
from datetime import datetime, timedelta
import math
import io
import os

st.set_page_config(page_title="Chandigarh University — Thermal Digital Twin",
                   layout="wide")

# ---------------------------
# Load Data / Model
# ---------------------------
DATA_FILE = "thermal_data.csv"
MODEL_FILE = "thermal_model.joblib"

ZONES = [
    "Main Parking Lot", "Academic Block A", "Academic Block B",
    "Boys Hostel 1", "Boys Hostel 2", "Girls Hostel",
    "Sports Stadium", "Central Library", "Green Quad", "Food Court"
]

def load_data():
    try:
        return pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    except:
        return None

def load_model():
    try:
        return joblib.load(MODEL_FILE)
    except:
        return None

df = load_data()
pipeline = load_model()

if df is None:
    st.warning("Data file missing. Using generated demo data.")
    now = pd.Timestamp.now().floor("H")
    rows = []
    for z in ZONES:
        for h in range(48):
            ts = now - pd.Timedelta(hours=h)
            temp = 28 + 6*np.sin(h/24*np.pi) + np.random.normal(0,1)
            uv = max(0, round(8*np.sin(h/24*np.pi) + np.random.normal(0,1),1))
            rows.append({"timestamp": ts, "zone": z, "temp": temp, "uv": uv})
    df = pd.DataFrame(rows)

latest = df.sort_values("timestamp").groupby("zone").last().reset_index()

def classify(temp):
    if temp > 40: return "Hotspot"
    if temp > 36: return "Medium"
    return "Safe"

latest["status"] = latest["temp"].apply(classify)

# ---------------------------
# UI HEADER
# ---------------------------
st.markdown("""
<style>
body, .stApp { font-family: 'Inter', sans-serif; }
.card {
    padding: 14px; 
    border-radius: 10px;
    background: #ffffffdd;
    border: 1px solid #e5e5e5;
}
.section-title { 
    font-size: 22px; 
    font-weight: 700; 
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🌿 Chandigarh University — Thermal Digital Twin")
st.caption("Real-time Monitoring • Climate Safety • Predictive Insights")

# ---------------------------
# LIVE SNAPSHOT
# ---------------------------
st.markdown("### 🔥 Live Thermal Snapshot")
cols = st.columns(3)

for i, row in latest.iterrows():
    color = "#4caf50" if row["status"]=="Safe" else "#ff9800" if row["status"]=="Medium" else "#d32f2f"
    with cols[i % 3]:
        st.markdown(f"""
        <div class='card' style='border-left: 6px solid {color};'>
        <b>{row['zone']}</b><br>
        <span style="font-size:22px; font-weight:700; color:{color}">{row['temp']:.1f}°C</span><br>
        UV Index: {row['uv']}<br>
        Status: <b>{row['status']}</b>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ---------------------------
# FORECAST
# ---------------------------
st.markdown("### 📈 Forecast (Model Prediction)")
fc_col1, fc_col2 = st.columns([3,2])

with fc_col1:
    zone = st.selectbox("Select Zone", ZONES)
    date = st.date_input("Select Date", datetime.now().date())
    time = st.time_input("Select Time", (datetime.now()+timedelta(hours=1)).time())
    if st.button("Run Forecast"):
        timestamp = pd.to_datetime(f"{date} {time}")
        hour = timestamp.hour
        dow = timestamp.dayofweek
        month = timestamp.month

        if pipeline:
            X = pd.DataFrame([{"hour":hour,"dayofweek":dow,"month":month, **{f"zone_{z}":int(z==zone) for z in ZONES}}])
            pred = pipeline.predict(X)
            predicted_temp = float(pred[0][0])
        else:
            predicted_temp = latest[latest["zone"]==zone]["temp"].values[0] + np.random.normal(0,1)

        st.success(f"Predicted Temperature: **{predicted_temp:.2f}°C**")

with fc_col2:
    st.info("""
    **Understanding the Prediction**
    - Based on historical thermal trends
    - Zone characteristics are considered
    - Best suited for planning safe movement and outdoor events
    """)

st.markdown("---")

# ---------------------------
# ROI CALCULATOR
# ---------------------------
st.markdown("### 💰 ROI Calculator (Major Feature)")

cost = st.number_input("Installation Cost (₹)", min_value=0, value=25000)
saving = st.number_input("Estimated Annual Energy Saving (₹)", min_value=0, value=6000)
years = st.number_input("System Lifetime (Years)", min_value=1, value=5)

total = saving * years
roi = ((total - cost)/cost * 100) if cost else None

st.metric("Total Saving Over Lifetime", f"₹{int(total):,}")
st.metric("Estimated ROI", f"{roi:.1f}%")

st.markdown("---")

# ---------------------------
# ACTIONABLE INSIGHTS
# ---------------------------
st.markdown("### 🎯 Actionable Insights")
hotspots = latest[latest["status"]=="Hotspot"]["zone"].tolist()

if hotspots:
    st.error(f"⚠️ Immediate action required in: **{', '.join(hotspots)}**")
else:
    st.success("✅ All zones are currently safe")

st.markdown("---")

st.caption("Made with ❤️ at Chandigarh University")
