# streamlit_app.py  — PRITHVISENSE (Clean White Theme)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
from datetime import datetime, timedelta
import math
import io

st.set_page_config(page_title="PrithviSense", layout="wide",)

# ---------- STYLE ----------
st.markdown("""
<style>
body, .stApp {
    background-color: #F8FAF7;
    font-family: 'Inter', sans-serif;
}
h1 {
    font-weight: 800;
    color: #2E7D32;
}
h2, h3 {
    font-weight: 600;
    color: #2E7D32;
}
.card {
    background: #FFFFFF;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #EAEAEA;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.05);
}
.statbox {
    background: #F3F9F3;
    padding: 14px;
    border-radius: 10px;
    text-align: center;
    border-left: 5px solid #4CAF50;
}
.zonebox {
    padding: 10px;
    border-radius: 8px;
    border-left: 5px solid #4CAF50;
    background: #FFFFFF;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

# ---------- DATA ----------
ZONES = ["Main Parking Lot","Academic Block A","Academic Block B",
         "Boys Hostel 1","Boys Hostel 2","Girls Hostel",
         "Sports Stadium","Central Library","Green Quad","Food Court"]

def load_data():
    try:
        df = pd.read_csv("thermal_data.csv", parse_dates=["timestamp"])
        return df
    except:
        return None

def load_model():
    try:
        return joblib.load("thermal_model.joblib")
    except:
        return None

df = load_data()
model = load_model()

# Fallback sample data
if df is None:
    now = pd.Timestamp.now().floor("H")
    rows=[]
    for z in ZONES:
        for h in range(48):
            ts = now - pd.Timedelta(hours=h)
            temp = 29 + np.random.normal(0,1.8)
            uv = max(0, round(9*np.sin((ts.hour-9)/24*2*np.pi)+np.random.normal(0,0.4),1))
            rows.append({"timestamp":ts,"zone":z,"temp":round(temp,1),"uv":uv})
    df = pd.DataFrame(rows)

latest = df.sort_values("timestamp").groupby("zone").last().reset_index()

def status(t):
    return "Hotspot" if t>40 else ("Warm" if t>36 else "Comfortable")

latest["Status"] = latest["temp"].apply(status)

# ---------- HEADER ----------
st.markdown("<h1>PrithviSense</h1>", unsafe_allow_html=True)
st.markdown("### *Smart Thermal Awareness for Sustainable Campuses*")
st.write("")

# ---------- LIVE SNAPSHOT ----------
left, right = st.columns([1.1,1.9], gap="large")

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Current Zone Conditions")
    
    for zone in ZONES:
        row = latest[latest.zone==zone].iloc[0]
        st.markdown(
            f"<div class='zonebox'><b>{zone}</b><br>Temp: {row.temp}°C · UV: {row.uv} · {row.Status}</div>",
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- SIMPLE CLEAN MAP ----------
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Campus Overview (Simple Layout Map)")
    st.image("cu.jpeg", caption="Chandigarh University Campus Map", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- FORECAST ----------
st.write("")
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Forecast Temperature")

z = st.selectbox("Select Zone", ZONES)
d = st.date_input("Date", datetime.now().date())
t = st.time_input("Time", datetime.now().time())

if st.button("Predict"):
    ts = pd.to_datetime(f"{d} {t}")
    hour = ts.hour; dow = ts.dayofweek; month = ts.month

    if model is not None:
        X = pd.DataFrame([{ "hour":hour,"dayofweek":dow,"month":month, **{f"zone_{zn}":int(zn==z) for zn in ZONES}}])
        p = model.predict(X)[0][0]
    else:
        p = latest[latest.zone==z].temp.values[0] + np.random.normal(0,1)

    st.success(f"Estimated Temperature: {p:.1f}°C")

    hist = df[df.zone==z].tail(48)
    chart = alt.Chart(hist).mark_line().encode(x="timestamp:T", y="temp:Q")
    st.altair_chart(chart, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------- ROI CALCULATOR ----------
st.write("")
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("ROI Calculator")

cost = st.number_input("Installation Cost (₹)", value=25000)
saving = st.number_input("Estimated Yearly Saving (₹)", value=7000)
years = st.slider("Years", 1, 10, 5)

total = saving*years
roi = ((total-cost)/cost)*100

st.write(f"**Total Savings:** ₹{total:,}")
st.write(f"**ROI:** {roi:.1f}%")

st.markdown("</div>", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.write(" ")
st.markdown("<center><i>Made with care for Chandigarh University 💚</i></center>", unsafe_allow_html=True)
