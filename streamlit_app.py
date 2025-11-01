import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
from datetime import datetime, timedelta
import math
import io

st.set_page_config(
    page_title="Chandigarh University — Thermal Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------- Styles ----------------------
st.markdown(
    """
    <style>
    :root{
      --card:#ffffffcc;
      --muted:#6b7280;
      --accent:#2E7D32;
    }

    /* Background Image */
    html, body, [class*="main"] {
      background: url('cu_bg.jpg') no-repeat center center fixed !important;
      background-size: cover !important;
    }

    /* Blur + Soft Light Overlay */
    body:before {
      content:"";
      position: fixed;
      top:0; left:0;
      width:100%; height:100%;
      backdrop-filter: blur(6px) brightness(0.93);
      z-index:-1;
    }

    /* Glass Cards */
    .glass {
      background: var(--card);
      border-radius: 14px;
      padding: 18px;
      box-shadow: 0 8px 24px rgba(9,30,66,0.15);
      backdrop-filter: blur(10px) saturate(130%);
    }

    .zone-bubble {
      border-radius: 999px;
      padding: 10px 14px;
      font-weight:700;
      color: white;
      display:inline-block;
      cursor:pointer;
      transition: 0.25s;
    }
    .zone-bubble:hover {
      transform: translateY(-6px);
      box-shadow: 0 18px 36px rgba(0,0,0,0.18);
    }

    .small { font-size:0.9rem; color:#4b4b4b; font-style:italic; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------- Config ----------------------
DATA_FILE = "thermal_data.csv"
MODEL_FILE = "thermal_model.joblib"

ZONES = [
    "Main Parking Lot", "Academic Block A", "Academic Block B",
    "Boys Hostel 1", "Boys Hostel 2", "Girls Hostel",
    "Sports Stadium", "Central Library", "Green Quad", "Food Court"
]

def load_dataset(path=DATA_FILE):
    try:
        return pd.read_csv(path, parse_dates=["timestamp"])
    except:
        return None

def load_model(path=MODEL_FILE):
    try:
        return joblib.load(path)
    except:
        return None

def temp_status(t):
    if t > 40: return "Hotspot"
    if t > 36: return "Moderate"
    return "Safe"

def status_color(t):
    if t > 45: return "#b21010"
    if t > 40: return "#ff6b00"
    if t > 36: return "#ffd54f"
    return "#4caf50"

# ---------------------- Data ----------------------
st.sidebar.title("Chandigarh University")
st.sidebar.markdown("*Thermal Digital Twin — Live Demo*")

df = load_dataset()
pipeline = load_model()

if df is None:
    st.sidebar.warning("No dataset found — using temporary demo data…")
    now = pd.Timestamp.now().floor("H")
    rows=[]
    for z in ZONES:
        for h in range(48):
            ts = now - pd.Timedelta(hours=h)
            temp = round(28 + 6*np.sin((ts.hour-9)/24*2*np.pi) + np.random.normal(0,0.5),1)
            uv = round(max(0,6*np.sin((ts.hour)/24*2*np.pi)+np.random.normal(0,0.3)),1)
            rows.append((ts,z,temp,uv))
    df = pd.DataFrame(rows, columns=["timestamp","zone","temp","uv"])

latest = df.sort_values("timestamp").groupby("zone").last().reset_index()
latest["status"] = latest["temp"].apply(temp_status)
latest = latest.set_index("zone").reindex(ZONES).reset_index()

# ---------------------- Heading ----------------------
st.markdown(
    """
    <div style="text-align:center;">
        <h1 style="font-family:Poppins; font-weight:700; margin-bottom:2px;">
            Chandigarh University — Thermal Digital Twin
        </h1>
        <div class="small">Live Heat Awareness • Smart Movement Guidance • Eco Efficiency</div>
    </div>
    """, unsafe_allow_html=True
)
st.write("")

# ---------------------- Current Campus Status ----------------------
col1, col2 = st.columns([2,3], gap="large")

with col1:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Current Campus Status")

    cols = st.columns(3)
    for idx, row in latest.iterrows():
        box = cols[idx % 3]
        color = status_color(row["temp"])
        box.markdown(
            f"""
            <div style='background:#ffffffcc; border-radius:12px; padding:12px;'>
                <div style='font-size:12px; color:#666'>#{idx+1} — {row['zone']}</div>
                <div style='font-size:22px; font-weight:700; color:{color}'>{row['temp']}°C</div>
                <div style='font-size:13px; color:#444'>UV Index: {row['uv']}</div>
                <div style='margin-top:6px; font-size:13px; color:#666'>Status: {row['status']}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Campus Map (Zones)")
    bubble_html = "<div style='display:flex; flex-wrap:wrap; gap:10px; justify-content:center;'>"
    for _, r in latest.iterrows():
        bubble_html += f"<div class='zone-bubble' style='background:{status_color(r['temp'])};'>{r['zone'].split()[0]}<br><span style='font-size:11px'>{r['temp']}°C</span></div>"
    bubble_html += "</div>"
    st.components.v1.html(bubble_html, height=160)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- Zone Detail + Forecast ----------------------
st.write("")
st.markdown("<div class='glass'>", unsafe_allow_html=True)

selected_zone = st.selectbox("Select Zone", ZONES)
zone_row = latest[latest["zone"] == selected_zone].iloc[0]
st.metric(f"{selected_zone} — Current Temperature", f"{zone_row['temp']} °C")

hist = df[df.zone == selected_zone].sort_values("timestamp").tail(48)
st.altair_chart(
    alt.Chart(hist).mark_line(point=True).encode(
        x="timestamp:T", y="temp:Q"
    ).properties(height=200),
    use_container_width=True
)

st.subheader("Forecast Temperature")
with st.form("forecast"):
    date = st.date_input("Date", datetime.now().date())
    time = st.time_input("Time", datetime.now().time())
    run = st.form_submit_button("Run Forecast")

    if run:
        if pipeline:
            ts = pd.to_datetime(f"{date} {time}")
            X = pd.DataFrame([{"hour":ts.hour, "dayofweek":ts.dayofweek, "month":ts.month, **{f"zone_{z}":int(z==selected_zone) for z in ZONES}}])
            temp, uv = pipeline.predict(X)[0]
        else:
            temp = zone_row["temp"] + (np.random.rand()-.4)*2
            uv = zone_row["uv"] + (np.random.rand()-.4)*1.2

        st.markdown(
            f"""
            <div style="border:1px solid #ccc; border-radius:10px; padding:14px;">
                <strong>Predicted Temperature:</strong> {temp:.2f} °C<br>
                <strong>Predicted UV Index:</strong> {uv:.2f}
            </div>
            """, unsafe_allow_html=True
        )
        st.success(f"Forecast Ready → {selected_zone}: {temp:.2f}°C, UV {uv:.2f}")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- ROI Calculator ----------------------
st.write("")
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.subheader("ROI Calculator")

cost = st.number_input("Upfront Cost (₹)", value=25000)
saving = st.number_input("Yearly Savings (₹)", value=6000)
years = st.number_input("Lifetime (years)", value=5, min_value=1)
if st.checkbox("Apply 10% Subsidy"):
    cost *= 0.9

total = saving * years
roi = ((total-cost)/cost)*100 if cost else 0

st.metric("Total Lifetime Savings", f"₹{int(total):,}")
st.metric("ROI %", f"{roi:.1f}%")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- Footer ----------------------
st.markdown("<div style='text-align:center; color:#444; margin-top:25px;'>Built with ❤ at Chandigarh University Hackathon</div>", unsafe_allow_html=True)
