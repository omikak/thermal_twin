# streamlit_app.py
"""
PRITHVI SENSE — Tech Futuristic UI (Dark / Neon)
Place this file in repo root. Add thermal_data.csv and thermal_model.joblib (optional).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
from datetime import datetime, timedelta
import math
import io
import os

# -----------------------
# Page config
# -----------------------
st.set_page_config(
    page_title="PRITHVI SENSE — Thermal Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# Files & Zones
# -----------------------
DATA_FILE = "thermal_data.csv"
MODEL_FILE = "thermal_model.joblib"

ZONES = [
    "Main Parking Lot", "Academic Block A", "Academic Block B",
    "Boys Hostel 1", "Boys Hostel 2", "Girls Hostel",
    "Sports Stadium", "Central Library", "Green Quad", "Food Court"
]

# -----------------------
# Styling (Dark + Neon)
# -----------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    :root {
        --bg: #0b0f12;
        --card: rgba(12,16,20,0.85);
        --muted: #9aa6a6;
        --neon: #00E5FF; /* cyan */
        --neon-2: #7CFF6A; /* green */
        --accent: #00BFA5;
        --danger: #ff4d6d;
        --glass-radius: 12px;
    }
    .stApp { background: linear-gradient(180deg, #050607 0%, #0b0f12 100%); color: #e6f7f7; font-family: Inter, sans-serif; }
    .card {
        background: var(--card);
        padding:16px;
        border-radius:var(--glass-radius);
        box-shadow: 0 8px 30px rgba(0,0,0,0.6);
        border: 1px solid rgba(255,255,255,0.03);
        margin-bottom:14px;
    }
    h1 { color: var(--neon-2); margin:0; font-weight:800; letter-spacing:0.6px; }
    h2 { color: #bdebd0; margin:6px 0 10px 0; font-weight:700; }
    .muted { color: var(--muted); font-size:13px; }
    .neon-btn { background: linear-gradient(90deg, rgba(0,229,255,0.14), rgba(124,255,106,0.06)); padding:8px 12px; border-radius:8px; color:var(--neon); font-weight:700; border:1px solid rgba(0,229,255,0.12); }
    .tiny { font-size:12px; color:var(--muted); }

    /* abstract campus container */
    .map-outer { position: relative; width:100%; padding-top:54%; overflow:hidden; border-radius:10px; background: linear-gradient(180deg, rgba(0,20,20,0.2), rgba(0,0,0,0.2)); border:1px solid rgba(255,255,255,0.02); }
    .map-inner { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; }
    .grid { width:94%; height:86%; display:grid; grid-template-columns: repeat(6, 1fr); grid-template-rows: repeat(4, 1fr); gap:10px; align-items:center; justify-items:center; }
    .tile { width:100%; height:100%; border-radius:8px; background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); display:flex; align-items:center; justify-content:center; color:#cfeeee; font-weight:700; font-size:0.88rem; text-align:center; padding:8px; box-shadow: inset 0 0 12px rgba(0,0,0,0.2); border:1px solid rgba(255,255,255,0.02); }
    .tile-muted { opacity:0.55; color:#7eaead; font-weight:600; }
    .bubble {
        position:absolute; transform: translate(-50%, -50%); border-radius:999px; color:black; font-weight:800; display:inline-flex; align-items:center; justify-content:center; text-align:center; box-shadow: 0 12px 30px rgba(0,0,0,0.6);
        pointer-events:auto;
    }
    .bubble-label { background: rgba(0,0,0,0.6); padding:6px 10px; border-radius:8px; color:#e6f7f7; font-weight:700; font-size:12px; backdrop-filter: blur(4px); }
    @media (max-width:900px) {
        .grid { grid-template-columns: repeat(3,1fr); grid-template-rows: repeat(8,1fr); gap:8px; }
        h1 { font-size:20px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Helpers: load dataset & model
# -----------------------
def load_dataset(path=DATA_FILE):
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df
    except Exception:
        return None

def load_model(path=MODEL_FILE):
    try:
        m = joblib.load(path)
        return m
    except Exception:
        return None

def status(temp):
    if temp > 40: return "Hotspot"
    if temp > 36: return "Medium"
    return "Safe"

def color(temp):
    if temp > 45: return "#ff615a"
    if temp > 40: return "#ff9b3a"
    if temp > 36: return "#ffd54f"
    return "#7CFF6A"

# -----------------------
# Load data & model (fallback demo)
# -----------------------
df = load_dataset()
model = load_model()

if df is None:
    st.sidebar.warning("thermal_data.csv not found — demo data in use")
    now = pd.Timestamp.now().floor("H")
    rows = []
    for z in ZONES:
        for h in range(0, 24*2):  # 48 hours
            ts = now - pd.Timedelta(hours=h)
            base = 26 + 6*math.sin((ts.dayofyear % 365)/365.0 * 2*math.pi)
            hour_cycle = 6 * math.sin((ts.hour-9)/24*2*math.pi)
            offset = 3 if "Parking" in z or "Stadium" in z else -1 if "Quad" in z or "Library" in z else 0
            temp = round(base + hour_cycle + offset + np.random.normal(0,0.6),1)
            uv = max(0, round(9 * math.sin((ts.hour-9)/24*2*math.pi) + np.random.normal(0,0.5),1))
            rows.append({"timestamp": ts, "zone": z, "temp": temp, "uv": uv})
    df = pd.DataFrame(rows)
else:
    st.sidebar.success("Dataset loaded")

if model is None:
    st.sidebar.info("Model not found — using heuristic fallback for forecasts")
else:
    st.sidebar.success("Model loaded")

# -----------------------
# Latest snapshot
# -----------------------
latest = df.sort_values("timestamp").groupby("zone").last().reset_index()
latest["status"] = latest["temp"].apply(status)
latest = latest.set_index("zone").reindex(ZONES).reset_index()

# -----------------------
# Header: Title + one-line
# -----------------------
col1, col2 = st.columns([3,1])
with col1:
    st.markdown("<h1>PRITHVI SENSE</h1>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Intelligent Campus Thermal Twin & Sustainability Advisor</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div style='text-align:right'><button class='neon-btn'>Live</button></div>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------
# Top area: Live snapshot (left) + Abstract tech map (right)
# -----------------------
left, right = st.columns([1.2, 2], gap="large")

# LEFT: Live cards
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Live Zones Snapshot")
    # two-column stacked cards for neatness
    ccols = st.columns(2)
    for i, r in latest.iterrows():
        c = ccols[i % 2]
        colr = color(r["temp"])
        c.markdown(f"""
            <div style='padding:12px; border-radius:8px; margin-bottom:8px; background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-left:4px solid {colr};'>
              <div style='font-weight:800; color:#e6fff8'>{r['zone']}</div>
              <div style='font-size:20px; font-weight:900; color:{colr};'>{r['temp']}°C</div>
              <div class='tiny'>{r['status']} · UV {r['uv']}</div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# RIGHT: Abstract futuristic tech-grid map with neon bubbles
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Campus — Tech Map (Abstract)")
    # build grid HTML
    grid_slots = [
        "Gate", "Main Parking Lot", "Academic Block A", "Academic Block B", "Central Library", "Quad",
        "Green Area", "Sports Stadium", "Food Court", "Cafeteria", "Admin Block", "Solar Field",
        "Boys Hostel 1", "Boys Hostel 2", "Girls Hostel", "Research Labs", "Workshop", "Open Field",
        "Clinic", "Garden", "Auditorium", "Parking-2", "Lab Annex", "Pond"
    ]
    map_html = "<div class='map-outer'><div class='map-inner'><div class='grid'>"
    for s in grid_slots:
        if s in ZONES:
            map_html += f"<div class='tile' id='{s}' style='border:1px solid rgba(0,255,255,0.03);'>{s}</div>"
        else:
            map_html += f"<div class='tile tile-muted'>{s}</div>"
    map_html += "</div></div></div>"
    st.components.v1.html(map_html, height=420)

    # overlay neon bubbles (compute positions by grid index)
    zone_index = {
        "Main Parking Lot": 1, "Academic Block A": 2, "Academic Block B": 3,
        "Central Library": 4, "Green Quad": 5, "Sports Stadium": 7,
        "Food Court": 8, "Boys Hostel 1": 12, "Boys Hostel 2": 13, "Girls Hostel": 14
    }
    def center_pct(idx, cols=6, rows=4):
        col = idx % cols
        row = idx // cols
        x = (col + 0.5) / cols * 100
        y = (row + 0.5) / rows * 100
        return x, y

    overlay = "<div style='position:relative; margin-top:-460px; pointer-events:none;'>"
    tmin = latest['temp'].min()
    for _, r in latest.iterrows():
        z = r['zone']
        if z in zone_index:
            idx = zone_index[z]
            x, y = center_pct(idx)
            size = 36 + int(max(0, (r['temp'] - tmin)) * 3)
            bg = color(r['temp'])
            text_color = "#091011"
            overlay += f"<div class='bubble' style='left:{x}%; top:{y}%; width:{size}px; height:{size}px; line-height:{size}px; background:{bg}; pointer-events:auto;'><div class='bubble-label' title='{z} · {r['temp']}°C · UV {r['uv']}' style='color:{text_color};'>{int(r['temp'])}°</div></div>"
    overlay += "</div>"
    st.components.v1.html(overlay, height=0)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# Forecast + ROI (side-by-side)
# -----------------------
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
fcol, rcol = st.columns([2,1], gap="large")

with fcol:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Forecast — Predictive Temperature")
    fz, fd, ft = st.columns([1,1,1])
    zone_choice = fz.selectbox("Zone", ZONES, index=0)
    date_choice = fd.date_input("Date", datetime.now().date())
    time_choice = ft.time_input("Time", (datetime.now()+timedelta(hours=1)).time())
    if st.button("Run Forecast", key="runF1"):
        ts = pd.to_datetime(f"{date_choice} {time_choice}")
        hour = ts.hour; dow = ts.dayofweek; month = ts.month
        X = pd.DataFrame([{"hour":hour,"dayofweek":dow,"month":month, **{f"zone_{z}": int(z==zone_choice) for z in ZONES}}])
        if model is not None:
            try:
                pred = model.predict(X)
                p_temp = float(pred[0][0]); p_uv = float(pred[0][1])
            except Exception:
                p_temp = latest.loc[latest.zone==zone_choice,'temp'].values[0] + np.random.normal(0,1)
                p_uv = max(0, latest.loc[latest.zone==zone_choice,'uv'].values[0] + np.random.normal(0,0.6))
        else:
            p_temp = latest.loc[latest.zone==zone_choice,'temp'].values[0] + (np.random.rand()-0.45)*1.8
            p_uv = max(0, latest.loc[latest.zone==zone_choice,'uv'].values[0] + (np.random.rand()-0.45)*0.9)
        st.success(f"Predicted Temp: {p_temp:.2f}°C · Predicted UV: {p_uv:.2f}")

        # show interactive chart (history + forecast point)
        hist = df[df.zone==zone_choice].sort_values("timestamp").tail(72)
        fpt = pd.DataFrame([{"timestamp":ts,"temp":p_temp,"type":"forecast"}])
        hist2 = hist.assign(type="history")[["timestamp","temp","type"]]
        chart_df = pd.concat([hist2, fpt], ignore_index=True)
        line = alt.Chart(chart_df).mark_line(interpolate='monotone').encode(
            x=alt.X('timestamp:T', title='Time'),
            y=alt.Y('temp:Q', title='Temperature (°C)'),
            color=alt.Color('type:N', scale=alt.Scale(range=['#7CFF6A', '#ff9b3a']), legend=None)
        ).properties(height=300)
        fp = alt.Chart(chart_df[chart_df.type=='forecast']).mark_point(size=150, color='#ff4d6d').encode(x='timestamp:T', y='temp:Q')
        st.altair_chart(line + fp, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with rcol:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("💰 ROI Calculator")
    cost = st.number_input("Upfront cost (₹)", min_value=0, value=30000, step=500)
    annual = st.number_input("Estimated yearly energy saving (₹)", min_value=0, value=7000, step=100)
    lifetime = st.number_input("Expected lifetime (yrs)", min_value=1, value=5, step=1)
    subsidy = st.checkbox("Apply subsidy (10%)", value=False)
    effective_cost = cost * (0.9 if subsidy else 1.0)
    total_saved = annual * lifetime
    roi = ((total_saved - effective_cost) / effective_cost * 100) if effective_cost > 0 else None

    st.markdown(f"<div style='padding:10px; border-radius:10px; background:linear-gradient(90deg, rgba(0,255,220,0.04), rgba(124,255,106,0.02));'><div style='font-weight:800; font-size:18px;'>Lifetime Savings: ₹{int(total_saved):,}</div><div style='margin-top:8px; font-weight:900; font-size:20px; color:#7CFF6A;'>ROI: {'{:.1f}%'.format(roi) if roi is not None else '—'}</div></div>", unsafe_allow_html=True)

    # payback line
    years = list(range(0, lifetime+1))
    cum = [annual * y for y in years]
    dfroi = pd.DataFrame({"year": years, "savings": cum, "cost":[effective_cost]*len(years)})
    area = alt.Chart(dfroi).mark_area(opacity=0.28, color="#00E5FF").encode(x='year:O', y='savings:Q')
    cost_line = alt.Chart(dfroi).mark_line(color='#ff4d6d', strokeDash=[4,4]).encode(x='year:O', y='cost:Q')
    st.altair_chart(area + cost_line, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# Actionable insights & UV popup
# -----------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Actionable Insights")

hot = latest[latest["temp"]>40]
mid = latest[(latest["temp"]>36) & (latest["temp"]<=40)]

if len(hot)>0:
    st.error(f"IMMEDIATE ACTION REQUIRED — Hotspots: {', '.join(hot['zone'].tolist())}")
elif len(mid)>0:
    st.warning(f"Moderate risk zones: {', '.join(mid['zone'].tolist())}")
else:
    st.success("All zones safe — no immediate hotspots")

with st.expander("UV Index Guide (safety)"):
    st.write("""
    - 0–2: Low — safe for outdoor activity.
    - 3–5: Moderate — sunscreen, hat, sunglasses recommended.
    - 6–7: High — protection required; reduce exposure.
    - 8–10: Very High — avoid midday sun; seek shade.
    - 11+: Extreme — stay indoors if possible.
    """)
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# Footer + snapshot download
# -----------------------
st.markdown("<div style='display:flex; justify-content:space-between; align-items:center; gap:12px; margin-top:12px;'>", unsafe_allow_html=True)
st.markdown("<div style='color:#9fbfb7; font-weight:700;'>PRITHVI SENSE • Chandigarh University • Demo</div>", unsafe_allow_html=True)
buf = io.StringIO(); latest.to_csv(buf, index=False)
st.download_button("Download snapshot (CSV)", data=buf.getvalue().encode(), file_name="latest_snapshot.csv", mime="text/csv")
st.markdown("</div>", unsafe_allow_html=True)
