# streamlit_app.py
"""
Soft Green Eco-Theme Thermal Digital Twin (Streamlit)
- Uses campus_map.png / campus_map.jpg (fallback to demo image)
- Stylized 3D-like floating bubbles with glow for hotspots
- Prominent ROI calculator, Forecast charts, UV popup, Actionable Insights
- Tweak `zone_positions` to align bubbles precisely over your image
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

st.set_page_config(page_title="Chandigarh University — Thermal Digital Twin",
                   layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Files & constants
# ---------------------------
DATA_FILE = "thermal_data.csv"
MODEL_FILE = "thermal_model.joblib"
MAP_FILES = ["campus_map.png", "campus_map.jpg", "cu.jpeg"]  # check these names
FALLBACK_MAP_URL = "https://images.unsplash.com/photo-1501004318641-b39e6451bec6?auto=format&fit=crop&w=1600&q=60"

ZONES = [
    "Main Parking Lot", "Academic Block A", "Academic Block B",
    "Boys Hostel 1", "Boys Hostel 2", "Girls Hostel",
    "Sports Stadium", "Central Library", "Green Quad", "Food Court"
]

# ---------------------------
# CSS / style (soft green theme + bubble glow)
# ---------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg: #f2fbf2;
  --card: rgba(255,255,255,0.88);
  --muted:#6b7280;
  --accent:#2E7D32;
  --glass-radius:14px;
}
.stApp { font-family: 'Inter', sans-serif; background: var(--bg); }
.glass {
  background: var(--card);
  border-radius: var(--glass-radius);
  padding: 16px;
  box-shadow: 0 10px 30px rgba(20,40,20,0.06);
  backdrop-filter: blur(4px) saturate(120%);
  margin-bottom: 14px;
}

/* Map container */
.map-wrap { position: relative; width:100%; overflow:hidden; border-radius:10px; }
.map-img { width:100%; height:auto; display:block; filter: saturate(1.05) contrast(1.02); }

/* Bubble styles */
.bubble {
  position: absolute;
  transform: translate(-50%, -50%);
  border-radius:999px;
  color: white;
  font-weight:700;
  display:inline-flex;
  align-items:center;
  justify-content:center;
  text-align:center;
  box-shadow: 0 18px 40px rgba(46,125,50,0.11);
  transition: transform .18s ease, box-shadow .18s ease;
  cursor: pointer;
}
.bubble:hover { transform: translate(-50%, -58%) scale(1.06); box-shadow: 0 28px 60px rgba(0,0,0,0.2); }

/* Hotspot glow animation */
@keyframes pulseGlow {
  0% { box-shadow: 0 6px 18px rgba(255,107,0,0.18), 0 0 0 0 rgba(255,107,0,0.12); transform: translate(-50%,-50%) scale(1); }
  50% { box-shadow: 0 20px 48px rgba(255,107,0,0.22), 0 0 30px 10px rgba(255,107,0,0.08); transform: translate(-50%,-54%) scale(1.04); }
  100% { box-shadow: 0 6px 18px rgba(255,107,0,0.18), 0 0 0 0 rgba(255,107,0,0.12); transform: translate(-50%,-50%) scale(1); }
}
.hotspot { animation: pulseGlow 2s infinite; }

/* Legend / small helpers */
.legend { display:flex; gap:8px; flex-wrap:wrap; }
.legend .item { padding:8px; border-radius:8px; min-width:120px; text-align:center; font-size:13px; }

/* small screens */
@media (max-width:900px) {
  .map-wrap { max-height:420px; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Helpers (load data & model)
# ---------------------------
def load_dataset(path=DATA_FILE):
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df
    except Exception:
        return None

def load_model(path=MODEL_FILE):
    try:
        model = joblib.load(path)
        return model
    except Exception:
        return None

def temp_status(temp):
    if temp > 40: return "Hotspot"
    if temp > 36: return "Medium"
    return "Safe"

def status_color(temp):
    if temp > 45: return "#b21010"
    if temp > 40: return "#ff6b00"
    if temp > 36: return "#ffd54f"
    return "#2E7D32"

# ---------------------------
# Load dataset + model, or create demo
# ---------------------------
df = load_dataset()
pipeline = load_model()

if df is None:
    st.sidebar.warning("thermal_data.csv not found — generating demo data for UI.")
    now = pd.Timestamp.now().floor("H")
    rows = []
    for z in ZONES:
        for h in range(0, 24*3):  # last 3 days to keep lightweight
            ts = now - pd.Timedelta(hours=h)
            base = 26 + 6*math.sin((ts.dayofyear % 365)/365.0 * 2*math.pi)
            hour_cycle = 6 * math.sin((ts.hour-9)/24*2*math.pi)
            offset = 3 if "Parking" in z or "Stadium" in z else -2 if "Quad" in z or "Library" in z else 0
            temp = round(base + hour_cycle + offset + np.random.normal(0,0.6),1)
            uv = max(0, round(9 * math.sin((ts.hour-9)/24*2*math.pi) + np.random.normal(0,0.6),1))
            rows.append({"timestamp": ts, "zone": z, "temp": temp, "uv": uv})
    df = pd.DataFrame(rows)
else:
    st.sidebar.success("Dataset loaded")

if pipeline is None:
    st.sidebar.warning("Model file not found — forecast will use fallback heuristic.")
else:
    st.sidebar.success("Model loaded")

# ---------------------------
# Latest snapshot
# ---------------------------
latest = df.sort_values("timestamp").groupby("zone").last().reset_index()
latest["status"] = latest["temp"].apply(temp_status)
latest = latest.set_index("zone").reindex(ZONES).reset_index()

# ---------------------------
# Map image selection
# ---------------------------
map_src = None
for name in MAP_FILES:
    if os.path.exists(name):
        map_src = name
        break
if map_src is None:
    map_src = FALLBACK_MAP_URL

# ---------------------------
# Zone positions (x_pct, y_pct)
# - Edit these if bubble misaligns. Values are fractions 0..1
# ---------------------------
zone_positions = {
    "Main Parking Lot": (0.12, 0.78),
    "Academic Block A": (0.45, 0.28),
    "Academic Block B": (0.60, 0.30),
    "Boys Hostel 1": (0.85, 0.66),
    "Boys Hostel 2": (0.78, 0.72),
    "Girls Hostel": (0.74, 0.56),
    "Sports Stadium": (0.18, 0.44),
    "Central Library": (0.50, 0.48),
    "Green Quad": (0.55, 0.62),
    "Food Court": (0.35, 0.70)
}

# ---------------------------
# Header
# ---------------------------
left, right = st.columns([3,1])
with left:
    st.markdown("<h1 style='margin:4px 0 2px 0;'>🌿 Chandigarh University — Thermal Digital Twin</h1>", unsafe_allow_html=True)
    st.markdown("<div style='color:#2E7D32; font-weight:600;'>Sustainable • Predictive • Actionable</div>", unsafe_allow_html=True)
with right:
    theme = st.selectbox("Theme", ["Light","Dark"], index=0)
    if theme == "Dark":
        st.markdown("<script>document.querySelector('.stApp').classList.add('dark')</script>", unsafe_allow_html=True)
    else:
        st.markdown("<script>document.querySelector('.stApp').classList.remove('dark')</script>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------
# Top area: Live snapshot (left) + Map (right)
# ---------------------------
col1, col2 = st.columns([1.2,2], gap="large")

with col1:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Live Zones Snapshot")
    # grid of cards (3 per row)
    card_cols = st.columns([1,1,1])
    idx = 0
    for _, r in latest.iterrows():
        col = card_cols[idx % 3]
        color = status_color(r["temp"])
        col.markdown(f"""
            <div style='border-radius:10px; padding:12px; margin-bottom:8px; background:linear-gradient(180deg, rgba(255,255,255,0.95), rgba(245,255,240,0.9)); border-left:6px solid {color};'>
              <div style='font-weight:700'>{r['zone']}</div>
              <div style='font-size:20px; font-weight:800; color:{color};'>{r['temp']}°C</div>
              <div style='color:#666; font-size:13px;'>UV: {r['uv']} · {r['status']}</div>
            </div>
        """, unsafe_allow_html=True)
        idx += 1
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Campus Map — Soft Green Eco Theme")
    # Build HTML: container + image + bubbles (absolute)
    map_html = f"<div class='map-wrap'><img src='{map_src}' class='map-img'/>"
    # create bubble elements
    tmin = latest['temp'].min()
    for _, row in latest.iterrows():
        z = row['zone']
        x_pct, y_pct = zone_positions.get(z, (0.5,0.5))
        color = status_color(row['temp'])
        # bubble size (scale by temp difference)
        size = 46 + int((row['temp'] - tmin) * 2.8)
        tooltip = f"{z} · {row['temp']}°C · UV {row['uv']}"
        # hotspot class will add pulse
        hotspot_cls = "hotspot" if row['status']=="Hotspot" else ""
        map_html += f"""<div class='bubble {hotspot_cls}' style='left:{x_pct*100}%; top:{y_pct*100}%; width:{size}px; height:{size}px; line-height:{size}px; background:{color};' title="{tooltip}">{int(row['temp'])}°</div>"""
    map_html += "</div>"
    st.components.v1.html(map_html, height=520)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# UV info & legend expander
# ---------------------------
with st.expander("ℹ️ UV Index — Safety Guide (click to open)"):
    st.markdown("""
    **UV Index Guide**  
    - **0–2 (Low)** — Minimal risk. Safe for outdoor activity.  
    - **3–5 (Moderate)** — Take precautions: sunscreen, hats, sunglasses.  
    - **6–7 (High)** — Protection essential; avoid midday sun.  
    - **8–10 (Very High)** — Extra protection required; reduce time in sun.  
    - **11+ (Extreme)** — Avoid sun exposure; stay indoors if possible.
    """)
    st.markdown("""
    <div class='legend' style='margin-top:8px;'>
      <div class='item' style='background:#4caf50; color:white'>0-2 Low</div>
      <div class='item' style='background:#ffeb3b; color:#222'>3-5 Moderate</div>
      <div class='item' style='background:#ff9800; color:white'>6-7 High</div>
      <div class='item' style='background:#f44336; color:white'>8-10 Very High</div>
      <div class='item' style='background:#9c27b0; color:white'>11+ Extreme</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# Forecast (left) + ROI (right)
# ---------------------------
left, right = st.columns([2,1], gap="large")

with left:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Forecast (Model Prediction)")
    a1, a2, a3 = st.columns([1,1,1])
    zone_choice = a1.selectbox("Zone", ZONES, index=0)
    date_choice = a2.date_input("Date", datetime.now().date())
    time_choice = a3.time_input("Time", (datetime.now()+timedelta(hours=1)).time())
    if st.button("Run Forecast", key="f1"):
        ts = pd.to_datetime(f"{date_choice} {time_choice}")
        hour = ts.hour; dow = ts.dayofweek; month = ts.month
        Xpred = pd.DataFrame([{"hour":hour, "dayofweek":dow, "month":month, **{f"zone_{z}": int(z==zone_choice) for z in ZONES}}])
        if pipeline is not None:
            try:
                pred = pipeline.predict(Xpred)
                p_temp = float(pred[0][0]); p_uv = float(pred[0][1])
                st.success(f"Predicted Temp: {p_temp:.2f} °C · UV: {p_uv:.2f}")
            except Exception as e:
                st.error("Model prediction error — using fallback heuristic.")
                p_temp = latest.loc[latest['zone']==zone_choice,'temp'].values[0] + np.random.normal(0,1)
                p_uv = max(0, latest.loc[latest['zone']==zone_choice,'uv'].values[0] + np.random.normal(0,0.6))
        else:
            p_temp = latest.loc[latest['zone']==zone_choice,'temp'].values[0] + (np.random.rand()-0.45)*2
            p_uv = max(0, latest.loc[latest['zone']==zone_choice,'uv'].values[0] + (np.random.rand()-0.45)*1.2)
        # show chart of last 48 hours + forecast point
        hist = df[df['zone']==zone_choice].sort_values('timestamp').tail(72)
        forecast_pt = pd.DataFrame([{"timestamp":ts, "temp":p_temp, "type":"forecast"}])
        hist2 = hist.assign(type="history")[["timestamp","temp","type"]]
        chart_df = pd.concat([hist2, forecast_pt], ignore_index=True)
        base_line = alt.Chart(chart_df).mark_line(interpolate='monotone').encode(
            x=alt.X('timestamp:T', title='Time'),
            y=alt.Y('temp:Q', title='Temperature (°C)'),
            color=alt.Color('type:N', scale=alt.Scale(range=['#2E7D32','#ff6b00']), legend=None)
        ).properties(height=280)
        point = alt.Chart(chart_df[chart_df['type']=='forecast']).mark_point(size=120, color='#d32f2f', filled=True).encode(x='timestamp:T', y='temp:Q')
        st.altair_chart(base_line + point, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("💰 ROI Calculator (Major Feature)")
    cost = st.number_input("Upfront Cost (₹)", min_value=0, value=25000, step=500)
    energy_saved = st.number_input("Estimated yearly energy saving (₹)", min_value=0, value=7000, step=100)
    lifetime = st.number_input("Expected lifetime (years)", min_value=1, value=5, step=1)
    apply_subsidy = st.checkbox("Apply government subsidy (10%)", value=False)
    effective_cost = cost * (0.9 if apply_subsidy else 1.0)
    total_saved = energy_saved * lifetime
    roi = ((total_saved - effective_cost) / effective_cost * 100) if effective_cost>0 else None
    st.markdown(f"<div style='padding:10px; border-radius:10px; background:linear-gradient(90deg, rgba(46,125,50,0.06), rgba(79,195,247,0.03));'>"
                f"<div style='font-weight:700'>Total Saved (lifetime): ₹{int(total_saved):,}</div>"
                f"<div style='margin-top:6px; font-weight:700'>Estimated ROI: {'{:.1f}%'.format(roi) if roi is not None else '—'}</div>"
                f"</div>", unsafe_allow_html=True)
    # payback timeline chart
    years = list(range(0, lifetime+1))
    cum = [energy_saved * y for y in years]
    df_roi = pd.DataFrame({"year": years, "savings": cum, "cost": [effective_cost]*len(years)})
    area = alt.Chart(df_roi).mark_area(opacity=0.38).encode(x='year:O', y='savings:Q', color=alt.value('#2E7D32'))
    cost_line = alt.Chart(df_roi).mark_line(color='#d32f2f', strokeDash=[4,4]).encode(x='year:O', y='cost:Q')
    st.altair_chart(area + cost_line, use_container_width=True, theme='streamlit')
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Actionable insights (color-coded)
# ---------------------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.subheader("Actionable Insights")
hot = latest[latest['status']=="Hotspot"]
mid = latest[latest['status']=="Medium"]
if len(hot)>0:
    st.markdown(f"<div style='padding:12px; border-radius:8px; background:#ffe6e6; color:#8b0000; font-weight:700;'>IMMEDIATE ACTION REQUIRED — Hotspots: {', '.join(hot['zone'].tolist())}</div>", unsafe_allow_html=True)
elif len(mid)>0:
    st.markdown(f"<div style='padding:12px; border-radius:8px; background:#fff8e6; color:#b35a00; font-weight:700;'>Moderate risk — Monitor zones: {', '.join(mid['zone'].tolist())}</div>", unsafe_allow_html=True)
else:
    st.markdown("<div style='padding:12px; border-radius:8px; background:#eef9ee; color:#006400; font-weight:700;'>All zones safe — no immediate hotspots.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Legend + downloads
# ---------------------------
l, r = st.columns([3,1])
with l:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Temperature Ranges — Quick Guide")
    st.markdown("""
    <div style='display:flex; gap:12px; flex-wrap:wrap;'>
      <div style='background:#eef9ee; padding:8px; border-radius:8px; min-width:160px;'><b>Safe</b><br/>Temp ≤ 36°C</div>
      <div style='background:#fff7e0; padding:8px; border-radius:8px; min-width:160px;'><b>Medium</b><br/>36°C < Temp ≤ 40°C</div>
      <div style='background:#ffe6e6; padding:8px; border-radius:8px; min-width:160px;'><b>Hotspot</b><br/>Temp > 40°C</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with r:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Download latest snapshot")
    buf = io.StringIO()
    latest.to_csv(buf, index=False)
    st.download_button("Download CSV", data=buf.getvalue().encode(), file_name="latest_snapshot.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown("<div style='text-align:center; color:#6b7280; margin-top:8px;'>Made for Hackathon • Chandigarh University • Built with ♥</div>", unsafe_allow_html=True)
