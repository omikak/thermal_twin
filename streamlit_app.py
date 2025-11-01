# streamlit_app.py
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

# ----------------------
# Styles (glassmorphism + simple animations)
# ----------------------
st.markdown(
    """
    <style>
    :root{
      --bg:#FAFAF7;
      --card:#ffffffcc;
      --muted:#6b7280;
      --accent:#2E7D32;
      --accent-2:#4FC3F7;
    }
    html,body, [class*="main"] {
      background: linear-gradient(180deg, #f7faf2 0%, #edf7f0 100%);
    }
    .glass {
      background: var(--card);
      border-radius: 14px;
      padding: 18px;
      box-shadow: 0 8px 24px rgba(9,30,66,0.08), inset 0 1px 0 rgba(255,255,255,0.6);
      backdrop-filter: blur(6px) saturate(120%);
    }
    .zone-bubble {
      border-radius: 999px;
      padding: 10px 14px;
      font-weight:700;
      color: white;
      display:inline-block;
      box-shadow: 0 6px 18px rgba(46,125,50,0.18);
      transform: translateY(0px);
      transition: transform .25s ease, box-shadow .25s ease;
      cursor: pointer;
    }
    .zone-bubble:hover { transform: translateY(-6px); box-shadow: 0 18px 36px rgba(46,125,50,0.22); }
    .small { font-size:0.78rem; color:var(--muted); }
    .header { font-family: 'Poppins', sans-serif; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------
# Configuration
# ----------------------
DATA_FILE = "thermal_data.csv"
MODEL_FILE = "thermal_model.joblib"

ZONES = [
    "Main Parking Lot", "Academic Block A", "Academic Block B",
    "Boys Hostel 1", "Boys Hostel 2", "Girls Hostel",
    "Sports Stadium", "Central Library", "Green Quad", "Food Court"
]

# ----------------------
# Helper utilities
# ----------------------
def load_dataset(path=DATA_FILE):
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df
    except Exception as e:
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
    return "#4caf50"

# ----------------------
# Data + Model load
# ----------------------
st.sidebar.title("Chandigarh University")
st.sidebar.markdown("*Thermal Digital Twin* — Live demo")

df = load_dataset()
pipeline = load_model()

if df is None:
    st.sidebar.warning("thermal_data.csv not found — using generated mock data (safe for demo).")
    # create mock hourly data for last 48 hours for each zone
    now = pd.Timestamp.now().floor("H")
    rows = []
    for z in ZONES:
        for h in range(0, 48):
            ts = now - pd.Timedelta(hours=h)
            base = 28 + 6 * math.sin((ts.dayofyear % 365)/365.0 * 2*math.pi)
            hour_cycle = 6 * math.sin((ts.hour-9)/24*2*math.pi)
            offset = 3 if "Parking" in z or "Stadium" in z else -1 if "Quad" in z or "Library" in z else 0
            temp = round(base + hour_cycle + offset + np.random.normal(0,0.6),1)
            uv = max(0, round(10 * math.sin((ts.hour-9)/24*2*math.pi) + np.random.normal(0,0.5),1))
            rows.append({"timestamp": ts, "zone": z, "temp": temp, "uv": uv})
    df = pd.DataFrame(rows)
else:
    st.sidebar.success("Dataset loaded.")

if pipeline is None:
    st.sidebar.warning("Model not found — forecast will use a simple heuristic.")
else:
    st.sidebar.success("Model loaded.")

# ----------------------
# Live snapshot for dashboard
# ----------------------
latest = df.sort_values("timestamp").groupby("zone").last().reset_index()
latest["status"] = latest["temp"].apply(temp_status)
# ensure zones order
latest = latest.set_index("zone").reindex(ZONES).reset_index()

# ----------------------
# Layout: header
# ----------------------
st.markdown("<div class='header'><h1 style='margin-bottom:4px'>Chandigarh University — Thermal Digital Twin</h1></div>", unsafe_allow_html=True)
st.markdown("<div class='small'>Environmental theme • Live monitoring • ROI calculator • Hackathon-ready</div>", unsafe_allow_html=True)
st.write("")

# ----------------------
# Top row: live cards + map
# ----------------------
col1, col2 = st.columns([2,3], gap="large")

with col1:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Live Zones Snapshot")
    cols = st.columns(3)
    for i, row in latest.iterrows():
        box = cols[i % 3]
        color = status_color(row["temp"])
        box.markdown(f"<div style='background:linear-gradient(135deg,#ffffffcc, #f5fff5cc); border-radius:12px; padding:12px;'>"
                     f"<div style='font-size:12px; color:#666'>{row['zone']}</div>"
                     f"<div style='font-size:20px; font-weight:700; color:{color}'>{row['temp']}°C</div>"
                     f"<div style='font-size:12px; color:#888'>UV {row['uv']}</div>"
                     f"<div style='margin-top:6px; font-size:12px; color:#666'>Status: {row['status']}</div>"
                     f"</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Campus Map — Click a bubble")
    # We will show simple floating bubbles (no heavy 3D). Each bubble links to zone details.
    bubble_html = "<div style='display:flex; flex-wrap:wrap; gap:10px; align-items:center; justify-content:center; padding:12px;'>"
    for _, r in latest.iterrows():
        color = status_color(r["temp"])
        bubble_html += f"<div class='zone-bubble' style='background:{color};' onclick=\"window.dispatchEvent(new CustomEvent('select-zone', {{detail: '{r['zone']}'}}))\">{r['zone'].split(' ')[0]}<div style='font-size:11px; opacity:0.9'>{r['temp']}°C</div></div>"
    bubble_html += "</div>"
    st.components.v1.html(bubble_html, height=160)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# Middle row: selected zone details + chart
# ----------------------
st.write("")
st.markdown("<div class='glass' style='padding:18px;'>", unsafe_allow_html=True)
sel1, sel2 = st.columns([2,1])
with sel1:
    st.subheader("Selected Zone Details")
    selected_zone = st.selectbox("Choose zone", options=ZONES, index=0)
    zone_row = latest[latest["zone"] == selected_zone].iloc[0]
    st.metric(label=f"{selected_zone} — Current Temp", value=f"{zone_row['temp']} °C", delta=None)
    st.write(f"UV Index: *{zone_row['uv']}*")
    st.write(f"Status: *{zone_row['status']}*")
    # show a small altair chart with historical temps
    hist = df[df["zone"] == selected_zone].sort_values("timestamp").tail(48)
    if not hist.empty:
        ch = alt.Chart(hist).mark_line(point=True).encode(
            x=alt.X('timestamp:T', title='Time'),
            y=alt.Y('temp:Q', title='Temperature (°C)'),
            tooltip=['timestamp:T','temp']
        ).properties(height=200, width=700)
        st.altair_chart(ch, use_container_width=True)
    else:
        st.info("No history available for this zone.")
with sel2:
    st.subheader("Quick Actions")
    if st.button("Refresh Data"):
        st.experimental_rerun()
    st.write("Download snapshot:")
    buf = io.StringIO()
    latest.to_csv(buf, index=False)
    b = buf.getvalue().encode()
    st.download_button("Download CSV", data=b, file_name="latest_snapshot.csv", mime="text/csv")
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# Forecast + ROI row
# ----------------------
left, right = st.columns([2,1])
with left:
    st.markdown("<div class='glass' style='padding:18px;'>", unsafe_allow_html=True)
    st.subheader("Forecast (Model Prediction)")
    with st.form("forecast_form"):
        zone_in = st.selectbox("Zone", options=ZONES, index=ZONES.index(selected_zone))
        date_in = st.date_input("Date", value=datetime.now().date())
        time_in = st.time_input("Time", value=(datetime.now() + timedelta(hours=1)).time())
        submitted = st.form_submit_button("Run Forecast")
        if submitted:
            # prepare input features similar to training pipeline
            timestamp = pd.to_datetime(f"{date_in} {time_in}")
            hour = timestamp.hour
            dayofweek = timestamp.dayofweek
            month = timestamp.month
            Xpred = pd.DataFrame([{
                "hour": hour, "dayofweek": dayofweek, "month": month,
                **{f"zone_{z}": int(z==zone_in) for z in ZONES}
            }])
            if pipeline is not None:
                try:
                    pred = pipeline.predict(Xpred)
                    p_temp = float(pred[0][0]); p_uv = float(pred[0][1])
                    st.success(f"Predicted Temp: {p_temp:.2f} °C · UV: {p_uv:.2f}")
                except Exception as e:
                    st.error("Model prediction failed. (Check model compatibility)")
                    st.exception(e)
            else:
                # fallback heuristic
                base = latest[latest["zone"]==zone_in]["temp"].values[0]
                p_temp = base + (np.random.rand()-0.4)*2
                p_uv = max(0, latest[latest["zone"]==zone_in]["uv"].values[0] + (np.random.rand()-0.4)*1.2)
                st.info("Using heuristic fallback prediction.")
                st.success(f"Predicted Temp: {p_temp:.2f} °C · UV: {p_uv:.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='glass' style='padding:18px;'>", unsafe_allow_html=True)
    st.subheader("ROI Calculator")
    # inputs
    cost = st.number_input("Upfront Cost (₹)", value=25000, min_value=0)
    energy_saved = st.number_input("Estimated yearly energy saving (₹)", value=6000, min_value=0)
    lifetime = st.number_input("Expected lifetime (years)", value=5, min_value=1)
    apply_subsidy = st.checkbox("Apply government subsidy (10%)", value=False)
    if apply_subsidy:
        cost = cost * 0.9
    total_saved = energy_saved * lifetime
    roi = None
    if cost > 0:
        roi = ((total_saved - cost) / cost) * 100
    st.metric("Total Saved (Lifetime)", f"₹{int(total_saved):,}")
    st.metric("Estimated ROI", f"{roi:.1f}% " if roi is not None and math.isfinite(roi) else "—")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# Insights and tips
# ----------------------
st.write("")
st.markdown("<div class='glass' style='padding:18px;'>", unsafe_allow_html=True)
st.subheader("Actionable Insights")
hotspots = latest[latest["status"]=="Hotspot"]
if not hotspots.empty:
    st.warning(f"Hotspots detected: {', '.join(hotspots['zone'].tolist())}. Consider immediate shade or cooling.")
else:
    st.success("No immediate hotspots detected.")

st.markdown("""
*Quick Recommendations*
- Plant shade trees in parking & stadium to cut peak temps (~2–4°C).
- Increase reflective surfaces on roof of academic buildings.
- Schedule outdoor activities in mornings when UV < 4.
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# Footer
# ----------------------
st.markdown("<div style='text-align:center; padding-top:12px; color:#6b7280;'>Made for Hackathon • Chandigarh University • Built with ♥</div>", unsafe_allow_html=True)