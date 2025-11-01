# streamlit_app.py
"""
PRITHVI SENSE — Streamlit (Dark / Neon) single-file app
Place thermal_data.csv and thermal_model.joblib in repo root if available.
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

# ---------------- page config ----------------
st.set_page_config(
    page_title="PRITHVI SENSE — Thermal Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- constants ----------------
DATA_FILE = "thermal_data.csv"
MODEL_FILE = "thermal_model.joblib"

ZONES = [
    "Main Parking Lot", "Academic Block A", "Academic Block B",
    "Boys Hostel 1", "Boys Hostel 2", "Girls Hostel",
    "Sports Stadium", "Central Library", "Green Quad", "Food Court"
]

# neon colors
NEON_GREEN = "#7CFF6A"
NEON_CYAN = "#00E5FF"
WARN = "#FFB86B"
DANGER = "#FF5C74"
SAFE = "#8EE28E"

# ---------------- CSS / styling ----------------
st.markdown(
    f"""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <style>
      .stApp {{ background: linear-gradient(180deg,#050607 0%, #0b0f12 100%); color:#e6f7f7; font-family:'Inter',sans-serif; }}
      .card {{ background: rgba(255,255,255,0.02); padding:16px; border-radius:12px; border:1px solid rgba(255,255,255,0.03); box-shadow: 0 8px 20px rgba(0,0,0,0.6); }}
      h1 {{ color: {NEON_GREEN}; margin:0; font-size:28px; font-weight:800; }}
      .subtitle {{ color: {NEON_CYAN}; margin-bottom:6px; font-weight:600; }}
      .muted {{ color:#94a3b8; font-size:13px; }}
      .bubble {{ position:absolute; transform:translate(-50%,-50%); border-radius:999px; display:inline-flex; align-items:center; justify-content:center; font-weight:800; cursor:pointer; color:#0b0f12; }}
      .bubble-label {{ background: rgba(255,255,255,0.94); padding:6px 8px; border-radius:8px; color:#071018; font-weight:700; font-size:12px; }}
      .legend-item { display:inline-block; padding:8px 10px; border-radius:8px; margin-right:8px; color:#042026; font-weight:700; }
      @media (max-width:900px) {{
        h1 {{ font-size:20px; }}
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- helpers ----------------
def load_dataset(path=DATA_FILE):
    try:
        return pd.read_csv(path, parse_dates=["timestamp"])
    except Exception:
        return None

def load_model(path=MODEL_FILE):
    try:
        return joblib.load(path)
    except Exception:
        return None

def temp_status(temp):
    if temp > 40: return "Hotspot"
    if temp > 36: return "Medium"
    return "Safe"

def temp_color(temp):
    if temp > 45: return DANGER
    if temp > 40: return WARN
    if temp > 36: return NEON_CYAN
    return NEON_GREEN

# ---------------- load data & model (fallback demo) ----------------
df = load_dataset()
model = load_model()

if df is None:
    st.sidebar.warning("thermal_data.csv not found — using demo data.")
    # small demo dataset: last 48 hours per zone
    now = pd.Timestamp.now().floor("H")
    rows = []
    for z in ZONES:
        for h in range(0, 48):
            ts = now - pd.Timedelta(hours=h)
            base = 26 + 6 * math.sin((ts.dayofyear % 365)/365.0 * 2*math.pi)
            hour_cycle = 6 * math.sin((ts.hour-9)/24*2*math.pi)
            offset = 3 if ("Parking" in z or "Stadium" in z) else -2 if ("Quad" in z or "Library" in z) else 0
            temp = round(base + hour_cycle + offset + np.random.normal(0,0.6), 1)
            uv = round(max(0, 9 * math.sin((ts.hour-9)/24*2*math.pi) + np.random.normal(0,0.5)),1)
            rows.append({"timestamp": ts, "zone": z, "temp": temp, "uv": uv})
    df = pd.DataFrame(rows)
else:
    st.sidebar.success("Dataset loaded.")

if model is None:
    st.sidebar.info("Model not found — forecasts will use heuristic fallback.")
else:
    st.sidebar.success("Model loaded.")

# ---------------- compute latest snapshot ----------------
latest = df.sort_values("timestamp").groupby("zone").last().reset_index()
latest["status"] = latest["temp"].apply(temp_status)
latest = latest.set_index("zone").reindex(ZONES).reset_index()

# ---------------- header ----------------
col1, col2 = st.columns([3,1])
with col1:
    st.markdown("<h1>PRITHVI SENSE</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Intelligent Campus Thermal Twin — Chandigarh University</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Live monitoring • ROI-first recommendations • Predictive alerts</div>", unsafe_allow_html=True)
with col2:
    # theme toggle (visual only)
    theme_choice = st.selectbox("Theme", ["Tech (Dark)"], index=0)
st.markdown("---")

# ---------------- top area: snapshot + map ----------------
left, right = st.columns([1.1, 1.9], gap="large")

# left: live snapshot
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Live Zones Snapshot")
    # neat two-column list
    c1, c2 = st.columns(2)
    for i, r in latest.iterrows():
        target = c1 if (i % 2 == 0) else c2
        clr = temp_color(r["temp"])
        target.markdown(f"""
            <div style="padding:10px; border-radius:8px; margin-bottom:8px; background:linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.007)); border-left:4px solid {clr};">
              <div style="font-weight:800; color:#eafff0;">{r['zone']}</div>
              <div style="font-size:18px; font-weight:900; color:{clr}; margin-top:6px;">{r['temp']}°C</div>
              <div style="color:#9fb7b7; font-size:13px; margin-top:6px;">UV {r['uv']} · {r['status']}</div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# right: abstract campus map with neon bubbles (positions by percentages)
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Campus Map — Neon Bubbles (click to copy zone name)")
    st.markdown("<div style='position:relative; padding-top:56%;'><!-- map container -->", unsafe_allow_html=True)

    # map background: simple dark panel with grid labels (abstract, responsive)
    grid_html = "<div style='position:absolute; inset:6px; border-radius:8px; overflow:hidden; background:linear-gradient(180deg,#061014,#08121a); border:1px solid rgba(255,255,255,0.02);'>"
    grid_html += "<div style='width:100%; height:100%; display:grid; grid-template-columns: repeat(6, 1fr); grid-template-rows: repeat(4, 1fr); gap:8px; padding:8px;'>"
    labels = [
        "Gate","Main Parking Lot","Academic Block A","Academic Block B","Central Library","Green Quad",
        "Green Area","Sports Stadium","Food Court","Cafeteria","Admin Block","Garden",
        "Boys Hostel 1","Boys Hostel 2","Girls Hostel","Labs","Workshop","Open Field",
        "Solar Field","Clinic","Auditorium","Parking-2","Pond","Parkland"
    ]
    for lab in labels:
        grid_html += f"<div style='border-radius:8px; display:flex; align-items:center; justify-content:center; color:#8fd9d9; font-weight:700; font-size:12px; opacity:0.7;'>{lab}</div>"
    grid_html += "</div></div>"

    # bubble positions (approx grid percentages) - tweak if needed
    zone_positions = {
        "Main Parking Lot": (17, 12),
        "Academic Block A": (33, 12),
        "Academic Block B": (50, 12),
        "Central Library": (67, 12),
        "Green Quad": (83, 12),
        "Sports Stadium": (25, 35),
        "Food Court": (42, 35),
        "Boys Hostel 1": (58, 62),
        "Boys Hostel 2": (70, 62),
        "Girls Hostel": (82, 62),
    }

    overlay = ""
    tmin = latest['temp'].min()
    for _, r in latest.iterrows():
        name = r["zone"]
        if name in zone_positions:
            x, y = zone_positions[name]
            size = 40 + int(max(0, (r["temp"] - tmin)) * 3)
            bg = temp_color(r["temp"])
            # bubble html - clicking copies zone name (JS)
            overlay += f"""
            <div class="bubble" style="left:{x}%; top:{y}%; width:{size}px; height:{size}px; line-height:{size}px; background:{bg}; border:1px solid rgba(255,255,255,0.06);"
                 onclick="navigator.clipboard.writeText('{name}')">
              <div class="bubble-label">{int(r['temp'])}°</div>
            </div>
            """

    st.components.v1.html(grid_html + f"<div style='position:absolute; inset:6px; pointer-events:none;'>{overlay}</div>", height=420, scrolling=False)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- forecast + ROI row ----------------
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
fleft, fright = st.columns([2,1], gap="large")

with fleft:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Forecast — Model Prediction")
    c1, c2, c3 = st.columns([1,1,1])
    zone_in = c1.selectbox("Zone", ZONES, index=0)
    date_in = c2.date_input("Date", datetime.now().date())
    time_in = c3.time_input("Time", (datetime.now()+timedelta(hours=1)).time())
    if st.button("Run Forecast", key="forecast"):
        ts = pd.to_datetime(f"{date_in} {time_in}")
        hour = ts.hour; dow = ts.dayofweek; month = ts.month
        Xpred = pd.DataFrame([{"hour":hour, "dayofweek":dow, "month":month, **{f"zone_{z}": int(z==zone_in) for z in ZONES}}])
        try:
            if model is not None:
                pred = model.predict(Xpred)
                # safe unpack: model may return shape (n,2) or list
                p_temp = float(pred[0][0]) if hasattr(pred[0], "__len__") else float(pred[0])
                # try uv as second output if available
                p_uv = float(pred[0][1]) if hasattr(pred[0], "__len__") and len(pred[0])>1 else np.nan
            else:
                # fallback heuristic
                base = latest.loc[latest.zone==zone_in,"temp"].values[0]
                p_temp = float(base + (np.random.rand()-0.45)*1.8)
                p_uv = float(max(0, latest.loc[latest.zone==zone_in,"uv"].values[0] + (np.random.rand()-0.45)*0.9))
        except Exception:
            p_temp = latest.loc[latest.zone==zone_in,"temp"].values[0] + np.random.normal(0,1)
            p_uv = latest.loc[latest.zone==zone_in,"uv"].values[0] + np.random.normal(0,0.6)

        st.success(f"Predicted Temp: {p_temp:.2f} °C — Predicted UV: {p_uv:.2f}")
        # show last 48h + forecast point
        hist = df[df["zone"]==zone_in].sort_values("timestamp").tail(72)
        fpoint = pd.DataFrame([{"timestamp": ts, "temp": p_temp, "type":"forecast"}])
        h2 = hist.assign(type="history")[["timestamp","temp","type"]]
        chart_df = pd.concat([h2, fpoint], ignore_index=True)
        base = alt.Chart(chart_df).mark_line(interpolate='monotone').encode(
            x=alt.X('timestamp:T', title='Time'),
            y=alt.Y('temp:Q', title='Temperature (°C)'),
            color=alt.Color('type:N', scale=alt.Scale(range=[NEON_GREEN, WARN]), legend=None)
        ).properties(height=300)
        point = alt.Chart(chart_df[chart_df.type=="forecast"]).mark_point(size=130, color=DANGER).encode(x='timestamp:T', y='temp:Q')
        st.altair_chart(base + point, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with fright:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("ROI Calculator — Priority")
    upfront = st.number_input("Upfront Cost (₹)", min_value=0, value=30000, step=500)
    yearly = st.number_input("Estimated yearly saving (₹)", min_value=0, value=7000, step=100)
    lifetime = st.number_input("Lifetime (years)", min_value=1, value=5, step=1)
    subsidy = st.checkbox("Apply government subsidy (10%)", value=False)
    effective = upfront * (0.9 if subsidy else 1.0)
    lifetime_saved = yearly * lifetime
    roi = ((lifetime_saved - effective) / effective * 100) if effective>0 else None
    st.markdown(f"<div style='padding:10px; border-radius:10px; background:linear-gradient(90deg, rgba(124,255,106,0.04), rgba(0,229,255,0.02));'><div style='font-weight:800;'>Lifetime saved: ₹{int(lifetime_saved):,}</div><div style='font-weight:900; font-size:20px; color:{NEON_GREEN}; margin-top:6px;'>ROI: {'{:.1f}%'.format(roi) if roi is not None else '—'}</div></div>", unsafe_allow_html=True)
    # simple payback chart
    years = list(range(0, lifetime+1))
    cum = [yearly * y for y in years]
    df_roi = pd.DataFrame({"year": years, "savings": cum, "cost": [effective]*len(years)})
    area = alt.Chart(df_roi).mark_area(opacity=0.28, color=NEON_CYAN).encode(x='year:O', y='savings:Q')
    cost_line = alt.Chart(df_roi).mark_line(color=DANGER, strokeDash=[4,4]).encode(x='year:O', y='cost:Q')
    st.altair_chart(area + cost_line, use_container_width=True, height=200)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- actionable insights + UV guide ----------------
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Actionable Insights — Automatic")

hot = latest[latest["temp"] > 40]
mid = latest[(latest["temp"] > 36) & (latest["temp"] <= 40)]
if len(hot) > 0:
    st.error(f"IMMEDIATE ACTION — Hotspots: {', '.join(hot['zone'].tolist())}")
elif len(mid) > 0:
    st.warning(f"Moderate risk: {', '.join(mid['zone'].tolist())}")
else:
    st.success("All zones safe — no immediate hotspots.")

with st.expander("UV Index — Safety Guide"):
    st.write("""
    - 0–2: Low — safe.
    - 3–5: Moderate — use sunscreen/hats.
    - 6–7: High — protection required.
    - 8–10: Very High — minimize outdoor exposure.
    - 11+: Extreme — avoid prolonged sun exposure.
    """)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- footer + download ----------------
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
leftf, rightf = st.columns([3,1])
with leftf:
    st.markdown("<div style='color:#9fbfb7; font-weight:700;'>PRITHVI SENSE — Chandigarh University • Demo</div>", unsafe_allow_html=True)
with rightf:
    buf = io.StringIO(); latest.to_csv(buf, index=False)
    st.download_button("Download latest snapshot (CSV)", data=buf.getvalue().encode(), file_name="latest_snapshot.csv", mime="text/csv")
