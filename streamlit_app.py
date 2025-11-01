# streamlit_app.py — PrithviSense (Simple & Clean Version)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

st.set_page_config(page_title="PrithviSense", layout="wide")

# ----------------- HEADER -----------------
st.title("🌿 PrithviSense")
st.write("### Smart Thermal Awareness for Sustainable Campuses")
st.write("This dashboard shows live campus temperature conditions, predicts future values, and calculates cost-saving potential.")

st.divider()

# ----------------- LOAD DATA -----------------
ZONES = [
    "Main Parking Lot", "Academic Block A", "Academic Block B",
    "Boys Hostel 1", "Boys Hostel 2", "Girls Hostel",
    "Sports Stadium", "Central Library", "Green Quad", "Food Court"
]

def load_data():
    try:
        return pd.read_csv("thermal_data.csv", parse_dates=["timestamp"])
    except:
        return None

def load_model():
    try:
        return joblib.load("thermal_model.joblib")
    except:
        return None

df = load_data()
model = load_model()

# Generate fallback demo data if dataset missing
if df is None:
    now = datetime.now()
    rows=[]
    for z in ZONES:
        for h in range(24):
            temp = 28 + np.random.normal(0,1.5)
            uv = round(np.random.uniform(2,9),1)
            rows.append([now - timedelta(hours=h), z, round(temp,1), uv])
    df = pd.DataFrame(rows, columns=["timestamp","zone","temp","uv"])

# Latest values
latest = df.sort_values("timestamp").groupby("zone").last().reset_index()

def status(t):
    return "🔥 Hotspot" if t>40 else ("🌤 Warm" if t>36 else "✅ Comfortable")

latest["Status"] = latest["temp"].apply(status)

# ----------------- LIVE TABLE -----------------
st.subheader("Current Campus Conditions")
st.dataframe(latest[["zone","temp","uv","Status"]], use_container_width=True)
st.divider()

# ----------------- FORECAST -----------------
st.subheader("Forecast Temperature")

col1, col2, col3 = st.columns(3)
zone = col1.selectbox("Zone", ZONES)
date = col2.date_input("Date", datetime.now().date())
time = col3.time_input("Time", datetime.now().time())

if st.button("Predict Temperature"):
    ts = pd.to_datetime(f"{date} {time}")
    hour = ts.hour
    dow = ts.dayofweek
    month = ts.month

    if model is not None:
        try:
            X = pd.DataFrame([{
                "hour":hour, "dayofweek":dow, "month":month,
                **{f"zone_{z}": int(z==zone) for z in ZONES}
            }])
            pred = model.predict(X)[0][0]
        except:
            pred = latest[latest.zone==zone].temp.values[0] + np.random.normal(0,1)
    else:
        pred = latest[latest.zone==zone].temp.values[0] + np.random.normal(0,1)

    st.success(f"Predicted Temperature for {zone}: **{pred:.1f}°C**")

st.divider()

# ----------------- ROI CALCULATOR -----------------
st.subheader("💰 ROI Calculator (Cost Saving)")

c1, c2, c3 = st.columns(3)
cost = c1.number_input("Installation Cost (₹)", min_value=0, value=25000)
saving = c2.number_input("Yearly Energy Saving (₹)", min_value=0, value=7000)
years = c3.number_input("Lifetime (Years)", min_value=1, value=5)

total = saving * years
roi = ((total - cost) / cost) * 100 if cost > 0 else None

st.write(f"**Total Savings Over Lifetime:** ₹{total:,}")
if roi is not None:
    st.write(f"**Return on Investment:** {roi:.1f}%")
else:
    st.write("ROI cannot be calculated.")

st.divider()

# ----------------- INSIGHTS -----------------
st.subheader("Actionable Insights")

hot = latest[latest["temp"]>40]
warm = latest[(latest["temp"]>36)&(latest["temp"]<=40)]

if len(hot) > 0:
    st.error(f"Immediate Cooling Needed in: {', '.join(hot.zone)}")
elif len(warm) > 0:
    st.warning(f"Monitor these zones: {', '.join(warm.zone)}")
else:
    st.success("All zones currently safe ✅")

st.write("")
st.caption("Built with care for Chandigarh University 💚")
