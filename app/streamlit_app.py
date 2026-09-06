"""Streamlit dashboard for the Pearls AQI Predictor."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import city_from_env
from src.data.health import AQI_CATEGORIES, category_for, is_hazardous
from src.inference.predict import predict_next_3_days
from src.store.hopsworks_client import read_features

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌫️", layout="wide")

city = city_from_env()

st.title(f"🌫️ Pearls AQI Predictor — {city.name}")
st.caption("Serverless forecast of Air Quality Index 3 days ahead. Data: AQICN + Open-Meteo. Feature store: Hopsworks.")


@st.cache_data(ttl=600, show_spinner="Reading feature store…")
def load_features() -> pd.DataFrame:
    df = read_features()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp")


@st.cache_data(ttl=600, show_spinner="Loading models & predicting…")
def load_forecast() -> pd.DataFrame:
    return predict_next_3_days()


features = load_features()
forecast = load_forecast()

# ---------- Current AQI header ----------
latest = features.iloc[-1]
current_aqi = float(latest["aqi"])
cat = category_for(current_aqi)

hdr1, hdr2, hdr3 = st.columns([1.5, 2, 2])
with hdr1:
    st.metric(
        label=f"Current AQI ({latest['timestamp']:%Y-%m-%d %H:%M UTC})",
        value=int(current_aqi),
        help=cat.advice,
    )
    st.markdown(
        f"<div style='background:{cat.color};color:white;padding:0.5em 1em;"
        f"border-radius:8px;text-align:center;font-weight:bold;'>{cat.label}</div>",
        unsafe_allow_html=True,
    )
with hdr2:
    st.metric("PM2.5", f"{latest.get('pm25', float('nan')):.1f}")
    st.metric("PM10", f"{latest.get('pm10', float('nan')):.1f}")
with hdr3:
    st.metric("Temperature (°C)", f"{latest.get('temperature', float('nan')):.1f}")
    st.metric("Wind speed", f"{latest.get('wind_speed', float('nan')):.1f}")

# ---------- Alert banner ----------
worst_future = forecast["predicted_aqi"].max()
if is_hazardous(worst_future):
    worst_cat = category_for(worst_future)
    st.error(
        f"⚠️ Health alert: forecast reaches **{int(worst_future)} ({worst_cat.label})** "
        f"in the next 72 h. {worst_cat.advice}"
    )
elif is_hazardous(current_aqi):
    st.warning(f"⚠️ Current AQI is {int(current_aqi)} — {cat.label}. {cat.advice}")
else:
    st.success(f"Current AQI is {int(current_aqi)} — {cat.label}.")

# ---------- Forecast chart ----------
st.subheader("Forecast — next 72 hours")

past = features.tail(24 * 7)  # last 7 days for context
fc = forecast.copy()
fc["forecast_for"] = pd.to_datetime(fc["forecast_for"], utc=True)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=past["timestamp"], y=past["aqi"],
        name="Recent AQI (7 days)", line=dict(color="#3b82f6", width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=fc["forecast_for"], y=fc["predicted_aqi"],
        name="Forecast", mode="lines+markers",
        line=dict(color="#ef4444", width=3, dash="dot"),
        marker=dict(size=10),
    )
)
# Health zone bands
for c in AQI_CATEGORIES:
    fig.add_hrect(y0=c.lo, y1=min(c.hi, 500), line_width=0,
                  fillcolor=c.color, opacity=0.05)
fig.update_layout(
    yaxis_title="AQI", xaxis_title="Time (UTC)",
    height=450, hovermode="x unified", legend=dict(orientation="h", y=1.05),
)
st.plotly_chart(fig, use_container_width=True)

# ---------- Forecast table ----------
st.subheader("Forecast detail")
tbl = fc.copy()
tbl["category"] = tbl["predicted_aqi"].apply(lambda a: category_for(a).label)
tbl = tbl[["horizon_h", "forecast_for", "predicted_aqi", "category", "model"]]
tbl.columns = ["Horizon (h)", "For (UTC)", "Predicted AQI", "Category", "Model"]
st.dataframe(tbl, use_container_width=True)

# ---------- Feature importance ----------
with st.expander("Feature importance (SHAP)"):
    try:
        from app.shap_panel import render_shap_panel

        render_shap_panel(features, forecast)
    except Exception as e:
        st.info(f"SHAP panel unavailable: {e}")

st.caption("Model registry: Hopsworks · Automation: GitHub Actions · Built for the Pearls AQI Predictor challenge.")
