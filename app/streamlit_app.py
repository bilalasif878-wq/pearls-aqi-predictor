"""Streamlit dashboard for the Pearls AQI Predictor.

Editorial data-journalism aesthetic. Warm cream on near-black.
Category color used as a single, disciplined accent that tints the hero.
"""
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

# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Design tokens
BG          = "#0b0b0f"     # near-black, warm undertone
BG_ALT      = "#131318"     # subtle surface
INK         = "#f4efe6"     # warm cream
MUTED       = "#8e857a"     # warm gray
DIM         = "#4a4640"     # dimmer
HAIRLINE    = "#22212a"     # divider
ACCENT_LINE = "#c9c1b4"     # observed line color (cream)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
city = city_from_env()


@st.cache_data(ttl=600, show_spinner=False)
def load_features() -> pd.DataFrame:
    df = read_features()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp")


@st.cache_data(ttl=600, show_spinner=False)
def load_forecast() -> pd.DataFrame:
    return predict_next_3_days()


with st.spinner(""):
    features = load_features()
    forecast = load_forecast()

latest = features.iloc[-1]
current_aqi = float(latest["aqi"])
current_cat = category_for(current_aqi)
worst_future = forecast["predicted_aqi"].max()
worst_cat = category_for(worst_future)

# The color that ties the whole page together, derived from the current AQI
TINT = current_cat.color


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {{ color-scheme: dark; }}
    html, body, .stApp {{
        background: {BG};
        color: {INK};
    }}
    .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 400;
    }}

    #MainMenu, header, footer {{ visibility: hidden; }}
    .stDeployButton {{ display: none; }}
    .block-container {{
        padding: 3.5rem 4rem 5rem 4rem;
        max-width: 1240px;
    }}

    /* --------- Masthead ---------- */
    .masthead {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        border-bottom: 1px solid {HAIRLINE};
        padding-bottom: 1rem;
        margin-bottom: 3rem;
    }}
    .brand {{
        font-family: 'Instrument Serif', serif;
        font-size: 1.35rem;
        font-style: italic;
        color: {INK};
    }}
    .brand-sub {{
        color: {MUTED};
        font-size: 0.72rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        margin-left: 0.8rem;
    }}
    .stamp {{
        color: {MUTED};
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
    }}

    /* --------- Hero ---------- */
    .hero {{
        display: grid;
        grid-template-columns: 1.05fr 1fr;
        gap: 4rem;
        align-items: end;
        padding: 1rem 0 3.5rem 0;
        border-bottom: 1px solid {HAIRLINE};
    }}
    .hero-eyebrow {{
        color: {MUTED};
        font-size: 0.7rem;
        letter-spacing: 0.24em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }}
    .hero-city {{
        font-family: 'Instrument Serif', serif;
        font-size: 3.4rem;
        line-height: 1;
        letter-spacing: -0.02em;
        margin: 0 0 0.6rem 0;
        color: {INK};
    }}
    .hero-country {{
        font-family: 'Instrument Serif', serif;
        font-style: italic;
        font-size: 1.35rem;
        color: {MUTED};
        margin: 0 0 1.8rem 0;
    }}
    .hero-summary {{
        color: {INK};
        font-size: 1.05rem;
        line-height: 1.55;
        max-width: 30rem;
    }}
    .hero-summary em {{
        font-family: 'Instrument Serif', serif;
        font-style: italic;
        color: {TINT};
        font-size: 1.1rem;
        font-weight: 400;
    }}

    .aqi-block {{
        text-align: right;
    }}
    .aqi-glyph {{
        font-family: 'Instrument Serif', serif;
        font-weight: 400;
        font-size: 12rem;
        line-height: 0.85;
        letter-spacing: -0.05em;
        color: {TINT};
        text-shadow: 0 0 60px {TINT}22;
    }}
    .aqi-label {{
        color: {MUTED};
        font-size: 0.7rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }}
    .aqi-cat {{
        color: {TINT};
        font-family: 'Instrument Serif', serif;
        font-style: italic;
        font-size: 1.35rem;
        margin-top: 0.4rem;
    }}

    /* --------- Metric strip ---------- */
    .metric-strip {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        border-bottom: 1px solid {HAIRLINE};
        padding: 2rem 0;
    }}
    .metric {{
        padding: 0 1.5rem;
        border-left: 1px solid {HAIRLINE};
    }}
    .metric:first-child {{ border-left: none; padding-left: 0; }}
    .metric-label {{
        color: {MUTED};
        font-size: 0.65rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }}
    .metric-value {{
        font-family: 'Instrument Serif', serif;
        font-size: 2.4rem;
        font-weight: 400;
        color: {INK};
        line-height: 1;
        letter-spacing: -0.02em;
    }}
    .metric-unit {{
        color: {MUTED};
        font-size: 0.85rem;
        margin-left: 0.4rem;
        font-family: 'Inter', sans-serif;
    }}

    /* --------- Section ---------- */
    .section-head {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin: 3.5rem 0 1.4rem 0;
    }}
    .section-title {{
        font-family: 'Instrument Serif', serif;
        font-size: 1.7rem;
        color: {INK};
    }}
    .section-sub {{
        color: {MUTED};
        font-size: 0.72rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
    }}

    /* --------- Alert ---------- */
    .banner {{
        border-left: 3px solid {TINT};
        padding: 1rem 1.4rem;
        background: {BG_ALT};
        margin: 2rem 0 0 0;
        color: {INK};
        font-size: 0.95rem;
        line-height: 1.55;
    }}
    .banner em {{
        font-family: 'Instrument Serif', serif;
        font-style: italic;
        color: {TINT};
    }}

    /* --------- Table override ---------- */
    div[data-testid="stDataFrame"] {{
        background: {BG};
        border: 1px solid {HAIRLINE};
        border-radius: 0;
        font-family: 'JetBrains Mono', monospace !important;
    }}
    div[data-testid="stDataFrame"] * {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        color: {INK} !important;
    }}

    /* --------- Expander ---------- */
    .stExpander, .stExpander > details {{
        background: {BG_ALT} !important;
        border: 1px solid {HAIRLINE} !important;
        border-radius: 0 !important;
    }}
    .stExpander summary p {{
        color: {MUTED} !important;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-size: 0.78rem !important;
    }}

    /* --------- Colophon ---------- */
    .colophon {{
        margin-top: 4rem;
        padding-top: 1.5rem;
        border-top: 1px solid {HAIRLINE};
        color: {DIM};
        font-size: 0.75rem;
        letter-spacing: 0.02em;
        line-height: 1.7;
    }}
    .colophon strong {{
        color: {MUTED};
        font-weight: 500;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-size: 0.68rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Masthead
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="masthead">
        <div>
            <span class="brand">Pearls AQI</span>
            <span class="brand-sub">Air Quality Intelligence</span>
        </div>
        <div class="stamp">{latest['timestamp']:%d %b %Y . %H:%M UTC}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
trend_word = (
    "improving" if worst_future < current_aqi - 5
    else "worsening" if worst_future > current_aqi + 5
    else "steady"
)

st.markdown(
    f"""
    <div class="hero">
        <div>
            <div class="hero-eyebrow">Air Quality Index</div>
            <div class="hero-city">{city.name}</div>
            <div class="hero-country">Pakistan</div>
            <div class="hero-summary">
                Air quality in {city.name} is currently <em>{current_cat.label.lower()}</em>,
                with the three-day forecast trending <em>{trend_word}</em> to
                <em>{int(worst_future)}</em> ({worst_cat.label.lower()}).
                Forecasts are generated hourly from Copernicus CAMS observations
                joined with ERA5 meteorology.
            </div>
        </div>
        <div class="aqi-block">
            <div class="aqi-label">Now</div>
            <div class="aqi-glyph">{int(current_aqi)}</div>
            <div class="aqi-cat">{current_cat.label}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Metric strip
# ---------------------------------------------------------------------------
def metric_html(label: str, value, unit: str, fmt: str = "{:.1f}") -> str:
    display = fmt.format(value) if value is not None and pd.notna(value) else "."
    return f"""
    <div class="metric">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{display}<span class="metric-unit">{unit}</span></div>
    </div>
    """


st.markdown(
    f"""
    <div class="metric-strip">
        {metric_html("PM 2.5", latest.get("pm25"), "μg/m³")}
        {metric_html("PM 10",  latest.get("pm10"), "μg/m³")}
        {metric_html("Temperature", latest.get("temperature"), "°C")}
        {metric_html("Wind", latest.get("wind_speed"), "km/h")}
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
if is_hazardous(worst_future):
    st.markdown(
        f"""<div class="banner">
        Forecast reaches <em>{int(worst_future)}</em>, <em>{worst_cat.label.lower()}</em>, within 72 hours. {worst_cat.advice}
        </div>""",
        unsafe_allow_html=True,
    )
elif is_hazardous(current_aqi):
    st.markdown(
        f"""<div class="banner">
        Current air quality is <em>{current_cat.label.lower()}</em>. {current_cat.advice}
        </div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Forecast chart
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="section-head">
        <div class="section-title">72-hour forecast</div>
        <div class="section-sub">Observed . Predicted</div>
    </div>
    """,
    unsafe_allow_html=True,
)

past = features.tail(24 * 7)
fc = forecast.copy()
fc["forecast_for"] = pd.to_datetime(fc["forecast_for"], utc=True)

fig = go.Figure()

# Subtle category bands
for c in AQI_CATEGORIES:
    fig.add_hrect(
        y0=c.lo, y1=min(c.hi, 500),
        line_width=0, fillcolor=c.color, opacity=0.03,
        layer="below",
    )

# Observed line
fig.add_trace(go.Scatter(
    x=past["timestamp"], y=past["aqi"],
    name="Observed",
    line=dict(color=ACCENT_LINE, width=1.4),
    hovertemplate="%{x|%b %d, %H:%M}<br>AQI %{y:.0f}<extra></extra>",
))

# Bridge line
bridge_x = [past["timestamp"].iloc[-1], fc["forecast_for"].iloc[0]]
bridge_y = [past["aqi"].iloc[-1], fc["predicted_aqi"].iloc[0]]
fig.add_trace(go.Scatter(
    x=bridge_x, y=bridge_y,
    line=dict(color=TINT, width=1.6, dash="dot"),
    showlegend=False, hoverinfo="skip",
))
# Forecast
fig.add_trace(go.Scatter(
    x=fc["forecast_for"], y=fc["predicted_aqi"],
    name="Predicted",
    mode="lines+markers",
    line=dict(color=TINT, width=2, dash="dot"),
    marker=dict(size=9, color=TINT, line=dict(color=BG, width=2)),
    hovertemplate="%{x|%b %d, %H:%M}<br>Predicted AQI %{y:.0f}<extra></extra>",
))

fig.update_layout(
    height=460,
    margin=dict(l=10, r=10, t=10, b=30),
    plot_bgcolor=BG,
    paper_bgcolor=BG,
    font=dict(color=INK, family="Inter, sans-serif", size=12),
    hovermode="x unified",
    hoverlabel=dict(bgcolor=BG_ALT, bordercolor=HAIRLINE, font=dict(color=INK, family="JetBrains Mono, monospace")),
    legend=dict(
        orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=10.5),
    ),
    xaxis=dict(
        gridcolor=HAIRLINE, showgrid=True, zeroline=False,
        tickfont=dict(color=MUTED, size=10.5, family="JetBrains Mono, monospace"),
        title="", showline=True, linecolor=HAIRLINE,
    ),
    yaxis=dict(
        gridcolor=HAIRLINE, showgrid=True, zeroline=False,
        tickfont=dict(color=MUTED, size=10.5, family="JetBrains Mono, monospace"),
        title="", showline=False,
    ),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Forecast table
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="section-head">
        <div class="section-title">Point forecast</div>
        <div class="section-sub">Three horizons</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tbl = fc.copy()
tbl["category"] = tbl["predicted_aqi"].apply(lambda a: category_for(a).label)
tbl["forecast_for"] = tbl["forecast_for"].dt.strftime("%d %b, %H:%M UTC")
tbl["predicted_aqi"] = tbl["predicted_aqi"].round(1)
tbl = tbl.rename(columns={
    "horizon_h": "Horizon",
    "forecast_for": "Valid at",
    "predicted_aqi": "AQI",
    "category": "Category",
    "model": "Model",
})
st.dataframe(
    tbl[["Horizon", "Valid at", "AQI", "Category", "Model"]],
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="section-head">
        <div class="section-title">Attribution</div>
        <div class="section-sub">SHAP feature contributions</div>
    </div>
    """,
    unsafe_allow_html=True,
)
with st.expander("Explain the 24-hour prediction"):
    try:
        from app.shap_panel import render_shap_panel

        render_shap_panel(features, forecast)
    except Exception as e:
        st.info(f"SHAP panel unavailable: {e}")


# ---------------------------------------------------------------------------
# Colophon
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="colophon">
        <strong>Sources</strong> Open-Meteo (Copernicus CAMS air quality, ERA5 meteorology).
        <strong style="margin-left:1rem;">Infrastructure</strong> Hopsworks feature store and model registry, GitHub Actions for hourly ingest and daily training, Streamlit Community Cloud for delivery.
        <strong style="margin-left:1rem;">Models</strong> Persistence, Ridge, Random Forest, XGBoost and LightGBM evaluated per horizon; best by test RMSE promoted to the registry.
    </div>
    """,
    unsafe_allow_html=True,
)
