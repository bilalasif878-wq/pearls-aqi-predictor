"""SHAP explanation panel for the Streamlit dashboard."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from src.config import HORIZONS, MODEL_NAME_TEMPLATE
from src.features.build import build_features
from src.inference.predict import _load_model  # small utility import


@st.cache_data(ttl=1800, show_spinner="Computing SHAP…")
def _shap_values_for_horizon(features_raw: pd.DataFrame, horizon: int):
    est, meta = _load_model(horizon)
    engineered = build_features(features_raw, drop_na_targets=False).dropna(how="all")
    row = engineered.tail(1)
    feats = [c for c in meta["feature_cols"] if c in row.columns]
    background = engineered[feats].tail(500).dropna()
    try:
        explainer = shap.TreeExplainer(est)
        sv = explainer(row[feats])
    except Exception:
        explainer = shap.KernelExplainer(est.predict, background.sample(min(50, len(background)), random_state=0))
        sv = explainer(row[feats])
    return sv, meta


def render_shap_panel(features_raw: pd.DataFrame, forecast: pd.DataFrame) -> None:
    horizon = st.selectbox("Horizon", HORIZONS, index=0)
    sv, meta = _shap_values_for_horizon(features_raw, horizon)
    st.write(f"Model: `{MODEL_NAME_TEMPLATE.format(horizon=horizon)}` ({meta['model_type']})")
    fig, _ = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(sv[0], max_display=12, show=False)
    st.pyplot(fig, clear_figure=True)
