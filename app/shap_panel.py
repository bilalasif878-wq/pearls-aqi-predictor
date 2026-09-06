"""SHAP explanation panel for the Streamlit dashboard.

Handles both sklearn Pipelines (Ridge with scaling) and bare tree models
(RF, XGBoost, LightGBM) transparently.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.linear_model._base import LinearModel
from sklearn.pipeline import Pipeline

from src.config import HORIZONS, MODEL_NAME_TEMPLATE
from src.features.build import build_features
from src.inference.predict import _load_model


def _pick_explainer(est, background: pd.DataFrame, X_row: pd.DataFrame):
    """Choose the right SHAP explainer for whatever model type we have.

    - sklearn Pipeline (Ridge + Scaler) → LinearExplainer on the last step,
      after transforming through everything before it.
    - Any linear model → LinearExplainer.
    - Tree models (RF / XGB / LGB) → TreeExplainer.
    - Anything else → KernelExplainer (slow fallback).
    """
    if isinstance(est, Pipeline):
        # Split preprocessing from the final estimator
        final = est.steps[-1][1]
        pre = Pipeline(est.steps[:-1]) if len(est.steps) > 1 else None
        bg = pre.transform(background) if pre is not None else background.to_numpy()
        row_t = pre.transform(X_row) if pre is not None else X_row.to_numpy()
        if isinstance(final, LinearModel):
            explainer = shap.LinearExplainer(final, bg, feature_names=list(X_row.columns))
            sv = explainer(row_t)
            # Restore column names for readable plots
            sv.feature_names = list(X_row.columns)
            return sv
        # Non-linear final step in a pipeline , try Tree first, else Kernel
        try:
            explainer = shap.TreeExplainer(final)
            return explainer(row_t)
        except Exception:
            explainer = shap.KernelExplainer(final.predict, bg[:50])
            return explainer(row_t)
    if isinstance(est, LinearModel):
        explainer = shap.LinearExplainer(est, background, feature_names=list(X_row.columns))
        return explainer(X_row)
    try:
        explainer = shap.TreeExplainer(est)
        return explainer(X_row)
    except Exception:
        explainer = shap.KernelExplainer(est.predict, background.sample(min(50, len(background)), random_state=0))
        return explainer(X_row)


@st.cache_data(ttl=1800, show_spinner="Computing SHAP…")
def _shap_values_for_horizon(features_raw: pd.DataFrame, horizon: int):
    est, meta = _load_model(horizon)
    engineered = build_features(features_raw, drop_na_targets=False).dropna(how="all")
    feats = [c for c in meta["feature_cols"] if c in engineered.columns]
    engineered = engineered.dropna(subset=feats)
    if engineered.empty:
        raise RuntimeError("No non-NaN engineered rows for SHAP background")
    row = engineered.tail(1)[feats]
    background = engineered[feats].tail(500)
    sv = _pick_explainer(est, background, row)
    return sv, meta


def render_shap_panel(features_raw: pd.DataFrame, forecast: pd.DataFrame) -> None:
    horizon = st.selectbox("Horizon", HORIZONS, index=0)
    sv, meta = _shap_values_for_horizon(features_raw, horizon)
    st.write(f"Model: `{MODEL_NAME_TEMPLATE.format(horizon=horizon)}` ({meta['model_type']})")
    fig, _ = plt.subplots(figsize=(8, 5))
    try:
        shap.plots.waterfall(sv[0], max_display=12, show=False)
    except Exception:
        # Waterfall wants a specific shape; fall back to bar plot
        shap.plots.bar(sv[0], max_display=12, show=False)
    st.pyplot(fig, clear_figure=True)
