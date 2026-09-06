"""Inference: load latest features + models and produce a 3-day forecast."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import HORIZONS, MODEL_NAME_TEMPLATE
from src.features.build import build_features
from src.store.hopsworks_client import get_model_registry, read_features

logger = logging.getLogger(__name__)


@dataclass
class ForecastPoint:
    horizon_h: int
    forecast_for: pd.Timestamp
    predicted_aqi: float
    model: str


def _load_model(horizon: int) -> tuple[Any, dict]:
    """Fetch the best registered model for one horizon."""
    mr = get_model_registry()
    name = MODEL_NAME_TEMPLATE.format(horizon=horizon)
    model = mr.get_best_model(name=name, metric="rmse", direction="min")
    model_dir = Path(model.download())
    est = joblib.load(model_dir / "model.joblib")
    meta = json.loads((model_dir / "metadata.json").read_text())
    return est, meta


def _pick_inference_row(engineered: pd.DataFrame, required_feats: list[str]) -> pd.DataFrame:
    """The most recent row whose required feature columns are all non-NaN.

    We need this because build_features can leave NaN in the last few rows
    (missing hours, freshly-added columns, ffill exhausted) and Ridge refuses
    NaN inputs. Falls back to zero-imputing the last row if no clean row exists.
    """
    present = [c for c in required_feats if c in engineered.columns]
    valid = engineered.dropna(subset=present)
    if not valid.empty:
        return valid.tail(1)
    # Fallback: zero-impute the very last row so inference still returns something
    row = engineered.tail(1).copy()
    row[present] = row[present].fillna(0.0)
    logger.warning("No row with all non-NaN features; falling back to zero-imputation.")
    return row


def predict_next_3_days() -> pd.DataFrame:
    raw = read_features()
    engineered = build_features(raw, drop_na_targets=False)
    engineered = engineered.dropna(how="all")
    if engineered.empty:
        raise RuntimeError("No engineered rows available for inference.")

    # Use the 24h model's feature list as the reference for choosing the row
    _, meta_ref = _load_model(HORIZONS[0])
    row = _pick_inference_row(engineered, meta_ref["feature_cols"])
    ts_now = row.index[-1]
    logger.info("Inference row timestamp: %s", ts_now)

    points: list[ForecastPoint] = []
    for h in HORIZONS:
        est, meta = _load_model(h)
        feats = [c for c in meta["feature_cols"] if c in row.columns]
        # Belt-and-braces: if this model's feature list has NaNs in the chosen row,
        # zero-impute just for it (rare — feature lists are the same across horizons).
        X = row[feats]
        if X.isna().any().any():
            X = X.fillna(0.0)
        pred = float(est.predict(X)[0])
        points.append(
            ForecastPoint(
                horizon_h=h,
                forecast_for=ts_now + pd.Timedelta(hours=h),
                predicted_aqi=pred,
                model=meta["model_type"],
            )
        )

    return pd.DataFrame([p.__dict__ for p in points])
