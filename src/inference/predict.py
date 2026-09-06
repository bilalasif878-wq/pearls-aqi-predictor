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


def _latest_row(features: pd.DataFrame) -> pd.DataFrame:
    """The most recent single row of engineered features."""
    engineered = build_features(features, drop_na_targets=False)
    engineered = engineered.dropna(how="all")
    return engineered.tail(1)


def predict_next_3_days() -> pd.DataFrame:
    raw = read_features()
    row = _latest_row(raw)
    if row.empty:
        raise RuntimeError("No engineered rows available for inference.")
    ts_now = row.index[-1]

    points: list[ForecastPoint] = []
    for h in HORIZONS:
        est, meta = _load_model(h)
        feats = [c for c in meta["feature_cols"] if c in row.columns]
        pred = float(est.predict(row[feats])[0])
        points.append(
            ForecastPoint(
                horizon_h=h,
                forecast_for=ts_now + pd.Timedelta(hours=h),
                predicted_aqi=pred,
                model=meta["model_type"],
            )
        )

    return pd.DataFrame([p.__dict__ for p in points])
