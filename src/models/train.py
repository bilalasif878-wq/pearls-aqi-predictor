"""Training utilities: candidates, time-splits, and per-horizon fit."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build import build_features
from src.models.baseline import PersistenceRegressor
from src.models.evaluate import Metrics, compute_metrics

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Model catalogue
# ------------------------------------------------------------------

def _candidates() -> dict[str, Any]:
    """All candidate models we evaluate per horizon.

    We import xgboost / lightgbm lazily so a plain scikit-learn install still runs
    something useful.
    """
    models: dict[str, Any] = {
        "persistence": PersistenceRegressor(),
        "ridge": Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=0))]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=0
        ),
    }
    try:
        from xgboost import XGBRegressor

        models["xgboost"] = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=0,
            n_jobs=-1,
            tree_method="hist",
        )
    except Exception as e:
        logger.warning("xgboost not available: %s", e)

    try:
        from lightgbm import LGBMRegressor

        models["lightgbm"] = LGBMRegressor(
            n_estimators=500,
            max_depth=-1,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=0,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception as e:
        logger.warning("lightgbm not available: %s", e)


    try:
        from src.models.lstm import LSTMForecaster

        models["lstm"] = LSTMForecaster(
            hidden=64, num_layers=1, epochs=15, batch_size=256, lr=2e-3, dropout=0.2,
        )
    except Exception as e:
        logger.warning("PyTorch LSTM not available: %s", e)

    return models


# ------------------------------------------------------------------
# Time-based splitting
# ------------------------------------------------------------------

@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def time_split(df: pd.DataFrame, test_days: int = 30, val_days: int = 30) -> Split:
    """Split a timestamp-indexed dataframe into train / val / test tail-first."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("dataframe must be timestamp-indexed")
    end = df.index.max()
    test_start = end - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)
    test = df[df.index > test_start]
    val = df[(df.index > val_start) & (df.index <= test_start)]
    train = df[df.index <= val_start]
    return Split(train=train, val=val, test=test)


# ------------------------------------------------------------------
# Feature / target selection
# ------------------------------------------------------------------

def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Everything except targets and non-numeric metadata."""
    drop = [c for c in df.columns if c.startswith("aqi_target_")]
    drop += ["city"]
    drop += [c for c in df.columns if df[c].dtype == "object"]
    return [c for c in df.columns if c not in drop]


# ------------------------------------------------------------------
# Fit one (model, horizon) pair
# ------------------------------------------------------------------

@dataclass
class TrainedModel:
    horizon: int
    name: str
    estimator: Any
    metrics_val: Metrics
    metrics_test: Metrics
    feature_cols: list[str]


def fit_and_eval(
    split: Split,
    horizon: int,
    name: str,
    estimator: Any,
) -> TrainedModel:
    target = f"aqi_target_{horizon}h"
    feats = _feature_columns(split.train)

    X_tr, y_tr = split.train[feats], split.train[target]
    X_v, y_v = split.val[feats], split.val[target]
    X_te, y_te = split.test[feats], split.test[target]

    # Drop rows with NaNs in features/target (lags at the head, targets at the tail)
    tr_mask = ~(X_tr.isna().any(axis=1) | y_tr.isna())
    v_mask = ~(X_v.isna().any(axis=1) | y_v.isna())
    te_mask = ~(X_te.isna().any(axis=1) | y_te.isna())

    logger.info("horizon=%dh model=%s train=%d val=%d test=%d",
                horizon, name, tr_mask.sum(), v_mask.sum(), te_mask.sum())

    estimator.fit(X_tr[tr_mask], y_tr[tr_mask])
    m_val = compute_metrics(y_v[v_mask], estimator.predict(X_v[v_mask]))
    m_te = compute_metrics(y_te[te_mask], estimator.predict(X_te[te_mask]))
    logger.info("  val  %s", m_val.as_dict())
    logger.info("  test %s", m_te.as_dict())

    return TrainedModel(
        horizon=horizon,
        name=name,
        estimator=estimator,
        metrics_val=m_val,
        metrics_test=m_te,
        feature_cols=feats,
    )


def train_all(features_raw: pd.DataFrame, horizons: tuple[int, ...] = (24, 48, 72)) -> list[TrainedModel]:
    """Full training loop. Returns every (horizon, model) result."""
    df = build_features(features_raw)
    split = time_split(df, test_days=30, val_days=30)
    logger.info("Split shapes: train=%s val=%s test=%s",
                split.train.shape, split.val.shape, split.test.shape)

    results: list[TrainedModel] = []
    for h in horizons:
        for name, est in _candidates().items():
            # Fresh copies so persistence's fit doesn't pollute
            from copy import deepcopy

            results.append(fit_and_eval(split, h, name, deepcopy(est)))
    return results


def pick_best(results: list[TrainedModel]) -> dict[int, TrainedModel]:
    """Choose the model with lowest test RMSE per horizon."""
    best: dict[int, TrainedModel] = {}
    for r in results:
        if r.horizon not in best or r.metrics_test.rmse < best[r.horizon].metrics_test.rmse:
            best[r.horizon] = r
    return best


def results_table(results: list[TrainedModel]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "horizon_h": r.horizon,
                "model": r.name,
                "val_rmse": r.metrics_val.rmse,
                "val_mae": r.metrics_val.mae,
                "val_r2": r.metrics_val.r2,
                "test_rmse": r.metrics_test.rmse,
                "test_mae": r.metrics_test.mae,
                "test_r2": r.metrics_test.r2,
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon_h", "test_rmse"])
