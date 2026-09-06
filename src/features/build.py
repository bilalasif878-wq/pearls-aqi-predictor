"""Feature engineering for AQI forecasting.

Given a wide dataframe with columns
  [timestamp, aqi, pm25, pm10, o3, no2, so2, co,
   temperature, humidity, dew_point, pressure, wind_speed, wind_direction,
   cloud_cover, precipitation]
we compute:
  - time features (cyclical)
  - lag features (t-1h, t-3h, t-6h, t-12h, t-24h, t-48h)
  - rolling features (mean / max / std over 6h, 12h, 24h)
  - derived: aqi_change_{3,24}h
  - three targets: aqi at t+24h, t+48h, t+72h
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import HORIZONS, LAG_HOURS, ROLLING_WINDOWS

POLLUTANT_COLS = ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co"]
WEATHER_COLS = [
    "temperature",
    "humidity",
    "dew_point",
    "pressure",
    "wind_speed",
    "wind_direction",
    "cloud_cover",
    "precipitation",
]


def _cyc(x: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    theta = 2 * np.pi * x / period
    return np.sin(theta), np.cos(theta)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Assumes df is indexed by timestamp (UTC)."""
    ts = df.index
    out = df.copy()
    out["hour"] = ts.hour
    out["day_of_week"] = ts.dayofweek
    out["day_of_year"] = ts.dayofyear
    out["month"] = ts.month
    out["is_weekend"] = (ts.dayofweek >= 5).astype(int)
    # Cyclical encodings — the model sees Monday and Sunday as neighbors, hour 23 next to hour 0
    out["hour_sin"], out["hour_cos"] = _cyc(pd.Series(ts.hour, index=ts), 24)
    out["dow_sin"], out["dow_cos"] = _cyc(pd.Series(ts.dayofweek, index=ts), 7)
    out["month_sin"], out["month_cos"] = _cyc(pd.Series(ts.month, index=ts), 12)
    return out


def add_lag_features(df: pd.DataFrame, cols: list[str], lags: tuple[int, ...] = LAG_HOURS) -> pd.DataFrame:
    """Assumes df is sorted ascending by timestamp with an hourly frequency."""
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        for lag in lags:
            out[f"{col}_lag_{lag}h"] = out[col].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame, cols: list[str], windows: tuple[int, ...] = ROLLING_WINDOWS
) -> pd.DataFrame:
    """Rolling stats, computed on shifted values so they never include the current row."""
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        shifted = out[col].shift(1)  # exclude current
        for w in windows:
            out[f"{col}_roll_{w}_mean"] = shifted.rolling(w, min_periods=max(2, w // 2)).mean()
            out[f"{col}_roll_{w}_max"] = shifted.rolling(w, min_periods=max(2, w // 2)).max()
            out[f"{col}_roll_{w}_std"] = shifted.rolling(w, min_periods=max(2, w // 2)).std()
    return out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "aqi" in out:
        out["aqi_change_3h"] = out["aqi"] - out["aqi"].shift(3)
        out["aqi_change_24h"] = out["aqi"] - out["aqi"].shift(24)
    if {"wind_speed", "wind_direction"}.issubset(out.columns):
        theta = np.deg2rad(out["wind_direction"])
        out["wind_u"] = -out["wind_speed"] * np.sin(theta)  # eastward component
        out["wind_v"] = -out["wind_speed"] * np.cos(theta)  # northward component
    return out


def add_targets(df: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Add aqi_target_{h}h columns = aqi shifted -h (future value)."""
    out = df.copy()
    if "aqi" not in out.columns:
        return out
    for h in horizons:
        out[f"aqi_target_{h}h"] = out["aqi"].shift(-h)
    return out


def build_features(raw: pd.DataFrame, *, drop_na_targets: bool = True) -> pd.DataFrame:
    """End-to-end feature build.

    Expects `raw` to have a `timestamp` column (UTC) plus pollutant / weather columns.
    Returns a dataframe indexed by timestamp, sorted, with all features and targets.
    """
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    # Regularise to strict hourly grid — forward-fill short gaps only
    df = df.asfreq("h")
    df[POLLUTANT_COLS + WEATHER_COLS] = df.reindex(
        columns=POLLUTANT_COLS + WEATHER_COLS
    ).ffill(limit=3)

    df = add_time_features(df)
    df = add_lag_features(df, POLLUTANT_COLS + ["temperature", "humidity", "wind_speed", "pressure"])
    df = add_rolling_features(df, ["aqi", "pm25", "pm10", "temperature", "wind_speed"])
    df = add_derived_features(df)
    df = add_targets(df)

    if drop_na_targets:
        target_cols = [c for c in df.columns if c.startswith("aqi_target_")]
        df = df.dropna(subset=target_cols)
    return df
