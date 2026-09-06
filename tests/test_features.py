"""Unit tests for feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build import (
    add_lag_features,
    add_rolling_features,
    add_targets,
    add_time_features,
    build_features,
)


def _synthetic_raw(hours: int = 240) -> pd.DataFrame:
    """A clean, dense hourly frame with a daily sinusoidal AQI pattern."""
    ts = pd.date_range("2026-01-01", periods=hours, freq="h", tz="UTC")
    aqi = 100 + 40 * np.sin(2 * np.pi * ts.hour / 24) + np.random.default_rng(0).normal(0, 5, hours)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "aqi": aqi,
            "pm25": aqi * 0.6,
            "pm10": aqi * 0.8,
            "o3": np.random.default_rng(1).uniform(10, 40, hours),
            "no2": np.random.default_rng(2).uniform(5, 30, hours),
            "so2": np.random.default_rng(3).uniform(1, 10, hours),
            "co": np.random.default_rng(4).uniform(0.2, 2.0, hours),
            "temperature": 20 + 5 * np.sin(2 * np.pi * ts.hour / 24),
            "humidity": np.random.default_rng(5).uniform(30, 90, hours),
            "dew_point": np.random.default_rng(6).uniform(5, 20, hours),
            "pressure": np.random.default_rng(7).uniform(990, 1020, hours),
            "wind_speed": np.random.default_rng(8).uniform(0, 10, hours),
            "wind_direction": np.random.default_rng(9).uniform(0, 360, hours),
            "cloud_cover": np.random.default_rng(10).uniform(0, 100, hours),
            "precipitation": np.zeros(hours),
        }
    )
    return df


def test_time_features_present():
    df = _synthetic_raw().set_index("timestamp")
    out = add_time_features(df)
    for c in ["hour", "day_of_week", "is_weekend", "hour_sin", "hour_cos"]:
        assert c in out.columns
    assert out["is_weekend"].isin([0, 1]).all()


def test_lag_features_no_leak():
    df = _synthetic_raw().set_index("timestamp")
    out = add_lag_features(df, ["aqi"], lags=(1, 3))
    # Row i should have lag_1h == df.aqi[i-1]
    for i in range(3, 20):
        assert out["aqi_lag_1h"].iloc[i] == df["aqi"].iloc[i - 1]
        assert out["aqi_lag_3h"].iloc[i] == df["aqi"].iloc[i - 3]


def test_rolling_features_use_only_past():
    df = _synthetic_raw().set_index("timestamp")
    out = add_rolling_features(df, ["aqi"], windows=(3,))
    # rolling mean at row i must equal mean of aqi[i-3:i], never include i itself
    for i in range(10, 20):
        expected = df["aqi"].iloc[i - 3 : i].mean()
        got = out["aqi_roll_3_mean"].iloc[i]
        assert abs(expected - got) < 1e-6, f"leak at row {i}"


def test_targets_shifted():
    df = _synthetic_raw().set_index("timestamp")
    out = add_targets(df, horizons=(24,))
    # target at row i must equal aqi at row i+24
    for i in range(50, 80):
        assert out["aqi_target_24h"].iloc[i] == df["aqi"].iloc[i + 24]


def test_build_features_end_to_end():
    df = _synthetic_raw(hours=300)
    out = build_features(df)
    assert not out.empty
    # No NaN targets after drop
    for c in ["aqi_target_24h", "aqi_target_48h", "aqi_target_72h"]:
        assert not out[c].isna().any()
    # Some engineered columns exist
    assert "hour_sin" in out.columns
    assert "aqi_lag_24h" in out.columns
    assert "aqi_roll_24_mean" in out.columns
