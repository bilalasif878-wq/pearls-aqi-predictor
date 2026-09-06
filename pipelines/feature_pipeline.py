"""Hourly feature pipeline.

Fetches the latest hour of pollutant + weather data, combines them, and appends
one row to the Hopsworks feature group. Designed to be idempotent — running it
twice in the same hour will simply overwrite the same (city, timestamp) key.

Run manually:
    python -m pipelines.feature_pipeline

Runs automatically via .github/workflows/feature_pipeline.yml every hour.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import aqicn_token, city_from_env
from src.data.aqicn import fetch_station
from src.data.openmeteo import fetch_weather_current
from src.features.build import build_features
from src.store.hopsworks_client import insert_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
logger = logging.getLogger("feature_pipeline")


def collect_current_row() -> pd.DataFrame:
    """Combine one AQICN reading + Open-Meteo current weather into a single row."""
    city = city_from_env()

    aqi = fetch_station(city.aqicn_station, aqicn_token())
    logger.info("AQICN reading: aqi=%s pm25=%s station=%s", aqi.aqi, aqi.pm25, aqi.station_name)

    weather = fetch_weather_current(city.lat, city.lon)
    logger.info("Weather rows fetched: %d", len(weather))

    # Snap AQICN timestamp down to the hour to align with Open-Meteo grid
    ts = pd.Timestamp(aqi.timestamp).tz_convert("UTC").floor("h")
    row = {
        "timestamp": ts,
        "aqi": aqi.aqi,
        "pm25": aqi.pm25,
        "pm10": aqi.pm10,
        "o3": aqi.o3,
        "no2": aqi.no2,
        "so2": aqi.so2,
        "co": aqi.co,
    }

    # Prefer Open-Meteo's weather (more complete) with AQICN as a fallback
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True).dt.floor("h")
    weather_row = weather[weather["timestamp"] == ts]
    if len(weather_row) == 1:
        for col in ["temperature", "humidity", "dew_point", "pressure",
                    "wind_speed", "wind_direction", "cloud_cover", "precipitation"]:
            if col in weather_row.columns:
                row[col] = float(weather_row.iloc[0][col])
    else:
        logger.warning("No Open-Meteo row aligned to %s; falling back to AQICN weather fields", ts)
        row.update({
            "temperature": aqi.temperature,
            "humidity": aqi.humidity,
            "pressure": aqi.pressure,
            "wind_speed": aqi.wind_speed,
        })

    return pd.DataFrame([row])


def main() -> int:
    city = city_from_env()
    logger.info("Feature pipeline start city=%s now=%s", city.name, datetime.now(timezone.utc))

    raw_row = collect_current_row()
    logger.info("Current-hour raw row:\n%s", raw_row.to_string(index=False))

    # Building features for a single row is meaningless (no history) — so we
    # write the raw row itself and let training-time feature engineering assemble
    # lag / rolling features from the accumulated history in the feature store.
    # (Feature engineering runs on the whole feature group when it's read back.)
    raw_row["city"] = city.name
    insert_features(raw_row, city.name)
    logger.info("Feature pipeline done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
