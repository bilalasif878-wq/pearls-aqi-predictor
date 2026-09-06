"""Hourly feature pipeline.

Pulls the latest hour of air quality + weather from Open-Meteo (CAMS global model
+ ERA5-derived weather) and writes one row to the Hopsworks feature group.

Design note — why not AQICN:
  AQICN's Lahore coverage is currently offline (US Embassy station stopped
  reporting Feb 2025). Open-Meteo's air-quality is the CAMS global reanalysis
  which gives consistent hourly coverage anywhere on Earth. Same source for
  live and historical means one schema, no timestamp alignment issues.

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

from src.config import city_from_env
from src.data.openmeteo import fetch_air_quality_current, fetch_weather_current
from src.store.hopsworks_client import insert_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
logger = logging.getLogger("feature_pipeline")

POLLUTANT_COLS = ["aqi", "pm25", "pm10", "ozone", "nitrogen_dioxide",
                  "sulphur_dioxide", "carbon_monoxide"]
WEATHER_COLS = ["temperature", "humidity", "dew_point", "pressure",
                "wind_speed", "wind_direction", "cloud_cover", "precipitation"]

# Rename Open-Meteo full names to short pollutant column names used everywhere else
POLLUTANT_RENAME = {
    "ozone": "o3",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "carbon_monoxide": "co",
}


def collect_current_row() -> pd.DataFrame:
    """Combine one hour of Open-Meteo air quality + weather into a single row."""
    city = city_from_env()

    aq = fetch_air_quality_current(city.lat, city.lon)
    logger.info("Air quality rows: %d", len(aq))
    wx = fetch_weather_current(city.lat, city.lon)
    logger.info("Weather rows: %d", len(wx))

    aq["timestamp"] = pd.to_datetime(aq["timestamp"], utc=True).dt.floor("h")
    wx["timestamp"] = pd.to_datetime(wx["timestamp"], utc=True).dt.floor("h")

    # Rename to canonical short names
    aq = aq.rename(columns=POLLUTANT_RENAME)

    merged = pd.merge(aq, wx, on="timestamp", how="inner")
    if merged.empty:
        raise RuntimeError("No overlapping hour between air quality and weather fetches.")

    # Take the most recent row that has non-null AQI
    merged = merged.sort_values("timestamp").dropna(subset=["aqi"])
    if merged.empty:
        raise RuntimeError("No non-null AQI row available from Open-Meteo.")
    row = merged.tail(1).copy()
    logger.info("Latest hour: %s  AQI=%.1f  PM2.5=%.1f",
                row["timestamp"].iloc[0], row["aqi"].iloc[0], row["pm25"].iloc[0])
    return row


def main() -> int:
    city = city_from_env()
    logger.info("Feature pipeline start city=%s now=%s", city.name, datetime.now(timezone.utc))

    row = collect_current_row()
    row["city"] = city.name
    logger.info("Row to insert:\n%s", row.to_string(index=False))

    insert_features(row, city.name)
    logger.info("Feature pipeline done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
