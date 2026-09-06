"""Open-Meteo API client (weather + air quality, current + historical).

Open-Meteo is free and requires no API key. It gives us two things AQICN can't:
  1. Historical air quality (PM2.5, PM10, O3, NO2, SO2, CO, US AQI) going back years
     — via https://air-quality-api.open-meteo.com/v1/air-quality
  2. Historical weather (temperature, humidity, precip, wind, pressure) via the
     archive endpoint — https://archive-api.open-meteo.com/v1/archive
Both accept start_date / end_date and return hourly arrays.

We use openmeteo-requests with requests-cache + retry-requests for polite calls.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Iterable

import numpy as np
import openmeteo_requests
import pandas as pd
from pathlib import Path
import requests_cache
from retry_requests import retry

logger = logging.getLogger(__name__)

# Cached, retrying HTTP session shared by both endpoints
_cache_dir = Path.home() / ".cache" / "pearls_aqi"
_cache_dir.mkdir(parents=True, exist_ok=True)
_cache_session = requests_cache.CachedSession(str(_cache_dir / "openmeteo"), expire_after=3600)
_retry_session = retry(_cache_session, retries=5, backoff_factor=0.5)
_client = openmeteo_requests.Client(session=_retry_session)


AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

AIR_QUALITY_HOURLY = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
]

WEATHER_HOURLY = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "cloud_cover",
]


def _hourly_to_df(response, variables: Iterable[str]) -> pd.DataFrame:
    """Convert an openmeteo-requests hourly response into a tidy DataFrame."""
    hourly = response.Hourly()
    ts = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )
    data: dict[str, np.ndarray] = {"timestamp": ts}
    for i, name in enumerate(variables):
        data[name] = hourly.Variables(i).ValuesAsNumpy()
    return pd.DataFrame(data)


def fetch_air_quality_history(
    lat: float,
    lon: float,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Fetch hourly air-quality series for [start, end] inclusive."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": AIR_QUALITY_HOURLY,
        "timezone": "UTC",
    }
    responses = _client.weather_api(AIR_QUALITY_URL, params=params)
    df = _hourly_to_df(responses[0], AIR_QUALITY_HOURLY)
    df = df.rename(columns={"pm2_5": "pm25", "us_aqi": "aqi"})
    return df


def fetch_weather_history(
    lat: float,
    lon: float,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Fetch hourly weather series for [start, end] inclusive."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": WEATHER_HOURLY,
        "timezone": "UTC",
    }
    responses = _client.weather_api(ARCHIVE_URL, params=params)
    df = _hourly_to_df(responses[0], WEATHER_HOURLY)
    df = df.rename(
        columns={
            "temperature_2m": "temperature",
            "relative_humidity_2m": "humidity",
            "dew_point_2m": "dew_point",
            "wind_speed_10m": "wind_speed",
            "wind_direction_10m": "wind_direction",
            "surface_pressure": "pressure",
        }
    )
    return df


def fetch_weather_current(lat: float, lon: float) -> pd.DataFrame:
    """One row of the current hour's weather (used by the live feature pipeline)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": WEATHER_HOURLY,
        "past_hours": 1,
        "forecast_hours": 1,
        "timezone": "UTC",
    }
    responses = _client.weather_api(FORECAST_URL, params=params)
    df = _hourly_to_df(responses[0], WEATHER_HOURLY)
    df = df.rename(
        columns={
            "temperature_2m": "temperature",
            "relative_humidity_2m": "humidity",
            "dew_point_2m": "dew_point",
            "wind_speed_10m": "wind_speed",
            "wind_direction_10m": "wind_direction",
            "surface_pressure": "pressure",
        }
    )
    return df


def fetch_air_quality_current(lat: float, lon: float) -> pd.DataFrame:
    """One row (or two) of the current hour's air quality (used by the live feature pipeline)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": AIR_QUALITY_HOURLY,
        "past_hours": 1,
        "forecast_hours": 1,
        "timezone": "UTC",
    }
    responses = _client.weather_api(AIR_QUALITY_URL, params=params)
    df = _hourly_to_df(responses[0], AIR_QUALITY_HOURLY)
    df = df.rename(columns={"pm2_5": "pm25", "us_aqi": "aqi"})
    return df
