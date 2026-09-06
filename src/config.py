"""Central configuration. Reads from environment variables (via .env in local dev)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _required(name: str) -> str:
    val = os.environ.get(name)
    if not val or val.startswith("your_"):
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            f"Copy .env.example to .env and fill it in."
        )
    return val


@dataclass(frozen=True)
class CityConfig:
    name: str
    lat: float
    lon: float
    aqicn_station: str  # "@8574" or a numeric id string
    tz: str = "Asia/Karachi"


@dataclass(frozen=True)
class HopsworksConfig:
    api_key: str
    project: str
    host: str = "c.app.hopsworks.ai"


# Forecast horizons in hours
HORIZONS = (24, 48, 72)

# Rolling window sizes in hours (for lag/rolling features)
LAG_HOURS = (1, 3, 6, 12, 24, 48)
ROLLING_WINDOWS = (6, 12, 24)

# Feature group names in Hopsworks
FG_AQI_FEATURES = "aqi_features"
FG_AQI_FEATURES_VERSION = 1
FG_AQI_TARGETS = "aqi_targets"
FG_AQI_TARGETS_VERSION = 1

# Model naming
MODEL_NAME_TEMPLATE = "aqi_lahore_h{horizon}"


def city_from_env() -> CityConfig:
    return CityConfig(
        name=os.environ.get("CITY_NAME", "Lahore"),
        lat=float(os.environ.get("CITY_LAT", "31.5497")),
        lon=float(os.environ.get("CITY_LON", "74.3436")),
        aqicn_station=os.environ.get("AQICN_STATION", "@8574"),
    )


def hopsworks_from_env() -> HopsworksConfig:
    return HopsworksConfig(
        api_key=_required("HOPSWORKS_API_KEY"),
        project=_required("HOPSWORKS_PROJECT"),
        host=os.environ.get("HOPSWORKS_HOST", "c.app.hopsworks.ai"),
    )


def aqicn_token() -> str:
    return _required("AQICN_TOKEN")
