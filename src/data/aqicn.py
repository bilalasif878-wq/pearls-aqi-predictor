"""AQICN (World Air Quality Index) API client.

Docs: https://aqicn.org/api/
Endpoints used:
  - GET /feed/{station}/?token=...   -> real-time reading for a station
  - GET /search/?keyword=...&token=... -> station discovery
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

BASE_URL = "https://api.waqi.info"


class AqicnError(RuntimeError):
    """AQICN returned status != ok, or the payload was unusable."""


@dataclass
class AqicnReading:
    aqi: float
    timestamp: datetime  # local station time
    pm25: float | None
    pm10: float | None
    o3: float | None
    no2: float | None
    so2: float | None
    co: float | None
    temperature: float | None
    humidity: float | None
    pressure: float | None
    wind_speed: float | None
    station_id: int
    station_name: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "aqi": self.aqi,
            "timestamp": self.timestamp,
            "pm25": self.pm25,
            "pm10": self.pm10,
            "o3": self.o3,
            "no2": self.no2,
            "so2": self.so2,
            "co": self.co,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "pressure": self.pressure,
            "wind_speed": self.wind_speed,
            "station_id": self.station_id,
            "station_name": self.station_name,
        }


def _iaqi(data: dict, key: str) -> float | None:
    v = data.get("iaqi", {}).get(key, {}).get("v")
    return float(v) if v is not None else None


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((requests.RequestException, AqicnError)),
    reraise=True,
)
def fetch_station(station: str, token: str, timeout: int = 20) -> AqicnReading:
    """Fetch the latest reading for one station.

    `station` accepts either "@1234" or a numeric idx. We pass it through verbatim
    to AQICN's /feed/{station}/ endpoint.
    """
    url = f"{BASE_URL}/feed/{station}/"
    resp = requests.get(url, params={"token": token}, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("status") != "ok":
        raise AqicnError(f"AQICN status={payload.get('status')} data={payload.get('data')}")

    data = payload["data"]
    ts_iso = data["time"].get("iso") or data["time"]["s"]
    ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))

    return AqicnReading(
        aqi=float(data["aqi"]) if data.get("aqi") not in (None, "-") else float("nan"),
        timestamp=ts,
        pm25=_iaqi(data, "pm25"),
        pm10=_iaqi(data, "pm10"),
        o3=_iaqi(data, "o3"),
        no2=_iaqi(data, "no2"),
        so2=_iaqi(data, "so2"),
        co=_iaqi(data, "co"),
        temperature=_iaqi(data, "t"),
        humidity=_iaqi(data, "h"),
        pressure=_iaqi(data, "p"),
        wind_speed=_iaqi(data, "w"),
        station_id=int(data.get("idx", 0)),
        station_name=data.get("city", {}).get("name", ""),
    )


def search_stations(keyword: str, token: str, timeout: int = 20) -> list[dict[str, Any]]:
    """Discover stations by city or place name."""
    url = f"{BASE_URL}/search/"
    resp = requests.get(url, params={"keyword": keyword, "token": token}, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "ok":
        raise AqicnError(f"AQICN status={payload.get('status')}")
    return payload.get("data", [])
