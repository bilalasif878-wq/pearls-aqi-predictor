"""Quick verification that AQICN token works and returns real Lahore data."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import aqicn_token, city_from_env
from src.data.aqicn import fetch_station, search_stations
from src.data.health import category_for


def main() -> int:
    token = aqicn_token()
    city = city_from_env()

    print(f"Testing AQICN for {city.name} (station={city.aqicn_station})")
    print(f"Token: {token[:6]}…{token[-4:]}  ({len(token)} chars)")
    print()

    # 1. Search
    print("--- Nearby stations from search ---")
    stations = search_stations(city.name, token)
    for s in stations[:8]:
        uid = s.get("uid")
        name = s.get("station", {}).get("name")
        aqi = s.get("aqi")
        print(f"  @{uid:<8}  aqi={aqi:<5}  {name}")
    print()

    # 2. Fetch the configured station
    print(f"--- Live reading for {city.aqicn_station} ---")
    r = fetch_station(city.aqicn_station, token)
    cat = category_for(r.aqi)
    print(f"  Station:     {r.station_name} (id={r.station_id})")
    print(f"  Timestamp:   {r.timestamp}")
    print(f"  AQI:         {r.aqi}  ({cat.label})")
    print(f"  PM2.5:       {r.pm25}")
    print(f"  PM10:        {r.pm10}")
    print(f"  O3:          {r.o3}")
    print(f"  NO2:         {r.no2}")
    print(f"  Temp:        {r.temperature}")
    print(f"  Humidity:    {r.humidity}")
    print(f"  Wind:        {r.wind_speed}")
    print()
    print("AQICN connectivity OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
