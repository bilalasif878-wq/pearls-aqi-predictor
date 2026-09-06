"""Print AQICN stations for a given search keyword.

    python scripts/discover_stations.py lahore
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import aqicn_token
from src.data.aqicn import search_stations


def main() -> int:
    keyword = sys.argv[1] if len(sys.argv) > 1 else "lahore"
    stations = search_stations(keyword, aqicn_token())
    print(f"{len(stations)} stations matching '{keyword}':\n")
    print(f"{'@UID':<10} {'AQI':<6} STATION")
    print("-" * 60)
    for s in stations:
        uid = s.get("uid", "?")
        aqi = s.get("aqi", "?")
        name = s.get("station", {}).get("name", "?")
        print(f"@{uid:<9} {str(aqi):<6} {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
