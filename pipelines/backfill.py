"""Historical backfill.

Uses Open-Meteo's air-quality + archive endpoints to pull hourly pollutant and
weather series for a date range, joins them, and bulk-inserts into the same
Hopsworks feature group the live pipeline writes to.

Run:
    python -m pipelines.backfill --days 400
    python -m pipelines.backfill --start 2024-09-01 --end 2026-09-01

Backfill takes a while — Open-Meteo returns everything in a couple of API calls
but Hopsworks insertion of ~10k rows can run several minutes.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import city_from_env
from src.data.openmeteo import fetch_air_quality_history, fetch_weather_history

POLLUTANT_RENAME = {
    "ozone": "o3",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "carbon_monoxide": "co",
}
from src.store.hopsworks_client import insert_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
logger = logging.getLogger("backfill")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill AQI + weather history")
    p.add_argument("--start", type=lambda s: date.fromisoformat(s), help="YYYY-MM-DD")
    p.add_argument("--end", type=lambda s: date.fromisoformat(s), help="YYYY-MM-DD")
    p.add_argument("--days", type=int, default=400,
                   help="If --start/--end not given, pull the last N days")
    p.add_argument("--dry-run", action="store_true",
                   help="Only fetch + join; do not write to Hopsworks")
    p.add_argument("--out", type=str, default=None,
                   help="Optional parquet path to save the assembled dataframe")
    return p.parse_args()


def chunked_ranges(start: date, end: date, chunk_days: int = 90):
    """Yield (chunk_start, chunk_end) pairs of at most chunk_days each."""
    cur = start
    while cur <= end:
        cend = min(end, cur + timedelta(days=chunk_days - 1))
        yield cur, cend
        cur = cend + timedelta(days=1)


def fetch_range(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """Fetch and merge air quality + weather for a single chunk."""
    logger.info("Fetching %s -> %s", start, end)
    aq = fetch_air_quality_history(lat, lon, start, end)
    wx = fetch_weather_history(lat, lon, start, end)
    aq = aq.rename(columns=POLLUTANT_RENAME)
    merged = pd.merge(aq, wx, on="timestamp", how="inner")
    logger.info("  aq=%d wx=%d merged=%d", len(aq), len(wx), len(merged))
    return merged


def main() -> int:
    args = parse_args()
    city = city_from_env()

    if args.start and args.end:
        start, end = args.start, args.end
    else:
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=args.days)

    logger.info("Backfilling %s: %s -> %s", city.name, start, end)

    frames = []
    for cs, ce in chunked_ranges(start, end, chunk_days=90):
        frames.append(fetch_range(city.lat, city.lon, cs, ce))
    full = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
    logger.info("Total rows assembled: %d (%s -> %s)",
                len(full), full["timestamp"].min(), full["timestamp"].max())

    if args.out:
        full.to_parquet(args.out, index=False)
        logger.info("Saved parquet to %s", args.out)

    if args.dry_run:
        logger.info("Dry-run — not writing to Hopsworks.")
        print(full.head().to_string())
        print("...")
        print(full.tail().to_string())
        return 0

    full["city"] = city.name
    insert_features(full, city.name)
    logger.info("Backfill complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
