"""Thin Hopsworks feature-store + model-registry wrapper."""
from __future__ import annotations

import logging
from functools import lru_cache

import hopsworks
import pandas as pd

from src.config import (
    FG_AQI_FEATURES,
    FG_AQI_FEATURES_VERSION,
    hopsworks_from_env,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_project():
    cfg = hopsworks_from_env()
    logger.info("Logging into Hopsworks project=%s host=%s", cfg.project, cfg.host)
    project = hopsworks.login(
        api_key_value=cfg.api_key,
        project=cfg.project,
        host=cfg.host,
    )
    return project


def get_feature_store():
    return get_project().get_feature_store()


def get_model_registry():
    return get_project().get_model_registry()


def get_or_create_features_fg():
    """Create the AQI features feature group if it doesn't exist."""
    fs = get_feature_store()
    try:
        fg = fs.get_feature_group(name=FG_AQI_FEATURES, version=FG_AQI_FEATURES_VERSION)
        return fg
    except Exception:
        pass
    fg = fs.create_feature_group(
        name=FG_AQI_FEATURES,
        version=FG_AQI_FEATURES_VERSION,
        description="Hourly AQI + weather features and forecast targets for Lahore.",
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        online_enabled=True,
    )
    return fg


def insert_features(df: pd.DataFrame, city: str) -> None:
    """Insert a feature dataframe into the Hopsworks feature group."""
    df = df.copy()
    if "city" not in df.columns:
        df["city"] = city
    if df.index.name == "timestamp":
        df = df.reset_index()
    fg = get_or_create_features_fg()
    fg.insert(df, write_options={"wait_for_job": True})
    logger.info("Inserted %d rows into feature group %s v%d", len(df), FG_AQI_FEATURES, FG_AQI_FEATURES_VERSION)


def read_features() -> pd.DataFrame:
    """Read the full history back for training."""
    fs = get_feature_store()
    fg = fs.get_feature_group(name=FG_AQI_FEATURES, version=FG_AQI_FEATURES_VERSION)
    return fg.read()
