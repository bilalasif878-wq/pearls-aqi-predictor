"""Persistence baseline: predict future AQI = current AQI.

Any real model must beat this or we're wasting compute.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class PersistenceRegressor(BaseEstimator, RegressorMixin):
    """Ignores everything except the `aqi` column and predicts it verbatim."""

    def __init__(self, aqi_col: str = "aqi"):
        self.aqi_col = aqi_col

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.aqi_col].to_numpy()
