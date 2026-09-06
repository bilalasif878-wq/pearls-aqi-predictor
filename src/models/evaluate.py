"""Metrics + evaluation helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class Metrics:
    rmse: float
    mae: float
    r2: float

    def as_dict(self) -> dict[str, float]:
        return {"rmse": self.rmse, "mae": self.mae, "r2": self.r2}


def compute_metrics(y_true, y_pred) -> Metrics:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return Metrics(
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
    )
