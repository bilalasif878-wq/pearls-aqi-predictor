"""LSTM regressor that plugs into the sklearn-style training loop.

Reconstructs a short historical sequence from the lag features already produced
by src.features.build (values at t-48, t-24, t-12, t-6, t-3, t-1 and t itself),
then feeds it through a 1-layer LSTM and a small MLP head.

Kept intentionally small so it trains on CPU in a couple of minutes.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_OK = True
except Exception:  # pragma: no cover - handled in _candidates()
    _TORCH_OK = False

from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger(__name__)

# Sequence positions (oldest first). "0" means the current hour and is appended
# as the final timestep at inference time.
LAG_HOURS: Sequence[int] = (48, 24, 12, 6, 3, 1)

# Variables that have lag features in the engineered frame.
SEQ_VARS: tuple[str, ...] = (
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "wind_speed", "pressure",
)


class _LSTMNet(nn.Module if _TORCH_OK else object):
    def __init__(self, input_dim: int, hidden: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class LSTMForecaster(BaseEstimator, RegressorMixin):
    """A sklearn-compatible wrapper around a small PyTorch LSTM.

    The training loop expects fit(X, y) / predict(X) where X is the flat
    engineered feature dataframe (rows = hours, cols = features). We ignore
    everything except the {var}_lag_{h}h and {var} columns, from which we
    reconstruct a length-7 sequence per row.
    """

    def __init__(
        self,
        hidden: int = 64,
        num_layers: int = 1,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.2,
        device: str | None = None,
        random_state: int = 0,
    ):
        self.hidden = hidden
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.device = device
        self.random_state = random_state

    # ------------------------------------------------------------------
    def _sequences(self, X: pd.DataFrame) -> np.ndarray:
        """Build (n, seq_len, n_vars) tensor from the flat feature frame."""
        n = len(X)
        seq_len = len(LAG_HOURS) + 1  # + current hour
        n_vars = len(SEQ_VARS)
        out = np.zeros((n, seq_len, n_vars), dtype=np.float32)
        for vi, var in enumerate(SEQ_VARS):
            for ti, lag in enumerate(LAG_HOURS):
                col = f"{var}_lag_{lag}h"
                if col in X.columns:
                    out[:, ti, vi] = X[col].to_numpy(dtype=np.float32)
                elif var in X.columns:
                    out[:, ti, vi] = X[var].to_numpy(dtype=np.float32)
            if var in X.columns:
                out[:, -1, vi] = X[var].to_numpy(dtype=np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y):
        if not _TORCH_OK:
            raise RuntimeError("PyTorch not installed; cannot fit LSTMForecaster.")

        torch.manual_seed(self.random_state)
        device = self.device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        self.device_ = device

        seq = self._sequences(X)
        y_arr = np.asarray(y, dtype=np.float32)

        # Standardise
        self.x_mean_ = seq.mean(axis=(0, 1), keepdims=True)
        self.x_std_ = seq.std(axis=(0, 1), keepdims=True) + 1e-6
        self.y_mean_ = float(y_arr.mean())
        self.y_std_ = float(y_arr.std()) + 1e-6

        seq_n = (seq - self.x_mean_) / self.x_std_
        y_n = (y_arr - self.y_mean_) / self.y_std_

        ds = TensorDataset(torch.from_numpy(seq_n), torch.from_numpy(y_n))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.model_ = _LSTMNet(
            input_dim=len(SEQ_VARS),
            hidden=self.hidden,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        self.model_.train()
        for ep in range(self.epochs):
            losses = []
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                pred = self.model_(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            logger.info("  lstm epoch %d/%d mse=%.4f", ep + 1, self.epochs, float(np.mean(losses)))
        return self

    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        seq = self._sequences(X)
        seq_n = (seq - self.x_mean_) / self.x_std_
        self.model_.eval()
        with torch.no_grad():
            xb = torch.from_numpy(seq_n).to(self.device_)
            pred_n = self.model_(xb).squeeze(-1).cpu().numpy()
        return pred_n * self.y_std_ + self.y_mean_
