"""FastAPI JSON endpoint for AQI predictions."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI

from src.config import city_from_env
from src.data.health import category_for
from src.inference.predict import predict_next_3_days

app = FastAPI(title="Pearls AQI Predictor API", version="0.1")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/predict")
def predict() -> dict:
    city = city_from_env()
    fc = predict_next_3_days()
    return {
        "city": city.name,
        "generated_at": fc["forecast_for"].min().isoformat(),
        "forecast": [
            {
                "horizon_h": int(r["horizon_h"]),
                "for": r["forecast_for"].isoformat(),
                "predicted_aqi": round(float(r["predicted_aqi"]), 1),
                "category": category_for(r["predicted_aqi"]).label,
                "model": r["model"],
            }
            for _, r in fc.iterrows()
        ],
    }
