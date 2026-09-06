"""Save trained models to the Hopsworks model registry."""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

import joblib

from src.config import MODEL_NAME_TEMPLATE
from src.models.train import TrainedModel
from src.store.hopsworks_client import get_model_registry, get_project

logger = logging.getLogger(__name__)


def register_model(tm: TrainedModel) -> None:
    """Save the model + metadata to Hopsworks."""
    mr = get_model_registry()

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        joblib.dump(tm.estimator, d / "model.joblib")
        (d / "metadata.json").write_text(
            json.dumps(
                {
                    "horizon_h": tm.horizon,
                    "model_type": tm.name,
                    "feature_cols": tm.feature_cols,
                    "metrics_val": tm.metrics_val.as_dict(),
                    "metrics_test": tm.metrics_test.as_dict(),
                },
                indent=2,
            )
        )

        model_name = MODEL_NAME_TEMPLATE.format(horizon=tm.horizon)
        model = mr.python.create_model(
            name=model_name,
            metrics=tm.metrics_test.as_dict(),
            description=f"AQI predictor for Lahore, horizon={tm.horizon}h, algo={tm.name}",
        )
        model.save(str(d))
        logger.info("Registered model %s version=%s", model_name, getattr(model, "version", "?"))
