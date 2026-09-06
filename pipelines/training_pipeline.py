"""Daily training pipeline.

Reads all features from Hopsworks, retrains every candidate model for every
forecast horizon, picks the best-by-test-RMSE per horizon, and registers it.

Run manually:
    python -m pipelines.training_pipeline

Runs daily via .github/workflows/training_pipeline.yml.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.register import register_model
from src.models.train import pick_best, results_table, train_all
from src.store.hopsworks_client import read_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
logger = logging.getLogger("training_pipeline")


def main() -> int:
    logger.info("Reading features from Hopsworks…")
    raw = read_features()
    logger.info("Got %d rows, %d cols", len(raw), raw.shape[1])

    results = train_all(raw)
    table = results_table(results)
    logger.info("Results table:\n%s", table.to_string(index=False))

    out_dir = Path(__file__).resolve().parent.parent / "reports"
    out_dir.mkdir(exist_ok=True)
    table.to_csv(out_dir / "model_comparison.csv", index=False)
    logger.info("Saved reports/model_comparison.csv")

    best = pick_best(results)
    for horizon, tm in sorted(best.items()):
        logger.info("Best for %dh: %s  test_rmse=%.3f", horizon, tm.name, tm.metrics_test.rmse)
        register_model(tm)

    logger.info("Training pipeline done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
