# Pearls AQI Predictor

Serverless machine-learning pipeline that forecasts Air Quality Index (AQI) for **Lahore, Pakistan** three days into the future.

## Architecture

Four independent pipelines connected only through a shared Hopsworks feature store and model registry.

1. **Feature pipeline** (`pipelines/feature_pipeline.py`) — pulls the latest hour from AQICN and Open-Meteo, computes features, writes to Hopsworks. Runs hourly via GitHub Actions.
2. **Backfill** (`pipelines/backfill.py`) — same feature computation over the last 1–2 years of historical hourly data.
3. **Training pipeline** (`pipelines/training_pipeline.py`) — reads historical features, trains multiple models per forecast horizon (24 h / 48 h / 72 h), registers the best in Hopsworks. Runs daily.
4. **Web app** (`app/streamlit_app.py`) — loads the latest features and models from Hopsworks and shows a 3-day forecast, health alert, and SHAP explanation.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
# fill AQICN_TOKEN, HOPSWORKS_API_KEY, HOPSWORKS_PROJECT in .env
python pipelines/feature_pipeline.py  # single-hour smoke test
```

## Deployment

- Automation: GitHub Actions (`.github/workflows/`)
- Dashboard: Streamlit Community Cloud
- Feature store & model registry: Hopsworks

## Report

See `reports/report.md` for the full write-up: problem framing, data, features, EDA, model comparison, SHAP analysis, and limitations.
