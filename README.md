# Pearls AQI Predictor

A serverless machine-learning system that forecasts the Air Quality Index for **Lahore, Pakistan** three days ahead. Data is pulled hourly from Open-Meteo, features and models are stored in Hopsworks, GitHub Actions handles scheduling, and the dashboard runs on Streamlit Community Cloud.

**Live dashboard:** https://pearls-aqi-lahore.streamlit.app

---

## What it does

- Ingests the current hour of air quality (PM2.5, PM10, O3, NO2, SO2, CO, US AQI) and weather (temperature, humidity, dew point, precipitation, wind, pressure, cloud cover) every hour.
- Backfilled with 400 days of history (9,600+ hourly rows).
- Retrains six candidate models per horizon every day and promotes the best.
- Shows the current AQI, a 3-day forecast with health alerts, a color-coded chart, and SHAP feature attributions for the top drivers.

## Architecture

Four independent pipelines communicating through Hopsworks:

```
                     +----------------------+
                     |  Open-Meteo (CAMS)   |
                     |  air quality + ERA5  |
                     +----------+-----------+
                                |
    +-------------+     hourly  |     +----------+
    |  Backfill   |------------>+     | Feature  |
    | (one-shot)  |             |     | pipeline |  <-- GH Actions cron '5 * * * *'
    +-------------+             v     +----+-----+
                     +----------------------+
                     |  Hopsworks feature   |
                     |    store  (HUDI)     |
                     +----------+-----------+
                                |
                                v
                     +----------------------+     +----------------+
                     |  Training pipeline   |---->| Hopsworks      |
                     |  6 models x 3 hzn    |     | model registry |
                     +----------------------+     +--------+-------+
                       <-- GH Actions daily                |
                                                           v
                                                   +----------------+
                                                   | Streamlit app  |
                                                   |  (public URL)  |
                                                   +----------------+
```

## Results

Test-set RMSE by horizon (lower is better). Ridge wins every horizon, LSTM comes last.

| Horizon | Best model | Test RMSE | Test R² | Beats persistence? |
|---------|------------|-----------|---------|--------------------|
| 24 h    | Ridge      | 18.02     | 0.20    | Yes |
| 48 h    | Ridge      | 19.94     | 0.08    | Yes |
| 72 h    | Ridge      | 21.74     | -0.07   | Yes (barely) |

Six candidates trained per horizon: persistence baseline, Ridge, Random Forest, XGBoost, LightGBM, PyTorch LSTM. Full comparison in `reports/model_comparison.csv` and in the project report.

## Tech stack

- **Data**: Open-Meteo (Copernicus CAMS air quality, ERA5 weather)
- **Storage**: Hopsworks feature store + model registry (free serverless tier)
- **Training**: scikit-learn, XGBoost, LightGBM, PyTorch
- **Explainability**: SHAP (LinearExplainer for the winning Ridge pipeline)
- **Dashboard**: Streamlit + Plotly
- **API**: FastAPI (defined, not deployed)
- **Automation**: GitHub Actions (hourly features, daily training)
- **Hosting**: Streamlit Community Cloud (dashboard), Hopsworks (state)

## Project layout

```
pearls-aqi-predictor/
|-- src/
|   |-- config.py                 # env + Streamlit secrets loading
|   |-- data/
|   |   |-- openmeteo.py          # air quality + weather clients
|   |   |-- aqicn.py              # AQICN client (kept for reference; unused live)
|   |   \-- health.py             # AQI category and health advice
|   |-- features/build.py         # time / lag / rolling / target engineering
|   |-- store/hopsworks_client.py # feature store + registry wrapper
|   |-- models/
|   |   |-- baseline.py           # persistence baseline
|   |   |-- lstm.py               # PyTorch LSTM regressor
|   |   |-- train.py              # candidate models + time-based split
|   |   |-- evaluate.py           # RMSE / MAE / R2
|   |   \-- register.py           # write best model to Hopsworks registry
|   \-- inference/predict.py      # 3-day forecast for the dashboard
|-- pipelines/
|   |-- feature_pipeline.py       # runs hourly on GitHub Actions
|   |-- backfill.py               # one-shot historical loader
|   \-- training_pipeline.py      # runs daily on GitHub Actions
|-- app/
|   |-- streamlit_app.py          # dashboard
|   |-- shap_panel.py             # SHAP explanation panel
|   \-- api.py                    # FastAPI /predict endpoint
|-- tests/                        # 5 unit tests for the feature builder
|-- .github/workflows/            # hourly + daily cron jobs
|-- scripts/                      # setup + station discovery
|-- reports/                      # model comparison CSV, report builder
\-- requirements.txt
```

## Running locally

```bash
git clone https://github.com/bilalasif878-wq/pearls-aqi-predictor.git
cd pearls-aqi-predictor
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env to add HOPSWORKS_API_KEY and HOPSWORKS_PROJECT
```

Smoke-test one hour of data:
```bash
python -m pipelines.feature_pipeline
```

Backfill 400 days of history (one-time):
```bash
python -m pipelines.backfill --days 400
```

Train models:
```bash
python -m pipelines.training_pipeline
```

Launch the dashboard locally:
```bash
streamlit run app/streamlit_app.py
```

## Automation

`.github/workflows/feature_pipeline.yml` runs every hour at HH:05 UTC.
`.github/workflows/training_pipeline.yml` runs daily at 02:30 UTC.

Both read Hopsworks credentials from repository secrets (`HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT`, `HOPSWORKS_HOST`).

## Tests

```bash
pytest tests/
```

Five tests cover time features, lag features (no leakage), rolling features (only past values), target shifting, and end-to-end feature build.

## Report

Full write-up including problem framing, data limitations, feature engineering, per-horizon results, SHAP interpretation, and honest limitations is at `Pearls_AQI_Predictor_Report.docx` (build it from `reports/build_report.js` with `node reports/build_report.js`).

## Author

Bilal Asif. Built as a submission for the Pearls AQI Predictor challenge.
