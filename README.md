# Forecasting MLOps — AI Forecast Review

A Streamlit dashboard for comparing time-series forecasting models, reviewing chronological holdout metrics, plotting forecasts, and exporting results. Includes a GitHub Actions CI workflow and a Docker image build.

## 🚀 Streamlit dashboard

**[Deploy / open this app on Streamlit Community Cloud](https://share.streamlit.io/deploy?repository=AkankshaB123%2Fforecasting&branch=main&mainModule=app.py)**

This link opens Streamlit Community Cloud's deployment flow for this repository. The public `.streamlit.app` URL is assigned by Streamlit after the app is deployed; it cannot be truthfully linked until that deployment has been completed. Once deployed, add the assigned public URL here as the live dashboard link.

## Features

- Upload a CSV or try the embedded synthetic sample data.
- Select date and target columns; configure holdout length, forecast horizon, and seasonal period.
- Compare Naive, ARIMA, SARIMA, SARIMAX, Prophet, and optional CNN models.
- Review MAE, RMSE, WAPE, and sMAPE on a chronological holdout.
- Plot historical data, holdout predictions, and future forecasts.
- Download forecast and metric CSVs.
- GitHub Actions checks Python syntax, runs tests, and builds a Docker image.

## Quick start

Python 3.11 or 3.12 is recommended.

```bash
git clone https://github.com/AkankshaB123/forecasting.git
cd forecasting
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The dashboard starts with a synthetic monthly series. To use your own data, upload a CSV containing a date/time column and a numeric target, for example:

```csv
date,value
2024-01-01,120.4
2024-02-01,124.8
2024-03-01,131.2
```

Choose the appropriate columns and settings in the sidebar, then select **Run forecast review**. Use synthetic data only to smoke-test the UI, not for business decisions.

## Models

| Model | Implementation | Notes |
|---|---|---|
| Naive | Repeats the last training observation | Baseline for comparison |
| ARIMA | statsmodels ARIMA(1,1,1) | Fixed starter order; tune for the series |
| SARIMA | Seasonal ARIMA via statsmodels | Uses the selected seasonal period |
| SARIMAX | statsmodels SARIMAX with seasonal terms | No external regressors are configured in this starter |
| Prophet | Prophet trend and automatic yearly seasonality | Optional runtime dependency; needs sufficient history |
| CNN | Small 1D convolutional neural network | Optional TensorFlow dependency; install with `pip install "tensorflow>=2.15,<3"` |

Optional models that fail to import or fit are reported in the metrics table instead of being assigned fabricated results. The current model parameters are baselines, not automatically tuned production models.

## Evaluation and workflow

The latest contiguous observations are held out from training. Metrics are calculated only on this holdout:

- **MAE**: mean absolute error, in target units.
- **RMSE**: root mean squared error, penalizes larger misses.
- **WAPE**: weighted absolute percentage error; unstable when actual totals are near zero.
- **sMAPE**: symmetric percentage-style error; interpret carefully around zero.

The current dashboard uses one chronological holdout and fixed starter configurations. Before production use, add rolling-origin cross-validation, seasonal-naive baselines, business-specific metrics, data-quality contracts, drift monitoring, and model/version tracking. Do not select a production model using a single score alone.

### CI/CD

Workflow: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

On pushes to `main`, feature/fix branches, pull requests to `main`, and manual dispatch, GitHub Actions installs dependencies, checks Python syntax, runs tests, and builds the Docker image. This workflow does **not** publish a container image or deploy the live app automatically.

### Docker

```bash
docker build -t forecasting-mlops .
docker run --rm -p 8501:8501 forecasting-mlops
```

Open http://localhost:8501. The container runs as a non-root user and includes a Streamlit health check.

## Repository layout

```text
.
├── app.py
├── requirements.txt
├── Dockerfile
├── .streamlit/config.toml
├── tests/test_metrics.py
└── .github/workflows/ci.yml
```

## Limitations and data handling

- Univariate starter dashboard; no persistent model registry, scheduled retraining, calibrated prediction intervals, or external regressors are configured.
- Forecast quality depends on data quality, seasonal structure, and parameter choices. Validate with representative historical backtests before acting on predictions.
- Do not upload sensitive or personally identifiable data to an unapproved public app. Configure access controls and review your organization's data-retention requirements before deployment.
- Add the license appropriate for your project.
