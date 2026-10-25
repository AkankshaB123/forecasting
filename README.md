# Forecasting MLOps — AI Forecast Review

A modular Streamlit dashboard for comparing time-series forecasting models, reviewing chronological holdout metrics, plotting forecasts, and exporting results. GitHub Actions validates code and tests on pull requests and pushes, then builds a Docker image in CI.

## Live dashboard

**[Open the live forecasting dashboard](https://akankshab123-forecasting-app-cd6xcn.streamlit.app/)**

## Features

- Upload a CSV or try deterministic synthetic monthly sample data.
- Select date and target columns; configure holdout length, forecast horizon, and seasonal period.
- Compare Naive, ARIMA, SARIMA, SARIMAX, Prophet, and optional CNN models.
- Review MAE, RMSE, WAPE, and sMAPE on a chronological holdout.
- Plot history, holdout predictions, and future forecasts; download CSV outputs.
- Automated CI for Python 3.11/3.12, unit tests, and Docker image build.

## Repository structure

```text
.
├── app.py
├── src/forecasting/
│   ├── __init__.py
│   ├── data.py
│   └── engine.py
├── tests/
│   ├── test_data.py
│   └── test_metrics.py
├── .github/workflows/
│   ├── ci.yml
│   └── deploy.yml.example
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

## Quick start

Python 3.11 or 3.12 is recommended.

```bash
git clone https://github.com/AkankshaB123/forecasting.git
cd forecasting
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt -e .
streamlit run app.py
```

Upload a CSV containing a date/time column and a numeric target, e.g.:

```csv
date,value
2024-01-01,120.4
2024-02-01,124.8
2024-03-01,131.2
```

The data preparation layer parses dates and numeric values, drops invalid rows, sorts observations, and keeps the last row for duplicate dates. The application requires enough observations for the selected holdout.

## Models and evaluation

| Model | Implementation | Notes |
|---|---|---|
| Naive | Repeats the last training observation | Baseline |
| ARIMA | statsmodels ARIMA(1,1,1) | Fixed starter order |
| SARIMA | Seasonal SARIMAX model | Selected seasonal period, bounded by history length |
| SARIMAX | statsmodels SARIMAX with seasonal terms | No external regressors configured |
| Prophet | Prophet trend and automatic yearly seasonality | Optional at runtime; needs sufficient history |
| CNN | Small 1D convolutional network | Optional TensorFlow dependency; install with `pip install "tensorflow>=2.15,<3"` |

Metrics are evaluated on the latest chronological holdout: MAE and RMSE are in target units; WAPE and sMAPE are percentages and need careful interpretation when actuals are near zero. Model parameters are baselines, not automatically tuned production models. Use rolling-origin backtests and business-specific metrics before production decisions.

## CI/CD

- **CI**: `.github/workflows/ci.yml` runs on pull requests and pushes. It installs the package, compiles Python files, runs unit tests across Python 3.11 and 3.12, and builds a Docker image.
- **CD example**: `.github/workflows/deploy.yml.example` is intentionally disabled as a workflow because GitHub Actions only executes YAML files ending in `.yml` or `.yaml`. To use it with a compatible hosting provider, copy it to `.github/workflows/deploy.yml`, add a repository Actions secret named `DEPLOY_HOOK_URL`, and adjust the trigger/provider as needed.
- The deployment example triggers a hosting-provider webhook only after a successful CI run on `main` (or manually). It does not deploy the Streamlit Community Cloud app automatically; that service normally deploys from its connected GitHub repository.

## Docker

```bash
docker build -t forecasting-app .
docker run --rm -p 8501:8501 forecasting-app
```

Open http://localhost:8501. The image uses a non-root user and includes a Streamlit health check.

## Limitations and data handling

This is a starter dashboard: no model registry, scheduled retraining, drift monitoring, prediction intervals, or external regressors are configured. Validate with representative data before acting on forecasts. Do not upload sensitive or personally identifiable data to an unapproved public app; review access controls and retention requirements before deployment.
