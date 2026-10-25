# Forecasting MLOps — AI Forecast Review

A production-oriented starting point for time-series model comparison, evaluation, and CI/CD. The Streamlit interface is inspired by forecast-review workbenches: historical series, model forecast overlays, metric comparison, and downloadable forecast tables.

> **Status:** Starter implementation. Validate dependencies, model suitability, security, data contracts, and deployment configuration against your own workload before production use. No model is guaranteed to win on every dataset.

## Capabilities

- Streamlit forecast review dashboard with uploadable CSV, configurable columns, frequency, horizon, and backtest holdout.
- Common model interface for **Naive baseline, ARIMA, XGBoost, Prophet, CNN (optional TensorFlow), and AutoML champion selection**.
- Chronological holdout evaluation with MAE, RMSE, WAPE, and sMAPE.
- Forecast overlays, evaluation table, error reporting for unavailable models, and CSV exports.
- GitHub Actions CI for supported Python versions, linting, unit tests, and Docker image build.
- Non-root Docker runtime and health check.

## Architecture

```text
CSV / source adapter
       |
       v
Validation + frequency normalization
       |
       v
Model adapters ----> chronological backtest ----> metrics
       |                                           |
       v                                           v
Full-history refit ----------------------> Forecast Review UI
                                                   |
                                      CSV export / deployment
```

## Quick start

Requires Python 3.10–3.12. Prophet and XGBoost may require platform-specific build/runtime dependencies.

```bash
git clone https://github.com/AkankshaB123/forecasting.git
cd forecasting
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal. Upload a CSV with at least 12 valid observations and columns such as:

```csv
date,value
2024-01-01,120.4
2024-02-01,124.8
2024-03-01,131.2
```

Select the date/target columns, frequency, forecast horizon, and holdout length, then click **Run forecast review**. The app includes a synthetic sample for UI smoke testing; do not treat synthetic output as a real business forecast.

## Model behavior

| Model | Implementation | Notes |
|---|---|---|
| Naive | Last observation repeated | Required baseline; always compare more complex models against it |
| ARIMA | statsmodels ARIMA(1,1,1) | Baseline order is intentionally simple; tune order/seasonality per dataset |
| XGBoost | Lag-window autoregression | Recursive multi-step predictions can accumulate error; add calendar/exogenous features when appropriate |
| Prophet | Trend and automatic calendar seasonality | Seasonal components need enough observations; tune holidays and changepoints for the use case |
| CNN | Small 1D convolutional neural network | Optional TensorFlow dependency; needs more history and compute than simple baselines |
| AutoML | Selects lowest holdout MAE among successfully fitted candidates | This is a transparent baseline selector, not a managed AutoML service; use rolling-origin CV and tuning for stronger selection |

Model libraries are imported lazily. If an optional library is missing or a model fails, the dashboard displays that model's error instead of fabricating predictions. Install TensorFlow separately to enable CNN:

```bash
pip install "tensorflow>=2.15,<3"
```

## Forecasting process

1. **Define the target.** Identify the business quantity to forecast and the grain (for example, demand per SKU per day). Avoid mixing entities or units in one univariate series.
2. **Validate the input.** Parse dates and numeric target values, sort timestamps, aggregate duplicate timestamps by mean, and reject missing required columns. Internal gaps may be interpolated only when frequency is known; confirm this policy is appropriate for your business.
3. **Set frequency explicitly when needed.** Supported examples include `D` daily, `W` weekly, `MS` month-start, `QS` quarter-start, and `h` hourly. Do not infer a frequency from irregular timestamps if a reliable business calendar exists.
4. **Choose a time-aware split.** The holdout is the latest contiguous block of observations. Random train/test splits are inappropriate for ordinary forecasting because they can leak future information into training.
5. **Fit on training history.** Each candidate predicts the holdout without training on the holdout observations.
6. **Evaluate candidates.** Compare multiple error measures and inspect forecast plots and residuals. MAE is interpretable in target units; RMSE penalizes large misses; WAPE scales total absolute error by total actual magnitude; sMAPE is a symmetric percentage-style measure. Percentage metrics can be unstable around zero.
7. **Select a champion.** Current AutoML selects the eligible candidate with lowest holdout MAE. In a production rollout, use multiple rolling-origin folds, business-weighted metrics, and a minimum improvement threshold before replacing the current champion.
8. **Refit on all available observations.** After model comparison, the chosen model can be refit on all history to generate the requested future horizon.
9. **Review and publish.** Export forecasts and metrics. Record the dataset version, configuration, code commit, package versions, training timestamp, and evaluation results with each production run.

## Evaluation metrics

For actuals \(y_t\) and predictions \(\hat y_t\):

- **MAE:** mean of \(|y_t - \hat y_t|\).
- **RMSE:** square root of mean squared error.
- **WAPE:** \(100 \times \sum |y_t-\hat y_t| / \sum |y_t|\). Undefined when the denominator is zero.
- **sMAPE:** mean of \(200|y_t-\hat y_t|/(|y_t|+|\hat y_t|)\), with zero-denominator terms handled as zero in this implementation.

Do not choose a model on a single metric alone. Compare against naive/seasonal-naive baselines and report metrics by forecast horizon, product/region, and time window when relevant. Forecast intervals should be calibrated from backtest errors rather than assumed to be reliable.

## CI/CD

GitHub Actions runs on pushes to `main`, feature/fix branches, pull requests, and manual dispatch:

1. Set up Python 3.10, 3.11, and 3.12.
2. Install dependencies.
3. Run Ruff lint checks.
4. Run unit tests.
5. Build the Docker image without publishing it.

To enable CI, commit `.github/workflows/ci.yml` to the repository. The workflow uses read-only repository permissions. Pin third-party GitHub Actions to reviewed commit SHAs according to your organization's supply-chain policy.

### Deployment

The included Dockerfile runs as a non-root user and exposes Streamlit on port 8501. Build and run locally:

```bash
docker build -t forecasting-mlops .
docker run --rm -p 8501:8501 forecasting-mlops
```

Deployment is intentionally provider-neutral. For production, add a protected deployment environment and short-lived cloud credentials/OIDC, publish images to a registry, scan dependencies/images, deploy only after CI passes, and retain a rollback to the previous image. The sample deploy workflow is a template, not an active deployment.

## Production hardening checklist

### Data and modeling
- [ ] Enforce a documented data contract, timezone, business calendar, target units, and missing-data policy.
- [ ] Support multiple series with entity keys and separate model selection per series or segment.
- [ ] Add rolling-origin cross-validation and horizon-wise evaluation.
- [ ] Tune model parameters on training folds only; never tune on the final test set.
- [ ] Add seasonal-naive and domain-specific baselines.
- [ ] Persist prediction intervals and validate empirical interval coverage.
- [ ] Detect outliers and data drift without automatically deleting legitimate spikes.
- [ ] Add model explainability and residual diagnostics appropriate to model type.

### MLOps and reliability
- [ ] Persist model artifacts and metadata in a versioned artifact store (for example, MLflow plus object storage).
- [ ] Track dataset fingerprints, code commit, config, library versions, run ID, and metric history.
- [ ] Add scheduled retraining, champion/challenger evaluation, approval gates, and rollback.
- [ ] Monitor forecast error after actuals arrive, missing data, drift, latency, and failed runs.
- [ ] Add structured logs, health checks, retry/timeouts, and alerting.
- [ ] Add authentication/authorization, secret management, TLS, and upload size limits before exposing the app publicly.
- [ ] Add integration tests with representative datasets and dependency/security scans.
- [ ] Establish a model review process and business-owner approval for high-impact forecasts.

### Known limitations of this starter
- It is a univariate dashboard and does not yet implement entity-level batch forecasting, exogenous regressors, a persistent model registry, automated retraining scheduler, or calibrated prediction intervals.
- AutoML uses one chronological holdout rather than full rolling-origin cross-validation; use rolling-origin validation before production champion promotion.
- ARIMA parameters are fixed, and CNN is an optional minimal adapter. Production use should tune models for the target data and workload.
- The dashboard runs training in the Streamlit process. For larger workloads, move training/inference to a job queue or API service and keep Streamlit as a review client.
- A successful Docker build and CI run do not by themselves certify the system as production-ready.

## Repository layout

```text
.
├── app.py
├── src/forecasting/
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

## Troubleshooting

- **Cannot infer frequency:** choose a frequency explicitly in the sidebar and verify the timestamps follow it.
- **Model unavailable:** install the named optional dependency; the dashboard lists import or fitting errors.
- **Not enough observations:** reduce the holdout or gather more history. Deep learning and seasonal models generally need substantially more data.
- **Unrealistic results:** check target scaling, duplicate aggregation, gaps, outliers, seasonality, leakage, and whether the holdout resembles the deployment period.

## License and data handling

Add the license appropriate for your project. Do not upload sensitive or personally identifiable data to an unapproved deployment. Review your organization's data-retention and access policies before operating the dashboard.
