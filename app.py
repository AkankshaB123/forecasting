from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Forecasting MLOps | Forecast Review", page_icon="📈", layout="wide")

@st.cache_data
def sample_data(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2016-01-01", periods=n, freq="MS")
    trend = np.linspace(80, 180, n)
    seasonal = 15 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise = rng.normal(0, 5, n)
    return pd.DataFrame({"date": dates, "value": trend + seasonal + noise})

def metric_row(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    denom = np.sum(np.abs(actual))
    smape_denom = np.abs(actual) + np.abs(predicted)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "WAPE (%)": float(100 * np.sum(np.abs(actual - predicted)) / denom) if denom else float("nan"),
        "sMAPE (%)": float(np.mean(np.divide(200 * np.abs(actual-predicted), smape_denom,
                                               out=np.zeros_like(actual), where=smape_denom != 0))),
    }

def naive_forecast(train: pd.Series, steps: int) -> np.ndarray:
    return np.repeat(float(train.iloc[-1]), steps)

def fit_forecast(name: str, train: pd.Series, steps: int, seasonal_period: int) -> np.ndarray:
    if name == "Naive":
        return naive_forecast(train, steps)
    if name == "ARIMA":
        return np.asarray(ARIMA(train, order=(1, 1, 1)).fit().forecast(steps), dtype=float)
    if name == "SARIMA":
        period = max(2, min(int(seasonal_period), max(2, len(train) // 2)))
        return np.asarray(SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 0, period),
                                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False).forecast(steps), dtype=float)
    if name == "SARIMAX":
        return np.asarray(SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 0, max(2, seasonal_period)),
                                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False).forecast(steps), dtype=float)
    if name == "Prophet":
        from prophet import Prophet
        frame = pd.DataFrame({"ds": train.index, "y": train.values})
        model = Prophet(yearly_seasonality="auto", weekly_seasonality=False, daily_seasonality=False)
        model.fit(frame)
        future = pd.DataFrame({"ds": pd.date_range(train.index[-1], periods=steps + 1, freq=pd.infer_freq(train.index) or "D")[1:]})
        return model.predict(future)["yhat"].to_numpy(dtype=float)
    if name == "CNN":
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Conv1D, Dense, Flatten
        values = train.to_numpy(dtype=float).reshape(-1, 1)
        window = min(12, max(3, len(values) // 5))
        if len(values) <= window + 5:
            raise ValueError("CNN needs more history; use at least 20 observations.")
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(values).ravel()
        X = np.array([scaled[i-window:i] for i in range(window, len(scaled))]).reshape(-1, window, 1)
        y = scaled[window:]
        model = Sequential([Conv1D(16, 2, activation="relu", input_shape=(window, 1)), Flatten(), Dense(16, activation="relu"), Dense(1)])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=20, batch_size=16, verbose=0)
        history = list(scaled[-window:])
        preds = []
        for _ in range(steps):
            p = float(model.predict(np.array(history[-window:]).reshape(1, window, 1), verbose=0)[0, 0])
            preds.append(p)
            history.append(p)
        return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).ravel()
    raise ValueError(f"Unknown model: {name}")

st.title("📈 Forecasting MLOps — AI Forecast Review")
st.caption("Compare statistical and deep-learning forecasting candidates with chronological backtesting. Results are decision support, not a guarantee of future performance.")

with st.sidebar:
    st.header("Data & evaluation")
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    st.caption("CSV should contain a date/time column and a numeric target.")
    holdout = st.slider("Backtest holdout observations", min_value=3, max_value=36, value=12)
    horizon = st.slider("Future forecast horizon", min_value=1, max_value=24, value=12)
    seasonal_period = st.selectbox("Seasonal period", [4, 7, 12, 24, 52], index=2)
    selected_models = st.multiselect("Models to compare", ["Naive", "ARIMA", "SARIMA", "SARIMAX", "Prophet", "CNN"],
                                     default=["Naive", "ARIMA", "SARIMA", "SARIMAX", "Prophet"])

try:
    raw = pd.read_csv(uploaded) if uploaded is not None else sample_data()
    st.sidebar.success("Using uploaded data" if uploaded is not None else "Using embedded synthetic sample")
    if uploaded is not None:
        date_col = st.sidebar.selectbox("Date column", raw.columns, index=0)
        numeric_cols = raw.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.error("CSV must include at least one numeric target column.")
            st.stop()
        target_col = st.sidebar.selectbox("Target column", numeric_cols)
        raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
        raw[target_col] = pd.to_numeric(raw[target_col], errors="coerce")
        data = raw[[date_col, target_col]].dropna().rename(columns={date_col: "date", target_col: "value"})
    else:
        data = raw
    data = data.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    series = data["value"].astype(float)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if len(series) < max(12, holdout + 6):
        st.error(f"Need at least {max(12, holdout + 6)} valid observations for the selected holdout. Current rows: {len(series)}.")
        st.stop()
    freq = pd.infer_freq(series.index)
    if freq:
        series = series.asfreq(freq)
        series = series.interpolate(limit_direction="both")
except Exception as exc:
    st.error(f"Could not load the input data: {exc}")
    st.stop()

top = st.columns(3)
top[0].metric("Observations", f"{len(series):,}")
top[1].metric("Start", str(series.index.min().date()))
top[2].metric("End", str(series.index.max().date()))
with st.expander("Preview data"):
    st.dataframe(series.rename("value").to_frame().tail(20), use_container_width=True)

st.subheader("Historical series")
st.plotly_chart(px.line(series.rename("Observed").reset_index(), x="date", y="Observed", title="Observed target"), use_container_width=True)

if st.button("Run forecast review", type="primary", use_container_width=True):
    train, test = series.iloc[:-holdout], series.iloc[-holdout:]
    rows, backtest_frames, future_frames = [], [], []
    if not selected_models:
        st.warning("Select at least one model in the sidebar.")
        st.stop()
    with st.spinner("Fitting models and evaluating the chronological holdout…"):
        for name in selected_models:
            try:
                pred = fit_forecast(name, train, len(test), int(seasonal_period))
                if len(pred) != len(test) or not np.isfinite(pred).all():
                    raise ValueError("Model returned invalid predictions.")
                rows.append({"Model": name, **metric_row(test.values, pred), "Status": "OK"})
                backtest_frames.append(pd.DataFrame({"date": test.index, "series": "Actual", "value": test.values}))
                backtest_frames.append(pd.DataFrame({"date": test.index, "series": name, "value": pred}))
                future = fit_forecast(name, series, horizon, int(seasonal_period))
                future_dates = pd.date_range(series.index[-1], periods=horizon + 1, freq=pd.infer_freq(series.index) or "D")[1:]
                future_frames.append(pd.DataFrame({"date": future_dates, "model": name, "forecast": future}))
            except Exception as exc:
                rows.append({"Model": name, "MAE": np.nan, "RMSE": np.nan, "WAPE (%)": np.nan, "sMAPE (%)": np.nan,
                             "Status": f"Unavailable/error: {exc}"})
    metrics = pd.DataFrame(rows).sort_values("MAE", na_position="last")
    st.subheader("Backtest metrics")
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    successful = metrics[metrics["Status"] == "OK"]
    if not successful.empty:
        st.success(f"Best holdout MAE: {successful.iloc[0]['Model']} ({successful.iloc[0]['MAE']:.3f}). Compare other metrics and business costs before selecting a champion.")
    if backtest_frames:
        st.subheader("Holdout forecast comparison")
        st.plotly_chart(px.line(pd.concat(backtest_frames), x="date", y="value", color="series", title="Actual vs model predictions"), use_container_width=True)
    if future_frames:
        st.subheader("Future forecasts")
        future_df = pd.concat(future_frames)
        st.plotly_chart(px.line(future_df, x="date", y="forecast", color="model", title="Forecast horizon"), use_container_width=True)
        st.download_button("Download future forecasts CSV", future_df.to_csv(index=False), "forecasts.csv", "text/csv")
    st.download_button("Download metrics CSV", metrics.to_csv(index=False), "forecast_metrics.csv", "text/csv")
    st.info("Backtest uses the latest contiguous observations as a holdout. This starter uses fixed baseline orders and a single split; validate with rolling-origin backtests before production use.")
else:
    st.info("Choose your data and settings, then click **Run forecast review** to compare models.")
