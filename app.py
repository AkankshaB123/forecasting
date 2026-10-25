from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from forecasting.data import prepare_series, sample_data
from forecasting.engine import MODEL_NAMES, fit_forecast, metric_row

st.set_page_config(page_title="Forecasting MLOps | Forecast Review", page_icon="📈", layout="wide")
st.title("📈 Forecasting MLOps — AI Forecast Review")
st.caption("Compare statistical and deep-learning forecasting candidates with chronological backtesting. Results are decision support, not a guarantee of future performance.")

with st.sidebar:
    st.header("Data & evaluation")
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    st.caption("CSV should contain a date/time column and a numeric target.")
    holdout = st.slider("Backtest holdout observations", min_value=3, max_value=36, value=12)
    horizon = st.slider("Future forecast horizon", min_value=1, max_value=24, value=12)
    seasonal_period = st.selectbox("Seasonal period", [4, 7, 12, 24, 52], index=2)
    selected_models = st.multiselect("Models to compare", MODEL_NAMES,
        default=["Naive", "ARIMA", "SARIMA", "SARIMAX", "Prophet"])

try:
    if uploaded is None:
        series = prepare_series(sample_data(), "date", "value")
        st.sidebar.success("Using embedded synthetic sample")
    else:
        raw = pd.read_csv(uploaded)
        if raw.empty:
            st.error("The uploaded CSV is empty.")
            st.stop()
        date_col = st.sidebar.selectbox("Date column", raw.columns, index=0)
        numeric_cols = raw.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.error("CSV must include at least one numeric target column.")
            st.stop()
        target_col = st.sidebar.selectbox("Target column", numeric_cols)
        series = prepare_series(raw, date_col, target_col)
        st.sidebar.success("Using uploaded data")
    if len(series) < max(12, holdout + 6):
        st.error(f"Need at least {max(12, holdout + 6)} valid observations for the selected holdout. Current rows: {len(series)}.")
        st.stop()
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
st.plotly_chart(px.line(series.rename("Observed").rename_axis("date").reset_index(),
    x="date", y="Observed", title="Observed target"), use_container_width=True)

if st.button("Run forecast review", type="primary", use_container_width=True):
    if not selected_models:
        st.warning("Select at least one model in the sidebar.")
        st.stop()
    train, test = series.iloc[:-holdout], series.iloc[-holdout:]
    rows, backtest_frames, future_frames = [], [], []
    with st.spinner("Fitting models and evaluating the chronological holdout…"):
        for name in selected_models:
            try:
                pred = fit_forecast(name, train, len(test), int(seasonal_period))
                if len(pred) != len(test) or not np.isfinite(pred).all():
                    raise ValueError("Model returned invalid predictions.")
                rows.append({"Model": name, **metric_row(test.to_numpy(), pred), "Status": "OK"})
                backtest_frames.extend([
                    pd.DataFrame({"date": test.index, "series": "Actual", "value": test.to_numpy()}),
                    pd.DataFrame({"date": test.index, "series": name, "value": pred}),
                ])
                future = fit_forecast(name, series, horizon, int(seasonal_period))
                freq = pd.infer_freq(series.index) or "D"
                future_dates = pd.date_range(series.index[-1], periods=horizon + 1, freq=freq)[1:]
                future_frames.append(pd.DataFrame({"date": future_dates, "model": name, "forecast": future}))
            except Exception as exc:
                rows.append({"Model": name, "MAE": np.nan, "RMSE": np.nan,
                    "WAPE (%)": np.nan, "sMAPE (%)": np.nan,
                    "Status": f"Unavailable/error: {exc}"})
    metrics = pd.DataFrame(rows).sort_values("MAE", na_position="last")
    st.subheader("Backtest metrics")
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    successful = metrics[metrics["Status"] == "OK"]
    if not successful.empty:
        st.success(f"Best holdout MAE: {successful.iloc[0]['Model']} ({successful.iloc[0]['MAE']:.3f}). Compare other metrics and business costs before selecting a champion.")
    if backtest_frames:
        st.subheader("Holdout forecast comparison")
        st.plotly_chart(px.line(pd.concat(backtest_frames), x="date", y="value",
            color="series", title="Actual vs model predictions"), use_container_width=True)
    if future_frames:
        st.subheader("Future forecasts")
        future_df = pd.concat(future_frames)
        st.plotly_chart(px.line(future_df, x="date", y="forecast", color="model",
            title="Forecast horizon"), use_container_width=True)
        st.download_button("Download future forecasts CSV", future_df.to_csv(index=False),
            "forecasts.csv", "text/csv")
    st.download_button("Download metrics CSV", metrics.to_csv(index=False),
        "forecast_metrics.csv", "text/csv")
    st.info("Backtest uses the latest contiguous observations as a holdout. This starter uses fixed baseline orders and a single split; validate with rolling-origin backtests before production use.")
else:
    st.info("Choose your data and settings, then click **Run forecast review** to compare models.")
