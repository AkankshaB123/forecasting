"""Forecast models and evaluation metrics, independent of the Streamlit UI."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

MODEL_NAMES = ["Naive", "ARIMA", "SARIMA", "SARIMAX", "Prophet", "CNN"]


def metric_row(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("actual and predicted must be non-empty arrays of equal shape")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("actual and predicted values must be finite")
    denom = np.sum(np.abs(actual))
    smape_denom = np.abs(actual) + np.abs(predicted)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "WAPE (%)": float(100 * np.sum(np.abs(actual - predicted)) / denom) if denom else float("nan"),
        "sMAPE (%)": float(np.mean(np.divide(200 * np.abs(actual - predicted), smape_denom,
            out=np.zeros_like(actual), where=smape_denom != 0))),
    }


def naive_forecast(train: pd.Series, steps: int) -> np.ndarray:
    if len(train) == 0:
        raise ValueError("train must contain at least one observation")
    if steps < 1:
        raise ValueError("steps must be positive")
    return np.repeat(float(train.iloc[-1]), steps)


def fit_forecast(name: str, train: pd.Series, steps: int, seasonal_period: int = 12) -> np.ndarray:
    if len(train) < 3:
        raise ValueError("At least 3 training observations are required")
    if steps < 1:
        raise ValueError("steps must be positive")
    if name == "Naive":
        return naive_forecast(train, steps)
    if name == "ARIMA":
        return np.asarray(ARIMA(train, order=(1, 1, 1)).fit().forecast(steps), dtype=float)
    if name in {"SARIMA", "SARIMAX"}:
        period = max(2, min(int(seasonal_period), max(2, len(train) // 2)))
        seasonal_order = (1, 1, 0, period) if name == "SARIMA" else (1, 1, 0, max(2, int(seasonal_period)))
        return np.asarray(SARIMAX(train, order=(1, 1, 1), seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False).forecast(steps), dtype=float)
    if name == "Prophet":
        from prophet import Prophet
        frame = pd.DataFrame({"ds": pd.to_datetime(train.index), "y": train.to_numpy(dtype=float)})
        model = Prophet(yearly_seasonality="auto", weekly_seasonality=False, daily_seasonality=False)
        model.fit(frame)
        freq = pd.infer_freq(train.index) or "D"
        future = pd.DataFrame({"ds": pd.date_range(train.index[-1], periods=steps + 1, freq=freq)[1:]})
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
        history, preds = list(scaled[-window:]), []
        for _ in range(steps):
            pred = float(model.predict(np.array(history[-window:]).reshape(1, window, 1), verbose=0)[0, 0])
            preds.append(pred)
            history.append(pred)
        return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).ravel()
    raise ValueError(f"Unknown model: {name}")
