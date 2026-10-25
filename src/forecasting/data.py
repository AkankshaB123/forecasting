"""Data preparation helpers for the forecasting dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sample_data(n: int = 120) -> pd.DataFrame:
    """Return a deterministic synthetic monthly time series."""
    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(42)
    dates = pd.date_range("2016-01-01", periods=n, freq="MS")
    trend = np.linspace(80, 180, n)
    seasonal = 15 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise = rng.normal(0, 5, n)
    return pd.DataFrame({"date": dates, "value": trend + seasonal + noise})


def prepare_series(frame: pd.DataFrame, date_col: str, target_col: str) -> pd.Series:
    """Validate and normalize a dataframe into a sorted datetime-indexed series."""
    missing = [col for col in (date_col, target_col) if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    data = frame[[date_col, target_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.dropna().rename(columns={date_col: "date", target_col: "value"})
    data = data.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    if data.empty:
        raise ValueError("No valid rows remain after parsing dates and numeric target values.")
    series = data["value"].astype(float).sort_index()
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError("Target values must be finite.")
    freq = pd.infer_freq(series.index) if len(series) >= 3 else None
    if freq:
        series = series.asfreq(freq).interpolate(limit_direction="both")
    return series
