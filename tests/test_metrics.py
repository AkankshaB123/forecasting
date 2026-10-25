import numpy as np
import pandas as pd
import pytest

from forecasting.engine import metric_row, naive_forecast


def test_metric_row_exact_prediction():
    result = metric_row(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert result["MAE"] == 0
    assert result["RMSE"] == 0
    assert result["WAPE (%)"] == 0
    assert result["sMAPE (%)"] == 0


def test_metric_row_known_errors():
    result = metric_row(np.array([10.0, 20.0]), np.array([12.0, 16.0]))
    assert result["MAE"] == 3
    assert result["RMSE"] == pytest.approx(np.sqrt(10))
    assert result["WAPE (%)"] == pytest.approx(20)


def test_metric_row_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="equal shape"):
        metric_row(np.array([1, 2]), np.array([1]))


def test_naive_forecast_repeats_last_value():
    assert naive_forecast(pd.Series([2.0, 4.0, 7.0]), 3).tolist() == [7.0, 7.0, 7.0]


def test_naive_forecast_rejects_empty_series():
    with pytest.raises(ValueError, match="at least one"):
        naive_forecast(pd.Series(dtype=float), 1)
