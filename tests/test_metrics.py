import numpy as np
from app import metric_row, naive_forecast
import pandas as pd

def test_metric_row_exact_prediction():
    result = metric_row(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert result["MAE"] == 0
    assert result["RMSE"] == 0
    assert result["WAPE (%)"] == 0

def test_naive_forecast_repeats_last_value():
    train = pd.Series([2.0, 4.0, 7.0])
    assert naive_forecast(train, 3).tolist() == [7.0, 7.0, 7.0]
