import pandas as pd
import pytest

from forecasting.data import prepare_series, sample_data


def test_sample_data_is_deterministic_and_monthly():
    first = sample_data(24)
    second = sample_data(24)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 24
    assert first["date"].is_monotonic_increasing


def test_prepare_series_sorts_deduplicates_and_drops_invalid_rows():
    frame = pd.DataFrame({"when": ["2024-02-01", "2024-01-01", "2024-02-01", "bad"],
                          "sales": [20, 10, 25, 99]})
    series = prepare_series(frame, "when", "sales")
    assert series.index.is_monotonic_increasing
    assert len(series) == 2
    assert series.iloc[-1] == 25


def test_prepare_series_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing required"):
        prepare_series(pd.DataFrame({"date": []}), "date", "value")


def test_prepare_series_rejects_all_invalid_rows():
    with pytest.raises(ValueError, match="No valid rows"):
        prepare_series(pd.DataFrame({"date": ["bad"], "value": [1]}), "date", "value")
