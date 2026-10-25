"""
Unit Tests for src.data_loader
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    clean_price_data,
    compute_returns,
    run_adf_stationarity_test,
)


def test_clean_price_data_governance():
    # Synthetic test data with duplicates and missing values
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    df = pd.DataFrame({
        "SPY": [100.0, 101.0, np.nan, 102.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
        "AGG": [50.0, 50.1, 50.2, 50.3, 50.4, 50.5, 50.6, 50.7, 50.8, 50.9],
    }, index=dates)

    clean_df, audit = clean_price_data(df)
    assert not clean_df.isnull().values.any(), "Cleaned price matrix must contain no NaNs"
    assert len(clean_df) == len(df), "Forward fill should preserve observation count"
    assert audit["initial_rows"] == 10


def test_compute_returns_properties():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    df = pd.DataFrame({
        "SPY": [100.0, 110.0, 121.0, 108.9, 119.79],
    }, index=dates)

    log_ret, simple_ret = compute_returns(df)
    assert len(log_ret) == 4
    assert len(simple_ret) == 4

    # Verify first simple return: (110 - 100) / 100 = 0.10
    assert np.isclose(simple_ret["SPY"].iloc[0], 0.10, atol=1e-5)
    # Verify first log return: ln(110 / 100) = ln(1.10)
    assert np.isclose(log_ret["SPY"].iloc[0], np.log(1.10), atol=1e-5)


def test_run_adf_stationarity_test():
    # 1. Test on stationary white noise (should reject null -> Stationary I(0))
    np.random.seed(42)
    stationary_series = pd.Series(np.random.normal(0, 1, 500))
    res_stat = run_adf_stationarity_test(stationary_series)

    assert res_stat["is_stationary"] is True
    assert 0.0 <= res_stat["p_value"] <= 1.0
    assert res_stat["p_value"] < 0.05
    assert res_stat["conclusion"] == "Stationary I(0)"
    assert res_stat["adf_statistic"] < res_stat["crit_5pct"]
    assert "crit_1pct" in res_stat
    assert "crit_5pct" in res_stat
    assert "crit_10pct" in res_stat
    # Critical values must be monotonically increasing (more negative at 1%)
    assert res_stat["crit_1pct"] < res_stat["crit_5pct"] < res_stat["crit_10pct"] < 0.0

    # 2. Test on non-stationary random walk / cumulative sum (should fail to reject null -> Non-Stationary I(1))
    np.random.seed(42)
    non_stationary_series = pd.Series(np.cumsum(np.random.normal(0.001, 0.02, 500)) + 100.0)
    res_non_stat = run_adf_stationarity_test(non_stationary_series)

    assert 0.0 <= res_non_stat["p_value"] <= 1.0
    assert res_non_stat["is_stationary"] is False
    assert res_non_stat["p_value"] > 0.05
    assert res_non_stat["conclusion"] == "Non-Stationary I(1)"
    assert res_non_stat["adf_statistic"] > res_non_stat["crit_5pct"]

