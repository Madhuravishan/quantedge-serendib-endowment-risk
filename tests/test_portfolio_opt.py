"""
Unit Tests for src.portfolio_opt
"""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_opt import (
    compute_portfolio_return_series,
    compute_drawdown_series,
    calculate_portfolio_kpis,
    optimize_markowitz_portfolio,
    compute_efficient_frontier,
)


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    ret_spy = np.random.normal(0.0005, 0.012, 252)
    ret_agg = np.random.normal(0.0001, 0.003, 252)
    return pd.DataFrame({"SPY": ret_spy, "AGG": ret_agg}, index=dates)


def test_compute_portfolio_return_series(sample_returns):
    w = {"SPY": 0.60, "AGG": 0.40}
    p_ret = compute_portfolio_return_series(sample_returns, w)

    expected = 0.60 * sample_returns["SPY"] + 0.40 * sample_returns["AGG"]
    np.testing.assert_allclose(p_ret.values, expected.values, atol=1e-7)


def test_compute_drawdown_series():
    # +10%, -20%, +10%
    ret = pd.Series([0.10, -0.20, 0.10])
    wealth, dd = compute_drawdown_series(ret)

    # Day 0: 1.10, peak 1.10, dd 0
    # Day 1: 1.10 * 0.80 = 0.88, peak 1.10, dd = (0.88 - 1.10) / 1.10 = -0.20
    # Day 2: 0.88 * 1.10 = 0.968, peak 1.10, dd = (0.968 - 1.10) / 1.10 = -0.12
    assert np.isclose(wealth.iloc[1], 0.88, atol=1e-5)
    assert np.isclose(dd.iloc[1], -0.20, atol=1e-5)
    assert np.isclose(dd.min(), -0.20, atol=1e-5)


def test_markowitz_optimization(sample_returns):
    # Min volatility optimization
    w_arr, w_dict = optimize_markowitz_portfolio(sample_returns, target="min_volatility")
    assert np.isclose(np.sum(w_arr), 1.0, atol=1e-4)
    assert np.all(w_arr >= -1e-5)
    # AGG has lower volatility so its weight should exceed SPY's
    assert w_dict["AGG"] > w_dict["SPY"]

    # Max Sharpe optimization
    w_arr_s, w_dict_s = optimize_markowitz_portfolio(sample_returns, target="max_sharpe")
    assert np.isclose(np.sum(w_arr_s), 1.0, atol=1e-4)


def test_calculate_portfolio_kpis(sample_returns):
    w = {"SPY": 0.50, "AGG": 0.50}
    p_ret = compute_portfolio_return_series(sample_returns, w)
    kpis = calculate_portfolio_kpis(
        p_ret,
        weights=w,
        name="TestPortfolio",
        benchmark_series=sample_returns["SPY"],
    )

    assert kpis.name == "TestPortfolio"
    assert kpis.annual_volatility > 0.0
    assert kpis.max_drawdown <= 0.0
    assert 0.0 <= kpis.beta <= 2.0
