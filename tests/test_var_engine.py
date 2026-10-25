"""
Unit Tests for src.var_engine
"""

import numpy as np
import pandas as pd
import pytest

from src.var_engine import (
    compute_historical_var,
    compute_parametric_gaussian_var,
    compute_parametric_student_t_var,
    compute_cornish_fisher_var,
    compute_monte_carlo_var,
    run_comprehensive_var_suite,
)


@pytest.fixture
def synthetic_portfolio_returns():
    np.random.seed(42)
    # Heavy-tailed Student-t returns
    ret = np.random.standard_t(df=5, size=1000) * 0.01 + 0.0004
    return pd.Series(ret)


def test_historical_var_monotonicity(synthetic_portfolio_returns):
    var_90 = compute_historical_var(synthetic_portfolio_returns, 0.90)
    var_95 = compute_historical_var(synthetic_portfolio_returns, 0.95)
    var_99 = compute_historical_var(synthetic_portfolio_returns, 0.99)

    assert 0.0 < var_90 < var_95 < var_99, "VaR must strictly increase with confidence level"


def test_parametric_gaussian_var(synthetic_portfolio_returns):
    var_95 = compute_parametric_gaussian_var(synthetic_portfolio_returns, 0.95)
    var_99 = compute_parametric_gaussian_var(synthetic_portfolio_returns, 0.99)

    assert var_95 > 0.0
    assert var_99 > var_95


def test_monte_carlo_var():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="B")
    df_ret = pd.DataFrame({
        "A": np.random.normal(0.0005, 0.01, 200),
        "B": np.random.normal(0.0002, 0.005, 200),
    }, index=dates)

    weights = {"A": 0.5, "B": 0.5}
    mc_var, sim_paths = compute_monte_carlo_var(
        returns_df=df_ret,
        weights=weights,
        confidence_level=0.99,
        n_sims=5000,
    )

    assert mc_var > 0.0
    assert len(sim_paths) == 5000
