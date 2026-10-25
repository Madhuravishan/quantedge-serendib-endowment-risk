"""
Unit Tests for src.stress_test
"""

import numpy as np
import pandas as pd
import pytest

from src.stress_test import (
    compute_euler_risk_budgeting,
    evaluate_hypothetical_stagflation_shock,
    perform_reverse_stress_test,
)


@pytest.fixture
def synthetic_asset_returns():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=250, freq="B")
    return pd.DataFrame({
        "SPY": np.random.normal(0.0006, 0.012, 250),
        "AGG": np.random.normal(0.0001, 0.003, 250),
        "GLD": np.random.normal(0.0004, 0.009, 250),
    }, index=dates)


def test_euler_risk_budgeting_sum_property(synthetic_asset_returns):
    weights = {"SPY": 0.50, "AGG": 0.30, "GLD": 0.20}
    items, df, val = compute_euler_risk_budgeting(synthetic_asset_returns, weights, confidence_level=0.99)

    # Verify Euler Homogeneous summation: sum of percentage contributions = 100%
    assert val["Euler_Decomposition_Verified"] is True
    assert np.isclose(val["Sum_Percent_VaR"], 100.0, atol=1e-2)
    assert np.isclose(val["Sum_Percent_ES"], 100.0, atol=1e-2)


def test_hypothetical_stagflation_shock():
    weights = {"SPY": 0.25, "AGG": 0.35, "GLD": 0.15, "EFA": 0.10, "XLE": 0.10, "EEM": 0.05, "XLK": 0.0}
    res = evaluate_hypothetical_stagflation_shock(weights)

    # Serendib return: 0.25*(-0.20) + 0.35*(-0.12) + 0.15*(0.15) + 0.10*(-0.22) + 0.10*(0.25) + 0.05*(-0.25)
    # = -0.05 - 0.042 + 0.0225 - 0.022 + 0.025 - 0.0125 = -0.079 (-7.90%)
    assert np.isclose(res["portfolio_return"], -0.079, atol=1e-4)
    # Benchmark 60/40: 0.60*(-0.20) + 0.40*(-0.12) = -0.12 - 0.048 = -0.168 (-16.80%)
    assert np.isclose(res["benchmark_60_40_return"], -0.168, atol=1e-4)
    # Capital saved: (-0.079 - (-0.168)) * 10,000,000 = +$890,000
    assert np.isclose(res["capital_saved_usd"], 890_000.0, atol=1.0)


def test_reverse_stress_testing():
    weights = {"SPY": 0.25, "AGG": 0.35, "GLD": 0.15, "EFA": 0.10, "XLE": 0.10, "EEM": 0.05, "XLK": 0.0}
    rev = perform_reverse_stress_test(weights, loss_threshold_pct=0.15)

    assert rev["crash_multiplier_required"] > 1.0
    assert "critical_spy_drop" in rev
    assert "circuit-breaker is breached" in rev["board_defense_narrative"]
