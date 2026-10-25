"""
Unit Tests for src.evt_engine
"""

import numpy as np
import pandas as pd
import pytest

from src.evt_engine import (
    fit_gpd_pot,
    compute_evt_risk_metrics,
    compute_mean_excess_function,
    compute_hill_estimator,
)


@pytest.fixture
def fat_tailed_returns():
    np.random.seed(42)
    # Generate 1500 returns with extreme student-t tail (df=3)
    r = np.random.standard_t(df=3, size=1500) * 0.012 + 0.0003
    return pd.Series(r)


def test_fit_gpd_pot(fat_tailed_returns):
    fit_res, excesses = fit_gpd_pot(fat_tailed_returns, threshold_percentile=0.95)

    assert fit_res.threshold_percentile == 0.95
    assert fit_res.exceedances_count_nu > 20
    assert fit_res.scale_sigma > 0.0
    # For Student-t df=3, shape xi should be positive (Fréchet fat tail)
    assert fit_res.shape_xi > 0.0
    assert "Heavy-Tailed" in fit_res.tail_type
    assert len(excesses) == fit_res.exceedances_count_nu


def test_compute_evt_risk_metrics(fat_tailed_returns):
    m_95, _ = compute_evt_risk_metrics(fat_tailed_returns, confidence_level=0.95)
    m_99, _ = compute_evt_risk_metrics(fat_tailed_returns, confidence_level=0.99)

    assert m_99.evt_var > m_95.evt_var
    assert m_99.evt_es >= m_99.evt_var, "EVT Expected Shortfall must be >= EVT VaR"
    assert m_99.evt_var_usd > 0.0


def test_diagnostics_computation(fat_tailed_returns):
    losses = -fat_tailed_returns.values
    mef_df = compute_mean_excess_function(losses)
    hill_df = compute_hill_estimator(losses)

    assert not mef_df.empty
    assert "Mean_Excess" in mef_df.columns
    assert not hill_df.empty
    assert "Hill_Xi" in hill_df.columns
