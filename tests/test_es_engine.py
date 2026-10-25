"""
Unit Tests for src.es_engine
"""

import numpy as np
import pandas as pd
import pytest

from src.es_engine import (
    compute_historical_es,
    compute_parametric_gaussian_es,
    compute_parametric_student_t_es,
    verify_subadditivity_properties,
)
from src.var_engine import (
    compute_parametric_gaussian_var,
    compute_parametric_student_t_var,
)


@pytest.fixture
def return_series():
    np.random.seed(42)
    # 1000 days of Student-t fat-tailed returns
    r = np.random.standard_t(df=4, size=1000) * 0.01 + 0.0003
    return pd.Series(r)


def test_expected_shortfall_greater_than_var(return_series):
    es_95, var_95 = compute_historical_es(return_series, confidence_level=0.95)
    es_99, var_99 = compute_historical_es(return_series, confidence_level=0.99)

    assert es_95 >= var_95, "Expected Shortfall must always be >= Value-at-Risk"
    assert es_99 >= var_99
    assert es_99 > es_95, "ES must be monotonic with confidence level"
    # Verify core risk monotonicity: ES_99 >= VaR_99 >= VaR_95
    assert es_99 >= var_99 >= var_95, "Risk monotonicity violated: must have ES_99 >= VaR_99 >= VaR_95"


def test_risk_monotonicity_hierarchy(return_series):
    """
    Verify fundamental risk measure monotonicity across confidence tiers:
    ES_99 >= VaR_99 >= VaR_95 >= VaR_90 and ES_99 >= ES_95 >= ES_90
    across Historical, Parametric Gaussian, and Student-t models.
    """
    # 1. Historical Simulation
    es_90_h, var_90_h = compute_historical_es(return_series, 0.90)
    es_95_h, var_95_h = compute_historical_es(return_series, 0.95)
    es_99_h, var_99_h = compute_historical_es(return_series, 0.99)

    assert es_99_h >= var_99_h >= var_95_h >= var_90_h
    assert es_99_h >= es_95_h >= es_90_h

    # 2. Parametric Gaussian
    es_90_g = compute_parametric_gaussian_es(return_series, 0.90)
    var_90_g = compute_parametric_gaussian_var(return_series, 0.90)
    es_95_g = compute_parametric_gaussian_es(return_series, 0.95)
    var_95_g = compute_parametric_gaussian_var(return_series, 0.95)
    es_99_g = compute_parametric_gaussian_es(return_series, 0.99)
    var_99_g = compute_parametric_gaussian_var(return_series, 0.99)

    assert es_99_g >= var_99_g >= var_95_g >= var_90_g
    assert es_99_g >= es_95_g >= es_90_g

    # 3. Parametric Student-t
    es_90_t = compute_parametric_student_t_es(return_series, 0.90)
    var_90_t, _ = compute_parametric_student_t_var(return_series, 0.90)
    es_95_t = compute_parametric_student_t_es(return_series, 0.95)
    var_95_t, _ = compute_parametric_student_t_var(return_series, 0.95)
    es_99_t = compute_parametric_student_t_es(return_series, 0.99)
    var_99_t, _ = compute_parametric_student_t_var(return_series, 0.99)

    assert es_99_t >= var_99_t >= var_95_t >= var_90_t
    assert es_99_t >= es_95_t >= es_90_t




def test_subadditivity_verification():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=500, freq="B")
    r1 = pd.Series(np.random.normal(0, 0.015, 500), index=dates)
    r2 = pd.Series(np.random.normal(0, 0.008, 500), index=dates)

    axioms = verify_subadditivity_properties(r1, r2, confidence_level=0.99)
    assert axioms["Subadditivity"]["ES_Holds"] is True
    assert "Strictly Coherent" in axioms["Coherent_Measure_Verdict"]["ES"]
