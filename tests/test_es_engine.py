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


def test_subadditivity_verification():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=500, freq="B")
    r1 = pd.Series(np.random.normal(0, 0.015, 500), index=dates)
    r2 = pd.Series(np.random.normal(0, 0.008, 500), index=dates)

    axioms = verify_subadditivity_properties(r1, r2, confidence_level=0.99)
    assert axioms["Subadditivity"]["ES_Holds"] is True
    assert "Strictly Coherent" in axioms["Coherent_Measure_Verdict"]["ES"]
