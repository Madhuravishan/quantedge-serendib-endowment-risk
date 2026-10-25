"""
Unit Tests for src.backtest
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    run_kupiec_pof_test,
    run_christoffersen_independence_test,
    evaluate_basel_traffic_light,
)


def test_kupiec_pof_test_passing():
    # 250 days, 99% VaR, expected exceptions = 2.5.
    # 2 exceptions should comfortably pass
    res_pass = run_kupiec_pof_test(exceptions_count=2, total_observations=250, confidence_level=0.99)
    assert res_pass.verdict == "Pass"
    assert res_pass.p_value >= 0.05
    assert res_pass.lr_statistic < res_pass.critical_value_95


def test_kupiec_pof_test_failing():
    # 250 days, 99% VaR, 12 exceptions is an extreme failure
    res_fail = run_kupiec_pof_test(exceptions_count=12, total_observations=250, confidence_level=0.99)
    assert res_fail.verdict == "Fail"
    assert res_fail.p_value < 0.01


def test_christoffersen_independence_test():
    # Sequence with no clustering: alternating breaches spaced out
    series_no_cluster = pd.Series([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0] * 20)
    res_ind = run_christoffersen_independence_test(series_no_cluster, confidence_level=0.99)
    assert res_ind.indep_verdict == "Pass"

    # Sequence with heavy clustering: breaches back-to-back
    series_clustered = pd.Series([0, 0, 0, 1, 1, 1, 0, 0, 0, 0] * 20)
    res_clust = run_christoffersen_independence_test(series_clustered, confidence_level=0.99)
    # Consecutive breaches trigger lower independence p-value
    assert res_clust.n11 > 0


def test_basel_traffic_light_zones():
    # Green: 0 to 4 exceptions
    g_res = evaluate_basel_traffic_light(exceptions_count=3, window_days=250, confidence_level=0.99)
    assert g_res.zone == "GREEN"
    assert g_res.capital_multiplier == 3.00

    # Yellow: 5 to 9 exceptions
    y_res = evaluate_basel_traffic_light(exceptions_count=6, window_days=250, confidence_level=0.99)
    assert y_res.zone == "YELLOW"
    assert y_res.capital_multiplier == 3.50

    # Red: >= 10 exceptions
    r_res = evaluate_basel_traffic_light(exceptions_count=11, window_days=250, confidence_level=0.99)
    assert r_res.zone == "RED"
    assert r_res.capital_multiplier == 4.00
