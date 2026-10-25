"""
Layer 5: Model Validation & Basel Regulatory Backtesting Engine
Mandate: The Serendib University Endowment Fund (USD 10,000,000 AUM)
QuantEdge 1.0 National STEM Competition - Team 'Pandas Predictors'

Key Validation Frameworks:
1. Out-of-Sample Window Split:
   - In-sample calibration: 2006 to 2023-12-31
   - Out-of-sample backtesting: 2024-01-01 to 2026-10-20 (or rolling 250/500-day window)
2. Statistical Coverage Tests:
   - Kupiec's Proportion of Failures (POF) Likelihood Ratio Test (Unconditional coverage, chi2(1))
   - Christoffersen's Independence Test (Breach clustering, Markov chain transitions, chi2(1))
   - Christoffersen Combined Conditional Coverage Test (chi2(2))
3. Basel Committee Traffic Light Framework:
   - Regulatory benchmark: 250 trading days, 99% 1-day VaR
   - Green Zone (0-4 exceptions): Model accepted, capital multiplier k = 3.00
   - Yellow Zone (5-9 exceptions): Supervisory review, capital multiplier k = 3.40 - 3.85
   - Red Zone (10+ exceptions): Model definitively rejected, capital multiplier k = 4.00
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
from scipy.stats import chi2, binom

from src.config import (
    BASEL_WINDOW_DAYS,
    BASEL_CONFIDENCE_LEVEL,
    BASEL_ZONES,
    IN_SAMPLE_TRAIN_END,
    OUT_OF_SAMPLE_START,
)
from src.var_engine import (
    compute_historical_var,
    compute_parametric_gaussian_var,
    compute_parametric_student_t_var,
)
from src.evt_engine import compute_evt_risk_metrics

logger = logging.getLogger(__name__)


@dataclass
class KupiecTestResult:
    """Kupiec POF test statistical outputs."""
    total_observations: int
    exceptions_count: int
    expected_exceptions: float
    failure_rate: float
    nominal_p: float
    lr_statistic: float
    p_value: float
    critical_value_95: float  # 3.841
    verdict: str              # 'Pass' (Fail to reject H0) or 'Fail' (Reject H0)


@dataclass
class ChristoffersenTestResult:
    """Christoffersen Independence and Conditional Coverage test outputs."""
    n00: int
    n01: int
    n10: int
    n11: int
    pi_01: float
    pi_11: float
    pi_uncond: float
    lr_indep_statistic: float
    p_value_indep: float
    indep_verdict: str         # 'Pass' (Independent) or 'Fail' (Clustered)
    lr_cc_statistic: float
    p_value_cc: float
    cc_verdict: str            # 'Pass' or 'Fail'


@dataclass
class BaselTrafficLightResult:
    """Basel Committee on Banking Supervision Traffic Light categorization."""
    window_days: int
    exceptions_count: int
    zone: str                 # 'GREEN', 'YELLOW', 'RED'
    status: str
    capital_multiplier: float
    description: str
    color: str
    cumulative_binom_prob: float


@dataclass
class ModelValidationReport:
    """Consolidated model validation and audit report."""
    model_name: str
    confidence_level: float
    kupiec: KupiecTestResult
    christoffersen: ChristoffersenTestResult
    basel: BaselTrafficLightResult
    exceptions_dates: List[pd.Timestamp]

    def to_dict(self) -> Dict[str, any]:
        return {
            "Model Name": self.model_name,
            "Confidence": f"{int(self.confidence_level * 100)}%",
            "Obs / Exceptions": f"{self.kupiec.total_observations} / {self.kupiec.exceptions_count} (Exp: {self.kupiec.expected_exceptions:.1f})",
            "Kupiec POF p-val": f"{self.kupiec.p_value:.4f} ({self.kupiec.verdict})",
            "Independence p-val": f"{self.christoffersen.p_value_indep:.4f} ({self.christoffersen.indep_verdict})",
            "Conditional Coverage": f"{self.christoffersen.p_value_cc:.4f} ({self.christoffersen.cc_verdict})",
            "Basel Traffic Light": f"{self.basel.zone} ({self.basel.status})",
            "Capital Multiplier": f"{self.basel.capital_multiplier:.2f}x",
        }


def run_kupiec_pof_test(
    exceptions_count: int,
    total_observations: int,
    confidence_level: float = 0.99,
) -> KupiecTestResult:
    """
    Perform Kupiec's Proportion of Failures (POF) Likelihood Ratio Test.
    
    Hypothesis:
    H_0 : p = p_0 = 1 - alpha (Unconditional coverage matches target)
    H_1 : p != p_0
    
    Likelihood Ratio:
    LR_POF = -2 * ln[ ((1 - p_0)^(N - x) * p_0^x) / ((1 - x/N)^(N - x) * (x/N)^x) ]
    Under H0, LR_POF ~ Chi-Square(1). Critical value at 95% is 3.841.
    """
    N = int(total_observations)
    x = int(exceptions_count)
    p0 = 1.0 - confidence_level
    p_hat = x / N if N > 0 else 0.0

    if x == 0:
        # Edge case: 0 exceptions
        # LR = -2 * ln( (1 - p0)^N ) = -2 * N * ln(1 - p0)
        lr_stat = -2.0 * N * np.log(1.0 - p0)
    elif x == N:
        lr_stat = -2.0 * N * np.log(p0)
    else:
        log_num = (N - x) * np.log(1.0 - p0) + x * np.log(p0)
        log_den = (N - x) * np.log(1.0 - p_hat) + x * np.log(p_hat)
        lr_stat = -2.0 * (log_num - log_den)

    lr_stat = max(0.0, float(lr_stat))
    p_val = float(1.0 - chi2.cdf(lr_stat, df=1))
    crit_val = float(chi2.ppf(0.95, df=1))  # 3.841

    verdict = "Pass" if p_val >= 0.05 else "Fail"

    return KupiecTestResult(
        total_observations=N,
        exceptions_count=x,
        expected_exceptions=N * p0,
        failure_rate=p_hat,
        nominal_p=p0,
        lr_statistic=lr_stat,
        p_value=p_val,
        critical_value_95=crit_val,
        verdict=verdict,
    )


def run_christoffersen_independence_test(
    breach_indicator_series: pd.Series,
    confidence_level: float = 0.99,
) -> ChristoffersenTestResult:
    """
    Perform Christoffersen's Interval Forecast Test for Independence of Breaches.
    
    Tests whether VaR exceptions cluster together in time using a 2-state Markov chain:
    - n_00: No breach followed by no breach
    - n_01: No breach followed by breach
    - n_10: Breach followed by no breach
    - n_11: Breach followed by breach (Clustering!)
    
    LR_ind = -2 * ln[ L(pi) / L(pi_01, pi_11) ] ~ Chi-Square(1)
    """
    clean_series = breach_indicator_series.dropna().astype(int).values
    n = len(clean_series)

    if n < 2:
        return ChristoffersenTestResult(
            n00=0, n01=0, n10=0, n11=0, pi_01=0, pi_11=0, pi_uncond=0,
            lr_indep_statistic=0, p_value_indep=1.0, indep_verdict="Pass",
            lr_cc_statistic=0, p_value_cc=1.0, cc_verdict="Pass"
        )

    # Count Markov transitions
    lagged = clean_series[:-1]
    current = clean_series[1:]

    n00 = int(np.sum((lagged == 0) & (current == 0)))
    n01 = int(np.sum((lagged == 0) & (current == 1)))
    n10 = int(np.sum((lagged == 1) & (current == 0)))
    n11 = int(np.sum((lagged == 1) & (current == 1)))

    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi_uncond = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0.0

    # Independence Likelihood Ratio
    # If n11 == 0 or n01 == 0, avoid log(0)
    eps = 1e-12
    if (n01 == 0 and n11 == 0) or (pi_01 == 0 or pi_11 == 0):
        lr_ind = 0.0
    else:
        # L_null: independent breaches with probability pi_uncond
        ll_null = (n00 + n10) * np.log(max(eps, 1.0 - pi_uncond)) + (n01 + n11) * np.log(max(eps, pi_uncond))
        # L_alt: Markov dependent breaches
        term1 = n00 * np.log(max(eps, 1.0 - pi_01)) if n00 > 0 else 0.0
        term2 = n01 * np.log(max(eps, pi_01)) if n01 > 0 else 0.0
        term3 = n10 * np.log(max(eps, 1.0 - pi_11)) if n10 > 0 else 0.0
        term4 = n11 * np.log(max(eps, pi_11)) if n11 > 0 else 0.0
        ll_alt = term1 + term2 + term3 + term4
        lr_ind = -2.0 * (ll_null - ll_alt)

    lr_ind = max(0.0, float(lr_ind))
    p_indep = float(1.0 - chi2.cdf(lr_ind, df=1))
    indep_verdict = "Pass" if p_indep >= 0.05 else "Fail"

    # Combined Conditional Coverage Test (Kupiec LR + Independence LR)
    x_total = int(np.sum(clean_series))
    p0 = 1.0 - confidence_level
    p_hat = x_total / n

    if x_total == 0:
        lr_pof = -2.0 * n * np.log(1.0 - p0)
    else:
        log_num = (n - x_total) * np.log(1.0 - p0) + x_total * np.log(p0)
        log_den = (n - x_total) * np.log(1.0 - p_hat) + x_total * np.log(p_hat)
        lr_pof = -2.0 * (log_num - log_den)

    lr_cc = max(0.0, float(lr_pof + lr_ind))
    p_cc = float(1.0 - chi2.cdf(lr_cc, df=2))
    cc_verdict = "Pass" if p_cc >= 0.05 else "Fail"

    return ChristoffersenTestResult(
        n00=n00,
        n01=n01,
        n10=n10,
        n11=n11,
        pi_01=float(pi_01),
        pi_11=float(pi_11),
        pi_uncond=float(pi_uncond),
        lr_indep_statistic=lr_ind,
        p_value_indep=p_indep,
        indep_verdict=indep_verdict,
        lr_cc_statistic=lr_cc,
        p_value_cc=p_cc,
        cc_verdict=cc_verdict,
    )


def evaluate_basel_traffic_light(
    exceptions_count: int,
    window_days: int = BASEL_WINDOW_DAYS,
    confidence_level: float = BASEL_CONFIDENCE_LEVEL,
) -> BaselTrafficLightResult:
    """
    Classify model into Basel Committee Traffic Light zones (Green, Yellow, Red).
    
    Standard: 250 trading days, 99% confidence (expected exceptions = 2.5).
    - Green: 0 to 4 exceptions (multiplier 3.00)
    - Yellow: 5 to 9 exceptions (multiplier 3.40 to 3.85)
    - Red: 10+ exceptions (multiplier 4.00, model rejected)
    """
    p = 1.0 - confidence_level
    cum_prob = float(binom.cdf(exceptions_count, window_days, p))

    yellow_multipliers = {
        5: 3.40,
        6: 3.50,
        7: 3.65,
        8: 3.75,
        9: 3.85,
    }

    if exceptions_count <= 4:
        zone = "GREEN"
        status = BASEL_ZONES["GREEN"]["status"]
        mult = BASEL_ZONES["GREEN"]["multiplier"]
        desc = BASEL_ZONES["GREEN"]["description"]
        color = BASEL_ZONES["GREEN"]["color"]
    elif 5 <= exceptions_count <= 9:
        zone = "YELLOW"
        status = BASEL_ZONES["YELLOW"]["status"]
        mult = yellow_multipliers.get(exceptions_count, 3.85)
        desc = BASEL_ZONES["YELLOW"]["description"]
        color = BASEL_ZONES["YELLOW"]["color"]
    else:
        zone = "RED"
        status = BASEL_ZONES["RED"]["status"]
        mult = BASEL_ZONES["RED"]["multiplier"]
        desc = BASEL_ZONES["RED"]["description"]
        color = BASEL_ZONES["RED"]["color"]

    return BaselTrafficLightResult(
        window_days=window_days,
        exceptions_count=exceptions_count,
        zone=zone,
        status=status,
        capital_multiplier=mult,
        description=desc,
        color=color,
        cumulative_binom_prob=cum_prob,
    )


def run_rolling_out_of_sample_backtest(
    returns_series: pd.Series,
    model_type: str = "historical",
    confidence_level: float = 0.99,
    estimation_window: int = 500,
    out_of_sample_start: Optional[str] = OUT_OF_SAMPLE_START,
) -> Tuple[ModelValidationReport, pd.DataFrame]:
    """
    Execute rigorous out-of-sample backtest without look-ahead bias.
    
    Parameters
    ----------
    returns_series : pd.Series
        Daily portfolio return series.
    model_type : str
        'historical', 'gaussian', 'student_t', or 'evt'.
    confidence_level : float
        Target confidence (e.g. 0.99).
    estimation_window : int
        Rolling lookback days for model calibration (e.g. 500 days).
    out_of_sample_start : str, optional
        Filter evaluation strictly after this date (e.g. '2024-01-01').
        
    Returns
    -------
    Tuple[ModelValidationReport, pd.DataFrame]
        Validation report dataclass and detailed daily evaluation DataFrame.
    """
    clean_ret = returns_series.dropna()
    n_total = len(clean_ret)

    if n_total <= estimation_window + 20:
        raise ValueError(f"Insufficient history ({n_total} days) for {estimation_window}-day window.")

    dates = clean_ret.index
    var_forecasts = []
    actual_returns = []
    eval_dates = []

    logger.info(f"Running rolling out-of-sample backtest for '{model_type}' at {confidence_level*100:.0f}% confidence...")

    for i in range(estimation_window, n_total):
        curr_date = dates[i]
        if out_of_sample_start and curr_date < pd.to_datetime(out_of_sample_start):
            continue

        # Strictly historical calibration window [i - estimation_window : i]
        train_window = clean_ret.iloc[i - estimation_window : i]
        actual_val = float(clean_ret.iloc[i])

        # Forecast 1-day ahead VaR
        if model_type == "historical":
            var_pred = compute_historical_var(train_window, confidence_level=confidence_level)
        elif model_type == "gaussian":
            var_pred = compute_parametric_gaussian_var(train_window, confidence_level=confidence_level)
        elif model_type == "student_t":
            var_pred, _ = compute_parametric_student_t_var(train_window, confidence_level=confidence_level)
        elif model_type == "evt":
            metrics, _ = compute_evt_risk_metrics(train_window, confidence_level=confidence_level)
            var_pred = metrics.evt_var
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        var_forecasts.append(var_pred)
        actual_returns.append(actual_val)
        eval_dates.append(curr_date)

    eval_df = pd.DataFrame({
        "Date": eval_dates,
        "Actual_Return": actual_returns,
        "VaR_Forecast": var_forecasts,
    }).set_index("Date")

    # Breach condition: actual return < - VaR
    eval_df["Breach"] = (eval_df["Actual_Return"] < -eval_df["VaR_Forecast"]).astype(int)

    # Statistical Tests
    total_obs = len(eval_df)
    n_exceptions = int(eval_df["Breach"].sum())
    exception_dates = list(eval_df[eval_df["Breach"] == 1].index)

    kupiec_res = run_kupiec_pof_test(
        exceptions_count=n_exceptions,
        total_observations=total_obs,
        confidence_level=confidence_level,
    )

    christoffersen_res = run_christoffersen_independence_test(
        breach_indicator_series=eval_df["Breach"],
        confidence_level=confidence_level,
    )

    # Basel standard: if evaluation window is > 250 days, evaluate on the most recent 250 days
    # or scale to 250 days
    if total_obs >= 250:
        recent_250_df = eval_df.iloc[-250:]
        recent_exceptions = int(recent_250_df["Breach"].sum())
        basel_res = evaluate_basel_traffic_light(recent_exceptions, window_days=250, confidence_level=confidence_level)
    else:
        basel_res = evaluate_basel_traffic_light(n_exceptions, window_days=total_obs, confidence_level=confidence_level)

    model_title = f"{model_type.capitalize()} VaR ({int(confidence_level * 100)}%)"

    report = ModelValidationReport(
        model_name=model_title,
        confidence_level=confidence_level,
        kupiec=kupiec_res,
        christoffersen=christoffersen_res,
        basel=basel_res,
        exceptions_dates=exception_dates,
    )

    return report, eval_df


def run_all_models_backtest_suite(
    returns_series: pd.Series,
    confidence_level: float = 0.99,
    estimation_window: int = 500,
    out_of_sample_start: Optional[str] = OUT_OF_SAMPLE_START,
) -> Tuple[Dict[str, ModelValidationReport], pd.DataFrame]:
    """
    Compare all 4 risk models side-by-side in out-of-sample backtesting:
    1. Historical Simulation
    2. Parametric Gaussian
    3. Parametric Student-t
    4. Extreme Value Theory (EVT-POT)
    """
    models = ["historical", "gaussian", "student_t", "evt"]
    reports = {}
    summary_records = []

    for m in models:
        rep, _ = run_rolling_out_of_sample_backtest(
            returns_series=returns_series,
            model_type=m,
            confidence_level=confidence_level,
            estimation_window=estimation_window,
            out_of_sample_start=out_of_sample_start,
        )
        reports[m] = rep
        summary_records.append(rep.to_dict())

    return reports, pd.DataFrame(summary_records)


if __name__ == "__main__":
    from src.data_loader import load_market_dataset
    from src.portfolio_opt import compute_portfolio_return_series
    from src.config import SERENDIB_CONSERVATIVE_WEIGHTS

    print("Testing backtest module...")
    md = load_market_dataset()
    port_ret = compute_portfolio_return_series(md.simple_returns, SERENDIB_CONSERVATIVE_WEIGHTS)

    reports, summary_table = run_all_models_backtest_suite(
        returns_series=port_ret,
        confidence_level=0.99,
        estimation_window=500,
        out_of_sample_start="2024-01-01",
    )

    print("\n--- Out-of-Sample Backtest & Basel Compliance Matrix (2024 - 2026) ---")
    print(summary_table.to_string())
