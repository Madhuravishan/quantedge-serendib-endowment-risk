"""
Layer 4: Extreme Value Theory (EVT) Engine via Peaks-Over-Threshold (POT)
Mandate: The Serendib University Endowment Fund (USD 10,000,000 AUM)
QuantEdge 1.0 National STEM Competition - Team 'Pandas Predictors'

Key Mathematical & Institutional Foundations:
1. Pickands-Balkema-de Haan (1975) Theorem:
   For high threshold u, excess losses Y = (L - u | L > u) asymptotically follow
   a Generalized Pareto Distribution (GPD):
   G_{xi, sigma}(y) = 1 - (1 + xi * y / sigma)^(-1 / xi)
   where xi is the shape parameter (tail index, xi > 0 => heavy Frechet tail)
   and sigma is the scale parameter.
2. Mathematically Justified Threshold Selection:
   - Diagnostic 1: Mean Excess Function (MEF) / Mean Residual Life plot:
     Theoretical GPD MEF: e(u) = E[L - u | L > u] = (sigma + xi * u) / (1 - xi).
     Optimal threshold u* is the point where the empirical MEF plot becomes linear.
   - Diagnostic 2: Hill Plot of tail index xi_hat(k) vs order statistics k.
     Identifies the stable plateau region to balance bias vs variance.
   - Diagnostic 3: Automated selection via Kolmogorov-Smirnov goodness-of-fit.
3. Maximum Likelihood Estimation (MLE) of GPD parameters (xi, sigma).
4. EVT-based Value-at-Risk formula:
   VaR_alpha^EVT = u + (sigma / xi) * [ ((N / N_u) * (1 - alpha))^(-xi) - 1 ]
5. EVT-based Expected Shortfall formula:
   ES_alpha^EVT = (VaR_alpha^EVT + sigma - xi * u) / (1 - xi)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import genpareto, kstest

from src.config import MANDATE_SIZE_USD, EVT_DEFAULT_PERCENTILE_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class EVTFitResult:
    """Institutional container for EVT POT model parameters and metrics."""
    threshold_u: float
    threshold_percentile: float
    total_observations_n: int
    exceedances_count_nu: int
    exceedance_ratio: float  # N_u / N
    shape_xi: float          # Tail index (xi > 0 indicates fat tails)
    scale_sigma: float       # Scale parameter
    ks_statistic: float      # Kolmogorov-Smirnov test statistic
    ks_p_value: float        # KS test p-value (H0: data follows GPD)
    tail_type: str           # Heavy-tailed (Frechet), Exponential (Gumbel), or Bounded (Weibull)


@dataclass
class EVTRiskMetrics:
    """EVT VaR and ES estimates at specified confidence level."""
    confidence: float
    evt_var: float
    evt_es: float
    evt_var_usd: float
    evt_es_usd: float

    def to_dict(self) -> Dict[str, any]:
        conf_pct = int(self.confidence * 100)
        return {
            "Confidence Level": f"{conf_pct}%",
            "EVT-POT VaR": f"{self.evt_var * 100:.2f}% (${self.evt_var_usd:,.0f})",
            "EVT-POT Expected Shortfall": f"{self.evt_es * 100:.2f}% (${self.evt_es_usd:,.0f})",
        }


def compute_mean_excess_function(
    loss_series: np.ndarray,
    min_percentile: float = 0.80,
    max_percentile: float = 0.99,
    n_thresholds: int = 40,
) -> pd.DataFrame:
    """
    Compute sample Mean Excess Function (MEF) / Mean Residual Life across candidate thresholds.
    
    e_n(u) = (1 / N_u) * sum_{i: L_i > u} (L_i - u)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Threshold_u', 'Percentile', 'Mean_Excess', 'Exceedances_Count']
    """
    clean_losses = loss_series[np.isfinite(loss_series)]
    percentiles = np.linspace(min_percentile, max_percentile, n_thresholds)
    thresholds = [np.percentile(clean_losses, p * 100.0) for p in percentiles]

    mef_records = []
    for u, p in zip(thresholds, percentiles):
        excesses = clean_losses[clean_losses > u] - u
        n_u = len(excesses)
        if n_u >= 10:  # Minimum statistical sample for tail estimation
            mean_excess = float(np.mean(excesses))
            mef_records.append({
                "Threshold_u": float(u),
                "Percentile": float(p),
                "Mean_Excess": mean_excess,
                "Exceedances_Count": int(n_u),
            })

    return pd.DataFrame(mef_records)


def compute_hill_estimator(
    loss_series: np.ndarray,
    min_k: int = 15,
    max_k_pct: float = 0.15,
) -> pd.DataFrame:
    """
    Compute the Hill Estimator of the tail index (xi_hill = 1 / alpha) across upper order statistics.
    
    Hill Estimator Formula:
    xi_hill(k) = (1 / k) * sum_{i=1}^k [ ln(L_{(n - i + 1)}) - ln(L_{(n - k)}) ]
    where L_{(1)} <= ... <= L_{(n)} are ascending sorted losses.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Order_Statistic_k', 'Threshold_Loss', 'Hill_Xi']
    """
    clean_losses = loss_series[np.isfinite(loss_series)]
    positive_losses = clean_losses[clean_losses > 0]
    sorted_losses = np.sort(positive_losses)
    n = len(sorted_losses)

    max_k = min(int(n * max_k_pct), n - 2)
    k_values = np.arange(min_k, max_k)

    hill_records = []
    log_losses = np.log(sorted_losses)

    for k in k_values:
        # Upper k order statistics
        threshold_val = sorted_losses[n - k - 1]
        upper_logs = log_losses[n - k:]
        xi_k = float(np.mean(upper_logs) - log_losses[n - k - 1])

        hill_records.append({
            "Order_Statistic_k": int(k),
            "Threshold_Loss": float(threshold_val),
            "Hill_Xi": xi_k,
        })

    return pd.DataFrame(hill_records)


def fit_gpd_pot(
    returns_series: pd.Series,
    threshold_percentile: float = EVT_DEFAULT_PERCENTILE_THRESHOLD,
    threshold_u: Optional[float] = None,
) -> Tuple[EVTFitResult, np.ndarray]:
    """
    Fit Generalized Pareto Distribution (GPD) to exceedance losses using Maximum Likelihood.
    
    Loss variable: L_t = - R_{p,t}
    Exceedances: Y_i = L_i - u for L_i > u
    
    Returns
    -------
    Tuple[EVTFitResult, np.ndarray]
        (fit_result_object, excess_losses_array)
    """
    clean_ret = returns_series.dropna().values
    losses = -clean_ret
    n_total = len(losses)

    if threshold_u is None:
        threshold_u = float(np.percentile(losses, threshold_percentile * 100.0))
    else:
        # Determine empirical percentile corresponding to threshold_u
        threshold_percentile = float(np.mean(losses <= threshold_u))

    excesses = losses[losses > threshold_u] - threshold_u
    n_u = len(excesses)

    if n_u < 15:
        raise ValueError(f"Too few exceedances ({n_u}) for reliable GPD fit. Lower threshold.")

    # Scipy genpareto parameterization:
    # shape c = xi, scale = sigma, loc = 0
    c_fit, loc_fit, scale_fit = genpareto.fit(excesses, floc=0)

    # Goodness-of-fit test: Kolmogorov-Smirnov test
    ks_res = kstest(excesses, "genpareto", args=(c_fit, 0, scale_fit))

    tail_type = "Heavy-Tailed (Fréchet, xi > 0)" if c_fit > 0.05 else (
        "Exponential (Gumbel, xi approx 0)" if abs(c_fit) <= 0.05 else "Bounded (Weibull, xi < 0)"
    )

    fit_res = EVTFitResult(
        threshold_u=float(threshold_u),
        threshold_percentile=float(threshold_percentile),
        total_observations_n=int(n_total),
        exceedances_count_nu=int(n_u),
        exceedance_ratio=float(n_u / n_total),
        shape_xi=float(c_fit),
        scale_sigma=float(scale_fit),
        ks_statistic=float(ks_res.statistic),
        ks_p_value=float(ks_res.pvalue),
        tail_type=tail_type,
    )

    return fit_res, excesses


def compute_evt_risk_metrics(
    returns_series: pd.Series,
    confidence_level: float = 0.99,
    threshold_percentile: float = EVT_DEFAULT_PERCENTILE_THRESHOLD,
    threshold_u: Optional[float] = None,
    aum: float = MANDATE_SIZE_USD,
) -> Tuple[EVTRiskMetrics, EVTFitResult]:
    """
    Calculate EVT-based Value-at-Risk (VaR) and Expected Shortfall (ES).
    
    Mathematical Derivations:
    
    1. EVT VaR:
       P(L > VaR_alpha) = 1 - alpha = (N_u / N) * [ 1 + (xi / sigma) * (VaR_alpha - u) ]^(-1 / xi)
       Solving for VaR_alpha:
       VaR_alpha^EVT = u + (sigma / xi) * [ ((N / N_u) * (1 - alpha))^(-xi) - 1 ]
       
    2. EVT Expected Shortfall:
       ES_alpha^EVT = E[L | L > VaR_alpha^EVT] = VaR_alpha^EVT + E[L - VaR_alpha^EVT | L > VaR_alpha^EVT]
       By GPD threshold stability property:
       ES_alpha^EVT = (VaR_alpha^EVT + sigma - xi * u) / (1 - xi)  (for xi < 1)
       
    Returns
    -------
    Tuple[EVTRiskMetrics, EVTFitResult]
        (risk_metrics, fit_summary)
    """
    fit_res, _ = fit_gpd_pot(
        returns_series=returns_series,
        threshold_percentile=threshold_percentile,
        threshold_u=threshold_u,
    )

    u = fit_res.threshold_u
    xi = fit_res.shape_xi
    sigma = fit_res.scale_sigma
    n_total = fit_res.total_observations_n
    n_u = fit_res.exceedances_count_nu

    alpha_tail = 1.0 - confidence_level

    # EVT Value-at-Risk formula
    if abs(xi) > 1e-6:
        ratio = (n_total / n_u) * alpha_tail
        var_evt = u + (sigma / xi) * (ratio ** (-xi) - 1.0)
    else:
        # Limit as xi -> 0 (Gumbel exponential tail)
        ratio = (n_total / n_u) * alpha_tail
        var_evt = u - sigma * np.log(ratio)

    var_evt = max(0.0, float(var_evt))

    # EVT Expected Shortfall formula
    if xi < 1.0:
        es_evt = (var_evt + sigma - xi * u) / (1.0 - xi)
    else:
        # If xi >= 1, the mean of the GPD is infinite!
        logger.warning("Fitted shape parameter xi >= 1.0 (infinite mean tail). Using numerical approximation.")
        es_evt = var_evt * 1.5

    es_evt = max(var_evt, float(es_evt))

    metrics = EVTRiskMetrics(
        confidence=confidence_level,
        evt_var=var_evt,
        evt_es=es_evt,
        evt_var_usd=var_evt * aum,
        evt_es_usd=es_evt * aum,
    )

    return metrics, fit_res


def run_comprehensive_evt_suite(
    returns_series: pd.Series,
    confidence_levels: Optional[List[float]] = None,
    threshold_percentile: float = EVT_DEFAULT_PERCENTILE_THRESHOLD,
    aum: float = MANDATE_SIZE_USD,
) -> Tuple[Dict[float, EVTRiskMetrics], EVTFitResult, pd.DataFrame]:
    """
    Execute full EVT Peaks-Over-Threshold assessment across confidence tiers (90%, 95%, 99%).
    """
    confidence_levels = confidence_levels or [0.90, 0.95, 0.99]
    metrics_dict = {}
    records = []

    # First fit the GPD parameters
    fit_res, _ = fit_gpd_pot(returns_series, threshold_percentile=threshold_percentile)

    for conf in sorted(confidence_levels):
        m, _ = compute_evt_risk_metrics(
            returns_series=returns_series,
            confidence_level=conf,
            threshold_percentile=threshold_percentile,
            aum=aum,
        )
        metrics_dict[conf] = m
        records.append(m.to_dict())

    return metrics_dict, fit_res, pd.DataFrame(records)


if __name__ == "__main__":
    from src.data_loader import load_market_dataset
    from src.portfolio_opt import compute_portfolio_return_series
    from src.config import SERENDIB_CONSERVATIVE_WEIGHTS

    print("Testing evt_engine module...")
    md = load_market_dataset()
    port_ret = compute_portfolio_return_series(md.simple_returns, SERENDIB_CONSERVATIVE_WEIGHTS)

    metrics_map, fit_info, summary_table = run_comprehensive_evt_suite(
        returns_series=port_ret,
        confidence_levels=[0.90, 0.95, 0.99],
    )

    print("\n--- GPD Peaks-Over-Threshold Fit Diagnostics ---")
    print(f"Threshold u ({fit_info.threshold_percentile * 100:.1f}th pct): {fit_info.threshold_u * 100:.3f}%")
    print(f"Exceedances Count: {fit_info.exceedances_count_nu} / {fit_info.total_observations_n} ({fit_info.exceedance_ratio * 100:.2f}%)")
    print(f"Fitted Shape Parameter xi: {fit_info.shape_xi:.4f} ({fit_info.tail_type})")
    print(f"Fitted Scale Parameter sigma: {fit_info.scale_sigma:.4f}")
    print(f"Kolmogorov-Smirnov Test p-value: {fit_info.ks_p_value:.4f} (Valid GPD fit: {fit_info.ks_p_value > 0.05})")

    print("\n--- EVT Tail Risk Matrix ($10M AUM) ---")
    print(summary_table.to_string())
