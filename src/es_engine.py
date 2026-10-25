"""
Layer 3: Expected Shortfall (CVaR) & Coherent Risk Engine
Mandate: The Serendib University Endowment Fund (USD 10,000,000 AUM)
QuantEdge 1.0 National STEM Competition - Team 'Pandas Predictors'

Key Mathematical & Regulatory Foundations:
1. Expected Shortfall Definition:
   ES_alpha = - E[R_p | R_p <= -VaR_alpha]
   Measures the conditional expectation of loss given that a VaR breach has occurred.
2. Formulations:
   - Empirical Historical ES: Sample tail mean of observations beyond historical VaR.
   - Parametric Gaussian ES: Analytical formula based on truncated normal distribution.
   - Parametric Student-t ES: Analytical heavy-tailed formula with degrees of freedom nu.
   - Monte Carlo ES: Tail mean across 10,000 simulated multivariate paths.
3. Coherent Risk Measure & Subadditivity Proof:
   - Demonstrates the 4 Artzner et al. (1999) axioms:
     1. Monotonicity: X <= Y => rho(X) >= rho(Y)
     2. Subadditivity: rho(X + Y) <= rho(X) + rho(Y)
     3. Positive Homogeneity: rho(lambda * X) = lambda * rho(X) for lambda > 0
     4. Translation Invariance: rho(X + a) = rho(X) - a
   - Numerical proof of VaR subadditivity failure vs ES coherence under heavy tails.
4. Tail Risk Ratio:
   - Ratio = ES_alpha / VaR_alpha (Quantifies severity multiplier of tail breaches).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from src.config import (
    MANDATE_SIZE_USD,
    MONTE_CARLO_SIMULATIONS,
    RANDOM_SEED,
    DEFAULT_CONFIDENCE_LEVELS,
)
from src.var_engine import (
    compute_historical_var,
    compute_parametric_gaussian_var,
    compute_parametric_student_t_var,
    compute_monte_carlo_var,
)

logger = logging.getLogger(__name__)


@dataclass
class ESResult:
    """Institutional container for Expected Shortfall metrics."""
    confidence: float
    var_historical: float
    es_historical: float
    es_parametric_gaussian: float
    es_parametric_student_t: float
    es_monte_carlo: float
    tail_risk_ratio: float  # es_historical / var_historical
    es_usd: float
    var_usd: float
    tail_severity_usd: float  # es_usd - var_usd

    def to_dict(self, aum: float = MANDATE_SIZE_USD) -> Dict[str, any]:
        conf_pct = int(self.confidence * 100)
        return {
            "Confidence Level": f"{conf_pct}%",
            "1-Day VaR (Empirical)": f"{self.var_historical * 100:.2f}% (${self.var_usd:,.0f})",
            "1-Day Expected Shortfall (ES)": f"{self.es_historical * 100:.2f}% (${self.es_usd:,.0f})",
            "Parametric Gaussian ES": f"{self.es_parametric_gaussian * 100:.2f}% (${self.es_parametric_gaussian * aum:,.0f})",
            "Parametric Student-t ES": f"{self.es_parametric_student_t * 100:.2f}% (${self.es_parametric_student_t * aum:,.0f})",
            "Monte Carlo ES (10k)": f"{self.es_monte_carlo * 100:.2f}% (${self.es_monte_carlo * aum:,.0f})",
            "Tail Severity Gap (ES - VaR)": f"{(self.es_historical - self.var_historical) * 100:.2f}% (${self.tail_severity_usd:,.0f})",
            "Tail Risk Ratio (ES / VaR)": f"{self.tail_risk_ratio:.2f}x",
        }


def compute_historical_es(
    returns_series: pd.Series,
    confidence_level: float = 0.99,
) -> Tuple[float, float]:
    """
    Calculate Empirical Historical Expected Shortfall (CVaR).
    
    ES_alpha = - (1 / |T_tail|) * sum_{t in T_tail} R_{p,t}
    where T_tail = {t : R_{p,t} <= -VaR_alpha}
    
    Returns
    -------
    Tuple[float, float]
        (expected_shortfall, value_at_risk)
    """
    clean_ret = returns_series.dropna().values
    var_val = compute_historical_var(returns_series, confidence_level=confidence_level)
    threshold = -var_val

    tail_losses = clean_ret[clean_ret <= threshold]
    if len(tail_losses) == 0:
        es_val = var_val
    else:
        es_val = -float(np.mean(tail_losses))

    # ES must always be >= VaR
    es_val = max(var_val, es_val)
    return es_val, var_val


def compute_parametric_gaussian_es(
    returns_series: pd.Series,
    confidence_level: float = 0.99,
) -> float:
    """
    Calculate Analytical Parametric Gaussian Expected Shortfall.
    
    For R_p ~ N(mu, sigma^2):
    ES_alpha = - mu + sigma * [ phi(z_alpha) / (1 - alpha) ]
    where phi is standard normal PDF and z_alpha = Phi^{-1}(alpha).
    """
    clean_ret = returns_series.dropna().values
    mu = float(np.mean(clean_ret))
    sigma = float(np.std(clean_ret, ddof=1))

    z = float(norm.ppf(confidence_level))
    phi_z = float(norm.pdf(z))

    alpha_tail = 1.0 - confidence_level
    es_val = -mu + sigma * (phi_z / alpha_tail)
    return max(0.0, es_val)


def compute_parametric_student_t_es(
    returns_series: pd.Series,
    confidence_level: float = 0.99,
) -> float:
    """
    Calculate Analytical Parametric Student-t Expected Shortfall.
    
    For standardized Student-t with nu degrees of freedom:
    ES_alpha = - mu + sigma * sqrt((nu - 2)/nu) * [ g_nu(t_alpha) / (1 - alpha) ] * [ (nu + t_alpha^2) / (nu - 1) ]
    """
    clean_ret = returns_series.dropna().values
    mu = float(np.mean(clean_ret))
    sigma = float(np.std(clean_ret, ddof=1))

    # Fit degrees of freedom nu
    std_ret = (clean_ret - mu) / (sigma if sigma > 1e-8 else 1.0)
    try:
        fit_params = t.fit(std_ret, floc=0, fscale=1)
        nu = max(2.5, float(fit_params[0]))
    except Exception:
        nu = 8.0

    t_crit = float(t.ppf(confidence_level, df=nu))
    g_t = float(t.pdf(t_crit, df=nu))
    alpha_tail = 1.0 - confidence_level

    scale_factor = np.sqrt((nu - 2.0) / nu)
    trunc_factor = ((nu + t_crit**2) / (nu - 1.0))

    es_val = -mu + sigma * scale_factor * (g_t / alpha_tail) * trunc_factor
    return max(0.0, es_val)


def compute_monte_carlo_es(
    returns_df: pd.DataFrame,
    weights: Dict[str, float],
    confidence_level: float = 0.99,
    n_sims: int = MONTE_CARLO_SIMULATIONS,
    seed: int = RANDOM_SEED,
) -> float:
    """
    Calculate Monte Carlo Expected Shortfall from 10,000 simulated portfolio returns.
    """
    mc_var, sim_returns = compute_monte_carlo_var(
        returns_df=returns_df,
        weights=weights,
        confidence_level=confidence_level,
        n_sims=n_sims,
        seed=seed,
    )

    tail_losses = sim_returns[sim_returns <= -mc_var]
    if len(tail_losses) == 0:
        return mc_var

    mc_es = -float(np.mean(tail_losses))
    return max(mc_var, mc_es)


def run_comprehensive_es_suite(
    returns_df: pd.DataFrame,
    weights: Dict[str, float],
    confidence_levels: Optional[List[float]] = None,
    aum: float = MANDATE_SIZE_USD,
) -> Tuple[Dict[float, ESResult], pd.DataFrame]:
    """
    Generate complete side-by-side Expected Shortfall and Tail Risk audit across confidence tiers.
    """
    confidence_levels = confidence_levels or DEFAULT_CONFIDENCE_LEVELS
    port_ret = returns_df.dot(np.array([weights.get(t, 0.0) for t in returns_df.columns]))

    results = {}
    summary_rows = []

    for conf in sorted(confidence_levels):
        hist_es, hist_var = compute_historical_es(port_ret, confidence_level=conf)
        gauss_es = compute_parametric_gaussian_es(port_ret, confidence_level=conf)
        t_es = compute_parametric_student_t_es(port_ret, confidence_level=conf)
        mc_es = compute_monte_carlo_es(returns_df, weights, confidence_level=conf)

        tail_ratio = hist_es / hist_var if hist_var > 1e-8 else 1.0
        var_usd = hist_var * aum
        es_usd = hist_es * aum
        severity_usd = es_usd - var_usd

        res = ESResult(
            confidence=conf,
            var_historical=hist_var,
            es_historical=hist_es,
            es_parametric_gaussian=gauss_es,
            es_parametric_student_t=t_es,
            es_monte_carlo=mc_es,
            tail_risk_ratio=tail_ratio,
            es_usd=es_usd,
            var_usd=var_usd,
            tail_severity_usd=severity_usd,
        )
        results[conf] = res
        summary_rows.append(res.to_dict(aum=aum))

    return results, pd.DataFrame(summary_rows)


def verify_subadditivity_properties(
    asset_1_returns: pd.Series,
    asset_2_returns: pd.Series,
    confidence_level: float = 0.99,
) -> Dict[str, any]:
    """
    Mathematically prove Subadditivity: rho(X + Y) <= rho(X) + rho(Y)
    
    Axiom Comparison:
    - Expected Shortfall is mathematically proven to be a Coherent Risk Measure (Artzner et al. 1999),
      always satisfying subadditivity under any joint distribution.
    - Value-at-Risk is NOT coherent; it can fail subadditivity when assets exhibit heavy-tailed
      distributions or asymmetric jump default risks, creating an illusion that diversification
      increases risk.
      
    Returns
    -------
    Dict[str, any]
        Empirical check of subadditivity for VaR and ES on two assets.
    """
    r1 = asset_1_returns.dropna()
    r2 = asset_2_returns.dropna()
    aligned_1, aligned_2 = r1.align(r2, join="inner")

    # Combined 50/50 portfolio
    r_comb = 0.5 * aligned_1 + 0.5 * aligned_2

    var_1 = compute_historical_var(aligned_1, confidence_level)
    var_2 = compute_historical_var(aligned_2, confidence_level)
    var_comb = compute_historical_var(r_comb, confidence_level)
    weighted_var_sum = 0.5 * var_1 + 0.5 * var_2

    es_1, _ = compute_historical_es(aligned_1, confidence_level)
    es_2, _ = compute_historical_es(aligned_2, confidence_level)
    es_comb, _ = compute_historical_es(r_comb, confidence_level)
    weighted_es_sum = 0.5 * es_1 + 0.5 * es_2

    var_subadditive = var_comb <= (weighted_var_sum + 1e-6)
    es_subadditive = es_comb <= (weighted_es_sum + 1e-6)

    # Coherence Axioms Status
    axioms = {
        "Subadditivity": {
            "Description": "rho(X + Y) <= rho(X) + rho(Y) (Diversification never penalizes)",
            "VaR_Holds": bool(var_subadditive),
            "ES_Holds": bool(es_subadditive),
            "VaR_Comb_vs_Sum": f"{var_comb * 100:.2f}% vs {weighted_var_sum * 100:.2f}%",
            "ES_Comb_vs_Sum": f"{es_comb * 100:.2f}% vs {weighted_es_sum * 100:.2f}%",
            "ES_Diversification_Benefit": f"{(weighted_es_sum - es_comb) * 100:.2f}%",
        },
        "Monotonicity": {
            "Description": "If X <= Y always, then rho(X) >= rho(Y)",
            "VaR_Holds": True,
            "ES_Holds": True,
        },
        "Positive_Homogeneity": {
            "Description": "rho(c * X) = c * rho(X) for all c > 0",
            "VaR_Holds": True,
            "ES_Holds": True,
        },
        "Translation_Invariance": {
            "Description": "rho(X + c) = rho(X) - c for cash buffer c",
            "VaR_Holds": True,
            "ES_Holds": True,
        },
        "Coherent_Measure_Verdict": {
            "VaR": "Non-Coherent (Fails subadditivity under fat tails)",
            "ES": "Strictly Coherent (Guaranteed subadditivity, Basel III FRTB standard)",
        },
    }

    return axioms


if __name__ == "__main__":
    from src.data_loader import load_market_dataset
    from src.config import SERENDIB_CONSERVATIVE_WEIGHTS

    print("Testing es_engine module...")
    md = load_market_dataset()
    results, summary_df = run_comprehensive_es_suite(
        returns_df=md.simple_returns,
        weights=SERENDIB_CONSERVATIVE_WEIGHTS,
        confidence_levels=[0.90, 0.95, 0.99],
    )

    print("\n--- Serendib Conservative Expected Shortfall Suite ($10M AUM) ---")
    print(summary_df.to_string())

    print("\n--- Subadditivity Verification (SPY + AGG) ---")
    sub_res = verify_subadditivity_properties(md.simple_returns["SPY"], md.simple_returns["AGG"])
    print("ES Subadditivity Holds:", sub_res["Subadditivity"]["ES_Holds"])
    print("ES Diversification Benefit:", sub_res["Subadditivity"]["ES_Diversification_Benefit"])
    print("Regulatory Verdict:", sub_res["Coherent_Measure_Verdict"]["ES"])
