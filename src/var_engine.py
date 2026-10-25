"""
Layer 2: Value-at-Risk (VaR) Engine (90%, 95%, 99%)
Mandate: The Serendib University Endowment Fund (USD 10,000,000 AUM)
QuantEdge 1.0 National STEM Competition - Team 'Pandas Predictors'

Key Methodological Framework:
1. Method A: Historical Simulation VaR (Empirical rolling and full-sample quantiles).
2. Method B: Parametric (Variance-Covariance) VaR:
   - Gaussian Analytical VaR
   - Student-t Heavy-Tailed VaR (Degrees of freedom nu fitted via MLE)
   - Cornish-Fisher Semi-Parametric Expansion VaR (Adjusted for skewness & excess kurtosis)
3. Method C: Monte Carlo VaR (Cholesky factorization of covariance matrix, 10,000 multivariate paths).
4. Risk Gap Analysis:
   - Risk Gap = VaR_Empirical - VaR_Gaussian
   - Quantifies the dollar underestimation on $10M AUM caused by Gaussian thin-tail assumptions.
5. Rolling VaR estimation across historical market regimes.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, t, skew, kurtosis

from src.config import (
    MANDATE_SIZE_USD,
    MONTE_CARLO_SIMULATIONS,
    RANDOM_SEED,
    DEFAULT_CONFIDENCE_LEVELS,
)

logger = logging.getLogger(__name__)


@dataclass
class VaRResult:
    """Institutional container for VaR estimates across confidence levels."""
    confidence: float
    historical_var: float
    parametric_gaussian_var: float
    parametric_student_t_var: float
    cornish_fisher_var: float
    monte_carlo_var: float
    student_t_dof: float
    risk_gap_gaussian: float
    risk_gap_usd: float

    def to_dict(self, aum: float = MANDATE_SIZE_USD) -> Dict[str, any]:
        conf_pct = int(self.confidence * 100)
        return {
            "Confidence Level": f"{conf_pct}%",
            "Historical Sim VaR": f"{self.historical_var * 100:.2f}% (${self.historical_var * aum:,.0f})",
            "Parametric Gaussian VaR": f"{self.parametric_gaussian_var * 100:.2f}% (${self.parametric_gaussian_var * aum:,.0f})",
            "Parametric Student-t VaR": f"{self.parametric_student_t_var * 100:.2f}% (${self.parametric_student_t_var * aum:,.0f})",
            "Cornish-Fisher VaR": f"{self.cornish_fisher_var * 100:.2f}% (${self.cornish_fisher_var * aum:,.0f})",
            "Monte Carlo VaR (10k)": f"{self.monte_carlo_var * 100:.2f}% (${self.monte_carlo_var * aum:,.0f})",
            "Risk Gap (Hist - Gauss)": f"{self.risk_gap_gaussian * 100:.2f}% (${self.risk_gap_usd:,.0f})",
            "Fitted Student-t DoF": f"{self.student_t_dof:.2f}",
        }


def compute_historical_var(
    returns_series: pd.Series,
    confidence_level: float = 0.99,
) -> float:
    """
    Calculate Historical Simulation Value-at-Risk (Non-parametric empirical quantile).
    
    VaR_alpha = - Quantile(R_p, 1 - alpha)
    
    Returns
    -------
    float
        Positive loss fraction (e.g. 0.0215 for 2.15% loss).
    """
    clean_ret = returns_series.dropna()
    tail_quantile = 1.0 - confidence_level
    var_val = -float(np.percentile(clean_ret, tail_quantile * 100.0))
    return max(0.0, var_val)


def compute_parametric_gaussian_var(
    returns_series: pd.Series,
    confidence_level: float = 0.99,
) -> float:
    """
    Calculate Parametric (Variance-Covariance) Gaussian VaR.
    
    VaR_alpha = - (mu_p + z_{1-alpha} * sigma_p) = - mu_p + z_alpha * sigma_p
    """
    clean_ret = returns_series.dropna()
    mu = float(clean_ret.mean())
    sigma = float(clean_ret.std(ddof=1))
    z = float(norm.ppf(confidence_level))
    var_val = -(mu - z * sigma)
    return max(0.0, var_val)


def compute_parametric_student_t_var(
    returns_series: pd.Series,
    confidence_level: float = 0.99,
) -> Tuple[float, float]:
    """
    Calculate Parametric Student-t VaR by fitting degrees of freedom (nu).
    
    Formula:
    VaR_alpha = - mu_p + t_inv(alpha, nu) * sqrt((nu - 2)/nu) * sigma_p
    
    Returns
    -------
    Tuple[float, float]
        (student_t_var, fitted_degrees_of_freedom)
    """
    clean_ret = returns_series.dropna().values
    mu = float(np.mean(clean_ret))
    sigma = float(np.std(clean_ret, ddof=1))

    # Fit standardized Student-t distribution via MLE
    try:
        # Standardize returns
        std_ret = (clean_ret - mu) / (sigma if sigma > 1e-8 else 1.0)
        fit_params = t.fit(std_ret, floc=0, fscale=1)
        nu = max(2.1, float(fit_params[0]))  # Ensure nu > 2 for finite variance
    except Exception as e:
        logger.debug(f"Student-t MLE fit failed ({e}). Falling back to kurtosis method of moments.")
        # Method of moments: excess kurtosis K = 6 / (nu - 4) => nu = 6/K + 4
        ex_kurt = float(kurtosis(clean_ret))
        nu = max(2.5, min(30.0, 6.0 / ex_kurt + 4.0)) if ex_kurt > 0.2 else 30.0

    t_crit = float(t.ppf(confidence_level, df=nu))
    # Scaling factor so variance matches sample variance: sqrt((nu - 2)/nu)
    scale_factor = np.sqrt((nu - 2.0) / nu)
    var_val = -(mu - t_crit * scale_factor * sigma)
    return max(0.0, var_val), nu


def compute_cornish_fisher_var(
    returns_series: pd.Series,
    confidence_level: float = 0.99,
) -> float:
    """
    Calculate Cornish-Fisher Semi-Parametric Expansion VaR.
    Adjusts the Gaussian quantile for sample skewness (S) and excess kurtosis (K).
    
    z_CF = z + (1/6)(z^2 - 1)S + (1/24)(z^3 - 3z)K - (1/36)(2z^3 - 5z)S^2
    VaR_alpha = - mu + z_CF * sigma
    """
    clean_ret = returns_series.dropna()
    mu = float(clean_ret.mean())
    sigma = float(clean_ret.std(ddof=1))

    s = float(skew(clean_ret))
    k = float(kurtosis(clean_ret))  # Fisher excess kurtosis

    z = float(norm.ppf(confidence_level))

    z_cf = (
        z
        + (1.0 / 6.0) * (z**2 - 1.0) * s
        + (1.0 / 24.0) * (z**3 - 3.0 * z) * k
        - (1.0 / 36.0) * (2.0 * z**3 - 5.0 * z) * (s**2)
    )

    # Ensure monotonic valid quantile (safeguard against polynomial inversion when z < sqrt(3))
    z_cf = max(z * 0.75, z_cf)

    var_val = -(mu - z_cf * sigma)
    return max(0.0, var_val)



def compute_monte_carlo_var(
    returns_df: pd.DataFrame,
    weights: Dict[str, float],
    confidence_level: float = 0.99,
    n_sims: int = MONTE_CARLO_SIMULATIONS,
    seed: int = RANDOM_SEED,
    distribution: str = "normal",
) -> Tuple[float, np.ndarray]:
    """
    Calculate Monte Carlo VaR via Cholesky Factorization of asset covariance matrix.
    
    Multivariate Simulation Steps:
    1. Estimate mean vector mu and covariance matrix Sigma.
    2. Factorize Sigma = L * L^T via Cholesky decomposition.
    3. Generate n_sims independent standard shocks Z ~ N(0, I_N).
    4. Correlated returns: X = mu + L * Z.
    5. Portfolio simulated returns: R_p,sim = w^T * X.
    6. Derive empirical percentile from simulated distribution.
    
    Returns
    -------
    Tuple[float, np.ndarray]
        (mc_var, simulated_portfolio_returns)
    """
    np.random.seed(seed)
    tickers = list(returns_df.columns)
    w_vec = np.array([weights.get(t, 0.0) for t in tickers])
    if not np.isclose(np.sum(w_vec), 1.0, atol=1e-3):
        w_vec = w_vec / np.sum(w_vec)

    mu = returns_df.mean().values
    cov = returns_df.cov().values

    # Cholesky factorization with eigenvalue jitter if needed
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        logger.warning("Covariance matrix not strictly positive definite. Applying Higham eigenvalue jitter.")
        min_eig = np.min(np.real(np.linalg.eigvals(cov)))
        cov_jittered = cov + 1e-6 * np.eye(len(tickers)) if min_eig < 1e-8 else cov
        L = np.linalg.cholesky(cov_jittered)

    n_assets = len(tickers)

    if distribution == "student_t":
        # Multivariate Student-t simulation using normal-inverse-gamma mixture
        nu = 5.0
        z_raw = np.random.normal(0, 1, size=(n_sims, n_assets))
        w_inv_gamma = np.random.chisquare(df=nu, size=n_sims) / nu
        z_shocks = z_raw / np.sqrt(w_inv_gamma)[:, np.newaxis]
    else:
        # Standard Multivariate Normal
        z_shocks = np.random.normal(0, 1, size=(n_sims, n_assets))

    # X = mu + Z * L^T
    simulated_asset_returns = mu + np.dot(z_shocks, L.T)
    simulated_port_returns = np.dot(simulated_asset_returns, w_vec)

    tail_pct = (1.0 - confidence_level) * 100.0
    mc_var = -float(np.percentile(simulated_port_returns, tail_pct))

    return max(0.0, mc_var), simulated_port_returns


def run_comprehensive_var_suite(
    returns_df: pd.DataFrame,
    weights: Dict[str, float],
    confidence_levels: Optional[List[float]] = None,
    aum: float = MANDATE_SIZE_USD,
) -> Tuple[Dict[float, VaRResult], pd.DataFrame]:
    """
    Run all three VaR methodologies across multiple confidence tiers (90%, 95%, 99%).
    Produces a side-by-side comparison matrix and Risk Gap analysis.
    
    Returns
    -------
    Tuple[Dict[float, VaRResult], pd.DataFrame]
        Dictionary of VaRResult objects and clean comparison summary DataFrame.
    """
    confidence_levels = confidence_levels or DEFAULT_CONFIDENCE_LEVELS
    port_ret = returns_df.dot(np.array([weights.get(t, 0.0) for t in returns_df.columns]))

    results = {}
    summary_rows = []

    for conf in sorted(confidence_levels):
        hist_var = compute_historical_var(port_ret, confidence_level=conf)
        gauss_var = compute_parametric_gaussian_var(port_ret, confidence_level=conf)
        t_var, dof = compute_parametric_student_t_var(port_ret, confidence_level=conf)
        cf_var = compute_cornish_fisher_var(port_ret, confidence_level=conf)
        mc_var, _ = compute_monte_carlo_var(returns_df, weights, confidence_level=conf)

        # Risk Gap: Difference between empirical loss and normal parametric loss
        risk_gap = hist_var - gauss_var
        risk_gap_usd = risk_gap * aum

        res = VaRResult(
            confidence=conf,
            historical_var=hist_var,
            parametric_gaussian_var=gauss_var,
            parametric_student_t_var=t_var,
            cornish_fisher_var=cf_var,
            monte_carlo_var=mc_var,
            student_t_dof=dof,
            risk_gap_gaussian=risk_gap,
            risk_gap_usd=risk_gap_usd,
        )
        results[conf] = res
        summary_rows.append(res.to_dict(aum=aum))

    summary_df = pd.DataFrame(summary_rows)
    return results, summary_df


def compute_rolling_var(
    returns_series: pd.Series,
    window: int = 252,
    confidence_level: float = 0.99,
) -> pd.DataFrame:
    """
    Compute rolling Historical and Gaussian VaR series to track dynamic tail risk over time.
    """
    clean_ret = returns_series.dropna()
    z = norm.ppf(confidence_level)

    # Rolling Historical VaR
    rolling_hist_var = clean_ret.rolling(window=window).apply(
        lambda s: -np.percentile(s, (1.0 - confidence_level) * 100.0),
        raw=True,
    )

    # Rolling Parametric Gaussian VaR
    rolling_mean = clean_ret.rolling(window=window).mean()
    rolling_std = clean_ret.rolling(window=window).std()
    rolling_gauss_var = -(rolling_mean - z * rolling_std)

    df_rolling = pd.DataFrame({
        "Portfolio_Return": clean_ret,
        "Rolling_Hist_VaR": rolling_hist_var,
        "Rolling_Gauss_VaR": rolling_gauss_var,
    }).dropna()

    return df_rolling


if __name__ == "__main__":
    from src.data_loader import load_market_dataset
    from src.config import SERENDIB_CONSERVATIVE_WEIGHTS

    print("Testing var_engine module...")
    md = load_market_dataset()
    results, summary_df = run_comprehensive_var_suite(
        returns_df=md.simple_returns,
        weights=SERENDIB_CONSERVATIVE_WEIGHTS,
        confidence_levels=[0.90, 0.95, 0.99],
    )

    print("\n--- Serendib Conservative Portfolio VaR Suite ($10M AUM) ---")
    print(summary_df.to_string())
