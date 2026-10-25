"""
Layer 1: Foundations - Portfolio Theory, Optimization & Institutional KPIs
Mandate: The Serendib University Endowment Fund (USD 10,000,000 AUM)
QuantEdge 1.0 National STEM Competition - Team 'Pandas Predictors'

Key Mathematical Capabilities:
1. Markowitz Mean-Variance Optimization via Quadratic Programming (SLSQP).
2. Efficient Frontier derivation and tangency portfolio solving.
3. Prescribed Model Portfolios:
   - Serendib Conservative Endowment Allocation (Defended human-justified mandate)
   - Max Sharpe Portfolio
   - Minimum Variance Portfolio
   - Classic 60/40 Equity/Bond Benchmark
   - Equal Weight (1/N)
4. Institutional Performance & Risk KPI Engine:
   - Annualized Return, Volatility, Sharpe Ratio
   - Downside Semi-Deviation, Sortino Ratio
   - CAPM Beta, Jensen's Alpha, Treynor Ratio
   - Tracking Error, Information Ratio (vs SPY benchmark)
   - Peak-to-Trough Drawdown series, Maximum Drawdown (MDD), Calmar Ratio
   - Gain-to-Pain Ratio, Skewness, Kurtosis
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis

from src.config import (
    TICKERS,
    MANDATE_SIZE_USD,
    TRADING_DAYS_PER_YEAR,
    SERENDIB_CONSERVATIVE_WEIGHTS,
    CLASSIC_60_40_WEIGHTS,
    EQUAL_WEIGHTS,
    BENCHMARK_EQUITY,
)

logger = logging.getLogger(__name__)


@dataclass
class PortfolioKPIs:
    """Institutional KPI summary container."""
    name: str
    weights: Dict[str, float]
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    downside_deviation: float
    sortino_ratio: float
    beta: float
    jensens_alpha: float
    treynor_ratio: float
    tracking_error: float
    information_ratio: float
    max_drawdown: float
    calmar_ratio: float
    gain_to_pain: float
    skewness: float
    kurtosis: float
    cagr: float
    final_wealth_usd: float

    def to_dict(self) -> Dict[str, any]:
        return {
            "Portfolio": self.name,
            "CAGR": f"{self.cagr * 100:.2f}%",
            "Annual Return": f"{self.annual_return * 100:.2f}%",
            "Annual Volatility": f"{self.annual_volatility * 100:.2f}%",
            "Sharpe Ratio (Rf)": f"{self.sharpe_ratio:.3f}",
            "Downside Volatility": f"{self.downside_deviation * 100:.2f}%",
            "Sortino Ratio": f"{self.sortino_ratio:.3f}",
            "CAPM Beta (vs SPY)": f"{self.beta:.3f}",
            "Jensen's Alpha (ann)": f"{self.jensens_alpha * 100:.2f}%",
            "Treynor Ratio": f"{self.treynor_ratio * 100:.2f}%",
            "Tracking Error": f"{self.tracking_error * 100:.2f}%",
            "Information Ratio": f"{self.information_ratio:.3f}",
            "Max Drawdown (MDD)": f"{self.max_drawdown * 100:.2f}%",
            "Calmar Ratio": f"{self.calmar_ratio:.3f}",
            "Gain-to-Pain Ratio": f"{self.gain_to_pain:.3f}",
            "Skewness": f"{self.skewness:.3f}",
            "Excess Kurtosis": f"{self.kurtosis:.3f}",
            "End Wealth ($10M AUM)": f"${self.final_wealth_usd:,.2f}",
        }


def compute_portfolio_return_series(
    returns_df: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    """
    Calculate the cross-sectional weighted portfolio return series.
    
    r_{p,t} = sum_{i=1}^N w_i r_{i,t}
    """
    w_vec = np.array([weights.get(col, 0.0) for col in returns_df.columns])
    if not np.isclose(np.sum(w_vec), 1.0, atol=1e-3):
        # Normalize weights if slight rounding discrepancy
        w_vec = w_vec / np.sum(w_vec)
    port_ret = returns_df.dot(w_vec)
    port_ret.name = "Portfolio_Return"
    return port_ret


def compute_drawdown_series(returns_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Derive the cumulative wealth path and peak-to-trough drawdown series.
    
    Wealth_t = Wealth_0 * prod_{s=1}^t (1 + R_s)
    Peak_t = max_{s <= t} Wealth_s
    DD_t = (Wealth_t - Peak_t) / Peak_t
    
    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (wealth_index, drawdown_series)
    """
    wealth_index = (1.0 + returns_series).cumprod()
    peak = wealth_index.cummax()
    drawdown = (wealth_index - peak) / peak
    return wealth_index, drawdown


def calculate_portfolio_kpis(
    returns_series: pd.Series,
    weights: Dict[str, float],
    name: str = "Portfolio",
    benchmark_series: Optional[pd.Series] = None,
    risk_free_series: Optional[pd.Series] = None,
    initial_aum: float = MANDATE_SIZE_USD,
) -> PortfolioKPIs:
    """
    Calculate full suite of institutional performance and risk KPIs.
    """
    clean_ret = returns_series.dropna()
    n_days = len(clean_ret)
    if n_days < 20:
        raise ValueError("Insufficient data to compute KPIs.")

    # Risk-free rate handling
    if risk_free_series is not None:
        rf_daily = risk_free_series.reindex(clean_ret.index).ffill().bfill()
        rf_mean_ann = float(rf_daily.mean() * TRADING_DAYS_PER_YEAR)
    else:
        rf_mean_ann = 0.035
        rf_daily = pd.Series(rf_mean_ann / TRADING_DAYS_PER_YEAR, index=clean_ret.index)

    # 1. Annualized Return and Volatility
    mean_daily = float(clean_ret.mean())
    std_daily = float(clean_ret.std(ddof=1))
    ann_return = mean_daily * TRADING_DAYS_PER_YEAR
    ann_volatility = std_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

    # 2. Sharpe Ratio
    excess_ret_daily = clean_ret - rf_daily
    excess_ret_ann = ann_return - rf_mean_ann
    sharpe = excess_ret_ann / ann_volatility if ann_volatility > 1e-8 else 0.0

    # 3. Downside Deviation & Sortino Ratio
    # Downside semivariance relative to daily risk-free rate
    downside_diff = np.minimum(0.0, excess_ret_daily)
    downside_std_ann = np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sortino = excess_ret_ann / downside_std_ann if downside_std_ann > 1e-8 else 0.0

    # 4. Wealth, Drawdowns & CAGR
    wealth_index, dd_series = compute_drawdown_series(clean_ret)
    max_dd = float(dd_series.min())  # Negative number (e.g. -0.15)
    total_periods_years = n_days / TRADING_DAYS_PER_YEAR
    final_mult = float(wealth_index.iloc[-1])
    cagr = (final_mult ** (1.0 / total_periods_years) - 1.0) if total_periods_years > 0 else 0.0
    calmar = abs(cagr / max_dd) if abs(max_dd) > 1e-6 else 0.0
    final_wealth = initial_aum * final_mult

    # 5. CAPM Beta, Jensen's Alpha, Treynor Ratio, Tracking Error, Information Ratio
    if benchmark_series is not None:
        bench_clean = benchmark_series.reindex(clean_ret.index).dropna()
        aligned_ret, aligned_bench = clean_ret.align(bench_clean, join="inner")
        aligned_rf = rf_daily.reindex(aligned_ret.index).ffill()

        bench_excess = aligned_bench - aligned_rf
        port_excess = aligned_ret - aligned_rf

        cov_matrix = np.cov(port_excess, bench_excess)
        bench_var = cov_matrix[1, 1]
        beta = float(cov_matrix[0, 1] / bench_var) if bench_var > 1e-8 else 1.0

        bench_ann_ret = float(aligned_bench.mean() * TRADING_DAYS_PER_YEAR)
        # Jensen's Alpha: alpha = (R_p - R_f) - beta * (R_m - R_f)
        jensens_alpha = (ann_return - rf_mean_ann) - beta * (bench_ann_ret - rf_mean_ann)

        treynor = excess_ret_ann / beta if abs(beta) > 1e-6 else 0.0

        # Tracking Error & Information Ratio
        active_ret = aligned_ret - aligned_bench
        tracking_error = float(active_ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        info_ratio = (ann_return - bench_ann_ret) / tracking_error if tracking_error > 1e-8 else 0.0
    else:
        beta = 1.0
        jensens_alpha = 0.0
        treynor = sharpe
        tracking_error = 0.0
        info_ratio = 0.0

    # 6. Higher Moments & Ratios
    sk = float(skew(clean_ret))
    kt = float(kurtosis(clean_ret))  # Fisher excess kurtosis (0 for normal)
    pos_returns = clean_ret[clean_ret > 0].sum()
    neg_returns = abs(clean_ret[clean_ret < 0].sum())
    gain_to_pain = float(pos_returns / neg_returns) if neg_returns > 1e-8 else 0.0

    return PortfolioKPIs(
        name=name,
        weights=weights,
        annual_return=ann_return,
        annual_volatility=ann_volatility,
        sharpe_ratio=sharpe,
        downside_deviation=float(downside_std_ann),
        sortino_ratio=sortino,
        beta=beta,
        jensens_alpha=jensens_alpha,
        treynor_ratio=treynor,
        tracking_error=tracking_error,
        information_ratio=info_ratio,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        gain_to_pain=gain_to_pain,
        skewness=sk,
        kurtosis=kt,
        cagr=cagr,
        final_wealth_usd=final_wealth,
    )


def optimize_markowitz_portfolio(
    returns_df: pd.DataFrame,
    target: str = "max_sharpe",
    risk_free_rate: float = 0.035,
    custom_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    target_return: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Solve Markowitz Mean-Variance Portfolio Optimization problem via Quadratic Programming.
    
    Formulation:
    - Min Volatility:
        min_w  w^T Sigma w
        s.t.   sum w_i = 1,  0 <= w_i <= 1
    - Max Sharpe:
        max_w  (w^T mu - r_f) / sqrt(w^T Sigma w)
        s.t.   sum w_i = 1,  0 <= w_i <= 1
    - Target Return:
        min_w  w^T Sigma w
        s.t.   w^T mu = r_target,  sum w_i = 1,  0 <= w_i <= 1
        
    Parameters
    ----------
    returns_df : pd.DataFrame
        Asset returns DataFrame.
    target : str
        'max_sharpe', 'min_volatility', or 'target_return'.
    risk_free_rate : float
        Annualized risk-free rate.
    custom_bounds : dict, optional
        Asset-specific weight bounds {ticker: (min_w, max_w)}.
    target_return : float, optional
        Annualized required return if target == 'target_return'.
        
    Returns
    -------
    Tuple[np.ndarray, Dict[str, float]]
        (optimal_weights_array, optimal_weights_dict)
    """
    tickers = list(returns_df.columns)
    n_assets = len(tickers)

    # Annualized mean vector and covariance matrix
    mu_daily = returns_df.mean().values
    mu_ann = mu_daily * TRADING_DAYS_PER_YEAR
    cov_daily = returns_df.cov().values
    cov_ann = cov_daily * TRADING_DAYS_PER_YEAR

    # Set asset bounds (default 0 to 1, no shorting)
    if custom_bounds:
        bounds = [custom_bounds.get(t, (0.0, 1.0)) for t in tickers]
    else:
        bounds = [(0.0, 1.0) for _ in range(n_assets)]

    # Initial guess: equal weight
    init_weights = np.ones(n_assets) / n_assets

    # Constraint: sum of weights = 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if target == "min_volatility":
        def objective(w):
            return np.sqrt(np.dot(w.T, np.dot(cov_ann, w)))

    elif target == "max_sharpe":
        def objective(w):
            port_ret = np.dot(w, mu_ann)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_ann, w)))
            if port_vol < 1e-8:
                return 0.0
            return -(port_ret - risk_free_rate) / port_vol

    elif target == "target_return":
        if target_return is None:
            raise ValueError("target_return must be provided for target_return optimization.")
        def objective(w):
            return np.dot(w.T, np.dot(cov_ann, w))
        constraints.append({
            "type": "eq",
            "fun": lambda w: np.dot(w, mu_ann) - target_return,
        })
    else:
        raise ValueError(f"Unknown target: {target}")

    res = minimize(
        objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    if not res.success:
        logger.warning(f"Optimization warning: {res.message}. Returning normalized result.")

    opt_w = np.clip(res.x, 0.0, 1.0)
    opt_w = opt_w / np.sum(opt_w)

    w_dict = {tickers[i]: float(opt_w[i]) for i in range(n_assets)}
    return opt_w, w_dict


def compute_efficient_frontier(
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0.035,
    n_points: int = 60,
) -> pd.DataFrame:
    """
    Generate points along the Markowitz Efficient Frontier.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Target_Return', 'Volatility', 'Sharpe_Ratio'] + weight columns.
    """
    tickers = list(returns_df.columns)
    _, min_vol_dict = optimize_markowitz_portfolio(returns_df, target="min_volatility")
    _, max_sharpe_dict = optimize_markowitz_portfolio(returns_df, target="max_sharpe", risk_free_rate=risk_free_rate)

    mu_ann = returns_df.mean().values * TRADING_DAYS_PER_YEAR
    cov_ann = returns_df.cov().values * TRADING_DAYS_PER_YEAR

    min_vol_w = np.array([min_vol_dict[t] for t in tickers])
    r_min = float(np.dot(min_vol_w, mu_ann))
    r_max = float(mu_ann.max())

    target_returns = np.linspace(r_min, r_max, n_points)
    frontier_points = []

    for r_target in target_returns:
        try:
            w_opt, w_dict = optimize_markowitz_portfolio(
                returns_df,
                target="target_return",
                target_return=r_target,
            )
            vol = float(np.sqrt(np.dot(w_opt.T, np.dot(cov_ann, w_opt))))
            sharpe = (r_target - risk_free_rate) / vol if vol > 1e-8 else 0.0

            pt = {
                "Target_Return": r_target,
                "Volatility": vol,
                "Sharpe_Ratio": sharpe,
            }
            pt.update(w_dict)
            frontier_points.append(pt)
        except Exception:
            continue

    return pd.DataFrame(frontier_points)


def simulate_random_portfolios(
    returns_df: pd.DataFrame,
    n_portfolios: int = 3000,
    risk_free_rate: float = 0.035,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate random long-only portfolios for Monte Carlo visual scatter on the risk-return plane.
    """
    np.random.seed(seed)
    tickers = list(returns_df.columns)
    n_assets = len(tickers)

    mu_ann = returns_df.mean().values * TRADING_DAYS_PER_YEAR
    cov_ann = returns_df.cov().values * TRADING_DAYS_PER_YEAR

    # Dirichlet distribution gives uniform sample on simplex sum(w) = 1
    raw_weights = np.random.dirichlet(np.ones(n_assets), size=n_portfolios)

    returns = np.dot(raw_weights, mu_ann)
    # Vectorized portfolio volatility computation
    variances = np.einsum("ij,jk,ik->i", raw_weights, cov_ann, raw_weights)
    volatilities = np.sqrt(variances)
    sharpes = (returns - risk_free_rate) / volatilities

    sim_df = pd.DataFrame({
        "Annual_Return": returns,
        "Volatility": volatilities,
        "Sharpe_Ratio": sharpes,
    })
    for i, t in enumerate(tickers):
        sim_df[f"w_{t}"] = raw_weights[:, i]

    return sim_df


def build_standard_portfolios_suite(
    returns_df: pd.DataFrame,
    benchmark_col: str = BENCHMARK_EQUITY,
    risk_free_series: Optional[pd.Series] = None,
) -> Dict[str, Tuple[Dict[str, float], PortfolioKPIs, pd.Series]]:
    """
    Construct, compute, and compare all 5 institutional standard allocations:
    1. Serendib Conservative (Mandate Recommendation)
    2. Max Sharpe Portfolio
    3. Minimum Volatility Portfolio
    4. Classic 60/40 Equity/Bond
    5. Equal Weight (1/N)
    
    Returns
    -------
    Dict[str, Tuple[weights, kpi_object, daily_return_series]]
    """
    tickers = list(returns_df.columns)
    benchmark_series = returns_df[benchmark_col] if benchmark_col in returns_df.columns else None

    # 1. Optimal Portfolios
    _, max_sharpe_w = optimize_markowitz_portfolio(returns_df, target="max_sharpe")
    _, min_vol_w = optimize_markowitz_portfolio(returns_df, target="min_volatility")

    # 2. Portfolios Catalog
    catalogs = {
        "Serendib Conservative (Recommended)": SERENDIB_CONSERVATIVE_WEIGHTS,
        "Markowitz Max Sharpe": max_sharpe_w,
        "Minimum Variance": min_vol_w,
        "Classic 60/40 Benchmark": CLASSIC_60_40_WEIGHTS,
        "Equal Weight (1/N)": EQUAL_WEIGHTS,
    }

    results = {}
    for name, w in catalogs.items():
        ret_series = compute_portfolio_return_series(returns_df, w)
        kpis = calculate_portfolio_kpis(
            returns_series=ret_series,
            weights=w,
            name=name,
            benchmark_series=benchmark_series,
            risk_free_series=risk_free_series,
        )
        results[name] = (w, kpis, ret_series)

    return results


if __name__ == "__main__":
    from src.data_loader import load_market_dataset

    print("Testing portfolio_opt module...")
    md = load_market_dataset()
    suite = build_standard_portfolios_suite(md.simple_returns, risk_free_series=md.risk_free_daily)

    kpi_records = [suite[p][1].to_dict() for p in suite]
    kpi_df = pd.DataFrame(kpi_records)
    print("\n--- Model Portfolios Institutional KPIs ---")
    print(kpi_df[["Portfolio", "CAGR", "Annual Volatility", "Sharpe Ratio (Rf)", "Max Drawdown (MDD)", "Sortino Ratio", "End Wealth ($10M AUM)"]].to_string())
