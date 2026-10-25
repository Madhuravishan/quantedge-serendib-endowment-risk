"""
Data Governance, Ingestion & Stationarity Module
Mandate: The Serendib University Endowment Fund (USD 10,000,000 AUM)
QuantEdge 1.0 National STEM Competition - Team 'Pandas Predictors'

Key Responsibilities:
1. Ingest daily adjusted close prices (2006-01-01 to 2026-10-20) for 7-ETF universe.
2. Ingest risk-free rate proxy (FRED DTB3 / Yahoo Finance ^IRX).
3. Institutional data cleaning: calendar alignment, stale price detection, jump anomaly checks.
4. Calculate continuous logarithmic returns (statistical modeling) and simple returns (portfolio cross-section).
5. Formal Augmented Dickey-Fuller (ADF) unit-root tests proving I(1) non-stationarity of prices vs I(0) stationarity of returns.
6. Deterministic, local file caching for speed and auditability.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import io
import warnings

import numpy as np
import pandas as pd
import requests
import statsmodels.tsa.stattools as smt
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)


from src.config import (
    TICKERS,
    FULL_SAMPLE_START,
    FULL_SAMPLE_END,
    RISK_FREE_FRED_ID,
    RISK_FREE_YF_SYMBOL,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    TRADING_DAYS_PER_YEAR,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@dataclass
class MarketData:
    """Institutional container for cleansed market prices, returns, and diagnostics."""
    prices: pd.DataFrame
    log_returns: pd.DataFrame
    simple_returns: pd.DataFrame
    risk_free_daily: pd.Series
    risk_free_annual: pd.Series
    stationarity_report: pd.DataFrame
    quality_audit: Dict[str, any]


def fetch_raw_etf_prices(
    tickers: Optional[List[str]] = None,
    start: str = FULL_SAMPLE_START,
    end: str = FULL_SAMPLE_END,
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Ingest daily adjusted close prices for the 7-asset universe.
    
    Parameters
    ----------
    tickers : list of str, optional
        ETF symbols to ingest. Defaults to config.TICKERS.
    start : str
        Start date string (YYYY-MM-DD).
    end : str
        End date string (YYYY-MM-DD).
    force_reload : bool
        If True, ignore local cache and re-download from Yahoo Finance.
        
    Returns
    -------
    pd.DataFrame
        Daily adjusted close price matrix with DateTimeIndex.
    """
    tickers = tickers or TICKERS
    cache_csv = DATA_RAW_DIR / "etf_prices_raw.csv"
    cache_parquet = DATA_RAW_DIR / "etf_prices_raw.parquet"

    if not force_reload:
        if cache_parquet.exists():
            try:
                df = pd.read_parquet(cache_parquet)
                logger.info(f"Loaded {len(df)} price observations from parquet cache.")
                return df
            except Exception as e:
                logger.warning(f"Failed to read parquet cache ({e}). Trying CSV.")
        if cache_csv.exists():
            df = pd.read_csv(cache_csv, index_col=0, parse_dates=True)
            logger.info(f"Loaded {len(df)} price observations from CSV cache.")
            return df

    logger.info(f"Downloading historical ETF data for {tickers} from {start} to {end}...")
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    if raw.empty:
        raise ValueError(f"Yahoo Finance returned empty data for tickers: {tickers}")

    # Extract Adjusted Close
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.levels[0]:
            prices = raw["Adj Close"].copy()
        elif "Close" in raw.columns.levels[0]:
            prices = raw["Close"].copy()
        else:
            raise KeyError(f"Neither 'Adj Close' nor 'Close' found in columns: {raw.columns}")
    else:
        prices = raw.copy()

    # Reorder columns to match canonical TICKERS list
    avail_cols = [t for t in tickers if t in prices.columns]
    prices = prices[avail_cols]
    prices.index.name = "Date"

    # Persist cache
    prices.to_csv(cache_csv)
    try:
        prices.to_parquet(cache_parquet)
    except Exception as e:
        logger.debug(f"Parquet save skipped ({e}). CSV preserved.")

    logger.info(f"Successfully downloaded and cached {len(prices)} trading days for {len(avail_cols)} assets.")
    return prices


def fetch_risk_free_rate(
    start: str = FULL_SAMPLE_START,
    end: str = FULL_SAMPLE_END,
    force_reload: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """
    Ingest risk-free rate proxy from Federal Reserve Economic Data (FRED DTB3: 3-Month T-Bill),
    with automatic failover to Yahoo Finance ^IRX (13-Week Treasury Bill).
    
    Parameters
    ----------
    start : str
        Start date string.
    end : str
        End date string.
    force_reload : bool
        If True, re-download even if cache exists.
        
    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (risk_free_daily_rate, risk_free_annual_rate) in decimal format (e.g. 0.045 for 4.5%).
    """
    cache_csv = DATA_RAW_DIR / "risk_free_rate.csv"

    if not force_reload and cache_csv.exists():
        df_rf = pd.read_csv(cache_csv, index_col=0, parse_dates=True)
        logger.info(f"Loaded risk-free rate from cache ({len(df_rf)} rows).")
        return df_rf["rf_daily"], df_rf["rf_annual"]

    logger.info("Attempting to ingest 3-Month T-Bill from St. Louis FRED (DTB3)...")
    rf_series: Optional[pd.Series] = None

    try:
        fred_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={RISK_FREE_FRED_ID}"
        resp = requests.get(fred_url, timeout=10)
        if resp.status_code == 200:
            df_fred = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
            # FRED uses '.' to represent market holidays
            df_fred[RISK_FREE_FRED_ID] = pd.to_numeric(df_fred[RISK_FREE_FRED_ID], errors="coerce")
            df_fred = df_fred.dropna()
            df_fred = df_fred.loc[(df_fred.index >= start) & (df_fred.index <= end)]
            if not df_fred.empty:
                rf_series = df_fred[RISK_FREE_FRED_ID] / 100.0  # Convert percent to decimal
                logger.info(f"Successfully loaded FRED DTB3 series ({len(rf_series)} dates).")
    except Exception as e:
        logger.warning(f"FRED ingestion failed: {e}. Falling back to Yahoo Finance ^IRX...")

    # Fallback to Yahoo Finance ^IRX if FRED failed
    if rf_series is None or rf_series.empty:
        logger.info(f"Ingesting ^IRX from Yahoo Finance...")
        raw_irx = yf.download(RISK_FREE_YF_SYMBOL, start=start, end=end, progress=False)
        if not raw_irx.empty:
            if isinstance(raw_irx.columns, pd.MultiIndex):
                close_col = raw_irx["Adj Close"] if "Adj Close" in raw_irx.columns.levels[0] else raw_irx["Close"]
                if isinstance(close_col, pd.DataFrame):
                    rf_series = close_col.iloc[:, 0] / 100.0
                else:
                    rf_series = close_col / 100.0
            else:
                rf_series = raw_irx["Close"] / 100.0
            rf_series = rf_series.dropna()
            logger.info(f"Successfully loaded ^IRX series ({len(rf_series)} dates).")
        else:
            # Conservative static fallback: 3.5% long-term historical average
            logger.warning("Could not ingest live risk-free series. Utilizing statutory 3.50% constant.")
            dates = pd.date_range(start=start, end=end, freq="B")
            rf_series = pd.Series(0.035, index=dates)

    # Convert annual yield to daily compounding yield
    # r_daily = (1 + r_annual)^(1/252) - 1 or simple r_annual / 252
    rf_series.index = pd.to_datetime(rf_series.index)
    rf_daily = rf_series / TRADING_DAYS_PER_YEAR

    df_out = pd.DataFrame({"rf_annual": rf_series, "rf_daily": rf_daily})
    df_out.index.name = "Date"
    df_out.to_csv(cache_csv)

    return df_out["rf_daily"], df_out["rf_annual"]


def clean_price_data(
    df_prices: pd.DataFrame,
    max_jump_threshold: float = 0.35,
    max_flatline_days: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Audit and clean price matrix following strict data governance principles.
    
    Data Cleaning Rules:
    1. Verify date index monotonicity, eliminate duplicate timestamps.
    2. Check for missing values and document holiday alignment.
    3. Detect stale / flatlined prices (> max_flatline_days identical prices).
    4. Detect extreme unadjusted jumps (> max_jump_threshold daily shift).
    5. Forward-fill non-overlapping exchange holidays, drop residual initial NaNs.
    
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, any]]
        Cleaned prices DataFrame and audit dictionary.
    """
    audit: Dict[str, any] = {
        "initial_rows": len(df_prices),
        "initial_start": str(df_prices.index.min()),
        "initial_end": str(df_prices.index.max()),
        "tickers": list(df_prices.columns),
        "missing_values_by_ticker": df_prices.isnull().sum().to_dict(),
        "flatline_warnings": {},
        "jump_warnings": {},
        "dropped_rows_count": 0,
    }

    # Ensure chronological order and remove duplicate timestamps
    df = df_prices[~df_prices.index.duplicated(keep="first")].sort_index()

    # Forward fill trading calendar discrepancies (e.g. minor holiday differences)
    df_ffill = df.ffill()

    # Drop any leading rows where assets hadn't commenced trading yet
    df_clean = df_ffill.dropna()
    audit["dropped_rows_count"] = len(df) - len(df_clean)
    audit["final_rows"] = len(df_clean)
    audit["final_start"] = str(df_clean.index.min())
    audit["final_end"] = str(df_clean.index.max())

    # Detect price jumps and flatlines
    for col in df_clean.columns:
        pct_chg = df_clean[col].pct_change().dropna()
        extreme_jumps = pct_chg[pct_chg.abs() > max_jump_threshold]
        if not extreme_jumps.empty:
            audit["jump_warnings"][col] = [
                {"date": str(d), "return": float(r)} for d, r in extreme_jumps.items()
            ]

        # Flatlines
        diffs = df_clean[col].diff().fillna(1.0)
        zero_runs = (diffs == 0).astype(int)
        # Consecutive zero run length
        streak = zero_runs.groupby((zero_runs != zero_runs.shift()).cumsum()).cumsum()
        max_streak = streak.max()
        if max_streak >= max_flatline_days:
            audit["flatline_warnings"][col] = int(max_streak)

    logger.info(
        f"Data cleaning complete: {audit['initial_rows']} -> {audit['final_rows']} rows. "
        f"Jump warnings: {len(audit['jump_warnings'])}, Flatlines: {len(audit['flatline_warnings'])}."
    )
    return df_clean, audit


def compute_returns(df_prices: pd.DataFrame, save_cache: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate logarithmic and simple returns from cleaned price series.
    
    Mathematical Formulation:
    - Logarithmic Returns: r_t = ln(P_t / P_{t-1})
      Property: Time-additive (sum_t r_t = ln(P_T / P_0)), suited for diffusion modeling,
      ADF stationarity, and EVT tail characterization.
    - Simple Arithmetic Returns: R_t = (P_t - P_{t-1}) / P_{t-1}
      Property: Cross-sectionally additive (R_p,t = sum_i w_i R_i,t), required for accurate
      portfolio valuation and financial accounting.
      
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (log_returns, simple_returns)
    """
    # Simple Returns
    simple_ret = df_prices.pct_change().dropna()

    # Log Returns: ln(P_t) - ln(P_{t-1})
    log_ret = np.log(df_prices / df_prices.shift(1)).dropna()

    # Persist processed datasets if requested
    if save_cache:
        log_ret.to_csv(DATA_PROCESSED_DIR / "returns_log.csv")
        simple_ret.to_csv(DATA_PROCESSED_DIR / "returns_simple.csv")

    return log_ret, simple_ret


def run_adf_stationarity_test(series: pd.Series, maxlag: Optional[int] = None) -> Dict[str, any]:
    """
    Perform formal Augmented Dickey-Fuller (ADF) test for unit root non-stationarity.
    
    Model with constant:
    Delta y_t = alpha + beta * t + gamma * y_{t-1} + sum_{i=1}^p delta_i Delta y_{t-i} + epsilon_t
    
    Hypothesis:
    H_0 : gamma = 0 (Presence of unit root -> Non-Stationary I(1))
    H_1 : gamma < 0 (Rejection of unit root -> Stationary I(0))
    
    Returns
    -------
    Dict[str, any]
        Test statistic, p-value, critical values, and stationarity verdict.
    """
    clean_s = series.dropna()
    try:
        result = smt.adfuller(clean_s, maxlag=maxlag, regression="c", autolag="AIC", result_object=False)
    except TypeError:
        result = smt.adfuller(clean_s, maxlag=maxlag, regression="c", autolag="AIC")

    stat, p_val, usedlag, nobs, crit_vals, _ = result

    # Standard 5% significance threshold
    is_stat = p_val < 0.05

    return {
        "adf_statistic": float(stat),
        "p_value": float(p_val),
        "lags_used": int(usedlag),
        "observations": int(nobs),
        "crit_1pct": float(crit_vals["1%"]),
        "crit_5pct": float(crit_vals["5%"]),
        "crit_10pct": float(crit_vals["10%"]),
        "is_stationary": bool(is_stat),
        "conclusion": "Stationary I(0)" if is_stat else "Non-Stationary I(1)",
    }


def generate_stationarity_report(
    df_prices: pd.DataFrame,
    df_log_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Execute ADF tests across all 7 assets for both raw prices and return series.
    Provides formal mathematical proof of price non-stationarity I(1) vs return stationarity I(0).
    """
    records = []

    for ticker in df_prices.columns:
        # 1. Price Series (Log Price)
        log_price = np.log(df_prices[ticker])
        p_res = run_adf_stationarity_test(log_price)
        records.append({
            "Ticker": ticker,
            "Series": "Log Price ln(P_t)",
            "ADF Statistic": round(p_res["adf_statistic"], 4),
            "p-value": f"{p_res['p_value']:.4e}",
            "Critical 1%": round(p_res["crit_1pct"], 4),
            "Critical 5%": round(p_res["crit_5pct"], 4),
            "Stationary (5%)": p_res["is_stationary"],
            "Order of Integration": "I(1) Unit Root" if not p_res["is_stationary"] else "I(0)",
        })

        # 2. Return Series (Log Return)
        r_res = run_adf_stationarity_test(df_log_returns[ticker])
        records.append({
            "Ticker": ticker,
            "Series": "Log Return r_t",
            "ADF Statistic": round(r_res["adf_statistic"], 4),
            "p-value": f"{r_res['p_value']:.4e}",
            "Critical 1%": round(r_res["crit_1pct"], 4),
            "Critical 5%": round(r_res["crit_5pct"], 4),
            "Stationary (5%)": r_res["is_stationary"],
            "Order of Integration": "I(0) Stationary" if r_res["is_stationary"] else "I(1)",
        })

    report_df = pd.DataFrame(records)
    report_df.to_csv(DATA_PROCESSED_DIR / "adf_stationarity_report.csv", index=False)
    return report_df


def load_market_dataset(force_reload: bool = False) -> MarketData:
    """
    High-level orchestration function to fetch, clean, process, and validate the
    entire market dataset for the Serendib Endowment Fund mandate.
    """
    raw_prices = fetch_raw_etf_prices(force_reload=force_reload)
    clean_prices, quality_audit = clean_price_data(raw_prices)
    log_ret, simple_ret = compute_returns(clean_prices, save_cache=True)

    rf_daily, rf_annual = fetch_risk_free_rate(
        start=str(clean_prices.index.min().strftime("%Y-%m-%d")),
        end=str(clean_prices.index.max().strftime("%Y-%m-%d")),
        force_reload=force_reload,
    )

    # Align risk-free series with trading dates
    rf_daily_aligned = rf_daily.reindex(log_ret.index).ffill().bfill()
    rf_annual_aligned = rf_annual.reindex(log_ret.index).ffill().bfill()

    # Generate stationarity audit
    stat_report = generate_stationarity_report(clean_prices, log_ret)

    logger.info("Market dataset fully prepared, verified, and aligned.")
    return MarketData(
        prices=clean_prices,
        log_returns=log_ret,
        simple_returns=simple_ret,
        risk_free_daily=rf_daily_aligned,
        risk_free_annual=rf_annual_aligned,
        stationarity_report=stat_report,
        quality_audit=quality_audit,
    )


if __name__ == "__main__":
    # Self-test execution
    print("Testing data_loader module...")
    md = load_market_dataset(force_reload=False)
    print("Prices shape:", md.prices.shape)
    print("Returns shape:", md.log_returns.shape)
    print("\nADF Stationarity Summary:")
    print(md.stationarity_report.to_string())
