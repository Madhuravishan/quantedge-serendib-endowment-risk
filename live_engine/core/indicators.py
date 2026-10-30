"""
Technical Indicators & Market Trend Analysis Engine
QuantEdge 1.0 - Serendib University Endowment Risk Engine (Phase 2)
"""

import math
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd


TECHNICAL_DISCLAIMER = (
    "Institutional Technical Disclaimer: Technical indicators (SMA 20/50/200, RSI 14, and rolling volatility) "
    "are purely empirical mathematical transformations of past market price history. They do NOT constitute "
    "guaranteed predictions of future price movement, financial advice, or investment recommendations."
)


def compute_rsi_14(series: pd.Series, periods: int = 14) -> pd.Series:
    """
    Calculate 14-period Relative Strength Index (RSI) using standard Wilder smoothing.
    
    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss
    """
    if len(series) < periods + 1:
        return pd.Series([np.nan] * len(series), index=series.index)

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's exponential smoothing
    avg_gain = gain.ewm(com=periods - 1, min_periods=periods).mean()
    avg_loss = loss.ewm(com=periods - 1, min_periods=periods).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # Where loss is 0 and gain > 0, RSI is 100; where both 0, RSI is 50
    rsi = rsi.fillna(100.0)
    rsi[avg_gain == 0] = 0.0
    return rsi


def compute_technical_indicators(
    df_observations: pd.DataFrame,
    trading_days_per_year: int = 252,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Calculate technical indicators and market trend classifications from price series.
    
    Parameters
    ----------
    df_observations : pd.DataFrame
        DataFrame containing at minimum 'Date' and 'Close' columns.
        Can be sorted ascending or descending.
    trading_days_per_year : int
        Annualization factor for volatility (default: 252).
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        1. Enriched DataFrame with technical indicator series (sorted ascending by Date).
        2. Summary metrics dictionary containing latest values and trend classification.
    """
    if df_observations is None or df_observations.empty or "Close" not in df_observations.columns:
        empty_metrics = {
            "status": "Insufficient data",
            "observation_count": 0,
            "latest_close": None,
            "daily_return": None,
            "sma_20": "Insufficient data",
            "sma_50": "Insufficient data",
            "sma_200": "Insufficient data",
            "rsi_14": "Insufficient data",
            "rolling_vol_20": "Insufficient data",
            "recent_direction": "Unknown",
            "trend_classification": "Neutral",
            "disclaimer": TECHNICAL_DISCLAIMER,
        }
        return pd.DataFrame(), empty_metrics

    # Work with a copy and ensure ascending date order for chronological calculations
    df = df_observations.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(by="Date", ascending=True).reset_index(drop=True)

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    n_obs = len(df)

    # 1. Daily return
    df["Daily_Return"] = df["Close"].pct_change()

    # 2. Simple Moving Averages
    df["SMA_20"] = df["Close"].rolling(window=20, min_periods=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50, min_periods=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200, min_periods=200).mean()

    # 3. Relative Strength Index (RSI 14)
    df["RSI_14"] = compute_rsi_14(df["Close"], periods=14)

    # 4. Rolling 20-Day Annualized Volatility
    df["Rolling_Vol_20"] = df["Daily_Return"].rolling(window=20, min_periods=20).std(ddof=1) * math.sqrt(trading_days_per_year) * 100.0

    # Extract latest observations
    latest_row = df.iloc[-1]
    prev_row = df.iloc[-2] if n_obs >= 2 else latest_row

    latest_close = float(latest_row["Close"]) if pd.notna(latest_row["Close"]) else None
    daily_ret = float(latest_row["Daily_Return"] * 100.0) if pd.notna(latest_row["Daily_Return"]) else None

    sma_20_val = float(latest_row["SMA_20"]) if pd.notna(latest_row["SMA_20"]) else None
    sma_50_val = float(latest_row["SMA_50"]) if pd.notna(latest_row["SMA_50"]) else None
    sma_200_val = float(latest_row["SMA_200"]) if pd.notna(latest_row["SMA_200"]) else None
    rsi_14_val = float(latest_row["RSI_14"]) if pd.notna(latest_row["RSI_14"]) else None
    vol_20_val = float(latest_row["Rolling_Vol_20"]) if pd.notna(latest_row["Rolling_Vol_20"]) else None

    # Recent 5-day price direction
    if n_obs >= 5:
        five_day_change = (df["Close"].iloc[-1] - df["Close"].iloc[-5]) / df["Close"].iloc[-5]
        if five_day_change > 0.005:
            recent_direction = "Upward (+)"
        elif five_day_change < -0.005:
            recent_direction = "Downward (-)"
        else:
            recent_direction = "Sideways / Flat"
    else:
        recent_direction = "Upward (+)" if (daily_ret and daily_ret > 0) else "Downward (-)" if (daily_ret and daily_ret < 0) else "Flat"

    # Descriptive Trend Classification (Bullish / Bearish / Neutral)
    bullish_signals = 0
    bearish_signals = 0

    if sma_20_val is not None and latest_close is not None:
        if latest_close > sma_20_val:
            bullish_signals += 1
        else:
            bearish_signals += 1

    if sma_50_val is not None and latest_close is not None:
        if latest_close > sma_50_val:
            bullish_signals += 1
        else:
            bearish_signals += 1

    if sma_20_val is not None and sma_50_val is not None:
        if sma_20_val > sma_50_val:
            bullish_signals += 1
        else:
            bearish_signals += 1

    if rsi_14_val is not None:
        if rsi_14_val > 55.0:
            bullish_signals += 1
        elif rsi_14_val < 45.0:
            bearish_signals += 1

    if bullish_signals >= 3:
        classification = "Bullish"
    elif bearish_signals >= 3:
        classification = "Bearish"
    else:
        classification = "Neutral"

    summary_metrics = {
        "status": "Computed",
        "observation_count": n_obs,
        "latest_close": latest_close,
        "daily_return_pct": daily_ret,
        "sma_20": f"${sma_20_val:.2f}" if sma_20_val is not None else "Insufficient data",
        "sma_50": f"${sma_50_val:.2f}" if sma_50_val is not None else "Insufficient data",
        "sma_200": f"${sma_200_val:.2f}" if sma_200_val is not None else "Insufficient data",
        "sma_20_numeric": sma_20_val,
        "sma_50_numeric": sma_50_val,
        "sma_200_numeric": sma_200_val,
        "rsi_14": f"{rsi_14_val:.1f}" if rsi_14_val is not None else "Insufficient data",
        "rsi_14_numeric": rsi_14_val,
        "rolling_vol_20": f"{vol_20_val:.2f}%" if vol_20_val is not None else "Insufficient data",
        "rolling_vol_20_numeric": vol_20_val,
        "recent_direction": recent_direction,
        "trend_classification": classification,
        "disclaimer": TECHNICAL_DISCLAIMER,
    }

    # Format Date column back to YYYY-MM-DD string for display
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    return df, summary_metrics
