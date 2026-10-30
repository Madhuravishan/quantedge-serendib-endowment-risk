"""
Market Trend Intelligence Module
QuantEdge 1.0 - Serendib University Endowment Risk Engine (Phase 3)

Analyzes historical price data for the endowment's 7 ETFs to generate:
- Comprehensive moving average analysis (SMA 20, 50, 200)
- Momentum and RSI (14) evaluations
- Transparent Bullish / Bearish / Neutral classifications
- Explicit explanations of contributing factors
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from live_engine.core.indicators import compute_technical_indicators, TECHNICAL_DISCLAIMER

ETF_NAMES = {
    "SPY": "SPDR S&P 500 ETF Trust",
    "EFA": "iShares MSCI EAFE ETF",
    "EEM": "iShares MSCI Emerging Markets ETF",
    "AGG": "iShares Core U.S. Aggregate Bond ETF",
    "GLD": "SPDR Gold Shares",
    "XLE": "Energy Select Sector SPDR Fund",
    "XLK": "Technology Select Sector SPDR Fund",
}


@dataclass
class TrendInsight:
    """Encapsulates detailed trend intelligence for an asset."""
    symbol: str
    name: str
    classification: str  # "Bullish", "Bearish", "Neutral"
    strength_score: float  # 0 to 100
    latest_price: Optional[float]
    daily_return_pct: Optional[float]
    momentum_5d_pct: Optional[float]
    momentum_20d_pct: Optional[float]
    sma_20: Optional[float]
    sma_50: Optional[float]
    sma_200: Optional[float]
    rsi_14: Optional[float]
    rolling_vol_20: Optional[float]
    golden_cross: Optional[bool]  # True if SMA 50 > SMA 200
    contributing_factors: List[str] = field(default_factory=list)
    disclaimer: str = TECHNICAL_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Symbol": self.symbol,
            "Name": self.name,
            "Classification": self.classification,
            "Strength Score": f"{self.strength_score:.1f}/100",
            "Price": f"${self.latest_price:.2f}" if self.latest_price else "N/A",
            "Daily Return": f"{self.daily_return_pct:+.2f}%" if self.daily_return_pct is not None else "N/A",
            "5-Day Momentum": f"{self.momentum_5d_pct:+.2f}%" if self.momentum_5d_pct is not None else "N/A",
            "20-Day Momentum": f"{self.momentum_20d_pct:+.2f}%" if self.momentum_20d_pct is not None else "N/A",
            "RSI (14)": f"{self.rsi_14:.1f}" if self.rsi_14 is not None else "N/A",
            "SMA 20": f"${self.sma_20:.2f}" if self.sma_20 else "N/A",
            "SMA 50": f"${self.sma_50:.2f}" if self.sma_50 else "N/A",
            "SMA 200": f"${self.sma_200:.2f}" if self.sma_200 else "N/A",
            "Rolling Vol (20d)": f"{self.rolling_vol_20:.2f}%" if self.rolling_vol_20 is not None else "N/A",
            "Golden Cross": "Active (SMA50 > SMA200)" if self.golden_cross is True else "Inactive (SMA50 < SMA200)" if self.golden_cross is False else "N/A",
            "Contributing Factors Count": len(self.contributing_factors),
        }


def analyze_asset_trend(symbol: str, df_history: pd.DataFrame) -> TrendInsight:
    """
    Perform deep trend intelligence on an asset's historical price observations.
    
    Parameters
    ----------
    symbol : str
        ETF symbol (e.g. 'SPY').
    df_history : pd.DataFrame
        DataFrame with 'Date' and 'Close' columns.
        
    Returns
    -------
    TrendInsight
        Dataclass containing classification, score, and explicit contributing explanations.
    """
    sym = symbol.strip().upper()
    asset_name = ETF_NAMES.get(sym, sym)

    if df_history is None or df_history.empty or "Close" not in df_history.columns or len(df_history) < 5:
        return TrendInsight(
            symbol=sym,
            name=asset_name,
            classification="Neutral",
            strength_score=50.0,
            latest_price=None,
            daily_return_pct=None,
            momentum_5d_pct=None,
            momentum_20d_pct=None,
            sma_20=None,
            sma_50=None,
            sma_200=None,
            rsi_14=None,
            rolling_vol_20=None,
            golden_cross=None,
            contributing_factors=["Insufficient historical observations (<5 trading days) to establish trend."],
        )

    # Sort ascending for chronological calculation
    df = df_history.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(by="Date", ascending=True).reset_index(drop=True)

    df_enriched, metrics = compute_technical_indicators(df)
    n_obs = len(df_enriched)

    latest_price = metrics.get("latest_close")
    daily_ret = metrics.get("daily_return_pct")
    sma_20 = metrics.get("sma_20_numeric")
    sma_50 = metrics.get("sma_50_numeric")
    sma_200 = metrics.get("sma_200_numeric")
    rsi_14 = metrics.get("rsi_14_numeric")
    vol_20 = metrics.get("rolling_vol_20_numeric")

    # Momentum calculations
    mom_5d = None
    if n_obs >= 5:
        p_latest = df_enriched["Close"].iloc[-1]
        p_5d_ago = df_enriched["Close"].iloc[-5]
        if p_5d_ago > 0:
            mom_5d = float((p_latest - p_5d_ago) / p_5d_ago * 100.0)

    mom_20d = None
    if n_obs >= 20:
        p_latest = df_enriched["Close"].iloc[-1]
        p_20d_ago = df_enriched["Close"].iloc[-20]
        if p_20d_ago > 0:
            mom_20d = float((p_latest - p_20d_ago) / p_20d_ago * 100.0)

    # Golden cross / Death cross evaluation
    golden_cross = None
    if sma_50 is not None and sma_200 is not None:
        golden_cross = (sma_50 > sma_200)

    # Factor Evaluation & Score Aggregation
    bullish_points = 0.0
    total_evaluations = 0.0
    factors: List[str] = []

    # 1. Price vs SMA 20 (Short-term support)
    if latest_price and sma_20:
        total_evaluations += 1.0
        diff_pct = (latest_price - sma_20) / sma_20 * 100.0
        if latest_price > sma_20:
            bullish_points += 1.0
            factors.append(f"Price (${latest_price:.2f}) trades {diff_pct:+.2f}% above 20-day SMA (${sma_20:.2f}), confirming short-term momentum.")
        else:
            factors.append(f"Price (${latest_price:.2f}) trades {diff_pct:+.2f}% below 20-day SMA (${sma_20:.2f}), indicating short-term weakness.")

    # 2. Price vs SMA 50 (Intermediate trend benchmark)
    if latest_price and sma_50:
        total_evaluations += 1.5
        diff_pct = (latest_price - sma_50) / sma_50 * 100.0
        if latest_price > sma_50:
            bullish_points += 1.5
            factors.append(f"Price is {diff_pct:+.2f}% above intermediate 50-day SMA (${sma_50:.2f}), sustaining medium-term structural support.")
        else:
            factors.append(f"Price is {diff_pct:+.2f}% below intermediate 50-day SMA (${sma_50:.2f}), indicating medium-term overhead pressure.")

    # 3. Price vs SMA 200 (Long-term regime)
    if latest_price and sma_200:
        total_evaluations += 1.5
        diff_pct = (latest_price - sma_200) / sma_200 * 100.0
        if latest_price > sma_200:
            bullish_points += 1.5
            factors.append(f"Price trades {diff_pct:+.2f}% above secular 200-day SMA (${sma_200:.2f}), maintaining long-term secular bull trend.")
        else:
            factors.append(f"Price trades {diff_pct:+.2f}% below secular 200-day SMA (${sma_200:.2f}), reflecting long-term macro headwind.")

    # 4. Moving Average Alignment (SMA 20 vs SMA 50)
    if sma_20 and sma_50:
        total_evaluations += 1.0
        if sma_20 > sma_50:
            bullish_points += 1.0
            factors.append(f"Short-term 20-day SMA leads 50-day SMA (${sma_20:.2f} > ${sma_50:.2f}), verifying upward moving average slope.")
        else:
            factors.append(f"Short-term 20-day SMA lags 50-day SMA (${sma_20:.2f} < ${sma_50:.2f}), reflecting downward moving average curvature.")

    # 5. Golden Cross / Death Cross
    if golden_cross is not None:
        total_evaluations += 1.0
        if golden_cross:
            bullish_points += 1.0
            factors.append(f"Golden Cross active: 50-day SMA (${sma_50:.2f}) is above 200-day SMA (${sma_200:.2f}).")
        else:
            factors.append(f"Death Cross active: 50-day SMA (${sma_50:.2f}) is below 200-day SMA (${sma_200:.2f}).")

    # 6. RSI (14) Momentum Oscillator
    if rsi_14 is not None:
        total_evaluations += 1.0
        if rsi_14 > 70.0:
            bullish_points += 0.5  # Strong but overbought risk
            factors.append(f"RSI (14) is elevated at {rsi_14:.1f} (Overbought > 70): strong momentum but elevated short-term pullback risk.")
        elif rsi_14 > 50.0:
            bullish_points += 1.0
            factors.append(f"RSI (14) is healthy at {rsi_14:.1f} (> 50): bullish momentum without overbought stretch.")
        elif rsi_14 > 30.0:
            factors.append(f"RSI (14) is subdued at {rsi_14:.1f} (< 50): bearish momentum dominate buyers.")
        else:
            factors.append(f"RSI (14) is deeply oversold at {rsi_14:.1f} (< 30): persistent downward selling pressure.")

    # 7. Recent Momentum (5-day & 20-day)
    if mom_5d is not None:
        total_evaluations += 0.5
        if mom_5d > 0.5:
            bullish_points += 0.5
            factors.append(f"Recent 5-day momentum is positive ({mom_5d:+.2f}%).")
        elif mom_5d < -0.5:
            factors.append(f"Recent 5-day momentum is negative ({mom_5d:+.2f}%).")
        else:
            bullish_points += 0.25
            factors.append(f"Recent 5-day momentum is flat / consolidating ({mom_5d:+.2f}%).")

    # Calculate normalized strength score
    if total_evaluations > 0:
        strength_score = (bullish_points / total_evaluations) * 100.0
    else:
        strength_score = 50.0

    # Determine classification
    if strength_score >= 62.5:
        classification = "Bullish"
    elif strength_score <= 37.5:
        classification = "Bearish"
    else:
        classification = "Neutral"

    return TrendInsight(
        symbol=sym,
        name=asset_name,
        classification=classification,
        strength_score=round(strength_score, 1),
        latest_price=latest_price,
        daily_return_pct=daily_ret,
        momentum_5d_pct=mom_5d,
        momentum_20d_pct=mom_20d,
        sma_20=sma_20,
        sma_50=sma_50,
        sma_200=sma_200,
        rsi_14=rsi_14,
        rolling_vol_20=vol_20,
        golden_cross=golden_cross,
        contributing_factors=factors,
    )


def analyze_universe_trends(history_dict: Dict[str, pd.DataFrame]) -> Dict[str, TrendInsight]:
    """
    Analyze trend intelligence for the full 7-asset ETF universe.
    
    Parameters
    ----------
    history_dict : Dict[str, pd.DataFrame]
        Dictionary mapping ETF symbols to historical price DataFrames.
        
    Returns
    -------
    Dict[str, TrendInsight]
        Dictionary of trend insights keyed by symbol.
    """
    results: Dict[str, TrendInsight] = {}
    for sym in ["SPY", "EFA", "EEM", "AGG", "GLD", "XLE", "XLK"]:
        df = history_dict.get(sym)
        results[sym] = analyze_asset_trend(sym, df)
    return results


def get_trend_breadth(insights: Dict[str, TrendInsight]) -> Dict[str, Any]:
    """
    Calculate cross-asset trend breadth metrics.
    """
    total = len(insights)
    if total == 0:
        return {
            "total_analyzed": 0,
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "bullish_ratio": 0.0,
            "bearish_ratio": 0.0,
            "market_breadth_label": "No Historical Data Available",
        }

    bull_count = sum(1 for t in insights.values() if t.classification == "Bullish")
    bear_count = sum(1 for t in insights.values() if t.classification == "Bearish")
    neut_count = sum(1 for t in insights.values() if t.classification == "Neutral")

    return {
        "total_analyzed": total,
        "bullish_count": bull_count,
        "bearish_count": bear_count,
        "neutral_count": neut_count,
        "bullish_ratio": bull_count / total,
        "bearish_ratio": bear_count / total,
        "market_breadth_label": "Broad Bullish Breadth" if (bull_count / total) >= 0.57 else "Broad Bearish Breadth" if (bear_count / total) >= 0.57 else "Mixed / Divergent Breadth",
    }
