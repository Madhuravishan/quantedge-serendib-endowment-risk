"""
Market Intelligence Executive Summary Module
QuantEdge 1.0 - Serendib University Endowment Risk Engine (Phase 3)

Synthesizes outputs from all intelligence sub-modules:
- Trend Intelligence (Breadth across the 7 ETFs)
- News Sentiment Intelligence (Positive/Negative sentiment scores & macro events)
- Economic Intelligence (FRED yield curve inversion & interest rate stance)
- Statistical Anomaly Alerts (Price, volume, and volatility shocks)
- Research Screening Opportunities (Passed/Failed counts)

Produces a unified Executive Macro Stance and summary dashboard state.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd

from live_engine.core.data_validator import DataFreshness


@dataclass
class ExecutiveMarketSummary:
    """Consolidated institutional macro intelligence summary."""
    timestamp: str
    macro_regime: str  # "🟢 Risk-On / Favorable", "🟡 Neutral / Transition Regime", "🔴 Risk-Off / Defensive"
    macro_regime_rationale: str
    bullish_assets_count: int
    bearish_assets_count: int
    neutral_assets_count: int
    total_assets_evaluated: int
    sentiment_stance: str
    news_sentiment_score: float
    positive_news_count: int
    negative_news_count: int
    yield_curve_regime: str
    yield_curve_spread_bps: Optional[float]
    active_anomalies_count: int
    high_severity_anomalies_count: int
    screening_opportunities_count: int
    data_freshness: DataFreshness
    key_takeaways: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Analysis Timestamp": self.timestamp,
            "Macro Regime": self.macro_regime,
            "Rationale": self.macro_regime_rationale,
            "ETF Breadth (Bull/Bear/Neut)": f"{self.bullish_assets_count} / {self.bearish_assets_count} / {self.neutral_assets_count}",
            "News Sentiment": f"{self.sentiment_stance} ({self.news_sentiment_score:+.2f})",
            "Yield Curve": f"{self.yield_curve_regime} ({self.yield_curve_spread_bps:+.0f} bps)" if self.yield_curve_spread_bps is not None else self.yield_curve_regime,
            "Active Anomalies": f"{self.active_anomalies_count} ({self.high_severity_anomalies_count} High Severity)",
            "Screening Opportunities": f"{self.screening_opportunities_count} Assets Passed",
            "Data Freshness": self.data_freshness.value,
        }


def compile_market_intelligence_summary(
    trend_insights: Dict[str, Any],
    news_analysis: Dict[str, Any],
    economic_analysis: Dict[str, Any],
    anomaly_analysis: Dict[str, Any],
    screening_analysis: Dict[str, Any],
    data_freshness: DataFreshness = DataFreshness.FRESH,
) -> ExecutiveMarketSummary:
    """
    Synthesize all intelligence sub-engines into an executive macro assessment.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Trend Breadth
    total_assets = len(trend_insights)
    bull_count = sum(1 for t in trend_insights.values() if getattr(t, "classification", "") == "Bullish")
    bear_count = sum(1 for t in trend_insights.values() if getattr(t, "classification", "") == "Bearish")
    neut_count = sum(1 for t in trend_insights.values() if getattr(t, "classification", "") == "Neutral")

    # 2. News Sentiment
    sentiment_stance = news_analysis.get("sentiment_stance", "Balanced / Neutral Sentiment")
    sentiment_score = float(news_analysis.get("average_sentiment", 0.0))
    pos_news = int(news_analysis.get("positive_count", 0))
    neg_news = int(news_analysis.get("negative_count", 0))

    # 3. Economic & Yield Curve
    yc = economic_analysis.get("yield_curve")
    yc_regime = getattr(yc, "status", "Unavailable") if yc else "Unavailable"
    yc_bps = getattr(yc, "spread_bps", None) if yc else None

    # 4. Anomalies
    total_anomalies = int(anomaly_analysis.get("total_anomalies", 0))
    high_anomalies = int(anomaly_analysis.get("high_severity_count", 0))

    # 5. Screener
    passed_screener = int(screening_analysis.get("passed_count", 0))

    # 6. Synthesize Macro Regime
    # Scoring system (-3.0 to +3.0)
    macro_score = 0.0

    # Trend component (-1.0 to +1.0)
    if total_assets > 0:
        net_breadth = (bull_count - bear_count) / total_assets
        macro_score += net_breadth * 1.5

    # Yield curve component (-1.0 to +1.0)
    if yc_regime == "Inverted Yield Curve":
        macro_score -= 1.0
    elif yc_regime == "Normal / Upward Sloping":
        macro_score += 0.5

    # Sentiment component (-0.5 to +0.5)
    macro_score += sentiment_score * 0.5

    # Anomaly component
    if high_anomalies >= 2:
        macro_score -= 0.5

    takeaways: List[str] = []

    if macro_score >= 0.75:
        regime = "🟢 Risk-On / Favorable"
        rationale = "Constructive equity momentum, healthy market breadth, and supportive macro conditions favor growth allocations."
        takeaways.append(f"Strong ETF breadth with {bull_count}/{total_assets} universe assets exhibiting bullish moving average alignment.")
    elif macro_score <= -0.75:
        regime = "🔴 Risk-Off / Defensive Regime"
        rationale = "Elevated downside momentum, potential yield curve inversion, or risk anomalies warrant prioritizing defensive allocations (AGG, GLD)."
        takeaways.append(f"Weak ETF breadth with {bear_count}/{total_assets} assets under downward pressure.")
    else:
        regime = "🟡 Neutral / Transition Regime"
        rationale = "Divergent asset co-movements and mixed macro indicators suggest balanced portfolio maintenance within benchmark corridors."
        takeaways.append(f"Mixed asset breadth ({bull_count} Bullish, {bear_count} Bearish, {neut_count} Neutral) reflects transitional sector rotation.")

    if yc_regime == "Inverted Yield Curve":
        takeaways.append(f"Active 10Y-3M Yield Curve Inversion ({yc_bps:+.0f} bps) signals elevated medium-term recession risk.")
    elif yc_regime == "Normal / Upward Sloping":
        takeaways.append(f"Normal upward-sloping yield curve ({yc_bps:+.0f} bps) confirms healthy term premia.")

    if total_anomalies > 0:
        takeaways.append(f"{total_anomalies} statistical anomalies detected across the ETF universe ({high_anomalies} high severity).")
    else:
        takeaways.append("Zero abnormal statistical shocks or volume spikes detected across the universe.")

    if passed_screener > 0:
        takeaways.append(f"{passed_screener} ETF assets meet all systematic momentum and volatility screening criteria.")
    else:
        takeaways.append("No assets currently satisfy the strict multi-factor research criteria.")

    return ExecutiveMarketSummary(
        timestamp=now_str,
        macro_regime=regime,
        macro_regime_rationale=rationale,
        bullish_assets_count=bull_count,
        bearish_assets_count=bear_count,
        neutral_assets_count=neut_count,
        total_assets_evaluated=total_assets,
        sentiment_stance=sentiment_stance,
        news_sentiment_score=sentiment_score,
        positive_news_count=pos_news,
        negative_news_count=neg_news,
        yield_curve_regime=yc_regime,
        yield_curve_spread_bps=yc_bps,
        active_anomalies_count=total_anomalies,
        high_severity_anomalies_count=high_anomalies,
        screening_opportunities_count=passed_screener,
        data_freshness=data_freshness,
        key_takeaways=takeaways,
    )
