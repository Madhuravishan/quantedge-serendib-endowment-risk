"""
Research Opportunity Screener Module
QuantEdge 1.0 - Serendib University Endowment Risk Engine (Phase 3)

Systematic screening engine filtering the endowment's 7 ETFs against configurable research criteria:
1. Positive Momentum: Short-term (5-day) or intermediate (20-day) momentum > 0%
2. Trend Alignment: Price trades above key moving averages (SMA 20 or SMA 50)
3. Volatility Constraint: Realized 20-day annualized volatility <= maximum tolerable limit
4. Momentum Health: 14-period RSI within healthy bounds (e.g. 40 <= RSI <= 70)
5. Overall Classification: Trend is Bullish or Neutral-to-Bullish

Provides granular pass/fail auditing for every criterion across every asset.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd

from live_engine.intelligence.trend_analyzer import TrendInsight, analyze_asset_trend

SCREENER_DISCLAIMER = (
    "Research Signal Notice: Screening results represent systematic algorithmic filtering based on mathematical "
    "criteria for quantitative research and institutional monitoring only. They do NOT constitute automated "
    "trading recommendations, buy/sell directives, investment advice, or execution mandates."
)


@dataclass
class ScreeningCriteria:
    """Configurable threshold parameters for opportunity screening."""
    require_positive_5d_momentum: bool = True
    require_positive_20d_momentum: bool = False
    require_price_above_sma20: bool = True
    require_price_above_sma50: bool = False
    max_rolling_volatility_pct: float = 25.0
    min_rsi_14: float = 40.0
    max_rsi_14: float = 70.0
    allowed_classifications: List[str] = field(default_factory=lambda: ["Bullish", "Neutral"])


@dataclass
class AssetScreeningResult:
    """Audit evaluation of a single ETF against all screening criteria."""
    symbol: str
    name: str
    overall_pass: bool
    passed_rules_count: int
    total_rules_count: int
    rule_results: Dict[str, Dict[str, Any]]
    failure_reasons: List[str]
    trend_insight: TrendInsight

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Symbol": self.symbol,
            "Name": self.name,
            "Screening Status": "PASS" if self.overall_pass else "FAIL",
            "Rules Passed": f"{self.passed_rules_count}/{self.total_rules_count}",
            "Classification": self.trend_insight.classification,
            "Price": f"${self.trend_insight.latest_price:.2f}" if self.trend_insight.latest_price else "N/A",
            "5d Mom": f"{self.trend_insight.momentum_5d_pct:+.2f}%" if self.trend_insight.momentum_5d_pct is not None else "N/A",
            "RSI 14": f"{self.trend_insight.rsi_14:.1f}" if self.trend_insight.rsi_14 is not None else "N/A",
            "Vol 20": f"{self.trend_insight.rolling_vol_20:.2f}%" if self.trend_insight.rolling_vol_20 is not None else "N/A",
            "Failure Reasons": "; ".join(self.failure_reasons) if self.failure_reasons else "None (Passed All Criteria)",
        }


def screen_asset(
    symbol: str,
    trend_insight: TrendInsight,
    criteria: ScreeningCriteria,
) -> AssetScreeningResult:
    """
    Evaluate an asset against research screening criteria and generate an audit record.
    """
    rules: Dict[str, Dict[str, Any]] = {}
    failures: List[str] = []

    # Rule 1: Positive 5-day momentum
    if criteria.require_positive_5d_momentum:
        val = trend_insight.momentum_5d_pct
        passed = (val is not None and val > 0.0)
        rules["Positive 5D Momentum"] = {"passed": passed, "observed": f"{val:+.2f}%" if val is not None else "N/A", "target": "> 0.0%"}
        if not passed:
            failures.append(f"5-day momentum is negative or unavailable ({rules['Positive 5D Momentum']['observed']})")

    # Rule 2: Positive 20-day momentum
    if criteria.require_positive_20d_momentum:
        val = trend_insight.momentum_20d_pct
        passed = (val is not None and val > 0.0)
        rules["Positive 20D Momentum"] = {"passed": passed, "observed": f"{val:+.2f}%" if val is not None else "N/A", "target": "> 0.0%"}
        if not passed:
            failures.append(f"20-day momentum is negative ({rules['Positive 20D Momentum']['observed']})")

    # Rule 3: Price above SMA 20
    if criteria.require_price_above_sma20:
        p = trend_insight.latest_price
        ma20 = trend_insight.sma_20
        passed = (p is not None and ma20 is not None and p > ma20)
        obs_str = f"${p:.2f} vs ${ma20:.2f}" if (p and ma20) else "N/A"
        rules["Price > SMA 20"] = {"passed": passed, "observed": obs_str, "target": "Price > SMA 20"}
        if not passed:
            failures.append(f"Price is below 20-day moving average ({obs_str})")

    # Rule 4: Price above SMA 50
    if criteria.require_price_above_sma50:
        p = trend_insight.latest_price
        ma50 = trend_insight.sma_50
        passed = (p is not None and ma50 is not None and p > ma50)
        obs_str = f"${p:.2f} vs ${ma50:.2f}" if (p and ma50) else "N/A"
        rules["Price > SMA 50"] = {"passed": passed, "observed": obs_str, "target": "Price > SMA 50"}
        if not passed:
            failures.append(f"Price is below 50-day moving average ({obs_str})")

    # Rule 5: Volatility Constraint
    if criteria.max_rolling_volatility_pct is not None:
        vol = trend_insight.rolling_vol_20
        passed = (vol is not None and vol <= criteria.max_rolling_volatility_pct)
        obs_str = f"{vol:.2f}%" if vol is not None else "N/A"
        rules["Volatility Tolerance"] = {"passed": passed, "observed": obs_str, "target": f"<= {criteria.max_rolling_volatility_pct:.1f}%"}
        if not passed:
            failures.append(f"Rolling volatility ({obs_str}) exceeds ceiling ({criteria.max_rolling_volatility_pct:.1f}%)")

    # Rule 6: RSI Momentum Health
    if criteria.min_rsi_14 is not None and criteria.max_rsi_14 is not None:
        rsi = trend_insight.rsi_14
        passed = (rsi is not None and criteria.min_rsi_14 <= rsi <= criteria.max_rsi_14)
        obs_str = f"{rsi:.1f}" if rsi is not None else "N/A"
        rules["RSI Health Range"] = {"passed": passed, "observed": obs_str, "target": f"{criteria.min_rsi_14:.0f} - {criteria.max_rsi_14:.0f}"}
        if not passed:
            failures.append(f"RSI ({obs_str}) outside target bounds ({criteria.min_rsi_14:.0f}-{criteria.max_rsi_14:.0f})")

    # Rule 7: Trend Classification
    if criteria.allowed_classifications:
        passed = (trend_insight.classification in criteria.allowed_classifications)
        rules["Allowed Classification"] = {"passed": passed, "observed": trend_insight.classification, "target": ", ".join(criteria.allowed_classifications)}
        if not passed:
            failures.append(f"Classification is {trend_insight.classification} (must be {', '.join(criteria.allowed_classifications)})")

    total_rules = len(rules)
    passed_rules = sum(1 for r in rules.values() if r["passed"])
    overall_pass = (passed_rules == total_rules) and (total_rules > 0)

    return AssetScreeningResult(
        symbol=symbol,
        name=trend_insight.name,
        overall_pass=overall_pass,
        passed_rules_count=passed_rules,
        total_rules_count=total_rules,
        rule_results=rules,
        failure_reasons=failures,
        trend_insight=trend_insight,
    )


def run_universe_screening(
    insights_dict: Dict[str, TrendInsight],
    criteria: Optional[ScreeningCriteria] = None,
) -> Dict[str, Any]:
    """
    Execute systematic research screening across all 7 universe assets.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with passed assets, failed assets, audit DataFrame, and summary counts.
    """
    crit = criteria or ScreeningCriteria()
    results: List[AssetScreeningResult] = []

    for sym, insight in insights_dict.items():
        res = screen_asset(sym, insight, crit)
        results.append(res)

    passed_list = [r for r in results if r.overall_pass]
    failed_list = [r for r in results if not r.overall_pass]

    audit_records = [r.to_dict() for r in results]
    audit_df = pd.DataFrame(audit_records)

    return {
        "criteria": crit,
        "total_screened": len(results),
        "passed_count": len(passed_list),
        "failed_count": len(failed_list),
        "passed_assets": passed_list,
        "failed_assets": failed_list,
        "audit_df": audit_df,
        "disclaimer": SCREENER_DISCLAIMER,
    }
