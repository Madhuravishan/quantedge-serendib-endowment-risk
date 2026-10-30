"""
Unit and Integration Tests for Live Market Intelligence Engine (Phase 3)
QuantEdge 1.0 - Serendib University Endowment Risk Engine
"""

import math
from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from live_engine.core.data_validator import DataFreshness
from live_engine.intelligence.trend_analyzer import (
    TrendInsight,
    analyze_asset_trend,
    analyze_universe_trends,
    get_trend_breadth,
)
from live_engine.intelligence.news_analyzer import (
    score_text_sentiment,
    detect_macro_events,
    detect_related_etfs,
    analyze_article,
    analyze_news_feed,
)
from live_engine.intelligence.economic_analyzer import (
    SeriesAnalysis,
    YieldCurveAnalysis,
    analyze_single_series,
    analyze_yield_curve_slope,
    analyze_all_economic_data,
)
from live_engine.intelligence.anomaly_detector import (
    AnomalyAlert,
    detect_asset_anomalies,
    scan_universe_anomalies,
)
from live_engine.intelligence.opportunity_screener import (
    ScreeningCriteria,
    AssetScreeningResult,
    screen_asset,
    run_universe_screening,
    SCREENER_DISCLAIMER,
)
from live_engine.intelligence.market_summary import (
    ExecutiveMarketSummary,
    compile_market_intelligence_summary,
)


# =====================================================================
# 1. TREND INTELLIGENCE TESTS
# =====================================================================

def test_trend_analyzer_bullish_and_bearish():
    # 1. Bullish uptrend dataset (250 days, monotonic increase with small noise)
    dates = pd.date_range("2024-01-01", periods=250, freq="B")
    bull_prices = np.linspace(100.0, 200.0, 250) + np.sin(np.linspace(0, 10, 250))
    df_bull = pd.DataFrame({"Date": dates, "Close": bull_prices})

    insight_bull = analyze_asset_trend("SPY", df_bull)
    assert insight_bull.symbol == "SPY"
    assert insight_bull.classification == "Bullish"
    assert insight_bull.strength_score > 60.0
    assert insight_bull.latest_price is not None
    assert insight_bull.sma_20 is not None
    assert insight_bull.sma_50 is not None
    assert insight_bull.sma_200 is not None
    assert insight_bull.golden_cross is True
    assert len(insight_bull.contributing_factors) >= 2
    bull_dict = insight_bull.to_dict()
    assert bull_dict["Symbol"] == "SPY"
    assert bull_dict["Classification"] == "Bullish"

    # 2. Bearish downtrend dataset (250 days, monotonic decrease)
    bear_prices = np.linspace(200.0, 100.0, 250) - np.sin(np.linspace(0, 10, 250))
    df_bear = pd.DataFrame({"Date": dates, "Close": bear_prices})

    insight_bear = analyze_asset_trend("AGG", df_bear)
    assert insight_bear.symbol == "AGG"
    assert insight_bear.classification == "Bearish"
    assert insight_bear.strength_score < 40.0
    assert insight_bear.golden_cross is False
    assert len(insight_bear.contributing_factors) >= 2


def test_trend_analyzer_insufficient_data():
    # Less than 5 rows
    df_short = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=3),
        "Close": [100.0, 101.0, 102.0],
    })
    res = analyze_asset_trend("GLD", df_short)
    assert res.classification == "Neutral"
    assert res.strength_score == 50.0
    assert "Insufficient historical observations" in res.contributing_factors[0]

    # None input
    res_none = analyze_asset_trend("GLD", None)
    assert res_none.classification == "Neutral"


def test_analyze_universe_trends_and_breadth():
    dates = pd.date_range("2024-01-01", periods=250, freq="B")
    df_bull = pd.DataFrame({"Date": dates, "Close": np.linspace(100, 150, 250)})
    df_bear = pd.DataFrame({"Date": dates, "Close": np.linspace(150, 100, 250)})

    universe_data = {
        "SPY": df_bull,
        "XLK": df_bull,
        "AGG": df_bear,
    }
    insights = analyze_universe_trends(universe_data)
    # The universe always evaluates all 7 ETFs, defaulting unprovided ones to Neutral
    assert len(insights) == 7
    assert insights["SPY"].classification == "Bullish"
    assert insights["XLK"].classification == "Bullish"
    assert insights["AGG"].classification == "Bearish"
    assert insights["EFA"].classification == "Neutral"

    breadth = get_trend_breadth(insights)
    assert breadth["bullish_count"] == 2
    assert breadth["bearish_count"] == 1
    assert breadth["neutral_count"] == 4
    assert pytest.approx(breadth["bullish_ratio"], 0.01) == 2 / 7
    assert pytest.approx(breadth["bearish_ratio"], 0.01) == 1 / 7


# =====================================================================
# 2. NEWS INTELLIGENCE TESTS
# =====================================================================

def test_news_analyzer_sentiment_and_events():
    # 1. Positive text
    pos_headline = "Tech stocks rally strongly as Nvidia reports surging quarterly profit beat"
    score_pos, label_pos, terms_pos = score_text_sentiment(pos_headline)
    assert score_pos > 0.3
    assert label_pos == "Positive"
    assert any(t in terms_pos for t in ["rally", "surging", "profit", "beat"])

    # 2. Negative text
    neg_headline = "Recession fears escalate as bond yields slump and retail sales plunge heavily"
    score_neg, label_neg, terms_neg = score_text_sentiment(neg_headline)
    assert score_neg < -0.3
    assert label_neg == "Negative"
    assert any(t in terms_neg for t in ["recession", "slump", "plunge"])

    # 3. Macro Event Detection
    events_fomc = detect_macro_events("Powell signals Fed may cut interest rate at upcoming FOMC")
    assert "Monetary Policy / FOMC" in events_fomc

    events_cpi = detect_macro_events("Consumer price index CPI report shows persistent inflation pressures")
    assert "Inflation & Prices" in events_fomc or "Inflation & Prices" in events_cpi

    # 4. Related ETF detection
    etfs_tech = detect_related_etfs("Apple and Microsoft lead semiconductor chips and tech rally")
    assert "XLK" in etfs_tech

    etfs_energy = detect_related_etfs("Crude oil prices jump as OPEC announces production cuts")
    assert "XLE" in etfs_energy

    etfs_gold = detect_related_etfs("Investors rush to gold bullion as sovereign safe haven hedge")
    assert "GLD" in etfs_gold


def test_analyze_news_feed():
    mock_articles = [
        {
            "headline": "S&P 500 rallies to record high driven by tech sector optimism",
            "summary": "Equities surged with strong gains across major software companies.",
            "source": "MarketWatch",
            "datetime": 1700000000,
            "url": "https://example.com/article1",
        },
        {
            "headline": "Crude oil plunges sharply as OPEC talks collapse amidst recession warning",
            "summary": "Energy markets saw heavy losses and sudden drop in prices.",
            "source": "Reuters",
            "datetime": 1700000100,
            "url": "https://example.com/article2",
        },
    ]

    result = analyze_news_feed(mock_articles)
    assert result["total_articles"] == 2
    assert result["positive_count"] == 1
    assert result["negative_count"] == 1
    assert len(result["articles"]) == 2
    art1 = result["articles"][0]
    assert art1.sentiment_label == "Positive"
    assert "SPY" in art1.related_etfs or "XLK" in art1.related_etfs
    art2 = result["articles"][1]
    assert art2.sentiment_label == "Negative"
    assert "XLE" in art2.related_etfs


def test_analyze_news_feed_empty():
    res = analyze_news_feed([])
    assert res["total_articles"] == 0
    assert res["positive_ratio"] == 0.0
    assert res["negative_ratio"] == 0.0
    assert res["positive_count"] == 0
    assert res["articles"] == []


def test_get_trend_breadth_empty():
    res = get_trend_breadth({})
    assert res["total_analyzed"] == 0
    assert res["bullish_ratio"] == 0.0
    assert res["bearish_ratio"] == 0.0
    assert "market_breadth_label" in res


# =====================================================================
# 3. ECONOMIC INTELLIGENCE TESTS
# =====================================================================

def test_economic_analyzer_single_series():
    df_dgs10 = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=10, freq="B").strftime("%Y-%m-%d"),
        "Value": [4.10, 4.15, 4.20, 4.22, 4.25, 4.28, 4.30, 4.35, 4.38, 4.45],
    })
    analysis = analyze_single_series("DGS10", df_dgs10)
    assert analysis.series_id == "DGS10"
    assert analysis.latest_value == 4.45
    assert analysis.prev_value == 4.38
    assert analysis.change_1p == pytest.approx(0.07, 0.001)
    assert analysis.trend_direction == "Increasing"
    assert "10-Year" in analysis.observable_fact or "Treasury" in analysis.observable_fact


def test_yield_curve_slope_inversion():
    # 1. Inverted Curve: 10Y (3.80%) < 3M (4.80%) -> spread = -1.00% (-100 bps)
    yc_inverted = analyze_yield_curve_slope(dgs10_value=3.80, dtb3_value=4.80)
    assert yc_inverted.status == "Inverted Yield Curve"
    assert yc_inverted.spread_value == pytest.approx(-1.00, 0.01)
    assert yc_inverted.spread_bps == pytest.approx(-100.0, 0.1)
    assert "inverted" in yc_inverted.observable_fact.lower() or "inversion" in yc_inverted.observable_fact.lower()
    assert "recession" in yc_inverted.economic_interpretation.lower()

    # 2. Normal Upward Sloping: 10Y (4.50%) > 3M (3.50%) -> spread = +1.00% (+100 bps)
    yc_normal = analyze_yield_curve_slope(dgs10_value=4.50, dtb3_value=3.50)
    assert yc_normal.status == "Normal / Upward Sloping"
    assert yc_normal.spread_value == pytest.approx(1.00, 0.01)
    assert yc_normal.spread_bps == pytest.approx(100.0, 0.1)

    # 3. Missing data
    yc_missing = analyze_yield_curve_slope(dgs10_value=None, dtb3_value=4.50)
    assert yc_missing.status == "Unavailable"


def test_analyze_all_economic_data():
    dates = pd.date_range("2024-01-01", periods=5, freq="B").strftime("%Y-%m-%d")
    mock_econ = {
        "DGS10": pd.DataFrame({"Date": dates, "Value": [4.0, 4.1, 4.2, 4.3, 4.4]}),
        "DTB3": pd.DataFrame({"Date": dates, "Value": [4.8, 4.8, 4.8, 4.8, 4.8]}),
        "UNRATE": pd.DataFrame({"Date": ["2024-01-01", "2024-02-01"], "Value": [3.8, 3.9]}),
    }
    res = analyze_all_economic_data(mock_econ)
    assert "series_analyses" in res
    assert "yield_curve" in res
    assert res["yield_curve"].status == "Inverted Yield Curve"
    assert res["series_analyses"]["DGS10"].trend_direction == "Increasing"


# =====================================================================
# 4. ANOMALY DETECTION TESTS
# =====================================================================

def test_anomaly_detector_return_and_volume_shocks():
    # Create 60 days of tranquil data, then introduce an extreme return shock & volume surge
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    prices = 100.0 + np.cumsum(np.random.normal(0.05, 0.5, 60))
    volumes = np.random.normal(1_000_000, 50_000, 60)

    # Day 59: Normal
    # Inject massive return shock (-6%) and volume surge (4x) on last day
    prices[-1] = prices[-2] * 0.94
    volumes[-1] = 4_000_000

    df_shock = pd.DataFrame({"Date": dates, "Close": prices, "Volume": volumes})

    alerts = detect_asset_anomalies("SPY", df_shock, z_threshold_price=2.0, volume_multiplier_threshold=2.0)
    assert len(alerts) >= 2

    alert_types = [a.anomaly_type for a in alerts]
    assert "Price Return Shock" in alert_types
    assert "Abnormal Volume Surge" in alert_types

    shock_alert = next(a for a in alerts if a.anomaly_type == "Price Return Shock")
    assert shock_alert.severity in ["High", "Medium"]
    assert shock_alert.z_score is not None
    assert abs(shock_alert.z_score) >= 2.0

    vol_alert = next(a for a in alerts if a.anomaly_type == "Abnormal Volume Surge")
    assert vol_alert.severity in ["High", "Medium"]
    assert vol_alert.current_value >= 4_000_000


def test_scan_universe_anomalies():
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    df_calm = pd.DataFrame({
        "Date": dates,
        "Close": np.linspace(100, 102, 60),
        "Volume": [1_000_000] * 60,
    })
    df_shock = pd.DataFrame({
        "Date": dates,
        "Close": list(np.linspace(100, 102, 59)) + [85.0],
        "Volume": [1_000_000] * 59 + [5_000_000],
    })

    universe_data = {"AGG": df_calm, "XLE": df_shock}
    scan = scan_universe_anomalies(universe_data)
    assert scan["total_anomalies"] >= 1
    assert scan["has_active_anomalies"] is True
    assert scan["asset_counts"]["XLE"] >= 1
    assert any(a.asset == "XLE" for a in scan["alerts"])


# =====================================================================
# 5. OPPORTUNITY SCREENER TESTS
# =====================================================================

def test_opportunity_screener_rules_and_audit():
    # 1. Construct a winning TrendInsight
    winning_insight = TrendInsight(
        symbol="SPY",
        name="SPDR S&P 500 ETF Trust",
        classification="Bullish",
        strength_score=85.0,
        latest_price=510.0,
        daily_return_pct=0.8,
        momentum_5d_pct=2.4,
        momentum_20d_pct=4.1,
        sma_20=500.0,
        sma_50=485.0,
        sma_200=450.0,
        rsi_14=58.0,
        rolling_vol_20=14.5,
        golden_cross=True,
    )

    criteria = ScreeningCriteria(
        require_positive_5d_momentum=True,
        require_price_above_sma20=True,
        max_rolling_volatility_pct=20.0,
        min_rsi_14=40.0,
        max_rsi_14=70.0,
        allowed_classifications=["Bullish"],
    )

    result_win = screen_asset("SPY", winning_insight, criteria)
    assert result_win.overall_pass is True
    assert result_win.passed_rules_count == result_win.total_rules_count
    assert len(result_win.failure_reasons) == 0
    dict_win = result_win.to_dict()
    assert dict_win["Screening Status"] == "PASS"

    # 2. Construct a failing TrendInsight (RSI overbought, high vol)
    losing_insight = TrendInsight(
        symbol="XLK",
        name="Technology Select Sector SPDR Fund",
        classification="Bullish",
        strength_score=90.0,
        latest_price=220.0,
        daily_return_pct=-0.5,
        momentum_5d_pct=-1.2,  # Fails 5d momentum
        momentum_20d_pct=3.0,
        sma_20=210.0,
        sma_50=200.0,
        sma_200=180.0,
        rsi_14=78.0,           # Fails max RSI (78 > 70)
        rolling_vol_20=28.0,   # Fails max vol (28% > 20%)
        golden_cross=True,
    )

    result_lose = screen_asset("XLK", losing_insight, criteria)
    assert result_lose.overall_pass is False
    assert result_lose.passed_rules_count < result_lose.total_rules_count
    assert len(result_lose.failure_reasons) >= 2
    assert any("momentum" in r.lower() for r in result_lose.failure_reasons)
    assert any("rsi" in r.lower() for r in result_lose.failure_reasons)


def test_run_universe_screening():
    insight1 = TrendInsight("SPY", "SPDR S&P 500", "Bullish", 80.0, 500.0, 0.5, 1.5, 3.0, 490.0, 480.0, 460.0, 55.0, 15.0, True)
    insight2 = TrendInsight("AGG", "Core Bond", "Bearish", 30.0, 95.0, -0.2, -0.8, -1.5, 96.0, 97.0, 99.0, 42.0, 8.0, False)

    universe_insights = {"SPY": insight1, "AGG": insight2}
    criteria = ScreeningCriteria(require_positive_5d_momentum=True, allowed_classifications=["Bullish"])

    screening_res = run_universe_screening(universe_insights, criteria)
    assert screening_res["total_screened"] == 2
    assert screening_res["passed_count"] == 1
    assert screening_res["failed_count"] == 1
    passed_symbols = [r.symbol for r in screening_res["passed_assets"]]
    assert "SPY" in passed_symbols
    assert "AGG" not in passed_symbols
    assert screening_res["disclaimer"] == SCREENER_DISCLAIMER


# =====================================================================
# 6. EXECUTIVE MARKET SUMMARY TESTS
# =====================================================================

def test_market_summary_compilation():
    insight_bull = TrendInsight("SPY", "S&P 500", "Bullish", 80.0, 500.0, 0.5, 1.5, 3.0, 490.0, 480.0, 460.0, 55.0, 15.0, True)
    insight_bull2 = TrendInsight("XLK", "Tech", "Bullish", 85.0, 220.0, 0.8, 2.0, 4.0, 210.0, 205.0, 195.0, 60.0, 18.0, True)
    insight_bear = TrendInsight("AGG", "Bond", "Bearish", 30.0, 95.0, -0.2, -0.8, -1.5, 96.0, 97.0, 99.0, 42.0, 8.0, False)

    mock_trends = {"SPY": insight_bull, "XLK": insight_bull2, "AGG": insight_bear}
    mock_news = {
        "overall_stance": "Positive / Constructive",
        "average_sentiment_score": 0.45,
        "positive_count": 8,
        "negative_count": 2,
    }
    mock_econ = {
        "yield_curve": YieldCurveAnalysis(4.5, 3.5, 1.0, 100.0, "Normal / Upward Sloping", "10Y > 3M", "Expansionary"),
    }
    mock_anomalies = {
        "total_anomalies_detected": 1,
        "high_severity_count": 0,
    }
    mock_screener = {
        "passed_count": 2,
    }

    summary = compile_market_intelligence_summary(
        trend_insights=mock_trends,
        news_analysis=mock_news,
        economic_analysis=mock_econ,
        anomaly_analysis=mock_anomalies,
        screening_analysis=mock_screener,
        data_freshness=DataFreshness.FRESH,
    )

    assert summary.macro_regime == "🟢 Risk-On / Favorable"
    assert summary.bullish_assets_count == 2
    assert summary.bearish_assets_count == 1
    assert summary.screening_opportunities_count == 2
    assert len(summary.key_takeaways) >= 3

    sum_dict = summary.to_dict()
    assert sum_dict["Macro Regime"] == "🟢 Risk-On / Favorable"
    assert "2 / 1 / 0" in sum_dict["ETF Breadth (Bull/Bear/Neut)"]
