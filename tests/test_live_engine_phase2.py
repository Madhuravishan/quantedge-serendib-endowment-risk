"""
Unit Test Suite for Phase 2: Expanded Live Market Data Engine
QuantEdge 1.0 - Serendib University Endowment Risk Engine

Validates:
1. All 7 ETF selections (SPY, EFA, EEM, AGG, GLD, XLE, XLK) quotes
2. All 7 ETF historical daily observations
3. Technical indicators calculation (SMA 20/50/200, RSI 14, Volatility, Trend Classification)
4. Technical indicators handling of insufficient data
5. Economic indicator retrieval across all 5 FRED series (DTB3, DGS10, FEDFUNDS, CPIAUCSL, UNRATE)
6. Financial news retrieval (General & Ticker-specific)
7. SEC filing retrieval across multiple entities with direct document links
8. In-memory cache manager TTL and freshness states
"""

from datetime import datetime
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from live_engine.core.data_validator import ProviderStatus, DataFreshness, LiveResult
from live_engine.core.indicators import compute_technical_indicators, compute_rsi_14
from live_engine.core.cache_manager import LiveCacheManager
from live_engine.collectors.finnhub_collector import FinnhubCollector
from live_engine.collectors.alphavantage_collector import AlphaVantageCollector
from live_engine.collectors.fred_collector import FREDCollector, FRED_SERIES_CATALOG
from live_engine.collectors.sec_collector import SECCollector, SEC_ENTITY_CATALOG
from live_engine.core.api_manager import LiveAPIManager


ETF_UNIVERSE = ["SPY", "EFA", "EEM", "AGG", "GLD", "XLE", "XLK"]


# ===========================================================================
# 1. Test All 7 ETF Selections Quote Retrieval
# ===========================================================================

@patch("requests.get")
def test_all_seven_etf_quotes(mock_get):
    """Verify that quote retrieval works for all 7 ETFs in the universe."""
    for symbol in ETF_UNIVERSE:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "c": 125.50,
            "d": 1.20,
            "dp": 0.965,
            "h": 126.00,
            "l": 124.80,
            "o": 125.00,
            "pc": 124.30,
            "t": 1729886400,
        }
        mock_get.return_value = mock_resp

        res = FinnhubCollector.fetch_etf_quote(symbol=symbol, api_key="mock_key")
        assert res.status == ProviderStatus.CONNECTED
        assert res.metrics["symbol"] == symbol
        assert res.metrics["current_price"] == 125.50
        assert res.dataframe is not None


# ===========================================================================
# 2. Test All 7 ETF Historical Daily Observations
# ===========================================================================

@patch("requests.get")
def test_all_seven_etf_historical_daily(mock_get):
    """Verify daily time series retrieval works across all 7 ETFs."""
    for symbol in ETF_UNIVERSE:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "Meta Data": {
                "2. Symbol": symbol,
                "3. Last Refreshed": "2024-10-25",
            },
            "Time Series (Daily)": {
                "2024-10-25": {"1. open": "100.0", "2. high": "102.0", "3. low": "99.0", "4. close": "101.5", "5. volume": "1000000"},
                "2024-10-24": {"1. open": "99.0", "2. high": "101.0", "3. low": "98.5", "4. close": "100.0", "5. volume": "950000"},
            },
        }
        mock_get.return_value = mock_resp

        res = AlphaVantageCollector.fetch_etf_daily(symbol=symbol, api_key="mock_key")
        assert res.status == ProviderStatus.CONNECTED
        assert res.record_count == 2
        assert res.metrics["symbol"] == symbol
        assert res.dataframe is not None
        assert "Close" in res.dataframe.columns


# ===========================================================================
# 3. Test Technical Indicators Calculation
# ===========================================================================

def test_technical_indicators_calculation():
    """Verify computation of SMA 20, 50, 200, RSI 14, Volatility, and Trend Classification."""
    # Create 220 synthetic price points with an upward trend
    dates = pd.date_range("2024-01-01", periods=220, freq="B")
    base_price = 100.0
    prices = [base_price + i * 0.25 + np.sin(i / 5.0) * 2.0 for i in range(220)]
    df_raw = pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Close": prices})

    df_enriched, metrics = compute_technical_indicators(df_raw)

    assert not df_enriched.empty
    assert "Daily_Return" in df_enriched.columns
    assert "SMA_20" in df_enriched.columns
    assert "SMA_50" in df_enriched.columns
    assert "SMA_200" in df_enriched.columns
    assert "RSI_14" in df_enriched.columns
    assert "Rolling_Vol_20" in df_enriched.columns

    # Check metrics
    assert metrics["observation_count"] == 220
    assert metrics["sma_20_numeric"] is not None
    assert metrics["sma_50_numeric"] is not None
    assert metrics["sma_200_numeric"] is not None
    assert metrics["rsi_14_numeric"] is not None
    assert 0 <= metrics["rsi_14_numeric"] <= 100
    assert metrics["rolling_vol_20_numeric"] is not None
    assert metrics["trend_classification"] in ("Bullish", "Bearish", "Neutral")
    assert "disclaimer" in metrics


def test_technical_indicators_insufficient_data():
    """Verify graceful handling when fewer than 20 observations exist."""
    df_short = pd.DataFrame({
        "Date": ["2024-10-01", "2024-10-02", "2024-10-03"],
        "Close": [100.0, 101.0, 102.0],
    })
    _, metrics = compute_technical_indicators(df_short)

    assert metrics["sma_20"] == "Insufficient data"
    assert metrics["sma_50"] == "Insufficient data"
    assert metrics["sma_200"] == "Insufficient data"
    assert metrics["rolling_vol_20"] == "Insufficient data"


# ===========================================================================
# 4. Test Economic Indicator Retrieval (All 5 FRED Series)
# ===========================================================================

@patch("requests.get")
def test_all_five_economic_indicators(mock_get):
    """Verify retrieval across all 5 required FRED macroeconomic series."""
    series_list = ["DTB3", "DGS10", "FEDFUNDS", "CPIAUCSL", "UNRATE"]

    for sid in series_list:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "realtime_start": "2024-10-25",
            "observations": [
                {"date": "2024-10-01", "value": "4.25"},
                {"date": "2024-09-01", "value": "4.30"},
            ],
        }
        mock_get.return_value = mock_resp

        res = FREDCollector.fetch_economic_series(series_id=sid, api_key="mock_fred_key")
        assert res.status == ProviderStatus.CONNECTED
        assert res.metrics["series_id"] == sid
        assert res.metrics["latest_value"] == 4.25
        assert res.record_count == 2
        assert sid in FRED_SERIES_CATALOG


# ===========================================================================
# 5. Test Financial News Retrieval
# ===========================================================================

@patch("requests.get")
def test_financial_news_general(mock_get):
    """Verify general market news parsing."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {
            "headline": "Fed Signals Measured Policy Easing Path",
            "summary": "Federal Reserve officials noted balanced inflation risks...",
            "source": "Reuters",
            "datetime": 1729886400,
            "related": "General",
            "url": "https://example.com/article1",
        },
        {
            "headline": "S&P 500 Closes Higher Led by Tech and Energy",
            "summary": "Broad market index advances...",
            "source": "Bloomberg",
            "datetime": 1729886100,
            "related": "SPY",
            "url": "https://example.com/article2",
        },
    ]
    mock_get.return_value = mock_resp

    res = FinnhubCollector.fetch_financial_news(category="general", api_key="mock_key")
    assert res.status == ProviderStatus.CONNECTED
    assert res.record_count == 2
    articles = res.metrics["articles"]
    assert len(articles) == 2
    assert articles[0]["source"] == "Reuters"
    assert "https://example.com/article1" == articles[0]["url"]


@patch("requests.get")
def test_financial_news_symbol_specific(mock_get):
    """Verify symbol-filtered company news parsing."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {
            "headline": "SPY ETF Assets Hit New Milestones",
            "summary": "Investor inflows into broad market ETFs continue...",
            "source": "CNBC",
            "datetime": 1729886400,
            "related": "SPY",
            "url": "https://example.com/spy-article",
        }
    ]
    mock_get.return_value = mock_resp

    res = FinnhubCollector.fetch_financial_news(symbol="SPY", api_key="mock_key")
    assert res.status == ProviderStatus.CONNECTED
    assert res.record_count == 1
    assert "SPY" in res.data_source


# ===========================================================================
# 6. Test SEC Filing Retrieval Across Multiple Entities
# ===========================================================================

@patch("requests.get")
def test_sec_filings_multiple_entities(mock_get):
    """Verify SEC filings retrieval and document URL generation for multiple entities."""
    test_entities = ["0000884394", "0000320193", "0000789019"]

    for cik in test_entities:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "cik": cik,
            "name": SEC_ENTITY_CATALOG[cik]["name"],
            "tickers": [SEC_ENTITY_CATALOG[cik]["ticker"]],
            "filings": {
                "recent": {
                    "form": ["10-K", "10-Q"],
                    "filingDate": ["2024-10-15", "2024-07-20"],
                    "reportDate": ["2024-09-30", "2024-06-30"],
                    "primaryDocDescription": ["Annual Report", "Quarterly Report"],
                    "accessionNumber": ["0000320193-24-000106", "0000320193-24-000085"],
                    "primaryDocument": ["aapl-20240930.htm", "aapl-20240630.htm"],
                }
            },
        }
        mock_get.return_value = mock_resp

        res = SECCollector.fetch_recent_filings(
            cik=cik,
            user_agent="SerendibAudit/1.0 (audit@serendib.ac.lk)",
        )
        assert res.status == ProviderStatus.CONNECTED
        assert res.record_count == 2
        assert res.dataframe is not None
        assert "Document Link" in res.dataframe.columns
        assert res.dataframe["Document Link"].iloc[0].startswith("https://www.sec.gov/Archives/edgar/data/")


# ===========================================================================
# 7. Test In-Memory Cache Manager & Freshness
# ===========================================================================

def test_cache_manager_ttl_and_freshness():
    """Verify that cached results are returned with DataFreshness.CACHED and respect TTL."""
    cache = LiveCacheManager()
    cache.clear()

    dummy_result = LiveResult(
        provider="TestProvider",
        status=ProviderStatus.CONNECTED,
        record_count=5,
        freshness=DataFreshness.FRESH,
    )

    cache.set("test_key", dummy_result)

    # Retrieval should be CACHED
    cached = cache.get("test_key", max_age_seconds=10)
    assert cached is not None
    assert cached.freshness == DataFreshness.CACHED

    # Expired check
    expired = cache.get("test_key", max_age_seconds=-1)
    assert expired is None

    cache.clear()
    assert cache.get("test_key") is None
