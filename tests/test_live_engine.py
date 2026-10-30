"""
Automated Unit Tests for Live Market Data Engine
QuantEdge 1.0 - Serendib University Endowment Risk Engine

Validates all external API collector behaviors using mocked HTTP responses:
- Valid API keys
- Invalid API keys (401/403/400)
- Missing API keys (Not Configured)
- Rate-limit responses (429 & API info notes)
- Network failure / timeouts
- Empty responses
- Failure isolation (one provider fails while others succeed)
- Secret sanitization
"""

from unittest.mock import MagicMock, patch
import pytest
import requests

from live_engine.core.data_validator import ProviderStatus, LiveResult, sanitize_secret
from live_engine.collectors.finnhub_collector import FinnhubCollector
from live_engine.collectors.alphavantage_collector import AlphaVantageCollector
from live_engine.collectors.fred_collector import FREDCollector
from live_engine.collectors.sec_collector import SECCollector
from live_engine.core.api_manager import LiveAPIManager


# ===========================================================================
# 1. Finnhub Collector Tests
# ===========================================================================

def test_finnhub_missing_key():
    """Verify Finnhub returns NOT_CONFIGURED when key is empty without HTTP call."""
    res = FinnhubCollector.fetch_spy_quote(api_key=None)
    assert res.status == ProviderStatus.NOT_CONFIGURED
    assert res.provider == "Finnhub"

    res_empty = FinnhubCollector.fetch_spy_quote(api_key="   ")
    assert res_empty.status == ProviderStatus.NOT_CONFIGURED


@patch("requests.get")
def test_finnhub_valid_key(mock_get):
    """Verify Finnhub successfully parses real-time quote metrics."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "c": 582.45,
        "d": 3.12,
        "dp": 0.5385,
        "h": 584.10,
        "l": 580.20,
        "o": 581.00,
        "pc": 579.33,
        "t": 1729886400,
    }
    mock_get.return_value = mock_resp

    res = FinnhubCollector.fetch_spy_quote(api_key="mock_finnhub_token")
    assert res.status == ProviderStatus.CONNECTED
    assert res.http_status == 200
    assert res.record_count == 1
    assert res.metrics["current_price"] == 582.45
    assert res.metrics["previous_close"] == 579.33
    assert res.dataframe is not None
    assert len(res.dataframe) > 0


@patch("requests.get")
def test_finnhub_invalid_key(mock_get):
    """Verify Finnhub handles 401 authentication failure."""
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Invalid API key"
    mock_get.return_value = mock_resp

    res = FinnhubCollector.fetch_spy_quote(api_key="invalid_token")
    assert res.status == ProviderStatus.FAILED
    assert res.http_status == 401
    assert "Invalid Finnhub API Key" in res.error_message


@patch("requests.get")
def test_finnhub_rate_limited(mock_get):
    """Verify Finnhub handles 429 rate limit."""
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_get.return_value = mock_resp

    res = FinnhubCollector.fetch_spy_quote(api_key="mock_token")
    assert res.status == ProviderStatus.RATE_LIMITED
    assert res.http_status == 429


@patch("requests.get")
def test_finnhub_network_failure(mock_get):
    """Verify Finnhub handles network timeout gracefully without crashing."""
    mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

    res = FinnhubCollector.fetch_spy_quote(api_key="mock_token", timeout=5)
    assert res.status == ProviderStatus.FAILED
    assert "timed out" in res.error_message


@patch("requests.get")
def test_finnhub_empty_response(mock_get):
    """Verify Finnhub handles blank/zero quote."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"c": 0, "d": 0, "dp": 0, "h": 0, "l": 0, "o": 0, "pc": 0, "t": 0}
    mock_get.return_value = mock_resp

    res = FinnhubCollector.fetch_spy_quote(api_key="mock_token")
    assert res.status == ProviderStatus.FAILED
    assert "empty quote" in res.error_message.lower()


# ===========================================================================
# 2. Alpha Vantage Collector Tests
# ===========================================================================

def test_alphavantage_missing_key():
    """Verify Alpha Vantage returns NOT_CONFIGURED when key is missing."""
    res = AlphaVantageCollector.fetch_spy_daily(api_key="")
    assert res.status == ProviderStatus.NOT_CONFIGURED
    assert res.provider == "Alpha Vantage"


@patch("requests.get")
def test_alphavantage_valid_key(mock_get):
    """Verify Alpha Vantage successfully parses daily observations."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "Meta Data": {
            "1. Information": "Daily Prices and Volumes",
            "2. Symbol": "SPY",
            "3. Last Refreshed": "2024-10-25",
        },
        "Time Series (Daily)": {
            "2024-10-25": {
                "1. open": "583.15",
                "2. high": "586.20",
                "3. low": "581.30",
                "4. close": "584.50",
                "5. volume": "45120300",
            },
            "2024-10-24": {
                "1. open": "580.00",
                "2. high": "583.00",
                "3. low": "578.50",
                "4. close": "582.00",
                "5. volume": "38200100",
            },
        },
    }
    mock_get.return_value = mock_resp

    res = AlphaVantageCollector.fetch_spy_daily(api_key="mock_av_key")
    assert res.status == ProviderStatus.CONNECTED
    assert res.record_count == 2
    assert res.dataframe is not None
    assert "Close" in res.dataframe.columns
    assert res.dataframe.iloc[0]["Close"] == 584.50


@patch("requests.get")
def test_alphavantage_rate_limit_information(mock_get):
    """Verify Alpha Vantage detects 'Information' rate-limit message."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "Information": "Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute and 100 calls per day."
    }
    mock_get.return_value = mock_resp

    res = AlphaVantageCollector.fetch_spy_daily(api_key="mock_av_key")
    assert res.status == ProviderStatus.RATE_LIMITED
    assert "Rate limit notice" in res.error_message


@patch("requests.get")
def test_alphavantage_error_message(mock_get):
    """Verify Alpha Vantage detects 'Error Message' invalid call."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "Error Message": "Invalid API call. Please check documentation."
    }
    mock_get.return_value = mock_resp

    res = AlphaVantageCollector.fetch_spy_daily(api_key="mock_av_key")
    assert res.status == ProviderStatus.FAILED
    assert "Alpha Vantage error" in res.error_message


# ===========================================================================
# 3. FRED Collector Tests
# ===========================================================================

def test_fred_missing_key():
    """Verify FRED returns NOT_CONFIGURED when key is missing."""
    res = FREDCollector.fetch_dtb3_series(api_key=None)
    assert res.status == ProviderStatus.NOT_CONFIGURED
    assert res.provider == "Federal Reserve (FRED)"


@patch("requests.get")
def test_fred_valid_key(mock_get):
    """Verify FRED parses DTB3 interest-rate series."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "realtime_start": "2024-10-25",
        "observations": [
            {"date": "2024-10-24", "value": "4.58"},
            {"date": "2024-10-23", "value": "4.60"},
            {"date": "2024-10-22", "value": "."},  # Holiday marker
        ],
    }
    mock_get.return_value = mock_resp

    res = FREDCollector.fetch_dtb3_series(api_key="mock_fred_key_32chars_hex")
    assert res.status == ProviderStatus.CONNECTED
    assert res.record_count == 3
    assert res.metrics["latest_yield_pct"] == 4.58
    assert res.dataframe is not None


@patch("requests.get")
def test_fred_invalid_key(mock_get):
    """Verify FRED handles 400 invalid API key response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.json.return_value = {
        "error_code": 400,
        "error_message": "The api_key you provided is not valid.",
    }
    mock_get.return_value = mock_resp

    res = FREDCollector.fetch_dtb3_series(api_key="bad_key")
    assert res.status == ProviderStatus.FAILED
    assert res.http_status == 400
    assert "not valid" in res.error_message


# ===========================================================================
# 4. SEC EDGAR Collector Tests
# ===========================================================================

def test_sec_invalid_user_agent():
    """Verify SEC rejects User-Agents missing contact email."""
    res = SECCollector.fetch_recent_filings(user_agent="GenericClientNoEmail")
    assert res.status == ProviderStatus.FAILED
    assert "Invalid SEC User-Agent" in res.error_message


@patch("requests.get")
def test_sec_valid_submissions(mock_get):
    """Verify SEC collector parses recent filings for SPDR ETF Trust."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "cik": "0000884394",
        "name": "SPDR S&P 500 ETF TRUST",
        "tickers": ["SPY"],
        "filings": {
            "recent": {
                "form": ["497", "N-PORT", "24F-2NT"],
                "filingDate": ["2024-10-21", "2024-09-30", "2024-09-15"],
                "reportDate": ["2024-10-21", "2024-07-31", "2024-06-30"],
                "primaryDocDescription": ["Prospectus Supplement", "Monthly Portfolio", "Annual Filing"],
                "accessionNumber": ["0001193125-24-001", "0001193125-24-002", "0001193125-24-003"],
            }
        },
    }
    mock_get.return_value = mock_resp

    res = SECCollector.fetch_recent_filings(
        user_agent="SerendibEndowment/1.0 (audit@serendib.ac.lk)",
        limit=5,
    )
    assert res.status == ProviderStatus.CONNECTED
    assert res.record_count == 3
    assert res.metrics["entity_name"] == "SPDR S&P 500 ETF TRUST"
    assert res.dataframe is not None
    assert len(res.dataframe) == 3


@patch("requests.get")
def test_sec_forbidden_403(mock_get):
    """Verify SEC handles HTTP 403 Forbidden."""
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.text = "Forbidden: User-Agent policy violated."
    mock_get.return_value = mock_resp

    res = SECCollector.fetch_recent_filings(
        user_agent="BlockedAgent/1.0 (test@domain.com)",
    )
    assert res.status == ProviderStatus.FAILED
    assert res.http_status == 403
    assert "403 Forbidden" in res.error_message


# ===========================================================================
# 5. LiveAPIManager Multi-Provider Failure Isolation Test
# ===========================================================================

@patch("live_engine.collectors.finnhub_collector.FinnhubCollector.fetch_spy_quote")
@patch("live_engine.collectors.alphavantage_collector.AlphaVantageCollector.fetch_spy_daily")
@patch("live_engine.collectors.fred_collector.FREDCollector.fetch_dtb3_series")
@patch("live_engine.collectors.sec_collector.SECCollector.fetch_recent_filings")
def test_api_manager_independent_execution(mock_sec, mock_fred, mock_av, mock_fh):
    """
    Verify that one provider failing catastrophically does not crash or block the others.
    Scenario:
    - Finnhub crashes with an uncaught exception
    - Alpha Vantage succeeds
    - FRED is rate-limited
    - SEC succeeds
    """
    mock_fh.side_effect = RuntimeError("Catastrophic Finnhub network socket error")
    mock_av.return_value = LiveResult(provider="Alpha Vantage", status=ProviderStatus.CONNECTED, record_count=10)
    mock_fred.return_value = LiveResult(provider="FRED", status=ProviderStatus.RATE_LIMITED, record_count=0)
    mock_sec.return_value = LiveResult(provider="SEC EDGAR", status=ProviderStatus.CONNECTED, record_count=5)

    results = LiveAPIManager.run_all_queries(
        finnhub_key="key1",
        alphavantage_key="key2",
        fred_key="key3",
        sec_user_agent="Test/1.0 (test@domain.com)",
    )

    assert "finnhub" in results
    assert "alphavantage" in results
    assert "fred" in results
    assert "sec" in results

    # Finnhub failed gracefully
    assert results["finnhub"].status == ProviderStatus.FAILED
    assert "Catastrophic Finnhub" in results["finnhub"].error_message

    # Other providers executed unimpeded
    assert results["alphavantage"].status == ProviderStatus.CONNECTED
    assert results["fred"].status == ProviderStatus.RATE_LIMITED
    assert results["sec"].status == ProviderStatus.CONNECTED


# ===========================================================================
# 6. Secret Credential Sanitization Test
# ===========================================================================

def test_secret_sanitization():
    """Verify API keys and sensitive tokens are completely scrubbed from error output."""
    raw_secret = "c12345_super_secret_finnhub_token_999"
    error_str = f"Connection failed to https://finnhub.io/api/v1/quote?symbol=SPY&token={raw_secret}"

    sanitized = sanitize_secret(error_str, secrets=[raw_secret])
    assert raw_secret not in sanitized
    assert "[REDACTED]" in sanitized


# ===========================================================================
# 7. Additional Resilience Tests (Rate limits & Timeouts)
# ===========================================================================

@patch("requests.get")
def test_fred_rate_limited(mock_get):
    """Verify FRED handles 429 rate limit."""
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_get.return_value = mock_resp

    res = FREDCollector.fetch_dtb3_series(api_key="mock_key")
    assert res.status == ProviderStatus.RATE_LIMITED
    assert res.http_status == 429


@patch("requests.get")
def test_fred_network_timeout(mock_get):
    """Verify FRED handles network timeout."""
    mock_get.side_effect = requests.exceptions.Timeout("FRED connection timeout")

    res = FREDCollector.fetch_dtb3_series(api_key="mock_key")
    assert res.status == ProviderStatus.FAILED
    assert "timed out" in res.error_message


@patch("requests.get")
def test_sec_rate_limited(mock_get):
    """Verify SEC handles 429 rate limit."""
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_get.return_value = mock_resp

    res = SECCollector.fetch_recent_filings(user_agent="Test/1.0 (test@serendib.ac.lk)")
    assert res.status == ProviderStatus.RATE_LIMITED
    assert res.http_status == 429


@patch("requests.get")
def test_sec_network_timeout(mock_get):
    """Verify SEC handles network timeout."""
    mock_get.side_effect = requests.exceptions.Timeout("SEC connection timeout")

    res = SECCollector.fetch_recent_filings(user_agent="Test/1.0 (test@serendib.ac.lk)")
    assert res.status == ProviderStatus.FAILED
    assert "timed out" in res.error_message


@patch("requests.get")
def test_alphavantage_network_timeout(mock_get):
    """Verify Alpha Vantage handles network timeout."""
    mock_get.side_effect = requests.exceptions.Timeout("Alpha Vantage timeout")

    res = AlphaVantageCollector.fetch_spy_daily(api_key="mock_key")
    assert res.status == ProviderStatus.FAILED
    assert "timed out" in res.error_message
