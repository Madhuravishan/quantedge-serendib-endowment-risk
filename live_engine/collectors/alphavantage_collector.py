"""
Alpha Vantage Market Data Collector
QuantEdge 1.0 - Serendib University Endowment Risk Engine (Phase 2)
"""

import logging
from typing import Optional
import pandas as pd
import requests

from live_engine.core.data_validator import ProviderStatus, LiveResult, sanitize_secret

logger = logging.getLogger(__name__)


class AlphaVantageCollector:
    """Retrieves daily time series data for ETFs via Alpha Vantage REST API."""

    BASE_URL = "https://www.alphavantage.co/query"
    PROVIDER_NAME = "Alpha Vantage"
    DATA_SOURCE = "Alpha Vantage TIME_SERIES_DAILY"

    @classmethod
    def fetch_etf_daily(
        cls,
        symbol: str,
        api_key: Optional[str],
        outputsize: str = "compact",
        timeout: int = 10,
    ) -> LiveResult:
        """
        Retrieve daily time series observations for any specified ETF symbol.
        
        Parameters
        ----------
        symbol : str
            ETF ticker symbol (e.g. 'SPY', 'AGG', 'GLD', 'EFA', 'EEM', 'XLE', 'XLK').
        api_key : str, optional
            Alpha Vantage personal API key.
        outputsize : str
            'compact' (latest 100 observations) or 'full' (up to 20+ years).
        timeout : int
            HTTP connection & read timeout in seconds.
            
        Returns
        -------
        LiveResult
            Standardized result envelope with observations DataFrame and metadata.
        """
        sym = (symbol or "SPY").strip().upper()
        data_source = f"Alpha Vantage TIME_SERIES_DAILY ({sym})"

        if not api_key or not api_key.strip():
            return LiveResult(
                provider=cls.PROVIDER_NAME,
                status=ProviderStatus.NOT_CONFIGURED,
                data_source=data_source,
                error_message="Alpha Vantage API key is not configured.",
            )

        clean_key = api_key.strip()
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": sym,
            "apikey": clean_key,
            "outputsize": outputsize,
        }
        headers = {"User-Agent": "SerendibEndowmentRiskEngine/1.0"}

        try:
            response = requests.get(
                cls.BASE_URL,
                params=params,
                headers=headers,
                timeout=timeout,
            )
            http_status = response.status_code

            if http_status == 429:
                return LiveResult(
                    provider=cls.PROVIDER_NAME,
                    status=ProviderStatus.RATE_LIMITED,
                    http_status=http_status,
                    data_source=data_source,
                    error_message="Alpha Vantage rate limit reached (HTTP 429).",
                )

            if http_status != 200:
                err_msg = sanitize_secret(response.text[:200], [clean_key])
                return LiveResult(
                    provider=cls.PROVIDER_NAME,
                    status=ProviderStatus.FAILED,
                    http_status=http_status,
                    data_source=data_source,
                    error_message=f"HTTP {http_status}: {err_msg}",
                )

            data = response.json()

            if not isinstance(data, dict):
                return LiveResult(
                    provider=cls.PROVIDER_NAME,
                    status=ProviderStatus.FAILED,
                    http_status=http_status,
                    data_source=data_source,
                    error_message="Unexpected response format from Alpha Vantage.",
                )

            # Check for Alpha Vantage rate-limiting info messages
            if "Information" in data:
                info_msg = data["Information"]
                return LiveResult(
                    provider=cls.PROVIDER_NAME,
                    status=ProviderStatus.RATE_LIMITED,
                    http_status=http_status,
                    data_source=data_source,
                    raw_data=data,
                    error_message=sanitize_secret(f"Rate limit notice: {info_msg}", [clean_key]),
                )

            if "Note" in data:
                note_msg = data["Note"]
                return LiveResult(
                    provider=cls.PROVIDER_NAME,
                    status=ProviderStatus.RATE_LIMITED,
                    http_status=http_status,
                    data_source=data_source,
                    raw_data=data,
                    error_message=sanitize_secret(f"Call frequency limit: {note_msg}", [clean_key]),
                )

            # Check for error message
            if "Error Message" in data:
                err_msg = data["Error Message"]
                return LiveResult(
                    provider=cls.PROVIDER_NAME,
                    status=ProviderStatus.FAILED,
                    http_status=http_status,
                    data_source=data_source,
                    raw_data=data,
                    error_message=sanitize_secret(f"Alpha Vantage error: {err_msg}", [clean_key]),
                )

            time_series = data.get("Time Series (Daily)")
            if not time_series or not isinstance(time_series, dict):
                return LiveResult(
                    provider=cls.PROVIDER_NAME,
                    status=ProviderStatus.FAILED,
                    http_status=http_status,
                    data_source=data_source,
                    raw_data=data,
                    error_message=f"No 'Time Series (Daily)' section found in Alpha Vantage response for {sym}.",
                )

            meta_data = data.get("Meta Data", {})
            market_time = meta_data.get("3. Last Refreshed", "N/A")

            records = []
            for date_str, values in time_series.items():
                try:
                    records.append({
                        "Date": date_str,
                        "Open": float(values.get("1. open", 0.0)),
                        "High": float(values.get("2. high", 0.0)),
                        "Low": float(values.get("3. low", 0.0)),
                        "Close": float(values.get("4. close", 0.0)),
                        "Volume": int(float(values.get("5. volume", 0))),
                    })
                except Exception:
                    continue

            if not records:
                return LiveResult(
                    provider=cls.PROVIDER_NAME,
                    status=ProviderStatus.FAILED,
                    http_status=http_status,
                    data_source=data_source,
                    error_message=f"Could not parse daily observation rows for {sym}.",
                )

            df = pd.DataFrame(records)
            df = df.sort_values(by="Date", ascending=False).reset_index(drop=True)

            latest_close = df.iloc[0]["Close"] if len(df) > 0 else None
            metrics = {
                "symbol": sym,
                "latest_close": latest_close,
                "observation_count": len(df),
                "market_timestamp": market_time,
                "latest_date": df.iloc[0]["Date"] if len(df) > 0 else "N/A",
            }

            return LiveResult(
                provider=cls.PROVIDER_NAME,
                status=ProviderStatus.CONNECTED,
                http_status=http_status,
                market_timestamp=market_time,
                record_count=len(df),
                data_source=data_source,
                raw_data={"Meta Data": meta_data, "Sample Rows": len(df)},
                dataframe=df,
                metrics=metrics,
            )

        except requests.exceptions.Timeout:
            return LiveResult(
                provider=cls.PROVIDER_NAME,
                status=ProviderStatus.FAILED,
                data_source=data_source,
                error_message=f"Request timed out after {timeout} seconds.",
            )
        except requests.exceptions.ConnectionError:
            return LiveResult(
                provider=cls.PROVIDER_NAME,
                status=ProviderStatus.FAILED,
                data_source=data_source,
                error_message="Network connection failed while reaching Alpha Vantage.",
            )
        except Exception as e:
            logger.warning(f"Unexpected Alpha Vantage collector exception: {e}")
            return LiveResult(
                provider=cls.PROVIDER_NAME,
                status=ProviderStatus.FAILED,
                data_source=data_source,
                error_message=sanitize_secret(f"Unexpected error: {str(e)}", [clean_key]),
            )

    @classmethod
    def fetch_spy_daily(cls, api_key: Optional[str], timeout: int = 10) -> LiveResult:
        """Backward-compatible alias for SPY daily series."""
        return cls.fetch_etf_daily(symbol="SPY", api_key=api_key, outputsize="compact", timeout=timeout)
