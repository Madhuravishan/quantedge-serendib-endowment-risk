"""
Finnhub Market Data Collector
QuantEdge 1.0 - Serendib University Endowment Risk Engine (Phase 2)
"""

from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional
import pandas as pd
import requests

from live_engine.core.data_validator import ProviderStatus, LiveResult, sanitize_secret

logger = logging.getLogger(__name__)


class FinnhubCollector:
    """Retrieves real-time market quote data and financial news via Finnhub REST API."""

    BASE_URL = "https://finnhub.io/api/v1"
    PROVIDER_NAME = "Finnhub"
    DATA_SOURCE = "Finnhub /v1/quote"

    @classmethod
    def fetch_etf_quote(cls, symbol: str, api_key: Optional[str], timeout: int = 8) -> LiveResult:
        """
        Retrieve real-time market quote for any specified ETF symbol from Finnhub.
        
        Parameters
        ----------
        symbol : str
            ETF ticker symbol (e.g. 'SPY', 'AGG', 'GLD', 'EFA', 'EEM', 'XLE', 'XLK').
        api_key : str, optional
            Finnhub personal API key.
        timeout : int
            HTTP connection & read timeout in seconds.
            
        Returns
        -------
        LiveResult
            Standardized result envelope with quote metrics and formatted DataFrame.
        """
        sym = (symbol or "SPY").strip().upper()
        data_source = f"Finnhub /v1/quote ({sym})"

        if not api_key or not api_key.strip():
            return LiveResult(
                provider=cls.PROVIDER_NAME,
                status=ProviderStatus.NOT_CONFIGURED,
                data_source=data_source,
                error_message="Finnhub API key is not configured.",
            )

        clean_key = api_key.strip()
        headers = {
            "X-Finnhub-Token": clean_key,
            "User-Agent": "SerendibEndowmentRiskEngine/1.0",
        }
        params = {"symbol": sym}

        try:
            response = requests.get(
                f"{cls.BASE_URL}/quote",
                params=params,
                headers=headers,
                timeout=timeout,
            )
            http_status = response.status_code

            if http_status in (401, 403):
                return LiveResult(
                    provider=cls.PROVIDER_NAME,
                    status=ProviderStatus.FAILED,
                    http_status=http_status,
                    data_source=data_source,
                    error_message="Authentication failed: Invalid Finnhub API Key.",
                )

            if http_status == 429:
                return LiveResult(
                    provider=cls.PROVIDER_NAME,
                    status=ProviderStatus.RATE_LIMITED,
                    http_status=http_status,
                    data_source=data_source,
                    error_message="Finnhub rate limit reached (Free tier: max 60 calls/minute).",
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
                    error_message="Unexpected response format from Finnhub.",
                )

            current_price = data.get("c")
            prev_close = data.get("pc")
            epoch_time = data.get("t", 0)

            if current_price is None or (current_price == 0 and prev_close == 0 and epoch_time == 0):
                return LiveResult(
                    provider=cls.PROVIDER_NAME,
                    status=ProviderStatus.FAILED,
                    http_status=http_status,
                    data_source=data_source,
                    raw_data=data,
                    error_message=f"Finnhub returned an empty quote for {sym}.",
                )

            # Format market timestamp
            market_time_str = "N/A"
            if epoch_time and epoch_time > 0:
                try:
                    dt = datetime.fromtimestamp(epoch_time, tz=timezone.utc)
                    market_time_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                except Exception:
                    market_time_str = str(epoch_time)

            metrics = {
                "symbol": sym,
                "current_price": float(current_price) if current_price is not None else None,
                "change": float(data.get("d", 0.0)),
                "percent_change": float(data.get("dp", 0.0)),
                "high_price": float(data.get("h", 0.0)),
                "low_price": float(data.get("l", 0.0)),
                "open_price": float(data.get("o", 0.0)),
                "previous_close": float(prev_close) if prev_close is not None else None,
                "market_timestamp": market_time_str,
            }

            quote_table = pd.DataFrame([
                {"Field": "Symbol", "Value": sym},
                {"Field": "Current Price", "Value": f"${metrics['current_price']:.2f}" if metrics['current_price'] else "N/A"},
                {"Field": "Change ($)", "Value": f"${metrics['change']:+.2f}"},
                {"Field": "Change (%)", "Value": f"{metrics['percent_change']:+.2f}%"},
                {"Field": "Day High", "Value": f"${metrics['high_price']:.2f}"},
                {"Field": "Day Low", "Value": f"${metrics['low_price']:.2f}"},
                {"Field": "Open Price", "Value": f"${metrics['open_price']:.2f}"},
                {"Field": "Previous Close", "Value": f"${metrics['previous_close']:.2f}" if metrics['previous_close'] else "N/A"},
                {"Field": "Market Timestamp", "Value": market_time_str},
            ])

            return LiveResult(
                provider=cls.PROVIDER_NAME,
                status=ProviderStatus.CONNECTED,
                http_status=http_status,
                market_timestamp=market_time_str,
                record_count=1,
                data_source=data_source,
                raw_data=data,
                dataframe=quote_table,
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
                error_message="Network connection failed while reaching Finnhub.",
            )
        except Exception as e:
            logger.warning(f"Unexpected Finnhub collector exception: {e}")
            return LiveResult(
                provider=cls.PROVIDER_NAME,
                status=ProviderStatus.FAILED,
                data_source=data_source,
                error_message=sanitize_secret(f"Unexpected error: {str(e)}", [clean_key]),
            )

    @classmethod
    def fetch_spy_quote(cls, api_key: Optional[str], timeout: int = 8) -> LiveResult:
        """Backward-compatible alias for SPY quote."""
        return cls.fetch_etf_quote(symbol="SPY", api_key=api_key, timeout=timeout)

    @classmethod
    def fetch_financial_news(
        cls,
        category: str = "general",
        symbol: Optional[str] = None,
        api_key: Optional[str] = None,
        limit: int = 15,
        timeout: int = 8,
    ) -> LiveResult:
        """
        Retrieve financial market news from Finnhub.
        
        Parameters
        ----------
        category : str
            News category ('general', 'forex', 'crypto', 'merger').
        symbol : str, optional
            Filter news by specific ETF or company symbol if specified.
        api_key : str, optional
            Finnhub personal API key.
        limit : int
            Maximum number of news articles to return.
        timeout : int
            HTTP timeout in seconds.
            
        Returns
        -------
        LiveResult
            Standardized result envelope with news items DataFrame and metadata.
        """
        data_source = f"Finnhub /v1/news ({symbol or category})"

        if not api_key or not api_key.strip():
            return LiveResult(
                provider=cls.PROVIDER_NAME,
                status=ProviderStatus.NOT_CONFIGURED,
                data_source=data_source,
                error_message="Finnhub API key is not configured.",
            )

        clean_key = api_key.strip()
        headers = {
            "X-Finnhub-Token": clean_key,
            "User-Agent": "SerendibEndowmentRiskEngine/1.0",
        }

        try:
            # If symbol provided, query company news; otherwise general market news
            if symbol and symbol.strip().upper() not in ("ALL", "GENERAL"):
                sym = symbol.strip().upper()
                today = datetime.now().strftime("%Y-%m-%d")
                week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                endpoint = f"{cls.BASE_URL}/company-news"
                params = {"symbol": sym, "from": week_ago, "to": today}
                data_source = f"Finnhub /v1/company-news ({sym})"
            else:
                endpoint = f"{cls.BASE_URL}/news"
                params = {"category": category}
                data_source = f"Finnhub /v1/news ({category})"

            response = requests.get(endpoint, params=params, headers=headers, timeout=timeout)
            http_status = response.status_code

            if http_status in (401, 403):
                return LiveResult(
                    provider=cls.PROVIDER_NAME,
                    status=ProviderStatus.FAILED,
                    http_status=http_status,
                    data_source=data_source,
                    error_message="Authentication failed: Invalid Finnhub API Key.",
                )

            if http_status == 429:
                return LiveResult(
                    provider=cls.PROVIDER_NAME,
                    status=ProviderStatus.RATE_LIMITED,
                    http_status=http_status,
                    data_source=data_source,
                    error_message="Finnhub rate limit reached (Free tier: max 60 calls/minute).",
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

            raw_news = response.json()
            if not isinstance(raw_news, list):
                return LiveResult(
                    provider=cls.PROVIDER_NAME,
                    status=ProviderStatus.FAILED,
                    http_status=http_status,
                    data_source=data_source,
                    error_message="Unexpected news format received from Finnhub.",
                )

            articles: List[Dict[str, Any]] = []
            for item in raw_news[:limit]:
                epoch_t = item.get("datetime", 0)
                pub_time = "N/A"
                if epoch_t and epoch_t > 0:
                    try:
                        pub_time = datetime.fromtimestamp(epoch_t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                    except Exception:
                        pub_time = str(epoch_t)

                articles.append({
                    "headline": item.get("headline", "No headline"),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", "Finnhub"),
                    "datetime": pub_time,
                    "related": item.get("related", symbol or "General"),
                    "url": item.get("url", "#"),
                })

            df = pd.DataFrame(articles)
            latest_time = articles[0]["datetime"] if articles else "N/A"

            return LiveResult(
                provider=cls.PROVIDER_NAME,
                status=ProviderStatus.CONNECTED,
                http_status=http_status,
                market_timestamp=latest_time,
                record_count=len(articles),
                data_source=data_source,
                dataframe=df,
                metrics={"articles": articles, "count": len(articles)},
            )

        except requests.exceptions.Timeout:
            return LiveResult(
                provider=cls.PROVIDER_NAME,
                status=ProviderStatus.FAILED,
                data_source=data_source,
                error_message=f"News request timed out after {timeout} seconds.",
            )
        except requests.exceptions.ConnectionError:
            return LiveResult(
                provider=cls.PROVIDER_NAME,
                status=ProviderStatus.FAILED,
                data_source=data_source,
                error_message="Network connection failed while reaching Finnhub News.",
            )
        except Exception as e:
            return LiveResult(
                provider=cls.PROVIDER_NAME,
                status=ProviderStatus.FAILED,
                data_source=data_source,
                error_message=sanitize_secret(f"Unexpected news error: {str(e)}", [clean_key]),
            )
