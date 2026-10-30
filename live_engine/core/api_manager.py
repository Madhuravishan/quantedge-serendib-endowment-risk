"""
Live API Manager & Multi-Provider Orchestrator
QuantEdge 1.0 - Serendib University Endowment Risk Engine (Phase 2)
"""

import logging
from typing import Dict, List, Optional

from live_engine.core.data_validator import DataFreshness, ProviderStatus, LiveResult, sanitize_secret
from live_engine.core.cache_manager import LiveCacheManager

logger = logging.getLogger(__name__)


class LiveAPIManager:
    """
    Coordinates independent execution of external market data queries.
    Integrates memory caching, failure isolation, and rate-limit preservation.
    """

    _cache = LiveCacheManager()

    @classmethod
    def fetch_etf_quote(
        cls,
        symbol: str,
        finnhub_key: Optional[str] = None,
        use_cache: bool = True,
        ttl_seconds: int = 180,
    ) -> LiveResult:
        """Fetch quote for an ETF with caching."""
        sym = (symbol or "SPY").strip().upper()
        cache_key = f"quote_{sym}"

        if use_cache:
            cached = cls._cache.get(cache_key, max_age_seconds=ttl_seconds)
            if cached and cached.status == ProviderStatus.CONNECTED:
                return cached

        from live_engine.collectors.finnhub_collector import FinnhubCollector
        res = FinnhubCollector.fetch_etf_quote(symbol=sym, api_key=finnhub_key)
        if res.status == ProviderStatus.CONNECTED:
            res.freshness = DataFreshness.FRESH
            cls._cache.set(cache_key, res)
        return res

    @classmethod
    def fetch_etf_daily_series(
        cls,
        symbol: str,
        alphavantage_key: Optional[str] = None,
        outputsize: str = "compact",
        use_cache: bool = True,
        ttl_seconds: int = 900,
    ) -> LiveResult:
        """Fetch daily time series for an ETF with caching."""
        sym = (symbol or "SPY").strip().upper()
        cache_key = f"daily_{sym}_{outputsize}"

        if use_cache:
            cached = cls._cache.get(cache_key, max_age_seconds=ttl_seconds)
            if cached and cached.status == ProviderStatus.CONNECTED:
                return cached

        from live_engine.collectors.alphavantage_collector import AlphaVantageCollector
        res = AlphaVantageCollector.fetch_etf_daily(symbol=sym, api_key=alphavantage_key, outputsize=outputsize)
        if res.status == ProviderStatus.CONNECTED:
            res.freshness = DataFreshness.FRESH
            cls._cache.set(cache_key, res)
        return res

    @classmethod
    def fetch_economic_indicator(
        cls,
        series_id: str,
        fred_key: Optional[str] = None,
        limit: int = 60,
        use_cache: bool = True,
        ttl_seconds: int = 900,
    ) -> LiveResult:
        """Fetch FRED economic series with caching."""
        sid = (series_id or "DTB3").strip().upper()
        cache_key = f"fred_{sid}_{limit}"

        if use_cache:
            cached = cls._cache.get(cache_key, max_age_seconds=ttl_seconds)
            if cached and cached.status == ProviderStatus.CONNECTED:
                return cached

        from live_engine.collectors.fred_collector import FREDCollector
        res = FREDCollector.fetch_economic_series(series_id=sid, api_key=fred_key, limit=limit)
        if res.status == ProviderStatus.CONNECTED:
            res.freshness = DataFreshness.FRESH
            cls._cache.set(cache_key, res)
        return res

    @classmethod
    def fetch_news(
        cls,
        category: str = "general",
        symbol: Optional[str] = None,
        finnhub_key: Optional[str] = None,
        limit: int = 15,
        use_cache: bool = True,
        ttl_seconds: int = 300,
    ) -> LiveResult:
        """Fetch market news with caching."""
        cache_key = f"news_{category}_{symbol or 'ALL'}_{limit}"

        if use_cache:
            cached = cls._cache.get(cache_key, max_age_seconds=ttl_seconds)
            if cached and cached.status == ProviderStatus.CONNECTED:
                return cached

        from live_engine.collectors.finnhub_collector import FinnhubCollector
        res = FinnhubCollector.fetch_financial_news(
            category=category,
            symbol=symbol,
            api_key=finnhub_key,
            limit=limit,
        )
        if res.status == ProviderStatus.CONNECTED:
            res.freshness = DataFreshness.FRESH
            cls._cache.set(cache_key, res)
        return res

    @classmethod
    def fetch_sec_submissions(
        cls,
        cik: Optional[str] = None,
        user_agent: Optional[str] = None,
        limit: int = 15,
        use_cache: bool = True,
        ttl_seconds: int = 900,
    ) -> LiveResult:
        """Fetch SEC filings with caching."""
        from live_engine.collectors.sec_collector import SECCollector
        target_cik = (cik or SECCollector.DEFAULT_CIK).strip().zfill(10)
        cache_key = f"sec_{target_cik}_{limit}"

        if use_cache:
            cached = cls._cache.get(cache_key, max_age_seconds=ttl_seconds)
            if cached and cached.status == ProviderStatus.CONNECTED:
                return cached

        res = SECCollector.fetch_recent_filings(
            user_agent=user_agent,
            cik=target_cik,
            limit=limit,
        )
        if res.status == ProviderStatus.CONNECTED:
            res.freshness = DataFreshness.FRESH
            cls._cache.set(cache_key, res)
        return res

    @classmethod
    def run_all_queries(
        cls,
        finnhub_key: Optional[str] = None,
        alphavantage_key: Optional[str] = None,
        fred_key: Optional[str] = None,
        sec_user_agent: Optional[str] = None,
        sec_cik: Optional[str] = None,
        timeout: int = 10,
    ) -> Dict[str, LiveResult]:
        """
        Execute baseline queries across all 4 configured data providers independently.
        Maintains backward compatibility with Phase 1.
        """
        all_keys: List[str] = [k for k in (finnhub_key, alphavantage_key, fred_key) if k]
        results: Dict[str, LiveResult] = {}

        from live_engine.collectors.finnhub_collector import FinnhubCollector
        from live_engine.collectors.alphavantage_collector import AlphaVantageCollector
        from live_engine.collectors.fred_collector import FREDCollector
        from live_engine.collectors.sec_collector import SECCollector

        # 1. Finnhub
        try:
            results["finnhub"] = FinnhubCollector.fetch_spy_quote(
                api_key=finnhub_key,
                timeout=timeout,
            )
        except Exception as e:
            logger.error(f"Uncaught exception in Finnhub collector: {e}")
            results["finnhub"] = LiveResult(
                provider=FinnhubCollector.PROVIDER_NAME,
                status=ProviderStatus.FAILED,
                data_source=FinnhubCollector.DATA_SOURCE,
                error_message=sanitize_secret(f"Execution error: {str(e)}", all_keys),
            )

        # 2. Alpha Vantage
        try:
            results["alphavantage"] = AlphaVantageCollector.fetch_spy_daily(
                api_key=alphavantage_key,
                timeout=timeout,
            )
        except Exception as e:
            logger.error(f"Uncaught exception in Alpha Vantage collector: {e}")
            results["alphavantage"] = LiveResult(
                provider=AlphaVantageCollector.PROVIDER_NAME,
                status=ProviderStatus.FAILED,
                data_source=AlphaVantageCollector.DATA_SOURCE,
                error_message=sanitize_secret(f"Execution error: {str(e)}", all_keys),
            )

        # 3. FRED
        try:
            results["fred"] = FREDCollector.fetch_dtb3_series(
                api_key=fred_key,
                timeout=timeout,
            )
        except Exception as e:
            logger.error(f"Uncaught exception in FRED collector: {e}")
            results["fred"] = LiveResult(
                provider=FREDCollector.PROVIDER_NAME,
                status=ProviderStatus.FAILED,
                data_source=FREDCollector.DATA_SOURCE,
                error_message=sanitize_secret(f"Execution error: {str(e)}", all_keys),
            )

        # 4. SEC EDGAR
        try:
            results["sec"] = SECCollector.fetch_recent_filings(
                user_agent=sec_user_agent,
                cik=sec_cik,
                timeout=timeout,
            )
        except Exception as e:
            logger.error(f"Uncaught exception in SEC collector: {e}")
            results["sec"] = LiveResult(
                provider=SECCollector.PROVIDER_NAME,
                status=ProviderStatus.FAILED,
                data_source=SECCollector.DATA_SOURCE,
                error_message=sanitize_secret(f"Execution error: {str(e)}", all_keys),
            )

        return results
