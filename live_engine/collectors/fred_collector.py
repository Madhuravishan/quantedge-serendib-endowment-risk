"""
Federal Reserve Economic Data (FRED) Collector
QuantEdge 1.0 - Serendib University Endowment Risk Engine (Phase 2)
"""

import logging
from typing import Any, Dict, Optional
import pandas as pd
import requests

from live_engine.core.data_validator import ProviderStatus, LiveResult, sanitize_secret

logger = logging.getLogger(__name__)

FRED_SERIES_CATALOG: Dict[str, Dict[str, str]] = {
    "DTB3": {
        "name": "3-Month Treasury Bill Secondary Market Rate",
        "description": "Short-term sovereign borrowing yield. Primary institutional proxy for the risk-free rate ($R_f$) and cash equivalent yield.",
        "frequency": "Daily",
        "units": "Percent (%)",
        "category": "Interest Rates",
    },
    "DGS10": {
        "name": "10-Year Treasury Constant Maturity Rate",
        "description": "Benchmark long-term US sovereign risk-free yield. Foundation for asset pricing models, equity hurdle rates, and discount cash flows.",
        "frequency": "Daily",
        "units": "Percent (%)",
        "category": "Interest Rates",
    },
    "FEDFUNDS": {
        "name": "Federal Funds Effective Rate",
        "description": "Volume-weighted median rate at which commercial banks lend reserve balances to each other overnight. Key indicator of Federal Reserve monetary policy stance.",
        "frequency": "Monthly",
        "units": "Percent (%)",
        "category": "Monetary Policy",
    },
    "CPIAUCSL": {
        "name": "Consumer Price Index for All Urban Consumers",
        "description": "Headline inflation measure tracking price changes for a basket of consumer goods. Direct benchmark for the endowment's 2.5% inflation hurdle.",
        "frequency": "Monthly",
        "units": "Index (1982-1984=100)",
        "category": "Inflation & Prices",
    },
    "UNRATE": {
        "name": "Civilian Unemployment Rate",
        "description": "Percentage of the civilian labor force actively seeking employment. Core macroeconomic indicator of labor market slack and business cycle health.",
        "frequency": "Monthly",
        "units": "Percent (%)",
        "category": "Labor & Growth",
    },
}


class FREDCollector:
    """Retrieves macroeconomic and interest-rate series from Federal Reserve Economic Data (FRED)."""

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    PROVIDER_NAME = "Federal Reserve (FRED)"

    @classmethod
    def fetch_economic_series(
        cls,
        series_id: str,
        api_key: Optional[str],
        limit: int = 60,
        timeout: int = 8,
    ) -> LiveResult:
        """
        Retrieve observations for any supported FRED series identifier.
        
        Parameters
        ----------
        series_id : str
            FRED series identifier (e.g. 'DTB3', 'DGS10', 'FEDFUNDS', 'CPIAUCSL', 'UNRATE').
        api_key : str, optional
            FRED personal API key (32-character hex).
        limit : int
            Number of recent observations to retrieve.
        timeout : int
            HTTP connection & read timeout in seconds.
            
        Returns
        -------
        LiveResult
            Standardized result envelope containing observations DataFrame and metadata.
        """
        sid = (series_id or "DTB3").strip().upper()
        meta = FRED_SERIES_CATALOG.get(sid, {
            "name": f"FRED Series {sid}",
            "description": "Federal Reserve Economic Data series.",
            "frequency": "Variable",
            "units": "Value",
            "category": "Economic Data",
        })
        data_source = f"FRED API (Series: {sid} - {meta['name']})"

        if not api_key or not api_key.strip():
            return LiveResult(
                provider=cls.PROVIDER_NAME,
                status=ProviderStatus.NOT_CONFIGURED,
                data_source=data_source,
                error_message="FRED API key is not configured.",
            )

        clean_key = api_key.strip()
        params = {
            "series_id": sid,
            "api_key": clean_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
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
                    error_message=f"FRED API rate limit reached for {sid} (HTTP 429).",
                )

            if http_status in (400, 403):
                err_text = f"Authentication failed: Invalid FRED API Key or bad request for {sid}."
                try:
                    err_json = response.json()
                    if "error_message" in err_json:
                        err_text = sanitize_secret(err_json["error_message"], [clean_key])
                except Exception:
                    pass

                return LiveResult(
                    provider=cls.PROVIDER_NAME,
                    status=ProviderStatus.FAILED,
                    http_status=http_status,
                    data_source=data_source,
                    error_message=err_text,
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
            if not isinstance(data, dict) or "observations" not in data:
                return LiveResult(
                    provider=cls.PROVIDER_NAME,
                    status=ProviderStatus.FAILED,
                    http_status=http_status,
                    data_source=data_source,
                    error_message=f"Unexpected response format from FRED API for {sid}.",
                )

            observations = data.get("observations", [])
            if not observations:
                return LiveResult(
                    provider=cls.PROVIDER_NAME,
                    status=ProviderStatus.CONNECTED,
                    http_status=http_status,
                    data_source=data_source,
                    record_count=0,
                    error_message=f"Zero observations returned by FRED for {sid}.",
                )

            val_col_name = f"Value ({meta['units']})"
            records = []
            for obs in observations:
                val_str = obs.get("value", "")
                val_float = None
                if val_str and val_str != ".":
                    try:
                        val_float = float(val_str)
                    except ValueError:
                        val_float = None

                records.append({
                    "Date": obs.get("date", "N/A"),
                    val_col_name: val_float,
                    "Raw Value": val_str,
                })

            df = pd.DataFrame(records)

            latest_valid_val = None
            market_time = "N/A"
            for r in records:
                if r[val_col_name] is not None:
                    latest_valid_val = r[val_col_name]
                    market_time = r["Date"]
                    break

            metrics = {
                "series_id": sid,
                "series_name": meta["name"],
                "description": meta["description"],
                "frequency": meta["frequency"],
                "units": meta["units"],
                "latest_value": latest_valid_val,
                "market_timestamp": market_time,
                "observation_count": len(df),
            }

            return LiveResult(
                provider=cls.PROVIDER_NAME,
                status=ProviderStatus.CONNECTED,
                http_status=http_status,
                market_timestamp=market_time,
                record_count=len(df),
                data_source=data_source,
                raw_data={"count": len(observations), "first_sample": observations[0]},
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
                error_message="Network connection failed while reaching FRED.",
            )
        except Exception as e:
            logger.warning(f"Unexpected FRED collector exception: {e}")
            return LiveResult(
                provider=cls.PROVIDER_NAME,
                status=ProviderStatus.FAILED,
                data_source=data_source,
                error_message=sanitize_secret(f"Unexpected error: {str(e)}", [clean_key]),
            )

    @classmethod
    def fetch_dtb3_series(cls, api_key: Optional[str], limit: int = 30, timeout: int = 8) -> LiveResult:
        """Backward-compatible alias for DTB3 series."""
        res = cls.fetch_economic_series(series_id="DTB3", api_key=api_key, limit=limit, timeout=timeout)
        # Ensure 'Yield (%)' column exists for backward compatibility with existing tests
        if res.dataframe is not None and "Yield (%)" not in res.dataframe.columns:
            val_cols = [c for c in res.dataframe.columns if c.startswith("Value (")]
            if val_cols:
                res.dataframe["Yield (%)"] = res.dataframe[val_cols[0]]
            if "latest_value" in res.metrics:
                res.metrics["latest_yield_pct"] = res.metrics["latest_value"]
        return res
