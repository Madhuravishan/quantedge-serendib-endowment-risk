"""
SEC EDGAR Public Submissions Collector
QuantEdge 1.0 - Serendib University Endowment Risk Engine (Phase 2)
"""

import logging
from typing import Any, Dict, List, Optional
import pandas as pd
import requests

from live_engine.core.data_validator import ProviderStatus, LiveResult, sanitize_secret

logger = logging.getLogger(__name__)

SEC_ENTITY_CATALOG: Dict[str, Dict[str, str]] = {
    "0000884394": {
        "name": "SPDR S&P 500 ETF Trust",
        "ticker": "SPY",
        "type": "Unit Investment Trust (ETF)",
        "role": "Core endowment equity compounder.",
    },
    "0000320193": {
        "name": "Apple Inc.",
        "ticker": "AAPL",
        "type": "Operating Public Corporation",
        "role": "Top weight constituent in SPY / US equity index.",
    },
    "0000789019": {
        "name": "Microsoft Corporation",
        "ticker": "MSFT",
        "type": "Operating Public Corporation",
        "role": "Top weight constituent in SPY and enterprise software anchor.",
    },
    "0001045810": {
        "name": "NVIDIA Corporation",
        "ticker": "NVDA",
        "type": "Operating Public Corporation",
        "role": "Leading semiconductor and AI infrastructure provider.",
    },
    "0001137360": {
        "name": "iShares Trust",
        "ticker": "AGG / EEM / EFA",
        "type": "Open-End Investment Management Company",
        "role": "Fund family issuer for endowment bond and international allocations.",
    },
}


class SECCollector:
    """Retrieves public regulatory filings from SEC EDGAR API (no API key required)."""

    BASE_URL = "https://data.sec.gov/submissions"
    PROVIDER_NAME = "SEC EDGAR"
    DATA_SOURCE = "SEC EDGAR Submissions API"
    DEFAULT_CIK = "0000884394"  # SPDR S&P 500 ETF TRUST
    DEFAULT_USER_AGENT = "SerendibEndowmentRiskEngine/1.0 (endowment-audit@serendib.ac.lk)"

    @classmethod
    def fetch_recent_filings(
        cls,
        user_agent: Optional[str] = None,
        cik: Optional[str] = None,
        limit: int = 15,
        timeout: int = 8,
    ) -> LiveResult:
        """
        Retrieve recent corporate regulatory filings for an entity from SEC EDGAR.
        
        Parameters
        ----------
        user_agent : str, optional
            SEC-compliant User-Agent header (format: "AppName/Version (email@domain.com)").
        cik : str, optional
            10-digit Central Index Key (CIK). Defaults to SPDR S&P 500 ETF Trust (0000884394).
        limit : int
            Number of recent filings to display.
        timeout : int
            HTTP timeout in seconds.
            
        Returns
        -------
        LiveResult
            Standardized result envelope with filings DataFrame and document URLs.
        """
        ua = (user_agent or "").strip()
        if not ua:
            ua = cls.DEFAULT_USER_AGENT

        # SEC mandates an email or contact information in User-Agent
        if "@" not in ua or "." not in ua:
            return LiveResult(
                provider=cls.PROVIDER_NAME,
                status=ProviderStatus.FAILED,
                data_source=cls.DATA_SOURCE,
                error_message="Invalid SEC User-Agent: SEC policy requires a contact email in the header (e.g. 'Name/1.0 (user@domain.com)').",
            )

        target_cik = (cik or cls.DEFAULT_CIK).strip().zfill(10)
        url = f"{cls.BASE_URL}/CIK{target_cik}.json"
        headers = {
            "User-Agent": ua,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }

        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            http_status = response.status_code

            if http_status == 403:
                return LiveResult(
                    provider=cls.PROVIDER_NAME,
                    status=ProviderStatus.FAILED,
                    http_status=http_status,
                    data_source=cls.DATA_SOURCE,
                    error_message="SEC EDGAR rejected request (HTTP 403 Forbidden). Ensure User-Agent contains a valid organization and contact email.",
                )

            if http_status == 429:
                return LiveResult(
                    provider=cls.PROVIDER_NAME,
                    status=ProviderStatus.RATE_LIMITED,
                    http_status=http_status,
                    data_source=cls.DATA_SOURCE,
                    error_message="SEC EDGAR rate limit exceeded (Max 10 requests per second allowed).",
                )

            if http_status != 200:
                err_msg = sanitize_secret(response.text[:200])
                return LiveResult(
                    provider=cls.PROVIDER_NAME,
                    status=ProviderStatus.FAILED,
                    http_status=http_status,
                    data_source=cls.DATA_SOURCE,
                    error_message=f"HTTP {http_status}: {err_msg}",
                )

            data = response.json()
            if not isinstance(data, dict):
                return LiveResult(
                    provider=cls.PROVIDER_NAME,
                    status=ProviderStatus.FAILED,
                    http_status=http_status,
                    data_source=cls.DATA_SOURCE,
                    error_message="Unexpected response format from SEC EDGAR.",
                )

            entity_name = data.get("name", "Unknown Entity")
            tickers = data.get("tickers", [])
            ticker_label = tickers[0] if tickers else "N/A"
            filings_block = data.get("filings", {}).get("recent", {})

            forms = filings_block.get("form", [])
            filing_dates = filings_block.get("filingDate", [])
            report_dates = filings_block.get("reportDate", [])
            descriptions = filings_block.get("primaryDocDescription", [])
            accessions = filings_block.get("accessionNumber", [])
            primary_docs = filings_block.get("primaryDocument", [])

            total_filings = len(forms)
            if total_filings == 0:
                return LiveResult(
                    provider=cls.PROVIDER_NAME,
                    status=ProviderStatus.CONNECTED,
                    http_status=http_status,
                    data_source=f"SEC EDGAR (CIK {target_cik} - {entity_name})",
                    record_count=0,
                    error_message="No recent filings found for this entity.",
                )

            sample_count = min(limit, total_filings)
            records = []
            cik_int = str(int(target_cik))  # SEC archive folder removes leading zeros
            for i in range(sample_count):
                acc = accessions[i] if i < len(accessions) else ""
                acc_clean = acc.replace("-", "")
                p_doc = primary_docs[i] if i < len(primary_docs) else ""

                if acc_clean and p_doc:
                    doc_link = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{p_doc}"
                elif acc:
                    doc_link = f"https://www.sec.gov/edgar/browse/?CIK={target_cik}"
                else:
                    doc_link = "#"

                records.append({
                    "Form": forms[i] if i < len(forms) else "",
                    "Filing Date": filing_dates[i] if i < len(filing_dates) else "",
                    "Report Date": report_dates[i] if i < len(report_dates) else "",
                    "Description": descriptions[i] if i < len(descriptions) else "",
                    "Accession Number": acc,
                    "Document Link": doc_link,
                })

            df = pd.DataFrame(records)
            latest_market_time = filing_dates[0] if filing_dates else "N/A"

            metrics = {
                "entity_name": entity_name,
                "cik": target_cik,
                "ticker": ticker_label,
                "total_recent_filings": total_filings,
                "latest_filing_form": forms[0] if forms else "N/A",
                "market_timestamp": latest_market_time,
                "sample_count": sample_count,
            }

            return LiveResult(
                provider=cls.PROVIDER_NAME,
                status=ProviderStatus.CONNECTED,
                http_status=http_status,
                market_timestamp=latest_market_time,
                record_count=sample_count,
                data_source=f"SEC EDGAR (CIK {target_cik} - {entity_name})",
                raw_data={"name": entity_name, "sample_records": sample_count},
                dataframe=df,
                metrics=metrics,
            )

        except requests.exceptions.Timeout:
            return LiveResult(
                provider=cls.PROVIDER_NAME,
                status=ProviderStatus.FAILED,
                data_source=cls.DATA_SOURCE,
                error_message=f"Request timed out after {timeout} seconds.",
            )
        except requests.exceptions.ConnectionError:
            return LiveResult(
                provider=cls.PROVIDER_NAME,
                status=ProviderStatus.FAILED,
                data_source=cls.DATA_SOURCE,
                error_message="Network connection failed while reaching SEC EDGAR.",
            )
        except Exception as e:
            logger.warning(f"Unexpected SEC collector exception: {e}")
            return LiveResult(
                provider=cls.PROVIDER_NAME,
                status=ProviderStatus.FAILED,
                data_source=cls.DATA_SOURCE,
                error_message=sanitize_secret(f"Unexpected error: {str(e)}"),
            )
