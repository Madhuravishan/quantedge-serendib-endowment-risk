"""
Data Models & Validation Utilities for Live Market Data Collectors
QuantEdge 1.0 - Serendib University Endowment Risk Engine (Phase 2)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
from typing import Any, Dict, List, Optional
import pandas as pd


class ProviderStatus(str, Enum):
    CONNECTED = "Connected"
    FAILED = "Failed"
    RATE_LIMITED = "Rate Limited"
    NOT_CONFIGURED = "Not Configured"


class DataFreshness(str, Enum):
    FRESH = "Fresh"
    CACHED = "Cached"
    DELAYED = "Delayed"
    UNAVAILABLE = "Unavailable"


@dataclass
class LiveResult:
    """Standardized result envelope for external data provider queries."""
    provider: str
    status: ProviderStatus
    http_status: Optional[int] = None
    retrieval_timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    market_timestamp: Optional[str] = None
    record_count: int = 0
    data_source: str = ""
    freshness: DataFreshness = DataFreshness.FRESH
    error_message: Optional[str] = None
    raw_data: Optional[Any] = None
    dataframe: Optional[pd.DataFrame] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Provider": self.provider,
            "Status": self.status.value,
            "HTTP Status": self.http_status or "N/A",
            "Retrieval Time": self.retrieval_timestamp,
            "Market Time": self.market_timestamp or "N/A",
            "Freshness": self.freshness.value,
            "Records": self.record_count,
            "Data Source": self.data_source,
            "Error": self.error_message or "None",
        }


def sanitize_secret(message: Optional[str], secrets: Optional[List[str]] = None) -> str:
    """
    Sanitize error strings and URLs to prevent credential leakage.
    Redacts any passed secrets as well as standard URL query credentials.
    """
    if not message:
        return ""

    sanitized = str(message)

    # Redact explicit secret strings if provided and non-trivial
    if secrets:
        for secret in secrets:
            if secret and len(secret.strip()) >= 4:
                sanitized = sanitized.replace(secret.strip(), "[REDACTED]")

    # Redact common credential patterns in URLs / query strings
    patterns = [
        r'(token=)[^&"\']+',
        r'(apikey=)[^&"\']+',
        r'(api_key=)[^&"\']+',
        r'(key=)[^&"\']+',
        r'(bearer\s+)[a-zA-Z0-9_\-\.]+',
    ]
    for pattern in patterns:
        sanitized = re.sub(pattern, r'\1[REDACTED]', sanitized, flags=re.IGNORECASE)

    return sanitized
