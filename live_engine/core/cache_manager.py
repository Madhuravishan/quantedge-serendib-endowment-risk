"""
In-Memory Cache Manager with TTL & Freshness Tracking
QuantEdge 1.0 - Serendib University Endowment Risk Engine (Phase 2)
"""

from datetime import datetime, timedelta
import threading
from typing import Dict, Optional, Tuple

from live_engine.core.data_validator import DataFreshness, LiveResult


class LiveCacheManager:
    """
    Thread-safe in-memory cache for external market data queries.
    Prevents repetitive requests, rate-limit consumption, and provides explicit freshness metadata.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._store: Dict[str, Tuple[datetime, LiveResult]] = {}
            return cls._instance

    def get(self, key: str, max_age_seconds: int = 300) -> Optional[LiveResult]:
        """
        Retrieve a cached LiveResult if present and within max_age_seconds.
        Marks the result's freshness as CACHED.
        """
        with self._lock:
            if key not in self._store:
                return None

            cached_time, result = self._store[key]
            age = (datetime.now() - cached_time).total_seconds()

            if age > max_age_seconds:
                return None

            # Return a shallow copy with freshness marked as CACHED
            result.freshness = DataFreshness.CACHED
            return result

    def set(self, key: str, result: LiveResult):
        """Store a LiveResult in cache with current timestamp."""
        with self._lock:
            self._store[key] = (datetime.now(), result)

    def clear(self):
        """Wipe all cached entries."""
        with self._lock:
            self._store.clear()

    def get_age_seconds(self, key: str) -> Optional[float]:
        """Get the age in seconds of a cached entry."""
        with self._lock:
            if key in self._store:
                return (datetime.now() - self._store[key][0]).total_seconds()
            return None
