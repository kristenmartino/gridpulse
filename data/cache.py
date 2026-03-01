"""
SQLite-based caching layer with TTL support.

Provides persistent caching for API responses (EIA, Open-Meteo, NOAA)
on Cloud Run's ephemeral disk. Survives across requests within the same
instance but is lost on instance recycle (acceptable — data refreshes from APIs).

Features:
- TTL-based expiration with configurable per-key TTL
- Stale-data fallback: returns expired data when API is down
- WAL mode for concurrent reads with gunicorn workers
- JSON serialization for DataFrames and dicts
"""

from __future__ import annotations

import io
import json
import sqlite3
import time
from contextlib import contextmanager
from typing import Any

import pandas as pd
import structlog

from config import CACHE_DB_PATH, CACHE_TTL_SECONDS

log = structlog.get_logger()


class Cache:
    """SQLite cache with TTL-based expiration and stale fallback."""

    def __init__(self, db_path: str = CACHE_DB_PATH, default_ttl: int = CACHE_TTL_SECONDS):
        self.db_path = db_path
        self.default_ttl = default_ttl
        self._init_db()

    def _init_db(self) -> None:
        """Create cache table if it doesn't exist. Enable WAL mode."""
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    ttl_seconds INTEGER NOT NULL
                )
            """)
            conn.commit()
        log.debug("cache_initialized", db_path=self.db_path)

    @contextmanager
    def _connect(self):
        """Context manager for SQLite connections."""
        conn = sqlite3.connect(self.db_path, timeout=10)
        try:
            yield conn
        finally:
            conn.close()

    def get(self, key: str, allow_stale: bool = False) -> Any | None:
        """
        Retrieve a cached value by key.

        Args:
            key: Cache key.
            allow_stale: If True, return expired data with a warning log.
                         Used for API fallback when external services are down.

        Returns:
            Cached value (DataFrame, dict, or str), or None if not found.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value, data_type, created_at, ttl_seconds FROM cache WHERE key = ?",
                (key,),
            ).fetchone()

        if row is None:
            log.debug("cache_miss", key=key)
            return None

        value_str, data_type, created_at, ttl_seconds = row
        age_seconds = time.time() - created_at
        is_expired = age_seconds > ttl_seconds

        if is_expired and not allow_stale:
            log.debug("cache_expired", key=key, age_seconds=round(age_seconds))
            return None

        if is_expired and allow_stale:
            log.warning(
                "cache_serving_stale",
                key=key,
                age_seconds=round(age_seconds),
                ttl_seconds=ttl_seconds,
            )

        log.debug(
            "cache_hit",
            key=key,
            age_seconds=round(age_seconds),
            stale=is_expired,
        )
        return self._deserialize(value_str, data_type)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        Store a value in the cache.

        Args:
            key: Cache key.
            value: Value to cache (DataFrame, dict, list, or str).
            ttl: TTL in seconds. Defaults to CACHE_TTL_SECONDS.
        """
        ttl = ttl if ttl is not None else self.default_ttl
        value_str, data_type = self._serialize(value)

        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO cache (key, value, data_type, created_at, ttl_seconds)
                   VALUES (?, ?, ?, ?, ?)""",
                (key, value_str, data_type, time.time(), ttl),
            )
            conn.commit()

        log.debug("cache_set", key=key, data_type=data_type, ttl_seconds=ttl)

    def delete(self, key: str) -> None:
        """Remove a key from the cache."""
        with self._connect() as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
        log.debug("cache_deleted", key=key)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._connect() as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()
        log.info("cache_cleared")

    def get_age_seconds(self, key: str) -> float | None:
        """Get the age of a cached entry in seconds. Returns None if not found."""
        with self._connect() as conn:
            row = conn.execute("SELECT created_at FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return time.time() - row[0]

    def is_stale(self, key: str) -> bool | None:
        """Check if a cached entry is past its TTL. Returns None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT created_at, ttl_seconds FROM cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return (time.time() - row[0]) > row[1]

    def _serialize(self, value: Any) -> tuple[str, str]:
        """Serialize a value to JSON string with type tag."""
        if isinstance(value, pd.DataFrame):
            return value.to_json(orient="split", date_format="iso"), "dataframe"
        elif isinstance(value, (dict, list)):
            return json.dumps(value), "json"
        elif isinstance(value, str):
            return value, "string"
        else:
            return json.dumps(value), "json"

    def _deserialize(self, value_str: str, data_type: str) -> Any:
        """Deserialize a JSON string back to its original type."""
        if data_type == "dataframe":
            # Use StringIO to prevent pandas from interpreting JSON as file path
            return pd.read_json(io.StringIO(value_str), orient="split")
        elif data_type == "json":
            return json.loads(value_str)
        elif data_type == "string":
            return value_str
        else:
            return json.loads(value_str)


# Module-level singleton
_cache: Cache | None = None


def get_cache() -> Cache:
    """Get or create the module-level cache singleton."""
    global _cache
    if _cache is None:
        _cache = Cache()
    return _cache
