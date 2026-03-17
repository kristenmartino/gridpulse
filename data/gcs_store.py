"""
GCS Parquet persistence for durable API data caching.

Writes successful API fetches to GCS as Parquet so data survives container
recycles. Reads from GCS as fallback when both the API and SQLite cache
are unavailable.

Path convention:
    gs://{bucket}/{prefix}/demand/{region}/latest.parquet
    gs://{bucket}/{prefix}/weather/{region}/latest.parquet

Design:
- Writes are fire-and-forget (background thread) — no latency on happy path
- Reads are synchronous (fallback path — better than demo data)
- All errors swallowed — GCS unavailability must never crash the app
- Lazy client init — safe to import even when GCS is disabled
"""

from __future__ import annotations

import io
import threading
from typing import TYPE_CHECKING

import pandas as pd
import structlog

from config import GCS_BUCKET_NAME, GCS_ENABLED, GCS_PATH_PREFIX

if TYPE_CHECKING:
    from google.cloud.storage import Client

log = structlog.get_logger()

_client: Client | None = None
_client_lock = threading.Lock()


def _get_client() -> Client | None:
    """Lazily initialize the GCS client singleton.

    Uses Application Default Credentials (ADC) — works on Cloud Run
    without explicit credential configuration.
    """
    global _client
    if not GCS_ENABLED or not GCS_BUCKET_NAME:
        return None
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        try:
            from google.cloud.storage import Client as StorageClient

            _client = StorageClient()
            log.info("gcs_client_initialized", bucket=GCS_BUCKET_NAME)
        except Exception as e:
            log.warning("gcs_client_init_failed", error=str(e))
            return None
    return _client


def _blob_path(data_type: str, region: str) -> str:
    """Build the GCS object path."""
    return f"{GCS_PATH_PREFIX}/{data_type}/{region}/latest.parquet"


def write_parquet(df: pd.DataFrame, data_type: str, region: str) -> None:
    """Write a DataFrame to GCS as Parquet in a background thread.

    Fire-and-forget — never blocks the caller, never raises.

    Args:
        df: DataFrame to persist.
        data_type: 'demand' or 'weather'.
        region: Balancing authority code.
    """
    if not GCS_ENABLED or df.empty:
        return

    try:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False, engine="pyarrow")
        parquet_bytes = buf.getvalue()
    except Exception as e:
        log.warning(
            "gcs_parquet_serialize_failed",
            data_type=data_type,
            region=region,
            error=str(e),
        )
        return

    def _upload() -> None:
        try:
            client = _get_client()
            if client is None:
                return
            bucket = client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(_blob_path(data_type, region))
            blob.upload_from_string(parquet_bytes, content_type="application/octet-stream")
            log.info(
                "gcs_parquet_written",
                data_type=data_type,
                region=region,
                size_bytes=len(parquet_bytes),
            )
        except Exception as e:
            log.warning("gcs_write_failed", data_type=data_type, region=region, error=str(e))

    threading.Thread(target=_upload, daemon=True).start()


def read_parquet(data_type: str, region: str) -> pd.DataFrame | None:
    """Read a DataFrame from GCS Parquet. Returns None on any failure.

    Synchronous — called on the fallback path where latency is acceptable
    (the alternative is showing demo data).

    Args:
        data_type: 'demand' or 'weather'.
        region: Balancing authority code.

    Returns:
        DataFrame or None if GCS is unavailable or blob doesn't exist.
    """
    if not GCS_ENABLED:
        return None

    try:
        client = _get_client()
        if client is None:
            return None
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(_blob_path(data_type, region))
        if not blob.exists():
            log.debug("gcs_blob_not_found", data_type=data_type, region=region)
            return None
        data = blob.download_as_bytes()
        df = pd.read_parquet(io.BytesIO(data), engine="pyarrow")
        log.info("gcs_parquet_read", data_type=data_type, region=region, rows=len(df))
        return df
    except Exception as e:
        log.warning("gcs_read_failed", data_type=data_type, region=region, error=str(e))
        return None
