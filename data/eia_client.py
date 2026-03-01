"""
EIA API v2 client for energy data ingestion.

Fetches hourly demand, generation by fuel type, and interchange data
for 8 balancing authorities. Implements pagination, caching, and
exponential backoff for rate limiting.

API docs: https://www.eia.gov/opendata/documentation.php
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import structlog

from config import (
    EIA_API_KEY,
    EIA_BASE_URL,
    EIA_ENDPOINTS,
    REGION_COORDINATES,
    CACHE_TTL_SECONDS,
)
from data.cache import get_cache

log = structlog.get_logger()

# EIA API pagination limit
EIA_PAGE_SIZE = 5000
# Max retries for rate limiting (HTTP 429)
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2.0

# Map internal region names to EIA API respondent codes
# EIA uses abbreviated codes that differ from common names
EIA_REGION_CODES = {
    "ERCOT": "ERCO",
    "FPL": "FPL",
    "CAISO": "CISO",
    "PJM": "PJM",
    "MISO": "MISO",
    "NYISO": "NYIS",
    "ISONE": "ISNE",
    "SPP": "SWPP",
}


def _get_eia_code(region: str) -> str:
    """Convert internal region name to EIA API respondent code."""
    return EIA_REGION_CODES.get(region, region)


def fetch_demand(
    region: str,
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch hourly demand data from EIA API for a balancing authority.

    Args:
        region: Balancing authority code (e.g., "ERCOT", "FPL").
        start: Start date ISO format (default: 90 days ago).
        end: End date ISO format (default: now).
        use_cache: Whether to check cache first.

    Returns:
        DataFrame with columns: [timestamp, demand_mw, forecast_mw, region]
    """
    if region not in REGION_COORDINATES:
        raise ValueError(f"Unknown region: {region}. Valid: {list(REGION_COORDINATES.keys())}")

    if start is None:
        start = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%dT00")
    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%dT23")

    # Use date-level granularity for cache key so it doesn't change every hour
    cache_key = f"eia_demand_{region}_{start[:10]}_{end[:10]}"
    cache = get_cache()

    if use_cache:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    log.info("eia_fetching_demand", region=region, start=start, end=end)

    eia_code = _get_eia_code(region)
    params = {
        "api_key": EIA_API_KEY,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": eia_code,
        "facets[type][]": ["D", "DF"],  # D=Demand, DF=Day-ahead demand forecast
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": EIA_PAGE_SIZE,
    }

    all_records = _paginated_fetch(EIA_ENDPOINTS["demand"], params)

    if not all_records:
        log.warning("eia_no_data", region=region, start=start, end=end)
        # Try stale cache
        stale = cache.get(cache_key, allow_stale=True)
        if stale is not None:
            return stale
        return pd.DataFrame(columns=["timestamp", "demand_mw", "forecast_mw", "region"])

    df = _parse_demand_records(all_records, region)
    cache.set(cache_key, df, ttl=CACHE_TTL_SECONDS)
    log.info("eia_demand_cached", region=region, rows=len(df))
    return df


def fetch_generation_by_fuel(
    region: str,
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch hourly generation by fuel type from EIA API.

    Args:
        region: Balancing authority code.
        start: Start date ISO format.
        end: End date ISO format.
        use_cache: Whether to check cache first.

    Returns:
        DataFrame with columns: [timestamp, fuel_type, generation_mw, region]
    """
    if region not in REGION_COORDINATES:
        raise ValueError(f"Unknown region: {region}")

    if start is None:
        start = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%dT%H")
    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")

    cache_key = f"eia_gen_{region}_{start}_{end}"
    cache = get_cache()

    if use_cache:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    log.info("eia_fetching_generation", region=region, start=start, end=end)

    eia_code = _get_eia_code(region)
    params = {
        "api_key": EIA_API_KEY,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": eia_code,
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": EIA_PAGE_SIZE,
    }

    all_records = _paginated_fetch(EIA_ENDPOINTS["fuel_type"], params)

    if not all_records:
        log.warning("eia_no_generation_data", region=region)
        stale = cache.get(cache_key, allow_stale=True)
        if stale is not None:
            return stale
        return pd.DataFrame(columns=["timestamp", "fuel_type", "generation_mw", "region"])

    df = _parse_generation_records(all_records, region)
    cache.set(cache_key, df, ttl=CACHE_TTL_SECONDS)
    log.info("eia_generation_cached", region=region, rows=len(df))
    return df


def fetch_interchange(
    region: str,
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch hourly interchange (power flow between BAs) from EIA API.

    Returns:
        DataFrame with columns: [timestamp, from_ba, to_ba, interchange_mw]
    """
    if region not in REGION_COORDINATES:
        raise ValueError(f"Unknown region: {region}")

    if start is None:
        start = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%dT%H")
    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")

    cache_key = f"eia_interchange_{region}_{start}_{end}"
    cache = get_cache()

    if use_cache:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    log.info("eia_fetching_interchange", region=region)

    eia_code = _get_eia_code(region)
    params = {
        "api_key": EIA_API_KEY,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[fromba][]": eia_code,
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": EIA_PAGE_SIZE,
    }

    all_records = _paginated_fetch(EIA_ENDPOINTS["interchange"], params)

    if not all_records:
        stale = cache.get(cache_key, allow_stale=True)
        if stale is not None:
            return stale
        return pd.DataFrame(columns=["timestamp", "from_ba", "to_ba", "interchange_mw"])

    df = _parse_interchange_records(all_records)
    cache.set(cache_key, df, ttl=CACHE_TTL_SECONDS)
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _paginated_fetch(endpoint: str, params: dict) -> list[dict]:
    """
    Fetch all pages from an EIA API endpoint with retry on rate limiting.

    Returns:
        List of all data records across pages.
    """
    all_records: list[dict] = []
    offset = 0

    while True:
        params["offset"] = offset
        url = f"{EIA_BASE_URL}/{endpoint}/data/"

        response = _request_with_backoff(url, params)
        if response is None:
            break

        data = response.get("response", {})
        records = data.get("data", [])
        total = int(data.get("total", 0) or 0)

        all_records.extend(records)
        log.debug("eia_page_fetched", offset=offset, records=len(records), total=total)

        offset += EIA_PAGE_SIZE
        if offset >= total or not records:
            break

    return all_records


def _request_with_backoff(url: str, params: dict) -> dict | None:
    """Make an HTTP GET with exponential backoff on 429/5xx."""
    backoff = INITIAL_BACKOFF_SECONDS

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)

            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                log.warning("eia_rate_limited", attempt=attempt, backoff=backoff)
                time.sleep(backoff)
                backoff *= 2
            elif resp.status_code >= 500:
                log.warning("eia_server_error", status=resp.status_code, attempt=attempt)
                time.sleep(backoff)
                backoff *= 2
            else:
                log.error("eia_request_failed", status=resp.status_code, body=resp.text[:200])
                return None

        except requests.RequestException as e:
            log.error("eia_request_exception", error=str(e), attempt=attempt)
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2

    log.error("eia_max_retries_exceeded", url=url)
    return None


def _parse_demand_records(records: list[dict], region: str) -> pd.DataFrame:
    """Parse EIA demand records into a clean DataFrame."""
    rows = []
    for r in records:
        rows.append({
            "timestamp": pd.Timestamp(r["period"], tz="UTC"),
            "value": float(r.get("value", 0) or 0),
            "type": r.get("type", ""),
            "region": region,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "demand_mw", "forecast_mw", "region"])

    # Pivot: D → demand_mw, DF → forecast_mw
    demand = df[df["type"] == "D"][["timestamp", "value", "region"]].rename(
        columns={"value": "demand_mw"}
    )
    forecast = df[df["type"] == "DF"][["timestamp", "value"]].rename(
        columns={"value": "forecast_mw"}
    )

    if not forecast.empty:
        result = demand.merge(forecast, on="timestamp", how="left")
    else:
        result = demand.copy()
        result["forecast_mw"] = None

    result = result.sort_values("timestamp").reset_index(drop=True)
    return result


def _parse_generation_records(records: list[dict], region: str) -> pd.DataFrame:
    """Parse EIA generation-by-fuel records."""
    rows = []
    for r in records:
        rows.append({
            "timestamp": pd.Timestamp(r["period"], tz="UTC"),
            "fuel_type": r.get("fueltype", r.get("type-name", "unknown")),
            "generation_mw": float(r.get("value", 0) or 0),
            "region": region,
        })
    df = pd.DataFrame(rows)
    return df.sort_values("timestamp").reset_index(drop=True)


def _parse_interchange_records(records: list[dict]) -> pd.DataFrame:
    """Parse EIA interchange records."""
    rows = []
    for r in records:
        rows.append({
            "timestamp": pd.Timestamp(r["period"], tz="UTC"),
            "from_ba": r.get("fromba", ""),
            "to_ba": r.get("toba", ""),
            "interchange_mw": float(r.get("value", 0) or 0),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("timestamp").reset_index(drop=True)
