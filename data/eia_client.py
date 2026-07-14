"""
EIA API v2 client for energy data ingestion.

Fetches hourly demand, generation by fuel type, and interchange data
for the 51 balancing authorities defined in ``config.REGION_COORDINATES``
(~100% of contiguous-US lower-48 load). Implements pagination, caching,
and exponential backoff for rate limiting.

API docs: https://www.eia.gov/opendata/documentation.php
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import requests
import structlog

from config import (
    CACHE_TTL_SECONDS,
    EIA_API_KEY,
    EIA_BASE_URL,
    EIA_ENDPOINTS,
    REGION_COORDINATES,
)
from data.cache import get_cache

log = structlog.get_logger()

# EIA API pagination limit
EIA_PAGE_SIZE = 5000
# Max retries for rate limiting (HTTP 429)
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2.0

# Circuit breaker (see _EIACircuitBreaker). During a sustained EIA outage the
# MAX_RETRIES x 30s + backoff retry loop costs ~150s per failing call; across
# 51 BAs x 3 endpoints that overruns the scoring job's task timeout long before
# the per-call fallbacks (stale cache / GCS) can serve last-known data. The
# breaker trips after this many consecutive hard failures, then fail-fasts
# subsequent calls (one probe every PROBE_INTERVAL suppressed calls lets it
# recover mid-run). State is process-local and resets every fresh job process.
EIA_CIRCUIT_TRIP_THRESHOLD = 3
EIA_CIRCUIT_PROBE_INTERVAL = 30

# Map internal region names to EIA API respondent codes
# EIA uses abbreviated codes that differ from common names
EIA_REGION_CODES = {
    # Original 8.
    "ERCOT": "ERCO",
    "FPL": "FPL",
    "CAISO": "CISO",
    "PJM": "PJM",
    "MISO": "MISO",
    "NYISO": "NYIS",
    "ISONE": "ISNE",
    "SPP": "SWPP",
    # V1.α — 8 new BAs. Internal name == EIA-930 respondent code for all
    # of these; listed explicitly to keep the mapping comprehensive rather
    # than relying on the identity fallback in `_get_eia_code`.
    "SOCO": "SOCO",
    "TVA": "TVA",
    "DUK": "DUK",
    "CPLE": "CPLE",
    "BPAT": "BPAT",
    "AZPS": "AZPS",
    "NEVP": "NEVP",
    "PSCO": "PSCO",
    # V3.ζ — 35 remaining EIA-930 BAs in the contiguous US. All are
    # identity mappings (internal code == EIA-930 respondent code).
    # Listed explicitly for the same reason as V1.α — keeps the mapping
    # exhaustive and self-documenting.
    "FPC": "FPC",
    "TEC": "TEC",
    "FMPP": "FMPP",
    "JEA": "JEA",
    "TAL": "TAL",
    "GVL": "GVL",
    "SEC": "SEC",
    "HST": "HST",
    "SC": "SC",
    "SCEG": "SCEG",
    "CPLW": "CPLW",
    "LGEE": "LGEE",
    "AECI": "AECI",
    "EPE": "EPE",
    "SPA": "SPA",
    "PACE": "PACE",
    "PACW": "PACW",
    "PGE": "PGE",
    "PSEI": "PSEI",
    "SCL": "SCL",
    "TPWR": "TPWR",
    "AVA": "AVA",
    "IPCO": "IPCO",
    "NWMT": "NWMT",
    "GCPD": "GCPD",
    "CHPD": "CHPD",
    "DOPD": "DOPD",
    "BANC": "BANC",
    "LDWP": "LDWP",
    "IID": "IID",
    "TIDC": "TIDC",
    "SRP": "SRP",
    "TEPC": "TEPC",
    "PNM": "PNM",
    "WALC": "WALC",
}


def _get_eia_code(region: str) -> str:
    """Convert internal region name to EIA API respondent code."""
    return EIA_REGION_CODES.get(region, region)


class EIAIncompleteFetchError(RuntimeError):
    """A paginated fetch hard-failed partway through, leaving fewer rows than the
    server-reported total. Raised by :func:`_paginated_fetch` so the caller routes
    to last-known-good instead of caching/persisting a truncated series that would
    poison the 24h cache and overwrite the GCS fallback (#269 / P2-06)."""


def _last_known_good(
    cache,
    cache_key: str,
    data_type: str,
    region: str,
    empty_cols: list[str],
) -> pd.DataFrame:
    """Serve the most recent *real* data for a key without writing anything.

    The uniform ``stale cache -> GCS parquet -> typed-empty`` chain is the #174
    outage-resilience contract. Centralised here so every path that must fall
    back — an empty upstream response, a truncated multi-page fetch (#269), and
    a 200 that parses to zero rows (#270) — shares the same last-known-good
    lookup and, critically, never caches or persists over prior real data.
    """
    stale = cache.get(cache_key, allow_stale=True)
    if stale is not None:
        log.info("eia_stale_fallback", region=region, data_type=data_type, rows=len(stale))
        return stale
    from data.gcs_store import read_parquet

    gcs_df = read_parquet(data_type, region)
    if gcs_df is not None and not gcs_df.empty:
        log.info("eia_gcs_fallback", region=region, data_type=data_type, rows=len(gcs_df))
        return gcs_df
    return pd.DataFrame(columns=empty_cols)


def _fetch_eia(
    *,
    region: str,
    data_type: str,
    cache_key: str,
    endpoint: str,
    params: dict,
    parser: Callable[[list[dict]], pd.DataFrame],
    empty_cols: list[str],
    use_cache: bool,
    value_col: str | None = None,
) -> pd.DataFrame:
    """Shared fetch skeleton for the demand / generation / interchange endpoints.

    Each public fetcher computes its own default dates, cache key, params, and
    parser, then delegates the identical flow here (#185): cache read ->
    paginated fetch -> parse -> cache + GCS write, with a ``stale cache -> GCS
    parquet -> typed-empty`` fallback (:func:`_last_known_good`) whenever the
    fetch does not yield a complete, usable frame. ``data_type`` is the GCS
    parquet key + the logging label.

    ``value_col`` names the single measurement column (e.g. ``"demand_mw"``)
    that must carry at least one real observation for the frame to be usable;
    when given, a parsed frame whose ``value_col`` is entirely NaN is treated as
    a miss (see below). Leave it ``None`` for endpoints whose zero/near-zero
    readings are legitimate (generation can be ~0 for a fuel; interchange can be
    ~0 on a tie), where only a row-count-empty frame is a miss.

    Four failure modes route to last-known-good *without* writing, so none of
    them can poison the 24h cache or overwrite the durable GCS fallback:

    * the upstream response is empty (#174);
    * a page hard-fails mid-pagination, truncating the series (#269 / P2-06) —
      surfaced as :class:`EIAIncompleteFetchError`;
    * a 200 parses to a row-empty frame, e.g. a demand response carrying only
      day-ahead ``DF`` rows and no ``D`` observations (#270 / P2-07);
    * a 200 parses to rows whose ``value_col`` is entirely NaN, e.g. a demand
      window in which every ``D`` observation is null/zero-filled — real rows,
      no usable signal (#270 sibling; would otherwise overwrite the GCS
      last-known-good with an all-NaN frame the ``df.empty`` test misses).
    """
    cache = get_cache()
    if use_cache:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    log.info("eia_fetching", region=region, data_type=data_type)
    try:
        all_records = _paginated_fetch(endpoint, params)
    except EIAIncompleteFetchError as exc:
        # Truncated multi-page fetch — the partial rows must not be cached or
        # persisted over last-known-good. (#269 / P2-06)
        log.warning("eia_incomplete_fetch", region=region, data_type=data_type, error=str(exc))
        return _last_known_good(cache, cache_key, data_type, region, empty_cols)

    df = parser(all_records) if all_records else pd.DataFrame(columns=empty_cols)

    # A frame with no usable data is a miss, not a result — caching it would
    # blank the surface for 24h and shadow the stale/GCS last-known-good, so fall
    # back without writing. Two shapes qualify: a row-empty frame (empty upstream
    # response, or a 200 that parsed to zero rows — #270), and a rows-present
    # frame whose measurement column is entirely NaN (a demand window of all
    # null/zero-filled observations — the #270 sibling the row-count test misses).
    all_nan = bool(
        not df.empty
        and value_col is not None
        and value_col in df.columns
        and df[value_col].isna().all()
    )
    if df.empty or all_nan:
        log.warning(
            "eia_no_usable_data",
            region=region,
            data_type=data_type,
            raw_records=len(all_records),
            reason="all_nan" if all_nan else "empty",
        )
        return _last_known_good(cache, cache_key, data_type, region, empty_cols)

    cache.set(cache_key, df, ttl=CACHE_TTL_SECONDS)
    from data.gcs_store import write_parquet

    write_parquet(df, data_type, region)
    log.info("eia_cached", region=region, data_type=data_type, rows=len(df))
    return df


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
        start = (datetime.now(UTC) - timedelta(days=90)).strftime("%Y-%m-%dT00")
    if end is None:
        end = datetime.now(UTC).strftime("%Y-%m-%dT23")

    eia_code = _get_eia_code(region)
    return _fetch_eia(
        region=region,
        data_type="demand",
        # Date-level granularity so the key doesn't change every hour.
        cache_key=f"eia_demand_{region}_{start[:10]}_{end[:10]}",
        endpoint=EIA_ENDPOINTS["demand"],
        params={
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
        },
        parser=lambda records: _parse_demand_records(records, region),
        empty_cols=["timestamp", "demand_mw", "forecast_mw", "region"],
        use_cache=use_cache,
        # A demand window whose D observations are all null/zero (coerced to NaN)
        # carries no usable signal — treat it as a miss so it can't overwrite the
        # GCS last-known-good. (#270 sibling.)
        value_col="demand_mw",
    )


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
        start = (datetime.now(UTC) - timedelta(days=90)).strftime("%Y-%m-%dT%H")
    if end is None:
        end = datetime.now(UTC).strftime("%Y-%m-%dT%H")

    eia_code = _get_eia_code(region)
    return _fetch_eia(
        region=region,
        data_type="generation",
        cache_key=f"eia_gen_{region}_{start}_{end}",
        endpoint=EIA_ENDPOINTS["fuel_type"],
        params={
            "api_key": EIA_API_KEY,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": eia_code,
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": EIA_PAGE_SIZE,
        },
        parser=lambda records: _parse_generation_records(records, region),
        empty_cols=["timestamp", "fuel_type", "generation_mw", "region"],
        use_cache=use_cache,
        # P2-08 (#273): with nulls now preserved as NaN, an all-null window
        # must route to the last-known-good chain like demand does — NOT
        # parse to a rows-present all-NaN frame that gets cached 24h and
        # written over the GCS last-known-good (verification HIGH). Safe
        # only because _parse_mw_value has no 0→NaN coercion: a legitimate
        # all-zero window parses to 0.0 and can never trip this gate.
        value_col="generation_mw",
    )


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
        start = (datetime.now(UTC) - timedelta(days=30)).strftime("%Y-%m-%dT%H")
    if end is None:
        end = datetime.now(UTC).strftime("%Y-%m-%dT%H")

    eia_code = _get_eia_code(region)
    return _fetch_eia(
        region=region,
        data_type="interchange",
        cache_key=f"eia_interchange_{region}_{start}_{end}",
        endpoint=EIA_ENDPOINTS["interchange"],
        params={
            "api_key": EIA_API_KEY,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[fromba][]": eia_code,
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": EIA_PAGE_SIZE,
        },
        parser=lambda records: _parse_interchange_records(records),
        empty_cols=["timestamp", "from_ba", "to_ba", "interchange_mw"],
        use_cache=use_cache,
        # P2-08 (#273): same all-NaN gate as generation — see the comment
        # there. Keeps a transient all-null window on the last-known-good
        # snapshot instead of poisoning cache/GCS and blanking the chip.
        value_col="interchange_mw",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _EIACircuitBreaker:
    """Process-local circuit breaker that fail-fasts EIA calls during an outage.

    The retry/backoff loop in :func:`_request_with_backoff` costs ~150s per
    failing call (``MAX_RETRIES`` x 30s + backoff). Across 51 BAs x 3 endpoints
    that overruns the scoring job's task timeout before the per-call fallbacks
    (stale cache / GCS) can serve last-known data. This breaker trips after a
    few consecutive hard failures and then short-circuits subsequent calls to
    ~0s so callers fall back immediately and the run completes within budget.
    A periodic single-attempt probe lets it close again if EIA recovers
    mid-run. State is module-level and resets every fresh (per-tick) job
    process; :meth:`reset` exists for tests and explicit run boundaries.
    """

    def __init__(self, trip_threshold: int, probe_interval: int) -> None:
        self._trip_threshold = trip_threshold
        self._probe_interval = probe_interval
        self._consecutive_failures = 0
        self._tripped = False
        self._suppressed_since_probe = 0

    @property
    def tripped(self) -> bool:
        """True while the breaker is open (calls should fail fast)."""
        return self._tripped

    def record_success(self) -> None:
        """Register a successful request; closes the breaker."""
        if self._tripped:
            log.info("eia_circuit_closed")
        self._consecutive_failures = 0
        self._tripped = False
        self._suppressed_since_probe = 0

    def record_failure(self) -> None:
        """Register a hard failure (retries exhausted); may trip the breaker."""
        self._consecutive_failures += 1
        if not self._tripped and self._consecutive_failures >= self._trip_threshold:
            self._tripped = True
            log.warning(
                "eia_circuit_tripped",
                consecutive_failures=self._consecutive_failures,
                trip_threshold=self._trip_threshold,
            )

    def allow_request(self) -> bool:
        """Whether to attempt a network call now.

        Returns True normally. When tripped, returns False (fail fast) except
        once every ``probe_interval`` suppressed calls, when it lets a single
        recovery probe through.
        """
        if not self._tripped:
            return True
        self._suppressed_since_probe += 1
        if self._suppressed_since_probe >= self._probe_interval:
            self._suppressed_since_probe = 0
            log.debug("eia_circuit_probe")
            return True
        return False

    def reset(self) -> None:
        """Reset all state (test helper / explicit run boundary)."""
        self._consecutive_failures = 0
        self._tripped = False
        self._suppressed_since_probe = 0


_circuit_breaker = _EIACircuitBreaker(EIA_CIRCUIT_TRIP_THRESHOLD, EIA_CIRCUIT_PROBE_INTERVAL)


def _paginated_fetch(endpoint: str, params: dict) -> list[dict]:
    """
    Fetch all pages from an EIA API endpoint with retry on rate limiting.

    Returns:
        List of all data records across pages.

    Raises:
        EIAIncompleteFetchError: if a page hard-fails (``_request_with_backoff``
            returns ``None``) while the server-reported ``total`` says more rows
            remain. The accumulated rows are a truncated series that must not be
            cached/persisted over last-known-good, so this signals the caller to
            fall back instead (#269 / P2-06). A *first*-page failure (nothing
            accumulated, ``total`` still 0) is not a truncation — it returns an
            empty list and takes the ordinary empty-response fallback.
    """
    all_records: list[dict] = []
    offset = 0
    total = 0
    page_failed = False

    while True:
        params["offset"] = offset
        url = f"{EIA_BASE_URL}/{endpoint}/data/"

        response = _request_with_backoff(url, params)
        if response is None:
            page_failed = True
            break

        data = response.get("response", {})
        records = data.get("data", [])
        total = int(data.get("total", 0) or 0)

        all_records.extend(records)
        log.debug("eia_page_fetched", offset=offset, records=len(records), total=total)

        offset += EIA_PAGE_SIZE
        if offset >= total or not records:
            break

    # A page hard-failing mid-pagination — while ``total`` says rows remain —
    # leaves a truncated series. Gate on ``page_failed`` (not a bare
    # ``len < total``) so an imprecise/over-counted ``total`` on a cleanly
    # completed fetch can't force a permanent false-positive fallback. (#269)
    #
    # Deliberately NOT flagged: a page that returns a valid 200 with an *empty*
    # data array while ``offset < total`` (loop then exits via the ``not records``
    # break, ``page_failed`` False). That shape is indistinguishable from the
    # benign case where EIA over-reports ``total`` and the real data simply ends
    # on a page boundary — both surface as ``records == [] while offset < total``.
    # Raising here would reintroduce false-positive fallbacks on that common
    # quirk to catch an anomalous mid-stream empty page EIA v2 does not emit under
    # contiguous pagination. See test_empty_mid_page_is_not_flagged_as_truncation.
    if page_failed and len(all_records) < total:
        log.warning(
            "eia_truncated_fetch", endpoint=endpoint, fetched=len(all_records), expected=total
        )
        raise EIAIncompleteFetchError(
            f"paginated fetch truncated: {len(all_records)}/{total} rows for {endpoint}"
        )

    return all_records


def _request_with_backoff(url: str, params: dict) -> dict | None:
    """Make an HTTP GET with exponential backoff on 429/5xx.

    Guarded by a process-local circuit breaker (:data:`_circuit_breaker`): once
    EIA has returned several consecutive hard failures in this run, calls fail
    fast (return ``None`` without touching the network) so callers fall back to
    last-known data and the job completes within its task budget instead of
    timing out. While tripped, a call that is let through is a single-attempt
    recovery probe rather than the full retry budget.
    """
    if not _circuit_breaker.allow_request():
        log.debug("eia_circuit_open_skip", url=url)
        return None

    # While tripped, a permitted call is a recovery probe: one attempt, not the
    # full retry budget, so a continued outage stays cheap.
    max_attempts = 1 if _circuit_breaker.tripped else MAX_RETRIES
    backoff = INITIAL_BACKOFF_SECONDS

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)

            if resp.status_code == 200:
                _circuit_breaker.record_success()
                return resp.json()
            elif resp.status_code == 429:
                log.warning("eia_rate_limited", attempt=attempt, backoff=backoff)
                if attempt < max_attempts:
                    time.sleep(backoff)
                    backoff *= 2
            elif resp.status_code >= 500:
                log.warning("eia_server_error", status=resp.status_code, attempt=attempt)
                if attempt < max_attempts:
                    time.sleep(backoff)
                    backoff *= 2
            else:
                # Sanitize response body to avoid leaking API keys in logs
                import re

                sanitized = re.sub(r"api_key=[^&\s\"']+", "api_key=***", resp.text[:200])
                log.error("eia_request_failed", status=resp.status_code, body=sanitized)
                # A 4xx is a request problem, not a transient outage — return
                # without tripping the breaker.
                return None

        except requests.RequestException as e:
            log.error("eia_request_exception", error=str(e), attempt=attempt)
            if attempt < max_attempts:
                time.sleep(backoff)
                backoff *= 2

    _circuit_breaker.record_failure()
    log.error("eia_max_retries_exceeded", url=url)
    return None


def _parse_demand_records(records: list[dict], region: str) -> pd.DataFrame:
    """Parse EIA demand records into a clean DataFrame.

    EIA returns ``null`` for missing observations. Preserve that as ``NaN``
    (not ``0``) so downstream preprocessing can interpolate short gaps and
    flag long ones. Also coerce literal ``0`` to ``NaN`` — a balancing
    authority serving a state's load never reads 0 MW, so zeros are always
    missing-data artifacts (commonly a null that was zero-filled upstream).
    """
    rows = []
    for r in records:
        raw = r.get("value")
        try:
            val = float(raw) if raw not in (None, "") else float("nan")
        except (TypeError, ValueError):
            val = float("nan")
        if val == 0:
            val = float("nan")
        rows.append(
            {
                "timestamp": pd.Timestamp(r["period"], tz="UTC"),
                "value": val,
                "type": r.get("type", ""),
                "region": region,
            }
        )

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


def _parse_mw_value(raw: Any) -> float:
    """Guarded EIA ``value`` conversion: null / ``""`` / unparseable → NaN.

    P2-08 (#273): the generation/interchange parsers coerced EIA nulls to
    ``0.0 MW`` — fabricating readings that deflated renewable share, filled
    the fuel-mix pivot with fake zeros, and made the interchange sparse-data
    ``dropna`` contract dead code. This mirrors the demand parser's
    null→NaN policy but WITHOUT demand's 0→NaN coercion: a true ~0 MW
    reading is legitimate here (a fuel type can genuinely produce nothing
    for an hour; a tie can genuinely sit at zero flow).
    """
    try:
        return float(raw) if raw not in (None, "") else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _parse_generation_records(records: list[dict], region: str) -> pd.DataFrame:
    """Parse EIA generation-by-fuel records (null values preserved as NaN)."""
    rows = []
    for r in records:
        rows.append(
            {
                "timestamp": pd.Timestamp(r["period"], tz="UTC"),
                "fuel_type": r.get("fueltype", r.get("type-name", "unknown")),
                "generation_mw": _parse_mw_value(r.get("value")),
                "region": region,
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("timestamp").reset_index(drop=True)


def _parse_interchange_records(records: list[dict]) -> pd.DataFrame:
    """Parse EIA interchange records (null values preserved as NaN)."""
    rows = []
    for r in records:
        rows.append(
            {
                "timestamp": pd.Timestamp(r["period"], tz="UTC"),
                "from_ba": r.get("fromba", ""),
                "to_ba": r.get("toba", ""),
                "interchange_mw": _parse_mw_value(r.get("value")),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("timestamp").reset_index(drop=True)
