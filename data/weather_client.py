"""
Open-Meteo API client for weather data ingestion.

Fetches historical and forecast weather data for each balancing authority
centroid. No API key required for non-commercial use.

Key design decisions:
- Always request Fahrenheit/mph (CDD/HDD use 65°F baseline, sliders use mph)
- ``fetch_weather`` stitches the archive (ERA5) endpoint for deep history
  with the /forecast endpoint for recent + future (#161). The older
  single ``/forecast?past_days=92`` call silently degraded its historical
  coverage in 2026-05, so the bulk of history now comes from the archive
  endpoint, which is purpose-built for it.
- All 17 weather variables requested; the archive endpoint lacks 3 of
  them (wind_speed_80m/120m, soil_temperature_0cm) on deep history —
  those stay imputed by ``engineer_features`` (see #161).

API docs: https://open-meteo.com/en/docs
"""

import numpy as np
import pandas as pd
import requests
import structlog

from config import (
    CACHE_TTL_SECONDS,
    OPEN_METEO_BASE_URL,
    OPEN_METEO_FORECAST_DAYS,
    REGION_COORDINATES,
    WEATHER_VARIABLES,
)
from data.cache import get_cache

log = structlog.get_logger()

# Open-Meteo is generous with rate limits, but be respectful
REQUEST_TIMEOUT = 30

# ERA5 archive reanalysis lags real-time by ~5 days. Beyond that boundary
# only the /forecast endpoint has data; before it the archive has full
# coverage for 14 of the 17 variables. The 3 it lacks
# (wind_speed_80m/120m, soil_temperature_0cm) stay imputed on deep history
# by ``data.feature_engineering.engineer_features`` (#161 option A).
ARCHIVE_LAG_DAYS = 5
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _fetch_forecast_endpoint(
    region: str, past_days: int, forecast_days: int, model: str | None = None
) -> pd.DataFrame:
    """Lean GET against Open-Meteo's /forecast endpoint (recent + future).

    No cache / GCS / fallback of its own — ``fetch_weather`` owns those
    once for the combined result. Raises ``requests.RequestException`` on
    HTTP failure so the caller can run the existing fallback chain.

    ``model`` selects a specific Open-Meteo model (ADR-011: NBM) instead
    of the default ``best_match`` blend. Models with shorter horizons
    (NBM: ~11.5 days) return null rows beyond their coverage rather than
    erroring — the composite's per-value fill handles the tail.
    """
    coords = REGION_COORDINATES[region]
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": ",".join(WEATHER_VARIABLES),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
    }
    if model:
        params["models"] = model
    resp = requests.get(f"{OPEN_METEO_BASE_URL}/forecast", params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return _parse_weather_response(resp.json())


def _composite_nbm(
    base_df: pd.DataFrame, nbm_df: pd.DataFrame, now: pd.Timestamp
) -> tuple[pd.DataFrame, int, int]:
    """Overlay NBM values onto the base frame — the measured ADR-011 arm.

    Three rules, each tied to the study configuration
    (docs/WEATHER_MODEL_AB.md):

    * **Future hours only** (``ts > now``): the study conditioned future
      weather; recent-past rows stay base (and the ERA5 stitch replaces
      most of them anyway).
    * **``NBM_FORCE_FILL_VARS`` always keep base** — the measured arm was
      base-filled for radiation/DNI/diffuse/pressure/120m-wind; live NBM's
      patchy radiation is an unmeasured configuration.
    * **Any NBM null keeps base** — covers NBM's ~11.5-day horizon tail
      inside the 16-day frame and any patchy field.

    Returns ``(composited copy, n_values_overlaid, n_values_base_kept)``
    where the counts cover the future-hour region of NBM-eligible columns.
    """
    from config import NBM_FORCE_FILL_VARS

    out = base_df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)
    future = ts > now
    if not future.any() or nbm_df is None or nbm_df.empty:
        return out, 0, 0

    nbm = nbm_df.copy()
    nbm["timestamp"] = pd.to_datetime(nbm["timestamp"], utc=True)
    nbm = nbm.set_index("timestamp")

    overlaid = 0
    kept = 0
    eligible = [v for v in WEATHER_VARIABLES if v not in NBM_FORCE_FILL_VARS]
    nbm_aligned = nbm.reindex(ts[future])
    for var in eligible:
        if var not in nbm_aligned.columns:
            kept += int(future.sum())
            continue
        vals = pd.to_numeric(nbm_aligned[var], errors="coerce").to_numpy()
        mask = np.isfinite(vals)
        col = out.loc[future, var].to_numpy(dtype=float).copy()
        col[mask] = vals[mask]
        out.loc[future, var] = col
        overlaid += int(mask.sum())
        kept += int((~mask).sum())
    return out, overlaid, kept


def _fetch_archive_endpoint(region: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Lean GET against Open-Meteo's archive (ERA5) endpoint for deep history.

    No cache / fallback of its own (distinct from the public
    ``fetch_historical_weather``, which has both — reusing that here would
    pollute ``fetch_weather``'s cache-call accounting). Raises
    ``requests.RequestException`` on HTTP failure.
    """
    coords = REGION_COORDINATES[region]
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": ",".join(WEATHER_VARIABLES),
        "start_date": start_date,
        "end_date": end_date,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return _parse_weather_response(resp.json())


# ── ADR-012 (#336): multi-point coordinates + fetch ──────────
#
# Every helper below is best-effort by construction: anything unexpected
# returns None so ``fetch_weather`` runs its UNCHANGED single-point path.
# "Multi-point off or broken" must equal today's behavior — the #161
# lesson (a silent coverage collapse here took down forecasts fleet-wide).

_MULTIPOINT_COORDS: dict[str, list[list[float]]] | None = None


def _load_multipoint_coords() -> dict[str, list[list[float]]]:
    """Lazy-load the committed coordinate artifact; ``{}`` on any problem.

    Generated offline by ``scripts/generate_multipoint_coords.py`` — the
    census download and matplotlib point-in-polygon live there, keeping
    the production import path dependency-free. A missing or corrupt
    artifact degrades every BA to single-point rather than raising.
    """
    global _MULTIPOINT_COORDS
    if _MULTIPOINT_COORDS is not None:
        return _MULTIPOINT_COORDS

    import json
    from pathlib import Path

    path = Path(__file__).parent.parent / "assets" / "multipoint_coordinates.json"
    try:
        payload = json.loads(path.read_text())
        coords = payload.get("coordinates", {})
        _MULTIPOINT_COORDS = coords if isinstance(coords, dict) else {}
    except Exception as exc:
        log.warning("multipoint_coords_load_failed", error=str(exc))
        _MULTIPOINT_COORDS = {}
    return _MULTIPOINT_COORDS


def _multipoint_coords(region: str) -> list[list[float]] | None:
    """The BA's footprint cells, or None when it should stay single-point.

    Compact BAs are absent from the artifact by design (the study's own
    fallback), so ``None`` is a normal, expected answer.
    """
    from config import feature_enabled

    if not feature_enabled("multipoint_weather"):
        return None
    coords = _load_multipoint_coords().get(region)
    return coords if coords else None


def _parse_weather_response_multi(payload: object) -> list[pd.DataFrame]:
    """Parse a multi-point response into per-point frames, in submitted
    order. Open-Meteo returns a LIST for multiple coordinates and a plain
    dict for one — handle both."""
    elements = payload if isinstance(payload, list) else [payload]
    return [_parse_weather_response(el) for el in elements]


def _fetch_forecast_endpoint_multi(
    coords: list[list[float]],
    past_days: int,
    forecast_days: int,
    model: str | None = None,
) -> list[pd.DataFrame]:
    """One /forecast call covering every coordinate (not one call each —
    K separate calls would blow the free tier at 51 BAs hourly)."""
    params = {
        "latitude": ",".join(f"{lat:.4f}" for lat, _ in coords),
        "longitude": ",".join(f"{lon:.4f}" for _, lon in coords),
        "hourly": ",".join(WEATHER_VARIABLES),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
    }
    if model:
        params["models"] = model
    resp = requests.get(f"{OPEN_METEO_BASE_URL}/forecast", params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return _parse_weather_response_multi(resp.json())


def _fetch_archive_endpoint_multi(
    coords: list[list[float]], start_date: str, end_date: str
) -> list[pd.DataFrame]:
    """One archive (ERA5) call covering every coordinate."""
    params = {
        "latitude": ",".join(f"{lat:.4f}" for lat, _ in coords),
        "longitude": ",".join(f"{lon:.4f}" for _, lon in coords),
        "hourly": ",".join(WEATHER_VARIABLES),
        "start_date": start_date,
        "end_date": end_date,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return _parse_weather_response_multi(resp.json())


def _frames_ok(frames: list[pd.DataFrame] | None, expected: int, region: str, what: str) -> bool:
    """Shape guard — wrong count or an empty frame means fail open."""
    if frames is None or len(frames) != expected or any(f is None or f.empty for f in frames):
        log.warning(
            "weather_multipoint_shape_mismatch",
            region=region,
            endpoint=what,
            got=0 if frames is None else len(frames),
            expected=expected,
        )
        return False
    return True


def _try_multipoint_forecast(
    coords: list[list[float]], past_days: int, forecast_days: int, region: str
) -> list[pd.DataFrame] | None:
    """Best-effort multi-point /forecast. NEVER raises — any trouble
    returns None so the caller runs the untouched single-point path (whose
    own RequestException still drives the #161 fallback chain)."""
    try:
        frames = _fetch_forecast_endpoint_multi(coords, past_days, forecast_days)
    except Exception as e:
        log.warning("weather_multipoint_forecast_failed", region=region, error=str(e))
        return None
    return frames if _frames_ok(frames, len(coords), region, "forecast") else None


def _try_multipoint_archive(
    coords: list[list[float]], start_date: str, end_date: str, region: str
) -> pd.DataFrame | None:
    """Best-effort multi-point archive, already aggregated. None on any
    trouble so the caller retries the single-point archive — dropping
    straight to forecast-only would re-introduce the #161 loss of deep
    history."""
    from data.weather_aggregate import aggregate_weather

    try:
        frames = _fetch_archive_endpoint_multi(coords, start_date, end_date)
    except Exception as e:
        log.warning("weather_multipoint_archive_failed", region=region, error=str(e))
        return None
    if not _frames_ok(frames, len(coords), region, "archive"):
        return None
    try:
        return aggregate_weather(frames)
    except Exception as e:
        log.warning("weather_multipoint_aggregate_failed", region=region, error=str(e))
        return None


def _stitch_weather(
    archive_df: pd.DataFrame | None,
    forecast_df: pd.DataFrame,
    boundary: pd.Timestamp,
) -> pd.DataFrame:
    """Combine archive history (ts <= boundary) with forecast (ts > boundary).

    Archive (ERA5 reanalysis) is preferred for the overlap region because
    it's more accurate for past hours than the forecast model's backfill.
    Concatenates, de-dups on timestamp (keeping the archive row in any
    tie), and sorts ascending.
    """
    parts: list[pd.DataFrame] = []
    if archive_df is not None and not archive_df.empty:
        a = archive_df.copy()
        a["timestamp"] = pd.to_datetime(a["timestamp"], utc=True)
        parts.append(a[a["timestamp"] <= boundary])
    if forecast_df is not None and not forecast_df.empty:
        f = forecast_df.copy()
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
        parts.append(f[f["timestamp"] > boundary])

    if not parts:
        return pd.DataFrame(columns=["timestamp"] + WEATHER_VARIABLES)

    combined = pd.concat(parts, ignore_index=True)
    # archive rows were appended first; keep="first" prefers them on a tie
    combined = (
        combined.drop_duplicates(subset="timestamp", keep="first")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return combined


def fetch_weather(
    region: str,
    past_days: int = 92,
    forecast_days: int = OPEN_METEO_FORECAST_DAYS,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical + forecast weather data for a balancing authority centroid.

    Stitches two Open-Meteo endpoints (#161):

    * **/forecast** — recent days + the ``forecast_days`` horizon. Fetched
      first; it's the data the demand forecast actually consumes, and its
      failure drives the stale-cache → GCS → empty fallback chain.
    * **archive (ERA5)** — the deep historical window
      ``[today-past_days, today-ARCHIVE_LAG_DAYS]``. Fetched second as
      enrichment; archive failure degrades gracefully to forecast-only.

    The older single ``/forecast?past_days=92`` call silently lost most of
    its historical coverage in 2026-05 (one variable down to ~5% of rows),
    which collapsed the feature pipeline below the model threshold and
    took down forecasts for all regions — see #161. Sourcing deep history
    from the purpose-built archive endpoint restores ~full coverage for
    14 of 17 variables; the 3 the archive lacks (wind_speed_80m/120m,
    soil_temperature_0cm) stay imputed by ``engineer_features``.

    Args:
        region: Balancing authority code (e.g., "ERCOT", "FPL").
        past_days: Total historical days to include (default 92 = ~3 months).
        forecast_days: Forecast horizon days (default
            ``config.OPEN_METEO_FORECAST_DAYS`` = 16, Open-Meteo's free GFS
            horizon). See ADR-008 in PRD.md.
        use_cache: Whether to check cache first.

    Returns:
        DataFrame with columns: [timestamp] + all 17 WEATHER_VARIABLES
    """
    if region not in REGION_COORDINATES:
        raise ValueError(f"Unknown region: {region}. Valid: {list(REGION_COORDINATES.keys())}")

    # ADR-012: the multi-point intent (flag on AND this BA has footprint
    # cells) is known before any fetch, so it can key the cache. Prod runs
    # fresh containers per job so a flag flip can't serve a stale frame,
    # but dev keeps cache.db across toggles — the suffix stops a
    # single-point frame being served as multi-point (or vice versa).
    mp_coords = _multipoint_coords(region)
    cache_key = f"weather_{region}_past{past_days}_fc{forecast_days}"
    if mp_coords:
        cache_key += "_mp"
    cache = get_cache()

    if use_cache:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    coords = REGION_COORDINATES[region]
    log.info("weather_fetching", region=region, lat=coords["lat"], lon=coords["lon"])

    def _fallback(reason: str) -> pd.DataFrame:
        """Shared stale-cache → GCS → empty fallback (unchanged behavior)."""
        log.warning("weather_request_failed", region=region, reason=reason)
        stale = cache.get(cache_key, allow_stale=True)
        if stale is not None:
            return stale
        from data.gcs_store import read_parquet

        gcs_df = read_parquet("weather", region)
        if gcs_df is not None and not gcs_df.empty:
            log.info("weather_gcs_fallback", region=region, rows=len(gcs_df))
            return gcs_df
        return pd.DataFrame(columns=["timestamp"] + WEATHER_VARIABLES)

    # 1. Forecast endpoint (recent + future). Short past_days — just enough
    #    to overlap the archive boundary; deep history comes from archive.
    #
    #    ADR-012: try multi-point first. ``_try_multipoint_forecast`` never
    #    raises — on ANY trouble it returns None and the single-point path
    #    below runs exactly as it does today, including its
    #    RequestException → _fallback (stale → GCS → empty) chain. So the
    #    worst case of multi-point is one wasted request, never a worse
    #    outcome than today.
    forecast_frames: list[pd.DataFrame] | None = None
    if mp_coords:
        forecast_frames = _try_multipoint_forecast(
            mp_coords, ARCHIVE_LAG_DAYS + 2, forecast_days, region
        )
    mp_active = forecast_frames is not None

    if not mp_active:
        try:
            forecast_df = _fetch_forecast_endpoint(region, ARCHIVE_LAG_DAYS + 2, forecast_days)
        except requests.RequestException as e:
            return _fallback(str(e))

        if forecast_df.empty:
            return _fallback("empty_forecast_response")
        forecast_frames = [forecast_df]

    # 1b. ADR-011 (#332): NBM-composite enrichment — flag-gated, fail-open.
    #     Any failure (HTTP or composite bug) serves the base frame
    #     unchanged; the base fetch's fallback chain above is untouched.
    #     Prod jobs run in fresh containers, so the 24h SQLite cache can
    #     never serve a pre-flip frame across a flag flip (dev-only nuance).
    from config import NBM_MODEL, feature_enabled

    if feature_enabled("nbm_weather"):
        try:
            now_ts = pd.Timestamp.now(tz="UTC").floor("h")
            if mp_active:
                # ADR-012 × ADR-011: composite PER POINT, then aggregate.
                # Aggregate-then-overlay would be wrong — _composite_nbm's
                # "any NBM null keeps base" rule is per-cell, so a point
                # whose NBM is null at some future hour (NBM's ~11.5-day
                # CONUS horizon, or a border cell) must contribute ITS OWN
                # base to the average; overlaying after aggregation would
                # instead overwrite the hour from only the finite points.
                # It is also the only configuration ADR-011 measured.
                nbm_frames = _fetch_forecast_endpoint_multi(
                    mp_coords, ARCHIVE_LAG_DAYS + 2, forecast_days, model=NBM_MODEL
                )
                if _frames_ok(nbm_frames, len(forecast_frames), region, "nbm"):
                    composited = [
                        _composite_nbm(base, nbm, now_ts)
                        for base, nbm in zip(forecast_frames, nbm_frames, strict=True)
                    ]
                    forecast_frames = [c for c, _, _ in composited]
                    log.info(
                        "weather_nbm_composited",
                        region=region,
                        points=len(composited),
                        n_overlaid=sum(o for _, o, _ in composited),
                        n_base_kept=sum(k for _, _, k in composited),
                    )
                # shape mismatch → keep the base frames (fail open)
            else:
                nbm_df = _fetch_forecast_endpoint(
                    region, ARCHIVE_LAG_DAYS + 2, forecast_days, model=NBM_MODEL
                )
                forecast_frames[0], n_overlaid, n_kept = _composite_nbm(
                    forecast_frames[0], nbm_df, now_ts
                )
                log.info(
                    "weather_nbm_composited",
                    region=region,
                    n_overlaid=n_overlaid,
                    n_base_kept=n_kept,
                )
        except Exception as e:
            log.warning("weather_nbm_failed", region=region, error=str(e))

    # ADR-012: collapse K → 1 right after the forecast fetch, so the stitch,
    # empty-guard, cache and GCS write below are untouched. When multi-point
    # is off, aggregate_weather is never called and this is the identical
    # single frame.
    if mp_active:
        from data.weather_aggregate import aggregate_weather

        try:
            forecast_df = aggregate_weather(forecast_frames)
            log.info("weather_multipoint_aggregated", region=region, points=len(forecast_frames))
        except Exception as e:
            log.warning("weather_multipoint_aggregate_failed", region=region, error=str(e))
            forecast_df = forecast_frames[0]
    else:
        forecast_df = forecast_frames[0]

    # 2. Archive endpoint (deep history). Enrichment only — on failure we
    #    keep the forecast-only result rather than dropping to fallback.
    from datetime import UTC, datetime, timedelta

    today = datetime.now(UTC).date()
    boundary = pd.Timestamp(today - timedelta(days=ARCHIVE_LAG_DAYS), tz="UTC") + pd.Timedelta(
        hours=23
    )
    archive_start = (today - timedelta(days=past_days)).isoformat()
    archive_end = (today - timedelta(days=ARCHIVE_LAG_DAYS)).isoformat()

    archive_df: pd.DataFrame | None = None
    if mp_active:
        # Already aggregated; None on any multi-point trouble.
        archive_df = _try_multipoint_archive(mp_coords, archive_start, archive_end, region)

    if archive_df is None:
        # Not attempted, or multi-point failed. Retry SINGLE-POINT before
        # degrading: #161 was precisely about losing deep history, so a
        # multi-point hiccup must not skip straight to forecast-only.
        try:
            archive_df = _fetch_archive_endpoint(region, archive_start, archive_end)
        except requests.RequestException as e:
            log.warning("weather_archive_failed_forecast_only", region=region, error=str(e))

    # 3. Stitch. Never return empty when forecast had data.
    df = _stitch_weather(archive_df, forecast_df, boundary)
    if df.empty:
        df = forecast_df

    cache.set(cache_key, df, ttl=CACHE_TTL_SECONDS)
    from data.gcs_store import write_parquet

    write_parquet(df, "weather", region)
    log.info(
        "weather_cached",
        region=region,
        rows=len(df),
        archive_rows=0 if archive_df is None else len(archive_df),
        forecast_rows=len(forecast_df),
    )
    return df


def fetch_historical_weather(
    region: str,
    start_date: str,
    end_date: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical weather data for a specific date range.

    Uses Open-Meteo's archive endpoint for data back to 1940 (ERA5 reanalysis).

    Args:
        region: Balancing authority code.
        start_date: Start date "YYYY-MM-DD".
        end_date: End date "YYYY-MM-DD".
        use_cache: Whether to check cache first.

    Returns:
        DataFrame with columns: [timestamp] + all 17 WEATHER_VARIABLES
    """
    if region not in REGION_COORDINATES:
        raise ValueError(f"Unknown region: {region}")

    # Validate date format to prevent cache key pollution and API abuse
    from datetime import datetime as _dt

    for label, val in [("start_date", start_date), ("end_date", end_date)]:
        try:
            _dt.strptime(val, "%Y-%m-%d")
        except (ValueError, TypeError):
            raise ValueError(f"Invalid {label} format: expected YYYY-MM-DD, got {val!r}") from None

    cache_key = f"weather_hist_{region}_{start_date}_{end_date}"
    cache = get_cache()

    if use_cache:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    coords = REGION_COORDINATES[region]
    log.info("weather_fetching_historical", region=region, start=start_date, end=end_date)

    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": ",".join(WEATHER_VARIABLES),
        "start_date": start_date,
        "end_date": end_date,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
    }

    # Historical API uses different base URL
    archive_url = "https://archive-api.open-meteo.com/v1/archive"
    try:
        resp = requests.get(
            archive_url,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        log.error("weather_historical_request_failed", region=region, error=str(e))
        stale = cache.get(cache_key, allow_stale=True)
        if stale is not None:
            return stale
        return pd.DataFrame(columns=["timestamp"] + WEATHER_VARIABLES)

    df = _parse_weather_response(data)
    # Historical data is stable — cache aggressively (24h)
    cache.set(cache_key, df, ttl=86400)
    log.info("weather_historical_cached", region=region, rows=len(df))
    return df


def _parse_weather_response(data: dict) -> pd.DataFrame:
    """
    Parse Open-Meteo JSON response into a DataFrame.

    Open-Meteo returns:
    {
        "hourly": {
            "time": ["2025-01-01T00:00", ...],
            "temperature_2m": [45.2, ...],
            ...
        }
    }
    """
    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame(columns=["timestamp"] + WEATHER_VARIABLES)

    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Ensure all expected columns exist (fill missing with NaN)
    for var in WEATHER_VARIABLES:
        if var not in df.columns:
            df[var] = None

    # Reorder columns
    cols = ["timestamp"] + [v for v in WEATHER_VARIABLES if v in df.columns]
    df = df[cols]

    return df.sort_values("timestamp").reset_index(drop=True)
