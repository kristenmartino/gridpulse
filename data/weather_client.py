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

    cache_key = f"weather_{region}_past{past_days}_fc{forecast_days}"
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
    try:
        forecast_df = _fetch_forecast_endpoint(region, ARCHIVE_LAG_DAYS + 2, forecast_days)
    except requests.RequestException as e:
        return _fallback(str(e))

    if forecast_df.empty:
        return _fallback("empty_forecast_response")

    # 1b. ADR-011 (#332): NBM-composite enrichment — flag-gated, fail-open.
    #     Any failure (HTTP or composite bug) serves the base frame
    #     unchanged; the base fetch's fallback chain above is untouched.
    #     Prod jobs run in fresh containers, so the 24h SQLite cache can
    #     never serve a pre-flip frame across a flag flip (dev-only nuance).
    from config import NBM_MODEL, feature_enabled

    if feature_enabled("nbm_weather"):
        try:
            nbm_df = _fetch_forecast_endpoint(
                region, ARCHIVE_LAG_DAYS + 2, forecast_days, model=NBM_MODEL
            )
            now_ts = pd.Timestamp.now(tz="UTC").floor("h")
            forecast_df, n_overlaid, n_kept = _composite_nbm(forecast_df, nbm_df, now_ts)
            log.info(
                "weather_nbm_composited",
                region=region,
                n_overlaid=n_overlaid,
                n_base_kept=n_kept,
            )
        except Exception as e:
            log.warning("weather_nbm_failed", region=region, error=str(e))

    # 2. Archive endpoint (deep history). Enrichment only — on failure we
    #    keep the forecast-only result rather than dropping to fallback.
    from datetime import UTC, datetime, timedelta

    today = datetime.now(UTC).date()
    boundary = pd.Timestamp(today - timedelta(days=ARCHIVE_LAG_DAYS), tz="UTC") + pd.Timedelta(
        hours=23
    )
    archive_df: pd.DataFrame | None = None
    try:
        archive_df = _fetch_archive_endpoint(
            region,
            (today - timedelta(days=past_days)).isoformat(),
            (today - timedelta(days=ARCHIVE_LAG_DAYS)).isoformat(),
        )
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
