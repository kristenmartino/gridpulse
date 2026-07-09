"""Multi-year weather-normal artifact for the days-17-30 forecast tail (#283).

Past Open-Meteo's ~16-day forecast horizon there is no real weather signal, so
the demand model must be driven by a seasonal climatology. #281 shipped a
recent-28-day workaround; the #283 Phase-0 backtest showed a **weather-normal**
— a per-``(day_of_year, hour)`` average over many prior years — is a materially
better estimate of days-17-30 weather (esp. at seasonal turns), because it is
calendar-correct rather than "the next month looks like the last month".

This module builds that artifact (training-job path) and persists it so the
scoring job can drive XGBoost's tail off a normal weather year — the textbook
Medium-Term Load Forecasting method. The scoring wiring is #283 Phase 2.

Design notes:
- **Derived features are averaged directly** (CDD/HDD/wind_power/solar_cf via
  ``engineer_exogenous_features``), not derived-from-mean-temperature. Because
  CDD/HDD are convex (``max(0, T-65)``), ``E[CDD] >= CDD(E[T])`` — averaging the
  derived value avoids a systematic shoulder-season underestimate.
- **Mean** (not median) is the aggregator, to keep that Jensen benefit (median
  commutes through the monotonic CDD transform and would lose it).
- A **trailing ~10-year** window (``WEATHER_NORMAL_YEARS``), not the full ERA5
  archive — decades of warming would re-introduce a cold bias.
- A circular **±7-day day-of-year smoothing** turns ~10 raw samples/cell into a
  stable normal without blurring the seasonal shape.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pandas as pd
import structlog

log = structlog.get_logger()

# Weather + weather-derived columns that get a (doy, hour) normal. Time/calendar
# deterministic features (hour_sin, dow_*, is_holiday, temp_x_hour) are NOT here
# — they are recomputed per-timestamp when the forecast frame is built, so a
# normal of them would be redundant and, for interactions, wrong.
NORMAL_FEATURE_COLS = [
    # raw ERA5 weather
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_speed_80m",
    "shortwave_radiation",
    "cloud_cover",
    "precipitation",
    "surface_pressure",
    # derived (averaged directly — see module docstring on Jensen)
    "cooling_degree_days",
    "heating_degree_days",
    "wind_power_estimate",
    "solar_capacity_factor",
    "temperature_deviation",
]

# ERA5 reanalysis lags real-time by ~5 days; end the window a touch before that.
_ARCHIVE_LAG_DAYS = 6
_DOY_SMOOTH_HALF_WINDOW = 7  # ±7 days circular smoothing over day-of-year
_MIN_YEARS_FRACTION = 0.6  # require ≥60% of the requested window to trust a build
WEATHER_NORMAL_DATA_TYPE = "weather_normals"  # GCS parquet key + Redis suffix


def _smooth_doy(pivot: pd.DataFrame) -> pd.DataFrame:
    """Circular ±``_DOY_SMOOTH_HALF_WINDOW``-day rolling mean over the day-of-year
    index (rows = doy 1..366, cols = hour). Wraps Dec↔Jan so late-December and
    early-January smooth into each other."""
    n = len(pivot)
    w = 2 * _DOY_SMOOTH_HALF_WINDOW + 1
    if n <= w:
        return pivot
    # Pad head/tail with the opposite end for a circular window, roll, then trim.
    padded = pd.concat(
        [pivot.iloc[-_DOY_SMOOTH_HALF_WINDOW:], pivot, pivot.iloc[:_DOY_SMOOTH_HALF_WINDOW]]
    )
    rolled = padded.rolling(window=w, center=True, min_periods=1).mean()
    return rolled.iloc[_DOY_SMOOTH_HALF_WINDOW : _DOY_SMOOTH_HALF_WINDOW + n]


def build_weather_normal(
    region: str,
    years: int | None = None,
    asof: datetime | None = None,
) -> pd.DataFrame | None:
    """Build the per-``(day_of_year, hour)`` weather normal for ``region``.

    Fetches ~``years`` of ERA5 archive weather (via the cached
    :func:`data.weather_client.fetch_historical_weather`), computes the derived
    weather features, and averages each feature by ``(day_of_year, hour)`` with a
    circular day-of-year smoothing.

    Returns a DataFrame with columns ``[doy, hour, <NORMAL_FEATURE_COLS...>]``
    (≤ 366×24 rows), or ``None`` when history is unavailable/too thin.
    """
    from config import WEATHER_NORMAL_YEARS
    from data.feature_engineering import engineer_exogenous_features
    from data.weather_client import fetch_historical_weather

    years = years or WEATHER_NORMAL_YEARS
    asof = asof or datetime.now(UTC)
    end = asof - timedelta(days=_ARCHIVE_LAG_DAYS)
    # +15d so the ±7d doy smoothing has full coverage at the window edges.
    start = end - timedelta(days=365 * years + 15)

    try:
        raw = fetch_historical_weather(region, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    except Exception as e:  # noqa: BLE001
        log.warning("weather_normal_fetch_failed", region=region, error=str(e))
        return None
    if raw is None or raw.empty or "temperature_2m" not in raw.columns:
        log.warning("weather_normal_no_data", region=region)
        return None

    raw = raw.dropna(subset=["temperature_2m"]).copy()
    span_days = (raw["timestamp"].max() - raw["timestamp"].min()) / pd.Timedelta(days=1)
    if span_days < 365 * years * _MIN_YEARS_FRACTION:
        log.warning(
            "weather_normal_thin_history",
            region=region,
            span_days=round(span_days),
            want=365 * years,
        )
        return None

    feat = engineer_exogenous_features(raw)
    feat["doy"] = feat["timestamp"].dt.dayofyear
    feat["hour"] = feat["timestamp"].dt.hour
    cols = [c for c in NORMAL_FEATURE_COLS if c in feat.columns]

    # Mean of the derived features (Jensen) per (doy, hour), then smooth per hour
    # across day-of-year. Pivot doy×hour per feature so smoothing is vectorised.
    out: dict[str, pd.Series] = {}
    grouped = feat.groupby(["doy", "hour"])[cols].mean()
    for col in cols:
        pivot = grouped[col].unstack("hour")  # rows=doy, cols=hour
        smoothed = _smooth_doy(pivot)
        out[col] = smoothed.stack()
    normal = pd.DataFrame(out).reset_index()  # columns: doy, hour, <cols>
    normal = normal.sort_values(["doy", "hour"]).reset_index(drop=True)
    log.info(
        "weather_normal_built", region=region, rows=len(normal), years=years, features=len(cols)
    )
    return normal


def _to_payload(region: str, normal: pd.DataFrame, years: int) -> dict:
    feature_cols = [c for c in normal.columns if c not in ("doy", "hour")]
    return {
        "region": region,
        "updated_at": datetime.now(UTC).isoformat(),
        "years": years,
        "features": feature_cols,
        "rows": normal.round(4).to_dict(orient="records"),
    }


def persist_weather_normal(region: str, normal: pd.DataFrame, years: int | None = None) -> None:
    """Persist the normal to Redis (fast scoring read) + GCS parquet (durable).

    Uses a long Redis TTL (``WEATHER_NORMAL_TTL_SECONDS``) so the artifact
    survives between quarterly rebuilds. GCS is the durable copy.
    """
    from config import WEATHER_NORMAL_TTL_SECONDS, WEATHER_NORMAL_YEARS
    from data.gcs_store import write_parquet
    from data.redis_client import persist, redis_key

    years = years or WEATHER_NORMAL_YEARS
    payload = _to_payload(region, normal, years)
    persist(
        redis_key(f"{WEATHER_NORMAL_DATA_TYPE}:{region}"),
        payload,
        ttl=WEATHER_NORMAL_TTL_SECONDS,
    )
    write_parquet(normal, WEATHER_NORMAL_DATA_TYPE, region)
    log.info("weather_normal_persisted", region=region, rows=len(normal))


def load_weather_normal(region: str) -> pd.DataFrame | None:
    """Load the normal for ``region`` — Redis-first, GCS parquet fallback.

    Returns a DataFrame ``[doy, hour, <features...>]`` or ``None``.
    """
    from data.redis_client import redis_get, redis_key

    try:
        payload = redis_get(redis_key(f"{WEATHER_NORMAL_DATA_TYPE}:{region}"))
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list) and payload["rows"]:
            return pd.DataFrame(payload["rows"])
    except Exception as e:  # noqa: BLE001
        log.debug("weather_normal_redis_miss", region=region, error=str(e))

    from data.gcs_store import read_parquet

    gcs = read_parquet(WEATHER_NORMAL_DATA_TYPE, region)
    if gcs is not None and not gcs.empty:
        return gcs
    return None


def normal_age_days(region: str) -> float | None:
    """Age (days) of the persisted normal from its Redis ``updated_at``; ``None``
    when absent/unreadable (→ treat as needing a rebuild)."""
    from data.redis_client import redis_get, redis_key

    try:
        payload = redis_get(redis_key(f"{WEATHER_NORMAL_DATA_TYPE}:{region}"))
        if isinstance(payload, dict) and payload.get("updated_at"):
            updated = datetime.fromisoformat(payload["updated_at"])
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=UTC)
            return (datetime.now(UTC) - updated).total_seconds() / 86400.0
    except Exception as e:  # noqa: BLE001
        log.debug("weather_normal_age_unknown", region=region, error=str(e))
    return None


def refresh_weather_normals(
    regions: list[str],
    min_age_days: int | None = None,
    max_rebuild: int | None = None,
    throttle_s: float = 1.0,
) -> dict:
    """Rebuild stale/missing weather normals, capped + throttled.

    The normal is expensive to build (a ~10-year multi-BA ERA5 fetch), so this is
    a quarterly job, not per-scoring-tick. Called from the daily training job: it
    skips regions whose normal is younger than ``min_age_days`` and rebuilds at
    most ``max_rebuild`` per run, so a cold-start backfill spreads over several
    days rather than hammering the archive API (which rate-limits) in one run.

    Returns a summary dict ``{built, skipped, failed}``.
    """
    from config import WEATHER_NORMAL_MAX_REBUILD_PER_RUN, WEATHER_NORMAL_REFRESH_DAYS

    min_age_days = min_age_days if min_age_days is not None else WEATHER_NORMAL_REFRESH_DAYS
    max_rebuild = max_rebuild if max_rebuild is not None else WEATHER_NORMAL_MAX_REBUILD_PER_RUN

    built, skipped, failed = [], [], []
    for region in regions:
        if len(built) >= max_rebuild:
            skipped.append(region)  # deferred to a later run (backfill spread)
            continue
        age = normal_age_days(region)
        if age is not None and age < min_age_days:
            skipped.append(region)  # still fresh
            continue
        try:
            normal = build_weather_normal(region)
            if normal is None or normal.empty:
                failed.append(region)
                continue
            persist_weather_normal(region, normal)
            built.append(region)
            time.sleep(throttle_s)  # stay under the archive burst limit
        except Exception as e:  # noqa: BLE001
            log.warning("weather_normal_refresh_failed", region=region, error=str(e))
            failed.append(region)

    summary = {"built": built, "skipped": skipped, "failed": failed}
    log.info(
        "weather_normals_refreshed",
        built=len(built),
        skipped=len(skipped),
        failed=len(failed),
    )
    return summary
