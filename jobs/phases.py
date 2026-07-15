"""
Shared phase functions for the GridPulse scheduled jobs.

Both the hourly scoring job and the daily training job need to:

1. Fetch demand + weather + generation for a region.
2. Engineer features.
3. Write actuals / weather / generation payloads to Redis for the web tier
   to read.

The scoring job additionally predicts forward-looking demand, writes
forecast / alerts / diagnostics / weather-correlation Redis entries, and
refreshes ``gridpulse:meta:last_scored``.

The training job additionally trains new model artifacts, persists them to
GCS via :mod:`models.persistence`, recomputes backtests, and refreshes
``gridpulse:meta:last_trained``.

Design:
- Every phase returns a structured result (``PhaseResult``) rather than
  raising so a single region's failure can't abort a whole job run.
- No module-level state. Both jobs should be safely invokable from a
  single container without cross-talk.
"""

from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import structlog

from config import (
    EIA_API_KEY,
    PRECOMPUTE_MAX_WORKERS,
    REGION_COORDINATES,
)

log = structlog.get_logger()

# Redis keys + TTL kept in sync with components/callbacks.py consumers.
REDIS_TTL = 86400
DEFAULT_BACKTEST_EXOG_MODE = "forecast_exog"
BACKTEST_HORIZONS = (24, 168, 720)
FORECAST_HORIZON_HOURS = 720

# Window (days) of *recent* history used to build the (hour, dow) climatology
# baseline for the beyond-Open-Meteo forecast horizon. A season-agnostic mean
# over the full ~90-day training window understates peak-summer demand — for a
# July forecast it dilutes in cooler April–June data (on DUK the full-window
# baseline ran 9.4°F cooler than current and halved CDD, driving ~84% of the
# residual 30-day decline, #281). The most recent weeks are the closest
# available proxy for the forecast season, so the climatology is restricted to
# them (falling back to the full history when the recent window is too thin).
CLIMATOLOGY_WINDOW_DAYS = 28
_CLIMATOLOGY_MIN_ROWS = 7 * 24  # ≥ 1 week before trusting the recent window

# Decay timescale (hours) for the #283 Phase 3 seam anomaly-blend: the current
# weather anomaly (real − normal at the Open-Meteo boundary) persists into the
# tail as exp(−Δh / τ). ~5 days ⇒ ~37% of the anomaly survives 5 days past the
# boundary, ~6% by day 30 — matching how long a weather regime typically holds
# before reverting to climatology.
_SEAM_ANOMALY_TAU_HOURS = 120.0

# #296 serve-time horizon guard: each served model series (and the ensemble)
# is checked against a band derived from this many hours of recent real
# demand; the horizons checked mirror the UI's 24h/7d/30d views. Thresholds
# live in config (LONG_HORIZON_GUARD_*).
_GUARD_RECENT_ROWS = 28 * 24
_GUARD_HORIZONS = (24, 168, 720)

# PR-E (2026-05-20) — depth of recursive autoregressive-feature inference.
# For future hours 1..RECURSIVE_AUTOREGRESSIVE_HOURS, the XGBoost predict
# loop computes ``demand_lag_*`` / ``ramp_rate`` / ``demand_roll_*`` from
# recent actuals + prior predictions (chained), matching the inference
# behavior validated by the training holdout. Past this depth the
# autoregressive features fall back to the climatology baseline built
# by ``_build_future_feature_frame``. The boundary aligns with
# ``config.OPEN_METEO_FORECAST_HOURS`` (384) so the "real signal"
# regime ends at the same day-16 mark for both weather and
# autoregressive features — see ADR-008 in PRD.md.
from config import OPEN_METEO_FORECAST_HOURS as _OM_HOURS  # noqa: E402

RECURSIVE_AUTOREGRESSIVE_HOURS = _OM_HOURS

_EIA_FUEL_MAP = {
    "SUN": "solar",
    "WND": "wind",
    "NG": "gas",
    "NUC": "nuclear",
    "COL": "coal",
    "WAT": "hydro",
    "OTH": "other",
}


# ── Result types ─────────────────────────────────────────────


@dataclass
class PhaseResult:
    """Result of a single-region phase execution."""

    region: str
    ok: bool
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class RegionData:
    """Per-region data payload shared across phases."""

    region: str
    demand_df: pd.DataFrame
    weather_df: pd.DataFrame
    featured_df: pd.DataFrame | None = None


# ── Region ordering ──────────────────────────────────────────


def ordered_regions(default_region: str | None = None) -> list[str]:
    """Return all known regions, putting ``default_region`` first when supplied."""
    all_regions = list(REGION_COORDINATES.keys())
    if default_region and default_region in all_regions:
        all_regions.remove(default_region)
        all_regions.insert(0, default_region)
    return all_regions


def _has_eia_key() -> bool:
    return bool(EIA_API_KEY) and EIA_API_KEY != "your_eia_api_key_here"


# ── Phase: data fetch ────────────────────────────────────────


def fetch_region_data(region: str) -> RegionData | None:
    """Fetch demand + weather for a region. Returns ``None`` on failure."""
    if not _has_eia_key():
        log.warning("job_fetch_skipped_no_api_key", region=region)
        return None

    from data.eia_client import fetch_demand
    from data.weather_client import fetch_weather

    try:
        demand_df = fetch_demand(region)
    except Exception as e:
        log.warning("job_fetch_demand_failed", region=region, error=str(e))
        return None

    try:
        weather_df = fetch_weather(region)
    except Exception as e:
        log.warning("job_fetch_weather_failed", region=region, error=str(e))
        return None

    if demand_df is None or weather_df is None or demand_df.empty or weather_df.empty:
        log.warning(
            "job_fetch_partial",
            region=region,
            has_demand=demand_df is not None and not demand_df.empty,
            has_weather=weather_df is not None and not weather_df.empty,
        )
        return None

    log.info(
        "job_data_fetched",
        region=region,
        demand_rows=len(demand_df),
        weather_rows=len(weather_df),
    )
    return RegionData(region=region, demand_df=demand_df, weather_df=weather_df)


def fetch_all_regions(regions: list[str], max_workers: int | None = None) -> dict[str, RegionData]:
    """Fetch data for every region in parallel."""
    workers = max_workers or PRECOMPUTE_MAX_WORKERS
    out: dict[str, RegionData] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_region_data, r): r for r in regions}
        for fut in as_completed(futures):
            region = futures[fut]
            try:
                data = fut.result()
                if data is not None:
                    out[region] = data
            except Exception as e:
                log.warning("job_fetch_error", region=region, error=str(e))
    return out


def engineer_region_features(data: RegionData) -> pd.DataFrame | None:
    """Run feature engineering and store the result on ``data``."""
    from data.feature_engineering import engineer_features
    from data.preprocessing import merge_demand_weather

    try:
        merged = merge_demand_weather(data.demand_df, data.weather_df)
        featured = engineer_features(merged).dropna(subset=["demand_mw"])
        if len(featured) < 168:
            log.warning(
                "job_insufficient_feature_rows",
                region=data.region,
                rows=len(featured),
            )
            return None
        data.featured_df = featured.reset_index(drop=True)
        return data.featured_df
    except Exception as e:
        log.warning("job_feature_engineering_failed", region=data.region, error=str(e))
        return None


# ── Phase: Redis writes (shared by both jobs) ────────────────


def _ts_list(series: Any) -> list[str]:
    return [t.isoformat() if hasattr(t, "isoformat") else str(t) for t in series]


def write_actuals_and_weather(data: RegionData) -> PhaseResult:
    """Write actuals + weather JSON payloads to Redis."""
    from data.redis_client import persist, redis_key

    region = data.region
    try:
        demand_df = data.demand_df
        weather_df = data.weather_df

        # scored_at lets the web tier MEASURE freshness from the payload's
        # own age instead of asserting "fresh" at render time (P1-3).
        scored_at = datetime.now(UTC).isoformat()
        actuals_payload = {
            "region": region,
            "scored_at": scored_at,
            "timestamps": _ts_list(demand_df["timestamp"]),
            "demand_mw": demand_df["demand_mw"].tolist(),
        }
        persist(redis_key(f"actuals:{region}"), actuals_payload, ttl=REDIS_TTL)

        weather_payload: dict[str, Any] = {
            "region": region,
            "scored_at": scored_at,
            "timestamps": _ts_list(weather_df["timestamp"]),
        }
        for col in weather_df.columns:
            if col == "timestamp":
                continue
            weather_payload[col] = weather_df[col].tolist()
        persist(redis_key(f"weather:{region}"), weather_payload, ttl=REDIS_TTL)

        return PhaseResult(
            region=region,
            ok=True,
            details={
                "demand_rows": len(demand_df),
                "weather_rows": len(weather_df),
            },
        )
    except Exception as e:
        log.warning("job_write_actuals_failed", region=region, error=str(e))
        return PhaseResult(region=region, ok=False, error=str(e))


def write_generation(region: str) -> PhaseResult:
    """Fetch generation-by-fuel for a region and write a pivoted payload to Redis."""
    from data.eia_client import fetch_generation_by_fuel
    from data.redis_client import redis_key, redis_set

    if not _has_eia_key():
        return PhaseResult(region=region, ok=False, error="no_eia_api_key")

    try:
        gen_df = fetch_generation_by_fuel(region)
        if gen_df is None or gen_df.empty:
            log.info("job_generation_empty", region=region)
            return PhaseResult(region=region, ok=False, error="empty")

        gen_df["fuel_type"] = (
            gen_df["fuel_type"].map(_EIA_FUEL_MAP).fillna(gen_df["fuel_type"].str.lower())
        )
        gen_df["timestamp"] = pd.to_datetime(gen_df["timestamp"])

        # P2-08 (#273): the parser now preserves EIA nulls as NaN instead of
        # fabricating readings, and an ALL-null window returns an honest
        # empty result below instead of serving zeros (the upstream
        # value_col gate in eia_client additionally routes that case to
        # last-known-good before it ever reaches here). KNOWN RESIDUAL,
        # deliberately unchanged in this pass: a null for ONE fuel at an
        # hour where other fuels report still reads 0 in the served series,
        # because after dropna the pivot's fillna(0) can't distinguish a
        # parsed null from a fuel-column alignment gap. Fixing that needs
        # nullable payload lists plus NaN-aware aggregation in all three
        # consumer surfaces — tracked as a #273 follow-up, not claimed here.
        gen_df = gen_df.dropna(subset=["generation_mw"])
        if gen_df.empty:
            log.info("job_generation_all_null", region=region)
            return PhaseResult(region=region, ok=False, error="empty")

        pivot = gen_df.pivot_table(
            index="timestamp",
            columns="fuel_type",
            values="generation_mw",
            aggfunc="sum",
        ).fillna(0)

        payload: dict[str, Any] = {
            "region": region,
            "timestamps": _ts_list(pivot.index),
        }
        for col in pivot.columns:
            payload[col] = pivot[col].tolist()

        total = pivot.sum(axis=1)
        ren_cols = [c for c in ("wind", "solar", "hydro") if c in pivot.columns]
        if ren_cols and total.mean() > 0:
            ren_pct = (pivot[ren_cols].sum(axis=1) / total * 100).tolist()
        else:
            ren_pct = [0.0] * len(pivot)
        payload["renewable_pct"] = ren_pct

        redis_set(redis_key(f"generation:{region}"), payload, ttl=REDIS_TTL)
        avg_ren = float(np.mean(ren_pct)) if ren_pct else 0.0
        log.info(
            "job_generation_written",
            region=region,
            rows=len(pivot),
            avg_renewable_pct=round(avg_ren, 1),
        )
        return PhaseResult(
            region=region,
            ok=True,
            details={"rows": len(pivot), "avg_renewable_pct": avg_ren},
        )
    except Exception as e:
        log.warning("job_generation_failed", region=region, error=str(e))
        return PhaseResult(region=region, ok=False, error=str(e))


# ── Phase: interchange (V3.α) ────────────────────────────────


def write_interchange(region: str) -> PhaseResult:
    """Fetch BA-to-BA hourly interchange and write a per-region snapshot to Redis.

    Output Redis key: ``gridpulse:interchange:{region}:1h``. Schema::

        {
            "region": "PJM",
            "scored_at": "<iso>",
            "latest_hour": "<iso>",
            "net_mw": -1234.5,            # signed: + export / - import
            "counterparties": [
                {"to_ba": "MISO", "mw": -1200.5},
                {"to_ba": "NYISO", "mw": -800.0},
                {"to_ba": "DUK",  "mw":  350.0},
            ],
        }

    Counterparties are the top 3 by absolute interchange in the latest
    available hour. Empty fetches (BA not in EIA-930 or sparse data)
    write a placeholder with ``net_mw=None`` so the UI renders ``"—"``
    instead of guessing.
    """
    from data.eia_client import fetch_interchange
    from data.redis_client import redis_key, redis_set

    if not _has_eia_key():
        return PhaseResult(region=region, ok=False, error="no_eia_api_key")

    try:
        flow_df = fetch_interchange(region)
    except Exception as e:
        log.warning("job_interchange_fetch_failed", region=region, error=str(e))
        return PhaseResult(region=region, ok=False, error=str(e))

    payload: dict[str, Any] = {
        "region": region,
        "scored_at": datetime.now(UTC).isoformat(),
        "latest_hour": None,
        "net_mw": None,
        "counterparties": [],
    }

    if flow_df is None or flow_df.empty:
        log.info("job_interchange_empty", region=region)
        redis_set(redis_key(f"interchange:{region}:1h"), payload, ttl=REDIS_TTL)
        return PhaseResult(region=region, ok=True, details={"net_mw": None, "rows": 0})

    flow_df = flow_df.dropna(subset=["interchange_mw"])
    if flow_df.empty:
        redis_set(redis_key(f"interchange:{region}:1h"), payload, ttl=REDIS_TTL)
        return PhaseResult(region=region, ok=True, details={"net_mw": None, "rows": 0})

    latest_ts = flow_df["timestamp"].max()
    latest = flow_df[flow_df["timestamp"] == latest_ts]
    by_counterparty = (
        latest.groupby("to_ba")["interchange_mw"].sum().sort_values(key=abs, ascending=False)
    )
    top3 = by_counterparty.head(3)
    counterparties = [
        {"to_ba": str(to_ba), "mw": round(float(mw), 2)} for to_ba, mw in top3.items()
    ]
    net_mw = round(float(by_counterparty.sum()), 2)

    payload.update(
        {
            "latest_hour": latest_ts.isoformat() if hasattr(latest_ts, "isoformat") else None,
            "net_mw": net_mw,
            "counterparties": counterparties,
        }
    )
    redis_set(redis_key(f"interchange:{region}:1h"), payload, ttl=REDIS_TTL)
    log.info(
        "job_interchange_written",
        region=region,
        net_mw=net_mw,
        n_counterparties=len(counterparties),
    )
    return PhaseResult(
        region=region,
        ok=True,
        details={"net_mw": net_mw, "n_counterparties": len(counterparties)},
    )


# ── Phase: forecast (scoring) ────────────────────────────────


def _overlay_weather_forecast(
    future_df: pd.DataFrame,
    featured: pd.DataFrame,
    weather_df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Overlay actual Open-Meteo forecast values onto a climatology-built ``future_df``.

    For future timestamps covered by ``weather_df`` (typically the next
    ~16 days from Open-Meteo's ``/forecast`` endpoint), raw weather columns
    are replaced with their forecasted values and the derived weather
    features (CDD/HDD/wind_power/solar_capacity_factor/temp_x_hour/
    temperature_deviation) are recomputed from those forecasted raw values.
    Hours beyond the forecast horizon keep their (hour, dow) climatology
    values from the existing builder. Returns a NEW DataFrame; does not
    mutate inputs.

    Why this exists (2026-05-20, PR-C of the forecast-pipeline audit):
    pre-fix, ``_build_future_feature_frame`` populated future weather
    features entirely from historical (hour, day-of-week) group means.
    The model was trained on actual weather but served with climatology —
    a train/serve gap that caused weather to barely move the demand
    forecast in production. After this overlay, the first ~384 hours
    of the forecast horizon use real Open-Meteo data; beyond that
    we still fall back to climatology because Open-Meteo's free GFS
    forecast caps at 16 days.
    """
    from config import WEATHER_VARIABLES
    from data.feature_engineering import (
        compute_cdd,
        compute_hdd,
        compute_solar_capacity_factor,
        compute_temp_hour_interaction,
        compute_temperature_deviation,
        compute_wind_power,
    )

    future_df = future_df.copy()

    if weather_df is None or weather_df.empty:
        return future_df

    wx = weather_df.copy()
    wx["timestamp"] = pd.to_datetime(wx["timestamp"], utc=True)
    wx = wx.drop_duplicates(subset="timestamp", keep="last")

    # Restrict to the raw weather columns we actually use as model features.
    raw_in_wx = [c for c in WEATHER_VARIABLES if c in wx.columns]
    if not raw_in_wx:
        return future_df

    # Index by timestamp for fast point lookup
    wx_indexed = wx.set_index("timestamp")[raw_in_wx]
    ts = pd.to_datetime(future_df["timestamp"], utc=True)

    # Coverage diagnostic — useful for confirming Open-Meteo forecast hours
    # actually align with the demand forecast horizon in production logs.
    n_covered = int(ts.isin(wx_indexed.index).sum())
    log.info(
        "future_frame_weather_forecast_coverage",
        horizon=horizon,
        forecast_covered_hours=n_covered,
        climatology_fallback_hours=horizon - n_covered,
    )

    if n_covered == 0:
        # Nothing to overlay — climatology stays as-is.
        return future_df

    # Overlay each raw weather column where forecast exists. Other rows
    # keep their climatological value from the existing builder.
    for col in raw_in_wx:
        forecast_values = ts.map(
            lambda t, c=col: (
                float(wx_indexed.loc[t, c])
                if t in wx_indexed.index and pd.notna(wx_indexed.loc[t, c])
                else np.nan
            )
        )
        if col not in future_df.columns:
            future_df[col] = np.nan
        mask = forecast_values.notna()
        future_df.loc[mask, col] = forecast_values[mask].values

    # Recompute derived weather features from the (now possibly-updated)
    # raw values. Apply to the whole frame so derived columns stay
    # internally consistent with raw columns regardless of which source
    # (forecast or climatology) provided each row.
    if "temperature_2m" in future_df.columns:
        future_df["cooling_degree_days"] = compute_cdd(future_df["temperature_2m"]).values
        future_df["heating_degree_days"] = compute_hdd(future_df["temperature_2m"]).values

        # temperature_deviation = current_temp - 720h rolling mean. The
        # rolling window must include historical context (the 30 days
        # preceding `now`) or the deviation collapses to ~0 for all
        # future rows. Concatenate hist + future, compute, take tail.
        if "temperature_2m" in featured.columns and len(featured) > 0:
            hist_temp = featured["temperature_2m"].reset_index(drop=True)
            future_temp = future_df["temperature_2m"].reset_index(drop=True)
            combined = pd.concat([hist_temp, future_temp], ignore_index=True)
            deviation = compute_temperature_deviation(combined)
            future_df["temperature_deviation"] = deviation.tail(horizon).values

    if "wind_speed_80m" in future_df.columns:
        future_df["wind_power_estimate"] = compute_wind_power(future_df["wind_speed_80m"]).values

    if "shortwave_radiation" in future_df.columns:
        future_df["solar_capacity_factor"] = compute_solar_capacity_factor(
            future_df["shortwave_radiation"]
        ).values

    if "temperature_2m" in future_df.columns and "hour_sin" in future_df.columns:
        future_df["temp_x_hour"] = compute_temp_hour_interaction(
            future_df["temperature_2m"], future_df["hour_sin"]
        ).values

    return future_df


def _overlay_weather_normal_tail(
    future_df: pd.DataFrame,
    featured: pd.DataFrame,
    weather_df: pd.DataFrame | None,
    horizon: int,
) -> pd.DataFrame:
    """#283 Phase 2: past the Open-Meteo coverage boundary, drive the forecast
    tail off a **normal weather year** instead of the recent-28d climatology.

    For the tail hours (those with no real Open-Meteo forecast), the WEATHER +
    derived feature columns are replaced with the per-BA ``(day_of_year, hour)``
    weather-normal (``data.weather_normals``); ``AUTOREGRESSIVE_DEMAND_FEATURES``
    are left on the recent-28d window so they keep anchoring the tail to *current*
    load (which is how load growth is handled without an explicit ratio). The
    stored derived normals (CDD/HDD/wind_power/solar_cf) are injected DIRECTLY —
    they were averaged at hourly resolution to avoid the Jensen underestimate of
    CDD(mean-temp) at shoulder temps — while ``temp_x_hour`` and
    ``temperature_deviation`` are recomputed from the injected normal temps
    (``temperature_deviation``'s stored normal is a seasonal slope, not a level).

    No-op — returns the input unchanged — when the ``weather_normal_tail`` flag is
    off, the region's artifact isn't built yet, or every hour is Open-Meteo-covered,
    so a flag-off run is byte-identical to the recent-28d path. The scoring job is
    the only caller; it reads the normal from GCS (via an in-process cache), not the
    web tier.
    """
    from config import feature_enabled

    if not feature_enabled("weather_normal_tail"):
        return future_df
    if featured.empty or "region" not in featured.columns:
        return future_df
    region = str(featured["region"].iloc[0])

    from data.weather_normals import NORMAL_FEATURE_COLS, load_weather_normal_cached

    normal = load_weather_normal_cached(region)
    if normal is None or normal.empty or "doy" not in normal.columns:
        return future_df  # flag on but artifact not backfilled yet → recent-28d

    future_df = future_df.copy()
    ts = pd.to_datetime(future_df["timestamp"], utc=True)

    # Tail = future hours NOT covered by the real Open-Meteo forecast (mirrors the
    # coverage the overlay used), so the normal only fills the beyond-day-16 gap.
    covered = pd.Series(False, index=future_df.index)
    if weather_df is not None and not weather_df.empty:
        wx_ts = set(pd.to_datetime(weather_df["timestamp"], utc=True))
        covered = ts.isin(wx_ts)
    tail = (~covered).to_numpy()
    if not tail.any():
        return future_df

    lut = normal.drop_duplicates(["doy", "hour"]).set_index(["doy", "hour"])
    doy = ts.dt.dayofyear.to_numpy()
    hour = ts.dt.hour.to_numpy()
    inject_cols = [
        c
        for c in NORMAL_FEATURE_COLS
        if c != "temperature_deviation" and c in lut.columns and c in future_df.columns
    ]
    for col in inject_cols:
        series = lut[col]
        vals = np.array(
            [series.get((d, h), np.nan) for d, h in zip(doy, hour, strict=False)], dtype=float
        )
        m = tail & ~np.isnan(vals)
        future_df.loc[m, col] = vals[m]

    # Seam anomaly-blend (#283 Phase 3): carry the CURRENT weather anomaly
    # (real − normal at the last covered day, per hour-of-day) into the near tail
    # with exponential decay, so (a) there's no discontinuity where the real
    # Open-Meteo forecast hands off to the normal at the ~day-16 boundary, and
    # (b) the current regime persists a few days before reverting to the normal —
    # anomaly persistence, which the Phase-0 winter-persistence finding validated.
    # Only runs when there ARE covered hours (nothing to persist otherwise), and
    # only shifts hours strictly past the boundary (decay=0 elsewhere), so covered
    # rows and pre-boundary Open-Meteo gaps keep their exact values.
    #
    # Blend as a CONVEX combination — tail = (1−w)·normal[tail] + w·real[boundary],
    # w = decay — NOT an additive anomaly. A weighted average of two physically
    # valid values (the Jensen-correct tail normal and a real covered observation)
    # stays in-bounds; an additive `normal[tail] + (real − normal[boundary])` can
    # drive convex/bounded derived features out of range (CDD<0, solar_cf>1) when
    # the boundary-day and tail-day normals sit at different seasonal levels. At
    # w→1 the tail continues from the current regime (continuity); at w→0 it is the
    # Jensen-correct normal (the Phase-2 tail), so the deep tail keeps that benefit.
    covered_idx = np.where(covered.to_numpy())[0]
    if covered_idx.size:
        last_cov = int(covered_idx.max())
        pos = np.arange(len(future_df))
        decay = np.where(pos > last_cov, np.exp(-(pos - last_cov) / _SEAM_ANOMALY_TAU_HOURS), 0.0)
        last_day = covered_idx[covered_idx > last_cov - 24]  # the last covered day
        # Circular (wind_direction) and categorical (weather_code) features can't be
        # linearly blended — leave them at the injected normal.
        blend_cols = [c for c in inject_cols if c not in ("wind_direction_10m", "weather_code")]
        for col in blend_cols:
            real_by_hour: dict[int, float] = {}
            col_now = future_df[col].to_numpy(dtype=float)  # covered=real, tail=normal
            for i in last_day:
                rv = col_now[i]
                if np.isfinite(rv):
                    real_by_hour[int(hour[i])] = float(rv)  # the current (real) value
            if not real_by_hour:
                continue
            real_vec = np.array([real_by_hour.get(int(h), np.nan) for h in hour])
            w = np.where(np.isnan(real_vec), 0.0, decay)  # no real for this hour → no blend
            future_df[col] = col_now * (1.0 - w) + np.nan_to_num(real_vec) * w

    # Recompute temp_x_hour + temperature_deviation from the injected+blended temps
    # (days 1-16 are unchanged, so their values recompute identically).
    from data.feature_engineering import (
        compute_temp_hour_interaction,
        compute_temperature_deviation,
    )

    if "temperature_2m" in future_df.columns and "hour_sin" in future_df.columns:
        future_df["temp_x_hour"] = compute_temp_hour_interaction(
            future_df["temperature_2m"], future_df["hour_sin"]
        ).values
    if (
        "temperature_2m" in future_df.columns
        and "temperature_2m" in featured.columns
        and len(featured) > 0
    ):
        combined = pd.concat(
            [
                featured["temperature_2m"].reset_index(drop=True),
                future_df["temperature_2m"].reset_index(drop=True),
            ],
            ignore_index=True,
        )
        future_df["temperature_deviation"] = (
            compute_temperature_deviation(combined).tail(horizon).values
        )

    log.info("weather_normal_tail_applied", region=region, tail_hours=int(tail.sum()))
    return future_df


def _resolve_forecast_start(
    featured: pd.DataFrame,
    demand_df: pd.DataFrame,
) -> pd.Timestamp:
    """Pick the timestamp for hour 0 of the forecast.

    Used by ``predict_and_write_forecast`` to close the EIA-publishing-lag
    gap (#129) on the Forecast tab chart. Returns
    ``last_real_demand_hour + 1h`` whenever a real-demand hour can be
    identified — that anchor matches where the actuals trace ends, so
    the forecast trace picks up immediately after it without a visible
    multi-hour gap.

    "Real demand" = non-NaN AND strictly positive. EIA-930 publishes
    null for not-yet-available observations (which ``eia_client``
    preserves as NaN). Literal zero is coerced to NaN upstream, but
    we filter ``> 0`` defensively in case any zero-demand row slips
    through (a balancing authority cannot have truly zero load).

    Fallback chain when real demand can't be identified (degenerate
    cases — empty demand_df, all-NaN demand, brand-new region during
    first scoring tick):

    1. Last real demand from ``demand_df`` (the desired anchor) — ``+ 1h``
    2. Last timestamp in ``featured`` (pre-fix behavior, may leave a gap)
    3. ``demand_df.timestamp.max() + 1h`` as the last-resort floor

    Args:
        featured: Engineered DataFrame (post-merge, post-dropna).
        demand_df: Raw EIA demand DataFrame from ``data.demand_df``.

    Returns:
        Forecast-start timestamp (timezone-aware UTC).
    """
    last_featured_ts = featured["timestamp"].max()

    if demand_df is None or demand_df.empty:
        return last_featured_ts + pd.Timedelta(hours=1)

    # Filter to real demand readings — must be non-NaN AND strictly
    # positive. A balancing authority cannot have zero load; any zero
    # is a missing-data artifact.
    mask = demand_df["demand_mw"].notna() & (demand_df["demand_mw"] > 0)
    real_demand = demand_df.loc[mask, "timestamp"]
    if real_demand.empty:
        return last_featured_ts + pd.Timedelta(hours=1)

    last_real_demand = real_demand.max()

    # Cap at last_featured so we don't generate a forecast row for
    # which we can't compute autoregressive lag context. This means
    # ``last_real > last_featured`` (theoretically possible if
    # feature-engineering drops rows for reasons unrelated to demand
    # NaN-ness) falls back to last_featured + 1h.
    anchor = min(last_real_demand, last_featured_ts)
    return anchor + pd.Timedelta(hours=1)


def _build_future_feature_frame(
    featured: pd.DataFrame,
    horizon: int,
    weather_df: pd.DataFrame | None = None,
    start_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build a feature frame for the forecast horizon.

    Two-stage build:

    1. **Climatology baseline (always)** — every non-time feature column
       in ``featured`` is filled from per-(hour, dow) historical group
       means. This is the existing behavior; it produces a usable
       feature frame even when no weather forecast is available.
    2. **Weather-forecast overlay (when ``weather_df`` is provided)** —
       for future timestamps covered by ``weather_df`` (typically the
       next ~16 days from Open-Meteo), raw weather columns are
       overwritten with actual forecast values and derived weather
       features are recomputed. Hours beyond the forecast horizon keep
       their climatology values.

    Args:
        featured: Engineered historical DataFrame (post-merge,
            post-feature-engineering). Drives the climatology baseline
            and provides historical temperature for the rolling
            ``temperature_deviation`` window.
        horizon: Number of future hours to build.
        weather_df: Optional raw weather DataFrame (from
            ``data.weather_client.fetch_weather``) covering both the
            historical and forecast periods. Only the forecast portion
            (timestamps after the forecast start) is used. When
            ``None``, the function falls back to the pre-PR-C
            climatology-only behavior.
        start_ts: Optional explicit timestamp for hour 0 of the forecast.
            When ``None``, defaults to ``featured["timestamp"].max() +
            1h`` (the existing behavior). Passed explicitly by
            ``predict_and_write_forecast`` to anchor the forecast at
            ``last_real_demand_hour + 1h`` instead of
            ``featured.timestamp.max() + 1h`` — closes the 1-4h gap
            on the Forecast tab between the last EIA-published actual
            and the start of the forecast trace when EIA's publishing
            lag is non-zero. See #129.

    Note on autoregressive features: the climatology values placed here
    by this function are used **only as the long-horizon fallback** —
    XGBoost prediction overrides them per-row for the first
    ``RECURSIVE_AUTOREGRESSIVE_HOURS`` (384) via
    ``_predict_xgboost_with_recursive_autoregressive`` (PR-E). Beyond
    that hour the climatology values produced here are what the model
    actually sees. ARIMA and Prophet don't use these columns.
    """
    if start_ts is None:
        start_ts = featured["timestamp"].max() + pd.Timedelta(hours=1)
    future_timestamps = pd.date_range(
        start=start_ts,
        periods=horizon,
        freq="h",
    )
    future_df = pd.DataFrame({"timestamp": future_timestamps})
    future_df["hour"] = future_df["timestamp"].dt.hour
    future_df["day_of_week"] = future_df["timestamp"].dt.dayofweek
    future_df["month"] = future_df["timestamp"].dt.month
    future_df["day_of_year"] = future_df["timestamp"].dt.dayofyear
    future_df["hour_sin"] = np.sin(2 * np.pi * future_df["hour"] / 24)
    future_df["hour_cos"] = np.cos(2 * np.pi * future_df["hour"] / 24)
    future_df["dow_sin"] = np.sin(2 * np.pi * future_df["day_of_week"] / 7)
    future_df["dow_cos"] = np.cos(2 * np.pi * future_df["day_of_week"] / 7)
    future_df["is_weekend"] = (future_df["day_of_week"] >= 5).astype(int)
    # P2-14 (#273): is_holiday is calendar-derivable — compute it directly
    # from the future timestamps. It previously fell through to the
    # (hour, dow) group-mean imputer below, which (a) never set 1 for real
    # holidays inside the horizon and (b) smeared any holiday in the recent
    # 28d window onto every future week at that (hour, dow) as a fractional
    # value (~0.25 with 4 samples per key). Because the column now exists
    # before the imputer builds its column set, it is skipped there
    # automatically.
    from data.feature_engineering import compute_holiday_flag

    future_df["is_holiday"] = compute_holiday_flag(future_df["timestamp"]).to_numpy()

    feature_cols = [c for c in featured.columns if c not in ("timestamp", "demand_mw", "region")]

    # Restrict the climatology baseline to a recent trailing window so it tracks
    # the forecast season instead of regressing toward the (cooler) annual mean
    # of the full training window (#281). Fall back to the full history when the
    # recent slice is too thin to form stable (hour, dow) group means.
    hist = featured.copy()
    if "timestamp" in hist.columns and len(hist):
        cutoff = hist["timestamp"].max() - pd.Timedelta(days=CLIMATOLOGY_WINDOW_DAYS)
        recent = hist[hist["timestamp"] >= cutoff]
        if len(recent) >= _CLIMATOLOGY_MIN_ROWS:
            hist = recent.copy()
    hist["_hour"] = hist["timestamp"].dt.hour
    hist["_dow"] = hist["timestamp"].dt.dayofweek

    non_time_cols = [c for c in feature_cols if c not in future_df.columns]
    numeric_cols = [c for c in non_time_cols if c in hist.columns]
    if numeric_cols:
        group_means = hist.groupby(["_hour", "_dow"])[numeric_cols].mean()
        future_hour = future_df["timestamp"].dt.hour
        future_dow = future_df["timestamp"].dt.dayofweek
        last_row = featured.iloc[-1]
        for col in numeric_cols:
            values = np.empty(horizon, dtype=float)
            for i in range(horizon):
                key = (future_hour.iloc[i], future_dow.iloc[i])
                if key in group_means.index:
                    values[i] = group_means.loc[key, col]
                else:
                    values[i] = float(last_row[col]) if col in last_row.index else 0.0
            future_df[col] = values

    for col in feature_cols:
        if col not in future_df.columns:
            future_df[col] = 0

    # Overlay actual Open-Meteo forecast where available. For hours
    # beyond the forecast horizon (~day 16+), climatology stays.
    if weather_df is not None and not weather_df.empty:
        future_df = _overlay_weather_forecast(future_df, featured, weather_df, horizon)

    # #283 Phase 2: past the Open-Meteo boundary, swap the recent-28d climatology
    # weather for the (day_of_year, hour) weather-normal. No-op when the flag is
    # off / the artifact isn't built, so a flag-off run is byte-identical to today.
    future_df = _overlay_weather_normal_tail(future_df, featured, weather_df, horizon)

    return future_df


def _predict_xgboost_with_recursive_autoregressive(
    model: Any,
    featured: pd.DataFrame,
    future_df: pd.DataFrame,
    horizon: int,
    recursive_hours: int = RECURSIVE_AUTOREGRESSIVE_HOURS,
) -> np.ndarray:
    """XGBoost prediction with recursive autoregressive features for hours 1..N.

    For the first ``recursive_hours`` of the forecast (default 384, aligned
    with Open-Meteo's forecast horizon per ADR-008), the autoregressive
    features ``demand_lag_*`` / ``ramp_rate`` / ``demand_roll_*`` are
    computed via ``compute_autoregressive_snapshot`` from a growing
    demand history — initial seed is recent actuals from ``featured``,
    each predicted hour appends its prediction. This shares the exact
    recursive protocol (``data.feature_engineering.recursive_autoregressive_forecast``)
    used to score the persisted XGBoost holdout since #195, so the
    published holdout MAPE is measured the way production forecasts.

    Past hour ``recursive_hours``, the climatology-shaped autoregressive
    features already present in ``future_df`` (built by
    ``_build_future_feature_frame``) are used as-is. The vectorized
    XGBoost predict over the tail of ``future_df`` produces the
    long-horizon predictions in one call.

    Returns a 1D array of length ``horizon`` (or shorter if the model
    predicts fewer rows due to a column mismatch).
    """
    from data.feature_engineering import recursive_autoregressive_forecast
    from models.xgboost_model import predict_xgboost

    n_recursive = min(recursive_hours, horizon)

    # Recursive zone: chain predictions hour by hour, seeded from recent
    # actuals, via the shared helper that is the single source of truth for
    # both production scoring and holdout evaluation (#195/#186). The helper
    # filters the seed to real demand readings (non-NaN, > 0) — a single zero
    # in the history poisons the next 168 rolling-window features (#129).
    recursive_preds = recursive_autoregressive_forecast(
        model,
        featured["demand_mw"].tolist(),
        future_df.iloc[:n_recursive],
        predict_xgboost,
    )

    if horizon <= n_recursive:
        return recursive_preds

    # Climatology zone (hours N+1 to horizon): vectorized predict on the
    # tail of future_df, which already has climatology-shaped autoregressive
    # features from ``_build_future_feature_frame``. Weather features here
    # are also climatology (per ADR-008), so both signals degrade together.
    clim_df = future_df.iloc[n_recursive:horizon].copy()
    clim_preds = np.asarray(predict_xgboost(model, clim_df), dtype=float)
    return np.concatenate([np.asarray(recursive_preds, dtype=float), clim_preds])


def _gap_forward_frame(
    featured: pd.DataFrame,
    future_df: pd.DataFrame,
    anchor_end: Any | None,
    start_ts: Any,
) -> pd.DataFrame:
    """Build a feature frame spanning ``(anchor_end, start_ts) + forward``.

    Prophet/SARIMAX are pickled at daily training; the hourly scoring tick runs
    later, so their forecast origin (``anchor_end`` = the model's training end)
    precedes ``start_ts`` (= ``forecast_start``) by the train→score gap. To make
    the anchored predict return real values for the horizon, hand it the gap
    hours' REAL weather (from ``featured``) followed by the forward frame,
    rather than forward-filling/padding across the gap (#194). Returns
    ``future_df`` unchanged when there's no known anchor or no gap rows.
    """
    if anchor_end is None:
        return future_df
    ts = featured["timestamp"]
    anchor = pd.Timestamp(anchor_end)
    start = pd.Timestamp(start_ts)
    tz = ts.dt.tz
    if tz is not None:
        anchor = anchor.tz_localize(tz) if anchor.tz is None else anchor.tz_convert(tz)
        start = start.tz_localize(tz) if start.tz is None else start.tz_convert(tz)
    else:
        anchor = anchor.tz_localize(None) if anchor.tz is not None else anchor
        start = start.tz_localize(None) if start.tz is not None else start
    gap = featured[(ts > anchor) & (ts < start)]
    if gap.empty:
        return future_df
    cols = [c for c in future_df.columns if c in gap.columns]
    return pd.concat([gap[cols], future_df[cols]], ignore_index=True)


def _gap_actual_demand(featured: pd.DataFrame, anchor_end: Any, start_ts: Any) -> np.ndarray | None:
    """Real demand observed across ``(anchor_end, start_ts)`` — the actuals a
    daily-trained SARIMAX hasn't seen at hourly-scoring time (#226). Returns the
    LEADING contiguous run of non-NaN demand (so it can be appended to advance
    the frozen Kalman state), aligned hour-for-hour with ``_gap_forward_frame``'s
    leading rows. Returns ``None`` when there's no anchor, no gap, or no demand
    column. Trailing NaNs (EIA publish lag, #129) end the run."""
    if anchor_end is None or "demand_mw" not in featured.columns:
        return None
    ts = featured["timestamp"]
    anchor = pd.Timestamp(anchor_end)
    start = pd.Timestamp(start_ts)
    tz = ts.dt.tz
    if tz is not None:
        anchor = anchor.tz_localize(tz) if anchor.tz is None else anchor.tz_convert(tz)
        start = start.tz_localize(tz) if start.tz is None else start.tz_convert(tz)
    else:
        anchor = anchor.tz_localize(None) if anchor.tz is not None else anchor
        start = start.tz_localize(None) if start.tz is not None else start
    gap = featured[(ts > anchor) & (ts < start)]
    if gap.empty:
        return None
    dem = gap["demand_mw"].to_numpy(dtype=float)
    n = 0
    while n < dem.size and np.isfinite(dem[n]):
        n += 1
    return dem[:n] if n > 0 else None


def _predict_one(
    model_name: str,
    model: Any,
    featured: pd.DataFrame,
    future_df: pd.DataFrame,
    horizon: int,
    start_ts: Any | None = None,
) -> np.ndarray | None:
    """Dispatch a single model to its predict function and return point forecasts.

    All three models return the ``horizon``-long window whose first hour is
    ``start_ts`` (= ``forecast_start``), so the caller can write predictions
    positionally against ``future_ts`` (#194). XGBoost is row-feature based over
    ``future_df`` and is already anchored there. Prophet/SARIMAX forecast from
    their frozen training end, so we pass ``start_ts`` (and a gap-spanning
    feature frame) and use their returned, re-anchored window.

    XGBoost runs through ``_predict_xgboost_with_recursive_autoregressive``
    (PR-E, 2026-05-20) which uses recursive autoregressive features for
    the first ``RECURSIVE_AUTOREGRESSIVE_HOURS`` (384) hours of the
    horizon, falling back to climatology beyond.

    Returns ``None`` on a per-model failure so the caller can degrade gracefully
    (other models in the dispatch dict still get their predictions written).
    """
    try:
        if model_name == "xgboost":
            return _predict_xgboost_with_recursive_autoregressive(
                model, featured, future_df, horizon
            )
        if model_name == "prophet":
            from models.prophet_model import predict_prophet

            if start_ts is not None:
                hist_end = pd.Timestamp(model.history["ds"].max())
                frame = _gap_forward_frame(featured, future_df, hist_end, start_ts)
                result = predict_prophet(model, frame, periods=horizon, start_ts=start_ts)
            else:
                result = predict_prophet(model, featured, periods=horizon)
            preds = result.get("forecast")
            return np.asarray(preds, dtype=float) if preds is not None else None
        if model_name == "arima":
            from models.arima_model import predict_arima

            # SARIMAX takes the future-feature frame as a DataFrame; it extracts
            # its own exog columns (ARIMA_EXOG_COLS) via _get_exog internally.
            if start_ts is not None:
                train_end = model.get("train_end") if isinstance(model, dict) else None
                frame = _gap_forward_frame(featured, future_df, train_end, start_ts)
                # #226: advance the frozen Kalman state through the gap actuals
                # so the 1h-ahead origin is the last real value, not train_end.
                gap_actuals = _gap_actual_demand(featured, train_end, start_ts)
                res = predict_arima(
                    model, frame, periods=horizon, start_ts=start_ts, gap_actuals=gap_actuals
                )
            else:
                res = predict_arima(model, future_df, periods=horizon)
            preds = res["forecast"] if isinstance(res, dict) else res
            arr = np.asarray(preds, dtype=float)
            # train_arima returns NaN-filled forecast on failure; treat that as
            # a per-model failure so the row layer skips ARIMA cleanly.
            if arr.size == 0 or not np.isfinite(arr).all():
                return None
            return arr
    except Exception as exc:  # pragma: no cover — defensive; per-model isolation
        log.warning(
            "scoring_predict_failed",
            model=model_name,
            error=str(exc),
        )
        return None
    return None


def _horizon_guard_for_series(
    series: np.ndarray,
    recent_demand: np.ndarray,
) -> dict[str, Any] | None:
    """Check a served forecast series at each UI horizon slice (#296).

    Runs ``models.evaluation.check_long_horizon_sanity`` on the 24h/168h/
    720h prefixes of ``series`` against recent real demand. The drift
    check inside the checker only engages on ≥15-day slices, so a
    legitimate weather swing across a 24h/168h view is never flagged as
    drift — short slices are judged on the band alone.

    Returns:
        ``None`` when every checkable horizon passes; otherwise a dict
        ``{"max_ok_horizon": <largest passing horizon, 0 if none>,
        "flagged_horizon": <first failing horizon>, "reason": <str>}``
        for the payload's ``horizon_guard`` map.
    """
    from models.evaluation import check_long_horizon_sanity

    passing: list[int] = []
    first_reason: str | None = None
    first_failed: int | None = None
    for h in _GUARD_HORIZONS:
        if len(series) < h:
            continue
        reason = check_long_horizon_sanity(series[:h], recent_demand)
        if reason is None:
            passing.append(h)
        elif first_reason is None:
            first_reason, first_failed = reason, h
    if first_reason is None:
        return None
    return {
        "max_ok_horizon": max(passing, default=0),
        "flagged_horizon": first_failed,
        "reason": first_reason,
    }


def predict_and_write_forecast(
    data: RegionData,
    models: dict[str, Any] | None,
    model_mapes: dict[str, float | None] | None = None,
    model_metrics: dict[str, dict[str, float]] | None = None,
) -> PhaseResult:
    """Run all loaded forward forecasters and write ``gridpulse:forecast:{region}:1h``.

    Each model in ``models`` (e.g. ``{"xgboost": <m>, "prophet": <m>,
    "arima": <m>}``) is dispatched through ``_predict_one``. The per-row
    Redis payload carries every model that produced a finite prediction
    under its name (``row["xgboost"]`` / ``row["prophet"]`` / ``row["arima"]``)
    plus a ``predicted_demand_mw`` key set to the primary forecast (XGBoost
    when available, else first successful model).

    Stage 3 of plans/scoring-job-multi-model.md adds a weighted ``ensemble``
    key to each row when at least 2 models produced finite predictions.
    Weights come from inverse MAPE (``compute_ensemble_weights``) over
    ``model_mapes``; missing MAPE values fall back to equal weighting.

    #131 (2026-05-20): ``model_metrics`` rides along on the payload's
    top-level so the web tier can read training-time holdout metrics
    from Redis without needing meta.json files on its container disk
    (which it doesn't have — those live only on this Job container).
    See ``models.model_service.get_model_metrics`` Layer 0.

    Args:
        data: Per-region payload with ``featured_df`` populated.
        models: Mapping of model name → loaded model object.
        model_mapes: Optional mapping of model name → recent MAPE (%). Drives
            ensemble weighting when present.
        model_metrics: Optional mapping of model name → full holdout dict
            ``{mape, rmse, mae, r2}`` for that model, sourced from each
            model's ``meta.extra["holdout_metrics"]`` plus the ensemble
            row from ``xgb_meta.extra["ensemble_holdout_metrics"]``.
            Persisted as the ``model_metrics`` field on the Redis payload.
    """
    from data.redis_client import persist, redis_key
    from models.ensemble import compute_ensemble_weights, ensemble_combine

    region = data.region
    if not models:
        return PhaseResult(region=region, ok=False, error="no_models")
    if data.featured_df is None:
        return PhaseResult(region=region, ok=False, error="no_features")

    try:
        featured = data.featured_df

        # Anchor the forecast at ``last_real_demand_hour + 1h`` instead
        # of ``featured.timestamp.max() + 1h`` (#129, 2026-05-21). When
        # EIA's publishing lag is non-zero, ``featured`` can extend past
        # the last hour with real demand — either via trailing zero rows
        # that survive ``dropna(subset=["demand_mw"])`` or via the
        # asymmetric publishing lag between EIA (demand) and Open-Meteo
        # (weather). Anchoring on the last real demand reading closes
        # the 1-4h gap that was visible on the Forecast tab chart
        # between actuals end and forecast start. When there's no
        # publishing-lag gap, this is a no-op:
        # ``last_real_demand == featured.timestamp.max()``.
        forecast_start = _resolve_forecast_start(featured, data.demand_df)

        # Pass the raw weather DataFrame so the future-feature builder can
        # overlay actual Open-Meteo forecast values (next ~16 days) onto
        # the climatology baseline. See ``_overlay_weather_forecast``.
        future_df = _build_future_feature_frame(
            featured,
            FORECAST_HORIZON_HOURS,
            weather_df=data.weather_df,
            start_ts=forecast_start,
        )
        future_ts = future_df["timestamp"]

        # Run every model defensively — a single per-model failure can't
        # abort the phase. Preserves single-model behavior when others
        # aren't loaded (e.g. training job hasn't produced their pickle yet).
        predictions_by_model: dict[str, np.ndarray] = {}
        for name, model in models.items():
            preds = _predict_one(
                name, model, featured, future_df, FORECAST_HORIZON_HOURS, start_ts=forecast_start
            )
            if preds is None or len(preds) < FORECAST_HORIZON_HOURS:
                continue
            # Hard physical floor: demand is strictly non-negative. Prophet's
            # logistic ``floor=0`` bounds only the trend, not the additive
            # composite (trend + seasonality + regressors), so a served yhat can
            # still go negative (#281). Clip every model uniformly here — the
            # single choke point every series (incl. the ``predicted_demand_mw``
            # primary and the ensemble inputs) flows through.
            predictions_by_model[name] = np.maximum(preds[:FORECAST_HORIZON_HOURS], 0.0)

        if not predictions_by_model:
            return PhaseResult(region=region, ok=False, error="all_models_failed")

        # Stage 3: weighted ensemble of every model that succeeded.
        # Skip when only one model survived — its "ensemble" would equal
        # the model itself and just add noise to the Redis row.
        ensemble_preds: np.ndarray | None = None
        ensemble_weights: dict[str, float] | None = None
        if len(predictions_by_model) >= 2:
            mape_input: dict[str, float] = {}
            for name in predictions_by_model:
                m = (model_mapes or {}).get(name)
                if m is not None and m > 0 and np.isfinite(m):
                    mape_input[name] = float(m)
            try:
                # Only inverse-MAPE-weight when EVERY predicting model has a
                # valid MAPE. Partial coverage silently degrades the ensemble
                # to whichever model happens to have its MAPE recorded — fall
                # back to equal weights so each predicting model contributes.
                if mape_input and len(mape_input) == len(predictions_by_model):
                    ensemble_weights = compute_ensemble_weights(mape_input)
                    total = sum(ensemble_weights.values()) or 1.0
                    ensemble_weights = {k: v / total for k, v in ensemble_weights.items()}
                else:
                    n = len(predictions_by_model)
                    ensemble_weights = {name: 1.0 / n for name in predictions_by_model}
                    if mape_input:
                        log.info(
                            "scoring_ensemble_equal_weights_fallback",
                            region=region,
                            have_mape=sorted(mape_input.keys()),
                            missing_mape=sorted(set(predictions_by_model) - set(mape_input)),
                        )
                # Floored inputs make the weighted blend non-negative already;
                # clip again so the guarantee survives any future change to
                # ensemble_combine. (#281)
                ensemble_preds = np.maximum(
                    ensemble_combine(predictions_by_model, ensemble_weights), 0.0
                )
            except Exception as exc:  # pragma: no cover — defensive
                log.warning("scoring_ensemble_failed", region=region, error=str(exc))
                ensemble_preds = None
                ensemble_weights = None

        # #296: serve-time long-horizon sanity guard, uniform across every
        # served series (per-model AND ensemble). A series whose 24h/168h/720h
        # slice exits the recent-demand band gets a ``horizon_guard`` entry in
        # the payload; the Forecast tab withholds that model at flagged
        # horizons and says why, instead of drawing a degenerate line. The
        # flagged series stays in the payload rows for transparency, and it
        # still enters the ensemble blend: the fit-time d-cap removes the
        # known degeneracy at the source, inverse-MAPE³ weighting keeps a bad
        # model's contribution small, and all three affected BAs' ensembles
        # verified sane — revisit if a flagged *ensemble* ever shows up here.
        horizon_guard: dict[str, dict[str, Any]] = {}
        series_to_guard: dict[str, np.ndarray] = dict(predictions_by_model)
        if ensemble_preds is not None:
            series_to_guard["ensemble"] = ensemble_preds
        if "demand_mw" in featured.columns:
            recent_demand = featured["demand_mw"].tail(_GUARD_RECENT_ROWS).to_numpy(dtype=float)
            for name, series in series_to_guard.items():
                guard = _horizon_guard_for_series(series, recent_demand)
                if guard is not None:
                    horizon_guard[name] = guard
                    log.warning(
                        "scoring_horizon_guard_flagged",
                        region=region,
                        model=name,
                        **guard,
                    )

        # Pick the primary that powers ``predicted_demand_mw`` for back-compat.
        # XGBoost when available; otherwise the first successful model.
        primary_name = (
            "xgboost"
            if "xgboost" in predictions_by_model
            else next(iter(predictions_by_model.keys()))
        )
        primary = predictions_by_model[primary_name]

        scored_at = datetime.now(UTC).isoformat()
        fl: list[dict[str, Any]] = []
        for i in range(FORECAST_HORIZON_HOURS):
            row: dict[str, Any] = {
                "timestamp": future_ts.iloc[i].isoformat(),
                "predicted_demand_mw": float(primary[i]),
            }
            for name, preds in predictions_by_model.items():
                row[name] = float(preds[i])
            if ensemble_preds is not None:
                row["ensemble"] = float(ensemble_preds[i])
            fl.append(row)

        redis_payload: dict[str, Any] = {
            "region": region,
            "scored_at": scored_at,
            "granularity": "1h",
            "primary_model": primary_name,
            "forecasts": fl,
        }
        if ensemble_weights is not None:
            redis_payload["ensemble_weights"] = {
                k: round(v, 4) for k, v in ensemble_weights.items()
            }
        if horizon_guard:
            redis_payload["horizon_guard"] = horizon_guard

        # #131: write per-model holdout metrics into the forecast payload
        # so the web tier can read them from Redis instead of falling
        # through to ``_simulate_forecasts``-derived values via
        # ``get_model_metrics``'s layer-6 fallback. Sanitize incoming
        # values so a malformed model_metrics dict from the caller can't
        # corrupt the payload.
        if model_metrics:
            sanitized: dict[str, dict[str, float]] = {}
            for name, mvals in model_metrics.items():
                if not isinstance(mvals, dict):
                    continue
                cleaned: dict[str, float] = {}
                for field in ("mape", "rmse", "mae", "r2"):
                    val = mvals.get(field)
                    if val is None:
                        continue
                    try:
                        f = float(val)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(f):
                        cleaned[field] = f
                if cleaned:
                    sanitized[name] = cleaned
            if sanitized:
                redis_payload["model_metrics"] = sanitized

        # persist() (not redis_set) so a dropped forecast write raises → the
        # except below returns ok=False, and the region is counted as failed,
        # not scored (#268 → #267). A forecast that computed but never landed in
        # Redis must not read as a success.
        persist(
            redis_key(f"forecast:{region}:1h"),
            redis_payload,
            ttl=REDIS_TTL,
        )
        models_in_row = sorted(predictions_by_model.keys())
        if ensemble_preds is not None:
            models_in_row.append("ensemble")
        return PhaseResult(
            region=region,
            ok=True,
            details={
                "horizon": FORECAST_HORIZON_HOURS,
                "points": FORECAST_HORIZON_HOURS,
                "models": models_in_row,
            },
        )
    except Exception as e:
        log.warning("job_forecast_write_failed", region=region, error=str(e))
        return PhaseResult(region=region, ok=False, error=str(e))


# ── Phase: drift (scoring) — #121 part 1 ─────────────────────


def read_existing_forecast(region: str) -> dict[str, Any] | None:
    """Read the *current* ``gridpulse:forecast:{region}:1h`` payload from Redis.

    Called before ``predict_and_write_forecast`` overwrites the key so the
    drift phase can compare the about-to-be-stale 1-hour-ahead prediction
    against the now-known actual. Returns ``None`` for first-time scoring
    or any Redis-side error — the caller treats absence as a no-op.
    """
    from data.redis_client import redis_get, redis_key

    try:
        payload = redis_get(redis_key(f"forecast:{region}:1h"))
        if isinstance(payload, dict):
            return payload
        return None
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("drift_previous_forecast_read_failed", region=region, error=str(exc))
        return None


def write_drift_metrics(
    region: str,
    previous_forecast: dict[str, Any] | None,
    demand_df: pd.DataFrame,
) -> PhaseResult:
    """Update the rolling per-model drift window at ``gridpulse:drift:{region}``.

    #121 part 1: continuous 1-hour-ahead drift signal. At each scoring tick
    the previous tick's forecast for the *current* hour has a knowable
    actual; we compute the per-model absolute % error and append it to a
    rolling window (default 30 days). Headline 7-day and 30-day MAPEs are
    persisted alongside the underlying records so downstream UI / alerting
    has both the summary and the series.

    The phase is a no-op (``ok=True`` with ``details["skipped"]=...``) when:
    - First-ever scoring tick for the region (``previous_forecast is None``)
    - The previous forecast has no row matching any recent actual hour
    - The actuals dataframe is empty / missing required columns

    Failures here MUST NOT block the broader scoring run — drift is a
    secondary signal, not a critical path.
    """
    from data.redis_client import redis_get, redis_key, redis_set
    from models.drift import (
        build_records_from_actuals,
        compute_drift_payload,
        probe_actual_revisions,
    )

    if previous_forecast is None:
        return PhaseResult(region=region, ok=True, details={"skipped": "no_previous_forecast"})

    if demand_df is None or demand_df.empty or "demand_mw" not in demand_df.columns:
        return PhaseResult(region=region, ok=True, details={"skipped": "no_actuals"})

    try:
        # Build {timestamp_iso -> actual_mw} from the just-fetched demand
        # frame. We only care about hours where the actual is finite —
        # EIA's publishing-lag NaN rows can't anchor a drift record.
        df = demand_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.dropna(subset=["demand_mw"])
        actuals: dict[str, float] = {
            row["timestamp"].isoformat(): float(row["demand_mw"])
            for _, row in df.iterrows()
            if np.isfinite(row["demand_mw"]) and float(row["demand_mw"]) > 0
        }

        new_records = build_records_from_actuals(previous_forecast, actuals)
        if not new_records:
            return PhaseResult(
                region=region,
                ok=True,
                details={"skipped": "no_matchable_actual_hour"},
            )

        existing = redis_get(redis_key(f"drift:{region}"))
        existing_payload = existing if isinstance(existing, dict) else None

        # #304 probe (observability only): the stored records' actuals are the
        # PRELIMINARY values EIA had published when each record was created;
        # ``actuals`` here is the current fetch, where those hours have since
        # settled. If the two differ materially, the live drift MAPE measures
        # prediction-vs-preliminary rather than forecast skill — the leading
        # hypothesis for BPAT reporting 11.7% while its served forecast
        # measures 0.58% against settled data. Never allowed to fail the phase.
        try:
            revisions = probe_actual_revisions(existing_payload, actuals)
            for model_name, stats in revisions.items():
                log.info("drift_actual_revision_probe", region=region, model=model_name, **stats)
        except Exception as exc:  # pragma: no cover — probe must never break drift
            log.warning("drift_actual_revision_probe_failed", region=region, error=str(exc))

        payload = compute_drift_payload(region, existing_payload, new_records)

        redis_set(redis_key(f"drift:{region}"), payload, ttl=REDIS_TTL)

        # Compact summary for the scoring-job log line.
        models_with_records = sorted(payload["models"].keys())
        sample_model = models_with_records[0] if models_with_records else None
        sample = payload["models"][sample_model] if sample_model else {}
        log.info(
            "drift_updated",
            region=region,
            models=models_with_records,
            new_record_ts=next(iter(new_records.values())).timestamp,
            sample_rolling_smape_7d=sample.get("rolling_smape_7d"),
            sample_rolling_mape_7d=sample.get("rolling_mape_7d"),
            sample_low_actual_excluded_7d=sample.get("n_low_actual_excluded_7d"),
        )
        return PhaseResult(
            region=region,
            ok=True,
            details={
                "models": models_with_records,
                "new_records": len(new_records),
                "total_records": sum(m["n_records"] for m in payload["models"].values()),
            },
        )
    except Exception as exc:
        log.warning("drift_write_failed", region=region, error=str(exc))
        return PhaseResult(region=region, ok=False, error=str(exc))


def write_horizon_drift_metrics(
    region: str,
    forecast_payload: dict[str, Any] | None,
    demand_df: pd.DataFrame,
) -> PhaseResult:
    """Update the horizon-matched drift series at ``gridpulse:drift_horizon:{region}``.

    #227: the 1-hour signal (``write_drift_metrics``) structurally penalizes the
    multi-step models. This phase snapshots the latest forward forecast's 24h /
    48h / 72h predictions and resolves previously-snapshotted predictions whose
    target hour now has an actual, grading each horizon against its OWN
    ``MAPE_BY_HORIZON`` band. It reuses the same inputs as the 1h phase — the
    about-to-be-overwritten forecast and the just-fetched demand (the ~1h
    snapshot staleness is negligible at these horizons).

    Runs even when ``forecast_payload`` is None so pending snapshots still
    resolve. Non-critical: an error here never blocks the scoring run.
    """
    from data.redis_client import redis_get, redis_key, redis_set
    from models.drift import compute_horizon_drift_payload

    if demand_df is None or demand_df.empty or "demand_mw" not in demand_df.columns:
        return PhaseResult(region=region, ok=True, details={"skipped": "no_actuals"})

    try:
        df = demand_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.dropna(subset=["demand_mw"])
        actuals: dict[str, float] = {
            row["timestamp"].isoformat(): float(row["demand_mw"])
            for _, row in df.iterrows()
            if np.isfinite(row["demand_mw"]) and float(row["demand_mw"]) > 0
        }

        existing = redis_get(redis_key(f"drift_horizon:{region}"))
        existing_payload = existing if isinstance(existing, dict) else None
        payload = compute_horizon_drift_payload(region, existing_payload, forecast_payload, actuals)
        redis_set(redis_key(f"drift_horizon:{region}"), payload, ttl=REDIS_TTL)

        n_pending = len(payload.get("pending", []))
        n_records = sum(
            block.get("n_records", 0)
            for model in payload["models"].values()
            for block in model.values()
        )
        log.info(
            "horizon_drift_updated",
            region=region,
            models=sorted(payload["models"].keys()),
            pending=n_pending,
            total_records=n_records,
        )
        return PhaseResult(
            region=region,
            ok=True,
            details={"pending": n_pending, "total_records": n_records},
        )
    except Exception as exc:
        log.warning("horizon_drift_write_failed", region=region, error=str(exc))
        return PhaseResult(region=region, ok=False, error=str(exc))


# ── Phase: backtests (training) ──────────────────────────────


def write_backtests(data: RegionData) -> PhaseResult:
    """Run walk-forward backtests for the configured horizons and write to Redis.

    Imports ``_run_backtest_for_horizon`` lazily so the Dash callbacks module
    isn't pulled into the job container unless this phase runs.
    """
    from components.callbacks import _run_backtest_for_horizon
    from data.redis_client import redis_key, redis_set

    region = data.region
    written: list[int] = []
    for horizon in BACKTEST_HORIZONS:
        try:
            bt = _run_backtest_for_horizon(
                data.demand_df,
                data.weather_df,
                horizon,
                "xgboost",
                region,
                DEFAULT_BACKTEST_EXOG_MODE,
                bypass_redis_guard=True,
            )
            if "error" in bt:
                log.warning(
                    "job_backtest_skipped",
                    region=region,
                    horizon=horizon,
                    reason=bt["error"],
                )
                continue

            metrics = bt["metrics"]
            actual = np.asarray(bt["actual"]).tolist()
            preds = np.asarray(bt["predictions"]).tolist()
            timestamps = [pd.Timestamp(t).isoformat() for t in bt["timestamps"]]
            residuals = (np.asarray(bt["actual"]) - np.asarray(bt["predictions"])).tolist()
            redis_set(
                redis_key(f"backtest:{DEFAULT_BACKTEST_EXOG_MODE}:{region}:{horizon}"),
                {
                    "horizon": horizon,
                    "exog_mode": DEFAULT_BACKTEST_EXOG_MODE,
                    "exog_source": bt.get("exog_source", "climatology/naive baseline"),
                    "metrics": {
                        "xgboost": {
                            "mape": round(float(metrics["mape"]), 2),
                            "rmse": round(float(metrics["rmse"]), 2),
                            "mae": round(float(metrics["mae"]), 2),
                            "r2": round(float(metrics["r2"]), 4),
                        }
                    },
                    "actual": actual,
                    "predictions": {"xgboost": preds},
                    "timestamps": timestamps,
                    "residuals": residuals,
                },
                ttl=REDIS_TTL,
            )
            written.append(horizon)
        except Exception as e:
            log.warning(
                "job_backtest_error",
                region=region,
                horizon=horizon,
                error=str(e),
            )

    return PhaseResult(
        region=region,
        ok=len(written) > 0,
        details={"horizons_written": written},
    )


# ── Phase: diagnostics / weather-correlation / alerts ───────


def write_weather_correlation(data: RegionData) -> PhaseResult:
    """Write the weather-correlation payload consumed by the Weather tab."""
    from data.feature_engineering import compute_solar_capacity_factor, compute_wind_power
    from data.redis_client import redis_key, redis_set

    region = data.region
    try:
        wc_merged = data.demand_df.merge(data.weather_df, on="timestamp", how="inner")
        corr_cols = [
            c
            for c in (
                "demand_mw",
                "temperature_2m",
                "wind_speed_80m",
                "shortwave_radiation",
                "relative_humidity_2m",
                "cloud_cover",
                "surface_pressure",
            )
            if c in wc_merged.columns
        ]
        if len(corr_cols) < 2:
            return PhaseResult(region=region, ok=False, error="insufficient_weather_cols")

        corr = wc_merged[corr_cols].corr()
        importance = corr["demand_mw"].drop("demand_mw").abs().sort_values(ascending=True)

        wp_arr = (
            compute_wind_power(wc_merged["wind_speed_80m"])
            if "wind_speed_80m" in wc_merged.columns
            else []
        )
        scf_arr = (
            compute_solar_capacity_factor(wc_merged["shortwave_radiation"])
            if "shortwave_radiation" in wc_merged.columns
            else []
        )

        demand_ts = wc_merged.set_index("timestamp")["demand_mw"].resample("h").mean().dropna()
        trend = demand_ts.rolling(168, center=True).mean()
        residual = demand_ts - trend

        payload: dict[str, Any] = {
            "region": region,
            "timestamps": _ts_list(wc_merged["timestamp"]),
            "demand_mw": wc_merged["demand_mw"].tolist(),
            "wind_power": wp_arr.tolist() if hasattr(wp_arr, "tolist") else list(wp_arr),
            "solar_cf": scf_arr.tolist() if hasattr(scf_arr, "tolist") else list(scf_arr),
            "correlation_matrix": {
                "cols": corr.columns.tolist(),
                "values": corr.values.tolist(),
            },
            "importance": {
                "names": importance.index.tolist(),
                "values": importance.values.tolist(),
            },
            "seasonal": {
                "timestamps": _ts_list(demand_ts.index),
                "original": demand_ts.values.tolist(),
                "trend": [float(v) if not np.isnan(v) else None for v in trend.values],
                "residual": [float(v) if not np.isnan(v) else None for v in residual.values],
            },
        }
        for col in (
            "temperature_2m",
            "wind_speed_80m",
            "shortwave_radiation",
            "relative_humidity_2m",
            "cloud_cover",
            "surface_pressure",
        ):
            payload[col] = wc_merged[col].tolist() if col in wc_merged.columns else []

        redis_set(redis_key(f"weather-correlation:{region}"), payload, ttl=REDIS_TTL)
        return PhaseResult(region=region, ok=True, details={"rows": len(wc_merged)})
    except Exception as e:
        log.warning("job_weather_correlation_failed", region=region, error=str(e))
        return PhaseResult(region=region, ok=False, error=str(e))


def _real_feature_importance(xgb_model: dict | None) -> dict | None:
    """Top-10 feature importances from a real trained model, or None.

    Never the hardcoded ``[10, 9, 8, …]`` placeholder — an absent model must
    yield None so the SHAP panel renders an honest empty state rather than
    fabricated importances (2026-07 review, #166 sibling).
    """
    if xgb_model and isinstance(xgb_model, dict) and xgb_model.get("feature_importances"):
        sorted_feats = sorted(
            xgb_model["feature_importances"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        return {"names": [f[0] for f in sorted_feats], "values": [f[1] for f in sorted_feats]}
    return None


def write_diagnostics(data: RegionData, xgb_model: dict | None) -> PhaseResult:
    """Write the model-diagnostics payload (residuals + importance) from REAL
    walk-forward backtest results (#166 / #220).

    History: residual diagnostics need a real prediction series to compare
    actuals against. The original implementation substituted actual demand as
    the "prediction" and wrote identically-zero residuals (2026-07 review
    P2-32); the #166 interim fix wrote an honest ``unavailable`` marker — but
    it sourced from the legacy v1 ``get_forecasts``, which is strict-gated in
    production (#149) and NEVER produces a series on the job container, so the
    Models tab's four residual panels were permanently empty in prod (#220).

    Now the residual series comes from the Redis walk-forward backtest payload
    (``backtest:{exog_mode}:{region}:{horizon}``, written by the nightly
    training job): genuine holdout ``actual`` / ``predictions`` / ``residuals``
    — the same source the Forecast tab's P10–P90 band calibrates on. The 24h
    horizon is preferred (day-ahead error, the operational standard), falling
    back to 168h/720h. Provenance rides on the payload (``residual_source``)
    so the UI can disclose horizon + model. When no backtest exists yet (fresh
    deploy before the first training run) the honest ``unavailable`` marker is
    written with the TRUE self-heal reason — ``no_backtest_yet`` — rather than
    the old copy's false promise that the next scoring tick would fill it.
    """
    from data.redis_client import redis_get, redis_key, redis_set

    region = data.region
    try:
        feature_importance = _real_feature_importance(xgb_model)

        # Locate the freshest USABLE backtest payload: prefer the 24h horizon.
        # "Usable" validates the whole shape — ≥24 ALIGNED rows across actual,
        # the chosen prediction series, AND timestamps — so downstream never
        # defends against a partially-formed payload. (Verify catch on the
        # first cut: a residual series written with missing/short timestamps
        # crashed the Models-tab renderer and synthesized hour-of-day by index
        # — a malformed payload now simply loses to the next horizon.)
        bt_payload: dict | None = None
        bt_horizon: int | None = None
        pred_model: str | None = None
        n = 0
        for horizon in BACKTEST_HORIZONS:  # (24, 168, 720)
            cached = redis_get(
                redis_key(f"backtest:{DEFAULT_BACKTEST_EXOG_MODE}:{region}:{horizon}")
            )
            if not isinstance(cached, dict):
                continue
            preds_map = cached.get("predictions") or {}
            if not (isinstance(preds_map, dict) and preds_map):
                continue
            model = "xgboost" if "xgboost" in preds_map else next(iter(preds_map))
            usable = min(
                len(cached.get("actual") or []),
                len(preds_map.get(model) or []),
                len(cached.get("timestamps") or []),
            )
            if usable >= 24:
                bt_payload, bt_horizon, pred_model, n = cached, horizon, model, usable
                break

        # No backtest yet (fresh deploy, pre-first-training-run) → honest
        # unavailable marker, no fabricated residuals.
        if bt_payload is None:
            redis_set(
                redis_key(f"diagnostics:{region}"),
                {
                    "region": region,
                    "diagnostics_source": "unavailable",
                    "reason": "no_backtest_yet",
                    "metrics": {},
                    "feature_importance": feature_importance,
                },
                ttl=REDIS_TTL,
            )
            log.info("job_diagnostics_unavailable", region=region, reason="no_backtest_yet")
            return PhaseResult(region=region, ok=True, details={"diagnostics": "unavailable"})

        # The gate above guarantees ≥ n aligned rows in all three arrays.
        diag_pred = np.asarray(bt_payload["predictions"][pred_model], dtype=float)[:n]
        diag_actual = np.asarray(bt_payload["actual"], dtype=float)[:n]
        diag_residuals = diag_actual - diag_pred
        diag_ts = pd.to_datetime(pd.Series(bt_payload["timestamps"][:n]))
        error_by_hour = pd.DataFrame({"hour": diag_ts.dt.hour, "abs_error": np.abs(diag_residuals)})
        hourly_error = error_by_hour.groupby("hour")["abs_error"].mean()

        redis_set(
            redis_key(f"diagnostics:{region}"),
            {
                "region": region,
                "diagnostics_source": "backtest",
                # Provenance for the UI's disclosure caption: which series these
                # residuals actually are. (#220 — never imply live-forecast
                # residuals when they're holdout-backtest residuals.)
                "residual_source": {
                    "kind": "walk_forward_backtest",
                    "horizon": int(bt_horizon),
                    "model": pred_model,
                    "exog_mode": DEFAULT_BACKTEST_EXOG_MODE,
                },
                "timestamps": _ts_list(diag_ts),
                "actual": diag_actual.tolist(),
                # Canonical field name; the old payload called this "ensemble",
                # which would mislabel an XGBoost backtest series.
                "predicted": diag_pred.tolist(),
                "residuals": diag_residuals.tolist(),
                "metrics": dict(bt_payload.get("metrics", {})),
                "hourly_error": {
                    "hours": hourly_error.index.tolist(),
                    "values": hourly_error.values.tolist(),
                },
                "feature_importance": feature_importance,
            },
            ttl=REDIS_TTL,
        )
        return PhaseResult(
            region=region,
            ok=True,
            details={"residual_horizon": int(bt_horizon), "points": int(n)},
        )
    except Exception as e:
        log.warning("job_diagnostics_failed", region=region, error=str(e))
        return PhaseResult(region=region, ok=False, error=str(e))


# Cap the alert cards persisted per region — a storm-season state can carry
# 50+ active NWS alerts; the payload discloses the uncapped total via
# ``alerts_total`` so truncation is never silent.
_ALERTS_PAYLOAD_CAP = 20


def _alert_payload_entry(alert) -> dict[str, Any]:
    """Trim a ``WeatherAlert`` to the fields the Risk tab renders."""
    return {
        "id": alert.id,
        "event": alert.event,
        "headline": alert.headline,
        "severity": alert.severity,
        "expires": alert.expires.isoformat() if alert.expires else None,
        "areas": alert.areas[:3],
        "urgency": alert.urgency,
    }


def _live_noaa_alerts(region: str) -> tuple[list[dict[str, Any]], int, int, int, int]:
    """Fetch live NOAA alerts for ``region`` and shape them for the payload.

    Returns ``(alerts, n_critical, n_warning, n_info, alerts_total)``.
    Expired alerts are dropped (the region/state caches and the stale-cache
    outage fallback can hold entries past their expiry); counts reflect ALL
    live alerts while the persisted card list is capped at
    ``_ALERTS_PAYLOAD_CAP`` with the true total in ``alerts_total``.
    Raises on total fetch failure (see ``data.noaa_client``).
    """
    from data.noaa_client import fetch_alerts_for_region

    fetched = fetch_alerts_for_region(region)
    now = datetime.now(UTC)
    live = []
    for a in fetched:
        exp = a.expires
        if exp is not None:
            if exp.tzinfo is None:
                # Rare naive timestamp — keep the alert rather than guess.
                pass
            elif exp <= now:
                continue
        live.append(a)

    n_crit = sum(1 for a in live if a.severity == "critical")
    n_warn = sum(1 for a in live if a.severity == "warning")
    n_info = sum(1 for a in live if a.severity == "info")
    if len(live) > _ALERTS_PAYLOAD_CAP:
        log.info("job_alerts_capped", region=region, total=len(live), cap=_ALERTS_PAYLOAD_CAP)
    alerts = [_alert_payload_entry(a) for a in live[:_ALERTS_PAYLOAD_CAP]]
    return alerts, n_crit, n_warn, n_info, len(live)


def write_alerts(data: RegionData) -> PhaseResult:
    """Write the alerts / stress / anomaly payload for the Risk tab.

    Alert-feed honesty (2026-07 review P0-1 lineage): live alerts come from
    NOAA/NWS via ``data.noaa_client`` (``alerts_source="noaa"``). On any
    fetch failure the payload degrades to an explicitly-empty
    ``alerts_source="unavailable"`` state — never fabricated content, and an
    outage is never disguised as "no active alerts" (the client raises
    rather than returning empty on total failure). Demo alerts are emitted
    only when ``config.USE_DEMO_DATA`` is set and are labeled
    ``alerts_source="demo"`` so the UI can disclose them. The anomaly and
    temperature sections are always real (derived from fetched demand/weather).
    """
    import config as _config
    from data.redis_client import redis_key, redis_set

    region = data.region
    try:
        stress: int | None
        alerts_total = 0
        if _config.USE_DEMO_DATA:
            from data.demo_data import generate_demo_alerts

            alerts = generate_demo_alerts(region)
            alerts_source = "demo"
            alerts_total = len(alerts)
            n_crit = sum(1 for a in alerts if a["severity"] == "critical")
            n_warn = sum(1 for a in alerts if a["severity"] == "warning")
            n_info = sum(1 for a in alerts if a["severity"] == "info")
        else:
            try:
                alerts, n_crit, n_warn, n_info, alerts_total = _live_noaa_alerts(region)
                alerts_source = "noaa"
            except Exception as noaa_err:
                log.warning("job_alerts_noaa_unavailable", region=region, error=str(noaa_err))
                alerts = []
                alerts_source = "unavailable"
                n_crit = n_warn = n_info = 0

        # Grid stress = supply tightness (current demand ÷ nameplate capacity),
        # NOT a count of NWS alerts (#265). The old alert-count heuristic
        # saturated to 100 for nearly every BA — a multi-state footprint always
        # has some active advisory. Alert counts ride along as context in
        # alert_counts below; stress is independent of the alert feed.
        from models.pricing import grid_stress

        _dseries = (
            data.demand_df["demand_mw"].dropna()
            if data.demand_df is not None and not data.demand_df.empty
            else None
        )
        current_demand = (
            float(_dseries.iloc[-1]) if _dseries is not None and len(_dseries) else None
        )
        stress, stress_label = grid_stress(region, current_demand)

        # Compute the ±2σ band over the FULL demand series, then slice to the
        # displayed 168h window, so the 24h rolling window is already warm at the
        # window's start. Computing rolling(24) on the 168h slice left the first
        # 24h NaN, so the bands rendered a day after the demand line began (bands
        # started ~24h in while the demand line started at hour 0).
        demand_full = data.demand_df
        roll_mean_full = demand_full["demand_mw"].rolling(24, min_periods=1).mean()
        roll_std_full = demand_full["demand_mw"].rolling(24, min_periods=2).std()
        recent = demand_full.tail(168).copy()
        upper = (roll_mean_full + 2 * roll_std_full).tail(168)
        lower = (roll_mean_full - 2 * roll_std_full).tail(168)
        anomalies = recent[recent["demand_mw"] > upper]

        recent_w = (
            data.weather_df.tail(168).copy()
            if data.weather_df is not None and not data.weather_df.empty
            else pd.DataFrame()
        )

        payload: dict[str, Any] = {
            "region": region,
            "scored_at": datetime.now(UTC).isoformat(),
            "alerts": alerts,
            "alerts_source": alerts_source,
            "alerts_total": alerts_total,
            "stress_score": stress,
            "stress_label": stress_label,
            "alert_counts": {"critical": n_crit, "warning": n_warn, "info": n_info},
            "anomaly": {
                "timestamps": _ts_list(recent["timestamp"]),
                "demand": recent["demand_mw"].tolist(),
                "upper": [float(v) if not np.isnan(v) else None for v in upper.values],
                "lower": [float(v) if not np.isnan(v) else None for v in lower.values],
                "anomaly_timestamps": _ts_list(anomalies["timestamp"])
                if not anomalies.empty
                else [],
                "anomaly_values": anomalies["demand_mw"].tolist() if not anomalies.empty else [],
            },
        }
        if not recent_w.empty and "temperature_2m" in recent_w.columns:
            payload["temperature"] = {
                "timestamps": _ts_list(recent_w["timestamp"]),
                "values": recent_w["temperature_2m"].tolist(),
            }

        # Latest reading for the Risk tab's "Current Conditions" cards. Without
        # this the web tier only had the temperature series above, so it could
        # render a lone Temperature card — no wind / humidity / cloud (the fields
        # _build_weather_context needs). Emit whichever the weather frame carries.
        if not recent_w.empty:
            last_w = recent_w.iloc[-1]
            payload["weather_current"] = {
                col: (
                    float(last_w[col])
                    if col in recent_w.columns and pd.notna(last_w[col])
                    else None
                )
                for col in (
                    "temperature_2m",
                    "wind_speed_80m",
                    "wind_speed_10m",
                    "relative_humidity_2m",
                    "cloud_cover",
                )
            }

        redis_set(redis_key(f"alerts:{region}"), payload, ttl=REDIS_TTL)
        return PhaseResult(
            region=region,
            ok=True,
            details={
                "n_critical": n_crit,
                "n_warning": n_warn,
                "n_info": n_info,
                "stress": stress,
            },
        )
    except Exception as e:
        log.warning("job_alerts_failed", region=region, error=str(e))
        return PhaseResult(region=region, ok=False, error=str(e))


# ── Meta keys ────────────────────────────────────────────────


def write_meta(key: str, extra: dict[str, Any] | None = None) -> None:
    """Write a ``gridpulse:meta:{key}`` marker with current UTC timestamp."""
    from data.redis_client import redis_key, redis_set

    payload = {
        "updated_at": datetime.now(UTC).isoformat(),
    }
    if extra:
        payload.update(extra)
    redis_set(redis_key(f"meta:{key}"), payload, ttl=REDIS_TTL)


# ── Orchestration helpers ────────────────────────────────────


def summarize(results: list[PhaseResult], phase: str) -> None:
    """Log a per-phase summary line with pass/fail counts."""
    ok = [r.region for r in results if r.ok]
    failed = [(r.region, r.error) for r in results if not r.ok]
    log.info(
        "job_phase_summary",
        phase=phase,
        ok_count=len(ok),
        fail_count=len(failed),
        failed=failed or None,
    )


def safe_phase(name: str, region: str, fn, *args, **kwargs) -> PhaseResult:
    """Invoke a phase function and wrap any exception as a failed ``PhaseResult``."""
    t0 = time.time()
    try:
        res = fn(*args, **kwargs)
    except Exception as e:
        log.warning(
            "job_phase_crashed",
            phase=name,
            region=region,
            error=str(e),
            tb=traceback.format_exc()[-400:],
        )
        return PhaseResult(region=region, ok=False, error=str(e))
    elapsed = time.time() - t0
    if isinstance(res, PhaseResult):
        res.details.setdefault("elapsed_s", round(elapsed, 2))
        return res
    # Allow phase functions to return ``bool`` / primitives for simplicity.
    return PhaseResult(
        region=region,
        ok=bool(res),
        details={"elapsed_s": round(elapsed, 2), "result": str(res)},
    )


__all__ = [
    "BACKTEST_HORIZONS",
    "DEFAULT_BACKTEST_EXOG_MODE",
    "FORECAST_HORIZON_HOURS",
    "PhaseResult",
    "REDIS_TTL",
    "RegionData",
    "engineer_region_features",
    "fetch_all_regions",
    "fetch_region_data",
    "ordered_regions",
    "predict_and_write_forecast",
    "safe_phase",
    "summarize",
    "write_actuals_and_weather",
    "write_alerts",
    "write_backtests",
    "write_diagnostics",
    "write_generation",
    "write_meta",
    "write_weather_correlation",
]


# ── Backwards-compat helpers (unused by callers, kept for readability) ──


def _unused_json_dumps_placeholder() -> None:  # pragma: no cover
    json.dumps({})  # keeps ``json`` import in use if redis_set paths change
