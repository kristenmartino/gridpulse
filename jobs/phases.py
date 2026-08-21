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
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import structlog

import config
from config import (
    EIA_API_KEY,
    PRECOMPUTE_MAX_WORKERS,
    REGION_COORDINATES,
)
from observability import collect_substeps, substep

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Sub-phase timing (#389 follow-up)
# ---------------------------------------------------------------------------
# `scoring_job.timed` measures whole phases, which is how we learned `forecast`
# is 60.1% of worker time. It cannot say WHICH PART of forecast, and that gap
# has now cost two wrong optimisation targets: local profiling with real SPA
# pickles puts the entire predict path at ~3.4s/BA against a phase that
# measures ~60s/BA in production. An 18x discrepancy is not something to
# reason about from a laptop — the missing steps (notably the multi-point
# weather overlay, ADR-012) are only exercised with production-shaped inputs.
#
# So: measure it where it runs. One tick answers it.
#
# Deliberately kept OUT of `summary["timings"]`. `_phase_rollup` sums every
# key it finds there, so injecting `forecast.*` alongside `forecast` would
# double-count and silently corrupt the very percentage (60.1%) this exists to
# refine. Sub-steps ride in `summary["subtimings"]` and roll up separately.
#
# Sub-step timing lives in ``observability`` so the DATA layer can record its
# own sub-steps too (#389 follow-up: the `fetch` phase needed them, and
# ``data/`` must not import ``jobs/``). Re-exported here because every existing
# call site and test refers to ``jobs.phases.substep``.


# Redis keys + TTL kept in sync with components/callbacks.py consumers.
REDIS_TTL = 86400

from models.drift import WINDOW_30D_HOURS as DRIFT_WINDOW_HOURS  # noqa: E402

#: TTL for ``gridpulse:drift:{region}`` — DERIVED from the window it stores, not
#: set independently (#512). The drift payload is a 30-day rolling history, and it
#: was previously written with the generic 24h ``REDIS_TTL`` every snapshot key
#: uses. Because the phase returns *before* persisting when there is no matchable
#: actual hour, a BA that cannot grade for 24 consecutive hours never refreshed
#: the TTL and lost its ENTIRE 30-day window, silently restarting from one record.
#:
#: That is not hypothetical: AZPS had a 25.0h gap on 2026-08-04 (inside the #389
#: incident) and its window restarted, sitting at 54 records against 720 for every
#: other BA a week later. Grading cadence varies by an order of magnitude across
#: the fleet — LDWP grades ~2x/day against a median of 24x/day — so the broken-feed
#: BAs live near this cliff permanently.
#:
#: Expressed as ``window + margin`` so the two cannot drift apart: if the window
#: ever grows, the TTL follows. A test asserts the relationship rather than the
#: literal.
DRIFT_TTL_MARGIN_HOURS = 48
DRIFT_REDIS_TTL = (DRIFT_WINDOW_HOURS + DRIFT_TTL_MARGIN_HOURS) * 3600
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
    #: Readings the #309 quality guard NaN-coerced out of ``demand_df``
    #: (``[{"ts", "mw", "reason"}]``) — stamped onto the actuals payload so
    #: the tiles and /grid/summary can disclose the exclusion.
    artifact_exclusions: list[dict[str, Any]] = field(default_factory=list)
    #: The anchor-conditioned frame (ADR-009), populated only for broken-class
    #: regions when the ``anchor_conditioning`` flag is on. THE FORK IS
    #: LOAD-BEARING: actuals/tiles, drift, alerts, weather-correlation, and
    #: diagnostics keep reading the real ``demand_df``; only the
    #: feature/forecast path prefers this frame. Never mutate ``demand_df``.
    conditioned_demand_df: pd.DataFrame | None = None
    #: Origin (``forecasts[0]["timestamp"]``) of the payload currently in Redis,
    #: set from ``read_existing_forecast`` before this tick overwrites it. The
    #: forecast phase refuses to replace it with an OLDER one (#537).
    previous_forecast_origin: pd.Timestamp | None = None
    #: ``{canonical_hour: was_placeholder}`` for the vintage window, handed
    #: across in memory by :func:`write_vintage_records` so the forecast phase
    #: can record what its anchor was seeded with (#547) WITHOUT re-reading the
    #: ~65KB ``vintage:{region}`` key on every region every tick — the scoring
    #: job is under an active runtime budget (#389).
    #:
    #: ``None`` means the question was never asked (vintage phase skipped,
    #: failed, or Redis unconfigured), which is NOT the same fact as "the
    #: anchor was metered". Readers must preserve that distinction.
    placeholder_by_hour: dict[str, bool] | None = None

    @property
    def anchor_frame(self) -> pd.DataFrame:
        """The frame the FORECAST path anchors on — conditioned when present."""
        return (
            self.conditioned_demand_df if self.conditioned_demand_df is not None else self.demand_df
        )


def apply_demand_quality_guard(data: RegionData) -> PhaseResult:
    """NaN-coerce implausible trailing readings out of ``data.demand_df`` (#309).

    Runs ONCE per region, AFTER vintage capture (which must record the raw
    values — it is the study of exactly these artifacts) and BEFORE every
    consumer of the frame: the actuals payload (tiles), drift (stops scoring
    forecasts against 730-MW partials), and feature engineering (the anchor —
    ``_resolve_forecast_start`` then anchors on the last real hour).

    Never fatal: on any error the raw frame stands and the run proceeds — a
    broken guard must degrade to today's behavior, not take out the tick.
    """
    from data.quality import coerce_demand_artifacts

    try:
        cleaned, exclusions = coerce_demand_artifacts(data.demand_df)
        if exclusions:
            data.demand_df = cleaned
            data.artifact_exclusions = exclusions
            log.info(
                "demand_artifacts_excluded",
                region=data.region,
                n=len(exclusions),
                newest=exclusions[-1]["ts"],
                reasons=[e["reason"] for e in exclusions],
            )
        return PhaseResult(region=data.region, ok=True, details={"excluded": len(exclusions)})
    except Exception as exc:
        log.warning("demand_quality_guard_failed", region=data.region, error=str(exc))
        return PhaseResult(region=data.region, ok=False, error=str(exc))


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

    # #389 follow-up: `fetch` is 13.0% of worker time and, like `forecast`
    # before it, the phase-level number cannot say which part. It is also the
    # one phase whose total is dominated by an upstream we do not control, so
    # the EIA leg has to be separable from the three Open-Meteo legs (named
    # inside ``data.weather_client``) before any weather-side change can be
    # sized against it.
    try:
        with substep("eia_demand"):
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
        # ADR-009 seam: features (and therefore every AR anchor + SARIMAX's
        # gap-actuals, which read ``featured``) build from the anchor frame —
        # conditioned when present, the real frame otherwise.
        merged = merge_demand_weather(data.anchor_frame, data.weather_df)
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


#: #401: minimum observed hours before per-BA temperature percentiles are
#: published. Below this the high quantiles are dominated by a handful of
#: readings, and a fleet-uniform line the chart already labels as generic is
#: better than a region-specific one that is mostly noise.
_TEMP_PCT_MIN_HOURS = 24 * 14


def _temperature_percentiles(weather_df: pd.DataFrame) -> dict[str, Any] | None:
    """p90/p95/p99 of OBSERVED temperature for one BA, or ``None``.

    #401. The Risk tab drew reference lines at 95/100/105 °F on all 51 BAs —
    a mark that is operationally meaningful in Florida and unreachable in the
    Pacific Northwest. #273 made the chart disclose that; this makes it
    unnecessary.

    **Observed hours only.** ``weather_df`` carries roughly 90 days of history
    *and* a 16-day forecast; including the forecast would compute a threshold
    partly from predictions, which is exactly the kind of quiet circularity
    this codebase keeps finding. Rows at or before ``now`` only.

    The window is the frame's own trailing history — NOT the 10-year ERA5
    normal. That is a deliberate departure from #401's sketch: the normal is a
    per-(day_of_year, hour) *mean* and cannot yield a percentile, and a fresh
    10-year fetch per BA per tick is not affordable. A seasonal threshold is
    also the more useful one — "hot for this BA lately" beats "hot for this BA
    in February" when the chart is read in August. The window length is
    published with the numbers so the label can say what it is.
    """
    if weather_df is None or weather_df.empty or "temperature_2m" not in weather_df.columns:
        return None
    df = weather_df[["timestamp", "temperature_2m"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    observed = df[df["timestamp"] <= pd.Timestamp.now(tz=UTC)]
    vals = pd.to_numeric(observed["temperature_2m"], errors="coerce").dropna()
    if len(vals) < _TEMP_PCT_MIN_HOURS:
        return None
    q = vals.quantile([0.90, 0.95, 0.99])
    out = {
        "p90": round(float(q.loc[0.90]), 1),
        "p95": round(float(q.loc[0.95]), 1),
        "p99": round(float(q.loc[0.99]), 1),
        "n_hours": int(len(vals)),
        "window_days": int(round(len(vals) / 24)),
        "source": "observed_trailing",
    }
    if not all(np.isfinite(out[k]) for k in ("p90", "p95", "p99")):
        return None
    return out


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
            # #309: readings the quality guard excluded this tick — the series
            # above is already cleaned (NaN at excluded hours); this field is
            # the disclosure the tiles and /grid/summary render.
            "artifact_excluded": data.artifact_exclusions,
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

        # #401: per-BA temperature percentiles, so the Risk tab's heat lines
        # can stop being fleet-uniform. Computed here because this phase
        # already holds the frame — no new upstream fetch. Best-effort: a
        # failure must not cost the weather payload above, which is the thing
        # this phase actually exists to write.
        pct = _temperature_percentiles(weather_df)
        if pct is not None:
            try:
                persist(
                    redis_key(f"weather_percentiles:{region}"),
                    {"region": region, "scored_at": scored_at, **pct},
                    ttl=REDIS_TTL,
                )
            except Exception as e:  # pragma: no cover — advisory payload
                log.warning("weather_percentiles_persist_failed", region=region, error=str(e))

        return PhaseResult(
            region=region,
            ok=True,
            details={
                "demand_rows": len(demand_df),
                "weather_rows": len(weather_df),
                "temp_percentiles": bool(pct),
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
        # #427: `generation` is 323.4s of worker time (10.8%) and the phase
        # total cannot say whether that is EIA latency or our own pivot work.
        # Naming the fetch splits it: the remainder is `phases.generation`
        # minus this, so two numbers fully attribute the phase.
        with substep("eia_generation"):
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
        # #427: same split as `generation` above — 118.6s (4.0%) of worker
        # time, attribution unknown until the EIA call is named.
        with substep("eia_interchange"):
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


def _ar_seed_bridge(
    featured: pd.DataFrame,
    demand_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Real-demand hours that ``dropna`` deleted from the tail of ``featured``.

    ``engineer_features`` drops every row whose autoregressive lag source was
    null, so one null hour deletes the rows 1, 2, 3, 24 and 168 hours after it
    — including, when the null hour sits 1-3/24/168 hours before the frame end,
    the *tail*. Those deleted rows carry **real** demand: they were dropped for
    what their lag pointed at, not for what they hold.

    This returns exactly those hours, as the maximal run of **contiguous**
    hourly rows carrying real demand (non-NaN and strictly positive — the same
    #129 predicate ``_resolve_forecast_start`` selects on) beginning at
    ``last_featured_ts + 1h``. Empty whenever the very next hour is missing,
    unusable, or not hourly-adjacent, so a caller that treats "empty" as
    "cannot advance" degrades to today's behaviour.

    Contiguity is the point, not an implementation detail: it is what lets the
    recursion's near lags (``demand_lag_1h/2h/3h``, ``ramp_rate``) at an origin
    past the bridge resolve to the hours they name.

    Args:
        featured: Engineered frame (post-``dropna``).
        demand_df: The frame the forecast anchors on (``RegionData.anchor_frame``).

    Returns:
        ``timestamp`` / ``demand_mw`` rows, oldest first. Possibly empty.
    """
    empty = pd.DataFrame({"timestamp": [], "demand_mw": []})
    if featured is None or featured.empty or "timestamp" not in featured.columns:
        return empty
    if demand_df is None or demand_df.empty:
        return empty
    if "timestamp" not in demand_df.columns or "demand_mw" not in demand_df.columns:
        return empty

    last_featured_ts = featured["timestamp"].max()
    stamps = demand_df["timestamp"]
    # Fail closed on anything that does not line up — a non-datetime column, a
    # tz mismatch, an empty tail. The comparisons below would raise, and the
    # honest answer to "can we bridge?" on frames that do not line up is no.
    try:
        stamps_tz = stamps.dt.tz
    except (AttributeError, TypeError):
        return empty
    if stamps_tz != getattr(last_featured_ts, "tz", None) or pd.isna(last_featured_ts):
        return empty

    mask = (
        demand_df["demand_mw"].notna() & (demand_df["demand_mw"] > 0) & (stamps > last_featured_ts)
    )
    tail = demand_df.loc[mask, ["timestamp", "demand_mw"]].sort_values("timestamp")
    if tail.empty:
        return empty

    tail = tail.reset_index(drop=True)
    hour = pd.Timedelta(hours=1)
    if tail["timestamp"].iloc[0] != last_featured_ts + hour:
        return empty
    breaks = np.asarray(tail["timestamp"].diff() != hour, dtype=bool).copy()
    breaks[0] = False  # the leading NaT diff is not a break
    n = int(np.argmax(breaks)) if breaks.any() else len(tail)
    return tail.iloc[:n]


def _resolve_forecast_start(
    featured: pd.DataFrame,
    demand_df: pd.DataFrame,
    *,
    region: str | None = None,
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

    #559: when ``featured``'s tail is older than the last real demand hour —
    ``dropna(subset=autoregressive)`` deleting rows whose lag source was null —
    the anchor may advance across ``_ar_seed_bridge``'s contiguous run of real
    demand hours, but ONLY under ``temporal_ar_seed``. See the comment at the
    cap for why the positional seed can never take that advance safely.

    Args:
        featured: Engineered DataFrame (post-merge, post-dropna).
        demand_df: Raw EIA demand DataFrame from ``data.demand_df``.

    Returns:
        Forecast-start timestamp (timezone-aware UTC).
    """
    from config import feature_enabled

    last_featured_ts = featured["timestamp"].max()

    def _fallback(reason: str) -> pd.Timestamp:
        start = last_featured_ts + pd.Timedelta(hours=1)
        log.info(
            "forecast_start_resolved",
            region=region,
            forecast_start=start.isoformat(),
            last_real_demand=None,
            last_featured_ts=last_featured_ts.isoformat(),
            binding_term=reason,
        )
        return start

    if demand_df is None or demand_df.empty:
        return _fallback("no_demand_frame")

    # Filter to real demand readings — must be non-NaN AND strictly
    # positive. A balancing authority cannot have zero load; any zero
    # is a missing-data artifact.
    mask = demand_df["demand_mw"].notna() & (demand_df["demand_mw"] > 0)
    real_demand = demand_df.loc[mask, "timestamp"]
    if real_demand.empty:
        return _fallback("no_real_demand")

    last_real_demand = real_demand.max()

    # The cap exists so we never forecast an hour whose autoregressive lag
    # context we cannot seed. ``last_featured_ts`` is a STRICTER question than
    # that — it asks whether the origin's predecessor row survived
    # ``dropna(subset=autoregressive)``, which is feature-frame warm-up
    # bookkeeping, not a fact about the demand we hold. A single null hour
    # deletes the rows 1/2/3/24/168 hours later, so the tail of ``featured``
    # can end 16 hours behind demand that arrived and is real, and the origin
    # freezes there while fresh hours keep landing (#559 candidate 1, driving
    # the #537 drift shortfall).
    #
    # The question the cap SHOULD ask is "do we hold real hourly demand for the
    # hours immediately before the origin?" — answerable from the demand grid,
    # independent of ``dropna``. ``_ar_seed_bridge`` answers it.
    #
    # GATED ON ``temporal_ar_seed``, and this is load-bearing rather than
    # cautious. The recursion's positional seed reads ``demand_lag_1h`` as
    # "the last surviving entry", so an origin advanced past the tail of
    # ``featured`` would index it to ``last_featured_ts`` instead of
    # ``start - 1h`` — turning a stall into a silently wrong value, which is
    # worse than the stall. ``positional_seed_matches_hours`` is the exact
    # condition for the positional arm being right, and it requires the seed's
    # last entry to BE ``start - 1h``; it is therefore false by construction
    # for every advanced origin, so no positional advance is ever provably
    # safe. Only the hour-indexed path (``temporal_ar_seed``, #584/#615)
    # resolves the bridged origin's lags to the hours they name. Flag off →
    # byte-identical to the pre-#559 behaviour.
    anchor = min(last_real_demand, last_featured_ts)
    bridge_hours = 0
    if last_featured_ts < last_real_demand and feature_enabled("temporal_ar_seed"):
        bridge = _ar_seed_bridge(featured, demand_df)
        if not bridge.empty:
            anchor = bridge["timestamp"].max()
            bridge_hours = len(bridge)
    start = anchor + pd.Timedelta(hours=1)

    # #537: the resolved origin was previously observable ONLY by reconstructing
    # it from the drift log a tick later (``new_record_ts - lead_hours + 1``),
    # which is how two multi-day origin freezes went unnoticed. Emit the value
    # AND both terms of the ``min()`` above, because which one binds is the
    # difference between the two failure modes: ``featured`` binding means a
    # NaN hole deleted rows from the tail of the feature frame (the lags are
    # positional and ``dropna`` drops the rows that read them), while
    # ``real_demand`` binding means the demand series itself ends there.
    #
    # ``binding_term`` names the term the RESOLVED start actually sits on, so
    # a fully bridged stall reports ``real_demand`` — which is true, the demand
    # series is what bounds it now. ``featured_bridged`` is the partial case:
    # the bridge ran into a hole before reaching ``last_real_demand``.
    if start == last_real_demand + pd.Timedelta(hours=1):
        binding_term = "real_demand"
    elif start == last_featured_ts + pd.Timedelta(hours=1):
        binding_term = "featured"
    else:
        binding_term = "featured_bridged"
    log.info(
        "forecast_start_resolved",
        region=region,
        forecast_start=start.isoformat(),
        last_real_demand=last_real_demand.isoformat(),
        last_featured_ts=last_featured_ts.isoformat(),
        binding_term=binding_term,
        bridge_hours=bridge_hours,
    )
    return start


def _ar_seed_for_origin(
    featured: pd.DataFrame,
    demand_df: pd.DataFrame | None,
    forecast_start: pd.Timestamp,
    *,
    region: str | None = None,
) -> tuple[pd.Timestamp, pd.DataFrame | None]:
    """Couple an advanced origin to the seed that can actually serve it.

    ``_resolve_forecast_start`` may advance the origin past the tail of
    ``featured`` across ``_ar_seed_bridge``. Those bridge hours are absent from
    ``featured`` — that is the defect — so the recursion must be handed them
    explicitly, or the hour-indexed seed would *impute* hours we hold: on
    LGEE's 16-hour hole ``HourIndexedHistory.lag`` would fall through
    interpolation into the 24h step-back regime and read demand from two days
    earlier for ``demand_lag_1h``. Strictly worse than the stall.

    So the two must not be able to drift apart. This returns them together, and
    **clamps the origin back** to ``last_featured_ts + 1h`` if a bridge that
    reaches it cannot be produced — making "origin advanced, seed did not"
    unreachable rather than merely unlikely.

    Returns:
        ``(forecast_start, seed_frame)``. ``seed_frame`` is ``None`` whenever
        the origin needs no bridge, in which case callers seed from
        ``featured`` exactly as before.
    """
    hour = pd.Timedelta(hours=1)
    if featured is None or featured.empty or "timestamp" not in featured.columns:
        return forecast_start, None
    last_featured_ts = featured["timestamp"].max()
    if pd.isna(last_featured_ts) or forecast_start <= last_featured_ts + hour:
        return forecast_start, None

    bridge = _ar_seed_bridge(featured, demand_df)
    if bridge.empty or bridge["timestamp"].max() + hour != forecast_start:
        log.error(
            "forecast_origin_bridge_unavailable",
            region=region,
            forecast_start=forecast_start.isoformat(),
            last_featured_ts=last_featured_ts.isoformat(),
            bridge_hours=len(bridge),
        )
        return last_featured_ts + hour, None

    seed = pd.concat(
        [featured[["timestamp", "demand_mw"]], bridge[["timestamp", "demand_mw"]]],
        ignore_index=True,
    )
    log.info(
        "forecast_origin_bridged",
        region=region,
        forecast_start=forecast_start.isoformat(),
        last_featured_ts=last_featured_ts.isoformat(),
        bridge_hours=len(bridge),
    )
    return forecast_start, seed


def _demand_at(frame: pd.DataFrame | None, ts: pd.Timestamp) -> float | None:
    """``demand_mw`` at exactly ``ts``, or ``None`` when absent or unusable.

    ``None`` for a missing row, a NaN, or a non-positive value — the same
    "real demand" predicate ``_resolve_forecast_start`` selects on, so a value
    this returns is one the anchor could actually have been seeded with.
    """
    if frame is None or frame.empty or "demand_mw" not in frame.columns:
        return None
    hit = frame.loc[frame["timestamp"] == ts, "demand_mw"]
    if hit.empty:
        return None
    value = hit.iloc[-1]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) and out > 0 else None


def _anchor_provenance(
    data: RegionData,
    featured: pd.DataFrame,
    forecast_start: pd.Timestamp,
) -> dict[str, Any]:
    """What the forecast was seeded with, recorded at forecast time (#547).

    ``docs/BENCHMARK_METHODOLOGY.md`` limit 11: where EIA has not metered an
    hour yet it publishes the BA's own day-ahead value in the ``D`` field, and
    ``_resolve_forecast_start`` selects on positive ``D`` rather than on
    *metered* ``D`` — so our recursion is sometimes seeded with the very series
    the benchmark scores us against. The rate is published per BA as
    ``placeholder_pct``; which *forecasts* it touched was never recorded, so
    the materiality was stated as unmeasured rather than as small. This is the
    instrument that makes it measurable, and it CANNOT be backfilled: the
    payload that would prove a past run's anchor is overwritten every tick.

    Derived from ``forecast_start`` rather than by re-running the selection, so
    the recorded anchor and the anchor actually used cannot drift apart. That
    holds across every branch of the resolver's fallback chain, since all of
    them return ``anchor + 1h``.

    Every field is tri-state. ``None`` means the question could not be
    answered, which is a different fact from a confirmed negative — a record
    written before the vintage window covered its anchor hour must not claim
    the anchor was metered.

    Returns:
        ``{anchor_ts, anchor_mw, anchor_was_placeholder, anchor_conditioned}``.
    """
    from data.vintage import canonical_hour

    anchor_ts = forecast_start - pd.Timedelta(hours=1)

    # The seed of ``demand_lag_1h`` normally comes from the FEATURED frame
    # (data/feature_engineering.py filters it to positive non-NaN and takes
    # history[-1]), so that is the authoritative source for the value. The
    # anchor frame is the fallback for the branches where the hour survived
    # selection but not feature engineering — which since #559 includes the
    # ordinary bridged case: an origin advanced across ``_ar_seed_bridge``
    # anchors on an hour ``dropna`` deleted, and the bridge took that hour's
    # value from this same anchor frame. So the fallback is not a degenerate
    # path there; it is the one that names the value actually seeded.
    anchor_mw = _demand_at(featured, anchor_ts)
    if anchor_mw is None:
        anchor_mw = _demand_at(data.anchor_frame, anchor_ts)

    was_placeholder: bool | None = None
    if data.placeholder_by_hour is not None:
        key = canonical_hour(anchor_ts)
        if key is not None:
            # ``.get`` returns None for an hour outside the window — unknown,
            # not metered. Vintage captures the RAW frame while the anchor
            # resolves on the guard-cleaned one, so the two can disagree about
            # which hours exist and that disagreement must read as unknown.
            was_placeholder = data.placeholder_by_hour.get(key)

    # ADR-009 is the OTHER way an anchor becomes the operator's own forecast,
    # and it is deliberate: for broken-class feeds we substitute ``forecast_mw``
    # into the trailing hours on a forked frame, because it was measured better
    # (58.2% wrong vs 14.5%). Vintage records the raw ``D``, so such an anchor
    # reads ``was_placeholder=False`` while the value that seeded the model was
    # their day-ahead figure — a true field whose framing asserts something
    # false. Recorded separately rather than folded in.
    #
    # Decided by comparing the two frames at the anchor hour, so it is a fact
    # about the value used and not an inference from the conditioning window.
    if data.conditioned_demand_df is None:
        # ``anchor_frame`` IS ``demand_df`` here, so this is a confirmed
        # negative rather than an absence of evidence.
        conditioned: bool | None = False
    else:
        raw_mw = _demand_at(data.demand_df, anchor_ts)
        cond_mw = _demand_at(data.conditioned_demand_df, anchor_ts)
        conditioned = None if cond_mw is None else cond_mw != raw_mw

    return {
        "anchor_ts": anchor_ts.isoformat(),
        "anchor_mw": round(anchor_mw, 2) if anchor_mw is not None else None,
        "anchor_was_placeholder": was_placeholder,
        "anchor_conditioned": conditioned,
    }


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
    with substep("frame_climatology"):
        if numeric_cols:
            group_means = hist.groupby(["_hour", "_dow"])[numeric_cols].mean()
            future_hour = future_df["timestamp"].dt.hour
            future_dow = future_df["timestamp"].dt.dayofweek
            last_row = featured.iloc[-1]

            # Vectorised (2026-08-05). This was `for col in numeric_cols: for i
            # in range(horizon)` — ~49 x 720 = 35,280 scalar `.iloc`/`.loc`
            # lookups into a MultiIndex, per BA, per tick.
            #
            # It measured 1,081.9s of summed worker time in production, 33.1%
            # of the whole forecast phase and its single largest sub-step
            # (`scoring_phase_rollup.forecast_substeps`, 2026-08-05T14:11).
            #
            # A LOCAL BENCHMARK SAID 1.33s/call, i.e. ~68s across 51 BAs, and
            # on that basis this was dismissed as ~1% and left alone. The
            # local number was 16x optimistic because the loop is PURE PYTHON
            # and therefore holds the GIL: at PRECOMPUTE_MAX_WORKERS=8 the
            # eight regions being scored concurrently serialise against each
            # other here. A single-threaded benchmark cannot see that, which
            # is the whole reason the sub-phase instrumentation exists.
            # Vectorising does not just make this step faster — it stops it
            # blocking the other seven workers.
            #
            # Semantics preserved exactly, including two things a naive
            # rewrite gets wrong:
            #   1. A key PRESENT in group_means whose mean is NaN must stay
            #      NaN. Only a MISSING key falls back to last_row. `reindex`
            #      alone cannot tell those apart, hence the explicit `missing`
            #      mask rather than `fillna`.
            #   2. Assignment stays column-by-column in `numeric_cols` order so
            #      column insertion order — and therefore the resulting frame's
            #      column order — is byte-identical to the loop's.
            future_keys = pd.MultiIndex.from_arrays(
                [future_hour.to_numpy(), future_dow.to_numpy()],
                names=["_hour", "_dow"],
            )
            aligned = group_means.reindex(future_keys)
            missing = ~future_keys.isin(group_means.index)
            any_missing = bool(missing.any())
            for col in numeric_cols:
                values = aligned[col].to_numpy(dtype=float, copy=True)
                if any_missing:
                    values[missing] = float(last_row[col]) if col in last_row.index else 0.0
                future_df[col] = values

        for col in feature_cols:
            if col not in future_df.columns:
                future_df[col] = 0

    # Overlay actual Open-Meteo forecast where available. For hours
    # beyond the forecast horizon (~day 16+), climatology stays.
    with substep("frame_weather_overlay"):
        if weather_df is not None and not weather_df.empty:
            future_df = _overlay_weather_forecast(future_df, featured, weather_df, horizon)

    # #283 Phase 2: past the Open-Meteo boundary, swap the recent-28d climatology
    # weather for the (day_of_year, hour) weather-normal. No-op when the flag is
    # off / the artifact isn't built, so a flag-off run is byte-identical to today.
    with substep("frame_normal_tail"):
        future_df = _overlay_weather_normal_tail(future_df, featured, weather_df, horizon)

    return future_df


def _predict_xgboost_with_recursive_autoregressive(
    model: Any,
    featured: pd.DataFrame,
    future_df: pd.DataFrame,
    horizon: int,
    recursive_hours: int = RECURSIVE_AUTOREGRESSIVE_HOURS,
    force_temporal: bool | None = None,
    seed_frame: pd.DataFrame | None = None,
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
    # #559: ``seed_frame`` carries the hours ``dropna`` deleted from the tail of
    # ``featured`` when the origin was bridged past them (``_ar_seed_for_origin``).
    # It only ever differs from ``featured`` on ticks the origin advanced, which
    # requires ``temporal_ar_seed`` — so the positional arm always sees exactly
    # today's seed.
    seed = featured if seed_frame is None else seed_frame
    recursive_preds = recursive_autoregressive_forecast(
        model,
        seed["demand_mw"].tolist(),
        future_df.iloc[:n_recursive],
        predict_xgboost,
        seed_timestamps=seed.get("timestamp"),
        force_temporal=force_temporal,
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


def _gate_decision(judged: list[dict[str, Any]]) -> bool:
    """The #326 decision rule over judged gate anchors.

    A live-anchor failure (no ``truth_median_ape`` — the frame about to
    serve) rejects outright; otherwise up to
    ``MODEL_GATE_MAX_OFFSET_FAILURES`` offset-anchor dive pockets are
    tolerated, and a pattern of failures rejects.
    """
    from config import MODEL_GATE_MAX_OFFSET_FAILURES

    live_ok = all(v["ok"] for v in judged if "truth_median_ape" not in v)
    n_failures = sum(1 for v in judged if not v["ok"])
    return live_ok and n_failures <= MODEL_GATE_MAX_OFFSET_FAILURES


def serve_path_gate(
    model_dict: Any,
    featured: pd.DataFrame,
    weather_df: pd.DataFrame | None,
    region: str,
) -> dict[str, Any]:
    """Judge a candidate XGBoost by replaying it through the real serve path.

    Daily retrains are a fit lottery (#326): ~27% of persisted LDWP vintages
    produce recursive forecasts that collapse overnight demand into a
    phantom regime, the failure is condition-dependent, and the published
    holdout carries zero signal — it scores a freshly retrained model on a
    sliced historical frame and never runs the candidate through
    ``_build_future_feature_frame`` + the recursion. This gate does exactly
    that, at persist time, from ``MODEL_GATE_PROBE_ANCHORS`` anchors stepped
    ``MODEL_GATE_PROBE_STEP_HOURS`` apart (multiple anchors because a single
    window measurably undercounts divers).

    Two judgment modes, chosen per anchor by what exists:

    * **Offset anchors** replay into known history, so they are judged
      against settled truth — median APE (median, not mean: a stray
      artifact row in the unguarded training frame must not fail an honest
      replay) plus replay-trough vs truth-trough. Calibration forced this:
      judging offsets by the trailing-week band false-rejected an honest
      model whose replay tracked a genuine demand dip at 5% MAPE (0715).
    * **The live anchor** has no truth yet — its replay is judged against
      the trailing week: trough vs the 5th-percentile floor and mean within
      the ``MODEL_GATE_LEVEL_RATIO_*`` band. Divers measured 0.27–0.49 on
      the trough ratio at their own training moments; sane fits >= 0.90.

    Decision rule: a live-anchor failure rejects outright (that exact frame
    serves next); otherwise one offset-anchor dive pocket is tolerated
    (``MODEL_GATE_MAX_OFFSET_FAILURES`` — the lottery is a spectrum and
    rejecting every pocket would streak rejections into stale pointers),
    while a pattern of failing anchors rejects.

    Fail-open by design: a gate that errors (or lacks history) must not
    freeze model rollout on a harness bug, so exceptions log
    ``model_gate_error`` and count as pass. Refusals are the caller's job —
    the verdict's ``passed`` drives ``save_model(update_latest=...)``, so a
    rejected candidate is still persisted (the forensic record the #326
    study depended on) but ``latest.json`` keeps pointing at yesterday's
    accepted model.

    Returns a JSON-serializable verdict:
    ``{"passed": bool, "anchors": [...], "skipped"?: str}``.
    """
    from config import (
        MODEL_GATE_LEVEL_RATIO_MAX,
        MODEL_GATE_LEVEL_RATIO_MIN,
        MODEL_GATE_PROBE_ANCHORS,
        MODEL_GATE_PROBE_HORIZON_HOURS,
        MODEL_GATE_PROBE_STEP_HOURS,
        MODEL_GATE_TROUGH_FRACTION,
        MODEL_GATE_TRUTH_MEDIAN_APE_MAX,
        feature_enabled,
    )
    from data.feature_engineering import recursive_autoregressive_forecast
    from models.xgboost_model import predict_xgboost

    if not feature_enabled("model_serve_gate"):
        return {"passed": True, "skipped": "flag_off", "anchors": []}

    horizon = MODEL_GATE_PROBE_HORIZON_HOURS
    ts = pd.to_datetime(featured["timestamp"], utc=True)
    frame_end = ts.max()
    anchors = [
        frame_end - pd.Timedelta(hours=MODEL_GATE_PROBE_STEP_HOURS * i)
        for i in range(MODEL_GATE_PROBE_ANCHORS)
    ]

    verdicts: list[dict[str, Any]] = []
    for anchor in anchors:
        try:
            seed = featured[ts <= anchor]
            if len(seed) < 168:
                continue
            recent = seed["demand_mw"].tail(168).astype(float)
            ref_trough = float(recent.quantile(0.05))
            ref_mean = float(recent.mean())
            if not (np.isfinite(ref_trough) and ref_trough > 0 and ref_mean > 0):
                continue
            future = _build_future_feature_frame(
                seed,
                horizon,
                weather_df=weather_df,
                start_ts=anchor + pd.Timedelta(hours=1),
            )
            preds = np.asarray(
                recursive_autoregressive_forecast(
                    model_dict,
                    seed["demand_mw"].tolist(),
                    future.iloc[:horizon],
                    predict_xgboost,
                    seed_timestamps=seed.get("timestamp"),
                ),
                dtype=float,
            )
            if preds.size == 0:
                continue
            trough_ratio = float(preds.min()) / ref_trough
            level_ratio = float(preds.mean()) / ref_mean
            entry: dict[str, Any] = {
                "anchor": anchor.isoformat(),
                "trough_ratio": round(trough_ratio, 3),
                "level_ratio": round(level_ratio, 3),
            }

            truth = featured[(ts > anchor)].head(horizon)["demand_mw"].astype(float).to_numpy()
            n = min(len(truth), len(preds))
            real = truth[:n] > 0
            if int(real.sum()) >= 12:
                # Offset anchor — judge against settled truth.
                t = truth[:n][real]
                p = preds[:n][real]
                median_ape = float(np.median(np.abs(p - t) / t)) * 100.0
                truth_trough_ratio = float(p.min()) / float(t.min())
                anchor_ok = (
                    median_ape <= MODEL_GATE_TRUTH_MEDIAN_APE_MAX
                    and truth_trough_ratio >= MODEL_GATE_TROUGH_FRACTION
                )
                entry["truth_median_ape"] = round(median_ape, 2)
                entry["truth_trough_ratio"] = round(truth_trough_ratio, 3)
            else:
                # Live anchor — no truth exists yet; the trailing week is
                # the only reference.
                anchor_ok = (
                    trough_ratio >= MODEL_GATE_TROUGH_FRACTION
                    and MODEL_GATE_LEVEL_RATIO_MIN <= level_ratio <= MODEL_GATE_LEVEL_RATIO_MAX
                )
            entry["ok"] = anchor_ok
            verdicts.append(entry)
        except Exception as e:
            log.warning(
                "model_gate_error",
                region=region,
                anchor=str(anchor),
                error=str(e),
            )
            verdicts.append({"anchor": anchor.isoformat(), "error": str(e)})

    judged = [v for v in verdicts if "ok" in v]
    if not judged:
        return {"passed": True, "skipped": "insufficient_history", "anchors": verdicts}

    result = {"passed": _gate_decision(judged), "anchors": verdicts}
    passed = result["passed"]
    if passed:
        log.info("model_gate_passed", region=region, model="xgboost", anchors=verdicts)
    else:
        log.warning("model_gate_rejected", region=region, model="xgboost", anchors=verdicts)
    return result


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
    seed_frame: pd.DataFrame | None = None,
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
                model, featured, future_df, horizon, seed_frame=seed_frame
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


def _baseline_substitution(
    region: str, demand_df: pd.DataFrame, horizon_h: int
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Seasonal-naive series for a region whose model loses to it, or None.

    Returns None — meaning "keep the model" — for every failure: the flag is
    off, no skill signal, too little history, or a deficit inside the noise
    band. That asymmetry is deliberate. Substituting when we should not is
    worse than the reverse, because the model is right on 35 of 44 regions
    and a bug here would replace all of them.

    The comparison is like-for-like on purpose. Both sides are measured over
    the drift instrument's own 7-day window: an earlier version of this
    analysis compared a 30-day baseline against a 7-day model and concluded
    the model won at 48h/72h, which reversed once the windows matched.
    """
    from config import feature_enabled

    if not feature_enabled("baseline_substitution"):
        return None
    try:
        from data.redis_client import redis_get, redis_key
        from models.skill import (
            SEASONAL_NAIVE_LAG_H,
            seasonal_naive_forecast,
            should_serve_baseline,
            skill_payload,
        )

        if demand_df is None or demand_df.empty or "demand_mw" not in demand_df.columns:
            return None
        d = demand_df.dropna(subset=["demand_mw"]).copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        # asfreq exposes gaps as NaN. Without it the lag reaches across a gap
        # and compares hours that are days apart, flattering the baseline —
        # which here would mean substituting a model that was fine.
        series = d.sort_values("timestamp").set_index("timestamp").asfreq("h")["demand_mw"]
        window = series[series.index > series.index.max() - pd.Timedelta(days=7)]
        y = window.to_numpy(dtype=float)
        if y.size <= SEASONAL_NAIVE_LAG_H:
            return None

        horizon = redis_get(redis_key(f"drift_horizon:{region}"))
        block = (((horizon or {}).get("models") or {}).get("ensemble") or {}).get("24h") or {}
        model_mape = block.get("rolling_mape_7d")
        if model_mape is None:
            return None

        # One definition of this block, in models.skill — an inline copy here
        # had already drifted from it (no `beats_baseline`, no non-finite
        # guards) while `should_serve_baseline` consumed it regardless.
        skill_block = skill_payload(float(model_mape), y, window_days=7)
        serve, reason = should_serve_baseline(skill_block)
        skill_block["decision"] = reason
        if not serve:
            return None

        values = seasonal_naive_forecast(series.to_numpy(dtype=float), horizon_h)
        if values.size != horizon_h or not np.isfinite(values).all():
            log.warning("baseline_substitution_unusable", region=region)
            return None
        log.warning("baseline_substituted", region=region, **skill_block)
        return values, skill_block
    except Exception as exc:  # pragma: no cover — never block a forecast
        log.warning("baseline_substitution_failed", region=region, error=str(exc))
        return None


#: Rolling depth of the shadow-weight record window. Matches
#: ``drift.DEFAULT_MAX_RECORDS`` (30 days hourly) so both arms are graded over
#: the same span; #478 asks for >=14 days before evaluating.
_SHADOW_MAX_RECORDS = 720


def _write_shadow_weights(
    region: str,
    predictions_by_model: dict[str, np.ndarray],
    model_mapes_shadow: dict[str, float | None] | None,
    served_weights: dict[str, float] | None,
    rows: list[dict[str, Any]],
    demand_df: Any,
) -> bool:
    """Record the not-served weighting alongside the served one (#478).

    #451 proved the WAPE half of the smoothed-weights question with a replay and
    could not prove the bias half, because a replayed vintage over-forecasts by
    ~6% in the *control* arm — a harness whose control fails a constraint cannot
    certify the treatment against it. This closes that by measuring both arms on
    production forecasts instead of replayed ones.

    Cheap by construction: both arms consume the same ``predictions_by_model``
    arrays, so the marginal cost over the served blend is one weighted sum.

    **Enrichment only.** Every failure path returns ``False`` and leaves the
    served forecast — already in Redis by the time this runs — untouched.
    Returns whether a shadow payload was written.
    """
    from config import feature_enabled
    from data.redis_client import persist, redis_get, redis_key
    from models.drift import build_records_from_actuals
    from models.ensemble import ensemble_combine, resolve_ensemble_weights
    from models.shadow_eval import regrade_records as regrade_shadow_records

    if len(predictions_by_model) < 2 or not served_weights:
        return False
    try:
        shadow_input = {
            name: float(v)
            for name, v in (model_mapes_shadow or {}).items()
            if name in predictions_by_model and v is not None and np.isfinite(v) and v > 0
        }
        # A shadow arm missing any member would fall back to equal weights and
        # measure "equal vs cubed" rather than the question asked. Skip instead:
        # a missing comparison is honest, a mislabelled one is not.
        if len(shadow_input) != len(predictions_by_model):
            log.info(
                "shadow_weights_incomplete",
                region=region,
                have=sorted(shadow_input),
                missing=sorted(set(predictions_by_model) - set(shadow_input)),
            )
            return False

        shadow_weights, shadow_rule = resolve_ensemble_weights(
            list(predictions_by_model), shadow_input
        )
        if shadow_rule != "inverse_mape_cubed":
            return False
        shadow_preds = np.maximum(ensemble_combine(predictions_by_model, shadow_weights), 0.0)

        key = redis_key(f"shadow_weights:{region}")
        previous = redis_get(key)
        previous = previous if isinstance(previous, dict) else None

        # Grade the PREVIOUS payload against actuals that have landed since,
        # reusing the drift primitives so both arms are graded by identical
        # code — including the "most recent matchable hour" rule and its
        # lead-hours bookkeeping (P2-19).
        records = list((previous or {}).get("records") or [])
        if previous is not None and demand_df is not None and not demand_df.empty:
            try:
                df = demand_df.copy()
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.dropna(subset=["demand_mw"])
                actuals = {
                    r["timestamp"].isoformat(): float(r["demand_mw"])
                    for _, r in df.iterrows()
                    if np.isfinite(r["demand_mw"]) and float(r["demand_mw"]) > 0
                }
                # Settled-grade the stored window (#541), exactly as
                # ``write_drift_metrics`` does for drift. Each record froze
                # ``actual`` at the tick that created it — a PRELIMINARY EIA
                # value, 15-70% wrong for high-revision BAs. Without this the
                # window stays prediction-vs-preliminary forever, which is what
                # made IID read +86.49% here against +2.8% from drift on the
                # same forecasts, same hours, byte-identical predictions.
                # Re-running is a no-op, so the corrupt history self-heals.
                records, regrade_stats = regrade_shadow_records(records, actuals)
                if regrade_stats.get("n_regraded"):
                    log.info("shadow_weights_regraded", region=region, **regrade_stats)

                graded = build_records_from_actuals(previous, actuals)
                if graded:
                    entry: dict[str, Any] = {"lead_hours": None}
                    for arm in ("served", "shadow"):
                        rec = graded.get(arm)
                        if rec is None:
                            entry = {}
                            break
                        entry["timestamp"] = rec.timestamp
                        entry["actual"] = rec.actual
                        entry["lead_hours"] = rec.lead_hours
                        entry[f"{arm}_predicted"] = rec.predicted
                    if entry:
                        records.append(entry)
            except Exception as exc:  # pragma: no cover — grading is advisory
                log.warning("shadow_weights_grade_failed", region=region, error=str(exc))

        payload = {
            "region": region,
            "scored_at": datetime.now(UTC).isoformat(),
            "served_weights": {k: round(v, 4) for k, v in served_weights.items()},
            "shadow_weights": {k: round(v, 4) for k, v in shadow_weights.items()},
            "shadow_arm": "raw" if feature_enabled("smoothed_ensemble_weights") else "ewma",
            "members": sorted(predictions_by_model),
            # Same row shape the drift primitives expect, carrying exactly two
            # series so nothing else can be mistaken for a model.
            "forecasts": [
                {
                    "timestamp": row["timestamp"],
                    "served": float(row["ensemble"]),
                    "shadow": float(shadow_preds[i]),
                }
                for i, row in enumerate(rows)
                if "ensemble" in row and i < len(shadow_preds)
            ],
            "records": records[-_SHADOW_MAX_RECORDS:],
        }
        persist(redis_key(f"shadow_weights:{region}"), payload, ttl=REDIS_TTL)
        # Log the SUCCESS path, not only the skips. Without this the strongest
        # available evidence that the shadow is running is the *absence* of
        # ``shadow_weights_incomplete`` across 51 BAs — which is indistinguishable
        # from the phase never executing at all. That is the "configured and
        # inert" pattern this project keeps finding (docs/monitoring/README.md),
        # and #478 cannot be evaluated from a key nobody can confirm is being
        # written. ``n_records`` is the one to watch: it must climb toward the
        # 14 days #478 asks for, and a flat count means grading is failing while
        # the write succeeds.
        log.info(
            "shadow_weights_written",
            region=region,
            members=sorted(predictions_by_model),
            n_records=len(payload["records"]),
            n_forecast_rows=len(payload["forecasts"]),
            shadow_arm=payload["shadow_arm"],
        )
        return True
    except Exception as exc:  # pragma: no cover — enrichment, never fatal
        log.warning("shadow_weights_failed", region=region, error=str(exc))
        return False


#: #559: how many BAs have already run a second recursion this tick. Regions are
#: scored concurrently in one process, so this is module-level and guarded; it
#: resets naturally because every tick is a fresh job process, exactly like
#: ``data.eia_client._EIACircuitBreaker``. ``reset`` exists for tests.
_seed_shadow_budget_lock = threading.Lock()
_seed_shadow_spent = 0


def _claim_seed_shadow_budget() -> bool:
    """Take one slot of this tick's second-recursion budget, or decline.

    Capping rather than only gating, per CLAUDE.md's "bound what ONE RUN can
    cost". The gate is data-dependent and could admit the whole fleet on a bad
    EIA day; shedding is whole-BA, so an unbounded enrichment buys shadow data
    with other regions' forecasts.
    """
    global _seed_shadow_spent
    with _seed_shadow_budget_lock:
        if _seed_shadow_spent >= config.SEED_SHADOW_MAX_REGIONS_PER_TICK:
            return False
        _seed_shadow_spent += 1
        return True


def reset_seed_shadow_budget() -> None:
    """Reset the per-tick budget. For tests and explicit run boundaries."""
    global _seed_shadow_spent
    with _seed_shadow_budget_lock:
        _seed_shadow_spent = 0


def _is_seed_shadow_audit_region(region: str, origin: pd.Timestamp) -> bool:
    """Pick one region per hour to shadow even though it should be identical.

    The gate below skips any BA whose seed has no hole, on the grounds that the
    two arms are then byte-identical. That is a *claim*, and a gate that quietly
    starts skipping everything is indistinguishable from one that is working —
    the "configured and inert" pattern this project keeps rediscovering, most
    recently the parity fixture that agreed by construction (#559).

    So one region per tick is shadowed against the prediction that it will NOT
    diverge, and a nonzero divergence there is an alarm about the gate rather
    than a finding about the seed. Chosen by clock arithmetic rather than at
    random so it is reproducible from the log line alone, and so each region
    comes round about every two days without any cross-region coordination —
    regions are scored concurrently and cannot see each other's choices.
    """
    import config

    regions = sorted(config.REGION_COORDINATES)
    if region not in regions:
        return False
    hours = int(pd.Timestamp(origin).value // 3_600_000_000_000)
    return regions.index(region) == hours % len(regions)


def _write_seed_shadow(
    *,
    region: str,
    model: Any,
    featured: pd.DataFrame,
    future_df: pd.DataFrame,
    horizon: int,
    forecast_start: pd.Timestamp,
    served_preds: Any,
    rows: list[dict],
    demand_df: pd.DataFrame | None,
    xgboost_weight: float | None,
) -> bool:
    """#559: record the temporal-seed arm beside the served positional one.

    Observation only — the shadow series is written to its own key and **never**
    into ``redis_payload``, for the reason spelled out at the call site: the
    drift primitives treat every numeric key in a forecast row as a model, so a
    shadow series added there would silently acquire drift records and a place
    in the published rolling MAPE.

    Two arms, one difference: both are XGBoost through the same recursion on the
    same frame, and only the seed indexing moves. That keeps this commensurable
    with the offline replay in ``docs/POSITIONAL_LAG_SEED_STUDY.md``, which
    compared the same two things. The served *headline* is the ensemble, but its
    delta is exactly ``xgboost_weight × delta_xgboost`` because the blend is a
    weighted sum, so the weight is recorded rather than a third arm computed.

    **This does not decide the flag and cannot.** At the natural rate gaps
    occur, a decisive accuracy verdict is 1.2-6.6 years away. It is here to
    answer the questions that *are* answerable before a rollout: does the
    temporal path run clean against real production frames, what does it
    actually cost, and does its divergence match the 2.1-2.7% the offline replay
    predicted.
    """
    from config import feature_enabled

    if not feature_enabled("temporal_ar_seed_shadow"):
        return False

    try:
        from data.feature_engineering import (
            positional_seed_matches_hours,
            seed_divergence_reason,
        )
        from data.redis_client import persist, redis_get, redis_key
        from models.drift import build_records_from_actuals
        from models.shadow_eval import regrade_records as regrade_shadow_records

        key = redis_key(f"seed_shadow:{region}")
        previous = redis_get(key)
        previous = previous if isinstance(previous, dict) else None

        seed_ts = featured.get("timestamp")
        identical = positional_seed_matches_hours(seed_ts, forecast_start)
        # #624: "diverges" collapsed two situations that are not equally
        # informative. A seed that reaches ``origin - 1h`` but has a hole
        # further back gives a correctly-sized array and clean evidence about
        # temporal indexing; a seed that stops SHORT of it under-sizes the
        # array, so the tail of the horizon is silently discarded and the
        # observation is partly about that bug. The gate selects for the
        # second case by construction, so recording which one this was is what
        # lets the comparison be stratified later instead of thrown away.
        gate_reason, seed_tail_gap_h = seed_divergence_reason(seed_ts, forecast_start)
        audit = identical and _is_seed_shadow_audit_region(region, forecast_start)

        shadow_preds: Any = None
        divergence_pct: float | None = None
        wanted = (not identical) or audit
        # Out of this tick's budget is recorded, never silent: a missing
        # observation that reads as "no gap" would bias the sample toward
        # quiet ticks.
        budget_declined = wanted and not _claim_seed_shadow_budget()
        if budget_declined:
            log.info("seed_shadow_budget_exhausted", region=region)
        if wanted and not budget_declined:
            shadow_preds = _predict_xgboost_with_recursive_autoregressive(
                model, featured, future_df, horizon, force_temporal=True
            )
            served = np.asarray(served_preds, dtype=float)[: len(shadow_preds)]
            shadow = np.asarray(shadow_preds, dtype=float)[: len(served)]
            ok = np.isfinite(served) & np.isfinite(shadow)
            denom = float(np.abs(served[ok]).sum())
            divergence_pct = (
                float(100 * np.abs(shadow[ok] - served[ok]).sum() / denom) if denom else None
            )
            if audit:
                # The gate said these arms cannot differ. If they do, the gate is
                # wrong and every BA it skipped is an unrecorded observation.
                if divergence_pct:
                    log.warning(
                        "seed_shadow_audit_diverged",
                        region=region,
                        divergence_pct=round(divergence_pct, 6),
                        origin=forecast_start.isoformat(),
                    )
                else:
                    log.info("seed_shadow_audit_ok", region=region)

        # Grade the PREVIOUS payload whether or not a new arm was computed —
        # actuals land continuously, and skipping this on a quiet tick would
        # leave records permanently unresolved.
        records = list((previous or {}).get("records") or [])
        if previous is not None and demand_df is not None and not demand_df.empty:
            try:
                df = demand_df.copy()
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.dropna(subset=["demand_mw"])
                actuals = {
                    r["timestamp"].isoformat(): float(r["demand_mw"])
                    for _, r in df.iterrows()
                    if np.isfinite(r["demand_mw"]) and float(r["demand_mw"]) > 0
                }
                # Settled-grade the window (#541). A record froze ``actual`` at
                # the tick that created it — a preliminary EIA value that is
                # 15-70% wrong on high-revision BAs. IID read +86.49% bias
                # against drift's +2.8% on byte-identical predictions for
                # exactly this reason. Re-running is a no-op, so a corrupt
                # window self-heals rather than needing a migration.
                records, regrade_stats = regrade_shadow_records(records, actuals)
                if regrade_stats.get("n_regraded"):
                    log.info("seed_shadow_regraded", region=region, **regrade_stats)

                graded = build_records_from_actuals(previous, actuals)
                if graded:
                    entry: dict[str, Any] = {"lead_hours": None}
                    for arm in ("served", "shadow"):
                        rec = graded.get(arm)
                        if rec is None:
                            # All arms or none: a record with one side missing
                            # would bias whichever arm survived.
                            entry = {}
                            break
                        entry["timestamp"] = rec.timestamp
                        entry["actual"] = rec.actual
                        entry["lead_hours"] = rec.lead_hours
                        entry[f"{arm}_predicted"] = rec.predicted
                    if entry:
                        records.append(entry)
            except Exception as exc:  # pragma: no cover — grading is advisory
                log.warning("seed_shadow_grade_failed", region=region, error=str(exc))

        forecasts: list[dict] = []
        if shadow_preds is not None:
            served = np.asarray(served_preds, dtype=float)
            forecasts = [
                {
                    "timestamp": row["timestamp"],
                    "served": float(served[i]),
                    "shadow": float(shadow_preds[i]),
                }
                for i, row in enumerate(rows)
                if i < len(shadow_preds)
                and i < len(served)
                and np.isfinite(served[i])
                and np.isfinite(shadow_preds[i])
            ]

        payload = {
            "region": region,
            "scored_at": datetime.now(UTC).isoformat(),
            "origin": forecast_start.isoformat(),
            "arms": {"served": "positional_seed", "shadow": "temporal_seed"},
            # Why this tick did or did not compute a second arm. Without it a
            # quiet key is ambiguous between "no gaps" and "not running".
            "gate": gate_reason,
            "seed_tail_gap_h": seed_tail_gap_h,
            "audited": bool(audit),
            "computed": shadow_preds is not None,
            "budget_declined": budget_declined,
            "divergence_pct": round(divergence_pct, 6) if divergence_pct is not None else None,
            # The served headline is the ensemble; its delta is this weight
            # times the XGBoost delta, because the blend is a weighted sum.
            "xgboost_weight": round(xgboost_weight, 4) if xgboost_weight is not None else None,
            "forecasts": forecasts,
            "records": records[-_SHADOW_MAX_RECORDS:],
        }
        # Keep the previous forecast rows when this tick computed nothing, so
        # the next tick still has something to grade against.
        if not forecasts and previous is not None:
            payload["forecasts"] = list(previous.get("forecasts") or [])

        # Nothing computed and nothing graded means nothing to store. Skipping
        # the write keeps 44-of-51 never-gapping BAs from re-persisting an empty
        # payload every hour. The LOG below still fires on every invocation, so
        # "ran, nothing to do" stays distinguishable from "never ran" — which is
        # the property that matters and the reason this is safe to skip.
        wrote = bool(payload["forecasts"] or payload["records"])
        if wrote:
            persist(key, payload, ttl=REDIS_TTL)
        # Log the success path, not only the skips: the absence of a warning
        # across 51 BAs is indistinguishable from the phase never running.
        log.info(
            "seed_shadow_written",
            region=region,
            gate=payload["gate"],
            seed_tail_gap_h=payload["seed_tail_gap_h"],
            audited=payload["audited"],
            computed=payload["computed"],
            budget_declined=budget_declined,
            divergence_pct=payload["divergence_pct"],
            n_records=len(payload["records"]),
            n_forecast_rows=len(payload["forecasts"]),
            persisted=wrote,
        )
        return True
    except Exception as exc:  # pragma: no cover — enrichment, never fatal
        log.warning("seed_shadow_failed", region=region, error=str(exc))
        return False


def predict_and_write_forecast(
    data: RegionData,
    models: dict[str, Any] | None,
    model_mapes: dict[str, float | None] | None = None,
    model_metrics: dict[str, dict[str, float]] | None = None,
    model_mapes_shadow: dict[str, float | None] | None = None,
    weight_input: dict[str, str] | None = None,
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
    from models.ensemble import ensemble_combine, resolve_ensemble_weights

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
        # ADR-009 seam 2 (SARIMAX consistency): forecast_start must resolve
        # from the SAME frame the features anchored on, or substituted hours
        # land outside the Kalman gap window and get re-forecast.
        with substep("resolve_start"):
            forecast_start = _resolve_forecast_start(featured, data.anchor_frame, region=region)
            # #559: an origin bridged past the tail of ``featured`` and the seed
            # that can serve it are resolved together — see ``_ar_seed_for_origin``.
            # ``seed_frame`` is None on every unbridged tick, which is all of them
            # while ``temporal_ar_seed`` is off.
            forecast_start, seed_frame = _ar_seed_for_origin(
                featured, data.anchor_frame, forecast_start, region=region
            )

        # #537: a forecast origin must never go BACKWARDS. It is recomputed from
        # scratch every tick with no memory of the last one, so when EIA retracts
        # hours it had already published — measured: 25 of LGEE's 26 regressed
        # ticks carried fewer frame hours than the payload spanned — the anchor
        # collapses to just before the retracted block and this tick would
        # publish an origin OLDER than one already served. Downstream reads
        # ``forecasts[0]`` as the current nowcast, so LGEE's 2026-08-14 regression
        # relabelled 40-to-63-hour-ahead rows as one-hour-ahead for 24 ticks.
        #
        # Keep the payload already in Redis rather than overwriting it. It is
        # strictly the more current of the two and still covers the horizon; the
        # alternative — clamping the origin forward — would forecast from an hour
        # whose antecedent demand the frame no longer holds. ``ok=True`` because
        # a live, newer payload is not a failed region (the deadline-shed branch
        # in ``scoring_job`` reasons the same way); the refusal is carried in
        # ``details`` and on a WARNING line instead.
        prior_origin = data.previous_forecast_origin
        if prior_origin is not None and forecast_start < prior_origin:
            log.warning(
                "forecast_origin_regressed",
                region=region,
                resolved_start=forecast_start.isoformat(),
                served_origin=prior_origin.isoformat(),
                regression_hours=int((prior_origin - forecast_start).total_seconds() // 3600),
            )
            # #559: the seed shadow lives ~300 lines below and is unreachable
            # from here, so an origin-regressed tick used to leave NO trace in
            # the shadow's record at all — the region was simply absent, with
            # no error and no warning. That is missing-not-at-random: the guard
            # fires when EIA withdraws published hours, which is gap-adjacent,
            # so the absences correlate with the very condition the shadow
            # exists to observe. LGEE regressed for 24 consecutive ticks during
            # #537; that whole episode would have been an unexplained hole.
            #
            # It records the SKIP rather than computing an arm. Production did
            # not forecast on this tick, so there is no served prediction to
            # compare against, and manufacturing one would put "what production
            # would have done" in the same sample as "what production did".
            # Coverage you can measure beats a counterfactual you cannot.
            from config import feature_enabled

            if feature_enabled("temporal_ar_seed_shadow"):
                log.info(
                    "seed_shadow_skipped",
                    region=region,
                    reason="origin_regressed",
                    regression_hours=int((prior_origin - forecast_start).total_seconds() // 3600),
                )
            return PhaseResult(
                region=region,
                ok=True,
                details={
                    "skipped": "origin_regressed",
                    "resolved_start": forecast_start.isoformat(),
                    "served_origin": prior_origin.isoformat(),
                },
            )

        # #547: record what this run anchored on, while the frames that prove it
        # are still in scope. Deliberately AFTER the #537 guard: on a regressed
        # origin no payload is written, so there is nothing to stamp and an
        # anchor computed here would describe a forecast that was never served.
        #
        # The two are closely related. #537's regression was EIA publishing 19
        # hours as placeholders (``D == DF``) and then withdrawing them — so on
        # exactly those ticks this block records ``anchor_was_placeholder=True``,
        # which is the signal that diagnosis had to be reconstructed without.
        anchor = _anchor_provenance(data, featured, forecast_start)

        # Pass the raw weather DataFrame so the future-feature builder can
        # overlay actual Open-Meteo forecast values (next ~16 days) onto
        # the climatology baseline. See ``_overlay_weather_forecast``.
        # `future_frame` is the wrapper total; the three `frame_*` sub-steps
        # inside it partition it.
        with substep("future_frame"):
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
            # Per-model, not one `predict` total: the three have completely
            # different cost shapes (XGBoost recurses 384 steps, Prophet and
            # SARIMAX forecast the gap+720 in one call), so a combined number
            # would hide which one to go after.
            with substep(f"predict_{name}"):
                preds = _predict_one(
                    name,
                    model,
                    featured,
                    future_df,
                    FORECAST_HORIZON_HOURS,
                    start_ts=forecast_start,
                    seed_frame=seed_frame,
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
                # P2-16 (#273): the weighting rule is now shared with the
                # training job via ``resolve_ensemble_weights`` — cubed weights
                # only when EVERY predicting model has a usable MAPE, equal
                # weights otherwise. Membership still differs by necessity
                # (training has holdout payloads, scoring has forecast arrays),
                # so the composition actually served is recorded on the payload
                # and compared against the persisted one below.
                ensemble_weights, weight_rule = resolve_ensemble_weights(
                    list(predictions_by_model), mape_input
                )
                if weight_rule == "equal" and mape_input:
                    log.info(
                        "scoring_ensemble_equal_weights_fallback",
                        region=region,
                        have_mape=sorted(mape_input.keys()),
                        missing_mape=sorted(set(predictions_by_model) - set(mape_input)),
                    )
                # Floored inputs make the weighted blend non-negative already;
                # clip again so the guarantee survives any future change to
                # ensemble_combine. (#281)
                with substep("ensemble"):
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
                with substep("horizon_guard"):
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
        # 720 rows x (timestamp isoformat + one float per model) built one dict
        # at a time. Cheap per row, but it is the only O(horizon) Python loop
        # left in the phase after the models return, so it is worth naming
        # rather than assuming.
        fl: list[dict[str, Any]] = []
        with substep("build_rows"):
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
            # #547. TOP LEVEL, never inside a ``forecasts`` row: the drift
            # extractor treats every extra numeric key on a row as a model
            # (see the note on the horizon-guard block below), so a per-row
            # ``anchor_mw`` would acquire its own drift records, a Models-tab
            # entry, and a place in the rolling MAPE the visibility gate reads.
            # One anchor per run, so per-row would be redundant regardless.
            "anchor": anchor,
        }

        # Where the model measurably loses to "yesterday, same hour", serve the
        # baseline instead. The per-model rows are LEFT INTACT: the substitution
        # replaces the headline series a reader consumes, and hiding the models
        # would destroy the evidence for why it happened. `served_series` is the
        # disclosure — nothing may present this as a model forecast.
        with substep("baseline_substitution"):
            substitution = _baseline_substitution(region, data.demand_df, len(fl))
        if substitution is not None:
            values, skill_block = substitution
            for i, row in enumerate(fl):
                row["predicted_demand_mw"] = float(values[i])
                row["baseline"] = float(values[i])
            redis_payload["served_series"] = "seasonal-naive"
            redis_payload["served_reason"] = skill_block["decision"]
            redis_payload["skill"] = skill_block
        else:
            redis_payload["served_series"] = "model"
        if ensemble_weights is not None:
            redis_payload["ensemble_weights"] = {
                k: round(v, 4) for k, v in ensemble_weights.items()
            }
            # P2-16 (#273): publish the composition actually SERVED, and warn
            # when the persisted holdout metric describes a different one. The
            # metric and the served blend are computed from different inputs
            # by necessity, so they can legitimately differ — what must never
            # happen again is differing silently under one name.
            # #514: the INPUT is published too, not just the membership and
            # the rule. Scoring may weight by the EWMA while the persisted
            # metric weights by within-window holdout MAPEs — legitimate, and
            # by design, but a difference the two fields below are the only way
            # to see. Sorted+deduped because the per-model fallback can leave a
            # fleet genuinely mixed on the first run after a flag flip.
            served_inputs = sorted(
                {(weight_input or {}).get(m) or "unknown" for m in ensemble_weights}
            )
            redis_payload["ensemble_composition"] = {
                "members": sorted(ensemble_weights),
                "weight_rule": weight_rule,
                "weight_input": served_inputs,
            }
            persisted = ((model_metrics or {}).get("ensemble") or {}) if model_metrics else {}
            persisted_members = persisted.get("members")
            persisted_input = persisted.get("weight_input")
            members_differ = bool(
                persisted_members and sorted(persisted_members) != sorted(ensemble_weights)
            )
            # A basis mismatch counts even when membership matches — that is
            # exactly the P2-16 shape a flag flip would otherwise reintroduce
            # silently, since the members are identical and only the numbers
            # feeding them change.
            input_differs = bool(persisted_input and served_inputs != sorted({persisted_input}))
            if members_differ or input_differs:
                log.info(
                    "ensemble_composition_divergence",
                    region=region,
                    served=sorted(ensemble_weights),
                    persisted_metric=sorted(persisted_members or []),
                    served_rule=weight_rule,
                    persisted_rule=persisted.get("weight_rule"),
                    served_input=served_inputs,
                    persisted_input=persisted_input,
                    differs=("members" if members_differ else "")
                    + ("input" if input_differs else ""),
                )
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
        # The 720-row payload is the largest single write the job makes, and it
        # is JSON-serialised inside persist() — so this covers serialisation,
        # not just network time.
        with substep("redis_write"):
            persist(
                redis_key(f"forecast:{region}:1h"),
                redis_payload,
                ttl=REDIS_TTL,
            )
        # #478: the shadow weighting. Blends the SAME per-model forecasts under
        # the arm that is not being served, so the two differ only in weights —
        # and writes it to its own key, never into ``redis_payload``. That last
        # part is deliberate: ``drift.extract_one_hour_ahead_predictions``
        # iterates every numeric key in a forecast row, so a shadow series added
        # there would silently acquire drift records, a Models-tab entry and a
        # place in the rolling MAPE the visibility gate reads. The served
        # payload is already persisted above and is not touched here.
        #
        # Guarded at the CALL SITE as well as inside the helper. The phase's
        # outer ``except`` returns ok=False, and #267 makes ``ok`` the signal
        # for whether this region was scored — so an exception escaping here
        # would report a region as failed whose forecast is already in Redis.
        # That is the #268 mistake inverted, and a test pins it.
        try:
            with substep("shadow_weights"):
                _write_shadow_weights(
                    region=region,
                    predictions_by_model=predictions_by_model,
                    model_mapes_shadow=model_mapes_shadow,
                    served_weights=ensemble_weights,
                    rows=fl,
                    demand_df=data.demand_df,
                )
        except Exception as exc:  # pragma: no cover — belt and braces
            log.warning("shadow_weights_call_failed", region=region, error=str(exc))

        # #559: the seed shadow. Unlike the weights shadow above, this one runs
        # a SECOND XGBoost recursion, so it is gated to the BAs where the two
        # seed conventions can actually differ — a hole inside the 168h lookback
        # — plus one rotating BA that should be identical, as a live check that
        # the gate is still deciding. On 2026-08-20 that was 3 of 51 BAs, about
        # 3 CPU-seconds; ungated it would be roughly +380 CPU-s on a job whose
        # worst recent tick used 1155s of its 1800s budget.
        #
        # Same call-site guard as above and for the same reason: the forecast is
        # already in Redis, and an exception escaping here would report a scored
        # region as failed (#268 inverted).
        try:
            with substep("seed_shadow"):
                _write_seed_shadow(
                    region=region,
                    model=(models or {}).get("xgboost"),
                    featured=featured,
                    future_df=future_df,
                    horizon=FORECAST_HORIZON_HOURS,
                    forecast_start=forecast_start,
                    served_preds=predictions_by_model.get("xgboost"),
                    rows=fl,
                    demand_df=data.demand_df,
                    xgboost_weight=(ensemble_weights or {}).get("xgboost"),
                )
        except Exception as exc:  # pragma: no cover — belt and braces
            log.warning("seed_shadow_call_failed", region=region, error=str(exc))

        # #127: the what-if grid. Computed HERE rather than as its own phase
        # because it needs `future_df`, and rebuilding that costs ~4.3s/BA —
        # more than the grid itself. Enrichment only: it runs after the
        # forecast has already landed in Redis, and every failure path leaves
        # the forecast write untouched.
        with substep("scenario_grid"):
            scenario_written = _write_scenario_grid(
                region=region,
                featured=featured,
                future_df=future_df,
                models=models,
                baseline=predictions_by_model.get("xgboost"),
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
                "scenario_grid": scenario_written,
                # #547: flat scalars so they land as queryable jsonPayload.*
                # on the per-region completion log (#306). This is the signal
                # that says the instrument is running, per BA, per tick.
                "anchor_was_placeholder": anchor["anchor_was_placeholder"],
                "anchor_conditioned": anchor["anchor_conditioned"],
            },
        )
    except Exception as e:
        log.warning("job_forecast_write_failed", region=region, error=str(e))
        return PhaseResult(region=region, ok=False, error=str(e))


def _write_scenario_grid(
    region: str,
    featured: pd.DataFrame,
    future_df: pd.DataFrame,
    models: dict[str, Any],
    baseline: np.ndarray | None,
) -> bool:
    """Compute and persist ``gridpulse:scenario_grid:{region}`` (#127).

    **Enrichment only, and fail-open by construction.** The forecast has
    already been written by the time this runs; every return path below
    leaves it alone. A region with no grid falls back to the analytical
    heuristic in the web tier, which is what shipped before #127 — a
    degraded simulator, not a missing forecast.

    The baseline passed in is XGBoost's, and the grid re-runs XGBoost
    through ``_predict_xgboost_with_recursive_autoregressive`` — the same
    recursive protocol, on the same ``future_df``, seeded from the same
    ``featured``. Only the weather differs between the two sides of the
    ratio. Using the ensemble baseline instead would mean re-running Prophet
    81 times per region, which is ~14x the cost for a second opinion on a
    *delta* whose weather sensitivity lives almost entirely in XGBoost's
    engineered features.

    Returns:
        True when a grid was written.
    """
    from config import SCENARIO_GRID_HORIZON_HOURS, feature_enabled

    if not feature_enabled("scenario_grid"):
        return False

    model = (models or {}).get("xgboost")
    if model is None or baseline is None:
        return False

    horizon = min(SCENARIO_GRID_HORIZON_HOURS, len(future_df), len(baseline))
    if horizon < 2:
        return False

    try:
        from data.feature_engineering import batched_recursive_autoregressive_forecast
        from data.redis_client import persist, redis_key
        from models.xgboost_model import predict_xgboost
        from simulation.scenario_grid import build_scenario_grid

        # Batched: one predict call per STEP across all 80 variants, not one
        # per variant. The cell-at-a-time version cost 2.7x tick runtime and
        # was reverted (#462); the per-scenario chaining is unchanged and its
        # parity with the single-frame SSOT is a differential test.
        def forecaster(frames: list[pd.DataFrame]) -> list[np.ndarray]:
            return batched_recursive_autoregressive_forecast(
                model,
                featured["demand_mw"],
                frames,
                predict_xgboost,
                seed_timestamps=featured.get("timestamp"),
            )

        payload = build_scenario_grid(
            featured=featured,
            future_df=future_df.iloc[:horizon],
            baseline=np.asarray(baseline, dtype=float)[:horizon],
            forecaster=forecaster,
            horizon=horizon,
        )
        payload["generated_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        payload["region"] = region

        persist(redis_key(f"scenario_grid:{region}"), payload, ttl=REDIS_TTL)
        return True
    except Exception as e:
        # Deliberately broad: this is the last thing the forecast phase does,
        # and nothing it can raise is worth failing an already-written
        # forecast over.
        log.warning("scenario_grid_write_failed", region=region, error=str(e))
        return False


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


def forecast_payload_origin(payload: dict[str, Any] | None) -> pd.Timestamp | None:
    """Origin of a forecast payload — ``forecasts[0]["timestamp"]``, or None.

    The same row ``models.drift._lead_hours`` measures lead against, so a lead
    recovered from the drift log and this value are the same quantity (#537).
    """
    if not isinstance(payload, dict):
        return None
    rows = payload.get("forecasts") or []
    if not rows:
        return None
    try:
        return pd.Timestamp(rows[0].get("timestamp"))
    except (AttributeError, TypeError, ValueError):
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
    from data.redis_client import RedisReadError, redis_key, redis_set
    from models.drift import (
        _normalize_ts as _normalize_drift_ts,
    )
    from models.drift import (
        build_records_from_actuals,
        compute_drift_payload,
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

        try:
            existing = _read_window_strict(redis_key(f"drift:{region}"))
        except RedisReadError as exc:
            # #313: a nil-read-during-outage must not rebuild the window —
            # post-#318 these records carry re-graded history that a fresh
            # window cannot recompute. One skipped tick beats 720 lost records.
            log.warning("drift_history_read_failed", region=region, error=str(exc))
            return PhaseResult(region=region, ok=False, error=f"history read failed: {exc}")
        existing_payload = existing if isinstance(existing, dict) else None

        # Settled-grade drift (#304 endgame): the freshly fetched, guard-
        # cleaned frame covers every stored record's target hour — re-grade
        # history against EIA's current view so the displayed metric
        # converges to prediction-vs-settled as revisions land.
        payload = compute_drift_payload(region, existing_payload, new_records, actuals=actuals)

        # Ephemeral — the caller's log line, never persisted.
        regrade_stats = payload.pop("_regrade_stats", None) or {}
        if regrade_stats.get("n_regraded"):
            log.info("drift_regraded", region=region, **regrade_stats)

        redis_set(redis_key(f"drift:{region}"), payload, ttl=DRIFT_REDIS_TTL)

        # Compact summary for the scoring-job log line.
        #
        # #170: these figures used to come from ``models_with_records[0]`` — the
        # ALPHABETICAL first model, i.e. always ``arima``, typically the weakest
        # and the one carrying the least ensemble weight. The user-facing
        # headline (Overview, ``_resolve_forecast_mape``) and the visibility gate
        # both read the **ensemble**, so the live log could not confirm the number
        # anyone actually sees: during the PR-G9 verification LDWP's arima drift
        # read 188% MAPE while the ensemble's was unobservable from outside the
        # VPC.
        #
        # Headline on the ensemble, fall back to the alphabetical first only when
        # a BA has no ensemble record, and **say which model these came from** —
        # the underlying defect was not the choice of model, it was that the line
        # never named the one it was describing.
        models_with_records = sorted(payload["models"].keys())
        headline_model = (
            "ensemble"
            if "ensemble" in payload["models"]
            else (models_with_records[0] if models_with_records else None)
        )
        headline = payload["models"].get(headline_model, {}) if headline_model else {}
        # P2-19 (#273): the lead this tick's record actually had, and how many
        # OTHER matchable hours were dropped to keep one-record-per-tick. Both
        # are logged rather than acted on — the pooled statistic still calls
        # itself 1-hour-ahead, and this is the evidence for whether it can.
        sample_record = next(iter(new_records.values()))
        n_matchable = sum(
            1
            for r in (previous_forecast.get("forecasts") or [])
            if _normalize_drift_ts(r.get("timestamp", "")) in actuals
        )
        log.info(
            "drift_updated",
            region=region,
            models=models_with_records,
            new_record_ts=sample_record.timestamp,
            lead_hours=sample_record.lead_hours,
            matchable_hours=n_matchable,
            dropped_hours=max(0, n_matchable - 1),
            headline_model=headline_model,
            rolling_smape_7d=headline.get("rolling_smape_7d"),
            rolling_mape_7d=headline.get("rolling_mape_7d"),
            n_low_actual_excluded_7d=headline.get("n_low_actual_excluded_7d"),
            # #542: the healing signal. ``n_lead_unknown_7d`` declining tick
            # over tick is re-grading no longer blanking leads; ``n_lead_
            # excluded_7d`` rising is the P2-19 filter re-engaging on records
            # that had been bypassing it. Both were already computed and
            # published in the payload — and neither was ever read, which is
            # how the defect survived. Logging them puts the counters where
            # the post-deploy check actually looks.
            n_lead_excluded_7d=headline.get("n_lead_excluded_7d"),
            n_lead_unknown_7d=headline.get("n_lead_unknown_7d"),
            # #547: the anchor-provenance instrument's own accrual.
            # ``n_anchor_unknown_7d`` falling tick over tick is the field
            # actually being recorded; it plateauing above zero once the 7d
            # window has turned over means something is not carrying it.
            # Logged, not merely published — see #542.
            n_anchor_placeholder_7d=headline.get("n_anchor_placeholder_7d"),
            n_anchor_unknown_7d=headline.get("n_anchor_unknown_7d"),
            # #512: the window's DEPTH, logged every write. A key expiry drops
            # this from 720 to 1 and nothing else in the system notices — the
            # job is stateless, so it cannot tell "history lost" from "new BA".
            # AZPS lost 30 days on 2026-08-05 and it surfaced six days later,
            # via the public API, by accident.
            n_records=headline.get("n_records"),
            # Per-model depth, because the models do NOT have to agree: a model
            # absent from a forecast row grades no record that tick, so its
            # window is shallower. #512 added the depth signal reading only the
            # alphabetical sample, which is the same #170 defect one field over.
            n_records_by_model={
                m: payload["models"][m].get("n_records") for m in models_with_records
            },
        )
        return PhaseResult(
            region=region,
            ok=True,
            details={
                "models": models_with_records,
                "new_records": len(new_records),
                "lead_hours": sample_record.lead_hours,
                "dropped_hours": max(0, n_matchable - 1),
                "total_records": sum(m["n_records"] for m in payload["models"].values()),
            },
        )
    except Exception as exc:
        log.warning("drift_write_failed", region=region, error=str(exc))
        return PhaseResult(region=region, ok=False, error=str(exc))


def condition_anchor_frame(data: RegionData) -> PhaseResult:
    """ADR-009: substitute the BA's own day-ahead forecast for the trailing
    unsettled hours of BROKEN-feed regions, on a fork of the frame.

    The study (docs/ANCHOR_CONDITIONING_STUDY.md, from real vintage data):
    broken-class anchors average 58.2% wrong where the hour-matched DF runs
    14.5% (90% win rate). Churn and bulk classes measured AGAINST
    substitution and are never conditioned; clean/unknown untouched by
    policy. Flag-dark via ``anchor_conditioning`` (byte-identical no-op off).

    The revision class comes from ``vintage_summary:{region}`` — written
    earlier THIS tick (scoring_job ordering), fail-soft read: missing or
    non-qualifying class → no conditioning.

    Never fatal; never touches ``data.demand_df`` (the fork invariant).
    """
    from config import (
        ANCHOR_CONDITIONING_CLASSES,
        ANCHOR_CONDITIONING_TRAILING_HOURS,
        feature_enabled,
    )
    from data.redis_client import redis_get, redis_key

    region = data.region
    try:
        if not feature_enabled("anchor_conditioning"):
            return PhaseResult(region=region, ok=True, details={"skipped": "flag_off"})

        summary = redis_get(redis_key(f"vintage_summary:{region}"))
        cls = (summary or {}).get("revision_class") if isinstance(summary, dict) else None
        if cls not in ANCHOR_CONDITIONING_CLASSES:
            return PhaseResult(
                region=region, ok=True, details={"skipped": f"class_{cls or 'unknown'}"}
            )

        df = data.demand_df
        if df is None or df.empty or "forecast_mw" not in df.columns:
            return PhaseResult(region=region, ok=True, details={"skipped": "no_frame_or_df"})

        conditioned = df.copy()
        n = len(conditioned)
        start = max(0, n - ANCHOR_CONDITIONING_TRAILING_HOURS)
        demand_col = conditioned.columns.get_loc("demand_mw")
        substituted = 0
        deltas: list[float] = []
        for i in range(start, n):
            df_mw = conditioned["forecast_mw"].iloc[i]
            if df_mw is None or not np.isfinite(df_mw) or df_mw <= 0:
                continue
            prior = conditioned["demand_mw"].iloc[i]
            conditioned.iloc[i, demand_col] = float(df_mw)
            substituted += 1
            if prior is not None and np.isfinite(prior) and prior > 0:
                deltas.append(abs(float(df_mw) - float(prior)))

        if not substituted:
            return PhaseResult(region=region, ok=True, details={"skipped": "no_df_values"})

        data.conditioned_demand_df = conditioned
        log.info(
            "anchor_conditioned",
            region=region,
            revision_class=cls,
            n_substituted=substituted,
            mean_abs_delta_mw=round(float(np.mean(deltas)), 1) if deltas else 0.0,
        )
        return PhaseResult(
            region=region, ok=True, details={"conditioned": substituted, "class": cls}
        )
    except Exception as exc:
        log.warning("anchor_conditioning_failed", region=region, error=str(exc))
        return PhaseResult(region=region, ok=False, error=str(exc))


def _read_window_strict(key: str) -> Any:
    """Read a stateful rolling-window payload, refusing to conflate failure
    with absence (#313).

    One retry with a short pause (the observed anomaly was transient — the
    same region read fine the next tick), then :class:`RedisReadError`
    propagates to the caller, which must FAIL its phase rather than rebuild:
    a window rebuilt from a nil-read-during-outage silently discards up to
    720 records of history. Genuinely-absent keys return ``None`` — the
    legitimate first-run path.
    """
    from data.redis_client import RedisReadError, redis_get_strict

    try:
        return redis_get_strict(key)
    except RedisReadError:
        time.sleep(0.5)
        return redis_get_strict(key)


def write_vintage_records(
    region: str,
    demand_df: pd.DataFrame,
    data: RegionData | None = None,
) -> PhaseResult:
    """Record what EIA first said about each hour, at ``gridpulse:vintage:{region}``.

    #309. The forecast anchors on the newest EIA reading, EIA revises it, and
    ``corr(revision, settled error) = 0.88`` — but we cannot study that, because
    ``gcs_store`` overwrites ``{region}/latest.parquet`` hourly and nothing keeps
    what EIA *first* said. This phase is the recorder: it pins ``first_seen_d``
    (the value the anchor actually used) against ``last_d`` (settled), so the
    revision question becomes answerable from history instead of from a live
    probe and a hypothesis.

    Capture only — reads the demand frame, writes its own key, changes no
    forecast behavior. Like drift, a failure here MUST NOT block the run: this
    is a measurement, not a critical path.

    ## The #313 defense — never rebuild first-sight history from an ambiguous read

    On 2026-07-16 prod re-pinned four regions' windows: an unexplained **nil**
    read (no error logged, no eviction, no TTL expiry, single execution) made
    the naive path treat 720 hours of accumulated ``first_seen_d`` as "no
    history" and overwrite them with current values — the one silent failure
    this recorder exists to prevent. Three rules follow:

    1. History is read with ``redis_get_strict`` (+1 retry): an infrastructure
       failure fails the phase; it never masquerades as an empty past.
    2. A ``vintage_seeded:{region}`` tombstone outlives the data key. Window
       absent while the tombstone survives ⇒ the #313 anomaly, not a first run
       ⇒ **refuse to write** and log ``vintage_window_missing_but_seeded`` at
       error level. Capture resumes next tick; one lost tick beats 720 lost
       first-sights.
    3. Writes go through ``persist`` (#268), so a dropped write fails the
       phase instead of silently diverging from what we logged.

    ## The #547 hand-off

    When ``data`` is supplied, the placeholder verdict for each captured hour
    is stashed on it as :attr:`RegionData.placeholder_by_hour` so the forecast
    phase — which runs later in the same ``_score_region`` — can record whether
    its anchor was seeded by EIA's own day-ahead value. Handed across in memory
    rather than re-read from Redis: this phase already holds the deserialised
    records, and a re-read would cost ~65KB per region per tick and could
    observe a different tick's window.

    Populated ONLY on the success path. Every early return leaves it ``None``,
    which reads downstream as "unknown", never as "metered".
    """
    from data.redis_client import (
        RedisReadError,
        persist,
        redis_configured,
        redis_key,
    )
    from data.vintage import (
        canonical_hour,
        classify_region,
        deserialize_records,
        serialize_records,
        summarize,
        update_vintage_records,
    )

    if demand_df is None or demand_df.empty or "demand_mw" not in demand_df.columns:
        return PhaseResult(region=region, ok=True, details={"skipped": "no_demand"})
    if not redis_configured():
        # Dev / offline: nothing to protect and nowhere to write.
        return PhaseResult(region=region, ok=True, details={"skipped": "redis_not_configured"})

    data_key = redis_key(f"vintage:{region}")
    seed_key = redis_key(f"vintage_seeded:{region}")

    try:
        existing_payload = _read_window_strict(data_key)
        tombstone = _read_window_strict(seed_key) if existing_payload is None else None
    except RedisReadError as exc:
        log.warning("vintage_history_read_failed", region=region, error=str(exc))
        return PhaseResult(region=region, ok=False, error=f"history read failed: {exc}")

    if existing_payload is None and tombstone is not None:
        # The #313 signature: the window key is gone but the tombstone —
        # written only after successful seeds/updates — survives. Rebuilding
        # now would re-pin first_seen_d for the whole window. Refuse, loudly.
        log.error(
            "vintage_window_missing_but_seeded",
            region=region,
            tombstone=str(tombstone),
        )
        return PhaseResult(
            region=region,
            ok=False,
            error="window absent but tombstone present — refusing to re-pin",
        )

    if existing_payload is not None and not (
        isinstance(existing_payload, dict) and existing_payload.get("records")
    ):
        # A payload we can read but not use (wrong shape, no records) is a
        # failure too — this phase never writes a record-less payload, so
        # overwriting it would destroy something another writer produced.
        log.error(
            "vintage_payload_unusable", region=region, payload_type=type(existing_payload).__name__
        )
        return PhaseResult(
            region=region, ok=False, error="existing payload unusable — refusing to overwrite"
        )

    first_seed = existing_payload is None

    try:
        existing = deserialize_records(existing_payload.get("records")) if existing_payload else []

        records = update_vintage_records(existing, demand_df)
        if not records:
            return PhaseResult(region=region, ok=True, details={"skipped": "no_usable_readings"})

        if data is not None:
            # #547. Keyed by ``canonical_hour`` — the same normalisation the
            # drift records join on — so the forecast phase can look up its
            # anchor hour without re-deriving a timestamp format.
            data.placeholder_by_hour = {
                key: r.was_placeholder
                for r in records
                if (key := canonical_hour(r.timestamp)) is not None
            }

        stats = summarize(records, region=region)
        persist(
            data_key,
            {"region": region, "records": serialize_records(records), **stats},
            ttl=REDIS_TTL,
        )
        # PR 3 groundwork: a ~250B summary the WEB TIER may read — the compact
        # stats + the revision-class verdict, without the 65KB records array.
        # Consumed by the provenance callouts; classification is heuristic v1
        # (see data/vintage.py constants) and hedges to "unknown".
        classification = classify_region(records)
        persist(
            redis_key(f"vintage_summary:{region}"),
            {"region": region, **stats, **classification},
            ttl=REDIS_TTL,
        )
        stats = {**stats, **classification}
        # Tombstone AFTER the data write, refreshed on every success so its
        # 7-day TTL only lapses when capture has been dead for a week anyway.
        persist(
            seed_key,
            {"last_write": datetime.now(UTC).isoformat(), "n_records": stats["n_records"]},
            ttl=REDIS_TTL * 7,
        )
        if first_seed:
            # Loud by design: any future re-seed of an established region is
            # the #313 corruption becoming visible in one log query.
            log.info("vintage_window_seeded", region=region, n_records=stats["n_records"])

        # The shadow signal. Answers two things currently unknown fleet-wide:
        # how often the anchor's seed is a day-ahead placeholder rather than a
        # measurement (12/43 BAs at the newest hour, observed once by hand), and
        # how far readings move afterwards. Flat scalars → queryable
        # jsonPayload.* since #306.
        log.info("demand_vintage", **stats)

        # GCS mirror (anchor-redesign PR A): best-effort durability + LOCAL
        # replay access — prod Redis is VPC-only, so the anchor-conditioning
        # study reads this parquet instead. Fire-and-forget by design (daemon
        # thread, never raises, never touches this phase's ok-flag): the Redis
        # key remains the source of truth; the mirror closes #312's
        # acknowledged "a Redis flush loses the study" fragility. Single-slot
        # latest.parquet is sufficient by explicit decision — the window
        # itself encodes first-sight-vs-settled per hour.
        try:
            from data.gcs_store import write_parquet

            write_parquet(pd.DataFrame(serialize_records(records)), "vintage", region)
        except Exception as exc:  # pragma: no cover — mirror must never bite
            log.warning("vintage_gcs_mirror_failed", region=region, error=str(exc))

        return PhaseResult(
            region=region,
            ok=True,
            details={
                "n_records": stats["n_records"],
                "n_placeholder": stats["n_placeholder"],
                "n_revised": stats["n_revised"],
            },
        )
    except Exception as exc:
        log.warning("vintage_write_failed", region=region, error=str(exc))
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
    from data.redis_client import RedisReadError, redis_key, redis_set
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

        try:
            existing = _read_window_strict(redis_key(f"drift_horizon:{region}"))
        except RedisReadError as exc:
            log.warning("horizon_drift_history_read_failed", region=region, error=str(exc))
            return PhaseResult(region=region, ok=False, error=f"history read failed: {exc}")
        existing_payload = existing if isinstance(existing, dict) else None
        payload = compute_horizon_drift_payload(region, existing_payload, forecast_payload, actuals)
        redis_set(redis_key(f"drift_horizon:{region}"), payload, ttl=REDIS_TTL)

        n_pending = len(payload.get("pending", []))
        n_records = sum(
            block.get("n_records", 0)
            for model in payload["models"].values()
            for block in model.values()
        )
        # #537: the two silent channels an hour can drop out of n_7d by,
        # logged the same way #542 taught this module to log a counter —
        # where the post-deploy check actually looks, not only published and
        # left for nobody to read. Same headline-model convention as
        # write_drift_metrics (#170): prefer ensemble, else the alphabetical
        # first model with any block, since the count is identical across
        # every model present for a horizon (see _horizon_rollup_block).
        models_present = sorted(payload["models"].keys())
        headline_model = (
            "ensemble"
            if "ensemble" in payload["models"]
            else (models_present[0] if models_present else None)
        )
        headline_by_horizon = payload["models"].get(headline_model, {}) if headline_model else {}
        horizons = payload.get("horizons", [])
        n_dedup_skipped_7d = {
            h: headline_by_horizon.get(h, {}).get("n_dedup_skipped_7d") for h in horizons
        }
        n_expired_unresolved_7d = {
            h: headline_by_horizon.get(h, {}).get("n_expired_unresolved_7d") for h in horizons
        }
        n_malformed_7d = {h: headline_by_horizon.get(h, {}).get("n_malformed_7d") for h in horizons}
        log.info(
            "horizon_drift_updated",
            region=region,
            models=models_present,
            pending=n_pending,
            total_records=n_records,
            headline_model=headline_model,
            n_dedup_skipped_7d=n_dedup_skipped_7d,
            n_expired_unresolved_7d=n_expired_unresolved_7d,
            n_malformed_7d=n_malformed_7d,
        )
        return PhaseResult(
            region=region,
            ok=True,
            details={
                "pending": n_pending,
                "total_records": n_records,
                "n_dedup_skipped_7d": n_dedup_skipped_7d,
                "n_expired_unresolved_7d": n_expired_unresolved_7d,
                "n_malformed_7d": n_malformed_7d,
            },
        )
    except Exception as exc:
        log.warning("horizon_drift_write_failed", region=region, error=str(exc))
        return PhaseResult(region=region, ok=False, error=str(exc))


# ── Phase: public forecast benchmark (E0) ────────────────────


#: Nominal horizon → hours, for the two leads the benchmark scores. MUST
#: agree with ``models.drift._HORIZON_HOURS``, which decides which hour a
#: "24h" snapshot actually targets; pinned by a test rather than imported so
#: a divergence fails CI loudly instead of degrading to "no observation".
_BENCHMARK_LEAD_HOURS: dict[str, int] = {"24h": 24, "48h": 48}


def _observed_lead_hours(
    previous_forecast: dict[str, Any] | None,
) -> dict[str, float]:
    """This tick's REALIZED lead per nominal horizon.

    ``_resolve_forecast_start`` anchors row 0 at the last *real* demand
    hour, so with EIA's publishing lag a "24h" row is not 24h ahead —
    measured at 23.8–24.0h across the fleet
    (``docs/BENCHMARK_PROVENANCE.md``). Computing it from the forecast
    payload the phase already reads avoids adding ``made_at`` to the live
    drift record schema, which would churn four models × three horizons ×
    51 regions to serve one field.

    Two things here are load-bearing, and both were wrong in the first cut:

    * **The rows live under ``forecasts``.** That is the Redis payload's
      key; only the API reshapes it to ``forecast``. Reading the API's name
      off the Redis payload silently produced *no* observation, which left
      ``lead_basis`` on ``"nominal"`` and withheld the conservative label
      fleet-wide.
    * **The horizon is measured from row 0 + H, not from row index H−1** —
      exactly how ``models.drift.snapshot_horizon_predictions`` picks the
      target hour it later scores. Indexing instead measured the hour
      *before* the one the benchmark grades, understating every lead by an
      hour.
    """
    if not isinstance(previous_forecast, dict):
        return {}
    rows = (
        previous_forecast.get("forecasts")
        or previous_forecast.get("forecast")
        or previous_forecast.get("rows")
        or []
    )
    scored_at = previous_forecast.get("scored_at")
    if not rows or not scored_at:
        return {}
    try:
        made = pd.Timestamp(scored_at)
        origin_ts = rows[0].get("timestamp") or rows[0].get("ts")
        if origin_ts is None:
            return {}
        origin = pd.Timestamp(origin_ts)
        return {
            label: (origin + pd.Timedelta(hours=hours) - made).total_seconds() / 3600.0
            for label, hours in _BENCHMARK_LEAD_HOURS.items()
        }
    except Exception:  # pragma: no cover — observation only, never fatal
        return {}


def write_benchmark_metrics(
    region: str,
    previous_forecast: dict[str, Any] | None = None,
    demand_df: pd.DataFrame | None = None,
) -> PhaseResult:
    """Score GridPulse against the BA's OWN day-ahead forecast (E0).

    The product competes with a free incumbent — EIA-930 publishes each BA's
    day-ahead forecast — so the benchmark is the evidence that decides
    whether GridPulse is worth anything to a third party. It rides
    instrumentation that already exists rather than a bolted-on replay:

    * official arm  ← ``vintage:{region}`` ``first_seen_df``
    * GridPulse arm ← ``drift_horizon:{region}`` resolved 24h/48h records
    * truth         ← vintage ``last_d`` (settled), for BOTH arms
    * exclusions    ← ``vintage_summary:{region}`` class + DF coverage

    Ordering: must run AFTER the vintage, vintage-summary and horizon-drift
    phases in the same tick, all of which it reads. Non-critical — a
    benchmark error never blocks scoring.
    """
    try:
        # Imported inside the guard: an import-time failure here must degrade
        # the benchmark, never take down a scoring run for a measurement.
        from data.redis_client import redis_get, redis_key, redis_set
        from data.vintage import deserialize_records
        from models.benchmark import compute_benchmark_payload, revised_df_from_frame

        raw = redis_get(redis_key(f"vintage:{region}"))
        rows = raw.get("records") if isinstance(raw, dict) else None
        records = deserialize_records(rows)
        if not records:
            return PhaseResult(region=region, ok=True, details={"skipped": "no_vintage"})

        summary = redis_get(redis_key(f"vintage_summary:{region}"))
        summary = summary if isinstance(summary, dict) else {}
        horizon = redis_get(redis_key(f"drift_horizon:{region}"))

        payload = compute_benchmark_payload(
            region,
            records,
            horizon if isinstance(horizon, dict) else None,
            summary.get("revision_class"),
            mean_revision_pct=summary.get("mean_fresh_revision_pct"),
            # The live frame's forecast_mw IS EIA's post-revision DF, so the
            # conservative official arm costs no extra capture.
            revised_df_by_ts=revised_df_from_frame(demand_df),
            observed_lead_h=_observed_lead_hours(previous_forecast),
            # #348. `previous_forecast` is this region's forecast payload,
            # which is where the substitution is recorded — so naming what we
            # actually serve costs no extra Redis read.
            served_series=(previous_forecast or {}).get("served_series"),
        )
        # Per-BA freshness. The fleet rollup's own timestamp stamps the
        # rollup; a reader of one region's row has no way to tell how old it
        # is without this, and the per-region API route has no fleet key to
        # fall back on.
        payload["scored_at"] = datetime.now(UTC).isoformat()
        redis_set(redis_key(f"benchmark:{region}"), payload, ttl=REDIS_TTL)

        headline = (payload.get("leads") or {}).get("24h") or {}
        log.info(
            "benchmark_scored",
            region=region,
            scoreable=payload.get("scoreable"),
            reason=payload.get("reason"),
            n=headline.get("n"),
            winner=headline.get("winner"),
            delta_mape=headline.get("delta_mape"),
        )
        return PhaseResult(
            region=region,
            ok=True,
            details={
                "scoreable": payload.get("scoreable"),
                "reason": payload.get("reason"),
                "winner": headline.get("winner"),
            },
        )
    except Exception as exc:
        log.warning("benchmark_write_failed", region=region, error=str(exc))
        return PhaseResult(region=region, ok=False, error=str(exc))


# ── Phase: backtests (training) ──────────────────────────────


def _backtest_is_fresh(payload: Any, refresh_days: int) -> bool:
    """True when an existing backtest payload is younger than the refresh window.

    Anything unparseable — no payload, no ``computed_at``, a bad timestamp, a
    payload written before this field existed — is treated as STALE, so the
    failure mode is a recomputation rather than a number that never refreshes
    again. Fail toward doing the work.
    """
    if refresh_days <= 0 or not isinstance(payload, dict):
        return False
    stamp = payload.get("computed_at")
    if not isinstance(stamp, str):
        return False
    try:
        computed = datetime.fromisoformat(stamp)
    except ValueError:
        return False
    if computed.tzinfo is None:
        computed = computed.replace(tzinfo=UTC)
    age = datetime.now(UTC) - computed
    return age < timedelta(days=refresh_days)


def write_backtests(data: RegionData) -> PhaseResult:
    """Run walk-forward backtests for the configured horizons and write to Redis.

    Recomputes at most every ``BACKTEST_REFRESH_DAYS`` (default 7), not every
    run. This is the training job's largest single cost: 12 folds per BA and
    **every fold trains its own booster** — walk-forward requires it, each
    fold's model must see only that fold's training window. Measured,
    `train_xgboost(cross_validate=False)` is 11.26s per fold against 0.42s for
    the fold's recursive predict loop: **training is 96% of fold cost**, ~141s
    per BA, ~120 minutes across 51 BAs, roughly 29% of the job's ~25,000
    task-seconds.

    It only ever ran daily because the payloads carried a 24h TTL — the refresh
    interval and the expiry were the same number, so skipping a day blanked the
    Models tab. Nothing about the measurement wants daily: it scores a model
    ARCHITECTURE against a trailing window, and neither moves meaningfully in a
    day.

    The real cost of this is staleness, so it is made explicit rather than
    hidden: every payload carries ``computed_at``, the TTL outlives the refresh
    interval so a key cannot expire between runs, and every skip is logged with
    the payload's age. A reader can always tell how old the number is.

    Imports ``_run_backtest_for_horizon`` lazily so the Dash callbacks module
    isn't pulled into the job container unless this phase runs.
    """
    from components.callbacks import _run_backtest_for_horizon
    from config import BACKTEST_REFRESH_DAYS, BACKTEST_TTL_SECONDS
    from data.redis_client import redis_get, redis_key, redis_set

    region = data.region
    written: list[int] = []
    skipped: list[int] = []
    for horizon in BACKTEST_HORIZONS:
        key = redis_key(f"backtest:{DEFAULT_BACKTEST_EXOG_MODE}:{region}:{horizon}")
        # redis_get is the fail-SOFT client: a Redis blip returns None, which
        # reads as stale and recomputes. Costly, but never wrong — the opposite
        # choice would skip on an outage and let a payload silently expire.
        existing = redis_get(key)
        if _backtest_is_fresh(existing, BACKTEST_REFRESH_DAYS):
            skipped.append(horizon)
            log.info(
                "job_backtest_fresh_skip",
                region=region,
                horizon=horizon,
                computed_at=existing.get("computed_at"),
                refresh_days=BACKTEST_REFRESH_DAYS,
            )
            continue
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
                key,
                {
                    "horizon": horizon,
                    "exog_mode": DEFAULT_BACKTEST_EXOG_MODE,
                    "exog_source": bt.get("exog_source", "climatology/naive baseline"),
                    # The disclosure that makes a weekly cadence honest. Without
                    # it a reader cannot distinguish today's number from one
                    # measured six days ago, and the skip gate above has nothing
                    # to read.
                    "computed_at": datetime.now(UTC).isoformat(),
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
                ttl=BACKTEST_TTL_SECONDS,
            )
            written.append(horizon)
            # The alert signal. Deliberately counts RECOMPUTES, not skips.
            #
            # A metric built on `job_backtest_fresh_skip` looks like the
            # obvious choice and is a trap: when the gate breaks, skips stop
            # being emitted, the logs-based counter has NO data, and a
            # threshold-below condition never evaluates. The alert would go
            # quiet at exactly the moment it should fire. Counting the thing
            # that INCREASES on failure avoids that entirely.
            #
            # Normal operation emits 153 of these (51 BAs x 3 horizons) on one
            # day in seven and none on the other six, so a 72h window holds at
            # most one recompute day. See
            # docs/monitoring/backtest_recompute_alert.json.
            log.info(
                "job_backtest_recomputed",
                region=region,
                horizon=horizon,
                refresh_days=BACKTEST_REFRESH_DAYS,
                previous_computed_at=(
                    existing.get("computed_at") if isinstance(existing, dict) else None
                ),
            )
        except Exception as e:
            log.warning(
                "job_backtest_error",
                region=region,
                horizon=horizon,
                error=str(e),
            )

    return PhaseResult(
        region=region,
        # A run that skipped everything because every payload was fresh is a
        # SUCCESS, not a no-op failure. Only "nothing written and nothing
        # deliberately skipped" means the phase actually got nowhere.
        ok=bool(written or skipped),
        details={"horizons_written": written, "horizons_fresh_skipped": skipped},
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


def check_backtest_recompute_cadence(recomputed_regions: int) -> bool:
    """Emit ``backtest_recompute_unexpected`` if backtests recomputed too soon.

    **Why this lives in code and not in a Cloud Monitoring condition.** The
    signal is a FREQUENCY: one recompute day per week is correct, two in three
    days is a regression, and no `conditionMatchedLog` can tell those apart --
    it fires on both. The natural fix is a metric threshold over a multi-day
    window, and Cloud Monitoring rejects it:

        Alignment periods longer than 25h are not supported.

    A <=25h window cannot express it either. The training job runs once daily,
    so consecutive recompute days land in ADJACENT windows and never the same
    one -- any 24h window sees at most one day's worth (153) whether backtests
    run weekly or every single run. So the comparison has to happen where the
    state is: here.

    Called ONCE PER RUN from the epilogue, never per region. A per-region check
    would write the fleet-wide marker on the first region and then see a fresh
    marker for the other fifty, reporting an anomaly on every single one.

    Returns True when the cadence looks wrong, so the caller can surface it.
    The marker is refreshed regardless -- a missed detection is better than a
    stuck marker that suppresses every future one.
    """
    from config import BACKTEST_REFRESH_DAYS
    from data.redis_client import redis_get, redis_key

    if recomputed_regions <= 0:
        return False

    previous = redis_get(redis_key("meta:last_backtest_recompute"))
    prev_at = previous.get("updated_at") if isinstance(previous, dict) else None
    unexpected = False
    age_days: float | None = None

    if isinstance(prev_at, str):
        try:
            then = datetime.fromisoformat(prev_at)
            if then.tzinfo is None:
                then = then.replace(tzinfo=UTC)
            age_days = (datetime.now(UTC) - then).total_seconds() / 86400
            # One day of slack: the job runs at a fixed hour, so consecutive
            # legitimate recomputes are ~REFRESH_DAYS apart give or take
            # scheduling jitter. Comparing against the bare threshold would
            # alert on a run that started a few minutes early.
            unexpected = age_days < (BACKTEST_REFRESH_DAYS - 1)
        except ValueError:
            # Unparseable marker: treat as no marker rather than as an anomaly.
            # Fail toward silence here -- the opposite would cry wolf on the
            # first run after any format change.
            age_days = None

    if unexpected:
        log.warning(
            "backtest_recompute_unexpected",
            regions_recomputed=recomputed_regions,
            days_since_last_recompute=round(age_days, 2) if age_days is not None else None,
            refresh_days=BACKTEST_REFRESH_DAYS,
        )

    write_meta("last_backtest_recompute", extra={"regions_recomputed": recomputed_regions})
    return unexpected


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
    "collect_substeps",
    "engineer_region_features",
    "fetch_all_regions",
    "fetch_region_data",
    "ordered_regions",
    "predict_and_write_forecast",
    "safe_phase",
    "substep",
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
