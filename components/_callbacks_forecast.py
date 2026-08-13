"""Forecast / demand-outlook tab helpers extracted from ``components/callbacks.py``.

Step 8 of the ``callbacks.py`` decomposition tracked in issue #87.
Continues the per-tab split established by:

* #98 — shared infrastructure (``_callbacks_shared.py``)
* #99 — US Grid tab (``_callbacks_us_grid.py``)
* #100 — Models tab (``_callbacks_models.py``)
* #101 — Alerts tab (``_callbacks_alerts.py``)
* #102 — Generation tab (``_callbacks_generation.py``)
* #103 — Weather tab (``_callbacks_weather.py``)
* #104 / #105 / #106 — Overview tab (``_callbacks_overview.py``)

## What lives here

Six helpers that compose the demand-forecast surface:

* ``_confidence_half_width`` — horizon-scaled heuristic envelope used as
  fallback when no empirical backtest residuals are available.
* ``_add_confidence_bands`` — adds upper/lower indicative-range traces
  to a forecast figure. Prefers empirical intervals from backtests
  (via ``_empirical_interval_from_backtests`` in shared) and falls back
  to the heuristic envelope when calibration data is scarce.
* ``_add_trailing_actuals`` — overlays trailing actual demand as a
  lead-in trace on the forecast chart.
* ``_run_forecast_outlook`` — the v1 inline-compute path for generating
  forward-looking forecasts. 3-tier cache (in-memory → SQLite → train
  fresh). In production this returns a ``"warming"`` status instead of
  training when ``REQUIRE_REDIS=True``, since the scoring Cloud Run Job
  owns forecast generation.
* ``_create_future_features`` — feature-engineering for future
  timestamps using (hour, day_of_week) historical group means so models
  see realistic daily/weekly patterns instead of frozen values.
* ``_outlook_tab_from_redis`` — Redis fast path that builds the entire
  Demand Forecast tab (figure + 7 KPI strings + insight card) from the
  scoring job's hourly ``gridpulse:forecast:{region}:1h`` payload.

## Cross-tab dependency factoring

This extraction lifted three helpers from callbacks.py to
``_callbacks_shared.py`` because both Forecast (here) and Backtest
(later in callbacks.py until Step 9) need them:

* ``_compute_data_hash`` — data signature for cache correctness
* ``_collect_backtest_residuals`` — residual collector across cache layers
* ``_empirical_interval_from_backtests`` — empirical quantile estimator

Living in shared lets each tab module import them directly without
introducing a sideways dependency between sibling tab modules.

## Public-import surface

``components/callbacks.py`` re-imports each function by name. Tests
import via ``from components.callbacks import _run_forecast_outlook``
etc — the re-export shim keeps those import sites valid without any
caller-side changes. ``register_callbacks`` continues to call the
helpers directly through the same namespace.

When patching for tests, target the function's *new* namespace:

    @patch("components._callbacks_forecast.redis_get")  # ✓
    @patch("components.callbacks.redis_get")            # ✗ (no effect)
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import structlog
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from components._callbacks_shared import (
    _CACHE_VERSION,
    _EIA_FUEL_MAP,
    _GENERATION_CACHE,
    _MODEL_BAND_COLORS,
    _MODEL_CACHE,
    _PREDICTION_CACHE,
    COLORS,
    _compute_data_hash,
    _empirical_interval_from_backtests,
    _empty_figure,
    _guard_max_ok,
    _layout,
    _pipeline_alive,
    _read_ensemble_forecast_from_redis,
    _scoring_pass_completed_since_actuals,
    _widening_interval_from_backtests,
)
from components.accessibility import LINE_STYLES, model_display_name
from components.cards import build_metrics_bar
from config import (
    CACHE_TTL_SECONDS,
    OPEN_METEO_FORECAST_HOURS,
    REGION_CAPACITY_MW,
    REQUIRE_REDIS,
)
from data.redis_client import redis_get, redis_key

log = structlog.get_logger()

# Recent trailing window (days) for the (hour, dow) climatology used by
# ``_create_future_features``, with a min-rows guard before trusting it.
# Duplicated from ``jobs.phases`` (CLIMATOLOGY_WINDOW_DAYS /
# _CLIMATOLOGY_MIN_ROWS) rather than imported — the web tier deliberately
# does not import the jobs module (see ``_callbacks_shared`` for the same
# convention). Keep the values in sync with jobs/phases.py; rationale for
# the recent-window restriction is documented there (#281/#282).
_CLIMATOLOGY_WINDOW_DAYS = 28
_CLIMATOLOGY_MIN_ROWS = 7 * 24  # ≥ 1 week before trusting the recent window


def _confidence_half_width(horizon_hours: int) -> float:
    """Return the indicative-range half-width as a fraction, scaled by horizon.

    These are heuristic percentages, NOT statistically calibrated confidence
    intervals.  They provide a visual sense of increasing uncertainty at
    longer horizons but should not be interpreted as probabilistic coverage
    guarantees.  When empirical backtest residuals are available,
    ``_add_confidence_bands`` uses those instead (see ``_empirical_interval_from_backtests``).
    """
    if horizon_hours <= 24:
        return 0.03  # ±3%
    if horizon_hours <= 168:
        return 0.06  # ±6%
    return 0.10  # ±10% for 30-day


def _add_forecast_horizon_divider(
    fig: go.Figure,
    timestamps,
    horizon_hours: int,
) -> bool:
    """Mark the boundary between Open-Meteo forecast and climatology fallback.

    Open-Meteo's free ``/forecast`` endpoint covers 16 days (384 hours).
    Beyond that, ``jobs/phases._build_future_feature_frame`` drives the
    weather features from the per-BA (day_of_year, hour) weather-normal
    (#283; seam anomaly-blend at the boundary), falling back to the
    recent-28d (hour, dow) climatology where a BA's normal artifact isn't
    backfilled yet. The model still produces a forecast there, but its
    weather inputs are seasonal-average shaped, not actual forward-looking
    values — which is exactly what the divider discloses.

    This helper makes that distinction visible on the chart:

    1. A dotted vertical line at the 16-day boundary
    2. A subtle background shade on the climatology segment
    3. Annotations labeling both segments

    Returns ``True`` if the divider was added (horizon extends past
    Open-Meteo coverage), ``False`` otherwise (24h / 7d views are all
    real forecast — no divider needed).

    See ADR-008 in PRD.md for the architectural decision behind this
    behavior and the alternatives that were considered.
    """
    if horizon_hours <= OPEN_METEO_FORECAST_HOURS:
        # All shown hours are within Open-Meteo's real-forecast coverage.
        # No climatology section → no divider needed.
        return False

    # Coerce to a pandas-friendly index so positional access is reliable
    # whether the caller hands us a DatetimeIndex, np.ndarray, or list.
    ts = pd.DatetimeIndex(timestamps)
    if len(ts) <= OPEN_METEO_FORECAST_HOURS:
        # Shorter slice than the horizon constant — defensive guard.
        return False

    # Plotly's ``add_vline(annotation_text=...)`` positions its
    # annotation by computing ``mean([x0, x1])``, which fails on
    # pandas Timestamp objects (no scalar ``__add__``). Pass ISO
    # strings to side-step the arithmetic; add the annotations
    # separately via ``add_annotation`` for full control.
    boundary_iso = pd.Timestamp(ts[OPEN_METEO_FORECAST_HOURS]).isoformat()
    start_iso = pd.Timestamp(ts[0]).isoformat()
    end_iso = pd.Timestamp(ts[-1]).isoformat()

    # Vertical divider line — dotted, deliberately subtle so it reads as
    # a guide rail rather than a primary visual element.
    fig.add_vline(
        x=boundary_iso,
        line=dict(color="rgba(160,180,200,0.45)", width=1, dash="dot"),
    )

    # Faint background shade past the boundary — communicates "this is
    # different" without competing with the forecast trace itself.
    fig.add_vrect(
        x0=boundary_iso,
        x1=end_iso,
        fillcolor="rgba(160,180,200,0.05)",
        line_width=0,
        layer="below",
    )

    # Right-side label for the climatology segment, anchored just past
    # the boundary line. Uses paper coords for vertical positioning so
    # the label stays put when the y-axis rescales.
    fig.add_annotation(
        x=boundary_iso,
        y=1.0,
        xref="x",
        yref="paper",
        text="climatology baseline →",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=10, color="rgba(160,180,200,0.85)"),
        yshift=2,
    )

    # Left-side label for the real-forecast segment.
    fig.add_annotation(
        x=start_iso,
        y=1.0,
        xref="x",
        yref="paper",
        text="← Open-Meteo forecast",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=10, color="rgba(160,180,200,0.85)"),
        yshift=2,
    )

    return True


def _add_confidence_bands(
    fig: go.Figure,
    timestamps: pd.DatetimeIndex | np.ndarray,
    predictions: np.ndarray,
    horizon_hours: int,
    region: str | None = None,
    model_name: str = "ensemble",
) -> dict[str, float | int | bool | str]:
    """Add upper/lower indicative range traces to a forecast figure.

    When empirical backtest residuals are available the range is data-driven.
    Otherwise a heuristic percentage envelope is used (clearly labelled as
    such so users do not mistake it for a calibrated confidence interval).
    """
    from models.evaluation import apply_empirical_interval

    interval_meta = {"method": "heuristic", "target_coverage": 0.80}
    lower = upper = None

    # #283 Phase 3b: lead-time-resolved P10–P90 band. Anchor error quantiles
    # from the 24h/168h/720h backtests — pinned at each pool's EFFECTIVE lead
    # (~H/2, since a horizon-H backtest pools residuals over leads 1..H, so
    # its quantiles measure roughly the mid-window error) — interpolate across
    # the chart's lead axis, and enforce monotone widening: forecast
    # uncertainty cannot shrink with lead time; a non-monotone wiggle between
    # anchors is single-origin backtest sampling noise, not signal. np.interp
    # holds the ends constant outside the anchor range. The P50 of the fan is
    # the forecast line itself.
    widening = _widening_interval_from_backtests(region, model_name) if region else None
    if widening and bool(widening.get("available")):
        n = len(predictions)
        lead = np.arange(1, n + 1, dtype=float)
        hs = np.array(
            [a.get("effective_lead", a["horizon"]) for a in widening["anchors"]], dtype=float
        )
        lo_anchor = np.array([a["lower_error"] for a in widening["anchors"]], dtype=float)
        up_anchor = np.array([a["upper_error"] for a in widening["anchors"]], dtype=float)
        lo_vec = np.minimum.accumulate(np.interp(lead, hs, lo_anchor))  # non-increasing
        up_vec = np.maximum.accumulate(np.interp(lead, hs, up_anchor))  # non-decreasing
        upper = predictions + up_vec
        # Physical floor: demand is non-negative (#282) — the deep-tail P10
        # offset can exceed a small forecast value.
        lower = np.maximum(predictions + lo_vec, 0.0)
        # Edge ordering: a systematically over-forecasting model can have a
        # NEGATIVE q90 (upper_error < 0), which after the lower floor could
        # invert the band (upper < lower). Clamp so the fill renders sanely.
        upper = np.maximum(upper, lower)
        interval_meta = {"method": "empirical_widening", **widening}

    if lower is None:
        empirical = None
        if region:
            empirical = _empirical_interval_from_backtests(region, model_name, horizon_hours)
        if empirical and bool(empirical.get("available")):
            lower, upper = apply_empirical_interval(
                predictions,
                float(empirical["lower_error"]),
                float(empirical["upper_error"]),
            )
            interval_meta = {"method": "empirical", **empirical}
        else:
            hw = _confidence_half_width(horizon_hours)
            upper = predictions * (1 + hw)
            lower = predictions * (1 - hw)

    if interval_meta["method"] in ("empirical", "empirical_widening"):
        # Disclose the calibration source when the residuals came from a
        # substitute model — the prod backtest payload only carries XGBoost
        # predictions, so a Prophet/SARIMAX/ensemble band is typically
        # XGBoost-calibrated (2026-07 critical-review finding P1-2/F6-003).
        calib = interval_meta.get("calibration_model")
        calib_note = "" if calib in (None, model_name) else f" ({calib}-calibrated)"
        if interval_meta["method"] == "empirical_widening":
            band_name = f"P10–P90 empirical range, widens with lead{calib_note}"
        else:
            band_name = f"80% empirical prediction interval{calib_note}"
    else:
        band_name = "80% indicative range"

    band_fill = _MODEL_BAND_COLORS.get(model_name, COLORS["confidence"])

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=band_fill,
            name=band_name,
            hovertemplate="%{fullData.name}<br>%{y:,.0f} MW<extra></extra>",
        )
    )
    return interval_meta


def _interval_caption(interval_meta: dict, model_name: str) -> str:
    """Chart-subtitle disclosure for the uncertainty band, shared by the Redis
    fast path and the inline-compute path so the two can't drift (#283 Phase
    3b verification). Returns "" for the heuristic envelope (its band legend
    already labels it "indicative").
    """
    method = interval_meta.get("method")
    calib = interval_meta.get("calibration_model")
    calib_note = "" if calib in (None, model_name) else f", {calib}-calibrated"
    if method == "empirical_widening":
        anchor_hs = "/".join(f"{a['horizon']}h" for a in interval_meta.get("anchors", []))
        return (
            f"<br><sup>P10–P90 empirical outcome range — widens with lead time "
            f"(anchored on {anchor_hs} backtest residuals{calib_note})</sup>"
        )
    if method == "empirical":
        return (
            f"<br><sup>80% empirical prediction interval "
            f"(calibration window: last {int(interval_meta.get('calibration_window_hours', 0))}h"
            f"{calib_note})</sup>"
        )
    return ""


def _add_trailing_actuals(
    fig: go.Figure,
    demand_json: str | None,
    tail_hours: int = 48,
) -> None:
    """Add trailing actual demand as a lead-in trace on the forecast chart."""
    if not demand_json:
        return
    try:
        demand_df = pd.read_json(io.StringIO(demand_json))
        demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])
        demand_df = demand_df.sort_values("timestamp")
        tail = demand_df.tail(tail_hours)
        if tail.empty:
            return
        fig.add_trace(
            go.Scatter(
                x=tail["timestamp"],
                y=tail["demand_mw"],
                mode="lines",
                name="Actual",
                line=dict(color=COLORS["actual"], width=2, dash="dot"),
            )
        )
    except Exception:
        pass  # Non-critical — chart still works without actuals


def _run_forecast_outlook(
    demand_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    horizon_hours: int,
    model_name: str,
    region: str,
) -> dict:
    """Generate forward-looking forecast using cached model when possible."""
    import time

    from data.feature_engineering import engineer_features
    from data.preprocessing import merge_demand_weather

    data_hash = _compute_data_hash(demand_df, weather_df, region)
    cache_key = (region, horizon_hours, model_name)

    # Check prediction cache first (fastest path)
    if cache_key in _PREDICTION_CACHE:
        cached_pred, cached_ts, cached_hash, cached_time = _PREDICTION_CACHE[cache_key]
        if cached_hash == data_hash and (time.time() - cached_time) < CACHE_TTL_SECONDS:
            log.info("forecast_cache_hit", region=region, horizon=horizon_hours, model=model_name)
            return {"timestamps": cached_ts, "predictions": cached_pred}

    # Check SQLite cache (survives page refresh / server restart)
    try:
        from data.cache import get_cache

        sqlite_cache = get_cache()
        sqlite_key = f"forecast:{region}:{horizon_hours}:{model_name}"
        cached_sqlite = sqlite_cache.get(sqlite_key)
        if (
            cached_sqlite is not None
            and isinstance(cached_sqlite, dict)
            and "predictions" in cached_sqlite
            and cached_sqlite.get("cache_version") == _CACHE_VERSION
            and cached_sqlite.get("data_hash") == data_hash
        ):
            cached_sqlite["timestamps"] = pd.to_datetime(cached_sqlite["timestamps"])
            cached_sqlite["predictions"] = np.array(cached_sqlite["predictions"])
            _PREDICTION_CACHE[cache_key] = (
                cached_sqlite["predictions"],
                cached_sqlite["timestamps"],
                data_hash,
                time.time(),
            )
            log.info(
                "forecast_sqlite_cache_hit", region=region, horizon=horizon_hours, model=model_name
            )
            return cached_sqlite
    except Exception as e:
        log.debug("forecast_sqlite_cache_miss", error=str(e))

    # REQUIRE_REDIS: the scheduled scoring job owns forecast generation.
    # If neither the in-memory cache nor the SQLite cache has a hit, surface
    # a degraded state rather than training inline. The Dash UI treats this
    # like any other degraded state and renders a skeleton.
    if REQUIRE_REDIS:
        # P2-35 (#273): "warming — will appear shortly" is only honest while
        # the pipeline is genuinely cold (first runs after a deploy/flush).
        # Two escalations, each claiming only the permanence its evidence
        # earns (verification pass on the first cut of this fix):
        # (a) the forecast payload EXISTS but can't serve this selection
        #     (model column missing / horizon not covered) — a per-RUN state
        #     a single-tick model failure heals within the hour, so the copy
        #     must not claim permanence ("unavailable_selection");
        # (b) the forecast key is ABSENT while the pipeline is alive AND a
        #     full scoring pass completed since this region's actuals landed
        #     — the pipeline provably had its chance and produced no
        #     forecast ("unavailable"). The completion check keeps the first
        #     pass after a flush honest: actuals land minutes before the
        #     forecast within one pass.
        status = None
        try:
            forecast_payload = redis_get(redis_key(f"forecast:{region}:1h"))
            if isinstance(forecast_payload, dict) and forecast_payload.get("forecasts"):
                status = "unavailable_selection"
            elif _pipeline_alive(region) and _scoring_pass_completed_since_actuals(region):
                status = "unavailable"
        except Exception:  # pragma: no cover — defensive; keep the softer copy
            pass
        if status is not None:
            log.info(
                "forecast_unavailable_state",
                region=region,
                horizon=horizon_hours,
                model=model_name,
                kind=status,
            )
            return {
                "error": status,
                "status": status,
                "message": "The pipeline is live but no forecast exists for this selection.",
            }
        log.info(
            "forecast_warming_state",
            region=region,
            horizon=horizon_hours,
            model=model_name,
        )
        return {
            "error": "warming",
            "status": "warming",
            "message": "Forecasts are being refreshed by the scheduled job.",
        }

    # Merge and engineer features
    merged_df = merge_demand_weather(demand_df, weather_df)
    featured_df = engineer_features(merged_df)
    featured_df = featured_df.dropna(subset=["demand_mw"])

    if len(featured_df) < 168:
        return {"error": "Insufficient training data"}

    train_df = featured_df.copy()
    last_ts = train_df["timestamp"].max()
    future_timestamps = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1), periods=horizon_hours, freq="h", tz="UTC"
    )
    future_df = _create_future_features(train_df, future_timestamps)

    # NEXD-13: SHAP data for inline tooltips (populated for XGBoost only)
    shap_data = None
    xgb_model_dict = None

    try:
        if model_name == "xgboost":
            from models.xgboost_model import predict_xgboost, train_xgboost

            # Only XGBoost is cached in _MODEL_CACHE (small tree structure)
            xgb_model = None
            mck = (region, "xgboost", 0)
            if mck in _MODEL_CACHE:
                cached_model, cached_hash, cached_time = _MODEL_CACHE[mck]
                if cached_hash == data_hash and (time.time() - cached_time) < CACHE_TTL_SECONDS:
                    xgb_model = cached_model
                    log.info("model_cache_hit", region=region, model="xgboost")
            if xgb_model is None:
                log.info("model_training_start", region=region, model="xgboost")
                xgb_model = train_xgboost(train_df)
                _MODEL_CACHE[mck] = (xgb_model, data_hash, time.time())
                log.info("model_cached", region=region, model="xgboost")
            predictions = predict_xgboost(xgb_model, future_df)[:horizon_hours]

            # Compute SHAP values for per-point tooltips (NEXD-13)
            xgb_model_dict = xgb_model
            try:
                from config import feature_enabled

                if feature_enabled("inline_tooltips"):
                    from models.xgboost_model import compute_shap_values

                    shap_result = compute_shap_values(xgb_model, future_df)
                    shap_data = {
                        "shap_values": shap_result["shap_values"][:horizon_hours],
                        "feature_names": shap_result["feature_names"],
                    }
            except Exception:
                log.debug("shap_computation_skipped", model=model_name, region=region)

        elif model_name == "prophet":
            from models.prophet_model import predict_prophet, train_prophet

            prophet_model = None
            mck = (region, "prophet", 0)
            if mck in _MODEL_CACHE:
                cached_model, cached_hash, cached_time = _MODEL_CACHE[mck]
                if cached_hash == data_hash and (time.time() - cached_time) < CACHE_TTL_SECONDS:
                    prophet_model = cached_model
                    log.info("model_cache_hit", region=region, model="prophet")
            if prophet_model is None:
                log.info("model_training_start", region=region, model="prophet")
                prophet_model = train_prophet(train_df)
                _MODEL_CACHE[mck] = (prophet_model, data_hash, time.time())
                log.info("model_cached", region=region, model="prophet")
            prophet_result = predict_prophet(prophet_model, future_df, periods=horizon_hours)
            predictions = prophet_result["forecast"][:horizon_hours]

        elif model_name == "arima":
            from models.arima_model import predict_arima, train_arima

            arima_model = None
            mck = (region, "arima", 0)
            if mck in _MODEL_CACHE:
                cached_model, cached_hash, cached_time = _MODEL_CACHE[mck]
                if cached_hash == data_hash and (time.time() - cached_time) < CACHE_TTL_SECONDS:
                    arima_model = cached_model
                    log.info("model_cache_hit", region=region, model="arima")
            if arima_model is None:
                log.info("model_training_start", region=region, model="arima")
                arima_model = train_arima(train_df)
                _MODEL_CACHE[mck] = (arima_model, data_hash, time.time())
                log.info("model_cached", region=region, model="arima")
            predictions = predict_arima(arima_model, future_df, periods=horizon_hours)[
                :horizon_hours
            ]

        elif model_name == "ensemble":
            # Equal-weight ensemble (no actuals for MAPE weighting).
            # Strategy: reuse cached individual-model predictions when available,
            # then only train/predict for models that aren't cached yet.
            # SARIMAX is excluded beyond 168h — its integrated component
            # compounds errors at long horizons and actively degrades
            # ensemble quality.
            from concurrent.futures import ThreadPoolExecutor, as_completed

            ensemble_models = (
                ["xgboost", "prophet"] if horizon_hours > 168 else ["xgboost", "prophet", "arima"]
            )

            preds = {}

            # Fast path: check if individual model predictions are already cached
            for sub_model in ensemble_models:
                sub_key = (region, horizon_hours, sub_model)
                if sub_key in _PREDICTION_CACHE:
                    cp, ct, ch, ctm = _PREDICTION_CACHE[sub_key]
                    if ch == data_hash and (time.time() - ctm) < CACHE_TTL_SECONDS:
                        preds[sub_model] = cp
                        log.info("ensemble_reuse_cached", model=sub_model, horizon=horizon_hours)

            # Only train models whose predictions we don't already have
            missing = [m for m in ensemble_models if m not in preds]

            if missing:

                def _forecast_xgb():
                    from models.xgboost_model import predict_xgboost, train_xgboost

                    xgb_model = None
                    mck = (region, "xgboost", 0)
                    if mck in _MODEL_CACHE:
                        cached_model, cached_hash_c, cached_time_c = _MODEL_CACHE[mck]
                        if (
                            cached_hash_c == data_hash
                            and (time.time() - cached_time_c) < CACHE_TTL_SECONDS
                        ):
                            xgb_model = cached_model
                    if xgb_model is None:
                        xgb_model = train_xgboost(train_df)
                        _MODEL_CACHE[mck] = (xgb_model, data_hash, time.time())
                    p = predict_xgboost(xgb_model, future_df)[:horizon_hours]
                    _PREDICTION_CACHE[(region, horizon_hours, "xgboost")] = (
                        p,
                        future_timestamps,
                        data_hash,
                        time.time(),
                    )
                    return "xgboost", p

                def _forecast_prophet():
                    from models.prophet_model import predict_prophet, train_prophet

                    pm = None
                    mck = (region, "prophet", 0)
                    if mck in _MODEL_CACHE:
                        cached_model, cached_hash_c, cached_time_c = _MODEL_CACHE[mck]
                        if (
                            cached_hash_c == data_hash
                            and (time.time() - cached_time_c) < CACHE_TTL_SECONDS
                        ):
                            pm = cached_model
                    if pm is None:
                        pm = train_prophet(train_df)
                        _MODEL_CACHE[mck] = (pm, data_hash, time.time())
                    pr = predict_prophet(pm, future_df, periods=horizon_hours)
                    p = pr["forecast"][:horizon_hours]
                    _PREDICTION_CACHE[(region, horizon_hours, "prophet")] = (
                        p,
                        future_timestamps,
                        data_hash,
                        time.time(),
                    )
                    return "prophet", p

                def _forecast_arima():
                    from models.arima_model import predict_arima, train_arima

                    am = None
                    mck = (region, "arima", 0)
                    if mck in _MODEL_CACHE:
                        cached_model, cached_hash_c, cached_time_c = _MODEL_CACHE[mck]
                        if (
                            cached_hash_c == data_hash
                            and (time.time() - cached_time_c) < CACHE_TTL_SECONDS
                        ):
                            am = cached_model
                    if am is None:
                        am = train_arima(train_df)
                        _MODEL_CACHE[mck] = (am, data_hash, time.time())
                    # Fill NaN in exog columns to prevent SARIMAX forecast failure
                    safe_future = future_df.copy()
                    for col in [
                        "temperature_2m",
                        "wind_speed_80m",
                        "shortwave_radiation",
                        "cooling_degree_days",
                        "heating_degree_days",
                    ]:
                        if col in safe_future.columns:
                            safe_future[col] = safe_future[col].ffill().bfill().fillna(0)
                    p = predict_arima(am, safe_future, periods=horizon_hours)[:horizon_hours]
                    _PREDICTION_CACHE[(region, horizon_hours, "arima")] = (
                        p,
                        future_timestamps,
                        data_hash,
                        time.time(),
                    )
                    return "arima", p

                model_fns = {
                    "xgboost": _forecast_xgb,
                    "prophet": _forecast_prophet,
                    "arima": _forecast_arima,
                }
                with ThreadPoolExecutor(max_workers=3) as pool:
                    futures = {pool.submit(model_fns[m]): m for m in missing}
                    for future in as_completed(futures):
                        model_label = futures[future]
                        try:
                            name, pred = future.result()
                            preds[name] = pred
                        except Exception as e:
                            log.warning(
                                "forecast_ensemble_model_failed", model=model_label, error=str(e)
                            )

            log.info(
                "forecast_ensemble_combined",
                models=list(preds.keys()),
                count=len(preds),
                cached=len(preds) - len(missing) if missing else len(preds),
            )

            if preds:
                # Equal weights for forward forecast (no actuals to compute MAPE)
                all_preds = list(preds.values())
                predictions = np.mean(all_preds, axis=0)
            else:
                return {"error": "No models trained successfully"}
        else:
            return {"error": f"Unknown model: {model_name}"}

        # Cache predictions (in-memory)
        _PREDICTION_CACHE[cache_key] = (predictions, future_timestamps, data_hash, time.time())

        # Write-through to SQLite cache (survives page refresh / server restart)
        try:
            from data.cache import get_cache

            sqlite_cache = get_cache()
            sqlite_key = f"forecast:{region}:{horizon_hours}:{model_name}"
            serializable = {
                "cache_version": _CACHE_VERSION,
                "data_hash": data_hash,
                "timestamps": [str(t) for t in future_timestamps],
                "predictions": predictions.tolist()
                if hasattr(predictions, "tolist")
                else list(predictions),
            }
            sqlite_cache.set(sqlite_key, serializable, ttl=CACHE_TTL_SECONDS)
            log.debug(
                "forecast_sqlite_cache_written",
                region=region,
                horizon=horizon_hours,
                model=model_name,
            )
        except Exception as e:
            log.debug("forecast_sqlite_write_failed", error=str(e))

    except Exception as e:
        log.warning("outlook_model_failed", model=model_name, error=str(e))
        return {"error": str(e)}

    result = {
        "timestamps": future_timestamps,
        "predictions": predictions,
    }
    if shap_data is not None:
        result["shap_data"] = shap_data
    if xgb_model_dict is not None:
        result["model_dict"] = xgb_model_dict

    # Save snapshot for replay (NEXD-14)
    try:
        from config import feature_enabled

        if feature_enabled("forecast_replay"):
            from data.forecast_history import save_forecast_snapshot

            save_forecast_snapshot(
                region=region,
                horizon_hours=horizon_hours,
                model_name=model_name,
                timestamps=[str(t) for t in future_timestamps],
                predictions=predictions.tolist()
                if hasattr(predictions, "tolist")
                else list(predictions),
            )
    except Exception:
        log.debug("forecast_snapshot_save_failed", region=region, model=model_name)

    return result


def _create_future_features(
    train_df: pd.DataFrame, future_timestamps: pd.DatetimeIndex
) -> pd.DataFrame:
    """Create feature dataframe for future predictions.

    Fills weather, demand lag, and rolling features using historical
    hour-of-day + day-of-week averages from training data so that the
    model sees realistic daily/weekly patterns instead of a single frozen
    value repeated across the forecast horizon.

    The (hour, dow) group means are computed over the most recent
    ``_CLIMATOLOGY_WINDOW_DAYS`` of training data (full history when the
    recent slice is thinner than ``_CLIMATOLOGY_MIN_ROWS``) so the future
    features track the forecast season instead of regressing toward the
    cooler mean of the full ~90-day window — mirrors the #281/#282 fix in
    ``jobs.phases._build_future_feature_frame``.
    """
    feature_cols = [c for c in train_df.columns if c not in ["timestamp", "demand_mw", "region"]]

    future_df = pd.DataFrame({"timestamp": future_timestamps})

    # Time-based features (always computed from the actual future timestamps)
    future_df["hour"] = future_df["timestamp"].dt.hour
    future_df["day_of_week"] = future_df["timestamp"].dt.dayofweek
    future_df["month"] = future_df["timestamp"].dt.month
    future_df["day_of_year"] = future_df["timestamp"].dt.dayofyear
    future_df["hour_sin"] = np.sin(2 * np.pi * future_df["hour"] / 24)
    future_df["hour_cos"] = np.cos(2 * np.pi * future_df["hour"] / 24)
    future_df["dow_sin"] = np.sin(2 * np.pi * future_df["day_of_week"] / 7)
    future_df["dow_cos"] = np.cos(2 * np.pi * future_df["day_of_week"] / 7)
    future_df["is_weekend"] = (future_df["day_of_week"] >= 5).astype(int)
    # P2-14 (#273): calendar-derivable, computed directly — mirrors the
    # prod builder in jobs/phases.py so the dev inline path doesn't smear
    # is_holiday through the (hour, dow) imputer (same keep-in-sync
    # convention as the #291 recent-window mirror above).
    from data.feature_engineering import compute_holiday_flag

    future_df["is_holiday"] = compute_holiday_flag(future_df["timestamp"]).to_numpy()

    horizon = len(future_timestamps)
    last_row = train_df.iloc[-1]

    # Use historical (hour, day_of_week) averages so models see realistic
    # daily demand curves and weather patterns instead of a single frozen
    # value repeated for every future hour. Restrict to a recent trailing
    # window so the baseline tracks the forecast season rather than the
    # full-history mean (#281/#282); fall back to the full history when
    # the recent slice is too thin for stable (hour, dow) group means.
    hist = train_df.copy()
    if "timestamp" in hist.columns and len(hist):
        cutoff = hist["timestamp"].max() - pd.Timedelta(days=_CLIMATOLOGY_WINDOW_DAYS)
        recent = hist[hist["timestamp"] >= cutoff]
        if len(recent) >= _CLIMATOLOGY_MIN_ROWS:
            hist = recent.copy()
    hist["_hour"] = hist["timestamp"].dt.hour
    hist["_dow"] = hist["timestamp"].dt.dayofweek

    # Compute (hour, dow) group means for all numeric feature columns
    non_time_cols = [c for c in feature_cols if c not in future_df.columns]
    numeric_cols = [c for c in non_time_cols if c in hist.columns]

    group_means = hist.groupby(["_hour", "_dow"])[numeric_cols].mean()

    # Map future timestamps to their (hour, dow) historical averages
    future_hour = future_df["timestamp"].dt.hour
    future_dow = future_df["timestamp"].dt.dayofweek

    for col in numeric_cols:
        values = np.empty(horizon)
        for i in range(horizon):
            key = (future_hour.iloc[i], future_dow.iloc[i])
            if key in group_means.index:
                values[i] = group_means.loc[key, col]
            else:
                values[i] = last_row[col] if col in last_row.index else 0
        future_df[col] = values

    # Fill any remaining feature columns not in training data
    for col in feature_cols:
        if col not in future_df.columns:
            future_df[col] = 0

    return future_df


# #296: human-readable copy for the serve-time horizon guard's reason codes
# (jobs/phases.py writes them via models.evaluation.check_long_horizon_sanity).
_GUARD_REASON_COPY = {
    "below_recent_band": "its trajectory falls far below the recent demand range",
    "above_recent_band": "its trajectory climbs far above the recent demand range",
    "sustained_drift": "its trajectory drifts steadily away from recent demand",
    "non_finite": "it produced non-numeric values",
}


#: Label for a region served the seasonal-naive baseline instead of a model
#: (``models.skill``). Not a model name — every surface that resolves a
#: served model must be able to say "this is not a model" in the same breath.
BASELINE_SERIES_LABEL = "seasonal-naive baseline"


def is_baseline_served(cached: dict | None) -> bool:
    """Is this payload's headline series the baseline rather than a model?

    The scoring job substitutes a seasonal-naive series for regions whose
    model measurably loses to it (``models.skill.should_serve_baseline``).
    That series lands in ``predicted_demand_mw`` — the same key a model
    forecast uses — so any surface plotting it without checking this flag
    presents a baseline as a model, which is the failure the substitution
    exists to correct.
    """
    return isinstance(cached, dict) and cached.get("served_series") == "seasonal-naive"


def _served_model_for_payload(cached: dict, model_name: str) -> str:
    """Resolve which model's series the outlook chart plots for a payload
    and a dropdown selection (P2-26/#273).

    A requested model present in the rows serves itself. Only the
    "xgboost" selection can be substituted: when its column is absent the
    chart plots ``predicted_demand_mw``, which mirrors the payload's
    PRIMARY model (the first model that succeeded that scoring run).
    Legacy payloads without ``primary_model`` keep the xgboost attribution
    — nothing better is knowable for them.

    A region served the BASELINE resolves to neither: the headline series is
    not any model's output, so naming a model here would attribute a naive
    projection to a trained forecaster on every label the tab renders.

    Shared by the chart render path and the model-metrics card so the two
    can't disagree about which model the tab is describing.
    """
    forecasts = cached.get("forecasts") or []
    if is_baseline_served(cached) and (not forecasts or model_name not in forecasts[0]):
        return BASELINE_SERIES_LABEL
    if not forecasts or model_name in forecasts[0]:
        return model_name
    return str(cached.get("primary_model") or "xgboost")


def _guarded_outlook_state(
    region, model_name, horizon_hours, guard, data_through_str, ensemble_ok, requested_model=None
):
    """Honest unavailable state for a guard-withheld model+horizon (#296).

    The scoring job flagged this model's forecast as degenerate at this
    horizon (``horizon_guard`` on the Redis payload). Rendering the line
    anyway would draw fiction — SC/PSCO SARIMAX decayed to 0 MW, BPAT grew
    ~2x — so the chart states what happened and what still works instead.

    ``ensemble_ok`` gates the guidance copy: recommending "the Ensemble
    model" would be wrong when the ensemble itself is the flagged series,
    is flagged at this horizon too, or is absent from the payload.

    ``requested_model`` (P2-26/#273): when the withheld series is the
    payload primary substituting for a missing requested model, the copy
    must bridge the gap — otherwise the user selected XGBoost and gets a
    screen about PROPHET with no explanation.
    """
    max_ok = _guard_max_ok(guard) or 0
    reason_copy = _GUARD_REASON_COPY.get(
        str(guard.get("reason")), "it failed the long-horizon sanity check"
    )
    horizon_labels = {24: "24-hour", 168: "7-day", 720: "30-day"}
    label = horizon_labels.get(horizon_hours, f"{horizon_hours}h")
    if max_ok > 0:
        verified = (
            f"Verified up to {max_ok}h — try a shorter horizon or the Ensemble model."
            if ensemble_ok
            else f"Verified up to {max_ok}h — try a shorter horizon."
        )
    else:
        verified = (
            "Try the Ensemble model."
            if ensemble_ok
            else "Try a different model, or check back after the next scoring run."
        )
    substitution_note = ""
    if requested_model and requested_model != model_name:
        substitution_note = (
            f"<br>Requested {requested_model.upper()} is unavailable this scoring run — "
            f"{model_name.upper()} (payload primary) is the served series."
        )
    fig = go.Figure()
    fig.update_layout(**_layout(uirevision=f"{region}:{horizon_hours}"))
    fig.add_annotation(
        text=(
            f"{model_name.upper()} is withheld at the {label} horizon for {region}"
            f"<br><sup>The scoring job's long-horizon sanity guard flagged this "
            f"forecast — {reason_copy}.{substitution_note}<br>{verified}</sup>"
        ),
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(color="#71717a", size=14),  # tertiary — disclosure, not alarm
    )
    return (
        fig,
        data_through_str,
        "—",
        "",
        "—",
        "—",
        "",
        "—",
        html.Div(),
    )


def _outlook_tab_from_redis(
    region, horizon_hours, model_name, demand_json, weather_json, persona_id
):
    """Redis fast path for the outlook (demand forecast) tab.

    Returns a 9-tuple (fig, data_through, peak_str, peak_time, avg_str,
    min_str, min_time, range_str, insight_card) or None if cache miss
    or insufficient data.
    """
    granularity = "1h"
    cached = redis_get(redis_key(f"forecast:{region}:{granularity}"))
    if cached is None or not cached.get("forecasts"):
        return None

    log.info("outlook_redis_hit", region=region, granularity=granularity)
    forecasts = cached["forecasts"]

    # Model availability check: skip Redis if requested model isn't stored.
    # "xgboost" is exempt for back-compat: it falls back to the payload's
    # ``predicted_demand_mw`` primary series below. Every other model miss
    # returns None (warming / inline fallback).
    if model_name != "xgboost" and model_name not in forecasts[0]:
        log.info("outlook_redis_model_miss", model=model_name)
        return None

    timestamps = pd.to_datetime([f["timestamp"] for f in forecasts])
    pred_key = model_name if model_name in forecasts[0] else "predicted_demand_mw"

    # P2-26 (#273): resolve the model whose numbers are ACTUALLY plotted.
    # When the xgboost exemption above falls back to ``predicted_demand_mw``,
    # that column mirrors the payload's PRIMARY model — the first model that
    # succeeded that scoring run, not necessarily XGBoost. Every label on
    # this chart (trace name, title, band calibration, insights, withheld
    # state) must name the served model, never the requested one — the old
    # behavior titled another model's series "XGBOOST Demand Forecast" and
    # wrapped XGBoost-calibrated bands around it.
    served_model = _served_model_for_payload(cached, model_name)

    # #296: honor the scoring job's serve-time horizon guard. When the
    # series about to be drawn is flagged as degenerate at this horizon,
    # render an honest withheld state instead of the line. The guard map
    # is keyed by model name — i.e. by ``served_model``. Malformed guard
    # shapes fail OPEN (normal render + warning log) — see ``_guard_max_ok``.
    guard_map = cached.get("horizon_guard")
    if not isinstance(guard_map, dict):
        if guard_map is not None:
            log.warning("outlook_horizon_guard_malformed", region=region, shape="map")
        guard_map = {}
    guard = guard_map.get(served_model)
    max_ok = _guard_max_ok(guard)
    if guard is not None and max_ok is None:
        log.warning("outlook_horizon_guard_malformed", region=region, model=served_model)
    if max_ok is not None and horizon_hours > max_ok:
        log.info(
            "outlook_horizon_guard_withheld",
            region=region,
            model=served_model,
            horizon=horizon_hours,
            reason=guard.get("reason"),
        )
        # Recommend the ensemble only when it exists in this payload, is a
        # different series than the one being withheld, and is itself
        # unflagged (or verified) at this horizon.
        ensemble_guard = guard_map.get("ensemble")
        ensemble_max_ok = _guard_max_ok(ensemble_guard)
        ensemble_ok = (
            served_model != "ensemble"
            and "ensemble" in forecasts[0]
            and (
                ensemble_guard is None
                or (ensemble_max_ok is not None and horizon_hours <= ensemble_max_ok)
            )
        )
        data_through_str = cached.get("scored_at", "Unknown")
        if data_through_str != "Unknown":
            import contextlib

            with contextlib.suppress(Exception):
                data_through_str = pd.Timestamp(data_through_str).strftime("%Y-%m-%d %H:%M UTC")
        # Name the series that is actually flagged (P2-26): when the primary
        # substitutes for a missing xgboost column, the withheld copy must
        # attribute the degeneracy to the primary, not to XGBoost.
        return _guarded_outlook_state(
            region,
            served_model,
            horizon_hours,
            guard,
            data_through_str,
            ensemble_ok,
            requested_model=model_name,
        )

    predictions = np.array([f.get(pred_key, f.get("predicted_demand_mw", 0)) for f in forecasts])

    # Sufficiency check: Redis must cover the requested horizon
    if len(predictions) < horizon_hours:
        log.warning(
            "outlook_redis_insufficient",
            region=region,
            available=len(predictions),
            requested=horizon_hours,
        )
        return None

    # Log the substitution only once it will actually render (an
    # insufficient payload above falls through to the degraded states —
    # unavailable_selection in prod, inline compute in dev — so an earlier
    # log would record a serve that never happened).
    if served_model != model_name:
        log.info(
            "outlook_serving_primary_substitute",
            region=region,
            requested=model_name,
            served=served_model,
        )

    # Limit to requested horizon
    if len(predictions) > horizon_hours:
        timestamps = timestamps[:horizon_hours]
        predictions = predictions[:horizon_hours]

    data_through_str = cached.get("scored_at", "Unknown")
    if data_through_str != "Unknown":
        import contextlib

        with contextlib.suppress(Exception):
            data_through_str = pd.Timestamp(data_through_str).strftime("%Y-%m-%d %H:%M UTC")

    peak_val = float(np.max(predictions))
    peak_idx = int(np.argmax(predictions))
    peak_time = timestamps[peak_idx].strftime("%a %H:%M")
    min_val = float(np.min(predictions))
    min_idx = int(np.argmin(predictions))
    min_time = timestamps[min_idx].strftime("%a %H:%M")
    avg_val = float(np.mean(predictions))
    range_val = peak_val - min_val

    fig = go.Figure()
    # P2-26 (#273): style + name by the SERVED model, never the requested one.
    model_style = LINE_STYLES.get(
        served_model, {"color": COLORS["ensemble"], "width": 2, "dash": "solid"}
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=predictions,
            mode="lines",
            name=(
                BASELINE_SERIES_LABEL.title()
                if served_model == BASELINE_SERIES_LABEL
                else f"{served_model.upper()} Forecast"
            ),
            line=dict(
                color=COLORS.get(served_model, COLORS["ensemble"]),
                width=model_style.get("width", 2),
                dash=model_style.get("dash", "solid"),
            ),
            fill="tozeroy",
            fillcolor="rgba(56,208,255,0.10)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[timestamps[peak_idx]],
            y=[peak_val],
            mode="markers+text",
            name="Peak",
            marker=dict(color="#FF5C7A", size=12, symbol="triangle-up"),
            text=[f"Peak: {peak_val:,.0f} MW"],
            textposition="top center",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[timestamps[min_idx]],
            y=[min_val],
            mode="markers+text",
            name="Min",
            marker=dict(color="#3b82f6", size=10, symbol="triangle-down"),
            text=[f"Min: {min_val:,.0f} MW"],
            textposition="bottom center",
            showlegend=False,
        )
    )
    interval_meta = _add_confidence_bands(
        fig, timestamps, predictions, horizon_hours, region=region, model_name=served_model
    )
    _add_trailing_actuals(fig, demand_json)
    # Mark the Open-Meteo / climatology boundary on long-horizon views.
    # Only the 30-day view actually crosses the day-16 boundary; on 24h
    # and 7-day views the helper is a no-op. See ADR-008.
    has_climatology_segment = _add_forecast_horizon_divider(fig, timestamps, horizon_hours)
    horizon_labels = {24: "24-Hour", 168: "7-Day", 720: "30-Day"}
    interval_caption = _interval_caption(interval_meta, served_model)
    # On the 30-day view, surface in the subtitle that days 17-30 are
    # climatology baseline rather than real forecast. Users browsing
    # the chart shouldn't have to hover the divider line to understand
    # the regime split. See ADR-008.
    horizon_caption = ""
    if has_climatology_segment:
        # "Seasonal climatology baseline" is the honest umbrella for both tail
        # modes during the artifact backfill: the (day_of_year, hour)
        # weather-normal where a BA's artifact exists (#283), the recent-28d
        # (hour, dow) climatology otherwise. The web tier can't tell which the
        # scoring job used per-BA, so the label claims only what is always true.
        horizon_caption = (
            "<br><sup>Days 1-16: real Open-Meteo forecast · "
            "Days 17-30: seasonal climatology baseline</sup>"
        )
    # P2-26 (#273): when the primary substitutes for a missing xgboost
    # column, say so on the chart — the title alone naming the served model
    # doesn't explain why the requested model isn't shown.
    substitution_caption = ""
    if is_baseline_served(cached):
        # Disclose the substitution whichever series is plotted. Two distinct
        # statements, and conflating them is how the first cut of this went
        # wrong: what this REGION is served (the baseline), and what this
        # CHART is drawing (whatever the model selector asked for). A chart
        # titled "baseline" over a model's line is the same class of lie the
        # substitution exists to correct.
        reason = str(cached.get("served_reason") or "")
        plotted_is_baseline = served_model == BASELINE_SERIES_LABEL
        substitution_caption = (
            "<br><sup><b>"
            + (
                "Not a model forecast."
                if plotted_is_baseline
                else f"Shown: {served_model.upper()}, for comparison only."
            )
            + "</b> GridPulse serves this region a seasonal-naive baseline — the "
            "same clock hour from the most recent observed day — because its "
            f"trained models measurably lose to it{(': ' + reason) if reason else ''}."
            + ("" if plotted_is_baseline else " That baseline is the served series, not this line.")
            + "</sup>"
        )
    elif served_model != model_name:
        substitution_caption = (
            f"<br><sup>Requested {model_name.upper()} unavailable this scoring run — "
            f"showing {served_model.upper()} (payload primary)</sup>"
        )
    fig.update_layout(
        **_layout(
            uirevision=f"{region}:{horizon_hours}",
            title=(
                f"{horizon_labels.get(horizon_hours, '')} "
                f"{BASELINE_SERIES_LABEL.title() if served_model == BASELINE_SERIES_LABEL else served_model.upper()}"
                f" Demand Forecast — {region}"
                f"{substitution_caption}{interval_caption}{horizon_caption}"
            ),
            xaxis_title="Date/Time",
            yaxis_title="Demand (MW)",
            hovermode="x unified",
        )
    )

    from components.insights import build_insight_card, generate_tab2_insights

    persona = persona_id or "grid_ops"
    weather_df = pd.read_json(io.StringIO(weather_json)) if weather_json else pd.DataFrame()
    tab2_insights = generate_tab2_insights(
        persona,
        region or "FPL",
        predictions,
        timestamps,
        model_name=served_model,
        horizon_hours=horizon_hours,
        weather_df=weather_df,
    )
    insight_card = build_insight_card(tab2_insights, persona, "tab-outlook")

    return (
        fig,
        data_through_str,
        f"{peak_val:,.0f} MW",
        peak_time,
        f"{avg_val:,.0f} MW",
        f"{min_val:,.0f} MW",
        min_time,
        f"{range_val:,.0f} MW",
        insight_card,
    )


# ── Callback registration (Step 10c — register_callbacks split) ──────


def register_forecast_callbacks(app):
    """Register Forecast / Demand Outlook tab callbacks with the Dash app.

    Step 10c of the ``register_callbacks`` decomposition (issue #87).
    Owns the entire Forecast tab callback surface:

    * Three clientside-callback panel toggles (drivers / generation /
      scenarios collapse open/close).
    * Three inline panel content callbacks (drivers / generation /
      scenarios — all lazy: only fire when collapse is open and the
      Forecast tab is active).
    * Scenario preset handler + three clientside slider readouts.
    * Page title + model metrics card.
    * The big 9-output ``update_demand_outlook`` callback with Redis
      fast path (``_outlook_tab_from_redis``) and v1 compute fallback
      (``_run_forecast_outlook``).
    * Two Forecast-Replay callbacks (NEXD-14 selector + overlay).

    The three panel builders (drivers / generation / scenarios) now live
    in THIS module. They were retained by the Overview module through
    Step 7b/7c and imported lazily from here to keep the dependency
    one-way; §1c of the GP-P1-04 proposal moved them to the tab that has
    always rendered them, so there is no cross-module import left to
    keep one-way.
    """
    from components.cards import build_model_metrics_card, build_page_title
    from config import REGION_NAMES

    # ── FORECAST TAB (R4a — v2 linear stack + inline panels) ────
    # Hero chart + 4-up MetricsBar + InsightCard are still driven by
    # ``update_demand_outlook`` below (existing 9-output callback,
    # preserved). Small new callbacks fill the v2 title block, the
    # ModelMetricsCard slot, and (R4a-2) the inline Drivers panel
    # rendered when its collapse opens.

    # 3 clientside toggles — flip is_open on each panel collapse.
    # Generic JS could pattern-match, but explicit is clearer.
    for _panel_key in ("drivers", "generation", "scenarios"):
        app.clientside_callback(
            "function(n, is_open) { return n ? !is_open : is_open; }",
            Output(f"forecast-panel-{_panel_key}-collapse", "is_open"),
            Input(f"forecast-panel-toggle-{_panel_key}", "n_clicks"),
            State(f"forecast-panel-{_panel_key}-collapse", "is_open"),
            prevent_initial_call=True,
        )

    @app.callback(
        Output("forecast-drivers-content", "children"),
        [
            Input("forecast-panel-drivers-collapse", "is_open"),
            Input("weather-store", "data"),
        ],
        State("dashboard-tabs", "active_tab"),
    )
    def update_forecast_drivers_panel(is_open, weather_json, active_tab):
        """Render the 3-up Drivers KPI grid (Temperature / Wind / Solar).

        Lazy: only computes when the collapse is open and the user is
        on the Forecast tab (avoid spending render cost while collapsed).
        """
        if active_tab != "tab-outlook" or not is_open:
            return no_update
        return _build_drivers_panel(weather_json)

    @app.callback(
        Output("forecast-generation-content", "children"),
        [
            Input("forecast-panel-generation-collapse", "is_open"),
            Input("region-selector", "value"),
            Input("demand-store", "data"),
        ],
        State("dashboard-tabs", "active_tab"),
    )
    def update_forecast_generation_panel(is_open, region, demand_json, active_tab):
        """Render the stacked-area fuel mix + 3-up sub-MetricsBar.

        Lazy: only computes when the collapse is open and the user is
        on the Forecast tab.
        """
        if active_tab != "tab-outlook" or not is_open:
            return no_update
        return _build_generation_panel(region, demand_json)

    # Scenarios panel — preset chip click writes deltas into the 3 sliders.
    @app.callback(
        [
            Output("forecast-scn-temp", "value"),
            Output("forecast-scn-wind", "value"),
            Output("forecast-scn-solar", "value"),
        ],
        Input({"type": "scenario-preset", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def apply_scenario_preset(_clicks):
        """Apply a preset's temperature/wind/solar deltas to the three sliders."""
        from components.tab_demand_outlook import _SCENARIO_PRESETS

        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or "index" not in triggered:
            return no_update, no_update, no_update
        # Ignore noop dispatches with all-zero clicks
        if not any(c for c in (_clicks or []) if c):
            return no_update, no_update, no_update
        preset = _SCENARIO_PRESETS.get(triggered["index"])
        if not preset:
            return no_update, no_update, no_update
        deltas = preset["deltas"]
        return deltas["temp"], deltas["wind"], deltas["solar"]

    # Slider readouts (clientside — instant, no Python round-trip)
    for _key, _unit in (("temp", "°F"), ("wind", "mph"), ("solar", "W/m²")):
        app.clientside_callback(
            f"function(v) {{ if (v === null || v === undefined) return '0 {_unit}'; "
            f"const sign = v > 0 ? '+' : ''; return sign + v + ' {_unit}'; }}",
            Output(f"forecast-scn-{_key}-readout", "children"),
            Input(f"forecast-scn-{_key}", "value"),
        )

    @app.callback(
        [
            Output("forecast-scenarios-kpis", "children"),
            Output("forecast-scenarios-chart", "figure"),
        ],
        [
            Input("forecast-panel-scenarios-collapse", "is_open"),
            Input("forecast-scn-temp", "value"),
            Input("forecast-scn-wind", "value"),
            Input("forecast-scn-solar", "value"),
            Input("region-selector", "value"),
            Input("demand-store", "data"),
            Input("dashboard-tabs", "active_tab"),
        ],
    )
    def update_forecast_scenarios_panel(
        is_open, temp_d, wind_d, solar_d, region, demand_json, active_tab
    ):
        """Render the 4-up delta MetricsBar + baseline-vs-scenario chart.

        Lazy: only renders when the collapse is open and the user is on the
        Forecast tab.

        ``active_tab`` is an INPUT, not State. As State the callback never
        fired on a tab change, so arriving at the Forecast tab with the panel
        already open — which is what a bookmark, a reload, or opening the
        panel from another tab all produce — left the KPIs permanently blank
        until the user happened to move a slider. Observed live on
        2026-08-10: panel rendered, sliders responded, KPI row empty.

        Values come from the precomputed scenario grid when the scoring job
        has written one (#127), and from the #119 linear heuristic when it has
        not. ``_scenario_factors`` reports which, and the panel says so rather
        than claiming one unconditionally.
        """
        if active_tab != "tab-outlook" or not is_open:
            return no_update, no_update
        return _build_scenarios_panel(temp_d, wind_d, solar_d, region, demand_json)

    @app.callback(
        Output("outlook-title", "children"),
        [
            Input("region-selector", "value"),
            Input("dashboard-tabs", "active_tab"),
        ],
    )
    def update_outlook_title(region, active_tab):
        """Page-title block for the Forecast tab."""
        if active_tab != "tab-outlook":
            return no_update

        region = region or "FPL"
        region_name = REGION_NAMES.get(region, region)
        return build_page_title(
            "Forecast",
            f"24h–30d demand outlook with confidence bands · {region_name}",
        )

    @app.callback(
        Output("outlook-model-card", "children"),
        [
            Input("region-selector", "value"),
            Input("outlook-model", "value"),
            Input("dashboard-tabs", "active_tab"),
        ],
    )
    def update_outlook_model_card(region, model_name, active_tab):
        """Render the horizontal MAPE/RMSE/MAE/R² bar for the active model.

        P2-26 (#273): the card must describe the model the chart below it
        actually plots. When the xgboost selection is substituted by the
        payload primary (see ``_served_model_for_payload``), showing
        XGBoost's MAPE above a PROPHET-served chart invites the reader to
        attribute one model's accuracy to another's line.
        """
        if active_tab != "tab-outlook":
            return no_update

        try:
            from models.model_service import get_model_metrics, is_trained
        except ImportError:
            return html.Div()

        region = region or "FPL"
        try:
            cached = redis_get(redis_key(f"forecast:{region}:1h"))
            if isinstance(cached, dict):
                model_name = _served_model_for_payload(cached, model_name)
        except Exception:  # pragma: no cover — defensive; card keeps dropdown model
            pass
        metrics_dict = get_model_metrics(region) or {}
        if model_name not in metrics_dict:
            # Fall back to any available model
            if not metrics_dict:
                return html.Div()
            model_name = next(iter(metrics_dict.keys()))

        m = metrics_dict[model_name]

        def _fmt(key: str, spec: str, suffix: str = "") -> str:
            # An absent metric must render as unavailable, not a perfect 0
            # (partial metric dicts are a supported prod payload state; #201).
            value = m.get(key)
            if value is None:
                return "—"
            return f"{value:{spec}}{suffix}"

        formatted = {
            "MAPE": _fmt("mape", ".1f", "%"),
            "RMSE": _fmt("rmse", ",.0f", " MW"),
            "MAE": _fmt("mae", ",.0f", " MW"),
            "R²": _fmt("r2", ".3f"),
        }
        name = model_display_name(model_name)
        badge = "trained" if is_trained(region) else "simulated"
        return build_model_metrics_card(model_name=name, metrics=formatted, badge=badge)

    @app.callback(
        [
            Output("outlook-chart", "figure"),
            Output("outlook-data-through", "children"),
            Output("outlook-peak", "children"),
            Output("outlook-peak-time", "children"),
            Output("outlook-avg", "children"),
            Output("outlook-min", "children"),
            Output("outlook-min-time", "children"),
            Output("outlook-range", "children"),
            Output("tab2-insight-card", "children"),
        ],
        [
            Input("outlook-horizon", "value"),
            Input("outlook-model", "value"),
            Input("dashboard-tabs", "active_tab"),
            Input("demand-store", "data"),
            Input("persona-selector", "value"),
        ],
        [
            State("weather-store", "data"),
            State("region-selector", "value"),
        ],
        prevent_initial_call=True,
    )
    def update_demand_outlook(
        horizon,
        model_name,
        active_tab,
        demand_json,
        persona_id,
        weather_json,
        region,
    ):
        """Generate forward-looking demand forecast."""
        # Only run when this tab is active — avoids 10s+ model training on page load
        if active_tab != "tab-outlook":
            return [no_update] * 9

        log.info("outlook_callback_start", horizon=horizon, model=model_name, region=region)

        horizon_hours = int(horizon)
        empty_insight = html.Div()

        # ── v2 Redis fast path ──────────────────────────────
        if region:
            redis_result = _outlook_tab_from_redis(
                region, horizon_hours, model_name, demand_json, weather_json, persona_id
            )
            if redis_result is not None:
                return redis_result

        # uirevision keyed on region + horizon so zoom/legend state persists
        # across data refresh but resets when the user picks a new horizon.
        uirev = f"{region}:{horizon_hours}"

        # ── v1 compute fallback ─────────────────────────────
        if not demand_json or not weather_json:
            fig = go.Figure()
            fig.update_layout(**_layout(uirevision=uirev))
            fig.add_annotation(
                text="Loading data...", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
            return (
                fig,
                "Loading...",
                "Loading...",
                "",
                "Loading...",
                "Loading...",
                "",
                "Loading...",
                empty_insight,
            )

        try:
            demand_df = pd.read_json(io.StringIO(demand_json))
            weather_df = pd.read_json(io.StringIO(weather_json))
        except Exception as e:
            log.error("outlook_parse_error", error=str(e))
            fig = go.Figure()
            fig.update_layout(**_layout(uirevision=uirev))
            return (
                fig,
                "Error",
                "No data",
                "",
                "No data",
                "No data",
                "",
                "No data",
                empty_insight,
            )

        # Get the data through date (last timestamp in demand data)
        demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])
        data_through = demand_df["timestamp"].max()
        data_through_str = data_through.strftime("%Y-%m-%d %H:%M UTC")

        # Run the forecast
        result = _run_forecast_outlook(demand_df, weather_df, horizon_hours, model_name, region)

        if "error" in result:
            # Soften the warming case (pipeline still populating Redis after
            # a deploy / cache eviction) — that's an expected transient state,
            # not a hard failure. The two unavailable cases (P2-35/#273) are
            # calm but claim exactly the permanence their evidence supports:
            # a missing SELECTION in an existing payload is a per-run state
            # (a one-tick model failure heals next hour), while a missing
            # forecast KEY after a completed pass will not self-heal. Keep
            # the loud message for genuine errors.
            is_warming = result["error"] == "warming"
            is_unavailable = result["error"] == "unavailable"
            is_selection = result["error"] == "unavailable_selection"
            if is_warming:
                text = "Pipeline is warming up — forecast will appear shortly"
            elif is_selection:
                text = (
                    f"No {model_name.upper()} data in the current scoring payload"
                    f" for {region or 'this region'}"
                    "<br><sup>The pipeline is live; a single-run model failure heals on "
                    "the next hourly run. If this persists across runs, the model isn't "
                    "training or its forecast phase is failing.</sup>"
                )
            elif is_unavailable:
                text = (
                    f"Forecast unavailable for {region or 'this region'}"
                    "<br><sup>The data pipeline is live and a full scoring pass has "
                    "completed, but no forecast exists for this region — typically its "
                    "models haven't trained yet or its forecast phase is failing. "
                    "This won't resolve on its own.</sup>"
                )
            else:
                text = f"Forecast failed: {result['error']}"
            soft = is_warming or is_unavailable or is_selection
            color = "#71717a" if soft else "#f87171"  # tertiary | danger
            fig = go.Figure()
            fig.update_layout(**_layout(uirevision=uirev))
            fig.add_annotation(
                text=text,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(color=color, size=14),
            )
            return (
                fig,
                data_through_str,
                "No data",
                "",
                "No data",
                "No data",
                "",
                "No data",
                empty_insight,
            )

        timestamps = pd.to_datetime(result["timestamps"])
        predictions = result["predictions"]

        # Build per-point tooltip strings (NEXD-13)
        tooltips = None
        try:
            from config import feature_enabled

            if feature_enabled("inline_tooltips"):
                from data.explainability import build_tooltip_strings

                shap_data = result.get("shap_data")
                tooltips = build_tooltip_strings(
                    shap_values=shap_data.get("shap_values") if shap_data else None,
                    feature_names=shap_data.get("feature_names") if shap_data else None,
                    model_dict=result.get("model_dict"),
                    n_points=len(predictions),
                    model_name=model_name,
                )
        except Exception:
            log.debug("tooltip_build_skipped")

        # Calculate KPIs
        peak_val = np.max(predictions)
        peak_idx = np.argmax(predictions)
        peak_time = timestamps[peak_idx].strftime("%a %H:%M")

        min_val = np.min(predictions)
        min_idx = np.argmin(predictions)
        min_time = timestamps[min_idx].strftime("%a %H:%M")

        avg_val = np.mean(predictions)
        range_val = peak_val - min_val

        # Build chart
        fig = go.Figure()

        # Forecast line (model-aware color + dash pattern)
        model_style = LINE_STYLES.get(
            model_name, {"color": COLORS["ensemble"], "width": 2, "dash": "solid"}
        )
        # Forecast trace with optional SHAP tooltips (NEXD-13)
        forecast_kwargs: dict = dict(
            x=timestamps,
            y=predictions,
            mode="lines",
            name=f"{model_name.upper()} Forecast",
            line=dict(
                color=COLORS.get(model_name, COLORS["ensemble"]),
                width=model_style.get("width", 2),
                dash=model_style.get("dash", "solid"),
            ),
            fill="tozeroy",
            fillcolor="rgba(56,208,255,0.10)",
        )
        if tooltips and any(tooltips):
            forecast_kwargs["customdata"] = tooltips
            forecast_kwargs["hovertemplate"] = (
                "<b>%{x|%a %b %d %H:%M}</b><br>Demand: %{y:,.0f} MW<br>%{customdata}<extra></extra>"
            )
        fig.add_trace(go.Scatter(**forecast_kwargs))

        # Add peak marker
        fig.add_trace(
            go.Scatter(
                x=[timestamps[peak_idx]],
                y=[peak_val],
                mode="markers+text",
                name="Peak",
                marker=dict(color="#FF5C7A", size=12, symbol="triangle-up"),
                text=[f"Peak: {peak_val:,.0f} MW"],
                textposition="top center",
                showlegend=False,
            )
        )
        # Add min marker
        fig.add_trace(
            go.Scatter(
                x=[timestamps[min_idx]],
                y=[min_val],
                mode="markers+text",
                name="Min",
                marker=dict(color="#3b82f6", size=10, symbol="triangle-down"),
                text=[f"Min: {min_val:,.0f} MW"],
                textposition="bottom center",
                showlegend=False,
            )
        )
        interval_meta = _add_confidence_bands(
            fig, timestamps, predictions, horizon_hours, region=region, model_name=model_name
        )
        _add_trailing_actuals(fig, demand_json)

        # Layout
        horizon_labels = {24: "24-Hour", 168: "7-Day", 720: "30-Day"}
        interval_caption = _interval_caption(interval_meta, model_name)
        fig.update_layout(
            **_layout(
                uirevision=uirev,
                title=(
                    f"{horizon_labels.get(horizon_hours, '')} {model_name.upper()} Demand Forecast — {region}"
                    f"{interval_caption}"
                ),
                xaxis_title="Date/Time",
                yaxis_title="Demand (MW)",
                hovermode="x unified",
            )
        )

        # Format KPI strings
        peak_str = f"{peak_val:,.0f} MW"
        avg_str = f"{avg_val:,.0f} MW"
        min_str = f"{min_val:,.0f} MW"
        range_str = f"{range_val:,.0f} MW"

        # Generate insights
        from components.insights import build_insight_card, generate_tab2_insights

        persona = persona_id or "grid_ops"
        tab2_insights = generate_tab2_insights(
            persona,
            region or "FPL",
            predictions,
            timestamps,
            model_name=model_name,
            horizon_hours=horizon_hours,
            weather_df=weather_df,
        )
        insight_card = build_insight_card(tab2_insights, persona, "tab-outlook")

        log.info("outlook_callback_complete", horizon=horizon_hours, peak=peak_str)
        return (
            fig,
            data_through_str,
            peak_str,
            peak_time,
            avg_str,
            min_str,
            min_time,
            range_str,
            insight_card,
        )

    # ── FORECAST REPLAY SELECTOR (NEXD-14) ──────────────────────

    @app.callback(
        [
            Output("replay-selector", "options"),
            Output("replay-selector", "value"),
            Output("replay-container", "style"),
        ],
        [
            Input("outlook-horizon", "value"),
            Input("outlook-model", "value"),
            Input("dashboard-tabs", "active_tab"),
            Input("region-selector", "value"),
        ],
        prevent_initial_call=True,
    )
    def populate_replay_selector(horizon, model_name, active_tab, region):
        """Populate the replay dropdown with available forecast snapshots."""
        from config import feature_enabled

        hidden = {"display": "none"}
        default_opts = [{"label": "Current", "value": "current"}]

        if active_tab != "tab-outlook" or not feature_enabled("forecast_replay"):
            return default_opts, "current", hidden

        try:
            from data.forecast_history import build_replay_options

            horizon_hours = int(horizon) if horizon else 168
            options = build_replay_options(region or "FPL", horizon_hours, model_name or "xgboost")
            # Hide if only "Current" (no historical snapshots to compare)
            visible = {"display": "block"} if len(options) > 1 else hidden
            return options, "current", visible
        except Exception:
            log.debug("replay_selector_populate_failed")
            return default_opts, "current", hidden

    # ── FORECAST REPLAY OVERLAY (NEXD-14) ───────────────────────

    @app.callback(
        [
            Output("outlook-chart", "figure", allow_duplicate=True),
            Output("replay-label", "children"),
        ],
        [Input("replay-selector", "value")],
        [
            State("outlook-chart", "figure"),
            State("outlook-horizon", "value"),
            State("outlook-model", "value"),
            State("region-selector", "value"),
        ],
        prevent_initial_call=True,
    )
    def overlay_replay_snapshot(replay_value, current_fig, horizon, model_name, region):
        """Overlay a historical forecast snapshot on the current chart.

        This is a lightweight callback — it only reads from SQLite and
        patches the existing figure.  It never recomputes a forecast.
        """
        if current_fig is None:
            return no_update, ""

        # Fast path: if "current" and no replay traces exist, skip figure round-trip
        has_replay_traces = any(
            (t.get("name") or "").startswith("Forecast from ")
            for t in (current_fig.get("data") or [])
        )
        if (not replay_value or replay_value == "current") and not has_replay_traces:
            return no_update, ""

        fig = go.Figure(current_fig)

        # Strip any previously added replay traces
        fig.data = [t for t in fig.data if not (t.name or "").startswith("Forecast from ")]

        if not replay_value or replay_value == "current":
            return fig, ""

        try:
            from config import feature_enabled

            if not feature_enabled("forecast_replay"):
                return fig, ""

            from data.forecast_history import get_forecast_snapshot

            horizon_hours = int(horizon) if horizon else 168
            snap = get_forecast_snapshot(
                region or "FPL", horizon_hours, model_name or "xgboost", replay_value
            )
            if snap:
                from datetime import datetime as _dt

                try:
                    snap_label = _dt.fromisoformat(snap["scored_at"]).strftime("%b %d %H:%M UTC")
                except (ValueError, TypeError):
                    snap_label = snap["scored_at"][:16]

                fig.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(snap["timestamps"]),
                        y=snap["predictions"],
                        mode="lines",
                        name=f"Forecast from {snap_label}",
                        line=dict(color="#A8B3C7", width=2, dash="dash"),
                        opacity=0.6,
                    )
                )
                return fig, f"Comparing with forecast from {snap_label}"
        except Exception:
            log.debug("replay_overlay_failed")

        return fig, ""


# ── Relocated from ``_callbacks_overview`` (§1c of the GP-P1-04 proposal) ──
#
# Drivers, generation and scenarios render THIS tab's collapsible panels.
# They were prototyped in the Overview module and every caller has always
# been here; the import across the boundary was the anomaly.


def _generation_df_from_redis(region: str) -> pd.DataFrame | None:
    """Read the scoring job's ``gridpulse:generation:{region}`` payload and
    unpivot it to the long ``[timestamp, fuel_type, generation_mw, region]``
    frame the Generation panel expects.

    The scoring job writes a wide payload (``{timestamps, <fuel>: [...],
    renewable_pct: [...]}``, fuel names already normalized); this reverses that
    pivot so the web tier can render generation without touching EIA (#199).
    """
    payload = redis_get(redis_key(f"generation:{region}"))
    if not isinstance(payload, dict):
        return None
    timestamps = payload.get("timestamps")
    if not timestamps:
        return None
    skip = {"region", "timestamps", "renewable_pct", "scored_at"}
    rows: list[dict] = []
    for fuel, vals in payload.items():
        if fuel in skip or not isinstance(vals, list):
            continue
        for ts, mw in zip(timestamps, vals, strict=False):
            rows.append({"timestamp": ts, "fuel_type": fuel, "generation_mw": mw, "region": region})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _fetch_generation_cached(region: str) -> pd.DataFrame | None:
    """Return generation-by-fuel for a region, Redis-first.

    The stateless web tier must not fetch EIA in the request path — the scoring
    job writes ``gridpulse:generation:{region}`` hourly, and this reads it
    (#199 / the CLAUDE.md post-#130 web-tier I/O guardrail). Under
    ``REQUIRE_REDIS`` (staging/prod) a Redis miss returns None (warming state),
    never a live EIA call. The in-memory + EIA fetch tiers run only in
    development, where the scoring job may not be populating Redis.

    Returns a DataFrame ``[timestamp, fuel_type, generation_mw, region]`` or None.
    """
    import time as _time

    # Redis fast path (the only prod path).
    redis_df = _generation_df_from_redis(region)
    if redis_df is not None and not redis_df.empty:
        return redis_df

    if REQUIRE_REDIS:
        log.info("generation_warming", region=region)
        return None

    # ── development-only fallback (no scoring job populating Redis) ──
    # Tier 1: In-memory cache (5-minute TTL)
    if region in _GENERATION_CACHE:
        cached_df, cached_ts = _GENERATION_CACHE[region]
        if (_time.time() - cached_ts) < 300:
            log.info("generation_memory_cache_hit", region=region)
            return cached_df

    # Tier 2+3: fetch_generation_by_fuel handles SQLite cache + API call
    try:
        from config import EIA_API_KEY

        if EIA_API_KEY and EIA_API_KEY != "your_eia_api_key_here":
            from data.eia_client import fetch_generation_by_fuel

            gen_df = fetch_generation_by_fuel(region)
            if gen_df is not None and not gen_df.empty:
                # Normalize fuel type codes
                gen_df["fuel_type"] = (
                    gen_df["fuel_type"].map(_EIA_FUEL_MAP).fillna(gen_df["fuel_type"].str.lower())
                )
                _GENERATION_CACHE[region] = (gen_df, _time.time())
                log.info("generation_eia_fetched", region=region, rows=len(gen_df))
                return gen_df
    except Exception as e:
        log.warning("generation_eia_failed", region=region, error=str(e))

    # No demo fallback — return None so callers show "No data" or use
    # whatever is already in Redis rather than overwriting with fake values.
    log.warning("generation_no_data", region=region)
    return None


def _build_drivers_panel(weather_json: str | None) -> list:
    """3-up KPI cells (Temperature / Wind / Solar) with current value + 24h sparkline.

    The Forecast tab's Drivers inline panel calls this when its collapse
    opens. Each cell is a .gp-driver-cell with eyebrow / value / unit /
    sparkline. Sparkline reuses the same v2 minimal-axis style as
    _build_overview_sparkline.
    """
    if not weather_json:
        return _drivers_empty()

    try:
        wdf = pd.read_json(io.StringIO(weather_json))
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("forecast_drivers_parse_failed", error=str(exc))
        return _drivers_empty()

    if wdf.empty or "timestamp" not in wdf.columns:
        return _drivers_empty()

    wdf = wdf.copy()
    wdf["timestamp"] = pd.to_datetime(wdf["timestamp"])
    wdf = wdf.sort_values("timestamp")
    # Window: latest 24 rows (assume hourly cadence)
    horizon = wdf.tail(24)

    drivers = [
        {
            "label": "Temperature",
            "column": "temperature_2m",
            "unit": "°F",
            "color": "#3b82f6",
            "fillcolor": "rgba(59, 130, 246, 0.10)",
            "fmt": lambda v: f"{v:.0f}",
        },
        {
            "label": "Wind",
            "column": "wind_speed_80m",
            "unit": "mph",
            "color": "#34d399",
            "fillcolor": "rgba(52, 211, 153, 0.10)",
            "fmt": lambda v: f"{v:.1f}",
        },
        {
            "label": "Solar",
            "column": "shortwave_radiation",
            "unit": "W/m²",
            "color": "#f97316",
            "fillcolor": "rgba(249, 115, 22, 0.10)",
            "fmt": lambda v: f"{v:.0f}",
        },
    ]

    cells: list = []
    for d in drivers:
        col = d["column"]
        if col not in horizon.columns or horizon[col].isna().all():
            cells.append(_driver_cell_empty(d["label"]))
            continue
        latest = float(horizon[col].iloc[-1])
        avg = float(horizon[col].mean())
        delta = latest - avg
        delta_class = (
            "gp-metric-value--negative"
            if delta > 0.5
            else ("gp-metric-value--positive" if delta < -0.5 else "")
        )
        cells.append(
            html.Div(
                [
                    html.Div(d["label"], className="gp-metric-label"),
                    html.Div(
                        [
                            html.Span(
                                d["fmt"](latest),
                                className="gp-metric-value gp-metric-value--hero tabular",
                            ),
                            html.Span(d["unit"], className="gp-metric-unit"),
                        ],
                        className="gp-metric-value-row",
                    ),
                    html.Div(
                        [
                            html.Span(
                                f"{delta:+.1f} vs 24h avg",
                                className=f"gp-metric-sub {delta_class}",
                            ),
                        ],
                    ),
                    dcc.Graph(
                        figure=_driver_sparkline(horizon, col, d["color"], d["fillcolor"]),
                        config={"displayModeBar": False, "responsive": True},
                        style={"height": "60px"},
                    ),
                ],
                className="gp-driver-cell",
            )
        )
    return cells


def _drivers_empty() -> list:
    cells = []
    for label in ("Temperature", "Wind", "Solar"):
        cells.append(_driver_cell_empty(label))
    return cells


def _driver_cell_empty(label: str) -> html.Div:
    return html.Div(
        [
            html.Div(label, className="gp-metric-label"),
            html.Span("—", className="gp-metric-value tabular"),
            html.Div("No weather data", className="gp-metric-sub"),
        ],
        className="gp-driver-cell",
    )


# Fuel ordering: heaviest emissions at the bottom of the stack, zero-carbon
# on top. Within each bucket: dispatchable before intermittent.
_FUEL_STACK_ORDER: tuple[str, ...] = (
    "coal",
    "oil",
    "gas",
    "biomass",
    "other",
    "nuclear",
    "hydro",
    "wind",
    "solar",
)

_FUEL_DISPLAY: dict[str, dict[str, str]] = {
    "coal": {"label": "Coal", "color": "#71717a", "fill": "rgba(113, 113, 122, 0.85)"},
    "oil": {"label": "Oil", "color": "#52525b", "fill": "rgba(82, 82, 91, 0.85)"},
    "gas": {"label": "Gas", "color": "#f97316", "fill": "rgba(249, 115, 22, 0.85)"},
    "biomass": {"label": "Biomass", "color": "#a16207", "fill": "rgba(161, 98, 7, 0.85)"},
    "other": {"label": "Other", "color": "#a1a1aa", "fill": "rgba(161, 161, 170, 0.85)"},
    "nuclear": {"label": "Nuclear", "color": "#a855f7", "fill": "rgba(168, 85, 247, 0.85)"},
    "hydro": {"label": "Hydro", "color": "#3b82f6", "fill": "rgba(59, 130, 246, 0.85)"},
    "wind": {"label": "Wind", "color": "#34d399", "fill": "rgba(52, 211, 153, 0.85)"},
    "solar": {"label": "Solar", "color": "#fbbf24", "fill": "rgba(251, 191, 36, 0.85)"},
}


def _build_generation_panel(region: str | None, demand_json: str | None) -> html.Div:
    """Stacked-area fuel mix + 3-up sub-MetricsBar (Net Load / Renewable / Largest)."""
    region = region or "FPL"

    try:
        gen_df = _fetch_generation_cached(region)
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("forecast_generation_fetch_failed", region=region, error=str(exc))
        return _generation_empty()

    if gen_df is None or gen_df.empty:
        return _generation_empty()

    df = gen_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Window: latest 24 hours
    cutoff = df["timestamp"].max() - pd.Timedelta(hours=24)
    df = df[df["timestamp"] >= cutoff]
    if df.empty:
        return _generation_empty()

    pivot = (
        df.pivot_table(
            index="timestamp",
            columns="fuel_type",
            values="generation_mw",
            aggfunc="sum",
        )
        .fillna(0)
        .clip(lower=0)
    )

    # Sort columns by emissions order (any unknown fuels go to the end)
    ordered_fuels = [f for f in _FUEL_STACK_ORDER if f in pivot.columns]
    extras = [f for f in pivot.columns if f not in _FUEL_STACK_ORDER]
    pivot = pivot[ordered_fuels + extras]

    # ── KPIs ───────────────────────────────────────────────────
    total_per_ts = pivot.sum(axis=1)
    avg_total = float(total_per_ts.mean()) if not total_per_ts.empty else 0.0

    fuel_avg = pivot.mean(axis=0).sort_values(ascending=False)
    largest_fuel = fuel_avg.index[0] if len(fuel_avg) else None
    largest_label = _FUEL_DISPLAY.get(str(largest_fuel), {}).get(
        "label", str(largest_fuel).title() if largest_fuel else "—"
    )
    largest_share_pct = (
        float(fuel_avg.iloc[0] / fuel_avg.sum() * 100.0)
        if len(fuel_avg) and fuel_avg.sum() > 0
        else 0.0
    )

    renewable_cols = [c for c in ("wind", "solar", "hydro") if c in pivot.columns]
    if renewable_cols and avg_total > 0:
        renewable_pct = float((pivot[renewable_cols].sum(axis=1) / total_per_ts * 100.0).mean())
    else:
        renewable_pct = 0.0

    # Net load (Demand - Wind - Solar) if demand available
    # P2-23 (#273): net load is Demand − Wind − Solar. The old code silently
    # fell back to average TOTAL generation — a differently-defined quantity —
    # under the "Net Load (avg)" label whenever demand was missing, unparsable,
    # or misaligned (which is EVERY cold/warming page load). Render an honest
    # degraded cell instead; never substitute a different metric under this
    # label.
    net_load_avg: float | None = None
    if demand_json:
        try:
            ddf = pd.read_json(io.StringIO(demand_json))
            ddf["timestamp"] = pd.to_datetime(ddf["timestamp"])
            ddf = ddf.sort_values("timestamp")
            common = pivot.index.intersection(ddf.set_index("timestamp").index)
            if len(common) >= 2:
                d_aligned = ddf.set_index("timestamp").loc[common, "demand_mw"]
                wind_aligned = pivot.loc[common].get("wind", pd.Series(0.0, index=common))
                solar_aligned = pivot.loc[common].get("solar", pd.Series(0.0, index=common))
                net_load_series = d_aligned - wind_aligned - solar_aligned
                candidate = float(net_load_series.mean())
                # All-NaN demand over the aligned window yields NaN — that
                # must degrade honestly, not render "nan MW".
                if np.isfinite(candidate):
                    net_load_avg = candidate
        except Exception as exc:  # pragma: no cover
            log.warning("forecast_generation_netload_failed", region=region, error=str(exc))
    if net_load_avg is None:
        log.info("forecast_generation_netload_unavailable", region=region)
        net_load_item = {
            "label": "Net Load (avg)",
            "value": "—",
            "unit": "MW",
            "hero": True,
            "tone": "secondary",
            "subtext": "demand data unavailable",
        }
    else:
        net_load_item = {
            "label": "Net Load (avg)",
            "value": f"{net_load_avg:,.0f}",
            "unit": "MW",
            "hero": True,
        }

    sub_metrics = build_metrics_bar(
        [
            net_load_item,
            {
                "label": "Renewable Share",
                "value": f"{renewable_pct:.1f}%",
                "tone": "positive" if renewable_pct >= 25 else "secondary",
            },
            {
                "label": "Largest Source",
                "value": largest_label,
                "unit": f"{largest_share_pct:.0f}%",
                "tone": "secondary",
            },
        ]
    )
    # Override the default 5-up class for a 3-up grid.
    sub_metrics.className = "gp-metrics-bar gp-metrics-bar--3up"

    # ── Stacked-area chart ─────────────────────────────────────
    fig = go.Figure()
    for fuel in pivot.columns:
        cfg = _FUEL_DISPLAY.get(
            str(fuel),
            {
                "label": str(fuel).title(),
                "color": "#a1a1aa",
                "fill": "rgba(161,161,170,0.7)",
            },
        )
        fig.add_trace(
            go.Scatter(
                x=pivot.index,
                y=pivot[fuel],
                mode="lines",
                stackgroup="gen",
                name=cfg["label"],
                line=dict(width=0, color=cfg["color"]),
                fillcolor=cfg["fill"],
                hovertemplate=(
                    f"<b>{cfg['label']}</b><br>%{{x|%H:%M}}<br>%{{y:,.0f}} MW<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        **_layout(
            uirevision=f"gen-{region}",
            showlegend=True,
            xaxis=dict(
                showgrid=False,
                linecolor="rgba(255,255,255,0.04)",
                tickfont=dict(color="#71717a", size=10),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.04)",
                zeroline=False,
                tickformat=",.0f",
                tickfont=dict(color="#71717a", size=10),
                title=None,
            ),
            margin=dict(l=48, r=16, t=16, b=64),
        ),
    )

    return html.Div(
        [
            sub_metrics,
            dcc.Graph(
                figure=fig,
                config={"displayModeBar": False, "responsive": True},
                style={"height": "320px"},
            ),
        ],
        className="gp-generation-stack",
    )


def _generation_empty() -> html.Div:
    return html.Div(
        "No generation data available for this region.",
        className="gp-panel__placeholder",
    )


def _scenario_demand_factor(temp_delta: float, wind_delta: float, solar_delta: float) -> float:
    """Linear demand-sensitivity factor for the scenario simulator heuristic.

    Returns a multiplicative factor to apply to a baseline 24h forecast.
    Coefficients are order-of-magnitude-defensible against load-research
    norms (not physically rigorous — full-fidelity physics lives in
    ``simulation/scenario_engine.py``):

      * temp_delta: ±2.5 % per 5 °F (existing — dominant driver)
      * solar_delta: +1.5 % per 100 W/m² (sun load → AC demand;
        meaningful for summer-peaking BAs like FPL/ERCOT/PJM)
      * wind_delta: +0.5 % per 10 mph (wind chill → heating demand;
        meaningful for winter-peaking BAs)

    All three combine linearly. Pulled out as a pure function so the
    heuristic is unit-testable without spinning up the Plotly render.
    """
    return 1.0 + (temp_delta / 5.0) * 0.025 + solar_delta * 0.00015 + wind_delta * 0.0005


def _scenario_factors(
    region: str | None,
    temp_delta: float,
    wind_delta: float,
    solar_delta: float,
    n_hours: int,
) -> tuple[np.ndarray, str]:
    """Hourly demand factors for the simulator, physics first (#127).

    Reads the precomputed grid the scoring job wrote and interpolates it to
    the slider position. Falls back to ``_scenario_demand_factor`` — the
    analytical heuristic that shipped in #119 — whenever the grid is absent,
    stale-shaped, or unreadable, which covers a cold Redis, a region the
    scoring job shed at its soft deadline, and the flag being off.

    Reading Redis rather than re-running the models is what keeps this
    inside the web-tier I/O guardrail: no model touches the request path,
    and the slider stays as responsive as it was with the heuristic.

    Returns:
        ``(factors, source)`` where ``factors`` is length ``n_hours`` and
        ``source`` is ``"grid"`` or ``"heuristic"`` for the UI to label.
    """
    from config import feature_enabled

    flat = _scenario_demand_factor(temp_delta, wind_delta, solar_delta)
    fallback = np.full(n_hours, flat, dtype=float)

    if not region or not feature_enabled("scenario_grid"):
        return fallback, "heuristic"

    try:
        from data.redis_client import redis_get, redis_key
        from simulation.scenario_grid import interpolate_scenario_factors

        payload = redis_get(redis_key(f"scenario_grid:{region}"))
        if not payload:
            return fallback, "heuristic"

        curve = interpolate_scenario_factors(payload, temp_delta, wind_delta, solar_delta)
        if curve is None or curve.size == 0:
            return fallback, "heuristic"

        # The grid is 24h and the chart may be shorter or longer; hold the
        # last factor rather than letting the curve run out mid-chart.
        if curve.size < n_hours:
            curve = np.concatenate([curve, np.full(n_hours - curve.size, curve[-1])])

        # Outside the booster's observed range the response saturates and then
        # wanders (measured 2026-08-11). Label it rather than presenting an
        # extrapolation as a re-forecast.
        env = payload.get("envelope") or {}
        axes = payload.get("axes") or {}

        def _in(axis: str, value: float) -> bool:
            flags, positions = env.get(axis), axes.get(axis)
            if not flags or not positions or len(flags) != len(positions):
                return True
            i = min(range(len(positions)), key=lambda j: abs(positions[j] - value))
            return bool(flags[i])

        if not all(
            (
                _in("temp_f", temp_delta),
                _in("wind_mph", wind_delta),
                _in("solar_wm2", solar_delta),
            )
        ):
            return curve[:n_hours], "grid_extrapolated"
        return curve[:n_hours], "grid"
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("scenario_grid_read_failed", region=region, error=str(exc))
        return fallback, "heuristic"


def _build_scenarios_panel(
    temp_delta: int | float | None,
    wind_delta: int | float | None,
    solar_delta: int | float | None,
    region: str | None,
    demand_json: str | None,
) -> tuple[html.Div, go.Figure]:
    """Heuristic scenario impact + baseline-vs-scenario comparison chart.

    Returns ``(kpi_bar, figure)``. The math is a deliberate simplification:
    no model re-run, just a linear demand-sensitivity factor against the
    current 24h forecast. Real ensemble simulation lives in the (now hidden)
    Scenarios tab and the simulation/scenario_engine module — exposing
    full-fidelity here would need model loading on every slider drag.

    Demand sensitivities — see ``_scenario_demand_factor`` for coefficients.

    Renewable-share sensitivities (independent of demand):
      * wind_delta: ±0.6 pp per mph (caps at 30 pp)
      * solar_delta: ±0.05 pp per W/m² (caps at 30 pp)

    Confidence sensitivity: −1 pp per 5 °F of |temp_delta|, capped at −10 pp
    (forecast residuals grow with abs(temp_delta) outside ±10 °F).
    """
    region = region or "FPL"
    temp_delta = float(temp_delta or 0)
    wind_delta = float(wind_delta or 0)
    solar_delta = float(solar_delta or 0)

    # Base forecast (next 24h ensemble)
    horizon = 24
    base_y: np.ndarray | None = None
    last_actual_ts: pd.Timestamp | None = None

    if demand_json:
        try:
            demand_df = pd.read_json(io.StringIO(demand_json))
            demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])
            demand_df = demand_df.sort_values("timestamp")

            # Baseline = the real scored ensemble from Redis (the scoring job's
            # own output), not model_service.get_forecasts — which on the
            # stateless web tier is strict-gated to "unavailable" in prod
            # (#149) and echoed actuals as a fake forecast in dev when only
            # "ensemble" is requested (2026-07 review P2-31). This is the same
            # reader the Overview hero uses.
            forecast_payload = _read_ensemble_forecast_from_redis(region)
            if forecast_payload is not None:
                _fc_ts, ensemble_arr, _scored_at = forecast_payload
                if ensemble_arr is not None and len(ensemble_arr) >= horizon:
                    base_y = np.asarray(ensemble_arr[:horizon], dtype=float)
                    last_actual_ts = demand_df["timestamp"].iloc[-1]
        except Exception as exc:  # pragma: no cover
            log.warning("forecast_scenario_baseline_failed", region=region, error=str(exc))

    if base_y is None or last_actual_ts is None:
        kpi_empty = build_metrics_bar(
            [
                {"label": "Δ Peak", "value": "—", "tone": "secondary", "hero": True},
                {"label": "Δ Headroom", "value": "—", "tone": "secondary"},
                {"label": "Δ Renewable", "value": "—", "tone": "secondary"},
                {"label": "Δ Confidence", "value": "—", "tone": "secondary"},
            ]
        )
        kpi_empty.className = "gp-metrics-bar gp-metrics-bar--4up"
        return (kpi_empty, _empty_figure("Awaiting baseline forecast"))

    # ── Scenario forecast ──────────────────────────────────────
    # #127: real per-hour physics from the precomputed grid when the scoring
    # job has written one, the #119 linear heuristic when it has not. The
    # grid's factor varies BY HOUR — the heuristic's single scalar could not
    # express that a +15 °F afternoon and a +15 °F 4am differ, which is most
    # of what makes a demand response a curve rather than a scaling.
    scenario_factors, scenario_source = _scenario_factors(
        region, temp_delta, wind_delta, solar_delta, len(base_y)
    )
    scenario_y = base_y * scenario_factors

    base_peak = float(np.max(base_y))
    scenario_peak = float(np.max(scenario_y))
    delta_peak_mw = scenario_peak - base_peak
    delta_peak_pct = (delta_peak_mw / base_peak * 100.0) if base_peak > 0 else 0.0

    from models.pricing import capacity_headroom_pct

    capacity = REGION_CAPACITY_MW.get(region, 100_000)
    base_headroom = capacity_headroom_pct(base_peak, capacity)
    scenario_headroom = capacity_headroom_pct(scenario_peak, capacity)
    delta_headroom_pp = scenario_headroom - base_headroom

    # Renewable share heuristic — wind: 0.6 %/mph; solar: 0.05 %/(W/m²)
    delta_renewable_pp = wind_delta * 0.6 + solar_delta * 0.05
    delta_renewable_pp = max(min(delta_renewable_pp, 30.0), -30.0)

    # Confidence delta: bigger temp swings widen the band roughly linearly
    # (forecast residuals grow with abs(temp_delta) outside ±10°F band).
    delta_confidence_pp = -min(abs(temp_delta) / 5.0, 10.0)  # negative pp

    # ── KPI bar ────────────────────────────────────────────────
    peak_tone = (
        "negative"
        if delta_peak_pct > 0.5
        else ("positive" if delta_peak_pct < -0.5 else "secondary")
    )
    headroom_tone = (
        "positive"
        if delta_headroom_pp > 0.1
        else ("negative" if delta_headroom_pp < -0.1 else "secondary")
    )
    renewable_tone = (
        "positive"
        if delta_renewable_pp > 0.5
        else ("negative" if delta_renewable_pp < -0.5 else "secondary")
    )
    kpis = build_metrics_bar(
        [
            {
                "label": "Δ Peak",
                "value": f"{delta_peak_mw:+,.0f}",
                "unit": f"MW ({delta_peak_pct:+.1f}%)",
                "tone": peak_tone,
                "hero": True,
            },
            {
                "label": "Δ Headroom",
                "value": f"{delta_headroom_pp:+.1f}",
                "unit": "pp",
                "tone": headroom_tone,
                "help": "Change in capacity headroom (nameplate) at peak under this scenario — not a NERC reserve margin (#243).",
            },
            {
                "label": "Δ Renewable",
                "value": f"{delta_renewable_pp:+.1f}",
                "unit": "pp",
                "tone": renewable_tone,
            },
            {
                "label": "Δ Confidence",
                "value": f"{delta_confidence_pp:+.1f}",
                "unit": "pp",
                "tone": "secondary",
            },
        ]
    )
    kpis.className = "gp-metrics-bar gp-metrics-bar--4up"

    # Say which engine produced these numbers. The static panel copy asserted
    # "not a model re-forecast" for a fortnight while production served
    # exactly that (#127) — static text cannot track a feature flag, so the
    # claim belongs with the results it describes.
    source_note = html.P(
        (
            "Real ensemble re-forecast — 81 precomputed weather scenarios, "
            "interpolated to these slider positions."
            if scenario_source == "grid"
            else "Beyond this region's observed weather — the model is "
            "extrapolating and its response flattens here. Directional only."
            if scenario_source == "grid_extrapolated"
            else "Illustrative linear weather-sensitivity — not a model "
            "re-forecast. Directional stress-testing only, not calibrated "
            "predictions."
        ),
        className="gp-panel__disclosure",
        **{"data-scenario-source": scenario_source},
    )
    kpis = html.Div([kpis, source_note])

    # ── Baseline vs scenario chart ─────────────────────────────
    forecast_ts = pd.date_range(
        start=last_actual_ts + pd.Timedelta(hours=1),
        periods=horizon,
        freq="h",
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast_ts,
            y=base_y,
            mode="lines",
            name="Baseline",
            line=dict(color="#3b82f6", width=1.75),
            hovertemplate="<b>Baseline</b><br>%{x|%H:%M}<br>%{y:,.0f} MW<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_ts,
            y=scenario_y,
            mode="lines",
            name="Scenario",
            line=dict(color="#f97316", width=1.75, dash="dash"),
            hovertemplate="<b>Scenario</b><br>%{x|%H:%M}<br>%{y:,.0f} MW<extra></extra>",
        )
    )
    fig.update_layout(
        **_layout(
            uirevision=f"scn-{region}",
            xaxis=dict(
                showgrid=False,
                linecolor="rgba(255,255,255,0.04)",
                tickfont=dict(color="#71717a", size=10),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.04)",
                zeroline=False,
                tickformat=",.0f",
                tickfont=dict(color="#71717a", size=10),
                title=None,
            ),
            margin=dict(l=48, r=16, t=16, b=36),
            showlegend=True,
        ),
    )
    return kpis, fig


def _driver_sparkline(df: pd.DataFrame, column: str, color: str, fillcolor: str) -> go.Figure:
    """60px sparkline matching the v2 minimal-axes treatment."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df[column],
            mode="lines",
            line=dict(color=color, width=1.5),
            fill="tozeroy",
            fillcolor=fillcolor,
            hovertemplate="%{x|%H:%M}<br>%{y:,.1f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        **_layout(
            uirevision=column,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=0, r=0, t=4, b=4),
        )
    )
    return fig


# ── Overview briefing block (Step 7c — sparklines / briefing / digest / spotlights / persona) ──

__all__ = [
    "_confidence_half_width",
    "_add_confidence_bands",
    "_add_trailing_actuals",
    "_run_forecast_outlook",
    "_create_future_features",
    "_outlook_tab_from_redis",
    "register_forecast_callbacks",
]
