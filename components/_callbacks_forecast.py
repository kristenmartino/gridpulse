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
from dash import ALL, Input, Output, State, ctx, html, no_update

from components._callbacks_shared import (
    _CACHE_VERSION,
    _MODEL_BAND_COLORS,
    _MODEL_CACHE,
    _PREDICTION_CACHE,
    COLORS,
    _compute_data_hash,
    _empirical_interval_from_backtests,
    _layout,
    _widening_interval_from_backtests,
)
from components.accessibility import LINE_STYLES
from config import CACHE_TTL_SECONDS, OPEN_METEO_FORECAST_HOURS, REQUIRE_REDIS
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
    Beyond that, ``jobs/phases._build_future_feature_frame`` falls back
    to per-(hour, dow) climatological group means for the future weather
    features. The model still produces a forecast there, but its weather
    inputs are seasonal-average shaped, not actual forward-looking values.

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
        # predictions, so a Prophet/ARIMA/ensemble band is typically
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
    # a warming state rather than training inline. The Dash UI treats this
    # like any other degraded state and renders a skeleton.
    if REQUIRE_REDIS:
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
            # ARIMA is excluded beyond 168h — SARIMAX compounds errors at long
            # horizons and actively degrades ensemble quality.
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
    # Redis only contains XGBoost predictions — never serve them as "ensemble"
    # (which should be a weighted combination of multiple models).
    if model_name != "xgboost" and model_name not in forecasts[0]:
        log.info("outlook_redis_model_miss", model=model_name)
        return None

    timestamps = pd.to_datetime([f["timestamp"] for f in forecasts])
    pred_key = model_name if model_name in forecasts[0] else "predicted_demand_mw"
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
    model_style = LINE_STYLES.get(
        model_name, {"color": COLORS["ensemble"], "width": 2, "dash": "solid"}
    )
    fig.add_trace(
        go.Scatter(
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
        fig, timestamps, predictions, horizon_hours, region=region, model_name=model_name
    )
    _add_trailing_actuals(fig, demand_json)
    # Mark the Open-Meteo / climatology boundary on long-horizon views.
    # Only the 30-day view actually crosses the day-16 boundary; on 24h
    # and 7-day views the helper is a no-op. See ADR-008.
    has_climatology_segment = _add_forecast_horizon_divider(fig, timestamps, horizon_hours)
    horizon_labels = {24: "24-Hour", 168: "7-Day", 720: "30-Day"}
    interval_caption = _interval_caption(interval_meta, model_name)
    # On the 30-day view, surface in the subtitle that days 17-30 are
    # climatology baseline rather than real forecast. Users browsing
    # the chart shouldn't have to hover the divider line to understand
    # the regime split. See ADR-008.
    horizon_caption = ""
    if has_climatology_segment:
        horizon_caption = (
            "<br><sup>Days 1-16: real Open-Meteo forecast · "
            "Days 17-30: (hour-of-day, day-of-week) climatology baseline</sup>"
        )
    fig.update_layout(
        **_layout(
            uirevision=f"{region}:{horizon_hours}",
            title=(
                f"{horizon_labels.get(horizon_hours, '')} {model_name.upper()} Demand Forecast — {region}"
                f"{interval_caption}{horizon_caption}"
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
        model_name=model_name,
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

    The callbacks reach into ``_callbacks_overview`` for three panel
    builders that originally lived in callbacks.py (drivers /
    generation / scenarios) — those have always belonged to the
    Forecast tab in spirit but the Overview module retained the
    helpers during Step 7b/7c. Importing them lazily here keeps the
    dependency graph one-way (Forecast → Overview) and avoids a
    circular import at module load.
    """
    from components._callbacks_overview import (
        _build_drivers_panel,
        _build_generation_panel,
        _build_scenarios_panel,
    )
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
        ],
        State("dashboard-tabs", "active_tab"),
    )
    def update_forecast_scenarios_panel(
        is_open, temp_d, wind_d, solar_d, region, demand_json, active_tab
    ):
        """Render the 4-up delta MetricsBar + baseline-vs-scenario chart.

        Lazy: only fires when the collapse is open and the user is on the
        Forecast tab. Uses a heuristic impact model (no full ensemble re-run)
        so the slider feels instant; the existing Scenarios tab still hosts
        the trained-model simulation for full-fidelity what-ifs.
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
        """Render the horizontal MAPE/RMSE/MAE/R² bar for the active model."""
        if active_tab != "tab-outlook":
            return no_update

        try:
            from models.model_service import get_model_metrics, is_trained
        except ImportError:
            return html.Div()

        region = region or "FPL"
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
        name = "XGBoost" if model_name == "xgboost" else model_name.title()
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
            # not a hard failure. Keep the loud message for genuine errors.
            is_warming = result["error"] == "warming"
            text = (
                "Pipeline is warming up — forecast will appear shortly"
                if is_warming
                else f"Forecast failed: {result['error']}"
            )
            color = "#71717a" if is_warming else "#f87171"  # tertiary | danger
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


__all__ = [
    "_confidence_half_width",
    "_add_confidence_bands",
    "_add_trailing_actuals",
    "_run_forecast_outlook",
    "_create_future_features",
    "_outlook_tab_from_redis",
    "register_forecast_callbacks",
]
