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
  scoring job's hourly ``wattcast:forecast:{region}:1h`` payload.

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

from components._callbacks_shared import (
    _CACHE_VERSION,
    _MODEL_BAND_COLORS,
    _MODEL_CACHE,
    _PREDICTION_CACHE,
    COLORS,
    _compute_data_hash,
    _empirical_interval_from_backtests,
    _empty_figure,  # noqa: F401 — used implicitly via outlook tab pipeline
    _layout,
)
from components.accessibility import LINE_STYLES
from config import CACHE_TTL_SECONDS, REQUIRE_REDIS
from data.redis_client import redis_get

log = structlog.get_logger()


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

    band_name = (
        "80% empirical prediction interval"
        if interval_meta["method"] == "empirical"
        else "80% indicative range"
    )

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
    # value repeated for every future hour.
    hist = train_df.copy()
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
    cached = redis_get(f"wattcast:forecast:{region}:{granularity}")
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
    horizon_labels = {24: "24-Hour", 168: "7-Day", 720: "30-Day"}
    interval_caption = ""
    if interval_meta.get("method") == "empirical":
        interval_caption = (
            f"<br><sup>80% empirical prediction interval "
            f"(calibration window: last {int(interval_meta.get('calibration_window_hours', 0))}h)</sup>"
        )
    fig.update_layout(
        **_layout(
            uirevision=f"{region}:{horizon_hours}",
            title=(
                f"{horizon_labels.get(horizon_hours, '')} {model_name.upper()} Demand Forecast — {region}"
                f"{interval_caption}"
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


__all__ = [
    "_confidence_half_width",
    "_add_confidence_bands",
    "_add_trailing_actuals",
    "_run_forecast_outlook",
    "_create_future_features",
    "_outlook_tab_from_redis",
]
