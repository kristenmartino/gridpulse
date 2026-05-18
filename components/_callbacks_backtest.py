"""Backtest tab helpers extracted from ``components/callbacks.py``.

Step 9 of the ``callbacks.py`` decomposition tracked in issue #87.
Final per-tab extraction. Continues the per-tab split established by:

* #98 — shared infrastructure (``_callbacks_shared.py``)
* #99 — US Grid tab (``_callbacks_us_grid.py``)
* #100 — Models tab (``_callbacks_models.py``)
* #101 — Alerts tab (``_callbacks_alerts.py``)
* #102 — Generation tab (``_callbacks_generation.py``)
* #103 — Weather tab (``_callbacks_weather.py``)
* #104 / #105 / #106 — Overview tab (``_callbacks_overview.py``)
* #107 — Forecast tab (``_callbacks_forecast.py``)

After this lands, callbacks.py contains only ``register_callbacks``
plus ``_load_data_from_redis``. The final step splits
``register_callbacks`` into per-tab ``register_X_callbacks(app)``
functions.

## What lives here

Seven helpers covering the Backtest surface:

* ``_normalize_backtest_exog_mode`` — coerce optional UI input into
  one of {``oracle_exog``, ``forecast_exog``}.
* ``_describe_exog_mode`` — human-readable label for chart titles.
* ``_build_forecast_exog_fold`` — production-like exogenous weather
  for a backtest fold, with Redis snapshot + climatology fallback.
* ``_backtest_tab_from_redis`` — Redis fast path that builds the
  entire Backtest tab from the scoring job's hourly
  ``gridpulse:backtest:{exog_mode}:{region}:{horizon_hours}`` payload.
* ``_predict_single_fold`` — train+predict for one fold of one model.
* ``_ensemble_fold`` — equal-weight ensemble across all models for
  one fold. Uses uniform averaging (not 1/MAPE) to avoid data leakage
  — see the docstring for the data-leakage argument.
* ``_run_backtest_for_horizon`` — walk-forward expanding-window
  cross-validation orchestrator with 3-tier cache (in-memory →
  SQLite → train fresh) and warming gate for production.

## Why these stay together (vs splitting fast-path from compute path)

``_backtest_tab_from_redis`` and ``_run_backtest_for_horizon`` are the
two entry points the tab uses. The remaining five helpers are private
to one or the other. Splitting fast-path from compute-path would
leave the leakage-control rationale (``_ensemble_fold``'s uniform-
averaging docstring) and the exog-mode normalization wiring (``payload_mode
= _normalize_backtest_exog_mode(...)``) straddling two modules. Keeping
them together preserves the cohesion.

## Public-import surface

``components/callbacks.py`` re-imports each user-facing function by
name. Tests import via ``from components.callbacks import
_run_backtest_for_horizon`` etc — the re-export shim keeps those import
sites valid without any caller-side changes.

When patching for tests, target the function's *new* namespace:

    @patch("components._callbacks_backtest.redis_get")  # ✓
    @patch("components.callbacks.redis_get")            # ✗ (no effect)

The ``REQUIRE_REDIS`` symbol referenced by the warming gate inside
``_run_backtest_for_horizon`` is imported from ``config`` at module
load — integration tests that ``monkeypatch.setattr(cbs, "REQUIRE_REDIS",
True)`` should also patch ``components._callbacks_backtest.REQUIRE_REDIS``
to ensure the gate trips (same pattern as the forecast extraction).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import structlog

from components._callbacks_shared import (
    _BACKTEST_CACHE,
    _CACHE_VERSION,
    BACKTEST_EXOG_MODES,
    COLORS,
    DEFAULT_BACKTEST_EXOG_MODE,
    _compute_data_hash,
    _layout,
)
from config import CACHE_TTL_SECONDS, REQUIRE_REDIS, WEATHER_VARIABLES
from data.redis_client import redis_get, redis_key

log = structlog.get_logger()


def _normalize_backtest_exog_mode(exog_mode: str | None) -> str:
    """Normalize requested backtest exogenous mode."""
    mode = (exog_mode or DEFAULT_BACKTEST_EXOG_MODE).strip().lower()
    if mode not in BACKTEST_EXOG_MODES:
        return DEFAULT_BACKTEST_EXOG_MODE
    return mode


def _describe_exog_mode(exog_mode: str | None, exog_source: str | None = None) -> str:
    """Human-readable exogenous mode/source label for UI copy."""
    mode = _normalize_backtest_exog_mode(exog_mode)
    if mode == "oracle_exog":
        return "Oracle exogenous weather (actual future weather)"
    if exog_source:
        return f"Forecast exogenous weather ({exog_source})"
    return "Forecast exogenous weather (production-like baseline)"


def _build_forecast_exog_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    region: str,
    horizon_hours: int,
) -> tuple[pd.DataFrame, str]:
    """Build production-like exogenous weather for a fold.

    Priority:
    1) Archived forecast snapshots from Redis (if available and aligned)
    2) Hour-of-week climatology baseline from training data
    """
    weather_cols = [c for c in WEATHER_VARIABLES if c in test_df.columns and c in train_df.columns]
    if not weather_cols:
        return test_df.copy(), "no-weather-columns"

    out = test_df.copy()
    test_ts = pd.to_datetime(out["timestamp"])
    out["timestamp"] = test_ts

    snapshot_keys = [
        redis_key(f"weather-forecast-snapshot:{region}:{horizon_hours}"),
        redis_key(f"weather-forecast:{region}:{horizon_hours}"),
        redis_key(f"weather-forecast-snapshot:{region}"),
    ]

    for key in snapshot_keys:
        cached = redis_get(key)
        if not isinstance(cached, dict):
            continue
        rows = cached.get("forecasts")
        if not isinstance(rows, list) or not rows:
            continue
        snap_df = pd.DataFrame(rows)
        if "timestamp" not in snap_df.columns:
            continue
        keep_cols = ["timestamp"] + [c for c in weather_cols if c in snap_df.columns]
        if len(keep_cols) <= 1:
            continue
        snap_df = snap_df[keep_cols].copy()
        snap_df["timestamp"] = pd.to_datetime(snap_df["timestamp"], errors="coerce")
        snap_df = snap_df.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"])
        merged = out[["timestamp"]].merge(snap_df, on="timestamp", how="left")
        coverage = float(merged[keep_cols[1:]].notna().all(axis=1).mean()) if len(merged) else 0.0
        if coverage >= 0.8:
            for col in keep_cols[1:]:
                out[col] = merged[col].ffill().bfill()
            return out, "archived forecast snapshot"

    # Fallback: climatology / naive hour-of-week baseline from train period
    train = train_df.copy()
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    train["dow"] = train["timestamp"].dt.dayofweek
    train["hour"] = train["timestamp"].dt.hour
    out["dow"] = out["timestamp"].dt.dayofweek
    out["hour"] = out["timestamp"].dt.hour

    for col in weather_cols:
        by_dow_hour = train.groupby(["dow", "hour"])[col].mean()
        by_hour = train.groupby("hour")[col].mean()
        global_mean = float(train[col].mean()) if train[col].notna().any() else 0.0
        values = []
        for d, h in zip(out["dow"], out["hour"], strict=False):
            if (d, h) in by_dow_hour.index and pd.notna(by_dow_hour.loc[(d, h)]):
                values.append(float(by_dow_hour.loc[(d, h)]))
            elif h in by_hour.index and pd.notna(by_hour.loc[h]):
                values.append(float(by_hour.loc[h]))
            else:
                values.append(global_mean)
        out[col] = values

    return out.drop(columns=["dow", "hour"], errors="ignore"), "climatology/naive baseline"


def _backtest_tab_from_redis(region, horizon_hours, model_name, persona_id):
    """Redis fast path for backtest tab.

    Returns a 7-tuple (fig, mape_str, rmse_str, mae_str, r2_str,
    explanation, insight_card) or None if cache miss.
    """
    exog_mode = DEFAULT_BACKTEST_EXOG_MODE
    cached = redis_get(redis_key(f"backtest:{exog_mode}:{region}:{horizon_hours}"))
    if cached is None:
        cached = redis_get(redis_key(f"backtest:{region}:{horizon_hours}"))
    if cached is None:
        return None

    log.info("backtest_redis_hit", region=region, horizon=horizon_hours)

    # Model availability check: skip Redis if requested model isn't stored
    all_predictions = cached.get("predictions", {})
    if model_name not in all_predictions and model_name not in ("ensemble",):
        log.info("backtest_redis_model_miss", model=model_name)
        return None

    timestamps = pd.to_datetime(cached.get("timestamps", []))
    actual = np.array(cached.get("actual", []))

    # Get predictions for the requested model, fall back to ensemble
    all_predictions = cached.get("predictions", {})
    if model_name in all_predictions:
        predictions = np.array(all_predictions[model_name])
    elif "ensemble" in all_predictions:
        predictions = np.array(all_predictions["ensemble"])
    elif all_predictions:
        predictions = np.array(next(iter(all_predictions.values())))
    else:
        predictions = actual  # shouldn't happen

    # Get metrics for the requested model
    all_metrics = cached.get("metrics", {})
    if model_name in all_metrics:
        metrics = all_metrics[model_name]
    elif "ensemble" in all_metrics:
        metrics = all_metrics["ensemble"]
    elif all_metrics:
        metrics = next(iter(all_metrics.values()))
    else:
        metrics = {"mape": 0, "rmse": 0, "mae": 0, "r2": 0}
    residuals = actual - predictions
    interval_monitor = {"recent_coverage": 0.0, "drift": -0.8}
    interval_window = 0
    interval_available = False
    try:
        from models.evaluation import (
            apply_empirical_interval,
            compute_interval_coverage_drift,
            empirical_error_quantiles,
        )

        interval_window = int(min(len(residuals), max(horizon_hours * 5, 120)))
        recent_resid = residuals[-interval_window:] if interval_window else residuals
        q = empirical_error_quantiles(recent_resid, lower_q=0.10, upper_q=0.90)
        lower_band, upper_band = apply_empirical_interval(
            predictions, q["lower_error"], q["upper_error"]
        )
        interval_monitor = compute_interval_coverage_drift(actual, lower_band, upper_band, 0.80)
        interval_available = bool(q.get("sample_size", 0) > 0)
    except Exception:
        lower_band, upper_band = predictions, predictions

    # Build the chart
    fig = go.Figure()
    model_colors = {
        "xgboost": COLORS.get("ensemble", "#2DE2C4"),
        "prophet": COLORS.get("prophet", "#E69F00"),
        "arima": COLORS.get("arima", "#009E73"),
        "ensemble": COLORS.get("ensemble", "#2DE2C4"),
    }
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=actual,
            mode="lines",
            name="Actual Demand",
            line=dict(color=COLORS["actual"], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=predictions,
            mode="lines",
            name=f"{model_name.upper()} Forecast",
            line=dict(color=model_colors.get(model_name, "#2DE2C4"), width=2, dash="dash"),
        )
    )
    if interval_available:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=upper_band,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=lower_band,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=COLORS["confidence"],
                name="80% empirical prediction interval",
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=list(timestamps) + list(timestamps[::-1]),
            y=list(predictions) + list(actual[::-1]),
            fill="toself",
            fillcolor="rgba(255,92,122,0.12)",
            line=dict(width=0),
            name="Forecast Error",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    horizon_labels = {24: "24-Hour", 168: "7-Day", 720: "30-Day"}
    payload_mode = _normalize_backtest_exog_mode(cached.get("exog_mode", exog_mode))
    exog_caption = _describe_exog_mode(payload_mode, cached.get("exog_source"))
    explanations = {
        24: "24-hour ahead: Forecast made 1 day before. Best for day-ahead scheduling.",
        168: "7-day ahead: Forecast made 1 week before. Tests medium-term accuracy.",
        720: "30-day ahead: Forecast made 1 month before. Tests long-term planning reliability.",
    }
    fig.update_layout(
        **_layout(
            uirevision=f"{region}:{horizon_hours}:{model_name}",
            title=(
                f"{horizon_labels.get(horizon_hours, '')} Pre-computed Backtest: "
                f"{model_name.upper()} vs Actual — {region}<br><sup>{exog_caption}</sup>"
            ),
            xaxis_title="Date/Time",
            yaxis_title="Demand (MW)",
            hovermode="x unified",
        )
    )

    mode_suffix = f" ({payload_mode})"
    mape_str = f"{metrics.get('mape', 0):.2f}%{mode_suffix}"
    rmse_str = f"{metrics.get('rmse', 0):,.0f} MW{mode_suffix}"
    mae_str = f"{metrics.get('mae', 0):,.0f} MW{mode_suffix}"
    r2_str = f"{metrics.get('r2', 0):.3f}{mode_suffix}"
    coverage_str = f"{interval_monitor.get('recent_coverage', 0.0) * 100:.1f}%"
    drift_pp = interval_monitor.get("drift", 0.0) * 100.0

    from components.insights import build_insight_card, generate_tab3_insights

    persona = persona_id or "grid_ops"
    tab3_insights = generate_tab3_insights(
        persona,
        region or "FPL",
        {model_name: metrics},
        model_name=model_name,
        horizon_hours=horizon_hours,
        actual=actual,
        predictions=predictions,
        timestamps=timestamps,
        num_folds=0,
    )
    insight_card = build_insight_card(tab3_insights, persona, "tab-backtest")

    log.info("backtest_redis_complete", mape=mape_str, region=region)
    return (
        fig,
        mape_str,
        rmse_str,
        mae_str,
        r2_str,
        (
            f"{explanations.get(horizon_hours, '')} Exogenous mode: {exog_caption}. "
            f"Interval: 80% empirical prediction interval (calibration window: last {interval_window}h). "
            f"Recent coverage: {coverage_str} (drift vs 80% target: {drift_pp:+.1f} pp)."
        ),
        insight_card,
    )


def _predict_single_fold(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    exog_mode: str = DEFAULT_BACKTEST_EXOG_MODE,
) -> np.ndarray | None:
    """Train a model on train_df and predict on test_df for one backtest fold.

    Returns predictions array or None on failure.
    """
    _ = _normalize_backtest_exog_mode(exog_mode)
    n_test = len(test_df)

    if model_name == "xgboost":
        from data.feature_engineering import compute_autoregressive_snapshot
        from models.xgboost_model import predict_xgboost, train_xgboost

        model = train_xgboost(train_df)
        demand_history = train_df["demand_mw"].tolist()
        preds: list[float] = []
        for i in range(n_test):
            row = test_df.iloc[[i]].copy()
            for col, val in compute_autoregressive_snapshot(demand_history).items():
                row[col] = val
            row = row.ffill().bfill().fillna(0)
            step_pred = float(predict_xgboost(model, row)[0])
            preds.append(step_pred)
            demand_history.append(step_pred)
        return np.array(preds, dtype=float)

    elif model_name == "prophet":
        from models.prophet_model import predict_prophet, train_prophet

        model = train_prophet(train_df)
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        result = predict_prophet(model, full_df, periods=n_test)
        return result["forecast"][:n_test]

    elif model_name == "arima":
        from models.arima_model import predict_arima, train_arima

        test_clean = test_df.copy()
        for col in [
            "temperature_2m",
            "wind_speed_80m",
            "shortwave_radiation",
            "cooling_degree_days",
            "heating_degree_days",
        ]:
            if col in test_clean.columns:
                test_clean[col] = test_clean[col].ffill().bfill().fillna(0)
        model = train_arima(train_df)
        return predict_arima(model, test_clean, periods=n_test)[:n_test]

    return None


def _ensemble_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    exog_mode: str = DEFAULT_BACKTEST_EXOG_MODE,
    actual: np.ndarray | None = None,
) -> np.ndarray | None:
    """Train all models on train_df, combine via equal weighting for one fold.

    Uses uniform averaging to avoid data leakage — computing 1/MAPE weights
    from the same fold's holdout actuals would optimise on the scoring data,
    producing optimistic backtest metrics.  Forward forecasts already use
    equal weights (no actuals available), so this keeps backtest and
    production behaviour consistent.

    Returns equal-weight ensemble predictions or None if all models fail.
    """
    preds: dict[str, np.ndarray] = {}

    for name in ["xgboost", "prophet", "arima"]:
        try:
            pred = _predict_single_fold(name, train_df, test_df, exog_mode=exog_mode)
            if pred is not None and not np.all(np.isnan(pred)):
                preds[name] = pred
        except Exception as e:
            log.warning("ensemble_fold_model_failed", model=name, error=str(e))

    if not preds:
        return None

    return np.mean(list(preds.values()), axis=0)


def _run_backtest_for_horizon(
    demand_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    horizon_hours: int,
    model_name: str,
    region: str,
    exog_mode: str = DEFAULT_BACKTEST_EXOG_MODE,
    bypass_redis_guard: bool = False,
) -> dict:
    """
    Walk-forward backtest for a specific forecast horizon.

    Uses expanding-window cross-validation: slides non-overlapping test
    windows across the data (up to 5 folds), training on all data before
    each window. Metrics are aggregated across all folds for a robust
    accuracy estimate.

    Args:
        demand_df: Full demand dataframe with timestamp and demand_mw
        weather_df: Full weather dataframe
        horizon_hours: Forecast horizon (24, 168, or 720 hours)
        model_name: Model to use (xgboost, prophet, arima, ensemble)
        region: Region code
        exog_mode: Backtest exogenous-weather mode.
        bypass_redis_guard: When True, skip the REQUIRE_REDIS warming
            early-return. Reserved for the nightly training job which is
            the authoritative compute path — if that job short-circuits
            on REQUIRE_REDIS, nothing ever populates the backtest keys.

    Returns:
        Dict with predictions, actuals, timestamps, metrics, num_folds,
        and fold_boundaries.
    """
    import time

    from data.feature_engineering import engineer_exogenous_features, engineer_features
    from data.preprocessing import merge_demand_weather
    from models.evaluation import (
        apply_empirical_interval,
        compute_all_metrics,
        compute_interval_coverage_drift,
        empirical_error_quantiles,
    )

    exog_mode = _normalize_backtest_exog_mode(exog_mode)
    data_hash = _compute_data_hash(demand_df, weather_df, region)
    cache_key = (region, horizon_hours, model_name, exog_mode)

    # Layer 1: In-memory cache
    if cache_key in _BACKTEST_CACHE:
        cached_result, cached_hash, cached_time = _BACKTEST_CACHE[cache_key]
        if cached_hash == data_hash and (time.time() - cached_time) < CACHE_TTL_SECONDS:
            log.info("backtest_cache_hit", region=region, horizon=horizon_hours, model=model_name)
            return cached_result

    # Layer 2: SQLite cache (survives page refresh / server restart)
    try:
        from data.cache import get_cache

        sqlite_cache = get_cache()
        sqlite_key = f"backtest:{exog_mode}:{region}:{horizon_hours}:{model_name}"
        cached_sqlite = sqlite_cache.get(sqlite_key)
        if (
            cached_sqlite is not None
            and isinstance(cached_sqlite, dict)
            and "actual" in cached_sqlite
            and cached_sqlite.get("cache_version") == _CACHE_VERSION
            and cached_sqlite.get("data_hash") == data_hash
        ):
            cached_sqlite["timestamps"] = pd.to_datetime(cached_sqlite["timestamps"]).values
            cached_sqlite["actual"] = np.array(cached_sqlite["actual"])
            cached_sqlite["predictions"] = np.array(cached_sqlite["predictions"])
            cached_sqlite.setdefault("num_folds", 1)
            cached_sqlite.setdefault("fold_boundaries", [0])
            cached_sqlite.setdefault("exog_mode", exog_mode)
            cached_sqlite.setdefault("exog_source", "unknown")
            if isinstance(cached_sqlite.get("interval"), dict):
                if "lower" in cached_sqlite["interval"]:
                    cached_sqlite["interval"]["lower"] = np.array(
                        cached_sqlite["interval"]["lower"]
                    )
                if "upper" in cached_sqlite["interval"]:
                    cached_sqlite["interval"]["upper"] = np.array(
                        cached_sqlite["interval"]["upper"]
                    )
            _BACKTEST_CACHE[cache_key] = (cached_sqlite, data_hash, time.time())
            log.info(
                "backtest_sqlite_cache_hit", region=region, horizon=horizon_hours, model=model_name
            )
            return cached_sqlite
    except Exception as e:
        log.debug("backtest_sqlite_cache_miss", error=str(e))

    # REQUIRE_REDIS: the daily training job recomputes backtests and writes
    # them to Redis. A miss at this layer means the training job hasn't
    # produced results yet — surface warming instead of running walk-forward
    # training inline (which would time out a web request). The training
    # job itself passes bypass_redis_guard=True so it can actually do the
    # compute this guard exists to defer.
    if REQUIRE_REDIS and not bypass_redis_guard:
        log.info(
            "backtest_warming_state",
            region=region,
            horizon=horizon_hours,
            model=model_name,
        )
        return {
            "error": "warming",
            "status": "warming",
            "message": "Backtests are being refreshed by the nightly training job.",
        }

    # Merge once; features are built fold-by-fold from train-history-only slices
    merged_df = merge_demand_weather(demand_df, weather_df)
    base_df = merged_df.dropna(subset=["demand_mw"]).reset_index(drop=True)

    min_train_size = 720  # 30 days minimum training data
    n_total = len(base_df)

    if n_total < min_train_size + horizon_hours:
        return {"error": "Insufficient data"}

    # Calculate number of non-overlapping folds (max 5)
    max_possible_folds = (n_total - min_train_size) // horizon_hours
    num_folds = min(5, max(1, max_possible_folds))

    log.info(
        "backtest_walk_forward_start",
        region=region,
        horizon=horizon_hours,
        model=model_name,
        exog_mode=exog_mode,
        num_folds=num_folds,
        data_points=n_total,
    )

    all_actual: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_timestamps: list[np.ndarray] = []
    fold_boundaries: list[int] = []
    exog_sources: set[str] = set()

    try:
        for fold_idx in range(num_folds):
            # Non-overlapping test windows from end, oldest first
            offset_from_end = (num_folds - fold_idx) * horizon_hours
            test_start = n_total - offset_from_end
            test_end = test_start + horizon_hours

            if test_start < min_train_size:
                log.debug(
                    "backtest_fold_skipped", fold=fold_idx + 1, reason="insufficient_train_data"
                )
                continue

            train_slice = base_df.iloc[:test_start].copy()
            test_slice = base_df.iloc[test_start:test_end].copy()
            train_df = (
                engineer_features(train_slice).dropna(subset=["demand_mw"]).reset_index(drop=True)
            )
            test_df = engineer_exogenous_features(test_slice).reset_index(drop=True)

            log.info(
                "backtest_fold_start",
                fold=fold_idx + 1,
                num_folds=num_folds,
                train_rows=len(train_df),
                test_rows=len(test_df),
            )

            fold_test_df = test_df
            fold_exog_source = "actual future weather"
            if exog_mode == "forecast_exog":
                fold_test_df, fold_exog_source = _build_forecast_exog_fold(
                    train_df, test_df, region, horizon_hours
                )
            exog_sources.add(fold_exog_source)

            # Get predictions for this fold
            fold_actual = test_slice["demand_mw"].values
            if model_name == "ensemble":
                fold_preds = _ensemble_fold(
                    train_df, fold_test_df, exog_mode=exog_mode, actual=fold_actual
                )
            else:
                fold_preds = _predict_single_fold(
                    model_name, train_df, fold_test_df, exog_mode=exog_mode
                )

            if fold_preds is None:
                log.warning("backtest_fold_failed", fold=fold_idx + 1, model=model_name)
                continue

            # NaN guard per fold
            if np.any(np.isnan(fold_preds)):
                nan_pct = np.isnan(fold_preds).sum() / len(fold_preds) * 100
                log.warning("backtest_fold_nan", fold=fold_idx + 1, nan_pct=round(nan_pct, 1))
                fold_preds = np.where(np.isnan(fold_preds), np.mean(fold_actual), fold_preds)

            # Track fold boundary (index in concatenated array)
            fold_boundaries.append(sum(len(a) for a in all_actual))
            all_actual.append(fold_actual)
            all_predictions.append(fold_preds)
            all_timestamps.append(test_slice["timestamp"].values)

            log.info("backtest_fold_complete", fold=fold_idx + 1, num_folds=num_folds)

    except Exception as e:
        log.warning("backtest_walk_forward_failed", model=model_name, error=str(e))
        return {"error": str(e)}

    if not all_actual:
        return {"error": "All folds failed"}

    # Concatenate across all folds and compute aggregate metrics
    actual = np.concatenate(all_actual)
    predictions = np.concatenate(all_predictions)
    timestamps = np.concatenate(all_timestamps)
    metrics = compute_all_metrics(actual, predictions)
    residuals = actual - predictions
    calibration_window = int(min(len(residuals), max(horizon_hours * 5, 120)))
    recent_residuals = residuals[-calibration_window:] if calibration_window else residuals
    q = empirical_error_quantiles(recent_residuals, lower_q=0.10, upper_q=0.90)
    lower_interval, upper_interval = apply_empirical_interval(
        predictions, q["lower_error"], q["upper_error"]
    )
    interval_monitor = compute_interval_coverage_drift(actual, lower_interval, upper_interval, 0.80)

    result = {
        "timestamps": timestamps,
        "actual": actual,
        "predictions": predictions,
        "metrics": metrics,
        "num_folds": len(fold_boundaries),
        "fold_boundaries": fold_boundaries,
        "exog_mode": exog_mode,
        "exog_source": ", ".join(sorted(exog_sources)) if exog_sources else "unknown",
        "interval": {
            "method": "empirical",
            "target_coverage": 0.80,
            "calibration_window_hours": calibration_window,
            "sample_size": int(q["sample_size"]),
            "lower_error": float(q["lower_error"]),
            "upper_error": float(q["upper_error"]),
            "lower": lower_interval,
            "upper": upper_interval,
            "coverage_monitor": interval_monitor,
        },
    }

    log.info(
        "backtest_walk_forward_complete",
        region=region,
        horizon=horizon_hours,
        model=model_name,
        exog_mode=exog_mode,
        folds=len(fold_boundaries),
        mape=round(metrics["mape"], 2),
        interval_recent_coverage=round(interval_monitor["recent_coverage"], 4),
        interval_drift_pp=round(interval_monitor["drift"] * 100.0, 2),
    )
    if abs(interval_monitor["drift"]) > 0.05:
        log.warning(
            "prediction_interval_coverage_drift",
            region=region,
            horizon=horizon_hours,
            model=model_name,
            recent_coverage=round(interval_monitor["recent_coverage"], 4),
            target=0.80,
            drift_pp=round(interval_monitor["drift"] * 100.0, 2),
        )

    # Cache the result (in-memory)
    _BACKTEST_CACHE[cache_key] = (result, data_hash, time.time())

    # Persist to SQLite for cross-restart durability
    try:
        from data.cache import get_cache

        sqlite_cache = get_cache()
        sqlite_key = f"backtest:{exog_mode}:{region}:{horizon_hours}:{model_name}"
        serializable = {
            "cache_version": _CACHE_VERSION,
            "data_hash": data_hash,
            "timestamps": [str(t) for t in result["timestamps"]],
            "actual": result["actual"].tolist(),
            "predictions": result["predictions"].tolist(),
            "metrics": result["metrics"],
            "num_folds": result["num_folds"],
            "fold_boundaries": result["fold_boundaries"],
            "exog_mode": result["exog_mode"],
            "exog_source": result["exog_source"],
            "interval": {
                **{k: v for k, v in result["interval"].items() if k not in ("lower", "upper")},
                "lower": result["interval"]["lower"].tolist(),
                "upper": result["interval"]["upper"].tolist(),
            },
        }
        sqlite_cache.set(sqlite_key, serializable, ttl=CACHE_TTL_SECONDS)
    except Exception as e:
        log.debug("backtest_sqlite_write_failed", error=str(e))

    return result


__all__ = [
    "_normalize_backtest_exog_mode",
    "_describe_exog_mode",
    "_build_forecast_exog_fold",
    "_backtest_tab_from_redis",
    "_predict_single_fold",
    "_ensemble_fold",
    "_run_backtest_for_horizon",
]
