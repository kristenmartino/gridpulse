"""
All Dash callbacks for the Energy Demand Forecasting Dashboard.

Sprint 5 changes:
- D2: Forecast audit trail integration (data/audit.py records every forecast)
- I1: Pipeline transformation logging (observability.PipelineLogger)
- A4+E3: Per-widget confidence badges (widget-confidence-bar callback)
- C9: Meeting-ready mode (strips chrome for projection/PDF)

Sprint 4 changes:
- Model service integration (replaces simulated noise with deterministic forecasts)
- Tab 1 KPI callback (peak demand, MAPE, reserve margin, alerts)
- Persona tab visibility (AC-7.5)
- Orphan layout ID fixes (tab4-renewable-delta, tab5-stress-breakdown)
- All pd.read_json uses io.StringIO (pandas 2.x compat)
- G2: API fallback banners + header freshness badge (data-freshness-store)
- C2: Scenario bookmarks (URL state serialize/restore via dcc.Location)
"""

import io

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import structlog
from dash import ALL, Input, Output, State, ctx, html, no_update
from plotly.subplots import make_subplots

from components.cards import (
    build_alert_card,
    build_insight_card,
    build_kpi_row,
    build_metrics_bar,
    build_model_metrics_card,
    build_news_feed,
    build_page_title,
    build_welcome_card,
)
from config import (
    CACHE_TTL_SECONDS,
    EIA_API_KEY,
    REGION_CAPACITY_MW,
    REQUIRE_REDIS,
    WEATHER_VARIABLES,
)
from data.redis_client import redis_get
from hash_utils import stable_int_seed
from personas.config import get_persona, get_welcome_card

log = structlog.get_logger()

import threading  # noqa: E402

_cache_lock = threading.Lock()
_CACHE_VERSION = 3

_MODEL_CACHE: dict = {}  # {(region, model_name, horizon): (model, data_hash, timestamp)}
_PREDICTION_CACHE: dict = {}  # {(region, horizon): (predictions, timestamps, data_hash, time)}
_BACKTEST_CACHE: dict = {}  # {(region, horizon, model, exog_mode): (result_dict, data_hash, time)}
_GENERATION_CACHE: dict = {}  # {region: (gen_df, fetch_timestamp)}
BACKTEST_EXOG_MODES = {"oracle_exog", "forecast_exog"}
DEFAULT_BACKTEST_EXOG_MODE = "forecast_exog"

# EIA fuel type code normalization
_EIA_FUEL_MAP: dict[str, str] = {
    "SUN": "solar",
    "WND": "wind",
    "NG": "gas",
    "NUC": "nuclear",
    "COL": "coal",
    "WAT": "hydro",
    "OTH": "other",
    "Solar": "solar",
    "Wind": "wind",
    "Natural Gas": "gas",
    "Nuclear": "nuclear",
    "Coal": "coal",
    "Hydro": "hydro",
    "Other": "other",
}

# Plotly dark theme defaults
PLOT_TEMPLATE = "plotly_dark"
PLOT_LAYOUT = dict(
    template=PLOT_TEMPLATE,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        color="#a1a1aa",  # --text-secondary (v2 palette)
        size=11,
        family="Inter, 'Segoe UI', system-ui, sans-serif",
    ),
    margin=dict(l=48, r=16, t=24, b=36),
    legend=dict(
        orientation="h",
        y=-0.18,
        font=dict(size=11),
        bgcolor="rgba(0,0,0,0)",
    ),
)


def _layout(*, uirevision: str | None = None, **overrides) -> dict:
    """Compose a Plotly layout dict from PLOT_LAYOUT + per-call overrides.

    ``uirevision`` is the Plotly hook that preserves user-set UI state
    (zoom, pan, legend toggles) across figure re-renders. Tie it to
    *user-meaningful* state — region, horizon, model — NOT to a raw data
    hash, so that auto-refreshes (5-min interval, store updates) reuse
    the same revision string and the chart's interaction state survives.
    Changing the string forces Plotly to re-initialize the view, which
    is the desired behavior when the user picks a new region/horizon.
    """
    layout = {**PLOT_LAYOUT, **overrides}
    if uirevision is not None:
        layout["uirevision"] = uirevision
    return layout


# Color palette (colorblind-safe — Wong 2011)
from datetime import UTC  # noqa: E402

from components.accessibility import CB_PALETTE, LINE_STYLES  # noqa: E402

COLORS = {
    "actual": CB_PALETTE["blue"],
    "prophet": CB_PALETTE["orange"],
    "arima": CB_PALETTE["green"],
    "xgboost": CB_PALETTE["sky_blue"],
    "ensemble": CB_PALETTE["vermillion"],
    "eia_forecast": "#7f7f7f",
    "temperature": CB_PALETTE["yellow"],
    "confidence": "rgba(213,94,0,0.15)",
    "gas": CB_PALETTE["orange"],
    "nuclear": CB_PALETTE["purple"],
    "coal": "#7f7f7f",
    "wind": CB_PALETTE["green"],
    "solar": CB_PALETTE["yellow"],
    "hydro": CB_PALETTE["blue"],
    "other": "#b0b0b0",
}

# Model-aware confidence band fill colors (base model color at 12% opacity)
_MODEL_BAND_COLORS = {
    "xgboost": "rgba(86,180,233,0.12)",  # sky_blue
    "prophet": "rgba(230,159,0,0.12)",  # orange
    "arima": "rgba(0,158,115,0.12)",  # green
    "ensemble": "rgba(213,94,0,0.12)",  # vermillion
}


def _compute_data_hash(demand_df: pd.DataFrame, weather_df: pd.DataFrame, region: str) -> str:
    """Compute stable input signature for cache correctness.

    Signature includes:
    - region
    - row counts
    - normalized start/end timestamps
    - lightweight content checksums over key columns
    """
    import hashlib
    import json

    def _normalize_ts(ts) -> str:
        """Strip timezone to produce a stable string regardless of tz-aware vs tz-naive."""
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        return str(t)

    def _frame_sig(df: pd.DataFrame, key_cols: list[str]) -> dict:
        frame_sig: dict[str, str | int] = {
            "rows": int(len(df)),
            "start": "",
            "end": "",
            "checksum": "",
        }
        if df.empty:
            return frame_sig

        if "timestamp" in df.columns:
            ts_bounds = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            if ts_bounds.notna().any():
                frame_sig["start"] = _normalize_ts(ts_bounds.min())
                frame_sig["end"] = _normalize_ts(ts_bounds.max())

        cols = [c for c in key_cols if c in df.columns]
        if not cols:
            return frame_sig

        sample = df.loc[:, cols].copy()
        if "timestamp" in sample.columns:
            ts = pd.to_datetime(sample["timestamp"], utc=True, errors="coerce")
            sample["timestamp"] = ts.astype("int64").fillna(-1).astype("int64")
            sample = sample.sort_values("timestamp", kind="mergesort")
        for col in cols:
            if col != "timestamp" and pd.api.types.is_numeric_dtype(sample[col]):
                sample[col] = sample[col].round(6)

        hashed = pd.util.hash_pandas_object(sample.fillna("<NA>"), index=False).to_numpy(
            dtype=np.uint64
        )
        frame_sig["checksum"] = f"{int(hashed.sum(dtype=np.uint64)):016x}"
        return frame_sig

    signature_payload = {
        "region": region,
        "demand": _frame_sig(demand_df, ["timestamp", "demand_mw"]),
        "weather": _frame_sig(
            weather_df,
            [
                "timestamp",
                "temperature_2m",
                "wind_speed_80m",
                "shortwave_radiation",
                "relative_humidity_2m",
            ],
        ),
    }
    signature_json = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(signature_json.encode("utf-8")).hexdigest()


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


def _collect_backtest_residuals(region: str, model_name: str, horizon_hours: int) -> np.ndarray:
    """Collect recent backtest residuals by model/region/horizon from cache layers."""
    residual_chunks: list[np.ndarray] = []

    # In-memory cache (most recent in-process compute path)
    for (r, h, m, _mode), (cached_result, _hash, _time) in _BACKTEST_CACHE.items():
        if r != region or h != horizon_hours:
            continue
        if m != model_name and not (model_name == "ensemble" and m in ("ensemble",)):
            continue
        actual = np.asarray(cached_result.get("actual", []), dtype=float)
        pred = np.asarray(cached_result.get("predictions", []), dtype=float)
        n = min(len(actual), len(pred))
        if n > 0:
            residual_chunks.append(actual[:n] - pred[:n])

    # Redis pre-computed backtests (common production path)
    for key in (
        f"wattcast:backtest:{DEFAULT_BACKTEST_EXOG_MODE}:{region}:{horizon_hours}",
        f"wattcast:backtest:{region}:{horizon_hours}",
    ):
        cached = redis_get(key)
        if not isinstance(cached, dict):
            continue
        actual = np.asarray(cached.get("actual", []), dtype=float)
        preds_map = cached.get("predictions", {})
        if isinstance(preds_map, dict):
            if model_name in preds_map:
                pred = np.asarray(preds_map.get(model_name, []), dtype=float)
            elif "ensemble" in preds_map:
                pred = np.asarray(preds_map.get("ensemble", []), dtype=float)
            elif preds_map:
                pred = np.asarray(next(iter(preds_map.values())), dtype=float)
            else:
                pred = np.array([], dtype=float)
            n = min(len(actual), len(pred))
            if n > 0:
                residual_chunks.append(actual[:n] - pred[:n])

    if not residual_chunks:
        return np.array([], dtype=float)
    residuals = np.concatenate(residual_chunks)
    return residuals[np.isfinite(residuals)]


def _empirical_interval_from_backtests(
    region: str,
    model_name: str,
    horizon_hours: int,
    target_coverage: float = 0.80,
) -> dict[str, float | int | bool]:
    """Estimate empirical prediction interval from recent backtest residuals."""
    from models.evaluation import empirical_error_quantiles

    residuals = _collect_backtest_residuals(region, model_name, horizon_hours)
    if residuals.size < max(24, horizon_hours // 2):
        return {"available": False}

    tail_size = int(min(residuals.size, max(horizon_hours * 5, 120)))
    recent = residuals[-tail_size:]
    alpha = (1.0 - target_coverage) / 2.0
    q = empirical_error_quantiles(recent, lower_q=alpha, upper_q=1.0 - alpha)
    return {
        "available": True,
        "lower_error": float(q["lower_error"]),
        "upper_error": float(q["upper_error"]),
        "sample_size": int(q["sample_size"]),
        "target_coverage": float(target_coverage),
        "calibration_window_hours": tail_size,
    }


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
        f"wattcast:weather-forecast-snapshot:{region}:{horizon_hours}",
        f"wattcast:weather-forecast:{region}:{horizon_hours}",
        f"wattcast:weather-forecast-snapshot:{region}",
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


def _add_confidence_bands(
    fig: "go.Figure",
    timestamps: "pd.DatetimeIndex | np.ndarray",
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
    fig: "go.Figure",
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


def _fetch_generation_cached(region: str) -> pd.DataFrame | None:
    """Fetch generation data with 3-tier caching: memory -> SQLite/API -> demo.

    Args:
        region: Balancing authority code.

    Returns:
        DataFrame with [timestamp, fuel_type, generation_mw, region] or None.
    """
    import time as _time

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


def _load_data_from_redis(region):
    """Redis fast path for load_data callback.

    Returns a 5-tuple (demand_json, weather_json, freshness_json, audit_json,
    pipeline_json) when Redis has both actuals and weather for the region,
    or None if either cache miss occurs.
    """
    import json
    from datetime import UTC, datetime

    from data.audit import audit_trail
    from observability import PipelineLogger

    cached_actuals = redis_get(f"wattcast:actuals:{region}")
    cached_weather = redis_get(f"wattcast:weather:{region}")
    if cached_actuals is None or cached_weather is None:
        return None

    pipe = PipelineLogger("load_data", region=region)
    freshness = {
        "demand": "fresh",
        "weather": "fresh",
        "alerts": "fresh",
        "timestamp": datetime.now(UTC).isoformat(),
    }

    log.info("load_data_redis_hit", region=region)
    pipe.step("fetch_demand", rows=len(cached_actuals.get("demand_mw", [])), source="redis")
    pipe.step("fetch_weather", rows=len(cached_weather.get("timestamps", [])), source="redis")

    # Convert parallel-arrays to DataFrame JSON. Sanitize spurious zero-demand
    # readings to NaN — a balancing authority never truly reads 0 MW; these
    # are missing-data artifacts (commonly legacy Redis entries written
    # before the EIA ingestion fix). Charts/KPIs handle NaN correctly; zeros
    # otherwise render as misleading dips to zero.
    demand_df = pd.DataFrame(
        {
            "timestamp": cached_actuals["timestamps"],
            "demand_mw": cached_actuals["demand_mw"],
        }
    )
    demand_df.loc[demand_df["demand_mw"] <= 0, "demand_mw"] = np.nan
    weather_cols = {k: v for k, v in cached_weather.items() if k not in ("region",)}
    weather_df = pd.DataFrame(weather_cols)
    if "timestamps" in weather_df.columns:
        weather_df = weather_df.rename(columns={"timestamps": "timestamp"})

    freshness["demand"] = "fresh"
    freshness["weather"] = "fresh"
    if len(demand_df) > 0:
        freshness["latest_data"] = str(demand_df["timestamp"].iloc[-1])

    pipe.step(
        "serialize",
        demand_cols=len(demand_df.columns),
        weather_cols=len(weather_df.columns),
    )
    audit_record = audit_trail.record_forecast(
        region=region,
        demand_source="redis",
        weather_source="redis",
        demand_rows=len(demand_df),
        weather_rows=len(weather_df),
        demand_range=(
            str(demand_df["timestamp"].iloc[0]),
            str(demand_df["timestamp"].iloc[-1]),
        )
        if len(demand_df) > 0
        else ("", ""),
        weather_range=(
            str(weather_df["timestamp"].iloc[0]),
            str(weather_df["timestamp"].iloc[-1]),
        )
        if "timestamp" in weather_df.columns and len(weather_df) > 0
        else ("", ""),
        forecast_source="redis",
    )
    pipe.step("audit_recorded", record_id=audit_record.record_id)
    pipeline_summary = pipe.done()
    return (
        demand_df.to_json(date_format="iso"),
        weather_df.to_json(date_format="iso"),
        json.dumps(freshness),
        audit_record.to_json(),
        json.dumps(pipeline_summary, default=str),
    )


def _weather_tab_from_redis(region):
    """Redis fast path for update_weather_tab callback.

    Returns a 6-tuple of Plotly figures or None if cache miss.
    """
    cached = redis_get(f"wattcast:weather-correlation:{region}")
    if cached is None:
        return None

    log.info("weather_redis_hit", region=region)
    corr_data = cached.get("correlation_matrix", {})
    imp_data = cached.get("importance", {})
    seasonal = cached.get("seasonal", {})

    # Scatter: Temperature vs Demand
    fig_temp = go.Figure(
        go.Scatter(
            x=cached.get("temperature_2m", []),
            y=cached.get("demand_mw", []),
            mode="markers",
            marker=dict(size=3, color=COLORS["actual"], opacity=0.4),
        )
    )
    fig_temp.update_layout(
        **_layout(uirevision=region, xaxis_title="Temperature (°F)", yaxis_title="Demand (MW)")
    )

    # Scatter: Wind
    fig_wind = go.Figure(
        go.Scatter(
            x=cached.get("wind_speed_80m", []),
            y=cached.get("wind_power", []),
            mode="markers",
            marker=dict(size=3, color=COLORS["wind"], opacity=0.5),
        )
    )
    fig_wind.update_layout(
        **_layout(
            uirevision=region,
            xaxis_title="Wind Speed (mph)",
            yaxis_title="Wind Power Estimate",
        )
    )

    # Scatter: Solar
    fig_solar = go.Figure(
        go.Scatter(
            x=cached.get("shortwave_radiation", []),
            y=cached.get("solar_cf", []),
            mode="markers",
            marker=dict(size=3, color=COLORS["solar"], opacity=0.5),
        )
    )
    fig_solar.update_layout(
        **_layout(
            uirevision=region,
            xaxis_title="GHI (W/m²)",
            yaxis_title="Solar Capacity Factor",
        )
    )

    # Heatmap
    corr_cols = corr_data.get("cols", [])
    corr_vals = corr_data.get("values", [])
    fig_heatmap = go.Figure(
        go.Heatmap(
            z=corr_vals,
            x=corr_cols,
            y=corr_cols,
            colorscale="RdBu",
            zmid=0,
            text=np.round(np.array(corr_vals), 2) if corr_vals else [],
            texttemplate="%{text}",
        )
    )
    fig_heatmap.update_layout(**_layout(uirevision=region))

    # Feature importance
    imp_names = imp_data.get("names", [])
    imp_vals = imp_data.get("values", [])
    fig_importance = go.Figure(
        go.Bar(
            x=imp_vals,
            y=imp_names,
            orientation="h",
            marker_color=COLORS["ensemble"],
        )
    )
    fig_importance.update_layout(**_layout(uirevision=region, xaxis_title="Correlation Strength"))

    # Seasonal decomposition
    s_ts = pd.to_datetime(seasonal.get("timestamps", []))
    s_orig = seasonal.get("original", [])
    s_trend = seasonal.get("trend", [])
    s_resid = seasonal.get("residual", [])
    fig_seasonal = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Original", "Trend (7-day)", "Residual"],
    )
    fig_seasonal.add_trace(
        go.Scatter(
            x=s_ts,
            y=s_orig,
            line=dict(color=COLORS["actual"], width=1),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig_seasonal.add_trace(
        go.Scatter(
            x=s_ts,
            y=s_trend,
            line=dict(color=COLORS["ensemble"], width=2),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig_seasonal.add_trace(
        go.Scatter(
            x=s_ts,
            y=s_resid,
            line=dict(color=COLORS["arima"], width=1),
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig_seasonal.update_layout(**_layout(uirevision=region, height=350))

    return fig_temp, fig_wind, fig_solar, fig_heatmap, fig_importance, fig_seasonal


def _format_metric(m: dict, key: str, fmt: str) -> str:
    """Format a metric value, returning 'N/A' when the key is missing or None.

    Prevents unavailable metrics from displaying as ``0`` which users can
    misread as a real (and suspiciously perfect) model score.
    """
    val = m.get(key)
    if val is None:
        return "N/A"
    return fmt.format(val)


def _models_tab_from_redis(region, selected_models: list[str] | None = None):
    """Redis fast path for update_models_tab callback.

    Returns a 6-tuple (table, fig_resid_time, fig_resid_hist, fig_resid_pred,
    fig_heatmap, fig_shap) or None if cache miss.
    """
    default_models = ["prophet", "arima", "xgboost", "ensemble"]
    selected_models = selected_models or default_models
    # Current Redis diagnostics payload is ensemble-only for residual charts.
    # Keep this fast path only when callback explicitly requests ensemble-only.
    if selected_models is not default_models and set(selected_models) != {"ensemble"}:
        return None

    cached = redis_get(f"wattcast:diagnostics:{region}")
    if cached is None:
        return None

    # uirevision keyed on region + sorted model selection so zoom/legend
    # state survives Redis refresh but resets when the user picks new models.
    uirev = f"{region}:{','.join(sorted(selected_models))}"

    log.info("diagnostics_redis_hit", region=region)
    metrics = cached.get("metrics", {})
    timestamps = pd.to_datetime(cached.get("timestamps", []))
    ensemble = np.array(cached.get("ensemble", []))
    residuals = np.array(cached.get("residuals", []))
    hourly_err = cached.get("hourly_error", {})
    fi = cached.get("feature_importance", {})

    # Metrics table
    name_map = {
        "Prophet": "prophet",
        "SARIMAX": "arima",
        "XGBoost": "xgboost",
        "Ensemble": "ensemble",
    }
    rows = []
    for display_name, key in name_map.items():
        if key not in selected_models:
            continue
        m = metrics.get(key, {})
        rows.append(
            html.Tr(
                [
                    html.Td(display_name, style={"fontWeight": "600"}),
                    html.Td(_format_metric(m, "mape", "{:.2f}%")),
                    html.Td(_format_metric(m, "rmse", "{:.0f}")),
                    html.Td(_format_metric(m, "mae", "{:.0f}")),
                    html.Td(_format_metric(m, "r2", "{:.4f}")),
                ]
            )
        )
    table = html.Table(
        [
            html.Thead(html.Tr([html.Th(h) for h in ["Model", "MAPE", "RMSE", "MAE", "R²"]])),
            html.Tbody(rows),
        ],
        className="metrics-table",
    )

    fig_resid_time = go.Figure(
        go.Scatter(
            x=timestamps,
            y=residuals,
            mode="lines",
            line=dict(color=COLORS["arima"], width=1),
        )
    )
    fig_resid_time.add_hline(y=0, line=dict(color="#F7FAFC", dash="dash", width=0.5))
    fig_resid_time.update_layout(**_layout(uirevision=uirev, yaxis_title="Residual (MW)"))

    fig_resid_hist = go.Figure(
        go.Histogram(x=residuals, nbinsx=50, marker_color=COLORS["ensemble"])
    )
    fig_resid_hist.update_layout(
        **_layout(uirevision=uirev, xaxis_title="Residual (MW)", yaxis_title="Count")
    )

    fig_resid_pred = go.Figure(
        go.Scatter(
            x=ensemble,
            y=residuals,
            mode="markers",
            marker=dict(size=2, color=COLORS["xgboost"], opacity=0.3),
        )
    )
    fig_resid_pred.add_hline(y=0, line=dict(color="#F7FAFC", dash="dash", width=0.5))
    fig_resid_pred.update_layout(
        **_layout(
            uirevision=uirev,
            xaxis_title="Predicted (MW)",
            yaxis_title="Residual (MW)",
        )
    )

    h_hours = hourly_err.get("hours", list(range(24)))
    h_vals = hourly_err.get("values", [0] * 24)
    h_median = float(np.median(h_vals)) if h_vals else 0
    fig_heatmap = go.Figure(
        go.Bar(
            x=h_hours,
            y=h_vals,
            marker_color=[COLORS["ensemble"] if e > h_median else COLORS["actual"] for e in h_vals],
        )
    )
    fig_heatmap.update_layout(
        **_layout(
            uirevision=uirev,
            xaxis_title="Hour of Day",
            yaxis_title="Mean |Error| (MW)",
        )
    )

    if "xgboost" in selected_models:
        fi_names = fi.get("names", [])
        fi_vals = fi.get("values", [])
        fig_shap = go.Figure(
            go.Bar(
                x=fi_vals[::-1],
                y=fi_names[::-1],
                orientation="h",
                marker_color=COLORS["xgboost"],
            )
        )
        fig_shap.update_layout(**_layout(uirevision=uirev, xaxis_title="Feature Importance"))
    else:
        fig_shap = _empty_figure("SHAP is available only for XGBoost. Select XGBoost above.")

    return table, fig_resid_time, fig_resid_hist, fig_resid_pred, fig_heatmap, fig_shap


def _generation_tab_from_redis(region, range_hours, demand_json, persona_id):
    """Redis fast path for update_generation_tab callback.

    Returns a 7-tuple (fig_hero, fig_mix, ren_pct, peak_ramp, min_net,
    curtailment, insight_card) or None if cache miss.
    """
    cached_gen = redis_get(f"wattcast:generation:{region}")
    if cached_gen is None or not cached_gen.get("timestamps"):
        return None

    # uirevision keyed on region + range so zoom survives Redis refresh
    # but resets when user picks a new date range.
    uirev = f"{region}:{range_hours}"

    log.info("generation_redis_hit", region=region)
    # Convert parallel-arrays to DataFrame
    gen_cols = {k: v for k, v in cached_gen.items() if k not in ("region",)}
    if "timestamps" in gen_cols:
        gen_cols["timestamp"] = gen_cols.pop("timestamps")
    gen_redis_df = pd.DataFrame(gen_cols)
    gen_redis_df["timestamp"] = pd.to_datetime(gen_redis_df["timestamp"])

    # Filter by date range
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=range_hours)
    if gen_redis_df["timestamp"].dt.tz is None:
        cutoff = cutoff.tz_localize(None)
    gen_redis_df = gen_redis_df[gen_redis_df["timestamp"] >= cutoff]

    if gen_redis_df.empty:
        return None

    # Fuel columns (everything except timestamp, region, renewable_pct)
    fuel_cols = [
        c for c in gen_redis_df.columns if c not in ("timestamp", "region", "renewable_pct")
    ]
    pivot = gen_redis_df.set_index("timestamp")[fuel_cols]
    total_gen = pivot.sum(axis=1)

    # Demand from demand-store or approximate as total gen
    demand_series = total_gen
    if demand_json:
        try:
            demand_df = pd.read_json(io.StringIO(demand_json))
            demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])
            demand_aligned = demand_df.set_index("timestamp")["demand_mw"]
            common_idx = pivot.index.intersection(demand_aligned.index)
            if len(common_idx) > 24:
                demand_series = demand_aligned.loc[common_idx]
                pivot = pivot.loc[common_idx]
                total_gen = pivot.sum(axis=1)
        except Exception:
            pass

    wind_gen = pivot.get("wind", pd.Series(0, index=pivot.index))
    solar_gen = pivot.get("solar", pd.Series(0, index=pivot.index))
    net_load = demand_series - wind_gen - solar_gen
    common_idx = pivot.index

    # KPIs
    ren_cols = [c for c in ["wind", "solar", "hydro"] if c in pivot.columns]
    if ren_cols and total_gen.mean() > 0:
        renewable_pct = float((pivot[ren_cols].sum(axis=1) / total_gen * 100).mean())
    else:
        renewable_pct = 0.0
    net_load_diff = net_load.diff()
    peak_ramp = float(net_load_diff.max()) if not net_load_diff.isna().all() else 0.0
    min_net_load = float(net_load.min())
    peak_net_load = float(net_load.max())
    curtailment_hours = int((net_load < peak_net_load * 0.2).sum()) if peak_net_load > 0 else 0

    # Charts
    fig_hero = go.Figure()
    fig_hero.add_trace(
        go.Scatter(
            x=common_idx,
            y=demand_series.values if hasattr(demand_series, "values") else demand_series,
            mode="lines",
            name="Total Demand",
            line=dict(color=COLORS["actual"], width=2),
        )
    )
    fig_hero.add_trace(
        go.Scatter(
            x=common_idx,
            y=net_load.values,
            mode="lines",
            name="Net Load",
            line=dict(color=COLORS["ensemble"], width=2.5),
        )
    )
    fig_hero.add_trace(
        go.Scatter(
            x=common_idx,
            y=demand_series.values if hasattr(demand_series, "values") else demand_series,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig_hero.add_trace(
        go.Scatter(
            x=common_idx,
            y=net_load.values,
            mode="lines",
            name="Renewable Contribution",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(45,226,196,0.15)",
        )
    )
    fig_hero.update_layout(
        **_layout(
            uirevision=uirev,
            yaxis_title="MW",
            hovermode="x unified",
            title=f"Demand vs Net Load \u2014 {region}",
        )
    )

    fig_mix = go.Figure()
    fuel_order = [
        "nuclear",
        "coal",
        "gas",
        "natural_gas",
        "hydro",
        "wind",
        "solar",
        "oil",
        "other",
    ]
    for fuel in fuel_order:
        if fuel in pivot.columns:
            fig_mix.add_trace(
                go.Scatter(
                    x=pivot.index,
                    y=pivot[fuel],
                    mode="lines",
                    name=fuel.replace("_", " ").title(),
                    stackgroup="one",
                    line=dict(width=0),
                    fillcolor=COLORS.get(fuel, "#95a5a6"),
                )
            )
    fig_mix.update_layout(
        **_layout(
            uirevision=uirev,
            yaxis_title="Generation (MW)",
            hovermode="x unified",
            title=f"Generation Mix \u2014 {region}",
        )
    )

    from components.insights import build_insight_card, generate_tab4_insights

    persona = persona_id or "grid_ops"
    insights = generate_tab4_insights(
        persona_id=persona,
        region=region,
        net_load=net_load,
        demand=demand_series,
        renewable_pct=renewable_pct,
        pivot=pivot,
        timestamps=pd.DatetimeIndex(common_idx),
    )
    insight_card = build_insight_card(insights, persona, "tab-generation")

    return (
        fig_hero,
        fig_mix,
        f"{renewable_pct:.1f}%",
        f"{peak_ramp:,.0f} MW/hr",
        f"{min_net_load:,.0f} MW",
        str(curtailment_hours),
        insight_card,
    )


def _alerts_tab_from_redis(region):
    """Redis fast path for update_alerts_tab callback.

    Returns an 8-tuple (alert_cards, stress_str, stress_label_span, breakdown,
    fig_anomaly, fig_temp, fig_timeline, weather_context) or None if cache miss.
    """
    cached = redis_get(f"wattcast:alerts:{region}")
    if cached is None:
        return None

    empty = _empty_figure("Loading...")
    log.info("alerts_redis_hit", region=region)
    alerts = cached.get("alerts", [])
    stress = cached.get("stress_score", 20)
    stress_label = cached.get("stress_label", "Normal")
    counts = cached.get("alert_counts", {})
    anomaly = cached.get("anomaly", {})
    temp_data = cached.get("temperature", {})

    # Build alert cards
    alert_cards = []
    if alerts:
        for a in alerts:
            alert_cards.append(
                build_alert_card(
                    event=a["event"],
                    headline=a["headline"],
                    severity=a["severity"],
                    expires=a.get("expires", "")[:16] if a.get("expires") else None,
                )
            )
    else:
        alert_cards = [
            html.P(
                "No active alerts",
                style={"color": "#A8B3C7", "textAlign": "center", "padding": "20px"},
            )
        ]

    stress_color = "positive" if stress < 30 else ("negative" if stress >= 60 else "neutral")
    n_crit = counts.get("critical", 0)
    n_warn = counts.get("warning", 0)
    n_info = counts.get("info", 0)
    breakdown_items = []
    if n_crit:
        breakdown_items.append(
            html.Div(
                f"\U0001f534 Critical: {n_crit}",
                style={"fontSize": "0.75rem", "color": "#FF5C7A"},
            )
        )
    if n_warn:
        breakdown_items.append(
            html.Div(
                f"\U0001f7e1 Warning: {n_warn}",
                style={"fontSize": "0.75rem", "color": "#FFB84D"},
            )
        )
    if n_info:
        breakdown_items.append(
            html.Div(
                f"\U0001f535 Info: {n_info}",
                style={"fontSize": "0.75rem", "color": "#56B4E9"},
            )
        )
    if not alerts:
        breakdown_items.append(
            html.Div(
                "No active alerts",
                style={"fontSize": "0.75rem", "color": "#A8B3C7"},
            )
        )
    breakdown = html.Div(breakdown_items)

    # Anomaly detection chart
    a_ts = pd.to_datetime(anomaly.get("timestamps", []))
    a_demand = anomaly.get("demand", [])
    a_upper = anomaly.get("upper", [])
    a_lower = anomaly.get("lower", [])
    a_anom_ts = pd.to_datetime(anomaly.get("anomaly_timestamps", []))
    a_anom_vals = anomaly.get("anomaly_values", [])

    if len(a_ts) > 0:
        fig_anomaly = go.Figure()
        fig_anomaly.add_trace(
            go.Scatter(x=a_ts, y=a_demand, name="Demand", line=dict(color=COLORS["actual"]))
        )
        fig_anomaly.add_trace(
            go.Scatter(
                x=a_ts,
                y=a_upper,
                name="Upper (2\u03c3)",
                line=dict(color="#FF5C7A", dash="dash", width=1),
            )
        )
        fig_anomaly.add_trace(
            go.Scatter(
                x=a_ts,
                y=a_lower,
                name="Lower (2\u03c3)",
                line=dict(color="#FF5C7A", dash="dash", width=1),
            )
        )
        if len(a_anom_ts) > 0:
            fig_anomaly.add_trace(
                go.Scatter(
                    x=a_anom_ts,
                    y=a_anom_vals,
                    mode="markers",
                    name="Anomaly",
                    marker=dict(color="#FF5C7A", size=8, symbol="diamond"),
                )
            )
        fig_anomaly.update_layout(**_layout(uirevision=region, yaxis_title="MW"))
    else:
        fig_anomaly = empty

    # Temperature chart
    t_ts = pd.to_datetime(temp_data.get("timestamps", []))
    t_vals = temp_data.get("values", [])
    if len(t_ts) > 0:
        fig_temp = go.Figure()
        fig_temp.add_trace(
            go.Scatter(x=t_ts, y=t_vals, name="Temperature", line=dict(color=COLORS["temperature"]))
        )
        for t in [95, 100, 105]:
            fig_temp.add_hline(
                y=t,
                line=dict(color="#FF5C7A", dash="dot", width=1),
                annotation_text=f"{t}\u00b0F",
                annotation_position="right",
            )
        fig_temp.update_layout(**_layout(uirevision=region, yaxis_title="\u00b0F"))
    else:
        fig_temp = empty

    # Historical event timeline (static)
    events = [
        ("2021-02-15", "Winter Storm Uri", "ERCOT", 95),
        ("2022-09-06", "CA Heat Wave", "CAISO", 80),
        ("2023-07-20", "Heat Dome", "CAISO", 85),
        ("2024-04-08", "Solar Eclipse", "PJM", 40),
    ]
    fig_timeline = go.Figure()
    for date, name, reg, sev in events:
        color = COLORS["ensemble"] if reg == region else "#A8B3C7"
        fig_timeline.add_trace(
            go.Scatter(
                x=[date],
                y=[sev],
                mode="markers+text",
                text=[name],
                textposition="top center",
                marker=dict(size=12, color=color),
                showlegend=False,
            )
        )
    fig_timeline.update_layout(
        **_layout(
            uirevision=region,
            xaxis_title="Date",
            yaxis_title="Severity Score",
            yaxis_range=[0, 100],
        )
    )

    # Weather context: build from cached temperature if available
    weather_context = html.Div()
    if t_vals:
        import dash_bootstrap_components as dbc

        last_temp = float(t_vals[-1])
        color = "#FF5C7A" if last_temp >= 95 else ("#FFB84D" if last_temp >= 85 else "#2BD67B")
        weather_context = dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.P("TEMPERATURE", className="kpi-label"),
                            html.H4(
                                f"{last_temp:.0f}°F",
                                className="kpi-value",
                                style={"fontSize": "1.3rem"},
                            ),
                        ],
                        className="kpi-card",
                        style={"borderTop": f"3px solid {color}"},
                    ),
                    md=3,
                ),
            ],
            className="g-2",
        )

    return (
        alert_cards,
        str(stress),
        html.Span(stress_label, className=f"kpi-delta {stress_color}"),
        breakdown,
        fig_anomaly,
        fig_temp,
        fig_timeline,
        weather_context,
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


def _backtest_tab_from_redis(region, horizon_hours, model_name, persona_id):
    """Redis fast path for backtest tab.

    Returns a 7-tuple (fig, mape_str, rmse_str, mae_str, r2_str,
    explanation, insight_card) or None if cache miss.
    """
    exog_mode = DEFAULT_BACKTEST_EXOG_MODE
    cached = redis_get(f"wattcast:backtest:{exog_mode}:{region}:{horizon_hours}")
    if cached is None:
        cached = redis_get(f"wattcast:backtest:{region}:{horizon_hours}")
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


def register_callbacks(app):
    """Register all callbacks with the Dash app."""

    # ── 1. DATA LOADING ───────────────────────────────────────

    @app.callback(
        [
            Output("demand-store", "data"),
            Output("weather-store", "data"),
            Output("data-freshness-store", "data"),
            Output("audit-store", "data"),
            Output("pipeline-log-store", "data"),
        ],
        [Input("region-selector", "value"), Input("refresh-interval", "n_intervals")],
    )
    def load_data(region, _n):
        """Load demand + weather data for selected region.

        G2: Tracks which sources served fresh vs stale data.
        D2: Records audit trail for every forecast.
        I1: Logs each pipeline transformation step.

        v2: Reads pre-computed data from Redis when available.
        """
        import json
        from datetime import datetime

        from data.audit import audit_trail
        from observability import PipelineLogger

        pipe = PipelineLogger("load_data", region=region)
        freshness = {
            "demand": "fresh",
            "weather": "fresh",
            "alerts": "fresh",
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # ── v2 Redis fast path ──────────────────────────────
        if region:
            redis_result = _load_data_from_redis(region)
            if redis_result is not None:
                return redis_result

        # ── REQUIRE_REDIS gate ───────────────────────────────
        # In staging/production the scoring job owns the pipeline. A Redis
        # miss means either the first scoring run hasn't finished yet or
        # Redis was flushed. Surface this as a "warming" state so the UI
        # can render a skeleton — never block the request on API fetches
        # or inline training.
        if REQUIRE_REDIS:
            log.info("load_data_warming_state", region=region)
            pipe.step("redis_miss_warming", source="redis")
            freshness = {
                "demand": "warming",
                "weather": "warming",
                "alerts": "warming",
                "timestamp": datetime.now(UTC).isoformat(),
            }
            pipeline_summary = pipe.done()
            empty_demand = pd.DataFrame(columns=["timestamp", "demand_mw"]).to_json(
                date_format="iso"
            )
            empty_weather = pd.DataFrame(columns=["timestamp"]).to_json(date_format="iso")
            return (
                empty_demand,
                empty_weather,
                json.dumps(freshness),
                "{}",
                json.dumps(pipeline_summary, default=str),
            )

        # ── v1 compute fallback (dev only) ───────────────────
        try:
            if EIA_API_KEY and EIA_API_KEY != "your_eia_api_key_here":
                from data.eia_client import fetch_demand
                from data.weather_client import fetch_weather

                try:
                    demand_df = fetch_demand(region)
                    if demand_df.empty:
                        log.warning("demand_empty_fallback_to_demo", region=region)
                        from data.demo_data import generate_demo_demand

                        demand_df = generate_demo_demand(region)
                        freshness["demand"] = "stale"
                        pipe.step("fetch_demand", rows=len(demand_df), source="demo_fallback")
                    else:
                        pipe.step("fetch_demand", rows=len(demand_df), source="eia_api")
                except Exception as e:
                    log.warning("demand_fallback_to_demo", region=region, error=str(e))
                    from data.demo_data import generate_demo_demand

                    demand_df = generate_demo_demand(region)
                    freshness["demand"] = "stale"
                    pipe.step("fetch_demand", rows=len(demand_df), source="demo_fallback")
                try:
                    weather_df = fetch_weather(region)
                    if weather_df.empty:
                        log.warning("weather_empty_fallback_to_demo", region=region)
                        from data.demo_data import generate_demo_weather

                        weather_df = generate_demo_weather(region)
                        freshness["weather"] = "stale"
                        pipe.step("fetch_weather", rows=len(weather_df), source="demo_fallback")
                    else:
                        pipe.step("fetch_weather", rows=len(weather_df), source="open_meteo")
                except Exception as e:
                    log.warning("weather_fallback_to_demo", region=region, error=str(e))
                    from data.demo_data import generate_demo_weather

                    weather_df = generate_demo_weather(region)
                    freshness["weather"] = "stale"
                    pipe.step("fetch_weather", rows=len(weather_df), source="demo_fallback")
            else:
                from data.demo_data import generate_demo_demand, generate_demo_weather

                demand_df = generate_demo_demand(region)
                weather_df = generate_demo_weather(region)
                freshness["demand"] = "demo"
                freshness["weather"] = "demo"
                pipe.step("fetch_demand", rows=len(demand_df), source="demo")
                pipe.step("fetch_weather", rows=len(weather_df), source="demo")

            pipe.step(
                "serialize",
                demand_cols=len(demand_df.columns),
                weather_cols=len(weather_df.columns),
            )

            # D2: Record audit trail
            demand_range = ("", "")
            weather_range = ("", "")
            if "timestamp" in demand_df.columns and len(demand_df) > 0:
                demand_range = (
                    str(demand_df["timestamp"].min()),
                    str(demand_df["timestamp"].max()),
                )
            if "timestamp" in weather_df.columns and len(weather_df) > 0:
                weather_range = (
                    str(weather_df["timestamp"].min()),
                    str(weather_df["timestamp"].max()),
                )

            # Add latest data timestamp to freshness for display
            if demand_range[1]:
                freshness["latest_data"] = demand_range[1]

            audit_record = audit_trail.record_forecast(
                region=region,
                demand_source=freshness["demand"],
                weather_source=freshness["weather"],
                demand_rows=len(demand_df),
                weather_rows=len(weather_df),
                demand_range=demand_range,
                weather_range=weather_range,
                forecast_source="simulated" if freshness["demand"] == "demo" else "api",
            )
            pipe.step("audit_recorded", record_id=audit_record.record_id)

            pipeline_summary = pipe.done()

            return (
                demand_df.to_json(date_format="iso"),
                weather_df.to_json(date_format="iso"),
                json.dumps(freshness),
                audit_record.to_json(),
                json.dumps(pipeline_summary, default=str),
            )
        except Exception as e:
            log.error("data_load_failed", region=region, error=str(e))
            from data.demo_data import generate_demo_demand, generate_demo_weather

            freshness["demand"] = "error"
            freshness["weather"] = "error"
            pipe.step("error_fallback", error=str(e)[:100])
            pipeline_summary = pipe.done()
            return (
                generate_demo_demand(region).to_json(date_format="iso"),
                generate_demo_weather(region).to_json(date_format="iso"),
                json.dumps(freshness),
                "{}",
                json.dumps(pipeline_summary, default=str),
            )

    # ── 2. PERSONA SWITCHING ──────────────────────────────────

    @app.callback(
        [
            Output("welcome-card", "children"),
            Output("kpi-cards", "children"),
            Output("dashboard-tabs", "active_tab"),
        ],
        [
            Input("persona-selector", "value"),
            Input("region-selector", "value"),
        ],
        [
            State("demand-store", "data"),
            State("weather-store", "data"),
            State("dashboard-tabs", "active_tab"),
        ],
    )
    def switch_persona(persona_id, region, demand_json, weather_json, current_tab):
        """Reconfigure dashboard for selected persona with live data.

        Fires on persona change AND region change (immediate, no API wait).
        Only switches active tab when the persona selector triggered the callback.
        Hides standalone welcome/KPI cards when on overview tab (overview has its own).
        """
        persona = get_persona(persona_id)

        # Only switch active tab when persona changed (not on region/data change)
        triggered = ctx.triggered_id
        active_tab = persona.default_tab if triggered == "persona-selector" else no_update

        # Determine which tab we'll land on
        landing_tab = active_tab if active_tab is not no_update else current_tab

        # On overview tab, hide standalone welcome/KPI (overview has its own inline versions)
        if landing_tab == "tab-overview":
            return html.Div(), html.Div(), active_tab

        card_data = get_welcome_card(persona_id)

        from personas.welcome import generate_welcome_message

        # Parse from the demand-store; Redis fast path has already populated it.
        demand_df = None
        weather_df = None
        if demand_json:
            demand_df = pd.read_json(io.StringIO(demand_json))
            if weather_json:
                weather_df = pd.read_json(io.StringIO(weather_json))

        message = generate_welcome_message(persona_id, region, demand_df, weather_df)

        welcome = build_welcome_card(
            title=card_data["title"],
            message=message,
            avatar=card_data["avatar"],
            color=card_data["color"],
        )
        kpis = _build_persona_kpis(persona_id, region, demand_df, weather_df)

        return welcome, kpis, active_tab

    # ── 2b. PERSONA TAB VISIBILITY (AC-7.5) ──────────────────
    # NOTE: dbc.Tab uses tab_id (not id) and does not support dynamic
    # "disabled" toggling via callbacks.  Persona-based tab prioritisation
    # is handled by default_tab selection in the persona switcher above.

    # ── 3. OVERVIEW TAB (R2 — v2 linear stack) ────────────────
    # Single callback drives the 5 dynamic regions of the 7-section
    # mission-control stack. The footer is rendered statically in the
    # tab_overview layout. Persona affects only the InsightCard tone;
    # all charts and KPIs are persona-independent (matches v2 dashboard).

    @app.callback(
        [
            Output("overview-title", "children"),
            Output("overview-metrics-bar", "children"),
            Output("overview-spotlight-chart", "figure"),
            Output("overview-model-card", "children"),
            Output("overview-insight-card", "children"),
        ],
        [
            Input("demand-store", "data"),
            Input("dashboard-tabs", "active_tab"),
            Input("persona-selector", "value"),
        ],
        [
            State("weather-store", "data"),
            State("region-selector", "value"),
            State("data-freshness-store", "data"),
        ],
    )
    def update_overview_tab(
        demand_json, active_tab, persona_id, weather_json, region, freshness_data
    ):
        """Render the v2 linear-stack Overview: title, metrics, chart, model, insight."""
        if active_tab != "tab-overview":
            return [no_update] * 5

        persona_id = persona_id or "grid_ops"
        region = region or "FPL"

        try:
            demand_df = None
            if demand_json:
                demand_df = pd.read_json(io.StringIO(demand_json))
            # weather_json + freshness_data reserved for future inline drivers panel
            del weather_json, freshness_data

            # 1. Title block (region name + subtitle)
            title = _build_overview_title(region)

            # 2. MetricsBar (5-up KPI row)
            metrics_bar = build_metrics_bar(_build_overview_metrics_items(demand_df))

            # 3. Hero forecast chart (actual + dashed forecast + confidence band)
            chart = _build_overview_hero_chart(region, demand_df)

            # 4. ModelMetricsCard
            model_card = _build_overview_model_card(region)

            # 5. InsightCard
            insight = _build_overview_insight(region, demand_df, persona_id)

            return (title, metrics_bar, chart, model_card, insight)
        except Exception as exc:
            log.exception("update_overview_tab_failed")
            err_msg = f"{type(exc).__name__}: {exc}"
            err_div = html.Div(
                err_msg,
                style={"color": "var(--danger)", "fontSize": "0.8rem", "padding": "8px"},
            )
            return (err_div, html.Div(), _empty_figure(err_msg), html.Div(), err_div)

    # ── 3b. NEXD-8: SESSION CHANGE DETECTION ──────────────────
    # Snapshot store kept around even though the "What Changed" card is gone
    # in R2 — reserved for future R-phase reuse (R4a Forecast may wire it
    # into the Drivers panel as "what changed since last visit").

    @app.callback(
        [
            Output("session-snapshot-store", "data"),
            Output("changes-store", "data"),
        ],
        [
            Input("demand-store", "data"),
            Input("audit-store", "data"),
            Input("data-freshness-store", "data"),
        ],
        [
            State("session-snapshot-store", "data"),
            State("region-selector", "value"),
            State("persona-selector", "value"),
        ],
        prevent_initial_call=False,
    )
    def compute_session_changes(
        demand_json, audit_json, freshness_json, prev_snapshots, region, persona
    ):
        """Compute snapshot of current data and diff against last visit."""
        from config import feature_enabled

        if not feature_enabled("what_changed"):
            return no_update, no_update

        import json as _json

        from data.session_diff import SessionSnapshot, compute_diff, compute_snapshot

        region = region or "FPL"
        persona = persona or "grid_ops"

        # Parse inputs
        demand_df = None
        if demand_json:
            demand_df = pd.read_json(io.StringIO(demand_json))

        audit_data = None
        if audit_json:
            audit_data = _json.loads(audit_json) if isinstance(audit_json, str) else audit_json

        freshness_data = None
        if freshness_json:
            freshness_data = (
                _json.loads(freshness_json) if isinstance(freshness_json, str) else freshness_json
            )

        # Build current snapshot
        current = compute_snapshot(region, persona, demand_df, audit_data, freshness_data)

        # Load previous snapshots dict (may be None on first visit)
        snapshots = prev_snapshots if isinstance(prev_snapshots, dict) else {}

        # Diff against previous for this region
        changes = []
        prev_entry = snapshots.get(region)
        if prev_entry and isinstance(prev_entry, dict):
            prev_snap_data = prev_entry.get("snapshot")
            if prev_snap_data:
                prev_snap = SessionSnapshot.from_dict(prev_snap_data)
                changes = [c.to_dict() for c in compute_diff(prev_snap, current, persona)]

        # Save current snapshot for this region
        snapshots[region] = {
            "snapshot": current.to_dict(),
            "timestamp": current.timestamp,
        }

        return snapshots, _json.dumps(changes)

    # ── 4. TAB 1: DEMAND FORECAST ─────────────────────────────

    @app.callback(
        Output("tab1-forecast-chart", "figure"),
        [
            Input("demand-store", "data"),
            Input("weather-store", "data"),
            Input("tab1-weather-overlay", "value"),
            Input("tab1-timerange", "value"),
            Input("tab1-model-toggle", "value"),
        ],
        [State("region-selector", "value")],
        prevent_initial_call=True,
    )
    def update_forecast_chart(demand_json, weather_json, overlay, timerange, models_shown, region):
        """Build the historical demand chart (actual demand only)."""
        if not demand_json:
            return _empty_figure("No demand data loaded")

        demand_df = pd.read_json(io.StringIO(demand_json))
        demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])

        hours = int(timerange)
        if len(demand_df) > hours:
            demand_df = demand_df.tail(hours)

        fig = go.Figure()

        # Actual demand only. Mask spurious zero-demand hours (EIA returns 0
        # for missing observations; a balancing authority never truly reads
        # 0 MW) so plotly draws a gap instead of a misleading dip to zero.
        demand_y = demand_df["demand_mw"].where(demand_df["demand_mw"] > 0)
        fig.add_trace(
            go.Scatter(
                x=demand_df["timestamp"],
                y=demand_y,
                mode="lines",
                name="Actual Demand",
                line=dict(color=COLORS["actual"], width=2),
                fill="tozeroy",
                fillcolor="rgba(56,208,255,0.10)",
                connectgaps=False,
            )
        )

        # Weather overlay
        if overlay and "temp" in overlay and weather_json:
            weather_df = pd.read_json(io.StringIO(weather_json))
            weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
            weather_df = weather_df[weather_df["timestamp"].isin(demand_df["timestamp"])]
            if not weather_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=weather_df["timestamp"],
                        y=weather_df["temperature_2m"],
                        mode="lines",
                        name="Temperature (°F)",
                        line=dict(color=COLORS["temperature"], width=1.5),
                        yaxis="y2",
                    )
                )
                fig.update_layout(
                    yaxis2=dict(
                        title="Temperature (°F)",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                        color=COLORS["temperature"],
                    )
                )

        fig.update_layout(
            **_layout(
                uirevision=f"{region}:{timerange}",
                title=f"Historical Demand - {region}",
                xaxis_title="Time (UTC)",
                yaxis_title="Demand (MW)",
                hovermode="x unified",
            )
        )
        return fig

    # ── 4b. TAB 1 KPIs ────────────────────────────────────────

    @app.callback(
        [
            Output("tab1-peak-value", "children"),
            Output("tab1-peak-time", "children"),
            Output("tab1-mape-value", "children"),
            Output("tab1-reserve-value", "children"),
            Output("tab1-reserve-status", "children"),
            Output("tab1-alerts-count", "children"),
            Output("tab1-alerts-summary", "children"),
        ],
        [Input("demand-store", "data"), Input("weather-store", "data")],
        [State("region-selector", "value")],
        prevent_initial_call=True,
    )
    def update_tab1_kpis(demand_json, weather_json, region):
        """Update Tab 1 KPI cards with historical demand stats."""
        if not demand_json:
            return "No data", "", "No data", "No data", "", "0", ""

        demand_df = pd.read_json(io.StringIO(demand_json))
        demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])

        # Filter to valid demand data
        valid_data = demand_df.dropna(subset=["demand_mw"])
        if valid_data.empty:
            return "No data", "", "No data", "No data", "", "0", ""

        # Peak demand
        peak_mw = valid_data["demand_mw"].max()
        peak_idx = valid_data["demand_mw"].idxmax()
        peak_time = valid_data.loc[peak_idx, "timestamp"]
        peak_str = f"{int(peak_mw):,} MW"
        peak_time_str = peak_time.strftime("%b %d %H:%M UTC")

        # Average demand
        avg_mw = valid_data["demand_mw"].mean()
        avg_str = f"{int(avg_mw):,} MW"

        # Min demand
        min_mw = valid_data["demand_mw"].min()
        min_idx = valid_data["demand_mw"].idxmin()
        min_time = valid_data.loc[min_idx, "timestamp"]
        min_str = f"{int(min_mw):,} MW"
        min_time_str = min_time.strftime("%b %d %H:%M UTC")

        # Data points count
        data_points = len(valid_data)
        data_str = f"{data_points:,}"
        days_str = html.Span(
            f"~{data_points // 24} days",
            className="kpi-delta neutral",
            style={"fontSize": "0.75rem"},
        )

        return peak_str, peak_time_str, avg_str, min_str, min_time_str, data_str, days_str

    # ── 4c. TAB 1 INSIGHT CARD ───────────────────────────────────

    @app.callback(
        Output("tab1-insight-card", "children"),
        [
            Input("demand-store", "data"),
            Input("weather-store", "data"),
            Input("tab1-timerange", "value"),
            Input("persona-selector", "value"),
        ],
        [State("region-selector", "value")],
        prevent_initial_call=True,
    )
    def update_tab1_insights(demand_json, weather_json, timerange, persona_id, region):
        """Generate persona-aware insights for Historical Demand tab."""
        from components.insights import build_insight_card, generate_tab1_insights

        if not demand_json:
            return html.Div()

        demand_df = pd.read_json(io.StringIO(demand_json))
        weather_df = None
        if weather_json:
            weather_df = pd.read_json(io.StringIO(weather_json))

        timerange_hours = int(timerange) if timerange else 168
        persona = persona_id or "grid_ops"

        insights = generate_tab1_insights(
            persona, region or "FPL", demand_df, weather_df, timerange_hours
        )
        return build_insight_card(insights, persona, "tab-forecast")

    # ── 5. TAB 2: WEATHER CORRELATION ─────────────────────────

    @app.callback(
        [
            Output("tab2-scatter-temp", "figure"),
            Output("tab2-scatter-wind", "figure"),
            Output("tab2-scatter-solar", "figure"),
            Output("tab2-heatmap", "figure"),
            Output("tab2-feature-importance", "figure"),
            Output("tab2-seasonal", "figure"),
        ],
        [
            Input("demand-store", "data"),
            Input("weather-store", "data"),
            Input("dashboard-tabs", "active_tab"),
        ],
        [State("region-selector", "value")],
        prevent_initial_call=True,
    )
    def update_weather_tab(demand_json, weather_json, active_tab, region):
        """Update all Tab 2 charts."""
        if active_tab != "tab-weather":
            return [no_update] * 6
        if not demand_json or not weather_json:
            empty = _empty_figure("Loading...")
            return empty, empty, empty, empty, empty, empty

        # ── v2 Redis fast path ──────────────────────────────
        if region:
            redis_result = _weather_tab_from_redis(region)
            if redis_result is not None:
                return redis_result

        # ── v1 compute fallback ─────────────────────────────
        demand_df = pd.read_json(io.StringIO(demand_json))
        weather_df = pd.read_json(io.StringIO(weather_json))
        demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])
        weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
        merged = demand_df.merge(weather_df, on="timestamp", how="inner")

        fig_temp = go.Figure(
            go.Scatter(
                x=merged["temperature_2m"],
                y=merged["demand_mw"],
                mode="markers",
                marker=dict(size=3, color=COLORS["actual"], opacity=0.4),
            )
        )
        fig_temp.update_layout(
            **_layout(uirevision=region, xaxis_title="Temperature (°F)", yaxis_title="Demand (MW)")
        )

        from data.feature_engineering import compute_solar_capacity_factor, compute_wind_power

        merged["wind_power"] = compute_wind_power(merged["wind_speed_80m"])
        fig_wind = go.Figure(
            go.Scatter(
                x=merged["wind_speed_80m"],
                y=merged["wind_power"],
                mode="markers",
                marker=dict(size=3, color=COLORS["wind"], opacity=0.5),
            )
        )
        fig_wind.update_layout(
            **_layout(
                uirevision=region,
                xaxis_title="Wind Speed (mph)",
                yaxis_title="Wind Power Estimate",
            )
        )

        merged["solar_cf"] = compute_solar_capacity_factor(merged["shortwave_radiation"])
        fig_solar = go.Figure(
            go.Scatter(
                x=merged["shortwave_radiation"],
                y=merged["solar_cf"],
                mode="markers",
                marker=dict(size=3, color=COLORS["solar"], opacity=0.5),
            )
        )
        fig_solar.update_layout(
            **_layout(
                uirevision=region,
                xaxis_title="GHI (W/m²)",
                yaxis_title="Solar Capacity Factor",
            )
        )

        corr_cols = [
            c
            for c in [
                "demand_mw",
                "temperature_2m",
                "wind_speed_80m",
                "shortwave_radiation",
                "relative_humidity_2m",
                "cloud_cover",
                "surface_pressure",
            ]
            if c in merged.columns
        ]
        corr = merged[corr_cols].corr()
        fig_heatmap = go.Figure(
            go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="RdBu",
                zmid=0,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
            )
        )
        fig_heatmap.update_layout(**_layout(uirevision=region))

        importance = corr["demand_mw"].drop("demand_mw").abs().sort_values(ascending=True)
        fig_importance = go.Figure(
            go.Bar(
                x=importance.values,
                y=importance.index,
                orientation="h",
                marker_color=COLORS["ensemble"],
            )
        )
        fig_importance.update_layout(
            **_layout(uirevision=region, xaxis_title="Correlation Strength")
        )

        demand_ts = merged.set_index("timestamp")["demand_mw"].resample("h").mean().dropna()
        trend = demand_ts.rolling(168, center=True).mean()
        residual = demand_ts - trend
        fig_seasonal = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            subplot_titles=["Original", "Trend (7-day)", "Residual"],
        )
        fig_seasonal.add_trace(
            go.Scatter(
                x=demand_ts.index,
                y=demand_ts.values,
                line=dict(color=COLORS["actual"], width=1),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig_seasonal.add_trace(
            go.Scatter(
                x=trend.index,
                y=trend.values,
                line=dict(color=COLORS["ensemble"], width=2),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig_seasonal.add_trace(
            go.Scatter(
                x=residual.index,
                y=residual.values,
                line=dict(color=COLORS["arima"], width=1),
                showlegend=False,
            ),
            row=3,
            col=1,
        )
        fig_seasonal.update_layout(**_layout(uirevision=region, height=350))

        return fig_temp, fig_wind, fig_solar, fig_heatmap, fig_importance, fig_seasonal

    # ── 6. MODELS TAB: COMPARISON & DIAGNOSTICS ────────────────────────────

    @app.callback(
        [
            Output("tab3-metrics-table", "children"),
            Output("tab3-residuals-time", "figure"),
            Output("tab3-residuals-hist", "figure"),
            Output("tab3-residuals-pred", "figure"),
            Output("tab3-error-heatmap", "figure"),
            Output("tab3-shap", "figure"),
        ],
        [
            Input("demand-store", "data"),
            Input("dashboard-tabs", "active_tab"),
            Input("tab3-model-selector", "value"),
        ],
        [State("region-selector", "value")],
        prevent_initial_call=True,
    )
    def update_models_tab(demand_json, active_tab, selected_models, region):
        """Update Models tab diagnostics using model service."""
        if active_tab != "tab-models":
            return [no_update] * 6
        if not demand_json:
            empty = _empty_figure("Loading...")
            return html.P("Loading..."), empty, empty, empty, empty, empty
        if not selected_models:
            empty = _empty_figure("Select at least one model to view diagnostics.")
            return html.P("No model selected."), empty, empty, empty, empty, empty

        # Redis fast path is valid only for ensemble-only diagnostics payloads.
        if region:
            redis_result = _models_tab_from_redis(region, selected_models)
            if redis_result is not None:
                return redis_result

        # ── v1 compute fallback ─────────────────────────────
        demand_df = pd.read_json(io.StringIO(demand_json))
        demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])
        actual = demand_df["demand_mw"].values

        from models.model_service import get_forecasts

        forecasts = get_forecasts(region, demand_df, selected_models)
        metrics = forecasts.get("metrics", {})
        model_order = ["prophet", "arima", "xgboost", "ensemble"]
        selected = [m for m in model_order if m in set(selected_models)]

        # Metrics table
        name_map = {
            "Prophet": "prophet",
            "SARIMAX": "arima",
            "XGBoost": "xgboost",
            "Ensemble": "ensemble",
        }
        rows = []
        for display_name, key in name_map.items():
            if key not in selected:
                continue
            m = metrics.get(key, {})
            rows.append(
                html.Tr(
                    [
                        html.Td(display_name, style={"fontWeight": "600"}),
                        html.Td(_format_metric(m, "mape", "{:.2f}%")),
                        html.Td(_format_metric(m, "rmse", "{:.0f}")),
                        html.Td(_format_metric(m, "mae", "{:.0f}")),
                        html.Td(_format_metric(m, "r2", "{:.4f}")),
                    ]
                )
            )
        table = html.Table(
            [
                html.Thead(html.Tr([html.Th(h) for h in ["Model", "MAPE", "RMSE", "MAE", "R²"]])),
                html.Tbody(rows),
            ],
            className="metrics-table",
        )

        timestamps = demand_df["timestamp"]
        model_labels = {
            "prophet": "Prophet",
            "arima": "SARIMAX",
            "xgboost": "XGBoost",
            "ensemble": "Ensemble",
        }
        model_colors = {
            "prophet": COLORS["prophet"],
            "arima": COLORS["arima"],
            "xgboost": COLORS["xgboost"],
            "ensemble": COLORS["ensemble"],
        }
        model_residuals: dict[str, np.ndarray] = {}
        model_predictions: dict[str, np.ndarray] = {}
        for model_key in selected:
            pred = forecasts.get(model_key)
            if isinstance(pred, np.ndarray) and len(pred) == len(actual):
                model_predictions[model_key] = pred
                model_residuals[model_key] = actual - pred

        if not model_residuals:
            empty = _empty_figure("No residual diagnostics available for the selected model(s).")
            return (
                table,
                empty,
                empty,
                empty,
                empty,
                _empty_figure("Select XGBoost to view SHAP feature importance."),
            )

        # uirevision keyed on region + sorted model selection so zoom/legend
        # state survives data refresh but resets when the user picks new models.
        uirev = f"{region}:{','.join(sorted(selected))}"

        fig_resid_time = go.Figure()
        for model_key, residuals in model_residuals.items():
            fig_resid_time.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=residuals,
                    mode="lines",
                    name=model_labels.get(model_key, model_key.title()),
                    line=dict(color=model_colors.get(model_key, COLORS["actual"]), width=1),
                )
            )
        fig_resid_time.add_hline(y=0, line=dict(color="#F7FAFC", dash="dash", width=0.5))
        fig_resid_time.update_layout(**_layout(uirevision=uirev, yaxis_title="Residual (MW)"))

        fig_resid_hist = go.Figure()
        for model_key, residuals in model_residuals.items():
            fig_resid_hist.add_trace(
                go.Histogram(
                    x=residuals,
                    nbinsx=50,
                    name=model_labels.get(model_key, model_key.title()),
                    marker_color=model_colors.get(model_key, COLORS["actual"]),
                    opacity=0.6,
                )
            )
        fig_resid_hist.update_layout(
            **_layout(
                uirevision=uirev,
                barmode="overlay",
                xaxis_title="Residual (MW)",
                yaxis_title="Count",
            )
        )

        fig_resid_pred = go.Figure()
        for model_key, residuals in model_residuals.items():
            preds = model_predictions[model_key]
            fig_resid_pred.add_trace(
                go.Scatter(
                    x=preds,
                    y=residuals,
                    mode="markers",
                    name=model_labels.get(model_key, model_key.title()),
                    marker=dict(
                        size=3,
                        color=model_colors.get(model_key, COLORS["actual"]),
                        opacity=0.35,
                    ),
                )
            )
        fig_resid_pred.add_hline(y=0, line=dict(color="#F7FAFC", dash="dash", width=0.5))
        fig_resid_pred.update_layout(
            **_layout(uirevision=uirev, xaxis_title="Predicted (MW)", yaxis_title="Residual (MW)")
        )

        hours_of_day = timestamps.dt.hour
        fig_heatmap = go.Figure()
        for model_key, residuals in model_residuals.items():
            error_by_hour = pd.DataFrame({"hour": hours_of_day, "abs_error": np.abs(residuals)})
            hourly_error = error_by_hour.groupby("hour")["abs_error"].mean()
            fig_heatmap.add_trace(
                go.Bar(
                    x=hourly_error.index,
                    y=hourly_error.values,
                    name=model_labels.get(model_key, model_key.title()),
                    marker_color=model_colors.get(model_key, COLORS["actual"]),
                    opacity=0.85,
                )
            )
        fig_heatmap.update_layout(**_layout(uirevision=uirev, barmode="group"))
        fig_heatmap.update_layout(
            **_layout(uirevision=uirev, xaxis_title="Hour of Day", yaxis_title="Mean |Error| (MW)")
        )

        if "xgboost" in selected:
            feature_names, importance_vals = _get_feature_importance(region)
            fig_shap = go.Figure(
                go.Bar(
                    x=importance_vals[::-1],
                    y=feature_names[::-1],
                    orientation="h",
                    marker_color=COLORS["xgboost"],
                )
            )
            fig_shap.update_layout(**_layout(uirevision=uirev, xaxis_title="Feature Importance"))
        else:
            fig_shap = _empty_figure("SHAP is available only for XGBoost. Select XGBoost above.")

        return table, fig_resid_time, fig_resid_hist, fig_resid_pred, fig_heatmap, fig_shap

    # ── 7. TAB 4: GENERATION & NET LOAD ──────────────────────

    @app.callback(
        [
            Output("tab4-net-load-chart", "figure"),
            Output("tab4-gen-mix-chart", "figure"),
            Output("tab4-renewable-pct", "children"),
            Output("tab4-peak-ramp", "children"),
            Output("tab4-min-net-load", "children"),
            Output("tab4-curtailment-hours", "children"),
            Output("tab4-insight-card", "children"),
        ],
        [
            Input("region-selector", "value"),
            Input("dashboard-tabs", "active_tab"),
            Input("persona-selector", "value"),
            Input("gen-date-range", "value"),
        ],
        [State("demand-store", "data")],
        prevent_initial_call=True,
    )
    def update_generation_tab(region, active_tab, persona_id, date_range_hours, demand_json):
        """Update Generation & Net Load tab with real EIA data."""
        empty = _empty_figure("Select a region to view generation data")
        empty_insight = html.Div()
        defaults = (empty, empty, "No data", "No data", "No data", "No data", empty_insight)

        # Active tab guard
        if active_tab != "tab-generation":
            return [no_update] * 7

        if not region:
            return defaults

        # Parse date range (hours). Default to 168 (7 days).
        try:
            range_hours = int(date_range_hours) if date_range_hours else 168
        except (TypeError, ValueError):
            range_hours = 168

        log.info("generation_tab_start", region=region, range_hours=range_hours)

        # ── v2 Redis fast path ──────────────────────────────
        redis_result = _generation_tab_from_redis(region, range_hours, demand_json, persona_id)
        if redis_result is not None:
            return redis_result

        # ── v1 compute fallback ─────────────────────────────
        # Fetch generation data (memory → SQLite → API → demo)
        gen_df = _fetch_generation_cached(region)
        if gen_df is None or gen_df.empty:
            return defaults

        # ── Parse demand data for net load calculation ──
        demand_df = None
        if demand_json:
            demand_df = pd.read_json(io.StringIO(demand_json))
            demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])

        # ── Filter by selected time range ──
        gen_df["timestamp"] = pd.to_datetime(gen_df["timestamp"])
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=range_hours)
        # Strip timezone for comparison if gen_df timestamps are tz-naive
        if gen_df["timestamp"].dt.tz is None:
            cutoff = cutoff.tz_localize(None)
        gen_df = gen_df[gen_df["timestamp"] >= cutoff]
        if gen_df.empty:
            return defaults

        if demand_df is not None and not demand_df.empty:
            demand_cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=range_hours)
            if demand_df["timestamp"].dt.tz is None:
                demand_cutoff = demand_cutoff.tz_localize(None)
            demand_df = demand_df[demand_df["timestamp"] >= demand_cutoff]

        # ── Pivot generation by fuel type ──
        pivot = gen_df.pivot_table(
            index="timestamp",
            columns="fuel_type",
            values="generation_mw",
            aggfunc="sum",
        ).fillna(0)

        total_gen = pivot.sum(axis=1)

        # ── Net Load = Demand - Wind - Solar ──
        wind_gen = pivot.get("wind", pd.Series(0, index=pivot.index))
        solar_gen = pivot.get("solar", pd.Series(0, index=pivot.index))

        if demand_df is not None and not demand_df.empty and "demand_mw" in demand_df.columns:
            demand_aligned = demand_df.set_index("timestamp")["demand_mw"]
            common_idx = pivot.index.intersection(demand_aligned.index)
            if len(common_idx) > 24:
                demand_series = demand_aligned.loc[common_idx]
                pivot_aligned = pivot.loc[common_idx]
                wind_aligned = pivot_aligned.get("wind", pd.Series(0, index=common_idx))
                solar_aligned = pivot_aligned.get("solar", pd.Series(0, index=common_idx))
                net_load = demand_series - wind_aligned - solar_aligned
            else:
                # Insufficient overlap; approximate demand as total gen
                demand_series = total_gen
                net_load = total_gen - wind_gen - solar_gen
                common_idx = pivot.index
        else:
            # No demand data: approximate demand as total generation
            demand_series = total_gen
            net_load = total_gen - wind_gen - solar_gen
            common_idx = pivot.index

        # ── KPI 1: Renewable penetration % ──
        ren_cols = [c for c in ["wind", "solar", "hydro"] if c in pivot.columns]
        if ren_cols and total_gen.mean() > 0:
            renewable_gen = pivot[ren_cols].sum(axis=1)
            renewable_pct = float((renewable_gen / total_gen * 100).mean())
        else:
            renewable_pct = 0.0

        # ── KPI 2: Peak net load ramp (MW/hr) ──
        net_load_diff = net_load.diff()
        peak_ramp = float(net_load_diff.max()) if not net_load_diff.isna().all() else 0.0

        # ── KPI 3: Min net load (duck curve belly) ──
        min_net_load = float(net_load.min())

        # ── KPI 4: Curtailment risk hours (net load < 20% of peak) ──
        peak_net_load = float(net_load.max())
        curtailment_hours = int((net_load < peak_net_load * 0.2).sum()) if peak_net_load > 0 else 0

        # ── Hero Chart: Demand vs Net Load ──
        fig_hero = go.Figure()

        fig_hero.add_trace(
            go.Scatter(
                x=common_idx,
                y=demand_series.values if hasattr(demand_series, "values") else demand_series,
                mode="lines",
                name="Total Demand",
                line=dict(color=COLORS["actual"], width=2),
            )
        )

        fig_hero.add_trace(
            go.Scatter(
                x=common_idx,
                y=net_load.values,
                mode="lines",
                name="Net Load",
                line=dict(color=COLORS["ensemble"], width=2.5),
            )
        )

        # Shaded area between demand and net load (= renewable contribution)
        fig_hero.add_trace(
            go.Scatter(
                x=common_idx,
                y=demand_series.values if hasattr(demand_series, "values") else demand_series,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig_hero.add_trace(
            go.Scatter(
                x=common_idx,
                y=net_load.values,
                mode="lines",
                name="Renewable Contribution",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(45,226,196,0.15)",
            )
        )

        fig_hero.update_layout(
            **_layout(
                uirevision=f"{region}:{range_hours}",
                yaxis_title="MW",
                hovermode="x unified",
                title=f"Demand vs Net Load \u2014 {region}",
            )
        )

        # ── Supporting Chart: Generation Mix Stacked Area ──
        fig_mix = go.Figure()
        fuel_order = ["nuclear", "coal", "gas", "hydro", "wind", "solar", "other"]
        for fuel in fuel_order:
            if fuel in pivot.columns:
                fig_mix.add_trace(
                    go.Scatter(
                        x=pivot.index,
                        y=pivot[fuel],
                        mode="lines",
                        name=fuel.title(),
                        stackgroup="one",
                        line=dict(width=0),
                        fillcolor=COLORS.get(fuel, "#95a5a6"),
                    )
                )
        fig_mix.update_layout(
            **_layout(
                uirevision=f"{region}:{range_hours}",
                yaxis_title="Generation (MW)",
                hovermode="x unified",
                title=f"Generation Mix \u2014 {region}",
            )
        )

        # ── Insight Card ──
        from components.insights import build_insight_card, generate_tab4_insights

        persona = persona_id or "grid_ops"
        insights = generate_tab4_insights(
            persona_id=persona,
            region=region,
            net_load=net_load,
            demand=demand_series,
            renewable_pct=renewable_pct,
            pivot=pivot.loc[common_idx] if len(common_idx) > 0 else pivot,
            timestamps=pd.DatetimeIndex(common_idx),
        )
        insight_card = build_insight_card(insights, persona, "tab-generation")

        return (
            fig_hero,
            fig_mix,
            f"{renewable_pct:.1f}%",
            f"{peak_ramp:,.0f} MW/hr",
            f"{min_net_load:,.0f} MW",
            str(curtailment_hours),
            insight_card,
        )

    # ── 8. TAB 5: ALERTS ─────────────────────────────────────

    @app.callback(
        [
            Output("tab5-alerts-list", "children"),
            Output("tab5-stress-score", "children"),
            Output("tab5-stress-label", "children"),
            Output("tab5-stress-breakdown", "children"),
            Output("tab5-anomaly-chart", "figure"),
            Output("tab5-temp-exceedance", "figure"),
            Output("tab5-timeline", "figure"),
            Output("tab5-weather-context", "children"),
        ],
        [
            Input("region-selector", "value"),
            Input("demand-store", "data"),
            Input("weather-store", "data"),
            Input("dashboard-tabs", "active_tab"),
        ],
        prevent_initial_call=True,
    )
    def update_alerts_tab(region, demand_json, weather_json, active_tab):
        """Update Tab 5 alerts and stress indicators."""
        if active_tab != "tab-alerts":
            return [no_update] * 8
        empty = _empty_figure("Loading...")

        # ── v2 Redis fast path ──────────────────────────────
        if region:
            redis_result = _alerts_tab_from_redis(region)
            if redis_result is not None:
                return redis_result

        # ── v1 compute fallback ─────────────────────────────
        from data.demo_data import generate_demo_alerts

        alerts = generate_demo_alerts(region)

        alert_cards = []
        if alerts:
            for a in alerts:
                alert_cards.append(
                    build_alert_card(
                        event=a["event"],
                        headline=a["headline"],
                        severity=a["severity"],
                        expires=a.get("expires", "")[:16] if a.get("expires") else None,
                    )
                )
        else:
            alert_cards = [
                html.P(
                    "No active alerts",
                    style={"color": "#A8B3C7", "textAlign": "center", "padding": "20px"},
                )
            ]

        n_crit = sum(1 for a in alerts if a["severity"] == "critical")
        n_warn = sum(1 for a in alerts if a["severity"] == "warning")
        n_info = sum(1 for a in alerts if a["severity"] == "info")
        stress = min(100, n_crit * 30 + n_warn * 15 + 20)
        stress_label = "Normal" if stress < 30 else ("Elevated" if stress < 60 else "Critical")
        stress_color = "positive" if stress < 30 else ("negative" if stress >= 60 else "neutral")

        breakdown_items = []
        if n_crit:
            breakdown_items.append(
                html.Div(
                    f"🔴 Critical: {n_crit}", style={"fontSize": "0.75rem", "color": "#FF5C7A"}
                )
            )
        if n_warn:
            breakdown_items.append(
                html.Div(f"🟡 Warning: {n_warn}", style={"fontSize": "0.75rem", "color": "#FFB84D"})
            )
        if n_info:
            breakdown_items.append(
                html.Div(f"🔵 Info: {n_info}", style={"fontSize": "0.75rem", "color": "#56B4E9"})
            )
        if not alerts:
            breakdown_items.append(
                html.Div("No active alerts", style={"fontSize": "0.75rem", "color": "#A8B3C7"})
            )
        breakdown = html.Div(breakdown_items)

        if demand_json:
            demand_df = pd.read_json(io.StringIO(demand_json))
            demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])
            recent = demand_df.tail(168)
            rolling_mean = recent["demand_mw"].rolling(24).mean()
            rolling_std = recent["demand_mw"].rolling(24).std()
            upper = rolling_mean + 2 * rolling_std
            lower = rolling_mean - 2 * rolling_std
            anomalies = recent[recent["demand_mw"] > upper]

            fig_anomaly = go.Figure()
            fig_anomaly.add_trace(
                go.Scatter(
                    x=recent["timestamp"],
                    y=recent["demand_mw"],
                    name="Demand",
                    line=dict(color=COLORS["actual"]),
                )
            )
            fig_anomaly.add_trace(
                go.Scatter(
                    x=recent["timestamp"],
                    y=upper,
                    name="Upper (2σ)",
                    line=dict(color="#FF5C7A", dash="dash", width=1),
                )
            )
            fig_anomaly.add_trace(
                go.Scatter(
                    x=recent["timestamp"],
                    y=lower,
                    name="Lower (2σ)",
                    line=dict(color="#FF5C7A", dash="dash", width=1),
                )
            )
            if not anomalies.empty:
                fig_anomaly.add_trace(
                    go.Scatter(
                        x=anomalies["timestamp"],
                        y=anomalies["demand_mw"],
                        mode="markers",
                        name="Anomaly",
                        marker=dict(color="#FF5C7A", size=8, symbol="diamond"),
                    )
                )
            fig_anomaly.update_layout(**_layout(uirevision=region, yaxis_title="MW"))
        else:
            fig_anomaly = empty

        if weather_json:
            weather_df = pd.read_json(io.StringIO(weather_json))
            weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
            recent_w = weather_df.tail(168)
            fig_temp = go.Figure()
            fig_temp.add_trace(
                go.Scatter(
                    x=recent_w["timestamp"],
                    y=recent_w["temperature_2m"],
                    name="Temperature",
                    line=dict(color=COLORS["temperature"]),
                )
            )
            for t in [95, 100, 105]:
                fig_temp.add_hline(
                    y=t,
                    line=dict(color="#FF5C7A", dash="dot", width=1),
                    annotation_text=f"{t}°F",
                    annotation_position="right",
                )
            fig_temp.update_layout(**_layout(uirevision=region, yaxis_title="°F"))
        else:
            fig_temp = empty

        events = [
            ("2021-02-15", "Winter Storm Uri", "ERCOT", 95),
            ("2022-09-06", "CA Heat Wave", "CAISO", 80),
            ("2023-07-20", "Heat Dome", "CAISO", 85),
            ("2024-04-08", "Solar Eclipse", "PJM", 40),
        ]
        fig_timeline = go.Figure()
        for date, name, reg, sev in events:
            color = COLORS["ensemble"] if reg == region else "#A8B3C7"
            fig_timeline.add_trace(
                go.Scatter(
                    x=[date],
                    y=[sev],
                    mode="markers+text",
                    text=[name],
                    textposition="top center",
                    marker=dict(size=12, color=color),
                    showlegend=False,
                )
            )
        fig_timeline.update_layout(
            **_layout(
                uirevision=region,
                xaxis_title="Date",
                yaxis_title="Severity Score",
                yaxis_range=[0, 100],
            )
        )

        # Build weather context from latest reading
        weather_context = html.Div()
        if weather_json:
            try:
                w_df = pd.read_json(io.StringIO(weather_json))
                if not w_df.empty:
                    weather_context = _build_weather_context(w_df.iloc[-1])
            except Exception:
                log.warning("weather_context_build_failed")

        return (
            alert_cards,
            str(stress),
            html.Span(stress_label, className=f"kpi-delta {stress_color}"),
            breakdown,
            fig_anomaly,
            fig_temp,
            fig_timeline,
            weather_context,
        )

    # ── 9. TAB 6: SCENARIO SIMULATOR ─────────────────────────

    @app.callback(
        [
            Output("sim-forecast-chart", "figure"),
            Output("sim-demand-delta", "children"),
            Output("sim-demand-delta-pct", "children"),
            Output("sim-price-impact", "children"),
            Output("sim-price-delta", "children"),
            Output("sim-reserve-margin", "children"),
            Output("sim-reserve-status", "children"),
            Output("sim-renewable-impact", "children"),
            Output("sim-renewable-detail", "children"),
            Output("sim-price-chart", "figure"),
            Output("sim-renewable-chart", "figure"),
        ],
        [
            Input("sim-run-btn", "n_clicks"),
            Input({"type": "preset-btn", "index": ALL}, "n_clicks"),
            # Auto-fire the default scenario as soon as demand-store hydrates
            # or the user lands on the tab, so the KPIs never sit on their
            # initial placeholder text.
            Input("demand-store", "data"),
            Input("dashboard-tabs", "active_tab"),
        ],
        [
            State("sim-temp", "value"),
            State("sim-wind", "value"),
            State("sim-cloud", "value"),
            State("sim-humidity", "value"),
            State("sim-solar", "value"),
            State("sim-duration", "value"),
            State("region-selector", "value"),
        ],
    )
    def run_scenario(
        run_clicks,
        preset_clicks,
        demand_json,
        active_tab,
        temp,
        wind,
        cloud,
        humidity,
        solar_irr,
        duration,
        region,
    ):
        """Run scenario simulation and update impact dashboard.

        Fires on button clicks, preset selection, and whenever demand data
        arrives or the simulator tab becomes active. The active-tab guard
        avoids wasted work when the user is on another tab. The warming
        guard handles the REQUIRE_REDIS path where load_data() serializes
        an empty DataFrame while the scoring job is still populating Redis.
        """
        if active_tab != "tab-simulator":
            return [no_update] * 11

        # Parse defensively — warming-state JSON is a valid-but-empty frame,
        # and the first page render may pass demand_json=None.
        try:
            demand_df = pd.read_json(io.StringIO(demand_json)) if demand_json else pd.DataFrame()
        except ValueError:
            demand_df = pd.DataFrame()

        if demand_df.empty or "demand_mw" not in demand_df.columns:
            warming_fig = _empty_figure("Data warming up — refresh shortly.")
            warming = "Warming up"
            return (
                warming_fig,
                warming,
                "",
                warming,
                "",
                warming,
                "",
                warming,
                "",
                warming_fig,
                warming_fig,
            )

        triggered = ctx.triggered_id
        if isinstance(triggered, dict) and triggered.get("type") == "preset-btn":
            from simulation.presets import get_preset

            preset = get_preset(triggered["index"])
            w = preset["weather"]
            temp = w.get("temperature_2m", temp)
            wind = w.get("wind_speed_80m", wind)
            cloud = w.get("cloud_cover", cloud)
            humidity = w.get("relative_humidity_2m", humidity)
            solar_irr = w.get("shortwave_radiation", solar_irr)
            region = preset.get("region", region)

        demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])
        baseline = demand_df["demand_mw"].tail(duration).values
        capacity = REGION_CAPACITY_MW.get(region, 50000)

        # Baseline weather corresponds to the default slider values so the
        # scenario collapses to the baseline when sliders are unmoved.
        baseline_temp = 75.0  # °F
        baseline_humidity = 60.0  # %
        baseline_cloud = 50.0  # %

        cdd = max(0.0, temp - 65.0)
        hdd = max(0.0, 65.0 - temp)
        cdd_base = max(0.0, baseline_temp - 65.0)
        hdd_base = max(0.0, 65.0 - baseline_temp)

        # Standard utility load-temperature elasticities: cooling load grows
        # ~0.5%/°F above 65°F and heating load ~0.3%/°F below it. The
        # previous `/65` divisor flattened this to ~0.03%/°F and made the
        # slider visually inert.
        temp_effect = (cdd - cdd_base) * 0.005 + (hdd - hdd_base) * 0.003

        # Humidity amplifies cooling load (heat-index effect); only on hot
        # days. Cloud cover dampens cooling load via reduced solar heat
        # gain on buildings. Both are zero when temperature is at/below 65.
        humidity_effect = cdd * 0.0015 * (humidity - baseline_humidity) / 100.0
        cloud_effect = -cdd * 0.0010 * (cloud - baseline_cloud) / 100.0

        temp_factor = 1.0 + temp_effect + humidity_effect + cloud_effect

        seed = stable_int_seed(("scenario_simulation", region, temp, wind, cloud, humidity))
        rng = np.random.RandomState(seed)
        scenario = baseline * temp_factor + rng.normal(0, capacity * 0.005, len(baseline))
        scenario = np.maximum(scenario, 0)

        delta = scenario - baseline
        mean_delta = np.mean(delta)

        from data.feature_engineering import compute_solar_capacity_factor, compute_wind_power

        wind_power = float(compute_wind_power(pd.Series([wind])).iloc[0])
        solar_cf = float(compute_solar_capacity_factor(pd.Series([solar_irr])).iloc[0])

        # Approximate nameplate renewable shares of total capacity (US avg
        # across the 8 modeled BAs). Per-region splits vary but these are
        # representative enough for the heuristic path; price and reserve
        # then respond to net load (= demand − renewable gen), so solar
        # and wind sliders actually move the economics.
        wind_share = 0.15
        solar_share = 0.10
        renewable_gen = wind_power * wind_share * capacity + solar_cf * solar_share * capacity
        net_load = np.maximum(scenario - renewable_gen, 0)

        from models.pricing import estimate_price_impact

        base_price = estimate_price_impact(np.mean(baseline), capacity)
        scen_price = estimate_price_impact(np.mean(net_load), capacity)
        reserve = (capacity - np.max(net_load)) / capacity * 100

        timestamps = demand_df["timestamp"].tail(duration)
        fig_forecast = go.Figure()
        fig_forecast.add_trace(
            go.Scatter(
                x=timestamps,
                y=baseline,
                name="Baseline",
                line=dict(color=COLORS["actual"], width=2),
            )
        )
        fig_forecast.add_trace(
            go.Scatter(
                x=timestamps,
                y=scenario,
                name="Scenario",
                line=dict(color=COLORS["ensemble"], width=2.5),
            )
        )
        fig_forecast.add_trace(
            go.Scatter(
                x=timestamps,
                y=delta,
                name="Delta",
                line=dict(color=COLORS["temperature"], width=1, dash="dot"),
                yaxis="y2",
            )
        )
        fig_forecast.update_layout(
            **_layout(
                uirevision=region,
                yaxis_title="Demand (MW)",
                yaxis2=dict(title="Delta (MW)", overlaying="y", side="right", showgrid=False),
                hovermode="x unified",
            )
        )

        utilizations = np.linspace(0.5, 1.1, 100)
        prices = estimate_price_impact(utilizations * capacity, capacity)
        current_util = np.mean(net_load) / capacity
        fig_price = go.Figure()
        fig_price.add_trace(
            go.Scatter(
                x=utilizations * 100,
                y=prices,
                name="Price Curve",
                line=dict(color=COLORS["ensemble"]),
            )
        )
        fig_price.add_vline(
            x=current_util * 100,
            line=dict(color="#F7FAFC", dash="dash"),
            annotation_text=f"Scenario: {current_util * 100:.0f}%",
        )
        fig_price.update_layout(
            **_layout(uirevision=region, xaxis_title="Utilization %", yaxis_title="$/MWh")
        )

        fig_renewable = go.Figure()
        fig_renewable.add_trace(
            go.Bar(
                x=["Wind Power", "Solar CF"],
                y=[wind_power * 100, solar_cf * 100],
                marker_color=[COLORS["wind"], COLORS["solar"]],
                text=[f"{wind_power * 100:.0f}%", f"{solar_cf * 100:.0f}%"],
                textposition="auto",
            )
        )
        fig_renewable.update_layout(
            **_layout(uirevision=region, yaxis_title="Capacity Factor %", yaxis_range=[0, 110])
        )

        delta_dir = "positive" if mean_delta < 0 else "negative"
        reserve_status = "positive" if reserve > 15 else ("negative" if reserve < 5 else "neutral")

        return (
            fig_forecast,
            f"{mean_delta:+,.0f} MW",
            html.Span(
                f"{mean_delta / np.mean(baseline) * 100:+.1f}% vs baseline",
                className=f"kpi-delta {delta_dir}",
            ),
            f"${scen_price:.0f}/MWh",
            html.Span(
                f"{'↑' if scen_price > base_price else '↓'}{abs(scen_price - base_price):.0f} vs base",
                className=f"kpi-delta {'negative' if scen_price > base_price else 'positive'}",
            ),
            f"{reserve:.1f}%",
            html.Span(
                "Adequate" if reserve > 15 else ("Low" if reserve > 5 else "CRITICAL"),
                className=f"kpi-delta {reserve_status}",
            ),
            f"Wind: {wind_power * 100:.0f}%",
            html.Span(f"Solar CF: {solar_cf * 100:.0f}%", className="kpi-delta neutral"),
            fig_price,
            fig_renewable,
        )

    # ── SLIDER DISPLAY UPDATES ────────────────────────────────

    for slider_id, unit in [
        ("sim-temp", "°F"),
        ("sim-wind", "mph"),
        ("sim-cloud", "%"),
        ("sim-humidity", "%"),
        ("sim-solar", "W/m²"),
    ]:

        @app.callback(
            Output(f"{slider_id}-display", "children"),
            Input(slider_id, "value"),
        )
        def update_slider_display(val, u=unit):
            return f"{val}{u}"

    # ── PRESET → SLIDER SYNC ─────────────────────────────────

    @app.callback(
        [
            Output("sim-temp", "value"),
            Output("sim-wind", "value"),
            Output("sim-cloud", "value"),
            Output("sim-humidity", "value"),
            Output("sim-solar", "value"),
        ],
        Input({"type": "preset-btn", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def apply_preset_to_sliders(clicks):
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return no_update, no_update, no_update, no_update, no_update
        from simulation.presets import get_preset

        preset = get_preset(triggered["index"])
        w = preset["weather"]
        return (
            w.get("temperature_2m", 75),
            w.get("wind_speed_80m", 15),
            w.get("cloud_cover", 50),
            w.get("relative_humidity_2m", 60),
            w.get("shortwave_radiation", 500),
        )

    # ── NEXD-12: RESET SCENARIO SLIDERS ───────────────────────

    @app.callback(
        [
            Output("sim-temp", "value", allow_duplicate=True),
            Output("sim-wind", "value", allow_duplicate=True),
            Output("sim-cloud", "value", allow_duplicate=True),
            Output("sim-humidity", "value", allow_duplicate=True),
            Output("sim-solar", "value", allow_duplicate=True),
            Output("sim-duration", "value", allow_duplicate=True),
        ],
        Input("sim-reset-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_sim_sliders(n_clicks):
        """NEXD-12: Reset scenario sliders to defaults."""
        if not n_clicks:
            return [no_update] * 6

        from data.user_prefs import SIM_DURATION_DEFAULT, SIM_SLIDER_DEFAULTS, SIM_SLIDERS

        return [SIM_SLIDER_DEFAULTS[sid] for sid in SIM_SLIDERS] + [SIM_DURATION_DEFAULT]

    # ── SPRINT 4: G2 — API FALLBACK BANNER ────────────────────

    @app.callback(
        Output("fallback-banner", "children"),
        Input("data-freshness-store", "data"),
        prevent_initial_call=True,
    )
    def update_fallback_banner(freshness_json):
        """G2: Show warning banner when data sources are serving stale/fallback data."""
        import json

        if not freshness_json:
            return no_update

        freshness = json.loads(freshness_json)
        warnings = []
        icons = {"stale": "⚠️", "error": "🔴", "demo": "🧪"}

        for source in ("demand", "weather", "alerts"):
            status = freshness.get(source, "fresh")
            if status == "stale":
                warnings.append(
                    f"{icons['stale']} {source.title()}: serving cached data (API unavailable)"
                )
            elif status == "error":
                warnings.append(
                    f"{icons['error']} {source.title()}: data load failed — using fallback"
                )
            elif status == "demo":
                warnings.append(
                    f"{icons['demo']} {source.title()}: demo data (no API key configured)"
                )

        if not warnings:
            return html.Div()

        return dbc.Alert(
            [html.Strong("Data Source Status"), html.Br()]
            + [html.Span(w, style={"display": "block", "fontSize": "0.85rem"}) for w in warnings],
            color="warning" if "error" not in freshness_json else "danger",
            dismissable=True,
            className="mb-2 mt-1",
            style={"fontSize": "0.85rem"},
        )

    # ── SPRINT 4: G2 — HEADER FRESHNESS BADGE ───────────────────

    @app.callback(
        Output("header-freshness", "children"),
        Input("data-freshness-store", "data"),
    )
    def update_header_freshness(freshness_json):
        """G2: Compact freshness badge in the header bar."""
        import json
        from datetime import datetime

        if not freshness_json:
            return html.Span("⏳ Loading…", style={"color": "#A8B3C7", "fontSize": "0.75rem"})

        freshness = json.loads(freshness_json)
        statuses = [freshness.get(s, "fresh") for s in ("demand", "weather", "alerts")]

        if all(s == "fresh" for s in statuses):
            color, icon, label = "#2BD67B", "🟢", "Live"
        elif all(s == "demo" for s in statuses):
            color, icon, label = "#A8B3C7", "🧪", "Demo"
        elif any(s == "error" for s in statuses):
            color, icon, label = "#FF5C7A", "🔴", "Degraded"
        else:
            color, icon, label = "#FFB84D", "🟡", "Partial"

        # Show latest data timestamp (when the actual data is from)
        latest_data = freshness.get("latest_data", "")
        data_time_text = ""
        if latest_data:
            try:
                # Parse the timestamp string
                latest_dt = datetime.fromisoformat(latest_data.replace("Z", "+00:00"))
                data_time_text = latest_dt.strftime("%b %d %H:%M UTC")
            except (ValueError, TypeError):
                data_time_text = ""

        return html.Span(
            [
                html.Span(f"{icon} {label}", style={"marginRight": "8px"}),
                html.Span(
                    f"Data through: {data_time_text}" if data_time_text else "",
                    style={"color": "#A8B3C7", "fontSize": "0.7rem"},
                ),
            ],
            style={"color": color, "fontSize": "0.75rem", "fontWeight": "500"},
        )

    # ── SPRINT 4: C2 — SCENARIO BOOKMARKS (URL STATE) ─────────

    @app.callback(
        [
            Output("region-selector", "value", allow_duplicate=True),
            Output("persona-selector", "value", allow_duplicate=True),
            Output("dashboard-tabs", "active_tab", allow_duplicate=True),
            # NEXD-12: tracked filters
            Output("tab1-timerange", "value", allow_duplicate=True),
            Output("outlook-horizon", "value", allow_duplicate=True),
            Output("outlook-model", "value", allow_duplicate=True),
            Output("backtest-horizon", "value", allow_duplicate=True),
            Output("gen-date-range", "value", allow_duplicate=True),
            Output("tab3-model-selector", "value", allow_duplicate=True),
            Output("sim-duration", "value", allow_duplicate=True),
            # NEXD-12: sim sliders
            Output("sim-temp", "value", allow_duplicate=True),
            Output("sim-wind", "value", allow_duplicate=True),
            Output("sim-cloud", "value", allow_duplicate=True),
            Output("sim-humidity", "value", allow_duplicate=True),
            Output("sim-solar", "value", allow_duplicate=True),
        ],
        Input("url", "search"),
        prevent_initial_call=True,
    )
    def restore_bookmark(search):
        """C2+NEXD-12: Restore full dashboard state from URL query parameters.

        Supports core params (region, persona, tab), tracked filters (f.*),
        and scenario slider values (s.*).  Old URLs without f.*/s.* still work.
        """
        n_outputs = 15  # 3 core + 7 filters + 5 sliders
        if not search:
            return [no_update] * n_outputs

        from data.user_prefs import (
            SIM_SLIDERS,
            TRACKED_FILTERS,
            deserialize_bookmark_params,
        )

        state = deserialize_bookmark_params(search)
        if not state:
            return [no_update] * n_outputs

        outputs = [
            state.get("region", no_update),
            state.get("persona", no_update),
            state.get("tab", no_update),
        ]

        filters = state.get("filters", {})
        for fid in TRACKED_FILTERS:
            outputs.append(filters.get(fid, no_update))

        sim_sliders = state.get("sim_sliders", {})
        for sid in SIM_SLIDERS:
            outputs.append(sim_sliders.get(sid, no_update))

        return outputs

    @app.callback(
        [Output("url", "search"), Output("bookmark-toast", "children")],
        Input("bookmark-btn", "n_clicks"),
        [
            State("region-selector", "value"),
            State("persona-selector", "value"),
            State("dashboard-tabs", "active_tab"),
            # NEXD-12: tracked filters
            State("tab1-timerange", "value"),
            State("outlook-horizon", "value"),
            State("outlook-model", "value"),
            State("backtest-horizon", "value"),
            State("gen-date-range", "value"),
            State("tab3-model-selector", "value"),
            State("sim-duration", "value"),
            # NEXD-12: sim sliders
            State("sim-temp", "value"),
            State("sim-wind", "value"),
            State("sim-cloud", "value"),
            State("sim-humidity", "value"),
            State("sim-solar", "value"),
        ],
        prevent_initial_call=True,
    )
    def create_bookmark(n_clicks, region, persona, tab, *state_values):
        """C2+NEXD-12: Serialize full dashboard state into a shareable URL."""
        if not n_clicks:
            return no_update, no_update

        from data.user_prefs import (
            SIM_SLIDERS,
            TRACKED_FILTERS,
            serialize_bookmark_params,
        )

        filter_values = state_values[: len(TRACKED_FILTERS)]
        slider_values = state_values[len(TRACKED_FILTERS) :]

        filters: dict = {}
        for fid, val in zip(TRACKED_FILTERS, filter_values, strict=False):
            if val is not None:
                filters[fid] = val

        sim_sliders: dict = {}
        for sid, val in zip(SIM_SLIDERS, slider_values, strict=False):
            if val is not None:
                sim_sliders[sid] = val

        search = serialize_bookmark_params(region, persona, tab, filters, sim_sliders)

        toast = dbc.Toast(
            "Bookmark saved! URL updated — copy it to share this view.",
            header="Bookmark Created",
            dismissable=True,
            duration=4000,
            is_open=True,
            style={"backgroundColor": "#11182D", "color": "#DDE6F2", "border": "1px solid #263556"},
        )
        return search, toast

    # ── NEXD-9: SMART DEFAULTS — PERSIST & RESTORE FILTERS ───

    @app.callback(
        [
            Output("region-selector", "value", allow_duplicate=True),
            Output("persona-selector", "value", allow_duplicate=True),
            Output("dashboard-tabs", "active_tab", allow_duplicate=True),
            Output("tab1-timerange", "value", allow_duplicate=True),
            Output("outlook-horizon", "value", allow_duplicate=True),
            Output("outlook-model", "value", allow_duplicate=True),
            Output("backtest-horizon", "value", allow_duplicate=True),
            Output("gen-date-range", "value", allow_duplicate=True),
            Output("tab3-model-selector", "value", allow_duplicate=True),
            Output("sim-duration", "value", allow_duplicate=True),
        ],
        Input("url", "pathname"),
        [
            State("user-prefs-store", "data"),
            State("url", "search"),
        ],
        prevent_initial_call="initial_duplicate",
    )
    def restore_user_prefs(pathname, prefs_data, search):
        """NEXD-9: Restore saved filter state from localStorage on page load."""
        from config import feature_enabled

        n_outputs = 10
        if not feature_enabled("smart_defaults"):
            return [no_update] * n_outputs

        # If URL has bookmark params, let restore_bookmark handle it
        if search and "=" in search:
            return [no_update] * n_outputs

        if not prefs_data or not isinstance(prefs_data, dict):
            return [no_update] * n_outputs

        from data.user_prefs import TRACKED_FILTERS, validate_prefs

        prefs = validate_prefs(prefs_data)

        # Build output list: region, persona, tab, then each tracked filter
        outputs = [
            prefs.region or no_update,
            prefs.persona or no_update,
            prefs.tab or no_update,
        ]

        for fid in TRACKED_FILTERS:
            val = prefs.filters.get(fid)
            outputs.append(val if val is not None else no_update)

        return outputs

    @app.callback(
        Output("user-prefs-store", "data"),
        [
            Input("region-selector", "value"),
            Input("persona-selector", "value"),
            Input("dashboard-tabs", "active_tab"),
            Input("tab1-timerange", "value"),
            Input("outlook-horizon", "value"),
            Input("outlook-model", "value"),
            Input("backtest-horizon", "value"),
            Input("gen-date-range", "value"),
            Input("tab3-model-selector", "value"),
            Input("sim-duration", "value"),
        ],
        prevent_initial_call=True,
    )
    def save_user_prefs(region, persona, tab, *filter_values):
        """NEXD-9: Persist current filter state to localStorage."""
        from config import feature_enabled

        if not feature_enabled("smart_defaults"):
            return no_update

        from datetime import UTC, datetime

        from data.user_prefs import TRACKED_FILTERS

        filters = {}
        for fid, val in zip(TRACKED_FILTERS, filter_values, strict=False):
            if val is not None:
                filters[fid] = val

        return {
            "region": region or "FPL",
            "persona": persona or "grid_ops",
            "tab": tab or "tab-overview",
            "filters": filters,
            "updated_at": datetime.now(UTC).isoformat(),
        }

    # ── NEXD-11: CROSS-TAB CONTEXTUAL LINKS ──────────────────

    @app.callback(
        Output("dashboard-tabs", "active_tab", allow_duplicate=True),
        Input({"type": "cross-tab-link", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def handle_cross_tab_link(n_clicks_list):
        """NEXD-11: Navigate to the tab indicated by a cross-tab insight link."""
        from config import feature_enabled

        if not feature_enabled("cross_tab_links"):
            return no_update

        triggered = ctx.triggered_id
        if not triggered or not isinstance(triggered, dict):
            return no_update

        # Only act when a click actually occurred
        if not any(n for n in n_clicks_list if n):
            return no_update

        return triggered["index"]

    @app.callback(
        Output("dashboard-tabs", "active_tab", allow_duplicate=True),
        Input({"type": "quick-nav-btn", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def handle_quick_nav_click(n_clicks_list):
        """NEXD-11: Navigate to the tab indicated by a quick-nav card click."""
        from config import feature_enabled

        if not feature_enabled("cross_tab_links"):
            return no_update

        triggered = ctx.triggered_id
        if not triggered or not isinstance(triggered, dict):
            return no_update

        if not any(n for n in n_clicks_list if n):
            return no_update

        return triggered["index"]

    # ── SPRINT 5: A4+E3 — PER-WIDGET CONFIDENCE BADGES ───────

    @app.callback(
        Output("widget-confidence-bar", "children"),
        Input("data-freshness-store", "data"),
    )
    def update_widget_confidence(freshness_json):
        """A4+E3: Show per-source confidence badges below header.

        Each data source gets a green/amber/red/demo badge with age.
        """
        import json
        from datetime import datetime

        if not freshness_json:
            return ""

        freshness = json.loads(freshness_json)
        ts = freshness.get("timestamp", "")

        age_seconds = None
        if ts:
            try:
                fetched = datetime.fromisoformat(ts)
                age_seconds = (datetime.now(UTC) - fetched).total_seconds()
            except (ValueError, TypeError):
                pass

        from components.error_handling import widget_confidence_bar

        return widget_confidence_bar(freshness, age_seconds).children

    # ── SPRINT 5: C9 — MEETING-READY MODE ─────────────────────

    @app.callback(
        [
            Output("meeting-mode-store", "data"),
            Output("dashboard-header", "className"),
            Output("welcome-card", "style"),
            Output("widget-confidence-bar", "style"),
            Output("fallback-banner", "style"),
        ],
        Input("meeting-mode-btn", "n_clicks"),
        State("meeting-mode-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_meeting_mode(n_clicks, current_mode):
        """C9: Toggle meeting-ready mode.

        Meeting mode strips navigation chrome, filters, and sidebars.
        Reformats for projection/PDF: charts expand, narrative becomes
        slide title, annotations remain.
        """
        is_meeting = current_mode != "true"
        new_mode = "true" if is_meeting else "false"

        if is_meeting:
            # Hide non-essential UI elements
            header_class = "dashboard-header meeting-mode"
            welcome_style = {"display": "none"}
            confidence_style = {"display": "none"}
            banner_style = {"display": "none"}
        else:
            # Restore normal mode
            header_class = "dashboard-header"
            welcome_style = {}
            confidence_style = {}
            banner_style = {}

        return new_mode, header_class, welcome_style, confidence_style, banner_style

    # ── DEMAND OUTLOOK TAB ──────────────────────────────────────

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
            fig = go.Figure()
            fig.update_layout(**_layout(uirevision=uirev))
            fig.add_annotation(
                text=f"Forecast failed: {result['error']}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
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
        interval_caption = ""
        if interval_meta.get("method") == "empirical":
            interval_caption = (
                f"<br><sup>80% empirical prediction interval "
                f"(calibration window: last "
                f"{int(interval_meta.get('calibration_window_hours', 0))}h)</sup>"
            )
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

    # ── VALIDATION TAB: BACKTEST & ACCURACY ─────────────────────

    @app.callback(
        [
            Output("backtest-chart", "figure"),
            Output("backtest-mape", "children"),
            Output("backtest-rmse", "children"),
            Output("backtest-mae", "children"),
            Output("backtest-r2", "children"),
            Output("backtest-horizon-explanation", "children"),
            Output("tab3-insight-card", "children"),
        ],
        [
            Input("backtest-horizon", "value"),
            Input("backtest-model", "value"),
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
    def update_backtest_chart(
        horizon, model_name, active_tab, demand_json, persona_id, weather_json, region
    ):
        """Build the backtest chart comparing forecast vs actual."""
        # Only run when this tab is active — avoids expensive model evaluation on page load
        if active_tab != "tab-backtest":
            return [no_update] * 7

        log.info(
            "backtest_callback_start",
            horizon=horizon,
            model=model_name,
            region=region,
            has_demand=bool(demand_json),
            has_weather=bool(weather_json),
        )

        horizon_hours = int(horizon)
        empty_insight = html.Div()

        # uirevision keyed on region + horizon + model so zoom/legend state
        # persists across data refresh but resets when the user picks a new
        # horizon or model.
        uirev = f"{region}:{horizon_hours}:{model_name}"

        # Horizon explanations
        explanations = {
            24: "24-hour ahead: Forecast made 1 day before. Best for day-ahead scheduling.",
            168: "7-day ahead: Forecast made 1 week before. Tests medium-term accuracy.",
            720: "30-day ahead: Forecast made 1 month before. Tests long-term planning reliability.",
        }

        # ── v2 Redis fast path (critical — this is the callback that times out) ──
        if region:
            redis_result = _backtest_tab_from_redis(region, horizon_hours, model_name, persona_id)
            if redis_result is not None:
                return redis_result

        # ── v1 compute fallback ─────────────────────────────
        if not demand_json:
            log.debug("backtest_no_data")
            fig = go.Figure()
            fig.update_layout(**_layout(uirevision=uirev))
            fig.add_annotation(
                text="No data available. Select a region to load data.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return (
                fig,
                "No data",
                "No data",
                "No data",
                "No data",
                explanations.get(horizon_hours, ""),
                empty_insight,
            )

        # Parse data
        demand_df = pd.read_json(io.StringIO(demand_json))
        demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])

        weather_df = pd.DataFrame()
        if weather_json:
            weather_df = pd.read_json(io.StringIO(weather_json))
            weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])

        # Run backtest
        result = _run_backtest_for_horizon(
            demand_df,
            weather_df,
            horizon_hours,
            model_name,
            region,
            exog_mode=DEFAULT_BACKTEST_EXOG_MODE,
        )

        if "error" in result:
            fig = go.Figure()
            fig.update_layout(**_layout(uirevision=uirev))
            fig.add_annotation(
                text=f"Backtest failed: {result['error']}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return (
                fig,
                "No data",
                "No data",
                "No data",
                "No data",
                explanations.get(horizon_hours, ""),
                empty_insight,
            )

        timestamps = pd.to_datetime(result["timestamps"])
        actual = result["actual"]
        predictions = result["predictions"]
        metrics = result["metrics"]
        exog_mode = _normalize_backtest_exog_mode(
            result.get("exog_mode", DEFAULT_BACKTEST_EXOG_MODE)
        )
        exog_caption = _describe_exog_mode(exog_mode, result.get("exog_source"))

        # Build figure
        fig = go.Figure()

        # Actual demand line
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=actual,
                mode="lines",
                name="Actual Demand",
                line=dict(color=COLORS["actual"], width=2),
            )
        )

        # Forecast line
        model_colors = {
            "xgboost": COLORS.get("ensemble", "#2DE2C4"),
            "prophet": COLORS.get("prophet", "#E69F00"),
            "arima": COLORS.get("arima", "#009E73"),
            "ensemble": COLORS.get("ensemble", "#2DE2C4"),
        }

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=predictions,
                mode="lines",
                name=f"{model_name.upper()} Forecast",
                line=dict(color=model_colors.get(model_name, "#2DE2C4"), width=2, dash="dash"),
            )
        )
        interval_meta = result.get("interval", {})
        if isinstance(interval_meta, dict) and len(interval_meta.get("lower", [])) == len(
            predictions
        ):
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=interval_meta["upper"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=interval_meta["lower"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=COLORS["confidence"],
                    name="80% empirical prediction interval",
                    hoverinfo="skip",
                )
            )

        # Error shading (where forecast differs from actual), segmented by fold
        # to avoid implying visual continuity across retrained fold boundaries.
        num_folds = result.get("num_folds", 1)
        fold_boundaries = result.get("fold_boundaries", [0])
        # Build (start, end) pairs for each fold segment
        fold_ranges = []
        for i, start_idx in enumerate(fold_boundaries):
            end_idx = fold_boundaries[i + 1] if i + 1 < len(fold_boundaries) else len(timestamps)
            fold_ranges.append((start_idx, end_idx))

        for fold_i, (f_start, f_end) in enumerate(fold_ranges):
            fold_ts = timestamps[f_start:f_end]
            fold_pred = predictions[f_start:f_end]
            fold_act = actual[f_start:f_end]
            fig.add_trace(
                go.Scatter(
                    x=list(fold_ts) + list(fold_ts[::-1]),
                    y=list(fold_pred) + list(fold_act[::-1]),
                    fill="toself",
                    fillcolor="rgba(255,92,122,0.12)",
                    line=dict(width=0),
                    name="Forecast Error" if fold_i == 0 else None,
                    showlegend=(fold_i == 0),
                    hoverinfo="skip",
                )
            )

        # Fold boundary lines
        for boundary_idx in fold_boundaries[1:]:
            fig.add_vline(
                x=timestamps[boundary_idx],
                line_dash="dot",
                line_color="rgba(255, 255, 255, 0.3)",
                line_width=1,
            )

        # Layout
        horizon_labels = {24: "24-Hour", 168: "7-Day", 720: "30-Day"}
        horizon_label = horizon_labels.get(horizon_hours, "")
        fold_label = f" ({num_folds} folds)" if num_folds > 1 else ""
        fig.update_layout(
            **_layout(
                uirevision=uirev,
                title=(
                    f"{horizon_label} Walk-Forward Backtest{fold_label}: "
                    f"{model_name.upper()} vs Actual — {region}<br><sup>{exog_caption}</sup>"
                ),
                xaxis_title="Date/Time",
                yaxis_title="Demand (MW)",
                hovermode="x unified",
            )
        )

        # Format metrics
        mode_suffix = f" ({exog_mode})"
        mape_str = f"{metrics['mape']:.2f}%{mode_suffix}"
        rmse_str = f"{metrics['rmse']:,.0f} MW{mode_suffix}"
        mae_str = f"{metrics['mae']:,.0f} MW{mode_suffix}"
        r2_str = f"{metrics['r2']:.3f}{mode_suffix}"
        monitor = (
            interval_meta.get("coverage_monitor", {}) if isinstance(interval_meta, dict) else {}
        )
        coverage_str = f"{monitor.get('recent_coverage', 0.0) * 100:.1f}%"
        drift_pp = monitor.get("drift", 0.0) * 100.0
        calibration_window = int(interval_meta.get("calibration_window_hours", 0) or 0)

        # Generate insights
        from components.insights import build_insight_card, generate_tab3_insights

        persona = persona_id or "grid_ops"
        # Collect all model metrics for cross-model comparison
        all_metrics = {model_name: metrics}
        tab3_insights = generate_tab3_insights(
            persona,
            region or "FPL",
            all_metrics,
            model_name=model_name,
            horizon_hours=horizon_hours,
            actual=actual,
            predictions=predictions,
            timestamps=timestamps,
            num_folds=num_folds,
        )
        insight_card = build_insight_card(tab3_insights, persona, "tab-backtest")

        log.info(
            "backtest_callback_complete", mape=mape_str, model=model_name, horizon=horizon_hours
        )
        return (
            fig,
            mape_str,
            rmse_str,
            mae_str,
            r2_str,
            (
                f"{explanations.get(horizon_hours, '')} Exogenous mode: {exog_caption}. "
                f"Interval: 80% empirical prediction interval (calibration window: last {calibration_window}h). "
                f"Recent coverage: {coverage_str} (drift vs 80% target: {drift_pp:+.1f} pp)."
            ),
            insight_card,
        )


# ── HELPER FUNCTIONS ──────────────────────────────────────────


# ── Overview helpers (R2 — v2 linear stack) ─────────────────────────


def _build_overview_title(region: str) -> html.Div:
    """Page-title block: region name + 1-line subtitle."""
    from config import REGION_NAMES

    region_name = REGION_NAMES.get(region, region)
    subtitle = f"Demand forecast and grid intelligence · {region}"
    return build_page_title(region_name, subtitle)


def _build_overview_metrics_items(demand_df: pd.DataFrame | None) -> list[dict]:
    """Compose the 5-up MetricsBar cells (Now / 7d Peak / 7d Low / Average / 24h Trend)."""
    placeholder_labels = ["Now", "7d Peak", "7d Low", "Average", "24h Trend"]
    if demand_df is None or demand_df.empty or "demand_mw" not in demand_df.columns:
        items = [
            {"label": label, "value": "—", "unit": None, "tone": "secondary"}
            for label in placeholder_labels
        ]
        items[0]["hero"] = True
        items[0]["tone"] = "primary"
        return items

    df = demand_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Strip spurious zero-demand rows (EIA's missing-observation marker)
    nonzero = df[df["demand_mw"] > 0]
    last_7d = nonzero.tail(168)

    now_value = float(df["demand_mw"].iloc[-1]) if not df.empty else 0.0
    peak_7d = float(last_7d["demand_mw"].max()) if not last_7d.empty else 0.0
    low_7d = float(last_7d["demand_mw"].min()) if not last_7d.empty else 0.0
    avg_7d = float(last_7d["demand_mw"].mean()) if not last_7d.empty else 0.0

    # 24h trend (now vs 24 hours ago)
    ago_24h = float(df["demand_mw"].iloc[-25]) if len(df) >= 25 else now_value
    trend_pct = ((now_value - ago_24h) / ago_24h * 100.0) if ago_24h else 0.0
    # Inverted semantic: rising demand reads as "warning" (negative tone),
    # falling demand reads as "positive" — matches v2 MetricsBar.tsx:64.
    trend_tone = (
        "negative" if trend_pct > 0.5 else ("positive" if trend_pct < -0.5 else "secondary")
    )

    return [
        {"label": "Now", "value": f"{now_value:,.0f}", "unit": "MW", "hero": True},
        {"label": "7d Peak", "value": f"{peak_7d:,.0f}", "unit": "MW", "tone": "secondary"},
        {"label": "7d Low", "value": f"{low_7d:,.0f}", "unit": "MW", "tone": "secondary"},
        {"label": "Average", "value": f"{avg_7d:,.0f}", "unit": "MW", "tone": "secondary"},
        {"label": "24h Trend", "value": f"{trend_pct:+.1f}%", "unit": None, "tone": trend_tone},
    ]


def _build_overview_hero_chart(
    region: str,
    demand_df: pd.DataFrame | None,
) -> go.Figure:
    """7d actual demand + 24h forecast bridge with confidence band.

    Mirrors gridpulse-v2 components/DemandChart.tsx — blue solid actual with
    a faint area fill, orange dashed forecast bridged from the last actual
    point, and an orange-tinted confidence ribbon under the forecast.
    """
    if demand_df is None or demand_df.empty or "demand_mw" not in demand_df.columns:
        return _empty_figure("No demand data")

    df = demand_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    actual = df.tail(168)
    if actual.empty:
        return _empty_figure("No recent demand")

    last_ts = actual["timestamp"].iloc[-1]
    last_mw = float(actual["demand_mw"].iloc[-1])

    fig = go.Figure()

    # Actual demand: blue solid + faint area fill below
    fig.add_trace(
        go.Scatter(
            x=actual["timestamp"],
            y=actual["demand_mw"].where(actual["demand_mw"] > 0),
            mode="lines",
            name="Actual",
            line=dict(color="#3b82f6", width=1.75),
            fill="tozeroy",
            fillcolor="rgba(59, 130, 246, 0.08)",
            hovertemplate="<b>%{x|%b %d, %H:%M}</b><br>%{y:,.0f} MW<extra></extra>",
        )
    )

    # 24h forecast bridge (orange dashed) + confidence band
    try:
        from models.model_service import get_forecasts

        forecasts = get_forecasts(region, df, models_shown=["ensemble"])
        ensemble = forecasts.get("ensemble")
        upper_80 = forecasts.get("upper_80")
        lower_80 = forecasts.get("lower_80")
        if ensemble is not None and len(ensemble) > 0:
            horizon = min(24, len(ensemble))
            forecast_ts = pd.date_range(
                start=last_ts + pd.Timedelta(hours=1),
                periods=horizon,
                freq="h",
            )
            ensemble_y = list(ensemble[:horizon])

            # Confidence band — drawn first so it sits behind the line
            if upper_80 is not None and lower_80 is not None and len(upper_80) >= horizon:
                upper_y = list(upper_80[:horizon])
                lower_y = list(lower_80[:horizon])
                fig.add_trace(
                    go.Scatter(
                        x=list(forecast_ts) + list(forecast_ts[::-1]),
                        y=upper_y + lower_y[::-1],
                        fill="toself",
                        fillcolor="rgba(249, 115, 22, 0.12)",
                        line=dict(width=0),
                        hoverinfo="skip",
                        showlegend=False,
                        name="80% confidence",
                    )
                )

            # Forecast line (bridged from last actual point — no gap)
            bridge_x = [last_ts, *forecast_ts]
            bridge_y = [last_mw, *ensemble_y]
            fig.add_trace(
                go.Scatter(
                    x=bridge_x,
                    y=bridge_y,
                    mode="lines",
                    name="Forecast (24h)",
                    line=dict(color="#f97316", width=1.75, dash="dash"),
                    hovertemplate=(
                        "<b>%{x|%b %d, %H:%M}</b><br>%{y:,.0f} MW · forecast<extra></extra>"
                    ),
                )
            )
    except Exception as exc:  # pragma: no cover — fall back to actual-only chart
        log.warning("overview_hero_forecast_failed", region=region, error=str(exc))

    fig.update_layout(
        **_layout(uirevision=region, showlegend=False),
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
    )
    return fig


def _build_overview_model_card(region: str) -> html.Div:
    """Horizontal model-performance bar (top/bottom borders only)."""
    try:
        from models.model_service import get_model_metrics, is_trained
    except ImportError:  # pragma: no cover — defensive
        return html.Div()

    metrics_dict = get_model_metrics(region)
    if not metrics_dict:
        return html.Div()

    # Prefer ensemble; fall back to xgboost; finally first available
    if "ensemble" in metrics_dict:
        primary_key = "ensemble"
    elif "xgboost" in metrics_dict:
        primary_key = "xgboost"
    else:
        primary_key = next(iter(metrics_dict.keys()), None)
    if primary_key is None:
        return html.Div()

    m = metrics_dict[primary_key]
    formatted = {
        "MAPE": f"{m.get('mape', 0.0):.1f}%",
        "RMSE": f"{m.get('rmse', 0.0):,.0f} MW",
        "MAE": f"{m.get('mae', 0.0):,.0f} MW",
        "R²": f"{m.get('r2', 0.0):.3f}",
    }
    name = "XGBoost" if primary_key == "xgboost" else primary_key.title()
    badge = "trained" if is_trained(region) else "simulated"
    return build_model_metrics_card(model_name=name, metrics=formatted, badge=badge)


def _build_overview_insight(
    region: str,
    demand_df: pd.DataFrame | None,
    persona_id: str,
) -> html.Div:
    """3-sentence narrative paragraph with semantic-color delta spans."""
    if demand_df is None or demand_df.empty or "demand_mw" not in demand_df.columns:
        return build_insight_card(
            "Summary",
            (
                "Awaiting demand data for this region. The forecast will populate once "
                "the next pipeline cycle completes."
            ),
        )

    df = demand_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    nonzero = df[df["demand_mw"] > 0]
    last_7d = nonzero.tail(168)

    now_value = float(df["demand_mw"].iloc[-1])
    avg_7d = float(last_7d["demand_mw"].mean()) if not last_7d.empty else 0.0
    delta_pct = ((now_value - avg_7d) / avg_7d * 100.0) if avg_7d else 0.0
    direction = "above" if delta_pct >= 0 else "below"
    # Inverted semantic: rising demand reads as warning (matches v2)
    delta_class = (
        "gp-insight-card__delta--negative" if delta_pct >= 0 else "gp-insight-card__delta--positive"
    )

    last_24h = df.tail(24)
    if not last_24h.empty:
        peak_idx = last_24h["demand_mw"].idxmax()
        peak_mw = float(last_24h.loc[peak_idx, "demand_mw"])
        peak_ts = pd.to_datetime(last_24h.loc[peak_idx, "timestamp"])
        peak_str = f"{peak_mw:,.0f} MW at {peak_ts.strftime('%H:%M')}"
    else:
        peak_str = "—"

    forecast_clause = "Next-cycle forecast confidence is updating."
    try:
        from models.model_service import get_forecasts

        forecasts = get_forecasts(region, df, models_shown=["ensemble"])
        ensemble = forecasts.get("ensemble")
        if ensemble is not None and len(ensemble) > 0:
            horizon = min(24, len(ensemble))
            f_arr = np.asarray(ensemble[:horizon])
            f_peak = float(f_arr.max())
            f_peak_idx = int(f_arr.argmax())
            f_peak_ts = df["timestamp"].iloc[-1] + pd.Timedelta(hours=f_peak_idx + 1)
            metrics_dict = forecasts.get("metrics") or {}
            ens_metrics = metrics_dict.get("ensemble") or metrics_dict.get("xgboost") or {}
            mape = float(ens_metrics.get("mape", 0.0))
            forecast_clause = (
                f"Next-24h forecast peaks at {f_peak:,.0f} MW around "
                f"{f_peak_ts.strftime('%H:%M')} (MAPE {mape:.1f}%)."
            )
    except Exception as exc:  # pragma: no cover
        log.warning("overview_insight_forecast_failed", region=region, error=str(exc))

    body = [
        "Demand is ",
        html.Span(f"{abs(delta_pct):.1f}% {direction}", className=delta_class),
        " the 7-day average. ",
        "Recent peak: ",
        html.Span(peak_str, className="gp-insight-card__strong"),
        ". ",
        forecast_clause,
    ]
    # Persona influences eyebrow only — keeps the card tonally consistent
    eyebrow_map = {
        "grid_ops": "Operating summary",
        "renewables": "Renewables outlook",
        "trader": "Market signal",
        "data_scientist": "Model summary",
    }
    eyebrow = eyebrow_map.get(persona_id, "Summary")
    return build_insight_card(eyebrow, body)


def _build_overview_sparkline(demand_df: pd.DataFrame | None, region: str) -> go.Figure:
    """Build a compact 24h demand sparkline for the overview tab."""
    if demand_df is None or demand_df.empty or "demand_mw" not in demand_df.columns:
        return _empty_figure("No demand data")

    df = demand_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    last_24h = df.tail(24)

    if last_24h.empty:
        return _empty_figure("No recent demand data")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=last_24h["timestamp"],
            y=last_24h["demand_mw"],
            mode="lines",
            line=dict(color=CB_PALETTE["blue"], width=2),
            fill="tozeroy",
            fillcolor="rgba(0,114,178,0.15)",
            name="Demand",
            hovertemplate="%{x|%H:%M}<br>%{y:,.0f} MW<extra></extra>",
        )
    )
    sparkline_layout = _layout(
        uirevision=region,
        showlegend=False,
        margin=dict(l=40, r=10, t=10, b=30),
    )
    fig.update_layout(
        **sparkline_layout,
        xaxis=dict(
            showgrid=False,
            tickformat="%H:%M",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            tickformat=",.0f",
            title="MW",
        ),
    )
    return fig


def _build_overview_briefing(
    persona_id: str,
    region: str,
    demand_df: pd.DataFrame | None,
    weather_df: pd.DataFrame | None,
) -> html.Div:
    """Build the AI executive briefing section."""
    from data.ai_briefing import generate_briefing

    try:
        result = generate_briefing(persona_id, region, demand_df, weather_df)
    except Exception as exc:
        log.error("overview_briefing_failed", error=str(exc))
        return html.Div(
            "Briefing unavailable",
            style={"color": "#A8B3C7", "fontStyle": "italic"},
        )

    persona = get_persona(persona_id)

    children = [
        html.P(
            result.summary,
            style={
                "color": "#DDE6F2",
                "fontSize": "0.9rem",
                "lineHeight": "1.6",
                "marginBottom": "12px",
            },
        ),
    ]

    if result.observations:
        obs_items = []
        for obs in result.observations:
            obs_items.append(
                html.Li(
                    obs,
                    style={
                        "color": "#A8B3C7",
                        "fontSize": "0.82rem",
                        "marginBottom": "4px",
                        "lineHeight": "1.5",
                    },
                )
            )
        children.append(
            html.Ul(
                obs_items,
                style={"paddingLeft": "20px", "marginBottom": "8px"},
            )
        )

    source_label = "AI Analysis" if result.source == "claude" else "Data Summary"
    children.append(
        html.Span(
            source_label,
            style={
                "fontSize": "0.65rem",
                "color": "#A8B3C7",
                "textTransform": "uppercase",
                "letterSpacing": "0.5px",
            },
        )
    )

    return html.Div(
        children,
        style={"borderLeft": f"4px solid {persona.color}"},
        className="briefing-card-content",
    )


def _build_weather_context(latest: "pd.Series") -> html.Div:
    """Build a row of weather KPI mini-cards from the latest weather reading."""
    import dash_bootstrap_components as dbc

    temp = latest.get("temperature_2m")
    wind = latest.get("wind_speed_80m", latest.get("wind_speed_10m"))
    humidity = latest.get("relative_humidity_2m")
    cloud = latest.get("cloud_cover")

    cards = []

    if temp is not None:
        t = float(temp)
        color = "#FF5C7A" if t >= 95 else ("#FFB84D" if t >= 85 else "#2BD67B")
        cards.append(
            dbc.Col(
                html.Div(
                    [
                        html.P("TEMPERATURE", className="kpi-label"),
                        html.H4(
                            f"{t:.0f}°F",
                            className="kpi-value",
                            style={"fontSize": "1.3rem"},
                        ),
                    ],
                    className="kpi-card",
                    style={"borderTop": f"3px solid {color}"},
                ),
                md=3,
            )
        )

    if wind is not None:
        w = float(wind)
        color = "#FF5C7A" if w >= 40 else ("#FFB84D" if w >= 25 else "#2BD67B")
        cards.append(
            dbc.Col(
                html.Div(
                    [
                        html.P("WIND SPEED", className="kpi-label"),
                        html.H4(
                            f"{w:.0f} mph",
                            className="kpi-value",
                            style={"fontSize": "1.3rem"},
                        ),
                    ],
                    className="kpi-card",
                    style={"borderTop": f"3px solid {color}"},
                ),
                md=3,
            )
        )

    if humidity is not None:
        h = float(humidity)
        color = "#FFB84D" if h >= 80 else "#2BD67B"
        cards.append(
            dbc.Col(
                html.Div(
                    [
                        html.P("HUMIDITY", className="kpi-label"),
                        html.H4(
                            f"{h:.0f}%",
                            className="kpi-value",
                            style={"fontSize": "1.3rem"},
                        ),
                    ],
                    className="kpi-card",
                    style={"borderTop": f"3px solid {color}"},
                ),
                md=3,
            )
        )

    if cloud is not None:
        c = float(cloud)
        color = "#A8B3C7"
        cards.append(
            dbc.Col(
                html.Div(
                    [
                        html.P("CLOUD COVER", className="kpi-label"),
                        html.H4(
                            f"{c:.0f}%",
                            className="kpi-value",
                            style={"fontSize": "1.3rem"},
                        ),
                    ],
                    className="kpi-card",
                    style={"borderTop": f"3px solid {color}"},
                ),
                md=3,
            )
        )

    if not cards:
        return html.Div()

    return dbc.Row(cards, className="g-2")


def _build_changes_card(
    changes_json: str | None,
    persona: str,
    region: str,
    snapshots: dict | None,
) -> html.Div:
    """Build the 'Since your last visit' card for the Overview tab (NEXD-8)."""
    import json as _json

    from data.session_diff import format_relative_time

    if not changes_json:
        return html.Div()

    try:
        changes = _json.loads(changes_json) if isinstance(changes_json, str) else changes_json
    except Exception:
        return html.Div()

    if not changes:
        return html.Div()

    # Get previous visit timestamp for this region
    prev_timestamp = None
    if snapshots and isinstance(snapshots, dict):
        entry = snapshots.get(region)
        if entry and isinstance(entry, dict):
            prev_timestamp = entry.get("timestamp")

    rel_time = format_relative_time(prev_timestamp) if prev_timestamp else ""

    # Build bullet items
    items = []
    for change in changes[:5]:
        icon = change.get("icon", "")
        text = change.get("text", "")
        items.append(
            html.Div(
                [
                    html.Span(icon, className="change-icon"),
                    html.Span(text),
                ],
                className="change-item",
            )
        )

    header_parts = [
        html.Span("Since your last visit", className="changes-title"),
    ]
    if rel_time:
        header_parts.append(html.Span(rel_time, className="changes-timestamp"))

    return html.Div(
        [
            html.Div(header_parts, className="changes-header"),
            *items,
        ],
        className="changes-card",
    )


def _build_overview_data_health(freshness_data: dict | None) -> html.Div:
    """Build data health badges showing per-source freshness."""
    if not freshness_data:
        return html.Div()

    source_config = {
        "demand": {"label": "EIA Demand", "icon": "\u26a1"},
        "weather": {"label": "Weather", "icon": "\u2601"},
        "alerts": {"label": "NOAA Alerts", "icon": "\u26a0"},
    }

    status_colors = {
        "fresh": "#2BD67B",
        "stale": "#FFB84D",
        "demo": "#A8B3C7",
        "error": "#FF5C7A",
    }

    badges = []
    for source, status in freshness_data.items():
        if source == "timestamp":
            continue
        cfg = source_config.get(source, {"label": source.title(), "icon": "\u25cf"})
        color = status_colors.get(status, "#A8B3C7")
        status_text = status.upper() if status != "fresh" else "LIVE"
        badges.append(
            html.Div(
                [
                    html.Span(cfg["icon"], style={"marginRight": "6px"}),
                    html.Span(
                        cfg["label"],
                        style={"fontWeight": "600", "marginRight": "6px"},
                    ),
                    html.Span(
                        status_text,
                        style={
                            "fontSize": "0.65rem",
                            "padding": "1px 6px",
                            "borderRadius": "3px",
                            "background": f"{color}20",
                            "color": color,
                        },
                    ),
                ],
                className="data-health-badge",
                style={
                    "display": "inline-flex",
                    "alignItems": "center",
                    "fontSize": "0.75rem",
                    "color": "#A8B3C7",
                    "padding": "4px 12px",
                    "marginRight": "12px",
                },
            )
        )

    if not badges:
        return html.Div()

    return html.Div(
        [
            html.Span(
                "DATA SOURCES",
                style={
                    "fontSize": "0.65rem",
                    "color": "#A8B3C7",
                    "textTransform": "uppercase",
                    "letterSpacing": "1px",
                    "marginRight": "16px",
                },
            ),
            *badges,
        ],
        style={
            "display": "flex",
            "flexWrap": "wrap",
            "alignItems": "center",
            "padding": "8px 12px",
            "background": "#11182D",
            "borderRadius": "6px",
        },
    )


def _build_overview_spotlight(
    persona_id: str,
    region: str,
    demand_df: pd.DataFrame | None,
    weather_df: pd.DataFrame | None,
) -> go.Figure:
    """Build persona-specific spotlight chart for overview."""
    if persona_id == "renewables":
        return _spotlight_renewables(weather_df, region)
    if persona_id == "trader":
        return _spotlight_trader(demand_df, region)
    if persona_id == "data_scientist":
        return _spotlight_model_accuracy(region)
    # Default: grid_ops → demand sparkline
    return _build_overview_sparkline(demand_df, region)


def _spotlight_renewables(weather_df: pd.DataFrame | None, region: str) -> go.Figure:
    """Renewable generation potential chart."""
    if weather_df is None or weather_df.empty:
        return _empty_figure("No weather data for renewable outlook")

    df = weather_df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    last_48h = df.tail(48)

    fig = go.Figure()
    if "wind_speed_80m" in last_48h.columns:
        fig.add_trace(
            go.Scatter(
                x=last_48h.get("timestamp", list(range(len(last_48h)))),
                y=last_48h["wind_speed_80m"],
                mode="lines",
                line=dict(color=CB_PALETTE.get("sky", "#56B4E9"), width=2),
                name="Wind (mph)",
                hovertemplate="%{y:.0f} mph<extra>Wind</extra>",
            )
        )
    if "shortwave_radiation" in last_48h.columns:
        fig.add_trace(
            go.Scatter(
                x=last_48h.get("timestamp", list(range(len(last_48h)))),
                y=last_48h["shortwave_radiation"],
                mode="lines",
                line=dict(color=CB_PALETTE.get("orange", "#E69F00"), width=2),
                name="Solar (W/m\u00b2)",
                yaxis="y2",
                hovertemplate="%{y:.0f} W/m\u00b2<extra>Solar</extra>",
            )
        )

    renew_layout = _layout(
        uirevision=region,
        margin=dict(l=45, r=45, t=35, b=40),
        legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
    )
    fig.update_layout(
        **renew_layout,
        title=dict(text="Renewable Potential (48h)", font=dict(size=13, color="#DDE6F2")),
        showlegend=True,
        yaxis=dict(
            title="Wind (mph)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis2=dict(
            title="Solar (W/m\u00b2)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        xaxis=dict(showgrid=False, tickformat="%b %d %H:%M"),
    )
    return fig


def _spotlight_trader(demand_df: pd.DataFrame | None, region: str) -> go.Figure:
    """Demand vs capacity utilization chart for traders."""
    capacity = REGION_CAPACITY_MW.get(region, 50000)

    if demand_df is None or demand_df.empty:
        return _empty_figure("No demand data for market view")

    df = demand_df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    last_48h = df.tail(48)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=last_48h.get("timestamp", list(range(len(last_48h)))),
            y=last_48h["demand_mw"],
            mode="lines",
            line=dict(color=CB_PALETTE["blue"], width=2),
            fill="tozeroy",
            fillcolor="rgba(0,114,178,0.15)",
            name="Demand",
            hovertemplate="%{y:,.0f} MW<extra>Demand</extra>",
        )
    )

    # Capacity line
    fig.add_hline(
        y=capacity,
        line_dash="dot",
        line_color="#FF5C7A",
        annotation_text=f"Capacity: {capacity:,.0f} MW",
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color="#FF5C7A",
    )

    # Pricing tier thresholds
    for pct, label, color in [
        (0.85, "High tier (85%)", "#FFB84D"),
        (0.70, "Moderate (70%)", "#A8B3C7"),
    ]:
        fig.add_hline(
            y=capacity * pct,
            line_dash="dot",
            line_color=color,
            line_width=1,
            annotation_text=label,
            annotation_position="bottom left",
            annotation_font_size=9,
            annotation_font_color=color,
        )

    trader_layout = _layout(uirevision=region, margin=dict(l=50, r=10, t=35, b=30))
    fig.update_layout(
        **trader_layout,
        title=dict(text="Demand vs Capacity", font=dict(size=13, color="#DDE6F2")),
        showlegend=False,
        xaxis=dict(showgrid=False, tickformat="%b %d %H:%M"),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            tickformat=",.0f",
            title="MW",
        ),
    )
    return fig


def _spotlight_model_accuracy(region: str) -> go.Figure:
    """Model accuracy bar chart for data scientists."""
    # Pull from backtest cache if available
    models = ["prophet", "arima", "xgboost"]
    mape_values = []

    for model_name in models:
        mape = None
        for horizon in [168, 24, 720]:
            bt_key = (region, horizon, model_name, DEFAULT_BACKTEST_EXOG_MODE)
            if bt_key in _BACKTEST_CACHE:
                result_dict, _, _ = _BACKTEST_CACHE[bt_key]
                if isinstance(result_dict, dict) and "mape" in result_dict:
                    mape = result_dict["mape"]
                    break
        mape_values.append(mape if mape is not None else 4.5 + len(model_name) * 0.3)

    colors = [
        CB_PALETTE.get("vermillion", "#D55E00"),
        CB_PALETTE.get("blue", "#0072B2"),
        CB_PALETTE.get("green", "#009E73"),
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[m.title() for m in models],
            y=mape_values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in mape_values],
            textposition="outside",
            textfont=dict(color="#DDE6F2", size=11),
            hovertemplate="%{x}: %{y:.2f}% MAPE<extra></extra>",
        )
    )

    model_layout = _layout(uirevision=region, margin=dict(l=40, r=10, t=35, b=30))
    fig.update_layout(
        **model_layout,
        title=dict(text="Model MAPE Comparison", font=dict(size=13, color="#DDE6F2")),
        showlegend=False,
        yaxis=dict(
            title="MAPE (%)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
        ),
        xaxis=dict(showgrid=False),
    )
    return fig


def _build_overview_digest(
    persona_id: str,
    region: str,
    demand_df: pd.DataFrame | None,
    weather_df: pd.DataFrame | None,
) -> html.Div:
    """Aggregate top insights from all tabs into a cross-tab digest."""
    from components.insights import (
        Insight,
        build_insight_card,
        generate_tab1_insights,
        generate_tab2_insights,
        generate_tab3_insights,
        generate_tab4_insights,
    )

    all_insights: list[Insight] = []

    # Tab 1: Historical demand insights
    try:
        tab1 = generate_tab1_insights(persona_id, region, demand_df, weather_df)
        all_insights.extend(tab1)
    except Exception:
        pass

    # Tab 2: Forecast insights (need predictions — skip if unavailable)
    try:
        for horizon in [168, 24]:
            pred_key = (region, horizon)
            if pred_key in _PREDICTION_CACHE:
                preds, timestamps, _, _ = _PREDICTION_CACHE[pred_key]
                tab2 = generate_tab2_insights(persona_id, region, preds, timestamps)
                all_insights.extend(tab2)
                break
    except Exception:
        pass

    # Validation tab: model accuracy insights (from backtest cache)
    try:
        for model_name in ["xgboost", "prophet", "arima"]:
            for horizon in [168, 24, 720]:
                bt_key = (region, horizon, model_name, DEFAULT_BACKTEST_EXOG_MODE)
                if bt_key in _BACKTEST_CACHE:
                    result_dict, _, _ = _BACKTEST_CACHE[bt_key]
                    if isinstance(result_dict, dict) and "mape" in result_dict:
                        metrics = {model_name: result_dict}
                        tab3 = generate_tab3_insights(
                            persona_id,
                            region,
                            metrics,
                            model_name=model_name,
                            horizon_hours=horizon,
                        )
                        all_insights.extend(tab3)
                        break
            else:
                continue
            break
    except Exception:
        pass

    # Tab 4: Grid insights (from generation cache)
    try:
        if region in _GENERATION_CACHE:
            gen_df, _ = _GENERATION_CACHE[region]
            if gen_df is not None and not gen_df.empty:
                gen_copy = gen_df.copy()
                gen_copy["timestamp"] = pd.to_datetime(gen_copy["timestamp"])
                pivot = gen_copy.pivot_table(
                    index="timestamp",
                    columns="fuel_type",
                    values="generation_mw",
                    aggfunc="sum",
                ).fillna(0)
                total_gen = pivot.sum(axis=1)
                renewable_cols = [
                    c for c in pivot.columns if c.lower() in ("wind", "solar", "hydro")
                ]
                renewable_gen = (
                    pivot[renewable_cols].sum(axis=1) if renewable_cols else total_gen * 0
                )
                renewable_pct = (
                    (renewable_gen.sum() / total_gen.sum() * 100) if total_gen.sum() > 0 else 0.0
                )
                net_load = total_gen - renewable_gen
                tab4 = generate_tab4_insights(
                    persona_id=persona_id,
                    region=region,
                    net_load=net_load,
                    demand=total_gen,
                    renewable_pct=renewable_pct,
                    pivot=pivot,
                    timestamps=pd.DatetimeIndex(pivot.index),
                )
                all_insights.extend(tab4)
    except Exception:
        pass

    if not all_insights:
        return html.Div(
            html.P(
                "No insights available yet. Explore tabs to generate data.",
                style={"color": "#A8B3C7", "fontSize": "0.82rem", "fontStyle": "italic"},
            )
        )

    # Sort by severity (warning first)
    severity_order = {"warning": 0, "notable": 1, "info": 2}
    all_insights.sort(key=lambda i: severity_order.get(i.severity, 2))

    return build_insight_card(all_insights, persona_id, "Overview", max_insights=5)


def _build_overview_news() -> html.Div:
    """Fetch and render energy news for the overview tab."""
    from data.news_client import fetch_energy_news

    try:
        articles = fetch_energy_news(page_size=10)
        if not articles:
            from data.news_client import _get_demo_news

            articles = _get_demo_news()
        return build_news_feed(articles)
    except Exception as e:
        log.error("overview_news_failed", error=str(e))
        from data.news_client import _get_demo_news

        return build_news_feed(_get_demo_news())


def _empty_figure(message: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **_layout(
            uirevision="empty",
            annotations=[
                dict(
                    text=message,
                    showarrow=False,
                    font=dict(size=14, color="#A8B3C7"),
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                )
            ],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
    )
    return fig


def _get_feature_importance(region: str, top_n: int = 10) -> tuple[list[str], np.ndarray]:
    """Extract real feature importances from cached XGBoost model, or return defaults."""
    if (region, "xgboost", 0) in _MODEL_CACHE:
        model_dict, _, _ = _MODEL_CACHE[(region, "xgboost", 0)]
        if isinstance(model_dict, dict) and "feature_importances" in model_dict:
            imp = model_dict["feature_importances"]
            sorted_feats = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
            names = [f[0] for f in sorted_feats]
            vals = np.array([f[1] for f in sorted_feats])
            if vals.sum() > 0:
                return names, vals

    names = [
        "temperature_2m",
        "demand_lag_24h",
        "hour_sin",
        "cooling_degree_days",
        "wind_speed_80m",
        "demand_roll_24h_mean",
        "heating_degree_days",
        "solar_capacity_factor",
        "relative_humidity_2m",
        "is_holiday",
    ]
    vals = np.array([0.25, 0.18, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04])
    return names, vals


def _build_persona_kpis(
    persona_id: str,
    region: str,
    demand_df: pd.DataFrame | None = None,
    weather_df: pd.DataFrame | None = None,
) -> dbc.Row:
    """Build persona-specific KPI cards from live demand/weather data."""
    capacity = REGION_CAPACITY_MW.get(region, 50000)

    # Extract real stats from demand data
    peak_mw = None
    avg_mw = None
    min_mw = None
    pct_of_capacity = None
    if demand_df is not None and "demand_mw" in demand_df.columns:
        # Drop NaN and non-positive values — EIA occasionally emits spurious
        # zero rows (e.g. NYISO 2026-02-10 had 6 consecutive zero-hours)
        # that collapse Demand Range to the full peak. Utility demand is
        # never physically 0 or negative for any covered BA.
        valid = demand_df.dropna(subset=["demand_mw"])
        valid = valid[valid["demand_mw"] > 0]
        if not valid.empty:
            peak_mw = valid["demand_mw"].max()
            avg_mw = valid["demand_mw"].mean()
            min_mw = valid["demand_mw"].min()
            pct_of_capacity = peak_mw / capacity * 100 if capacity > 0 else 0

    # Fallback: read demand stats from Redis
    if peak_mw is None:
        actuals_redis = redis_get(f"wattcast:actuals:{region}")
        if actuals_redis and actuals_redis.get("demand_mw"):
            demand_vals = [v for v in actuals_redis["demand_mw"] if v is not None and v > 0]
            if demand_vals:
                peak_mw = max(demand_vals)
                avg_mw = sum(demand_vals) / len(demand_vals)
                min_mw = min(demand_vals)
                pct_of_capacity = peak_mw / capacity * 100 if capacity > 0 else 0

    # Extract weather stats. pandas ``.mean()`` is NaN-aware but returns NaN
    # when every value is null — coerce that to ``None`` so the downstream
    # ``is not None`` guards (and Redis fallback) behave consistently.
    avg_wind = None
    avg_solar = None
    if weather_df is not None:
        if "wind_speed_80m" in weather_df.columns:
            mean = weather_df["wind_speed_80m"].mean()
            avg_wind = float(mean) if pd.notna(mean) else None
        if "shortwave_radiation" in weather_df.columns:
            mean = weather_df["shortwave_radiation"].mean()
            avg_solar = float(mean) if pd.notna(mean) else None

    # Fallback: read weather stats from Redis. Gate each metric independently
    # so a missing column in ``weather_df`` still triggers the Redis lookup
    # for that metric (previously both had to be ``None`` for any fallback).
    # Also filter None/NaN entries before averaging — Redis arrays can have
    # gaps where Open-Meteo emitted nulls.
    if avg_wind is None or avg_solar is None:
        weather_redis = redis_get(f"wattcast:weather:{region}")
        if weather_redis:
            if avg_wind is None and "wind_speed_80m" in weather_redis:
                vals = [v for v in weather_redis["wind_speed_80m"] if pd.notna(v)]
                if vals:
                    avg_wind = sum(vals) / len(vals)
            if avg_solar is None and "shortwave_radiation" in weather_redis:
                vals = [v for v in weather_redis["shortwave_radiation"] if pd.notna(v)]
                if vals:
                    avg_solar = sum(vals) / len(vals)

    # Get backtest MAPE from cache if available
    backtest_mape = None
    backtest_rmse = None
    for horizon in [168, 24, 720]:  # prefer 7-day, then 24h, then 30d
        bt_key = (region, horizon, "xgboost", DEFAULT_BACKTEST_EXOG_MODE)
        if bt_key in _BACKTEST_CACHE:
            cached_result, _, _ = _BACKTEST_CACHE[bt_key]
            if "metrics" in cached_result:
                backtest_mape = cached_result["metrics"].get("mape")
                backtest_rmse = cached_result["metrics"].get("rmse")
                break

    # Fallback: read from Redis if in-memory cache is empty
    if backtest_mape is None:
        for horizon in [168, 24, 720]:
            bt_redis = redis_get(
                f"wattcast:backtest:{DEFAULT_BACKTEST_EXOG_MODE}:{region}:{horizon}"
            )
            if bt_redis is None:
                bt_redis = redis_get(f"wattcast:backtest:{region}:{horizon}")
            if bt_redis and "metrics" in bt_redis:
                xgb_metrics = bt_redis["metrics"].get("xgboost", {})
                if xgb_metrics:
                    backtest_mape = xgb_metrics.get("mape")
                    backtest_rmse = xgb_metrics.get("rmse")
                    break

    # Compute derived metrics
    reserve_margin_pct = (100.0 - pct_of_capacity) if pct_of_capacity is not None else None
    demand_range = (peak_mw - min_mw) if peak_mw is not None and min_mw is not None else None

    # Wind capacity factor (approximate: avg_wind / rated_wind)
    wind_cf = None
    if avg_wind is not None:
        from config import WIND_CUTOUT_SPEED_MPH

        wind_cf = min(avg_wind / WIND_CUTOUT_SPEED_MPH * 100, 100.0)

    # Solar capacity factor (approximate: avg_irradiance / rated_irradiance)
    solar_cf = None
    if avg_solar is not None:
        from config import SOLAR_RATED_IRRADIANCE

        solar_cf = avg_solar / SOLAR_RATED_IRRADIANCE * 100

    # Estimate price from utilization (merit-order approximation)
    from config import PRICING_BASE_USD_MWH

    price_estimate = None
    if pct_of_capacity is not None:
        utilization = pct_of_capacity / 100
        if utilization < 0.70:
            price_estimate = PRICING_BASE_USD_MWH
        elif utilization < 0.90:
            price_estimate = PRICING_BASE_USD_MWH * (1 + (utilization - 0.70) * 5)
        else:
            price_estimate = PRICING_BASE_USD_MWH * (2 + (utilization - 0.90) * 20)

    # Format values
    peak_str = f"{int(peak_mw):,} MW" if peak_mw is not None else "No data"
    avg_str = f"{int(avg_mw):,} MW" if avg_mw is not None else "No data"
    cap_str = f"{pct_of_capacity:.0f}% of capacity" if pct_of_capacity is not None else ""
    mape_str = f"{backtest_mape:.1f}%" if backtest_mape is not None else "No data"
    mape_dir = "positive" if backtest_mape is not None and backtest_mape < 5 else "negative"
    rmse_str = f"{int(backtest_rmse):,} MW" if backtest_rmse is not None else "No data"

    persona_kpis = {
        "grid_ops": [
            {
                "label": "Peak Demand",
                "value": peak_str,
                "delta": cap_str,
                "direction": "negative" if pct_of_capacity and pct_of_capacity > 80 else "neutral",
            },
            {
                "label": "Reserve Margin",
                "value": f"{reserve_margin_pct:.0f}%"
                if reserve_margin_pct is not None
                else "No data",
                "delta": "Below 15% is tight",
                "direction": "negative"
                if reserve_margin_pct is not None and reserve_margin_pct < 15
                else "positive"
                if reserve_margin_pct is not None
                else "neutral",
            },
            {
                "label": "Forecast Error",
                "value": mape_str,
                "delta": f"Walk-forward MAPE ({DEFAULT_BACKTEST_EXOG_MODE})",
                "direction": mape_dir,
            },
            {
                "label": "Demand Range",
                "value": f"{int(demand_range):,} MW" if demand_range is not None else "No data",
                "delta": "Peak - Min",
                "direction": "neutral",
            },
        ],
        "renewables": [
            {
                "label": "Wind CF",
                "value": f"{wind_cf:.0f}%" if wind_cf is not None else "No data",
                "delta": "Capacity factor",
                "direction": "positive" if wind_cf is not None and wind_cf > 25 else "neutral",
            },
            {
                "label": "Solar CF",
                "value": f"{solar_cf:.0f}%" if solar_cf is not None else "No data",
                "delta": "Capacity factor",
                "direction": "positive" if solar_cf is not None and solar_cf > 15 else "neutral",
            },
            {
                "label": "Avg Wind",
                "value": f"{avg_wind:.1f} mph" if avg_wind is not None else "No data",
                "delta": "80m hub height",
                "direction": "neutral",
            },
            {
                "label": "Avg Solar",
                "value": f"{avg_solar:.0f} W/m\u00b2" if avg_solar is not None else "No data",
                "delta": "Shortwave radiation",
                "direction": "neutral",
            },
        ],
        "trader": [
            {
                "label": "Est. Price",
                "value": f"${price_estimate:.0f}/MWh" if price_estimate is not None else "No data",
                "delta": "Merit-order estimate",
                "direction": "negative"
                if price_estimate is not None and price_estimate > 100
                else "neutral",
            },
            {
                "label": "Peak Demand",
                "value": peak_str,
                "delta": cap_str,
                "direction": "neutral",
            },
            {
                "label": "Avg Demand",
                "value": avg_str,
                "delta": f"Range: {int(demand_range):,} MW" if demand_range is not None else "",
                "direction": "neutral",
            },
            {
                "label": "Forecast Error",
                "value": mape_str,
                "delta": f"Walk-forward MAPE ({DEFAULT_BACKTEST_EXOG_MODE})",
                "direction": mape_dir,
            },
        ],
        "data_scientist": [
            {
                "label": "XGBoost MAPE",
                "value": mape_str,
                "delta": f"Target: <5% ({DEFAULT_BACKTEST_EXOG_MODE})",
                "direction": mape_dir,
            },
            {
                "label": "RMSE",
                "value": rmse_str,
                "delta": "Walk-forward backtest",
                "direction": "neutral",
            },
            {
                "label": "Peak Demand",
                "value": peak_str,
                "delta": cap_str,
                "direction": "neutral",
            },
            {
                "label": "Demand Range",
                "value": f"{int(demand_range):,} MW" if demand_range is not None else "No data",
                "delta": "Max variability",
                "direction": "neutral",
            },
        ],
    }
    kpis = persona_kpis.get(persona_id, persona_kpis["grid_ops"])
    return build_kpi_row(kpis)


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
