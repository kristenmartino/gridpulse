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
from datetime import UTC

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import structlog
from dash import ALL, Input, Output, State, ctx, html, no_update

# Shared infrastructure — caches, layout helpers, color tokens, basemap
# constants — lives in ``_callbacks_shared.py``. The explicit re-import
# list below makes every name accessible at
# ``from components.callbacks import <X>`` so the 40+ import sites in
# ``app.py`` and ``tests/`` continue to resolve. Explicit (not star) so
# ``ruff`` can statically verify name resolution everywhere downstream.
# Step 1 of the decomposition tracked in issue #87; per-tab helpers
# move into ``_callbacks_<tab>.py`` modules in follow-up PRs.
# noqa: F401 on the re-exports below — these aren't unused, they're
# the public-import shim. Anything imported from this module via
# ``from components.callbacks import <X>`` resolves through here.
from components._callbacks_alerts import _alerts_tab_from_redis
from components._callbacks_forecast import (
    _add_confidence_bands,
    _add_trailing_actuals,
    _confidence_half_width,  # noqa: F401 — re-export (tests/unit/test_callbacks_helpers.py)
    _create_future_features,  # noqa: F401 — re-export (tests/unit/test_callbacks_helpers.py)
    _outlook_tab_from_redis,
    _run_forecast_outlook,
)
from components._callbacks_generation import (
    _generation_tab_from_redis,  # noqa: F401 — re-export (tests/unit/test_redis_fast_paths.py); fast path is currently orphaned in register_callbacks
)
from components._callbacks_models import (
    _format_metric,
    _get_feature_importance,
    _models_tab_from_redis,
)
from components._callbacks_overview import (
    _build_changes_card,  # noqa: F401 — re-export (tests + helper-callable surface)
    _build_drivers_panel,
    _build_generation_panel,
    _build_models_leaderboard,
    _build_overview_briefing,  # noqa: F401 — re-export (tests + helper-callable surface)
    _build_overview_data_health,  # noqa: F401 — re-export (tests/unit/test_tab_overview.py)
    _build_overview_digest,  # noqa: F401 — re-export (tests/unit/test_tab_overview.py)
    _build_overview_hero_chart,
    _build_overview_insight,
    _build_overview_metrics_items,
    _build_overview_model_card,
    _build_overview_news,  # noqa: F401 — re-export (tests/unit/test_tab_overview.py)
    _build_overview_sparkline,  # noqa: F401 — re-export (tests/unit/test_tab_overview.py)
    _build_overview_spotlight,  # noqa: F401 — re-export (tests/unit/test_tab_overview.py)
    _build_overview_title,
    _build_persona_kpis,  # noqa: F401 — re-export (tests/unit/test_callbacks_helpers.py)
    _build_risk_insight,
    _build_scenarios_panel,
    _build_weather_context,
    _fetch_generation_cached,  # noqa: F401 — re-export (tests/unit/test_callbacks_*)
    _spotlight_model_accuracy,  # noqa: F401 — re-export (tests/unit/test_tab_overview.py)
    _spotlight_renewables,  # noqa: F401 — re-export (tests/unit/test_tab_overview.py)
    _spotlight_trader,  # noqa: F401 — re-export (tests/unit/test_tab_overview.py)
)
from components._callbacks_shared import (
    _BACKTEST_CACHE,
    _CACHE_VERSION,
    _EIA_FUEL_MAP,  # noqa: F401 — re-export (tests/unit/test_callbacks_helpers.py::TestModuleConstants)
    _GENERATION_CACHE,  # noqa: F401 — re-export (test fixture `_clear_module_caches`)
    _MAP_BORDER_COLOR,  # noqa: F401 — re-export (tests/unit/test_us_grid_*)
    _MAP_COLORSCALE,  # noqa: F401 — re-export (tests/unit/test_us_grid_*)
    _MAP_LAND_COLOR,  # noqa: F401 — re-export (tests/unit/test_us_grid_*)
    _MODEL_BAND_COLORS,  # noqa: F401 — re-export (Backtest helpers in callbacks.py call it)
    _MODEL_CACHE,  # noqa: F401 — re-export (test fixture `_clear_module_caches`)
    _PREDICTION_CACHE,  # noqa: F401 — re-export (test fixture `_clear_module_caches`)
    _STRESS_RELIABLE_CEILING,  # noqa: F401 — re-export (tests/unit/test_us_grid_stress_cap)
    BACKTEST_EXOG_MODES,  # noqa: F401 — re-export
    COLORS,
    DEFAULT_BACKTEST_EXOG_MODE,
    PLOT_LAYOUT,  # noqa: F401 — re-export
    PLOT_TEMPLATE,  # noqa: F401 — re-export
    _cache_lock,  # noqa: F401 — re-export (app.py uses for cache stats)
    _collect_backtest_residuals,  # noqa: F401 — re-export (tests + Backtest helpers in callbacks.py still call it)
    _compute_data_hash,
    _empirical_interval_from_backtests,  # noqa: F401 — re-export (tests/unit/test_callbacks_helpers.py)
    _empty_figure,
    _latest_real_demand,  # noqa: F401 — re-export (tests/unit/test_us_grid_nan_guard.py)
    _layout,
)
from components._callbacks_weather import (
    _weather_tab_from_redis,  # noqa: F401 — re-export (tests/unit/test_redis_fast_paths.py); fast path is currently orphaned in register_callbacks
)
from components.accessibility import LINE_STYLES
from components.cards import (
    build_alert_card,
    build_metrics_bar,
    build_model_metrics_card,
    build_page_title,
)
from config import (
    CACHE_TTL_SECONDS,
    EIA_API_KEY,
    REGION_NAMES,
    REQUIRE_REDIS,
    WEATHER_VARIABLES,
)
from data.redis_client import redis_get
from personas.config import get_persona

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

    # ── 2. PERSONA SWITCHING (R3) ─────────────────────────────
    # R3 deleted the standalone ``welcome-card`` and ``kpi-cards`` divs that
    # used to render between the header and the tab strip. Every visible
    # tab now ships its own page-title block + KPI bar; persona switching
    # only redirects the user to the persona's preferred default tab.

    @app.callback(
        Output("dashboard-tabs", "active_tab"),
        Input("persona-selector", "value"),
        prevent_initial_call=True,
    )
    def switch_persona_default_tab(persona_id):
        """Land the user on the persona's default tab when the persona changes.

        Region changes no longer trigger a tab redirect — it's a noisy UX
        when the user is mid-task in a non-default tab.
        """
        if not persona_id:
            return no_update
        persona = get_persona(persona_id)
        return persona.default_tab

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

    # ── 3a-bis. V1.β + V1.γ: US GRID SMALL-MULTIPLES TAB ──────
    # Bird's-eye view of all 16 BAs as a card grid OR a Plotly scatter_geo
    # of BA centroids (Cards | Map toggle). Each card / map point click
    # drills down into the Forecast tab for that region. Reads per-region
    # actuals from Redis (warming/cold regions render an "—" placeholder
    # in cards view; map view drops them).

    @app.callback(
        [
            Output("us-grid-title", "children"),
            Output("us-grid-metrics-bar", "children"),
            Output("us-grid-region-grid", "children"),
        ],
        [
            Input("dashboard-tabs", "active_tab"),
            Input("refresh-interval", "n_intervals"),
            Input("us-grid-view-toggle", "value"),
        ],
    )
    def update_us_grid_snapshot(active_tab, _n_intervals, view):
        """Render the US Grid tab's title, MetricsBar, and body (cards or map)."""
        if active_tab != "tab-us-grid":
            return [no_update] * 3

        view = view or "cards"

        try:
            region_data = _collect_us_grid_region_data()
            title = _build_us_grid_title(region_data)
            metrics_items = _build_us_grid_metrics_items(region_data)
            metrics_bar = build_metrics_bar(metrics_items)
            metrics_bar.className = f"gp-metrics-bar gp-metrics-bar--{len(metrics_items)}up"

            if view == "map":
                body = _build_us_grid_map(region_data)
            elif view == "polygons":
                body = _build_us_grid_choropleth(region_data)
            else:
                # Cards are grouped geographically. Section headers
                # span the full grid row via ``grid-column: 1 / -1`` —
                # see ``.gp-region-grid__section-header`` in custom.css.
                # V3.ζ follow-up: skip codes filtered out by the
                # forecast quality gate (already absent from
                # ``region_data``) so we don't render orphan section
                # headers or placeholder cards for hidden regions.
                from config import REGION_GROUPS

                grid_children: list = []
                for group_name, codes in REGION_GROUPS.items():
                    visible = [c for c in codes if c in region_data]
                    if not visible:
                        continue
                    grid_children.append(
                        html.Div(
                            group_name,
                            className="gp-region-grid__section-header",
                        )
                    )
                    grid_children.extend(
                        _build_us_grid_region_card(code, region_data[code]) for code in visible
                    )
                body = html.Div(grid_children, className="gp-region-grid")

            return (title, metrics_bar, body)
        except Exception as exc:
            log.exception("update_us_grid_snapshot_failed")
            err_msg = f"{type(exc).__name__}: {exc}"
            err_div = html.Div(
                err_msg,
                style={"color": "var(--danger)", "fontSize": "0.8rem", "padding": "8px"},
            )
            return (err_div, html.Div(), err_div)

    @app.callback(
        [
            Output("region-selector", "value", allow_duplicate=True),
            Output("dashboard-tabs", "active_tab", allow_duplicate=True),
        ],
        Input({"type": "us-grid-region-card", "region": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def drilldown_from_us_grid(n_clicks_list):
        """Click a region card → open Forecast tab pre-set to that region."""
        if not n_clicks_list or not any(n for n in n_clicks_list if n):
            return no_update, no_update
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or triggered.get("type") != "us-grid-region-card":
            return no_update, no_update
        region = triggered.get("region")
        if not region:
            return no_update, no_update
        return region, "tab-outlook"

    @app.callback(
        [
            Output("region-selector", "value", allow_duplicate=True),
            Output("dashboard-tabs", "active_tab", allow_duplicate=True),
        ],
        Input("us-grid-map", "clickData"),
        prevent_initial_call=True,
    )
    def drilldown_from_us_grid_map(click_data):
        """Click a map point → open Forecast tab pre-set to that region.

        Mirrors ``drilldown_from_us_grid`` (cards). Same outputs, same
        downstream effect; the map hands the region code through
        ``customdata`` instead of a pattern-matching component ID.

        Tolerates two ``customdata`` shapes so the same callback works
        for both the ``scatter_geo`` view (1-D array of region codes)
        and the V3.β ``Choropleth`` view (2-D array where index 0 is
        the region code, indexes 1-2 carry hover text fields).
        """
        if not click_data:
            return no_update, no_update
        points = click_data.get("points") or []
        if not points:
            return no_update, no_update
        cd = points[0].get("customdata")
        if cd is None:
            return no_update, no_update
        # Choropleth → list/tuple per point; scatter → string per point
        region = cd[0] if isinstance(cd, (list, tuple)) else cd
        if not region:
            return no_update, no_update
        return region, "tab-outlook"

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
        Output("models-title", "children"),
        [
            Input("region-selector", "value"),
            Input("dashboard-tabs", "active_tab"),
        ],
    )
    def update_models_title(region, active_tab):
        """Page title for the Models tab."""
        if active_tab != "tab-models":
            return no_update

        region = region or "FPL"
        region_name = REGION_NAMES.get(region, region)
        return build_page_title(
            "Models",
            f"Forecast accuracy, residuals, and feature importance · {region_name}",
        )

    @app.callback(
        Output("models-leaderboard", "children"),
        [
            Input("region-selector", "value"),
            Input("dashboard-tabs", "active_tab"),
        ],
    )
    def update_models_leaderboard(region, active_tab):
        """5-up MetricsBar leaderboard — one column per model with MAPE."""
        if active_tab != "tab-models":
            return no_update
        return _build_models_leaderboard(region)

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

        from models.model_service import get_forecasts, get_model_metrics

        forecasts = get_forecasts(region, demand_df, selected_models)
        # Read metrics from the same Redis-first source the leaderboard
        # uses (``get_model_metrics``) so the table and the leaderboard
        # always show consistent numbers — even if ``get_forecasts``
        # fell back to fresh simulated forecasts whose computed metrics
        # would otherwise diverge from the Redis payload's metrics.
        metrics = get_model_metrics(region) or forecasts.get("metrics", {})
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
        Output("risk-title", "children"),
        [
            Input("region-selector", "value"),
            Input("dashboard-tabs", "active_tab"),
        ],
    )
    def update_risk_title(region, active_tab):
        """Page-title block for the Risk tab."""
        if active_tab != "tab-alerts":
            return no_update

        region = region or "FPL"
        region_name = REGION_NAMES.get(region, region)
        return build_page_title(
            "Risk",
            f"Active alerts, demand anomalies, and grid stress · {region_name}",
        )

    @app.callback(
        Output("risk-insight-card", "children"),
        [
            Input("dashboard-tabs", "active_tab"),
            Input("region-selector", "value"),
            Input("demand-store", "data"),
            Input("weather-store", "data"),
        ],
    )
    def update_risk_insight(active_tab, region, demand_json, weather_json):
        """Narrative summary for the Risk tab — 'all systems nominal' or
        a 1-sentence elevated-risk note."""
        if active_tab != "tab-alerts":
            return no_update
        return _build_risk_insight(region, demand_json, weather_json)

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

        from components.icons import icon as _icon

        breakdown_items = []
        if n_crit:
            breakdown_items.append(
                html.Div(
                    [
                        _icon(
                            "alert-triangle",
                            size="xs",
                            className="gp-stress-row__icon gp-stress-row__icon--critical",
                        ),
                        html.Span(f"Critical: {n_crit}"),
                    ],
                    className="gp-stress-row gp-stress-row--critical",
                )
            )
        if n_warn:
            breakdown_items.append(
                html.Div(
                    [
                        _icon(
                            "alert-circle",
                            size="xs",
                            className="gp-stress-row__icon gp-stress-row__icon--warning",
                        ),
                        html.Span(f"Warning: {n_warn}"),
                    ],
                    className="gp-stress-row gp-stress-row--warning",
                )
            )
        if n_info:
            breakdown_items.append(
                html.Div(
                    [
                        _icon(
                            "info",
                            size="xs",
                            className="gp-stress-row__icon gp-stress-row__icon--info",
                        ),
                        html.Span(f"Info: {n_info}"),
                    ],
                    className="gp-stress-row gp-stress-row--info",
                )
            )
        if not alerts:
            breakdown_items.append(
                html.Div("No active alerts", className="gp-stress-row gp-stress-row--empty")
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
        Output("fallback-banner", "children"),
        Input("data-freshness-store", "data"),
        prevent_initial_call=True,
    )
    def update_fallback_banner(freshness_json):
        """G2: Show warning banner when data sources are serving stale/fallback data."""
        import json

        from components.icons import icon as _icon

        if not freshness_json:
            return no_update

        freshness = json.loads(freshness_json)
        # Map degraded-source state → (icon name, semantic class, message tail)
        states = {
            "stale": ("alert-circle", "warning", "serving cached data (API unavailable)"),
            "error": ("alert-triangle", "critical", "data load failed — using fallback"),
            "demo": ("flask", "info", "demo data (no API key configured)"),
        }

        rows: list = []
        for source in ("demand", "weather", "alerts"):
            status = freshness.get(source, "fresh")
            entry = states.get(status)
            if entry is None:
                continue
            icon_name, severity, tail = entry
            rows.append(
                html.Div(
                    [
                        _icon(
                            icon_name,
                            size="xs",
                            className=f"gp-fallback-row__icon gp-fallback-row__icon--{severity}",
                        ),
                        html.Span(f"{source.title()}: {tail}"),
                    ],
                    className=f"gp-fallback-row gp-fallback-row--{severity}",
                )
            )

        if not rows:
            return html.Div()

        return dbc.Alert(
            [html.Strong("Data Source Status"), html.Br(), *rows],
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
            Output("outlook-horizon", "value", allow_duplicate=True),
            Output("outlook-model", "value", allow_duplicate=True),
            Output("tab3-model-selector", "value", allow_duplicate=True),
            # NEXD-12: sim sliders
        ],
        Input("url", "search"),
        prevent_initial_call=True,
    )
    def restore_bookmark(search):
        """C2+NEXD-12: Restore full dashboard state from URL query parameters.

        Supports core params (region, persona, tab), tracked filters (f.*),
        and scenario slider values (s.*).  Old URLs without f.*/s.* still work.
        """
        n_outputs = 6  # 3 core + 3 filters (V2.1: dropped 4 hidden filters + 5 sliders)
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
            State("outlook-horizon", "value"),
            State("outlook-model", "value"),
            State("tab3-model-selector", "value"),
            # NEXD-12: sim sliders
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
            Output("outlook-horizon", "value", allow_duplicate=True),
            Output("outlook-model", "value", allow_duplicate=True),
            Output("tab3-model-selector", "value", allow_duplicate=True),
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

        n_outputs = 6  # V2.1: was 10 (3 core + 7 filters); now 3 core + 3 filters
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
            Input("outlook-horizon", "value"),
            Input("outlook-model", "value"),
            Input("tab3-model-selector", "value"),
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

    # ── SPRINT 5: C9 — BRIEFING MODE (R5c) ─────────────────────
    # Clientside mirror — observes meeting-mode-store and toggles
    # `body.briefing` so the CSS rules in assets/custom.css can hide
    # the tab strip, scale type, and stamp the watermark. Also
    # rewrites the toggle button's label between "Briefing Mode" and
    # "Exit Briefing" so the user has a clear way out.
    app.clientside_callback(
        """
        function(mode) {
            const isOn = mode === 'true';
            document.body.classList.toggle('briefing', isOn);
            return isOn ? 'Exit Briefing' : 'Briefing Mode';
        }
        """,
        Output("meeting-mode-btn", "children"),
        Input("meeting-mode-store", "data"),
    )

    @app.callback(
        [
            Output("meeting-mode-store", "data"),
            Output("dashboard-header", "className"),
            Output("widget-confidence-bar", "style"),
            Output("fallback-banner", "style"),
        ],
        Input("meeting-mode-btn", "n_clicks"),
        State("meeting-mode-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_meeting_mode(n_clicks, current_mode):
        """C9: Toggle meeting-ready (Briefing Mode) projection chrome.

        Strips navigation, freshness banner, and confidence-bar carriers.
        Charts and narrative remain. The clientside callback above mirrors
        meeting-mode-store onto body.briefing for the v2 projection CSS.
        """
        is_meeting = current_mode != "true"
        new_mode = "true" if is_meeting else "false"

        if is_meeting:
            header_class = "dashboard-header gp-header meeting-mode"
            confidence_style = {"display": "none"}
            banner_style = {"display": "none"}
        else:
            header_class = "dashboard-header gp-header"
            confidence_style = {}
            banner_style = {}

        return new_mode, header_class, confidence_style, banner_style

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
        formatted = {
            "MAPE": f"{m.get('mape', 0.0):.1f}%",
            "RMSE": f"{m.get('rmse', 0.0):,.0f} MW",
            "MAE": f"{m.get('mae', 0.0):,.0f} MW",
            "R²": f"{m.get('r2', 0.0):.3f}",
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


# ── HELPER FUNCTIONS ──────────────────────────────────────────


# ── Defensive readers ──────────────────────────────────────────
#
# EIA-930 has a publishing lag for the most recent hour, especially for
# newer / smaller BAs (PSCO, NEVP, AZPS observed 2026-05-02). Until the
# row catches up the demand series ends with NaN — sometimes preceded by
# zero (EIA's other "missing observation" marker). Anywhere we read the
# most recent value off a demand list or Series we route through this
# helper so downstream code never has to ``isnan`` or guard ``prev / 0``.
#
# Companion to the ``> 0`` filter in ``_build_overview_metrics_items`` —
# this formalizes that pattern as the single source of truth, so future
# tabs (US Grid, Risk, Models) can't accidentally re-introduce the bug.


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


# ── V1.β: US GRID TAB HELPERS ────────────────────────────────
#
# Extracted into ``components/_callbacks_us_grid.py`` per issue #87.
# The explicit re-import block below preserves the public-import surface
# so existing ``from components.callbacks import <X>`` sites
# (notably the US Grid test files) continue to resolve unchanged.

from components._callbacks_us_grid import (  # noqa: E402
    _BA_POLYGONS_CACHE,  # noqa: F401 — re-export
    _build_interchange_chip,  # noqa: F401 — re-export
    _build_us_grid_choropleth,  # noqa: F401 — re-export
    _build_us_grid_map,  # noqa: F401 — re-export
    _build_us_grid_metrics_items,  # noqa: F401 — re-export
    _build_us_grid_region_card,  # noqa: F401 — re-export
    _build_us_grid_sparkline,  # noqa: F401 — re-export
    _build_us_grid_title,  # noqa: F401 — re-export
    _collect_us_grid_region_data,  # noqa: F401 — re-export
    _is_real_positive,  # noqa: F401 — re-export
    _load_ba_polygons,  # noqa: F401 — re-export
)
