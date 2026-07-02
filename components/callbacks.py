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
from components._callbacks_alerts import (
    _alerts_tab_from_redis,  # noqa: F401 — re-export (callback now lives in _callbacks_alerts)
    register_alerts_callbacks,
)
from components._callbacks_backtest import (
    _backtest_tab_from_redis,  # noqa: F401 — re-export (tests/unit/test_redis_fast_paths.py)
    _build_forecast_exog_fold,  # noqa: F401 — re-export (tests + register_callbacks symmetry)
    _describe_exog_mode,  # noqa: F401 — re-export (tests/unit/test_callbacks_*)
    _ensemble_fold,  # noqa: F401 — re-export (tests/unit/test_callbacks_v1_paths.py)
    _normalize_backtest_exog_mode,  # noqa: F401 — re-export (tests)
    _predict_single_fold,  # noqa: F401 — re-export (tests/unit/test_callbacks_v1_paths.py)
    _run_backtest_for_horizon,  # noqa: F401 — re-export (tests/unit/test_callbacks_*)
)
from components._callbacks_forecast import (
    _add_confidence_bands,  # noqa: F401 — re-export (callback now lives in _callbacks_forecast)
    _add_forecast_horizon_divider,  # noqa: F401 — re-export (ADR-008 day-16 marker)
    _add_trailing_actuals,  # noqa: F401 — re-export (callback now lives in _callbacks_forecast)
    _confidence_half_width,  # noqa: F401 — re-export (tests/unit/test_callbacks_helpers.py)
    _create_future_features,  # noqa: F401 — re-export (tests/unit/test_callbacks_helpers.py)
    _outlook_tab_from_redis,  # noqa: F401 — re-export (callback now lives in _callbacks_forecast)
    _run_forecast_outlook,  # noqa: F401 — re-export (callback now lives in _callbacks_forecast)
    register_forecast_callbacks,
)
from components._callbacks_generation import (
    _generation_tab_from_redis,  # noqa: F401 — re-export (tests/unit/test_redis_fast_paths.py); fast path is currently orphaned in register_callbacks
)
from components._callbacks_models import (
    _build_drift_panel,  # noqa: F401 — re-export (tests/unit/test_drift_panel.py — #121 part 2)
    _format_metric,  # noqa: F401 — re-export (callback now lives in _callbacks_models)
    _get_feature_importance,  # noqa: F401 — re-export (callback now lives in _callbacks_models)
    _models_tab_from_redis,  # noqa: F401 — re-export (callback now lives in _callbacks_models)
    register_models_callbacks,
)
from components._callbacks_overview import (
    _build_changes_card,  # noqa: F401 — re-export (tests + helper-callable surface)
    _build_models_leaderboard,  # noqa: F401 — re-export (tests/unit/test_models_tab_consistency.py)
    _build_overview_briefing,  # noqa: F401 — re-export (tests + helper-callable surface)
    _build_overview_data_health,  # noqa: F401 — re-export (tests/unit/test_tab_overview.py)
    _build_overview_digest,  # noqa: F401 — re-export (tests/unit/test_tab_overview.py)
    _build_overview_hero_chart,  # noqa: F401 — re-export (callback now lives in _callbacks_overview)
    _build_overview_insight,  # noqa: F401 — re-export (callback now lives in _callbacks_overview)
    _build_overview_metrics_items,  # noqa: F401 — re-export (tests/unit/test_overview_metrics_nan_guard.py)
    _build_overview_model_card,  # noqa: F401 — re-export (callback now lives in _callbacks_overview)
    _build_overview_news,  # noqa: F401 — re-export (tests/unit/test_tab_overview.py)
    _build_overview_sparkline,  # noqa: F401 — re-export (tests/unit/test_tab_overview.py)
    _build_overview_spotlight,  # noqa: F401 — re-export (tests/unit/test_tab_overview.py)
    _build_overview_title,  # noqa: F401 — re-export (callback now lives in _callbacks_overview)
    _build_persona_kpis,  # noqa: F401 — re-export (tests/unit/test_callbacks_helpers.py)
    _fetch_generation_cached,  # noqa: F401 — re-export (tests/unit/test_callbacks_*)
    _spotlight_model_accuracy,  # noqa: F401 — re-export (tests/unit/test_tab_overview.py)
    _spotlight_renewables,  # noqa: F401 — re-export (tests/unit/test_tab_overview.py)
    _spotlight_trader,  # noqa: F401 — re-export (tests/unit/test_tab_overview.py)
    register_overview_callbacks,
)
from components._callbacks_shared import (
    _BACKTEST_CACHE,  # noqa: F401 — re-export (test fixture `_clear_module_caches`)
    _CACHE_VERSION,  # noqa: F401 — re-export (tests/unit/test_callbacks_v1_paths.py)
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
    COLORS,  # noqa: F401 — re-export (tests/unit/test_callbacks_helpers.py::TestModuleConstants)
    DEFAULT_BACKTEST_EXOG_MODE,  # noqa: F401 — re-export (tests + Backtest module imports it elsewhere)
    PLOT_CONFIG,  # noqa: F401 — re-export
    PLOT_LAYOUT,  # noqa: F401 — re-export
    PLOT_TEMPLATE,  # noqa: F401 — re-export
    _cache_lock,  # noqa: F401 — re-export (app.py uses for cache stats)
    _collect_backtest_residuals,  # noqa: F401 — re-export (tests)
    _compute_data_hash,  # noqa: F401 — re-export (tests/unit/test_callbacks_*)
    _empirical_interval_from_backtests,  # noqa: F401 — re-export (tests/unit/test_callbacks_helpers.py)
    _empty_figure,  # noqa: F401 — re-export (tests/unit/test_callbacks_helpers.py::TestEmptyFigure)
    _latest_real_demand,  # noqa: F401 — re-export (tests/unit/test_us_grid_nan_guard.py)
    _layout,  # noqa: F401 — re-export (test trace inspection)
)
from components._callbacks_weather import (
    _weather_tab_from_redis,  # noqa: F401 — re-export (tests/unit/test_redis_fast_paths.py); fast path is currently orphaned in register_callbacks
)
from config import (
    EIA_API_KEY,
    FRESHNESS_DEMAND_LAG_ALLOWANCE_HOURS,
    FRESHNESS_FRESH_MAX_AGE_HOURS,
    REGION_NAMES,  # noqa: F401 — re-export (tests/unit/test_forecast_quality_gate.py patches this)
    REQUIRE_REDIS,
)
from data.redis_client import redis_get, redis_key
from personas.config import get_persona

log = structlog.get_logger()


def _freshness_from_payload(payload, now) -> str | None:
    """Measure a payload's freshness from its own ``scored_at`` stamp.

    Returns ``"fresh"``/``"stale"``, or ``None`` when the payload carries no
    (parseable) stamp — callers then apply a payload-specific fallback.
    Freshness must be measured, never asserted at render time (2026-07
    critical-review finding P1-3).
    """
    if not isinstance(payload, dict):
        return None
    iso = payload.get("scored_at")
    if not iso:
        return None
    try:
        scored = pd.Timestamp(iso)
        if scored.tzinfo is None:
            scored = scored.tz_localize("UTC")
    except (ValueError, TypeError):
        return None
    age_h = (now - scored).total_seconds() / 3600.0
    return "fresh" if age_h <= FRESHNESS_FRESH_MAX_AGE_HOURS else "stale"


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

    cached_actuals = redis_get(redis_key(f"actuals:{region}"))
    cached_weather = redis_get(redis_key(f"weather:{region}"))
    if cached_actuals is None or cached_weather is None:
        return None

    pipe = PipelineLogger("load_data", region=region)
    now = pd.Timestamp.now(tz="UTC")
    # Statuses are filled in below from each payload's own age — the
    # defaults here are the pessimistic values for payloads we can't date.
    freshness = {
        "demand": "stale",
        "weather": "stale",
        "alerts": "warming",
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
    weather_cols = {k: v for k, v in cached_weather.items() if k not in ("region", "scored_at")}
    weather_df = pd.DataFrame(weather_cols)
    if "timestamps" in weather_df.columns:
        weather_df = weather_df.rename(columns={"timestamps": "timestamp"})

    # Measured freshness (P1-3). Payloads carry scored_at since 2026-07;
    # legacy demand payloads fall back to the age of the newest observation
    # with EIA publishing-lag allowance (~1-4h is normal; see #129).
    demand_status = _freshness_from_payload(cached_actuals, now)
    if demand_status is None and len(demand_df) > 0:
        last_obs = pd.Timestamp(demand_df["timestamp"].iloc[-1])
        if last_obs.tzinfo is None:
            last_obs = last_obs.tz_localize("UTC")
        age_h = (now - last_obs).total_seconds() / 3600.0
        demand_status = "fresh" if age_h <= FRESHNESS_DEMAND_LAG_ALLOWANCE_HOURS else "stale"
    freshness["demand"] = demand_status or "stale"
    freshness["weather"] = _freshness_from_payload(cached_weather, now) or freshness["demand"]

    cached_alerts = redis_get(redis_key(f"alerts:{region}"))
    if isinstance(cached_alerts, dict):
        alerts_src = cached_alerts.get("alerts_source", "demo")
        if alerts_src in ("unavailable", "demo"):
            freshness["alerts"] = alerts_src
        else:
            freshness["alerts"] = _freshness_from_payload(cached_alerts, now) or "fresh"

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
    # mission-control stack. Step 10a of the register_callbacks split
    # (issue #87) moves the decorator block into ``_callbacks_overview.py``
    # so the Overview tab's read path is end-to-end inside that module.
    register_overview_callbacks(app)

    # ── 3a-bis. V1.β + V1.γ: US GRID SMALL-MULTIPLES TAB ──────
    # Step 10b of the register_callbacks split (issue #87) moves the
    # three US-Grid callbacks (snapshot + card drilldown + map drilldown)
    # into ``_callbacks_us_grid.py`` so the tab's read path is end-to-end
    # inside that module.
    register_us_grid_callbacks(app)

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

    # ── 4. MODELS / DIAGNOSTICS TAB ────────────────────────────
    # Step 10c of the register_callbacks split (issue #87) moves the
    # three Models tab callbacks into ``_callbacks_models.py``. The
    # section header was historically labeled "TAB 1: DEMAND FORECAST"
    # — corrected here to match the actual tab the callbacks update.
    register_models_callbacks(app)

    # ── 5. RISK / ALERTS TAB ──────────────────────────────────
    # Step 10d of the register_callbacks split (issue #87) moves the
    # three Risk tab callbacks (title, insight, the 8-output
    # ``update_alerts_tab`` with v1 demo-data fallback) into
    # ``_callbacks_alerts.py``. The previous section header
    # ``# ── 7. TAB 4: GENERATION & NET LOAD ──`` was a stale breadcrumb
    # from an earlier round of refactoring that deleted the Generation
    # callbacks but left the comment; the outputs in that block were
    # actually all ``tab5-*`` (Risk tab) ids.
    register_alerts_callbacks(app)

    # ── 6. FALLBACK BANNER (cross-cutting — degraded data sources) ────
    # The previous section header ``# ── 9. TAB 6: SCENARIO SIMULATOR ──``
    # was another stale breadcrumb. The block below renders the
    # fallback banner shown when data sources are degraded — it's
    # cross-cutting (visible on every tab) so stays in callbacks.py.

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

    # ── FORECAST TAB (R4a + Replay) ─────────────────────────────
    # Step 10c of the register_callbacks split (issue #87) moves the
    # 9 Forecast tab callbacks (3 panel toggles + 3 inline panels +
    # scenario presets + title + model card + the big 9-output
    # ``update_demand_outlook`` + 2 Forecast Replay callbacks) into
    # ``_callbacks_forecast.py``.
    register_forecast_callbacks(app)


# ── HELPER FUNCTIONS ──────────────────────────────────────────
#
# Per the callbacks.py decomposition (issue #87) every tab-specific
# helper now lives in its corresponding ``_callbacks_<tab>.py`` module.
# Cross-cutting utilities (caches, layout, prediction intervals,
# defensive readers) live in ``_callbacks_shared.py``. The explicit
# re-imports at the top of this file expose every helper under the
# ``components.callbacks`` namespace so existing import sites
# (``from components.callbacks import <X>``) continue to resolve.


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
    register_us_grid_callbacks,
)
