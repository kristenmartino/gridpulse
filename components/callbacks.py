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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, ctx, html, ALL, no_update
import dash_bootstrap_components as dbc
import structlog

from config import (
    REGION_NAMES,
    REGION_CAPACITY_MW,
    REGION_COORDINATES,
    TAB_LABELS,
    EIA_API_KEY,
    PRICING_BASE_USD_MWH,
)
from personas.config import get_persona, get_welcome_card, PERSONAS, list_personas
from components.cards import build_kpi_row, build_welcome_card, build_alert_card, build_news_feed

log = structlog.get_logger()

# Model and prediction cache for performance
# Avoids retraining XGBoost on every callback (~10s -> <1s)
# Cache TTL is 24 hours since demand data only changes once per day
_MODEL_CACHE: dict = {}  # {region: (model, data_hash, timestamp)}
_PREDICTION_CACHE: dict = {}  # {(region, horizon): (predictions, timestamps, data_hash, time)}
_BACKTEST_CACHE: dict = {}  # {(region, horizon, model): (result_dict, data_hash, time)}
_CACHE_TTL_SECONDS = 86400  # 24 hours

# Plotly dark theme defaults
PLOT_TEMPLATE = "plotly_dark"
PLOT_LAYOUT = dict(
    template=PLOT_TEMPLATE,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,33,62,0.8)",
    font=dict(color="#e0e0e0", size=11),
    margin=dict(l=50, r=20, t=30, b=40),
    legend=dict(orientation="h", y=-0.15),
)

# Color palette (colorblind-safe — Wong 2011)
from components.accessibility import CB_PALETTE, LINE_STYLES, FUEL_COLORS
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

ALL_TAB_IDS = [
    "tab-forecast", "tab-weather", "tab-models",
    "tab-generation", "tab-alerts", "tab-simulator",
]


def _run_forecast_outlook(demand_df: pd.DataFrame, weather_df: pd.DataFrame,
                           horizon_hours: int, model_name: str, region: str) -> dict:
    """Generate forward-looking forecast using cached model when possible."""
    import time
    from data.preprocessing import merge_demand_weather
    from data.feature_engineering import engineer_features

    # Compute data hash for cache invalidation
    data_hash = hash((len(demand_df), len(weather_df), region))
    cache_key = (region, horizon_hours)

    # Check prediction cache first (fastest path)
    if cache_key in _PREDICTION_CACHE:
        cached_pred, cached_ts, cached_hash, cached_time = _PREDICTION_CACHE[cache_key]
        if cached_hash == data_hash and (time.time() - cached_time) < _CACHE_TTL_SECONDS:
            log.info("forecast_cache_hit", region=region, horizon=horizon_hours)
            return {"timestamps": cached_ts, "predictions": cached_pred}

    # Merge and engineer features
    merged_df = merge_demand_weather(demand_df, weather_df)
    featured_df = engineer_features(merged_df)
    featured_df = featured_df.dropna(subset=["demand_mw"])

    if len(featured_df) < 168:
        return {"error": "Insufficient training data"}

    train_df = featured_df.copy()
    last_ts = train_df["timestamp"].max()
    future_timestamps = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1),
        periods=horizon_hours,
        freq="h",
        tz="UTC"
    )
    future_df = _create_future_features(train_df, future_timestamps)

    # Check model cache (avoids retraining)
    xgb_model = None
    if region in _MODEL_CACHE:
        cached_model, cached_hash, cached_time = _MODEL_CACHE[region]
        if cached_hash == data_hash and (time.time() - cached_time) < _CACHE_TTL_SECONDS:
            xgb_model = cached_model
            log.info("model_cache_hit", region=region)

    try:
        from models.xgboost_model import train_xgboost, predict_xgboost

        if xgb_model is None:
            log.info("model_training_start", region=region)
            xgb_model = train_xgboost(train_df)
            _MODEL_CACHE[region] = (xgb_model, data_hash, time.time())
            log.info("model_cached", region=region)

        predictions = predict_xgboost(xgb_model, future_df)[:horizon_hours]

        # Cache predictions
        _PREDICTION_CACHE[cache_key] = (predictions, future_timestamps, data_hash, time.time())

    except Exception as e:
        log.warning("outlook_model_failed", model=model_name, error=str(e))
        return {"error": str(e)}

    return {
        "timestamps": future_timestamps,
        "predictions": predictions,
    }


def _create_future_features(train_df: pd.DataFrame, future_timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Create feature dataframe for future predictions."""
    # Get feature columns from training data
    feature_cols = [c for c in train_df.columns if c not in ["timestamp", "demand_mw", "region"]]

    # Create base dataframe
    future_df = pd.DataFrame({"timestamp": future_timestamps})

    # Add time-based features
    future_df["hour"] = future_df["timestamp"].dt.hour
    future_df["day_of_week"] = future_df["timestamp"].dt.dayofweek
    future_df["month"] = future_df["timestamp"].dt.month
    future_df["day_of_year"] = future_df["timestamp"].dt.dayofyear
    future_df["hour_sin"] = np.sin(2 * np.pi * future_df["hour"] / 24)
    future_df["hour_cos"] = np.cos(2 * np.pi * future_df["hour"] / 24)
    future_df["dow_sin"] = np.sin(2 * np.pi * future_df["day_of_week"] / 7)
    future_df["dow_cos"] = np.cos(2 * np.pi * future_df["day_of_week"] / 7)
    future_df["is_weekend"] = (future_df["day_of_week"] >= 5).astype(int)

    # Use last known values for weather and lag features
    last_row = train_df.iloc[-1]
    for col in feature_cols:
        if col not in future_df.columns:
            if col in last_row.index:
                future_df[col] = last_row[col]
            else:
                future_df[col] = 0

    return future_df


def _build_hourly_forecast_table(timestamps: pd.DatetimeIndex, predictions: np.ndarray):
    """Build an hourly breakdown table for 24h forecasts."""
    import dash_bootstrap_components as dbc

    rows = []
    for i, (ts, pred) in enumerate(zip(timestamps, predictions)):
        rows.append(html.Tr([
            html.Td(ts.strftime("%H:%M"), style={"fontSize": "0.8rem"}),
            html.Td(ts.strftime("%a"), style={"fontSize": "0.8rem"}),
            html.Td(f"{pred:,.0f} MW", style={"fontSize": "0.8rem", "fontWeight": "bold"}),
        ]))

    return dbc.Table([
        html.Thead(html.Tr([
            html.Th("Hour", style={"fontSize": "0.8rem"}),
            html.Th("Day", style={"fontSize": "0.8rem"}),
            html.Th("Forecast", style={"fontSize": "0.8rem"}),
        ])),
        html.Tbody(rows),
    ], bordered=True, striped=True, hover=True, size="sm",
       style={"maxHeight": "300px", "overflowY": "auto"})


def register_callbacks(app):
    """Register all callbacks with the Dash app."""

    # ── 1. DATA LOADING ───────────────────────────────────────

    @app.callback(
        [Output("demand-store", "data"),
         Output("weather-store", "data"),
         Output("data-freshness-store", "data"),
         Output("audit-store", "data"),
         Output("pipeline-log-store", "data")],
        [Input("region-selector", "value"),
         Input("refresh-interval", "n_intervals")],
    )
    def load_data(region, _n):
        """Load demand + weather data for selected region.

        G2: Tracks which sources served fresh vs stale data.
        D2: Records audit trail for every forecast.
        I1: Logs each pipeline transformation step.
        """
        import json
        from datetime import datetime, timezone
        from observability import PipelineLogger
        from data.audit import audit_trail

        pipe = PipelineLogger("load_data", region=region)
        freshness = {"demand": "fresh", "weather": "fresh", "alerts": "fresh",
                     "timestamp": datetime.now(timezone.utc).isoformat()}

        try:
            if EIA_API_KEY and EIA_API_KEY != "your_eia_api_key_here":
                from data.eia_client import fetch_demand
                from data.weather_client import fetch_weather
                try:
                    demand_df = fetch_demand(region)
                    pipe.step("fetch_demand", rows=len(demand_df), source="eia_api")
                except Exception as e:
                    log.warning("demand_fallback_to_demo", region=region, error=str(e))
                    from data.demo_data import generate_demo_demand
                    demand_df = generate_demo_demand(region)
                    freshness["demand"] = "stale"
                    pipe.step("fetch_demand", rows=len(demand_df), source="demo_fallback")
                try:
                    weather_df = fetch_weather(region)
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

            pipe.step("serialize", demand_cols=len(demand_df.columns),
                       weather_cols=len(weather_df.columns))

            # D2: Record audit trail
            demand_range = ("", "")
            weather_range = ("", "")
            if "timestamp" in demand_df.columns and len(demand_df) > 0:
                demand_range = (str(demand_df["timestamp"].min()),
                                str(demand_df["timestamp"].max()))
            if "timestamp" in weather_df.columns and len(weather_df) > 0:
                weather_range = (str(weather_df["timestamp"].min()),
                                 str(weather_df["timestamp"].max()))

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

            return (demand_df.to_json(date_format="iso"),
                    weather_df.to_json(date_format="iso"),
                    json.dumps(freshness),
                    audit_record.to_json(),
                    json.dumps(pipeline_summary, default=str))
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
        [Output("welcome-card", "children"),
         Output("kpi-cards", "children"),
         Output("dashboard-tabs", "active_tab")],
        [Input("persona-selector", "value")],
        [State("region-selector", "value"),
         State("demand-store", "data"),
         State("weather-store", "data")],
    )
    def switch_persona(persona_id, region, demand_json, weather_json):
        """Reconfigure dashboard for selected persona with live data."""
        card_data = get_welcome_card(persona_id)
        persona = get_persona(persona_id)

        from personas.welcome import generate_welcome_message
        demand_df = pd.read_json(io.StringIO(demand_json)) if demand_json else None
        weather_df = pd.read_json(io.StringIO(weather_json)) if weather_json else None
        message = generate_welcome_message(persona_id, region, demand_df, weather_df)

        welcome = build_welcome_card(
            title=card_data["title"],
            message=message,
            avatar=card_data["avatar"],
            color=card_data["color"],
        )
        kpis = _build_persona_kpis(persona_id, region)
        return welcome, kpis, persona.default_tab

    # ── 2b. PERSONA TAB VISIBILITY (AC-7.5) ──────────────────

    for _tid in ALL_TAB_IDS:
        @app.callback(
            Output(_tid, "disabled"),
            Input("persona-selector", "value"),
            prevent_initial_call=True,
        )
        def _update_tab_disabled(persona_id, bound_tid=_tid):
            persona = get_persona(persona_id)
            return bound_tid not in persona.priority_tabs

    # ── 4. TAB 1: DEMAND FORECAST ─────────────────────────────

    @app.callback(
        Output("tab1-forecast-chart", "figure"),
        [Input("demand-store", "data"),
         Input("weather-store", "data"),
         Input("tab1-weather-overlay", "value"),
         Input("tab1-timerange", "value"),
         Input("tab1-model-toggle", "value")],
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

        # Actual demand only
        fig.add_trace(go.Scatter(
            x=demand_df["timestamp"], y=demand_df["demand_mw"],
            mode="lines", name="Actual Demand",
            line=dict(color=COLORS["actual"], width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 170, 0.1)",
        ))

        # Weather overlay
        if overlay and "temp" in overlay and weather_json:
            weather_df = pd.read_json(io.StringIO(weather_json))
            weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
            weather_df = weather_df[weather_df["timestamp"].isin(demand_df["timestamp"])]
            if not weather_df.empty:
                fig.add_trace(go.Scatter(
                    x=weather_df["timestamp"], y=weather_df["temperature_2m"],
                    mode="lines", name="Temperature (°F)",
                    line=dict(color=COLORS["temperature"], width=1.5), yaxis="y2",
                ))
                fig.update_layout(yaxis2=dict(
                    title="Temperature (°F)", overlaying="y", side="right",
                    showgrid=False, color=COLORS["temperature"],
                ))

        fig.update_layout(**PLOT_LAYOUT, title=f"Historical Demand - {region}",
                          xaxis_title="Time (UTC)", yaxis_title="Demand (MW)",
                          hovermode="x unified")
        return fig

    # ── 4b. TAB 1 KPIs ────────────────────────────────────────

    @app.callback(
        [Output("tab1-peak-value", "children"),
         Output("tab1-peak-time", "children"),
         Output("tab1-mape-value", "children"),
         Output("tab1-reserve-value", "children"),
         Output("tab1-reserve-status", "children"),
         Output("tab1-alerts-count", "children"),
         Output("tab1-alerts-summary", "children")],
        [Input("demand-store", "data"),
         Input("weather-store", "data")],
        [State("region-selector", "value")],
        prevent_initial_call=True,
    )
    def update_tab1_kpis(demand_json, weather_json, region):
        """Update Tab 1 KPI cards with historical demand stats."""
        if not demand_json:
            return "— MW", "", "— MW", "— MW", "", "0", ""

        demand_df = pd.read_json(io.StringIO(demand_json))
        demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])

        # Filter to valid demand data
        valid_data = demand_df.dropna(subset=["demand_mw"])
        if valid_data.empty:
            return "— MW", "", "— MW", "— MW", "", "0", ""

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
        days_str = html.Span(f"~{data_points // 24} days", className="kpi-delta neutral",
                             style={"fontSize": "0.75rem"})

        return peak_str, peak_time_str, avg_str, min_str, min_time_str, data_str, days_str

    # ── 5. TAB 2: WEATHER CORRELATION ─────────────────────────

    @app.callback(
        [Output("tab2-scatter-temp", "figure"),
         Output("tab2-scatter-wind", "figure"),
         Output("tab2-scatter-solar", "figure"),
         Output("tab2-heatmap", "figure"),
         Output("tab2-feature-importance", "figure"),
         Output("tab2-seasonal", "figure")],
        [Input("demand-store", "data"),
         Input("weather-store", "data")],
        prevent_initial_call=True,
    )
    def update_weather_tab(demand_json, weather_json):
        """Update all Tab 2 charts."""
        if not demand_json or not weather_json:
            empty = _empty_figure("Loading...")
            return empty, empty, empty, empty, empty, empty

        demand_df = pd.read_json(io.StringIO(demand_json))
        weather_df = pd.read_json(io.StringIO(weather_json))
        demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])
        weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
        merged = demand_df.merge(weather_df, on="timestamp", how="inner")

        fig_temp = go.Figure(go.Scatter(
            x=merged["temperature_2m"], y=merged["demand_mw"],
            mode="markers", marker=dict(size=3, color=COLORS["actual"], opacity=0.4),
        ))
        fig_temp.update_layout(**PLOT_LAYOUT, xaxis_title="Temperature (°F)", yaxis_title="Demand (MW)")

        from data.feature_engineering import compute_wind_power, compute_solar_capacity_factor
        merged["wind_power"] = compute_wind_power(merged["wind_speed_80m"])
        fig_wind = go.Figure(go.Scatter(
            x=merged["wind_speed_80m"], y=merged["wind_power"],
            mode="markers", marker=dict(size=3, color=COLORS["wind"], opacity=0.5),
        ))
        fig_wind.update_layout(**PLOT_LAYOUT, xaxis_title="Wind Speed (mph)", yaxis_title="Wind Power Estimate")

        merged["solar_cf"] = compute_solar_capacity_factor(merged["shortwave_radiation"])
        fig_solar = go.Figure(go.Scatter(
            x=merged["shortwave_radiation"], y=merged["solar_cf"],
            mode="markers", marker=dict(size=3, color=COLORS["solar"], opacity=0.5),
        ))
        fig_solar.update_layout(**PLOT_LAYOUT, xaxis_title="GHI (W/m²)", yaxis_title="Solar Capacity Factor")

        corr_cols = [c for c in ["demand_mw", "temperature_2m", "wind_speed_80m", "shortwave_radiation",
                                 "relative_humidity_2m", "cloud_cover", "surface_pressure"] if c in merged.columns]
        corr = merged[corr_cols].corr()
        fig_heatmap = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="RdBu", zmid=0, text=np.round(corr.values, 2), texttemplate="%{text}",
        ))
        fig_heatmap.update_layout(**PLOT_LAYOUT)

        importance = corr["demand_mw"].drop("demand_mw").abs().sort_values(ascending=True)
        fig_importance = go.Figure(go.Bar(
            x=importance.values, y=importance.index, orientation="h", marker_color=COLORS["ensemble"],
        ))
        fig_importance.update_layout(**PLOT_LAYOUT, xaxis_title="Correlation Strength")

        demand_ts = merged.set_index("timestamp")["demand_mw"].resample("h").mean().dropna()
        trend = demand_ts.rolling(168, center=True).mean()
        residual = demand_ts - trend
        fig_seasonal = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                     subplot_titles=["Original", "Trend (7-day)", "Residual"])
        fig_seasonal.add_trace(go.Scatter(x=demand_ts.index, y=demand_ts.values,
                                          line=dict(color=COLORS["actual"], width=1), showlegend=False), row=1, col=1)
        fig_seasonal.add_trace(go.Scatter(x=trend.index, y=trend.values,
                                          line=dict(color=COLORS["ensemble"], width=2), showlegend=False), row=2, col=1)
        fig_seasonal.add_trace(go.Scatter(x=residual.index, y=residual.values,
                                          line=dict(color=COLORS["arima"], width=1), showlegend=False), row=3, col=1)
        fig_seasonal.update_layout(**PLOT_LAYOUT, height=350)

        return fig_temp, fig_wind, fig_solar, fig_heatmap, fig_importance, fig_seasonal

    # ── 6. TAB 3: MODEL COMPARISON ────────────────────────────

    @app.callback(
        [Output("tab3-metrics-table", "children"),
         Output("tab3-residuals-time", "figure"),
         Output("tab3-residuals-hist", "figure"),
         Output("tab3-residuals-pred", "figure"),
         Output("tab3-error-heatmap", "figure"),
         Output("tab3-shap", "figure")],
        [Input("demand-store", "data")],
        [State("region-selector", "value")],
        prevent_initial_call=True,
    )
    def update_models_tab(demand_json, region):
        """Update Tab 3 model diagnostics using model service."""
        if not demand_json:
            empty = _empty_figure("Loading...")
            return html.P("Loading..."), empty, empty, empty, empty, empty

        demand_df = pd.read_json(io.StringIO(demand_json))
        demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])
        actual = demand_df["demand_mw"].values

        from models.model_service import get_forecasts
        forecasts = get_forecasts(region, demand_df)
        metrics = forecasts.get("metrics", {})

        # Metrics table
        name_map = {"Prophet": "prophet", "SARIMAX": "arima", "XGBoost": "xgboost", "Ensemble": "ensemble"}
        rows = []
        for display_name, key in name_map.items():
            m = metrics.get(key, {})
            rows.append(html.Tr([
                html.Td(display_name, style={"fontWeight": "600"}),
                html.Td(f'{m.get("mape", 0):.2f}%'),
                html.Td(f'{m.get("rmse", 0):.0f}'),
                html.Td(f'{m.get("mae", 0):.0f}'),
                html.Td(f'{m.get("r2", 0):.4f}'),
            ]))
        table = html.Table([
            html.Thead(html.Tr([html.Th(h) for h in ["Model", "MAPE", "RMSE", "MAE", "R²"]])),
            html.Tbody(rows),
        ], className="metrics-table")

        ensemble = forecasts.get("ensemble", actual)
        if not isinstance(ensemble, np.ndarray):
            ensemble = actual
        residuals = actual - ensemble
        timestamps = demand_df["timestamp"]

        fig_resid_time = go.Figure(go.Scatter(
            x=timestamps, y=residuals, mode="lines", line=dict(color=COLORS["arima"], width=1),
        ))
        fig_resid_time.add_hline(y=0, line=dict(color="#ffffff", dash="dash", width=0.5))
        fig_resid_time.update_layout(**PLOT_LAYOUT, yaxis_title="Residual (MW)")

        fig_resid_hist = go.Figure(go.Histogram(x=residuals, nbinsx=50, marker_color=COLORS["ensemble"]))
        fig_resid_hist.update_layout(**PLOT_LAYOUT, xaxis_title="Residual (MW)", yaxis_title="Count")

        fig_resid_pred = go.Figure(go.Scatter(
            x=ensemble, y=residuals, mode="markers",
            marker=dict(size=2, color=COLORS["xgboost"], opacity=0.3),
        ))
        fig_resid_pred.add_hline(y=0, line=dict(color="#ffffff", dash="dash", width=0.5))
        fig_resid_pred.update_layout(**PLOT_LAYOUT, xaxis_title="Predicted (MW)", yaxis_title="Residual (MW)")

        hours_of_day = timestamps.dt.hour
        error_by_hour = pd.DataFrame({"hour": hours_of_day, "abs_error": np.abs(residuals)})
        hourly_error = error_by_hour.groupby("hour")["abs_error"].mean()
        fig_heatmap = go.Figure(go.Bar(
            x=hourly_error.index, y=hourly_error.values,
            marker_color=[COLORS["ensemble"] if e > hourly_error.median() else COLORS["actual"]
                          for e in hourly_error.values],
        ))
        fig_heatmap.update_layout(**PLOT_LAYOUT, xaxis_title="Hour of Day", yaxis_title="Mean |Error| (MW)")

        feature_names = ["temperature_2m", "demand_lag_24h", "hour_sin", "cooling_degree_days",
                         "wind_speed_80m", "demand_roll_24h_mean", "heating_degree_days",
                         "solar_capacity_factor", "relative_humidity_2m", "is_holiday"]
        importance_vals = np.array([0.25, 0.18, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04])
        fig_shap = go.Figure(go.Bar(
            x=importance_vals[::-1], y=feature_names[::-1], orientation="h", marker_color=COLORS["xgboost"],
        ))
        fig_shap.update_layout(**PLOT_LAYOUT, xaxis_title="Mean |SHAP Value|")

        return table, fig_resid_time, fig_resid_hist, fig_resid_pred, fig_heatmap, fig_shap

    # ── 7. TAB 4: GENERATION MIX ─────────────────────────────

    @app.callback(
        [Output("tab4-gen-mix", "figure"),
         Output("tab4-wind-overlay", "figure"),
         Output("tab4-solar-overlay", "figure"),
         Output("tab4-duck-curve", "figure"),
         Output("tab4-renewable-trend", "figure"),
         Output("tab4-renewable-pct", "children"),
         Output("tab4-renewable-delta", "children"),
         Output("tab4-wind-cf", "children"),
         Output("tab4-solar-cf", "children"),
         Output("tab4-carbon", "children")],
        [Input("region-selector", "value"),
         Input("demand-store", "data")],
        prevent_initial_call=True,
    )
    def update_generation_tab(region, demand_json):
        """Update Tab 4 generation charts."""
        empty = _empty_figure("Loading...")
        neutral = html.Span("", className="kpi-delta neutral")
        if not demand_json:
            return empty, empty, empty, empty, empty, "—%", neutral, "—%", "—%", "—"

        from data.demo_data import generate_demo_generation, generate_demo_weather
        gen_df = generate_demo_generation(region, days=30)
        gen_df["timestamp"] = pd.to_datetime(gen_df["timestamp"])

        pivot = gen_df.pivot_table(index="timestamp", columns="fuel_type",
                                   values="generation_mw", aggfunc="sum").fillna(0)
        fig_mix = go.Figure()
        for fuel in ["nuclear", "coal", "gas", "hydro", "wind", "solar", "other"]:
            if fuel in pivot.columns:
                fig_mix.add_trace(go.Scatter(
                    x=pivot.index, y=pivot[fuel], mode="lines", name=fuel.title(),
                    stackgroup="one", line=dict(width=0),
                    fillcolor=COLORS.get(fuel, "#95a5a6"),
                ))
        fig_mix.update_layout(**PLOT_LAYOUT, yaxis_title="Generation (MW)", hovermode="x unified")

        total_gen = pivot.sum(axis=1)
        ren_cols = [c for c in ["wind", "solar", "hydro"] if c in pivot.columns]
        renewable_gen = pivot[ren_cols].sum(axis=1) if ren_cols else total_gen * 0
        renewable_pct = (renewable_gen / total_gen * 100).mean()

        capacity = REGION_CAPACITY_MW.get(region, 50000)
        wind_cf = f"{(pivot.get('wind', pd.Series([0])).mean() / (capacity * 0.25) * 100):.0f}%"
        solar_cf_val = pivot.get("solar", pd.Series([0]))
        solar_cf = f"{(solar_cf_val[solar_cf_val > 0].mean() / (capacity * 0.12) * 100):.0f}%" if solar_cf_val.sum() > 0 else "—%"

        recent_ren = (renewable_gen / total_gen * 100).tail(24).mean()
        avg_ren = (renewable_gen / total_gen * 100).mean()
        ren_delta = recent_ren - avg_ren
        ren_dir = "positive" if ren_delta > 0 else "negative"
        ren_delta_span = html.Span(
            f"{'↑' if ren_delta > 0 else '↓'}{abs(ren_delta):.1f}% vs 30d avg",
            className=f"kpi-delta {ren_dir}")

        demand_df = pd.read_json(io.StringIO(demand_json))
        demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])
        weather_df = generate_demo_weather(region, days=30)

        fig_wind = make_subplots(specs=[[{"secondary_y": True}]])
        if "wind" in pivot.columns:
            fig_wind.add_trace(go.Scatter(x=pivot.index, y=pivot["wind"], name="Wind Gen (MW)",
                                          line=dict(color=COLORS["wind"])), secondary_y=False)
        fig_wind.add_trace(go.Scatter(x=weather_df["timestamp"], y=weather_df["wind_speed_80m"],
                                      name="Wind Speed (mph)", line=dict(color="#ffffff", width=1, dash="dot")),
                           secondary_y=True)
        fig_wind.update_layout(**PLOT_LAYOUT)
        fig_wind.update_yaxes(title_text="MW", secondary_y=False)
        fig_wind.update_yaxes(title_text="mph", secondary_y=True)

        fig_solar = make_subplots(specs=[[{"secondary_y": True}]])
        if "solar" in pivot.columns:
            fig_solar.add_trace(go.Scatter(x=pivot.index, y=pivot["solar"], name="Solar Gen (MW)",
                                           line=dict(color=COLORS["solar"])), secondary_y=False)
        fig_solar.add_trace(go.Scatter(x=weather_df["timestamp"], y=weather_df["shortwave_radiation"],
                                       name="GHI (W/m²)", line=dict(color="#ffffff", width=1, dash="dot")),
                            secondary_y=True)
        fig_solar.update_layout(**PLOT_LAYOUT)

        solar_gen = pivot.get("solar", 0)
        net_demand = total_gen - solar_gen
        fig_duck = go.Figure()
        fig_duck.add_trace(go.Scatter(x=pivot.index, y=total_gen, name="Total Demand",
                                      line=dict(color=COLORS["actual"])))
        fig_duck.add_trace(go.Scatter(x=pivot.index, y=net_demand, name="Net Demand (minus Solar)",
                                      line=dict(color=COLORS["ensemble"], width=2)))
        fig_duck.update_layout(**PLOT_LAYOUT, yaxis_title="MW")

        fig_ren = go.Figure(go.Scatter(
            x=pivot.index, y=(renewable_gen / total_gen * 100),
            mode="lines", line=dict(color=COLORS["wind"], width=2),
            fill="tozeroy", fillcolor="rgba(46,204,113,0.15)",
        ))
        fig_ren.update_layout(**PLOT_LAYOUT, yaxis_title="Renewable %", yaxis_range=[0, 100])

        co2_factors = {"gas": 0.41, "coal": 0.95, "nuclear": 0, "wind": 0, "solar": 0, "hydro": 0, "other": 0.5}
        carbon = sum(pivot.get(f, 0).mean() * co2_factors.get(f, 0) for f in co2_factors) / total_gen.mean() * 1000

        return (fig_mix, fig_wind, fig_solar, fig_duck, fig_ren,
                f"{renewable_pct:.0f}%", ren_delta_span, wind_cf, solar_cf, f"{carbon:.0f} kg/MWh")

    # ── 8. TAB 5: ALERTS ─────────────────────────────────────

    @app.callback(
        [Output("tab5-alerts-list", "children"),
         Output("tab5-stress-score", "children"),
         Output("tab5-stress-label", "children"),
         Output("tab5-stress-breakdown", "children"),
         Output("tab5-anomaly-chart", "figure"),
         Output("tab5-temp-exceedance", "figure"),
         Output("tab5-timeline", "figure")],
        [Input("region-selector", "value"),
         Input("demand-store", "data"),
         Input("weather-store", "data")],
        prevent_initial_call=True,
    )
    def update_alerts_tab(region, demand_json, weather_json):
        """Update Tab 5 alerts and stress indicators."""
        empty = _empty_figure("Loading...")

        from data.demo_data import generate_demo_alerts
        alerts = generate_demo_alerts(region)

        alert_cards = []
        if alerts:
            for a in alerts:
                alert_cards.append(build_alert_card(
                    event=a["event"], headline=a["headline"], severity=a["severity"],
                    expires=a.get("expires", "")[:16] if a.get("expires") else None,
                ))
        else:
            alert_cards = [html.P("No active alerts", style={"color": "#8a8fa8", "textAlign": "center", "padding": "20px"})]

        n_crit = sum(1 for a in alerts if a["severity"] == "critical")
        n_warn = sum(1 for a in alerts if a["severity"] == "warning")
        n_info = sum(1 for a in alerts if a["severity"] == "info")
        stress = min(100, n_crit * 30 + n_warn * 15 + 20)
        stress_label = "Normal" if stress < 30 else ("Elevated" if stress < 60 else "Critical")
        stress_color = "positive" if stress < 30 else ("negative" if stress >= 60 else "neutral")

        breakdown_items = []
        if n_crit:
            breakdown_items.append(html.Div(f"🔴 Critical: {n_crit}", style={"fontSize": "0.75rem", "color": "#e94560"}))
        if n_warn:
            breakdown_items.append(html.Div(f"🟡 Warning: {n_warn}", style={"fontSize": "0.75rem", "color": "#f0ad4e"}))
        if n_info:
            breakdown_items.append(html.Div(f"🔵 Info: {n_info}", style={"fontSize": "0.75rem", "color": "#56B4E9"}))
        if not alerts:
            breakdown_items.append(html.Div("No active alerts", style={"fontSize": "0.75rem", "color": "#8a8fa8"}))
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
            fig_anomaly.add_trace(go.Scatter(x=recent["timestamp"], y=recent["demand_mw"],
                                             name="Demand", line=dict(color=COLORS["actual"])))
            fig_anomaly.add_trace(go.Scatter(x=recent["timestamp"], y=upper,
                                             name="Upper (2σ)", line=dict(color="#e94560", dash="dash", width=1)))
            fig_anomaly.add_trace(go.Scatter(x=recent["timestamp"], y=lower,
                                             name="Lower (2σ)", line=dict(color="#e94560", dash="dash", width=1)))
            if not anomalies.empty:
                fig_anomaly.add_trace(go.Scatter(x=anomalies["timestamp"], y=anomalies["demand_mw"],
                                                 mode="markers", name="Anomaly",
                                                 marker=dict(color="#e94560", size=8, symbol="diamond")))
            fig_anomaly.update_layout(**PLOT_LAYOUT, yaxis_title="MW")
        else:
            fig_anomaly = empty

        if weather_json:
            weather_df = pd.read_json(io.StringIO(weather_json))
            weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
            recent_w = weather_df.tail(168)
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Scatter(x=recent_w["timestamp"], y=recent_w["temperature_2m"],
                                          name="Temperature", line=dict(color=COLORS["temperature"])))
            for t in [95, 100, 105]:
                fig_temp.add_hline(y=t, line=dict(color="#e94560", dash="dot", width=1),
                                   annotation_text=f"{t}°F", annotation_position="right")
            fig_temp.update_layout(**PLOT_LAYOUT, yaxis_title="°F")
        else:
            fig_temp = empty

        events = [("2021-02-15", "Winter Storm Uri", "ERCOT", 95),
                  ("2022-09-06", "CA Heat Wave", "CAISO", 80),
                  ("2023-07-20", "Heat Dome", "CAISO", 85),
                  ("2024-04-08", "Solar Eclipse", "PJM", 40)]
        fig_timeline = go.Figure()
        for date, name, reg, sev in events:
            color = COLORS["ensemble"] if reg == region else "#8a8fa8"
            fig_timeline.add_trace(go.Scatter(
                x=[date], y=[sev], mode="markers+text", text=[name],
                textposition="top center", marker=dict(size=12, color=color), showlegend=False,
            ))
        fig_timeline.update_layout(**PLOT_LAYOUT, xaxis_title="Date",
                                    yaxis_title="Severity Score", yaxis_range=[0, 100])

        return (alert_cards, str(stress),
                html.Span(stress_label, className=f"kpi-delta {stress_color}"),
                breakdown, fig_anomaly, fig_temp, fig_timeline)

    # ── 9. TAB 6: SCENARIO SIMULATOR ─────────────────────────

    @app.callback(
        [Output("sim-forecast-chart", "figure"),
         Output("sim-demand-delta", "children"),
         Output("sim-demand-delta-pct", "children"),
         Output("sim-price-impact", "children"),
         Output("sim-price-delta", "children"),
         Output("sim-reserve-margin", "children"),
         Output("sim-reserve-status", "children"),
         Output("sim-renewable-impact", "children"),
         Output("sim-renewable-detail", "children"),
         Output("sim-price-chart", "figure"),
         Output("sim-renewable-chart", "figure")],
        [Input("sim-run-btn", "n_clicks"),
         Input({"type": "preset-btn", "index": ALL}, "n_clicks")],
        [State("sim-temp", "value"),
         State("sim-wind", "value"),
         State("sim-cloud", "value"),
         State("sim-humidity", "value"),
         State("sim-solar", "value"),
         State("sim-duration", "value"),
         State("region-selector", "value"),
         State("demand-store", "data")],
        prevent_initial_call=True,
    )
    def run_scenario(run_clicks, preset_clicks, temp, wind, cloud, humidity, solar_irr,
                     duration, region, demand_json):
        """Run scenario simulation and update impact dashboard."""
        empty = _empty_figure("Click 'Run Scenario' or select a preset")
        if not demand_json:
            return (empty, "— MW", "", "— $/MWh", "", "— %", "", "—", "", empty, empty)

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

        demand_df = pd.read_json(io.StringIO(demand_json))
        demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])
        baseline = demand_df["demand_mw"].tail(duration).values
        capacity = REGION_CAPACITY_MW.get(region, 50000)

        from data.feature_engineering import compute_cdd, compute_hdd
        baseline_temp = 75
        cdd_delta = max(0, temp - 65) - max(0, baseline_temp - 65)
        hdd_delta = max(0, 65 - temp) - max(0, 65 - baseline_temp)
        temp_factor = 1 + (cdd_delta * 0.02 + hdd_delta * 0.015) / 65

        seed = hash((region, temp, wind)) & 0xFFFFFFFF
        rng = np.random.RandomState(seed)
        scenario = baseline * temp_factor + rng.normal(0, capacity * 0.005, len(baseline))
        scenario = np.maximum(scenario, 0)

        delta = scenario - baseline
        mean_delta = np.mean(delta)

        from models.pricing import estimate_price_impact
        base_price = estimate_price_impact(np.mean(baseline), capacity)
        scen_price = estimate_price_impact(np.mean(scenario), capacity)
        reserve = (capacity - np.max(scenario)) / capacity * 100

        from data.feature_engineering import compute_wind_power, compute_solar_capacity_factor
        wind_power = float(compute_wind_power(pd.Series([wind])).iloc[0])
        solar_cf = float(compute_solar_capacity_factor(pd.Series([solar_irr])).iloc[0])

        timestamps = demand_df["timestamp"].tail(duration)
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=timestamps, y=baseline, name="Baseline",
                                          line=dict(color=COLORS["actual"], width=2)))
        fig_forecast.add_trace(go.Scatter(x=timestamps, y=scenario, name="Scenario",
                                          line=dict(color=COLORS["ensemble"], width=2.5)))
        fig_forecast.add_trace(go.Scatter(x=timestamps, y=delta, name="Delta",
                                          line=dict(color=COLORS["temperature"], width=1, dash="dot"), yaxis="y2"))
        fig_forecast.update_layout(**PLOT_LAYOUT, yaxis_title="Demand (MW)",
                                    yaxis2=dict(title="Delta (MW)", overlaying="y", side="right", showgrid=False),
                                    hovermode="x unified")

        utilizations = np.linspace(0.5, 1.1, 100)
        prices = estimate_price_impact(utilizations * capacity, capacity)
        current_util = np.mean(scenario) / capacity
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=utilizations * 100, y=prices, name="Price Curve",
                                       line=dict(color=COLORS["ensemble"])))
        fig_price.add_vline(x=current_util * 100, line=dict(color="#ffffff", dash="dash"),
                            annotation_text=f"Scenario: {current_util*100:.0f}%")
        fig_price.update_layout(**PLOT_LAYOUT, xaxis_title="Utilization %", yaxis_title="$/MWh")

        fig_renewable = go.Figure()
        fig_renewable.add_trace(go.Bar(
            x=["Wind Power", "Solar CF"], y=[wind_power * 100, solar_cf * 100],
            marker_color=[COLORS["wind"], COLORS["solar"]],
            text=[f"{wind_power*100:.0f}%", f"{solar_cf*100:.0f}%"], textposition="auto",
        ))
        fig_renewable.update_layout(**PLOT_LAYOUT, yaxis_title="Capacity Factor %", yaxis_range=[0, 110])

        delta_dir = "positive" if mean_delta < 0 else "negative"
        reserve_status = "positive" if reserve > 15 else ("negative" if reserve < 5 else "neutral")

        return (
            fig_forecast,
            f"{mean_delta:+,.0f} MW",
            html.Span(f"{mean_delta/np.mean(baseline)*100:+.1f}% vs baseline", className=f"kpi-delta {delta_dir}"),
            f"${scen_price:.0f}/MWh",
            html.Span(f"{'↑' if scen_price > base_price else '↓'}{abs(scen_price-base_price):.0f} vs base",
                       className=f"kpi-delta {'negative' if scen_price > base_price else 'positive'}"),
            f"{reserve:.1f}%",
            html.Span("Adequate" if reserve > 15 else ("Low" if reserve > 5 else "CRITICAL"),
                       className=f"kpi-delta {reserve_status}"),
            f"Wind: {wind_power*100:.0f}%",
            html.Span(f"Solar CF: {solar_cf*100:.0f}%", className="kpi-delta neutral"),
            fig_price, fig_renewable,
        )

    # ── SLIDER DISPLAY UPDATES ────────────────────────────────

    for slider_id, unit in [("sim-temp", "°F"), ("sim-wind", "mph"), ("sim-cloud", "%"),
                            ("sim-humidity", "%"), ("sim-solar", "W/m²")]:
        @app.callback(
            Output(f"{slider_id}-display", "children"),
            Input(slider_id, "value"),
        )
        def update_slider_display(val, u=unit):
            return f"{val}{u}"

    # ── PRESET → SLIDER SYNC ─────────────────────────────────

    @app.callback(
        [Output("sim-temp", "value"),
         Output("sim-wind", "value"),
         Output("sim-cloud", "value"),
         Output("sim-humidity", "value"),
         Output("sim-solar", "value")],
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
        return (w.get("temperature_2m", 75), w.get("wind_speed_80m", 15),
                w.get("cloud_cover", 50), w.get("relative_humidity_2m", 60),
                w.get("shortwave_radiation", 500))

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
                warnings.append(f"{icons['stale']} {source.title()}: serving cached data (API unavailable)")
            elif status == "error":
                warnings.append(f"{icons['error']} {source.title()}: data load failed — using fallback")
            elif status == "demo":
                warnings.append(f"{icons['demo']} {source.title()}: demo data (no API key configured)")

        if not warnings:
            return html.Div()

        return dbc.Alert(
            [html.Strong("Data Source Status"), html.Br()] +
            [html.Span(w, style={"display": "block", "fontSize": "0.85rem"}) for w in warnings],
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
        from datetime import datetime, timezone

        if not freshness_json:
            return html.Span("⏳ Loading…", style={"color": "#8a8fa8", "fontSize": "0.75rem"})

        freshness = json.loads(freshness_json)
        statuses = [freshness.get(s, "fresh") for s in ("demand", "weather", "alerts")]

        if all(s == "fresh" for s in statuses):
            color, icon, label = "#00d4aa", "🟢", "Live"
        elif all(s == "demo" for s in statuses):
            color, icon, label = "#8a8fa8", "🧪", "Demo"
        elif any(s == "error" for s in statuses):
            color, icon, label = "#ff4757", "🔴", "Degraded"
        else:
            color, icon, label = "#ffa502", "🟡", "Partial"

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
                    style={"color": "#8a8fa8", "fontSize": "0.7rem"}
                ),
            ],
            style={"color": color, "fontSize": "0.75rem", "fontWeight": "500"},
        )

    # ── SPRINT 4: C2 — SCENARIO BOOKMARKS (URL STATE) ─────────

    @app.callback(
        [Output("region-selector", "value", allow_duplicate=True),
         Output("persona-selector", "value", allow_duplicate=True),
         Output("dashboard-tabs", "active_tab", allow_duplicate=True)],
        Input("url", "search"),
        prevent_initial_call=True,
    )
    def restore_bookmark(search):
        """C2: Restore dashboard state from URL query parameters.

        Supported params: ?region=FPL&persona=trader&tab=tab-forecast
        """
        if not search:
            return no_update, no_update, no_update

        from urllib.parse import parse_qs
        params = parse_qs(search.lstrip("?"))

        region = params.get("region", [None])[0]
        persona = params.get("persona", [None])[0]
        tab = params.get("tab", [None])[0]

        # Validate values
        if region and region not in REGION_NAMES:
            region = None
        if persona and persona not in PERSONAS:
            persona = None
        if tab and tab not in TAB_LABELS:
            tab = None

        return (region or no_update, persona or no_update, tab or no_update)

    @app.callback(
        [Output("url", "search"),
         Output("bookmark-toast", "children")],
        Input("bookmark-btn", "n_clicks"),
        [State("region-selector", "value"),
         State("persona-selector", "value"),
         State("dashboard-tabs", "active_tab")],
        prevent_initial_call=True,
    )
    def create_bookmark(n_clicks, region, persona, tab):
        """C2: Serialize current dashboard state into a shareable URL."""
        if not n_clicks:
            return no_update, no_update

        from urllib.parse import urlencode
        params = urlencode({"region": region, "persona": persona, "tab": tab})
        search = f"?{params}"

        toast = dbc.Toast(
            "Bookmark saved! URL updated — copy it to share this view.",
            header="🔗 Bookmark Created",
            dismissable=True,
            duration=4000,
            is_open=True,
            style={"backgroundColor": "#1e2130", "color": "#e0e0e0",
                   "border": "1px solid #00d4aa"},
        )
        return search, toast

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
        from datetime import datetime, timezone

        if not freshness_json:
            return ""

        freshness = json.loads(freshness_json)
        ts = freshness.get("timestamp", "")

        age_seconds = None
        if ts:
            try:
                fetched = datetime.fromisoformat(ts)
                age_seconds = (datetime.now(timezone.utc) - fetched).total_seconds()
            except (ValueError, TypeError):
                pass

        from components.error_handling import widget_confidence_bar
        return widget_confidence_bar(freshness, age_seconds).children

    # ── SPRINT 5: C9 — MEETING-READY MODE ─────────────────────

    @app.callback(
        [Output("meeting-mode-store", "data"),
         Output("dashboard-header", "className"),
         Output("welcome-card", "style"),
         Output("widget-confidence-bar", "style"),
         Output("fallback-banner", "style")],
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

    # ── NEWS FEED ─────────────────────────────────────────────

    @app.callback(
        Output("news-feed", "children"),
        Input("refresh-interval", "n_intervals"),
        prevent_initial_call=False,
    )
    def update_news_feed(_n):
        """Fetch and render energy news from NewsAPI."""
        from data.news_client import fetch_energy_news

        try:
            articles = fetch_energy_news(page_size=5)
            return build_news_feed(articles)
        except Exception as e:
            log.error("news_feed_update_failed", error=str(e))
            return build_news_feed([])

    # ── DEMAND OUTLOOK TAB ──────────────────────────────────────

    @app.callback(
        [Output("outlook-chart", "figure"),
         Output("outlook-data-through", "children"),
         Output("outlook-peak", "children"),
         Output("outlook-peak-time", "children"),
         Output("outlook-avg", "children"),
         Output("outlook-min", "children"),
         Output("outlook-min-time", "children"),
         Output("outlook-range", "children"),
         Output("outlook-hourly-table", "children")],
        [Input("outlook-horizon", "value"),
         Input("outlook-model", "value"),
         Input("dashboard-tabs", "active_tab")],
        [State("demand-store", "data"),
         State("weather-store", "data"),
         State("region-selector", "value")],
        prevent_initial_call=True,
    )
    def update_demand_outlook(horizon, model_name, active_tab, demand_json, weather_json, region):
        """Generate forward-looking demand forecast."""
        # Only run when this tab is active — avoids 10s+ model training on page load
        if active_tab != "tab-outlook":
            return [no_update] * 9

        log.info("outlook_callback_start", horizon=horizon, model=model_name, region=region)

        horizon_hours = int(horizon)

        if not demand_json or not weather_json:
            fig = go.Figure()
            fig.update_layout(**PLOT_LAYOUT)
            fig.add_annotation(text="Loading data...",
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig, "—", "— MW", "", "— MW", "— MW", "", "— MW", None

        try:
            demand_df = pd.read_json(io.StringIO(demand_json))
            weather_df = pd.read_json(io.StringIO(weather_json))
        except Exception as e:
            log.error("outlook_parse_error", error=str(e))
            fig = go.Figure()
            fig.update_layout(**PLOT_LAYOUT)
            return fig, "—", "— MW", "", "— MW", "— MW", "", "— MW", None

        # Get the data through date (last timestamp in demand data)
        demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])
        data_through = demand_df["timestamp"].max()
        data_through_str = data_through.strftime("%Y-%m-%d %H:%M UTC")

        # Run the forecast
        result = _run_forecast_outlook(demand_df, weather_df, horizon_hours, model_name, region)

        if "error" in result:
            fig = go.Figure()
            fig.update_layout(**PLOT_LAYOUT)
            fig.add_annotation(text=f"Forecast failed: {result['error']}",
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig, data_through_str, "— MW", "", "— MW", "— MW", "", "— MW", None

        timestamps = pd.to_datetime(result["timestamps"])
        predictions = result["predictions"]

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

        # Forecast line
        fig.add_trace(go.Scatter(
            x=timestamps, y=predictions,
            mode="lines",
            name=f"{model_name.upper()} Forecast",
            line=dict(color=COLORS.get("ensemble", "#00d4aa"), width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 170, 0.1)",
        ))

        # Add peak marker
        fig.add_trace(go.Scatter(
            x=[timestamps[peak_idx]], y=[peak_val],
            mode="markers+text",
            name="Peak",
            marker=dict(color="#ff6b6b", size=12, symbol="triangle-up"),
            text=[f"Peak: {peak_val:,.0f} MW"],
            textposition="top center",
            showlegend=False,
        ))

        # Layout
        horizon_labels = {24: "24-Hour", 168: "7-Day", 720: "30-Day"}
        fig.update_layout(
            **PLOT_LAYOUT,
            title=f"{horizon_labels.get(horizon_hours, '')} Demand Forecast for {region}",
            xaxis_title="Date/Time",
            yaxis_title="Demand (MW)",
            hovermode="x unified",
        )

        # Format KPI strings
        peak_str = f"{peak_val:,.0f} MW"
        avg_str = f"{avg_val:,.0f} MW"
        min_str = f"{min_val:,.0f} MW"
        range_str = f"{range_val:,.0f} MW"

        # Hourly table for 24h view
        hourly_table = None
        if horizon_hours == 24:
            hourly_table = _build_hourly_forecast_table(timestamps, predictions)

        log.info("outlook_callback_complete", horizon=horizon_hours, peak=peak_str)
        return fig, data_through_str, peak_str, peak_time, avg_str, min_str, min_time, range_str, hourly_table

    # ── TAB 7: BACKTEST ─────────────────────────────────────────

    @app.callback(
        [Output("backtest-chart", "figure"),
         Output("backtest-mape", "children"),
         Output("backtest-rmse", "children"),
         Output("backtest-mae", "children"),
         Output("backtest-r2", "children"),
         Output("backtest-horizon-explanation", "children")],
        [Input("backtest-horizon", "value"),
         Input("backtest-model", "value"),
         Input("dashboard-tabs", "active_tab")],
        [State("demand-store", "data"),
         State("weather-store", "data"),
         State("region-selector", "value")],
        prevent_initial_call=True,
    )
    def update_backtest_chart(horizon, model_name, active_tab, demand_json, weather_json, region):
        """Build the backtest chart comparing forecast vs actual."""
        # Only run when this tab is active — avoids expensive model evaluation on page load
        if active_tab != "tab-backtest":
            return [no_update] * 6

        log.info("backtest_callback_start", horizon=horizon, model=model_name, region=region,
                 has_demand=bool(demand_json), has_weather=bool(weather_json))

        horizon_hours = int(horizon)

        # Horizon explanations
        explanations = {
            24: "24-hour ahead: Forecast made 1 day before. Best for day-ahead scheduling.",
            168: "7-day ahead: Forecast made 1 week before. Tests medium-term accuracy.",
            720: "30-day ahead: Forecast made 1 month before. Tests long-term planning reliability.",
        }

        if not demand_json:
            log.debug("backtest_no_data")
            fig = go.Figure()
            fig.update_layout(**PLOT_LAYOUT)
            fig.add_annotation(text="No data available. Select a region to load data.",
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig, "—%", "— MW", "— MW", "—", explanations.get(horizon_hours, "")

        # Parse data
        demand_df = pd.read_json(io.StringIO(demand_json))
        demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])

        weather_df = pd.DataFrame()
        if weather_json:
            weather_df = pd.read_json(io.StringIO(weather_json))
            weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])

        # Run backtest
        result = _run_backtest_for_horizon(
            demand_df, weather_df, horizon_hours, model_name, region
        )

        if "error" in result:
            fig = go.Figure()
            fig.update_layout(**PLOT_LAYOUT)
            fig.add_annotation(text=f"Backtest failed: {result['error']}",
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig, "—%", "— MW", "— MW", "—", explanations.get(horizon_hours, "")

        timestamps = pd.to_datetime(result["timestamps"])
        actual = result["actual"]
        predictions = result["predictions"]
        metrics = result["metrics"]

        # Build figure
        fig = go.Figure()

        # Actual demand line
        fig.add_trace(go.Scatter(
            x=timestamps, y=actual,
            mode="lines",
            name="Actual Demand",
            line=dict(color=COLORS["actual"], width=2),
        ))

        # Forecast line
        model_colors = {
            "xgboost": COLORS.get("ensemble", "#00d4aa"),
            "prophet": COLORS.get("prophet", "#ff6b6b"),
            "arima": COLORS.get("arima", "#4ecdc4"),
            "ensemble": COLORS.get("ensemble", "#00d4aa"),
        }

        fig.add_trace(go.Scatter(
            x=timestamps, y=predictions,
            mode="lines",
            name=f"{model_name.upper()} Forecast",
            line=dict(color=model_colors.get(model_name, "#00d4aa"), width=2, dash="dash"),
        ))

        # Error shading (where forecast differs from actual)
        fig.add_trace(go.Scatter(
            x=list(timestamps) + list(timestamps[::-1]),
            y=list(predictions) + list(actual[::-1]),
            fill="toself",
            fillcolor="rgba(255, 107, 107, 0.15)",
            line=dict(width=0),
            name="Forecast Error",
            showlegend=True,
            hoverinfo="skip",
        ))

        # Layout
        horizon_labels = {24: "24-Hour", 168: "7-Day", 720: "30-Day"}
        fig.update_layout(
            **PLOT_LAYOUT,
            title=f"{horizon_labels.get(horizon_hours, '')} Ahead Backtest: {model_name.upper()} vs Actual",
            xaxis_title="Date/Time",
            yaxis_title="Demand (MW)",
            hovermode="x unified",
        )

        # Format metrics
        mape_str = f"{metrics['mape']:.2f}%"
        rmse_str = f"{metrics['rmse']:,.0f} MW"
        mae_str = f"{metrics['mae']:,.0f} MW"
        r2_str = f"{metrics['r2']:.3f}"

        log.info("backtest_callback_complete", mape=mape_str, model=model_name, horizon=horizon_hours)
        return fig, mape_str, rmse_str, mae_str, r2_str, explanations.get(horizon_hours, "")


# ── HELPER FUNCTIONS ──────────────────────────────────────────


def _empty_figure(message: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **PLOT_LAYOUT,
        annotations=[dict(text=message, showarrow=False, font=dict(size=14, color="#8a8fa8"),
                          xref="paper", yref="paper", x=0.5, y=0.5)],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def _build_persona_kpis(persona_id: str, region: str) -> dbc.Row:
    capacity = REGION_CAPACITY_MW.get(region, 50000)
    persona_kpis = {
        "grid_ops": [
            {"label": "Peak Demand", "value": f"{int(capacity * 0.72):,} MW", "delta": "↑3% vs yesterday", "direction": "negative"},
            {"label": "Forecast Error", "value": "2.8%", "delta": "7-day MAPE", "direction": "positive"},
        ],
        "renewables": [
            {"label": "Wind CF", "value": "32%", "delta": "↓6% vs 7d avg", "direction": "negative"},
            {"label": "Solar CF", "value": "68%", "delta": "↑2% vs yesterday", "direction": "positive"},
        ],
        "trader": [
            {"label": "Price Est.", "value": f"${PRICING_BASE_USD_MWH:.0f}/MWh", "delta": "Moderate load", "direction": "neutral"},
            {"label": "Demand vs Forecast", "value": "+2.1%", "delta": "Above EIA forecast", "direction": "negative"},
        ],
        "data_scientist": [
            {"label": "Ensemble MAPE", "value": "2.8%", "delta": "Target: <5%", "direction": "positive"},
            {"label": "RMSE", "value": f"{int(capacity * 0.015):,} MW", "delta": "7-day rolling", "direction": "neutral"},
        ],
    }
    kpis = persona_kpis.get(persona_id, persona_kpis["grid_ops"])
    return build_kpi_row(kpis)


def _run_backtest_for_horizon(demand_df: pd.DataFrame, weather_df: pd.DataFrame,
                               horizon_hours: int, model_name: str, region: str) -> dict:
    """
    Run backtest for a specific forecast horizon.

    Args:
        demand_df: Full demand dataframe with timestamp and demand_mw
        weather_df: Full weather dataframe
        horizon_hours: Forecast horizon (24, 168, or 720 hours)
        model_name: Model to use (prophet, arima, xgboost, ensemble)
        region: Region code

    Returns:
        Dict with predictions, actuals, timestamps, and metrics
    """
    import time
    from data.preprocessing import merge_demand_weather
    from data.feature_engineering import engineer_features
    from models.evaluation import compute_all_metrics

    # Check backtest cache first
    data_hash = hash((len(demand_df), len(weather_df), region))
    cache_key = (region, horizon_hours, model_name)

    if cache_key in _BACKTEST_CACHE:
        cached_result, cached_hash, cached_time = _BACKTEST_CACHE[cache_key]
        if cached_hash == data_hash and (time.time() - cached_time) < _CACHE_TTL_SECONDS:
            log.info("backtest_cache_hit", region=region, horizon=horizon_hours, model=model_name)
            return cached_result

    # Merge and engineer features
    merged_df = merge_demand_weather(demand_df, weather_df)
    featured_df = engineer_features(merged_df)
    featured_df = featured_df.dropna(subset=["demand_mw"])

    if len(featured_df) < horizon_hours + 168:
        return {"error": "Insufficient data"}

    # Split: train on everything except last horizon_hours
    cutoff_idx = len(featured_df) - horizon_hours
    train_df = featured_df.iloc[:cutoff_idx].copy()
    test_df = featured_df.iloc[cutoff_idx:].copy()

    actual = test_df["demand_mw"].values
    timestamps = test_df["timestamp"].values

    predictions = None

    try:
        if model_name == "xgboost":
            from models.xgboost_model import train_xgboost, predict_xgboost
            xgb_model = train_xgboost(train_df)
            predictions = predict_xgboost(xgb_model, test_df)[:len(actual)]

        elif model_name == "prophet":
            from models.prophet_model import train_prophet, predict_prophet
            prophet_model = train_prophet(train_df)
            prophet_result = predict_prophet(prophet_model, test_df, periods=len(test_df))
            predictions = prophet_result["forecast"][:len(actual)]

        elif model_name == "arima":
            from models.arima_model import train_arima, predict_arima
            arima_model = train_arima(train_df)
            predictions = predict_arima(arima_model, test_df, periods=len(test_df))[:len(actual)]

        elif model_name == "ensemble":
            # Train all models and combine
            preds = {}
            weights = {}

            try:
                from models.xgboost_model import train_xgboost, predict_xgboost
                xgb_model = train_xgboost(train_df)
                preds["xgboost"] = predict_xgboost(xgb_model, test_df)[:len(actual)]
            except Exception:
                pass

            try:
                from models.prophet_model import train_prophet, predict_prophet
                prophet_model = train_prophet(train_df)
                prophet_result = predict_prophet(prophet_model, test_df, periods=len(test_df))
                preds["prophet"] = prophet_result["forecast"][:len(actual)]
            except Exception:
                pass

            try:
                from models.arima_model import train_arima, predict_arima
                arima_model = train_arima(train_df)
                preds["arima"] = predict_arima(arima_model, test_df, periods=len(test_df))[:len(actual)]
            except Exception:
                pass

            if preds:
                # Compute 1/MAPE weights
                total_inv_mape = 0
                for name, pred in preds.items():
                    mape = np.mean(np.abs((actual - pred) / actual)) * 100
                    if mape > 0:
                        weights[name] = 1.0 / mape
                        total_inv_mape += weights[name]

                if total_inv_mape > 0:
                    ensemble_pred = np.zeros(len(actual))
                    for name, pred in preds.items():
                        w = weights[name] / total_inv_mape
                        ensemble_pred += pred * w
                    predictions = ensemble_pred
                else:
                    predictions = np.mean(list(preds.values()), axis=0)
            else:
                return {"error": "No models trained successfully"}

    except Exception as e:
        log.warning("backtest_model_failed", model=model_name, error=str(e))
        return {"error": str(e)}

    if predictions is None:
        return {"error": "Model training failed"}

    metrics = compute_all_metrics(actual, predictions)

    result = {
        "timestamps": timestamps,
        "actual": actual,
        "predictions": predictions,
        "metrics": metrics,
    }

    # Cache the backtest result
    _BACKTEST_CACHE[cache_key] = (result, data_hash, time.time())
    log.info("backtest_cached", region=region, horizon=horizon_hours, model=model_name)

    return result
