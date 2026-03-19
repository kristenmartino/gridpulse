"""
FastAPI Server — Read-Only Serving Layer.


CRITICAL DESIGN RULE:
    This server NEVER runs ML models, builds features, or queries raw data.
    It ONLY reads pre-computed results from Redis.

    Every endpoint is a cache read. If Redis is empty, we return
    503 "pipeline not ready" — we do NOT fall back to computing.

    The ONE exception is POST /scenarios/simulate, which runs on-demand
    inference (~2s). It lazy-imports the scenario service to keep
    model code out of the module-level scope.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from enum import StrEnum
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.api.cache import ForecastCache
from src.config import FORECAST_GRANULARITIES, GRID_REGIONS, DatabaseConfig

logger = logging.getLogger(__name__)

# ── Lifespan: init/cleanup connections ─────────
cache: ForecastCache | None = None
db_conn = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cache, db_conn
    cache = ForecastCache()
    try:
        import psycopg2

        db_config = DatabaseConfig()
        db_conn = psycopg2.connect(db_config.url)
    except Exception as e:
        logger.warning("Postgres connection failed (audit/freshness unavailable): %s", e)
        db_conn = None
    logger.info("Server started — Redis cache connection established")
    yield
    cache.close()
    if db_conn:
        db_conn.close()
    logger.info("Server stopped — connections closed")


# ── App ─────────────────────────────────────────
app = FastAPI(
    title="WattCast v2 API",
    description="Pre-computed energy demand forecasts. Zero computation at request time.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten for production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Enums ──────────────────────────────────────
class Granularity(StrEnum):
    fifteen_min = "15min"
    one_hour = "1h"
    one_day = "1d"


# ── Pydantic Response Models ──────────────────


class ForecastResponse(BaseModel):
    region: str
    scored_at: str
    models_trained_at: str | None = None
    granularity: str
    forecasts: list[dict]


class HealthResponse(BaseModel):
    status: str
    last_scored: str | None
    models_trained_at: str | None = None
    cache_healthy: bool
    regions_available: int
    scoring_interval_min: int
    scoring_mode: str


class BacktestResponse(BaseModel):
    horizon: int | None = None
    actual: list[float] | None = None
    timestamps: list[str] | None = None
    metrics: dict[str, Any] | None = None
    predictions: dict[str, Any] | None = None
    residuals: list[float] | None = None
    error_by_hour: list[dict] | None = None


class ResidualsResponse(BaseModel):
    region: str
    horizon: int
    residuals: list[float]
    timestamps: list[str]


class ErrorByHourResponse(BaseModel):
    region: str
    horizon: int
    error_by_hour: list[dict]


class ActualsResponse(BaseModel):
    region: str
    timestamps: list[str]
    demand_mw: list[float]
    forecast_mw: list[float | None] | None = None


class WeatherResponse(BaseModel):
    region: str
    timestamps: list[str]
    model_config = {"extra": "allow"}  # Allow dynamic weather columns


class ModelMetricsResponse(BaseModel):
    region: str
    metrics: dict[str, Any]
    updated_at: str | None = None


class WeightsResponse(BaseModel):
    weights: dict[str, float]
    metrics: dict[str, Any] | None = None
    updated_at: str | None = None


class GenerationResponse(BaseModel):
    region: str
    timestamps: list[str]
    renewable_pct: list[float] | None = None
    model_config = {"extra": "allow"}  # Allow dynamic fuel-type columns


class AlertsResponse(BaseModel):
    region: str
    alerts: list[dict]
    stress_score: int
    stress_label: str


class ScenarioRequest(BaseModel):
    region: str
    temperature_2m: float | None = None
    wind_speed_80m: float | None = None
    cloud_cover: float | None = None
    relative_humidity_2m: float | None = None
    shortwave_radiation: float | None = None
    duration_hours: int = Field(default=24, ge=1, le=168)


class ScenarioResponse(BaseModel):
    region: str
    duration_hours: int
    weather_overrides: dict[str, float]
    baseline: list[float]
    scenario: list[float]
    delta_mw: list[float]
    delta_pct: float
    pricing: dict[str, float]
    reserve_margin: dict[str, Any]
    renewable_impact: dict[str, float]
    computed_at: str | None = None


class PresetListResponse(BaseModel):
    presets: list[dict]


class PersonaResponse(BaseModel):
    id: str
    name: str
    title: str
    avatar: str
    default_tab: str
    priority_tabs: list[str]
    kpi_metrics: list[str]
    alert_threshold: str
    welcome_title: str
    welcome_message: str
    color: str


class WelcomeResponse(BaseModel):
    persona_id: str
    region: str
    message: str


class NewsResponse(BaseModel):
    articles: list[dict]


class FreshnessResponse(BaseModel):
    sources: list[dict]


class AuditResponse(BaseModel):
    region: str
    scored_at: str | None = None
    demand_source: str | None = None
    weather_source: str | None = None
    demand_rows: int | None = None
    weather_rows: int | None = None
    model_versions: Any | None = None
    ensemble_weights: Any | None = None
    feature_count: int | None = None
    feature_hash: str | None = None
    mape: Any | None = None
    peak_forecast_mw: float | None = None
    scoring_mode: str | None = None
    created_at: str | None = None


# ── Helpers ────────────────────────────────────


def _validate_region(region: str) -> str:
    """Validate and normalize region code."""
    upper = region.upper()
    if upper not in GRID_REGIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown region '{region}'. Available: {GRID_REGIONS}",
        )
    return upper


def _require_data(data, entity: str, region: str = ""):
    """Raise 503 if data is None (pipeline hasn't run yet)."""
    if data is None:
        detail = f"No {entity} available"
        if region:
            detail += f" for {region}"
        detail += ". The batch pipeline may not have run yet. Check /health."
        raise HTTPException(status_code=503, detail=detail)


# ── Routes ─────────────────────────────────────

# ─── Health ────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Pipeline freshness check."""
    from src.config import SCORING_INTERVAL_MINUTES, ModelConfig

    meta = cache.get_pipeline_metadata()
    is_healthy = cache.is_healthy()
    model_config = ModelConfig()

    # Determine status: healthy, degraded (stale models), or stale
    status = "healthy" if is_healthy else "stale"
    models_trained_at = meta.get("models_trained_at") if meta else None

    return HealthResponse(
        status=status,
        last_scored=meta["scored_at"] if meta else None,
        models_trained_at=models_trained_at,
        cache_healthy=is_healthy,
        regions_available=meta["regions_scored"] if meta else 0,
        scoring_interval_min=SCORING_INTERVAL_MINUTES,
        scoring_mode="fast (XGBoost only)" if model_config.fast_mode else "full (3-model ensemble)",
    )


# ─── Forecasts (Tab 2) ────────────────────────


@app.get("/forecasts/{region}", response_model=ForecastResponse)
async def get_forecast(
    region: str,
    granularity: Granularity = Query(default=Granularity.fifteen_min),
):
    """Get pre-computed demand forecast for a region."""
    region = _validate_region(region)
    result = cache.get_forecast(region, granularity.value)
    _require_data(result, "forecast", region)
    return ForecastResponse(**result)


@app.get("/forecasts", response_model=list[ForecastResponse])
async def get_all_forecasts(
    granularity: Granularity = Query(default=Granularity.one_hour),
):
    """Get forecasts for all 8 grid regions in a single request."""
    results = cache.get_all_regions(granularity.value)
    if not results:
        raise HTTPException(
            status_code=503,
            detail="No forecasts available. Check /health for pipeline status.",
        )
    return [ForecastResponse(**r) for r in results]


# ─── Actuals / Historical Demand (Tab 1) ──────


@app.get("/actuals/{region}")
async def get_actuals(
    region: str,
    hours: int = Query(default=168, ge=1, le=2160),
):
    """Historical demand actuals for a region."""
    region = _validate_region(region)
    data = cache.get_actuals(region)
    _require_data(data, "actuals", region)
    # Trim to requested hours (data is 15-min intervals, so 4 points/hour)
    max_points = hours * 4
    for key in ("timestamps", "demand_mw", "forecast_mw"):
        if key in data and data[key] is not None:
            data[key] = data[key][-max_points:]
    return data


@app.get("/actuals/{region}/weather-overlay")
async def get_actuals_weather_overlay(region: str):
    """Temperature series for overlaying on historical demand chart."""
    region = _validate_region(region)
    weather = cache.get_weather(region)
    _require_data(weather, "weather", region)
    return {
        "region": region,
        "timestamps": weather.get("timestamps", []),
        "temperature_2m": weather.get("temperature_2m", []),
    }


# ─── Backtests (Tab 3) ────────────────────────


@app.get("/backtests/{region}", response_model=BacktestResponse)
async def get_backtest(
    region: str,
    horizon: int = Query(default=24, ge=1),
    model: str | None = Query(default=None),
):
    """Pre-computed backtest results for a region and horizon."""
    region = _validate_region(region)
    data = cache.get_backtest(region, horizon, model=model)
    _require_data(data, "backtest", region)
    return BacktestResponse(**data)


@app.get("/backtests/{region}/residuals", response_model=ResidualsResponse)
async def get_backtest_residuals(
    region: str,
    horizon: int = Query(default=24, ge=1),
):
    """Residual array from a backtest."""
    region = _validate_region(region)
    data = cache.get_backtest_residuals(region, horizon)
    _require_data(data, "backtest residuals", region)
    return ResidualsResponse(**data)


@app.get("/backtests/{region}/error-by-hour", response_model=ErrorByHourResponse)
async def get_error_by_hour(
    region: str,
    horizon: int = Query(default=24, ge=1),
):
    """Mean absolute error by hour-of-day from a backtest."""
    region = _validate_region(region)
    data = cache.get_backtest_error_by_hour(region, horizon)
    _require_data(data, "error-by-hour", region)
    return ErrorByHourResponse(**data)


# ─── Weather (Tab 4) ──────────────────────────


@app.get("/weather/{region}")
async def get_weather(region: str):
    """Latest weather data for a region (17 variables)."""
    region = _validate_region(region)
    data = cache.get_weather(region)
    _require_data(data, "weather", region)
    return data


@app.get("/weather/{region}/correlation")
async def get_weather_correlation(region: str):
    """
    Pre-computed correlation between weather variables and demand.

    Returns correlation coefficients for key weather-demand pairs.
    """
    region = _validate_region(region)
    weather = cache.get_weather(region)
    actuals = cache.get_actuals(region)
    if weather is None or actuals is None:
        _require_data(None, "weather/demand correlation data", region)

    # Compute correlations from cached arrays
    import numpy as np

    demand = np.array(actuals.get("demand_mw", []))
    weather_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_80m",
        "cloud_cover",
        "shortwave_radiation",
        "surface_pressure",
    ]

    correlations = {}
    for var in weather_vars:
        arr = np.array(weather.get(var, []))
        min_len = min(len(demand), len(arr))
        if min_len > 10:
            corr = np.corrcoef(demand[-min_len:], arr[-min_len:])[0, 1]
            correlations[var] = round(float(corr), 4) if not np.isnan(corr) else 0.0
        else:
            correlations[var] = 0.0

    return {"region": region, "correlations": correlations}


# ─── Models (Tab 5) ───────────────────────────


@app.get("/models/{region}/metrics")
async def get_model_metrics(region: str):
    """Per-model MAPE, RMSE, MAE, R² from latest scoring run."""
    region = _validate_region(region)
    data = cache.get_ensemble_weights(region)
    _require_data(data, "model metrics", region)
    return {
        "region": region,
        "metrics": data.get("metrics", {}),
        "updated_at": data.get("updated_at"),
    }


@app.get("/models/{region}/weights", response_model=WeightsResponse)
async def get_model_weights(region: str):
    """Ensemble weights for a region."""
    region = _validate_region(region)
    data = cache.get_ensemble_weights(region)
    _require_data(data, "ensemble weights", region)
    return WeightsResponse(**data)


@app.get("/models/{region}/feature-importance")
async def get_feature_importance(region: str):
    """
    XGBoost feature importance (top features).

    Read from the backtest results where feature importances are stored.
    """
    region = _validate_region(region)
    # Feature importance is embedded in the forecast or backtest data
    # For now, return from the 24h backtest if available
    bt = cache.get_backtest(region, 24)
    if bt is None:
        _require_data(None, "feature importance", region)

    return {
        "region": region,
        "note": "Feature importance is computed during training. See backtest metrics for model-level details.",
        "metrics": bt.get("metrics", {}),
    }


# ─── Generation Mix (Tab 6) ───────────────────


@app.get("/generation/{region}")
async def get_generation(region: str):
    """Fuel-type generation breakdown + renewable percentage."""
    region = _validate_region(region)
    data = cache.get_generation(region)
    _require_data(data, "generation", region)
    return data


@app.get("/generation/{region}/capacity-factors")
async def get_capacity_factors(region: str):
    """Wind CF, solar CF, carbon intensity estimates."""
    region = _validate_region(region)
    gen = cache.get_generation(region)
    _require_data(gen, "generation", region)

    import numpy as np

    wind = np.array(gen.get("wind", []))
    solar = np.array(gen.get("solar", []))
    renewable_pct = gen.get("renewable_pct", [])

    return {
        "region": region,
        "wind_cf_pct": round(float(np.mean(wind) / max(np.max(wind), 1) * 100), 1)
        if len(wind) > 0
        else 0,
        "solar_cf_pct": round(float(np.mean(solar) / max(np.max(solar), 1) * 100), 1)
        if len(solar) > 0
        else 0,
        "avg_renewable_pct": round(float(np.mean(renewable_pct)), 1) if renewable_pct else 0,
    }


# ─── Alerts (Tab 7) ───────────────────────────


@app.get("/alerts/{region}", response_model=AlertsResponse)
async def get_alerts(region: str):
    """Active alerts + stress score for a region."""
    region = _validate_region(region)
    data = cache.get_alerts(region)
    _require_data(data, "alerts", region)
    return AlertsResponse(**data)


@app.get("/alerts/{region}/anomalies")
async def get_anomalies(region: str):
    """Demand anomaly detection results."""
    region = _validate_region(region)
    alerts = cache.get_alerts(region)
    _require_data(alerts, "alerts", region)
    # Anomalies are embedded in alerts payload if available
    return {
        "region": region,
        "anomalies": [a for a in alerts.get("alerts", []) if a.get("type") == "anomaly"],
        "stress_score": alerts.get("stress_score", 0),
    }


@app.get("/alerts/{region}/extreme-events")
async def get_extreme_events(region: str):
    """Historical extreme event timeline."""
    region = _validate_region(region)
    alerts = cache.get_alerts(region)
    _require_data(alerts, "alerts", region)
    return {
        "region": region,
        "events": [a for a in alerts.get("alerts", []) if a.get("severity") == "critical"],
    }


# ─── Scenarios (Tab 8) ────────────────────────


@app.get("/scenarios/presets")
async def list_scenario_presets():
    """List available scenario presets."""
    try:
        from simulation.presets import list_presets

        presets = list_presets()
    except Exception:
        presets = []
    return {"presets": presets}


@app.get("/scenarios/presets/{key}")
async def get_scenario_preset(
    key: str,
    region: str = Query(default="ERCOT"),
):
    """Get a pre-computed scenario preset result for a region."""
    region = _validate_region(region)
    data = cache.get_scenario_preset(region, key)
    _require_data(data, f"scenario preset '{key}'", region)
    return data


@app.post("/scenarios/simulate", response_model=ScenarioResponse)
async def simulate_scenario(request: ScenarioRequest):
    """
    Run a custom weather scenario simulation (~2s).

    This is the ONE endpoint that performs on-demand computation.
    All other endpoints are pre-computed cache reads.
    """
    region = _validate_region(request.region)

    # Build weather overrides from non-None fields
    overrides = {}
    for field in (
        "temperature_2m",
        "wind_speed_80m",
        "cloud_cover",
        "relative_humidity_2m",
        "shortwave_radiation",
    ):
        val = getattr(request, field, None)
        if val is not None:
            overrides[field] = val

    if not overrides:
        raise HTTPException(
            status_code=400,
            detail="At least one weather override must be provided.",
        )

    try:
        # Lazy import — keeps model code out of module scope
        from src.processing.scenario_service import simulate_custom_scenario

        result = simulate_custom_scenario(
            region=region,
            weather_overrides=overrides,
            duration_hours=request.duration_hours,
        )
        return ScenarioResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("Scenario simulation failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Scenario simulation failed: {e}",
        ) from e


# ─── Personas ──────────────────────────────────


@app.get("/personas")
async def list_personas():
    """List available user personas."""
    try:
        from personas.config import list_personas as _list

        return {"personas": _list()}
    except Exception:
        return {"personas": []}


@app.get("/personas/{persona_id}")
async def get_persona(persona_id: str):
    """Get persona configuration."""
    try:
        from personas.config import get_persona as _get

        persona = _get(persona_id)
        return PersonaResponse(
            id=persona.id,
            name=persona.name,
            title=persona.title,
            avatar=persona.avatar,
            default_tab=persona.default_tab,
            priority_tabs=persona.priority_tabs,
            kpi_metrics=persona.kpi_metrics,
            alert_threshold=persona.alert_threshold,
            welcome_title=persona.welcome_title,
            welcome_message=persona.welcome_message,
            color=persona.color,
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Persona '{persona_id}' not found: {e}") from e


@app.get("/personas/{persona_id}/welcome")
async def get_welcome_message(
    persona_id: str,
    region: str = Query(default="FPL"),
):
    """Generate a data-driven welcome message for a persona."""
    region = _validate_region(region)
    try:
        from personas.welcome import generate_welcome_message

        message = generate_welcome_message(persona_id, region)
        return WelcomeResponse(
            persona_id=persona_id,
            region=region,
            message=message,
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Welcome generation failed: {e}") from e


# ─── Cross-cutting ─────────────────────────────


@app.get("/news", response_model=NewsResponse)
async def get_news():
    """Energy news feed."""
    data = cache.get_news()
    _require_data(data, "news")
    return NewsResponse(**data)


@app.get("/data-freshness", response_model=FreshnessResponse)
async def get_data_freshness():
    """Per-source data freshness status."""
    if db_conn is None:
        return FreshnessResponse(
            sources=[
                {
                    "source": "Database",
                    "status": "unavailable",
                    "error": "No database connection",
                }
            ]
        )
    try:
        from src.processing.audit import get_data_freshness as _freshness

        sources = _freshness(db_conn)
        return FreshnessResponse(sources=sources)
    except Exception as e:
        logger.warning("Data freshness check failed: %s", e)
        return FreshnessResponse(
            sources=[
                {
                    "source": "Database",
                    "status": "error",
                    "error": str(e),
                }
            ]
        )


@app.get("/audit/{region}", response_model=AuditResponse)
async def get_audit(region: str):
    """Latest audit trail record for a region."""
    region = _validate_region(region)
    if db_conn is None:
        raise HTTPException(status_code=503, detail="Database connection unavailable.")
    try:
        from src.processing.audit import read_latest_audit

        record = read_latest_audit(db_conn, region)
        if record is None:
            _require_data(None, "audit record", region)
        record["region"] = region
        return AuditResponse(**record)
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Audit read failed for %s: %s", region, e)
        raise HTTPException(status_code=500, detail=f"Audit read failed: {e}") from e


@app.get("/regions")
async def list_regions():
    """List available grid regions."""
    return {"regions": GRID_REGIONS}


@app.get("/granularities")
async def list_granularities():
    """List available forecast granularities."""
    return {"granularities": list(FORECAST_GRANULARITIES.keys())}
