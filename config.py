"""
Global configuration for the Energy Demand Forecasting Dashboard.

All constants, API URLs, region definitions, and environment-based settings.
Source of truth for values referenced across data/, models/, simulation/, personas/.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Environment (J1 — Environment Config Matrix)
# ---------------------------------------------------------------------------
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG" if ENVIRONMENT == "development" else "INFO")
PORT = int(os.getenv("PORT", "8080"))
DEBUG = ENVIRONMENT == "development"

# J1: Environment-specific defaults.  Every value is overridable via env var.
# ┌──────────────────────┬────────────┬───────────┬────────────┐
# │ Setting              │ dev        │ staging   │ production │
# ├──────────────────────┼────────────┼───────────┼────────────┤
# │ LOG_LEVEL            │ DEBUG      │ INFO      │ WARNING    │
# │ CACHE_TTL_SECONDS    │ 86400 (24h)│ 86400(24h)│ 86400 (24h)│
# │ MIN_INSTANCES        │ 0          │ 0         │ 1          │
# │ MAX_INSTANCES        │ 1          │ 2         │ 4          │
# │ USE_DEMO_DATA        │ True       │ False     │ False      │
# │ ENABLE_PROFILING     │ True       │ False     │ False      │
# │ GUNICORN_WORKERS     │ 1          │ 2         │ 2          │
# │ GCS_ENABLED          │ False      │ True      │ True       │
# └──────────────────────┴────────────┴───────────┴────────────┘
_ENV_DEFAULTS: dict[str, dict] = {
    "development": {
        "cache_ttl": 86400,
        "min_inst": 0,
        "max_inst": 1,
        "demo": True,
        "profile": True,
        "workers": 1,
        "gcs_enabled": False,
    },
    "staging": {
        "cache_ttl": 86400,
        "min_inst": 0,
        "max_inst": 2,
        "demo": False,
        "profile": False,
        "workers": 2,
        "gcs_enabled": True,
    },
    "production": {
        "cache_ttl": 86400,
        "min_inst": 1,
        "max_inst": 4,
        "demo": False,
        "profile": False,
        "workers": 2,
        "gcs_enabled": True,
    },
}
_env = _ENV_DEFAULTS.get(ENVIRONMENT, _ENV_DEFAULTS["development"])

USE_DEMO_DATA = os.getenv("USE_DEMO_DATA", str(_env["demo"])).lower() in ("true", "1", "yes")
ENABLE_PROFILING = os.getenv("ENABLE_PROFILING", str(_env["profile"])).lower() in (
    "true",
    "1",
    "yes",
)
GUNICORN_WORKERS = int(os.getenv("GUNICORN_WORKERS", str(_env["workers"])))
MIN_INSTANCES = int(os.getenv("MIN_INSTANCES", str(_env["min_inst"])))
MAX_INSTANCES = int(os.getenv("MAX_INSTANCES", str(_env["max_inst"])))

# ---------------------------------------------------------------------------
# API Keys & URLs
# ---------------------------------------------------------------------------
EIA_API_KEY = os.getenv("EIA_API_KEY", "")
EIA_BASE_URL = "https://api.eia.gov/v2"

OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1"
OPEN_METEO_PARAMS = "&temperature_unit=fahrenheit&wind_speed_unit=mph"

NOAA_BASE_URL = "https://api.weather.gov"

# AI Briefing (Overview tab executive briefing via Claude)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# Redis (v2 pre-computation pipeline)
# ---------------------------------------------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
CACHE_DB_PATH = os.getenv("CACHE_DB_PATH", "cache.db")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", str(_env["cache_ttl"])))

# ---------------------------------------------------------------------------
# GCS Persistence (Parquet fallback for container recycle + API failure)
# ---------------------------------------------------------------------------
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "")
GCS_ENABLED = os.getenv("GCS_ENABLED", str(_env.get("gcs_enabled", False))).lower() in (
    "true",
    "1",
    "yes",
)
GCS_PATH_PREFIX = os.getenv("GCS_PATH_PREFIX", "cache")

# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "trained_models")
MODEL_REFRESH_INTERVAL = int(os.getenv("MODEL_REFRESH_INTERVAL", "86400"))  # 24h
TRAINING_WINDOW_DAYS = 365
FORECAST_HORIZON_DAYS = 7

# ---------------------------------------------------------------------------
# Regions — 8 Balancing Authorities
# ---------------------------------------------------------------------------
REGION_COORDINATES: dict[str, dict] = {
    "ERCOT": {"lat": 31.0, "lon": -97.0, "name": "Texas (ERCOT)"},
    "CAISO": {"lat": 37.0, "lon": -120.0, "name": "California (CAISO)"},
    "PJM": {"lat": 39.5, "lon": -77.0, "name": "Mid-Atlantic (PJM)"},
    "MISO": {"lat": 41.0, "lon": -89.0, "name": "Midwest (MISO)"},
    "NYISO": {"lat": 42.5, "lon": -74.0, "name": "New York (NYISO)"},
    "FPL": {"lat": 26.9, "lon": -80.1, "name": "Florida (FPL/NextEra)"},
    "SPP": {"lat": 35.5, "lon": -97.5, "name": "Southwest (SPP)"},
    "ISONE": {"lat": 42.3, "lon": -71.8, "name": "New England (ISO-NE)"},
}

REGION_NAMES: dict[str, str] = {k: v["name"] for k, v in REGION_COORDINATES.items()}

# ---------------------------------------------------------------------------
# Generation Capacity (MW) — from EIA-860 data
# Used by the scenario simulator's merit-order pricing model.
# Update annually or fetch from: electricity/operating-generator-capacity
# ---------------------------------------------------------------------------
REGION_CAPACITY_MW: dict[str, int] = {
    "ERCOT": 130_000,
    "CAISO": 80_000,
    "PJM": 185_000,
    "MISO": 175_000,
    "NYISO": 38_000,
    "FPL": 32_000,
    "SPP": 90_000,
    "ISONE": 30_000,
}

# ---------------------------------------------------------------------------
# NOAA State → Balancing Authority Mapping
# Multiple states map to each BA; alerts for any state trigger for that BA.
# Some states appear in multiple BAs (e.g., TX in ERCOT, MISO, SPP).
# ---------------------------------------------------------------------------
STATE_TO_BA: dict[str, list[str]] = {
    "ERCOT": ["TX"],
    "CAISO": ["CA"],
    "PJM": ["PA", "NJ", "MD", "DE", "VA", "WV", "OH", "DC", "NC", "IN", "IL", "MI", "KY", "TN"],
    "MISO": ["MN", "WI", "IA", "IL", "IN", "MI", "MO", "AR", "LA", "MS", "TX"],
    "NYISO": ["NY"],
    "FPL": ["FL"],
    "SPP": ["KS", "OK", "NE", "SD", "ND", "AR", "LA", "MO", "NM", "TX"],
    "ISONE": ["CT", "MA", "ME", "NH", "RI", "VT"],
}

# Reverse lookup: state → list of BAs
BA_FOR_STATE: dict[str, list[str]] = {}
for ba, states in STATE_TO_BA.items():
    for state in states:
        BA_FOR_STATE.setdefault(state, []).append(ba)

# ---------------------------------------------------------------------------
# Open-Meteo Weather Variables
# All 17 variables pulled for each BA centroid.
# ---------------------------------------------------------------------------
WEATHER_VARIABLES: list[str] = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_speed_120m",
    "wind_direction_10m",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "cloud_cover",
    "precipitation",
    "snowfall",
    "surface_pressure",
    "soil_temperature_0cm",
    "weather_code",
]

# ---------------------------------------------------------------------------
# Feature Engineering Constants
# ---------------------------------------------------------------------------
CDD_HDD_BASELINE_F = 65.0  # Fahrenheit baseline for CDD/HDD
WIND_CUTOUT_SPEED_MS = 25.0  # m/s — turbines shut down above this
WIND_CUTOUT_SPEED_MPH = 56.0  # mph equivalent
MPH_TO_MS = 0.44704  # conversion factor
SOLAR_RATED_IRRADIANCE = 1000.0  # W/m² — standard test conditions for solar panels
AIR_DENSITY_KG_M3 = 1.225  # standard air density at sea level

# ---------------------------------------------------------------------------
# EIA API Endpoints
# ---------------------------------------------------------------------------
EIA_ENDPOINTS = {
    "demand": "electricity/rto/region-data",
    "fuel_type": "electricity/rto/fuel-type-data",
    "interchange": "electricity/rto/interchange-data",
}

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
TAB_IDS: list[str] = [
    "tab-overview",
    "tab-forecast",
    "tab-outlook",
    "tab-backtest",
    "tab-generation",
    "tab-weather",
    "tab-models",
    "tab-alerts",
    "tab-simulator",
]

TAB_LABELS: dict[str, str] = {
    "tab-overview": "Overview",
    "tab-forecast": "History",
    "tab-outlook": "Forecast",
    "tab-backtest": "Validation",
    "tab-generation": "Grid",
    "tab-weather": "Conditions",
    "tab-models": "Models",
    "tab-alerts": "Risk",
    "tab-simulator": "Scenarios",
}

# ---------------------------------------------------------------------------
# Pricing Model Constants (merit-order)
# ---------------------------------------------------------------------------
PRICING_BASE_USD_MWH = 50.0
PRICING_TIER_MODERATE = 0.70  # utilization threshold
PRICING_TIER_HIGH = 0.90
PRICING_TIER_EMERGENCY = 1.00
PRICING_EMERGENCY_MULTIPLIER = 20.0

# ---------------------------------------------------------------------------
# Data Quality / Staleness Thresholds (Backlog E2)
# ---------------------------------------------------------------------------
STALENESS_THRESHOLDS_SECONDS: dict[str, int] = {
    "weather": 7200,  # 2 hours
    "generation": 300,  # 5 minutes
    "pricing": 900,  # 15 minutes
    "demand": 3600,  # 1 hour
    "alerts": 1800,  # 30 minutes
}

# ---------------------------------------------------------------------------
# Model Accuracy Thresholds (Backlog H2)
# Governance: models exceeding ROLLBACK threshold are auto-disabled.
# ---------------------------------------------------------------------------
MAPE_THRESHOLD_EXCELLENT = 3.0  # % — exceeding expectations
MAPE_THRESHOLD_TARGET = 5.0  # % — performing well
MAPE_THRESHOLD_ACCEPTABLE = 10.0  # % — model is usable
MAPE_THRESHOLD_ROLLBACK = 15.0  # % — model disabled, fallback to next-best

# Per-horizon targets (longer horizons get more slack)
MAPE_BY_HORIZON: dict[str, dict[str, float]] = {
    "24h": {"excellent": 2.0, "target": 3.5, "acceptable": 7.0, "rollback": 12.0},
    "48h": {"excellent": 3.0, "target": 5.0, "acceptable": 10.0, "rollback": 15.0},
    "72h": {"excellent": 4.0, "target": 6.5, "acceptable": 12.0, "rollback": 18.0},
    "7d": {"excellent": 6.0, "target": 9.0, "acceptable": 15.0, "rollback": 22.0},
}


def mape_grade(mape: float, horizon: str = "48h") -> str:
    """Return a governance grade for the given MAPE and forecast horizon.

    Returns one of: 'excellent', 'target', 'acceptable', 'rollback'.
    """
    thresholds = MAPE_BY_HORIZON.get(horizon, MAPE_BY_HORIZON["48h"])
    if mape <= thresholds["excellent"]:
        return "excellent"
    if mape <= thresholds["target"]:
        return "target"
    if mape <= thresholds["acceptable"]:
        return "acceptable"
    return "rollback"


# ---------------------------------------------------------------------------
# Performance Targets (Backlog F1)
# ---------------------------------------------------------------------------
TAB_LOAD_P95_SECONDS = 2.0  # p95 tab load time target

# ---------------------------------------------------------------------------
# Rate Limiting (D3 — API Key Rotation & Rate Limiting)
# ---------------------------------------------------------------------------
INITIAL_BACKOFF_SECONDS = 1.0
MAX_RETRIES = 4
RATE_LIMIT_ALERT_THRESHOLD = 3  # consecutive 429s before alerting

# EIA API key management:
# - dev/staging:  EIA_API_KEY in .env file
# - production:   GCP Secret Manager → `eia-api-key`
# - rotation:     Regenerate at https://www.eia.gov/opendata/register.php
#                 Update secret: gcloud secrets versions add eia-api-key --data-file=key.txt
#                 Cloud Run picks up new version on next cold start.

# ---------------------------------------------------------------------------
# Precomputation (Startup Cache Warming)
# ---------------------------------------------------------------------------
PRECOMPUTE_ENABLED = os.getenv("PRECOMPUTE_ENABLED", "true").lower() in ("true", "1", "yes")
PRECOMPUTE_DEFAULT_REGION = os.getenv("PRECOMPUTE_DEFAULT_REGION", "FPL")
PRECOMPUTE_ALL_REGIONS = os.getenv("PRECOMPUTE_ALL_REGIONS", "true").lower() in ("true", "1", "yes")
PRECOMPUTE_MAX_WORKERS = int(os.getenv("PRECOMPUTE_MAX_WORKERS", "4"))
PRECOMPUTE_INTERVAL_HOURS = int(os.getenv("PRECOMPUTE_INTERVAL_HOURS", "8"))
PRECOMPUTE_ALL_MODELS = os.getenv("PRECOMPUTE_ALL_MODELS", "true").lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# Feature Flags (Backlog J2 — simple in-code toggles)
# ---------------------------------------------------------------------------
FEATURE_FLAGS: dict[str, bool] = {
    "tab_forecast": True,
    "tab_weather": True,
    "tab_models": True,
    "tab_generation": True,
    "tab_alerts": True,
    "tab_simulator": True,
    "persona_switcher": True,
    "scenario_presets": True,
    "scenario_bookmarks": True,  # Sprint 4 (C2)
    "api_fallback_badges": True,  # Sprint 4 (G2)
    "ai_insights": True,  # Persona-aware insight cards on all tabs
    "ai_briefing": True,  # Claude-powered executive briefing on Overview tab
}


def feature_enabled(flag: str) -> bool:
    """Check if a feature flag is enabled. Unknown flags default to True."""
    return FEATURE_FLAGS.get(flag, True)
