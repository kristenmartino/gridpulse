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
        # Dev: allow the in-process precompute thread so contributors can
        # work without Cloud Run Jobs + Redis. The web callbacks keep their
        # v1 compute fallback when REQUIRE_REDIS is false.
        "precompute_enabled": True,
        "require_redis": False,
    },
    "staging": {
        "cache_ttl": 86400,
        "min_inst": 0,
        "max_inst": 2,
        "demo": False,
        "profile": False,
        "workers": 2,
        "gcs_enabled": True,
        # Staging + production: pipeline runs in Cloud Run Jobs on a cron
        # schedule (hourly scoring, daily training). The web service is a
        # Redis-only reader — never fetches/trains inline.
        "precompute_enabled": False,
        "require_redis": True,
    },
    "production": {
        "cache_ttl": 86400,
        "min_inst": 1,
        "max_inst": 4,
        "demo": False,
        "profile": False,
        "workers": 2,
        "gcs_enabled": True,
        "precompute_enabled": False,
        "require_redis": True,
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

# When True the web callbacks (load_data, _run_forecast_outlook,
# _run_backtest_for_horizon) must NOT fall back to synchronous API fetches
# or inline model training. Cache misses surface as a degraded / "warming"
# UI state. Staging and production default to True because Cloud Run Jobs
# own the pipeline; dev defaults to False so contributors can run the app
# end-to-end without Redis or the scheduled jobs.
REQUIRE_REDIS = os.getenv("REQUIRE_REDIS", str(_env.get("require_redis", False))).lower() in (
    "true",
    "1",
    "yes",
)

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
# Regions — 16 Balancing Authorities (~98% of US load coverage)
#
# Lat/lon are weather-lookup proxies — the load center of each BA's territory,
# not a geometric centroid. Open-Meteo pulls all 17 weather variables at this
# coordinate; we treat it as representative of the BA's demand-driving climate.
# ---------------------------------------------------------------------------
REGION_COORDINATES: dict[str, dict] = {
    # Original 8 (ISOs/RTOs + FPL).
    "ERCOT": {"lat": 31.0, "lon": -97.0, "name": "Texas (ERCOT)"},
    "CAISO": {"lat": 37.0, "lon": -120.0, "name": "California (CAISO)"},
    "PJM": {"lat": 39.5, "lon": -77.0, "name": "Mid-Atlantic (PJM)"},
    "MISO": {"lat": 41.0, "lon": -89.0, "name": "Midwest (MISO)"},
    "NYISO": {"lat": 42.5, "lon": -74.0, "name": "New York (NYISO)"},
    "FPL": {"lat": 26.9, "lon": -80.1, "name": "Florida (FPL/NextEra)"},
    "SPP": {"lat": 35.5, "lon": -97.5, "name": "Southwest (SPP)"},
    "ISONE": {"lat": 42.3, "lon": -71.8, "name": "New England (ISO-NE)"},
    # V1.α expansion — 8 utility/federal BAs covering the Southeast, Pacific
    # Northwest, Desert Southwest, and Front Range. Coordinates anchor the
    # primary load center (Atlanta for SOCO, Nashville for TVA, etc.).
    "SOCO": {"lat": 33.7, "lon": -84.4, "name": "Southeast (Southern Co.)"},
    "TVA": {"lat": 36.2, "lon": -86.8, "name": "Tennessee Valley (TVA)"},
    "DUK": {"lat": 35.2, "lon": -80.8, "name": "Carolinas West (DEC)"},
    "CPLE": {"lat": 35.8, "lon": -78.6, "name": "Carolinas East (DEP)"},
    "BPAT": {"lat": 45.5, "lon": -122.7, "name": "Pacific NW (BPA)"},
    "AZPS": {"lat": 33.4, "lon": -112.1, "name": "Arizona (APS)"},
    "NEVP": {"lat": 36.2, "lon": -115.1, "name": "Southern Nevada (NV Energy)"},
    "PSCO": {"lat": 39.7, "lon": -105.0, "name": "Colorado (Xcel)"},
}

REGION_NAMES: dict[str, str] = {k: v["name"] for k, v in REGION_COORDINATES.items()}

# ---------------------------------------------------------------------------
# Generation Capacity (MW)
# Used by the scenario simulator's merit-order pricing model.
# Values reflect total installed nameplate capacity from each BA's most
# recent 2024–2025 publication (ISOs/RTOs are nonprofits and do not issue
# shareholder reports; their State-of-the-Market / Power Trends / Regional
# System Plan reports are the equivalent source of truth). For investor-owned
# utility BAs, figures come from each utility's 10-K, integrated resource
# plan (IRP), or the regulator's electric resource plan (ERP) of record.
#
# Original 8 (ISOs/RTOs + FPL):
#   ERCOT — ~153,000 MW installed (CDR summer 2025; wind, solar, battery,
#           thermal). https://www.ercot.com/gridinfo/resource
#   CAISO — 86,000 MW (CAISO 2024 Annual Report on Market Issues &
#           Performance, published Aug 2025).
#           https://www.caiso.com/documents/2024-annual-report-on-market-issues-and-performance.pdf
#   PJM   — 184,202 MW (2025 State of the Market Report, Monitoring
#           Analytics).
#           https://www.monitoringanalytics.com/reports/PJM_State_of_the_Market/2025/
#   MISO  — 186,986 MW (MISO Fast Facts 2025).
#           https://www.misoenergy.org/about/miso-strategy-and-value-proposition/miso-fast-facts/
#   NYISO — 37,375 MW (NYISO 2025 Power Trends, summer 2024 capability).
#           https://www.nyiso.com/documents/20142/2223020/2025-Power-Trends.pdf
#   FPL   — 35,963 MW net generating capacity (NextEra Energy 2024 10-K,
#           filed Feb 2025).
#           https://www.investor.nexteraenergy.com/financial-information/sec-filings
#   SPP   — 102,376 MW nameplate (derived from SPP Fast Facts 2025: wind
#           35,740 MW = 34.9% of total nameplate).
#           https://www.spp.org/about-us/fast-facts/
#   ISONE — 30,000 MW (ISO-NE 2025 Regional System Plan — "nearly
#           30,000 MW of generating capacity").
#           https://www.iso-ne.com/static-assets/documents/100030/final_2025_rsp.pdf
#
# V1.α expansion (utility / federal BAs):
#   SOCO  — 46,000 MW rate-regulated retail generation across Alabama
#           Power, Georgia Power, and Mississippi Power (Southern Company
#           2024 Annual Report). Excludes Southern Power's wholesale 13 GW.
#           https://s27.q4cdn.com/273397814/files/doc_financials/2023/ar/2024-annual-report.pdf
#   TVA   — ~35,000 MW (TVA fleet summary; 7 nuclear units, 4 coal plants,
#           18 gas plants, 29 hydro dams, 1 pumped storage, 14 solar sites).
#           https://www.tva.com/energy/our-power-system
#   DUK   — 20,800 MW (Duke Energy Carolinas — DEC; 2025 Carolinas Resource
#           Plan filing, Sep 2025).
#           https://news.duke-energy.com/releases/duke-energy-files-2025-carolinas-resource-plan-continues-modernizing-energy-infrastructure-to-support-future-growth
#   CPLE  — 13,700 MW (Duke Energy Progress East — DEP; derived from the
#           same 2025 Carolinas Resource Plan: combined DEC+DEP fleet is
#           34,500 MW, less the 20,800 MW DEC figure above).
#   BPAT  — 17,462 MW federal-system capacity (predominantly hydro; BPA
#           2024 White Book — Pacific Northwest Loads and Resources Study).
#           https://www.bpa.gov/-/media/Aep/power/white-book/2024-white-book.pdf
#   AZPS  — 9,400 MW (Arizona Public Service 2023 IRP, ACC-approved 2024).
#           https://www.aps.com/en/About/Our-Company/Doing-Business-with-Us/Resource-Planning
#   NEVP  — ~8,000 MW (Nevada Power 2024 IRP filed with PUCN; approximate —
#           NV Energy's NEVP-only fleet is not separately disclosed, this
#           is derived from EIA-930 BA-level annual generation of ~37 TWh
#           at typical IOU capacity factor and the 2024 IRP capacity
#           additions). Verify against EIA-860 form on next data refresh.
#           https://www.nvenergy.com/integrated-resource-plan
#   PSCO  — 9,080 MW (Public Service Co. of Colorado / Xcel Colorado;
#           EIA-860 BA-level installed capacity per Form 860 Schedule 6,
#           reflected in EIA grid monitor PSCO BA page).
#           https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/balancing_authority/PSCO
# ---------------------------------------------------------------------------
REGION_CAPACITY_MW: dict[str, int] = {
    # Original 8.
    "ERCOT": 153_000,
    "CAISO": 86_000,
    "PJM": 184_202,
    "MISO": 186_986,
    "NYISO": 37_375,
    "FPL": 35_963,
    "SPP": 102_376,
    "ISONE": 30_000,
    # V1.α — 8 new BAs.
    "SOCO": 46_000,
    "TVA": 35_000,
    "DUK": 20_800,
    "CPLE": 13_700,
    "BPAT": 17_462,
    "AZPS": 9_400,
    "NEVP": 8_000,
    "PSCO": 9_080,
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
    # V1.α expansion — mapping primary load states. Some states overlap with
    # ISO/RTO BAs (e.g. NC is partially Duke and partially PJM); alerts in
    # those states fire across all relevant BAs.
    "SOCO": ["AL", "GA", "MS"],
    "TVA": ["TN", "AL", "KY", "MS"],
    "DUK": ["NC", "SC"],
    "CPLE": ["NC", "SC"],
    "BPAT": ["WA", "OR", "ID", "MT"],
    "AZPS": ["AZ"],
    "NEVP": ["NV"],
    "PSCO": ["CO"],
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
    "tab-models",
    "tab-backtest",
    "tab-generation",
    "tab-weather",
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
# In dev this keeps the legacy in-process precompute thread available so
# contributors without Cloud Run Jobs can still warm caches on startup.
# In staging/production the daily training job + hourly scoring job own
# the pipeline, so the default is False (see `_ENV_DEFAULTS`).
PRECOMPUTE_ENABLED = os.getenv(
    "PRECOMPUTE_ENABLED", str(_env.get("precompute_enabled", False))
).lower() in ("true", "1", "yes")
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
    "what_changed": True,  # NEXD-8: session-aware change detection
    "smart_defaults": True,  # NEXD-9: remember last filter state in localStorage
    "cross_tab_links": True,  # NEXD-11: contextual links between tabs
    "inline_tooltips": True,  # NEXD-13: SHAP-based per-point forecast tooltips
    # NEXD-14 / shell-redesign post-R6: replay surfaces stale snapshots and
    # competes with the v2 Forecast tab's hero rhythm. Re-enable once the
    # snapshot pipeline is producing fresh data; backtesting belongs in the
    # Models tab in the meantime.
    "forecast_replay": False,
}


def feature_enabled(flag: str) -> bool:
    """Check if a feature flag is enabled. Unknown flags default to True."""
    return FEATURE_FLAGS.get(flag, True)
