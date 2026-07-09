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

# Free-tier ``/forecast`` endpoint covers 16 days (GFS-based) hourly.
# Single source of truth shared by ``data.weather_client.fetch_weather``
# (forecast_days default) and ``components._callbacks_forecast`` (the
# day-16 boundary marker on the Forecast tab where real-forecast data
# transitions to climatological fallback). See ADR-008 in PRD.md.
OPEN_METEO_FORECAST_DAYS = 16
OPEN_METEO_FORECAST_HOURS = OPEN_METEO_FORECAST_DAYS * 24  # 384

NOAA_BASE_URL = "https://api.weather.gov"

# AI Briefing (Overview tab executive briefing via Claude)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# Redis (v2 pre-computation pipeline)
# ---------------------------------------------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Key prefix for every Redis namespace this app touches. Renamed from
# ``wattcast`` (an older project name) to ``gridpulse`` to match the
# product (issue #91). Override via the ``REDIS_KEY_PREFIX`` env var if
# you need to point at a different namespace — e.g. when running an
# experimental scoring job that shouldn't clobber production keys.
#
# On the deploy that flipped this default, the web service returned
# "Data warming up" until the next hourly scoring run populated the
# ``gridpulse:*`` keys. Old ``wattcast:*`` keys TTL out within 24h.
REDIS_KEY_PREFIX = os.getenv("REDIS_KEY_PREFIX", "gridpulse")

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
# Freshness measurement (2026-07 critical review, finding P1-3)
# ---------------------------------------------------------------------------
# Freshness must be MEASURED from each Redis payload's own scored_at, never
# asserted at render time. Rationale for the thresholds: the scoring job
# runs hourly, so one missed tick (<=2h) is a tolerable hiccup while two is
# an outage signal; demand actuals additionally tolerate EIA-930's normal
# publishing lag (~1-4h; see #129) before the *data itself* counts as stale.
FRESHNESS_FRESH_MAX_AGE_HOURS = float(os.getenv("FRESHNESS_FRESH_MAX_AGE_HOURS", "2.0"))
FRESHNESS_DEMAND_LAG_ALLOWANCE_HOURS = float(
    os.getenv("FRESHNESS_DEMAND_LAG_ALLOWANCE_HOURS", "6.0")
)

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
# Regions — 51 Balancing Authorities (~100% of contiguous-US lower-48 load)
#
# Original 8 (ISOs/RTOs + FPL) + V1.α expansion (8 utility/federal BAs) +
# V3.ζ expansion (35 remaining EIA-930 BAs in the lower 48).
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
    # V3.ζ expansion — adds the remaining 35 EIA-930 BAs in the contiguous
    # US, bringing coverage to ~100% of US lower-48 demand. Coordinates
    # anchor each BA's primary load center / utility HQ. Tiny BAs (CPLW
    # 42 MW, HST 36 MW, GVL 600 MW, SPA federal hydro marketer) are
    # included for completeness — they may produce noisy forecasts that
    # downstream NaN guards (PR #71) handle gracefully.
    # Southeast (11):
    "FPC": {"lat": 27.95, "lon": -82.46, "name": "Florida (Duke FL)"},
    "TEC": {"lat": 27.95, "lon": -82.46, "name": "Tampa Bay (TECO)"},
    "FMPP": {"lat": 28.54, "lon": -81.38, "name": "Florida Muni Pool"},
    "JEA": {"lat": 30.33, "lon": -81.66, "name": "Jacksonville (JEA)"},
    "TAL": {"lat": 30.44, "lon": -84.28, "name": "Tallahassee"},
    "GVL": {"lat": 29.65, "lon": -82.32, "name": "Gainesville (GRU)"},
    "SEC": {"lat": 27.95, "lon": -82.46, "name": "Seminole Electric"},
    "HST": {"lat": 25.47, "lon": -80.48, "name": "Homestead"},
    "SC": {"lat": 33.20, "lon": -80.01, "name": "Santee Cooper"},
    "SCEG": {"lat": 34.00, "lon": -81.03, "name": "Carolinas Mid (Dominion SC)"},
    "CPLW": {"lat": 35.60, "lon": -82.55, "name": "DEP-West (NC mountains)"},
    # Central (4 — all Mississippi-watershed / non-WECC interior):
    "LGEE": {"lat": 38.25, "lon": -85.76, "name": "Kentucky (LG&E + KU)"},
    "AECI": {"lat": 37.21, "lon": -93.30, "name": "Missouri (AECI)"},
    "EPE": {"lat": 31.76, "lon": -106.49, "name": "El Paso (EPE)"},
    "SPA": {"lat": 36.15, "lon": -95.99, "name": "SW Power Admin"},
    # West (20 — Pacific NW, California, Mountain West, Desert SW):
    "PACE": {"lat": 40.76, "lon": -111.89, "name": "Inland West (PacifiCorp E)"},
    "PACW": {"lat": 45.51, "lon": -122.68, "name": "Pacific NW (PacifiCorp W)"},
    "PGE": {"lat": 45.51, "lon": -122.68, "name": "Portland General"},
    "PSEI": {"lat": 47.61, "lon": -122.20, "name": "Puget Sound Energy"},
    "SCL": {"lat": 47.61, "lon": -122.33, "name": "Seattle (SCL)"},
    "TPWR": {"lat": 47.25, "lon": -122.44, "name": "Tacoma Power"},
    "AVA": {"lat": 47.66, "lon": -117.43, "name": "Spokane (Avista)"},
    "IPCO": {"lat": 43.62, "lon": -116.20, "name": "Idaho (Idaho Power)"},
    "NWMT": {"lat": 46.00, "lon": -112.53, "name": "Montana (NorthWestern)"},
    "GCPD": {"lat": 47.32, "lon": -119.55, "name": "Grant County PUD"},
    "CHPD": {"lat": 47.42, "lon": -120.31, "name": "Chelan County PUD"},
    "DOPD": {"lat": 47.42, "lon": -120.30, "name": "Douglas County PUD"},
    "BANC": {"lat": 38.58, "lon": -121.49, "name": "Sacramento (BANC)"},
    "LDWP": {"lat": 34.05, "lon": -118.24, "name": "Los Angeles (LADWP)"},
    "IID": {"lat": 32.79, "lon": -115.56, "name": "Imperial Valley (IID)"},
    "TIDC": {"lat": 37.49, "lon": -120.85, "name": "Turlock ID"},
    "SRP": {"lat": 33.45, "lon": -112.07, "name": "Phoenix (SRP)"},
    "TEPC": {"lat": 32.22, "lon": -110.93, "name": "Tucson (TEP)"},
    "PNM": {"lat": 35.08, "lon": -106.65, "name": "New Mexico (PNM)"},
    "WALC": {"lat": 33.45, "lon": -112.07, "name": "Desert SW (WAPA-DSW)"},
}

REGION_NAMES: dict[str, str] = {k: v["name"] for k, v in REGION_COORDINATES.items()}

# ---------------------------------------------------------------------------
# Regional groupings — used by the header dropdown and the US Grid card
# grid to surface geographic context. Groups are sorted A-Z by group
# name; BA codes within each group are sorted A-Z by code (the
# dictionary literal order below IS the render order — Python ≥3.7
# preserves insertion order, and ``test_config.py`` guards both
# orderings + total coverage of ``REGION_COORDINATES``).
# ---------------------------------------------------------------------------
REGION_GROUPS: dict[str, list[str]] = {
    "Central": ["AECI", "EPE", "ERCOT", "LGEE", "MISO", "SPA", "SPP"],
    "Northeast": ["ISONE", "NYISO", "PJM"],
    "Southeast": [
        "CPLE",
        "CPLW",
        "DUK",
        "FMPP",
        "FPC",
        "FPL",
        "GVL",
        "HST",
        "JEA",
        "SC",
        "SCEG",
        "SEC",
        "SOCO",
        "TAL",
        "TEC",
        "TVA",
    ],
    "West": [
        "AVA",
        "AZPS",
        "BANC",
        "BPAT",
        "CAISO",
        "CHPD",
        "DOPD",
        "GCPD",
        "IID",
        "IPCO",
        "LDWP",
        "NEVP",
        "NWMT",
        "PACE",
        "PACW",
        "PGE",
        "PNM",
        "PSCO",
        "PSEI",
        "SCL",
        "SRP",
        "TEPC",
        "TIDC",
        "TPWR",
        "WALC",
    ],
}


def grouped_regions() -> list[tuple[str, list[str]]]:
    """Return ``REGION_GROUPS`` items as a list — convenience for callers
    that iterate (group_name, codes) tuples without importing the dict
    directly. Order matches ``REGION_GROUPS`` (groups A-Z, codes A-Z).
    """
    return list(REGION_GROUPS.items())


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
#   NEVP  — 15,445 MW operating (EIA-860M February 2026, retrieved via
#           EIA API v2 on 2026-05-01: 261 generators in the NEVP BA
#           summed at the nameplate-capacity-mw field, filtered to
#           statusDescription="Operating"). This is the BA-level fleet
#           total — every generator in the territory including IPPs and
#           wholesale sellers, not just NV Energy's own fleet. Supersedes
#           the 8,000 MW first-pass estimate from V1.α (which conflated
#           NV Energy's utility-owned fleet with the BA total).
#           https://api.eia.gov/v2/electricity/operating-generator-capacity/
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
    # V1.α — 8 new BAs. V3.η (2026-05-02) corrected SOCO / DUK / CPLE /
    # PSCO from EIA-860M generator capacity to peak-demand × 1.15
    # reserve margin because in-territory generation is below served
    # demand for these net-importer utility BAs (capacity / peak ratios
    # were 0.93–0.96, indicating regular imports). Peak source: EIA-930
    # 12-month max demand per ``data/eia_client.fetch_demand``.
    "SOCO": 54_980,  # was 46,000 (EIA-860M); peak 47,809 × 1.15
    "TVA": 35_000,
    "DUK": 25_513,  # was 20,800 (EIA-860M); peak 22,186 × 1.15
    "CPLE": 16_478,  # was 13,700 (EIA-860M); peak 14,329 × 1.15
    "BPAT": 17_462,
    "AZPS": 9_400,
    "NEVP": 15_445,
    "PSCO": 12_238,  # was 9,080 (EIA-860M); peak 10,642 × 1.15
    # V3.ζ — 35 remaining EIA-930 BAs in the contiguous US. Capacities
    # all sourced from EIA-860M Feb 2026 (one batch query, methodology
    # mirrors V3.ε for NEVP — sum nameplate-capacity-mw rows filtered
    # to statusDescription="Operating" via the EIA API v2 endpoint
    # /electricity/operating-generator-capacity/data/, retrieved on
    # 2026-05-02). SCL fell back to Seattle City Light's 2024 annual
    # report (~1,800 MW; the API timed out for that one BA).
    # Southeast (11):
    "FPC": 16_535,
    "TEC": 8_747,
    "FMPP": 4_574,  # V3.η: was 3,908 (EIA-860M); peak 3,978 × 1.15
    "JEA": 3_214,
    "TAL": 1_024,
    "GVL": 600,
    "SEC": 2_826,
    # V3.η: Homestead is import-dominated (peak 147 MW vs 36 MW
    # in-territory generation = 4.08×). Corrected to peak × 1.15 and
    # tagged in IS_IMPORT_DOMINATED so the UI hides its stress chip.
    "HST": 169,  # was 36; peak 147 × 1.15
    "SC": 5_968,
    "SCEG": 8_122,
    # V3.η: DEP-West (CPLW, NC mountains) is severely import-dominated
    # (peak 1,261 MW vs 42 MW in-territory generators = 30×). User-
    # reported 2026-05-02 as "Highest-Stress: CPLW · 1071%". Corrected
    # to peak × 1.15 and tagged in IS_IMPORT_DOMINATED.
    "CPLW": 1_450,  # was 42; peak 1,261 × 1.15
    # Central (4):
    "LGEE": 9_074,
    "AECI": 6_650,
    "EPE": 2_849,
    "SPA": 2_559,
    # West (20):
    "PACE": 18_692,
    "PACW": 2_628,
    "PGE": 4_168,
    "PSEI": 2_960,
    "SCL": 1_800,
    "TPWR": 725,
    "AVA": 2_272,
    "IPCO": 4_724,
    "NWMT": 4_235,
    "GCPD": 1_220,
    "CHPD": 1_923,
    "DOPD": 1_289,
    "BANC": 5_583,
    "LDWP": 9_808,
    "IID": 2_196,
    "TIDC": 808,
    "SRP": 18_020,
    "TEPC": 4_445,
    "PNM": 7_544,
    "WALC": 7_096,
}

# ---------------------------------------------------------------------------
# Import-dominated BAs
# ---------------------------------------------------------------------------
# Some EIA-930 balancing authorities serve far more demand than their
# in-territory generators can produce — they're structurally dependent on
# imports from neighbours. Showing these BAs' "stress" against their
# nameplate capacity is misleading: a 30× import multiplier (CPLW) makes
# every reading look critical, and even after the V3.η peak × 1.15
# capacity correction, the underlying number is still an *estimate* of
# the resource pool the BA can draw on, not a measured plate capacity.
#
# The UI uses this set to:
#   1. Suppress the BA from the "highest-stress" KPI candidate pool so
#      a 30× multiplier doesn't always win (PR #76 capped the display
#      at 100% but didn't fix candidacy).
#   2. Annotate the polygon hover + drilldown with an "import-dominated
#      · capacity is estimated" footnote so users understand what the
#      utilization % is actually measured against.
#
# Inclusion criterion: in-territory generation < ~50% of 12-month peak
# demand (i.e. multiplier ≥ 2×). Sourced from EIA-860M generator rows
# vs EIA-930 demand peaks. Re-evaluate annually as new generators
# come online.
IS_IMPORT_DOMINATED: frozenset[str] = frozenset(
    {
        "CPLW",  # DEP-West, NC mountains. Peak 1,261 MW vs 42 MW gen = 30×.
        "HST",  # Homestead, FL. Peak 147 MW vs 36 MW gen = 4.08×.
        "SPA",  # Southwestern Power Admin — federal hydro *marketer*,
        #        not a vertically integrated utility. Its 2,559 MW
        #        nameplate is the federal dam fleet; the served load
        #        is far larger via long-term power contracts.
        "PACW",  # PacifiCorp West (OR/WA/N.CA). In-territory nameplate
        #         (2,628 MW) ≈ served load — the PacifiCorp fleet is mostly
        #         in the East (PACE, WY/UT coal); PACW imports across the
        #         system, so util-vs-nameplate crowned it "100% stress" (#225).
    }
)

# ---------------------------------------------------------------------------
# Peak-derived-capacity BAs (#254)
# ---------------------------------------------------------------------------
# These BAs carry ``REGION_CAPACITY_MW = 12-month peak demand × 1.15`` (a
# reserve-margin proxy) rather than a measured EIA-860M nameplate — their
# in-territory generation runs below served load (V3.η, 2026-05-02). That
# makes utilization = demand / (peak × 1.15) **self-referential**: at its own
# historical peak a BA reads exactly ~87% (1 / 1.15) and can never surface as
# stressed above that, so the ratio is not a meaningful stress signal. They are
# therefore excluded from ``national_utilization_pct`` / ``top_stress`` and the
# stress sort, the same way import-dominated BAs are, and the public API labels
# their capacity ``capacity_source = "peak_estimate"`` rather than "nameplate".
#
# HST and CPLW are peak-derived too but are ALSO in IS_IMPORT_DOMINATED (their
# import multiplier is the dominant story), so the union below covers them.
# SPA is deliberately absent: it is import-dominated but its 2,559 MW IS a true
# nameplate (the federal dam fleet), so it stays "nameplate" for the API.
PEAK_DERIVED_CAPACITY: frozenset[str] = frozenset(
    {"SOCO", "DUK", "CPLE", "PSCO", "FMPP", "HST", "CPLW"}
)

#: BAs excluded from stress/utilization aggregates because their capacity figure
#: is not a reliable measured plate — import-dominated (served load >> in-territory
#: generation) OR peak-derived (plate is a peak × 1.15 estimate). Single source of
#: truth for "does this BA's utilization mean anything as a stress reading."
UNRELIABLE_CAPACITY: frozenset[str] = IS_IMPORT_DOMINATED | PEAK_DERIVED_CAPACITY


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
    # V3.ζ expansion — primary load states for the remaining 35 BAs.
    # Florida muni / cooperative BAs are co-located with FPL territory.
    "FPC": ["FL"],
    "TEC": ["FL"],
    "FMPP": ["FL"],
    "JEA": ["FL"],
    "TAL": ["FL"],
    "GVL": ["FL"],
    "SEC": ["FL"],
    "HST": ["FL"],
    "SC": ["SC"],
    "SCEG": ["SC"],
    "CPLW": ["NC"],
    "LGEE": ["KY"],
    "AECI": ["MO", "IA", "OK", "AR"],
    "EPE": ["TX", "NM"],
    "SPA": ["AR", "MO", "OK", "KS", "LA"],
    "PACE": ["UT", "WY", "ID"],
    "PACW": ["OR", "CA", "WA"],
    "PGE": ["OR"],
    "PSEI": ["WA"],
    "SCL": ["WA"],
    "TPWR": ["WA"],
    "AVA": ["WA", "ID"],
    "IPCO": ["ID", "OR"],
    "NWMT": ["MT"],
    "GCPD": ["WA"],
    "CHPD": ["WA"],
    "DOPD": ["WA"],
    "BANC": ["CA"],
    "LDWP": ["CA"],
    "IID": ["CA"],
    "TIDC": ["CA"],
    "SRP": ["AZ"],
    "TEPC": ["AZ"],
    "PNM": ["NM"],
    "WALC": ["AZ", "NV", "CA", "NM"],
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
    "tab-us-grid",
    "tab-outlook",
    "tab-alerts",
    "tab-models",
]

TAB_LABELS: dict[str, str] = {
    "tab-overview": "Overview",
    "tab-us-grid": "US Grid",
    "tab-outlook": "Forecast",
    "tab-alerts": "Risk",
    "tab-models": "Models",
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

# Per-horizon targets (longer horizons get more slack). The ``1h`` band is for
# the live 1-hour-ahead drift metric (models/drift.py) and is the tightest —
# anchored to real short-term load forecasting (~0.5–2% MAPE one-hour-ahead),
# so a poor 1h number can't be laundered as "acceptable" by a looser band.
MAPE_BY_HORIZON: dict[str, dict[str, float]] = {
    "1h": {"excellent": 1.0, "target": 2.5, "acceptable": 5.0, "rollback": 8.0},
    "24h": {"excellent": 2.0, "target": 3.5, "acceptable": 7.0, "rollback": 12.0},
    "48h": {"excellent": 3.0, "target": 5.0, "acceptable": 10.0, "rollback": 15.0},
    "72h": {"excellent": 4.0, "target": 6.5, "acceptable": 12.0, "rollback": 18.0},
    "7d": {"excellent": 6.0, "target": 9.0, "acceptable": 15.0, "rollback": 22.0},
}

# Ensemble weighting exponent (ADR-004 refinement, resolves #181). The served
# ensemble weight_i is proportional to (1/MAPE_i)^k. k=1.0 is plain inverse-MAPE;
# k=3.0 sharpens toward the best model, blending meaningfully only when peers are
# genuinely close. Validated on the 51-BA recursive holdout (2026-07-04): k=3
# beats k=1 on 47/51 BAs (median 4.19% -> 3.90%) and holds up under a held-out
# even/odd-hour split, so it is not overfit to one window. Plain inverse-MAPE
# (k=1) kept 15-30% weight on models running 3-5x worse than the leader. See
# docs/BACKTEST_RESULTS.md "Ensemble weighting" and PRD.md ADR-004.
ENSEMBLE_WEIGHT_EXPONENT: float = 3.0


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
# Scoring-job runtime headroom guardrail (#171)
# ---------------------------------------------------------------------------
# The hourly scoring job runs under a Cloud Run ``--task-timeout``. Runtime
# creeps up as BAs/features grow (the 2026-06-01 incident tipped it over the
# then-900s cap), but the PR-G10 failure alert only fires on an OUTRIGHT
# timeout — too late, a tick is already killed. This guardrail warns on
# APPROACH: when a completed run's ``elapsed_s`` exceeds
# ``SCORING_RUNTIME_HEADROOM_FRACTION`` of the timeout for
# ``SCORING_RUNTIME_CREEP_RUNS`` consecutive runs, the job emits a
# ``scoring_runtime_creep`` alert log (matched by the Cloud Monitoring policy in
# docs/monitoring/). Keep ``SCORING_TASK_TIMEOUT_S`` in sync with the
# ``--task-timeout`` in deploy-prod.yml / deploy-dev.yml.
SCORING_TASK_TIMEOUT_S = int(os.getenv("SCORING_TASK_TIMEOUT_S", "1800"))
SCORING_RUNTIME_HEADROOM_FRACTION = float(os.getenv("SCORING_RUNTIME_HEADROOM_FRACTION", "0.70"))
SCORING_RUNTIME_CREEP_RUNS = int(os.getenv("SCORING_RUNTIME_CREEP_RUNS", "3"))
# #267: a hourly scoring run that forecasts fewer than this many BAs emits an
# alertable ``scoring_partial_failure`` ERROR (matched by a log-based Cloud
# Monitoring policy) and marks the freshness meta degraded, so a catastrophic
# partial failure (e.g. 1/51 forecasts) can't hide behind a 0 exit code. A
# normal run scores ~48-51 (a few untrained/new BAs legitimately have no model),
# so this absolute floor tolerates the expected no-model tail without noise.
SCORING_MIN_OK_REGIONS = int(os.getenv("SCORING_MIN_OK_REGIONS", "40"))

# ---------------------------------------------------------------------------
# Web-tier operational guard (#253)
# ---------------------------------------------------------------------------
# The public JSON API (#250/#251) made the stateless web tier publicly
# programmable, but the project's operational tooling (job-failure alerting,
# deep /health, circuit breaker) protects only the JOB tier. These bound the
# blast radius of the now-public request path on personal billing.
#
# Per-IP request rate limits (fixed-window, Redis-backed). Tunable via env so
# a flood can be clamped without a redeploy.
API_RATE_LIMIT_PER_MIN = int(os.getenv("API_RATE_LIMIT_PER_MIN", "120"))
DASH_RATE_LIMIT_PER_MIN = int(os.getenv("DASH_RATE_LIMIT_PER_MIN", "600"))
# Trusted source IPs that bypass rate limiting entirely — for a known
# shared-NAT egress (e.g. a Grid Ops control room where many operators sit
# behind one corporate IP and would otherwise share a single per-IP bucket).
# Comma-separated env; empty by default. Entries may be exact IPs or CIDR
# prefixes (e.g. a corporate /24 or an IPv6 /64). Keyed on the same
# spoof-resistant IP the limiter uses, so it can't be forged via X-Forwarded-For.
RATE_LIMIT_EXEMPT_IPS: frozenset[str] = frozenset(
    ip.strip() for ip in os.getenv("RATE_LIMIT_EXEMPT_IPS", "").split(",") if ip.strip()
)
# Rate limiting is enforced only when the web tier is in Redis-only mode
# (staging/prod, i.e. REQUIRE_REDIS) AND the ``rate_limiting`` flag is on — dev
# is unthrottled and needs no Redis. Callers gate on ``rate_limit_active()``
# below; the limiter itself fails OPEN on any Redis error so a Redis blip never
# self-inflicts an availability outage.
# Reject oversized request bodies before they buffer into a 4Gi worker (an OOM
# amplifier). The API is GET-only and Dash callbacks POST small JSON, so 2 MiB
# is generous headroom; Flask returns 413 automatically past this.
MAX_REQUEST_BYTES = int(os.getenv("MAX_REQUEST_BYTES", str(2 * 1024 * 1024)))

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
    # V3.ζ follow-up (#255): hide a BA from the dropdown + US Grid cards only
    # when its *best served* forecast — the champion across the ensemble + 3
    # base models, not XGBoost-alone — exceeds the 7d rollback threshold
    # (22% — see MAPE_BY_HORIZON["7d"]["rollback"]). Gating on the served
    # champion keeps a BA visible whenever *any* served model forecasts it
    # acceptably (e.g. SEC: XGBoost 38.6% but ensemble 13.6% → visible); at the
    # 2026-07-03 training 0/51 trip it. Disable in dev to debug noisy regions.
    "forecast_quality_gate": True,
    # #253: per-IP request rate limiting on the public web surfaces
    # (/api/v1/* + the Dash callback route), Redis-backed. Enforced only when
    # the web tier is in Redis-only mode (staging/prod); see rate_limit_active().
    "rate_limiting": True,
}


def feature_enabled(flag: str) -> bool:
    """Check if a feature flag is enabled.

    Unknown flags default to ``False`` (fail-closed). A typo in a flag
    name should never silently *enable* behavior — it should disable it
    and surface a warning so the typo is caught. Every flag actually
    read in the codebase has an explicit entry in ``FEATURE_FLAGS``, so
    this default only fires on a genuine mistake.

    Changed 2026-05-29 (PR-G8 / #145) from fail-open to fail-closed per
    the production-readiness review: ``FEATURE_FLAGS.get(flag, True)``
    meant an unregistered flag silently turned a feature ON.
    """
    if flag not in FEATURE_FLAGS:
        # Lazy import keeps config.py free of a module-level logging
        # dependency (it's imported very early, by nearly everything).
        import structlog

        structlog.get_logger().warning(
            "feature_flag_unknown",
            flag=flag,
            known_flags=sorted(FEATURE_FLAGS.keys()),
            defaulting_to=False,
        )
        return False
    return FEATURE_FLAGS[flag]


def rate_limit_active() -> bool:
    """True when per-IP request rate limiting should be enforced (#253).

    Only in Redis-only mode (staging/prod, ``REQUIRE_REDIS``) with the
    ``rate_limiting`` flag on. Dev is unthrottled (no Redis to back it). Checked
    at request time so the flag/env can change without recomputing a constant.
    """
    return REQUIRE_REDIS and feature_enabled("rate_limiting")
