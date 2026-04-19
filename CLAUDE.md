# CLAUDE.md — Project Conventions for GridPulse

## Start here

This repo already has multiple context layers. Read them in this order:

1. `CLAUDE.md` — architecture, conventions, code standards, execution guardrails
2. `EXECUTION_BRIEF.md` — prioritization, redesign direction, product-shell changes, execution order
3. `README.md` — current public framing and deployment overview
4. `PRD.md` — product requirements, personas, ADRs, descoping rationale
5. `TECHNICAL_SPEC.md` — data/model/system details

### Agent objective

GridPulse is evolving from a technically credible energy demand forecasting dashboard into a more cohesive **energy intelligence platform** for forecast confidence, grid visibility, and operational decision support.

Your job is to improve product coherence, positioning, and UX **without breaking core functionality**.

### Required working style
- Inspect first
- Plan briefly
- Implement in small increments
- Preserve working behavior unless explicitly asked to change it
- Validate after meaningful changes
- Summarize what changed, why, and what remains

### Guardrails
- Do not rewrite unrelated systems.
- Do not change frameworks.
- Do not destabilize data ingestion, caching, model training, or the scheduled-jobs pipeline for surface-level UI work.
- Do not remove personas, model validation, or operational context just to simplify the UI.
- Do not add unsupported marketing claims.

---

## Product context

GridPulse is a **Dash/Plotly** application for weather-aware energy forecasting and grid analysis across 8 US balancing authorities.
It combines:
- demand data
- weather data
- multiple ML/statistical models
- backtesting and model validation
- generation and net load context
- alerts/extreme events concepts
- scenario simulation
- role-based views and briefings

### Working product framing
Use this framing unless a human directs otherwise:
- **Category:** Energy Intelligence Platform
- **Positioning:** Forecast confidence, grid visibility, and decision support
- **Tagline:** See demand sooner. Decide with confidence.

This framing should guide UI copy, navigation naming, and landing-page work. For prioritization of those changes, follow `EXECUTION_BRIEF.md`.

---

## Architecture

This is a **Dash/Plotly** dashboard application for weather-aware energy demand forecasting.
It uses 3 ML models (Prophet, SARIMAX, XGBoost) combined via a weighted ensemble
to forecast hourly electricity demand for 8 US balancing authorities.

### Runtime split (production)
- **Cloud Run Service (`gridpulse`)** — stateless Dash/Flask web app. Reads
  from Redis only; never fetches EIA/Open-Meteo or trains models in the
  request path. When Redis is cold, renders a `warming` degraded state.
- **Cloud Run Jobs (scheduled by Cloud Scheduler)**
  - `gridpulse-scoring-job` — hourly. Fetches EIA/weather, loads latest
    models from GCS, writes forecasts + alerts + diagnostics +
    weather-correlation to Redis. Entry point: `python -m jobs scoring`.
  - `gridpulse-training-job` — daily at 04:00 UTC. Trains XGBoost/Prophet/
    SARIMAX, persists to `gs://nextera-portfolio-energy-cache/models/`,
    writes backtests to Redis. Entry point: `python -m jobs training`.
- **Model store** — GCS at `gs://nextera-portfolio-energy-cache/models/` via
  `models/persistence.py`. Layout: `{region}/{model_name}/{version}.pkl` +
  `.meta.json`, atomically pointed to by `latest.json`. Scoring job pulls
  via `load_model()` with local disk cache at `/app/trained_models/`.
- **Redis gating** — `REQUIRE_REDIS` flag (true in staging/production, false
  in development) controls whether callbacks fall back to inline compute.
  See `components/callbacks.py` for the three warming gates.

Setup + bootstrap procedure: `docs/SCHEDULED_JOBS.md`.

### Active top-level tabs in the current shell
- Overview
- Historical Demand
- Demand Forecast
- Backtest
- Generation & Net Load
- Weather Correlation
- Model Diagnostics
- Extreme Events
- Scenario Simulator

### Key Decisions (ADRs)
- **ADR-001**: Dash + Plotly (not Streamlit) — callback architecture scales to many interaction groups
- **ADR-002**: SQLite cache on Cloud Run ephemeral disk — survives across requests, acceptable to lose on recycle
- **ADR-003**: Open-Meteo (not NOAA NWS) for weather — no API key, 17 variables in one call, historical + forecast support
- **ADR-004**: 1/MAPE weighted ensemble — simpler than stacking, self-correcting, bounded by individual models
- **ADR-005**: Scenario engine copies features, never mutates — pure function, safe for concurrent callbacks
- **ADR-006**: Full multi-tab architecture — overview → forecast/history → validate → grid/generation → weather/risk → simulator

### Module Map
```text
app.py                    → Dash app entry point, registers layout + callbacks
config.py                 → ALL constants: regions, API URLs, thresholds, pricing tiers, feature flags
observability.py          → Structured logging + pipeline transformation logging
components/
  layout.py               → Main layout: header, persona/view selector, region selector, tab shell
  callbacks.py            → ALL Dash callbacks and shared data-loading flows
  cards.py                → Reusable KPI, welcome, alert, briefing, and supporting cards
  error_handling.py       → Confidence badges, loading spinners, empty/error states
  accessibility.py        → Colorblind palette, ARIA helpers
  insights.py             → Persona-aware insight engine
  tab_overview.py         → Overview / mission-control screen
  tab_forecast.py         → Historical Demand tab (past actuals + EIA overlay)
  tab_demand_outlook.py   → Demand Forecast tab (forward predictions + confidence bands)
  tab_backtest.py         → Backtest tab (model evaluation on holdout)
  tab_generation.py       → Generation & Net Load tab (fuel mix, renewable share)
  tab_weather.py          → Weather-Energy Correlation tab
  tab_models.py           → Model comparison & diagnostics
  tab_alerts.py           → Extreme Events / alerts / stress indicators
  tab_simulator.py        → Scenario Simulator tab
data/
  cache.py                → SQLite cache with TTL + stale fallback behavior where applicable
  eia_client.py           → EIA API v2: demand, generation, interchange
  weather_client.py       → Open-Meteo: 17 weather vars, historical + forecast
  noaa_client.py          → NOAA/NWS: severe weather alerts
  news_client.py          → External news feed integration
  preprocessing.py        → Merge, align UTC, interpolate gaps <6h, flag gaps ≥6h
  feature_engineering.py  → 43 derived features: CDD/HDD, wind power, solar CF, lags, rolling
  demo_data.py            → Synthetic data generator for offline/demo mode where explicitly used
  audit.py                → Forecast audit trail (model version, data hash, feature hash)
models/
  prophet_model.py        → Prophet with weather regressors
  arima_model.py          → SARIMAX with pmdarima auto-order
  xgboost_model.py        → XGBoost with TimeSeriesSplit CV + SHAP
  ensemble.py             → 1/MAPE weighted combination
  evaluation.py           → MAPE, RMSE, MAE, R², residuals, error-by-hour
  model_service.py        → Forecast service layer: get_forecasts() with trained→simulated fallback
  training.py             → Orchestrator: train all → validate → compute weights → serialize
  pricing.py              → Merit-order pricing model
simulation/
  scenario_engine.py      → Copy → Override → Recompute → Reforecast → Delta
  presets.py              → Historical extreme scenarios
personas/
  config.py               → 4 personas: Grid Ops, Renewables, Trader, Data Scientist
  welcome.py              → Data-driven welcome messages
```

---

## Execution priorities for redesign work

When the task is related to branding, shell UX, navigation, or product coherence, prioritize in this order:

### P0
1. Product positioning and naming cleanup
2. Navigation / IA cleanup
3. Visual token and shell refresh
4. Overview redesign
5. Forecast refinement

### P1
1. Risk/alerts consolidation
2. Models/validation framing cleanup
3. Scenarios polish
4. Briefings / intelligence layer cleanup
5. Documentation alignment

### P2
1. Module/suite scaffolding
2. Broader landing-page and marketing assets
3. Mobile awareness patterns

Detailed guidance lives in `EXECUTION_BRIEF.md`. Use that file for sequencing and acceptance criteria.

---

## Code Standards

### Type Hints
All functions have type hints. Use `X | None` not `Optional[X]`.

### Logging
Always use structlog: `log = structlog.get_logger()`. Key-value pairs, not f-strings.
```python
log.info("data_loaded", region=region, rows=len(df))  # ✓
log.info(f"Loaded {len(df)} rows for {region}")       # ✗
```

### Docstrings
Google style. First line = what it does. Args/Returns sections for public functions.

### Commits
Format: `type(scope): description`
Types: feat, fix, refactor, test, docs, chore
Scopes: data, models, sim, personas, ui, infra

### Testing
- Unit tests: `tests/unit/test_*.py` — pure functions, no I/O
- Integration: `tests/integration/` — mocked API calls, cache roundtrips
- E2E: `tests/e2e/` — full dashboard rendering, tab switching
- Run: `pytest tests/ -v --cov=data --cov=models --cov=simulation --cov=personas --cov=components`

---

## Common Patterns

### API client pattern
1. Check cache
2. Fetch from API
3. Parse
4. Cache
5. Return

On API failure: prefer serving stale real data with warning logs when available.
Do not overwrite real cached data with fake/demo data during production failure paths.

### Feature engineering
All features are backward-looking only (no future data leakage).
Temperature in °F, wind in mph, CDD/HDD baseline = 65°F.
43 total features: 17 raw weather + 25+ derived.

### Scenario engine
ALWAYS copy the feature matrix. NEVER mutate input. Recompute ALL derived features after override.

### Callbacks
All callbacks live in `components/callbacks.py`.
Tab layouts are stateless functions.
Typical flow: region-selector → data stores → tab-specific chart callbacks.

### UI changes
When making shell/UI changes:
- preserve IDs unless intentionally refactoring callbacks
- prefer relabeling/restructuring over unnecessary behavior changes
- maintain accessibility and focus states
- keep semantic color use consistent: default product identity should not rely on alert colors

---

## Spec References
- `EXECUTION_BRIEF.md` — prioritization, redesign direction, execution order
- `PRD.md` — requirements, personas, descoping rationale, ADRs
- `TECHNICAL_SPEC.md` — data sources, features, models, caching
- `docs/BACKTEST_RESULTS.md` — real EIA holdout accuracy
- `docs/SCHEDULED_JOBS.md` — Cloud Run Jobs deploy + bootstrap procedure
- `tests/TEST_PYRAMID.md` — coverage targets and testing strategy
- `specs/archive/` — historical reference only; do not treat as current truth unless cross-verified

---

## Sprint 5 / trust-and-readiness conventions

### Backlog Items Implemented
- **D2**: Forecast Model Input Audit Trail — `data/audit.py`
- **I1**: Pipeline Transformation Logging — `observability.py`
- **A4+E3**: Per-Widget Data Freshness + Confidence Badges — `components/error_handling.py`
- **C9**: Meeting-Ready Mode — toggle button strips chrome for projection/PDF
- **H3**: Test Pyramid Definition — `tests/TEST_PYRAMID.md`

### Environment Config (J1)
- ALL env-specific values via `_ENV_DEFAULTS` matrix in config.py
- `ENVIRONMENT` selects tier: `development` / `staging` / `production`
- Explicit env vars ALWAYS override matrix defaults
- Never hardcode tier-specific values outside config.py

### MAPE Governance (H2)
- Use `mape_grade(mape, horizon)` — never raw threshold comparison
- Horizons: `24h`, `48h`, `72h`, `7d` — longer horizons are more tolerant
- Rollback grade means a model should be disabled and logged as an alert

### Data freshness / fallback behavior (G2)
- `data-freshness-store` tracks per-source status such as `fresh | stale | warming | demo | error`
- `warming` is emitted in production when `REQUIRE_REDIS=True` and Redis has
  no entry for the requested key yet (e.g. before the first scoring-job run,
  or after a Redis flush). The UI renders a "Data warming up" message instead
  of spinning callbacks.
- `fallback-banner` renders warnings only when degraded
- Production fallback paths should prefer stale real data over fake data when possible
- Demo data is for offline/demo contexts and must not silently overwrite real cached data during production incidents

### Bookmarks (C2)
- `dcc.Location(id="url")` manages query params
- Supported params: `region`, `persona`, `tab`
- Always validate param values against known sets before applying

### Feature Flags
- All flags in `config.FEATURE_FLAGS`
- Use `config.feature_enabled(flag)` — unknown flags default to True

---

## Final instruction

Treat GridPulse as a technically serious product that is being upgraded into a clearer, calmer, more premium platform experience.

Do not optimize for superficial polish alone.
Optimize for:
- product coherence
- trustworthy workflows
- strong information hierarchy
- preserved technical credibility
- execution in small, safe steps
