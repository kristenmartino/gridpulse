# CLAUDE.md — Project Conventions for Energy Forecast Dashboard

## Architecture

This is a **Dash/Plotly** dashboard for weather-aware energy demand forecasting.
It uses 3 ML models (Prophet, SARIMAX, XGBoost) combined via a weighted ensemble
to forecast hourly electricity demand for 8 US balancing authorities.

**Active tabs (8):** Historical Demand, Demand Forecast, Backtest, Generation & Net Load, Weather Correlation, Model Diagnostics, Extreme Events, Scenario Simulator.

### Key Decisions (ADRs)
- **ADR-001**: Dash + Plotly (not Streamlit) — callback architecture scales to 21 groups
- **ADR-002**: SQLite cache on Cloud Run ephemeral disk — survives across requests, acceptable to lose on recycle
- **ADR-003**: Open-Meteo (not NOAA NWS) for weather — no API key, 17 variables in one call, &past_days parameter
- **ADR-004**: 1/MAPE weighted ensemble — simpler than stacking, self-correcting, bounded by individual models
- **ADR-005**: Scenario engine copies features, never mutates — pure function, safe for concurrent callbacks
- **ADR-006**: 8-tab full architecture — all planned tabs active (history → forecast → validate → generation → weather → models → alerts → simulator)

### Module Map
```
app.py                    → Dash app entry point, registers layout + callbacks
config.py                 → ALL constants: regions, API URLs, thresholds, pricing tiers
observability.py          → Pipeline transformation logger
components/
  layout.py               → Main layout: header, persona switcher, region selector, 8-tab container
  callbacks.py            → ALL Dash callbacks (21 groups: data loading, tab rendering, interactions)
  cards.py                → Reusable: KPI cards, welcome cards, alert cards, news feed
  error_handling.py       → Confidence badges, loading spinners, empty/error states
  accessibility.py        → Colorblind palette, ARIA helpers
  insights.py             → Persona-aware insight engine: rule-based analysis for all 8 tabs
  tab_forecast.py         → Historical Demand tab (past actuals + EIA overlay)
  tab_demand_outlook.py   → Demand Forecast tab (forward predictions + confidence bands)
  tab_backtest.py         → Backtest tab (model evaluation on holdout)
  tab_generation.py       → Generation & Net Load tab (fuel mix, renewable share)
  tab_weather.py          → Weather-Energy Correlation tab (scatter plots, heatmaps, feature importance)
  tab_models.py           → Model Comparison & Diagnostics tab (metrics, residuals, SHAP)
  tab_alerts.py           → Extreme Events tab (NOAA alerts, anomaly detection, stress indicator)
  tab_simulator.py        → Scenario Simulator tab (weather overrides, presets, impact dashboard)
data/
  cache.py                → SQLite cache with TTL + stale fallback
  eia_client.py           → EIA API v2: demand, generation, interchange
  weather_client.py       → Open-Meteo: 17 weather vars, historical + forecast
  noaa_client.py          → NOAA/NWS: severe weather alerts
  news_client.py          → NewsAPI: energy news feed
  preprocessing.py        → Merge, align UTC, interpolate gaps <6h, flag gaps ≥6h
  feature_engineering.py  → 43 derived features: CDD/HDD, wind power, solar CF, lags, rolling
  demo_data.py            → Synthetic data generator for offline/demo mode
  audit.py                → Forecast audit trail (model version, data hash, feature hash)
models/
  prophet_model.py        → Prophet with 7 weather regressors
  arima_model.py          → SARIMAX with pmdarima auto-order
  xgboost_model.py        → XGBoost with TimeSeriesSplit CV + SHAP
  ensemble.py             → 1/MAPE weighted combination
  evaluation.py           → MAPE, RMSE, MAE, R², residuals, error-by-hour
  model_service.py        → Forecast service layer: get_forecasts() with trained→simulated fallback
  training.py             → Orchestrator: train all → validate → compute weights → serialize
  pricing.py              → Merit-order: base/moderate/exponential/emergency tiers
simulation/
  scenario_engine.py      → Copy→Override→Recompute→Reforecast→Delta
  presets.py              → 6 historical extremes (Uri, Heat Dome, Irma, etc.)
personas/
  config.py               → 4 personas: Grid Ops, Renewables, Trader, Data Scientist
  welcome.py              → Data-driven welcome messages
```

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

### Common Patterns

**API client pattern:**
1. Check cache → 2. Fetch from API → 3. Parse → 4. Cache → 5. Return
On API failure: serve stale cache data with warning log.

**Feature engineering:**
All features are backward-looking only (no future data leakage).
Temperature in °F, wind in mph, CDD/HDD baseline = 65°F.
43 total features: 17 raw weather + 25+ derived.

**Scenario engine:**
ALWAYS copy the feature matrix. NEVER mutate input. Recompute ALL derived features after override.

**Callbacks:**
All callbacks in components/callbacks.py. Tab layouts are stateless functions.
Data flows: region-selector → data stores → tab-specific chart callbacks.

## Spec References
- PRD: PRD.md (requirements, personas, descoping rationale, ADRs)
- Technical spec: TECHNICAL_SPEC.md (data sources, features, models, caching)
- Backtest results: docs/BACKTEST_RESULTS.md (real EIA holdout accuracy)
- Test strategy: tests/TEST_PYRAMID.md (coverage targets, pyramid)
- Planning artifacts: specs/archive/ (original expanded spec, backlog, system definition — historical reference only)

## Sprint 5 Conventions

Sprint 5 focus: **Trust, Audit & Production Readiness**

### Backlog Items Implemented
- **D2**: Forecast Model Input Audit Trail — `data/audit.py` (AuditRecord, AuditTrail)
- **I1**: Pipeline Transformation Logging — `observability.py` (PipelineLogger)
- **A4+E3**: Per-Widget Data Freshness + Confidence Badges — `error_handling.py` (confidence_badge, widget_confidence_bar)
- **C9**: Meeting-Ready Mode — toggle button strips chrome for projection/PDF
- **H3**: Test Pyramid Definition — `tests/TEST_PYRAMID.md` (coverage targets, scope, flows)

### Key Files Changed
- `data/audit.py` — NEW: audit record dataclass + trail singleton
- `observability.py` — ADDED: PipelineLogger with step tracking
- `components/error_handling.py` — ADDED: confidence levels, badges, widget bar
- `components/layout.py` — ADDED: meeting-mode-btn, widget-confidence-bar, audit-store, pipeline-log-store
- `components/callbacks.py` — UPDATED: load_data outputs audit+pipeline, new callbacks (widget confidence, meeting mode)
- `tests/TEST_PYRAMID.md` — NEW: coverage targets, test scope
- `tests/unit/test_sprint5.py` — NEW: 40+ tests for all Sprint 5 items

### Environment Config (J1)
- ALL env-specific values via `_ENV_DEFAULTS` matrix in config.py
- ENVIRONMENT var selects tier: `development` / `staging` / `production`
- Explicit env vars ALWAYS override matrix defaults
- Never hardcode tier-specific values outside config.py

### MAPE Governance (H2)
- Use `mape_grade(mape, horizon)` — never raw threshold comparison
- Horizons: `24h`, `48h`, `72h`, `7d` — longer = more tolerant
- Rollback grade means model should be disabled; log an alert

### API Fallback (G2)
- `data-freshness-store` tracks per-source status: `fresh | stale | demo | error`
- `fallback-banner` renders warning only when degraded
- Demo data generators are the ultimate fallback — must never raise

### Bookmarks (C2)
- dcc.Location id="url" manages query params
- Supported params: `region`, `persona`, `tab`
- Always validate param values against known sets before applying

### Feature Flags
- All flags in `config.FEATURE_FLAGS` dict
- Use `config.feature_enabled(flag)` — unknown flags default to True
