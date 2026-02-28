# GridPulse — Product Requirements Document

## Problem

Energy grid operators make generation scheduling decisions worth millions of dollars per day based on demand forecasts. When forecasts miss, the consequences are immediate: blackouts during under-prediction, wasted capacity during over-prediction, and FERC/NERC penalties for inadequate reserve margins.

The operational reality at utilities like NextEra Energy is that forecasting workflows are fragmented across systems. Grid operators check one tool for demand data, another for weather, a third for model outputs. Renewable analysts and energy traders need the same underlying data viewed through different lenses. No single tool serves all four roles with the context each needs.

Three gaps define the problem:

1. **No weather-aware forecasting.** Existing tools treat demand as a purely time-series problem. Temperature alone explains 40–60% of demand variance in southern regions.
2. **No model validation in the workflow.** Operators consume forecasts without visibility into which model produced them, how accurate that model has been historically, or how confidence degrades over longer horizons.
3. **No role-based views.** A grid ops manager needs peak demand and reserve margins. A data scientist needs residual analysis and SHAP values. Forcing both through the same interface means neither gets what they need.

## Product Vision

An interactive dashboard that combines real grid data (EIA) with meteorological features (Open-Meteo) to forecast electricity demand across 8 U.S. balancing authorities. Three focused views — historical demand, forward forecast, and backtest validation — serve four distinct operational personas through a single interface.

## Target Users

| Persona | Role | Primary Decision | Default View |
|---------|------|-----------------|--------------|
| **Sarah** — Grid Ops Manager | Generation scheduling at a regional BA | "Do I need to dispatch peaker units in the next 72 hours?" | Historical Demand |
| **James** — Renewables Analyst | Wind/solar portfolio management | "How will tomorrow's weather affect my generation assets?" | Demand Forecast |
| **Maria** — Energy Trader | Electricity spot/futures trading | "Where are the demand-supply imbalances I can position against?" | Demand Forecast |
| **Dev** — Data Scientist | Model improvement and feature engineering | "Which model is degrading and why?" | Backtest |

Each persona gets a different default tab, KPI card set, and contextual welcome briefing. Switching personas reconfigures the interface without reloading data.

## Requirements

### Data Ingestion

| ID | Requirement | Priority |
|----|-------------|----------|
| R1.1 | Hourly demand from EIA API v2 for 8 balancing authorities (ERCOT, CAISO, PJM, MISO, NYISO, FPL, SPP, ISONE) | Must Have |
| R1.2 | 17 weather variables from Open-Meteo (temperature, wind at 3 heights, solar radiation, humidity, precipitation, etc.) | Must Have |
| R1.3 | Severe weather alerts from NOAA/NWS mapped to regions | Must Have |
| R1.4 | SQLite cache with configurable TTL; serve stale data when APIs are down | Must Have |
| R1.5 | Demo data generators for all regions when no API keys are configured | Must Have |

### Feature Engineering

| ID | Requirement | Priority |
|----|-------------|----------|
| R2.1 | CDD/HDD from temperature (65°F baseline) | Must Have |
| R2.2 | Wind power estimate using cubic relationship at hub height (80m) | Must Have |
| R2.3 | Solar capacity factor from GHI | Must Have |
| R2.4 | Cyclical encoding (sin/cos) for hour-of-day and day-of-week | Must Have |
| R2.5 | Lag features (t-24h, t-168h) with no future data leakage | Must Have |
| R2.6 | Rolling statistics (24h/72h/168h) backward-looking only | Must Have |
| R2.7 | Total feature matrix: 17 raw weather + 25+ derived = ~43 features | Must Have |

### Forecasting Models

| ID | Requirement | Priority |
|----|-------------|----------|
| R3.1 | Prophet with weather regressors and multiplicative seasonality | Must Have |
| R3.2 | SARIMAX with auto-order via pmdarima | Must Have |
| R3.3 | XGBoost with TimeSeriesSplit CV and SHAP explanations | Must Have |
| R3.4 | Inverse-MAPE weighted ensemble | Must Have |
| R3.5 | Model service abstraction: trained → simulated fallback transparent to UI | Must Have |
| R3.6 | Forecast audit trail: model version, data vintage, feature hash per prediction | Must Have |

### Dashboard

| ID | Requirement | Priority |
|----|-------------|----------|
| R4.1 | **Historical Demand tab**: actual demand + EIA forecast overlay, time range selector, weather overlay, comparative KPIs | Must Have |
| R4.2 | **Demand Forecast tab**: forward model predictions, confidence bands (80%/95%), horizon selector, model toggles | Must Have |
| R4.3 | **Backtest tab**: model vs actuals on holdout, per-model MAPE cards, residual histogram | Must Have |
| R4.4 | Persona switcher: 4 roles with per-role default tab, KPI cards, welcome briefing | Must Have |
| R4.5 | Region selector: all 8 BAs including FPL (NextEra subsidiary) | Must Have |
| R4.6 | Per-widget data confidence badges (green/amber/red based on data freshness) | Must Have |
| R4.7 | Meeting-ready mode: one-click strips chrome for projection | Should Have |

### Infrastructure

| ID | Requirement | Priority |
|----|-------------|----------|
| R5.1 | Multi-stage Dockerfile, non-root user, healthcheck endpoint | Must Have |
| R5.2 | CI pipeline (lint, test, build) via GitHub Actions | Must Have |
| R5.3 | Structured JSON logging via structlog | Must Have |
| R5.4 | WCAG 2.1 AA colorblind-safe palette, ARIA labels | Should Have |
| R5.5 | Environment config matrix (dev/staging/production) | Must Have |

## Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Tab load time | p95 < 2 seconds |
| Forecast accuracy (XGBoost, ERCOT) | MAPE < 5% on 21-day holdout |
| Test coverage | 80%+ unit, 70%+ integration, all tabs render |
| Graceful degradation | Dashboard always renders; stale/demo data with visible indicator |
| API resilience | 5 retries with exponential backoff; stale cache fallback |

## Descoped Items

Considered and deliberately excluded. Documented so they aren't re-proposed without context.

| Item | Why Not |
|------|---------|
| Real-time streaming (sub-second) | Energy forecasting operates on hourly cadences. Unnecessary infra cost. |
| AI chatbot overlay | Narrative mode + command palette achieve 80% of the value at 5% of the cost. |
| Multi-tenant architecture | Don't build isolation until there's a second customer. |
| Mobile-native app | Responsive meeting-ready mode on tablet covers the real use case. |
| AI-generated recommended actions | Too much liability in energy operations. Surface data, let humans decide. |
| Real-time collaboration | Meeting-ready mode solves the actual problem (presenting together). |
| Full RBAC with permission matrices | Overkill until 50+ users. Row-level security by region covers the real requirement. |
| Blockchain audit trails | A SQLite audit table does the same job. |
| Natural language query interface | High cost, low marginal value over structured persona views. |

## Architecture Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| ADR-001 | Dash + Plotly (not Streamlit) | Callback architecture scales to 21 callback groups across 3 tabs + persona logic. Streamlit's re-run model can't support this. |
| ADR-002 | SQLite cache (not Redis) | Survives across Cloud Run requests, acceptable to lose on container recycle. Zero infrastructure dependency. |
| ADR-003 | Open-Meteo (not NOAA for weather) | No API key, 17 variables in one call, historical + forecast in same API. |
| ADR-004 | 1/MAPE ensemble weighting | Self-correcting: poor models get downweighted automatically. Simpler than stacking, bounded by individual models. |
| ADR-005 | XGBoost as primary model | Backtesting on real EIA data: 3.13% MAPE vs Prophet (50%) and ARIMA (40%) on 43-day training window. Feature engineering matters more than model complexity. |
| ADR-006 | 3-tab focused architecture | Reduced from 8 tabs. Three views (history, forecast, backtest) cover the core operational loop. Dormant tabs preserved for reactivation. |
