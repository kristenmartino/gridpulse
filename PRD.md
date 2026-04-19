# GridPulse — Product Requirements Document

_Last updated: 2026-04-09_

## 1. Product Summary

GridPulse is an **energy intelligence platform** for forecast confidence, grid visibility, and operational decision support.

It combines:
- weather-aware demand forecasting
- role-based operational views
- model validation and backtesting
- generation and net load context
- severe-condition and anomaly signals
- scenario-based analysis
- briefing-ready outputs for stakeholders

### Product framing
- **Category:** Energy Intelligence Platform
- **Tagline:** See demand sooner. Decide with confidence.
- **Primary outcome:** help energy teams interpret changing conditions faster and act with more confidence

---

## 2. Problem

Energy grid operators, analysts, and trading/planning teams make decisions worth millions of dollars per day based on incomplete, fragmented signals. When demand forecasts miss, the consequences can include:
- under-preparation and reserve pressure
- wasted capacity and inefficient dispatch
- poor market positioning
- reduced trust in analytical outputs
- slower response to severe weather or abnormal conditions

The operational reality is that relevant signals are often scattered across tools:
- demand data in one system
- weather in another
- model outputs somewhere else
- validation and confidence often missing entirely
- generation and supply-side context disconnected from forecast interpretation

Three persistent gaps define the opportunity:

1. **Forecasts lack enough context.** Teams need weather, system conditions, and generation context alongside the number.
2. **Confidence is often hidden.** Users need validation, uncertainty, and model accountability in the workflow.
3. **Different roles need different decision views.** Operators, traders, renewables analysts, and data scientists use the same underlying signals differently.

---

## 3. Product Vision

GridPulse provides a unified operating layer for energy teams by combining demand forecasting, confidence-aware analytics, grid visibility, and scenario-ready decision support in one interface.

It should feel less like a static dashboard and more like a role-aware platform for understanding:
- what changed
- what matters now
- where risk is rising
- how confident the forecast is
- what to inspect next

---

## 4. Target Users

| Persona | Role | Primary Decision | Default Product Area |
|---|---|---|---|
| **Sarah** — Grid Ops Manager | Generation scheduling at a regional BA | "Do I need to dispatch additional capacity or prepare for abnormal demand?" | Overview / Historical Demand |
| **James** — Renewables Analyst | Wind/solar portfolio management | "How will weather and net load conditions affect generation expectations?" | Demand Forecast / Grid |
| **Maria** — Energy Trader | Electricity spot/futures trading | "Where are demand-supply imbalances or forecast shifts I can position against?" | Demand Forecast / Risk |
| **Dev** — Data Scientist | Model improvement and validation | "Which model is degrading, and how reliable is the forecast right now?" | Models / Backtest |

Each persona gets a different default view, KPI emphasis, and contextual welcome/briefing layer. Switching personas reconfigures the interface without reloading core data.

---

## 5. Product Areas

The current product is implemented as a multi-tab application. Conceptually, GridPulse should map to the following product areas:

| Product Area | Purpose |
|---|---|
| **Overview** | Mission-control summary of what changed, what matters, and where confidence/risk stand |
| **Historical Demand** | Past demand, EIA overlays, weather context, operational comparison |
| **Demand Forecast** | Forward-looking predictions, confidence bands, horizon views, model comparisons |
| **Models / Backtest** | Validation, trust, accuracy, residuals, model accountability |
| **Grid** | Generation mix, net load, renewable share, supply-side context |
| **Risk / Extreme Events** | Severe conditions, anomalies, stress signals, degraded states |
| **Scenarios** | What-if analysis, presets, sensitivity testing |
| **Briefings** | Narrative summaries, meeting/presentation mode, stakeholder-ready views |

This structure is both a product framing model and a guide for future IA cleanup.

---

## 6. Requirements

### 6.1 Data Ingestion

| ID | Requirement | Priority |
|---|---|---|
| R1.1 | Hourly demand from EIA API v2 for 8 balancing authorities (ERCOT, CAISO, PJM, MISO, NYISO, FPL, SPP, ISONE) | Must Have |
| R1.2 | Hourly weather variables from Open-Meteo including temperature, wind, radiation, humidity, precipitation, and related signals | Must Have |
| R1.3 | Severe weather / alert context from NOAA/NWS mapped to regions | Must Have |
| R1.4 | Cache-backed data access with stale-data fallback for degraded conditions | Must Have |
| R1.5 | Explicit offline/demo data support when live APIs or credentials are unavailable in demo contexts | Must Have |
| R1.6 | Data freshness tracking surfaced to the UI where relevant | Must Have |

### 6.2 Feature Engineering

| ID | Requirement | Priority |
|---|---|---|
| R2.1 | CDD/HDD from temperature (65°F baseline) | Must Have |
| R2.2 | Wind power estimate using wind-speed relationships at hub height | Must Have |
| R2.3 | Solar capacity factor estimate from radiation inputs | Must Have |
| R2.4 | Cyclical encoding for hour-of-day and day-of-week | Must Have |
| R2.5 | Lag features with no future data leakage | Must Have |
| R2.6 | Backward-looking rolling statistics | Must Have |
| R2.7 | Feature matrix with ~43 total features across raw + engineered inputs | Must Have |

### 6.3 Forecasting and Model Layer

| ID | Requirement | Priority |
|---|---|---|
| R3.1 | Prophet with weather regressors and multiplicative seasonality | Must Have |
| R3.2 | SARIMAX with auto-order selection via pmdarima | Must Have |
| R3.3 | XGBoost with TimeSeriesSplit CV and SHAP explanations | Must Have |
| R3.4 | Inverse-MAPE weighted ensemble | Must Have |
| R3.5 | Model service abstraction so the UI is insulated from training/runtime details | Must Have |
| R3.6 | Forecast audit trail with model version, data vintage, and feature lineage | Must Have |
| R3.7 | Confidence / evaluation context available alongside forecast views | Must Have |

### 6.4 Product Experience

| ID | Requirement | Priority |
|---|---|---|
| R4.1 | **Overview** with key signals, KPI hierarchy, narrative context, and a rapid understanding of what changed | Must Have |
| R4.2 | **Historical Demand** view with actual demand, comparison overlays, weather context, and comparative KPIs | Must Have |
| R4.3 | **Demand Forecast** view with forward predictions, confidence bands, horizon selector, and model toggles | Must Have |
| R4.4 | **Models / Backtest** view with holdout evaluation, per-model metrics, and validation context | Must Have |
| R4.5 | **Grid** view with generation mix, renewable share, and net load context | Must Have |
| R4.6 | **Risk / Extreme Events** view with alerting, stress signals, anomalies, and degraded-condition visibility | Should Have |
| R4.7 | **Scenarios** with what-if controls, presets, and impact comparisons | Should Have |
| R4.8 | Persona / view switcher with role-specific defaults, KPI emphasis, and welcome briefing | Must Have |
| R4.9 | Region selector covering all 8 supported balancing authorities | Must Have |
| R4.10 | Per-widget data confidence or freshness signaling | Must Have |
| R4.11 | Briefing / meeting-ready mode for presentation and stakeholder review | Should Have |

### 6.5 Infrastructure and Platform Readiness

| ID | Requirement | Priority |
|---|---|---|
| R5.1 | Multi-stage Dockerfile, non-root user, and healthcheck endpoint | Must Have |
| R5.2 | CI pipeline (lint, test, build) via GitHub Actions | Must Have |
| R5.3 | Structured JSON logging via structlog | Must Have |
| R5.4 | WCAG-aware palette and ARIA support | Should Have |
| R5.5 | Environment config matrix (dev/staging/production) | Must Have |
| R5.6 | Scheduled scoring (hourly) + training (daily) via Cloud Run Jobs; Redis-only web read path | Must Have |

---

## 7. Non-Functional Requirements

| Requirement | Target |
|---|---|
| Tab / screen load time | p95 < 2 seconds when Redis is warm (populated by the hourly scoring job) |
| Forecast accuracy (XGBoost, ERCOT reference) | MAPE < 5% on 21-day holdout |
| Test coverage | 80%+ unit, 70%+ integration, all major screens render |
| Graceful degradation | App continues rendering with visible freshness/degraded-state indicators |
| API resilience | retries + stale-cache fallback where applicable |
| Accessibility | keyboard navigation, focus visibility, color-safe semantics |

---

## 8. Product Principles

These principles should guide future work:

1. **Signal over noise** — highlight what matters first.
2. **Confidence must be visible** — do not show forecasts without trust context.
3. **Role-aware, not role-fragmented** — one platform, multiple decision lenses.
4. **Human-in-the-loop** — support judgment; do not over-automate critical decisions.
5. **Operational calm** — risk should be visible without the interface feeling frantic.
6. **Technical credibility first** — product polish should not come at the expense of rigor.

---

## 9. Descoped or Deliberately Limited Items

These items are intentionally not first-class priorities right now:

| Item | Why Not |
|---|---|
| Real-time streaming / sub-second updates | Current operating cadence is hourly; not worth the infra complexity yet |
| Fully autonomous recommended actions | Too much operational liability; surface data and confidence instead |
| Multi-tenant architecture | Premature before broader adoption or multiple customers |
| Full mobile-native parity | Tablet / briefing-mode support covers the higher-value use cases for now |
| Broad NL chatbot overlay | Lower value than improving structured product views and briefings |
| Complex collaboration layer | Presentation/briefing workflows solve the immediate need more directly |
| Heavy RBAC matrices | Too much complexity for current scale |

---

## 10. Architecture Decisions (ADRs)

| # | Decision | Rationale |
|---|---|---|
| ADR-001 | Dash + Plotly (not Streamlit) | Callback architecture and component control scale better for the current multi-view interaction model |
| ADR-002 | Cache-first + fallback strategy | Operational apps need resilience, visible freshness, and predictable degraded behavior |
| ADR-003 | Open-Meteo for weather inputs | No API key, broad variable coverage, historical + forecast support in one family of endpoints |
| ADR-004 | 1/MAPE ensemble weighting | Simple, bounded, self-correcting weighting strategy |
| ADR-005 | XGBoost as primary model | Strong empirical performance on the current feature-engineered demand problem |
| ADR-006 | Multi-view shell instead of one flat dashboard | Supports different operational questions without forking the product into separate tools |
| ADR-007 | Scenario engine must avoid input mutation | Safer callback behavior and more predictable state handling |

---

## 11. Roadmap Direction

GridPulse should be able to support a modular suite architecture over time. The current product can be understood as the foundation for future product modules such as:
- GridPulse Forecast
- GridPulse Risk
- GridPulse Grid
- GridPulse Scenarios
- GridPulse Models
- GridPulse Briefings
- GridPulse API

This is a positioning and IA direction, not a requirement to split the codebase immediately.

---

## 12. Success Criteria

The product is succeeding when:
- energy stakeholders can quickly understand what changed and why it matters
- forecast users can see trust and validation context without leaving the workflow
- the UI supports different personas without fragmenting the platform
- GridPulse reads as a coherent energy intelligence product, not just a technical demo
- the system remains technically credible under inspection by data and engineering stakeholders
