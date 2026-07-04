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

> **Addendum (2026-05-01) — implementation IA.** The conceptual product areas above were never a 1:1 mapping to the implementation. The shipping app went through three IA passes:
>
> - **R3 / R4** ([PR #38](https://github.com/kristenmartino/gridpulse/pull/38)) reduced the original nine implementation tabs to **four visible** (Overview / Forecast / Risk / Models) and absorbed the rest (Historical, Backtest, Generation, Weather, Simulator) into those four, while keeping their layouts DOM-resident under `tab_class_name="d-none"` for callback safety.
> - **V2.1** ([PR #63](https://github.com/kristenmartino/gridpulse/pull/63)) deleted the five hidden modules and their dedicated callbacks; the surface stayed at four visible tabs.
> - **V1.β / V1.γ** ([PR #64](https://github.com/kristenmartino/gridpulse/pull/64)) added **US Grid** as a fifth visible tab — a small-multiples + map view across all balancing authorities.
>
> Current visible tabs: **Overview · US Grid · Forecast · Risk · Models** (5 total, 0 hidden). Generation, Weather, and Scenarios live as inline content / panels inside Overview and Forecast; Briefings ships as the **Briefing Mode** header toggle rather than a tab.

---

## 6. Requirements

### 6.1 Data Ingestion

| ID | Requirement | Priority |
|---|---|---|
| R1.1 | Hourly demand from EIA API v2 for 51 balancing authorities — original 8 (ERCOT, CAISO, PJM, MISO, NYISO, FPL, SPP, ISONE), V1.α expansion of 8 utility/federal BAs (SOCO, TVA, DUK, CPLE, BPAT, AZPS, NEVP, PSCO) ([PR #61](https://github.com/kristenmartino/gridpulse/pull/61)), and V3.ζ expansion of the remaining 35 EIA-930 BAs in the lower 48 for ~100% contiguous-US coverage | Must Have |
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
| R4.9 | Region selector covering all 51 supported balancing authorities (~100% lower-48 coverage) | Must Have |
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
| ADR-004 | Sharpened inverse-MAPE ensemble weighting — weight ∝ (1/MAPE)³ | Follows the best model, blends only when peers are close; refined from plain 1/MAPE after the recursive re-measure (#181). Value is error-decorrelation, not tail-robustness |
| ADR-005 | XGBoost as primary model | Strong empirical performance on the current feature-engineered demand problem |
| ADR-006 | Multi-view shell instead of one flat dashboard | Supports different operational questions without forking the product into separate tools |
| ADR-007 | Scenario engine must avoid input mutation | Safer callback behavior and more predictable state handling |
| ADR-008 | Climatology fallback for days 17-30 of the forecast horizon, labeled visibly | Open-Meteo's free `/forecast` endpoint covers 16 days; atmospheric chaos limits NWP skill past ~14 days regardless; the operational user value is in days 1-7, not 17-30 |

### ADR-004 detail — Ensemble weighting exponent (2026-07-04)

**Context.** The served forecast is a weighted blend of XGBoost, Prophet, and
SARIMAX. The original weighting was plain inverse-MAPE (`weight_i = (1/MAPE_i) /
Σ 1/MAPE_j`). After the holdout was re-measured on the honest recursive protocol
(#209), the blend visibly trailed the best single model — median 4.82% vs
best-base 4.12% — because inverse-MAPE still hands 15–30% weight to models
running 3–5× worse than the leader (#181).

**Decision.** Raise the weighting to a power: `weight_i ∝ (1/MAPE_i)^k` with
`k = ENSEMBLE_WEIGHT_EXPONENT = 3`, so the blend follows the best model and
blends meaningfully only when peers are genuinely close.

**Evidence.** Regenerated the per-model recursive holdout series for all 51 BAs
and swept `k`. `k=3` beats `k=1` on **47/51** BAs (median 4.19% → 3.90%), sits
within ~0.15pp of the convex-optimal oracle, and holds up under a held-out
even/odd-hour split (not overfit to one window). Full table:
`docs/BACKTEST_RESULTS.md` → "Ensemble weighting".

**Corrected rationale.** The prior justification — "the ensemble's value is tail
variance-reduction" — does **not** hold on recursive data: a single model
(XGBoost) has the best tail. The blend's real value is **error-decorrelation**
on the minority of BAs where two models are comparably good (CAISO 4.55% →
3.51%, AZPS 13.4% → 8.2%). Sharpening keeps those wins while dropping the median
cost.

**Alternatives considered.** (1) Keep `k=1` — rejected, dominated on 49/51 BAs.
(2) Winner-take-all (best-model-per-BA) — rejected: forgoes the decorrelation
wins and generalizes worse across weeks (a model that wins one week can lose the
next). (3) Serve XGBoost-only — simplest and competitive on the tail, but gives
up the per-BA decorrelation gains; retained as a documented fallback if the
ensemble ever regresses. `k=3` is the middle path that dominates plain
inverse-MAPE at one-line, reversible cost.

### ADR-008 detail — Forecast horizon beyond Open-Meteo coverage (2026-05-20)

**Context.** `FORECAST_HORIZON_HOURS = 720` (30 days). Open-Meteo's free `/forecast` endpoint provides 16 days (384 hours) of GFS-based weather forecast. That leaves 14 days (336 hours) of the demand forecast horizon without real weather inputs.

**Decision.** Use per-(hour-of-day, day-of-week) climatological group means computed from the 92-day historical weather window to fill days 17-30. Render a visible day-16 boundary marker on the Forecast tab so users see the regime transition.

**Alternatives considered.**

1. **Shorten the horizon to 16 days.** Honest about the data, but loses the 30-day-view feature that's already shipped and used for monthly capacity-planning context. Net regression for some user workflows.
2. **ECMWF subseasonal-to-seasonal (S2S) forecasts.** Open-Meteo paid tier or direct Copernicus access provides 46-day ensembles. Modest skill improvement over climatology (~10-20% MAE reduction in well-sampled regimes), but real recurring cost, more API complexity, and skill varies materially by region/season. Pre-revenue, not worth the ops burden.
3. **Light conditional climatology (anomaly persistence).** Compute recent (last 30 days) weather anomaly vs long-term seasonal baseline; project that anomaly forward with exponential decay. ~10-15% MAE reduction at days 17-30, ~half day of work. Deferred — see below.
4. **Heavy conditional climatology (teleconnection-based).** Filter historical samples by current ENSO/NAO/MJO state, take the mean. Requires 30+ years of weather history per region (we have 92 days). Multi-week research project with unclear regional payoff. Deferred indefinitely.

**Rationale for raw climatology over Light conditional climatology.**

Operational use cases concentrate at 24-168h:

- Grid Ops: day-ahead unit commitment (24-72h)
- Renewables Trader: intraday + week-ahead (24-168h)
- Energy Trader: forward curves (24-168h)
- Data Scientist: backtest / drift analysis, not point accuracy at day 25

Days 17-30 exist primarily for visual completeness on the Forecast tab and as the longer horizon over which the scenario simulator can run hypothetical perturbations. A ~10-15% MAE improvement on a portion of the horizon that doesn't drive operational decisions is real but small. Engineering complexity (regional skill validation, anomaly-decay tuning) is not.

**Honesty over accuracy.** The UI labels the boundary explicitly:

- Dotted vertical divider at hour 384 (day 16)
- "← Open-Meteo forecast" annotation on the left segment
- "climatology baseline →" annotation on the right segment
- Subtitle: "Days 1-16: real Open-Meteo forecast · Days 17-30: (hour-of-day, day-of-week) climatology baseline"
- Faint background shade past the divider so the climatology segment reads as visually distinct

Users seeing demand changes past day 16 can correctly interpret them as seasonal/diurnal patterns rather than forward-looking forecast signal.

**Revisit triggers.** Light conditional climatology becomes worth doing if:

- Production usage analytics show meaningful engagement with days 17-30 (e.g., persona views beyond Data Scientist regularly hitting the 30-day toggle), OR
- A specific user-research signal indicates the climatology baseline is being mistaken for a real forecast despite the UI labels, OR
- A live drift MAPE measurement at days 17-30 shows the climatology baseline is meaningfully worse than what a simple anomaly-persistence model would produce on the same regions.

Heavy conditional climatology (S2S or teleconnection-conditioned) remains deferred until/unless GridPulse has paying customers with specific extended-range accuracy requirements.

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
