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
| ADR-009 | Class-conditional anchor conditioning: broken-feed BAs anchor on their own day-ahead forecast | EIA's newest reading averages 58.2% wrong on broken-class feeds vs the hour-matched DF's 14.5% (90% win rate, measured via the vintage instrument); churn/bulk classes measured AGAINST substitution and ship unchanged |
| ADR-010 | Serve-path acceptance gate: a retrained model must replay sanely through the real serve path before `latest.json` repoints to it | Daily retrains are a measured fit lottery (~27% of persisted LDWP vintages produce degenerate recursive forecasts) and the published holdout is provably blind to it; stale-but-sane beats fresh-but-insane, the same principle as the data-fallback policy |
| ADR-011 | NBM-composite forecast weather: NOAA's National Blend of Models overlaid on the base fetch for future hours, base-filled for the variables NBM lacks | The weather-model A/B study measured NBM temperatures 16–27% more accurate with ~zero bias, worth +0.921 sMAPE pts of demand accuracy through the real serve path (AZPS +3.70, SEC +1.88); the composite ships the exact configuration the study measured |
| ADR-012 | Multi-point weather: sample each BA's footprint at up to 12 static cells and aggregate (unweighted), instead of one representative point | The multi-point study measured +1.14 sMAPE pts (MISO +1.77, PJM +1.41, SPP +1.45; compact BAs ~0, GVL control exactly 0.000) — and measured population *weighting* as adding nothing, so production carries coordinates only and needs no runtime census data |

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

**Decision.** Use per-(hour-of-day, day-of-week) climatological group means to fill days 17-30. Render a visible day-16 boundary marker on the Forecast tab so users see the regime transition.

**Update (#281).** The group means are computed from a **recent trailing window** (`CLIMATOLOGY_WINDOW_DAYS = 28`), not the full 92-day history. A season-agnostic `(hour, dow)` mean over 92 days is seasonally biased: for a July forecast it dilutes in cooler April–June data, so the baseline understates peak-summer demand (measured on DUK: 9.4°F cooler than current, CDD halved), producing an implausible downward slope past day 16. Restricting the climatology to recent history is a lightweight recency-weighting — a partial adoption of the "Light conditional climatology" idea below — that removes most of the bias without the full anomaly-persistence machinery.

**Update (#283, 2026-07-11) — the tail is now a weather-normalized "normal weather year", level-anchored.** The days-17-30 weather inputs come from a per-BA **(day_of_year, hour) weather-normal** built from a trailing ~10-year ERA5 window (all 17 raw weather vars + Jensen-correct derived features, ±7-day circular day-of-year smoothing), with:

- a **seam anomaly-blend** at the Open-Meteo boundary — the current weather anomaly persists into the near tail as a convex blend decaying over ~5 days (the anomaly-persistence half of option 3 below, realized), so there is no regime discontinuity at hour 384;
- **level anchoring via the autoregressive demand features**, which stay on the recent window — the tail tracks *current* load levels (year-over-year growth handled with no explicit trend term);
- **per-BA graceful fallback** to the #281 recent-28d climatology wherever a BA's normal artifact isn't backfilled yet (the nightly training job builds ≤10 per run; full 51-BA coverage ~2026-07-15).

This realizes — and goes beyond — option 3 ("Light conditional climatology"), chosen via a 5-method study (issue #283) and admitted through two evidence gates: a **weather backtest** across 6 climate-diverse BAs (the normal beats the recent-28d baseline ~10:2 on season-relevant error at seasonal turns, often halving temperature MAE; a wash mid-season) and a **retrospective demand spot-check** (DUK, 2026-06-10 origin scored against realized actuals: days-17-30 MAE 3,442 → 3,146 MW, **−8.6%**, the origin straddling the early-summer ramp — the exact phase-lag case a recency-only window gets wrong). The P10–P90 empirical fan and the day-16 divider remain the honesty envelope around the whole horizon.

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
- Subtitle: "Days 1-16: real Open-Meteo forecast · Days 17-30: seasonal climatology baseline" (weather-normal where the per-BA artifact exists; recent-28d otherwise — both are climatologies, so the umbrella label stays honest across the backfill)
- Faint background shade past the divider so the climatology segment reads as visually distinct

Users seeing demand changes past day 16 can correctly interpret them as seasonal/diurnal patterns rather than forward-looking forecast signal.

**Revisit triggers** *(fired 2026-07: the #281 incident was exactly trigger 2/3-class evidence; option 3 was implemented and extended as the #283 weather-normal tail — see the update above)*. Originally, Light conditional climatology becomes worth doing if:

- Production usage analytics show meaningful engagement with days 17-30 (e.g., persona views beyond Data Scientist regularly hitting the 30-day toggle), OR
- A specific user-research signal indicates the climatology baseline is being mistaken for a real forecast despite the UI labels, OR
- A live drift MAPE measurement at days 17-30 shows the climatology baseline is meaningfully worse than what a simple anomaly-persistence model would produce on the same regions.

Heavy conditional climatology (S2S or teleconnection-conditioned) remains deferred until/unless GridPulse has paying customers with specific extended-range accuracy requirements.

---


### ADR-009 detail — Class-conditional anchor conditioning (2026-07-17)

**Context.** The forecast seeds `demand_lag_1h` + 20 sibling autoregressive
features (and SARIMAX's Kalman origin, #226) from EIA's newest published
demand reading. A week of direct measurement (#309 → #312 vintage instrument)
established that EIA's newest hour is *never* a measurement — it cycles
stub → partial → settled, mandated by Form EIA-930's 60-minute deadline and
"submit best estimates, resubmit within 3 days" rule — and that for a small
class of BAs the published partials are catastrophically wrong (LDWP 69%
mean fresh-hour revision, IID 53%, AZPS). The result on the settled-grade
meter: anchor-fed models at 26–56% live error where anchor-free Prophet ran
12.4%. The anchor was the liability, but only for that class.

**Decision.** A per-tick, data-driven policy keyed on the vintage
classifier's `revision_class`: **`broken` regions substitute the BA's own
hour-matched day-ahead forecast (`forecast_mw`, already on every frame) for
the trailing unsettled hours — on a forked frame consumed only by the
feature/forecast path.** Every other class keeps the unmodified anchor. The
fork invariant is structural: tiles, drift, alerts, and diagnostics always
read real data.

**Evidence** (`docs/ANCHOR_CONDITIONING_STUDY.md`, run against real vintage
records): broken-class anchors averaged **58.2%** wrong vs the DF's
**14.5%** (90.1% win rate, 103 fresh hours). Tier-2 end-to-end replay with
production pickles agreed on every sign: LDWP 16.4→14.3 MAPE, IID
28.2→26.7, and the counterexample validated — PSCO *worsens* 14.8→17.7
under substitution.

**Alternatives considered.**
- *Skip-to-stale (drop unsettled hours):* measured worse — a persistence
  proxy scored DF-anchoring 6.55% vs 7.72% for skipping (9/12 BAs), because
  demand ramps faster than readings settle. The #315 guard's NaN answer
  remains only the artifact backstop, not the anchor policy.
- *Condition `churn` too:* the plan's original policy, **refuted by the
  study** — the class mixes BPAT (~14% revisions) with mild churners, and at
  class level DF loses (3.20% vs 4.92%). Ships unchanged.
- *Condition `bulk`:* refuted by design and by replay — PSCO runs 118–121%
  of its own DF; substitution measurably hurts.
- *Static per-region config sets:* rejected for the live classifier —
  feeds change, and the policy self-corrects as classifications do.
- *Cron offsets / fetch reordering:* refuted earlier in the arc — the
  publication lifecycle has no universally good fetch time.

**Update trail.** Shipped dark in PR C (#324), flipped in PR D on the
study's verdict; post-flip verification is the settled-grade drift meter
(#318) — the same instrument arc that made the study possible.

### ADR-010 detail — Serve-path acceptance gate (2026-07-18)

**Context.** The night anchor conditioning went live, LDWP's forecast dove
to 1,302 MW overnight off provably clean inputs. The diagnosis study
(`docs/FORECAST_DIVE_DIAGNOSIS.md`, #326) exonerated every input and named
the mechanism: **each daily retrain is an independent draw, and ~27% of the
67 persisted LDWP XGBoost vintages produce recursive forecasts that
collapse overnight demand into a phantom regime** — condition-dependent
(the same pickle dives on one frame and not another), expressed only in
the recursive serve regime, and invisible to the published holdout, which
scores a *freshly retrained* model on a sliced historical frame and never
runs the deployed candidate through `_build_future_feature_frame` + the
recursion.

**Decision.** At persist time the training job replays the **actual
candidate pickle** through the **real serve path** from three anchors
stepped 24 h apart (`jobs/phases.py::serve_path_gate`). Offset anchors
replay into known history and are judged against settled truth (median APE
+ trough-vs-truth); the live anchor — the frame about to serve — is judged
against the trailing week (5th-percentile trough floor, mean-level band).
A live-anchor failure rejects outright; one offset dive pocket is
tolerated; a pattern of failures rejects. **A rejected candidate is still
persisted (the forensic record the diagnosis depended on) but never
repoints `latest.json`** — yesterday's accepted model keeps serving.
Stale-but-sane over fresh-but-insane, the same principle as the
stale-real-data-over-fake fallback policy. Fail-open on gate errors and
thin bootstrap history: an availability guard must not freeze rollout on
its own bug.

**Evidence.** Calibrated by replaying real vintages at their own training
moments: the gate rejects 0708/0710/0717 (live-anchor trough ratios
0.27–0.49) and 0715 (two transient dive pockets), passes 0711/0716/0718
and the PNM control. Under this rule the counterfactual timeline never
serves the 1,302 MW dive — 0716, proven sane on the Jul-18 frame, would
have kept serving.

**Alternatives considered.**
- *Harden the holdout instead:* refuted by the study — the holdout's
  blindness is model-identity (it scores a different model), not frame
  construction (rung 2 measured the frames identical). Only replaying the
  candidate itself closes it.
- *Reject on any anchor failure:* refuted by calibration — the lottery is
  a spectrum and many sane fits carry one transient pocket; rejecting all
  of them streaks rejections into stale pointers.
- *Fix the fit variance at the source (seeds, regularization):* the right
  long-term lever, but tuning work (draft PR #229's territory) — the gate
  is the safety property that makes tuning experiments safe to run at all.

**Update trail.** Shipped ON (flag `model_serve_gate`) in the PR that
closed #326; per-candidate verdicts land in each model's meta
(`extra["serve_gate"]`) and in `model_gate_passed` / `model_gate_rejected`
logs.

### ADR-011 detail — NBM-composite forecast weather (2026-07-21)

**Context.** The data-source research ranked higher-resolution NOAA
weather the top external accuracy candidate but could not quantify the
demand impact. The weather-model A/B study
(`docs/WEATHER_MODEL_AB.md`, #332) measured it through the real serve
path: lead-honest forecast vintages (Open-Meteo Previous Runs API),
current prod pickles, paired replays over 8 BAs × 11 anchors × 168 h. It
also discovered that `best_match` already resolves to the GFS+HRRR blend
for CONUS — production was already consuming HRRR, making that arm a
measured noise floor (+0.04 pts) rather than an upgrade.

**Decision.** Overlay **NBM** (`ncep_nbm_conus`, same vendor, same
endpoint, one extra request per region) onto the base fetch for **future
hours only**, keeping base values for (a) the five variables the studied
arm base-filled (`NBM_FORCE_FILL_VARS`: radiation ×3, surface pressure,
120 m wind — live NBM serves patchy radiation, but shipping it would ship
an unmeasured configuration), (b) any NBM null (its ~11.5-day horizon
inside the 16-day frame — ADR-008's boundary is untouched), and (c) all
past hours (the study conditioned future weather only). The composite is
**enrichment-only and fail-open**: any NBM failure logs
`weather_nbm_failed` and serves the base frame; the base fetch's
stale-cache → GCS fallback chain is untouched. One client-level change
means scoring and training switch together (train/serve consistency by
construction).

**Evidence.** Tier 1: NBM temperature RMSE 2.69 vs 3.22 °F at day-1,
25–27% better at days 3–7, bias ~zero vs the control's −0.4..−1.9 °F cold
bias. Tier 2 (decisive): mean **+0.921 sMAPE pts** paired; AZPS **+3.70**
and SEC **+1.88** — tail BAs the research's negative finding said no
external source could reach; worst BA MISO −0.33, inside the −0.5 veto
(and the multi-point aggregation follow-up targets exactly MISO's
single-point weakness).

**Alternatives considered.**
- *Raw NBM swap (`models=` only):* rejected — NBM lacks
  radiation/DNI/diffuse/pressure/120 m-wind; a naive swap starves
  `shortwave_radiation`, a top SHAP feature fleet-wide.
- *Per-value-only fill (trust live NBM radiation where present):*
  rejected on evidence fidelity — the measured arm base-filled those
  variables; live NBM radiation is patchy and unmeasured.
- *Switching to the `/v1/gfs` endpoint:* dissolved by measurement —
  `best_match` is already the seamless GFS+HRRR blend for CONUS.
- *NODD/Herbie self-fetch:* the licensing escape hatch if the project
  commercializes (#256 track), not an accuracy decision.

**Update trail.** Shipped DARK (flag `nbm_weather`) with the composite +
tests; flipped in a follow-up PR once the deploy verified. Post-flip
verification: `weather_nbm_composited` on 51/51 regions, the ADR-010 gate
green on the next training, and AZPS/SEC live sMAPE descending — the
study's prediction made visible on the drift meters.

### ADR-012 detail — Multi-point weather aggregation (2026-07-22)

**Context.** Every BA drew weather from ONE representative lat/lon
(`config.REGION_COORDINATES`) — for MISO that is a single point in rural
Illinois standing in for fifteen states. The load-forecasting literature
(Hong/Wang/White 2015; Sobhani et al. 2019) established that multiple
stations plus population weighting beat single-point at utility-*zonal*
scale, but it was unmeasured at BA scale; and MISO was the one BA that
got slightly *worse* under ADR-011, implicating its weather sampling
rather than its weather model.

**Decision.** Sample each BA's footprint at up to **12 static cells** and
aggregate, **unweighted**. Cells are chosen offline (census county
centroids inside the BA polygon, grid-snapped to the 0.25° ERA5 cell,
top-K by population) and committed as `assets/multipoint_coordinates.json`
— 36 BAs; the 15 compact single-metro BAs are omitted and keep their
single point. The census download and matplotlib point-in-polygon live in
`scripts/generate_multipoint_coords.py`; **the production import path
carries no new dependency and does no census work at runtime.**
Aggregation (`data/weather_aggregate.py`): circular mean for
`wind_direction_10m`, mode for the ordinal `weather_code`, renormalizing
`nanmean` for the other 15.

**Evidence** (`docs/MULTIPOINT_WEATHER_STUDY.md`): retrain-per-arm over 5
large BAs × 10 rolled windows, paired with a fixed seed so the
fit-lottery cancels — **mean +1.14 sMAPE pts**, 90% of paired windows,
no BA worse; MISO **+1.77**, SPP +1.45, PJM +1.41, ERCOT +0.79, SOCO
+0.31, and the GVL control (one county, falls back to single-point)
exactly **0.000**. The per-BA gradient tracks geographic spread, which is
the hypothesis made visible.

**Alternatives considered.**
- *Population-weighted aggregation:* the study's own arm C — **rejected
  by its own evidence** (C−B ≈ 0). The gain is spatial averaging, not
  load-weighting; at BA-aggregate scale the demand series has already
  integrated the load distribution. Dropping weights removed the entire
  runtime census dependency.
- *Loop the single-point pipeline K times:* rejected on cost — K separate
  calls ≈ 44k requests/day against a 10k/day tier. Open-Meteo accepts
  comma-separated coordinates and returns a list in one call, so
  multi-point costs 3 calls/BA/run regardless of K.
- *Aggregate first, then overlay NBM:* rejected on correctness — ADR-011's
  "any NBM null keeps base" rule is per-cell, so a point whose NBM is
  null must contribute its own base to the average; overlaying after
  aggregation would overwrite the hour from only the finite points, and
  is a configuration ADR-011 never measured. NBM composites **per point**,
  then aggregates.
- *Porting the study's aggregation verbatim:* rejected — its plain branch
  used `nansum`, counting a null point as zero. Harmless offline
  (land-only centroids, ERA5) but in production a cell over water or
  outside CONUS would drag every value toward zero: a silent coverage
  collapse of the #161 flavor. Production renormalizes.

**Safety.** The change lives in the function whose single-point
predecessor took down forecasts fleet-wide (#161), so it **fails open at
every seam**: a missing point-set, an HTTP error, or a wrong-shape
response all fall back to the untouched single-point path, whose own
`RequestException` still drives the stale → GCS → empty chain; a
multi-point archive failure retries *single-point* archive before
degrading to forecast-only, because #161 was precisely about losing deep
history. Flag off is byte-identical: `aggregate_weather` is never called.

**Update trail.** Shipped DARK (flag `multipoint_weather`); flipped in a
follow-up PR after deploy + shadow verification. Post-flip: the ADR-010
serve gate green at the next training, and MISO/PJM/SPP live sMAPE
descending toward the predicted +1.4–1.8.

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
