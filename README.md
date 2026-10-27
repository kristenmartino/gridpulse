# GridPulse — Electricity Demand Forecasting

**[gridpulse.kristenmartino.ai](https://gridpulse.kristenmartino.ai)** · [Methodology and context](https://gridpulse.kristenmartino.ai/about)

Product direction and delivery by Kristen Martino.

GridPulse compares hourly electricity demand forecasts with reported EIA demand and operator forecasts across 51 US balancing authorities. It also shows weather data, model error, data-quality notes, and weather scenarios.

**Product description:** _Electricity demand forecast review_

The primary user is a forecast review analyst deciding whether a published forecast needs closer examination. Start with the next peak, identify the forecast shown, then check its matching error evidence and data warnings before investigating in Forecast or Models.

Kristen set the product scope and priorities, defined the review workflow, and directed delivery of the live dashboard and read-only API, including how forecast sources, accuracy, and data quality are presented.

---

## What GridPulse Does

Use GridPulse to inspect reported demand and forecasts, compare model error, review weather alerts, and explore weather assumptions.

### Core questions the dashboard answers

| Tab | Question | What it shows |
|---|---|---|
| **Overview** | Does the published forecast need closer review? | Published next-24h peak, source, same-series 24h-horizon diagnostic when available, separate training reference, expandable history |
| **US Grid** | How does demand compare across regions? | 51-BA Cards / Map / Polygon views with utilization + capacity context |
| **Forecast** | Which forecast am I reviewing? | Published default or a named model comparison; source and same-horizon error above the chart; trained-model bands use backtest error ranges when available and heuristic envelopes otherwise |
| **Risk** | Which alerts need review? | Weather alerts, demand anomalies, demand/nameplate comparison |
| **Models** | How have the models performed? | Per-model MAPE / RMSE / MAE / R² (real holdout metrics from training), residuals, built-in XGBoost feature importance |

The default view is Forecast Review Analyst. Three secondary role-based views (Renewables Analyst, Trader, Data Scientist) tune the narrative layer: the insight card's framing, and which insights surface and how many. The data, KPI tiles, and forecasts are identical across personas. (Per-persona KPI cards and alert thresholds are declared in `personas/config.py` but not yet wired into the shell — see [#188](https://github.com/kristenmartino/gridpulse/issues/188).)

Overview keeps the published source, next peak, and matching observed-error label above the demand chart. Its 24-hour and seven-day history controls change the reported-demand history while retaining the next 24 hours of published forecasts. Aggregate scoring coverage and the separate training reference sit alongside the chart; the hatched portion represents unscored targets in the recorded window, not the timing or cause of particular gaps. Unknown coverage remains unknown. **Inspect evidence** separates the scoring summary from the hourly forecast targets, with source, timing, selection-policy, and sampling qualifications available in expandable details.

---

## Forecast identity and accuracy

The scoring job writes the published default to `predicted_demand_mw`. `primary_model` identifies its model unless `served_series=seasonal-naive` marks a repeated-day baseline. Overview and Forecast's default selection read that published column; explicit model columns remain available for comparison. A baseline never inherits a trained model's name, holdout score, or prediction interval.

The visible evidence panel separates **observed error** from the **training reference**. Observed error is seven-day MAPE for the displayed series, shown only with at least six matching observations. New hourly observations are sampled at least H and less than H+1 hours before their target (H = 24/48/72), independently of the forecast’s input origin. This is sampling lead, not original publication lead. Until six new observations are scored, the panel identifies the shown metric as the legacy forecast-start-offset score; the two histories are never pooled. The display-count rule does not establish sample sufficiency. Coverage and scored target dates are shown when recorded; legacy totals are not substituted for seven-day counts. It describes past forecasts at that horizon, not every point in the chart or expected future accuracy. The 7-day and 30-day views have no matching live lead score and say so. The training reference is that model's last 168-hour recursive holdout across multiple leads; missing or unidentified series never inherit another model's metrics.

The operator benchmark scores the named model series, not necessarily the published default. Each row identifies its model. Its aggregate median summarizes a stated group of authorities; authorities reported separately are excluded from that median. Forecast leads are not fully matched with the operator submissions, so the comparison does not establish operational value.

## Models

- **XGBoost** — 49 engineered features, TimeSeriesSplit CV, built-in feature importance. Best base model for 42 of 51 BAs in the dated 2026-08-07 evaluation; 1.89% MAPE on ERCOT and 3.69% median across all BAs (168-hour **recursive** holdout). These are historical reference results, not a current fleet measurement.
- **Prophet** — weather regressors with logistic growth + floor=0 (structurally prevents negative forecasts). Daily / weekly / yearly seasonality.
- **SARIMAX** — auto-order selection via `pmdarima` on cold runs; cached `(p,d,q)(P,D,Q,m)` order on warm runs (skips the auto-select stepwise search, ~10× speedup on daily refresh).
- **Ensemble** — sharpened inverse-MAPE weighted combination ((1/MAPE)³ per ADR-004). Lower measured error receives more weight; missing input scores trigger equal weights. In the 2026-08-07 evaluation it beat XGBoost alone on 21 of 51 BAs, while its median MAPE was worse: 4.35% versus 3.69%. Results depend on region and evaluation window.

**Holdout provenance.** Each model's MAPE / RMSE / MAE / R² shown in the Models tab is the training-job's last 168-hour holdout, scored **recursively** (multi-step — the model's own predictions feed forward, so errors compound as in a real forecast), persisted to each pickle's `meta.json` in GCS and read at request time. The ensemble uses held-out predictions too; its weighting and scoring split is described in the backtest document. Holdout error does not by itself establish live forecast performance.

See [docs/BACKTEST_RESULTS.md](docs/BACKTEST_RESULTS.md) for the dated evaluation, including past leakage corrections and cases where the ensemble underperforms. Heuristic bands are indicative ranges, not calibrated probabilities; backtest-derived bands also need coverage checks for the forecast horizon. Demand/nameplate comparisons do not establish available reserves or grid reliability.

---

## Architecture

```text
┌────────────────────────────────────────────────────────────────────┐
│  Browser — gridpulse.kristenmartino.ai                             │
│  ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────────┐  │
│  │ Persona    │ │ Region   │ │ KPI Bar  │ │ Briefings /        │  │
│  │ Selector   │ │ Selector │ │          │ │ Signals            │  │
│  └────────────┘ └──────────┘ └──────────┘ └────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Overview | US Grid | Forecast | Risk | Models                │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────┬──────────────────────────────────────────┘
                          │ Callback + store-driven rendering
                          ▼
┌────────────────────────────────────────────────────────────────────┐
│  Redis (Memorystore) ← populated by Cloud Run Jobs                │
│    · scoring job  (hourly 0 * * * *)                              │
│    · training job (daily  0 4 * * *)                              │
│  Web service is Redis-only: never fetches APIs or trains          │
│  models in the request path. Cold Redis → "warming" UI state.     │
└───────────┬─────────────────────────────┬──────────────────────────┘
      ┌─────▼─────┐                 ┌─────▼──────┐
      │ Data Layer│                 │ ML Models  │
      │ EIA v2    │                 │ XGBoost    │
      │ Open-Meteo│                 │ Prophet    │
      │ NOAA/NWS  │                 │ SARIMAX    │
      │ News RSS  │                 │ Ensemble   │
      │ SQLite    │                 │ SHAP       │
      └───────────┘                 └────────────┘
```

**Data flow:** the scoring job runs hourly, fetching EIA + weather, loading the latest pickled models from GCS, and writing forecasts / alerts / diagnostics to Redis. The training job runs daily at 04:00 UTC, retraining each region's models on the last 90 days and persisting them back to GCS. The web service does cache-backed reads only — no API calls or model training in the request path.

---

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
# → http://localhost:8080
```

For live EIA data, set `EIA_API_KEY` (free at [eia.gov/opendata](https://www.eia.gov/opendata/)).

---

## Project Structure

```text
.
├── app.py                          # Entry point
├── config.py                       # Constants, labels, feature flags, environment config
├── components/
│   ├── layout.py                   # Main layout and top-level shell
│   ├── callbacks.py                # Shared data flows + interaction callbacks
│   ├── cards.py                    # KPI, alert, welcome, briefing, supporting cards
│   ├── tab_overview.py             # Overview screen
│   ├── tab_us_grid.py              # US Grid screen (Cards / Map / Polygons)
│   ├── tab_demand_outlook.py       # Forecast screen
│   ├── tab_alerts.py               # Risk / extreme events screen
│   └── tab_models.py               # Models screen (per-model metrics + diagnostics)
├── data/
│   ├── eia_client.py               # EIA API v2
│   ├── weather_client.py           # Open-Meteo
│   ├── noaa_client.py              # NOAA/NWS alerts
│   ├── redis_client.py             # Memorystore client (read-only at request time)
│   ├── cache.py                    # SQLite cache with TTL
│   ├── preprocessing.py            # Merge, align, interpolate, LTTB downsample
│   ├── feature_engineering.py      # 49 features: 17 raw weather + 32 derived (CDD/HDD, lags, rolling stats)
│   └── audit.py                    # Forecast audit trail
├── models/
│   ├── model_service.py            # Forecast service abstraction (meta.json-first)
│   ├── persistence.py              # GCS-backed model store (atomic latest.json pointer)
│   ├── prophet_model.py            # Prophet model
│   ├── arima_model.py              # SARIMAX model (cached-order fast path)
│   ├── xgboost_model.py            # XGBoost model + SHAP
│   ├── ensemble.py                 # Inverse-MAPE weighting
│   ├── evaluation.py               # MAPE, RMSE, MAE, R²
│   └── training.py                 # Training orchestrator
├── jobs/                           # Cloud Run Jobs (hourly scoring, daily training)
│   ├── scoring_job.py              # Reads GCS pickles → writes Redis
│   └── training_job.py             # Trains all 51 BAs → persists to GCS
├── personas/                       # 4 persona configs + welcome logic
├── tests/                          # ~3,000 tests across unit / integration / smoke
├── Dockerfile                      # Multi-stage, non-root, healthcheck
└── .github/workflows/              # CI + deploy-prod workflows
```

---

## Deployment

**Production** is deployed automatically on push to `main` via GitHub Actions.

```text
Cloud Run Service (gridpulse)  →  gridpulse.kristenmartino.ai
  ├── Memorystore (Redis)       →  forecasts, backtests, alerts, diagnostics
  ├── VPC Connector             →  links service + jobs to Redis
  └── GCS (models/)             →  pickled XGBoost / Prophet / SARIMAX + latest.json

Cloud Scheduler  →  Cloud Run Jobs
  ├── gridpulse-scoring-hourly  →  gridpulse-scoring-job   (0 * * * *)
  └── gridpulse-training-daily  →  gridpulse-training-job  (0 4 * * *)
```

The web service is stateless and **Redis-only**: it never calls EIA, Open-Meteo, or
trains models in the request path. All expensive work runs in the two scheduled
Cloud Run Jobs. When Redis is cold, the UI renders a "warming" state.

Setup + bootstrap procedure for Cloud Scheduler, IAM, and the first run is in
[docs/SCHEDULED_JOBS.md](docs/SCHEDULED_JOBS.md).

### Manual deployment

```bash
docker build -t gridpulse .
docker run -p 8080:8080 -e EIA_API_KEY=your_key gridpulse
```

---

## Public API (v1)

Read-only JSON access to the same Redis-backed data the dashboard renders —
no key required. Base URL: `https://gridpulse.kristenmartino.ai/api/v1`

```bash
# Endpoint index
curl https://gridpulse.kristenmartino.ai/api/v1

# The 51 balancing authorities + metadata (capacity + capacity_source, import-dominated, quality gate)
curl https://gridpulse.kristenmartino.ai/api/v1/regions

# Next-24h ensemble forecast for a region (per-model series + holdout metrics included)
curl "https://gridpulse.kristenmartino.ai/api/v1/forecast/FPL?horizon=24"

# National roll-up: total demand, simultaneous 24h peak, utilization, top-stress BA
curl https://gridpulse.kristenmartino.ai/api/v1/grid/summary

# Model drift: live 1h nowcast + horizon-matched (24/48/72h) accuracy grades
curl https://gridpulse.kristenmartino.ai/api/v1/drift/ERCOT
```

Semantics worth knowing (the API keeps the dashboard's honesty rules):

- Every payload carries provenance — `scored_at`, model identity, ensemble
  weights. Data refreshes hourly from the scoring pipeline.
- A cold cache returns **503 `{"status": "warming"}`**, never fabricated data.
- Unknown regions return **404** with the valid-region list.
- Forecast horizon is capped at **168h** — the week driven by numerical
  weather forecasts; the dashboard's 30-day tail leans on climatology
  (ADR-008) and is deliberately not exported.
- Prediction intervals are omitted until per-model calibration lands (#196);
  `capacity_mw` is EIA-860M **nameplate** for most BAs or a peak×1.15 **estimate**
  for 7 peak-derived BAs (`capacity_source` disambiguates; #254), never accredited
  capacity (#243).

---

## Testing

```bash
pytest tests/ -v
pytest tests/unit/ -v
pytest tests/smoke/ -v
```

The suite includes unit, integration, and smoke checks. Run it at the revision
you are reviewing for current counts and results.
[docs/CANONICAL_FACTS.md](docs/CANONICAL_FACTS.md) records dated measurements;
those counts are historical, not evidence that the present revision was tested.

See [tests/TEST_PYRAMID.md](tests/TEST_PYRAMID.md) for coverage targets and test strategy.

---

## Published forecast evidence

The forecast API now follows the same published column as the dashboard;
clients needing the ensemble comparison can read `by_model.ensemble` explicitly.
The timing object distinguishes generation, target range, and input anchor.
Operator issue time is unknown unless supplied; capture time is not substituted.

The scoring job prospectively archives published issues for the initial
FPL/PJM/LGEE pilot. `/api/v1/publications/{region}` lists cached receipts;
`/api/v1/publications/{region}/{id}` returns their immutable values and provenance.
The read cache has a rolling 30-day index, bounded to 720 issues per region;
this does not promise complete capture or a 30-day physical GCS deletion policy.
Legacy chart replay is separate. The historical operating-day review is gated
on verified publication history and has not been introduced by this foundation.

The drift API also exposes separate `hourly_sampled` summaries. Coverage categories distinguish startup, missing observations, unavailable series, pending actuals, and exclusions. See the [scoring-coverage investigation](docs/audits/SCORING_COVERAGE_2026-10-25.md).

See the [evidence audit](docs/audits/FORECAST_EVIDENCE_2026-10-25.md) and
[implementation scope](docs/internal/PRACTITIONER_WORKFLOW_SCOPE.md).

## Project notes

GridPulse is an independent project using public data. Practitioner validation of the workflow and its operational value is the next step; analyst adoption and business impact have not been validated.

Developed with AI assistance.

## Documentation

| Doc | Purpose |
|---|---|
| [docs/BACKTEST_RESULTS.md](docs/BACKTEST_RESULTS.md) | Model accuracy on real EIA holdout data |
| [TECHNICAL_SPEC.md](TECHNICAL_SPEC.md) | Data sources, features, models, caching, pipeline details |
| [PRD.md](PRD.md) | Product requirements, personas, ADRs, scope decisions |
| [docs/SCHEDULED_JOBS.md](docs/SCHEDULED_JOBS.md) | Cloud Run Jobs deploy + first-run bootstrap |
| [CLAUDE.md](CLAUDE.md) | AI coding assistant context, architecture, conventions, execution guardrails |
| [docs/internal/EXECUTION_BRIEF.md](docs/internal/EXECUTION_BRIEF.md) | _(internal)_ Agent-ready prioritization layer for redesign and repositioning work |

---

## Scope and next work

The repository contains a forecasting dashboard, scheduled data/model jobs, and
a read-only API. Current engineering work is tracked in
[GitHub issues](https://github.com/kristenmartino/gridpulse/issues).
