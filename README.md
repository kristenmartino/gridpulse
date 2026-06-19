# GridPulse — Energy Intelligence Platform

**[gridpulse.kristenmartino.ai](https://gridpulse.kristenmartino.ai)**

Weather-aware energy intelligence. GridPulse combines real grid data (EIA), meteorological signals (Open-Meteo), multiple forecasting models, model validation, generation context, and scenario analysis in one decision-ready platform.

**Tagline:** _See demand sooner. Decide with confidence._

---

## What GridPulse Does

GridPulse helps energy teams move from fragmented monitoring to a unified operating view. The product organizes around four decision pillars:

- **Forecast with context** — unify demand, weather, and time-series patterns in a forecasting experience built for real operating conditions
- **See confidence, not just output** — understand model reliability, forecast uncertainty, and recent performance before acting on a number
- **Surface risk earlier** — track anomalies, severe conditions, and forecast instability in one operational view
- **Plan through scenarios** — test assumptions and understand how changing conditions alter expected demand and decision windows

### Core questions the dashboard answers

| Tab | Question | What it shows |
|---|---|---|
| **Overview** | What changed, and what matters now? | Mission-control summary, key KPIs, role-aware briefing |
| **US Grid** | Which balancing authorities are stressed? | 51-BA Cards / Map / Polygon views with utilization + capacity context |
| **Forecast** | What will happen? | Forward-looking predictions (Prophet, SARIMAX, XGBoost, Ensemble) with confidence bands |
| **Risk** | Where is operating risk rising? | Severe-weather signals, anomalies, stress indicators, degraded conditions |
| **Models** | How trustworthy is the forecast? | Per-model MAPE / RMSE / MAE / R² (real holdout metrics from training), residuals, SHAP feature importance |

Four role-based personas (Grid Ops, Renewables Analyst, Trader, Data Scientist) reconfigure the default tab, KPI cards, and welcome briefing — each reflects a different decision-making context for the same underlying data.

---

## Models

- **XGBoost** — 49 engineered features, TimeSeriesSplit CV, SHAP explanations. Best base model for 48 of 51 BAs; reference run 0.98% MAPE on ERCOT (168-hour holdout, 2026-06-19).
- **Prophet** — weather regressors with logistic growth + floor=0 (structurally prevents negative forecasts). Daily / weekly / yearly seasonality.
- **SARIMAX** — auto-order selection via `pmdarima` on cold runs; cached `(p,d,q)(P,D,Q,m)` order on warm runs (skips the auto-select stepwise search, ~10× speedup on daily refresh).
- **Ensemble** — inverse-MAPE weighted combination (self-correcting; underperforming models lose weight automatically).

**Real holdout metrics, not simulated.** Each model's MAPE / RMSE / MAE / R² shown in the Models tab is the training-job's last 168-hour holdout, persisted to each pickle's `meta.json` in GCS and read at request time. The ensemble row is computed from the same holdout predictions so all four model metrics share provenance.

See [docs/BACKTEST_RESULTS.md](docs/BACKTEST_RESULTS.md) for full accuracy analysis on real EIA data.

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

**Data flow:** the scoring job runs hourly, fetching EIA + weather, loading the latest pickled models from GCS, and writing forecasts / alerts / diagnostics to Redis. The training job runs daily at 04:00 UTC, retraining each region's models on the last 60 days and persisting them back to GCS. The web service does cache-backed reads only — no API calls or model training in the request path.

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
│   ├── news_client.py              # External news feed integration
│   ├── redis_client.py             # Memorystore client (read-only at request time)
│   ├── cache.py                    # SQLite cache with TTL
│   ├── preprocessing.py            # Merge, align, interpolate, LTTB downsample
│   ├── feature_engineering.py      # 43 derived features (CDD/HDD, lags, rolling stats)
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
├── tests/                          # 1681 tests across unit / integration / e2e
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

## Testing

```bash
pytest tests/ -v
pytest tests/unit/ -v
pytest tests/e2e/ -v
```

See [tests/TEST_PYRAMID.md](tests/TEST_PYRAMID.md) for coverage targets and test strategy.

---

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

## Roadmap direction

GridPulse is being shaped to support a modular product architecture over time, including concepts like:
- GridPulse Forecast
- GridPulse Risk
- GridPulse Grid
- GridPulse Scenarios
- GridPulse Models
- GridPulse Briefings
- GridPulse API

The current repo should be treated as a technically credible foundation for that broader platform direction.
