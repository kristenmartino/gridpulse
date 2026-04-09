# GridPulse — Energy Intelligence Platform

**[gridpulse.kristenmartino.ai](https://gridpulse.kristenmartino.ai)**

Weather-aware energy intelligence for 8 U.S. balancing authorities. GridPulse combines real grid data (EIA), meteorological signals (Open-Meteo), multiple forecasting models, model validation, generation context, and scenario analysis in one decision-ready platform.

Built for the NextEra Analytics portfolio on a stack aligned with modern data and analytics workflows: Python, Dash/Plotly, XGBoost, Prophet, and Cloud Run.

**Tagline:** _See demand sooner. Decide with confidence._

---

## What GridPulse Does

GridPulse is designed to help energy teams move from fragmented monitoring to a more unified operating view. It brings together:
- demand visibility
- weather-aware forecasting
- forecast confidence and backtesting
- generation and net load context
- alerts and extreme-event monitoring
- scenario analysis
- role-based views and briefings

### Core questions it helps answer

| Product Area | Question | What It Shows |
|---|---|---|
| **Overview** | What changed, and what matters now? | Mission-control summary, key KPIs, context, and role-aware briefing |
| **Historical Demand** | What happened? | Actual recorded demand, EIA day-ahead forecast overlay, weather context, comparative KPIs |
| **Demand Forecast** | What will happen? | Forward-looking model predictions (Prophet, SARIMAX, XGBoost, Ensemble) with confidence bands |
| **Backtest / Models** | How trustworthy is the forecast? | Model vs actuals, per-model MAPE, residuals, validation context |
| **Generation & Net Load** | What is happening on the supply side? | Generation mix breakdown, renewable share, net load trends |
| **Risk / Extreme Events** | Where is operating risk rising? | Severe-weather signals, anomalies, stress indicators, degraded conditions |
| **Scenarios** | What changes if conditions shift? | What-if analysis, weather overrides, scenario presets, impact comparisons |

Four role-based personas (Grid Ops, Renewables Analyst, Trader, Data Scientist) reconfigure the default tab, KPI cards, and welcome briefing. Each persona reflects a different decision-making context for the same underlying data.

---

## Regions

ERCOT · CAISO · PJM · MISO · NYISO · **FPL (NextEra)** · SPP · ISO-NE

---

## Models

- **XGBoost**: 43 engineered features, TimeSeriesSplit CV, SHAP explanations — 3.13% MAPE on ERCOT 21-day holdout
- **Prophet**: weather regressors with multiplicative seasonality
- **SARIMAX**: auto-order selection via pmdarima
- **Ensemble**: inverse-MAPE weighted combination (self-correcting)

See [docs/BACKTEST_RESULTS.md](docs/BACKTEST_RESULTS.md) for full accuracy analysis on real EIA data.

---

## Positioning

GridPulse is not just a forecasting dashboard. It is evolving into an **energy intelligence platform** for:
- forecast confidence
- grid visibility
- role-aware operational decision support
- scenario-ready analysis

This positioning matters because the product already includes more than raw forecasting: it combines data, context, validation, and workflow support for multiple energy personas.

---

## Architecture

```text
┌────────────────────────────────────────────────────────────────────┐
│  Browser — gridpulse.kristenmartino.ai                             │
│  ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────────┐  │
│  │ View /     │ │ Region   │ │ KPI Bar  │ │ Briefings /        │  │
│  │ Persona    │ │ Selector │ │          │ │ Signals            │  │
│  │ Selector   │ │          │ │          │ │                    │  │
│  └────────────┘ └──────────┘ └──────────┘ └────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Overview | History | Forecast | Models | Grid | Risk | ... │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────┬──────────────────────────────────────────┘
                          │ Callback + store-driven rendering
                          ▼
┌────────────────────────────────────────────────────────────────────┐
│  Redis (Memorystore) ← pre-computed by Cloud Run Job (12h cron)   │
│  Fallback chain: Redis → live API → stale cache → explicit demo   │
│  mode / no-data states depending on context                       │
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

**Data flow:** Region selection triggers cache-backed reads. If pre-computed or cached data exists, charts render quickly. If not, the app attempts live fetch and compute paths. The system is designed to prefer real/stale operational data over fake data in degraded production paths, while still supporting explicit offline/demo contexts when needed.

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
│   ├── tab_forecast.py             # Historical Demand screen
│   ├── tab_demand_outlook.py       # Demand Forecast screen
│   ├── tab_backtest.py             # Backtest screen
│   ├── tab_generation.py           # Generation & Net Load screen
│   ├── tab_weather.py              # Weather / correlation screen
│   ├── tab_models.py               # Model diagnostics screen
│   ├── tab_alerts.py               # Extreme events / alerts screen
│   └── tab_simulator.py            # Scenario simulator screen
├── data/
│   ├── eia_client.py               # EIA API v2
│   ├── weather_client.py           # Open-Meteo
│   ├── noaa_client.py              # NOAA/NWS alerts
│   ├── news_client.py              # External news feed integration
│   ├── cache.py                    # SQLite cache with TTL
│   ├── preprocessing.py            # Merge, align, interpolate, validate
│   ├── feature_engineering.py      # 43 derived features
│   ├── audit.py                    # Forecast audit trail
│   └── demo_data.py                # Synthetic/offline demo data utilities
├── models/
│   ├── model_service.py            # Forecast service abstraction
│   ├── prophet_model.py            # Prophet model
│   ├── arima_model.py              # SARIMAX model
│   ├── xgboost_model.py            # XGBoost model + SHAP
│   ├── ensemble.py                 # Ensemble weighting logic
│   ├── evaluation.py               # MAPE, RMSE, MAE, R²
│   └── pricing.py                  # Merit-order pricing model
├── scaling-analytics/              # Scaled / precompute scaffold
├── personas/                       # Persona configs and welcome logic
├── tests/                          # Unit / integration / e2e tests
├── Dockerfile                      # Multi-stage, non-root, healthcheck
└── .github/workflows/              # CI / deploy workflows
```

---

## Deployment

**Production** is deployed automatically on push to `main` via GitHub Actions.

```text
Cloud Run (gridpulse)  →  gridpulse.kristenmartino.ai
  ├── Memorystore (Redis)  →  pre-computed forecasts + backtests
  ├── Cloud Run Job        →  populate-redis (12h cron via Cloud Scheduler)
  └── VPC Connector        →  links Run to Redis
```

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
| [CLAUDE.md](CLAUDE.md) | AI coding assistant context, architecture, conventions, execution guardrails |
| [EXECUTION_BRIEF.md](EXECUTION_BRIEF.md) | Agent-ready prioritization layer for redesign and repositioning work |
| [PRD.md](PRD.md) | Product requirements, personas, ADRs, scope decisions |
| [TECHNICAL_SPEC.md](TECHNICAL_SPEC.md) | Data sources, features, models, caching, pipeline details |
| [docs/BACKTEST_RESULTS.md](docs/BACKTEST_RESULTS.md) | Model accuracy on real EIA holdout data |

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
