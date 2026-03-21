# GridPulse — Energy Demand Forecasting Dashboard

**[gridpulse.kristenmartino.ai](https://gridpulse.kristenmartino.ai)**

Weather-aware energy demand forecasting for 8 U.S. balancing authorities. Combines real grid data (EIA), 17 meteorological variables (Open-Meteo), and ML models to predict hourly electricity demand.

Built for the NextEra Analytics portfolio on the stack NextEra uses: Python, Dash/Plotly, XGBoost, Prophet, and Cloud Run.

---

## What It Does

Four tabs, each answering a distinct operational question:

| Tab | Question | What It Shows |
|-----|----------|---------------|
| **Historical Demand** | What happened? | Actual recorded demand + EIA day-ahead forecast, weather overlay, comparative KPIs (peak, avg, min, EIA MAPE) |
| **Demand Forecast** | What will happen? | Forward-looking model predictions (Prophet, SARIMAX, XGBoost, Ensemble) with widening 80%/95% confidence bands |
| **Backtest** | How accurate are the models? | Model vs actuals on holdout periods, per-model MAPE, residual histograms |
| **Generation & Net Load** | Where does the power come from? | Generation mix breakdown, renewable share, net load trends |

Four role-based personas (Grid Ops, Renewables Analyst, Trader, Data Scientist) reconfigure the default tab, KPI cards, and welcome briefing. Each persona reflects a different decision-making context for the same underlying data.

### Regions

ERCOT · CAISO · PJM · MISO · NYISO · **FPL (NextEra)** · SPP · ISO-NE

### Models

- **XGBoost**: 43 engineered features, TimeSeriesSplit CV, SHAP explanations — 3.13% MAPE on ERCOT 21-day holdout
- **Prophet**: 7 weather regressors, multiplicative seasonality
- **SARIMAX**: Auto-order selection via pmdarima
- **Ensemble**: Inverse-MAPE weighted combination (self-correcting)

See [docs/BACKTEST_RESULTS.md](docs/BACKTEST_RESULTS.md) for full accuracy analysis on real EIA data.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser — gridpulse.kristenmartino.ai                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────────┐ │
│  │ Persona  │ │ Region   │ │ KPI Bar  │ │ Energy News Ticker│ │
│  │ Switcher │ │ Selector │ │          │ │ (Google News RSS) │ │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────────┘ │
│  ┌──────────────────────────────────────────────┐              │
│  │ [History] [Forecast] [Backtest] [Generation] │              │
│  └──────────────────────────────────────────────┘              │
└────────────────────────┬────────────────────────────────────────┘
                         │ 21 Callback Groups
                         ▼
┌──────────────────────────────────────────────────┐
│  Redis (Memorystore)  ←  Pre-computed by         │
│  Read-only serving       Cloud Run Job (12h cron)│
│                          ↓                       │
│  Fallback: v1 compute   EIA API + Open-Meteo     │
│  path (EIA → features   → XGBoost train          │
│  → train → predict)     → forecasts + backtests  │
└──────────┬───────────────┬───────────────────────┘
     ┌─────▼─────┐   ┌─────▼──────┐
     │ Data Layer│   │ ML Models  │
     │ EIA v2   │   │ XGBoost    │
     │ Open-Meteo│   │ Prophet    │
     │ Google   │   │ SARIMAX    │
     │ News RSS │   │ Ensemble   │
     │ SQLite   │   │ SHAP       │
     └───────────┘   └────────────┘
```

**Data flow:** Region selection triggers a Redis read. If cached data exists (pre-computed every 12h), charts render instantly. If Redis is unavailable, the v1 compute path activates: API fetch → feature engineering → model training → prediction. Every external dependency has a fallback chain: Redis → live API → stale cache → demo data.

---

## Quick Start

```bash
cd energy-forecast
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
# → http://localhost:8080
```

No API keys required — the app runs in demo mode with synthetic data for all 8 regions. For live data, set `EIA_API_KEY` (free at [eia.gov/opendata](https://www.eia.gov/opendata/)).

---

## Project Structure

```
energy-forecast/
├── app.py                          # Entry point (port 8080)
├── config.py                       # All constants, regions, thresholds
├── components/
│   ├── layout.py                   # Main layout (4-tab container)
│   ├── callbacks.py                # 21 callback groups
│   ├── cards.py                    # KPI, welcome, news ticker cards
│   ├── tab_forecast.py             # Historical Demand tab
│   ├── tab_demand_outlook.py       # Demand Forecast tab
│   ├── tab_backtest.py             # Backtest tab
│   └── tab_generation.py           # Generation & Net Load tab
├── data/
│   ├── eia_client.py               # EIA API v2 (demand, generation)
│   ├── weather_client.py           # Open-Meteo (17 weather variables)
│   ├── news_client.py              # Google News RSS (energy headlines)
│   ├── redis_client.py             # Redis read layer (Memorystore)
│   ├── preprocessing.py            # Merge, align, interpolate, validate
│   ├── feature_engineering.py      # 43 derived features
│   ├── cache.py                    # SQLite with TTL + stale fallback
│   ├── audit.py                    # Forecast audit trail
│   └── demo_data.py                # Synthetic data (offline mode)
├── models/
│   ├── model_service.py            # Forecast service (trained ↔ simulated)
│   ├── prophet_model.py            # Prophet with weather regressors
│   ├── arima_model.py              # SARIMAX with auto-order
│   ├── xgboost_model.py            # XGBoost + SHAP
│   ├── ensemble.py                 # 1/MAPE weighted combination
│   ├── evaluation.py               # MAPE, RMSE, MAE, R²
│   └── pricing.py                  # Merit-order pricing model
├── scaling-analytics/              # v2 pre-computation scaffold (see below)
├── personas/                       # 4 role-based persona configs
├── tests/                          # 19 test files (unit/integration/e2e)
├── Dockerfile                      # Multi-stage, non-root, healthcheck
└── .github/workflows/              # CI, staging deploy, prod deploy
```

### Scaling Analytics (v2 Scaffold)

The `scaling-analytics/` directory contains the full pre-computation pipeline architecture: Airflow DAGs, Kafka consumers/producers, FastAPI server, batch scorer, and Postgres schema. This is designed for production-scale deployment with Cloud Composer and managed Kafka. Currently, the production dashboard uses a simplified version of this pipeline — a Cloud Run Job on a 12-hour Cloud Scheduler cron that populates Redis (Memorystore).

---

## Deployment

**Production** is deployed automatically on push to `main` via GitHub Actions.

```
Cloud Run (gridpulse)  →  gridpulse.kristenmartino.ai
  ├── Memorystore (Redis)  →  pre-computed forecasts + backtests
  ├── Cloud Run Job        →  populate-redis (12h cron via Cloud Scheduler)
  └── VPC Connector        →  wattcast-connector (links Run to Redis)
```

### Manual deployment

```bash
# Docker
docker build -t gridpulse .
docker run -p 8080:8080 -e EIA_API_KEY=your_key gridpulse

# Google Cloud Run
gcloud builds submit --tag us-east1-docker.pkg.dev/nextera-portfolio/portfolio/gridpulse
gcloud run deploy gridpulse \
  --image us-east1-docker.pkg.dev/nextera-portfolio/portfolio/gridpulse \
  --platform managed --allow-unauthenticated \
  --memory 2Gi --timeout 300 \
  --set-env-vars EIA_API_KEY=your_key,REDIS_HOST=<memorystore-ip>
```

---

## Testing

```bash
pytest tests/ -v                    # Full suite (440+ tests)
pytest tests/unit/ -v               # Fast feedback
pytest tests/e2e/ -v                # Dashboard rendering
```

See [tests/TEST_PYRAMID.md](tests/TEST_PYRAMID.md) for coverage targets and test strategy.

---

## Documentation

| Doc | Purpose |
|-----|---------|
| [PRD.md](PRD.md) | Product requirements, personas, descoping rationale |
| [TECHNICAL_SPEC.md](TECHNICAL_SPEC.md) | Data sources, feature engineering, model config, caching |
| [CLAUDE.md](CLAUDE.md) | AI coding assistant context and project conventions |
| [docs/BACKTEST_RESULTS.md](docs/BACKTEST_RESULTS.md) | Model accuracy on real EIA holdout data |
