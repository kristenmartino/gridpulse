# GridPulse — Energy Demand Forecasting Dashboard

Weather-aware energy demand forecasting for 8 U.S. balancing authorities. Combines real grid data (EIA), 17 meteorological variables (Open-Meteo), and ML models to predict hourly electricity demand.

Built for the NextEra Analytics portfolio on the stack NextEra uses: Python, Dash/Plotly, XGBoost, Prophet, and Cloud Run.

---

## Prerequisites

- **Python 3.11+** — required by Prophet and type hint syntax used throughout
- **pip** — comes with Python; used for dependency installation

```bash
# macOS (if you don't have Python 3.11+)
brew install python@3.11

# Verify
python3 --version  # should be 3.11.x or higher
```

No other system-level dependencies are required — all ML libraries (XGBoost, Prophet, SHAP) install via pip with pre-built wheels.

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

## What It Does

Three tabs, each answering a distinct operational question:

| Tab | Question | What It Shows |
|-----|----------|---------------|
| **Historical Demand** | What happened? | Actual recorded demand + EIA day-ahead forecast, weather overlay, comparative KPIs (peak, avg, min, EIA MAPE) |
| **Demand Forecast** | What will happen? | Forward-looking model predictions (Prophet, SARIMAX, XGBoost, Ensemble) with widening 80%/95% confidence bands |
| **Backtest** | How accurate are the models? | Model vs actuals on holdout periods, per-model MAPE, residual histograms |

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
┌─────────────────────────────────────────────────┐
│  Browser — Dash/Plotly Dark Theme               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Persona  │ │ Region   │ │ KPI Bar  │       │
│  │ Switcher │ │ Selector │ │          │       │
│  └──────────┘ └──────────┘ └──────────┘       │
│  ┌─────────────────────────────────────┐       │
│  │  [History] [Forecast] [Backtest]    │       │
│  └─────────────────────────────────────┘       │
└─────────────────┬───────────────────────────────┘
                  │ 21 Callback Groups
                  ▼
┌─────────────────────────────────────────────────┐
│  Model Service Layer                             │
│  get_forecasts() → trained model or simulation   │
│  Audit trail: model version, data vintage, hash  │
└─────────┬───────────────┬───────────────────────┘
          │               │
    ┌─────▼─────┐   ┌─────▼──────┐
    │ Data Layer│   │ ML Models  │
    │ EIA v2   │   │ Prophet    │
    │ Open-Meteo│   │ SARIMAX    │
    │ NOAA/NWS │   │ XGBoost    │
    │ SQLite   │   │ Ensemble   │
    └───────────┘   └────────────┘
```

**Data flow:** Region selection → API fetch (or demo fallback) → dcc.Store → tab callbacks → model service → Plotly figures. Every external dependency has a fallback chain: live API → stale cache → demo data.

---

## Project Structure

```
energy-forecast/
├── app.py                          # Entry point (port 8080)
├── config.py                       # All constants, regions, thresholds
├── components/
│   ├── layout.py                   # Main layout (3-tab container)
│   ├── callbacks.py                # 21 callback groups
│   ├── cards.py                    # KPI, welcome, alert cards
│   ├── tab_forecast.py             # Historical Demand tab
│   ├── tab_demand_outlook.py       # Demand Forecast tab
│   └── tab_backtest.py             # Backtest tab
├── data/
│   ├── eia_client.py               # EIA API v2 (demand, generation)
│   ├── weather_client.py           # Open-Meteo (17 weather variables)
│   ├── noaa_client.py              # NOAA/NWS severe weather alerts
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
├── simulation/                     # Scenario engine (dormant — see below)
├── personas/                       # 4 role-based persona configs
├── tests/                          # 19 test files (unit/integration/e2e)
├── Dockerfile                      # Multi-stage, non-root, healthcheck
└── .github/workflows/              # CI, staging deploy, prod deploy
```

**Dormant modules:** `simulation/`, `components/tab_weather.py`, `tab_models.py`, `tab_generation.py`, `tab_alerts.py`, and `tab_simulator.py` contain completed implementations from earlier sprints. They were removed from the active tab set during an architecture review that focused the dashboard on its three core views. Code is preserved for future reactivation.

---

## Deployment

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
  --set-env-vars EIA_API_KEY=your_key

# Health check
curl http://localhost:8080/health
```

## Testing

```bash
pytest tests/ -v                    # Full suite (361 tests)
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
