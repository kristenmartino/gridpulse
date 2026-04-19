# GridPulse Technical Specification

_Last updated: 2026-04-09_

Technical documentation for the GridPulse **Energy Intelligence Platform**.

This document covers the implementation details behind GridPulse’s forecasting, grid visibility, validation, and supporting platform behaviors.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Sources](#2-data-sources)
3. [Region Definitions](#3-region-definitions)
4. [Feature Engineering](#4-feature-engineering)
5. [Forecasting and Model Layer](#5-forecasting-and-model-layer)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Caching and Fallback Strategy](#7-caching-and-fallback-strategy)
8. [Data Processing Pipeline](#8-data-processing-pipeline)
9. [Product-Shell Mapping](#9-product-shell-mapping)
10. [Assumptions and Limitations](#10-assumptions-and-limitations)
11. [Environment Variables](#11-environment-variables)
12. [Repository Structure](#12-repository-structure)

---

## 1. System Overview

GridPulse is a Dash/Plotly-based application that combines:
- real energy demand data
- weather inputs
- multiple forecasting models
- model validation and confidence context
- generation and net load context
- alert/risk-oriented views
- scenario simulation
- role-aware presentation and briefing patterns

### Primary technical capabilities
- hourly demand forecasting across 8 balancing authorities
- feature-engineered model inputs built from weather and time-series signals
- validation and backtest workflows for trust and accountability
- cache-backed serving with resilience patterns
- role-specific product surfaces across a shared data/model core

### Current runtime architecture
- Dash app served via Flask/Gunicorn-compatible entrypoint on Cloud Run
- Redis/Memorystore used as the **sole** production read path; the web
  service never fetches EIA/Open-Meteo or trains models inline
- GCS (`gs://nextera-portfolio-energy-cache/models/`) stores trained
  XGBoost / Prophet / SARIMAX pickles via `models/persistence.py`
- SQLite used for local or app-layer caching patterns (dev mode)
- Scheduled Cloud Run Jobs populate Redis and GCS:
  - `gridpulse-scoring-job` — hourly, writes forecasts/alerts/diagnostics
  - `gridpulse-training-job` — daily (04:00 UTC), persists new models
  - Shared phase logic lives in `jobs/phases.py`, CLI dispatcher in
    `jobs/__main__.py` (`python -m jobs {scoring|training}`)
- `REQUIRE_REDIS` config flag gates the legacy inline compute fallback
  (true in staging/production, false in development)

---

## 2. Data Sources

## 2.1 EIA API v2 (Demand / Generation)

**Source:** U.S. Energy Information Administration  
**Base URL:** `https://api.eia.gov/v2`  
**Authentication:** API key required (`EIA_API_KEY`)  

### Primary demand source
**Endpoint family:** `electricity/rto/region-data`

**Representative demand fields:**

| Field | Type | Description |
|---|---|---|
| `timestamp` | datetime (UTC) | Hour of observation |
| `demand_mw` | float | Actual demand in megawatts |
| `region` | string | Balancing authority code |

**Representative request parameters:**
- demand type facets
- hourly frequency
- ascending time sort
- bounded historical fetch window

**Default historical range:** approximately 90 days

### Generation source
Generation-by-fuel data is also fetched for supply-side context, fuel mix views, and net-load-style analysis where available.

### Error handling intent
- retries with exponential backoff
- cache-backed fallback when possible
- avoid overwriting real cached data with fake data in degraded production paths

---

## 2.2 Open-Meteo API (Weather)

**Source:** Open-Meteo  
**Forecast URL:** `https://api.open-meteo.com/v1/forecast`  
**Archive URL:** `https://archive-api.open-meteo.com/v1/archive`  
**Authentication:** None required  

### Representative hourly variables

| Variable | Unit | Description |
|---|---|---|
| `temperature_2m` | °F | Air temperature at 2m |
| `apparent_temperature` | °F | Feels-like temperature |
| `relative_humidity_2m` | % | Relative humidity |
| `dew_point_2m` | °F | Dew point |
| `wind_speed_10m` | mph | Wind at 10m |
| `wind_speed_80m` | mph | Wind at hub height |
| `wind_speed_120m` | mph | Wind at 120m |
| `wind_direction_10m` | ° | Wind direction |
| `shortwave_radiation` | W/m² | Global horizontal irradiance |
| `direct_normal_irradiance` | W/m² | Direct beam radiation |
| `diffuse_radiation` | W/m² | Diffuse sky radiation |
| `cloud_cover` | % | Cloud cover |
| `precipitation` | mm | Total precipitation |
| `snowfall` | cm | Snowfall |
| `surface_pressure` | hPa | Atmospheric pressure |
| `soil_temperature_0cm` | °F | Surface soil temperature |
| `weather_code` | WMO code | Encoded condition |

### Representative request parameters
- `past_days`
- `forecast_days`
- Fahrenheit temperature units
- mph wind-speed units
- UTC timezone handling

---

## 2.3 NOAA / NWS Alerts

**Source:** NOAA / National Weather Service  
**Purpose:** severe weather and alert context for regions  

This source supports alert- and extreme-event-oriented views. It is not the primary weather forecasting source but provides risk context layered over energy and forecast views.

---

## 2.4 External News / Signals Feed

GridPulse includes an external news/signals integration for contextual headlines. This is auxiliary to the forecasting core and should be treated as a supporting signal rather than a primary system dependency.

### Behavior notes
- feed responses may be cached depending on implementation path
- demo/static fallback behavior may exist for news-like surfaces
- this layer should not be treated as a critical forecasting dependency

---

## 3. Region Definitions

## 3.1 Balancing Authorities

| Code | Full Name | Centroid (lat, lon) | Approx. Capacity (MW) |
|---|---|---|---|
| ERCOT | Texas (ERCOT) | 31.0, -97.0 | 130,000 |
| CAISO | California (CAISO) | 37.0, -120.0 | 80,000 |
| PJM | Mid-Atlantic (PJM) | 39.5, -77.0 | 185,000 |
| MISO | Midwest (MISO) | 41.0, -89.0 | 175,000 |
| NYISO | New York (NYISO) | 42.5, -74.0 | 38,000 |
| FPL | Florida (FPL/NextEra) | 26.9, -80.1 | 32,000 |
| SPP | Southwest (SPP) | 35.5, -97.5 | 90,000 |
| ISONE | New England (ISO-NE) | 42.3, -71.8 | 30,000 |

### Note on FPL
FPL represents Florida Power & Light’s service territory, not all statewide Florida demand.

## 3.2 Weather Location Assumption
Weather data is fetched for centroid coordinates representing each balancing authority. This is a pragmatic simplification and can reduce fidelity for geographically large regions.

---

## 4. Feature Engineering

All features below are used as model inputs. Most are not directly displayed in the UI but influence forecast quality.

## 4.1 Input Requirements
- `timestamp`: UTC timezone-aware datetime
- `demand_mw`: actual demand in MW
- weather variables from Open-Meteo

## 4.2 Derived Feature Categories

### Temperature-based features
| Feature | Formula / Logic | Purpose |
|---|---|---|
| `cooling_degree_days` | `max(0, temp_F - 65)` | Cooling load proxy |
| `heating_degree_days` | `max(0, 65 - temp_F)` | Heating load proxy |
| `temperature_deviation` | temp vs rolling historical mean | Anomaly signal |

### Time-based features
| Feature | Logic |
|---|---|
| `hour_sin` / `hour_cos` | cyclic encoding of hour-of-day |
| `dow_sin` / `dow_cos` | cyclic encoding of day-of-week |
| `is_weekend` | binary weekend flag |

### Lag features
| Feature | Offset | Purpose |
|---|---|---|
| `demand_lag_24h` | t - 24h | same time yesterday |
| `demand_lag_168h` | t - 168h | same time last week |
| `ramp_rate` | demand_t - demand_t-1 | hour-over-hour change |

### Rolling statistics
Multiple windows are used (for example 24h / 72h / 168h) with statistics such as:
- mean
- std
- min
- max

### Interaction and domain-specific features
Examples include:
- wind power estimate
- solar capacity factor estimate
- temperature × hour interactions
- anomaly-style deviations from rolling temperature context

## 4.3 Processing Rules
1. No future leakage — all engineered features are backward-looking only.
2. Lag-induced NaNs are handled explicitly.
3. Timestamps are aligned to hourly cadence.
4. Resulting model matrix is roughly 43 features depending on path.

---

## 5. Forecasting and Model Layer

## 5.1 XGBoost (Primary Model)

**Library:** `xgboost.XGBRegressor`

### Typical hyperparameter profile
| Parameter | Representative Value |
|---|---|
| `n_estimators` | 500 |
| `max_depth` | 6 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_alpha` | 0.1 |
| `reg_lambda` | 1.0 |
| `random_state` | 42 |

### Validation approach
- time-series-aware cross-validation
- no future leakage across train/validation splits
- MAPE emphasized as a core business metric

### Model output behavior
- predictions are clamped or constrained to avoid nonsensical negative demand values

---

## 5.2 Prophet

Prophet is used as a comparative model with weather regressors and seasonality structure.

---

## 5.3 SARIMAX / ARIMA

SARIMAX provides a statistical benchmark path with auto-order support via pmdarima.

---

## 5.4 Ensemble

The ensemble combines multiple models using inverse-MAPE weighting.

### Weight logic
```text
weight_i = (1 / MAPE_i) / Σ(1 / MAPE_j)
```

This keeps better-performing models more influential while bounding behavior to the constituent models.

---

## 5.5 Model Service and Auditability

The model layer is abstracted behind service logic so the UI is insulated from direct model-training concerns.

Supporting technical concerns include:
- trained vs simulated fallback behavior where applicable
- forecast audit trail metadata
- evaluation artifacts
- confidence and validation context for UI surfaces

---

## 6. Evaluation Metrics

| Metric | Formula / Meaning | Units |
|---|---|---|
| **MAPE** | mean absolute percentage error | % |
| **RMSE** | root mean squared error | MW |
| **MAE** | mean absolute error | MW |
| **R²** | coefficient of determination | unitless |

### Typical MAPE interpretation
| MAPE | Interpretation |
|---|---|
| < 3% | Excellent |
| 3–5% | Good |
| 5–10% | Acceptable |
| > 10% | Poor |

GridPulse also uses horizon-aware governance and grade concepts in parts of the system rather than relying only on a single raw threshold.

---

## 7. Caching and Fallback Strategy

## 7.1 Cache Locations
Representative cache layers include:

| Data Type | Cache Location | Typical Behavior |
|---|---|---|
| Demand / weather data | Redis (prod) / SQLite (dev) | TTL-based reuse; web reads only |
| News/signals | feed-specific path | may be refreshed more frequently |
| Trained models | GCS `models/` + local disk cache | region-scoped, pickled, `latest.json` pointer |
| Predictions / backtests | Redis, written by scoring/training jobs | region/horizon scoped |

## 7.2 Invalidation Principles
- TTL expiry (Redis keys, 24h default)
- input/data-hash change recorded in GCS model metadata
- explicit refresh or new scheduled job run
- atomic `latest.json` flip on training-job success

## 7.3 Fallback Behavior
### Intended production behavior
1. Serve fresh Redis data written by the hourly scoring job.
2. Serve stale Redis entries (with staleness badges) when the scoring job
   has missed a cycle but values still exist.
3. Surface a `warming` state when `REQUIRE_REDIS=True` and Redis has no
   entry yet — the UI renders a "Data warming up" message instead of
   blocking on an inline fetch or training.
4. Use demo/synthetic data explicitly in demo/offline contexts, not as a
   silent overwrite of real operational state.

This behavior matters because operational credibility is reduced when fake data overwrites previously valid real data during upstream outages.

---

## 8. Data Processing Pipeline

## 8.1 Representative Pipeline

```text
1. fetch_demand(region)
2. fetch_weather(region)
3. merge_demand_weather()
4. engineer_features()
5. train_or_load_models()
6. produce forecasts / backtests / supporting artifacts
7. render via callback-driven UI surfaces
```

## 8.2 Data Merging
- left-join style alignment on hourly timestamps
- timestamp normalization to consistent hourly cadence
- duplicate handling by stable preference rules

## 8.3 Missing Value Handling
| Gap Size | Treatment |
|---|---|
| small gaps | interpolation or controlled fill strategy |
| large gaps | explicit degraded handling / no-fill behavior |

## 8.4 Timezone Handling
- timestamps normalized to UTC for internal processing
- local/display conversion handled separately where needed

---

## 9. Product-Shell Mapping

The current codebase implements multiple tabs. Conceptually, these map to the product shell as follows:

| Current/Implementation Surface | Product Meaning |
|---|---|
| `tab_overview.py` | Overview / mission-control |
| `tab_forecast.py` | Historical Demand |
| `tab_demand_outlook.py` | Demand Forecast |
| `tab_backtest.py` + `tab_models.py` | Models / validation / trust |
| `tab_generation.py` | Grid |
| `tab_alerts.py` + parts of `tab_weather.py` | Risk / conditions |
| `tab_simulator.py` | Scenarios |
| cards + meeting mode + summaries | Briefings / stakeholder layer |

This mapping is important for product-shell redesign work and should guide naming updates before deep refactors.

---

## 10. Assumptions and Limitations

## 10.1 Data Assumptions
| Assumption | Implication |
|---|---|
| EIA data is sufficiently reliable for operational analytics | independent validation is limited |
| Single weather point per region is acceptable | large regions may be underrepresented |
| Historical patterns remain informative | structural changes can degrade performance |
| Hourly cadence is sufficient | sub-hourly peaks are not captured |

## 10.2 Known Limitations
| Limitation | Impact |
|---|---|
| Single weather point per region | lower fidelity for PJM, MISO, and other broad regions |
| FPL is not all of Florida | geographic interpretation must stay precise |
| No explicit demand-response modeling | DR events may not be reflected |
| No outage-aware generation modeling | major unit outages may not be captured |
| Non-real-time architecture | not designed for sub-hour operational dispatch |

---

## 11. Environment Variables

| Variable | Required | Default / Notes | Description |
|---|---|---|---|
| `EIA_API_KEY` | Yes for live EIA data | none | EIA API authentication |
| `ENVIRONMENT` | No | development | deployment tier |
| `PORT` | No | 8080 | server port |
| `CACHE_DB_PATH` | No | app default | SQLite cache path |
| `CACHE_TTL_SECONDS` | No | app default | cache TTL |
| other service-specific vars | context-dependent | varies | Redis, metrics, secrets, etc. |

Some older documentation may reference news-specific keys or older fallback assumptions. Prefer the code path and current config when those differ.

---

## 12. Repository Structure

```text
.
├── app.py                     # Dash app entry point
├── config.py                  # Constants, labels, env config, feature flags
├── components/
│   ├── layout.py              # Product shell / header / tab structure
│   ├── callbacks.py           # Callback orchestration and data-loading flows
│   ├── cards.py               # KPI, alert, briefing, and supporting cards
│   ├── error_handling.py      # Confidence badges, loading states, degraded-state UI
│   ├── accessibility.py       # Accessibility helpers and color-safe support
│   ├── tab_overview.py        # Overview surface
│   ├── tab_forecast.py        # Historical demand
│   ├── tab_demand_outlook.py  # Demand forecast
│   ├── tab_backtest.py        # Backtest
│   ├── tab_generation.py      # Generation and net load
│   ├── tab_weather.py         # Weather/correlation
│   ├── tab_models.py          # Models / diagnostics
│   ├── tab_alerts.py          # Extreme events / alerts
│   └── tab_simulator.py       # Scenario simulation
├── data/
│   ├── cache.py               # Cache layer
│   ├── eia_client.py          # EIA client
│   ├── weather_client.py      # Open-Meteo client
│   ├── noaa_client.py         # NOAA/NWS client
│   ├── news_client.py         # External news/signals client
│   ├── preprocessing.py       # Data merge / align / validate
│   ├── feature_engineering.py # Derived features
│   ├── audit.py               # Forecast audit trail
│   └── demo_data.py           # Demo/offline data generation
├── models/
│   ├── model_service.py       # Forecast service abstraction
│   ├── xgboost_model.py       # XGBoost path + SHAP
│   ├── prophet_model.py       # Prophet path
│   ├── arima_model.py         # SARIMAX path
│   ├── ensemble.py            # Ensemble logic
│   ├── evaluation.py          # Metrics
│   ├── training.py            # Training orchestration
│   └── pricing.py             # Pricing/support logic
├── simulation/                # Scenario engine and presets
├── personas/                  # Role/view configuration
├── observability.py           # Logging / pipeline observability
├── jobs/                      # Cloud Run Jobs (scoring, training, phases, CLI)
├── tests/                     # Unit / integration / e2e
├── Dockerfile                 # Container spec
└── .github/workflows/         # CI / deploy automation
```

---

*Document Version: Updated for platform framing and current implementation guidance*