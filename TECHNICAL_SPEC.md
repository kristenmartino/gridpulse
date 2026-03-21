# GridPulse Technical Specification

Technical documentation for the GridPulse Energy Demand Forecasting Dashboard.

---

## Table of Contents

1. [Data Sources](#1-data-sources)
2. [Region Definitions](#2-region-definitions)
3. [Feature Engineering](#3-feature-engineering)
4. [Forecasting Model](#4-forecasting-model)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Caching Strategy](#6-caching-strategy)
7. [Data Processing Pipeline](#7-data-processing-pipeline)
8. [Assumptions & Limitations](#8-assumptions--limitations)

---

## 1. Data Sources

### 1.1 EIA API v2 (Energy Demand)

**Source:** U.S. Energy Information Administration
**Base URL:** `https://api.eia.gov/v2`
**Authentication:** API key required (env: `EIA_API_KEY`)
**Endpoint:** `electricity/rto/region-data`

**Data Fields Retrieved:**

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime (UTC) | Hour of observation |
| `demand_mw` | float | Actual demand in megawatts |
| `region` | string | Balancing authority code |

**API Parameters:**
- `facets[type][]`: "D" (actual demand)
- `frequency`: hourly
- `length`: 5000 (page size)
- `sort`: period ascending

**Default Date Range:** 90 days historical

**EIA Region Code Mapping:**

| Dashboard Code | EIA API Code |
|----------------|--------------|
| ERCOT | ERCO |
| CAISO | CISO |
| PJM | PJM |
| MISO | MISO |
| NYISO | NYIS |
| FPL | FPL |
| SPP | SWPP |
| ISONE | ISNE |

**Error Handling:**
- 5 retries with exponential backoff (starting 2 seconds)
- Stale cache fallback on API failure
- Demo data fallback if no cache available

---

### 1.2 Open-Meteo API (Weather)

**Source:** Open-Meteo (free, no API key)
**Forecast URL:** `https://api.open-meteo.com/v1/forecast`
**Archive URL:** `https://archive-api.open-meteo.com/v1/archive`

**17 Weather Variables (Hourly):**

| Variable | Unit | Description |
|----------|------|-------------|
| `temperature_2m` | °F | Air temperature at 2m |
| `apparent_temperature` | °F | Feels-like temperature |
| `relative_humidity_2m` | % | Relative humidity |
| `dew_point_2m` | °F | Dew point temperature |
| `wind_speed_10m` | mph | Wind at 10m height |
| `wind_speed_80m` | mph | Wind at 80m (turbine hub height) |
| `wind_speed_120m` | mph | Wind at 120m height |
| `wind_direction_10m` | ° | Wind direction (0-360) |
| `shortwave_radiation` | W/m² | Global horizontal irradiance (GHI) |
| `direct_normal_irradiance` | W/m² | Direct beam radiation |
| `diffuse_radiation` | W/m² | Diffuse sky radiation |
| `cloud_cover` | % | Total cloud cover |
| `precipitation` | mm | Total precipitation |
| `snowfall` | cm | Snowfall amount |
| `surface_pressure` | hPa | Atmospheric pressure |
| `soil_temperature_0cm` | °F | Surface soil temperature |
| `weather_code` | WMO code | Weather condition code |

**Request Parameters:**
- `past_days`: 92 (historical data)
- `forecast_days`: 7 (future forecast)
- `temperature_unit`: fahrenheit
- `wind_speed_unit`: mph
- `timezone`: UTC

---

### 1.3 Google News RSS (Energy News)

**Source:** Google News RSS
**URL:** `https://news.google.com/rss/search?q=electricity+grid+OR+renewable+energy+OR+solar+power+OR+wind+power+OR+power+demand+OR+energy+prices+OR+ERCOT+OR+CAISO+OR+power+grid&hl=en-US&gl=US&ceid=US:en`
**Authentication:** None required (free, no API key)

**Response Format:** RSS/XML with `<item>` elements containing:
- `<title>` — headline (source name appended after ` - `)
- `<link>` — article URL
- `<pubDate>` — RFC 2822 date
- `<source>` — publisher name

**Parameters:**
- `page_size`: 10 articles (displayed in auto-scrolling ticker)

**Caching:** SQLite with 30-minute TTL. Falls back to demo articles with real URLs (EIA, DOE, NextEra) if RSS fetch fails.

---

## 2. Region Definitions

### 2.1 Balancing Authorities

| Code | Full Name | Centroid (lat, lon) | Capacity (MW) |
|------|-----------|---------------------|---------------|
| ERCOT | Texas (ERCOT) | 31.0, -97.0 | 130,000 |
| CAISO | California (CAISO) | 37.0, -120.0 | 80,000 |
| PJM | Mid-Atlantic (PJM) | 39.5, -77.0 | 185,000 |
| MISO | Midwest (MISO) | 41.0, -89.0 | 175,000 |
| NYISO | New York (NYISO) | 42.5, -74.0 | 38,000 |
| FPL | Florida (FPL/NextEra)* | 26.9, -80.1 | 32,000 |
| SPP | Southwest (SPP) | 35.5, -97.5 | 90,000 |
| ISONE | New England (ISO-NE) | 42.3, -71.8 | 30,000 |

**Capacity Source:** EIA-860 (annual generator data)

*\*FPL represents Florida Power & Light's service territory (~50% of Florida), not statewide demand. Florida has 9 separate balancing authorities.*

### 2.2 Weather Location

Weather data is fetched for the **centroid coordinates** of each balancing authority (single point per region).

---

## 3. Feature Engineering

All features below are computed and used as **inputs to the XGBoost model**. They are not displayed in the UI but influence forecast accuracy.

### 3.1 Input Requirements

- `timestamp`: UTC timezone-aware datetime
- `demand_mw`: Actual demand (MW)
- 17 weather variables from Open-Meteo (all used as model features)

### 3.2 Derived Features (all used as model inputs)

#### Temperature-Based (3 features)

| Feature | Formula | Purpose |
|---------|---------|---------|
| `cooling_degree_days` | max(0, temp_F - 65) | AC load proxy |
| `heating_degree_days` | max(0, 65 - temp_F) | Heating load proxy |
| `temperature_deviation` | temp_F - rolling_mean(720h) | Anomaly detection |

**Baseline temperature:** 65°F (standard HVAC reference)

#### Time-Based (5 features)

| Feature | Formula | Range |
|---------|---------|-------|
| `hour_sin` | sin(2π × hour / 24) | [-1, 1] |
| `hour_cos` | cos(2π × hour / 24) | [-1, 1] |
| `dow_sin` | sin(2π × dayofweek / 7) | [-1, 1] |
| `dow_cos` | cos(2π × dayofweek / 7) | [-1, 1] |
| `is_weekend` | 1 if Saturday/Sunday | {0, 1} |

*Cyclical encoding preserves proximity (hour 23 is near hour 0).*

#### Lag Features (3 features)

| Feature | Offset | Purpose |
|---------|--------|---------|
| `demand_lag_24h` | t - 24 hours | Same time yesterday |
| `demand_lag_168h` | t - 168 hours | Same time last week |
| `ramp_rate` | demand_t - demand_{t-1} | Hour-over-hour change |

#### Rolling Statistics (12 features)

Three window sizes × four statistics:

| Window | Features |
|--------|----------|
| 24h (1 day) | mean, std, min, max |
| 72h (3 days) | mean, std, min, max |
| 168h (1 week) | mean, std, min, max |

Format: `demand_roll_{window}_{statistic}`

#### Interaction Features (1 feature)

| Feature | Formula | Purpose |
|---------|---------|---------|
| `temp_x_hour` | temperature × hour_sin | Peak temp × peak demand interaction |

### 3.3 Processing Rules

1. **No future leakage:** All features use backward-looking windows only
2. **NaN handling:** Rows with NaN from lag features dropped
3. **Total features:** 17 raw weather + 24 derived = ~41 features for model training

**Additional model input features** (computed but not listed above):
- `wind_power_estimate` - normalized wind power [0,1]
- `solar_capacity_factor` - solar output ratio [0,1]
- `temperature_deviation` - temp vs 30-day rolling average
- `temp_x_hour` - temperature × hour interaction

---

## 4. Forecasting Model

### 4.1 XGBoost (Primary Model)

**Type:** Gradient Boosted Decision Trees
**Library:** `xgboost.XGBRegressor`

**Hyperparameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_estimators` | 500 | Number of trees |
| `max_depth` | 6 | Tree depth limit |
| `learning_rate` | 0.05 | Shrinkage rate |
| `subsample` | 0.8 | Row sampling per tree |
| `colsample_bytree` | 0.8 | Feature sampling per tree |
| `reg_alpha` | 0.1 | L1 regularization |
| `reg_lambda` | 1.0 | L2 regularization |
| `random_state` | 42 | Reproducibility seed |

**Cross-Validation:**
- Method: TimeSeriesSplit (5 folds)
- Constraint: `max(train_idx) < min(val_idx)` (no data leakage)
- Metric: MAPE on each validation fold

**Model Inputs:** All 41 features from Section 3 (17 raw weather + 24 derived)

**Features Excluded:**
- timestamp, region, data_quality (metadata)
- demand_mw (target variable)

**Output:** Predictions clamped to ≥ 0 MW

### 4.2 Backtest Ensemble (Optional)

When "Ensemble" model is selected in Backtest tab, combines:
- XGBoost
- Prophet
- ARIMA

**Weight Calculation:** Inverse-MAPE weighted average
```
weight_i = (1 / MAPE_i) / Σ(1 / MAPE_j)
```

Lower error models get higher weights.

---

## 5. Evaluation Metrics

### 5.1 Metrics Used

| Metric | Formula | Units |
|--------|---------|-------|
| **MAPE** | mean(\|actual - pred\| / \|actual\|) × 100 | % |
| **RMSE** | sqrt(mean((actual - pred)²)) | MW |
| **MAE** | mean(\|actual - pred\|) | MW |
| **R²** | 1 - SS_res / SS_tot | unitless |

### 5.2 Interpretation

| MAPE | Quality |
|------|---------|
| < 3% | Excellent |
| 3-5% | Good |
| 5-10% | Acceptable |
| > 10% | Poor |

---

## 6. Caching Strategy

### 6.1 Cache Configuration

| Data Type | Cache Location | TTL | Notes |
|-----------|----------------|-----|-------|
| News | None | - | Always fetched fresh |
| Demand (EIA) | SQLite | 24 hours | `cache.db` file |
| Weather | SQLite | 24 hours | `cache.db` file |
| Trained Models | In-memory | 24 hours | Per-region |
| Forecast Predictions | In-memory | 24 hours | Per region+horizon |
| Backtest Results | In-memory | 24 hours | Per region+horizon+model |

### 6.2 Cache Invalidation

In-memory caches use data hash for invalidation:
```python
data_hash = hash((len(demand_df), len(weather_df), region))
```

Cache is invalidated when:
1. Data hash changes (new data fetched)
2. TTL expires (24 hours)

### 6.3 Fallback Behavior

On API failure:
1. Serve stale cache if available
2. If no cache, generate demo data
3. Log warning about degraded data

---

## 7. Data Processing Pipeline

### 7.1 Pipeline Steps

```
1. fetch_demand(region)      → demand_df [timestamp, demand_mw, region]
2. fetch_weather(region)     → weather_df [timestamp, 17 weather vars]
3. merge_demand_weather()    → merged_df (left join on timestamp)
4. engineer_features()       → featured_df (~45 columns)
5. train_xgboost()           → model (if not cached)
6. predict_xgboost()         → forecast array
```

### 7.2 Data Merging

- **Join type:** Left join on timestamp
- **Alignment:** Timestamps rounded to nearest hour
- **Duplicates:** Keep last occurrence

### 7.3 Missing Value Handling

| Gap Size | Treatment |
|----------|-----------|
| < 6 hours | Linear interpolation |
| ≥ 6 hours | Flagged as "gap", not interpolated |

### 7.4 Timezone Handling

- All timestamps converted to UTC
- Naive timestamps assumed UTC
- Display may convert to local time

---

## 8. Assumptions & Limitations

### 8.1 Data Assumptions

| Assumption | Implication |
|------------|-------------|
| EIA data is accurate | No independent validation performed |
| Single weather point per region | May not represent large regions well |
| Historical patterns persist | Structural changes may reduce accuracy |
| Hourly granularity | Sub-hourly peaks not captured |

### 8.2 Model Assumptions

| Assumption | Value Used |
|------------|------------|
| Temperature baseline | 65°F for CDD/HDD |

### 8.3 Known Limitations

| Limitation | Impact |
|------------|--------|
| Single weather point per region | Less accurate for large regions (PJM, MISO) |
| FPL ≠ all of Florida | Only ~50% of Florida demand |
| No demand response modeling | DR events not predicted |
| No outage data | Plant outages not factored in |
| Minimum 1-hour data latency | Not real-time |

### 8.4 Capacity Data

- **Source:** EIA-860 (published annually)
- **Update frequency:** Manual
- **Risk:** New plants or retirements not reflected until updated

---

## Appendix A: Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `EIA_API_KEY` | Yes | - | EIA API authentication |
| `NEWS_API_KEY` | No | - | NewsAPI authentication (falls back to demo) |
| `ENVIRONMENT` | No | development | dev/staging/production |
| `PORT` | No | 8080 | Server port |
| `CACHE_DB_PATH` | No | cache.db | SQLite cache file path |
| `CACHE_TTL_SECONDS` | No | 86400 | Cache time-to-live (24h) |

## Appendix B: File Structure

```
energy-forecast/
├── app.py                     # Dash app entry point
├── config.py                  # All constants and configuration
├── components/
│   ├── layout.py              # Dashboard layout (3 tabs)
│   ├── callbacks.py           # All Dash callbacks (21 groups)
│   ├── cards.py               # KPI, welcome, alert card components
│   ├── error_handling.py      # Loading spinners, confidence badges
│   ├── accessibility.py       # Colorblind palette, ARIA helpers
│   ├── tab_forecast.py        # Historical Demand tab
│   ├── tab_demand_outlook.py  # Demand Forecast tab
│   └── tab_backtest.py        # Backtest tab
├── data/
│   ├── cache.py               # SQLite caching with TTL + stale fallback
│   ├── eia_client.py          # EIA API v2 client
│   ├── weather_client.py      # Open-Meteo client (17 variables)
│   ├── noaa_client.py         # NOAA/NWS severe weather alerts
│   ├── news_client.py         # NewsAPI client (energy news feed)
│   ├── preprocessing.py       # Merge, align, interpolate, validate
│   ├── feature_engineering.py # 43 derived features
│   ├── audit.py               # Forecast audit trail (model version, data hash)
│   └── demo_data.py           # Synthetic data generators (offline mode)
├── models/
│   ├── model_service.py       # Forecast service (trained ↔ simulated fallback)
│   ├── xgboost_model.py       # XGBoost training/prediction + SHAP
│   ├── prophet_model.py       # Prophet with weather regressors
│   ├── arima_model.py         # SARIMAX with auto-order selection
│   ├── ensemble.py            # 1/MAPE weighted combination
│   ├── evaluation.py          # MAPE, RMSE, MAE, R²
│   ├── training.py            # Training orchestrator
│   └── pricing.py             # Merit-order pricing model
├── simulation/                # Scenario engine + 6 presets (dormant)
├── personas/                  # 4 role-based persona configurations
├── observability.py           # Pipeline transformation logger
├── tests/                     # 19 test files (unit/integration/e2e)
├── Dockerfile                 # Multi-stage, non-root, healthcheck
└── .github/workflows/         # CI, staging deploy, prod deploy
```

---

*Document Version: Sprint 5*
*Last Updated: 2026-02-22*
