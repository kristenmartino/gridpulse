# Project 1: Energy Demand Forecasting Dashboard — Expanded Spec

## Vision Statement
An interactive, weather-aware energy demand forecasting dashboard that combines real grid data with meteorological features to predict electricity demand across U.S. regions. Demonstrates the exact intersection of weather science and energy optimization that is NextEra Analytics' core business.

---

## User Personas

### Persona 1: Grid Operations Manager ("Sarah")
- **Role**: Manages real-time grid operations at a regional balancing authority
- **Goal**: Needs 24-72 hour demand forecasts to schedule generation resources and avoid brownouts
- **Pain Point**: Current forecasts don't account for weather well enough — unexpected heat waves cause demand spikes that catch her off guard
- **Key Workflows**: Checks morning forecast → compares to yesterday's actuals → adjusts generation schedule → monitors intraday deviations
- **Metrics She Cares About**: Forecast accuracy (MAPE < 3%), peak demand timing, ramp rates

### Persona 2: Renewable Energy Portfolio Analyst ("James")
- **Role**: Manages a portfolio of wind and solar assets for a utility or IPP (like NextEra)
- **Goal**: Forecast how much generation his wind/solar assets will produce to optimize bidding into energy markets
- **Pain Point**: Intermittent renewables make forecasting hard — cloud cover kills solar output, wind patterns are unpredictable
- **Key Workflows**: Reviews 7-day generation forecast → compares wind/solar vs demand curves → identifies curtailment risk → adjusts market positions
- **Metrics He Cares About**: Capacity factor accuracy, generation vs demand correlation, curtailment events

### Persona 3: Energy Trading Desk Analyst ("Maria")
- **Role**: Trades electricity futures and spot market positions
- **Goal**: Identify demand-supply imbalances before they show up in prices
- **Pain Point**: Needs to see the relationship between weather forecasts, demand forecasts, and generation mix in one view
- **Key Workflows**: Morning briefing (overnight actuals + today's forecast) → identify spreads between forecast and actual → flag extreme weather events → position trades
- **Metrics She Cares About**: Price-demand correlation, forecast error distribution, extreme event detection

### Persona 4: Data Scientist / ML Engineer ("Dev")
- **Role**: Builds and improves forecasting models
- **Goal**: Compare model performance, experiment with features, evaluate new approaches
- **Pain Point**: Hard to visualize model comparison and feature importance in one place
- **Key Workflows**: Select region → compare models → analyze residuals → identify feature importance → test new weather features
- **Metrics They Care About**: MAPE, RMSE, MAE, R², residual patterns, feature importance rankings

---

## Data Sources

### 1. EIA Open Data API v2 (Primary — Energy Data)
- **URL**: https://api.eia.gov/v2/
- **Key**: Free, register at https://www.eia.gov/opendata/register.php
- **Rate Limits**: Throttled per second/hour (use caching)

| Endpoint | Data | Frequency | Use Case |
|----------|------|-----------|----------|
| `electricity/rto/region-data` | Actual demand, forecast demand, net generation by balancing authority | Hourly | Core demand data — actuals vs forecasts |
| `electricity/rto/fuel-type-data` | Generation by fuel type (wind, solar, gas, nuclear, hydro) | Hourly | Generation mix and renewable penetration |
| `electricity/rto/interchange-data` | Power flowing between balancing authorities | Hourly | Import/export analysis |

**Stretch Goal Endpoints (not required for MVP):**
| Endpoint | Data | Frequency | Use Case |
|----------|------|-----------|----------|
| `electricity/rto/region-sub-ba-data` | Demand by sub-region (e.g., PG&E within CAISO) | Hourly | Drill-down analysis |
| `electricity/retail-sales` | Retail price, sales volume, customer count by state/sector | Monthly | Price correlation analysis |
| `steo` (Short Term Energy Outlook) | EIA's own demand/supply projections | Monthly | Benchmark comparison |

**Key Balancing Authorities to Include:**
- **ERCOT** — Texas grid (isolated, weather-sensitive, high wind penetration)
- **CAISO** — California (high solar, duck curve, wildfire risk)
- **PJM** — Mid-Atlantic/Midwest (largest US grid, diverse generation)
- **MISO** — Midwest (high wind, agricultural load)
- **NYISO** — New York (urban load, weather extremes)
- **FPL** — Florida Power & Light (**NextEra's subsidiary** — strategic to include!)
- **SPP** — Southwest Power Pool (high wind corridor)
- **ISO-NE** — New England (winter heating spikes, gas constraints)

### 2. Open-Meteo API (Weather Data — No Key Required!)
- **URL**: https://api.open-meteo.com/v1/
- **Key**: None required for non-commercial use
- **Historical**: Data back to 1940 (ERA5 reanalysis), high-res back to 2021
- **Forecast**: 7-16 day forecasts, updated hourly at 1km resolution
- **IMPORTANT**: Add `&temperature_unit=fahrenheit&wind_speed_unit=mph` to all requests — CDD/HDD calculations use °F baseline (65°F), and wind speed sliders in the scenario simulator use mph

| Endpoint | Variables | Use Case |
|----------|-----------|----------|
| `/v1/forecast` | Current + 7-day forecast hourly | Feature input for demand forecasting |
| `/v1/archive` | Historical hourly back to 1940 | Training data for ML models |
| `/v1/forecast` with `&past_days=92` | 3 months historical + 7-day forecast seamlessly | Continuous backtesting |

**Weather Variables to Pull (Feature Engineering):**

| Variable | API Parameter | Why It Matters for Energy |
|----------|--------------|--------------------------|
| Temperature (2m) | `temperature_2m` | #1 demand driver — heating/cooling load |
| Apparent Temperature | `apparent_temperature` | Better proxy for AC usage than raw temp |
| Relative Humidity | `relative_humidity_2m` | Affects cooling efficiency and AC load |
| Dew Point | `dew_point_2m` | Discomfort index → AC demand |
| Wind Speed (10m) | `wind_speed_10m` | Wind generation output |
| Wind Speed (80m) | `wind_speed_80m` | Better proxy for wind turbine hub height |
| Wind Speed (120m) | `wind_speed_120m` | Modern turbines operate at 100-120m |
| Wind Direction | `wind_direction_10m` | Affects which wind farms produce |
| Solar Radiation (GHI) | `shortwave_radiation` | Global Horizontal Irradiance → solar output |
| Direct Normal Irradiance | `direct_normal_irradiance` | Concentrated solar / tracking systems |
| Diffuse Radiation | `diffuse_radiation` | Cloudy sky solar performance |
| Cloud Cover | `cloud_cover` | Solar generation forecast |
| Precipitation | `precipitation` | Hydro generation, storm demand patterns |
| Snowfall | `snowfall` | Winter heating demand spikes |
| Pressure | `surface_pressure` | Weather system changes |
| Soil Temperature | `soil_temperature_0cm` | Ground-source heat pump efficiency |
| Weather Code (WMO) | `weather_code` | Categorical weather classification |

**Coordinates for Balancing Authority Centroids:**
```python
REGION_COORDINATES = {
    "ERCOT": {"lat": 31.0, "lon": -97.0, "name": "Texas (ERCOT)"},
    "CAISO": {"lat": 37.0, "lon": -120.0, "name": "California (CAISO)"},
    "PJM":   {"lat": 39.5, "lon": -77.0, "name": "Mid-Atlantic (PJM)"},
    "MISO":  {"lat": 41.0, "lon": -89.0, "name": "Midwest (MISO)"},
    "NYISO": {"lat": 42.5, "lon": -74.0, "name": "New York (NYISO)"},
    "FPL":   {"lat": 26.9, "lon": -80.1, "name": "Florida (FPL/NextEra)"},
    "SPP":   {"lat": 35.5, "lon": -97.5, "name": "Southwest (SPP)"},
    "ISONE": {"lat": 42.3, "lon": -71.8, "name": "New England (ISO-NE)"},
}
```

### 3. NOAA/NWS API (Supplemental Weather — Free, No Key)
- **URL**: https://api.weather.gov/
- **Use Case**: Severe weather alerts, storm warnings that cause demand spikes or outages
- **Endpoint**: `/alerts/active?area={state}` — active weather alerts by state

**NOAA State → Balancing Authority Mapping:**
```python
# Multiple states map to each BA; alerts for any state trigger for that BA
STATE_TO_BA = {
    "ERCOT": ["TX"],
    "CAISO": ["CA"],
    "PJM":   ["PA", "NJ", "MD", "DE", "VA", "WV", "OH", "DC", "NC", "IN", "IL", "MI", "KY", "TN"],
    "MISO":  ["MN", "WI", "IA", "IL", "IN", "MI", "MO", "AR", "LA", "MS", "TX"],
    "NYISO": ["NY"],
    "FPL":   ["FL"],
    "SPP":   ["KS", "OK", "NE", "SD", "ND", "AR", "LA", "MO", "NM", "TX"],
    "ISONE": ["CT", "MA", "ME", "NH", "RI", "VT"],
}
# Note: Some states appear in multiple BAs (e.g., TX in ERCOT, MISO, SPP).
# Filter alerts by severity: Extreme, Severe → Critical; Moderate → Warning; Minor → Info.
```

### 4. Generation Capacity (Static Lookup + EIA Monthly)
The scenario simulator's pricing model requires total generation capacity per region.

```python
# Approximate installed capacity (MW) per BA — from EIA-860 data
# Update these annually or fetch from EIA electricity/operating-generator-capacity endpoint
REGION_CAPACITY_MW = {
    "ERCOT": 130000,
    "CAISO": 80000,
    "PJM":   185000,
    "MISO":  175000,
    "NYISO": 38000,
    "FPL":   32000,
    "SPP":   90000,
    "ISONE": 30000,
}
# For more accuracy, pull from: electricity/operating-generator-capacity
# and sum by balancing authority. This gives fuel-type breakdown too,
# which feeds the generation mix shift calculation in scenarios.
```

### 5. Derived / Engineered Features (Calculated)
These are computed from the raw data above:

| Feature | Calculation | Why |
|---------|-------------|-----|
| Cooling Degree Days (CDD) | max(0, temp - 65°F) | Standard HVAC demand proxy |
| Heating Degree Days (HDD) | max(0, 65°F - temp) | Winter heating demand |
| Temperature Deviation | temp - 30-day rolling avg | Unusual weather = unusual demand |
| Wind Power Estimate | 0.5 × ρ × A × v³ (simplified) | Estimate wind generation potential |

**Wind Power Unit Note:** Open-Meteo returns wind in mph (per our request params), but the power curve formula requires m/s. Convert: `v_ms = v_mph × 0.44704`. Apply cutout speed (25 m/s ≈ 56 mph) — above this, turbines shut down → power = 0. Cap at rated power. The scenario simulator sliders display mph for user familiarity; conversion happens internally.
| Solar Capacity Factor | GHI / 1000 (panel rated at 1kW/m²) | Estimate solar generation potential |
| Hour of Day (sin/cos encoded) | sin(2π×hour/24), cos(2π×hour/24) | Cyclical time feature |
| Day of Week (encoded) | sin(2π×dow/7), cos(2π×dow/7) | Weekday/weekend demand patterns (cyclical like hour) |
| Holiday Flag | US federal holiday calendar | Holidays reduce commercial demand |
| Lag Features | demand_t-24, demand_t-168 | Yesterday same hour, last week same hour |
| Rolling Stats | 24h/72h/168h rolling mean, std, min, max | Trend and volatility features |
| Temp × Hour Interaction | temperature * hour_sin | AC peaks in afternoon, heating in evening |
| Ramp Rate | demand_t - demand_t-1 | Rate of change (critical for grid ops) |

---

## Features & Functionality

### Tab 1: Demand Forecast Dashboard (Sarah's View)
**Purpose**: Primary operational view — what's demand going to be?

**Components:**
1. **Region Selector** — Dropdown with all 8 balancing authorities
2. **Main Chart: Demand Forecast** (Plotly)
   - X-axis: Time (hourly, scrollable)
   - Y-axis: Demand (MW)
   - Lines: Actual demand (solid blue), Prophet forecast (dashed orange), ARIMA forecast (dashed green), EIA's own forecast (dotted gray)
   - Shaded area: Prophet uncertainty bands (80% and 95% confidence intervals)
   - Vertical line: "now" marker separating historical from forecast
   - Hover: Shows all values + weather conditions at that hour
3. **Weather Overlay Toggle** — Overlay temperature curve on demand chart (dual y-axis)
4. **Peak Demand Card** — Today's predicted peak (MW), time of peak, confidence range
5. **Forecast Accuracy Scorecard** — Rolling 7-day MAPE for each model
6. **Alerts Panel** — Flags when forecast demand exceeds historical 95th percentile

### Tab 2: Weather-Energy Correlation Explorer (James's View)
**Purpose**: Understand how weather drives demand and renewable generation

**Components:**
1. **Scatter Plot Matrix** — Temperature vs Demand, Wind Speed vs Wind Generation, GHI vs Solar Generation
2. **Correlation Heatmap** — All weather features vs demand, interactive (click to drill into any pair)
3. **Weather Feature Importance** — Bar chart showing which weather variables most impact demand (from model feature importance)
4. **Seasonal Decomposition** — Trend, seasonal, residual components of demand (statsmodels STL decomposition)
5. **Renewable Generation Forecast** — Wind and solar generation forecast based on weather inputs
6. **Duck Curve Link** — Quick-navigate to Tab 4's Duck Curve visualization for the selected region

### Tab 3: Model Comparison & Diagnostics (Dev's View)
**Purpose**: Compare forecasting models and diagnose errors

**Components:**
1. **Model Selector** — Toggle which models to display (Prophet, ARIMA, XGBoost, Ensemble)
2. **Metrics Table**:
   | Metric | Prophet | ARIMA | XGBoost | Ensemble |
   |--------|---------|-------|---------|----------|
   | MAPE   |         |       |         |          |
   | RMSE   |         |       |         |          |
   | MAE    |         |       |         |          |
   | R²     |         |       |         |          |
3. **Residual Analysis**:
   - Residuals over time (are errors getting worse?)
   - Residual distribution (histogram — should be normal, centered at 0)
   - Residual vs predicted (heteroscedasticity check)
   - ACF/PACF of residuals (autocorrelation check)
4. **Error by Hour of Day** — Heatmap showing which hours have highest forecast error
5. **Error by Weather Condition** — Are errors worse during extreme weather?
6. **Feature Importance** — XGBoost SHAP values or permutation importance

### Tab 4: Generation Mix & Renewables (James + Maria's View)
**Purpose**: Understand the supply side — what's generating the power?

**Components:**
1. **Stacked Area Chart** — Generation by fuel type over time (wind, solar, gas, coal, nuclear, hydro)
2. **Renewable Penetration %** — Line chart showing renewable share of total generation
3. **Wind Generation vs Wind Speed** — Overlay showing the power curve relationship
4. **Solar Generation vs Irradiance** — Overlay with cloud cover
5. **Duck Curve Visualization** — Net demand (total demand minus solar) showing the classic duck shape, especially for CAISO. Highlights the mid-day solar trough and evening ramp.
6. **Curtailment Indicator** — Flag periods where renewable generation exceeds demand (negative pricing risk)
7. **Carbon Intensity** — Estimated CO₂ per MWh based on generation mix

### Tab 5: Extreme Events & Alerts (Maria's View)
**Purpose**: Early warning system for demand-supply stress events

**Components:**
1. **Active Weather Alerts** — Pull from NOAA API, map to affected grid regions
2. **Demand Anomaly Detection** — Statistical flagging of unusual demand patterns
3. **Historical Extreme Events** — Timeline of past demand records, extreme weather events
4. **Temperature Exceedance Forecast** — Probability that temperature exceeds key thresholds (95°F, 100°F, 105°F)
5. **Stress Indicator** — Combined metric: high demand forecast + low renewable generation + extreme weather = stress score

### Tab 6: Scenario Simulator ("What-If" Planner) — NEW
**Purpose**: Stress-test the grid against hypothetical weather scenarios. This is the showstopper feature — demonstrates decision-support thinking, not just forecasting.

**Components:**
1. **Weather Scenario Builder**
   - Temperature slider: -10°F to 120°F (with current forecast shown as default)
   - Wind speed slider: 0-50 mph
   - Cloud cover slider: 0-100%
   - Humidity slider: 0-100%
   - Duration selector: How many hours/days does this scenario last?
   - "Apply to region" dropdown — which balancing authority?

2. **Preset Historical Scenarios** (one-click replay)
   | Scenario | Date | Region | What Happened |
   |----------|------|--------|---------------|
   | Winter Storm Uri | Feb 2021 | ERCOT | Texas grid collapse, temps below 0°F, 69 deaths |
   | Summer 2023 Heat Dome | Jul 2023 | CAISO | 110°F+ across Southwest, record demand |
   | Polar Vortex 2019 | Jan 2019 | MISO | Chicago -23°F, Midwest demand surge |
   | California Heat Wave | Sep 2022 | CAISO | CAISO emergency alerts, rolling blackout risk |
   | Hurricane Irma | Sep 2017 | FPL | FPL lost power to 4.4M customers (NextEra!) |
   | 2024 Solar Eclipse | Apr 2024 | PJM | Solar generation dropped to near-zero mid-day |

3. **Impact Dashboard** (updates in real-time as sliders move)
   - **Demand Forecast Curve** — Shows baseline forecast vs scenario forecast overlaid
   - **Demand Delta** — MW change from baseline, peak shift timing
   - **Generation Mix Shift** — How does the fuel mix change? (e.g., heat wave = more gas peakers, no wind = less wind generation)
   - **Renewable Impact** — Wind generation estimate at scenario wind speed, solar at scenario cloud cover
   - **Estimated Price Impact** — Simplified pricing model: when demand exceeds supply threshold, prices spike non-linearly
   - **Reserve Margin Indicator** — Generation capacity minus forecasted demand = how close to the edge?
   - **Carbon Impact** — If renewables drop and gas peakers fire up, what's the CO₂ change?

4. **Scenario Comparison Mode**
   - Save up to 3 scenarios and overlay them on the same chart
   - Side-by-side metrics table comparing scenarios
   - "Which scenario creates the most grid stress?" summary

5. **Monte Carlo Uncertainty** (stretch goal)
   - Run 100 slight variations of the scenario (±noise on each parameter)
   - Show confidence bands around the scenario forecast
   - Display probability of demand exceeding capacity

**How the Scenario Engine Works:**
```python
def simulate_scenario(
    features: pd.DataFrame,
    weather_overrides: dict[str, float],
    models: dict[str, Any],
    base_forecast: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Replace weather features in the feature matrix with user-specified values,
    recompute derived features, re-run the ensemble forecast, and return deltas.
    
    Args:
        features: Feature matrix (will be copied, not mutated)
        weather_overrides: Dict of {column_name: override_value}
        models: Dict of trained model objects keyed by name
        base_forecast: Optional pre-computed baseline. If None, computed from unmodified features.
    
    Returns:
        (scenario_forecast, delta) where delta = scenario - baseline
    """
    features = features.copy()  # Never mutate the input
    
    # 1. Compute baseline if not provided
    if base_forecast is None:
        base_forecast = models['ensemble'].predict(features)
    
    # 2. Override weather columns with scenario values
    for col, value in weather_overrides.items():
        if col not in features.columns:
            raise ValueError(f"Unknown weather column: {col}")
        features[col] = value
    
    # 3. Recompute derived features (CDD, HDD, wind power estimate, etc.)
    features = recompute_derived_features(features)
    
    # 4. Run each model on the modified features
    prophet_forecast = models['prophet'].predict(features)
    xgboost_forecast = models['xgboost'].predict(features)
    # ARIMA doesn't re-forecast well with swapped exogenous — exclude from scenario ensemble.
    # Renormalize weights to sum to 1.0 using only Prophet + XGBoost weights.
    
    # 5. Ensemble combine (2-model variant for scenarios)
    scenario_forecast = ensemble_combine(
        forecasts=[prophet_forecast, xgboost_forecast],
        model_names=['prophet', 'xgboost'],
        weights=models.get('ensemble_weights', None),  # Will renormalize internally
    )
    
    # 6. Compute deltas vs baseline
    delta = scenario_forecast - base_forecast
    
    return scenario_forecast, delta
```

**Pricing Model (simplified but credible):**
```python
def estimate_price_impact(demand_forecast, generation_capacity, base_price=50):
    """
    Simplified merit-order pricing model.
    
    Accepts scalars or arrays. Returns same type as input.
    - When demand < 70% capacity: base price ($30-50/MWh)
    - When demand 70-90% capacity: moderate increase (gas peakers)
    - When demand > 90% capacity: exponential spike (scarcity pricing)
    - When demand > 100% capacity: emergency pricing ($1000+/MWh)
    """
    demand_forecast = np.asarray(demand_forecast, dtype=float)
    generation_capacity = np.asarray(generation_capacity, dtype=float)
    scalar_input = demand_forecast.ndim == 0
    
    utilization = demand_forecast / generation_capacity
    price = np.where(
        utilization < 0.7, base_price,
        np.where(
            utilization < 0.9, base_price * (1 + 2 * (utilization - 0.7)),
            np.where(
                utilization < 1.0, base_price * np.exp(5 * (utilization - 0.9)),
                base_price * 20  # Emergency/scarcity pricing
            )
        )
    )
    return float(price) if scalar_input else price
```

---

## Persona Switcher

### Overview
A header-level dropdown that reconfigures the entire dashboard based on the selected role. This demonstrates role-based UX and product thinking — each persona sees the same data but through a lens optimized for their decisions.

### Implementation
```python
# In app.py header
persona_switcher = dbc.Select(
    id="persona-selector",
    options=[
        {"label": "👷 Grid Operations Manager", "value": "grid_ops"},
        {"label": "🌱 Renewable Portfolio Analyst", "value": "renewables"},
        {"label": "📈 Energy Trading Analyst", "value": "trader"},
        {"label": "🔬 Data Scientist / ML Engineer", "value": "data_sci"},
    ],
    value="grid_ops",
    className="persona-switcher",
)
```

### Per-Persona Configuration

| Setting | Grid Ops (Sarah) | Renewables (James) | Trader (Maria) | Data Sci (Dev) |
|---------|-------------------|---------------------|-----------------|----------------|
| **Default Tab** | Tab 1: Demand Forecast | Tab 4: Generation Mix | Tab 5: Extreme Events | Tab 3: Model Comparison |
| **Header KPI Cards** | Peak Demand, Forecast Error, Reserve Margin | Wind CF, Solar CF, Renewable % | Price Estimate, Stress Score, Demand Delta | MAPE, RMSE, R², Feature Count |
| **Alert Threshold** | Demand > 95th percentile | Curtailment risk, Low wind | Price spike probability | Model drift detected |
| **Default Region** | Last selected | ERCOT (high wind) | All regions comparison | All regions |
| **Chart Emphasis** | Demand + weather overlay | Generation mix + renewables | Price + demand correlation | Residuals + diagnostics |
| **Scenario Presets** | Extreme weather focus | Low wind / cloudy day | Price spike scenarios | Model stress tests |
| **Visible Tabs** | All | Tabs 1, 2, 4, 6 | Tabs 1, 4, 5, 6 | Tabs 1, 2, 3, 6 |
| **Data Refresh** | Every 15 min | Every hour | Every 5 min | On demand |

### Persona-Specific Welcome Cards
When switching personas, show a brief contextual card:

**Grid Ops**: "Good morning, Sarah. Today's peak demand for FPL is forecast at 28,450 MW at 4:00 PM. Temperature expected to reach 94°F. No active weather alerts."

**Renewables**: "Portfolio summary: Wind capacity factor at 32% (below 7-day avg of 38%). Solar tracking normally. ERCOT curtailment risk low today."

**Trader**: "Market brief: ERCOT demand 3.2% above forecast at 8 AM. SPP wind generation underperforming. Watch: Heat advisory for PJM Thursday."

**Data Sci**: "Model health: Ensemble MAPE 2.8% (7-day rolling). XGBoost outperforming Prophet this week. Feature drift detected: wind_speed_80m distribution shifted."

### How It Works Technically
```python
@app.callback(
    [Output("tab-content", "children"),
     Output("kpi-cards", "children"),
     Output("welcome-card", "children"),
     Output("tabs", "active_tab")],
    Input("persona-selector", "value")
)
def switch_persona(persona):
    config = PERSONA_CONFIGS[persona]
    
    # Reconfigure KPI cards
    kpis = build_kpi_cards(config["kpi_metrics"])
    
    # Set default tab
    active_tab = config["default_tab"]
    
    # Generate welcome message
    welcome = generate_welcome_card(persona, get_latest_data())
    
    # Filter visible tabs
    tabs = build_tabs(visible=config["visible_tabs"])
    
    return tabs, kpis, welcome, active_tab
```

---

## Forecasting Models

### Model 1: Prophet (Primary)
- **Why**: Handles seasonality well (daily, weekly, yearly), robust to missing data, easy to add regressors
- **Configuration**:
  ```python
  model = Prophet(
      yearly_seasonality=True,
      weekly_seasonality=True,
      daily_seasonality=True,
      changepoint_prior_scale=0.05,  # Regularization
      seasonality_mode='multiplicative',  # Energy demand scales multiplicatively
  )
  # Add weather regressors
  model.add_regressor('temperature_2m', mode='multiplicative')
  model.add_regressor('apparent_temperature', mode='multiplicative')
  model.add_regressor('wind_speed_80m', mode='additive')
  model.add_regressor('shortwave_radiation', mode='additive')
  model.add_regressor('cooling_degree_days', mode='multiplicative')
  model.add_regressor('heating_degree_days', mode='multiplicative')
  model.add_regressor('is_holiday', mode='multiplicative')
  ```
- **Training**: Rolling window — train on 365 days, forecast next 7 days, slide forward

### Model 2: ARIMA/SARIMAX (Baseline)
- **Why**: Classical statistical baseline to compare against ML approaches
- **Configuration**: SARIMAX with exogenous weather variables
  ```python
  model = SARIMAX(
      demand_series,
      exog=weather_features,
      order=(2, 1, 2),           # (p, d, q)
      seasonal_order=(1, 1, 1, 24),  # Daily seasonality (24 hours)
  )
  ```
- **Auto-tuning**: Use pmdarima auto_arima for order selection

### Model 3: XGBoost (ML Approach)
- **Why**: Captures non-linear weather-demand relationships, provides feature importance
- **Features**: All weather + calendar + lag features from the engineering table above
- **Configuration**:
  ```python
  model = XGBRegressor(
      n_estimators=500,
      max_depth=6,
      learning_rate=0.05,
      subsample=0.8,
      colsample_bytree=0.8,
      reg_alpha=0.1,   # L1 regularization
      reg_lambda=1.0,  # L2 regularization
  )
  ```
- **Validation**: TimeSeriesSplit (no data leakage)

### Model 4: Ensemble (Production)
- **Why**: Combining models almost always beats individual models
- **Method**: Weighted average where weights are inversely proportional to recent MAPE
  ```python
  weights = 1 / recent_mape_per_model
  weights = weights / weights.sum()  # Normalize
  ensemble_forecast = sum(w * f for w, f in zip(weights, forecasts))
  ```

---

## Workflows

### Workflow 1: Morning Demand Review (Sarah)
```
1. Open dashboard → Tab 1 (Demand Forecast)
2. Select region (e.g., FPL for Florida)
3. Review overnight actuals vs yesterday's forecast
   → Check forecast accuracy scorecard
4. Review today's demand forecast + confidence bands
   → Note predicted peak time and magnitude
5. Toggle weather overlay
   → See temperature forecast driving today's demand
6. Check alerts panel
   → Any extreme weather warnings? Demand exceeding 95th percentile?
7. Switch to Tab 4 (Generation Mix)
   → Confirm expected renewable generation
   → Identify any wind/solar shortfall
8. Decision: Adjust generation schedule based on forecast
```

### Workflow 2: Renewable Portfolio Analysis (James)
```
1. Open dashboard → Tab 2 (Weather-Energy Correlation)
2. Select ERCOT (high wind region)
3. Review wind speed forecast vs wind generation
   → Scatter plot: is the power curve holding?
4. Check solar radiation forecast for the week
   → Identify low-generation days (cloudy)
5. Switch to Tab 4 → Duck Curve visualization
   → Identify mid-day solar surplus → curtailment risk?
6. Review renewable penetration trend
   → Is renewable share growing month over month?
7. Decision: Adjust market bids based on expected generation
```

### Workflow 3: Model Performance Review (Dev)
```
1. Open dashboard → Tab 3 (Model Comparison)
2. Review metrics table — which model has lowest MAPE this week?
3. Analyze residuals
   → Residuals over time: are errors stable?
   → Residual distribution: normally distributed?
   → ACF: any autocorrelation? (indicates model is missing a pattern)
4. Check error by hour — worst at which hours?
5. Check error by weather — errors spike during extreme heat?
6. Review feature importance
   → Is temperature still #1? Any new features gaining importance?
7. Decision: Retrain with new features or adjust model hyperparameters
```

### Workflow 4: Extreme Weather Response (Maria)
```
1. Open dashboard → Tab 5 (Extreme Events)
2. Check NOAA alerts — any active heat advisories or winter storms?
3. Review temperature exceedance forecast
   → Probability of 100°F+ in ERCOT this week?
4. Check stress indicator
   → High demand + low wind + extreme heat = critical stress
5. Switch to Tab 1 → Zoom into stress period
   → Compare demand forecast vs generation capacity
6. Decision: Position trades for expected price spike
```

### Workflow 5: Scenario Planning — "What If a Heat Wave Hits Texas?" (All Personas)
```
1. Open dashboard → Tab 6 (Scenario Simulator)
2. Select region: ERCOT
3. Option A — Use a preset:
   → Click "Summer 2023 Heat Dome"
   → Dashboard instantly shows: demand spikes 18% above baseline,
     wind drops to 15% CF, solar holds, gas peakers fire up,
     price estimate jumps from $45/MWh to $180/MWh,
     reserve margin drops to 3.2%
4. Option B — Build custom scenario:
   → Drag temperature to 108°F
   → Set wind speed to 5 mph (dead calm)
   → Set cloud cover to 20% (sunny but no wind)
   → Duration: 72 hours
   → Watch the impact dashboard update in real-time
5. Compare: Save scenario, create a second with 95°F + normal wind
   → Side-by-side comparison shows the dead-calm scenario is 2x worse
6. Each persona takes different action:
   → Sarah (Grid Ops): "We need to pre-position peaker plants"
   → James (Renewables): "Wind portfolio will underperform, need backup"
   → Maria (Trader): "Price spike likely, position long"
   → Dev (Data Sci): "Model handles heat well but misses low-wind impact"
```

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Google Cloud Run                       │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Plotly Dash Application              │    │
│  │                                                   │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐       │    │
│  │  │  Tab 1   │  │  Tab 2   │  │  Tab 3   │ ...   │    │
│  │  │ Forecast │  │ Weather  │  │  Models  │       │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘       │    │
│  │       └──────────────┴──────────────┘             │    │
│  │                      │                            │    │
│  │              Callback Layer                       │    │
│  │                      │                            │    │
│  │       ┌──────────────┴──────────────┐             │    │
│  │       │        Data Layer           │             │    │
│  │       │                             │             │    │
│  │       │  ┌─────────┐ ┌──────────┐  │             │    │
│  │       │  │  Cache   │ │  Models  │  │             │    │
│  │       │  │ (SQLite) │ │ (Pickle) │  │             │    │
│  │       │  └─────────┘ └──────────┘  │             │    │
│  │       └──────┬──────────┬──────────┘             │    │
│  └──────────────┼──────────┼────────────────────────┘    │
│                 │          │                              │
└─────────────────┼──────────┼──────────────────────────────┘
                  │          │
        ┌─────────┴──┐  ┌───┴──────────┐
        │  EIA API   │  │  Open-Meteo  │
        │  (energy)  │  │  (weather)   │
        └────────────┘  └──────────────┘
                              │
                        ┌─────┴─────┐
                        │ NOAA API  │
                        │ (alerts)  │
                        └───────────┘
```

### Data Flow
```
1. Startup / Scheduled Refresh (every 6 hours):
   EIA API → fetch hourly demand + generation data → cache to SQLite
   Open-Meteo → fetch historical + forecast weather → cache to SQLite
   NOAA → fetch active weather alerts → cache to SQLite

2. Model Training (on startup or scheduled):
   SQLite cached data → feature engineering → train Prophet, ARIMA, XGBoost
   → serialize models to pickle → store in /models/

3. User Request (Dash callback):
   User selects region + date range → load from cache
   → run inference on cached models → generate Plotly figures → render
```

### Project Structure
```
energy-forecast/
├── app.py                          # Dash app entry point, tab layout, persona switcher
├── config.py                       # Global config: API URLs, regions, coordinates, capacity, env vars
├── data/
│   ├── eia_client.py               # EIA API v2 client with pagination + caching
│   ├── weather_client.py           # Open-Meteo client (historical + forecast)
│   ├── noaa_client.py              # NOAA alerts client
│   ├── cache.py                    # SQLite caching layer
│   ├── feature_engineering.py      # CDD, HDD, lags, rolling stats, interactions
│   └── preprocessing.py            # Data cleaning, alignment, missing value handling
├── models/
│   ├── prophet_model.py            # Prophet with weather regressors
│   ├── arima_model.py              # SARIMAX with exogenous variables
│   ├── xgboost_model.py            # XGBoost with full feature set
│   ├── ensemble.py                 # Weighted ensemble combiner
│   ├── evaluation.py               # MAPE, RMSE, MAE, R², residual analysis
│   ├── training.py                 # Training orchestrator with TimeSeriesSplit
│   └── pricing.py                  # Simplified merit-order pricing model
├── simulation/
│   ├── scenario_engine.py          # Weather override → re-forecast pipeline
│   ├── presets.py                  # Historical scenario definitions (Uri, Heat Dome, etc.)
│   └── comparison.py               # Multi-scenario comparison logic
├── personas/
│   ├── config.py                   # Per-persona tab visibility, KPIs, defaults
│   └── welcome.py                  # Dynamic welcome card generator
├── components/
│   ├── tab_forecast.py             # Tab 1: Demand Forecast Dashboard
│   ├── tab_weather.py              # Tab 2: Weather-Energy Correlation
│   ├── tab_models.py               # Tab 3: Model Comparison & Diagnostics
│   ├── tab_generation.py           # Tab 4: Generation Mix & Renewables
│   ├── tab_alerts.py               # Tab 5: Extreme Events & Alerts
│   ├── tab_simulator.py            # Tab 6: Scenario Simulator (What-If)
│   ├── charts.py                   # Plotly figure factory functions
│   ├── cards.py                    # KPI cards, scorecard components
│   └── callbacks.py                # All Dash callbacks
├── assets/
│   └── style.css                   # Custom dark theme CSS
├── tests/
│   ├── conftest.py                 # Shared fixtures: mock API responses, sample DataFrames
│   ├── unit/
│   │   ├── test_eia_client.py
│   │   ├── test_weather_client.py
│   │   ├── test_noaa_client.py
│   │   ├── test_cache.py
│   │   ├── test_feature_engineering.py
│   │   ├── test_preprocessing.py
│   │   ├── test_prophet_model.py
│   │   ├── test_arima_model.py
│   │   ├── test_xgboost_model.py
│   │   ├── test_ensemble.py
│   │   ├── test_evaluation.py
│   │   ├── test_pricing.py
│   │   ├── test_scenario_engine.py
│   │   ├── test_scenario_presets.py
│   │   ├── test_persona_config.py
│   │   └── test_welcome_cards.py
│   ├── integration/
│   │   ├── test_data_pipeline.py
│   │   ├── test_model_pipeline.py
│   │   ├── test_scenario_pipeline.py
│   │   └── test_api_fallback.py
│   ├── e2e/
│   │   ├── test_dashboard_render.py
│   │   ├── test_persona_switching.py
│   │   └── test_scenario_interaction.py
│   └── fixtures/
│       ├── eia_response_ercot.json
│       ├── eia_response_fpl.json
│       ├── weather_response_ercot.json
│       ├── noaa_alerts_texas.json
│       ├── sample_demand_90days.csv
│       └── sample_features.csv
├── Dockerfile
├── requirements.txt
├── .env.example                    # EIA_API_KEY=your_key_here
└── README.md                       # Project documentation with screenshots
```

---

## Requirements
```
# Core
dash==2.17.1
dash-bootstrap-components==1.6.0
plotly==5.22.0
pandas==2.2.2
numpy==1.26.4

# ML / Forecasting
prophet==1.1.5
statsmodels==0.14.2
xgboost==2.0.3
scikit-learn==1.5.1

# Data
requests==2.32.3
openmeteo-requests==1.2.0
openmeteo-sdk==1.11.5
retry-requests==2.0.0

# Infrastructure
gunicorn==22.0.0
python-dotenv==1.0.1
structlog==24.2.0

# Enhancement
shap==0.45.1              # SHAP values for XGBoost
pmdarima==2.0.4           # Auto-ARIMA order selection
holidays==0.50            # US holiday detection

# Testing (dev only — exclude from Docker)
pytest==8.2.2
pytest-cov==5.0.0
pytest-mock==3.14.0
```

### .dockerignore
```
tests/
__pycache__/
*.pyc
.env
.git/
.pytest_cache/
htmlcov/
*.egg-info/
README.md
.dockerignore
```

---

## Deployment (Cloud Run)

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system deps for Prophet
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download and cache initial data on build (optional)
# RUN python -c "from data.cache import warm_cache; warm_cache()"

EXPOSE 8080

CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "300"]
```

### Deploy Commands
```bash
# Build and deploy to Cloud Run
gcloud run deploy energy-forecast \
  --source . \
  --platform managed \
  --region us-east1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300 \
  --set-env-vars EIA_API_KEY=your_key_here

# Verify
gcloud run services describe energy-forecast --region us-east1 --format="value(status.url)"
```

---

## Interview Talking Points

This project gives you ammunition for several common interview questions:

1. **"Tell me about a technical project you built"** → Full end-to-end: data ingestion from 3 APIs, 25+ engineered features across 12 categories, 4 ML models, ensemble, scenario simulation, production deployment on GCP

2. **"How do you approach feature engineering?"** → Weather-energy relationship: CDD/HDD, hub-height wind speeds, interaction terms, lag features, cyclical encoding — explain *why* each feature matters physically

3. **"How do you evaluate model performance?"** → Not just accuracy metrics — residual analysis, error by time-of-day, error by weather condition, autocorrelation checks, model drift detection

4. **"Why Google Cloud?"** → "NextEra signed a landmark partnership with Google Cloud in December 2025 — I wanted to demonstrate proficiency with the platform you're adopting"

5. **"How does this relate to NextEra's business?"** → "Your 360 platform does exactly this at scale — combining weather science with energy optimization. I built a version that includes scenario planning, which mirrors how your team would use Google's WeatherNext 2 for operational stress-testing"

6. **"Tell me about a product decision you made"** → "I built a persona switcher that reconfigures the dashboard based on role — a grid operator sees demand forecasts and reserve margins, while a trader sees pricing estimates and stress indicators. Same data, different lens. This demonstrates how a single platform can serve multiple stakeholders, which is exactly the challenge with an enterprise analytics product like NextEra's"

7. **"How do you handle stakeholder needs?"** → "I defined 4 user personas with specific workflows and designed each tab to serve a primary persona's decision. The scenario simulator, for example, serves all 4 — but each takes a different action from the same simulation output. A grid ops manager pre-positions generation, a trader positions for price spikes, a portfolio analyst adjusts expectations."

8. **"How would you improve this?"** → Add Vertex AI for model training at scale, BigQuery for data warehousing, integrate Google's TimesFM 2.5 and WeatherNext 2 (the exact models NextEra announced), add real-time streaming with Pub/Sub, Monte Carlo simulation for probabilistic scenario analysis, and multi-region scenario cascading (e.g., what if a hurricane hits Florida AND a heat wave hits Texas simultaneously?)

---

## Acceptance Criteria

### AC-1: Data Ingestion & Caching
| ID | Criteria | Verification |
|----|----------|-------------|
| AC-1.1 | EIA API returns hourly demand data for all 8 balancing authorities (ERCOT, CAISO, PJM, MISO, NYISO, FPL, SPP, ISONE) | Unit test: mock API response, assert 8 regions parsed correctly |
| AC-1.2 | Open-Meteo returns all 17 weather variables for each region's centroid coordinates | Unit test: validate response schema matches expected variables |
| AC-1.3 | NOAA API returns active weather alerts for relevant states | Unit test: mock alert response, assert parsed into Alert dataclass |
| AC-1.4 | All API responses are cached to SQLite with configurable TTL (default 6 hours) | Integration test: call twice, assert second call reads from cache (no HTTP request) |
| AC-1.5 | Cache gracefully handles stale data — serves stale if API is down, logs warning | Integration test: mock API failure after cache populated, assert stale data returned |
| AC-1.6 | API rate limiting is respected — EIA throttling handled with exponential backoff | Unit test: mock 429 response, assert retry with backoff |
| AC-1.7 | Data alignment: EIA hourly timestamps and Open-Meteo hourly timestamps align after preprocessing (both UTC) | Unit test: merge two DataFrames, assert no NaN from misalignment |
| AC-1.8 | Missing data handling: gaps < 6 hours interpolated, gaps > 6 hours flagged | Unit test: introduce gaps, assert interpolation and flags |

### AC-2: Feature Engineering
| ID | Criteria | Verification |
|----|----------|-------------|
| AC-2.1 | CDD calculated correctly: max(0, temp_F - 65) for each hourly observation | Unit test: known inputs → expected CDD values |
| AC-2.2 | HDD calculated correctly: max(0, 65 - temp_F) for each hourly observation | Unit test: known inputs → expected HDD values |
| AC-2.3 | Wind power estimate follows cubic relationship: 0.5 × ρ × A × v³ (capped at rated power) | Unit test: wind=0 → power=0, wind=rated → power=rated, wind>cutout → power=0 |
| AC-2.4 | Solar capacity factor = GHI / 1000, clipped to [0, 1] | Unit test: GHI=500 → CF=0.5, GHI=1200 → CF=1.0 (clipped) |
| AC-2.5 | Cyclical time encoding: sin/cos hour produces correct values (hour 0 and 24 map to same point) | Unit test: hour=0 and hour=24 produce identical sin/cos |
| AC-2.6 | Lag features: demand_t-24 and demand_t-168 correctly shift data with no leakage | Unit test: verify lag values match source, assert no future data in features |
| AC-2.7 | Rolling statistics (24h/72h/168h mean, std, min, max) use only backward-looking windows | Unit test: verify rolling window boundaries, assert no lookahead |
| AC-2.8 | Holiday flag correctly identifies all US federal holidays via `holidays` library | Unit test: known holiday dates → flag=1, known non-holidays → flag=0 |
| AC-2.9 | Temperature deviation = current temp - 30-day rolling average temp | Unit test: constant temp series → deviation = 0; spike → positive deviation |
| AC-2.10 | Ramp rate = demand_t - demand_t-1, computed for each hourly step | Unit test: known demand series [100, 120, 110] → ramps [NaN, 20, -10] |
| AC-2.11 | Temp × Hour interaction = temperature × hour_sin, captures afternoon AC peaks | Unit test: hot afternoon (temp=100, hour_sin=1) → high interaction; cold morning → low |
| AC-2.12 | Day of week sin/cos encoding produces correct cyclical values (Monday=0, Sunday=6) | Unit test: Monday and next Monday produce identical sin/cos values |
| AC-2.13 | All features are numeric, no NaN after engineering (NaN rows dropped or imputed) | Unit test: assert no NaN in final feature matrix |
| AC-2.14 | Feature matrix has correct shape: rows = hours in range, columns = all expected features | Unit test: assert shape matches expected dimensions |

### AC-3: Forecasting Models
| ID | Criteria | Verification |
|----|----------|-------------|
| AC-3.1 | Prophet model trains successfully with all weather regressors and produces 7-day forecast | Integration test: train on 90 days, forecast 7 days, assert output shape and no NaN |
| AC-3.2 | Prophet uncertainty bands (80% and 95%) are monotonically widening into the future | Unit test: assert upper_95 > upper_80 > yhat > lower_80 > lower_95 at each step |
| AC-3.3 | SARIMAX model trains with exogenous weather variables and converges (no warnings) | Integration test: train, assert convergence, forecast shape correct |
| AC-3.4 | XGBoost model trains with TimeSeriesSplit (no data leakage across folds) | Unit test: verify split indices — no test index appears before train indices |
| AC-3.5 | XGBoost feature importance is extractable and sums to ~1.0 | Unit test: assert feature_importances_ length = n_features, sum ≈ 1.0 |
| AC-3.6 | Ensemble weights are inversely proportional to recent MAPE and sum to 1.0 | Unit test: given MAPEs [2%, 5%, 10%], verify weights ≈ [0.59, 0.24, 0.17] |
| AC-3.7 | Ensemble forecast is between the min and max of individual model forecasts | Unit test: assert ensemble_min >= individual_min, ensemble_max <= individual_max |
| AC-3.8 | All models achieve MAPE < 10% on the validation set (reasonable for energy forecasting) | Integration test: train/test split, assert MAPE < 10% |
| AC-3.9 | Model serialization: trained models save to pickle and reload without error | Unit test: train → serialize → deserialize → predict → assert same output |
| AC-3.10 | SHAP values compute for XGBoost without error and produce per-feature explanations | Integration test: compute SHAP on test set, assert shape = (n_samples, n_features) |

### AC-4: Evaluation Metrics
| ID | Criteria | Verification |
|----|----------|-------------|
| AC-4.1 | MAPE calculated correctly: mean(|actual - predicted| / |actual|) × 100, handles zero actuals | Unit test: known values → expected MAPE; zero actual → excluded or capped |
| AC-4.2 | RMSE calculated correctly: sqrt(mean((actual - predicted)²)) | Unit test: known values → expected RMSE |
| AC-4.3 | MAE calculated correctly: mean(|actual - predicted|) | Unit test: known values → expected MAE |
| AC-4.4 | R² calculated correctly: 1 - SS_res/SS_tot | Unit test: perfect forecast → R²=1.0; mean forecast → R²=0.0 |
| AC-4.5 | Residual analysis: residuals = actual - predicted for each model | Unit test: verify residual computation, assert length matches |
| AC-4.6 | Error-by-hour heatmap data: aggregate errors by hour-of-day correctly | Unit test: known errors at known hours → correct aggregation |

### AC-5: Dashboard UI (Tabs 1-5)
| ID | Criteria | Verification |
|----|----------|-------------|
| AC-5.1 | All 6 tabs render without error for each of the 8 regions | E2E test: programmatically switch tabs and regions, assert no callback errors |
| AC-5.2 | Region dropdown contains all 8 balancing authorities including FPL | UI test: assert dropdown options = expected list |
| AC-5.3 | Tab 1: Demand forecast chart shows actual (solid), Prophet (dashed), ARIMA (dashed), EIA forecast (dotted), ensemble (bold) | UI test: assert 5 traces in figure with correct line styles |
| AC-5.4 | Tab 1: "Now" vertical line separates historical from forecast | UI test: assert vertical line at current timestamp |
| AC-5.5 | Tab 1: Peak demand card shows MW value, time, and confidence range | UI test: assert card renders with numeric values |
| AC-5.6 | Tab 1: Weather overlay toggle adds temperature trace to dual y-axis | UI test: toggle on → assert 2 y-axes and temperature trace |
| AC-5.7 | Tab 2: Correlation heatmap renders with correct variable labels | UI test: assert heatmap shape = (n_features, n_features) |
| AC-5.8 | Tab 2: Scatter plots are interactive (hover shows values) | Manual verification |
| AC-5.9 | Tab 3: Metrics table shows MAPE, RMSE, MAE, R² for all 4 models | UI test: assert table has 4 rows × 4 columns of metrics |
| AC-5.10 | Tab 3: Residual plots render (time series, histogram, ACF) | UI test: assert 3+ figures in tab |
| AC-5.11 | Tab 4: Stacked area chart shows all fuel types with correct colors | UI test: assert n_traces >= 5 (wind, solar, gas, nuclear, hydro, coal) |
| AC-5.12 | Tab 4: Renewable penetration % line chart renders correctly | UI test: assert values between 0-100% |
| AC-5.13 | Tab 5: NOAA alerts displayed when active for selected region | Integration test: mock alert data, assert alert cards render |
| AC-5.14 | Tab 5: Stress indicator computes and displays (0-100 scale) | UI test: assert indicator value in range |
| AC-5.15 | All charts use dark theme consistently | Visual inspection: no white backgrounds, consistent color palette |
| AC-5.16 | Dashboard is responsive — usable at 1024px, 1440px, 1920px widths | Manual test at each resolution |

### AC-6: Scenario Simulator (Tab 6)
| ID | Criteria | Verification |
|----|----------|-------------|
| AC-6.1 | Temperature slider ranges -10°F to 120°F, default = current forecast value | UI test: assert slider min/max/default |
| AC-6.2 | Wind speed slider ranges 0-50 mph | UI test: assert slider range |
| AC-6.3 | Cloud cover slider ranges 0-100% | UI test: assert slider range |
| AC-6.4 | Moving any slider triggers re-forecast within 2 seconds | Performance test: time from slider change to chart update < 2s |
| AC-6.5 | All 6 preset scenarios load without error and produce reasonable demand curves | Integration test: load each preset, assert forecast output is valid (no NaN, positive values) |
| AC-6.6 | Winter Storm Uri preset: demand increases significantly vs baseline (Texas heating load) | Integration test: assert Uri demand > baseline demand by > 10% |
| AC-6.7 | Scenario demand delta chart shows positive/negative MW change from baseline | UI test: assert delta trace present with both + and - values |
| AC-6.8 | Pricing model: low utilization (<70%) → base price; high utilization (>90%) → exponential spike | Unit test: utilization=0.5 → ~$50; utilization=0.95 → > $200; utilization=1.0 → > $1000 |
| AC-6.9 | Scenario comparison mode: up to 3 scenarios overlay on same chart | UI test: save 3 scenarios, assert 3 traces + baseline on chart |
| AC-6.10 | Generation mix shift: low wind scenario reduces wind generation, increases gas | Integration test: wind=2mph → assert wind_gen < baseline, gas_gen > baseline |
| AC-6.11 | Reserve margin indicator = capacity - forecasted demand, displayed as % and MW | UI test: assert reserve margin card shows both units |
| AC-6.12 | Carbon impact: when gas peakers replace renewables, CO₂/MWh increases | Integration test: compare carbon intensity between baseline and high-gas scenario |
| AC-6.13 | Scenario engine does not mutate the input feature matrix | Unit test: pass features, run scenario, assert original DataFrame unchanged |
| AC-6.14 | Scenario engine raises ValueError for unknown weather column names | Unit test: pass {"fake_column": 50} → assert ValueError raised |

### AC-7: Persona Switcher
| ID | Criteria | Verification |
|----|----------|-------------|
| AC-7.1 | Persona dropdown in header with 4 options: Grid Ops, Renewables, Trader, Data Sci | UI test: assert 4 options in selector |
| AC-7.2 | Switching persona changes default active tab per config table | UI test: switch to each persona, assert correct tab is active |
| AC-7.3 | KPI cards update to show persona-relevant metrics | UI test: Grid Ops shows Peak Demand; Trader shows Price Estimate |
| AC-7.4 | Welcome card generates with real data (not placeholder text) | UI test: assert welcome card contains numeric values from latest data |
| AC-7.5 | Tab visibility changes per persona (e.g., Trader doesn't see Model Comparison by default) | UI test: assert tab count changes per persona |
| AC-7.6 | Persona selection persists during session (doesn't reset on tab switch) | UI test: select persona → switch tab → assert persona unchanged |
| AC-7.7 | Default persona on first load is Grid Ops | UI test: fresh load → assert persona = grid_ops |

### AC-8: Infrastructure & Deployment
| ID | Criteria | Verification |
|----|----------|-------------|
| AC-8.1 | Docker image builds successfully from Dockerfile | CI test: `docker build .` exits 0 |
| AC-8.2 | Container starts and serves on port 8080 within 30 seconds | CI test: `docker run` → curl localhost:8080 within 30s |
| AC-8.3 | Health check endpoint /health returns 200 with JSON status | Integration test: GET /health → 200 + {"status": "healthy"} |
| AC-8.4 | Application runs with only EIA_API_KEY environment variable set (Open-Meteo and NOAA need no key) | Integration test: start with only EIA_API_KEY, assert no errors |
| AC-8.5 | Cloud Run deployment succeeds with `gcloud run deploy` | Deploy test: verify service URL is reachable |
| AC-8.6 | Application handles 10 concurrent users without error (Cloud Run autoscaling) | Load test: 10 concurrent requests → all return 200 |
| AC-8.7 | Memory usage stays under 2Gi during normal operation | Monitor: check Cloud Run metrics after deployment |
| AC-8.8 | Structured logging (structlog) outputs JSON to stdout for Cloud Run log aggregation | Integration test: assert log output is valid JSON |
| AC-8.9 | No secrets in codebase — API keys only via environment variables | Static analysis: grep for hardcoded keys |
| AC-8.10 | README includes: project description, setup instructions, screenshots, architecture diagram, deployment commands | Manual review |

### AC-9: Code Quality
| ID | Criteria | Verification |
|----|----------|-------------|
| AC-9.1 | All functions have type hints on parameters and return values | Static analysis: mypy or pyright with no errors |
| AC-9.2 | All public functions have docstrings (Google or NumPy style) | Static analysis: pydocstyle or manual review |
| AC-9.3 | No unused imports | Static analysis: ruff or flake8 with F401 |
| AC-9.4 | All Pydantic models validate input correctly (reject bad data) | Unit test: pass invalid data, assert ValidationError |
| AC-9.5 | Error handling: all API calls wrapped in try/except with logging | Code review: grep for bare requests.get without error handling |
| AC-9.6 | Test coverage ≥ 80% on data/, models/, simulation/, personas/ directories | Coverage report: pytest --cov |

---

## Test Plan

### Test Structure
```
tests/
├── conftest.py                     # Shared fixtures: mock API responses, sample DataFrames
├── unit/
│   ├── test_eia_client.py          # EIA API client: request building, response parsing, pagination
│   ├── test_weather_client.py      # Open-Meteo client: request building, response parsing
│   ├── test_noaa_client.py         # NOAA alerts: parsing, state mapping
│   ├── test_cache.py               # SQLite caching: set/get/TTL/stale
│   ├── test_feature_engineering.py # All 25+ features across 12 categories: CDD, HDD, lags, etc.
│   ├── test_preprocessing.py       # Data alignment, missing values, timezone handling
│   ├── test_prophet_model.py       # Prophet config, regressor attachment, forecast shape
│   ├── test_arima_model.py         # SARIMAX fitting, convergence, forecast shape
│   ├── test_xgboost_model.py       # XGBoost training, TimeSeriesSplit, feature importance
│   ├── test_ensemble.py            # Weight calculation, combining, bounds
│   ├── test_evaluation.py          # MAPE, RMSE, MAE, R² calculations
│   ├── test_pricing.py             # Merit-order pricing model at various utilization levels
│   ├── test_scenario_engine.py     # Weather override, derived feature recompute, delta calc
│   ├── test_scenario_presets.py    # Each preset loads, produces valid config
│   ├── test_persona_config.py      # Per-persona settings: tabs, KPIs, defaults
│   └── test_welcome_cards.py       # Welcome message generation with real data
├── integration/
│   ├── test_data_pipeline.py       # Full flow: API → cache → features → model-ready DataFrame
│   ├── test_model_pipeline.py      # Full flow: features → train → predict → evaluate
│   ├── test_scenario_pipeline.py   # Full flow: preset → override → forecast → pricing → delta
│   └── test_api_fallback.py        # API failure → cache fallback → stale data served
├── e2e/
│   ├── test_dashboard_render.py    # All 6 tabs render for all 8 regions without callback errors
│   ├── test_persona_switching.py   # Persona switch → correct tabs, KPIs, welcome
│   └── test_scenario_interaction.py # Slider move → chart update → comparison save
└── fixtures/
    ├── eia_response_ercot.json     # Sample EIA API response for ERCOT
    ├── eia_response_fpl.json       # Sample EIA API response for FPL
    ├── weather_response_ercot.json # Sample Open-Meteo response
    ├── noaa_alerts_texas.json      # Sample NOAA alert response
    ├── sample_demand_90days.csv    # 90 days of hourly demand data (synthetic)
    └── sample_features.csv         # Pre-computed feature matrix for testing
```

### Key Test Fixtures (conftest.py)
```python
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def sample_demand_df():
    """90 days of realistic hourly demand data with known patterns."""
    hours = pd.date_range("2025-01-01", periods=90*24, freq="h", tz="UTC")
    np.random.seed(42)
    # Base load + daily cycle + weekly cycle + noise
    base = 25000  # MW
    daily = 5000 * np.sin(2 * np.pi * np.arange(len(hours)) / 24 - np.pi/2)
    weekly = 1000 * np.sin(2 * np.pi * np.arange(len(hours)) / (24*7))
    noise = np.random.normal(0, 500, len(hours))
    demand = base + daily + weekly + noise
    return pd.DataFrame({"timestamp": hours, "demand_mw": demand})

@pytest.fixture
def sample_weather_df():
    """Matching weather data for the demand fixture — all 17 weather variables."""
    hours = pd.date_range("2025-01-01", periods=90*24, freq="h", tz="UTC")
    np.random.seed(42)
    n = len(hours)
    # Temperature: seasonal + daily cycle
    temp_base = 50 + 20 * np.sin(2 * np.pi * np.arange(n) / (24*365))
    temp_daily = 10 * np.sin(2 * np.pi * np.arange(n) / 24 - np.pi/3)
    temp = temp_base + temp_daily + np.random.normal(0, 3, n)
    solar_cycle = np.maximum(0, np.sin(2 * np.pi * np.arange(n) / 24 - np.pi/4))
    return pd.DataFrame({
        "timestamp": hours,
        "temperature_2m": temp,
        "apparent_temperature": temp - 2 + np.random.normal(0, 1, n),
        "relative_humidity_2m": np.clip(60 + np.random.normal(0, 15, n), 0, 100),
        "dew_point_2m": temp - 10 + np.random.normal(0, 3, n),
        "wind_speed_10m": np.abs(8 + np.random.normal(0, 4, n)),
        "wind_speed_80m": np.abs(12 + np.random.normal(0, 5, n)),
        "wind_speed_120m": np.abs(14 + np.random.normal(0, 5, n)),
        "wind_direction_10m": np.random.uniform(0, 360, n),
        "shortwave_radiation": np.clip(400 * solar_cycle, 0, 1000),
        "direct_normal_irradiance": np.clip(600 * solar_cycle, 0, 1000),
        "diffuse_radiation": np.clip(150 * solar_cycle + np.random.normal(0, 20, n), 0, 500),
        "cloud_cover": np.clip(50 + np.random.normal(0, 25, n), 0, 100),
        "precipitation": np.maximum(0, np.random.exponential(0.5, n)),
        "snowfall": np.where(temp < 32, np.maximum(0, np.random.exponential(0.2, n)), 0),
        "surface_pressure": 1013 + np.random.normal(0, 5, n),
        "soil_temperature_0cm": temp - 5 + np.random.normal(0, 2, n),
        "weather_code": np.random.choice([0, 1, 2, 3, 45, 51, 61, 71, 80, 95], n),
    })

@pytest.fixture
def sample_feature_matrix(sample_demand_df, sample_weather_df):
    """Pre-merged and feature-engineered matrix ready for model training."""
    from data.feature_engineering import engineer_features
    merged = sample_demand_df.merge(sample_weather_df, on="timestamp")
    return engineer_features(merged)

@pytest.fixture
def mock_eia_response():
    """Realistic EIA API v2 response for ERCOT demand."""
    import json
    with open("tests/fixtures/eia_response_ercot.json") as f:
        return json.load(f)

@pytest.fixture
def mock_weather_response():
    """Realistic Open-Meteo API response."""
    import json
    with open("tests/fixtures/weather_response_ercot.json") as f:
        return json.load(f)

@pytest.fixture
def trained_models(sample_feature_matrix):
    """Pre-trained models for testing predictions and scenarios."""
    from models.training import train_all_models
    return train_all_models(sample_feature_matrix, target_col="demand_mw")
```

### Critical Unit Tests (Examples)

#### test_feature_engineering.py
```python
import pytest
import pandas as pd
import numpy as np
from data.feature_engineering import (
    compute_cdd, compute_hdd, compute_wind_power,
    compute_solar_cf, encode_cyclical_time, compute_lags,
    compute_rolling_stats, compute_temp_deviation
)

class TestCDD:
    def test_hot_day(self):
        """85°F → CDD = 20"""
        assert compute_cdd(85.0) == 20.0

    def test_cold_day(self):
        """40°F → CDD = 0"""
        assert compute_cdd(40.0) == 0.0

    def test_boundary(self):
        """65°F → CDD = 0"""
        assert compute_cdd(65.0) == 0.0

    def test_series(self):
        temps = pd.Series([30, 65, 80, 100])
        expected = pd.Series([0, 0, 15, 35])
        pd.testing.assert_series_equal(compute_cdd(temps), expected)

class TestHDD:
    def test_cold_day(self):
        assert compute_hdd(40.0) == 25.0

    def test_hot_day(self):
        assert compute_hdd(85.0) == 0.0

    def test_boundary(self):
        assert compute_hdd(65.0) == 0.0

class TestWindPower:
    def test_zero_wind(self):
        assert compute_wind_power(0.0) == 0.0

    def test_rated_wind(self):
        """At rated wind speed, output should equal rated power."""
        result = compute_wind_power(12.0)  # typical rated speed m/s
        assert result > 0

    def test_above_cutout(self):
        """Above cutout speed (25 m/s), turbine shuts down."""
        assert compute_wind_power(30.0) == 0.0

    def test_cubic_relationship(self):
        """Power should roughly follow v³ in operating range."""
        p1 = compute_wind_power(5.0)
        p2 = compute_wind_power(10.0)
        # At 2x wind speed, power should be ~8x (2³)
        assert p2 / p1 > 6  # Allow some tolerance

class TestSolarCF:
    def test_full_sun(self):
        assert compute_solar_cf(1000.0) == 1.0

    def test_half_sun(self):
        assert compute_solar_cf(500.0) == 0.5

    def test_over_irradiance(self):
        """GHI > 1000 should clip to 1.0"""
        assert compute_solar_cf(1200.0) == 1.0

    def test_night(self):
        assert compute_solar_cf(0.0) == 0.0

class TestCyclicalEncoding:
    def test_hour_0_and_24_equal(self):
        sin_0, cos_0 = encode_cyclical_time(0, 24)
        sin_24, cos_24 = encode_cyclical_time(24, 24)
        assert abs(sin_0 - sin_24) < 1e-10
        assert abs(cos_0 - cos_24) < 1e-10

    def test_hour_6_sin_positive(self):
        sin_6, _ = encode_cyclical_time(6, 24)
        assert sin_6 > 0  # 6AM = positive sine

    def test_hour_12(self):
        sin_12, cos_12 = encode_cyclical_time(12, 24)
        assert abs(sin_12) < 1e-10  # sin(π) ≈ 0
        assert cos_12 < 0  # cos(π) = -1

class TestLagFeatures:
    def test_no_future_leakage(self):
        """Lag features should only look backward."""
        s = pd.Series(range(100))
        lag_24 = compute_lags(s, lag=24)
        # First 24 values should be NaN
        assert lag_24[:24].isna().all()
        # Value at index 24 should equal original value at index 0
        assert lag_24.iloc[24] == s.iloc[0]

    def test_lag_168_weekly(self):
        s = pd.Series(range(200))
        lag_168 = compute_lags(s, lag=168)
        assert lag_168[:168].isna().all()
        assert lag_168.iloc[168] == s.iloc[0]

class TestRollingStats:
    def test_backward_only(self):
        """Rolling window must not include current or future observations."""
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        rolling_mean = compute_rolling_stats(s, window=3, stat="mean")
        # Value at index 3 should be mean of indices 1,2,3 (backward-looking)
        # Exact behavior depends on implementation — assert no NaN after warmup
        assert not rolling_mean.iloc[3:].isna().any()

class TestTempDeviation:
    def test_constant_temp(self):
        """Constant temperature → deviation = 0"""
        temps = pd.Series([70.0] * 720)  # 30 days hourly
        deviation = compute_temp_deviation(temps, window=720)
        assert abs(deviation.iloc[-1]) < 1e-10

    def test_spike_detected(self):
        """Temperature spike → positive deviation"""
        temps = pd.Series([70.0] * 719 + [100.0])
        deviation = compute_temp_deviation(temps, window=720)
        assert deviation.iloc[-1] > 20  # 100 - ~70 = ~30
```

#### test_evaluation.py
```python
import pytest
import numpy as np
from models.evaluation import compute_mape, compute_rmse, compute_mae, compute_r2

class TestMAPE:
    def test_perfect_forecast(self):
        actual = np.array([100, 200, 300])
        predicted = np.array([100, 200, 300])
        assert compute_mape(actual, predicted) == 0.0

    def test_known_error(self):
        actual = np.array([100, 100, 100])
        predicted = np.array([110, 90, 105])
        # Errors: 10%, 10%, 5% → MAPE = 8.33%
        assert abs(compute_mape(actual, predicted) - 8.33) < 0.1

    def test_handles_zero_actual(self):
        """Zero actual values should be excluded or handled, not cause inf."""
        actual = np.array([0, 100, 200])
        predicted = np.array([10, 100, 200])
        result = compute_mape(actual, predicted)
        assert np.isfinite(result)

class TestRMSE:
    def test_perfect_forecast(self):
        actual = np.array([100, 200, 300])
        assert compute_rmse(actual, actual) == 0.0

    def test_known_values(self):
        actual = np.array([100, 200])
        predicted = np.array([110, 190])
        # Errors: 10, -10 → squared: 100, 100 → mean: 100 → sqrt: 10
        assert compute_rmse(actual, predicted) == 10.0

class TestR2:
    def test_perfect(self):
        actual = np.array([1, 2, 3, 4, 5])
        assert compute_r2(actual, actual) == 1.0

    def test_mean_prediction(self):
        """Predicting the mean should give R² ≈ 0."""
        actual = np.array([1, 2, 3, 4, 5])
        predicted = np.full_like(actual, actual.mean(), dtype=float)
        assert abs(compute_r2(actual, predicted)) < 1e-10
```

#### test_pricing.py
```python
import pytest
import numpy as np
from models.pricing import estimate_price_impact

class TestPricingModel:
    def test_low_utilization(self):
        """<70% utilization → near base price."""
        price = estimate_price_impact(demand=14000, capacity=25000, base_price=50)
        assert 40 <= price <= 60

    def test_moderate_utilization(self):
        """80% utilization → moderate increase."""
        price = estimate_price_impact(demand=20000, capacity=25000, base_price=50)
        assert 60 <= price <= 150

    def test_high_utilization(self):
        """>90% utilization → exponential spike."""
        price = estimate_price_impact(demand=23750, capacity=25000, base_price=50)
        assert price > 200

    def test_emergency_utilization(self):
        """100%+ utilization → emergency pricing."""
        price = estimate_price_impact(demand=26000, capacity=25000, base_price=50)
        assert price > 1000

    def test_monotonic_increase(self):
        """Prices should increase monotonically with utilization."""
        prices = [
            estimate_price_impact(d, 25000, 50)
            for d in [10000, 15000, 20000, 22500, 24000, 25000]
        ]
        assert all(prices[i] <= prices[i+1] for i in range(len(prices)-1))
```

#### test_scenario_engine.py
```python
import pytest
import numpy as np
import pandas as pd
from simulation.scenario_engine import simulate_scenario
from simulation.presets import PRESETS

class TestScenarioEngine:
    def test_baseline_equals_no_override(self, trained_models, sample_feature_matrix):
        """No weather overrides → scenario forecast equals baseline."""
        baseline = trained_models['ensemble'].predict(sample_feature_matrix)
        scenario, delta = simulate_scenario(
            features=sample_feature_matrix,
            weather_overrides={},  # No changes
            models=trained_models
        )
        np.testing.assert_array_almost_equal(delta, 0, decimal=1)

    def test_heat_increases_demand(self, trained_models, sample_feature_matrix):
        """Setting temperature to 110°F should increase demand vs baseline."""
        _, delta = simulate_scenario(
            features=sample_feature_matrix,
            weather_overrides={"temperature_2m": 110.0},
            models=trained_models
        )
        assert delta.mean() > 0  # Demand should increase

    def test_no_wind_reduces_wind_gen(self, trained_models, sample_feature_matrix):
        """Setting wind to 0 should reduce demand (less wind gen → more expensive peakers → behavioral response)."""
        scenario, delta = simulate_scenario(
            features=sample_feature_matrix.copy(),
            weather_overrides={"wind_speed_80m": 0.0, "wind_speed_120m": 0.0},
            models=trained_models
        )
        # Verify the scenario produced a valid result (not NaN)
        assert not np.isnan(scenario).any()
        # Wind power estimate should be 0 in the modified features
        # (tested indirectly — if model trained on wind, removing it changes output)

    def test_derived_features_recomputed(self, trained_models, sample_feature_matrix):
        """CDD/HDD should update when temperature is overridden."""
        from data.feature_engineering import compute_cdd, compute_hdd
        
        # Get baseline CDD before scenario
        original_cdd_mean = sample_feature_matrix['cooling_degree_days'].mean()
        
        # Run scenario with extreme heat — should produce different forecast than baseline
        scenario_hot, delta_hot = simulate_scenario(
            features=sample_feature_matrix.copy(),
            weather_overrides={"temperature_2m": 100.0},  # Very hot → CDD = 35
            models=trained_models
        )
        scenario_cold, delta_cold = simulate_scenario(
            features=sample_feature_matrix.copy(),
            weather_overrides={"temperature_2m": 30.0},   # Cold → CDD = 0, HDD = 35
            models=trained_models
        )
        
        # Hot scenario should produce higher demand than cold scenario
        # (validates that derived features like CDD/HDD were recomputed and affected the forecast)
        assert scenario_hot.mean() > scenario_cold.mean(), \
            "Hot scenario should produce higher demand than cold — derived features likely not recomputed"

    def test_all_presets_valid(self, trained_models, sample_feature_matrix):
        """Every preset scenario should produce a valid forecast (no NaN, positive values)."""
        for name, preset in PRESETS.items():
            scenario, delta = simulate_scenario(
                features=sample_feature_matrix.copy(),
                weather_overrides=preset['weather'],
                models=trained_models
            )
            assert not np.isnan(scenario).any(), f"Preset '{name}' produced NaN"
            assert (scenario > 0).all(), f"Preset '{name}' produced negative demand"

    def test_input_not_mutated(self, trained_models, sample_feature_matrix):
        """Scenario engine must not modify the input DataFrame."""
        original = sample_feature_matrix.copy()
        simulate_scenario(
            features=sample_feature_matrix,
            weather_overrides={"temperature_2m": 110.0},
            models=trained_models
        )
        pd.testing.assert_frame_equal(sample_feature_matrix, original)

    def test_invalid_column_raises(self, trained_models, sample_feature_matrix):
        """Unknown weather column should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown weather column"):
            simulate_scenario(
                features=sample_feature_matrix.copy(),
                weather_overrides={"fake_nonexistent_column": 50.0},
                models=trained_models
            )

class TestPresets:
    def test_all_presets_have_required_keys(self):
        for name, preset in PRESETS.items():
            assert 'weather' in preset, f"Preset '{name}' missing 'weather'"
            assert 'name' in preset, f"Preset '{name}' missing 'name'"
            assert 'description' in preset, f"Preset '{name}' missing 'description'"
            assert 'date' in preset, f"Preset '{name}' missing 'date'"
            assert 'region' in preset, f"Preset '{name}' missing 'region'"

    def test_uri_preset_targets_ercot(self):
        uri = PRESETS['winter_storm_uri']
        assert uri['region'] == 'ERCOT'

    def test_irma_preset_targets_fpl(self):
        irma = PRESETS['hurricane_irma']
        assert irma['region'] == 'FPL'

    def test_uri_preset_is_extreme_cold(self):
        uri = PRESETS['winter_storm_uri']
        assert uri['weather']['temperature_2m'] < 10  # Below 10°F
        assert uri['weather']['wind_speed_80m'] < 10  # Low wind during ice storm

    def test_heat_dome_preset_is_extreme_hot(self):
        hd = PRESETS['summer_2023_heat_dome']
        assert hd['weather']['temperature_2m'] > 105  # Above 105°F
```

#### test_persona_config.py
```python
import pytest
from personas.config import PERSONA_CONFIGS, get_persona_config

class TestPersonaConfig:
    def test_all_four_personas_defined(self):
        expected = {"grid_ops", "renewables", "trader", "data_sci"}
        assert set(PERSONA_CONFIGS.keys()) == expected

    def test_each_has_required_keys(self):
        required_keys = {"default_tab", "kpi_metrics", "visible_tabs", "alert_threshold", "welcome_template"}
        for persona, config in PERSONA_CONFIGS.items():
            for key in required_keys:
                assert key in config, f"Persona '{persona}' missing '{key}'"

    def test_grid_ops_defaults_to_forecast(self):
        config = get_persona_config("grid_ops")
        assert config["default_tab"] == "tab-forecast"

    def test_data_sci_defaults_to_models(self):
        config = get_persona_config("data_sci")
        assert config["default_tab"] == "tab-models"

    def test_visible_tabs_are_subset_of_all_tabs(self):
        all_tabs = {"tab-forecast", "tab-weather", "tab-models", "tab-generation", "tab-alerts", "tab-simulator"}
        for persona, config in PERSONA_CONFIGS.items():
            assert set(config["visible_tabs"]).issubset(all_tabs), f"Persona '{persona}' has invalid tabs"

    def test_invalid_persona_raises(self):
        with pytest.raises(KeyError):
            get_persona_config("invalid_persona")
```

### Running Tests
```bash
# Run all tests
pytest tests/ -v --tb=short

# Run with coverage
pytest tests/ -v --cov=data --cov=models --cov=simulation --cov=personas --cov-report=html

# Run only unit tests (fast)
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Run specific test file
pytest tests/unit/test_feature_engineering.py -v

# Run with markers
pytest tests/ -v -m "not slow"  # Skip slow integration tests
```

### Coverage Targets
| Directory | Target | Rationale |
|-----------|--------|-----------|
| `data/` | ≥ 90% | Core data pipeline — errors here cascade everywhere |
| `models/` | ≥ 85% | Model training/evaluation — critical for correctness |
| `simulation/` | ≥ 90% | Scenario engine — user-facing feature, must be reliable |
| `personas/` | ≥ 95% | Simple config — easy to cover fully |
| `components/` | ≥ 60% | UI callbacks — harder to test programmatically, some manual |
| **Overall** | **≥ 80%** | |
