# Forecast Backtest Results

**Date:** 2026-02-21
**Holdout Period:** 21 days (504 hours)
**Training Period:** ~43 days
**Test Period:** 2026-01-31 to 2026-02-20

## Summary

Backtested forecast accuracy for Prophet, ARIMA, and XGBoost models against actual EIA demand data using a 21-day holdout period.

| Region | Best Model | MAPE | RMSE (MW) | R² |
|--------|------------|------|-----------|-----|
| **ERCOT** | XGBoost | **3.13%** | 2,198 | 0.853 |
| **FPL** | XGBoost | **7.51%** | 2,106 | 0.649 |

**Key Finding:** XGBoost significantly outperforms Prophet and ARIMA for both regions.

---

## ERCOT Results

**Mean Actual Demand:** 50,101 MW
**Std Actual Demand:** 5,729 MW

### Model Performance

| Model | MAPE % | RMSE (MW) | MAE (MW) | R² |
|-------|--------|-----------|----------|-----|
| Prophet | 50.18 | 29,891 | 24,544 | -26.22 |
| ARIMA | 40.51 | 20,779 | 19,313 | -12.15 |
| **XGBoost** | **3.13** | **2,198** | **1,544** | **0.853** |
| Ensemble | 3.87 | 2,461 | 1,915 | 0.815 |

### Ensemble Weights (1/MAPE)
- Prophet: 0.05
- ARIMA: 0.07
- XGBoost: 0.88

### XGBoost Top Features
1. heating_degree_days
2. demand_roll_24h_min
3. demand_lag_24h
4. temperature_2m
5. apparent_temperature

### Daily MAPE Analysis

**Best Days:**
- 2026-02-12: 1.24%
- 2026-02-11: 1.32%
- 2026-02-13: 1.42%
- 2026-02-10: 1.91%
- 2026-02-14: 2.12%

**Worst Days:**
- 2026-02-07: 9.27%
- 2026-02-03: 8.48%
- 2026-02-06: 7.11%
- 2026-02-08: 6.33%
- 2026-02-02: 4.77%

### Error by Hour of Day

**Highest Error Hours:** 13:00-14:00, 08:00-10:00 (morning ramp-up, afternoon peak)
**Lowest Error Hours:** 01:00-03:00, 17:00-18:00 (overnight stable, evening)

---

## FPL (Florida Power & Light) Results

**Mean Actual Demand:** 14,979 MW
**Std Actual Demand:** 3,555 MW

### Model Performance

| Model | MAPE % | RMSE (MW) | MAE (MW) | R² |
|-------|--------|-----------|----------|-----|
| Prophet | 24.51 | 4,932 | 3,743 | -0.925 |
| ARIMA | 15.81 | 3,661 | 2,690 | -0.061 |
| **XGBoost** | **7.51** | **2,106** | **1,282** | **0.649** |
| Ensemble | 10.39 | 2,648 | 1,778 | 0.445 |

### Ensemble Weights (1/MAPE)
- Prophet: 0.17
- ARIMA: 0.27
- XGBoost: 0.56

### XGBoost Top Features
1. demand_lag_24h
2. demand_lag_168h
3. demand_roll_24h_min
4. cooling_degree_days
5. demand_roll_24h_max

### Daily MAPE Analysis

**Best Days:**
- 2026-02-07: 3.05%
- 2026-02-10: 3.20%
- 2026-02-11: 3.59%
- 2026-02-12: 4.07%
- 2026-02-09: 4.26%

**Worst Days:**
- 2026-02-01: 26.66%
- 2026-02-02: 24.67%
- 2026-02-03: 20.41%
- 2026-02-20: 19.97%
- 2026-02-19: 16.57%

### Error by Hour of Day

**Highest Error Hours:** 12:00-16:00 (afternoon peak demand)
**Lowest Error Hours:** 00:00, 06:00-08:00, 23:00 (overnight/early morning)

---

## Analysis & Recommendations

### Why XGBoost Outperforms

1. **Feature Engineering:** XGBoost leverages 43 engineered features including:
   - Lagged demand (24h, 168h)
   - Rolling statistics (min, max, mean)
   - Temperature-derived features (CDD, HDD)
   - Calendar features (hour, day of week)

2. **Non-linear Relationships:** Captures complex interactions between weather and demand that linear models miss.

3. **Cross-Validation:** 5-fold TimeSeriesSplit prevents data leakage and ensures robust validation.

### Why Prophet/ARIMA Underperform

1. **Limited Training Data:** Only 43 days of training may be insufficient for capturing seasonal patterns that Prophet and ARIMA rely on.

2. **No Weather Integration:** ARIMA doesn't use weather features; Prophet only uses 7 regressors vs XGBoost's 43.

3. **Cold Start:** Prophet performs better with 1-2+ years of historical data to learn yearly seasonality.

### Recommendations

1. **Production Deployment:** Use XGBoost as the primary model with higher ensemble weight (0.8+).

2. **Retrain Frequency:** Weekly retraining recommended to capture recent demand patterns.

3. **Feature Importance:** Focus on maintaining data quality for:
   - 24-hour lag features (most predictive)
   - Temperature and degree-day calculations
   - Rolling statistics

4. **Model Improvement:**
   - Collect more historical data (6+ months) for Prophet
   - Consider LSTM or Transformer models for sequence learning
   - Add external features: holidays, economic indicators, major events

---

## Running the Backtest

```bash
# Activate virtual environment
source venv/bin/activate

# Run backtest for a specific region
python scripts/backtest.py --region ERCOT --holdout-days 21
python scripts/backtest.py --region FPL --holdout-days 21

# Available regions: ERCOT, FPL, CAISO, PJM, MISO, NYISO, ISONE, SPP
```

## Test Environment

- Python: 3.13
- XGBoost: 3.2.0
- Prophet: 1.3.0
- pmdarima: 2.1.1
- scikit-learn: 1.8.0
