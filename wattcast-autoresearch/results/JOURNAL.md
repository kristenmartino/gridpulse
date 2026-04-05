# ⚡ WattCast Autoresearch Journal
*Auto-generated at 2026-04-04 21:17:52*

## Executive Summary

| Metric | Value |
|--------|-------|
| Total experiments | 31 |
| Improvements kept | 14 |
| Baseline MAPE | 1.7232% |
| Best MAPE | 1.4411% |
| Total improvement | 0.2821% (16.4% relative) |
| Compute time | 1246s (20.8 min) |

## Key Patterns & Insights

- ⚠️ **Plateau detected** — Last 5 experiments show <0.01% MAPE variation. Consider shifting strategy (e.g., from hyperparameter tuning to feature engineering, or vice versa).
- 🔥 **Strong improvement trend** — 7+ of last 10 experiments beat baseline. Current strategy is working well.
- 🗺️ **Regional gap**: ISONE is hardest (1.497% MAPE) while CAISO is easiest (1.368%). Targeting ISONE-specific features could yield the biggest gains.

## Region Performance Trends

| Region | Start MAPE | Current MAPE | Improvement |
|--------|-----------|-------------|-------------|
| 🟢 ISONE | 1.899% | 1.497% | 0.401% (21.1%) |
| 🟢 NYISO | 1.886% | 1.489% | 0.397% (21.1%) |
| 🟢 SECO | 1.695% | 1.419% | 0.276% (16.3%) |
| 🟢 CAISO | 1.623% | 1.368% | 0.254% (15.7%) |
| 🟢 ERCOT | 1.735% | 1.465% | 0.270% (15.6%) |
| 🟢 SPP | 1.673% | 1.415% | 0.258% (15.4%) |
| 🟢 MISO | 1.626% | 1.390% | 0.236% (14.5%) |
| 🟢 PJM | 1.650% | 1.486% | 0.164% (9.9%) |

## Most Consistently Important Features

*Averaged across kept experiments:*

 1. `demand_lag_168h` — 0.5954 ███████████████████████████████████████████████
 2. `demand_lag_1h` — 0.2759 ██████████████████████
 3. `demand_lag_24h` — 0.0811 ██████
 4. `demand_rolling_mean_6h` — 0.0179 █
 5. `demand_lag_2h` — 0.0067 
 6. `day_of_week` — 0.0020 
 7. `dow_sin` — 0.0018 
 8. `demand_rolling_mean_24h` — 0.0017 
 9. `hour_cos` — 0.0016 
10. `demand_lag_48h` — 0.0014 

## Experiment History

### Experiment 1 — ✅ KEEP
- **MAPE**: 1.7232%  ⬇️ inf% improvement
- **Best region**: CAISO (1.623%)
- **Worst region**: ISONE (1.899%)
- **Features used**: 59
- **Training time**: 5.5s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `day_of_week`

### Experiment 2 — ✅ KEEP
- **MAPE**: 1.6851%  ⬇️ 0.0381% improvement
- **Best region**: MISO (1.571%)
- **Worst region**: NYISO (1.873%)
- **Features used**: 61
- **Training time**: 4.0s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `demand_rolling_mean_24h`

### Experiment 3 — ❌ DISCARD
- **MAPE**: 1.6946%  ⬆️ 0.0095% regression
- **Best region**: MISO (1.570%)
- **Worst region**: ISONE (1.898%)
- **Features used**: 62
- **Training time**: 4.5s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `demand_rolling_mean_24h`

### Experiment 4 — ❌ DISCARD
- **MAPE**: 1.6942%  ⬆️ 0.0091% regression
- **Best region**: CAISO (1.576%)
- **Worst region**: ISONE (1.895%)
- **Features used**: 62
- **Training time**: 4.7s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `demand_rolling_mean_24h`

### Experiment 5 — ❌ DISCARD
- **MAPE**: 1.7025%  ⬆️ 0.0174% regression
- **Best region**: CAISO (1.582%)
- **Worst region**: NYISO (1.909%)
- **Features used**: 62
- **Training time**: 4.6s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `demand_rolling_mean_24h`

### Experiment 6 — ❌ DISCARD
- **MAPE**: 1.6907%  ⬆️ 0.0056% regression
- **Best region**: CAISO (1.579%)
- **Worst region**: NYISO (1.894%)
- **Features used**: 63
- **Training time**: 4.9s
- **Top features**: `demand_lag_1h`, `demand_lag_168h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `dow_sin`

### Experiment 7 — ✅ KEEP
- **MAPE**: 1.5690%  ⬇️ 0.1161% improvement
- **Best region**: CAISO (1.462%)
- **Worst region**: NYISO (1.714%)
- **Features used**: 61
- **Training time**: 9.1s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `demand_rolling_mean_24h`

### Experiment 8 — ✅ KEEP
- **MAPE**: 1.5299%  ⬇️ 0.0391% improvement
- **Best region**: CAISO (1.409%)
- **Worst region**: NYISO (1.651%)
- **Features used**: 61
- **Training time**: 18.3s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `day_of_week`

### Experiment 9 — ✅ KEEP
- **MAPE**: 1.4922%  ⬇️ 0.0377% improvement
- **Best region**: CAISO (1.377%)
- **Worst region**: NYISO (1.603%)
- **Features used**: 61
- **Training time**: 35.1s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `day_of_week`

### Experiment 10 — ✅ KEEP
- **MAPE**: 1.4818%  ⬇️ 0.0104% improvement
- **Best region**: CAISO (1.351%)
- **Worst region**: NYISO (1.584%)
- **Features used**: 61
- **Training time**: 51.7s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `demand_lag_2h`

### Experiment 11 — ❌ DISCARD
- **MAPE**: 1.5537%  ⬆️ 0.0719% regression
- **Best region**: MISO (1.438%)
- **Worst region**: NYISO (1.758%)
- **Features used**: 61
- **Training time**: 38.5s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `demand_rolling_mean_24h`

### Experiment 12 — ❌ DISCARD
- **MAPE**: 1.4943%  ⬆️ 0.0125% regression
- **Best region**: CAISO (1.348%)
- **Worst region**: PJM (1.596%)
- **Features used**: 61
- **Training time**: 84.9s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `demand_lag_2h`

### Experiment 13 — ❌ DISCARD
- **MAPE**: 1.4828%  ⬆️ 0.0010% regression
- **Best region**: CAISO (1.362%)
- **Worst region**: NYISO (1.576%)
- **Features used**: 61
- **Training time**: 54.5s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `heat_index_proxy`

### Experiment 14 — ✅ KEEP
- **MAPE**: 1.4803%  ⬇️ 0.0015% improvement
- **Best region**: CAISO (1.359%)
- **Worst region**: NYISO (1.579%)
- **Features used**: 61
- **Training time**: 52.8s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `demand_lag_2h`

### Experiment 15 — ❌ DISCARD
- **MAPE**: 1.4815%  ⬆️ 0.0012% regression
- **Best region**: CAISO (1.358%)
- **Worst region**: NYISO (1.573%)
- **Features used**: 61
- **Training time**: 55.6s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `demand_lag_2h`

### Experiment 16 — ❌ DISCARD
- **MAPE**: 1.4878%  ⬆️ 0.0075% regression
- **Best region**: CAISO (1.369%)
- **Worst region**: NYISO (1.590%)
- **Features used**: 61
- **Training time**: 74.2s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `demand_lag_2h`

### Experiment 17 — ✅ KEEP
- **MAPE**: 1.4767%  ⬇️ 0.0036% improvement
- **Best region**: CAISO (1.353%)
- **Worst region**: NYISO (1.577%)
- **Features used**: 54
- **Training time**: 53.4s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `hour_cos`

### Experiment 18 — ❌ DISCARD
- **MAPE**: 1.4792%  ⬆️ 0.0025% regression
- **Best region**: CAISO (1.354%)
- **Worst region**: NYISO (1.575%)
- **Features used**: 52
- **Training time**: 50.9s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `dow_sin`

### Experiment 19 — ❌ DISCARD
- **MAPE**: 1.4777%  ⬆️ 0.0010% regression
- **Best region**: CAISO (1.356%)
- **Worst region**: NYISO (1.590%)
- **Features used**: 51
- **Training time**: 44.5s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_rolling_mean_6h`, `demand_lag_24h`, `demand_lag_2h`

### Experiment 20 — ✅ KEEP
- **MAPE**: 1.4569%  ⬇️ 0.0198% improvement
- **Best region**: CAISO (1.339%)
- **Worst region**: NYISO (1.561%)
- **Features used**: 55
- **Training time**: 42.8s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `dow_sin`

### Experiment 21 — ✅ KEEP
- **MAPE**: 1.4442%  ⬇️ 0.0127% improvement
- **Best region**: CAISO (1.351%)
- **Worst region**: PJM (1.508%)
- **Features used**: 56
- **Training time**: 40.8s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `demand_lag_2h`

### Experiment 22 — ❌ DISCARD
- **MAPE**: 1.4479%  ⬆️ 0.0037% regression
- **Best region**: CAISO (1.360%)
- **Worst region**: PJM (1.511%)
- **Features used**: 57
- **Training time**: 41.3s
- **Top features**: `demand_lag_168h`, `demand_lag_24h`, `demand_lag_1h`, `demand_rolling_mean_6h`, `dow_sin`

### Experiment 23 — ✅ KEEP
- **MAPE**: 1.4424%  ⬇️ 0.0018% improvement
- **Best region**: CAISO (1.355%)
- **Worst region**: PJM (1.498%)
- **Features used**: 56
- **Training time**: 50.2s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `demand_lag_2h`

### Experiment 24 — ✅ KEEP
- **MAPE**: 1.4418%  ⬇️ 0.0006% improvement
- **Best region**: CAISO (1.356%)
- **Worst region**: NYISO (1.496%)
- **Features used**: 56
- **Training time**: 49.4s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `demand_lag_2h`

### Experiment 25 — ❌ DISCARD
- **MAPE**: 1.4445%  ⬆️ 0.0027% regression
- **Best region**: CAISO (1.365%)
- **Worst region**: ISONE (1.496%)
- **Features used**: 59
- **Training time**: 53.6s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `demand_lag_2h`

### Experiment 26 — ❌ DISCARD
- **MAPE**: 1.6621%  ⬆️ 0.2203% regression
- **Best region**: CAISO (1.590%)
- **Worst region**: PJM (1.726%)
- **Features used**: 51
- **Training time**: 49.1s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `demand_lag_2h`

### Experiment 27 — ❌ DISCARD
- **MAPE**: 1.4418%  ⬆️ 0.0000% regression
- **Best region**: CAISO (1.358%)
- **Worst region**: ISONE (1.497%)
- **Features used**: 56
- **Training time**: 54.7s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `demand_lag_2h`

### Experiment 28 — ❌ DISCARD
- **MAPE**: 1.4467%  ⬆️ 0.0049% regression
- **Best region**: CAISO (1.361%)
- **Worst region**: NYISO (1.501%)
- **Features used**: 57
- **Training time**: 55.0s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `demand_lag_2h`

### Experiment 29 — ✅ KEEP
- **MAPE**: 1.4413%  ⬇️ 0.0005% improvement
- **Best region**: CAISO (1.357%)
- **Worst region**: NYISO (1.500%)
- **Features used**: 56
- **Training time**: 53.1s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `demand_lag_2h`

### Experiment 30 — ✅ KEEP
- **MAPE**: 1.4411%  ⬇️ 0.0002% improvement
- **Best region**: CAISO (1.368%)
- **Worst region**: ISONE (1.497%)
- **Features used**: 56
- **Training time**: 52.2s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `demand_lag_2h`

### Experiment 31 — ❌ DISCARD
- **MAPE**: 1.4411%  ⬆️ 0.0000% regression
- **Best region**: CAISO (1.368%)
- **Worst region**: ISONE (1.497%)
- **Features used**: 56
- **Training time**: 52.5s
- **Top features**: `demand_lag_168h`, `demand_lag_1h`, `demand_lag_24h`, `demand_rolling_mean_6h`, `demand_lag_2h`

## Suggested Next Steps

1. **Per-region models**: Overall MAPE is strong. Train separate models for the hardest regions.
2. **Ensemble methods**: Combine XGBoost with LightGBM for additional gains.
3. **Target ISONE**: This region has the highest MAPE at 1.497%. Region-specific features could help.

---
*Journal generated by autoresearch journal.py — 31 experiments analyzed*