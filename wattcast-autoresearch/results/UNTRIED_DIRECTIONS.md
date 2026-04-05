# Untried Research Directions

Directions from `program.md` that were **not attempted** during the 30-experiment session,
or were only partially explored. Organized by priority for a future autoresearch run.

**Current best:** 1.4411% MAPE (16.4% improvement from 1.7232% baseline)

---

## High Priority — Likely to yield gains

### Per-region models (Phase 4)
Train a separate XGBoost model per balancing authority instead of one global model.
ERCOT and SECO have the highest individual MAPE due to extreme weather sensitivity.
Per-region models could specialize to regional demand patterns without interference.
**Why untried:** Phase 4 scope; session ended at experiment 30.

### Stacking ensemble (Phase 4)
XGBoost + LightGBM ensemble with a meta-learner. LightGBM may capture different
interactions than XGBoost, and stacking can exploit complementary errors.
**Why untried:** Phase 4 scope.

### Target transform: log(demand) (Phase 4)
Predicting `log(demand_mw)` instead of raw demand stabilizes variance and can
improve MAPE since percentage errors become absolute errors in log-space.
**Why untried:** Phase 4 scope.

### Correlated feature pair removal (Phase 2)
Systematically identify highly correlated features (e.g., `wind_speed_10m` vs
`wind_speed_80m` vs `wind_speed_120m`) and keep only the strongest from each
group. Experiment 16 (weather pruning) showed this direction works.
**Why untried:** Only one round of pruning was done (exp 16). A systematic
correlation-based pass was never attempted.

### L1-heavy regularization for feature selection (Phase 2)
Use very high `reg_alpha` (L1) to zero out unimportant features, then retrain
with only the surviving features. Different from manual pruning — lets the
model decide what to drop.
**Why untried:** Regularization was only tuned for performance (exps 13, 26),
not used as a feature selection tool.

---

## Medium Priority — Mixed signals from related experiments

### Piecewise temperature features (Phase 1)
Multi-breakpoint temperature features (e.g., separate slopes for <40F, 40-65F,
65-80F, >80F) to capture the U-shaped demand-temperature curve. CDD/HDD are a
two-breakpoint version of this; finer breakpoints may help.
**Why untried:** Cubic temperature was tried (exp 27) and hurt. But piecewise
linear is fundamentally different from polynomial — tree models can already
do piecewise splits, but explicit features might help with fewer trees.
**Risk:** XGBoost handles non-linearity natively (exp 27 lesson).

### Rolling weather stats (Phase 1)
Min/max temperature in last 24h, precipitation accumulation, sustained wind
periods. Captures weather persistence and extremes.
**Why untried:** Rolling temp stats were tried (exp 24: temp_max/min/range_24h)
and hurt. Precipitation and wind persistence were never tried.
**Risk:** Exp 24 showed rolling temp stats added noise. Other weather stats
may behave similarly.

### Regional encoding (Phase 1)
Encode region as a categorical or one-hot feature so one global model can learn
region-specific patterns. Currently the model has no explicit region awareness.
**Why untried:** Not attempted. May overlap with per-region models direction.
**Risk:** One-hot encoding 8 regions adds 7 features, which is noise if the
model is already lag-dominated.

### Quantile features (Phase 4)
Replace or supplement rolling mean/std with rolling quantiles (10th, 50th, 90th
percentile). Quantiles are more robust to outliers than mean/std.
**Why untried:** Phase 4 scope. Rolling std was shown to be critical (exp 25:
removing it caused +0.22% MAPE regression), so quantiles as a replacement
are risky. As supplements, they may add signal.

### Custom objective: weighted MAPE (Phase 4)
Custom XGBoost loss function that penalizes peak-hour errors more heavily.
Peak-hour accuracy matters most for grid operations and pricing.
**Why untried:** Phase 4 scope.

---

## Low Priority — Related experiments suggest limited upside

### Hour x is_weekend interaction (Phase 1)
**Why deprioritized:** Hour x month interaction (exp 3) and temp x hour
interaction (exp 4) both hurt. XGBoost learns interactions via tree splits.
Explicit interaction terms consistently added noise in this model.

### Weather change rates (Phase 1)
Temperature delta over last 3, 6, 12 hours.
**Why deprioritized:** Tried as exp 5 (temp_delta_3h, temp_delta_6h) and it
hurt (+0.0056%). The model computes these implicitly from raw + lagged temps.

### Demand acceleration (2nd derivative) (Phase 1)
**Why deprioritized:** Tried as exp 21 and hurt (+0.0037%). First-order
momentum was enough; higher-order derivatives add noise.

### Calendar features: daylight savings transitions (Phase 4)
Binary flag for DST transition days.
**Why deprioritized:** Very narrow signal (2 days/year). Holiday flag (already
implemented) is a more impactful calendar feature.

---

## Key Learnings to Guide Future Runs

1. **Training strategy dominated** — Moving from 500 to 6000 trees with lower LR
   accounted for ~75% of total improvement. Always check if the model is
   undertrained before adding features.

2. **XGBoost handles interactions natively** — Explicit interaction terms
   (exps 3, 4, 27) consistently hurt. Save interaction features for linear models.

3. **Ratio/normalization features work** — demand_ratio_24h and demand_ratio_168h
   were the biggest feature engineering wins. Normalizing by rolling mean captures
   relative deviations that raw lags miss.

4. **Rolling std is critical** — Never remove rolling volatility features (exp 25
   lesson: +0.22% MAPE regression).

5. **Diminishing returns after exp 20** — Last 10 experiments yielded only 0.0031%
   total improvement. Major gains require a strategy shift (per-region models,
   stacking, or target transform).
