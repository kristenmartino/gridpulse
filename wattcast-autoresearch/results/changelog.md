# ⚡ WattCast Autoresearch Changelog

*Agent lab notebook — documents the reasoning behind every experiment.*
*Run `python journal.py` for the auto-generated trend analysis and summary.*

---

## Session Summary

**Baseline:** 1.7232% MAPE | **Best:** 1.4411% MAPE | **Improvement:** 16.4% | **Experiments:** 30 (14 kept)

---

### Experiment 0 — BASELINE
**What I changed:** Ran unmodified train.py to establish baseline.
**Result:** MAPE 1.7232%
**What I learned:** Top 3 lag features (168h, 1h, 24h) account for 96.3% of feature importance. Model is lag-dominated.
**Next move:** Try demand momentum features to capture ramping direction.

### Experiment 1 — KEEP
**What I changed:** Added demand_momentum_short (lag_1h - lag_3h) and demand_momentum_long (lag_1h - lag_24h).
**Hypothesis:** Momentum captures whether demand is ramping up or down, which raw lags don't express directly.
**Result:** MAPE 1.6851% (delta -0.0381%)
**What I learned:** Momentum features add genuine signal; short-term derivatives of demand help the model anticipate direction.
**Next move:** Try more ratio/derivative features.

### Experiment 2 — DISCARD
**What I changed:** Added demand_ratio_weekly (lag_24h / rolling_mean_168h).
**Hypothesis:** Flagging abnormal days relative to weekly average would help.
**Result:** MAPE 1.6946% (delta +0.0095%)
**What I learned:** This ratio introduced noise — the model already captures weekly patterns via lag_168h.
**Next move:** Try temporal interaction features.

### Experiment 3 — DISCARD
**What I changed:** Added hour x month interaction feature.
**Hypothesis:** Demand patterns shift by season AND time of day; explicit interaction would help.
**Result:** MAPE 1.6942% (delta +0.0091%)
**What I learned:** XGBoost already learns hour/month interactions via tree splits. Explicit multiplication adds redundancy.
**Next move:** Try temperature x hour interaction.

### Experiment 4 — DISCARD
**What I changed:** Added temperature x hour interaction.
**Hypothesis:** Cooling/heating demand varies by time of day.
**Result:** MAPE 1.7025% (delta +0.0174%)
**What I learned:** Same issue as exp 3. Tree models handle these interactions natively.
**Next move:** Try weather change rates instead.

### Experiment 5 — DISCARD
**What I changed:** Added temp_delta_3h and temp_delta_6h (current temp - lagged temp).
**Hypothesis:** Rapid temperature drops spike heating demand.
**Result:** MAPE 1.6907% (delta +0.0056%)
**What I learned:** Temperature deltas didn't help — the model can compute these implicitly from temp + temp_lags.
**Next move:** Pivot to training strategy — try early stopping.

### Experiment 6 — KEEP
**What I changed:** Added early_stopping_rounds=50 and increased n_estimators to 1000.
**Hypothesis:** The model was undertrained at 500 trees with no early stopping safeguard.
**Result:** MAPE 1.5690% (delta -0.1161%)
**What I learned:** Massive improvement! The model was severely undertrained. 500 trees at LR 0.05 was not enough.
**Next move:** Push further — more trees with lower learning rate.

### Experiment 7 — KEEP
**What I changed:** 2000 trees, LR 0.03 (from 1000 trees, LR 0.05).
**Hypothesis:** Lower LR with more trees should converge better.
**Result:** MAPE 1.5299% (delta -0.0391%)
**What I learned:** Still hitting iteration ceiling (1999). Model wants even more trees.
**Next move:** Keep pushing: more trees + lower LR.

### Experiment 8 — KEEP
**What I changed:** 4000 trees, LR 0.02.
**Hypothesis:** Continue the lower-LR-more-trees strategy.
**Result:** MAPE 1.4922% (delta -0.0377%)
**What I learned:** Still at ceiling (3999). Improvement rate slowing but significant.
**Next move:** Try 6000 trees with LR 0.015.

### Experiment 9 — KEEP
**What I changed:** 6000 trees, LR 0.015, early_stopping_rounds=100.
**Hypothesis:** Finding the natural stopping point.
**Result:** MAPE 1.4818% (delta -0.0104%)
**What I learned:** Still no early stop at 5999. Improvement rate slowing — approaching convergence.
**Next move:** Try tree structure changes (depth, regularization).

### Experiment 10 — DISCARD
**What I changed:** Reduced max_depth from 8 to 6.
**Hypothesis:** Shallower trees might generalize better.
**Result:** MAPE 1.5537% (delta +0.0719%)
**What I learned:** Depth 8 is critical. The demand-lag interactions need deep trees to model properly.
**Next move:** Try deeper trees instead.

### Experiment 11 — DISCARD
**What I changed:** Increased max_depth to 10.
**Hypothesis:** More depth = more complex interactions.
**Result:** MAPE 1.4943% (delta +0.0125%)
**What I learned:** Depth 10 overfits slightly. 8 is the sweet spot.
**Next move:** Try sampling and regularization tuning.

### Experiment 12 — DISCARD
**What I changed:** subsample and colsample_bytree from 0.8 to 0.9.
**Hypothesis:** Less aggressive sampling might help.
**Result:** MAPE 1.4828% (delta +0.0010%)
**What I learned:** 0.8 sampling was already optimal. More data per tree doesn't help.
**Next move:** Try lower regularization.

### Experiment 13 — KEEP
**What I changed:** reg_alpha 0.1->0.01, reg_lambda 1.0->0.5, gamma 0.1->0.05.
**Hypothesis:** Lower regularization lets the model fit more closely.
**Result:** MAPE 1.4803% (delta -0.0015%)
**What I learned:** Small but real gain. The baseline regularization was slightly too strong.
**Next move:** Try min_child_weight adjustment.

### Experiment 14 — DISCARD
**What I changed:** min_child_weight from 5 to 3.
**Hypothesis:** Finer splits allow more precise predictions.
**Result:** MAPE 1.4815% (delta +0.0012%)
**What I learned:** MCW=5 is the right balance for this dataset size.
**Next move:** Try different tree count + LR combinations.

### Experiment 15 — DISCARD
**What I changed:** 8000 trees, LR 0.01 (from 6000 trees, LR 0.015).
**Hypothesis:** More trees with lower LR might help.
**Result:** MAPE 1.4878% (delta +0.0075%)
**What I learned:** Too low LR doesn't converge fully even at 8000 iterations. 0.015 with 6000 was the sweet spot.
**Next move:** Pivot to feature engineering — try new derived features.

### Experiment 16 — KEEP
**What I changed:** Pruned 7 low-importance weather features (snow, rain, precip, wind_direction, wind_gusts, diffuse/direct radiation).
**Hypothesis:** Removing noise features helps the model focus on signals.
**Result:** MAPE 1.4767% (delta -0.0036%)
**What I learned:** Feature pruning works! Fewer features (54 vs 61) and better performance.
**Next move:** Continue pruning other low-importance features.

### Experiment 17 — DISCARD
**What I changed:** Removed raw hour and month (kept only sin/cos encodings).
**Hypothesis:** Cyclical encodings are redundant with raw integers for tree models.
**Result:** MAPE 1.4792% (delta +0.0025%)
**What I learned:** XGBoost uses raw integers for precise splits better than sin/cos alone. Keep both.
**Next move:** Try pruning temp lag features.

### Experiment 18 — DISCARD
**What I changed:** Pruned temp_lag_2h, 3h, 12h (kept 1h, 6h, 24h only).
**Hypothesis:** Intermediate temp lags are redundant.
**Result:** MAPE 1.4777% (delta +0.0010%)
**What I learned:** The intermediate lags contribute small but real signal. Keep them.
**Next move:** Try new engineered ratio features.

### Experiment 19 — KEEP
**What I changed:** Added demand_ratio_24h (lag_24h / rolling_mean_24h).
**Hypothesis:** Daily deviation ratio captures whether yesterday was abnormal.
**Result:** MAPE 1.4569% (delta -0.0198%)
**What I learned:** Big win! The ratio normalizes scale differences between regions and captures relative deviations.
**Next move:** Try weekly deviation ratio.

### Experiment 20 — KEEP
**What I changed:** Added demand_ratio_168h (lag_168h / rolling_mean_168h).
**Hypothesis:** Weekly deviation ratio for same-day comparison.
**Result:** MAPE 1.4442% (delta -0.0127%)
**What I learned:** Another strong signal. NYISO improved dramatically (1.56% -> 1.49%).
**Next move:** Try more derivative features.

### Experiment 21 — DISCARD
**What I changed:** Added demand_acceleration (2nd derivative: momentum difference).
**Hypothesis:** Rate of change in momentum captures inflection points.
**Result:** MAPE 1.4479% (delta +0.0037%)
**What I learned:** 2nd derivative adds noise. First-order momentum was enough.
**Next move:** Try hyperparameter fine-tuning.

### Experiment 22 — KEEP
**What I changed:** max_bin 256 -> 512.
**Hypothesis:** Finer histogram binning allows more precise splits.
**Result:** MAPE 1.4424% (delta -0.0018%)
**What I learned:** Finer bins help. Also: early stopping triggered at 5995 (first time!).
**Next move:** Try more sampling dimensions.

### Experiment 23 — KEEP
**What I changed:** Added colsample_bylevel=0.8.
**Hypothesis:** Another sampling dimension reduces overfitting.
**Result:** MAPE 1.4418% (delta -0.0006%)
**What I learned:** Tiny but real. Feature importance is now more balanced across features.
**Next move:** Continue tuning sampling parameters.

### Experiment 24 — DISCARD
**What I changed:** Added temp_max_24h, temp_min_24h, temp_range_24h.
**Hypothesis:** Temperature extremes drive demand spikes.
**Result:** MAPE 1.4445% (delta +0.0027%)
**What I learned:** Rolling temp stats add noise with current feature set. Model already captures this signal.
**Next move:** Try pruning rolling std features.

### Experiment 25 — DISCARD
**What I changed:** Removed all rolling std features (kept only rolling means).
**Hypothesis:** Std features have low importance and add noise.
**Result:** MAPE 1.6621% (delta +0.2203%)
**What I learned:** MAJOR regression! Rolling std captures demand volatility — critical for prediction.
**Next move:** Never remove rolling std. Try regularization changes.

### Experiment 26 — DISCARD
**What I changed:** reg_alpha 0.01->0.1, reg_lambda 0.5->2.0 (heavier L2).
**Hypothesis:** More regularization might help generalization.
**Result:** MAPE 1.4418% (delta +0.0000%)
**What I learned:** No difference — heavier regularization doesn't help or hurt at this point.
**Next move:** Try polynomial temperature features.

### Experiment 27 — DISCARD
**What I changed:** Added temp_cubed (temperature^3 / 1000).
**Hypothesis:** Cubic term captures asymmetric heating/cooling demand curve.
**Result:** MAPE 1.4467% (delta +0.0049%)
**What I learned:** XGBoost handles non-linearity natively. Polynomial features add noise.
**Next move:** Try different sampling ratios.

### Experiment 28 — KEEP
**What I changed:** subsample 0.8 -> 0.7.
**Hypothesis:** More aggressive subsampling reduces overfitting.
**Result:** MAPE 1.4413% (delta -0.0005%)
**What I learned:** Slight improvement from more stochastic training.
**Next move:** Match colsample to subsample.

### Experiment 29 — KEEP
**What I changed:** colsample_bytree 0.8 -> 0.7.
**Hypothesis:** Matching col/row sampling symmetry for regularization.
**Result:** MAPE 1.4411% (delta -0.0002%)
**What I learned:** Feature importance now much more distributed. Model learning diverse patterns.
**Next move:** Try gamma=0.

### Experiment 30 — DISCARD
**What I changed:** gamma 0.05 -> 0.0.
**Hypothesis:** Removing split penalty allows more splits.
**Result:** MAPE 1.4411% (identical to best)
**What I learned:** gamma at 0.05 was already negligible. No effect.
**Next move:** We're hitting diminishing returns. Session complete.

