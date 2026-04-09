# Senior Data Science Review: Charts, Tables, and Forecasting Logic

Date: 2026-04-09

Scope reviewed:
- Chart/table rendering logic in `components/callbacks.py`
- Forecast generation and diagnostics logic in `components/callbacks.py` and `models/model_service.py`
- Core model wrappers in `models/prophet_model.py`, `models/arima_model.py`, and `models/xgboost_model.py`

## Executive summary

The codebase has several strong patterns (walk-forward fold structure, cache layering, explicit model modules), but there are **critical methodological issues** that can materially misstate model quality in charts/tables and create non-production-like behavior.

Top concerns:
1. **Backtest data leakage in ensemble weighting** (uses test actuals to set weights).
2. **Misleading uncertainty presentation** (fixed-width band labeled as 80% CI).
3. **Invalid 95% interval derivation in Prophet wrapper**.
4. **Diagnostics tab compares actuals to predictions not generated for those timestamps/features**.
5. **Model selection bug forces XGBoost computation even when not selected**.

---

## Findings

### 1) Critical — Ensemble backtest leakage (uses holdout actuals to compute weights)

**Where:** `components/callbacks.py`, `_ensemble_fold`.

The ensemble weights are computed using `compute_mape(actual, pred)` where `actual` is the current fold's holdout truth, then immediately used to blend that same fold's predictions.

Why this is incorrect:
- This is target leakage inside evaluation.
- It makes the ensemble backtest optimistic relative to deploy-time behavior, where holdout actuals are unknown.

Industry-standard fix:
- Learn weights only from prior folds (or a nested validation split inside training), then apply those fixed weights to the current test fold.
- Alternatively, use static pre-declared weights for backtest to mimic production.

### 2) High — "80% CI" band is deterministic heuristic, not calibrated uncertainty

**Where:** `components/callbacks.py`, `_add_confidence_bands` and `_confidence_half_width`.

The band is defined as `prediction * (1 ± constant)` with hardcoded percentages by horizon, then labeled "80% CI".

Why this is incorrect:
- This is not a statistical confidence/prediction interval.
- No calibration step verifies nominal coverage (e.g., 80% of actuals inside band).

Industry-standard fix:
- Label as "heuristic range" unless calibrated.
- Build empirical quantile models or conformal intervals with backtest-based coverage validation.

### 3) High — Prophet "95%" bounds are mathematically invalid

**Where:** `models/prophet_model.py`, `predict_prophet`.

The code derives 95% bounds by scaling 80-ish outputs:
- `lower_95 = yhat_lower * 0.95`
- `upper_95 = yhat_upper * 1.05`

Why this is incorrect:
- Multiplying lower/upper bounds by constants does not transform interval coverage.
- Coverage becomes unknown and misleading.

Industry-standard fix:
- Request true quantiles from the model/posterior samples, or estimate empirical residual quantiles from rolling backtests.

### 4) Critical — Diagnostics tab residual charts are likely misaligned/non-comparable

**Where:** `components/callbacks.py`, `update_models_tab`; `models/model_service.py`, `_predict_from_trained`.

`update_models_tab` passes raw `demand_df` into `get_forecasts`, then computes residuals as `actual - pred` for the same timestamps. But model wrappers are not consistently producing in-sample aligned predictions for those timestamps:
- Prophet path uses `make_future_dataframe(periods=n)` and takes the tail (future horizon behavior).
- XGBoost path expects feature-engineered inputs; raw demand frame can trigger missing-feature zero-filling.

Why this is incorrect:
- Residual charts/tables can compare actuals against forecasts for a different target period or feature context.
- This invalidates model diagnostics and can mislead users.

Industry-standard fix:
- Build a clear diagnostic pipeline: either strict in-sample fitted values or strict out-of-sample backtest predictions with aligned timestamps and engineered features.

### 5) Medium — Model filter logic still computes XGBoost when unselected

**Where:** `models/model_service.py`, `_predict_from_trained`.

Condition:
```python
if models_shown and name not in models_shown and name != "xgboost":
    continue
```
This means XGBoost bypasses the filter and is always executed.

Why this is incorrect:
- Violates user selection semantics.
- Adds unnecessary compute and can alter downstream ensemble/diagnostic behavior.

Industry-standard fix:
- Respect `models_shown` for all models consistently.

### 6) Medium — "Simulated" forecast fallback can be misread as real model quality

**Where:** `models/model_service.py`, `_simulate_forecasts` and public return payload.

Simulated predictions are generated as `actual * (1 + noise)` and metrics are computed against the same actual series.

Why this is problematic:
- These are synthetic perturbations of truth, not model forecasts.
- If surfaced in charts/tables without unmistakable labeling, users may overtrust reported metrics.

Industry-standard fix:
- Gate simulated mode behind explicit "demo/synthetic" banner and separate metric namespace.
- Avoid displaying synthetic metrics alongside trained-model KPIs without clear segregation.

### 7) Medium — Short-horizon future feature generation freezes lag-like signals

**Where:** `components/callbacks.py`, `_create_future_features`.

For horizons `<168`, non-time features are copied from `last_row` across all future rows.

Why this is problematic:
- Autoregressive and weather-sensitive models receive unrealistic static feature trajectories.
- Can flatten intraday dynamics and produce overconfident/illogical charts.

Industry-standard fix:
- For autoregressive features, use recursive rollout.
- For exogenous weather, use forecast drivers or scenario-based trajectories.

### 8) Medium — Backtest chart error polygon can obscure interpretation over discontinuous folds

**Where:** `components/callbacks.py`, `update_backtest_chart`.

Error shading is drawn as one continuous polygon over concatenated folds; discontinuities between folds can create visual artifacts and imply continuity.

Why this is problematic:
- Visual miscommunication in model-review contexts.

Industry-standard fix:
- Shade per fold segment or facet by fold to preserve temporal/test-window boundaries.

---

## Quality standard assessment

Against standard forecasting QA expectations (leakage control, valid uncertainty, timestamp alignment, faithful diagnostics):
- **Fails critical standards** on leakage and diagnostic comparability.
- **Partially meets** engineering robustness (caching, exception handling), but methodological correctness issues dominate.

## Recommended remediation order

1. Fix ensemble backtest leakage (Finding 1).
2. Rebuild diagnostics pipeline with strict timestamp/feature alignment (Finding 4).
3. Remove/rename pseudo-CI and replace with calibrated intervals (Findings 2–3).
4. Correct model selection semantics and simulated-mode presentation (Findings 5–6).
5. Improve future feature generation realism and fold-aware plotting (Findings 7–8).
