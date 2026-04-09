# Unified Review: Charts, Tables, and Forecasting Pipeline (Production Readiness)

Date: 2026-04-09

Scope reviewed:
- Chart/table rendering logic in `components/callbacks.py`
- Forecast generation and diagnostics logic in `components/callbacks.py` and `models/model_service.py`
- Core model wrappers in `models/prophet_model.py`, `models/arima_model.py`, and `models/xgboost_model.py`
- Scenario, backtest, and Redis/model-service fallback paths

## Executive summary

The codebase has good structural foundations (modular model wrappers, walk-forward folds, cache layering), but it currently fails key production forecasting quality standards due to leakage risk, timestamp/feature misalignment, and misleading uncertainty/diagnostic presentation.

Most urgent issues:
1. **Backtest data leakage in ensemble weighting** (weights are derived from holdout actuals).
2. **Prediction input misalignment** (trained-model path and Prophet regressor alignment issues).
3. **Diagnostics and scenario semantic inconsistencies** (comparisons and labels can be non-comparable or historically framed as forecast).
4. **Uncertainty representation is not statistically valid as currently labeled**.

---

## Consolidated findings

### P0 — Ensemble backtest leakage (critical)
**Where:** `components/callbacks.py`, `_ensemble_fold`.

Ensemble weights are computed from the same fold holdout actuals used for scoring, then applied to that fold’s predictions.

**Impact:** optimistic backtest error; non-production-like evaluation.

**Recommended fix:** learn weights only from prior folds (or nested validation inside train), then apply fixed weights to current fold.

### P0 — Trained inference path uses non-guaranteed feature-engineered inputs (critical)
**Where:** `models/model_service.py`, `_predict_from_trained`.

The trained-model path can pass raw `demand_df` directly to model predictors. For XGBoost, missing engineered features can be silently zero-filled.

**Impact:** predictions can be materially wrong while still appearing valid in charts/tables.

**Recommended fix:** enforce preprocessing + feature engineering before inference, validate required schema, and fail closed on missing mandatory features.

### P0 — Prophet regressor alignment to forecast timestamps is unreliable (critical)
**Where:** `models/prophet_model.py`, `predict_prophet`.

Regressors are assigned positionally to a `future` frame containing history + future rows. When provided regressors are horizon-only or differently indexed, forecast-period regressors can be flattened/misaligned.

**Impact:** weather/exogenous effects can be incorrectly applied with no explicit error.

**Recommended fix:** join regressors by timestamp (`ds`) and explicitly separate history vs future prediction windows.

### P1 — Diagnostics residual comparisons can be non-comparable (high)
**Where:** `components/callbacks.py`, `update_models_tab`; `models/model_service.py`, `_predict_from_trained`.

Residual views compute `actual - pred` while prediction generation is not consistently guaranteed to be aligned to those exact timestamps/features.

**Impact:** diagnostics may compare actuals against predictions produced for different contexts.

**Recommended fix:** establish one strict diagnostics mode: either aligned in-sample fitted values or aligned out-of-sample backtest predictions with identical feature lineage.

### P1 — Uncertainty bands are heuristic but labeled as confidence intervals (high)
**Where:** `components/callbacks.py`, `_confidence_half_width`, `_add_confidence_bands`; `models/prophet_model.py`, `predict_prophet`.

- App-level bands are deterministic percent envelopes by horizon.
- Prophet “95%” bounds are derived by scaling lower/upper values (multiplicative widening), which does not preserve interval coverage.

**Impact:** statistical confidence can be overstated/misrepresented.

**Recommended fix:** relabel as heuristic uncertainty until calibrated, or produce empirical/conformal quantiles from rolling residuals with measured coverage.

### P1 — Scenario tab semantics and controls are inconsistent with displayed meaning (high)
**Where:** `components/callbacks.py`, `run_scenario`.

- Several UI controls do not materially influence scenario demand as implied.
- “Forecast” baseline/scenario timeline can be built from historical tail timestamps rather than a true future horizon.

**Impact:** users can infer causal control/forecast meaning that the implementation does not provide.

**Recommended fix:** map each control to explicit modeled effects, generate future timestamps from latest observed time, and clearly separate “historical replay” from “forward forecast.”

### P1 — Backtest exogenous retrieval may not be vintage-correct (high)
**Where:** `components/callbacks.py`, `_build_forecast_exog_fold`.

Backtest exogenous inputs may be sourced by coverage match rather than strict as-of issuance time.

**Impact:** potential look-ahead bias and poor reproducibility.

**Recommended fix:** version/store exogenous forecasts by issuance timestamp and enforce as-of retrieval at fold origin.

### P2 — Model selection filter inconsistency (medium)
**Where:** `models/model_service.py`, `_predict_from_trained`.

Current selection logic can still execute XGBoost when unselected.

**Impact:** unnecessary compute and inconsistent user-selected model behavior.

**Recommended fix:** apply `models_shown` filtering uniformly across all models.

### P2 — Horizon boundary mismatch at 168h (medium)
**Where:** `components/callbacks.py`, `_create_future_features`.

Docstring intent (`<= 168h`) and implementation branch (`< 168`) diverge.

**Impact:** surprising behavior change exactly at 7-day horizon; QA baseline ambiguity.

**Recommended fix:** align code and docs intentionally and add boundary tests at 167/168/169.

### P2 — Simulated fallback metrics can be mistaken for real model quality (medium)
**Where:** `models/model_service.py`, `_simulate_forecasts`.

Synthetic forecasts are generated from perturbed actuals, then scored against those actuals.

**Impact:** synthetic mode can appear comparable to real trained-model performance if not clearly isolated.

**Recommended fix:** hard-label demo/synthetic mode and separate synthetic metrics from production KPIs.

### P3 — Backtest error-area rendering across concatenated folds can mislead (low/medium)
**Where:** `components/callbacks.py`, `update_backtest_chart`.

A single continuous error polygon over discontinuous folds can visually imply continuity.

**Impact:** interpretability artifacts in review dashboards.

**Recommended fix:** render fold-segmented bands or facet by fold.

### P3 — Missing metrics rendered as zero (low/medium)
**Where:** `components/callbacks.py`, `_models_tab_from_redis`, `update_models_tab`.

Unavailable metrics may display as `0` instead of null/NA.

**Impact:** users can misread data absence as valid zero values.

**Recommended fix:** render `—`/NA and attach data-quality/source indicators.

---

## Quality standard assessment

Against common production forecasting QA criteria (no leakage, timestamp/feature alignment, calibrated uncertainty, and trustworthy diagnostics):
- **Does not yet meet production-readiness bar** due to P0/P1 methodological correctness gaps.
- **Engineering foundations are solid**, but reliability of decision-support outputs depends on resolving the above issues.

## Prioritized remediation sequence

1. **P0:** eliminate ensemble leakage; enforce feature-engineered inference; fix Prophet regressor alignment.
2. **P1:** rebuild aligned diagnostics; correct scenario semantics/timeline; enforce vintage-correct exogenous backtests.
3. **P2:** fix model filtering, 168h boundary behavior, and simulated-mode isolation.
4. **P3:** improve fold-aware visualization and missing-metric display semantics.
