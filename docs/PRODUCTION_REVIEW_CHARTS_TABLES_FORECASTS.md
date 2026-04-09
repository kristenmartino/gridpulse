# Charts, Tables, and Forecast Pipeline Review (Production SaaS Standards)

Date: 2026-04-09
Scope reviewed:
- Forecast generation and model wrappers (`models/*`, `components/callbacks.py`)
- Chart/table assembly for forecast, backtest, model diagnostics, and scenario tabs
- Redis/model-service fallback paths

## Executive Summary

I found **6 high-impact correctness/quality issues** and **2 medium issues** that should be addressed before relying on these outputs for production decision support.

---

## High Impact Findings

### 1) Prophet regressors are misaligned to forecast horizon (likely incorrect forecasts)
**Where**: `models/prophet_model.py` (`predict_prophet`)

`predict_prophet` builds `future` with `make_future_dataframe(periods=periods)`, which includes historical rows + future rows. Regressors are then assigned from `df` by position, padding with the final observed regressor value when shorter than `future`. In the common case where `df` only contains the forecast horizon, this makes most forecast-period regressor rows flat/constant and not timestamp-aligned.

**Why this is a production issue**
- Forecasts can be materially wrong because weather regressors drive load shape.
- Silent failure mode: no exception, but degraded quality.

**Fix recommendation**
- Build a regressor frame keyed by `ds` and `merge` on timestamp (not positional assignment).
- For forward-only inference, predict only on truly future timestamps (or explicitly split history/future and populate each correctly).

---

### 2) Trained-model prediction path uses raw demand frame instead of engineered feature frame
**Where**: `models/model_service.py` (`_predict_from_trained`)

The service calls Prophet/ARIMA/XGBoost predict functions directly with `demand_df` (raw columns). For XGBoost specifically, missing features are defaulted to `0.0`, which effectively bypasses real model signal.

**Why this is a production issue**
- Predictions appear “trained” while running on invalid feature inputs.
- Diagnostics, tables, and charts can look plausible but be incorrect.

**Fix recommendation**
- Always run preprocessing + feature engineering before model inference.
- Enforce schema validation for required columns and fail closed when missing.
- Never silently fill all missing model features with zero in production paths.

---

### 3) Scenario simulator ignores several user controls for demand curve generation
**Where**: `components/callbacks.py` (`run_scenario`)

Scenario demand is computed from baseline plus temperature factor and random noise only. `cloud`, `humidity`, and `solar_irr` do not affect scenario demand; `wind` only affects random seed, not physics-based load/generation coupling.

**Why this is a production issue**
- UI implies multi-factor what-if behavior that model does not implement.
- Decision support can be misleading (“control does not control outcome”).

**Fix recommendation**
- Tie each slider to explicit modeled effects (load and/or net load).
- Surface model assumptions in tooltip/help text and include sensitivity checks.

---

### 4) Scenario “forecast” chart uses historical tail as baseline timeline
**Where**: `components/callbacks.py` (`run_scenario`)

The chart labeled baseline vs scenario forecast uses `demand_df.tail(duration)` timestamps and values (historical segment), not future timestamps.

**Why this is a production issue**
- Semantically incorrect labeling (“forecast” over historical window).
- Can cause operator confusion and incorrect operational interpretation.

**Fix recommendation**
- Generate future timeline from latest timestamp + 1h.
- Distinguish “historical replay” vs “forward forecast” in UX and code paths.

---

### 5) Backtest “forecast exogenous” mode may use non-vintage-aligned snapshots
**Where**: `components/callbacks.py` (`_build_forecast_exog_fold`)

Backtest fold exogenous values can come from current Redis snapshot keys, matched by timestamp coverage only. There is no strict constraint that weather forecasts were generated at fold forecast-issue time.

**Why this is a production issue**
- Risks optimistic bias or non-reproducible backtests.
- Violates common MLOps standard of vintage/time-travel correctness for exogenous features.

**Fix recommendation**
- Store and retrieve forecast snapshots keyed by issuance timestamp + horizon.
- During backtest, only use exogenous forecasts available as-of fold origin time.

---

### 6) Forecast feature-generation horizon branch contradicts docstring at boundary (168h)
**Where**: `components/callbacks.py` (`_create_future_features`)

Docstring says short horizon behavior applies to `<= 168h`, but implementation uses `if horizon < 168`, pushing exactly 168h into long-horizon branch.

**Why this is a production issue**
- Non-obvious behavior change at a key product horizon (7 days).
- Inconsistent expectation for model behavior and QA baselines.

**Fix recommendation**
- Align code and docs (`<= 168` or update docs intentionally).
- Add boundary tests for 167/168/169 horizons.

---

## Medium Findings

### 7) Confidence intervals are heuristic percent bands, not model-derived uncertainty
**Where**: `components/callbacks.py` (`_confidence_half_width`, `_add_confidence_bands`)

Fixed percentage bands by horizon are presented as “80% CI.”

**Risk**
- Potentially overstates statistical confidence and calibration quality.

**Recommendation**
- Rename to “uncertainty band (heuristic)” or calibrate quantiles from residual distributions per model/horizon/season.

### 8) Metrics tables default missing values to 0
**Where**: `components/callbacks.py` (`_models_tab_from_redis`, `update_models_tab`)

Missing metric values render as `0` rather than null/NA.

**Risk**
- Users may interpret unavailable metrics as perfect/valid zero values.

**Recommendation**
- Render `—` for missing values and add data-quality badges/source labels.

---

## Prioritized Remediation Plan

1. **P0**: Fix Prophet regressor timestamp alignment.
2. **P0**: Enforce feature-engineered inference in model_service.
3. **P1**: Correct scenario semantics (future timeline + slider-to-effect wiring).
4. **P1**: Implement vintage-correct exogenous backtest snapshots.
5. **P2**: Align 168h branch condition and add boundary tests.
6. **P2**: Replace/rename heuristic CIs; stop rendering missing metrics as zero.

