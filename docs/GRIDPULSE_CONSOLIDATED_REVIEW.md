# GridPulse Consolidated Product & Technical Review

**Date:** 2026-04-09
**Basis:** Synthesis of `CHART_TABLE_FORECAST_REVIEW.md`, `GRIDPULSE_360_PRODUCT_REVIEW.md`, `GRIDPULSE_MERGED_360_PRODUCT_REVIEW.md`, plus independent code audit of `callbacks.py`, `precompute.py`, `populate_redis.py`, `prophet_model.py`, `model_service.py`, and test suite.

> **Historical note (2026-04-19):** This review captures the state of the
> codebase when `precompute.py` ran as an in-process daemon thread and
> `populate_redis.py` ran as a 12h cron Cloud Run Job. That architecture
> has since been retired. Scoring and training now run as two scheduled
> Cloud Run Jobs (`gridpulse-scoring-job`, hourly; `gridpulse-training-job`,
> daily), shared phase logic lives in `jobs/phases.py`, and models are
> persisted to GCS via `models/persistence.py`. Findings below that
> reference `precompute.py` or `populate_redis.py` describe the prior
> system; read them as historical record, not as current code pointers.
> See [`docs/SCHEDULED_JOBS.md`](SCHEDULED_JOBS.md) for the current pipeline.

---

## How to read this document

Every finding is classified by **verification status**:

- **CONFIRMED** -- reproduced in code, root cause identified
- **VERIFIED FIX** -- was a real bug, already patched on main
- **PLAUSIBLE** -- code pattern is risky but no confirmed failure case yet
- **OVERSTATED** -- concern is valid in theory but severity is inflated or already mitigated
- **ALREADY ADDRESSED** -- existing code or recent work covers this
- **NOT APPLICABLE** -- finding is incorrect or doesn't apply to the primary code path

Priority labels follow the original reviews (P0/P1/P2/P3) but are re-assessed where warranted.

---

## Part 1: Technical Correctness (Code-Verified)

### 1.1 Ensemble backtest weight leakage -- CONFIRMED (P0)

**Where:** `components/callbacks.py:4336`, `_ensemble_fold`

**What:** `_ensemble_fold` receives the fold's holdout actuals and computes 1/MAPE weights from them. Those weights are then applied to that same fold's predictions. This means the weighting is optimized on the scoring data.

```python
# Line 4370-4375: weights derived from the fold being scored
if len(actual) > 0:
    for name, pred in preds.items():
        mape = compute_mape(actual, pred)
        if mape > 0:
            weights[name] = 1.0 / mape
```

**Impact:** Backtest metrics are optimistic. The ensemble appears to outperform individual models by an amount that won't reproduce in production.

**Fix options:**
1. **Use equal weights in backtests** (simple). Forward forecasts already use equal weights (line 708). Apply the same policy to backtests. This removes the leakage entirely and matches production behavior.
2. **Carry forward weights from prior folds** (more complex). Compute 1/MAPE weights from fold N-1, apply them frozen to fold N. Requires sequential fold processing.

**Recommendation:** Option 1. The 1/MAPE weighting provides marginal improvement over equal weights in energy demand contexts, and the leakage risk outweighs the benefit.

---

### 1.2 Prophet regressor alignment is positional, not timestamp-keyed -- CONFIRMED (P0)

**Where:** `models/prophet_model.py:149-164`, `predict_prophet`

**What:** Prophet's `make_future_dataframe` creates a frame covering history + future periods. Regressors from the input `df` are sliced positionally (`available[:len(future)]`) or padded with the last value. There is no join on timestamp.

```python
# Line 152-164: positional assignment, not timestamp-keyed
available = df[regressor_name].values
if len(available) >= len(future):
    future[regressor_name] = available[: len(future)]
else:
    padded = np.concatenate([
        available,
        np.full(len(future) - len(available), available[-1] ...)
    ])
    future[regressor_name] = padded
```

**Impact:** If `df` has a different length or time range than `future` (which includes the full training history plus forecast horizon), regressors are misaligned. Weather values from one hour are applied to a different hour's prediction. The error is silent -- predictions appear valid but weather effects are shifted.

**Fix:** Join regressors on timestamp. Construct the regressor DataFrame with a `ds` column, merge into `future` on `ds`, then forward-fill any gaps in the forecast period.

---

### 1.3 Ensemble never precomputed (was silently skipped) -- VERIFIED FIX

**Where:** `precompute.py:314` (before fix)

**What:** `_free_precompute_memory()` cleared `_region_featured` at line 305. The ensemble phase at line 314 then filtered on `_region_featured`, which was empty. Result: ensemble predictions were never generated during precompute. Every ensemble request fell through to on-demand computation, which required training all three models from scratch.

**Fix applied:** Changed filter to check `_region_data` (which persists) instead of `_region_featured`. Commit `eddb908`.

---

### 1.4 Redis fast path served XGBoost as ensemble -- VERIFIED FIX

**Where:** `components/callbacks.py:1473` (before fix)

**What:** Redis only stores XGBoost predictions (populated by `populate_redis.py`). The Redis fast path special-cased `"ensemble"` to bypass the model availability check, silently serving single-model XGBoost data when users requested the ensemble forecast. Users saw a single-model forecast labeled as ensemble, with no confidence bands or model diversity.

**Fix applied:** Removed `"ensemble"` from the Redis bypass. Ensemble requests now correctly fall through to the multi-model compute path. Commit `eddb908`.

---

### 1.5 Horizon ordering caused suboptimal cache warming -- VERIFIED FIX

**Where:** `precompute.py:54` (before fix)

**What:** `_ALL_HORIZONS = [168, 24, 720]` processed 720h last. On Cloud Run with container recycling, the most valuable horizon (30-day, which can also serve shorter horizons by slicing) was least likely to complete before timeout.

**Fix applied:** Changed to `[720, 168, 24]`. Commit `eddb908`.

---

### 1.6 168h boundary mismatch in `_create_future_features` -- CONFIRMED (P2)

**Where:** `components/callbacks.py:782`

**What:** Docstring says "For short horizons (<=168h)" but code branches on `horizon < 168`. At exactly 168h (the 7-day horizon), the long-horizon path runs, using historical hour/dow groupby averages instead of last-known values. This is inconsistent with the documented intent and creates a discontinuity at the exact boundary of the default 7-day forecast.

```python
# Line 758: docstring says <=168h
# Line 782: code says < 168
if horizon < 168:
```

**Fix:** Change to `horizon <= 168` and add boundary tests at 167/168/169.

---

### 1.7 Confidence bands are heuristic, labeled as CI -- CONFIRMED (P1)

**Where:** `components/callbacks.py:193-203`, `_confidence_half_width`

**What:** Bands are fixed percentage envelopes: +/-3% at 24h, +/-6% at 168h, +/-10% at 720h. These are not derived from residual distributions, prediction variance, or any calibration against historical coverage. The function is named `_confidence_half_width` and the docstring says "80% CI."

Additionally, Prophet's "95% confidence" bounds (`prophet_model.py:177`) are fabricated by scaling the 80% bounds: `lower_95 = yhat_lower * 0.95`, `upper_95 = yhat_upper * 1.05`. This has no statistical basis.

**Impact:** Users making risk-sensitive decisions (trading, reserve commitment) may overweight these bounds as statistically calibrated when they're arbitrary.

**Fix:** Rename to "Indicative Range" or "Heuristic Uncertainty." For a real improvement, compute empirical prediction intervals from backtest residuals -- the infrastructure for this exists (`_collect_backtest_residuals` at line 206).

---

### 1.8 `model_service.py` passes raw DataFrame to predictors -- CONFIRMED (P1)

**Where:** `models/model_service.py:148-164`, `_predict_from_trained`

**What:** The trained-model path passes `demand_df` directly to `predict_prophet`, `predict_arima`, and `predict_xgboost` without running feature engineering. XGBoost will silently zero-fill missing features. Prophet receives regressor values from the raw demand DataFrame (which may not contain weather columns).

**Mitigating factor:** This path is only hit when pre-trained models are loaded from disk (`MODEL_DIR`). The primary production path through `callbacks.py:_run_forecast_outlook` always runs `engineer_features` before prediction. However, the `model_service.py` path is the fallback for the diagnostics tab and `get_forecasts()` consumers.

**Fix:** Add a feature engineering gate in `_predict_from_trained`, or mark this path as deprecated in favor of the `_run_forecast_outlook` path.

---

### 1.9 ARIMA at 720h horizon degrades ensemble quality -- CONFIRMED (P1, new finding)

**Where:** `components/callbacks.py:555-681` (ARIMA in ensemble path)

**What:** SARIMAX with `pmdarima` auto-order is fundamentally unsuited for 720-hour predictions. The model's autoregressive structure compounds errors over long horizons, producing forecasts that flatten or diverge. Including ARIMA in the 30-day ensemble actively degrades forecast quality compared to XGBoost + Prophet only.

The ARIMA path also requires special handling -- NaN filling in exogenous columns (`callbacks.py:664-673`) -- that the other models don't need, indicating it's already fragile at this horizon.

**Fix:** Exclude ARIMA from the ensemble when `horizon_hours > 168`. This is a one-line gate in the ensemble's `missing` model list.

---

### 1.10 Model selection filter skips XGBoost -- CONFIRMED (P2)

**Where:** `models/model_service.py:143`

```python
if models_shown and name not in models_shown and name != "xgboost":
    continue
```

XGBoost always executes regardless of `models_shown` filter. If a user selects only Prophet, XGBoost still runs and contributes to the ensemble.

**Fix:** Remove the `and name != "xgboost"` exception, or document why XGBoost must always run (e.g., it's needed for feature importance).

---

## Part 2: Infrastructure & Performance (Code-Verified)

### 2.1 Precompute pipeline has no mention in original reviews -- NEW

**Where:** `precompute.py` (entire module)

**What:** The precompute pipeline is the most performance-critical component for user experience. It runs as a background daemon thread on startup, warming all caches. Three bugs were found and fixed (1.3, 1.4, 1.5 above). The original reviews don't mention this module at all.

Key facts:
- **3-tier cache:** Redis -> in-memory dict -> SQLite -> full compute
- **Phase ordering:** Data fetch -> XGBoost -> Prophet -> Ensemble -> Backtests -> Generation
- **OS-level locking:** `fcntl.flock` ensures only one gunicorn worker runs precompute
- **Memory management:** Model objects and feature DataFrames are freed between phases to stay within Cloud Run's 4Gi limit

**Remaining risks:**
- No timeout on individual phase execution -- a stuck API call blocks all subsequent phases
- Backtests run in a thread pool but share the GIL with the main request handler
- No health check endpoint to verify precompute completion status

---

### 2.2 Cloud Run container lifecycle vs precompute timing -- NEW

**Where:** Deployment config, `precompute.py`

**What:** Cloud Run can recycle containers. The precompute pipeline takes significant time (backtests for 8 regions x 3 horizons x walk-forward folds). If a container is recycled mid-precompute, all warm cache is lost and must restart. The `--no-cpu-throttling` and `--min-instances=1` flags mitigate this, but there's no checkpoint/resume mechanism.

**Recommendation:** Add phase-level checkpointing to SQLite so a recycled container can resume where it left off rather than restarting from scratch.

---

### 2.3 `populate_redis.py` only stores XGBoost -- CONFIRMED (informational)

**Where:** `populate_redis.py`

**What:** Redis is populated with 720 hours of XGBoost-only predictions per region. Prophet, ARIMA, and ensemble predictions are not stored in Redis. This means the Redis fast path only works for XGBoost and (after the fix) correctly returns `None` for other models.

**Recommendation:** If Redis is intended as the fastest cache tier, consider storing ensemble predictions there too -- especially now that ensemble precompute actually works (fix 1.3).

---

## Part 3: Design & UX (From Reviews, Assessed)

### 3.1 Tab rename set -- AGREED, actionable

**Source:** 360 Review Section 9

Current names and proposed replacements:

| Current | Proposed | Assessment |
|---------|----------|------------|
| Historical Demand | Load History | Clearer, action-oriented |
| Demand Forecast (Demand Outlook) | Forecast | Removes ambiguity between two similar names |
| Extreme Events | Alerts & Events | More intuitive for operators |
| Model Diagnostics | Model QA (Advanced) | Correctly signals this is a power-user tab |
| Scenario Simulator | What-If Planner | More accessible language |
| Present (meeting mode) | Presentation Mode | Self-explanatory |
| Backtest | Backtest | No change needed -- accurate and domain-standard |
| Generation & Net Load | Generation & Net Load | No change needed |

**Effort:** Low. Rename tab labels in `components/layout.py`, update `CLAUDE.md` module map.

---

### 3.2 Three-tier information architecture -- AGREED, high impact

**Source:** 360 Review Section 3A

Current flat 8-tab structure should be reorganized into tiers:

**Tier 1 -- Operations (default landing):**
- New Overview page: top risks, what changed, required actions
- Forecast (forward-looking demand)
- Alerts & Events

**Tier 2 -- Analysis:**
- Load History
- Generation & Net Load
- Weather Correlation
- What-If Planner

**Tier 3 -- Advanced (data science / model governance):**
- Backtest
- Model QA

This is the single highest-impact UX change available. It makes the product feel purpose-built for operators rather than exhaustive for everyone.

---

### 3.3 Personas should be decision-centric, not tab-centric -- AGREED

**Source:** 360 Review Sections 2.3, 4

Current state: `personas/config.py` defines personas with `default_tab`, `welcome_focus`, and insight emphasis. Switching personas changes insight text and welcome messages but not navigation structure, visible metrics, or available actions.

**What would make personas functional:**
- Each persona defines a set of priority KPIs shown on the Overview page
- Each persona defines which tier 2/3 tabs appear in their nav by default
- "What changed since last login" is scoped to the persona's focus area
- Insight text tone shifts per the review's recommendation: ops = directive, analytics = explanatory

**What is NOT needed:**
- RBAC / permission enforcement (this is a portfolio product, not multi-tenant SaaS)
- Approval chains or role-bound action controls
- Named individual personas with emoji (the current approach works, but the review is right that role labels are more appropriate for the target audience)

---

### 3.4 Placeholder states need explicit semantics -- AGREED

**Source:** 360 Review Section 5

Current state: KPI cards render `--` (em-dash) for missing data. This is ambiguous -- loading, unavailable, stale, and errored all look the same.

Sprint 5 shipped `error_handling.py` with `confidence_badge` and loading spinners, but these aren't used consistently across all KPI cards in `cards.py`.

**Fix:** Define four explicit states and render them distinctly:
- **Loading** -- spinner or skeleton
- **No data** -- "No data available" with context
- **Stale** -- show last value with "Updated 2h ago" timestamp
- **Error** -- "Data source unavailable" with fallback indicator

---

### 3.5 Overview landing page -- AGREED, new work needed

**Source:** 360 Review Section 3A

No overview page exists today. Users land on whichever tab their persona defaults to. An overview page should show:
- Current demand vs forecast delta for selected region
- Active alerts count and severity
- Top persona-specific KPIs (3-5 max)
- Data freshness status for all sources
- "Since last visit" change summary

This replaces the current welcome card pattern with a live operational summary.

---

### 3.6 Accessibility gaps -- AGREED (P1)

**Source:** 360 Review Section 7

Valid concerns:
- Typography baseline is small for control-room displays
- Red is overused outside alert contexts
- News ticker animation competes with data reading
- Semantic HTML landmarks (`header`, `nav`, `main`) should replace div-based layout

Existing mitigation: `accessibility.py` has colorblind-safe palettes and ARIA helpers. Meeting-ready mode strips chrome. But no WCAG AA audit has been done.

**Priority:** P1. These are usability issues, not decision-correctness issues.

---

### 3.7 Business impact translation -- DIRECTIONALLY CORRECT, scope-dependent

**Source:** 360 Review Section 3D, Merged Review Section 7

The review asks for cost/risk translation: "estimated imbalance exposure," "peak-risk cost band," "curtailment opportunity/penalty."

We already have `models/pricing.py` with merit-order tiers (base/moderate/exponential/emergency). Surfacing the pricing tier alongside the forecast peak would demonstrate business value awareness with minimal new code.

Full cost modeling (imbalance penalties, curtailment economics) is a significant scope expansion and should only be pursued if the portfolio use case demands it.

---

## Part 4: Findings From Reviews That Are Overstated or Wrong

### 4.1 "No decision workflow closure" rated P0 -- OVERSTATED

**Source:** 360 Review Section 2.2, Merged Review P0.2

The reviews describe incident management, alert triage workflows, shift handoffs, approval chains, and immutable decision logs as P0 production blockers. This conflates "analytics dashboard" with "operational workflow platform."

GridPulse is a forecasting and intelligence tool. It should inform decisions, not manage operational workflows. The appropriate scope is:
- **Lightweight annotation:** Save a scenario result with a note
- **Export with context:** Meeting-ready mode already exists; extend with "changes since last export"
- **Alert acknowledgment:** Let users dismiss/acknowledge alerts in the Events tab

Full incident management (assign, escalate, close, audit trail) is a different product category (ServiceNow, PagerDuty, custom SCADA).

---

### 4.2 Enterprise governance (SSO, RBAC, compliance) rated P0 -- NOT APPLICABLE at current stage

**Source:** 360 Review Sections 3E, 8; Merged Review Section 6

SSO/SAML, RBAC matrices, data retention disclosures, and compliance surfaces are enterprise procurement checklist items. They're true for every B2B SaaS product and not specific findings about GridPulse. At the current stage (portfolio demonstration / early deployment), these are future considerations, not production blockers.

---

### 4.3 "Product narrative drift" rated P0 -- OVERSTATED

**Source:** 360 Review Section 2.1, Merged Review P0.4

Calling tab naming and product positioning a "production blocker" inflates severity. The rename suggestions are good (see 3.1 above), but inconsistent terminology doesn't prevent users from making correct decisions. This is P2 polish work.

---

### 4.4 Backtest exogenous vintage -- ALREADY ADDRESSED

**Source:** Chart Review P1, Merged Review P1.4

The reviews claim backtest exogenous retrieval may not be vintage-correct. Recent branch merges added `exog_mode` with explicit `oracle_exog` and `forecast_exog` modes. `_build_forecast_exog_fold` constructs synthetic forecast-vintage exogenous data. This is more sophisticated than the review acknowledges.

**Remaining gap:** We don't store historical weather forecast issuances from Open-Meteo, so "forecast exogenous" backtests use reconstructed data rather than actual as-issued forecasts. This is an inherent limitation of the data source, not a code bug.

---

### 4.5 Diagnostics residual misalignment -- PLAUSIBLE but not demonstrated

**Source:** Chart Review P1

The review says diagnostics may compare actuals against predictions from different contexts. The diagnostics tab uses `_run_forecast_outlook` (same path as the forecast tab) and `_run_backtest_for_horizon` (same path as the backtest tab). Both paths produce aligned timestamp/prediction pairs. The review doesn't cite a specific misalignment case.

The `model_service.py` path (see 1.8) could produce misaligned predictions, but it's the legacy path, not what the diagnostics tab uses by default.

**Status:** Worth a targeted audit, but not a confirmed bug.

---

### 4.6 Scenario controls "don't materially influence demand" -- UNVERIFIED

**Source:** Chart Review P1

The review claims "several UI controls do not materially influence scenario demand as implied" but doesn't specify which controls. The scenario engine (`simulation/scenario_engine.py`) copies features, applies weather overrides, recomputes all derived features, and re-forecasts. Without specifics on which controls are inert, this isn't actionable.

**Status:** Needs the reviewer to identify which specific controls are the concern.

---

### 4.7 Model IO audit record -- ALREADY EXISTS

**Source:** Merged Review P0.1 item 4

`data/audit.py` ships `AuditRecord` (model version, data hash, feature hash, prediction timestamp range) and `AuditTrail` singleton. This was delivered in Sprint 5. The `audit-store` is wired into the layout and populated by `load_data` callbacks.

---

## Part 5: Prioritized Action Plan

### Immediate (model correctness)

| # | Item | Severity | Effort |
|---|------|----------|--------|
| 1 | Fix ensemble backtest leakage: use equal weights in backtests | P0 | Low |
| 2 | Fix Prophet regressor alignment: join on timestamp, not position | P0 | Medium |
| 3 | Fix 168h boundary: change `< 168` to `<= 168`, add boundary tests | P2 | Low |
| 4 | Exclude ARIMA from ensemble at horizons > 168h | P1 | Low |
| 5 | Relabel confidence bands as "Indicative Range" (not CI) | P1 | Low |

### Near-term (UX/design)

| # | Item | Impact | Effort |
|---|------|--------|--------|
| 6 | Rename tabs per Section 3.1 | Medium | Low |
| 7 | Build Overview landing page with top KPIs and alert summary | High | Medium |
| 8 | Restructure nav into 3 tiers (Ops / Analysis / Advanced) | High | Medium |
| 9 | Make personas decision-centric (priority KPIs, scoped nav) | High | Medium |
| 10 | Replace em-dash placeholders with explicit loading/error/stale states | Medium | Low |

### Later (infrastructure & enhancements)

| # | Item | Impact | Effort |
|---|------|--------|--------|
| 11 | Add ensemble predictions to Redis store | Medium | Low |
| 12 | Add precompute phase checkpointing for container recycle resilience | Medium | Medium |
| 13 | Build empirical prediction intervals from backtest residuals | High | Medium |
| 14 | Surface pricing tier alongside forecast peak | Medium | Low |
| 15 | Accessibility audit (contrast, typography, semantic landmarks) | Medium | Medium |
| 16 | Add precompute health check endpoint | Low | Low |
| 17 | Gate or deprecate `model_service.py` trained path | Low | Low |
| 18 | Lightweight scenario save/export with notes | Medium | Medium |

---

## Appendix: Review Source Quality Assessment

| Document | Strengths | Weaknesses |
|----------|-----------|------------|
| `CHART_TABLE_FORECAST_REVIEW.md` | Code-grounded, cites specific functions, correct priority ordering | Missed `precompute.py` entirely, some findings on secondary code path, didn't acknowledge `exog_mode` work |
| `GRIDPULSE_360_PRODUCT_REVIEW.md` | Good rename set, 3-tier IA, persona critique, accessibility callouts | Inflated severity on naming/governance, enterprise checklist is generic, no code verification |
| `GRIDPULSE_MERGED_360_PRODUCT_REVIEW.md` | Well-structured synthesis, explicit acceptance criteria | Inherited weaknesses from both sources, added timeline estimates that contradict project conventions, most "expanded" content is enterprise boilerplate |
| This document | Code-verified findings, includes bugs found and fixed, separates confirmed from speculative | Written by the person who fixed the bugs (potential bias toward infrastructure findings) |
