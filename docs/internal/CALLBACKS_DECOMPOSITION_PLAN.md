# `callbacks.py` decomposition plan

_Drafted 2026-05-06. Targets GitHub issue #87. Execution scoped for a single fresh-focus session._

## Why this is queued, not shipped

Tonight's session shipped 13 PRs across V3.η, real-metrics, training parallelism, portfolio updates. Attempting a 7,167-line refactor at end-of-day with 40 import-site dependencies is the move that ships a subtle bug nobody catches for a week. Deferring to fresh focus is the senior call. This document captures the design so tomorrow's session is execution, not planning.

## Current state

- `components/callbacks.py` — 7,167 lines
- `register_callbacks(app)` — 2,013 lines (lines 2089–4100ish)
- 64 module-level functions
- ~40 symbols imported by `app.py` and `tests/`

## Target structure

A `components/callbacks/` package replaces the single file. Public import surface unchanged via re-export in `__init__.py`.

```
components/
  callbacks/
    __init__.py        # re-exports everything for backward compat
                       # owns top-level register_callbacks(app) orchestrator
    shared.py          # caches, constants, color tokens, _layout helper,
                       # _compute_data_hash, _is_real_positive, _empty_figure,
                       # _format_metric, _latest_real_demand, _MAP_* constants,
                       # PLOT_LAYOUT, COLORS, _EIA_FUEL_MAP
    overview.py        # _build_overview_* helpers + register_overview_callbacks
    us_grid.py         # _build_us_grid_*, _collect_us_grid_region_data,
                       # _load_ba_polygons + register_us_grid_callbacks
    forecast.py        # _run_forecast_outlook (367 lines!), _confidence_*,
                       # _collect_backtest_residuals, _empirical_interval_*,
                       # _build_forecast_exog_fold, _add_confidence_bands,
                       # _add_trailing_actuals, _create_future_features,
                       # _outlook_tab_from_redis + register_forecast_callbacks
    backtest.py        # _run_backtest_for_horizon (308 lines),
                       # _predict_single_fold, _ensemble_fold,
                       # _backtest_tab_from_redis
                       # (hidden tab but still called via v1 fallback)
    models.py          # _models_tab_from_redis (206 lines),
                       # _get_feature_importance
    alerts.py          # _alerts_tab_from_redis (180 lines)
    generation.py      # _fetch_generation_cached, _generation_tab_from_redis
                       # (hidden tab; still called via v1 fallback)
    weather.py         # _weather_tab_from_redis (hidden tab)
    data.py            # _load_data_from_redis (shared data path)
```

## File-allocation principles

- **Shared.py is for code with no tab affinity** — caches, layout helpers, evaluation utilities, design tokens. If a helper is used by exactly one tab, it goes in that tab's module.
- **Hidden-tab modules still exist** — `backtest.py`, `generation.py`, `weather.py`, `scenarios.py` (if present). The tabs are removed from the visible IA per V2.1, but the logic is still called via the v1-inline-compute fallback path and is tested.
- **Each tab module exports a `register_<tab>_callbacks(app)`** — the per-tab portion of the current 2K-line `register_callbacks`. The top-level orchestrator calls each in sequence.
- **No tab module imports from another tab module.** Cross-tab needs go through `shared.py`. Prevents circular imports.
- **`__init__.py` re-exports every public symbol** that the test suite or `app.py` imports today. Backward-compat shim. No test or app code needs to change.

## Risks to plan around

1. **Circular imports.** Risk: tab module A imports from tab module B because they share a helper. Mitigation: anything shared between tabs goes in `shared.py`. The discipline: if you're about to write `from components.callbacks.overview import X` inside `forecast.py`, the answer is no — promote X to `shared.py` instead.

2. **The 2,013-line `register_callbacks`.** Splitting this is the highest-risk step. The callbacks reference module-level helpers; if those helpers moved to per-tab modules, the imports inside the closure functions need updating too. Mitigation: do this AFTER the helper-extraction step, as the second commit in the PR. Each step's tests run independently.

3. **`from components.callbacks import X` import sites.** 40 known sites. Risk: a private helper that was incidentally importable is no longer re-exported. Mitigation: `__init__.py` does `from .shared import *; from .overview import *; ...` so all public symbols flow through. Then run the full test suite — any `ImportError` is a missing re-export.

4. **`app.py:179` imports `_BACKTEST_CACHE, _MODEL_CACHE, _PREDICTION_CACHE` for the `/health` endpoint.** These caches must live in `shared.py` and be re-exported. They're module-level singletons used across tabs.

## Execution checklist

1. Create `components/callbacks/` directory with empty `__init__.py`
2. Create `shared.py`. Move:
   - All module-level constants (`_CACHE_VERSION`, caches, `PLOT_LAYOUT`, `COLORS`, `_EIA_FUEL_MAP`, `BACKTEST_EXOG_MODES`, `_STRESS_RELIABLE_CEILING`, `_MAP_*`)
   - Cross-cutting helpers (`_layout`, `_compute_data_hash`, `_empty_figure`, `_is_real_positive`, `_format_metric`, `_latest_real_demand`)
   - `_cache_lock`
3. `__init__.py` does `from .shared import *`. Run tests. Fix any missing re-exports.
4. Create `us_grid.py`. Move all `_build_us_grid_*` helpers + `_collect_us_grid_region_data` + `_load_ba_polygons` + `_build_us_grid_choropleth`. Inside `us_grid.py`, replace any reference to a moved helper with `from .shared import X`.
5. `__init__.py` adds `from .us_grid import *`. Run tests.
6. Repeat for each remaining tab: `models`, `alerts`, `overview`, `forecast`, `backtest`, `generation`, `weather`, `data`.
7. After all helpers are extracted, callbacks.py is reduced to just `register_callbacks(app)` (~2K lines). Now split that function:
   - Each tab gets a `register_<tab>_callbacks(app)` function in its module
   - The top-level `register_callbacks(app)` in `__init__.py` calls each in sequence
   - Inner `@app.callback` decorators move with their tab
8. Delete the original `components/callbacks.py` file (its contents are now in the package).
9. Run full test suite. Target: 1688 passing, no behavioral changes.
10. Run `app.py` locally, click each visible tab, verify no console errors.

## Acceptance

- [ ] No file in `components/callbacks/` over ~800 lines
- [ ] `__init__.py` is a thin orchestrator < 100 lines
- [ ] All existing tests pass without modification (no `from components.callbacks import` line is updated in tests)
- [ ] App boots locally, every visible tab renders, no console errors
- [ ] `ruff check` + `ruff format --check` clean
- [ ] `app.py:179` (`from components.callbacks import _BACKTEST_CACHE, _MODEL_CACHE, _PREDICTION_CACHE`) still works without modification

## Estimated effort

- Helper extraction: 2–3 hours of careful work
- `register_callbacks` split: 1–2 hours
- Test + iterate: 1 hour
- **Total: half-day of focused work**, single PR

## What's intentionally out of scope

- **No behavior changes.** Pure code motion.
- **No new tests.** The existing 1688 tests are the safety net.
- **No callback signature changes.** All `@app.callback` decorators preserved.
- **No `dash_bootstrap_components` or `dash` version bumps.**
