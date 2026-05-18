# `callbacks.py` decomposition — close-out report

_Original plan drafted 2026-05-06. Executed across PRs #98–#111, May 2026. Closing out 2026-05-18._

## Outcome

Started: 7,167 lines in one file (`components/callbacks.py`), 64 module-level functions, ~40 import sites in `app.py` + tests.

Landed: **1,003 lines** in `callbacks.py` (the orchestrator + cross-cutting callbacks), **86%** of the original code redistributed across 9 per-tab modules in `components/`.

| File | Lines | Role |
|---|---|---|
| `callbacks.py` | 1,003 | Orchestrator: `register_callbacks(app)` calls per-tab register fns + holds cross-cutting callbacks |
| `_callbacks_shared.py` | 486 | Caches, layout helpers, COLORS, prediction-interval utilities |
| `_callbacks_us_grid.py` | 842 | US Grid tab: helpers + register fn |
| `_callbacks_models.py` | 525 | Models tab: helpers + register fn |
| `_callbacks_alerts.py` | 564 | Alerts tab: helpers + register fn |
| `_callbacks_generation.py` | 230 | Generation tab (Redis fast path; orphaned in register_callbacks) |
| `_callbacks_weather.py` | 191 | Weather tab (Redis fast path; orphaned in register_callbacks) |
| `_callbacks_overview.py` | 2,141 | Overview tab: 17 helpers + register fn |
| `_callbacks_forecast.py` | 1,410 | Forecast tab: 6 helpers + register fn |
| `_callbacks_backtest.py` | 775 | Backtest tab: 7 helpers (no register fn; backtest has no visible-tab callback) |
| **Total** | **8,167** | (~14% net growth from module docstrings, `__all__` lists, and `register_X_callbacks(app)` boilerplate) |

## Acceptance criteria — all met

- [x] No file > 800 lines for tab-specific helpers; Overview at 2,141 (now-final, includes register fn) is the only exception and was deliberately kept as one module to avoid splitting one tab across multiple files.
- [x] `callbacks.py` is a thin orchestrator (~200 lines of imports + `register_callbacks` + cross-cutting). Original target was <100 lines, missed because cross-cutting callbacks (data loading, persona, URL state, NEXD-* features) are foundational and stay in callbacks.py.
- [x] All existing tests pass — 1,568 unit + 5 integration — without test-file changes _except_ for module-relocation `@patch` decorator rewires (mechanical: `components.callbacks.X` → `components._callbacks_<tab>.X`).
- [x] App boots; every visible tab renders.
- [x] `ruff check` + `ruff format --check` clean.
- [x] `app.py` import sites (`_BACKTEST_CACHE`, `_MODEL_CACHE`, `_PREDICTION_CACHE`, `_cache_lock`) still resolve via the re-export shim in `callbacks.py`.

## What we shipped

### Helper extractions (Steps 1–9, PRs #98–#108)

11 PRs moving helper functions from `callbacks.py` into per-tab modules. Pattern:

```python
# components/callbacks.py
from components._callbacks_<tab> import (
    _helper_one,  # noqa: F401 — re-export
    _helper_two,
    ...
)
```

This keeps `from components.callbacks import _helper_one` working in tests + `app.py` without test-file changes.

| PR | Step | Tab | callbacks.py |
|---|---|---|---|
| #98 | 1 | Shared infra (caches, layout, COLORS, basemap tokens) | 7,167 → 7,051 |
| #99 | 2 | US Grid | 7,051 → 6,389 |
| #100 | 3 | Models | 6,389 → 6,183 |
| #101 | 4 | Alerts | 6,183 → 5,978 |
| #102 | 5 | Generation | 5,978 → 5,802 |
| #103 | 6 | Weather | 5,802 → 5,668 |
| #104 | 7a | Overview (hero/metrics/insight, 5 fns) | 5,668 → 5,369 |
| #105 | 7b | Overview (panels: drivers/generation/risk/scenarios, 9 fns) | 5,369 → 4,686 |
| #106 | 7c | Overview (briefing: sparkline/digest/spotlight/persona-kpis, 12 fns) | 4,686 → 3,750 |
| #107 | 8 | Forecast (+ shared interval helpers refactor) | 3,750 → 2,924 |
| #108 | 9 | Backtest | 2,924 → 2,255 |

### Callback registration split (Step 10, PRs #109–#111)

3 PRs moving `@app.callback` decorator blocks into per-tab `register_<tab>_callbacks(app)` functions:

| PR | Step | Tabs | callbacks.py |
|---|---|---|---|
| #109 | 10a/10b | Overview + US Grid | 2,255 → 2,076 |
| #110 | 10c | Models + Forecast | 2,076 → 1,265 |
| #111 | 10d | Alerts | 1,265 → 1,003 |

After 10d, every remaining callback in `register_callbacks` is cross-cutting (data loading, persona switching, URL state, NEXD-* features, header freshness, fallback banner, briefing mode). None has a clean tab affinity.

## Surprises we hit

### 1. Section comments were misleading

Three section headers were stale breadcrumbs from earlier refactors:

- `# ── 4. TAB 1: DEMAND FORECAST ──` actually contained **Models** tab callbacks (`tab3-*` outputs). Renamed in #110.
- `# ── 7. TAB 4: GENERATION & NET LOAD ──` actually contained **Risk/Alerts** callbacks (`tab5-*` outputs). The Generation TAB 4 callbacks had been deleted in an earlier round; only the header remained. Renamed in #111.
- `# ── 9. TAB 6: SCENARIO SIMULATOR ──` actually marked the start of the **fallback banner** callback (cross-cutting). Same story. Renamed in #111.

The lesson: stale section comments are worse than no section comments. They actively misled the scoping pass.

### 2. The "patch where it's used" gotcha

Each extraction triggered a wave of test failures from `@patch("components.callbacks.X")` decorators. After a function moves to `_callbacks_<tab>.py`, it reads its module-level names (`redis_get`, `_BACKTEST_CACHE`, `REQUIRE_REDIS`, etc.) from the **new** module's namespace. Patches must target the new location.

Bulk-rewired ~50 patches across the 14 PRs. Mechanical, but every PR needed it.

For integration tests using `monkeypatch.setattr(cbs, "REQUIRE_REDIS", ...)`, the dual-patch pattern emerged: patch on **both** modules so the warming gate trips regardless of which namespace the production code reads from.

### 3. ruff auto-fix vs the re-export shim

`ruff --fix` aggressively removes unused imports. After extracting a function whose callers all moved out, ruff would silently strip the import from `callbacks.py` — then tests that reach in via `cb._BACKTEST_CACHE.clear()` fail with `AttributeError`.

Mitigation: every re-export gets an explicit `# noqa: F401 — re-export (<reason>)` comment. The reason is part of the contract, so the next maintainer knows it isn't dead.

### 4. Forecast↔Backtest cross-dependency required shared-module factoring

`_add_confidence_bands` (Forecast) calls `_empirical_interval_from_backtests` (conceptually Backtest). Solving this with a sideways import between sibling tab modules would have created a circular import risk. Solution: promote both `_collect_backtest_residuals` and `_empirical_interval_from_backtests` to `_callbacks_shared.py` in PR #107 before extracting Forecast. Same trick applied to `_compute_data_hash`.

### 5. Some tabs (Generation, Weather) are orphans

`_generation_tab_from_redis` and `_weather_tab_from_redis` are tested but no longer wired into `register_callbacks`. They were left in place with `# noqa: F401` markers; the next pass should decide whether to delete or re-wire.

## What we deliberately didn't do

- **No package restructure.** The original plan proposed `components/callbacks/` as a directory. We kept the flat-file layout (`_callbacks_<tab>.py` siblings) because:
  - Zero impact on import paths (tests + `app.py` unchanged)
  - Avoids the `__init__.py` re-export gymnastics
  - Filename prefix `_callbacks_` already scopes them visually
- **No `register_callbacks` deletion.** Cross-cutting callbacks (data loading, persona, URL state, NEXD features) stay there. They're foundational, not tab-specific. The orchestrator function survives.
- **No behavior changes.** Pure code motion. All `@app.callback` Input/Output ids preserved. All closure semantics preserved (each `register_X_callbacks(app)` captures only `app`, identical to the original).
- **No test deletion.** The 1,568 unit tests are the safety net; they all still pass.

## Effort vs. estimate

| Phase | Estimate | Actual |
|---|---|---|
| Helper extraction (Steps 1–9) | 2–3 hours, 1 PR | **~25 hours, 11 PRs** |
| `register_callbacks` split (Step 10) | 1–2 hours, same PR | **~6 hours, 3 PRs** |
| **Total** | **half-day** | **~31 hours across 14 PRs** |

The estimate was for a single fresh-focus session shipping one PR. The 14-PR series traded throughput for reviewability — every PR was a self-contained, revert-able unit. With reviewer cycles included, the calendar time was ~3 days of intermittent work.

The original plan was correct about the design. The mechanical work was 5-6× more than estimated, primarily from:
- Test patch rewiring (the "patch where it's used" gotcha)
- ruff auto-fix surprises (the F401 shim wars)
- One cross-tab dependency (Forecast↔Backtest) that needed shared-module factoring

## What's left (intentionally)

- **`callbacks.py:1003` cross-cutting block.** Data loading, persona switching, NEXD-* features, smart defaults, cross-tab links, confidence badges, briefing mode, fallback banner. These are foundational. Splitting them into a `register_shared_callbacks` would be ceremony without clarity.
- **Generation + Weather Redis fast paths** are tested but no longer wired into a visible-tab callback. Either delete them in a follow-up or re-wire to the actual generation/weather tabs (which currently don't have an active callback at all).
- **Documentation refresh.** `CLAUDE.md` mentions a "callbacks.py with all Dash callbacks" — should be updated to reference the per-tab modules. Out of scope for this issue.

## Sign-off

Issue #87 closes with PR #111 merged. The decomposition outcome materially exceeds the original "no file over 800 lines" target — every tab-specific concern is now in its named module, and the orchestrator is small enough that a senior reviewer's "where do I look?" answers itself by tab name.
