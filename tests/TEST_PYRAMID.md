# Test Strategy (Backlog H3)

## Overview

Test strategy for GridPulse — the tiers, what each is for, and what is
deliberately **not** covered.

> **Renamed 2026-08-10 (#399).** This file used to describe a 55/30/15
> unit/integration/E2E pyramid and a `tests/e2e/` tier. The tree was
> **93/6/1**, and the "E2E" tier ran no browser, no HTTP client and no
> callbacks — it called `layout()` and asserted the result was not `None`.
> The gap was closed by renaming the tier to what it is rather than by
> claiming the shape it was not: `tests/e2e/` → **`tests/smoke/`**.
>
> **GridPulse therefore has no end-to-end tier.** That is a deliberate
> position, recorded here so it stays a decision rather than an oversight —
> see "What is not covered" below.

## Shape

This suite is **unit-dominant on purpose**, not a pyramid that failed to
fill out. The product's risk concentrates in data and model correctness —
gap policy, feature leakage, recursive inference, ensemble weighting,
freshness gating — which are pure functions and belong in unit tests. The UI
is a thin read-only render over Redis, so breadth of interaction testing buys
comparatively little.

```
    ┌──────────────┐  smoke        24   (0.7%)  — does it construct?
    ├──────────────┤  integration  215  (6.5%)  — do the seams line up?
    │              │
    │     unit     │              3,044 (92.7%) — is the logic right?
    └──────────────┘
```

*Counts measured 2026-08-10, 3,283 collected. They move most weeks —
`pytest tests/ --collect-only -q` is the source, not this file.*

## Coverage Targets

| Layer        | Coverage target | Scope                                  | Speed budget | Last measured |
|-------------|--------|---------------------------------------------|---------|---------|
| Unit         | 80%+   | Pure functions, models, config, utils       | < 60s   | **45–47s** |
| Integration  | 70%+   | Data pipeline, callback dispatch, wiring    | < 30s   | **25.0s** |
| Smoke        | every tab + card builder | Constructs without raising | < 5s | **1.0s** |

*Measured 2026-08-10 at `5fed8ee`, `pytest tests/<tier> -q`. Re-measure rather
than trust the column.*

*No percentage-of-suite target. The previous 55/30/15 was never pursued and
its only effect was to make the tree look broken against a number nobody was
working toward.*

**On the speed budgets (#399 finding 2).** Both bottom-tier budgets were
breached when the issue was filed — unit **38.5s against < 10s**, integration
**43.5s against < 30s**. The unit budget is restated to < 60s rather than
defended: 3,041 tests cannot run in 10s, the number was never derived from
anything, and a budget nobody can meet is not a budget. Integration moved the
other way and now **meets its original < 30s** unchanged, so that one stands.
The measured column exists because the original failure was not the budgets —
it was that nothing re-measured them, so they drifted from reality in silence
for months.

## What is not covered

Stated plainly, because the rename removes the word that used to imply
otherwise:

- **No browser or rendered-DOM assertion.** Nothing verifies a tab paints,
  a chart draws, or CSS applies. Visual regressions are caught by looking.
- **No clientside callbacks.** `app.clientside_callback` runs in the
  browser's JS engine, so the driver below cannot reach it.
- **No user *input* — only its effect.** The driver sends the values a
  component would have produced; nothing clicks, types or drags. A control
  wired to the wrong callback still dispatches correctly when asked
  directly.

If a browser tier is ever wanted, it is a **new tier** — not a rename back.

### What changed 2026-08-10 (second half of #399)

The three bullets above used to read "no callback execution" and "no real
user flow". Both are now false, and the entry is kept rather than quietly
edited because the claim was published:
`tests/integration/test_callback_driver.py` dispatches real
`POST /_dash-update-component` requests through `app.server.test_client()`,
so callbacks execute with Dash's own argument binding and real
serialisation. A region switch, a bookmark restore and the `REQUIRE_REDIS`
warming gate are asserted through it.

## Unit Tests (`tests/unit/`)

Cover individual functions without external dependencies.

### Module Targets

| Module                 | File                      | Target | Critical Functions |
|-----------------------|---------------------------|--------|-------------------|
| Config                | test_config.py            | 95%    | Region lookup, staleness thresholds |
| Feature Engineering   | test_feature_engineering.py| 85%   | CDD/HDD, wind power, solar CF |
| Ensemble              | test_ensemble.py          | 90%    | Weight computation, combination |
| Evaluation            | test_evaluation.py        | 90%    | MAPE, RMSE, MAE, R² |
| Model Service         | test_model_service.py     | 80%    | Forecast generation, metrics |
| Preprocessing         | test_preprocessing.py     | 85%    | Merge, gap handling, validation |
| Pricing               | test_pricing.py           | 85%    | Price impact, reserve margin |
| Personas              | test_personas.py          | 90%    | Persona config, welcome cards |
| Scenarios             | test_scenario.py          | 85%    | Presets, derived features |
| Cache                 | test_cache.py             | 80%    | Set/get, TTL, staleness |
| Sprint 3 (A11y)       | test_sprint3.py           | 80%    | WCAG, error handling, observability |
| Sprint 4 (Ops)        | test_sprint4.py           | 90%    | Persona tabs, KPI contracts |
| Sprint 4 Features     | test_sprint4_features.py  | 95%    | All 7 backlog items |
| Sprint 5 (Trust)      | test_sprint5.py           | 90%    | Audit, pipeline, confidence, meeting |

## Integration Tests (`tests/integration/`)

Test data flow between components.

### Scope

| Test File                   | What It Tests                          |
|-----------------------------|----------------------------------------|
| test_callback_driver.py     | **Real callback execution** (see below) |
| test_callback_data_flow.py  | JSON roundtrip, timestamp alignment    |
| test_callbacks_redis_only.py| `REQUIRE_REDIS` warming gates          |
| test_data_pipeline.py       | EIA → preprocess → features → model    |
| test_infrastructure.py      | Docker, logging, secrets, health       |

### Key Contracts Tested
- Demand JSON preserves timestamps after roundtrip
- Weather + demand merge produces expected column set
- All 51 balancing authorities in `REGION_COORDINATES` produce valid merged data
- Every callback `Output` id exists in the layout — all 77, read from
  `app.callback_map`
- Pipeline logger records all steps

### The callback driver

`dash_driver.py` reads Dash's callback registry, builds a well-formed
`POST /_dash-update-component`, and sends it through
`app.server.test_client()`. Dash resolves the callback and invokes the real
function, so the test sees the real return value after real serialisation.
No browser and no new dependency — `dash` and `flask` are already required.

What it catches that no other tier does:

- an `Output` id that no longer exists in the layout (Dash does not raise —
  the panel just renders empty forever)
- an `Input` list wired in the wrong order, which a direct call like
  `load_data("ERCOT", 0)` reproduces rather than detects
- a return value that cannot be serialised to the browser

```bash
pytest tests/integration/test_callback_driver.py -v
```

## Smoke Tests (`tests/smoke/`)

> **This tier does not test end to end — hence the name (#399).**
> Everything in `tests/smoke/` calls functions directly — no browser, no
> HTTP client, no Dash test server, and no callbacks fire. Callbacks are
> executed one tier up, by `tests/integration/test_callback_driver.py`;
> this tier stays below it deliberately, because it needs no app import and
> so still gives a sub-second signal on a broken layout function.
> The rows below describe what they actually assert.

### What the tier actually asserts

| Flow | What it really does | Tests |
|------|-------------|-------|
| Tab Render | Calls each of the 5 tabs' `layout()` and asserts it constructs. **US Grid was missing entirely until #399.** The id contract that briefly lived here moved to the callback driver, which reads it from Dash instead of by regex. | test_dashboard_smoke.py |
| Persona Switch | Asserts the 4 persona configs produce welcome cards and carry valid default tabs. Does not switch anything. | test_dashboard_smoke.py |
| Region Switch | Asserts `generate_demo_demand` / `_weather` / `_generation` return well-formed frames for all 51 BAs. **This is the synthetic demo generator, not the real data path** — the row previously read "All 51 BAs load data successfully", which is not what it checks. | test_dashboard_smoke.py |
| Scenario Presets | Asserts the 6 presets in `simulation/presets.py` are well-formed. That module has **no live UI importer** — the shipped Scenarios panel uses the linear heuristic — so this covers code the product does not run. | test_dashboard_smoke.py |
| Card Components | Calls the KPI / alert / welcome / chart card builders and asserts they construct. | test_dashboard_smoke.py |

## Test Naming Convention

```
test_{module}_{behavior}_{condition}
```

Examples:
- `test_merge_demand_weather_basic` — happy path merge
- `test_mape_all_zeros_returns_inf` — edge case
- `test_persona_tab_disabled_loop` — interaction between features

## Running Tests

```bash
# All tests
pytest tests/ -v

# Unit only (fast feedback)
pytest tests/unit/ -v --timeout=10

# Integration (medium)
pytest tests/integration/ -v --timeout=30

# Smoke (fast)
pytest tests/smoke/ -v

# Coverage report
pytest tests/ --cov=. --cov-report=html --cov-report=term-missing

# Sprint 5 only
pytest tests/unit/test_sprint5.py -v
```

## Fixtures

Common fixtures are defined in `tests/conftest.py`:

| Fixture | Provides |
|---------|----------|
| `tmp_cache` | Temporary SQLite cache |
| `sample_demand_df` | 168-row demand DataFrame |
| `sample_weather_df` | 168-row weather DataFrame |
| `mock_eia_response` | Mocked EIA API response |
| `mock_weather_response` | Mocked Open-Meteo response |
| `mock_noaa_alerts_response` | Mocked NOAA alerts |
| `feature_df` | Merged + engineered features |

## Is the suite any good? (mutation testing)

Everything above measures **how much** is tested. None of it measures whether
the tests would **notice** if the code were wrong — and a suite can be 97%
covered and still assert nothing that matters.

Mutation testing answers the second question by breaking the code on purpose
and checking whether the suite fails. A **survivor** is a mutant nothing
caught: a line that can be broken with CI green.

```bash
python scripts/mutation_test.py                        # all targets
python scripts/mutation_test.py --module models/skill.py
python scripts/mutation_test.py --skip-run             # re-report only
```

Scope is the seven decision-critical modules in `[tool.mutmut] only_mutate`
(pyproject.toml) — pure logic where a silently wrong number reaches a published
result. Measured baselines, the survivor ledger, and the rules for reading a
mutation score are in [`docs/TEST_QUALITY.md`](../docs/TEST_QUALITY.md).

The relationship to coverage, concretely: `models/ensemble.py` is **85%**
line-covered and scored **61.6%** on behavioural mutants. Two rounds of pinning
took that to **91.6%** while coverage did not move at all — the two instruments
measure different things. Coverage is a floor, not a verdict.

Fleet logic score is **88.7%** across the seven scoped modules; the per-module
table and the adjudicated survivor ledger are in
[`docs/TEST_QUALITY.md`](../docs/TEST_QUALITY.md).

## Quality Gates (CI)

Before merge:
1. All unit, integration, and smoke tests pass (one instrumented run)
2. Total coverage ≥ 70% over `data/models/simulation/personas/components`
3. Changed-lines coverage reported by `diff-cover` (advisory — see PR comment)
4. No hardcoded secrets detected
5. `ruff check` and `ruff format --check` pass
6. MAPE thresholds met (H2) — rollback grade blocks deploy

Not a gate: mutation score. It runs weekly and on demand, and is advisory —
see `docs/TEST_QUALITY.md` for the conditions under which that changes.

Coverage artefacts on every run: `htmlcov/` (line-level HTML), `coverage.xml`,
`junit.xml`, and a PR comment with the per-file table.
