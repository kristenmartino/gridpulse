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
    ┌──────────────┐  smoke        23   (0.7%)  — does it construct?
    ├──────────────┤  integration  204  (6.3%)  — do the seams line up?
    │              │
    │     unit     │              3,034 (93.0%) — is the logic right?
    └──────────────┘
```

*Counts measured 2026-08-10, 3,261 collected. They move most weeks —
`pytest tests/ --collect-only -q` is the source, not this file.*

## Coverage Targets

| Layer        | Target | Scope                                       | Speed   |
|-------------|--------|---------------------------------------------|---------|
| Unit         | 80%+   | Pure functions, models, config, utils       | < 45s   |
| Integration  | 70%+   | Data pipeline, callback contracts           | < 60s   |
| Smoke        | every tab + card builder | Constructs without raising; callback-id contract | < 5s |

*No percentage-of-suite target. The previous 55/30/15 was never pursued and
its only effect was to make the tree look broken against a number nobody was
working toward.*

## What is not covered

Stated plainly, because the rename removes the word that used to imply
otherwise:

- **No browser or rendered-DOM assertion.** Nothing verifies a tab paints,
  a chart draws, or CSS applies. Visual regressions are caught by looking.
- **No callback execution.** No `dash.testing`, no `dash_duo`, no Flask
  `test_client` anywhere under `tests/`. A callback that raises at runtime
  passes this suite — the smoke tier's callback-**id** contract is the
  closest guard, and it only catches an Output whose id has gone missing.
- **No real user flow.** Nothing switches a region, changes a persona or
  moves a slider and asserts what follows.

If an end-to-end tier is ever wanted, it is a **new tier** with a real
driver — not a rename back.

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
| test_callback_data_flow.py  | JSON roundtrip, timestamp alignment    |
| test_data_pipeline.py       | EIA → preprocess → features → model    |
| test_infrastructure.py      | Docker, logging, secrets, health       |

### Key Contracts Tested
- Demand JSON preserves timestamps after roundtrip
- Weather + demand merge produces expected column set
- All 51 balancing authorities in `REGION_COORDINATES` produce valid merged data
- Callback outputs match layout IDs
- Pipeline logger records all steps

## Smoke Tests (`tests/smoke/`)

> **This tier does not test end to end — hence the name (#399).**
> Everything in `tests/smoke/` calls functions directly — no browser,
> no HTTP client, no Dash test server, and no callbacks fire. There is no
> `dash.testing`, `dash_duo` or Flask `test_client` anywhere under `tests/`.
> The rows below describe what they actually assert.

### What the tier actually asserts

| Flow | What it really does | Tests |
|------|-------------|-------|
| Tab Render | Calls each of the 5 tabs' `layout()` and asserts it constructs. **US Grid was missing entirely until #399** and now also asserts every id its callbacks target is present — a renamed id orphans a callback silently, since Dash does not raise on an Output with no matching layout id. | test_dashboard_render.py |
| Persona Switch | Asserts the 4 persona configs produce welcome cards and carry valid default tabs. Does not switch anything. | test_dashboard_render.py |
| Region Switch | Asserts `generate_demo_demand` / `_weather` / `_generation` return well-formed frames for all 51 BAs. **This is the synthetic demo generator, not the real data path** — the row previously read "All 51 BAs load data successfully", which is not what it checks. | test_dashboard_render.py |
| Scenario Presets | Asserts the 6 presets in `simulation/presets.py` are well-formed. That module has **no live UI importer** — the shipped Scenarios panel uses the linear heuristic — so this covers code the product does not run. | test_dashboard_render.py |
| Card Components | Calls the KPI / alert / welcome / chart card builders and asserts they construct. | test_dashboard_render.py |

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

Fleet logic score is **88.4%** across the seven scoped modules; the per-module
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
