# Test Pyramid Definition (Backlog H3)

## Overview

Test strategy for GridPulse.
Defines coverage targets, test scope, and critical user flows.

## Test Pyramid

```
         ╱╲
        ╱E2E╲           ~15% — Critical user flows
       ╱──────╲
      ╱ Integr. ╲       ~30% — Data pipeline, callback contracts
     ╱────────────╲
    ╱    Unit       ╲    ~55% — Pure functions, models, config
   ╱──────────────────╲
```

> **Measured 2026-08-10, against the shape above: unit 3,034 (93.0%),
> integration 204 (6.3%), e2e 23 (0.7%) of 3,261 collected.** At the intended
> 15%, the e2e tier would hold ~490 tests; it holds 23, in one file. The
> percentages above are the *target*, not a description of the tree — stated
> here so a reader does not mistake one for the other. Reconciling them is
> [#399](https://github.com/kristenmartino/gridpulse/issues/399): either rebalance the suite or restate the
> target. This note takes no position on which.

## Coverage Targets

| Layer        | Target | Scope                                  | Speed   |
|-------------|--------|----------------------------------------|---------|
| Unit         | 80%+   | Pure functions, models, config, utils  | < 10s   |
| Integration  | 70%+   | Data pipeline, callback contracts      | < 30s   |
| E2E          | 100%   | Critical user flows (5 visible tabs)   | < 60s   |

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

## E2E Tests (`tests/e2e/`)

> **This tier does not currently test end to end, and the table below says so
> (#399).** Everything in `tests/e2e/` calls functions directly — no browser,
> no HTTP client, no Dash test server, and no callbacks fire. There is no
> `dash.testing`, `dash_duo` or Flask `test_client` anywhere under `tests/`.
> The rows are described as what they actually assert; #399 tracks whether to
> add a real driver or rename the tier.

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

# E2E (full)
pytest tests/e2e/ -v --timeout=60

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
1. All unit, integration, and E2E tests pass (one instrumented run)
2. Total coverage ≥ 70% over `data/models/simulation/personas/components`
3. Changed-lines coverage reported by `diff-cover` (advisory — see PR comment)
4. No hardcoded secrets detected
5. `ruff check` and `ruff format --check` pass
6. MAPE thresholds met (H2) — rollback grade blocks deploy

Not a gate: mutation score. It runs weekly and on demand, and is advisory —
see `docs/TEST_QUALITY.md` for the conditions under which that changes.

Coverage artefacts on every run: `htmlcov/` (line-level HTML), `coverage.xml`,
`junit.xml`, and a PR comment with the per-file table.
