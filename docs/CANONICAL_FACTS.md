# Canonical Facts — GridPulse

> Cross-doc fact source. When `README.md`, `PRD.md`, `TECHNICAL_SPEC.md`,
> case-study copy in `portfolio-v2`, or interview material need to cite
> a number/list/name, **they reference this file** rather than restating.
> When a value here changes, update one place and consumers pick it up.
>
> Verified: **2026-05-20** (PR-C1).
> Next planned re-verification: per the CLAUDE.md end-of-PR check, any
> PR that moves a value here updates this file in the same commit.

## Scale

| Fact | Value | Source of truth |
|---|---|---|
| Balancing authorities covered | **51** | [`config.REGION_COORDINATES`](../config.py) |
| Demand coverage (contiguous US lower-48) | **~100%** | derived from EIA-930 demand vs. our covered set |
| EIA-930 total BAs (contiguous US) | **63** | [Aug 2025 Federal Register PRA renewal](https://www.govinfo.gov/content/pkg/FR-2025-08-28/pdf/2025-16450.pdf) |
| BA-count coverage of EIA-930 | **~81%** (51 of 63) | derived |
| Expansion history | Original 8 → V1.α +8 → V3.ζ +35 | [`docs/internal/NEXT_UP.md`](internal/NEXT_UP.md) |

## Models

| Fact | Value | Source of truth |
|---|---|---|
| Base ML models | **3**: Prophet, SARIMAX, XGBoost | [`models/`](../models/) |
| Ensemble method | Sharpened inverse-MAPE — `weight_i ∝ (1/MAPE_i)³` normalized (ADR-004, `ENSEMBLE_WEIGHT_EXPONENT=3`, #181) | [`models/ensemble.py`](../models/ensemble.py) |
| User-selectable forecasts in UI | **4**: XGBoost, Prophet, ARIMA, Ensemble | [`components/_callbacks_forecast.py`](../components/_callbacks_forecast.py) |
| Total engineered features | **49** (17 raw weather + 32 derived) | [`data/feature_engineering.py`](../data/feature_engineering.py) |
| Forecast horizons | 24h, 7d, 30d (UI selectable) | [`components/_callbacks_forecast.py`](../components/_callbacks_forecast.py) |
| Confidence interval | 80% empirical, last 120h calibration window | [`models/evaluation.py`](../models/evaluation.py) |

## Architecture

| Fact | Value | Source of truth |
|---|---|---|
| Web tier | Cloud Run Service `gridpulse` | [`.github/workflows/deploy-prod.yml`](../.github/workflows/deploy-prod.yml) |
| Scheduled work | 2 Cloud Run Jobs (`-scoring-job` hourly, `-training-job` daily 04:00 UTC) | [`docs/SCHEDULED_JOBS.md`](SCHEDULED_JOBS.md) |
| Scoring job runtime | ~855s baseline, 1083–1333s under elevated upstream latency, for 51 BAs at `PRECOMPUTE_MAX_WORKERS=4`; 30 min (1800s) Cloud Run task timeout, guarded by a runtime-creep alert at 70% ([#171](https://github.com/kristenmartino/gridpulse/issues/171)) | 855s observed 2026-06-01, 1083–1333s the three post-incident runs; timeout bumped from 15 min after 4 consecutive timeouts; creep guardrail added 2026-07-04 |
| Training job runtime | ~3 hours for 51 BAs (5h Cloud Run task timeout) | observed; bumped from 2h after timeouts |
| Model storage | `gs://nextera-portfolio-energy-cache/models/{region}/{model}/` | [`models/persistence.py`](../models/persistence.py) |
| Model rollback mechanism | edit `latest.json` to point at older version | [`models/persistence.py`](../models/persistence.py) |
| Redis namespace prefix | `gridpulse:` (was `wattcast:` until [#114](https://github.com/kristenmartino/gridpulse/pull/114)) | [`data/redis_client.REDIS_KEY_PREFIX`](../data/redis_client.py) |
| Visible tabs | **5**: Overview, US Grid, Forecast, Risk, Models | `config._VISIBLE_TABS` |
| Tabs original / current | 9 visible → 5 visible (R3 redesign 2026) | [`components/layout.py`](../components/layout.py) |
| Public API base URL | `https://gridpulse.kristenmartino.ai/api/v1` (read-only, no key; 5 routes; 60s public cache on 200s; forecast horizon capped at 168h) | [`api.py`](../api.py) ([#250](https://github.com/kristenmartino/gridpulse/issues/250)) |

## Product framing

| Fact | Value |
|---|---|
| Category | Energy Intelligence Platform |
| Positioning | Forecast confidence, grid visibility, decision support |
| Tagline | See demand sooner. Decide with confidence. |
| Personas | 4: Grid Operations, Renewables, Trader, Data Scientist |
| Production URL | https://gridpulse.kristenmartino.ai |
| Test count | 1,589 passing as of [#119](https://github.com/kristenmartino/gridpulse/pull/119) |

## Data sources

| Source | Endpoint | Notes |
|---|---|---|
| Demand | EIA API v2 `/electricity/rto/region-data/` | Hourly per BA |
| Generation by fuel | EIA API v2 `/electricity/rto/fuel-type-data/` | Hourly per BA |
| Interchange | EIA API v2 `/electricity/rto/interchange-data/` | Hourly tie-line flows |
| Weather | Open-Meteo (no API key) | 17 vars, historical + forecast |
| Severe weather alerts | NOAA NWS | State-scoped |
| Capacity (most BAs) | EIA-860M Feb 2026 | Sum nameplate-MW filtered to `Operating` |
| Capacity (import-dominated BAs) | Peak demand × 1.15 reserve margin | V3.η fix for CPLW/HST/SPA |

## Forecast accuracy (from holdout backtests)

Accuracy is **per-BA** — never quote a single pooled "across-51" number.
Distributions over all 51 BAs, 168h holdout, **recursive multi-step** (three
views). Recursive scoring — the model's own predictions feed forward as lags
— runs ~2× the teacher-forced one-step numbers published before
[#209](https://github.com/kristenmartino/gridpulse/issues/209); it is the
honest 7-day-forecast number, not a nowcast:

| Stat | XGBoost-only | Best-base per BA | Ensemble (served) |
|---|---|---|---|
| min | 1.76% (PSEI) | 1.66% (ERCOT) | 1.48% (ERCOT) |
| median | 4.32% | 4.12% | 4.82% |
| mean | 5.99% | 5.38% | 6.22% |
| p90 | 9.90% | 9.90% | 12.64% |
| max | 38.63% (SEC) | 21.13% (SPA) | 22.81% (SPA) |

XGBoost is best base for 44 of 51 BAs (Prophet for CHPD/DOPD/SEC/SOCO, ARIMA
for ERCOT/SC/WALC). **The ensemble trails best-base in aggregate** (median
4.82% vs 4.12%) but under recursive scoring now beats XGBoost-alone on 17 of
51 BAs (up from 4) — the inverse-MAPE blend (ADR-004) damps compounding
single-model drift on the tail (SEC 38.63% → 13.61%) while landing above
XGBoost where it is already strongest. Quote the ensemble for *what production
serves*, best-base for *best achievable per BA*. Tail BAs (SPA, IID, PSCO,
SEC, AZPS) swing run-to-run.

(Source: generated 2026-07-03 from production GCS via
`scripts/export_holdout_metrics.py`; models trained 2026-07-03. Ensemble
holdout column populated for all 51 BAs since
[#176](https://github.com/kristenmartino/gridpulse/issues/176) fixed the
holdout-NaN crash. Per-BA holdout metrics are produced every daily training
run, persisted to each model's GCS `meta.json`, and surfaced live in the
Models tab via Redis `model_metrics`. Full per-BA, per-model table:
[`docs/BACKTEST_RESULTS.md`](BACKTEST_RESULTS.md).)

Latest ensemble weights example (FPL, 2026-05-01 09:00 UTC scoring run):
`{xgboost: 0.578, prophet: 0.293, arima: 0.130}`.

## Key Architecture Decisions

| ID | Decision | Why |
|---|---|---|
| ADR-001 | Dash + Plotly (not Streamlit) | Callback architecture scales to many interaction groups |
| ADR-002 | SQLite cache on Cloud Run ephemeral disk | Survives across requests, acceptable to lose on recycle |
| ADR-003 | Open-Meteo (not NOAA NWS) for weather | No API key, 17 vars in one call, historical + forecast |
| ADR-004 | 1/MAPE weighted ensemble | Simpler than stacking, self-correcting, bounded |
| ADR-005 | Scenario engine copies features, never mutates | Pure function, safe for concurrent callbacks |
| ADR-006 | Full multi-tab architecture | Mission control + drill-downs |

## How this file gets maintained

- **Per-PR**: any PR that moves a value here updates it in the same commit (CLAUDE.md end-of-PR check item #2)
- **Audit cadence**: re-verify each row against its source quarterly (or after every 20 PRs at high velocity)
- This file is **derived from code/data**, not authoritative on its own — if a value here disagrees with the linked source, **the source wins**
