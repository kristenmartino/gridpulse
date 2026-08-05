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
| UI-visible count | **51** (quality-gated) | a BA is hidden only when its best *served* model — ensemble or champion base, **not** XGBoost-alone ([#255](https://github.com/kristenmartino/gridpulse/issues/255)) — is in the 7d rollback grade **as measured on the training holdout** (the generous question; the horizon-matched serve-path grade is published beside it but does not hide, [#349](https://github.com/kristenmartino/gridpulse/issues/349)); **0 hidden**, re-measured 2026-07-28 |
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
| Scoring job runtime | **~820s baseline** at the pre-2026-08-05 config (`PRECOMPUTE_MAX_WORKERS=4` on `--cpu 2` / 4Gi), daily median stable 07-17→08-04. Raised to **8 workers on `--cpu 4` / 8Gi on 2026-08-05**; post-bump baseline **not yet measured**. 30 min (1800s) Cloud Run task timeout. Under upstream degradation it reaches **the cap** — 2026-08-04 saw 1004 → 1283 → **two ticks killed at 1800s** → 1792s → 1375s → 1625s. Guarded at 70% by the runtime-creep alert ([#171](https://github.com/kristenmartino/gridpulse/issues/171)) and at 85% by the soft deadline. **#171's `<600s` acceptance criterion has never been met** | 855s observed 2026-06-01; 20-day median from `scoring_job_complete` logs 2026-08-04; the row previously said "1083–1333s under elevated upstream latency", which 2026-08-04 falsified |
| Cloud Run Jobs cost (us-east1) | **$0.000018 / vCPU-second**, **$0.000002 / GiB-second**. Monthly: training **~$80**, scoring **~$27** pre-bump (**~$36** expected after, break-even at a 2× speedup) — jobs total **~$106–115** against a **$150** budget, before the web service, Memorystore, GCS and egress. **The training job is 3× the scoring job** | Cloud Billing Catalog API, SKUs "Jobs CPU/Memory in us-east1", read 2026-08-05; runtimes from job logs |
| Scoring-job runtime budget rule | `--task-timeout × (max-retries + 1)` must stay **below the scheduler interval**. At `1800 × 2 = 3600s` against an hourly cadence there is **zero margin** — do not raise the timeout | 2026-08-04: the 19:00 execution's retry finished at 20:01, after the 20:00 execution had started. The `deploy-prod.yml` comment claiming runs "can't overlap" was wrong |
| Weather archive leg, per BA per fetch | 0.28 MB single-point, 3.40 MB at 12 points (ADR-012), 87 days × 17 variables; ~109 MB/tick fleet-wide | **Synthetic** — a response of production dimensions, not a production measurement. Its effect on end-to-end runtime is **not** visible in the daily medians across the ADR-011/012 flips (07-22/07-23), which stayed flat |
| Training job runtime | ~2h45m wall for 51 BAs across 3 parallel tasks (5h per-task Cloud Run timeout). **Billing is the SUM of per-task durations, not 3× wall** — 27,541 task-seconds on 2026-08-04 (8152 / 10025 / 9364) | `gcloud run jobs executions` + per-task log spans, 2026-08-04..05; the row previously said "~3 hours" without distinguishing wall from billed |
| Training job cost | **~$73/mo** at 4 vCPU / 8 GiB (82% vCPU, 18% memory). Reduced 2026-08-05 by skipping discarded CV in the backtest folds (**−~$29**) and 8→4 GiB (**−$6.6**) → **~$40/mo** expected; post-change figure **not yet measured** | Cloud Billing Catalog rates × measured task-seconds |
| Training job resource utilization | Memory peak **1.24 GiB of 8** (15.5%) → cut to 4Gi. CPU **0.66 mean / 0.82 peak of 4 vCPU** (~2.6 cores busy) → **kept at 4**; `--cpu 2` is not the free win it appears to be | Cloud Monitoring `container/memory/utilizations` + `container/cpu/utilizations`, 3-day window read 2026-08-05 |
| XGBoost fits per BA per training run | **13.9** measured, against **3** models saved. 12 of the 14 are backtest walk-forward folds; each ran a 5-fold CV whose `cv_scores` the caller discarded — 60 of ~80 boosters per BA. Removed 2026-08-05; the production fit still cross-validates because it reads `cv_scores` as a fallback MAPE feeding ADR-004 weights | `xgboost_trained` / `model_saved` log counts over one run (709 / 153 for 51 BAs) |
| Model storage | `gs://nextera-portfolio-energy-cache/models/{region}/{model}/` | [`models/persistence.py`](../models/persistence.py) |
| Model rollback mechanism | edit `latest.json` to point at older version | [`models/persistence.py`](../models/persistence.py) |
| Redis namespace prefix | `gridpulse:` (was `wattcast:` until [#114](https://github.com/kristenmartino/gridpulse/pull/114)) | [`data/redis_client.REDIS_KEY_PREFIX`](../data/redis_client.py) |
| Visible tabs | **5**: Overview, US Grid, Forecast, Risk, Models | `config._VISIBLE_TABS` |
| Tabs original / current | 9 visible → 5 visible (R3 redesign 2026) | [`components/layout.py`](../components/layout.py) |
| Public API base URL | `https://gridpulse.kristenmartino.ai/api/v1` (read-only, no key; **6 data endpoints** — `regions`, `forecast/{region}`, `grid/summary`, `drift/{region}`, `benchmark`, `benchmark/{region}` — plus a self-describing index at `/`; 60s public cache on 200s; forecast horizon capped at 168h) | [`api.py`](../api.py) ([#250](https://github.com/kristenmartino/gridpulse/issues/250)) |

## Product framing

| Fact | Value |
|---|---|
| Category | Energy Intelligence Platform |
| Positioning | Forecast confidence, grid visibility, decision support |
| Tagline | See demand sooner. Decide with confidence. |
| Personas | 4: Grid Operations, Renewables, Trader, Data Scientist |
| Production URL | https://gridpulse.kristenmartino.ai |
| Test count | **2,989 collected** — 2,986 passed, 3 skipped, in ~79s. Split: unit 2,762 / integration 204 / e2e 23. Measured 2026-08-05 via `pytest tests/ -q` at `db13c06` (previously "1,589 passing as of #119" — stale by ~1,400). The 3 skips are environment-conditional: `test_scenarios_heuristic.py` skips when no Redis-backed ensemble forecast is reachable. **This number moves most weeks** — re-measure at the merge base rather than citing this row's figure back at it (it went 2,977 → 2,989 across a single intervening PR while this row was being written) |

## Data sources

| Source | Endpoint | Notes |
|---|---|---|
| Demand | EIA API v2 `/electricity/rto/region-data/` | Hourly per BA |
| Generation by fuel | EIA API v2 `/electricity/rto/fuel-type-data/` | Hourly per BA |
| Interchange | EIA API v2 `/electricity/rto/interchange-data/` | Hourly tie-line flows |
| Weather | Open-Meteo (no API key) | 17 vars, historical + forecast; future hours are the **NBM composite** since ADR-011 (2026-07-22): `ncep_nbm_conus` overlaid on `best_match`, base-filled for radiation ×3 / surface pressure / 120 m wind + NBM's ~11.5-day tail. Sampled **multi-point** since ADR-012 (2026-07-23): 36 BAs aggregate up to 12 footprint cells (`assets/multipoint_coordinates.json`, unweighted); the 15 compact BAs stay single-point |
| Severe weather alerts | NOAA NWS | State-scoped |
| Capacity (most BAs) | EIA-860M Feb 2026 | Sum nameplate-MW filtered to `Operating` |
| Capacity (7 peak-derived BAs) | Peak demand × 1.15 (V3.η) | SOCO, DUK, CPLE, PSCO, FMPP, HST, CPLW — in-territory generation runs below served load, so the plate is a peak-based estimate (`capacity_source = peak_estimate` in the API); excluded from utilization/top-stress (#254). NOT a reserve margin. **SPA is import-dominated but its 2,559 MW is a true nameplate** (federal dam fleet), so it stays `nameplate`. |

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
[`docs/BACKTEST_RESULTS.md`](BACKTEST_RESULTS.md). Also cited on the
`/about` landing page (`web/landing.html` — the "4.8% median per-BA" chip);
update it when this table regenerates.)

Latest ensemble weights example (FPL, 2026-05-01 09:00 UTC scoring run):
`{xgboost: 0.578, prophet: 0.293, arima: 0.130}`.

## Key Architecture Decisions

| ID | Decision | Why |
|---|---|---|
| ADR-001 | Dash + Plotly (not Streamlit) | Callback architecture scales to many interaction groups |
| ADR-002 | SQLite cache on Cloud Run ephemeral disk | Survives across requests, acceptable to lose on recycle |
| ADR-003 | Open-Meteo (not NOAA NWS) for weather | No API key, 17 vars in one call, historical + forecast |
| ADR-004 | Sharpened (1/MAPE)³ weighted ensemble | Simpler than stacking; value is error decorrelation (beats XGBoost-alone on 17/51 — see "Forecast accuracy" above), not a guarantee of dominance |
| ADR-005 | Scenario engine copies features, never mutates | Pure function, safe for concurrent callbacks |
| ADR-006 | Full multi-tab architecture | Mission control + drill-downs |

## How this file gets maintained

- **Per-PR**: any PR that moves a value here updates it in the same commit (CLAUDE.md end-of-PR check item #2)
- **Audit cadence**: re-verify each row against its source quarterly (or after every 20 PRs at high velocity)
- This file is **derived from code/data**, not authoritative on its own — if a value here disagrees with the linked source, **the source wins**
