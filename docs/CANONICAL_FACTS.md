# Canonical Facts — GridPulse

> Cross-doc fact source. When `README.md`, `PRD.md`, `TECHNICAL_SPEC.md`,
> case-study copy in `portfolio-v2`, or interview material need to cite a
> number/list/name, **they reference this file** rather than restating.
> When a value here changes, update one place and consumers pick it up.
>
> Verified: **2026-05-20** (initial population from PR #123 wider replan).
> Next planned re-verification: with PR-C1 (next session).

## Scale

| Fact | Value | Source of truth |
|---|---|---|
| Balancing authorities covered | **51** | [`config.REGION_COORDINATES`](../config.py) |
| Demand coverage (contiguous US lower-48) | **~100%** | derived from EIA-930 demand vs. our covered set |
| EIA-930 total BAs (contiguous US) | **63** | [Aug 2025 Federal Register PRA renewal](https://www.govinfo.gov/content/pkg/FR-2025-08-28/pdf/2025-16450.pdf) |
| Expansion history | Original 8 → V1.α +8 → V3.ζ +35 | [`docs/internal/NEXT_UP.md`](internal/NEXT_UP.md) |

## Models

| Fact | Value | Source of truth |
|---|---|---|
| Base ML models | **3**: Prophet, SARIMAX, XGBoost | [`models/`](../models/) |
| Ensemble method | Inverse-MAPE weighted | [`models/ensemble.py`](../models/ensemble.py) |
| User-selectable forecasts in UI | **4**: XGBoost, Prophet, ARIMA, Ensemble | [`components/_callbacks_forecast.py`](../components/_callbacks_forecast.py) |
| Total engineered features | **43** (17 raw weather + 26 derived) | [`data/feature_engineering.py`](../data/feature_engineering.py) |

## Architecture

| Fact | Value | Source of truth |
|---|---|---|
| Web tier | Cloud Run Service `gridpulse` | [`.github/workflows/deploy-prod.yml`](../.github/workflows/deploy-prod.yml) |
| Scheduled work | 2 Cloud Run Jobs (`-scoring-job` hourly, `-training-job` daily 04:00 UTC) | [`docs/SCHEDULED_JOBS.md`](SCHEDULED_JOBS.md) |
| Model storage | `gs://nextera-portfolio-energy-cache/models/` | [`models/persistence.py`](../models/persistence.py) |
| Redis namespace prefix | `gridpulse:` (was `wattcast:` until [#114](https://github.com/kristenmartino/gridpulse/pull/114)) | [`data/redis_client.REDIS_KEY_PREFIX`](../data/redis_client.py) |
| Visible tabs | **5**: Overview, US Grid, Forecast, Risk, Models | `config._VISIBLE_TABS` |

## Product framing

| Fact | Value |
|---|---|
| Category | Energy Intelligence Platform |
| Positioning | Forecast confidence, grid visibility, decision support |
| Tagline | See demand sooner. Decide with confidence. |
| Personas | 4: Grid Operations, Renewables, Trader, Data Scientist |
| Production URL | https://gridpulse.kristenmartino.ai |

## Data sources

| Fact | Value | Notes |
|---|---|---|
| Demand | EIA API v2 (`/electricity/rto/region-data/`) | Hourly per BA |
| Generation by fuel | EIA API v2 (`/electricity/rto/fuel-type-data/`) | Hourly |
| Interchange | EIA API v2 (`/electricity/rto/interchange-data/`) | Hourly tie-line flows |
| Weather | Open-Meteo (no API key required) | 17 vars, historical + forecast |
| Severe weather alerts | NOAA NWS | State-scoped |
| Capacity | EIA-860M Feb 2026 | Except import-dominated BAs (V3.η) which use peak × 1.15 |

## Forecast accuracy (from holdout backtests)

| Fact | Value | Source |
|---|---|---|
| Holdout MAPE — Prophet (PJM 24h) | 11.04% | 2026-05-01 training run |
| Holdout MAPE — ARIMA (PJM 24h) | 5.19% | 2026-05-01 training run |
| Holdout MAPE — Prophet (FPL 24h) | 7.88% | 2026-05-01 training run |
| Holdout MAPE — ARIMA (FPL 24h) | 5.55% | 2026-05-01 training run |
| Latest ensemble weights (FPL) | xgboost=0.578, prophet=0.293, arima=0.130 | 2026-05-01 09:00 UTC scoring run |

(Wider MAPE table for all 51 BAs / all horizons lives in
[`docs/BACKTEST_RESULTS.md`](BACKTEST_RESULTS.md).)

## How this file gets maintained

- Per-PR: when a PR changes a value cited here, update in the same commit
- Quarterly audit (or after every 20 PRs at high velocity, whichever first):
  re-verify each row against its source
- This file is **derived from code/data**, not authoritative on its own —
  if a value here disagrees with the linked source, the source wins
