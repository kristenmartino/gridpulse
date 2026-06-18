# Forecast Backtest Results

> **Provenance:** Generated 2026-06-17 from production GCS via
> `scripts/export_holdout_metrics.py` (per-BA rolling 7-day / 168h holdout).
> Models trained 2026-06-17. The full `{mape, rmse, mae, r2}` per BA per
> model lives in the **regenerable** `holdout_metrics.csv` (untracked by
> design — it goes stale every training run; regenerate it, don't read it
> from git). This document holds the human-readable MAPE summary.

> **Leakage caveat (now resolved — kept for history).** Figures in the
> *previous* version of this doc (the 2026-02-21 ERCOT+FPL snapshot) were
> measured under a training regime that leaked the current row's
> `demand_mw` into `ramp_rate` and the `demand_roll_{24,72,168}h_*`
> aggregations (pandas' trailing rolling window includes the current row;
> `demand.diff()` returns `demand[i] - demand[i-1]`). That was fixed in
> `fix/training-feature-leakage` ([#135](https://github.com/kristenmartino/gridpulse/issues/135)):
> every autoregressive feature now reads from `demand.shift(1)` before
> rolling/diffing, matching the inference-time
> `compute_autoregressive_snapshot` definition row-for-row. **The numbers
> below are the current, post-leakage-fix reference** — produced by daily
> training runs on the de-leaked feature definitions.

## Methodology

- **Holdout:** the final **168 hours (7 days)** of each region's training
  data, held out and scored every daily training run.
- **Metrics:** MAPE / RMSE / MAE / R² per base model (XGBoost, Prophet,
  ARIMA), computed in `jobs/training_job.py` and persisted to each model's
  GCS `meta.json` under `extra.holdout_metrics`. The same numbers surface
  live in the **Models tab** via Redis `model_metrics`.
- **Coverage:** all **51** balancing authorities (`config.REGION_COORDINATES`).

## Accuracy distribution (best base per BA)

Accuracy is **per-BA** — a single pooled "across-51" figure hides the tail
(e.g. SPA, AZPS), so we report the **distribution** of each BA's *best base
model* (the lowest MAPE of XGBoost / Prophet / ARIMA for that BA):

| Statistic | Best-base MAPE |
|---|---|
| n | 51 |
| min | **0.79%** (ERCOT) |
| median | **2.28%** |
| mean | **3.38%** |
| p90 | **6.57%** |
| max | **21.00%** (SPA) |

**Worst 5 BAs (by best-base MAPE):** SPA 21.00% (xgboost) · AZPS 11.90%
(arima) · SEC 9.36% (xgboost) · LDWP 8.99% (xgboost) · WALC 6.61% (xgboost).

**Best base model:** XGBoost wins **50 of 51** BAs; ARIMA wins **AZPS** (1).
Reporting best-base rather than XGBoost-only matters for the tail — AZPS is
29.42% on XGBoost but 11.90% on ARIMA, which also pulls the distribution max
down from 29.42% to 21.00%.

## Ensemble holdout — pending (not currently persisted)

The ensemble column is intentionally **blank**: `extra.ensemble_holdout_metrics`
is absent from every XGBoost `meta.json` in GCS, so there are no ensemble
holdout numbers to report. **These are not fabricated or inferred from the
base models.** (The post-hoc `write_extra_to_meta` ensemble write in
`jobs/training_job.py` isn't landing; root cause is tracked separately.) The
live ensemble forecast still runs — only its *holdout backtest metric* is
missing here.

## Per-BA holdout MAPE (current)

Per-model MAPE plus the best base and that model's R². Full RMSE/MAE/R² for
every model is in `holdout_metrics.csv`.

| BA | Region | XGBoost | Prophet | ARIMA | Best base | Best-base R² |
|---|---|---|---|---|---|---|
| AECI | Missouri (AECI) | 1.86% | 17.48% | 20.05% | xgboost | 0.989 |
| AVA | Spokane (Avista) | 1.70% | 3.85% | 3.77% | xgboost | 0.967 |
| AZPS | Arizona (APS) | 29.42% | 19.51% | 11.90% | arima | 0.296 |
| BANC | Sacramento (BANC) | 4.25% | 12.94% | 10.73% | xgboost | 0.908 |
| BPAT | Pacific NW (BPA) | 1.69% | 4.05% | 4.83% | xgboost | 0.927 |
| CAISO | California (CAISO) | 5.39% | 14.48% | 16.23% | xgboost | 0.668 |
| CHPD | Chelan County PUD | 2.09% | 8.89% | 5.42% | xgboost | 0.962 |
| CPLE | Carolinas East (DEP) | 2.97% | 9.99% | 12.45% | xgboost | 0.941 |
| CPLW | DEP-West (NC mountains) | 3.89% | 8.30% | 12.23% | xgboost | 0.889 |
| DOPD | Douglas County PUD | 1.64% | 8.81% | 9.19% | xgboost | 0.957 |
| DUK | Carolinas West (DEC) | 2.61% | 13.97% | 14.44% | xgboost | 0.956 |
| EPE | El Paso (EPE) | 2.12% | 9.98% | 5.09% | xgboost | 0.986 |
| ERCOT | Texas (ERCOT) | 0.79% | 10.15% | 16.55% | xgboost | 0.994 |
| FMPP | Florida Muni Pool | 2.09% | 6.19% | 7.70% | xgboost | 0.973 |
| FPC | Florida (Duke FL) | 1.48% | 18.14% | 5.80% | xgboost | 0.992 |
| FPL | Florida (FPL/NextEra) | 1.55% | 3.27% | 3.85% | xgboost | 0.990 |
| GCPD | Grant County PUD | 1.33% | 10.54% | 3.17% | xgboost | 0.911 |
| GVL | Gainesville (GRU) | 1.64% | 7.63% | 8.37% | xgboost | 0.986 |
| HST | Homestead | 1.89% | 5.43% | 5.58% | xgboost | 0.983 |
| IID | Imperial Valley (IID) | 6.57% | 21.65% | 17.27% | xgboost | 0.937 |
| IPCO | Idaho (Idaho Power) | 1.52% | 6.45% | 5.74% | xgboost | 0.962 |
| ISONE | New England (ISO-NE) | 3.51% | 14.32% | 12.35% | xgboost | 0.885 |
| JEA | Jacksonville (JEA) | 2.14% | 6.42% | 5.98% | xgboost | 0.972 |
| LDWP | Los Angeles (LADWP) | 8.99% | 12.04% | 12.85% | xgboost | 0.576 |
| LGEE | Kentucky (LG&E + KU) | 2.74% | 20.40% | 11.19% | xgboost | 0.928 |
| MISO | Midwest (MISO) | 1.88% | 13.26% | 4.83% | xgboost | 0.960 |
| NEVP | Southern Nevada (NV Energy) | 3.68% | 6.84% | 5.71% | xgboost | 0.897 |
| NWMT | Montana (NorthWestern) | 1.08% | 2.99% | 4.30% | xgboost | 0.978 |
| NYISO | New York (NYISO) | 2.52% | 14.61% | 10.58% | xgboost | 0.928 |
| PACE | Inland West (PacifiCorp E) | 2.26% | 7.49% | 3.98% | xgboost | 0.954 |
| PACW | Pacific NW (PacifiCorp W) | 3.67% | 7.09% | 5.16% | xgboost | 0.802 |
| PGE | Portland General | 2.89% | 4.37% | 4.24% | xgboost | 0.861 |
| PJM | Mid-Atlantic (PJM) | 2.10% | 15.09% | 61.32% | xgboost | 0.937 |
| PNM | New Mexico (PNM) | 2.74% | 8.77% | 3.60% | xgboost | 0.922 |
| PSCO | Colorado (Xcel) | 3.34% | 10.94% | 11.21% | xgboost | 0.956 |
| PSEI | Puget Sound Energy | 1.45% | 5.31% | 5.76% | xgboost | 0.970 |
| SC | Santee Cooper | 2.71% | 6.04% | 14.66% | xgboost | 0.938 |
| SCEG | Carolinas Mid (Dominion SC) | 4.21% | 6.26% | 14.81% | xgboost | 0.886 |
| SCL | Seattle (SCL) | 2.28% | 6.11% | 7.76% | xgboost | 0.940 |
| SEC | Seminole Electric | 9.36% | 14.26% | 10.51% | xgboost | 0.375 |
| SOCO | Southeast (Southern Co.) | 1.17% | 6.35% | 7.99% | xgboost | 0.991 |
| SPA | SW Power Admin | 21.00% | 42.56% | 48.17% | xgboost | 0.093 |
| SPP | Southwest (SPP) | 1.16% | 8.50% | 8.77% | xgboost | 0.988 |
| SRP | Phoenix (SRP) | 5.58% | 26.11% | 21.17% | xgboost | 0.890 |
| TAL | Tallahassee | 2.47% | 9.73% | 6.38% | xgboost | 0.963 |
| TEC | Tampa Bay (TECO) | 1.16% | 7.30% | 3.51% | xgboost | 0.994 |
| TEPC | Tucson (TEP) | 2.42% | 4.86% | 3.82% | xgboost | 0.968 |
| TIDC | Turlock ID | 2.31% | 7.29% | 19.28% | xgboost | 0.968 |
| TPWR | Tacoma Power | 2.07% | 7.17% | 9.04% | xgboost | 0.958 |
| TVA | Tennessee Valley (TVA) | 2.06% | 16.11% | 4.77% | xgboost | 0.961 |
| WALC | Desert SW (WAPA-DSW) | 6.61% | 19.99% | 18.11% | xgboost | 0.794 |

## Why XGBoost dominates the base models

1. **Feature engineering.** XGBoost uses all 49 engineered features —
   lagged demand (24h/168h), rolling statistics, degree-days (CDD/HDD), and
   calendar features — capturing non-linear weather↔demand interactions that
   Prophet (7 regressors) and ARIMA (no weather) miss.
2. **Tail behaviour.** Where XGBoost struggles it is usually a low-load or
   data-quality regime (SPA, LDWP, SEC) rather than a modelling gap; in one
   such case (AZPS) ARIMA's smoother extrapolation wins, which is exactly why
   the headline uses best-base-per-BA.
3. **No-leakage validation.** Post-#135, the holdout uses the honest
   `demand.shift(1)` autoregressive snapshot, so these numbers reflect true
   forward-prediction skill, not reconstructed targets.

## Regenerating this table

```bash
# Reads the live per-BA holdout metrics straight from the GCS model store.
# No training, no EIA/weather fetch — runs in minutes across all 51 BAs.
gcloud auth application-default login            # if ADC not already set
ENVIRONMENT=production GCS_BUCKET_NAME=nextera-portfolio-energy-cache \
  python scripts/export_holdout_metrics.py \
    --out-md docs/_holdout_table.md --out-csv holdout_metrics.csv
```

`scripts/backtest.py --region <BA> --holdout-days 21` still exists for an
*independent* from-scratch recompute on a 21-day holdout (different
methodology — fetches 90 days, retrains locally), useful for spot-checking a
single BA but not for refreshing this table.
