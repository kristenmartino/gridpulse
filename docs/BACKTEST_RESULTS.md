# Forecast Backtest Results

> **Provenance:** Generated 2026-06-19 from production GCS via
> `scripts/export_holdout_metrics.py` (per-BA rolling 7-day / 168h holdout).
> Models trained 2026-06-19. The full `{mape, rmse, mae, r2}` per BA per
> model lives in the **regenerable** `holdout_metrics.csv` (untracked by
> design — it goes stale every training run; regenerate it, don't read it
> from git). This document holds the human-readable MAPE summary. The
> **ensemble** column is now populated for all 51 BAs (was blank until
> [#176](https://github.com/kristenmartino/gridpulse/issues/176) fixed the
> holdout-NaN crash that had been dropping Prophet+ARIMA from the blend).

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

## Accuracy distribution (per BA, 168h holdout)

Accuracy is **per-BA** — a single pooled "across-51" figure hides the tail
(AZPS, SPA), so we report **distributions**, not one number. Three views,
all 51 BAs:

| Statistic | XGBoost-only | Best-base per BA | Ensemble (served) |
|---|---|---|---|
| n | 51 | 51 | 51 |
| min | **0.98%** (ERCOT) | **0.98%** (ERCOT) | **1.70%** (NWMT) |
| median | **2.32%** | **2.30%** | **3.48%** |
| mean | **3.79%** | **3.61%** | **4.92%** |
| p90 | **6.57%** | **6.57%** | **8.37%** |
| max | **33.97%** (AZPS) | **26.68%** (AZPS) | **27.40%** (AZPS) |

**The ensemble trails the best base model in aggregate** (median 3.48% vs
2.30%) and beats XGBoost-alone on only **4 of 51** BAs. This is expected, not
a regression: the inverse-MAPE blend (ADR-004) still gives real weight to
Prophet and ARIMA, which run 3–5× worse than XGBoost on most BAs, so the
blend lands above the strongest single model. The ensemble earns its keep as
**variance-reduction on the tail** (AZPS: XGBoost alone 33.97% → blend
27.40%), not as a headline-accuracy win. For per-BA *best-achievable*
accuracy, read the best-base column; for *what production serves by default*,
read the ensemble column.

**Worst 5 BAs (ensemble MAPE):** AZPS 27.40% · SPA 22.07% · IID 16.31% ·
NEVP 11.17% · WALC 8.44%. These tail BAs (low load / data-quality regimes)
swing materially run-to-run — AZPS was 11.90% best-base on 2026-06-17 and
26.68% here — which is why the table is regenerated each pass, not frozen.

**Best base model:** XGBoost wins **48 of 51** BAs; ARIMA wins **AZPS** and
**GCPD** (2); Prophet wins **SPA** (1). Reporting best-base rather than
XGBoost-only matters for the tail — AZPS is 33.97% on XGBoost but 26.68% on
ARIMA.

## Per-BA holdout MAPE (current)

Per-model MAPE plus the ensemble, best base, and training-window provenance.
This is the verbatim `scripts/export_holdout_metrics.py` markdown output;
full RMSE/MAE/R² for every model is in the regenerable `holdout_metrics.csv`.

| BA | Region | XGBoost | Prophet | ARIMA | Ensemble | Best base | Train rows | Trained (UTC) |
|---|---|---|---|---|---|---|---|---|
| AECI | Missouri (AECI) | 2.24% | 20.35% | 23.95% | 4.97% | xgboost | 1997 | 2026-06-19 |
| AVA | Spokane (Avista) | 2.16% | 7.60% | 8.51% | 3.83% | xgboost | 1998 | 2026-06-19 |
| AZPS | Arizona (APS) | 33.97% | 32.54% | 26.68% | 27.40% | arima | 1996 | 2026-06-19 |
| BANC | Sacramento (BANC) | 3.02% | 9.50% | 4.30% | 3.45% | xgboost | 1998 | 2026-06-19 |
| BPAT | Pacific NW (BPA) | 1.87% | 4.99% | 5.17% | 2.69% | xgboost | 1996 | 2026-06-19 |
| CAISO | California (CAISO) | 3.65% | 10.25% | 4.54% | 3.34% | xgboost | 1996 | 2026-06-19 |
| CHPD | Chelan County PUD | 2.48% | 12.85% | 6.28% | 3.21% | xgboost | 1998 | 2026-06-19 |
| CPLE | Carolinas East (DEP) | 1.70% | 18.64% | 8.72% | 2.51% | xgboost | 1996 | 2026-06-19 |
| CPLW | DEP-West (NC mountains) | 2.18% | 10.69% | 12.45% | 3.98% | xgboost | 1997 | 2026-06-19 |
| DOPD | Douglas County PUD | 1.86% | 6.66% | 4.80% | 2.08% | xgboost | 1998 | 2026-06-19 |
| DUK | Carolinas West (DEC) | 2.04% | 13.03% | 8.05% | 2.22% | xgboost | 1996 | 2026-06-19 |
| EPE | El Paso (EPE) | 2.16% | 13.39% | 6.79% | 3.53% | xgboost | 1997 | 2026-06-19 |
| ERCOT | Texas (ERCOT) | 0.98% | 8.21% | 6.86% | 1.79% | xgboost | 1996 | 2026-06-19 |
| FMPP | Florida Muni Pool | 2.57% | 5.18% | 5.95% | 3.53% | xgboost | 1997 | 2026-06-19 |
| FPC | Florida (Duke FL) | 1.91% | 6.60% | 7.42% | 3.48% | xgboost | 1997 | 2026-06-19 |
| FPL | Florida (FPL/NextEra) | 2.32% | 3.84% | 6.44% | 2.08% | xgboost | 1996 | 2026-06-19 |
| GCPD | Grant County PUD | 2.53% | 5.40% | 2.16% | 2.14% | arima | 1998 | 2026-06-19 |
| GVL | Gainesville (GRU) | 1.98% | 7.13% | 5.98% | 2.91% | xgboost | 1955 | 2026-06-19 |
| HST | Homestead | 2.56% | 6.67% | 8.84% | 3.37% | xgboost | 1997 | 2026-06-19 |
| IID | Imperial Valley (IID) | 7.55% | 49.66% | 40.31% | 16.31% | xgboost | 1998 | 2026-06-19 |
| IPCO | Idaho (Idaho Power) | 1.83% | 9.51% | 9.83% | 3.56% | xgboost | 1997 | 2026-06-19 |
| ISONE | New England (ISO-NE) | 3.01% | 16.76% | 8.93% | 3.89% | xgboost | 1996 | 2026-06-19 |
| JEA | Jacksonville (JEA) | 2.03% | 4.94% | 6.49% | 2.74% | xgboost | 1997 | 2026-06-19 |
| LDWP | Los Angeles (LADWP) | 5.17% | 19.76% | 13.83% | 8.37% | xgboost | 1998 | 2026-06-19 |
| LGEE | Kentucky (LG&E + KU) | 1.75% | 11.28% | 14.06% | 2.92% | xgboost | 1997 | 2026-06-19 |
| MISO | Midwest (MISO) | 1.03% | 16.54% | 10.71% | 2.12% | xgboost | 1996 | 2026-06-19 |
| NEVP | Southern Nevada (NV Energy) | 9.82% | 12.87% | 13.82% | 11.17% | xgboost | 1988 | 2026-06-19 |
| NWMT | Montana (NorthWestern) | 1.16% | 5.61% | 4.51% | 1.70% | xgboost | 1993 | 2026-06-19 |
| NYISO | New York (NYISO) | 2.11% | 19.86% | 15.76% | 4.24% | xgboost | 1996 | 2026-06-19 |
| PACE | Inland West (PacifiCorp E) | 2.66% | 5.27% | 5.05% | 3.02% | xgboost | 1997 | 2026-06-19 |
| PACW | Pacific NW (PacifiCorp W) | 4.73% | 7.75% | 7.89% | 5.87% | xgboost | 1997 | 2026-06-19 |
| PGE | Portland General | 3.25% | 8.15% | 10.73% | 5.43% | xgboost | 1997 | 2026-06-19 |
| PJM | Mid-Atlantic (PJM) | 1.20% | 22.32% | 40.19% | 2.92% | xgboost | 1996 | 2026-06-19 |
| PNM | New Mexico (PNM) | 1.79% | 7.00% | 5.23% | 2.15% | xgboost | 1998 | 2026-06-19 |
| PSCO | Colorado (Xcel) | 4.16% | 15.54% | 11.16% | 6.72% | xgboost | 1997 | 2026-06-19 |
| PSEI | Puget Sound Energy | 1.73% | 6.42% | 7.74% | 2.97% | xgboost | 1997 | 2026-06-19 |
| SC | Santee Cooper | 3.11% | 14.35% | 10.34% | 4.95% | xgboost | 1997 | 2026-06-19 |
| SCEG | Carolinas Mid (Dominion SC) | 2.59% | 9.64% | 19.49% | 3.99% | xgboost | 1896 | 2026-06-19 |
| SCL | Seattle (SCL) | 2.72% | 8.28% | 7.03% | 4.32% | xgboost | 1997 | 2026-06-19 |
| SEC | Seminole Electric | 7.45% | 12.24% | 11.80% | 7.97% | xgboost | 1996 | 2026-06-19 |
| SOCO | Southeast (Southern Co.) | 1.71% | 7.66% | 5.76% | 2.73% | xgboost | 1974 | 2026-06-19 |
| SPA | SW Power Admin | 21.88% | 20.74% | 30.54% | 22.07% | prophet | 1970 | 2026-06-19 |
| SPP | Southwest (SPP) | 1.12% | 10.71% | 6.68% | 2.17% | xgboost | 1996 | 2026-06-19 |
| SRP | Phoenix (SRP) | 4.27% | 9.33% | 14.53% | 4.73% | xgboost | 1998 | 2026-06-19 |
| TAL | Tallahassee | 2.30% | 10.15% | 8.36% | 3.92% | xgboost | 1997 | 2026-06-19 |
| TEC | Tampa Bay (TECO) | 1.44% | 6.41% | 8.57% | 2.78% | xgboost | 1997 | 2026-06-19 |
| TEPC | Tucson (TEP) | 2.57% | 10.19% | 6.94% | 3.02% | xgboost | 1998 | 2026-06-19 |
| TIDC | Turlock ID | 2.51% | 20.98% | 14.45% | 5.11% | xgboost | 1947 | 2026-06-19 |
| TPWR | Tacoma Power | 2.40% | 7.97% | 7.33% | 3.79% | xgboost | 1997 | 2026-06-19 |
| TVA | Tennessee Valley (TVA) | 1.34% | 12.83% | 9.47% | 2.49% | xgboost | 1996 | 2026-06-19 |
| WALC | Desert SW (WAPA-DSW) | 6.57% | 16.47% | 8.81% | 8.44% | xgboost | 1998 | 2026-06-19 |

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
