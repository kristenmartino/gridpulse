# Forecast Backtest Results

> **Provenance:** Generated 2026-07-03 from production GCS via
> `scripts/export_holdout_metrics.py` (per-BA rolling 7-day / 168h holdout).
> Models trained 2026-07-03. These are **recursive multi-step** holdout
> numbers (see the recursive-holdout note below); they supersede the lower
> teacher-forced one-step figures published before
> [#209](https://github.com/kristenmartino/gridpulse/issues/209). The full
> `{mape, rmse, mae, r2}` per BA per model lives in the **regenerable**
> `holdout_metrics.csv` (untracked by design — it goes stale every training
> run; regenerate it, don't read it from git). This document holds the
> human-readable MAPE summary. The **ensemble** column is populated for all
> 51 BAs (was blank until
> [#176](https://github.com/kristenmartino/gridpulse/issues/176) fixed the
> holdout-NaN crash that had been dropping Prophet+ARIMA from the blend).
>
> **Recursive-holdout caveat (why these numbers are ~2× the old ones).**
> Figures published before [#209](https://github.com/kristenmartino/gridpulse/issues/209)
> (2026-06-19 and earlier) were measured **teacher-forced**: each hour of the
> 168h holdout was predicted from the *real* demand of the preceding hours,
> because the autoregressive lag features were built from actuals. That
> answers "how good is a one-hour-ahead nowcast," not "how good is a 7-day
> forecast." Production now scores the holdout **recursively**
> (`recursive_autoregressive_forecast`): the model's own predictions feed
> forward as the lags for the next step, so errors compound exactly as they
> do in a real forward forecast. Recursive MAPE runs ~2× the teacher-forced
> number (median 4.32% vs 2.32% on XGBoost) — not a regression, the honest
> number. The Models tab, `meta.json`, and this doc all report it.

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
  ARIMA), computed **recursively** (multi-step autoregressive — the model's
  own predictions feed the next step's lags) in `jobs/training_job.py` and
  persisted to each model's GCS `meta.json` under `extra.holdout_metrics`.
  The same numbers surface live in the **Models tab** via Redis
  `model_metrics`.
- **Coverage:** all **51** balancing authorities (`config.REGION_COORDINATES`).

## Accuracy distribution (per BA, 168h holdout)

Accuracy is **per-BA** — a single pooled "across-51" figure hides the tail
(AZPS, SPA), so we report **distributions**, not one number. Three views,
all 51 BAs:

| Statistic | XGBoost-only | Best-base per BA | Ensemble (served) |
|---|---|---|---|
| n | 51 | 51 | 51 |
| min | **1.76%** (PSEI) | **1.66%** (ERCOT) | **1.48%** (ERCOT) |
| median | **4.32%** | **4.12%** | **4.82%** |
| mean | **5.99%** | **5.38%** | **6.22%** |
| p90 | **9.90%** | **9.90%** | **12.64%** |
| max | **38.63%** (SEC) | **21.13%** (SPA) | **22.81%** (SPA) |

**The ensemble trails the best base model in aggregate** (median 4.82% vs
4.12%) but under recursive scoring it now **beats XGBoost-alone on 17 of 51**
BAs (up from 4 of 51 teacher-forced): as errors compound over the horizon,
blending in Prophet and ARIMA damps the worst single-model drift. On the
majority where XGBoost is strongest the inverse-MAPE blend (ADR-004) still
lands above it, because Prophet and ARIMA run 3–5× worse there. The ensemble
earns its keep as **variance-reduction on the tail** (SEC: XGBoost alone
38.63% → blend 13.61%), not as a headline-accuracy win. For per-BA
*best-achievable* accuracy, read the best-base column; for *what production
serves by default*, read the ensemble column.

**Worst 5 BAs (ensemble MAPE):** SPA 22.81% · IID 15.37% · PSCO 14.69% ·
SEC 13.61% · AZPS 12.70%. These tail BAs (low load / data-quality regimes)
swing materially run-to-run, which is why the table is regenerated each pass,
not frozen.

**Best base model:** XGBoost wins **44 of 51** BAs; Prophet wins **CHPD,
DOPD, SEC, SOCO** (4); ARIMA wins **ERCOT, SC, WALC** (3). Reporting
best-base rather than XGBoost-only matters for the tail — SEC is 38.63% on
XGBoost but 11.22% on Prophet.

## Accuracy by forecast horizon

The 168h figure above is the hardest operating point — but it is barely harder
than day-ahead. An independent recursive recompute (XGBoost only, all 51 BAs, on
a hold-out week sourced from the ERA5 archive — the three archive-missing vars
are imputed exactly as production imputes its own deep history; cross-checked
within ~0.2pp of the all-17-variable recent-window measurement) shows how
cumulative MAPE grows with lead time:

| Horizon | Median | Mean | p90 |
|---|---|---|---|
| 1h (nowcast) | **0.96%** | 2.65% | 4.70% |
| 24h (day-ahead) | **4.14%** | 5.10% | 7.61% |
| 48h | **4.32%** | 5.43% | 8.41% |
| 72h | **4.26%** | 5.50% | 9.46% |
| 168h (7-day) | **4.12%** | 5.34% | 7.51% |

Two things stand out:

1. **The 1-hour nowcast is ~1%.** With real recent demand still in the lag
   features, the model is genuinely excellent very-short-term — this is the
   number the Models-tab live-drift panel tracks, and it is competitive with
   industry nowcasting.
2. **Error jumps at day-ahead, then plateaus — it does not compound.** By 24h
   out the recursive forecast has lost its real-demand anchor and runs on
   weather + calendar + its own predictions, so the error saturates around ~4%
   and holds roughly flat from day-ahead through 7 days (day-ahead actually
   beats the 7-day figure on only **24 of 51** BAs — a coin flip). The practical
   consequence: the ~4.8% ensemble headline is representative of the entire
   day-ahead-to-week operating range, not an artifact of the longest horizon.

**Versus industry.** Best-in-class day-ahead short-term load forecasting runs
1–3% MAPE. GridPulse's strongest large BAs land in or near that band (PJM 1.2%,
ERCOT 3.6%, MISO 3.8% day-ahead), but the fleet median (~4%) sits above it. That
gap is honest and has named causes: the recursive protocol (not teacher-forced),
~90-day training windows (utilities train on multi-year histories), and a fleet
that deliberately includes many small, noisy BAs whose load is intrinsically
harder to predict. We report the gap rather than average it away.

> Methodology note: this per-horizon table is an *independent* recompute
> (XGBoost, archive-sourced week ending ~5 days ago), so its 168h median
> (~4.1%) is close to but not identical to the production XGBoost 168h median
> (4.32%, most-recent week, from GCS `meta.json`) — the difference is
> week-to-week variance, not a methodology change. Both are recursive.

## Ensemble weighting

The served ensemble weights each model by `(1/MAPE_i)^k` (ADR-004,
`config.ENSEMBLE_WEIGHT_EXPONENT`). `k` was plain inverse-MAPE (`k=1`) until
[#181](https://github.com/kristenmartino/gridpulse/issues/181); it is now
**`k=3`**. On honest recursive data `k=1` was dominated — it kept 15–30% weight
on models running 3–5× worse than the leader, so the blend trailed the best
single model (ensemble median 4.82% vs best-base 4.12%).

Sweeping `k` on the per-model recursive holdout series (all 51 BAs, weights and
scoring on the same 168h window):

| Exponent k | median MAPE | p90 | beats k=1 on |
|---|---|---|---|
| 1.0 (old) | 4.19% | 10.16% | — |
| 2.0 | 3.98% | 8.28% | 48/51 |
| **3.0 (served)** | **3.90%** | **7.95%** | **47/51** |
| 5.0 | 3.90% | 7.47% | 44/51 |
| best-model (k→∞) | 4.07% | 7.41% | 38/51 |
| convex-optimal oracle | 3.75% | — | — |

`k=3` captures nearly all the achievable gain (within ~0.15pp of the oracle),
generalizes (in a held-out even/odd-hour split it beats `k=1` on both median and
tail — 3.88% / 6.78% vs 4.18% / 10.11%), and beats even winner-take-all — because
it still blends where two models are comparably good. That is where the ensemble
earns its keep: **error-decorrelation**, not tail variance-reduction (a single
model owns the tail). Examples where blending genuinely helps: CAISO 4.55% →
3.51%, AZPS 13.4% → 8.2%.

> The weighting change is offline-validated — it re-combines existing model
> outputs, so no retrain is needed — but it is still a served-forecast change;
> watch live ensemble drift after deploy. Numbers independently reproduced and
> red-teamed with a held-out split.

## Per-BA holdout MAPE (current)

Per-model MAPE plus the ensemble, best base, and training-window provenance.
This is the verbatim `scripts/export_holdout_metrics.py` markdown output;
full RMSE/MAE/R² for every model is in the regenerable `holdout_metrics.csv`.

| BA | Region | XGBoost | Prophet | ARIMA | Ensemble | Best base | Train rows | Trained (UTC) |
|---|---|---|---|---|---|---|---|---|
| AECI | Missouri (AECI) | 5.94% | 11.86% | 18.39% | 9.48% | xgboost | 1997 | 2026-07-03 |
| AVA | Spokane (Avista) | 4.12% | 11.26% | 9.81% | 5.24% | xgboost | 1998 | 2026-07-03 |
| AZPS | Arizona (APS) | 12.27% | 19.57% | 13.98% | 12.70% | xgboost | 1996 | 2026-07-03 |
| BANC | Sacramento (BANC) | 4.46% | 7.94% | 7.23% | 5.39% | xgboost | 1998 | 2026-07-03 |
| BPAT | Pacific NW (BPA) | 2.14% | 3.54% | 5.53% | 1.74% | xgboost | 1996 | 2026-07-03 |
| CAISO | California (CAISO) | 3.10% | 4.08% | 4.87% | 2.46% | xgboost | 1996 | 2026-07-03 |
| CHPD | Chelan County PUD | 4.92% | 4.08% | 4.62% | 2.95% | prophet | 1998 | 2026-07-03 |
| CPLE | Carolinas East (DEP) | 3.40% | 7.71% | 5.21% | 3.29% | xgboost | 1996 | 2026-07-03 |
| CPLW | DEP-West (NC mountains) | 5.83% | 8.81% | 12.22% | 7.44% | xgboost | 1997 | 2026-07-03 |
| DOPD | Douglas County PUD | 3.75% | 3.51% | 4.67% | 2.96% | prophet | 1998 | 2026-07-03 |
| DUK | Carolinas West (DEC) | 3.92% | 5.80% | 10.21% | 4.91% | xgboost | 1996 | 2026-07-03 |
| EPE | El Paso (EPE) | 2.53% | 14.91% | 7.62% | 4.25% | xgboost | 1997 | 2026-07-03 |
| ERCOT | Texas (ERCOT) | 2.03% | 4.16% | 1.66% | 1.48% | arima | 1996 | 2026-07-03 |
| FMPP | Florida Muni Pool | 3.70% | 4.37% | 4.69% | 3.89% | xgboost | 1997 | 2026-07-03 |
| FPC | Florida (Duke FL) | 4.70% | 5.94% | 5.92% | 4.48% | xgboost | 1997 | 2026-07-03 |
| FPL | Florida (FPL/NextEra) | 3.60% | 4.18% | 3.65% | 2.85% | xgboost | 1996 | 2026-07-03 |
| GCPD | Grant County PUD | 2.71% | 4.82% | 4.55% | 3.11% | xgboost | 1998 | 2026-07-03 |
| GVL | Gainesville (GRU) | 4.99% | 5.93% | 6.17% | 5.07% | xgboost | 1955 | 2026-07-03 |
| HST | Homestead | 3.70% | 5.75% | 5.93% | 4.06% | xgboost | 1997 | 2026-07-03 |
| IID | Imperial Valley (IID) | 14.01% | 27.75% | 29.34% | 15.37% | xgboost | 1927 | 2026-07-03 |
| IPCO | Idaho (Idaho Power) | 4.32% | 26.86% | 21.75% | 9.17% | xgboost | 1997 | 2026-07-03 |
| ISONE | New England (ISO-NE) | 7.12% | 24.58% | 15.33% | 10.76% | xgboost | 1996 | 2026-07-03 |
| JEA | Jacksonville (JEA) | 4.11% | 5.29% | 6.10% | 4.51% | xgboost | 1997 | 2026-07-03 |
| LDWP | Los Angeles (LADWP) | 5.93% | 9.08% | 12.62% | 6.23% | xgboost | 1998 | 2026-07-03 |
| LGEE | Kentucky (LG&E + KU) | 7.23% | 9.90% | 14.16% | 6.76% | xgboost | 1997 | 2026-07-03 |
| MISO | Midwest (MISO) | 6.18% | 7.30% | 13.41% | 7.86% | xgboost | 1996 | 2026-07-03 |
| NEVP | Southern Nevada (NV Energy) | 3.43% | 17.75% | 16.45% | 6.09% | xgboost | 1988 | 2026-07-03 |
| NWMT | Montana (NorthWestern) | 4.92% | 9.75% | 11.71% | 6.68% | xgboost | 1983 | 2026-07-03 |
| NYISO | New York (NYISO) | 8.11% | 31.41% | 14.05% | 12.64% | xgboost | 1996 | 2026-07-03 |
| PACE | Inland West (PacifiCorp E) | 3.59% | 5.34% | 9.13% | 3.82% | xgboost | 1997 | 2026-07-03 |
| PACW | Pacific NW (PacifiCorp W) | 3.01% | 7.06% | 12.12% | 2.77% | xgboost | 1997 | 2026-07-03 |
| PGE | Portland General | 2.96% | 5.63% | 11.86% | 4.00% | xgboost | 1997 | 2026-07-03 |
| PJM | Mid-Atlantic (PJM) | 8.27% | 20.44% | 8.49% | 7.66% | xgboost | 1996 | 2026-07-03 |
| PNM | New Mexico (PNM) | 3.11% | 3.56% | 3.62% | 2.59% | xgboost | 1998 | 2026-07-03 |
| PSCO | Colorado (Xcel) | 9.90% | 27.20% | 23.43% | 14.69% | xgboost | 1979 | 2026-07-03 |
| PSEI | Puget Sound Energy | 1.76% | 6.94% | 7.69% | 2.68% | xgboost | 1997 | 2026-07-03 |
| SC | Santee Cooper | 3.41% | 5.63% | 2.75% | 2.56% | arima | 1997 | 2026-07-03 |
| SCEG | Carolinas Mid (Dominion SC) | 3.69% | 6.56% | 4.31% | 3.92% | xgboost | 1997 | 2026-07-03 |
| SCL | Seattle (SCL) | 3.04% | 6.05% | 4.87% | 3.63% | xgboost | 1997 | 2026-07-03 |
| SEC | Seminole Electric | 38.63% | 11.22% | 12.94% | 13.61% | prophet | 1997 | 2026-07-03 |
| SOCO | Southeast (Southern Co.) | 5.28% | 5.06% | 9.03% | 3.53% | prophet | 1996 | 2026-07-03 |
| SPA | SW Power Admin | 21.13% | 25.83% | 36.96% | 22.81% | xgboost | 1950 | 2026-07-03 |
| SPP | Southwest (SPP) | 5.24% | 10.05% | 11.23% | 7.87% | xgboost | 1996 | 2026-07-03 |
| SRP | Phoenix (SRP) | 7.24% | 16.53% | 10.34% | 4.23% | xgboost | 1998 | 2026-07-03 |
| TAL | Tallahassee | 3.88% | 4.94% | 9.80% | 4.82% | xgboost | 1997 | 2026-07-03 |
| TEC | Tampa Bay (TECO) | 4.84% | 7.47% | 6.51% | 5.70% | xgboost | 1997 | 2026-07-03 |
| TEPC | Tucson (TEP) | 3.97% | 14.39% | 6.51% | 4.66% | xgboost | 1998 | 2026-07-03 |
| TIDC | Turlock ID | 6.81% | 17.47% | 12.89% | 9.50% | xgboost | 1932 | 2026-07-03 |
| TPWR | Tacoma Power | 2.10% | 5.37% | 6.38% | 2.25% | xgboost | 1998 | 2026-07-03 |
| TVA | Tennessee Valley (TVA) | 4.90% | 7.37% | 10.15% | 5.51% | xgboost | 1996 | 2026-07-03 |
| WALC | Desert SW (WAPA-DSW) | 11.54% | 15.59% | 10.27% | 10.53% | arima | 1998 | 2026-07-03 |

## Why XGBoost dominates the base models

1. **Feature engineering.** XGBoost uses all 49 engineered features —
   lagged demand (24h/168h), rolling statistics, degree-days (CDD/HDD), and
   calendar features — capturing non-linear weather↔demand interactions that
   Prophet (7 regressors) and ARIMA (no weather) miss.
2. **Tail behaviour.** Where XGBoost struggles it is usually a low-load or
   data-quality regime (SPA, LDWP, SEC) rather than a modelling gap; in one
   such case (AZPS) ARIMA's smoother extrapolation wins, which is exactly why
   the headline uses best-base-per-BA.
3. **No-leakage, recursive validation.** Post-#135 the holdout uses the
   honest `demand.shift(1)` autoregressive snapshot; post-#209 it also feeds
   the model's own predictions forward step-by-step, so these numbers reflect
   true multi-step forward-prediction skill — not one-step nowcasts, not
   reconstructed targets.

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
