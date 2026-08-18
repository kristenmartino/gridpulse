# Forecast Backtest Results

> **Provenance:** Regenerated **2026-08-07** from production GCS
> (`models/{region}/{model}/{version}.meta.json` at `latest.json`), per-BA
> rolling 7-day / 168h holdout. Models trained 2026-08-07. **The ensemble
> column is scored out-of-sample as of [#404](https://github.com/kristenmartino/gridpulse/pull/404) — see the note under the
> distribution table; figures published before 2026-08-05 were in-sample and
> optimistically biased.** These are **recursive multi-step** holdout
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
> number (median 4.32% vs 2.32% on XGBoost, as measured on the 2026-07-03 run when the two protocols were compared side by side; the table below is a later training run, so its XGBoost median differs) — not a regression, the honest
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
| min | **1.89%** (ERCOT) | **1.72%** (ERCOT) | **1.38%** (SPP) |
| median | **3.69%** | **3.69%** | **4.35%** |
| mean | **5.37%** | **4.95%** | **6.27%** |
| p90 | **9.87%** | **9.87%** | **14.27%** |
| max | **37.50%** (SPA) | **23.26%** (SPA) | **30.86%** (IID) |

> **The ensemble column is now scored strictly out-of-sample (ledger-23,
> [#404](https://github.com/kristenmartino/gridpulse/pull/404)).** Until 2026-08-05 the inverse-MAPE weights were fitted on
> the *same* 168 hours the blend was then scored against — the combination
> rule saw the answers it was graded on, so every ensemble figure this table
> ever published was optimistically biased. Weights are now fitted on the
> leading half of the holdout and the metric scores only the trailing half.
> The base-model columns were never affected: they come from
> `_holdout_metrics_*`, which this did not touch.
>
> **Do not read the whole change as the bias correction.** The table also
> moved a month of retraining (2026-07-03 → 2026-08-07). The two are
> separable, and were measured separately on the 2026-08-07 run — same 51
> BAs, same data, same models, the old estimator logged alongside the new:
>
> | ensemble median | |
> |---|---|
> | previously published (2026-07-03 data, in-sample) | 4.82% |
> | 2026-08-07 data, in-sample — what the old code would say today | 3.89% |
> | **2026-08-07 data, out-of-sample — published above** | **4.35%** |
>
> So retraining bought **−0.93 pts** and the bias correction gave back
> **+0.46 pts** (mean **+1.32**, worse on **33 of 51** BAs — the correction
> is far larger in the tail than at the median). Net, the published median
> improved; the honesty of it improved more.

**The ensemble trails the best base model in aggregate** (median 4.35% vs
3.69%) but under recursive scoring it **beats XGBoost-alone on 21 of 51**
BAs: as errors compound over the horizon, blending in Prophet and ARIMA damps
the worst single-model drift. On the majority where XGBoost is strongest the
inverse-MAPE blend (ADR-004) still lands above it, because Prophet and ARIMA
run 3–5× worse there. For per-BA *best-achievable* accuracy, read the
best-base column; for *what production serves by default*, read the ensemble
column.

**The tail is where the blend now looks worst, and that is the corrected
picture, not a regression.** Ensemble p90 is **14.27%** against XGBoost-only's
9.87%, and the fleet-worst ensemble BA (IID, 30.86%) is materially worse than
the worst base model would be. The "variance-reduction on the tail" case this
section used to make rested on **SEC: 38.63% → 13.61%** — on the 2026-08-07
run SEC is XGBoost **13.68%** and ensemble **14.72%**, i.e. the blend is now
*worse* there. That example is withdrawn rather than replaced: the honest
current statement is that the ensemble buys error decorrelation on *some*
BAs and costs accuracy on others, and the per-BA table below is the only
reliable guide to which.

**Worst 5 BAs (ensemble MAPE):** IID 30.86% · SPA 28.71% · AZPS 24.24% ·
HST 15.44% · SEC 14.72%. These tail BAs (low load / data-quality regimes)
swing materially run-to-run, which is why the table is regenerated each pass,
not frozen.

**Best base model:** XGBoost wins **42 of 51** BAs; Prophet wins **ERCOT,
LGEE, SC, TEC, TIDC** (5); ARIMA wins **AZPS, NEVP, SEC, SPA** (4). Which
model wins a given BA is not stable across retrains — on the 2026-07-03 run
Prophet won CHPD/DOPD/SEC/SOCO and ARIMA won ERCOT/SC/WALC, and only SEC and
SC held their column. Reporting best-base rather than XGBoost-only still
matters for the tail: SPA is **37.50%** on XGBoost and **23.26%** on ARIMA.

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
   consequence: the ~4.35% ensemble headline is representative of the entire
   day-ahead-to-week operating range, not an artifact of the longest horizon.
   *(Read 4.82% before 2026-08-07 — this line was not updated with the
   distribution table above it, and said `~4.8%` for eleven days after that
   table moved.)*

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
single model (on the 2026-08-07 run, ensemble median 4.35% vs best-base
3.69%; the 2026-07-03 figures behind the sweep below were 4.82% vs 4.12%).

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
| AECI | Missouri (AECI) | 4.68% | 9.60% | 13.98% | 3.39% | xgboost | 1997 | 2026-08-07 |
| AVA | Spokane (Avista) | 4.21% | 12.69% | 6.64% | 6.21% | xgboost | 1997 | 2026-08-07 |
| AZPS | Arizona (APS) | 17.67% | 14.52% | 14.36% | 24.24% | arima | 1996 | 2026-08-07 |
| BANC | Sacramento (BANC) | 4.36% | 5.78% | 5.45% | 5.60% | xgboost | 1997 | 2026-08-07 |
| BPAT | Pacific NW (BPA) | 2.08% | 4.96% | 3.72% | 1.86% | xgboost | 1996 | 2026-08-07 |
| CAISO | California (CAISO) | 3.55% | 3.81% | 3.84% | 2.85% | xgboost | 1996 | 2026-08-07 |
| CHPD | Chelan County PUD | 3.69% | 9.42% | 9.74% | 5.16% | xgboost | 1997 | 2026-08-07 |
| CPLE | Carolinas East (DEP) | 3.26% | 6.64% | 5.09% | 3.93% | xgboost | 1996 | 2026-08-07 |
| CPLW | DEP-West (NC mountains) | 2.81% | 9.98% | 4.55% | 2.58% | xgboost | 1997 | 2026-08-07 |
| DOPD | Douglas County PUD | 2.89% | 7.50% | 7.23% | 3.66% | xgboost | 1997 | 2026-08-07 |
| DUK | Carolinas West (DEC) | 2.03% | 4.55% | 5.02% | 2.08% | xgboost | 1996 | 2026-08-07 |
| EPE | El Paso (EPE) | 3.92% | 14.24% | 9.43% | 5.24% | xgboost | 1997 | 2026-08-07 |
| ERCOT | Texas (ERCOT) | 1.89% | 1.72% | 3.37% | 1.56% | prophet | 1996 | 2026-08-07 |
| FMPP | Florida Muni Pool | 4.57% | 4.87% | 6.59% | 6.49% | xgboost | 1997 | 2026-08-07 |
| FPC | Florida (Duke FL) | 4.66% | 10.02% | 8.22% | 6.90% | xgboost | 1996 | 2026-08-07 |
| FPL | Florida (FPL/NextEra) | 4.31% | 6.24% | 8.78% | 6.92% | xgboost | 1996 | 2026-08-07 |
| GCPD | Grant County PUD | 3.38% | 6.63% | 6.26% | 4.35% | xgboost | 1997 | 2026-08-07 |
| GVL | Gainesville (GRU) | 4.67% | 7.36% | 10.72% | 5.04% | xgboost | 1955 | 2026-08-07 |
| HST | Homestead | 5.56% | 12.10% | 13.99% | 15.44% | xgboost | 1997 | 2026-08-07 |
| IID | Imperial Valley (IID) | 16.88% | 27.43% | 18.86% | 30.86% | xgboost | 1885 | 2026-08-07 |
| IPCO | Idaho (Idaho Power) | 3.43% | 7.69% | 6.33% | 5.20% | xgboost | 1997 | 2026-08-07 |
| ISONE | New England (ISO-NE) | 3.68% | 5.21% | 11.72% | 3.48% | xgboost | 1996 | 2026-08-07 |
| JEA | Jacksonville (JEA) | 4.77% | 11.74% | 7.37% | 7.80% | xgboost | 1997 | 2026-08-07 |
| LDWP | Los Angeles (LADWP) | 3.53% | 5.48% | 4.81% | 4.44% | xgboost | 1996 | 2026-08-07 |
| LGEE | Kentucky (LG&E + KU) | 3.50% | 2.95% | 6.03% | 1.86% | prophet | 1997 | 2026-08-07 |
| MISO | Midwest (MISO) | 2.05% | 5.24% | 5.85% | 1.81% | xgboost | 1996 | 2026-08-07 |
| NEVP | Southern Nevada (NV Energy) | 5.34% | 5.66% | 4.74% | 4.24% | arima | 1991 | 2026-08-07 |
| NWMT | Montana (NorthWestern) | 2.19% | 10.14% | 7.45% | 3.46% | xgboost | 1986 | 2026-08-07 |
| NYISO | New York (NYISO) | 3.58% | 5.58% | 9.14% | 2.16% | xgboost | 1996 | 2026-08-07 |
| PACE | Inland West (PacifiCorp E) | 5.76% | 7.98% | 6.11% | 7.61% | xgboost | 1997 | 2026-08-07 |
| PACW | Pacific NW (PacifiCorp W) | 2.81% | 10.71% | 5.97% | 2.59% | xgboost | 1997 | 2026-08-07 |
| PGE | Portland General | 2.12% | 8.23% | 6.87% | 2.24% | xgboost | 1997 | 2026-08-07 |
| PJM | Mid-Atlantic (PJM) | 1.92% | 6.29% | 4.91% | 1.64% | xgboost | 1996 | 2026-08-07 |
| PNM | New Mexico (PNM) | 3.21% | 5.49% | 3.58% | 2.98% | xgboost | 1998 | 2026-08-07 |
| PSCO | Colorado (Xcel) | 8.99% | 13.23% | 11.76% | 14.27% | xgboost | 1946 | 2026-08-07 |
| PSEI | Puget Sound Energy | 2.55% | 8.91% | 6.07% | 2.40% | xgboost | 1997 | 2026-08-07 |
| SC | Santee Cooper | 3.42% | 2.64% | 3.86% | 1.71% | prophet | 1997 | 2026-08-07 |
| SCEG | Carolinas Mid (Dominion SC) | 3.93% | 5.54% | 5.51% | 4.72% | xgboost | 1997 | 2026-08-07 |
| SCL | Seattle (SCL) | 2.73% | 3.08% | 4.47% | 2.42% | xgboost | 1997 | 2026-08-07 |
| SEC | Seminole Electric | 13.68% | 15.85% | 12.88% | 14.72% | arima | 1997 | 2026-08-07 |
| SOCO | Southeast (Southern Co.) | 4.31% | 6.84% | 5.71% | 5.11% | xgboost | 1996 | 2026-08-07 |
| SPA | SW Power Admin | 37.50% | 30.07% | 23.26% | 28.71% | arima | 1975 | 2026-08-07 |
| SPP | Southwest (SPP) | 2.93% | 3.97% | 9.62% | 1.38% | xgboost | 1996 | 2026-08-07 |
| SRP | Phoenix (SRP) | 9.87% | 17.38% | 10.49% | 11.70% | xgboost | 1997 | 2026-08-07 |
| TAL | Tallahassee | 5.61% | 5.66% | 11.72% | 3.92% | xgboost | 1997 | 2026-08-07 |
| TEC | Tampa Bay (TECO) | 5.23% | 4.71% | 9.70% | 4.13% | prophet | 1996 | 2026-08-07 |
| TEPC | Tucson (TEP) | 3.42% | 4.55% | 4.16% | 4.64% | xgboost | 1997 | 2026-08-07 |
| TIDC | Turlock ID | 4.51% | 4.13% | 7.65% | 6.03% | prophet | 1949 | 2026-08-07 |
| TPWR | Tacoma Power | 3.75% | 8.16% | 6.06% | 4.53% | xgboost | 1997 | 2026-08-07 |
| TVA | Tennessee Valley (TVA) | 2.14% | 5.39% | 5.39% | 1.73% | xgboost | 1996 | 2026-08-07 |
| WALC | Desert SW (WAPA-DSW) | 10.27% | 20.03% | 10.81% | 11.97% | xgboost | 1997 | 2026-08-07 |

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
