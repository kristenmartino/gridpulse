# Training-window study (#231 follow-up) — working brief

**Status:** in progress. Written 2026-08-21 as a durable handoff so the
findings below — several of them corrections to earlier findings in the same
investigation — do not have to be re-derived or re-broken by a later session.

**Read the confidence tiers.** This document deliberately separates what was
measured from what was relayed from what was inferred. Three conclusions in
this investigation were stated confidently and later retracted; the tiering
exists so the next reader can tell which claims have actually been checked.

---

## 1. What #231 asked, and why it cannot be done as written

[#231](https://github.com/kristenmartino/gridpulse/issues/231) proposed
extending the training fetch to **365 days (Prophet-scoped)** to re-enable
`yearly_seasonality`.

**VERIFIED — the literal ask is unreachable.**
`models/prophet_model.py:53` gates the feature at
`YEARLY_SEASONALITY_MIN_DAYS = 730`, not 365, following #281 (a 365-day
window produced a spurious −11.8 GW phantom seasonal decline). A 365-day
fetch would leave `yearly_seasonality` off in every window. The issue's
title describes a change that cannot have its intended effect.

**VERIFIED — production trains on ~90 days.** Traced end to end:
`jobs/training_job.py:807` → `phases.fetch_region_data(region)` →
`jobs/phases.py:275` `fetch_demand(region)` (no override) →
`data/eia_client.py:325` default of 90 days. `config.TRAINING_WINDOW_DAYS = 365`
is dead config, referenced only in a docstring.

> Note for future readers: an earlier pass asserted this same conclusion while
> citing only `jobs/phases.py:275` — a **scoring**-job call site — as evidence
> about the **training** job. The conclusion happened to survive tracing, but
> the evidence originally offered did not support it. Deposited as
> `wrong-artifact-cited`.

---

## 2. Measured results

Scripts: `scripts/xgboost_730d_fetch_study.py` (v1, **defective — see §3**),
`scripts/xgboost_730d_fetch_study_v2.py` (corrected),
`scripts/arima_730d_fetch_study.py`, `scripts/prophet_730d_fetch_study.py`,
`scripts/xgboost_recency_weight_study.py` (v3, running).
All route verdicts through `models/rolling_eval.py` per
`docs/EVALUATION_POLICY.md`.

### XGBoost — corrected (v2), 6 BAs, fixed trailing windows, recursive scoring

| BA | verdict | WAPE 90d→730d | bias 90d→730d | satisficing |
|---|---|---|---|---|
| CAISO | inconclusive | 5.45 → 5.57 | −1.60 → −4.18 | FAIL |
| ERCOT | **control wins** | 2.19 → 2.60 | −0.92 → −2.18 | FAIL |
| TVA | inconclusive | 3.55 → 3.64 | −0.58 → −2.39 | FAIL |
| DUK | inconclusive | 3.54 → 3.01 | −0.72 → −0.94 | PASS |
| SCEG | inconclusive | 3.88 → 3.50 | −1.57 → −0.42 | PASS |
| CPLW | inconclusive | 4.11 → 3.65 | −1.65 → −0.60 | PASS |

**Zero decisive wins for the longer window.** Three of six push
under-forecasting past the ±2% satisficing veto.

### SARIMAX — 730d vs 90d, 6 BAs

Zero unvetoed wins. **5 of 6 fail the bias constraint** (−2.49% to −7.45%,
all under-forecasting). SCEG's WAPE win (86% of windows, 3.2× stderr) was
vetoed on bias — a metric-only read would have called it a win.

### Prophet — 800d + `yearly_seasonality` ON vs 90d (partial: 3 of 6 BAs, run interrupted)

CAISO −1.447 pts (control wins 100% of windows, 6.3× stderr, fails both
constraints); ERCOT −1.185 (control wins 86%, fails MAPE regression);
TVA inconclusive. **No wins.** `yearly_seasonality` was confirmed ON in all
treatment windows, so this is a clean read of the feature, not a silent no-op.

### Production baseline

> **Provenance — these numbers are COPIED, not measured here.** Source:
> [`docs/CANONICAL_FACTS.md`](CANONICAL_FACTS.md) (and
> [`docs/BACKTEST_RESULTS.md`](BACKTEST_RESULTS.md)), read 2026-08-21, models
> trained 2026-08-07. **`CANONICAL_FACTS.md` is authoritative; if it and this
> table ever disagree, it wins and this table is stale.** They are restated
> here only as the yardstick the study's own numbers are read against.
> Restating a canonical fact is how `BACKTEST_RESULTS.md` ended up publishing
> a "~4.8% ensemble headline" 90 lines below its own corrected table and
> leaving it wrong for 11 days (PR #404) — hence this banner rather than a
> bare copy.

Every comparison in this study should be read against what production
actually serves, not against XGBoost alone in a harness:

| | best base model | ensemble (served) |
|---|---|---|
| median MAPE | 3.69% | **4.35%** |
| mean | 4.95% | 6.27% |
| p90 | 9.87% | 14.27% |
| max | 23.26% (SPA) | 30.86% (IID) |

**The ensemble trails the best base model in aggregate** (4.35% vs 3.69%).
Worst BAs: IID 30.86%, SPA 28.71%, AZPS 24.24% — note SPA is also the BA
whose apparent WAPE win was vetoed on a +14.13% over-forecast bias in the
(withdrawn) v1 study, and IID/SPA sit at the bottom of production too.

**Cumulative MAPE by lead time** (XGBoost | Prophet | SARIMAX):

| lead | XGB | Prophet | SARIMAX |
|---|---|---|---|
| 1h | 0.96% | 2.65% | 4.70% |
| 24h | 4.14% | 5.10% | 7.61% |
| 48h | 4.32% | 5.43% | 8.41% |
| 72h | 4.26% | 5.50% | 9.46% |
| 168h | 4.12% | 5.34% | 7.51% |

Error jumps sharply in the first 24h then **plateaus** — 4.32 → 4.26 → 4.12
across 48/72/168h. That is weak evidence *against* unbounded error
compounding in the recursive regime, but it measures magnitude (MAPE), not
direction, so it does not settle the bias question. `BACKTEST_RESULTS.md`
also states **recursive MAPE runs ~2× teacher-forced**, which is precisely
why the v2 scoring correction (§3) changed the answer.

### Cost facts (VERIFIED, and both contradict in-repo assumptions)

- **SARIMAX's fit scales ~linearly, not cubically.** Measured on CAISO:
  2,160 rows → 11s, 4,320 → 25s, 8,640 → 49s (fitted exponent ≈ 1.08). The
  `# ARIMA is O(n³)` comment at `models/arima_model.py:75` overstates by
  orders of magnitude. A Kalman filter's forward pass is O(n) in
  observations, so near-linear is the expected result.
- **`auto_arima`'s selected order is NOT stable across rolling windows.**
  8 CAISO windows produced **5 distinct** `(order, seasonal_order)` combos,
  confirmed deterministic on identical input. Caching one order across
  backtest windows — the tempting cost saving, and what production's
  `cached_order` fast path does on a *daily* refresh — would change what a
  study measures, not just what it costs.
- **EIA coverage is not a constraint.** All 6 sampled BAs return ~100%
  hourly coverage over 1,500 days (4.1y), *including* CPLW (~42 MW). The
  expectation from #283 that V3.ζ BAs would be too thin for multi-year
  training does not hold at this horizon.

---

## 3. RETRACTED: the v1 result (21/28 BAs "decisive wins")

An earlier pass reported 21 of 28 BAs as decisive wins for a 730-day window.
**That result is withdrawn.** Two defects, each sufficient on its own:

1. **Strawman control.** `rolling_origin_splits` returns `slice(0, test_start)`
   — train on everything before the holdout. Fed a 90-day dataset, the newest
   window trained on ~83 days but the oldest on ~35. Six of seven windows
   compared a full-history treatment against a starved control, conflating
   "more history helps" with "the control was crippled."
2. **Wrong scoring regime.** Scoring was one-shot against holdout rows carrying
   *real* `demand_lag_1h`. Production forecasts recursively, feeding the model
   its own predictions (ADR-010 exists because the non-recursive holdout is
   blind to that regime). 419 of 420 fits put an autoregressive lag top of the
   feature-importance list.

Correcting both flipped **all four** re-tested wins to inconclusive or
control-wins, and reversed ERCOT's sign.

A third instrument was also invalid: the **train/holdout MAE gap** used as an
overfitting check. With 6,000 trees and no early stopping on the final fit,
the 90d arm memorizes its training set (measured train MAE **0.1 MW** vs
17.6 MW at 730d), so the two arms' gaps were never comparable. Decomposing
CAISO's gap showed the shrink came from holdout error falling (−144 MW), not
train error rising (+17.5) — the *conclusion* survived, the *instrument* did
not. Dropped rather than repaired; the rolling holdout covers this.

---

## 4. The live caveat: none of the above tested seasonality

**All test windows in §2 fell in June–August.** `rolling_origin_splits` walks
consecutively backward, so 8 windows span 56 days — structurally incapable of
crossing a seasonal turn. Verified on CAISO: windows covered
2026-06-22 … 2026-08-16, months `{Jun, Jul, Aug}`.

This matters because #283's Phase 0 found the weather-normal beat recent-28d
**~10:2 at seasonal turns and was "a wash mid-season."** If the benefit of
more history is concentrated at seasonal turns, then a null result measured
entirely in mid-summer is exactly what that prior predicts — and is
uninformative about the seasonality argument rather than evidence against it.

So §2 supports: *"more history does not help in mid-summer."* It does **not**
support "more history does not help." Deposited as `seasonally-blind-holdout`.

### RESULT (2026-08-21): with seasonal spread, the sign REVERSES

`scripts/xgboost_recency_weight_study.py` — 12 windows at 720h stride, all
12 calendar months covered, sweeping *half-life* rather than window length
(a hard window is already a weighting scheme: 1 inside, 0 outside; decay is
its smooth generalisation). Turn/peak segmentation pre-declared in source.

**CAISO** (control `hard_90d` 5.24% WAPE, bias −1.34%):

| arm | WAPE | bias | all | turn | peak |
|---|---|---|---|---|---|
| hard_730d | 4.66% | −2.11% | +0.58~ | +0.53~ | +0.64~ (SAT-FAIL) |
| decay_hl_180d | **4.61%** | −1.82% | +0.63~ | +0.60~ | +0.66~ |
| decay_hl_730d | 4.66% | −1.93% | +0.59~ | +0.42**WIN** | +0.75~ |

**DUK** (control `hard_90d` 4.49% WAPE, bias +0.29%):

| arm | WAPE | bias | all | turn | peak |
|---|---|---|---|---|---|
| hard_730d | 2.97% | −0.74% | +1.52**WIN** | +1.96**WIN** | +1.09~ |
| decay_hl_180d | **2.96%** | −0.67% | +1.54**WIN** | +1.69**WIN** | +1.38~ |
| decay_hl_365d | 2.96% | −1.02% | +1.53**WIN** | +1.76**WIN** | +1.30~ |
| decay_hl_30d | 4.16% | −0.83% | +0.33~ | +0.77**WIN** | −0.11~ |

Findings:

1. **The sign reverses versus the summer-only studies.** CAISO v2 (Jun–Aug):
   90d 5.45% vs 730d 5.57%, treatment *worse*. CAISO v3 (all months): 5.24%
   vs 4.66%, treatment *better*. DUK v2 was inconclusive; DUK v3 is a
   decisive win at **4.49% → 2.96%, −34% relative**. The only change is
   seasonal coverage of the test windows.
2. **The pre-declared segmentation earned its keep.** DUK's turn-month
   deltas are decisive (+1.44 to +1.96) while peak-month deltas are positive
   but inconclusive (+0.91 to +1.38) — #283 Phase 0's "large at turns, wash
   mid-season" shape, reproduced on demand data. An aggregate-only report
   would have diluted this into one ambiguous number.
3. **The half-life curve is non-monotone.** `decay_hl_30d` is the weakest
   long arm on DUK (+0.33, inconclusive) — too short to hold a season.
   `decay_hl_180d` is at/near the top on both BAs. The answer to "how much
   history" is neither 90 days nor everything: a **180–365 day half-life**.
4. **Decay dominates hard truncation on the bias constraint** — the
   constraint that had been vetoing long windows. CAISO at matched volume:
   `hard_730d` −2.11% (SAT-FAIL) vs `decay_hl_730d` −1.93% (passes);
   `decay_hl_30d` −1.10% beats even the control's −1.34%.

**This retracts the "one coherent negative result" framing** given earlier in
this investigation. §2's null results measured mid-summer only and did not
test the regime where the effect lives. #231's *underlying intuition* — more
history helps — is vindicated for XGBoost.

**It does NOT rescue #231 as written.** 365 days still fails to clear the
730-day `yearly_seasonality` gate, and Prophet with the yearly term ON lost
decisively on 2 of 3 BAs. The win here is XGBoost + recency weighting, a
different mechanism than the issue proposed.

### FULL 6-BA RESULT (2026-08-21): mixed, not a fleet-wide win

The 2-BA pilot above overstated it. Extending to all six:

| BA | control WAPE | best long arm | Δ | verdict | bias |
|---|---|---|---|---|---|
| DUK | 4.49% | 2.96% | +1.53 | **decisive win** | passes |
| CPLW | 4.79% | 3.10% | +1.69 | **decisive, all 3 segments** | passes |
| SCEG | 5.57% | 3.74% | +1.84 | decisive WAPE | **VETOED** (+2.96% over-forecast) |
| CAISO | 5.24% | 4.61% | +0.63 | inconclusive | mixed |
| ERCOT | 3.01% | 2.93% | +0.07 | **negative** — most arms worse | 6/8 fail |
| TVA | — | — | — | **VOID — see below** | — |

**2 clean wins, 1 vetoed, 1 weak positive, 1 negative** across the five valid
BAs. The effect is real for some BAs and absent-to-harmful for others.
**Do not extend the production window fleet-wide on this evidence.**

#### TVA is void: corrupt EIA data, not a modelling result

TVA's long-history arms returned 18.48% WAPE (`decay_hl_365d`) and 10.46%
(`hard_730d`) against a 4.65% control. Cause: TVA's EIA history contains
**two ~9.9 million MW spikes and two non-positive values** (normal range
~12,000–31,000 MW), concentrated in 2024-Q1/Q2. Short arms end recently and
never see them; every long arm trains straight through them.

An outlier scan of all six BAs found **TVA is the only one affected** —
4 bad records out of 35,856 rows (0.011%). The other five results stand.

Two consequences:

1. **Outlier filtering is a hard prerequisite for any window extension.**
   This study went `fetch_demand → merge → engineer_features` with no guard.
   Production trains on 90 days so a corrupt patch ages out within a quarter;
   at 730 days you permanently inherit every bad record in the history.
2. **Extreme sensitivity is itself a finding.** 0.011% of rows moved a BA
   from 4.65% to 18.48% WAPE. With 6,000 trees, no early stopping, and no
   input guard, a handful of records is catastrophic.

**No past production incident — an earlier draft of this section claimed one
and was wrong.** The 9.9M MW records are dated January 2024, and that draft
suggested production's trailing 90-day window would have contained them at
the time. Two checks kill it:

- **GridPulse did not exist then.** Initial commit 2026-02-28; the scheduled
  Cloud Run Jobs that write vintages began 2026-04-19. There are no 2024
  vintages to inspect.
- **The arithmetic never worked anyway.** A trailing 90-day window in 2026
  covers ~May–Aug 2026. January 2024 is ~2.5 years back, outside every
  configuration production has ever run.

**What survives is the inverse, and it matters more:** those records sit in
EIA's history *now*, invisible to production precisely because the window is
short. Any extension past ~2.5 years pulls them in. The outlier guard is
therefore a **precondition for the proposed change**, not remediation for a
past event — and production's short window has been accidentally protecting
it, which is worth weighing in the decision.

**Caveats:** six BAs of 51, one of them void. Effect size varies widely
(DUK/CPLW +1.5 to +1.7 decisive; CAISO +0.6 inconclusive; ERCOT negative).

---

## 5. Confidence tiers on the research directions

Alternative model families were **never considered** in this repo — grep of
`PRD.md`, `TECHNICAL_SPEC.md`, `docs/internal/NEXT_UP.md`, `CLAUDE.md` for
LightGBM/CatBoost/LSTM/N-BEATS/transformer/TFT/Chronos returns nothing.
Training is strictly per-BA (`for region in regions:`,
`jobs/training_job.py:1066`). `models/skill.py` benchmarks only against
seasonal-naive, which its own docstring calls "deliberately the dumbest
defensible one."

**VERIFIED (checked in this repo):** the three facts above; weighting support
(`XGBRegressor.fit` accepts `sample_weight`; `Prophet.fit(self, df, **kwargs)`
does not; statsmodels SARIMAX has none); forecast vintages are stored
(`write_vintage_records`, `jobs/phases.py:3196`).

**RELAYED, NOT VERIFIED — treat as leads, not evidence.** That Chronos-2
(Oct 2025) gains substantially from covariates came from a *search-result
summary*, not a primary paper, and carries no effect size. Direction without
magnitude cannot support a build decision — 0.2% and 8% imply opposite
answers.

### Reconciliation: effect sizes from primary sources (2026-08-21)

**Positive, on real electricity data, at the level we serve.** Brégère &
Huard (EDF/INRIA; 1,545 UK households, 136-node hierarchy) report
**bottom-level MSE improvements of 8.0% (GAM base), 18.8% (Random Forest),
14.7% (autoregressive)** — comparable to or larger than their
total-consumption gains. This matters because GridPulse serves *per-BA*
forecasts; reconciliation that only improved aggregates would be worthless
here. Wickramasuriya et al. (2019, original MinT — tourism, not electricity)
corroborate: *"for all levels of disaggregation, MinT(Shrink) forecasts
improve"*, with bottom-up by contrast making things **worse** at almost all
levels.

**Three caveats on those numbers:**
1. They are **MSE** — the most flattering scale. ~13% MSE ≈ ~6.7% RMSE;
   WAPE/MAE gains would be lower still. Do not read them as WAPE points.
2. Brégère & Huard used **OLS/projection reconciliation, not MinT-shrink** —
   same family, different estimator.
3. Wickramasuriya's own text notes MinT improves "almost always... with a
   few rare exceptions" — bottom-level degradation is possible.

**The gap:** the Athanasopoulos et al. (2023) review surveys *every*
published MinT-on-electricity application (EDF tariff groups; 13 Texas
buildings; Ben Taieb's 5,701 smart meters; 318 CA solar plants; ISO-NE) and
reports **no quantitative effect size for any of them**. The specific
combination we would build — MinT + electricity load + ~51 nodes — has no
verifiable published number.

**Two cautions:** the one real-electricity paper found gains **plateau or
reverse above ~32 nodes** in some configurations, and we are at 51. And both
GEFCom2012 and GEFCom2017 shipped hierarchical load data that **no
participant exploited** — soft evidence (competition time pressure, metrics
that may not reward coherence), but not nothing.

**Cost is a non-issue:** MinT at n≈50 measures **0.172s per reconciliation
pass**. FlowRec's O(n² log n) is real but unnecessary at this scale.

### CORRECTED 2026-08-21: the global-model case is much weaker than first stated

An earlier pass in this investigation ranked a global (cross-BA) model as the
single largest opportunity, on the reasoning that pooling 51 series would let
small noisy BAs borrow strength from large ones. Primary-source extraction
contradicts that on three counts:

- **Model-dependent sign flip in the closest analogue.** arXiv 2507.11729
  (42 Alberta load series — nearest published match to our n=51, same domain):
  globalized **LightGBM improved MAPE 12.80% → 11.57%**, but globalized
  **XGBoost regressed 12.86% → 13.11%**. Ridge regressed on nMAE and lost
  −2.15% to −2.59% under data drift. XGBoost is our primary forecaster
  (ADR-005), and it is the arm that got worse.
- **The stated mechanism is backwards.** arXiv 2204.00493 (1,000 load series,
  4 aggregation tiers) found the **smallest, noisiest series regressed** under
  pooling (−0.28%) while large smooth aggregates gained 3.24–4.38%; global beat
  local on 35% of individual series vs 95–98% of aggregate series. Small BAs
  are the risk case, not the expected beneficiary.
- **Heterogeneity, not series count, is decisive — and we are heterogeneous.**
  A controlled simulation at n=100 heterogeneous series found **local beat
  global by ~4.8% SMAPE**. Two papers disagree outright on the same 72-series
  CIF2016 dataset. No paper tests n≈51 across a 42 MW–100 GW scale span.
  M5-scale results (30,000+ series) are explicitly non-transferable per the
  authors who discuss it.

**What survives:** test **LightGBM** rather than XGBoost if globalizing;
**cluster by scale/behaviour before pooling** (the variant the literature
supports) rather than naive 51-way pooling (the variant it warns about); and
expect small BAs to be the risk. Also **no paper measured 1 global fit vs N
local fits** — the "global is cheaper" claim is inference, not evidence.

**INFERRED (reasoning, not measurement):** that reconciliation and conformal
prediction are cheapest because they are post-processing on stored vintages;
that pooling should most help the small noisy BAs; the infra/cost estimates;
the recommended ordering.

**RESOLVED 2026-08-21 — compounding is real but NOT universal.** The
hypothesis that recursive error compounds with horizon (used twice: to
explain the under-forecast bias, and to motivate stepwise conformal
prediction) was measured. Mean signed % error by horizon step:

| step | CAISO 90d | CAISO 730d | DUK 90d | DUK 730d |
|---|---|---|---|---|
| 1 | −0.75% | −0.80% | −0.20% | −0.06% |
| 24 | −6.68% | −7.61% | −4.76% | −5.15% |
| 48 | −3.54% | −8.27% | −2.97% | −1.69% |
| 96 | −2.19% | −10.41% | −1.11% | −0.13% |
| 168 | **+1.79%** | **−11.18%** | **+0.49%** | **−0.00%** |

**Only CAISO's 730d arm compounds** — monotonically worse from step 24 to
−11.18%. The other three dip near step 24 and recover toward zero. This maps
exactly onto the satisficing results: CAISO 730d failed the bias veto
(−4.18%), DUK 730d passed (−0.94%). CAISO's aggregate −4.18% is therefore
**not a uniform level shift but the mean of a drifting curve**, which makes
the 730d failure mode *late-horizon drift* — potentially addressable with a
horizon-scoped remedy rather than by abandoning long windows.

*Lead, not finding:* every arm dips near step 24, which is exactly where
`demand_lag_24h` becomes self-referential in recursive inference (real data
before, own predictions after). Untested.

*Limitations:* n=2 BAs; the script's own first-24h-vs-last-24h "drift" label
is crude and mislabels non-monotonic dip-and-recover curves as "IMPROVING" —
read the per-step numbers, not the label.

**Consequence for the recommendations:** stepwise conformal prediction is
*partially* justified — horizon-dependent intervals would help the
compounding cases specifically, but compounding is not the universal
phenomenon an earlier pass of this investigation implied.

---

## 6. Plan

**Phase 0 — settle the unknown.** Step-wise bias diagnostic (CPU-bound; queue
behind the running pilot). Effect-size extraction from primary papers
(I/O-bound; runs in parallel). Establish the real production baseline from
`docs/BACKTEST_RESULTS.md` + the benchmark payload so comparisons are against
the *served ensemble*, not XGBoost alone in a harness. Add a Tao Hong vanilla
MLR benchmark — seconds to fit, and it answers whether three models and 49
features earn their complexity over a well-specified linear model.

**Phase 1 — retrospective, on stored vintages.** MinT reconciliation and
conformal calibration both evaluate against forecasts already produced. No
new inference runs, no infra change.

**This is now the recommended next build, and the reason is test-cost
asymmetry rather than effect size.** Reconciliation converts literature
inference into a real number on our own 51-BA hierarchy for near-zero cost —
vintages are already stored, compute is sub-second, nothing in the serving
path changes until we decide it should. The global model requires building a
new training pipeline (breaking the per-BA model store and ADR-010's
acceptance gate) *before* it yields any evidence, on a weaker prior. Learn
cheaply first.

**Phase 2 — global (cross-BA) model,** on the v3 harness. This attacks the
data-quantity question on the *cross-sectional* axis (51 BAs × current data)
rather than the *temporal* axis (one BA × stale data), which is the axis that
has now failed repeatedly. **Run it as a genuine experiment with an uncertain
prior, not as an expected win** — see the correction in §5. Concretely: use
LightGBM (not XGBoost), cluster BAs by scale/behaviour before pooling, and
report small-BA results separately, since that is where the literature
predicts regression.

**Phase 3 — Chronos-2 zero-shot with covariates** as a benchmark, to find out
whether the current stack beats a pretrained model that required no training.

Sequencing note: CPU is the binding constraint, not context. Studies must run
serially on one machine — concurrent runs slow each other *and* corrupt the
timing measurements that half of this analysis depends on.

---

## 7. Phase 1 RESULT (2026-08-21): reconciliation does not help this fleet

`scripts/reconciliation_study.py`, two complete sub-hierarchies, base
forecasts through the production recursive path, `W` estimated only from
strictly-earlier windows.

| | Northeast (3 BAs) | Southeast (16 BAs) |
|---|---|---|
| independent aggregate model | 11.38% WAPE | 24.14% WAPE |
| sum-of-parts at top | 4.18% | **3.35%** |
| bottom-level base | 5.53% | 5.96% |
| `mint_shrink` vs base | −0.042 (noise) | −0.163 (wins 51% of windows) |
| `ols` vs base | −12.2 pts | **−246.7 pts** |

**The hypothesis is refuted where it should have been strongest.** The
published mechanism is small, noisy series borrowing strength from a more
reliable aggregate (Brégère & Huard's bottom level was individual
households). Southeast's small half — 8 BAs averaging 991 MW — got **worse**
under MinT (6.90% → 7.08%), as did the large half (5.02% → 5.15%). The size
split was pre-declared precisely so a real small-BA gain could not be
averaged away; there was none to hide.

**Why, and it is structural rather than a modelling failure.** Sum-of-parts
(3.35%) beats the bottom-level mean (5.96%), so aggregation genuinely
denoises — 16 independent recursive trajectories have partially cancelling
errors. A single 168-step recursive trajectory of the aggregate drifts
freely and lands at 24.14%. The gap widened from 2.7x (Northeast) to 7x
(Southeast): more series means more cancellation for bottom-up and no help
for the direct model. So the top-down forecast carries no information worth
reconciling toward. MinT correctly discounts it; OLS trusts it and produces
252% WAPE.

A better aggregate model might reach 5–6%, but it cannot capture error
cancellation across independent trajectories — that advantage exists only in
the bottom-up direction.

**Scope of this negative result.** It tests the **168h recursive** regime,
where drift dominates. At short horizons recursive drift is far smaller and
a direct aggregate model could be competitive — and the `drift_horizon`
benchmark scores **24h/48h leads**. Reconciliation is not dead in general;
it is dead for the long-horizon recursive product, which is the main one
GridPulse serves. A short-horizon retest is the one avenue this result does
not close.

**Two harness bugs found on the way, both by the cheap Northeast validation
rather than by review:**
1. The aggregate forecast was first built as `F.sum(axis=0)` — coherent by
   construction, making every arm a mathematical no-op with byte-identical
   WAPE across four methods.
2. `wx_cols` selected every numeric column of the *featured* frame, so demand
   lags and rolling means were being load-weighted across BAs. The lag of a
   sum is not the weighted sum of lags.

A third fix — per-BA unaveraged weather, replacing a spatial mean — was
applied and **did not help** (11.31% → 11.38%), which is what redirected the
diagnosis from "my features are wrong" to the structural explanation above.
