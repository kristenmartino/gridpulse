# ARIMA order selection: does the search need the weather regressors? (#297)

**Verdict: no.** The order search stays univariate. The bug that made it
univariate by accident is removed; the behaviour it produced is kept,
deliberately.

Fleet run **2026-07-29, all 51 BAs, zero failures**. Reproduce:
`python -m scripts.arima_order_exog_study --all`. Raw results:
[`ARIMA_ORDER_EXOG_STUDY.json`](ARIMA_ORDER_EXOG_STUDY.json).

> **This supersedes the 10-BA run of 2026-07-28** that PR #365 shipped on. The
> conclusion is unchanged and better supported. Several *individual* numbers
> from that run did not reproduce — one BA reversed sign — and §"How stable is
> this?" below is the correction, kept rather than quietly overwritten.

---

## The defect, as filed

`models/arima_model.py::_auto_select_order` passed `exogenous=exog_sub` to
`pm.auto_arima`. pmdarima 2.x renamed that parameter to `X`, and `auto_arima`
accepts `**fit_args` — so the old name raised nothing, was swallowed, and the
stepwise (p,q,P,Q) search ran on a **univariate** view of demand while the
final `SARIMAX(..., exog=exog)` fit used all five weather regressors.

The code stated an intent it did not execute. #297's fix sketch was the
obvious one: `exogenous=` → `X=`, then re-measure before merging.

## What the fleet measurement found

Both arms fit the **same** final model (SARIMAX with exog — that half was
never broken). Only the selected order differs:

- **A — control:** order selected univariately (the pre-#297 behaviour).
- **B — "fixed":** order selected with `X=exog`.

| | |
|---|---|
| BAs measured | **51** (0 failures, 0 skips) |
| Order actually changed | **38** — the other 13 are exact no-ops |
| Better / worse | 18 / 20 |
| Median Δ | **0.00 pts** |
| Mean Δ | **−0.92 pts** |
| **Total gained** across all BAs that improved | **+14.88 pts** |
| **Total lost** across all BAs that worsened | **−61.71 pts** |
| Worst single BA | **ISONE −19.18** (13.93 → 33.11 sMAPE) |
| Best single BA | CAISO +3.87 (8.44 → 4.57) |
| Search cost | 21.9 s → 58.7 s median (**2.7×**) |

**The argument is asymmetry, not average.** On a coin-flip count the fix is
even (18 up, 20 down, 13 unchanged). What decides it is that **the losses are
4.1× the gains in aggregate**, and the tail is not comparable: the worst
outcome is a BA getting 2.4× worse, the best is a BA getting ~1.8× better.
Taking this change means accepting a much heavier left tail for no expected
gain, at 2.7× the search cost.

## Why it loses — mechanism, fleet-validated

The harm concentrates precisely where the exog-aware search **drops the
seasonal moving-average term** (`Q: 1 → 0`):

| exog-aware search… | n | mean Δ | median Δ | worst |
|---|---:|---:|---:|---:|
| **drops the seasonal MA term** | 14 | **−2.99** | −1.75 | −19.18 |
| keeps it | 24 | −0.21 | +0.14 | −6.75 |

Given five weather columns, AIC credits them for variance the seasonal MA was
carrying and prunes it as redundant. Defensible in-sample; wrong across 168
recursive hours, where that term carries the daily cycle robustly and
pointwise weather regression has to be right at *every* step to replace it.

Note the effect is specific to the **MA** term, not to seasonal complexity in
general: bucketing instead by "did total seasonal order (P+Q) go down" gives
−1.45 vs −1.13, which separates almost nothing. Seasonal AR is not what is
load-bearing here.

**The study was generous to the arm that lost.** It fed arm B *perfect* future
weather. Production feeds forecast weather, whose error grows with horizon —
so the exog-aware orders would fare no better in production, and plausibly
worse.

## How stable is this? (the correction)

A 10-BA run on **2026-07-28** and the 51-BA run on **2026-07-29** differ only
in that the rolling window moved one day. Per-BA results moved a great deal:

| BA | Δ on 07-28 | Δ on 07-29 | note |
|---|---:|---:|---|
| CAISO | **−7.24** | **+3.87** | **sign reversed** |
| WALC | +4.59 | +0.04 | gain evaporated |
| PJM | −9.04 | −5.01 | |
| MISO | −2.90 | −3.86 | control sMAPE 7.62 → 3.91 |
| ERCOT | −4.41 | −6.75 | control sMAPE 8.57 → 4.98 |
| LDWP | −0.71 | 0.00 | order stopped changing |
| PSCO | +0.17 | 0.00 | order stopped changing |

The *control* arm alone moved by 3.6 points on ERCOT and halved on MISO
between adjacent windows. **A single 168-hour holdout is not enough to rank an
individual BA**, and any per-BA figure from this study should be read as one
draw, not as that BA's number.

What *is* stable is the aggregate and the mechanism: both runs agree the fix
is net-negative with a much heavier downside tail, and both show the damage
tracking the loss of the seasonal MA term. That is what the decision rests on.

Per-BA deltas from the earlier run are preserved in the JSON as
`prior_run_delta_smape_pts` so the variance stays visible.

## What shipped

The search stays univariate, and now says so. The dead `exogenous=` kwarg is
gone — the code no longer states an intent it does not execute — and no `X` is
passed either, with the reasoning at the call site.

`tests/unit/test_arima_auto_order_kwargs.py` pins:

1. no kwarg in the call is absent from the installed pmdarima's signature
   (the *class* of bug — the next rename cannot hide the same way);
2. `X` is not passed, so re-applying the "obvious fix" fails with a pointer
   to this document;
3. every `SARIMAX` construction still passes `exog` — univariate *order
   selection* is the decision, a univariate *model* would not be;
4. `auto_arima` still accepts `**kwargs`, so guard 1 is known to be guarding
   something real.

## An operational note

`_auto_select_order` is reached only when a region has **no cached order**.
`jobs/training_job.py::_read_cached_arima_order` reads `order`/`seasonal_order`
from the previous model's persisted metadata, `train_arima` skips the search
when both are present, and the newly-trained model re-persists them. There is
no invalidation path, so **a once-selected order propagates forever**.

- Every currently persisted order was selected univariately → this change is a
  no-op in steady state; it governs new BAs and any cold model store.
- The 2.7× search cost is likewise a cold-cache cost, not a daily one.
- 13 of 51 BAs select the same order either way, so for a quarter of the fleet
  the question is moot regardless.

If a future change *should* re-select orders fleet-wide, it needs an explicit
invalidation; nothing in the pipeline provides one today.

## Limits

1. **One holdout window per run** (168 h, ending 7 days back). Demonstrably the
   binding limit — see the correction above. A defensible per-BA ranking needs
   several windows; the fleet verdict does not.
2. **Known future weather**, which favours the arm that lost.
3. **sMAPE**, matching the other A/B studies in this repo; the serving
   scorecards use mean MAPE (`docs/BACKTEST_RESULTS.md` §8 on cross-artifact
   metric mixing applies).
4. ARIMA is one ensemble member, weighted `(1/MAPE)³` under ADR-004. A
   per-model sMAPE delta is not an ensemble delta.
5. The mechanism split (seasonal-MA-dropped vs not) is observational across 38
   BAs, not a controlled experiment — nothing here forces `Q` and re-measures.
