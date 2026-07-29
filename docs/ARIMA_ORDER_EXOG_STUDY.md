# ARIMA order selection: does the search need the weather regressors? (#297)

**Verdict: no — and passing them is actively harmful.** The order search stays
univariate. The bug that made it univariate by accident is removed; the
behaviour it produced is kept, deliberately.

Run 2026-07-28. Reproduce: `python -m scripts.arima_order_exog_study --all`.
Raw results: [`ARIMA_ORDER_EXOG_STUDY.json`](ARIMA_ORDER_EXOG_STUDY.json).

---

## The defect, as filed

`models/arima_model.py::_auto_select_order` passed `exogenous=exog_sub` to
`pm.auto_arima`. pmdarima 2.x renamed that parameter to `X`, and `auto_arima`
accepts `**fit_args` — so the old name raised nothing, was swallowed, and the
stepwise (p,q,P,Q) search ran on a **univariate** view of demand while the
final `SARIMAX(..., exog=exog)` fit used all five weather regressors.

The code said one thing and did another. #297's fix sketch was the obvious
one: `exogenous=` → `X=`, then re-measure before merging.

## What the measurement found

Both arms fit the **same** final model (SARIMAX with exog — that half was
never broken). Only the selected order differs:

- **A — control:** order selected univariately (the pre-#297 behaviour).
- **B — "fixed":** order selected with `X=exog`.

Per BA: 120 days of settled EIA demand + ERA5 archive weather ending 7 days
back, engineered through the production `engineer_features`, last 168 h held
out, both arms forecasting the holdout with **known** future weather.

| BA | control order | "fixed" order | control sMAPE | "fixed" sMAPE | Δ pts |
|---|---|---|---:|---:|---:|
| **PJM** | (2,0,0)(1,1,0,24) | (1,0,0)(0,1,0,24) | 9.18 | **18.22** | **−9.04** |
| **CAISO** | (1,0,0)(1,1,1,24) | (1,0,0)(0,1,0,24) | 5.18 | **12.42** | **−7.24** |
| **ERCOT** | (2,0,0)(1,1,0,24) | (1,0,0)(1,1,0,24) | 8.57 | **12.98** | **−4.41** |
| **MISO** | (2,0,0)(0,1,1,24) | (1,0,0)(1,1,0,24) | 7.63 | **10.52** | **−2.90** |
| LDWP | (1,0,0)(0,1,1,24) | (1,0,1)(0,1,1,24) | 13.06 | 13.77 | −0.72 |
| FPL | (2,0,1)(0,1,1,24) | (1,0,1)(0,1,1,24) | 4.44 | 4.60 | −0.17 |
| PACE | (2,0,0)(0,1,1,24) | (1,0,0)(1,1,0,24) | 3.94 | 3.88 | +0.06 |
| IPCO | (2,0,1)(0,1,1,24) | (2,0,0)(0,1,1,24) | 5.57 | 5.47 | +0.10 |
| PSCO | (1,0,1)(0,1,1,24) | (2,0,1)(0,1,1,24) | 9.84 | 9.67 | +0.17 |
| WALC | (2,0,1)(0,1,1,24) | (1,0,0)(0,1,1,24) | 13.80 | **9.20** | **+4.59** |

**Δ > 0 means the "fix" is better.** 4 better, 6 worse; median **−0.44 pts**,
mean **−1.95 pts**. The order changed in **10 of 10** BAs, so this is not a
no-op that happens to look bad — the search genuinely selects differently.

Search cost roughly triples: median **19.9 s → 55.4 s** (2.8×).

## Why the "correct" fix loses

The regressions concentrate on the largest BAs, and they share a mechanism:
**the exog-aware search prunes seasonal structure.**

- CAISO (1,1,1,24) → (0,1,0,24) — seasonal AR *and* MA both dropped
- PJM (1,1,0,24) → (0,1,0,24) — seasonal AR dropped
- MISO (0,1,1,24) → (1,1,0,24) — seasonal MA traded for seasonal AR

Given five weather columns, AIC credits the regressors for variance the
seasonal terms were carrying and prunes those terms as redundant. In-sample
that is defensible. Over a 168-hour forecast it is not: the seasonal terms
carry the daily cycle robustly, whereas pointwise weather regression has to
be right at every step to substitute for them.

**The study was generous to the losing arm.** It fed arm B *perfect* future
weather. Production feeds ARIMA forecast weather, whose error grows with
horizon — so the exog-aware orders would fare no better in production than
they do here, and plausibly worse.

## What shipped

The search stays univariate, and now says so. The dead `exogenous=` kwarg is
gone — the code no longer states an intent it does not execute — and no `X` is
passed either, with the reasoning and these numbers at the call site.

`tests/unit/test_arima_auto_order_kwargs.py` pins three things:

1. no kwarg in the call is absent from the installed pmdarima's signature
   (the *class* of bug — the next rename cannot hide the same way);
2. `X` is not passed, so re-applying the "obvious fix" fails with a pointer
   to this document;
3. every `SARIMAX` construction still passes `exog` — univariate *order
   selection* is the decision, a univariate *model* would not be.

## An operational note found on the way

`_auto_select_order` is reached only when a region has **no cached order**.
`jobs/training_job.py::_read_cached_arima_order` reads `order`/`seasonal_order`
from the previous model's persisted metadata and `train_arima` skips the
search when both are present — and the newly-trained model re-persists them.
There is no invalidation path, so **a once-selected order propagates forever**.

Two consequences worth holding:

- Every currently persisted order was selected univariately, so this change
  is a no-op in steady state — it governs new BAs and any cold model store.
- The 2.8× search cost is likewise a cold-cache cost, not a daily one.

If a future change *should* re-select orders fleet-wide, it needs an explicit
invalidation; nothing in the pipeline provides one today.

## Limits

1. **Ten BAs, not 51.** Chosen as the four major ISOs plus the six where ARIMA
   is currently the 24h champion — where a real gain would have to show. A
   fleet run is `--all`.
2. **One holdout window** (168 h, ending 7 days back). No seasonal replication.
3. **Known future weather**, which favours the arm that lost.
4. **sMAPE**, matching the other A/B studies in this repo; the serving
   scorecards use mean MAPE (`docs/BACKTEST_RESULTS.md` §8 on cross-artifact
   metric mixing applies).
5. ARIMA is one ensemble member, weighted `(1/MAPE)³` under ADR-004. A
   per-model sMAPE delta is not an ensemble delta.
