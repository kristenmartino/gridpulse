# Holdout stability study — P2-17

> **Question.** A single unsmoothed 168-hour holdout drives two things: the
> ADR-004 ensemble weights and the forecast-quality visibility gate. P2-17
> (#273) asserts it flaps run-to-run and proposes a "smoothed/rolling holdout"
> so both consume a stabilised metric. Does it flap, does that matter, and is
> smoothing the right answer?
>
> **Verdict.** It flaps, it matters for the gate, and **smoothing is the wrong
> answer for the gate**. Hysteresis on the decision ships. The weights half is
> **not decided here** — see "What is deliberately not concluded".

## Method

No retraining. GCS keeps every persisted model version's `meta.json`, each
carrying that run's holdout MAPE, so the estimator's own history is already
recorded. Pulled the last 45 versions for 3 models × 12 BAs — weighted toward
the tail, because the gate only ever decides there — and aligned by training
date.

- **1,620** holdout-MAPE observations
- **540** training runs where all three models are present
- 12 BAs: SPA, IID, AZPS, SEC, HST, PSCO, ERCOT, PJM, MISO, FPL, CAISO, BPAT

The centred-median level used below is a 5-point non-causal filter. It is not
truth; it is the best available estimate of the underlying level, and it is
used only to compare causal estimators against each other.

## 1. The flap is real

| | run-to-run change in holdout MAPE |
|---|---|
| fleet median | **12.0%** |
| fleet p90 | **43.3%** |

Worst per-BA medians: PJM/arima 28.2%, ERCOT/arima 19.0%, HST/arima 19.4%,
SEC/xgboost 18.8%, AZPS/xgboost 18.3%. Single-run maxima reach 282 points
(SPA/xgboost) — the tail BAs swing enormously.

## 2. What it does to the two consumers

**Ensemble weights** move **0.220 L1 per day** at the fleet median (p90 0.547,
max 1.499 — on a 3-model simplex, L1 = 0.22 means roughly 11% of the blend
mass relocating between models *every day*).

**The visibility gate flips on 2.7% of run-transitions** — 14 flips in 528 —
and every one is in **SPA (8), AZPS (4), IID (2)**. Those are exactly the BAs
whose champion MAPE oscillates across the 22-point bar; SPA's ranges 17.5–225.8.
A BA appearing and disappearing from the UI is the user-visible consequence,
and it undermines the "51 BAs" claim the gate exists to keep honest.

## 3. Smoothing works, and costs more than it is worth — for the gate

EWMA over the holdout series, all 540 runs:

| α | weight churn | vs raw | gate flips | lag err |
|---|---|---|---|---|
| raw | 0.220 | — | 14 (2.7%) | **0.217** |
| 0.5 | 0.123 | −44% | 8 (1.5%) | 0.424 |
| 0.4 | 0.101 | −54% | 6 (1.1%) | 0.499 |
| 0.3 | 0.075 | −66% | 4 (0.8%) | 0.645 |
| 0.2 | 0.054 | −75% | 2 (0.4%) | 0.974 |

*lag err = median |estimate − centred-median level|.*

The stability gains are large and real. **But the raw value tracks the
underlying level better than any smoothed version**, monotonically — which is
the expected variance-for-bias trade of a causal filter, and it means the
day-to-day movement is not purely noise around a stable level.

For the gate there is a second, decisive objection that has nothing to do with
the trade-off: a smoothed gate reads one number while the Models tab shows
another. **Two numbers under one name is the exact defect class #273 exists to
remove.** Shipping it here would have closed the cluster by committing its
signature error.

## 4. What shipped: hysteresis on the decision

The gate's problem is not that its input is noisy — it is that a *threshold*
turns small input noise into a binary flip. Hysteresis targets that directly:

| band (pts) | flips | rate | runs hidden | effect |
|---|---|---|---|---|
| 0.0 (today) | 14 | 2.7% | 35/540 | — |
| 1.0 | 12 | 2.3% | 40/540 | −14% |
| **3.0 (shipped)** | **8** | **1.5%** | **45/540** | **−43%** |
| 5.0 | 6 | 1.1% | 52/540 | −57% |

A visible BA hides when its champion crosses the 22-point bar; a hidden one
reappears only once it is back under `22 − 3`. Costs: 10 extra hidden run-days
out of 540 — a recovered BA waits slightly longer to return.

Why this over EWMA:

- **No published figure changes.** The metric stays raw and honest everywhere;
  only the transition is sticky. There is still exactly one number.
- **No lag in any estimate.** The bias EWMA introduces is not paid at all.
- **The cost is bounded and explainable** — "a recovered BA reappears once it
  is clearly back under the bar" — rather than distributed invisibly across
  every published MAPE.

`currently_visible=None` reproduces the bare threshold exactly, so a BA the
system has never judged is unaffected, and an unreadable `gate_status` map
degrades to today's behaviour rather than to stickiness.

## What is deliberately not concluded

**The ensemble-weights half of P2-17 is not decided by this study.** Weight
churn is real (0.220 L1/day) and smoothing demonstrably reduces it, but the
only question that matters for weights is *do smoothed weights forecast
better* — and nothing here answers that. Measuring it means generating
forecasts under both weightings and scoring them across rolling origins, which
needs real training runs per arm, and per CLAUDE.md the verdict has to route
through `models/rolling_eval.py` with its satisficing constraints.

Two specific reasons not to guess:

1. The lag column says the raw estimator tracks the level *better*. If
   day-to-day movement partly reflects genuine changes in model quality, then
   weights that respond to it are doing their job and smoothing degrades them.
2. `verdict()` is allowed to return inconclusive, and on a metric this noisy
   that is a plausible outcome. Shipping a weights change on a stability
   argument alone would be exactly the in-sample reasoning ledger-23 was
   fixed for.

Tracked separately as a pre-registered A/B.

## Reproduce

```bash
# 1,620 observations from persisted meta.json; no retraining
gcloud storage ls "gs://nextera-portfolio-energy-cache/cache/models/{BA}/{model}/"
gcloud storage cat ".../{version}.meta.json"   # extra.holdout_metrics.mape
```
