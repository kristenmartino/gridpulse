# Pre-registration — #559 re-run, with the absent-hour policy decided

**Committed before the re-run exists. Nothing below was chosen after seeing a
re-run result.**

This is a **new** pre-registration, not an amendment to
[`POSITIONAL_LAG_INJECTION_PREREGISTRATION.md`](POSITIONAL_LAG_INJECTION_PREREGISTRATION.md).
That one's stopping rule was one run, and it has been spent. Re-running a
changed treatment arm against it would be exactly the re-tuning it forbids.

## 1. What changed, and why a re-run is warranted

The first run did **not** confirm its hypothesis
([`POSITIONAL_LAG_INJECTION_STUDY.md`](POSITIONAL_LAG_INJECTION_STUDY.md)), and
the diagnosis was that the treatment arm carried a defect of its own:
`compute_temporal_autoregressive_snapshot` returned NaN for an absent hour and
the shared `row.fillna(0)` turned that into `demand_lag_24h = 0 MW` — the #129
poison — on **13% of forecast steps, 22.6% on IID**.

So that run measured *temporal-indexing-plus-zero-fill*, which is not the
hypothesis. The policy is now decided rather than inherited
(`HourIndexedHistory.lag`):

1. **Hole ≤ 6 hours → linear interpolation** between the observations either
   side. The bound is `MAX_INTERPOLATION_GAP_HOURS`, reused from
   `data.preprocessing` rather than invented, and it covers the dominant case:
   25 of the 31 measured gap runs are a single hour.
2. **Longer hole → same clock hour, previous days**, stepping back in 24-hour
   multiples up to 7 (matching `demand_lag_168h`'s own reach). Interpolating
   across a 16-hour hole would smooth over a diurnal cycle; this preserves phase.

**Measured effect of the fix, before any accuracy claim:** on scored windows the
NaN-lag rate is **0.00%** across all nine probed BAs, against 13.08–22.57%
before. No lag is zero-filled any more.

**Stated plainly:** this imputation is **serve-only and out of distribution by
construction.** `engineer_features` drops any training row whose lag source was
NaN, so the model never saw an imputed lag. The goal is not to be right about
the missing hour; it is to keep the row inside the distribution the model was
fit on, which a plausible value does and zero does not. That is a hypothesis
about distribution shift, and it is what this run tests.

## 2. The hypothesis, stated in advance

> With the absent-hour policy applied, resolving autoregressive lags by
> timestamp produces lower WAPE than resolving them by position, on origins
> whose 168-hour lookback contains a gap.

Same direction as before, and the same reasoning: training features are
temporally correct, so the temporal seed moves inference toward the convention
the model was fit under.

## 3. Everything held fixed from the first run

Unchanged, deliberately, so the two runs are comparable and only the policy moved:

* injected gaps are **NaN values on present rows**, run lengths drawn from the
  frozen empirical distribution (1×25, 2×2, 3×1, 13×1, 16×1, 24×1);
* one gap run per window, placed uniformly at random in the 168-hour lookback,
  seeded per (BA, window) — **same seed, 559**, so the same holes land in the
  same places and the comparison is paired against the first run;
* 48-hour horizon, non-overlapping windows, archived vintage live at each origin;
* strata A (7 naturally gapped) and B (12 never gapped), reported separately;
* verdict through `models/rolling_eval.py`.

## 4. Power

Unchanged and already known, since the window population is identical:

| stratum | n (first run) | MDE |
|---|---:|---:|
| A — naturally gapped | 249 | 0.327 |
| B — never gapped | 432 | 0.180 |

Stratum A remains structurally capped and remains **underpowered for a 0.18-pt
effect**. That was true last time and is not fixed by this change.

## 5. What counts as confirmation

Confirmed only if all four hold:

1. **Stratum A decisive and positive** — `verdict()` returns `decisive` with
   `winner == "treatment"`.
2. **Null control exact** — gap-free origins give Δ of exactly 0. Voids the run
   otherwise.
3. **Satisficing passes**, control-arm bias checked first.
4. **The zero-fill is actually gone** — NaN-lag rate 0.00% on scored windows,
   re-measured in the run rather than assumed from §1.

## 6. Pre-committed readings of the outcomes that are not confirmation

Fixed now so none of them can be chosen afterwards:

* **A decisive positive, B decisive positive** → confirmed; the policy was the
  blocker and temporal indexing helps.
* **A inconclusive, B decisive positive** → the mechanism is real at a size B can
  see and A cannot. **Not** licence to quote B's number as A's, and **not**
  sufficient to flip the flag on stratum A's population.
* **A still negative** → the zero-fill was *not* the explanation, the earlier
  diagnosis was wrong, and temporal indexing genuinely costs accuracy on gapped
  BAs. The flag should then be considered for removal, not just left off.
* **Both inconclusive** → the effect is below 0.18 pts and the flag does not
  matter enough to ship either way. A publishable answer, and the most likely
  one on the first run's numbers.

## 7. Stopping rule

**One run.** No re-runs with adjusted policy parameters, gap placement, horizon,
BA set or window count. A third attempt requires a third pre-registration, and
if this one also fails to confirm, the honest conclusion is that the question is
not worth more compute.

## 8. Known confounds

All five from the first run carry over unchanged: uniform gap placement against
possibly non-uniform real timing; injection destroying real information equally
in both arms; archived vintages and mirrored weather; XGBoost only; one gap per
window. Two more are specific to this run:

* **The policy is itself a modelling choice**, and a different one (carry
  forward, nearest neighbour, positional fallback for that lag alone) could give
  a different answer. This run tests the policy as specified, not the space of
  policies. Comparing policies is a separate study and would need its own
  multiplicity treatment.
* **Interpolation uses the hour *after* the hole**, which at serve time is real
  history rather than a future value — but it is information the positional arm
  also has, so the arms remain comparable and neither sees the future.
