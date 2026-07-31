# Pre-registration — #230 direct multi-horizon, 51-BA run

**Committed before the run. Nothing below was chosen after seeing fleet results.**

The 10-BA study (`DIRECT_MULTIHORIZON_STUDY.md`) found a conditional signal —
direct beats recursive where recursion is already struggling — but the
difficulty threshold was chosen **after** seeing the data, which makes it
hypothesis-generating only. This run tests it properly.

## 1. The hypothesis, stated in advance

> Direct multi-horizon forecasting outperforms recursive forecasting on BAs
> where the forecasting problem is harder, and does not outperform it
> elsewhere.

## 2. The threshold, fixed now

**Difficulty = mean of the two arms' WAPE** (symmetric, so it shares no term
with the delta — the 10-BA writeup's `corr(recursive, Δ) = +0.872` was
inflated by exactly that shared term; the honest figure was +0.737).

**A BA is "hard" when difficulty ≥ 5.00 WAPE.** Taken from the 10-BA run's
median (5.05), rounded to a round number, and **frozen here**. It will not be
re-tuned to fit the fleet result.

## 3. What counts as confirmation

The conditional rule is **confirmed** only if all three hold:

1. **Pooled hard-BA effect is positive and decisive** — mean Δ > 0 on the
   pooled paired windows of hard BAs, at ≥ 2× stderr.
2. **Pooled easy-BA effect is not positive-decisive** — direct does not also
   win on easy BAs, or the rule is not conditional, it is just "direct wins".
3. **Separation** — hard-BA mean Δ exceeds easy-BA mean Δ by ≥ 0.5 pts.

Anything less is **not confirmed**, and #230 stays open as a rejected lever
rather than a per-BA option.

## 4. Power, and why the unit of analysis changes

Per-BA verdicts at 5–6 windows are underpowered, and the 10-BA run's own
numbers show how badly. Minimum detectable effect = 2 × stderr:

| BA | observed Δ | MDE at n=5 | detectable? |
|---|---:|---:|:--|
| NYISO | +4.292 | 5.747 | **no** |
| ISONE | +2.267 | 3.067 | **no** |
| CAISO | +0.274 | 2.417 | no |
| PJM | +1.124 | 0.806 | yes |
| SPP | −1.206 | 0.519 | yes |

**The two largest effects were undetectable by construction.** Detecting a
0.5 pt effect would need a median of 14 windows — and 117 for CAISO, 660 for
NYISO. Per-BA verdicts at this window count are therefore reported as
**estimates with their MDE**, not as verdicts.

The **pooled** test is the well-powered one: 51 BAs × 6 windows = 306 paired
observations, MDE ≈ **0.21 pts**. That is the unit the confirmation criteria
above are evaluated on, and it is fixed here so it cannot be swapped for
whichever unit happens to look better.

## 5. Stopping rule

One run. 51 BAs, 6 windows, 168h horizon. No re-runs with adjusted
parameters; if it fails, it fails, and any follow-up is a new pre-registration.

## 6. Known confounds, acknowledged in advance

* Perfect future weather for both arms — plausibly favours direct.
* XGBoost only; says nothing about Prophet, SARIMAX or the served ensemble.
* Rolling windows overlap in training data, so they are not independent draws
  and the stderr is optimistic. It is a decision rule, not a p-value.
* Summer window; the seasonal caveat from `ERROR_ANALYSIS.md` applies.
