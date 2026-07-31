# #230 fleet test: the pre-registered rule is NOT confirmed

**51 BAs, 6 rolling windows each, 168h horizon, zero skips or errors.
306 paired windows. The conditional rule fails its own pre-registered test.**

Run 2026-07-31 against
[`DIRECT_MULTIHORIZON_PREREGISTRATION.md`](DIRECT_MULTIHORIZON_PREREGISTRATION.md),
committed before the run. Raw:
[`DIRECT_MULTIHORIZON_FLEET.json`](DIRECT_MULTIHORIZON_FLEET.json).

---

## The pre-registered verdict

| criterion | required | result | |
|---|---|---|:--|
| 1. hard pool positive **and decisive** | yes | mean **+1.589** but wins only **54%** of windows | **FAIL** |
| 2. easy pool not positive-decisive | yes | mean −0.029, within noise | pass |
| 3. separation ≥ 0.5 pts | yes | **+1.618 pts** | pass |

**CONFIRMED: false.**

Criterion 1 fails on **reliability, not size**. The hard pool's mean is +1.589
at 3.6× stderr — magnitude is comfortable. But direct wins only 54% of the 138
hard-BA windows, and the pool's **median is +0.198 against a mean of +1.589**.
That gap is the outlier-domination signature: the average is carried by a
minority of windows with very large gains, while the typical hard-BA window is
close to a coin flip.

Per the pre-registration's own terms, **#230 stays a rejected lever rather
than becoming a per-BA option.**

This is the pre-registration doing its job. Without it, "hard-BA pool: +1.589
pts at 3.6× stderr, separation 1.6 pts" reads like a win, and the
sign-consistency requirement — fixed in advance — is the only thing standing
between that number and a rewrite.

## The fleet picture

| | |
|---|---|
| BAs | 51 (23 hard, 28 easy) |
| better / worse | **25 / 26** |
| mean Δ | +0.700 pts |
| **median Δ** | **−0.002 pts** |
| decisive for direct | 11 |
| decisive for recursive | 5 |
| inconclusive | 35 |
| ship (decisive + satisficing) | 9 |

**The median effect across 51 BAs is essentially exactly zero.** The +0.700
mean is concentrated: **NWMT (+11.0) and NYISO (+8.7) alone contribute 55% of
it**, from 2 of 51 BAs.

### Decisive per-BA outcomes

| direction | BAs |
|---|---|
| **direct** (11) | NWMT +11.04, NYISO +8.69, SPA +5.81, AZPS +5.60, SEC +1.90, PSEI +0.90, AVA +0.85, SCL +0.78, SC +0.48, DUK +0.45, BPAT +0.44 |
| **recursive** (5) | SCEG −1.28, LDWP −1.27, SPP −1.19, TAL −1.04, FPC −0.58 |

16 of 51 decisive (31%) is well above the ~5% chance would produce, so **real
effects exist**. Which specific BAs carry them is a different question, and
this run does not answer it — the decisive-direct group has a mean |Δ| of
**3.36 pts**, i.e. the wins that survive the filter are the large ones, which
is what a filter tuned for large effects will always select.

## Why per-BA adoption is not the fallback

The tempting move is to skip the failed rule and simply adopt direct on the 9
BAs that ship. That is invalid here: those 9 were **selected because they
won**. Using the same data to choose them and to justify them is the
cherry-pick the whole harness exists to prevent — the same error as reading
ISONE's +0.328 out of the cooling study.

Making it legitimate needs out-of-sample validation: choose the BAs on one
period, verify on a later one. That is a new pre-registration, not a
reinterpretation of this one.

## Power, and what the per-BA column is worth

**Only 17 of 51 per-BA results are detectable at 6 windows** — the other 34
have observed effects smaller than their own minimum detectable effect. Median
per-BA MDE is **0.90 pts**, which is larger than most of the effects in play.

This is why the pre-registration moved the unit of analysis to the pooled
test: at 306 paired windows the MDE is ≈0.21 pts. The per-BA column is
reported with `mde_pts` and `detectable` attached, and should be read as
estimates, not verdicts.

## Bias

| | mean bias |
|---|---:|
| recursive | −0.775% |
| direct | −1.211% |

Direct still under-forecasts more — the operationally expensive direction —
though the gap is narrower at fleet scale than the 10-BA run suggested
(−1.848% there). Not disqualifying on its own, and moot given the verdict.

## What this closes and what it leaves

**Closed:** direct multi-horizon as a general strategy change, and as a
difficulty-conditional one. Two hypotheses, both tested, both rejected.

**Left open:** whether *specific* BAs benefit reliably. NWMT +11.04 and NYISO
+8.69 are large enough that noise alone is an uncomfortable explanation, and
both are BAs where recursion performs badly. A properly designed follow-up
would select on one period and validate on another — but note this is the
fourth consecutive experiment whose honest answer is "conditional, per-BA, and
not demonstrated," which is itself information about where the remaining
accuracy is not.

## Limits

1. **Summer window** (June–July 2026), 6 windows per BA.
2. **XGBoost only** — says nothing about Prophet, SARIMAX, or the served
   ensemble.
3. **Perfect future weather** for both arms; plausibly favours direct, since
   recursion at least re-anchors on its own trajectory.
4. Rolling windows share training data, so they are not independent draws and
   the stderr is optimistic. The verdict rule is a decision rule, not a
   p-value.
5. Origins strided at 12h; denser origins might raise the direct arm's
   ceiling.
6. Production re-anchors hourly, so 168h measures cold-forecast capability
   rather than lived accuracy — as #230 itself notes.
