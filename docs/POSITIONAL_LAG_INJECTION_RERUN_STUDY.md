# #559 re-run — the policy fix worked, and the hypothesis is still not confirmed

**Run 2026-08-20**, to
[`POSITIONAL_LAG_INJECTION_RERUN_PREREGISTRATION.md`](POSITIONAL_LAG_INJECTION_RERUN_PREREGISTRATION.md).
Same seed, same holes, same strata as the first run; only the absent-hour policy
moved.

## Result

**Not confirmed.** Criterion 1 required stratum A decisive and positive. It is
positive and it is not decisive.

| stratum | n | mean Δ WAPE | median | MDE | sign consistency | verdict |
|---|---:|---:|---:|---:|---:|---|
| **A** — naturally gapped | 249 | **+0.268** | +0.132 | 0.172 | **0.594** | not decisive |
| **B** — never gapped | 432 | **+0.614** | +0.309 | 0.107 | **0.729** | not decisive |

Both strata now: positive, mean and median agreeing in sign, magnitude clearing
MDE — A by 1.6×, B by **5.7×**. Both fail on **one** rule, sign consistency,
and B misses it by **2.1 percentage points**.

`verdict()`'s reason, both times: *"real on average but not reliable enough to
ship."* That is exactly the case the 75% rule exists to catch, and it is not a
rule to argue down after seeing the number.

## The policy fix did what the diagnosis predicted

The runs are paired by construction, and the **control arm is byte-identical
across them** (`max|diff| = 0.0000000000`), which is the check that the policy
touched only the treatment arm:

| stratum | mean Δ before | mean Δ after | swing |
|---|---:|---:|---:|
| A | −0.265 | **+0.268** | **+0.533** |
| B | +0.187 | **+0.614** | **+0.427** |

Stratum A's sign flipped. The first run's conclusion — that the zero-fill, not
temporal indexing, was what made A run backwards — is confirmed.

Per BA, the worst first-run regressions are gone: IID **−1.47 → −0.01**,
NEVP **−0.51 → −0.005**, LGEE **−0.25 → +0.18**. PSCO remains the one BA
consistently worse (−1.12 → −0.46), and it is the BA whose real gaps are
clock-aligned rather than random — a possible interaction with the uniform
placement this study pre-registered as a confound.

**Criterion 4 met**: re-measured across **32,832 scored forecast steps** in both
strata, **0** with a NaN lag and **0** with a non-positive lag, against
13.08–22.57% before. **Criterion 2 met**: null control exact on six BAs.
**Criterion 3**: B passes satisficing; A fails on treatment bias −2.27% against
±2.0% — but its **control** arm is already at −2.03%, so the harness cannot
certify either arm against that bound, which is a limit of the replay regime and
not a property of the treatment.

## An honest note about the pre-registration

§6 pre-committed a reading for "both inconclusive": *"the effect is below 0.18
pts and the flag does not matter enough to ship either way."*

**That reading does not fit what happened, and stretching it to fit would be
dishonest.** It assumed inconclusive would mean *small*. The effect is not small
— +0.268 and +0.614, well above the MDEs — it fails on **consistency**, not
magnitude. Pre-registering a reading for an outcome and then meeting that outcome
by a different route is a real gap, and the correct response is to say the
pre-committed interpretation was wrong rather than to apply it anyway.

What the evidence supports, stated plainly: **the temporal seed with a decided
absent-hour policy is better on average by a margin that clears its own
detection threshold several times over, and it is worse on roughly a quarter of
windows.** Both halves are true, and the second is why it does not ship.

## What follows

1. **The flag stays off.** Criterion 1 is unmet, and 73% is not 75%.
2. **The stopping rule is spent.** One run, and this was it. There is no third
   attempt on this question under these registrations.
3. **The live question is now different, and it is a good one**: *where* does the
   losing quarter of windows sit? If those losses concentrate on an identifiable
   feature — gap length, gap hour, BA character — a conditional rule could ship
   what works and skip what does not. PSCO's clock-aligned gaps are the obvious
   first place to look. That is a **new** question and needs its own
   pre-registration; the artifacts here record `gap_len`, `gap_hour_utc` and
   `gap_lead_h` per window precisely so it can be asked without re-running.
4. **The seed shadow is more useful than it was, but not yet clean.** It forces
   this same treatment arm, and the zero-fill defect is genuinely gone. The
   conclusion that its output is *therefore* evidence about temporal indexing
   does **not** follow, because the arm carries a second, unrelated defect:
   [#624](https://github.com/kristenmartino/gridpulse/issues/624) —
   `HourIndexedHistory.build` sizes its array from the last **present** seed
   hour, so a trailing gap makes `set()` silently discard the recursion's own
   later predictions (measured: an array ending 11 hours into a 48-hour
   horizon). Production seeds that history from the post-`dropna` `featured`
   frame, so the gap that motivates the temporal path is also what under-sizes
   its storage. Read shadow divergence with #624 named until it is fixed.

   Found by the replication run's criterion-4 control channel (below), not by
   this study's own hypothesis, and not fixed here — the stopping rule forbids
   it.

## Independently replicated

This study was run twice, concurrently and without coordination, by two
sessions working from the same pre-registration and separate implementations.
The second run's per-window artifacts are committed here as
`POSITIONAL_LAG_INJECTION_RERUN_REPLICATION_{A,B}.json`.

| | this run | replication |
|---|---:|---:|
| A mean / median | +0.268 / +0.132 | +0.275 / +0.155 |
| A MDE / consistency | 0.172 / 0.594 | 0.169 / 0.610 |
| B mean / median | +0.614 / +0.309 | +0.620 / +0.298 |
| B MDE / consistency | 0.107 / 0.729 | 0.106 / 0.743 |
| criterion 4 (NaN-lag) | 0 / 32,832 steps | 0 / 32,688 steps |
| control-arm bias | −2.03% | −2.015% |
| null control | exact | `max|diff| = 0.0000000000` |

Same verdict, same §6 reading, and both runs independently observed that §6's
*rationale* for "both inconclusive" ("the effect is below 0.18 pts") did not
hold while its bucket did.

**The step counts differ — 32,832 against 32,688 — and that is the point.**
Identical figures would suggest shared code; near-identical ones from separately
written instruments are the stronger evidence. This investigation has twice had
a harness agree with itself for the wrong reason
(`docs/FORECAST_ORIGIN_REGRESSION.md` §2, §11), so a second implementation
landing in the same place is worth more here than a tighter single number.

The replication's figures are **not** restatements of this run's and should not
be quoted as such. Where a per-BA number is cited anywhere, cite one run.

## Limits

All confounds from the pre-registration carry over: uniform gap placement against
possibly non-uniform real timing (and PSCO is the case that makes this concrete);
injection destroying real information equally in both arms; archived vintages and
mirrored weather; XGBoost only; one gap per window. Stratum A remains
structurally capped near 280 windows and underpowered relative to B — though at
+0.268 against an MDE of 0.172, power was not what stopped it this time.
