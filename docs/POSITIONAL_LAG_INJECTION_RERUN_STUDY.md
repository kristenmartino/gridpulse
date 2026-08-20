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
4. **The seed shadow is now more useful than it was.** It forces this same
   treatment arm, which no longer carries the zero-fill defect, so what it
   records in production is finally evidence about temporal indexing.

## Limits

All confounds from the pre-registration carry over: uniform gap placement against
possibly non-uniform real timing (and PSCO is the case that makes this concrete);
injection destroying real information equally in both arms; archived vintages and
mirrored weather; XGBoost only; one gap per window. Stratum A remains
structurally capped near 280 windows and underpowered relative to B — though at
+0.268 against an MDE of 0.172, power was not what stopped it this time.
