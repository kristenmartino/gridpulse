# #559 — the injection study: hypothesis NOT confirmed, and why

**Run 2026-08-20**, to the pre-registration in
[`POSITIONAL_LAG_INJECTION_PREREGISTRATION.md`](POSITIONAL_LAG_INJECTION_PREREGISTRATION.md)
(committed, with its amendment, before the runner existed — the ordering is
checkable in git).

## Result

**The pre-registered hypothesis is not confirmed.** Criterion 1 required stratum
A to be decisive and positive. It is neither.

| stratum | n | mean Δ WAPE | median | MDE | sign consistency | verdict |
|---|---:|---:|---:|---:|---:|---|
| **A** — naturally gapped | 249 | **−0.265** | +0.067 | 0.327 | 0.478 | not decisive |
| **B** — never gapped | 432 | **+0.187** | +0.187 | 0.180 | 0.641 | not decisive |

Positive Δ = treatment (temporal) better.

- **A** fails on the outlier signature: mean and median **disagree in sign**, so
  `verdict()` reports tail risk rather than a win. Satisficing also fails —
  treatment bias **−3.14%** against a ±2.0% bound, and the control's own −2.03%
  is already at the edge.
- **B** clears its magnitude test (|+0.187| > MDE 0.180) and satisfices, but
  wins only **64%** of windows against the 75% required: real on average, not
  reliable enough to ship.

**The null control is exact.** Six BAs across both strata, gap-free origins
selected by the contiguity predicate rather than assumed:
`max|diff| = 0.0000000000`. The harness measures the seed convention and
nothing else.

One number worth noting: stratum B's **+0.187** lands almost exactly on the
observational study's **+0.181** at the same horizon, from disjoint data and a
different gap population. The mechanism replicates; it is the *decision* that
does not follow.

## Why the strata disagree — a defect in the treatment arm

Stratum A's mean runs *against* the hypothesis. The cause is not the temporal
indexing, and this is the finding that matters most.

`compute_temporal_autoregressive_snapshot` returns **NaN** when the hour a lag
asks for is absent. That is the honest answer, and it is the whole point of
resolving by hour. But the shared row build then does `row.fillna(0)`, so the
model is handed **`demand_lag_24h = 0 MW`** — the exact #129 poison the seed
filter exists to keep out. The positional arm never does this: its history always
holds ≥168 entries after warm-up, so it always supplies a *plausible* value, just
from the wrong hour.

Measured over the injected windows, share of forecast steps where at least one
AR lag is NaN and therefore zero-filled:

| stratum | BA | NaN-lag steps | rate |
|---|---|---:|---:|
| B | MISO, PJM, CAISO, FPL | 226 / 1728 | **13.08%** |
| A | LGEE | 226 / 1728 | 13.08% |
| A | NWMT | 238 / 1728 | 13.77% |
| A | TIDC | 267 / 1728 | 15.45% |
| A | PSCO | 292 / 1728 | 16.90% |
| A | **IID** | 390 / 1728 | **22.57%** |

Stratum A carries real gaps *on top of* the injected one, so it zero-fills more
often — and the two worst per-BA regressions are IID (−1.47 WAPE, 22.57%) and
PSCO (−1.12, 16.90%). The correlation is not clean (TIDC is at 15.45% and still
*improves* by +1.07), so this is a plausible mechanism, not a proven one.

**So this run compared temporal-indexing-**plus-zero-fill** against positional
indexing.** That is not the comparison the hypothesis names. It is exactly the
limit recorded, and left unmeasured, at §4.4 of the observational study:

> Absent-hour policy is a modelling choice the treatment arm makes implicitly.
> [...] it is unmeasured deeper into the horizon and a production fix must decide
> this explicitly rather than inherit it.

It is now measured. The policy is wrong, and it was inherited rather than chosen.

## What follows

1. **The flag stays off.** Nothing here supports turning it on, and stratum A
   mildly argues against it.
2. **The absent-hour policy needs deciding, not inheriting** — carry the last
   known hour forward, interpolate across the hole, or fall back to the
   positional value for that lag alone. Each is a modelling decision with a
   different failure mode, and the choice must be pinned by a test.
3. **A re-run is a NEW pre-registration.** The stopping rule here was one run,
   and re-running with a changed treatment arm and reporting it against *this*
   pre-registration is precisely the re-tuning that rule forbids.
4. **The seed shadow (#597) would have recorded this defect**, since it forces
   the same treatment arm. Worth landing anyway — it is gated, capped and off —
   but its arm should carry the policy fix before anyone reads its output as
   evidence about temporal indexing.

## Limits

* **Placement is uniform, real gaps may not be** — pre-registered as a confound.
  Gap hour-of-day is recorded in the artifacts for a later descriptive cut.
* **The zero-fill confound is now the dominant one**, and it is not corrected
  here by design: correcting it mid-analysis would have turned a pre-registered
  test into a fishing expedition.
* Stratum A is capped at ~280 windows by the 7 naturally-gapped BAs and a 90-day
  mirror; 249 scored after vintage and truth requirements. It was pre-registered
  as powered for 0.25 pts, not for 0.18.
* XGBoost only; archived vintages and mirrored weather, common-mode across arms.
* **Disclosure:** two smoke runs (LGEE; MISO+PJM) preceded the recorded run and
  their output was seen. They validated the harness and exposed a null-control
  bug — the control had assumed "no injected gap" meant "no gap", which is false
  for stratum A. Fixing that was required by criterion 2. No study parameter was
  changed in response to any result.
