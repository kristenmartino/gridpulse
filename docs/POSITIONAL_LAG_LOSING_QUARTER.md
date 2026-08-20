# #559 — the losing windows are diffuse, and both hypotheses were backwards

**Run 2026-08-20**, to
[`POSITIONAL_LAG_LOSING_QUARTER_PREREGISTRATION.md`](POSITIONAL_LAG_LOSING_QUARTER_PREREGISTRATION.md),
committed before the analysis existed.

**EXPLORATORY**, as pre-registered: this re-cuts a dataset that already produced
a verdict, so nothing here can turn `temporal_ar_seed` on.

## Result: nothing flagged, 20 cells

No cell met the pre-registered bar (n ≥ 30, win rate < 50%, bootstrap 95%
interval excluding zero on the losing side). At a 5% level ~1 false flag was
expected by chance; **0 appeared**.

Per the pre-registration, that settles it: **the losing quarter is diffuse.
There is no subgroup to carve out.**

## Both pre-specified hypotheses point the wrong way

Reported in full, flagged or not, as required.

### H1 — "losses concentrate on long gaps." No; long gaps help *more*.

| stratum | gap len | n | mean Δ | win rate |
|---|---|---:|---:|---:|
| B | 1h | 349 | +0.513 | 0.736 |
| B | 2-3h | 42 | **+1.057** | 0.714 |
| B | 13-24h | 41 | **+1.021** | 0.683 |
| A | 1h | 203 | +0.275 | 0.596 |
| A | 2-3h | 22 | +0.166 | 0.682 |
| A | 13-24h | 24 | +0.304 | 0.500 |

The reasoning behind H1 — that the seasonal branch is a weaker estimate than
interpolation — is not what the data shows. On stratum B a long gap yields
**twice** the mean benefit of a 1-hour gap. Win rate does decline monotonically
(0.736 → 0.714 → 0.683), so the benefit is larger but slightly less reliable;
neither bin comes near the flag bar.

### H2 — "losses concentrate where the gap is recent." No; that is where it helps most.

| stratum | gap lead | n | mean Δ | win rate |
|---|---|---:|---:|---:|
| B | 1-24h | 52 | **+1.143** | **0.808** |
| B | 25-72h | 122 | +0.655 | 0.754 |
| B | 73-168h | 258 | +0.488 | 0.702 |
| A | 1-24h | 27 | +0.101 | **0.444** |
| A | 25-72h | 67 | +0.338 | 0.642 |
| A | 73-168h | 155 | +0.268 | 0.600 |

**This is the clearest structure in the data, and it is the exact inverse of the
hypothesis.** On stratum B, both mean and win rate fall monotonically as the gap
moves away from the origin: the fix helps most precisely where the defect bites
hardest, when a recent hole corrupts `demand_lag_1h` and `demand_lag_3h`.

Stratum A's 1-24h cell is the only sub-50% win rate anywhere (0.444) — and it has
**n = 27**, below the pre-registered floor of 30, with an interval spanning zero.
It is reported because everything is reported, not because it means anything. It
is also the one place the two strata disagree in direction, which is a reason to
trust neither reading without fresh data.

### H3 (post-hoc) — no hour-of-day effect

Win rates by gap hour UTC are flat: **B 0.683–0.783**, **A 0.542–0.649**, with no
ordering. The PSCO-derived suspicion does not generalise into an hour effect.

## PSCO, post-hoc — and it would not have rescued the verdict anyway

| | n | mean Δ | win rate |
|---|---:|---:|---:|
| PSCO | 36 | **−0.462** | **0.361** |
| stratum A excluding PSCO | 213 | +0.392 | 0.634 |

PSCO is genuinely anomalous. But this look was suggested by a result already
seen, so it is hypothesis-generating only — and the number that matters is the
second row: **removing the worst BA entirely still leaves stratum A at a 0.634
win rate, well short of the 0.75 the policy requires.** Even the most generous
post-hoc carve-out does not ship.

That is worth stating plainly because it closes the obvious escape route. The
consistency failure is not one bad BA.

## What follows

1. **The flag stays off**, and this line of inquiry is **closed**. There is no
   subgroup rule to write, and the pre-registration fixed that reading in advance
   precisely so this could not be re-litigated.
2. **No confirmatory follow-up is warranted.** §5 reserved that for a coherent
   pattern in the *losing* direction. The one coherent pattern found (H2 on
   stratum B) runs the other way and describes where the fix already works.
3. **Two hypotheses were falsified, cheaply**, on data already collected. Both
   were mechanically plausible and both were wrong, which is the argument for
   writing them down before looking rather than after.

## Limits

* Exploratory by construction; no claim here is confirmatory.
* Gap placement is uniform, so `gap_hour_utc` is uniform by construction — H3
  tests only whether the *effect* varies by hour, not whether real gaps cluster
  there. PSCO's real 10:00 UTC clustering is **not** represented at its true
  frequency, so this cannot exonerate or convict that mechanism.
* `gap_len` bins inherit the empirical imbalance (~81% of windows carry a 1h
  gap), so the long-gap intervals are wide.
* The H2 gradient is not adjusted for anything that might correlate with gap
  lead; the stopping rule was one pass with no added covariates, and that has
  been honoured.
* Stratum A carries real gaps on top of injected ones, so its `gap_len` label
  describes only the injected hole.
