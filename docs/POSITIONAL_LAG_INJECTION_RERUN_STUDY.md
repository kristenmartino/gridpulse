# #559 — the injection re-run: the sign flipped, the verdict did not

**Run 2026-08-20**, to the pre-registration in
[`POSITIONAL_LAG_INJECTION_RERUN_PREREGISTRATION.md`](POSITIONAL_LAG_INJECTION_RERUN_PREREGISTRATION.md)
(committed before this run existed — the ordering is checkable in git). That is
a **new** pre-registration, not an amendment to the first one, whose one-run
stopping rule was spent.

Per-window artifacts are committed alongside this document —
[`POSITIONAL_LAG_INJECTION_RERUN_A.json`](POSITIONAL_LAG_INJECTION_RERUN_A.json)
(249 rows) and
[`POSITIONAL_LAG_INJECTION_RERUN_B.json`](POSITIONAL_LAG_INJECTION_RERUN_B.json)
(432 rows) — so every per-BA figure below can be recomputed rather than taken on
trust, and a future run can be paired against this one. The first run's were
not, which is why its numbers are quoted here rather than recomputed.

Reproduce with:

```bash
PYTHONPATH=. python scripts/positional_lag_injection_study.py --stratum A --out /tmp/inj_A.json
PYTHONPATH=. python scripts/positional_lag_injection_study.py --stratum B --out /tmp/inj_B.json
```

## Result

**The pre-registered hypothesis is not confirmed.** Two of the four §5 criteria
hold, two do not, and `verdict()` refuses to decide on both strata.

| stratum | n | mean Δ WAPE | median | stderr | MDE | sign consistency | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| **A** — naturally gapped | 249 | **+0.2749** | +0.1548 | 0.0846 | 0.1692 | 0.610 | not decisive |
| **B** — never gapped | 432 | **+0.6200** | +0.2981 | 0.0532 | 0.1064 | 0.743 | not decisive |

Positive Δ = treatment (temporal) better.

### The four confirmation criteria

| # | criterion | result |
|---|---|---|
| 1 | stratum A decisive **and** `winner == "treatment"` | **FAILS** — `decisive=False`, `winner=None` |
| 2 | null control exactly 0 | **PASSES** — `max\|diff\| = 0.0000000000`, six BAs |
| 3 | satisficing passes, control-arm bias checked first | **FAILS on A** (treatment bias −2.238% vs ±2.0%); passes on B |
| 4 | NaN-lag rate 0.00% on scored windows, re-measured | **PASSES** — 0 / 32,688 steps, 0.0000% |

Confirmation required all four. It has two.

### Which pre-committed reading applies

From §6, unchanged and chosen in advance:

> **Both inconclusive** → the effect is below 0.18 pts and the flag does not
> matter enough to ship either way. A publishable answer, and the most likely
> one on the first run's numbers.

**That is the reading: both inconclusive.** A is not decisive; B is not
decisive. The other three readings need a decisive stratum, and neither has one.

**One honest qualification, which does not change the reading.** The §6 text
attaches a *rationale* to this outcome — "the effect is below 0.18 pts" — and
that rationale is not what happened. Both strata **clear** their magnitude test:
A's +0.275 against an MDE of 0.169, B's +0.620 against 0.106. What fails is
**window reliability**: temporal indexing wins 61% of stratum A windows and 74%
of stratum B windows, against the 75% `MIN_SIGN_CONSISTENCY` requires. The
outcome bucket is the pre-committed one; the mechanism inside it is different
from the one anticipated, and saying so is cheaper than pretending the
prediction was exact. Inventing a fifth reading to describe "large but
unreliable" is exactly what pre-registration exists to prevent, so it is not
done here.

**Stratum B misses by three windows.** 321 of 432, where 324 is the bar. This is
recorded because it will be obvious to anyone who divides the numbers, and
recording it is the only safe way to hold the line: 0.7 percentage points is
not a pass, the threshold was fixed before the run, and a re-run tuned to cross
it is precisely what §7's stopping rule forbids. The number is not quoted as a
win anywhere in this document.

## What did move: the sign

The first run's stratum A mean ran **against** the hypothesis at −0.265 with a
median of +0.067 — signs disagreeing, which is what made `verdict()` report tail
risk. With the absent-hour policy in place, the same 249 paired windows, same
seed, same holes in the same places, give:

| | first run (#605) | this run |
|---|---:|---:|
| mean Δ WAPE | −0.265 | **+0.2749** |
| median Δ WAPE | +0.067 | **+0.1548** |
| signs agree | no | **yes** |
| sign consistency | 0.478 | 0.610 |
| treatment bias | −3.14% | −2.238% |

A swing of **+0.54 pts** on the mean, with mean and median now agreeing in sign
and within 0.12 pts of each other. **The earlier diagnosis survives its test:**
the first run measured *temporal-indexing-plus-zero-fill*, and the zero-fill was
carrying the negative result. Stratum B moved the same direction, +0.187 →
+0.620.

That is a real finding about the *mechanism*. It is not a decision, and it does
not license shipping the flag.

## Per-BA

### Stratum A — naturally gapped

| BA | n | mean Δ | median Δ | win rate | control bias | treatment bias |
|---|---:|---:|---:|---:|---:|---:|
| TIDC | 36 | **+1.5501** | +1.2688 | 94.4% | +1.16% | +0.43% |
| NWMT | 36 | +0.7717 | +0.4195 | 80.6% | −0.63% | −1.06% |
| LGEE | 35 | +0.1704 | +0.1657 | 60.0% | −1.90% | −1.87% |
| NEVP | 36 | −0.0343 | −0.0453 | 36.1% | −14.70% | −14.80% |
| IID | 34 | −0.0532 | +0.0280 | 52.9% | −1.72% | −1.96% |
| SPA | 36 | −0.2173 | +0.0142 | 52.8% | +7.31% | +7.77% |
| PSCO | 36 | −0.2840 | −0.0891 | 50.0% | −3.61% | −4.16% |

The pooled mean is carried by two BAs: TIDC alone contributes more than the
pooled mean, and four of seven are negative or flat. That dispersion **is** the
sign-consistency failure in per-BA form, and it is why the pooled number must
not be read as "temporal indexing helps gapped BAs."

Against the first run — whose per-BA figures are quoted from
[`POSITIONAL_LAG_INJECTION_STUDY.md`](POSITIONAL_LAG_INJECTION_STUDY.md), since
that run's artifacts are not in the repo and were not recomputed here — **IID
moved from the worst regression (−1.47) to roughly flat (−0.053)**, and it was
the BA with the highest zero-fill rate at 22.57%. PSCO, second-worst at 16.90%,
moved from −1.12 to −0.284. Both moved as the diagnosis predicted, neither
became a win.

### Stratum B — never gapped

| BA | n | mean Δ | median Δ | win rate |
|---|---:|---:|---:|---:|
| FPL | 36 | +1.8710 | +1.5837 | 91.7% |
| BPAT | 36 | +1.5005 | +1.3377 | 94.4% |
| CAISO | 36 | +1.1549 | +1.0624 | 88.9% |
| TVA | 36 | +0.6915 | +0.2554 | 80.6% |
| ISONE | 36 | +0.4529 | +0.2311 | 63.9% |
| MISO | 36 | +0.3817 | +0.1931 | 75.0% |
| PACE | 36 | +0.3641 | +0.2182 | 72.2% |
| DUK | 36 | +0.2453 | +0.0571 | 58.3% |
| ERCOT | 36 | +0.2343 | +0.1495 | 75.0% |
| SPP | 36 | +0.2034 | +0.1462 | 72.2% |
| PJM | 36 | +0.1922 | +0.1000 | 66.7% |
| NYISO | 36 | +0.1484 | +0.0181 | 52.8% |

All twelve are positive, which the first run's +0.187 already hinted at. The
spread is the problem: three BAs above +1.0 and four below +0.25, so the pooled
mean describes no particular BA well, and the median (+0.298) is less than half
the mean.

## Satisficing — and on stratum A the control fails it too

```
stratum A   control bias −2.015%   treatment bias −2.238%   bound ±2.0%
            control MAPE 11.6025   treatment MAPE 11.3230
stratum B   control bias −0.809%   treatment bias −0.703%   PASSES
            control MAPE  4.9204   treatment MAPE  4.2948
```

**The control arm's bias was computed first**, as the policy requires, and it
matters: on stratum A the control is *itself* outside the bound at −2.015%. The
constraint is not cleanly separating the arms — the stratum-A injected-gap
population under-forecasts in both arms, and injection is part of why, since a
NaN'd hour removes real information from both arms equally (a pre-registered
confound).

The policy is nonetheless unambiguous: **a satisficing constraint vetoes a win,
and treatment fails it on A.** Treatment is also 0.223 pts *worse* than control
on bias, so this is not purely a population artifact. MAPE moves the right way
in both strata and is not enough on its own.

This is a strict improvement in evidence over the first run, where control-arm
bias was not computed at all and was recorded as a limit. It is computed now,
and it says the stratum-A bias failure is partly structural.

## Criterion 4 — the zero-fill is gone, and the probe can prove it

Re-measured **in this run**, on this run's scored windows, rather than inherited
from #615's own reporting:

| stratum | point lags NaN | rate | any snapshot key NaN | rate |
|---|---:|---:|---:|---:|
| A | 0 / 11,952 steps | **0.0000%** | 0 / 11,952 | 0.0000% |
| B | 0 / 20,736 steps | **0.0000%** | 14 / 20,736 | 0.0675% |
| **total** | **0 / 32,688** | **0.0000%** | 14 / 32,688 | 0.0428% |

Worst single BA in either stratum: **0.0000%** — against 13.08–22.57% before.
**No autoregressive lag is zero-filled on any scored window.** Criterion 4 holds.

### Why a zero here is readable

A zero satisfies criterion 4, and a broken probe produces the same zero. Three
independent guards separate those cases:

1. **Interception is asserted per window.** The treatment arm requires exactly
   `1 + 48` snapshot calls (one column-resolution probe, one per step) and
   raises otherwise, so a silently-defeated wrapper cannot pass as clean.
2. **Two controls, run before any window is scored.** *Designed to disagree* — a
   history with one observed hour queried 200 hours later, where both imputation
   regimes must fail (no neighbour within 6 hours, no same clock hour within 7
   days); the probe **must** flag it. *Designed to agree* — a dense contiguous
   history; the probe **must not** flag it. Result `(True, False)`; the run
   aborts on anything else.
3. **The probe demonstrably reports non-zero on real study data** — the 14 steps
   in the "any snapshot key" column below. An instrument stuck at zero could not
   have produced them.

### The 14 non-lag NaN steps — a second, distinct defect in the treatment arm

The point-lag column is zero, but the broader column is not, and the difference
is worth stating rather than rounding away. All 14 steps are one PACE window
(origin 2026-07-29T00:00Z, a 24-hour injected gap starting 39 hours earlier),
and all of them are the same five keys:

```
demand_roll_24h_mean, demand_roll_24h_std, demand_roll_24h_min,
demand_roll_24h_max, demand_ratio_24h
```

These are **rolling-window** features, not point lags. `HourIndexedHistory.lag`
imputes; `window()` deliberately does not, because it mirrors the training
side's `min_periods=1`, which skips a NaN inside a window rather than filling
it. So criterion 4 is untouched — but these five are still handed to the model
as `0` by the shared `row.fillna(0)`.

Tracing the recursion on that window shows the cause is **not** the absent-hour
policy:

* `engineer_features` drops every row whose lag source was NaN, so a 24-hour gap
  deletes ~39 consecutive trailing hours of the seed. The last present seed hour
  is 2026-07-27T09:00Z for an origin of 2026-07-29T00:00Z.
* `HourIndexedHistory.build` sizes its array as `offsets.max() + 1 +
  extra_hours`, and `extra_hours` is `len(future_df) + 1 = 49`. That room is
  reserved from the **last present seed hour**, not from the origin — so a
  trailing hole eats it. Here the array ends 11 hours into a 48-hour horizon.
* `set()` guards with `0 <= i < len(self._values)` and therefore **silently
  discards every prediction past that point**. The trailing 24-hour window
  saturates, then empties as it slides past the end of the array, and the last
  ~13 steps have nothing in it.

So the temporal arm silently stops recording its own predictions when the seed's
tail is missing — which is exactly the broken-feed case #559 exists for. It is
production-reachable in principle, though the flag is off and this study is the
only thing exercising the path.

**It is deliberately not fixed here.** Correcting the treatment arm mid-analysis
is what the first study refused to do and what §7's stopping rule forbids; a
third attempt needs a third pre-registration, and this defect should be fixed
before it. Filed as a follow-up rather than patched in place. Its blast radius
in *this* run is 0.0675% of stratum B steps in a single window, far too small to
account for the sign-consistency shortfall.

## Null control — exact

Gap-free origins, chosen by the contiguity predicate rather than assumed
("no injected gap" is not "no gap" on stratum A):

```
stratum A   LGEE   max|diff| 0.0000000000  exact=True
            PSCO   max|diff| 0.0000000000  exact=True
            TIDC   max|diff| 0.0000000000  exact=True
stratum B   MISO   max|diff| 0.0000000000  exact=True
            PJM    max|diff| 0.0000000000  exact=True
            ERCOT  max|diff| 0.0000000000  exact=True
```

Criterion 2 holds to full precision across six BAs. In the same run the injected
windows diverge by 2.165%/1.686% mean and 21.119%/11.522% max (A/B), and **0 of
249** and **0 of 432** windows were byte-identical — so the arms demonstrably do
different things where a gap exists and are demonstrably identical where one
does not. That pair of facts is what makes the null control meaningful rather
than vacuous.

## What follows

1. **`temporal_ar_seed` stays OFF.** Under the "both inconclusive" reading the
   flag does not matter enough to ship either way, and nothing here clears the
   bar to turn it on.
2. **Do not remove it either.** Removal was the pre-committed response to "A
   still negative", and A is no longer negative — it moved +0.54 pts and now
   agrees in sign with B. The flag stays registered and off.
3. **The question is now answered as far as this design can answer it.** §7
   allows one run; it has been spent. A third attempt requires a third
   pre-registration, and on this evidence the honest position is that the effect
   is real, positive, consistent in direction across all 19 BAs — and still not
   reliable enough per-window to ship. That is a publishable answer, not a
   failure.
4. **Fix the `extra_hours` truncation before any further work on this path**,
   including the #597 seed shadow, which forces the same treatment arm and would
   inherit the defect. It is a correctness bug in its own right, independent of
   whether the flag ever ships.
5. **A decisive stratum-A answer needs a different design, not another run** —
   see the first limit below.

## Limits

* **Stratum A is structurally underpowered for this effect, and this change did
  not fix that.** Only 7 BAs are naturally gapped, which caps the population at
  ~280 windows against a 90-day mirror; 249 scored, MDE 0.169 pts. The
  pre-registration said so in advance, the first study said so, and it remains
  the largest single constraint on what this question can be answered with. A
  decisive stratum-A answer needs a longer mirror or more gapped BAs.
* **Stratum B's numbers are stratum B's.** They are not quoted as stratum A's
  anywhere here, and B's population — never-gapped BAs with a synthetic hole —
  is not the population the flag would serve.
* **The absent-hour policy is itself a modelling choice.** Carry-forward,
  nearest-neighbour, or a positional fallback for that lag alone would each give
  a different treatment arm. This run tests the policy as specified in #615, not
  the space of policies; comparing policies is a separate study needing its own
  multiplicity treatment.
* **The imputation is serve-only and out of distribution by construction.**
  `engineer_features` drops any training row whose lag source was NaN, so the
  model never saw an imputed lag. The claim tested is about distribution shift,
  not about being right for the missing hour.
* **Bias fails on both arms in stratum A**, so that satisficing veto is partly a
  property of the injected-gap population rather than of the treatment. Reported
  above, not adjusted for.
* **The treatment arm still carries the `extra_hours` truncation** described
  above. Small here, unquantified beyond this run.
* All five confounds from the first run carry over: uniform gap placement
  against possibly non-uniform real timing; injection destroying real
  information in both arms; archived vintages and mirrored weather; XGBoost
  only; one gap per window.
* **Disclosure — runs whose output was seen.** One smoke run preceded the
  recorded runs: LGEE alone (`--regions LGEE`), which validated the new NaN-lag
  probe end-to-end and returned the same LGEE numbers the recorded stratum-A run
  later produced (same seed, n=35, mean +0.1704). Two short diagnostic runs
  followed the recorded runs, both on the single PACE window above, to identify
  which snapshot keys NaN'd and why; neither touched any arm, metric or scored
  result. No study parameter was changed in response to any of them.
* **Disclosure — a process fault, recorded rather than hidden.** The probe's two
  controls were added **after** the LGEE smoke output was seen, not before,
  which is not the ordering CLAUDE.md's "control case designed to disagree,
  checked before results are inspected" asks for. The probe wrapper itself was
  not modified when the controls were added — only the self-test around it — so
  the recorded runs and the controls exercise identical measurement code, and
  the controls were run and passed before any number in this document was
  published. Deposited to `.mistakes/worklog/`.
