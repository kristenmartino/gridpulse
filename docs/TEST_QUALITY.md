# Test quality — coverage, mutation testing, and what each one can prove

> **Whole table re-measured 2026-08-07** after #377, #383, #385, #386, #416,
> #426 and #434 — baseline and adjudication in one pass. `models/evaluation.py`
> (#442) and `models/skill.py` (#441) were re-run separately after it; the
> adjudication counts follow them by arithmetic, as the note below says.
> Regenerate with `python scripts/mutation_test.py`; re-adjudicate with
> `python scripts/adjudicate_mutants.py --fast` (~25 min each).

Most of this repo's tests are agent-written. That is fine, but it means "the
suite is green" carries less information than usual: a test that asserts
nothing useful is also green. This file is about replacing trust with
measurement.

Two instruments, answering two different questions:

| instrument | question it answers | where |
|---|---|---|
| **Coverage** | did this line *run*? | CI, every PR — HTML artifact + PR comment + `diff-cover` |
| **Mutation testing** | would anything have *noticed* if this line were wrong? | weekly + on demand, advisory |

Coverage is a floor, not a verdict. `models/ensemble.py` was **85%**
line-covered and scored **61.6%** against behavioural mutants — every line ran,
and a third of the ways to break them went unnoticed. Pinning its guards took
that to **73.3%** without touching a line of production code or moving coverage
at all, which is the whole point: the two instruments measure different things.

---

## Reading a mutation score

A **mutant** is one small deliberate change to the source — `>` becomes `>=`,
`and` becomes `or`, a constant shifts. The suite runs. If it fails, the mutant
is **killed**. If it passes, the mutant **survived**, and you have a precise,
reproducible statement: *this line can be broken and CI stays green.*

Survivors are the output. The score is only a summary of them.

**Two scores are published, because one of them is misleading on its own.**
mutmut rewrites string constants (`"x"` → `"XXxXX"`) and structlog arguments
(`log.info(None, ...)`) prolifically. Nothing asserts on either, so they survive
by construction — 204 of this baseline's 473 survivors are that. The **logic
score** drops those from both sides of the ratio. Neither number is the "real"
one; the gap between them tells you how much of a low score is noise.

**A survivor is not automatically a bug.** Some mutants are *equivalent* — they
change the source without changing behaviour. `(1.0 / v) ** k` → `(2.0 / v) ** k`
inside `compute_ensemble_weights` survives because the result is normalised by
its own total immediately after, so the constant cancels. No test can kill that,
and none should try. Adjudication is a human step, and it is the point.

---

## Baseline

2,372 mutants scored, **whole table re-measured in one pass**, with
`models/evaluation.py` (#442) and `models/skill.py` (#441) re-run separately
afterwards — see the note under the table. Full run ≈ 25 min on an 8-core
laptop; a single module is ≈ 1 min.

| module | mutants | killed | logic surv. | noise surv. | score | logic score | |
|---|---:|---:|---:|---:|---:|---:|---|
| `data/feature_engineering.py` | 919 | 780 | 82 | 57 | 84.9% | **90.5%** | ↑ 83.8% |
| `data/quality.py` | 188 | 164 | 20 | 4 | 87.2% | **89.1%** | ↑ 71.6% |
| `models/skill.py` | 194 | 172 | 18 | 4 | 88.7% | **90.5%** | ↑ 88.6% |
| `models/rolling_eval.py` | 327 | 273 | 41 | 13 | 83.5% | **86.9%** | ↑ 76.7% |
| `simulation/scenario_engine.py` | 288 | 181 | 35 | 72 | 62.8% | **83.8%** | ↑ 77.1% |
| `models/evaluation.py` | 373 | 320 | 49 | 4 | 85.8% | **86.7%** | ↑ 77.1% |
| `models/ensemble.py` | 221 | 177 | 11 | 33 | 80.1% | **94.1%** | ↑ 91.6% |
| **overall** | **2,510** | **2,067** | **256** | **187** | **82.4%** | **89.0%** | ↑ 88.7% |

> **Three modules were re-run separately**, not one: `models/evaluation.py` in
> #442, `models/skill.py` in #441, and `models/ensemble.py` in #445. The other
> four rows are the whole-table pass. The overall row is the **column sums of
> the seven rows above it** — no PR's own figure is right on its own, because
> each is computed against a shared base and so omits the others' kills (#442
> published 1,891, #441 published 1,851, #445 published 1,948; the truth is
> 1,956). **Anything that re-measures one module must re-sum the column, not
> add its delta to whatever total it last read.**

Eleven rounds of fixes (#377, #383, #385, #386, #416, #426, #441, #442, #445,
#484, #487) took the overall logic score **78.6% → 89.0%** and killed **314**
mutants, without changing production behaviour anywhere except the one crash
#386 fixed. The mutant total rose from 2,349 to 2,510: #386's fix added lines,
#423 rewrote `recursive_autoregressive_forecast` for performance, #441 added a
parameter, #444 added `resolve_ensemble_weights`, #451/#478 added the smoothed
and shadow weighting surface, and #458 added `apply_weather_deltas`.

> **This sentence was wrong for two rounds**, reading 88.4% and nine rounds
> while the table beside it read 89.0% and listed eleven. #484 and #487 each
> added a row to the rounds table without touching the prose above it. Recorded
> rather than quietly corrected because it is the failure mode this document
> exists to prevent in code: **a derived number maintained by hand, next to the
> table it is derived from.** If it drifts again, compute it from the table
> instead of editing it.

**The most useful number here is not the total.** Compare what equal effort
bought in different places:

| round | scope | overall | that module |
|---|---|---:|---:|
| #377 | five decision boundaries in `rolling_eval` / `quality` | +0.2 pts | — |
| #383 | four guards in `ensemble` | +0.4 pts | **+11.7** |
| #385 | the `coerce_demand_artifacts` cluster | — | **+17.5** |
| #416 | the `skill.py` clusters | — | **+16.5** |
| #426 | four `feature_engineering` clusters | +2.8 pts | **+6.7** |
| #442 | the `evaluation` clusters + the unexercised defaults | +1.6 pts | **+9.6** |
| #441 | one definition of the skill block + its last two survivors | +0.2 pts | **+1.9** |
| #445 | the `ensemble` fallback + the warn-only bounds block | +0.8 pts | **+18.3** |
| #484 | the usability boundary on `ensemble`'s new #451/#478 surface | +0.3 pts | **+2.5** |
| #487 | `scenario_engine`'s live half — the part #458 put in production | +0.3 pts | **+6.7** |

A 2,372-mutant denominator makes every real fix look like rounding error, which
is exactly why the gate policy below is **per-module**. It is also why the
"logic score" column is the one to read: six modules carry an ↑, several by
10–18 points, while the headline moved 9.

**#426 is the exception that shows the rule.** It is the only round that moved
the headline noticeably (+2.8 pts), because `feature_engineering` is 39% of the
mutant population — the same work on a small module is invisible at the top
level and decisive at the module level.

`simulation/scenario_engine.py` is the clearest illustration of why both
columns are published: 53.4% raw looks alarming, but 75 of its 111 survivors
are log-argument rewrites. Its behavioural score is 77.9%, mid-pack — and it is
dormant code besides.

**The score is not perfectly repeatable.** A second run of `models/skill.py`
alone, from a clean checkout, scored 129/192 rather than 130/192 — one mutant
flipped, 0.5 pts. Small, but measured rather than assumed, and the reason the
policy below asks for several consecutive runs before anyone attaches a
threshold to this number.

**A third run reproduced the published row exactly**, which is the more useful
data point about the jitter: re-running `models/skill.py` against the
pre-#441 tree returned 192 / 164 / 21 / 7 — every cell identical to the row
this table had published. So the instrument is stable enough that a
same-machine, back-to-back A/B is worth more than comparing a fresh run against
a figure measured weeks earlier on a different tree. That is how #441's delta
below was attributed rather than assumed.

### Scope

The seven modules in `[tool.mutmut] only_mutate` — pure logic where a silently
wrong number reaches a published result. Deliberately **not** the whole
codebase: I/O clients and persistence produce slow mutants that are mostly
equivalent, and would bury the signal.

---

## Adjudicated survivors

**All 269 logic survivors are machine-verified** with
`scripts/adjudicate_mutants.py`: apply the mutant to the real source, run
tests, restore. Re-run in full against the current tree, so these counts match
the baseline above rather than trailing it.

| | |
|---|---:|
| confirmed (no test notices) | **267** |
| false survivors (mutmut was wrong) | **2** |

Both false ones are in `ensemble_combine`, both killed by
`test_stable_hash_reproducibility.py` — the subprocess blind spot of
limitation 1. **The tool's false-survivor rate is 2 in 269, or 0.74%.**

> **The adjudicator was not re-run for #442 or #441**; 304 → 269 is
> arithmetic, not a re-measurement. #442 killed 32 adjudicated logic survivors
> and #441 killed 3, neither added any, and the false pair is in
> `ensemble_combine`, which neither touches — so the numerator is unaffected
> either way.
>
> **This section briefly disagreed with the table above it**, which is the
> failure it is written to prevent: #442 re-measured the baseline to 272 logic
> survivors while leaving this section at 304, so the sentence promising these
> counts "match the baseline above rather than trailing it" was false by 32.
> Reconciled here as part of merging #441. A per-module re-run needs to carry
> its subtraction into this section, or the two halves of the ledger drift
> apart silently.
>
> The **shape table below is likewise not re-derived.** Its shares were
> computed against the 2026-08-05 population of 304; recomputing them needs the
> full seven-module adjudication. What is certain is that none of the 35 were
> `dtype=float`, so that class is still exactly 81 and its share rises to
> **~30%** against the new denominator, leaving an actionable population of
> **188**. Every remaining number in this section is from the full 2026-08-05
> re-adjudication.

> **This re-run existed to check a specific doubt, and the doubt was
> unfounded.** #386 found that the adjudicator could reuse stale bytecode:
> CPython invalidates a `.pyc` on `(mtime, size)` with mtime at one-second
> resolution, and many mutations preserve length exactly, so an
> apply/run/restore loop finishing inside a second could test code that was
> never actually mutated. The failure is silent and **biased in one
> direction** — it can only manufacture false *survivors*.
>
> That put a published number in doubt, so the whole population was
> re-adjudicated with the fix in place. **It found the same two, in the same
> function, killed by the same test.** The earlier figure was right; it just
> had not been *shown* to be right.
>
> Three full re-adjudications now agree on the *same two mutants*: 2/457
> (0.44%), 2/363 (0.55%), 2/304 (0.66%). **The numerator has never moved.**
> The percentage drifts upward only because each round of fixes shrinks the
> denominator, which is a good reason to quote the pair rather than the
> ratio.

> **Correcting an earlier claim in this file.** A previous revision said
> "roughly a third of what a mutation run reports is not a gap," from a sample
> of seven. That sample was chosen *because* the entries looked interesting,
> which is exactly how you get a biased estimate. Measured over the whole
> population it is 2 in 304. The lesson survives — adjudicate before acting —
> but the number was wrong and the reasoning behind it was worse.

Verification protocol: apply the mutation **one at a time**, run tests, revert.
One at a time matters — two mutations applied together produced a failure that
neither caused alone, which would have mis-adjudicated both. For a *fix*, the
protocol runs twice: the mutation must fail the new test and pass without it,
or the test is decorative.

### Confirmed does not mean broken

`confirmed` means *no test noticed*. It does not rank severity, and the 302 are
not 302 defects. Two things matter more than the count:

**They cluster by untested function, not by independent defect.** The top
clusters are one missing test each, not dozens of bugs:

| confirmed | site | |
|---:|---|---|
| 26 | `simulation/scenario_engine.py::_run_ensemble` | dormant — see below |
| 9 | `models/evaluation.py::check_long_horizon_sanity` | the #296 guard — worked in #442 |
| 14 | `data/feature_engineering.py::compute_heat_index` | |
| 13 | `data/feature_engineering.py::compute_autoregressive_snapshot` | |
| 12 | `data/feature_engineering.py::engineer_features` | |
| 11 | `data/quality.py::coerce_demand_artifacts` | |
| 10 | `models/skill.py::seasonal_naive_forecast` | |

**Every cluster that has ever topped this table has now been worked**, and the
table has flattened: the largest is 26 (in dead code) where it used to be 40,
and nothing else clears 20.

Worked so far: `coerce_demand_artifacts` 40 → 11 (#385), `verdict` 32 → 6
(#386), `skill_payload` + `seasonal_naive_forecast` 37 → 13 (#416), the
four `feature_engineering` clusters 85 → 32 (#426), the three
`evaluation` clusters 44 → 21 (#442), and `ensemble_combine` 20 → 8 (#445).

**`simulation/scenario_engine.py` is the only cluster left**, and it is dead
code. Every live module is now at 86.7% or better.

**`models/evaluation.py` is now worked** (#442): 81 → 49, and 42 of the 49 that
remain are the closed `dtype=float` class. Its three clusters went 44 → 21.

**`models/ensemble.py` grew and was re-closed** (#484). #451 (EWMA-smoothed
holdout MAPE) and #478 (shadow weighting) added 58 mutants — 160 → 251 lines —
and **their own tests killed all but three**. Those three were one shape, in
all three new functions: the usability filter `v > 0` relaxed to `v > 1`,
which rejects any MAPE in (0, 1] as unusable.

That band is the *best* models on the easiest BAs, not noise — this repo
quotes 1.6% as a **baseline** figure. So the mutation drops the strongest
measurements, which is the worst direction for a filter meant to reject junk.
The sharpest consequence is in `shadow_weighting_mape`, which returns `None`
when the alternative is unavailable *by design* (a fallback would make both
A/B arms identical and the comparison vacuously "no difference"): under the
mutation the best models vanish from the shadow arm entirely, **biasing a live
experiment toward whatever the remaining, worse models support**.

Module now **94.1%**, and the 11 remaining survivors are the same set proved
equivalent in #445.

**`models/ensemble.py` was first closed** (#445): 23 → 8, logic **73.3% → 91.6%**,
the largest single-module move of any round. Every one of the 8 that remain
carries a proof — six equivalent (three `zip(..., strict=)` variants, which
cannot differ because `arrays` is built by iterating `model_names`; three
constants that cancel under normalisation, verified across 1.0 / 2.0 / 7.0) and
two exact-float-equality tolerance variants. There is nothing left to write a
test for in `compute_ensemble_weights` or `ensemble_combine`.

Scoped deliberately to those two, because a third arrived mid-flight. #444
added `resolve_ensemble_weights` to this module while #445 was in review, so
the row above is **re-measured against the merged result**, not against the
`8f28cea` baseline the work started from — the #423 lesson, which cost #426 a
wrong commit subject. What the re-measure costs is visible in the row: 30 more
mutants, so the module reads **91.6%** rather than the 92.2% measured before
the rebase, on strictly more code and with the same 8 survivors in the two
functions this round touched.

`resolve_ensemble_weights` landed clean: 30 mutants, 25 killed by its own
tests, 3 logic survivors — and all 3 are equivalent for the same reason three
of `ensemble_combine`'s are. It renormalises the output of
`compute_ensemble_weights`, which already sums to 1.0 (checked across five MAPE
shapes; worst deviation 1.1e-16). So `or 1.0` → `or 2.0`, `or 1.0` → `and 1.0`,
and `v / total` → `v * total` are all no-ops on a `total` that is 1.0 to float
precision. Belt-and-braces renormalisation is a reasonable thing to write and
an unkillable thing to mutate.

**`simulation/scenario_engine.py` is no longer dormant, and the ledger said so
for three days after it stopped being true.** #458 put `apply_weather_deltas`
and `_recompute_derived_features` on the hourly scoring path — 81 calls per
region, 51 regions — while this row still read "dormant". The module also grew
238 → 288 mutants. **A module's dormancy is a fact about its callers, not a
property of the module, and nothing re-checks it.**

Split by caller rather than by count (#487):

| | survivors | on the served path? |
|---|---:|---|
| `_run_ensemble` | 26 | no production caller |
| `compute_scenario_impact` | 7 | no production caller |
| `simulate_scenario` | 2 | no production caller |
| `apply_weather_deltas` | 11 → **0** | **live** — scoring job, hourly, 51 BAs |
| `_recompute_derived_features` | 1 → **0** | **live** |

**Every survivor on the served path is now pinned; all 35 that remain are in
code nothing calls.** That is a more useful closure statement than the 83.8%.

The gap worth recording: **inverting the solar sign passed the entire suite.**
`test_derived_features_follow_the_drivers` asserted
`solar_capacity_factor > 0` off a 300 W/m² baseline, so a +200 delta that
silently became −200 still left 100 W/m² and a positive factor. A relationship
assertion over a lenient fixture cannot see a sign flip — the #426 finding
again, this time in code written *by* the person who had just documented it.

`_run_ensemble` still survives *everything*, including `forecasts = None` and
`ensemble_combine(None, weights)`. That is not 26 findings. It is one function
with no test — and one that does not run in production. Note it is deliberately
bypassed: `scenario_grid` uses the production recursive forecaster instead,
because a scenario and a baseline from different inference paths report the gap
between the paths as the response to weather (ADR-013).

**Shape predicts value.** Grouping the 302 by what the mutation actually
changed:

| share | shape | what it usually means |
|---:|---|---|
| 26.6% | `dtype=float` dropped | **closed** — real guards, unreachable input; see below |
| 11.9% | arithmetic operator/constant | usually real |
| 10.9% | control flow (`continue`/`break`/`return`) | usually real — produced a fixed gap in #377 |
| 8.3% | comparison boundaries | real — produced nine fixed gaps across #377/#383/#385/#416 |
| 7.6% | assignment nulled (`x = None`) | crashes on mutation; concentrated in untested functions |
| 2.6% | `and` → `or` | often equivalent where NaN propagates anyway |

The `dtype=float` share rose every round — 17.8% → 22.4% → 26.6% → **~30%** —
while the absolute count never moved off 81. Nothing was multiplying; the
behavioural classes around them kept getting pinned. After #442 that is close
to the whole story in `models/evaluation.py`: 42 of its 49 remaining survivors
are this class, and the other 7 are float-boundary variants (`> 1e-10` →
`>= 1e-10`, `> R2_THRESHOLD` → `>=`) that require landing on an exact float
equality no real series produces.

**That class is now closed** (below): the guards are real, the input they guard
is no longer reachable, and #434 fixed the single line that produced it. They
are equivalent *in practice*, which means **the actionable population is 188,
not 269.** Read the remaining shares against that denominator.

### The `dtype=float` class — CLOSED, and the fix did not move the score

81 survivors drop `dtype=float` from an `np.asarray(...)`. This was the largest
open question in this file for three revisions. It is now answered, and the
answer is worth more than the number.

**They are real guards.** On an **object-dtype** array `np.isfinite` raises
`TypeError`, while `dtype=float` converts `None` to `nan` and carries on:

```python
np.isfinite(np.asarray(obj_col, dtype=None))   # TypeError
np.asarray(obj_col, dtype=float)               # [100.  nan 200.]
```

An earlier pass here called them "equivalent noise" and only a direct probe
caught it. Shape-based triage is a prioritisation aid, not a verdict.

**But the input they guard is no longer reachable.** A sweep of every way an
object-dtype numeric column could enter (#434):

| producer | dtype |
|---|---|
| `_parse_demand_records` `demand_mw` | float64 always — every value goes through `float(raw)` with a NaN fallback |
| `_parse_demand_records` `forecast_mw` | float64 **since #434** — the no-forecast branch assigned `None`, building an object column; it was the only such line in the codebase |
| `recursive_autoregressive_forecast` history | float64 — built with an explicit `float(v)` and a None/NaN filter |
| `jobs/phases.py` skill path | float64 — `window.to_numpy(dtype=float)` |
| anywhere assigning `= None` to a numeric column | none remain (grepped across `data/`, `models/`, `jobs/`) |
| anywhere using `dtype=object` / `astype(object)` | none exist |

**Verdict: equivalent in practice, not in principle.** The guards stay — they
are correct, and these helpers are importable by code that has not been written
yet. But no reachable path distinguishes them, so this class is **not
actionable** and should stop being counted as though it were. That is now an
evidenced position rather than a preference.

> **The re-measure after #434 was identical: 2,370 / 1,843 / 304, logic 85.8%,
> and the dtype class still exactly 81.** It had to be — those mutants survive
> because no *test* passes object dtype, and #434 changed production code, not
> tests. Recorded because the run could have flattered the story and did not:
> the reclassification above rests on the reachability sweep, not on any number
> moving. A fix that makes a whole class of survivor unreachable is invisible
> to the score that pointed at it.

**The rule this generalises to, which is the durable part:** a large,
homogeneous mutation class usually points at a *missing invariant upstream*,
not at N missing tests. 81 near-identical guards was the score showing where
duplication lived; the fix was one line at the source, not 81 tests. A small,
heterogeneous cluster is the opposite — that really is N decisions.

### A dead function that three tests pin

`models/skill.py::skill_payload` has **no production caller**. `jobs/phases.py`
imports `mape`, `skill_score`, `should_serve_baseline` and
`seasonal_naive_forecast` from that module — but builds the skill block inline
instead of calling `skill_payload`, and the two have already diverged:

| field | `skill_payload` | the inline block |
|---|---|---|
| `beats_baseline` | yes — its docstring calls this "the field worth acting on" | **absent** |
| `window_days`, `decision` | absent | yes |
| non-finite guards | `None if not np.isfinite(...)` | none — `round(float(...), 3)` directly |

So #416's `should_serve_baseline` and `seasonal_naive_forecast` work is on the
serving path, and its `skill_payload` work is not. Recorded plainly rather than
left to inflate that PR's value. Worth an issue: either call the function or
delete it, because a tested payload that production does not emit is a
liability dressed as coverage.

| # | site | mutation | verdict |
|---|---|---|---|
| 1 | `models/rolling_eval.py` `satisficing_check` | `regression > max_mape_regression_pts` → `>=` | **REAL GAP → FIXED** |
| 2 | `models/rolling_eval.py` `satisficing_check` | `abs(bias) > max_abs_bias_pct` → `>=` | **REAL GAP → FIXED** |
| 3 | `models/rolling_eval.py` `verdict` | `consistency < min_sign_consistency` → `<=` | **REAL GAP → FIXED** |
| 4 | `models/rolling_eval.py` `verdict` | `abs(mean) < min_t * stderr` → `<=` | **REAL GAP → FIXED** |
| 5 | `data/quality.py` `coerce_demand_artifacts` | `continue` → `break` on an absent reading | **REAL GAP → FIXED** |
| 6 | `models/rolling_eval.py` `verdict` | `winner = "treatment" if mean > 0` → `>=` | **EQUIVALENT** — unreachable, see below |
| 7 | `models/ensemble.py` `ensemble_combine` | `len(set(lengths)) > 1` → `>= 1` | **FALSE SURVIVOR** — actually killed (see limitation 1) |

Every one of 1–5 survived all 2,687 unit tests before the fix and fails after
it.

### Priority: five clusters fixed, one to ignore, one clearly next

**`models/ensemble.py` — was highest value, now FIXED.** ADR-004 weights feed
every served forecast, and three guards were unpinned. Each mapped to a
*reachable* crash, not a hypothetical one — the real code handled all three
correctly and nothing asserted that it did:

| mutation | consequence if the guard were weakened |
|---|---|
| `v > 0` → `v >= 0` | a MAPE of exactly 0 is admitted → `ZeroDivisionError` |
| `and np.isfinite(v)` → `or` | a non-finite MAPE is admitted → `ZeroDivisionError` |
| `weights.get(k, 0)` → `weights.get(k)` | a model missing from `weights` → `TypeError` |
| `1.0 / n` (fallback) | equal weights stop summing to 1, silently rescaling |

Reachability is not theoretical: `compute_mape` returns `inf` for all-zero
actuals, and TIDC publishes zeros (STATUS.md). An infinite MAPE reaching
`compute_ensemble_weights` is a documented scenario in this system.

**Fixed** by seven tests in `tests/unit/test_ensemble.py`, each verified by
re-applying its mutation one at a time. Measured effect — much larger than the
0.2 pts the first round of boundary fixes moved, because these landed on the
weakest module rather than its best-tested one:

| | before | after |
|---|---:|---:|
| mutants killed | 53 | **63** |
| logic survivors | 33 | **23** |
| logic score | 61.6% | **73.3%** |

`models/ensemble.py` is no longer the lowest-scoring module.

Why the pre-existing `test_handles_inf_mape` did not cover any of it: it pairs
`inf` with a *healthy* model, which leaves a non-zero denominator, so the
weights come out right **even with the guard broken**. That is the recurring
shape of these gaps — a test exists, exercises the line, and asserts something
true in the one arrangement where the guard does not matter.

One mutant in `ensemble_combine` was checked and deliberately left unpinned:
`weights = {name: 1.0 / len(model_names)}` is renormalised on the next line, so
any non-zero constant gives identical output (verified across 0.333 / 0.667 /
3.0 / 7.0). **Equivalent** — a test for it would be theatre.

> **Correction (#445): that verdict was right, and it was applied to the wrong
> line as well.** The *identical* expression appears twice in
> `ensemble_combine`, six lines apart. On the first it is renormalised and the
> constant cancels. On the second — inside `if total == 0:` — it is followed by
> a hard-coded `total = 1.0`, so nothing renormalises it and the constant is
> the answer. Mutated there, a stale weights dict returns **300.0, 600.0 or
> 75.0** for an input whose answer is 150.0: the served forecast scaled by 2x,
> 4x or 0.5x, still finite and still plausibly shaped. Five survivors sat in
> that branch, reachable three ways (renamed model keys, an empty dict, an
> all-zero dict), none covered. **The lesson is about adjudication, not about
> ensembles: an equivalence verdict is a claim about a line's context, not
> about its text, and it does not travel to a line that looks the same.**

**`data/quality.py::coerce_demand_artifacts` (40 → 11) — fixed in #385.** The
day-ahead signal was never exercised through the series function, and the
disclosure payload was asserted by count and substring only.

**`models/rolling_eval.py::verdict` (32 → 6) — fixed in #386,** which turned up
a **live crash** rather than a test gap: identical deltas in every window raised
`ZeroDivisionError`. The `stderr == 0` branch was handled deliberately at the
top of the function and then divided by `stderr` anyway in the closing reason
string. Four survivors sat in that branch because nothing reached it, and
nothing reaching it is exactly why the crash went unnoticed.

**`models/skill.py` (49 → 21) — fixed in #416.** The substitution boundary that
decides whether a region is served a naive forecast, and the edges of the
projection that replaces the model.

**...and the tests landed on a function production did not call.** #416's
exact-payload assertions pinned `skill_payload` field by field. Nothing
outside the test file imported it: the scoring job hand-rolled its own copy of
the same block inline, and the two had already diverged — production emitted
`window_days` and `decision` but no `beats_baseline`, the field the module
docstring calls "the field worth acting on", and it skipped the non-finite
guards. `should_serve_baseline` consumed the inline dict and happened to work
because it reads only two keys.

**A mutation score cannot see this.** Every mutant of `skill_payload` was
killed, so the module scored well precisely *because* the dead function was
well tested; the served block had no direct coverage at all. Coverage and
mutation testing both answer "is this code tested" and neither answers "is
this code reached". Worth a `grep` for a caller before reading a strong
per-function score as reassurance.

Resolved 2026-08-07 (#441): the job now builds the block via `skill_payload(...,
window_days=7)` and attaches `decision` from `should_serve_baseline`, so there
is one definition and the tests pin the shape that is served. `beats_baseline`
joins the published payload — additive, and always `False` where it appears,
since the block is only written on ticks where substitution fired.

**Re-measured, A/B, same machine, back to back** — the pre-change tree returned
the published row exactly (192 / 164 / 21 / 7), so the delta is attributable
rather than assumed:

| | before | unified | + the last two tests |
|---|---:|---:|---:|
| mutants | 192 | 194 | 194 |
| killed | 164 | 170 | 172 |
| logic survivors | 21 | 20 | **18** |
| noise survivors | 7 | 4 | 4 |
| logic score | 88.6% | 89.5% | **90.5%** |

**4 newly killed, 0 newly surviving**, and the composition is the interesting
part. Three are the `"skill not measurable"` reason string, now asserted
exactly rather than by truthiness. The fourth is `mape`'s `return float("nan")`
mutated to `float(None)` — which raises. It survived before **because no test
had ever executed that line**: at 98% the module had exactly one uncovered
statement, and it was the one deciding what `mape` returns when nothing is
measurable. That is the same branch the non-finite-guard story runs through,
and the test that reaches it now is the one asserting the guard keeps the
model. Coverage is 100%.

**The last two were then killed too, and both were worth killing** — neither
was the theatre that "pin the remaining survivors" usually becomes:

- `skill_payload` forwards `lag_h` into the baseline call. Drop that one
  argument and the call falls back to the 24h default, so the block publishes
  a number measured at 24h **under a label that says 48h**. Every prior test
  used the default lag, where label and number are indistinguishable. On the
  fixture the test now uses, the mutation inverts `beats_baseline` from False
  to True — the one field the module exists to get right.
- `seasonal_naive_forecast`'s loop starts at lead 1. Starting it at 0 probes
  the origin hour itself. At a horizon of 24 or more that is invisible: lead
  24 reads the same index and the extra write is overwritten before return.
  Below a day it is not, because the origin hour becomes the only index whose
  gap can veto the whole forecast — and a gap in the most recently reported
  hour is the likeliest one there is. Every other test in that class uses a
  horizon of 24 or 72.

**All 18 remaining logic survivors are the `dtype=float` class — zero others.**
So this module is now pinned everywhere except a question that was already
closed as deliberate.

**`simulation/scenario_engine.py` (36) — still deliberately ignored.** Nothing
in `components/`, `jobs/`, `api.py` or `app.py` imports `simulation`. The live
Scenarios feature runs a heuristic; issue #127 tracks replacing it with this
engine. So these are real test gaps in code that **does not run in
production** — zero operational risk today, and worth re-adjudicating if #127
lands. A raw mutation score cannot tell you this; it is why the score is not
a gate.

**`data/feature_engineering.py` (141 → 82) — fixed in #426.** It held 39% of
all remaining survivors and four of the top eight clusters. The finding worth
carrying forward is *why* so much survived there: the existing tests asserted
**relationships** rather than values. `test_hour_0_and_24_equal` compares two
midnights, which agree under any period; `test_rated_wind` asserts `> 0.5`,
true across a wide span of wrong physics; the snapshot tests assert parity with
the training path, so both could drift together. None is a bad test — each pins
a real invariant — but a mutation that preserves the relationship walks
straight through. That is a review heuristic independent of this tooling.

**`models/evaluation.py` (81 → 49) — fixed in #442.** It held three of the top
ten clusters: `check_long_horizon_sanity` (18), the #296 guard against
degenerate forecasts reaching the serve path; `compute_interval_coverage_drift`
(16); and `compute_interval_coverage` (10).

The cause was one sentence long and it is the most transferable finding in this
document so far: **every existing test passed every parameter explicitly, so
none of the published defaults were ever executed.** `lower_q=0.10`,
`upper_q=0.90`, `target_coverage=0.80`, `window_size=168` — each could be
changed to any value at all with the full suite green. Those four numbers are
the contract behind the "80% empirical prediction interval" the Forecast tab
renders; that 80% is `0.10` and `0.90` in this file and nowhere else.

It is a blind spot neither coverage nor a normal review catches. The defaults
are *covered* — the lines execute on every call. They are simply never
*exercised*, because a test that passes `lower_q=0.2` is testing the caller's
number, not the module's. Calling a function the way production calls it —
with the arguments omitted — is a different test from calling it with the
arguments spelled out, and the parameterised version silently replaces it.

Two smaller findings from the same round:

- `compute_error_by_hour` had no value assertion at all. Its one test built
  random inputs and asserted `len(result) == 24` plus a column name, so
  `abs_errors = None` survived: the Backtest heatmap could render 24 empty
  cells with CI green.
- `compute_r2` returns `0.0`, not `1.0`, when the actual series has no
  variance. Nothing exercised that branch, and flipped it would publish a
  *perfect* R² for an hour of flat demand off a forecast never tested.

### 1. The decision boundaries were not pinned — **fixed**

[`EVALUATION_POLICY.md`](EVALUATION_POLICY.md) states the constraints that veto
an A/B win: `|bias| ≤ 2%`, `MAPE regression ≤ 0.5 pts`. Both boundary
comparisons in [`satisficing_check`](../models/rolling_eval.py) could be flipped
to `>=` with the entire unit suite green. A treatment landing *exactly* on the
threshold was decided by an untested branch.

The same class of survivor ran through [`verdict`](../models/rolling_eval.py):
`consistency < min_sign_consistency` → `<=`, and
`abs(mean) < min_t * stderr` → `<=`.

This matters more than the raw numbers suggest: CLAUDE.md makes
`rolling_eval.verdict()` the mandatory route for *every* model change. It was
the most load-bearing decision function in the repo, with the least pinned
boundaries in it.

**Fixed** by four boundary tests in `tests/unit/test_rolling_eval.py`. Each was
verified by re-applying its mutation to the working copy, one at a time, and
confirming the new test fails. The noise-threshold test injects `min_t` as the
exact ratio the data produces, so the comparison really lands on equality
rather than near it.

The policy question these tests settle, which is the actual value: **the
thresholds are inclusive.** Exactly 2% bias ships; exactly 0.5 pts of MAPE
regression ships; exactly 75% sign consistency ships. That matches
EVALUATION_POLICY.md's "≤" wording, and it is now enforced rather than implied.

#### One that looked like a gap and is not

An earlier reading of this survivor list flagged
`winner = "treatment" if mean > 0` → `>=` as a fifth boundary gap. It is an
**equivalent mutant**. `verdict` returns early on `if mean == 0:` ("no
difference") well before that line, so `mean > 0` and `mean >= 0` cannot
disagree there. No test can kill it and none should be written for it —
demonstrated directly:

```python
verdict([1.0, -1.0, 1.0, -1.0])["reason"]   # 'no difference' — returns early
```

Recorded because a wrong entry in a survivor ledger is worse than no ledger:
it sends someone to write a test that cannot fail.

### 2. The demand-artifact guard's skip was not pinned — **fixed**

In [`coerce_demand_artifacts`](../data/quality.py), the trailing-hours loop
`continue`s past an absent reading. Changed to `break`, the loop stops at the
first gap and every later artifact goes uncoerced — and no test noticed. This
is the #309 guard that keeps implausible EIA partials out of the forecast
anchor, so a silent early exit is exactly the failure it exists to prevent.

Why the gap existed: every prior test put its NaNs at the *end* of the frame,
where skipping and stopping are indistinguishable. The arrangement that tells
them apart — a gap **followed by** an artifact — is ordinary EIA behaviour for
the broken-feed BAs the guard was written for, and it was the one shape
untested.

**Fixed** by `test_a_gap_does_not_stop_the_scan` in
`tests/unit/test_quality.py`, verified the same way.

### 3. Why #7 was false — and why that is worth writing down

mutmut reported the `ensemble_combine` length-guard as survived. It is not:
`test_stable_hash_reproducibility.py::test_simulated_forecasts_identical_across_processes`
kills it. mutmut never ran that test against the mutant, because it does its
work in a **subprocess** — coverage tracing sees nothing, so the stats pass maps
it to *zero* functions:

```python
# functions mutmut associates with that test: NONE
# is it in duration_by_test at all?  True   (it ran; it just traced nothing)
```

Caught only because the finding was verified before publishing. This is the
argument for the adjudication step, in one example.

**Follow-up (#445): the kill is real by the score's definition and is not
protection.** Look at *how* that test kills it. It runs a forecast in a
subprocess and parses stdout as JSON:

```
E   json.decoder.JSONDecodeError: Extra data: line 1 column 5 (char 4)
```

The mutant makes `log.warning` fire where it should not; structlog writes to
stdout; `json.loads` then chokes on the extra line. The test asserts that two
runs produce identical forecasts — it has no opinion about bounds checking and
does not import this module. Routing structlog to stderr, or making the test
read only its last line, silently removes the kill and nothing reports it.

So a mutant can be "covered" by an assertion that would not survive a
five-minute refactor of an unrelated file. #445 kills all three deliberately,
with `structlog.testing.capture_logs`, so the protection now sits in the test
that names the behaviour.

### 4. A warn-only block cannot be tested through its return value

Eleven of `ensemble_combine`'s survivors were in one block: it computes whether
the ensemble left the pointwise min/max band of its inputs, logs a warning, and
returns `result` untouched. Fail-open is the right design — a bounds violation
is worth telling an operator about, not worth refusing to serve a forecast over
— but it means **no assertion on the return value can reach any of those
eleven**. They are not equivalent (behaviour changes: the warning stops firing,
or fires on every call) and not covered. They are *unobservable* through the
interface every existing test used.

Naming that third category is the useful part. The fix is to assert the
diagnostic, which #445 does. Two consequences worth recording:

- **Nine of the eleven die immediately.** The two that do not need a result
  landing exactly on `min - 1e-6` — a float equality no real series produces.
- **The module's "noise" count fell 47 → 30** (the table reads 32, after
  #444's function landed). Seventeen structlog-argument
  mutants were previously classified as surviving *by construction*, which is
  true only while nothing asserts on logs. Where a log line is a load-bearing
  diagnostic rather than a trace, its arguments are behaviour and the noise
  classification understates the module. That is a caveat on the logic score,
  not a defect in it — but it means a high logic score on a module that
  communicates mainly through logs deserves a second look.

---

## Known limitations

These bound what the score can prove. None of them are silent.

1. **Subprocess tests cannot kill mutants.** mutmut selects per-mutant tests
   from coverage tracing, which is blind to work done in a child process. Any
   mutant that only such a test would catch is reported as a false survivor.

   **The blind spot is now enumerated, and it is small.**
   `grep -rln "subprocess\|multiprocessing\|Popen" tests/unit/` returns exactly
   two files — `test_stable_hash_reproducibility.py` and `test_cache.py` — and
   only the first has been observed killing anything. That bound is what makes
   `scripts/adjudicate_mutants.py --fast` sound: mutmut has already run every
   *traced* test against a survivor, so confirming one needs only the tests
   tracing could not see.

   **Cross-validated, not assumed.** The same 69 mutants were adjudicated both
   ways — full suite and `--fast`:

   | | full suite | `--fast` |
   |---|---:|---:|
   | confirmed | 67 | 67 |
   | false survivors | 2 | 2 |
   | wall clock | 84 min | **4 min** |

   **Zero disagreements.** 25x faster for the same answers, which is what made
   adjudicating the whole population practical rather than aspirational.

   Re-derive the list if tests start spawning processes elsewhere. Running the
   verifier without `--fast` is always the unconditional check.

2. **Equivalent mutants are counted as survivors.** No tool can identify them
   automatically. They are a permanent floor under 100%, which is one reason
   there is no target score.

3. **Unit tests only.** Integration and smoke tests are not in the selection —
   they need infrastructure the `mutants/` tree does not have.

4. **Threading is pinned to one thread per pool.** `scripts/mutation_test.py`
   sets `OMP_NUM_THREADS=1` and friends before invoking mutmut. Without it,
   forking a parent with a live threaded BLAS pool deadlocks the workers — they
   sit at 0% CPU indefinitely and mutmut's wall-clock timeout never fires
   because nothing is running.

---

## Verifying a survivor

`scripts/mutation_test.py` reports what mutmut *found*.
`scripts/adjudicate_mutants.py` checks whether it is *true*.

```bash
python scripts/adjudicate_mutants.py --fast                       # everything
python scripts/adjudicate_mutants.py --fast --module models/ensemble.py
python scripts/adjudicate_mutants.py --module data/quality.py     # full suite
```

For each survivor it applies the mutant to the real source file, runs tests,
and restores the file:

* tests **pass** -> `confirmed` — genuinely unnoticed
* tests **fail** -> `false-survivor` — mutmut missed the killing test, which is
  named in the output

It refuses to start against a dirty working tree, because it restores files
from memory and a mid-run kill would otherwise leave a mutated source behind.

`confirmed` is still not `bug`. An **equivalent** mutant changes the source
without changing behaviour and nothing can ever kill it. That last call is
human; the script's job is to shrink the field it has to be made over.

---

## Running it

```bash
python scripts/mutation_test.py                        # all seven, ~25 min
python scripts/mutation_test.py --module models/skill.py   # one module, seconds
python scripts/mutation_test.py --skip-run             # re-report existing state
python scripts/mutation_test.py --max-children 4       # cap parallelism
```

Outputs `mutation-report.json` (full survivor diffs) and `mutation-report.md`
(score table + logic survivors). Both gitignored — this file is the durable
record.

`mutants/` is a throwaway tree mutmut copies the source into. The script never
edits tracked files.

> **If you apply a mutation by hand** to check a finding: **commit first.**
> Reverting with `git checkout <file>` destroys any uncommitted work in that
> file, not just the mutation.

CI: [`.github/workflows/mutation.yml`](../.github/workflows/mutation.yml) —
weekly (Mondays 06:00 UTC) and `workflow_dispatch`, with an optional single
module input. Score table lands in the job summary; JSON and markdown are
uploaded as artifacts for 90 days.

---

## Policy: why this is not a gate

It is advisory, and stays that way until all three hold:

1. ⬜ The baseline is stable across at least four consecutive weekly runs (no
   unexplained swings from test ordering or flakiness). One re-run of
   `models/skill.py` already moved by a single mutant, so this is not
   theoretical — though a later re-run of the same module reproduced its
   published row cell-for-cell, so the jitter looks like an occasional
   single-mutant flip rather than a persistent drift.
2. ✅ **Resolved, and re-verified twice.** The false-survivor rate is
   **measured at 2/304 (0.66%)** — the same two mutants found by all three
   full re-adjudications — the blind spot behind it is enumerated (limitation 1), and
   the stale-bytecode defect that could have masked more of them is fixed and
   the whole population re-checked — same two, same function, same killing
   test. The deselected slow test that was once limitation 2 is gone;
   re-baselining after it landed changed **no** verdict, so it had been killing
   zero mutants.
3. ✅ **Resolved.** All 269 logic survivors are machine-verified, clustered by
   function, and grouped by mutation shape above.

Only (1) is outstanding, and it needs four weeks of scheduled runs rather than
any decision. Note that condition (1) is about *stability*, not *stasis*: the
whole table has been re-measured on demand four times in a week, and each
re-run cost ~25 min. The weekly job exists so that stability can be observed
without anyone asking for it. **When it clears, the honest threshold is per-module logic score
with a no-regression rule** — "this module's logic score may not fall" — rather
than an absolute bar. The floor moves as modules are fixed (`models/ensemble.py`
went 61.6% -> 73.3%), so any absolute bar is either instantly stale or set so
low it can never fire. A no-regression rule needs no number at all.

A threshold picked before these are known would be picked from nothing, fire on
noise, and get switched off within a month. The failure mode of mutation
testing in practice is not a low score; it is abandonment.

When it does become a gate, the unit is the **logic score per module**, not the
overall number — `data/feature_engineering.py` is 919 of 2,370 mutants and
would otherwise drown `models/ensemble.py`'s 133. #426 is the demonstration:
the same class of work moved the headline 2.8 points there and ~0 points on the
small modules.
