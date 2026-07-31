# Test quality — coverage, mutation testing, and what each one can prove

> Baseline measured 2026-07-31 on `main` @ `ebbb48f`.
> Regenerate with `python scripts/mutation_test.py`.

Most of this repo's tests are agent-written. That is fine, but it means "the
suite is green" carries less information than usual: a test that asserts
nothing useful is also green. This file is about replacing trust with
measurement.

Two instruments, answering two different questions:

| instrument | question it answers | where |
|---|---|---|
| **Coverage** | did this line *run*? | CI, every PR — HTML artifact + PR comment + `diff-cover` |
| **Mutation testing** | would anything have *noticed* if this line were wrong? | weekly + on demand, advisory |

Coverage is a floor, not a verdict. `models/ensemble.py` is **85%**
line-covered and scores **61.6%** against behavioural mutants. Every line ran;
a third of the ways to break them go unnoticed.

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
by construction — 259 of this baseline's 722 survivors are that. The **logic
score** drops those from both sides of the ratio. Neither number is the "real"
one; the gap between them tells you how much of a low score is noise.

**A survivor is not automatically a bug.** Some mutants are *equivalent* — they
change the source without changing behaviour. `(1.0 / v) ** k` → `(2.0 / v) ** k`
inside `compute_ensemble_weights` survives because the result is normalised by
its own total immediately after, so the constant cancels. No test can kill that,
and none should try. Adjudication is a human step, and it is the point.

---

## Baseline

2,349 mutants scored. Full run ≈ 25 min on an 8-core laptop.

| module | mutants | killed | logic surv. | noise surv. | score | logic score |
|---|---:|---:|---:|---:|---:|---:|
| `data/feature_engineering.py` | 903 | 705 | 141 | 57 | 78.1% | **83.3%** |
| `models/evaluation.py` | 373 | 272 | 81 | 20 | 72.9% | **77.1%** |
| `simulation/scenario_engine.py` | 238 | 127 | 36 | 75 | 53.4% | **77.9%** |
| `models/rolling_eval.py` | 322 | 220 | 67 | 35 | 68.3% | **76.7%** |
| `models/skill.py` | 192 | 130 | 49 | 13 | 67.7% | **72.6%** |
| `data/quality.py` | 188 | 126 | 50 | 12 | 67.0% | **71.6%** |
| `models/ensemble.py` | 133 | 53 | 33 | 47 | 39.8% | **61.6%** |
| **overall** | **2,349** | **1,633** | **457** | **259** | **69.5%** | **78.1%** |

`models/rolling_eval.py` and `data/quality.py` are re-measured after the fixes
below; the other five rows are the original baseline. **The fixes moved the
overall score by 0.2 points** — 69.3% → 69.5%, six mutants. That is the honest
scale of pinning five boundaries in a 2,349-mutant population, and it is worth
stating plainly: the value was never the score. It was learning that the
function CLAUDE.md makes mandatory for every model change had unpinned
decision boundaries.

`simulation/scenario_engine.py` is the clearest illustration of why both
columns are published: 53.4% raw looks alarming, but 75 of its 111 survivors
are log-argument rewrites. Its behavioural score is 77.9%, mid-pack.

**The score is not perfectly repeatable.** A second run of `models/skill.py`
alone, from a clean checkout, scored 129/192 rather than 130/192 — one mutant
flipped, 0.5 pts. Small, but measured rather than assumed, and the reason the
policy below asks for several consecutive runs before anyone attaches a
threshold to this number.

### Scope

The seven modules in `[tool.mutmut] only_mutate` — pure logic where a silently
wrong number reaches a published result. Deliberately **not** the whole
codebase: I/O clients and persistence produce slow mutants that are mostly
equivalent, and would bury the signal.

---

## Adjudicated survivors

**All 457 logic survivors have now been machine-verified** with
`scripts/adjudicate_mutants.py`: apply the mutant to the real source, run
tests, restore. Result:

| | |
|---|---:|
| confirmed (no test notices) | **455** |
| false survivors (mutmut was wrong) | **2** |

Both false ones are in `ensemble_combine`, both killed by
`test_stable_hash_reproducibility.py` — the subprocess blind spot of
limitation 1. **The tool's false-survivor rate is 0.4%, not a third.**

> **Correcting an earlier claim in this file.** A previous revision said
> "roughly a third of what a mutation run reports is not a gap," from a sample
> of seven. That sample was chosen *because* the entries looked interesting,
> which is exactly how you get a biased estimate. Measured over the whole
> population it is 2 in 457. The lesson survives — adjudicate before acting —
> but the number was wrong and the reasoning behind it was worse.

Verification protocol: apply the mutation **one at a time**, run tests, revert.
One at a time matters — two mutations applied together produced a failure that
neither caused alone, which would have mis-adjudicated both. For a *fix*, the
protocol runs twice: the mutation must fail the new test and pass without it,
or the test is decorative.

### Confirmed does not mean broken

`confirmed` means *no test noticed*. It does not rank severity, and the 455 are
not 455 defects. Two things matter more than the count:

**They cluster by untested function, not by independent defect.** The top
clusters are one missing test each, not dozens of bugs:

| confirmed | site | |
|---:|---|---|
| 40 | `data/quality.py::coerce_demand_artifacts` | |
| 32 | `models/rolling_eval.py::verdict` | |
| 26 | `simulation/scenario_engine.py::_run_ensemble` | dormant, see below |
| 24 | `models/ensemble.py::ensemble_combine` | |
| 23 | `data/feature_engineering.py::compute_cyclical_dow` | |

`_run_ensemble` survives *everything*, including `forecasts = None` and
`ensemble_combine(None, weights)`. That is not 26 findings. It is one function
with no test.

**Shape predicts value.** Grouping the 455 by what the mutation actually
changed:

| share | shape | what it usually means |
|---:|---|---|
| 17.8% | `dtype=float` dropped | defensive coercion — see below, **not** equivalent |
| 16.9% | arithmetic operator/constant | usually real |
| 8.6% | control flow (`continue`/`break`/`return`) | usually real — this class produced a fixed gap |
| 7.9% | `round()` precision | cosmetic unless the rounding is contractual |
| 7.7% | assignment nulled (`x = None`) | crashes on mutation; concentrated in untested functions |
| 11.6% | comparison boundaries, `and`/`or`, negation | real — this class produced four fixed gaps |

### The `dtype=float` class is load-bearing, and I nearly got that wrong

81 survivors drop `dtype=float` from an `np.asarray(...)`. The obvious reading
is that they are equivalent: for lists of floats and for int arrays the values
come out identical, and `compute_mape` returns the same number either way.

That reading is wrong. On an **object-dtype** array — which pandas produces
routinely here from mixed or missing EIA data — `np.isfinite` raises
`TypeError` without the coercion, while `dtype=float` converts `None` to `nan`
and carries on:

```python
np.isfinite(np.asarray(obj_col, dtype=None))   # TypeError
np.asarray(obj_col, dtype=float)               # [100.  nan 200.]
```

So they are unpinned **defensive guards**, in the same family as the ensemble
ones below: low severity (a crash on bad input, not a wrong number), but real,
and deletable in a refactor with CI green.

Recorded because the first pass through this class classified it as
"equivalent noise" and only a direct probe caught it. Shape-based triage is a
prioritisation aid, not a verdict.

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

### Priority: two clusters worth fixing, one worth ignoring

**`models/ensemble.py` — highest value.** ADR-004 weights feed every served
forecast, and three guards there are unpinned. Each maps to a *reachable*
crash, not a hypothetical one — the real code handles all three correctly and
nothing asserts that it does:

| mutation | consequence if the guard were weakened |
|---|---|
| `v > 0` → `v >= 0` | a MAPE of exactly 0 is admitted → `ZeroDivisionError` |
| `and np.isfinite(v)` → `or` | a non-finite MAPE is admitted → `ZeroDivisionError` |
| `weights.get(k, 0)` → `weights.get(k)` | a model missing from `weights` → `TypeError` |

Reachability is not theoretical: `compute_mape` returns `inf` for all-zero
actuals, and TIDC publishes zeros (STATUS.md). An infinite MAPE reaching
`compute_ensemble_weights` is a documented scenario in this system.

**`data/quality.py::coerce_demand_artifacts` (40) and
`models/rolling_eval.py::verdict` (32) — next.** Both are live decision code,
and both have already yielded one confirmed real gap each.

**`simulation/scenario_engine.py` (36) — ignore for now.** Nothing in
`components/`, `jobs/`, `api.py` or `app.py` imports `simulation`. The live
Scenarios feature runs a heuristic; issue #127 tracks replacing it with this
engine. So these are real test gaps in code that **does not run in
production** — zero operational risk today, and worth re-adjudicating if #127
lands. A raw mutation score cannot tell you this; it is why the score is not
a gate.

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
   adjudicating all 457 practical rather than aspirational.

   Re-derive the list if tests start spawning processes elsewhere. Running the
   verifier without `--fast` is always the unconditional check.

2. **Equivalent mutants are counted as survivors.** No tool can identify them
   automatically. They are a permanent floor under 100%, which is one reason
   there is no target score.

3. **Unit tests only.** Integration and e2e tests are not in the selection —
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
   theoretical.
2. ✅ **Resolved.** The false-survivor rate is **measured at 2/457 (0.4%)**,
   the blind spot behind it is enumerated (limitation 1), and the deselected
   slow test that was limitation 2 is gone — re-baselining after it landed
   changed **no** verdict, so it had been killing zero mutants.
3. ✅ **Resolved.** All 457 logic survivors are machine-verified, clustered by
   function, and grouped by mutation shape above.

Only (1) is outstanding, and it needs four weeks of scheduled runs rather than
any decision. **When it clears, the honest threshold is per-module logic score
with a no-regression rule** — "this module's logic score may not fall" — rather
than an absolute bar. An absolute bar would have to be set below
`models/ensemble.py`'s 61.6%, which would make it unable to fire anywhere else.

A threshold picked before these are known would be picked from nothing, fire on
noise, and get switched off within a month. The failure mode of mutation
testing in practice is not a low score; it is abandonment.

When it does become a gate, the unit is the **logic score per module**, not the
overall number — a 900-mutant module would otherwise drown a 133-mutant one.
