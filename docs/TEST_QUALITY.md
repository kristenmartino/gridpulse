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
| `models/rolling_eval.py` | 322 | 215 | 72 | 35 | 66.8% | **74.9%** |
| `models/skill.py` | 192 | 130 | 49 | 13 | 67.7% | **72.6%** |
| `data/quality.py` | 188 | 125 | 51 | 12 | 66.5% | **71.0%** |
| `models/ensemble.py` | 133 | 53 | 33 | 47 | 39.8% | **61.6%** |
| **overall** | **2,349** | **1,627** | **463** | **259** | **69.3%** | **77.8%** |

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

**463 logic survivors exist. Three have been hand-verified.** The rest are
recorded but unadjudicated — do not read the table below as "and the other 460
are fine." Adjudication is incremental; this is where it accumulates.

Verification protocol: apply the mutation to the working copy **one at a time**,
run the whole unit suite, revert. One at a time matters — two mutations applied
together produced a failure that neither caused alone, which would have
mis-adjudicated both.

| # | site | mutation | verdict |
|---|---|---|---|
| 1 | `models/rolling_eval.py` `satisficing_check` | `regression > max_mape_regression_pts` → `>=` | **REAL GAP** — 2,687 unit tests pass |
| 2 | `data/quality.py` `coerce_demand_artifacts` | `continue` → `break` on an absent reading | **REAL GAP** — 2,687 unit tests pass |
| 3 | `models/ensemble.py` `ensemble_combine` | `len(set(lengths)) > 1` → `>= 1` | **FALSE SURVIVOR** — actually killed (see limitation 1) |

### 1. The satisficing constraint boundaries are not pinned

[`EVALUATION_POLICY.md`](EVALUATION_POLICY.md) states the constraints that veto
an A/B win: `|bias| ≤ 2%`, `MAPE regression ≤ 0.5 pts`. Both boundary
comparisons in [`satisficing_check`](../models/rolling_eval.py) can be flipped
to `>=` with the entire unit suite green. A treatment landing *exactly* on the
threshold is decided by an untested branch.

The same class of survivor appears throughout
[`verdict`](../models/rolling_eval.py) — `consistency < min_sign_consistency`
→ `<=`, and `winner = "treatment" if mean > 0` → `>=` (an exactly-zero mean
picks a winner either way, unnoticed).

This matters more here than the raw numbers suggest: CLAUDE.md makes
`rolling_eval.verdict()` the mandatory route for *every* model change. It is
the most load-bearing decision function in the repo and its own boundaries are
the least pinned part of it.

### 2. The demand-artifact guard's skip is not pinned

In [`coerce_demand_artifacts`](../data/quality.py), the trailing-hours loop
`continue`s past an absent reading. Changed to `break`, the loop stops at the
first gap and every later artifact goes uncoerced — and no test notices. This
is the #309 guard that keeps implausible EIA partials out of the forecast
anchor, so a silent early exit is exactly the failure it exists to prevent.

### 3. Why #3 was false — and why that is worth writing down

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
   Affects at least `tests/unit/test_stable_hash_reproducibility.py`.

2. **One test is deselected.**
   `test_callbacks_helpers.py::TestRunForecastOutlook::test_sqlite_cache_hit`
   takes **36.8s** standalone — 76% of the whole 2,690-test unit suite's
   runtime, ~121s under instrumentation. It touches
   `data/feature_engineering.py` incidentally, so mutmut re-ran it for every one
   of that module's 904 mutants, which does not finish. It is deselected in
   `[tool.mutmut] pytest_add_cli_args`. Consequence: any feature-engineering
   mutant only that test would kill is reported as a survivor. Tracked
   separately as its own fix; when it is fast, delete the deselect and this note.

3. **Equivalent mutants are counted as survivors.** No tool can identify them
   automatically. They are a permanent floor under 100%, which is one reason
   there is no target score.

4. **Unit tests only.** Integration and e2e tests are not in the selection —
   they need infrastructure the `mutants/` tree does not have.

5. **Threading is pinned to one thread per pool.** `scripts/mutation_test.py`
   sets `OMP_NUM_THREADS=1` and friends before invoking mutmut. Without it,
   forking a parent with a live threaded BLAS pool deadlocks the workers — they
   sit at 0% CPU indefinitely and mutmut's wall-clock timeout never fires
   because nothing is running.

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

1. The baseline is stable across at least four consecutive weekly runs (no
   unexplained swings from test ordering or flakiness).
2. Limitations 1 and 2 above are resolved or quantified, so the false-survivor
   rate is known rather than guessed.
3. Enough of the 463 logic survivors are adjudicated to know what fraction are
   equivalent mutants — which is what sets an achievable threshold.

A threshold picked before those are known would be picked from nothing, fire on
noise, and get switched off within a month. The failure mode of mutation
testing in practice is not a low score; it is abandonment.

When it does become a gate, the unit is the **logic score per module**, not the
overall number — a 900-mutant module would otherwise drown a 133-mutant one.
