# Scenario grid — real physics for the what-if simulator (#127)

The scenario simulator shipped in [#119](https://github.com/kristenmartino/gridpulse/pull/119)
with an analytical heuristic:

```python
demand_factor = 1.0 + (temp_delta / 5.0) * 0.025 + solar_delta * 0.00015 + wind_delta * 0.0005
```

Three coefficients, order-of-magnitude-defensible against load-research norms,
applied as a single scalar to the whole 24h curve. The real physics sat in
`simulation/scenario_engine.py`, which **nothing outside tests imported**.

This document records why it stayed that way for so long, what actually
unblocked it, and the one correctness trap that shaped the design.

---

## Why it was parked, and why the stated reason was wrong

Issue #127 parked the work on a cost argument: running the ensemble per
slider drag would put model inference in the web-tier request path, which
[`CLAUDE.md`](../CLAUDE.md)'s I/O guardrail forbids. That part was right, and
it still is.

Its recommended alternative — Approach B, precompute a grid in the scoring
job — was priced at **~200 ms per ensemble re-run**, giving 9×5×5 × 51
regions ≈ **40 minutes** added to the job.

Both numbers were wrong, in opposite directions, and the second error is the
one that mattered.

**The per-run cost was far worse than 200 ms.** The measured production
figure ([`CANONICAL_FACTS.md`](CANONICAL_FACTS.md)) is **763 ms/BA** for
XGBoost's 384-step recursive loop alone, and ~11 s/BA including Prophet's
gap+720 predict. So the honest version of the issue's own plan was hours, not
40 minutes — and 40 minutes was already fatal, because the hourly job runs
under an **1800 s task timeout** (`SCORING_TASK_TIMEOUT_S`) that it has
actually hit: two ticks were SIGKILLed on 2026-08-04.

**The horizon was never questioned.** The issue assumed a full-horizon
re-forecast. But the simulator charts **24 hours** — `_scenario_demand_factor`
says so in its own docstring, "a multiplicative factor to apply to a baseline
24h forecast". Twenty-four recursive steps instead of 384 is **~16× cheaper**,
and that single observation is what moved this from impossible to routine.

| grid | variants | added worker time | added wall (7.65× parallel) | p95 | $/mo |
|---|---:|---:|---:|---:|---:|
| 9×5×5 (the issue's) | 225 | ~551 s | ~72 s | 560 s | $6.10 |
| **9×3×3 (shipped)** | **81** | **~198 s** | **~26 s** | **514 s** | **$2.20** |
| 5×3×3 | 45 | ~110 s | ~14 s | 502 s | $1.19 |

Costs are marginal, not average: at a 406 s median the job has **already
consumed the monthly Cloud Run free tier**, so every added second bills at
full rate. Scaling the measured $26.06/mo proportionally would understate it
by a third.

**The money was never the constraint** — even the issue's full grid is 20
cents a day. What is scarce is **p95 headroom against
[#171](https://github.com/kristenmartino/gridpulse/issues/171)'s 600 s
criterion**, and the extra 144 variants of a 9×5×5 grid would have spent more
than half of what remains to buy interpolation accuracy finer than a slider
step.

---

## The correctness trap: two inference paths, one ratio

This is the part worth carrying to other work.

`simulation/scenario_engine._run_ensemble` calls `predict_xgboost` directly —
a plain vectorised predict over whatever autoregressive features are already
in the frame. Production scoring calls
`data.feature_engineering.recursive_autoregressive_forecast`, which chains
each hour's prediction into the next hour's lag features and whose docstring
calls it *"the single source of truth for both production scoring and holdout
evaluation"*.

Those two paths **disagree on identical weather, by construction.** So wiring
the engine up as written would have produced:

```
scenario (vectorised path)  ÷  baseline (recursive path)
```

and reported the difference between the *paths* as the response to
*weather*. The simulator would have looked more sophisticated and been less
truthful than the heuristic it replaced.

This is the third instance of the same bug class in this repo:

- [#437](https://github.com/kristenmartino/gridpulse/pull/437) — the backtest
  carried its own copy of the recursive loop, in the code that publishes
  holdout MAPE
- [#444](https://github.com/kristenmartino/gridpulse/pull/444) — training and
  scoring each decided ensemble weighting independently
- **#127 (this work)** — the scenario engine is a third forecaster

The fix is the same shape each time: **the module takes the forecaster as a
parameter and the caller passes the production one.**
`simulation/scenario_grid.build_scenario_grid` has no idea how to forecast;
`jobs/phases._write_scenario_grid` binds
`_predict_xgboost_with_recursive_autoregressive`, seeded from the same
`featured` frame and run over the same `future_df` as the baseline. Only the
weather differs between the two sides of the ratio.

`tests/unit/test_scenario_grid_serving.py::test_the_grid_runs_the_same_recursive_path_as_the_baseline`
pins it at the seam.

### What the engine does contribute

The half of `scenario_engine.py` worth keeping is the feature work — copy the
frame, offset the drivers, recompute everything downstream of them (CDD/HDD,
wind power, solar capacity factor, temp×hour). That is exposed as
`apply_weather_deltas()`, and it is genuinely load-bearing: a scenario that
shifted `temperature_2m` and left `cooling_degree_days` alone would hand the
model a contradiction — hot weather with no cooling load — and the forecast
would barely move.

`apply_weather_deltas` is **relative** where `simulate_scenario` is absolute.
That distinction matters: setting `temperature_2m` to a constant, which is
what an absolute override does, erases the diurnal cycle that gives the demand
response its shape.

---

## Design

**Axes** (`config.SCENARIO_GRID_*`) span the slider domains in
`components/tab_demand_outlook.py::_scenario_slider` **exactly**, so every
reachable slider position interpolates and none extrapolates:

| axis | points | values |
|---|---:|---|
| temperature | 9 | −20 … +20 °F, step 5 |
| wind | 3 | −10, 0, +10 mph |
| solar | 3 | −200, 0, +200 W/m² |

Temperature gets the fine axis because **CDD/HDD are piecewise-linear in
temperature with a kink at 65 °F** — precisely where a coarse grid, or the
linear heuristic this replaces, is most wrong. Wind and solar enter through
smooth monotone transforms, so endpoints plus zero carry them.

**Payload** — `gridpulse:scenario_grid:{region}`, factors indexed
`[temp][wind][solar]`, each a 24-length ratio curve against the baseline.
Ratios rather than absolute MW so the web tier can apply them to whichever
baseline it reads, which may be a tick older than the grid. ~15 KB/region.

**The origin cell is defined, not computed.** (0, 0, 0) is the baseline by
definition; re-running the forecaster there would spend a cell reproducing a
row of 1.0s, and any drift it showed would be nondeterminism rendered as a
weather response to moving no slider.

**Serving** — `_scenario_factors()` reads Redis, interpolates trilinearly,
and falls back to the #119 heuristic on a cold cache, a malformed payload, a
Redis failure, or the flag being off. The factor now varies **by hour**,
which the single scalar could not express: a +15 °F afternoon and a +15 °F
4 a.m. are not the same event.

**Failure posture** — the grid is computed *after* the forecast has already
been persisted, and every failure path returns `False` without touching it.
A region with no grid gets the heuristic. The failure mode is a degraded
simulator, never a missing forecast (the #268 → #267 rule).

---

## Status and what is not yet verified

Shipped behind `FEATURE_FLAGS["scenario_grid"]`. **Enabled 2026-08-10 (#460)**
on the scoring job at image `46b7fb8`.

Enabled rather than held back because the cost cannot be confirmed any other
way — there are no trained models on a dev box or in CI, so every estimate of
this phase is arithmetic until a real tick runs it. The estimate's known blind
spot is per-call fixed overhead in the recursive helper: the grid makes 4,080
short calls where production makes 51 long ones, and if that overhead
dominates, the true figure is several times ~26 s.

Note the flag is **not** an env-var override. `feature_enabled` reads
`FEATURE_FLAGS` directly, so both flipping it and rolling it back are a code
change plus a redeploy — budget a CI cycle, not a config push.

Verified: 39 unit tests over the grid maths, the axes/slider contract, the
interpolation, both integration seams, and the fail-open behaviour.

**Not yet verified, and it needs production to verify:** the end-to-end
physics claim. Issue #127's acceptance criterion asks for a test asserting
that a +25 °F Heat Dome scenario produces uplift consistent with the trained
model's real sensitivity. That cannot be asserted from unit tests — there are
no trained models on a dev box or in CI, and a synthetic booster would only
re-measure the stub. It needs one scoring run with the flag on, against real
persisted models.

So the honest claim today is: **the plumbing is correct and the physics are
wired to the right forecaster.** Whether the resulting sensitivities are
*better than the heuristic's* is measurable only once the flag is on, and the
first thing to check when it is:

1. Compare grid factors against `_scenario_demand_factor` at the grid points
   for a summer-peaking BA (FPL/ERCOT) and a winter-peaking one (SPA/NWMT).
   The heuristic is BA-independent; the grid should not be.
2. Confirm the added wall time lands near the costed ~26 s, and that the
   runtime-creep alert (70% of task timeout) stays quiet.
3. Confirm the 65 °F kink is visible — the temperature response should not be
   a straight line through zero.
