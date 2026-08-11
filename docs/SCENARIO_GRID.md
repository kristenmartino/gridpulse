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

## MEASURED 2026-08-10: the estimate was wrong by ~28x, and the flag is back off

Enabled in #460, measured on the first flag-on tick, reverted the same hour.

| | estimate | measured |
|---|---:|---:|
| per grid cell | ~48 ms | **~1.4 s** |
| per region | ~4 s | **~113 s** |
| whole tick | ~26 s added | **+714 s** (1,139 s vs a 411/439/411/451 s baseline) |

Per-region elapsed came back at **168.9 s and 169.9 s** against a ~55.6 s
baseline. The tick succeeded — 51/51, no shed, no kill — and that is precisely
the trap: it clears the 1800 s timeout by 661 s **on a quiet upstream**, and the
2026-08-04 EIA outage alone cost ~800 s. The next degraded tick with this on is
a SIGKILL.

**Why the estimate was wrong.** The 48 ms/cell figure came from scaling a
384-step production measurement down to 24 steps. That assumes cost is linear
in steps and per-call setup is free. It is not:
`recursive_autoregressive_forecast` pays a fixed seed/snapshot cost **per
call**, and the grid makes 4,080 short calls where production makes 51 long
ones. Fixed overhead, not step count, is the entire bill — which is why 24
steps did not cost 1/16 of 384.

**The fix was to batch the cells, not shrink the grid** (#464). The 80 variants
differ only in weather, so their step-*i* rows can travel through the model
together: one `predict` call per step instead of one per step *per cell*, which
is **1,920 single-row predicts per region down to 24**. Per-cell chaining is
unchanged — each variant still appends only its own predictions to its own
history — and parity with the single-frame SSOT is a differential test that
compares both the outputs and the exact frames handed to the model.

Two measurements worth keeping from that work:

- **A free stub cannot measure batching.** The first benchmark used a
  zero-cost `predict` and reported the batched path as 1.7x *slower*, because
  with the model free the only thing left to measure is pandas. Re-run with a
  realistic ~0.85 ms per-call cost it is **11x faster**. The per-call cost is
  the entire thing batching removes, so a benchmark that omits it measures the
  opposite of the question.
- **Fancy-indexing ate the win.** Gathering each step's 80 rows with
  `.iloc[[...]]` was slower than 80 separate one-row slices. Stacking
  step-major so a step's rows are contiguous — plus filling once up front
  rather than per step — is what made the batched path faster in absolute
  terms, not just in call count.

**Do not re-enable by shrinking the grid.** At ~1.4 s/cell even 5x3x3 is ~63 s
per region, still a ~2x tick. The fix has to make a *cell* cheap — hoist the
per-call setup out of the loop, or push all 80 variants through one seeded
pass — and the re-measure has to be a real tick, because this is exactly the
number that cannot be obtained offline.

**What this validated.** The fail-open design held: the grid runs after the
forecast is persisted, and the 1,139 s tick still wrote 51/51 forecasts. The
cost was runtime, never correctness.

## MEASURED 2026-08-10 (second attempt): 506s, affordable

After #465 batched the cells, the first flag-on tick came in at **506s**
(23:00:09 -> 23:08:35, 51/51) against a 411/439/411/451/427/442 s flag-off
baseline. **1.18x, +79s** — against 1,139s and 2.7x for the cell-at-a-time
version. Clears #171's 600s criterion.

**Updated 2026-08-11 — now n=3: 506s, 461s, 515s** (23:00, 00:00, 01:00
ticks), mean 494s, all inside #171's 600s criterion. The n=1 caveat below is
retained because it was the right caution at the time, and because 461-515s is
a ~54s spread on three observations — a median over a full day is still the
number to quote if this ever needs defending.

The original caveats, kept: 506s **overshot the ~495s projection**, the third
time in this line of work a cost estimate came in optimistic, so weight the
measurement and discount the estimates. And this file's own scoring-runtime row
records a 370.7s reading that a 17-tick median later revised to 406s — small
samples of this job have misled before.

Headroom is the real question rather than the criterion: 506s leaves 1,294s to
the hard timeout, and the 2026-08-04 EIA outage added ~800s on its own. A
degraded tick would land near 1,300s — past the 1260s creep alert, under the
kill. It survives the historical worst case with much less margin than 427s
had.

## What the 506s tick did NOT prove

It proved the grid gets **written** affordably. It proved nothing about the
grid being **read**, and driving the deployed UI immediately turned up two
reasons that mattered (#471):

- **The panel rendered empty.** `active_tab` was a `State` on the scenario
  callback rather than an `Input`, so switching to the Forecast tab never
  re-fired it. Arriving with the panel already open — a bookmark, a reload,
  opening the panel from another tab — left the KPI row blank until the user
  moved a slider.
- **The copy under it said "not a model re-forecast"** while production served
  exactly that. Static text cannot track a feature flag, and `_scenario_factors`
  had reported the source since #458 with nothing rendering it.

Both are fixed in #471, along with `GET /api/v1/scenario/{region}` — gated on
the `scenario_grid` flag and 404 when it is off, so the public surface appears
only while the data behind it does — which
exists so the physics checks below go through `interpolate_scenario_factors`,
the same helper the web tier uses. Reading Redis directly would confirm the
payload exists and skip the serving path, which is the half that was untested.

## The physics checks, run 2026-08-11

Both were run against `/api/v1/scenario/{region}` on the live grid.

**Check 1 — BA-dependence: PASSED, decisively.** At +20 F the #119 heuristic
returns **+10.0% for every BA by construction** (it takes only the three
deltas). The grid returns:

| BA | mean factor at +20 F |
|---|---:|
| ERCOT | 1.1007 |
| NWMT | 1.0830 |
| ISONE | 1.0506 |
| FPL | 1.0417 |
| **SPA** | **0.9677** |

A 13.3-point spread, and **SPA's demand falls as it warms** — the
winter-peaking signature a single positive coefficient cannot produce at any
parameter value. The response also varies BY HOUR within each BA, which a
scalar factor cannot express at all.

**Check 2 — the 65 F kink: the check was badly specified, and the answer is
more interesting than the question.** The axis is a DELTA from each BA's own
baseline weather, not an absolute temperature, so a kink at absolute 65 F
lands at a different delta for every BA and cannot be read off this axis. What
the walk did show is saturation:

| BA | -20 | -10 | +5 | +10 | +20 |
|---|---:|---:|---:|---:|---:|
| FPL | 0.9790 | **0.9786** | 1.0228 | **1.0264** | **1.0264** |
| ERCOT | 0.9445 | 0.9502 | 1.0113 | **1.0123** | **1.0123** |
| SPA | 0.9925 | 0.9968 | **0.9498** | **0.9498** | **0.9498** |

Identical to four decimals across 10-15 F spans. **XGBoost is a tree ensemble
and does not extrapolate**: past the training range every split routes the same
way and the response stops depending on the input. FPL saturating on the COLD
side is the tell — Florida has no cold training data, so -10 and -20 land in
the same empty region.

SPA is worse than flat: it saturates above +5 F and then WANDERS, with the mean
turning down through +15 (0.9917) to +20 (0.9677). Outside the envelope a tree
model is unconstrained, not merely constant.

Two hypotheses were tested and rejected on the way: it is not a diverged
recursive forecast (the decline does not compound with hour index) and not a
path-parity offset (the first-hour factor does vary with temperature for FPL,
ERCOT and ISONE).

**The grid is faithfully reporting what the model says. The model has nothing
to say out there.** That is a product-honesty problem, not a defect in the
batching or the serving path — both of which these checks exercised end to end.

## Envelope flags and the origin parity cell (#472)

**Envelope.** Each payload now carries `envelope.{temp_f,wind_mph,solar_wm2}`,
one flag per axis position, computed by comparing the shifted forecast series
against that BA's own observed range in `featured`. A position is in-envelope
only if EVERY hour stays inside. The API returns `extrapolated: true/false` for
the requested position and the panel says so instead of calling it a
re-forecast.

Known limitation, stated rather than papered over: the check is **per-axis, not
per-cell**. A cell can be in-envelope on all three axes and still sit in a
sparse corner the booster never saw jointly.

**The origin is now computed.** It was defined as `1.0` to save one forecast in
81 — and that saved cell was the only thing that could have caught the two
sides of every ratio coming from different inference paths, the exact failure
this module was built to prevent. `origin_drift` is now in the payload and
warns above 0.001.

## The origin cell found a bug on its first run (#474)

Measured 2026-08-11, the first tick that computed the origin instead of
defining it as 1.0:

| BA | origin_drift |
|---|---:|
| FPL | **0.013** |
| CAISO | 0.00473 |
| ISONE | 0.0041 |
| ERCOT | 0.00405 |
| MISO | 0.00114 |
| PJM | 0.00067 |

Five of six above the 0.001 warn threshold. A zero-delta scenario was **not**
reproducing its own baseline, so every ratio in every payload carried a
non-weather component of up to 1.3%.

**Cause.** `compute_temperature_deviation` is a **720-hour rolling mean**, and
`apply_weather_deltas` called `_recompute_derived_features` unconditionally on
the simulator's **24-row** frame. With `min_periods=1` that is an expanding
mean over 24 hours, not a 30-day reference — so identical weather produced a
different feature. Drift tracks how much a BA's temperature swings within a
day, which is why FPL is worst and the large thermally-sluggish BAs (PJM,
MISO) are lowest.

**Fix.** CDD/HDD, wind power, solar CF and temp x hour are pointwise functions
of their drivers and are still recomputed. `temperature_deviation` is now
carried from the input and shifted by the delta: the 30-day reference is
dominated by history the scenario does not touch (24 shifted hours against a
720-hour window), so a uniform shift of d moves the deviation by d. At zero
delta that is exactly a no-op, which is what makes the origin a parity check
rather than a measurement of this bug.

**What survived the contamination.** The BA-dependence result stands: a
13.3-point spread against <=1.3% contamination is roughly 10x the noise, and
SPA's sign flip is not explicable by a 1% offset. The saturation result also
stands, because the offset is common-mode within a payload and identical cells
stay identical. What does NOT stand is the **precision**: factors were quoted
to four decimals off a zero point that was wrong by up to 1.3%. Read the
pre-#474 numbers as ~+4%, not 4.17%.

**CONFIRMED FIXED 2026-08-11.** The first tick after #474 deployed (grids
written 02:02:21-25) returned `origin_drift` of **exactly 0.0 on all six BAs**
— FPL 0.013 -> 0.0, CAISO 0.00473 -> 0.0, ISONE 0.0041 -> 0.0, ERCOT
0.00405 -> 0.0, MISO 0.00114 -> 0.0, PJM 0.00067 -> 0.0. `implausible_cells`
was empty everywhere, so #475's narrowed 0.6-1.7 band does not bind on real
physics either.

Exact zeros matter here rather than merely small ones: a zero-delta scenario
now reproduces its baseline bit for bit, so both sides of every ratio come from
the same place. **That retroactively validates the physics results above** —
the BA-dependence spread and SPA's sign flip are pure weather response with no
path or feature contamination in them, and the precision caveat attached to the
pre-#474 numbers can be lifted for grids written from 02:00 onward.

The diagnosis is confirmed rather than merely consistent: 0.0 was predicted
from a specific mechanism (a 720-hour rolling feature recomputed on a 24-row
frame), and fixing exactly that produced exactly that.

**The generalisable part.** The origin cell was removed on the argument that it
could only ever reproduce a row of 1.0s. The first time it ran, it did not.
A parity check whose expected value is "obviously" a constant is exactly the
check worth keeping, because that is what makes its failure legible.

## The plausibility band (#475)

Was `0.25-4.0`, and silent. Both wrong.

**Too wide.** Measured hourly factors at the slider extremes span ~0.91 to
~1.20 across FPL/ERCOT/SPA/NWMT/ISONE. A band four times wider than that in
both directions can only catch a catastrophe, and this feature's realistic
failure is not a catastrophe — it is a cell that *wanders* outside the
training envelope, which is what SPA does above +5 F. Now `0.6-1.7`: roughly
3x the observed excursion, so it should never bind on real physics.

**Silent was the worse half.** `np.clip` clamped an out-of-band cell to the
edge and stored it as if it were a measurement — a diverged forecast reaching
the simulator as "demand drops 75%", with nothing anywhere saying otherwise.
An out-of-band cell is now treated exactly like a non-finite one: dropped to
the baseline, logged as `scenario_grid_cell_implausible` with its coordinates
and extremes, and listed in the payload as `implausible_cells`.

The distinction that matters: a clamped value is indistinguishable from a real
one at the point of use. A dropped-and-counted one is not.

## Criterion 3 — the physics test (#481)

`tests/unit/test_scenario_physics.py`. The blocker was that CI has no trained
models and "a synthetic booster only re-measures the stub". That holds for a
hand-written stub function; it does NOT hold for a **real XGBoost trained on
data with a deliberately encoded temperature response**, which is what this
does. The test then asserts the whole pipeline — override, derived-feature
recompute, recursive inference, ratio against baseline — recovers a
relationship that genuinely exists in the training data.

Two fixtures: a cooling-driven model with its forecast window in the cooling
regime (FPL in August) and a heating-driven one in the heating regime (SPA in
January).

**Two drafts of this test were wrong, and both failures were informative.**

1. The first placed both windows at the tail of a synthetic seasonal cycle,
   which landed at ~29 F. The "cooling-driven" model then correctly showed
   demand FALLING as it warmed, because **the regime of the forecast window
   decides the sign, not the size of the coefficients**. That is the same
   reason FPL and SPA differ in production, and the fixture now places each
   window deliberately.
2. The second asserted "the grid differs from the heuristic's 1.10 at +20 F"
   and failed at a 0.004 gap — the fixture's cooling model happened to return
   1.0959. **Coincidental agreement on one region is not evidence the
   coefficients are still in use**, so a magnitude threshold measures the
   fixture's tuning rather than the code path. It now asserts what the
   heuristic *cannot* do at any coefficient values: return different answers
   for two regions given the same slider, and return a value below 1.0 for
   warming.

What it buys: the machinery is pinned end to end and a grid that stopped
responding to weather would fail. What it does not: it says nothing about
whether the PRODUCTION boosters are well calibrated — only the live checks
above answer that.

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
