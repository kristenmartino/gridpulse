# #559 — the positional AR seed: real, live, and not measurably an accuracy defect

**Run 2026-08-18.** Replay through the production serve path against archived GCS
model vintages. No retrain.

## Summary

| question | answer |
|---|---|
| Is `demand_lag_168h` reading the wrong hour in production? | **Yes.** Verified live: 34 hours off on LGEE at today's origin. |
| Does fixing it change forecast values? | **Yes.** 2.1–2.7% of demand on affected BAs, max 12.8%. |
| Does fixing it improve accuracy? | **Not established.** Inconclusive at 168h *and* 48h; the effect is inside the harness's MDE. |
| Does it need a retrain? | **No.** The training features were never wrong. |

The honest reading: this is a **correctness** defect with a measurable effect on
output and **no demonstrated accuracy benefit** to fixing it. That is a real
outcome, not a failed study — `docs/EVALUATION_POLICY.md` treats "inconclusive"
as publishable, and it is the right answer here.

## 1. What is actually broken

Not what [#559](https://github.com/kristenmartino/gridpulse/issues/559) says, and
not quite what [PR #578](https://github.com/kristenmartino/gridpulse/pull/578)
says either.

The **demand** frame is a complete hourly grid on 50 of 51 BAs — EIA reports gap
hours as rows with null values, and `_parse_demand_records` never manufactures
rows. So `shift(24)` in `add_autoregressive_demand_features` is temporally exact,
the training features are correct, and there is nothing to reindex. (#578's
census, reproduced independently here: 7 absent rows, all SPA, all May; 78 null
rows.)

But `engineer_features` then drops every row whose lag source was null, and **that
is what punches holes into the frame the serve path seeds from**. On LGEE:

```
featured rows 2171 -> 1969   (dropped_rows=202)
   jump of 19h ending at 2026-08-13 09:00:00+00:00
   jump of 17h ending at 2026-08-14 06:00:00+00:00
   34 missing hours inside the featured span
```

`jobs/phases.py:1289` seeds the recursion with `featured["demand_mw"].tolist()`,
and `compute_autoregressive_snapshot` indexes that list **positionally**. At the
origin production resolves right now:

```
LIVE origin 2026-08-18 11:00:00+00:00
   lag_24h:  reads 2026-08-17 11:00   want 2026-08-17 11:00   err  +0h
   lag_168h: reads 2026-08-10 01:00   want 2026-08-11 11:00   err +34h
   demand_roll_168h_* spans 201 real hours, not 167
```

Because `dropna` deletes rows at +1/+3/+24/+168 from each null hour, **one null
hour corrupts `lag_24h` for 24 subsequent origins and `lag_168h` for 168** — a
seven-day blast radius, on every tick, not only the ticks whose origin stalls.

Fleet-wide, 7 BAs carry a corrupt origin and 44 carry none.

## 2. Method

Two arms, one difference. Same archived vintage (the one live at that origin),
same weather, same origins, same future frame:

- **control** — `recursive_autoregressive_forecast`, production, positional
- **treatment** — the same loop with the history keyed by timestamp; `lag_k`
  resolves to `origin − k hours` or NaN

`scripts/positional_lag_value_study.py`. Origins are restricted to those where the
two arms actually differ, spaced at least one horizon apart, requiring settled
truth across the window. Verdicts route through `models/rolling_eval.py`.

**Null control.** Ungapped BAs must produce **byte-identical** arms. MISO and PJM
both returned `0.000000000%` divergence across 47 origins. The harness is sound.

## 3. Results

Positive Δ = treatment better.

| horizon | pooled n | mean Δ WAPE | median | stderr | MDE | verdict |
|---|---:|---:|---:|---:|---:|---|
| 168h | 24 | +0.090 | +0.126 | 0.233 | 0.466 | **inconclusive** |
| 48h | 85 | +0.181 | +0.141 | 0.203 | 0.406 | **inconclusive** |

Both means sit well inside the minimum detectable effect. `verdict()` declined to
decide in both cases, for the same stated reason: *within window-to-window noise*.

Per BA, with each BA's own MDE:

| BA | n (48h) | Δ WAPE | MDE | detectable? | divergence |
|---|---:|---:|---:|:--|---:|
| **TIDC** | 23 | **+1.296** | 0.640 | **yes** | 2.56% |
| IID | 12 | −1.102 | 1.568 | no | 3.29% |
| NEVP | 7 | +0.468 | 0.554 | no | 1.15% |
| NWMT | 10 | +0.457 | 0.817 | no | 1.85% |
| PSCO | 24 | −0.355 | 0.777 | no | 3.20% |
| SPA | 7 | +0.058 | 0.311 | no | 2.94% |
| LGEE | 2 | −0.456 | — | no | 2.44% |
| MISO, PJM | 37 each | +0.000 | — | — | **0.000%** |

**Only TIDC clears its own MDE**, and it does so at both horizons (+0.779 at
168h, MDE 0.458). TIDC is also the BA with the most frequent scattered nulls.
That is suggestive — but it is a **post-hoc per-BA look at an inconclusive pooled
result, and is hypothesis-generating only**. It is not a finding, and it must not
be used to justify shipping on its own.

## 4. Limits

1. **Not pre-registered.** Arms, metric, verdict route and null control were fixed
   before running; the per-BA cut in §3 was not. Treat the pooled result as the
   study and the TIDC line as a hypothesis.
2. **Underpowered by construction.** Detecting the observed +0.18 would need on
   the order of 600 non-overlapping windows. Only 26 exist at 168h, because the
   defect requires a gap and gaps are rare. This study cannot be made decisive by
   running it harder.
3. **LGEE, the worked example, is nearly unscoreable** — its holes are recent
   enough that settled truth does not yet extend a full horizon past them.
4. **Absent-hour policy is a modelling choice the treatment arm makes implicitly.**
   A lag whose hour is missing returns NaN and is then zero-filled by the shared
   `row.fillna(0)`. Sampled at step 0, this fired once in 97 origins — the
   recursion fills forward, so it is rare — but it is unmeasured deeper into the
   horizon and a production fix must decide this explicitly rather than inherit it.
5. Replay weather is the archived mirror, not what production held at the time;
   both arms share it, so it is common-mode.
6. The 48h satisficing check failed on treatment bias (−2.47%). Control-arm bias
   was not computed, so this is **not** attributable to the treatment — the #451
   harness has a known bias floor in this regime.

## 5. What follows

The fix cannot be argued as an accuracy improvement — the measurement does not
support that, at either horizon. It can be argued as a **correctness** fix: a
feature named and documented as "demand 168 hours ago" demonstrably is not, the
error is live today, and no existing test can see it.

Against shipping: it moves forecast values 2–3% on seven BAs for no measured
gain, and two BAs move the wrong way (inside noise, but still).

The narrower change in PR #578 — teaching `_resolve_forecast_start`'s cap to ask
whether AR context can be seeded — fixes the origin stall without touching any
feature value, and is orthogonal to this. It should land regardless.

## 6. What shipped

`temporal_ar_seed`, registered in `config.FEATURE_FLAGS`, **default off**.

- `HourIndexedHistory` stores demand densely by hour, NaN where we have none —
  **the same shape the training path already works in**, which is what makes the
  two paths mean the same thing.
- `compute_temporal_autoregressive_snapshot` resolves each lag to `now - k
  hours`, returning NaN when that hour is absent rather than reaching further.
- Both recursion helpers take an optional `seed_timestamps` and use it only when
  the flag is on. Fail-open at every seam: no timestamps, no `timestamp` column,
  or a length mismatch all fall through to the positional path **byte-identical**
  — pinned by `test_flag_off_ignores_seed_timestamps_entirely`.
- The batched scenario-grid helper moves with the single-frame one, because
  `test_scenario_grid_batching.py` pins them equal.

Verified end-to-end: with the flag on, production's
`recursive_autoregressive_forecast` reproduces this study's independently written
treatment arm to **max abs diff 0.0000000000** on TIDC, and flag-off vs flag-on
diverges 1.269% — consistent with §3. So the measurement and the implementation
are the same thing.

Default off is the honest setting for an inconclusive result. Turning it on is a
decision that wants a shadow run, not this study.

### Cost

The first implementation keyed the history by timestamp in a `dict`, which
rebuilt a list of up to 168 lookups per rolling window per step: **79x** the
positional snapshot, **+74.9s per scoring tick** fleet-wide, +4.6% of a forecast
phase on a job that has SIGKILLed at its 1800s timeout (#389). That would have
blocked ever flipping the flag on, and it was not measured before it shipped.

The dense hour-indexed array replaced it. A window is a slice, so it is
**faster than the positional path it replaces**:

| per BA, 384 recursive steps | snapshot cost | full loop, fleet |
|---|---:|---:|
| positional list (flag off) | 17.8 ms | — |
| temporal dict (first cut) | 1418.5 ms (79.8x) | +74.9s |
| **temporal array (shipped)** | **9.7 ms (0.5x)** | **+0.1s** |

Forecasts are unchanged by the swap: the array implementation reproduces this
study's independent treatment arm to **0.0000000000** on TIDC, PSCO and IID, so
every number in §3 still describes what ships.

## 7. The shadow run — what it is, and what it cannot be

The flag is off because §3 was inconclusive. The obvious next move is a
production shadow: compute both arms live, serve the control, grade later.
**That will not settle the accuracy question, and it is worth being precise
about why before anyone waits on it.**

The defect only produces an observation when a gap occurs, and gaps are rare.
Extrapolating §3's own accrual:

| unit | observed | accrual | needed for a verdict | wait |
|---|---:|---:|---:|---:|
| 168h windows | 24 in 90 days | ~1.9/wk | ~643 | **6.6 years** |
| 48h windows | 85 in 90 days | ~6.6/wk | ~428 | **1.2 years** |

More production time does not fix this; the effect is small relative to
window-to-window variance and the sampling rate is set by how often EIA drops an
hour. So the shadow is built as a **pre-rollout safety instrument**, and
`scripts/seed_shadow_eval.py` prints its own MDE and the implied wait next to
every comparison so it cannot be mistaken for a verdict.

What it does answer, and what no offline replay can:

- does the temporal path run clean against **real production frames**, not
  mirrors;
- what the second recursion actually costs in the live job;
- whether live divergence matches the 2.1-2.7% §3 predicted — a large gap would
  mean production frames differ from the mirrors the replay used;
- whether the gate is still deciding.

**Gating.** A second XGBoost recursion is not free, so it runs only where the
arms *can* differ. `positional_seed_matches_hours` is the exact condition, not a
proxy: every lag and rolling window reaches at most 168 entries back, and the
recursion appends its own predictions contiguously, so the arms are identical
whenever the last 168 seed entries are contiguous hours ending at `origin - 1h`.
On 2026-08-20 that gated in **3 of 51** BAs (~3 CPU-seconds); ungated it would
be roughly +380 CPU-s on a job whose worst recent tick used 1155s of 1800s.

Membership is recomputed every tick and moves quickly — LGEE alone on 08-18,
LGEE/SPA/TIDC on 08-20 — so a static allowlist would already be stale.

**Gated is not bounded.** The gate is data-dependent, so a bad EIA day could
admit the whole fleet. `_write_seed_shadow` runs after the served payload
persists and cannot lose its own BA's forecast — but work-shedding is whole-BA,
so an unbounded enrichment would push the run past the soft deadline and buy
shadow data with *later* regions' real forecasts. Hard-capped at
`SEED_SHADOW_MAX_REGIONS_PER_TICK` (12, about 4x the observed population),
counted per process exactly like `_EIACircuitBreaker`, with a declined tick
recorded rather than silent — a dropped observation that read as "no gap" would
bias the sample toward quiet ticks.

**The gate audits itself.** One region per hour that the gate says is identical
is shadowed anyway, asserting zero divergence. Without it a gate that quietly
started skipping everything would be indistinguishable from a fleet with no
gaps — the failure mode this very PR found in the parity fixture. A nonzero
audit divergence is an alarm about the *gate*, not a finding about the seed.

Verified end to end on real LGEE data (origin `2026-08-20T12:00Z`): the gate
said diverges, the second arm ran, and divergence was **2.82%** — consistent
with §3, and slightly higher because LGEE was +50h out that day against +34h at
study time.

**Where the accuracy verdict comes from instead:** injecting synthetic gaps that
match the observed length distribution, offline, through the same serve path.
The corruption is deterministic — a hole is a hole — so an injected one produces
the same class of error, and `n` stops being set by EIA's outage schedule.

## 8. Why #186 is not "unify the implementations"

[#186](https://github.com/kristenmartino/gridpulse/issues/186) asks for a single
shared core, on the reasoning that the training and inference paths "match today
only by coincidence." Two corrections.

**They do not match.** Not on gapped frames, and not for as long as gaps have
existed. #186 is a divergence-*repair* issue, not a divergence-prevention one.

**And a shared core is the expensive way to fix it.** Measured on a real 90-day
frame (2171 rows):

| unification | cost |
|---|---|
| training calls the per-row core | **611x** the vectorised path — +372s per fleet training run, roughly double once the holdout replays it |
| inference calls the pandas core on a trailing frame | **60x** the positional snapshot — +54s per scoring tick fleet-wide |

Both buy a structural guarantee that a property test provides for free, against
CLAUDE.md's #389 rule to bound what one run can cost.

What actually delivers #186's intent is cheaper and already here: the two paths
now share a **representation** (a continuous hourly grid with NaN holes) and a
**semantic** (a lag is an hour, an absent hour is NaN), and
`TestParityProperty` fuzzes randomised gap patterns to assert they agree —
including a companion test asserting the *positional* path **fails** that same
property, so the fuzz cannot quietly stop generating gaps that matter.

Three implementations remain only because the flag needs both inference paths
alive. Whichever way the flag decision lands, one of them is deleted, and the
count goes to two: bulk-vectorised for training, incremental for inference,
which is an inherent difference in shape rather than duplicated logic.

**Also worth fixing regardless of the above:**
`test_training_features_match_inference_snapshot_row_by_row`
(`tests/unit/test_feature_engineering.py:651`) compares the two AR
implementations on a **gapless** fixture, where they agree by construction. It is
the guard that should have caught this and cannot. Logged to `MISTAKES.md`, and
`tests/unit/test_temporal_ar_seed.py` now carries the gapped-fixture version —
including a characterisation test that pins the defect itself, and one that
demonstrates *why* a gapless fixture is blind to it.

A note on that: the first draft of the gapped fixture put its hole **outside** the
168-hour lookback, where positional and temporal agree — the same trap, one level
up, and it took a failing assertion to notice.
