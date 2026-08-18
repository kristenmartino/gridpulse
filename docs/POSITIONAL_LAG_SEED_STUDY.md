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

**Also worth fixing regardless of the above:**
`test_training_features_match_inference_snapshot_row_by_row`
(`tests/unit/test_feature_engineering.py:651`) compares the two AR
implementations on a **gapless** fixture, where they agree by construction. It is
the guard that should have caught this and cannot. Logged to `MISTAKES.md`.
