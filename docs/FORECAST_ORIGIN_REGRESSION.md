# The forecast origin froze, then went backwards — two mechanisms, one line

**Issue:** [#537](https://github.com/kristenmartino/gridpulse/issues/537), the
open half. The drift-side half is settled in
[`docs/DRIFT_LEAD_REGRADE.md`](DRIFT_LEAD_REGRADE.md) (#542) and is not revisited
here.

**All figures measured against production.** Replay and log sweep captured
2026-08-18T09:00Z over the tick window 2026-08-11T12:00Z → 2026-08-18T07:00Z.
Live numbers move hourly.

---

## 1. What was open

A forecast's origin is `forecasts[0]["timestamp"]` — the hour the payload calls
its first prediction. LGEE's, reconstructed from the drift log as
`new_record_ts − lead_hours + 1`:

* **2026-08-13T14:00** — held for 15 ticks while fresh demand kept arriving.
* **2026-08-12T15:00** — then **23 hours OLDER than the vintage already served**,
  held for 24 more, carrying leads out to 63h under a "1-hour-ahead" label.

A monotonic origin should never go backwards. Why it froze, and why it then
regressed, was left open as a scoring-path question.

## 2. Method

The repo's standing method: don't reason about candidate mechanisms — diff two
things that should agree, then look for structure in the disagreement.

* **CARRIED** — the origin the payload actually shipped, from `drift_updated`.
* **COMPUTED** — what `_resolve_forecast_start` returns when replayed against the
  frame that BA actually held at that tick.

`scripts/forecast_origin_replay.py` runs the **real** primitives —
`engineer_features`, `_resolve_forecast_start` — never a reimplementation, and
reads the GCS vintage mirror locally (no Redis, no VPC).

**Reconstructing the per-tick frame.** The mechanism is driven by which hours
were *missing*, not by their values, and `VintageRecord.captured_at` records when
an hour was first seen carrying a positive `D`. So `demand[h]` is NaN at tick `T`
iff `captured_at(h)` falls in a later tick than `T`.

Two things this got wrong on the first pass, both caught by the validation below
rather than by inspection, and both worth stating because either alone
manufactures agreement:

1. `captured_at` is stamped a few minutes *into* the tick that first saw the hour
   (12:04:40 for the 12:00 tick). Comparing instants excludes the very hour that
   tick captured. Compare on the capture's **hour**.
2. A tick's drift record grades the payload written by the **previous** tick
   (`read_existing_forecast` runs before the forecast phase), so CARRIED at `T`
   must be diffed against COMPUTED at `T−1`.

Together those two errors cancel, and the first run scored ~100% agreement on a
frame that was an hour short throughout. A harness that agrees for the wrong
reason is worse than one that disagrees.

**Independence check, run before anything downstream was believed.** COMPUTED
must reproduce CARRIED wherever nothing went wrong:

| BA | agree | of | |
|---|---:|---:|---|
| TAL | 162 | 162 | never-frozen control |
| CPLW | 163 | 163 | never-frozen control |
| FMPP | 162 | 162 | never-frozen control |
| PSCO | 144 | 151 | |
| LGEE | 112 | 138 | |
| SPA | 79 | 124 | |

**487 of 487 on the three control BAs.** The harness is exact where production is
healthy, so the disagreements are the anomaly and not the instrument.

## 3. Mechanism 1 — the freeze: a hole 24 hours back deletes the tail of `featured`

`_resolve_forecast_start` ([jobs/phases.py](../jobs/phases.py)) returns
`min(last_real_demand, last_featured_ts) + 1h`. The second term is the one that
freezes.

Autoregressive features are built by **positional** shift —
`demand_lag_1h/3h/24h/168h`, plus `ramp_rate` reading two rows back — and
`engineer_features` then does `dropna(subset=autoregressive)`. Rolling features
are NaN-tolerant (`min_periods=1`) and are **not** involved. So a NaN at row *i*
deletes rows *i+1*, *i+2*, *i+3*, *i+24* and *i+168*.

LGEE's demand series carries a contiguous 16-hour hole,
**2026-08-12T14:00 → 2026-08-13T05:00** — hours EIA never metered. Twenty-four
rows later that hole deletes exactly the 16 rows
**2026-08-13T14:00 → 2026-08-14T05:00** through `demand_lag_24h`. `featured`
therefore ends at 2026-08-13T13:00, and the origin sits at **2026-08-13T14:00**
while fresh demand keeps arriving one hour per tick.

Reproduced exactly, with `binding_term = featured` on every tick: **28 stalled
ticks for LGEE and 14 for PSCO** across the week, at eight distinct frozen
origins for LGEE alone. This is recurrent, not a single event.

Note what is *not* implicated. `demand_vintage.newest_hour` advances one hour per
tick with no repeat and no regression straight through the freeze, and
`scoring_region_complete` reports `ok=True` with `timings.forecast` of 11–58 s on
every tick. The data arrived, the models ran, the payload was rewritten. Only the
anchor stood still.

## 4. Mechanism 2 — the regression: upstream retracts hours it already published

The regression is a **different** condition reaching the **same** line, and the
offline replay cannot reproduce it — by construction. The vintage window is
monotone: it records first sight and settled value, never *absence*. An hour
withdrawn after publication leaves no trace in it.

Production's own log proves it instead. `write_drift_metrics` builds `actuals`
directly from that tick's demand frame (`dropna`, then `> 0`), and
`matchable_hours` counts how many rows of the previous payload it finds there. It
is a direct probe of how many hours the frame still held.

At the 2026-08-14T06:05Z tick the payload spanned 2026-08-13T14:00 onward and the
newest actual was 2026-08-14T05:00 — an intact frame gives
`matchable_hours = 16`. **It read 1.** Fifteen hours that had been published,
scored, and used were gone from the frame. `last_real_demand` collapsed back to
the last hour before the older hole, and the origin regressed to
**2026-08-12T15:00**.

Those fifteen hours are identifiable. **2026-08-13T10:00 → 2026-08-14T04:00 is a
19-hour run in which every hour was first seen as a placeholder** — `first_seen_d
== first_seen_df`, EIA publishing its own day-ahead value in the `D` field for
hours it has not metered (the #539 disclosure). Every one later revised. The run
ends at 2026-08-14T04:00; the first non-placeholder hour is 2026-08-14T05:00 —
the one hour that still matched.

Across the week, cross-tabulating replay agreement against frame shortfall
(`matchable_hours < lead_hours`):

| BA | agree/intact | agree/short | disagree/intact | disagree/short |
|---|---:|---:|---:|---:|
| LGEE | 101 | 10 | 1 | **25** |
| SPA | 77 | 1 | 18 | **27** |
| PSCO | 136 | 7 | **7** | 0 |
| TAL | 161 | 0 | 0 | 0 |
| CPLW | 162 | 0 | 0 | 0 |
| FMPP | 161 | 0 | 0 | 0 |

**25 of LGEE's 26 regressed ticks show a short frame; the three controls show 0
of 484.** And every regressed tick on every affected BA carried an origin that BA
had computed at an *earlier* tick — LGEE 26 of 26, PSCO 7 of 7, SPA 34 of 45.

**PSCO's variant is clock-aligned.** Six of its seven regressions land at exactly
**10:00 UTC on consecutive days**, each exactly **3 hours** — a daily upstream
re-baselining, mild enough that the frame still reads "intact" by the
`matchable ≥ lead` test but large enough to move the anchor backwards.

**SPA is the retraction class only** — 0 featured-bound stalls in 124 ticks, all
of its disagreement in the frame-shortfall column. So the two mechanisms are
separable across BAs, not two descriptions of one thing.

## 5. Why one line produces both

`_resolve_forecast_start` recomputes the origin from scratch every tick and has
no memory of the one it last served. Whatever this tick's frame says, it
publishes. When feature engineering loses its tail the anchor stalls; when
upstream withdraws hours the anchor walks backwards. Nothing in the path compares
the new origin to the old, and until now nothing logged the value either — it was
recoverable only by reconstructing it from the drift log a tick later, which is
how two multi-day freezes went unnoticed.

## 6. What shipped

**The origin may not regress.** When the freshly resolved start is older than the
origin already in Redis, the forecast phase keeps the served payload and does not
overwrite it. That payload is strictly the more current of the two and still
covers the horizon. The alternative — clamping the origin forward — would
forecast from an hour whose antecedent demand the frame no longer holds.

The phase returns `ok=True`: a live, newer payload is not a failed region, and
the deadline-shed branch in `scoring_job` reasons the same way. The refusal rides
on `PhaseResult.details` and a `forecast_origin_regressed` WARNING carrying the
resolved start, the served origin, and the regression in hours.

**The origin is now logged.** `forecast_start_resolved` emits the resolved value
and **both** terms of the `min()` — because which term binds is exactly the
difference between mechanism 1 and mechanism 2. The degenerate fallbacks report
too, under `binding_term` values `no_demand_frame` / `no_real_demand`.

**The guard is strictly `<`.** A *stalled* origin still republishes: the models
and the weather behind it have moved even when the anchor has not, and
suppressing that would hide mechanism 1 rather than fix it.

## 7. What did NOT ship, and why

**The positional-lag defect is left open and filed separately.** On a series with
holes, `shift(24)` means "24 rows back", which is not 24 hours back — and it is
the direct cause of mechanism 1. It is not fixed here because
`data/feature_engineering.py` is shared by the training job and the scoring path,
so the models were *trained* under the same convention. Reindexing to a
continuous hourly grid at serve time alone creates train/serve skew; doing it
properly means retraining 51 BAs × 3 models behind the ADR-010 serve-path gate.
Related: [#186](https://github.com/kristenmartino/gridpulse/issues/186).

**No published number moves at merge.** The guard is forward-only — it changes
which payload is served on a future regressed tick and rewrites nothing already
published. Features, model inputs, and every historical record are untouched.

## 8. Limits, stated

* **The replay cannot reproduce the retraction class**, because the vintage
  window is monotone. Mechanism 2 rests on production's own `matchable_hours`,
  not on the replay — a different instrument, which is why both are reported.
* **Reconstructed values are settled values.** Revision *timing* is not
  recoverable per hour (the record keeps `n_updates`, not a history). This does
  not touch the measurement: the origin is set by `dropna`, which sees NaN-ness
  and not magnitude. The 487/487 control result is the test of that claim.
* ~~**PSCO's 7 intact-frame regressions are characterised, not explained.**~~
  **Resolved post-deploy — see §10.** They are mechanism 1, and the replay
  misattributed them.
* ~~**SPA has 4 ticks where the payload carried a *newer* origin than the replay
  computed.**~~ **Resolved — see §11.** A harness artifact, and it exposed a
  reconstruction ambiguity that brackets some of §4's replay counts (though not
  §2's controls and not §3's result).

## 9. Reproducing

```bash
python scripts/forecast_origin_replay.py LGEE,SPA,PSCO,TAL,CPLW,FMPP \
  2026-08-11T12:00:00Z 2026-08-18T07:00:00Z drift_origins.csv out.json
```

`drift_origins.csv` is `timestamp,region,new_record_ts,lead_hours` rows from
`drift_updated`, via `gcloud logging read`. Needs ADC and `GCS_ENABLED`; reads
`cache/{demand,vintage}/{BA}/latest.parquet`. **Bucket versioning is Suspended
and each object is overwritten hourly**, so a replay run today reconstructs from
today's mirror — re-running it against a later mirror will not reproduce these
exact frames once the 30-day vintage window has rolled past this episode.

---

## 10. Post-deploy: the instrumentation resolved a residual on its first tick

Deployed 2026-08-18 at merge **`86d87c8`** — the commit that carries the code;
everything after it in this doc's history is prose. Verified by image SHA on all
three surfaces.

**Verify it by ancestry, not equality.** Several sessions merge to this repo
concurrently — `main` moved five times in the twenty minutes after this merge —
so the deployed tag will not equal `86d87c8` and an equality check reads as a
failed deploy when nothing is wrong:

```bash
git merge-base --is-ancestor 86d87c8 "$(gcloud run jobs describe gridpulse-scoring-job \
  --region us-east1 --format='value(spec.template.spec.template.spec.containers[0].image)' \
  | sed 's/.*://')" && echo LIVE
```

Two things that fall out of the same churn and cost time here. A commit can get a
CI run and **no deploy run at all** — the docs-only follow-up `eeb371b` was
superseded before its guard job fired, so waiting for "the deploy of my SHA" waits
forever. And the three surfaces are **not always on the same tag mid-flight**:
scoring and training read one tag while the web service read a later one, minutes
apart. Check all three, and judge each by ancestry. The first post-deploy scoring tick — **10:00Z**, the exact hour of
PSCO's daily signature — carried `forecast_start_resolved` for all 51 BAs:

| binding term | BAs |
|---|---:|
| `real_demand` | 49 |
| `featured` | 2 |

```
PSCO  forecast_start 07:00  last_real_demand 09:00  last_featured_ts 06:00
TIDC  forecast_start 08:00  last_real_demand 08:00  last_featured_ts 07:00
```

**PSCO's origin resolved to 07:00 when its demand reached 09:00 — three hours
short, with `featured` binding.** That is mechanism 1, at exactly the hour and
exactly the magnitude §4 recorded as PSCO's daily signature.

So §4's classification of PSCO's seven regressions as "intact-frame" was an
artefact of the replay, and the honest reading is narrower than what was written
there: `binding_term` in the replay is the **replay's** binding, and on precisely
those ticks the replay disagreed with production. The reconstructed frame did not
carry the hole, because the missing hours arrived later and the vintage window —
which records first sight and never absence — reports them as present. Production
saw the hole; the replay could not. **PSCO is not a third mechanism.** The
`matchable ≥ lead` test that classed those ticks "intact" remains correct on its
own terms: PSCO's frame was not short. It is the feature frame that was.

TIDC is not a stall: its `featured` tail sits one hour back, but `min()` returns
`07:00 + 1h = 08:00`, which is `last_real_demand + 1h` anyway.

No `forecast_origin_regressed` refusal fired on this tick, which is expected —
the regressions are episodic, and the guard exists for the episode.

**What this changes about §8's limits.** One residual is closed. The other
stands: SPA still has 4 of 124 ticks carrying a *newer* origin than the replay
computes, and the new log line does not speak to it. And the general caution is
now demonstrated rather than hypothetical — **the replay's `binding_term` is
evidence only on ticks where it agrees with production.** On disagreeing ticks
it describes a frame production did not have.

---

## 11. SPA's four "newer origin" ticks were the harness, and what they exposed

The four ticks where the payload carried a **newer** origin than the replay
computed — the opposite direction from the defect — are an artefact of the
reconstruction, not a production behaviour. Chasing them was worth it, because
the cause generalises.

### What the frame actually looked like

SPA's hours `2026-08-13T00:00 → 03:00` arrived **days** late: captured at
12:05Z the same day, then 08-15T12:12Z, then 08-16T12:11Z twice — lags of 12,
59, 82 and 81 hours, against SPA's median capture lag of 1.16 h. Hours 04:00 and
05:00 arrived on time, at +1.08 h. So at the 06:00 tick the four hours were
genuinely not ours yet, and the replay was right about *when*.

It was wrong about *how*. `frame_as_of` NaN-filled them, keeping the rows — and
a NaN row deletes the rows 1, 2, 3, 24 and 168 positions after it (§3). That
manufactured a hole, killed rows `08-13T04:00` and `05:00`, and dragged the
replay's origin back to `08-13T01:00` while production published `06:00`.

**EIA omitted them instead.** `job_data_fetched.demand_rows` for SPA settles it
without inference — the count froze for four consecutive ticks and then stepped:

```
00:05  2151      04:05  2151        08:05  2155  +1
01:04  2151  +0  05:04  2152  +1    ...
02:04  2151  +0  06:05  2153  +1    12:05  2160  +2   <-- new hour AND the
03:05  2151  +0  07:04  2154  +1                          backfilled 00:00
```

A frozen count is a response with nothing new in it, and the `+2` is one new
hour arriving alongside one three-day-late backfill — as a **new row**. Absent
rows delete nothing, so production had no hole and its origin was correct.

### The ambiguity this exposes, and its size

An hour that has not arrived can be either **a null row** (EIA reports the hour
with no value) or **absent** (EIA does not report the hour). The two are not
interchangeable here: one deletes five rows downstream, the other deletes none.
Nothing we retain records which it was at tick time — the vintage window stores
first sight, not the shape of the response.

Re-running the whole sweep under both models bounds it:

| BA | NaN-fill model | absent-row model |
|---|---:|---:|
| TAL | 162/162 | 162/162 |
| CPLW | 163/163 | 163/163 |
| FMPP | 162/162 | 162/162 |
| PSCO | 144/151 | 130/151 |
| LGEE | 112/138 | 103/138 |
| SPA | 79/124 | **82/124** |

**Neither model dominates**, which is the honest result: LGEE and PSCO get worse
under the absent-row model and SPA gets better, so EIA genuinely does both,
per hour, and no single rule reconstructs it. The truth is bracketed, not
pinned.

### What this does and does not touch

* **§2's independence check is unaffected** — the three controls read 487/487
  under *both* models. They have no irregular publication, so the choice cannot
  reach them. That is why they were the right controls.
* **§3's result is unaffected — verified, not assumed.** Re-run across the
  freeze-1 window, both models reproduce production on **17 of 17** ticks. They
  must: LGEE's hole is hours that were *never* published, so they are NaN rows
  in the mirror today and stay NaN rows under either rule. The mechanism-1
  finding does not depend on the ambiguity.
* **§4's replay-derived counts are bracketed**, and the disagreement columns move
  by up to 14 ticks. The `matchable_hours` columns do **not** move — that is
  production's own counter, computed from the frame the job actually held, and it
  is why mechanism 2 was argued from it rather than from the replay.
* **§4's "SPA is the retraction class only" should be read narrowly.** It rests on
  SPA having no `featured`-bound *agreeing* ticks; SPA's low agreement rate is
  substantially this artefact, not evidence about SPA.

The transferable point: the replay reconstructs *values and timing* faithfully
and **the shape of the upstream response not at all**. When a mechanism turns on
shape — and this one does, because `dropna` reads row presence — a reconstruction
can be right about every fact it stores and still be wrong.