<!--
How this file gets maintained:
- Per-PR: updated in the same commit as material work that changes
  active focus, next-3, blocked-on, or recent decisions
- End-of-session: agent re-verifies against gh issue list / gh pr list
- Pre-external-use: user re-reads top-to-bottom (~1 min)
If this file disagrees with gh, the live sources win — patch in a
follow-up commit.
-->

# Status — updated 2026-08-20

> Canonical pointer for "where am I, what's next." This file +
> [GitHub Projects board](https://github.com/users/kristenmartino/projects/1)
> + the issue tracker are the single source of truth for project state.
> See [`docs/internal/NEXT_UP.md`](docs/internal/NEXT_UP.md) for the full
> historical roadmap; see [`CLAUDE.md`](CLAUDE.md) for the pre-session
> sanity-check ritual.

## Active focus + open question

**2026-08-20 — [#559](https://github.com/kristenmartino/gridpulse/issues/559)
the injection study says NOT CONFIRMED, and found a defect in the fix itself.**

Pre-registered, run, and the hypothesis (temporal seed lowers WAPE on gapped
origins) **fails criterion 1**. Null control exact across six BAs
(`max|diff| = 0.0000000000`), so the harness is sound.

| stratum | n | mean Δ WAPE | median | MDE | consistency | verdict |
|---|---:|---:|---:|---:|---:|---|
| **A** naturally gapped | 249 | **−0.265** | +0.067 | 0.327 | 0.478 | not decisive |
| **B** never gapped | 432 | **+0.187** | +0.187 | 0.180 | 0.641 | not decisive |

A is outlier-dominated (mean and median disagree in sign) and **fails
satisficing on bias, −3.14% against ±2.0%**. B clears magnitude and satisfices
but wins only 64% of windows against 75% required. B's +0.187 replicates the
observational study's +0.181 from disjoint data — the mechanism is real; the
decision does not follow.

**The reason A runs backwards is a defect in the treatment arm, not in temporal
indexing.** `compute_temporal_autoregressive_snapshot` returns NaN for an absent
hour — correct, and the point of the fix — but the shared `row.fillna(0)` then
hands the model `demand_lag_24h = 0 MW`, the #129 poison the seed filter exists
to exclude. The positional arm never does this: its history always has ≥168
entries, so it feeds a plausible value from the wrong hour. Measured:
**13.08% of treatment steps zero-fill a lag on stratum B, rising to 22.57% on
IID** — whose −1.47 WAPE is the worst per-BA regression. So the run compared
temporal-indexing-**plus-zero-fill** against positional indexing, which is not
the hypothesis. This is §4.4 of the observational study, recorded as a limit and
left unmeasured; it is measured now.

**Decisions:** the flag stays **off**. The absent-hour policy must be *decided*
(carry-forward / interpolate / positional fallback for that lag), not inherited,
and pinned by a test. Re-running with a changed treatment arm requires a **new
pre-registration** — the stopping rule here was one run. **Evidence:**
[`docs/POSITIONAL_LAG_INJECTION_STUDY.md`](docs/POSITIONAL_LAG_INJECTION_STUDY.md).

**Open:** [#597](https://github.com/kristenmartino/gridpulse/pull/597) (seed
shadow) forces the same treatment arm, so it would record the defect too. Still
worth landing — gated, capped, off — but its arm needs the policy fix before its
output reads as evidence about temporal indexing.

---

**2026-08-20 — [#559](https://github.com/kristenmartino/gridpulse/issues/559)
seed shadow built and shipped DARK. It is a safety instrument, not the thing
that decides the flag — and that distinction is the finding.**

`temporal_ar_seed` is off because the offline replay was inconclusive at both
horizons. The natural next step is a production shadow, so it exists now — but
**it cannot settle accuracy either, and the arithmetic says so up front.** The
defect only yields an observation when a gap occurs: at the observed accrual a
verdict is **1.2 years** (48h windows) to **6.6 years** (168h) away. More
production time does not fix that. `scripts/seed_shadow_eval.py` prints its own
MDE and the implied wait beside every comparison so it cannot be read as a
verdict.

**What the shadow does answer**, and no offline replay can: does the temporal
path run clean against real production frames, what the second recursion costs
live, and whether live divergence matches the 2.1-2.7% the replay predicted.

**Gated on the exact divergence condition, not a proxy.**
`positional_seed_matches_hours` asks whether the last 168 seed entries are
contiguous hours ending at `origin - 1h` — the precise condition under which the
two arms are byte-identical. **3 of 51 BAs on 2026-08-20** (~3 CPU-seconds);
ungated it would be ~+380 CPU-s on a job whose worst recent tick used 1155s of
1800s. Membership is recomputed per tick and moves fast: LGEE alone on 08-18,
LGEE/SPA/TIDC on 08-20.

**Capped, not just gated** at `SEED_SHADOW_MAX_REGIONS_PER_TICK` (12): shedding
is whole-BA, so an unbounded enrichment would buy shadow data with later
regions' forecasts (CLAUDE.md #389, "bound what one run can cost").

**The gate audits itself.** One region per hour that the gate calls identical is
shadowed anyway, asserting zero divergence — because a gate that silently
skipped everything would look exactly like a fleet with no gaps, which is the
failure mode #584 found in the parity fixture. A nonzero audit divergence is an
alarm about the gate, not a finding about the seed.

Verified end to end on real LGEE data (origin `2026-08-20T12:00Z`): gate said
diverges, second arm ran, **divergence 2.82%** — consistent with the replay, and
higher than the 34h-era figure because LGEE was **+50h** out that day.

**Blocked on:** nothing in code. Needs a deploy, then
`temporal_ar_seed_shadow: True` and a second deploy — flags have no env
override. **Next:** the accuracy verdict comes from synthetic gap injection
offline, not from waiting on this.

---

**2026-08-18 — [#559](https://github.com/kristenmartino/gridpulse/issues/559)
MEASURED: the positional AR seed reads the wrong hour, and fixing it buys no
measured accuracy. Shipped behind `temporal_ar_seed`, default OFF.**

#559 prescribed a reindex plus a 51-BA × 3-model retrain behind the ADR-010 gate.
**Its premise does not hold.** `_parse_demand_records` never manufactures rows, so
the demand frame is whatever EIA returned — and EIA reports gap hours as rows with
null values. Across 51 BAs over 90 days: **7 absent rows** (all SPA, all May)
against **78 null rows**. On 50 of 51 the grid is already continuous, `shift(24)`
is temporally exact, and the training features were never wrong. Nothing to
reindex, no retrain to justify. ([PR #578](https://github.com/kristenmartino/gridpulse/pull/578)
reached the same census independently.)

**The real defect is one layer down.** `dropna` deletes every row whose lag source
was null, putting real holes into `featured` — LGEE 2171 → 1969 rows with 19h and
17h discontinuities — and `jobs/phases.py` seeds the recursion with that frame,
which `compute_autoregressive_snapshot` indexes **positionally**. At the origin
production resolved on 2026-08-18, `demand_lag_168h` read `08-10T01:00` against a
correct `08-11T11:00`: **34 hours off, live**, with `demand_roll_168h_*` spanning
201 real hours instead of 167. One null hour corrupts `lag_24h` for 24 subsequent
origins and `lag_168h` for 168 — a seven-day blast radius, on every tick, not only
the ticks whose origin stalls. Seven BAs carry a corrupt origin; 44 carry none.

**But the accuracy case is not there.** Replaying both seed conventions through the
serve path against archived vintages — same model, same weather, same origins —
`verdict()` declined twice: mean **+0.090** WAPE at 168h (n=24, MDE 0.466) and
**+0.181** at 48h (n=85, MDE 0.406). The arms diverge **2.1–2.7% of demand**, so
values genuinely move; the accuracy effect does not clear noise. Only TIDC beats
its own MDE, which is a post-hoc per-BA look at an inconclusive pooled result and
is hypothesis-generating only. The study cannot be rescued by running it harder:
detecting +0.18 needs ~600 non-overlapping windows and 26 exist, because the defect
requires a gap. Null control held exactly — MISO and PJM at `0.000000000%`
divergence across 47 origins.

**Shipped:** `temporal_ar_seed`, default **off**, fail-open at every seam
(flag-off is byte-identical, pinned by test). Production with the flag on
reproduces the study's independent treatment arm to **0.0000000000**.

**Open:** whether to turn it on. That wants a shadow run, not this study — the
honest framing is a correctness fix with no demonstrated accuracy benefit, and two
BAs move the wrong way inside noise. **Evidence:**
[`docs/POSITIONAL_LAG_SEED_STUDY.md`](docs/POSITIONAL_LAG_SEED_STUDY.md).

**Also fixed:** the parity test that should have caught this compared both AR
implementations on a **gapless** fixture, where they agree by construction.
`tests/unit/test_temporal_ar_seed.py` carries the gapped version, plus a fuzz
over randomised gap patterns and a companion test asserting the positional path
**fails** that same property — so the fuzz cannot quietly stop testing anything.

**[#186](https://github.com/kristenmartino/gridpulse/issues/186) re-scoped, not
closed.** Its premise ("the two paths match today only by coincidence") is wrong
— they do not match, and have not for as long as gaps have existed. And its
Option A is priced: a per-row shared core is **611x** the vectorised training
path (+372s per fleet run, ~double with the holdout), and routing inference
through pandas is **60x** the snapshot (+54s/tick). Both buy what the property
test gives free, against #389's rule to bound per-run cost. The paths now share a
representation (continuous hourly grid, NaN holes) and a semantic instead. Three
implementations exist only while the flag keeps both inference paths alive;
the flag decision deletes one.

**Caught late, worth recording:** the first temporal implementation keyed history
by timestamp in a `dict` and cost **79x** the positional snapshot — **+74.9s per
scoring tick** fleet-wide, which would have blocked flipping the flag on at all.
It was not measured before it shipped. The dense hour-indexed array that replaced
it runs at **0.5x** the positional path, and forecasts are byte-identical to the
studied arm.

---

**2026-08-18 — [#537](https://github.com/kristenmartino/gridpulse/issues/537)
ROOT-CAUSED: the forecast origin has no memory, so it stalls when feature
engineering loses its tail and walks BACKWARDS when EIA withdraws hours.**

Two mechanisms, one line. `_resolve_forecast_start` returns
`min(last_real_demand, last_featured_ts) + 1h`, recomputed from scratch every
tick. **The stall:** AR lags are computed by *positional* shift, so LGEE's
contiguous 16-hour demand hole (`08-12T14:00 → 08-13T05:00`) deletes exactly the
16 rows 24 positions later via `demand_lag_24h`, freezing the origin at
`08-13T14:00`. **The regression:** EIA published `08-13T10:00 → 08-14T04:00` as
placeholders (`D == DF`, 19 hours) and then withdrew them, collapsing the anchor
to `08-12T15:00` — **23 hours older than a vintage already served** — for 24
ticks, at leads to 63h.

**The replay proves one and cannot prove the other, so both were measured with
different instruments.** `scripts/forecast_origin_replay.py` reruns the real
primitives against the frame each tick actually held, reconstructed from
`captured_at`: **487 of 487 exact on the three never-frozen control BAs**, and it
reproduces every stalled tick (LGEE 28, PSCO 14) with `binding_term=featured`. It
*cannot* reproduce a retraction — the vintage window is monotone by construction —
so that half rests on production's own `matchable_hours`, which read **1 where an
intact frame gives 16**. Fleet-wide: **25 of LGEE's 26 regressed ticks show a
short frame, against 0 of 484 on the controls.** PSCO's variant is clock-aligned —
six regressions at exactly 10:00 UTC on consecutive days, each exactly 3 hours.

**Two errors in the harness cancelled each other and scored ~100% agreement on a
frame that was an hour short throughout** — `captured_at` is stamped minutes into
its own tick, and a drift record grades the *previous* tick's payload. Caught by
the control-BA check, not by inspection.

**Shipped:** the forecast phase refuses to overwrite a served payload with an
older-origin one (`ok=True` — a live newer payload is not a failed region), and
`forecast_start_resolved` now logs the origin plus both `min()` terms. Guard is
strictly `<`, so a *stalled* origin still republishes. Forward-only: no published
number moves at merge. **Evidence:**
[`docs/FORECAST_ORIGIN_REGRESSION.md`](docs/FORECAST_ORIGIN_REGRESSION.md).

**Deliberately NOT fixed:** the row-deletion defect that causes the stall
([#559](https://github.com/kristenmartino/gridpulse/issues/559)). **Both this
file's and #559's first framing were overstated and are corrected**: the defect
was argued as temporally-wrong lag values needing a reindex plus a 51-BA × 3-model
retrain, but absent rows are **7 of 110,704 fleet-wide (0.0063%), all in SPA, all
from May** — the frames are complete grids and `shift(24)` is exact on 99.9937%
of rows. What is real is `dropna` deleting rows whose lag source is **null** (IID
37, LGEE 16, PSCO 11, TIDC 10), which is fixable serve-side without touching a
feature value. Live: 4 of 102 BA-ticks bind on `featured`, max stall 3h — PSCO 3h,
LDWP 2h, AZPS 2h, TIDC 1h, and **LDWP/AZPS were not in the replay set**, so the
exposure is broader than that sample showed. Full correction:
[`docs/FORECAST_ORIGIN_REGRESSION.md`](docs/FORECAST_ORIGIN_REGRESSION.md) §7.

**Deployed** at merge `86d87c8`, image SHA verified on job/training/service. The
first post-deploy tick (10:00Z — PSCO's own signature hour) closed a residual:
49 of 51 BAs bound on `real_demand`, and **PSCO resolved to 07:00 with demand at
09:00, `featured` binding**. Its daily 3-hour anomaly is mechanism 1, not a third
thing; the replay had misattributed it, because the vintage window records first
sight and never absence, so the reconstructed frame lacked a hole production had.

**SPA's 4 newer-origin ticks: resolved, and a harness artifact.** Those hours
arrived 12-82h late against SPA's 1.16h median, and the replay NaN-filled them —
but `job_data_fetched.demand_rows` froze for four ticks and then stepped
`+1,+1,…,+2` (a new hour plus a three-day-late backfill arriving as a NEW ROW),
so EIA had **omitted** them. Absent rows delete nothing; NaN rows delete five
positions downstream. Production was right; the replay manufactured a hole.

**The ambiguity that exposes is now bracketed rather than hidden.** An unarrived
hour is either a null row or an absent row, EIA does **both** per hour, and
nothing we retain records which. Re-run under both models: controls 487/487
either way, SPA 79→82, but LGEE 112→103 and PSCO 144→130 — **neither model
dominates**. The mechanism-1 result is model-independent (**17 of 17** freeze
ticks under both, since LGEE's hole is hours never published), and the
`matchable_hours` evidence behind mechanism 2 is production's own counter and
does not move. The replay now takes an `unarrived` model so the bracket is
reproducible.

**Open:** nothing specific. The general lesson worth carrying: the replay
reconstructs values and timing faithfully and the **shape** of the upstream
response not at all — and `binding_term` from it is evidence only on ticks where
it agrees with production.

---

**2026-08-18 — [#542](https://github.com/kristenmartino/gridpulse/issues/542)
FIXED: re-grading erased each drift record's lead, and the counter that would
have shown it had been published, unread, for weeks.**

`models.drift.regrade_records` dropped `lead_hours` on every rebuild.
`filter_by_lead` keeps unknown-lead records *by design* — they were assumed to
predate the field — so every EIA revision moved one more observation past the
P2-19 filter. At capture (2026-08-18T07:06Z) **82.7% of retained records, 81.3%
of the 30-day window and 55.3% of the 7-day window** carried no lead, sorted by
each BA's revision rate rather than by age: PACW 162 of 163 in-window against
PJM 3 of 150.

**The fix is one line; the work was proving what it moves**, since
`/api/v1/drift` feeds the Models Live Drift panel, the Overview headline and
the `n_7d ≥ 24` visibility gate. The code repairs nothing already blanked — a
blanked lead is unrecoverable *inside* the pipeline — so a merge-time
before/after would have read 0.00 everywhere. It was recoverable from outside:
`drift_updated` has logged `(region, target hour, lead)` every tick since
2026-08-05 (#407), and 31 days of that rebuilds the erased map. Harness
validated before it was believed — reproduces the published `n_7d` on **204 of
204** blocks; lead recovery inside the 7-day window **100%** (4,285/4,285).

**The published number moves both ways, and the prediction was wrong.** I
expected a uniform improvement (error grows with lead). Ensemble 7d: **17 BAs
better, 20 worse, 14 unmoved**; fleet 3.706 → 3.630 (−0.076). The tail is the
story — LDWP **12.085 → 8.231** and IID **16.148 → 13.643**, against AZPS
**9.779 → 11.427** and PSCO **9.639 → 10.766** getting *worse*. 21 of 408
blocks move >1 pt. **LDWP crosses the visibility gate on all four models**
(`n_7d` 25 → 15) and correctly falls back to "Warming": its published figure
rested on 25 records of which 10 were never 1-hour-ahead.

**Two things fell out that were not the assignment.** `reconcile.py` — the
checker for this exact panel — never lead-filtered its settled arm, so it
graded a different *population*; invisible while leads were blanked, **1 → 12**
false A1 findings once repaired, **12 → 0** with the filter mirrored. And
[#537](https://github.com/kristenmartino/gridpulse/issues/537) is answered as a
side effect: it **shares an upstream phenomenon with #542 and no code**. A
frozen forecast origin makes the 1h path grade at a growing lead *and* makes
the horizon path re-derive a target it already has. LGEE's origin froze for 15
ticks, then served a **23-hour-older vintage for another 24** — leads to
**63h**, horizon 24h coverage 82 of 168. `_expire_pending`, the standing
hypothesis, is **ruled out** (LGEE pending 117 vs PJM 126). Fleet correlation
is direct: 0 frozen ticks → 166 of 168.

**Evidence:** [`docs/DRIFT_LEAD_REGRADE.md`](docs/DRIFT_LEAD_REGRADE.md).
**`serve_grade` does not move** — 442,537 horizon records, 0 carry a lead.
**Open:** why LGEE's forecast payload froze and regressed (#537's remaining
half, a scoring-path question). Backfill of the six >1 pt BAs runs post-deploy.

---
---

**2026-08-18 — [#539](https://github.com/kristenmartino/gridpulse/issues/539)
DISCLOSED: the benchmark's independence held on the feature side and leaked on
the anchor side.**

`forecast_mw` is genuinely not a model feature, and ADR-009 substitution is
genuinely scoped to `broken`. But `_resolve_forecast_start` anchors on the last
hour carrying a *positive* `D`, and for hours EIA has not metered yet EIA
publishes the BA's day-ahead value in that field — so on those hours the seed
of our recursion is the series we score against. `pair_hours` drops the hour
from **scoring** and nothing drops the hour that **seeded** the forecast. The
methodology answered this exact objection for broken feeds and was silent about
it for the scored set, which is what made the silence read as a claim.

**Verified independently before acting.** Recomputed for all 51 BAs from the
GCS vintage mirror's raw columns, not by calling `scoreability()`: MISO 36.58%,
CAISO 26.56%, NEVP 18.50%, SCEG 12.24%, ERCOT 10.99%, fleet median 3.34% —
matching the live payload to the decimal. Two things that pass established
beyond agreement: the post-#535 `df_at` guard is **load-bearing** (SCEG reads
31.7% on raw `D == DF` equality and 12.24% with it), and the flag is not an
artefact of exact equality — flagged hours later revise on 97.9-100% of
records against 1.3-5.7% for non-placeholder hours on CAISO/PJM/NYISO.

**It is a disclosure, not a bias with a sign**, and it must not be "fixed" by
refusing to anchor: measured at 6.55% against 7.72% mean error, 9 of 12 BAs.

**Cheaper than it looked.** The per-BA rate was *already published twice* —
`placeholder_pct` on every payload row, `stub_pct` in the scoreability snapshot
— under two opaque names with nothing saying what either meant. No new field,
so the #535 export-parity guard is not in play. Shipped: methodology §5/§7/§11
+ limit 11 + change log, one `_BENCHMARK_NOTES` entry making the existing
one-sided note symmetric, and two guard tests. Two live prose defects fixed in
the same pass: §7 said the forecast anchors on the last *real* demand hour when
the selector admits placeholders, and §5's broken-feed row implied the
exclusion had disposed of the self-reference on the scored set.

**DF-as-feature — NOT pre-registered, and the reason is not caution.** Earlier
handoffs ranked it P3 "blocked on #535". It is unblocked and the reframing makes
it **less** defensible: the official arm *is* `forecast_mw`, so a DF-fed model
scored against DF is circular and the benchmark cannot arbitrate it; the
direction argument that makes the anchor disclosable inverts for a feature
chosen *because* it imports their skill; and it would overturn a defect already
recorded under fire (`scripts/error_analysis.py:190-196` — DF leaked in via
merge suffixes, became the top feature, was logged as leaking the arm we
measure against into the arm being measured). It **is** runnable offline, and
that is the trap: `make_day_ahead_safe` strips `demand_lag_1h` and the 6-7 day
archive lag removes placeholders, so the harness measures cleanly precisely
because it lacks the production path in question — the ADR-010 failure mode.

**Open question:** the materiality of limit 11 is **unmeasured**, not small —
and that is now a gap in *evidence*, no longer a gap in *capability*. **Both
halves of the original blocker are gone, same day.**
[#542](https://github.com/kristenmartino/gridpulse/issues/542) is fixed, so
records carry a lead again, and the erased history was reconstructed for the
7-day window at 100% from 31 days of `drift_updated` log lines
(`docs/DRIFT_LEAD_REGRADE.md`). And
[#547](https://github.com/kristenmartino/gridpulse/issues/547) shipped the
instrument (#555): every forecast payload now carries an `anchor` block —
hour, value, was-it-a-placeholder, was-it-conditioned — riding onto both drift
paths. It has measured **nothing** yet, and limit 11 is guarded by a test that
fails if "instrumented" is ever allowed to read as "measured".

**Two claims of mine that #555 refuted, kept rather than quietly dropped.**
This entry said the anchor hour "is not instrumented … `DriftRecord` and
`PairedHour` carry none", and #547's body went further and said it could not be
recovered retrospectively at all because `lead_hours` is the realized lead
rather than the anchor. **That impossibility claim is false, and it is the
intuitive version.** Row 0 of a forecast *is* `anchor + 1h` by construction
(`_build_future_feature_frame` starts the frame at `forecast_start`) and
`_lead_hours` counts from row 0 — so `anchor = target − lead_hours` is exact on
the 1h path, and `anchor = target − H − 1h` on the horizon path needs no lead
at all. A bounded retrospective measurement was available over the vintage
mirror's rolling ~30-day window the whole time. I asserted unmeasurability from
the record schemas without reading the frame builder; forward recording is
justified by *reach*, not by impossibility.

**2026-08-18 — [#535](https://github.com/kristenmartino/gridpulse/issues/535)
ANSWERED and fixed: the `df-coverage` exclusion was measuring our own
collector and publishing the result as a fact about the balancing authority.**

`/api/v1/benchmark` served **25 of 51 scoreable** for ~3 weeks, excluding five
of the seven large ISOs, under a reason reading "The BA publishes a day-ahead
forecast for under 80% of hours". That sentence was false for all but one of
the BAs it was applied to.

**Root cause.** `_readings` admits an hour only once EIA publishes a positive
metered `D`, snapshotting `DF` at that instant; `update_vintage_records` then
short-circuited whenever `D` had not moved and copied `first_seen_df`
unconditionally. A NaN captured at first sight was **permanent**, and
`scoreability()` counted those NaNs as the BA's publishing behaviour.

**Evidence, measured not inferred.** Against EIA directly over the live
payload's own window: ISONE publishes DF for 100% of hours where we recorded
66.8%, NYISO 96.7 vs 58.0, ERCOT 93.3 vs 64.1, MISO 93.3 vs 63.8. Swept across
all 51 BAs, **exactly two fall below the gate upstream** — SPP (53.8%) and SPA
(78.7%, already excluded as a broken feed). The hours we lost form a diurnal
block aligned to each BA's *local* early morning, near-identical across
unrelated BAs in different interconnects; and the values are genuine forecasts,
not placeholders (of 210 ERCOT / 278 NYISO / 137 PJM recovered hours, 0/1/0
have `DF == D`).

**#536's leading hypothesis is REFUTED.** It named the frozen `fetch_demand`
cache and said the decisive test — distinct `captured_at` in `vintage:NYISO`,
~30 confirms / ~720 refutes — needed in-VPC Redis. The **GCS mirror carries
the same rows**: NYISO reads **672 of 719**, ERCOT 675, CAISO 669. Capture is
per-tick. Two further #536 claims corrected: ERCOT/MISO/ISONE do *not* have a
"real ~13% upstream gap" (6.7 / 6.7 / **0%**), and SPP is sparse upstream
rather than absent.

**Verified by replay before deploy** — the new capture applied to the real
production vintage window for all 51 BAs: **25 → 46 scoreable, 21 restored, 0
newly excluded**, with post-fix coverage matching the independent upstream
measurement on every restored BA. SPP stays out at 53.6%. (Live serves **45** —
see the 2026-08-18 note below; the replay measured the coverage gate, and MISO
is demoted a step later by paired hours.)

**The load-bearing half is the rail, not the fill.** `df_at` dates the DF
observation and `pair_hours` now grades staleness on it, so a late-filled DF
is still dropped from the as-issued arm. Without that line the fix would put
post-revision values on the as-issued arm at scale — the #358/#392 defect,
worse than the bug it repairs.

**Also shipped:** the count now lives only in the live payload (prose that
asserts it fails a new guard test); the page names its population and no
longer claims an ERCOT carve-out that is not there; and two policies watch the
count *and* warn on BAs approaching the gate. **Both are now applied and
enabled** — `alertPolicies/1095567904752750375` (drop) and
`alertPolicies/15888827698887105322` (at-risk), verified against the Monitoring
API; `_KNOWN_UNAPPLIED` is empty.

**2026-08-18 — the at-risk band fired for real within hours, and its own
justification turned out to be pre-fix.** First firing 06:18Z: **TEC at
`df_coverage` 80.1%**, a tenth of a point above the 0.80 gate,
`df_asissued_coverage` 49.0%. **It is a true positive and not the #535
asymmetry** — measured against EIA over the payload's own 719-hour window, EIA
published DF for **576** hours and we recorded **576**. TEC's DF stopped at
`2026-08-17T04Z` while its metered `D` kept flowing; on that trajectory it
crosses the gate around `2026-08-19T06Z`. Decision: **let it drop and watch
it** — the gate is correct as written and the drop alert (floor 40) is nowhere
near firing.

But the band was argued from **CAISO 82.9% / PJM 81.0%**, and those were the
*broken pre-fix* readings — the very defect the same commit repaired. Post-fix
they measure **100.0% / 99.7%**. Six sites carried that dead evidence into the
runbook an on-call reader would follow; all now cite TEC. **A rationale written
in the same breath as a fix is measured on the pre-fix world.**

**Live count is 45, not the 46 the replay predicted**, and the difference is
not coverage. **MISO** clears the gate at 93.2% and is excluded for
`insufficient-paired-hours` (175 against `MIN_PAIRED_HOURS` 200) — the 24h
lead's pairing verdict demoting a BA the coverage gate passed, which is
[#539](https://github.com/kristenmartino/gridpulse/issues/539) surfacing as an
exclusion.

**The corrected runbook could not be delivered to the console.**
`PATCH ?updateMask=documentation` on a `conditionMatchedLog` policy returns
**HTTP 200**, fails with `validity code 13` ("Recompilation of log match
condition failed during update"), and **flips `enabled` to false** — it
disarms the alert as a side effect of documenting it. The at-risk alert was
disabled twice during this and re-enabled within ~1 min each time; all 11
policies verified `enabled: true`, `validity: ok` afterwards. So
`benchmark_coverage_at_risk_alert.json` was **declared-correct and
applied-stale**, the one gap `test_monitoring_policies_applied.py` could not
see: it compares files to a table of ids, never documentation to documentation.

**2026-08-18 — RESOLVED, and the diagnosis above was wrong.** The cause is a
**4000-character cap on `documentation.content`**, not log-match immutability.
The corrected runbook shipped at **4035** characters, so it was un-appliable by
any route from the moment it merged — `policies create` rejects it cleanly
(`INVALID_ARGUMENT: 'description' must not be more than 4000 characters`),
while `PATCH` on a log-match policy reports the same over-length body as a
bogus recompilation failure and disables the alert. **No delete-and-recreate
was needed and no new policy id was minted**; the at-risk alert keeps
`alertPolicies/15888827698887105322`. Fixed by moving the 587-char design
rationale ("why it counts the failing direction") into
[`docs/monitoring/README.md`](docs/monitoring/README.md), bringing the runbook
to **3575** (425 to spare), then patching it in place.

**Why the first diagnosis held up:** every failing attempt carried that same
4035-char body, so "fails even byte-identical" and "fails because too long"
predicted identical outcomes. Settled on a throwaway policy by varying only
length — two short patches applied cleanly **in sequence** (refuting "only the
first succeeds"), the 4035-char one failed, and a short one **after** it
restored `enabled: true` and cleared `validity` (refuting "stays invalid").
**A reproduction is not a diagnosis** — re-running the failure proved it
repeatable, not that the stated cause was the operative one, and the control
that separated them was the one never run. `test_monitoring_policies_applied.py`
now fails the build on any runbook over the cap.

**Split out:**
[#554](https://github.com/kristenmartino/gridpulse/issues/554) — the length
check closes *one* reason committed and applied can diverge; the likelier one
is untouched, since applying a documentation edit is still a manual step
outside CI and nothing compares the applied runbook to the committed file. The
same issue covers noticing a policy left `enabled: false` — which happened
twice during this session and both times was caught only because someone was
looking. Modelled on `deploy-divergence.yml`, which already answers the
"needs a live API call" objection.

**2026-08-18 — 554 is BUILT and MERGED, and it found a second drift on its
first run.** Shipped in
[PR #560](https://github.com/kristenmartino/gridpulse/pull/560) (squashed as
`029065d`); the issue is closed.
`scripts/check_monitoring_divergence.py` runs hourly as a step in
`deploy-divergence.yml` and compares every applied policy's
`documentation.content`, `enabled`, `validity` and `notificationChannels`
against the committed files, resolving policy → file through the README's
applied table. Granted `roles/monitoring.viewer` to `github-actions-deploy@`
(read-only, weaker than the `run.admin` it held).

**The drift it found was 14 days old and nobody knew.**
`scoring_runtime_creep` was serving the pre-#389 runbook — 1153 characters,
three generic steps — while the repo has carried the 3555-character rewrite
since 2026-08-04 (`61b5686`). The applied copy was last mutated **2026-07-08**
and never again. So the console's runbook for the runtime-creep alert omitted
the entire partial-degradation triage: the "exceptions high but
`eia_max_retries_exceeded` at zero ⇒ the breaker cannot trip, runtime will keep
climbing" branch, which is the incident class that killed two scoring ticks on
2026-08-04. Patched to the committed text and verified by read; the check now
exits 0 across all 11 policies.

**What this corrects in the reasoning above, not just in the config.** The
README argued this was *deliberately not built* because "drift has not been
observed here", with revisit triggers about console edits, a second person
getting access, or a wrong id reaching `main`. Drift had already been observed
twice, and **no listed trigger could have fired for either** — both kept the
correct id the whole time, which is exactly why the guard test stayed green.
The triggers were written about identity; the failure was about content and
state. A "revisit if" list is only as good as its guess about how the next
failure will look.

**Verified in CI, not just locally.** The step had never run on a runner before
merge — `deploy-divergence.yml` checks out `ref: main`, so a branch dispatch
would have run main's copy, which did not have the script yet. Dispatched
against the merged main
([run 32126906804](https://github.com/kristenmartino/gridpulse/actions/runs/32126906804)):
step `success`, output `applied-table rows: 11   policies compared: 11` /
`OK: applied policies match the repo`. That is what confirms the parts local
runs could not — WIF resolves to the deploy SA, `roles/monitoring.viewer` is
sufficient, and the GA `gcloud monitoring policies` group is present on a stock
`setup-gcloud` runner (the alpha group is not, which is what blocked checking
this by hand earlier the same day). Now on the hourly `:23` schedule alongside
the deploy check.

**ANSWERED, and the premise was wrong:**
[#549](https://github.com/kristenmartino/gridpulse/issues/549) asked how to
tell chronic sparsity from episodic blackout. Measured across all 51 BAs on
2026-08-18, **that distinction does not exist in this fleet** — every BA with
any absence has 92–100% of its absent hours in runs of ≥3h, SPP included.
SPP's absence is ONE contiguous **341-hour** block (feed stopped
`2026-08-04T06Z`, never resumed); TEC's is six whole-day blocks with a live
feed. Both confirmed upstream against EIA.

What separates them is **liveness**, cleanly: hours since the newest published
DF are SPP 341, TEC 30, every other BA ≤6. So coverage stopped gating and
`MAX_DF_GAP_HOURS = 168` gates instead — the longest DF gap anywhere in the
window, not just the trailing one (#587). Replayed over the live window:
**46 scoreable before, 46 after, zero newly excluded** — the change is
population-neutral today and prevents TEC's false exclusion when it crosses.
`MIN_DF_COVERAGE` is unchanged at 0.80; it just decides nothing.

**Split out:** [#537](https://github.com/kristenmartino/gridpulse/issues/537) —
the horizon-drift 7-day window is short fleet-wide (151-167 of 168) and LGEE
loses ~half (80/86/89). **Different cause**: the low-actual filter is ruled out
for LGEE (zero hours below threshold, median 5,328 MW), upstream nulls explain
8.3% of ~50%, and `_expire_pending` is the untested leading candidate.

**2026-08-07 — the 2026-08-04 incident is CLOSED, and the cost work it turned
into is measured rather than projected.**

**Scoring runtime: 406s median over 17 ticks** (2026-08-06T20:00Z → 08-07T12:30Z),
51/51 every tick, p95 488s. [#171](https://github.com/kristenmartino/gridpulse/issues/171)'s
`<600s` criterion now holds on a **population**, not the single 370.7s run. At
4 vCPU / 8 GiB that is **$26.06/mo against $26.72 pre-bump — cheaper AND ~2x
faster** (break-even median 416s).

**Attribution is not isolable and should not be claimed.** The 8-worker bump and
the forecast-path work (#405/#413/#423) landed the same day, and `forecast` is
57.7% of worker time against `fetch`'s 15.0% — the perf work plausibly did more
of it than the concurrency did.

**The measurement trap this line kept setting.** A fast median means nothing
without `eia_gcs_fallback` and `eia_circuit_tripped` beside it. My first pull
covered the trailing 24h, returned the same 406s, and looked clean — it
contained a real EIA outage on **2026-08-06 17:00–19:00** (426 fallbacks, 5
circuit trips) whose ticks were fast *because they had stopped fetching*.
Identical to 2026-08-04T23:00, when a 736s run read as a 2.2x win and was the
fleet serving last-known data. Both are now in CANONICAL_FACTS.

> **Superseded 2026-08-11 — the ~$180 below was a projection and it was
> pessimistic.** With the BigQuery billing export live, the whole-project
> run-rate **measures $115.91/mo** (net of credits, mean 2026-08-07..09), and
> steady state is ~$110 once the weekly backtest cadence is fully reflected —
> **under the $150 budget**, not $30 over it. By service: Cloud Run $67.66,
> Memorystore $35.28, Artifact Registry $9.41, Cloud Storage $3.09. The
> per-item narrative below still reads correctly as *what was projected at the
> time*; only the total moved. The projection's one systematic error was not
> crediting the GCP free tier. **Memorystore is now the second-largest line and
> the largest untouched one.** Detail, method and caveats: the whole-project
> run-rate row in [`docs/CANONICAL_FACTS.md`](docs/CANONICAL_FACTS.md).

**Bill: ~$317/mo → ~$180/mo** against a $150 budget. Web tier $114 → ~$38
(resize + `--cpu-throttling` + `--min-instances 0`); Artifact Registry $32 →
~$9 (**measured** 407 → 124 versions, ~207 GiB, ~$20.74/mo — the policy existed
but sat in `cleanupPolicyDryRun: true`); training job $73 → ~$45 (backtest folds
were fitting 60 discarded CV boosters per BA, plus 8→4 GiB on a measured
1.24 GiB peak). **Memorystore $36 ruled out, not reduced**: Basic 1 GB is the
tier floor, `maxmemory_policy` is `volatile-lru` (read via `INFO memory` —
`CONFIG GET` is blocked on Memorystore), `evicted_keys: 0`. Nothing to fix.

**Alerting: 8 policies, all applied, `_KNOWN_UNAPPLIED` empty.**
`scoring_partial_failure` had been committed-and-inert since 2026-07-08 — and
could not have fired even if applied, because a SIGKILLed run never reaches its
`log.error`. It and the soft deadline only became useful as a pair.

**The pattern worth keeping, now in `docs/monitoring/README.md`:** seven controls
this week were *configured and inert* — an unapplied alert, a `--force` flag with
no caller, a registry policy in dry-run, a Redis write failure logged through
stdlib logging where no policy could match it, a soft deadline that could not
fire for its own workload shape, a deploy comment arguing against its own flag,
and a PR opened and never merged. Each looked correct in the place you would
naturally check. **Assert the enforcement, not the declaration** —
`tests/unit/test_monitoring_policies_applied.py` is the shape to copy.

---

**2026-08-04 — production incident: EIA partial degradation killed two scoring
ticks, and every defense we had was keyed to the wrong failure shape.**
[#389](https://github.com/kristenmartino/gridpulse/issues/389).

`api.eia.gov` began returning 502/504 and 30s read timeouts at ~16:00 UTC.
Runtime went 1004 → 1283 → **two ticks KILLED at the 1800s cap** → 1792s (8s of
margin, on a retry) → 1375s → 1317s. Nothing shipped that day; the deployed
image was 5 days old.

**The decisive evidence is what did NOT happen: 0 `eia_max_retries_exceeded`,
0 `eia_gcs_fallback`, 0 `eia_stale_fallback`, 0 `eia_rate_limited`, 0
`eia_circuit_tripped`.** Every call eventually *succeeded* on retry. No data
was ever lost and EIA never throttled us. The job spent its entire budget
paying retry tax on calls that would have worked.

Third scoring-job timeout, third distinct cause — and this one has no defense:

| | 2026-06-01 (#171) | 2026-06-04 (#174) | 2026-08-04 |
|---|---|---|---|
| cause | our runtime crept into the ceiling | upstream **vanished** | upstream got **slow and flaky** |
| defense built | creep alert at 70% | consecutive-failure breaker | — |

**Why the #174 breaker cannot help.** It counts *consecutive* hard failures
against a threshold of 3, and `record_failure()` only fires when the whole
retry budget is exhausted while `record_success()` zeroes the counter. A call
that timed out four times and succeeded on the fifth burns ~134s and registers
as a **success**. The breaker keys on the *shape* of failure; this failure has
only a *rate*.

**Four more findings, each its own defect:**
- **No headroom ever existed.** #171's acceptance criterion — healthy run
  "well under half the task timeout, target `<600s`" — was **never met**;
  baseline is ~820s. Nothing could have caught that: the creep alarm's
  threshold is `0.70 × 1800 = 1260s`, so it **defends the ceiling, not the
  criterion**. A run at 820s is 37% over the criterion and 0% of the way to
  the alarm.
- **The killed runs had already done the work.** Both reached ~49–51 of 51 BAs,
  and per-BA Redis writes are incremental — but `write_meta("last_scored")`
  sits *after* the fan-out, so neither recorded any of it. `last_scored` stayed
  pinned at 16:22 until 20:01: **~2 hours of deep-health degraded for work that
  had actually been done.**
- **The runs overlapped.** `--task-timeout × (--max-retries + 1)` = 3600s
  against an hourly cadence is zero margin. The 19:00 retry finished at 20:01,
  after 20:00 had started — two scoring processes against a dependency that
  was failing because it was overloaded. The `deploy-prod.yml` comment
  asserting runs "can't overlap" was wrong.
- **The runbook told on-call to do nothing.** `docs/SCHEDULED_JOBS.md` said
  that on `Read timed out`, "since #174 the EIA circuit breaker self-mitigates
  this ... wait it out." False under partial degradation, and the documented
  response while two ticks died.

**Shipped:** per-phase instrumentation + `eia_client_stats` (the EIA latency
distribution the 30s timeout had never been sized against); a **per-call
wall-clock budget** — worst case 169s → 40s measured, which models every hour
of the incident back under the creep threshold; a **soft deadline** at 85% so a
squeezed run writes its meta and exits 0 instead of being SIGKILLed with
nothing recorded; and the runbook rewritten to branch on total-outage vs
partial-degradation, with the log signature that distinguishes them.

**Deliberately NOT done: making the breaker trip on a failure rate.** Zero data
was lost — a breaker tripping at 8–15% would fail-fast the remaining BAs onto
last-known-good, trading fresh data we could actually get for runtime the
budget recovers more cheaply. Two characterization tests exist so nobody
"fixes" that into tripping.

**Open question — the fix is modelled, not proven.** EIA was already recovering
when this landed, so no before/after is attributable. The real proof arrives at
the next upstream wobble.

**Update 2026-08-05 — the worker bump is measured, and it is inconclusive.**
The 8-worker / `--cpu 4` / 8Gi config went live at **01:44 UTC**. Three runs
since: **1041.8 / 699.4 / 667.9s**, all 51/51 ok. That looks like a win against
the ~808s pre-bump median (n=48) and it should not be read as one — n=3, the
window overlaps EIA's recovery, and **the best post-bump run sits inside the
pre-bump range**, whose minimum was 665.6s. Same discipline as
`EVALUATION_POLICY.md`: one window is not a verdict.

**What the phase rollup does settle, and it redirects the next lever.**
`scoring_phase_rollup` on the 667.9s run: **`forecast` is 60.1%** of all worker
time (3085.5s of 5131.0s; slowest BA SPA at 80.4s), then `fetch` 13.0%,
`generation` 11.1%, `model_load` 8.7%, `interchange` 4.0%, everything else under
1%. Effective parallelism is already **7.7×** — 5131.0s of work retired in
667.9s of wall clock. **In-container workers are therefore spent**: raising
`PRECOMPUTE_MAX_WORKERS` again buys nothing without more CPU. #171's `<600s` is
still unmet (best 667.9s) and the two levers that can actually reach it are
cheaper 720h recursive inference — the forecast phase is 60% of the bill — or
more vCPU, which is what fanning across parallel Cloud Run tasks would buy.
That fan-out still needs design first, because `run()`'s fleet-wide steps
(`last_scored`, gate-status merge, benchmark rollup) are genuinely fleet-scoped.

**[#171](https://github.com/kristenmartino/gridpulse/issues/171) reopened 2026-08-05.** It was closed on 2026-07-04 with its
acceptance criterion — a healthy run "well under half the task timeout, target
`<600s`" — **never met**, and it still is not: the best run ever observed is
**667.9s**, 11% over. A closed issue carrying a live unmet criterion is how a
target quietly evaporates, so it now has a home again and a current-state
comment. Note that its own prescription — *"parallelize per-BA work instead of
raising the ceiling"* — has largely happened (4→8 workers, 2→4 vCPU) and was
not enough. Two honest ways to shut it: a sustained sub-600s day with the
mechanism named, **or** an explicit decision that `<600s` was the wrong bar,
replaced by one derived from the real constraint (`--task-timeout ×
(max-retries + 1)` = 3600s against a 3600s hourly window — zero margin).
Retiring the target is legitimate; leaving it unmet and unowned is not.

> **Superseded — [#171](https://github.com/kristenmartino/gridpulse/issues/171)
> was re-closed 2026-08-05T20:23Z, the same day it reopened, by the first of
> those two exits.** The block above is preserved as the reasoning that
> reopened it; read it as history, not as current state. What shut it is in
> the active-focus block at the top of this file: a **406s median over 17
> ticks** on a verified-quiet upstream, which is a population rather than the
> single 667.9s run this paragraph was written against. Flagged 2026-08-11 —
> `gh` said CLOSED while this paragraph still read present-tense, and the
> CLAUDE.md rule is that GitHub wins.

**A note on #389's diagnosis, which was wrong.** A concurrent session, working
**without production access**, attributed the alert to ADR-012 making the ERA5
archive leg ~12× heavier. The runtime record refutes it: daily medians across
the flips (07-22/07-23) are flat — 762/745/768/781/965 before, 838/824/830/826
after — and 07-21, *pre*-flip, has a higher median and max than any post-flip
day until the incident. The archive leg plausibly costs ~50–70s and is worth
reclaiming, but it did not move the median and is not why the alert fired. Its
instrumentation half was correct and is what shipped here; its cross-run
archive cache **merged DARK on 2026-08-05** (`9997d07`, #414) together with the
`fetch` sub-step timing that has to size it — see the recent-decisions entry.

---

**2026-08-04 — #358: backfilled hours cannot supply an as-issued forecast.**
The official arm is `first_seen_df`, documented as "the earliest day-ahead
forecast we observed". For an hour first seen *after* it passed — the seed
backfill, or any reseed — that value is already post-revision, so scoring it
as as-issued collapsed the distinction the dual arm (#341) exists to draw.

Shipped a `stale_capture` drop (lag > `FRESH_CAPTURE_LAG_HOURS`), counted in
the published `excluded_hours`, evaluated **before** the stub rules because it
disqualifies the official arm's *provenance* — so per-reason counts are no
longer comparable to pre-#358 payloads.

**§14 answered with data, not a prediction:** each lead publishes
`stale_capture_impact` — the same hours rescored *without* the filter. The
direction is not uniform (revisions improve some BAs' forecasts and worsen
others'), so a single fleet sentence would be wrong for about half the fleet.

**Not necessarily self-healed:** the issue expected the seed to age out of the
30-day window, but #313 documented vintage windows re-pinned through
2026-07-17 — inside today's window. The API exposes no capture-lag evidence,
so **publishing the count is the measurement**; it lands on the next tick.

One definition of capture lag now: `data.vintage.capture_lag_hours` is public
and imported rather than reimplemented, pinned by a test — the
`OFFICIAL_DOCUMENTED_LEAD_H` lesson applied.

**2026-08-04 — the zonal effect is a COOLING-SEASON phenomenon. First mechanism
to survive; recommend closing the line.**
[`docs/WINTER_RUN_STUDY.md`](docs/WINTER_RUN_STUDY.md).

Pre-registered prediction 2 confirmed — NYISO's gain collapses in winter:

| | summer | winter |
|---|---:|---:|
| **NYISO gain** | **+0.729** decisive | **+0.149** inconclusive |
| NYISO top-down WAPE | 3.958 | **2.237** |
| CAISO gain | +0.283 | +0.015 |

**Mechanism, coherent:** winter error is ~40% lower before any decomposition —
NY heats with gas/oil, so winter *electric* load is far less
temperature-sensitive. Steep summer AC response → zones diverge → splitting
pays. Shallow winter response → nothing to exploit. Same shape as the probe's
finding that diversity mattered in cool/hot bands and vanished in the mid band.

**It does not close the cross-ISO gap:** at matched season NYISO +0.729 vs
CAISO +0.283. Seasonality explains within-ISO variation, not between-ISO.

**Recommendation: CLOSE the zonal line.** The motivating +0.729 exists only in
cooling season, on one ISO, unexplained across ISOs — so the annualised ceiling
is a seasonal fraction of a one-ISO effect. Reopening entry point would be a
third ISO in summer (PJM/ISO-NE, needs credentials a human must create).

**2026-08-03 — component viability fails too. Six mechanisms, six refutations;
the effect is real, replicated, and unexplained.**
[`docs/COMPONENT_VIABILITY_STUDY.md`](docs/COMPONENT_VIABILITY_STUDY.md).

**Not confirmed on either pre-registered criterion.** Folding CAISO's two
negligible areas (MWD 1.0%, VEA 0.4%) into SCE moved the gain **+0.283 →
+0.348** — inconclusive (t=1.375, MDE 0.507), against a required ≥+0.50.

| | CAISO 5 | CAISO 3 | NYISO 5 | NYISO 11 |
|---|---:|---:|---:|---:|
| gain | +0.283 | +0.348 | **+0.745** | **+0.729** |
| pure-load channel | +0.023 | +0.039 | +0.261 | +0.349 |

**Sharpest statement:** the pure-load-decomposition channel works on NYISO and
is **dead on CAISO**, an order of magnitude apart, stable across every grouping
tried on both sides.

**Six explanations, six refutations:** anchor staleness (+0.014), weather
diversity (CAISO more spread, less gain), granularity (5=11), heterogeneity
(CAISO less correlated, less gain), component viability (this run), and lossy
zonal data — CAISO's zone sum vs its own published total is **0.000% WAPE,
ratio 1.0000**, arithmetically exact, so nothing is lost at source.

**Stopping rule now binding — no more regroupings.** Different evidence needed:
a third ISO (PJM/ISO-NE, gated behind free registration, **needs a human**), or
a winter run. **Do not build zonal ingestion.**

**2026-07-31 — granularity is not the driver either. Three mechanisms, three
refutations.** [`docs/SUPERZONE_STUDY.md`](docs/SUPERZONE_STUDY.md).

Pre-registered prediction **failed**: halving NYISO 11 → 5 zones left the gain
unchanged, **+0.745 vs +0.729**, both decisive, both 100% sign consistency.
Five weather points did as much as eleven.

| | NYISO 11 | NYISO 5 | CAISO 5 |
|---|---:|---:|---:|
| gain | +0.729 | **+0.745** | +0.283 |
| verdict | decisive | **decisive** | inconclusive |

At **equal zone count** NYISO gains decisively and CAISO does not. Not granularity.

**Three candidate mechanisms, all contradicted:** weather diversity (CAISO has
more spread, gains less), granularity (5 = 11 on NYISO), component
heterogeneity (CAISO inter-zone corr **0.412** vs NYISO **0.826** — less
correlated, gains less).

**Untested observation that survives:** CAISO's zones are not comparable —
SCE 0.48, PGE 0.42, SDGE 0.09, **MWD 0.01, VEA 0.004**. Two are rounding
errors, each contributing its own model error for no signal. Bottom-up may
need comparably-sized components. **Not tested** — the pre-registration forbade
re-grouping after seeing results, and that is exactly what dropping MWD/VEA
would be. Needs its own pre-registration.

**Still do not build zonal ingestion.** A robust one-ISO effect with no working
explanation and one failed replication is not a foundation. This run's value is
negative: zone count is off the candidate list.

**2026-07-31 — bottom-up does NOT generalise, and the mechanism is
contradicted.** [`docs/ZONAL_GENERALISATION.md`](docs/ZONAL_GENERALISATION.md).

**PJM and ISO-NE could not be tested — both gated.** Verified: PJM Data Miner 2
and ISO-NE web services both **HTTP 401** without a key; four open ISO-NE
static paths **404**. Free registration exists but needs account creation.
**CAISO used instead** (5 TAC areas, open OASIS).

| | NYISO | CAISO |
|---|---:|---:|
| gain | **+0.729** (decisive, 3.7× se) | +0.283 (**inconclusive**) |
| detectable | yes | **no** (MDE 0.334) |
| from load decomposition | 48% | **8%** |

**The weather-diversity mechanism is contradicted.** CAISO has **19.0°F** mean
hourly zonal temp spread against NYISO's **12.6°F** — 51% more — and gets less
than half the gain. If diversity were the driver this should run the other
way. More likely: zone count / load heterogeneity (11 zones incl. a dense
metro pocket vs 5 large utility territories). That is a hypothesis, not a
finding.

**Reconciliation gap replicates:** zone sum vs EIA `D` is 2.70% WAPE (NYISO)
and 3.34% (CAISO) — a general ISO-vs-EIA property, not a NYISO quirk.

**Do not build zonal ingestion yet.** Next: get PJM/ISO-NE keys; then test the
zone-count hypothesis directly by aggregating NYISO's 11 zones into 5 and
re-running — if the gain drops toward CAISO's, the weather story is dead.

**2026-07-31 — bottom-up beats top-down on NYISO. First decisive win in this
line.** [`docs/NYISO_BOTTOM_UP_STUDY.md`](docs/NYISO_BOTTOM_UP_STUDY.md).

| arm | WAPE | gain |
|---|---:|---:|
| top-down (1 model, BA weather) | 3.958 | — |
| bottom-up, **BA** weather | 3.609 | **+0.349** |
| **bottom-up, zonal weather** | **3.229** | **+0.729** |

**100% of 6 windows, 3.7× stderr, satisficing clean, ship=true**, and
**detectable** (MDE 0.390 vs effect 0.729). It also lands exactly where the
probe predicted (0.5–0.8 pts) — a confirmed forecast, not a fishing result.

**The attribution matters more than the headline.** Bottom-up gains eleven
load histories *and* eleven weather points at once. Ablation splits it almost
evenly: **load decomposition alone +0.349 (48%, itself decisive)**, zonal
weather adds the other **+0.380**. Staged adoption is therefore possible.

**The caveat that governs everything:** the target is the **zone sum, not EIA
`D`** — they differ by **2.70% WAPE hourly** (means agree at 1.0003; hourly
ratio 0.94–1.07). That is over half our NYISO error budget. Production
forecasts `D`, so this does not transfer directly. **Reconciling that gap is
strictly prior to any adoption.**

**Next:** reconcile → zonal load only → zonal weather. And run the same
experiment on PJM and ISONE first — the #230 fleet run is a standing reminder
that a handful of BAs can change character at 51.

**2026-07-31 — NYISO zonal probe: 1 of 3 predictions survives.**
[`docs/NYISO_ZONAL_PROBE.md`](docs/NYISO_ZONAL_PROBE.md). 985 hours, 11 zones.

**Weather diversity — supported.** Abs residual by zonal temperature-spread
quintile: 3.37, 3.35, 3.66, 3.27, **4.67**. A threshold, not a line (which is
why corr is only +0.123). **Survives the temperature control** — spread and
level correlate at +0.257, but the within-band effect is cool **+0.512**, mid
+0.061, hot **+0.804**. The mid band showing nothing is what makes it
credible: mild weather means flat load response, so zonal disagreement
shouldn't matter there. Top-spread quintile: 25.6% of error on 20.1% of hours.

**Load mix — fails** (corr +0.085, non-monotonic). **Mix instability — fails**
(corr −0.008, U-shaped).

**Caution the finding owes:** 1.27× concentration is the *same shape* that
produced two consecutive failures — hot hours were 1.43× and both the cooling
pack and BTM died on it. Concentration says where error is, not that it is
addressable.

**Next step is one BA, not seven integrations:** bottom-up (11 zonal models +
zonal weather, summed) vs top-down on NYISO, through the existing harness. If
diversity really costs us, bottom-up should recover the 0.5–0.8 pts.

**2026-07-31 — ISO real-time feeds: the anchor case is dead. My reasoning was
wrong.** [`docs/ISO_REALTIME_FEEDS.md`](docs/ISO_REALTIME_FEEDS.md).

I recommended these across several turns on the premise that "EIA-930 is
hourly and lagged" and that a fresher anchor would help. Measured:

| | claimed | measured |
|---|---|---|
| EIA publication lag | "lagged" | **1.7h, identical across all 51 BAs** |
| trailing stub hours | implied material | **median 0** (only ERCOT, 6h) |

**And staleness costs nothing anyway.** Holding scored hours fixed and varying
only anchor age: **2h costs +0.014 pts (median −0.036)**; even **24h costs
+0.087**. The 0–8h range is noise around zero, several entries negative. At a
24h horizon the forecast is driven by weather and calendar, not recent demand.

**A correction inside the correction:** my first probe reported 19.7h median
staleness. That was my own bug — EIA's `D` and `DF` cover different ranges, so
every future forecast-only hour was counted as a stale actual. The "18h" was
the forecast horizon.

**What ISO feeds verifiably offer instead** (fetched live, auth-free): NYISO
publishes 5-min load *and* day-ahead forecasts **per zone** — 11 zones where
EIA-930 gives one number. **Zonal decomposition is the real differentiator,
not freshness** — a different hypothesis from the one I argued, and untested.

Cost of that path: ~7 of 51 BAs (but ~62% of fleet MW error), seven separate
integrations, zonal weather needed to match, and ERCOT gates its modern API.

**Recommendation: do not build the ingestion I was recommending.** If ISO data
is pursued, probe NYISO zonal structure against BA-level residuals *first*.

**2026-07-31 — #230 fleet test: the pre-registered rule is NOT confirmed.**
[`docs/DIRECT_MULTIHORIZON_FLEET.md`](docs/DIRECT_MULTIHORIZON_FLEET.md).
51 BAs × 6 windows, 306 paired observations, zero skips.

| criterion | result | |
|---|---|:--|
| hard pool positive **and decisive** | mean +1.589 but wins only **54%** of windows | **FAIL** |
| easy pool not positive-decisive | −0.029, noise | pass |
| separation ≥ 0.5 pts | +1.618 | pass |

Criterion 1 fails on **reliability, not size** — 3.6× stderr on magnitude, but
the hard pool's median is **+0.198 against a mean of +1.589**. The average is
carried by a minority of windows.

**Fleet: 25 better / 26 worse. Median Δ = −0.002 pts.** The +0.700 mean is 55%
attributable to 2 of 51 BAs (NWMT +11.0, NYISO +8.7). 11 decisive for direct,
5 for recursive, 35 inconclusive.

**Only 17 of 51 per-BA results are detectable at 6 windows** (median MDE 0.90
pts), which is why the pre-registration moved the unit to the pooled test
(MDE ≈0.21).

**Per-BA adoption is not the fallback.** The 9 that ship were selected
*because* they won; choosing and justifying on the same data is the
cherry-pick the harness exists to stop. Legitimising it needs out-of-sample
validation — a new pre-registration, not a reinterpretation.

**The pre-registration earned its keep.** Without it, "hard pool +1.589 at
3.6× stderr, separation 1.6 pts" reads as a win.

**2026-07-30 — #230 direct multi-horizon: not a rewrite, possibly a per-BA
choice.** [`docs/DIRECT_MULTIHORIZON_STUDY.md`](docs/DIRECT_MULTIHORIZON_STUDY.md).

10 BAs × 5 windows at 168h. **One decisive win (PJM +1.124) and one decisive
loss (SPP −1.206) that cancel; 5 better / 5 worse; 1 of 10 ships.** Mean +0.646
vs median +0.091 — the outlier-domination gap the harness flags.

**The conditional signal is the real finding:** direct helps where recursion is
already struggling. corr(mean-of-arms WAPE, Δ) = **+0.737** on a symmetric
axis; harder half mean **+1.350** (4/5 better), easier half −0.058 (1/5).
Exactly what error accumulation predicts. **But the threshold was chosen after
seeing the data, and SPP is a decisive counterexample inside the harder half.**

**Bias is why it is not free:** direct −1.848% vs recursive −0.636% mean — it
under-forecasts ~3× more, the expensive direction.

**Recommendation:** don't rewrite; don't discard. Next test is a 51-BA run with
the threshold *pre-registered*, and if it survives, per-BA strategy selection
rather than a rewrite — ADR-010's serve-path gate is already shaped to carry it.

**Process note worth keeping.** The first version sampled 14 training horizons
and let `horizon_h` interpolate; trees split rather than interpolate, so
unsampled horizons were mis-served. That version measured **−0.757** on PJM.
Training on all 168 flipped it to **+0.754** — a 1.5-point swing from my own
sampling choice, which would have rejected #230 on an implementation artifact.

**2026-07-30 — BTM solar: not supported. Two of three predictions fail.**
[`docs/BTM_SOLAR_PROBE.md`](docs/BTM_SOLAR_PROBE.md). Probed residuals rather
than building features — the explicit lesson from the cooling pack.

**Prediction 1 (sign) fails, and it is decisive.** Unmodelled rooftop PV means
*over*-forecasting at high irradiance. Observed: signed error is positive at
top-quintile irradiance in only **3 of 8** BAs, and the mean daylight signed
error is **−0.636%** — we systematically *under*-forecast in daylight, the
opposite direction. No external data needed to read that.

**Prediction 3 (dose-response) fails:** corr(BTM rank, gradient) = **+0.14**,
and CAISO (highest penetration) shows +1.85 while FPL (second) shows −0.93.
Weak evidence though — the ranking is my own guess, not EIA-861M.

**What the residual actually is:** we under-forecast worst on **hot,
low-irradiance** hours — muggy, high-cooling-load, no solar relief. Opposite
shape to a PV story.

**The premise itself checked out.** Hot hours carry **24.3% of MW served** but
**34.7% of MW error** — **1.43×**, so genuinely harder, not just bigger, in 7
of 8 BAs. (FPL is 1.04 — there it *was* just load size; drop FPL from hot-hour
work.)

Hot-hour cause now: temperature representation **rejected**, BTM solar
**rejected**. Untested: demand response, and the economical explanation that
trees already learn these interactions.

**2026-07-30 — cooling-response features: measured, rejected, and the negative
result is the useful part.** [`docs/COOLING_RESPONSE_STUDY.md`](docs/COOLING_RESPONSE_STUDY.md).

Built the pack the error analysis pointed at — CDD accumulation (24h/72h),
CDD², NWS heat index, CDD×humidity, all from weather we already fetch. Ran it
against control over 6 rolling windows on the 8 addressable BAs.

**8 of 8 inconclusive. Mean effect −0.0033 WAPE pts. 6 of 8 slightly worse.**
Flag `cooling_response_features` stays **off**.

**The informative part:** both arms had *perfect* future weather. If explicit
accumulation, convexity and humidity terms cannot reduce hot-hour error when
temperature is known exactly, the hot-hour error is **not a
temperature-representation problem**. The error analysis was right about
*where* and wrong about *what*.

What survives as hypotheses, untested: **behind-the-meter solar** (hot
afternoons are peak-irradiance afternoons; rooftop PV suppresses net load
exactly at cooling peak and nothing in the feature set knows it exists), demand
response/curtailment, or simply that trees already learn these interactions.

**ISONE is the one row that moved** (+0.328, 83% sign consistency) — and it is
exactly where the analysis predicted the biggest payoff. Still below the bar;
picking it from eight would be the cherry-pick the harness exists to stop.
Follow-up, not a result.

**2026-07-30 — error analysis: the error is not where the scorecard points.**
[`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md). The step that had been
skipped for months — bucket the errors, size the buckets.

**In MW across the fleet we are 8.5% BETTER than the incumbents** (16.46M vs
17.99M MWh), the opposite of the published headline (operator closer on 27 of
43). Both true; only one is about megawatts. Top 10 BAs carry **77.5%** of all
fleet MW error. **SEC — several sessions of work — is 0.30% of it.**

The addressable gap (28 BAs where we lose) is 3.62M MWh/mo, **82% of it in 8
BAs**: MISO, ERCOT, ISONE, NYISO, PJM, TVA, FPL, SOCO.

Per-hour on those 8: **19–59% of our error is on hours the operator got
right**, with our arm given *perfect* weather — so not weather error, and
addressable.

**The one clear axis is temperature.** Hot quintile carries a mean **34.7%**
of error vs **11.9%** cold, monotone in 7 of 8 BAs. **Holidays** are 1.87×
over-represented in ISONE, 1.69× NYISO — against a single binary `is_holiday`
flag. **Ramp magnitude and hour-of-day are flat** — no signal.

Next, by evidence: cooling response (biggest bucket everywhere), holiday
features for the northeast BAs, ISONE specifically (worst on every axis).

**I was wrong to deprioritise [#230](https://github.com/kristenmartino/gridpulse/issues/230)** — this
reconstruction is a *direct* day-ahead model and beat production's live
numbers (PJM 2.72 vs 4.10, MISO 2.85 vs 3.41). Confounded by perfect weather,
but large enough to test rather than dismiss.

**Caveat that matters: summer only.** June–July windows. The hot-quintile
dominance may be seasonal — re-run in January before treating cooling response
as the year-round answer.

**2026-07-29 — evaluation policy: rolling origin, and we stopped optimising MAPE.**
Prompted by asking what to do next and getting an uncomfortable answer: the eval
harness could not support the decisions being made with it.

`models/rolling_eval.py` + [`docs/EVALUATION_POLICY.md`](docs/EVALUATION_POLICY.md).

**The proof.** CAISO re-run over 8 rolling windows gives per-window deltas
`-0.43 +2.31 -0.20 0.00 -6.93 +4.30 +1.57 -0.30` — mean +0.04, median −0.10, a
wash. The two numbers previously published (−7.24, +3.87) are **the two
extremes of that distribution**. The harness returns *inconclusive*.

**Metric change: optimise WAPE, publish MAPE.** MAPE grows without bound for
over-forecasts but caps at 100% for under-forecasts, so minimising it biases
toward **under**-forecasting demand — the expensive direction for a grid. It
also explodes on low denominators (SEC, ~300 MW) and cannot aggregate across
BAs meaningfully. MAPE stays the published number for comparability with EIA /
ISOs / vendors, protected as a constraint rather than optimised.

Satisficing vetoes a win: |bias| ≤ 2%, MAPE regression ≤ 0.5 pts, and an
*unmeasurable* constraint counts as failed.

**Incidental find worth pulling on:** across those 8 windows the control arm —
what production serves — under-forecasts CAISO by **−2.83%**. Dangerous
direction, and exactly what the new bias constraint is for. One BA, so not yet
a mechanism.

**Open question:** the serving gate still grades on MAPE, so a WAPE-optimising
experiment can disagree with it. Migrating the gate is the fix; not done.

**2026-07-29 — #297: the dead kwarg was load-bearing. Fix rejected on evidence.**
`_auto_select_order` passed `exogenous=` (pmdarima 1.x) where 2.x wants `X`;
`auto_arima` takes `**fit_args`, so it was swallowed and the order search ran
univariate while the fit used all five weather regressors.

The issue's own fix sketch said `exogenous=` → `X=`, then measure. **Measured
across all 51 BAs (2026-07-29, 0 failures), it is the wrong fix.**

Not because it loses everywhere — 18 improve, 20 worsen, **13 select the same
order either way**. Because of the asymmetry: losses total **−61.7 sMAPE pts
against +14.9 gained**, mean −0.92, worst single BA **−19.18** (ISONE 13.93 →
33.11), at 2.7× the search cost. A much heavier left tail for no expected gain.

**Mechanism, fleet-validated:** harm concentrates where the search drops the
**seasonal MA term** (Q 1→0) — those 14 BAs mean −2.99 vs −0.21 for the other
24. Specifically the MA term, not seasonal complexity generally (bucketing on
total P+Q separates almost nothing). The study also fed the losing arm
*perfect* future weather.

**Correction to the 10-BA run PR #365 shipped on:** per-BA numbers from a
single 168h window do not reproduce. One day of window shift reversed CAISO
(−7.24 → **+3.87**), evaporated WALC (+4.59 → +0.04), and halved MISO's
*control* sMAPE (7.62 → 3.91). The aggregate and the mechanism held; the
individual rows did not. The study now stamps its window, and per-BA figures
should be read as one draw.

Shipped: keep the behaviour, delete the lie — no dead kwarg, no `X`, numbers
at the call site, and tests that pin the *class* (every kwarg must exist in
the installed signature) plus a guard that fails if someone re-applies the
obvious fix. Study: `docs/ARIMA_ORDER_EXOG_STUDY.md`.

**Operational find:** `_auto_select_order` runs only on a **cold cache** — the
training job reads the previous model's persisted order, skips the search, and
re-persists it, with **no invalidation path anywhere**. Every live order was
selected univariately. Any future change that needs fleet-wide re-selection
must build that invalidation first; nothing provides it today.

**2026-07-28 — #348: the benchmark row that knew it was bad and didn't say so.**
Every unflattering fact on `/benchmark` is published deliberately except one:
a row our own drift monitor already grades `rollback` was rendered as an
ordinary comparison. SEC publishes **17.64%** against the operator's 8.14%
while we grade its ensemble `rollback` at 24h.

Shipped two markers, both payload-derived (no region list, no new capture):
`serve_grade` — the grade for the **exact** series that row scores, same
model and same lead — and `served_series` / `serves_scored_model`.

**The second one was the sharper find.** This arm always scores the
*ensemble*, and SEC is served the *seasonal-naive baseline* — so SEC's
published number is **worse than what it actually serves**. The arm is
deliberately not re-based onto the fallback (that would stop it measuring the
forecaster), which makes it a disclosure rather than a scoring change. Limit 6
of the methodology asserted "the GridPulse arm is the served ensemble"; that
was true when written and isn't any more, and is corrected.

Also corrected: a grade earns `rollback` by exceeding the **acceptable**
threshold (7.0 at 24h), not `MAPE_BY_HORIZON`'s `rollback` entry (12.0) —
`mape_grade` never uses that as a boundary. Draft page copy quoted 12.0, as
does the issue text. The applicable figure now ships as `acceptable_max`.

**2026-07-28 — #349: the gate's bar did not move, and that is the finding.**
The complaint was real — `is_forecast_quality_acceptable` grades the
**training holdout** against the **7-day** band (22%), while the number we
publish is 24h on the serve path. SEC passed at 6.96% holdout with all four
models `rollback` at 24h (ensemble 12.2, arima 16.9, xgboost 25.2, prophet
34.0).

**Measuring the blast radius before touching it changed the design.** The
gate hides **0 of 51** regions today; re-grading it on the 24h band would
hide **7** — SPA 25.3, SEC 12.2, IID 11.4, AZPS 9.6, WALC 7.7, LDWP 7.5,
CPLE 7.1 — and three of those sit within **0.7 points** of the threshold,
i.e. inside the tick-to-tick movement of a rolling window. They would
flicker in and out of the product on noise.

So the generous question stays the gate (hiding a BA is heavy-handed), and
the sharp question gets published instead of being absent: `live_horizon`
on the verdict, `operating_horizon_grade` + an explicit
`quality_gate_measurement` on `/api/v1/regions`, and a
`gate_live_horizon_disagreement` warning every tick a region passes the gate
while failing at 24h. **SEC stays visible and is now flagged**, which is
also the standing evidence for its baseline substitution.

Caveat worth keeping: for a substituted region the live grade describes the
**models**, not the served series.

**2026-07-28 — #313 closed: the vintage/drift window corruption stopped on
2026-07-17 and has not recurred.** The defence shipped in #313/#320 made the
anomaly *observable*; 13 days later it has been observed **zero** times —
`vintage_window_missing_but_seeded`, `horizon_drift_history_read_failed` and
`drift_history_read_failed` are all 0, as is any Redis error line. Query
shape validated against live events first (`baseline_substituted` = 24 hits
in 2 days, exactly one region hourly since the flip), because a mistyped
filter also returns zero.

**The unchecked audit item came back positive:** drift windows really were
wiped — **8 reset ticks, 15 window wipes** across CAISO/ERCOT/FPL/PJM
between 07-16 02:00 and 07-17 09:00, every one a drop to exactly 0. #320
landed 07-17 14:23 UTC; nothing since, through 07-28.

**New fact that sharpens the trigger, if it ever recurs:** half the reset
ticks hit *several regions at the same instant* (three-region wipes at
02:00:46, 12:00:55, 09:00:53). The issue framed the victims as "the
earliest-fetched regions", implying a per-region property — the simultaneity
says it is a property of the **tick**. Something returns nil for several
distinct keys at once in the first seconds of a run, then the same
connection serves the other 47 regions fine. Every reset is 46–68s past the
hour. Consistent with the cold-start init race found during #312.

`MARKET_POSITION.md` called this the top risk to the most differentiated
asset, "being corrupted by an unknown trigger" — that was read off the
issue's open state rather than evidence, and is corrected in the same PR.

**2026-07-28 — market position written down, and it inverts the pitch**
(internal note, not published). Triggered by a look at [orreryhq.com](https://orreryhq.com).

**Orrery is not a competitor today** — they sell derived *weather* by the call
(live API serves NOAA GFS 0.25° only; the advertised ECMWF/HRRR/NBM/CAMS are
not served yet), from a ~30-day-old domain, apparently one engineer, no
funding, no prices published despite marketing "published prices". They sit
at the weather-ingestion layer, closer to an Open-Meteo substitute than a
rival. **The watch item is real though:** `load`/`lmp`/`wind_generation` are
first-class in their schema and `/v1/energy/load` is a live BA-keyed route
returning `series: null` — declared surface, no data. They also independently
converged on our skill baseline ("24-hour persistence"), the same week.

**The honest position:** "we forecast demand better" is not sellable — our own
public scorecard contradicts it on **five of six major ISOs** (ERCOT 2.48 vs
1.44, NYISO 5.25 vs 2.06, ISONE 7.90 vs 3.43). We win big on small weak-forecast
BAs (PSEI 3.59 vs 40.9, FMPP 5.52 vs 28.15). **The wedge is a floor, not a
ceiling** — and the buyers with budget are on the ISOs, where we lose.

**What's differentiated is the measurement apparatus, not the forecast.**
Ranked: the vintage instrument is a real barrier but only a 30-day rolling
window (head start, not a moat); ADR-010's gate is hardest to arrive at
independently but fully published here; the benchmark methodology is cheap to
copy technically and expensive institutionally (nobody publishes a scorecard
they lose). Publishing accuracy vs the incumbent is **table stakes** — Enverus
and Amperon both do it. The white space is publishing *continuously,
pre-sale, including losses*.

**Recommendation is neither "sell A" nor "sell B" but keep measuring:** the
benchmark is 2 days old, skill 1 day, substitution flipped today — and the
asset with the shortest shelf life is the one a competitor starts matching 30
days after they begin.

**2026-07-28 — baseline substitution flipped ON; SEC now serves the
seasonal-naive baseline.** Flag `baseline_substitution` → True after
shadow-running the live decision across all 44 regions scoreable on that
date. (That count is a measurement, not a constant — see the #535 entry.)

**Shadow result: SEC alone**, at **−4.03** error points against the −2.0
bar (model 12.59% vs naive 8.56% on the trailing 7 days). 43 regions keep
their model; 0 were unmeasured. Stable across every window with enough
hours to decide — −2.88 at 5d, −4.03 at 7d, −2.79 at 10d — while the
nearest other region (FPC) sits at −1.68 and never clears the bar.

One apparent disagreement in the stability run was the policy working: the
5-day window substitutes *nothing*, not because SEC's margin fails but
because 120 hours is below the 168-hour minimum the policy requires before
it will decide at all. The guard fired exactly as intended.

**Disclosure is on both surfaces before the flip, deliberately.**
`/api/v1/forecast` reports `series_source: "seasonal-naive-baseline"`; the
dashboard resolves through `_served_model_for_payload` to a label that is
not a model name, so title, trace and caption all say so, and the caption
carries the scoring job's own reason. Per-model rows stay intact as the
evidence.

**Post-flip watch:** `baseline_substituted` log lines should name SEC and
only SEC each tick; the Forecast tab for SEC should title "Seasonal-Naive
Baseline Demand Forecast"; and SEC's benchmark row should move toward the
naive number (~8–11% rather than ~18%) as substituted hours accumulate in
the drift window. Rollback = flip back — the substitution is a read-time
swap with no persisted state.

**2026-07-28 — baseline substitution built, ships DARK; flip after shadow
verification.** Where a model measurably loses to "yesterday, same hour",
the honest thing to serve is the free thing. Flag `baseline_substitution`
→ False; flag-off is byte-identical to today and pinned by a test.

**A window mismatch nearly inverted this.** The first cut compared a
30-day baseline against a 7-day model and concluded we *won* SEC at 48h
(+3.36) and 72h (+2.75) — I was about to report that substitution wasn't
supported. On matched 7-day windows the baseline wins at **every** lead:
24h **8.56 vs 12.54**, 48h **9.55 vs 10.57**, 72h **8.01 vs 11.21**. The
helper now measures both sides on one window and carries a comment saying
why, because that error is one line away at all times.

**Policy, not a region list.** Substitution needs the baseline to beat the
model by ≥ 2.0 error points over ≥ 168 measured hours. Today that selects
SEC alone (−3.98 pts) and leaves the eight regions within ~1 point of the
line untouched. Every failure path keeps the model — flag off, no skill
signal, no drift record, thin history, unusable projection, any exception
— because the model is right on 35 of 44 regions and a bug that
substituted wrongly would replace all of them.

**Disclosure is part of the change, not a follow-up.** Payload carries
`served_series` / `served_reason` / the skill block; `/api/v1/forecast`
reports `series_source: "seasonal-naive-baseline"`, never `"ensemble"`.
Per-model rows stay intact so the evidence sits next to the decision.
4 assert-applied mutations killed (substitute-when-fine,
substitute-when-unmeasured, wrong source day, flag ignored).

**Next: shadow-verify the decision set is SEC alone across a few ticks,
then flip.** The UI still needs to show the substitution — the dashboard
reads `predicted_demand_mw` and would currently present a baseline series
without saying so, which is the one gap left in this arc.

**2026-07-28 — SEC isn't a bad region, it's a region where our model is
worse than free.** Investigating the benchmark's worst row produced a
finding no existing instrument could have surfaced, because none of them
measured skill against a baseline. Measured
([`docs/PERSISTENCE_SKILL.md`](docs/PERSISTENCE_SKILL.md), re-runnable):
**SEC's served ensemble reads 17.8% against a seasonal-naive "yesterday,
same hour" baseline's 11.5%** — 6.36 points of *negative* skill. A one-line
predictor with no model, no weather and no training beats three trained
models and their ensemble.

**It is not the fleet.** 35 of 44 BAs beat the baseline, median +0.83 pts,
best +3.50 (NYISO). Eight others sit within ~1 point of the line, which is
noise at this sample size. SEC is 6× the next worst.

**And it is not what I first diagnosed.** The error is *flat across
horizon* — 10.38% at 1h, 12.59% at 24h, 10.55% at 48h, 11.23% at 72h.
Compare NYISO at 1.70% → 5.84%, which is what recursion actually looks
like. A 10% error one hour ahead, with the actual known to the previous
hour, means the model never tracks this load at any lead. So it is **not**
recursion drift, **not** an anchoring problem, and **not** fixable with
more weather features — my earlier ensemble-weighting theory (ADR-004
cutting XGBoost to 3%) was wrong about the mechanism, and the ensemble at
13.8% already beats every member (ARIMA 18.8, XGB 26.1, Prophet 37.2).
SEC's load is simply the most volatile sampled: cv 33%, peak/trough 2.38,
median hour-over-hour step 7.54% against 1.7–5.0% elsewhere.

**Shipped: the missing primitive.** `models/skill.py` +
`scripts/persistence_skill.py`. Skill vs a naive baseline is the minimum
bar a forecasting product clears, and nothing here measured it — which is
how a worse-than-nothing forecast stayed invisible behind a 6.96% holdout.

**The serving decision is open and is the actual "fix":** serve the
baseline for negative-skill BAs (SEC 17.8% → 11.5%), gate them, or
disclose on the benchmark page ([#348](https://github.com/kristenmartino/gridpulse/issues/348)).
Recommend serving the baseline where skill is negative by a real margin,
with hysteresis so it can't flap — but that changes what users see, so it
is a product call. Continuous per-tick skill publication is the natural
next PR either way.

**2026-07-27 — the benchmark returned its first real numbers, and the
incumbent wins.** Across 43 scoreable BAs at the 24h headline, **the
operators' own day-ahead forecasts are closer on 28, GridPulse on 15**;
median mean-MAPE **theirs 3.80% vs ours 4.82%**. The 48h conservative arm
(label now genuinely earned — observed 47.82h > their documented 41h max)
does not rescue it: 14–30, 3.82% vs 5.16%.

**This is not an instrument fault, which is what makes it credible.** Our
median here is 4.76–4.82% against a published holdout claim of 4.8% — the
benchmark reproduces our own known accuracy exactly, and the operators are
simply better than it on most BAs. Instrument health is clean:
`lead_basis` is `observed` on all 44 (so #342's producer fix is live),
leads vary 22.74–23.94h per BA, 44 distinct drop-count combinations, no
verdict flips under the as-revised scoring, exclusions match the
scoreability report exactly.

**What the data does support**, and it is narrower than what this file
used to claim: not more accurate on a typical BA — **more consistent**
(our spread 8.3×, 2.18–18.03%; theirs 23.4×, 1.76–41.32%) and dramatically
better where an operator forecasts its own load poorly (PSEI 41.32 → 3.46,
FMPP 28.15 → 5.42, FPC 22.15 → 6.24). The page now states this result in
words above the first table, generated from the payload, so a one-line
skim can't stop at a scoreboard and a future run can't leave a stale boast
behind.

**SEC is the open technical item.** Our worst row at 18.02% vs their
8.24%, confirmed by every metric (median APE 13.47 vs 5.49, WAPE 17.22 vs
9.10) and by a second instrument — live horizon drift grades *every* SEC
model `rollback` at 24h. Mechanism: SEC's XGBoost holdout is 21.3% with
**R² = 0.085**, so the sharpened inverse-MAPE weighting (ADR-004) cut it to
**3%** and the ensemble runs on ARIMA 52% + Prophet 45% — both carrying
open defects (#297 ARIMA univariate, #299 Prophet seam-step). Root cause
looks like fit, not code: a ~308 MW generation co-op whose load follows
member scheduling, which our weather/calendar feature set cannot see.
**Governance gap — now [#349](https://github.com/kristenmartino/gridpulse/issues/349):** SEC reads `quality_gated=False`
because the gate judges the *holdout* champion (6.96%) against the **7d**
rollback band (22%) — wrong measurement (holdout, not serve path — the
ADR-010 blindness again, live is 2.6× holdout) and wrong horizon (a 7d
tolerance while our own drift grades it rollback against the 24h band of
12%). The disclosure half — `/benchmark` publishing a 24h scorecard for a
BA our own drift already grades `rollback`, with no marker — is
[#348](https://github.com/kristenmartino/gridpulse/issues/348).

**2026-07-27 — E0-3 shipped: the public benchmark is live-able.**
[`/benchmark`](web/benchmark.html) + `/api/v1/benchmark` (+ per-region).
The page holds no data of its own — it fetches the public endpoint in the
browser, so it cannot render a figure the API would not also return. It
publishes both official arms and **both** verdicts (an amber pill where a
revision flips a BA), per-row drop counts, fairness exclusions with their
reason and the direction of the bias, and the limits led by the two that cut
against us. API follows the existing contract: allow-listed export, 503
`{"status": "warming"}` when cold, per-BA `scored_at`.

An 8-agent adversarial review confirmed **17 defects**, all fixed. The two
that mattered: the excluded table lumped *fairness* exclusions together with
BAs merely **still accumulating hours**, under a lede claiming "most are
broken feeds" — false at launch, when the second group is larger, and it made
a young BA read as a hidden loss (now two sections, split on the published
reason). And the page said "their forecast", the exact phrasing §12.1
forbids — now "the earliest day-ahead we observed" throughout. Also: an
invisible focus ring (30%-alpha token, ~1.5:1, fails WCAG 1.4.11 — the same
bug was in `/about`, fixed in a follow-up), missing `aria-sort`, BA cells
promoted to `th[scope=row]`, and a loading placeholder that read "Loading…"
forever with JS off.

**E0 is complete: engine → measurement → methodology → public surface.**
Next: deploy and watch the first real payloads (the conservative label and
`observed_lead_h` only start populating once #342's fix is live), then decide
whether the `/benchmark` page is promoted from a side path.

**2026-07-27 — E0-4 methodology published, and writing it found two live
bugs in the lead instrument.** [`docs/BENCHMARK_METHODOLOGY.md`](docs/BENCHMARK_METHODOLOGY.md)
states the rules — sources, the single-truth discipline, the five hour-drop
rules and their bias direction, the exclusion tests, the dual official arm,
lead-time handling, metrics, windows, fleet aggregation, what the benchmark
is *not*, eight known limits, and a rule that any future change to the
scoring rules must state which direction it moves our own number. Numbers
stay in the generated artifacts so the doc can't go stale.

A 12-agent adversarial review of the draft confirmed **33 defects**, and the
two worst were mine repeating a number without checking its provenance: the
"median 649 / min 500 **paired** hours" sizing is actually *officially
scoreable* hours — the count before the `no_gridpulse` join — an error that
originated in `MIN_PAIRED_HOURS`'s own comment in #340 and was fixed in both
places; and a SOCO example meant to teach "always name your metric" itself
compared *their* median APE over 30d against *our* mean sMAPE over 7d, four
axes apart, using a 5.82% figure no artifact carries. Also added: the two
limits that cut in our favour — the headline arm is **not lead-matched**
(our ~23.9h vs their documented 17–41h, midpoint ~29h), and `no_gridpulse`
conditions the hour set on *our* availability, dropping hours the operator
did forecast. And §10's "ours is comparatively flat" is now labelled a
hypothesis: no committed artifact publishes our spread yet.

Writing it forced a read of the code rather than the code's docstrings, and
`_observed_lead_hours` was wrong twice: it read the **API's** key
(`forecast`) off the **Redis** payload (`forecasts`), so it returned `{}`
every tick — the conservative label was being withheld fleet-wide and
`lead_basis` never left `"nominal"` — and it measured row index H−1 instead
of row 0 + H, the hour the drift snapshot actually targets, understating
every lead by exactly 1h. Both failed conservative, which is why nothing
looked broken. **The consumer was well tested and the producer was not**: a
producer returning `{}` is indistinguishable from a BA with no forecast yet.
Fixed, pinned by five producer tests including a cross-module invariant
against `snapshot_horizon_predictions`, and both original bugs reproduced as
assert-applied mutations. Corrected leads: nominal-24h realized
**23.80–23.95h**, nominal-48h **47.80–47.95h** (probe re-run, artifact
regenerated) — the conservative claim holds by a wider margin than reported.

**2026-07-27 — E0 measurement pass: both provenance gates measured, both
pass; publication unblocked** (PR
[#341](https://github.com/kristenmartino/gridpulse/pull/341),
`docs/BENCHMARK_PROVENANCE.md`, re-runnable via
`scripts/benchmark_provenance_probe.py`).

*Gate 1 — does EIA revise DF after we bank it?* Yes, unevenly.
PJM/MISO/ERCOT/CAISO/GVL/SPP/NYISO revise **0%**; SOCO 24.2%; PSEI 26.4%
(max Δ 34%); FMPP 5.2%, where revision makes them *worse*; fleet 6.78%.
Largest movement in any operator's own **median** APE is **1.43 pts** (PSEI
47.15% as-issued → 45.71% as-revised) — which bounds no head-to-head
verdict, since those are decided on *mean* MAPE and the probe never measures
it. The payload publishes `winner_vs_revised` per BA rather than asserting
the two agree. *Gate 2 — what lead do we actually forecast at?* A
nominal-24h record is a realized **23.80–23.95h**, so no "24 hours ahead"
claim ships unqualified; the nominal-48h arm's minimum **47.80h exceeds
the operators' documented 41h maximum**, so publishing it as the
*conservative* comparison is measurement-supported rather than assumed.

Encoded, not just documented: the official side is now scored **twice on
the same hours and the same settled truth** — as-issued (fair, primary)
and as-revised (conservative, since post-hour revision carries
hindsight) — with both verdicts published and any revision-driven flip
named rather than hidden; and the conservative label on the 48h arm is
**earned per tick** from the realized lead, lapsing automatically if
EIA's publishing lag grows. Known limit, stated everywhere: the probe
cannot see revision *before* our first capture, so the phrasing is always
"the earliest day-ahead forecast we observed." **Next: E0-4 methodology
doc, then E0-3 public page + `/api/v1/benchmark`** — every claim they
need now has a measurement and a script behind it. Caution the
measurement surfaced: per-BA figures move with metric, window AND arm —
SOCO's *own* forecast reads 1.84% median APE / 30d, while an indicative run
of *our* ensemble on the drift instrument read 5.82% mean sMAPE / 7d; four
axes differ, so the two were never comparable. The public page must carry
metric, window, `n` and arm on every row.

**2026-07-27 — E0 benchmark engine landed (PR 1 of the arc); two
provenance limits found by design review must be closed before anything
publishes.** *(Both closed by the measurement pass above.)* The gate epic for commercialization: GridPulse competes
against a *free* incumbent (EIA-930 publishes each BA's day-ahead
forecast), so relative accuracy is the value proposition. Engine rides
existing instrumentation — official arm from vintage `first_seen_df`,
GridPulse arm from `drift_horizon` 24h/48h records, settled `last_d` as
the single truth for both. Measured **as of 2026-07-27**: 44 of 51 BAs
scoreable, operators' own accuracy spanning 41× (ERCOT 1.15% → PSEI
47.21%) — content no incumbent publishes. **Both figures moved, and the
first one moved because of a bug** ([#535](https://github.com/kristenmartino/gridpulse/issues/535));
the live count ships as `n_scoreable` on `/api/v1/benchmark`, and the
spread is a property of whatever population is scoreable that day.

**Two limits block the public claim, both documented in
`models/benchmark.py`:** (1) `first_seen_df` is *not* the day-ahead value
as published — vintage only admits an hour once EIA publishes a metered
`D`, so the DF was re-read 0–3h after the target hour, and nothing yet
measures whether EIA revises DF in between; (2) our lead is *nominal*, not
realized — the forecast anchors on the last real demand hour, so a "24h"
record is a ~20–24h wall-clock lead, and the resolved drift records
discard `made_at`. Neither may be published as a claim until measured.
**Next: a DF-revision measurement week, then the realized-lead capture** —
both gate the methodology doc (E0-4) and the public page (E0-3).

~~Indicative sizing (10 BAs, approximate metric match): GridPulse 6,
official 4… we are ~3–5% everywhere.~~ **Superseded by the real
measurement on 2026-07-27 — see the entry at the top of this file. The
indicative run had the win count backwards and the "~3–5% everywhere"
range was wrong (our real range is 2.18–18.03%).** Left struck through
rather than deleted: it is the estimate the epic was planned against, and
how far off it was is the point.

**2026-07-23 — ADR-012 flipped ON: 36 BAs now forecast on aggregated
footprint weather.** Flag `multipoint_weather` → True (PR B of the #336
arc; the issue — auto-closed early by a close-keyword written in PR A's
prose, the documented CLAUDE.md trap and its *second* occurrence this
session after #332, reopened via API — closes properly with this PR).
Post-flip watch: `weather_multipoint_aggregated` on the 36 multi-point
BAs (the 15 compact ones correctly silent), zero
`weather_multipoint_shape_mismatch`, ADR-010 serve gate green at the next
04:00 Z training, and MISO/PJM/SPP live sMAPE descending toward the
predicted +1.4–1.8. Rollback = flip off (byte-identical).

**2026-07-22 — ADR-012 multi-point weather built, shipped DARK; flip PR
next.** The study's ADOPT verdict implemented as the measured arm,
weights-free: `assets/multipoint_coordinates.json` (36 BAs × up to 12
footprint cells, generated offline; the 15 compact BAs omitted → single
point), `data/weather_aggregate.py` (circular mean / mode /
**renormalizing nanmean** — a deliberate divergence from the study's
`nansum`, which counted a null point as zero and would have dragged
values toward zero in production), and multi-point fetchers that **fail
open at every seam** to the untouched single-point path (the #161
lesson). NBM composites **per point** before aggregation — its
null-keeps-base rule is per-cell, so aggregate-then-overlay would be
wrong. Flag `multipoint_weather` ships False. Live dev verification:
MISO 12 points → NBM per-point (34,452 = 12 × 2,871) → aggregated, 0
nulls, schema identical, 1.64 °F mean delta vs single-point; GVL
(compact, absent) exactly 0.00 °F. Three mutations killed (null-drop
reverted, flag removed, archive retry removed). Next: deploy, shadow
verification, then the flip PR closes #336.

**2026-07-22 — Multi-point weather study: ADOPT, and it's simpler than the
literature said.** Research rank 2 (MISO's fix — the one BA that LOST
−0.33 in the NBM study). `scripts/multipoint_weather_study.py` retrains a
fresh model per weather arm (no GCS needed) over 5 large BAs × 10 rolled
windows: aggregating ~12 footprint points beats the single representative
point by **mean +1.23 sMAPE pts** (88% of paired windows), largest where
spread is worst — **MISO +1.93**, PJM +1.66, SPP +1.63 — smallest on
compact SOCO/ERCOT, and the GVL control reads **exactly 0.000** (the
falsification gate passed). **Key finding: population weighting adds
nothing** (C−B ≈ 0) — the gain is pure spatial averaging, so a production
adoption can drop the census/population machinery entirely and
simple-average N footprint points (contradicts the literature's zonal
weighting>averaging claim — at BA-aggregate scale the demand already
integrated the load distribution). Evidence: `docs/MULTIPOINT_WEATHER_STUDY.md`.
Next: adoption PR (config point-sets + multi-point fetch + circular-mean/
weighted-mode aggregation, dark→flip, ADR-012) — a bigger lift than NBM
(changes the weather representation in BOTH jobs), gated on this verdict;
tracked in a new issue.

**2026-07-22 — ADR-011 flipped ON: the fleet forecasts on NBM-composite
weather.** Flag `nbm_weather` → True (PR B of the #332 arc; the issue —
briefly auto-closed by a close-keyword quote-trap in PR A's body, the
documented CLAUDE.md hazard, reopened via API — closes properly with this
PR). Post-flip watch: `weather_nbm_composited` on 51/51 in the next
scoring tick; ADR-010 gate green at the next 04:00 Z training (the first
NBM-fed training frames); AZPS/SEC live sMAPE descending over the week —
their measured deltas were +3.70/+1.88 pts. Rollback = flip off.

**2026-07-21 — ADR-011 NBM-composite weather built, shipped DARK; flip PR
is next.** The A/B study's ADOPT verdict (#332) implemented as the exact
measured arm: `_composite_nbm` overlays NBM onto the base fetch for
future hours only, `NBM_FORCE_FILL_VARS` (radiation ×3, pressure, 120 m
wind) always keep base per the rung-0 evidence, NBM nulls keep base (the
11.5-day tail — ADR-008 untouched), enrichment-only + fail-open
(`weather_nbm_failed` → base frame). Flag `nbm_weather` ships False.
Live-verified in dev: `weather_nbm_composited` n_overlaid=2871 per BA,
~260 future hours moved (mean Δ1.8–1.9 °F, max 12.4 °F), radiation
byte-identical, schema unchanged. After the deploy: the flip PR closes
#332; post-flip watch = composited logs 51/51, gate green at 04:00 Z,
AZPS/SEC live sMAPE descending (+3.70/+1.88 were their measured deltas).

**2026-07-21 — Weather-model A/B study: verdict ADOPT for the NBM
composite (+0.92 sMAPE pts), adoption tracked in #332.** The data-source
research's top candidate, measured the project's way:
`scripts/weather_model_ab_study.py` replayed 8 BAs × 11 anchors × 168h
through the real serve path with lead-honest forecast vintages (Open-Meteo
Previous Runs API). Two findings: (1) **best_match ≡ gfs_seamless for
CONUS** — production already consumes GFS+HRRR, so arm B's ~zero delta
(+0.04) became the harness noise floor; (2) **NBM is decisively better
weather** (temperature RMSE −16% at day-1, −25-27% at days 3-7, bias
~zero vs the control's cold bias) and it feeds through to demand: **mean
+0.921 sMAPE pts paired**, AZPS +3.70 and SEC +1.88 (the tail BAs the
research said nothing external could reach — better weather reached
them), worst BA MISO −0.33 (inside the veto; the multi-point follow-up
targets exactly MISO). Evidence: `docs/WEATHER_MODEL_AB.md`. Next:
adoption PR per #332 (NBM composite behind a dark flag, both jobs
together, ADR-011); the multi-point/population-weighted study follows.

**2026-07-21 — Marketing landing page shipped at `/about`
(portfolio-neutral, BSC-safe).** The archived landing spec's reopen clause
fired via the market-entry plan; the neutral subset now ships as a
self-contained static page (`web/landing.html` + `landing.py` blueprint;
dashboard keeps `/`, zero blast radius). Copy is canonical-framing only
(CLAUDE.md category/tagline, the five real tab names per GP-P2-03), every
number traces to CANONICAL_FACTS (4.8% median per-BA served-ensemble
holdout), the hero is a real headless-Chrome capture of the live ERCOT
Overview, and the "Built in the open" section replaces social proof with
the ADR trail + committed studies. The BSC guardrails (no demo/contact/
pricing CTAs, no "beats" claims) are enforced as posture-pin tests in
`tests/unit/test_landing.py` — flipping them post-process is a deliberate
edit there. Promotion to the front door, benchmark copy, and module
subpages stay parked.

**2026-07-18 — The overnight dive: fit lottery named (#326), the ADR-010
serve-path gate shipped.** LADWP's live XGBoost forecast dove to 1,302 MW
overnight off provably clean inputs. The ablation ladder
(`scripts/forecast_dive_diagnosis.py`, PR #327) reproduced the live curve
within 4.3% with the exact serving pickle and named the mechanism:
**per-training-day fit instability in the recursive serve regime** — 18/67
persisted LDWP vintages (27%) dive on a fixed replay window, and the
published holdout carries **zero signal** (it never runs the deployed
pickle through the serve path). Exonerated: serve-frame construction,
weather values, anchor conditioning (ADR-009 — its numbers were clean),
train/serve AR semantics, training-data contamination. The fix is the
**ADR-010 serve-path acceptance gate**: the training job replays each
candidate through the real serve path from 3 anchors (offset anchors
judged vs settled truth, live anchor vs the trailing week); a rejected
candidate persists as forensics but never repoints `latest.json`.
Calibrated on real vintages at their own training moments — rejects
0708/0710/0715/0717, accepts 0711/0716/0718 + PNM control; under the gate
the 1,302 MW night never happens. Flag `model_serve_gate` ON; first live
exercise is the next 04:00Z training run (verdicts land in meta
`extra["serve_gate"]` + `model_gate_passed|rejected` logs). The arc's
hygiene close-out also landed: the training job now runs the artifact
guard AND drops excluded hours from the training frame — test-first
caught that `engineer_features` imputes `demand_mw` (the 2026-05-29
outage defense), which would have resurrected guard-NaN'd artifacts as
ffill'd fabricated targets. Fit-variance tuning itself stays parked with
draft PR #229.

**2026-07-11 — Forecast honesty: #283 shipped end-to-end, audit critical tier
closed, #296 SARIMAX degeneracy found + fixed.** The #283 seasonal-forecasting
arc completed all phases and the `weather_normal_tail` flag is **ON in prod**
(demand-gated go-live; per-BA artifact backfill completes ~Jul 15). The
buried-ledger audit's standalone criticals are **all fixed** (#267–#272). A
verify-close sweep closed #193 and #174 on file:line evidence and re-scoped
#196 to its three concrete remnants. From user prod screenshots (SC, then
BPAT/PSCO), **#296** surfaced: 8/51 BAs' 30-day SARIMAX forecasts degenerate
(decay through 0 MW or grow ~2×) — root-caused to **double integration**
(auto-selected d=1 stacked on forced D=1 extrapolates the training window's
weather trend as a permanent line; all 8 flagged BAs are d+D=2, zero false
positives on the 39 d+D=1 BAs), fixed this PR with a d+D≤1 cap + fit-time
sanity refit + serve-time per-horizon guard + honest UI withheld state.
Remaining board: the audit ledger clusters **#273** (misleading live numbers,
~15), **#274** (reliability, 7), **#275** (doc/config honesty, 23), plus #196's
narrowed remnants.

**2026-07-08 — Post-keystone: honesty & robustness hardening.** The 2026-07
critical-review keystone is *behind us* — the recursive re-measure shipped
(2026-07-03) and was prod-verified in the 2026-07-06 click-through, which also
answered the drift keystone (horizon-graded Prophet/SARIMAX are healthy). The
surface has since grown — US Grid UX Phases 1–2 (#244/#249), a public read-only
JSON API (#250/#251), and the #225 demand-plausibility guard — so a **product +
open-issue blindspot pass (2026-07-07)** swept for gaps the existing issues
missed. It filed **#253–#256**, closed 3 verified-stale (#199/#194/#124), and
reframed 2 (#222 de-escalated, #153 recast as the honesty keystone). The three
cheap honesty wins already shipped (**PR #257**, merged 2026-07-08: data-source
attribution, scenario-panel "illustrative heuristic" disclosure, demo-vs-stale
freshness relabel). Sharpest open follow-up: **#254** — a circular-utilization
correctness bug in this session's own #244/#251 capacity code. The blindspot
follow-ups are queued under Next-3; the older keystone narrative is preserved
below for history.

**2026-07-03 — Critical-review remediation: CODE COMPLETE + DEPLOYED; RE-MEASURED + DOCS REFRESHED.**
The keystone re-measure has landed: the 2026-07-03 04:00 UTC `gridpulse-training-job`
run retrained all 51 BAs on the #209 recursive-holdout image, and
`scripts/export_holdout_metrics.py` against prod GCS refreshed the published tables —
`BACKTEST_RESULTS.md` / `_holdout_table.md` / `CANONICAL_FACTS.md` / `README.md` now
carry the honest recursive numbers (**XGBoost median 2.32% → 4.32%; ensemble now beats
XGBoost-alone on 17/51, up from 4/51**). Remaining keystone tail: watch
`drift:{region}.rolling_smape_7d` for Prophet/ARIMA, and the gated follow-ups below.


A 35-agent adversarially-verified review at `5000d6a`
(report: `docs/internal/CRITICAL_REVIEW_2026-07.md`) confirmed **2 P0 + 10 P1**
new findings beyond the #181–#189 elegance audit. **Both P0s and 9 of 10 P1s
are now merged to `main`** across PRs #191, #192, #204, #205, #207, #208,
#209, #210:

- **P0-1** (scoring job published `generate_demo_alerts()` as real,
  NOAA-attributed alerts) → honest-empty mitigation (#191) + real live-NOAA
  wiring (#204) + cache-datetime fix (#205). Issue #193.
- **P0-2** (Prophet/SARIMAX forecasts time-mislabeled up to ~24h) → explicit
  `start_ts` anchor + gap-spanning frame (#208). Issue #194.
- **P1s shipped:** freshness measured from `scored_at` (#192, #197), scored_at
  surfaced (#198), interval calibration disclosed (#192, #196-partial), sMAPE
  labeled (#202), alerts warming gate (#200), fabricated-perfection → "—" on
  Overview (#191) + Forecast card (#210, #201), Models-tab zero-residual
  honesty (#207/#166-interim), National-Peak math (#210, #203), generation
  Redis-first no request-path EIA (#210, #199), recursive commensurable
  XGBoost holdout (#209, #195).

**Issues #193–#203 stay OPEN pending prod deploy-verify** (repo convention;
close after verification). P2/P3 folded into the #189 tracker.

**THE KEYSTONE — `main` is already DEPLOYED to prod** (#215, Deploy→Production
succeeded 2026-07-02 20:08 UTC; web service + both Jobs on the current image —
verified via GH Actions). **Next action is the re-measurement (yours, needs prod
GCS/Cloud Run):** (1) trigger a `gridpulse-training-job` run so every BA
re-scores its holdout recursively — `gcloud run jobs execute
gridpulse-training-job --region us-east1` (or wait for the 04:00 UTC daily run).
**There is NO `force` flag** — it is not wired in `training_job.run()`; the
data-hash resume naturally invalidates because the training window has advanced
since the last pre-#209 run, so a normal execute retrains every active BA.
(2) `scripts/export_holdout_metrics.py` against prod GCS → refresh
`docs/BACKTEST_RESULTS.md` / `_holdout_table.md` / `CANONICAL_FACTS.md` /
`README.md` (~13 values move); (3) watch `drift:{region}.rolling_smape_7d` for
Prophet/ARIMA — #170's per-model live drift should drop now that they're
time-aligned (this also eases the #217 "Degraded" false alarm, which is inflated
by the stale teacher-forced holdout). This re-measurement is the gate for the
remaining follow-ups (below) and supplies the evidence for the INTERVIEW_PREP
STAR stories (§9–§11).

**Remaining after re-measurement:** #196 per-model residual calibration
(disclosure shipped; real per-model residuals need the backtest-payload work
#181 also needs); variance-smoothing fast-follow; weather-realism
follow-up; revisit **#170** on the commensurable, time-aligned data. (#181,
#195, #226 done — see recent decisions.)

**Strategic position: A — Portfolio + targeted credibility investment.**
The 2026-05-20 forecast-pipeline audit reframed the credibility surface
substantially: six PRs (#134, #135, #136, #137, #138, #139) shipped in
one day, aligning training, inference, and UI around honest signals.
The audit work itself is now one of the strongest interview narratives
in the repo — a story arc that goes "user reports MAPE looks too clean
→ senior-staff audit → one real bug (training-time target leakage) +
two architectural mismatches (train/serve climatology gap, calibration
provenance) → six-PR sequenced rollout with empirical validation gates
+ visible UI labeling + an ADR." Recent decisions section below has
the full bill of materials.

**Status of the strategic position:** still A. The audit pivot didn't
change the position; it produced more of the "targeted credibility
investment" the position is named after. The recruiter-facing
documentation surface (PR-C1 shipped, PR-C2 parked) is unchanged.

**Open question — 14-day success criterion (by 2026-06-03):** at least
2 of these must be true, or the PM infrastructure built this week is
theatrical and should be partially reverted:

- [x] (a) `docs/HOW_IT_WORKS.md` has real content (PR #125)
- [ ] (b) `docs/HOW_IT_WORKS.md` and `docs/INTERVIEW_PREP.md` have been used at least once for actual practice (read aloud + timed)
- [x] (c) [#121](https://github.com/kristenmartino/gridpulse/issues/121) has a draft PR or partial implementation (PRs [#126](https://github.com/kristenmartino/gridpulse/pull/126) backend writer + [#128](https://github.com/kristenmartino/gridpulse/pull/128) UI panel)
- [~] (d) Handoff quickstart run on another repo — **deferred** 2026-05-20. Discovery: `news-aggregator/` is a working folder with multiple version subdirs (sift_v1, v2, the-digest), not a git repo, so the quickstart can't run cleanly there. User has already set up "something similar" for sift's 3 repos independently — the cross-project validation the criterion was probing for has effectively happened, just outside this framework. Re-evaluate if/when a new project is bootstrapped from scratch.

**2 of 4 criteria satisfied (a + c) — the ≥2 "not theatrical" threshold is cleared. The PM infrastructure built this week is not theatrical.** Criterion (b) takes ~10 min of reading aloud and is yours to do off-keyboard.

## Next 3 (priority order)

*(Refreshed 2026-07-11 against `gh issue list` — the pre-#296 version of this
block still centered the long-completed 2026-07-03 re-measure keystone; the
2026-07 critical-review remediation and the buried-ledger critical tier
(#267–#272) are both fully shut.)*

*(Re-verified 2026-08-05 against `gh issue list`, twice — once before the
day's work and once at close. All three slots are still open and still
correctly ordered. End-of-day state:*

- *Slot 1, [#273](https://github.com/kristenmartino/gridpulse/issues/273): **12 of 15 done**, up from 8. Today cleared
  P2-44, P2-29 ([#402](https://github.com/kristenmartino/gridpulse/pull/402)), ledger-23 ([#404](https://github.com/kristenmartino/gridpulse/pull/404)) and
  P2-19's instrument half ([#407](https://github.com/kristenmartino/gridpulse/pull/407)). **Three remain**, and they are
  not interchangeable: **P2-17** needs a measured `rolling_eval` experiment
  plus a training run because it moves served weights and the visibility
  gate; **P2-15** and **P2-16** are latent — measured, and neither fires in
  production today.*
- *Slot 3, [#275](https://github.com/kristenmartino/gridpulse/issues/275): down to **20 of 23** (`ledger-8`/`-9`/`-11`
  cleared by [#396](https://github.com/kristenmartino/gridpulse/pull/396) + [#398](https://github.com/kristenmartino/gridpulse/pull/398)).*
- *Two new issues filed and **deliberately not** promoted into this block:
  [#399](https://github.com/kristenmartino/gridpulse/issues/399) (test pyramid inverted 92/7/1, no e2e coverage on US Grid)
  and [#401](https://github.com/kristenmartino/gridpulse/issues/401) (per-BA temperature percentiles). Both are invisible to
  users; slot 1 is user-visible wrong numbers.*
- *[#171](https://github.com/kristenmartino/gridpulse/issues/171) **reopened** — see the active-focus block above.*

*The 2026-08-04 incident ([#389](https://github.com/kristenmartino/gridpulse/issues/389)) remains the active focus, ahead
of all three slots.)*

*(Re-verified 2026-08-10 against `gh issue list` / `gh pr list` while
backfilling the decisions log, then **re-verified again an hour later because
it had already gone stale** — [#452](https://github.com/kristenmartino/gridpulse/pull/452) merged between the two
checks. **Slot 1, [#273](https://github.com/kristenmartino/gridpulse/issues/273), is 15 of 15 — every member is now
done**, all four on 2026-08-10: [#443](https://github.com/kristenmartino/gridpulse/pull/443) (**P2-19**),
[#444](https://github.com/kristenmartino/gridpulse/pull/444) (**P2-15**, **P2-16**) and
[#452](https://github.com/kristenmartino/gridpulse/pull/452) (**P2-17**, the one the 08-05 note flagged as not
interchangeable — it moves served ensemble weights and the visibility gate).
At the time of writing the issue was still OPEN and this pass deliberately did
not close it: closing a 15-item tracker is a judgement about all 15, not about
the four this pass verified, and #452 had explicitly split its own scope with
the weights half pre-registered as an A/B in
[#451](https://github.com/kristenmartino/gridpulse/issues/451). It also flagged that slot 1 was no longer
top-of-block on merit, without re-ordering. **Both of those were resolved
within the day — see the re-ordering note directly below, which walked the
checklist, closed the issue and re-verified slot 3's count.** Kept as written
because it records what was true at that pass; superseded on every forward-
looking point.)*

*(**Re-ordered 2026-08-10** on the close of [#273](https://github.com/kristenmartino/gridpulse/issues/273) — the
judgement the backfill note above correctly declined to make on its own. Its
15-item checklist was walked against `origin/main` @ `42d3ed9`: every member
carries a `#273` marker in shipped code except `ledger-3`, which was docs-only
by nature. The issue is now CLOSED with a summary comment. Slots 2 and 3 each
move up one; the new slot 3 is the old slot-3 remnants minus #275. Slot 2's
count was **re-verified** this pass rather than carried forward — it is
genuinely unchanged at 20 of 23.)*

~~1. #273 — misleading numbers on live surfaces.~~ ✅ **CLOSED 2026-08-10,
   15 of 15.** See recent-decisions for the arc. Its two deliberate
   follow-ons are [#451](https://github.com/kristenmartino/gridpulse/issues/451) (the ensemble-weights half of P2-17, as
   a pre-registered A/B — stability was measured, forecast accuracy was not)
   and [#401](https://github.com/kristenmartino/gridpulse/issues/401) (per-BA temperature percentiles, so the Risk
   tab's heat lines can stop being fleet-uniform). Neither is promoted into
   this block: #451 needs scheduled training runs rather than a session, and
   #401 is invisible to users.

1. **[#274](https://github.com/kristenmartino/gridpulse/issues/274) — backend
   reliability/correctness cluster (7 items).** Medium-severity failure modes
   in jobs/data paths, same family as the shipped #267–#272 criticals. Now the
   top of the block: with #273 shut, this is the largest remaining cluster and
   the only one whose members can still break a served number rather than
   mislabel one.
2. **[#275](https://github.com/kristenmartino/gridpulse/issues/275) — doc/config
   honesty sweep, 20 of 23 remaining.** Re-verified 2026-08-10: only
   `ledger-8`, `ledger-9` and `ledger-11` are cleared (by
   [#396](https://github.com/kristenmartino/gridpulse/pull/396) + [#398](https://github.com/kristenmartino/gridpulse/pull/398)); the other 20 have no
   trace in the tree. Same discipline as #273 and mostly one-line corrections,
   so it should move fast once started.
3. **[#196](https://github.com/kristenmartino/gridpulse/issues/196) remnants +
   [#299](https://github.com/kristenmartino/gridpulse/issues/299).** The
   successor to #273's *theme* — user-visible numbers that are wrong rather
   than merely undocumented. #196 is the "80% empirical prediction interval"
   calibrated from XGBoost residuals for **every** model selection; #299 is
   the Prophet seam-step. Historical context for this slot follows.

   **Historical (slot 3 as it read before the 2026-08-10 re-order):**
   ✅ The #296 follow-through completed 2026-07-14 and the issue is CLOSED on
   prod evidence: guard flagged the degenerate fleet on every pre-healing run
   (incl. Prophet collapses on JEA/SEC/LDWP and one flagged SCEG *ensemble*
   the screenshots never caught), the Jul 12 training run capped all 12
   doubly-integrated orders (verified in GCS metas: SC (2,0,0), PSCO (2,0,2),
   BPAT (2,0,1), PJM (0,0,2)…), flags collapsed to zero by Jul 14, and the
   SC/PSCO/BPAT 30-day ARIMA views verified level on the live site. Remaining
   in this slot: the #275 doc/config honesty sweep (23 small items), #196's
   narrowed remnants (per-model backtest prediction vectors; caption
   pooled-count semantics + legacy-key dedup; Overview hero flat-width), and
   #299 (Prophet seam bias-correction, filed 2026-07-11). Watch item:
   weather-normal backfill hits 51/51 ~Jul 15 (spot-check + close the #283
   tail). Newer follow-up also open: #297 (pmdarima univariate-search fix,
   needs its own fleet re-validation).

**Blindspot-pass follow-ups (filed 2026-07-07, #253–#256).** Priority order —
these are the newest tracked gaps and slot ahead of the P2/P3 elegance backlog:

- ✅ **[#254](https://github.com/kristenmartino/gridpulse/issues/254) — SHIPPED
  2026-07-08 (this PR).** The circular National-Utilization / top-stress bug for
  the 5 peak-derived-capacity BAs (SOCO/DUK/CPLE/PSCO/FMPP). Fixed via a canonical
  `config.PEAK_DERIVED_CAPACITY` / `UNRELIABLE_CAPACITY` set excluded from every
  stress aggregate (KPI + sort + per-card "est." chip + map hover + API
  `/grid/summary`), the API field renamed `nameplate_capacity_mw → capacity_mw`
  with a `capacity_source` enum, and the docs desynced by the bug corrected. See
  recent-decisions.
- ✅ **[#253](https://github.com/kristenmartino/gridpulse/issues/253) — CLOSED
  (#263 merged + deployed; noted 2026-07-11).** Original status for history: The now-live unauthenticated
  `/api/v1` had no request-path guard. Shipped in-app (#261, deployed): per-IP
  Redis rate limiting on `/api/v1/*` + the Dash callback route (fail-open),
  `MAX_CONTENT_LENGTH`, and `/health`/`/metrics` gated to liveness-only for the
  public; the exempt/allowlist matchers now take CIDR prefixes so a rotating
  IPv6 `/64` stays valid (#263). **Applied live in GCP 2026-07-08:** 5 alert
  policies (5xx, pinned-at-max, uptime, scoring-creep #171, job-failure), the
  public-`/health` uptime check, and a $150/mo budget with a forecast-anomaly
  rule — all bound to the email channel (ids recorded in `docs/monitoring/README`).
  `METRICS_ALLOWED_IPS` secret set to the operator IPv4 + IPv6 `/64`. **Remaining:**
  merge **#263** (carries the CIDR matcher, the `deploy-prod.yml`
  `METRICS_ALLOWED_IPS` wiring, and the `COMPARISON_GT` fix main was missing) →
  auto-deploy activates the secret. Edge-level (Cloud Armor) rate limiting stays
  a follow-up. Close #253 after #263 deploys.
- ✅ **[#255](https://github.com/kristenmartino/gridpulse/issues/255) — SHIPPED
  2026-07-08 (this PR).** The forecast-quality gate keyed off *XGBoost-alone*
  while prod serves the ensemble, hiding SEC (XGBoost 38.6%) despite its served
  ensemble being 13.6%. Now gates on the **best served model** (champion across
  ensemble + bases) via `get_best_holdout_mape`, Redis-first through
  `get_model_metrics`. SEC visible again; gating on the ensemble alone would
  have newly hidden SPA (22.8%), so the champion gate is the right call — **0/51
  hidden**. See recent-decisions.
- **[#256](https://github.com/kristenmartino/gridpulse/issues/256) — data-source
  attribution/licensing.** **Partially shipped** in PR #257 (runtime-sources
  notice + API `attribution` field + Open-Meteo CC-BY footer link). Remaining: the
  commercial-use posture decision on the Open-Meteo free-tier host.

**Deferred to #189 / #127:** the review's 52 P2 + 57 P3 findings are folded into
the #189 tech-debt tracker; P1-9 (Scenarios panel prod-dead) is tracked under #127
(its **honesty disclosure shipped** 2026-07-08 in PR #257 — the caption; the full
`scenario_engine` swap remains #127's open work).

**Superseded prod-readiness queue** (was Next-3 pre-review, now behind the above):
Phase 4 rigor — #151 deps, #152 mypy, #153 typed Redis payloads, #154 callbacks
decomposition; plus #170 drift logging, #171 scoring runtime, #166 write_diagnostics.

**Queued behind those:**

- **Phase 4 of `prod-readiness`** — engineering rigor ([#151](https://github.com/kristenmartino/gridpulse/issues/151) deps, [#152](https://github.com/kristenmartino/gridpulse/issues/152) mypy, [#153](https://github.com/kristenmartino/gridpulse/issues/153) typed Redis payloads, [#154](https://github.com/kristenmartino/gridpulse/issues/154) callbacks.py decomposition).
- **[#164](https://github.com/kristenmartino/gridpulse/issues/164)** — drop archive-unstable weather vars (wind_80m/120m, soil_temp) + retrain, IF a feature-importance ablation shows they're deadweight. P0 #161 follow-up, low priority.
- **[#170](https://github.com/kristenmartino/gridpulse/issues/170)** — observability: `drift_updated` logs only the alphabetical sample model (`arima`), not the `ensemble` headline, so the user-facing drift number can't be verified from logs. Surfaced during PR-G9 deploy verification. Small logging change.
- **[#171](https://github.com/kristenmartino/gridpulse/issues/171)** — scoring-job runtime (~855s) has no margin under task-timeout; parallelize per-BA fetch/score so runtime *drops* instead of the ceiling rising. Filed from the 2026-06-01 timeout incident (timeout bumped 900→1800s as mitigation). Real fix, not just a higher ceiling. **Promoted ahead of Phase 4 (2026-06-01):** the three post-fix runs measured 1083 / 1262 / 1333s (60–74% of the new cap, one already past the ≤60–70% headroom target), so this is the next substantive engineering fix — not a deferred follow-up.
- **[#174](https://github.com/kristenmartino/gridpulse/issues/174)** — EIA-outage resilience (circuit breaker + uniform GCS fallback). **Implemented + unit-tested this session** (2026-06-04); kept open only for opportunistic **prod-verification on the next real EIA outage** (the one acceptance criterion not unit-checkable). Not auto-closed by its PR by design.
- **[#121](https://github.com/kristenmartino/gridpulse/issues/121) part 3 — Ensemble weight integration** (`path-b`, timing-gated).
- **PR-C2** (`PITCH.md` + expanded STAR stories) — parked unless interview cycle demands it.

**`prod-readiness` Phase 1 + Phase 2 COMPLETE** (2026-05-29). Phase 1: #156/#157/#158. Phase 2: PR-G2 deploy-gating (#146 → PR #159), PR-G3 deep /health (#147 → PR #160), PR-G10 job-failure alerting (#148 → PR #165) — all merged + prod-verified. **P0 #161 fully resolved**: mitigation (A, #162) + proper fix (C, #163, archive ERA5 stitch) both deployed + prod-verified; historical weather coverage ~0 → 14/17 real vars, `/health?deep=1` healthy. Job-failure alerting now live in Cloud Monitoring (no more manual incident discovery). **Phase 3 COMPLETE** (2026-05-30): #149 (PR-G4 → #167) · #155 (PR-G9 → #168, closed #142) · #150 (PR-G11). Remaining campaign: **Phase 4 (#151 / #152 / #153 / #154)** — gated on a checkpoint.

**The production-readiness campaign keeps proving its own value:** PR-G3's deep `/health` (shipped 2026-05-29) caught a total forecast outage on its first production run — invisible to the `curl / → 200` check it replaced. Strongest STAR story in the set.

## Blocked / waiting on

- ✅ **Alert policies: nothing blocked.** All eleven are applied and
  `_KNOWN_UNAPPLIED` is empty as of 2026-08-18. `redis_write_failures`
  (`alertPolicies/16314898527819427981`) went live once its event was confirmed
  in the DEPLOYED image, not merely on main; `scoring_partial_failure`
  (`alertPolicies/1942403527399204858`) and `scoring_deadline_shed`
  (`alertPolicies/8524477981812373740`) preceded it.
  `tests/unit/test_monitoring_policies_applied.py` guards the next one.

- **Forecast tab chart 1–4h gap between actual end and forecast start**
  ([#129](https://github.com/kristenmartino/gridpulse/issues/129)) —
  EIA publishing lag visualized as empty. Fix is in
  `jobs/phases.predict_and_write_forecast`: backfill predictions for
  trailing NaN-demand rows so the forecast trace starts at
  `last_actual_demand_hour + 1h` instead of `featured.timestamp.max() + 1h`.
  Different code path than the audit fixes; ~3-4 hours when picked up.
  Surfaced in Next-3 above (#2).

- **Cross-link this Project to portfolio-v2 / sift / future repos**
  ([#124](https://github.com/kristenmartino/gridpulse/issues/124)) —
  trigger condition (≥2 repos with their own state-management setup)
  is technically met since sift's 3 repos have a parallel framework
  in place. But cross-linking requires deciding HOW (single user-level
  mega-board vs federated per-repo boards) and reconciling shape
  differences between sift's framework and the [`claude-templates`](https://github.com/kristenmartino/claude-templates)
  quickstart. Defer until that decision is worth making — likely when
  spanning ≥3 repos starts producing real navigation friction.
- **Scenario simulator: full-fidelity physics**
  ([#127](https://github.com/kristenmartino/gridpulse/issues/127)) —
  replace the analytical heuristic shipped in PR #119 with real
  `scenario_engine` re-runs (Approach B: pre-computed sensitivity grid
  in the scoring job, preserves Redis-only web tier). Parked until a
  real user / interviewer signal demands physics correctness — see
  issue body for trigger conditions.

**Resolved 2026-05-20:**
- ✅ [#131](https://github.com/kristenmartino/gridpulse/issues/131) — Overview model card MAPE showing simulated baseline values. Fixed by PR #132 (scoring job writes `model_metrics` into Redis payload; `get_model_metrics` reads them as Layer 0); reinforced by PR-A (#134) which switched Overview's MAPE clause to live drift MAPE.

## Recent decisions (last 7 days)
- **2026-08-18** **CI's test suite was never hermetic — it made 79 live API calls per run, and that was a correctness bug wearing a performance costume.** Chasing an 8m45s PR wall-clock turned up 40% CPU utilization on a suite that should be compute-bound, and per-test cost that grew with position in the run (one test: 4.8s alone, 13.6s after 220 tests, 29.3s in the full run) — the signature of an accumulating rate limiter, not slow code. Tests were reaching `api.eia.gov` and `archive-api.open-meteo.com` for real: every client fetch path is cache-first, so a miss falls through to the live API and a test that believed it asserted on `mock_eia_response` asserted on today's grid. `test_scoring_job.py` claimed "All external I/O is faked" while its interchange endpoint went out on all 13 of its tests. **Two mocks were also silently inert:** `patch("data.redis_client.redis")` was defeated by a function-local `import redis` (real DNS lookup, 4.5s), and a `CACHE_DB_PATH` default bound at def time meant a thread-safety test ran 16 threads against the real repo-root `cache.db`. Sockets are now blocked autouse (not `requests` patched — a raw socket or urllib would route around that), with `tests/unit/test_network_guard.py` pinning that the guard is actually installed. **Suite 135.6s → 77.7s serial, 30.4s on 4 workers; CPU 40% → 78%; coverage unchanged at 91%.** CI's four jobs also ran serially for no reason, making the critical path their sum. **Measured on CI: 8m45s → ~3m40s** (pytest step 266–298s → ~180s; dependency setup 66s → 12s on a cached venv). **One optimisation was rejected on its number, not its idea:** Docker buildx `cache-to: type=gha,mode=max` took the image build 83s → 371s, because mode=max exports every intermediate layer of an image carrying prophet/xgboost/shap/scipy — reverted, with the measurement left in a `ci.yml` comment so it is not retried. [PR #586]
- **2026-08-18** **#542's fix verified on live production after deploy — it holds going forward, and 70% of the drift window is still blanked history that no fix can reach.** Independent re-check at **10:30Z**, after [#548](https://github.com/kristenmartino/gridpulse/pull/548) (`71a60cc`) landed: confirmed through to behaviour rather than stopping at "merged", since those are different claims — `71a60cc` is an **ancestor** of the deployed image `437411b` (equality would have failed; eight commits had landed on top), and **new records now retain their lead**: newest-24 known **IID 24/24, PJM 22/24**. Fleet unknown-lead **2,275/2,880 (79%) → 2,029/2,880 (70%)**. **The residual is the operationally important part and it is not a defect:** those leads are *unrecoverable* — the payload proving them was overwritten — so outside the six backfilled BAs they clear only by **ageing out of the 30-day window, ~3–4 more weeks**. Until then the P2-19 filter is still mostly bypassed, and **a 7d-vs-30d comparison spans a repaired population against a partly-blanked one**; anyone leaning on drift's lead-filtered headline in that period should know which half they are reading. **One BA is a different question, not an incomplete fix:** SEC sits at 5/24 with newest leads reading `[None, None, None, 1, 1, 1, 1, 1]` — `_lead_hours` returns `None` when it cannot derive a sane positive lead, which fits a broken feed, so those are **write-time unknowns rather than erasure** and are out of #542's scope. The fix itself is better than the one #542 proposed: `dataclasses.replace` carries **every** field, so the next field added to `DriftRecord` cannot be dropped the same way — the omission class, closed rather than the instance. [docs/542-postdeploy-verification]
- **2026-08-18** **#547: anchor provenance is now recorded at forecast time — the instrument #539 could not build, and it measures nothing yet.** #539 disclosed that our anchor can be seeded by the operator's own day-ahead forecast (MISO 36.58%, CAISO 26.56%, fleet median 3.34%) and had to state that limit's materiality as **unmeasured rather than small** — honest, and not a stable position, because it invites a reader to assume "small". Every forecast payload now carries an `anchor` block (hour, value, was-it-a-placeholder, was-it-conditioned) which rides onto both drift paths. The doc explicitly refuses to let "instrumented" read as "measured" — on the day it lands it has measured nothing. **I repeated #547's own impossibility claim and it is false**: the issue said the anchor cannot be recovered because `lead_hours` is the realized lead rather than the anchor, but row 0 *is* `anchor + 1h` by construction (`_build_future_feature_frame` starts at `forecast_start`) and `_lead_hours` counts from row 0 — so `anchor = target − lead_hours` is exact on the 1h path, and `anchor = target − H − 1h` on the horizon path needs no lead at all. `drift_updated` has logged `(region, target, lead)` at write time since #407, so even the #542 erasure does not block it. A bounded retrospective measurement is therefore **available today** over the vintage mirror's rolling ~30-day window, and the doc now says so rather than asserting unmeasurability. Forward recording is justified by reach — unbounded, not resting on a row-0 assumption a frame-builder change would silently break, and covering the two fields no reconstruction reaches (`anchor_conditioned`, whose fork is never persisted, and the anchor value *as seeded*, which diverges from vintage `first_seen_d` exactly when the quality guard or conditioning touched it). **Three deliberate calls.** (1) Scope stops at drift accrual: no `PairedHour`, no benchmark payload, no `api.py`, because publishing a split computed over zero records recreates the problem #539 fixed, and the published shape should be designed with data in hand. (2) A **fourth** field, `anchor_conditioned` — vintage records the *raw* `D`, so an ADR-009-substituted anchor would otherwise read "metered" while the seed genuinely was their day-ahead value; a true field whose framing asserts something false, the §5 defect one layer down. (3) `regrade_records` converted from a hand-listed rebuild to `dataclasses.replace` — ours was the **second** field class to meet that omission, and the third was a matter of time. Sizing checked rather than assumed: +54 B/record, 12.9 → **20.9 MB** fleet-wide, which is 0.8% of the 1 GB Memorystore ceiling at 13% use, so the ISO `anchor_ts` stays readable rather than being packed as an offset. Ties directly to [#558](https://github.com/kristenmartino/gridpulse/pull/558), merged hours earlier: that origin regression was EIA publishing 19 hours as placeholders (`D == DF`) and then withdrawing them — precisely the ticks on which this block now records `anchor_was_placeholder=True`, so the diagnosis it had to reconstruct becomes a field read. Direction: **moves our own number by exactly nothing**. [feat/547-anchor-provenance]
- **2026-08-18** **Per-horizon champion-chasing measured and rejected: loses to a fixed model at all three horizons, and the reason is near-tie margins, not flip rate.** Reused the preserved `drift_horizon` GCS buffer (§2 of the shared readout), 204 BA-weeks. Chasing the previous window's champion: **4.980/5.452/5.634** WAPE at 24/48/72h, against best fixed model **4.909 (xgboost) / 5.326 / 5.625 (ensemble)** — chasing loses at every horizon. Median champion-vs-runner-up margin is only **0.212/0.316/0.307 pts**, which is why: with 4 models the 90.2/98.0/94.1% flip rates sit **at** their 98.4% null (4-window comparison) and carry no signal, while the 1h figure (35.3%) is **far below** its 75% null (2-window) — champions there are *more* stable than chance, the opposite of how the raw number reads. **Reproduces an independent measurement**: the fleet-baseline sign flip (xgboost ahead 24h, ensemble 48h/72h) matches paired xgb-ens −0.0635 → +0.0554 on a different metric and different windows. Oracle headroom is real (~0.4 pts every horizon) and unreachable by chasing for the same near-tie reason. **Data-completeness note:** windows carry median 161-162/168 records; #537 explains this as episodic and BA-specific (frozen-origin re-derivation dropped by `seen`, LGEE 44/140 ticks), not uniform noise — whether missingness correlates with which window a BA falls in was **not checked**, so cross-window comparability is not asserted. **#542 confirmed clean**: 442,537 horizon records, 0 carry a lead. **Limitation stated explicitly:** only chasing, fleet-fixed, and oracle were tested — a per-BA fixed assignment was not, and ADR-004's existing winner-take-all rejection (`PRD.md:227`) is neither closed nor reopened by this. No follow-up proposed. Evidence: `docs/PER_HORIZON_SELECTION_STUDY.md`. [exp/478-bias-measurability]
- **2026-08-18** **#542 fixed: `regrade_records` erased `lead_hours`, and the counter built to make that observable was published and unread for weeks.** One-line omission beside a correct line — the constructor deliberately reset sMAPE to a recompute sentinel (right: sMAPE is *derived* from the value that moved) and dropped `lead_hours` (wrong: a property of the observation a revision cannot touch). Because `filter_by_lead` keeps unknown leads by design, every revision moved one more record past the P2-19 filter: **82.7% of retained records blanked**, sorted by revision rate not age (PACW 162/163 in-window vs PJM 3/150). **The fix repairs nothing already blanked**, so the before/after had to be reconstructed from outside the pipeline — `drift_updated` had logged `(region, target hour, lead)` every tick since #407, and 31 days of it rebuilt the erased map. Harness validated first: published `n_7d` reproduced on **204/204** blocks, 7-day lead recovery **100%**. **I predicted the direction and was wrong** — the headline moves *both ways*, 17 BAs better and 20 worse (AZPS 9.779 → **11.427**, PSCO 9.639 → **10.766**), fleet only −0.076 while LDWP moves **−3.854** and crosses the `n_7d ≥ 24` gate 25 → 15. Two unplanned findings: `reconcile.py` graded a different *population* than the panel it checks (1 → **12** false A1s once leads were repaired, **12 → 0** with the filter mirrored), and **#537 shares an upstream phenomenon with this and no code** — a frozen forecast origin produces growing leads on the 1h path *and* re-derived targets on the horizon path; LGEE froze 15 ticks then served a 23-hour-older vintage for 24 more (leads to **63h**, horizon 82/168), and `_expire_pending` is **ruled out**. `serve_grade` unmoved: 442,537 horizon records, **0** carry a lead. Evidence: `docs/DRIFT_LEAD_REGRADE.md`. [fix/542-drift-regrade-lead-hours]
- **2026-08-18** **#541 root-caused: the shadow window graded against actuals it never refreshed — and the arm that looked broken was the instrument.** The shadow path reused **one of drift's three record primitives**: it graded with `build_records_from_actuals`, and skipped both filters *and* `regrade_records`. `DriftRecord.actual` is EIA's **current** view, re-graded every tick as revisions land (#304); a shadow record froze `actual` at the tick that created it — the **preliminary** value, which drift's own docstring measures at **15–70% wrong for high-revision BAs**. So the shadow window was prediction-vs-preliminary *forever* while drift converged to prediction-vs-settled. **The row-level diff is the whole argument** (both paths, lead 1, same window, 2026-08-18T05:07Z): predictions **byte-identical on every BA** (`pred_differs=0`), actuals diverging on **123 of 139 hours for IID** and **107 of 144 for SEC** against **3 of 142 for PJM**. IID's frozen actual sat at **339 MW on every row** while EIA settled those hours at **545–867** — the entirety of its +86.49% against drift's **+2.8%**. The BAs that looked broken are exactly the high-revision ones. **That also killed the two hypotheses I would have spent the day on:** the `enumerate(rows)`/`shadow_preds[i]` alignment is fine (`redis_payload["forecasts"]` and `rows=` are the *same list object*, and `pred_differs=0` proves it end to end), and it is **not** ADR-009 anchor conditioning — drift grades the same hours at the same lead from the same payload and reads +2.8%. **The prior session's "a shorter lead cannot be 52× worse than a longer one" was the right instinct pointed at the wrong suspect:** it was never the forecast, it was the yardstick. Fixed by `models.shadow_eval.regrade_records`, mirroring drift's semantics — absent hours skipped rather than treated as agreement, materiality at 2dp, and **idempotent**, which is what makes the corrupt history **self-heal next tick with no backfill**. Measured against production (2026-08-18T05:13Z, **4,037 of 6,963 records re-graded**): control per-BA **+3.263% → +0.739%**, pooled **+2.403% → +0.585%**; treatment +0.707% / +0.547%; `satisficing {'passed': True}`. **Both control figures are inside ±2% for the first time, so the constraint is finally measurable** — and **SEC reads +8.92% against drift's independent +8.96%**, the convergence check that matters more than either number. **No BA excluded, no threshold touched.** Median span **6.54 days**, so #478's 14-day minimum still governs (~2026-08-24). Spun out [#542](https://github.com/kristenmartino/gridpulse/issues/542): drift's own `regrade_records` **erases `lead_hours`**, leaving **2,275 of 2,880 records (79%) unknown-lead** — IID 704/720, PJM 443/720, tracking revision rate exactly — so the P2-19 lead filter is progressively defeated and its docstring's self-healing claim is now false. **Three harnesses, one question, three different instrument failures** (#451's imputed replay weather, #541's stale actuals, #542's blanked leads); the ±2% bound was never the problem. [exp/478-bias-measurability]
- **2026-08-18** **#478 is not waiting on days — the harness built to settle it is writing corrupt records, and that is the second measurement failure in a row on the same question.** [#541](https://github.com/kristenmartino/gridpulse/issues/541). `scripts/shadow_weights_eval.py` stops on `STOP_served_arm_itself_breaches_bias_constraint` with the served arm at **per-BA +9.421% / pooled +3.656%** — but that number is not a forecasting fact. **The shadow path reused the *grading* primitive (`build_records_from_actuals`) and none of the *filtering*:** `compute_drift_payload` runs every record through `filter_by_lead` then `filter_low_actuals` before it averages anything, and the shadow path ran neither, so the two paths graded identically and filtered differently. Fixed by `models/shadow_eval.filter_records`, called **once per region on the shared `actual`** so both arms keep identical hours (a per-arm gate would turn a weighting comparison into a coverage comparison) and deliberately **not idempotent**, since `filter_low_actuals` thresholds on the median of what it is given. Measured before → after: **per-BA +9.421% → +3.264%, pooled +3.656% → +2.412%**, almost all of it **415 records whose known lead exceeded 1h** in a window whose entire name is "1-hour-ahead" (production carried leads to **63h**). `filter_low_actuals` dropped **2 records fleet-wide** — it is region-*relative*, so a BA whose artifact hours are a large share of a short window has them **set the median** and stop reading as outliers. **The gate does not clear the bound, and the residual is one BA:** IID **+86.49% over 126 clean lead-1 records**, against **+1.65%** from the drift path over the same window at a *longer* 24h lead, on a feed whose actuals (339–960 MW) and predictions (397–882 MW) are both sane. **A shorter lead cannot be 52× worse than a longer one** — LDWP is the same shape at ~135× (+327.8% vs −2.42%). So the residual is a defect in how the shadow stream is *written*, not a bias in what the models forecast. **The pattern worth keeping:** #478 exists because #451's *replay* could not measure bias (control over-forecast ~6% against partly-imputed ERA5 weather); the shadow pass was built to escape that, and is measurement-limited in its own different way. Two harnesses, same question, both defeated by their instrumentation — the ±2% bound was never the problem and **no threshold was touched**. Naming the residual BAs is diagnosis, **not** a proposed exclusion: dropping a BA whose number looks wrong after seeing it is the post-hoc criterion-invention `EVALUATION_POLICY.md` forbids and #451 already called out. The 14-day minimum is also unmet (6.5 days), so the answer was never due today — but "answer arrives ~2026-08-24" was wrong about *why* it had not arrived. Found while preserving the horizon-drift buffer (30d × 51 BAs behind a 24h TTL on a no-persistence instance) to `gs://nextera-portfolio-energy-cache/cache/adhoc/`, which is what made the independent cross-check possible. [exp/478-bias-measurability]
- **2026-08-12** **The Overview's dead half is gone — ~3,184 lines across six PRs — and the docs describing it were wrong in four separate places.** GP-P1-04 ([#221](https://github.com/kristenmartino/gridpulse/issues/221)) had sat un-approved since 2026-07-02 with a standing verdict to delete the dead Overview briefing surface. Executed in five scoped steps, each independently revertible and green: news path ([#513](https://github.com/kristenmartino/gridpulse/pull/513), 957), briefing path ([#518](https://github.com/kristenmartino/gridpulse/pull/518), 599), dead builders ([#524](https://github.com/kristenmartino/gridpulse/pull/524), 937), the `tab3-insight-card` slot ([#525](https://github.com/kristenmartino/gridpulse/pull/525)), persona KPIs + the KPI cards they kept alive ([#526](https://github.com/kristenmartino/gridpulse/pull/526), 676). `components/_callbacks_overview.py` went **2,953 → 2,001 lines**; `data/ai_briefing.py` (424) and `data/news_client.py` (158) are gone entirely, and with them **the `anthropic` SDK from the production image** — `ai_briefing.py:100` held the tree's only `import anthropic`. **Three findings mattered more than the line count.** (1) *The fabrication was usually one layer below the dead builder.* `news_client` injected `_get_demo_news()` at three failure sites, so deleting `_build_overview_news` alone would have left the canned-headline generator in the tree with its trigger intact. (2) *"Dead" is not the same as "should delete."* `generate_tab1_insights` was orphaned by step 3 but is the working half of an unbuilt feature — seven persona-tagged insights that `_filter_for_persona` already ranks and caps; verified live that `grid_ops` and `data_scientist` get different bodies. Carved out with a docstring and [#523](https://github.com/kristenmartino/gridpulse/issues/523). The inverse test retired `cards.build_kpi_row`/`build_kpi_card`: live successor (`build_metrics_bar`), no filed use. (3) *§1c was never a blocker.* The proposal said six cross-tab helpers "must be relocated first or live tabs break" — true only for deleting the module. Function-by-function deletion left every import intact, taking the riskiest step off the critical path entirely; §1c stays open as coherence work. **Four doc-vs-reality contradictions, none found by reading docs.** STATUS said "#185 stays open" for the news-honesty product call — 185 closed COMPLETED the same day for its EIA-fetcher half, taking the reserved question with it (deletion answers it: no widget left to choose for). CLAUDE.md said `feature_enabled` defaults unknown flags to **True**; it has been fail-closed since PR-G8/#145. `tab_models.py` claimed a "1-output insight callback" filled `tab3-insight-card`; none was ever registered, and the smoke test *asserted the empty slot must be present* — the suite was guarding the bug. And `_callbacks_overview.py`'s own header still promised "the single home for every `_spotlight_*` / `_build_persona_kpis` helper" after they were deleted ([#528](https://github.com/kristenmartino/gridpulse/pull/528)). Each surfaced by touching the code the doc described. **Public copy was overstating too:** the landing page advertised the Overview's "role-aware briefing" — that briefing was the dead `ai_briefing` module, and what ships changes a four-word eyebrow label by persona and nothing else (`# Persona influences eyebrow only`). Reworded in [#522](https://github.com/kristenmartino/gridpulse/pull/522) rather than left to describe an intention; #523 carries the work to earn the word back, with "restore it to landing.html" in its acceptance criteria. Prerequisite fix: [#503](https://github.com/kristenmartino/gridpulse/pull/503) had made the spotlight MAPE chart honest (it drew bars from `4.5 + len(model_name)*0.3`) while the delete decision was pending; step 3 superseded it, as planned. [this session]
- **2026-08-11** **The tool written to stop a measurement mistake documented, and then printed, the command that guarantees it.** [#484](https://github.com/kristenmartino/gridpulse/pull/484) (`fdb735c`), following [#454](https://github.com/kristenmartino/gridpulse/pull/454). `gcloud logging read` defaults to **`--freshness=1d`** silently, and the archive cache's MISS arm accrues at **one observation per UTC day** because the ERA5 window only moves at 00Z. `scripts/analyze_phase_rollup.py` shipped with `--limit 48` and no freshness flag in its usage block — an invocation that **can never collect a second miss**, so its own archive verdict could never return anything but `INCONCLUSIVE` no matter how large the limit. **No wrong number came out of it**: the `n>=3` guard caught the consequence, which is exactly what it is for. But the trap sat in the usage block of the script whose stated purpose is to prevent this class of error — the same **configured-and-inert** shape as the seven controls recorded above, correct in the place you would naturally check. Fixed in **both** places it lived, and the second is the one that matters: the docstring only helps someone starting out, while the person already holding `MISS arm is n=1` is reaching for a bigger `--limit`, which buys only more hits. The `INCONCLUSIVE` verdict now prints **the day span of the ticks it just read** ("These ticks span 1 UTC day(s): 2026-08-10 — the miss arm is counted in DAYS, not ticks"), how many 00Z boundaries remain, and the flags. **The span is the diagnosis, not the advice** — 23 hits and 1 miss reads as a large sample, and the tick count, the number anyone naturally checks, is precisely what hides the problem. Two calls that could have gone the other way: it **also fires on an EMPTY miss arm** (the same one-day pull an hour earlier, when the reader has nothing and no indication why), and it **goes quiet at n>=3**, because advice that prints on a healthy result trains people to skim the block that matters. 4 tests (20 in that file); no production code path touched. **The `n=1` qualifier on the CANONICAL_FACTS archive row is unchanged** — this changes how the evidence is *collected*, not the figure — but the re-read that drops it is now self-directing: `--freshness=7d --limit 200`. [this PR]
- **2026-08-11** **One model wore three names, and the two PRs that fixed it disagreed with each other within the hour.** Reported as two tabs: Models said **SARIMAX**, Forecast said **ARIMA**, both `value="arima"`. Grepping the two literals found a third nobody reported — **"Arima"** — on the Overview hero card, its ensemble caption ("most accurate single model: Arima") and the data-scientist spotlight axis, all from `model_name.title()`. Each of those sites already hand-special-cased `"XGBoost" if key == "xgboost"` because `.title()` gives "Xgboost" too: **the same bug patched three times locally and generalised zero times**, which is the actual defect — two lines were never the problem, two lines being independently authoritative was. **Resolved to SARIMAX** on two independent grounds that agree: it is what the model *is* (`models/arima_model.py` fits a seasonal order with exogenous weather regressors; the **S** and the **X** are load-bearing and already pinned by `test_arima_model.py`), and it is what README/PRD/TECHNICAL_SPEC §5.3/CLAUDE.md's module map already said. ARIMA is shorter, but width was never the constraint — `tab_models.py` already rendered SARIMAX in the same `gp-segmented` class. `"arima"` stays the internal key (Redis payload, config, callback values): **a boundary, not a rename.** One map, `MODEL_DISPLAY_NAMES` in `components/accessibility.py` (zero-import leaf), read by 16 label sites across 7 files ([#495](https://github.com/kristenmartino/gridpulse/pull/495)). **Then it was contradicted by our own next PR.** [#496](https://github.com/kristenmartino/gridpulse/pull/496) (the SEO opener, entry below) moved `/about` from SARIMAX **to ARIMA** 18 minutes later, matching the Forecast tab — and it published to the internet, live, for ~30 min. Neither PR was wrong alone: #496's author saw two tabs disagree, matched one, and explicitly scoped the in-product split out; #495 had not merged yet. #495's own writeup *predicted* this exact hunk would need reverting — **so the outcome was foreseen and still happened, because nothing enforced it.** The gap was that #495's banned-literal sweep covered `components/*.py` only, and `web/` is the other surface where a model label is user-visible. [#504](https://github.com/kristenmartino/gridpulse/pull/504) reverted `/about` and extended the sweep to `web/*.html`, plus a test that the page still *names* the models (a page could otherwise satisfy the ban by deleting them, passing while being worse). **Two assertion traps, ten minutes apart, both the same shape.** `test_insights_extended` asserted `"ARIMA" in text` among assertions that read like real label coverage — **`"ARIMA"` is a substring of `"SARIMAX"`**, so it was true under either spelling and could not have failed on the split it appeared to guard; it would have gone green on the fix while proving nothing. Then the new AST sweep failed on its first run by flagging the canonical map as a violation of itself, for the identical reason. Both now match on word boundaries (`\bARIMA\b` does not match inside `SARIMAX`). **Mutation testing corrected the new guard a second time**: the first `.title()` rule flagged receivers whose *name* mentioned a model and caught **one shape of four** — the receivers that shipped were `model_name`, `primary_key`, `best_key`, and `m`. A heuristic tuned on the one example in hand scored 25% against the examples already fixed; inverted to an allowlist of the eight legitimate `.title()` calls (fuel, grade, source, baseline-series label), which cannot silently miss. Five reintroductions now fail. **Spun off, not folded in:** `_spotlight_model_accuracy` fabricated MAPE from `len(model_name)` on a cold backtest cache and rendered it as a real measurement — the #131 species — fixed separately in [#503](https://github.com/kristenmartino/gridpulse/pull/503). **A miss of my own, found late:** `_callbacks_forecast.py:580` read "ARIMA is excluded beyond 168h — SARIMAX compounds errors", one model under two names in one sentence ([#506](https://github.com/kristenmartino/gridpulse/pull/506)). It survived because the original sweep filtered `grep '\bARIMA\b' | grep -v SARIMAX`, **discarding every line carrying both spellings** — precisely the lines most likely to be confused. Same species as the substring trap, one layer up: filtering on the wrong relation. The AST guard is deliberately *not* extended to comments; `accessibility.py`'s own block and the test module's docstring name all three superseded spellings on purpose, so the carve-outs would be wider than the rule. Eight bare-`ARIMA` comments in `models/`/`jobs/` left alone as algorithm references (`ARIMA is O(n³)`), correct as written. **Verified in production, and the naive check would have said "not deployed":** live image is `53af9b2` (#505), *not* #504's merge commit, because the deploy guard skips when a strictly-newer commit has already passed CI. The right question is ancestry + content, not tag equality — #504 is an ancestor, and an HTTP fetch of `/about` serves "Prophet, SARIMAX, XGBoost, and the ensemble". **The guard has since held under load**: the remaining SEO stack (#498–#502) all merged afterward carrying ARIMA at their branch tips, and `landing.html` on main is still SARIMAX-only — they inherit that line from the merge-base rather than modifying it, verified per branch before relying on it. All five public pages clean. [this session]
- **2026-08-11** **Search Console is live and needed no DNS record; Bing needed a committed file.** Google **auto-verified** `kristenmartino.ai` as a **Domain property** through the domain-name provider (GoDaddy — nameservers are `domaincontrol.com`, *not* Cloud DNS, so there was never a `gcloud` path). Sitemap submitted: **Success, 5 pages discovered** — `/`, `/about`, `/methodology`, `/coverage`, `/benchmark`. Two `google-site-verification` TXT records already existed on the apex before this; **removing them would silently un-verify the property**, so a DNS tidy-up is now a footgun worth knowing about. Bing was different: its existing `kristenmartino.ai` property is **apex-only** and does not cover the `gridpulse.` subdomain, and Bing has no auto-verification for it. **Rejected the import-from-GSC path** — it is faster but grants Microsoft ongoing OAuth access to the Google Search Console account, which is a bigger standing permission than this warrants. Verification instead ships as `/BingSiteAuth.xml` from `seo.py`: reviewable, version-controlled beside `robots.txt`, and no third-party account access. It is a **route, not a static file** — the file must sit at the origin root, which the `assets/` mount cannot provide, and anything under `assets/` inherits the 1-year immutable header. The token was transcribed off a screen, so a test pins its shape (32 uppercase hex); a single wrong character fails verification with no useful message. Full suite 3,731 green.
- **2026-08-11** **`/coverage` ships, and it makes the opposite build choice from `/methodology` for the same reason.** `/methodology` had to be committed markup because `docs/` is in `.dockerignore`; `/coverage`'s 51-row table is **generated at request time from `config.py`**, which *does* ship in the image — so the page cannot disagree with what the product actually covers, and a committed copy of a 51-row list would be stale within a release. Tests assert the rendered code set equals `REGION_NAMES` exactly, that all seven `PEAK_DERIVED_CAPACITY` rows carry the estimate marker and no others do. **The page publishes both coverage percentages**, 81% by BA count and ~100% by demand, and says why they differ: counting BAs treats a 150,000 MW interconnection and a municipal utility as equals, and quoting only the flattering one would be the omission. It also states plainly that **peak × 1.15 is not a reserve margin** — the single most misreadable number on the page. **Accuracy is deliberately absent**: it is per-BA, moves every retrain, and copying it here would create a second place to go stale, which is the exact failure this whole effort started from; the page links to `/benchmark` instead. The percentage sweep caught `81%` unregistered, as designed. One self-inflicted catch: I measured horizontal overflow with `window.innerWidth`, which includes the scrollbar — a false positive; verified properly at 375px, where the 736px table scrolls inside its 327px container and the body does not. Full suite 3,581 green. **SEO work is complete across seven PRs.**
- **2026-08-11** **`/benchmark` shipped zero numbers in its initial HTML, and it is the one page whose entire value is a published scoreboard.** `#content` was `hidden` and every figure arrived from a single client-side `fetch`. Googlebot renders JS but defers it; Bingbot and most LLM fetchers do not. There was also a live failure mode: `_api_rate_limit` is per-IP and a render fleet shares egress IPs, so a rendered crawl could be 429'd into a blank page. **The route now server-renders fleet tiles, the verdict and a plain scored-row table**, then hydrates the sortable table over it and removes the server block so no reader sees two copies. **The posture that made this delicate is preserved by construction**: `landing.py` has always asserted the page "cannot render a figure the API would not also return", so the SSR path calls the extracted `api.build_benchmark_payload()` — same allow-list, same 30s memo — rather than reading Redis directly. Tests assert `_debug_scratch` and the fleet's `_leak` field still do not cross. **The verdict is derived server-side in both directions**, because a written-in conclusion becomes a lie the first time the numbers move; on the current fixture it correctly reports the operators as closer. **One real defect caught while testing:** my formatter used one decimal where the page's own `num()` defaults to two, so every figure would have visibly changed as the page hydrated. Now pinned by a test. Route cache dropped 3600s → **300s** — this page now carries live numbers, and an hour-stale scoreboard is worse than an hour-stale description of the product. Rendering fails open: any exception leaves the marker empty and the page behaves exactly as it did before SSR existed. Full suite 3,542 green.
- **2026-08-11** **`/methodology` ships — ~1,600 words that existed nowhere on the domain, and `/about` stops spending its authority on github.com.** Both of `/about`'s "how it works" links pointed off-domain, so the site's strongest page sent every reader asking the most valuable question to a domain that does not need the traffic. **It is hand-converted and committed, not rendered from `docs/HOW_IT_WORKS.md`** — `docs/` is in `.dockerignore`, so a route reading it would 404 in every environment that matters. That buys deploy safety and costs drift, and `test_methodology_page.py` pays the drift down the same way PR 1 did: nine figures asserted against the doc that sources them, failing on the *source* side. **The page leads with the withdrawn claim rather than burying it** — a `.counterpoint` block, visually distinct, stating that the ensemble trails best-base (4.35% vs 3.69% median, p90 14.27% vs 9.87%) and that the old tail-robustness claim is retracted. A page voluntarily publishing that its headline architecture is worse than the simpler alternative is the most credible thing on the site; a test asserts it stays in that section. **Two catches from tests I had just written.** The percentage sweep flagged 12 unsourced figures on the new page, which is exactly its job — and one of them was mine: I had rounded canonical `1.72%`/`23.26%` to "1.7%"/"23%" in prose, which is precisely how a page and its source quietly stop meaning the same thing. Now exact. The registry was also restructured from `literal → (page, source)` to `literal → source`, because a figure legitimately appears on more than one public page and both must move together. Posture pins applied to the new page too — "merit-order pricing model" became "cost model" rather than weakening the pin. Full suite 3,533 green.
- **2026-08-11** **The two content-rich public pages were orphans, and the fix could not go where it looked like it belonged.** `components/` contained **zero** `href="/..."` links: `/` linked to neither `/about` nor `/benchmark`, and `/about` never linked to `/benchmark` at all — so the benchmark, the project's strongest asset, was reachable only from GitHub. The obvious fix is links in `build_layout()`, and it would have been **invisible to most of the audience**: `app.layout` is served as JSON from `/_dash-layout` and rendered client-side, so no href inside it exists in the HTML the Slack/LinkedIn/X unfurlers or most LLM fetchers ever see. Three layers instead, one per consumer: the **sitemap** (machines, JS-independent), a **`<noscript>` nav** in `index_string` (agents that fetch but do not render — same links and anchor text as the JS version, so not cloaking, and it also fixes a real courtesy gap where a no-JS visitor got a blank page with no exit), and the **footer** (humans). The footer extends `cards.build_page_footer` rather than adding a page-level one: `.gp-footer` is already the calm `--text-disabled` 10px treatment [#222](https://github.com/kristenmartino/gridpulse/issues/222) asks for, all five tabs already render it, and a second element with that class would have collided. `.gp-footer__nav-link` joins `.gp-footer__source-link`'s existing rule — no new visual language. **Briefing Mode hides the nav but keeps the attribution**, because Open-Meteo's CC-BY-4.0 credit is a licence obligation and not decoration. Verified in the running app: nav `display:none` under `body.briefing`, sources still `block`. Rejected: header links beside Briefing Mode / Save View — the header is already dense and `@media (max-width:768px)` hides `.gp-header__link`, so they would vanish on phones. Full suite 3,470 green.
- **2026-08-11** **Every share of the production URL had been unfurling without a card, and nothing could have told us.** `app.py`'s `og:image`/`twitter:image` were *relative* (`/assets/og-image.png`); the Open Graph spec requires absolute URLs and no unfurler resolves a relative one, so Facebook, LinkedIn, Slack and X all rendered a bare link. `/benchmark` was worse — it declared `twitter:card = summary_large_image` and supplied no image, title or description at all. The existing test asserted `'property="og:image"' in idx`, which was true and useless. Fixed across all three surfaces, with `test_seo_head.py` as **the shared abstraction instead of shared markup**: `landing.py` deliberately keeps the three heads decoupled, and what actually broke was never editorial but mechanical (relative URLs, a card type with nothing behind it, a missing `lang`), which is what tests are for. The strongest invariant there is *`summary_large_image` implies image AND title AND description* — the one `/benchmark` violated. **`<html lang="en">` on the shell is a WCAG 3.1.1 failure**, not an SEO nit. **The canonical on `/` is static and that is the point**: the C2 bookmark flow generates unbounded `?region=&persona=&tab=` variants of one page, and a fixed canonical collapses the whole space. **`index_string` cannot become an f-string** — it contains `{%metas%}`/`{%css%}`/`{%app_entry%}`, so `{` opens a replacement field; absolute URLs substitute through a `__BASE__` sentinel. **JSON-LD ships only four types** and bans `offers`/`priceCurrency` — true (the app is free) and encouraged by Google, but commercial framing on a page whose posture suite exists to exclude it, and `priceCurrency` would have slipped past that suite's prose-only substring check. That posture test now scans structured data too. **Two pre-existing brittle tests surfaced**, both fixed rather than worked around: `test_benchmark_page.py` selected the data script by `body.split("<script>")[1]`, which broke the moment GA4 became the first bare `<script>` — it now selects by content; and my own new test initially depended on import order via `ENVIRONMENT`, which is worse than no test. GA4 added to `/benchmark` (a deliberate call, not a consistency fix — it was the one page whose traffic was invisible). Full suite 3,462 green.
- **2026-08-11** **The site had no 404, and the bug had no symptom.** Four `curl`s at paths that should not exist — `/robots.txt`, `/sitemap.xml`, `/this-page-does-not-exist-12345`, `/wp-admin` — returned four identical `200 text/html`, 10,918 bytes each. Dash registers a `<path:path>` catch-all pointed at its index, so by construction every unmatched URL rendered the app: no error, no 500, no log line that looked wrong, and an unbounded crawlable URL space where every URL is the same page. **The fix is not a path allowlist.** An allowlist duplicates Flask's routing table by hand and goes stale the day Dash adds an internal route; a `@errorhandler(404)` cannot fire at all, because the catch-all returns 200 and `NotFound` is never raised. `seo.register_not_found_guard` reads `request.url_rule.endpoint` and fires only when it equals Dash's catch-all — i.e. the request matched nothing else — so the allowlist *is* the routing table. The endpoint name is derived from `routes_pathname_prefix`, not hardcoded, so a prefix change can't silently disarm it. **`robots.txt` and `sitemap.xml` are generated, not static**, because they must be host-aware: the `*.run.app` origin is live and unauthenticated beside the custom domain and now gets `Disallow: /`, while a sitemap fetched *from* that origin still names only `config.PUBLIC_BASE_URL` (the test that pins this is the one that would catch building `<loc>` from `request.host_url`). **Two decisions pinned as tests rather than left as defaults:** `/metrics` is *absent* from robots.txt rather than disallowed (the file is world-readable, so naming the one IP-gated endpoint advertises it), and there are **no AI-crawler stanzas** — GridPulse earns nothing per pageview so there is no cannibalization to defend, and an assistant surfacing the project unprompted is a corpus property, not a retrieval one. **Adjacent bug fixed:** `_set_cache_headers` stamped the 1-year immutable header on *any* `/assets/` response including 404s, pinning a typo'd path's 404 for a year. Full suite 3,414 green.
- **2026-08-11** **SEO work opened, and the first finding was that the test enforcing our published accuracy number was the reason it was wrong.** `/about` shipped **4.8%** median per-BA holdout error for four days after the 2026-08-07 retrain moved it to **4.35%** — and `CANONICAL_FACTS.md:154` had *already flagged that exact chip* as needing update. It did not get fixed because `test_landing.py::test_numbers_are_the_canonical_ones` asserted `"4.8%" in body`. Its docstring claimed "every number traces to CANONICAL_FACTS.md"; it never opened that file. So the source moving broke nothing, and correcting the page would have failed CI. **The assertion is now inverted** (`test_public_copy_traces_to_canonical_facts.py`): each published figure is checked against its *source doc*, so a retrain fails on the source side and names the page to fix. **A sweep catches the other half** — any percentage in public prose that isn't in the registry, which a fixed list can't see by itself. **Counts got deleted rather than corrected.** Both stale claims were counts ("Ten architecture decision records" — there are 13; "the full list of eight" limits — there are ten, and *the same page said ten* 80 lines later). A count buys a reader nothing and goes stale whenever the list grows; a test now blocks publishing one. The count test found the second instance on its first run, which is the whole argument for it. Also fixed: `CANONICAL_FACTS.md`'s own ADR table listed six and mis-numbered them (labelled the scenario-engine rule ADR-005; `PRD.md:218-220` says that's ADR-007), and its ADR-004 row said 17/51 where the regenerated accuracy section says 21/51. Remaining SEO work is scoped as PRs 2–7 (real 404 + robots/sitemap, head/JSON-LD, internal links, `/methodology`, benchmark SSR, `/coverage`).
- **2026-08-10** **The shadow weighting ships — and the design decision that matters is where it is NOT written.** [#478](https://github.com/kristenmartino/gridpulse/issues/478), the successor to [#451](https://github.com/kristenmartino/gridpulse/issues/451)'s undecided bias half. The scoring job now blends the same per-model forecasts under the arm it is *not* serving and records both, so the constraint #451 could not evaluate gets measured on **production** forecasts rather than replayed ones. Cheap by construction: both arms consume the same `predictions_by_model` arrays, so the marginal cost over the served blend is **one weighted sum**. **The shadow goes in its own Redis key, never into the forecast payload rows.** `drift.extract_one_hour_ahead_predictions` treats **every numeric key in a row as a model** — so a shadow series added there would silently acquire drift records, a Models-tab entry, and a place in the rolling MAPE the **visibility gate** reads. A shadow that is not a shadow, and it would have looked like it was working. A test pins the row key set exactly, and a second asserts the served payload is byte-identical with and without a shadow arm. **A partial shadow arm is SKIPPED, not equal-weighted.** `resolve_ensemble_weights` falls back to equal weights on partial coverage, so letting one through would file an *equal-vs-cubed* comparison under the name *smoothed-vs-raw* — the mislabelling class this repo keeps finding. Grading reuses `build_records_from_actuals` so both arms are graded by identical code on the same hour; separate record streams could diverge on coverage and turn a weighting comparison into a coverage comparison. **One real bug, caught by its own test rather than by review:** an exception from the shadow write reached the forecast phase's outer `except`, which returns `ok=False` — and #267 makes `ok` the signal for *this region was scored*. A shadow failure would have marked a region failed whose forecast was already in Redis: **the #268 mistake inverted.** Guarded at the call site as well as inside the helper. **And one finding about my own code, from mutation testing:** deleting the explicit partial-membership guard changed **nothing observable** — the rule check catches it anyway — so that guard earns its place only through its log line, which is what tells an operator *which* models still lack an EWMA. That is the **unobservable-not-equivalent** category named in `docs/TEST_QUALITY.md`; fixed the same way, with `structlog.testing.capture_logs`. `scripts/shadow_weights_eval.py` **reports control bias first and stops if the served arm breaches the bound** — the wall #451 hit, made structural rather than remembered — and refuses to decide before 14 days. 15 tests, 6 mutations, all killed. Answer arrives ~2026-08-24. [this PR]
- **2026-08-10** **[#451](https://github.com/kristenmartino/gridpulse/issues/451) decided: smoothed weights win, the experiment needed none of the training runs it was parked for, and the confound I went looking for is real but points the opposite way.** This is the half of P2-17 [#452](https://github.com/kristenmartino/gridpulse/pull/452) deliberately left undecided — stability was measured, accuracy was not. **The issue priced itself at ~3h of training per arm and parked as a scheduled job. It needs zero:** the arms differ *only in the weight vector*, so the per-model forecasts are identical across them and the expensive half is paid once. GCS holds every daily vintage back to 2026-04-19 (3,495 for these 12 BAs), so each origin is a **replay** — load the model that was live at that moment, predict 168h forward through the production serve path. ~15 minutes, and being forward-looking it is the *serve* regime rather than the holdout regime. **The general lesson: 'this needs a scheduled run' was an assumption about the experiment's shape, not a fact about it.** **Result** (8 rolling origins x 12 BAs = 96 cases, verdict via `models/rolling_eval.py`): **`ewma_0.3` +0.524 WAPE pts, t=2.21, winning 7 of 8 windows — DECISIVE**, both satisficing constraints met (bias +0.08%, MAPE 13.07 → 12.54); `ewma_0.5` also decisive at half the effect. 10 of 12 BAs favour it, 66 of 96 cases. **The confound is the part worth keeping.** Smoothing compresses the spread between models' MAPEs, so cubed it *should* flatten the blend — making the finding really about ADR-004's exponent, with a different action attached. **Both halves of that are false, and backwards:** smoothing makes the blend **MORE** concentrated (HHI 0.617 vs 0.603), and every flatter exponent loses *decisively* and monotonically (k=2.5 −0.206 → k=1.0 −1.495). So the gain is not "spread the weight around" — and **ADR-004's k=3 is re-validated as a side effect**, independently of the #181 sweep that set it. What is left as the mechanism is that an EWMA estimates a model's *next-week* quality better than one noisy draw, supported by a dose-response the noise hypothesis cannot produce: gain correlates **+0.40** with how far the weights moved, +0.897 where they moved most against +0.152 where they barely moved. **Shipped OFF, and the reason is scope rather than doubt about the verdict.** The pre-registered criterion is met and is not being relitigated — but it was measured on **12 of 51 BAs** chosen for the hard tail, and flipping moves served forecasts fleet-wide. Two disclosures a reader is owed and that leave-one-out is *not* allowed to overturn (it was not a pre-registered criterion, and inventing one after seeing the numbers is the same sin in the other direction): **t=2.21 sits close to its own 2.0 filter and moved 2.26 → 2.21 between two runs hours apart**, and the verdict breaks if **FPL alone** is dropped, or any of 3 of the 8 windows. `update_smoothed_mape` persists the series **regardless of the flag**, so a later flip finds real history rather than one observation. **The 51-BA re-measure then moved the two halves in OPPOSITE directions, which is why it was worth running.** 408 cases. **The WAPE win got MORE robust**: +0.355 pts, **t=2.62**, still 7 of 8 windows, and it now survives dropping **any** single BA and **any** single window — the FPL fragility was a small-sample artifact. **And the ship criterion fails anyway, on the constraint rather than the effect:** |bias| +6.01% against a ±2.0% bound. **That bias is the harness, not the treatment** — `satisficing_check` only ever sees the treatment's bias, and the *control* measures **+6.042%**, with **26 of 51 BAs breaching ±2% before any smoothing is applied**; treatment-minus-control is **−0.029 pts**, i.e. smoothing very slightly *reduces* bias. A vintage replayed 168h forward against partly-imputed weather over-forecasts, and the 12-BA subset simply averaged near zero. **So the 12-BA run passed the bias constraint by luck of the subset, not because the harness could measure it** — a correction to what this entry said an hour ago. A replay whose control arm fails a constraint cannot certify the treatment against it, and `EVALUATION_POLICY.md` counts an unmeasurable constraint as failed. **Ships nothing.** **This retroactively justifies #451's own instinct for a reason the issue never stated:** it demanded real training runs per arm. Those are *not* needed to compare weights — the replay does that better, and answered it. They **are** needed to check the bias constraint, because only production-quality forecasts have production-quality bias. The scheduled run was the right call about the wrong half of the experiment. Settling it needs a shadow scoring pass writing both weightings, or #451's per-arm runs; the WAPE half would not be re-run. **#451 is CLOSED on outcome 2 — ships nothing** — with the WAPE half answered and out of scope for any successor, and the bias half carved out as [#478](https://github.com/kristenmartino/gridpulse/issues/478): a **shadow scoring pass** that records both weightings on production forecasts rather than paying for per-arm training runs. Both arms consume the same forecast arrays, so the marginal cost is a second weighted sum, and `mape_ewma` already persists on every meta regardless of the flag — the history the shadow arm needs exists from today. #478's acceptance requires **reporting control bias first**: if production forecasts also breach ±2% fleet-wide, that is a finding about the fleet, not about smoothing, and the experiment is again unable to decide. Evidence: [`docs/WEIGHTS_AB_STUDY.md`](docs/WEIGHTS_AB_STUDY.md); reproducible via `scripts/weights_ab_study.py`. 29 tests, 6 mutations, all killed. **The implementation shape is worth one line, because it is a deliberate echo of P2-16 ([#444](https://github.com/kristenmartino/gridpulse/pull/444)):** the flag gate lives in exactly ONE place — `weighting_mape` in `models/ensemble.py`, which scoring reads for all three models — while training writes the series unconditionally via `_carry_smoothed_mape` → `update_smoothed_mape` into `extra["mape_ewma"]`. So the two jobs cannot disagree about *which number* they weight by, the same way `resolve_ensemble_weights` stopped them disagreeing about the *rule* — and because the write is unconditional, the history accrues whether or not the flag is on, which is what made the #478 shadow arm free to stand up. (#473's own PR body described this as "both weight callers route through" the gate; on the tree it is writer-and-reader, not two readers.) [#473, #476]
- **2026-08-10** **[#401](https://github.com/kristenmartino/gridpulse/issues/401) verified in production and CLOSED — the fleet-uniform heat lines were wrong in *both* directions, and only measurement showed the second one.** The Risk tab drew reference lines at **95/100/105 °F on all 51 BAs**; [#273](https://github.com/kristenmartino/gridpulse/issues/273)'s P2-29 fixed that by *disclosure* (titling the chart "not calibrated per region"), and #401 was the parameterization half. **Checked against the deployed service the way a browser would** — one `POST /_dash-update-component` per BA against the Risk tab callback, reading the figure's own title and drawn lines; read-only, nothing written, no job triggered. This reuses the [#466](https://github.com/kristenmartino/gridpulse/pull/466) callback driver's technique against production, which is a **second use for it nobody planned**: a Redis key with no API surface is otherwise unobservable from outside the VPC, and the alternative was executing a Cloud Run job to dump key counts. **All four acceptance boxes pass: 51/51 calibrated**, every BA reporting `trailing 93 days`, none falling back. **The premise, now measured: p95 spans 34.2 °F across the fleet** — 78.2 °F at TPWR to 112.4 °F at IID, 44 distinct values over 51 BAs, all monotonic and physically plausible. **The number worth keeping is how badly one line served 51 BAs, and that it failed in opposite directions at once:** for **34 of 51 the lowest generic line (95 °F) sat ABOVE that BA's own p99** — a threshold that could essentially never be crossed (43 of 51 at 100 °F, 48 of 51 at 105 °F) — while for **6 BAs it sat BELOW their own p90** (IID, SRP, TEPC, AZPS, TIDC, EPE), crossing on more than a tenth of all hours. A dead line and a constantly-firing line carry the same amount of information, and **disclosure could fix neither** — it stopped the chart misleading without making it useful, which is exactly the boundary #273 drew when it carved this out. **One deliberate departure from the issue's own proposal**, documented in `_temperature_percentiles`' docstring rather than done quietly: the issue sketched a 10-year ERA5 window labelled `p95 · 10-yr ERA5`, and what shipped is the **trailing ~93-day observed window** from the frame the phase already holds. A 10-year fetch per BA per tick is not affordable, and the weather normal is a per-`(day_of_year, hour)` **mean** that cannot yield a percentile at all — the issue's own blocker 2, which its proposal then contradicted. The seasonal reading is also the more useful one ("hot for this BA lately" beats "hot for this BA in February" when the chart is read in August), and the window length is published with the numbers so the label states what it is rather than implying a climatology it is not. **Observed hours only** — the frame carries a 16-day forecast, and including it would compute a threshold partly from predictions. [#401 closed on production evidence]
- **2026-08-10** **The test tier was renamed to what it did before it was upgraded to do more — and the upgrade cost zero new dependencies.** **[#399](https://github.com/kristenmartino/gridpulse/issues/399) is CLOSED** — all four findings, in the order honesty-then-capability: [#463](https://github.com/kristenmartino/gridpulse/pull/463) gave `tab_us_grid` its first coverage (it was the densest rendering surface in the product and the one tab nothing touched, while the flow table claimed all 5), [#464](https://github.com/kristenmartino/gridpulse/pull/464) renamed `tests/e2e/` → `tests/smoke/` and replaced the documented 55/30/15 pyramid with the tree's real unit-dominant shape, and this PR adds `tests/integration/dash_driver.py` — **the first thing in this repo ever to execute a Dash callback.** **The rename mattered on its own:** filed under `e2e`, 23 tests that call `layout()` and assert `is not None` made the pyramid read as a tree we were failing to grow rather than a target we had stopped believing in, and the docstring claimed Dash testing utilities that appear nowhere in the tree. A tier named for what it does can be judged on what it does. **The driver is a Flask test client, not a browser.** A Dash callback invocation is just a `POST /_dash-update-component`, and Dash serves its own callback registry at `/_dash-dependencies`, so the driver reads the registry, builds a well-formed request and posts it through `app.server.test_client()` — `dash` and `flask` were already required, so a browser tier's Selenium/ChromeDriver bill bought nothing the named gap needed. **The gap was never rendering; it was wiring.** Unit tests call the helper *inside* a callback and the smoke tier calls `layout()`; nothing checked the two were connected. Concretely: `test_callbacks_redis_only.py` reaches `load_data` by standing up a second Dash app, scanning `callback_map` for a function of that name, unwrapping the decorator and calling `load_data("ERCOT", 0)` — a positional call that **would keep passing if the registered `Input` order were reversed**, which a user would see as the wrong region's data. Dispatching puts Dash in charge of the arguments, and reversing that order now fails two tests. **The id contract generalised and a weaker duplicate was deleted:** #463's version regexed `_callbacks_us_grid.py` for `Output("...")` as a stand-in when no driver existed; it now reads `app.callback_map` and covers **all 77 outputs across 41 callbacks** (all present today), so the regex test was removed rather than kept beside it — a weaker duplicate of a stronger check reads like extra coverage and is only maintenance. 11 tests, ~3.5s, **5 mutations verified by hand and all killed** (renamed layout id, reversed `Input` order, deleted `REQUIRE_REDIS` gate, dropped bookmark allow-list, callback ignoring its region argument), each paired where it could otherwise pass vacuously — the warming assertion has a twin proving it *fails* to report warming when the gate is off. **Still not covered, and now stated as the boundary rather than the shortfall:** no browser, so no CSS, no clientside callbacks (`app.clientside_callback` runs in the browser's JS engine and the driver cannot reach it), and no real *input* — the driver sends the values a control would have produced, so a button wired to the wrong callback still dispatches correctly when asked directly. A browser tier remains a **new** tier, not a rename back. **Two things the first three PRs missed, caught only by walking the checklist before closing** ([#468](https://github.com/kristenmartino/gridpulse/pull/468)): **finding 2 (speed budgets) had never been addressed** — unit is restated to `< 60s` rather than defended at `< 10s` (3,041 tests cannot run in 10s and the number was never derived from anything), integration moved the other way on its own and now **meets its original `< 30s`** unchanged, and both carry a dated measured value because the original failure was not the budgets but that nothing re-measured them; and **the issue's own supporting grep was wrong at the commit it measured** — "no `dash.testing`, `dash_duo` or Flask `test_client` anywhere under `tests/`" was false at `db13c06` (ten files, one already POSTing to `/_dash-update-component`), though the finding underneath survives intact because none of them **execute a callback**: every one asserts on a Flask route or on response headers, and the two callback-route POSTs send an empty body that 4xxs before dispatch. **The cost of not checking it: #463 copied that sentence into the smoke docstring verbatim, so a false claim sat for two PRs in the file whose entire purpose was correcting a false claim.** The durable rule is the one this repo keeps relearning — a plausible measurement quoted from an issue is still an unverified measurement. [#463, #464, #466, #468]
- **2026-08-10** **The scenario simulator now runs on real forecasts, and the issue that parked it for months was wrong about its own cost in both directions.** [#127](https://github.com/kristenmartino/gridpulse/issues/127) priced Approach B at **200 ms per ensemble re-run** → ~40 min added to the scoring job, and parked it as unaffordable. The measured production figure is **763 ms/BA** for XGBoost's 384-step recursive loop alone and ~11 s including Prophet — so the issue's own plan was *hours*, and 40 min was already fatal against an 1800 s task timeout the job has actually hit (two SIGKILLs, 2026-08-04). **But the horizon was never questioned:** the simulator charts **24 hours** — `_scenario_demand_factor`'s own docstring says so — and 24 recursive steps against 384 is ~16× cheaper. That one observation moved this from impossible to **~26 s added wall, $2.20/mo** at a 9×3×3 grid. **The money was never the constraint** (even the issue's 9×5×5 is 20¢/day); p95 headroom against #171's 600 s criterion is, which is why the grid is 81 variants and not 225 — the extra 144 buy accuracy finer than a slider step and would spend half the remaining margin. **Marginal ≠ average:** at 406 s the job has already consumed the Cloud Run free tier, so scaling $26.06/mo proportionally understates the true increment by a third. **The correctness trap, and it is the third instance of one bug class:** `scenario_engine._run_ensemble` calls `predict_xgboost` directly while production runs `recursive_autoregressive_forecast` — documented as "the single source of truth". Wiring the engine up as written would have divided a scenario from one inference path by a baseline from another and reported **the gap between the paths as the response to weather** — more sophisticated-looking and less truthful than the heuristic it replaces. Same shape as #437 (backtest carried its own copy of the loop) and #444 (training and scoring each decided weighting). Fix is the same each time: **the module takes the forecaster as a parameter and the caller passes the production one.** **Shipped dark** behind `scenario_grid`, fail-open to the #119 heuristic, computed after the forecast is already persisted so no failure path can cost a written forecast. 39 tests. **What is NOT verified and needs production:** #127's acceptance criterion asks for a physics assertion (+25 °F Heat Dome vs the model's real sensitivity) — unassertable in CI, which has no trained models, and a synthetic booster would only re-measure the stub. The honest claim today is that the plumbing is correct and wired to the right forecaster; whether the sensitivities beat the heuristic is measurable only with the flag on. Evidence + the three things to check first: [`docs/SCENARIO_GRID.md`](docs/SCENARIO_GRID.md). ADR-013. [#458]
- **2026-08-10** **#273 CLOSED — 15 of 15, and three times the issue's own framing lost to measurement.** The "misleading numbers on live surfaces" cluster is cleared; checklist walked against `origin/main` @ `42d3ed9`. Final four landed today: **P2-19** ([#443](https://github.com/kristenmartino/gridpulse/pull/443)), **P2-15 + P2-16** ([#444](https://github.com/kristenmartino/gridpulse/pull/444)), **P2-17** ([#452](https://github.com/kristenmartino/gridpulse/pull/452)). **Four numbers that were wrong on live surfaces:** the ensemble metric **graded itself on its own answers** (weights fitted on the same 168h they were scored against — measured at **+0.46 median / +1.32 mean, worse on 33 of 51 BAs**, republished in [#439](https://github.com/kristenmartino/gridpulse/pull/439)); **CV MAPE was published as holdout** (a different protocol, and itself biased because each fold early-stopped on the rows it was scored against); the Risk tab **printed the literal word "None"** as a grid-stress score for unreliable-capacity BAs and rendered an alerting outage as an all-clear; and historical events were plotted on a **0–100 "Severity Score" axis with invented values** (95/80/85/40 — withdrawn, not replaced). **The pattern is the durable output.** Three times the issue text prescribed one thing and production data said another, and each time the fix got *smaller*: (1) **batch order** — P2-15/P2-16 read as urgent and were *latent*, neither log line having ever fired, while ledger-23 and P2-17 further down the list were biasing every published number daily; (2) **P2-19** — "mixes lead horizons" implied a stratified statistic, but **97% of records are lead 1**, so a filter sufficed and the 3% tail could never have carried its own window; (3) **P2-17** — the item prescribed a smoothed holdout, and the study ([`docs/HOLDOUT_STABILITY_STUDY.md`](docs/HOLDOUT_STABILITY_STUDY.md), 1,620 persisted holdout MAPEs, **no retraining** — GCS already held the estimator's history) showed the raw value tracks the underlying level *better* than any EWMA, **and** that a smoothed gate would read one number while the Models tab showed another. That is the exact defect class the cluster spent 15 items removing, so hysteresis on the *decision* shipped instead: flips 14 → 8 for 10 extra hidden run-days out of 540, with no published figure changed and no lag introduced. **Mutation verification caught two of my own tests asserting nothing** — both used fixtures where the correct rule and the mutated rule give the same answer (ledger-23's constant-multiplier forecasts; P2-17's champion MAPE of 22.1, above *both* thresholds). Both fixed and recorded in their own commits, because it is the same failure mode as the code being fixed. **Deliberately not done:** [#451](https://github.com/kristenmartino/gridpulse/issues/451) — the ensemble-weights half of P2-17, pre-registered as an A/B. Weight churn is real (0.220 L1/day, ~11% of blend mass relocating daily) and smoothing reduces it, but stability is not the objective and whether smoothed weights *forecast* better is unmeasured; shipping on the stability argument alone would repeat the in-sample reasoning ledger-23 was fixed for. Inconclusive ships nothing. [this PR]
- **2026-08-10** **`models/ensemble.py` closed (23 → 8 survivors, logic **73.3% → 91.6%** — the largest single-module move of any round), and two inherited adjudications were wrong.** **(1) An equivalence verdict does not travel to a line that looks the same.** The expression `1.0 / len(model_names)` appears twice in `ensemble_combine`, six lines apart. On the first it is renormalised and the constant cancels — `docs/TEST_QUALITY.md` correctly called that equivalent, "a test for it would be theatre". On the second, inside `if total == 0:`, it is followed by a hard-coded `total = 1.0`, so nothing renormalises it. Mutated there, a stale weights dict returns **300.0, 600.0 or 75.0** where the answer is 150.0 — the served forecast scaled by 2x, 4x or 0.5x, still finite and plausibly shaped. Five survivors, reachable three ways (renamed model keys, an empty dict, an all-zero dict), none covered. **(2) A mutant can be "killed" by a test that has no idea the code exists.** The ledger recorded the length guard as a false survivor killed by `test_stable_hash_reproducibility.py`. It is — via `JSONDecodeError: Extra data`. That test runs a forecast in a subprocess and parses stdout as JSON; the mutant makes a `log.warning` fire, structlog writes to stdout, and `json.loads` chokes. Route structlog to stderr and the kill vanishes with nothing reporting it. **The new category worth naming: unobservable, distinct from equivalent.** Eleven survivors sat in a block that checks the pointwise-bounds invariant, logs a warning, and returns the input untouched — correct fail-open design, but no assertion on the *return value* can reach any of them. Fixed with `structlog.testing.capture_logs`; 9 of 11 die, the other 2 need an exact float equality. **Side effect worth a caveat on the logic score:** the module's noise count fell 47 → 30 (32 in the table, after #444's function landed), because 17 structlog-argument mutants "survive by construction" only while nothing asserts on logs. Where a log line is a load-bearing diagnostic rather than a trace, its arguments are behaviour. **15 mutations verified by hand and the 6 remaining equivalences proved** (three `zip(..., strict=)` variants cannot differ because `arrays` is built by iterating `model_names`; three constants cancel under normalisation, checked at 1.0/2.0/7.0). **Ledger: overall logic 87.6% → 88.4%**, nine rounds now 78.6% → 88.4%, 288 mutants killed. **Re-measured after #444 landed mid-review**, which added `resolve_ensemble_weights` to this same module: 30 more mutants, so the row reads 91.6% rather than the 92.2% measured pre-rebase, on strictly more code. #444's function landed clean (25/30 killed by its own tests, 3 equivalent survivors — it renormalises an already-normalised dict, so the constants cancel). Recorded because the alternative was publishing a number measured against a file that no longer exists, which is exactly what cost #426 its commit subject. **And a second concurrency lesson on top of it:** #441 re-measured `models/skill.py` on main while this was in review, so the fleet row had moved too. Per-PR fleet totals are wrong by construction when two rounds share a base — each omits the other's kills (#442 published 1,891, #441 published 1,851, this PR first published 1,948; the truth is 1,956). **The overall row is the column sums of the seven module rows and must be re-summed, never delta'd against whatever total you last read.** **`simulation/scenario_engine.py` is the only cluster left and it is dead code** — every live module is now ≥86.7%. No production code changed. [#445]
- **2026-08-10** **The same analysis was done by hand four times and the arithmetic was wrong in three of them — one of which reached this file.** [#447](https://github.com/kristenmartino/gridpulse/pull/447) (`78d0e27`). `scripts/analyze_phase_rollup.py` reads `scoring_phase_rollup` logs and prints phase shares, sub-step attribution and the archive cache's paired miss/hit arms. The case for it is the error log: `forecast` was reported at **91%** of worker time when it is **57.7%** (denominator taken from a different tick), summed worker time was said to have fallen **~3×** when it was **1.81×** — and that one **propagated into STATUS.md before anyone checked it** — and the archive leg was estimated at **50–70s** from payload bytes when it measured **294.0s**. None announced itself; each was a plausible number never checked against the payload it came from. The three properties the script enforces each map to one of those: it **never mixes ticks** (`Tick.worker_s` sums from that payload's own `phases` and cannot borrow another's, which is exactly how the bad denominator happened, and easy to do because gcloud will print either block alone), it **always prints n and names the binding constraint**, and it derives rather than assumes. Two new files, nothing else touched. The durable point is that hand-arithmetic over log payloads had a **75% error rate here** and produced numbers confident enough to publish.
- **2026-08-10** **The #389 arc, measured: the cache is a 19x win on its leg, `generation`/`interchange` are ~97% upstream latency, and four of my own numbers were wrong along the way.** Five PRs: **[#414](https://github.com/kristenmartino/gridpulse/pull/414)** `fetch` sub-steps + the cross-run GCS archive cache **dark**; **[#424](https://github.com/kristenmartino/gridpulse/pull/424)** fact corrections; **[#432](https://github.com/kristenmartino/gridpulse/pull/432)** flag flipped **ON** on the measurement; **[#440](https://github.com/kristenmartino/gridpulse/pull/440)** `generation`/`interchange` sub-steps (#427); **[#447](https://github.com/kristenmartino/gridpulse/pull/447)** `scripts/analyze_phase_rollup.py`. All merged and deployed; flag verified `True` on main, not merely merged.

  **THE CACHE WORKS — and the formal verdict is still INCONCLUSIVE, deliberately.** 22 hit ticks against the one 00Z forced miss (2026-08-09T18Z → 08-10T15Z): miss **301.6s**, hit **median 15.9s** — **19.0x on the leg, 285.7s summed, ~37s of a ~400s tick (~9%)**. Two independent uncached reads agree three days apart (**294.0s** 08-07, **301.6s** 08-10), and the hit arm's *worst* tick (81.9s) is still 3.7x under the miss, so **the arms do not overlap**. The script nonetheless refuses to conclude, because the MISS arm accrues at **one observation per UTC day**. That guard exists so a liked result cannot be waved through, and it is not being overridden: the honest statement is **strongly indicated, formally n=1**. Three more 00Z boundaries settle it. **[SUPERSEDED 2026-08-12 — CONFIRMED at n=4 misses / n=90 hits: 17.0x, 288.9s summed, arms separated by 3.1x. The summed saving is unchanged within noise; see the archive row in `docs/CANONICAL_FACTS.md`. The three boundaries this asked for took a `--since 2026-08-08` window, because the 7-day pull that first reached n>=3 got there by crossing the 08-07 flag flip.]**

  **[#427](https://github.com/kristenmartino/gridpulse/issues/427) is ANSWERED, and the answer is that there is nothing local to fix.** Over 21 clean ticks, `eia_generation` is **median 96.5%** of the `generation` phase (93.6-98.7) and `eia_interchange` **98.0%** of `interchange` (96.7-99.2) — **our own pivot/transform work is ~3.5% and ~2.0%**. That ~15% of worker time is upstream latency end to end. Per the issue's own criteria the honest close is *"this is what the data costs"*, unless multi-respondent batching (`facets[respondent][]` is an array in EIA API v2) is pursued — untested.

  **An EIA degradation event sits in the window and must not be pooled: the 13:19 tick.** Elapsed **1132.75s (2.8x the median)**; `eia_generation` 4531.5s, `eia_interchange` 1689.5s, `eia_demand` 1623.6s — all three EIA legs inflated together, the 2026-08-04 signature. Note `forecast` **SHRANK** to 565.2s against a ~1650s norm: workers blocked on I/O stop contending for the GIL, the same second-order effect the climatology vectorisation produced in reverse. **The script failed to flag it** — its confound check keys on `n != 51` and that tick had all 51. An elapsed-outlier check is the fix; found by using the tool, which is the point of having one.

  **Four wrong numbers, all mine, none caught by thinking harder.** The ADR-012 diagnosis of the alert (**refuted** — flat medians across the flips; the cause was EIA retry tax); `forecast` published at **91%** off a borrowed denominator (**57.7%**); a **"~3x"** denominator drop that is **1.81x**; and the archive leg estimated at 50-70s from payload bytes, measuring **294s** — low by 4.2-5.9x. A fifth was caught by CI: the unit test written to prevent cross-tick mixing was itself written by mixing ticks.

  **Operational notes.** `gcloud logging read` defaults to `--freshness=1d`, so `--limit 96` silently returns one day — use `--freshness=7d` to reach several 00Z boundaries. Post-cache the largest `fetch` leg is **`eia_demand`** (126-288s), with `weather_nbm` (70-120s) now ahead of `weather_archive` (6-26s). The hit arm's 81.9s maximum is the **01Z** tick, explained by the 00Z miss recording n=48: the three BAs that took the `fetch_weather` fallback had nothing cached and paid full price an hour later. And a merge is not a deploy — the pipeline guards on main's tip, so a commit ships via any later deploy rather than one of its own. [this PR]
- **2026-08-10** **A well-tested function nobody called, and the mutation score was high *because* of it.** [#441](https://github.com/kristenmartino/gridpulse/pull/441) (`65547b6`). `models/skill.py::skill_payload` had **no production caller**: `jobs/phases.py` imported four other symbols from the module and hand-rolled its own copy of the published skill block inline. The two had already drifted — production emitted `window_days` and `decision` but **not `beats_baseline`**, the field the module's own docstring calls "the field worth acting on", and it skipped the non-finite guards. `should_serve_baseline` consumed the inline dict and worked only because it happens to read two keys both versions carry. **The causality is the finding:** #416 had taken this module to 88.6% and every mutant of `skill_payload` died — there was a test for each — so the module scored well *because* the dead half was well tested, while the block production actually serves had no direct coverage and contributed no survivors either way. Coverage and mutation testing both answer "is this code tested"; neither answers "is this code reached." **Kept the function rather than deleting it**, which was the closer call: the inline block is what ships, but it lives inside a private method behind a feature flag, a Redis read and a DataFrame, while `skill_payload` is a pure function of two arguments — deleting would have moved nine tests onto a surface needing heavy mocking to test arithmetic. `beats_baseline` joins the published payload only after checking every consumer (`api.py` passes the block through; **no UI surface reads it**), and it is always `False` where it appears, which a test now enforces. Two docstring claims were false and are corrected: there is no `gridpulse:skill:{region}` key (the block ships nested in `gridpulse:forecast:{region}:1h`; the per-tick publication that key anticipated was never built), and nothing is "published on the Models tab". **Re-measured as a back-to-back A/B, not against the published figure** — the pre-change tree reproduced its row *cell-for-cell* (192/164/21/7), which is a better data point on this instrument's jitter than the one previously recorded flip. Logic score **88.6% → 90.5%**; the module is now pinned everywhere except the closed `dtype` class (**all 18 remaining logic survivors are that class, zero others**). Two of the kills are worth the retelling: `mape`'s `return float("nan")` survived because **no test had ever executed that line** — the module sat at 98% coverage with exactly one uncovered statement, and it was the one deciding what `mape` returns when nothing is measurable, the same branch the whole non-finite-guard argument runs through (now 100%); and `skill_payload`'s `lag_h` forwarding, which is the **inverse of [#442](https://github.com/kristenmartino/gridpulse/pull/442)'s finding the same day** — there, defaults were never executed because every test passed parameters explicitly; here, a parameter's forwarding was never exercised because every test used the default. Drop it and the block publishes a 24h number under a label reading "lag 48h", inverting `beats_baseline`. **The merge with #442 produced a third instance of the same shape and is the part to carry forward:** both PRs recomputed the ledger's overall row against the same shared base, so **each published a total omitting the other's kills** (1,891 and 1,851; the column sums give **1,899**), and #442 had re-measured the baseline to 272 logic survivors while leaving the adjudication section at 304 — under a sentence promising those counts "match the baseline above rather than trailing it", false by 32. Both reconciled here (overall now 2,372 / 1,899 / 269 / 204, **80.1% / 87.6%**), with the file now stating that the overall row *is* the column sums and that a per-module re-run must carry its subtraction into the adjudication section. The adjudicator itself was not re-run; none of the 35 killed were `dtype`, so that class is still exactly 81. Verified post-merge rather than assumed: 3,152 tests green and `models/skill.py` re-measured on the merged tree to the same 194/172/18/4.
- **2026-08-10** **The `maxiter` study I was about to scope isn't worth running, and the claim that prompted it was mine and wrong.** [#446](https://github.com/kristenmartino/gridpulse/pull/446) (`f24e883`). SARIMAX is **~58%** of the training job at **~49s** per fit, so `maxiter` is its largest cost knob — and nothing recorded whether the optimiser was using its 200-iteration budget, so every argument about the right value was a guess. The earlier claim that the optimiser *"never converges"* at 200 came from **one pathological synthetic series** (pure sine plus noise, no day-of-week term). On a realistic series it **converges at 123 iterations**: `maxiter=400` is byte-identical to 200 (same iterations, same llf, **0.0000%** parameter difference), while `maxiter=100` saves only **19%** and moves parameters by **88%** — a non-converged fit is a *different model*, not a cheaper one. The cost case was weak regardless: post-[#433](https://github.com/kristenmartino/gridpulse/pull/433) training is ~$21/mo and the whole SARIMAX block ~$12/mo, so even a successful 2× buys **~$6/mo**, and every remaining lever is a model change needing a rolling-origin study per `docs/EVALUATION_POLICY.md`. **But that measurement is synthetic** — the evidence class this codebase has been punished by repeatedly — so rather than close the question on it, two fields (`converged`, iteration count, already carried by `statsmodels`' `mle_retvals`) now ride on `arima_trained` to settle it against the **102 real fits** a production run performs. Either reading is useful: converging well inside 200 closes the knob question, and a meaningful fraction hitting the cap is a **model-quality** finding independent of cost, since SARIMAX is a third of the ensemble. The extractor is deliberately defensive — `mle_retvals` is optimiser-dependent, some spell the count `nit`, some paths omit it, and a helper that raised inside a log call would fail a fit that had already succeeded — with 11 tests covering every shape including a **stable key set**, because a field that sometimes vanishes makes the log-based question unanswerable for those rows. Also recorded: the refit census is **103 fits where 102 are structurally required** (51 BAs × 2 calls on different training frames, plus one drift retrain; the ADR-296 guard refit fired 0 times). There is no waste to remove. Answer arrives at the next 04:00 UTC run.
- **2026-08-10** **A cross-validation score that graded itself, then got published under the wrong name — both latent, and fixed on that basis.** [#444](https://github.com/kristenmartino/gridpulse/pull/444) (`e7dea2b`), closing **P2-15 and P2-16** of [#273](https://github.com/kristenmartino/gridpulse/issues/273). Measured as latent *before* the work: `ensemble_holdout_unavailable` has not fired in production and `scoring_ensemble_equal_weights_fallback` has **never** fired, so these are guards on a path that does not run today, not repairs to a number users are reading. **P2-15 part one:** each XGBoost CV fold early-stopped on `eval_set=[(X_val, y_val)]` and was then scored on those same rows, so the boosting-round count was selected to minimise the error being reported — a *fitted* quantity published as held-out. Early stopping now uses an inner split carved from the tail of the fold's own training portion; where that portion is too small, the fold fits **without** early stopping, on the reasoning that a slower, possibly over-fit booster is a better failure than a self-graded number. **Part two:** `saved_mape = holdout.mape or cv_mape` put a CV mean into `meta.mape`, which `get_model_metrics`, the Models tab and the ADR-004 ensemble weights all read as a holdout — and beyond the bias the two measure **different protocols**, teacher-forced one-step folds versus recursive multi-step. Now split into `meta.mape` (recursive holdout, or `None`), `extra.cv_mape` (never substituted) and `extra.mape_source`. **Equal weight on an unmeasured model is honest; a cubed weight from a mislabelled number is not.** The `or` was independently a latent bug — a holdout MAPE of exactly `0.0` is falsy and would have fallen through to CV — now pinned by test. **P2-16:** training weighted **always** inverse-MAPE³ over models with a valid holdout, while scoring weighted MAPE³ *only if all* predicting models had a MAPE and otherwise equal — so the persisted ensemble metric could describe a different blend, and a different membership, than production served, under one name. Both now call `models.ensemble.resolve_ensemble_weights`, so the rule is **shared by construction**; membership still differs by necessity (training holds holdout payloads, scoring holds forecast arrays), so each side records what it used (`members` + `weight_rule`, `ensemble_composition`) and scoring logs `ensemble_composition_divergence` when they disagree. They may legitimately differ; **what must not happen again is differing silently.** 10 new tests, 5 mutations, all killed.
- **2026-08-10** **The measurement argued against the obvious design, for the second time on this issue and again toward less machinery.** [#443](https://github.com/kristenmartino/gridpulse/pull/443) (`8f28cea`), finishing **P2-19** of [#273](https://github.com/kristenmartino/gridpulse/issues/273). [#407](https://github.com/kristenmartino/gridpulse/pull/407) had added `lead_hours` and deliberately left the pooling alone until the distribution was known; production now says **lead 1 is 97.0%** of records and leads 2–6 are the remaining 3.0%, with 97.5% of ticks discarding no other matchable hour. "Mixes lead horizons" reads like it wants a **stratified** statistic — a window per lead — and on this data that is over-engineering for a 3% tail that could never carry its own window. Filtering the headline to lead 1 costs 3% of records and ends it. **The load-bearing decision is that `lead_hours=None` is KEPT:** records predating the field cannot have their lead recovered, and dropping a *known* lead-6 record is a strict improvement while dropping an *unknown* one discards seven days of history to win three percent of purity — against a measurement saying that record is 97% likely to be lead 1 anyway. It **self-heals** (every record written since the field ships carries a lead, so the filter becomes exact once pre-field records age out of the 30-day window) and both transition counters ship — `n_lead_excluded_7d`, `n_lead_unknown_7d` — so convergence is *observable*, matching how `n_low_actual_excluded_7d` already works. Scope held deliberately narrow: `records` stays **unfiltered** because it is the history a later consumer would need to stratify; the filter runs before the low-actual filter and before every count so `n_7d`/`n_30d` and the rolling means describe one population; and it is **not** applied to `_horizon_rollup_block`, whose #227 records have a *designed* 24/48/72h horizon that lead-1 filtering would empty — with a test asserting `filter_by_lead` never appears in that function, because that is the one mistake here that would be silent. **This moves a published number:** `rolling_mape_7d`/`rolling_smape_7d` feed the Models tab and the visibility gate's per-model figures, and should drift **down** slightly, since what is removed is longer-lead records whose error is genuinely higher. 6 new tests, 5 mutations, all killed.
- *(Bookkeeping, closed out 2026-08-11: the placeholder that stood here claimed **#473** had no entry. **It does** — folded into the #451 decision above, which is attributed `[#473, #476]`, the same way #452 sits inside "#273 CLOSED" and #438 inside its own. Retired rather than left standing, since a note asserting that something is outstanding is worse than no note once it stops being true. Two placeholders in this log have now each been wrong within a day of being written; `gh pr list --state merged` is the check, not this list.)*
- **2026-08-07** **`models/evaluation.py` pinned (81 → 49 survivors, logic **77.1% → 86.7%**), and the cause was one sentence: every existing test passed every parameter explicitly, so none of the published defaults were ever executed.** `lower_q=0.10`, `upper_q=0.90`, `target_coverage=0.80`, `window_size=168` could each have been changed to any value at all with the full suite green. Those four numbers are the contract behind the "80% empirical prediction interval" the Forecast tab renders — that 80% is `0.10` and `0.90` in this file and nowhere else. **This is a blind spot neither coverage nor review catches:** the defaults are *covered* (the lines execute on every call) but never *exercised*, because a test that passes `lower_q=0.2` is testing the caller's number, not the module's. Calling a function the way production calls it — arguments omitted — is a **different test** from calling it with them spelled out, and the parameterised version silently replaces it. Two smaller findings: `compute_error_by_hour` had **no value assertion at all** (random inputs, `len(result) == 24`, a column name — so `abs_errors = None` survived and the Backtest heatmap could render 24 empty cells with CI green); and `compute_r2` returns `0.0`, not `1.0`, for a zero-variance actual series, an unexercised branch that flipped would publish a *perfect* R² for an hour of flat demand. **20 targeted mutations verified by hand, 19 killed and 1 proved equivalent** (`max(1, ...)` → `max(2, ...)` is a no-op when slicing a 1-element array), plus 5 more found in the leftovers and killed. **What remains is not a backlog:** 42 of the 49 are the `dtype=float` class closed in #435, and the other 7 are float-boundary variants (`> 1e-10` → `>=`) needing an exact float equality no real series produces. **Ledger: overall logic 85.8% → 87.4%**, seven rounds now 78.6% → 87.4%, 248 mutants killed. **Every module that is neither dormant nor `ensemble` is now ≥86.7%**; the remaining gate-policy condition is elapsed time (four consecutive stable weekly runs), not work. [#442]
- **2026-08-07** **The `dtype=float` reachability claim was short by three call sites — and the tests I wrote to prove the fix passed with the fix fully reverted.** [#438](https://github.com/kristenmartino/gridpulse/pull/438) (`aca49ff`), following #434 above. That PR's argument, not its one-line edit, is what mattered: object dtype is *"unreachable from inside this codebase"* was the stated basis for **not** pinning ~81 `dtype=float` survivors. The reasoning was sound and the sweep was incomplete. `pd.DataFrame(columns=[...])` also builds every column as object, and the client had three: `_last_known_good:159` — the **terminal return of the #174 stale→GCS→empty chain, whose own docstring calls it "typed-empty"** — plus `_fetch_eia:218` (empty upstream response) and `_parse_demand_records:731` (zero rows). All three serve **demand, generation and interchange**, so the shape #434 removed from one branch stayed reachable from three others across every endpoint. **This file was carrying the same false claim**: the #185 entry below describes the fallback as "stale→GCS→typed-empty". **The placement is worse than the original** — the first two run only when a fetch has failed *and* the stale cache missed *and* the GCS parquet missed, so the object frame appears **only during an upstream outage**, on the least-covered path, reached when an operator is already debugging something else; `np.isfinite` raises on an object array regardless of length, so a zero-row object frame crashes a consumer a zero-row float frame passes through. **Nothing was broken**, same as #434 — the same five defences absorb it and `jobs/phases.py:1967` guards `.empty` first — so this closes the *argument*, which is the thing licensing the survivors, not a live bug. `_typed_empty()` derives dtypes by letting pandas build a real one-row frame from prototype **values** and slicing to zero rows; hardcoded dtype strings would have been wrong within one release, since **pandas 3.0 gives `str` and `datetime64[us, UTC]` where 2.x gives `object` and `[ns]`**, and the parity test compares against each parser's own output so both sides move together on a version bump. It **fails open** — an unregistered column degrades to the old untyped frame rather than raising, because raising inside the outage fallback converts a degraded fetch into a hard failure — and `test_every_empty_cols_list_is_typed` reads the `empty_cols=` lists back out of the source so that escape hatch cannot be used silently. **The finding worth carrying forward is about the tests, not the dtype.** The first round was a parametrized class over all three endpoints, 118 green; reverting all three call sites failed **one** test. The class called `_typed_empty()` **directly**, so it proved the helper worked and nothing about whether anything *used* it — both outage sites were unpinned by construction, and a helper wired to nothing would have shipped with a green suite. **Mutating the helper is not mutating the call site.** `TestTotalOutageReturnsTypedFrame` now drives the real public fetchers through the terminal scenario; re-running the mutations fails 3 and 1 respectively. **One limitation stated rather than papered over:** the `_fetch_eia:218` frame is never actually returned (an empty frame there always routes to `_last_known_good`), so it is typed for consistency but no test can observe it. **Verification:** 3,082 passed / 3 skipped; 527 across the 11 files a `grep` — not a guess — says touch the changed symbols; mutations applied only *after* committing. **Deployed and confirmed by image SHA** (`cc697fd`, all three surfaces on one image, and the fix read back out of `git show cc697fd:data/eia_client.py`), not by a green deploy badge — the badge at merge time belonged to #437. **`eia_untyped_empty_frame`: 0 over 24h, with a control** (`scoring_job_complete` = 24 in the same window and query shape, so the zero is real and not an unmatched filter). What that cannot prove: a healthy tick never exercises this path, so a clean run is equally consistent with the fix working and with the branch never being entered — the real proof waits for the next upstream wobble.
- **2026-08-07** **The `dtype=float` question is closed — the answer was one line, and the re-measure that confirmed it moved nothing.** For three revisions `docs/TEST_QUALITY.md` carried "~81 unpinned `dtype=float` coercions, 26.8% of survivors" as an open **policy** question, framed as a choice between writing ~81 tests, scoping them out, or leaving them as a floor. It was not a policy question; it was answerable by looking. **The sweep:** `_parse_demand_records` already coerces every EIA value through `float(raw)` with a NaN fallback, so `demand_mw` is float64 whatever the API sends — nulls, strings, zeros. Exactly **one** line in the codebase produced an object-dtype numeric column: the no-forecast branch assigning `result["forecast_mw"] = None`, where the sibling `merge(..., how="left")` branch already yielded float64 NaN. Two branches, one right. **#434** makes them agree. **Nothing was broken** — every consumer survived via one of five separate defences (`pd.to_numeric` in quality/vintage, a `try/except` in benchmark, an `is None` check in phases, `np.asarray(dtype=float)` in the metric helpers) — and my first probe pass wrongly reported crashes in benchmark and phases because it used hand-built frames with bare `float()` instead of going through the real parser. **Two false bug reports if I had trusted the simulation**; the same failure mode as the fixture trap that recurred across three modules — the setup satisfied the assertion without exercising the thing. **The re-measure after the fix was byte-identical: 2,370 / 1,843 / 304, logic 85.8%, dtype class still exactly 81.** It had to be — those mutants survive because no *test* passes object dtype, and the fix changed production code. Recorded because the run could have flattered the story and did not: the reclassification rests on the reachability sweep, not on a number moving. **A fix that makes a whole class of survivor unreachable is invisible to the score that pointed at it.** Verdict: equivalent *in practice*, guards stay (they are correct, and the helpers are importable by code not yet written), class no longer counted as actionable — **the actionable population is 223, not 304**. **The generalisable rule, which is the durable part:** a large homogeneous mutation class points at a *missing invariant upstream*, not at N missing tests; a small heterogeneous cluster is the opposite. **Incidental finding, recorded against my own earlier claim:** `models/skill.py::skill_payload` has **no production caller** — `jobs/phases.py` imports four other symbols from that module and builds the skill block inline, and the two have already diverged (the inline block omits `beats_baseline`, the field `skill_payload`'s docstring calls "the field worth acting on", and adds `window_days`/`decision`). So #416's `should_serve_baseline` and `seasonal_naive_forecast` work is on the serving path and its `skill_payload` work is not; worth an issue to either call the function or delete it, because a tested payload production never emits is a liability dressed as coverage. [#435]
- **2026-08-07** **Two more modules pinned, the ledger re-measured end to end for the third time, and one merged commit's subject line is wrong.** **#416** (`models/skill.py`, 50 → 21 survivors, logic **72.1% → 88.6%**): the substitution boundary that decides whether a region is served a naive forecast instead of its model, the edges of the projection that replaces it, and the published Redis payload. `seasonal_naive_forecast` already had a *property*-based test — written that way deliberately, because its docstring records a previous bug where "the unit test encoded the same wrong arithmetic and passed" — and it still missed every edge, including a negative index that **wraps to the end of history**, serving the newest hours as the oldest. Also found: `skill > 0` relaxed to `>=` publishes `beats_baseline: True` for a model with exactly zero skill, which is the one claim the module exists to prevent. **#426** (`data/feature_engineering.py`, 140 → 82, logic **83.8% → 90.5%**): four clusters — the cyclical encoders, wind power, the autoregressive snapshot. **The reason 140 survived there is the transferable finding:** every existing assertion compared the output to *itself* or to a *range*. `test_hour_0_and_24_equal` compares two midnights, true under any period, so `/24` could become `/25`; `test_rated_wind` asserts `> 0.5`, true across a wide span of wrong physics; the snapshot tests assert *parity* with the training path, so both could drift together. None is a bad test — each pins a real invariant — but a mutation that preserves the *relationship* passes. Two things only the mutations surfaced: the `DatetimeIndex` branch of both encoders had **never been executed**, and both wind thresholds are inclusive (3 m/s generates, 25 m/s has not tripped) where the old fixtures sat far outside. **Ledger re-measured** (2,370 mutants, 304 logic survivors, overall logic **83.0% → 85.8%**; six rounds have now taken it 78.6% → 85.8% and killed 200 mutants). **The false-survivor rate has been re-adjudicated three times and found the same two mutants every time** — 2/457, 2/363, 2/304; the numerator has never moved and the percentage drifts up only because the denominator shrinks, so quote the pair not the ratio. **Correction to the record:** #426's merged commit *subject* says "141 survivors down to 83" — the pre-#423 figures. #423 rewrote `recursive_autoregressive_forecast` mid-work, both halves were re-measured against the new module, and the commit *body* carries the right numbers (919 mutants, 140 → 82). The subject was not amended and GitHub's squash took it over the PR title. **Next: `models/evaluation.py` (81 survivors, 77.1%)** — now the largest untouched surface, holding the #296 degenerate-forecast guard and both interval-coverage functions. **Open policy question, unchanged and now the largest item in the file:** the `dtype=float` class is 26.8% of survivors (81), rising every round because everything behavioural around it gets pinned; pinning them means ~81 tests passing object-dtype columns into numeric helpers, and the alternative is scoping them out deliberately. The score cannot decide that. [#427]
- **2026-08-07** **The archive leg is 294s, not 50-70s — the estimate was low by 4.2-5.9×, and the flag is now ON.** #389's cache shipped dark specifically to wait for this number, and `fetch_substeps` (2026-08-07T12:07:58Z, elapsed 448.1s) delivered it: **`weather_archive` 294.0s is the LARGEST leg of `fetch` at 42.6%**, ahead of `eia_demand` 175.3s (25.4%), `weather_forecast` 146.0s (21.2%), `weather_nbm` 74.2s (10.8%). 6.1s/BA average, 13.6s worst (NWMT). **Why the estimate was wrong:** it reasoned from payload size (3.40 MB at 12 points) and assumed wire time dominates. At 6.1s/BA the cost is ERA5 server-side extraction on Open-Meteo's free host — bytes were the wrong proxy. That is the fourth estimate in this thread that production overturned, and the one place I refused to act without measuring is the only reason this is a number rather than a story. **It also refutes a claim I had been repeating:** that `fetch` is dominated by EIA. On that tick `fetch` is **~75% weather, ~25% EIA**. **Flipped ON per the rule pre-registered on #389** (material → flip; small → delete), which existed precisely so 294s could not be argued down after the fact. **Not yet verified: the SAVING.** A hit swaps an Open-Meteo fetch for a GCS parquet read, which is not free. The verdict comes from the arms the design generates itself — the window moves at 00Z, so the first tick of each UTC day is a forced MISS and the other 23 are hits, a within-day paired contrast on the same code and BAs. Per `EVALUATION_POLICY`, several days of pairs, not one. **Also surfaced, unhandled:** `weather_archive` has **n=48, not 51** — three BAs returned early from `fetch_weather` via `_fallback`, meaning their base forecast leg failed and they served stale cache or GCS that tick. Small silent degradation, worth its own look. The legacy weather tests are now insulated from this flag's default via the module fixture, after it cost two test edits across two flips. [this PR]
- **2026-08-05** **#414 merged dark (`9997d07`) — and the denominator moved under it before the measurement could be taken.** Shipped: `fetch` sub-step timing (`scoring_phase_rollup.fetch_substeps` splits `eia_demand` from `weather_forecast` / `weather_nbm` / `weather_archive`), plus the cross-run GCS archive cache behind `weather_archive_cache`, **default `False`**. Flag-off is byte-identical, so the merge changed nothing in production. **Every share I quoted for this cache is now stale.** The vectorised climatology fill landed in the same window and took `forecast` 3,269.5s → 1,636.3s and wall clock 676.9s → 370.7s. **I first wrote "summed worker time fell roughly 3×" — wrong, and off a bad figure**: CANONICAL_FACTS said `forecast` was "1,636.3s of ~1,800s summed worker time", but the 17 `phases` entries in that rollup sum to **2,834.8s**. Corrected there in this PR. The real drop is **5,131.0s → 2,834.8s, 1.81×**, and `fetch` went **667s (13.0%) → 426.3s (15.0%)** — it fell in absolute terms with EIA's recovery while gaining share. So the old framing was roughly right after all: a ~50-70s archive leg is **12-16% of `fetch`, ~1.8-2.5% of worker time**. Still an estimate, still unmeasured. Meanwhile the runtime pressure that made it interesting has eased — **#171's `<600s` is met for the first time (370.7s, n=1)**. **Pending, and the first thing to check is not the number — it is whether the instrumentation is there at all.** The merge landed ~19:45 UTC and the 20:06 tick is the one CANONICAL_FACTS cites; whether that execution carried the new image is unverified, and there is a known gcloud asymmetry that has frozen the jobs on a stale image before. **Checked, and it is not there**: the 20:06:54Z rollup (execution `vbxlx`) carries `forecast_substeps` and `phases` but **no `fetch_substeps`** — that tick predates the deploy of `9997d07`. The measurement is still pending a tick on the new image, and the pre-check is what stopped a missing field being read as a small number. **Decision rule, committed before the number arrives:** material → flip the flag on and harvest the 00Z miss vs 23 hits as a within-day paired contrast over several days (per `EVALUATION_POLICY`, not one window); small → **delete the cache**. The `fetch` breakdown is reusable and stays either way, which is what makes deleting a clean outcome rather than a loss. Full runbook on [#389](https://github.com/kristenmartino/gridpulse/issues/389). [this PR]
- **2026-08-05** **#414 reduced to the archive cache, my diagnosis retracted, and the cache held dark behind a measurement it never had.** The concurrent-session PR carried two halves. Its **instrumentation half was already on main** — shipped there, not rewritten — and it was additionally un-mergeable against 54 commits of drift. Its **causal claim was wrong**: it blamed ADR-012 for making the ERA5 archive leg ~12× heavier, and the runtime record refutes that (flat daily medians across the 07-22/07-23 flips; the incident was EIA retry tax). What survives is the cross-run GCS cache for the archive segment — and the honest problem with it is that **its own value was never measured**: `fetch` is 13.0% of worker time, the archive leg's share of that was an estimate (~50-70s, ~1%), and `fetch` is dominated by EIA, whose variance swung runs 660s→1800s. A ~1% signal is not readable off that total. **So the phase now names its legs**, the way `forecast` does after the 60.1% finding: `fetch_substeps` splits `eia_demand` from `weather_forecast` / `weather_nbm` / `weather_archive`. The `substep` primitive moved to `observability.py` to get there — `data/` must not import `jobs/` — and `jobs.phases` re-exports it, so all 12 existing call sites are untouched. **The cache ships DARK** (ADR-011/012 discipline) and stays dark until `fetch_substeps.weather_archive` is read from a live tick. If that number is small, the honest outcome is deleting the cache, not enabling it. Once on, the arms generate themselves: the window moves at 00Z, so the first tick of each UTC day is a forced MISS and the other 23 are hits — a within-day paired contrast, same code, same BAs, repeating daily. Per `EVALUATION_POLICY`, a verdict wants several days of those, not one. Retraction recorded in the code comment, the flag comment, `HOW_IT_WORKS` and on #389. 3026 tests green. [this PR]
- **2026-08-05** **The mutation ledger is re-measured end to end, and the one number I had put in doubt survives.** `docs/TEST_QUALITY.md` had drifted five ways across #377/#383/#385/#386/#416 — three module scores, two cluster sizes, and an unqualified false-survivor rate. Whole table re-run in one pass so baseline and adjudication finally agree: **2,354 mutants, 1,768 killed, 363 logic survivors, overall logic 78.6% → 83.0%.** Five rounds of test work killed **125** mutants and changed production behaviour exactly once — the `verdict()` crash #386 fixed. **The doubt that prompted this:** #386 found the adjudicator could reuse stale bytecode (CPython invalidates a `.pyc` on `(mtime, size)`, mtime at one-second resolution; many mutations preserve length exactly, so an apply/run/restore loop finishing inside a second tests code that was never mutated). That failure is silent and **biased in one direction — it can only manufacture false survivors** — so the published 0.4% rate could only have been an undercount. Re-adjudicated all 363 with the fix in place: **the same two, in the same function, killed by the same test.** The figure was right; it just had not been *shown* to be right. It reads 0.55% now only because the denominator shrank from 457 to 363 — the numerator never moved. **What the table now says that the headline does not:** equal effort bought +0.2 pts overall on `rolling_eval`, but **+17.5** on `data/quality.py` and **+16.5** on `models/skill.py`, because those landed on weak modules. A 2,354-mutant denominator makes every real fix look like rounding error, which is the concrete argument for the per-module, no-regression gate the policy section now specifies. **Gate conditions 2 and 3 are resolved**; only baseline stability over four weekly runs remains, and that needs time rather than a decision. **What is next is now unambiguous:** `data/feature_engineering.py` holds **141 survivors — 39% of everything left** — and four of the top eight clusters, with `compute_autoregressive_snapshot` seeding the recursive forecast; its 83.3% is exactly the unremarkable mid-table number that stops attracting attention. Machine-readable verdicts replaced with a single dated file, `docs/data/adjudication-2026-08-05.json`. [this PR]
- **2026-08-05** **#273 pushed 8→12 of 15, and the batch order was wrong until it was measured.** Three PRs. **Batch D — [#402](https://github.com/kristenmartino/gridpulse/pull/402):** the Risk tab was built **twice** and the two builds disagreed. Production rendered the severity breakdown as emoji with inline styles while dev rendered `gp-stress-row` icon components whose CSS had shipped long ago and **nobody in production had ever seen**; the dev path did `str(stress)` where `grid_stress` returns `None` *deliberately* for BAs with no reliable capacity plate (#254), so those regions printed the literal word **"None"** as a grid-stress score; and with the NOAA feed down dev's empty state read "No active alerts" — **an outage displayed as an all-clear**. Now one `_render_risk_tab`, each disagreement resolved toward the more honest of the two, and the module docstring corrected (it claimed "a single function" with "no fallback compute path"; the dev fallback has computed stress inline since #265 — the same species of stale claim [#396](https://github.com/kristenmartino/gridpulse/pull/396) cleared from the docs, sitting in the file that describes it). **P2-29** in the same PR: historical events were plotted on a 0–100 axis labelled "Severity Score" with values 95/80/85/40 — **unsourced editorial judgement rendered as measurement**, beside charts fed by real demand; dates and regions are factual so it became a categorical timeline and the scores are gone. Temperature lines at 95/100/105 °F were fleet-uniform, implying each of 51 BAs had been assessed against its own climate; they now say they are generic. **Disclosure not parameterization, deliberately** — real per-region thresholds need per-BA percentiles the web tier may not compute in the request path (web-tier I/O guardrail), and the weather-normal artifact is a `(doy, hour)` **mean**, the wrong statistic for an exceedance threshold → filed [#401](https://github.com/kristenmartino/gridpulse/issues/401). **Batch E — [#404](https://github.com/kristenmartino/gridpulse/pull/404):** `_ensemble_holdout_metrics` fitted inverse-MAPE weights on the **same 168 hours it then scored the blend against** — the combination rule saw the answers it was graded on, so every published ensemble figure was optimistically biased, including BACKTEST_RESULTS and the Models tab. Split: weights on the leading half, scored on the trailing half. Measured on the null case (three equal-skill models, 400 trials) the old estimator reports **−0.057 pts better** than an equal-weight baseline and the new one **+0.057 worse** — **0.114 pts of bias removed**. Served forecasts unchanged: scoring recomputes weights from each model's `meta.mape` and never reads the persisted ones. **[#407](https://github.com/kristenmartino/gridpulse/pull/407):** P2-19's instrument half — `DriftRecord.lead_hours`, because the window called "1-hour-ahead" is not uniformly 1h (forecasts start at `last_real_demand_hour + 1`, EIA lags 1–4h) and `build_records_from_actuals` **permanently discards** every other matchable hour. Pooling untouched: filter-to-lead-1 vs stratify-by-lead depends on a distribution nobody could see, and the headline feeds the Models tab and the gate's inputs — evidence before a published-number change, the order used for the vintage instrument before ADR-009. **The measured re-ordering is the finding:** P2-15 and P2-16 read as urgent from the issue text and are **latent** — `ensemble_holdout_unavailable` has not fired in 3 days, `scoring_ensemble_equal_weights_fallback` has **never** fired — while ledger-23 and P2-17 bias every published number every day. Verification throughout: **15 mutations applied one at a time, 15 killed**, and **two of my own tests initially survived** (constant-multiplier fixtures make in-sample and out-of-sample coincide) — rewritten with mirrored halves that assert the separation first. Browser verification caught what tests could not: dropping the severity axis widened the plot and clipped the edge labels. [this PR]
- **2026-08-05** **Measured the post-bump scoring baseline instead of assuming it, and the phase rollup moved the next lever.** Two `CANONICAL_FACTS` rows were carrying **"not yet measured"** against a config change already live in production — the exact failure class the registry exists to prevent. Now measured from `scoring_job_complete` (n=48 pre-bump, median **807.8s**, range 665.6–903.4) against the three runs since the 8-worker / `--cpu 4` config went live at **01:44 UTC**: **1041.8 / 699.4 / 667.9s**, all 51/51 ok, 0 errored. **Published as inconclusive, deliberately** — n=3, the window overlaps EIA's recovery, and 667.9s is inside the pre-bump range (min 665.6s). The `scoring_phase_rollup` instrumentation #389 shipped is what actually paid: **`forecast` is 60.1%** of worker time (3085.5s of 5131.0s), `fetch` 13.0%, `generation` 11.1%, `model_load` 8.7%, `interchange` 4.0%, rest under 1% — and effective parallelism is already **7.7×**, so **in-container workers are spent**. That redirects #171's `<600s` (still unmet, best 667.9s) away from "more workers" toward cheaper 720h recursive inference or more vCPU via task fan-out. Training's post-change cost stays **honestly unmeasured** — its first post-change run (`ppgw8`, 04:00 UTC) was still in flight; 4Gi is confirmed live. Deploy verified by image SHA (`db13c06`), not by a green deploy badge. [this PR]
- **2026-08-05** **Seven doc claims the code contradicted, found by tracing every number in an end-to-end walkthrough back to source.** [#396](https://github.com/kristenmartino/gridpulse/pull/396) took six: scoring runtime `~5 min` → `~14 min` (the same file already said ~14 min 54 lines later, and the stale low number is what hid the 2026-06-01 creep); the model diagram still showing pre-#181 `weight_i = 1/MAPE_i` while the prose below it described the cubed form, with `CLAUDE.md`'s module map carrying the same stale formula; `4 horizons` → 3; README's training window `60 days` → 90; README's `43 derived features` → 49 = 17 raw + 32 derived; and API `5 routes` → 6 endpoints + index. **The widest gap was personas** — docs claimed persona switching reconfigures KPI selection, alert thresholds, scenario sets and a welcome briefing; none of it ships. `_build_persona_kpis` and `_build_overview_briefing` have zero live call sites (re-exported under `noqa: F401` for tests only), `alert_threshold` is never read outside its dataclass, and all four personas declare `default_tab="tab-overview"` so the redirect is a no-op. What ships is the insight-card eyebrow plus persona-filtered insight count. [#398](https://github.com/kristenmartino/gridpulse/pull/398) took the seventh, `ledger-9`: README said 1681 tests, CANONICAL_FACTS said 1,589, the tree collects **2,989** (2,986 passed / 3 skipped, CI agreeing exactly). **The count went stale mid-PR** — measured 2,977, then #397 landed 12 tests underneath it — so the row now warns it moves weekly and the README tree line says `~3,000` rather than a precise number nobody re-runs. Clears `ledger-8`, `ledger-9` and `ledger-11` of #275's 23; that issue stays open for the other 20. Filed [#399](https://github.com/kristenmartino/gridpulse/issues/399) for what a number fix could not honestly touch: the test pyramid is **92/7/1** against a documented 55/30/15, the e2e tier uses no Dash testing utilities despite its docstring saying so (`layout(); assert result is not None`), and **`tab_us_grid` has no e2e coverage at all** while the doc claims all 5 tabs.
- **2026-07-31** **The ensemble validity guards are pinned — the worst-scoring module is no longer the worst, and no production code changed.** `models/ensemble.py` sat at **61.6% logic** in the mutation baseline and was the highest-value target: ADR-004 weights feed every served forecast, so the function raising takes the whole ensemble down. Three guards were unpinned, and none is defensive decoration — each maps to a **reachable** crash that the real code already handles correctly while nothing asserted that it does: `v > 0` removed admits a MAPE of exactly 0 into `1.0/v` (`ZeroDivisionError`); `and np.isfinite(v)` weakened to `or` admits `inf`, whose `(1/inf)**k` is 0.0, so the weight total is 0.0 and normalisation raises; `weights.get(k, 0)` losing its default yields `None` and summing raises `TypeError`; and the equal-weights fallback's `1.0/n` stops summing to 1, silently rescaling every forecast it combines. **Reachability is documented, not hypothetical:** `compute_mape` returns `inf` when every actual is zero and TIDC publishes zeros, so a BA whose feed goes flat produces exactly that input. **Why the pre-existing `test_handles_inf_mape` covered none of it:** it pairs `inf` with a *healthy* model, leaving a non-zero denominator, so the weights come out right **even with the guard broken** — the recurring shape of these gaps is a test that exercises the line and asserts something true in the one arrangement where the guard does not matter. Seven tests, each verified by re-applying its mutation one at a time. **Measured: 53 → 63 mutants killed, 33 → 23 logic survivors, logic 61.6% → 73.3%** — and the overall score moved 0.4 pts (69.5% → 69.9%) against the first round's 0.2, same effort, because this one landed on the weakest module rather than a well-tested one. That contrast is now the argument in `docs/TEST_QUALITY.md` for a **per-module** gate: a 2,349-mutant denominator makes every real fix look like rounding error. One mutant deliberately left unpinned and recorded as **equivalent** — `ensemble_combine`'s `1.0 / len(model_names)` default is renormalised on the next line, so any non-zero constant gives identical output (verified across 0.333/0.667/3.0/7.0); a test for it would be theatre. The adjudication ledger is kept **as-measured** with a note that ten of its 455 confirmed survivors are now dead, rather than back-edited — a ledger that rewrites its own history is not evidence. [#383]
- **2026-07-31** **All 457 mutation survivors adjudicated — and the false-survivor rate I published was wrong by ~80x.** `docs/TEST_QUALITY.md` said seven were adjudicated and the other 450 unexamined. New `scripts/adjudicate_mutants.py` applies each reported survivor to the real source, runs tests, restores the file: **455 confirmed, 2 false** (both `ensemble_combine`, both killed by the subprocess test behind limitation 1). **The correction:** the file claimed "roughly a third of what a mutation run reports is not a gap" — from a sample of seven chosen *because* they looked interesting. Measured over the population it is **2 in 457 (0.4%)**. The lesson survives; the number and the reasoning behind it did not. **`--fast`, and why it is sound:** mutmut has already run every *traced* test against a survivor, so the only tests it can have missed are ones tracing cannot see — work in a child process. `grep -rln "subprocess\|multiprocessing\|Popen" tests/unit/` returns exactly **two files**, and that bound is the whole argument. Cross-validated rather than asserted: the same 69 mutants both ways gave 67/2 either way, **zero disagreements, 84 min → 4 min**. **What the 455 are** matters more than the count: they cluster by *untested function*, not by defect — `_run_ensemble` alone accounts for 26 and survives everything including `forecasts = None`, which is one missing test, not 26 findings. **A second correction, caught by probing:** the 81 `dtype=float` survivors were first classified as equivalent noise; on an **object-dtype** array (which pandas produces here from missing EIA data) `np.isfinite` *raises* without the coercion, so they are unpinned defensive guards. **Priority a raw score cannot express:** `models/ensemble.py` is highest value — ADR-004 weights feed every served forecast and three unpinned guards each map to a reachable crash (a MAPE of 0 and a non-finite MAPE both → `ZeroDivisionError`; `compute_mape` returns `inf` for all-zero actuals and TIDC publishes zeros, so this is documented behaviour); `simulation/scenario_engine.py`'s 36 are in **dead code** — nothing imports `simulation`, the live Scenarios feature runs a heuristic, #127 tracks the swap. **Gate policy: 2 of 3 conditions now resolved**; only baseline stability over four weekly runs remains, and the threshold that would work is recorded (per-module logic score, no-regression rule — an absolute bar would have to sit below ensemble.py's 61.6% and could then never fire). Machine-readable verdicts in `docs/data/`. [#381]
- **2026-07-31** **The 36.8s test was slow because it was not testing what its name said — the fix is the same edit as the speedup.** `TestRunForecastOutlook::test_sqlite_cache_hit` mocked a SQLite payload with `predictions` and `timestamps` only. The read guard in `_run_forecast_outlook` requires **four** things: `predictions` present, `cache_version == _CACHE_VERSION`, and `data_hash` matching the live frames. Commit `a618374` (2026-04-08) added the version/hash half of that guard to the production path and did not touch the test, so every run since fell straight through the cache branch into **real XGBoost training** — 5-fold `TimeSeriesSplit` CV plus SHAP, 35.8s standalone. Nothing failed, because the test only asserted `"predictions" in result` and `isinstance(..., np.ndarray)`, and inline training satisfies both. The three sibling `test_sqlite_cache_hit` tests (one in this file, two in `test_callbacks_v1_paths.py`) all pass `cache_version`/`data_hash` correctly — this one was the outlier, not the pattern. Fix: complete the payload, then assert the things a fall-through cannot fake — exact cached values, `mock_cache.get` called once with `forecast:ERCOT:24:xgboost`, the in-memory cache primed on the way out, and `train_xgboost` patched to raise so a future re-break fails in **0.9s** instead of passing in 36. **35.8s → ~1s standalone (it cost 52.7s in-suite, worse than isolation); the whole unit suite 91.39s → 41.64s measured back-to-back on one machine at `aa33cf2`, 2,692 passed 3 skipped both times.** Consequence for #376: the mutmut deselect existed only because this test dragged `data/feature_engineering.py` through 904 mutants × ~124s; the cache path never reaches the feature pipeline, so the deselect and the "Known limitations" note it earned are both deleted — that module's mutants are no longer scored against a false survivor. No `mark.slow` reintroduced; the slowness was a bug, not inherent cost.
- **2026-07-31** **Test quality became measurable: coverage is visible on every PR, and mutation testing found two real gaps in the code that decides model changes.** Two halves. **Coverage** (landed on `main` inside #375): it was already computed in CI but only as a pass/fail integer — no HTML, no PR comment, no per-file view. Now `htmlcov/` uploads on every run (including failed ones), a single PR comment carries the per-file table, and `diff-cover` reports **changed-lines** coverage, which is the only version of the number that can signal on a PR when total coverage is 88.83% across 12,500 lines. The test job also stopped running the suite **twice** — it ran unit/integration/e2e as three steps and then all of `tests/` again for coverage; one instrumented run replaces four. Two dead directives deleted: `-m "not slow"` matched **zero** tests (no `mark.slow` exists) and `pytest scaling-analytics/tests/` pointed at a directory that does not exist. Gate ordering fixed so a coverage shortfall reports as a coverage failure after the artifacts publish, not as a test failure that aborts before them. **Mutation testing** ([#376](https://github.com/kristenmartino/gridpulse/pull/376), `docs/TEST_QUALITY.md`): 2,349 mutants over the seven decision-critical modules — **69.3% raw, 77.8% behavioural**, ~25 min. The point coverage cannot make: `models/ensemble.py` is **85% line-covered and scores 61.6%**. **Two real gaps, each hand-verified against all 2,687 unit tests — and both now FIXED in the same PR:** `satisficing_check`'s `regression > max_mape_regression_pts` flipped to `>=` with the suite green (as did the `|bias|` bound, and `verdict()`'s sign-consistency and noise thresholds — five boundaries in all) — the function CLAUDE.md makes mandatory for every model change had the least-pinned boundaries in the repo; and `coerce_demand_artifacts`' trailing-hours `continue` flipped to `break` unnoticed, which would leave every artifact after the first gap uncoerced — the exact failure the #309 guard exists to prevent. Five boundary tests now pin them, each verified by re-applying its mutation and confirming the new test fails. **The tests settle a policy question, which is the real value: the thresholds are inclusive** — exactly 2% bias, exactly 0.5 pts of MAPE regression, and exactly 75% sign consistency all ship, matching EVALUATION_POLICY.md's "≤" wording. Score movement was **0.2 pts** (69.3% → 69.5%, six mutants); the value was never the score. **One false survivor caught before publishing**: mutmut called the `ensemble_combine` length-guard survived, but a subprocess test kills it — coverage tracing sees nothing in a child process, so mutmut maps that test to zero functions and never runs it. That is why adjudication is a step. Survivors are bucketed three ways because raw mutmut output is unreadable: 259 of 722 only rewrite string constants or structlog arguments and survive by construction. **No gate** — advisory weekly run until the baseline is stable, the limitations are quantified, and enough of the 463 logic survivors are adjudicated to know the equivalent-mutant fraction. Incidental find, since fixed (entry above): **one test takes 36.8s — 76% of the entire 2,690-test unit suite.** [#376]
- **2026-07-17** **Anchor redesign PR D — the flip. ADR-009 recorded; #304 closes; the arc is complete.** `anchor_conditioning` flipped ON per the study's two-tier verdict (blast radius: ~4 broken-class BAs whose forecasts were the fleet's worst; the settled-grade meter is the post-flip verification — LDWP/IID converging toward ~14% over the coming week, Feed-limited pills clearing themselves as the visible signal). ADR-009 lands in PRD §10 with the full alternatives trail (skip-to-stale measured worse; churn refuted by the study against my own plan; bulk refuted by the PSCO counterexample end-to-end; static config sets rejected for the live classifier; cron offsets refuted by the settle-curve measurement) + CLAUDE.md mirror + HOW_IT_WORKS anchor paragraph + the INTERVIEW_PREP capstone story (six refuted hypotheses → instrument-first). **Closes #304** — both halves now fixed: the drift-metric half by settled-grade re-scoring (#318), the forecast-anchor half by conditioning (#324 + this flip). [this PR]
- **2026-07-17** **Anchor redesign PRs A-C: the study ran against real vintage data and refuted my own plan one final time; conditioning ships flag-dark for `broken` ONLY.** PR A+B (#323, merged): the vintage GCS mirror (best-effort, mutation-pinned isolation; closes #312's flush fragility; unlocks local replay — the thing #309 originally declared impossible) + `scripts/anchor_conditioning_study.py`. **The study's verdict on full-fleet real data** (`docs/ANCHOR_CONDITIONING_STUDY.md`): `broken` — prod anchors **58.2%** wrong vs the BA's own day-ahead at **14.5%**, 90.1% win rate over 103 fresh hours → **CONDITION**; **`churn` → SKIP, refuting the plan's own policy row** — the class mixes BPAT (~14%) with mild churners, and at class level DF *loses* (3.20 vs 4.92, 29.6% win rate); `bulk` → SKIP (the PSCO counterexample confirmed at class scale, DF 9.03 vs prod 2.56); `clean` anchors are literally **0.00%** wrong. Tier-2 end-to-end model replay (prod pickles, same model both arms): every sign agrees — LDWP 16.4→14.3, IID 28.2→26.7, PSCO *worsens* 14.8→17.7 (as predicted), PNM slightly worsens (never condition). **PR C** (this PR): `condition_anchor_frame` — for broken-class regions (live `vintage_summary` read), substitute the frame's own hour-matched `forecast_mw` into the trailing 3 hours **on a fork** (`RegionData.conditioned_demand_df` + `anchor_frame` property); the fork invariant is mutation-pinned (actuals/drift/alerts/weather-corr/diagnostics keep reading the real frame; only features + `_resolve_forecast_start` repoint — the second seam is the SARIMAX Kalman-gap consistency hazard the explorer caught). Ships **dark** (`anchor_conditioning: False`); flag-off is byte-identical (pinned). 4 mutations kill (class-gate inversion, fork-break via no-copy, flag-guard removal, substitution-disable). 2,443 unit green. Next: PR D — flip after ≥48h of `anchor_conditioned` shadow logs, ADR-009, and the Feed-limited pills clearing themselves as the meter converges. [this PR]
- **2026-07-17** **PR 3 — provenance surfaces: the dashboard explains its own data (Feed-limited pills, the operating-summary data note, header-chip fix).** Post-#318 the Models-tab numbers are honest but the *prescription* was wrong: all four LADWP models pilled **Rollback** ("disable this model", H2) when the corrected numbers themselves prove the attribution — Prophet (no AR anchor, #299) at 12.4% vs anchor-fed models at 26-56%, with the vintage study measuring the feed revising 70%. **The attribution rule** (softens verdicts, never creates them; grades/bands/#217 untouched): a confirmed Rollback renders as **Feed-limited** (warning tone, evidence tooltip) only when the model is anchor-fed (`config.ANCHOR_FED_MODELS` — Prophet exempt), the region's `revision_class ∈ {broken, bulk, churn}` (#319 summary key), AND `mean_fresh_revision_pct ≥ 10%` (the magnitude floor — churn is defined by revision *frequency*, and a 1%-hourly-revising BA must not launder a real model failure into a feed excuse). Missing/unknown/clean → today's behavior exactly, which is what makes shipping before the classifier's shakedown completes safe. Headline splits feed-limited names from genuine degradations — an entirely feed-limited panel reads informational, not "investigate". **Operating summary** gains one class-conditional Data note (broken/bulk/churn copy, every number a measured field; clean/unknown/missing → silence — callouts stay rare, which keeps them credible). **Header-chip fix**: `widget_confidence_bar` iterated every freshness key skipping only `timestamp`, leaking "Artifact_Excluded · just now" (and, long unnoticed, "Latest_Data") as source chips — now an allow-list (demand/weather/alerts) matching its sibling callbacks, with the first-ever pin on the chip label set. Scope cut, documented: no `insights.py` integration (zero-I/O contract vs Redis-resident class data). 7-row decision-table matrix from measured prod cases; 4 genuine mutations kill (a first-cut "missing→broken" mutation was WEAK — the rev-guard masked it — replaced with the true both-fields inversion); 2,405 unit green. [this PR]
- **2026-07-17** **Settled-grade drift — the Models tab stops measuring EIA's feed and starts measuring the models (#304 endgame).** Live drift scored each forecast against the actual *as published at tick time* — for high-revision BAs 15-70% wrong and later revised — so LADWP showed **459%** drift with Rollback badges on healthy models and the Overview clause read "live 7d sMAPE 96.9%". The #309 guard only excludes *gross* partials; BPAT-class plausible-but-wrong intermediates still poisoned every record. **Now the metric self-corrects:** new `regrade_records` re-scores stored history against EIA's current view each tick (the fresh guard-cleaned frame covers every record hour — free data), wired into both `compute_drift_payload` (new `actuals` param) and the horizon payload (which already had `actuals` in hand). Load-bearing semantics, all mutation-pinned: **hours absent from the frame are skipped, never treated as agreement** (guard-excluded partials keep their prior value until a plausible revision lands); rebuild only when the value differs after the serializer's own 2dp rounding (no per-tick churn); sMAPE refreshed via the NaN-sentinel path. Emergent correctness for free: a partial-actual record re-graded to its real value **exits the low-actual exclusion** and re-enters the mean. **The #305 revision probe retires** (function + call + tests deleted): vintage (`first_seen_d`, immutability-pinned) owns tick-time truth; the new `drift_regraded` log (n, mean/max shift) replaces the probe's observability. `reconcile.py` untouched — the independence doctrine survives verbatim and Check A becomes the verifier that the regrade works. No schema change: consumers read only aggregates (explorer-verified: nothing reads `records`), so the panel/API/Overview heal with zero UI logic changes (one caption line added). Migration: the first post-deploy tick re-grades ≤720×4×51 records in one pass — LDWP ~459→~50s, BPAT ~12→~8.7 (the probe's own measured settled numbers); Rollback badges that clear were false alarms, any that remain are real. 4 mutations kill (regrade-disabled / absent-as-agreement / stale-sMAPE / churn-guard); 2,383 unit green. Remaining in the arc: PR 3 provenance callouts (narrating now-honest numbers), reconcile Phase 2 deploy, anchor redesign on the maturing vintage class table. [this PR]
- **2026-07-16** **#309 PR 2 — the demand-artifact guard: implausible EIA readings no longer reach the forecast anchor or the region tiles, and every exclusion is disclosed.** The LADWP screenshot case (NOW **730 MW** / 7D LOW 730 / trend −80.6% / "78.1% below average" — an EIA partial that settled to ~3,034 an hour later) is structurally closed. **What shipped:** the #225 detector promoted to `data/quality.py` (US-Grid + `/grid/summary` now import it; three thresholds to config), extended with a **third signal — `D < 0.5 × the BA's own day-ahead forecast`, low-side only, paired with a below-median co-signal**. Signal 3 is load-bearing, not belt-and-braces: stuck partials (IID frozen at 339 for 6+ h, AZPS frozen at 1959) evade both #225 signals and only the day-ahead ratio catches them — while the D==DF placeholder stub (ratio exactly 1.0) and PSCO's legitimate 118-121%-of-DF running can never fire it. The co-signal exists because a **test failure caught a real design edge**: PSEI-class BAs' own day-ahead runs ~47% high, so a bare ratio would false-flag genuine deep troughs. **Scoring job:** `coerce_demand_artifacts` NaN-coerces guard-failing trailing readings ONCE per tick — with the ordering invariant that **vintage capture runs first on the raw frame** (it is the study of these artifacts; integration-pinned: vintage sees 800, the payload sees NaN + disclosure) — then actuals/drift/anchor all consume the cleaned frame; drift stops scoring forecasts against 730-MW partials; `_resolve_forecast_start` anchors on the last real hour. **Web tier (disclosure only — the NaN-fallback math was already pinned):** the actuals payload carries `artifact_excluded [{ts, mw, reason}]`; NOW tile reads "as of HH:MM · N newer readings excluded" with the reason in the tooltip; the operating summary names the exclusion in prose; `/grid/summary`'s `artifact_excluded_regions` now sources from stamped verdicts (read-time detection would go blind on pre-cleaned payloads). **False-positive floor (the #296 lesson): 14,259 settled hours × 46 BAs replayed — 30 fires, ZERO on clean BAs**; IID/PSCO/AZPS/SEC fires are true artifacts persisting in settled data; SPA's 7 (median demand **24 MW** — a rounding-error BA) are unadjudicable and 4/7 fire under the pre-existing guard anyway. Documented residual: BPAT-class +20% HIGH partials are deliberately uncatchable (a high-side signal would false-flag real spikes). 4 mutations (ordering swap, silent-skip, threshold invert, co-signal drop) each kill their test; 2,376 unit green. PR 3 (provenance callouts narrating these verdicts + vintage classes) staged next. [this PR]

- **2026-07-16** **#313 — the vintage instrument was corrupting itself; defense shipped, trigger unidentified; and the #309 lifecycle is now fully verified.** A 3-skeptic adversarial pass over #312's first 14 prod ticks confirmed the fetch-timing findings (fetches spread min 0.7–22.2 across each run, deterministic per-BA order; BPAT's fresh-hour anchor ~16% wrong every hour, matching the #305 probe's 14.1% via an independent instrument; LDWP passes through *multiple* partials — confirmed live: stub 3280 → 554 → 511 → 3251) and **refuted the "synchronized ~12Z EIA re-publication" reading**: bulk daily-file true-ups arrive as a rolling wave 05Z–14Z hitting different BA cohorts at different ticks (magnitudes span 3 orders: DUK ≤0.6% vs SEC/SPA/IID/AZPS 52–77%). C9 resolved by a live daytime boundary probe: the stub→intermediate flip is an EIA **batch republication landing ~+2–4 min after the hour closes** — it beats our earliest fetches (~+4–9 min), so production never anchors on the good stub; and an hour's row is *born* mid-hour (~13 min before close) already carrying the day-ahead placeholder. Grounded in the Form EIA-930 instructions (fetched): same-day files due **within 60 min** of the operating hour, respondents explicitly told to submit **best estimates and resubmit within 3 days**, daily files due 7 a.m. ET — partials are mandated behavior, not a data bug. Ten regions never revise at all (PJM, PNM, PGE, CHPD, GVL, HST, JEA, TAL, TPWR + TIDC *falsely* — its zeros are coerced 0→NaN and skipped, so its brokenness is invisible to the metric). **The verification's incidental find became the urgent work (#313):** four regions' vintage windows (CAISO/ERCOT 12:00Z, FPL/PJM 13:01Z — exactly the earliest-fetched) were destructively re-pinned by unexplained **nil** Redis reads. Trigger systematically eliminated: not read errors (zero logged), not eviction (0 evicted keys, 10% memory), not TTL (24h, gaps ~60 min), not concurrency (taskCount=1, no double executions), not log contamination — genuinely unidentified. Defense makes the corruption impossible without naming the trigger: `RedisReadError` + `redis_get_strict()` (the read-side twin of #268's `persist`) so absence and failure are different answers; a `vintage_seeded:{region}` tombstone that outlives the data key — window absent + tombstone present ⇒ **refuse to write, log `vintage_window_missing_but_seeded` at error level** (turns the unknown trigger into a countable event); vintage writes through `persist` so dropped writes fail loud; record-less payloads refused. All three guards mutation-verified; 2,348 unit green. Fleet-class taxonomy (per-hour churn / bulk re-publication / clean / broken-feed) accumulates in `gridpulse:vintage:*` — the per-BA input the eventual anchor fix designs against. #309 closed by user; full record on the issue. [this PR]

- **2026-07-15** **#309 — demand-vintage capture shipped; three hypotheses about the anchor died first, and the fix is NOT what the issue claims.** #309 (which I filed) says the forecast's `demand_lag_1h` anchor reads EIA's preliminary value, that `corr(revision, settled error) = 0.88` proves it, and that a revision-robust anchor "may be the largest accuracy lever on the board." **Direct measurement against the EIA API refuted the mechanism twice in one session.** Findings, in order: (1) **12/43 BAs publish `D == DF` exactly at the newest hour** — the metered field carrying the BA's own day-ahead forecast for an hour nobody has reported. It is a stub, not a coincidence: equality holds at 0-3% of *settled* hours, and `DF` is published for hours that have not happened (+1h…+4h) while `D` never is, so the value can only flow forecast→actual. (2) Gross artifacts reach the anchor unfiltered: LDWP `1199` at evening peak (26% of its own day-ahead), IID **stuck at `339`** for 6+ hours, TIDC `0`, AZPS revising **1157 → 7815 in four minutes**. (3) I then "refuted" the stub theory — corr(day-ahead error, revision) = 0.15, and a persistence proxy said removing the stub makes things *worse* (6.55% → 7.72%, keeping it wins 9/12). (4) **A user question overturned that too, and it was the right question**: "are we running the forecasts too soon after the hour?" The scheduler is `0 * * * *` — the job fires the instant the hour closes, when the reading is a stub, a partial, or absent; it settles 30-70 min later. Direct stub→settled observation: **BPAT `9008` → `9803` (+8.1%)** against BPAT's 14.2% revision rating; **PNM `2153` → `2126` (−1.3%)** against PNM's 0.7% rating. Those track — my refutation had compared *today's* `DF` against `D` over settled history, but EIA updates `DF` too, so I measured a settled forecast, not the stub that was actually published. **I refuted the right idea with the wrong number.** It also means the skip-test posed a false choice (stub at H−1 vs measured at H−2) when a third option dominates: **the same hour, settled**, obtained by fetching later. **Why capture and not a fix:** `gcs_store` writes `{region}/latest.parquet`, one blob, overwritten hourly — nothing versions demand, so the 0.88 had to be measured by a live probe rather than from history, and no anchor change can be replayed or validated. `data/vintage.py` + a scoring-job phase now pin `first_seen_d` (what the anchor actually used at `:00`), `first_seen_df`, `last_d` (settled), `n_updates` at `gridpulse:vintage:{region}` — pure capture, new key, no forecast behavior change, never fails a run (the drift contract). It does **not** import `models.drift` (`models/` already imports `data/`); join compatibility is pinned by a test instead. Mutation-tested 5 ways, incl. the study-killing one (overwriting `first_seen_d` → 4 tests fail). Also fixed a fixture that built demand frames without `forecast_mw` — a schema the real client never returns, which would have hidden the stub detector entirely. 2,341 unit green. **Open:** a settle-curve measurement (per-BA minutes-after-close until the reading stops moving) decides whether the fix is a cron offset, an unsettled-reading guard, or both. #309 needs retitling — I over-claimed it. [this PR]
- **2026-07-14** **#273 Batch C — EIA null honesty (P2-08), calendar-true is_holiday (P2-14), drift-panel label truth (user-prompted).** **P2-08**: the generation/interchange parsers coerced EIA nulls to 0.0 MW — fabricated readings deflating renewable share and poisoning cache/GCS artifacts. Nulls now parse to NaN (shared `_parse_mw_value`; NO 0→NaN coercion — true zeros are legitimate for a fuel or a tie, unlike demand), an all-null window returns honest-empty at the serve layer, and — the verification pass's HIGH — both fetchers now pass `value_col` so an all-null window routes to the **#174 last-known-good chain** instead of caching a rows-present all-NaN frame for 24h and overwriting the GCS LKG (the first cut had made the outage case *worse than the zeros*: dark surface + destroyed durable copy). **Honestly-scoped residual, documented not hidden**: a PARTIAL null (one fuel at an hour where others report) still reads 0 in the served series — post-pivot fillna(0) can't distinguish a dropped null from an alignment gap; matches pre-fix behavior; fixing it needs nullable payload lists + NaN-aware aggregation across three consumer surfaces (tracked as a #273 follow-up; the first-cut comment/test falsely claimed it fixed and the verifiers caught the false claim). **P2-14**: future-frame `is_holiday` was (hour,dow) group-mean imputed — real in-horizon holidays never read 1, and one holiday in the 28d window smeared ~0.25 onto every future week at that (hour,dow), feeding Prophet's regressor and XGBoost fiction. Now computed directly from future timestamps (prod builder + dev inline mirror), imputer skips it structurally; pinned with a poison-sentinel test. **Drift panel** (user question exposed it): the "Live ÷ Holdout" ratio column invited the cross-horizon comparison the panel's own rules declared non-actionable — removed; headers now carry their leads ("Holdout (168h recursive)" / "Live 1h-ahead (7d/30d avg)"); caption states context-not-comparable; render-pinned. Verification: 17 agents, confirmed set all fixed (all-null LKG HIGH; 3× partial-null false-claim findings; label-fix fully revertible with a green suite; dev-mirror gaps) — 0 refuted. 2,252 unit green. [this PR]
- **2026-07-14** **#273 Batch B — warming honesty (P2-35) + drift sample-count gating (P2-21).** **P2-35**: "Pipeline is warming up — forecast will appear shortly" rendered FOREVER for regions whose forecast is persistently unavailable (models never trained / forecast phase failing past the 24h TTL). The escalation is **evidence-based, claiming only the permanence its evidence earns** (the first cut over-claimed and the adversarial pass caught it twice): (a) forecast payload exists but can't serve the selection → "unavailable_selection" with hedged per-run copy (a one-tick model failure heals next hour); (b) forecast key absent + `_pipeline_alive` (fresh actuals ≤3h) + **`_scoring_pass_completed_since_actuals`** (meta:last_scored ≥ actuals scored_at — the pipeline provably had its chance) → "unavailable" with the "won't resolve on its own" copy. The completion check exists because within one scoring pass actuals land minutes BEFORE forecast/alerts — without it the permanence claim would show during genuine post-flush warming. Alerts gate escalates identically ("Risk data unavailable" + Unavailable chip). **P2-21**: the Overview "live 7d sMAPE" headline gated on `n_records` — TOTAL retained history, count-trimmed not age-trimmed — so a week-scale accuracy claim could rest on a handful of in-window observations. Drift payloads (live + horizon rollup) now emit **n_7d/n_30d = per-window POST-low-actual-filter counts** (the honest denominators of the means); every consumer gates each window's figure on its own count (Overview headline, Models drift-panel warming, 24h-grade helper, drift-by-horizon cells — all with legacy fallback for one migration tick), warming rows no longer print the means the chip just declared meaningless, the panel's N column shows the gate's own denominator, and n_7d/n_30d are exported on the public API drift fields. Verification: 18 agents; confirmed set all fixed (headliners: the two permanence-over-claim MEDIUMs above; a HIGH unpinned-render-copy gap — the annotation copy could be reverted with a green suite; four more mutation-surfaced unpinned gates) and 3 refuted with reachability evidence (incl. the region-dead warming-forever case — the #174 last-known-good chain keeps actuals alive, so the escalation does fire for delisted-BA scenarios). 35 tests in the batch file; 2,230 unit green. [this PR]
- **2026-07-14** **#273 Batch A — "labels tell the truth": 4 of the 15 misleading-numbers ledger items fixed (P2-26, P2-23, ledger-3, P2-42).** A 15-agent triage first verified every ledger item against HEAD: **13 still live, 2 partially fixed** (the stress halves of P2-29/P2-44 died with #265/#266), none fully dead — and produced the batch plan (A: surface labels [this PR]; B: warming/drift-sample honesty P2-35+P2-21; C: data integrity P2-08+P2-14; D: risk tab P2-44-then-P2-29; E: measurement integrity P2-15/16/17+ledger-23, needs training-run validation; P2-19 standalone). Batch A: **P2-26** — the Forecast fast path titled the payload primary's series "XGBOOST Demand Forecast" (XGBoost-calibrated bands wrapped around another model's numbers) whenever the xgboost column was missing; every label now resolves through a shared `_served_model_for_payload` helper (trace/color/title/bands/caption/insights/withheld state + the model-metrics card, which had stayed dropdown-keyed — an XGBoost MAPE bar above a PROPHET-served chart) with an on-chart substitution disclosure on both render outcomes. **P2-23** — the Generation panel's "Net Load (avg)" hero silently substituted average TOTAL generation on every cold page load; now an honest degraded cell (— / "demand data unavailable"), incl. the all-NaN→"nan MW" edge. **ledger-3** — falsified ensemble claims ("almost always beats", "never worse than the worst", "self-correcting") purged from models/ensemble.py, README, CANONICAL_FACTS ADR-004, HOW_IT_WORKS (also stale 1/MAPE → (1/MAPE)³); all now state the measured story (error decorrelation; beats XGBoost-alone 17/51). **P2-42** — Alt+2/3/4 landed one tab left of intent and Models was unreachable (4-entry map positional against 5 pills); now Alt+1..5, browser-verified live; dead contradictory Python TAB_KEY_MAP deleted. Adversarial verification: 17 agents, **11 confirmed / 3 refuted**, all fixed pre-PR (headliners: the substitution-blind metrics card; withheld-state missing the substitution bridge; substitution log firing before the sufficiency check — recording serves that never happened; 5 mutation-verified label pins added). 2,195 unit tests green. [this PR]
- **2026-07-11** **#296 — 30-day SARIMAX forecasts degenerated on 8/51 BAs (SC/PSCO/SCEG/PJM/PACW decay toward or through 0 MW; AZPS/BPAT/LDWP grow 1.5–1.8×); root cause was double integration, not the mechanisms first hypothesized.** Found from user prod screenshots. First-pass hypotheses (no-intercept mean reversion; explosive AR roots) were **published to the issue and then overturned by payload forensics**: reconstructing the actual prod pickles (SC/PSCO/BPAT + healthy DUK control) showed every AR/MA root stationary-side and the decay *linear through* zero — drift. Mechanism: auto_arima could select d=1 on top of the force-enforced seasonal D=1; two zero-frequency unit roots put a **linear trend in the forecast function**, slope estimated from the training window — so the model converts the region's recent weather regime (PNW July heat ramp ⇒ BPAT up; CO monsoon cooldown / post-heat-wave Carolinas ⇒ PSCO/SC down) into a permanent line, while the exog weather response contributes only ±0.03–1 GW at day 30 vs 3–6 GW of drift. Fleet sweep on all 51 real payloads: **8 degenerate, all 8 d+D=2; 0 false positives across the 39 d+D=1 BAs** (also caught PJM decaying negative — nobody had screenshotted it). **Layered fix:** d pinned to 0 in auto-selection (D=1 stays), **d+D≤1 cap on every path** incl. cached meta.extra orders (healed order round-trips so the cache converges in one cycle), doubly-integrated defaults corrected, fit-time 720h sanity check w/ safe-default refit (never raises; `long_horizon_ok` observability field), **serve-time per-horizon guard uniform across every served series (each model + the ensemble)** writing `horizon_guard` {max_ok_horizon, flagged_horizon, reason} into the forecast payload, and a Forecast-tab **withheld state that says why** ("failed the long-horizon sanity guard") instead of drawing fiction — the #227 by-horizon philosophy on the serving surface. Ensembles stay unguarded-in-blend by documented decision (verified sane on all 3 reported BAs; inverse-MAPE³ keeps bad inputs small; cap heals at source). Guard band: floor 0.5× / ceil 1.6× recent 28d envelope + 0.40× sustained-drift on ≥15-day slices, `config.LONG_HORIZON_GUARD_*`. +34 tests across checker/fit-guard/serve-guard/UI; 430 green across the affected sweep. The **#282 floor** was working as designed throughout (clamped SC/PSCO at 0) — a floor can't fix a degenerate model, and BPAT's explosive polarity has no symmetric ceiling: the guard is the containment. [this PR]
- **2026-07-11** **Verify-close sweep: #193 and #174 closed on file:line evidence; #196 re-scoped to its real remnants; stale-comment note on #275.** Three parallel evidence-trace agents audited long-open issues against current `main`. **#193 (demo alerts as real)** — fully resolved: real NOAA wired (#204), demo paths dev-gated + disclosed, outages degrade honestly, stress decoupled from demo alerts; closed w/ 10 evidence items. **#174 (EIA outage resilience)** — fully resolved: `_EIACircuitBreaker` + uniform stale-cache→GCS fallback across demand/generation/interchange + tests, hardened further by #269/#270; closed w/ 11 evidence items. **#196 (interval calibration)** — partially resolved (holdout self-calibration, substitute disclosure, #283-3b lead-resolved widening all shipped); re-scoped to: per-model backtest prediction vectors, caption pooled-count semantics + legacy-key dedup, Overview hero flat-width. Roadmap now truthful: audit critical tier closed, #296 the newest live item. [gh-only, this session]
- **2026-07-11** **#272 (P2-11) — a GCS blip can no longer negative-cache the model store; failures now serve last-known-good.** The audit finding: one failed `latest.json` read stored the `{}` sentinel in the pointer cache, making all 51 regions × 3 models unloadable — originally for the process lifetime; #280's 300s TTL had already shrunk that to ≤5 min, but the residual was still real: the failure path **overwrote a previously-good pointer with `{}`** and cached the failure at the full success TTL, and during a scoring run (fresh process, ~all 51 region loads starting inside the window) a single blip at run start could silently skip most of the fleet's forecasts — silently, because pointer-miss reads as `no_model`, which the #267 semantics correctly treat as expected-not-an-error. **Fix (three layers in `_read_latest`): (1) in-call retry** (2 quick attempts, 0.5s/1s backoff) absorbs sub-second blips with no semantics change; **(2) last-known-good on failure** — a prior pointer (or a prior legitimately-empty read) keeps being served instead of `{}`; the pointer only changes on the daily training write, so stale-by-minutes is safe (the same last-known-good philosophy as the EIA #269 and gate-map #271 fixes); **(3) failure-aware TTL** — only a cold process with no prior value caches the sentinel, and then for `_LATEST_FAILURE_TTL=30s` (re-probe in seconds), not 300s. Legitimately-missing pointers (fresh bucket) and dev/GCS-off remain valid empty results at the normal TTL — failure and absence are no longer conflated. Writer-refresh and `invalidate_latest_cache` clear the failure flag. +5 tests (last-known-good served over `{}` — the P2-11 core; cold-failure 30s re-probe + recovery; missing-pointer keeps success TTL; in-call retry heals a blip silently; GCS-off not-a-failure) + reset-helper/invalidate updated; 300 green across the persistence/model-service/gate/jobs sweep; ruff clean. Closes #272 — the last standalone critical from the 2026-07 buried-ledger audit (#267–#272 now all fixed). [this PR]
- **2026-07-11** **#283 Phase 4 — `weather_normal_tail` flipped ON: the days-17-30 forecast tail is live on the weather-normal, gated on real demand evidence.** The go-live gate ran as a **retrospective demand spot-check** against realized actuals: DUK, forecast origin 2026-06-10 (so days 17-30 = Jun 26–Jul 10 had 336/336 realized hours), XGBoost trained only on pre-origin data, both tail modes on otherwise-identical features, using the **production 10-year artifact** (built by the nightly training job, verified 366×24/17-var). Result: **tail MAE 3,442 → 3,146 MW (−8.6%), MAPE 21.1% → 19.2%**, no negatives — the recent-28d window had filled the late-June tail with May-diluted 72.4°F weather while the normal correctly said 80.1°F; the origin straddles the early-summer ramp, exactly the phase-lag case the Phase-0 weather backtest predicted (normal ~10:2 at seasonal turns across 6 BAs). Honest framing recorded: both variants under-predict a hot late-June (~20% MAPE absolute) — a climatology-class tail cannot see a heat wave, which is what the P10–P90 fan + divider label disclose; the **delta** is the decision number. **Flip mechanics:** one flag + docs. Per-BA graceful fallback makes the early flip safe — 20/51 backfilled BAs upgrade immediately, the rest join as the nightly refresh lands (~Jul 15). Docs in the same PR (CLAUDE.md end-of-PR check): ADR-008 (PRD §10) gains the #283 update (weather-normalized normal-weather-year, level-anchored; option-3 "Light conditional climatology" marked realized-and-exceeded; revisit-triggers marked fired), HOW_IT_WORKS day-16 paragraph rewritten, and the Forecast-tab regime subtitle re-worded to the honest umbrella "seasonal climatology baseline" (the web tier can't tell which tail mode the scoring job used per-BA during backfill, so the label claims only what is always true). [this PR]
- **2026-07-10** **#220 — the Models tab's 4 residual panels were PERMANENTLY empty in prod (structural, not warming); now populated from real walk-forward-backtest residuals.** User-spotted live on prod (ERCOT: metrics + SHAP populated, all 4 residual panels showing "no scored forecast to compare against yet"). **Root cause:** `write_diagnostics` sourced from the legacy v1 `get_forecasts`, which the #149 strict gate makes return "unavailable" on the job container **every tick, for every region** — so the #166 interim honesty fix (correctly) wrote the unavailable marker forever, and the panel copy's "populates after the scoring job writes a forecast" was a false promise no scoring tick could keep. **Fix — populate from the data that already exists:** the diagnostics phase now reads the nightly training job's Redis backtest payloads (`backtest:{forecast_exog}:{region}:{horizon}` — genuine holdout actual/predictions/residuals, the same source the #283 Phase 3b P10–P90 band calibrates on), preferring the **24h horizon** (day-ahead error, the operational standard) with 168h/720h fallback. The payload now carries **provenance** (`residual_source`: kind/horizon/model/exog_mode) and names the series `predicted` (the old "ensemble" field name would mislabel an XGBoost backtest series; the UI reads `predicted` with a legacy-`ensemble` fallback for in-TTL payloads). The UI **captions all four charts** ("24h walk-forward backtest residuals · XGBOOST") so holdout residuals are never mistaken for live-forecast residuals, and the unavailable copy now states the TRUE self-heal condition ("populates after the nightly training job writes its walk-forward backtest") — reachable only pre-first-training-run. v1 `get_forecasts` dependence deleted from the phase. 10 tests rewritten/added (no-backtest honest marker w/ reason; real residuals + provenance; 24h-preferred + deeper-fallback; feature-importance None w/o model; unavailable render w/ truthful copy; populated render w/ provenance captions on all 4; legacy-field back-compat w/o false provenance); **A 3-lens adversarial verification confirmed 6 findings on the first cut** — notably a latent crash in my own defensive branch (a malformed backtest payload with short timestamps would have written residuals with empty timestamps, crashing the whole Models-tab callback in lttb_downsample — worse than the honest empty state — plus index-synthesized hour-of-day, a fabricated attribution) and a mutation-verified test gap (re-adding the v1 `get_forecasts` call — the #220 root cause — passed all tests). **Fixed:** the horizon gate now validates the WHOLE payload shape (≥24 aligned rows across actual + chosen series + timestamps; a malformed payload loses to the next horizon), the defective defensive branch is deleted, the reader gained a belt-and-braces length guard (mismatch → honest unavailable, never a dead tab), and `_FORBID_V1` (get_forecasts → AssertionError) is patched into every job-side test so the root cause can't silently return. Tests 10 → 15 (short-payload loses to deeper horizon / only-short → unavailable / xgboost preferred in multi-model payloads with residuals matching the xgboost series / misaligned-timestamps skipped / reader degrades honestly on malformed payloads); 269 green across the sweep; ruff clean. Closes #220 (found + diagnosed same-day from a user prod screenshot). [this PR]
- **2026-07-10** **#283 Phase 3b — the Forecast-tab uncertainty band now WIDENS with lead time (P10–P90 fan), replacing the flat empirical offset.** The pre-3b band applied one q10/q90 pair — pooled across every lead of a single backtest horizon — uniformly over the whole chart: too wide at hour 1, too narrow at hour 720, and visually implying day-30 is as knowable as day-1. New `_widening_interval_from_backtests` (components/_callbacks_shared) anchors the SAME empirical residual quantiles per backtest horizon (24h/168h/720h, real holdout error at increasing lead) and `_add_confidence_bands` interpolates them across the lead axis with **monotone widening enforced** (cummax/cummin — a 720h anchor sampling narrower than 168h is single-origin noise; uncertainty can't shrink with lead) and the band's P10 edge **floored at 0** (#282 consistency). P50 = the forecast line itself, so the band IS the P10/P50/P90 fan; past the day-16 divider it visibly flares — the honesty cue the ADR-008 label alone didn't carry. Caption + legend updated ("P10–P90 empirical outcome range — widens with lead time (anchored on 24h/168h/720h backtest residuals{, xgboost-calibrated})"), preserving the P1-2 substitute-calibration disclosure. Fallback chain preserved: <2 anchors → the old flat empirical band → the labeled heuristic envelope (all 278 pre-existing band tests pass unchanged on single-horizon fixtures). **Deliberate deviation from the issue scope:** bands are computed in the UI from the backtest payloads it already reads — NOT added as `quantile_bands` on the scoring payload — because that reuses the per-model calibration/disclosure machinery, works for all 4 model views, and avoids growing the hot payload; the public API keeps intervals withheld pending #196 (field allow-list unaffected). Not flag-gated: it's an honesty improvement to the EXISTING pipeline's displayed uncertainty, valid for both the current climatology tail and the future weather-normal tail. **A 3-lens adversarial verification confirmed 11/11 findings on the first cut — the harshest result of the #283 series — headlined by a production-defeating HIGH:** `_collect_backtest_residuals` treats the 168h training-holdout pool (`gridpulse:holdout:{region}`, horizon-AGNOSTIC) as an exact source that beats per-horizon substitutes, so for the default ensemble/prophet/arima views every anchor collapsed onto the SAME 168-residual pool — the 720h anchor silently dropped (size gate), the 24h-vs-168h spread was a window-subsetting artifact that cummax then monotone-locked into a **fake fan**, and the deep tail rendered under-wide beneath an 'anchored on backtest residuals' caption that was actually the training holdout. **Fix: the widening estimator collects horizon-RESOLVED residuals only** (`horizon_resolved=True` skips the holdout source — the flat estimator's 'right model beats right horizon' trade-off inverts for a lead-resolved band, so a disclosed per-horizon xgboost substitute wins; flat path unchanged + regression-pinned). Also from the pass: anchors now pinned at their pool's **effective lead (~H/2)** — a horizon-H backtest pools leads 1..H, so its quantiles measure mid-window error, not lead-H error (the remaining deep-tail known-narrow bias is documented in the docstring); an **edge-ordering clamp** (a systematically over-forecasting model's negative q90 could invert the band after the zero-floor); captions extracted to a shared `_interval_caption` helper so the two chart paths can't drift; `__all__` exports. +18 tests incl. the holdout-poisoning pin (720 anchor survives, spread is genuine, substitution disclosed), flat-still-uses-holdout regression pin, effective-lead interp, precedence (flat never consulted when widening available), degenerate-anchor ordering, and 3 caption tests; 336 green across the band sweep; ruff clean. [this PR]
- **2026-07-09** **#283 — days-17-30 forecast tail: fan-out study → weather-normalized ("normal weather year"), shipping in phases; Phase 2 wires it (flag off).** Follow-up to #281/#282: the acute negative-demand/summer-decline bug was fixed with a *recent-28d* climatology tail, which a user correctly flagged as a workaround. A **5-method fan-out study** (seasonal-analog / re-enable-model-yearly / weather-normals / specialized-long-range / SME-literature, scored on correctness/accuracy/cost/architecture-fit/load-growth/honesty/robustness) found the user's seasonal-analog *diagnosis* right (a seasonal-**phase** lag, not level bias) but the sharper *implementation* is the textbook-MTLF **weather-normal**: drive the existing XGBoost through the tail off a per-BA `(day_of_year, hour)` ERA5 normal — ERA5 reaches 1940 so **every** BA gets a normal (multi-year EIA demand is empty pre-2019 for DUK / thin for the 35 V3.ζ BAs), keeps one weather→demand model across all 720h, and leaving the **autoregressive** demand features on the recent-28d window anchors the tail to *current* load (load growth handled with no explicit ratio). **Phase 0** (weather-error backtest, `scripts/phase0_weather_normal_backtest.py`) → **GO**: across 6 climate-diverse BAs the normal beats recent-28d ~10:2 at seasonal turns (often halving temp MAE), a wash mid-season (satisfies the ship-gate), with a deep-winter-persistence caveat that *validates the hybrid's anomaly-blend*. **Phase 1** (#286) built the artifact (GCS-durable + tiny Redis staleness marker after verification caught a ~210MB Redis-bloat + a training-timeout-ordering bug). **Phase 2 (this PR)** wires the tail: `_overlay_weather_normal_tail` swaps recent-28d weather+derived for the normal past the Open-Meteo boundary (injecting the **directly-stored** derived normals — Jensen-correct — and recomputing only `temp_x_hour`/`temperature_deviation` from the injected temps), keeping autoregressive on recent-28d; behind `weather_normal_tail` (still **off**) with graceful recent-28d fallback when an artifact isn't backfilled; scoring-job-only, reading the normal from GCS via an in-process cache. **Validated on real DUK data:** the 369 Open-Meteo-covered hours are byte-identical off-vs-on, the 351 climatology-fallback hours get the calendar-correct Aug normal (78.7°F, tracks the recent 82.7°F — no cold-decline), autoregressive unchanged. Flag stays OFF pending prod artifact backfill (the training job builds them over days, capped) + a demand-MAE spot-check before the go-live flip. +10 tests (tail-injection flag-off no-op / injection / covered-hours-untouched / all-covered no-op; cached loader); 88 green across the affected sweep; ruff clean. Phase 3 (seam anomaly-blend + P10/P50/P90 honesty bands) + Phase 4 (ADR-008 doc flip) follow. [this PR]
- **2026-07-09** **#281 — the 30-day forecast declined through peak summer and went NEGATIVE; quantifying the cause overturned the leading hypothesis.** A user flagged the Forecast tab (DUK 30-day): Prophet/Ensemble sloped down through July–August and crossed **below 0 MW** (Prophet min −6,062), XGBoost stayed sane. Physically impossible + seasonally backwards, on the flagship surface. **Rather than fix from intuition, I reproduced DUK on real data and decomposed Prophet's `predict()` components — and the numbers rewrote the diagnosis.** I'd fingered the climatology weather reversion; it was only **21%**. The dominant **78%** was Prophet's **`yearly` seasonality**: `create_prophet_model` set `yearly_seasonality=True`, but the model trains on a ~90-day window — you can't identify a **365-day** Fourier cycle from <3 months, so it extrapolated a spurious **−11.8 GW** swing. The negatives themselves come from Prophet's **additive composite** (trend + seasonality + regressors): the docstrings claimed `floor=0` "structurally prevents negative forecasts" — **false**, floor/cap bound the *trend* only, and `predict_prophet` returned raw `yhat` unclipped. The tuned `changepoint_prior_scale=0.001` — the one knob touched to stop "long-horizon drift" — was the one thing working (trend Δ −159 MW). **Fix (evidence-ranked, each verified against the DUK decomposition): (1)** enable `yearly_seasonality` only when the span ≥ `YEARLY_SEASONALITY_MIN_DAYS`=730 (off in prod; the weather regressors already carry the annual signal) → min −2,543 → **+7,586**, 0 negatives, decline −15,204 → −3,955; **(2)** build the `(hour,dow)` climatology from a recent 28-day window (`CLIMATOLOGY_WINDOW_DAYS`) instead of the full 92 days (which diluted July in cool April–June data — measured 9.4°F cold-bias, CDD halved) → climatology 73.0°F → **80.6°F**, day-16 step −1,111 → +807, decline → **−1,164 MW** (a plausible ~7% wiggle); **(3)** hard non-negative floor at the serve layer (all models + ensemble in the scoring job) **and** in `predict_prophet` (covers the dev inline path + Prophet's band) — don't trust Prophet's floor. Final DUK: **min +10,150 MW, 0/720 negative** (was −6,062). Had I skipped the quantification and gone straight to "floor + climatology," a ~12 GW phantom decline would have survived. +8 tests (yearly-gate on/off, predict clip, recent-window climatology + short-history fallback, serve-layer floor across models/ensemble); reusable `scripts/diagnose_forecast_decomposition.py` added; 2224 unit+integration green (only the 4 pre-existing `pyarrow`-absent failures), ruff clean. Docs: ADR-008 (PRD §10) + HOW_IT_WORKS climatology-window update. Closes #281. [this PR]
- **2026-07-09** **#271 (P2-10) — the forecast-quality gate now reads a scoring-job-published verdict (Redis-only), instead of fataling open on an outage and sweeping GCS metas per render.** The gate decides which BAs appear in the dropdown + US-Grid. It had three coupled defects: it **failed open** (any Redis/GCS exception → `None` → pass, so an outage silently stopped hiding rollback-grade BAs and users could select unusable ones); it **swept per-render GCS metas** on cold Redis (`get_model_metrics → get_model_metadata → latest.json`, ×51 — a web-tier I/O-guardrail violation); and a **process-lifetime pointer pin** meant the web tier never reflected daily retraining. #255/#260 only changed *which* MAPE it read. **Fix — move the decision to where the data lives:** the scoring job computes each BA's verdict from the real holdout metrics it already writes and publishes a consolidated `gridpulse:meta:gate_status` (`{region: {acceptable, best_mape}}`, 24h TTL); the web gate reads that **one key, Redis-only** — no GCS in the request path. Its TTL makes it self-healing (a later scoring failure leaves the last-known verdict in place, so an outage no longer *forgets* which BAs are unusable — the same last-known-good spirit as the EIA fix). The champion-MAPE logic is now a pure `gate_verdict_from_metrics` (shared by the job's publish + the dev inline path). **Fail behaviour (user-chosen):** when Redis has *no* verdict at all (cold/flushed — the app is already warming everywhere), the gate passes but **logs** (`forecast_gate_status_unavailable_pass_open`) — no longer *silent*, no dropdown blackout, no new UI plumbing; health/freshness reflect the outage via the existing missing-scoring signal. **Part D (pointer pin):** `_read_latest` gained a 300s TTL so the web tier picks up retraining (bounded — ≤1 GCS read/TTL, not per render); writer-refresh + `invalidate_latest_cache` stamp/reset it. Dev/offline (`REQUIRE_REDIS` False, nothing publishes) still computes inline from local metrics — which is why all pre-existing gate tests pass unchanged. No HOW_IT_WORKS diagram impact — the change makes the gate *conform* to the already-documented Redis-only web tier (it was the exception). **A 3-lens adversarial-verify workflow caught a HIGH-severity bug in the first cut pre-merge:** a *completed-but-degraded* scoring run (model-store outage → `load_model` returns None for every BA → no verdicts) published an **empty** `gate_status` map that clobbered the last-known-good one on the same 24h key; the web tier read present-but-empty as "every region warming → visible", silently **un-hiding rollback-grade BAs** serving stale bad forecasts — the exact fail-open the fix set out to remove (my "self-healing" claim only held if the job *crashed*, not if it finished degraded). Fixed by **merging** each run's verdicts over the last-known map (preserving un-scored regions) and **skipping** the publish entirely when a run yields zero verdicts, plus web-side defense-in-depth (empty/malformed map → the pass-open+log path, never silent-visible). +22 tests (pure verdict logic, prod Redis-verdict path incl. asserting `get_model_metrics` is never called, no-verdict pass-and-log, **merge-preserves-last-known / skip-on-degraded / empty-map-not-silent / one-Redis-read-per-sweep cache / real `_score_region` wiring**, pointer TTL expiry/invalidate); 2217 unit+integration green (only the 4 pre-existing `pyarrow`-absent `test_gcs_store` failures remain, identical on clean main), ruff clean. Closes #271. [this PR]
- **2026-07-09** **#265 — Risk-tab "Grid Stress" was alert-count arithmetic (pinned at 100); replaced with demand/capacity utilization.** A user asked why grid stress reads 100 nearly everywhere. Root cause: the score was `min(100, n_crit*30 + n_warn*15 + 20)` over NWS alert counts (`jobs/phases.py` + duplicated twice in `_callbacks_alerts.py` — review P2-29/P2-44, folded into #189, never surfaced as a standalone issue). It saturates at ~3 alerts, and a multi-state BA pulls dozens of county-level advisories — so it measured "is there weather anywhere in the footprint" (≈ always), not stress. **The irony: wiring real NOAA (#204) made it worse** — the old demo alerts kept it at a tame 20-35; real volumes blow past the cap. **Fix (chosen option #1):** grid stress is now supply-tightness — `demand ÷ nameplate capacity` via a shared `models.pricing.grid_stress`, reusing the #254 `UNRELIABLE_CAPACITY` exclusion (import-dominated / peak-derived BAs → `—`, no fabricated number). Decoupled from the alert feed entirely (a NOAA outage no longer nulls it); NWS alerts demoted to context (the "Components" breakdown relabeled "Active NWS alerts"). Utilization bands: <70 Normal / 70-85 Elevated / ≥85 High. Collapsed the 3 duplicated formulas toward one (the job writes it, the fast path displays it; dev fallback shares the helper) — addresses P2-44's single-source ask. 5 new `grid_stress` unit tests + updated the alert-honesty / fast-path-tone / callback tests to the new behavior; 244 affected tests green, ruff clean. Filed as #265; found by surfacing a buried critical-review finding. [this PR]
- **2026-07-08** **#253 — web-tier operational guard: code shipped, GCP-apply steps handed off.** The genuine P0 from the blindspot pass: the public JSON API (#250/#251) made the stateless web tier *publicly programmable*, but every piece of the project's operational tooling (job-failure alerting, deep /health, circuit breaker) protected only the JOB tier — the now-public, `--allow-unauthenticated`, personal-billing request path had no rate limit, cost guardrail, or web-tier alerting. **Shipped in-app + tested:** (1) a Redis-backed fixed-window **per-IP rate limiter** (`ratelimit.py`) on `/api/v1/*` (120/min) and the Dash callback route `/_dash-update-component` (600/min) — global across the 1-4 instances, **fail-open** on any Redis error (a limiter that failed closed would self-inflict the outage it prevents), enforced only in Redis-only mode (staging/prod) so dev is unthrottled; (2) `MAX_CONTENT_LENGTH` (2 MiB) so an unbounded POST can't OOM a 4Gi worker; (3) **`/health` minimal-public** — public callers get liveness `{"status"}` only, the detailed body (Redis state / last-scored / cache counts) + `?deep=1` gated behind the `/metrics` IP allowlist via a shared `_is_internal_caller` (deny-by-default: `METRICS_ALLOWED_IPS` unset → localhost only). **Authored as-code, user applies on GCP** (billing + monitoring are account settings I don't touch): `web_service_5xx_alert.json`, `web_service_max_instances_alert.json`, and README `gcloud` recipes for a billing budget (+ forecasted-spend *anomaly* rule — the cheapest, highest-leverage guard) and an uptime check on public `/health`. The deep-`/health` alert the monitoring README had listed as a follow-up is now delivered (adapted to shallow `/health`, since deep is allowlist-gated). Rate limits are env-tunable (clamp a flood without a redeploy). 16 new tests (limiter fail-open/window/bucket + 429 contract + health/metrics gating) + full app-boot/health suite green (82); ruff clean. Docs: HOW_IT_WORKS §1 web-tier-guard note, deploy-prod.yml cost-ceiling comment. **A 3-lens adversarial-verify workflow caught 4 issues pre-merge** — a major one where the uptime-check content matcher `"status": "healthy"` (spaced) would never match Flask's compact `{"status":"healthy"}` and fire a permanent false-down alert; an `instance_count` `REDUCE_MAX` that undercounts the active/idle split; a missing shared-NAT exemption (added `RATE_LIMIT_EXEMPT_IPS` for the control-room persona); and a stale env-var name in the deploy comment — all fixed. #253 stays OPEN for the GCP apply steps + Cloud Armor (code half done). [this PR]
- **2026-07-08** **#255 fixed — the forecast-quality gate now judges the served model, not XGBoost-alone.** The gate (`is_forecast_quality_acceptable`, on in staging/prod) hid a BA when its **XGBoost-only** holdout was in the 7d rollback grade (>22%) — but production serves the ensemble (ADR-004). It hid **SEC** (XGBoost 38.63%) even though its served ensemble is 13.61% and best base (Prophet) 11.22%, silently making the flagship "51 BAs" grid render 50 in prod. **Fix:** gate on the **best achievable** holdout MAPE — the champion `min` across the ensemble + 3 base models (`get_best_holdout_mape`) — so a BA is hidden only when *no* served model reaches the acceptable grade. **Key subtlety caught pre-implementation from `_holdout_table.md`:** gating on the ensemble *alone* would have **newly hidden SPA** (ensemble 22.81% > 22, but XGBoost 21.13% < 22) — swapping one hidden BA for another. The champion gate keeps both visible; **0/51 hidden** now, so the "51 BAs" claim is honest again. Source: the new path reads via `get_model_metrics` (Redis-**first** in prod: `forecast:{region}:1h.model_metrics`, which the scoring job writes with per-model + ensemble rows), so in the normal warm-Redis case the gate does **0 GCS reads** (the old XGBoost-only gate always read a GCS meta) — a partial step on **P2-10/#189**. Honest caveat: on the cold-Redis warming window it still falls through to GCS metas (now 3 base metas vs the old 1), so it's Redis-*first*, not Redis-*only*; the full Redis-only + fail-closed redesign remains P2-10/#189. Reconciled the docs the change made stale: the US Grid "N hidden" **tooltip** + collector docstring, the `layout.py` + `config.py` gate comments (the config example wrongly implied the gate fires on SPA — now kept visible), API `quality_gated` comment, and `CANONICAL_FACTS` (new "UI-visible count is quality-gated" row) — a 3-lens adversarial-verify workflow caught the tooltip + comments + the earlier GCS overclaim. 10 gate tests rewritten to the served-champion seam + 3 new #255 cases (SEC visible, SPA not newly hidden, hidden-only-when-all-rollback); 260 gate/metrics tests green, ruff clean. Closes #255. [this PR]
- **2026-07-08** **#254 fixed — the circular National-Utilization / top-stress bug.** The blindspot pass's sharpest finding, and a wrong-number bug in this session's own #244/#251 capacity code. 5 BAs (SOCO/DUK/CPLE/PSCO/FMPP) carry `REGION_CAPACITY_MW = peak×1.15` (a reserve proxy, not a measured plate) yet weren't import-dominated, so utilization = demand/(peak×1.15) fed the KPIs **self-referentially** — a BA at its own peak reads exactly ~87% (1/1.15) and can neither win top-stress honestly nor mean anything in the national-utilization denominator. **Fix:** a canonical `config.PEAK_DERIVED_CAPACITY` (the 7 peak×1.15 BAs — those 5 + HST/CPLW, which are already import-dominated) + `UNRELIABLE_CAPACITY = IS_IMPORT_DOMINATED | PEAK_DERIVED_CAPACITY` as the single exclusion set, applied to every stress surface: the US Grid KPI top-stress + National Utilization, the stress sort, the per-card chip (peak-derived now shows a neutral **"est."** chip, not a circular util %), both map hovers (**"· est. cap"** suffix), and the public API `/api/v1/grid/summary`. The API `/regions` field was renamed **`nameplate_capacity_mw → capacity_mw`** with a `capacity_source` enum (`nameplate` | `peak_estimate`) — a pre-adoption contract change on the week-old API (#250/#251), honest before anyone depends on the wrong name. **SPA is the deliberate carve-out** — import-dominated but a *true* federal-dam nameplate, so it stays `capacity_source=nameplate`. Corrected the docs the bug had desynced (`CANONICAL_FACTS` wrongly listed SPA as peak×1.15 and omitted the 5; README/TECHNICAL_SPEC/api index notes overclaimed "nameplate"). **A 3-lens adversarial-verify workflow caught 3 real doc/note gaps pre-merge** (a `/grid/summary` note that conflated "excluded" with "peak_estimate" — false for SPA/PACW; two stale TECHNICAL_SPEC rows), all fixed. 10 new tests + updated API contract test; 218 US-grid/API tests green, ruff clean. Closes #254; accredited-capacity/ELCC remains #243. [this PR]
- **2026-07-08** **Three honesty quick-wins shipped (PR #257, merged) — partial #256 + partial #127.** The cheap, high-value half of the 2026-07-07 blindspot pass. **(1) Data-source attribution:** Open-Meteo weather is CC-BY-4.0 (its credit must travel with redistributed data); EIA-930 + NWS are public-domain but credited. Added a "Runtime data sources" table to `THIRD_PARTY_NOTICES.md`, an `attribution` field on the public-API index **and every data payload** (the redistribution surface — CORS is `*`), and a CC-BY credit **link** on the Open-Meteo footer token. **(2) Scenario what-if disclosure:** the Forecast Scenarios panel runs a hidden linear weather-sensitivity heuristic (`_scenario_demand_factor`), not a model re-forecast — added a one-line caption under the panel header so the Δ deltas aren't read as calibrated predictions. **(3) demo-vs-stale relabel:** the dev-only demo-fallback path tagged synthetic `generate_demo_*` output as freshness `"stale"` (which means *real cached data past its window*) → corrected to `"demo"` (the taxonomy's synthetic state, already used by the sibling no-API-key branch); bonus correctness, `forecast_source` now reads `"simulated"` not `"api"` for that path. Updated the 3 `test_callbacks_v1_paths` tests that encoded the old mislabel (their docstrings already said "demo data is used") + added 2 API attribution-contract tests; 100 + 111 targeted tests green, ruff clean, runtime-verified all three surfaces render. **Refs #256/#127 — both stay open** (the Open-Meteo commercial-posture decision + the full `scenario_engine` swap remain). [this session]
- **2026-07-07** **Product + open-issue blindspot pass → 4 filed, 3 closed, 2 reframed.** A dedicated adversarial sweep (4 parallel lenses; two independent lenses converged on the same #1) for gaps the existing 28 issues missed, now that the surface has grown (US Grid UX #244/#249, public API #250/#251, #225 guard). **Filed:** **#253** (public API + web-tier operational guard — rate-limit + budget alert + monitoring on the now-live *unauthenticated* `/api/v1` on personal GCP billing); **#254** (*a correctness bug in this session's own #244/#251 code* — National Utilization / top-stress is circular for the 5 peak-derived-capacity BAs where `cap = peak×1.15`, and the API mislabels that as "nameplate"); **#255** (the forecast-quality gate judges the *XGBoost-only* holdout while prod serves the *ensemble* — surfaced on SEC); **#256** (data-source attribution/licensing — since partially shipped, see the entry above). **Closed prod-verified/stale:** #199 (generation Redis-first, verified), #194 (Prophet/SARIMAX time-label, verified), #124 (cross-repo link — trigger stale). **Reframed:** #222 de-escalated (dropped `[P0]` — positioning-in-header is polish, not a launch blocker); #153 recast as the honesty *keystone* (typed Redis contracts structurally prevent the whole fabrication-vs-real class the 2026-07 review kept surfacing). **Re-scoped:** #121 (drift monitoring — parts 1–2 shipped via #126/#128, ensemble-weight integration remains). The two highest-stakes findings (#254 circular-util, #255 gate) were directly verified before filing; I owned that #254 is in my own #244/#251 code. **No product code changed in the pass itself** — issues + tracker hygiene only. [this session]
- **2026-07-07** **#225 closed — US Grid demand-plausibility guard + PACW capacity fix.** Two residual halves of the 2026-07-02 review's least-robust-numbers bug (the reserve half already shipped via #244). **(1) Demand-artifact guard strengthened:** `_is_implausible_demand_artifact` gained a **single-step-collapse** signal — a >60% one-hour drop from the prior real reading that also lands <60% of the day's median is a dropped/partial EIA point (the reported "APS 0.6 GW / −90.7%" and "LADWP −68.6%" cases, which sat *above* the old 10%-of-median floor and rendered as bogus 6%/10% utilization). Both conditions are required, so a gradual overnight trough (descends over many hours) and a return-from-spike (lands at median) are NOT flagged. `prev_mw` now threaded to all 4 render sites + the API's `/grid/summary`. **(2) PACW capacity:** PacifiCorp West (2,628 MW nameplate ≈ served load — the fleet is mostly in the East, PACW imports across the system) crowned the top-stress KPI at "100%". Added to `IS_IMPORT_DOMINATED` — the certain, precedent-matching fix (like SPA), so it now shows the "imports" chip and is excluded from stress ranking on all three surfaces (KPI/card/API). **Considered + rejected:** a data-driven `peak>nameplate → importer` generalization — it conflicts with the deliberate `_STRESS_RELIABLE_CEILING=2.0` design (which intentionally treats the 100-200% band as *genuine* high stress, capped-display) and would need a ~1.0× threshold I can't defend without prod data; caught during self-review before it broke `test_displayed_stress_caps_at_100_percent`. 9 new tests (6 artifact-signal + PACW); 154-test US-Grid/API sweep green. Note: wall-clock staleness (flag a real-but-hours-old reading) is NOT added — all reported symptoms were glitch/artifact values, not old-but-real ones; the timestamp isn't threaded into the render path. Closes #225. [this PR]
- **2026-07-07** **Public read-only JSON API v1 shipped (#250) — the "platform" claim gets a programmatic surface.** New `api.py` blueprint on the existing Flask server: `GET /api/v1` (index) · `/regions` (51 BAs + nameplate/import-dominated/quality-gate metadata) · `/forecast/{region}?horizon=` (ensemble + per-model hourly series, holdout metrics, ensemble weights; horizon deliberately capped at **168h** — the weather-driven week; the payload's 720-row climatology tail (ADR-008) is not exported as if weather-driven) · `/grid/summary` (national demand / simultaneous 24h peak / utilization / top-stress — same helpers **and same artifact filter** as the US Grid KPI bar, exclusions disclosed via `artifact_excluded_regions`) · `/drift/{region}` (live 1h + horizon-matched grades). Architecture cost ~zero: the web tier was already stateless + Redis-only, so the API is thin routes over the same `gridpulse:*` keys. **Honesty contract:** `scored_at`/model provenance everywhere; cold cache → 503 warming + Retry-After (never fabricated data); unknown region → 404, raw input never reflected; **intervals omitted until #196's per-model calibration**; capacity labeled nameplate (#243). **Adversarial 3-lens review (security/correctness/honesty) caught 6 substantive defects pre-merge, all fixed:** (1) DoS amplification — the fan-out endpoints (~100 Redis reads/request) had no server-side cache → 30s in-process memo of success bodies; (2) `/grid/summary` skipped the UI's `_is_implausible_demand_artifact` filter → API and dashboard would have *disagreed* on national totals whenever a #225-class glitch reading existed → now mirrored + disclosed; (3) deny-list field export would auto-publish any future cache-schema field (e.g. #196's uncalibrated intervals) → allow-lists on models AND fields; (4) `public, max-age` on 503s would let CDNs serve "warming" past the fix → errors now `no-store`; (5) false horizon-cap rationale (payload has 720 rows, not 168) → documented as a deliberate honesty cap; (6) missing doc obligations → HOW_IT_WORKS §1 (API-client edge + surfaces note), TECHNICAL_SPEC §2.5, CANONICAL_FACTS API row. Residual accepted: the quality gate can still touch GCS on a 10-min TTL miss (pre-existing UI path too) — follow-up candidate: scoring job writes the gate to Redis. 21 unit tests + real-app boot smoke; README "Public API" section. Closes #250. [this PR]
- **2026-07-06** **US Grid UX Phase 2 shipped (Haiku-orchestrated) — the Highest-Stress KPI is now a jump-to-card.** Clicking "Highest-Stress Region · PACW · N%" smooth-scrolls to that BA's card and flashes it (~2.4s ring), staying on the US Grid tab — pure clientside (pattern-matching id `{"type": "us-grid-kpi-jump", "region": <top>}` → `app.clientside_callback` parses the region from the triggered id, finds the card via its stringified dict-id in the DOM, `scrollIntoView` + flash class; dummy `dcc.Store` output; guarded against the re-mount-fires-with-n_clicks=0 pattern-input quirk). **Scope honesty:** the original Phase-2 plan made *two* KPIs clickable, but "Lowest Reserve" became **National Utilization** (#244) — a national aggregate with no card target — so Phase 2 is the Highest-Stress KPI only, clickable **only in the Cards view** (Map/Polygons have no card to scroll to; the items builder takes a `view` param). Foundation: `build_metrics_bar` gained an optional `cell_id` key (id + n_clicks + role=button + tabIndex + clickable class — same additive pattern as Phase 1's `help`). 4 Haiku file-owning agents (foundation-first) + verify; my diff review found the output faithful this time (no Phase-1-style layout bug). 6 new tests; 113-test US-Grid sweep green. Remaining from the proposal: **Phase 3** (Forecast grid-snapshot strip + cross-tab model-story reconciliation) and the optional Phase 4. [this PR]
- **2026-07-06** **#217 de-alarm shipped (P2-25) — the 1h Live-Drift panel now needs its verdict CONFIRMED before crying wolf.** With the horizon-drift keystone answered (resolved 24h grades exonerating Prophet/SARIMAX — see the click-through entry below), the panel's remaining dishonesty was issuing the H2 governance verdict "Rollback" (= disable the model) from the 1h band alone. New **verdict-confirmation rule** in `_build_drift_panel`: a model above the 5% 1h band is labeled "Rollback" **only when its own 24h-ahead grade** (`drift_horizon`, #227) is also `rollback`; otherwise a descriptive warning-tone **"Off band"** chip whose tooltip explains the missing-1h-anchor structure and points to Drift by Horizon. Headline alarms only on a confirmed rollback. **Adversarial 3-lens review caught 4 real defects pre-merge, all fixed:** (1) *unbounded fail-open* — the horizon key has a 24h TTL, so a dead horizon feed would have parked every off-band model on "still resolving" forever, silently disabling the alarm path → now **fails closed**: feed absent (it writes hourly even with 0 resolved records) ⇒ off-band models alarm as unverified; (2) present-tense "healthy at 24h" from ≥24h-lagged evidence → copy now says "healthy on matured 24h-ahead scores (≥24h lag)"; (3) unknown grade vocabulary vetoed the alarm as *health* → unknown grades map to pending; (4) `AttributeError` on non-dict payload intermediates → hardened. Trade-off disclosed: a genuine breakage's *confirmed* alarm lags ~24h (visible warning-tone "Off band" in the interim; the paging path for real outages is job alerting + deep /health, not this panel — and this panel's alarm history was 100% false positives). Header kept as "Status" (review: "1h Grade" misdescribed cross-horizon verdicts). 8 new/updated tests (28 drift-panel tests green). Closes #217. [this PR]
- **2026-07-06** **Live-app click-through — 8 issues closed prod-verified; THE KEYSTONE ANSWERED: horizon-graded Prophet/SARIMAX are healthy.** Browser walk of all 5 tabs on deployed `966e40f`. **The #235/#236 horizon-drift instrument resolved its first records and answered the drift keystone (FPL):** Prophet 24h **5.41% Acceptable** / 48h **2.59% Excellent**; SARIMAX 24h **2.87% Target** / 48h **1.92% Excellent**; ensemble 24h 3.59% / 48h 2.04% (72h "—", resolves from 07-07) — while the adjacent 1h Live-Drift panel still badges Prophet ×2.20 / SARIMAX ×2.67 "Rollback". Same models, same week: the cross-horizon cry-wolf (#217) is now *demonstrated on screen* by its own honest replacement; remaining #217 work is de-alarming the 1h panel (P2-25). The 1h live-7d window still contains mostly pre-#226 days — expect it to drop as the window turns over (~07-11). Also live-verified this session's UI arc: Phase-1 ⓘ tooltips + honest ensemble caption ("Production blend — most accurate single model: XGBoost 2.5%"), **National Utilization 57%** KPI, sort dropdown, self-labeling card badges (util/net/Δ1h/demand) with no legend, 4-card Risk weather row (83°F — no nan°F), anomaly band warm from display start, scored-at chips on Overview + Forecast, neutral operating summary, sMAPE labeled, real (non-fabricated) metrics on every model card. **Closed prod-verified: #197 #198 #200 #201 #202 #203 #218 #219** (evidence in each). Commented live evidence on #217 + #220 (residual panels now honest-empty + will populate as snapshots resolve; empty-state text renders clipped — micro-fix task spun off with the "typically exceeds" National-Peak tooltip nit). **Watch-list observations:** FPL ensemble holdout (2.32%) now beats XGBoost (2.50%) — the k=3 world visible in prod; Highest-Stress reads "PACW · 100%" (the #225 artifact class — #225 still real). [this PR]
- **2026-07-04** **Deploy + verify sweep — this session's work is live; drift keystone is now a genuine WAIT.** Merging #241/#242/#244/#245 auto-triggered Deploy→Production (`workflow_run` on CI-green main, not a manual step); it succeeded on main tip `966e40f`. **Verified live in prod:** `/health?deep=1` healthy (FPL 720-row forecast ok, Redis up, `last_scored` fresh ~17m); the reserve-relabel is deployed (prod HTML meta reads "monitor grid utilization", not "reserve margin"); scoring job runs hourly, clean (no WARN/ERROR in 90m) on the new image; **alert-honesty holding** (no `generate_demo_alerts` fallback firing — #191/#200 confirmed live in prod); both `drift_updated` (1h) and `horizon_drift_updated` (#235) fire across all regions. **Keystone drift-watch can't conclude yet, two ways:** (1) the #235 horizon-matched drift has **0 resolved records (pending≈39/region)** — snapshots resolve 24-72h out, so the honest day-ahead Prophet/ARIMA grade won't exist until ~2026-07-05→07; (2) per **#170**, the 1h `drift_updated` log exposes only the *sample* model (arima: e.g. WALC 13.7%, PNM 11.5% rolling-sMAPE-7d), so per-model Prophet drift isn't readable from logs — #170 is the confirmed blocker to log-based verification. **Action:** re-check horizon-drift resolved records in ~1-3 days to close the #170/#181/#226-vs-#194 loop; the rendering-dependent issue closures (#217-#220 etc.) still need a live-app click-through. Also closed #223 (relabel shipped #244; NERC → #243). [this PR]
- **2026-07-04** **Reserve-margin honesty (#223) — relabel to what it actually is; national KPI → utilization; true PRM deferred (#243).** #223 flagged that "reserve margin" ships as a utilization-complement, not NERC `(cap−peak)/peak`. A 3-agent grounding workflow found it worse than #223 said: **only 3 of 6 reserve sites are live, and none use the NERC formula** (US Grid used a current-demand complement; scenarios + briefing used `(cap−peak)/capacity`). The keystone finding: **`REGION_CAPACITY_MW` is EIA-860M *nameplate*, not NERC-accredited** — a literal NERC PRM on our data reads ~66% nationally (vs the real ~15-25%). Blindly shipping #223's formula would swap a visibly-wrong number for a plausible-but-fabricated one (accredited capacity needs per-fuel ELCC we don't have). **Decision (user-confirmed): don't fake it — relabel honestly + defer the accredited-capacity model to #243.** Shipped: one canonical helper `models/pricing.capacity_headroom_pct` (+ `utilization_pct`); the degenerate US-Grid "Lowest Reserve" KPI replaced by **National Utilization** (Σdemand ÷ Σnameplate over the reliable-capacity BA set — excludes import-dominated, so no absurd >100%; the national *average* complementing Highest-Stress's per-BA *max*); and every other "reserve margin" mislabel (scenario `Δ Headroom`, briefing capacity headroom, + 3 dead sites: persona-KPI, `scenario_engine` keys, accessibility ARIA) routed through the helper and renamed. Nothing on any surface now calls a nameplate number "reserve margin." Codebase-wide grep clean; 268 tests green + adversarial-verify pass. Filed #243 (ELCC model for a real PRM). Resolves the **presentational half of #223** (NERC-PRM half is #243) and the US_GRID_UX_PROPOSAL Open Question (Lowest-Reserve degeneracy). [this PR]
- **2026-07-04** **US Grid tab UX — Phase 1: definitions & honest labels (Haiku-orchestrated).** From a live screenshot review of the US Grid + Overview/Forecast tabs, wrote a phased proposal ([`docs/internal/US_GRID_UX_PROPOSAL.md`](docs/internal/US_GRID_UX_PROPOSAL.md)) covering 5 problems: undefined KPIs/badges, non-actionable summary KPIs, incoherent US-Grid→Forecast drilldown, an Overview-vs-Forecast horizon mismatch, and an inconsistent cross-tab model story. A 3-parallel-reader + adversarial-verify workflow grounded two of the user's questions: (a) **Overview peak/low/avg are 7-day *actuals* (fixed 168h) while Forecast peak/avg/min/range are the *forecast* over a user-selected horizon** (24h/7d/30d) — different data *and* window, so the fix is explicit labeling, not forcing a match; (b) the "Ensemble default for HST" is the **Overview model card hardcoding `if 'ensemble' in metrics: primary='ensemble'`** (`_callbacks_overview.py:579`), featuring the ensemble's 4.06% even though XGBoost (3.70%) is the best single model there — while the Forecast selector hard-defaults to XGBoost and the Backtest tab names the per-region best (three tabs, three stories). Not a math bug (the ensemble is the production blend per ADR-004, which trails on headline MAPE by design) — purely presentational. **Shipped Phase 1** (additive, no data/nav/behavior change): `build_metrics_bar` gained an optional `help` key → ⓘ tooltip, `build_model_metrics_card` gained a `caption`; US-Grid's 4 KPIs + Overview's 5 KPIs + Forecast's 4 metrics now carry definitions, a per-BA card-badge legend was added, and the Overview model card now captions "Production blend — most accurate single model: XGBoost X%". Orchestrated with 4 Haiku file-owning agents (one file each, foundation-first); **diff-review caught one Haiku layout bug** — the legend was a direct child of the `.gp-region-grid` CSS grid without `grid-column: 1/-1`, so it'd have squished into one card's width (fixed to span full width like the section headers). Verified: ruff clean, all 4 modules import, render smoke test passes. **Discovery worth a design call:** *Highest-Stress Region* and *Lowest Reserve* are the **same BA** showing complementary numbers (util% and 100−util%) — and the reserve half intersects **#223** (reserve should be NERC (cap−peak)/peak, not the utilization-complement my tooltip honestly documents). Phase 1 relates to #222/#223/#224 but closes none; Phases 2 (clickable scroll-to-BA KPIs) + 3 (Forecast grid-snapshot strip + cross-tab model reconciliation) remain. [this PR]
- **2026-07-04** **Risk-tab UI fixes — Current Conditions cards + anomaly-band warm-up.** Two prod bugs the user caught in a screenshot. **(1) Only a Temperature card showed** (and it looked cramped as a lone narrow card). Root cause: the scoring job's `alerts:{region}` payload wrote only the temperature series, so the web tier could render just one card — `_build_weather_context` already builds temp/wind/humidity/cloud but the Redis path never fed it. Fix: scoring job now emits `weather_current` (latest temp/wind/humidity/cloud); the Redis fast path builds the full 4-card row via `_build_weather_context`. Also hardened `_build_weather_context`: `pd.Series` coerces None→NaN and archive-unstable `wind_speed_80m` (#164) arrives NaN, so the old `is not None` guard would have rendered "nan" cards — now `pd.notna`-guarded, with a proper 80m→10m wind fallback. **(2) The ±2σ anomaly bands started ~24h after the demand line** (bands began Jun 29, demand Jun 28). Root cause: `rolling(24)` computed on the 168h *slice* left the first 24h NaN. Fix (both the scoring-job writer and the dev compute path): compute the rolling band on the **full** demand series, then slice to 168h, so the window is already warm at the display start. 5 new weather-context tests; full unit suite green. No API/behavior change beyond the payload's added `weather_current` field. [this PR]
- **2026-07-04** **#184 (parts 1+2) — one ensemble-combine path + vectorized ARIMA NaN-fill.** From the #189 audit. (1) The inverse-MAPE combine existed in 3 places; `training_job`/`phases` already routed through `models.ensemble.ensemble_combine`, leaving `model_service._predict_from_trained` and `_simulate_forecasts` hand-rolling their own weighted sums. Routed both through `ensemble_combine` — **provably parity-preserving**: `_predict_from_trained` pre-fills the old per-model `1/len` default for a base model missing from the weights dict (then `ensemble_combine` renormalizes over models present, equal-weights on zero-total = the old `np.mean` branch); `_simulate_forecasts`'s weights already sum to 1 so renormalization is a no-op. Both are dev/demo paths (strict-gated out of prod since #149). (2) Replaced the two hand-rolled O(n) per-element ARIMA NaN-fill loops (`arima_model.py` train + `_get_exog`) with vectorized `pd.DataFrame(exog).ffill().bfill().fillna(0)` — the same three-step fill `prophet_model.py` already uses — **byte-identical** to the loop (direct parity check on leading/interior/trailing/all-NaN). 86 model-service+ensemble tests, 102 arima+ensemble+service tests, full suite green. **Deferred (own decisions): part 3** (shared `_holdout` helper — touches the *published* holdout numbers, higher-stakes) and **part 4** (Prophet mode tuple — the modes document each regressor's natural mode even though the loop forces additive under logistic growth; keep-vs-drop is a real design call, not a mechanical de-lie). **#184 stays open** for those. Refs #184. [this PR]
- **2026-07-04** **#185 (part 1) — collapsed the 3 triplicated EIA fetchers into one shared helper.** `fetch_demand` / `fetch_generation_by_fuel` / `fetch_interchange` repeated the same ~30-line skeleton (cache → paginated fetch → stale→GCS→typed-empty fallback → parse → cache+GCS write) three times, differing only in data_type, params, parser, and empty columns; the #174 uniform GCS fallback was copied thrice. Extracted `_fetch_eia(*, region, data_type, cache_key, endpoint, params, parser, empty_cols, use_cache)`; the 3 public fetchers are now thin callers that just compute their dates/key/params. Behavior-preserving: all **73** `test_eia_client.py` tests green (verified no test/alert depends on the now-unified log event names), full unit suite green. **News-honesty part deferred as a product call:** #185 also flagged `news_client` returning demo articles on failure — but on investigation it's a *deliberate 2-layer* design (`news_client` failure paths **and** `_build_overview_news`'s empty/except branches both inject `_get_demo_news`), so the widget shows fabricated headlines rather than "No news available" (which `build_news_feed` supports). Whether to keep demo news for demos vs. honest-empty is the user's UX call; **#185 stays open** for it. Refs #185. [this PR]

  > **Superseded 2026-08-11 — the reserved product call was lost, and is now
  > answered by deletion.** This entry says "#185 stays open" for the
  > news-honesty decision; `gh` says 185 was closed COMPLETED that same day
  > (2026-07-04) when its EIA-fetcher half shipped, taking the reserved UX
  > question with it. Nothing has tracked it since — a deferred decision
  > parked on an issue that closes for its *other* half leaves no trace.
  > Answered the honest way in the GP-P1-04 step-1 delete: `data/news_client.py`,
  > `_build_overview_news`, `cards.build_news_feed` / `build_news_card`, the
  > orphan `news-store`, and the `.news-*` CSS are gone, so there is no widget
  > left to choose demo-vs-empty for. Scope + rationale: the step-1 comment on
  > 221.
- **2026-07-04** **#171 — scoring-runtime creep guardrail (the recurrence-preventer).** Picked up #171 ("parallelize scoring") and found the parallelization was **already done** — `run()` submits `_score_region` per BA to a 4-worker `ThreadPoolExecutor`. The genuine gap the 2026-06-01 incident exposed is the one the issue's acceptance criteria emphasize: **no early warning**. The PR-G10 alert only fires on an *outright* timeout (~1700s under the 1800s cap), by which point a tick is already killed; and STATUS's own post-incident runs (1083–1333s = 60–74% of the cap) show runtime is already creeping. Shipped the guardrail: `_check_runtime_headroom` tracks consecutive runs whose `elapsed_s` exceeds 70% of `SCORING_TASK_TIMEOUT_S` (Redis-persisted streak, since each run is a fresh process) and emits a `scoring_runtime_creep` ERROR log after 3, matched by a new Cloud-Monitoring **policy-as-code** alert (`docs/monitoring/scoring_runtime_creep_alert.json`, log-based on `jsonPayload.event`). Config `SCORING_TASK_TIMEOUT_S=1800` verified against the deploy `--task-timeout`; fixed a stale deploy-prod comment claiming the run is "sequential" (it's 4-worker parallel). 7 tests; guardrail is best-effort (never fails the run). **Runtime *reduction* (more workers — Open-Meteo throttles under bursts — or incremental trailing-hours fetch) is the follow-up the guardrail now makes safe to schedule instead of paging.** Updated CANONICAL_FACTS + SCHEDULED_JOBS. **Deliberately does NOT close #171** — the guardrail is done, but the runtime *reduction* (the issue's <600s target) remains its open work; #171 stays open for that, now with an alarm to schedule it. [this PR]
- **2026-07-04** **#227 shipped — horizon-matched drift series (24h/48h/72h).** The only live accuracy signal was 1-hour-ahead drift, which structurally condemns Prophet/SARIMAX (no last-value anchor). Built the deferred "part 2" of `models/drift.py`: a snapshot → resolve → grade pipeline that snapshots each forecast's +24/48/72h predictions, re-scores them against the now-known actual N hours later, and grades each horizon against **its own** `MAPE_BY_HORIZON` band — so a competent day-ahead model grades "excellent" at 24h instead of "degraded" at 1h. New Redis key `gridpulse:drift_horizon:{region}`; new `write_horizon_drift_metrics` phase wired after the 1h phase (isolated, can't block scoring). **Adversarial red-team (+ my own review) caught a critical bug pre-merge:** horizons were keyed off wall-clock `scored_at` (`datetime.now()`, sub-hour) vs hour-aligned forecast rows → snapshots would have been silently empty in prod; fixed to anchor on the first forecast row, guarded by a regression test. Verified: horizon-matched grading works, resolution math correct, pending buffer bounded at ~504 snapshots (~100KB/region) under sustained no-actuals. 13 new tests. **This is the instrument that lets the #226 stale-Kalman-vs-#194 live-drift split finally be measured**, and a prerequisite for horizon-aware ensemble weighting (deeper #181). UI surface for the series deferred as a follow-up. Closes #227. [this PR]
- **2026-07-04** **#226 fixed — SARIMAX served with a stale Kalman state.** The daily-trained SARIMAX is pickled lean (params + 240-row tail) and reconstructed + Kalman-filtered at predict time, anchoring its state at `train_end`; the hourly scoring tick then asked for a "1h-ahead" that was really a `(gap+1)`-step (up to ~24-step) forecast from a stale origin that never saw the intervening actuals — no analogue to XGBoost's `demand_lag_1h`. **Fix:** `predict_arima` now accepts `gap_actuals` and advances the state via statsmodels `append()` (re-filters with the same params, no re-estimation) before forecasting; `jobs/phases._predict_one` extracts the gap-span real demand from `featured` and passes it. **Offline-validated** (daily-train/hourly-score simulation, all real data): 1h-ahead MAPE ERCOT 0.87% → 0.34%, CAISO 6.27% → 1.79%, MISO 3.62% → 0.53% (2.5–7×), landing SARIMAX in the 0.3–1.8% range. Re-combines observed data (no retrain); guarded to fall back to the stale path on append failure. 9 new tests; full unit suite green (4 pre-existing gcs env failures only). **Gating caveat:** the issue notes some of the live 13% is a shared #194 time-mislabel artifact — the live-drift attribution needs the post-#194 re-score (prod); this fix isolates + fixes the stale-Kalman component. Also **closed #195** (holdout commensurability, resolved by #209/#232/#233). Closes #226. Added INTERVIEW_PREP §13. [this PR]
- **2026-07-04** **#181 resolved — ensemble re-weighted on the honest data ((1/MAPE)¹ → (1/MAPE)³).** With the commensurable recursive holdout published (#232), revisited the ADR-004 inverse-MAPE blend that #181 flagged as trailing best-base. Regenerated the per-model recursive holdout **prediction series** for all 51 BAs (reusing the production `_holdout_metrics_*` helpers so it's production-faithful) and swept the weighting exponent k. **Plain inverse-MAPE (k=1) is dominated** — beaten by k=3 on **47/51 BAs** (median 4.19% → 3.90%); only 2/51 BAs prefer k≤1. Set **`ENSEMBLE_WEIGHT_EXPONENT = 3`** (config + `compute_ensemble_weights`, one line, reversible; supersedes #229's k=1 groundwork). Confirmed not-overfit via a held-out even/odd-hour split (k=3 wins median + tail there too) + an independent reimplementation (my first adversarial-verify workflow silently failed its output schema, so I re-verified by hand). **Corrected ADR-004's rationale:** the blend's value is **error-decorrelation** on BAs with 2+ comparable models (CAISO 4.55% → 3.51%, AZPS 13.4% → 8.2%), NOT tail variance-reduction (XGBoost-only owns the tail) — the ADR had it backwards. Updated ADR-004 (PRD + CLAUDE), added the sweep table to `BACKTEST_RESULTS.md`, added INTERVIEW_PREP §12. Offline-validated (re-combines existing model outputs, no retrain needed) but a served-forecast change — watch live ensemble drift post-deploy. Closes #181. [this PR]
- **2026-07-03** **Keystone re-measure shipped — published accuracy tables refreshed to the honest recursive numbers.** The 2026-07-03 04:00 UTC `gridpulse-training-job` run retrained all 51 BAs on the deployed #209 recursive-holdout image (`trained_at` 05:30 UTC across every meta). Ran `scripts/export_holdout_metrics.py` against prod GCS and refreshed `BACKTEST_RESULTS.md` / `_holdout_table.md` / `CANONICAL_FACTS.md` / `README.md` — the whole 51-BA table + all three distributions now report **recursive multi-step** MAPE, superseding the pre-#209 teacher-forced numbers. **Before → after (headline moves):** XGBoost-only median **2.32% → 4.32%**, best-base median **2.30% → 4.12%**, ensemble (served) median **3.48% → 4.82%**; ERCOT best ensemble **1.79% → 1.48%**; worst-5 ensemble now SPA 22.81% / IID 15.37% / PSCO 14.69% / SEC 13.61% / AZPS 12.70%. **The ensemble's standing improved under honest scoring:** it now beats XGBoost-alone on **17/51 BAs (was 4/51)** — as errors compound over the horizon, blending Prophet+ARIMA damps single-model drift (SEC XGBoost 38.63% → blend 13.61%). Best-base winners: XGBoost 44 / Prophet 4 (CHPD, DOPD, SEC, SOCO) / ARIMA 3 (ERCOT, SC, WALC). Also added a per-horizon (day-ahead vs 7-day) recursive breakdown to `BACKTEST_RESULTS.md` and filled INTERVIEW_PREP §11 with the real before/after. **The number went up (~2×) and the trust went up with it — the recursive figure is the honest 7-day-forecast MAPE, not a one-step nowcast.** No product code changed — docs only. [this PR]
- **2026-07-02** **Product-judgment pass + live prod review → 9 issues filed; deploy-status corrected.** Synthesized user product notes into an Overview **decision-briefing** plan ([`docs/internal/OVERVIEW_DECISION_LAYER_PROPOSAL.md`](docs/internal/OVERVIEW_DECISION_LAYER_PROPOSAL.md)): **GP-P1-04 decided — delete the dead ~849-line Overview briefing surface, rebuild an honest Redis-backed DEMAND/MODEL/RISK/DECISION briefing**, grounded in industry standards (NERC RML 15%/10% + per-BA reserve-margin references §11, day-ahead MAPE 1–3%, CDD/HDD 65°F base). **Live-reviewed prod headless (Overview/Risk/Models/US-Grid/Forecast):** positioning is invisible in the header; the Forecast tab already does the DEMAND block well (reuse it); surfaced live issues — Models "Live Drift" labels every model "Degraded" (P2-25, cries wolf on XGBoost 1.53% live), `nan°F` current temp, alert-red on a routine metric, 4 empty Models panels, US-Grid reserve-margin artifact (PACW "0% reserve") + demand-data artifacts (APS 6% util at −90.7%). **Filed #217–#225** (4 live bugs, GP-P1-04 tracking, P0 positioning, reserve-margin convention, shell hygiene, US-Grid robustness); rest folded into the proposal doc / #189. **Deploy-status finding:** `main` is already deployed to prod (#215, Deploy→Production 20:08 UTC — verified via GH Actions); the pending keystone is the **training re-run + re-measure, NOT a deploy**, and STATUS's "force=True" was inaccurate (no force flag is wired; the data-hash resume invalidates naturally). Corrected the keystone + Next-3 above. **No product code changed — all plan + issues + docs.**
- **2026-06-19** **#176 verified fixed in prod + docs refreshed with the now-populated ensemble column.** PR #179 merged 01:39 UTC, Deploy → Production 01:48 UTC, so the 04:00 training run (`gridpulse-training-job-rx7q5`) was first on the fixed image: **`ensemble_holdout_persisted` 51/51, `ensemble_holdout_unavailable` 0** (was 51/51 on `fewer_than_two_holdouts`). Closed #176 via `gh issue close`. Re-ran the exporter and refreshed `BACKTEST_RESULTS.md` + `CANONICAL_FACTS.md` to show three distributions (XGBoost-only / best-base / ensemble). **Honest finding worth keeping:** the **ensemble trails best-base in aggregate** — ensemble median **3.48%** / p90 **8.37%** / max **27.40%** (AZPS) vs best-base median **2.30%** / p90 **6.57%** — and beats XGBoost-alone on only **4/51** BAs. Not a regression: the inverse-MAPE blend (ADR-004) still weights the 3–5×-weaker Prophet/ARIMA, so it lands above the strongest single model; its value is tail variance-reduction (AZPS XGBoost 33.97% → blend 27.40%), not a headline win. Docs now say: quote ensemble for *what production serves*, best-base for *best achievable per BA*. Tail BAs swing run-to-run (AZPS best-base 11.90%→26.68% in two days). Refs #176.
- **2026-06-18** **Root-caused the empty ensemble holdout column → model-boundary NaN fix ([PR #179](https://github.com/kristenmartino/gridpulse/pull/179)).** The ensemble holdout-MAPE was `—` for all 51 BAs (#176). The #178 self-heal diagnostics surfaced *why*: `ensemble_holdout_unavailable reason=fewer_than_two_holdouts valid_holdouts=['xgboost']` for every BA — Prophet **and** ARIMA holdout fits both die on NaN in the archive-unstable `wind_speed_80m` regressor (#164), leaving only XGBoost (which tolerates NaN natively). With `<2` valid base holdouts, `_ensemble_holdout_metrics` returns `None` and the ensemble row is never written to `meta.json`. Two distinct crashes: Prophet `fit` raises `Found NaN in column 'wind_speed_80m'`; ARIMA `_get_exog`'s `np.isnan` guard raises `ufunc 'isnan' not supported ... casting rule 'safe'` on an **object-dtype** column *before* its own ff/bf/zero fill could run. **Fix (this session, PR #179):** coerce exog to float in `_get_exog` before the isnan check, and sanitize each Prophet regressor (`to_numeric → ffill → bfill → fillna(0)`, matching `predict_prophet`) so a gappy column degrades gracefully (signal dropped, model retained) instead of dropping the model from the ensemble. The #178 diagnostics did their job — surfaced but didn't fix the root cause; this is the fix. **5 new unit tests; targeted suite 22 green; ruff clean.** Verification deferred to the next daily training run (confirm `extra.ensemble_holdout_metrics` populates + the exporter's ensemble column is no longer `—`). Refs #176.
- **2026-06-17** **Refreshed holdout accuracy to real all-51-BA numbers; surfaced the ensemble-metrics gap.** An audit of the headline "MAPE across 51 BAs" framing found it unbacked by code — accuracy is computed **per-BA, per-model** (168h holdout), there is no across-51 aggregate, and the only published numbers (`BACKTEST_RESULTS.md`: stale 2026-02-21 ERCOT 3.13% / FPL 7.51% snapshot, self-flagged as pre-leakage-fix). Ran `scripts/export_holdout_metrics.py` against production GCS (reads the per-BA holdout the daily training job already writes to each model's `meta.json` — no retraining), 51/51 BAs resolved. Refreshed `BACKTEST_RESULTS.md` (full 51-BA table) + `CANONICAL_FACTS.md` to the real **best-base-per-BA** distribution: median **2.28%**, p90 **6.57%**, max **21.0%** (SPA), min **0.79%** (ERCOT); XGBoost best for 50/51 BAs (ARIMA for AZPS — best-base pulls AZPS 29.4%→11.9% and the max 29.4%→21.0%). Regenerable CSV/intermediate gitignored. **Ensemble holdout left "pending"** — `extra.ensemble_holdout_metrics` is absent from every xgboost meta in prod (the training-job post-hoc write isn't landing), so the Models-tab ensemble row is empty; not fabricated. Filed #176 for that gap (root-cause fix tracked separately, not in this PR). Refs #176.
- **2026-06-04** **Second scoring-failure event — an EIA *API outage*, not runtime creep; built outage-resilience ([#174](https://github.com/kristenmartino/gridpulse/issues/174)).** The job-failure alert fired again; the 03:00 + 04:00 UTC scoring ticks timed out (each ran the full 1800s ×2, reaching ~16/51 BAs) and **self-healed at 05:00** — every tick since ran a healthy ~700s (well under the 1800s cap). **Resisted the reflex to bump the timeout again.** Root cause from `gridpulse-scoring-job-trjwg` logs: `api.eia.gov` returned sustained `HTTP 504` + 30s read-timeouts for ~2h. Two gaps turned a transient upstream outage into a job failure: (1) **retry amplification** — `_request_with_backoff` spends `MAX_RETRIES=5 ×30s + backoff ≈150s` per failing call *before* any fallback, so `~150s × 3 endpoints × N BAs` overran the task budget; (2) **uneven fallback** — `fetch_demand` falls back to GCS but `fetch_generation`/`fetch_interchange` only checked the cold cache and returned empty (and were never written to GCS). **Fix (this session):** a process-local **circuit breaker** in the EIA client (trip after 3 consecutive hard failures → fail fast to fallback, single-attempt probe to recover mid-run) + **uniform GCS write/read fallback** for generation + interchange. Bounds a total-EIA-outage run to well under the task timeout, serving last-known data. **21 new unit tests; full unit suite 1800 green; ruff clean.** Distinct from #171 (parallelize for normal-runtime margin) but reinforcing. The 06-01 vs 06-04 contrast is itself the lesson: same alert, different root cause — *don't pattern-match the fix to the last incident.* Also confirmed **not** the overdue-bill (billing enabled; zero gaps in hourly runs; a suspension would have shown as failures). STAR story added to INTERVIEW_PREP (#7).
- **2026-06-01** **Scoring-job timeout incident — runtime crept into the 900s cap; mitigated, durably fixed, real fix filed ([#171](https://github.com/kristenmartino/gridpulse/issues/171)).** The PR-G10 job-failure alert fired — **its first real production catch, confirming the email channel works end-to-end** (incident found by alert, not by manual check, exactly as G10 intended). Not a transient miss: **four consecutive hourly scoring ticks (09:00–12:00 UTC) timed out** at the 900s Cloud Run task limit (×2 retries each), each reaching only ~37/51 BAs; `/health?deep=1` went `degraded` with `last_scored` stale ~4.5h. **Root cause:** the healthy run is ~855s (last good `scoring_job_complete elapsed_s=854.97`) — only **45s under the cap** — and runtime had crept there as features landed (ERA5 archive+forecast stitch #163, per-tick drift stats PR-G9). Normal upstream-latency variance tipped 854→>900s. No new deploy (image `d65db55`, docs-only) — a latent creep, not a regression. **Mitigation:** live `--task-timeout` 900→1800s + forced make-up run `gridpulse-scoring-job-pv58v` — **completed all 51 BAs in 1262s** (vs the 855s baseline: upstream latency was elevated, so **1200s would *not* have sufficed — validates the 1800s choice**; but 1262s is already ~70% of the new cap, so #171's runtime reduction isn't optional). Recovery verified: `/health?deep=1` healthy, `last_scored` fresh, `forecast_sample` ok (720 rows). **Durable:** bump committed to `deploy-prod.yml` + `deploy-dev.yml`; corrected a stale `CANONICAL_FACTS` row (scoring "~5 min" → ~14 min) that had **hidden** the creep; added a "repeated timeouts" mitigation case to the `SCHEDULED_JOBS.md` runbook (it only covered transient + data-fault scoring failures). **Raising the ceiling is mitigation, not a fix** → [#171](https://github.com/kristenmartino/gridpulse/issues/171) filed for the real fix (parallelize per-BA fetch/score so runtime *drops*). Mirrors the training-job 7200→18000 bump (2026-05-03). STAR story added to INTERVIEW_PREP (#6). **Durable + scheduled path both confirmed:** PR [#172](https://github.com/kristenmartino/gridpulse/pull/172) merged to main 13:29 UTC, `Deploy → Production` for `6e40abd` succeeded 13:37 (live job timeout reapplied at 1800, not reverted to 900), and the **13:00 + 14:00 scheduled ticks both succeeded** (51/51 — on the live-hotfix then the deployed image), so the scheduled path self-heals under the committed config. Runtime across the three post-fix runs: **1083 / 1262 / 1333s** (60–74% of the cap) — already at the headroom limit, so #171 is promoted ahead of Phase 4 (data commented on the issue).
- **2026-05-30** **Phase 3 deployed + verified in production.** Final cumulative deploy (#169 merge `35cd535`, includes #168) CI-gated → **Deploy → Production success**; service rev `gridpulse-00242-tk6`. `/health?deep=1` → `status: healthy`, Redis up/required, `forecast_sample` ok (FPL, 720 rows), `last_scored` not stale. Scoring job confirmed running the new PR-G9 drift code (new `rolling_smape_7d` / `n_low_actual_excluded_7d` fields present in `drift_updated` logs). **sMAPE bounding confirmed live** (AZPS 266%→104%, LDWP 188%→53%, SPA 79%→43% MAPE→sMAPE) and **normal regions unaffected** (dozens at plausible 15–30%, sMAPE≈MAPE, 0 excluded). Issue state clean (#149/#155/#142/#150 closed, no churn). **One caveat → [#170](https://github.com/kristenmartino/gridpulse/issues/170):** the `drift_updated` log emits only the alphabetical sample model (`arima`, the weakest), not the `ensemble` headline, so the user-facing LDWP number isn't directly verifiable from logs (arima-for-LDWP is genuinely high — model weakness, not the artifact #142 addressed; the metric is now honestly reporting it, bounded + filtered). Checkpoint here before Phase 4.
- **2026-05-30** **Phase 3 closed — PR-G11 (#150) Prophet interval honesty.** Narrow honesty fix, not a calibration redesign: `predict_prophet` was emitting a fabricated "95%" band (`yhat_lower×0.95` / `yhat_upper×1.05`) — an uncalibrated scaling of the real 80% bounds. Investigation showed those `lower_95`/`upper_95` keys were **consumed by nothing outside two tests** (the Forecast/Overview tabs already display *empirical* 80% residual intervals via `models.evaluation`, the PR-B path), so the cleanest fix was to **remove them at the source** rather than fabricate or relabel. Kept Prophet's genuine 80% posterior (`interval_width` defaults to 0.80, so `yhat_lower/upper` are honest). Tests flipped to assert no `lower_95`/`upper_95` survives + the 80% band brackets the forecast. No UI/story change (display was already empirical). Considered Option 1 (real Prophet 95% via `interval_width=0.95`) and Option 3 (empirical 95%) but both add unused output for a band nothing renders — out of scope. **This closes Phase 3 of `prod-readiness`** (#149 fake-fallbacks removed · #155 live drift made honest · #150 interval claim made honest); checkpoint before Phase 4 per direction.
- **2026-05-30** **Phase 3 progress: PR-G4 (#149) merged; PR-G9 (#155) drift robustness in flight; #150 re-close-trap fixed.** (1) **PR-G4 / #149** merged as PR [#167](https://github.com/kristenmartino/gridpulse/pull/167) — strict prod-fallback gating: under `REQUIRE_REDIS`, `get_forecasts` returns `{"source":"unavailable"}` and `get_model_metrics` returns `{}` rather than simulated/hardcoded baselines (the #131-class bug source). (2) **PR-G9 / #155** (this branch) — robust live drift stats for near-zero actuals: added bounded **sMAPE** (`200·|a−p|/(|a|+|p|)`) as the headline drift metric + a **region-relative low-actual filter** (drop drift records with `actual < 0.10 × rolling-window median`, *not* a universal MW floor — a 50 MW record next to a 2.5 GW median is an EIA artifact, but 50 MW is legitimate load for a tiny BA). Spot-check through the real `models.drift` path: **LDWP 266.9% → 13.1% sMAPE / 14.0% filtered MAPE (9 artifacts excluded); FPL/MISO/SPP/NYISO/ISONE unchanged (0 excluded, MAPE identical).** Overview headline prefers sMAPE w/ MAPE fallback; Models panel keeps (now-filtered) MAPE for the holdout comparison. Live Memorystore is VPC-internal so the spot-check is representative data through real code; live confirmation lands on the next post-deploy scoring tick. (3) **#150 re-close-trap:** the 2026-05-29 bookkeeping-correction commit `bd95c42` *re-closed* #150 four minutes after the manual reopen — GitHub scans **commit messages/PR bodies** for close-keywords and **ignores backticks/context**, so quoting `` `Closes #150` `` to *describe* the bug fired it again. Fixed via a pure `gh issue reopen 150` (an API action no commit can undo) and hardened the CLAUDE.md rule: flip state via the gh CLI, and never put a live close-keyword adjacent to an issue you don't mean to close.
- **2026-05-29** **Bookkeeping correction — issue-number mismatch caught in review, fixed.** PR #165 (PR-G10 alerting) was written with `Closes #150`, but alerting is **#148**; #150 is Prophet interval honesty (NOT done). The bad `Closes` wrongly closed #150 and left #148 open. Also a systematic off-by-one had crept into STATUS's Phase 3/4 issue lists. Corrected against `gh issue view` ground truth: reopened #150, closed #148 (credited #165), fixed all `#150`→`#148` doc refs (`docs/monitoring/`, `SCHEDULED_JOBS.md`), and rewrote STATUS Phase 3 (#149/#150/#155) + Phase 4 (#151/#152/#153/#154). **Root cause:** `Closes #N` written from memory, not verified. **Prevention:** always `gh issue view <n>` to confirm title before writing `Closes #N` — adding this to the CLAUDE.md end-of-PR ritual. Exactly the failure the project-state system exists to prevent; caught + closed before it propagated into the next pass.
- **2026-05-29** **`prod-readiness` Phase 1 + Phase 2 (G2/G3) shipped — and the deep /health check caught a P0 forecast outage on its first prod run.** Phase 1 complete: PR-G1 app-imports smoke test (#156), PR-G7 metadata/docs sync (#157), PR-G8 `feature_enabled` fail-closed (#158). Phase 2: PR-G2 gate deploys behind CI via `workflow_run` (#159, **prod-verified** — deploy fires gated, WIF auth held through the trigger change), PR-G3 deep `/health` + meaningful post-deploy smoke (#160). **PR-G3 immediately earned its keep:** on first prod run it flagged `forecast_sample: degraded` → investigation found **all 51 regions producing zero forecasts** (filed P0 [#161](https://github.com/kristenmartino/gridpulse/issues/161)). Root cause: Open-Meteo `/forecast?past_days=92` degraded its historical coverage; `soil_temperature_0cm` arrived 103/2177 rows, and `engineer_features`' `dropna(subset=<all features>)` let one sparse column collapse every region below the 168-row model threshold. Mitigation (A) — impute exogenous weather, drop only on autoregressive warm-up — shipped (#162), CI-gated-deployed, manual scoring run triggered, **service restored + verified** (`/health?deep=1` → `forecast_sample: ok, rows: 720, status: healthy`). Option (C) (archive endpoint for real historical weather) fully designed in #161, queued as next focused pass (Option 1: stitch; the 3 archive-missing vars stay imputed). The `curl / → 200` check PR-G3 replaced would have shown the entire outage as healthy.
- **2026-05-22** **External code review → 13-issue `prod-readiness` campaign filed.** Second senior-staff review (engineering-rigor focused, complementing the SaaS-gap review from 2026-05-21) surfaced 11 real findings + 1 false positive — reviewer claimed `register_us_grid_callbacks` was called but not imported in `components/callbacks.py`; the import IS on line 1005, reviewer missed it. Verified before acting. Real findings: deploy not gated by CI, shallow `/health`, `requirements.txt` vs `.lock` drift, simulated fallback paths still in `models/model_service.py`, Prophet 95% interval is heuristic, stale docs (pyproject name, env example branding, Dockerfile/TEST_PYRAMID region counts), `feature_enabled()` defaults True for unknown flags, mypy installed in CI but not run, LDWP drift outlier (already filed as #142). Filed [#143](https://github.com/kristenmartino/gridpulse/issues/143)-[#155](https://github.com/kristenmartino/gridpulse/issues/155) (label `prod-readiness`), added to Roadmap project. Plan: Phase 1 (~3h quick credibility wins) → Phase 2 (~7h deploy + observability) → Phase 3 (~7h production safety) → Phase 4 (~1 day engineering rigor). **PR-G1 (#143) in flight** — app-imports smoke test that would've caught the reviewer's claimed bug if it had been real.
- **2026-05-21** **Scheduler retry fix + manual training cycle validated audit fixes in production.** 2026-05-21 04:00 UTC scheduled training silently failed (Cloud Run regional API 503, no retry). Diagnosed as transient infra blip, applied retry policy to both training-daily (3 retries) and scoring-hourly (1 retry) schedulers, documented in `docs/SCHEDULED_JOBS.md`, closed [#141](https://github.com/kristenmartino/gridpulse/issues/141). Manually triggered training cycle (`gridpulse-training-job-fkzsp`) — first run with all six audit-fix PRs deployed. Completed cleanly in 1h47m, three parallel tasks all succeeded. Hourly scoring picked up the new pickles. Spot-check confirmed `demand_roll_24h_min` no longer in top-5 XGBoost features (PR-D's leakage fix working). Filed [#142](https://github.com/kristenmartino/gridpulse/issues/142) for LDWP drift outlier (sustained ~200% rolling MAPE, robust-statistics issue).
- **2026-05-20** **Forecast pipeline audit closed — six PRs merged in one day.** User raised "those MAPE #s look too clean" → senior-staff audit found one real bug (training-time target leakage in `ramp_rate` and `demand_roll_*` features) and two architectural mismatches (train/serve climatology gap, mismatched confidence-band calibration across surfaces). Six PRs shipped:
  - [#134](https://github.com/kristenmartino/gridpulse/pull/134) PR-A — Overview honest signals (timestamp-based trend, live drift MAPE, label clarifications). 12 tests.
  - [#135](https://github.com/kristenmartino/gridpulse/pull/135) PR-D — De-leak training features (`shift(1)` before rolling/diff). 5 tests + empirical demo.
  - [#136](https://github.com/kristenmartino/gridpulse/pull/136) PR-C — Real Open-Meteo forecast in `_build_future_feature_frame` (16 days). 9 tests.
  - [#137](https://github.com/kristenmartino/gridpulse/pull/137) ADR-008 — Climatology fallback past day 16 + UI labeling (dotted divider on Forecast tab). 5 tests + full ADR in PRD.md §10.
  - [#138](https://github.com/kristenmartino/gridpulse/pull/138) PR-E — Recursive autoregressive features in production (cap aligned with weather boundary at hour 384). 5 tests + empirical validation script.
  - [#139](https://github.com/kristenmartino/gridpulse/pull/139) PR-B — Empirical CI on Overview hero chart (shared method with Forecast tab). 3 tests.

  ADR-008 logged in PRD.md §10; full alternatives considered (shorten horizon, ECMWF S2S, light conditional climatology, heavy teleconnection-based) and why we chose climatology + visible labeling. Cumulative: **1,717 unit tests passing** (39 new), **all 6 Deploy → Production runs succeeded** (web service + scoring job + training job redeployed each merge per `.github/workflows/deploy-prod.yml`). Methodology re-validated on fresh data 2026-05-21 morning — FPL holdout MAPE consistent across runs, `demand_roll_24h_min` no longer in top-5 features. **Watching live drift MAPE for ~7 days to confirm production effect.**
- **2026-05-20** [#131](https://github.com/kristenmartino/gridpulse/issues/131) closed — scoring job now writes per-model + ensemble holdout metrics into the `gridpulse:forecast:{region}:1h` payload as `model_metrics`. `get_model_metrics` reads them as Layer 0 (the production path; existing layers 1-6 remain as fallbacks). Eliminates the "MAPE 1.6%" simulated-baseline values the Overview model card had been showing in production. 18 new tests. Full suite: 1,670 pass. [This PR]
- **2026-05-20** [PR #130](https://github.com/kristenmartino/gridpulse/pull/130) — Overview hero chart + insight + `is_trained` all route to Redis instead of `_simulate_forecasts` / local-disk checks. User-reported "looks off" surfaced two related bugs (chart rendered noisy historical as forward forecast; `[simulated]` badge always shown). CLAUDE.md "Web tier I/O guardrail" added documenting the architectural rule. Filed [#129](https://github.com/kristenmartino/gridpulse/issues/129) for the Forecast-tab gap (separate code path). 20 new tests. Full suite: 1,660 pass.
- **2026-05-20** PR-D2 — [#121](https://github.com/kristenmartino/gridpulse/issues/121) part 2 shipped. Models tab drift panel: `_build_drift_panel` reads `gridpulse:drift:{region}` + holdout MAPEs, renders per-model status chips (on track / drifting / degraded) with mixed-state support. 15 new tests. Full suite: 1,640 pass. [PR #128]
- **2026-05-20** PR-D1 — [#121](https://github.com/kristenmartino/gridpulse/issues/121) part 1 shipped. `models/drift.py` (continuous 1-hour-ahead drift measurement) + `jobs/phases.write_drift_metrics` (hourly Redis writes to `gridpulse:drift:{region}`) + 36 new unit tests. Full suite: 1,625 pass. [PR #126]
- **2026-05-20** PR-C1 — Recall artifacts shipped. Real `HOW_IT_WORKS.md` + 5 Mermaid diagrams + populated `CANONICAL_FACTS.md` + `INTERVIEW_PREP.md` STAR-story content. [PR #125]
- **2026-05-20** Wider replan after multi-perspective review: confirmed Position A, deferred Path B beyond #121, reordered PR sequence to C → B (conditional) → D (deferred), and split PR-C into C1 (recall) + C2 (communication). [PR #123]
- **2026-05-19** Path A declared complete. [#120](https://github.com/kristenmartino/gridpulse/pull/120)
- **2026-05-19** Scenario simulator: heuristic over full-fidelity engine. [#119](https://github.com/kristenmartino/gridpulse/pull/119)
- **2026-05-19** Project-state lives in GitHub, not Markdown. [#123](https://github.com/kristenmartino/gridpulse/pull/123)
- **2026-05-18** Big-bang Redis namespace flip over phased migration. [#114](https://github.com/kristenmartino/gridpulse/pull/114)
