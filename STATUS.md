<!--
How this file gets maintained:
- Per-PR: updated in the same commit as material work that changes
  active focus, next-3, blocked-on, or recent decisions
- End-of-session: agent re-verifies against gh issue list / gh pr list
- Pre-external-use: user re-reads top-to-bottom (~1 min)
If this file disagrees with gh, the live sources win — patch in a
follow-up commit.
-->

# Status — updated 2026-08-21

> Canonical pointer for "where am I, what's next." This file +
> [GitHub Projects board](https://github.com/users/kristenmartino/projects/1)
> + the issue tracker are the single source of truth for project state.
> See [`docs/internal/NEXT_UP.md`](docs/internal/NEXT_UP.md) for the full
> historical roadmap; see [`CLAUDE.md`](CLAUDE.md) for the pre-session
> sanity-check ritual.

## Active focus + open question

**2026-08-21 — [#559](https://github.com/kristenmartino/gridpulse/issues/559)
the seed shadow had a blind spot correlated with what it observes. Fixed by
making the absence typed, not by inventing an observation.**

Found by tracing one region (IID) absent from the 00:00 tick. It had not failed
— it logged `forecast_origin_regressed`, and #558's monotonic-origin guard
returns from `predict_and_write_forecast` ~300 lines above the shadow call, so
the shadow never ran. No error, no warning, just a missing region.

**That is missing-not-at-random.** The guard fires when EIA withdraws published
hours, which is gap-adjacent — so the absences correlate with the exact
condition the shadow exists to observe. LGEE regressed for **24 consecutive
ticks** during #537; that entire episode would have been an unexplained hole.

**The fix records the skip rather than computing an arm.** On a regressed tick
production does not forecast at all, so there is no served prediction to compare
against; manufacturing one would put "what production would have done" in the
same sample as "what production did". `seed_shadow_skipped` carries the region,
the reason and the regression size, so coverage becomes measurable.
`scripts/seed_shadow_eval.py` now says so in its output rather than reporting
counts as if the denominator were complete.

**Why now rather than after the sample completed:** the "don't change an
instrument mid-collection" rule guards against changes motivated by *results*.
This one was found by tracing a missing region and would have been made
identically whatever the divergence numbers said. The sample was two ticks old,
and every further tick collected under a biased instrument is a tick that has to
be caveated later.

**2026-08-21 — [#600](https://github.com/kristenmartino/gridpulse/issues/600)
`/api/v1/benchmark` could serve a fleet aggregate computed from rows it was not
shipping. Fixed by an atomic export, and verified in production — after the
first verification passed for the wrong reason.**

Per-BA rows and the fleet rollup were separate Redis keys written at different
points in the same tick, and `build_benchmark_payload` assembled the response
from them at request time. A request landing mid-write paired one tick's
aggregate with another's rows, and `updated_at` came from the fleet meta, so
**nothing in the payload could distinguish the two states**. Live capture:
45 of 51 rows carrying `scored_at` from the 01:0x tick under an `updated_at`
of `00:09:56`.

**Shipped** ([#637](https://github.com/kristenmartino/gridpulse/pull/637)): the
scoring job writes `gridpulse:meta:benchmark_export` — the rollup and the rows
it was computed from, as one value — and the API reads that one key. Mixing is
impossible by construction rather than unlikely. The request path drops from
**52 Redis reads to 1**; the document is 130,889 B and costs 0.71 ms to
serialise, built from the list the rollup already holds, so **zero extra
reads**. Rows are stored unfiltered and the public allow-list stays on the read
path, so the new key never becomes a second trust boundary. Absent or malformed
document falls back to the old assembly; `updated_at` now stamps the export
document itself.

### The verification passed before it should have, and that is the entry

The first live check after deploy reported **CONSISTENT** — and it was
meaningless. The Cloud Run *service* picked up the image at 01:38 while the
*scoring job* stayed on the previous one until after its 01:09 tick, so for
about an hour production ran a **new reader against an old writer**. No export
document existed; the API was on its fail-open path and the check was measuring
the old code.

What settled it was not the recompute assertion but
**`benchmark_export_absent_split_read`** — the fallback's own log line.
Zero occurrences after the 02:09:41 write (51 regions, no failures) is what
makes the passing recompute mean anything. Verified end to end at 02:10Z:
fleet recomputes from rows at delta `0.000000` on all four statistics, and two
fetches 35 s apart held `updated_at` at `02:09:41.306320` with **0 of 51**
per-BA `scored_at` values differing — the #600 symptom, absent.

**A second false reading, corrected earlier in the same chain.** The checker
first reported four mismatches including a 26 % relative error, and two of them
were not this bug: `fleet_rollup` isolates ERCOT by design
(`isolate=("ERCOT",)`), and ERCOT is both the min official and min gridpulse
MAPE. The real deltas were the two maximums, 0.016 and 0.036 — roughly what
#600 estimated, not larger. The payload had declared its own population twice
(`isolated`, and `fleet.n=44` against 45 scoreable rows) and the checker had
consulted neither; it now asserts `fleet.n` matches the population it compares.

**The transferable part:** an intermittent race passes its own check most of
the time, so "the assertion is green" is not evidence a fix landed. The
evidence was the fallback going silent. A deploy that rolls a service and its
jobs at different times will always produce a window where the new reader meets
the old writer — and that window looks exactly like success.

---

**2026-08-20 — where the #537 / #559 batch landed, and the one decision it
did not make.** *Roundup. The entries below are the record; this is the state.*

Seven changes shipped and deployed together at `7ef1610e` (service, scoring job
and training job all on it, checked by ancestry):

| | what | result |
|---|---|---|
| [#621](https://github.com/kristenmartino/gridpulse/pull/621) | counters for the two silent drift-buffer loss channels | published on `/api/v1/drift/{BA}` |
| [#625](https://github.com/kristenmartino/gridpulse/pull/625) | the fleet-wide channel split | `docs/DRIFT_COVERAGE_CHANNELS.md` |
| [#620](https://github.com/kristenmartino/gridpulse/pull/620) + [#630](https://github.com/kristenmartino/gridpulse/pull/630) | the #559 re-run, and an independent replication of it | not confirmed; both strata inconclusive |
| [#627](https://github.com/kristenmartino/gridpulse/pull/627) | the origin stall fix (#559 candidate 1) | merged, **inert** — see below |
| [#631](https://github.com/kristenmartino/gridpulse/pull/631) | the shadow gate names *why* the arms differ; `set()` stops dropping writes silently | live |
| [#633](https://github.com/kristenmartino/gridpulse/pull/633) | `/benchmark` publishes the drift window's real coverage | live |
| [#629](https://github.com/kristenmartino/gridpulse/pull/629) | `temporal_ar_seed_shadow` on | recording |

**The headline number is now published rather than implied.** `/benchmark` said
a flagged row was scored "over the trailing 7 days"; it is scored on 94 of 168
hours for LGEE, 102 for JEA, 165 at the fleet ceiling. Every scored row now
carries its own count, and the page states that **168 is not reachable** —
81 % of the gap is a skipped origin, which is EIA's publication lag beating
against an hourly tick clock, not a backlog. [#628](https://github.com/kristenmartino/gridpulse/issues/628)
closed on that.

**Two framings were refuted by their own measurements, and both are recorded
rather than quietly dropped:** the unresolved-actual channel (proposed as JEA's
explanation) is 1.5 % of the shortfall with zero expiries; and the absent-hour
policy fix moved stratum A's sign from −0.265 to +0.27 without making it
decisive.

### The one decision this batch did not make — now [#635](https://github.com/kristenmartino/gridpulse/issues/635)

*Filed 2026-08-20. It had been tracked only in this file, so the `gh issue list`
half of CLAUDE.md's session-start check could not see it while #627 sat merged
and inert waiting on it. That is the failure mode this file's scope note now
exists to prevent.*

**`temporal_ar_seed` is still off, so [#627](https://github.com/kristenmartino/gridpulse/pull/627)
is in place and inert.** Channel B — the 16.9 % of the shortfall that is
actually ours to fix — is not being recovered. The flag is the gate because
under the positional seed `demand_lag_1h` means "the last surviving entry", so
no positional advance is provably safe; advancing without the temporal path
would trade a visible stall for invisible wrong values.

What is new is that the evidence for that decision is finally clean. The shadow
is live and every divergent observation so far lands in the `hole_in_lookback`
stratum — the one that is genuinely about temporal indexing — with
`seed_tail_gap_h` at 0 throughout. That is structural, not luck: with the flag
off the origin cap holds the gap at 0, and with it on #627's bridge does.

**Also open:** [#624](https://github.com/kristenmartino/gridpulse/issues/624) —
`HourIndexedHistory` sizes its array from the last *present* seed hour. Its
silent half is fixed (a dropped write is now counted and logged); the sizing
fix is a signature change and is not made. Measured live, it **cannot fire in
the current regime** — both paths that could open the trailing gap close it —
so it is latent correctness, not an active defect.

---

**2026-08-20 — [#537](https://github.com/kristenmartino/gridpulse/issues/537)
the fleet-wide drift shortfall is measured and split. The hypothesis it was
measured to test is refuted, and JEA is not the case anyone thought.**
*Measurement only — no code changed. Does not displace the #559 focus below.*

The suspected second loss channel — snapshots entering the pending buffer and
`_expire_pending` dropping them at 120 h because the actual never published — is
**1.5 % of the shortfall, and zero of it expired**. All 530 absent hours have a
settled actual in the vintage mirror; the expiry channel is bounded above at
**4 hours across all 51 BAs and the whole 168 h window**.

What is real is a third mechanism neither the issue nor the hypothesis named.
The origin **skips**: a 24h snapshot for target `T` needs some tick to see hour
`T−25h` as its newest, and when EIA publishes two hours in one tick no tick ever
does. The origin jumps, the target is never proposed, and there is no
re-proposal path. Tested against its own control — **82.7 %** of absent hours
show the same-tick signature against **0.16 %** of resolved ones.

| channel (ensemble 24h, 6069 h) | hours | share |
|---|---:|---:|
| **A** origin skip (never proposed) | **436** | 81.0 % |
| **B** origin freeze, `last_featured_ts` (#559) | **91** | 16.9 % |
| **C** unresolved actual — the hypothesis | **8** | 1.5 % |

**JEA is channel A, 64 of 65.** Its feed went dark and back-filled 24 hours at a
time (all of 2026-08-15 first seen at one instant, 08-17T16:04Z); the absent run
ends exactly 24 h after the feed recovered, the horizon's own offset. The six
hours that arrived **39–45 h late resolved fine** — the buffer tolerates a late
actual, so nothing was lost at resolution. `168 − 102 = 66` was arithmetic
across two buffers and does not survive either.

**Pre-deploy prediction for the origin-cap fix, on the record:** it reaches
channel B only — **at most 91 records, 91.14 % → 92.64 %**, on six BAs
(LGEE 21, SPA 18, PSCO 16, LDWP/IID/AZPS 10 each). JEA moves ≤ 1. If the median
BA improves, it is measuring something else.

**Filed as [#628](https://github.com/kristenmartino/gridpulse/issues/628), and
being paid:** channel A is not fixable — the orphaned hour's only forecast is
at a 23 h lead, and filing it in a 24h window is the P2-19 mislabelling. So
`n_7d = 168` is unreachable by design and `/benchmark` owed a standing coverage
disclosure. It now publishes the per-row count beside the trailing-7-day claim,
derived from `serve_grade.n_7d` with no literal on the page, and states the
ceiling as structural rather than as a backlog. The per-channel counters
(`n_dedup_skipped_7d` and siblings, #621) are deliberately **not** on the
public row: measured live on 2026-08-20 they sum to 1 of LGEE's 74 missing
hours and 0 of JEA's 66, so publishing them would present three near-zero
numbers against a large gap the measurement does explain. They stay on
`/api/v1/drift/{BA}`. **Evidence:**
[`docs/DRIFT_COVERAGE_CHANNELS.md`](docs/DRIFT_COVERAGE_CHANNELS.md).
[branch `fix/628-benchmark-drift-coverage`]

**Also refuted while checking it:** this file's 2026-07-16 claim that ten
regions never revise. Over the 30 days the vintage mirror now holds, all ten do
— PNM 27.5 %, PGE 33.7 %, JEA 16.1 %. That is `n_updates` on **`D`**, the actual
demand; it is **not** the day-ahead-`DF` revision rate in
`BENCHMARK_METHODOLOGY.md` §6, which is a different series and is unaffected.
The line below is left as dated history; do not build on it.

---

**2026-08-20 — [#559](https://github.com/kristenmartino/gridpulse/issues/559)
candidate 1: the origin stall is fixed, gated on `temporal_ar_seed`.**

`_resolve_forecast_start` capped the anchor at `min(last_real_demand,
last_featured_ts)`. The second term asks whether the origin's predecessor row
survived `dropna(subset=autoregressive)` — feature-frame bookkeeping — where it
means to ask whether we hold hourly demand for it. One null hour deletes the
rows 1/2/3/24/168 hours later, so the tail of `featured` ends behind demand
that arrived and is real. The anchor now advances across the contiguous run of
real demand hours after that tail, and the recursion is handed those hours.

**Gated on `temporal_ar_seed`, and the gate is the finding.** The positional
seed reads `demand_lag_1h` as "the last surviving entry", so an advanced origin
indexes it to the wrong hour. `positional_seed_matches_hours` requires the
seed's last entry to *be* `origin - 1h`, so it is false by construction for any
advanced origin — **no positional advance is ever provably safe**, which is why
this ships inert rather than live. Without the bridged seed the hour-indexed arm
is not safe either: measured on the 16h fixture, `demand_lag_1h` off by 691 MW,
`demand_lag_3h` by 662, because the hole is too long to interpolate and the
fallback steps back 24h into the hole. Origin and seed are therefore resolved
together and the origin clamps back if a reaching seed cannot be built.

**Cost, measured before merge (#389):** +1.61 ms per stalled BA, −0.01 ms per
unstalled one → **+14 ms per fleet tick** at the observed 9 stalls, +82 ms if
every BA stalled. Once per BA, not inside the 384-step recursion; the recursion
itself is unchanged (20.7 vs 21.2 ms/BA). Six mutations, six kills.

**The channel split landed ([#625](https://github.com/kristenmartino/gridpulse/pull/625),
`docs/DRIFT_COVERAGE_CHANNELS.md`) and this fix is the small channel.** Three
channels, not two: **A** origin *skip* 436 h (81.0%, unreachable — no
re-proposal path exists), **B** origin *freeze* **91 h (16.9%, this PR)**, **C**
unresolved actual 8 h (1.5%). So the prediction is concrete, not conditional:
**≤ 91 records recovered fleet-wide — 5,531 → 5,622 of 6,069, coverage 91.14% → 92.63%**, JEA moves ≤ 1
hour, **41 BAs do not move at all** (channel B is ten BAs: LGEE 21, SPA 18,
PSCO 16, LDWP/IID/AZPS 10 each, TIDC 3, PACE/SC/JEA 1). **If the median BA
improves, that is falsification, not a win.** No target implies `n_7d = 168` —
it is unreachable by construction, ceiling 165.

**Still held as a draft**: #625 is itself open, so the baseline the prediction
is tested against is not on `main` yet.

**And it is inert for as long as `temporal_ar_seed` is off — which the re-run
below leaves off, with its stopping rule spent.** That is the honest cost of
refusing the unsound version, and it is stated rather than worked around: the
stall stays live until the flag flips, because the alternative is an origin
whose near lags read hours nobody named. If the flag is never flipped, the
open question this raises is whether the origin should instead be advanced by
giving the *positional* recursion the bridge hours too — sound for
`demand_lag_1h/2h/3h` and `ramp_rate`, and provably nothing for `lag_24h` /
`lag_168h` / the rolling windows, which are already misindexed on exactly these
BAs. That is a **different, measurable** question, not this PR.

the losing quarter is DIFFUSE. Nothing to carve out, and both pre-specified
hypotheses were backwards. This line of inquiry is closed.**

Pre-registered as **exploratory** before the cut existed — it re-cuts data that
already produced a verdict, so nothing in it could turn the flag on. **20 cells
examined, 0 flagged** against a bar of n ≥ 30, win rate < 50%, and a bootstrap
interval excluding zero. ~1 false flag was expected by chance; none appeared.

**H1 (long gaps hurt): wrong, and inverted.** On stratum B a 13–24h gap yields
**+1.02** mean Δ against **+0.51** for a 1h gap — twice the benefit.

**H2 (recent gaps hurt): wrong, and the clearest structure in the data runs the
other way.** Stratum B falls monotonically as the gap moves away from the origin
— **+1.143 / win 0.808** at 1–24h lead, down to **+0.488 / 0.702** at 73–168h.
The fix helps most exactly where the defect bites hardest. Stratum A's 1–24h cell
is the only sub-50% win rate anywhere (0.444) and has **n = 27**, below the
pre-registered floor, with an interval spanning zero — and it is the one place
the strata disagree in direction.

**H3 (hour of day): flat.** B 0.683–0.783, A 0.542–0.649, no ordering.

**PSCO does not rescue it.** Post-hoc, PSCO is genuinely anomalous
(−0.462, win 0.361) — but stratum A **excluding PSCO** is still only **0.634**
win rate against the 0.75 the policy requires. Removing the worst BA entirely
does not ship. The consistency failure is not one bad BA, which closes the
obvious escape route.

**No confirmatory follow-up is warranted** — §5 reserved that for a coherent
pattern in the *losing* direction, and the one coherent pattern found describes
where the fix already works. **Evidence:**
[`docs/POSITIONAL_LAG_LOSING_QUARTER.md`](docs/POSITIONAL_LAG_LOSING_QUARTER.md).

---

**2026-08-20 — [#559](https://github.com/kristenmartino/gridpulse/issues/559)
the seed shadow is being turned ON, now that its arm is worth observing.**

It shipped dark because, at the time, an absent lag hour returned NaN and the
shared `row.fillna(0)` made it `demand_lag_24h = 0 MW` on **13% of forecast
steps** — so anything it recorded would have been evidence about that bug, not
about temporal indexing. With the absent-hour policy decided the rate is **0
across 32,832 scored steps**, and the arm is finally the thing we mean to watch.

`temporal_ar_seed` stays **off** — this changes what is *recorded*, never what is
served, and a test pins the served payload byte-identical either way.

**Two things the flip itself surfaced**, neither of which a unit-test-only run
would have shown:
- a test asserted `redis_set.call_count == 1`, which is a fact about which
  enrichment flags are on rather than about the model metrics it was testing.
  It now selects the forecast payload by key.
- with the flag on, 44-of-51 never-gapping BAs would have re-persisted an empty
  payload **every hour**. The write is now skipped when nothing was computed and
  nothing graded, and the `seed_shadow_written` log — which fires on every
  invocation — carries "ran, nothing to do" instead. That is strictly stronger
  than a key that only sometimes exists.

**Watch after deploy:** `seed_shadow_written` (`n_records` must climb) and
`seed_shadow_audit_diverged` (must stay absent — it is an alarm about the gate,
not a finding about the seed).

---

**2026-08-20 — [#559](https://github.com/kristenmartino/gridpulse/issues/559)
re-run: the policy fix worked, stratum A's sign flipped, and the hypothesis is
STILL not confirmed — now for a different reason.**

| stratum | n | mean Δ WAPE | median | MDE | consistency | verdict |
|---|---:|---:|---:|---:|---:|---|
| **A** naturally gapped | 249 | **+0.268** | +0.132 | 0.172 | **0.594** | not decisive |
| **B** never gapped | 432 | **+0.614** | +0.309 | 0.107 | **0.729** | not decisive |

Both positive, mean and median agreeing, magnitude clearing MDE — B by **5.7×**.
Both fail on **one** rule: sign consistency, and B misses 75% by **2.1 points**.
`verdict()`: *"real on average but not reliable enough to ship."* That is the
case the rule exists to catch, and it is not a threshold to argue down after
seeing the number.

**The diagnosis is confirmed.** Runs are paired and the **control arm is
byte-identical across them** (`0.0000000000`), so only the treatment moved:
A **−0.265 → +0.268** (swing +0.533), B **+0.187 → +0.614** (+0.427). The worst
first-run regressions are gone — IID **−1.47 → −0.01**, NEVP −0.51 → −0.005.
PSCO is the one BA still consistently worse (−0.46), and it is the BA whose real
gaps are clock-aligned rather than random.

**Criterion 4 met**: 0 NaN lags and 0 non-positive lags across **32,832 scored
steps**, against 13.08–22.57% before. Criterion 2 (null control) exact.

**Pre-registration gap, recorded rather than papered over:** §6's reading for
"both inconclusive" assumed inconclusive would mean *small*. It does not here —
the effect is large and fails on consistency. Applying that pre-committed reading
anyway would have been dishonest, so it is marked wrong instead.

**Flag stays off. The stopping rule is spent** — no third attempt on this
question. **The live question is now different and better:** *where* does the
losing quarter of windows sit? Gap length, gap hour and gap lead are recorded per
window so it can be asked without re-running — as a **new** pre-registration.
**Evidence:**
[`docs/POSITIONAL_LAG_INJECTION_RERUN_STUDY.md`](docs/POSITIONAL_LAG_INJECTION_RERUN_STUDY.md).

---

**2026-08-20 — [#559](https://github.com/kristenmartino/gridpulse/issues/559)
the absent-hour policy is decided rather than inherited, and the re-run is
pre-registered.**

The injection study did not confirm its hypothesis, and the diagnosis was a
defect in the treatment arm: an absent lag hour returned NaN and the shared
`row.fillna(0)` turned it into `demand_lag_24h = 0 MW` — the #129 poison — on
**13% of forecast steps, 22.6% on IID**. So that run measured
temporal-indexing-*plus-zero-fill*, not the hypothesis.

**Policy now chosen, not inherited** (`HourIndexedHistory.lag`): a hole of
**≤ 6 hours** is linearly interpolated across — the bound is
`MAX_INTERPOLATION_GAP_HOURS`, reused from `data.preprocessing` rather than
invented, and 25 of 31 measured gap runs are a single hour — and a longer hole
falls back to the **same clock hour on previous days**, up to 7, because
interpolating across a 16-hour hole would smooth over a diurnal cycle.

**Measured, before any accuracy claim:** NaN-lag rate on scored windows is
**0.00%** across nine BAs, against 13.08–22.57% before. Nothing is zero-filled.
The earlier 8.33% residual was an artifact of the probe, not the policy — four
early windows with too little history, which the study's own guard never scores.

**The parity invariant changed, and the tests say so.** Parity is only *defined*
where training kept the row: a NaN training lag means the row was dropped and the
model never saw it, while serve cannot skip a step and must impute. The fuzz now
skips those rows — and asserts it still compared ≥50 lags, so it cannot pass by
skipping everything, which is the failure this file was written about.

**Re-run pre-registered** as a NEW registration, not an amendment — the previous
stopping rule was one run and is spent. Same seed, same holes, same strata, so
the runs are paired and only the policy moved. All four outcomes have their
readings fixed in advance, including "A still negative", which would mean the
diagnosis was wrong and the flag should be considered for removal rather than
merely left off. **Evidence:**
[`docs/POSITIONAL_LAG_INJECTION_RERUN_PREREGISTRATION.md`](docs/POSITIONAL_LAG_INJECTION_RERUN_PREREGISTRATION.md).

**Flag stays off.** Stratum A remains structurally underpowered for a 0.18-pt
effect; that is not fixed by this change.

---

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

**Entries through 2026-07-31 are archived** in
[`docs/internal/JOURNAL_ARCHIVE_2026H1.md`](docs/internal/JOURNAL_ARCHIVE_2026H1.md).
They are the record, not current state, and are not edited. Split out on
2026-08-20 because this file had reached ~95k tokens against an active focus
of ~1,750 — and the `cat STATUS.md` in CLAUDE.md's session ritual was being
truncated rather than read.

## Next 3 — moved to GitHub

**This section is gone deliberately.** It duplicated `gh issue list` and went
stale within two weeks: on 2026-08-20 it still presented
[#273](https://github.com/kristenmartino/gridpulse/issues/273) and
[#171](https://github.com/kristenmartino/gridpulse/issues/171) as live slots,
both long closed. CLAUDE.md already declares GitHub the winner of any conflict
with this file, so a hand-maintained copy here carried no authority — only the
ability to mislead, which is what it did.

```bash
gh issue list --state open            # the queue
gh pr list --state open               # in flight
```

Priority lives on the [project board](https://github.com/users/kristenmartino/projects/1)
and in issue labels. **This file is a decision log, not a queue** — see the note
under "Blocked / waiting on".

## Blocked / waiting on — moved to GitHub

**Also gone deliberately.** It carried
[#129](https://github.com/kristenmartino/gridpulse/issues/129) as pending work
for three months after it closed — and the fix it described had since shipped as
[#627](https://github.com/kristenmartino/gridpulse/pull/627) under a different
number. Blocked state belongs where the blocking thing lives.

```bash
gh issue list --state open --label blocked
```

### What this file is for

The two sections above tried to hold **current state**, which GitHub already
holds and keeps correct. What has no home in GitHub is the **cross-issue
record** — *we measured X, it refuted Y, here is why we did not do Z*. That is
what the entries above are, and it is why they are dated and append-only.

The rule that follows: **an entry here describes what was true on its date and
is never edited to stay current.** The moment this file claims present state it
starts rotting, and it rotted in both directions at once — holding three closed
issues as live while the `temporal_ar_seed` decision
([#635](https://github.com/kristenmartino/gridpulse/issues/635)) existed only
here and was invisible to `gh`.

## Recent decisions (last 7 days)

**Entries before 2026-08-13 are archived** in [`docs/internal/JOURNAL_ARCHIVE.md`](docs/internal/JOURNAL_ARCHIVE.md). This section held **111 entries going back to 2026-05-18** under a "last 7 days" heading — the label had been wrong for three months.

- **2026-08-18** **CI's test suite was never hermetic — it made 79 live API calls per run, and that was a correctness bug wearing a performance costume.** Chasing an 8m45s PR wall-clock turned up 40% CPU utilization on a suite that should be compute-bound, and per-test cost that grew with position in the run (one test: 4.8s alone, 13.6s after 220 tests, 29.3s in the full run) — the signature of an accumulating rate limiter, not slow code. Tests were reaching `api.eia.gov` and `archive-api.open-meteo.com` for real: every client fetch path is cache-first, so a miss falls through to the live API and a test that believed it asserted on `mock_eia_response` asserted on today's grid. `test_scoring_job.py` claimed "All external I/O is faked" while its interchange endpoint went out on all 13 of its tests. **Two mocks were also silently inert:** `patch("data.redis_client.redis")` was defeated by a function-local `import redis` (real DNS lookup, 4.5s), and a `CACHE_DB_PATH` default bound at def time meant a thread-safety test ran 16 threads against the real repo-root `cache.db`. Sockets are now blocked autouse (not `requests` patched — a raw socket or urllib would route around that), with `tests/unit/test_network_guard.py` pinning that the guard is actually installed. **Suite 135.6s → 77.7s serial, 30.4s on 4 workers; CPU 40% → 78%; coverage unchanged at 91%.** CI's four jobs also ran serially for no reason, making the critical path their sum. **Measured on CI: 8m45s → ~3m40s** (pytest step 266–298s → ~180s; dependency setup 66s → 12s on a cached venv). **One optimisation was rejected on its number, not its idea:** Docker buildx `cache-to: type=gha,mode=max` took the image build 83s → 371s, because mode=max exports every intermediate layer of an image carrying prophet/xgboost/shap/scipy — reverted, with the measurement left in a `ci.yml` comment so it is not retried. [PR #586]
- **2026-08-18** **#542's fix verified on live production after deploy — it holds going forward, and 70% of the drift window is still blanked history that no fix can reach.** Independent re-check at **10:30Z**, after [#548](https://github.com/kristenmartino/gridpulse/pull/548) (`71a60cc`) landed: confirmed through to behaviour rather than stopping at "merged", since those are different claims — `71a60cc` is an **ancestor** of the deployed image `437411b` (equality would have failed; eight commits had landed on top), and **new records now retain their lead**: newest-24 known **IID 24/24, PJM 22/24**. Fleet unknown-lead **2,275/2,880 (79%) → 2,029/2,880 (70%)**. **The residual is the operationally important part and it is not a defect:** those leads are *unrecoverable* — the payload proving them was overwritten — so outside the six backfilled BAs they clear only by **ageing out of the 30-day window, ~3–4 more weeks**. Until then the P2-19 filter is still mostly bypassed, and **a 7d-vs-30d comparison spans a repaired population against a partly-blanked one**; anyone leaning on drift's lead-filtered headline in that period should know which half they are reading. **One BA is a different question, not an incomplete fix:** SEC sits at 5/24 with newest leads reading `[None, None, None, 1, 1, 1, 1, 1]` — `_lead_hours` returns `None` when it cannot derive a sane positive lead, which fits a broken feed, so those are **write-time unknowns rather than erasure** and are out of #542's scope. The fix itself is better than the one #542 proposed: `dataclasses.replace` carries **every** field, so the next field added to `DriftRecord` cannot be dropped the same way — the omission class, closed rather than the instance. [docs/542-postdeploy-verification]
- **2026-08-18** **#547: anchor provenance is now recorded at forecast time — the instrument #539 could not build, and it measures nothing yet.** #539 disclosed that our anchor can be seeded by the operator's own day-ahead forecast (MISO 36.58%, CAISO 26.56%, fleet median 3.34%) and had to state that limit's materiality as **unmeasured rather than small** — honest, and not a stable position, because it invites a reader to assume "small". Every forecast payload now carries an `anchor` block (hour, value, was-it-a-placeholder, was-it-conditioned) which rides onto both drift paths. The doc explicitly refuses to let "instrumented" read as "measured" — on the day it lands it has measured nothing. **I repeated #547's own impossibility claim and it is false**: the issue said the anchor cannot be recovered because `lead_hours` is the realized lead rather than the anchor, but row 0 *is* `anchor + 1h` by construction (`_build_future_feature_frame` starts at `forecast_start`) and `_lead_hours` counts from row 0 — so `anchor = target − lead_hours` is exact on the 1h path, and `anchor = target − H − 1h` on the horizon path needs no lead at all. `drift_updated` has logged `(region, target, lead)` at write time since #407, so even the #542 erasure does not block it. A bounded retrospective measurement is therefore **available today** over the vintage mirror's rolling ~30-day window, and the doc now says so rather than asserting unmeasurability. Forward recording is justified by reach — unbounded, not resting on a row-0 assumption a frame-builder change would silently break, and covering the two fields no reconstruction reaches (`anchor_conditioned`, whose fork is never persisted, and the anchor value *as seeded*, which diverges from vintage `first_seen_d` exactly when the quality guard or conditioning touched it). **Three deliberate calls.** (1) Scope stops at drift accrual: no `PairedHour`, no benchmark payload, no `api.py`, because publishing a split computed over zero records recreates the problem #539 fixed, and the published shape should be designed with data in hand. (2) A **fourth** field, `anchor_conditioned` — vintage records the *raw* `D`, so an ADR-009-substituted anchor would otherwise read "metered" while the seed genuinely was their day-ahead value; a true field whose framing asserts something false, the §5 defect one layer down. (3) `regrade_records` converted from a hand-listed rebuild to `dataclasses.replace` — ours was the **second** field class to meet that omission, and the third was a matter of time. Sizing checked rather than assumed: +54 B/record, 12.9 → **20.9 MB** fleet-wide, which is 0.8% of the 1 GB Memorystore ceiling at 13% use, so the ISO `anchor_ts` stays readable rather than being packed as an offset. Ties directly to [#558](https://github.com/kristenmartino/gridpulse/pull/558), merged hours earlier: that origin regression was EIA publishing 19 hours as placeholders (`D == DF`) and then withdrawing them — precisely the ticks on which this block now records `anchor_was_placeholder=True`, so the diagnosis it had to reconstruct becomes a field read. Direction: **moves our own number by exactly nothing**. [feat/547-anchor-provenance]
- **2026-08-18** **Per-horizon champion-chasing measured and rejected: loses to a fixed model at all three horizons, and the reason is near-tie margins, not flip rate.** Reused the preserved `drift_horizon` GCS buffer (§2 of the shared readout), 204 BA-weeks. Chasing the previous window's champion: **4.980/5.452/5.634** WAPE at 24/48/72h, against best fixed model **4.909 (xgboost) / 5.326 / 5.625 (ensemble)** — chasing loses at every horizon. Median champion-vs-runner-up margin is only **0.212/0.316/0.307 pts**, which is why: with 4 models the 90.2/98.0/94.1% flip rates sit **at** their 98.4% null (4-window comparison) and carry no signal, while the 1h figure (35.3%) is **far below** its 75% null (2-window) — champions there are *more* stable than chance, the opposite of how the raw number reads. **Reproduces an independent measurement**: the fleet-baseline sign flip (xgboost ahead 24h, ensemble 48h/72h) matches paired xgb-ens −0.0635 → +0.0554 on a different metric and different windows. Oracle headroom is real (~0.4 pts every horizon) and unreachable by chasing for the same near-tie reason. **Data-completeness note:** windows carry median 161-162/168 records; #537 explains this as episodic and BA-specific (frozen-origin re-derivation dropped by `seen`, LGEE 44/140 ticks), not uniform noise — whether missingness correlates with which window a BA falls in was **not checked**, so cross-window comparability is not asserted. **#542 confirmed clean**: 442,537 horizon records, 0 carry a lead. **Limitation stated explicitly:** only chasing, fleet-fixed, and oracle were tested — a per-BA fixed assignment was not, and ADR-004's existing winner-take-all rejection (`PRD.md:227`) is neither closed nor reopened by this. No follow-up proposed. Evidence: `docs/PER_HORIZON_SELECTION_STUDY.md`. [exp/478-bias-measurability]
- **2026-08-18** **#542 fixed: `regrade_records` erased `lead_hours`, and the counter built to make that observable was published and unread for weeks.** One-line omission beside a correct line — the constructor deliberately reset sMAPE to a recompute sentinel (right: sMAPE is *derived* from the value that moved) and dropped `lead_hours` (wrong: a property of the observation a revision cannot touch). Because `filter_by_lead` keeps unknown leads by design, every revision moved one more record past the P2-19 filter: **82.7% of retained records blanked**, sorted by revision rate not age (PACW 162/163 in-window vs PJM 3/150). **The fix repairs nothing already blanked**, so the before/after had to be reconstructed from outside the pipeline — `drift_updated` had logged `(region, target hour, lead)` every tick since #407, and 31 days of it rebuilt the erased map. Harness validated first: published `n_7d` reproduced on **204/204** blocks, 7-day lead recovery **100%**. **I predicted the direction and was wrong** — the headline moves *both ways*, 17 BAs better and 20 worse (AZPS 9.779 → **11.427**, PSCO 9.639 → **10.766**), fleet only −0.076 while LDWP moves **−3.854** and crosses the `n_7d ≥ 24` gate 25 → 15. Two unplanned findings: `reconcile.py` graded a different *population* than the panel it checks (1 → **12** false A1s once leads were repaired, **12 → 0** with the filter mirrored), and **#537 shares an upstream phenomenon with this and no code** — a frozen forecast origin produces growing leads on the 1h path *and* re-derived targets on the horizon path; LGEE froze 15 ticks then served a 23-hour-older vintage for 24 more (leads to **63h**, horizon 82/168), and `_expire_pending` is **ruled out**. `serve_grade` unmoved: 442,537 horizon records, **0** carry a lead. Evidence: `docs/DRIFT_LEAD_REGRADE.md`. [fix/542-drift-regrade-lead-hours]
- **2026-08-18** **#541 root-caused: the shadow window graded against actuals it never refreshed — and the arm that looked broken was the instrument.** The shadow path reused **one of drift's three record primitives**: it graded with `build_records_from_actuals`, and skipped both filters *and* `regrade_records`. `DriftRecord.actual` is EIA's **current** view, re-graded every tick as revisions land (#304); a shadow record froze `actual` at the tick that created it — the **preliminary** value, which drift's own docstring measures at **15–70% wrong for high-revision BAs**. So the shadow window was prediction-vs-preliminary *forever* while drift converged to prediction-vs-settled. **The row-level diff is the whole argument** (both paths, lead 1, same window, 2026-08-18T05:07Z): predictions **byte-identical on every BA** (`pred_differs=0`), actuals diverging on **123 of 139 hours for IID** and **107 of 144 for SEC** against **3 of 142 for PJM**. IID's frozen actual sat at **339 MW on every row** while EIA settled those hours at **545–867** — the entirety of its +86.49% against drift's **+2.8%**. The BAs that looked broken are exactly the high-revision ones. **That also killed the two hypotheses I would have spent the day on:** the `enumerate(rows)`/`shadow_preds[i]` alignment is fine (`redis_payload["forecasts"]` and `rows=` are the *same list object*, and `pred_differs=0` proves it end to end), and it is **not** ADR-009 anchor conditioning — drift grades the same hours at the same lead from the same payload and reads +2.8%. **The prior session's "a shorter lead cannot be 52× worse than a longer one" was the right instinct pointed at the wrong suspect:** it was never the forecast, it was the yardstick. Fixed by `models.shadow_eval.regrade_records`, mirroring drift's semantics — absent hours skipped rather than treated as agreement, materiality at 2dp, and **idempotent**, which is what makes the corrupt history **self-heal next tick with no backfill**. Measured against production (2026-08-18T05:13Z, **4,037 of 6,963 records re-graded**): control per-BA **+3.263% → +0.739%**, pooled **+2.403% → +0.585%**; treatment +0.707% / +0.547%; `satisficing {'passed': True}`. **Both control figures are inside ±2% for the first time, so the constraint is finally measurable** — and **SEC reads +8.92% against drift's independent +8.96%**, the convergence check that matters more than either number. **No BA excluded, no threshold touched.** Median span **6.54 days**, so #478's 14-day minimum still governs (~2026-08-24). Spun out [#542](https://github.com/kristenmartino/gridpulse/issues/542): drift's own `regrade_records` **erases `lead_hours`**, leaving **2,275 of 2,880 records (79%) unknown-lead** — IID 704/720, PJM 443/720, tracking revision rate exactly — so the P2-19 lead filter is progressively defeated and its docstring's self-healing claim is now false. **Three harnesses, one question, three different instrument failures** (#451's imputed replay weather, #541's stale actuals, #542's blanked leads); the ±2% bound was never the problem. [exp/478-bias-measurability]
- **2026-08-18** **#478 is not waiting on days — the harness built to settle it is writing corrupt records, and that is the second measurement failure in a row on the same question.** [#541](https://github.com/kristenmartino/gridpulse/issues/541). `scripts/shadow_weights_eval.py` stops on `STOP_served_arm_itself_breaches_bias_constraint` with the served arm at **per-BA +9.421% / pooled +3.656%** — but that number is not a forecasting fact. **The shadow path reused the *grading* primitive (`build_records_from_actuals`) and none of the *filtering*:** `compute_drift_payload` runs every record through `filter_by_lead` then `filter_low_actuals` before it averages anything, and the shadow path ran neither, so the two paths graded identically and filtered differently. Fixed by `models/shadow_eval.filter_records`, called **once per region on the shared `actual`** so both arms keep identical hours (a per-arm gate would turn a weighting comparison into a coverage comparison) and deliberately **not idempotent**, since `filter_low_actuals` thresholds on the median of what it is given. Measured before → after: **per-BA +9.421% → +3.264%, pooled +3.656% → +2.412%**, almost all of it **415 records whose known lead exceeded 1h** in a window whose entire name is "1-hour-ahead" (production carried leads to **63h**). `filter_low_actuals` dropped **2 records fleet-wide** — it is region-*relative*, so a BA whose artifact hours are a large share of a short window has them **set the median** and stop reading as outliers. **The gate does not clear the bound, and the residual is one BA:** IID **+86.49% over 126 clean lead-1 records**, against **+1.65%** from the drift path over the same window at a *longer* 24h lead, on a feed whose actuals (339–960 MW) and predictions (397–882 MW) are both sane. **A shorter lead cannot be 52× worse than a longer one** — LDWP is the same shape at ~135× (+327.8% vs −2.42%). So the residual is a defect in how the shadow stream is *written*, not a bias in what the models forecast. **The pattern worth keeping:** #478 exists because #451's *replay* could not measure bias (control over-forecast ~6% against partly-imputed ERA5 weather); the shadow pass was built to escape that, and is measurement-limited in its own different way. Two harnesses, same question, both defeated by their instrumentation — the ±2% bound was never the problem and **no threshold was touched**. Naming the residual BAs is diagnosis, **not** a proposed exclusion: dropping a BA whose number looks wrong after seeing it is the post-hoc criterion-invention `EVALUATION_POLICY.md` forbids and #451 already called out. The 14-day minimum is also unmet (6.5 days), so the answer was never due today — but "answer arrives ~2026-08-24" was wrong about *why* it had not arrived. Found while preserving the horizon-drift buffer (30d × 51 BAs behind a 24h TTL on a no-persistence instance) to `gs://nextera-portfolio-energy-cache/cache/adhoc/`, which is what made the independent cross-check possible. [exp/478-bias-measurability]
