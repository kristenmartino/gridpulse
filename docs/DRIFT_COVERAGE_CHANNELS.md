# Why the 24h drift window is short: the origin skips, it does not lose actuals

**Issue:** [#537](https://github.com/kristenmartino/gridpulse/issues/537), the
fleet-wide half. The LGEE freeze and the SPA retraction are settled in
[`docs/FORECAST_ORIGIN_REGRESSION.md`](FORECAST_ORIGIN_REGRESSION.md); this
document is about the other 49 BAs and about **JEA**, which appears in no table
on that issue.

**All figures measured against production, captured 2026-08-20T18:00Z** — one
Redis read of `gridpulse:drift_horizon:*` for all 51 BAs, joined to the GCS
vintage mirror. Drift numbers move hourly; every figure here is anchored to
that capture.

---

## 1. The question, and the hypothesis it started from

`/api/v1/drift/{BA}`, ensemble 24h, `n_7d` against a theoretical 168:

```
LGEE 94   JEA 102   SPA 113   TIDC 129   PSCO 136   |   median 161   |   TAL 165
```

**JEA at 102 is second-worst.** It is absent from #559's null-row census (which
is led by IID at 37 — and IID sits at a healthy 149), and the
`forecast_start_resolved` log added by [#558](https://github.com/kristenmartino/gridpulse/pull/558)
shows JEA with 51 distinct origins in 56 ticks, 5 repeats, **zero** stalls. Six
of the seven worst BAs track origin repetition. JEA does not.

The hypothesis under test was: JEA's hours entered the pending buffer normally,
the settled actual never published, and `models.drift._expire_pending` dropped
them at `PENDING_STALE_HOURS = 120`. A **second loss channel**, unresolved-actual
rather than never-proposed.

**That hypothesis is refuted.** Its predicted mechanism accounts for at most 4
of 8568 hours fleet-wide, and 0 for JEA. What is real is a *third* thing neither
the issue nor the hypothesis named: the origin **skips forward**, and every
skipped origin permanently loses one target hour.

---

## 2. Method

The pre-registration — windows, controls and the readings fixed in advance — was
written before any `pending`/`records` payload was read.

### The instrument is exact, not inferential

`gridpulse:drift_horizon:{BA}` holds, per (model, horizon), the full `records`
array keyed by target hour, plus one shared `pending` array of unresolved
snapshots. Reading both classifies every target hour with no inference:

| class | meaning |
|---|---|
| **RESOLVED** | hour is in ensemble/24h `records` |
| **PENDING** | hour is in `pending` with `horizon == "24h"` |
| **ABSENT** | neither — no snapshot exists and none is coming |

Nothing else can remove a record: `merge_and_trim` caps at 720 records (30 days,
and every BA sits at the cap, so the 7-day window is far inside it) and
`regrade_records` rewrites actuals but never drops rows. Both were read rather
than assumed.

### Windows

`t_ref` = the payload's own `last_updated_at`.

* **W_full = [t_ref − 119h, t_ref − 1h]** (119 h × 51 = 6069 hours). Entirely
  inside the 120 h expiry horizon, so **nothing in it has been expired yet** —
  which is what makes ABSENT mean *never proposed* rather than *proposed and
  erased*. This is the only window where the two are separable.
* The oldest 49 h of the 168 h window is **not** separable by this instrument
  and is not presented as though it were. §6 bounds it by a different argument.

### Controls, designated before the measurement

1. **Never-frozen controls** TAL, CPLW, FMPP, GCPD
   (`FORECAST_ORIGIN_REGRESSION.md` §2 scored 487/487 on the first three).
2. **Partition identity** — RESOLVED + PENDING + ABSENT must equal exactly 119
   on every BA.
3. **Cross-instrument reproduction** — recomputing `n_7d` from `records` with
   the real `_within_window` and `filter_low_actuals` must reproduce the number
   the production scoring job wrote, per BA.
4. **A control designed to disagree** — control 3 re-run with the window
   deliberately set to 167 h **must fail**. This investigation has twice had a
   harness score ~100% agreement for the wrong reason (§2 and §11 of
   `FORECAST_ORIGIN_REGRESSION.md`), and a check that passes on a knowingly
   wrong input proves nothing.

| control | result |
|---|---|
| C1 never-frozen controls, channel C | **0, 0, 0, 0** |
| C2 partition identity | **51 / 51** |
| C3 `n_7d` reproduced from records | **51 / 51** |
| C3b `n_low_actual_excluded_7d` reproduced | **51 / 51** |
| C4 wrong 167 h window disagrees | **50 / 51** — C3 is not vacuous |

The harness refuses to print a result if C3 fails, or if C4 *passes*.

### Reproducing it

`scripts/drift_coverage_channels.py` runs in two stages, because production
Redis is a private Memorystore. Stage 1 is inlined into a Cloud Run job
execution so it runs inside the VPC; the `^%%^` prefix sets a custom `--args`
delimiter so the source keeps its commas, and `%%` must not appear in the file:

```bash
SRC=$(cat scripts/drift_coverage_channels.py)
gcloud run jobs execute gridpulse-scoring-job --region us-east1 --wait \
  --args="^%%^-c%%$SRC%%--dump"
gsutil cp gs://nextera-portfolio-energy-cache/probe/drift_channel_split.json .
PYTHONPATH=. python scripts/drift_coverage_channels.py --analyze drift_channel_split.json
```

`python -c <source> --dump` leaves `sys.argv == ["-c", "--dump"]`, so the
script's own argparse sees the flag — the container's command is `python` and
`--args` replaces `-m,jobs,scoring` wholesale.

Stage 2 reads the GCS vintage mirror directly and needs only ADC — no VPC, no
Redis. It calls the real `_within_window`, `filter_low_actuals` and
`deserialize_records` rather than reimplementing them.

---

## 3. JEA — the verdict

**Hypothesis refuted.** Over W_full, JEA has **0** hours in the pending buffer
and **65** absent. Nothing was waiting to resolve; nothing was expired.

**All 65 absent hours have a settled actual in the vintage mirror.** The actual
published. So does every one of the 530 absent hours fleet-wide — 530 of 530.
"The actual never published" is false at the fleet level, not merely for JEA.

What JEA has instead is one contiguous 62-hour run of absent hours,
**2026-08-16T01:00 → 2026-08-18T14:00**, and an upstream feed that stopped and
then caught up in batches:

| target hours | all first seen at | first-sight lag |
|---|---|---|
| 2026-08-15T00:00 → 23:00 | 2026-08-17T16:04:22Z | 41–64 h |
| 2026-08-16T00:00 → 08-17T14:00 | 2026-08-17T15:04:11Z | 1–39 h |
| 2026-08-17T15:00 onward | hourly, ~1.15 h | normal |

Twenty-four hours arriving at one instant is a batch, not a trickle.

**The positive control is inside JEA's own window.** The six hours
2026-08-15T19:00 → 08-16T00:00 carry first-sight lags of **39–45 h** and are
**RESOLVED**. The pending buffer demonstrably tolerates a two-day-late actual;
it did not drop them. Whatever cost JEA its 65 hours happened *before*
resolution, not at it.

And the run's end is the tell. It stops at **2026-08-18T14:00**, exactly
**24 hours** after the feed returned to real time at 08-17T14:00 — the 24h
horizon's own offset. The damage tracks the *proposal* step, which is keyed on
the origin 24 h earlier, not the resolution step.

---

## 4. The mechanism: the origin skips

`snapshot_horizon_predictions` targets `origin + 24h`, where
`_resolve_forecast_start` returns `min(last_real_demand, last_featured_ts) + 1h`.
So target `T` is proposed only by a tick whose resolved origin is `T − 24h`,
which requires some tick to see hour `T − 25h` as its **newest published hour**.

Hour `h` is the newest hour for at least one tick iff a tick boundary falls
between `captured_at(h)` and `captured_at(h+1)`. **If EIA publishes `h` and
`h+1` in the same tick, no tick ever sees `h` as newest.** Origin `h+1` is
skipped, and target `h+25h` is never proposed. There is no re-proposal path: the
snapshot is keyed on the origin, and an origin that never occurred never
produces one.

This is a *skip*, not a *repeat* — which is precisely why the
`forecast_start_resolved` table, which counts repeated and stalled origins,
shows JEA clean. JEA's origins do not repeat. They jump.

**Tested, with the control designed to disagree stated first:** if ABSENT hours
show "origin hour published in the same tick as the hour before it", RESOLVED
hours must not. If both do, the test is vacuous.

| | same tick | different tick | share same |
|---|---:|---:|---:|
| **ABSENT** (no snapshot) | 436 | 91 | **82.7 %** |
| **RESOLVED** (control) | 9 | 5522 | **0.16 %** |

A 500-fold separation. JEA alone: 64 of 65 absent (98.5 %) against 0 of 54
resolved.

---

## 5. Fleet-wide split — the deliverable

Ensemble 24h over W_full, 6069 hours. **Coverage 5531 / 6069 = 91.14 %.**

| channel | hours | share of shortfall |
|---|---:|---:|
| **A — origin skip** (never proposed; upstream published two hours in one tick) | **436** | **81.0 %** |
| **B — origin freeze** (never proposed; origin held by `last_featured_ts`) | **91** | **16.9 %** |
| **C — unresolved actual** (proposed, still pending) | **8** | **1.5 %** |
| unknown (hour absent from the vintage mirror) | 3 | 0.6 % |

Channel C — the hypothesis — is **1.5 %**, and all 8 are transient: they sit in
the buffer and will resolve. **Zero were expired** (§6).

Per BA, sorted by `n_7d`. Channels A and B are both "never proposed"; they
differ in which term of the `min()` was responsible, and only B is reachable by
the origin-cap fix.

| BA | n_7d | resolved /119 | A skip | B freeze | C unres | ? |
|---|---:|---:|---:|---:|---:|---:|
| LGEE | 94 | 90 | 7 | **21** | 1 | 0 |
| JEA | 102 | 54 | **64** | 1 | 0 | 0 |
| SPA | 113 | 83 | 13 | **18** | 2 | 3 |
| TIDC | 129 | 87 | 28 | 3 | 1 | 0 |
| PSCO | 136 | 95 | 8 | **16** | 0 | 0 |
| CAISO | 143 | 95 | 24 | 0 | 0 | 0 |
| NYISO | 145 | 99 | 20 | 0 | 0 | 0 |
| FPL | 146 | 99 | 20 | 0 | 0 | 0 |
| MISO | 146 | 99 | 20 | 0 | 0 | 0 |
| SPP | 146 | 99 | 20 | 0 | 0 | 0 |
| ERCOT | 147 | 99 | 20 | 0 | 0 | 0 |
| LDWP | 147 | 106 | 2 | **10** | 1 | 0 |
| ISONE | 148 | 101 | 18 | 0 | 0 | 0 |
| IID | 149 | 105 | 3 | **10** | 1 | 0 |
| PJM | 149 | 101 | 18 | 0 | 0 | 0 |
| AZPS | 150 | 106 | 3 | **10** | 0 | 0 |
| SEC | 152 | 104 | 15 | 0 | 0 | 0 |
| NEVP | 155 | 107 | 12 | 0 | 0 | 0 |
| TVA | 158 | 110 | 9 | 0 | 0 | 0 |
| CPLE | 159 | 111 | 8 | 0 | 0 | 0 |
| SCL | 159 | 114 | 4 | 0 | 1 | 0 |
| DUK | 160 | 113 | 6 | 0 | 0 | 0 |
| BPAT | 161 | 114 | 5 | 0 | 0 | 0 |
| PACE | 161 | 113 | 5 | 1 | 0 | 0 |
| SOCO | 161 | 113 | 6 | 0 | 0 | 0 |
| TPWR | 161 | 113 | 5 | 0 | 1 | 0 |
| WALC | 161 | 114 | 5 | 0 | 0 | 0 |
| EPE | 162 | 115 | 4 | 0 | 0 | 0 |
| PNM | 162 | 114 | 5 | 0 | 0 | 0 |
| SC | 162 | 114 | 4 | 1 | 0 | 0 |
| DOPD | 163 | 115 | 4 | 0 | 0 | 0 |
| FPC | 163 | 116 | 3 | 0 | 0 | 0 |
| IPCO | 163 | 115 | 4 | 0 | 0 | 0 |
| NWMT | 163 | 116 | 3 | 0 | 0 | 0 |
| SCEG | 163 | 115 | 4 | 0 | 0 | 0 |
| TEC | 163 | 116 | 3 | 0 | 0 | 0 |
| AECI | 164 | 116 | 3 | 0 | 0 | 0 |
| AVA | 164 | 117 | 2 | 0 | 0 | 0 |
| BANC | 164 | 117 | 2 | 0 | 0 | 0 |
| CHPD | 164 | 116 | 3 | 0 | 0 | 0 |
| CPLW | 164 | 116 | 3 | 0 | 0 | 0 |
| PACW | 164 | 116 | 3 | 0 | 0 | 0 |
| SRP | 164 | 116 | 3 | 0 | 0 | 0 |
| TEPC | 164 | 117 | 2 | 0 | 0 | 0 |
| FMPP | 165 | 117 | 2 | 0 | 0 | 0 |
| GCPD | 165 | 117 | 2 | 0 | 0 | 0 |
| GVL | 165 | 117 | 2 | 0 | 0 | 0 |
| HST | 165 | 118 | 1 | 0 | 0 | 0 |
| PGE | 165 | 117 | 2 | 0 | 0 | 0 |
| PSEI | 165 | 117 | 2 | 0 | 0 | 0 |
| TAL | 165 | 117 | 2 | 0 | 0 | 0 |
| **FLEET** | **7869** | **5531** | **436** | **91** | **8** | **3** |

Two readings worth stating.

**Channel B is a ten-BA phenomenon** — LGEE 21, SPA 18, PSCO 16, and 10 each on
LDWP, IID and AZPS account for 85 of the 91. Forty-one BAs have zero. This is
also the resolution of the #559 puzzle in the issue's framing: the null-row
census (IID 37, LGEE 16, PSCO 11, TIDC 10) **does** predict channel B — the
overlap is five of its seven named BAs — it just does not predict `n_7d`,
because channel B is only 17 % of the shortfall. IID leads the census and sits
at 149 because its channel-B loss is 10 hours.

**Channel A is universal and mostly structural.** Every healthy BA loses 1–3
hours per 119 to it, including all four controls. That is not a defect firing
occasionally; it is EIA's ~1.02–1.30 h publication lag drifting against an
hourly tick clock until two hours occasionally land together.

---

## 6. The expiry channel is bounded at 4, not 66

`_expire_pending` drops a snapshot only when `target_ts < now − 120h`. A
snapshot therefore survives to resolve **iff** its hour's actual publishes
within 120 h. So the expiry channel is bounded above by the count of hours whose
actual published later than that, or never.

Across the full 168 h window and all 51 BAs, that count is **4**:

```
LGEE 2026-08-19T20:00     SPA 2026-08-19T02:00
SPA  2026-08-19T03:00     TIDC 2026-08-20T17:00
```

— four hours the mirror has not received at all, all within the last 36 hours,
i.e. plausibly just not yet published rather than never. The only hours anywhere
in the 30-day mirror with a first-sight lag exceeding 120 h are **10 SPA hours
from 2026-07-25 → 07-28**, far outside the window.

This is what closes the undecidable band. The oldest 49 h of the 168 h window
cannot be classified by the pending/records instrument, because expiry has
erased whatever was there — but it contains **no expiry-eligible hour**, so
expiry took nothing in it either. Its 161 missing records are channels A and B
in some proportion this measurement does not resolve, and the band's coverage
(2338 / 2499 = 93.6 %) is slightly *better* than W_full's 91.1 %, which is what
one expects if its losses are the same ordinary skips.

**`168 − 102 = 66` was suggestive arithmetic across two buffers, and it does not
survive contact with either.** JEA's expiry channel is 0.

---

## 7. What is not fixable, and why

**The skipped hour cannot be recovered at a 24-hour lead.** The payload written
at the tick *after* a skip does contain a prediction for the orphaned hour — but
at a 23-hour lead, from a different origin. Re-proposing it would file a 23h-lead
prediction in a window labelled 24h, which is exactly the mislabelling P2-19
([#273](https://github.com/kristenmartino/gridpulse/issues/273))'s `lead_hours`
field and `filter_by_lead` exist to prevent. There is no 24h-lead forecast for
that hour because the origin it would have needed never existed.

So `n_7d = 168` is **not achievable** under the current design, and the ceiling
is not a constant — it is set per BA by how the upstream feed's publication
timing interacts with the hourly tick clock. The observed ceiling in this
capture is 165 of 168.

Channel A is therefore a **disclosure obligation, not a bug**, and it warrants
its own issue rather than a fix: it converts a one-off finding into a standing
statement the benchmark page has to keep making. It is not filed here.

**Channel B is a real defect and is already filed** —
[#559](https://github.com/kristenmartino/gridpulse/issues/559), the
`dropna(subset=autoregressive)` row deletion that holds `last_featured_ts`
behind. That is the one the origin-cap fix addresses.

### Pre-deploy prediction for the origin-cap fix

The fix reaches channel B and nothing else. Its **upper bound** is therefore
**91 records over 119 h fleet-wide — 1.5 % of theoretical coverage**, moving
91.14 % → at most 92.64 %, concentrated on six BAs:

| BA | recoverable (upper bound, /119) |
|---|---:|
| LGEE | 21 |
| SPA | 18 |
| PSCO | 16 |
| LDWP | 10 |
| IID | 10 |
| AZPS | 10 |
| TIDC | 3 |
| PACE, SC, JEA | 1 each |

It should move **JEA by at most 1 hour**, and it should move the 41 zero-B BAs
not at all. A post-deploy reading that shows the median BA improving is
measuring something other than this fix.

---

## 8. Limits, stated

* **Channel A/B attribution is inferential, channel C is not.** RESOLVED /
  PENDING / ABSENT is read directly off the payload. The A-vs-B split within
  ABSENT is inferred from vintage capture instants, and rests on
  `_resolve_forecast_start` being the only thing that sets the origin. Its
  support is the 82.7 % vs 0.16 % separation in §4, not a per-tick log — the
  `forecast_start_resolved` line only exists post-#558 (2026-08-18T09:47Z) and
  covers ~2.5 days of a 7-day window. **The A/B numbers are not extrapolated
  across that gap; they are computed from the vintage mirror, which spans it.**
* **Capture instants are bucketed to the hour.** Two publications 3 minutes
  apart across a tick boundary read as one tick. This can only *over*-count
  channel A, so B's 91 is a lower bound and A's 436 an upper one.
* **The oldest 49 h of the 168 h window is unattributed between A and B.** §6
  rules out expiry there; it does not split the rest.
* **Absent rows and null rows remain indistinguishable** in the mirror
  (`FORECAST_ORIGIN_REGRESSION.md` §11). Nothing here depends on which: the
  origin-skip test reads capture *instants*, not values.
* **One capture, not a rolling measurement.** JEA's 62-hour episode is a single
  upstream event. Whether channel A's fleet baseline of 1–3 hours per BA is
  stable across weeks is not established by one snapshot.

---

## 9. Two inherited claims that did not survive re-checking

**`STATUS.md` (2026-07-16): "Ten regions never revise at all (PJM, PNM, PGE,
CHPD, GVL, HST, JEA, TAL, TPWR + TIDC)."**

*Instrument, stated because it is easy to cross-apply the wrong one.* This is
`VintageRecord.n_updates`, which counts movements of **`D` — the actual demand —**
beyond `REVISION_EPSILON_MW` after first sight, including a placeholder `D`
later replaced by a metered value. It is **not** the day-ahead-forecast (`DF`)
revision rate reported by
[`BENCHMARK_METHODOLOGY.md`](BENCHMARK_METHODOLOGY.md) §6 and
[`BENCHMARK_PROVENANCE.md`](BENCHMARK_PROVENANCE.md) ("seven never revise: PJM,
MISO, ERCOT, CAISO, GVL, SPP, NYISO"). Different series, different sample.
**Nothing here contradicts those two documents or the benchmark's two official
arms**, and the figures below must not be read across to them.

Measured over the 30 days the vintage mirror now holds — a window that begins
2026-07-21, entirely *after* the claim was written, so this is a fresh reading
rather than a re-run of theirs — all ten revise:

| | PJM | PNM | PGE | CHPD | GVL | HST | JEA | TAL | TPWR | TIDC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hours revised | 5.1% | **27.5%** | **33.7%** | 2.9% | 2.8% | 2.8% | 16.1% | 6.3% | 13.9% | 2.8% |

PNM and PGE revise a third of their hours. The claim is not a small drift; for
those two it is inverted. It is left in `STATUS.md` as dated history rather than
edited, but nothing should be built on it.

**"JEA is a low-revision, clean feed."** JEA revises 16.1 % of hours and, in this
window, went dark for two days and back-filled 24 hours at a time. Its `n_7d` is
low for a reason that has nothing to do with revision rate.
