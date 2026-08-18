# Public Forecast Benchmark — methodology

**What is measured, how, and what the result may and may not be used to claim.**

This file is the *rules*. The numbers live in the generated artifacts —
[`BENCHMARK_SCOREABILITY.md`](BENCHMARK_SCOREABILITY.md),
[`BENCHMARK_PROVENANCE.md`](BENCHMARK_PROVENANCE.md) — and in the live
`gridpulse:benchmark:{ba}` payload, so the numbers can move without this
document going stale. Implementation: [`models/benchmark.py`](../models/benchmark.py)
(pure, 45 unit tests) and the `write_benchmark_metrics` phase in
[`jobs/phases.py`](../jobs/phases.py).

---

## 1. The comparison

Every US balancing authority already publishes a day-ahead demand forecast,
free, through EIA-930 (the `DF` column). That is the incumbent. So the only
question worth measuring is **relative**: for the same hour, judged by the
same settled truth, whose number was closer?

| | |
|---|---|
| **Official arm** | The earliest day-ahead forecast we observed the BA publish for the target hour — **provided we saw it before the hour settled** (§12.1, §12.9; hours first seen later are dropped as `stale_capture`) |
| **GridPulse arm** | Our **ensemble's** forecast for that same hour — always the ensemble, which on a substituted BA is not the series we serve (limit 6) |
| **Truth** | EIA's settled value for that hour — the *same* value grades both |

This is a continuously recomputed measurement, not a one-off replay: it runs
every hourly scoring tick, per BA, on instrumentation the pipeline already
maintained for other reasons.

## 2. Where every number comes from

| Quantity | Source | Field |
|---|---|---|
| Official, as-issued | `gridpulse:vintage:{ba}` | `first_seen_df` — the earliest day-ahead forecast we observed for the hour |
| Official, as-revised | the live EIA demand frame | `forecast_mw` — EIA's *current* day-ahead value for the hour |
| GridPulse | `gridpulse:drift_horizon:{ba}` | `models.ensemble.{24h,48h}.records[].p` — the prediction as stored when it was made |
| Truth | `gridpulse:vintage:{ba}` | `last_d` — EIA's latest (settled) value |
| Exclusion inputs | `gridpulse:vintage_summary:{ba}` | revision class, DF coverage |
| Observed lead | `gridpulse:forecast:{ba}:1h` | `scored_at` vs row 0 + H |
| Output | `gridpulse:benchmark:{ba}` | the full payload, per BA, per tick |

Two properties of this wiring matter:

- **The GridPulse arm contributes only a prediction.** The drift record's own
  `actual` is ignored; the benchmark re-grades every prediction against the
  vintage `last_d` itself. Both arms are therefore scored by one yardstick,
  computed in one place — neither inherits its own pipeline's notion of truth.
- **The prediction value is fixed before the outcome is known.** It is
  snapshotted at forecast time (`made_at`, with the target hour) and carried
  unchanged into the record built once an actual exists — the record's
  timestamp is later, the number is not. Nothing here re-runs a model over
  history, so no later retrain, feature, or hindsight can reach our side of
  the comparison.

## 3. One truth, one hour set

Two disciplines, both load-bearing:

**Settled truth only.** EIA's first-published actual is preliminary *by
design*, not by accident: the [Form EIA-930 instructions][eia930] direct
respondents to "submit their best estimates on schedule and correct the data
with a resubmission within 3 days or as soon as the actual data is available,"
and to resubmit within 30 days when an unexpected error exceeds 10 MWh. So a
first-published value is an estimate the respondent is instructed to replace.
On high-revision feeds we have observed it up to ~70% wrong — that magnitude is
ours, measured; the fact that it revises at all is EIA's, documented. Scoring
against it would grade both arms on a number EIA itself later withdrew. Only
`last_d` counts — EIA's latest value for the hour, refreshed on every tick. Note what
that is *not*: no flag marks an hour final, so `last_d` is settled in
practice (revisions stop) rather than by declaration, and a very recent hour
may still move. Both arms move with it together.

**The same hours, always.** An hour enters the comparison only if *both*
arms have a value for it. A one-sided score would put a 30-day official
record against a partial GridPulse one and call the difference accuracy.

## 4. Which hours count — and what each rule costs

Every candidate hour passes five tests, in this order. The per-reason counts
ship in the payload as `excluded_hours`, because the drops are **not neutral
across BAs** — MISO loses ~20% of its hours to stubs and ~10% more to hours
with no published `DF` at all, while the cleanest feeds still lose a percent
or two (none loses nothing), and a reader who cannot see that will reasonably
assume the worst.

| Rule | Dropped when | Why |
|---|---|---|
| `no_df` | the BA published no day-ahead forecast for the hour | nothing to compare against |
| `unsettled` | no finite, positive settled actual yet | truth not available |
| `unresolved_stub` | settled value still equals the day-ahead forecast (within 0.01 MW) | see below |
| `first_seen_placeholder` | the hour was flagged `D == DF` at first sight | dropped conservatively, even if later corrected |
| `no_gridpulse` | we have no matured prediction for the hour | keeps the hour sets identical |

**The stub trap** is the one that would have quietly destroyed the whole
exercise. For hours a BA has not yet metered, EIA publishes *the official
forecast itself* as the actual. Score those hours and the official forecast
records a perfect prediction on hours it never made — while our forecast is
graded against their forecast rather than against reality. `unresolved_stub`
is the sharper of the two defences: it catches the hour whether or not it was
flagged at first sight, because it tests the *settled* value.

The 0.01 MW tolerance is not a fudge factor: EIA publishes to 2 decimal
places and the vintage store rounds identically, so it absorbs float noise
and nothing else.

## 5. Which BAs count

A BA is excluded only when it cannot be scored *fairly*, and the reason is
published alongside the exclusion — the exclusion list is part of the
product, not an omission from it.

| Reason | Test | Rationale |
|---|---|---|
| `broken-feed` | vintage revision class is `broken` | the feed's provisional readings revise so heavily that intraday scoring is meaningless — **and** (ADR-009) GridPulse anchors its own forecast on that BA's day-ahead value, which would make the comparison partly self-referential |
| `df-coverage` | the BA publishes `DF` for < 80% of hours | too sparse to score fairly |
| `insufficient-paired-hours` | fewer than 200 hours survive §4 | too thin for a per-BA verdict |

**The direction of this bias is against us.** The broken-feed exclusions are
BAs with sloppy data operations, which plausibly also forecast sloppily — so
the excluded set is disproportionately made of BAs where GridPulse would
likely win. The rule removes our best cases, not our worst.

**The broken-feed row must not be read as disposing of the self-reference
(#539).** It excludes the BAs where the anchor substitution is *deliberate* and
standing — ADR-009, live-classifier-driven, `broken` class only. It does not
establish that the scored set is free of the dependence, and it is not true
that it is: on scored BAs the anchor can still be seeded by `DF`, through a
second and undesigned path. That path, its measured size and why it is not
removed are **limit 11** (§12).

**Current standing is not stated here, deliberately (#535).** This paragraph
read "44 of 51 BAs scoreable" for three weeks while the live payload served
**25** — a prose sentence cannot track a number recomputed every hour, and
pretending otherwise is how the page and its own methodology came to disagree
in public. The count is published, live, as `n_scoreable` on
[`/api/v1/benchmark`](https://gridpulse.kristenmartino.ai/api/v1/benchmark).
Per-BA exclusions and their reasons are listed there too, and mirrored in the
dated snapshot at
[`BENCHMARK_SCOREABILITY.md`](BENCHMARK_SCOREABILITY.md).

### The gate measures the BA, not us (#535)

`df_coverage` is the share of hours **EIA published a day-ahead forecast for**.
It is the only coverage figure the exclusion gate acts on, because the
exclusion text it produces makes a claim about the balancing authority.

Until #535 it was not that number. `first_seen_df` was pinned on the tick that
first admitted the hour and never revisited, so a DF EIA published slightly
later was lost permanently — and the gate read our collector's timing as the
BA's publishing behaviour. Twenty-six BAs were excluded on it, five of them
large ISOs. Measured against EIA directly on 2026-08-17, exactly one of that
set (SPP, 53.8%) was genuinely below the threshold.

Our own capture rate did not disappear; it moved to `df_asissued_coverage`,
which is published beside the gate figure and decides nothing. The two answer
different questions and are no longer allowed to share a number.

### Is EIA's `DF` a strawman? Probed against NYISO's own feed (2026-08-18)

A fair objection to this whole benchmark: EIA-930 is a regulatory rollup, so
maybe we score operators on a degraded copy of a forecast they publish better
elsewhere. NYISO is the cheapest place to check — it publishes both its own
day-ahead load forecast (`isolf.csv`) and its integrated actual load
(`palIntegrated.csv`), auth-free.

**The truth side is settled outright: EIA's `D` for NYISO *is* NYISO's own
data.** Summed across NYISO's eleven zones and aligned to EIA's periods, the two
actual series agree to **0.001%, correlation 1.000000** over 144 hours. There is
no quality gap to close on the yardstick both arms are scored against.

**On the forecast side EIA's `DF` is not the weaker series.** Over 212 hours
(2026-08-07 → 08-17), scored against that shared settled truth:

| series | MAPE | median APE | bias |
|---|---:|---:|---:|
| EIA-930 `DF` — what this benchmark scores | **2.184%** | 2.043% | +0.30% |
| NYISO `isolf` — the operator's own | 2.526% | 2.107% | −0.58% |

The two are genuinely different — they diverge on **every one of the 212 hours**,
by 324 MW on average (1.59% of load) — but NYISO's own is closer on **111 of
212**, a coin flip, and slightly worse on the mean. We are not scoring against a
degraded copy.

**Limits, stated because this is a probe and not a study.** One BA, ~9 days, a
single window, and `isolf`'s archived vintage may already include intraday
revisions — which would flatter the operator, so the finding is conservative in
the direction that matters. Under
[`EVALUATION_POLICY.md`](EVALUATION_POLICY.md) this is nowhere near the bar for
a methodology change (8 rolling origins, WAPE, satisficing constraints); it is
sufficient only for the narrow claim it makes, which is that the strawman
objection does not hold for NYISO. Alignment was fixed empirically from the two
*actual* series, with no forecast involved — a first attempt using the naive
timezone conversion was off by one hour and reversed the conclusion.

## 6. Two official arms

EIA revises the day-ahead forecast for some BAs after the fact. Across a
**10-BA sample** chosen for a spread of feed behaviours — not the fleet —
seven never revise (PJM, MISO, ERCOT, CAISO, GVL, SPP, NYISO), while PSEI
revises 26.4%, SOCO 24.2% and FMPP 5.2%. FMPP is worth naming: revision
makes its forecast *worse*, so revision is not a one-way advantage to the
operator. Picking one value to score would invite a fair objection either
way, so the official side is scored **twice, on the same hours, against the
same settled truth**:

- **`official`** — as-issued: the earliest day-ahead forecast we observed,
  restricted to hours we saw within `FRESH_CAPTURE_LAG_HOURS` of the target
  (#358 — a backfilled first sighting is already post-revision).
  The fair comparison, and the primary one.
- **`official_revised`** — as-revised: EIA's current value, which for a
  revising BA carries hindsight our forecast never had. The conservative
  comparison.

Both verdicts are published (`winner` and `winner_vs_revised`,
`delta_mape` and `delta_mape_vs_revised`). Where a revision would flip a
BA's result, the payload says so rather than reporting only the favourable
one. For every BA that does not revise, the two arms are identical by
construction.

Measured effect: no sampled operator's own **median APE** moves by more than
**1.43 points** between the two scorings. Read that narrowly — it bounds a
median, on the official side only, and every verdict here is decided on
**mean** MAPE (§8), which the probe does not measure. A fat-tailed feed can
move a mean far more than a median, so nothing measured so far bounds a
head-to-head result. Whether a flip occurs is decided per BA in the payload,
which is why `winner_vs_revised` is published beside `winner` rather than
asserted to agree with it. See
[`BENCHMARK_PROVENANCE.md`](BENCHMARK_PROVENANCE.md).

## 7. Lead time

**Ours is measured, not labelled.** The forecast anchors on the last hour for
which EIA reports a positive `D` — which is not always a metered value, and
that distinction is limit 11 (§12), not a detail — so EIA's publishing lag
makes a nominal "24h" record slightly shorter than 24 hours in wall-clock
terms. The phase computes the realized
lead every tick from the forecast payload — row 0 + H versus `scored_at`,
the same target hour the drift pipeline snapshots and the benchmark later
grades — and the payload carries it as `observed_lead_h` with `lead_basis:
"observed"`.

**Theirs is documented, not observed.** The [Form EIA-930 instructions][eia930]
put the day-ahead submission at **17–41 hours** ahead depending on hour-of-day.
We cite that; we do not measure it, and the doc says so wherever it appears.

**The conservative arm is earned.** The benchmark also scores our nominal
**48h** snapshot. That arm may be *labelled* conservative only while the
tick's own observed lead exceeds the operators' documented 41h maximum —
the check runs per tick, so if EIA's publishing lag ever grew, the label
would lapse on its own rather than persist as a stale assertion.

**The 24h arm is not lead-matched, and the mismatch appears to run in our
favour.** Across the 15 BAs sampled, our realized lead is 23.80–23.95h —
inside their documented 17–41h window, whose midpoint is ~29h. We have never
observed an actual submission time (§12.2), so this is a comparison of our
measurement against their documentation, not of two measurements; on that
basis the operator plausibly forecast from further out than we did on a
typical hour. This is the reason the 48h arm exists, and the reason §12 lists
it as a limit rather than leaving a reader to find it.

**What the code enforces:** when a tick has no measurement, the block ships
`observed_lead_h: null` and `lead_basis: "nominal"` — the *label* degrades,
the block still publishes. So the rule is editorial and belongs to whoever
writes the surface: **never quote a lead from a block whose `lead_basis` is
`nominal`.** Where a realized number exists, it is the shorter one, and it is
the one to quote.

## 8. Metrics

Four, always reported together, always with `n`:

| Metric | Definition | Why it is here |
|---|---|---|
| **MAPE** | mean of \|a−p\| / a × 100 | the industry-standard headline, and the statistic the winner is decided on — but dominated by small-denominator hours, which systematically misrepresents small BAs |
| **median APE** | median of \|a−p\| / a × 100 | the typical hour, undragged by a fat tail; also the statistic the offline reports use |
| **MAE** | mean of \|a−p\|, in MW | scale-anchored; immune to the denominator problem |
| **WAPE** | Σ\|a−p\| / Σ\|a\| × 100 | the honest percentage for small BAs — the ones where this benchmark's most striking results live |

**Mean and median are both published, per arm.** They answer different
questions — "how much error accumulated" versus "what does a typical hour
look like" — and a wide gap between them is itself the finding: it says a
BA's error is tail-driven rather than pervasive. `delta_median_ape` ships
next to `delta_mape` for the same reason, so a reader can see at a glance
whether the headline verdict is metric-dependent rather than having to take
it on trust. **The winner is decided on mean MAPE**, always, and never
re-picked per BA to whichever statistic reads better.

**Two cautions when quoting across artifacts.** The offline reports
([scoreability](BENCHMARK_SCOREABILITY.md),
[provenance](BENCHMARK_PROVENANCE.md)) report median APE, and the live
payload's headline `mape` is a mean — so match the statistic before carrying
a figure between them. And even matched, the *hour sets differ*: the offline
reports score every eligible hour, while the payload scores only hours where
both arms have a value (§3). Same BA, same statistic, still not the same
population.

**Only MAPE decides anything.** `winner`, `winner_vs_revised` and every
fleet win/loss count compare mean MAPE. Median APE, MAE and WAPE are
published for interpretation and feed no decision — a BA that loses on MAPE
is reported as a loss even where WAPE would flatter it.

**Everything moves a per-BA number.** SOCO reads 1.84% median APE over 30
days on its *own* day-ahead forecast; an earlier indicative run of *our*
ensemble on the separate drift instrument read 5.82% mean sMAPE over 7 days.
Those two figures differ in arm, metric, statistic *and* window — four axes,
which is exactly why neither belongs in a sentence with the other, and why
the 5.82% figure appears in no artifact here. Every published per-BA row must
carry **metric, window and `n`**, and name its arm. A single unqualified
number per BA is not a supportable format.

## 9. Windows and sample size

- **Vintage window** — 720 hours (30 days) per BA.
- **Drift window** — 720 records per model per horizon, so the GridPulse arm
  spans a comparable ~30 days.
- **Verdict window** — 30 days. Over 30 days a BA has a median of 649
  *officially scoreable* hours (min 500, MISO) — the count after the four
  vintage-side drops but **before** the `no_gridpulse` join, measured in
  [`BENCHMARK_SCOREABILITY.md`](BENCHMARK_SCOREABILITY.md). Paired hours are a
  subset of that, bounded by how many of our snapshots have matured; the
  per-lead `n` in the live payload is the only true paired count. The
  equivalent 7-day figures (median 133, min 46 scoreable hours — measured
  off-artifact, no committed script regenerates them) are too thin to call a
  per-BA winner, so 7-day numbers may be shown as a trend, never as a
  verdict.
- **Minimum for a verdict** — 200 paired hours after §4.
- **Time to a first verdict** — snapshots resolve at most 24 per horizon per
  day, so a newly scoreable BA needs **at least ~9 days** of ticks before a
  verdict publishes, and longer where drops are heavy. A BA with no verdict
  yet is a BA that has not accumulated evidence, not a BA that lost.

## 10. Fleet aggregation

- **Medians, not means.** The fleet headline is the median of the per-BA
  MAPEs. One catastrophic BA should move its own row, not the fleet's.
- **Wins and losses** are counted on the headline (24h) lead.
- **ERCOT is reported separately**, not folded into the aggregate: its
  official forecast is the fleet's best at ~1.2%, and the argument there
  rests on quantiles rather than a point scorecard.
- **Spread is reported for both arms** — min, max, and ratio, as
  `official_spread` and `gridpulse_spread`, both computed on **mean** MAPE.
  Separately, the operators' own day-ahead accuracy spans a wide multiple
  across the scoreable set — that figure is a *median* APE spread from
  [`BENCHMARK_SCOREABILITY.md`](BENCHMARK_SCOREABILITY.md), not the payload
  field, and the two are not interchangeable (§8). The specific multiple is
  **not quoted here**: it is computed over the scoreable *population*, so it
  moves whenever the population does — as it did when #535 changed the
  population by 21 BAs, taking the previously-quoted "41× (1.15% to 47.21%)"
  with it. Read it off the snapshot, which carries its own date.

  Consistency across a fleet is a different claim from winning a head-to-head,
  and it is the more interesting one. It is also **not yet established**: no
  committed artifact publishes our spread, so "ours is flatter than theirs" is
  a hypothesis this benchmark exists to test, not a result it has returned.

## 11. What this is not

- **Not a market-quality claim.** Nothing here is scored on prices, reserve
  margins, or dispatch value — only on demand error.
- **Not a claim about how operators actually forecast.** `DF` is what a BA
  publishes to EIA. It need not be the forecast they dispatch on, and several
  operate more sophisticated internal models; a large error in `DF` is
  evidence about the published series, nothing more.
- **Not an input-independent comparison on every hour.** Where EIA has not
  metered an hour yet it republishes the BA's day-ahead value in the `D`
  field, and our forecast anchors on the last positive `D` — so on those
  hours the seed of our own recursion is the series we are scored against.
  It correlates our error with the operator's rather than shrinking it, it is
  measured and published per BA, and it is not removed because removing it
  forecasts worse (§12, limit 11).
- **Not horizon-complete.** Two leads (nominal 24h and 48h), not the whole
  curve.
- **Not lead-matched.** The headline arm compares our ~23.9h forecast against
  a submission made 17–41h out (§7, §12.4).
- **Not fleet-complete.** Some BAs are excluded, by published rule; the
  count and the per-BA reasons ship in the payload rather than being
  asserted here (§5).
- **Not peer-reviewed, and not a study.** It is a continuously recomputed
  measurement with published rules and reproducible scripts.

## 12. Known limits

1. **Pre-capture revision is invisible.** The vintage window admits an hour
   only once EIA publishes a metered `D`, so a revision landing before our
   first sight cannot be detected. Hence the phrasing everywhere: *the
   earliest day-ahead forecast we observed*, never *their day-ahead
   forecast*. Detecting it would need `DF` captured for hours that have no
   `D` yet — a separate instrument, not built.
2. **The official lead is documented, not observed.** The 17–41h figure
   comes from EIA's Form EIA-930 respondent instructions and lives in one
   place in code (`models.benchmark.OFFICIAL_DOCUMENTED_LEAD_H`); we have not
   measured when any BA actually submitted, and a BA that submits late would
   look better here than it deserves.
3. **Exclusions are not neutral**, and skew against us (§5). Stated rather
   than corrected for.
4. **The headline arm is not lead-matched, and the gap appears to favour
   us.** Our realized 23.80–23.95h (15-BA sample) sits inside the operators'
   documented 17–41h window, midpoint ~29h — so on a typical hour they
   plausibly forecast from further out than we did, on the arm every
   win/loss count is taken from. "Plausibly" because their side is
   documentation, not observation (limit 2). The 48h arm is the mitigation,
   not a cancellation: it publishes beside the headline, not instead of
   it.
5. **The hour set is conditioned on our own availability.** `no_gridpulse`
   (§4) drops an hour from *both* arms because *we* had no matured
   prediction — the operator's forecast for that hour exists and goes
   unscored. Our predictions exist only for hours a tick actually
   snapshotted *and* later resolved, and resolution reads a demand frame
   that has passed a quality guard the vintage capture has not. That does
   not change how an hour is graded (§2 — grading is always against vintage
   `last_d`); it changes *which* hours become eligible at all. Neither
   effect is corrected for; both are why the drop counts ship in the
   payload.
6. **One model, and not always the served one.** The GridPulse arm is always
   the **ensemble**. Nothing here is a per-model claim, and the arm changes
   when the ensemble changes. Where a BA has been substituted onto the
   seasonal-naive baseline (`models/skill.py`), the ensemble is no longer
   what that BA's users are served — the row still scores the ensemble,
   because re-basing the arm onto a fallback would stop it measuring the
   forecaster at all. That makes the published number **worse** than what
   the BA actually serves, and the row says so: `served_series` and
   `serves_scored_model` ship in the payload and mark the row on the page.
   SEC is the live case (published 17.64% for an ensemble it does not
   serve).
7. **Mean/median differ across artifacts** (§8) — a live footgun for anyone
   quoting across documents.
8. **Both arms inherit EIA's settled values.** If a settled value is itself
   wrong, both arms are graded against the same wrong number. Shared bias,
   not differential bias, but not zero.
9. **Backfilled hours are excluded, and the exclusion is not neutral.**
   An hour first seen more than `FRESH_CAPTURE_LAG_HOURS` (3h) after it passed
   cannot supply an "as-issued" forecast: its `first_seen_df` is already
   post-revision. Those hours are dropped as `stale_capture` (#358). Where any
   were dropped, `stale_capture_impact` publishes how their removal moved
   **both** arms, because the direction is not uniform — it favours the
   operator where its revisions improve its forecast and favours us where they
   worsen it. The drop is evaluated before the stub rules, so per-reason counts
   are not comparable to payloads published before this change.
10. **A published row may be one we already grade failing.** Our drift
   monitor grades every model on a rolling 7-day window against the band for
   its own horizon, and a row can be `rollback` there while appearing on this
   page as an ordinary comparison. Since #348 that grade travels with the
   row as `serve_grade` — same model, same lead as the row scores — and a
   `rollback` row is marked. Note the grade is earned by exceeding the
   **acceptable** threshold (7.0% at 24h), not `MAPE_BY_HORIZON`'s `rollback`
   entry (12.0), which `mape_grade` never uses as a boundary; the applicable
   figure ships as `acceptable_max` so the page cannot overstate how bad a
   flagged row must be.
11. **The anchor can be seeded by the operator's own forecast, on BAs this
   page scores (#539).** For an hour EIA has not metered yet it publishes the
   BA's day-ahead value in the `D` field — `D == DF`, exactly — and
   `_resolve_forecast_start` selects the anchor as the last hour carrying a
   *positive* `D`, not the last *metered* one. That value becomes
   `demand_lag_1h` and its autoregressive siblings, and carries through every
   recursive step of the horizon. On those hours our forecast is seeded with
   the series it is then scored against.

   **The protection that exists is one-sided.** §4 drops the placeholder hour
   from *scoring* (`first_seen_placeholder`), which stops the operator being
   credited with a perfect prediction on an hour it never predicted. Nothing
   drops the hour that *seeded* a forecast. The hour we score and the hour
   that anchored it are different hours, and only the first is protected.

   **Size, measured 2026-08-18 over the live 30-day vintage window** — share
   of hours whose first sighting was a placeholder:

   | MISO | CAISO | NEVP | SCEG | ERCOT | fleet median |
   |---:|---:|---:|---:|---:|---:|
   | **36.6%** | **26.6%** | 18.5% | 12.2% | 11.0% | 3.3% |

   A dated snapshot, not a standing figure (§5's rule). The live per-BA number
   ships on every row as `placeholder_pct`, and as `stub_pct` in
   [`BENCHMARK_SCOREABILITY.md`](BENCHMARK_SCOREABILITY.md) — it was already
   published in both places before this limit was written; what was missing
   was any statement of what it means. Note MISO carries the fleet's highest
   rate and is at present excluded for an unrelated reason
   (`insufficient-paired-hours`), so read the payload rather than this table
   for who is scored today.

   **The direction is not in our favour, which is why this is disclosed rather
   than corrected.** Seeding our anchor with the operator's forecast makes our
   error *correlated* with theirs; it does not make ours smaller.

   **It is not removed, because removing it is measurably worse.** On a
   persistence proxy over 14 days, anchoring on the placeholder scores 6.55%
   mean error against **7.72%** when the hour is skipped, winning 9 of 12 BAs
   (`data/vintage.py`) — a decent forecast *for the hour you want* beats a
   real measurement two hours stale, because demand ramps. Refusing to anchor
   would trade a disclosable dependence for a worse forecast, and the
   alternative was measured before it was rejected.

   **What this number is not.** `placeholder_pct` is the share of *hours in
   the window* whose first sighting was a placeholder. Each hour is the newest
   hour for roughly one tick, so it estimates how often a forecast run
   anchored on a placeholder — it does not measure it. The direct measurement
   needs the anchor hour recorded per forecast, which is not instrumented;
   until it is, scored hours cannot be split by anchor provenance and the
   materiality of this limit is stated as unmeasured rather than as small.

## 13. Reproducing it

```bash
python scripts/benchmark_scoreability.py --output docs/BENCHMARK_SCOREABILITY.md
```

```bash
python scripts/benchmark_provenance_probe.py --output docs/BENCHMARK_PROVENANCE.md
```

Both regenerate their committed artifact in place, so any figure quoted from
them is a re-runnable measurement rather than a screenshot. The scoring rules
themselves are pure functions in `models/benchmark.py`, covered by
`tests/unit/test_benchmark.py` (45 tests). Behaviours whose tests have been
verified by assert-applied mutation — deliberately breaking the code and
confirming a named test fails — are the stub predicates, the medians in the
fleet rollup, the shared-hour-set rule, the dual official arm, the earned
conservative label, the observed-lead producer (both its payload key and its
target-hour arithmetic), and the median-APE metric. That list is what has been checked, not a claim
about the suite as a whole. Live per-BA output: `gridpulse:benchmark:{ba}`.

## 14. Changing this methodology

Any change to the drop rules, the exclusions, the metrics, the windows or the
lead definitions must, in the same PR:

1. be recorded here, with the reason;
2. re-run both scripts and commit the regenerated artifacts;
3. state **which direction the change moves our own number**.

A rule that gets looser in our favour needs a materially stronger
justification than one that gets stricter. The point of writing the rules
down first is that we do not get to discover them after seeing the result.

### Change log

**2026-08-18 — the anchor's placeholder dependence is disclosed ([#539]).**
*Direction: moves our own number by exactly nothing.*

No drop rule, exclusion, metric, window or lead definition changed, and no
score was recomputed. This is a **labelling** of a fact the payload was
already publishing: the per-BA rate has shipped as `placeholder_pct` (and as
`stub_pct` in the snapshot) all along, with nothing anywhere saying what it
meant for our own input. Because no rule changed, §14's item 2 does not apply
and neither generated artifact was regenerated for this — their numbers are
unchanged in value, and now have a stated meaning.

Two prose defects were corrected in the same pass, both ours. §7 said the
forecast anchors on the last *real* demand hour when the selector admits
placeholders. §5's broken-feed row, by giving the ADR-009 self-reference as a
*reason for exclusion*, implied the scored set was free of it. Direction on
both: **against us** — each replaces an implied claim of independence with a
stated dependence.

**2026-08-18 — `df_coverage` measures the BA, not our collector ([#535]).**
*Direction: it grows the scoreable population by 21 BAs, which is the direction
that flatters us — so the evidence is given in full, and the rule it relaxes is
one that was measurably **wrong**, not merely strict.*

No drop rule loosened. `MIN_DF_COVERAGE` is unchanged at 0.80, every per-hour
exclusion in §4 still applies, and the `stale_capture` rule got *stricter*: it
is now evaluated on when we observed the **DF** (`df_at`) rather than when we
first saw the hour, so a day-ahead value filled in on a later tick cannot reach
the as-issued arm. What changed is the input to the gate.

`first_seen_df` was pinned on the tick that first admitted an hour and never
revisited, so a DF EIA published slightly later was lost permanently — and the
gate read our collector's timing as the BA's publishing behaviour, then
published that reading as an exclusion reason asserting the BA "publishes a
day-ahead forecast for under 80% of hours". `data.vintage` now gives the DF a
second look, so the number means what its label says.

**Why this is not a favourable rule change dressed up.** Measured against EIA
directly over the same 30-day window on 2026-08-17, the excluded BAs publish DF
for 93.3–100% of hours where we had recorded 58–83% — ISONE 100 vs 66.8, NYISO
96.7 vs 58.0, ERCOT 93.3 vs 64.1. Swept across all 51 BAs, exactly two fall
below the gate upstream, and one of those (SPA) was already excluded as a broken
feed. The hours we lost form a diurnal block aligned to each BA's *local* early
morning, near-identical across unrelated BAs in different interconnects. And the
values are genuine forecasts, not placeholders: of 210 ERCOT / 278 NYISO / 137
PJM recovered hours, 0 / 1 / 0 have `DF == D`.

**Verified by replay before deploy**, applying the new capture to the real
production vintage window for all 51 BAs: **25 → 46 scoreable, 21 restored, 0
newly excluded.** SPP stays out at 53.6% measured coverage, which is genuine.
The post-fix per-BA coverage matches the independent upstream measurement to the
decimal on every restored BA.

**What this does *not* do is make the arm easier on us.** Our capture rate did
not improve — it is now published separately as `df_asissued_coverage` (fleet
median 0.774), and hours whose DF arrived late are still dropped from the
as-issued arm by `stale_capture`. A restored BA is scored on the hours it was
always entitled to be scored on, and `MIN_PAIRED_HOURS` still governs whether
there are enough of them; MISO in particular is expected to sit near that floor
until the window refills.

[#535]: https://github.com/kristenmartino/gridpulse/issues/535


**2026-08-04 — `stale_capture` exclusion ([#358]).** *Direction: measured, not
predicted, and not uniform.* Hours first seen after they had already passed
were being scored on the as-issued arm carrying post-revision values. They are
now dropped.

This **is** a drop-rule change, so §14 applies in full. Rather than state a
single fleet direction — which would be wrong for roughly half the fleet, since
revisions improve some BAs' forecasts and worsen others' — every affected lead
now publishes `stale_capture_impact`: the same hours rescored *without* the
filter, so the shift in both arms is visible per BA and stays true as windows
roll. Positive means the exclusion improved that arm's MAPE.

The rule is **stricter**, and it removes hours where the official arm was
enjoying hindsight. Where a BA's revisions improve its forecast, that costs the
operator and helps us; where they worsen it, the reverse. Both cases are
published rather than summarised.

[#358]: https://github.com/kristenmartino/gridpulse/issues/358


**2026-07-28 — per-row `serve_grade` and served-series disclosure ([#348]).**
*Direction: moves our own number by exactly nothing.* No drop rule, exclusion,
metric, window or lead definition changed, and no score was recomputed — both
additions are context attached to rows that already publish their numbers.
Both move the page against us: one marks rows we already grade `rollback`
(SEC), the other says the ensemble scored on a substituted BA is not what that
BA serves. The second is the only one with a direction argument available at
all, and it runs *toward* disclosure — the alternative, re-basing the arm onto
the served baseline, would have improved SEC's published number while
destroying what the arm measures.

[#348]: https://github.com/kristenmartino/gridpulse/issues/348
[#539]: https://github.com/kristenmartino/gridpulse/issues/539
[eia930]: https://www.eia.gov/survey/form/eia_930/instructions.pdf
