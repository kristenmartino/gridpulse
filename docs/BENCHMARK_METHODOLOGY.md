# Public Forecast Benchmark — methodology

**What is measured, how, and what the result may and may not be used to claim.**

This file is the *rules*. The numbers live in the generated artifacts —
[`BENCHMARK_SCOREABILITY.md`](BENCHMARK_SCOREABILITY.md),
[`BENCHMARK_PROVENANCE.md`](BENCHMARK_PROVENANCE.md) — and in the live
`gridpulse:benchmark:{ba}` payload, so the numbers can move without this
document going stale. Implementation: [`models/benchmark.py`](../models/benchmark.py)
(pure, 41 unit tests) and the `write_benchmark_metrics` phase in
[`jobs/phases.py`](../jobs/phases.py).

---

## 1. The comparison

Every US balancing authority already publishes a day-ahead demand forecast,
free, through EIA-930 (the `DF` column). That is the incumbent. So the only
question worth measuring is **relative**: for the same hour, judged by the
same settled truth, whose number was closer?

| | |
|---|---|
| **Official arm** | The BA's own day-ahead forecast for the target hour |
| **GridPulse arm** | Our served ensemble's forecast for that same hour |
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
- **The prediction is read from a record written before the outcome was
  known.** Nothing here re-runs a model over history, so there is no way for
  a later retrain, a later feature, or hindsight of any kind to leak into our
  side of the comparison.

## 3. One truth, one hour set

Two disciplines, both load-bearing:

**Settled truth only.** EIA's first-published actual is preliminary and, on
high-revision feeds, has been observed up to ~70% wrong. Scoring against it
would grade both arms on a number EIA itself later withdrew. Only `last_d`
counts.

**The same hours, always.** An hour enters the comparison only if *both*
arms have a value for it. A one-sided score would put a 30-day official
record against a partial GridPulse one and call the difference accuracy.

## 4. Which hours count — and what each rule costs

Every candidate hour passes five tests, in this order. The per-reason counts
ship in the payload as `excluded_hours`, because the drops are **not neutral
across BAs** — a stub-heavy BA like MISO loses ~20% of its hours while a
clean feed loses none, and a reader who cannot see that will reasonably
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

**The direction of this bias is against us.** Four of the seven current
exclusions are broken feeds, and a BA with sloppy data operations plausibly
also forecasts sloppily — so the excluded set is disproportionately made of
BAs where GridPulse would likely win. The rule removes our best cases, not
our worst.

Current standing: **44 of 51 BAs scoreable**, with each exclusion and its
reason listed in [`BENCHMARK_SCOREABILITY.md`](BENCHMARK_SCOREABILITY.md).

## 6. Two official arms

EIA revises the day-ahead forecast for some BAs after the fact — measured at
0% for PJM, MISO, ERCOT, CAISO, GVL, SPP and NYISO, but 26.4% for PSEI and
24.2% for SOCO. Picking one value to score would invite a fair objection
either way, so the official side is scored **twice, on the same hours,
against the same settled truth**:

- **`official`** — as-issued: the earliest day-ahead forecast we observed.
  The fair comparison, and the primary one.
- **`official_revised`** — as-revised: EIA's current value, which for a
  revising BA carries hindsight our forecast never had. The conservative
  comparison.

Both verdicts are published (`winner` and `winner_vs_revised`,
`delta_mape` and `delta_mape_vs_revised`). Where a revision would flip a
BA's result, the payload says so rather than reporting only the favourable
one. For every BA that does not revise, the two arms are identical by
construction.

Measured effect on any verdict so far: at most **1.43 points**, flipping
nothing. See [`BENCHMARK_PROVENANCE.md`](BENCHMARK_PROVENANCE.md).

## 7. Lead time

**Ours is measured, not labelled.** The forecast anchors on the last *real*
demand hour, so EIA's publishing lag makes a nominal "24h" record slightly
shorter than 24 hours in wall-clock terms. The phase computes the realized
lead every tick from the forecast payload — row 0 + H versus `scored_at`,
the same target hour the drift pipeline snapshots and the benchmark later
grades — and the payload carries it as `observed_lead_h` with `lead_basis:
"observed"`.

**Theirs is documented, not observed.** The Form EIA-930 instructions put the
day-ahead submission at **17–41 hours** ahead depending on hour-of-day. We
cite that; we do not measure it, and the doc says so wherever it appears.

**The conservative arm is earned.** The benchmark also scores our nominal
**48h** snapshot. That arm may be *labelled* conservative only while the
tick's own observed lead exceeds the operators' documented 41h maximum —
the check runs per tick, so if EIA's publishing lag ever grew, the label
would lapse on its own rather than persist as a stale assertion.

**Rule:** no "*N* hours ahead" claim is published without the realized
number attached. The realized figure is the shorter one, and it is the one
to quote.

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

**Windows move verdicts too.** SOCO reads 1.84% median APE over 30 days and
5.82% mean sMAPE over 7 — same BA, same pipeline. Which is why every
published per-BA row must carry **metric, window and `n`**, and why a single
number per BA, unqualified, is not a supportable format.

## 9. Windows and sample size

- **Vintage window** — 720 hours (30 days) per BA.
- **Drift window** — 720 records per model per horizon, so the GridPulse arm
  spans a comparable ~30 days.
- **Verdict window** — 30 days. A 30-day window yields a median of 649 paired
  hours per BA (min 500); a 7-day window yields median 133 (min 46), which is
  too thin to call a per-BA winner. 7-day numbers may be shown as a trend,
  never as a verdict.
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
- **Spread is reported for both arms** — min, max, and ratio. This is the
  durable observation even in windows where the win count is close: the
  operators' own day-ahead accuracy spans a measured **41×** across the
  scoreable set, while ours is comparatively flat. Consistency across a fleet
  is a different claim from winning a head-to-head, and it is the better
  supported one.

## 11. What this is not

- **Not a market-quality claim.** Nothing here is scored on prices, reserve
  margins, or dispatch value — only on demand error.
- **Not a claim about how operators actually forecast.** `DF` is what a BA
  publishes to EIA. It need not be the forecast they dispatch on, and several
  operate more sophisticated internal models; a large error in `DF` is
  evidence about the published series, nothing more.
- **Not horizon-complete.** Two leads (nominal 24h and 48h), not the whole
  curve.
- **Not fleet-complete.** 7 of 51 BAs are excluded, by published rule.
- **Not peer-reviewed, and not a study.** It is a continuously recomputed
  measurement with published rules and reproducible scripts.

## 12. Known limits

1. **Pre-capture revision is invisible.** The vintage window admits an hour
   only once EIA publishes a metered `D`, so a revision landing before our
   first sight cannot be detected. Hence the phrasing everywhere: *the
   earliest day-ahead forecast we observed*, never *their day-ahead
   forecast*. Detecting it would need `DF` captured for hours that have no
   `D` yet — a separate instrument, not built.
2. **The official lead is documented, not observed.** We cite EIA's own
   instructions for 17–41h; we do not measure when a BA actually submitted.
3. **Exclusions are not neutral**, and skew against us (§5). Stated rather
   than corrected for.
4. **One model.** The GridPulse arm is the served ensemble. Nothing here is a
   per-model claim, and the arm changes when the ensemble changes.
5. **Mean/median differ across artifacts** (§8) — a live footgun for anyone
   quoting across documents.
6. **Both arms inherit EIA's settled values.** If a settled value is itself
   wrong, both arms are graded against the same wrong number. Shared bias,
   not differential bias, but not zero.

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
`tests/unit/test_benchmark.py` (41 tests, including assert-applied mutations
on the stub predicates, the shared-hour-set rule, the dual official arm and
the earned conservative label). Live per-BA output: `gridpulse:benchmark:{ba}`.

## 14. Changing this methodology

Any change to the drop rules, the exclusions, the metrics, the windows or the
lead definitions must, in the same PR:

1. be recorded here, with the reason;
2. re-run both scripts and commit the regenerated artifacts;
3. state **which direction the change moves our own number**.

A rule that gets looser in our favour needs a materially stronger
justification than one that gets stricter. The point of writing the rules
down first is that we do not get to discover them after seeing the result.
