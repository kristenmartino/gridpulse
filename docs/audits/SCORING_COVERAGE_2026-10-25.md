# Scoring coverage: missing target hours

## Measured production evidence

The read-only state probe captured all 51 regions at
2026-10-25 19:56:29 UTC. The accompanying
[machine-readable reconciliation](scoring_coverage_2026-10-25.json) reproduces
the stored seven-day ensemble counts from their retained records. It joins
27,447 origin-resolution, origin-regression, and region-completion log entries
from October 15 through the capture. All successful completions have matching
origin logs; the log range covers the required origins for every horizon.

Each region completed in all 168 hourly slots of its reporting window. Counts
below partition 8,568 region/target hours for each horizon:

| Legacy offset | Scored | No accepted matching origin | Awaiting usable actual | Low-actual exclusion |
|---|---:|---:|---:|---:|
| 24h | 6,681 | 1,833 | 54 | 0 |
| 48h | 5,886 | 2,625 | 56 | 1 |
| 72h | 5,810 | 2,701 | 56 | 1 |

No missing sample has an unexplained accepted origin in this capture. For
24h, 1,822 of the 1,833 absent targets have a positive latest-observed demand
value in the separate vintage mirror. That observation does not establish
that the quality-guarded scoring frame accepted the same value at every tick.
The exported seven-day expiry and malformed counters were zero for all regions.
SPA, at 82 scored 24h targets, is now the lowest-covered region; LGEE has 129.
The August rankings must not be reused as current state.

## Why this happens

The legacy sampler selects exactly `first_forecast_target + H`. Forecast starts
follow available demand and feature history, rather than advancing once per
wall-clock hour. A jump skips a target's only eligible origin; a stall repeats
an existing target. Successful hourly jobs therefore do not guarantee hourly
target samples. This is not primarily an actual-demand expiry problem in the
measured window. The older feature-origin investigation remains relevant to
forecast inputs, but changing trained-model inputs is not required to repair
this sampling design.

The missing old values cannot be reconstructed honestly: a different row from
a later forecast would change the observation. Historical offsets remain
available as a separate diagnostic and benchmark input.

## Repair contract

`models/hourly_scoring.py` observes the cached forecast before replacement and
selects the target at the next UTC hour plus H. An observation at 10:05 selects
11:00 + H, giving a sampling-to-target lead of at least H and less than H+1
hours. The original generation clock and optional publication identity are
retained separately. This is an hourly observation policy, not a claim to have
recovered the operator's issue time or the latest forecast at an exact cutoff.

The new `drift_hourly:{region}` ledger is separate from `drift_horizon:{region}`:

- First observation for a target/horizon wins, including concurrent retries.
- Forecast values are never synthesized for earlier unobserved slots.
- Missing actuals do not prevent future forecast capture. Unresolved targets
  remain for the full seven-day reporting window, so observations arriving
  after the legacy 120-hour pending limit can still be scored.
- Withheld/non-finite model values are not sampled. Baseline observations use
  the published column when a baseline is served.
- Actual revisions retain the original prediction, sampling clock, and first
  observed actual alongside the latest observed actual.
- Coverage partitions every reporting-window hour into scored, before
  collection, missing sample, unavailable series, pending actual, or excluded
  low actual. Startup is not presented as a scoring failure.
- A strict Redis WATCH transaction prevents concurrent overwrites and commits
  the ledger and its small web/API summary together. Failed or
  malformed state reads fail the phase rather than rebuilding history. The
  eleven-day key TTL covers the seven-day reporting window plus the longest
  future horizon.

The Overview/Forecast panel uses hourly scores only after six matching new
observations exist. Until then, it identifies the shown score as legacy and
displays collection progress. The Models view keeps the old offset table in a
separate expander. The six-observation rule is an existing display policy,
not evidence of statistical sufficiency. `/api/v1/drift/{region}` exports the
new summaries under `hourly_sampled`, without exporting raw sample arrays.
The benchmark continues using its documented legacy sample; the histories are
not pooled and their error values are not claimed to be directly comparable.

Two related defects are also repaired: resolved legacy targets can no longer
be resampled with later predictions, and a failed horizon-history write can
no longer report phase success. The legacy phase also captures future points
when the current actual-demand frame is empty.

## Verification and resource bounds

The synthetic ten-day continuity test freezes and jumps the input origin while
hourly observation and actual availability continue. It produces 168 scored
targets at every horizon with zero missing sampling slots. These are synthetic
values, not a measured accuracy improvement or reconstructed production history.
Other tests cover delayed/revised actuals, first-sample preservation, WATCH
conflicts, withheld values, unknown clocks, separate UI/API histories, and
failure reporting.

A synthetic mature ledger with five series retained 651 samples per region.
Across the real 51-region fan-out, serialized state totaled 20,355,051 bytes;
three sequential local update passes took 2.506, 1.9756, and 1.926 seconds.
This measures local CPU/storage behavior, not Cloud Run elapsed impact. A
read-only production Redis probe measured 229,496,424 used bytes against a
1,073,741,824-byte limit before this addition; Redis allocator overhead is
additional to serialized JSON size. Real phase timings and writes must still
be checked after deployment.

Production capture can be verified immediately. Observed scores require the
future target hours and their actuals to arrive; a complete seven-day score
cannot be claimed on deployment day. Historical operating-day review remains
out of scope until this coverage work is verified and adequate original
publication history is available.

Reproduce the dated audit locally with the retained read-only capture:

```sh
PYTHONPATH=. python scripts/audit_scoring_coverage.py \
  --state /path/to/raw-state.json.gz \
  --logs /path/to/origin-and-scoring-logs-10d.json \
  --output /path/to/reconciled.json
```
