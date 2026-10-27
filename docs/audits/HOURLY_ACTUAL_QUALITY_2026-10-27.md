# Retained-hour actual quality

## Production finding

PSCO's first hourly 24h target is October 26, 2026 at 21:00 UTC. Its raw demand
was revised from a zero placeholder to 25 MW and then 598 MW. At 598 MW it
continued to fail the configured day-ahead-ratio and below-normal-scale signals.
Nevertheless, the 04:00 UTC scoring run included it without a further source
correction.

A direct replay of that production demand frame reproduced the cause. The
frame had 2,164 rows, with six rows after this target. The default six-row
forecast-input guard no longer examined it. The same function, with a retained
window span, still rejected 598 MW against the same 3,535 MW day-ahead value.
Aging was incorrectly treated as actual eligibility. A nonzero score count
therefore did not prove that usable demand had arrived.

The read-only ledger capture was taken at 04:26:40 UTC across all 51 regions.
Its [machine-readable audit](hourly_actual_quality_2026-10-27.json) includes
source-frame hashes, object update times, retained clocks, and per-region
recheck results. The recheck found two already-scored observations flagged by
the configured signals: PSCO's unchanged 598 MW and a revised SPA value
(11 MW stored, 4 MW in the source frame). Only the PSCO case establishes the
unchanged-value aging defect. Source revisions can also change eligibility.
These flags are heuristic decisions, not certification of physical truth.

## Repair

The hourly scorer keeps a separate reference to the raw demand frame before
the short input guard replaces the model-input frame. Its actuals helper omits
future rows, retains a preceding day of context, and applies the same quality
signals throughout the seven-day scoreable target window. The forecast-input
guard's span and thresholds, model inputs, serving policy, legacy diagnostics,
and benchmark calculations are unchanged.

The ledger records an active actual-quality exclusion and the last rejection
value, reason, and clocks. Flagged observations do not enter MAPE, including
observations scored before this correction. Their stored actual and first
observation evidence remain intact, as do every original prediction and
sampling clock. A later source revision that passes the same checks clears the
active exclusion. An absent reading does not clear it. On upgrade, an old
stored actual absent from the current source remains pending until it can be
rechecked; its presence in the ledger does not establish eligibility.

API summaries identify `actual_quality_policy = retained_window_guard_v1`.
Coverage adds `quality_guard_excluded_hours` to its mutually exclusive
168-hour partition, separate from pending demand and low-actual exclusions.
The public API continues to omit raw sample arrays. The UI describes quality
exclusions separately; its existing six-observation display threshold does not
establish representative coverage.

## Verification and limits

Regression tests reproduce aging past the short tail, future DF-only rows,
previously scored observations becoming excluded, valid revisions clearing a
flag, preservation of original forecasts and first-observation evidence, and
phase handoff of the raw scoring frame without changing model inputs. The
initial focused quality tests passed 59 checks; integration, API and context
tests passed 121 checks. The local full suite then passed 4,212 tests with three
skips. The final affected quality, scoring, API and context suites passed 175 tests,
including the additional fail-closed upgrade case. CI must verify the final commit. Production terminal writes remain a separate release gate.

The offline 51-region recheck preserved prediction and first-observation
provenance. Local quality-helper CPU time totaled 1.5263 seconds, with a
0.1295-second maximum region. This is a bounded local measurement, not Cloud
Run elapsed impact or a forecast-accuracy improvement.

Production verification must read all 51 new policy summaries, inspect exact
retained target records and quality flags, reconcile the coverage partitions,
and check the scoring run's terminal write results. PSCO's first target must
remain excluded while its reading still fails these signals. Historical review
remains on hold until the coverage and publication-evidence prerequisites pass.
