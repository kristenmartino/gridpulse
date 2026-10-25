# Forecast evidence audit — 2026-10-25

The audit used bounded, read-only public API and GCS reads. The public capture
was recorded at 2026-10-25 18:29 UTC. It describes these samples, not all regions
or every scoring run. No models were retrained, historical forecasts recreated,
or upstream readings changed during the audit.

## Measured production samples

The accompanying [JSON summary](forecast_evidence_2026-10-25.json) preserves the
public API values before the foundation fixes.

| Region | Pre-fix API headline | Forecast-start-offset 24h sample count for each exported trained model | Legacy 24h XGBoost replay |
|---|---|---:|---|
| FPL | Seasonal baseline; selection reason recorded | 123 | 30 entries, last saved 2026-04-19 16:36 UTC |
| PJM | Ensemble | 120 | 30 entries, last saved 2026-04-19 16:38 UTC |
| LGEE | Ensemble | 129 | GCS replay prefix matched no objects |
| ERCOT | Ensemble | 120 | Not sampled |

The baseline diagnostic was not exported by the pre-fix drift API. The FPL
trained-model counts in this table do not describe its published baseline.
No sampled count establishes complete seven-day coverage. Current FPL/PJM/LGEE
demand-vintage parquet files each contained 719 records with target timestamps
through 2026-10-25 17:00 UTC. Their schema contains target hour, first observed
demand, first observation time, latest demand, revision count, earliest observed
operator forecast, and its capture time. It does not establish an original
operator issue time or final actual-demand status.

The sampled 24h ensemble diagnostic reported repeated-origin skip events of
27 (FPL), 30 (PJM), and 31 (LGEE), with zero expired-unresolved and malformed
events in those blocks. These counters do not establish the cause of every
missing target and are not summed into a fabricated complete loss breakdown.
The existing coverage investigation remains open.

Raw audit captures are retained as local review evidence outside the repository
in the task's `gridpulse-evidence-foundation` review directory. The committed
summary contains public fields only.

## Code findings and decisions

1. **Source divergence.** `jobs/phases.py` writes the primary model into
   `predicted_demand_mw`, with explicit seasonal-baseline substitution. The
   dashboard uses that column. The API preferred `ensemble` merely because the
   comparison column existed. Regression fixtures reproduce the mismatch.
   The API now uses the same published-series resolver as the dashboard;
   explicit comparisons remain separately named.
2. **Clock interpretation.** `snapshot_horizon_predictions` chooses
   `first_target + H`, not `generated_at + H`. Existing diagnostic labels cannot
   establish exact issue-to-target leads. The foundation labels this as a
   forecast-start offset without changing forecast values or grade bands.
   `scored_at` identifies generation/scoring time; it is not the original
   operator publication time or the time a user refreshed the page.
3. **Incomplete scoring evidence.** Public `n_7d` counts were available for
   trained models, but baseline records were omitted. The UI also accepted
   `n_records` as a substitute for an absent seven-day count. The fix exports
   baseline diagnostics, preserves unknown window counts, and reports actual
   unique scored targets and observed dates against the full seven-day hourly
   window. This denominator is not a claim the series existed throughout it.
   Future target hours and the left boundary are excluded from completed
   rolling windows. Existing loss counters may overlap and are not a complete
   additive explanation of every unscored hour.
4. **Baseline decision scope.** The selection policy compares the ensemble's
   rolling diagnostic with a seasonal-naive backcast over recent demand.
   Their target sets are not paired. This is a policy decision, not a matched
   benchmark proving the baseline superior to the published primary model.
   The foundation identifies the comparison series and basis prospectively,
   exposes the recorded reason, and leaves onset time unknown. Selection
   thresholds and forecast-serving policy are unchanged.
5. **Legacy replay is not a current production archive.** Replay snapshots
   are saved by a chart callback, stamped with that save time, deduplicated by
   prediction values, and bounded to 30 entries per combination. The sampled
   objects are old or absent. They cannot be relabeled as current, originally
   published forecast issues. They remain a separate legacy feature.
6. **Prospective receipt contract.** The scoring job prepares an identity
   from the published series, generation clock, target range, and source
   metadata. Only after a successful required Redis write does it create an
   immutable GCS receipt and read-cache entry. Failed publication cannot create
   an archival receipt. Retry writes preserve the first receipt, and concurrent
   read-index updates use Redis WATCH transactions.

## Pilot, retention, and failure handling

The initial publication pilot is FPL, PJM, and LGEE, representing a sampled
baseline, a regional model forecast, and the region in the existing coverage
investigation. These are audit candidates, not validated customer selections.
The setting is `FORECAST_PUBLICATION_REGIONS`; an empty value disables capture.

The archive captures the full available published target array, respecting
existing horizon guards. It does not include a preferred model's alternate
values or recreate unavailable history. Only the published series is archived
initially. The public archive explains that long horizons may use climatology.

GCS objects live under a new `publications/v1/{region}/{generation-date}/{id}`
prefix and use create-only writes. Redis holds issue bodies with a 31-day TTL
and a rolling 30-day index capped at 720 issues per region. Those are read-cache
limits, not a promise of complete capture or physical GCS deletion after 30 days.
GCS lifecycle cleanup and a broad historical restoration procedure remain
separate operational work; existing replay/vintage objects are not deleted.

Archive failures are logged and leave the live forecast-serving decision intact.
An identity in a live forecast is not a guarantee that its archival receipt is
currently retrievable. Archive reads use Redis only; web requests do not fetch
upstream feeds, create forecasts, or repair missing history. Empty indexes and
unavailable records return explicit states.

Before widening the pilot, measure production object sizes, index/cache use,
archive failures, and the `publication_archive` scoring substep. The audit does
not claim a measured archive cost or runtime impact before those writes occur.

## Historical-review gate

The sampled legacy files cannot supply a current operating-day review with
verified publication provenance. The implementation therefore establishes
prospective capture before introducing that workflow. A whole-day review must
use one eligible issue covering the day; later issues must not be stitched into
an apparently earlier prediction. First-seen/latest actuals must be explicitly
versioned when they are used in a review.

Practitioner usefulness and business impact remain unvalidated. This audit
establishes data and code contracts, not operational adoption.

## Benchmark wording follow-through

The benchmark page/API and methodology now describe `last_d` as the latest
observed EIA actual, with revisions possible. Earliest-observed operator
forecasts are not asserted to be original issues. Existing calculation fields
and exclusion keys remain compatible. The benchmark coverage explanation also
keeps the missing-hour investigation open instead of asserting that every
shortfall is a permanent feed-timing ceiling.

## First production capture verification — 2026-10-25

The foundation deployed at `a6898358`, followed by benchmark wording and
Models diagnostic-table containment at `56fb150f`. The latter was verified on
ready revision `gridpulse-00582-p2g` with all serving traffic; both job definitions
matched that release. The original captures remained readable across this web
revision change.

A verification execution, `gridpulse-scoring-job-qmkhl`, ran the foundation
image and completed at 19:16:51 UTC. Its terminal event reported 51 regions
scored, zero failures, and zero forecast errors. All three pilot issues
preserved 720 target hours; each issue's first 24 values and generation/source
metadata matched the live forecast with that same publication ID. The stored
GCS contents also passed the internal content-identity check.

| Region | Published series | Generated (UTC) | GCS object bytes |
|---|---|---|---:|
| FPL | Seasonal baseline | 19:10:29 | 45,874 |
| PJM | XGBoost | 19:10:31 | 49,193 |
| LGEE | XGBoost | 19:13:47 | 51,906 |

The run's `publication_archive` substeps reported three calls, 0.9 seconds
combined, with a 0.4-second maximum. These are first-run observations, not a
representative performance or cost estimate. The public indexes initially held
one issue each; they do not yet establish a historical operating-day record.
Production coverage totals reconciled to the 168-hour window, including the
FPL baseline's 12 scored targets. Missing-hour causes remain under investigation.
