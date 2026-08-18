# Drift lead-hours re-grade: what the fix moves, and why the new number is right

**Issue:** [#542](https://github.com/kristenmartino/gridpulse/issues/542).
**All figures measured against production, captured 2026-08-18T07:06:24Z.**
Drift numbers move hourly; every figure here is anchored to that capture.

---

## 1. The defect

`models.drift.regrade_records` rebuilds a `DriftRecord` when EIA revises an
hour's actual. It dropped `lead_hours` on the rebuild.

`lead_hours` is how far ahead the prediction actually reached.
`filter_by_lead` uses it to hold the "1-hour-ahead" headline to lead ≤ 1 — and
it **keeps** unknown-lead records, deliberately, because those were assumed to
be records that predate the field. Every revision therefore moved one more
observation past a filter built to exclude it.

The asymmetry that made the omission read as intentional sits two lines above
it: `smape` is explicitly reset to a NaN sentinel so `__post_init__` recomputes
it. That is correct — sMAPE is *derived* from the pair that just changed.
`lead_hours` is a property of the *observation*, which a revision to the actual
cannot touch. One had to be recomputed and one had to be carried; the code did
the first and forgot the second.

### How far it had progressed

Ensemble block, share of records carrying no lead:

| scope | unknown-lead | of | share |
|---|---:|---:|---:|
| whole retained window | 29,857 | 36,087 | **82.7%** |
| inside the 30-day window | 27,056 | 33,286 | **81.3%** |
| inside the 7-day window | 4,285 | 7,747 | **55.3%** |

The spread across BAs is the diagnostic, not the fleet number. Inside the 7-day
window PACW reads 162 of 163 unknown and FPC 161 of 163, while PJM reads 3 of
150 and ERCOT 5 of 138. Unknown-lead share tracks each BA's revision rate,
which is the signature of re-grading — pre-field history would decay uniformly
with age, not sort itself by how often a feed revises.

`filter_by_lead`'s docstring claimed the filter self-heals to
`n_unknown_kept == 0` once pre-field records age out. That claim was
**conditional on nothing else blanking the field**, and this defect broke the
condition: unknowns were manufactured faster than they aged out, so the counter
converged *upward*. The docstring is corrected in the same change.

---

## 2. Method

The code fix cannot repair history. A record already blanked has no recoverable
lead *inside the pipeline* — the forecast payload that would prove it is
overwritten the next tick. So on deploy day every published figure moves by
0.00, and convergence runs over 7 days (7d stats) and 30 days (30d stats).

That makes a naive before/after worthless as merge evidence, so the leads were
reconstructed from outside the pipeline.

**Instrument.** `jobs.phases.write_drift_metrics` has logged `drift_updated`
with `region`, `new_record_ts` and `lead_hours` on every tick since
2026-08-05 ([#407](https://github.com/kristenmartino/gridpulse/pull/407)). Lead
is a property of the (region, target hour) pair and not of the model, so that
log rebuilds exactly the map `regrade_records` erased. 33,445 entries swept
across 31 days.

**Counterfactual.** For each (region, model), both arms are computed at the
*same* `now_iso` — that block's own `last_updated_at`, so the window boundary
cannot differ between them — through the real `models.drift` primitives, never
a reimplementation:

* **BEFORE** — the records exactly as stored, leads erased.
* **AFTER** — the same records with recovered leads joined on.

**Recovery coverage.**

| window | blanked records | recovered | share |
|---|---:|---:|---:|
| 7d | 4,285 | 4,285 | **100.0%** |
| 30d | 27,056 | 8,130 | 30.0% |

Every one of the 18,926 unrecovered 30-day records has a target hour before
2026-08-05, when the field first shipped. Those predate the **field**, not the
blanking: they were written with no lead, and `filter_by_lead` was always
designed to keep them. So the 7-day counterfactual is exact, and the 30-day one
is correct by construction rather than merely conservative.

**Independence check, run before anything downstream was believed.** BEFORE
must reproduce what the payload publishes, or the harness is measuring itself:

* `n_7d` matches the published value on **204 of 204** (BA, model) blocks.
* `rolling_mape_7d` matches exactly on **200 of 204**. The other four (LDWP
  xgboost/prophet/arima, LGEE xgboost) differ in the sixth decimal — the
  serializer's 4dp rounding of `e` re-entering the mean, not a disagreement.

---

## 3. Direction — and why the new number is the right one

**The headline moves in both directions.** Ensemble, 7-day window: 17 BAs
improve, 20 get worse, 14 do not move. I expected a uniform improvement (error
grows with lead, so removing the multi-hour tail should lower the mean) and
that expectation was wrong. AZPS goes 9.779 → **11.427** and PSCO 9.639 →
**10.766**; both now publish a worse number than before the fix.

That two-sided movement is the argument. A filter that only ever flattered the
product would deserve suspicion; this one reports whatever the lead-1
population actually says. What the old figure was doing is not "over-reporting
error" or "under-reporting" it — it was averaging observations from a
population the window does not claim to cover. LGEE's window carried leads out
to **63 hours** under a label reading "1-hour-ahead" (§5). Removing them can
move the mean either way, depending on whether that BA's long leads happened to
be better or worse than its genuine nowcasts.

**The denominator must be read next to the numerator.** Fleet-wide `n_7d` falls
from 7,627 to 7,363 for every model, and the per-BA drops are very uneven:
LGEE 135 → 80, PSCO 154 → 120, SPA 119 → 82. A BA whose sMAPE barely moves
while its sample count falls by a third has not been unaffected — it has had a
third of its window reclassified as measuring something else.

**Fleet effect is small; the tail is not.** Ensemble 7d moves 3.706 → 3.630
(−0.076 pts). 21 of 408 (BA, model, window) blocks move more than 1.0 pt, and
they concentrate in six BAs.

---

## 4. Consequences that reach a user

**The visibility gate.** `components/_callbacks_models.py` and
`_callbacks_overview.py` both gate on `n_7d >= 24`. **LDWP crosses it on all
four models** (25 → 15): its Models panel row becomes "Warming" and the
Overview headline falls back to training holdout MAPE. That is the correct
outcome — LDWP's published 12.085% currently rests on 25 records of which 10
were never 1-hour-ahead, and a 15-record window is below the threshold this
project already decided was the minimum defensible one. No other block crosses
in either direction, at either window.

**`/api/v1/drift` `live_1h`** moves as tabulated in §6. The export allow-list
(`_EXPORTED_LIVE_DRIFT_FIELDS`) is unchanged; `n_lead_excluded_7d` and
`n_lead_unknown_7d` remain internal, per the
[#250](https://github.com/kristenmartino/gridpulse/issues/250) review, and are
now emitted on the `drift_updated` log line instead.

**`serve_grade` on `/benchmark` does NOT move — measured, not argued.**
Probed across production: **442,537 horizon records over 51 BAs, of which 0
carry a lead.** `resolve_horizon_snapshots` never sets the field and
`_horizon_rollup_block` never lead-filters (24/48/72h records have a *designed*
horizon, so filtering them to lead 1 would empty them — pinned by a test since
#273). Both halves are now pinned: the producer by a new test, the consumer by
the existing one. Every 24/48/72h figure, and therefore every `serve_grade`
marker on `web/benchmark.html`, is byte-identical across this change.

**`reconcile.py` Check A had to be fixed in the same PR.** It recomputed
settled MAPE with the low-actual filter but **not** the lead filter, then
diffed the result against the lead-filtered displayed figure and fired A1 above
2.0 pts. While leads were being blanked the mismatch was invisible — with
almost nothing to filter, the two populations agreed by accident. Repairing the
lead field parts them:

| settled arm | A1 firings, 204 blocks |
|---|---:|
| as shipped, before the lead repair | 1 |
| as shipped, after the lead repair | **12** |
| with the lead filter mirrored | **0** |

All 12 were differences in *which hours were scored*, not divergence from
settled demand — the worst being LDWP arima at an 8.5 pt gap. The module's
"independently re-implement" rule protects against inheriting the producer's
notion of a *window* or an *aggregate*; it was never a licence to grade a
different *population*. `RECONCILE_MAX_LEAD_HOURS` mirrors
`HEADLINE_LEAD_HOURS` on the same threshold-spec footing as
`RECONCILE_LOW_ACTUAL_FRACTION`.

*Approximation, stated:* the settled arm above uses each record's own `actual`
rather than a GCS parquet read, on the grounds that
[#304](https://github.com/kristenmartino/gridpulse/issues/304) re-grading
already converges it to EIA's settled view. The 12 → 0 result is far enough
from the threshold that a parquet read would not change the decision.

---

## 5. Relationship to #537 — shared phenomenon, different defect

[#537](https://github.com/kristenmartino/gridpulse/issues/537) (horizon-drift
7-day window short fleet-wide; LGEE loses about half) was cross-linked to this
issue as possibly sharing a mechanism. It does — one level up from the code.

**They share an upstream phenomenon and no code.** A forecast's origin is
`last_real_demand_hour + 1`, so it does not advance one hour per wall-clock
tick. When it stalls:

* the 1-hour path still grades, at a **growing lead** — the contamination #542
  is about; and
* the horizon path recomputes the **same** `(target_ts, horizon)` every tick,
  the `seen` dedup correctly drops it, and that hour is never snapshotted —
  the missing records #537 is about.

One cause, two symptoms, two unrelated code sites. **Fixing #542 does not fix
#537**, and the horizon path is untouched by this change (§4).

**Reconstructing LGEE's forecast origin from the lead log settles it.** Origin
= `target_hour − lead + 1`:

* 2026-08-13T15:00 → 2026-08-14T05:00 — origin frozen at 2026-08-13T14:00 for
  **15 consecutive ticks**, leads climbing 2 → 16.
* 2026-08-14T06:00 → 2026-08-15T05:00 — origin **regressed** to
  2026-08-12T15:00 (23 hours *older* than the previous tick's) and froze there
  for **24 consecutive ticks**, leads climbing 40 → **63**.

Across the fleet the correlation is direct — ticks spent on a frozen origin
against horizon coverage (168 hours available):

| BA | scoring ticks in 7d | ticks on a frozen origin | max lead | horizon 24h `n_7d` |
|---|---:|---:|---:|---:|
| LGEE | 140 | **44** | **63** | **82** |
| SPA | 126 | 18 | 7 | 109 |
| PSCO | 154 | 15 | 4 | 140 |
| CAISO | 146 | 5 | 9 | 150 |
| SRP | 164 | 1 | 3 | 165 |
| TAL / CPLW / FMPP | 164–165 | **0** | 2 | **166** |

**`_expire_pending` is ruled out**, contrary to the standing hypothesis. LGEE's
pending buffer holds 117 snapshots against PJM's 126 and ERCOT's 120 — it is
not accumulating, and during the regressed-origin freeze the 24h target lands
in the *past* and resolves immediately rather than ageing toward the 120-hour
cutoff. The fleet-wide shortfall of 1–17 hours is explained by ordinary missed
ticks plus the leads > 1 tail (SRP: 164 ticks, 3 skipped hours, 165 of 168).

**What is left open for #537** is the layer above: *why* LGEE's forecast
payload froze for 15 ticks and then served a 23-hour-older vintage for another
24. That is a scoring-path question, not a drift-module one, and it is reported
on the issue rather than fixed here.

---

## 6. Tables

`b→a` is BEFORE → AFTER. sMAPE is the headline metric; MAPE and both `n`
columns are published alongside it. Captured 2026-08-18T07:06:24Z.

## A. Per-BA movement — ensemble, both windows

| BA | 7d sMAPE b→a | Δ | 7d n b→a | 30d sMAPE b→a | Δ | 30d n b→a |
|---|---|---:|---|---|---:|---|
| LDWP | 12.085 → 8.231 | -3.854 | 25 → 15 | 5.463 → 4.991 | -0.472 | 314 → 297 |
| IID | 16.148 → 13.643 | -2.504 | 145 → 135 | 16.004 → 14.189 | -1.814 | 576 → 548 |
| AZPS | 9.779 → 11.427 | +1.648 | 41 → 31 | 7.734 → 8.660 | +0.927 | 87 → 67 |
| PSCO | 9.639 → 10.766 | +1.127 | 154 → 120 | 11.344 → 11.849 | +0.505 | 643 → 584 |
| LGEE | 3.117 → 2.435 | -0.682 | 135 → 80 | 2.560 → 2.416 | -0.144 | 630 → 562 |
| SPA | 24.347 → 24.630 | +0.283 | 119 → 82 | 25.240 → 25.040 | -0.201 | 512 → 451 |
| SCL | 3.852 → 3.948 | +0.096 | 163 → 156 | 4.467 → 4.498 | +0.031 | 685 → 675 |
| BPAT | 8.047 → 8.100 | +0.053 | 162 → 157 | 6.963 → 6.948 | -0.015 | 683 → 674 |
| WALC | 7.006 → 7.051 | +0.045 | 162 → 159 | 5.688 → 5.690 | +0.002 | 686 → 682 |
| NEVP | 3.780 → 3.739 | -0.041 | 159 → 151 | 3.723 → 3.714 | -0.009 | 666 → 654 |
| SRP | 3.194 → 3.159 | -0.035 | 165 → 163 | 3.395 → 3.370 | -0.025 | 688 → 685 |
| DUK | 1.273 → 1.243 | -0.031 | 159 → 155 | 1.136 → 1.132 | -0.003 | 675 → 667 |
| SOCO | 2.926 → 2.957 | +0.031 | 163 → 161 | 2.303 → 2.309 | +0.007 | 685 → 680 |
| TVA | 2.896 → 2.869 | -0.027 | 161 → 155 | 3.510 → 3.515 | +0.005 | 683 → 674 |
| PACE | 3.227 → 3.200 | -0.027 | 163 → 159 | 2.480 → 2.473 | -0.007 | 687 → 679 |
| SEC | 12.858 → 12.878 | +0.020 | 150 → 149 | 15.170 → 15.216 | +0.046 | 673 → 669 |
| SPP | 1.864 → 1.844 | -0.020 | 137 → 136 | 2.095 → 2.090 | -0.004 | 660 → 658 |
| SCEG | 2.449 → 2.465 | +0.016 | 165 → 163 | 2.984 → 2.985 | +0.000 | 689 → 685 |
| FPL | 1.816 → 1.829 | +0.014 | 147 → 137 | 1.452 → 1.449 | -0.003 | 671 → 659 |
| MISO | 2.257 → 2.244 | -0.013 | 148 → 134 | 2.397 → 2.402 | +0.006 | 671 → 654 |
| PSEI | 1.433 → 1.445 | +0.012 | 165 → 163 | 2.250 → 2.258 | +0.008 | 689 → 685 |
| CAISO | 1.623 → 1.634 | +0.011 | 134 → 129 | 1.991 → 1.995 | +0.004 | 652 → 646 |
| CPLE | 1.152 → 1.141 | -0.011 | 159 → 154 | 1.452 → 1.452 | -0.001 | 681 → 673 |
| FPC | 2.077 → 2.087 | +0.009 | 164 → 161 | 2.376 → 2.382 | +0.006 | 687 → 682 |
| PACW | 3.672 → 3.663 | -0.009 | 164 → 161 | 3.630 → 3.631 | +0.002 | 685 → 678 |
| PGE | 0.732 → 0.739 | +0.007 | 165 → 163 | 1.165 → 1.165 | +0.000 | 686 → 682 |
| TEC | 1.924 → 1.931 | +0.007 | 162 → 161 | 2.125 → 2.128 | +0.003 | 683 → 681 |
| BANC | 2.460 → 2.466 | +0.006 | 165 → 163 | 2.402 → 2.404 | +0.001 | 688 → 684 |
| AVA | 1.187 → 1.181 | -0.005 | 162 → 161 | 1.666 → 1.661 | -0.005 | 684 → 681 |
| GCPD | 1.149 → 1.143 | -0.005 | 164 → 163 | 1.461 → 1.456 | -0.006 | 688 → 685 |
| IPCO | 1.711 → 1.707 | -0.004 | 162 → 159 | 1.800 → 1.782 | -0.018 | 684 → 677 |
| FMPP | 3.745 → 3.748 | +0.003 | 165 → 163 | 3.069 → 3.063 | -0.006 | 686 → 681 |
| NWMT | 1.143 → 1.146 | +0.003 | 162 → 159 | 1.901 → 1.913 | +0.012 | 678 → 671 |
| JEA | 2.982 → 2.982 | +0.000 | 104 → 104 | 2.183 → 2.174 | -0.008 | 626 → 624 |
| TEPC | 3.399 → 3.399 | +0.000 | 161 → 161 | 2.400 → 2.394 | -0.006 | 678 → 676 |
| TIDC | 1.459 → 1.459 | +0.000 | 116 → 116 | 1.948 → 1.942 | -0.006 | 499 → 498 |

Unmoved on both windows (15): AECI, CHPD, CPLW, DOPD, EPE, ERCOT, GVL, HST, ISONE, NYISO, PJM, PNM, SC, TAL, TPWR

## B. Fleet roll-up — BA-mean, 51 BAs

| model | window | sMAPE before | sMAPE after | Δ | Σn before | Σn after | BAs moved |
|---|---|---:|---:|---:|---:|---:|---:|
| ensemble | 7d | 3.706 | 3.630 | -0.076 | 7627 | 7363 | 37/51 |
| ensemble | 30d | 3.665 | 3.641 | -0.024 | 33091 | 32635 | 50/51 |
| xgboost | 7d | 3.479 | 3.464 | -0.015 | 7627 | 7363 | 37/51 |
| xgboost | 30d | 3.642 | 3.631 | -0.011 | 33091 | 32635 | 50/51 |
| prophet | 7d | 6.403 | 6.409 | +0.006 | 7627 | 7363 | 37/51 |
| prophet | 30d | 6.177 | 6.184 | +0.006 | 33091 | 32635 | 50/51 |
| arima | 7d | 4.677 | 4.416 | -0.262 | 7627 | 7363 | 37/51 |
| arima | 30d | 4.296 | 4.249 | -0.047 | 33091 | 32635 | 50/51 |

## C. Blocks moving more than 1.0 sMAPE pt (21 of 408)

| BA | model | window | sMAPE b→a | Δ | n b→a |
|---|---|---|---|---:|---|
| LDWP | arima | 7d | 28.980 → 17.848 | -11.132 | 25 → 15 |
| LDWP | ensemble | 7d | 12.085 → 8.231 | -3.854 | 25 → 15 |
| LGEE | arima | 7d | 9.944 → 6.636 | -3.308 | 135 → 80 |
| IID | xgboost | 7d | 15.849 → 12.686 | -3.163 | 145 → 135 |
| LGEE | prophet | 7d | 6.852 → 4.324 | -2.527 | 135 → 80 |
| IID | ensemble | 7d | 16.148 → 13.643 | -2.504 | 145 → 135 |
| IID | xgboost | 30d | 17.474 → 15.264 | -2.211 | 576 → 548 |
| AZPS | xgboost | 7d | 10.239 → 12.290 | +2.050 | 41 → 31 |
| IID | ensemble | 30d | 16.004 → 14.189 | -1.814 | 576 → 548 |
| AZPS | arima | 7d | 9.663 → 11.455 | +1.792 | 41 → 31 |
| IID | arima | 7d | 23.424 → 21.742 | -1.682 | 145 → 135 |
| AZPS | ensemble | 7d | 9.779 → 11.427 | +1.648 | 41 → 31 |
| AZPS | prophet | 7d | 11.851 → 13.395 | +1.544 | 41 → 31 |
| SPA | prophet | 7d | 29.743 → 31.167 | +1.424 | 119 → 82 |
| LDWP | arima | 30d | 7.615 → 6.240 | -1.375 | 314 → 297 |
| IID | arima | 30d | 24.060 → 22.724 | -1.336 | 576 → 548 |
| PSCO | xgboost | 7d | 10.465 → 11.797 | +1.331 | 154 → 120 |
| AZPS | xgboost | 30d | 9.707 → 10.954 | +1.247 | 87 → 67 |
| PSCO | arima | 7d | 11.169 → 12.367 | +1.198 | 154 → 120 |
| AZPS | arima | 30d | 8.226 → 9.372 | +1.146 | 87 → 67 |
| PSCO | ensemble | 7d | 9.639 → 10.766 | +1.127 | 154 → 120 |

---

## 7. Backfill

Production self-heals: the 7-day headline recovers in 7 days and the 30-day
figure in 30, as blanked records age out. `scripts/backfill_drift_leads.py`
narrows that wait for the six BAs above the 1.0 pt bar or crossing the
visibility gate — LDWP, IID, AZPS, PSCO, LGEE, SPA — writing 1,098 recovered
lead values additively and recomputing through `compute_drift_payload`. The
other 45 BAs move by less than 0.1 pt and are left to heal on their own.

It must run **after** the fix is deployed: against the old image the next
hourly tick re-blanks every lead it writes. The script is dry-run unless the
GCS data artifact says otherwise, so the reviewed code is byte-identical
between rehearsal and real run. The AFTER column above is the prediction the
live payload is checked against once it lands.

### Running the backfill

The body is passed inline to `python -c`, because `scripts/` is in
`.dockerignore` and is not in the job image. `--args` splits on commas, which
the source contains, so use gcloud's custom-delimiter prefix:

```bash
cd <repo> && SRC=$(cat scripts/backfill_drift_leads.py) && gcloud run jobs execute gridpulse-scoring-job --region us-east1 --args="^%^-c%$SRC"
```

**The delimiter must be a character the source does not contain**, which is why
the command lives here and not in the script's own docstring — documenting a
delimiter inside the file being split puts that character into the file. Two job
executions were burned learning this: first with a pipe (the docstring showed a
pipe), then with a tilde (the docstring then showed a tilde). Both surfaced as
`SyntaxError: unterminated triple-quoted string literal`, which reads like a
broken script rather than a shredded argument. Check the split locally first:

```bash
python3 -c "import sys;src=open(sys.argv[1]).read();d=chr(37)*2;print(chr(10).join(['SPLIT OK'] if d not in src else ['DELIMITER PRESENT IN SOURCE']))" scripts/backfill_drift_leads.py
```

Dry-run is the default; the apply switch lives in the GCS artifact.

### Post-deploy confirmation (2026-08-18T08:28Z)

Merged `71a60cc`, deployed, and verified **inside the running container** rather
than by tag: a lead-6 record re-graded against a changed actual came back with
`lead_hours=6` (`FIX_LIVE=True`). Ancestry was checked rather than assumed — a
concurrent merge (`f826c9f`) superseded this one's deploy run, and `71a60cc` is
an ancestor of it, so a green workflow alone would not have been evidence.

Backfill applied to the six BAs, then re-read independently:

| BA | predicted AFTER (07:06Z) | live after backfill (08:28Z) | `n_7d` | `unk7d` | `excl7d` |
|---|---:|---:|---:|---:|---:|
| LDWP | 8.231 | **8.231** | 15 | 0 | 12 |
| PSCO | 10.766 | **10.766** | 120 | 0 | 33 |
| IID | 13.643 | 14.702 | 135 | 0 | 13 |
| AZPS | 11.427 | 11.793 | 30 | 0 | 10 |
| LGEE | 2.435 | 2.454 | 80 | 1 | 59 |
| SPA | 24.630 | 26.005 | 82 | 0 | 38 |

LDWP and PSCO reproduce the counterfactual **exactly**; they graded no new
records in the intervening 1.4 hours. The other four moved because their
windows advanced, which is the expected behaviour of a time-bounded statistic
and the reason every figure here carries a timestamp. `n_lead_unknown_7d`
reached **0** on five BAs and 1 on LGEE — one record whose lead the log could
not supply.

The GCS artifact was returned to `apply: false` after the run, so an accidental
re-execution is a dry run.
