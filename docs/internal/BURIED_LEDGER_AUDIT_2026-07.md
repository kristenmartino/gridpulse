# Buried-ledger audit — 2026-07-09

A fan-out re-verification of the 2026-07 critical review (`docs/internal/CRITICAL_REVIEW_2026-07.md`) against **current** code, to answer: which P2/ledger findings are *still live* and were *never surfaced* as a standalone issue (they were folded under the #189 umbrella and forgotten). 7 agents audited 80 findings; 51 survived as still-live + untracked. Each was assigned a tracking issue below.

| Tier | Tracking |
|---|---|
| 6 critical integrity/silent-failure | individual issues #267–#272 |
| ~15 misleading numbers on live surfaces | #273 |
| 7 medium backend reliability/correctness | #274 |
| 23 doc/config honesty (ledger) | #275 |


## HIGH (6)

### P2-01 — Scoring job reports success when only 1 of 51 regions actually scored  → #267 · 👁 user-visible
**Why:** A run where the forecast phase failed for all 51 BAs still exits 0, refreshes last_scored, and never trips the failed-execution alert, so ~50 regions can serve stale or no forecasts while health reports green. It masks exactly the failure class the alerting was built for.
**Evidence:** jobs/scoring_job.py:280 marks a region ok if ANY phase ran (`summary["ok"]=any(p.get("ok")...)`); :400 `return 0 if ok_count>0 else 1`; :377-384 writes last_scored unconditionally; health.py:107-113/:129-133 only checks age and samples FPL; regions_failed is written but nothing consumes it.
**Fix:** Redefine region-ok as 'forecast phase ok AND its Redis write ok'; fail the run (or emit a distinct alertable signal) below a threshold ok-fraction; write last_scored only from successful regions; have deep-health inspect regions_failed. File a standalone issue.

### P2-07 — HTTP-200 responses that parse to zero demand rows poison the 24h cache and blank the demand surface  → #270 · 👁 user-visible
**Why:** The fallback decision is made on the raw record list before parsing, so a 200 carrying only day-ahead (type-DF) rows yields a 0-row demand frame that is cached for 24h; later calls serve the poisoned empty frame without ever consulting the stale/GCS last-known-good, and the demand surface goes blank.
**Evidence:** data/eia_client.py:148 guards the pre-parse list (`if not all_records:`); _parse_demand_records filters type=="D" (:510) so a DF-only 200 yields 0 rows; :162 caches the empty frame (cache.set has no empty guard); :141-143 later serves the cached empty frame directly.
**Fix:** Move the fallback decision AFTER parsing (empty parsed frame -> stale->GCS chain) and never cache an empty frame over prior real data. File a standalone issue.

### P2-06 — Mid-pagination truncation is cached 24h and overwrites the GCS last-known-good  → #269 · 👁 user-visible
**Why:** When a page hard-fails partway through pagination the fetch returns partial records; the only guard is whether the list is empty, so a truncated frame is cached for 24h AND written over latest.parquet, corrupting the durable fallback during a partial outage. The server-reported total is available but never checked.
**Evidence:** data/eia_client.py:404-405 `if response is None: break` returns accumulated records; :409 reads `total` but only for the stop condition; :148 guard is only `if not all_records:`; :162 cache.set + :165 write_parquet then overwrite GCS.
**Fix:** Compare accumulated row count against the server-reported total; on shortfall raise into the stale->GCS fallback or mark the frame incomplete and skip the cache/GCS writes. Adjacent to #174 but not covered; file a standalone issue.

### P2-11 — One transient GCS blip negative-caches the entire model store ({} sentinel) for the run/process lifetime  → #272 · 👁 user-visible
**Why:** A single failed latest.json read caches an empty dict and returns it to every later caller with no retry: on the scoring job all 51x3 models become unloadable for the whole run (warming/degraded UI), and on the web tier it blanks the quality gate and meta layers until container restart. A momentary hiccup cascades into total model-store unavailability instead of self-healing.
**Evidence:** models/persistence.py:222-225 stores `_latest_cache={}` on ANY exception in _read_latest; :201-202 returns that pinned {} to every caller with no retry/TTL; no production code passes force=True or invalidates it.
**Fix:** Cache only successful reads (never the failure sentinel); add a short TTL or retry-on-empty so a blip self-heals within the run. Needs standalone tracking (#183 adjacent but does not name it).

### P2-10 — Forecast quality gate fails open, live-reads GCS in the request path, and pins a stale pointer for process lifetime  → #271 · 👁 user-visible
**Why:** The gate decides which BAs appear in the region dropdown and US-Grid view. It fails OPEN (any Redis/GCS exception returns None, which means pass), so during an outage it silently stops hiding rollback-grade regions and users can select BAs whose forecasts are unusable; it also pins latest.json for the whole process so it never reflects daily retraining, and on cold Redis it falls through to per-render GCS meta reads in the request path. #255/#260 only changed which MAPE it reads.
**Evidence:** models/model_service.py:371-387 get_best_holdout_mape returns None on exception, :412-416 None->pass; persistence.py:201-202 pins _latest_cache for process lifetime; components/layout.py:72-81 and _callbacks_us_grid.py:82-83,:128 sweep the gate per render.
**Fix:** Publish per-region gate status to a gridpulse:* Redis key from the scoring job; make the dropdown and US-Grid read it; make gate evaluation fail closed (or render a visible degraded state); drop the process-lifetime pointer pin. Give it a standalone issue (currently buried only under umbrella #189).

### P2-03 — Redis failure is structurally invisible to the scoring job  → #268
**Why:** A Redis blip at job start pins a failed client for the whole process, so an entire scoring tick can write nothing to Redis yet exit 0 and log scoring_job_complete; redis_set failures are swallowed into a discarded bool, so no phase or the run summary reflects that nothing persisted - the invisibility that lets P2-01 report success on an empty run.
**Evidence:** data/redis_client.py:18,25-28 `_redis_init_attempted` never reset (a failed first ping pins _redis_client=None for the process); :96-106 redis_set swallows all exceptions and returns False; :109-111 redis_available() is stale both directions; jobs/phases.py:224,235 discard the redis_set return and PhaseResult(ok=True) regardless.
**Fix:** Allow Redis re-init after failure (or TTL the lockout), propagate redis_set failures into PhaseResult ok-flags, and count Redis-write failures in the run summary that P2-01's exit policy consumes. File a standalone issue.


## MEDIUM (22)

### P2-29 — Risk tab presents invented constants as measured data across all 51 BAs  → #273 · 👁 user-visible
**Why:** The production Risk tab renders unsourced event 'Severity Scores', region-agnostic temperature-exceedance lines, and an unprovenanced stress-score formula as if they were measured grid data, identically for every one of 51 balancing authorities. This is a fabricated-number surface shown to every user.
**Evidence:** components/_callbacks_alerts.py:234-237 hardcodes severity scores (Uri=95, CA Heat=80, Heat Dome=85, Eclipse=40) on a 0-100 axis; :221 draws fixed 95/100/105F exceedance lines applied to every region; the stress score `min(100, n_crit*30+n_warn*15+20)` at :474-475; all in the Redis prod fast path. Distinct from #193's fabricated-alert cluster.
**Fix:** Parameterize thresholds per region from climatology/config, source or drop the severity scores, and document or replace the stress-score weights.

### P2-23 — 'Net Load (avg)' hero KPI silently falls back to average TOTAL generation  → #273 · 👁 user-visible
**Why:** When demand JSON is missing or fewer than 2 timestamps align, the KPI renders total generation (all fuels) under the 'Net Load (avg)' label, overstating net load by the entire wind+solar share with no visible degraded state.
**Evidence:** components/_callbacks_overview.py:1063 seeds `net_load_avg=avg_total`; the true net-load compute at :1074-1075 overwrites only when demand parses and len(common)>=2; the exception path :1076-1077 swallows failures; the label stays 'Net Load (avg)' at :1082 regardless.
**Fix:** On alignment/parse failure render an explicit degraded state or relabel 'Total Generation' instead of substituting a differently-defined quantity under the net-load label.

### P2-26 — Forecast fast path charts one model's series under another model's name  → #273 · 👁 user-visible
**Why:** When XGBoost is missing from a payload that still carries a primary series, the Forecast tab plots the primary (possibly Prophet) but titles and legends it 'XGBOOST Forecast' - the same badge-vs-data class as the PR #130 c2d6c20 bug, via a still-live mechanism with a real trigger.
**Evidence:** components/_callbacks_forecast.py:753 exempts xgboost from the payload-miss check; :758-759 falls back to `predicted_demand_mw`; trace name :801 and title :867 still say `{model_name.upper()} Forecast`. jobs/phases.py:1001-1006 sets predicted_demand_mw to xgboost-or-first-successful, :1015-1016 writes per-model keys only for successful models.
**Fix:** Remove the xgboost exemption (return None on model miss like every other model) or relabel the trace/title by the key actually plotted.

### P2-21 — Overview headline 'live 7d' MAPE can display from a single in-window observation  → #273 · 👁 user-visible
**Why:** The Overview headline shows a rolling-7d live MAPE users read as 'expected accuracy right now', but it can be computed from as few as 1-2 in-window samples, so it swings wildly while looking authoritative, and the intended thin-window->30d fallback can never fire because both branches gate on the same total record count.
**Evidence:** components/_callbacks_overview.py:204,219,221 gate 7d and 30d on `n_records=ens.get("n_records")`, which models/drift.py:497 sets to `len(merged)` (total rolling records up to 720), not the 7d-window count; the payload exposes no per-window sample count.
**Fix:** Have write_drift_metrics emit per-window sample counts (n_7d, n_30d) and gate the 7d headline / 30d fallback on those, not on total n_records.

### P2-15+ledger-22 — XGBoost CV MAPE is optimistically biased and silently substituted for the missing holdout, mislabeled as holdout  → #273 · 👁 user-visible
**Why:** Each fold's reported MAPE is scored on the same validation set that drove early stopping (selection-on-test, optimistic bias), while the persisted artifact is a different full fit no fold scored; when the holdout metric is missing, this biased cv_mape is promoted to persisted meta.mape on a claimed 'consistent metric basis' and feeds both inverse-MAPE ensemble weighting and the 22% quality gate - a too-good number that also skews weights and region visibility.
**Evidence:** models/xgboost_model.py:87-92 fits with `eval_set=[(X_val,y_val)]` then scores MAPE on the same X_val; :96-97 a separate full fit; jobs/training_job.py:366 cv_mape=mean(cv_scores); :371 `saved_mape=(holdout or {}).get("mape") or cv_mape` with the 'consistent metric basis' comment at :359-361.
**Fix:** Split an inner early-stopping eval set from the scored fold (or fix n_estimators from CV and refit); keep holdout and CV as distinct labeled quantities, never silently substitute CV MAPE into holdout-basis weighting, and persist which protocol produced meta.mape.

### P2-16 — Partial holdout failure: persisted ensemble metric describes a composition the served ensemble never uses  → #273 · 👁 user-visible
**Why:** When one model's holdout returns None but its full fit still saves, training persists an inverse-MAPE ensemble metric over the SURVIVING models only, while scoring serves an EQUAL-weight blend over ALL loaded models, so the displayed 'ensemble holdout' MAPE reports a formula and membership the served ensemble does not actually use.
**Evidence:** jobs/training_job.py:307-330 (_ensemble_holdout_metrics) filters to valid (non-None) models and inverse-MAPE-weights; jobs/phases.py:979-985 falls back to equal weights over every predicting model when mape_input is incomplete; jobs/scoring_job.py:202,212 passes possibly-None meta.mape through. #176/#179 handled the all-models crash, not this partial-failure divergence.
**Fix:** Make training and scoring share one membership/weight rule - either skip unweighted models in both, or persist the ensemble metric only for the exact composition scoring will serve.

### P2-17 — A single unsmoothed 168h holdout drives both ensemble weights and the 22% visibility gate, with documented run-to-run flap  → #273 · 👁 user-visible
**Why:** One 168h window with no CV or smoothing feeds compute_ensemble_weights AND the 22% rollback gate; because a BA's best-base MAPE can swing across 22% run-to-run (review documents AZPS 11.90%->26.68%), region visibility itself flaps - a BA appears in the dropdown one day and vanishes the next.
**Evidence:** jobs/training_job.py:40 `_HOLDOUT_HOURS=168` (single window, no smoothing); config.py:703 7d rollback threshold=22.0; the same metric feeds both weights and the gate. #181 (the ensemble-weighting issue) is CLOSED, so neither the framing nor the gate-flap delta is tracked.
**Fix:** Require a smoothed/rolling holdout so both the ensemble weights and the visibility gate consume a stabilized metric; file a standalone issue since #181 is closed.

### ledger-23 — Published ensemble numbers fit inverse-MAPE weights in-sample on the scored window  → #273 · 👁 user-visible
**Why:** Both the standalone backtest script and the training job fit ensemble weights on the same window they then score, while production applies previous-window weights forward - a directionally favorable bias baked into every published ensemble figure.
**Evidence:** scripts/backtest.py:173-185 computes weights from validation MAPE then scores the ensemble on that same window; jobs/training_job.py:325-330 computes inverse-MAPE weights from the fold's mape_scores then scores on the same forecasts. [EXT #181, CLOSED].
**Fix:** Fit weights on a leading window and score strictly out-of-sample to remove the in-sample bias from published numbers.

### P2-19 — The '1-hour-ahead' drift signal mixes lead horizons and permanently drops catch-up hours  → #273 · 👁 user-visible
**Why:** After an EIA publishing catch-up, build_drift_records records only the single most-recent matchable hour (possibly an N-hour-lead prediction) into the same rolling series as true 1h-lead records, and the skipped hours can never re-match, so they are permanently dropped - the 'live 7d' drift MAPE shown on Overview/Models is computed from a mix of lead horizons and a thinner-than-intended sample.
**Evidence:** models/drift.py:348-370 iterates actuals reverse-chronologically, picks the first timestamp present in the previous forecast, builds records for that hour only and returns; DriftRecord (:363-368) has no lead-hours field.
**Fix:** Record all matchable hours from the previous payload (each carries a known lead) and add a lead-hours field to DriftRecord so the 7d statistic can filter or stratify by lead.

### P2-14 — Future is_holiday is group-mean imputed from (hour,dow), never set to 1 inside the horizon  → #273 · 👁 user-visible
**Why:** Within the 720h horizon the models never see a real holiday flag, so holiday load drops (July 4, Thanksgiving, etc.) are systematically over-forecast - affecting both XGBoost (beyond the recursive window) and Prophet (is_holiday is a regressor) - producing a visibly wrong forecast on holidays.
**Evidence:** jobs/phases.py:635-643 builds only hour/dow/month/day_of_year/sin/cos/is_weekend; :651-666 everything else (including is_holiday) is filled from `hist.groupby([_hour,_dow]).mean()`, near-zero on holiday hours; compute_holiday_flag exists at data/feature_engineering.py:473 but is never called for the future frame.
**Fix:** Compute is_holiday (and other calendar-derivable features) directly from the future timestamps via compute_holiday_flag in _build_future_feature_frame instead of the group-mean imputer.

### P2-08 — Generation/interchange parsers coerce EIA nulls to 0.0 MW, contradicting the demand parser in the same file  → #273 · 👁 user-visible
**Why:** Missing fuel/interchange observations render as a real 0 MW in fuel-mix, renewable-share, and net-load views, silently understating generation and distorting net load, while the demand parser in the same file deliberately maps null/0 to NaN; unit tests lock the wrong behavior in.
**Evidence:** data/eia_client.py:535 `float(r.get("value",0) or 0)` (generation) and :552 (interchange) vs :489-495 demand parser coercing null/''/0 to NaN; tests/unit/test_eia_client.py:218 and :261 assert 0.0.
**Fix:** Apply the demand parser's null->NaN policy to both parsers (with downstream NaN handling in write_generation) and update the tests that pin the zero-fill.

### ledger-3 — Docs still claim the ensemble 'almost always beats' / 'can never be worse' / 'self-correcting'  → #273 · 👁 user-visible
**Why:** Multiple docs and a code header still assert the ensemble almost always wins, can never be worse than the worst model, and is self-correcting - directly contradicted by the project's own current measurement (ensemble trails the best base model in aggregate, median 4.82% vs 4.12%).
**Evidence:** models/ensemble.py:7; docs/HOW_IT_WORKS.md:163; docs/CANONICAL_FACTS.md:117; README.md:39 - contradicted by CANONICAL_FACTS.md:90-91 and BACKTEST_RESULTS.md:70-72. ADR-004 was corrected; these four remain. [EXT #181, CLOSED].
**Fix:** Reword the four remaining doc/code instances to match the honest distribution already in CANONICAL_FACTS/BACKTEST_RESULTS.

### P2-35 — Permanent forecast-failure regions show 'warming up - forecast will appear shortly' forever  → #273 · 👁 user-visible
**Why:** A region whose model never trains (e.g. a BA permanently hidden by the quality gate, so the scoring job never writes its forecast key) shows the Forecast tab a soft 'Pipeline is warming up - forecast will appear shortly' indefinitely - a transient message that becomes a standing false promise, never escalating to an honest 'unavailable'.
**Evidence:** components/_callbacks_forecast.py:359-370 cache-miss returns {'error':'warming'...} with no age input or escalation; :1227-1237 hard-codes the 'warming up shortly' text; error_handling.py:326-361 warming_state() has no age/escalation param. #188 item 5 is cosmetic-only; #189 umbrella does not count.
**Fix:** Thread the payload's last_scored/scored_at age (and whether a forecast key ever existed for the region) into the warming decision and escalate to a distinct 'Forecast unavailable for this region' past a threshold. File a standalone issue.

### P2-44 — Risk tab implemented twice: production and dev paths show different designs AND different stress-score definitions  → #273 · 👁 user-visible
**Why:** Production users see a different Risk-tab design and a different stress-score semantic than the dev path: prod renders the legacy emoji breakdown with the job's alert-count stress, while dev renders v2 gp-stress-row markup with a demand-over-capacity stress - the two have diverged further than the review described.
**Evidence:** components/_callbacks_alerts.py:66-323 (prod Redis path) reads cached stress_score (:81) and emoji breakdown (:151,158,165); the dev fallback :471-546 uses gp-stress-row + grid_stress(region,demand) (:484-494), differing from the job's min(100,n_crit*30+n_warn*15+20) (jobs/phases.py:1588,1597).
**Fix:** Extract one render function both paths call and make the job-computed stress the single source the UI reads (or push grid_stress into the job so both agree). File a standalone issue.

### P2-42 — Keyboard-shortcut map activates the wrong tabs in the shipped 5-tab shell; Models has no shortcut  → #273 · 👁 user-visible
**Why:** A user-facing accessibility bug: the JS clicks tabs positionally against a stale 4-tab map, so after US Grid's insertion at position 2 the shortcuts are off by one (Alt+2->US Grid, Alt+3->Forecast, Alt+4->Risk) and Models (Alt+5) has no handler; a dead 8-tab Python map compounds the rot.
**Evidence:** components/accessibility.py:152-161 dead 8-tab TAB_KEY_MAP + :165 'Alt+1..8'; assets/accessibility.js declares a 4-tab map but clicks visibleLinks[key-1]; live order layout.py:189-193 is [Overview, US Grid, Forecast(tab-outlook), Risk(tab-alerts), Models].
**Fix:** Drive the JS from a single 5-tab map keyed by tab_id (click the pill whose id matches, not by index); add the Models shortcut; delete or rewire the dead Python map. File a standalone issue.

### P2-02 — Training job exits 0 when most regions in a task fail, defeating Cloud Run retry  → #274
**Why:** A training task where 16 of 17 regions fail still exits 0, so Cloud Run sees success and never retries - defeating the per-region resume logic built to make retries cheap; those regions silently keep serving stale models with no alert.
**Evidence:** jobs/training_job.py:815 `return 0 if ok_count>0 else 1`; :772-780 wraps _train_region in try/except that swallows every per-region exception with no re-raise; :797 writes regions_failed but no alert consumer reads it.
**Fix:** Apply the same threshold-based exit policy as P2-01 and add a regions_failed-driven alert so a mostly-failed training task triggers the Cloud Run retry the resume logic supports. File a standalone issue.

### P2-04 — latest.json pointer-race exhaustion is reported as save success  → #274
**Why:** Under parallel training the optimistic-concurrency retry loop can exhaust its budget; when it does, save_model still logs model_saved and returns the version, so the training summary reports success while latest.json points at yesterday's model and the ensemble-metrics blob is orphaned.
**Evidence:** models/persistence.py:231 _write_latest returns None on every path; :296-304 on exhaustion logs model_latest_race_exhausted and returns None; no backoff/jitter (:259-295); :396 discards _write_latest, :398-406 returns version unconditionally.
**Fix:** Return a success/failure result from _write_latest, add jittered backoff, and make save_model propagate pointer-write failure into the training summary and P2-02's exit policy. File a standalone issue.

### P2-05 — GCS parquet backups ride unjoined daemon threads killed at job exit  → #274
**Why:** Each write_parquet upload runs on a fire-and-forget daemon thread with no join or atexit; scoring_job.run() returns right after the region loop, so in-flight tail-region latest.parquet refreshes are killed with zero signal, and the only failure log (gcs_write_failed) has no alert consumer - so the #174 outage-fallback-of-record ages silently until the next EIA outage exposes it.
**Evidence:** data/gcs_store.py:112 `threading.Thread(target=_upload,daemon=True).start()` with no future tracking; no atexit/join in jobs/ or gcs_store.py; :110 gcs_write_failed the sole signal; jobs/scoring_job.py:400 run() returns with no upload join.
**Fix:** Track upload futures and join them (bounded) before job exit, and surface upload-failure counts in the run summary. File a standalone issue.

### P2-48 — Unrecognized ENVIRONMENT value silently fails open to full development defaults  → #274
**Why:** A single operator typo in ENVIRONMENT (e.g. 'prod', 'Production', trailing space) silently selects the development tier - flipping require_redis=False, demo=True, gcs_enabled=False - removing every honesty gate at once, so a prod web tier could serve simulated forecasts and demo alerts with no degradation signal, and there is no import-time validation or warning.
**Evidence:** config.py:76 `_env=_ENV_DEFAULTS.get(ENVIRONMENT,_ENV_DEFAULTS["development"])`; repo-wide grep finds no membership check; the unknown-flag warning at ~:873 is scoped to feature flags only. Not covered by #187 (DRY/doc only).
**Fix:** Validate ENVIRONMENT against {development,staging,production} at import - fail hard on an unrecognized value, or at minimum log loudly and default to the STRICTEST tier (production), never development. File a standalone issue.

### P2-18 — ensemble_combine propagates NaN and its bounds self-check is dead for the NaN case; Prophet output not finite-guarded  → #274
**Why:** ensemble_combine has no non-finite handling and its min/max invariant silently passes on NaN (NaN comparisons are always False), so the guard cannot catch the one case it exists for; _predict_one finite-guards ARIMA but not Prophet, so a NaN Prophet yhat would reach the served payload and Python's json emits a non-standard NaN token that can break strict parsers. Latent today but unguarded.
**Evidence:** models/ensemble.py:104-107 weighted sum with no finite filter; :108-115 out_of_bounds compares against NaN min/max so the count stays 0; jobs/phases.py:840-841 returns Prophet preds with no isfinite guard vs the ARIMA guard at :862.
**Fix:** Add a non-finite guard in ensemble_combine (drop or renormalize over finite members and warn loudly) and finite-guard the Prophet branch of _predict_one like ARIMA.

### ledger-20 — config claims rollback-grade models are 'auto-disabled' - no such mechanism exists  → #274
**Why:** config comments assert an automatic model-disable/fallback governance mechanism that is not implemented anywhere; rollback-grade models keep contributing to the served ensemble while only region visibility is gated.
**Evidence:** config.py:687 'Governance: models exceeding ROLLBACK threshold are auto-disabled.' and :692 'model disabled, fallback to next-best'; the four MAPE_THRESHOLD_* constants (:689-692) have no non-test consumer in the main tree.
**Fix:** Either implement disable-and-fallback or reword the comments to describe the actual visibility-gate-only behavior.

### P2-47+ledger-18 — Merit-order pricing has ungrounded constants and drops ~28.6% as demand rises through 90% of capacity  → #274
**Why:** Price falls as demand rises across the u=0.90 tier boundary - physically backwards and contradicting the module's own 'exponential spike' docstring - and the PRICING_* constants carry no provenance or 'illustrative' label; latent today (Scenarios surface dormant) but ships live with the planned P1 Scenarios polish.
**Evidence:** models/pricing.py:47-59: linear tier ends at 1.4x base just below u=0.90, exponential tier restarts at base*exp(0)=1.0x at u=0.90 (a ~28.6% drop) then *EMERGENCY above; config.py:669-670 tiers; the only caller estimate_price_impact via simulation/scenario_engine.py:123-124 is dormant.
**Fix:** Anchor the exponential tier to the linear endpoint (continuity at 1.4x base), label the model illustrative, and add a tier-boundary monotonicity property test before Scenarios goes live. File a standalone issue.


## LOW (23)

### P2-22 — '7d Peak/Low/Average' KPIs use last-168-nonzero-rows, not a 7-day window, and disagree with the hero chart  → #275 · 👁 user-visible
**Why:** During EIA publishing gaps the KPI window silently stretches past 7 calendar days while the hero chart above uses a different frame, so a peak shown in '7d Peak' may not appear on the chart labeled the same 7 days.
**Evidence:** components/_callbacks_overview.py:272-282 filters zeros then `nonzero.tail(168)` (labels at :363,:371,:379 assert 'last 168h (7 days)') while the hero chart at :408 uses raw `df.tail(168)`; neither windows by timestamp.
**Fix:** Window both KPI and chart by timestamp (>= last_ts - 7d) and share one frame.

### ledger-5 — Coverage claim contradicts itself: ~100% vs ~80% vs ~99% (incl. a user-visible caption)  → #275 · 👁 user-visible
**Why:** The flagship coverage number is stated three mutually-contradicting ways, including a user-visible '51 (~80% of demand)' caption, with no derivation artifact.
**Evidence:** docs/CANONICAL_FACTS.md:18 '~100%' (+~81% BA-count row :20); components/_callbacks_us_grid.py:935 caption '51 (~80% of demand)'; docs/internal/NEXT_UP.md:184 '~99%' (plus ~98%/~85%/~94% elsewhere).
**Fix:** Pick one derived figure, document its derivation in CANONICAL_FACTS, and reconcile the us_grid caption and NEXT_UP references to it.

### ledger-11 — HOW_IT_WORKS scoring runtime self-contradicts (~5 min vs ~14 min) and mis-states horizon/model counts  → #275 · 👁 user-visible
**Why:** The same document states two different scoring runtimes and horizon/model counts, undermining trust in the operational description.
**Evidence:** docs/HOW_IT_WORKS.md:73 '~5 min' scoring + '5-hour timeouts' contradicts :125 '~14 minutes'; :182 '4 horizons x 4 models' vs the UI's 3 forecast horizons; CANONICAL_FACTS.md:39 anchors ~855s (~14 min).
**Fix:** Reconcile HOW_IT_WORKS to the measured ~14 min scoring runtime and the real horizon/model counts.

### ledger-9 — Test count published as '1,589' and '1681', disagreeing with each other and the tree  → #275 · 👁 user-visible
**Why:** The canonical registry built to prevent drift carries a stale test count that disagrees with README and with the actual collected count.
**Evidence:** docs/CANONICAL_FACTS.md:58 '1,589 passing as of #119'; README.md:134 '1681 tests'; review measured 1981 at its SHA - at least one is stale.
**Fix:** Recompute via `pytest --collect-only -q`, update both, and cite the same source.

### ledger-8 — README says training retrains 'on the last 60 days' (actually ~90)  → #275 · 👁 user-visible
**Why:** The public data-flow description understates the training window versus the actual ~90-day fetch used everywhere else in the docs and code.
**Evidence:** README.md:79 reads 'retraining each region's models on the last 60 days'; the traced default is ~90 days (matching HOW_IT_WORKS/TECHNICAL_SPEC and published train_rows).
**Fix:** Change README.md:79 to ~90 days.

### ledger-16 — Governance/calibration constants ungrounded; TECHNICAL_SPEC publishes a competing grading scale  → #275 · 👁 user-visible
**Why:** User-facing grading and drift/cap constants have no derivation, and the spec publishes a second, incompatible grading scale for the same decision.
**Evidence:** config.py:698 MAPE_BY_HORIZON (used by mape_grade :722) has no provenance; models/arima_model.py:139 `abs(drift)>np.std(y)*0.5`; models/prophet_model.py:106 `demand_cap=max*1.5`; TECHNICAL_SPEC.md:358-360 publishes a competing flat scale (<3%/3-5%/5-10%, no rollback tier).
**Fix:** Add one-line provenance comments to the gating constants and delete or reconcile the competing TECHNICAL_SPEC scale.

### ledger-26 — Models tab presents training-holdout metrics table and live diagnostics charts as one evaluation  → #275 · 👁 user-visible
**Why:** The metrics table (168h training holdout) and the residual/diagnostics charts directly below it (a different, live-scoring evaluation) are shown together with no badge or caption distinguishing their provenances, and the table itself mixes meta.json MAPE with Redis-payload RMSE/MAE/R2.
**Evidence:** components/_callbacks_models.py:144 table metrics from get_model_metrics (holdout), :146-147 charts from the gridpulse:diagnostics payload; comment :139-141 notes the mix; tab_models.py:192 carries only a generic 'most recent week of holdout data' note. [EXT #166, CLOSED].
**Fix:** Add a provenance caption/badge to each panel distinguishing training-holdout metrics from live diagnostics.

### P2-12 — auto_arima order selection silently ignores the weather exogenous matrix (wrong pmdarima 2.x kwarg)  → #275
**Why:** SARIMAX orders are selected against a no-exog model while the final fit includes exog, so the (p,d,q)(P,D,Q) order is chosen sub-optimally for the weather-driven demand problem - bounded impact (ARIMA is one of three, final fit still uses exog) but a silently mis-configured shipped model.
**Evidence:** models/arima_model.py:387-389 `pm.auto_arima(y_sub, exogenous=exog_sub,...)`; `exogenous` was removed in pmdarima 2.x (should be `X=`), requirements.lock:42 pins pmdarima==2.1.1, so the kwarg falls into **fit_args and every candidate is built with exog=None.
**Fix:** Pass `X=exog_sub` per the pmdarima 2.x API and add a unit test asserting order selection actually consumes the exog matrix.

### P2-20 — Scenario engine zeroes temperature_deviation for any constant temperature override (test-locked)  → #275
**Why:** A constant-temperature override (the typical 'heat wave to 112F' scenario) makes temperature_deviation identically 0, because it is recomputed as the overridden series minus its OWN rolling mean, so the 'unusual weather' feature reads exactly 0 during the most extreme scenarios and a unit test enshrines it. Limited impact today (scenario_engine not yet wired into the simulator UI).
**Evidence:** simulation/scenario_engine.py:86 broadcasts a constant; :149 recomputes via compute_temperature_deviation on that constant; data/feature_engineering.py:357-372 returns temp - rolling(720).mean() = 0 for constant input; tests/unit/test_scenario_extended.py:176-184 asserts deviation==0.
**Fix:** Compute deviation against the pre-override baseline history's rolling mean rather than the overridden series' self-mean, and correct the test.

### P2-36 — Scoring job writes a weather-correlation payload hourly for 51 BAs that nothing reads  → #275
**Why:** Wasted per-tick compute inside the runtime-constrained scoring job (~855s ceiling): the generation payload is now genuinely consumed, but the weather-correlation phase's only reader is an unregistered/orphaned callback, so it is pure dead compute.
**Evidence:** jobs/scoring_job.py:268 write_weather_correlation runs every tick (impl jobs/phases.py:1331); its only reader components/_callbacks_weather.py:58 is an orphaned unregistered fast path; generation IS read live (_callbacks_overview.py:780 into the Forecast panel), so drop only weather-correlation.
**Fix:** Drop write_weather_correlation from the scoring loop and delete the orphaned _callbacks_weather.py fast path to reclaim job headroom; keep write_generation. Re-verify the review's 'generation is dead' premise before acting (now false).

### P2-43 — Forecast-Replay subsystem (NEXD-14) permanently inert behind a flag with a circular re-enable condition  → #275
**Why:** 422 LOC dark behind a flag whose own re-enable comment can never be satisfied - the only snapshot producer is itself gated by the same flag.
**Evidence:** config.py:839 hard-sets `"forecast_replay": False` (comment: re-enable once snapshots produce fresh data); feature_enabled defaults unknown flags to False with no env override; the only snapshot producer save_forecast_snapshot is gated at _callbacks_forecast.py:652-653; no job imports data/forecast_history.py.
**Fix:** Either move snapshot production into the scoring job (breaking the circularity) then flip the flag, or delete data/forecast_history.py + the three gated callbacks. File a standalone decision issue.

### P2-45+ledger-25 — Two more ensemble implementations in components/ with policy drift and a false parity docstring  → #275
**Why:** Two component-layer combiners use equal weights (and one excludes ARIMA beyond 168h) diverging from production's inverse-MAPE weighting, and _ensemble_fold's docstring falsely asserts backtest/production parity - actively misleading maintainers.
**Evidence:** components/_callbacks_backtest.py:436-459 _ensemble_fold returns np.mean(preds) but the docstring :438-442 claims 'consistent with production'; components/_callbacks_forecast.py:463-473 _run_forecast_outlook equal-weights and drops ARIMA >168h; production weights by inverse holdout MAPE incl ARIMA (jobs/phases.py:887-912). [#184 CLOSED without consolidating these].
**Fix:** Delete _ensemble_fold's false parity claim and align _run_forecast_outlook's membership/weighting with production (or route both through one shared combiner). Reopen/replace #184's consolidation scope for these two sites.

### ledger-14 — TECHNICAL_SPEC presents five deleted tab modules as current and lists a nonexistent is_weekend feature  → #275
**Why:** The technical spec describes deleted surfaces as live and names an engineered feature that does not exist, misleading any reader treating it as current truth.
**Evidence:** TECHNICAL_SPEC.md:438-443 and :499-503 list tab_forecast/tab_backtest/tab_generation/tab_weather/tab_simulator.py as current; :244 lists is_weekend though the real feature is is_holiday (models/prophet_model.py:40).
**Fix:** Update TECHNICAL_SPEC sections 9/12 to the 5-tab shell and correct is_weekend to is_holiday.

### ledger-24 — Extended-holdout ship criterion divides by the wrong window, biasing toward ship  → #275
**Why:** The ship/no-ship audit script's documented criterion MAPE(169-384)/MAPE(1-168) is not what the code computes; it divides by MAPE(72-168h), a typically-worse short window, systematically shrinking the ratio and biasing toward 'ship'.
**Evidence:** scripts/audit/extended_holdout_check.py:21-22 docstring says '/MAPE(1-168)<1.5'; WINDOWS :58 are (0,24),(24,72),(72,168),(168,384) with no (0,168); :129-135 `ratio=mape_long/mape_full_short`=MAPE(168-384)/MAPE(72-168).
**Fix:** Compute a true MAPE(1-168) window and divide by it, or correct the docstring to describe the 72-168 basis.

### ledger-27 — Documented gap policy (interpolate <6h, flag >=6h) is not enforced by the live pipeline  → #275
**Why:** The advertised missing-data policy has no production caller; the live pipeline just drops NaN demand rows, so the documented gap interpolation/flagging never runs.
**Evidence:** data/preprocessing.py handle_missing_values/validate_dataframe have no non-test callers (only re-exported in data/__init__.py:13,27-28); the live pipeline is `engineer_features(merged).dropna(subset=["demand_mw"])`; CLAUDE.md module map advertises the <6h/>=6h policy.
**Fix:** Either wire handle_missing_values into the live pipeline or remove the documented policy.

### ledger-28 — D2 'Forecast Model Input Audit Trail' is a memory-only ring buffer the scoring job never writes  → #275
**Why:** D2 is listed as implemented for post-event lineage / FERC-NERC defensibility, but it is a per-process in-memory 1000-record buffer with no persistence, no production query consumer, and the scoring job (which actually produces forecasts) writes no audit records at all.
**Evidence:** data/audit.py:8-9,65-67 'In-memory audit log'; grep of jobs/ finds no audit write (only a comment at jobs/phases.py:416); records are written only in the dev/inline web path (#149-strict-gated in prod); CLAUDE.md Sprint-5 presents D2 as implemented for lineage.
**Fix:** Have the scoring job persist audit records, or downscope the D2 claim to its actual in-memory dev-only behavior.

### ledger-12 — CANONICAL_FACTS confidence-interval row states the wrong window and points to the wrong file  → #275
**Why:** The canonical-facts registry misstates the interval calibration window (120h is only the floor) and points to the wrong source file, so anyone citing it propagates a wrong number and a dead pointer.
**Evidence:** docs/CANONICAL_FACTS.md:32 '80% empirical, last 120h calibration window | models/evaluation.py'; actual window is min(available, max(horizon*5,120)) = 840h at the default 7d horizon, and the logic lives at components/_callbacks_shared.py:455, not models/evaluation.py.
**Fix:** Correct the window description and the file pointer in CANONICAL_FACTS.

### ledger-17 — Prophet logistic-growth cap defaults to a hardcoded 50,000 MW  → #275
**Why:** A hardcoded 50,000 MW cap default (below the load scale of PJM/MISO/ERCOT) is used as a silent fallback with no config linkage or log line whenever a model is missing its _demand_cap attribute (e.g. a stale pickle).
**Evidence:** models/prophet_model.py:190 `demand_cap=getattr(model,"_demand_cap",50000)`; training sets _demand_cap=max*1.5 (:106,:131), so 50000 only fires for a model missing the attribute but engages silently.
**Fix:** Derive the fallback from the region's capacity table or fail loud when _demand_cap is absent.

### ledger-19 — Scenario '2024 Solar Eclipse' preset uses cloud-cover instead of irradiance attenuation, values uncited  → #275
**Why:** Preset 'real historical event' weather values are uncited flat constants and the 2024 eclipse preset models the event as 100% cloud cover rather than solar-irradiance attenuation - the wrong physical mechanism. Latent (Scenarios surface hidden under #127).
**Evidence:** simulation/presets.py:101-114 sets `cloud_cover: 100.0 # eclipse blocks sun`; values applied as flat constants across the horizon.
**Fix:** Model the eclipse via shortwave_radiation attenuation, cite the preset values, and label them approximate.

### ledger-4 — xgboost code comment claims 'tuned via autoresearch: 30 experiments, 16.4% MAPE improvement' with no artifact  → #275
**Why:** A specific tuning-improvement figure is asserted in code with no measurement artifact anywhere in the repo, predating the leakage repair.
**Evidence:** models/xgboost_model.py:23 '# Default hyperparameters (tuned via autoresearch: 30 experiments, 16.4% MAPE improvement)'; no experiment journal exists; the sole provenance is a commit message.
**Fix:** Drop the unverifiable figure or attach a reproducible measurement artifact.

### ledger-10 — PRD R3.1 claims Prophet 'multiplicative seasonality'; implementation is additive  → #275
**Why:** A Must-Have requirement asserts a modeling choice the code deliberately does not make.
**Evidence:** PRD.md:131 'Prophet with weather regressors and multiplicative seasonality'; models/prophet_model.py:63 `seasonality_mode="additive"` (:5 'Additive seasonality'); some per-regressor modes are multiplicative (:34-40) but the seasonality itself is additive. [EXT #184, CLOSED].
**Fix:** Change PRD R3.1 to 'additive seasonality with mixed-mode weather regressors'.

### P2-49 — Dockerfile bakes a dev-tier PRECOMPUTE_ENABLED=true that deploys must remember to override; PRECOMPUTE_ALL_REGIONS is a dead knob  → #275
**Why:** The image ships a dev-tier default (PRECOMPUTE_ENABLED=true) that only stays safe because each web-service deploy remembers to override it to false; any deploy surface that forgets runs the inline scoring pipeline inside the stateless web container, contradicting the read-only web-tier guardrail. PRECOMPUTE_ALL_REGIONS is baked and defined but controls nothing.
**Evidence:** Dockerfile:43-44 set both true; deploy-prod.yml:99 / deploy-dev.yml:75 override for the web service, but the four Cloud Run Job blocks do not set it (inherit baked true); config.py:762 reads PRECOMPUTE_ALL_REGIONS into a constant no other .py consumes. #187's criteria cover flag de-dup/layer-cache, not removing wrong-tier ENV from the image.
**Fix:** Remove tier-specific ENV values (PRECOMPUTE_ENABLED, PRECOMPUTE_ALL_REGIONS) from the Dockerfile and let the J1 _ENV_DEFAULTS matrix + explicit deploy overrides own them; fold this concrete instance into #187's checklist or file a standalone issue.

### P2-50 — LOG_LEVEL is a dead knob with no consumer; configure_logging applies no level filter, so every tier emits DEBUG  → #275
**Why:** LOG_LEVEL is documented in the J1 matrix (dev=DEBUG, staging=INFO, prod=WARNING) and set in deploy YAML, but nothing reads it and configure_logging installs no level filter, so debug-level events are emitted in every tier regardless - log noise, GCP logging cost, and a potential sensitive-data-in-verbose-logs exposure operators believe they have turned off.
**Evidence:** repo-wide LOG_LEVEL grep returns only config.py:18 (definition) and :26 (comment); observability.configure_logging (:20-58) uses add_log_level (annotate-only), no make_filtering_bound_logger/setLevel anywhere; deploy-dev.yml:75 sets LOG_LEVEL=DEBUG under ENVIRONMENT=staging.
**Fix:** Wire a structlog level filter in configure_logging driven by config.LOG_LEVEL (e.g. make_filtering_bound_logger(getLevelName(LOG_LEVEL))) and reconcile the four surfaces to one owner. Track separately from #187's doc reconciliation.
