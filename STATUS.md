<!--
How this file gets maintained:
- Per-PR: updated in the same commit as material work that changes
  active focus, next-3, blocked-on, or recent decisions
- End-of-session: agent re-verifies against gh issue list / gh pr list
- Pre-external-use: user re-reads top-to-bottom (~1 min)
If this file disagrees with gh, the live sources win — patch in a
follow-up commit.
-->

# Status — updated 2026-05-29

> Canonical pointer for "where am I, what's next." This file +
> [GitHub Projects board](https://github.com/users/kristenmartino/projects/1)
> + the issue tracker are the single source of truth for project state.
> See [`docs/internal/NEXT_UP.md`](docs/internal/NEXT_UP.md) for the full
> historical roadmap; see [`CLAUDE.md`](CLAUDE.md) for the pre-session
> sanity-check ritual.

## Active focus + open question

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

1. **Phase 3 of `prod-readiness`** (production safety, ~7h): [#149](https://github.com/kristenmartino/gridpulse/issues/149) strict prod fallback gating in `model_service` — **in flight** (PR-G4: `get_forecasts`/`get_model_metrics` return unavailable/`{}` under `REQUIRE_REDIS`, no simulated/hardcoded values; surfaced follow-up [#166](https://github.com/kristenmartino/gridpulse/issues/166) `write_diagnostics` uses the simulated path), [#155](https://github.com/kristenmartino/gridpulse/issues/155) LDWP/drift robust sMAPE for low-demand regions (closes [#142](https://github.com/kristenmartino/gridpulse/issues/142) — the live ~200% drift number is a *currently-visible* blemish), [#150](https://github.com/kristenmartino/gridpulse/issues/150) Prophet interval honesty (the 95% is heuristic).
2. **Two manual one-clicks from this session** (yours, ~2 min): (a) verify the Cloud Monitoring email channel — `gcloud beta monitoring channels describe ...7265334362271951327 --format='value(verificationStatus)'` should read `VERIFIED`; (b) the `/health` deep-degraded uptime alert is a documented G10 follow-up worth adding (would catch a #161-style outage where infra is healthy but forecasts are absent).
3. **Watch live drift** (passive). Confirm `gridpulse:drift:{region}.rolling_mape_7d` healthy for top regions on the Overview tab.

**Queued behind those:**

- **Phase 4 of `prod-readiness`** — engineering rigor ([#151](https://github.com/kristenmartino/gridpulse/issues/151) deps, [#152](https://github.com/kristenmartino/gridpulse/issues/152) mypy, [#153](https://github.com/kristenmartino/gridpulse/issues/153) typed Redis payloads, [#154](https://github.com/kristenmartino/gridpulse/issues/154) callbacks.py decomposition).
- **[#164](https://github.com/kristenmartino/gridpulse/issues/164)** — drop archive-unstable weather vars (wind_80m/120m, soil_temp) + retrain, IF a feature-importance ablation shows they're deadweight. P0 #161 follow-up, low priority.
- **[#121](https://github.com/kristenmartino/gridpulse/issues/121) part 3 — Ensemble weight integration** (`path-b`, timing-gated).
- **PR-C2** (`PITCH.md` + expanded STAR stories) — parked unless interview cycle demands it.

**`prod-readiness` Phase 1 + Phase 2 COMPLETE** (2026-05-29). Phase 1: #156/#157/#158. Phase 2: PR-G2 deploy-gating (#146 → PR #159), PR-G3 deep /health (#147 → PR #160), PR-G10 job-failure alerting (#148 → PR #165) — all merged + prod-verified. **P0 #161 fully resolved**: mitigation (A, #162) + proper fix (C, #163, archive ERA5 stitch) both deployed + prod-verified; historical weather coverage ~0 → 14/17 real vars, `/health?deep=1` healthy. Job-failure alerting now live in Cloud Monitoring (no more manual incident discovery). Remaining campaign: **Phase 3 (#149 / #150 / #155)** + **Phase 4 (#151 / #152 / #153 / #154)**.

**The production-readiness campaign keeps proving its own value:** PR-G3's deep `/health` (shipped 2026-05-29) caught a total forecast outage on its first production run — invisible to the `curl / → 200` check it replaced. Strongest STAR story in the set.

## Blocked / waiting on

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
