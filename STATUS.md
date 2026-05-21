<!--
How this file gets maintained:
- Per-PR: updated in the same commit as material work that changes
  active focus, next-3, blocked-on, or recent decisions
- End-of-session: agent re-verifies against gh issue list / gh pr list
- Pre-external-use: user re-reads top-to-bottom (~1 min)
If this file disagrees with gh, the live sources win — patch in a
follow-up commit.
-->

# Status — updated 2026-05-21

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

1. **Watch live drift over the next ~7 days** (passive — no code work). The 2026-05-20 overnight training cycle (~04:00 UTC) was the first run with the audit fixes in place — PR-D's de-leaked features, PR-E's recursive inference, PR-C's real weather forecast, PR-B's empirical CI. Live drift MAPE in `gridpulse:drift:{region}.rolling_mape_7d` should drop notably for weather-sensitive regions (ERCOT, FPL, PJM, CAISO, MISO) as the new pickle accumulates predictions. **What to check Friday 2026-05-23:** rolling 7d MAPE on the Overview tab for top regions; Forecast-tab feature importance should no longer show `demand_roll_24h_min` in the top 5. If drift is stable or improved, the audit landed clean. If drift worsens, file an issue and investigate.
2. **#129 Forecast-tab chart 1–4h gap** (~3-4h, `bug`). EIA publishing lag visualized as empty space between actual end and forecast start. The fix is in `jobs/phases.predict_and_write_forecast`: backfill predictions for trailing NaN-demand rows so the forecast trace starts at `last_actual_demand_hour + 1h` instead of `featured.timestamp.max() + 1h`. This was the only OTHER bug found during the audit window that didn't make it into a PR. Different code path than the audit fixes.
3. **Practice the explanatory docs** (~10 min, off-keyboard). Read `HOW_IT_WORKS.md` aloud + pick one STAR story to rehearse. Satisfies criterion (b) of the 14-day success criterion. ADR-008 added to `HOW_IT_WORKS.md` §4 yesterday — practice now includes the audit narrative.

**Queued behind those:**

- **[#121](https://github.com/kristenmartino/gridpulse/issues/121) part 3 — Ensemble weight integration** (~2–3 days, `path-b`). Decision: incorporate live MAPE into ensemble weights, OR surface a stale-weights warning when holdout-vs-live diverges past threshold. **Timing-gated** — needs ~7 days of POST-AUDIT live records before the decision is data-informed. Don't start before 2026-05-28. The PR-D + PR-E changes will shift the residual distribution materially; the decision criterion needs to be re-evaluated against the new equilibrium.
- **PR-C2** (`PITCH.md` + expanded STAR stories) — parked unless interview cycle demands it. The audit work is now PRD.md §10 (ADR-008) and `INTERVIEW_PREP.md` may want a new STAR story for it.

The Next-3 is much thinner than yesterday's. The audit-driven correctness work is complete; the remaining open code work is one small bug (#129) plus passive monitoring. Good window for off-keyboard work (practice, polish) or to pick up a different project.

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
