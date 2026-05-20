<!--
How this file gets maintained:
- Per-PR: updated in the same commit as material work that changes
  active focus, next-3, blocked-on, or recent decisions
- End-of-session: agent re-verifies against gh issue list / gh pr list
- Pre-external-use: user re-reads top-to-bottom (~1 min)
If this file disagrees with gh, the live sources win — patch in a
follow-up commit.
-->

# Status — updated 2026-05-20

> Canonical pointer for "where am I, what's next." This file +
> [GitHub Projects board](https://github.com/users/kristenmartino/projects/1)
> + the issue tracker are the single source of truth for project state.
> See [`docs/internal/NEXT_UP.md`](docs/internal/NEXT_UP.md) for the full
> historical roadmap; see [`CLAUDE.md`](CLAUDE.md) for the pre-session
> sanity-check ritual.

## Active focus + open question

**Strategic position: A — Portfolio + targeted credibility investment.**
GridPulse's portfolio-grade build is shipped. Next moves polish the
recruiter-facing surface (PR-C1 in flight, PR-C2 next) AND close one
Path B item ([#121](https://github.com/kristenmartino/gridpulse/issues/121)
Model drift monitoring) as a high-quality interview asset. Path B
items beyond #121 stay deferred — see [`docs/internal/NEXT_UP.md`](docs/internal/NEXT_UP.md) §V4.

**Open question — 14-day success criterion (by 2026-06-03):** at least
2 of these must be true, or the PM infrastructure built this week is
theatrical and should be partially reverted:

- [x] (a) `docs/HOW_IT_WORKS.md` has real content (PR #125)
- [ ] (b) `docs/HOW_IT_WORKS.md` and `docs/INTERVIEW_PREP.md` have been used at least once for actual practice (read aloud + timed)
- [x] (c) [#121](https://github.com/kristenmartino/gridpulse/issues/121) has a draft PR or partial implementation (PRs [#126](https://github.com/kristenmartino/gridpulse/pull/126) backend writer + [#128](https://github.com/kristenmartino/gridpulse/pull/128) UI panel)
- [~] (d) Handoff quickstart run on another repo — **deferred** 2026-05-20. Discovery: `news-aggregator/` is a working folder with multiple version subdirs (sift_v1, v2, the-digest), not a git repo, so the quickstart can't run cleanly there. User has already set up "something similar" for sift's 3 repos independently — the cross-project validation the criterion was probing for has effectively happened, just outside this framework. Re-evaluate if/when a new project is bootstrapped from scratch.

**2 of 4 criteria satisfied (a + c) — the ≥2 "not theatrical" threshold is cleared. The PM infrastructure built this week is not theatrical.** Criterion (b) takes ~10 min of reading aloud and is yours to do off-keyboard.

## Next 3 (priority order)

1. **De-leak training features** (this PR, in flight, ~3h). Shift ``ramp_rate`` and every ``demand_roll_*`` to read from ``demand.shift(1)`` so training-time autoregressive features match the inference-time ``compute_autoregressive_snapshot`` definition. 5 new regression tests pin the leakage absence at the unit level. New training pickles flow nightly; live drift MAPE should improve over the following ~7 days as the new model accumulates predictions.
2. **PR-E — Recursive autoregressive features in production inference** (~3h, blocked on this PR landing + ≥1 training cycle). Replace climatological group-mean lag/rolling features in ``_build_future_feature_frame`` with recursive computation from recent actuals + prior predictions, mirroring the holdout-validation behavior. Closes the train/serve gap completely. Same-data parity test included.
3. **PR-C — Real weather forecast in ``_build_future_feature_frame``** (~4h, independent of training-leakage PR but best landed after). Replace (hour, dow) group-mean weather features with actual Open-Meteo forecast values. Addresses original "temperature predictions can't matter much" concern — weather will actually move the demand forecast.

**Queued behind those:**

- **PR-B — Empirical confidence interval on Overview hero chart** (~2h). Polish: replace ±3% heuristic band with same calibrated empirical residual-quantile method the Forecast tab uses. Best done LAST so the band calibrates against the post-de-leak model rather than re-calibrating after every model fix.
- **[#121](https://github.com/kristenmartino/gridpulse/issues/121) part 3 — Ensemble weight integration** (~2–3 days, `path-b`). Decision: incorporate live MAPE into ensemble weights, OR surface a stale-weights warning when holdout-vs-live diverges past threshold. **Timing-gated** — needs ~7 days of live records before the decision is data-informed. Don't start before 2026-05-27. The leakage fix may shift live MAPE meaningfully; revisit the decision criterion once the new model has accumulated ~7 days of post-fix predictions.
- **Practice the explanatory docs** (~10 min, off-keyboard). Read `HOW_IT_WORKS.md` aloud + pick one STAR story to rehearse. Satisfies criterion (b) of the 14-day success criterion.
- **PR-C2** (`PITCH.md` + expanded STAR stories) — parked unless interview cycle demands it.

The Next-3 is back-to-back substantive code work for the next ~12h of focused time. The audit-driven correctness fixes take priority over the timing-gated #121-part-3 work.

## Blocked / waiting on

- **Overview model card MAPE shows simulated baseline values**
  ([#131](https://github.com/kristenmartino/gridpulse/issues/131)) — the
  `is_trained` badge was fixed in PR #130 (now reads Redis), but the
  MAPE / RMSE / MAE / R² displayed on the card still come from
  `_simulate_forecasts` because the web tier can't read meta.json from
  the Job container's disk. Real fix: scoring job writes `model_metrics`
  into the forecast Redis payload, web tier reads from there. ~4 hours.

- **Forecast tab chart 1–4h gap between actual end and forecast start**
  ([#129](https://github.com/kristenmartino/gridpulse/issues/129)) —
  EIA publishing lag visualized as empty. Fix is in
  `jobs/phases.predict_and_write_forecast`: backfill predictions for
  trailing NaN-demand rows so the forecast trace starts at
  `last_actual_demand_hour + 1h` instead of `featured.timestamp.max() + 1h`.
  Different code path than this PR; ~3–4 hours when picked up.

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
- **PR-B (doc-drift CI) decision** — ship only if drift surfaces during
  PR-C2 / #121 work. Otherwise stays deferred.
- **PR-D (audit workflow)** — deferred indefinitely per [2026-05-20
  wider replan](https://github.com/kristenmartino/gridpulse/pull/123).
  Saved to [`claude-templates/DEFERRED_FOR_PRODUCTION.md`](https://github.com/kristenmartino/claude-templates/blob/main/DEFERRED_FOR_PRODUCTION.md)
  for revisit if/when GridPulse becomes production-grade or continuously
  updated.

## Recent decisions (last 7 days)

- **2026-05-20** **Forecast pipeline audit + four follow-up PRs queued.** Senior-staff audit of the per-model forecast calculations confirmed: (a) reported holdout MAPE is genuinely out-of-sample (last 168 hours, disjoint from train); (b) Prophet / ARIMA / live-drift paths are honest; (c) but XGBoost training features have direct target leakage — ``ramp_rate[i] = demand[i] - demand[i-1]`` and ``demand_roll_{24,72,168}h_*`` aggregations include the current row. Doesn't directly inflate the surfaced holdout MAPE (which uses the honest snapshot at val time), but contaminates model weights and creates a train/serve distribution shift. Empirically confirmed via ``/tmp/leakage_demo.py``: leaky model trains to 0.11% MAPE (memorizing), honest model trains to 0.24% (must generalize). Four PRs queued: this one (de-leak training), PR-E (recursive prod features), PR-C (real weather forecast in future features), PR-B (empirical CI bands).
- **2026-05-20** [PR #134](https://github.com/kristenmartino/gridpulse/pull/134) — Overview tab "honest signals" pass. Five user-visible fixes in one PR: timestamp-based 24h trend (was ``iloc[-25]``, drifted with EIA publishing gaps), freshness subtext on NOW (``as of HH:MM UTC``), live drift MAPE in the forecast clause (was citing training-time holdout, misleading), label clarifications on 7d Peak/Low/Average, "Recent peak" → "Last 24h peak". New traceability audit script ``scripts/audit/verify_overview_metrics.py`` confirms all 5 spot-checked regions reconcile from raw EIA actuals. 12 new tests, 1690 total. **All four follow-up PRs from the audit are now queued, listed in Next-3 below.**
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
