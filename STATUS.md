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
- [x] (c) [#121](https://github.com/kristenmartino/gridpulse/issues/121) has a draft PR or partial implementation (PR-D1, this PR — backend drift measurement)
- [ ] (d) The handoff quickstart has been run on sift-news or another repo

**2 of 4 criteria now satisfied — the ≥2 threshold is cleared. The PM infrastructure built this week is not theatrical.** Criterion (b) takes ~10 min of reading aloud; (d) takes ~60 min running the quickstart on sift-news.

## Next 3 (priority order)

1. **[#121](https://github.com/kristenmartino/gridpulse/issues/121) part 3 — Ensemble weight integration** (~2–3 days, `path-b`). Decision: incorporate live MAPE into ensemble weights, OR surface a stale-weights warning when holdout-vs-live diverges past threshold. Decide based on the live drift signal once part 2 (this PR's panel) has accumulated ~7 days of records in production.
2. **Run handoff quickstart on sift-news** (~60 min, satisfies (d) above + unblocks [#124](https://github.com/kristenmartino/gridpulse/issues/124) cross-linking).
3. **PR-C2** (`PITCH.md` + expanded STAR stories) — parked unless interview cycle demands it. Currently no signal.

## Blocked / waiting on

- **Cross-link this Project to portfolio-v2 / sift-news / future repos**
  ([#124](https://github.com/kristenmartino/gridpulse/issues/124)) —
  wait until ≥2 repos have their own STATUS.md before linking makes
  sense.
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

- **2026-05-20** PR-D2 — [#121](https://github.com/kristenmartino/gridpulse/issues/121) part 2 shipped. Models tab drift panel: `_build_drift_panel` reads `gridpulse:drift:{region}` + holdout MAPEs, renders per-model status chips (on track / drifting / degraded) with mixed-state support. 15 new tests. Full suite: 1,640 pass. [This PR]
- **2026-05-20** PR-D1 — [#121](https://github.com/kristenmartino/gridpulse/issues/121) part 1 shipped. `models/drift.py` (continuous 1-hour-ahead drift measurement) + `jobs/phases.write_drift_metrics` (hourly Redis writes to `gridpulse:drift:{region}`) + 36 new unit tests. Full suite: 1,625 pass. [PR #126]
- **2026-05-20** PR-C1 — Recall artifacts shipped. Real `HOW_IT_WORKS.md` + 5 Mermaid diagrams + populated `CANONICAL_FACTS.md` + `INTERVIEW_PREP.md` STAR-story content. [PR #125]
- **2026-05-20** Wider replan after multi-perspective review: confirmed Position A, deferred Path B beyond #121, reordered PR sequence to C → B (conditional) → D (deferred), and split PR-C into C1 (recall) + C2 (communication). [PR #123]
- **2026-05-19** Path A declared complete. [#120](https://github.com/kristenmartino/gridpulse/pull/120)
- **2026-05-19** Scenario simulator: heuristic over full-fidelity engine. [#119](https://github.com/kristenmartino/gridpulse/pull/119)
- **2026-05-19** Project-state lives in GitHub, not Markdown. [#123](https://github.com/kristenmartino/gridpulse/pull/123)
- **2026-05-18** Big-bang Redis namespace flip over phased migration. [#114](https://github.com/kristenmartino/gridpulse/pull/114)
