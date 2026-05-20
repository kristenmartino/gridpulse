<!--
How this file gets maintained:
- Per-PR: updated in the same commit as material work that changes
  active focus, next-3, blocked-on, or recent decisions
- End-of-session: agent re-verifies against `gh issue list`, `gh pr list`
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
recruiter-facing surface AND close one Path B item ([#121](https://github.com/kristenmartino/gridpulse/issues/121)
Model drift monitoring) as a high-quality interview asset. Path B
items beyond #121 stay deferred — see [`docs/internal/NEXT_UP.md`](docs/internal/NEXT_UP.md) §V4.

**Open question — 14-day success criterion (by 2026-06-03):** at least
2 of these must be true, or the PM infrastructure built this week is
theatrical and should be partially reverted:

- [ ] (a) [#121](https://github.com/kristenmartino/gridpulse/issues/121) has a draft PR or partial implementation
- [ ] (b) `docs/HOW_IT_WORKS.md` and `docs/INTERVIEW_PREP.md` have real content (not stubs) and have been used at least once for actual practice
- [ ] (c) The handoff quickstart has been run on sift-news or another repo, producing a STATUS.md there

## Next 3 (priority order)

1. **PR-C1 — Recall artifacts** (`HOW_IT_WORKS.md` + 5 Mermaid diagrams, ~90 min). Next session. Closes the user's literal stated need ("diagrams and how it works so I can share and recall").
2. **PR-C2 — Communication artifacts** (`PITCH.md` 3 lengths + `INTERVIEW_PREP.md` with 5 STAR stories, ~90 min). Session after that.
3. **[#121](https://github.com/kristenmartino/gridpulse/issues/121) — Model drift monitoring** (~1 week, `path-b`, `effort-week`). The 2026-05-19 PJM walkthrough surfaced a 47 GW model spread; closing this gap is both real product work AND generates the strongest STAR story this project will produce.

[#122](https://github.com/kristenmartino/gridpulse/issues/122) (V3.γ Hawaii) is queued but lower priority — blocked on HECO data-quality assessment AND less leverage on portfolio narrative than #121.

## Blocked / waiting on

- **Cross-link this Project to portfolio-v2 / sift-news / future repos**
  — wait until ≥2 repos have their own STATUS.md before linking makes
  sense. Tracked separately as an issue with `pm-followup` label.
- **PR-B (doc-drift CI) decision** — ship only if drift surfaces during
  PR-C1 / PR-C2 / #121 work. Otherwise stays deferred.
- **PR-D (audit workflow)** — deferred indefinitely per [2026-05-20
  wider replan](https://github.com/kristenmartino/gridpulse/pull/123).
  Saved to [`claude-templates/DEFERRED_FOR_PRODUCTION.md`](https://github.com/kristenmartino/claude-templates/blob/main/DEFERRED_FOR_PRODUCTION.md)
  for revisit if/when GridPulse becomes production-grade or continuously
  updated.

## Recent decisions (last 7 days)

- **2026-05-20** Wider replan after multi-perspective review: confirmed
  Position A, deferred Path B beyond #121, reordered PR sequence to
  C → B (conditional) → D (deferred). [PR #123 touch-up commit]
- **2026-05-19** Path A declared complete. [#120](https://github.com/kristenmartino/gridpulse/pull/120)
- **2026-05-19** Scenario simulator: heuristic over full-fidelity engine. [#119](https://github.com/kristenmartino/gridpulse/pull/119)
- **2026-05-19** Project-state lives in GitHub, not Markdown. [#123](https://github.com/kristenmartino/gridpulse/pull/123)
- **2026-05-18** Big-bang Redis namespace flip over phased migration. [#114](https://github.com/kristenmartino/gridpulse/pull/114)
