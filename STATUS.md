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

- [x] (a) `docs/HOW_IT_WORKS.md` has real content (shipped PR #125)
- [x] (a') `docs/INTERVIEW_PREP.md` has 5 full STAR stories + `docs/PITCH.md` has 3 length variants (PR-C1 + PR-C2 combined in PR #125)
- [ ] (b) `HOW_IT_WORKS`, `INTERVIEW_PREP`, `PITCH` have been used at least once for actual practice (read aloud, time, refine for stumble points)
- [ ] (c) [#121](https://github.com/kristenmartino/gridpulse/issues/121) has a draft PR or partial implementation
- [ ] (d) The handoff quickstart has been run on sift-news or another repo

## Next 3 (priority order)

1. **[#121](https://github.com/kristenmartino/gridpulse/issues/121) — Model drift monitoring** (~1 week, `path-b`, `effort-week`). The 2026-05-19 PJM walkthrough surfaced a 47 GW model spread; closing this gap is real product work AND generates the strongest STAR story this project will produce. (Promotes from #2 → #1 now that PR-C work is complete.)
2. **Run the handoff quickstart on sift-news** (~60 min, satisfies success criterion (d)). Validates the [`claude-templates`](https://github.com/kristenmartino/claude-templates) spec works cross-project; produces a second repo with STATUS.md to enable cross-linking ([#124](https://github.com/kristenmartino/gridpulse/issues/124)).
3. **[#122](https://github.com/kristenmartino/gridpulse/issues/122) — V3.γ Hawaii** (~3–5 days, `v3-open`, `effort-week`). Lower priority — blocked on HECO data-quality assessment, lower portfolio leverage than #121.

## Blocked / waiting on

- **Cross-link this Project to portfolio-v2 / sift-news / future repos**
  ([#124](https://github.com/kristenmartino/gridpulse/issues/124)) —
  wait until ≥2 repos have their own STATUS.md before linking makes
  sense.
- **PR-B (doc-drift CI) decision** — ship only if drift surfaces during
  PR-C2 / #121 work. Otherwise stays deferred.
- **PR-D (audit workflow)** — deferred indefinitely per [2026-05-20
  wider replan](https://github.com/kristenmartino/gridpulse/pull/123).
  Saved to [`claude-templates/DEFERRED_FOR_PRODUCTION.md`](https://github.com/kristenmartino/claude-templates/blob/main/DEFERRED_FOR_PRODUCTION.md)
  for revisit if/when GridPulse becomes production-grade or continuously
  updated.

## Recent decisions (last 7 days)

- **2026-05-20** PR-C1 + PR-C2 shipped combined in [PR #125](https://github.com/kristenmartino/gridpulse/pull/125). `HOW_IT_WORKS.md` (5 sections, 5 Mermaid diagrams) + `CANONICAL_FACTS.md` (populated) + `INTERVIEW_PREP.md` (5 full ~225-word STAR stories with practice notes) + `PITCH.md` (30s / 2min / 5min variants). STATUS.md restructured per review §5; CLAUDE.md caveat removed.
- **2026-05-20** Wider replan after multi-perspective review: confirmed Position A, deferred Path B beyond #121, reordered PR sequence to C → B (conditional) → D (deferred), and split PR-C into C1 (recall) + C2 (communication). [PR #123]
- **2026-05-19** Path A declared complete. [#120](https://github.com/kristenmartino/gridpulse/pull/120)
- **2026-05-19** Scenario simulator: heuristic over full-fidelity engine. [#119](https://github.com/kristenmartino/gridpulse/pull/119)
- **2026-05-19** Project-state lives in GitHub, not Markdown. [#123](https://github.com/kristenmartino/gridpulse/pull/123)
- **2026-05-18** Big-bang Redis namespace flip over phased migration. [#114](https://github.com/kristenmartino/gridpulse/pull/114)
