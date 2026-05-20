# Status — updated 2026-05-19

> Canonical pointer for "where am I, what's next." This file +
> [GitHub Projects board](https://github.com/users/kristenmartino/projects/1)
> + the issue tracker are the single source of truth for project state.
> See [`docs/internal/NEXT_UP.md`](docs/internal/NEXT_UP.md) for the full
> historical roadmap with acceptance criteria; see [`CLAUDE.md`](CLAUDE.md)
> for the pre-session sanity-check ritual that keeps this file from
> drifting.

## Active investment

**None active.** Path A (portfolio-grade complete) was declared done
2026-05-19 with the close of #115 / #117 / #118 / #119 / #120. The
roadmap is awaiting a strategic decision: invest in Path B #1
(Model drift monitoring) now, or pause.

The Path B #1 case is strengthened by a real observation —
see §"Open question" below.

## Next 3 (priority order)

1. **[#121](https://github.com/kristenmartino/gridpulse/issues/121) — V4 Path B #1: Model drift monitoring** (~1 week, `path-b`, `effort-week`). Recommended next investment. Concrete evidence: 2026-05-19 PJM walkthrough surfaced a 47 GW spread between XGBoost / Ensemble / Prophet / ARIMA on the same horizon — exactly the symptom drift monitoring is designed to catch.
2. **[#122](https://github.com/kristenmartino/gridpulse/issues/122) — V3.γ: Hawaii coverage** (3–5 days, `v3-open`, `effort-week`). Blocked on HECO data-quality assessment.
3. **V4 Path B #2: Observability infrastructure** (3–5 days). Not yet promoted to an issue — sketch lives in [`docs/internal/NEXT_UP.md` §V4 Path B](docs/internal/NEXT_UP.md). Promote to issue when committing to start.

The remaining 6 V4 Path B items (auth, API surface, alerting, DR,
data-quality monitoring, cost monitoring) are sketches in NEXT_UP.md.
They become issues when committed to, not before.

## Recent decisions (last 7 days)

- **Path A declared complete** (2026-05-19) — #21 / #25 / #26 / #87 / #91 all closed + the 2026-05-19 sweep (#116 / #118 / #119) merged. There is no remaining Path A backlog. [PR #120](https://github.com/kristenmartino/gridpulse/pull/120).
- **Scenario simulator: heuristic over full-fidelity engine** (#119) — adding wind/solar→demand coupling to the existing analytical approximation rather than wiring the real `simulation/scenario_engine.py` to a server-side debounced callback. Full physics is parked as a follow-up if there's ever a real user.
- **Big-bang Redis namespace flip over phased migration** (#114) — closed #91 by accepting ~1 hour of "warming" downtime instead of building a 4-phase zero-downtime cutover for a single-tenant portfolio app.
- **Project-state lives in GitHub, not Markdown** (this PR) — STATUS.md + GitHub Projects + Issues are canonical. `docs/internal/NEXT_UP.md` keeps the historical roadmap with acceptance criteria but is no longer the operational queue.

## Open question

**Path B #1 (Model drift monitoring) now, or pause?**

Arguments **for now**: the 2026-05-19 PJM spread is real evidence that
the inverse-MAPE ensemble weights drift between trainings. Today's
portfolio-grade product silently misrepresents forecast confidence in
that condition. Building drift monitoring closes that gap.

Arguments **for pause**: Path B is ~6–8 weeks of focused work end to
end. The marginal portfolio-recruiter value of "I built drift
monitoring" over the current shipped state is modest. The real return
is when there's a paying user / interviewer who probes the ML
operationalization story.

Recommend: **decide before next session.** The cron-triggered doc
audit will create an issue every month regardless — don't let the
audit be the only thing in the queue.

## How this file gets updated

- Per-PR: updated **in the same commit** as material work that
  changes active investment, next-3, or recent decisions
- End-of-session: agent re-reads STATUS.md and verifies it matches
  reality (PRs merged, issues closed, decisions made) — opens a
  small touch-up commit if drift
- Pre-external-use ritual (interview, networking event, portfolio
  share): user re-reads top-to-bottom; ~1 minute

If this file disagrees with `gh issue list` / `gh pr list` /
`config.py` / merged PR bodies, **the live sources win** — fix
STATUS.md in a follow-up commit.
