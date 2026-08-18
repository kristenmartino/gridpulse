# Mistakes

Two-part log: a cheap **Worklog** anyone deposits to mid-task, and an
**Analyzed** section that only a separate audit pass (or a backfill of an
already-understood incident) writes to. See
[`CLAUDE.md`](CLAUDE.md) → "Mistake logging & rule graduation" for the full
policy, and `.claude/skills/check-past-mistakes/` +
`.claude/skills/audit-mistakes-log/` for the two skills that use this file.

**The split exists on purpose.** A session mid-task deciding its own root
cause and proposing its own prevention is exactly the failure mode this
file is designed to avoid — it's biased by the same tunnel vision that
caused the mistake, and it's expensive enough per-entry that people stop
doing it. So depositing is a one-liner, no judgment call required. Deciding
whether something is a real pattern, what actually caused it, and what would
fix it happens later, separately, by something with none of the original
session's context.

**This file is an evidence store, not a runtime lookup — it is meant to sit
on disk mostly unread.** Rules get derived from it once a pattern emerges,
and it's what a rule gets audited against later if anyone questions why the
rule exists — that's why entries never get deleted on graduation. It is
**not** something a normal working session reads to check its own work; that
would both balloon this file's cost every single time and duplicate what the
always-loaded, deliberately small [`CLAUDE.md`](CLAUDE.md) invariants are
for. If you're reading this file mid-task for anything other than appending
one Worklog line, you're probably `audit-mistakes-log` — if you're not, stop
and go check `CLAUDE.md` instead.

---

## Worklog (undecided candidates)

One line per entry, newest first — **new entries go at the top of this
list**, right below this paragraph, not appended at the bottom. Format:
`- YYYY-MM-DD [category] one-line description — <ref: PR/issue/session>`

No root cause, no prevention, no status field — that analysis happens in
Audit, not here. `[category]` can be a best guess; the audit pass is what
actually decides whether two entries share a root cause. Anyone may append;
nothing here is authoritative until it's promoted to Analyzed.

- 2026-08-18 [stale-plan-premise] Nearly executed a saved plan that would have deleted a healthy, enabled, actively-firing alert policy and minted a new id — its stated premise (a runbook stuck applied-stale, un-editable in place) had been resolved by 553 landing on `main` after the plan was written. Caught by checking the premise against the live policy before acting. — ref: this session, no PR (nothing executed)
- 2026-08-18 [local-verification-narrower-than-ci] Reported lint clean after running `ruff check` only; CI's lint job also runs `ruff format --check`, which failed on a newly added script and cost a CI cycle. — ref: PR #560
- 2026-08-18 [configured-but-inert] The SessionStart nudge hook resolved `MISTAKES.md` by bare relative path behind an `[ -f ]` guard, so from any subdirectory it exited 0 with no output — identical to "checked, nothing to report." Caught by testing it from `docs/` before merge. — ref: PR #561
- 2026-08-18 [stale-branch-diff] Nearly opened a PR from `exp/478-bias-measurability` after its earlier commits had been squash-merged into `main` — `git log branch..main` still showed them as unmerged and the branch's `STATUS.md` predated 11 later commits, so the diff would have looked like a revert. Caught by diffing against `origin/main` before pushing. — ref: this session, no PR (caught pre-push)

---

## Analyzed

Pattern tally first (derived from the entries below — recount when Audit
promotes something new), then the full entries: what happened, root cause,
prevention, and whether it graduated into a CLAUDE.md rule.

### Pattern tally

| category | occurrences | status | rule / fix |
|---|---:|---|---|
| github-close-keywords | 2 | graduated | CLAUDE.md → "Verify every `Closes #N`" + "The backtick/quote trap" |
| reliability-timeout-budget | 2 (distinct root causes, same family) | graduated | CLAUDE.md → "Upstream-outage resilience" + "Partial degradation is a DIFFERENT failure class" |
| single-source-of-truth-drift | 2 | 1 graduated, 1 resolved-by-test | CLAUDE.md → End-of-PR check item 2 (grep rule); `MODEL_DISPLAY_NAMES` + AST sweep test |
| stale-branch-diff | 1 (in Worklog, not yet promoted) | open | none yet — watching for a repeat before drafting a rule |

### Entries

**2026-08-11 — One model, three display names, in three different places, patched three times locally and generalized zero times [single-source-of-truth-drift]**
- **What happened:** Models tab showed "SARIMAX," Forecast tab showed
  "ARIMA," Overview showed "Arima" (from `model_name.title()`) — three
  spellings, none of them a single source of truth. Fixed in PR #495 by
  introducing a canonical `MODEL_DISPLAY_NAMES` map, then contradicted 18
  minutes later by PR #496, which moved a fourth site (`/about`) to a
  fourth-ish spelling to match what it locally observed, unaware #495
  existed.
- **Root cause:** The same `.title()`-casing bug had already been hand-patched
  at three call sites independently — the defect was that the mapping was
  authoritative in zero places, not that any one site was wrong. Two PRs
  fixing the same underlying bug without knowing about each other is what a
  missing single source of truth looks like in practice.
- **Prevention:** A canonical map read by every label site, plus an
  AST-based sweep test that fails on any site constructing a label without
  going through it, plus the label ban extended to `web/*.html` (the surface
  the first sweep missed).
- **Status:** graduated (doc-drift half) → CLAUDE.md End-of-PR check item 2,
  which generalizes to "a cited fact moved → grep for the old literal
  outside the source of truth." **Resolved, not graduated** (naming half):
  the AST sweep test makes the mistake structurally impossible to
  reintroduce silently — nothing left for an agent to remember by hand.
- **Related:** PRs #495, #496, #504, #506.

**2026-08-07 — A retrain moved a published number; the public page didn't move with it for four days [single-source-of-truth-drift]**
- **What happened:** `/about` published `4.8%`. A 2026-08-07 retrain moved
  the real number to `4.35%` in `docs/CANONICAL_FACTS.md`. The public page
  kept showing `4.8%` for four days.
- **Root cause:** `tests/unit/test_public_copy_traces_to_canonical_facts.py`
  only asserted *a* literal was present on the page — never that it was the
  *current* one. A test that checks presence instead of provenance passes
  forever once it passes once, regardless of whether the fact underneath it
  has moved.
- **Prevention:** Rewrote the test to fail on the source side — it reads
  `CANONICAL_FACTS.md` and asserts the page matches it, not the reverse.
- **Status:** graduated → CLAUDE.md End-of-PR check item 2: when a cited
  fact moves, `grep -rn '<old literal>' web/` in the same PR, because a
  stale published number is worse than a stale internal one.
- **Related:** `docs/CANONICAL_FACTS.md`, `tests/unit/test_public_copy_traces_to_canonical_facts.py`.

**2026-08-04 — Zero hard failures, two SIGKILLs: the circuit breaker was built for the wrong failure shape [reliability-timeout-budget]**
- **What happened:** The scoring job burned ~800s and hit two SIGKILLs at
  the 1800s task timeout (#389), even though every EIA call eventually
  succeeded — zero `eia_max_retries_exceeded`, zero fallbacks triggered.
  The #174 circuit breaker never tripped: it counts *consecutive* hard
  failures and `record_success()` resets that counter on every interleaved
  success, so an upstream that's merely slow and flaky (8–15% of calls
  retrying, the rest fine) keeps the breaker closed by construction while
  still paying full retry cost every call.
- **Root cause:** The #174 fix bounds a *total outage* (all calls failing)
  but has no concept of "this call is expensive even though it will
  eventually succeed." Partial degradation is a different failure shape
  from total outage, and the existing guard's own success-path logic
  actively hid it.
- **Prevention:** Bound cost per call (`EIA_CALL_BUDGET_S`, split
  connect/read timeouts, clamped to time remaining) and cost per run (soft
  deadline that sheds remaining work and still writes completion metadata
  before exiting). Deliberately did **not** make the breaker trip on failure
  rate — that trades fresh-but-slow data for stale-but-fast, the wrong
  tradeoff here.
- **Status:** graduated → CLAUDE.md "Partial degradation is a DIFFERENT
  failure class," rules 3–5, pinned by two characterization tests.
- **Related:** [#389](https://github.com/kristenmartino/gridpulse/issues/389), `tests/unit/test_eia_client.py`. Linked to the entry below — same
  family, different failure shape; the second incident is the first fix's
  blind spot made visible.

**2026-06-04 — A 2-hour EIA outage overran the job timeout because retries multiply across 51 BAs [reliability-timeout-budget]**
- **What happened:** A 2-hour EIA 504 outage failed the scoring job.
  Per-call retry-to-exhaustion (`MAX_RETRIES × timeout + backoff`)
  multiplied across 51 BAs × multiple endpoints alone overran the job's
  task timeout before any per-call fallback got a chance to engage.
- **Root cause:** Retry budgets were reasoned about per-call, never summed
  across the fan-out. A negligible cost once looks completely different
  multiplied by 51.
- **Prevention:** Fallback to last-known data made uniform across every
  endpoint in a client, not just the primary one; a process-local circuit
  breaker that fail-fasts to the fallback after K consecutive hard
  failures, with a periodic probe to recover mid-run.
- **Status:** graduated → CLAUDE.md "Upstream-outage resilience," rules 1–2.
- **Related:** [#174](https://github.com/kristenmartino/gridpulse/issues/174).

**2026-05-29 — Quoting a bad `Closes #N` inside a commit message re-triggered it [github-close-keywords]**
- **What happened:** PR #165 wrote `Closes #150` when the intended issue was
  #148, silently closing the wrong issue. The follow-up commit written to
  *document* this mistake put `` `Closes #150` `` in backticks to quote it —
  and re-closed #150, which had just been reopened, because GitHub's
  close-keyword scanner reads commit messages and PR bodies and does not
  respect backticks, code spans, or surrounding prose.
- **Root cause:** Two failures stacked: a `Closes #N` written from memory
  instead of verified against the actual issue, and the mistaken belief
  that quoting a close-keyword in prose is inert to GitHub's scanner.
- **Prevention:** `gh issue view <N>` before writing any `Closes #N`, every
  time. Flip issue state with `gh issue reopen|close` directly rather than
  relying on a keyword edit. When a commit must mention a close-keyword it
  does not intend to fire, break the pattern (non-adjacent text, or a
  `#NNN` placeholder) instead of quoting it verbatim.
- **Status:** graduated → CLAUDE.md "Verify every `Closes #N`" and "The
  backtick/quote trap," both under End-of-PR check.
- **Related:** PR #165, issues #148/#150.
