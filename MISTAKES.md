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

One line per entry, newest first — **new entries go at the top of the list
below, immediately under the `audited-through` marker**, not above the marker
and not appended at the bottom. An entry above the marker reads as already
audited when it is not. Format:
`- YYYY-MM-DD [category] one-line description — <ref: PR/issue/session>`

No root cause, no prevention, no status field — that analysis happens in
Audit, not here. `[category]` can be a best guess; the audit pass is what
actually decides whether two entries share a root cause. Anyone may append;
nothing here is authoritative until it's promoted to Analyzed.

The marker below is what stops the SessionStart nudge from nagging forever.
`audit-mistakes-log` advances it to the run date **every** time it finishes,
including when it decides nothing graduates yet — "I looked, these can wait"
is a real outcome and needs somewhere to be recorded. The nudge counts only
entries dated after it, so a deliberate no-promotion decision buys quiet
until genuinely new candidates arrive. `never` means no audit has run.

<!-- audited-through: 2026-08-18 -->

- 2026-08-18 [guard-decision-without-force] The close-keyword guard fired correctly on a live reference and returned a PreToolUse `ask`, and the command then ran with no prompt — in permissive or auto-approving sessions an `ask` gates nothing, so a correct guard protected nothing. Found only because per-invocation telemetry distinguished "ran and was overruled" from "never ran". Backticked case switched to `deny`. — ref: PR #579
- 2026-08-18 [unverified-premise] Repeated an issue body's technical rationale ("the anchor cannot be recovered retrospectively because `lead_hours` is the realized lead") as established fact in `docs/BENCHMARK_METHODOLOGY.md` and a commit message; row 0 is `anchor + 1h` by construction, so the anchor is exact arithmetic and a bounded reconstruction was available. Corrected pre-merge after a challenge, not by checking. — ref: #547 / PR #555
- 2026-08-18 [test-validity] Wrote a unit assertion against a value that a monkeypatched fixture hardcodes (`_patch_predict_one` stubs `_build_future_feature_frame` and ignores `start_ts`), so the test exercised the stub's constant rather than the code under test. Caught by the assertion failing, not by review. — ref: #547 / PR #555
- 2026-08-18 [guard-coverage-gap] Shipped a guard test against stale published counts whose surface list omitted `docs/CANONICAL_FACTS.md` — the file CLAUDE.md's end-of-PR check routes a moved cited fact to, and so the likeliest place for one to be added. — ref: PR #538, fixed in #551
- 2026-08-18 [explanation-before-measurement] Wrote the causal claim into a shipped docstring ("this closes the defect", naming `filter_low_actuals` as the mechanism) before running the production measurement that would test it; the measurement showed that filter dropped 2 records fleet-wide and a different one did the work. Caught and corrected pre-merge. — ref: PR #543 / #541
- 2026-08-18 [destructive-step-chained-to-unchecked-outcome] Ran `gh pr merge 567` and the head-branch delete in one command without gating the delete on the merge result; the merge failed on a fresh conflict and the delete then closed the PR. Recovered from the intact local branch. — ref: PR #567
- 2026-08-18 [worklog-concurrent-deposit] Two sessions depositing at the same time collided: PRs #566, #570 and #567 all inserted at the top of this list, producing a merge conflict in this file on four separate rebase steps. The resolution is trivial (keep both) but every concurrent deposit hits it. — ref: PR #567
- 2026-08-18 [local-verification-narrower-than-ci] Reported lint clean after running `ruff check` only; CI's lint job also runs `ruff format --check`, which failed on a newly added script and cost a CI cycle. — ref: PR #560
- 2026-08-18 [configured-but-inert] The SessionStart nudge hook resolved `MISTAKES.md` by bare relative path behind an `[ -f ]` guard, so from any subdirectory it exited 0 with no output — identical to "checked, nothing to report." Caught by testing it from `docs/` before merge. — ref: PR #561

---

## Analyzed

Pattern tally first (derived from the entries below — recount when Audit
promotes something new), then the full entries: what happened, root cause,
prevention, and whether it graduated into a CLAUDE.md rule.

### Pattern tally

| category | occurrences | status | rule / fix |
|---|---:|---|---|
| reference-verification (was github-close-keywords) | 3 | graduated | CLAUDE.md → "Verify every `#N` reference" + "The backtick/quote trap" |
| reliability-timeout-budget | 2 (distinct root causes, same family) | graduated | CLAUDE.md → "Upstream-outage resilience" + "Partial degradation is a DIFFERENT failure class" |
| single-source-of-truth-drift | 2 | 1 graduated, 1 resolved-by-test | CLAUDE.md → End-of-PR check item 2 (grep rule); `MODEL_DISPLAY_NAMES` + AST sweep test |
| stale-repo-snapshot | 3 | graduated | CLAUDE.md → "Before recommending what's next" (re-derive the premise) |
| worklog-concurrent-deposit | 1 (in Worklog) | open | recurred during the 2026-08-18 audit itself — near the bar |
| explanation-before-measurement | 1 (in Worklog) | open | none yet — watching for a repeat |
| guard-coverage-gap | 1 (in Worklog) | open | none yet — watching for a repeat |
| destructive-step-chained-to-unchecked-outcome | 1 (in Worklog) | open | none yet — watching for a repeat |
| local-verification-narrower-than-ci | 1 (in Worklog) | open | none yet — watching for a repeat |
| configured-but-inert | 1 (in Worklog) | open | none yet — watching for a repeat |

### Entries

**2026-08-18 — A ref written for a PR that did not exist yet, in a file the close-keyword rule did not cover [reference-verification]**
- **What happened:** A `MISTAKES.md` Worklog line cited `ref: PR #566` for a PR
  that had not been opened when the line was written. By the time #566 existed
  it belonged to a different session's unrelated work, so the entry's evidence
  trail pointed at the wrong thing. Nothing was closed and no state was
  corrupted — the damage is that a future reader following the ref lands
  somewhere unrelated.
- **Root cause:** Same discipline failure as the 2026-05-29 close-keyword
  incidents — a reference written without confirming it resolves to the
  intended thing — but outside every boundary the rule drew. The rule named
  *issues* (`gh issue view`), this was a *PR*; it named *PR bodies, commit
  messages and STATUS.md*, this was `MISTAKES.md`; and it framed the harm as
  closing the wrong issue, where here nothing closes at all. There is also a
  mechanic the rule could not express: the number **did not exist yet**, so it
  was unverifiable rather than merely unverified, and PR/issue numbers are
  allocated globally and race between concurrent sessions.
- **Prevention:** Broadened the existing rule rather than adding a sibling —
  any `#N`, in any committed file, verified with `gh issue view` or
  `gh pr view` as appropriate; and a forward reference is written after the
  thing exists, or named by branch instead.
- **Status:** graduated → CLAUDE.md § Verify every `#N` reference — issue or PR
  (2026-08-18), which broadens the rule the 2026-05-29 entry below created.
- **Related:** the 2026-05-29 entry below (occurrences 1–2). Category renamed
  from `github-close-keywords`, which described the symptom of those two rather
  than the root cause; the old name is kept in the tally so the rename is
  traceable. **This grouping is the pass's least certain call** — those two
  corrupted issue *state* via GitHub's scanner, this one mis-attributed
  *evidence* in a file GitHub never reads. Recorded here so a later pass can
  overturn it rather than inherit it silently.

**2026-08-18 — Three artifacts captured `main` at authoring time, aged silently, and were nearly acted on [stale-repo-snapshot]**
- **What happened:** Three times in one day, work nearly proceeded from a stale
  view of `main`. (1) A branch (`exp/478-bias-measurability`) whose earlier
  commits had already been squash-merged would have opened a PR whose diff read
  as a revert, its `STATUS.md` predating 11 later commits. (2) A branch stacked
  on PR #565 stopped being an ancestor of `main` the moment #565 squash-merged.
  (3) A saved plan directed deleting `alertPolicies/15888827698887105322` and
  recreating it under a new id, on the premise that its runbook was
  applied-stale and un-editable in place — a premise resolved by 553 landing on
  `main` *after* the plan was written; executing it would have deleted a
  healthy, enabled, actively-firing production alert policy. All three were
  caught before acting; none ran.
- **Root cause:** A branch and a plan file both freeze repo state when they are
  authored and carry no signal that it has since moved. In each case the check
  a reader would naturally reach for reinforced the stale view rather than
  exposing it: `git log branch..main` lists squash-merged commits as unmerged,
  a squash merge silently severs ancestry for anything stacked on it, and the
  plan stated its premise as settled fact — it even instructed re-checking
  `git log` because `main` had moved once already, and `main` had moved again
  since. Squash-merging is the common accelerant: it rewrites the commit that
  branches and plans were reasoning about, so their view of `main` expires
  without any local signal.
- **Prevention:** Before acting on any artifact authored earlier, re-derive its
  premise from the authoritative source — `origin/main`, the live API, the
  running service — and stop and report when it no longer holds rather than
  executing as written. No mechanical guard is proposed: a plan document's
  premise is prose and cannot be validated by a test, which is what makes this
  a judgment call worth stating as a rule.
- **Status:** graduated → CLAUDE.md § Before recommending what's next (2026-08-18)
- **Related:** PRs #553, #560, #565; three Worklog lines consumed into this
  entry. Promoted on the **repeat** bar; the severity bar was also available
  (occurrence 3 would have deleted a live firing alert policy) and deliberately
  not invoked, since three occurrences is the cleaner justification. The audit
  that promoted this was itself interrupted by a fourth instance of the same
  pattern — `main` gained four Worklog entries, including occurrence (2), while
  the analysis was in flight, so the first draft of this entry counted two.

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
