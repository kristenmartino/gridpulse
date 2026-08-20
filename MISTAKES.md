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

**Pending candidates live in [`.mistakes/worklog/`](.mistakes/worklog/), one
file per deposit** — not in this file. Read them with:

```bash
cat .mistakes/worklog/2026-*.md     # every pending candidate
ls .mistakes/worklog/*.md | wc -l   # how many are waiting
```

They were an inline list here until 2026-08-18. Every session that deposited
had to insert at the same line, so parallel deposits conflicted — ten PRs
touched this file that day and three were open at once. The cost was never
the conflict itself but the hand-merge that resolves it, which can silently
drop someone else's deposit from the one file whose job is not losing
evidence. A directory has nothing to merge.

Format and rules for depositing:
[`.mistakes/worklog/README.md`](.mistakes/worklog/README.md).
`audit-mistakes-log` consumes those files and promotes what crosses the bar
into the Analyzed section below; `.mistakes/last-audit` records when it last
ran, so a deliberate "these can wait" buys quiet until new candidates arrive.

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
| single-source-of-truth-drift | 4 (addendum 2026-08-20) | graduated (strengthened 2026-08-20) | CLAUDE.md → End-of-PR check item 2 (grep now `web/ docs/`); `MODEL_DISPLAY_NAMES` + AST sweep test |
| stale-repo-snapshot | 6 | graduated (strengthened 2026-08-20, twice) | CLAUDE.md → "Before recommending what's next" (re-derive the premise; merge-safety sentence) |
| claim-shipped-before-measurement | 4 | graduated (2026-08-20) | CLAUDE.md → "Verify a claim before writing it as fact" |
| verification-checked-the-wrong-thing | 5 | graduated (2026-08-20) | CLAUDE.md → Testing § "Verify a mock actually intercepted" |
| unmeasured-performance-impact | 2 | graduated (2026-08-20) | CLAUDE.md → Required working style, "Validate after meaningful changes" |
| verification-instrument | 1 | graduated (2026-08-20) | CLAUDE.md → "Before recommending what's next", "Verify a production state by its terminal write" |
| reminder-blind-to-same-day | 1 | resolved (2026-08-20) | entries-seen counter (PR #582); already narrated in CLAUDE.md's mechanical-guard section |
| guard-decision-without-force | 1 | resolved (2026-08-20) | backticked case now `deny` (PR #579); already narrated in CLAUDE.md's mechanical-guard section |
| worklog-concurrent-deposit | 1 | resolved (2026-08-20) | one-file-per-deposit design (PR #588) structurally removes the conflict |
| environment-narrower-than-target | 5 (addenda 2026-08-20) | graduated (strengthened 2026-08-20) | CLAUDE.md → Testing § "Match the real target environment when validating" |
| guard-blind-by-construction | 2 | graduated (2026-08-20) | CLAUDE.md → Testing § "A guard needs a fixture that can trigger what it exists to catch" |
| metric-definition-blind-to-edge-case (was statistic-confounded-by-shape) | 5 (addenda 2026-08-20) | graduated (2026-08-20) | CLAUDE.md → Testing § "Write the domain requirement before the code" |
| unchecked-destructive-git-chaining | 2 | graduated (2026-08-20) | CLAUDE.md → "Before recommending what's next" § "Gate a destructive git step on its actual outcome" |
| ci-guard-intermittent | 1 (in Worklog) | open | none yet — still actively causing red CI, worth investigating on its own before the next audit pass |
| scratch-over-tracked-config | 1 (in Worklog) | open | none yet — thematically echoes the assumed-vs-verified-state throughline behind `unchecked-destructive-git-chaining`/`stale-repo-snapshot`, but the mechanism is distinct enough and severity low; watching for a repeat |
| watcher-predicate | 1 (in Worklog) | open | echoes a known personal pattern (verify deploy by SHA via ancestry, not equality) not yet codified in this repo — a distinct write-side predicate-design mechanism from `stale-repo-snapshot`'s read-side premise-checking; watching for a repeat before drafting |

### Entries

**2026-08-20 — Three local checks passed while validating against a substitute that didn't match the real target [environment-narrower-than-target]**
- **What happened:** Three incidents where a local check reported success
  because it validated against something narrower than the real target,
  never noticed until the real target ran. (1) `ruff check` was run alone
  and reported clean; CI's lint job also runs `ruff format --check`, which
  failed on a newly added script and cost a CI cycle (PR #560). (2) The
  audit-nudge hook interpolated deposit text into JSON output unescaped;
  every synthetic test fixture used quote-free text, so the bug was invisible
  until the hook first ran against real migrated entries, several of which
  quoted error messages and doc headlines (PR #588). (3) The Worklog
  migration script used `mapfile` and `declare -A`, which macOS's default
  bash (3.2) does not have — it died on first run against the real shell
  rather than whatever shell the script was drafted against (PR #588).
- **Root cause:** In all three, the check that ran locally was a narrower
  stand-in for the real target — a subset of the CI command, a synthetically
  clean fixture, an assumed shell version — and the narrowing was invisible
  because the substitute *looked* equivalent. Nothing in the local check's
  own design would have revealed the gap; only running against the real
  target (the full CI command, real quoted deposit text, the actual macOS
  bash) did.
- **Prevention:** Match the real target environment when validating locally:
  the exact CI command(s) (not a subset), real/production-shaped data rather
  than synthesizable-clean fixtures, and the actual runtime a script will
  execute under (macOS ships bash 3.2, not a newer bash).
- **Status:** graduated → CLAUDE.md § Testing, "Match the real target
  environment when validating" (2026-08-20).
- **Related:** PR #560, PR #588 (two occurrences from the same PR, distinct
  root-cause instances). Three Worklog lines consumed
  (`local-verification-narrower-than-ci`, `synthetic-fixture-narrower-than-real-data`,
  `bash-version-assumption`).
- **2026-08-20 addendum (4th–5th occurrences, rule strengthened):** (4) A
  SessionStart hook resolved `MISTAKES.md` by bare relative path, correct
  only from the repo root; from any other CWD it silently exited 0,
  indistinguishable from "nothing to report" — caught only by deliberately
  testing from `docs/` before merge (PR #561). (5) A diagnostic probe
  verifying an AR-seed fix reported a uniform 8.33% residual rate — nearly
  written up as a defect — because it omitted the history-length guard the
  real scored-window study applies; true rate 0.00% (#559 / PR #584). Unlike
  occurrences 1–3, this is a diagnostic tool reimplementing a subset of the
  real pipeline rather than an execution environment, which is why the
  prevention below was broadened rather than just restated. CLAUDE.md's
  Testing § "Match the real target environment when validating" now also
  says: *"...and a diagnostic/probe script that re-derives a result a real
  pipeline already computes must apply the same guards/filters that
  pipeline applies — a probe that omits one is a narrower stand-in even when
  its output looks like a genuine measurement."* **Related:** PR #561, #559,
  PR #584. Two Worklog lines consumed (`configured-but-inert`,
  `probe-artifact-read-as-residual`).

**2026-08-20 — A guard tested only on inputs that could not trigger the defect shape it exists to catch [guard-blind-by-construction]**
- **What happened:** Two guards that passed while blind to their own target
  defect. (1) The train/inference autoregressive parity test
  (`test_training_features_match_inference_snapshot_row_by_row`) compares
  both implementations on a gapless synthetic fixture, where they agree by
  construction — it cannot see the positional-vs-temporal divergence it
  exists to catch, because that divergence only appears on gapped series
  (`#559` / `#186`). (2) A public-copy percentage-sweep carried an exemption
  for a "30%" figure that was assumed to reach the sweep and be correctly
  waved through; `_prose_of` already strips comments before the sweep runs,
  so the exemption never actually exercised the sweep's mechanics, silently
  waving through a real unregistered product claim elsewhere (PR #609/#610).
- **Root cause:** A guard's own test suite validated that the guard runs and
  returns a result, not that the guard can actually detect the specific
  failure mode it was built for. A fixture engineered to be clean (no gaps,
  a comment-stripped exemption) structurally cannot exercise the code path
  that would reveal a real defect.
- **Prevention:** When a guard/test exists to catch a specific defect shape,
  include at least one fixture known to trigger that shape, and verify any
  exemption actually reaches the guard's mechanics rather than assuming it
  does.
- **Status:** graduated → CLAUDE.md § Testing, "A guard needs a fixture that
  can trigger what it exists to catch" (2026-08-20).
- **Related:** #559, #186, PR #609, PR #610. Two Worklog lines consumed
  (`guard-blind-by-construction` ×2, same tag, distinct incidents).

**2026-08-20 — A statistic's own test encoded the same narrow assumption as its implementation, so only manual adversarial reasoning caught the edge cases [metric-definition-blind-to-edge-case]**
- **What happened:** Three incidents in the benchmark-gating work around
  PR #580, all found by hand rather than by any test. (1)
  `absent_hours_bias_pct` shipped with a 20-absent-hour floor fitted to 3-4h
  noise cases; live WALC read -19.88% over 49 absent hours sitting in two
  contiguous ~48h blocks, because the statistic conflates *which* hours a BA
  skips with *what load did* during the outage — a blocked, contiguous
  absence is a structurally different case from scattered short gaps, and
  the statistic's own name promised a selection-bias correction it doesn't
  provide for the blocked shape. (2) A scoreability gate was correct while a
  feed was down but wrong on the very first tick after the feed resumed — the
  gate's definition never considered the resumption boundary as a distinct
  state. (3) A docstring and its test asserted `warn < gate` as a sufficient
  condition, when the real requirement depended on Redis read-staleness
  across two files that nothing in the check connected.
- **Root cause:** Each check's own test suite encoded the same boundary
  assumption as the implementation — it verified the code did what the code
  does, not what the domain actually requires. None of the three were caught
  by their test because the test and the implementation shared the same
  blind spot; all three surfaced only when someone reasoned adversarially
  about the metric's boundary (resumption after absence, blocked vs.
  scattered gaps, cross-file staleness) rather than trusting the test.
- **Prevention:** When shipping a statistic, gate, or invariant check, write
  the domain requirement in words before the code, and adversarially probe
  its boundary (resumption after absence, blocked vs. scattered gaps, values
  read back through an intermediate store) — a test that only encodes the
  implementation's own comparison won't catch this.
- **Status:** graduated → CLAUDE.md § Testing, "Write the domain requirement
  before the code" (2026-08-20).
- **Related:** PR #580 (all three occurrences). Renamed from
  `statistic-confounded-by-shape`, which named only occurrence (1); the
  broader pattern covers all three. Three Worklog lines consumed.
- **2026-08-20 addendum (4th–5th occurrences):** Two more metric/policy
  definitions blind to a domain edge case. (4) The "web service pinned at
  max instances" alert summed `ALIGN_MAX` across the active/idle `state`
  label and `revision_name`; a 20-merge, 19-revision deploy-rollover window
  read as a sustained 7-instance ceiling while true per-revision peak was
  2.0 against a limit of 4 — a transition state the aggregation
  dimensionality never considered (PR #583, issue #581). (5) A shared
  `row.fillna(0)` step, reused in a new AR-seed snapshot, silently turns a
  genuinely-absent lag hour into `demand_lag = 0 MW` — the #129 poison the
  seed filter exists to exclude — affecting 13% of forecast steps (22.6%
  IID) and flagged as the likely reason a pre-registered study came back
  inconclusive; documented as an unmeasured limitation rather than resolved.
  The flag defaults off, so not currently live, but unresolved before it can
  be flipped on. **Worth follow-up attention**: this stays unresolved, not
  just logged — it could taint the decision to ever enable `temporal_ar_seed`
  (#559 / PR #584). No new CLAUDE.md text needed; existing prevention already
  covers this shape. **Related:** PR #583, #581, #559, PR #584. Two Worklog
  lines consumed (`instrumentation`, `inherited-policy-not-decided`).

**2026-08-20 — A destructive git step ran without inspecting the actual outcome of the step before it [unchecked-destructive-git-chaining]**
- **What happened:** Two incidents where a destructive git command ran on an
  assumed rather than inspected outcome. (1) `gh pr merge 567` and the
  head-branch delete were chained in one command without gating the delete
  on the merge's result; the merge failed on a fresh conflict, and the
  delete then ran anyway, closing the PR (PR #567). (2) In a worktree whose
  base had gone stale relative to `origin/main`, `git reset --soft
  origin/main` followed by `git add -A` staged ~490 lines of two other
  sessions' already-merged work as if it were a "reversion" to be discarded
  — caught only later, by a rebase conflict, not by inspecting what `add -A`
  had actually staged (PR #603).
- **Root cause:** Both commands assumed the outcome of the preceding
  destructive step rather than checking it — the merge's success, and the
  reset's target being current — before taking the next irreversible action.
  In a repo under heavy concurrent use, a stale worktree base makes "assumed"
  and "actual" diverge silently; nothing about the commands themselves
  signals the mismatch.
- **Prevention:** Gate a destructive git step (branch delete, `add -A` after
  a `reset`) on inspecting the actual outcome/diff of the step before it, not
  the assumed one — especially in a worktree whose base may be stale
  relative to `origin/main`.
- **Status:** graduated → CLAUDE.md § "Before recommending what's next",
  "Gate a destructive git step on its actual outcome" (2026-08-20). Promoted
  on both bars: repeat (2 occurrences), and occurrence (2) alone is close to
  severity — it would have silently reverted shipped work from two other
  sessions in a now-heavily-concurrent repo.
- **Related:** PR #567, PR #603. Two Worklog lines consumed.

**2026-08-20 — A fix shipped 79x costlier than what it replaced, and a CI cache change made builds 4.5x slower, both unmeasured before landing [unmeasured-performance-impact]**
- **What happened:** Two incidents where a change's own resource cost was
  never measured against real invocation frequency before it shipped. (1) A
  correctness fix for the positional-vs-temporal AR seed divergence (`#559`)
  landed with a first implementation that cost +74.9s per scoring tick
  fleet-wide — 79x the code it replaced — on a job that has already
  SIGKILLed at its task timeout (`#389`). It stayed dormant only because the
  flag defaulted off; the cost was found afterward, by measuring a different
  question (the seed shadow study), not by anyone checking the fix's own
  wall-clock cost before merging it (PR #584). (2) CI's Docker buildx cache
  was changed to `cache-to: type=gha,mode=max` on the assumption that a
  layer cache beats a cold rebuild; `mode=max` exports every intermediate
  layer, and the image carries prophet/xgboost/shap/scipy, so the image
  build went from 83s to 371s and became the new critical path. Caught only
  because the very next CI run happened to be watched, and reverted before
  merge (PR #586).
- **Root cause:** Both changes were reasoned about qualitatively ("this is
  more correct," "a cache should be faster") without measuring the actual
  cost against the real invocation shape — per-tick × 51-BA fan-out in (1),
  per-CI-run image size in (2). A plausible-sounding optimization or fix can
  be a regression in wall-clock or resource terms, and nothing short of
  measuring against real invocation frequency catches that before it ships.
- **Prevention:** Validate after meaningful changes by measuring wall-clock
  or resource cost against real invocation frequency (per call × real
  fan-out, per CI run) — not assumed — before reporting a change done.
- **Status:** graduated → CLAUDE.md § Required working style, "Validate
  after meaningful changes" (2026-08-20, landed in commit `f67d8893`, backfilled
  here 2026-08-20).
- **Related:** #559, PR #584, PR #586, #389. Two Worklog lines consumed into
  this entry (`unmeasured-cost-of-own-fix`, `optimisation-made-it-worse`).

**2026-08-20 — Twice confirmed a production deploy from a signal adjacent to the claim, not the write that actually mattered [verification-instrument]**
- **What happened:** Two attempts to verify that a fix (`#549` / PR #580)
  had actually deployed to production both checked the wrong instrument.
  First, the check waited for the benchmark payload's `updated_at` field to
  advance — but a scoring tick already in flight advances that field
  regardless of whether the deploy landed, so it can pass before the new
  code ever ran. Second, the check waited for a specific per-BA field to
  appear — but the scoring job writes per-BA keys first and
  `meta:benchmark_fleet` last, so the fleet-level list (which is what
  actually determines exclusion) still carried the old reason well after
  the per-BA field had already updated. Neither check was wrong about
  whether the deploy eventually landed; both were wrong about what they
  had actually proven at the moment they reported success.
- **Root cause:** A field that updates independently of the claim being
  verified — because it's driven by an in-flight tick, or written earlier
  in the same pipeline than the field that matters — can advance without
  confirming the claim. Neither check identified the *specific* terminal
  write that constitutes "this landed" before treating an easier-to-observe
  neighbor as a stand-in for it.
- **Prevention:** To verify a production state, identify the specific field
  or artifact that is the actual claim — the last key a job writes, the
  served artifact's own version/tag — and check that directly, not an
  adjacent signal that merely correlates with it.
- **Status:** graduated → CLAUDE.md § Before recommending what's next,
  "Verify a production state by its terminal write, not an adjacent signal"
  (2026-08-20, landed in commit `f67d8893`, backfilled here 2026-08-20).
- **Related:** #549, PR #580. One Worklog line consumed into this entry
  (`verification-instrument`).

**2026-08-20 — One-file-per-deposit structurally removed the Worklog merge-conflict pattern [worklog-concurrent-deposit, resolved]**
- **What happened:** Before 2026-08-18, the Worklog was a single inline list
  in `MISTAKES.md`; every depositing session had to insert at the same
  line, so concurrent deposits collided. Recorded concretely: PRs #566,
  #570, and #567 all inserted at the top of the list, producing a merge
  conflict on four separate rebase steps in one day. The resolution was
  always trivial (keep both lines) but every concurrent deposit hit it, and
  a hand-merge is exactly the kind of step that can silently drop someone
  else's evidence — which is the one thing this file's job is to not do.
- **Root cause:** A single shared file with an insert-at-the-top convention
  has no way for two concurrent writers to avoid touching the same lines.
- **Prevention:** Already shipped, not a prose rule — PR #588 moved the
  Worklog to `.mistakes/worklog/`, one file per deposit, named by UTC
  timestamp. Two sessions depositing at the same moment now write different
  filenames and cannot conflict; the failure mode is structurally
  impossible rather than merely discouraged.
- **Status:** resolved — enforced by PR #588 (`.mistakes/worklog/` directory
  structure, landed as commit `af04d7cf`). No CLAUDE.md line needed: a
  guard a structural fix already owns doesn't need a second, weaker prose
  copy for an agent to remember by hand.
- **Related:** PRs #566, #567, #570, #588.

**2026-08-20 — Four claims shipped as fact before anyone measured them, one nearly justifying an unwarranted retrain [claim-shipped-before-measurement]**
- **What happened:** Four separate incidents where a causal, quantitative, or
  attribution claim was written into a committed artifact as established fact
  before it was checked against real code or data. (1) `#549` asserted SPP is
  "diffusely sparse" and the claim was repeated in `config.py`, a
  `benchmark.py` docstring, and a pinned test; the real shape is ONE
  contiguous 341h outage since 2026-08-04, surfaced only because an unrelated
  plan happened to require fitting the classifier to real data first. (2)
  `#559` was filed, and a doc section shipped, arguing the forecast-origin
  stall came from positional AR lags on gapped series and required a
  51-BA × 3-model retrain behind the ADR-010 gate; measured hours later,
  absent rows are 7 of 110,704 fleet-wide (0.0063%), all in one BA — the
  retrain's justification never existed. (3) An issue body's stated rationale
  ("the anchor cannot be recovered retrospectively because `lead_hours` is
  the realized lead") was repeated as fact in `docs/BENCHMARK_METHODOLOGY.md`
  and a commit message; row 0 is `anchor + 1h` by construction, so the
  anchor is exact arithmetic and a bounded reconstruction was available all
  along — corrected pre-merge, but only after a challenge, not by checking.
  (4) A shipped docstring named `filter_low_actuals` as the mechanism that
  "closes the defect" before the production measurement ran; the measurement
  showed that filter dropped 2 records fleet-wide and a different filter did
  the actual work.
- **Root cause:** In all four, the claim was plausible, came from a
  reasonable-sounding source (an issue's own stated rationale, an intuitive
  read of a bug's shape, a docstring's working theory), and was written down
  before anyone ran the check that would confirm or refute it. None were
  caught by a designed verification step — (1) and (2) surfaced only because
  an unrelated task forced a real-data measurement, (3) only because someone
  challenged it, (4) only because the measurement happened to run before
  merge. Writing the claim down first makes it load-bearing before it's true.
- **Prevention:** Before writing a causal, quantitative, or attribution claim
  into any committed artifact as established fact, check it against the
  actual code or data first — run the measurement, read the row, grep the
  function — rather than inheriting another artifact's stated rationale.
- **Status:** graduated → CLAUDE.md § "Verify a claim before writing it as
  fact" (2026-08-20). Promoted on both bars: 4 occurrences in one day, and
  occurrence (2) alone would justify severity (a doc-endorsed, gate-routed
  51-BA × 3-model retrain with no real justification, corrected only after
  shipping).
- **Related:** #549, #559 (PR #578), #547 (PR #555), PR #543 / #541. Four
  Worklog lines consumed into this entry (`evidence-verification`,
  `premise-not-measured-before-filing`, `unverified-premise`,
  `explanation-before-measurement`).

**2026-08-20 — Four checks reported success without exercising the thing they claimed to check [verification-checked-the-wrong-thing]**
- **What happened:** Four incidents, all surfaced on 2026-08-18, where a test
  or harness reported a passing/agreeing result while not actually exercising
  the real code path under test. (1) A unit assertion checked a value that
  `_patch_predict_one`'s stub hardcodes — the stub ignored `start_ts` and
  `_build_future_feature_frame` never ran for real — so the test exercised
  the stub's constant, not the code under test; caught only because the
  assertion happened to fail. (2) A per-tick replay reproducing production's
  forecast origin scored ~100% agreement on a frame that was an hour short
  throughout: `captured_at` is stamped minutes *into* the tick that records
  it, and a drift record grades the *previous* tick's payload, so the two
  bugs canceled each other; caught only because control BAs were designated
  before any output was inspected. (3) `patch("data.redis_client.redis")`
  was silently defeated by a function-local `import redis` inside the
  patched function (a real DNS lookup, 4.5s per call), and
  `monkeypatch.setattr(cache_mod, "CACHE_DB_PATH", ...)` was defeated by a
  default bound at def time — 16 threads hit the real repo-root `cache.db`
  instead. (4) The broader suite made 79 live calls per run to
  `api.eia.gov`/`archive-api.open-meteo.com` because cache-first clients fell
  through to the live API on a miss, so "mocked" tests asserted against
  today's real grid data.
- **Root cause:** A mock/patch that silently fails to intercept, or a
  measurement built from two quantities that can drift and cancel, produces
  output indistinguishable from a genuinely passing/agreeing check. Nothing
  in the check's own design verified that the substitution took effect or
  that the two measured quantities were independent — every catch here came
  from an external signal (an unrelated assertion failing, a pre-designated
  control group, anomalous timing/DB-write volume noticed during unrelated
  profiling), never from the check itself.
- **Prevention:** When a test relies on mocking/monkeypatching, assert the
  substitution was exercised (call count, absence of real I/O) rather than
  trusting the final assertion alone. For harnesses computing agreement
  between two derived measures, include a control case designed to
  disagree, checked before results are inspected.
- **Status:** graduated → CLAUDE.md § Testing, "Verify a mock actually
  intercepted" (2026-08-20).
- **Related:** #547 (PR #555), PR #558, branch `perf/ci-hermetic-and-parallel`
  (fixed by PR #586, per `git log`). Four Worklog lines consumed into this
  entry (`test-validity`, `harness-agrees-for-the-wrong-reason`,
  `mock-never-applied`, `test-hermeticity`). The specific instances are
  already fixed in code; the rule generalizes the lesson for future tests.
- **2026-08-20 addendum (5th occurrence):** A new prose guard passed clean
  against the exact break it was written to catch — `**` emphasis in the
  real instruction text defeated its regex, and one incidental, unrelated
  mention elsewhere in the file satisfied the presence check it also ran.
  Found only by mutation testing, and the first mutation run itself read as
  ambiguous between "the guard is weak" and "the mutation never applied" —
  the same shape the existing prevention already names (verify the check
  fires on a designed-to-fail case) but here applied to a text/regex guard
  rather than a mock or a harness-agreement measure. No new CLAUDE.md
  language needed; the existing prevention already generalizes to any
  guard, not just mocks. Related: PR #602. Worklog line
  `guard-missed-its-own-case` consumed into this entry.

**2026-08-20 — The audit-staleness nudge's own date bug, and the close-keyword guard's own `ask`-in-permissive-mode gap [resolved, not graduated]**
- **What happened:** Two Worklog entries documented bugs in this repo's own
  mistake-logging enforcement, both already fixed and already narrated in
  CLAUDE.md's text. (1) The SessionStart nudge compared entry dates against
  the marker date and counted only entries strictly *after* it, so any
  deposit made the same calendar day as an audit was permanently invisible —
  fixed by switching the marker to an `entries-seen` count (PR #582). (2) The
  close-keyword guard correctly returned a PreToolUse `ask` on a live
  reference, and the command then ran with no prompt, because `ask` gates
  nothing in permissive/auto-approving sessions — fixed by switching the
  backticked case to `deny`, the one case decidable with certainty (PR #579).
- **Root cause:** Both are instances of an enforcement mechanism silently not
  doing what its own author assumed — one a same-day blind spot in date
  comparison, the other a permission-mode gap between "the guard fired" and
  "the guard stopped anything." Each was caught only because per-invocation
  telemetry (`.claude/hook-activity.log`) distinguished "ran and had no
  effect" from "never ran."
- **Prevention:** Both fixes are mechanical and already shipped — the
  `entries-seen` counter and the `deny`-for-backticked split — and CLAUDE.md
  already narrates the mechanism and the reasoning in its "Where a rule is
  fully mechanical" section, so no additional prose rule is needed here.
- **Status:** resolved — enforced by PR #582 and PR #579; text already
  present in CLAUDE.md. Archived without a new diff, per the rule that a
  guard a fix already owns doesn't need a second, weaker prose copy.
- **Related:** PR #582, PR #579, `.claude/hooks/guard-close-keywords.sh`.

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
- **2026-08-20 addendum (4th–6th occurrences, rule strengthened again):** Three
  more instances landed the same day, all before the merge-safety strengthening
  below shipped. (4) A squash merge (`#594`) landed carrying its own change
  while silently removing 27 files belonging to two PRs (`#588`, `#590`)
  GitHub still reports as merged — including every `.mistakes/worklog`
  deposit and `#590`'s whole script — so `main` simply stopped containing
  work that had landed 13 minutes earlier; nothing failed, nothing warned.
  (5) A just-fixed incident was reported as still-live, and stale line
  references were published into two public issue comments, because the
  "is this still true" check ran against the read taken minutes earlier
  rather than being re-checked against `origin/main` before writing it down
  — by then `#598` had already restored the affected half. (6) Every
  "is this still true" check in a separate session ran against a local
  `main` eight commits behind `origin/main`, because the restore PR had
  merged twelve hours earlier straight into the remote and the local clone
  was never fetched — an already-fixed incident was reported as live damage
  a second time, from a different mechanism than (5) (a stale local clone,
  not a stale in-memory read). **Prevention (strengthened):** re-deriving a
  premise against "`origin/main`" is only as good as the last fetch — a
  local `main` branch is not `origin/main` until you fetch it. CLAUDE.md's
  merge-safety sentence (added same day) also now covers squash-merging
  itself as an action that must re-check its branch point, not only actions
  that read repo state. **This includes merging**: before squash-merging a
  branch — especially one touching infra/structural files — confirm its
  branch point is still current against `origin/main`, not just that CI is
  green.
- **Observation for a future pass:** six occurrences of the same root cause
  in three days, three of them on the day the rule was strengthened, is
  worth treating as evidence the prose rule alone may not be sufficient in a
  repo now under sustained concurrent multi-session use (this very audit was
  instructed to `git fetch` before starting, precisely because that can no
  longer be assumed). Worth a future pass considering a mechanical nudge
  (e.g., a SessionStart hook that runs `git fetch` and reports how far local
  is behind `origin/main`) rather than relying solely on prose discipline.
  Not drafted here — mechanism design is out of scope for a single audit
  pass, flagged for the human instead.
- **Related (addendum):** #594, #588, #590, #598, #537, #559. Three more
  Worklog lines consumed into this entry (`merged-work-silently-reverted`,
  `stale-repo-state-claim`, `verified-against-stale-local-main`).

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
- **2026-08-20 addendum (3rd occurrence, rule strengthened):**
  `docs/BACKTEST_RESULTS.md` republished its distribution table on
  2026-08-07 but left a "~4.8% ensemble headline" restatement ~90 lines
  below it *in the same file*, so the doc contradicted itself for 11 days
  while the public pages sourced from it were already correct (PR #404).
  The item-2 grep was scoped to `web/` only and would never have caught a
  restatement stranded inside the source doc itself — broadened to
  `grep -rn '<old literal>' web/ docs/` (CLAUDE.md, 2026-08-20).
- **2026-08-20 addendum (4th occurrence, resolved mechanically):**
  `test_benchmark_count_is_not_hardcoded.py`'s surface list omitted
  `docs/CANONICAL_FACTS.md` — the exact file CLAUDE.md's end-of-PR check
  routes a moved fact into, and so the likeliest place for one to be added
  (PR #538). Fixed by PR #551 adding the missing surface. No new CLAUDE.md
  text needed — the guard's own coverage was the fix. **Related:** PR #538,
  PR #551. One Worklog line consumed (`guard-coverage-gap`).

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
