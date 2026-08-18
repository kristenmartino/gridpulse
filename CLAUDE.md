# CLAUDE.md — Project Conventions for GridPulse

## Before recommending what's next

Don't rely on memory, the system prompt, or `docs/internal/NEXT_UP.md`.
Always run a state check at session start before suggesting work:

```bash
cat STATUS.md                # active focus + recent decisions + open question
gh pr list --state open      # in-flight work
gh issue list --state open   # committed queue
```

If `STATUS.md` contradicts what `gh` reports, **GitHub wins** — patch
STATUS.md in the same session. `docs/internal/NEXT_UP.md` is the
historical roadmap with acceptance criteria, **not** the operational
queue.

**Re-derive the premise of any saved plan, existing branch, or earlier
written decision before acting on it.** Each froze repo state when it was
authored and carries no signal that it has moved, and the obvious check can
reinforce the staleness — `git log branch..main` lists squash-merged commits
as unmerged, and a branch stacked on a squash-merged PR stops being an
ancestor of `main` the moment that PR lands. Check the claim against
`origin/main` or the live service; if it no longer holds, stop and report
rather than executing as written. Evidence: `MISTAKES.md` → "Three artifacts
captured `main` at authoring time".

## End-of-PR explanatory-doc check

For any non-trivial PR, before reporting "done":

1. **Architecture changed** (new service, swapped tech, removed component)?
   → update [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) + relevant
   Mermaid diagrams in same PR
2. **A cited fact moved** (value referenced across multiple docs)?
   → update [`docs/CANONICAL_FACTS.md`](docs/CANONICAL_FACTS.md) in same PR,
   **and `grep -rn '<old literal>' web/`** — the public pages at `/about` and
   `/benchmark` cite these numbers too, and a stale one there is published to
   the internet rather than merely wrong internally. `/about` shipped `4.8%`
   for four days after the 2026-08-07 retrain moved it to `4.35%`, because the
   test that claimed to check this only asserted the literal was on the page
   and never opened CANONICAL_FACTS. `tests/unit/test_public_copy_traces_to_canonical_facts.py`
   now fails on the *source* side instead; this grep catches it a step earlier.
3. **STAR-story trigger hit** (trade-off, debugging arc, surprising
   decision, recovery, scope-cut)?
   → add the story to [`docs/INTERVIEW_PREP.md`](docs/INTERVIEW_PREP.md)
   in same PR
4. **`STATUS.md` active focus, next-3, blocked-on, or open question
   changed**? → update [`STATUS.md`](STATUS.md) in the same PR
5. **Something went wrong** (measurement error, wrong assumption, wasted
   time, a trap you had to work around, a near-miss caught before it
   shipped)? → log it in [`MISTAKES.md`](MISTAKES.md). See "Mistake logging
   & rule graduation" below for format and for when it also needs a rule
   here.

Otherwise report: "no explanatory-doc impact."

### Verify every `#N` reference — issue or PR — before writing it

Before putting any `#N` in a commit message, PR body, or committed file,
confirm it is what you mean — `gh issue view <N> --json title,state`, or
`gh pr view <N> --json title,state` for a PR. A `Closes #N` written from
memory can close the *wrong* issue and leave the right one open —
silently corrupting the roadmap the project-state system exists to keep
trustworthy. This bit us on 2026-05-29 (PR #165 said `Closes #150`
when the alerting issue was #148; #150 was Prophet-interval honesty).
One `gh issue view` per reference prevents it.

Outside close-keywords the harm is provenance rather than state: a wrong
`#N` in a doc points its evidence trail at unrelated work. And a number you
have not created yet cannot be checked — issue and PR numbers race between
concurrent sessions — so write the ref after the thing exists, or name it by
branch. Evidence: `MISTAKES.md` → `[reference-verification]`.

**The backtick/quote trap (this bit us twice on 2026-05-29):** GitHub
scans *commit messages and PR bodies* for close-keywords and **ignores
backticks, code spans, and surrounding prose** — it does not scan file
contents. So even writing `` `Closes #NNN` `` inside a commit message to
*quote* or *describe* a bad reference still closes `#NNN`. The very
commit that documented the PR #165 mistake re-closed the issue it had
just reopened, because its message quoted the offending close-keyword.
Two rules follow:
1. Flip issue state with `gh issue reopen|close <N>` — a pure API action
   no later commit can undo. Never rely on keyword edits to reopen.
2. When a commit/PR must *mention* a close-keyword it does **not** intend
   to fire, break the pattern: write the keyword and number
   non-adjacently (e.g. "the close-keyword for 150") or use a placeholder
   like `#NNN`. Never put a live close-keyword next to an issue you don't
   mean to close.

## Mistake logging & rule graduation

Deposit one line to [`MISTAKES.md`](MISTAKES.md)'s Worklog whenever
something costs real time, nearly ships wrong, or would change how you'd
approach the next similar task: date, a best-guess `[category]` tag, one
sentence, a ref. **Stop there.** Don't diagnose root cause, don't propose a
fix, don't decide whether it's a repeat of anything — do that mid-task and
you're reasoning about your own mistake with the same tunnel vision that
produced it, which is exactly what makes early-generation `MISTAKES.md`
setups fail: high-effort entries that people stop writing, or biased
root-causing that ships a bad rule. Keep the deposit cheap enough that it
always happens.

**A separate pass decides everything else.** Something else — the
[`audit-mistakes-log`](.claude/skills/audit-mistakes-log/SKILL.md) skill,
run periodically with none of the depositing session's context — reads the
Worklog, tallies entries by category (grouping by *root cause*, not surface
symptom: #541's stale actuals and #542's blanked leads are one pattern, "an
instrument measured something other than what it was checking," not two),
and only once a category crosses the graduation bar drafts the full
Analyzed entry (what happened, root cause, prevention) and a candidate
CLAUDE.md diff. It surfaces the proposal; it does not merge it. **Graduate
on either bar**, both already this repo's practice before this section
existed:
- **Repeat** — the same root cause recurs (≥2 occurrences): how the
  backtick/quote trap, the `Closes #N` mistake, and the ARIMA/SARIMAX naming
  split all graduated.
- **Severity** — one incident costly or high-blast-radius enough
  (production-visible, silently wrong for days, corrupted state) that
  waiting for a repeat isn't worth the risk: how #174 and #389 graduated.

**A human approves every promotion.** A CLAUDE.md line is a durable,
overriding instruction for every future agent — it earns the same scrutiny
as any other standing-rule change, reviewed like any other doc PR. Self-edit
this file mid-task only for corrections to what's already here, never to
add a new invariant on your own authority.

**Keep the enforced set small — point at evidence instead of restating
it.** `MISTAKES.md` can grow; this file should not grow at the same rate.
Phrase a graduated rule as the invariant to hold (what to do, stated once,
plainly) and let the linked `MISTAKES.md` entry carry the narrative — don't
duplicate the story here. Prefer strengthening an existing rule's wording
over adding a near-duplicate new one. State rules as **positive invariants**
("verify X before Y," "bound what one call can cost") rather than "don't do
X" — a model reasoning from a list of prohibitions is more likely to invert
one under pressure than a model reasoning from what to do; this file's
existing rules already lean this way and new ones should too.

**When a promotion is approved:** add the concise invariant here, and mark
the source `MISTAKES.md` entry `graduated → CLAUDE.md § <heading> (<date>)`
— don't delete it; it's the evidence trail for why the rule exists. If a
fix instead makes the mistake structurally impossible (a test, an assertion,
a lint rule), the entry is marked `resolved — enforced by <X>` and does
**not** need a line here at all: a prose rule is for judgment calls nothing
mechanical can catch, and a guard a test already owns doesn't need a second,
weaker copy for an agent to remember by hand.

**Reassessment runs in both directions, not just forward:**
- *Worklog → Analyzed*: `audit-mistakes-log` re-scans for a pattern that no
  single entry crosses the bar on alone but the tally does in aggregate. It
  advances `MISTAKES.md`'s `audited-through` marker on every run, including
  runs that promote nothing — "I looked, these can wait" is a decision, and
  recording it is what keeps the reminder from repeating a decision you have
  already made until you learn to ignore it.
- *Existing rules → still true?*: a rule outlives the bug that produced it.
  When touching code a guardrail here cites, check it still describes
  reality — the same discipline "Verify every `Closes #N`" and the
  `CANONICAL_FACTS.md` grep rule already ask for. A guardrail that no
  longer matches the code is worse than none: it spends a future agent's
  attention on a solved problem while still sounding authoritative. If a
  rule's premise is gone, cut it in the same PR that removes what it was
  guarding against.

**`MISTAKES.md` is an evidence store, not a runtime lookup — it sits on disk
mostly unread.** Rules are derived from it and later audited against it; a
working session doesn't load the whole archive to check its own work,
because the archive is the thing this section exists to keep *out* of every
session's context by promoting its signal into the one file that's always
loaded. The only exceptions are the two processes that exist specifically to
touch the archive: a Worklog deposit (append one line, don't read the rest),
and `audit-mistakes-log` (reads all of it, deliberately, on its own separate
pass).

**The real-time layer is separate from the audit layer.** The
[`check-past-mistakes`](.claude/skills/check-past-mistakes/SKILL.md) skill
runs *in*-session — hooked after plan approval, and run by hand before a
commit or PR — and checks the plan or diff against the invariants already in
this file (already loaded, no extra read), so a known pattern gets caught
before it ships rather than logged after. If it catches something new, it
adds one Worklog line (never a full entry, never a re-read of the archive to
"check for similar entries" — that judgment call belongs to the audit pass)
and moves on.

**Where a rule is fully mechanical, a guard enforces it instead of a
reminder.** The close-keyword invariants above are decidable by pattern, so
`.claude/hooks/guard-close-keywords.sh` checks every `git commit` /
`gh pr create|edit` for a close keyword next to an issue number and asks for
confirmation — loudly when the keyword sits inside backticks, since GitHub
fires those too. It asks rather than blocks: a live `Closes #N` is
legitimate, and what the rule actually requires is that a human verified the
number. This is the same principle as marking an entry `resolved — enforced
by <X>`: prefer the guard, and keep prose for the judgment calls nothing
mechanical can decide.

**The enforcement layer reports on itself.** Every mistake hook appends a
line to `.claude/hook-activity.log` (gitignored, local, safe to delete) on
every invocation, silent runs included — so "no cause to fire" and "never
ran" stay distinguishable, which is exactly what this repo has failed to
tell apart before. `audit-mistakes-log` reads it and reports whether
enforcement is actually running. A guard nobody can confirm is running is
indistinguishable from one that isn't.

## Start here

This repo already has multiple context layers. Read them in this order:

1. `STATUS.md` — current focus, next 3, recent decisions (canonical state)
2. `CLAUDE.md` — architecture, conventions, code standards, execution guardrails
3. `docs/internal/EXECUTION_BRIEF.md` — prioritization, redesign direction, product-shell changes, execution order
4. `README.md` — current public framing and deployment overview
5. `PRD.md` — product requirements, personas, ADRs, descoping rationale
6. `TECHNICAL_SPEC.md` — data/model/system details
7. `docs/internal/NEXT_UP.md` — historical roadmap with acceptance criteria (reference, not queue)

### Agent objective

GridPulse is evolving from a technically credible energy demand forecasting dashboard into a more cohesive **energy intelligence platform** for forecast confidence, grid visibility, and operational decision support.

Your job is to improve product coherence, positioning, and UX **without breaking core functionality**.

### Required working style
- Inspect first
- Plan briefly
- Implement in small increments
- Preserve working behavior unless explicitly asked to change it
- Validate after meaningful changes
- Summarize what changed, why, and what remains

### Guardrails
- Do not rewrite unrelated systems.
- Do not change frameworks.
- Do not destabilize data ingestion, caching, model training, or the scheduled-jobs pipeline for surface-level UI work.
- Do not remove personas, model validation, or operational context just to simplify the UI.
- Do not add unsupported marketing claims.

---

## Product context

GridPulse is a **Dash/Plotly** application for weather-aware energy forecasting and grid analysis across 51 US balancing authorities (~100% of contiguous-US lower-48 load).
It combines:
- demand data
- weather data
- multiple ML/statistical models
- backtesting and model validation
- generation and net load context
- alerts/extreme events concepts
- scenario simulation
- role-based views and briefings

### Working product framing
Use this framing unless a human directs otherwise:
- **Category:** Energy Intelligence Platform
- **Positioning:** Forecast confidence, grid visibility, and decision support
- **Tagline:** See demand sooner. Decide with confidence.

This framing should guide UI copy, navigation naming, and landing-page work. For prioritization of those changes, follow `docs/internal/EXECUTION_BRIEF.md`.

---

## Architecture

This is a **Dash/Plotly** dashboard application for weather-aware energy demand forecasting.
It uses 3 ML models (Prophet, SARIMAX, XGBoost) combined via a weighted ensemble
to forecast hourly electricity demand for 51 US balancing authorities
(~100% of contiguous-US lower-48 load). See `config.REGION_COORDINATES` for
the canonical list; expansion history is `Original 8 → V1.α +8 → V3.ζ +35`.

### Runtime split (production)
- **Cloud Run Service (`gridpulse`)** — stateless Dash/Flask web app. Reads
  from Redis only; never fetches EIA/Open-Meteo or trains models in the
  request path. When Redis is cold, renders a `warming` degraded state.
- **Cloud Run Jobs (scheduled by Cloud Scheduler)**
  - `gridpulse-scoring-job` — hourly. Fetches EIA/weather, loads latest
    models from GCS, writes forecasts + alerts + diagnostics +
    weather-correlation to Redis. Entry point: `python -m jobs scoring`.
  - `gridpulse-training-job` — daily at 04:00 UTC. Trains XGBoost/Prophet/
    SARIMAX, persists to `gs://nextera-portfolio-energy-cache/models/`,
    writes backtests to Redis. Entry point: `python -m jobs training`.
- **Model store** — GCS at `gs://nextera-portfolio-energy-cache/models/` via
  `models/persistence.py`. Layout: `{region}/{model_name}/{version}.pkl` +
  `.meta.json`, atomically pointed to by `latest.json`. Scoring job pulls
  via `load_model()` with local disk cache at `/app/trained_models/`.
- **Redis gating** — `REQUIRE_REDIS` flag (true in staging/production, false
  in development) controls whether callbacks fall back to inline compute.
  See `components/callbacks.py` for the three warming gates.

Setup + bootstrap procedure: `docs/SCHEDULED_JOBS.md`.

### Web tier I/O guardrail (added 2026-05-20 after PR #130)

The Cloud Run Service container is **stateless** and has **no trained
models, no meta.json files, no pickles on disk**. Those files live only
on the Cloud Run Job container after GCS pull. Any call from
`components/` to a function that reads from local disk or GCS will
silently fall back to a simulated/baseline path in production.

**Watchlist** (functions that have a local-disk path):

- `models.model_service.get_forecasts(region, df)` — falls back to
  `_simulate_forecasts` (noisy actuals at forward timestamps) when
  no local pickle is present. **Strict-gated since #149 (2026-05-29):**
  when `REQUIRE_REDIS` is set (staging/prod) and no trained models are
  on disk, returns `{"source": "unavailable", ...}` with NO fabricated
  series instead of simulated. Simulated output is dev/demo-only.
- `models.model_service.is_trained(region)` — pre-2026-05-20 checked
  local disk; now Redis-first with local-pickle as dev fallback
- `models.model_service.get_model_metrics(region)` — 6-layer fallback
  chain; layers 1–3 and 5 require meta.json/pickle on local disk.
  **Strict-gated since #149:** when `REQUIRE_REDIS` is set, only the
  real sources (layer 0 Redis `model_metrics` + layers 1–3 meta
  holdout) are returned; the simulated/hardcoded fallbacks (layer 4
  diagnostics, layer 5 pickle, layer 6 baseline) are skipped, so a
  cold web tier returns `{}` (warming state) rather than the
  `MAPE 1.6%`-style **baseline** that surfaced the
  [#131](https://github.com/kristenmartino/gridpulse/issues/131) bug.

**The rule for component callbacks:**

> If a component callback needs model output, feature data, or model
> metadata in the request path, **it MUST read from Redis**, not from
> `models.model_service` or anywhere that touches disk. The scoring
> job is the only writer; the web tier is read-only.

When adding a new callback that needs ML-side data, the default
question is: **"is this value in a `gridpulse:*` Redis key
somewhere?"** If yes, use it. If no, the scoring job needs to write
it first — file an issue, don't paper over with an inline compute.

Two real bugs caused by violating this guardrail, both surfaced
2026-05-20 within one session:

- [PR #130 commit 7832633](https://github.com/kristenmartino/gridpulse/pull/130/commits/7832633) — Overview hero chart was
  rendering noisy historical actuals at forward timestamps for every
  page load
- [PR #130 commit c2d6c20](https://github.com/kristenmartino/gridpulse/pull/130/commits/c2d6c20) — Overview model card badge
  always said "simulated" even when forecasts in Redis were real

### Active top-level tabs in the current shell
- Overview
- US Grid
- Forecast
- Risk
- Models

Note: The legacy modules (Historical Demand, Demand Forecast, Backtest, Generation & Net Load, Weather Correlation, Model Diagnostics, Extreme Events, and Scenario Simulator) were absorbed into the visible tabs and removed in R4.

### Key Decisions (ADRs)
- **ADR-001**: Dash + Plotly (not Streamlit) — callback architecture scales to many interaction groups
- **ADR-002**: SQLite cache on Cloud Run ephemeral disk — survives across requests, acceptable to lose on recycle
- **ADR-003**: Open-Meteo (not NOAA NWS) for weather — no API key, 17 variables in one call, historical + forecast support
- **ADR-004**: Sharpened inverse-MAPE weighted ensemble (weight ∝ (1/MAPE)³, `config.ENSEMBLE_WEIGHT_EXPONENT`) — follows the best model, blends only when peers are close; refined from plain 1/MAPE after the #181 recursive re-measure. Its value is error-decorrelation, not tail-robustness.
- **ADR-005**: XGBoost as the primary single-model forecaster — strong empirical performance on the engineered-feature demand problem
- **ADR-006**: Full multi-tab architecture — overview → forecast/history → validate → grid/generation → weather/risk → simulator
- **ADR-007**: Scenario engine copies features, never mutates — pure function, safe for concurrent callbacks
- **ADR-008**: Climatology fallback for forecast horizon beyond Open-Meteo's 16-day coverage — operationally honest about extended-range uncertainty rather than fabricating signal; visibly labeled on the Forecast tab. Full rationale: PRD.md §10.
- **ADR-009**: Class-conditional anchor conditioning — broken-feed BAs anchor on their own day-ahead forecast (`forecast_mw`) for trailing unsettled hours, on a forked frame; policy driven live by the vintage classifier. Evidence: docs/ANCHOR_CONDITIONING_STUDY.md. Full rationale: PRD.md §10.
- **ADR-010**: Serve-path acceptance gate — daily retrains are a fit lottery (~27% of persisted LDWP vintages dive in the recursive serve regime; the holdout is blind to it), so the training job replays each candidate through the real serve path and a rejected candidate never repoints `latest.json`. Evidence: docs/FORECAST_DIVE_DIAGNOSIS.md. Full rationale: PRD.md §10.
- **ADR-011**: NBM-composite forecast weather — NOAA's National Blend of Models overlaid on the base fetch for future hours only, base-filled where NBM lacks variables (`NBM_FORCE_FILL_VARS`); enrichment-only, fail-open, flag `nbm_weather`. Measured +0.921 sMAPE pts through the real serve path. Evidence: docs/WEATHER_MODEL_AB.md. Full rationale: PRD.md §10.
- **ADR-013**: Precomputed scenario grid — the what-if simulator is served from
  81 real forecasts per region (9 temp × 3 wind × 3 solar, spanning the slider
  domains) computed by the scoring job over a **24h** horizon and interpolated
  trilinearly in the web tier. Keeps model inference out of the request path
  (the web-tier I/O guardrail) at unchanged slider latency. Affordable only
  because the simulator charts 24h, not the full horizon: 24 recursive steps
  against production's 384 is ~16× cheaper, ~26s added wall. **The grid is
  computed through the production recursive forecaster, not
  `scenario_engine._run_ensemble`** — a scenario from one inference path
  divided by a baseline from another reports the gap between the *paths* as
  the response to *weather*. Flag `scenario_grid`, fail-open to the #119
  heuristic. Evidence: docs/SCENARIO_GRID.md.
- **ADR-012**: Multi-point weather — each BA's footprint sampled at up to 12 static cells (`assets/multipoint_coordinates.json`, generated offline; 15 compact BAs omitted → single point) and aggregated **unweighted**; circular mean for wind direction, mode for `weather_code`, renormalizing nanmean otherwise. NBM composites per point BEFORE aggregation. Fails open to single-point at every seam (the #161 lesson). Flag `multipoint_weather`. Measured +1.14 sMAPE pts (MISO +1.77). Evidence: docs/MULTIPOINT_WEATHER_STUDY.md. Full rationale: PRD.md §10.

### Module Map
```text
app.py                    → Dash app entry point, registers layout + callbacks
config.py                 → ALL constants: regions, API URLs, thresholds, pricing tiers, feature flags
observability.py          → Structured logging + pipeline transformation logging
components/
  layout.py               → Main layout: header, persona/view selector, region selector, tab shell
  callbacks.py            → ALL Dash callbacks and shared data-loading flows
  cards.py                → Reusable KPI, welcome, alert, briefing, and supporting cards
  error_handling.py       → Confidence badges, loading spinners, empty/error states
  accessibility.py        → Colorblind palette, ARIA helpers
  insights.py             → Persona-aware insight engine
  tab_overview.py         → Overview tab (mission-control linear stack)
  tab_us_grid.py          → US Grid tab (small-multiples across the BA fleet)
  tab_demand_outlook.py   → Forecast tab (forward predictions + confidence bands)
  tab_alerts.py           → Risk tab (alerts, extreme events, stress indicators)
  tab_models.py           → Models tab (comparison, validation, diagnostics)
data/
  cache.py                → SQLite cache with TTL + stale fallback behavior where applicable
  eia_client.py           → EIA API v2: demand, generation, interchange
  weather_client.py       → Open-Meteo: 17 weather vars, historical + forecast
  noaa_client.py          → NOAA/NWS: severe weather alerts
  preprocessing.py        → Merge, align UTC, interpolate gaps <6h, flag gaps ≥6h
  feature_engineering.py  → 49 features (17 raw + 32 derived): CDD/HDD, wind power, solar CF, lags, rolling
  demo_data.py            → Synthetic data generator for offline/demo mode where explicitly used
  audit.py                → Forecast audit trail (model version, data hash, feature hash)
models/
  prophet_model.py        → Prophet with weather regressors
  arima_model.py          → SARIMAX with pmdarima auto-order
  xgboost_model.py        → XGBoost with TimeSeriesSplit CV + SHAP
  ensemble.py             → sharpened inverse-MAPE combination, weight ∝ (1/MAPE)³
  evaluation.py           → MAPE, RMSE, MAE, R², residuals, error-by-hour
  skill.py                → skill vs a seasonal-naive baseline (is the model beating "yesterday"?)
  model_service.py        → Forecast service layer: get_forecasts() with trained→simulated fallback
  training.py             → Orchestrator: train all → validate → compute weights → serialize
  pricing.py              → Merit-order pricing model
simulation/
  scenario_engine.py      → Copy → Override → Recompute → Reforecast → Delta
  presets.py              → Historical extreme scenarios
personas/
  config.py               → 4 personas: Grid Ops, Renewables, Trader, Data Scientist
  welcome.py              → Data-driven welcome messages
```

---

## Execution priorities for redesign work

When the task is related to branding, shell UX, navigation, or product coherence, prioritize in this order:

### P0
1. Product positioning and naming cleanup
2. Navigation / IA cleanup
3. Visual token and shell refresh
4. Overview redesign
5. Forecast refinement

### P1
1. Risk/alerts consolidation
2. Models/validation framing cleanup
3. Scenarios polish
4. Briefings / intelligence layer cleanup
5. Documentation alignment

### P2
1. Module/suite scaffolding
2. Broader landing-page and marketing assets
3. Mobile awareness patterns

Detailed guidance lives in `docs/internal/EXECUTION_BRIEF.md`. Use that file for sequencing and acceptance criteria.

---

## Code Standards

### Type Hints
All functions have type hints. Use `X | None` not `Optional[X]`.

### Logging
Always use structlog: `log = structlog.get_logger()`. Key-value pairs, not f-strings.
```python
log.info("data_loaded", region=region, rows=len(df))  # ✓
log.info(f"Loaded {len(df)} rows for {region}")       # ✗
```

### Docstrings
Google style. First line = what it does. Args/Returns sections for public functions.

### Commits
Format: `type(scope): description`
Types: feat, fix, refactor, test, docs, chore
Scopes: data, models, sim, personas, ui, infra

### Testing
- Unit tests: `tests/unit/test_*.py` — pure functions, no I/O
- Integration: `tests/integration/` — mocked API calls, cache roundtrips, and
  **real callback dispatch** via `tests/integration/dash_driver.py`, which
  posts to `/_dash-update-component` through a Flask test client so Dash
  binds the arguments and serialises the result. Use it for anything that
  depends on the *wiring* (an `Output` id, an `Input` order, a serialisable
  return); call the helper directly in a unit test for anything that does not.
- Smoke: `tests/smoke/` — import-level render checks (tab `layout()` builds,
  card builders construct). **Not** end-to-end: no browser, no HTTP client,
  no callbacks fire. Renamed from `tests/e2e/` in #399 because the old name
  claimed coverage that did not exist. There is still NO browser tier;
  adding one is a new tier, not a rename back.
- Run: `pytest tests/ -v --cov=data --cov=models --cov=simulation --cov=personas --cov=components`

---

## Common Patterns

### API client pattern
1. Check cache
2. Fetch from API
3. Parse
4. Cache
5. Return

On API failure: prefer serving stale real data with warning logs when available.
Do not overwrite real cached data with fake/demo data during production failure paths.

**Upstream-outage resilience (added 2026-06-04 after #174).** A *sustained*
upstream outage is different from a one-off failure: retry-to-exhaustion
(`MAX_RETRIES × timeout + backoff` per call) multiplied across 51 BAs ×
multiple endpoints can overrun a job's task timeout before per-call fallbacks
engage — this is what failed the scoring job on 2026-06-04 during a 2h EIA
504 outage. Two rules follow:
1. The fallback to last-known data (stale cache → GCS) must be **uniform
   across every endpoint** in a client, not just the primary one. `eia_client`
   writes *and* reads GCS for demand, generation, **and** interchange.
2. Guard the retry loop with a **process-local circuit breaker**
   (`data.eia_client._EIACircuitBreaker`): after K consecutive hard failures it
   fail-fasts subsequent calls straight to the fallback (periodic probe to
   recover mid-run), bounding total runtime during an outage. Per-process
   state, resets every fresh job run.

**Partial degradation is a DIFFERENT failure class (added 2026-08-04 after
#389).** Rules 1 and 2 above bound a *total* outage. They do nothing when an
upstream is merely slow and flaky — 8–15% of calls failing, the rest fine —
because the breaker counts **consecutive** hard failures and `record_success()`
resets that counter, so interleaved successes keep it closed by construction.
On 2026-08-04 every EIA call eventually succeeded (zero
`eia_max_retries_exceeded`, zero fallbacks) and the scoring job still burned
~800s → two SIGKILLs at the 1800s task timeout, paying full retry price for
work that landed. Three rules follow:

3. **Bound what ONE call can cost**, not just how many times it retries.
   `EIA_CALL_BUDGET_S` is a wall-clock ceiling across every attempt and sleep,
   with each attempt's read timeout clamped to the time remaining. Lowering the
   retry count alone does not help: 3 attempts at a 30s timeout is still ~96s.
   Split connect and read timeouts — a scalar makes a dead connect cost what a
   multi-MB body may.
4. **Bound what ONE RUN can cost, and always reach the epilogue.** A job that
   is SIGKILLed loses its bookkeeping even when the work is done: both killed
   ticks had scored ~49/51 BAs, but `write_meta("last_scored")` sits after the
   fan-out, so neither recorded any of it. Shed work at a soft deadline
   (`SCORING_SOFT_DEADLINE_FRACTION`), write the meta, exit 0.
5. **Do not answer a slow dependency by refusing to call it.** Making the
   breaker trip on a failure *rate* would trade fresh data we can still get for
   last-known-good. Make retries cheap instead. Two characterization tests in
   `tests/unit/test_eia_client.py` pin this decision.

### Feature engineering
All features are backward-looking only (no future data leakage).
Temperature in °F, wind in mph, CDD/HDD baseline = 65°F.
49 total features: 17 raw weather + 32 derived.

### Scenario engine
ALWAYS copy the feature matrix. NEVER mutate input. Recompute ALL derived features after override.

### Callbacks
All callbacks live in `components/callbacks.py`.
Tab layouts are stateless functions.
Typical flow: region-selector → data stores → tab-specific chart callbacks.

### UI changes
When making shell/UI changes:
- preserve IDs unless intentionally refactoring callbacks
- prefer relabeling/restructuring over unnecessary behavior changes
- maintain accessibility and focus states
- keep semantic color use consistent: default product identity should not rely on alert colors

---

## Deciding an experiment

Any A/B or model change routes its verdict through
[`models/rolling_eval.py`](models/rolling_eval.py). Full rationale:
[`docs/EVALUATION_POLICY.md`](docs/EVALUATION_POLICY.md).

- **Rolling origin, never one window.** A single 168h holdout reversed CAISO's
  sign between two adjacent days. Default 8 windows.
- **Optimise WAPE, publish MAPE.** MAPE is asymmetric against over-forecasting,
  so minimising it biases toward *under*-forecasting demand — the expensive
  direction for a grid. MAPE stays the published number for comparability and
  is protected as a constraint.
- **Satisficing constraints veto a win:** |bias| ≤ 2%, MAPE regression ≤ 0.5
  pts. An unmeasurable constraint counts as failed.
- **`verdict()` may refuse to decide.** Inconclusive is a valid, common, and
  publishable outcome — say so rather than reaching for the nearest number.

## Spec References
- `docs/internal/EXECUTION_BRIEF.md` — prioritization, redesign direction, execution order
- `PRD.md` — requirements, personas, descoping rationale, ADRs
- `TECHNICAL_SPEC.md` — data sources, features, models, caching
- `docs/BACKTEST_RESULTS.md` — real EIA holdout accuracy
- `docs/SCHEDULED_JOBS.md` — Cloud Run Jobs deploy + bootstrap procedure
- `tests/TEST_PYRAMID.md` — coverage targets and testing strategy
- `specs/archive/` — historical reference only; do not treat as current truth unless cross-verified

---

## Sprint 5 / trust-and-readiness conventions

### Backlog Items Implemented
- **D2**: Forecast Model Input Audit Trail — `data/audit.py`
- **I1**: Pipeline Transformation Logging — `observability.py`
- **A4+E3**: Per-Widget Data Freshness + Confidence Badges — `components/error_handling.py`
- **C9**: Meeting-Ready Mode — toggle button strips chrome for projection/PDF
- **H3**: Test Pyramid Definition — `tests/TEST_PYRAMID.md`

### Environment Config (J1)
- ALL env-specific values via `_ENV_DEFAULTS` matrix in config.py
- `ENVIRONMENT` selects tier: `development` / `staging` / `production`
- Explicit env vars ALWAYS override matrix defaults
- Never hardcode tier-specific values outside config.py

### MAPE Governance (H2)
- Use `mape_grade(mape, horizon)` — never raw threshold comparison
- Horizons: `24h`, `48h`, `72h`, `7d` — longer horizons are more tolerant
- Rollback grade means a model should be disabled and logged as an alert

### Data freshness / fallback behavior (G2)
- `data-freshness-store` tracks per-source status such as `fresh | stale | warming | demo | error`
- `warming` is emitted in production when `REQUIRE_REDIS=True` and Redis has
  no entry for the requested key yet (e.g. before the first scoring-job run,
  or after a Redis flush). The UI renders a "Data warming up" message instead
  of spinning callbacks.
- `fallback-banner` renders warnings only when degraded
- Production fallback paths should prefer stale real data over fake data when possible
- Demo data is for offline/demo contexts and must not silently overwrite real cached data during production incidents

### Bookmarks (C2)
- `dcc.Location(id="url")` manages query params
- Supported params: `region`, `persona`, `tab`
- Always validate param values against known sets before applying

### Feature Flags
- All flags in `config.FEATURE_FLAGS`
- Use `config.feature_enabled(flag)` — unknown flags default to **False**
  (fail-closed since PR-G8 / #145, 2026-05-29: a typo must never silently
  *enable* behavior). Register every flag you read in `FEATURE_FLAGS`.

---

## Final instruction

Treat GridPulse as a technically serious product that is being upgraded into a clearer, calmer, more premium platform experience.

Do not optimize for superficial polish alone.
Optimize for:
- product coherence
- trustworthy workflows
- strong information hierarchy
- preserved technical credibility
- execution in small, safe steps
