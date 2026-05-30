# Interview Prep — GridPulse

> **Status: STAR-story stubs identified, full drafts land in PR-C2
> (next session).** Each story below is a real event from the project's
> recent history; PR-C2 expands each into ~90 seconds of spoken
> narrative + practice notes.

## How to use this file

- Each story uses **STAR** (Situation / Task / Action / Result) compressed
  to ~90 seconds of speaking time
- Stories are drawn from real recent PRs, not synthesized
- Practice ritual: rotate which 3 you rehearse weekly so all stay fresh
- Time yourself reading aloud — most candidates run 50% over target on
  first attempt

## Seed stories identified from this week's work

### 1. "Tell me about a trade-off you made."
**The big-bang Redis flip (PR [#114](https://github.com/kristenmartino/gridpulse/pull/114)).**

Situation: Needed to rename a Redis namespace (`wattcast:` → `gridpulse:`) across a multi-component system — web tier, scoring job, training job, all reading/writing the same keys.

Trade-off: Textbook answer is dual-write + parallel-read + cutover (4 phases, zero downtime). I started building that. Mid-Phase-2 I asked: *why are we taking on this complexity for a single-tenant portfolio app?* The actual cost of cutting over was ~1 hour of "Data warming up" until the next scoring tick.

Action: Closed the in-progress phased-migration PR. Opened a single big-bang flip PR. Verified production warming behavior was already in place as a degraded-graceful state, so the cutover was safe.

Result: Shipped in 2 hours instead of 2 days. Documented the rationale so future-me doesn't reinvent the over-engineered approach.

**Lesson to convey**: *Complexity costs should match the problem's actual cost-of-failure. Zero-downtime migrations are expensive to build; for a single-tenant app, they're often theater.*

### 2. "Walk me through a bug you debugged."
**The xaxis collision (PR [#117](https://github.com/kristenmartino/gridpulse/pull/117)).**

Situation: Production threw `TypeError: plotly Figure.update_layout() got multiple values for keyword argument 'xaxis'`. Tests passed locally. The Overview hero chart was breaking on every load.

Hypothesis chain:
- H1: Plotly version mismatch between local and prod → ruled out via `pip freeze` comparison
- H2: A race condition in the callback → ruled out by timing analysis
- H3: A literal duplicate kwarg somewhere → searched, found it

Action: `update_layout(**_layout(...), xaxis=...)` was the pattern. `PLOT_LAYOUT` (the shared defaults) had been extended in PR #115 to include an `xaxis` default. The spread (`**_layout(...)`) and the explicit `xaxis=` were now colliding. Fixed at 2 sites; added a regression-test class that **builds each chart helper end-to-end** so future drift catches itself.

Result: Production restored. The new tests caught a latent bug pattern that existed in 2 other helpers I hadn't touched.

**Lesson to convey**: *Tests that check output shape miss errors that happen during the calls themselves. End-to-end "does this function actually run" tests are cheap insurance.*

### 3. "Tell me about a time you chose what to NOT do."
**The scenario simulator heuristic (PR [#119](https://github.com/kristenmartino/gridpulse/pull/119)).**

Situation: User reported the scenario simulator's wind and solar sliders produced **zero ΔPeak** while temperature worked fine. Three hypotheses about callback wiring; none right. The real answer: the panel intentionally **doesn't** call the full physics-based scenario engine. Why? Because the full engine requires loading trained models server-side on every slider drag.

Trade-off: Two options to fix —
- (A) Wire the full physics engine to a server-side debounced callback (~200ms latency per drag, real model re-run)
- (B) Add coupling coefficients to the existing analytical heuristic (~0ms latency, approximate)

Action: Chose B. Added two small terms — solar contributes +1.5% per 100 W/m² (AC load), wind +0.5% per 10 mph (wind chill) — calibrated against load-research norms. Temperature stays dominant (>60% of any combined delta) per a regression test. Documented Option C as a parked follow-up if there's ever a real user.

Result: Wind and solar deltas now produce visible (small) ΔPeak. Five scenario presets all show non-zero impacts. 10 regression tests lock the behavior. Latency unchanged.

**Lesson to convey**: *"Full fidelity" can be the wrong answer when the cost of fidelity exceeds the value at the current scale. Document the cheaper path AND the expensive path; choose the one matched to the actual user.*

### 4. "Tell me about a data-quality decision."
**Import-dominated balancing authorities (V3.η).**

Situation: User reported "Highest-Stress Region: CPLW · 1071%" / "Lowest Reserve: -971%" on the deployed US Grid metrics bar. The stress chart was showing impossible values.

Investigation: CPLW (Duke Energy Progress West, NC mountains) has 42 MW of in-territory generators serving ~449 MW of demand — a 10× import multiplier. The denominator (in-territory generator capacity from EIA-860M) was meaningless for utility BAs that import nearly all their power.

Action: Two complementary fixes —
1. **Data fix**: Replaced EIA-860M capacity with `peak_demand × 1.15` reserve margin for 7 affected BAs (SOCO, DUK, CPLE, PSCO, FMPP, HST, CPLW)
2. **Categorical fix**: Created an `IS_IMPORT_DOMINATED` frozenset for 3 BAs where the stress metric is *intrinsically* meaningless (CPLW, HST, SPA — the federal hydro marketer). UI suppresses these from the highest-stress KPI candidate pool and annotates hover with `· imports`.

Kept `_STRESS_RELIABLE_CEILING = 2.0` as defense-in-depth — catches future structurally-importing BAs that aren't yet tagged.

Result: Stress KPIs now reflect reality. The denominator change ships proper engineering; the categorical change ships honest UX.

**Lesson to convey**: *Wrong-looking outputs are usually a denominator problem. When the math is correct but the answer is nonsense, the units or the comparator are wrong — not the formula.*

### 5. "What's the biggest open issue you'd address with more time?"
**Model drift monitoring ([#121](https://github.com/kristenmartino/gridpulse/issues/121)).**

Situation: 2026-05-19 UI walkthrough surfaced a 47 GW spread for PJM at the same horizon — XGBoost 95k MW, Ensemble 106k MW, Prophet 122k MW, ARIMA 142k MW. Recent actuals ended at ~125-130 GW.

Diagnosis: Holdout MAPE is **training-time only**. The inverse-MAPE ensemble weights are computed during the daily training run and stay frozen until the next training. Between trainings, individual models can drift relative to live actuals — and the ensemble silently weights them as if they hadn't.

What I'd build: A scoring-job-side comparison that, every hourly tick, compares each model's earlier forecast against the realized actual. Persists per-model rolling-window MAPE (7d / 30d). UI surfaces drift in the Models tab. Alert (log + degraded confidence badge) when any model's live MAPE exceeds its holdout MAPE by a threshold.

Why I haven't built it yet: It's ~1 week of focused work. The portfolio bar was met without it. But it's the strongest argument that the system handles **change over time**, which is the single biggest gap between "demo" and "production" ML systems.

**Lesson to convey**: *Static holdout metrics tell you how the model performed yesterday. Continuous drift monitoring tells you how it's performing right now. Closing that gap is what separates a portfolio piece from a production system.*

**Update — since shipped (PR #126 backend writer + #128 UI panel), then hardened (PR-G9 / #155).** The 1-hour-ahead drift signal is live: each scoring tick scores the prior tick's forecast against the realized actual and persists rolling 7d/30d error to `gridpulse:drift:{region}`. The follow-on robustness story is a good "know your data" beat: LDWP's live rolling MAPE sat at a persistent ~200% while a comparable BA read ~25%. Not model failure — EIA-930 occasionally publishes ~50 MW sentinel actuals for a region whose true demand is ~2.5 GW, and `|a−p|/|a|` on a 50 MW actual is a ~4,900% per-record spike that a few hours pull the whole mean toward. **The trade-off I chose: do both.** Switch the headline to bounded **sMAPE** (`200·|a−p|/(|a|+|p|)`, can't exceed 200% per record) *and* add a **region-relative low-actual filter** — drop records below 10% of the rolling-window median, not a universal MW floor, because 50 MW is an artifact for LDWP but a legitimate load for a tiny BA. Result **on representative artifact-shaped data** (through the real code path): LDWP 266.9% → ~13%, five comparison regions untouched. **Lesson**: *a metric that's correct on average can be useless on the tails; robust statistics (bounded + scale-relative) beat a global threshold when your data has structural outliers.*

**Update — verified live in production (2026-05-30); verification found a more precise truth.** Deployed CI-gated, then read the live scoring logs rather than rubber-stamping the merge. Bounded-sMAPE behaviour and no-regression on normal regions both confirmed live (dozens of regions at plausible 15–30%, sMAPE≈MAPE, 0 records filtered; the filter does fire where applicable — AZPS excluded 1). **The honest production phrasing, which protects credibility, is this:** *in representative artifact-shaped data LDWP dropped to ~13%; in live production the new sMAPE + logging confirmed bounded drift, but the observed LDWP value came from ARIMA (188% MAPE → 53% sMAPE, 0 records filtered) and appears to reflect genuine model weakness rather than near-zero-actual artifacts — the region-relative filter correctly found nothing to exclude.* The ensemble headline users actually see wasn't externally readable (Memorystore is VPC-internal; the scoring log exposes only the alphabetical-first sample model, `arima`), so I filed [#170](https://github.com/kristenmartino/gridpulse/issues/170) to log the ensemble figure and close that observability gap. **The better production-readiness story — and the real lesson:** Phase 3 didn't *make LDWP good*; it made the drift metric **honest enough to reveal what's actually happening**. An honest metric's job is not to make a number look good — it's to distinguish a data-quality artifact from a genuine model weakness, and here the new metric did exactly that. A verification pass should find a *more precise truth*, not rubber-stamp the work. Never repeat "LDWP → ~13%" as a production fact: it was the synthetic case.

## Practice instructions (after PR-C2 expands these)

After PR-C2 lands each story as a full 90-second narrative:

- Read each aloud, time yourself (target ~90 sec)
- Record with Loom or QuickTime; review for verbal stumbles + filler words
- Rotate which 3 you rehearse weekly so all 5 stay fresh
- Before any interview cycle: re-read all 5 stories and time them as a final check
