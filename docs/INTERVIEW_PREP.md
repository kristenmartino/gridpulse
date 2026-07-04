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

### 6. "Tell me about a time you responded to a production incident."
**The scoring job that crept into its own timeout (2026-06-01, [#171](https://github.com/kristenmartino/gridpulse/issues/171)).**

Situation: A Cloud Monitoring alert (the job-failure policy from PR-G10) fired for `gridpulse-scoring-job`. The naive read of a single scoring miss is "transient — the next hourly tick self-heals." The runbook even says so.

Investigation: It wasn't one miss — it was **four consecutive hourly ticks** (09:00–12:00 UTC), each timing out at the 900s Cloud Run task limit after reaching only ~37 of 51 BAs. That ruled out "transient." But the deployed image was unchanged (latest commit was docs-only), so it wasn't a fresh regression either. The tell was in the last *successful* run's log: `scoring_job_complete elapsed_s=854.97` — **45 seconds under the 900s cap.** The job hadn't broken; it had spent months quietly creeping toward a ceiling as features were added (ERA5 archive+forecast weather stitch, per-tick drift stats), and a normal hour of upstream-API latency variance finally tipped 854s over 900s.

Action: Two-speed response. (1) **Restore service now** — bumped the live task-timeout 900→1800s and forced a make-up run; this mirrored a precedent the *training* job had set a month earlier (7200→18000 for the same reason). (2) **Don't let the mitigation masquerade as the fix** — raising a ceiling that runtime is still growing toward just resets the clock. Filed #171 for the real fix (parallelize the per-BA fetch/score loop so runtime *drops* instead of the ceiling rising) and made the timeout bump durable in the deploy workflow so CI wouldn't silently revert it. Along the way I found `CANONICAL_FACTS.md` claimed the scoring job took "~5 minutes" — it was ~14 — and that stale fact was part of why nobody saw the creep. Corrected it.

Result: Production recovered within one make-up run; `/health` back to healthy. The incident note + this story document *why* the ceiling moved, so the next person doesn't just bump it again.

**Lesson to convey**: *A slowly-creeping resource limit gives zero warning until the instant it crosses — and then the alert tells you "the job failed," not "the job was never given margin." When you mitigate by raising a ceiling, say out loud that it's mitigation, not a fix, and file the real one — otherwise the ceiling-raise becomes permanent and you're back here in a month with a bigger number.*

### 7. "Tell me about a time you resisted the obvious fix."
**The second scoring failure that wasn't the first one ([#174](https://github.com/kristenmartino/gridpulse/issues/174)).**

Situation: Three days after I fixed a scoring-job timeout by raising the Cloud Run task limit 900→1800s, the same failure alert fired again — two hourly ticks timed out. The obvious move, the one my own last PR practically scripted, was "bump the timeout again."

Investigation: I didn't. Two facts argued against it. First, it had **self-healed** — every tick after the two failures ran a healthy ~700s, nowhere near the 1800s cap. A creeping-runtime problem doesn't recover on its own. Second, the failed run's logs showed `api.eia.gov` returning `HTTP 504` and 30s read-timeouts for ~2 hours. This wasn't our runtime growing into the ceiling (the previous incident); it was an **external EIA API outage**. Bumping the timeout would have done almost nothing — during a hard upstream outage the job can't get data no matter how long you wait.

The real defect was how the job *handled* the outage: each EIA call retried 5×30s + backoff (~150s) **before** any fallback engaged, and across 51 BAs × 3 endpoints that retry budget overran the task limit. Worse, `fetch_demand` fell back to cached GCS data but `fetch_generation`/`fetch_interchange` didn't — they just returned empty.

Action: Built a process-local **circuit breaker** — after a few consecutive hard failures it trips and fail-fasts subsequent calls straight to the fallback (with a periodic single-attempt probe to recover mid-run), so a total EIA outage completes fast on last-known data instead of timing out. Plus a uniform GCS write/read fallback for the two endpoints that lacked it. 21 tests; full suite green.

Result: The job now degrades gracefully through an EIA outage rather than dying. And I filed it as a *separate* issue from the timeout-margin one, because they're different failure modes that happened to trip the same alert.

**Lesson to convey**: *The most dangerous moment after an incident is the next incident that looks identical. The same alert fired both times, but one was "our runtime outgrew its budget" and the other was "an upstream dependency vanished" — and the right fix for the first (more headroom) does nothing for the second (graceful degradation). Read the new evidence before reaching for the last fix.*

### 8. "Walk me through a subtle bug — and how a safety improvement exposed it."
**The identity check that ate the default Models view.**

Situation: The Models tab's residual charts rendered the placeholder "No residual diagnostics available for the selected model(s)." on production for the *default* view — every model selected, the state a user lands on. The same charts worked fine in dev. The screenshot even *looked* like a CSS bug: the message was clipped to its middle slice in the narrow 3-up cards.

Investigation: The clipping was a red herring — a one-line Plotly annotation overflowing a narrow card. The real question was why the placeholder showed at all. The Models tab has a Redis fast path that serves cached ensemble residuals, gated by `if selected_models is not default_models and set(selected_models) != {"ensemble"}: return None`. That `is not` is an **identity** check. The callback passes the checklist's *value* — `["prophet","arima","xgboost","ensemble"]`, a fresh list that *equals* `default_models` but is a different object — so the identity check was always true, and the default view always fell through to the compute path.

Why it only broke in production: that compute path calls `get_forecasts`, which #149 had recently **strict-gated** to return `unavailable` (no fabricated series) under `REQUIRE_REDIS`. In dev the fallthrough still produced simulated residuals, so the charts filled in and the bug stayed invisible. The honesty fix didn't *cause* the bug — it *revealed* a latent one that fake data had been masking.

Action: Compared by value, not identity (`if set(selected_models) not in ({"ensemble"}, set(default_models))`), so the default view serves the real ensemble charts that were in Redis all along. Added a regression test that passes the default selection as a distinct object (the exact call shape that fooled the identity check), and hardened the placeholder annotation to wrap so a genuine warming state never clips again.

Result: The default Models view renders real charts in production. One-line gate fix; full unit suite green.

**Lesson to convey**: *`is` is not `==`. An identity check on a value that's reconstructed on every callback is a time bomb — it works in the one test that passes the sentinel object and fails everywhere real. And removing fake fallbacks is double-edged: it makes you honest, but it also strips the camouflage off every latent bug the fake data was quietly covering. Budget for the bugs an honesty fix will surface.*

### 9. "Tell me about a time you found a serious problem others had missed."
**I ran an adversarial review on my own project and found two P0s I was shipping.**

> ⚠️ Numbers marked `‹fill after re-measure›` come from the post-deploy training
> run — drop them in once prod re-scores. Everything else is final.

Situation: GridPulse had already been through a four-reviewer "elegance audit"
that graded the code and concluded — verbatim — that only two correctness
defects touched fabricated data. I wasn't satisfied that an *elegance* pass had
actually looked for *integrity* bugs, so I ran a second review with a different
charter: an adversarial, multi-agent sweep (finders per code territory →
independent verifiers whose job was to *refute* each finding with a runnable
repro → dedup against everything already tracked). It surfaced **two P0s the
elegance audit had explicitly ruled out.**

Investigation: The first was the one that stung. The Risk tab showed live
severe-weather alerts, badged "NOAA · LIVE." They were fabricated — the hourly
scoring job called a `generate_demo_alerts()` helper with no environment gate
and wrote canned "Heat Advisory / Wind Advisory" content to Redis every tick;
the real NOAA client existed but had *no caller*. Worse, a well-meaning earlier
PR had "fixed" the Models/alerts surfaces by making the charts *render* — which
turned an honest "no data" placeholder into a wall of fabricated-perfect output.
The safety improvement had made the dishonesty more convincing, not less.

Action: I treated "no fake data on a production surface" as the invariant and
drove it end-to-end: gate the demo generator out of prod, publish an explicit
`alerts_source="unavailable"` state instead of canned content, then wire the
real NOAA feed (with stale-cache + circuit-breaker outage resilience so a NOAA
outage degrades honestly instead of silently emptying — and never gets
disguised as "no active alerts"). I verified against the *live* API, not mocks:
CAISO and NYISO carry real alerts today, ERCOT and Florida legitimately have
none — the empty state was correct, which is exactly why it had looked like
"working."

Result: The Risk tab now shows real NWS alerts or an honest unavailable state;
the fabricated path is unreachable in prod and pinned by a test that asserts the
demo generator can't be called. Two P0s + nine P1s from that review are merged.

**Lesson to convey**: *A review finds what its charter tells it to look for. An
elegance audit will grade your abstractions and miss that you're shipping fake
alerts stamped LIVE — because "is this honest?" was never the question it was
asked. And be suspicious of the fix that makes a broken thing render: sometimes
"now it shows something" is worse than "it showed nothing," because fake data is
most dangerous when it looks finished.*

### 10. "Walk me through a subtle correctness bug in a system you own."
**Two of my three models were forecasting for the wrong clock.**

Situation: The forecast pipeline trains models daily at 04:00 UTC and scores
hourly. The review flagged that Prophet and SARIMAX forecasts might be
time-mislabeled — their values anchored to the model's frozen *training* end,
but written into Redis rows timestamped from the *current* scoring tick. XGBoost
was fine.

Investigation: The mechanism was a discarded return value. Both Prophet's and
SARIMAX's predict functions forecast forward *from where their training data
ended* — that's baked into how they generate a future window. The scoring job
took those values and wrote them positionally against `forecast_start`
(now + 1h), silently throwing away the timestamps the models actually emitted.
Because training is daily and scoring hourly, the offset grows from zero right
after a retrain toward ~23 hours just before the next one — rotating the diurnal
demand curve so an evening peak could land at midday. XGBoost escaped because
it predicts row-by-row over the forward feature frame, so it was already
anchored correctly. The kicker: the per-model *live drift* that a prior issue
had chalked up to "genuine model weakness" (some regions showing wild ARIMA
error) was, at least in part, this — the drift monitor was scoring each
prediction against the *wrong hour's* actual.

Action: I gave the predict functions an explicit `start_ts` anchor — they now
forecast across the train-end→scoring-start gap and return the window labeled
from `start_ts`, with the default path kept byte-identical so the ~10 other
callers didn't move. The scoring job feeds them a gap-spanning feature frame
(real historical weather for the gap hours, not forward-fill) so the values are
honest, not just the labels. I verified end-to-end on real Prophet + SARIMAX
against a synthetic sinusoid with an 18-hour gap: SARIMAX's first prediction
landed *exactly* on the forecast-start hour's true value, versus a
full-amplitude miss under the old anchoring.

Result: All three models now write predictions labeled with the hours they
actually predicted. Post-deploy re-measurement of per-model live drift:
`‹fill after re-measure: Prophet/ARIMA rolling_smape_7d before → after for
LDWP/AZPS›`.

**Lesson to convey**: *The bug wasn't in the math — it was in the seam between a
model that forecasts "from its training end" and a job that assumes "from now."
A returned timestamp that everyone ignores is a landmine. And when one signal
looks anomalously bad ("this model is just weak here"), check whether you're
measuring it correctly before you conclude it's broken — I nearly inherited a
wrong diagnosis of a downstream symptom.*

### 11. "Tell me about a data or statistical-integrity decision."
**My published accuracy numbers were flattering — because I measured one model differently.**

Situation: GridPulse publishes a 168-hour holdout MAPE per model, and those
numbers do real work: they drive the inverse-MAPE ensemble weights and headline
the accuracy tables in the docs. The review found the comparison was
apples-to-oranges.

Investigation: XGBoost's holdout was scored **teacher-forced one-step-ahead** —
at every holdout hour it got to see the *real* previous-hour demand as a
feature, so it was effectively answering 168 easy one-hour questions. Prophet
and SARIMAX were scored as honest 168-hour multi-step forecasts. So XGBoost's
number wasn't just better, it was measuring a *different, easier task* — which
flattered its published accuracy and tilted the ensemble weights toward it. It
also meant a separate open question ("the ensemble trails the best single model
on 47 of 51 regions") rested on a contaminated comparison.

Action: I made all three models score the holdout the same way production
actually serves — a shared recursive protocol where each step's autoregressive
features come from the model's own prior *predictions*, not observed actuals.
I extracted it into one function that's now the single source of truth for both
scoring and evaluation (so the two can't silently diverge again), and — because
this changes published numbers — I logged *both* the new recursive MAPE and the
old teacher-forced one for one release, so the shift is visible before it moves
any weights or gates. The regression test is the tell: perturbing the in-window
actuals no longer changes the forecast (proving it's genuinely recursive), while
perturbing the seed history does.

Result: XGBoost's holdout MAPE rose to a comparable basis and the ensemble
weights shifted. Measured 2026-07-03 on the production recursive holdout
(all 51 BAs): **XGBoost's median holdout MAPE went 2.32% → 4.32%**, and the
**ensemble now beats XGBoost-alone on 17 of 51 BAs, up from 4** — because once
errors are allowed to compound over the horizon, blending in Prophet and ARIMA
damps the worst single-model drift (e.g. SEC: XGBoost 38.6% → ensemble 13.6%).
The headline number roughly doubled and became one I trust. The numbers went
down as a headline and *up* in trustworthiness.

**Lesson to convey**: *The most dangerous metric is the one that's wrong in your
favor — you don't go looking for it. "XGBoost is our best model" was true, but
partly because I was grading it on an easier exam. Fixing a measurement often
makes your headline number worse and your credibility better, and that's a trade
worth making every time. I'd rather report a 4% I trust than a 2% I have to
asterisk.*

### 12. "Tell me about a time you improved a model with evidence."
**We improved the ensemble — and learned it helped for a different reason than we thought.**

Situation: GridPulse serves an inverse-MAPE weighted ensemble of three models
(ADR-004), justified in the ADR as "tail variance-reduction." After I fixed a
dishonest holdout measurement (teacher-forced → recursive), the ensemble visibly
trailed the best single model — but the open question about it (#181) still
rested on the old contaminated numbers.

Task: Decide, on honest data, whether inverse-MAPE weighting was still the right
default — and if not, what to change.

Action: I regenerated the per-model recursive holdout series for all 51 BAs —
reusing the production holdout code so the numbers were production-faithful — and
swept the weighting exponent from equal-weight to winner-take-all. Crucially I
didn't just minimize error on one window: I ran two generalization tests (a
temporal split and an even/odd-hour split) so I wouldn't tune the exponent to one
week's noise, and I independently reimplemented the whole evaluation to catch my
own bugs (my first adversarial-verification pass had itself failed silently, so I
re-verified by hand).

Result: Plain inverse-MAPE (`k=1`) was too soft — it kept 15–30% weight on models
running 3–5× worse, and it was beaten by a sharper exponent (`k=3`) on 47 of 51
BAs; only 2 BAs preferred the current setting. Sharpening improved median and, in
the clean split, the tail too — a one-line, reversible config change. The deeper
find was mechanistic: the ADR's stated rationale was wrong. The ensemble's value
isn't tail-robustness (a single model owns the tail) — it's error-decorrelation
on the handful of BAs where two models are comparably good (CAISO 4.55% → 3.51%).
I was about to "keep it for the tail"; the data said keep it, weighted
differently, for a different reason.

**Lesson to convey**: *A default nobody has re-derived since the data changed is
worth re-deriving — the weighting had never been tuned, just assumed. And know
why your ensemble helps, not just that it helps: we thought it bought tail-
robustness and it actually bought error-decorrelation, and you only see that by
measuring the mechanism per-segment, not the headline average.*

## Practice instructions (after PR-C2 expands these)

After PR-C2 lands each story as a full 90-second narrative:

- Read each aloud, time yourself (target ~90 sec)
- Record with Loom or QuickTime; review for verbal stumbles + filler words
- Rotate which 3 you rehearse weekly so all 5 stay fresh
- Before any interview cycle: re-read all 5 stories and time them as a final check
