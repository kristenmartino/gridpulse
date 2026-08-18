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

### The metric that was measuring us — a public scorecard that lost 19 BAs and named the wrong culprit (2026-08-17 → 08-18)

**S.** GridPulse publishes a benchmark scoring its forecasts against each
balancing authority's own day-ahead forecast. The public page read **25 of 51
scoreable**, down from 44, with five of the seven large ISOs — ERCOT, MISO,
NYISO, ISONE, SPP — silently absent from its fleet medians. Each was excluded
under a published reason: *"The BA publishes a day-ahead forecast for under 80%
of hours."* It had been that way for about three weeks, found by an unrelated
scheduled recheck. Nothing failed — the job succeeded and exited 0 every hour.

**T.** Find out whether the balancing authorities had really stopped
publishing, or we had stopped recording. Those need opposite fixes, and getting
it wrong on a public page in the flattering direction is the expensive error.

**A.** I refused to take the exclusion reason at face value and went to the
source. Querying EIA directly over the same 30-day window the page used: ISONE
publishes a day-ahead forecast for **100%** of hours where we had recorded
66.8%; NYISO 96.7 vs 58.0; ERCOT 93.3 vs 64.1. Swept across all 51 BAs, exactly
**two** fell below the threshold upstream, and one of those was already excluded
under a different rule.

Two more checks made it conclusive. The hours we were missing formed a
**diurnal block aligned to each BA's local early morning** — ERCOT 06-09 UTC,
NYISO 05-10, CAISO 08-09 — near-identical across unrelated BAs in different
interconnects. Twenty-six operators do not change their filing behaviour on the
same local clock; that is the fingerprint of a collector. And the values we had
missed were **genuine forecasts, not placeholders**: of 210 ERCOT / 278 NYISO /
137 PJM recovered hours, 0 / 1 / 0 had the forecast equal to the actual.

The bug was four lines. We snapshot the day-ahead forecast at the moment EIA
first publishes a metered demand value for an hour, and never look again — so a
forecast EIA published slightly later was lost permanently. The gate counted
those permanent gaps and published them as a claim about the BA.

**The half that mattered was not the fix.** Simply back-filling the missing
forecasts would have put *post-revision* values onto an arm whose entire claim
is that its numbers were observed as-issued — reintroducing a defect we had
closed two weeks earlier, at scale, in the direction that flatters whoever it
flatters. So the fill records **when** the value was observed, and the existing
freshness rule now grades on that timestamp instead of the hour's. Filling a
value and earning the as-issued claim are different things.

I also refuted the standing hypothesis, which was mine to inherit: a prior
investigation had blamed a frozen upstream cache and said the decisive test
needed in-VPC Redis access. The same records were mirrored to cloud storage —
NYISO showed **672 distinct capture timestamps out of 719**. Capture was
per-tick. Two of that investigation's other claims were also wrong by
measurement, and I corrected both in the same PR.

**R.** Verified by **replay before deploying** — the new capture applied to the
real production window for all 51 BAs: **25 → 46 scoreable, 21 restored, zero
newly excluded**, with post-fix coverage matching the independent upstream
measurement BA by BA. SPP stayed excluded at 53.6%, because that one is real.

Three things shipped alongside, because the bug was only half the failure:
the count now lives **only** in the live payload — a guard test fails any doc
that asserts it in prose, since a sentence cannot track an hourly recomputation
and "just update the number" is the fix that had already failed; the page names
its own population, so a headline cannot silently change the fleet it describes;
and an alert watches the count *and* warns on BAs approaching the threshold.

**The coda, one day later.** That warn band was justified by CAISO at 82.9% and
PJM at 81.0% — one and three points from falling out with nothing watching them.
Those numbers were produced by the broken measurement the same commit repaired:
post-fix they read 100.0% and 99.7%. The threshold shipped with evidence its own
fix had destroyed, and I only noticed because the alert fired for real within
hours — **TEC at 80.1%**, which I then confirmed against EIA directly (576 DF
hours published over the payload's own 719-hour window, 576 recorded, so the
gap was the BA's and not our collector's). Two lessons, and the second is the
one I'd lead with: a rationale written in the same breath as a fix is *measured
on the pre-fix world*, so re-derive it after; and the fastest way to find out
whether an alert is any good is to read the first thing it catches.

**Takeaway.** The exclusion reason was a *sentence about the balancing
authority* generated from a *measurement of our own pipeline*, and nothing in
the type system, the tests or the page could tell the difference. Now they are
two published fields that answer two different questions, and only one of them
is allowed to exclude anybody.

### The honesty cluster — closing 15 items, and being overruled by my own data three times (2026-07-14 → 08-10)

**S.** A tracking issue listed ~15 places where GridPulse showed a user a
number that was wrong, mislabelled, or silently synthetic. Eight had been
cleared months earlier; the issue body still listed all fifteen as open.

**T.** Clear the rest — in a codebase whose entire discipline is *do not
publish a number you cannot support*, where the failure mode I was fixing was
exactly the one I was most likely to commit while fixing it.

**A.** The first move was not a fix. It was re-verifying all 15 against the
tree, which showed 8 already done — starting cold would have meant
re-litigating finished work. Then production measurement before each design
decision, and **three times it overruled the issue's own prescription, each
time making the fix smaller**. The batch order was backwards: two items that
read as urgent were *latent* — their log lines had never fired — while two
further down were biasing every published number daily. P2-19 said the drift
signal "mixes lead horizons", implying a statistic stratified by lead; the
distribution said **97% of records are lead 1**, so a filter sufficed and the
3% tail could never have carried its own window. P2-17 prescribed a smoothed
holdout; 1,620 persisted holdout MAPEs — pulled from GCS with no retraining,
because the estimator's own history was already recorded — showed the raw
value tracks the underlying level *better* than any EWMA, **and** that a
smoothed gate would read one number while the Models tab showed another. That
is precisely the defect the cluster existed to remove, so I shipped hysteresis
on the *decision* instead: flips 14 → 8, no published figure touched.

The sharpest single finding was an estimator that **graded itself on its own
answers** — the ensemble metric fitted its weights on the same 168 hours it
was then scored against. My synthetic estimate of the bias was 0.11 points.
The fleet said **+0.46 median, +1.32 mean, worse on 33 of 51 BAs**. I had
under-called it, in the flattering direction.

And mutation verification caught **two of my own tests asserting nothing** —
fixtures where the correct rule and the mutated rule return the same answer.
Both are recorded in their own commits rather than quietly fixed.

**R.** 15 of 15, 13 PRs. Four wrong numbers off live surfaces, the published
accuracy table re-derived from the corrected estimator, and one claim
**withdrawn rather than replaced** — the "ensemble reduces tail variance"
example stopped being true on retrained data, and a claim that needs a freshly
chosen example every retrain was never a claim. What I *didn't* ship is the
part I'd lead with: the ensemble-weights half is filed as a pre-registered A/B
because stability is not accuracy, and shipping it on the stability argument
alone would have repeated the exact in-sample reasoning I had just spent a PR
removing.

**Lesson to convey**: *The failure mode you are fixing is the one you are most
likely to commit while fixing it. Three times the issue told me what to build
and the data said build less; twice my own tests passed against mutations that
broke the thing they named. Measure first, mutate your tests, and let the
instrument overrule you — including when it says ship nothing.*


### The anchor arc — a week of measurement replacing six wrong hypotheses (2026-07-11 → 07-17)

**S.** A user screenshot showed LADWP's dashboard reading "NOW 730 MW,
demand 78% below average" — and the Models tab showing 459% forecast error
with every model flagged for rollback. The forecasting looked broken.

**T.** Find the real cause and fix it — in a system where the previous four
serious bugs had all been in the *measurement layer*, not the models.

**A.** Six times, a confident hypothesis died on contact with data: my own
published #296-style mechanism theory, the "revision-robust anchor is the
biggest lever" issue framing (I wrote it; direct EIA measurement refuted
it), the day-ahead-placeholder theory (refuted by my own proxy — which was
itself then refuted by the user's "are we fetching too soon?" question), a
scheduler-offset fix (the full settle-curve measurement showed no good fetch
time exists), and finally my planned churn-class policy (the study's
class-level data killed it). The response each time was the same: build the
instrument before the fix. Vintage capture recorded what EIA first said;
when an adversarial review caught the instrument corrupting itself, the
defense (strict reads + a tombstone) turned an unexplained production ghost
into a countable event whose 30-millisecond clustering identified a
cold-start init race nobody had hypothesized. Settled-grade drift made the
displayed numbers true; the classifier segmented the fleet; the study — the
replay the original issue had declared impossible — ran against the
instrument's own records and produced the final policy: condition exactly
one class, where the win was 58%→14%, and ship everything else untouched
because the data said so.

**R.** Ten merged PRs in six days. The dashboard stopped lying everywhere
(LADWP: 459% → ~50% *true* error → falling further as conditioning lands),
every threshold in the system traces to a measured number, and the
attribution pills built mid-arc became the progress meter for the fix that
ended it. The capstone lesson, in one line: when a system measures itself
wrongly, every downstream conclusion is folklore — instrument first, and
let the instrument refute you as many times as it takes.


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
**The scenario simulator: heuristic in v1 ([#119](https://github.com/kristenmartino/gridpulse/pull/119)), real physics in v2 ([#127](https://github.com/kristenmartino/gridpulse/issues/127)).**

Two chapters, and the second is the stronger half of the answer: deciding not to build something is only credible if you also know how to build it, and when the reason for not building it has expired.

Situation: User reported the scenario simulator's wind and solar sliders produced **zero ΔPeak** while temperature worked fine. Three hypotheses about callback wiring; none right. The real answer: the panel intentionally **doesn't** call the full physics-based scenario engine. Why? Because the full engine requires loading trained models server-side on every slider drag.

Trade-off: Two options to fix —
- (A) Wire the full physics engine to a server-side debounced callback (~200ms latency per drag, real model re-run)
- (B) Add coupling coefficients to the existing analytical heuristic (~0ms latency, approximate)

Action: Chose B. Added two small terms — solar contributes +1.5% per 100 W/m² (AC load), wind +0.5% per 10 mph (wind chill) — calibrated against load-research norms. Temperature stays dominant (>60% of any combined delta) per a regression test. Documented Option C as a parked follow-up if there's ever a real user.

Result: Wind and solar deltas now produce visible (small) ΔPeak. Five scenario presets all show non-zero impacts. 10 regression tests lock the behavior. Latency unchanged.

**Lesson to convey**: *"Full fidelity" can be the wrong answer when the cost of fidelity exceeds the value at the current scale. Document the cheaper path AND the expensive path; choose the one matched to the actual user.*

#### v2 (2026-08-10): unparking it — and finding the parked estimate was wrong in both directions

Also answers **"tell me about a time you were wrong"** and **"how do you decide when to revisit a decision?"**

Situation: the heuristic had been the right call for a year. When it was time to unpark, [#127](https://github.com/kristenmartino/gridpulse/issues/127) already contained the analysis — Approach B, precompute a grid in the scoring job, costed at ~200 ms per ensemble re-run giving **~40 minutes** added to an hourly job. That reads as "too expensive, stay parked."

Action: re-measured the assumption instead of re-litigating the decision. **Both numbers in it were wrong, in opposite directions.**
- The per-run cost was far *worse*: the measured production figure is **763 ms/BA** for XGBoost's 384-step recursive loop alone (~11 s including Prophet). So the issue's own plan was *hours*, and 40 minutes was already fatal against an 1800 s task timeout the job has actually hit.
- But **the horizon was never questioned.** The simulator charts **24 hours** — its own docstring said so. 24 recursive steps against 384 is ~16x cheaper. That single observation moved it from impossible to ~26 s.

A decision had sat parked for months on arithmetic nobody had re-checked, and the error that unblocked it was in the *scope* of the estimate, not its precision.

Result, and the parts that went wrong on the way:
- **First attempt measured 2.7x** (1,139 s vs a ~425 s baseline) and was reverted within the hour. Cause: each of the 80 grid cells ran its own recursive forecast — 1,920 single-row predicts per region against production's 384. Batching them into one predict call per step (24 per region) took it to **1.18x, 506/461/515 s**, inside the <600 s criterion.
- **The correctness trap was the interesting one.** The obvious implementation calls the existing scenario engine — which uses a *different* inference path from production scoring. A scenario from one path divided by a baseline from another reports **the gap between the paths as the response to weather**: more sophisticated-looking, and less truthful, than the heuristic it replaces. Third instance of that same bug class in this repo within a fortnight.
- **Verified, not assumed.** At +20 °F the heuristic returns +10.0% for every region *by construction*. The grid returns a 13.3-point spread — and for a winter-peaking BA, demand *falls* as it warms, a sign a single positive coefficient cannot produce at any parameter value.

**Lessons to convey**:
- *Re-measure the assumption, don't re-litigate the decision. The estimate that parks a piece of work is rarely re-checked, and it only has to be wrong about **scope** — here, the horizon — to be wrong by an order of magnitude.*
- *A cost estimate scaled out of a measurement taken in a different regime is a guess wearing a number's clothes. Mine were optimistic three times in one evening: 28x on the grid cost, wrong-signed on a benchmark that used a free stub, and 11 s over on the final projection.*
- *Ship it behind a flag, measure in production, revert in an hour. Some numbers genuinely cannot be obtained offline — there were no trained models in CI — and the fastest honest way to get them is to run it and be ready to undo it.*
- *Keep the check whose expected value is "obviously" a constant.* A parity cell was removed on the argument that it could only ever reproduce a row of 1.0s. Added back under challenge, **it read 0.013 on its first run** — a 720-hour rolling feature was being recomputed on a 24-row frame, contaminating every scenario by ~1%. A check that should be boring is exactly the one whose failure is legible.

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
**ensemble now beats XGBoost-alone on 21 of 51 BAs, up from 4** — because once
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

### 13. "Tell me about a subtle bug you debugged."
**A model that backtested at ~2% was drifting ~13% live — and the model was fine.**

Situation: SARIMAX's live 1-hour-ahead drift was ~13% — four-to-seven times its
training holdout. On paper it was a competent model; in production it looked
broken, and it was dragging the ensemble.

Task: Find why the *served* model was so much worse live than in backtest.

Action: I traced the serving path instead of the model. The models train daily
at 04:00 UTC but score hourly, and SARIMAX is pickled "lean" — just fitted
params plus a short training tail — then reconstructed and Kalman-filtered at
predict time, which anchors its state at the *training* boundary. So the
"1-hour-ahead" ask was really a `(gap+1)`-step-ahead forecast — up to ~24 steps —
from a stale origin that had never seen the last day of actuals. XGBoost didn't
have the problem because it reads the last real value as a lag feature; SARIMAX
had no equivalent. The fix leaned on a property of the Kalman filter most people
forget: you can `append()` the intervening actuals to advance the state through
them *without re-estimating the parameters*, then forecast a true one step.

Result: I proved it offline before touching production — on a daily-train/
hourly-score simulation across real data the 1-hour-ahead error dropped 2.5–7×
(CAISO 6.3% → 1.8%, MISO 3.6% → 0.5%), landing SARIMAX in the 0.3–1.8% range,
competitive with XGBoost. It's offline-validated (it re-combines observed data,
no retrain) and guarded to fall back to the old behavior if the state-append
ever fails.

**Lesson to convey**: *When a model's live number is 5× its backtest, suspect
the serving path before the model — here it was a state-management detail, not a
modelling gap. And know your tools: the fix wasn't more modelling, it was one
Kalman-filter operation. I also refused to ship it on the offline number alone —
it's staged to watch live drift post-deploy, because the issue flagged a second,
overlapping cause (a time-mislabel) whose share only the live re-score can settle.*

### 14. "Tell me about a time you were wrong — and how you found out."
**I published the wrong root cause for a fleet-wide forecast failure; a domain question and four pickles overturned it.**

Situation: Users spotted 30-day SARIMAX forecasts decaying to 0 MW on some
regions (Santee Cooper, Colorado) and exploding to ~2× on another (BPA). I
filed the issue with a plausible textbook mechanism — no intercept term plus
an unconstrained fit means mean-reversion to zero, or an explosive AR root on
the growth case — and started on that fix.

Task: Before implementing, a stakeholder asked whether I'd considered how US
regional weather patterns interact with each model. That question deserved
evidence, not a hand-wave — so I made the diagnosis prove itself first.

Action: I pulled the actual fitted model payloads for three degenerate regions
plus a healthy control from the production model store, reconstructed them
exactly as the serving path does, and decomposed the 720-hour forecast:
characteristic-root analysis plus a forecast run with weather inputs zeroed
out to separate the regression contribution from the time-series structure.
Both of my published claims died on contact: every AR/MA root was on the
stationary side, and the decay was *linear through* zero — drift, not
reversion, and an intercept wouldn't have fixed it. The real mechanism: the
code force-enforced seasonal differencing (D=1) "to prevent drift," but the
order search could still add d=1 — and a doubly-integrated process carries a
linear trend in its forecast function, slope estimated from the last weeks of
data, extrapolated forever. Which is exactly where the weather question
landed: the Pacific Northwest's July heat ramp became a permanent upward
line; Colorado's monsoon cooldown became a permanent decline. A 51-region
sweep on the real payloads sealed it — 8 degenerate, all 8 doubly integrated,
zero failures among the singly-integrated. I corrected the public issue
before shipping, then fixed it in layers: cap total integration at d+D≤1 on
every path including cached orders, a fit-time 720h sanity check with
safe-default refit, a serve-time per-horizon guard on every model *and* the
ensemble, and a UI state that says "withheld — failed the sanity guard"
instead of drawing fiction.

Result: The fix heals 5 of 8 regions outright at the next training run, the
guard withholds anything that still slips through, and the thresholds
validated with zero false positives across the 39 healthy regions. The wrong
mechanism never made it into code.

**Lesson to convey**: *A plausible mechanism that pattern-matches the
symptom is a hypothesis, not a diagnosis — the fix for "no intercept" and the
fix for "double integration" are different code, and shipping the first would
have left the bug alive. Payload forensics on four pickles was two hours;
it changed the fix, and it turned a stakeholder's domain instinct into the
actual causal story: the model was converting three weeks of regional weather
into a permanent climate trend.*

### 15. "Tell me about a hard bug where every obvious suspect was innocent."
**A forecast collapsed to one-third of reality off provably clean inputs — the defect was a lottery, and the fix was a gate, not a patch.**

Situation: The night a major data-quality fix went live, the Los Angeles
forecast dove to 1,302 MW overnight — deep in the range of EIA's known-bad
provisional readings and half of any real trough. Every obvious suspect had
just been engineered away: the artifact guard was excluding bad readings,
the anchor was conditioned on clean day-ahead data, and I could *prove*
from guard logs that the input frame was settled hour-by-hour. Weather was
verified normal. The model's holdout score said 8.8%; live error said 24%.

Task: Name the real mechanism with evidence — the failure had survived
every input-side fix, so guessing again would just burn another cycle.

Action: I built an ablation-ladder study against the production model
store: reproduce the live curve from mirrored data (matched within 4.3%
with the exact serving pickle), then vary one dimension per rung.
Teacher-forced predictions on real rows: 0.5% error — the model was sharp.
Serve-style frame vs holdout-style frame: identical — construction was
innocent. Then the decisive rung: replaying **all 67 persisted model
vintages** over one fixed window. Eighteen of them — 27% — collapsed, in
multi-day runs, while their neighbors were fine. Each daily retrain was an
independent draw, some draws degenerate only in the recursive serve regime,
and the published holdout was structurally blind: it scores a *freshly
retrained* model on sliced historical rows, never the deployed pickle
through the serving path. I then calibrated a persist-time acceptance gate
by replaying real vintages at their own training moments — which exposed
that a naive threshold would false-reject an honest model during a genuine
demand dip, so offset anchors judge against settled truth and only the
live anchor uses a statistical band.

Result: The gate ships in the training job: every candidate model must
replay sanely through the real serve path before the pointer moves; a bad
draw is persisted for forensics but never serves. Calibration showed the
counterfactual plainly — under the gate, the model that served the 1,302 MW
night would have been rejected and its sane predecessor kept serving; the
incident never happens. The published holdout's blind spot is closed by
construction, because the gate evaluates the exact artifact that will serve.

**Lesson to convey**: *When every input is provably clean and the output is
still wrong, stop debugging inputs — the defect can live in the artifact
itself, and it can be nondeterministic across retrains. "Which pickle?" is
a real diagnostic axis. And a validation metric that never touches the
deployed artifact on the serving path isn't a safety property; the gate
that replays the real thing is.*

### 16. "Tell me about a bug that hid because it failed *safely*."
**A feature that shipped green — tests, mutations, four CI checks — had never once executed in production, because "no measurement" and "measurement not available yet" were the same value.**

Situation: We publish a benchmark scoring our forecast against each grid
operator's own day-ahead forecast. One claim needed care: that our 48-hour
comparison hands the operators *more* lead time than their own documented
maximum of 41 hours. Rather than assert it, I made the label conditional —
granted per scoring tick only while the measured lead actually exceeded 41h,
lapsing on its own if the upstream publishing lag ever grew. It shipped with
unit tests, three assert-applied mutations, and green CI.

Task: Write the public methodology document — which meant describing, in
precise language, what the code actually did rather than what it intended.

Action: I read the implementation instead of its docstring. The helper that
measures the lead read `previous_forecast["forecast"]`, but the Redis payload
stores rows under `forecasts` — `forecast` is the *API's* reshaped name, and
I had mirrored the shape I'd most recently looked at. So it returned an empty
dict every tick, for every one of 51 regions. Empty is a legitimate state — a
region whose forecast hasn't been written yet — so the payload fell back to
the nominal label and nothing logged a problem. Following that thread turned
up a second defect: the helper measured row index H−1, while the drift
pipeline defines the 24h target as row 0 + 24h. The reported lead described
the hour *before* the one actually scored, understating every lead by exactly
an hour. Both defects failed conservative — the label withheld rather than
wrongly granted, the lead understated rather than overstated — which is
exactly why nothing looked broken.

Result: Both fixed, with five producer-level tests including a cross-module
invariant asserting the reported lead describes the same target hour the
drift snapshot actually grades, and both original bugs reproduced as
assert-applied mutations. Re-running the probe moved the measured leads from
22.8–23.0h to **23.80–23.95h**, and the conservative arm from 46.80h to
**47.80h** — the claim holds by a wider margin than we had published.

**Lesson to convey**: *The tests covered the consumer — inject a dict, assert
the label behaves. Nothing covered the producer, whose failure mode was a
legal-looking empty value. Test what produces the data, not only what reads
it. And notice when one payload has two names at two layers: a string key is
a join with no type system behind it. Fail-safe defaults are good
engineering, and they are also camouflage — a feature that never fires looks
identical to a feature with nothing to say.*

### 17. "Tell me about a time you measured the wrong thing."
**Three instruments agreed a forecast was bad. None of them could tell me it was worse than doing nothing, because none of them compared it to anything.**

Situation: A public benchmark I'd built showed one balancing authority —
a ~300 MW generation co-op — as our worst row by a wide margin: 18.0%
error against the operator's own 8.2%. Every instrument I had agreed it
was bad. The live drift monitor graded all four models `rollback`. The
holdout said 6.96%, which was its own puzzle.

Task: Work out whether it was fixable, and fix it.

Action: I formed a theory quickly and it was wrong. The ensemble weighting
had cut XGBoost to 3% because its R² was 0.085, leaving the forecast on
ARIMA and Prophet — both carrying open defects in our tracker. Tidy story,
and I nearly shipped against it. What stopped me was checking the error by
lead time first: **10.38% at one hour, 12.59% at 24h, 10.55% at 48h**. Flat.
A healthy region ran 1.70% at 1h growing to 5.84% at 24h — that's what
recursion looks like. Ours wasn't degrading with horizon; it was wrong
immediately, one hour out, with the actual known to the previous hour. So
it was never a recursion, anchoring, or weather-feature problem, and every
fix I'd been about to build addressed a mechanism that wasn't operating.

Then I asked the question none of the instruments asked: what would
*nothing* score? Seasonal-naive — "yesterday, same hour", no model, no
weather, no training — got **11.5%** where our three-model ensemble got
18.0%. I ran it across all 44 scoreable BAs: 35 beat the baseline by a
median of 0.83 points, eight sat within a point of the line, and this one
was 6.36 points *below* it — six times the next worst.

Result: The finding wasn't "a bad region", it was "a region where we
subtract information." I shipped the missing primitive — skill against a
naive baseline, as a tested module and a re-runnable study — because the
real defect was that a product built to forecast had no measurement of
whether its forecasts beat the trivial alternative. That absence is what
let a worse-than-nothing forecast sit behind a healthy-looking 6.96%
holdout, on a public page, indefinitely.

**Lesson to convey**: *Absolute error told me the model was bad. Only a
baseline could tell me it was harmful, and the difference changes the fix
entirely — you don't tune a model that should be switched off. Check the
error's shape across the axis you think is failing before you build for
that mechanism: flat-in-horizon and growing-in-horizon are different
diseases, and I had a fix half-built for the wrong one. And any forecasting
system without a skill score is reporting a number nobody can interpret —
"18%" means nothing until you know that free means 11.5%.*

### 18. "Tell me about a time the obvious fix was the wrong one."
**The bug report was "this gate is too lenient." Measuring first showed that tightening it would have hidden 7 of 51 regions — three of them by less than the noise.**

Situation: Our product hides a balancing authority when its forecast is bad
enough to be misleading. I'd found that the gate asked a strangely generous
question: it graded the **training holdout** against the **7-day** error
band, when the number we publicly publish is the **24-hour** one measured on
the live serve path. One region passed at 6.96% while every one of its four
served models graded `rollback` at 24h. The gate was answering a kinder
question than the one the product advertises, and nothing anywhere noticed
the gap.

Task: Close it. The issue I'd written myself proposed the direct fix — grade
the gate on the serve path, at the horizon we publish.

Action: Before changing a gate that removes regions from users, I measured
what the change would actually do. I pulled every region's live 24h grade
and replayed both rules across the fleet. Today's gate hides **zero of 51**.
The "correct" rule would hide **seven** — and three of those sat at 7.1%,
7.5% and 7.7% against a 7.0% threshold. Under 0.7 points of margin on a
rolling window that moves by more than that between ticks: those regions
would flicker in and out of the product on noise.

That reframed the defect. "Too lenient" was wrong. Hiding a region is a
heavy, user-visible act, and the generous question — *can we forecast this
at all?* — is the right one to gate on. The real defect was narrower and
worse: the sharp question had **no voice at all**. So I left the bar exactly
where it was and shipped the missing second opinion — a serve-path grade at
the published horizon, computed every tick from data we already wrote,
published beside the verdict, exposed on the API next to an explicit
statement of which measurement decides visibility, and logged as an alert
whenever a region passes the gate while failing at 24h.

Result: Nothing was hidden that wasn't hidden before, no region flickers,
and the silent state is now loud. I also caught a real gap in my own tests
while writing them: the existing gate test couldn't prove the new wiring
ran, because its harness starts with no drift history — so the value was
always `None` and the producer never executed. That is the exact shape of a
bug I'd shipped two weeks earlier, so I seeded the fixture and pinned the
producer, then mutated the implementation eight ways to confirm the tests
actually fail when it breaks.

**Lesson to convey**: *When a fix removes something from users, measure the
blast radius before you write it — "hides 7 of 51, three inside the noise"
is a fact that changes the design, and it took ten minutes to get. And check
what the complaint actually is: mine turned out not to be "the threshold is
wrong" but "the second measurement isn't published," which is a much smaller
and much safer change. Two instruments disagreeing is not a problem to
reconcile by moving one of them — it's information, and the fix is usually
to show both and say which one decides.*

### 19. "Tell me about a bug you decided not to fix."
**A kwarg that had been silently doing nothing for months. Fixing it made the forecast twice as bad on our largest market.**

Situation: A parameter in our ARIMA order-selection call used a name the
library had renamed two major versions earlier. Because that function
accepts arbitrary keyword arguments, the old name was accepted, swallowed,
and ignored — no error, no warning. The consequence was that the model-order
search ran on demand alone, while the model we actually fit included five
weather regressors. The code said one thing and did another, which is about
as clean a defect as you get.

Task: Fix it. The issue even had the one-word fix written in it, plus an
instruction I'd added when filing it: measure before merging.

Action: I made the change, then built the study. Both arms fit the identical
final model — only the selected order differed — over a 168-hour holdout on
ten balancing authorities, fed *known* future weather so the comparison
favoured the version I expected to win.

It lost — and I then ran it across all 51 regions rather than trusting the
ten. The full result is more interesting than the sample was: 18 regions
improved, 20 got worse, 13 were unaffected. A coin flip on the count. What
decided it was the asymmetry — **the losses totalled 61.7 points of error
against 14.9 gained**, with a worst case of one region going from 14% to 33%.
A much heavier left tail for no expected gain, at nearly triple the search
cost.

The mechanism was legible once I looked at the orders. Given five weather
columns, the in-sample selection criterion credits those columns for
variance the seasonal terms had been carrying, and prunes the seasonal
terms as redundant — one region lost both its seasonal autoregressive and
moving-average terms. That is defensible for one step ahead. Across 168
recursive hours it is not: the seasonal terms carry the daily cycle
robustly, and pointwise weather regression has to be right at every step to
replace them. The accident had been protecting us.

Result: I kept the behaviour and removed the lie — no dead parameter, and no
"corrected" one either, with the numbers and the reasoning at the call site.
The tests pin the class of defect rather than the instance: every keyword
must exist in the installed library's signature, so the next rename can't
hide the same way, and a test fails if someone re-applies the obvious fix,
pointing them at the study. I also found the search only runs on a cold
cache — orders are persisted and reused forever with no invalidation — so
this was a no-op in steady state either way, which is worth knowing before
anyone plans a change that depends on re-selection.

The fleet run also corrected me. Several per-region numbers from the ten-BA
sample did not reproduce a day later — one region reversed sign entirely, and
another's *control* error halved between adjacent windows. The conclusion
survived because it never rested on those rows, but I had quoted them as
headline evidence, and I had to go back and restate the argument on the
aggregate and the mechanism, which is what actually held.

**Lesson to convey**: *"The code doesn't do what it says" is a reason to
investigate, not a reason to know which way to change it. I had a one-word
fix, a filed issue agreeing with me, and a clean causal story — and the
measurement said the opposite, decisively. The valuable output wasn't the
diff, it was knowing that our order selection is load-bearing in a way
nobody had documented. And when you keep surprising behaviour, you owe the
next person the evidence and a test that stops them from "fixing" it — an
accident you've measured and chosen is a decision; leave it looking like an
accident and someone will helpfully undo it.*

### 20. "Tell me about a performance problem whose cause turned out to be a correctness problem."
**Our slowest test was 76% of the unit suite. It was slow because it wasn't testing the thing in its own name.**

Situation: One test took 36.8 seconds. The other 2,689 in the unit suite had
a median in the milliseconds and a maximum under 7. It was also blocking
mutation testing: it touched the feature-engineering module incidentally, so
the tool re-ran it for every one of that module's 904 mutants — 904 × ~124s,
which never finishes. It had been deselected as a documented workaround,
which meant any mutant only that test would kill was being reported as a
false survivor.

Task: Make it fast without weakening what it covered — a SQLite cache-hit
path in a forecast callback. Going in, I expected the honest answer might be
"this is inherently expensive, mark it slow and move on."

Action: Before optimising I checked what it actually executed, by spying on
`train_xgboost` while the test ran. It was called. A test named
"sqlite_cache_hit" was training a real gradient-boosted model — five-fold
time-series cross-validation plus SHAP — on every run.

The read guard requires four things to serve from cache: predictions
present, a matching cache version, and a data hash matching the live frames.
The test's mocked payload supplied only predictions. So the guard rejected
it and the function fell through to inline training. It stayed green because
its assertions were `"predictions" in result` and `isinstance(..., ndarray)`
— and inline training satisfies both. `git log -S` dated it: a commit four
months earlier had hardened the guard in the production path and left the
test behind. Three sibling tests of the same shape all set the fields
correctly, so this was one drifted outlier, not a pattern I'd have to argue
about.

Result: Completing the payload fixed the coverage and the runtime in the
same edit — **35.8s → ~1s standalone, and the whole unit suite from 91s
to 42s** on one machine, same command before and after. I then
made the class of defect fail loudly rather than slowly: the test now
asserts the exact cached values, that the cache was queried once with the
expected key, and that the in-memory cache was primed on the way out — none
of which a fall-through can fake — and patches `train_xgboost` to raise, so
if the guard ever changes again the test fails in 0.9 seconds instead of
passing in 36. The mutation-testing deselect and the "known limitation" note
it had earned both came out; that module's mutants are no longer scored
against a phantom. No `slow` marker: the slowness was a symptom, not a cost.

**Lesson to convey**: *An outlier in a runtime distribution is a question,
not a chore. Everything else in that suite was milliseconds — that gap was
the suite telling me the test was doing something categorically different
from what it claimed. The instinct is to optimise the slow thing or quarantine
it; both would have preserved the actual defect, which was three months of
zero coverage on a cache path, and quarantining it would have added a CI
filter to hide it. The general version: a test that is slow for the same
reason it is wrong will look, from the outside, exactly like a test that is
slow for a good reason. The only way to tell them apart is to check what it
runs, not how long it runs.*

### 21. "Tell me about a defense that was working, and still didn't help."
**The third scoring-job timeout, and the first with no defense at all (2026-08-04, [#389](https://github.com/kristenmartino/gridpulse/issues/389)).**

Situation: The `scoring_runtime_creep` alert fired — the early-warning guardrail I'd built after #171 precisely so a creeping runtime became a scheduled fix rather than a 3am page. It fired correctly. It was also already too late: by the time it went off, two hourly ticks had been killed at the 1800s Cloud Run cap.

Investigation: The runtime record said this wasn't creep. Daily medians had been flat at ~820s for three weeks, including that morning. Only the tail had broken — 1004s, 1283s, two kills, then a 1792s run that survived by 8 seconds. `api.eia.gov` had started returning 502/504 and 30s read timeouts at ~16:00 UTC. Nothing of ours had shipped in five days.

The decisive evidence was what was **absent**: zero `eia_max_retries_exceeded`, zero GCS fallbacks, zero stale fallbacks, zero rate limits, zero circuit trips. **Every call eventually succeeded.** No data was ever lost. The job had spent its entire budget paying retry tax on calls that worked.

That explained why the #174 circuit breaker — built after a *total* EIA outage — never fired. It counts *consecutive* hard failures, and `record_success()` zeroes the counter. A call that timed out four times and succeeded on the fifth burns ~134s and registers as a success. The breaker keys on the **shape** of a failure; this failure had only a **rate**.

Action: The tempting fix was to make the breaker rate-aware. I argued against my own instinct and didn't. Zero data was lost that day — a breaker tripping at 8–15% would have fail-fasted the remaining BAs onto last-known-good, trading fresh data we could actually *get* for runtime recoverable more cheaply elsewhere: a silent freshness regression no alert would catch. The correct response to a *slow* dependency is to make retries cheap, not to stop fetching. So the real fix was a hardcoded `30` on one line — replaced with split connect/read timeouts and a per-call wall-clock budget. Measured on a fake clock: worst case 169s → 40s. Modelled against the incident's own hourly exception counts (a model that reproduces the observed 1792s to within 4%), every hour lands back under the alert threshold.

Two things I'd have missed without digging. The killed runs had **already scored ~49 of 51 BAs** — per-BA Redis writes are incremental — but the freshness meta is written *after* the fan-out, so neither run recorded any of it. Two hours of degraded health for work that was done. That's now a soft deadline: the run sheds remaining BAs, writes what it has, and exits 0. And the runbook actively told on-call to **wait it out**, citing the breaker's self-mitigation — the documented response, while two ticks died.

I also had to overrule a concurrent session. It had opened an issue attributing the alert to a weather change that made one fetch 12× heavier. Plausible, carefully argued, and wrong: it had no production access, and the medians across that change's ship date are flat. Its instrumentation half was genuinely good and I shipped it rather than rewriting it; its cache is held until that same instrumentation can size it.

Result: Four commits, 31 new tests, every new guard verified by re-applying its mutation. Two of those tests are *characterization* tests that encode the decision not to trip the breaker, so nobody "fixes" it later without reading why.

**Lesson to convey**: *A defense keyed to the shape of a failure cannot see a failure that has only a rate — and the alert built to warn you early inherits the same blind spot, because it only evaluates the runs that survive. What generalizes across all three of these incidents isn't any dependency-specific guard; it's the budget. Bound what one call can cost, bound what one run can cost, and make the run always reach the point where it records what it did.*

**Epilogue (2026-08-05) — I refused to call the fix a win, and the instrumentation moved the next lever.** Alongside the budget we bumped the job to 8 workers on 4 vCPU. The next day two rows in the canonical-facts registry still read **"not yet measured"** against a config that was already live in production — which is precisely the failure that registry exists to prevent, so I went and measured it. Pre-bump: median **807.8s** over n=48. The three runs after the bump went live at 01:44 UTC: **1041.8 / 699.4 / 667.9s**, all 51/51.

That reads like a 17% improvement, and I published it as **inconclusive**. Three reasons: n=3; the window overlaps EIA's own recovery so nothing is cleanly attributable; and the best post-bump run, 667.9s, sits *inside* the pre-bump range, whose minimum was 665.6s. It is indistinguishable from a good day at the old config. That is the same rule `EVALUATION_POLICY.md` applies to model changes — one window is not a verdict — applied to infrastructure, where it's much less habitual.

The part that did pay was the per-phase attribution shipped during the incident. `forecast` is **60.1%** of all worker time (3085.5s of 5131.0s), and effective parallelism is already **7.7×** — 5131s of work retired in 668s of wall clock. So in-container workers are *spent*: turning that knob again buys nothing. The planned next lever had been "fan scoring across parallel tasks"; the data says the reason that would help is more vCPU, not more concurrency, and that the bigger prize is the 720-hour recursive inference that is 60% of the bill. **The measurement I took to confirm a fix ended up re-ordering the roadmap** — and the honest reading of it was the one that looked least impressive.

### 22. "Tell me about a time your own tests would have fooled you."
**A one-line dtype fix, and the tests that passed with it fully reverted (2026-08-07, follow-up to [#434](https://github.com/kristenmartino/gridpulse/pull/434)).**

Situation: A concurrent session shipped a careful one-line fix — an EIA parse branch assigned `None` where its sibling branch produced `NaN`, so pandas built an *object* column instead of float. The argument was the interesting part. It said the fix made object dtype "unreachable from inside this codebase," and that claim was load-bearing: it was the stated reason for **not** pinning ~81 mutation survivors, all of them defensive numeric coercions that exist only because nothing guaranteed the dtype at the source.

Investigation: The reasoning was sound and the reachability claim was short by three call sites. `pd.DataFrame(columns=[...])` also builds every column as object, and the client had three of those — including the terminal return of the #174 outage-fallback chain, whose own docstring called it "typed-empty." So the shape had been removed from one branch and left in three others, across all three endpoints. Worse placement than the original: those two only execute when a fetch has failed *and* the stale cache *and* the GCS parquet have both missed. The bad frame appears only during an upstream outage — least coverage, most operator pressure.

Then the part worth telling. I wrote the fix, wrote a parametrized test class for it, got 118 green, and ran the mutation — reverted all three call sites. **One test failed.** My new class called the helper directly, so it verified the helper worked while proving nothing about whether anything *used* it. Both outage-path sites were unpinned by construction. I could have shipped a helper wired to nothing and had a green suite say otherwise.

The rewrite drives the real public fetchers through the full outage scenario instead. Re-running both mutations now fails 3 tests and 1 test respectively.

Two judgment calls I'd defend. The helper **fails open** — an unregistered column degrades to the old untyped frame rather than raising, because raising inside the outage fallback converts a degraded fetch into a hard failure — and a test reads the `empty_cols=` lists back out of the source so that escape hatch can't be used silently. And the dtypes are derived by letting pandas build a real row and slicing it to zero, never hardcoded: pandas 3.0 gives `str` and `datetime64[us, UTC]` where 2.x gives `object` and `[ns]`, so a hardcoded map would have rotted on a version bump with no test able to say so.

**Lesson to convey**: *Mutating the helper is not mutating the call site — a unit test on a function proves the function works, not that the fix is connected to anything. And when a piece of reasoning is what licenses you to skip work, that reasoning is the thing to attack: "this is now unreachable" is a claim about every path in the file, not the one you just edited.*

### 23. "Tell me about a time a quality metric told you the opposite of the truth."
**A module scored 88.6% on mutation testing because its best-tested function was the one production never called (2026-08-07).**

Situation: `models/skill.py` answers the question the product exists to answer — does the forecast beat "yesterday, same hour"? After a round of mutation testing it was one of the better-scoring modules in the repo, 72.1% → 88.6% logic score, with exact-payload tests pinning `skill_payload` field by field.

Investigation: `grep -rn "skill_payload"` returned only the test file. The scoring job imported four other symbols from the module and hand-rolled its own copy of the same block inline. The two had drifted: production emitted `window_days` and `decision`, and omitted `beats_baseline` — the field the module's own docstring calls "the field worth acting on." The substitution policy consumed the inline dict and worked only because it happens to read two keys both versions carry. The module docstring also named a Redis key, `gridpulse:skill:{region}`, that was never built; the block actually ships nested inside the forecast payload and is passed through verbatim by the public API.

The uncomfortable part is the causality. The module scored *well* **because** the dead function was well tested. Every mutant of it died — there was a test for each one. The block production actually serves had no direct coverage at all, so it contributed no survivors either. Coverage and mutation testing both answer "is this code tested." Neither answers "is this code reached," and averaged together the dead half flattered the live half.

Action: I kept the function rather than deleting it, which was the closer call. The argument for deletion was that the inline block is what ships. The argument that won: `skill_payload` is a pure function of two arguments, while the inline copy lives inside a private method behind a feature flag, a Redis read, and a DataFrame — deleting the function would have moved nine tests onto a surface needing heavy mocking to test arithmetic. So the job now calls `skill_payload`, the function grew the two fields production needs, and the tests pin the shape that is served.

Before adding `beats_baseline` to a published payload I checked every consumer: the API passes the block through, no UI surface reads it. And because the block is only written on ticks where substitution fired — which requires the model to lose by a threshold — `beats_baseline` is always `False` where it appears. A test now forbids a `True`, since publishing one would mean the policy and the measurement disagreed about the same numbers.

Result: One definition. The non-finite guards came along with it, closing a seam that is currently unreachable *by accident*: the job measures over 7 days and the policy requires 168 hours, so a window can't hold enough observed hours to clear the gate while still being too sparse to compute a baseline. Those two constants are equal, in different files, with nothing connecting them. Widen the window and NaN reaches a `points > -threshold` comparison that is False for NaN, and the policy substitutes on a measurement that does not exist.

Re-measured as an A/B rather than against the published figure: 88.6% → 90.5%, and the module's last two non-`dtype` survivors died with it. One of them was the same species as the finding in story 22, inverted — there, defaults were never executed because every test passed parameters explicitly; here, a parameter's *forwarding* was never exercised because every test used the default. Both are the gap between how a function is tested and how production calls it.

The tail had one more instance. Merging this work alongside that concurrent PR, both had recomputed the same ledger's overall row against the same shared base, so **each published a total that omitted the other's kills** — 1,891 and 1,851, where summing the seven module rows gives 1,899. Neither number was wrong when written; both were wrong once the other landed.

**Lesson to convey**: *A per-function quality score is a statement about the tests, not about the system — it silently assumes every function is reachable. Before reading a strong score as reassurance, grep for a caller. And when a well-tested helper and an inline copy of it disagree, the tests will always defend the helper, because the helper is what they import; the divergence can only be found by asking which one runs.*

### 24. "Tell me about a small bug that turned out to be bigger than reported — and a test that couldn't have caught what it was named for."
**Two tabs called one model two names. The real count was three, and the assertion guarding it was true under every spelling (2026-08-11).**

Situation: A UI report, about as small as they come. The Models tab's compare-models checklist said **SARIMAX**; the Forecast tab's model selector said **ARIMA**; both carried `value="arima"`. Switch tabs, see two names for one model. The ask was to pick one.

Investigation: Grepping the two literals across `components/` found a third spelling nobody had reported — **"Arima"** — on three user-visible surfaces: the Overview hero metrics card, its ensemble caption ("most accurate single model: Arima"), and the data-scientist spotlight bar chart's x-axis. All three derived their label with `model_name.title()`. Each of those call sites *also* carried a hand-written `"XGBoost" if key == "xgboost" else ...` guard — because `.title()` gives "Xgboost" too. So the same bug had been found and patched three separate times, locally, and generalised zero times. That is the thing I'd want an interviewer to hear: the fix for "two tabs disagree" is not editing two lines, it's asking why two lines were ever independently authoritative.

The trade-off was genuine and I'll defend the losing side first. **ARIMA** is shorter, and it was already the Forecast tab's label — the tab a user is most likely to be looking at. **SARIMAX** won on two independent grounds that happened to agree. It is what the model *is*: `models/arima_model.py` fits a seasonal order with exogenous weather regressors, and both the **S** and the **X** are load-bearing and pinned by existing tests (`ARIMA_EXOG_COLS` must carry the weather columns; `d + D <= 1` guards the integrated component). And it is what the whole project outside those two tabs already published — README, PRD, TECHNICAL_SPEC §5.3, CLAUDE.md's module map, and the landing page. Calling it ARIMA in the UI understates the model to exactly the audience that would notice. `"arima"` stays the internal key — Redis payload, config, callback values — because none of that is user-visible; the fix is a boundary, not a rename.

Then the part worth telling. `test_insights_extended` asserted `"ARIMA" in comp_insights[0].text`, and its neighbours read like real coverage of model labelling. **`"ARIMA"` is a substring of `"SARIMAX"`.** The assertion was true under either spelling. It could never have failed on the inconsistency it appeared to guard, and it would not have failed on my fix either — it would have gone green and told me nothing.

Then it happened again, to me. I wrote an AST sweep forbidding superseded label literals in `components/`, and its first run failed on the canonical map itself: `"ARIMA" in "SARIMAX"` flagged the correct answer as a violation of itself. Same trap, ten minutes apart, once in a four-year-old test and once in the test written *because of* it. The fix is word-boundary matching — `\bARIMA\b` does not match inside `SARIMAX` — but the useful part is that substring containment is not the relation anyone means when they check a label.

The guard needed a second correction that only mutation testing found. My first `.title()` rule flagged receivers whose *name* mentioned a model. Reintroducing each real defect showed it caught one shape and missed three: the receivers that actually shipped were `model_name`, `primary_key`, `best_key`, and `m`. A heuristic tuned on the one example I had in front of me scored 25% against the examples I'd already fixed. Inverting it to an allowlist of the eight legitimate `.title()` calls in `components/` — fuel, grade, source, baseline-series label — makes the check unable to miss, at a cost of one line per new exception, and makes the declaration the review. Five reintroductions now fail; all five are pinned.

Result: One map, `MODEL_DISPLAY_NAMES`, read by every surface. The suite stayed green at 3,475. Worth flagging honestly: an unmerged branch moves the landing page the *other* way, to ARIMA, and its commit message explicitly noted the Models-tab disagreement and scoped it out. That hunk now needs reverting on merge — a decision made in one PR's blind spot doesn't stop being a decision.

**Lesson to convey**: *Two things disagreeing usually means N things disagree and you've found two — the report is a sample, not a census. And an assertion that is true under both the right and the wrong answer is not a weak test, it's a decoration: check that your guard fails before trusting that it passes.*

### 25. "Tell me about a bug with no symptom."

**Every unknown path on our production site returned HTTP 200 and the full app shell. No error, no 500, no log line that looked wrong — and the fix was not the one I first reached for.**

Situation: I picked up a discoverability task and started, as usual, by
measuring rather than reading. Four `curl`s at paths that should not exist —
`/robots.txt`, `/sitemap.xml`, `/this-page-does-not-exist-12345`, `/wp-admin`
— returned four identical `200 text/html`, 10,918 bytes each. The site had no
404. It had never had one.

Task: Understand why before writing anything, because a defect that produces
no error signal usually means the thing generating it is doing exactly what
it was told.

Action: It was. Dash registers a `<path:path>` catch-all pointed at its index,
so by construction every unmatched URL renders the app. Nothing was broken;
the framework's default was simply wrong for a public deployment. The obvious
fix is a path allowlist — let `/`, `/_dash-*`, `/assets/*`, `/api/v1/*`,
`/health` through and 404 the rest. I didn't write it, for two reasons. It
duplicates Flask's routing table by hand, so it goes stale the day Dash adds
an internal route or someone registers a blueprint. And a Flask
`@errorhandler(404)` cannot work here at all — the catch-all returns **200**,
so `NotFound` is never raised and the handler never fires. What I wanted was
not a list of valid paths but the question *did anything real match?* — which
the router already answers. The guard reads `request.url_rule.endpoint` and
fires only when it equals Dash's catch-all, meaning the request matched
nothing else. The allowlist maintains itself, because it *is* the routing
table. I derived the catch-all's endpoint name from
`routes_pathname_prefix` rather than hardcoding `"/<path:path>"`, so a
prefix change can't silently disarm it and restore 200-on-everything.

Two things fell out of looking closely. The `after_request` hook stamped
`Cache-Control: max-age=31536000, immutable` on *any* `/assets/` response —
including 404s, so one typo'd asset path pinned its own 404 for a year in
every browser and CDN that saw it. And the fix had a trap of its own: a 404
page is the classic place to echo the requested path back to the user, and
this codebase already has a contract that raw input is never reflected. A
test now requests `/<script>alert(1)</script>` and asserts neither the tag
nor the payload appears in the body.

Result: `/wp-admin` 404s, `no-store`, `X-Robots-Tag: noindex`; `/api/*` gets
JSON rather than an HTML page, because handing an API client a web page is a
second bug on top of the one it came for. Thirteen tests, including one class
that runs against the *real* url_map rather than a synthetic app — the guard's
entire safety argument is "it only fires when nothing matched," and that
claim is only worth anything against the routing table production actually
has. Before merging I checked the tests weren't vacuous: with the guard
removed, `/wp-admin` returns 200 again.

**Lesson to convey**: *A bug with no error signal is usually a default doing exactly what it promised, in a context nobody re-read it in. And when you find yourself about to hand-maintain a copy of something the system already knows — a route table, a schema, a list of valid states — that is the signal to go ask the system instead. The allowlist I didn't write would have been correct on the day I wrote it and wrong within a quarter.*

### 26. "Tell me about a time the experiment couldn't be trusted — and neither could the fix you were sure of."

**One question, three harnesses built to answer it, and each one defeated by a different failure in its own instrument. The BA that looked broken was the yardstick, not the forecast.**

Situation: An A/B on ensemble weighting was decided on its headline metric —
the treatment won, robustly — but blocked on a constraint: the **control** arm
over-forecast by ~6% against a ±2% bound. A harness whose control fails a
constraint cannot certify the treatment against it, so it shipped nothing.
The successor measured both arms on live production forecasts instead of a
replay. It also could not decide: control read **+9.4% per-BA**, with a handful
of balancing authorities reporting biases of +84% and +328%.

Task: Find out whether the fleet really over-forecasts, or whether we were
measuring wrong again — without touching the ±2% bound, and without excluding
the BAs whose numbers looked bad. Dropping a BA after seeing its number is
inventing your acceptance criterion from the answer, which is the exact sin
the earlier experiment had been called out for.

Action: I found that the evaluation path reused the drift pipeline's *grading*
function but none of its *filters*, and shipped that fix — measured, tested,
documented. It moved control from +9.42% to +3.26%. **I had also written into
the docstring that this closed the defect, and the production numbers came back
and said it didn't.** The filter I'd built the whole story around dropped **two
records fleet-wide**: it thresholds relative to a region's own median, so when a
BA's bad hours are a large share of a short window they *become* the median and
stop looking like outliers. The real improvement came from a different filter
than the one I'd argued for. I corrected the docstring before it merged.

That left one BA, IID, at +86% — and my justification for calling it corrupt
was that a 1-hour-ahead forecast can't be 52× worse than a 24-hour-ahead one.
True, but I'd compared against the wrong control: two different lead times.
The right comparison was the 1-hour drift path, same lead, same hours. It read
**+2.8%**. So I stopped reasoning about mechanisms and diffed the raw records
row by row. The predictions were **byte-identical** on every BA. The actuals
were not: they differed on **123 of 139 hours for IID** and **3 of 142 for
PJM**. IID's stored actual was frozen at **339 MW on every single row** while
EIA had settled those same hours at 545–867.

Result: EIA revises. The drift path re-grades its window every tick and
converges on settled values; the shadow path recorded the *preliminary* value
once and never looked again — 15–70% wrong for high-revision BAs, by the
codebase's own measurement. Three primitives in that pipeline, and the new path
had reused one. Control went to **+0.74% per-BA / +0.59% pooled**, inside the
bound for the first time, with no BA excluded and no threshold moved. The check
I trust more than either number: one BA now reads +8.92% against the
independent path's +8.96%. The diff also surfaced a separate latent bug — the
re-grade silently erases each record's lead metadata, leaving **79% of drift
records** bypassing a filter built specifically to exclude them.

**Follow-on (#542), and the part worth telling.** That latent bug was a
one-line omission next to a line that was correct: the constructor deliberately
reset sMAPE to a recompute sentinel — right, because sMAPE is *derived* from
the value that changed — and silently dropped `lead_hours`, which is a property
of the observation a revision cannot touch. The fix is one line. The work was
proving what it moves, because those numbers are published.

Two things made that measurable. First, the code fix repairs nothing already
broken — a blanked record's lead is unrecoverable inside the pipeline — so a
merge-time before/after would have read "0.00 everywhere, trust me." Second,
the lead had been in a log line the whole time, so I rebuilt the erased map
from 31 days of Cloud Logging and validated the harness against the payload
before believing any of it: reproduced the published sample count on **204 of
204** blocks. Recovery inside the 7-day window was **100%**.

**I predicted the direction and was wrong.** I expected the headline to fall
uniformly, since error grows with lead. It moved *both ways* — 17 BAs better,
20 worse, AZPS 9.78 → 11.43 and PSCO 9.64 → 10.77. That is a better result than
the one I predicted: a correction that only ever flattered the product would
deserve suspicion. And the tail is where it lives — the fleet moved 0.076 pts
while LDWP moved 3.85 and its sample count fell 25 → 15, below the threshold at
which the product is willing to show a number at all.

**Two things fell out that I did not go looking for.** The checker that
validates this very panel (`reconcile.py`) graded a *different population* than
the panel — it never lead-filtered. While the leads were blanked that mismatch
was invisible, because there was nothing to filter; repairing them took its
false-alarm count from 1 to 12, and mirroring the filter took it to 0. A
checker had been agreeing with the thing it checks by accident. And
reconstructing the leads answered a *different* open issue for free: one BA's
forecast origin had frozen for 15 ticks and then served a 23-hour-older
vintage for another 24, which is simultaneously why its "1-hour-ahead" window
carried leads out to **63 hours** and why its 24-hour horizon window was half
empty. One upstream phenomenon, two unrelated defects, and the leading
hypothesis for the second one (`_expire_pending`) was wrong.

**Lesson to convey**: *Three experiments, three instrument failures — imputed weather in the replay, stale actuals in the shadow, blanked leads in the drift. The bound everyone kept arguing about was never the problem. When a result is impossible, the instrument is the first suspect and your own last fix is the second: I shipped a correct fix with an incorrect explanation, and only the measurement caught it. And once you have two systems that should agree, stop theorising and diff them — `pred_differs=0` ended a debate that three hypotheses and a day of reasoning had not.*

### Shipping an instrument that measures nothing yet — and refusing to let it read as an answer (2026-08-18)

**S.** GridPulse's public benchmark scores its forecasts against each balancing
authority's own day-ahead forecast. A day earlier I had disclosed an awkward
dependence in it: for an hour EIA has not metered yet, EIA publishes the BA's
day-ahead value in the *actual* field, and our forecast anchors on the last
**positive** reading rather than the last **metered** one — so on those hours
our recursion is seeded with the very series we are then scored against. MISO
36.6% of hours, CAISO 26.6%, fleet median 3.3%. The disclosure had to state the
materiality as **unmeasured**, because nothing recorded which forecasts it
touched.

**T.** Make it measurable. "Unmeasured" is honest but unstable — it quietly
invites the reader to fill the gap with "small".

**A.** The measurement could not be reconstructed: the payload that would prove
a past run's anchor is overwritten every hour, and the obvious retrospective
join (anchor = target hour − lead) was both broken by a separate defect *and*
wrong in principle, because the lead is what the record *realized*, not what it
was seeded from. So the only move was to start recording. I scoped it
deliberately short — record the provenance, carry it onto the accruing drift
records, and **stop before publishing a split**, because a split computed over
zero records would recreate the exact problem the disclosure had just fixed. I
added a fourth field nobody asked for: a separate flag for the *deliberate*
anchor substitution we already do for broken feeds, because the placeholder
flag reads the raw upstream value and would otherwise have reported those
anchors as "metered" while the seed genuinely was the operator's forecast — a
true field whose framing asserts something false. And I found that the code
rebuilt these records field-by-field in two places, one of which had silently
dropped a field for weeks; rather than adding mine to the list, I replaced the
rebuild with a structural copy so the next field cannot repeat it.

**R.** Provenance now accrues from that date. The doc says the instrument
exists **and that it has measured nothing yet**, guarded by a test that fails
if a later edit lets "instrumented" quietly become "measured". Cost was checked
rather than assumed — +54 bytes a record, 12.9 → 20.9 MB fleet-wide, 0.8% of a
1 GB ceiling running at 13% — which is what let me keep the timestamp
human-readable instead of packing it.

**Lesson to convey**: *The honest gap is the unstable one. Saying "we don't
know" is correct but it decays into "it's probably fine" unless you put a clock
on it, and some measurements can only start today — data you didn't record is
gone, not merely inconvenient. The discipline is to ship the instrument and
then loudly refuse to let its existence be mistaken for a result: on the day it
lands it has measured exactly nothing, and the doc has to say so. When you find
the same class of bug you're about to reintroduce, fix the class, not your
instance of it.*
## Practice instructions (after PR-C2 expands these)

After PR-C2 lands each story as a full 90-second narrative:

- Read each aloud, time yourself (target ~90 sec)
- Record with Loom or QuickTime; review for verbal stumbles + filler words
- Rotate which 3 you rehearse weekly so all 5 stay fresh
- Before any interview cycle: re-read all 5 stories and time them as a final check

### 27. "Tell me about a time your harness agreed with production for the wrong reason."

**Two bugs in my measurement code cancelled each other and scored ~100%. What caught it wasn't inspection — it was picking control cases before looking at any numbers.**

Situation: A forecast payload for one balancing authority froze at a stale
origin for 15 hourly ticks, then published an origin **23 hours older than one
it had already served**, and held that for 24 more — quietly relabelling
40-to-63-hour-ahead predictions as one-hour-ahead. A monotonic value had gone
backwards. The origin was logged nowhere; it could only be reconstructed a tick
late, from an unrelated log line's lead field.

Task: Explain the freeze and the regression. The house method is not to reason
about candidate mechanisms — every pre-listed candidate in this project's
history has been wrong — but to diff two things that should agree and look for
structure in the disagreement. So: what the code computes, replayed against the
data each tick actually held, versus what the payload carried.

Action: I reconstructed each tick's frame from the vintage window's
first-sight timestamps and reran the real primitives. First pass: near-total
agreement, including across the freeze. I nearly believed it.

It was wrong twice. The capture timestamp is stamped a few minutes *into* the
tick that records it, so an instant comparison silently dropped the newest hour —
my frame was an hour short throughout. And a drift record grades the *previous*
tick's payload, so I was diffing values one tick apart. **Each error shifted the
answer by exactly one hour, in opposite directions.** They cancelled.

What exposed it was a decision made before any numbers existed: three balancing
authorities that had *never* frozen were designated controls, and the harness had
to reproduce them exactly or nothing downstream counted. Fixed, it scored 487 of
487 on the controls — and then, honestly, disagreed on the interesting cases.

Result: The disagreements were the finding. The freeze reproduced exactly:
autoregressive lags are computed by positional shift, so a 16-hour hole in demand
deletes the sixteen rows 24 positions later, and the origin is capped at the
feature frame's tail. The regression could **not** be reproduced — and that was
informative rather than a failure, because the vintage window is monotone by
construction and cannot record an hour being *withdrawn*. That half needed a
different instrument: a counter already in the log showed the frame holding **1**
hour where an intact frame gives 16, on 25 of 26 regressed ticks and 0 of 484
control ticks.

I shipped the narrow fix — refuse to publish an origin older than the one being
served — and deliberately did **not** fix the positional lags, because that file
is shared with the training job, so the models were trained under the same
convention and correcting it at serve time alone would be train/serve skew.

What I'd tell someone: a harness that agrees is not a harness that works.
Choose the cases it must reproduce *before* you look at any output, and prefer
controls where you're confident nothing went wrong — agreement on the
interesting cases is the thing you're trying to earn, so it can't also be the
evidence that you're entitled to it.

### 28. "Tell me about a time you found the reasoning wrong, not just the code."
**A written decision not to build a check rested on "drift has not been observed here." Drift had been observed twice, and no revisit trigger could have fired for either (2026-08-18, [#554](https://github.com/kristenmartino/gridpulse/issues/554)).**

Situation: Alert policies live as JSON in the repo and are applied to Cloud Monitoring by hand. A unit test guards that directory, but it compares committed files to a table of policy **ids** — it cannot see what GCP is actually serving. The README had already reasoned about closing that gap and decided against it, in writing: the shape was identified (a step in the hourly workflow that already had cloud credentials), the cost was known (one read-only IAM role), and it was marked **deliberately not built** because *"drift has not been observed here."* It listed three revisit triggers: someone edits a policy in the console, a second person gets project access, or a wrong id reaches `main`.

That is a good paragraph. It names the alternative, prices it, and says what would change its mind — better than most decisions get documented. It was also wrong, and the interesting part is *where*.

Investigation: The load-bearing clause was the empirical one, and it was false when it was written. Drift had happened days earlier: a runbook shipped at 4035 characters against a 4000-character API cap, so it was un-appliable from the moment it merged and the console served the previous copy for four days. Nobody classified that as drift at the time because it had been diagnosed as a different problem entirely.

More telling was the trigger list. All three triggers are about **identity** — the id — or about **who has access**. The failure that actually happened was about **content**: the id was correct and unchanged throughout, which is precisely why the guard test stayed green while the console served the wrong runbook. A "revisit if" list is a prediction about how the next failure will look, and this one predicted the wrong axis.

Action: I built the check the paragraph had already designed — hourly, in the existing workflow, one `gcloud monitoring policies list` call — comparing applied `documentation.content`, `enabled`, `validity` and `notificationChannels` against the committed files. Two deliberate choices. It **never mutates**: this API returns HTTP 200 on a failed update, so a write's own response is not evidence the write landed, and every assertion here is a read. And the headroom check — warn when a runbook is within 200 characters of the cap — **warns rather than fails**, because 3800 is a number I invented and 4000 is the one the API enforces; this directory has a standing rule to assert the enforcement, not the declaration.

Result: It failed on its first run, on a policy nobody was looking at. `scoring_runtime_creep` had been serving a 1153-character runbook since 2026-07-08 while the repo carried a 3555-character rewrite merged 2026-08-04 — **14 days**. What the console omitted was not cosmetic: the whole partial-degradation triage, including the branch that says if EIA exceptions are high but retry-exhaustion is at zero, the circuit breaker *cannot* trip because it counts consecutive failures, so runtime will keep climbing. That is the exact incident class that killed two scoring ticks on 2026-08-04. The runbook an on-call reader would have opened, during the incident it was written for, was the version from before anyone understood it.

I patched it to the committed text and verified by reading it back, not by the PATCH's status code. All 11 policies now converge.

**Lesson to convey**: *When a decision says "we haven't seen this happen," that clause is a factual claim with a date on it, and it is the one to re-check — not the design, which was fine. And be suspicious of a revisit-trigger list that is entirely about one dimension: these were all about identity, so a content failure could run for two weeks underneath them while every guard stayed green. The check that finds something on its first run is evidence the gap was real, not that you got lucky.*

### 29. "Tell me about an alert that was wrong, and the fix you argued against."
**A cost alert said the web tier was pinned at its 4-instance ceiling. It reported 7. A statistic that exceeds the ceiling it claims to measure is not a severe reading — it is the wrong reading (2026-08-18, [#581](https://github.com/kristenmartino/gridpulse/issues/581)).**

Situation: A Cloud Monitoring policy fired claiming the public Cloud Run service had sat at its `max-instances` ceiling of 4 for 15 minutes. Its runbook framed that as two things at once — a cost event on personal billing, and a traffic-flood signal against an `--allow-unauthenticated` surface. Both readings point at the same first move: go find the IP that is hammering you.

Investigation: The request rate was **0.07 req/s** and had not moved all day — before, during, or after. So I went to the metric instead, and the disproof was already sitting in the alert's own history: the value had reached **7**. The policy's documentation calls its number "the 4-instance ceiling." Nothing that measures instances against a ceiling of 4 returns 7.

The condition summed `ALIGN_MAX` across two dimensions its `groupByFields` did not preserve: the active/idle `state` label **and** `revision_name`. Per-series alignment runs before the cross-series reduction, so one instance that went active-then-idle inside a five-minute window counted twice, and each draining revision added its own on top. What had actually changed was deploy cadence — 20 merges to `main` in three hours, 19 revisions, one every five minutes. The alert was measuring rollover.

Action: Two changes, and the second is the one that matters. `ALIGN_MEAN`, because the metric is a gauge and sum-of-means equals mean-of-sums, which kills the double count. And `revision_name` in the grouping, because `--max-instances` is a per-revision bound. I checked whether the aligner alone was enough: mean-aligned but still summed across revisions peaks at **3.2** on the same window — three consecutive points over the threshold, so it would have fired anyway. Half a fix would have looked like a fix.

The harder call was the second one. The obvious follow-on is `paths-ignore`, so docs-only merges stop redeploying production — and it was what we were leaning toward. I argued against it. The hourly divergence check defines the expected commit as main's newest **CI**-green commit, and CI has no path filter, so skipping a docs deploy makes that check fail by construction, once per docs merge. Fixing *that* means teaching the checker the same path predicate — the deploy-skip rule now living in two files that must agree forever, one of which already has two production incidents to its name from exactly that kind of addition. And it fails silently in the dangerous direction: a checker that believes a commit shouldn't have deployed stops flagging one that should have. The saving was also imaginary — the repo is public, so the CI minutes are free, and Cloud Run billed for the two instances that actually ran regardless of how many revisions they spanned.

Result: Re-measured over the same three hours with the corrected aggregation, the per-revision peak is **2.0** and there are zero points over the threshold — it would not have fired. I rewrote the runbook so its first step is "confirm it is traffic": read `request_count` and list recent revisions, because deploy churn and a flood are indistinguishable from the logs. The churn itself got filed as its own issue with the measurement, including the precondition that would make `paths-ignore` safe — extract the deploy-eligibility predicate into one module both the workflow and the checker import.

I also had a number wrong mid-investigation: I told my reviewer the churn was all docs-only merges. It was 13 of 20; seven carried code, including that day's production fix. Correcting it weakened the case for the change we were considering, which is exactly when a correction is worth making out loud.

**Lesson to convey**: *An alert has two failure modes and only one of them is loud. It can miss, or it can fire on something other than what it names — and the second one costs more, because it spends the on-call reader's attention and trains them to distrust the next page. The cheapest check is whether the number is even in the range the alert's own description allows; ours was not, and had been that way since the day it was written. And when someone proposes the obvious efficiency, price what it costs to keep the detector honest, not just what it saves.*

### 30. "Tell me about a time a performance problem turned out to be a correctness problem."

**Situation**: CI took 8m45s on every PR, and because production deploys trigger
on CI's completion, it sat in front of every ship as well as every review.

**Task**: Make it faster. I started by measuring rather than guessing, which is
what turned a performance ticket into a correctness one.

**Action**: The job graph was serial — `lint` needed `security`, `test` needed
both, `docker` needed `test` — so the critical path was the *sum* of four jobs
instead of the max. That was worth ~2 minutes and it was the boring half.

The interesting half came from one number. The full local suite ran 135s wall
but only 52s of user CPU — **40% utilization**. A CPU-bound test suite does not
do that; it was waiting on something. I also noticed the cost was not uniform:
one test took 4.8s run alone, 13.6s after 220 tests, and 29.3s in the full
3,875-test run. Cost that scales with *position in the run* means something is
accumulating.

Running a single test with output unsuppressed showed it: a live fetch of 4,308
real records from `api.eia.gov`, and `429 Too Many Requests` from
`archive-api.open-meteo.com`. The suite was making **79 live API calls per run**.
The accumulation was the rate limiter warming up against us.

That reframed the whole thing. Every client fetch path is cache-first — check
cache, fetch, cache, return — so on a miss it falls through to the live API.
Tests that believed they were asserting on a fixture were asserting on today's
grid. One file's docstring literally said "All external I/O is faked" while its
interchange endpoint went out to the internet on all 13 of its tests.

Two mocks were also silently inert, and this is the part I found most
instructive. One did `patch("data.redis_client.redis")`, but the code under test
does a *function-local* `import redis`, which binds from `sys.modules` and
ignores the module attribute — so the test made a real DNS lookup for a host
named "nonexistent" and passed for the wrong reason. The other patched a module
attribute used as a *default argument value*, which Python binds at
function-definition time, so the patch could not take effect and 16 threads ran
against the real repo-root `cache.db`.

I blocked sockets outright in an autouse fixture rather than patching
`requests`, so no client can route around it via urllib or a raw socket, then
fixed each call site the guard exposed.

**Result**: Suite went 135s → 85s single-process, and CPU utilization 40% → 78%
— the remaining time is now our own code. With the network gone the suite was
genuinely CPU-bound and parallelized cleanly: **31s** on 4 workers, stable
across repeated runs, coverage gate unaffected at 91%. CI's critical path went
from ~8m45s to roughly 3 minutes. The suite also stopped depending on
third-party availability, which was the more valuable outcome — the old one
could have gone red because Open-Meteo was having a bad afternoon.

**Lesson to convey**: *"It's slow" and "it's wrong" are often the same finding
seen from different angles. The tell was a ratio, not a stopwatch: 40% CPU on a
suite that should be compute-bound, and per-test cost that grew with position in
the run. Neither is visible in a total. And a mock that doesn't apply is worse
than no mock, because it buys the confidence of isolation while quietly testing
production — both of ours passed for years while measuring the wrong thing.*
