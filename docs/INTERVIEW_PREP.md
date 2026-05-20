# Interview Prep — GridPulse

> 5 STAR stories drawn from real events in the project's recent history.
> Each targets ~90 seconds of spoken time (~225 words at conversational
> pace). Practice notes at the bottom of each story flag the moments
> where candidates typically stumble.

## How to use this file

- **STAR format** — Situation / Task / Action / Result, plus an explicit
  "lesson to convey" so you don't trail off without the punchline
- **Stories are real** — every PR / issue link is verifiable. No
  synthesized examples. That's the whole point.
- **Practice ritual** — rotate which 3 you rehearse weekly; record on
  Loom or QuickTime; time yourself; review for filler words
- **Pre-interview** — re-read all 5, time each one as a final check.
  Most candidates run 30–50% over target on first attempt.

---

## 1. "Tell me about a trade-off you made."

**The big-bang Redis flip — [PR #114](https://github.com/kristenmartino/gridpulse/pull/114).**

**Situation:** GridPulse uses Memorystore Redis to bridge between scheduled scoring jobs and the web tier. The namespace was originally `wattcast:` — a relic of an earlier project name. We needed to rename it to `gridpulse:` across the whole system: web tier reads, scoring job writes, training job writes, every cached key.

**Task / trade-off:** Two paths. The textbook answer is a four-phase zero-downtime migration: introduce a `REDIS_KEY_PREFIX` indirection, dual-write to both namespaces in the jobs, dual-read in the web tier, then cut over. I actually started building it — got through Phase 1 and was mid-Phase 2 when I caught myself.

**Action:** Stopped and asked: *why are we building zero-downtime infrastructure for a single-tenant portfolio app?* The actual cost of cutting over was ~1 hour of "Data warming up" until the next hourly scoring tick. We already had a degraded-graceful warming state for cold-cache scenarios — exactly this case. I closed the in-progress phased-migration PR, opened a single big-bang flip PR that bumped the prefix in one commit, force-executed the scoring job manually to repopulate, and verified within 5 minutes.

**Result:** Shipped in 2 hours instead of 2 days. Documented the rationale in the closed PR so future-me doesn't reinvent the over-engineered version.

**Lesson to convey:** *Complexity should match cost-of-failure. Zero-downtime patterns are expensive to build; for low-blast-radius systems, they're often theater.*

**Practice notes:** Target 90s. Stumble points: don't get into the technical weeds of dual-write — the listener cares about the *decision*, not the protocol. The "I caught myself mid-Phase 2" moment is the key beat — slow down there.

---

## 2. "Walk me through a bug you debugged."

**The xaxis collision — [PR #117](https://github.com/kristenmartino/gridpulse/pull/117).**

**Situation:** Production threw `TypeError: plotly.Figure.update_layout() got multiple values for keyword argument 'xaxis'` on every Forecast tab load. Users saw a blank chart with the error in browser console. Tests passed locally, CI was green. This was the worst kind of bug: visible to users, invisible to tests.

**Task:** Find the source of the duplicate kwarg without rolling back, because the chart-polish PR that introduced the bug had ~40 other intended improvements I didn't want to revert.

**Action:** Three hypotheses. (H1) Plotly version mismatch between local and prod — ruled out by `pip freeze` diff. (H2) Race condition in the callback — ruled out by timing analysis. (H3) An actual duplicate kwarg pattern somewhere — searched, found it. The pattern was `update_layout(**_layout(...), xaxis=...)`. The `_layout()` helper had recently been extended to include an `xaxis` default in `PLOT_LAYOUT`. The spread (`**_layout(...)`) and the explicit `xaxis=` were now colliding. The kicker: I fixed it at the two obvious sites, then asked *what made the tests miss this?* The tests checked output shape but never actually called `update_layout()` end-to-end. I added a regression-test class that builds each chart helper end-to-end, which immediately surfaced two **more** broken sites I hadn't touched.

**Result:** Production restored within 30 minutes. Four sites fixed; regression class added; the test category that would have caught this earlier now exists.

**Lesson to convey:** *Tests that verify output shape don't catch errors that happen during the call itself. End-to-end "does this function actually run" tests are cheap insurance for code that wraps third-party APIs.*

**Practice notes:** Target 90s. The "tests checked output shape but never called the function" beat is the most important — practice landing that without rushing. Use it as the answer to "what changed in your testing approach as a result?"

---

## 3. "Tell me about a time you chose what to NOT do."

**The scenario simulator heuristic — [PR #119](https://github.com/kristenmartino/gridpulse/pull/119).**

**Situation:** User opened the Forecast tab's scenario simulator and reported that the Wind and Solar sliders moved the renewable-share KPI but produced **zero** ΔPeak — only Temperature changed the demand forecast. Three sliders, only one worked. The "Stress-test demand against weather shifts" subhead was misleading.

**Task / diagnosis:** Three hypotheses about callback wiring. (H1) Slider IDs don't match scenario-engine column names — wrong. (H2) Trained model dropped wind/solar features in feature-selection — wrong. (H3) The full physics engine isn't being called at all — *correct.*

**Action:** Read the panel code. The docstring said it out loud: *"The math is a deliberate simplification: no model re-run, just a linear demand-sensitivity factor. Real ensemble simulation lives in `simulation/scenario_engine.py`."* Two options to fix: **(A) Wire the full physics engine** to a server-side debounced callback — real model re-runs, ~200ms per slider drag. **(B) Add coefficients to the existing analytical heuristic** — zero latency, approximate. I chose B. Added two small terms: solar contributes +1.5% per 100 W/m² (AC load), wind +0.5% per 10 mph (wind chill). Calibrated against load-research norms. A regression test pins temperature as still dominant (>60% of any combined delta). Documented Option A as a parked follow-up in a separate `DEFERRED.md` if there's ever a real user.

**Result:** Wind and solar deltas now move ΔPeak visibly. All 5 scenario presets show non-zero impacts. 10 regression tests lock the behavior. Latency unchanged.

**Lesson to convey:** *Full-fidelity is the wrong answer when fidelity's cost exceeds its value at the current scale. Document both the cheap path AND the expensive path; ship the one matched to the actual user.*

**Practice notes:** Target 90s. The docstring-as-evidence moment ("the code said it out loud") is memorable — don't rush past it. End with "Option A is documented" — interviewers love the deferred-but-not-forgotten signal.

---

## 4. "Tell me about a data-quality decision."

**Import-dominated balancing authorities — V3.η.**

**Situation:** Two days after launching the 51-BA US Grid map, a user reported a value that made no physical sense: *"Highest-Stress Region: CPLW · 1071%, Lowest Reserve: -971%"* on the deployed metrics bar. The math was technically working — utilization was being computed as `demand / capacity` — but the result was nonsense.

**Task:** Figure out whether this was a bug or a deeper data-modeling problem. Spoiler: deeper.

**Action:** CPLW is Duke Energy Progress West — the NC mountain region. EIA-860M (the federal generator-capacity dataset) reports it has **42 MW** of in-territory generators. But its peak demand is **~1,261 MW.** It's a 30× import multiplier. The denominator (in-territory capacity) was meaningless for utility BAs that import nearly all their power. I shipped two complementary fixes: **(1) Data fix** — replaced EIA-860M capacity with `peak_demand × 1.15` reserve margin for 7 affected BAs (SOCO, DUK, CPLE, PSCO, FMPP, HST, CPLW). **(2) Categorical fix** — created an `IS_IMPORT_DOMINATED` frozenset for 3 BAs where the stress metric is *intrinsically* meaningless (CPLW, HST, plus SPA — the Southwestern Power federal hydro marketer). The UI suppresses these from the highest-stress KPI candidate pool and annotates hover with `· imports`. Kept the existing `_STRESS_RELIABLE_CEILING = 2.0` filter as defense-in-depth — catches structurally-importing BAs that aren't yet tagged.

**Result:** Stress KPIs reflect reality. The denominator change ships proper engineering; the categorical change ships honest UX. Test coverage added so future BAs that hit the same pattern get caught.

**Lesson to convey:** *Wrong-looking outputs are usually a denominator problem. When the math is correct but the answer is nonsense, the units or the comparator are wrong — not the formula.*

**Practice notes:** Target 90s. The "30× multiplier" number is the hook — say it deliberately. Most listeners will perk up at the specificity. The data-fix-plus-categorical-fix pairing is the engineering signal — don't conflate them.

---

## 5. "What's the biggest open issue you'd address with more time?"

**Model drift monitoring — [#121](https://github.com/kristenmartino/gridpulse/issues/121).**

**Situation:** During a routine UI walkthrough on 2026-05-19, I noticed something off on the PJM forecast. The four model options (XGBoost, Prophet, ARIMA, Ensemble) were showing forecasts that spanned a **47 GW range** at the same hour — XGBoost at 95k MW, Ensemble at 106k, Prophet at 122k, ARIMA at 142k. Recent actuals had ended at ~125–130 GW. XGBoost was predicting a sharp drop; ARIMA was predicting we keep climbing.

**Diagnosis:** Holdout MAPE — the basis for our inverse-MAPE ensemble weights — is computed at **training time only**. The daily training run sets the weights, and they stay frozen until the next training. Between trainings, individual models can drift relative to live actuals — and the ensemble silently weights them as if they hadn't. The 47 GW spread is exactly the symptom you'd expect when one model has degraded against current conditions but the weights don't know.

**Action — what I'd build:** A scoring-job-side comparison that, every hourly tick, compares each model's earlier forecast against the realized actual N hours later. Persists per-model rolling-window MAPE (7d / 30d) to Redis. UI surfaces the drift indicator in the Models tab. Log line + degraded confidence badge when any model's live MAPE exceeds its holdout MAPE by more than a threshold. The ensemble weights either incorporate live MAPE or surface a stale-weights warning in the UI.

**Why I haven't built it yet:** It's ~1 week of focused work. The portfolio bar was met without it. But it's the strongest argument that the system handles **change over time** — the single biggest gap between "demo ML" and "production ML." It's also the best STAR story this project will produce when it ships.

**Lesson to convey:** *Static holdout metrics tell you how the model performed yesterday. Continuous drift monitoring tells you how it's performing right now. Closing that gap is what separates a portfolio piece from a production system.*

**Practice notes:** Target 90s. The "47 GW spread" number is the hook. Don't editorialize on why you haven't shipped — interviewers respect scope honesty more than hand-wavy excuses. End on the "portfolio vs production" framing — it gives them a follow-up to ask about.

---

## Practice schedule

- **Weekly (rotating 3 of 5)** — read aloud, time, record, review for filler words
- **Pre-interview cycle** — read all 5 in one sitting, time each. Aim for ≤95s; if any are >100s, trim before the interview
- **After any meaningful new PR** — check the CLAUDE.md end-of-PR explanatory-doc check item #3. If it triggers, append a 6th story here. STAR stories should grow over time; old ones don't need editing.
