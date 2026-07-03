# Overview Decision Layer — Strategy & Proposal

_Drafted 2026-07-02. Author: agent, synthesizing user product notes + a 7-agent
feasibility investigation (`docs/internal/CRITICAL_REVIEW_2026-07.md` P2-30/38/39/41/46
lineage). Status: **proposal, not yet approved.** Nothing here has shipped._

---

## 0. TL;DR

Two questions were on the table: (1) delete or resurrect the ~849 lines of dead
Overview "briefing" code, and (2) how to make the Overview surface *real, actionable
insight* instead of a chart wall. **They are the same question.** The dead code was an
un-honest first attempt at exactly the decision layer the notes describe — it faked the
numbers it couldn't source (MAPE from a model-name's string length; "System status is
nominal" asserted before any data loads; unlabeled demo news rendered as live). So the
recommendation is one coherent move, not two:

> **Delete the dead implementation. Keep the intent. Rebuild it honestly on Redis.**

The sharp, non-obvious finding a Head of Product would flag immediately: **the four
most decision-shaped lines in the notes are exactly the four with no honest source
today** — "within normal bounds," "elevated heat-demand risk," local-time risk windows,
and the AI "DECISION" line. That is not a coincidence. The most valuable claims are the
hardest to source. A weaker product fabricates them (the dead code did). GridPulse's
entire credibility rests on *refusing to* — so the play is to **ship the honest 80%
now, gate the seductive 20% behind three small, real sourcing steps, and make the
refusal-to-fabricate a visible, paid-tier trust feature** rather than a hidden limitation.

Phasing at a glance:

| Phase | What | New data needed |
|---|---|---|
| **0** | Record the GP-P1-04 delete decision; relocate 6 cross-tab helpers out of the dead module | none (pure refactor) |
| **1 — MVP** | Honest 3-block briefing (DEMAND / MODEL / RISK) + 1 so-what per persona, all `real_now` | none — reads existing `gridpulse:*` keys |
| **2** | Unlock reserve margin, "within normal," "elevated risk," local time | 3 tiny sources (below) |
| **3** | Generative briefing + labeled news (separate bets) | job-side writes |
| **cut** | Dead builders, request-path AI briefing, inline news, demo "Heat Advisory 87°F" string, temp/demand-ramp "signal" | — |

---

## 1. The dead-surface decision (GP-P1-04): **delete impl / keep intent / rebuild on Redis**

### 1a. What is actually dead
The live Overview (`components/tab_overview.py:25-62`) is a clean 5-slot stack —
`overview-title`, `-metrics-bar`, `-spotlight-chart`, `-model-card`, `-insight-card` —
filled by one honest callback (`update_overview_tab`, `_callbacks_overview.py:2489`)
that reads `gridpulse:forecast/actuals` and resolves MAPE from Redis. Behind it sits
~849 lines of **builders that no layout slot targets** and that survive only because
`callbacks.py:74-91` re-exports them and unit tests import them directly.

### 1b. Why "just un-hide it" is unsafe — the hazard table
Every non-trivial dead builder either **invents signal** or **reads process-local
caches that are always empty in the stateless web tier** or **makes a blocking network
call in the request path**. None reads `gridpulse:*` the way the web-tier I/O guardrail
requires.

| Dead builder | Defect | Class |
|---|---|---|
| `_spotlight_model_accuracy` (`:2038`) | MAPE fallback = `4.5 + len(model_name)*0.3` → Prophet 6.6%, Arima 6.0% from **string length**; and the real-cache check `"mape" in result_dict` never hits (writer nests under `["metrics"]`) | **fabrication** (#131 class) |
| `_build_overview_digest` (`:2092`) | scans `_PREDICTION_CACHE`/`_BACKTEST_CACHE` — empty on the web tier; tab2 branch uses a 2-tuple key no writer produces | dead / guardrail violation |
| `_briefing_grid_ops` (`ai_briefing.py:190`) | `"System status … is nominal."` asserted **before any data check**; reserve margin uses wrong denominator | **fabrication** |
| `_build_overview_briefing` → `ai_briefing.py:117` | model id `claude-haiku-4-20250414` **does not exist** → 404s every call after a blocking 10 s sync HTTP call **in the Dash request path** | latency + dead-path |
| `_build_overview_news` (`:2205`) | on any failure serves hardcoded `_get_demo_news()` through the same renderer with **no demo label** | **fabrication** |
| `_build_persona_kpis` (`:2223`) | Wind CF divides by `WIND_CUTOUT_SPEED_MPH=56` (the shut-down speed, not rated ~26-30 mph); nameplate-denominator reserve; reads empty cache before Redis | needs rework |
| `tab3-insight-card` (`tab_models.py:187`) | live layout slot **no callback ever fills**; docstring claims it is populated | dead slot |

Verdict: **delete the implementation** (the builders + their re-exports + `ai_briefing.py`
+ `generate_tab1_insights` + the dead `tab3-insight-card` slot + the direct-import tests
that pin them). The intent survives as the new honest surface (§3).

### 1c. Phase 0 precondition — relocate, don't delete, the 6 live helpers
Six helpers physically living in `_callbacks_overview.py` are **not dead** — live tabs
call them. They must be **moved** before the dead-code deletion, or live tabs break:

- `_build_models_leaderboard` → `_callbacks_models`
- `_build_drivers_panel`, `_build_generation_panel`, `_build_scenarios_panel` → `_callbacks_forecast`
- `_build_risk_insight`, `_build_weather_context` → `_callbacks_alerts`

(`test_layout_no_legacy_ids` already guards that the old slots are gone, so it stays green.)

### 1d. Record the decision first
`CRITICAL_REVIEW_2026-07.md:506` explicitly defers this to a "GP-P1-04 resurrection vs
deletion" call **that does not exist yet**. No dead-Overview code should move until that
call is recorded in STATUS.md. This proposal *is* the recommendation for GP-P1-04.

---

## 2. Product thesis: the honesty gradient is the moat

The investigation classified every insight in the notes as `real_now` / `needs_sourcing`
/ `fabrication_risk`. The pattern is the whole story:

- **`real_now` (ship today, backed by a live `gridpulse:*` key):** forecast peak/min/avg/
  range + timestamps, the empirical confidence band, MAPE/R²/grade + drift, live NOAA alert
  count/categories/top-3/expiry, current temperature, renewable share.
- **`needs_sourcing` (valuable, no honest basis *yet*):** reserve margin (convention +
  estimated-capacity footnote), "within normal bounds" (no seasonal baseline exists),
  "elevated heat-demand risk" (no persisted threshold), local-time windows (no per-BA
  timezone anywhere).
- **`fabrication_risk` (cut):** the demo "Heat Advisory 87°F … until 8 pm local" string
  (lifted verbatim from `demo_data.py:195`), news-as-risk (inline, unlabeled demo
  fallback), temp/demand-ramp "signal" (train-only, never persisted).

**The four crispest "so-what" lines the notes reach for are all in the middle bucket.**
That is the product in one sentence: a competitor ships those four by guessing; GridPulse
ships them only once each has a real source, and *shows its work* in the meantime. The
codebase already encodes this as a hard rule — `insights.py:8` constrains the insight
engine to **"OBSERVE and EXPLAIN, never RECOMMEND ACTIONS,"** and PRD §9 deliberately
descopes autonomous action on liability grounds. **That boundary — inform the decision,
never make it — is not a gap. It is the reason a grid operator or trader would trust the
number, and therefore the thing a paid tier sells.** (See §6.)

---

## 3. The decision briefing — design

### 3a. Where it renders
No new layout slots for the MVP. Enrich the existing, already-honest `overview-insight-card`
into the DEMAND + RISK + DECISION briefing; reuse the existing `overview-model-card` as the
MODEL block. Minimal, in-place, safe.

### 3b. Block structure — **implication first**
The notes proposed *Status → numbers → Implication → Watch*. A decision surface should
**invert that**: lead each block with the deterministic so-what, numbers as the evidence
beneath, watch-item last. A reader scanning for a decision shouldn't parse four figures to
reach the sentence that tells them what it means.

```
DEMAND   ▸ Peak sits inside the forecast confidence band — no tail-risk signal.
           Peak 68,400 MW · Mon 20:00 UTC · 80% within ±1,900 MW (empirical)
           Min 41,100 MW (Tue 05:00) · Avg 54,600 MW · Range 27,300 MW  [next 24h]
           Watch: re-check after the evening ramp.                     ⟢ forecast · scored 12:00 UTC

MODEL    ▸ High forecast confidence — model stable, no drift.
           Ensemble MAPE 2.2% (live 7d) · R² 0.877 · grade: excellent
           Plain English: on a typical hour the forecast is ~2% off actual demand.
           Watch: XGBoost drift nominal.                               ⟢ live 7d drift

RISK     ▸ 2 active NWS alerts; none demand-critical. Current temp 91°F.
           Most critical: Heat Advisory (warning) · until 21:00 UTC
           Renewable share 34% of current generation.
           Watch: reserve-margin trigger — Phase 2.                    ⟢ NWS live · 6 min ago

DECISION ▸ Demand within forecast confidence, model stable, no critical grid alert —
           no action indicated. (Deterministic composition of the above; not an LLM.)
```

_Values illustrative. The MVP renders only `real_now` fields; `needs_sourcing` fields
(reserve margin, "within normal," "elevated," local time) render as a designed
"Phase 2" placeholder, never a guessed number._

### 3c. One so-what per persona — the thing that makes it a *briefing*, not a metrics wall
`personas/config.py` already gives each persona a named **Primary Decision**. Route the
**same Redis keys** to **four different top-lines**:

| Persona | Primary Decision | Headline so-what (same keys, different framing) |
|---|---|---|
| **Grid Ops** | Is reserve adequate through peak? | reserve-margin band (P2) + alert tri-state |
| **Trader** | Is peak inside the band, or is there tail risk to price? | peak + empirical **interval width** |
| **Renewables** | How much of peak is renewable-covered? | `renewable_pct` vs forecast demand |
| **Data Scientist** | Is the model trustworthy right now? | MAPE grade + per-model drift |

### 3d. Trust affordances are first-class UI, not prose
- **Provenance + freshness chip per block** ("NWS live · 6 min ago", "scored 12:00 UTC",
  "capacity estimated · EIA-860M"). The machinery already exists (`error_handling.py`
  confidence badges + `data-freshness-store`, the A4+E3 work). This is the mechanism that
  lets the surface show honest numbers **and prove they're honest** — the paid-tier
  differentiator.
- **Four states per block** (fresh / stale / warming / unavailable), specified before
  building. A decision surface is defined by its worst state, not its happy path.
- **Independent degradation:** a NWS outage degrades only RISK; DEMAND and MODEL stay
  live. Never an all-or-nothing surface. The alert tri-state must render as three visually
  distinct states — collapsing "unavailable" into a green "0 alerts" is the exact
  fabrication the last two months eliminated.
- **Default window = 24h, default region = persona/bookmark** (`dcc.Location` already
  supports `region/persona/tab`), so first paint is already the user's decision context.

### 3e. Slide-in spec — build from the existing `.gp-*` components (dark theme)

The briefing must look native to the current **dark** shell (`--bg-base #0a0a0b`, cards
`--bg-raised #111113`, Inter/Sora), not like a foreign card. It slides in with **three
thin CSS additions and zero new colors** if built this way:

- **Container** `.gp-briefing` = a byte-for-byte clone of `.gp-chart-card` (`--bg-raised`
  on `--bg-base`, `1px solid --border-subtle`, `--radius-lg`, `--card-padding`,
  `--edge-highlight`) so it matches the hero chart directly above it. **Do not** reuse the
  legacy `.briefing-card` (custom.css:1962) — it carries drop-shadows from the dead
  AI-briefing surface we're deleting.
- **Blocks** stack flat, separated by a `border-top: 1px solid --border-subtle` hairline
  (the `.gp-model-card` idiom). No cards-within-cards.
- **Reuse, don't invent:** DEMAND/RISK/DECISION prose = the existing `.gp-insight-card`
  machinery (eyebrow + body + `__delta`/`__strong`); **MODEL block = the existing
  `overview-model-card` relocated** (never render two model cards); provenance chip = the
  existing `.gp-as-of-chip` (neutral mono); freshness state = the existing
  `error_handling.freshness_badge()`; watch line = `.gp-metric-sub`. The **only** new
  style is `.gp-briefing__implication` — a 14px (`--text-md`), weight-500, `--text-primary`
  lead line so the so-what out-ranks its numbers.
- **Color ruling (the crux — and the correction to the first mockup).** Per-block
  green/amber/blue status dots are **out** — they violate CLAUDE.md's "product identity
  must not rely on alert colors," and a green "0 alerts" dot collapses the NWS tri-state
  into a false all-clear (§3d). Correct treatment:
  - All three blocks use **identical neutral chrome** (`--bg-raised` / `--border-subtle`).
  - `--forecast` orange (`#f97316`) is the **sole signature accent**, used only on the
    forecast/DEMAND figures (it's the design system's "forecast signal" color).
  - Alert colors (`--danger`/`--warning`) are **earned only by a genuine RISK severity**
    (a real active NWS warning), via the existing `.alert-card` severity classes.
  - `--success` green appears **only** on a real freshness dot or a genuine positive delta
    — never as a block-level "all good" status.
  - The DECISION line gets a calm `border-left: 3px solid --accent-base` (the `.welcome-card`
    idiom), never an alert color.
- **Relationship to the current 5 slots:** the briefing **replaces** the single-paragraph
  `overview-insight-card` and **folds in** `overview-model-card` (one home). It does **not**
  subsume `overview-metrics-bar` — that 5-up strip stays as the scannable numeric anchor;
  the briefing is the interpretation layer above it (so window the DEMAND numbers to 24h so
  they don't just echo the bar). Fix the orphaned `.gp-metric-subtext` selector in passing
  (emitted by `cards.py:357`, no CSS rule exists).

---

## 4. Feasibility map — every insight in the notes, classified

Backing is the real `gridpulse:*` key or `file:line`. "Phase" is when it can ship *honestly*.

| # | Insight (from your notes) | Honesty | Backing | Phase |
|---|---|---|---|---|
| a | Forecast **peak** MW + datetime | `real_now` | `gridpulse:forecast:{r}:1h` series; `phases.py:975` | **1** |
| a | **Min** projected + time | `real_now` | same series | **1** |
| a | **Avg** projected | `real_now` | same series (window it — see note) | **1** |
| a | **Range** (peak−min) | `real_now` | derived | **1** |
| b | **Confidence band** on peak | `real_now` | `gridpulse:backtest/holdout` (#212); `_callbacks_forecast.py:205` | **1** |
| b | **MAPE / R² / RMSE / MAE + grade** | `real_now` | `model_metrics` on forecast payload (`phases.py:1005`); `mape_grade` | **1** |
| b | Per-model **drift** status | `real_now` | `gridpulse:drift:{r}.models.*`; use 1h-ahead trend, **not** the P2-25 cross-horizon ratio | **1** |
| c | **N active alerts** + categories + top-3 + expiry | `real_now` | `gridpulse:alerts:{r}` (#204/#205); tri-state | **1** |
| c | **Current temperature** | `real_now` | `gridpulse:weather:{r}.temperature_2m` nearest-now | **1** |
| c | **Renewable share** / supply context | `real_now` | `gridpulse:generation:{r}.renewable_pct` | **1** |
| a | **Reserve margin %** + "demand nears X MW" trigger | `needs_sourcing` | capacity is **real & sourced** (`config.REGION_CAPACITY_MW`, EIA-860M) but **convention is wrong** + estimated-capacity BAs need the footnote | **2** |
| d | **"Within normal bounds"** | `needs_sourcing` | **no seasonal baseline exists**; needs new `gridpulse:demand_normal:{r}` percentile write | **2** |
| c | **"Elevated heat-demand risk"** | `needs_sourcing` | CDD/HDD are **train-only, unpersisted**; needs a stated threshold (percentile or reserve band) | **2** |
| d | **Local-time** peak / risk window | `needs_sourcing` | **zero per-BA timezone anywhere**; needs a `REGION_TIMEZONE` map | **2** |
| — | Rapid **temp/demand ramp** "signal" | `fabrication_risk` | `ramp_rate` is a train-only MW diff, never in Redis | **cut** |
| — | **News**-driven risk | `fabrication_risk` | inline in request path, unlabeled demo fallback | **cut** |
| — | Demo **"Heat Advisory 87°F … 8 pm local"** line | `fabrication_risk` | verbatim from `demo_data.py:195` | **cut** |

**Two honest-precision notes** that the notes underweight and a reviewer would catch:
1. **Window the demand aggregates (default 24h).** The forecast is 720 h, but only the
   first ~384 h use real weather; beyond that it degrades to `(hour, dow)` climatology
   (ADR-008). A single "30-day peak" is a *technically honest max* that is **misleading by
   omission** because it silently mixes two regimes. Enforce the window at the data layer.
2. **Guard the capacity default.** `.get(region, 50000)` fabricates a 50 GW capacity for
   any unknown BA. Any capacity-derived figure must suppress/hard-fail for unknown BAs,
   never fall back to the default.

---

## 5. The three Phase-2 prerequisites (each small, each unlocks multiple claims)

The `needs_sourcing` items collapse onto **three** well-scoped sources. Do these three and
the entire "seductive 20%" becomes honest:

1. **`REGION_TIMEZONE` IANA map in `config.py`** (~51 hand-verified entries; deterministic
   from lat/lon; DST handled by the zone). **Pure conversion, no Redis write.** Unlocks
   local peak-time, local alert-expiry, and the local risk-window **all at once**. Care for
   multi-tz BAs (MISO/SPP span Central + a sliver) — hand-verify, don't guess from centroid.
2. **Fix + label the reserve-margin convention** (one code change in `models/pricing.py`).
   Today `pricing.py:104` and `ai_briefing.py:197` compute `(capacity − demand)/capacity`
   — a **utilization complement**, *not* the NERC planning reserve margin
   `(capacity − peak)/peak` that the word "reserve margin" means to a grid operator. They
   diverge materially, and **near the 15% NERC Reference Margin Level the divergence flips
   the adequacy verdict**: cap 115 GW / peak 100 GW → utilization-complement `(cap−peak)/cap`
   = **13.0%** (reads *below* the 15% bar) vs NERC `(cap−peak)/peak` = **15.0%** (reads *at*
   the bar). Same region, opposite call. **Recommendation: adopt NERC** `(cap−peak)/peak` because the
   surface literally shows the forecast *peak*, so pairing it with peak is the natural,
   correct definition — and carry the "capacity estimated (EIA-860M)" footnote /
   suppression for `IS_IMPORT_DOMINATED` and `peak×1.15` BAs. Unlocks reserve margin, the
   "demand nears X MW" trigger, **and** one honest path to "elevated risk."
3. **New scoring/training write `gridpulse:demand_normal:{region}`** = per-`(month, hour)`
   (or trailing-N-weeks same-hour) demand **percentile band** from `gridpulse:actuals`
   history. The raw material and the `group-by-(hour, dow)` machinery already exist
   (`_build_future_feature_frame:654`). Unlocks **both** "within normal bounds" (peak
   inside p10–p90) **and** the second path to "elevated risk," in one write.

Each is independently valuable and removes exactly one fabrication vector.

---

## 6. Monetization / positioning (Head of Monetization lens)

This is a portfolio/interview piece, so "monetization" = a credible narrative an
interviewer respects, **not** a claim of traction. The architecture backs exactly one
angle end-to-end, and the strongest move is to anchor on the trust boundary as the moat.

- **Lead: tiered access — free grid visibility vs. paid confidence + decision-support +
  alerting.** `natural`. The free/paid seam maps onto the product's own architecture: the
  read-only visibility tabs (Historical, Generation, Weather) are the free layer; the paid
  layer is the *confidence* surface (calibrated intervals #196/#212, drift/model-risk
  monitoring, persona briefings, live NOAA alerts #204/#205). Interview line: *"The
  dashboard is table stakes. You charge for the confidence, and for the alert that reaches
  you when you're not looking at the screen."* Every ingredient is a real wired capability.
- **Anchor the whole story on the boundary: "inform the decision, never make it."** PRD §9
  + `insights.py:8` deliberately stop at OBSERVE/EXPLAIN. Sell that as **judgment, not a
  gap**: refusing autonomous action (and refusing to fabricate a number) is *why* an energy
  buyer trusts the output. Trust is the moat; the honesty gradient (§2) is the product.
- **Roadmap hooks, labeled precisely (don't overclaim):**
  - **Alerting SLA / push subscriptions** — `plausible`. Detection exists (severities
    computed hourly); **delivery does not** (no email/Slack/PagerDuty, no subscription
    store; `NEXT_UP.md:418` scopes it ~2-3 days). Say "the signal is in Redis; the
    transport is the next build."
  - **API / data-feed** over `gridpulse:*` — `plausible`. The jobs already produce a
    complete, versioned, per-region forecast+confidence dataset; it's feed-shaped. But **no
    authenticated API exists** — the web tier only renders it. Say "the data asset is
    productizable as a feed," not "we have an API product."
  - **Per-seat by persona** — `plausible` but personas are **personalization, not
    entitlement** (PRD §9 descopes RBAC + multi-tenant; no accounts/auth). Frame as "the
    role model maps cleanly to seats; today it's personalization."
  - **White-label per-utility** — `stretch` / present as a **deliberate non-goal** that
    shows judgment (multi-tenant descoped on purpose), not a revenue line.
- **Do not conflate `models/pricing.py` (electricity $/MWh merit-order) with product
  pricing.** Calling that distinction out explicitly signals you read the code, not the
  tagline.

---

## 7. Phased plan + issues to file

**Phase 0 — decide & de-risk (no UI).**
- Record GP-P1-04 (delete-impl / keep-intent / rebuild-on-Redis) in STATUS.md + review doc.
- Relocate the 6 cross-tab helpers (§1c). Delete dead builders + re-exports + `ai_briefing.py`
  + `generate_tab1_insights` + dead `tab3-insight-card` + their direct-import tests.
- _Independently valuable: removes ~849 lines of dead code + a #131-class fabrication vector._

**Phase 1 — MVP decision briefing (ship_now only, zero new writes).**
- DEMAND block: windowed peak/min/avg/range (24h default) + empirical/indicative band.
- MODEL block: reuse the live MAPE/grade/R²/drift path verbatim.
- RISK block: alert count + categories + top-3 + expiry (tri-state, 3 distinct states);
  current temp; renewable share.
- One deterministic so-what per persona; DECISION line composed from real numbers (no LLM).
- Provenance/freshness chip per block; all four block states; UTC timestamps.

**Phase 2 — the three prerequisites (§5).** Timezone map → local time. Reserve-margin
convention fix + footnote → reserve margin + trigger + one "elevated" path.
`gridpulse:demand_normal` write → "within normal" + second "elevated" path. Rework
`_build_persona_kpis` arithmetic (rated-wind-speed constant, fixed reserve, drop cache read)
here if the persona-KPI layer is wanted.

**Phase 3 — separate bets (only if Phase-1 deterministic so-what proves insufficient).**
Generative `gridpulse:briefing:{region}` written **job-side** with a valid model id
(`claude-haiku-4-5`), read-only in the web tier. Labeled news via a job-side write.

**Cut permanently.** Request-path AI briefing, inline news, temp/demand-ramp "signal", any
resurrection of the dead builders' logic, and the demo "Heat Advisory 87°F" string.

---

## 8. Decisions this proposal needs from you

1. **GP-P1-04:** approve delete-impl / keep-intent / rebuild-on-Redis? (recommended: yes)
2. **Reserve-margin convention:** NERC `(cap−peak)/peak` (recommended) vs. keep the current
   utilization-complement — a labeling choice with a real numeric consequence.
3. **Scope:** ship **Phase 0 + 1** now (all honest, no new data), and treat Phase 2's three
   sources as a fast-follow? Or bundle Phase 2 in?
4. **Deploy interaction:** Phase 0/1 are deploy-independent and can land before the keystone
   re-measurement; Phase 2's `demand_normal` write is a scoring-job change that pairs
   naturally with that deploy. Sequence accordingly?
5. **Shell-coherence PR first?** Land the header positioning/tagline + selector labels +
   CLAUDE.md/`.gp-metric-subtext` hygiene (§10) as a small behavior-preserving PR **before**
   the briefing PR — it establishes the "platform" promise the briefing then pays off.
   (Recommend yes.)

---

## 9. Industry-standard anchors (what we can credibly cite)

Grounding claims in recognized standards raises credibility — but **citing a standard wrong
is its own fabrication.** Two rules: (a) an anchor earns **UI space** only if it has a live
`gridpulse:*` source *and* makes a number more trustworthy at a glance; otherwise it's
**interview narrative**; (b) always separate "cite the standard to *frame* our number" from
"we *meet* the standard" (usually the former).

| Standard | What it is | Honest GridPulse framing | Where | Verified |
|---|---|---|---|---|
| **NERC Reference Margin Level** — 15% (thermal) / 10% (hydro) | The planning-reserve threshold that flags a forecast capacity shortfall in NERC's Long-Term Reliability Assessment | Show the Phase-2 reserve-margin figure **against the 15% RML** as a reference line; note region-specific targets (**ERCOT 13.75%**, 1-in-10 LOLE) rather than one universal bar | UI (Phase-2 reserve figure) + narrative | ERCOT NERC-RML primer; EIA |
| **Planning reserve margin = (capacity − peak)/peak** | The standard definition (EIA states it as (capacity − demand)/demand with demand = peak) | Confirms adopting NERC over the code's utilization-complement; label the convention explicitly | UI label + narrative | EIA |
| **Day-ahead STLF MAPE ≈ 1–3%** (competitive ISO < 2.5%, < 5% common benchmark) | Typical utility/ISO short-term load-forecast accuracy | Our real holdout — best-base median **2.28%**, ERCOT **0.79%** — sits **within the typical professional range**. Claim "in line with / at the better end of typical day-ahead STLF," **not** "beats the industry" | Narrative (the MAPE grade is already in UI) | Industry summaries — cite as "commonly reported," not a single authority |
| **CDD/HDD base 65°F** | The NOAA/EIA standard degree-day base | We already compute CDD/HDD at 65°F (`feature_engineering`) — cite the standard directly | UI (Weather Correlation) + narrative | EIA degree-days; NOAA/NWS |
| **Net load / "duck curve"** | CAISO's term for load net of renewable output | Frames the Generation & Net Load tab and the Renewables persona's story | Narrative (+ existing net-load chart) | CAISO |
| **Horizon-aware MAPE grades** (excellent/target/acceptable/rollback) | Our own H2 governance banding — an *internal* honest standard, not external | Keep as the credibility grade on the MODEL block | UI | `config.mape_grade` |

**Do NOT claim** (would overstate): **operating / contingency reserves** (NERC BAL-002,
N-1 "largest single contingency") — we ingest no real-time reserve telemetry, so frame ours
strictly as a **planning** reserve margin, never operating reserve; **NERC Energy Emergency
Alert (EEA) levels 1/2/3** — our `stress_score` is self-declared uncalibrated, so keep a
coarse Normal/Elevated/Critical label explicitly marked "not an EEA level"; **ELCC**
(renewable effective capacity value) — we don't compute it, so it's a future frame only.

---

## 10. Current-shell polish (independent of the briefing)

**Reframe first:** the live shell is **already** a clean 5-tab decision IA —
Overview / US Grid / Forecast / Risk / Models — *not* the 9-tab feature sprawl CLAUDE.md
still lists (Extreme Events already folded into Risk, Model Diagnostics into Models, etc.).
So the "consolidate the tabs" question is largely already solved. The real gaps:

- **P0 — positioning is invisible in the running product.** "Energy Intelligence Platform"
  and the tagline "See demand sooner. Decide with confidence." live only in README/PRD/a CSS
  comment; the live header (`layout.py:82-132`) is monogram + wordmark + unlabeled controls,
  so the shell reads as a generic admin theme. Surface the descriptor/tagline in the header
  lockup in calm `--text-tertiary`/`--text-xs` (product identity, not accent). This is the
  highest-leverage unmet EXECUTION_BRIEF P0 — and it seeds the narrative the briefing pays off.
- **P1 — unlabeled selectors.** The region and persona `dbc.Select`s (`layout.py:98-109`)
  have no visible label and no `aria-label` — a bare "FPL" gives a newcomer no anchor. Add
  "Region" / "View" labels and action the "Persona → View / Role View" rename (preserve the
  `region-selector`/`persona-selector` IDs so callbacks are untouched).
- **P1 — model-card provenance.** The Overview model card states MAPE/R² with only a
  "trained" badge and no window, so it reads as *live* forecast accuracy when it's last
  night's **168h training holdout**. Label it "holdout · 168h / updated nightly" — this lands
  inside the briefing's MODEL block. Last unlabeled-precision surface after the no-fabrication work.
- **P1 — doc hygiene.** Patch CLAUDE.md's stale 9-tab list + Module Map to the real 5 tabs
  (README/PRD already match). Fix the orphaned `.gp-metric-subtext` selector.
- **P2 — typography resilience.** The external Google Fonts `@import` (`custom.css:10`) is a
  demo-time single point of failure for the Inter/Sora premium look; self-host the woff2 or
  specify a graceful fallback stack.

**Sequencing:** ship these as a small "shell coherence" PR **first**, then the briefing PR.

---

## 11. Appendix — per-BA reserve-margin reference thresholds (Phase 2 sourcing)

Reserve-margin *targets* are not set per EIA-930 balancing authority — they're published at
the **ISO/RTO or NERC assessment-area** level (and per-utility IRP for vertically-integrated
BAs). So the honest structure is a **tiered reference**, not 51 invented numbers:

- **Default (all 51):** the **NERC Reference Margin Level** — **15%** thermal-dominated /
  **10%** hydro-dominated — assigned by each BA's resource mix. NERC itself uses this as the
  default RML when an area doesn't publish one. Source: NERC LTRA; ERCOT NERC-RML primer.
- **Region-specific overrides** (only where a target is documented *and* on a basis roughly
  comparable to our nameplate margin):

| BAs | Reference | Basis / source | Notes |
|---|---|---|---|
| ERCOT | **13.75%** | ERCOT Board minimum target (1-in-10 LOLE) | accredited-capacity basis |
| PJM | **17.7%** (25/26); 19.1% (26/27) | PJM IRM — **ICAP**, comparable to nameplate | rises yearly |
| NYISO | **24.4%** (25/26) | NYSRC IRM — **ICAP**, comparable | very high; rising |
| CAISO | **18%** (26/27); 15% baseline | CPUC PRM on **NQC** (accredited) | not raw-nameplate basis |
| SPP | **16%** summer 2026 (was 15%) | SPP PRM on accredited cap | winter target **36%** — seasonal |
| MISO | use **15% RML** for display | MISO PRM is **7.9% UCAP** — *not comparable* to nameplate | flag; do not show 7.9% against a nameplate margin |
| ISONE | **~15%** | NERC reference; ISO-NE plans to an ICR, not a fixed margin | representative net reserve ~12–16% |
| FPL, FPC, TEC | **20%** | FPSC voluntary Total Reserve Margin criterion (investor-owned FL) | other FL BAs use 15% |
| DUK, CPLE, CPLW | **17%** (→ 22% by 2031) | Duke Carolinas IRP / NCUC-approved | rising |

- **Hydro-dominated → 10% (NERC RML hydro default):** `BPAT`, `SCL`, `TPWR`, `GCPD`,
  `CHPD`, `DOPD`, `WALC`, `SPA`.
- **All other BAs → 15% (NERC RML thermal default):** `SOCO`, `TVA`, `AZPS`, `NEVP`, `PSCO`,
  `FMPP`, `JEA`, `TAL`, `GVL`, `SEC`, `SC`, `SCEG`, `LGEE`, `AECI`, `EPE`, `PACE`, `PACW`,
  `PGE`, `PSEI`, `AVA`, `IPCO`, `NWMT`, `BANC`, `LDWP`, `IID`, `TIDC`, `SRP`, `TEPC`, `PNM`.
- **Import-dominated → indicative only:** `SPA`, `HST`, `CPLW`, `WALC` — their `REGION_CAPACITY_MW`
  is already a peak×1.15 estimate (`IS_IMPORT_DOMINATED`), so both the margin *and* its
  reference are indicative; suppress or label, mirroring the stress-KPI suppression.

**Three honesty caveats to carry in the UI/config:**
1. **Basis mismatch.** Our `(nameplate − peak)/peak` is a raw-nameplate margin; ISO IRM/PRM/RA
   targets are on accredited/derated capacity (UCAP/NQC), which is *lower* — so a raw-nameplate
   margin reads *higher* than the accredited target it's compared against. Prefer showing the
   NERC RML band (15%/10%) as the consistent reference and treat ISO numbers as context; never
   compare a nameplate margin directly to a UCAP figure (the MISO trap).
2. **Targets drift upward** (SPP 15→16, CAISO 15→18, Duke 17→22, PJM 17.7→19.1). Store with an
   `as_of` date; don't hardcode-and-forget — same failure mode as the static `REGION_CAPACITY_MW`.
3. **Seasonality.** Some targets are seasonal (SPP summer 16% / winter 36%); a demand-now surface
   should use the season-appropriate (or annual-peak) figure and say which.

**Recommended config shape** (mirrors `REGION_CAPACITY_MW` provenance style):
```python
# Planning reserve-margin *reference* per BA. Default = NERC Reference Margin Level
# (15% thermal / 10% hydro-dominated); overrides carry ISO/utility source + as_of.
REGION_RESERVE_MARGIN_PCT: dict[str, float] = { "ERCOT": 13.75, "PJM": 17.7, ... }
RESERVE_MARGIN_BASIS: dict[str, str] = { "ERCOT": "ERCOT target 2024", "PJM": "PJM IRM 25/26 (ICAP)", ... }
# helper: reserve_margin_reference(region) -> (pct, basis, is_indicative)
```
This lands in **Phase 2** alongside the reserve-margin convention fix (§5.2) — the reference is
useless without the corrected `(cap−peak)/peak` numerator and the estimated-capacity suppression.

### Sources (industry standards §9 + reserve-margin references §11, verified 2026-07)

_Note: ISO/utility reserve-margin targets change annually — re-verify before shipping. The
per-utility 15%/10% defaults are the NERC RML default by resource mix, not each utility's
published IRP target._

**NERC / reserve-margin definition + default RML (15% thermal / 10% hydro):**
- NERC Reference Margin Level primer (ERCOT): https://www.ercot.com/files/docs/2017/05/16/ERCOT_Primer_on_the_NERC_ReferenceMarginLevel_5-15-2017.pdf
- NERC Long-Term Reliability Assessment (LTRA): https://www.nerc.com/globalassets/our-work/assessments/nerc_ltra_2025.pdf
- NERC Planning Reserve Margin (M-1): https://www.nerc.com/pa/RAPA/ri/Pages/PlanningReserveMargin.aspx
- EIA — reserve capacity: https://www.eia.gov/todayinenergy/detail.php?id=6510

**Per-region overrides:**
- ERCOT 13.75%: https://www.ercot.com/news/release/02132025-ercot-releases-capacity · https://www.ercot.com/gridinfo/resource
- PJM IRM 17.7% (25/26) → 19.1% (26/27): https://www.pjm.com/-/media/DotCom/committees-groups/committees/mrc/2025/20250319/20250319-item-04---irm-fpr-and-elcc-for-26-27-bra---presentation.pdf · https://www.pcienergysolutions.com/2025/02/06/how-the-installed-reserve-margin-irm-ensures-reliability-in-pjm-markets/
- NYISO IRM 24.4% (25/26): https://www.nysrc.org/wp-content/uploads/2024/12/2025-IRM-Study-Technical-Report_Final_12062024_clean.pdf · https://www.nyiso.com/-/how-the-installed-reserve-margin-supports-reliability-in-new-york
- MISO PRM 7.9% UCAP (not comparable to nameplate): https://cdn.misoenergy.org/PY%202025-2026%20LOLE%20Study%20Report685316.pdf
- ISO-NE Installed Capacity Requirement: https://www.iso-ne.com/system-planning/system-plans-studies/installed-capacity-requirement
- SPP 16% summer / 36% winter (was 15%): https://www.spp.org/news-list/spp-board-approves-new-planning-reserve-margins-to-protect-against-high-winter-summer-use/
- CAISO/CPUC PRM 15% → 18% (26/27): https://www.cpuc.ca.gov/industries-and-topics/electrical-energy/electric-power-procurement/resource-adequacy-homepage · https://www.caiso.com/Documents/Resource-Adequacy-Fact-Sheet.pdf
- Florida 20% (FPL/DEF/TEC, FPSC criterion): https://www.utilitydive.com/news/frcc-florida-reserve-margin-to-remain-over-20-as-state-leans-on-gas-gener/426398/ · https://www.floridapsc.com/pscfiles/website-files/PDF/Utilities/Electricgas/TenYearSitePlans//2025/FRCC_RLRP.pdf
- Duke Carolinas 17% → 22% by 2031: https://www.duke-energy.com/our-company/about-us/irp-carolinas · https://news.duke-energy.com/releases/duke-energy-responds-to-constructive-carolinas-resource-plan-decision-by-north-carolina-utilities-commission

**Other industry-standard anchors (§9):**
- Degree-days base 65°F (CDD/HDD): https://www.eia.gov/energyexplained/units-and-calculators/degree-days.php · https://www.weather.gov/key/climate_heat_cool
- Day-ahead load-forecast MAPE (commonly 1–3%; cite as "commonly reported," not a single authority): https://www.amperon.co/blog/the-different-kinds-of-forecasting-metrics
- Net load / "duck curve": CAISO (term of art; no single canonical URL — attribute to CAISO).

---

## 12. Live production review — gridpulse.kristenmartino.ai (2026-07-02)

Rendered the live site (ERCOT) headless and reviewed Overview / Risk / Models. The plan folds
in cleanly, and the live product both **confirms** the recommendations and surfaces **four fresh,
visible issues** worth acting on.

### Confirmed against the live product
- **Positioning invisible (P0 holds).** Header on every tab is the "GridPulse" wordmark +
  region/View selectors + Briefing Mode / Save View — no category or tagline. Each tab does have
  a decent page subtitle ("Demand forecast and grid intelligence"; "Active alerts, demand
  anomalies, and grid stress"; "Forecast accuracy, residuals, and feature importance") — a
  foothold, but not product positioning.
- **Dark theme + blue/orange** — the corrected §3e mockup matches the real chrome.
- **Overview shape** = title+subtitle · metrics bar (NOW / 7D PEAK / 7D LOW / AVERAGE / 24H TREND —
  all 7-day **historical**, with freshness subtext) · hero chart (history + orange forecast + band,
  "forecast scored …") · model card · an "OPERATING SUMMARY" paragraph. The briefing folds in
  exactly as §3e planned: restructure the summary paragraph into implication-first
  DEMAND/MODEL/RISK/DECISION, fold the model card into MODEL (with provenance), keep the metrics
  bar (historical → complementary to the forward DEMAND block) and the hero chart.
- **Model-card provenance gap confirmed and vivid:** the card shows "Ensemble · [trained] · MAPE
  0.9% · R² 0.993" while the summary right below cites "live 7d sMAPE 4.7%" — two model-accuracy
  numbers on one screen, the rosy holdout one unlabeled. Exactly the misread §10 flags.
- **RISK content already exists on the Risk tab:** live NOAA alert (real "Flash Flood Warning …
  by NWS Midland/Odessa TX" with expiry — #204/#205 confirmed live in prod), stress score,
  "Demand sits within normal bounds" (±2σ band), "Heat-driven demand risk is elevated (peak
  108°F)". So the Overview RISK block **summarizes existing verdicts** — it is *not* blocked on
  new sources for the MVP; the seasonal-normal upgrade (§4) is a refinement, not a gate. Label
  honestly ("within recent ±2σ trend"; "heat-driven risk — forecast peak 108°F").

### New live issues (visible now)
1. **`nan°F` current temperature (P1 bug).** Risk tab → "Current Conditions → Temperature: nan°F"
   *with a green "fresh" dot*, while forecast peak temp (108°F) computes fine — a NaN leaking from
   the nearest-now temp lookup. The RISK block's current-temp must guard NaN → "—/unavailable".
2. **Models "LIVE DRIFT" panel cries wolf (P1 credibility — P2-25, live).** Every model is labeled
   **"Degraded"**: XGBoost live-7d **1.53%** (excellent, inside the 1–3% industry band) is flagged
   Degraded because live÷holdout = ×2.04 exceeds threshold — comparing 1h-ahead live drift to the
   168h **teacher-forced** holdout (incommensurable). Headline "One or more models drifting vs
   holdout baseline" is a **false alarm**. Two-part fix: the keystone deploy's recursive holdout
   (#209) raises holdout to a commensurable basis (shrinks the ratio), *and* the status logic must
   stop comparing cross-horizon (P2-25 — compare like-for-like or relabel as an indicative ratio,
   suppress "Degraded" when the terms aren't commensurable). **This is the most damaging live
   credibility issue** — the product's own health panel says everything is broken when XGBoost is
   running at 1.5%. Elevate from "entangled with deploy" to a near-term fix.
3. **4 of 6 Models-tab charts empty (P2 — product call).** Residuals over-time / distribution /
   vs-predicted / error-by-hour all show the honest empty state ("Populates after…") — correct
   no-fabrication behavior, but four blank panels dominate the tab. Either populate (needs the
   diagnostics/backtest write landing in prod — #166 area) or collapse/hide until data exists
   rather than showing four empty boxes. (SHAP importance + the metrics/drift tables render real.)
4. **"10.0% above" in alert-red (P2 color).** The OPERATING SUMMARY renders "Demand is 10.0% above
   the 7-day average" in `--danger` red — an alert color on a routine summer metric, the exact
   semantic-color rule the briefing enforces. The implication-first DEMAND line fixes it by design.

### Adjustments to the plan
- **RISK MVP is less gated than §4 implied** — "within normal" and "elevated heat risk" already
  ship on Risk (±2σ + temp-threshold); the Overview block can surface them now with honest labels.
  Reserve margin + local time remain Phase 2.
- **Elevate the Models "Degraded" false alarm** to a near-term credibility fix (it's the most
  visible trust problem on the live site).
- **Selectors render friendly values** ("Texas (ERCOT)", "Grid Operations Manager") — the "bare
  FPL" concern is milder than the code-only read; field labels remain a minor nicety.
