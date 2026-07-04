# US Grid tab — UX improvement proposal (2026-07-04)

From a live review of the Cards view. Three related problems, all grounded in
the current code (`components/_callbacks_us_grid.py`, `tab_demand_outlook.py`,
`cards.py`).

---

## Problem 1 — the metrics have no visible definitions

**A new reader can't tell what anything means.**

- **Top KPI bar** (`_build_us_grid_metrics_items` → `build_metrics_bar`): *Total
  Demand*, *National Peak (24h)*, *Highest-Stress Region*, *Lowest Reserve* —
  **zero tooltips or definitions.**
- **Per-BA card badges** carry three numbers with only terse **hover-only**
  `title=` tooltips (invisible, mobile-hostile) and no legend:
  - `29%` = utilization (`current_mw ÷ capacity_mw`)
  - `−0.2 GW` = net inter-BA interchange (`imports` chip for import-dominated BAs)
  - `−10.0%` = hour-over-hour demand change (`(current − prev) / prev`)
- **Specific confusions the labels create:**
  - `National Peak (24h) 711.5 GW` **>** `Total Demand 460.7 GW` reads as a
    contradiction. It's actually correct — the honest *simultaneous* cross-grid
    24h peak (`_simultaneous_national_peak_mw`, the #203 fix) — but nothing
    tells the reader that, so "peak > demand" just looks broken.
  - "Stress", "Reserve", "utilization" are undefined domain terms.

## Problem 2 — the summary KPIs aren't actionable

- `Highest-Stress Region · PACW · 69%` and `Lowest Reserve · 31%` are **static
  text**. You can see the tightest BA but can't jump to it.
- The per-BA **cards are already clickable** (`drilldown_from_us_grid`:
  card → set `region-selector` + open the Forecast tab), so the plumbing for
  "go to a BA" exists — the KPIs just don't use it.
- `Lowest Reserve` doesn't name its BA — and, verified in code
  (`_callbacks_us_grid.py:279`), it is the **same BA as Highest-Stress**:
  `lowest_reserve_pct = (1 − max(reliable_stress.values()))` reuses the identical
  `max` that picks `top_region`. So the two KPIs are complementary numbers
  (util% and 100−util%) for *one* region — a design smell worth a later look
  (§Open question), not just a labeling gap.

## Problem 3 — the US Grid → Forecast drilldown is incoherent

Clicking a BA card jumps to the **Forecast tab** (`tab-outlook`) scoped to that
region. But the two surfaces share no vocabulary:

| US Grid card (what you clicked) | Forecast tab (where you land) |
|---|---|
| current demand (**GW**) | Peak · Average · Min · Range (**MW**) |
| utilization % · reserve · net interchange · Δ%/hr | forward forecast + confidence band + model picker |

Unit mismatch (GW↔MW), metric mismatch, and **no continuity** — nothing on the
Forecast tab acknowledges you drilled in from the grid or carries the numbers
you were just looking at.

## Problem 4 — Overview and Forecast summarize demand over different (unlabeled) windows

Both tabs show "Peak / Low(Min) / Average," but they are **not the same
measurement** and nothing says so:

| | Overview KPI bar | Forecast KPI bar |
|---|---|---|
| **What** | *observed actuals* (EIA-930 demand) | *model forecast* (forward predictions) |
| **Window** | **fixed** last 168h (7 days) | the **user-selected** horizon (default 7d; options 24h / 7d / 30d) |
| **Extras** | Now, 24h Trend | **Range** (peak−min) + **hourly ramp** (insight card) — Forecast-only |

Verified in code: Overview uses `nonzero.tail(168)` on actuals
(`_callbacks_overview.py:273-282`); Forecast uses `predictions[:horizon_hours]`
on the forecast (`_callbacks_forecast.py:772-774, 1283-1293`). So the two "Peak"
numbers line up **only** when the Forecast horizon is left at its 7-day default
*and* you ignore that one is history and the other is a prediction. Flip the
horizon to 24h or 30d and they diverge with no visible reason. (A third window
hides on Overview itself: the insight card's "Last 24h peak" uses `df.tail(24)`.)

## Problem 5 — the "which model" story is inconsistent across tabs

Three surfaces name three different "primary" models for the same region (HST):

| Surface | Model featured | Why |
|---|---|---|
| **Overview** model card | **Ensemble** | `_build_overview_model_card` hardcodes `if "ensemble" in metrics: primary = "ensemble"` (`_callbacks_overview.py:579`) |
| **Forecast** model selector | **XGBoost** | `outlook-model` hardcodes `value="xgboost"` (`tab_demand_outlook.py:79`); no callback overrides it |
| **Backtest** insight | **XGBoost** (best single) | dynamically `min(mape)` per region (`insights.py:279`) |

For HST the holdout numbers are **XGBoost 3.70% < Ensemble 4.06%**
(`BACKTEST_RESULTS.md`), so the Overview card features the *less accurate* number
(Ensemble 4.06%) with no explanation, while the other two tabs feature XGBoost.
That's the "why is Ensemble the default when XGBoost is best?" you saw.

It is **not** a bug in the math: per ADR-004/ADR-005 the ensemble is the
*production* forecast — chosen for error-decorrelation / tail-robustness, not the
lowest headline MAPE (it documentedly *trails* the best base model in aggregate,
~4.82% vs 4.12%). The problem is purely presentational: the UI presents
"Ensemble 4.06%" as if it were the winner, contradicts itself tab-to-tab, and
never tells the reader the ensemble is the shipped blend rather than the most
accurate single model.

---

## Proposed fixes

### Fix 1 — definitions (recommended: info affordances + a badge legend)

- **(a) An `ⓘ` per KPI** with a tap/hover tooltip: one-line definition + formula
  + why it matters. Add an optional `help` field to `build_metrics_bar` items
  (small, clean, reusable across tabs). *Recommended.*
- **(b) A one-line legend** under the KPI bar / above the card grid decoding the
  three card badges and their color meaning. *Recommended, pairs with (a).*
- **(c) Clarify the confusing labels:**
  - `National Peak (24h)` → tooltip "Highest *simultaneous* cross-grid demand in
    the last 24h (BAs peak at different hours, so this exceeds current total
    demand)." Consider renaming to `Grid Peak (24h)`.
  - `Highest-Stress Region` → "Utilization = current demand ÷ estimated capacity;
    the highest across BAs."
  - `Lowest Reserve` → "Reserve = 1 − utilization; the tightest BA right now."
- **(d) Optional:** a collapsible "How to read this" / shared glossary the other
  tabs can reuse. Bigger; defer.

### Fix 2 — make the summary KPIs actionable (recommended: scroll-to-card)

- Make **Highest-Stress Region** (already names PACW) clickable → **smooth-scroll
  to that BA's card and briefly highlight it**, staying on the US Grid tab.
  Staying in-tab keeps context and side-steps Problem 3 entirely.
- **Lowest Reserve:** it resolves to the *same BA* as Highest-Stress (see
  Problem 2), so don't duplicate the name in the value (`PACW · 69%` beside
  `PACW · 31%` reads worse). Instead define it as the complement in a tooltip;
  its scroll-to-card target is that same card.
- **Mechanism:** cards already have deterministic ids
  (`{"type":"us-grid-region-card","region":…}`). A clientside callback scrolls to
  the id + toggles a highlight class — no server round-trip, no data change.

### Fix 3 — coherent drilldown (recommended: a grid-context strip on Forecast)

- **(a)** When the Forecast tab is scoped to a BA, show a compact **"current grid
  snapshot"** strip at the top: the *same four numbers the card showed* — current
  demand, utilization, reserve, net interchange — so the drilldown carries
  continuity into the forecast. Smallest bridge, biggest coherence win.
  *Recommended.*
- **(b)** Reconcile units: render the Forecast metrics in **GW** (or GW + MW) so
  `55.0 GW` on the grid isn't `55,000 MW` on the forecast.
- **(c)** A short "← from US Grid · <BA name>" breadcrumb so the transition is
  explained.
- **(d) Bigger:** a dedicated per-BA **detail view** (current conditions +
  forecast + models in one place) instead of reusing the generic Forecast tab.
  Larger; defer until the strip proves insufficient.

### Fix 4 — make the two demand summaries explicit (don't force them to match)

They measure different things on purpose (Overview = *observed*, backward;
Forecast = *predicted*, forward), so identical windows would be *wrong*. Make the
framing explicit instead:

- **Overview:** label the metrics as observed actuals over a fixed 7-day window —
  e.g. "7-Day Peak (actual)" + tooltip "observed demand, last 168h." *(Average
  already reads "7d hourly mean" — extend that pattern to Peak/Low.)*
- **Forecast:** put the horizon *in* the metric labels and make it dynamic —
  "Forecast Peak · next 7d" that re-reads "next 24h" / "next 30d" as the horizon
  control changes + tooltip "predicted demand over the selected horizon."
- **Range / hourly ramp** (Forecast-only): keep them, but state the window
  ("over next {horizon}") so they don't look orphaned.

### Fix 5 — one coherent model story, with the ensemble's role made honest

- **Pick one rule** for "primary model" and use it on every tab. *Recommended:*
  keep featuring the **Ensemble** (it is what actually ships) but stop presenting
  it as the accuracy winner — label it "Ensemble · production blend" and pair it
  with the best single model, e.g. "Most accurate single model: XGBoost 3.70%."
  That reconciles Overview with Forecast/Backtest and kills the "why the worse
  number?" confusion in one line.
- **Explain the ensemble** (ties to Fix 1): a tooltip — "Blended forecast; chosen
  for stability across conditions, not the single lowest error. See Models tab."
  Reuses the honest framing already in `BACKTEST_RESULTS.md`.
- **Align the Forecast selector**: either default it to the same featured model or
  add a small "best for this region: XGBoost" hint beside the selector, so the
  three tabs stop disagreeing.

---

## Phased sequence (lowest-risk / highest-leverage first)

- **Phase 1 — definitions & honest labels (½–1 day, additive, no data/nav
  change):** KPI `ⓘ` tooltips (`build_metrics_bar` `help` field) + card-badge
  legend + the three US-Grid label clarifications + define *Lowest Reserve* as
  the complement of Highest-Stress (same BA) in its tooltip + **make every
  peak/avg/min metric state its window** (Overview
  "7-day actual", Forecast "next {horizon}") + **relabel the Overview model
  card** ("Ensemble · production blend" with a best-single-model line and a
  what-is-the-ensemble tooltip). Fixes Problems 1, 4, and the presentational
  half of 5.
- **Phase 2 — actionable KPIs:** clientside scroll-to-card + highlight for
  *Highest-Stress* and *Lowest Reserve*.
- **Phase 3 — coherence:** the grid-context snapshot strip on the Forecast tab
  (unit-consistent) + breadcrumb + **reconcile the model story across
  Overview/Forecast/Backtest** (one primary-model rule; optional "best for this
  region" hint on the Forecast selector).
- **Phase 4 — optional:** glossary; per-BA detail view.

## Recommended first slice

**Phase 1.** It's the highest-leverage (addresses the core "unreadable /
inconsistent for someone new" complaint — now covering Problems 1, 4, and the
presentational half of 5), the lowest-risk (additive tooltips + a legend + label
copy — no data, no navigation, no callback restructuring), and self-contained.
Phases 2 and 3 build cleanly on top.

## Open question — RESOLVED (2026-07-04)

*Highest-Stress Region* and *Lowest Reserve* were the same BA showing
complementary numbers (util% and 100−util%). **Resolved:** *Lowest Reserve* is
replaced by **National Utilization** (Σdemand ÷ Σnameplate capacity over the
reliable-capacity BA set) — the national *average* that complements
Highest-Stress's per-BA *maximum*, killing the degeneracy.

A national NERC reserve-margin roll-up was considered and **rejected for now**:
`REGION_CAPACITY_MW` is EIA-860M *nameplate*, not NERC-accredited, so a literal
`(cap−peak)/peak` reads ~66% nationally (vs a real ~15-25%). Rather than fake it
with guessed ELCC factors, we labeled every nameplate-based number honestly
("capacity headroom" / "utilization", never "reserve margin") and filed the
accredited-capacity (ELCC) model as [#243](https://github.com/kristenmartino/gridpulse/issues/243).
This also resolves the presentational half of #223.

---

## Phase 1 — Haiku execution breakdown (file-partitioned, additive)

Each task owns **one file** (no parallel-edit conflicts). The shared component
change lands **first**; the three consumers then run in parallel. All copy is
fixed below so agents don't improvise. Shared ⓘ affordance (identical everywhere):

```python
html.Span("ⓘ", title=<HELP TEXT>, className="gp-metric-help",
          style={"marginLeft": "4px", "opacity": 0.45, "cursor": "help", "fontSize": "0.85em"})
```

**Task 0 — foundation · `components/cards.py`** *(must complete first)*
- `build_metrics_bar`: support an optional `help` key per item. When present,
  wrap the label in a `html.Div([label, ⓘ-span])` using the snippet above.
  Backward-compatible (no `help` → unchanged).
- `build_model_metrics_card`: add `caption: str | None = None`. When present,
  render one muted line (`gp-model-card__caption`) under the model name.

**Task 1 — `components/_callbacks_us_grid.py`** (top KPIs + card legend)
- `_build_us_grid_metrics_items`: add `help=` to the 4 items:
  - Total Demand → "Sum of current demand across all reporting balancing authorities (GW)."
  - National Peak (24h) → "Highest simultaneous cross-grid demand in the last 24h. BAs peak at different hours, so this exceeds current total demand."
  - Highest-Stress Region → "BA with the highest utilization = current demand ÷ estimated capacity (capped 100%). Import-dominated BAs excluded."
  - Lowest Reserve → "Operating headroom (100% − utilization) of the most-stressed BA — the complement of Highest-Stress Region."
- Card grid (assembled ~line 857): add a muted legend line above the grid:
  "Each card: current demand (GW) · utilization vs. capacity (%) · net interchange (GW) · change vs. previous hour (%)."

**Task 2 — `components/_callbacks_overview.py`** (window labels + honest model card)
- `_build_overview_metrics_items`: add `help=` making the *actuals* framing explicit:
  - 7d Peak → "Highest observed (actual) demand in the last 168h (7 days)."
  - 7d Low → "Lowest observed (actual) demand in the last 168h (7 days)."
  - Average → "Mean observed (actual) demand over the last 168h (7 days)."
  - Now → "Most recent actual demand reading (EIA-930)."
  - 24h Trend → "Percent change from ~24h ago to now."
- `_build_overview_model_card`: when `primary_key == "ensemble"`, compute the
  best single model (min MAPE among non-ensemble keys in `metrics_dict`) and pass
  `caption=f"Production blend — most accurate single model: {NAME} {mape:.1f}%"`
  (fallback caption "Production blend — combines all models for stability." if
  no single-model MAPE available).

**Task 3 — `components/tab_demand_outlook.py`** (forecast-over-horizon framing)
- `_metrics_bar()`: add `title=` to each of the 4 label `Div`s:
  - Peak Demand → "Highest forecasted demand over the selected horizon (default 7 days)."
  - Average → "Mean forecasted demand over the selected horizon."
  - Min Demand → "Lowest forecasted demand over the selected horizon."
  - Range → "Forecast peak − forecast min over the selected horizon."
  *(Dynamic "next 24h/7d/30d" labels need a new callback output — deferred to Phase 3.)*

**Verify** — `ruff format` + `ruff check` each changed file; import all four
modules; diff-review each change against this spec.
