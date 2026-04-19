# EXECUTION_BRIEF.md — GridPulse Rebrand, Redesign, and Execution Plan

_Last updated: 2026-04-09_

## 1. Purpose

This file is the **agent-ready decision and execution layer** for the GridPulse repo.

It is designed for coding agents such as Claude Code and Codex. The goal is to help an agent:
- understand what GridPulse should become
- prioritize work without getting lost in repo details
- identify the highest-leverage changes first
- start implementing in a practical sequence
- avoid rewriting large parts of the repo without a reason

This file is not meant to replace the existing repo docs. It should be used **with**:
- `CLAUDE.md` for repo conventions and architecture
- `README.md` for current public framing
- `PRD.md` for product requirements and ADRs
- `TECHNICAL_SPEC.md` for implementation details

---

## 2. Agent operating instructions

When working in this repo:

1. Read `CLAUDE.md` first.
2. Read this file fully.
3. Inspect the current relevant files before editing anything.
4. Prioritize changes using the P0 → P2 ordering in this file.
5. Preserve working functionality unless explicitly told to break or replace it.
6. Prefer small, reviewable improvements over broad rewrites.
7. Avoid changing multiple conceptual layers at once unless the task requires it.
8. For UI changes, preserve current data flows and callback logic where possible.
9. If documentation and code disagree, verify the code and update docs carefully.
10. After each major change, summarize:
   - what changed
   - why it changed
   - what files were affected
   - what remains next

### Working style
- Inspect first
- Plan briefly
- Implement in small increments
- Validate after changes
- Keep diffs understandable

### What not to do
- Do not rewrite the forecasting or data pipeline layers just because the UI or brand is changing.
- Do not convert the app to another framework unless explicitly instructed.
- Do not remove existing personas, forecasting logic, or dashboard features without confirming they are obsolete.
- Do not let marketing or branding work drift into fake claims unsupported by the repo.

---

## 3. Current repo reality

GridPulse is currently an energy demand forecasting dashboard built with Dash/Plotly and deployed on Cloud Run. It already includes:
- weather-aware forecasting
- multiple models
- role-based personas
- regional selection
- backtesting
- generation and net load views
- alerts/extreme events concepts
- a simulator/scenario capability
- meeting / presentation mode
- supporting technical documentation

The current product is strong conceptually, but its external framing and design language still lean more toward **portfolio dashboard / analytics prototype** than **premium energy intelligence platform**.

This execution brief is intended to close that gap.

---

## 4. Strategic objective

Reposition GridPulse from:

> an energy demand forecasting dashboard

into:

> an energy intelligence platform for forecast confidence, grid visibility, and operational decision support

This repositioning should happen without destroying the technical credibility already present in the repo.

---

## 5. Brand and product direction

### Recommended master brand
**GridPulse**

Keep the name unless there is a later strategic decision to move beyond energy-specific positioning.

### Recommended descriptor
**Energy Intelligence Platform**

### Recommended tagline
**See demand sooner. Decide with confidence.**

### Core positioning
GridPulse should be presented as a platform that unifies:
- demand forecasting
- confidence and model visibility
- grid and generation context
- risk signals
- scenario planning
- stakeholder-ready briefings

### Product personality
- credible
- calm
- precise
- modern
- technically literate
- operationally useful

### Design direction
**Modern Utility Control Room**

Not:
- generic admin dashboard
- consumer app
- hype-AI landing page
- cyberpunk energy brand

---

## 6. Product suite direction

GridPulse should be designed so it can expand into a modular product suite.

### Recommended module architecture
- **GridPulse Forecast** — demand forecasting and outlook
- **GridPulse Risk** — alerts, severe conditions, anomalies, forecast instability
- **GridPulse Grid** — generation mix, net load, regional operating context
- **GridPulse Scenarios** — simulation and what-if analysis
- **GridPulse Models** — backtests, model confidence, drift, validation
- **GridPulse Briefings** — narrative summaries, snapshots, presentation-ready outputs
- **GridPulse API** — future integrations layer

### Principle
Do not build all modules at once. Use this architecture to guide naming, IA, landing pages, and future extensibility.

---

## 7. Current UX and framing problems

These are the major issues this brief is trying to solve.

### Problem A — external framing is too narrow
The repo and product are framed primarily as a forecasting dashboard, which undersells the broader system intelligence already present.

### Problem B — design language is too theme-driven
The current UI has solid structure, but it still reads as Bootstrap-dark-dashboard rather than a distinctive product.

### Problem C — information architecture is mixed
The app includes multiple tabs and concepts, but the product story is not yet expressed as a clear operating workflow.

### Problem D — core signals are not elevated enough
Forecast confidence, risk, and narrative insight should be more central to the user experience.

### Problem E — suite readiness is implicit, not explicit
The foundation for module expansion exists conceptually, but naming and navigation do not yet make that future legible.

---

## 8. Priority framework

### P0 — highest leverage
These tasks should be tackled first because they improve coherence without requiring deep architectural churn.

1. Product positioning and naming cleanup
2. Navigation / IA cleanup
3. Visual system refresh
4. Overview screen redesign
5. Forecast screen refinement
6. Landing page and external copy alignment

### P1 — important next
1. Risk screen consolidation
2. Briefings / intelligence layer cleanup
3. Scenarios redesign
4. Models screen redesign and terminology cleanup
5. Documentation updates for new framing

### P2 — later expansion
1. Suite module pages
2. API and enterprise/platform marketing pages
3. richer responsive/mobile patterns
4. more formal design system packaging
5. broader screenshot and sales asset system

---

## 9. Constraints

### Technical constraints
- Keep Dash/Plotly.
- Preserve existing callback architecture unless there is a specific reason to refactor.
- Do not destabilize the data, modeling, caching, or scheduled-jobs layers for branding work.
- Preserve deployment compatibility.
- Keep tests passing where applicable.

### Product constraints
- Do not remove existing operational value just to simplify visuals.
- Preserve personas and region-selection capability.
- Keep model credibility and backtesting visible.
- Keep the system grounded in real grid and weather data concepts.

### Copy / marketing constraints
- Do not make unsupported customer, scale, or performance claims.
- Avoid exaggerated AI language.
- Lead with operational clarity and confidence, not empty hype.

---

## 10. Recommended navigation model

The current product should gradually move toward a clearer decision-oriented information architecture.

### Preferred top-level structure
1. **Overview**
2. **Forecast**
3. **Risk**
4. **Grid**
5. **Scenarios**
6. **Models**
7. **Briefings**
8. **Settings**

### Suggested mapping from likely current concepts
- Historical Demand → Overview or History
- Demand Forecast → Forecast
- Alerts / weather / instability → Risk
- Generation & Net Load → Grid
- Scenario Simulator → Scenarios
- Backtest / model diagnostics → Models
- AI briefing / news / saved views → Briefings

### Important note
This does **not** mean everything must be physically rewritten at once. The agent may start by changing labels, grouping concepts more clearly, or updating top-level navigation components before performing deeper content restructuring.

---

## 11. Recommended design system direction

### Theme
Modern Utility Control Room

### Color direction
Shift away from heavy red as the primary accent.

#### Recommended palette intent
- deep obsidian and midnight surfaces
- cloud / cool-gray text
- cyan / blue for primary selection and active states
- teal for confidence and intelligence cues
- lime for positive reserve/healthy status where appropriate
- amber and coral-red for warnings and critical states only

### Typography
- Display: **Sora**
- UI/body: **Inter**
- Optional mono: **JetBrains Mono**

### Component behavior
- restrained motion
- subtle borders and surface depth
- more hierarchy in KPI cards
- stronger focus states
- less admin-theme feel

---

## 12. UX redesign principles

1. Signal first
2. Confidence must be visible
3. Risk should feel calm, not noisy
4. Overview should become the strongest demo screen
5. Forecast should become the strongest analytical screen
6. Scenarios should feel interactive, not form-heavy
7. Models should provide trust and accountability
8. Briefings should unify intelligence, not feel bolted on

---

## 13. Workstreams

## Workstream A — external positioning and copy
Goal: update the product’s external framing.

### Outcomes
- stronger README framing
- better homepage/landing page language
- consistent module naming
- cleaner descriptor/tagline usage

## Workstream B — navigation and IA
Goal: make the product feel like a coherent operating workflow.

### Outcomes
- clearer top-level labels
- improved navigation structure
- clearer mapping of current tabs into product concepts

## Workstream C — visual refresh
Goal: make the app look more distinctive and productized.

### Outcomes
- new color tokens
- updated typography tokens
- improved card, nav, selector, and panel styling
- reduced theme-default feel

## Workstream D — screen redesign
Goal: improve the most important user flows.

### Outcomes
- Overview redesign
- Forecast redesign
- Risk restructuring
- Models cleanup
- Scenarios polish
- Briefings layer consolidation

## Workstream E — landing page and suite architecture
Goal: support commercial or portfolio storytelling.

### Outcomes
- homepage structure
- module architecture
- suite naming consistency
- stronger screenshot narrative

---

## 14. Task backlog

Each task includes:
- priority
- rationale
- likely files
- acceptance criteria

### GP-001 — Reframe product language in docs
**Priority:** P0  
**Why:** The current framing is too dashboard-centric.  
**Likely files:** `README.md`, possibly `PRD.md`, marketing docs added later  
**Acceptance criteria:**
- README describes GridPulse as an energy intelligence platform
- forecasting remains central, but confidence/risk/grid context are also visible
- language stays grounded in repo reality

### GP-002 — Add agent-facing execution brief
**Priority:** P0  
**Why:** Gives future agents a prioritization layer.  
**Likely files:** `EXECUTION_BRIEF.md`  
**Acceptance criteria:**
- file exists
- prioritization is explicit
- task order is usable

### GP-003 — Introduce design tokens for refreshed theme direction
**Priority:** P0  
**Why:** Visual consistency requires tokenized colors/typography/spacing direction.  
**Likely files:** `assets/custom.css`, possible config or token docs  
**Acceptance criteria:**
- updated surface, text, and accent colors
- reduced reliance on current red-as-primary usage
- styling remains accessible and coherent

### GP-004 — Clean up top-level labels/navigation
**Priority:** P0  
**Why:** Navigation currently undersells product coherence.  
**Likely files:** `components/layout.py`, `config.py`, relevant tab labels/constants  
**Acceptance criteria:**
- top-level labels align better with Overview / Forecast / Risk / Grid / Scenarios / Models / Briefings
- no broken callbacks due to label changes
- naming is consistent across selectors, cards, and visible UI labels

### GP-005 — Redesign Overview as mission-control screen
**Priority:** P0  
**Why:** Overview should become the strongest entry and demo surface.  
**Likely files:** `components/tab_overview.py`, `components/cards.py`, `components/callbacks.py`, `assets/custom.css`  
**Acceptance criteria:**
- overview clearly answers what changed, what matters, and where risk/confidence stand
- stronger hierarchy than generic tile layout
- narrative insight and key signals are visible quickly

### GP-006 — Refine Forecast screen
**Priority:** P0  
**Why:** Forecast is the product’s analytical center of gravity.  
**Likely files:** `components/tab_demand_outlook.py`, callbacks, CSS  
**Acceptance criteria:**
- confidence bands are clear and elegant
- forecast horizon and model comparison are easier to interpret
- screen better communicates confidence and change over time

### GP-007 — Consolidate Risk concepts
**Priority:** P1  
**Why:** Alerts, conditions, and instability should read as one operational idea.  
**Likely files:** `components/tab_alerts.py`, `components/tab_weather.py`, related cards/callbacks  
**Acceptance criteria:**
- risk-oriented grouping is clearer
- warning states are useful without being noisy
- copy and visual hierarchy support rapid scanning

### GP-008 — Reframe model/backtest area as Models
**Priority:** P1  
**Why:** Better naming and clearer accountability layer.  
**Likely files:** `components/tab_models.py`, `components/tab_backtest.py`, labels/docs  
**Acceptance criteria:**
- users can understand this area as trust/validation/model oversight
- naming is consistent with the product-suite direction

### GP-009 — Upgrade Scenarios experience
**Priority:** P1  
**Why:** A polished scenario layer is a strong differentiator.  
**Likely files:** `components/tab_simulator.py`, callbacks, CSS  
**Acceptance criteria:**
- scenarios feel interactive and decision-oriented
- before/after impact is easy to understand
- controls do not feel like a raw settings panel

### GP-010 — Consolidate intelligence / briefing layer
**Priority:** P1  
**Why:** News, AI briefings, and meeting mode should feel like one coherent communication layer.  
**Likely files:** `components/cards.py`, `components/layout.py`, any briefing-related files/callbacks  
**Acceptance criteria:**
- intelligence outputs feel intentional and integrated
- presentation/briefing mode is more polished
- naming supports stakeholder-ready usage

### GP-011 — Build landing page implementation scaffold
**Priority:** P0/P1 depending on request  
**Why:** External presentation matters for product strategy and demos.  
**Likely files:** may be new web/docs files or separate site assets  
**Acceptance criteria:**
- homepage sections reflect platform positioning
- module architecture is visible
- copy is coherent and premium without unsupported claims

### GP-012 — Refresh screenshots/demo assets
**Priority:** P2  
**Why:** High-value for presentation, but secondary to product coherence.  
**Likely files:** docs/assets or external design artifacts  
**Acceptance criteria:**
- screenshots reflect updated styling and framing
- reusable for README, landing page, and demos

---

## 15. Recommended first 10 actions for an agent

1. Read `CLAUDE.md`, `README.md`, and this file.
2. Inspect current navigation labels and visible top-level product terminology.
3. Update `README.md` framing to align with the new platform positioning.
4. Inspect `assets/custom.css` and identify the smallest safe token-level visual refresh.
5. Update accent strategy so red becomes semantic rather than primary.
6. Inspect `components/layout.py` and current tab labels.
7. Propose or implement label cleanup toward the target IA.
8. Inspect the Overview screen and identify what should become the primary signal hierarchy.
9. Implement a first-pass Overview redesign without changing the underlying data pipeline.
10. Refine the Forecast view next, keeping callbacks stable where possible.

---

## 16. Definition of done for phase 1

Phase 1 is done when:
- README and visible product framing describe GridPulse as an energy intelligence platform
- the UI no longer reads primarily as a generic dark dashboard theme
- Overview and Forecast feel like signature screens
- navigation labels are clearer and more coherent
- red is no longer the main visual identity color
- the product feels more premium without losing technical credibility

---

## 17. Landing page execution guidance

If the agent is asked to create a landing page or landing-page copy, the page should communicate:
1. GridPulse is an energy intelligence platform
2. It unifies forecast confidence, grid visibility, and decision support
3. It is built for real energy workflows
4. It can expand into modules
5. It is technically credible

### Recommended homepage sections
- Hero
- credibility strip
- value pillars
- product modules
- workflow audience section
- platform screenshot narrative
- technical credibility
- suite/platform vision
- closing CTA

### Recommended homepage headline
**Operational clarity for a volatile grid.**

### Recommended homepage subhead
GridPulse unifies demand forecasting, forecast confidence, grid visibility, and scenario-based decision support in one energy intelligence platform.

---

## 18. Copy rules

Prefer language like:
- operational clarity
- forecast confidence
- grid visibility
- decision support
- scenario planning
- model accountability
- energy intelligence
- signals
- readiness

Avoid language like:
- revolutionary
- disruptive
- cutting-edge
- game-changing
- AI-powered everything

Keep copy:
- concise
- specific
- credible
- technically literate

---

## 19. Open questions that should not block execution

These can be resolved later unless explicitly requested:
- whether GridPulse should eventually be renamed
- whether Briefings becomes Intelligence as a module name
- whether Weather remains a separate view or is fully absorbed into Risk
- how formal the suite architecture becomes in code vs. marketing
- whether a separate marketing site exists or will be created later

Do not let these questions block P0 execution.

---

## 20. Suggested future repo docs

These are optional and can be created later if useful:
- `AGENTS.md` — tool-neutral version of `CLAUDE.md`
- `BRAND_SYSTEM.md` — finalized brand system doc inside repo
- `LANDING_PAGE_SPEC.md` — implementation-ready homepage spec
- `DESIGN_TOKENS.md` — documented token system for UI and web

---

## 21. Final directive

The highest-value outcome is not a decorative redesign.

The highest-value outcome is this:

> GridPulse should feel like the operating layer for modern energy decisions.

Any implementation work should move the product closer to that outcome while preserving the strongest parts of the existing repo: technical credibility, forecasting depth, personas, and operational relevance.
