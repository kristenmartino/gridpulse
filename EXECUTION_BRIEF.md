# EXECUTION_BRIEF.md — GridPulse Redesign and Repositioning Master Brief

_Last updated: 2026-04-19_

> **Pipeline scheduling status:** Done. The web service is Redis-only in
> production; hourly scoring and daily training run as Cloud Run Jobs.
> Details: [`docs/SCHEDULED_JOBS.md`](docs/SCHEDULED_JOBS.md).

## 1. Mission

GridPulse is evolving from a technically credible energy demand forecasting dashboard into a more cohesive **energy intelligence platform**.

The repo already contains substantial product and engineering value:
- Dash/Plotly application architecture
- role-based personas
- regional energy views
- demand forecasting
- backtesting and model validation
- generation / net load context
- alerts and scenario concepts
- meeting/presentation mode
- real data + demo fallback patterns

The next phase is to make the product read more clearly as a **premium, decision-oriented platform** rather than a collection of dashboard tabs.

### Strategic objective
Improve product positioning, information architecture, terminology, and UI coherence **without breaking core functionality**.

### North star
GridPulse should feel like:
- an energy intelligence operating layer
- a credible product for operators, analysts, and technical stakeholders
- a modular platform that can expand into a suite

---

## 2. How the agent should operate

## Required process
1. Read `CLAUDE.md` first.
2. Read this file fully.
3. Inspect the current repo structure before editing.
4. Prioritize work from P0 to P2.
5. Start with the highest-leverage changes that improve structure, naming, and coherence.
6. Preserve working functionality unless explicitly asked to change behavior.
7. Make changes in small, reviewable increments.
8. Prefer updates that strengthen product positioning, IA, and design-system consistency.
9. Validate after each meaningful change.
10. Summarize what changed, why, and what remains.

## Guardrails
- Do not rewrite unrelated modules.
- Do not introduce unnecessary architectural churn.
- Do not degrade accessibility.
- Do not remove technical credibility in favor of superficial marketing polish.
- Do not replace the product’s operational character with generic startup branding.

---

## 3. Source-of-truth files

Read these in this order when relevant:
1. `CLAUDE.md` — repo architecture, conventions, standards
2. `README.md` — current product framing
3. `PRD.md` — product intent, personas, ADRs, requirements
4. `TECHNICAL_SPEC.md` — data/model/system details
5. `components/layout.py` — current IA and top-level shell
6. `assets/custom.css` — current visual design layer
7. tab modules under `components/` — current experience by screen

This file is the **execution and prioritization layer** on top of those sources.

---

## 4. Current state assessment

## What is already strong
- Real operational product substance exists.
- Product has a distinct energy/grid context.
- Personas provide decision-context differentiation.
- Forecasting and model validation create technical credibility.
- Meeting mode and multi-view analytics support demos and stakeholder walkthroughs.
- The app already hints at platform potential.

## What is currently weak or inconsistent
- Product framing still reads as “dashboard” more than “platform.”
- Navigation structure feels tab-heavy and mixed in abstraction level.
- The current dark theme is workable but still feels close to a stock admin/dashboard theme.
- Terminology is inconsistent across product, marketing, and strategic framing.
- Risk, confidence, and narrative insight are present but not yet first-class organizing ideas.
- Current styling leans too heavily on navy/red; red is better reserved for risk and severe states.

---

## 5. Desired end state

GridPulse should present as:
- **Category:** Energy Intelligence Platform
- **Positioning:** Forecast confidence, grid visibility, and decision support
- **Tagline:** See demand sooner. Decide with confidence.

## Product experience goals
- clearer hierarchy
- calmer premium visual system
- stronger screen purpose by module
- more coherent navigation
- confidence and risk surfaced earlier
- easier path to future module/suite expansion

---

## 6. Explicit priorities

## P0 — Must do first
These items create the most leverage and should happen before lower-value polishing.

1. Clarify product language and positioning in-product
2. Clean up primary information architecture / naming
3. Replace visual dependence on the current Bootstrap-dark-theme feel
4. Redefine the core top-level experience around decision-oriented modules
5. Improve header, navigation, and key overview surfaces

## P1 — Important after P0
1. Strengthen Forecast, Risk, and Models screen hierarchy
2. Upgrade design tokens and reusable visual styles
3. Improve Briefing / presentation patterns
4. Create stronger landing page / module marketing structure
5. Tighten chart styling and semantic color usage

## P2 — Valuable but not first
1. Broader suite scaffolding
2. Additional narrative layers and exports
3. Expanded mobile awareness patterns
4. More advanced screenshot/marketing asset generation

---

## 7. Product positioning decisions

## Master brand
Keep **GridPulse** as the master brand.

## Product descriptor
Use one of these consistently:
- **Energy Intelligence Platform**
- **Forecast Confidence & Grid Visibility**

Recommended:
**Energy Intelligence Platform** for external use
**Forecast Confidence & Grid Visibility** as a secondary supporting line in-product where useful

## Tagline
Use:
**See demand sooner. Decide with confidence.**

## Brand promise
Operational clarity for a volatile grid.

---

## 8. Naming and IA decisions

## Recommended top-level nav model
Use this as the target structure:
1. Overview
2. Forecast
3. Risk
4. Grid
5. Scenarios
6. Models
7. Briefings
8. Settings

## Map likely current concepts to target labels
- Historical Demand → Overview or History
- Demand Forecast → Forecast
- Backtest → Models or Validation
- Generation & Net Load → Grid
- Alerts / Extreme Events → Risk
- Weather / Weather Correlation → Risk or Conditions
- Scenario Simulator → Scenarios
- AI Briefing / news ribbon / summary layers → Briefings or Intelligence

## Label changes to prefer
- Persona → View or Role View
- Presentation Mode → Briefing Mode
- Bookmark → Save View or Save Snapshot
- Demand Forecasting & Analytics → Energy Intelligence Platform
- News Ticker → Grid Signals or Market & Grid Signals

---

## 9. Visual system direction

## Design direction
Modern Utility Control Room

## Desired visual attributes
- dark premium surfaces
- restrained electric accents
- calm, analytical polish
- clearer hierarchy
- reduced dashboard-theme feel
- less red as default brand accent

## Core palette direction
### Base neutrals
- `#0B1020` background
- `#11182D` surface-1
- `#17223B` surface-2
- `#263556` border
- `#F7FAFC` text primary
- `#DDE6F2` text secondary
- `#A8B3C7` text muted

### Brand accents
- `#38D0FF` pulse cyan
- `#4A7BFF` grid blue
- `#2DE2C4` aurora teal

### Semantic colors
- success `#2BD67B`
- warning `#FFB84D`
- danger `#FF5C7A`

## Typography
- Display: Sora
- UI/body: Inter
- Optional mono: JetBrains Mono

---

## 10. Workstreams

## Workstream A — Product shell and navigation
### Objective
Make the app feel like a platform with intentional decision flows.

### Likely files
- `components/layout.py`
- `components/callbacks.py`
- selected tab modules
- `config.py`

### Target outcomes
- clearer top-level structure
- cleaner header
- improved module naming
- better framing of region/view controls

---

## Workstream B — Visual design system refresh
### Objective
Replace the “theme-first dashboard” feel with a cleaner brand system.

### Likely files
- `assets/custom.css`
- reusable component modules
- chart styling helpers if present

### Target outcomes
- improved surfaces, borders, states, spacing, and hierarchy
- cyan/blue/teal accents for primary product identity
- red reserved more explicitly for risk states

---

## Workstream C — Overview / Forecast / Risk core experience
### Objective
Make the first-run experience stronger and more coherent.

### Likely files
- `components/tab_overview.py`
- `components/tab_forecast.py`
- `components/tab_demand_outlook.py`
- `components/tab_alerts.py`
- card/insight helpers

### Target outcomes
- stronger mission-control overview
- more elegant forecast experience
- risk surfaced as a first-class workflow

---

## Workstream D — Models / validation credibility layer
### Objective
Preserve and elevate technical trust.

### Likely files
- `components/tab_backtest.py`
- `components/tab_models.py`
- `models/`
- `data/audit.py`

### Target outcomes
- clearer model accountability
- stronger performance/validation narrative
- better organization of technical depth

---

## Workstream E — Briefings and narrative layer
### Objective
Unify AI briefing, presentation mode, and summary surfaces into a cleaner intelligence story.

### Likely files
- layout/header areas
- relevant card and briefing modules
- meeting/presentation mode logic

### Target outcomes
- coherent “Briefings” concept
- stronger stakeholder-ready output surfaces
- calmer alternative to a distracting ticker pattern

---

## Workstream F — Marketing and suite alignment
### Objective
Ensure product and external positioning match.

### Deliverables
- homepage messaging
- module architecture language
- suite naming
- screenshot direction

This may live outside the app codebase, but the terminology should align with in-product changes.

---

## 11. Prioritized task backlog

## P0 tasks

### GP-P0-01 — Add consistent product language
**Why:** The product currently reads too much like a dashboard and not enough like a platform.

**Tasks:**
- update visible product subtitle / supporting lines
- standardize terminology across primary shell
- align internal labels with target positioning where low-risk

**Likely files:**
- `components/layout.py`
- `assets/custom.css`
- tab headings / card headings

**Acceptance criteria:**
- the product can be described consistently as an energy intelligence platform
- visible copy does not conflict across major screens

---

### GP-P0-02 — Restructure or relabel top-level navigation
**Why:** Current nav is tab-heavy and mixed in abstraction.

**Tasks:**
- audit current tabs
- map them to target IA
- relabel without breaking behavior where possible
- identify which screens should remain primary vs secondary

**Likely files:**
- `components/layout.py`
- `config.py`
- tab modules

**Acceptance criteria:**
- top-level nav labels are clearer and more decision-oriented
- active modules feel like coherent product areas

---

### GP-P0-03 — Refresh core shell styling
**Why:** Current dark theme is directionally right but still too close to stock dashboard styling.

**Tasks:**
- update key color tokens
- shift default accent away from red
- improve header, nav, card, and panel styling
- preserve contrast and accessibility

**Likely files:**
- `assets/custom.css`

**Acceptance criteria:**
- interface reads as more premium and distinctive
- red is no longer the dominant default brand accent
- semantic meanings for warning/danger remain clear

---

### GP-P0-04 — Redesign Overview as a stronger mission-control screen
**Why:** This is likely the most important screen for demos, first impressions, and stakeholder understanding.

**Tasks:**
- elevate top KPIs and status narrative
- clarify “what changed / what matters” hierarchy
- add stronger pathways into Forecast, Risk, and Scenarios

**Likely files:**
- `components/tab_overview.py`
- card helpers
- callbacks as needed

**Acceptance criteria:**
- overview answers what changed, what matters, where risk is rising, and how confident we are
- overview feels like the central control screen

---

### GP-P0-05 — Improve header and workspace controls
**Why:** The current header is useful but still scaffold-like.

**Tasks:**
- strengthen logo/title hierarchy
- refine region and view controls
- simplify button prominence and spacing
- improve framing for briefing mode / save state if present

**Likely files:**
- `components/layout.py`
- `assets/custom.css`

**Acceptance criteria:**
- header feels intentional and premium
- controls are easier to scan and more coherent

---

## P1 tasks

### GP-P1-01 — Redesign Forecast screen hierarchy
**Goal:** Make forecast confidence and interpretation more legible.

### GP-P1-02 — Establish Risk as a first-class product area
**Goal:** Organize alerts/extreme events into a calmer operational risk surface.

### GP-P1-03 — Refine Models / Validation screen
**Goal:** Improve technical trust and model accountability.

### GP-P1-04 — Unify Briefings / meeting mode / intelligence surfaces
**Goal:** Make narrative layers more coherent.

### GP-P1-05 — Create reusable design token patterns
**Goal:** Reduce ad hoc visual inconsistency.

---

## P2 tasks

### GP-P2-01 — Suite scaffolding
Create terminology and navigational affordances that support future modules.

### GP-P2-02 — Mobile awareness patterns
Focus on overview, risk, and briefing surfaces.

### GP-P2-03 — External marketing alignment
Align landing page, screenshots, and module descriptions with in-product terminology.

---

## 12. Recommended first execution sequence

The agent should start here unless a human overrides priority.

### Step 1
Inspect:
- `CLAUDE.md`
- `README.md`
- `PRD.md`
- `components/layout.py`
- `assets/custom.css`
- overview/forecast/alerts/backtest tab files

### Step 2
Produce a short audit of:
- current nav labels
- current screen purposes
- visible product terminology
- current brand-color usage
- obvious P0 improvement opportunities

### Step 3
Implement GP-P0-01 and GP-P0-05 together if low-risk.

### Step 4
Implement GP-P0-03 to establish new visual tokens and shell styling.

### Step 5
Implement GP-P0-02 to improve top-level naming/IA.

### Step 6
Implement GP-P0-04 to strengthen Overview.

### Step 7
Reassess before moving into P1 Forecast/Risk/Models redesign work.

---

## 13. Definition of done

The work is successful when:
- GridPulse reads as an energy intelligence platform in the UI
- the header and top-level navigation feel intentional and coherent
- the default experience feels more premium and less like a stock dashboard theme
- Overview becomes a stronger control screen
- Forecast, Risk, and Models are easier to understand as distinct product areas
- terminology is cleaner and more consistent
- existing functionality remains intact unless intentionally changed

---

## 14. Nice-to-haves that should not block execution

These ideas are useful but should not block P0:
- full logo redesign inside the app
- advanced motion design
- deeper suite scaffolding
- complete mobile redesign
- large-scale architecture refactor unrelated to UX/product shell
- overproduction of marketing content inside core product work

---

## 15. Deliverables this brief should eventually enable

### In-repo product deliverables
- refined product shell
- stronger screen naming
- upgraded styling system
- clearer overview / forecast / risk / models surfaces
- better briefing-mode framing

### External deliverables
- landing page
- module pages
- screenshot set
- design system tokens
- suite architecture language

---

## 16. Final instruction to the agent

Do not treat this as a request for superficial theming.

Treat this as a request to improve:
- product positioning
- interaction hierarchy
- information architecture
- visual system coherence
- platform readiness

Preserve the product’s technical seriousness.
Make it calmer, clearer, and more valuable.
