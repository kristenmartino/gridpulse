# NextEra Energy Demand Forecasting Dashboard
## Master Backlog, Prioritization & PM Playbook

---

## Table of Contents

1. [Complete Feature & Enhancement Backlog](#backlog)
2. [Prioritization Framework](#prioritization)
3. [Prioritized Roadmap](#roadmap)
4. [Jira Board Setup](#jira)
5. [Workflow & Process](#workflow)
6. [Documentation Standards](#docs)
7. [Explicitly Descoped Items](#descoped)

---

## 1. Complete Feature & Enhancement Backlog {#backlog}

### Category A: Design / UI / Information Architecture

| ID | Item | Description | Complexity | Value |
|----|------|-------------|------------|-------|
| A1 | Progressive Disclosure (Briefing vs. Explore Mode) | Two depth modes per tab: VP-level headline KPIs with trend arrows (Briefing), and full interactive filter/slice experience (Explore). Default view is curated narrative; "Explore" unlocks all filters and chart interactions. | Medium | Critical |
| A2 | Annotation & Context Layers | Lightweight system for analysts to pin notes to specific data points or time ranges (e.g., "Hurricane Milton impact," "planned outage Zone 3"). Turns dashboard from monitoring tool into institutional memory. | Medium | High |
| A3 | Alert-Driven Entry Points | Anomaly-triggered notifications that deep-link directly to the relevant tab with the relevant time range pre-filtered. Dashboard comes to users rather than users hunting for anomalies. | Medium | High |
| A4 | Data Freshness Indicators Per Widget | Each individual widget shows its data vintage: "Pricing: 4 min ago" / "Weather: 2 hrs ago" / "Generation: real-time." Visible on each component, not buried in a page footer. | Low | High |
| A5 | Confidence Bands on Forecasts | Shaded uncertainty cone that widens as forecast horizon extends, replacing single-line point estimates. Communicates precision visually without requiring statistical literacy. | Low | High |
| A6 | Comparative Framing Everywhere | Design principle: never show a number alone. Every metric has automatic comparator — vs. yesterday, vs. same day last year, vs. forecast. "4,200 MW (↑6% vs. expected)" | Low | Critical |

### Category B: UX / Behavioral Design

| ID | Item | Description | Complexity | Value |
|----|------|-------------|------------|-------|
| B1 | Narrative Mode | Dashboard opens with auto-generated natural-language summary: "Today's demand forecast is 12% above seasonal baseline, driven by sustained heat in ERCOT Zone 3. Generation capacity is adequate but pricing is elevated." Templated sentence generation from threshold logic. | Medium | Critical |
| B2 | "What Changed Since Last Time?" | Session-aware diff surfaced on return: "Since your last visit: demand forecast revised upward 8%, two new weather alerts in Zone 5, pricing model v2.3 deployed." | Medium | High |
| B3 | Smart Defaults That Learn | localStorage remembering last 3 filter states per user. "Pick up where you left off" option. Not a full personalization engine — just remembers preferred region, time range, tab. | Low | Medium |
| B4 | Smooth Animated Transitions | Chart animations when switching time ranges (7-day to 30-day). Maintains spatial context. D3/charting libraries support natively. Difference between spreadsheet and product. | Low | Medium |

### Category C: Features / Functionality

| ID | Item | Description | Complexity | Value |
|----|------|-------------|------------|-------|
| C1 | Cross-Tab Contextual Links | When viewing a pricing spike, dashboard surfaces: "Related: demand anomaly detected on same timeline → view" linking to relevant chart on another tab with same time range pre-loaded. Event correlation on shared time axis. | Medium | High |
| C2 | Scenario Bookmarks | Save filter + time range + tab combinations as named shareable URLs. "Hurricane prep view," "Q1 budget review," "Zone 3 summer peak." Serialized filter state in URL params. | Low | High |
| C3 | Inline "Why" Tooltips on Model Outputs | Hover on forecast spike reveals top 3 contributing factors with weights: "temperature +4°F above normal (42%), weekday demand pattern (31%), industrial load increase Zone 2 (27%)." Model explainability in UX. | Medium | High |
| C4 | Time-Scrub Replay | Draggable playhead timeline at bottom of any tab. Scrub through time, all widgets animate together. Transforms post-event analysis from research task to 30-second visual scan. | Medium | Medium |
| C5 | Keyboard Command Palette | Cmd+K opens search bar: type "ERCOT 7-day pricing" to jump to right tab with right filters. Also searches saved scenarios by name. Creates power users who evangelize the tool. | Low | Medium |
| C6 | Decision Log | Sidebar where stakeholders record actions taken based on dashboard data: "Based on Zone 3 pricing spike at 14:00, dispatched peaker unit at 14:15 — J. Rodriguez." Organizational knowledge base connecting decisions to data states. Regulatory gold for FERC/NERC. | Medium | High |
| C7 | Contextual Mode Shifting | Information hierarchy adapts to operational context. Hurricane season: weather widgets auto-promote. Rate case periods: regulatory metrics surface. Based on calendar events, active alerts, or manual mode selection. | Medium | Medium |
| C8 | Context-Aware Export | Three export modes: "Briefing slide" (auto-formatted PowerPoint with narrative title), "Analyst dataset" (full CSV), "Copy chart to clipboard" (image). Closes loop between dashboard and meetings. | Medium | High |
| C9 | Meeting-Ready Mode | Single button strips navigation chrome, filters, sidebar. Reformats current view for projection/PDF. Narrative sentence becomes slide title, charts expand, annotations remain. | Low | High |

### Category D: Security

| ID | Item | Description | Complexity | Value |
|----|------|-------------|------------|-------|
| D1 | Row-Level Security by Balancing Authority Region | Data visibility scoped by user role and territory. Trivial to define at spec stage, nightmare to retrofit. Critical if dashboard serves different internal teams or eventually external 360 customers. | Medium | Critical |
| D2 | Forecast Model Input Audit Trail | Log which model version, weather data vintage, and feature encoding produced each forecast. Lineage for "what happened" investigations. Regulatory defensibility for FERC/NERC. | Medium | Critical |
| D3 | API Key Rotation & Rate Limiting | Define rotation cadence for NOAA/EIA/pricing feed keys, fallback behavior when external API is down (stale data indicator vs. hard failure), and rate limit handling. NOAA has documented rate limits and outages. | Low | High |

### Category E: Data Quality & Validation

| ID | Item | Description | Complexity | Value |
|----|------|-------------|------------|-------|
| E1 | Validation Rules Per Data Source | Define expected schema, acceptable value ranges, and null handling for NOAA, EIA, and pricing feeds. What happens when a weather station goes offline or pricing feed delivers nulls? | Medium | Critical |
| E2 | Staleness Thresholds | Per-source acceptable data age before triggering warnings. Weather: 2 hours. Pricing: 15 minutes. Generation: 5 minutes. Dashboard communicates data confidence to users. | Low | Critical |
| E3 | Data Confidence Communication | Visual system for communicating data quality to users. Pairs with A4 (freshness indicators) and E2 (staleness thresholds). Green/amber/red reliability badges per widget. | Low | High |

### Category F: Performance & Scalability

| ID | Item | Description | Complexity | Value |
|----|------|-------------|------------|-------|
| F1 | Target Query Response Times | Define p95 < 2s for any dashboard tab load. Specify per-tab performance budgets. | Low | Critical |
| F2 | Caching Strategy | Define caching rules: historical data (aggressive cache), near-real-time data (short TTL), forecast outputs (cache until model refresh). | Medium | Critical |
| F3 | Pre-Aggregation Rules | Define which rollups are pre-computed (daily/weekly/monthly aggregations) vs. computed on-demand. Prevents slow queries on large time ranges. | Medium | High |

### Category G: Error Handling & Degraded States

| ID | Item | Description | Complexity | Value |
|----|------|-------------|------------|-------|
| G1 | Per-Component Graceful Degradation | Define behavior for each widget when its upstream data source is unavailable: last-known-good with timestamp badge, grayed-out widget, partial view. Specified per tab. | Medium | Critical |
| G2 | External API Fallback Behavior | When NOAA/EIA/pricing feeds are down: show stale data with warning vs. error state vs. partial rendering. Must be explicitly defined per data source. | Low | High |

### Category H: Testing & Acceptance Criteria

| ID | Item | Description | Complexity | Value |
|----|------|-------------|------------|-------|
| H1 | Acceptance Criteria Audit | Every feature in spec has testable acceptance criterion. Gap analysis against current spec. | Medium | Critical |
| H2 | ML Model Accuracy Thresholds | Define accuracy threshold below which a model version gets rolled back. MAPE targets per forecast horizon. | Low | Critical |
| H3 | Test Pyramid Definition | Unit test coverage targets, integration test scope, E2E test cases for critical user flows. | Medium | High |

### Category I: Data Lineage & Observability

| ID | Item | Description | Complexity | Value |
|----|------|-------------|------------|-------|
| I1 | Pipeline Transformation Logging | Logging at each ETL transformation step. When a dashboard number looks wrong, engineer can trace back through pipeline. | Medium | High |
| I2 | Pipeline Health Monitoring | Monitoring and alerting on ETL pipeline health (not just dashboard health). Define what "healthy" means for each pipeline stage. | Medium | High |

### Category J: Configuration & Environment Management

| ID | Item | Description | Complexity | Value |
|----|------|-------------|------------|-------|
| J1 | Environment Config Matrix | Define which values are environment-configurable vs. hardcoded. How config differs between dev/staging/prod. | Low | High |
| J2 | Feature Flags | How to roll out new tabs, model versions, or features incrementally. Simple on/off flags per feature per environment. | Medium | Medium |

### Category K: Integration & API Contracts

| ID | Item | Description | Complexity | Value |
|----|------|-------------|------------|-------|
| K1 | External API Contract Definitions | For NOAA, EIA, pricing feeds: expected schema, response format, error codes, retry behavior, versioning. How to detect and handle upstream API format changes. | Medium | Critical |
| K2 | Internal Service Boundary Contracts | For internal service boundaries: define request/response schemas, error handling, timeout policies. | Medium | High |

### Category L: Accessibility & Compliance

| ID | Item | Description | Complexity | Value |
|----|------|-------------|------------|-------|
| L1 | WCAG 2.1 AA Compliance | Color-independent indicators (not just red/green), screen reader compatibility for data tables, keyboard navigation for all interactive elements. | Medium | High |
| L2 | Energy Regulatory Data Requirements | FERC/NERC requirements for data presentation, retention periods, and auditability. Pairs with D2 (audit trail). | Low | High |

### Category M: Deployment & Operations

| ID | Item | Description | Complexity | Value |
|----|------|-------------|------------|-------|
| M1 | Release & Rollback Strategy | Blue-green or canary deployment process. Rollback procedure for bad model versions. Release checklist. | Medium | Medium |
| M2 | Capacity Planning & Cost | Estimated compute/storage at current vs. projected scale. Cost review triggers. Prevent surprise cloud bills. | Low | Medium |

### Category N: Documentation & Change Management

| ID | Item | Description | Complexity | Value |
|----|------|-------------|------------|-------|
| N1 | Spec Sync Process | Lightweight process for keeping buildplan and expanded spec in sync. Audit already found drift. | Low | Medium |
| N2 | User Onboarding | In-app guidance or separate runbook for new analysts. How they learn to use the dashboard. | Medium | Medium |
| N3 | Change Communication | How users are notified of dashboard changes: release notes, in-app banners, etc. Prevents "who moved my chart" tickets. | Low | Medium |

---

## 2. Prioritization Framework {#prioritization}

### Scoring Methodology: RICE-Lite

Each item scored on two axes:

**Value** (to user adoption and operational impact):
- Critical = 4 — Dashboard fails or gets rejected without this
- High = 3 — Significant differentiation or risk reduction
- Medium = 2 — Nice improvement, noticeable quality lift
- Low = 1 — Polish, future-proofing

**Complexity** (engineering effort):
- Low = 1 — < 1 sprint, standard tooling
- Medium = 2 — 1–2 sprints, some integration work
- High = 3 — 3+ sprints, significant R&D or cross-system coordination

**Priority Score = Value / Complexity** (higher = do first)

---

## 3. Prioritized Roadmap {#roadmap}

### Tier 1: Non-Negotiable Foundation (Sprint 1–3)
*Without these, the dashboard is a demo, not a product.*

| ID | Item | Score | Rationale |
|----|------|-------|-----------|
| A6 | Comparative Framing Everywhere | 4.0 | Design principle, near-zero cost, transforms every widget |
| E2 | Staleness Thresholds | 4.0 | Trivial to define, prevents bad decisions |
| F1 | Target Query Response Times | 4.0 | One-line spec addition, governs all architecture decisions |
| D3 | API Key Rotation & Rate Limiting | 3.0 | Low effort, prevents production outages |
| G2 | External API Fallback Behavior | 3.0 | Low effort, critical operational resilience |
| H2 | ML Model Accuracy Thresholds | 4.0 | Low effort, defines model governance |
| A4 | Data Freshness Per Widget | 3.0 | Low effort, builds trust in every view |
| A5 | Confidence Bands | 3.0 | Low effort, honesty in forecasting |
| C2 | Scenario Bookmarks | 3.0 | Low effort, high adoption and communication value |
| J1 | Environment Config Matrix | 3.0 | Low effort, prevents dev/prod mismatches |

### Tier 2: Core Differentiation (Sprint 3–6)
*These make the dashboard remarkable — the features that get mentioned in exec reviews.*

| ID | Item | Score | Rationale |
|----|------|-------|-----------|
| B1 | Narrative Mode | 2.0 | Medium effort but single highest-impact adoption feature |
| A1 | Progressive Disclosure | 2.0 | Medium effort, solves VP vs. analyst problem completely |
| D1 | Row-Level Security | 2.0 | Medium effort, impossible to retrofit, blocks multi-team use |
| D2 | Forecast Model Input Audit Trail | 2.0 | Medium effort, regulatory requirement for FERC/NERC |
| E1 | Validation Rules Per Data Source | 2.0 | Medium effort, prevents catastrophic bad-data decisions |
| K1 | External API Contract Definitions | 2.0 | Medium effort, prevents integration failures |
| G1 | Per-Component Graceful Degradation | 2.0 | Medium effort, defines production behavior |
| H1 | Acceptance Criteria Audit | 2.0 | Medium effort, ensures testability of everything above |
| F2 | Caching Strategy | 2.0 | Medium effort, governs scalability |
| C9 | Meeting-Ready Mode | 3.0 | Low effort, closes the "dashboard to boardroom" gap |
| C8 | Context-Aware Export | 1.5 | Medium effort, but completes the meeting-ready story |

### Tier 3: Power User & Operational Excellence (Sprint 6–9)
*These separate a good dashboard from one analysts can't live without.*

| ID | Item | Score | Rationale |
|----|------|-------|-----------|
| B2 | "What Changed Since Last Time?" | 1.5 | Medium effort, respect users' time |
| A2 | Annotation & Context Layers | 1.5 | Medium effort, institutional memory |
| A3 | Alert-Driven Entry Points | 1.5 | Medium effort, proactive vs. reactive UX |
| C1 | Cross-Tab Contextual Links | 1.5 | Medium effort, makes 6 tabs feel like one product |
| C3 | Inline "Why" Tooltips | 1.5 | Medium effort, builds model trust |
| C6 | Decision Log | 1.5 | Medium effort, regulatory and operational value |
| E3 | Data Confidence Communication | 1.5 | Low effort (builds on E2 + A4), visual quality layer |
| F3 | Pre-Aggregation Rules | 1.5 | Medium effort, enables large time ranges |
| I1 | Pipeline Transformation Logging | 1.5 | Medium effort, debug capability |
| I2 | Pipeline Health Monitoring | 1.5 | Medium effort, operational maturity |
| L1 | WCAG 2.1 AA Compliance | 1.5 | Medium effort, required for 360 platform |
| L2 | Regulatory Data Requirements | 3.0 | Low effort, pairs with D2 |
| K2 | Internal Service Contracts | 1.5 | Medium effort, integration clarity |
| H3 | Test Pyramid Definition | 1.5 | Medium effort, quality assurance structure |

### Tier 4: Polish & Scale (Sprint 9+)
*Quality-of-life improvements that compound over time.*

| ID | Item | Score | Rationale |
|----|------|-------|-----------|
| C5 | Keyboard Command Palette | 2.0 | Low effort, creates power users |
| B3 | Smart Defaults That Learn | 2.0 | Low effort, "built for me" feeling |
| B4 | Smooth Animated Transitions | 2.0 | Low effort, spatial context retention |
| C4 | Time-Scrub Replay | 1.0 | Medium effort, powerful for post-event analysis |
| C7 | Contextual Mode Shifting | 1.0 | Medium effort, operational context awareness |
| J2 | Feature Flags | 1.0 | Medium effort, rollout safety |
| M1 | Release & Rollback Strategy | 1.0 | Medium effort, deployment maturity |
| M2 | Capacity Planning & Cost | 2.0 | Low effort, financial governance |
| N1 | Spec Sync Process | 2.0 | Low effort, prevents spec drift |
| N2 | User Onboarding | 1.0 | Medium effort, reduces support burden |
| N3 | Change Communication | 2.0 | Low effort, prevents user confusion |

---

## 4. Jira Board Setup {#jira}

### Project Configuration

**Project Name:** NEXD (NextEra Energy Dashboard)
**Project Type:** Scrum (2-week sprints)
**Board:** Kanban-style within Scrum framework

### Issue Types

| Type | Usage | Example |
|------|-------|---------|
| **Epic** | One per category (A through N) | "Design / UI / Information Architecture" |
| **Story** | One per backlog item | "As a VP, I see headline KPIs in Briefing mode so I can get a 10-second status read" |
| **Task** | Implementation work within a story | "Implement localStorage filter state persistence" |
| **Sub-task** | Granular work items | "Write unit tests for filter serialization" |
| **Bug** | Defects found during testing | "Staleness badge not updating on Weather widget" |
| **Spike** | Research/investigation needed | "Investigate NOAA API rate limit behavior under load" |

### Epic Structure

```
NEXD-EPIC-A: Design / UI / Information Architecture
  ├── NEXD-1: Progressive Disclosure (Briefing vs. Explore)
  ├── NEXD-2: Annotation & Context Layers
  ├── NEXD-3: Alert-Driven Entry Points
  ├── NEXD-4: Data Freshness Per Widget
  ├── NEXD-5: Confidence Bands on Forecasts
  └── NEXD-6: Comparative Framing (Design Principle)

NEXD-EPIC-B: UX / Behavioral Design
  ├── NEXD-7: Narrative Mode
  ├── NEXD-8: "What Changed Since Last Time?"
  ├── NEXD-9: Smart Defaults That Learn
  └── NEXD-10: Smooth Animated Transitions

NEXD-EPIC-C: Features / Functionality
  ├── NEXD-11: Cross-Tab Contextual Links
  ├── NEXD-12: Scenario Bookmarks
  ├── NEXD-13: Inline "Why" Tooltips on Model Outputs
  ├── NEXD-14: Time-Scrub Replay
  ├── NEXD-15: Keyboard Command Palette
  ├── NEXD-16: Decision Log
  ├── NEXD-17: Contextual Mode Shifting
  ├── NEXD-18: Context-Aware Export
  └── NEXD-19: Meeting-Ready Mode

NEXD-EPIC-D: Security
  ├── NEXD-20: Row-Level Security by Balancing Authority
  ├── NEXD-21: Forecast Model Input Audit Trail
  └── NEXD-22: API Key Rotation & Rate Limiting

NEXD-EPIC-E: Data Quality & Validation
  ├── NEXD-23: Validation Rules Per Data Source
  ├── NEXD-24: Staleness Thresholds
  └── NEXD-25: Data Confidence Communication

NEXD-EPIC-F: Performance & Scalability
  ├── NEXD-26: Target Query Response Times
  ├── NEXD-27: Caching Strategy
  └── NEXD-28: Pre-Aggregation Rules

NEXD-EPIC-G: Error Handling & Degraded States
  ├── NEXD-29: Per-Component Graceful Degradation
  └── NEXD-30: External API Fallback Behavior

NEXD-EPIC-H: Testing & Acceptance Criteria
  ├── NEXD-31: Acceptance Criteria Audit
  ├── NEXD-32: ML Model Accuracy Thresholds
  └── NEXD-33: Test Pyramid Definition

NEXD-EPIC-I: Data Lineage & Observability
  ├── NEXD-34: Pipeline Transformation Logging
  └── NEXD-35: Pipeline Health Monitoring

NEXD-EPIC-J: Configuration & Environment Management
  ├── NEXD-36: Environment Config Matrix
  └── NEXD-37: Feature Flags

NEXD-EPIC-K: Integration & API Contracts
  ├── NEXD-38: External API Contract Definitions
  └── NEXD-39: Internal Service Boundary Contracts

NEXD-EPIC-L: Accessibility & Compliance
  ├── NEXD-40: WCAG 2.1 AA Compliance
  └── NEXD-41: Energy Regulatory Data Requirements

NEXD-EPIC-M: Deployment & Operations
  ├── NEXD-42: Release & Rollback Strategy
  └── NEXD-43: Capacity Planning & Cost

NEXD-EPIC-N: Documentation & Change Management
  ├── NEXD-44: Spec Sync Process
  ├── NEXD-45: User Onboarding
  └── NEXD-46: Change Communication
```

### Custom Fields

| Field | Type | Purpose |
|-------|------|---------|
| Priority Tier | Single Select (1/2/3/4) | Roadmap tier alignment |
| Value Score | Number (1–4) | From prioritization framework |
| Complexity Score | Number (1–3) | From prioritization framework |
| Category | Single Select (A–N) | Cross-reference to backlog categories |
| Spec Section | Text | Links to section in buildplan/expanded spec |
| Dependencies | Issue Link | Blocked-by / blocks relationships |
| Acceptance Criteria | Text (multi-line) | Testable criteria per story |

### Board Columns (Workflow)

```
Backlog → Refinement → Ready → In Progress → In Review → QA → Done
```

| Column | Entry Criteria | Exit Criteria |
|--------|----------------|---------------|
| **Backlog** | Item exists in master backlog | — |
| **Refinement** | Prioritized for upcoming sprint | Story has acceptance criteria, estimated, dependencies identified |
| **Ready** | Refined, estimated, no blockers | Pulled into sprint |
| **In Progress** | Sprint commitment | Code complete, self-tested |
| **In Review** | PR submitted | Code review approved, no outstanding comments |
| **QA** | Review approved | Acceptance criteria verified, no regressions |
| **Done** | QA passed | Deployed to staging, demo-ready |

### Labels

```
label: spec-update      → Requires changes to buildplan or expanded spec
label: design-principle  → Not a feature, a standard to apply everywhere (e.g., A6)
label: regulatory        → FERC/NERC/compliance implications
label: quick-win         → Can be completed in < 1 day
label: cross-cutting     → Affects multiple tabs or components
label: spike-needed      → Requires research before estimation
```

### Sprint Cadence

- **Sprint Length:** 2 weeks
- **Sprint Planning:** Monday morning, Sprint Day 1
- **Daily Standup:** 15 min, async-friendly (Slack thread alternative)
- **Sprint Review/Demo:** Friday afternoon, Sprint Day 10
- **Retrospective:** Immediately after Sprint Review
- **Backlog Refinement:** Wednesday, Sprint Day 8 (prep for next sprint)

---

## 5. Workflow & Process {#workflow}

### Definition of Ready (DoR)

A story is "Ready" for sprint when ALL of the following are true:

- [ ] User story follows format: "As a [role], I [action] so that [outcome]"
- [ ] Acceptance criteria are written and testable
- [ ] Dependencies identified and unblocked (or explicitly accepted as risk)
- [ ] Complexity estimated (story points or T-shirt size)
- [ ] Design mockup or wireframe exists (for UI items)
- [ ] Spec section reference identified (which part of buildplan/expanded spec this touches)
- [ ] No open questions — all ambiguities resolved

### Definition of Done (DoD)

A story is "Done" when ALL of the following are true:

- [ ] Code complete and self-tested
- [ ] Code review approved (no outstanding comments)
- [ ] Acceptance criteria verified in QA
- [ ] No regressions in existing functionality
- [ ] Unit tests written (where applicable)
- [ ] Documentation updated (if user-facing change)
- [ ] Deployed to staging environment
- [ ] Demo-ready for Sprint Review
- [ ] Spec updated if implementation deviated from original spec

### Story Template (Jira Description)

```markdown
## User Story
As a [VP / analyst / engineer / system], I want to [action]
so that [measurable outcome].

## Context
[Why this matters. Link to backlog item ID. Reference to spec section.]

## Acceptance Criteria
- [ ] Given [context], when [action], then [expected result]
- [ ] Given [context], when [action], then [expected result]
- [ ] [Performance criteria if applicable]

## Technical Notes
[Architecture considerations, dependencies, known constraints]

## Out of Scope
[What this story explicitly does NOT include]

## Dependencies
- Blocked by: [NEXD-XX]
- Blocks: [NEXD-XX]

## Design Reference
[Link to mockup/wireframe if applicable]
```

### Spike Template

```markdown
## Research Question
[What do we need to learn?]

## Why This Matters
[What decision is blocked without this answer?]

## Time-Box
[Maximum time to spend: usually 1–3 days]

## Expected Output
[Document, prototype, recommendation, or decision]

## Success Criteria
[How do we know the spike answered the question?]
```

### Bug Template

```markdown
## Summary
[One-line description]

## Steps to Reproduce
1. [Step 1]
2. [Step 2]
3. [Step 3]

## Expected Behavior
[What should happen]

## Actual Behavior
[What actually happens]

## Environment
[Browser, OS, data source state, user role]

## Severity
- [ ] Critical: Dashboard unusable or data integrity at risk
- [ ] Major: Feature broken but workaround exists
- [ ] Minor: Visual/UX issue, not blocking decisions
- [ ] Trivial: Cosmetic

## Screenshots / Logs
[Attach if applicable]
```

---

## 6. Documentation Standards {#docs}

### Document Hierarchy

```
Confluence Space: NEXD - NextEra Energy Dashboard
├── 📁 Product
│   ├── Product Vision & Strategy (1-pager)
│   ├── Buildplan Overview (living document)
│   ├── Expanded Specification (living document)
│   ├── Master Backlog (this document, keep synced with Jira)
│   └── Roadmap (quarterly view)
│
├── 📁 Design
│   ├── Design System & Component Library
│   ├── Wireframes / Mockups (per feature)
│   ├── Design Principles (A6 comparative framing, etc.)
│   └── Accessibility Checklist
│
├── 📁 Engineering
│   ├── Architecture Decision Records (ADRs)
│   ├── API Contract Documentation
│   ├── Data Pipeline Documentation
│   ├── Environment Configuration Guide
│   ├── Deployment Runbook
│   └── Incident Response Playbook
│
├── 📁 Data & ML
│   ├── Model Registry & Version History
│   ├── Feature Engineering Documentation
│   ├── Model Accuracy Reports (per version)
│   ├── Data Source Catalog (NOAA, EIA, pricing feeds)
│   └── Data Quality Rules Reference
│
├── 📁 Process
│   ├── Sprint Ceremonies Guide
│   ├── Definition of Ready / Done
│   ├── Story / Spike / Bug Templates
│   ├── Release Process Checklist
│   └── Retrospective Notes Archive
│
└── 📁 Stakeholder
    ├── Sprint Review Decks
    ├── Release Notes Archive
    └── User Guide / Runbook
```

### Architecture Decision Records (ADRs)

Use ADRs for any non-obvious technical or product decision. Format:

```markdown
# ADR-001: [Decision Title]

## Status
[Proposed / Accepted / Deprecated / Superseded]

## Context
[What is the issue? Why does this decision need to be made?]

## Decision
[What was decided?]

## Consequences
[What becomes easier? What becomes harder? What are the trade-offs?]

## Alternatives Considered
[What else was evaluated and why was it rejected?]
```

**Trigger ADR creation when:**
- Choosing between two viable approaches
- Deviating from the spec
- Making a trade-off that affects future flexibility
- Deciding to NOT build something (document why)

### Spec Sync Protocol (Prevents Drift)

The audit already found buildplan and expanded spec drifting apart. Process to prevent recurrence:

1. **Single source of truth:** Expanded Spec is authoritative for implementation details. Buildplan is the executive summary.
2. **Sync check:** At every Sprint Review, compare any implemented changes against both documents.
3. **Update rule:** If implementation deviates from spec, update the spec in the same sprint. Never let it become "tech debt we'll fix later."
4. **Ownership:** PM owns buildplan. Tech lead owns expanded spec. Both review each other's changes.

---

## 7. Explicitly Descoped Items {#descoped}

Items considered and deliberately excluded. Documented here so future team members don't re-propose them without understanding the rationale.

| Item | Why Descoped |
|------|-------------|
| Full RBAC with custom permission matrices | Overkill until 50+ users with genuinely different access needs. Row-level security (D1) covers the real requirement. |
| Real-time streaming dashboards (sub-second refresh) | Energy forecasting operates on hourly/daily cadences, not milliseconds. Unnecessary infrastructure cost. |
| Multi-tenant architecture in v1 | Design for it conceptually but don't build isolation layer until there's a second customer. |
| Blockchain-based audit trails | PostgreSQL audit table does the same job without the complexity. D2 covers audit needs. |
| Dark mode | Zero business value for an internal energy tool. Adds testing surface. |
| Gamification / achievement badges | Wrong context for energy operations. |
| AI chatbot overlay ("ask the dashboard") | Narrative mode (B1) + command palette (C5) accomplish 80% of the same goal at 5% of the engineering cost. |
| Custom color theme picker | Just ship a well-designed default. |
| Real-time collaboration (shared cursors, live editing) | Meeting-Ready Mode (C9) solves the actual collaboration problem (presenting together) without infrastructure overhead. |
| Natural language query interface | Same rationale as AI chatbot. High cost, low marginal value over narrative mode. |
| Mobile-native app | Responsive meeting-ready mode on tablet covers the real use case. A dedicated mobile app is a separate product, not a dashboard feature. |
| AI-generated recommended actions | Too much liability in energy operations context. Surface data, let humans decide. |

---

## Appendix: Quick Reference — Item Count by Category

| Category | Count | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|----------|-------|--------|--------|--------|--------|
| A: Design/UI | 6 | 3 | 1 | 2 | 0 |
| B: UX/Behavioral | 4 | 0 | 1 | 1 | 2 |
| C: Features | 9 | 1 | 3 | 3 | 2 |
| D: Security | 3 | 1 | 2 | 0 | 0 |
| E: Data Quality | 3 | 2 | 1 | 0 | 0 |
| F: Performance | 3 | 1 | 1 | 1 | 0 |
| G: Error Handling | 2 | 1 | 1 | 0 | 0 |
| H: Testing | 3 | 1 | 1 | 1 | 0 |
| I: Observability | 2 | 0 | 0 | 2 | 0 |
| J: Config | 2 | 1 | 0 | 0 | 1 |
| K: Integration | 2 | 0 | 1 | 1 | 0 |
| L: Compliance | 2 | 0 | 0 | 2 | 0 |
| M: Deployment | 2 | 0 | 0 | 0 | 2 |
| N: Docs/Change Mgmt | 3 | 0 | 0 | 0 | 3 |
| **TOTAL** | **46** | **10** | **11** | **13** | **12** |

**Descoped items: 12**
