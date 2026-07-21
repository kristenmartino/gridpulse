# GridPulse Landing Page Wireframe + Homepage Copy Deck — _archived_
_Spec authored 2026-04-09. Archived to `docs/internal/` 2026-05-06._

## Status

**Partially implemented 2026-07-21** — the portfolio-neutral subset of this
spec now ships at [`/about`](https://gridpulse.kristenmartino.ai/about)
(`web/landing.html`, served by `landing.py`; the dashboard keeps `/`). The
reopen condition (commercial positioning, per the market-entry plan) fired.
The postmortem's exclusions below **held**: no demo/contact CTAs, no
social-proof strip, no "Solutions" nav, no suite-vision copy — and they are
now enforced as posture-pin tests in `tests/unit/test_landing.py`. Flipping
any of them post-BSC-process is a deliberate edit there.

_Original archive note (2026-05-06):_ this wireframe specs an
enterprise-B2B-SaaS marketing landing page (hero / social proof / value
pillars / module grid / "Request a demo" CTA) that doesn't fit a
portfolio-stage project. Audience for the GridPulse repo is overwhelmingly
recruiting / technical evaluation — those reviewers evaluate the live
dashboard + the GitHub repo + the test suite, not a separate marketing page.

Substantive copy from this spec that DOES belong on portfolio surfaces was migrated:

- Headline candidates → `README.md`'s H1 + tagline
- Value pillars (forecast with context · see confidence not just output · surface risk earlier · plan through scenarios) → `README.md` "What GridPulse Does"
- Persona role framing → already live in the dashboard's persona switcher
- Module names (Forecast / Risk / Grid / Scenarios / Models / Briefings / API) → `README.md` "Roadmap direction"
- Tech credibility bullets → `README.md` "Models" + ADRs in `PRD.md`

What's intentionally NOT migrated:
- "Request a demo" / "Schedule a call" CTAs — there's nothing to demo beyond the live URL
- "Social proof / credibility strip" with logos — would be stock-image theater without real customers
- "Solutions" nav — implies tailored verticals that don't exist
- "Suite expansion / platform vision" copy — overreach for a single-product portfolio piece

## When to reopen

Revisit if the project is positioning for an actual commercial deployment, energy-sector PM role applications where a marketing artifact is load-bearing, or a real customer signs up. Until then the README + deployed URL combination is doing the job a marketing page would attempt.

---

_The original wireframe content is preserved below for reference._

---

## Purpose

This document translates the GridPulse brand strategy into a landing page structure that can support:
- investor or portfolio presentation
- customer-facing product marketing
- recruiting / credibility signaling
- future product-suite expansion

This page should position GridPulse as an **energy intelligence platform**, not just a forecasting dashboard.

---

# 1. Homepage goal

The homepage should communicate five things within the first screen and one scroll:

1. GridPulse is for the energy sector
2. It helps teams forecast and interpret grid conditions
3. It combines visibility, confidence, and decision support
4. It feels technically credible and enterprise-grade
5. It can grow into a broader platform / product suite

---

# 2. Page strategy

## Primary audience
- energy operators
- analytics leaders
- planning and trading stakeholders
- technical hiring managers or reviewers
- enterprise buyers
- strategic partners

## Secondary audience
- investors
- broader technical audiences
- product/design reviewers
- portfolio evaluators

## Conversion goals
Choose one primary CTA depending on the site context:

### Option A — commercial
- **Request a demo**

### Option B — portfolio / showcase
- **Explore the platform**

### Option C — technical credibility
- **View the product walkthrough**

Recommended CTA pair:
- Primary: **Request a demo**
- Secondary: **Explore the platform**

---

# 3. Recommended sitemap

## Top navigation
- Platform
- Modules
- Solutions
- Resources
- Company
- Request Demo

## Footer navigation
- Platform Overview
- Forecast
- Risk
- Grid
- Scenarios
- Models
- Briefings
- API
- Documentation
- Contact

---

# 4. Wireframe overview

## Section order
1. Hero
2. Social proof / credibility strip
3. Value pillars
4. Product modules
5. Built for real energy workflows
6. Platform screenshot / UI narrative
7. Technical credibility
8. Suite expansion / platform vision
9. CTA close
10. Footer

---

# 5. Hero section

## Wireframe structure

### Left column
- eyebrow
- headline
- subhead
- CTA buttons
- small proof points

### Right column
- premium product mockup
- floating annotations
- confidence band / risk markers
- region selector / module label

## Copy

### Eyebrow
**Energy Intelligence Platform**

### Headline option A
**Operational clarity for a volatile grid.**

### Headline option B
**See demand sooner. Decide with confidence.**

### Recommended headline
**Operational clarity for a volatile grid.**

### Subhead
**GridPulse unifies demand forecasting, forecast confidence, grid visibility, and scenario-based decision support in one energy intelligence platform.**

### Primary CTA
**Request a demo**

### Secondary CTA
**Explore the platform**

### Proof chips under CTA
- Weather-aware forecasting
- Confidence-aware modeling
- Regional grid visibility
- Scenario-ready workflows

## Visual direction
Show a refined screen that includes:
- dark premium product surface
- selected region
- forecast curve
- confidence band
- a few high-signal KPIs
- one risk callout
- one short narrative briefing card

---

# 6. Social proof / credibility strip

This section should be restrained. Since this may be a portfolio-stage or early product, do not fake customer logos.

## If pre-commercial
Use credibility statements instead:
- Built on real grid and weather data
- Cloud-native deployment architecture
- Multi-model forecasting approach
- Designed for operational decision workflows

## Copy
**Built for real energy workflows, not generic dashboard demos.**

Optional proof row:
- Real grid data
- Weather-informed signals
- Historical backtesting
- Regional views
- Cloud-native architecture

---

# 7. Value pillars section

## Layout
3 or 4 cards in a grid

## Section headline
**A clearer operating layer for energy decisions**

## Supporting copy
GridPulse brings fragmented operational signals into one decision-ready system so teams can move faster without losing context or confidence.

## Pillar cards

### Card 1
**Forecast with context**  
Unify demand, weather, and time-series patterns in a forecasting experience designed for real operating conditions.

### Card 2
**See confidence, not just output**  
Understand model reliability, forecast uncertainty, and recent performance before acting on a number.

### Card 3
**Surface risk earlier**  
Track anomalies, severe conditions, and forecast instability in one operational view.

### Card 4
**Plan through scenarios**  
Test assumptions and understand how changing conditions can alter expected demand and decision windows.

---

# 8. Product modules section

## Section headline
**One platform. Multiple decision layers.**

## Supporting copy
Start with forecasting, then expand into risk, grid visibility, scenarios, and model oversight without fragmenting the workflow.

## Module grid

### GridPulse Forecast
Weather-aware demand forecasts with confidence bands, horizon views, and model-aware comparisons.

### GridPulse Risk
Operational risk visibility for anomalies, severe conditions, and forecast instability.

### GridPulse Grid
Generation mix, net load, and regional operating context in one view.

### GridPulse Scenarios
What-if planning tools for testing conditions before they become costly.

### GridPulse Models
Backtesting, model validation, drift awareness, and forecast accountability.

### GridPulse Briefings
Narrative summaries, executive snapshots, and presentation-ready operational updates.

### Optional module
**GridPulse API**  
Bring GridPulse signals into external systems and enterprise workflows.

---

# 9. Built for real energy workflows section

## Section headline
**Designed for the people who actually use the signal**

## Layout
Four role cards or tabs

## Role cards

### Utility Operations
See demand direction, risk windows, and high-priority conditions quickly.

### Renewables Planning
Understand weather-driven shifts and how they affect expected grid behavior.

### Trading & Market Analysis
Track demand movement, changing conditions, and forecast divergence with greater speed.

### Analytics & Model Teams
Monitor model performance, confidence degradation, and validation results in context.

---

# 10. Platform screenshot narrative section

## Section headline
**A system that moves from signal to decision**

## Layout
Large product screenshot on one side, explanation stack on the other

## Callout copy
- Start with a region or operating view
- Review today’s demand outlook and confidence
- Inspect rising risks and changing conditions
- Compare scenarios before action
- Share a briefing-ready summary with stakeholders

## Screenshot annotations
- Region selector
- Forecast confidence band
- Risk marker
- Scenario panel
- Model confidence panel
- Briefing mode / saved snapshot

---

# 11. Technical credibility section

## Section headline
**Built for technical scrutiny**

## Supporting copy
GridPulse is designed to be credible with both operators and technical stakeholders, combining real external signals, transparent model views, and cloud-ready deployment patterns.

## Credibility bullets
- Weather-informed forecasting inputs
- Historical backtesting and validation views
- Confidence bands and model comparisons
- Regional data workflows
- Cloud-native deployment model
- Structured architecture for scaling into a broader platform

## Optional subheading
**Not a black box. Not a static dashboard.**

---

# 12. Platform vision / suite expansion section

## Section headline
**From forecasting tool to energy intelligence platform**

## Supporting copy
GridPulse is structured to expand modularly, allowing teams to begin with the highest-value forecasting layer and grow into broader operational intelligence over time.

## Visual
A hub-and-spoke or modular system diagram:
- Forecast
- Risk
- Grid
- Scenarios
- Models
- Briefings
- API

---

# 13. Closing CTA

## Headline
**Bring forecasting, confidence, and grid visibility into one view.**

## Supporting copy
GridPulse helps energy teams interpret changing conditions faster and act with more confidence.

## CTA buttons
- Request a demo
- Explore the platform

---

# 14. Footer copy

## Short footer blurb
**GridPulse is an energy intelligence platform for forecast confidence, grid visibility, and operational decision support.**

---

# 15. Suggested homepage copy, assembled

## Hero assembled copy
**Energy Intelligence Platform**  
# **Operational clarity for a volatile grid.**  
GridPulse unifies demand forecasting, forecast confidence, grid visibility, and scenario-based decision support in one energy intelligence platform.

[Request a demo] [Explore the platform]

Weather-aware forecasting · Confidence-aware modeling · Regional grid visibility · Scenario-ready workflows

---

## Value section assembled copy
# **A clearer operating layer for energy decisions**  
GridPulse brings fragmented operational signals into one decision-ready system so teams can move faster without losing context or confidence.

**Forecast with context**  
Unify demand, weather, and time-series patterns in a forecasting experience designed for real operating conditions.

**See confidence, not just output**  
Understand model reliability, forecast uncertainty, and recent performance before acting on a number.

**Surface risk earlier**  
Track anomalies, severe conditions, and forecast instability in one operational view.

**Plan through scenarios**  
Test assumptions and understand how changing conditions can alter expected demand and decision windows.

---

## Modules section assembled copy
# **One platform. Multiple decision layers.**

**GridPulse Forecast**  
Weather-aware demand forecasts with confidence bands, horizon views, and model-aware comparisons.

**GridPulse Risk**  
Operational risk visibility for anomalies, severe conditions, and forecast instability.

**GridPulse Grid**  
Generation mix, net load, and regional operating context in one view.

**GridPulse Scenarios**  
What-if planning tools for testing conditions before they become costly.

**GridPulse Models**  
Backtesting, model validation, drift awareness, and forecast accountability.

**GridPulse Briefings**  
Narrative summaries, executive snapshots, and presentation-ready operational updates.

---

# 16. Visual design notes for the landing page

## Page style
- dark hero
- mixed light/dark sections for pacing
- strong typography hierarchy
- restrained gradients
- clear data screenshots
- minimal decorative illustration

## Recommended palette use
- Hero background: deep obsidian / midnight gradient
- CTA accent: pulse cyan
- Secondary accent: grid blue / teal
- Alert examples: amber / coral
- Light sections: very pale blue-gray backgrounds for contrast

## Typography
- Display: Sora
- Body/UI: Inter

---

# 17. Asset list for implementation

To implement the homepage properly, create:
1. hero product screenshot
2. 6 module icons
3. light and dark logo lockups
4. one suite architecture diagram
5. three product screenshot crops
6. CTA button styles
7. landing page section illustrations or abstract topology textures

---

# 18. Variants by use case

## A. Portfolio version
Primary CTA:
- Explore the platform

Sections to emphasize:
- product strategy
- technical architecture
- module thinking
- design system maturity

## B. Commercial version
Primary CTA:
- Request a demo

Sections to emphasize:
- business outcomes
- workflows
- trust and deployment readiness
- modular platform story

## C. Recruiting / case study version
Primary CTA:
- View the walkthrough

Sections to emphasize:
- design rationale
- analytics rigor
- role-based UX
- product architecture

---

# 19. Recommended next deliverables after landing page

1. final homepage Figma wireframe
2. product screenshot art direction brief
3. module page templates
4. logo explorations
5. web design tokens
6. copy deck for each module page
