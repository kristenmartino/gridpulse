# Jira Project Setup Guide — NEXD

## Step 1: Create the Jira Project

1. Go to your Jira instance → **Projects** → **Create project**
2. Select **Scrum** template
3. Configure:
   - **Name:** NextEra Energy Dashboard
   - **Key:** NEXD
   - **Lead:** (your name)
4. Click **Create**

## Step 2: Configure Custom Fields

Go to **Project Settings** → **Fields** and add:

| Field Name | Type | Values |
|-----------|------|--------|
| Priority Tier | Single Select | 1, 2, 3, 4 |
| Category | Single Select | A, B, C, D, E, F, G, H, I, J, K, L, M, N |
| Spec Section | Short Text | Free text |
| Value Score | Number | 1-4 |
| Complexity Score | Number | 1-3 |

## Step 3: Import the Backlog (CSV)

1. Go to **Project Settings** → **CSV Import** (or use the Jira CSV importer at `/secure/admin/ExternalImport1.jspa`)
2. Upload `scripts/jira-import.csv`
3. Map columns:
   - `Summary` → Summary
   - `Issue Type` → Issue Type
   - `Epic Name` → Epic Name (for Epic rows)
   - `Epic Link` → Epic Link (for Story rows)
   - `Priority` → Priority
   - `Labels` → Labels
   - `Description` → Description
   - `Story Points` → Story Points
   - `Sprint` → Sprint
   - `Custom field (Priority Tier)` → Priority Tier
   - `Custom field (Category)` → Category
4. Run import — should create **14 Epics** and **46 Stories**

## Step 4: Configure the Board

1. Go to **Board** → **Board Settings** → **Columns**
2. Set up columns:

```
Backlog → Refinement → Ready → In Progress → In Review → QA → Done
```

3. Configure WIP limits:
   - In Progress: 3
   - In Review: 2
   - QA: 2

## Step 5: Create Sprint 1

1. Go to **Backlog** view
2. Click **Create Sprint** → Name: "Sprint 1 — Data Pipeline & Core Models"
3. Set dates: 2 weeks from today
4. Drag these stories into Sprint 1:
   - All stories with `Sprint` = "Sprint 3" or "Sprint 4" in the CSV (these are Tier 1 items)
   - NOTE: Sprint 3/4 in the CSV refers to the roadmap tier, not the Jira sprint. For Sprint 1, focus on the infrastructure stories that other stories depend on.

**Recommended Sprint 1 stories (build-order critical):**

| Story | Why First |
|-------|-----------|
| Create data/eia_client.py + cache.py | Everything depends on data ingestion |
| Create data/weather_client.py | Features depend on weather data |
| Create data/noaa_client.py | Alerts tab depends on this |
| Create data/feature_engineering.py | Models depend on features |
| Create data/preprocessing.py | Data alignment is a prerequisite |
| Create models/ (all 4 + ensemble) | Tabs 1-3 depend on model output |
| Create config.py (already done) | Reference for all constants |
| Write tests for data/ and models/ | Coverage gates in CI |

These aren't in the backlog CSV (which tracks enhancement features) — they're the **core build tasks** from the expanded spec. Create them as Tasks under a new Epic: "NEXD-EPIC-0: Core Build."

## Step 6: Create Automation Rules

Go to **Project Settings** → **Automation** and create:

1. **Auto-assign reviewer on "In Review":**
   - Trigger: Issue transitioned to "In Review"
   - Action: Assign to tech lead

2. **Auto-transition on PR merge:**
   - Trigger: Pull request merged (requires GitHub for Jira app)
   - Condition: Issue status is "In Review"
   - Action: Transition to "QA"

3. **Stale issue reminder:**
   - Trigger: Issue in "In Progress" for > 3 days
   - Action: Add comment "⏰ This issue has been In Progress for 3+ days. Need help?"

## Step 7: Install GitHub for Jira

1. Go to **Jira Settings** → **Apps** → **Find new apps**
2. Search for "GitHub for Jira"
3. Install and connect to your GitHub organization
4. Authorize the `energy-forecast` repository
5. Now commits with `NEXD-XX` will auto-link to Jira issues

## Step 8: Create Labels

In Jira, create these labels (some are included in the CSV import):

```
spec-update        — Requires changes to buildplan or expanded spec
design-principle   — Design standard to apply everywhere (e.g., A6)
regulatory         — FERC/NERC/compliance implications
quick-win          — Completable in < 1 day
cross-cutting      — Affects multiple tabs or components
spike-needed       — Requires research before estimation
```

## Step 9: Sprint Cadence

| Ceremony | When | Duration |
|----------|------|----------|
| Sprint Planning | Monday AM, Day 1 | 1.5 hours |
| Daily Standup | Every morning | 15 min |
| Backlog Refinement | Wednesday, Day 8 | 1 hour |
| Sprint Review/Demo | Friday PM, Day 10 | 1 hour |
| Retrospective | After Sprint Review | 45 min |

## Verification Checklist

After setup, verify:

- [ ] 14 Epics visible in backlog (NEXD-EPIC-A through NEXD-EPIC-N)
- [ ] 46 Stories imported with correct Epic links
- [ ] Custom fields (Priority Tier, Category) populated
- [ ] Board columns match workflow
- [ ] Sprint 1 created with correct date range
- [ ] GitHub integration showing commits/PRs
- [ ] Automation rules active
