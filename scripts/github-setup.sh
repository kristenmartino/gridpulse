#!/bin/bash
# =============================================================================
# NEXD — GitHub Repository Setup Script
# Creates the repo, sets branch protection, and pushes the initial scaffold.
#
# Prerequisites:
#   1. gh CLI installed: https://cli.github.com/
#   2. Authenticated: gh auth login
#
# Usage:
#   chmod +x scripts/github-setup.sh
#   ./scripts/github-setup.sh
# =============================================================================

set -euo pipefail

REPO_NAME="energy-forecast"
DESCRIPTION="Weather-aware energy demand forecasting dashboard — NextEra Analytics portfolio project"
VISIBILITY="public"  # Change to "private" if desired

echo "================================================"
echo "  NEXD — GitHub Repository Setup"
echo "================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Create GitHub Repository
# ---------------------------------------------------------------------------
echo "📦 Step 1: Creating GitHub repository..."
if gh repo view "${REPO_NAME}" &>/dev/null; then
    echo "   Repository already exists. Skipping creation."
else
    gh repo create "${REPO_NAME}" \
        --description "${DESCRIPTION}" \
        --"${VISIBILITY}" \
        --clone=false
    echo "   ✅ Repository created."
fi

# ---------------------------------------------------------------------------
# Step 2: Initialize Local Git Repo and Push
# ---------------------------------------------------------------------------
echo ""
echo "📁 Step 2: Initializing local repository..."
cd "$(dirname "$0")/.."

if [ -d .git ]; then
    echo "   Git already initialized. Skipping."
else
    git init
    git branch -M main
fi

# Get the remote URL
REMOTE_URL=$(gh repo view "${REPO_NAME}" --json url -q .url)
if ! git remote get-url origin &>/dev/null; then
    git remote add origin "${REMOTE_URL}.git"
fi

echo "   ✅ Remote set to: ${REMOTE_URL}"

# ---------------------------------------------------------------------------
# Step 3: Initial Commit
# ---------------------------------------------------------------------------
echo ""
echo "📝 Step 3: Creating initial commit..."
git add -A
git commit -m "chore: initial project scaffold

- Repo structure: data/, models/, simulation/, personas/, components/
- CI/CD: GitHub Actions (ci.yml, deploy-staging.yml, deploy-prod.yml)
- Infrastructure: Dockerfile, requirements.txt, GCP setup script
- Config: config.py with all 8 BA regions, constants, thresholds
- Docs: README.md, CLAUDE.md, PR template, CODEOWNERS
- Jira: CSV import file with 14 epics, 46 stories

Phase 0.2 complete. Ready for Sprint 1 development."

git push -u origin main
echo "   ✅ Initial commit pushed to main."

# ---------------------------------------------------------------------------
# Step 4: Branch Protection Rules
# ---------------------------------------------------------------------------
echo ""
echo "🔒 Step 4: Configuring branch protection..."

# Note: Branch protection via API requires the repo to have at least 1 commit
gh api repos/{owner}/{repo}/branches/main/protection \
    --method PUT \
    --field "required_pull_request_reviews[required_approving_review_count]=1" \
    --field "required_pull_request_reviews[dismiss_stale_reviews]=true" \
    --field "required_status_checks[strict]=true" \
    --field "required_status_checks[contexts][]=lint" \
    --field "required_status_checks[contexts][]=test" \
    --field "enforce_admins=false" \
    --field "restrictions=null" \
    --field "allow_force_pushes=false" \
    --field "allow_deletions=false" \
    2>/dev/null && echo "   ✅ Branch protection set." || echo "   ⚠️  Branch protection requires GitHub Pro/Team for private repos. Set manually in Settings → Branches."

# ---------------------------------------------------------------------------
# Step 5: Create Labels
# ---------------------------------------------------------------------------
echo ""
echo "🏷️  Step 5: Creating labels..."

declare -A LABELS=(
    ["spec-update"]="d4c5f9:Requires changes to buildplan or expanded spec"
    ["design-principle"]="0e8a16:Design standard to apply everywhere"
    ["regulatory"]="b60205:FERC/NERC/compliance implications"
    ["quick-win"]="00ff00:Completable in less than 1 day"
    ["cross-cutting"]="fbca04:Affects multiple tabs or components"
    ["spike-needed"]="d93f0b:Requires research before estimation"
    ["tier-1"]="b60205:Non-negotiable foundation (Sprint 1-3)"
    ["tier-2"]="ff9900:Core differentiation (Sprint 3-6)"
    ["tier-3"]="0075ca:Power user features (Sprint 6-9)"
    ["tier-4"]="cccccc:Polish and scale (Sprint 9+)"
)

for LABEL in "${!LABELS[@]}"; do
    IFS=':' read -r COLOR DESC <<< "${LABELS[$LABEL]}"
    gh label create "${LABEL}" --color "${COLOR}" --description "${DESC}" --force 2>/dev/null || true
done
echo "   ✅ Labels created."

# ---------------------------------------------------------------------------
# Step 6: Create Milestone for Sprint 1
# ---------------------------------------------------------------------------
echo ""
echo "🎯 Step 6: Creating Sprint 1 milestone..."
gh api repos/{owner}/{repo}/milestones \
    --method POST \
    --field "title=Sprint 1 — Data Pipeline & Core Models" \
    --field "description=EIA/Open-Meteo/NOAA clients, SQLite cache, feature engineering, model training pipeline" \
    --field "due_on=$(date -d '+14 days' +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -v+14d +%Y-%m-%dT%H:%M:%SZ)" \
    2>/dev/null && echo "   ✅ Sprint 1 milestone created." || echo "   ⚠️  Milestone creation failed (may already exist)."

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================"
echo "  ✅ GitHub Setup Complete!"
echo "================================================"
echo ""
echo "  Repository:   ${REMOTE_URL}"
echo "  Branch:       main (protected)"
echo "  CI Status:    .github/workflows/ci.yml"
echo "  Labels:       10 custom labels created"
echo ""
echo "  Next steps:"
echo "  1. Add GitHub Secrets (Settings → Secrets → Actions):"
echo "     - WIF_PROVIDER (from GCP setup)"
echo "     - GCP_SA_EMAIL (from GCP setup)"
echo "  2. Install 'GitHub for Jira' app in Jira"
echo "  3. Import scripts/jira-import.csv into Jira project NEXD"
echo "  4. Start Sprint 1: git checkout -b feat/data/eia-client"
echo ""
