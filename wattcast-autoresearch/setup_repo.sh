#!/bin/bash
# ─────────────────────────────────────────────
# setup_repo.sh — Initialize & push to GitHub
# Run this once after extracting the tar.gz
# ─────────────────────────────────────────────

set -e

REPO_NAME="wattcast-autoresearch"
GITHUB_USER="your-github-username"  # ← change this

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  ⚡ WattCast Autoresearch — Repo Setup       ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Step 1: Create repo on GitHub via CLI
echo "  Creating GitHub repository..."
gh repo create "$REPO_NAME" \
  --public \
  --description "Autonomous ML optimization for energy demand forecasting. Applies Karpathy's autoresearch pattern to XGBoost pipeline targeting sub-3% MAPE across 8 U.S. grid regions." \
  --clone=false

# Step 2: Initialize local git
echo "  Initializing local repo..."
git init
git add -A
git commit -m "initial commit: wattcast autoresearch scaffold

- train.py: XGBoost pipeline (the editable asset)
- prepare.py: synthetic data generator for 8 grid regions
- program.md: research directives for the agent
- journal.py: auto-generated research journal
- results/changelog.md: agent lab notebook template

Baseline target: 3.13% MAPE
Stack: XGBoost / pandas / numpy
Pattern: Karpathy autoresearch loop"

# Step 3: Push
echo "  Pushing to GitHub..."
git branch -M main
git remote add origin "https://github.com/$GITHUB_USER/$REPO_NAME.git"
git push -u origin main

echo ""
echo "  ✅ Repository live at: https://github.com/$GITHUB_USER/$REPO_NAME"
echo ""
echo "  Next steps:"
echo "    1. python prepare.py          # generate training data"
echo "    2. python train.py            # verify baseline"
echo "    3. claude 'Read program.md and start running experiments'"
echo ""
