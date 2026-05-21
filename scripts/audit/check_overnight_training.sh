#!/usr/bin/env bash
# Check the state of the most recent Cloud Run training-job execution.
#
# Verifies the production effect of the 2026-05-20 forecast-pipeline audit:
#
# 1. Lists recent training-job executions (oldest scheduled-cron 04:00 UTC).
# 2. Tails the most recent execution's logs.
# 3. Greps for the post-PR-D health indicators:
#    - ``training_complete`` events per region with finite MAPE values
#    - feature-importance summary lines (``demand_roll_24h_min`` should
#      NOT appear in top-5 after PR-D's de-leak fix)
#    - ``training_failed`` events flagging a region's regression
# 4. Spot-checks live drift records for the top demand regions if a
#    REDIS_HOST env var is provided (the web tier reads
#    ``gridpulse:drift:{region}.models.ensemble.rolling_mape_7d`` —
#    same value the Overview tab's MAPE clause renders).
#
# Prerequisites:
# - ``gcloud`` authenticated against the ``nextera-portfolio`` project
#   (run ``gcloud auth login`` and ``gcloud config set project
#   nextera-portfolio`` once)
# - For the Redis spot-check: a machine with VPC access to the
#   Memorystore instance, plus REDIS_HOST exported.
#
# Usage::
#
#     bash scripts/audit/check_overnight_training.sh
#     REDIS_HOST=10.x.x.x bash scripts/audit/check_overnight_training.sh
#
# Output: human-readable summary printed to stdout. No persistence,
# no Redis writes, no GCS writes — read-only.

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-nextera-portfolio}"
REGION="${REGION:-us-east1}"
JOB_NAME="${JOB_NAME:-gridpulse-training-job}"
FRESHNESS="${FRESHNESS:-8h}"  # how far back to scan for logs

printf "\n%s\n" "================================================================"
printf "  Overnight training cycle check\n"
printf "%s\n" "================================================================"
printf "  Project:   %s\n" "$PROJECT_ID"
printf "  Region:    %s\n" "$REGION"
printf "  Job:       %s\n" "$JOB_NAME"
printf "  Log range: last %s\n" "$FRESHNESS"
printf "\n"

# ── 1. Recent executions ─────────────────────────────────────────
printf "── Recent executions ───────────────────────────────────────────\n"
gcloud run jobs executions list \
  --job="$JOB_NAME" \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --limit=3 \
  --format='table(metadata.name,status.startTime,status.completionTime,status.succeededCount,status.failedCount)' \
  2>&1 || {
    printf "  ✗ Failed to list executions. Check gcloud auth.\n" >&2
    exit 1
  }

# Most recent execution
LATEST_EXEC=$(gcloud run jobs executions list \
  --job="$JOB_NAME" \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --limit=1 \
  --format='value(metadata.name)' 2>/dev/null || true)

if [ -z "$LATEST_EXEC" ]; then
  printf "\n  ✗ No executions found in the last %s. Check Cloud Scheduler.\n" "$FRESHNESS"
  exit 1
fi

printf "\n  Most recent: %s\n\n" "$LATEST_EXEC"

# ── 2. Per-region training_complete events ───────────────────────
printf "── training_complete events ──────────────────────────────────\n"
gcloud logging read \
  "resource.type=cloud_run_job AND resource.labels.job_name=$JOB_NAME AND jsonPayload.event=training_complete" \
  --project="$PROJECT_ID" \
  --freshness="$FRESHNESS" \
  --limit=200 \
  --format='value(jsonPayload.region, jsonPayload.ensemble_mape, jsonPayload.xgboost_mape)' \
  2>/dev/null \
  | awk 'BEGIN { count=0 } { print "  " $0; count++ } END { print ""; print "  (" count " regions reported training_complete)" }'

# ── 3. Feature-importance check — the PR-D verification ──────────
printf "\n── Top XGBoost features (post-PR-D check) ────────────────────\n"
printf "  Looking for ``demand_roll_24h_min`` — should NOT appear in top-5\n\n"
gcloud logging read \
  "resource.type=cloud_run_job AND resource.labels.job_name=$JOB_NAME AND jsonPayload.event=xgboost_trained" \
  --project="$PROJECT_ID" \
  --freshness="$FRESHNESS" \
  --limit=20 \
  --format='value(jsonPayload.region, jsonPayload.top_features)' \
  2>/dev/null \
  | head -20 \
  | awk '/demand_roll_24h_min/ { print "  ⚠ " $0 " ← leaky feature still in top-5"; flagged++; next } { print "  ✓ " $0 } END { if (flagged) print ""; print "  ("flagged "+0" " regions still showing demand_roll_24h_min)" }'

# ── 4. Training failures (any region) ────────────────────────────
printf "\n── training_failed events ────────────────────────────────────\n"
FAILED=$(gcloud logging read \
  "resource.type=cloud_run_job AND resource.labels.job_name=$JOB_NAME AND severity=ERROR" \
  --project="$PROJECT_ID" \
  --freshness="$FRESHNESS" \
  --limit=50 \
  --format='value(jsonPayload.region, jsonPayload.error, textPayload)' \
  2>/dev/null \
  | head -50)
if [ -z "$FAILED" ]; then
  printf "  ✓ No ERROR-severity events in the last %s.\n" "$FRESHNESS"
else
  printf "%s\n" "$FAILED" | awk '{ print "  ✗ " $0 }'
fi

# ── 5. Optional: spot-check live drift MAPE via Redis ────────────
if [ -n "${REDIS_HOST:-}" ]; then
  printf "\n── Live drift MAPE (rolling 7d, top regions) ──────────────────\n"
  printf "  Redis: %s\n\n" "$REDIS_HOST"
  for REGION_CODE in ERCOT FPL PJM CAISO MISO; do
    KEY="gridpulse:drift:$REGION_CODE"
    PAYLOAD=$(redis-cli -h "$REDIS_HOST" --no-raw get "$KEY" 2>/dev/null || true)
    if [ -z "$PAYLOAD" ] || [ "$PAYLOAD" = "(nil)" ]; then
      printf "  %-7s — no drift records yet\n" "$REGION_CODE"
      continue
    fi
    # Strip leading/trailing quote that redis-cli --no-raw adds for JSON.
    PAYLOAD=$(printf "%s" "$PAYLOAD" | sed 's/^"//;s/"$//;s/\\"/"/g')
    MAPE_7D=$(printf "%s" "$PAYLOAD" | python3 -c "import sys,json; p=json.load(sys.stdin); print(p['models']['ensemble']['rolling_mape_7d'])" 2>/dev/null || echo "?")
    N_REC=$(printf "%s" "$PAYLOAD" | python3 -c "import sys,json; p=json.load(sys.stdin); print(p['models']['ensemble']['n_records'])" 2>/dev/null || echo "?")
    printf "  %-7s 7d MAPE = %s%%   (n=%s records)\n" "$REGION_CODE" "$MAPE_7D" "$N_REC"
  done
else
  printf "\n── Live drift spot-check skipped ──────────────────────────────\n"
  printf "  (Set REDIS_HOST env var to enable. Needs VPC access.)\n"
fi

printf "\n%s\n" "================================================================"
printf "  Done. Compare top features against yesterday's local check:\n"
printf "  Expected top-5: demand_lag_24h, demand_lag_1h, demand_ratio_24h,\n"
printf "                  demand_ratio_168h, demand_lag_168h\n"
printf "  (PR-D's de-leak removed demand_roll_24h_min from top features.)\n"
printf "%s\n" "================================================================"
