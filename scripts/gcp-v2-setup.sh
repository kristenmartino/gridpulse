#!/bin/bash
# =============================================================================
# WattCast v2 — GCP Infrastructure Setup (Memorystore + Cloud SQL + VPC)
#
# Provisions the managed services for the pre-computation pipeline.
# Run AFTER gcp-setup.sh has set up the base project.
#
# Prerequisites:
#   1. gcp-setup.sh has been run (project, SA, Artifact Registry exist)
#   2. gcloud CLI authenticated with project set
#
# Usage:
#   chmod +x scripts/gcp-v2-setup.sh
#   ./scripts/gcp-v2-setup.sh
# =============================================================================

set -euo pipefail

PROJECT_ID="nextera-portfolio"
REGION="us-east1"
SA_EMAIL="energy-forecast-sa@${PROJECT_ID}.iam.gserviceaccount.com"

echo "================================================"
echo "  WattCast v2 — Pipeline Infrastructure Setup"
echo "================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Enable additional APIs
# ---------------------------------------------------------------------------
echo "Step 1: Enabling v2 APIs..."
gcloud services enable \
    redis.googleapis.com \
    sqladmin.googleapis.com \
    vpcaccess.googleapis.com \
    composer.googleapis.com
echo "  Done."

# ---------------------------------------------------------------------------
# Step 2: Create VPC Connector (Cloud Run <-> Memorystore/Cloud SQL)
# ---------------------------------------------------------------------------
echo ""
echo "Step 2: Creating Serverless VPC Connector..."
if gcloud compute networks vpc-access connectors describe wattcast-connector \
    --region="${REGION}" &>/dev/null; then
    echo "  VPC connector already exists. Skipping."
else
    gcloud compute networks vpc-access connectors create wattcast-connector \
        --region="${REGION}" \
        --range="10.8.0.0/28" \
        --min-instances=2 \
        --max-instances=3
    echo "  Done."
fi

# ---------------------------------------------------------------------------
# Step 3: Create Memorystore (Redis) instance
# ---------------------------------------------------------------------------
echo ""
echo "Step 3: Creating Memorystore (Redis) instance..."
if gcloud redis instances describe wattcast-cache --region="${REGION}" &>/dev/null; then
    echo "  Redis instance already exists. Skipping."
else
    gcloud redis instances create wattcast-cache \
        --size=1 \
        --region="${REGION}" \
        --tier=basic \
        --redis-version=redis_7_0
    echo "  Done."
fi

REDIS_HOST=$(gcloud redis instances describe wattcast-cache \
    --region="${REGION}" --format="value(host)")
REDIS_PORT=$(gcloud redis instances describe wattcast-cache \
    --region="${REGION}" --format="value(port)")
echo "  Redis endpoint: ${REDIS_HOST}:${REDIS_PORT}"

# ---------------------------------------------------------------------------
# Step 4: Create Cloud SQL (Postgres) instance
# ---------------------------------------------------------------------------
echo ""
echo "Step 4: Creating Cloud SQL (Postgres) instance..."
if gcloud sql instances describe wattcast-db &>/dev/null; then
    echo "  Cloud SQL instance already exists. Skipping."
else
    gcloud sql instances create wattcast-db \
        --database-version=POSTGRES_15 \
        --tier=db-f1-micro \
        --region="${REGION}" \
        --storage-size=10GB \
        --storage-auto-increase \
        --no-assign-ip \
        --network=default
    echo "  Done."
fi

# Create database and user
gcloud sql databases create wattcast --instance=wattcast-db 2>/dev/null || echo "  Database 'wattcast' already exists."
DB_PASSWORD=$(openssl rand -base64 24)
gcloud sql users create wattcast --instance=wattcast-db --password="${DB_PASSWORD}" 2>/dev/null || echo "  User 'wattcast' already exists."

SQL_IP=$(gcloud sql instances describe wattcast-db --format="value(ipAddresses[0].ipAddress)" 2>/dev/null || echo "private-ip")
echo "  Cloud SQL: ${SQL_IP}"

# Apply init.sql schema
echo "  Applying database schema..."
if [ -f "scaling-analytics/db/init.sql" ]; then
    gcloud sql connect wattcast-db --user=wattcast --database=wattcast < scaling-analytics/db/init.sql 2>/dev/null || \
        echo "  Schema already applied or manual apply needed."
fi

# ---------------------------------------------------------------------------
# Step 5: Create GCS bucket for model artifacts
# ---------------------------------------------------------------------------
echo ""
echo "Step 5: Creating model artifacts bucket..."
MODELS_BUCKET="${PROJECT_ID}-wattcast-models"
if gsutil ls -b "gs://${MODELS_BUCKET}" &>/dev/null; then
    echo "  Bucket already exists. Skipping."
else
    gsutil mb -p "${PROJECT_ID}" -l "${REGION}" -b on "gs://${MODELS_BUCKET}"
    echo "  Done: gs://${MODELS_BUCKET}"
fi

# ---------------------------------------------------------------------------
# Step 6: Store secrets
# ---------------------------------------------------------------------------
echo ""
echo "Step 6: Storing v2 secrets..."

# Redis host
if gcloud secrets describe redis-host &>/dev/null; then
    echo "  Secret 'redis-host' already exists."
else
    echo -n "${REDIS_HOST}" | gcloud secrets create redis-host --data-file=-
    echo "  Done."
fi

# Database URL
if gcloud secrets describe database-url &>/dev/null; then
    echo "  Secret 'database-url' already exists."
else
    echo -n "postgresql://wattcast:${DB_PASSWORD}@${SQL_IP}:5432/wattcast" | \
        gcloud secrets create database-url --data-file=-
    echo "  Done."
fi

# Grant SA access to new secrets
for SECRET in redis-host database-url; do
    gcloud secrets add-iam-policy-binding "${SECRET}" \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="roles/secretmanager.secretAccessor" \
        --quiet
done
echo "  SA granted access to v2 secrets."

# ---------------------------------------------------------------------------
# Step 7: Grant SA additional roles for v2
# ---------------------------------------------------------------------------
echo ""
echo "Step 7: Granting additional IAM roles..."
for ROLE in roles/redis.editor roles/cloudsql.client roles/composer.worker; do
    gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="${ROLE}" \
        --quiet
done
echo "  Done."

# ---------------------------------------------------------------------------
# Step 8: Update Cloud Run with Redis
# ---------------------------------------------------------------------------
echo ""
echo "Step 8: Updating Cloud Run service with Redis..."
gcloud run services update gridpulse \
    --region="${REGION}" \
    --vpc-connector=wattcast-connector \
    --set-env-vars "REDIS_HOST=${REDIS_HOST},REDIS_PORT=${REDIS_PORT}" \
    --quiet 2>/dev/null || echo "  Cloud Run service not deployed yet. Will pick up env vars on next deploy."

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================"
echo "  v2 Infrastructure Setup Complete"
echo "================================================"
echo ""
echo "  Memorystore (Redis): ${REDIS_HOST}:${REDIS_PORT}"
echo "  Cloud SQL:           wattcast-db (${SQL_IP})"
echo "  VPC Connector:       wattcast-connector"
echo "  Model Artifacts:     gs://${MODELS_BUCKET}"
echo ""
echo "  Add to GitHub secrets:"
echo "    REDIS_HOST = ${REDIS_HOST}"
echo ""
echo "  Next steps:"
echo "    1. Set up Cloud Composer and upload DAGs from scaling-analytics/dags/"
echo "    2. Push to main branch to trigger deployment with Redis"
echo "    3. Trigger the training DAG to populate Memorystore"
echo ""
