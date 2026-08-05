#!/bin/bash
# =============================================================================
# NEXD — GCP Project Setup Script
# Run this ONCE to provision all Google Cloud infrastructure.
#
# Prerequisites:
#   1. gcloud CLI installed: https://cloud.google.com/sdk/docs/install
#   2. Authenticated: gcloud auth login
#   3. Billing account linked (script will prompt)
#   4. EIA API key ready: https://www.eia.gov/opendata/register.php
#
# Usage:
#   chmod +x scripts/gcp-setup.sh
#   ./scripts/gcp-setup.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ID="nextera-portfolio"
PROJECT_NAME="NextEra Portfolio"
REGION="us-east1"
REPO_NAME="portfolio"
SERVICE_NAME="energy-forecast"
SERVICE_ACCOUNT_NAME="energy-forecast-sa"

echo "================================================"
echo "  NEXD — GCP Infrastructure Setup"
echo "================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Create GCP Project
# ---------------------------------------------------------------------------
echo "📦 Step 1: Creating GCP project '${PROJECT_ID}'..."
if gcloud projects describe "${PROJECT_ID}" &>/dev/null; then
    echo "   Project already exists. Skipping creation."
else
    gcloud projects create "${PROJECT_ID}" --name="${PROJECT_NAME}"
    echo "   ✅ Project created."
fi
gcloud config set project "${PROJECT_ID}"

# ---------------------------------------------------------------------------
# Step 2: Link Billing
# ---------------------------------------------------------------------------
echo ""
echo "💳 Step 2: Linking billing account..."
BILLING_ACCOUNT=$(gcloud billing accounts list --format="value(ACCOUNT_ID)" | head -1)
if [ -z "${BILLING_ACCOUNT}" ]; then
    echo "   ❌ No billing account found. Set up billing at:"
    echo "      https://console.cloud.google.com/billing"
    exit 1
fi
echo "   Using billing account: ${BILLING_ACCOUNT}"
gcloud billing projects link "${PROJECT_ID}" --billing-account="${BILLING_ACCOUNT}"
echo "   ✅ Billing linked."

# ---------------------------------------------------------------------------
# Step 3: Enable APIs
# ---------------------------------------------------------------------------
echo ""
echo "🔌 Step 3: Enabling required APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    secretmanager.googleapis.com \
    storage.googleapis.com \
    iam.googleapis.com \
    iamcredentials.googleapis.com
echo "   ✅ APIs enabled."

# ---------------------------------------------------------------------------
# Step 4: Create Artifact Registry Repository
# ---------------------------------------------------------------------------
echo ""
echo "📦 Step 4: Creating Artifact Registry repository..."
if gcloud artifacts repositories describe "${REPO_NAME}" --location="${REGION}" &>/dev/null; then
    echo "   Repository already exists. Skipping."
else
    gcloud artifacts repositories create "${REPO_NAME}" \
        --repository-format=docker \
        --location="${REGION}" \
        --description="NextEra portfolio Docker images"
    echo "   ✅ Repository created."
fi

# Cleanup policy (2026-08-05). Applied unconditionally, including to an
# existing repo — this is the fix for a repo that had NO retention config at
# all and had grown to 319 GB across 398 gridpulse images since 2026-02-28,
# ~$32/mo at $0.10/GB-mo and rising ~$2.50/mo each month on its own. Every
# green merge to main and to dev pushes two tags of an ~800 MB
# Prophet/cmdstan/XGBoost image and nothing ever deleted one.
#
# Keep rules take precedence over Delete rules in Artifact Registry, so
# keep-recent wins for the newest 20 versions regardless of the 30d delete.
# 20 is chosen to cover realistic rollback depth, NOT to be minimal: the
# gridpulse Cloud Run service has ~389 revisions and each pins an image
# digest, so deleting an image makes rollback *to that revision* impossible.
# Recent revisions are the only ones anyone would roll back to.
#
# ALWAYS dry-run before the first real apply on a populated repo:
#   gcloud artifacts repositories set-cleanup-policies "${REPO_NAME}" \
#       --location="${REGION}" --policy=<file> --dry-run
#   gcloud artifacts repositories describe "${REPO_NAME}" \
#       --location="${REGION}"   # then read cleanupPolicyDryRun
echo "   Applying cleanup policy..."
cat > /tmp/ar-cleanup.json << 'ARCLEANUP'
[
  {
    "name": "keep-recent-versions",
    "action": {"type": "Keep"},
    "mostRecentVersions": {"keepCount": 20}
  },
  {
    "name": "delete-stale",
    "action": {"type": "Delete"},
    "condition": {"olderThan": "30d"}
  }
]
ARCLEANUP
gcloud artifacts repositories set-cleanup-policies "${REPO_NAME}" \
    --location="${REGION}" \
    --policy=/tmp/ar-cleanup.json
rm /tmp/ar-cleanup.json
echo "   ✅ Cleanup policy set: keep 20 most recent, delete older than 30d"

# ---------------------------------------------------------------------------
# Step 5: Store EIA API Key in Secret Manager
# ---------------------------------------------------------------------------
echo ""
echo "🔐 Step 5: Storing EIA API key in Secret Manager..."
if gcloud secrets describe eia-api-key &>/dev/null; then
    echo "   Secret 'eia-api-key' already exists."
    read -p "   Update it? (y/N): " UPDATE_SECRET
    if [[ "${UPDATE_SECRET}" =~ ^[Yy]$ ]]; then
        read -sp "   Enter your EIA API key: " EIA_KEY
        echo ""
        echo -n "${EIA_KEY}" | gcloud secrets versions add eia-api-key --data-file=-
        echo "   ✅ Secret updated."
    fi
else
    read -sp "   Enter your EIA API key: " EIA_KEY
    echo ""
    echo -n "${EIA_KEY}" | gcloud secrets create eia-api-key --data-file=-
    echo "   ✅ Secret created."
fi

# ---------------------------------------------------------------------------
# Step 5b: Create GCS Bucket for Parquet Persistence
# ---------------------------------------------------------------------------
echo ""
echo "📁 Step 5b: Creating GCS bucket for data persistence..."
BUCKET_NAME="${PROJECT_ID}-energy-cache"

if gsutil ls -b "gs://${BUCKET_NAME}" &>/dev/null; then
    echo "   Bucket '${BUCKET_NAME}' already exists. Skipping."
else
    gsutil mb -p "${PROJECT_ID}" -l "${REGION}" -b on "gs://${BUCKET_NAME}"
    echo "   ✅ Bucket created: gs://${BUCKET_NAME}"
fi

# Lifecycle rule: delete objects older than 90 days
cat > /tmp/lifecycle.json << 'LIFECYCLE'
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {"age": 90}
    }
  ]
}
LIFECYCLE
gsutil lifecycle set /tmp/lifecycle.json "gs://${BUCKET_NAME}"
rm /tmp/lifecycle.json
echo "   ✅ Lifecycle rule set: delete after 90 days"

# ---------------------------------------------------------------------------
# Step 6: Create Service Account for Cloud Run
# ---------------------------------------------------------------------------
echo ""
echo "👤 Step 6: Creating service account..."
SA_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
if gcloud iam service-accounts describe "${SA_EMAIL}" &>/dev/null; then
    echo "   Service account already exists. Skipping."
else
    gcloud iam service-accounts create "${SERVICE_ACCOUNT_NAME}" \
        --display-name="Energy Forecast Dashboard"
    echo "   ✅ Service account created."
fi

# Grant Secret Manager access
echo "   Granting Secret Manager access..."
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/secretmanager.secretAccessor" \
    --quiet

# Grant Cloud Run invoker (for health checks)
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/run.invoker" \
    --quiet
echo "   ✅ IAM bindings set."

# Grant GCS access for Parquet persistence
echo "   Granting GCS access..."
gsutil iam ch "serviceAccount:${SA_EMAIL}:roles/storage.objectAdmin" "gs://${BUCKET_NAME}"
echo "   ✅ GCS access granted."

# ---------------------------------------------------------------------------
# Step 7: Set Default Region
# ---------------------------------------------------------------------------
echo ""
echo "🌍 Step 7: Setting default region..."
gcloud config set run/region "${REGION}"
echo "   ✅ Default region: ${REGION}"

# ---------------------------------------------------------------------------
# Step 8: Set up Workload Identity Federation (for GitHub Actions)
# ---------------------------------------------------------------------------
echo ""
echo "🔗 Step 8: Setting up Workload Identity Federation for GitHub Actions..."
echo "   ⚠️  This requires your GitHub repo URL."
read -p "   Enter your GitHub repo (e.g., your-username/energy-forecast): " GITHUB_REPO

if [ -n "${GITHUB_REPO}" ]; then
    POOL_NAME="github-pool"
    PROVIDER_NAME="github-provider"
    
    # Create workload identity pool
    if ! gcloud iam workload-identity-pools describe "${POOL_NAME}" --location="global" &>/dev/null; then
        gcloud iam workload-identity-pools create "${POOL_NAME}" \
            --location="global" \
            --display-name="GitHub Actions Pool"
    fi

    # Create provider
    if ! gcloud iam workload-identity-pools providers describe "${PROVIDER_NAME}" \
        --location="global" --workload-identity-pool="${POOL_NAME}" &>/dev/null; then
        gcloud iam workload-identity-pools providers create-oidc "${PROVIDER_NAME}" \
            --location="global" \
            --workload-identity-pool="${POOL_NAME}" \
            --display-name="GitHub Provider" \
            --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
            --issuer-uri="https://token.actions.githubusercontent.com"
    fi

    # Allow GitHub repo to impersonate service account
    gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
        --role="roles/iam.workloadIdentityUser" \
        --member="principalSet://iam.googleapis.com/projects/$(gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)')/locations/global/workloadIdentityPools/${POOL_NAME}/attribute.repository/${GITHUB_REPO}" \
        --quiet

    # Grant Cloud Run and Artifact Registry permissions to SA
    for ROLE in roles/run.admin roles/artifactregistry.writer roles/cloudbuild.builds.builder roles/iam.serviceAccountUser; do
        gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
            --member="serviceAccount:${SA_EMAIL}" \
            --role="${ROLE}" \
            --quiet
    done

    WIF_PROVIDER="projects/$(gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)')/locations/global/workloadIdentityPools/${POOL_NAME}/providers/${PROVIDER_NAME}"
    
    echo ""
    echo "   ✅ Workload Identity Federation configured."
    echo ""
    echo "   ┌──────────────────────────────────────────────────┐"
    echo "   │ Add these as GitHub repository secrets:          │"
    echo "   ├──────────────────────────────────────────────────┤"
    echo "   │ WIF_PROVIDER = ${WIF_PROVIDER}"
    echo "   │ GCP_SA_EMAIL = ${SA_EMAIL}"
    echo "   └──────────────────────────────────────────────────┘"
else
    echo "   Skipped WIF setup. You can configure it later."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================"
echo "  ✅ GCP Setup Complete!"
echo "================================================"
echo ""
echo "  Project:          ${PROJECT_ID}"
echo "  Region:           ${REGION}"
echo "  Artifact Registry: ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"
echo "  Service Account:  ${SA_EMAIL}"
echo "  Secret:           eia-api-key"
echo ""
echo "  Next steps:"
echo "  1. Add GitHub secrets (WIF_PROVIDER, GCP_SA_EMAIL)"
echo "  2. Build and deploy:"
echo "     gcloud run deploy ${SERVICE_NAME} --source . \\"
echo "       --region ${REGION} --allow-unauthenticated \\"
echo "       --memory 2Gi --timeout 300 \\"
echo "       --set-secrets 'EIA_API_KEY=eia-api-key:latest'"
echo ""
