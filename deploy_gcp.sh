#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy_gcp.sh
# One-time GCP setup + manual deploy for SA Credit Stress Monitor.
#
# What this script does (in order):
#   1.  Sets the active GCP project
#   2.  Enables Cloud Run, Container Registry, and Cloud Build APIs
#   3.  Creates a service account for GitHub Actions CI/CD (once only)
#   4.  Grants that SA the minimum IAM roles needed (Cloud Run + GCR push)
#   5.  Exports the SA key as base64 — paste it into GitHub Secrets
#   6.  Trains the model locally if artefacts are missing
#   7.  Configures Docker auth for GCR
#   8.  Builds the Docker image
#   9.  Pushes to Google Container Registry
#   10. Deploys to Cloud Run (africa-south1 — Johannesburg)
#   11. Smoke-tests the live endpoint
#
# Prerequisites:
#   1. gcloud CLI installed: https://cloud.google.com/sdk/docs/install
#   2. Authenticated:        gcloud auth login
#   3. Project created in GCP console with billing enabled
#   4. Docker running locally
#
# Usage:
#   chmod +x deploy_gcp.sh
#   ./deploy_gcp.sh [gcp-project-id]
#
#   If no project ID is passed, defaults to: sa-credit-stress-prod
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config ─────────────────────────────────────────────────────────────────
PROJECT_ID="${1:-sa-credit-stress-prod}"   # default project ID
REGION="africa-south1"                     # Johannesburg — lowest latency for SA
SERVICE="sa-credit-stress-monitor"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE}"
SA_NAME="github-actions-deployer"          # service account for CI/CD
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
KEY_FILE="gcp-sa-key.json"                 # temp key file — deleted after export

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  SA Credit Stress Monitor — GCP Cloud Run Deployment"
echo "══════════════════════════════════════════════════════════════"
echo "  Project:  ${PROJECT_ID}"
echo "  Region:   ${REGION}  (Johannesburg)"
echo "  Service:  ${SERVICE}"
echo "  Image:    ${IMAGE}"
echo ""

# ── 1. Set active project ──────────────────────────────────────────────────
echo "▶ [1/11] Setting GCP project..."
gcloud config set project "${PROJECT_ID}"

# ── 2. Enable required APIs ────────────────────────────────────────────────
echo "▶ [2/11] Enabling Cloud Run + Container Registry + Cloud Build APIs..."
gcloud services enable \
  run.googleapis.com \
  containerregistry.googleapis.com \
  cloudbuild.googleapis.com \
  iam.googleapis.com

# ── 3. Create CI/CD service account (idempotent — skip if already exists) ──
echo "▶ [3/11] Creating GitHub Actions service account..."
if gcloud iam service-accounts describe "${SA_EMAIL}" &>/dev/null; then
  echo "  Service account already exists — skipping creation."
else
  gcloud iam service-accounts create "${SA_NAME}" \
    --display-name="GitHub Actions — SA Credit Stress Monitor deployer"
  echo "  Created: ${SA_EMAIL}"
fi

# ── 4. Grant minimum IAM roles ────────────────────────────────────────────
# Cloud Run Admin  — deploy new revisions
# Storage Admin    — push images to GCR (backed by Cloud Storage)
# Service Account User — allow Cloud Run to act as the SA
echo "▶ [4/11] Granting IAM roles to service account..."
for ROLE in \
  "roles/run.admin" \
  "roles/storage.admin" \
  "roles/iam.serviceAccountUser"; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${ROLE}" \
    --quiet
  echo "  Granted: ${ROLE}"
done

# ── 5. Export SA key → base64 for GitHub Secrets ─────────────────────────
echo "▶ [5/11] Exporting service account key..."
gcloud iam service-accounts keys create "${KEY_FILE}" \
  --iam-account="${SA_EMAIL}"

B64_KEY=$(base64 -w 0 "${KEY_FILE}" 2>/dev/null || base64 "${KEY_FILE}")
rm -f "${KEY_FILE}"   # delete the raw JSON immediately

echo ""
echo "  ┌──────────────────────────────────────────────────────────┐"
echo "  │  ACTION REQUIRED — add these to GitHub repo Secrets:     │"
echo "  │  Settings → Secrets and variables → Actions → New secret │"
echo "  ├──────────────────────────────────────────────────────────┤"
echo "  │  Secret name:  GCP_PROJECT_ID                            │"
echo "  │  Secret value: ${PROJECT_ID}"
echo "  │                                                          │"
echo "  │  Secret name:  GCP_SA_KEY                                │"
echo "  │  Secret value: (base64 key printed below)                │"
echo "  └──────────────────────────────────────────────────────────┘"
echo ""
echo "  ── GCP_SA_KEY value (copy everything between the lines) ───"
echo "  ${B64_KEY}"
echo "  ────────────────────────────────────────────────────────────"
echo ""
echo "  Also add your FRED API key if you have one:"
echo "  Secret name:  FRED_API_KEY"
echo "  Get one free: https://fred.stlouisfed.org/docs/api/api_key.html"
echo ""
read -rp "  Press ENTER once you have copied the key to GitHub Secrets..."
echo ""

# ── 6. Train model locally (if artefacts missing) ─────────────────────────
echo "▶ [6/11] Checking model artefacts..."
if [[ ! -f "data/processed/xgb_model.joblib" ]]; then
  echo "  Artefacts not found — training now (seed mode, ~30s)..."
  PYTHONPATH=. python -m src.models.train
  PYTHONPATH=. python -m src.models.explain
else
  echo "  Artefacts found — skipping training."
fi

# ── 7. Configure Docker auth for GCR ─────────────────────────────────────
# Must happen before docker build so the push step is already authenticated.
echo "▶ [7/11] Configuring Docker auth for GCR..."
gcloud auth configure-docker gcr.io --quiet

# ── 8. Build Docker image ─────────────────────────────────────────────────
echo "▶ [8/11] Building Docker image..."
docker build \
  --tag "${IMAGE}:latest" \
  --tag "${IMAGE}:$(git rev-parse --short HEAD 2>/dev/null || echo manual)" \
  .

# ── 9. Push to Google Container Registry ─────────────────────────────────
echo "▶ [9/11] Pushing image to GCR..."
docker push "${IMAGE}:latest"

# ── 10. Deploy to Cloud Run ───────────────────────────────────────────────
echo "▶ [10/11] Deploying to Cloud Run (${REGION})..."
gcloud run deploy "${SERVICE}" \
  --image="${IMAGE}:latest" \
  --region="${REGION}" \
  --platform=managed \
  --memory=1Gi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=3 \
  --concurrency=80 \
  --timeout=60s \
  --allow-unauthenticated \
  --port=8080

# ── 11. Smoke test ────────────────────────────────────────────────────────
echo "▶ [11/11] Running smoke test..."
SERVICE_URL=$(gcloud run services describe "${SERVICE}" \
  --region="${REGION}" \
  --format='value(status.url)')

HEALTH=$(curl -sf "${SERVICE_URL}/health")
echo "  Health response: ${HEALTH}"

python3 -c "
import json, sys
data = json.loads('${HEALTH}')
assert data.get('status') == 'ok',      f'Expected status=ok, got: {data}'
assert data.get('model_loaded') == True, f'Model not loaded: {data}'
print('  ✓ Health check passed')
"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  ✓ Deployment complete!"
echo ""
echo "  API base URL:  ${SERVICE_URL}"
echo "  Swagger docs:  ${SERVICE_URL}/docs"
echo "  Health check:  ${SERVICE_URL}/health"
echo "  Predict:       POST ${SERVICE_URL}/predict"
echo "  Historical:    GET  ${SERVICE_URL}/historical"
echo ""
echo "  Next: paste ${SERVICE_URL} into Streamlit Cloud Secrets"
echo "        as  API_BASE_URL = \"${SERVICE_URL}\""
echo "══════════════════════════════════════════════════════════════"
