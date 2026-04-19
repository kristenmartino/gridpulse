# GridPulse Scheduled Jobs

The expensive parts of GridPulse — EIA / weather fetches, feature
engineering, model training, forecasting, and backtests — run as two
Cloud Run Jobs driven by Cloud Scheduler:

| Job | Cadence | What it does |
|---|---|---|
| `gridpulse-scoring-job` | Hourly (`0 * * * *`) | Refresh Redis: actuals, weather, generation, forecasts, alerts, diagnostics, weather-correlation, `wattcast:meta:last_scored` |
| `gridpulse-training-job` | Daily (`0 4 * * *` UTC) | Retrain XGBoost / Prophet / SARIMAX per region, persist to GCS, recompute backtests, `wattcast:meta:last_trained` |

The Dash web service (`gridpulse`) reads Redis only. A Redis miss becomes
a degraded "warming" state in the UI — it never blocks the request on an
API fetch or inline training. See `config.REQUIRE_REDIS`.

Staging mirrors this topology with `-dev` suffixes
(`gridpulse-scoring-job-dev`, `gridpulse-training-job-dev`).

## Components

```
                 Cloud Scheduler
 ┌───────────────────────────────────┐
 │ gridpulse-scoring-hourly   0 * * * * │
 │ gridpulse-training-daily   0 4 * * * │
 └──────────────┬────────────────────┘
                │ OIDC invoke
                ▼
        Cloud Run Jobs
 ┌───────────────────────────────────┐
 │ gridpulse-scoring-job             │──── fetch EIA/weather, predict,
 │ gridpulse-training-job            │     write Redis + alerts/diag
 └──────┬───────────────┬────────────┘
        │               │
        ▼               ▼
  GCS (models/)   Redis (wattcast:*)
                        ▲
              ┌─────────┴────────────┐
              │ Cloud Run Service    │
              │ gridpulse (web tier) │
              │  - Redis-only reads  │
              │  - "warming" state   │
              │    if cache is cold  │
              └──────────────────────┘
```

## One-time setup

The workflows in `.github/workflows/deploy-{prod,dev}.yml` already deploy
the two Cloud Run Jobs. What lives outside the repo is the **scheduler
and IAM setup** — those are one-time `gcloud` commands.

### 1. Service accounts

Two SAs:

- `gridpulse-job@nextera-portfolio.iam.gserviceaccount.com` — runs the
  jobs themselves. Needs Secret Manager, GCS, and Redis VPC access.
- `gridpulse-scheduler@nextera-portfolio.iam.gserviceaccount.com` —
  identity Cloud Scheduler uses to invoke the jobs. Needs `run.invoker`
  on the specific jobs.

```bash
PROJECT=nextera-portfolio
REGION=us-east1

# SA for the job runtime
gcloud iam service-accounts create gridpulse-job \
  --project=$PROJECT \
  --display-name="GridPulse Cloud Run Jobs"

gcloud secrets add-iam-policy-binding eia-api-key \
  --project=$PROJECT \
  --member=serviceAccount:gridpulse-job@$PROJECT.iam.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor

gcloud storage buckets add-iam-policy-binding \
  gs://nextera-portfolio-energy-cache \
  --member=serviceAccount:gridpulse-job@$PROJECT.iam.gserviceaccount.com \
  --role=roles/storage.objectAdmin

# VPC connector usage (for Redis Memorystore)
gcloud projects add-iam-policy-binding $PROJECT \
  --member=serviceAccount:gridpulse-job@$PROJECT.iam.gserviceaccount.com \
  --role=roles/vpcaccess.user

# SA for Cloud Scheduler to invoke the jobs
gcloud iam service-accounts create gridpulse-scheduler \
  --project=$PROJECT \
  --display-name="GridPulse Cloud Scheduler"

for JOB in gridpulse-scoring-job gridpulse-training-job; do
  gcloud run jobs add-iam-policy-binding $JOB \
    --project=$PROJECT --region=$REGION \
    --member=serviceAccount:gridpulse-scheduler@$PROJECT.iam.gserviceaccount.com \
    --role=roles/run.invoker
done
```

The GitHub Actions workflows need to be updated (one-time) to pass
`--service-account=gridpulse-job@...` when deploying the jobs. The
current workflows rely on the default compute SA, which is acceptable
for staging — tighten this before relying on the jobs in production.

### 2. Cloud Scheduler entries

```bash
PROJECT=nextera-portfolio
REGION=us-east1
SA=gridpulse-scheduler@$PROJECT.iam.gserviceaccount.com

# Hourly scoring
gcloud scheduler jobs create http gridpulse-scoring-hourly \
  --project=$PROJECT --location=$REGION \
  --schedule="0 * * * *" \
  --time-zone="UTC" \
  --uri="https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT/jobs/gridpulse-scoring-job:run" \
  --http-method=POST \
  --oidc-service-account-email=$SA \
  --oidc-token-audience="https://$REGION-run.googleapis.com/"

# Daily training (04:00 UTC)
gcloud scheduler jobs create http gridpulse-training-daily \
  --project=$PROJECT --location=$REGION \
  --schedule="0 4 * * *" \
  --time-zone="UTC" \
  --uri="https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT/jobs/gridpulse-training-job:run" \
  --http-method=POST \
  --oidc-service-account-email=$SA \
  --oidc-token-audience="https://$REGION-run.googleapis.com/"
```

Staging mirrors with `gridpulse-scoring-hourly-dev` /
`gridpulse-training-daily-dev` pointing at the `-dev` jobs.

## Bootstrap (first deploy)

Models must exist in GCS before scoring can produce forecasts. First
deploy:

1. Push to the target branch → CI builds + deploys the web service and
   both Cloud Run Jobs.
2. Run the training job once manually so GCS has models:

   ```bash
   gcloud run jobs execute gridpulse-training-job \
     --region=us-east1 --wait
   ```

3. Run the scoring job once to populate Redis:

   ```bash
   gcloud run jobs execute gridpulse-scoring-job \
     --region=us-east1 --wait
   ```

4. Verify:

   ```bash
   # Models present in GCS
   gcloud storage cat gs://nextera-portfolio-energy-cache/cache/models/latest.json

   # Redis populated
   redis-cli -h $REDIS_HOST GET wattcast:meta:last_scored
   ```

5. Enable the Cloud Scheduler entries (they default to `ENABLED`, but
   `gcloud scheduler jobs pause/resume` is available if you need to
   stop the cron without deleting the schedule).

If the scoring job runs before models exist, it logs
`scoring_job_no_model_yet` per region and still refreshes actuals /
weather / generation. Forecasts and diagnostics appear once the training
job finishes.

## Local dev

Dev works without Cloud Run Jobs or Redis:

- Default `ENVIRONMENT=development` sets `PRECOMPUTE_ENABLED=True` and
  `REQUIRE_REDIS=False`.
- The Dash app spawns a background thread on the first request that
  invokes `jobs.scoring_job.run()` in-process. This is the dev
  equivalent of Cloud Scheduler.
- You can also run either job directly:

  ```bash
  python -m jobs scoring    # hourly-style refresh
  python -m jobs training   # retrains + writes to GCS
  ```

Set `REQUIRE_REDIS=true` locally to exercise the degraded / warming UI:

```bash
REQUIRE_REDIS=true python app.py
```

## Operations

- **Redis staleness**: `wattcast:meta:last_scored.updated_at` should be
  within the last ~90 minutes. Consider a Cloud Monitoring alert on
  absence.
- **Job failure**: Cloud Run Jobs exit non-zero only when every region
  fails. Watch the per-execution logs for per-region
  `scoring_job_region_crashed` / `training_job_region_crashed` entries.
- **Model promotion**: the training job updates
  `gs://.../cache/models/latest.json` last — this is the atomic pointer.
  The scoring job reads `latest.json` on every run, so a new training
  result propagates on the next hourly tick.
- **Rollback**: revert the previous `{version}` in `latest.json` by
  editing the file directly (it's just JSON). The local-disk cache on
  warm Cloud Run instances may still serve the older version until
  eviction, which is the desired behavior during rollback.

## File map

- `jobs/__main__.py` — CLI dispatcher (`python -m jobs {scoring|training}`)
- `jobs/scoring_job.py` — hourly entry point
- `jobs/training_job.py` — daily entry point
- `jobs/phases.py` — shared phase functions (data fetch, feature prep,
  Redis writers)
- `models/persistence.py` — GCS-backed model save / load with
  `latest.json` pointer and local-disk cache
- `.github/workflows/deploy-{prod,dev}.yml` — CI deploys all three
  surfaces (service + two jobs) from the same image
- `components/callbacks.py` — `REQUIRE_REDIS` gates at `load_data`,
  `_run_forecast_outlook`, `_run_backtest_for_horizon`
- `components/error_handling.py` — `warming` confidence level +
  `warming_state()` / `is_warming()` helpers
