# GridPulse Scheduled Jobs

The expensive parts of GridPulse — EIA / weather fetches, feature
engineering, model training, forecasting, and backtests — run as two
Cloud Run Jobs driven by Cloud Scheduler:

| Job | Cadence | What it does |
|---|---|---|
| `gridpulse-scoring-job` | Hourly (`0 * * * *`) | Refresh Redis: actuals, weather, generation, forecasts, alerts, diagnostics, weather-correlation, `gridpulse:meta:last_scored` |
| `gridpulse-training-job` | Daily (`0 4 * * *` UTC) | Retrain XGBoost / Prophet / SARIMAX per region, persist to GCS, recompute backtests, `gridpulse:meta:last_trained` |

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
  GCS (models/)   Redis (gridpulse:*)
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

Both `deploy-prod.yml` and `deploy-dev.yml` pin the Cloud Run Jobs to this
SA via `--service-account=gridpulse-job@$PROJECT.iam.gserviceaccount.com`,
so redeploys preserve the least-privilege binding rather than falling back
to the default compute SA.

### 2. Cloud Scheduler entries

```bash
PROJECT=nextera-portfolio
REGION=us-east1
SA=gridpulse-scheduler@$PROJECT.iam.gserviceaccount.com

# Hourly scoring — 1 retry covers a single transient 5xx without doubling up
# the next scheduled tick (which is only an hour away anyway).
gcloud scheduler jobs create http gridpulse-scoring-hourly \
  --project=$PROJECT --location=$REGION \
  --schedule="0 * * * *" \
  --time-zone="UTC" \
  --uri="https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT/jobs/gridpulse-scoring-job:run" \
  --http-method=POST \
  --oidc-service-account-email=$SA \
  --oidc-token-audience="https://$REGION-run.googleapis.com/" \
  --max-retry-attempts=1 \
  --min-backoff=2m \
  --max-backoff=5m

# Daily training (04:00 UTC) — 3 retries with longer backoff covers
# Cloud Run regional API blips up to ~1 hour. A miss here costs a full
# day vs a missed scoring tick (1 hour), so we tolerate retries more
# aggressively. See #141 — pre-2026-05-22 the default was zero retries
# and a single transient 503 on 2026-05-21 silently skipped the day's
# training cycle.
gcloud scheduler jobs create http gridpulse-training-daily \
  --project=$PROJECT --location=$REGION \
  --schedule="0 4 * * *" \
  --time-zone="UTC" \
  --uri="https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT/jobs/gridpulse-training-job:run" \
  --http-method=POST \
  --oidc-service-account-email=$SA \
  --oidc-token-audience="https://$REGION-run.googleapis.com/" \
  --max-retry-attempts=3 \
  --min-backoff=5m \
  --max-backoff=30m
```

If you're updating an existing scheduler entry rather than creating it
fresh, swap `create` for `update` and drop the create-only flags:

```bash
gcloud scheduler jobs update http gridpulse-training-daily \
  --location=$REGION --project=$PROJECT \
  --max-retry-attempts=3 --min-backoff=5m --max-backoff=30m

gcloud scheduler jobs update http gridpulse-scoring-hourly \
  --location=$REGION --project=$PROJECT \
  --max-retry-attempts=1 --min-backoff=2m --max-backoff=5m
```

Staging mirrors with `gridpulse-scoring-hourly-dev` /
`gridpulse-training-daily-dev` pointing at the `-dev` jobs.

## Deploy gating (CI → deploy)

Deploys are **gated behind CI** (PR-G2 / #146). The `deploy-prod.yml` /
`deploy-dev.yml` workflows do **not** run on push directly — they run on
`workflow_run` after the **CI** workflow completes, and only when CI
*succeeded*:

```
push to main ──▶ CI (security · lint · test · coverage · docker)
                  │
                  ├─ red  ──▶ deploy SKIPPED (if: conclusion == 'success')
                  └─ green ─▶ Deploy → Production
                              builds image tagged with the *exact* SHA
                              CI validated (workflow_run.head_sha), not
                              main's current HEAD — so a commit that lands
                              after CI passed can't ride out un-validated.
```

Before PR-G2 the deploy fired on every push to `main` independent of CI,
so a red build could ship. Now a failing test, lint error, or broken
Docker build blocks the deploy.

**One operational note:** under `workflow_run` the GitHub OIDC token's
`event_name` claim is `workflow_run` rather than `push`. If the GCP
Workload Identity Federation provider ever gets an attribute condition
pinned on `event_name`, deploy auth would break — the standard
`google-github-actions` WIF setup binds on `repository`, not event, so
this is not currently a problem. If a deploy ever fails at the
"Authenticate to Google Cloud" step right after this change, that's the
first thing to check. Rollback is a one-line revert of the `on:` trigger.

## Bootstrap (first deploy)

Models must exist in GCS before scoring can produce forecasts. First
deploy:

1. Push to the target branch → CI runs; **on green**, the deploy
   workflow builds + ships the web service and both Cloud Run Jobs.
   (A red CI blocks the deploy — see "Deploy gating" above.)
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
   redis-cli -h $REDIS_HOST GET gridpulse:meta:last_scored
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

## Cache-invalidating deploys

**Important:** Cloud Run **deploy** ≠ Cloud Run **execute**. The CI
workflow (`deploy-prod.yml`) builds a new image and updates BOTH Jobs'
container definitions on every push to `main` — but Jobs only *run*
when Cloud Scheduler triggers them (hourly for scoring, daily for
training). A fresh deploy with new code sits dormant until the next
cron tick.

Most deploys don't notice this: the previous scoring run already
populated Redis with the format the new web code expects, and the
next hourly tick refreshes it under the new code. The web service
keeps reading the same keys throughout.

**Any deploy that changes the Redis key namespace, payload schema, or
prefix invalidates that assumption.** After the deploy, the new web
code looks for keys that the old scoring run didn't write — the
dashboard shows "Data warming up" until the next hourly tick (up to
60 minutes for scoring, up to 24 hours for training-derived
backtests).

For these deploys, force-execute both Jobs immediately so keys
populate without the wait. **Do NOT manually redeploy the Jobs first**
— CI's ``deploy-prod.yml`` already ran ``gcloud run jobs deploy``
for both with the new image. The Jobs are sitting on the new code;
they just haven't been triggered yet.

```bash
# After CI finishes deploying:
gcloud run jobs execute gridpulse-scoring-job  --region us-east1 --wait
gcloud run jobs execute gridpulse-training-job --region us-east1
# (training takes ~3.5h with the 3-task parallel split — don't --wait)
```

If you ever DO need to manually update a Job's image (rolling back
to a previous commit, or recovering from a CI failure that pushed
to GHCR but didn't update Cloud Run), the real command is:

```bash
# Latest from CI's ``prod-latest`` tag:
gcloud run jobs update gridpulse-training-job \
  --image us-east1-docker.pkg.dev/nextera-portfolio/portfolio/gridpulse:prod-latest \
  --region us-east1

# Or a specific commit SHA (immutable — recommended for rollback):
gcloud run jobs update gridpulse-training-job \
  --image us-east1-docker.pkg.dev/nextera-portfolio/portfolio/gridpulse:<sha> \
  --region us-east1
```

**Cases that need this dance:**

- `REDIS_KEY_PREFIX` change (e.g. issue #91 `wattcast:` → `gridpulse:`)
- Adding a new Redis key the web reads (web 404s the key until the
  scoring job's next tick writes it)
- Payload schema break (web parses the new shape, last scoring run
  wrote the old shape → parse error or empty render)
- Changing the model artifact format (training job needs to re-emit
  `latest.json` in the new shape before the scoring job will load it)

**Symptom map for the live dashboard during the warming window:**

| What's missing | Means |
|---|---|
| Demand chart empty / "warming" | Redis `gridpulse:actuals:{region}` not populated — wait for next scoring tick or force-execute |
| Forecast chart empty | `gridpulse:forecast:{region}:1h` not populated — same fix |
| Backtest tab empty | `gridpulse:backtest:{exog_mode}:{region}:{horizon}` not populated — needs training tick (daily) or force-execute |
| Model accuracy card SHOWS numbers but charts are empty | Expected: MAPE/RMSE/MAE/R² come from GCS `meta.json` (persisted at training time, independent of Redis). The card is reading from the durable source while the live data flows from Redis are still warming. Not a bug. |

## Alerting + incident response (PR-G10 / #148)

A Cloud Monitoring alert policy (`GridPulse — Cloud Run Job failed
execution`, live since 2026-05-29) fires on a failed execution of either
job and emails `kristen.e.martino@gmail.com`. Policy-as-code +
re-apply commands live in [`docs/monitoring/`](monitoring/README.md).

This closes the gap behind two 2026-05 incidents that were found by
manual check rather than alert: the silent training-scheduler miss
(#141) and the all-region forecast outage (#161).

**One-time:** the email channel needs a verification click (sent on
creation) — confirm `gcloud beta monitoring channels describe <id>
--format='value(verificationStatus)'` reads `VERIFIED`.

### When the job-failure alert fires

1. **Identify**: `bash scripts/audit/check_overnight_training.sh`
   (training) or `gcloud run jobs executions list --job=gridpulse-scoring-job
   --region=us-east1` (scoring) — find the failed execution.
2. **Diagnose**: tail the failed execution's logs for the per-region
   cause (`scoring_job_region_crashed`, `job_insufficient_feature_rows`,
   `all_models_failed`, etc.).
3. **Mitigate**:
   - Training miss → trigger a make-up run:
     `gcloud run jobs execute gridpulse-training-job --region=us-east1`.
   - Scoring miss (single, transient) → the next hourly tick self-heals;
     force one now if urgent: `gcloud run jobs execute gridpulse-scoring-job
     --region=us-east1`.
   - **Repeated timeouts** (≥2 consecutive ticks; logs show "Terminating
     task because it has reached the maximum timeout" and each attempt
     reaches only a *partial* BA set) → **diagnose before you touch anything;
     the first two times this happened the right lever was different each
     time.** Work the "why it timed out" branch below first. Only if the cause
     is genuinely our own work growing — not upstream — is the timeout the
     right knob, and even then read the cadence rule first. First hit
     2026-06-01 (854s crept over the 900s cap → 1800s; [#171](https://github.com/kristenmartino/gridpulse/issues/171));
     the training job hit the same wall 2026-05-03.
     **Proactive since 2026-07-04 ([#171](https://github.com/kristenmartino/gridpulse/issues/171)):** a
     `scoring_runtime_creep` alert should reach you *before* an outright
     timeout — it fires when `elapsed_s` exceeds 70% of the `--task-timeout`
     for 3 consecutive runs (`docs/monitoring/scoring_runtime_creep_alert.json`),
     turning a creeping runtime into a scheduled fix, not a surprise 3am page.
     Note its blind spot: it only evaluates on runs that **complete**, and the
     streak resets on any non-breaching run — so killed ticks contribute
     nothing and the alert can arrive *after* the first timeouts, as it did
     2026-08-04.
     **But first check *why* it timed out.** Read `top_phases` off the
     `scoring_runtime_creep` log, or `jsonPayload.event="scoring_phase_rollup"`
     (emitted every run). If `fetch` leads, it is upstream, not our own creep —
     and there are **two upstream shapes that need opposite responses**. Count
     them:

     ```bash
     for e in eia_request_exception eia_max_retries_exceeded \
              eia_circuit_tripped eia_gcs_fallback; do
       printf '%s: ' "$e"
       gcloud logging read "resource.type=\"cloud_run_job\" AND jsonPayload.event=\"$e\"" \
         --project=nextera-portfolio --freshness=2h --format='value(timestamp)' | wc -l
     done
     ```

     - **Total outage** — `eia_circuit_tripped` present, `eia_gcs_fallback` /
       `eia_stale_fallback` non-zero. The [#174](https://github.com/kristenmartino/gridpulse/issues/174)
       breaker is working: it fail-fasts to last-known GCS data after a few
       *consecutive* hard failures, so runtime stays bounded and the next tick
       self-heals. **Wait it out**; `last_scored` catches up when EIA recovers.
       Bit us 2026-06-04 (03:00–04:00 UTC, EIA 504s; normal runtime ~700s).
     - **Partial degradation** — exceptions high but `eia_max_retries_exceeded`
       and the fallbacks at **ZERO**, and **no** `eia_circuit_tripped`. Every
       call is eventually *succeeding*; the run is paying the full retry budget
       for calls that work. **The breaker cannot help here** — it counts
       *consecutive* hard failures and `record_success()` resets that counter,
       so interleaved successes keep it closed by construction. This does
       **not** self-heal: runtime scales with the exception rate and keeps
       climbing. Bit us 2026-08-04 (~16:00 UTC onward, EIA 502/504 + 30s read
       timeouts; ~800s baseline → two ticks killed at the 1800s cap, one
       surviving at 1792s). **Act:** raise `PRECOMPUTE_MAX_WORKERS` (below).

     **The live lever (no deploy):** `PRECOMPUTE_MAX_WORKERS` is
     env-overridable (`config.py`) and is *not* set in either deploy workflow,
     so it is the only runtime knob you can turn without shipping an image.
     Raise `--cpu` with it — the compute-heavy phases already oversubscribe at
     `--cpu 2`, so more workers alone converts a fetch-bound win into a
     compute-bound loss. Snapshot first
     (`gcloud run jobs describe … --format=yaml > ~/scoring-job.pre.yaml`),
     then:

     ```bash
     gcloud run jobs update gridpulse-scoring-job --region=us-east1 \
       --update-env-vars=PRECOMPUTE_MAX_WORKERS=8 --cpu=4 --memory=8Gi
     ```

     ⚠ **`--update-env-vars`, never `--set-env-vars`.** `--set-env-vars`
     *replaces* the whole env map and would drop `REDIS_HOST`, `REDIS_PORT`,
     `GCS_BUCKET_NAME`, `GCS_ENABLED`, `ENVIRONMENT`, `LOG_LEVEL` — all set
     together in `deploy-prod.yml`. A scoring job without `REDIS_HOST` does
     not run slow; it writes **nothing**, turning a degraded incident into a
     total outage. Roll back with `--remove-env-vars=PRECOMPUTE_MAX_WORKERS`.
     Sanity-check `eia_rate_limited` afterwards: it was **zero** on
     2026-08-04, which is what made more concurrency safe. Under a 429 regime
     it is not — you would be adding load to something already refusing it.
     **Any merge to `main` silently reverts this** (CI redeploys with
     `--set-env-vars` and `--cpu 2`), so freeze prod deploys or land the
     change in both workflows.

     **The `--task-timeout` rule — why it is almost never the answer here.**
     `task-timeout × (max-retries + 1)` must stay below the scheduler
     interval. At `1800 × 2 = 3600s` against an hourly cadence, the scoring
     job **already has zero margin**, and on 2026-08-04 it went negative: the
     19:00 execution's retry finished at 20:01, *after* the 20:00 execution
     had started — two scoring processes at once, hammering a dependency that
     was failing because it was overloaded. Raising the ceiling makes that
     overlap a permanent config property rather than bad luck. (The
     `deploy-prod.yml` comment claiming runs "can't overlap" predates this and
     is wrong.) The **training** job may bump freely — `--max-retries 0` on a
     daily cadence has no overlap hazard — which is the real shape of the
     rule: it is about the ratio to the schedule, not about EIA. If you do
     raise it, bump `SCORING_TASK_TIMEOUT_S` in `config.py` to match, or the
     creep alarm silences itself against a stale ceiling.
   - Systemic (all regions) → check `/health?deep=1`, suspect a
     data-source or feature-pipeline fault (cf. #161), roll back the
     image via `latest.json` if code-caused.
4. **Confirm recovery**: `curl "$PROD_URL/health?deep=1"` →
   `status: healthy`, `forecast_sample: ok`.

Known follow-ups (see `docs/monitoring/README.md`): a Cloud Scheduler
error alert (catches a scheduler-side miss even when no execution is
created) and a deep-`/health` degraded uptime alert (would catch a
#161-style outage where infra is healthy but no forecasts exist).

## Operations

- **Redis staleness**: `gridpulse:meta:last_scored.updated_at` should be
  within the last ~90 minutes. Surfaced by `/health` `last_scored` check;
  a dedicated absence alert is a documented follow-up (above).
- **Job failure**: Cloud Run Jobs exit non-zero only when every region
  fails — but the alert policy (above) fires on ANY failed execution.
  Watch per-execution logs for per-region
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
