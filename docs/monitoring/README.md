# Cloud Monitoring — alerting (PR-G10 / #150)

Policy-as-code for GridPulse's production alerting. These close the
observability gap that let two 2026-05 incidents (the silent training
miss and the all-region forecast outage) go undetected until a manual
`/health` / log check found them.

## What's here

| File | Alert |
|---|---|
| `cloud_run_job_failure_alert.json` | Fires when `gridpulse-training-job` or `gridpulse-scoring-job` records a **failed execution** (`run.googleapis.com/job/completed_execution_count{result="failed"}` > 0, summed hourly per job). |

## Notification channel

One email channel (`GridPulse ops (Kristen)` →
`kristen.e.martino@gmail.com`), id
`projects/nextera-portfolio/notificationChannels/7265334362271951327`.

```bash
# (re)create the channel
gcloud beta monitoring channels create \
  --project=nextera-portfolio \
  --display-name="GridPulse ops (Kristen)" \
  --type=email \
  --channel-labels=email_address=YOUR_EMAIL
```

> ⚠ **Email channels require a one-time verification click.** After
> creation Google sends a verification email; until it's clicked, the
> policy evaluates + creates incidents but does NOT deliver email.
> Confirm status: `gcloud beta monitoring channels describe <id>
> --format='value(verificationStatus)'` → should read `VERIFIED`.

## Apply / re-apply a policy

```bash
gcloud beta monitoring policies create \
  --project=nextera-portfolio \
  --policy-from-file=docs/monitoring/cloud_run_job_failure_alert.json \
  --notification-channels="projects/nextera-portfolio/notificationChannels/7265334362271951327"
```

Live as of 2026-05-29: policy
`projects/nextera-portfolio/alertPolicies/5965243952275624431` (enabled).

## Verification (one manual step)

CLI confirms the policy is enabled, correctly filtered, and channel-bound.
What can't be checked from the CLI (no clean `incidents list`; email is
verification-gated) is end-to-end **fire + deliver**. To confirm once:

```bash
# Throwaway failing execution → produces a `failed` execution metric.
# Overrides the command so it exits non-zero WITHOUT running real scoring
# (no Redis writes). Leaves one fake "failed" row in the job history.
gcloud run jobs execute gridpulse-scoring-job --region=us-east1 \
  --command=python --args=-c,"import sys; sys.exit(1)"
# Then: wait ~10-15 min, confirm an incident appears in the Cloud
# Monitoring console (Alerting → Incidents) and an email arrives.
```

## Follow-ups (not yet implemented)

- **Cloud Scheduler error alert** — fiddlier (no clean error metric; alert
  on `cloudscheduler.googleapis.com/job/attempt_count` filtered to
  non-2xx `response_code`). Would catch a scheduler-side miss like the
  2026-05-21 503 even when no execution is created. Tracked in #150.
- **Deep-`/health` degraded alert** — an uptime check hitting
  `/health?deep=1` and alerting on `status != healthy` would have caught
  the 2026-05-29 forecast outage directly (it had healthy infra but no
  forecast payloads). Strong complement to the job-failure alert.
