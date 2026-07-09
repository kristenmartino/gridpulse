# Cloud Monitoring — alerting (PR-G10 / #148, web tier #253)

Policy-as-code for GridPulse's production alerting. The **job-tier** policies
close the observability gap that let two 2026-05 incidents (the silent training
miss and the all-region forecast outage) go undetected until a manual `/health`
/ log check found them. The **web-tier** policies (#253) extend that maturity to
the now-public request path (JSON API #250/#251) — which previously had *no*
alerting, rate limiting, or cost guardrail on personal `--allow-unauthenticated`
billing.

## What's here

| File | Alert |
|---|---|
| `cloud_run_job_failure_alert.json` | **Job tier.** Fires when `gridpulse-training-job` or `gridpulse-scoring-job` records a **failed execution** (`run.googleapis.com/job/completed_execution_count{result="failed"}` > 0, summed hourly per job). Fires *after* a timeout. |
| `scoring_runtime_creep_alert.json` | **Job tier, early warning (#171).** Log-based (`conditionMatchedLog` on `jsonPayload.event="scoring_runtime_creep"`) — fires when the scoring job's `elapsed_s` exceeds `SCORING_RUNTIME_HEADROOM_FRACTION` of the task timeout for `SCORING_RUNTIME_CREEP_RUNS` consecutive runs. Warns on *approach* (~70% of the cap), before a tick is killed — the gap that let 2026-06-01 happen. |
| `scoring_partial_failure_alert.json` | **Job tier (#267).** Log-based (`jsonPayload.event="scoring_partial_failure"`) — fires when a run forecast fewer than `SCORING_MIN_OK_REGIONS` BAs (default 40/51) but at least one succeeded, so it exits 0. Catches a catastrophic *partial* failure (e.g. 1/51) the failed-execution alert can't see. |
| `web_service_5xx_alert.json` | **Web tier (#253).** Fires when the `gridpulse` service returns sustained 5xx (`run.googleapis.com/request_count{response_code_class="5xx"}` summed > 25 / 5 min). The request-path equivalent of the job-failure alert. |
| `web_service_max_instances_alert.json` | **Web tier (#253).** Fires when the service sits at its `max-instances` ceiling (4) for 15 min — the cost ceiling *and* the traffic-flood signal on the public surface. |
| `web_service_uptime_alert.json` | **Web tier (#253).** Fires when the public `/health` uptime check fails from >1 probe location over 10 min (service down or shallow-degraded). Filter is check-id-specific — see the note in the file. |

Both web-tier policies apply the same way as the job policies (see "Apply /
re-apply" below). The **uptime check** and the **billing budget** are separate
GCP resource types — their `gcloud` recipes are in the two sections just below.

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

**Applied 2026-07-08 (#253 + #171):**

| Resource | Live id |
|---|---|
| Cloud Run Job failed execution | `alertPolicies/5965243952275624431` |
| scoring-job runtime creep (#171) | `alertPolicies/5813319064717268577` |
| web service sustained 5xx | `alertPolicies/14035657251363314798` |
| web service pinned at max instances | `alertPolicies/7343953142414788448` |
| /health uptime check failing (alert) | `alertPolicies/1577408926164424010` |
| Uptime check config — public `/health` | `uptimeCheckConfigs/gridpulse-health-162OIAwsIpE` |
| Monthly budget — $150 (billing acct `01D68B-6BF1D9-B54F3B`) | `budgets/3363cac4-5a23-46ea-a51f-ddbbadeca827` |

All five alert policies + the uptime check + the budget are live and bound to
the email channel. The budget also emails the billing-account admins by default.

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

## Web-tier cost guardrail — billing budget + anomaly alert (#253)

The **highest-leverage, cheapest** guard: nothing else tells you a flood is
happening on personal billing until the statement. A `max-instances=4` pin costs
~$456/mo vs ~$114 idle. Budgets are a **Cloud Billing** resource (not a
Monitoring policy), so they're applied with a different command — and because
they touch billing, **a human must run this** (it's not wired into deploy).

```bash
# Look up the billing account id.
gcloud billing accounts list

# Monthly budget with alert thresholds at 50/90/100% of forecasted spend.
# --all-updates-rule-* routes threshold breaches to a Pub/Sub topic or email.
gcloud billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="GridPulse monthly budget" \
  --budget-amount=150USD \
  --filter-projects="projects/nextera-portfolio" \
  --threshold-rule=percent=0.5 \
  --threshold-rule=percent=0.9 \
  --threshold-rule=percent=1.0 \
  --threshold-rule=percent=1.0,basis=forecasted-spend
```

> The `forecasted-spend` rule is the *anomaly* signal — it fires when GCP
> projects you'll blow the budget by month-end, i.e. mid-flood, not after.
> Budget email/Pub/Sub notifications are configured in the Billing console
> (Budgets & alerts → the budget → *Manage notifications*).

## Uptime check — public `/health` degraded/down alert (#253)

An uptime check hitting **public** `/health` with a content matcher on
`"healthy"` fires when the service is down *or* shallow-degraded (Redis down or
scoring stale — both surface in the public `{"status": ...}` liveness body).
Note: `?deep=1` (the forecast-payload probe) is deliberately gated behind the
`METRICS_ALLOWED_IPS` allowlist (#253), so the external prober uses shallow
`/health` — its `last_scored` check already degrades when forecasts go stale,
which is the signal the 2026-05-29 outage needed.

```bash
gcloud monitoring uptime create gridpulse-health \
  --resource-type=uptime-url \
  --resource-labels=host=gridpulse.kristenmartino.ai,project_id=nextera-portfolio \
  --protocol=https --path=/health --port=443 \
  --matcher-content='healthy' \
  --matcher-type=contains-string \
  --period=5 --timeout=10
```

> Match the bare word `healthy`, **not** `"status": "healthy"`: Flask's
> production `jsonify` emits compact JSON (`{"status":"healthy"}`, no space after
> the colon), so a spaced matcher would never match and the check would report
> the healthy service as permanently down. The status vocabulary is only
> `healthy` | `degraded`, so `healthy` is unambiguous and whitespace/key-order
> robust.

Then create an alert policy on `monitoring.googleapis.com/uptime_check/check_passed`
(`check_passed=false`) for that check, bound to the notification channel — the
console's "Create alert from uptime check" wizard is the least error-prone way.

## Follow-ups (not yet implemented)

- **Cloud Scheduler error alert** — fiddlier (no clean error metric; alert
  on `cloudscheduler.googleapis.com/job/attempt_count` filtered to
  non-2xx `response_code`). Would catch a scheduler-side miss like the
  2026-05-21 503 even when no execution is created. Tracked in #148.
- **Cloud Armor / edge rate limiting** — the app-layer per-IP limiter (#253)
  caps a single source, but a *distributed* flood can still pin instances.
  Cloud Armor (or an API gateway with quotas) is the edge-level defense; the
  `max-instances` alert + budget are the backstop until then.

> **Deep-`/health` degraded alert — done (#253):** delivered as the public
> uptime check above (adapted to shallow `/health` because `?deep=1` is now
> allowlist-gated; the shallow `last_scored` check covers the stale-forecast
> case the 2026-05-29 outage exhibited).
