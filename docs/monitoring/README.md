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
| `redis_write_failures_alert.json` | **Job tier (2026-08-05).** Log-based (`jsonPayload.event="redis_write_failures"`) — fires when a run dropped one or more *fail-soft* Redis writes. Secondary payloads only (generation, interchange, drift, benchmark, backtest, weather-correlation, alerts, meta); the critical ones use fail-loud `persist()`. Needed because all 15 `redis_set` call sites ignore the returned `False` by design, so a dropped write changes nothing observable — no phase fails, the region still counts as scored, the run still exits 0. |
| `scoring_deadline_shed_alert.json` | **Job tier (2026-08-04).** Log-based (`jsonPayload.event="scoring_deadline_shed"`) — fires when a run hits `SCORING_SOFT_DEADLINE_FRACTION` of its task timeout and stops starting new BAs. This is the guard *working*: the run completes, writes an honest `last_scored` (`deadline_hit`, `regions_deadline_skipped`) and exits 0, instead of being SIGKILLed having recorded nothing — which is what happened to two ticks on 2026-08-04 after they had already scored ~49/51. Still means runtime is at the ceiling and BAs are going unscored. |
| `backtest_recompute_alert.json` | **Job tier (2026-08-10).** Metric-threshold on a **logs-based counter** (`jsonPayload.event="job_backtest_recomputed"`) — fires when backtests recompute on more than one day in three, i.e. the `BACKTEST_REFRESH_DAYS` gate stopped holding. Backtests are **56.9% of the training job** (12,886s measured 2026-08-08); a silent regression here roughly triples the training bill while the job still succeeds and exits 0. Detection is in code (`check_backtest_recompute_cadence`) because Cloud Monitoring caps alignment windows at 25h — see the section below. |
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

## A frequency signal cannot be a Cloud Monitoring condition

`backtest_recompute_alert.json` originally used a metric threshold over a 72h
window, because the thing worth alerting on is a **frequency**: one backtest
recompute per week is correct, two in three days is a regression, and
`conditionMatchedLog` fires on both. Cloud Monitoring rejects that outright:

```
Alignment periods longer than 25h are not supported.
```

A `<=25h` window does not work either. The training job runs once daily, so
consecutive recompute days land in **adjacent** windows and never the same one —
any 24h window sees at most one day's worth (153 events) whether backtests run
weekly or every single run.

So the comparison lives in code, where the state is:
`jobs.phases.check_backtest_recompute_cadence` reads a
`gridpulse:meta:last_backtest_recompute` marker, runs **once per run from the
epilogue on task 0 only**, and emits `backtest_recompute_unexpected` only when
the cadence is genuinely wrong. The policy is then an ordinary
`conditionMatchedLog`, the same shape as every other one here — **no logs-based
metric is required.**

An earlier version of this section described creating a `backtest_recomputes`
logs-based counter. That metric was created while applying the first version
of the policy and has since been **deleted** — nothing referenced it. The
`job_backtest_recomputed` log it counted is still emitted and is the runbook's
root-cause diagnostic (`previous_computed_at` separates "payload missing" from
"gate rejected a valid payload"); it simply does not need a metric in front of
it.

**Applied 2026-08-10:** `alertPolicies/14801909132378911177`, enabled, bound to
the email channel — verified by reading the policy back rather than trusting
the create output. The guard is live from the **second** recompute onward: the
first has no prior marker to compare against.

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

The `File` column is **load-bearing, not decoration**:
`tests/unit/test_monitoring_policies_applied.py` parses this table and fails CI
if a policy JSON is committed without either a live id here or an explicit
not-yet-applied declaration. That check exists because
`scoring_partial_failure` sat committed-and-inert for a week with nobody
noticing (below).

| Resource | File | Live id |
|---|---|---|
| Cloud Run Job failed execution | `cloud_run_job_failure_alert.json` | `alertPolicies/5965243952275624431` |
| scoring-job runtime creep (#171) | `scoring_runtime_creep_alert.json` | `alertPolicies/5813319064717268577` |
| web service sustained 5xx | `web_service_5xx_alert.json` | `alertPolicies/14035657251363314798` |
| web service pinned at max instances | `web_service_max_instances_alert.json` | `alertPolicies/7343953142414788448` |
| /health uptime check failing (alert) | `web_service_uptime_alert.json` | `alertPolicies/1577408926164424010` |
| scoring-job partial failure (#267) | `scoring_partial_failure_alert.json` | `alertPolicies/1942403527399204858` |
| scoring-job shed BAs at the soft deadline | `scoring_deadline_shed_alert.json` | `alertPolicies/8524477981812373740` |
| Redis fail-soft writes dropped | `redis_write_failures_alert.json` | `alertPolicies/16314898527819427981` |
| backtests recomputing sooner than the cadence | `backtest_recompute_alert.json` | `alertPolicies/14801909132378911177` |
| Uptime check config — public `/health` | — | `uptimeCheckConfigs/gridpulse-health-162OIAwsIpE` |
| Monthly budget — $150 (billing acct `01D68B-6BF1D9-B54F3B`) | — | `budgets/3363cac4-5a23-46ea-a51f-ddbbadeca827` |

Five alert policies + the uptime check + the budget are live and bound to the
email channel. The budget also emails the billing-account admins by default.

> ✅ **`scoring_partial_failure_alert.json` (#267) applied 2026-08-05** as
> `alertPolicies/1942403527399204858`, after sitting committed-and-inert since
> 2026-07-08 — `jobs/scoring_job.py` emitted the event into a void for four
> weeks. **Landing the JSON was never landing the alert**, which is why
> `tests/unit/test_monitoring_policies_applied.py` exists.
>
> It also could not have fired under a timeout-shaped incident even once
> applied: a SIGKILLed run never reaches the `log.error` that emits the event.
> The soft deadline (2026-08-04) is what makes a squeezed run complete far
> enough to report itself — so this alert and that guard only became useful
> together.

> ✅ **`redis_write_failures_alert.json` applied 2026-08-05** as
> `alertPolicies/16314898527819427981`, once `redis_write_failures` was
> confirmed present in the **deployed image** (`ee85c2b`) rather than merely on
> main. It covers *fail-soft* `redis_set` drops — secondary payloads only;
> `actuals`/`weather`/`forecast`/`vintage` use fail-loud `persist()` and already
> fail their phase (#268). Before this, such a drop left only a stdlib-logging
> warning: `textPayload`, no `jsonPayload.event`, unmatched by any policy.
>
> **`_KNOWN_UNAPPLIED` is now empty.** Every committed policy is live. If you add
> one, this table is where it has to land — the test parses it.

> ✅ **`scoring_deadline_shed_alert.json` applied 2026-08-05** as
> `alertPolicies/8524477981812373740`, once the soft-deadline code was
> confirmed live in the deployed image — not before, because an alert on an
> event nothing can emit is the same void in the other direction.

### Log-based policies were inert until 2026-07-15 — read this before adding one

> ✅ **Fixed and verified in prod 2026-07-15.** Job logs now carry
> `jsonPayload.event` (`scoring_job_complete`, `job_cli_exit`, …) and
> `textPayload` is empty — nothing falls through unparsed. `scoring_runtime_creep`
> (#171) is genuinely unblocked. `scoring_partial_failure` (#267) still cannot
> fire, but now only because its policy was never applied (above) — the JSON
> blocker is gone.


Both `conditionMatchedLog` policies filter on **`jsonPayload.event="…"`**. That
field only exists if the process emits **JSON** to stdout. `configure_logging()`
(`observability.py`) does that — but it was called **only** by `app.py` (the web
tier). **`jobs/__main__.py` never called it**, so the jobs fell back to
structlog's default `ConsoleRenderer` and every job log arrived as
`textPayload`. `jsonPayload.event` never existed, so
`scoring_runtime_creep` (#171) and `scoring_partial_failure` (#267) **matched
nothing and could never fire** — the two alerts built specifically to catch the
2026-06-01 timeout and the #267 partial failure.

Fixed by calling `configure_logging()` in `jobs/__main__.py::main()`
(`Dockerfile` already sets `DASH_DEBUG=false`, so the JSON branch engages in
Cloud Run and local runs stay human-readable). Pinned by
`tests/unit/test_jobs_json_logging.py`.

**Confirm the pipe is alive before trusting any log-based alert** — this
returned nothing at all before the fix, and returns event names now:

```bash
gcloud logging read 'resource.type="cloud_run_job" AND jsonPayload.event:*' \
  --project=nextera-portfolio --limit=5 --freshness=2h \
  --format='value(jsonPayload.event)'
```

Use the `jsonPayload.event:*` **existence** form, not `jsonPayload.event!=""`.
Both work when pasted into a plain shell, but the `!=""` form carries embedded
double quotes that get mangled the moment the command is nested inside another
quoted context (CI step, Makefile, another shell). `:*` has no inner quotes.
`--freshness` matters too: the default look-back can silently miss a recent
job run and read as "still broken."

A new log-based policy also **requires** `alertStrategy.notificationRateLimit`
(both existing ones use `3600s`); without it the policy is rejected.

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
