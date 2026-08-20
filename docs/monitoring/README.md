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
| `benchmark_scoreability_drop_alert.json` | **Job tier (2026-08-18, #535).** Log-based (`jsonPayload.event="benchmark_scoreability_drop"`) — fires when the public `/benchmark` scorecard scores fewer than `BENCHMARK_MIN_SCOREABLE` BAs. It published **25 of 51** for ~3 weeks with five of seven large ISOs missing from its fleet medians while the job succeeded and exited 0 — the headline changed the population it described and nothing said so. Counts the *failing* direction: a threshold-below on the existing `benchmark_fleet_written` count would go quiet exactly when the phase stopped emitting. |
| `benchmark_df_gap_at_risk_alert.json` | **Job tier (2026-08-20, #587).** Log-based (`jsonPayload.event="benchmark_df_gap_at_risk"`) — the early warning on the gate that actually decides the population: a still-scored BA's **longest DF gap** entered the band below `MAX_DF_GAP_HOURS` (`BENCHMARK_DF_GAP_WARN_HOURS`, 120h against 168h). Warning at the gate would arrive too late, since a BA that has already fallen out is a page that is already wrong. **Replaces `benchmark_coverage_at_risk`**, retired 2026-08-20: it watched a rate while the gate measures a duration, so "warn before exclude" was an arithmetic coincidence of two unrelated constants and the window length — untestable, because the two numbers were not comparable. It had also become noise, firing on every tick after the #580 deploy for a healthy TEC at 0.822. Both are now hours of gap, so `warn < gate` is the whole proof and a unit test asserts it. Lower urgency than the drop alert — a lead, not an incident. Applied 2026-08-20 as `alertPolicies/13946097116379396982`. |
| `web_service_5xx_alert.json` | **Web tier (#253).** Fires when the `gridpulse` service returns sustained 5xx (`run.googleapis.com/request_count{response_code_class="5xx"}` summed > 25 / 5 min). The request-path equivalent of the job-failure alert. |
| `web_service_max_instances_alert.json` | **Web tier (#253).** Fires when a **single revision** sits at its `max-instances` ceiling (4) for 15 min — the cost ceiling *and* the traffic-flood signal on the public surface. Per-revision and mean-aligned since 2026-08-18; the original aggregation counted rollout overlap (below). |
| `web_service_uptime_alert.json` | **Web tier (#253).** Fires when the public `/health` uptime check fails from >1 probe location over 10 min (service down or shallow-degraded). Filter is check-id-specific — see the note in the file. |

Both web-tier policies apply the same way as the job policies (see "Apply /
re-apply" below). The **uptime check** and the **billing budget** are separate
GCP resource types — their `gcloud` recipes are in the two sections just below.

### Why the #535 alerts count the failing direction

Both benchmark policies are log-based and count *upward* as things get worse.
The obvious alternative is a metric threshold on `benchmark_fleet_written`,
which has carried the scoreable count since E0 — and it is the same trap
`backtest_recompute_alert.json` documents. It would have to be a
threshold-BELOW condition, so a benchmark phase that stops emitting leaves the
counter with no data, the condition never evaluates, and the alert goes quiet
exactly when it should fire. A phase that stops running is a different failure
and is covered by `cloud_run_job_failure_alert` and `scoring_partial_failure`.

(This lived in the `benchmark_coverage_at_risk` runbook until 2026-08-18. It is
design rationale rather than something an on-call reader acts on, and moving it
here is what brought that runbook back under the 4000-character documentation
cap — see the section on that cap below.)

### Why the gap gate replaced the coverage rate (#549, #587)

`df_coverage` — the share of window hours carrying a published day-ahead
forecast — gated benchmark scoreability until [#549](https://github.com/kristenmartino/gridpulse/issues/549).
It no longer gates anything; `MAX_DF_GAP_HOURS` does, and since
[#587](https://github.com/kristenmartino/gridpulse/issues/587) it compares the
*longest* no-DF stretch in the window rather than only the trailing one.

The reason is that **a publication rate cannot tell two different failures
apart.** A BA that half-publishes all month and a BA that published completely
and then stopped dead can report the same coverage figure, and only the second
one is unscoreable — its remaining hours no longer describe the same period as
every other row on the page.

Measured across all 51 BAs on 2026-08-18, **no BA is diffusely sparse**, which
is the shape the rate gate implicitly assumed: SPP's absence is ONE contiguous
341-hour block (feed stopped 2026-08-04T06Z), and TEC's is six whole-day blocks
with a live feed. The pre-#549 exclusion text asserted the diffuse shape for
both. A worked example of the integer-comparison step in that runbook: TEC read
576 DF rows both ways on 2026-08-18, settling "is the absence theirs or ours?"
in a single call.

(Moved out of the `benchmark_coverage_at_risk` runbook on 2026-08-20 for the
same reason as the section above — it is rationale, not an on-call action, and
the runbook was 187 characters from the cap that silently disarms the alert.
That runbook and its policy were retired hours later by #587, so this section
is now the only surviving copy of the rationale — which is the argument for
having moved it here rather than leaving it in a file that could be deleted.)

### The max-instances aggregation counted rollovers, not instances (2026-08-18)

`web_service_max_instances_alert.json` fired at 11:54Z claiming the service was
pinned at its 4-instance ceiling. It was not. Traffic was flat at **0.07 req/s**
all day and no revision exceeded **2** instances. What had changed was deploy
cadence: 20 merges to `main` in three hours produced **19 revisions**, one every
~5 minutes ([#581](https://github.com/kristenmartino/gridpulse/issues/581)).

The condition summed `ALIGN_MAX` across two dimensions its `groupByFields` did
not preserve — the `state` metric label **and** `revision_name`. Per-series
alignment runs *before* the cross-series reduction, so a single instance that
was active early in a 5-minute window and idle later contributed 1 to each
series' max, and every draining revision added its own on top.

**The tell was in the number itself.** The policy documented its statistic as
"the 4-instance ceiling", and the statistic reached **7**. A value that exceeds
the ceiling it claims to measure is not a severe reading of the right thing; it
is the wrong thing. Nothing else about the firing distinguished it from a real
flood, which is why the old runbook's second step — grep the logs for one IP —
sent the reader somewhere there was nothing to find.

Two changes, both load-bearing:

* **`ALIGN_MEAN`**, because `instance_count` is a GAUGE. Sum-of-means equals
  mean-of-sums, so the active/idle double count disappears. The cost is that a
  one-minute burst to 4 no longer fires — correct for a policy whose stated
  subject is *sustained* spend.
* **`revision_name` in `groupByFields`**, because `--max-instances` is a
  per-revision bound in Knative. This half is not optional: mean-aligned but
  still summed across revisions peaks at **3.2** on the same window, which is
  three consecutive points over the threshold and would have fired anyway.

Re-measured over 09:00–12:00Z with the new aggregation, the per-revision peak is
**2.0** — the alert would not have fired. The state label stays out of the
grouping, which preserves the original design note's correct half: a ceiling can
span active and idle, so both must be summed.

### Not a Cloud Monitoring policy: the deploy-divergence check

`.github/workflows/deploy-divergence.yml` runs
[`scripts/check_deploy_divergence.py`](../../scripts/check_deploy_divergence.py)
hourly and asks whether the three Cloud Run surfaces are running main's newest
**CI-validated** commit. It lives in GitHub Actions rather than here for a
simple reason: **Cloud Monitoring cannot see main's tip.** Half the comparison
is a git fact, so no metric or log filter can express the condition — the same
shape as the frequency signal below, resolved the same way, in code.

What it catches that nothing else does: **a deploy that was skipped is
indistinguishable from one that ran**, because `deploy-prod.yml`'s staleness
guard turns a superseded run into a no-op and the run still reports success.
The guard is correct to skip — it is what stopped an older commit shipping over
a newer one on 2026-08-05 — but its notice says *"a newer deploy covers it"*,
and that is an assumption about the future which fails three ways:

1. **The next commit is red.** The skipped commit never gets another turn,
   because the deploy workflow is gated on CI success. Production sits on the
   pre-merge image indefinitely with every workflow green.
2. **Merges outrun the pipeline.** Observed live 2026-08-11: four commits landed
   in 14 minutes, and each deploy found the tip had moved before its guard ran.
   Production stayed two commits behind for ~25 minutes and nothing reported it.
3. **A deploy half-lands.** 2026-08-04 (#418): `gcloud run jobs deploy` rejected
   a flag `gcloud run deploy` accepted, so the service advanced while **both
   jobs froze on a 12-hour-old image**. This check compares each surface
   separately and names that case ("partially deployed") distinctly, because a
   half-landed deploy is a different investigation from one that never started.

Exit codes are the alert — a non-zero exit fails the workflow and notifies:
`0` converged or still in flight (45-minute grace, since a mismatch is the
normal state for minutes after every merge), `1` diverged, `2` the check could
not reach a verdict. **`2` is deliberately non-zero.** A check that cannot run
is not protecting anything, and folding that into a pass is the exact failure
mode called out under "Log-based policies were inert" below.

Run it by hand any time — it needs `gh` and `gcloud` auth and touches nothing:

```bash
python3 scripts/check_deploy_divergence.py
```

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
| benchmark scorecard below the scoreable floor | `benchmark_scoreability_drop_alert.json` | `alertPolicies/1095567904752750375` |
| a benchmark BA approaching the DF-gap gate (#587) | `benchmark_df_gap_at_risk_alert.json` | `alertPolicies/13946097116379396982` |

**Applied 2026-08-20 (#587).** `alertPolicies/13946097116379396982`, enabled,
bound to the same notification channel as the rest of the job tier. Verified by
reading the policy back from the API rather than trusting the create call: the
filter matches `jsonPayload.event="benchmark_df_gap_at_risk"`, the runbook
round-tripped at 3,523 characters, and the rate limit and auto-close match the
committed file.

The policy it replaced — `15888827698887105322`, the `benchmark_coverage_at_risk`
rate warning — was **deleted** in the same pass. Leaving it would have been the
worse of the two options: its event is no longer emitted by any code path, and a
log-match policy whose event has been renamed away is indistinguishable from a
healthy one that simply has nothing to report. That is the exact failure mode
this directory is written against, so it was removed rather than disabled.
| Uptime check config — public `/health` | — | `uptimeCheckConfigs/gridpulse-health-162OIAwsIpE` |
| Monthly budget — $150 (billing acct `01D68B-6BF1D9-B54F3B`) | — | `budgets/3363cac4-5a23-46ea-a51f-ddbbadeca827` |

All eleven alert policies + the uptime check + the budget are live and bound
to the email channel — verified against the Monitoring API, not against this
table (`tests/unit/test_monitoring_policies_applied.py` is what keeps the two
honest). The budget also emails the billing-account admins by default.

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

## A policy's documentation is capped at 4000 characters (2026-08-18)

**`documentation.content` must be ≤ 4000 characters, and on a log-match policy
the cap is enforced by an error that names the wrong cause and disarms the
alert.**

`policies create` rejects an over-length body properly, saying what is wrong:

```
ERROR: INVALID_ARGUMENT: `description` must not be more than 4000 characters
```

`PATCH …/alertPolicies/<id>?updateMask=documentation` on a policy whose
condition is a `conditionMatchedLog` does not. The same over-length body gets:

```
HTTP 200
"validity": {"code": 13, "message": "Recompilation of log match condition failed during update."}
"enabled": false
```

Three things make the PATCH form worse than a plain rejection:

1. **It returns HTTP 200.** A caller that checks the status code sees success.
   The failure is only visible in the `validity` field of the response body,
   and the damage is only visible in `enabled`.
2. **It disarms a working alert.** `enabled` flips to false as a side effect of
   a documentation edit. Nothing else about the policy changes — the
   documentation is not updated either, so the edit silently accomplishes the
   opposite of its intent.
3. **It blames the log match condition**, which is untouched and fine. Nothing
   in the response mentions length. That misdirection is what made this cost a
   day: the obvious reading is that log-match policies are documentation-
   immutable, and that reading is wrong.

**A documentation edit under the cap applies cleanly**, repeatedly, on a
log-match policy. Keep runbooks well clear of 4000 —
`tests/unit/test_monitoring_policies_applied.py` fails the build at the cap so
this is caught at commit time rather than at apply time. When a runbook needs
to grow past it, move the *design rationale* into this README (that is why the
"why the #535 alerts count the failing direction" section above exists) and
leave the console copy operational.

**If you tripped this, recovery is a second, valid PATCH.** Sending
documentation under the cap clears `validity` and restores `enabled: true` in
one call — no separate re-enable is needed, though the explicit form still
works:

```bash
curl -s -X PATCH -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" -d '{"enabled": true}' \
  "https://monitoring.googleapis.com/v3/projects/nextera-portfolio/alertPolicies/<id>?updateMask=enabled"
```

Then verify with a read, not with the PATCH response:

```bash
curl -s -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  "https://monitoring.googleapis.com/v3/projects/nextera-portfolio/alertPolicies?fields=alertPolicies(displayName,enabled,validity)"
```

### The first diagnosis was wrong, and how it was settled (2026-08-18)

This section previously stated that log-match documentation **could not be
edited in place at all**, and that the only way to correct a runbook was
delete-and-recreate with a new policy id. That was wrong. The evidence for it
was real but confounded: every failing attempt happened to carry the same
4035-character body, so "it fails even with byte-identical text" and "it fails
because the text is too long" predicted the same outcome.

Isolated on a throwaway policy with a never-emitted filter, patching
documentation only:

| Test | doc chars | Result |
|---|---|---|
| A — first PATCH | 39 | 200, `validity` absent, `enabled: true` |
| B — second PATCH (the "subsequent update" control) | 40 | 200, `validity` absent, `enabled: true` |
| C — over-length PATCH | **4035** | 200, `validity code 13`, `enabled: false`, doc unchanged |
| D — PATCH after the failure | 44 | 200, `validity` absent, `enabled: true` |

**B refutes** "only the first update on a fresh policy succeeds." **D refutes**
"once a policy has taken one failed update it stays in the invalid state."
Only C fails, and length is the only variable it changes.

The lesson this file already teaches is **assert the enforcement, not the
declaration**. The correction adds a second: *a reproduction is not a
diagnosis.* Re-running the failure proved it was reproducible, not that the
stated cause was the operative one — the control that would have separated
them (vary the length, hold everything else) was the one never run.

## What the guard test does and does not cover

`tests/unit/test_monitoring_policies_applied.py` runs on every PR and is the
only automatic check on this directory. Being precise about its edge matters,
because a guard that looks broader than it is buys false confidence — the
failure mode this whole directory exists to prevent.

**It proves, from local files only:**

- every committed policy is either in the applied table with an
  `alertPolicies/<id>`-shaped id, or in `_KNOWN_UNAPPLIED` with a reason
- log-based policies carry a `notificationRateLimit` (GCP rejects them without)
- log-based filters key on `jsonPayload.event`, not `textPayload`
- **every filtered event name is one the source can actually emit** — added
  2026-08-18, covering both idioms: the literal `log.warning("name", ...)` and
  the `{"event": "name"}` dict that `jobs/scoring_job.py` later emits via
  `log.error(alert.pop("event"), **alert)`

That last one closes a rename: `benchmark_scoreability_drop` is a string in two
places nothing links — a filter in this directory and a call in `models/`.
Renaming the emitter is an ordinary refactor that no test, type or import
would object to, and it would silently disarm the alarm. It is the mirror of
the #267 failure in the module docstring, where the event was emitted and the
policy was never applied.

**It proves nothing about GCP.** Not that a policy exists, is enabled, routes
to a live channel, or still matches the committed JSON — the id check is a
*regex*, so `alertPolicies/9999999999` passes. Nor does it prove the emitting
branch is reachable, or that the running image contains it: on 2026-08-18 both
`#535` policies were correctly applied and enabled while the deployed job image
predated their emitter, so the alarm was armed and could not fire.

Everything in that paragraph except the last sentence is now covered hourly by
`scripts/check_monitoring_divergence.py` — see below. The unit test's edge is
unchanged; it just is not the only thing looking any more.

### The GCP half — built 2026-08-18 (was "deliberately not built")

This section previously said closing the GCP half was **deliberately not
built**, because *"drift has not been observed here, and the manual check below
covers it"*, with revisit triggers: someone edits policies in the console, a
second person gets project access, or a wrong id reaches `main`.

**The load-bearing clause was false when it was written.** Drift had been
observed — twice, and neither instance tripped a listed trigger:

| | committed | applied | how long | trigger fired |
|---|---|---|---|---|
| `benchmark_coverage_at_risk` (#553) | corrected runbook | pre-fix runbook | 4 days | none |
| `scoring_runtime_creep` (#389 rewrite) | 3555-char triage | 1153-char pre-#389 copy | **14 days** | none |

Both kept the correct id throughout, which is why the guard test stayed green
and why every revisit trigger missed: they were all about *identity* or *who
has access*, and the failure was about *content* and *state*. The second one
was found by the check built to replace this paragraph, on its first run — it
had been serving a runbook without the partial-degradation triage since
2026-07-08, for the incident class that killed two scoring ticks.

So it is built, in the shape this file already identified:
`scripts/check_monitoring_divergence.py`, as a step in
`.github/workflows/deploy-divergence.yml` (hourly, WIF already wired). It
needed `roles/monitoring.viewer` on `github-actions-deploy@` — granted
2026-08-18, weaker than the `run.admin` it already held.

**It fails when**, per policy in the applied table: the applied
`documentation.content` differs from the committed file (printing a diff, so
the direction is visible), `enabled` is not true, a `validity` field is
present, `notificationChannels` is empty, the id is absent from GCP, or a
policy exists in GCP with no row in the table. **It warns** when a committed
runbook is within 200 characters of the 4000 cap — a warning, not a failure,
because 3800 is invented and only 4000 is enforced; the standing rule here is
to assert the enforcement, not the declaration.

It never mutates, and it reports by reading. That is not fastidiousness: a
Monitoring PATCH returns HTTP 200 on failure, so a write's own response is
not evidence the write landed. Run it by hand the same way CI does:

```bash
python3 scripts/check_monitoring_divergence.py     # exit 0 ok, 1 diverged, 2 could not tell
```

Use the GA `gcloud monitoring policies` group, not `gcloud alpha monitoring`.
One `list --format=json` returns `documentation`, `enabled`, `validity` and
`notificationChannels` for every policy, and the alpha component is absent from
a fresh Cloud SDK — it prompts interactively to install, which is fatal in CI
and is what blocked checking this by hand on 2026-08-18.

```bash
# The one-liner this replaced, still handy for a quick eyeball.
gcloud monitoring policies list --project=nextera-portfolio \
  --format="value(name,displayName,enabled)"
```

**Still not covered:** applying is still manual. This makes a divergence
impossible to miss; it does not close one. Applying from CI is a larger change
with its own failure modes — the same reasoning `deploy-divergence.yml` records
for detecting a skipped deploy rather than making the deploy guard smarter.

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
