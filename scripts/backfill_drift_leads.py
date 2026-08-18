"""One-shot repair of `lead_hours` erased by the #542 defect.

`models.drift.regrade_records` dropped `lead_hours` on every re-grade, and
`filter_by_lead` keeps unknown-lead records by design, so multi-hour-ahead
observations progressively re-entered a window labelled "1-hour-ahead". The
code fix stops the bleeding but **cannot repair history**: the forecast payload
that would prove a blanked record's lead is overwritten the next tick.

It is recoverable from one place. `jobs.phases.write_drift_metrics` has logged
`drift_updated` with `region`, `new_record_ts` and `lead_hours` on every tick
since 2026-08-05 (#407), and lead is a property of the (region, target hour)
pair rather than of the model — which is exactly the map that was erased.

## Why this is narrow on purpose

Production self-heals: the 7-day headline recovers in 7 days and the 30-day
figure in 30, as blanked records age out. This runs only for the BAs where
waiting was judged too expensive — measured 2026-08-18T07:06Z, those moving
more than 1.0 sMAPE pt on any (model, window), or crossing the `n_7d >= 24`
visibility gate:

    LDWP  12.085 -> 8.231   n_7d 25 -> 15   (crosses the gate; arima -11.132)
    IID   16.148 -> 13.643  n_7d 145 -> 135
    AZPS   9.779 -> 11.427  n_7d 41 -> 31   (worse, and correct)
    PSCO   9.639 -> 10.766  n_7d 154 -> 120 (worse, and correct)
    LGEE   3.117 -> 2.435   n_7d 135 -> 80  (arima -3.308)
    SPA   24.347 -> 24.630  n_7d 119 -> 82

The other 45 BAs move by less than 0.1 pt and are left to heal on their own.

## Safety properties

* **Additive only.** An existing `l` is never overwritten and no other field is
  touched. A record whose lead the log cannot supply keeps saying unknown.
* **Refuses to write on a failed read.** Uses `_read_window_strict` (#313):
  a nil-read during an outage must not rebuild a window from nothing.
* **Recomputes through the production function.** Stats come from
  `compute_drift_payload` with no new records and no actuals, which is what the
  hourly writer would do — not a reimplementation.
* **Dry-run unless the artifact says otherwise.** The apply switch lives in the
  GCS data artifact, not in this file, so the reviewed code is byte-identical
  between the rehearsal and the real run.
* **Idempotent.** Re-running adds nothing once every recoverable lead is set.
* **Must run AFTER the #542 fix is deployed.** Against the old image the next
  hourly tick re-blanks every lead this writes.

## Running it

`scripts/` is in `.dockerignore` and is NOT in the job image, so `-m
scripts.backfill_drift_leads` fails with ModuleNotFoundError. Pass the body
inline instead, using gcloud's custom-delimiter form so the source may contain
commas:

    SRC=$(cat scripts/backfill_drift_leads.py)
    gcloud run jobs execute gridpulse-scoring-job --region us-east1 \
      --args="^|^-c|$SRC"

Reading the file into the argument keeps what runs identical to what is
reviewed here.
"""

import json

from config import GCS_BUCKET_NAME
from data.gcs_store import _get_client
from data.redis_client import redis_key, redis_set
from jobs.phases import DRIFT_REDIS_TTL, _read_window_strict
from models.drift import _normalize_ts, compute_drift_payload

ARTIFACT = "cache/adhoc/drift_lead_backfill_20260818.json"

blob = _get_client().bucket(GCS_BUCKET_NAME).blob(ARTIFACT)
spec = json.loads(blob.download_as_text())
apply_writes = bool(spec.get("apply"))
leads = spec["leads"]

print("BACKFILL_START apply=" + str(apply_writes) + " regions=" + str(sorted(leads)))

for region in sorted(leads):
    by_ts = {_normalize_ts(k): int(v) for k, v in leads[region].items()}
    try:
        payload = _read_window_strict(redis_key("drift:" + region))
    except Exception as exc:
        print("BACKFILL_SKIP " + region + " read_failed=" + str(exc))
        continue
    if not isinstance(payload, dict) or not payload.get("models"):
        print("BACKFILL_SKIP " + region + " no_window")
        continue

    filled = 0
    still_unknown = 0
    for block in payload["models"].values():
        for row in block.get("records") or []:
            if "l" in row:
                continue
            lead = by_ts.get(_normalize_ts(row.get("ts") or ""))
            if lead is None:
                still_unknown += 1
                continue
            row["l"] = lead
            filled += 1

    before = payload["models"].get("ensemble") or {}
    # No new records and no actuals: merge/trim is a no-op and the rolling
    # stats are recomputed off the enriched records, exactly as the hourly
    # writer would once the next tick lands.
    rebuilt = compute_drift_payload(region, payload, {})
    rebuilt.pop("_regrade_stats", None)
    after = rebuilt["models"].get("ensemble") or {}

    print(
        "BACKFILL_REGION "
        + region
        + " filled="
        + str(filled)
        + " still_unknown="
        + str(still_unknown)
        + " ens_n7d="
        + str(before.get("n_7d"))
        + "->"
        + str(after.get("n_7d"))
        + " ens_unk7d="
        + str(before.get("n_lead_unknown_7d"))
        + "->"
        + str(after.get("n_lead_unknown_7d"))
        + " ens_excl7d="
        + str(before.get("n_lead_excluded_7d"))
        + "->"
        + str(after.get("n_lead_excluded_7d"))
        + " ens_smape7d="
        + str(before.get("rolling_smape_7d"))
        + "->"
        + str(after.get("rolling_smape_7d"))
        + " ens_mape7d="
        + str(before.get("rolling_mape_7d"))
        + "->"
        + str(after.get("rolling_mape_7d"))
    )

    if apply_writes and filled:
        redis_set(redis_key("drift:" + region), rebuilt, ttl=DRIFT_REDIS_TTL)
        print("BACKFILL_WROTE " + region)

print("BACKFILL_DONE apply=" + str(apply_writes))
