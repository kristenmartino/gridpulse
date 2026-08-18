#!/usr/bin/env python3
"""Preserve the horizon-drift buffer to GCS — data insurance, not a study.

``gridpulse:drift_horizon:{region}`` (per-(model, horizon) prediction/actual
records, ~720 hourly entries deep — roughly four independent weekly windows)
sits behind a 24h TTL (``jobs/phases.py:REDIS_TTL``) on a Memorystore instance
with ``persistenceConfig: DISABLED``. A restart or a >24h scoring outage
zeroes a month of accrued history that cost a month to accrue.

Redis is only reachable from inside the VPC, so this is meant to run as a
one-off, read-only override of the deployed scoring job's container args —
not on a schedule, not wired into ``jobs/__main__.py``:

    gcloud run jobs execute gridpulse-scoring-job --region=us-east1 \\
        --args="-m,scripts.dump_drift_horizon" --wait

It writes the raw payload (including per-hour ``records``) straight to GCS —
NOT Cloud Logging, and NOT the public API's allow-listed export shape
(``api.py:_EXPORTED_HORIZON_DRIFT_FIELDS`` deliberately withholds raw
records per the #250 review; this script must not change or bypass that for
any live surface, it only writes a private backup object). Only per-region
success/failure counts are logged.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog  # noqa: E402

log = structlog.get_logger()

#: Below this, something is wrong (VPC/Redis reachability, wrong project) —
#: a one-off diagnostic run should fail loud rather than silently ship a
#: near-empty backup.
_MIN_REGIONS_OK = 45


def main() -> int:
    from config import GCS_BUCKET_NAME, GCS_ENABLED, GCS_PATH_PREFIX, REGION_COORDINATES
    from data.redis_client import redis_get, redis_key

    if not GCS_ENABLED or not GCS_BUCKET_NAME:
        log.error("dump_drift_horizon_gcs_disabled")
        return 1

    payloads: dict[str, dict] = {}
    missing: list[str] = []
    for region in sorted(REGION_COORDINATES):
        payload = redis_get(redis_key(f"drift_horizon:{region}"))
        if isinstance(payload, dict) and payload:
            payloads[region] = payload
        else:
            missing.append(region)

    log.info(
        "dump_drift_horizon_read",
        n_regions_ok=len(payloads),
        n_regions_missing=len(missing),
        missing=missing,
    )
    if len(payloads) < _MIN_REGIONS_OK:
        log.error(
            "dump_drift_horizon_too_few_regions",
            n_regions_ok=len(payloads),
            threshold=_MIN_REGIONS_OK,
        )
        return 1

    from google.cloud.storage import Client as StorageClient

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    blob_path = f"{GCS_PATH_PREFIX}/adhoc/drift_horizon_dump_{ts}.json"
    body = json.dumps(
        {"dumped_at": datetime.now(UTC).isoformat(), "regions": payloads}, default=str
    )

    client = StorageClient()
    bucket = client.bucket(GCS_BUCKET_NAME)
    bucket.blob(blob_path).upload_from_string(body, content_type="application/json")

    log.info(
        "dump_drift_horizon_written",
        bucket=GCS_BUCKET_NAME,
        path=blob_path,
        n_regions=len(payloads),
        bytes=len(body),
    )
    print(f"gs://{GCS_BUCKET_NAME}/{blob_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
