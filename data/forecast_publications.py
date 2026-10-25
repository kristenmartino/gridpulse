"""Immutable published-series evidence, captured by the scoring job after Redis writes.

The pilot covers FPL/PJM/LGEE. GCS holds immutable issues; Redis is a bounded
30-day read index/cache. UI requests never create issues or fetch upstream data.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime, timedelta

import structlog

from config import FORECAST_PUBLICATION_REGIONS
from data.forecast_evidence import baseline_selection, forecast_timing, utc_timestamp
from data.forecast_identity import resolve_forecast_identity

log = structlog.get_logger()
PILOT_REGIONS = FORECAST_PUBLICATION_REGIONS
INDEX_DAYS = 30
MAX_ISSUES = 24 * INDEX_DAYS
TTL = (INDEX_DAYS + 1) * 86400


def _digest(record: dict) -> str:
    core = {k: v for k, v in record.items() if k not in ("publication_id", "observed_published_at")}
    return hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def prepare_publication(payload: dict) -> dict | None:
    """Build evidence from the exact published column, never a preferred model."""
    if payload.get("region") not in PILOT_REGIONS:
        return None
    identity = resolve_forecast_identity(payload)
    rows = payload.get("forecasts") or []
    if identity is None:
        return None
    guard_map = payload.get("horizon_guard") or {}
    guard = guard_map.get(identity.metric_key) or {}
    max_hours = guard.get("max_ok_horizon")
    if isinstance(max_hours, (int, float)) and math.isfinite(max_hours):
        rows = rows[: max(0, int(max_hours))]
    timing = forecast_timing(payload, len(rows))
    if not timing["generated_at"]:
        return None
    points = []
    for row in rows:
        ts = utc_timestamp(row.get("timestamp"))
        value = row.get(identity.column)
        if not ts or isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not math.isfinite(value):
            return None
        points.append({"timestamp": ts, "demand_mw": float(value)})
    if not points or len({p["timestamp"] for p in points}) != len(points):
        return None
    dates = [datetime.fromisoformat(p["timestamp"]) for p in points]
    if any(b - a != timedelta(hours=1) for a, b in zip(dates, dates[1:], strict=False)):
        return None
    timing.pop("publication_id")
    record = {
        "schema_version": 1,
        "region": payload["region"],
        "producer": "GridPulse",
        "source_provenance": "scheduled_publication",
        "series_source": identity.metric_key,
        "series_label": identity.label,
        "timing": timing,
        "baseline_selection": baseline_selection(payload),
        "forecast": points,
        "notes": [
            "Captured only after the scheduled job successfully writes the published forecast.",
            "Long-horizon inputs may use climatology; this is not a weather-only forecast.",
            "Actual-demand observations and scores are not part of this immutable forecast issue.",
        ],
    }
    record["publication_id"] = _digest(record)
    return record


def publication_summary(record: dict) -> dict:
    return {
        "publication_id": record["publication_id"],
        "generated_at": record["timing"]["generated_at"],
        "observed_published_at": record["observed_published_at"],
        "series_source": record["series_source"],
        "series_label": record["series_label"],
        "forecast_start_at": record["timing"]["forecast_start_at"],
        "target_end_at": record["timing"]["target_end_at"],
        "target_hours": len(record["forecast"]),
    }


def _cache_receipt(receipt: dict) -> None:
    """Watch the index so concurrent successful publications cannot lose entries."""
    from redis.exceptions import WatchError

    from data.redis_client import _get_redis, persist, redis_key

    region = receipt["region"]
    persist(redis_key(f"publication:{region}:{receipt['publication_id']}"), receipt, ttl=TTL)
    client = _get_redis()
    if client is None:
        raise RuntimeError("publication cache unavailable")
    index_key = redis_key(f"publication_index:{region}")
    for _ in range(5):
        with client.pipeline() as pipe:
            try:
                pipe.watch(index_key)
                raw = pipe.get(index_key)
                existing = json.loads(raw) if raw else []
                if not isinstance(existing, list):
                    raise ValueError("publication index is malformed")
                cutoff = (datetime.now(UTC) - timedelta(days=INDEX_DAYS)).isoformat()
                by_id = {
                    row["publication_id"]: row
                    for row in existing
                    if isinstance(row, dict)
                    and row.get("publication_id")
                    and row.get("generated_at", "") > cutoff
                }
                if receipt["timing"]["generated_at"] > cutoff:
                    by_id[receipt["publication_id"]] = publication_summary(receipt)
                index = sorted(by_id.values(), key=lambda r: r["generated_at"], reverse=True)[
                    :MAX_ISSUES
                ]
                pipe.multi()
                pipe.set(index_key, json.dumps(index, allow_nan=False), ex=TTL)
                pipe.execute()
                return
            except WatchError:
                continue
    raise RuntimeError("publication index changed during five update attempts")


def export_publication(record: dict) -> dict:
    """Explicit public schema; internal/future cache fields never pass through."""
    timing_fields = (
        "generated_at",
        "forecast_start_at",
        "target_end_at",
        "target_hours",
        "timezone",
        "interval_convention",
        "input_anchor_at",
        "input_anchor_was_placeholder",
        "input_anchor_conditioned",
        "operator_issued_at",
    )
    selection_fields = (
        "reason",
        "decision_recorded_at",
        "selected_since",
        "compared_series",
        "comparison_basis",
    )
    selection = record.get("baseline_selection")
    return {
        **{
            k: record.get(k)
            for k in (
                "schema_version",
                "publication_id",
                "region",
                "producer",
                "source_provenance",
                "series_source",
                "series_label",
                "observed_published_at",
            )
        },
        "timing": {k: record.get("timing", {}).get(k) for k in timing_fields},
        "baseline_selection": (
            {k: selection.get(k) for k in selection_fields} if isinstance(selection, dict) else None
        ),
        "forecast": [
            {"timestamp": r.get("timestamp"), "demand_mw": r.get("demand_mw")}
            for r in record.get("forecast", [])[:720]
        ],
        "notes": [
            "Immutable GridPulse issue; observations/scores require a separate review context.",
            "Long-horizon inputs may use climatology. See /methodology.",
        ],
    }


def archive_publication(record: dict, *, observed_at: str | None = None) -> bool:
    """Best-effort durable receipt; failure never changes which forecast is served.

    Call AFTER successful publication. GCS create-only writes prevent retries
    from changing an issue/receipt. An unavailable archive stays unavailable.
    """
    from google.api_core.exceptions import PreconditionFailed

    from config import GCS_BUCKET_NAME, GCS_PATH_PREFIX
    from data.gcs_store import _get_client

    region = record["region"]
    try:
        if _digest(record) != record["publication_id"]:
            raise ValueError("publication content does not match identity")
        client = _get_client()
        if client is None or not GCS_BUCKET_NAME:
            log.warning("publication_archive_unavailable", region=region, reason="storage_disabled")
            return False
        receipt = dict(
            record,
            observed_published_at=utc_timestamp(observed_at) or datetime.now(UTC).isoformat(),
        )
        date = receipt["timing"]["generated_at"][:10]
        path = f"{GCS_PATH_PREFIX}/publications/v1/{region}/{date}/{record['publication_id']}.json"
        blob = client.bucket(GCS_BUCKET_NAME).blob(path)
        try:
            blob.upload_from_string(
                json.dumps(receipt, allow_nan=False, separators=(",", ":")),
                content_type="application/json",
                if_generation_match=0,
            )
        except PreconditionFailed:
            receipt = json.loads(blob.download_as_text())
            if (
                receipt.get("publication_id") != record["publication_id"]
                or _digest(receipt) != record["publication_id"]
            ):
                raise ValueError("existing archive content does not match identity") from None
        _cache_receipt(receipt)
        log.info(
            "publication_archived",
            region=region,
            publication_id=record["publication_id"],
            bytes=blob.size,
        )
        return True
    except Exception as exc:
        log.warning("publication_archive_failed", region=region, error=str(exc))
        return False
