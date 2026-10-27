"""Atomic read/modify/write for the bounded hourly scoring ledger."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from redis.exceptions import WatchError

from data.redis_client import _get_redis, redis_key
from models.hourly_scoring import compute_hourly_scoring, export_hourly_scoring


def update_hourly_scoring(
    region: str, forecast: dict | None, actuals: dict, *, actual_exclusions: dict | None = None
) -> dict:
    """Concurrent retries cannot replace an already captured hourly sample."""
    client = _get_redis()
    if client is None:
        raise RuntimeError("Hourly scoring cache is unavailable")
    key = redis_key(f"drift_hourly:{region}")
    observed_at = datetime.now(UTC).isoformat()
    for _ in range(5):
        with client.pipeline() as pipe:
            try:
                pipe.watch(key)
                raw = pipe.get(key)
                previous = json.loads(raw) if raw is not None else None
                if previous is not None and not isinstance(previous, dict):
                    raise ValueError("Hourly scoring history is malformed")
                payload = compute_hourly_scoring(
                    region,
                    previous,
                    forecast,
                    actuals,
                    now_iso=observed_at,
                    actual_exclusions=actual_exclusions,
                )
                pipe.multi()
                # Include the 72h future horizon as well as the 7d reporting
                # life: an outage must not expire still-scoreable samples.
                pipe.set(key, json.dumps(payload, allow_nan=False), ex=11 * 86400)
                # Web requests read the small summary, never deserialize the
                # retained observation ledger just to render a metric.
                pipe.set(
                    key + ":summary",
                    json.dumps(export_hourly_scoring(payload), allow_nan=False),
                    ex=11 * 86400,
                )
                pipe.execute()
                return payload
            except WatchError:
                continue
    raise RuntimeError("Hourly scoring changed during five update attempts")
