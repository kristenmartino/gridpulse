"""Application health reporting for the ``/health`` endpoint (PR-G3 / #147).

Two depths:

* **shallow** (default ``/health``) — process-up + in-memory cache counts
  + Redis ping + last-scored freshness. All cheap, safe for frequent
  liveness polling (the Dockerfile ``HEALTHCHECK`` and Cloud Run / load
  balancer probes hit this).
* **deep** (``/health?deep=1``) — additionally reads a real forecast
  payload from Redis and validates its shape. Heavier (deserializes a
  full forecast), so it's opt-in: the post-deploy smoke check and ops
  monitoring use it, the liveness probe does not.

Status semantics:

* ``"healthy"`` — every check that matters *in this environment* passed.
* ``"degraded"`` — the process is up and serving, but a check that
  matters here failed (Redis unreachable in production, forecasts
  stale, forecast payload malformed).

The endpoint returns **HTTP 200 in both cases**. A degraded-but-serving
instance (e.g. cold Redis during warming) renders the correct "warming"
UI and must NOT be killed by the load balancer — so health is carried in
the ``status`` *field*, not the HTTP code. Consumers that care about full
functionality (deploy smoke, monitoring) read ``status``.

Redis-dependent checks are gated on ``config.REQUIRE_REDIS``: when Redis
isn't required (development / CI containers), its absence is reported as
``"skipped"`` rather than ``"degraded"``, so a Redis-less container still
reports ``healthy``. This keeps the CI Docker health check (which runs
the container with no Redis) green.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

# Scoring runs hourly (cron ``0 * * * *``). Allowing for ~12 min job
# runtime + EIA publishing lag, anything older than 90 minutes means a
# full scoring cycle was missed — that's the degraded threshold.
LAST_SCORED_STALE_SECONDS = 90 * 60


def _check_redis() -> dict[str, Any]:
    """Ping Redis. Gated on REQUIRE_REDIS — absence is only a problem when
    Redis is the required source of truth (staging / production)."""
    from config import REQUIRE_REDIS
    from data.redis_client import redis_available

    required = bool(REQUIRE_REDIS)
    try:
        up = redis_available()
    except Exception as exc:  # pragma: no cover — defensive
        up = False
        return {
            "status": "degraded" if required else "skipped",
            "required": required,
            "up": False,
            "error": str(exc),
        }

    if up:
        return {"status": "ok", "required": required, "up": True}
    # Redis down
    return {
        "status": "degraded" if required else "skipped",
        "required": required,
        "up": False,
    }


def _check_last_scored() -> dict[str, Any]:
    """Read ``gridpulse:meta:last_scored`` and report its age.

    Only meaningful when Redis is up. When Redis is required and the
    last scoring run is older than ``LAST_SCORED_STALE_SECONDS`` (or the
    marker is missing entirely), the check is degraded.
    """
    from config import REQUIRE_REDIS
    from data.redis_client import redis_available, redis_get, redis_key

    required = bool(REQUIRE_REDIS)
    try:
        if not redis_available():
            return {"status": "skipped", "reason": "redis_unavailable"}
        payload = redis_get(redis_key("meta:last_scored"))
    except Exception as exc:  # pragma: no cover — defensive
        return {"status": "degraded" if required else "skipped", "error": str(exc)}

    if not isinstance(payload, dict) or "updated_at" not in payload:
        # No scoring marker yet (fresh deploy, pre-first-run). Degraded
        # only when Redis is required — otherwise we don't expect one.
        return {
            "status": "degraded" if required else "skipped",
            "reason": "no_last_scored_marker",
        }

    try:
        updated = datetime.fromisoformat(payload["updated_at"])
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=UTC)
        age = (datetime.now(UTC) - updated).total_seconds()
    except (ValueError, TypeError) as exc:
        return {"status": "degraded" if required else "skipped", "error": str(exc)}

    stale = age > LAST_SCORED_STALE_SECONDS
    # #267: a recent-but-mostly-failed run must not read healthy. The scoring job
    # flags partial_failure when its forecast succeeded for < SCORING_MIN_OK_REGIONS
    # BAs, so a fresh timestamp over ~50 stale forecasts still degrades.
    partial = bool(payload.get("partial_failure"))
    return {
        "status": "degraded" if (required and (stale or partial)) else "ok",
        "age_seconds": int(age),
        "threshold_seconds": LAST_SCORED_STALE_SECONDS,
        "stale": stale,
        "partial_failure": partial,
        "regions_scored": payload.get("regions_scored"),
        "regions_total": payload.get("regions_total"),
        "updated_at": payload["updated_at"],
    }


def _check_forecast_sample() -> dict[str, Any]:
    """Validate one real forecast payload can be read + has the expected
    shape. The heavy check (deserializes a full forecast), so deep-only.

    Reads ``gridpulse:forecast:{PRECOMPUTE_DEFAULT_REGION}:1h`` and
    asserts a non-empty ``forecasts`` list whose first row carries a
    numeric ``predicted_demand_mw``.
    """
    from config import PRECOMPUTE_DEFAULT_REGION, REQUIRE_REDIS
    from data.redis_client import redis_available, redis_get, redis_key

    required = bool(REQUIRE_REDIS)
    region = PRECOMPUTE_DEFAULT_REGION
    try:
        if not redis_available():
            return {"status": "skipped", "reason": "redis_unavailable"}
        payload = redis_get(redis_key(f"forecast:{region}:1h"))
    except Exception as exc:  # pragma: no cover — defensive
        return {"status": "degraded" if required else "skipped", "error": str(exc)}

    if not isinstance(payload, dict):
        return {
            "status": "degraded" if required else "skipped",
            "region": region,
            "reason": "no_forecast_payload",
        }

    forecasts = payload.get("forecasts")
    if not isinstance(forecasts, list) or not forecasts:
        return {
            "status": "degraded" if required else "skipped",
            "region": region,
            "reason": "empty_forecasts",
        }

    first = forecasts[0]
    if not isinstance(first, dict) or not isinstance(
        first.get("predicted_demand_mw"), (int, float)
    ):
        return {
            "status": "degraded" if required else "skipped",
            "region": region,
            "reason": "malformed_forecast_row",
        }

    return {
        "status": "ok",
        "region": region,
        "rows": len(forecasts),
    }


def build_health_report(deep: bool = False) -> tuple[dict[str, Any], int]:
    """Build the ``/health`` response body + HTTP status code.

    Args:
        deep: When True, additionally runs the forecast-payload check.

    Returns:
        ``(body, http_status)``. ``http_status`` is always 200 — see the
        module docstring for why a degraded instance still returns 200.
    """
    # In-memory cache counts (cheap, no I/O) — preserved from the original
    # shallow /health for continuity with existing dashboards.
    try:
        from components.callbacks import _BACKTEST_CACHE, _MODEL_CACHE, _PREDICTION_CACHE

        precompute = {
            "models_cached": len(_MODEL_CACHE),
            "predictions_cached": len(_PREDICTION_CACHE),
            "backtests_cached": len(_BACKTEST_CACHE),
        }
    except Exception:  # pragma: no cover — defensive
        precompute = {}

    checks: dict[str, Any] = {
        "redis": _check_redis(),
        "last_scored": _check_last_scored(),
    }
    if deep:
        checks["forecast_sample"] = _check_forecast_sample()

    # Overall: degraded if any check is degraded; otherwise healthy.
    # "skipped" and "ok" both count as not-degraded.
    degraded = any(c.get("status") == "degraded" for c in checks.values())
    status = "degraded" if degraded else "healthy"

    body = {
        "status": status,
        "deep": deep,
        "checks": checks,
        "precompute": precompute,
    }
    # Always 200 — see module docstring. Status is carried in the body.
    return body, 200
