"""
Thin Redis client for dual-mode Dash callbacks.

When REDIS_HOST is set, reads pre-computed data from Redis (v2 pipeline).
When REDIS_HOST is empty or Redis is unreachable, returns None so
callbacks fall through to the existing v1 compute path.

Never raises exceptions — all errors return None with a warning log.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

_redis_client = None
_redis_init_attempted = False


def _get_redis():
    """Lazy-init a Redis connection. Returns client or None."""
    global _redis_client, _redis_init_attempted

    if _redis_init_attempted:
        return _redis_client

    _redis_init_attempted = True
    host = os.getenv("REDIS_HOST", "")
    if not host:
        logger.debug("REDIS_HOST not set — v1 compute mode")
        return None

    port = int(os.getenv("REDIS_PORT", "6379"))
    try:
        import redis

        client = redis.Redis(
            host=host,
            port=port,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        client.ping()
        _redis_client = client
        logger.info("Redis connected: %s:%s", host, port)
        return _redis_client
    except Exception as exc:
        logger.warning("Redis unavailable (%s:%s): %s — falling back to v1", host, port, exc)
        return None


def redis_key(suffix: str) -> str:
    """Compose a fully-qualified Redis key from the configured prefix and a suffix.

    Phase 1 of the ``wattcast:`` → ``gridpulse:`` migration (issue #91).
    Callers pass the part of the key *after* the prefix, e.g.
    ``redis_key("actuals:FPL")`` returns ``"wattcast:actuals:FPL"``
    (default) or ``"gridpulse:actuals:FPL"`` when ``REDIS_KEY_PREFIX``
    is set. Putting the indirection in this module — rather than at
    every callsite — means future migrations are a single-line change.

    The prefix is read once per process at import time of ``config``,
    so an env-var flip requires a process restart. That matches the
    Cloud Run deploy boundary: changing ``REDIS_KEY_PREFIX`` in the
    service/job env definition triggers a new revision, which gets a
    new prefix on its first ``redis_key`` call.
    """
    # Imported lazily to avoid a circular: config -> nothing, this module
    # -> config is fine, but importing at module top would force config
    # to load before logging is configured in tests that monkeypatch
    # ``os.environ``.
    from config import REDIS_KEY_PREFIX

    return f"{REDIS_KEY_PREFIX}:{suffix}"


def redis_get(key: str) -> dict | list | None:
    """Read a JSON value from Redis. Returns parsed object or None."""
    client = _get_redis()
    if client is None:
        return None
    try:
        raw = client.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:
        logger.warning("Redis read error for %s: %s", key, exc)
        return None


def redis_set(key: str, value: dict | list, ttl: int = 86400) -> bool:
    """Write a JSON value to Redis with TTL (default 24h). Returns True on success."""
    client = _get_redis()
    if client is None:
        return False
    try:
        client.setex(key, ttl, json.dumps(value))
        return True
    except Exception as exc:
        logger.warning("Redis write error for %s: %s", key, exc)
        return False


def redis_publish(suffix: str, value: dict | list, ttl: int = 86400) -> bool:
    """Write a JSON value to the primary prefix and (optionally) a dual-write prefix.

    Phase 2 of the ``wattcast:`` → ``gridpulse:`` migration (issue #91).
    Producer jobs (``jobs/phases.py``) call this with a suffix like
    ``"actuals:{region}"``; the helper composes the full key from
    ``REDIS_KEY_PREFIX`` and writes there. When
    ``REDIS_DUAL_WRITE_PREFIX`` is also set (Phase 2 ops), the same
    payload is written to ``{dual_prefix}:{suffix}`` so the new
    namespace is populated before the Phase 3 read-cutover happens.

    Both writes are attempted independently. Return value is ``True`` only
    when the **primary** write succeeds — the dual write is best-effort
    (it's a migration safety net, not the source of truth). Dual-write
    failures log a warning but don't fail the primary path; the next
    scoring/training cycle will retry.

    Args:
        suffix: Key suffix after the prefix, e.g. ``"actuals:FPL"``.
        value: JSON-serializable payload.
        ttl: TTL in seconds (default 24h, matching scoring/training cadence).

    Returns:
        ``True`` if the primary write succeeded, ``False`` otherwise.
    """
    # Imported lazily for the same reason as ``redis_key`` — avoid forcing
    # config to load before test fixtures monkeypatch the env.
    from config import REDIS_DUAL_WRITE_PREFIX, REDIS_KEY_PREFIX

    primary_key = f"{REDIS_KEY_PREFIX}:{suffix}"
    primary_ok = redis_set(primary_key, value, ttl=ttl)

    if REDIS_DUAL_WRITE_PREFIX and REDIS_DUAL_WRITE_PREFIX != REDIS_KEY_PREFIX:
        dual_key = f"{REDIS_DUAL_WRITE_PREFIX}:{suffix}"
        # redis_set already swallows + logs exceptions, so this just
        # produces a structured warning on failure without raising.
        dual_ok = redis_set(dual_key, value, ttl=ttl)
        if not dual_ok:
            logger.warning(
                "redis_dual_write_failed primary=%s dual=%s — dual is best-effort",
                primary_key,
                dual_key,
            )

    return primary_ok


def redis_available() -> bool:
    """Check if Redis is connected and responsive."""
    return _get_redis() is not None
