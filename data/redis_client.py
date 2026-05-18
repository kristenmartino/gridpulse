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

    Callers pass the part of the key *after* the prefix, e.g.
    ``redis_key("actuals:FPL")`` returns ``"gridpulse:actuals:FPL"``
    (default). Putting the indirection in this module — rather than at
    every callsite — means future renames are a single-line change.

    The prefix is read every call from ``config.REDIS_KEY_PREFIX``, so
    tests can override it via ``importlib.reload(config)`` after
    monkeypatching the env. In production it's effectively constant per
    process — Cloud Run revisions get their prefix from the env on first
    request and don't see env changes until the next deploy.

    Issue #91 tracked the original ``wattcast`` → ``gridpulse`` rename;
    this helper was introduced as part of that work and stays useful as
    a hedge against the next rename.
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


def redis_available() -> bool:
    """Check if Redis is connected and responsive."""
    return _get_redis() is not None
