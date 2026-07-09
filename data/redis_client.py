"""
Thin Redis client for dual-mode Dash callbacks.

When REDIS_HOST is set, reads pre-computed data from Redis (v2 pipeline).
When REDIS_HOST is empty or Redis is unreachable, returns None so
callbacks fall through to the existing v1 compute path.

The read path never raises — all errors return None with a warning log. The
write path has two flavours: ``redis_set`` swallows failures and returns False,
while ``persist`` raises ``RedisWriteError`` so a caller that must know its write
landed (the scoring-job phases) can surface a dropped write (#268).
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

_redis_client = None
#: Monotonic time of the last *failed/absent* connection attempt. Before #268
#: a single failed first ping pinned the client off for the whole process
#: (``_redis_init_attempted`` was never reset), so a momentary blip at job start
#: silenced Redis for the entire scoring tick. Now a failure is retried at most
#: once per ``_REDIS_RETRY_INTERVAL_S`` so it self-heals mid-run.
_redis_last_attempt = 0.0
_REDIS_RETRY_INTERVAL_S = 30.0


def _get_redis():
    """Lazy-init a Redis connection. Returns client or None.

    A healthy client is cached and reused. A failed/absent connection is
    re-probed at most once per ``_REDIS_RETRY_INTERVAL_S`` (backoff), so a
    transient blip recovers within the same process instead of pinning Redis
    off for its lifetime (#268 / P2-03).
    """
    global _redis_client, _redis_last_attempt

    if _redis_client is not None:
        return _redis_client

    now = time.monotonic()
    if _redis_last_attempt and (now - _redis_last_attempt) < _REDIS_RETRY_INTERVAL_S:
        return None  # recent failed attempt — back off rather than re-probe every call
    _redis_last_attempt = now

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


class RedisWriteError(RuntimeError):
    """A required Redis write failed. Raised by :func:`persist` so a phase that
    computed a result but couldn't persist it surfaces as *failed* rather than
    silently reporting ok (#268 / P2-03)."""


def persist(key: str, value: dict | list, ttl: int = 86400) -> None:
    """Write to Redis, raising :class:`RedisWriteError` on any failure.

    The strict counterpart of :func:`redis_set` (which swallows failures and
    returns False). Used by the scoring-job write phases so a dropped write
    propagates into the phase's ok-flag — a forecast that computed but couldn't
    persist must not count as scored (#268; feeds the #267 region-ok logic).
    """
    if not redis_set(key, value, ttl=ttl):
        raise RedisWriteError(f"redis write failed for key {key!r}")


def redis_available() -> bool:
    """Check if Redis is connected and responsive."""
    return _get_redis() is not None
