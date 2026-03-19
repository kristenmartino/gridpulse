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


def redis_available() -> bool:
    """Check if Redis is connected and responsive."""
    return _get_redis() is not None
