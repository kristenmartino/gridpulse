"""
Thin Redis client for dual-mode Dash callbacks.

When REDIS_HOST is set, reads pre-computed data from Redis (v2 pipeline).
When REDIS_HOST is empty or Redis is unreachable, returns None so
callbacks fall through to the existing v1 compute path.

Both paths have two flavours, split along the same line (#268 / #313):

* **Fail-soft** — ``redis_get`` returns None on any error, ``redis_set``
  returns False. For the web tier, where a Redis blip should degrade to the
  warming state, never crash a callback.
* **Fail-loud** — ``redis_get_strict`` raises :class:`RedisReadError` so that
  *absence* and *failure* are different answers; ``persist`` raises
  :class:`RedisWriteError` on a dropped write. For stateful read-modify-write
  consumers (the scoring-job window phases), where mistaking an outage for
  "no history" destructively rebuilds state — the #313 vintage re-pin.
"""

import json
import logging
import os
import threading
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
#: Serializes the FIRST connect so concurrent callers wait (~1s) for the
#: winner instead of taking the silent backoff-None. Before #313's trigger
#: fix, a scoring container's 51 worker threads raced this init at cold
#: start: the first thread set ``_redis_last_attempt`` and spent ~1s
#: connecting over the VPC, and every thread arriving inside that window got
#: a silent ``None`` — indistinguishable from "Redis is down", with no log.
#: That None is what destructively re-pinned four regions' vintage windows
#: on 2026-07-16 (caught live by the #314 tripwire at 09:00:52.85-.88Z on
#: 2026-07-17: three failures 30ms apart, CAISO succeeding 1.5s later).
_redis_init_lock = threading.Lock()


def _get_redis():
    """Lazy-init a Redis connection. Returns client or None.

    A healthy client is cached and reused. The first connect is serialized
    behind ``_redis_init_lock`` — concurrent callers block briefly for the
    winner's result rather than being told "no Redis" while a connect is in
    flight (#313). A genuinely FAILED attempt is re-probed at most once per
    ``_REDIS_RETRY_INTERVAL_S`` (backoff), so a transient blip recovers
    within the same process instead of pinning Redis off for its lifetime
    (#268 / P2-03).
    """
    global _redis_client, _redis_last_attempt

    if _redis_client is not None:
        return _redis_client

    with _redis_init_lock:
        # Double-checked: the winner may have connected while we waited.
        if _redis_client is not None:
            return _redis_client

        now = time.monotonic()
        if _redis_last_attempt and (now - _redis_last_attempt) < _REDIS_RETRY_INTERVAL_S:
            return None  # recent FAILED attempt — back off rather than re-probe every call
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
            # A SUCCESSFUL connect clears the attempt stamp so an unrelated
            # later re-init (tests, client reset) is not backoff-blocked.
            _redis_last_attempt = 0.0
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


class RedisReadError(RuntimeError):
    """A Redis read failed for infrastructure reasons — client unavailable,
    connection/command error, or an unparseable payload. Distinct from a key
    that is genuinely absent (a nil reply on a healthy connection).

    The read-side twin of :class:`RedisWriteError` (#268). Exists because of
    #313: ``redis_get`` collapses failure and absence into one ``None``, and a
    stateful consumer that mistakes an outage for "no history" destructively
    rebuilds its window — prod re-pinned four regions' vintage first-sight
    records exactly this way on 2026-07-16.
    """


def redis_get_strict(key: str) -> dict | list | None:
    """Read a JSON value, refusing to conflate failure with absence (#313).

    Returns the parsed object, or ``None`` **only** when Redis affirmatively
    reports the key absent. Every other outcome — no client, command error,
    unparseable payload — raises :class:`RedisReadError`. Callers whose next
    action differs between "the key is not there" and "I could not find out"
    (window phases doing read-modify-write) must use this instead of
    :func:`redis_get`.
    """
    client = _get_redis()
    if client is None:
        raise RedisReadError(f"redis unavailable for read of {key}")
    try:
        raw = client.get(key)
    except Exception as exc:
        raise RedisReadError(f"read failed for {key}: {exc}") from exc
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError) as exc:
        # A value that exists but cannot be parsed is a failure, not an
        # absence — overwriting it would destroy whatever it was.
        raise RedisReadError(f"unparseable payload at {key}: {exc}") from exc


def redis_configured() -> bool:
    """True when this environment points at a Redis endpoint.

    Lets job phases distinguish "dev, no Redis — skip quietly" from
    "production Redis errored — protect state and fail the phase" without
    poking at private client state.
    """
    return bool(os.getenv("REDIS_HOST", ""))


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
