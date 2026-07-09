"""Redis-backed per-IP request rate limiting for the public web tier (#253).

The public JSON API (#250/#251) made the stateless web tier publicly
programmable. This is a lightweight **fixed-window** limiter, backed by the
Redis the web tier already talks to, so the limit is shared across the 1-4
Cloud Run instances (an in-process counter would let a flood multiply by the
instance count). It writes only ephemeral ``gridpulse:ratelimit:*`` counter
keys with a short TTL — not model data — so it doesn't violate the "scoring job
is the only writer" guardrail.

**Fail-open by design:** any Redis error (or no Redis configured) returns
*allowed* rather than blocking. A rate limiter that fails closed would turn a
Redis blip into a self-inflicted availability outage — the exact failure the
limiter exists to prevent. Enforcement is gated by ``config.rate_limit_active()``
(staging/prod only), so dev is unthrottled and needs no Redis.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import structlog

from data.redis_client import _get_redis, redis_key
from observability import untrusted_client_ip

log = structlog.get_logger()


class RateLimitResult(NamedTuple):
    """Outcome of a rate-limit check.

    ``retry_after`` is the seconds until the current fixed window resets
    (0 when the request is allowed).
    """

    allowed: bool
    remaining: int
    retry_after: int


def caller_ip(flask_request) -> str:
    """Spoof-resistant caller IP used to key the limit.

    Mirrors the ``/metrics`` allowlist resolution (rightmost, edge-appended
    ``X-Forwarded-For`` hop — a client can't prepend a fake key; P2-52).
    """
    return untrusted_client_ip(
        flask_request.headers.get("X-Forwarded-For"), flask_request.remote_addr
    )


def is_exempt(ip: str) -> bool:
    """True when ``ip`` is a trusted source that bypasses rate limiting.

    For a known shared-NAT egress (a control room behind one corporate IP)
    where many operators would otherwise share a single per-IP bucket and
    collect spurious 429s. Keyed on the same spoof-resistant IP as the limit
    (``RATE_LIMIT_EXEMPT_IPS``, empty by default). Entries may be exact IPs or
    CIDR prefixes (e.g. a corporate ``/24`` or an IPv6 ``/64``).
    """
    from config import RATE_LIMIT_EXEMPT_IPS
    from observability import ip_in_allowlist

    return ip_in_allowlist(ip, RATE_LIMIT_EXEMPT_IPS)


def check_rate_limit(bucket: str, identity: str, limit: int, window_s: int = 60) -> RateLimitResult:
    """Increment and test a fixed-window counter for ``(bucket, identity)``.

    Args:
        bucket: logical surface, e.g. ``"api"`` or ``"dash"`` (keeps the API
            and callback budgets independent).
        identity: caller key, typically the resolved IP.
        limit: max requests permitted per window.
        window_s: window length in seconds (default 60).

    Returns:
        ``RateLimitResult`` — ``allowed=True`` (fail-open) whenever Redis is
        absent or errors, so the limiter never self-inflicts an outage.
    """
    client = _get_redis()
    if client is None:
        return RateLimitResult(allowed=True, remaining=limit, retry_after=0)

    now = int(time.time())
    window_start = now - (now % window_s)
    key = redis_key(f"ratelimit:{bucket}:{identity}:{window_start}")
    try:
        pipe = client.pipeline()
        pipe.incr(key)
        # Refresh the TTL on every hit; the key is window-scoped so it can
        # outlive the window by at most ``window_s`` before Redis reaps it.
        pipe.expire(key, window_s)
        count = int(pipe.execute()[0])
    except Exception as exc:  # fail-open on any Redis error
        log.warning("rate_limit_redis_error", bucket=bucket, error=str(exc))
        return RateLimitResult(allowed=True, remaining=limit, retry_after=0)

    if count <= limit:
        return RateLimitResult(allowed=True, remaining=max(0, limit - count), retry_after=0)
    return RateLimitResult(allowed=False, remaining=0, retry_after=window_s - (now % window_s))
