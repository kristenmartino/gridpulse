"""Unit tests for the Redis-backed per-IP rate limiter (``ratelimit.py``, #253).

The limiter is a fixed-window counter. Two invariants matter most:
- it **fails open** (allows) whenever Redis is absent or errors, so it can
  never self-inflict an availability outage;
- it blocks strictly *after* the limit is exceeded within a window.

Time is pinned so tests never straddle a window boundary.
"""

from __future__ import annotations

from unittest.mock import patch


class _FakePipeline:
    def __init__(self, store):
        self.store = store
        self._last = None

    def incr(self, key):
        self.store[key] = self.store.get(key, 0) + 1
        self._last = self.store[key]
        return self

    def expire(self, key, ttl):
        return self

    def execute(self):
        return [self._last, True]


class _FakeRedis:
    def __init__(self):
        self.store = {}

    def pipeline(self):
        return _FakePipeline(self.store)


class _ErroringRedis:
    def pipeline(self):
        raise RuntimeError("synthetic redis outage")


def _fixed_time():
    return 1_000_000.0  # any value inside a single window


class TestCheckRateLimit:
    def test_allows_up_to_the_limit_then_blocks(self):
        import ratelimit

        fake = _FakeRedis()
        with (
            patch("ratelimit._get_redis", return_value=fake),
            patch("ratelimit.time.time", _fixed_time),
        ):
            results = [ratelimit.check_rate_limit("api", "1.2.3.4", limit=3) for _ in range(4)]

        assert [r.allowed for r in results] == [True, True, True, False]
        assert [r.remaining for r in results] == [2, 1, 0, 0]
        # Blocked result carries a positive retry-after (seconds to window reset).
        assert results[-1].retry_after > 0

    def test_separate_identities_have_separate_budgets(self):
        import ratelimit

        fake = _FakeRedis()
        with (
            patch("ratelimit._get_redis", return_value=fake),
            patch("ratelimit.time.time", _fixed_time),
        ):
            a = ratelimit.check_rate_limit("api", "1.1.1.1", limit=1)
            b = ratelimit.check_rate_limit("api", "2.2.2.2", limit=1)
            a2 = ratelimit.check_rate_limit("api", "1.1.1.1", limit=1)

        assert a.allowed is True
        assert b.allowed is True  # different IP, own budget
        assert a2.allowed is False  # same IP, second hit over limit=1

    def test_separate_buckets_have_separate_budgets(self):
        import ratelimit

        fake = _FakeRedis()
        with (
            patch("ratelimit._get_redis", return_value=fake),
            patch("ratelimit.time.time", _fixed_time),
        ):
            api = ratelimit.check_rate_limit("api", "1.1.1.1", limit=1)
            dash = ratelimit.check_rate_limit("dash", "1.1.1.1", limit=1)

        assert api.allowed is True
        assert dash.allowed is True  # api budget doesn't spend dash budget

    def test_fails_open_when_redis_absent(self):
        import ratelimit

        with patch("ratelimit._get_redis", return_value=None):
            # Even far past the limit, no Redis → allowed (never self-DoS).
            for _ in range(100):
                assert ratelimit.check_rate_limit("api", "1.2.3.4", limit=1).allowed is True

    def test_fails_open_on_redis_error(self):
        import ratelimit

        with (
            patch("ratelimit._get_redis", return_value=_ErroringRedis()),
            patch("ratelimit.time.time", _fixed_time),
        ):
            result = ratelimit.check_rate_limit("api", "1.2.3.4", limit=1)
        assert result.allowed is True


class TestIsExempt:
    def test_exempt_ip_bypasses(self):
        import ratelimit

        with patch("config.RATE_LIMIT_EXEMPT_IPS", frozenset({"9.9.9.9"})):
            assert ratelimit.is_exempt("9.9.9.9") is True
            assert ratelimit.is_exempt("1.2.3.4") is False

    def test_empty_ip_is_never_exempt(self):
        import ratelimit

        with patch("config.RATE_LIMIT_EXEMPT_IPS", frozenset({""})):
            # A resolution miss ("") must not accidentally match an empty entry.
            assert ratelimit.is_exempt("") is False


class TestCallerIp:
    def test_uses_rightmost_xff_hop_spoof_resistant(self):
        import ratelimit

        class _Req:
            headers = {"X-Forwarded-For": "127.0.0.1, 9.9.9.9"}
            remote_addr = "10.0.0.1"

        # Leftmost (127.0.0.1) is client-controlled; the real edge-appended hop
        # is rightmost (9.9.9.9). The limiter must key on the latter.
        assert ratelimit.caller_ip(_Req()) == "9.9.9.9"

    def test_falls_back_to_remote_addr_without_xff(self):
        import ratelimit

        class _Req:
            headers: dict = {}
            remote_addr = "10.0.0.1"

        assert ratelimit.caller_ip(_Req()) == "10.0.0.1"
