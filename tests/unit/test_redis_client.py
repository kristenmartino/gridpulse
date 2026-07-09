"""Unit tests for data/redis_client.py — dual-mode Redis client."""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestRedisClientGracefulFallback:
    """Verify that redis_client returns None when Redis is unavailable."""

    def test_no_redis_host_returns_none(self):
        """Without REDIS_HOST, redis_get always returns None."""
        import data.redis_client as rc

        rc._redis_client = None
        rc._redis_last_attempt = 0.0

        with patch.dict("os.environ", {"REDIS_HOST": ""}, clear=False):
            result = rc.redis_get("gridpulse:actuals:FPL")
            assert result is None

    def test_redis_unavailable_returns_none(self):
        """When Redis host is set but unreachable, redis_get returns None."""
        import data.redis_client as rc

        rc._redis_client = None
        rc._redis_last_attempt = 0.0

        with (
            patch.dict(
                "os.environ", {"REDIS_HOST": "nonexistent", "REDIS_PORT": "6379"}, clear=False
            ),
            patch("data.redis_client.redis", create=True) as mock_redis_mod,
        ):
            mock_client = MagicMock()
            mock_client.ping.side_effect = ConnectionError("Connection refused")
            mock_redis_mod.Redis.return_value = mock_client
            # Need to reimport to pick up the mock
            rc._redis_last_attempt = 0.0
            result = rc.redis_get("gridpulse:actuals:FPL")
            assert result is None

    def test_redis_available_returns_true(self):
        """redis_available() returns True when connected."""
        import data.redis_client as rc

        rc._redis_client = MagicMock()
        rc._redis_last_attempt = 0.0
        assert rc.redis_available() is True

    def test_redis_available_returns_false_no_host(self):
        """redis_available() returns False when no REDIS_HOST."""
        import data.redis_client as rc

        rc._redis_client = None
        rc._redis_last_attempt = 0.0
        assert rc.redis_available() is False


class TestRedisClientReads:
    """Verify redis_get correctly parses JSON from Redis."""

    def test_redis_get_parses_json(self):
        """redis_get returns parsed JSON dict."""
        import data.redis_client as rc

        mock_client = MagicMock()
        payload = {"region": "FPL", "demand_mw": [100, 200, 300]}
        mock_client.get.return_value = json.dumps(payload)
        rc._redis_client = mock_client
        rc._redis_last_attempt = 0.0

        result = rc.redis_get("gridpulse:actuals:FPL")
        assert result == payload
        mock_client.get.assert_called_once_with("gridpulse:actuals:FPL")

    def test_redis_get_returns_none_for_missing_key(self):
        """redis_get returns None when key doesn't exist."""
        import data.redis_client as rc

        mock_client = MagicMock()
        mock_client.get.return_value = None
        rc._redis_client = mock_client
        rc._redis_last_attempt = 0.0

        result = rc.redis_get("gridpulse:actuals:MISSING")
        assert result is None

    def test_redis_get_handles_corrupt_json(self):
        """redis_get returns None on invalid JSON without raising."""
        import data.redis_client as rc

        mock_client = MagicMock()
        mock_client.get.return_value = "not-valid-json{{"
        rc._redis_client = mock_client
        rc._redis_last_attempt = 0.0

        result = rc.redis_get("gridpulse:actuals:FPL")
        assert result is None

    def test_redis_get_handles_read_exception(self):
        """redis_get returns None on Redis read errors."""
        import data.redis_client as rc

        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("Connection lost")
        rc._redis_client = mock_client
        rc._redis_last_attempt = 0.0

        result = rc.redis_get("gridpulse:actuals:FPL")
        assert result is None


class TestRedisBacktestFormat:
    """Verify backtest data from Redis is in the expected format."""

    def test_backtest_has_expected_keys(self):
        """Backtest cached data has metrics, actual, predictions, timestamps."""
        import data.redis_client as rc

        backtest_payload = {
            "horizon": 24,
            "metrics": {
                "xgboost": {"mape": 3.45, "rmse": 1250.5, "mae": 950.25, "r2": 0.92},
            },
            "actual": [44500.0, 44600.0],
            "predictions": {
                "xgboost": [44520.5, 44620.3],
            },
            "timestamps": ["2024-01-15T14:00:00", "2024-01-15T15:00:00"],
        }

        mock_client = MagicMock()
        mock_client.get.return_value = json.dumps(backtest_payload)
        rc._redis_client = mock_client
        rc._redis_last_attempt = 0.0

        result = rc.redis_get("gridpulse:backtest:FPL:24")
        assert result is not None
        assert "metrics" in result
        assert "actual" in result
        assert "predictions" in result
        assert "timestamps" in result
        assert result["metrics"]["xgboost"]["mape"] == 3.45


class TestRedisKeyPrefix:
    """Verify the ``redis_key()`` helper composes the prefix correctly.

    The helper lives in ``data/redis_client.py`` and reads
    ``REDIS_KEY_PREFIX`` from ``config`` at every call. Tests that want
    to exercise an override re-import config after monkeypatching the
    env so the new value propagates.
    """

    def test_default_prefix_is_gridpulse(self):
        """The default prefix matches the product name."""
        import importlib

        import config
        import data.redis_client as rc

        # Clear any env-var override and re-import config so REDIS_KEY_PREFIX
        # picks up the actual default. Without this, a sibling test that
        # patched the env could leak into this one.
        with patch.dict("os.environ", {}, clear=False) as env:
            env.pop("REDIS_KEY_PREFIX", None)
            importlib.reload(config)
            importlib.reload(rc)
            assert rc.redis_key("actuals:FPL") == "gridpulse:actuals:FPL"
            assert rc.redis_key("forecast:ERCOT:1h") == "gridpulse:forecast:ERCOT:1h"

    def test_env_var_override_changes_prefix(self):
        """Setting ``REDIS_KEY_PREFIX`` flips the prefix on next import.

        Uses the historical ``wattcast`` value (issue #91) to verify the
        override path — pointing at any non-default namespace exercises
        the same code path. In production this is what you'd reach for
        when running an experimental scoring job that shouldn't clobber
        live keys.
        """
        import importlib

        import config
        import data.redis_client as rc

        with patch.dict("os.environ", {"REDIS_KEY_PREFIX": "wattcast"}, clear=False):
            importlib.reload(config)
            importlib.reload(rc)
            assert rc.redis_key("actuals:FPL") == "wattcast:actuals:FPL"
            assert rc.redis_key("backtest:forecast_exog:PJM:24") == (
                "wattcast:backtest:forecast_exog:PJM:24"
            )

        # Restore default so subsequent tests aren't tainted by the reload above.
        importlib.reload(config)
        importlib.reload(rc)

    def test_suffix_is_composed_verbatim(self):
        """No escaping, no validation — caller owns the suffix shape."""
        import data.redis_client as rc

        # Empty suffix (edge case)
        assert rc.redis_key("") == "gridpulse:"
        # Colons in the suffix pass through (this is how multi-part keys work)
        assert rc.redis_key("a:b:c") == "gridpulse:a:b:c"


class TestRedisReinit:
    """#268 (P2-03) — a failed connection self-heals instead of pinning Redis
    off for the whole process."""

    def test_failed_connection_retries_after_backoff(self):
        import sys

        import data.redis_client as rc

        rc._redis_client = None
        rc._redis_last_attempt = 0.0
        client = MagicMock()
        client.ping.side_effect = [ConnectionError("blip"), True]  # fail, then recover
        fake_redis = MagicMock()
        fake_redis.Redis.return_value = client
        # _get_redis does a local ``import redis``, so inject the mock into
        # sys.modules (a module-attr patch wouldn't intercept the import).
        with (
            patch.dict("os.environ", {"REDIS_HOST": "h", "REDIS_PORT": "6379"}, clear=False),
            patch.dict(sys.modules, {"redis": fake_redis}),
        ):
            assert rc._get_redis() is None  # 1st attempt fails
            assert rc._get_redis() is None  # within backoff → no re-probe
            assert client.ping.call_count == 1  # did NOT hammer the connection

            rc._redis_last_attempt = 0.0  # simulate the backoff window elapsing
            assert rc._get_redis() is client  # re-probes and recovers
            assert client.ping.call_count == 2
        rc._redis_client = None
        rc._redis_last_attempt = 0.0

    def test_healthy_client_is_cached_without_reprobe(self):
        import data.redis_client as rc

        cached = MagicMock()
        rc._redis_client = cached
        rc._redis_last_attempt = 0.0
        assert rc._get_redis() is cached
        rc._redis_client = None


class TestPersist:
    """persist() raises on a dropped write so scoring phases surface it (#268)."""

    def test_persist_raises_on_write_failure(self):
        import data.redis_client as rc

        with (
            patch("data.redis_client.redis_set", return_value=False),
            pytest.raises(rc.RedisWriteError),
        ):
            rc.persist("gridpulse:x", {"a": 1})

    def test_persist_succeeds_silently(self):
        import data.redis_client as rc

        with patch("data.redis_client.redis_set", return_value=True):
            assert rc.persist("gridpulse:x", {"a": 1}) is None
