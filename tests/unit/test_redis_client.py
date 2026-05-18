"""Unit tests for data/redis_client.py — dual-mode Redis client."""

import json
from unittest.mock import MagicMock, patch


class TestRedisClientGracefulFallback:
    """Verify that redis_client returns None when Redis is unavailable."""

    def test_no_redis_host_returns_none(self):
        """Without REDIS_HOST, redis_get always returns None."""
        import data.redis_client as rc

        rc._redis_client = None
        rc._redis_init_attempted = False

        with patch.dict("os.environ", {"REDIS_HOST": ""}, clear=False):
            result = rc.redis_get("wattcast:actuals:FPL")
            assert result is None

    def test_redis_unavailable_returns_none(self):
        """When Redis host is set but unreachable, redis_get returns None."""
        import data.redis_client as rc

        rc._redis_client = None
        rc._redis_init_attempted = False

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
            rc._redis_init_attempted = False
            result = rc.redis_get("wattcast:actuals:FPL")
            assert result is None

    def test_redis_available_returns_true(self):
        """redis_available() returns True when connected."""
        import data.redis_client as rc

        rc._redis_client = MagicMock()
        rc._redis_init_attempted = True
        assert rc.redis_available() is True

    def test_redis_available_returns_false_no_host(self):
        """redis_available() returns False when no REDIS_HOST."""
        import data.redis_client as rc

        rc._redis_client = None
        rc._redis_init_attempted = True
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
        rc._redis_init_attempted = True

        result = rc.redis_get("wattcast:actuals:FPL")
        assert result == payload
        mock_client.get.assert_called_once_with("wattcast:actuals:FPL")

    def test_redis_get_returns_none_for_missing_key(self):
        """redis_get returns None when key doesn't exist."""
        import data.redis_client as rc

        mock_client = MagicMock()
        mock_client.get.return_value = None
        rc._redis_client = mock_client
        rc._redis_init_attempted = True

        result = rc.redis_get("wattcast:actuals:MISSING")
        assert result is None

    def test_redis_get_handles_corrupt_json(self):
        """redis_get returns None on invalid JSON without raising."""
        import data.redis_client as rc

        mock_client = MagicMock()
        mock_client.get.return_value = "not-valid-json{{"
        rc._redis_client = mock_client
        rc._redis_init_attempted = True

        result = rc.redis_get("wattcast:actuals:FPL")
        assert result is None

    def test_redis_get_handles_read_exception(self):
        """redis_get returns None on Redis read errors."""
        import data.redis_client as rc

        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("Connection lost")
        rc._redis_client = mock_client
        rc._redis_init_attempted = True

        result = rc.redis_get("wattcast:actuals:FPL")
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
        rc._redis_init_attempted = True

        result = rc.redis_get("wattcast:backtest:FPL:24")
        assert result is not None
        assert "metrics" in result
        assert "actual" in result
        assert "predictions" in result
        assert "timestamps" in result
        assert result["metrics"]["xgboost"]["mape"] == 3.45


class TestRedisKeyPrefix:
    """Verify the ``redis_key()`` helper composes the prefix correctly.

    Phase 1 of the ``wattcast:`` → ``gridpulse:`` migration tracked in
    issue #91. The helper lives in ``data/redis_client.py`` and reads
    ``REDIS_KEY_PREFIX`` from ``config`` at call time, so an env-var
    flip after process start does not propagate (matches the Cloud Run
    deploy boundary — each new revision picks up the new prefix).
    """

    def test_default_prefix_is_wattcast(self):
        """Until Phase 3 ops flip, the default prefix matches the legacy literal."""
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
            assert rc.redis_key("actuals:FPL") == "wattcast:actuals:FPL"
            assert rc.redis_key("forecast:ERCOT:1h") == "wattcast:forecast:ERCOT:1h"

    def test_env_var_override_changes_prefix(self):
        """Setting ``REDIS_KEY_PREFIX`` flips the prefix on next import."""
        import importlib

        import config
        import data.redis_client as rc

        with patch.dict("os.environ", {"REDIS_KEY_PREFIX": "gridpulse"}, clear=False):
            importlib.reload(config)
            importlib.reload(rc)
            assert rc.redis_key("actuals:FPL") == "gridpulse:actuals:FPL"
            assert rc.redis_key("backtest:forecast_exog:PJM:24") == (
                "gridpulse:backtest:forecast_exog:PJM:24"
            )

        # Restore default so subsequent tests aren't tainted by the reload above.
        importlib.reload(config)
        importlib.reload(rc)

    def test_suffix_is_composed_verbatim(self):
        """No escaping, no validation — caller owns the suffix shape."""
        import data.redis_client as rc

        # Empty suffix (edge case)
        assert rc.redis_key("") == "wattcast:"
        # Colons in the suffix pass through (this is how multi-part keys work)
        assert rc.redis_key("a:b:c") == "wattcast:a:b:c"


class TestRedisPublishDualWrite:
    """Verify ``redis_publish()`` writes to primary and (optionally) dual-write prefix.

    Phase 2 of the ``wattcast:`` → ``gridpulse:`` migration tracked in
    issue #91. The helper composes the primary key from
    ``REDIS_KEY_PREFIX`` and, when ``REDIS_DUAL_WRITE_PREFIX`` is set,
    mirrors the same payload to the secondary namespace.
    """

    def _reload_modules(self):
        """Reload config + redis_client so the env vars set in the test propagate."""
        import importlib

        import config
        import data.redis_client as rc

        importlib.reload(config)
        importlib.reload(rc)
        return rc

    def test_single_write_when_dual_prefix_unset(self):
        """Default mode (no dual-write env) writes once to the primary prefix."""
        with patch.dict(
            "os.environ",
            {"REDIS_KEY_PREFIX": "wattcast"},
            clear=False,
        ) as env:
            env.pop("REDIS_DUAL_WRITE_PREFIX", None)
            rc = self._reload_modules()

            with patch("data.redis_client.redis_set", return_value=True) as mock_set:
                ok = rc.redis_publish("actuals:FPL", {"x": 1}, ttl=60)

            assert ok is True
            assert mock_set.call_count == 1
            args, kwargs = mock_set.call_args
            assert args[0] == "wattcast:actuals:FPL"
            assert args[1] == {"x": 1}
            assert kwargs == {"ttl": 60}

    def test_dual_write_when_prefix_set(self):
        """When ``REDIS_DUAL_WRITE_PREFIX`` is set, writes to both prefixes."""
        with patch.dict(
            "os.environ",
            {"REDIS_KEY_PREFIX": "wattcast", "REDIS_DUAL_WRITE_PREFIX": "gridpulse"},
            clear=False,
        ):
            rc = self._reload_modules()

            with patch("data.redis_client.redis_set", return_value=True) as mock_set:
                ok = rc.redis_publish("forecast:ERCOT:1h", {"y": 2}, ttl=120)

            assert ok is True
            assert mock_set.call_count == 2
            primary_call, dual_call = mock_set.call_args_list
            assert primary_call.args[0] == "wattcast:forecast:ERCOT:1h"
            assert dual_call.args[0] == "gridpulse:forecast:ERCOT:1h"
            # Same payload + TTL on both writes
            assert primary_call.args[1] == dual_call.args[1] == {"y": 2}
            assert primary_call.kwargs == dual_call.kwargs == {"ttl": 120}

    def test_dual_write_failure_doesnt_break_primary(self):
        """A failed dual-write logs a warning but still reports primary success.

        Justifies the "best-effort" claim in the docstring: during Phase 2
        rollout, a transient failure on the new prefix shouldn't degrade
        the production write path. The next scoring cycle overwrites both
        keys within an hour, so missed dual-writes self-heal.
        """
        with patch.dict(
            "os.environ",
            {"REDIS_KEY_PREFIX": "wattcast", "REDIS_DUAL_WRITE_PREFIX": "gridpulse"},
            clear=False,
        ):
            rc = self._reload_modules()

            # First call (primary) succeeds, second (dual) fails.
            with patch("data.redis_client.redis_set", side_effect=[True, False]):
                ok = rc.redis_publish("diagnostics:PJM", {"z": 3})

            # Primary success → caller sees True even though dual failed.
            assert ok is True

    def test_primary_failure_reports_false(self):
        """When the primary write fails, return False regardless of dual outcome."""
        with patch.dict(
            "os.environ",
            {"REDIS_KEY_PREFIX": "wattcast", "REDIS_DUAL_WRITE_PREFIX": "gridpulse"},
            clear=False,
        ):
            rc = self._reload_modules()

            # Primary fails, dual succeeds. Still report failure — the
            # source of truth (current prefix) didn't write.
            with patch("data.redis_client.redis_set", side_effect=[False, True]):
                ok = rc.redis_publish("alerts:CAISO", {"w": 4})

            assert ok is False

    def test_same_prefix_for_both_skips_dual_write(self):
        """If ops accidentally set DUAL == PRIMARY, write once not twice.

        Defensive: avoids a redundant Redis round-trip and a confusing
        log line if someone configures the same prefix for both env vars.
        """
        with patch.dict(
            "os.environ",
            {"REDIS_KEY_PREFIX": "gridpulse", "REDIS_DUAL_WRITE_PREFIX": "gridpulse"},
            clear=False,
        ):
            rc = self._reload_modules()

            with patch("data.redis_client.redis_set", return_value=True) as mock_set:
                rc.redis_publish("meta:last_scored", {"t": "now"})

            assert mock_set.call_count == 1
            assert mock_set.call_args.args[0] == "gridpulse:meta:last_scored"

        # Restore default for sibling tests.
        self._reload_modules()
