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
