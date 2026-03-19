"""Tests for the ForecastCache Redis read helper."""
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest


class TestForecastCache:
    """Tests for src.api.cache.ForecastCache."""

    def test_get_forecast_returns_none_on_empty_redis(self, mock_redis):
        """When Redis has no data, get_forecast returns None."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = mock_redis
        result = cache.get_forecast("ERCOT", "1h")
        assert result is None

    def test_get_forecast_returns_dict_when_populated(self, populated_redis, sample_forecast):
        """When Redis has data, get_forecast returns the parsed dict."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = populated_redis
        result = cache.get_forecast("ERCOT", "1h")
        assert result is not None
        assert result["region"] == "ERCOT"
        assert "forecasts" in result
        assert len(result["forecasts"]) == 2

    def test_get_forecast_returns_none_for_unknown_region(self, populated_redis):
        """Requesting an uncached region returns None."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = populated_redis
        result = cache.get_forecast("UNKNOWN", "1h")
        assert result is None

    def test_get_all_regions_empty(self, mock_redis):
        """get_all_regions returns empty list when Redis is empty."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = mock_redis
        result = cache.get_all_regions("1h")
        assert result == []

    def test_get_all_regions_with_data(self, populated_redis):
        """get_all_regions returns list with populated regions."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = populated_redis
        result = cache.get_all_regions("1h")
        # Only ERCOT is populated in the fixture
        assert len(result) == 1
        assert result[0]["region"] == "ERCOT"

    def test_get_pipeline_metadata_none(self, mock_redis):
        """get_pipeline_metadata returns None when no metadata."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = mock_redis
        result = cache.get_pipeline_metadata()
        assert result is None

    def test_get_pipeline_metadata_populated(self, populated_redis, sample_metadata):
        """get_pipeline_metadata returns dict when metadata exists."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = populated_redis
        result = cache.get_pipeline_metadata()
        assert result is not None
        assert result["regions_scored"] == 8

    def test_is_healthy_true_when_recent(self, populated_redis):
        """is_healthy returns True when last scored is recent."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = populated_redis
        assert cache.is_healthy() is True

    def test_is_healthy_false_when_empty(self, mock_redis):
        """is_healthy returns False when Redis is empty."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = mock_redis
        assert cache.is_healthy() is False

    def test_is_healthy_false_when_stale(self, mock_redis):
        """is_healthy returns False when last scored is too old."""
        from src.api.cache import ForecastCache

        stale_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        mock_redis.set(
            "wattcast:meta:last_scored",
            json.dumps({"scored_at": stale_time, "regions_scored": 8}),
        )
        cache = ForecastCache()
        cache.client = mock_redis
        assert cache.is_healthy() is False

    def test_get_forecast_handles_corrupt_json(self, mock_redis):
        """Corrupt JSON in Redis returns None, not an exception."""
        from src.api.cache import ForecastCache

        mock_redis.setex("wattcast:forecast:ERCOT:1h", 3600, "not-valid-json{{{")
        cache = ForecastCache()
        cache.client = mock_redis
        result = cache.get_forecast("ERCOT", "1h")
        assert result is None
