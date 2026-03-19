"""Tests for the extended ForecastCache (all new read methods)."""
import json
from datetime import datetime, timezone

import pytest


class TestBacktestCache:

    def test_get_backtest_returns_data(self, populated_redis, sample_backtest):
        """get_backtest returns full backtest data."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = populated_redis
        result = cache.get_backtest("ERCOT", 24)

        assert result is not None
        assert result["horizon"] == 24
        assert "metrics" in result
        assert "xgboost" in result["metrics"]
        assert "actual" in result
        assert len(result["actual"]) == 3

    def test_get_backtest_with_model_filter(self, populated_redis):
        """get_backtest with model= filters to that model."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = populated_redis
        result = cache.get_backtest("ERCOT", 24, model="xgboost")

        assert result is not None
        assert "xgboost" in result["metrics"]
        assert "ensemble" not in result["metrics"]

    def test_get_backtest_returns_none_for_missing(self, mock_redis):
        """get_backtest returns None when no data."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = mock_redis
        assert cache.get_backtest("ERCOT", 24) is None

    def test_get_backtest_residuals(self, populated_redis):
        """get_backtest_residuals returns residual array."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = populated_redis
        result = cache.get_backtest_residuals("ERCOT", 24)

        assert result is not None
        assert result["region"] == "ERCOT"
        assert result["horizon"] == 24
        assert len(result["residuals"]) == 3

    def test_get_backtest_error_by_hour(self, populated_redis):
        """get_backtest_error_by_hour returns error breakdown."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = populated_redis
        result = cache.get_backtest_error_by_hour("ERCOT", 24)

        assert result is not None
        assert len(result["error_by_hour"]) == 3


class TestActualsCache:

    def test_get_actuals_returns_data(self, populated_redis, sample_actuals):
        """get_actuals returns historical demand data."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = populated_redis
        result = cache.get_actuals("ERCOT")

        assert result is not None
        assert result["region"] == "ERCOT"
        assert len(result["demand_mw"]) == 3

    def test_get_actuals_returns_none_for_missing(self, mock_redis):
        """get_actuals returns None when no data."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = mock_redis
        assert cache.get_actuals("ERCOT") is None


class TestWeatherCache:

    def test_get_weather_returns_data(self, populated_redis, sample_weather):
        """get_weather returns weather data with all variables."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = populated_redis
        result = cache.get_weather("ERCOT")

        assert result is not None
        assert result["region"] == "ERCOT"
        assert "temperature_2m" in result
        assert "wind_speed_80m" in result
        assert len(result["temperature_2m"]) == 3


class TestWeightsCache:

    def test_get_ensemble_weights_returns_data(self, populated_redis, sample_weights):
        """get_ensemble_weights returns weights and metrics."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = populated_redis
        result = cache.get_ensemble_weights("ERCOT")

        assert result is not None
        assert "weights" in result
        assert result["weights"]["xgboost"] == 0.50
        assert result["weights"]["prophet"] == 0.30
        assert "updated_at" in result


class TestScenarioCache:

    def test_get_scenario_preset_returns_data(self, populated_redis, sample_scenario_preset):
        """get_scenario_preset returns pre-computed preset data."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = populated_redis
        result = cache.get_scenario_preset("ERCOT", "winter_storm_uri")

        assert result is not None
        assert result["region"] == "ERCOT"
        assert "baseline" in result
        assert "scenario" in result
        assert "delta_mw" in result
        assert result["reserve_margin"]["status"] == "CRITICAL"

    def test_get_scenario_preset_returns_none_for_missing(self, populated_redis):
        """get_scenario_preset returns None for unknown preset."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = populated_redis
        assert cache.get_scenario_preset("ERCOT", "nonexistent") is None

    def test_get_all_scenario_presets(self, populated_redis):
        """get_all_scenario_presets returns all presets for a region."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = populated_redis
        result = cache.get_all_scenario_presets("ERCOT")

        assert isinstance(result, list)
        assert len(result) >= 1
        assert result[0]["region"] == "ERCOT"


class TestGenerationCache:

    def test_get_generation_returns_data(self, populated_redis, sample_generation):
        """get_generation returns generation mix data."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = populated_redis
        result = cache.get_generation("ERCOT")

        assert result is not None
        assert result["region"] == "ERCOT"
        assert "wind" in result
        assert "solar" in result
        assert "renewable_pct" in result


class TestAlertsCache:

    def test_get_alerts_returns_data(self, populated_redis, sample_alerts):
        """get_alerts returns alerts and stress score."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = populated_redis
        result = cache.get_alerts("ERCOT")

        assert result is not None
        assert result["region"] == "ERCOT"
        assert result["stress_score"] == 45
        assert result["stress_label"] == "Elevated"
        assert len(result["alerts"]) == 2


class TestNewsCache:

    def test_get_news_returns_data(self, populated_redis, sample_news):
        """get_news returns news articles."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = populated_redis
        result = cache.get_news()

        assert result is not None
        assert "articles" in result
        assert len(result["articles"]) == 1

    def test_get_news_returns_none_when_empty(self, mock_redis):
        """get_news returns None when no data."""
        from src.api.cache import ForecastCache

        cache = ForecastCache()
        cache.client = mock_redis
        assert cache.get_news() is None
