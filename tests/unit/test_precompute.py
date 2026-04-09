"""Tests for startup precomputation module."""

import time

import pytest

from components.callbacks import _BACKTEST_CACHE, _MODEL_CACHE, _PREDICTION_CACHE

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear all in-memory caches before and after each test."""
    from precompute import _region_data

    _MODEL_CACHE.clear()
    _PREDICTION_CACHE.clear()
    _BACKTEST_CACHE.clear()
    _region_data.clear()
    yield
    _MODEL_CACHE.clear()
    _PREDICTION_CACHE.clear()
    _BACKTEST_CACHE.clear()
    _region_data.clear()


@pytest.fixture()
def precompute_region():
    """Fetch data + train model + predictions for a single region."""
    from precompute import _fetch_data, _region_data, _train_region

    def _run(region: str):
        demand_df, weather_df = _fetch_data(region)
        assert demand_df is not None
        _region_data[region] = (demand_df, weather_df)
        _train_region(region)

    return _run


class TestPrecomputeRegion:
    """Test precomputation for a single region using demo data."""

    def test_precompute_populates_model_cache(self, precompute_region):
        """After precomputing, _MODEL_CACHE has an entry for the region."""
        precompute_region("ERCOT")
        assert ("ERCOT", "xgboost", 0) in _MODEL_CACHE

    def test_precompute_populates_prediction_cache(self, precompute_region):
        """After precomputing, _PREDICTION_CACHE has entries for all horizons."""
        precompute_region("ERCOT")
        assert ("ERCOT", 24, "xgboost") in _PREDICTION_CACHE
        assert ("ERCOT", 168, "xgboost") in _PREDICTION_CACHE
        assert ("ERCOT", 720, "xgboost") in _PREDICTION_CACHE

    def test_precompute_with_backtest(self, precompute_region):
        """Backtest results are cached after _precompute_backtest."""
        from precompute import _precompute_backtest

        precompute_region("ERCOT")
        _precompute_backtest("ERCOT", 24)
        _precompute_backtest("ERCOT", 168)
        assert ("ERCOT", 24, "xgboost", "forecast_exog") in _BACKTEST_CACHE
        assert ("ERCOT", 168, "xgboost", "forecast_exog") in _BACKTEST_CACHE

    def test_precompute_without_backtest(self, precompute_region):
        """Backtest cache stays empty when backtests not run."""
        precompute_region("ERCOT")
        assert len(_BACKTEST_CACHE) == 0

    def test_model_cache_structure(self, precompute_region):
        """Model cache entries have the expected (model, hash, timestamp) structure."""
        precompute_region("FPL")
        model, data_hash, ts = _MODEL_CACHE[("FPL", "xgboost", 0)]
        assert model is not None
        assert isinstance(data_hash, int)
        assert isinstance(ts, float)
        assert ts <= time.time()

    def test_prediction_cache_structure(self, precompute_region):
        """Prediction cache entries have timestamps and predictions arrays."""
        precompute_region("FPL")
        predictions, timestamps, data_hash, ts = _PREDICTION_CACHE[("FPL", 24, "xgboost")]
        assert len(predictions) == 24
        assert len(timestamps) == 24


class TestPrecomputeDisabled:
    """Test that precompute respects the PRECOMPUTE_ENABLED flag."""

    def test_disabled_via_env_var(self, monkeypatch):
        """PRECOMPUTE_ENABLED=false skips all precomputation."""
        monkeypatch.setenv("PRECOMPUTE_ENABLED", "false")
        import importlib

        import config

        importlib.reload(config)
        assert config.PRECOMPUTE_ENABLED is False
        # Restore
        monkeypatch.setenv("PRECOMPUTE_ENABLED", "false")
        importlib.reload(config)


class TestPrecomputeFetchData:
    """Test data fetching with fallback to demo data."""

    def test_fetch_data_returns_dataframes(self):
        """_fetch_data returns (demand_df, weather_df) with demo data."""
        from precompute import _fetch_data

        demand_df, weather_df = _fetch_data("ERCOT")
        assert demand_df is not None
        assert weather_df is not None
        assert len(demand_df) > 0
        assert len(weather_df) > 0
        assert "timestamp" in demand_df.columns
        assert "demand_mw" in demand_df.columns
        assert "timestamp" in weather_df.columns

    def test_fetch_data_uses_demo_fallback(self):
        """_fetch_data falls back to demo data when no API key."""
        from precompute import _fetch_data

        demand_df, weather_df = _fetch_data("ERCOT")
        assert demand_df is not None
        assert weather_df is not None
        assert "demand_mw" in demand_df.columns


class TestPrecomputeAll:
    """Test the full precompute_all entry point."""

    def test_precompute_all_never_raises(self):
        """precompute_all() never raises, even with errors."""
        from precompute import precompute_all

        precompute_all()
        # All regions should be precomputed
        assert len(_MODEL_CACHE) >= 1
