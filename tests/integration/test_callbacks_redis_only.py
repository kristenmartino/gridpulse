"""Integration tests for the REQUIRE_REDIS gates in components.callbacks.

These tests assert that when ``REQUIRE_REDIS=True`` and Redis returns no
data, the callbacks surface a "warming" state rather than running
synchronous data fetches or inline model training.

The three gates covered:
- ``_run_forecast_outlook`` at components/callbacks.py:518
- ``load_data`` (registered callback) at components/callbacks.py:2038
- ``_run_backtest_for_horizon`` at components/callbacks.py:5947
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


def _sample_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.date_range("2024-01-01", periods=24 * 30, freq="h", tz="UTC")
    n = len(ts)
    demand = pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": 40_000 + 5000 * np.sin(2 * np.pi * np.arange(n) / 24),
            "region": "ERCOT",
        }
    )
    weather = pd.DataFrame(
        {
            "timestamp": ts,
            "temperature_2m": 70.0,
            "apparent_temperature": 70.0,
            "relative_humidity_2m": 60.0,
            "dew_point_2m": 50.0,
            "wind_speed_10m": 8.0,
            "wind_speed_80m": 12.0,
            "wind_speed_120m": 15.0,
            "wind_direction_10m": 180.0,
            "shortwave_radiation": 100.0,
            "direct_normal_irradiance": 0.0,
            "diffuse_radiation": 0.0,
            "cloud_cover": 40.0,
            "precipitation": 0.0,
            "snowfall": 0.0,
            "surface_pressure": 1013.0,
            "soil_temperature_0cm": 65.0,
            "weather_code": 0,
        }
    )
    return demand, weather


@pytest.fixture
def clear_caches():
    """Clear in-memory callback caches between tests so gate assertions are stable."""
    import components.callbacks as cbs

    cbs._PREDICTION_CACHE.clear()
    cbs._BACKTEST_CACHE.clear()
    cbs._MODEL_CACHE.clear()
    yield
    cbs._PREDICTION_CACHE.clear()
    cbs._BACKTEST_CACHE.clear()
    cbs._MODEL_CACHE.clear()


@pytest.fixture
def empty_sqlite_cache(monkeypatch, tmp_path):
    """Force the SQLite cache to return None for every key."""
    import data.cache as cache_mod

    class _EmptyCache:
        def get(self, key):
            return None

        def set(self, key, value, ttl=None):
            return None

    monkeypatch.setattr(cache_mod, "get_cache", lambda: _EmptyCache())


class TestForecastOutlookWarming:
    def test_returns_warming_when_require_redis_true(
        self, monkeypatch, clear_caches, empty_sqlite_cache
    ) -> None:
        """With REQUIRE_REDIS=True and a cache miss, _run_forecast_outlook returns warming."""
        import components.callbacks as cbs

        monkeypatch.setattr(cbs, "REQUIRE_REDIS", True)

        # Also patch fetchers to explode — a warming return means they
        # must NOT have been called.
        import data.eia_client as eia
        import data.weather_client as weather

        def _boom(*a, **kw):
            raise AssertionError("fetch called during warming state")

        monkeypatch.setattr(eia, "fetch_demand", _boom)
        monkeypatch.setattr(weather, "fetch_weather", _boom)

        demand_df, weather_df = _sample_frames()
        result = cbs._run_forecast_outlook(demand_df, weather_df, 24, "xgboost", "ERCOT")

        assert isinstance(result, dict)
        assert result.get("status") == "warming"
        assert result.get("error") == "warming"

    def test_returns_predictions_when_require_redis_false(
        self, monkeypatch, clear_caches, empty_sqlite_cache
    ) -> None:
        """With REQUIRE_REDIS=False the v1 compute fallback runs (we stub training)."""
        import components.callbacks as cbs

        monkeypatch.setattr(cbs, "REQUIRE_REDIS", False)

        # Stub out training + prediction so the test doesn't need xgboost.
        import models.xgboost_model as xgb_mod

        monkeypatch.setattr(
            xgb_mod,
            "train_xgboost",
            lambda df: {"model": "fake", "feature_importances": {}},
        )
        monkeypatch.setattr(
            xgb_mod,
            "predict_xgboost",
            lambda model, x: np.full(len(x), 40_000.0),
        )

        demand_df, weather_df = _sample_frames()
        result = cbs._run_forecast_outlook(demand_df, weather_df, 24, "xgboost", "ERCOT")
        # Not warming — either a real result or a different fallback error.
        assert result.get("status") != "warming"


class TestBacktestWarming:
    def test_returns_warming_when_require_redis_true(
        self, monkeypatch, clear_caches, empty_sqlite_cache
    ) -> None:
        import components.callbacks as cbs

        monkeypatch.setattr(cbs, "REQUIRE_REDIS", True)

        demand_df, weather_df = _sample_frames()
        result = cbs._run_backtest_for_horizon(
            demand_df,
            weather_df,
            24,
            "xgboost",
            "ERCOT",
            "forecast_exog",
        )
        assert isinstance(result, dict)
        assert result.get("error") == "warming"


class TestLoadDataWarming:
    def test_load_data_returns_warming_empty_frames(self, monkeypatch, clear_caches) -> None:
        """When REQUIRE_REDIS=True and Redis returns None, load_data returns empty-frame warming."""
        import components.callbacks as cbs

        monkeypatch.setattr(cbs, "REQUIRE_REDIS", True)
        # Redis fast path returns None → triggers the warming gate.
        monkeypatch.setattr(cbs, "_load_data_from_redis", lambda region: None)

        # Fetchers must not be called.
        import data.eia_client as eia
        import data.weather_client as weather

        def _boom(*a, **kw):
            raise AssertionError("fetch called during warming state")

        monkeypatch.setattr(eia, "fetch_demand", _boom)
        monkeypatch.setattr(weather, "fetch_weather", _boom)

        # load_data is a closure inside register_callbacks. Rebuild a minimal
        # Dash app so we can look it up in callback_map.
        import dash
        import dash_bootstrap_components as dbc

        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            suppress_callback_exceptions=True,
        )
        from components.layout import build_layout

        app.layout = build_layout()
        cbs.register_callbacks(app)

        load_data = None
        for _key, val in app.callback_map.items():
            fn = val.get("callback")
            if fn and getattr(fn, "__name__", "") == "load_data":
                load_data = getattr(fn, "__wrapped__", fn)
                break
        assert load_data is not None, "load_data callback not found"

        result = load_data("ERCOT", 0)
        # Return shape is (demand_json, weather_json, freshness_json, audit_json, pipeline_json)
        assert isinstance(result, tuple)
        assert len(result) == 5
        demand_json, weather_json, freshness_json, _, _ = result

        # Frames are empty
        demand_parsed = json.loads(demand_json)
        assert demand_parsed.get("data") in (None, []) or len(demand_parsed.get("data", [])) == 0
        weather_parsed = json.loads(weather_json)
        assert weather_parsed.get("data") in (None, []) or len(weather_parsed.get("data", [])) == 0

        # Freshness says warming across sources
        freshness = json.loads(freshness_json)
        assert freshness.get("demand") == "warming"
        assert freshness.get("weather") == "warming"
        assert freshness.get("alerts") == "warming"

    def test_load_data_uses_redis_when_available(self, monkeypatch, clear_caches) -> None:
        """When Redis has data, the warming gate is bypassed and Redis values are used."""
        import components.callbacks as cbs

        monkeypatch.setattr(cbs, "REQUIRE_REDIS", True)

        sentinel = ("demand-json", "weather-json", "freshness-json", "{}", "{}")
        monkeypatch.setattr(cbs, "_load_data_from_redis", lambda region: sentinel)

        import dash
        import dash_bootstrap_components as dbc

        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            suppress_callback_exceptions=True,
        )
        from components.layout import build_layout

        app.layout = build_layout()
        cbs.register_callbacks(app)

        load_data = None
        for _key, val in app.callback_map.items():
            fn = val.get("callback")
            if fn and getattr(fn, "__name__", "") == "load_data":
                load_data = getattr(fn, "__wrapped__", fn)
                break
        assert load_data is not None

        result = load_data("ERCOT", 0)
        assert result == sentinel
