"""
Tests for v1 compute fallback paths in callback closures and remaining
untested module-level functions in components/callbacks.py.

Covers the following uncovered line ranges:
- 322-453: ensemble forecast in _run_forecast_outlook (ThreadPoolExecutor)
- 1548-1673: load_data() v1 fallback (EIA API -> demo -> error fallback)
- 1705-1734: switch_persona() callback
- 1937-2075: update_weather_tab() v1 fallback
- 2094-2210: update_models_tab() v1 fallback
- 2235-2431: update_generation_tab() v1 fallback
- 2463-2622: update_alerts_tab() v1 fallback
- 2674-2790: run_scenario() callback (simulator)
- 3146-3280: update_demand_outlook() v1 fallback
- 3322-3508: update_backtest_chart() v1 fallback

Also covers module-level helpers:
- _run_forecast_outlook
- _run_backtest_for_horizon
- _predict_single_fold
- _ensemble_fold
"""

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from dash import html, no_update

# ---------------------------------------------------------------------------
# Shared test data helpers
# ---------------------------------------------------------------------------


def _demand_json(n=168, base_mw=35000):
    """Create realistic demand JSON for callback inputs."""
    start = datetime(2024, 6, 1, tzinfo=UTC)
    ts = pd.date_range(start, periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    demand = base_mw + 5000 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 300, n)
    df = pd.DataFrame({"timestamp": ts, "demand_mw": demand, "region": "FPL"})
    return df.to_json(date_format="iso")


def _weather_json(n=168):
    """Create realistic weather JSON for callback inputs."""
    start = datetime(2024, 6, 1, tzinfo=UTC)
    ts = pd.date_range(start, periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "temperature_2m": 85 + rng.normal(0, 5, n),
            "wind_speed_80m": 12 + rng.normal(0, 3, n),
            "shortwave_radiation": np.maximum(0, 400 * np.sin(2 * np.pi * np.arange(n) / 24)),
            "relative_humidity_2m": 60 + rng.normal(0, 10, n),
            "cloud_cover": 50 + rng.normal(0, 15, n),
            "surface_pressure": 1013 + rng.normal(0, 3, n),
        }
    )
    return df.to_json(date_format="iso")


def _demand_df(n=200):
    """Create a demand DataFrame (not JSON)."""
    start = datetime(2024, 6, 1, tzinfo=UTC)
    ts = pd.date_range(start, periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    demand = 35000 + 5000 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 300, n)
    return pd.DataFrame({"timestamp": ts, "demand_mw": demand})


def _weather_df(n=200):
    """Create a weather DataFrame (not JSON)."""
    start = datetime(2024, 6, 1, tzinfo=UTC)
    ts = pd.date_range(start, periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "temperature_2m": 85 + rng.normal(0, 5, n),
            "wind_speed_80m": 12 + rng.normal(0, 3, n),
            "shortwave_radiation": np.maximum(0, 400 * np.sin(2 * np.pi * np.arange(n) / 24)),
            "relative_humidity_2m": 60 + rng.normal(0, 10, n),
            "cloud_cover": 50 + rng.normal(0, 15, n),
            "surface_pressure": 1013 + rng.normal(0, 3, n),
        }
    )


def _train_df(n=800):
    """Create a full featured training DataFrame for model tests."""
    start = datetime(2024, 1, 1, tzinfo=UTC)
    ts = pd.date_range(start, periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": 35000
            + 5000 * np.sin(2 * np.pi * np.arange(n) / 24)
            + rng.normal(0, 300, n),
            "hour": ts.hour,
            "day_of_week": ts.dayofweek,
            "month": ts.month,
            "day_of_year": ts.dayofyear,
            "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
            "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
            "dow_sin": np.sin(2 * np.pi * ts.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * ts.dayofweek / 7),
            "is_weekend": (ts.dayofweek >= 5).astype(int),
            "temperature_2m": 85 + rng.normal(0, 5, n),
            "wind_speed_80m": 12 + rng.normal(0, 3, n),
            "shortwave_radiation": np.maximum(0, 400 * np.sin(2 * np.pi * np.arange(n) / 24)),
            "cooling_degree_days": rng.uniform(0, 30, n),
            "heating_degree_days": rng.uniform(0, 5, n),
        }
    )


# ---------------------------------------------------------------------------
# Fixtures: registered app + callback extraction
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registered_app():
    """Create a Dash app with all callbacks registered."""
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
    )

    from components.layout import build_layout

    app.layout = build_layout()

    from components.callbacks import register_callbacks

    register_callbacks(app)
    return app


@pytest.fixture(scope="module")
def callbacks(registered_app):
    """Extract unwrapped callback functions into a dict keyed by function name."""
    fns = {}
    for _key, val in registered_app.callback_map.items():
        fn = val.get("callback")
        if fn and hasattr(fn, "__name__"):
            raw = getattr(fn, "__wrapped__", fn)
            fns[fn.__name__] = raw
    return fns


# ===================================================================
# TestRunForecastOutlook: module-level function _run_forecast_outlook
# ===================================================================


class TestRunForecastOutlook:
    """Tests for _run_forecast_outlook (lines 217-489)."""

    def test_prediction_cache_hit(self):
        """When prediction cache has valid entry, return immediately."""
        import time

        from components.callbacks import _run_forecast_outlook

        demand = _demand_df(200)
        weather = _weather_df(200)

        # Pre-compute the hash the function will use
        from components.callbacks import _compute_data_hash

        data_hash = _compute_data_hash(demand, weather, "FPL")
        cache_key = ("FPL", 24, "xgboost")
        fake_preds = np.array([40000.0] * 24)
        fake_ts = pd.date_range("2024-06-10", periods=24, freq="h")

        with patch.dict(
            "components.callbacks._PREDICTION_CACHE",
            {cache_key: (fake_preds, fake_ts, data_hash, time.time())},
        ):
            result = _run_forecast_outlook(demand, weather, 24, "xgboost", "FPL")

        assert "error" not in result
        assert "predictions" in result
        np.testing.assert_array_equal(result["predictions"], fake_preds)

    def test_sqlite_cache_hit(self):
        """When in-memory cache misses but SQLite has data, return from SQLite."""
        from components.callbacks import _CACHE_VERSION, _compute_data_hash, _run_forecast_outlook

        demand = _demand_df(200)
        weather = _weather_df(200)
        data_hash = _compute_data_hash(demand, weather, "FPL")

        sqlite_result = {
            "cache_version": _CACHE_VERSION,
            "data_hash": data_hash,
            "timestamps": ["2024-06-10T00:00:00"] * 24,
            "predictions": [40000.0] * 24,
        }

        mock_cache = MagicMock()
        mock_cache.get.return_value = sqlite_result

        with (
            patch.dict("components.callbacks._PREDICTION_CACHE", {}, clear=True),
            patch("components.callbacks.get_cache", return_value=mock_cache, create=True),
            patch("data.cache.get_cache", return_value=mock_cache),
        ):
            result = _run_forecast_outlook(demand, weather, 24, "xgboost", "FPL")

        assert "error" not in result
        assert "predictions" in result

    def test_xgboost_path(self):
        """Test XGBoost training and prediction path."""
        from components.callbacks import _run_forecast_outlook

        demand = _demand_df(200)
        weather = _weather_df(200)
        featured = _train_df(200)

        fake_model = {"model": "xgb"}
        mock_sqlite = MagicMock()
        mock_sqlite.get.return_value = None

        with (
            patch.dict("components.callbacks._PREDICTION_CACHE", {}, clear=True),
            patch.dict("components.callbacks._MODEL_CACHE", {}, clear=True),
            patch("data.cache.get_cache", return_value=mock_sqlite),
            patch("data.preprocessing.merge_demand_weather", return_value=featured),
            patch("data.feature_engineering.engineer_features", return_value=featured),
            patch("models.xgboost_model.train_xgboost", return_value=fake_model) as mock_train,
            patch(
                "models.xgboost_model.predict_xgboost",
                return_value=np.array([40000.0] * 48),
            ) as mock_predict,
        ):
            result = _run_forecast_outlook(demand, weather, 24, "xgboost", "FPL")

        assert "error" not in result
        assert len(result["predictions"]) == 24
        mock_train.assert_called_once()
        mock_predict.assert_called_once()

    def test_prophet_path(self):
        """Test Prophet training and prediction path."""
        from components.callbacks import _run_forecast_outlook

        demand = _demand_df(200)
        weather = _weather_df(200)
        featured = _train_df(200)

        fake_model = MagicMock()
        mock_sqlite = MagicMock()
        mock_sqlite.get.return_value = None

        prophet_result = {"forecast": np.array([39000.0] * 48)}

        with (
            patch.dict("components.callbacks._PREDICTION_CACHE", {}, clear=True),
            patch.dict("components.callbacks._MODEL_CACHE", {}, clear=True),
            patch("data.cache.get_cache", return_value=mock_sqlite),
            patch("data.preprocessing.merge_demand_weather", return_value=featured),
            patch("data.feature_engineering.engineer_features", return_value=featured),
            patch("models.prophet_model.train_prophet", return_value=fake_model),
            patch("models.prophet_model.predict_prophet", return_value=prophet_result),
        ):
            result = _run_forecast_outlook(demand, weather, 24, "prophet", "FPL")

        assert "error" not in result
        assert len(result["predictions"]) == 24

    def test_arima_path(self):
        """Test ARIMA training and prediction path."""
        from components.callbacks import _run_forecast_outlook

        demand = _demand_df(200)
        weather = _weather_df(200)
        featured = _train_df(200)

        fake_model = MagicMock()
        mock_sqlite = MagicMock()
        mock_sqlite.get.return_value = None

        with (
            patch.dict("components.callbacks._PREDICTION_CACHE", {}, clear=True),
            patch.dict("components.callbacks._MODEL_CACHE", {}, clear=True),
            patch("data.cache.get_cache", return_value=mock_sqlite),
            patch("data.preprocessing.merge_demand_weather", return_value=featured),
            patch("data.feature_engineering.engineer_features", return_value=featured),
            patch("models.arima_model.train_arima", return_value=fake_model),
            patch(
                "models.arima_model.predict_arima",
                return_value=np.array([38000.0] * 48),
            ),
        ):
            result = _run_forecast_outlook(demand, weather, 24, "arima", "FPL")

        assert "error" not in result
        assert len(result["predictions"]) == 24

    def test_ensemble_path_concurrent(self):
        """Test ensemble path with ThreadPoolExecutor running all 3 models."""
        from components.callbacks import _run_forecast_outlook

        demand = _demand_df(200)
        weather = _weather_df(200)
        featured = _train_df(200)

        fake_xgb_model = {"model": "xgb"}
        fake_prophet_model = MagicMock()
        fake_arima_model = MagicMock()
        mock_sqlite = MagicMock()
        mock_sqlite.get.return_value = None

        prophet_result = {"forecast": np.array([39000.0] * 48)}

        with (
            patch.dict("components.callbacks._PREDICTION_CACHE", {}, clear=True),
            patch.dict("components.callbacks._MODEL_CACHE", {}, clear=True),
            patch("data.cache.get_cache", return_value=mock_sqlite),
            patch("data.preprocessing.merge_demand_weather", return_value=featured),
            patch("data.feature_engineering.engineer_features", return_value=featured),
            patch("models.xgboost_model.train_xgboost", return_value=fake_xgb_model),
            patch("models.xgboost_model.predict_xgboost", return_value=np.array([40000.0] * 48)),
            patch("models.prophet_model.train_prophet", return_value=fake_prophet_model),
            patch("models.prophet_model.predict_prophet", return_value=prophet_result),
            patch("models.arima_model.train_arima", return_value=fake_arima_model),
            patch("models.arima_model.predict_arima", return_value=np.array([38000.0] * 48)),
        ):
            result = _run_forecast_outlook(demand, weather, 24, "ensemble", "FPL")

        assert "error" not in result
        assert len(result["predictions"]) == 24
        # Ensemble should average the three model predictions
        expected_mean = np.mean([40000.0, 39000.0, 38000.0])
        np.testing.assert_allclose(result["predictions"][0], expected_mean, rtol=0.01)

    def test_unknown_model_returns_error(self):
        """Unknown model name should return an error dict."""
        from components.callbacks import _run_forecast_outlook

        demand = _demand_df(200)
        weather = _weather_df(200)
        featured = _train_df(200)
        mock_sqlite = MagicMock()
        mock_sqlite.get.return_value = None

        with (
            patch.dict("components.callbacks._PREDICTION_CACHE", {}, clear=True),
            patch("data.cache.get_cache", return_value=mock_sqlite),
            patch("data.preprocessing.merge_demand_weather", return_value=featured),
            patch("data.feature_engineering.engineer_features", return_value=featured),
        ):
            result = _run_forecast_outlook(demand, weather, 24, "unknown_model", "FPL")

        assert "error" in result
        assert "Unknown model" in result["error"]

    def test_exception_handling_returns_error(self):
        """If merge_demand_weather raises inside the try block, return an error dict."""
        from components.callbacks import _run_forecast_outlook

        demand = _demand_df(200)
        weather = _weather_df(200)
        # Return a valid merged df so we get past merge, but have engineer_features
        # raise to trigger the outer exception handler.
        merged = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-06-01", periods=200, freq="h"),
                "demand_mw": np.random.uniform(35000, 45000, 200),
            }
        )
        mock_sqlite = MagicMock()
        mock_sqlite.get.return_value = None

        with (
            patch.dict("components.callbacks._PREDICTION_CACHE", {}, clear=True),
            patch("data.cache.get_cache", return_value=mock_sqlite),
            patch("data.preprocessing.merge_demand_weather", return_value=merged),
            patch(
                "data.feature_engineering.engineer_features",
                return_value=merged,
            ),
            # Model training raises -> caught by the inner try/except
            patch(
                "models.xgboost_model.train_xgboost",
                side_effect=ValueError("training exploded"),
            ),
        ):
            result = _run_forecast_outlook(demand, weather, 24, "xgboost", "FPL")

        assert "error" in result

    def test_insufficient_data_returns_error(self):
        """If featured data has fewer than 168 rows, return error."""
        from components.callbacks import _run_forecast_outlook

        # Only 50 rows - should be insufficient
        demand = _demand_df(50)
        weather = _weather_df(50)
        mock_sqlite = MagicMock()
        mock_sqlite.get.return_value = None

        # Return a small df that will be < 168 after dropna
        small_merged = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-06-01", periods=50, freq="h"),
                "demand_mw": np.random.uniform(35000, 45000, 50),
            }
        )

        with (
            patch.dict("components.callbacks._PREDICTION_CACHE", {}, clear=True),
            patch("data.cache.get_cache", return_value=mock_sqlite),
            patch("data.preprocessing.merge_demand_weather", return_value=small_merged),
            patch("data.feature_engineering.engineer_features", return_value=small_merged),
        ):
            result = _run_forecast_outlook(demand, weather, 24, "xgboost", "FPL")

        assert "error" in result
        assert "Insufficient" in result["error"]


# ===================================================================
# TestPredictSingleFold
# ===================================================================


class TestPredictSingleFold:
    """Tests for _predict_single_fold helper."""

    def test_xgboost_fold(self):
        """XGBoost path trains and predicts."""
        from components.callbacks import _predict_single_fold

        train = _train_df(100)
        test = _train_df(24)

        fake_model = {"model": "xgb"}
        with (
            patch("models.xgboost_model.train_xgboost", return_value=fake_model),
            patch(
                "models.xgboost_model.predict_xgboost",
                return_value=np.array([40000.0] * 30),
            ),
        ):
            result = _predict_single_fold("xgboost", train, test)

        assert result is not None
        assert len(result) == 24

    def test_prophet_fold(self):
        """Prophet path trains and predicts."""
        from components.callbacks import _predict_single_fold

        train = _train_df(100)
        test = _train_df(24)
        fake_model = MagicMock()
        prophet_result = {"forecast": np.array([39000.0] * 30)}

        with (
            patch("models.prophet_model.train_prophet", return_value=fake_model),
            patch("models.prophet_model.predict_prophet", return_value=prophet_result),
        ):
            result = _predict_single_fold("prophet", train, test)

        assert result is not None
        assert len(result) == 24

    def test_arima_fold(self):
        """ARIMA path trains and predicts."""
        from components.callbacks import _predict_single_fold

        train = _train_df(100)
        test = _train_df(24)
        fake_model = MagicMock()

        with (
            patch("models.arima_model.train_arima", return_value=fake_model),
            patch(
                "models.arima_model.predict_arima",
                return_value=np.array([38000.0] * 30),
            ),
        ):
            result = _predict_single_fold("arima", train, test)

        assert result is not None
        assert len(result) == 24

    def test_unknown_model_returns_none(self):
        """Unknown model returns None."""
        from components.callbacks import _predict_single_fold

        result = _predict_single_fold("bogus_model", _train_df(100), _train_df(24))
        assert result is None


# ===================================================================
# TestEnsembleFold
# ===================================================================


class TestEnsembleFold:
    """Tests for _ensemble_fold helper."""

    def test_all_models_succeed(self):
        """When all models succeed, returns weighted ensemble."""
        from components.callbacks import _ensemble_fold

        train = _train_df(100)
        test = _train_df(24)

        def mock_predict_single(name, tr, te, **kwargs):
            base = {"xgboost": 40000.0, "prophet": 39000.0, "arima": 38000.0}
            return np.full(len(te), base.get(name, 39000.0))

        with (
            patch("components.callbacks._predict_single_fold", side_effect=mock_predict_single),
            patch("models.evaluation.compute_mape", return_value=5.0),
        ):
            result = _ensemble_fold(train, test)

        assert result is not None
        assert len(result) == 24

    def test_all_models_fail_returns_none(self):
        """When all models fail, returns None."""
        from components.callbacks import _ensemble_fold

        train = _train_df(100)
        test = _train_df(24)

        with patch(
            "components.callbacks._predict_single_fold",
            side_effect=ValueError("model failed"),
        ):
            result = _ensemble_fold(train, test)

        assert result is None


# ===================================================================
# TestRunBacktestForHorizon
# ===================================================================


class TestRunBacktestForHorizon:
    """Tests for _run_backtest_for_horizon (lines 3891-4088)."""

    def test_in_memory_cache_hit(self):
        """In-memory cache returns cached result."""
        import time

        from components.callbacks import _compute_data_hash, _run_backtest_for_horizon

        demand = _demand_df(200)
        weather = _weather_df(200)
        data_hash = _compute_data_hash(demand, weather, "FPL")
        cache_key = ("FPL", 24, "xgboost", "forecast_exog")
        cached = {"timestamps": [], "actual": [], "predictions": [], "metrics": {"mape": 5.0}}

        with patch.dict(
            "components.callbacks._BACKTEST_CACHE",
            {cache_key: (cached, data_hash, time.time())},
        ):
            result = _run_backtest_for_horizon(demand, weather, 24, "xgboost", "FPL")

        assert result is cached

    def test_sqlite_cache_hit(self):
        """SQLite cache returns cached result."""
        from components.callbacks import (
            _CACHE_VERSION,
            _compute_data_hash,
            _run_backtest_for_horizon,
        )

        demand = _demand_df(200)
        weather = _weather_df(200)
        data_hash = _compute_data_hash(demand, weather, "FPL")

        sqlite_result = {
            "cache_version": _CACHE_VERSION,
            "data_hash": data_hash,
            "timestamps": ["2024-06-10T00:00:00"] * 24,
            "actual": [40000.0] * 24,
            "predictions": [39000.0] * 24,
            "metrics": {"mape": 3.0, "rmse": 500, "mae": 400, "r2": 0.95},
        }
        mock_cache = MagicMock()
        mock_cache.get.return_value = sqlite_result

        with (
            patch.dict("components.callbacks._BACKTEST_CACHE", {}, clear=True),
            patch("data.cache.get_cache", return_value=mock_cache),
        ):
            result = _run_backtest_for_horizon(demand, weather, 24, "xgboost", "FPL")

        assert "error" not in result
        assert "actual" in result

    def test_insufficient_data_returns_error(self):
        """When data is too small, return error."""
        from components.callbacks import _run_backtest_for_horizon

        demand = _demand_df(50)
        weather = _weather_df(50)
        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        small_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-06-01", periods=50, freq="h"),
                "demand_mw": np.random.uniform(35000, 45000, 50),
            }
        )

        with (
            patch.dict("components.callbacks._BACKTEST_CACHE", {}, clear=True),
            patch("data.cache.get_cache", return_value=mock_cache),
            patch("data.preprocessing.merge_demand_weather", return_value=small_df),
            patch("data.feature_engineering.engineer_features", return_value=small_df),
        ):
            result = _run_backtest_for_horizon(demand, weather, 24, "xgboost", "FPL")

        assert "error" in result


# ===================================================================
# TestLoadDataV1: load_data callback closure
# ===================================================================


class TestLoadDataV1:
    """Tests for the load_data callback v1 fallback (lines 1548-1679)."""

    def test_no_api_key_uses_demo(self, callbacks):
        """When EIA_API_KEY is empty, demo data is used."""
        with (
            patch("components.callbacks.EIA_API_KEY", ""),
            patch("components.callbacks.redis_get", return_value=None),
        ):
            result = callbacks["load_data"]("FPL", 0)

        assert len(result) == 5
        demand_json, weather_json, freshness_json, audit_json, pipeline_json = result
        assert demand_json is not None
        freshness = json.loads(freshness_json)
        assert freshness["demand"] == "demo"
        assert freshness["weather"] == "demo"

    def test_api_key_fetches_demand_and_weather(self, callbacks):
        """When EIA_API_KEY is set, fetch_demand and fetch_weather are called."""
        fake_demand = _demand_df(168)
        fake_weather = _weather_df(168)

        with (
            patch("components.callbacks.EIA_API_KEY", "test_key_12345"),
            patch("components.callbacks.redis_get", return_value=None),
            patch("data.eia_client.fetch_demand", return_value=fake_demand),
            patch("data.weather_client.fetch_weather", return_value=fake_weather),
        ):
            result = callbacks["load_data"]("FPL", 0)

        demand_json, weather_json, freshness_json, audit_json, pipeline_json = result
        freshness = json.loads(freshness_json)
        assert freshness["demand"] == "fresh"
        assert freshness["weather"] == "fresh"

    def test_demand_api_failure_falls_to_demo(self, callbacks):
        """When fetch_demand raises, demo data is used for demand."""
        fake_weather = _weather_df(168)

        with (
            patch("components.callbacks.EIA_API_KEY", "test_key_12345"),
            patch("components.callbacks.redis_get", return_value=None),
            patch("data.eia_client.fetch_demand", side_effect=ConnectionError("timeout")),
            patch("data.weather_client.fetch_weather", return_value=fake_weather),
        ):
            result = callbacks["load_data"]("FPL", 0)

        freshness = json.loads(result[2])
        assert freshness["demand"] == "stale"
        assert freshness["weather"] == "fresh"

    def test_weather_api_failure_falls_to_demo(self, callbacks):
        """When fetch_weather raises, demo data is used for weather."""
        fake_demand = _demand_df(168)

        with (
            patch("components.callbacks.EIA_API_KEY", "test_key_12345"),
            patch("components.callbacks.redis_get", return_value=None),
            patch("data.eia_client.fetch_demand", return_value=fake_demand),
            patch("data.weather_client.fetch_weather", side_effect=ConnectionError("timeout")),
        ):
            result = callbacks["load_data"]("FPL", 0)

        freshness = json.loads(result[2])
        assert freshness["demand"] == "fresh"
        assert freshness["weather"] == "stale"

    def test_empty_demand_fallback_to_demo(self, callbacks):
        """When fetch_demand returns empty DataFrame, demo is used."""
        fake_weather = _weather_df(168)

        with (
            patch("components.callbacks.EIA_API_KEY", "test_key_12345"),
            patch("components.callbacks.redis_get", return_value=None),
            patch("data.eia_client.fetch_demand", return_value=pd.DataFrame()),
            patch("data.weather_client.fetch_weather", return_value=fake_weather),
        ):
            result = callbacks["load_data"]("FPL", 0)

        freshness = json.loads(result[2])
        assert freshness["demand"] == "stale"

    def test_total_failure_falls_to_error_demo(self, callbacks):
        """When audit_trail.record_forecast raises, error fallback produces demo."""
        with (
            patch("components.callbacks.EIA_API_KEY", ""),
            patch("components.callbacks.redis_get", return_value=None),
            patch(
                "data.demo_data.generate_demo_demand",
                side_effect=[ValueError("first call broken"), _demand_df(168)],
            ),
            patch("data.demo_data.generate_demo_weather", return_value=_weather_df(168)),
        ):
            result = callbacks["load_data"]("FPL", 0)

        freshness = json.loads(result[2])
        assert freshness["demand"] == "error"
        assert freshness["weather"] == "error"


# ===================================================================
# TestSwitchPersona: switch_persona callback closure
# ===================================================================


class TestSwitchPersona:
    """Tests for the switch_persona callback (lines 1698-1734)."""

    def test_basic_persona_switch(self, callbacks):
        """Basic persona switch returns welcome card, KPIs, and tab."""
        mock_ctx = MagicMock()
        mock_ctx.triggered_id = "persona-selector"

        with (
            patch("components.callbacks.ctx", mock_ctx),
            patch("precompute._region_data", {}, create=True),
        ):
            # 5th arg = current_tab (State)
            result = callbacks["switch_persona"]("grid_ops", "FPL", None, None, "tab-forecast")

        assert len(result) == 3
        welcome, kpis, active_tab = result
        assert welcome is not None
        assert kpis is not None
        # persona-selector triggered, so active_tab should be set
        assert active_tab is not no_update

    def test_region_change_does_not_switch_tab(self, callbacks):
        """When region-selector triggers, active_tab should be no_update."""
        mock_ctx = MagicMock()
        mock_ctx.triggered_id = "region-selector"

        with (
            patch("components.callbacks.ctx", mock_ctx),
            patch("precompute._region_data", {}, create=True),
        ):
            result = callbacks["switch_persona"]("grid_ops", "FPL", None, None, "tab-forecast")

        welcome, kpis, active_tab = result
        assert active_tab is no_update

    def test_with_demand_data(self, callbacks):
        """When demand_json is provided but not in precompute cache."""
        mock_ctx = MagicMock()
        mock_ctx.triggered_id = "region-selector"

        with (
            patch("components.callbacks.ctx", mock_ctx),
            patch("precompute._region_data", {}, create=True),
        ):
            result = callbacks["switch_persona"](
                "renewables", "FPL", _demand_json(), _weather_json(), "tab-forecast"
            )

        assert len(result) == 3
        assert result[0] is not None


# ===================================================================
# TestWeatherTabV1: update_weather_tab callback closure
# ===================================================================


class TestWeatherTabV1:
    """Tests for update_weather_tab v1 fallback (lines 1937-2075)."""

    def test_tab_guard_returns_no_update(self, callbacks):
        """When active_tab is not tab-weather, return no_update list."""
        result = callbacks["update_weather_tab"](
            _demand_json(), _weather_json(), "tab-forecast", "FPL"
        )
        assert result == [no_update] * 6

    def test_no_data_returns_empty_figures(self, callbacks):
        """When demand/weather are empty, return empty figures."""
        result = callbacks["update_weather_tab"](None, None, "tab-weather", "FPL")
        assert len(result) == 6
        # Each should be a go.Figure
        for fig in result:
            assert isinstance(fig, go.Figure)

    def test_v1_fallback_with_valid_data(self, callbacks):
        """V1 compute fallback produces 6 figures from demand+weather data."""
        with patch("components.callbacks.redis_get", return_value=None):
            result = callbacks["update_weather_tab"](
                _demand_json(), _weather_json(), "tab-weather", "FPL"
            )

        assert len(result) == 6
        for fig in result:
            assert isinstance(fig, go.Figure)


# ===================================================================
# TestModelsTabV1: update_models_tab callback closure
# ===================================================================


class TestModelsTabV1:
    """Tests for update_models_tab v1 fallback (lines 2094-2210)."""

    def test_tab_guard(self, callbacks):
        """When active_tab is not tab-models, return no_update list."""
        result = callbacks["update_models_tab"](_demand_json(), "tab-forecast", ["ensemble"], "FPL")
        assert result == [no_update] * 6

    def test_no_demand_returns_loading(self, callbacks):
        """When demand_json is None, return loading placeholders."""
        result = callbacks["update_models_tab"](None, "tab-models", ["ensemble"], "FPL")
        assert len(result) == 6

    def test_no_model_selected_returns_empty_state(self, callbacks):
        """When no models are selected, callback returns empty-state visuals."""
        result = callbacks["update_models_tab"](_demand_json(), "tab-models", [], "FPL")
        assert len(result) == 6
        assert isinstance(result[0], html.P)
        for fig in result[1:]:
            assert isinstance(fig, go.Figure)

    def test_v1_fallback_with_valid_data(self, callbacks):
        """V1 compute fallback produces table + 5 figures."""
        fake_forecasts = {
            "metrics": {
                "prophet": {"mape": 5.0, "rmse": 500, "mae": 400, "r2": 0.95},
                "arima": {"mape": 6.0, "rmse": 600, "mae": 450, "r2": 0.93},
                "xgboost": {"mape": 4.0, "rmse": 400, "mae": 350, "r2": 0.96},
                "ensemble": {"mape": 3.5, "rmse": 350, "mae": 300, "r2": 0.97},
            },
            "prophet": np.random.uniform(35000, 45000, 168),
            "arima": np.random.uniform(35000, 45000, 168),
            "xgboost": np.random.uniform(35000, 45000, 168),
            "ensemble": np.random.uniform(35000, 45000, 168),
        }

        with (
            patch("components.callbacks.redis_get", return_value=None),
            patch("models.model_service.get_forecasts", return_value=fake_forecasts),
        ):
            result = callbacks["update_models_tab"](
                _demand_json(), "tab-models", ["prophet", "arima", "xgboost", "ensemble"], "FPL"
            )

        assert len(result) == 6
        # First element is a table
        assert isinstance(result[0], html.Table)
        # Remaining are figures
        for fig in result[1:]:
            assert isinstance(fig, go.Figure)

    def test_shap_empty_state_when_xgboost_not_selected(self, callbacks):
        """SHAP panel is intentionally blank when XGBoost is not in model selector."""
        fake_forecasts = {
            "metrics": {"prophet": {"mape": 5.0, "rmse": 500, "mae": 400, "r2": 0.95}},
            "prophet": np.random.uniform(35000, 45000, 168),
        }
        with patch("models.model_service.get_forecasts", return_value=fake_forecasts):
            result = callbacks["update_models_tab"](
                _demand_json(), "tab-models", ["prophet"], "FPL"
            )
        shap_fig = result[5]
        assert isinstance(shap_fig, go.Figure)
        assert shap_fig.layout.annotations


# ===================================================================
# TestGenerationTabV1: update_generation_tab callback closure
# ===================================================================


class TestGenerationTabV1:
    """Tests for update_generation_tab v1 fallback (lines 2235-2431)."""

    def test_tab_guard(self, callbacks):
        """Non-active tab returns no_update."""
        result = callbacks["update_generation_tab"](
            "FPL", "tab-forecast", "grid_ops", 168, _demand_json()
        )
        assert result == [no_update] * 7

    def test_no_region_returns_defaults(self, callbacks):
        """No region returns defaults."""
        result = callbacks["update_generation_tab"](
            None, "tab-generation", "grid_ops", 168, _demand_json()
        )
        assert len(result) == 7

    def test_v1_fallback_with_generation_data(self, callbacks):
        """V1 compute with generation data returns figures and KPIs."""
        n = 168
        ts = pd.date_range("2024-06-01", periods=n, freq="h", tz="UTC")
        gen_rows = []
        for fuel in ["gas", "nuclear", "wind", "solar"]:
            for i in range(n):
                gen_rows.append(
                    {
                        "timestamp": ts[i],
                        "fuel_type": fuel,
                        "generation_mw": np.random.uniform(1000, 5000),
                        "region": "FPL",
                    }
                )
        gen_df = pd.DataFrame(gen_rows)

        with (
            patch("components.callbacks.redis_get", return_value=None),
            patch("components.callbacks._fetch_generation_cached", return_value=gen_df),
        ):
            result = callbacks["update_generation_tab"](
                "FPL", "tab-generation", "grid_ops", 168, _demand_json()
            )

        assert len(result) == 7
        # First two should be figures
        assert isinstance(result[0], go.Figure)
        assert isinstance(result[1], go.Figure)
        # Renewable pct should be a formatted string or "No data"
        assert "%" in result[2] or result[2] == "No data"


# ===================================================================
# TestAlertsTabV1: update_alerts_tab callback closure
# ===================================================================


class TestAlertsTabV1:
    """Tests for update_alerts_tab v1 fallback (lines 2463-2622)."""

    def test_tab_guard(self, callbacks):
        """Non-active tab returns no_update."""
        result = callbacks["update_alerts_tab"](
            "FPL", _demand_json(), _weather_json(), "tab-forecast"
        )
        assert result == [no_update] * 8

    def test_v1_with_alerts(self, callbacks):
        """V1 fallback with demo alerts returns alert cards and figures."""
        fake_alerts = [
            {
                "event": "Heat Advisory",
                "headline": "Excessive heat warning",
                "severity": "warning",
                "expires": "2024-06-15T12:00:00",
            },
            {
                "event": "Grid Stress",
                "headline": "High demand expected",
                "severity": "critical",
                "expires": "2024-06-15T18:00:00",
            },
        ]

        with (
            patch("components.callbacks.redis_get", return_value=None),
            patch("data.demo_data.generate_demo_alerts", return_value=fake_alerts),
        ):
            result = callbacks["update_alerts_tab"](
                "FPL", _demand_json(), _weather_json(), "tab-alerts"
            )

        assert len(result) == 8
        (
            alert_cards,
            stress,
            stress_label,
            breakdown,
            fig_anomaly,
            fig_temp,
            fig_timeline,
            weather_ctx,
        ) = result
        assert len(alert_cards) == 2
        assert int(stress) > 0
        assert isinstance(fig_anomaly, go.Figure)
        assert isinstance(fig_temp, go.Figure)
        assert isinstance(fig_timeline, go.Figure)

    def test_v1_no_demand_weather(self, callbacks):
        """V1 with no demand/weather still produces alerts and timeline."""
        fake_alerts = [
            {
                "event": "Storm Watch",
                "headline": "Approaching storm",
                "severity": "info",
            },
        ]

        with (
            patch("components.callbacks.redis_get", return_value=None),
            patch("data.demo_data.generate_demo_alerts", return_value=fake_alerts),
        ):
            result = callbacks["update_alerts_tab"]("FPL", None, None, "tab-alerts")

        assert len(result) == 8
        # fig_anomaly should be the empty figure (no demand)
        assert isinstance(result[4], go.Figure)

    def test_v1_no_alerts(self, callbacks):
        """V1 with no alerts produces 'no active alerts' message."""
        with (
            patch("components.callbacks.redis_get", return_value=None),
            patch("data.demo_data.generate_demo_alerts", return_value=[]),
        ):
            result = callbacks["update_alerts_tab"]("FPL", None, None, "tab-alerts")

        assert len(result) == 8
        alert_cards = result[0]
        # Should contain a single "No active alerts" paragraph
        assert len(alert_cards) == 1


# ===================================================================
# TestDemandOutlookV1: update_demand_outlook callback closure
# ===================================================================


class TestDemandOutlookV1:
    """Tests for update_demand_outlook v1 fallback (lines 3146-3280)."""

    def test_tab_guard(self, callbacks):
        """Non-active tab returns no_update."""
        result = callbacks["update_demand_outlook"](
            24,
            "xgboost",
            "tab-forecast",
            _demand_json(),
            "grid_ops",
            "current",
            _weather_json(),
            "FPL",
        )
        assert result == [no_update] * 10

    def test_no_data_returns_loading(self, callbacks):
        """No demand/weather data returns loading placeholder."""
        with patch("components.callbacks.redis_get", return_value=None):
            result = callbacks["update_demand_outlook"](
                24, "xgboost", "tab-outlook", None, "grid_ops", "current", None, "FPL"
            )

        assert len(result) == 10
        assert isinstance(result[0], go.Figure)

    def test_v1_with_valid_forecast(self, callbacks):
        """V1 compute with successful forecast returns figure and KPIs."""
        fake_result = {
            "timestamps": pd.date_range("2024-06-10", periods=24, freq="h"),
            "predictions": np.random.uniform(35000, 45000, 24),
        }

        with (
            patch("components.callbacks.redis_get", return_value=None),
            patch("components.callbacks._run_forecast_outlook", return_value=fake_result),
        ):
            result = callbacks["update_demand_outlook"](
                24,
                "xgboost",
                "tab-outlook",
                _demand_json(),
                "grid_ops",
                "current",
                _weather_json(),
                "FPL",
            )

        assert len(result) == 10
        fig = result[0]
        peak = result[2]
        avg = result[4]
        assert isinstance(fig, go.Figure)
        assert "MW" in peak
        assert "MW" in avg

    def test_v1_forecast_error(self, callbacks):
        """When forecast returns error, display error annotation."""
        with (
            patch("components.callbacks.redis_get", return_value=None),
            patch(
                "components.callbacks._run_forecast_outlook",
                return_value={"error": "Model training failed"},
            ),
        ):
            result = callbacks["update_demand_outlook"](
                24,
                "xgboost",
                "tab-outlook",
                _demand_json(),
                "grid_ops",
                "current",
                _weather_json(),
                "FPL",
            )

        assert len(result) == 10
        # KPIs should show explicit error placeholders
        assert result[2] == "No data"


# ===================================================================
# TestBacktestV1: update_backtest_chart callback closure
# ===================================================================


class TestBacktestV1:
    """Tests for update_backtest_chart v1 fallback (lines 3322-3508)."""

    def test_tab_guard(self, callbacks):
        """Non-active tab returns no_update."""
        result = callbacks["update_backtest_chart"](
            24, "xgboost", "tab-forecast", _demand_json(), "grid_ops", _weather_json(), "FPL"
        )
        assert result == [no_update] * 7

    def test_no_demand_data(self, callbacks):
        """No demand data returns placeholder figure."""
        with patch("components.callbacks.redis_get", return_value=None):
            result = callbacks["update_backtest_chart"](
                24, "xgboost", "tab-backtest", None, "grid_ops", None, "FPL"
            )

        assert len(result) == 7
        assert isinstance(result[0], go.Figure)
        assert result[1] == "No data"

    def test_v1_with_valid_backtest(self, callbacks):
        """V1 compute with successful backtest returns figure and metrics."""
        ts = pd.date_range("2024-06-01", periods=24, freq="h")
        fake_result = {
            "timestamps": ts.values,
            "actual": np.random.uniform(35000, 45000, 24),
            "predictions": np.random.uniform(35000, 45000, 24),
            "metrics": {"mape": 4.5, "rmse": 500, "mae": 400, "r2": 0.95},
            "num_folds": 3,
            "fold_boundaries": [0, 8, 16],
        }

        with (
            patch("components.callbacks.redis_get", return_value=None),
            patch("components.callbacks._run_backtest_for_horizon", return_value=fake_result),
        ):
            result = callbacks["update_backtest_chart"](
                24, "xgboost", "tab-backtest", _demand_json(), "grid_ops", _weather_json(), "FPL"
            )

        assert len(result) == 7
        fig, mape, rmse, mae, r2, explanation, insight = result
        assert isinstance(fig, go.Figure)
        assert mape == "4.50% (forecast_exog)"
        assert "MW" in rmse

    def test_v1_backtest_error(self, callbacks):
        """When backtest returns error, display error annotation."""
        with (
            patch("components.callbacks.redis_get", return_value=None),
            patch(
                "components.callbacks._run_backtest_for_horizon",
                return_value={"error": "Insufficient data"},
            ),
        ):
            result = callbacks["update_backtest_chart"](
                24, "xgboost", "tab-backtest", _demand_json(), "grid_ops", _weather_json(), "FPL"
            )

        assert len(result) == 7
        assert result[1] == "No data"


# ===================================================================
# TestRunScenario: run_scenario callback closure
# ===================================================================


class TestRunScenario:
    """Tests for run_scenario callback (lines 2661-2811)."""

    def test_no_demand_returns_empty(self, callbacks):
        """When demand_json is None, return empty placeholders."""
        mock_ctx = MagicMock()
        mock_ctx.triggered_id = "sim-run-btn"

        with patch("components.callbacks.ctx", mock_ctx):
            result = callbacks["run_scenario"](1, [], 95, 15, 50, 60, 500, 24, "FPL", None)

        assert len(result) == 11
        assert result[1] == "No data"

    def test_run_button_click(self, callbacks):
        """Run button produces forecast, price, renewable figures."""
        mock_ctx = MagicMock()
        mock_ctx.triggered_id = "sim-run-btn"

        with patch("components.callbacks.ctx", mock_ctx):
            result = callbacks["run_scenario"](
                1, [], 95, 15, 50, 60, 500, 24, "FPL", _demand_json()
            )

        assert len(result) == 11
        (
            fig_forecast,
            delta_mw,
            delta_pct,
            price,
            price_delta,
            reserve,
            reserve_status,
            renewable,
            renewable_detail,
            fig_price,
            fig_renewable,
        ) = result
        assert isinstance(fig_forecast, go.Figure)
        assert isinstance(fig_price, go.Figure)
        assert isinstance(fig_renewable, go.Figure)
        assert "MW" in delta_mw
        assert "$" in price

    def test_preset_click(self, callbacks):
        """Preset button applies preset weather values."""
        mock_ctx = MagicMock()
        mock_ctx.triggered_id = {"type": "preset-btn", "index": "winter_storm_uri"}

        preset_data = {
            "name": "Winter Storm Uri",
            "weather": {
                "temperature_2m": 10,
                "wind_speed_80m": 5,
                "cloud_cover": 90,
                "relative_humidity_2m": 80,
                "shortwave_radiation": 100,
            },
            "region": "ERCOT",
        }

        with (
            patch("components.callbacks.ctx", mock_ctx),
            patch("simulation.presets.get_preset", return_value=preset_data),
        ):
            result = callbacks["run_scenario"](
                0, [1], 75, 15, 50, 60, 500, 24, "FPL", _demand_json()
            )

        assert len(result) == 11
        assert isinstance(result[0], go.Figure)


# ===================================================================
# TestApplyPresetToSliders: apply_preset_to_sliders callback closure
# ===================================================================


class TestApplyPresetToSliders:
    """Tests for apply_preset_to_sliders callback."""

    def test_non_dict_trigger_returns_no_update(self, callbacks):
        """When trigger is not a dict, return no_update for all sliders."""
        mock_ctx = MagicMock()
        mock_ctx.triggered_id = "sim-run-btn"

        with patch("components.callbacks.ctx", mock_ctx):
            result = callbacks["apply_preset_to_sliders"]([None])

        assert result == (no_update, no_update, no_update, no_update, no_update)

    def test_preset_trigger_returns_weather_values(self, callbacks):
        """When preset-btn triggers, return weather values from preset."""
        mock_ctx = MagicMock()
        mock_ctx.triggered_id = {"type": "preset-btn", "index": "winter_storm_uri"}

        preset_data = {
            "name": "Winter Storm Uri",
            "weather": {
                "temperature_2m": 10,
                "wind_speed_80m": 5,
                "cloud_cover": 90,
                "relative_humidity_2m": 80,
                "shortwave_radiation": 100,
            },
        }

        with (
            patch("components.callbacks.ctx", mock_ctx),
            patch("simulation.presets.get_preset", return_value=preset_data),
        ):
            result = callbacks["apply_preset_to_sliders"]([1])

        assert result == (10, 5, 90, 80, 100)


# ===================================================================
# TestSliderDisplayUpdate: update_slider_display callbacks
# ===================================================================


class TestSliderDisplayUpdate:
    """Tests for slider display update callbacks."""

    def test_slider_display_registered(self, callbacks):
        """Verify at least one slider display update callback exists."""
        assert "update_slider_display" in callbacks

    def test_slider_display_formats_value(self, callbacks):
        """Slider display formats value with appropriate unit."""
        result = callbacks["update_slider_display"](95)
        # The unit depends on which slider callback was last registered in the loop.
        # The last one is sim-solar with W/m^2. It should format as "95<unit>".
        assert "95" in str(result)
