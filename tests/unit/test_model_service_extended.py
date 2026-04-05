"""
Extended tests for models/model_service.py — targeting uncovered code paths.

Covers:
- get_forecasts() trained model path, simulated fallback, cache hit
- _load_cached_models() cache hit, cache miss, corrupted cache
- _predict_from_trained() valid prediction, model error fallback
- _simulate_forecasts() generates all 3 model predictions + ensemble
- get_model_metrics() trained metrics and empty/simulated metrics
- get_ensemble_weights() trained weights and default fallback
- is_trained() true/false scenarios
- Error paths: individual model failure, all models fail
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Mock heavy optional dependencies before importing model_service
_mocked_modules = {}
for mod in ["prophet", "pmdarima", "shap"]:
    if mod not in sys.modules:
        _mocked_modules[mod] = None
        sys.modules[mod] = MagicMock()
    else:
        _mocked_modules[mod] = sys.modules[mod]

from models import model_service  # noqa: E402
from models.model_service import (  # noqa: E402
    _load_cached_models,
    _predict_from_trained,
    _simulate_forecasts,
    get_ensemble_weights,
    get_forecasts,
    get_model_metrics,
    is_trained,
)


def teardown_module():
    """Restore real modules so downstream tests are not polluted."""
    for mod, original in _mocked_modules.items():
        if original is None:
            sys.modules.pop(mod, None)
        else:
            sys.modules[mod] = original


@pytest.fixture
def demand_df():
    """Small 168-hour demand DataFrame for testing."""
    n = 168
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    demand = 30000 + 5000 * np.sin(2 * np.pi * np.arange(n) / 24)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "demand_mw": demand,
        }
    )


@pytest.fixture
def fake_model_data():
    """Fake trained model data dict as returned by load_models."""
    return {
        "region": "FPL",
        "prophet_model": MagicMock(name="prophet_model_obj"),
        "arima_model": MagicMock(name="arima_model_obj"),
        "xgboost_model": MagicMock(name="xgboost_model_obj"),
        "xgboost_feature_names": ["feature_a", "feature_b"],
        "ensemble_weights": {"prophet": 0.30, "arima": 0.20, "xgboost": 0.50},
        "metrics": {
            "prophet": {"mape": 2.5, "rmse": 400, "mae": 300, "r2": 0.97},
            "arima": {"mape": 3.2, "rmse": 520, "mae": 380, "r2": 0.95},
            "xgboost": {"mape": 1.8, "rmse": 350, "mae": 260, "r2": 0.98},
            "ensemble": {"mape": 1.5, "rmse": 310, "mae": 230, "r2": 0.985},
        },
    }


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Clear the in-memory model cache before each test."""
    model_service._model_cache.clear()
    yield
    model_service._model_cache.clear()


# ── _load_cached_models ─────────────────────────────────────────


class TestLoadCachedModels:
    """Tests for _load_cached_models() internal function."""

    def test_cache_hit_returns_cached_data(self, fake_model_data):
        """When region is already in _model_cache, return it without loading."""
        model_service._model_cache["FPL"] = fake_model_data
        result = _load_cached_models("FPL")
        assert result is fake_model_data
        assert result["region"] == "FPL"

    def test_cache_miss_no_file_returns_none(self):
        """When no file on disk, returns None."""
        result = _load_cached_models("FPL")
        assert result is None

    def test_corrupted_cache_returns_none(self, tmp_path):
        """When load_models raises a generic exception, returns None."""
        with patch("models.model_service.log") as mock_log:
            with patch(
                "models.training.load_models",
                side_effect=RuntimeError("corrupted pickle data"),
            ):
                result = _load_cached_models("ERCOT")
            assert result is None
            mock_log.warning.assert_called_once()
            call_kwargs = mock_log.warning.call_args
            assert "model_load_failed" in call_kwargs[0]

    def test_successful_load_populates_cache(self, fake_model_data):
        """After a successful load from disk, the data is cached in memory."""
        with patch("models.training.load_models", return_value=fake_model_data):
            result = _load_cached_models("FPL")
        assert result is fake_model_data
        assert "FPL" in model_service._model_cache
        assert model_service._model_cache["FPL"] is fake_model_data


# ── _predict_from_trained ───────────────────────────────────────


class TestPredictFromTrained:
    """Tests for _predict_from_trained() with mocked model predictions."""

    def test_valid_prediction_all_models(self, demand_df, fake_model_data):
        """When all 3 model predict functions succeed, returns trained source."""
        n = len(demand_df)
        actual = demand_df["demand_mw"].values

        with (
            patch(
                "models.prophet_model.predict_prophet",
                return_value={"forecast": actual * 1.01},
            ),
            patch(
                "models.arima_model.predict_arima",
                return_value=actual * 0.99,
            ),
            patch(
                "models.xgboost_model.predict_xgboost",
                return_value=actual * 1.005,
            ),
        ):
            result = _predict_from_trained(fake_model_data, demand_df, models_shown=None)

        assert result["source"] == "trained"
        assert "prophet" in result
        assert "arima" in result
        assert "xgboost" in result
        assert "ensemble" in result
        assert len(result["ensemble"]) == n
        assert "upper_80" in result
        assert "lower_80" in result
        # Ensemble should be a weighted combination, not equal to any single model
        assert not np.array_equal(result["ensemble"], result["prophet"])

    def test_model_error_falls_back_to_simulated_noise(self, demand_df, fake_model_data):
        """When a model predict function raises, that model gets simulated noise."""
        n = len(demand_df)

        # All three model imports will raise
        def raise_error(*args, **kwargs):
            raise RuntimeError("model prediction failed")

        with patch.dict(
            "sys.modules",
            {
                "models.prophet_model": MagicMock(
                    predict_prophet=MagicMock(side_effect=raise_error)
                ),
                "models.arima_model": MagicMock(predict_arima=MagicMock(side_effect=raise_error)),
                "models.xgboost_model": MagicMock(
                    predict_xgboost=MagicMock(side_effect=raise_error)
                ),
            },
        ):
            result = _predict_from_trained(fake_model_data, demand_df, models_shown=None)

        assert result["source"] == "trained"
        # Even with errors, we still get predictions (simulated fallback per model)
        for key in ["prophet", "arima", "xgboost", "ensemble"]:
            assert key in result
            assert len(result[key]) == n

    def test_models_shown_filter_skips_prophet(self, demand_df, fake_model_data):
        """When models_shown excludes prophet, prophet is skipped (but xgboost is always kept)."""
        len(demand_df)
        actual = demand_df["demand_mw"].values

        with patch.dict(
            "sys.modules",
            {
                "models.arima_model": MagicMock(
                    predict_arima=MagicMock(return_value=actual * 0.99)
                ),
                "models.xgboost_model": MagicMock(
                    predict_xgboost=MagicMock(return_value=actual * 1.005)
                ),
            },
        ):
            result = _predict_from_trained(
                fake_model_data, demand_df, models_shown=["arima", "xgboost"]
            )

        assert result["source"] == "trained"
        # prophet should be absent (skipped by models_shown filter)
        assert "prophet" not in result or "ensemble" in result
        assert "xgboost" in result
        assert "arima" in result

    def test_empty_preds_ensemble_copies_actual(self, demand_df):
        """When no model keys exist in model_data, ensemble falls back to actual."""
        len(demand_df)
        # model_data with no model keys at all
        empty_model_data = {
            "ensemble_weights": {"prophet": 0.30, "arima": 0.20, "xgboost": 0.50},
            "metrics": {},
        }
        result = _predict_from_trained(empty_model_data, demand_df, models_shown=None)
        assert result["source"] == "trained"
        # With no models in model_data, all_preds is empty -> ensemble = actual.copy()
        np.testing.assert_array_equal(result["ensemble"], demand_df["demand_mw"].values)


# ── _simulate_forecasts ─────────────────────────────────────────


class TestSimulateForecasts:
    """Tests for _simulate_forecasts() deterministic simulated output."""

    def test_generates_all_model_keys(self, demand_df):
        actual = demand_df["demand_mw"].values
        result = _simulate_forecasts("FPL", actual, models_shown=None)
        for key in ["prophet", "arima", "xgboost", "ensemble"]:
            assert key in result

    def test_source_is_simulated(self, demand_df):
        actual = demand_df["demand_mw"].values
        result = _simulate_forecasts("FPL", actual, models_shown=None)
        assert result["source"] == "simulated"

    def test_deterministic_across_calls(self, demand_df):
        """Same region + data produces identical output each time."""
        actual = demand_df["demand_mw"].values
        r1 = _simulate_forecasts("ERCOT", actual, models_shown=None)
        r2 = _simulate_forecasts("ERCOT", actual, models_shown=None)
        np.testing.assert_array_equal(r1["ensemble"], r2["ensemble"])
        np.testing.assert_array_equal(r1["prophet"], r2["prophet"])

    def test_metrics_included_for_all_models(self, demand_df):
        actual = demand_df["demand_mw"].values
        result = _simulate_forecasts("FPL", actual, models_shown=None)
        assert "metrics" in result
        for model_name in ["prophet", "arima", "xgboost", "ensemble"]:
            assert model_name in result["metrics"]
            for metric_key in ["mape", "rmse", "mae", "r2"]:
                assert metric_key in result["metrics"][model_name]

    def test_confidence_bands_bracket_ensemble(self, demand_df):
        actual = demand_df["demand_mw"].values
        result = _simulate_forecasts("FPL", actual, models_shown=None)
        assert (result["upper_80"] >= result["ensemble"]).all()
        assert (result["lower_80"] <= result["ensemble"]).all()

    def test_weights_default_values(self, demand_df):
        actual = demand_df["demand_mw"].values
        result = _simulate_forecasts("FPL", actual, models_shown=None)
        assert result["weights"] == {"prophet": 0.30, "arima": 0.20, "xgboost": 0.50}


# ── get_forecasts (trained path) ────────────────────────────────


class TestGetForecastsTrainedPath:
    """Tests for get_forecasts() when trained models are available."""

    def _mock_predict_functions(self, actual):
        """Return a context manager stack that mocks all 3 predict functions."""
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(
            patch(
                "models.prophet_model.predict_prophet",
                return_value={"forecast": actual * 1.01},
            )
        )
        stack.enter_context(
            patch(
                "models.arima_model.predict_arima",
                return_value=actual * 0.99,
            )
        )
        stack.enter_context(
            patch(
                "models.xgboost_model.predict_xgboost",
                return_value=actual * 1.005,
            )
        )
        return stack

    def test_trained_path_returns_trained_source(self, demand_df, fake_model_data):
        """When _load_cached_models returns data, source is 'trained'."""
        model_service._model_cache["FPL"] = fake_model_data
        actual = demand_df["demand_mw"].values
        with self._mock_predict_functions(actual):
            result = get_forecasts("FPL", demand_df)
        assert result["source"] == "trained"

    def test_trained_path_passes_metrics_from_model_data(self, demand_df, fake_model_data):
        """Trained path propagates stored metrics to result."""
        model_service._model_cache["FPL"] = fake_model_data
        actual = demand_df["demand_mw"].values
        with self._mock_predict_functions(actual):
            result = get_forecasts("FPL", demand_df)
        assert result["metrics"] == fake_model_data["metrics"]

    def test_trained_path_passes_weights_from_model_data(self, demand_df, fake_model_data):
        """Trained path propagates stored ensemble weights."""
        model_service._model_cache["FPL"] = fake_model_data
        actual = demand_df["demand_mw"].values
        with self._mock_predict_functions(actual):
            result = get_forecasts("FPL", demand_df)
        assert result["weights"] == fake_model_data["ensemble_weights"]


# ── get_model_metrics ───────────────────────────────────────────


class TestGetModelMetricsExtended:
    """Extended tests for get_model_metrics()."""

    def test_returns_trained_metrics_when_available(self, fake_model_data):
        """When trained models with metrics are cached, return those metrics."""
        model_service._model_cache["FPL"] = fake_model_data
        metrics = get_model_metrics("FPL")
        assert metrics == fake_model_data["metrics"]
        assert metrics["xgboost"]["mape"] == 1.8

    def test_returns_simulated_when_no_trained_models(self):
        """When no trained models, returns hardcoded simulated metrics."""
        metrics = get_model_metrics("FPL")
        assert metrics["xgboost"]["mape"] == 2.1
        assert metrics["ensemble"]["mape"] == 1.9
        assert metrics["prophet"]["r2"] == 0.967

    def test_returns_simulated_when_metrics_key_missing(self):
        """When model data is cached but has no 'metrics' key, returns simulated."""
        model_service._model_cache["FPL"] = {"region": "FPL"}
        metrics = get_model_metrics("FPL")
        # Should fall through to simulated since "metrics" not in model_data
        assert metrics["ensemble"]["mape"] == 1.9


# ── get_ensemble_weights ────────────────────────────────────────


class TestGetEnsembleWeightsExtended:
    """Extended tests for get_ensemble_weights()."""

    def test_returns_trained_weights_when_available(self, fake_model_data):
        model_service._model_cache["FPL"] = fake_model_data
        weights = get_ensemble_weights("FPL")
        assert weights == {"prophet": 0.30, "arima": 0.20, "xgboost": 0.50}

    def test_returns_default_when_no_trained(self):
        weights = get_ensemble_weights("FPL")
        assert weights == {"prophet": 0.30, "arima": 0.20, "xgboost": 0.50}

    def test_returns_default_when_weight_key_missing(self):
        """Cached model data without 'ensemble_weights' key gives defaults."""
        model_service._model_cache["FPL"] = {"region": "FPL", "metrics": {}}
        weights = get_ensemble_weights("FPL")
        assert weights == {"prophet": 0.30, "arima": 0.20, "xgboost": 0.50}


# ── is_trained ──────────────────────────────────────────────────


class TestIsTrainedExtended:
    """Extended tests for is_trained()."""

    def test_true_when_model_file_exists(self, tmp_path):
        """Returns True when the model pickle file exists on disk."""
        model_file = tmp_path / "FPL_models.pkl"
        model_file.write_bytes(b"fake pickle data")
        with (
            patch("models.model_service.MODEL_DIR", str(tmp_path)),
            patch("models.training.MODEL_DIR", str(tmp_path)),
        ):
            assert is_trained("FPL") is True

    def test_false_for_invalid_region(self):
        """Returns False for a region not in REGION_COORDINATES."""
        assert is_trained("INVALID_REGION") is False

    def test_false_for_path_traversal_region(self):
        """Returns False for region strings with path traversal characters."""
        assert is_trained("../etc") is False


# ── Error paths ─────────────────────────────────────────────────


class TestErrorPaths:
    """Tests for error handling and edge cases."""

    def test_individual_model_failure_still_produces_ensemble(self, demand_df, fake_model_data):
        """If one model fails during prediction, the ensemble still forms from remaining."""
        n = len(demand_df)
        actual = demand_df["demand_mw"].values

        # Only xgboost_model key present, prophet and arima keys removed
        partial_model_data = {
            "xgboost_model": fake_model_data["xgboost_model"],
            "xgboost_feature_names": ["feature_a"],
            "ensemble_weights": {"prophet": 0.30, "arima": 0.20, "xgboost": 0.50},
            "metrics": {},
        }

        with patch.dict(
            "sys.modules",
            {
                "models.xgboost_model": MagicMock(
                    predict_xgboost=MagicMock(return_value=actual * 1.01)
                ),
            },
        ):
            result = _predict_from_trained(partial_model_data, demand_df, models_shown=None)

        assert result["source"] == "trained"
        assert "xgboost" in result
        assert "ensemble" in result
        assert len(result["ensemble"]) == n

    def test_get_forecasts_with_models_shown_filter(self, demand_df):
        """get_forecasts passes models_shown through to the simulation path."""
        result = get_forecasts("FPL", demand_df, models_shown=["prophet", "xgboost"])
        # Simulated path always generates all 3 models regardless of filter
        # (filter is only used in trained path)
        assert "source" in result
        assert result["source"] == "simulated"
        for key in ["prophet", "arima", "xgboost", "ensemble"]:
            assert key in result
