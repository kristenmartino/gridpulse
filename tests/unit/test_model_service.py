"""
Tests for models/model_service.py — the forecast service layer.

Verifies:
- Simulated fallback produces deterministic, correctly-shaped output
- Metrics API returns valid structure
- Ensemble weights sum to 1.0
- Region consistency (same input → same output)
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Track which modules we mock so we can restore them after
_mocked_modules = {}
for mod in ["prophet", "pmdarima", "shap"]:
    if mod not in sys.modules:
        _mocked_modules[mod] = None  # didn't exist before
        sys.modules[mod] = MagicMock()
    else:
        _mocked_modules[mod] = sys.modules[mod]  # save original

from config import REGION_COORDINATES  # noqa: E402
from models.model_service import (  # noqa: E402
    get_ensemble_weights,
    get_forecasts,
    get_model_metrics,
    is_trained,
)


def teardown_module():
    """Restore real modules so downstream tests (e.g. test_precompute) aren't polluted."""
    for mod, original in _mocked_modules.items():
        if original is None:
            sys.modules.pop(mod, None)
        else:
            sys.modules[mod] = original


@pytest.fixture
def sample_demand():
    """168-hour demand DataFrame."""
    from data.demo_data import generate_demo_demand

    return generate_demo_demand("FPL", days=7)


class TestGetForecasts:
    """Test the main forecast interface."""

    def test_returns_all_model_keys(self, sample_demand):
        result = get_forecasts("FPL", sample_demand)
        for key in ["prophet", "arima", "xgboost", "ensemble"]:
            assert key in result, f"Missing key: {key}"

    def test_returns_confidence_bands(self, sample_demand):
        result = get_forecasts("FPL", sample_demand)
        assert "upper_80" in result
        assert "lower_80" in result

    def test_forecast_shapes_match_input(self, sample_demand):
        result = get_forecasts("FPL", sample_demand)
        n = len(sample_demand)
        for key in ["prophet", "arima", "xgboost", "ensemble"]:
            assert len(result[key]) == n, f"{key} has wrong length"

    def test_forecasts_are_positive(self, sample_demand):
        result = get_forecasts("FPL", sample_demand)
        for key in ["prophet", "arima", "xgboost", "ensemble"]:
            assert (result[key] > 0).all(), f"{key} has non-positive values"

    def test_ensemble_bounded_by_individuals(self, sample_demand):
        result = get_forecasts("FPL", sample_demand)
        individual_min = np.minimum.reduce([result["prophet"], result["arima"], result["xgboost"]])
        individual_max = np.maximum.reduce([result["prophet"], result["arima"], result["xgboost"]])
        # Ensemble should be within individual range (±small tolerance for rounding)
        assert (result["ensemble"] >= individual_min * 0.99).all()
        assert (result["ensemble"] <= individual_max * 1.01).all()

    def test_confidence_bands_ordered(self, sample_demand):
        result = get_forecasts("FPL", sample_demand)
        assert (result["upper_80"] >= result["ensemble"]).all()
        assert (result["lower_80"] <= result["ensemble"]).all()

    def test_deterministic_output(self, sample_demand):
        """Same region + data → same forecasts (no random flicker)."""
        r1 = get_forecasts("FPL", sample_demand)
        r2 = get_forecasts("FPL", sample_demand)
        np.testing.assert_array_equal(r1["ensemble"], r2["ensemble"])

    def test_different_regions_differ(self, sample_demand):
        """Different regions produce different forecasts."""
        from data.demo_data import generate_demo_demand

        r1 = get_forecasts("FPL", sample_demand)
        r2 = get_forecasts("ERCOT", generate_demo_demand("ERCOT", days=7))
        # Forecasts should differ (different seeds + different demand)
        assert not np.array_equal(r1["ensemble"], r2["ensemble"])

    def test_source_is_simulated_when_no_models(self, sample_demand):
        result = get_forecasts("FPL", sample_demand)
        assert result["source"] == "simulated"

    def test_metrics_included_in_result(self, sample_demand):
        result = get_forecasts("FPL", sample_demand)
        assert "metrics" in result
        assert "ensemble" in result["metrics"]

    @pytest.mark.parametrize("region", list(REGION_COORDINATES.keys()))
    def test_all_regions_produce_forecasts(self, region):
        from data.demo_data import generate_demo_demand

        df = generate_demo_demand(region, days=2)
        result = get_forecasts(region, df)
        assert len(result["ensemble"]) == len(df)


class TestGetModelMetrics:
    """Test metrics retrieval."""

    def test_returns_all_models(self):
        metrics = get_model_metrics("FPL")
        for model in ["prophet", "arima", "xgboost", "ensemble"]:
            assert model in metrics

    def test_metrics_have_required_keys(self):
        metrics = get_model_metrics("FPL")
        for model, m in metrics.items():
            for key in ["mape", "rmse", "mae", "r2"]:
                assert key in m, f"{model} missing {key}"

    def test_mape_is_reasonable(self):
        metrics = get_model_metrics("FPL")
        for model, m in metrics.items():
            assert 0 < m["mape"] < 20, f"{model} MAPE={m['mape']} out of range"

    def test_r2_is_reasonable(self):
        metrics = get_model_metrics("FPL")
        for model, m in metrics.items():
            assert 0.5 < m["r2"] <= 1.0, f"{model} R²={m['r2']} out of range"


class TestGetEnsembleWeights:
    """Test ensemble weight retrieval."""

    def test_weights_sum_to_one(self):
        weights = get_ensemble_weights("FPL")
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_all_weights_positive(self):
        weights = get_ensemble_weights("FPL")
        for name, w in weights.items():
            assert w > 0, f"{name} weight is non-positive"

    def test_has_all_model_names(self):
        weights = get_ensemble_weights("FPL")
        for name in ["prophet", "arima", "xgboost"]:
            assert name in weights


class TestIsTrained:
    """Test model availability check."""

    def test_returns_false_when_no_models(self):
        # No models saved on disk in test env
        assert is_trained("FPL") is False
