"""Unit tests for models/ensemble.py."""

import numpy as np
import pytest

from models.ensemble import compute_ensemble_weights, ensemble_combine


class TestComputeEnsembleWeights:
    """Weight computation: proportional to (1/MAPE)^k, k=ENSEMBLE_WEIGHT_EXPONENT (ADR-004)."""

    def test_basic_weights(self):
        mapes = {"prophet": 5.0, "arima": 10.0, "xgboost": 5.0}
        weights = compute_ensemble_weights(mapes)
        assert sum(weights.values()) == pytest.approx(1.0)
        assert weights["prophet"] == pytest.approx(weights["xgboost"])
        assert weights["prophet"] > weights["arima"]

    def test_single_model(self):
        weights = compute_ensemble_weights({"xgboost": 3.0})
        assert weights["xgboost"] == pytest.approx(1.0)

    def test_equal_mape(self):
        mapes = {"a": 5.0, "b": 5.0, "c": 5.0}
        weights = compute_ensemble_weights(mapes)
        for w in weights.values():
            assert w == pytest.approx(1.0 / 3)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_ensemble_weights({})

    def test_handles_inf_mape(self):
        mapes = {"a": 5.0, "b": float("inf")}
        weights = compute_ensemble_weights(mapes)
        assert weights["a"] == pytest.approx(1.0)

    def test_exponent_sharpens_toward_best_model(self):
        """#181: k>1 concentrates weight on the low-MAPE model far more than
        plain inverse-MAPE. With MAPE 2 vs 10, plain inverse (k=1) gives the
        leader 0.833; the sharpened default (k=3) gives it ~0.992."""
        from config import ENSEMBLE_WEIGHT_EXPONENT

        assert ENSEMBLE_WEIGHT_EXPONENT >= 1.0
        weights = compute_ensemble_weights({"xgboost": 2.0, "arima": 10.0})
        k = ENSEMBLE_WEIGHT_EXPONENT
        num_x, num_a = (1 / 2.0) ** k, (1 / 10.0) ** k
        assert weights["xgboost"] == pytest.approx(num_x / (num_x + num_a))
        if k > 1.0:
            # Strictly sharper than plain inverse-MAPE's 0.833 for the leader.
            assert weights["xgboost"] > 0.833


class TestEnsembleCombine:
    """Weighted forecast combination."""

    def test_equal_weights(self):
        forecasts = {
            "a": np.array([100.0, 200.0]),
            "b": np.array([200.0, 300.0]),
        }
        result = ensemble_combine(forecasts)
        np.testing.assert_array_almost_equal(result, [150.0, 250.0])

    def test_weighted(self):
        forecasts = {
            "a": np.array([100.0]),
            "b": np.array([200.0]),
        }
        weights = {"a": 0.75, "b": 0.25}
        result = ensemble_combine(forecasts, weights)
        assert result[0] == pytest.approx(125.0)

    def test_single_model(self):
        forecasts = {"only": np.array([100.0, 200.0])}
        result = ensemble_combine(forecasts)
        np.testing.assert_array_equal(result, [100.0, 200.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ensemble_combine({})

    def test_mismatched_lengths_truncates(self):
        forecasts = {
            "a": np.array([100.0, 200.0, 300.0]),
            "b": np.array([150.0, 250.0]),
        }
        result = ensemble_combine(forecasts)
        assert len(result) == 2

    def test_ensemble_bounded_by_individuals(self):
        """Ensemble should be between min and max of individual forecasts."""
        np.random.seed(42)
        forecasts = {
            "a": np.random.uniform(100, 200, 50),
            "b": np.random.uniform(100, 200, 50),
            "c": np.random.uniform(100, 200, 50),
        }
        result = ensemble_combine(forecasts)
        stacked = np.stack(list(forecasts.values()))
        assert (result >= stacked.min(axis=0) - 1e-6).all()
        assert (result <= stacked.max(axis=0) + 1e-6).all()
