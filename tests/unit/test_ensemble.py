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

    # ------------------------------------------------------------------
    # The validity filter. `v > 0 and np.isfinite(v)` is not defensive
    # decoration — every clause in it stands between a real input and a
    # ZeroDivisionError, and mutation testing found all of them unpinned
    # (docs/TEST_QUALITY.md). ADR-004 weights feed every served forecast, so
    # this function raising would take the whole ensemble down.
    #
    # test_handles_inf_mape above does not cover it: pairing inf with a
    # healthy model leaves a non-zero denominator, so the weights come out
    # right even with the guard broken. The cases below are the ones where
    # the guard is load-bearing.
    # ------------------------------------------------------------------

    def test_a_zero_mape_is_excluded_rather_than_divided_by(self):
        """A perfect score must not become ``1.0 / 0``.

        Pins the ``v > 0`` clause: relaxed to ``v >= 0``, a MAPE of exactly
        zero enters the inverse and raises ZeroDivisionError.
        """
        weights = compute_ensemble_weights({"perfect": 0.0, "normal": 5.0})

        assert weights == {"normal": pytest.approx(1.0)}
        assert "perfect" not in weights, "a zero MAPE is unusable, not infinitely good"

    def test_a_non_finite_mape_is_excluded_even_as_the_only_model(self):
        """An infinite MAPE alone must fall back, not divide by zero.

        Pins the ``np.isfinite(v)`` clause. Weakened to ``or``, ``inf`` passes
        the filter, ``(1/inf)**k`` is 0.0, the weight total is 0.0, and the
        normalisation raises.

        Reachable, not hypothetical: ``compute_mape`` returns ``inf`` when
        every actual is zero, and TIDC publishes zeros (STATUS.md). A BA whose
        feed goes flat produces exactly this input.
        """
        weights = compute_ensemble_weights({"broken": float("inf")})

        assert weights == {"broken": pytest.approx(1.0)}, "equal-weights fallback, not a crash"

    def test_a_negative_mape_is_excluded(self):
        """A negative MAPE is corrupt input, and must not produce a negative weight.

        Also pins ``np.isfinite`` against ``or``: ``-5.0`` is finite, so the
        weakened filter admits it, ``(1/-5)**3`` is negative, and it cancels
        the healthy model's weight to a total of zero.
        """
        weights = compute_ensemble_weights({"bad": -5.0, "good": 5.0})

        assert weights == {"good": pytest.approx(1.0)}
        assert all(w >= 0 for w in weights.values()), "no model may carry a negative weight"

    def test_the_equal_weights_fallback_still_sums_to_one(self):
        """When nothing is usable, the fallback must still be a distribution.

        Pins the ``1.0 / n``: mutated to ``1.0 * n`` or ``2.0 / n`` the
        function still returns a weight per model and still looks plausible,
        but the weights no longer sum to 1 — so the "ensemble" silently
        scales every forecast it combines.
        """
        weights = compute_ensemble_weights({"a": 0.0, "b": float("nan")})

        assert sum(weights.values()) == pytest.approx(1.0), "a fallback is still a distribution"
        assert weights == {"a": pytest.approx(0.5), "b": pytest.approx(0.5)}

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

    def test_a_model_missing_from_weights_contributes_nothing(self):
        """An unweighted model gets weight 0, not ``None``.

        Pins the default in ``weights.get(k, 0)``. Dropped, the missing model
        yields ``None``, and summing the weights raises TypeError — so a
        weights dict that has fallen out of step with the served models (a
        newly added model, a renamed key) would take down the combine instead
        of degrading to the models it does know about.
        """
        forecasts = {"a": np.array([100.0]), "b": np.array([200.0])}

        result = ensemble_combine(forecasts, {"a": 1.0})

        assert result[0] == pytest.approx(100.0), "b is unweighted, so it contributes nothing"

    def test_weights_are_renormalised_to_the_models_present(self):
        """Weights that do not sum to 1 are rescaled, not applied raw.

        Without renormalisation this returns 30.0 — a forecast an order of
        magnitude below either input, from two models that agree closely.
        """
        forecasts = {"a": np.array([100.0]), "b": np.array([200.0])}

        result = ensemble_combine(forecasts, {"a": 0.1, "b": 0.1})

        assert result[0] == pytest.approx(150.0)

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
