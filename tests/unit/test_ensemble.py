"""Unit tests for models/ensemble.py."""

import numpy as np
import pytest
from structlog.testing import capture_logs

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

    # ------------------------------------------------------------------
    # The `total == 0` fallback. Five mutants live in these two lines and
    # every one of them silently rescales the served forecast, because —
    # unlike the `weights is None` fallback six lines above — this branch is
    # NOT renormalised afterwards. It hard-codes `total = 1.0` and uses the
    # weights as written.
    #
    # docs/TEST_QUALITY.md had the two fallbacks adjudicated together as
    # "equivalent — renormalised on the next line, a test for it would be
    # theatre". That is true of the first and false of this one. The
    # expression is identical; what follows it is not.
    # ------------------------------------------------------------------

    def test_weights_that_have_gone_entirely_stale_fall_back_to_an_equal_blend(self):
        """No overlap between the weights dict and the served models.

        Reachable three ways — renamed model keys, an empty dict, an all-zero
        dict — and none of them was covered. Mutated, this branch returns
        300.0, 600.0 or 75.0 for an input whose answer is 150.0: a forecast
        scaled by 2x, 4x or 0.5x, still finite, still plausibly shaped, with
        nothing in the output marking it as wrong.
        """
        forecasts = {"a": np.array([100.0]), "b": np.array([200.0])}
        expected = ensemble_combine(forecasts)[0]  # the equal-weight answer

        assert expected == pytest.approx(150.0)

        stale_keys = ensemble_combine(forecasts, {"prophet": 1.0, "sarimax": 1.0})
        assert stale_keys[0] == pytest.approx(expected), "renamed models, not a rescaled forecast"

        empty = ensemble_combine(forecasts, {})
        assert empty[0] == pytest.approx(expected)

        all_zero = ensemble_combine(forecasts, {"a": 0.0, "b": 0.0})
        assert all_zero[0] == pytest.approx(expected)

    def test_the_stale_weights_fallback_is_a_distribution(self):
        """The fallback weights must sum to 1 across however many models there are.

        Checked at three model counts because `1.0 / n` and `1.0 * n` agree at
        n = 1: a single-model fixture cannot see the difference, and a
        two-model one cannot separate `2.0 / n` from a doubled total.
        """
        for n in (1, 2, 4):
            forecasts = {f"m{i}": np.full(3, 500.0) for i in range(n)}

            result = ensemble_combine(forecasts, {"stale": 1.0})

            np.testing.assert_allclose(
                result, np.full(3, 500.0), err_msg=f"{n} identical models must blend to 500"
            )


class TestEnsembleBoundsWarning:
    """The pointwise-bounds invariant, which is checked but only *warned* about.

    Eleven mutants live in this block and no assertion on the return value can
    reach any of them: it computes a diagnostic, logs, and returns `result`
    untouched. That is a deliberate fail-open design — a bounds violation is
    worth telling an operator about, not worth refusing to serve a forecast
    over — so the honest test asserts the warning rather than the return.

    Before this class the block was not untested, which is the more
    interesting part. Four of its mutants were killed by
    `test_stable_hash_reproducibility.py`, a test that has no idea this module
    exists: it runs a forecast in a subprocess and parses stdout as JSON, so
    an extra structlog line breaks `json.loads` and the test fails with
    `JSONDecodeError: Extra data`. That is a real kill by the score's
    definition and no protection at all — routing structlog to stderr, a
    change nobody would think to question, silently removes it.
    """

    FORECASTS = {"a": np.array([100.0]), "b": np.array([200.0])}

    def _warnings(self, forecasts, weights=None):
        with capture_logs() as caps:
            result = ensemble_combine(forecasts, weights)
        return result, [c for c in caps if c["event"] == "ensemble_out_of_bounds"]

    def test_a_well_formed_combine_says_nothing(self):
        """`if out_of_bounds > 0` — relaxed to `>= 0` this fires on every call.

        51 BAs on an hourly scoring job makes that ~1,200 spurious warnings a
        day on the one event that is supposed to mean something is wrong.
        """
        _, warned = self._warnings(self.FORECASTS)

        assert warned == []

    def test_a_forecast_sitting_exactly_on_its_bound_is_not_a_violation(self):
        """The 1e-6 tolerance is two-sided, and a single model sits on both bounds.

        With one model the ensemble *is* the min and *is* the max. Flip either
        tolerance inward (`min - 1e-6` → `min + 1e-6`, `max + 1e-6` →
        `max - 1e-6`) and this exactly-correct forecast is reported as out of
        bounds on every call.
        """
        _, warned = self._warnings({"only": np.array([100.0, 200.0])})

        assert warned == []

    def test_a_violation_is_detected_and_counted(self):
        """Negative weights are the only way past the bound, and they reach it.

        `weights.get(k, 0)` admits any float the caller supplies; a corrupt or
        hand-tuned weights dict with a negative entry produces 0.0 MW from two
        models forecasting 100 and 200. Pins `> 1` (the count is exactly one)
        and the `|` that makes either side of the band sufficient — narrowed
        to `&`, a below-min result must ALSO be above max, so nothing is ever
        flagged.
        """
        result, warned = self._warnings(self.FORECASTS, {"a": 2.0, "b": -1.0})

        assert result[0] == pytest.approx(0.0), "below both individual forecasts"
        assert len(warned) == 1
        assert warned[0]["count"] == 1

    def test_the_band_is_per_hour_not_across_the_whole_horizon(self):
        """`stacked.min(axis=0)` / `max(axis=0)` — a bound per timestamp.

        Dropped to `axis=None` the check collapses to two scalars over the
        entire horizon, so an hour that is wildly wrong still falls inside the
        horizon-wide range and goes unreported. Demand has a diurnal shape;
        this is the normal case, not a corner one. Both fixtures below violate
        their own hour's band while sitting comfortably inside the global one.
        """
        # Per-hour min [100, 500]; global min 100. Hour 1's 400 is only a
        # violation if the band is per-hour.
        peaked = {"a": np.array([100.0, 500.0]), "b": np.array([200.0, 600.0])}

        below, warned = self._warnings(peaked, {"a": 2.0, "b": -1.0})
        np.testing.assert_allclose(below, [0.0, 400.0])
        assert warned[0]["count"] == 2, "both hours are below their own hour's minimum"

        above, warned = self._warnings(peaked, {"a": -1.0, "b": 2.0})
        np.testing.assert_allclose(above, [300.0, 700.0])
        assert warned[0]["count"] == 2, "both hours are above their own hour's maximum"

    def test_the_tolerance_is_float_noise_not_a_megawatt(self):
        """`1e-6`, not `1.000001`.

        Both tolerances widen to ~1 MW under mutation. A 1 MW allowance sounds
        harmless and is not: it is a real violation on the small BAs, where
        SPA's median demand is ~24 MW. These fixtures overshoot by 0.5 MW —
        past float noise, inside a 1 MW window — so only the tight tolerance
        reports them.
        """
        _, below = self._warnings(self.FORECASTS, {"a": 1.005, "b": -0.005})
        assert below[0]["count"] == 1, "99.5 is below the 100.0 floor"

        _, above = self._warnings(self.FORECASTS, {"a": -0.005, "b": 1.005})
        assert above[0]["count"] == 1, "200.5 is above the 200.0 ceiling"


class TestEnsembleLengthMismatch:
    """The truncation guard, whose only observable effect is its warning.

    `len(set(lengths)) > 1` → `>= 1` takes the truncating branch on every
    call, and when the lengths already agree `min_len` is that length and the
    slice is a no-op — so the returned array is byte-identical. Equivalent by
    return value; not equivalent by behaviour, because the operator is now
    told every hour that forecasts of matching length were truncated.

    docs/TEST_QUALITY.md carried this one as a FALSE SURVIVOR "actually
    killed". It is killed — by the same `json.loads` accident described above,
    in a test about hash stability. Asserted properly here.
    """

    def test_matching_lengths_are_not_reported_as_truncated(self):
        with capture_logs() as caps:
            ensemble_combine({"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])})

        assert [c for c in caps if c["event"] == "ensemble_length_mismatch"] == []

    def test_a_mismatch_is_reported_with_what_was_truncated(self):
        """`test_mismatched_lengths_truncates` asserts `len(result) == 2` and
        stops there — it never checks *which* two, so truncating from the end
        instead of the start passes it. The values are asserted here."""
        forecasts = {"a": np.array([100.0, 200.0, 300.0]), "b": np.array([150.0, 250.0])}

        with capture_logs() as caps:
            result = ensemble_combine(forecasts)

        np.testing.assert_allclose(
            result, [125.0, 225.0], err_msg="the first two hours, not the last"
        )

        warned = [c for c in caps if c["event"] == "ensemble_length_mismatch"]
        assert len(warned) == 1
        assert warned[0]["truncating_to"] == 2
        assert warned[0]["lengths"] == {"a": 3, "b": 2}
