"""Unit tests for models/evaluation.py."""

import numpy as np
import pandas as pd
import pytest

from models.evaluation import (
    apply_empirical_interval,
    compute_all_metrics,
    compute_error_by_hour,
    compute_interval_coverage,
    compute_interval_coverage_drift,
    compute_mae,
    compute_mape,
    compute_r2,
    compute_residuals,
    compute_rmse,
    empirical_error_quantiles,
)


class TestMAPE:
    def test_perfect_forecast(self):
        actual = np.array([100, 200, 300])
        assert compute_mape(actual, actual) == 0.0

    def test_known_error(self):
        actual = np.array([100.0, 200.0])
        predicted = np.array([110.0, 180.0])
        # |10/100| + |20/200| = 0.10 + 0.10 = 0.20 → 10%
        assert compute_mape(actual, predicted) == pytest.approx(10.0)

    def test_excludes_zero_actuals(self):
        actual = np.array([0.0, 100.0])
        predicted = np.array([10.0, 110.0])
        result = compute_mape(actual, predicted)
        assert np.isfinite(result)

    def test_all_zeros_returns_inf(self):
        assert compute_mape(np.zeros(5), np.ones(5)) == float("inf")


class TestRMSE:
    def test_perfect_forecast(self):
        actual = np.array([100, 200])
        assert compute_rmse(actual, actual) == 0.0

    def test_known_value(self):
        actual = np.array([100.0])
        predicted = np.array([110.0])
        assert compute_rmse(actual, predicted) == pytest.approx(10.0)


class TestMAE:
    def test_perfect_forecast(self):
        actual = np.array([100, 200])
        assert compute_mae(actual, actual) == 0.0

    def test_symmetric_errors(self):
        actual = np.array([100.0, 200.0])
        predicted = np.array([110.0, 190.0])
        assert compute_mae(actual, predicted) == pytest.approx(10.0)


class TestR2:
    def test_perfect_forecast(self):
        actual = np.array([100, 200, 300])
        assert compute_r2(actual, actual) == pytest.approx(1.0)

    def test_mean_forecast(self):
        actual = np.array([100, 200, 300])
        mean_pred = np.full(3, 200.0)
        assert compute_r2(actual, mean_pred) == pytest.approx(0.0)

    def test_a_flat_actual_series_scores_zero_not_one(self):
        """`if ss_tot < 1e-10: return 0.0` — the undefined case, pinned.

        R² is undefined when the actual series never varies: there is no
        variance to explain, so even a perfect forecast explains none of it.
        The function returns 0.0 rather than 1.0, and nothing exercised that
        branch — flipped, an hour of flat demand would publish a *perfect* R²
        on the Models tab off a forecast that was never tested.
        """
        flat = np.full(5, 100.0)

        assert compute_r2(flat, flat) == 0.0
        assert compute_r2(flat, np.full(5, 999.0)) == 0.0

    def test_worse_than_mean(self):
        actual = np.array([100, 200, 300])
        bad_pred = np.array([300, 100, 200])
        assert compute_r2(actual, bad_pred) < 0.0


class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        actual = np.array([100, 200, 300])
        predicted = np.array([105, 195, 310])
        metrics = compute_all_metrics(actual, predicted)
        assert set(metrics.keys()) == {"mape", "rmse", "mae", "r2"}

    def test_values_are_finite(self):
        actual = np.array([100, 200, 300])
        predicted = np.array([105, 195, 310])
        metrics = compute_all_metrics(actual, predicted)
        for v in metrics.values():
            assert np.isfinite(v)


class TestResiduals:
    def test_basic(self):
        actual = np.array([100, 200])
        predicted = np.array([90, 210])
        residuals = compute_residuals(actual, predicted)
        np.testing.assert_array_equal(residuals, [10, -10])


class TestErrorByHour:
    def test_groups_by_hour(self):
        ts = pd.date_range("2024-01-01", periods=48, freq="h")
        actual = np.random.uniform(100, 200, 48)
        predicted = actual + np.random.normal(0, 5, 48)
        result = compute_error_by_hour(ts, actual, predicted)
        assert len(result) == 24
        assert "mean_abs_error" in result.columns

    def test_each_hours_error_is_the_mean_of_that_hours_absolute_errors(self):
        """The test above asserts a shape; nothing asserted a value.

        With random inputs and only `len == 24` checked, the whole computation
        could be replaced by `abs_errors = None` and the suite stays green —
        the heatmap on the Backtest tab would render 24 empty cells and no test
        would notice. This pins the three published columns.

        The two samples per hour straddle the forecast (one over, one under) so
        that dropping `np.abs` and averaging signed errors gives a different
        answer at every hour.
        """
        ts = pd.date_range("2024-01-01", periods=48, freq="h")
        hour = np.arange(48) % 24
        day = np.arange(48) // 24
        actual = np.full(48, 500.0)
        # day 0 over-forecasts by h+1, day 1 under-forecasts by h+3
        predicted = np.where(day == 0, actual + (hour + 1), actual - (hour + 3))

        result = compute_error_by_hour(ts, actual, predicted).set_index("hour")

        expected_mean = np.arange(24) + 2.0  # ((h+1) + (h+3)) / 2
        np.testing.assert_allclose(result["mean_abs_error"].to_numpy(), expected_mean)
        # sample std (ddof=1) of {h+1, h+3} is the same at every hour
        np.testing.assert_allclose(result["std_abs_error"].to_numpy(), np.full(24, np.sqrt(2.0)))
        np.testing.assert_array_equal(result["count"].to_numpy(), np.full(24, 2))
        np.testing.assert_array_equal(result.index.to_numpy(), np.arange(24))


class TestEmpiricalIntervals:
    def test_empirical_quantiles(self):
        residuals = np.array([-20, -10, 0, 10, 20])
        q = empirical_error_quantiles(residuals, lower_q=0.2, upper_q=0.8)
        assert q["sample_size"] == 5
        assert q["lower_error"] == pytest.approx(-12.0)
        assert q["upper_error"] == pytest.approx(12.0)

    def test_apply_empirical_interval(self):
        pred = np.array([100.0, 200.0])
        lower, upper = apply_empirical_interval(pred, -10.0, 15.0)
        np.testing.assert_allclose(lower, np.array([90.0, 190.0]))
        np.testing.assert_allclose(upper, np.array([115.0, 215.0]))

    def test_interval_coverage(self):
        actual = np.array([100.0, 105.0, 120.0])
        lower = np.array([95.0, 100.0, 110.0])
        upper = np.array([110.0, 106.0, 115.0])
        assert compute_interval_coverage(actual, lower, upper) == pytest.approx(2 / 3)

    # ------------------------------------------------------------------
    # 34 survivors across these three functions (docs/TEST_QUALITY.md), and
    # the reason is visible above: every test passes `lower_q`, `upper_q`,
    # `target_coverage` and `window_size` EXPLICITLY, so the defaults are
    # never executed. They could be changed to anything — 0.10 to 1.1, 0.80
    # to 1.8 — with the suite green.
    #
    # Those defaults are the published contract. The Forecast tab renders an
    # "80% empirical prediction interval"; that 80% is `lower_q=0.10` and
    # `upper_q=0.90` here and nowhere else.
    # ------------------------------------------------------------------

    def test_the_default_quantiles_are_p10_and_p90(self):
        """The interval the UI calls "80%" is these two numbers.

        Called with no quantile arguments — the way production calls it — so
        the defaults are actually exercised rather than shadowed by the test.
        """
        residuals = np.arange(-100, 101, dtype=float)

        q = empirical_error_quantiles(residuals)

        assert q["lower_error"] == pytest.approx(np.quantile(residuals, 0.10))
        assert q["upper_error"] == pytest.approx(np.quantile(residuals, 0.90))
        assert q["lower_error"] == pytest.approx(-80.0)
        assert q["upper_error"] == pytest.approx(80.0)
        assert q["sample_size"] == 201

    def test_the_default_coverage_target_is_80_percent_over_a_week(self):
        """`target_coverage=0.80`, `window_size=168` — an 80% interval judged
        over the trailing week, both unexercised until now."""
        n = 200
        actual = np.zeros(n)
        lower, upper = np.full(n, -1.0), np.full(n, 1.0)

        perfect = compute_interval_coverage_drift(actual, lower, upper)

        # Every point covered, so drift is the full distance above target.
        assert perfect["overall_coverage"] == 1.0
        assert perfect["drift"] == pytest.approx(1.0 - 0.80)

        # The recent window is the last 168 hours, not the whole series: put
        # the first 100 outside and overall/recent must disagree.
        degraded_actual = actual.copy()
        degraded_actual[:100] = 99.0
        degraded = compute_interval_coverage_drift(degraded_actual, lower, upper)

        # The window spans indices 32..199: 68 of them uncovered, 100 covered.
        assert degraded["overall_coverage"] == pytest.approx(0.5)
        assert degraded["recent_coverage"] == pytest.approx(100 / 168)
        assert degraded["recent_coverage"] != degraded["overall_coverage"]

    def test_an_interval_includes_its_own_endpoints(self):
        """`y >= lo` and `y <= hi` — a point exactly on a bound is covered.

        Tightening either to a strict inequality quietly reports a calibrated
        interval as under-covering, which is the direction that triggers
        recalibration.
        """
        actual = np.array([10.0, 20.0])
        lower = np.array([10.0, 5.0])  # first point sits ON the lower bound
        upper = np.array([50.0, 20.0])  # second sits ON the upper bound

        assert compute_interval_coverage(actual, lower, upper) == 1.0

    def test_empty_input_reports_zero_coverage_and_a_full_deficit(self):
        """The degenerate path returns real numbers, and each one is asserted.

        Nothing exercised these returns, so every constant in them was free to
        change — including `drift`, which is `-target_coverage` so that "no
        data" reads as the largest possible shortfall rather than as zero
        drift (which would look healthy).
        """
        empty = np.array([])

        assert empirical_error_quantiles(empty) == {
            "lower_error": 0.0,
            "upper_error": 0.0,
            "sample_size": 0,
        }
        assert compute_interval_coverage(empty, empty, empty) == 0.0
        assert compute_interval_coverage_drift(empty, empty, empty) == {
            "overall_coverage": 0.0,
            "recent_coverage": 0.0,
            "drift": -0.80,
        }

    def test_mismatched_lengths_truncate_to_the_shortest_of_all_three(self):
        """`n = min(y.size, lo.size, hi.size)` — all three, not any two.

        `actual` is deliberately the shortest here. A fixture where one of the
        *bounds* is shortest cannot see a `min` that has dropped `y.size`,
        because the answer comes out the same either way; this one reads past
        the end of `actual` instead.
        """
        drift = compute_interval_coverage_drift(np.zeros(3), np.full(5, -1.0), np.full(4, 1.0))

        assert drift["overall_coverage"] == 1.0, "three covered points, not four"

        # Same contract from the other two directions: whichever array is
        # shortest sets the length. Dropping either bound from the `min` reads
        # past the end of it, so these two calls are what pin `lo.size` and
        # `hi.size` specifically rather than "some pair of the three".
        lo_shortest = compute_interval_coverage_drift(
            np.zeros(5), np.full(3, -1.0), np.full(5, 1.0)
        )
        assert lo_shortest["overall_coverage"] == 1.0

        hi_shortest = compute_interval_coverage_drift(
            np.zeros(5), np.full(5, -1.0), np.full(3, 1.0)
        )
        assert hi_shortest["overall_coverage"] == 1.0

    def test_a_one_point_window_looks_at_one_point(self):
        """`recent_n = max(1, min(window_size, n))` — the floor is one, not two.

        Every other fixture uses a window of 3 or the 168-hour default, where
        `max(1, ...)` and `max(2, ...)` agree. At `window_size=1` they do not,
        and the reported "recent" coverage silently becomes a two-point average
        — the drift signal lagging the data by an hour.
        """
        actual = np.array([0.0, 0.0, 99.0, 0.0])  # third point outside
        lower, upper = np.full(4, -1.0), np.full(4, 1.0)

        drift = compute_interval_coverage_drift(actual, lower, upper, window_size=1)

        assert drift["recent_coverage"] == 1.0, "only the final point is recent"
        assert drift["overall_coverage"] == pytest.approx(0.75)

    def test_a_single_observation_is_measured_not_treated_as_empty(self):
        """`if n == 0` — one point is a real, if thin, measurement.

        Widened to `n == 1` the function short-circuits to the empty-input
        constants, reporting zero coverage and a full deficit for a point that
        was in fact covered. The observation has to be *covered* to see it: an
        uncovered single point happens to produce the same numbers as the
        empty return.
        """
        covered = compute_interval_coverage_drift(np.zeros(1), np.full(1, -1.0), np.full(1, 1.0))

        assert covered["overall_coverage"] == 1.0
        assert covered["recent_coverage"] == 1.0
        assert covered["drift"] == pytest.approx(0.20)

    def test_interval_coverage_drift(self):
        actual = np.array([10.0, 11.0, 9.0, 12.0, 13.0])
        lower = np.array([9.0, 10.0, 8.0, 11.0, 12.0])
        upper = np.array([11.0, 12.0, 10.0, 13.0, 14.0])
        monitor = compute_interval_coverage_drift(
            actual, lower, upper, target_coverage=0.8, window_size=3
        )
        assert monitor["overall_coverage"] == pytest.approx(1.0)
        assert monitor["recent_coverage"] == pytest.approx(1.0)
        assert monitor["drift"] == pytest.approx(0.2)
