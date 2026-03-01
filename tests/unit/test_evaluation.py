"""Unit tests for models/evaluation.py."""

import numpy as np
import pandas as pd
import pytest

from models.evaluation import (
    compute_all_metrics,
    compute_error_by_hour,
    compute_mae,
    compute_mape,
    compute_r2,
    compute_residuals,
    compute_rmse,
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
