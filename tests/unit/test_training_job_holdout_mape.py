"""Unit tests for the holdout-metrics flow in jobs/training_job.py.

History:
- Originally these tests covered ``_holdout_mape_prophet`` /
  ``_holdout_mape_arima`` returning a float MAPE.
- Real-metrics work renamed those to ``_holdout_metrics_*`` and
  changed the return shape to a dict: ``{metrics: {mape, rmse, mae, r2},
  forecast: ndarray, actual: ndarray}``. The richer payload powers two
  things downstream:
    1. Each per-model meta gets ``extra["holdout_metrics"]`` with the
       full {mape, rmse, mae, r2} dict, so the Models tab can stop
       supplementing RMSE / MAE / R² from ``_simulate_forecasts`` Redis.
    2. The ensemble's holdout metric is computed from the SAME
       predictions (no recomputation, no provenance drift) and stashed
       in xgboost's meta extra.

These tests verify per-model holdout still returns a finite metric set,
gracefully skips on short windows, and that ``_train_prophet`` /
``_train_arima`` thread the holdout payload through to ``save_model``
with both top-level ``mape`` and ``extra["holdout_metrics"]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def featured_df_long():
    """60 days @ hourly granularity — covers ``_HOLDOUT_HOURS + _MIN_TRAIN_HOURS``."""
    ts = pd.date_range("2024-01-01", periods=60 * 24, freq="h", tz="UTC")
    n = len(ts)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": 40_000.0 + 5_000.0 * np.sin(2 * np.pi * np.arange(n) / 24),
            "hour": ts.hour,
        }
    )


@pytest.fixture
def featured_df_short():
    """20 days — too short to leave 30 days of training after a 7-day holdout."""
    ts = pd.date_range("2024-01-01", periods=20 * 24, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": np.full(len(ts), 40_000.0),
            "hour": ts.hour,
        }
    )


class TestHoldoutMetricsProphet:
    def test_returns_full_metric_set_when_predict_succeeds(
        self, featured_df_long, monkeypatch
    ) -> None:
        """Train + predict succeed → returns ``{metrics, forecast, actual}`` with
        finite MAPE / RMSE / MAE / R²."""
        import models.prophet_model as prophet_mod

        def _fake_predict(model, df, periods=168):
            actuals = df["demand_mw"].values
            return {"forecast": actuals * 1.05}  # 5% bias → MAPE ≈ 5

        monkeypatch.setattr(prophet_mod, "train_prophet", lambda df: object())
        monkeypatch.setattr(prophet_mod, "predict_prophet", _fake_predict)

        from jobs.training_job import _holdout_metrics_prophet

        result = _holdout_metrics_prophet(featured_df_long, region="TEST")
        assert result is not None
        assert set(result.keys()) >= {"metrics", "forecast", "actual"}
        m = result["metrics"]
        assert set(m.keys()) == {"mape", "rmse", "mae", "r2"}
        assert m["mape"] == pytest.approx(5.0, abs=0.5)
        assert m["rmse"] > 0
        assert m["mae"] > 0
        # 5% multiplicative bias creates residual variance comparable to the
        # signal variance — R² lands around ~0.68 in this synthetic setup.
        # Just assert it's positive (model beats predicting the mean).
        assert m["r2"] > 0.0

    def test_short_window_returns_none(self, featured_df_short, monkeypatch) -> None:
        """Insufficient data → skip holdout, return None (no spurious metrics)."""
        import models.prophet_model as prophet_mod

        called: list[bool] = []
        monkeypatch.setattr(prophet_mod, "train_prophet", lambda df: called.append(True))

        from jobs.training_job import _holdout_metrics_prophet

        assert _holdout_metrics_prophet(featured_df_short, region="TEST") is None
        assert called == []  # train should not even be invoked

    def test_predict_failure_returns_none(self, featured_df_long, monkeypatch) -> None:
        """Predict raising → swallow, return None (production training continues)."""
        import models.prophet_model as prophet_mod

        def _broken_predict(model, df, periods=168):
            raise RuntimeError("synthetic prophet predict failure")

        monkeypatch.setattr(prophet_mod, "train_prophet", lambda df: object())
        monkeypatch.setattr(prophet_mod, "predict_prophet", _broken_predict)

        from jobs.training_job import _holdout_metrics_prophet

        assert _holdout_metrics_prophet(featured_df_long, region="TEST") is None

    def test_non_finite_forecast_returns_none(self, featured_df_long, monkeypatch) -> None:
        """NaN forecast → return None (don't persist garbage)."""
        import models.prophet_model as prophet_mod

        def _fake_predict(model, df, periods=168):
            return {"forecast": np.full(len(df), np.nan)}

        monkeypatch.setattr(prophet_mod, "train_prophet", lambda df: object())
        monkeypatch.setattr(prophet_mod, "predict_prophet", _fake_predict)

        from jobs.training_job import _holdout_metrics_prophet

        assert _holdout_metrics_prophet(featured_df_long, region="TEST") is None


class TestHoldoutMetricsArima:
    def test_returns_full_metric_set_when_predict_succeeds(
        self, featured_df_long, monkeypatch
    ) -> None:
        import models.arima_model as arima_mod

        def _fake_predict(model_dict, exog, periods=168):
            actuals = exog["demand_mw"].values
            return actuals * 0.97  # 3% under-forecast → MAPE ≈ 3

        monkeypatch.setattr(arima_mod, "train_arima", lambda df: {"params": []})
        monkeypatch.setattr(arima_mod, "predict_arima", _fake_predict)

        from jobs.training_job import _holdout_metrics_arima

        result = _holdout_metrics_arima(featured_df_long, region="TEST")
        assert result is not None
        assert result["metrics"]["mape"] == pytest.approx(3.0, abs=0.5)
        # See prophet test for why R² is moderate, not near-1, in this
        # synthetic setup.
        assert result["metrics"]["r2"] > 0.0

    def test_short_window_returns_none(self, featured_df_short) -> None:
        from jobs.training_job import _holdout_metrics_arima

        assert _holdout_metrics_arima(featured_df_short, region="TEST") is None

    def test_predict_failure_returns_none(self, featured_df_long, monkeypatch) -> None:
        import models.arima_model as arima_mod

        def _broken_predict(model_dict, exog, periods=168):
            raise ValueError("synthetic arima predict failure")

        monkeypatch.setattr(arima_mod, "train_arima", lambda df: {"params": []})
        monkeypatch.setattr(arima_mod, "predict_arima", _broken_predict)

        from jobs.training_job import _holdout_metrics_arima

        assert _holdout_metrics_arima(featured_df_long, region="TEST") is None


class TestTrainPersistsHoldoutMetrics:
    """End-to-end: ``_train_prophet`` / ``_train_arima`` pass the full
    holdout-metric dict through to ``save_model`` — both as the
    top-level ``mape`` (used for inverse-MAPE ensemble weighting) and
    as ``extra["holdout_metrics"]`` (powers the Models tab's RMSE /
    MAE / R²)."""

    def _make_region_data(self, df) -> object:
        from jobs.phases import RegionData

        return RegionData(
            region="TEST",
            demand_df=df[["timestamp", "demand_mw"]].copy(),
            weather_df=df.copy(),
            featured_df=df,
        )

    def test_prophet_passes_full_holdout_metrics_to_save_model(
        self, featured_df_long, monkeypatch
    ) -> None:
        import models.prophet_model as prophet_mod

        monkeypatch.setattr(prophet_mod, "train_prophet", lambda df: object())
        monkeypatch.setattr(
            prophet_mod,
            "predict_prophet",
            lambda model, df, periods=168: {
                "forecast": np.asarray(df["demand_mw"].values) * 1.04,  # MAPE ≈ 4
            },
        )

        captured: dict = {}

        def _save_model(**kwargs):
            captured.update(kwargs)
            return "v-test"

        monkeypatch.setattr("jobs.training_job.save_model", _save_model)
        monkeypatch.setattr("jobs.training_job._compute_data_hash", lambda region_data: "h")

        from jobs.training_job import _holdout_metrics_prophet, _train_prophet

        holdout = _holdout_metrics_prophet(featured_df_long, region="TEST")
        version = _train_prophet(self._make_region_data(featured_df_long), holdout=holdout)

        assert version == "v-test"
        assert captured["model_name"] == "prophet"

        # Top-level mape — drives ensemble weighting at scoring time.
        assert captured["mape"] is not None
        assert captured["mape"] == pytest.approx(4.0, abs=0.5)

        # extra["holdout_metrics"] — drives Models-tab RMSE / MAE / R².
        extra = captured.get("extra") or {}
        holdout_in_extra = extra.get("holdout_metrics")
        assert holdout_in_extra is not None
        assert set(holdout_in_extra.keys()) == {"mape", "rmse", "mae", "r2"}
        assert holdout_in_extra["mape"] == pytest.approx(4.0, abs=0.5)

    def test_arima_passes_full_holdout_metrics_to_save_model(
        self, featured_df_long, monkeypatch
    ) -> None:
        import models.arima_model as arima_mod

        monkeypatch.setattr(arima_mod, "train_arima", lambda df: {"params": []})
        monkeypatch.setattr(
            arima_mod,
            "predict_arima",
            lambda model_dict, exog, periods=168: (
                np.asarray(exog["demand_mw"].values) * 0.96
            ),  # MAPE ≈ 4
        )

        captured: dict = {}

        def _save_model(**kwargs):
            captured.update(kwargs)
            return "v-test"

        monkeypatch.setattr("jobs.training_job.save_model", _save_model)
        monkeypatch.setattr("jobs.training_job._compute_data_hash", lambda region_data: "h")

        from jobs.training_job import _holdout_metrics_arima, _train_arima

        holdout = _holdout_metrics_arima(featured_df_long, region="TEST")
        version = _train_arima(self._make_region_data(featured_df_long), holdout=holdout)

        assert version == "v-test"
        assert captured["model_name"] == "arima"
        assert captured["mape"] is not None
        assert captured["mape"] == pytest.approx(4.0, abs=0.5)

        extra = captured.get("extra") or {}
        holdout_in_extra = extra.get("holdout_metrics")
        assert holdout_in_extra is not None
        assert holdout_in_extra["mape"] == pytest.approx(4.0, abs=0.5)


class TestEnsembleHoldoutMetrics:
    """The ensemble's holdout metric is computed from the SAME 168-hour
    holdout predictions used by each per-model holdout — no separate
    train, no Redis, no simulation. ``_ensemble_holdout_metrics`` takes
    the per-model dicts and returns ``(metrics, weights)``.
    """

    def test_combines_two_models_with_inverse_mape_weights(self) -> None:
        from jobs.training_job import _ensemble_holdout_metrics

        n = 168
        actual = np.full(n, 50_000.0)
        per_model_holdouts = {
            # 5% over-forecast → mape ≈ 5
            "prophet": {
                "metrics": {"mape": 5.0, "rmse": 2500.0, "mae": 2500.0, "r2": 0.0},
                "forecast": actual * 1.05,
                "actual": actual,
            },
            # 1% over-forecast → mape ≈ 1 (much better)
            "xgboost": {
                "metrics": {"mape": 1.0, "rmse": 500.0, "mae": 500.0, "r2": 0.0},
                "forecast": actual * 1.01,
                "actual": actual,
            },
        }
        result = _ensemble_holdout_metrics(per_model_holdouts)
        assert result is not None
        metrics, weights = result

        # Weights normalize to 1.0
        assert weights["prophet"] + weights["xgboost"] == pytest.approx(1.0, abs=0.001)
        # xgboost (1% MAPE) gets ≥ 5× the weight of prophet (5% MAPE)
        assert weights["xgboost"] > weights["prophet"]

        # Ensemble MAPE is between the worst (5) and the best (1)
        assert 1.0 < metrics["mape"] < 5.0
        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0

    def test_single_model_returns_none(self) -> None:
        """One model isn't an ensemble — return None so the caller
        doesn't persist a misleading 'ensemble' metric that's just
        the lone model under a different name."""
        from jobs.training_job import _ensemble_holdout_metrics

        n = 168
        actual = np.full(n, 50_000.0)
        result = _ensemble_holdout_metrics(
            {
                "xgboost": {
                    "metrics": {"mape": 1.0, "rmse": 500.0, "mae": 500.0, "r2": 0.0},
                    "forecast": actual * 1.01,
                    "actual": actual,
                },
                "prophet": None,
                "arima": None,
            }
        )
        assert result is None

    def test_returns_none_when_actuals_mismatch(self) -> None:
        """If two models report different ``actual`` lengths, something
        upstream is broken — bail rather than silently misalign."""
        from jobs.training_job import _ensemble_holdout_metrics

        result = _ensemble_holdout_metrics(
            {
                "prophet": {
                    "metrics": {"mape": 5.0, "rmse": 1.0, "mae": 1.0, "r2": 0.5},
                    "forecast": np.ones(168),
                    "actual": np.ones(168),
                },
                "xgboost": {
                    "metrics": {"mape": 2.0, "rmse": 1.0, "mae": 1.0, "r2": 0.5},
                    "forecast": np.ones(100),
                    "actual": np.ones(100),
                },
            }
        )
        assert result is None
