"""Holdout-commensurability regression tests (#195, Workstream C — PR 2).

Before #195 the persisted XGBoost holdout MAPE was measured teacher-forced
one-step-ahead — ``predict_xgboost(model, val_df)`` where ``val_df`` already
carries real-demand lag features — while Prophet/SARIMAX holdouts are honest
multi-step forecasts. The numbers were incommensurable and biased the
inverse-MAPE ensemble weights toward XGBoost.

The fix routes XGBoost's holdout through the same recursive protocol production
serves: ``data.feature_engineering.recursive_autoregressive_forecast``. These
tests pin that the score is recursive (chains its own predictions) and not
teacher-forced (ignores observed in-window actuals) — the exact inverse of the
old behavior — and that the shared helper is the single source of truth.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from data.feature_engineering import recursive_autoregressive_forecast


def _future_df(n: int) -> pd.DataFrame:
    # Carries an autoregressive column the fake model reads, plus an in-window
    # "demand_mw" that a *teacher-forced* scorer would (wrongly) leak from.
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-20", periods=n, freq="h"),
            "demand_lag_1h": np.full(n, -999.0),  # sentinel: overwritten by the snapshot
            "demand_mw": np.arange(1000.0, 1000.0 + n),  # in-window actuals
        }
    )


def _lag1_predict(model, row):
    """Fake predict: returns the demand_lag_1h feature it is given. So the
    forecast reveals which lag source was used (recursive history vs the
    sentinel/leaked column)."""
    return [float(row["demand_lag_1h"].iloc[0])]


class TestRecursiveProtocol:
    def test_uses_recursive_history_not_the_sentinel_column(self):
        seed = [500.0] * 200  # real history; last value 500
        out = recursive_autoregressive_forecast(
            _lag1_predict, seed_demand=seed, future_df=_future_df(24), predict_fn=_lag1_predict
        )
        # Step 0's lag_1h == seed[-1] == 500 (from history), NOT the -999 sentinel
        # that sat in future_df["demand_lag_1h"]. Then each pred chains on itself.
        assert out[0] == 500.0
        assert np.allclose(out, 500.0)

    def test_perturbing_in_window_actuals_does_not_change_forecast(self):
        """Teacher-forcing would read val_df['demand_mw']; the recursive score
        must not — this is the core #195 assertion."""
        seed = [500.0] * 200
        base = recursive_autoregressive_forecast(None, seed, _future_df(24), _lag1_predict)

        perturbed = _future_df(24)
        perturbed["demand_mw"] = perturbed["demand_mw"] * 10.0  # wreck in-window actuals
        after = recursive_autoregressive_forecast(None, seed, perturbed, _lag1_predict)

        assert np.array_equal(base, after)  # unchanged → not teacher-forced

    def test_perturbing_seed_history_changes_forecast(self):
        base = recursive_autoregressive_forecast(None, [500.0] * 200, _future_df(24), _lag1_predict)
        changed = recursive_autoregressive_forecast(
            None, [800.0] * 200, _future_df(24), _lag1_predict
        )
        assert not np.array_equal(base, changed)
        assert changed[0] == 800.0

    def test_seed_filters_zero_and_nan(self):
        # Trailing 0 / NaN in seed must not become the lag (they'd poison it).
        seed = [500.0] * 200 + [0.0, float("nan")]
        out = recursive_autoregressive_forecast(None, seed, _future_df(4), _lag1_predict)
        assert out[0] == 500.0


class TestSharedSourceOfTruth:
    def test_helper_matches_phases_recursive_zone(self):
        """The production scorer and the holdout scorer must be the same code
        (#186 parity): phases._predict_xgboost_with_recursive_autoregressive's
        recursive zone == recursive_autoregressive_forecast on identical input."""
        import jobs.phases as phases

        featured = pd.DataFrame({"demand_mw": [500.0] * 200})
        future_df = _future_df(24)

        with patch("models.xgboost_model.predict_xgboost", side_effect=_lag1_predict):
            via_phases = phases._predict_xgboost_with_recursive_autoregressive(
                None, featured, future_df, horizon=24, recursive_hours=24
            )
        via_helper = recursive_autoregressive_forecast(
            None, featured["demand_mw"].tolist(), future_df.iloc[:24], _lag1_predict
        )
        assert np.array_equal(via_phases, via_helper)


class TestHoldoutReturnsRecursive:
    def _featured(self, n=900):
        # > _HOLDOUT_HOURS(168) + _MIN_TRAIN_HOURS(720)
        ts = pd.date_range("2026-01-01", periods=n, freq="h")
        return pd.DataFrame(
            {
                "timestamp": ts,
                "demand_mw": 1000.0 + np.arange(n),
                "demand_lag_1h": 0.0,
            }
        )

    def test_holdout_scores_recursively_and_logs_both(self):
        from jobs import training_job

        featured = self._featured()
        with (
            patch("models.xgboost_model.train_xgboost", return_value={"fake": True}),
            patch("models.xgboost_model.predict_xgboost", side_effect=_lag1_predict),
            patch.object(training_job.log, "info") as mock_log,
        ):
            out = training_job._holdout_metrics_xgboost(featured, "ERCOT")

        assert out is not None
        assert set(out) == {"metrics", "forecast", "actual"}
        assert len(out["forecast"]) == len(out["actual"]) == 168
        # The observability log carries both the recursive headline and the
        # teacher-forced comparison for one release.
        kwargs = mock_log.call_args.kwargs
        assert kwargs.get("protocol") == "recursive"
        assert "mape_teacher_forced" in kwargs
