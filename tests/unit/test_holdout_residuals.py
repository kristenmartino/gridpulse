"""Per-model holdout-residuals groundwork tests (#196 / #181).

The training job persists every model's holdout forecast vs the shared actuals
to ``gridpulse:holdout:{region}`` (zero extra compute). The web tier's interval
collector then uses each model's OWN residuals to calibrate its prediction band,
instead of substituting XGBoost's residuals (#196, P1-2) — and the same payload
is the per-model, commensurable data #181 needs.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from jobs import training_job


def _holdout(forecast, actual):
    return {
        "metrics": {"mape": 3.0, "rmse": 1.0, "mae": 1.0, "r2": 0.9},
        "forecast": np.asarray(forecast, dtype=float),
        "actual": np.asarray(actual, dtype=float),
    }


class TestPersistHoldoutResiduals:
    def test_persists_per_model_predictions_plus_ensemble(self):
        actual = [100.0, 110.0, 120.0, 130.0]
        per_model = {
            "xgboost": _holdout([101.0, 109.0, 119.0, 131.0], actual),
            "prophet": _holdout([95.0, 105.0, 118.0, 128.0], actual),
            "arima": _holdout([102.0, 112.0, 121.0, 129.0], actual),
        }
        # Equal weights → ensemble forecast is the per-model mean.
        ensemble_summary = ({"mape": 2.5}, {"xgboost": 1 / 3, "prophet": 1 / 3, "arima": 1 / 3})

        captured: dict = {}
        with patch(
            "data.redis_client.redis_set",
            side_effect=lambda k, p, ttl=None: captured.update(key=k, payload=p, ttl=ttl) or True,
        ):
            training_job._persist_holdout_residuals("ERCOT", per_model, ensemble_summary)

        payload = captured["payload"]
        assert captured["key"] == "gridpulse:holdout:ERCOT"
        assert payload["source"] == "training_holdout"
        assert payload["actual"] == actual
        preds = payload["predictions"]
        assert set(preds) == {"xgboost", "prophet", "arima", "ensemble"}
        # Ensemble is the equal-weight mean of the three per-model forecasts.
        expected_ens0 = (101.0 + 95.0 + 102.0) / 3
        assert abs(preds["ensemble"][0] - expected_ens0) < 1e-6

    def test_skips_models_that_dropped_out(self):
        actual = [100.0, 110.0, 120.0]
        per_model = {
            "xgboost": _holdout([101.0, 109.0, 119.0], actual),
            "prophet": None,  # holdout failed
            "arima": None,
        }
        captured: dict = {}
        with patch(
            "data.redis_client.redis_set",
            side_effect=lambda k, p, ttl=None: captured.update(payload=p) or True,
        ):
            training_job._persist_holdout_residuals("ERCOT", per_model, None)

        preds = captured["payload"]["predictions"]
        assert set(preds) == {"xgboost"}  # only the survivor; no ensemble (needs >=2)

    def test_all_dropped_out_writes_nothing(self):
        calls: list = []
        with patch("data.redis_client.redis_set", side_effect=lambda *a, **k: calls.append(a)):
            training_job._persist_holdout_residuals(
                "ERCOT", {"xgboost": None, "prophet": None, "arima": None}, None
            )
        assert calls == []


class TestCollectorUsesExactModelHoldout:
    def _holdout_payload(self):
        # ensemble residuals are distinctly non-XGBoost so we can tell them apart.
        actual = [100.0] * 10
        return {
            "region": "ERCOT",
            "predictions": {
                "xgboost": [101.0] * 10,  # residual -1 each
                "ensemble": [90.0] * 10,  # residual +10 each
            },
            "actual": actual,
        }

    def test_ensemble_uses_own_residuals_not_xgboost_substitute(self):
        import components._callbacks_shared as shared

        # No backtest key (only xgboost lived there before); holdout has both.
        def fake_get(key):
            return self._holdout_payload() if key.endswith("holdout:ERCOT") else None

        with patch.object(shared, "redis_get", side_effect=fake_get):
            residuals, calib = shared._collect_backtest_residuals("ERCOT", "ensemble", 24)

        # Exact ensemble residuals (+10), NOT the xgboost substitute (-1).
        assert calib == "ensemble"
        assert np.allclose(residuals, 10.0)

    def test_exact_backtest_still_preferred_over_holdout(self):
        import components._callbacks_shared as shared

        backtest = {
            "actual": [100.0] * 5,
            "predictions": {"ensemble": [100.5] * 5},  # residual -0.5
        }

        def fake_get(key):
            if "backtest" in key:
                return backtest
            if key.endswith("holdout:ERCOT"):
                return self._holdout_payload()
            return None

        with patch.object(shared, "redis_get", side_effect=fake_get):
            residuals, calib = shared._collect_backtest_residuals("ERCOT", "ensemble", 24)

        assert calib == "ensemble"
        # Both the backtest (-0.5) and holdout (+10) exact chunks are pooled;
        # the point is neither substitutes XGBoost.
        assert -0.5 in np.round(residuals, 3)
