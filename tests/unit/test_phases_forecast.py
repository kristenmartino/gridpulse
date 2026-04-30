"""Unit tests for the multi-model forecast phase in jobs/phases.py.

Covers ``predict_and_write_forecast``'s ensemble-weighting policy:

- Inverse-MAPE weights when *every* predicting model has a valid MAPE.
- Equal-weights fallback when MAPE coverage is partial — without this
  fallback the inverse-MAPE blend silently degrades to whichever model
  happens to have its MAPE recorded (the bug behind ``ensemble = xgboost``
  observed in production after option B Stage 3 shipped).
- Equal-weights fallback when no model has a MAPE.
- No ensemble row when only one model produces predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

HORIZON = 720


@pytest.fixture
def fake_redis(monkeypatch):
    """Capture ``redis_set`` writes in an in-memory dict."""
    store: dict[str, dict] = {}

    def _set(key: str, value, ttl: int = 86400) -> bool:
        store[key] = value
        return True

    import data.redis_client as rc

    monkeypatch.setattr(rc, "redis_set", _set)
    return store


@pytest.fixture
def region_data():
    """Minimal RegionData with a featured_df just large enough for the phase."""
    from jobs.phases import RegionData

    ts = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    demand_df = pd.DataFrame({"timestamp": ts, "demand_mw": np.full(len(ts), 40_000.0)})
    featured_df = pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": np.full(len(ts), 40_000.0),
            "hour": ts.hour,
        }
    )
    return RegionData(
        region="ERCOT",
        demand_df=demand_df,
        weather_df=demand_df.copy(),
        featured_df=featured_df,
    )


def _patch_predict_one(monkeypatch, predictions_by_name):
    """Patch ``_predict_one`` to dispatch by model name with synthetic arrays."""
    import jobs.phases as phases

    def _fake_predict_one(name, model, featured, future_df, horizon):
        return predictions_by_name.get(name)

    monkeypatch.setattr(phases, "_predict_one", _fake_predict_one)
    monkeypatch.setattr(
        phases,
        "_build_future_feature_frame",
        lambda featured, horizon: pd.DataFrame(
            {"timestamp": pd.date_range("2024-02-01", periods=horizon, freq="h", tz="UTC")}
        ),
    )


class TestPredictAndWriteForecast:
    def test_full_mape_uses_inverse_mape_weights(self, fake_redis, region_data, monkeypatch):
        """When every predicting model has a valid MAPE, weights ∝ 1/MAPE."""
        from jobs import phases

        xgb_preds = np.full(HORIZON, 41_000.0)
        prophet_preds = np.full(HORIZON, 39_000.0)
        arima_preds = np.full(HORIZON, 40_000.0)
        _patch_predict_one(
            monkeypatch,
            {"xgboost": xgb_preds, "prophet": prophet_preds, "arima": arima_preds},
        )

        result = phases.predict_and_write_forecast(
            region_data,
            models={"xgboost": object(), "prophet": object(), "arima": object()},
            model_mapes={"xgboost": 1.0, "prophet": 2.0, "arima": 4.0},
        )

        assert result.ok
        payload = fake_redis["wattcast:forecast:ERCOT:1h"]
        weights = payload["ensemble_weights"]
        # 1/1 : 1/2 : 1/4 = 0.5714 : 0.2857 : 0.1429 (rounded to 4dp)
        assert weights["xgboost"] == pytest.approx(0.5714, abs=1e-3)
        assert weights["prophet"] == pytest.approx(0.2857, abs=1e-3)
        assert weights["arima"] == pytest.approx(0.1429, abs=1e-3)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)

        # Ensemble pred for any row = 0.5714*41000 + 0.2857*39000 + 0.1429*40000
        row0 = payload["forecasts"][0]
        expected = 0.5714 * 41_000 + 0.2857 * 39_000 + 0.1429 * 40_000
        assert row0["ensemble"] == pytest.approx(expected, rel=1e-3)
        assert row0["xgboost"] == 41_000.0
        assert row0["prophet"] == 39_000.0
        assert row0["arima"] == 40_000.0

    def test_partial_mape_falls_back_to_equal_weights(self, fake_redis, region_data, monkeypatch):
        """Only xgboost has MAPE → 1/3 each across all 3 predicting models.

        This is the production state observed in 2026-04-30 V0 verification:
        before the fix, the ensemble collapsed to ``{xgboost: 1.0}`` and the
        ``ensemble`` field in Redis was identical to xgboost.
        """
        from jobs import phases

        xgb_preds = np.full(HORIZON, 41_000.0)
        prophet_preds = np.full(HORIZON, 39_000.0)
        arima_preds = np.full(HORIZON, 40_000.0)
        _patch_predict_one(
            monkeypatch,
            {"xgboost": xgb_preds, "prophet": prophet_preds, "arima": arima_preds},
        )

        result = phases.predict_and_write_forecast(
            region_data,
            models={"xgboost": object(), "prophet": object(), "arima": object()},
            model_mapes={"xgboost": 5.0, "prophet": None, "arima": None},
        )

        assert result.ok
        payload = fake_redis["wattcast:forecast:ERCOT:1h"]
        weights = payload["ensemble_weights"]
        assert weights == {"xgboost": 1 / 3, "prophet": 1 / 3, "arima": 1 / 3} or (
            weights["xgboost"] == pytest.approx(1 / 3, abs=1e-3)
            and weights["prophet"] == pytest.approx(1 / 3, abs=1e-3)
            and weights["arima"] == pytest.approx(1 / 3, abs=1e-3)
        )

        # Equal-weights ensemble = mean(41000, 39000, 40000) = 40000
        row0 = payload["forecasts"][0]
        assert row0["ensemble"] == pytest.approx(40_000.0, abs=1)
        # Critical: ensemble must NOT equal xgboost-alone (the prior bug).
        assert row0["ensemble"] != row0["xgboost"]

    def test_no_mape_falls_back_to_equal_weights(self, fake_redis, region_data, monkeypatch):
        """No model has MAPE → equal weights, every model contributes."""
        from jobs import phases

        _patch_predict_one(
            monkeypatch,
            {
                "xgboost": np.full(HORIZON, 41_000.0),
                "prophet": np.full(HORIZON, 39_000.0),
            },
        )

        result = phases.predict_and_write_forecast(
            region_data,
            models={"xgboost": object(), "prophet": object()},
            model_mapes=None,
        )

        assert result.ok
        payload = fake_redis["wattcast:forecast:ERCOT:1h"]
        weights = payload["ensemble_weights"]
        assert weights["xgboost"] == pytest.approx(0.5, abs=1e-6)
        assert weights["prophet"] == pytest.approx(0.5, abs=1e-6)
        assert payload["forecasts"][0]["ensemble"] == pytest.approx(40_000.0, abs=1)

    def test_single_model_omits_ensemble(self, fake_redis, region_data, monkeypatch):
        """Only one model predicted → no ``ensemble`` field, no ``ensemble_weights``."""
        from jobs import phases

        _patch_predict_one(monkeypatch, {"xgboost": np.full(HORIZON, 41_000.0)})

        result = phases.predict_and_write_forecast(
            region_data,
            models={"xgboost": object(), "prophet": object()},  # prophet returns None
            model_mapes={"xgboost": 5.0},
        )

        assert result.ok
        payload = fake_redis["wattcast:forecast:ERCOT:1h"]
        assert "ensemble_weights" not in payload
        row0 = payload["forecasts"][0]
        assert "ensemble" not in row0
        assert row0["xgboost"] == 41_000.0
        assert row0["predicted_demand_mw"] == 41_000.0
