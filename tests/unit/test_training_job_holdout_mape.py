"""Unit tests for the holdout-MAPE flow in jobs/training_job.py.

Before this change, ``_train_prophet`` and ``_train_arima`` always passed
``mape=None`` to ``save_model``. Scoring then dropped Prophet/ARIMA from
the inverse-MAPE blend (because their MAPE was ``None``) and the
"ensemble" forecast in Redis collapsed to xgboost-only. These tests
verify the holdout MAPE computation path now feeds real values into
``save_model``, and gracefully falls back when the holdout window is
too short or predict raises.
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


class TestHoldoutMapeProphet:
    def test_returns_mape_when_predict_succeeds(self, featured_df_long, monkeypatch) -> None:
        """Train + predict succeed → returns finite positive MAPE."""
        import models.prophet_model as prophet_mod

        # Forecast offset by a constant so MAPE is non-zero.
        def _fake_predict(model, df, periods=168):
            actuals = df["demand_mw"].values
            return {"forecast": actuals * 1.05}  # 5% bias → MAPE ≈ 5

        monkeypatch.setattr(prophet_mod, "train_prophet", lambda df: object())
        monkeypatch.setattr(prophet_mod, "predict_prophet", _fake_predict)

        from jobs.training_job import _holdout_mape_prophet

        mape = _holdout_mape_prophet(featured_df_long, region="TEST")
        assert mape is not None
        assert mape == pytest.approx(5.0, abs=0.5)

    def test_short_window_returns_none(self, featured_df_short, monkeypatch) -> None:
        """Insufficient data → skip holdout, return None (no spurious MAPE)."""
        import models.prophet_model as prophet_mod

        called: list[bool] = []
        monkeypatch.setattr(prophet_mod, "train_prophet", lambda df: called.append(True))

        from jobs.training_job import _holdout_mape_prophet

        assert _holdout_mape_prophet(featured_df_short, region="TEST") is None
        assert called == []  # train should not even be invoked

    def test_predict_failure_returns_none(self, featured_df_long, monkeypatch) -> None:
        """Predict raising → swallow, return None (production training continues)."""
        import models.prophet_model as prophet_mod

        def _broken_predict(model, df, periods=168):
            raise RuntimeError("synthetic prophet predict failure")

        monkeypatch.setattr(prophet_mod, "train_prophet", lambda df: object())
        monkeypatch.setattr(prophet_mod, "predict_prophet", _broken_predict)

        from jobs.training_job import _holdout_mape_prophet

        assert _holdout_mape_prophet(featured_df_long, region="TEST") is None

    def test_non_finite_mape_returns_none(self, featured_df_long, monkeypatch) -> None:
        """Forecast with zero actuals → MAPE blows up → return None."""
        import models.prophet_model as prophet_mod

        def _fake_predict(model, df, periods=168):
            # NaN forecast → compute_mape returns NaN
            return {"forecast": np.full(len(df), np.nan)}

        monkeypatch.setattr(prophet_mod, "train_prophet", lambda df: object())
        monkeypatch.setattr(prophet_mod, "predict_prophet", _fake_predict)

        from jobs.training_job import _holdout_mape_prophet

        assert _holdout_mape_prophet(featured_df_long, region="TEST") is None


class TestHoldoutMapeArima:
    def test_returns_mape_when_predict_succeeds(self, featured_df_long, monkeypatch) -> None:
        import models.arima_model as arima_mod

        def _fake_predict(model_dict, exog, periods=168):
            actuals = exog["demand_mw"].values
            return actuals * 0.97  # 3% under-forecast → MAPE ≈ 3

        monkeypatch.setattr(arima_mod, "train_arima", lambda df: {"params": []})
        monkeypatch.setattr(arima_mod, "predict_arima", _fake_predict)

        from jobs.training_job import _holdout_mape_arima

        mape = _holdout_mape_arima(featured_df_long, region="TEST")
        assert mape is not None
        assert mape == pytest.approx(3.0, abs=0.5)

    def test_short_window_returns_none(self, featured_df_short) -> None:
        from jobs.training_job import _holdout_mape_arima

        assert _holdout_mape_arima(featured_df_short, region="TEST") is None

    def test_predict_failure_returns_none(self, featured_df_long, monkeypatch) -> None:
        import models.arima_model as arima_mod

        def _broken_predict(model_dict, exog, periods=168):
            raise ValueError("synthetic arima predict failure")

        monkeypatch.setattr(arima_mod, "train_arima", lambda df: {"params": []})
        monkeypatch.setattr(arima_mod, "predict_arima", _broken_predict)

        from jobs.training_job import _holdout_mape_arima

        assert _holdout_mape_arima(featured_df_long, region="TEST") is None


class TestTrainPersistsMape:
    """End-to-end: train_prophet/_train_arima pass real MAPE to save_model.

    This is the key behavior the scoring-side fix relies on — without a
    real MAPE in the saved metadata, scoring's inverse-MAPE blend has
    nothing to weight by and falls back to equal weights.
    """

    def _make_region_data(self, df) -> object:
        from jobs.phases import RegionData

        return RegionData(
            region="TEST",
            demand_df=df[["timestamp", "demand_mw"]].copy(),
            weather_df=df.copy(),
            featured_df=df,
        )

    def test_prophet_passes_holdout_mape_to_save_model(self, featured_df_long, monkeypatch) -> None:
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
        # Stub the data hash helper to avoid pulling in the full feature pipeline.
        monkeypatch.setattr("jobs.training_job._compute_data_hash", lambda region_data: "h")

        from jobs.training_job import _train_prophet

        version = _train_prophet(self._make_region_data(featured_df_long))
        assert version == "v-test"
        assert captured["model_name"] == "prophet"
        assert captured["mape"] is not None
        assert captured["mape"] == pytest.approx(4.0, abs=0.5)

    def test_arima_passes_holdout_mape_to_save_model(self, featured_df_long, monkeypatch) -> None:
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

        from jobs.training_job import _train_arima

        version = _train_arima(self._make_region_data(featured_df_long))
        assert version == "v-test"
        assert captured["model_name"] == "arima"
        assert captured["mape"] is not None
        assert captured["mape"] == pytest.approx(4.0, abs=0.5)
