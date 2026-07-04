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

    def _fake_predict_one(name, model, featured, future_df, horizon, start_ts=None):
        return predictions_by_name.get(name)

    monkeypatch.setattr(phases, "_predict_one", _fake_predict_one)
    monkeypatch.setattr(
        phases,
        "_build_future_feature_frame",
        # PR-C (2026-05-20): function gained ``weather_df`` kwarg.
        # #129 (2026-05-21): function gained ``start_ts`` kwarg.
        # Lambda accepts and ignores both — tests here exercise
        # ``predict_and_write_forecast``'s ensemble logic, not the
        # future-frame builder itself (which has its own dedicated
        # test classes below).
        lambda featured, horizon, weather_df=None, start_ts=None: pd.DataFrame(
            {"timestamp": pd.date_range("2024-02-01", periods=horizon, freq="h", tz="UTC")}
        ),
    )


class TestPredictAndWriteForecast:
    def test_full_mape_uses_inverse_mape_weights(self, fake_redis, region_data, monkeypatch):
        """When every predicting model has a valid MAPE, weights ∝ (1/MAPE)^k,
        k=ENSEMBLE_WEIGHT_EXPONENT=3 (ADR-004 / #181 sharpened blend)."""
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
        payload = fake_redis["gridpulse:forecast:ERCOT:1h"]
        weights = payload["ensemble_weights"]
        # (1/1)^3 : (1/2)^3 : (1/4)^3 = 1 : 0.125 : 0.015625, normalized (k=3)
        assert weights["xgboost"] == pytest.approx(0.8767, abs=1e-3)
        assert weights["prophet"] == pytest.approx(0.1096, abs=1e-3)
        assert weights["arima"] == pytest.approx(0.0137, abs=1e-3)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)

        # Ensemble pred for any row = 0.8767*41000 + 0.1096*39000 + 0.0137*40000
        row0 = payload["forecasts"][0]
        expected = 0.8767 * 41_000 + 0.1096 * 39_000 + 0.0137 * 40_000
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
        payload = fake_redis["gridpulse:forecast:ERCOT:1h"]
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
        payload = fake_redis["gridpulse:forecast:ERCOT:1h"]
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
        payload = fake_redis["gridpulse:forecast:ERCOT:1h"]
        assert "ensemble_weights" not in payload
        row0 = payload["forecasts"][0]
        assert "ensemble" not in row0
        assert row0["xgboost"] == 41_000.0
        assert row0["predicted_demand_mw"] == 41_000.0


# ────────────────────────────────────────────────────────────────────────
# PR-C (2026-05-20) — Open-Meteo forecast overlay on future feature frame
# ────────────────────────────────────────────────────────────────────────


def _build_featured_hist(n_hours: int = 168 * 2, last_ts: str = "2026-05-20 14:00") -> pd.DataFrame:
    """Build a synthetic engineered-historical DataFrame ending at ``last_ts``.

    Includes raw weather columns + derived features so it looks like
    real output from ``engineer_features``. Used to exercise the
    climatology baseline + weather overlay path.
    """
    end = pd.Timestamp(last_ts, tz="UTC")
    ts = pd.date_range(end=end, periods=n_hours, freq="h")
    hours = np.arange(n_hours)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": 20_000 + 5_000 * np.sin(2 * np.pi * hours / 24),
            "temperature_2m": 70.0 + 15.0 * np.sin(2 * np.pi * hours / 24),
            "apparent_temperature": 72.0 + 15.0 * np.sin(2 * np.pi * hours / 24),
            "wind_speed_80m": 12.0 + 3.0 * np.sin(2 * np.pi * hours / 12),
            "shortwave_radiation": np.maximum(0, 600 * np.sin(2 * np.pi * hours / 24)),
            "cloud_cover": 30.0 + 20.0 * np.cos(2 * np.pi * hours / 24),
            "cooling_degree_days": np.maximum(0, 70.0 + 15.0 * np.sin(2 * np.pi * hours / 24) - 65),
            "heating_degree_days": np.maximum(
                0, 65 - (70.0 + 15.0 * np.sin(2 * np.pi * hours / 24))
            ),
            "temperature_deviation": np.zeros(n_hours),
            "wind_power_estimate": 0.4 + 0.1 * np.sin(2 * np.pi * hours / 12),
            "solar_capacity_factor": np.maximum(0, 0.6 * np.sin(2 * np.pi * hours / 24)),
            "demand_lag_24h": 20_000 + 5_000 * np.sin(2 * np.pi * (hours - 24) / 24),
        }
    )


def _build_weather_forecast(
    start_ts: str = "2026-05-20 15:00",
    n_hours: int = 168,
    temperature: float = 95.0,  # deliberately HOT, distinct from historical baseline
) -> pd.DataFrame:
    """Build a synthetic weather forecast DataFrame for the first ``n_hours``
    after ``start_ts``. Constant temperature so test assertions are easy.
    """
    start = pd.Timestamp(start_ts, tz="UTC")
    ts = pd.date_range(start=start, periods=n_hours, freq="h")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "temperature_2m": np.full(n_hours, temperature),
            "apparent_temperature": np.full(n_hours, temperature + 2),
            "wind_speed_80m": np.full(n_hours, 18.0),
            "shortwave_radiation": np.full(n_hours, 750.0),
            "cloud_cover": np.full(n_hours, 5.0),  # clear sky
        }
    )


class TestBuildFutureFeatureFrameNoOverlay:
    """PR-C invariants: without weather_df, behavior matches pre-PR-C climatology."""

    def test_no_weather_df_falls_back_to_climatology(self):
        from jobs.phases import _build_future_feature_frame

        featured = _build_featured_hist()
        future_df = _build_future_feature_frame(featured, horizon=24)

        # All future temperatures should come from (hour, dow) group means.
        # Historical mean temperature is ~70°F (the baseline of our synthetic
        # signal), so future temperatures cluster near 70°F.
        assert future_df["temperature_2m"].between(50.0, 90.0).all()
        # NOT the test forecast value (95) — that's only present when
        # weather_df is passed.
        assert (future_df["temperature_2m"] == pytest.approx(95.0, abs=1e-3)).sum() == 0


class TestOverlayWeatherForecast:
    """PR-C — actual forecast overlay onto climatology baseline."""

    def test_overlay_within_horizon_uses_forecast_values(self):
        """For future hours covered by weather_df, raw weather columns
        must match the forecast values, NOT climatology."""
        from jobs.phases import _build_future_feature_frame

        featured = _build_featured_hist()
        weather_df = _build_weather_forecast(n_hours=168, temperature=95.0)
        future_df = _build_future_feature_frame(featured, horizon=168, weather_df=weather_df)

        # Every hour of the 168-hour horizon is covered by weather_df.
        # Use np.allclose for series-wide approx comparisons —
        # ``Series == pytest.approx(scalar)`` doesn't broadcast as expected.
        assert np.allclose(future_df["temperature_2m"].values, 95.0)
        assert np.allclose(future_df["wind_speed_80m"].values, 18.0)
        assert np.allclose(future_df["shortwave_radiation"].values, 750.0)

    def test_overlay_recomputes_derived_features(self):
        """When raw weather is overlaid with forecast, derived features
        (CDD/HDD/wind_power/solar_cf/temp_x_hour) must be recomputed
        from the FORECAST values — not left at climatological values."""
        from data.feature_engineering import (
            compute_cdd,
            compute_hdd,
            compute_solar_capacity_factor,
            compute_wind_power,
        )
        from jobs.phases import _build_future_feature_frame

        featured = _build_featured_hist()
        # Forecast temp 95°F → CDD = 30, HDD = 0
        weather_df = _build_weather_forecast(n_hours=168, temperature=95.0)
        future_df = _build_future_feature_frame(featured, horizon=168, weather_df=weather_df)

        expected_cdd = float(compute_cdd(pd.Series([95.0])).iloc[0])
        expected_hdd = float(compute_hdd(pd.Series([95.0])).iloc[0])
        expected_wind = float(compute_wind_power(pd.Series([18.0])).iloc[0])
        expected_solar = float(compute_solar_capacity_factor(pd.Series([750.0])).iloc[0])

        assert future_df["cooling_degree_days"].iloc[0] == pytest.approx(expected_cdd, abs=1e-6)
        assert future_df["heating_degree_days"].iloc[0] == pytest.approx(expected_hdd, abs=1e-6)
        assert future_df["wind_power_estimate"].iloc[0] == pytest.approx(expected_wind, abs=1e-6)
        assert future_df["solar_capacity_factor"].iloc[0] == pytest.approx(expected_solar, abs=1e-6)

    def test_overlay_partial_coverage_falls_back_to_climatology_beyond(self):
        """If weather_df covers only the first 168 of 720 hours, the
        remaining 552 hours must use climatological values (not zero,
        not NaN, not the last forecast value)."""
        from jobs.phases import _build_future_feature_frame

        featured = _build_featured_hist()
        # Forecast covers only 168 of 720 horizon hours
        weather_df = _build_weather_forecast(n_hours=168, temperature=95.0)
        future_df = _build_future_feature_frame(featured, horizon=720, weather_df=weather_df)

        # First 168 hours: actual forecast
        assert np.allclose(future_df["temperature_2m"].iloc[:168].values, 95.0)

        # Beyond hour 168: climatology, which should be near the
        # historical 70°F baseline of our synthetic series. Should NOT
        # be 95°F (the forecast value) or 0 (a NaN-fill mistake).
        beyond_temp = future_df["temperature_2m"].iloc[168:]
        assert beyond_temp.between(50.0, 90.0).all()
        # No more than a trivial number of climatology rows happen to
        # equal 95 by coincidence — strict zero on a synthetic series.
        assert int(np.isclose(beyond_temp.values, 95.0, atol=1e-3).sum()) == 0

    def test_overlay_with_no_overlap_keeps_climatology(self):
        """When weather_df timestamps don't overlap the future horizon
        at all (e.g., stale weather cache), behavior reverts to climatology."""
        from jobs.phases import _build_future_feature_frame

        featured = _build_featured_hist()
        # Weather forecast starts AFTER our 24-hour horizon ends
        wx_start = pd.Timestamp("2026-06-20 00:00", tz="UTC")
        weather_df = pd.DataFrame(
            {
                "timestamp": pd.date_range(start=wx_start, periods=168, freq="h"),
                "temperature_2m": np.full(168, 95.0),
            }
        )
        future_df = _build_future_feature_frame(featured, horizon=24, weather_df=weather_df)

        # No row in the 24-hour horizon overlaps weather_df → climatology
        # 95°F should not appear anywhere.
        assert int(np.isclose(future_df["temperature_2m"].values, 95.0, atol=1e-3).sum()) == 0
        # And climatology should produce reasonable temperatures
        assert future_df["temperature_2m"].between(50.0, 90.0).all()

    def test_overlay_preserves_time_features(self):
        """The overlay must not corrupt time features (hour_sin, dow_sin,
        is_weekend) that are computed from future timestamps."""
        from jobs.phases import _build_future_feature_frame

        featured = _build_featured_hist()
        weather_df = _build_weather_forecast(n_hours=168, temperature=95.0)
        future_df = _build_future_feature_frame(featured, horizon=168, weather_df=weather_df)

        # hour_sin should range over [-1, 1] across a 24-hour window
        first_24 = future_df["hour_sin"].iloc[:24]
        assert first_24.min() < 0 and first_24.max() > 0
        # is_weekend should be 0 or 1
        assert future_df["is_weekend"].isin([0, 1]).all()

    def test_overlay_temperature_deviation_uses_historical_context(self):
        """temperature_deviation = current_temp - 720h rolling mean. The
        rolling window must include historical context, otherwise
        deviation collapses to ~0 for future rows when the forecast is
        constant."""
        from jobs.phases import _build_future_feature_frame

        # Historical baseline ~70°F, forecast is constant 95°F.
        # If rolling context is included: deviation ≈ 95 - 70 = 25°F.
        # If rolling computed over future rows alone (forecast constant):
        # deviation ≈ 95 - 95 = 0°F.
        featured = _build_featured_hist(n_hours=720 * 2)  # 2 months of history
        weather_df = _build_weather_forecast(n_hours=168, temperature=95.0)
        future_df = _build_future_feature_frame(featured, horizon=168, weather_df=weather_df)

        # Deviation should be substantially > 0 — the forecast is much
        # hotter than the historical rolling 30-day mean.
        deviation_at_hour_24 = float(future_df["temperature_deviation"].iloc[24])
        assert deviation_at_hour_24 > 5.0, (
            f"temperature_deviation collapsed to {deviation_at_hour_24} — "
            "rolling window probably not including historical context"
        )

    def test_overlay_missing_columns_silently_skipped(self):
        """If weather_df is missing some raw columns (e.g., older Open-Meteo
        format), the overlay should only touch the columns it has and
        leave the rest at their climatology values."""
        from jobs.phases import _build_future_feature_frame

        featured = _build_featured_hist()
        # Only provide temperature; other raw columns absent from forecast
        weather_df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2026-05-20 15:00", periods=168, freq="h", tz="UTC"
                ),
                "temperature_2m": np.full(168, 95.0),
            }
        )
        future_df = _build_future_feature_frame(featured, horizon=168, weather_df=weather_df)

        # Temperature got forecast values
        assert np.allclose(future_df["temperature_2m"].values, 95.0)
        # wind_speed_80m stayed at climatology (not 0, not NaN)
        assert future_df["wind_speed_80m"].notna().all()
        assert (future_df["wind_speed_80m"] == 0).sum() < len(future_df)

    def test_overlay_with_empty_weather_df_no_op(self):
        """An empty weather_df should produce identical output to the
        no-weather_df case (climatology baseline)."""
        from jobs.phases import _build_future_feature_frame

        featured = _build_featured_hist()
        empty_wx = pd.DataFrame(columns=["timestamp", "temperature_2m"])

        future_with_empty = _build_future_feature_frame(featured, horizon=24, weather_df=empty_wx)
        future_no_wx = _build_future_feature_frame(featured, horizon=24)

        pd.testing.assert_frame_equal(future_with_empty, future_no_wx)


# ────────────────────────────────────────────────────────────────────────
# PR-E (2026-05-20) — XGBoost recursive autoregressive prediction
# ────────────────────────────────────────────────────────────────────────


class _FakeXgbModel:
    """Minimal XGBoost-model stub for the recursive predict path.

    Returns a prediction that's a deterministic function of
    ``demand_lag_1h`` (the chained-prediction history's most recent
    value). Lets the recursion be observed directly: pred[i] = f(pred[i-1]).
    """

    def __init__(self, feature_names: list[str], multiplier: float = 1.02):
        self._feature_names = feature_names
        self._mult = multiplier

    def __getitem__(self, key):  # match dict-style access used by predict_xgboost
        if key == "feature_names":
            return self._feature_names
        if key == "model":
            return self
        raise KeyError(key)


def _fake_predict_xgboost(model_dict, df):
    """Stand-in for ``predict_xgboost`` — returns ``demand_lag_1h * 1.02``.

    Used by the recursive test path so we can verify the chaining works
    (each step's input lag_1h equals the previous step's prediction).
    """
    lag_1h = df["demand_lag_1h"].fillna(20_000.0).astype(float).values
    return lag_1h * 1.02


class TestPredictXgboostRecursive:
    """``_predict_xgboost_with_recursive_autoregressive`` runs a
    chained per-hour predict loop for the recursive zone, then a
    vectorized predict for the climatology tail. PR-E (#138).

    These tests use a fake predict_xgboost that returns
    ``demand_lag_1h * 1.02`` so we can observe chaining directly:
    pred[i] = pred[i-1] * 1.02 once the chain starts.
    """

    @staticmethod
    def _featured(n_hours: int = 200, last_demand: float = 20_000.0) -> pd.DataFrame:
        ts = pd.date_range("2026-05-01", periods=n_hours, freq="h", tz="UTC")
        return pd.DataFrame(
            {
                "timestamp": ts,
                "demand_mw": np.full(n_hours, last_demand),
            }
        )

    @staticmethod
    def _future(n_hours: int) -> pd.DataFrame:
        ts = pd.date_range("2026-05-20", periods=n_hours, freq="h", tz="UTC")
        # Climatology-shaped autoregressive baseline that the helper
        # will override row-by-row in the recursive zone.
        return pd.DataFrame(
            {
                "timestamp": ts,
                "hour": ts.hour,
                "demand_lag_1h": np.full(n_hours, 30_000.0),  # baseline, will be overwritten
                "demand_lag_3h": np.full(n_hours, 30_000.0),
                "demand_lag_24h": np.full(n_hours, 30_000.0),
                "demand_lag_168h": np.full(n_hours, 30_000.0),
                "ramp_rate": np.zeros(n_hours),
                "demand_roll_24h_mean": np.full(n_hours, 30_000.0),
                "demand_roll_24h_std": np.full(n_hours, 100.0),
                "demand_roll_24h_min": np.full(n_hours, 29_500.0),
                "demand_roll_24h_max": np.full(n_hours, 30_500.0),
                "demand_roll_72h_mean": np.full(n_hours, 30_000.0),
                "demand_roll_72h_std": np.full(n_hours, 100.0),
                "demand_roll_72h_min": np.full(n_hours, 29_500.0),
                "demand_roll_72h_max": np.full(n_hours, 30_500.0),
                "demand_roll_168h_mean": np.full(n_hours, 30_000.0),
                "demand_roll_168h_std": np.full(n_hours, 100.0),
                "demand_roll_168h_min": np.full(n_hours, 29_500.0),
                "demand_roll_168h_max": np.full(n_hours, 30_500.0),
                "demand_momentum_short": np.zeros(n_hours),
                "demand_momentum_long": np.zeros(n_hours),
                "demand_ratio_24h": np.ones(n_hours),
                "demand_ratio_168h": np.ones(n_hours),
            }
        )

    def test_recursive_zone_chains_from_recent_actuals(self, monkeypatch):
        """First prediction uses the most recent actual (20,000 MW)
        from ``featured`` — NOT the climatology baseline in future_df.
        Each subsequent prediction uses the prior prediction. Verifies
        the chain is seeded correctly."""
        import jobs.phases as phases

        monkeypatch.setattr("models.xgboost_model.predict_xgboost", _fake_predict_xgboost)

        featured = self._featured(last_demand=20_000.0)
        future_df = self._future(n_hours=5)
        model = _FakeXgbModel(feature_names=list(future_df.columns))

        preds = phases._predict_xgboost_with_recursive_autoregressive(
            model, featured, future_df, horizon=5, recursive_hours=5
        )

        # Chain: 20000 → 20400 → 20808 → 21224 → 21649 (×1.02 each step)
        # The first prediction reads lag_1h from history (20000 actuals),
        # not 30000 (the climatology baseline).
        expected = [20_000.0 * 1.02 ** (i + 1) for i in range(5)]
        np.testing.assert_allclose(preds, expected, rtol=1e-6)

    def test_recursive_then_climatology_horizon(self, monkeypatch):
        """When horizon exceeds recursive_hours, first N predictions
        chain from history, remaining predictions use the climatology-
        shaped features in future_df (lag_1h=30000)."""
        import jobs.phases as phases

        monkeypatch.setattr("models.xgboost_model.predict_xgboost", _fake_predict_xgboost)

        featured = self._featured(last_demand=20_000.0)
        future_df = self._future(n_hours=10)
        model = _FakeXgbModel(feature_names=list(future_df.columns))

        preds = phases._predict_xgboost_with_recursive_autoregressive(
            model, featured, future_df, horizon=10, recursive_hours=3
        )

        # First 3: recursive chain from 20000 ×1.02 each step
        recursive_expected = [20_000.0 * 1.02 ** (i + 1) for i in range(3)]
        np.testing.assert_allclose(preds[:3], recursive_expected, rtol=1e-6)

        # Remaining 7: climatology predictions = 30000 × 1.02 = 30600 (all same)
        clim_expected = [30_000.0 * 1.02] * 7
        np.testing.assert_allclose(preds[3:], clim_expected, rtol=1e-6)

    def test_recursive_hours_caps_at_horizon(self, monkeypatch):
        """If recursive_hours > horizon, we just chain for ``horizon``
        hours and skip the climatology tail."""
        import jobs.phases as phases

        monkeypatch.setattr("models.xgboost_model.predict_xgboost", _fake_predict_xgboost)

        featured = self._featured()
        future_df = self._future(n_hours=4)
        model = _FakeXgbModel(feature_names=list(future_df.columns))

        preds = phases._predict_xgboost_with_recursive_autoregressive(
            model, featured, future_df, horizon=4, recursive_hours=384
        )

        assert len(preds) == 4
        # All four are recursive
        expected = [20_000.0 * 1.02 ** (i + 1) for i in range(4)]
        np.testing.assert_allclose(preds, expected, rtol=1e-6)

    def test_default_recursive_hours_matches_open_meteo_horizon(self):
        """The default recursive depth (``RECURSIVE_AUTOREGRESSIVE_HOURS``)
        must equal ``OPEN_METEO_FORECAST_HOURS`` so the two regimes —
        "real signal" and "climatology baseline" — break at the same
        day-16 boundary as ADR-008."""
        from config import OPEN_METEO_FORECAST_HOURS
        from jobs.phases import RECURSIVE_AUTOREGRESSIVE_HOURS

        assert RECURSIVE_AUTOREGRESSIVE_HOURS == OPEN_METEO_FORECAST_HOURS
        assert RECURSIVE_AUTOREGRESSIVE_HOURS == 384

    def test_predict_one_xgboost_uses_recursive_path(self, monkeypatch):
        """``_predict_one`` for XGBoost must dispatch through
        ``_predict_xgboost_with_recursive_autoregressive`` so the
        production scoring job picks up PR-E's behavior."""
        import jobs.phases as phases

        called: dict[str, bool] = {"recursive": False}

        def _spy(model, featured, future_df, horizon, **kw):
            called["recursive"] = True
            return np.zeros(horizon, dtype=float)

        monkeypatch.setattr(phases, "_predict_xgboost_with_recursive_autoregressive", _spy)

        featured = self._featured()
        future_df = self._future(n_hours=24)
        model = _FakeXgbModel(feature_names=list(future_df.columns))

        result = phases._predict_one("xgboost", model, featured, future_df, horizon=24)
        assert result is not None
        assert called["recursive"] is True


# ────────────────────────────────────────────────────────────────────────
# #129 — Forecast tab gap fix (anchor on last_real_demand_hour + 1h)
# ────────────────────────────────────────────────────────────────────────


class TestResolveForecastStart:
    """``_resolve_forecast_start`` picks the timestamp for hour 0 of the
    forecast. The normal case (no publishing-lag gap) returns
    ``featured.timestamp.max() + 1h``; the gap case returns
    ``last_real_demand_hour + 1h``. See #129.
    """

    @staticmethod
    def _featured(end_ts: str, n_hours: int = 200) -> pd.DataFrame:
        end = pd.Timestamp(end_ts, tz="UTC")
        ts = pd.date_range(end=end, periods=n_hours, freq="h")
        return pd.DataFrame({"timestamp": ts, "demand_mw": np.full(n_hours, 20_000.0)})

    @staticmethod
    def _demand_df(end_ts: str, n_hours: int = 200) -> pd.DataFrame:
        end = pd.Timestamp(end_ts, tz="UTC")
        ts = pd.date_range(end=end, periods=n_hours, freq="h")
        return pd.DataFrame({"timestamp": ts, "demand_mw": np.full(n_hours, 20_000.0)})

    def test_no_gap_returns_featured_max_plus_1h(self):
        """Normal case: demand_df and featured end at the same timestamp
        (EIA fully caught up). Forecast starts at that timestamp + 1h."""
        from jobs.phases import _resolve_forecast_start

        same_end = "2026-05-20 14:00"
        featured = self._featured(same_end)
        demand_df = self._demand_df(same_end)

        forecast_start = _resolve_forecast_start(featured, demand_df)
        assert forecast_start == pd.Timestamp(same_end, tz="UTC") + pd.Timedelta(hours=1)

    def test_publishing_lag_gap_anchors_on_real_demand(self):
        """Gap case: featured extends to 14:00 UTC but demand_df has
        real readings only through 10:00 UTC (4-hour EIA publishing
        lag — the production scenario from #129). Forecast must start
        at 11:00 UTC, not 15:00 UTC."""
        from jobs.phases import _resolve_forecast_start

        featured = self._featured("2026-05-20 14:00")
        demand_df = self._demand_df("2026-05-20 10:00")

        forecast_start = _resolve_forecast_start(featured, demand_df)
        assert forecast_start == pd.Timestamp("2026-05-20 11:00", tz="UTC")

    def test_trailing_nan_demand_treated_as_missing(self):
        """If demand_df includes trailing rows with NaN demand (EIA's
        sentinel for unpublished hours), those don't count as 'real
        demand' — the anchor is the last non-NaN hour."""
        from jobs.phases import _resolve_forecast_start

        featured = self._featured("2026-05-20 14:00")
        demand_df = self._demand_df("2026-05-20 14:00")
        # Last 4 hours have NaN demand (unpublished)
        demand_df.loc[demand_df.index[-4:], "demand_mw"] = np.nan

        forecast_start = _resolve_forecast_start(featured, demand_df)
        # Last real demand hour = 10:00 UTC; forecast starts at 11:00 UTC
        assert forecast_start == pd.Timestamp("2026-05-20 11:00", tz="UTC")

    def test_trailing_zero_demand_treated_as_missing(self):
        """Defense in depth: even though ``eia_client`` coerces 0 → NaN,
        any zero rows that slip through (e.g., via cache from a
        pre-fix version) are still treated as 'missing' since a BA
        cannot have zero demand."""
        from jobs.phases import _resolve_forecast_start

        featured = self._featured("2026-05-20 14:00")
        demand_df = self._demand_df("2026-05-20 14:00")
        demand_df.loc[demand_df.index[-3:], "demand_mw"] = 0.0

        forecast_start = _resolve_forecast_start(featured, demand_df)
        # Last real demand = 11:00 UTC (last index minus 3 → 14:00 - 3h)
        assert forecast_start == pd.Timestamp("2026-05-20 12:00", tz="UTC")

    def test_empty_demand_df_falls_back_to_featured(self):
        """Defensive — empty demand_df → fall back to old behavior."""
        from jobs.phases import _resolve_forecast_start

        featured = self._featured("2026-05-20 14:00")
        empty = pd.DataFrame(columns=["timestamp", "demand_mw"])

        forecast_start = _resolve_forecast_start(featured, empty)
        assert forecast_start == pd.Timestamp("2026-05-20 15:00", tz="UTC")

    def test_all_nan_demand_falls_back_to_featured(self):
        """If every demand row is NaN (degenerate fetch failure), fall
        back to ``featured.max + 1h`` rather than failing the phase."""
        from jobs.phases import _resolve_forecast_start

        featured = self._featured("2026-05-20 14:00")
        demand_df = self._demand_df("2026-05-20 14:00")
        demand_df["demand_mw"] = np.nan

        forecast_start = _resolve_forecast_start(featured, demand_df)
        assert forecast_start == pd.Timestamp("2026-05-20 15:00", tz="UTC")

    def test_last_real_demand_after_featured_caps_at_featured(self):
        """If last_real_demand somehow exceeds featured.max (e.g., feature
        engineering dropped trailing rows for reasons unrelated to
        demand-NaN), cap the anchor at featured.max so we don't
        generate forecast rows without lag context."""
        from jobs.phases import _resolve_forecast_start

        featured = self._featured("2026-05-20 12:00")
        demand_df = self._demand_df("2026-05-20 14:00")

        forecast_start = _resolve_forecast_start(featured, demand_df)
        assert forecast_start == pd.Timestamp("2026-05-20 13:00", tz="UTC")


class TestBuildFutureFeatureFrameStartTs:
    """``_build_future_feature_frame`` accepts an explicit ``start_ts``
    kwarg (#129). Default behavior unchanged when ``start_ts=None``."""

    def test_explicit_start_ts_anchors_first_row(self):
        """When ``start_ts`` is provided, the first future row's
        timestamp equals that anchor (NOT ``featured.max + 1h``)."""
        from jobs.phases import _build_future_feature_frame

        featured = _build_featured_hist(n_hours=168 * 2, last_ts="2026-05-20 14:00")
        # Anchor forecast at 11:00 UTC (4 hours BEFORE featured.max)
        anchor = pd.Timestamp("2026-05-20 11:00", tz="UTC")
        future_df = _build_future_feature_frame(featured, horizon=24, start_ts=anchor)

        assert pd.Timestamp(future_df["timestamp"].iloc[0]) == anchor

    def test_no_start_ts_preserves_old_behavior(self):
        """Without ``start_ts``, the function still anchors at
        ``featured.max + 1h`` — pre-#129 behavior."""
        from jobs.phases import _build_future_feature_frame

        featured = _build_featured_hist(n_hours=168 * 2, last_ts="2026-05-20 14:00")
        future_df = _build_future_feature_frame(featured, horizon=24)

        expected_first = pd.Timestamp("2026-05-20 15:00", tz="UTC")
        assert pd.Timestamp(future_df["timestamp"].iloc[0]) == expected_first


class TestRecursivePredictDemandHistorySeed:
    """``_predict_xgboost_with_recursive_autoregressive`` filters its
    demand_history seed against NaN/zero values (#129). A single zero
    or NaN trailing row would otherwise poison the next 168 rolling
    features computed by ``compute_autoregressive_snapshot``.
    """

    def test_seed_filters_nan_and_zero_demand(self, monkeypatch):
        """Build a ``featured`` whose last 4 rows have NaN/zero demand.
        The recursive predict's first prediction should be seeded from
        the LAST GOOD demand (20,000.0), not from NaN/zero."""
        import jobs.phases as phases

        # Fake predict_xgboost returns lag_1h × 1.02 — same as the
        # PR-E test infrastructure. Lets us observe what got seeded.
        def _fake_predict(model, df):
            lag = df["demand_lag_1h"].fillna(-1.0).astype(float).values
            return lag * 1.02

        monkeypatch.setattr("models.xgboost_model.predict_xgboost", _fake_predict)

        ts = pd.date_range("2026-05-01", periods=200, freq="h", tz="UTC")
        # First 196 rows: real demand 20,000. Last 4 rows: NaN/zero noise.
        demand = np.full(200, 20_000.0)
        demand[-4:-2] = np.nan
        demand[-2:] = 0.0
        featured = pd.DataFrame({"timestamp": ts, "demand_mw": demand})

        future_ts = pd.date_range("2026-05-20", periods=5, freq="h", tz="UTC")
        future_df = pd.DataFrame(
            {
                "timestamp": future_ts,
                "demand_lag_1h": np.full(5, 30_000.0),  # baseline overridden by recursion
            }
        )
        # Add the autoregressive feature columns the snapshot fills
        for col in [
            "demand_lag_3h",
            "demand_lag_24h",
            "demand_lag_168h",
            "ramp_rate",
            "demand_momentum_short",
            "demand_momentum_long",
            "demand_ratio_24h",
            "demand_ratio_168h",
            "demand_roll_24h_mean",
            "demand_roll_24h_std",
            "demand_roll_24h_min",
            "demand_roll_24h_max",
            "demand_roll_72h_mean",
            "demand_roll_72h_std",
            "demand_roll_72h_min",
            "demand_roll_72h_max",
            "demand_roll_168h_mean",
            "demand_roll_168h_std",
            "demand_roll_168h_min",
            "demand_roll_168h_max",
        ]:
            future_df[col] = 0.0

        model = _FakeXgbModel(feature_names=list(future_df.columns))
        preds = phases._predict_xgboost_with_recursive_autoregressive(
            model, featured, future_df, horizon=5, recursive_hours=5
        )

        # First prediction = lag_1h × 1.02 = LAST REAL demand × 1.02
        # = 20,000 × 1.02 = 20,400. NOT 0 × 1.02 = 0 (which would
        # happen if NaN/zero values made it into demand_history).
        assert preds[0] == pytest.approx(20_400.0, rel=1e-6)
        # The chain continues at 1.02× per step
        assert preds[1] == pytest.approx(20_808.0, rel=1e-6)
