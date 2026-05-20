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
        # PR-C (2026-05-20): function gained ``weather_df`` kwarg. Lambda
        # accepts and ignores it — tests here exercise ``predict_and_write_forecast``
        # ensemble logic, not the future-frame builder itself (which has
        # its own dedicated test class below).
        lambda featured, horizon, weather_df=None: pd.DataFrame(
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
        payload = fake_redis["gridpulse:forecast:ERCOT:1h"]
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
