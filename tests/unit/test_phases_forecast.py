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

    def test_degenerate_series_gets_horizon_guard_entry(self, fake_redis, region_data, monkeypatch):
        """#296: a model whose 30-day trajectory decays out of the recent
        demand band gets a ``horizon_guard`` payload entry with the largest
        still-sane horizon; sane models (and a sane ensemble) get none."""
        from jobs import phases

        # ARIMA decays linearly 40k → −10k over 720h (the SC/PSCO shape);
        # the serve floor clips the tail at 0, far below the 20k band floor.
        # The 24h/168h prefixes stay in-band → max_ok_horizon = 168.
        arima_preds = 40_000.0 - np.linspace(0.0, 50_000.0, HORIZON)
        _patch_predict_one(
            monkeypatch,
            {"xgboost": np.full(HORIZON, 41_000.0), "arima": arima_preds},
        )

        result = phases.predict_and_write_forecast(
            region_data,
            models={"xgboost": object(), "arima": object()},
            # xgboost dominates the blend (inverse-MAPE³), keeping the
            # ensemble sane despite the degenerate arima input.
            model_mapes={"xgboost": 1.0, "arima": 8.0},
        )

        assert result.ok
        payload = fake_redis["gridpulse:forecast:ERCOT:1h"]
        guard = payload["horizon_guard"]
        assert set(guard.keys()) == {"arima"}
        assert guard["arima"]["max_ok_horizon"] == 168
        assert guard["arima"]["flagged_horizon"] == 720
        assert guard["arima"]["reason"] == "below_recent_band"
        # The flagged series stays in the rows for transparency.
        assert "arima" in payload["forecasts"][0]

    def test_all_sane_series_write_no_horizon_guard(self, fake_redis, region_data, monkeypatch):
        from jobs import phases

        _patch_predict_one(
            monkeypatch,
            {"xgboost": np.full(HORIZON, 41_000.0), "prophet": np.full(HORIZON, 39_000.0)},
        )
        result = phases.predict_and_write_forecast(
            region_data,
            models={"xgboost": object(), "prophet": object()},
            model_mapes={"xgboost": 2.0, "prophet": 3.0},
        )
        assert result.ok
        assert "horizon_guard" not in fake_redis["gridpulse:forecast:ERCOT:1h"]

    def test_degenerate_ensemble_is_guarded_too(self, fake_redis, region_data, monkeypatch):
        """When every input decays, the blend decays with them — the served
        ``ensemble`` series must carry its own guard entry."""
        from jobs import phases

        decay = 40_000.0 - np.linspace(0.0, 50_000.0, HORIZON)
        _patch_predict_one(monkeypatch, {"xgboost": decay.copy(), "arima": decay.copy()})

        result = phases.predict_and_write_forecast(
            region_data,
            models={"xgboost": object(), "arima": object()},
            model_mapes={"xgboost": 2.0, "arima": 2.0},
        )
        assert result.ok
        guard = fake_redis["gridpulse:forecast:ERCOT:1h"]["horizon_guard"]
        assert set(guard.keys()) == {"xgboost", "arima", "ensemble"}

    def test_negative_predictions_floored_at_zero(self, fake_redis, region_data, monkeypatch):
        """#281: any model emitting negative demand is clipped at 0 at the serve
        layer — covers all models + the ensemble, not just Prophet's own clip."""
        from jobs import phases

        prophet_preds = np.full(HORIZON, 39_000.0)
        prophet_preds[:10] = -2_000.0  # the #281 negative excursion
        _patch_predict_one(
            monkeypatch,
            {"xgboost": np.full(HORIZON, 41_000.0), "prophet": prophet_preds},
        )

        result = phases.predict_and_write_forecast(
            region_data,
            models={"xgboost": object(), "prophet": object()},
            model_mapes={"xgboost": 5.0, "prophet": 5.0},
        )

        assert result.ok
        payload = fake_redis["gridpulse:forecast:ERCOT:1h"]
        for row in payload["forecasts"]:
            assert row["prophet"] >= 0.0
            assert row["ensemble"] >= 0.0
            assert row["predicted_demand_mw"] >= 0.0
        assert payload["forecasts"][0]["prophet"] == 0.0  # −2000 → 0

    def test_ensemble_floor_survives_negative_ensemble_combine(
        self, fake_redis, region_data, monkeypatch
    ):
        """#281: the serve-layer ``np.maximum(ensemble_combine(...), 0)`` must
        floor the *ensemble* even when ``ensemble_combine`` itself emits a
        negative — the comment on that clip states it must survive a future
        ``ensemble_combine`` that returns a negative composite.

        ``test_negative_predictions_floored_at_zero`` above cannot exercise this:
        every per-model pred is floored at line ~984 *before* the blend, so its
        negative never reaches ``ensemble_combine``'s output. Here all per-model
        preds are positive and ``ensemble_combine`` is patched to return
        negatives, leaving only the ensemble ``np.maximum`` between the negative
        and Redis. Delete that clip and this test fails.

        NB: ``predict_and_write_forecast`` does a *function-local*
        ``from models.ensemble import ensemble_combine`` (jobs/phases.py:933), so
        the effective patch target is the source module ``models.ensemble`` —
        patching ``jobs.phases`` would be shadowed by that local import.
        """
        import models.ensemble as ens
        from jobs import phases

        _patch_predict_one(
            monkeypatch,
            {
                "xgboost": np.full(HORIZON, 41_000.0),
                "prophet": np.full(HORIZON, 39_000.0),
            },
        )

        # Blend returns negatives for the first 10 hours regardless of its
        # (already-floored) inputs — stands in for a future ensemble_combine
        # that can emit a negative composite.
        negative_blend = np.full(HORIZON, 40_000.0)
        negative_blend[:10] = -3_000.0

        def _fake_ensemble_combine(preds_by_model, weights):
            return negative_blend

        monkeypatch.setattr(ens, "ensemble_combine", _fake_ensemble_combine)

        result = phases.predict_and_write_forecast(
            region_data,
            models={"xgboost": object(), "prophet": object()},
            model_mapes={"xgboost": 5.0, "prophet": 5.0},
        )

        assert result.ok
        payload = fake_redis["gridpulse:forecast:ERCOT:1h"]
        for row in payload["forecasts"]:
            assert row["ensemble"] >= 0.0
            assert row["predicted_demand_mw"] >= 0.0
        assert payload["forecasts"][0]["ensemble"] == 0.0  # −3000 blend → 0


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


class TestClimatologyRecentWindow:
    """#281: the (hour, dow) climatology baseline is built from a RECENT trailing
    window, so it tracks the forecast season instead of regressing toward the
    cooler annual mean of the full training history."""

    def _split_featured(self, old_temp: float, recent_temp: float, total_days: int = 60):
        from jobs.phases import CLIMATOLOGY_WINDOW_DAYS

        end = pd.Timestamp("2026-07-09 23:00", tz="UTC")
        n = total_days * 24
        ts = pd.date_range(end=end, periods=n, freq="h")
        cutoff = end - pd.Timedelta(days=CLIMATOLOGY_WINDOW_DAYS)
        temp = np.where(ts >= cutoff, recent_temp, old_temp).astype(float)
        return pd.DataFrame(
            {
                "timestamp": ts,
                "demand_mw": 15000.0,
                "temperature_2m": temp,
                "cooling_degree_days": np.maximum(0.0, temp - 65),
            }
        )

    def test_climatology_reflects_recent_window_not_full_history(self):
        """Recent 28d at 85°F, older 32d at 50°F. Full-history mean would be
        ~66°F; the recent-window climatology must land near 85°F."""
        from jobs.phases import _build_future_feature_frame

        featured = self._split_featured(old_temp=50.0, recent_temp=85.0)
        future_df = _build_future_feature_frame(featured, horizon=48)  # no weather_df
        assert future_df["temperature_2m"].mean() == pytest.approx(85.0, abs=1.0)
        # CDD tracks it (85-65=20), not the diluted full-history value.
        assert future_df["cooling_degree_days"].mean() == pytest.approx(20.0, abs=1.5)

    def test_short_history_falls_back_without_crashing(self):
        """A history shorter than the recent-window min-rows guard falls back to
        the full history and still produces a valid frame."""
        from jobs.phases import _build_future_feature_frame

        featured = self._split_featured(old_temp=70.0, recent_temp=70.0, total_days=3)
        future_df = _build_future_feature_frame(featured, horizon=24)
        assert len(future_df) == 24
        assert future_df["temperature_2m"].between(60.0, 80.0).all()


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


class TestForecastOriginNeverRegresses:
    """#537 — a forecast origin must never go backwards.

    ``_resolve_forecast_start`` is recomputed from scratch each tick with no
    memory of the last one, so when EIA retracts hours it had already published
    the anchor collapses to just before the retracted block. Measured in
    production on 2026-08-14: LGEE's origin went from 2026-08-13T14:00 to
    2026-08-12T15:00 — 23 hours OLDER than a vintage already served — and stayed
    there for 24 ticks, relabelling 40-to-63-hour-ahead rows as one-hour-ahead.
    Every one of the six frozen-origin BAs' regressed ticks carried an origin the
    BA had computed at an earlier tick; the never-frozen BAs had none in 484.
    """

    @staticmethod
    def _origin(ts: str) -> pd.Timestamp:
        return pd.Timestamp(ts, tz="UTC")

    def test_older_origin_does_not_overwrite_the_served_payload(
        self, region_data, fake_redis, monkeypatch
    ):
        """The reported defect: an older-origin payload must not be published."""
        from jobs.phases import predict_and_write_forecast

        _patch_predict_one(monkeypatch, {"xgboost": np.full(HORIZON, 41_000.0)})
        # featured/demand end at 2024-01-09 07:00, so the resolved start is
        # 08:00 — far older than what Redis is already serving.
        region_data.previous_forecast_origin = self._origin("2024-02-01 00:00")

        result = predict_and_write_forecast(region_data, {"xgboost": object()})

        assert result.ok is True, "a live, newer payload is not a failed region"
        assert result.details["skipped"] == "origin_regressed"
        assert result.details["served_origin"] == "2024-02-01T00:00:00+00:00"
        assert fake_redis == {}, "the newer payload must be left in place"

    def test_equal_origin_still_writes(self, region_data, fake_redis, monkeypatch):
        """A stalled origin is a different defect and must not be suppressed.

        The guard is strictly ``<``. An origin that repeats — the freeze half of
        #537, caused by a NaN hole deleting rows from the tail of the feature
        frame — still republishes, because the models and the weather behind it
        have moved even when the anchor has not.
        """
        from jobs.phases import predict_and_write_forecast

        _patch_predict_one(monkeypatch, {"xgboost": np.full(HORIZON, 41_000.0)})
        region_data.previous_forecast_origin = self._origin("2024-01-09 08:00")

        result = predict_and_write_forecast(region_data, {"xgboost": object()})

        assert result.ok is True
        assert "skipped" not in result.details
        assert any(k.endswith("forecast:ERCOT:1h") for k in fake_redis)

    def test_first_ever_tick_has_no_prior_origin_and_writes(
        self, region_data, fake_redis, monkeypatch
    ):
        """No payload in Redis yet — the guard must not block the first write."""
        from jobs.phases import predict_and_write_forecast

        _patch_predict_one(monkeypatch, {"xgboost": np.full(HORIZON, 41_000.0)})
        assert region_data.previous_forecast_origin is None

        result = predict_and_write_forecast(region_data, {"xgboost": object()})

        assert result.ok is True
        assert any(k.endswith("forecast:ERCOT:1h") for k in fake_redis)


class TestForecastPayloadOrigin:
    """``forecast_payload_origin`` reads the same row the drift module measures
    lead against, so an origin recovered from the drift log and this value are
    the same quantity (#537)."""

    def test_reads_first_row_timestamp(self):
        from jobs.phases import forecast_payload_origin

        payload = {
            "forecasts": [
                {"timestamp": "2026-08-14T06:00:00+00:00"},
                {"timestamp": "2026-08-14T07:00:00+00:00"},
            ]
        }
        assert forecast_payload_origin(payload) == pd.Timestamp("2026-08-14 06:00", tz="UTC")

    def test_agrees_with_the_drift_module_lead_definition(self):
        """``lead = target - origin + 1`` must invert to this origin exactly.

        The whole #537 reconstruction rests on that identity: every historical
        origin was recovered from ``drift_updated`` as ``new_record_ts -
        lead_hours + 1``. If the two ends drift apart the reconstruction is
        measuring itself, so pin them against each other rather than by eye.
        """
        from jobs.phases import forecast_payload_origin
        from models.drift import _lead_hours

        rows = [{"timestamp": f"2026-08-14T{h:02d}:00:00+00:00"} for h in range(6, 12)]
        target = "2026-08-14T09:00:00+00:00"

        lead = _lead_hours(rows, target)
        origin = forecast_payload_origin({"forecasts": rows})

        assert pd.Timestamp(target) - pd.Timedelta(hours=lead - 1) == origin

    @pytest.mark.parametrize("payload", [None, {}, {"forecasts": []}, "not-a-dict"])
    def test_absent_or_malformed_yields_none(self, payload):
        """None disables the guard — an unreadable payload must never be treated
        as an origin of 0 and block every future write."""
        from jobs.phases import forecast_payload_origin

        assert forecast_payload_origin(payload) is None


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


class TestGapActualDemand:
    """#226: ``_gap_actual_demand`` extracts the leading contiguous real demand
    across the train->score gap so SARIMAX can advance its Kalman state."""

    def _featured(self, n=30, tz="UTC"):
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz=tz)
        return pd.DataFrame({"timestamp": ts, "demand_mw": np.arange(n, dtype=float) * 100 + 1000})

    def test_extracts_leading_contiguous_gap(self):
        from jobs.phases import _gap_actual_demand

        fe = self._featured(30)
        anchor = fe["timestamp"].iloc[10]  # train_end
        start = fe["timestamp"].iloc[20]  # start_ts
        ga = _gap_actual_demand(fe, anchor, start)
        # hours strictly between index 10 and 20 -> rows 11..19 = 9
        assert ga is not None and len(ga) == 9
        assert ga[0] == fe["demand_mw"].iloc[11]
        assert ga[-1] == fe["demand_mw"].iloc[19]

    def test_none_when_no_anchor(self):
        from jobs.phases import _gap_actual_demand

        fe = self._featured()
        assert _gap_actual_demand(fe, None, fe["timestamp"].iloc[20]) is None

    def test_trailing_nan_truncates_run(self):
        # EIA publish lag: the last gap hours have no demand yet.
        from jobs.phases import _gap_actual_demand

        fe = self._featured(30)
        fe.loc[15:19, "demand_mw"] = np.nan
        anchor = fe["timestamp"].iloc[10]
        start = fe["timestamp"].iloc[20]
        ga = _gap_actual_demand(fe, anchor, start)
        # gap rows 11..19; leading non-NaN = 11,12,13,14 -> 4
        assert ga is not None and len(ga) == 4

    def test_empty_gap_returns_none(self):
        from jobs.phases import _gap_actual_demand

        fe = self._featured(30)
        anchor = fe["timestamp"].iloc[10]
        start = fe["timestamp"].iloc[11]  # adjacent -> nothing strictly between
        assert _gap_actual_demand(fe, anchor, start) is None

    def test_leading_nan_yields_none(self):
        from jobs.phases import _gap_actual_demand

        fe = self._featured(30)
        fe.loc[11, "demand_mw"] = np.nan  # first gap hour missing
        anchor = fe["timestamp"].iloc[10]
        start = fe["timestamp"].iloc[20]
        assert _gap_actual_demand(fe, anchor, start) is None


class TestForecastWriteFailureIsFailed:
    """#268 → #267: a forecast that computed but couldn't persist must return
    ok=False, so the region is counted failed (not scored)."""

    def test_persist_failure_marks_phase_failed(self, region_data, monkeypatch):
        import data.redis_client as rc
        from jobs import phases

        _patch_predict_one(monkeypatch, {"xgboost": np.full(HORIZON, 41_000.0)})
        # persist() calls redis_set; force it to fail so persist raises.
        monkeypatch.setattr(rc, "redis_set", lambda *a, **k: False)

        result = phases.predict_and_write_forecast(
            region_data,
            models={"xgboost": object()},
            model_mapes={"xgboost": 1.0},
        )
        assert result.ok is False
        assert "redis write failed" in (result.error or "")


class TestFutureFrameHolidayFlag:
    """P2-14 (#273): is_holiday is computed directly from the future
    timestamps, never smeared by the (hour, dow) group-mean imputer."""

    def _featured(self, n=400, holiday_poison=0.7):
        ts = pd.date_range("2026-11-01", periods=n, freq="h", tz="UTC")
        return pd.DataFrame(
            {
                "timestamp": ts,
                "demand_mw": np.full(n, 40_000.0),
                "hour": ts.hour,
                # Poison sentinel: if the imputer ever touches is_holiday
                # again, future values inherit this fractional mean.
                "is_holiday": np.full(n, holiday_poison),
                "temperature_2m": np.full(n, 60.0),
            }
        )

    def test_real_holiday_inside_horizon_flagged_one(self):
        """Christmas 2026 (Fri Dec 25 — fixed-date, non-observed-shifted)
        must read 1.0 for all 24 hours; surrounding days 0.0."""
        from jobs.phases import _build_future_feature_frame

        start = pd.Timestamp("2026-12-24 00:00", tz="UTC")
        future = _build_future_feature_frame(self._featured(), 72, weather_df=None, start_ts=start)
        flags = future.set_index("timestamp")["is_holiday"]
        dec25 = flags[flags.index.date == pd.Timestamp("2026-12-25").date()]
        assert len(dec25) == 24
        assert (dec25 == 1.0).all()
        dec26 = flags[flags.index.date == pd.Timestamp("2026-12-26").date()]
        assert (dec26 == 0.0).all()

    def test_never_fractional_despite_poisoned_history(self):
        """The pre-fix imputer produced fractional smears (a holiday in the
        28d window imprinted ~0.25 on every future week at that (hour, dow)).
        With the poison sentinel at 0.7, any imputer leak is caught."""
        from jobs.phases import _build_future_feature_frame

        start = pd.Timestamp("2026-12-24 00:00", tz="UTC")
        future = _build_future_feature_frame(self._featured(), 168, weather_df=None, start_ts=start)
        assert set(future["is_holiday"].unique()) <= {0.0, 1.0}


class TestWriteGenerationNullHonesty:
    """P2-08 (#273): an ALL-null generation window must never serve/cache an
    all-zero payload (the poison class #279 closed for demand).

    Scope note (verification catch): a PARTIAL null — one fuel null at an
    hour where others report — still reads 0 in the served series, because
    the post-pivot fillna(0) cannot distinguish a dropped null row from a
    fuel-column alignment gap. That residual matches pre-fix behavior and
    is tracked as a #273 follow-up (nullable payload lists + NaN-aware
    consumers); these tests deliberately do NOT claim it fixed."""

    def _run(self, gen_df, fake_redis, monkeypatch):
        import jobs.phases as phases

        monkeypatch.setattr(phases, "_has_eia_key", lambda: True)
        monkeypatch.setattr("data.eia_client.fetch_generation_by_fuel", lambda region: gen_df)
        return phases.write_generation("ERCOT")

    def test_all_null_window_writes_nothing(self, fake_redis, monkeypatch):
        """An all-null response (EIA outage artifact) previously parsed to an
        all-zero frame that was served and cached — now it is an honest
        empty result, and no payload lands in Redis."""
        ts = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
        gen_df = pd.DataFrame(
            {
                "timestamp": ts,
                "fuel_type": ["NG"] * 6,
                "generation_mw": [np.nan] * 6,
                "region": ["ERCOT"] * 6,
            }
        )
        result = self._run(gen_df, fake_redis, monkeypatch)
        assert result.ok is False
        assert result.error == "empty"
        assert not any("generation" in k for k in fake_redis)

    def test_partial_nulls_dropped_not_zero_filled(self, fake_redis, monkeypatch):
        """Null hours drop out; present readings serve normally."""
        ts = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
        gen_df = pd.DataFrame(
            {
                "timestamp": list(ts) * 2,
                "fuel_type": ["NG"] * 3 + ["WND"] * 3,
                "generation_mw": [1000.0, 1000.0, 1000.0, 500.0, np.nan, 500.0],
                "region": ["ERCOT"] * 6,
            }
        )
        result = self._run(gen_df, fake_redis, monkeypatch)
        assert result.ok
        payload = fake_redis["gridpulse:generation:ERCOT"]
        assert len(payload["timestamps"]) == 3


class TestBaselineSubstitutionBlock:
    """The skill block ``_baseline_substitution`` publishes.

    The block is nested under ``skill`` in ``gridpulse:forecast:{region}:1h``
    and passed through verbatim by ``/api/v1/forecast``, so its shape is a
    public surface. ``tests/unit/test_skill.py`` pins ``skill_payload``, which
    builds it — but until 2026-08-07 the job hand-rolled its own copy and the
    two had drifted, with every test defending the copy nothing called.

    These tests cover the seam that divergence hid in: that the job really
    does compose the published block out of ``skill_payload`` and
    ``should_serve_baseline``, rather than reproducing either by hand.
    """

    #: 7 days flat then one day 25% higher — the lag-24 baseline is wrong only
    #: on the final day, so it beats a deliberately terrible model MAPE.
    HOURS = 192

    def _run(self, monkeypatch, model_mape=20.0):
        import config
        import data.redis_client as rc
        from jobs import phases

        monkeypatch.setitem(config.FEATURE_FLAGS, "baseline_substitution", True)
        monkeypatch.setattr(
            rc,
            "redis_get",
            lambda key: {"models": {"ensemble": {"24h": {"rolling_mape_7d": model_mape}}}},
        )
        ts = pd.date_range("2026-08-01", periods=self.HOURS, freq="h", tz="UTC")
        demand = np.array([100.0] * (self.HOURS - 24) + [125.0] * 24)
        df = pd.DataFrame({"timestamp": ts, "demand_mw": demand})
        return phases._baseline_substitution("SEC", df, 24)

    def test_the_published_block_carries_every_field(self, monkeypatch):
        """Pinned as a key set, not field by field.

        The numeric values are ``skill_payload``'s job and are pinned there.
        What this asserts is the part that drifted: which keys reach Redis.
        A field dropped here becomes ``null`` in an API response with the
        whole suite green — dropping ``window_days=7`` from the call site did
        exactly that, and nothing failed.
        """
        result = self._run(monkeypatch)
        assert result is not None, "a model 6x worse than the baseline must substitute"
        _, block = result

        assert set(block) == {
            "model_mape",
            "baseline_mape",
            "baseline",
            "skill",
            "points_vs_baseline",
            "beats_baseline",
            "n_hours",
            "window_days",
            "decision",
        }

    def test_the_window_is_disclosed_as_the_seven_days_it_measures(self, monkeypatch):
        """``window_days`` is not decoration.

        The job compares a 7-day baseline against the drift instrument's
        7-day model MAPE deliberately — an earlier analysis mismatched the
        windows and reversed its own conclusion about which side won. A
        reader given the numbers without the window cannot check that.
        """
        _, block = self._run(monkeypatch)
        assert block["window_days"] == 7
        assert block["n_hours"] <= 24 * 7

    def test_beats_baseline_is_published_and_false(self, monkeypatch):
        """The field the module exists to serve, absent from the hand-rolled
        copy for as long as it existed. It is always False here: the block is
        only written when substitution fires, which requires the model to
        lose by more than the threshold."""
        _, block = self._run(monkeypatch)
        assert block["beats_baseline"] is False

    def test_the_decision_string_is_the_policy_s_own_reason(self, monkeypatch):
        """``decision`` must come from ``should_serve_baseline``, not be
        re-phrased at the call site — it is the disclosure a user gets for
        why their region's forecast changed source."""
        from models.skill import should_serve_baseline

        _, block = self._run(monkeypatch)
        expected = should_serve_baseline({k: block[k] for k in ("points_vs_baseline", "n_hours")})

        assert block["decision"] == expected[1]

    def test_a_winning_model_is_not_substituted(self, monkeypatch):
        assert self._run(monkeypatch, model_mape=0.5) is None


class TestEnsembleCompositionDivergence:
    """The guard that must fire when a flag flip changes the weight INPUT (#514).

    #444 made training and scoring share the weighting *rule* and record their
    *membership*, and logs ``ensemble_composition_divergence`` when the two
    memberships disagree. It does not compare the numbers fed to that rule.

    That gap is harmless while `smoothed_ensemble_weights` is off, because
    ``weighting_mape`` is then an identity function and both sides weight by the
    same holdout MAPE. Flip it and scoring weights by the EWMA while the
    persisted metric still weights by within-window holdout MAPEs — same three
    members, same rule, different numbers. That is P2-16's shape exactly, and
    before #514 the comparison was silent on it.
    """

    def _run(self, monkeypatch, fake_redis, region_data, *, weight_input, persisted_input):
        from jobs import phases

        _patch_predict_one(
            monkeypatch,
            {
                "xgboost": np.full(HORIZON, 41_000.0),
                "prophet": np.full(HORIZON, 39_000.0),
                "arima": np.full(HORIZON, 40_000.0),
            },
        )
        return phases.predict_and_write_forecast(
            region_data,
            models={"xgboost": object(), "prophet": object(), "arima": object()},
            model_mapes={"xgboost": 1.0, "prophet": 2.0, "arima": 4.0},
            model_metrics={
                "ensemble": {
                    "mape": 3.0,
                    "members": ["arima", "prophet", "xgboost"],
                    "weight_rule": "inverse_mape_cubed",
                    "weight_input": persisted_input,
                }
            },
            weight_input=weight_input,
        )

    ALL_HOLDOUT = {"xgboost": "holdout_mape", "prophet": "holdout_mape", "arima": "holdout_mape"}
    ALL_EWMA = {"xgboost": "mape_ewma", "prophet": "mape_ewma", "arima": "mape_ewma"}

    def test_the_served_basis_is_published(self, fake_redis, region_data, monkeypatch):
        """A reader of the payload can tell which number produced the weights."""
        result = self._run(
            monkeypatch,
            fake_redis,
            region_data,
            weight_input=self.ALL_HOLDOUT,
            persisted_input="holdout_mape",
        )
        assert result.ok
        comp = fake_redis["gridpulse:forecast:ERCOT:1h"]["ensemble_composition"]
        assert comp["weight_input"] == ["holdout_mape"]
        assert comp["members"] == ["arima", "prophet", "xgboost"]

    @staticmethod
    def _divergence_calls(recorder):
        return [
            c
            for c in recorder.info.call_args_list
            if c.args and c.args[0] == "ensemble_composition_divergence"
        ]

    def test_matching_bases_do_not_warn(self, fake_redis, region_data, monkeypatch):
        """No divergence when both sides weight by the same basis.

        Patches the module logger rather than using ``structlog.testing.\
        capture_logs``, for the reason recorded in ``test_shadow_weights.py``:
        capture_logs is defeated by a bound logger cached before it installed
        its processor, so it passes alone and fails in the full suite depending
        on which test configured structlog first. This assertion does not touch
        structlog's global state.
        """
        from unittest.mock import MagicMock, patch

        recorder = MagicMock()
        with patch("jobs.phases.log", recorder):
            result = self._run(
                monkeypatch,
                fake_redis,
                region_data,
                weight_input=self.ALL_HOLDOUT,
                persisted_input="holdout_mape",
            )
        assert result.ok
        assert not self._divergence_calls(recorder)

    def test_a_basis_mismatch_warns_even_with_identical_membership(
        self, fake_redis, region_data, monkeypatch
    ):
        """The whole point. Same three members, same rule — only the input
        differs, which is precisely what flipping the flag does."""
        from unittest.mock import MagicMock, patch

        recorder = MagicMock()
        with patch("jobs.phases.log", recorder):
            result = self._run(
                monkeypatch,
                fake_redis,
                region_data,
                weight_input=self.ALL_EWMA,
                persisted_input="holdout_mape",
            )
        assert result.ok
        hit = self._divergence_calls(recorder)
        assert len(hit) == 1, "a basis-only difference must not be silent"
        kw = hit[0].kwargs
        assert kw["served_input"] == ["mape_ewma"]
        assert kw["persisted_input"] == "holdout_mape"
        assert kw["served"] == kw["persisted_metric"], (
            "membership is identical — the input is the only difference"
        )

    def test_a_mixed_fleet_is_reported_as_mixed_not_collapsed(
        self, fake_redis, region_data, monkeypatch
    ):
        """The first run after a flip can genuinely be mixed: models whose meta
        predates the field fall back to the raw MAPE. Publishing one basis for
        the blend would hide that, so both are listed."""
        mixed = dict(self.ALL_EWMA, arima="holdout_mape")
        result = self._run(
            monkeypatch,
            fake_redis,
            region_data,
            weight_input=mixed,
            persisted_input="holdout_mape",
        )
        assert result.ok
        comp = fake_redis["gridpulse:forecast:ERCOT:1h"]["ensemble_composition"]
        assert comp["weight_input"] == ["holdout_mape", "mape_ewma"]

    def test_a_payload_without_the_field_still_publishes_a_basis(
        self, fake_redis, region_data, monkeypatch
    ):
        """``weight_input`` is optional on the call. Absent, the served basis
        must read ``unknown`` rather than silently claiming ``holdout_mape`` —
        an unlabelled run is not a holdout-weighted one."""
        result = self._run(
            monkeypatch,
            fake_redis,
            region_data,
            weight_input=None,
            persisted_input=None,
        )
        assert result.ok
        comp = fake_redis["gridpulse:forecast:ERCOT:1h"]["ensemble_composition"]
        assert comp["weight_input"] == ["unknown"]


class TestAnchorProvenanceRecording:
    """#547: record what ``_resolve_forecast_start`` anchored on, at forecast
    time, because nothing downstream can reconstruct it.

    ``docs/BENCHMARK_METHODOLOGY.md`` limit 11: where EIA has not metered an
    hour it publishes the BA's own day-ahead value in ``D``, and the resolver
    selects on *positive* ``D`` rather than on *metered* ``D`` — so our
    recursion is sometimes seeded with the series the benchmark scores us
    against. The per-BA rate ships as ``placeholder_pct``; which forecasts it
    touched was never recorded, so the materiality is stated as unmeasured.
    """

    @staticmethod
    def _frame(end_ts: str, n_hours: int = 48, mw: float = 20_000.0) -> pd.DataFrame:
        end = pd.Timestamp(end_ts, tz="UTC")
        ts = pd.date_range(end=end, periods=n_hours, freq="h")
        return pd.DataFrame(
            {
                "timestamp": ts,
                "demand_mw": np.full(n_hours, mw),
                "forecast_mw": np.full(n_hours, mw * 1.05),
            }
        )

    def _data(self, end_ts: str = "2026-08-18 06:00", **kw):
        from jobs.phases import RegionData

        frame = self._frame(end_ts)
        return RegionData(
            region="CAISO",
            demand_df=frame,
            weather_df=pd.DataFrame(),
            **kw,
        )

    def _call(self, data, featured=None, start="2026-08-18 07:00"):
        from jobs.phases import _anchor_provenance

        return _anchor_provenance(
            data,
            featured if featured is not None else data.demand_df,
            pd.Timestamp(start, tz="UTC"),
        )

    # ── the anchor hour and its value ───────────────────────────────────────

    def test_anchor_ts_is_the_hour_resolved_from_not_the_forecast_start(self):
        """The issue is explicit: the anchor hour, not the ``+1h`` start.

        Derived from ``forecast_start - 1h`` rather than by re-running the
        selection, so the recorded anchor and the anchor actually used cannot
        drift apart — and that identity holds on every branch of the
        resolver's fallback chain, since all of them return ``anchor + 1h``.
        """
        out = self._call(self._data())

        assert out["anchor_ts"] == pd.Timestamp("2026-08-18 06:00", tz="UTC").isoformat()

    def test_anchor_mw_is_the_value_that_seeds_demand_lag_1h(self):
        """Pinned against the real seeder, not against a hand-copied number.

        ``compute_autoregressive_snapshot`` sets ``demand_lag_1h`` to
        ``history[-1]`` after filtering the featured demand column to positive
        non-NaN, so the two must agree by construction.
        """
        from data.feature_engineering import compute_autoregressive_snapshot

        data = self._data()
        out = self._call(data)

        history = [float(v) for v in data.demand_df["demand_mw"] if v > 0]
        snapshot = compute_autoregressive_snapshot(history)

        assert out["anchor_mw"] == pytest.approx(snapshot["demand_lag_1h"])

    def test_an_anchor_hour_absent_from_every_frame_is_unknown(self):
        out = self._call(self._data(), start="2026-09-01 00:00")

        assert out["anchor_mw"] is None

    # ── tri-state ───────────────────────────────────────────────────────────

    def test_placeholder_is_none_when_the_vintage_window_never_answered(self):
        """Absent provenance and a confirmed metered anchor are DIFFERENT facts.

        ``False`` here would let a record written before the vintage phase ran
        claim its anchor was metered.
        """
        out = self._call(self._data())

        assert out["anchor_was_placeholder"] is None, "no map handed over -> unknown"

    def test_placeholder_is_none_for_an_hour_outside_the_window(self):
        """Vintage captures the RAW frame; the anchor resolves on the
        guard-cleaned one, so the two can disagree about which hours exist.
        That disagreement must read as unknown, not as metered."""
        out = self._call(self._data(placeholder_by_hour={"2026-08-01T00:00:00+00:00": True}))

        assert out["anchor_was_placeholder"] is None

    @pytest.mark.parametrize("flag", [True, False])
    def test_a_recorded_verdict_is_carried_through(self, flag):
        from data.vintage import canonical_hour

        key = canonical_hour(pd.Timestamp("2026-08-18 06:00", tz="UTC"))
        out = self._call(self._data(placeholder_by_hour={key: flag}))

        assert out["anchor_was_placeholder"] is flag

    # ── ADR-009, the OTHER way an anchor becomes the operator's forecast ────

    def test_an_unconditioned_region_records_a_confirmed_negative(self):
        """``anchor_frame`` IS ``demand_df`` here, so this is evidence, not an
        absence of it — and unlike the placeholder flag it can be ``False``."""
        out = self._call(self._data())

        assert out["anchor_conditioned"] is False

    def test_a_conditioned_anchor_is_not_reported_as_metered(self):
        """The defect this fourth field exists to prevent.

        ADR-009 substitutes the BA's own ``forecast_mw`` into the trailing
        hours of broken-class feeds, deliberately, because it measured better
        (58.2% wrong vs 14.5%). Vintage records the RAW ``D``, so such an
        anchor reads ``was_placeholder=False`` while the value that seeded the
        model was their day-ahead figure — a true field whose framing asserts
        something false.
        """
        from data.vintage import canonical_hour

        data = self._data()
        conditioned = data.demand_df.copy()
        conditioned.loc[conditioned.index[-1], "demand_mw"] = 21_000.0
        data.conditioned_demand_df = conditioned
        key = canonical_hour(pd.Timestamp("2026-08-18 06:00", tz="UTC"))
        data.placeholder_by_hour = {key: False}

        out = self._call(data, featured=conditioned)

        assert out["anchor_was_placeholder"] is False, "the raw D genuinely was not a placeholder"
        assert out["anchor_conditioned"] is True, "but the seed WAS their day-ahead value"
        assert out["anchor_mw"] == pytest.approx(21_000.0), "and the substituted value is recorded"

    def test_a_conditioned_region_whose_anchor_hour_was_not_substituted(self):
        """Conditioning touches only the trailing hours, so the flag is decided
        by comparing the frames at the anchor hour — a fact about the value
        used, not an inference from the region's class."""
        data = self._data()
        conditioned = data.demand_df.copy()
        conditioned.loc[conditioned.index[-1], "demand_mw"] = 21_000.0
        data.conditioned_demand_df = conditioned

        out = self._call(data, featured=conditioned, start="2026-08-18 05:00")

        assert out["anchor_conditioned"] is False


class TestAnchorNeverLandsOnAForecastRow:
    """#547 guard. The anchor block belongs at the payload TOP LEVEL.

    ``extract_one_hour_ahead_predictions`` iterates every key on a forecast row
    and treats each remaining numeric value as a model, so a per-row
    ``anchor_mw`` would silently acquire its own drift records, a Models-tab
    entry, and a place in the rolling MAPE the visibility gate reads.
    """

    def test_a_row_carrying_an_anchor_would_be_read_as_a_model(self):
        """States the hazard as an executable fact rather than a comment."""
        from models.drift import extract_one_hour_ahead_predictions

        payload = {
            "forecasts": [
                {
                    "timestamp": "2026-08-18T06:00:00+00:00",
                    "predicted_demand_mw": 4200.0,
                    "xgboost": 4200.0,
                    "anchor_mw": 4100.0,
                }
            ]
        }
        preds = extract_one_hour_ahead_predictions(payload, "2026-08-18T06:00:00+00:00")

        assert "anchor_mw" in preds, "this is WHY the anchor must stay off the rows"

    def test_the_written_payload_keeps_the_anchor_off_every_row(self):
        import inspect

        from jobs import phases

        src = inspect.getsource(phases.predict_and_write_forecast)
        row_build = src.split("row: dict[str, Any] = {")[1].split("fl.append(row)")[0]

        assert "anchor" not in row_build, "the anchor must not be built into a forecast row"
        assert '"anchor": anchor,' in src, "and must be on the payload top level"


class TestAnchorAndTheOriginGuardCompose:
    """#547 x #537: the two changes meet at the same call site.

    ``_anchor_provenance`` runs immediately after ``_resolve_forecast_start``,
    and the origin-regression guard returns between them. The ordering is
    load-bearing in one direction only, and it is not obvious from either
    change read alone — which is exactly the kind of seam that survives both
    test suites and fails in production.
    """

    @staticmethod
    def _origin(ts: str) -> pd.Timestamp:
        return pd.Timestamp(ts, tz="UTC")

    def test_a_regressed_origin_stamps_no_anchor_because_it_serves_no_payload(
        self, region_data, fake_redis, monkeypatch
    ):
        """An anchor must describe a forecast that was actually served.

        On a regressed origin the phase keeps the newer payload already in
        Redis and writes nothing. If the anchor were computed and stamped
        before the guard, it would describe a forecast this tick declined to
        publish — and, worse, could be paired against the *previous* payload's
        rows by the drift phases.
        """
        from jobs.phases import predict_and_write_forecast

        _patch_predict_one(monkeypatch, {"xgboost": np.full(HORIZON, 41_000.0)})
        region_data.previous_forecast_origin = self._origin("2024-02-01 00:00")

        result = predict_and_write_forecast(region_data, {"xgboost": object()})

        assert result.details["skipped"] == "origin_regressed"
        assert "anchor_was_placeholder" not in result.details, (
            "a declined tick must not report an anchor it never served"
        )
        assert fake_redis == {}

    def test_a_normal_tick_still_carries_both(self, region_data, fake_redis, monkeypatch):
        """The guard must not cost the instrument its reading on healthy ticks."""
        from jobs.phases import predict_and_write_forecast

        _patch_predict_one(monkeypatch, {"xgboost": np.full(HORIZON, 41_000.0)})
        region_data.previous_forecast_origin = self._origin("2024-01-01 00:00")

        result = predict_and_write_forecast(region_data, {"xgboost": object()})

        assert "skipped" not in result.details
        assert "anchor_was_placeholder" in result.details

        key = next(k for k in fake_redis if k.endswith("forecast:ERCOT:1h"))
        anchor = fake_redis[key]["anchor"]
        # The resolved start is 2024-01-09T08:00 (the frame ends at 07:00), so
        # the anchor is the hour before it.
        assert anchor["anchor_ts"] == "2024-01-09T07:00:00+00:00"

        # NOT asserted here: that ``anchor_ts == forecasts[0] - 1h``. It is true
        # in production and is pinned by the integration test, but this class's
        # ``_patch_predict_one`` stubs ``_build_future_feature_frame`` with a
        # lambda that IGNORES ``start_ts`` and hardcodes a 2024-02-01 range — so
        # asserting it here would test the stub's constant, not the coupling.
        # Left explicit because the assertion looks obviously correct and passes
        # nowhere it would mean anything.
