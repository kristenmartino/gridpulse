"""Integration tests for jobs/scoring_job.py.

All external I/O is faked:
- EIA + Open-Meteo fetchers are monkeypatched to return synthetic DataFrames.
- ``data.redis_client.redis_set`` is replaced with an in-memory dict writer.
- ``models.persistence.load_model`` is monkeypatched to return a tiny fake model.

The tests assert the scoring job writes the expected wattcast:* keys and
returns a success exit code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fake_redis(monkeypatch):
    """In-memory replacement for redis_set used by phases."""
    store: dict[str, dict] = {}

    def _set(key: str, value, ttl: int = 86400) -> bool:
        store[key] = value
        return True

    # Patch in every module that imports redis_set lazily.
    import data.redis_client as rc

    monkeypatch.setattr(rc, "redis_set", _set)
    return store


@pytest.fixture
def synthetic_region_frames():
    """Build 30 days of synthetic demand + weather + generation-by-fuel."""
    ts = pd.date_range("2024-01-01", periods=30 * 24, freq="h", tz="UTC")
    n = len(ts)
    demand = pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": 40_000 + 5000 * np.sin(2 * np.pi * np.arange(n) / 24)
            + np.random.normal(0, 200, n),
            "region": "ERCOT",
        }
    )
    weather = pd.DataFrame(
        {
            "timestamp": ts,
            "temperature_2m": 70 + 10 * np.sin(2 * np.pi * np.arange(n) / 24),
            "apparent_temperature": 70.0,
            "relative_humidity_2m": 60.0,
            "dew_point_2m": 50.0,
            "wind_speed_10m": 8.0,
            "wind_speed_80m": 12.0,
            "wind_speed_120m": 15.0,
            "wind_direction_10m": 180.0,
            "shortwave_radiation": np.maximum(
                0, 500 * np.sin(2 * np.pi * (np.arange(n) - 6) / 24)
            ),
            "direct_normal_irradiance": 0.0,
            "diffuse_radiation": 0.0,
            "cloud_cover": 40.0,
            "precipitation": 0.0,
            "snowfall": 0.0,
            "surface_pressure": 1013.0,
            "soil_temperature_0cm": 65.0,
            "weather_code": 0,
        }
    )
    generation = pd.DataFrame(
        {
            "timestamp": ts[:168],
            "fuel_type": ["NG"] * 168,
            "generation_mw": 20_000 + np.random.normal(0, 500, 168),
        }
    )
    return demand, weather, generation


@pytest.fixture
def patch_data_sources(monkeypatch, synthetic_region_frames):
    """Replace EIA + weather client fetchers with synthetic data."""
    demand_df, weather_df, gen_df = synthetic_region_frames

    import data.eia_client as eia
    import data.weather_client as weather

    def _fetch_demand(region, **kwargs):
        df = demand_df.copy()
        df["region"] = region
        return df

    def _fetch_weather(region, **kwargs):
        return weather_df.copy()

    def _fetch_generation_by_fuel(region, **kwargs):
        return gen_df.copy()

    monkeypatch.setattr(eia, "fetch_demand", _fetch_demand)
    monkeypatch.setattr(eia, "fetch_generation_by_fuel", _fetch_generation_by_fuel)
    monkeypatch.setattr(weather, "fetch_weather", _fetch_weather)
    # Ensure _has_eia_key() returns True without depending on environment
    import jobs.phases as phases

    monkeypatch.setattr(phases, "_has_eia_key", lambda: True)


@pytest.fixture
def patch_single_region(monkeypatch):
    """Limit scoring to a single region for test speed."""
    import jobs.phases as phases

    monkeypatch.setattr(phases, "ordered_regions", lambda *a, **kw: ["ERCOT"])


def _fake_xgb_model() -> dict:
    """Tiny fake xgboost model payload that predict_xgboost can tolerate."""

    class _FakeBooster:
        def predict(self, X):
            return np.full(len(X), 40_000.0)

    return {
        "model": _FakeBooster(),
        "feature_importances": {"temperature_2m": 1.0},
        "feature_cols": ["hour", "day_of_week"],
    }


class TestScoringJob:
    def test_scoring_job_happy_path(
        self,
        fake_redis,
        patch_data_sources,
        patch_single_region,
        monkeypatch,
    ) -> None:
        """Scoring job writes the expected Redis keys and returns exit code 0."""
        # Patch model load to supply a fake XGBoost model.
        from models import persistence as mp

        fake_model = _fake_xgb_model()
        fake_meta = mp.ModelMetadata(
            region="ERCOT",
            model_name="xgboost",
            version="v-test",
            data_hash="h",
            trained_at="",
            train_rows=1,
            mape=5.0,
            lib_versions={},
            extra={},
        )
        monkeypatch.setattr(
            "jobs.scoring_job.load_model",
            lambda region, model_name: (fake_model, fake_meta),
        )

        # Patch predict_xgboost to bypass feature alignment complexity.
        import models.xgboost_model as xgb_mod

        monkeypatch.setattr(
            xgb_mod,
            "predict_xgboost",
            lambda model, X: np.full(len(X), 41_000.0),
        )

        # Patch the diagnostics path's forecast service to avoid training.
        import models.model_service as model_service

        monkeypatch.setattr(
            model_service,
            "get_forecasts",
            lambda region, df: {"ensemble": df["demand_mw"].values, "metrics": {}},
        )

        from jobs import scoring_job

        exit_code = scoring_job.run()
        assert exit_code == 0

        # Must have refreshed the core Redis keys for ERCOT.
        expected_keys = {
            "wattcast:actuals:ERCOT",
            "wattcast:weather:ERCOT",
            "wattcast:generation:ERCOT",
            "wattcast:forecast:ERCOT:1h",
            "wattcast:weather-correlation:ERCOT",
            "wattcast:diagnostics:ERCOT",
            "wattcast:alerts:ERCOT",
            "wattcast:meta:last_scored",
        }
        missing = expected_keys - set(fake_redis.keys())
        assert not missing, f"Missing Redis keys: {missing}"

        # last_scored must record the successful region count.
        meta = fake_redis["wattcast:meta:last_scored"]
        assert meta["regions_scored"] == 1
        assert meta["mode"] == "scoring-job"

    def test_scoring_job_missing_model_still_writes_actuals(
        self,
        fake_redis,
        patch_data_sources,
        patch_single_region,
        monkeypatch,
    ) -> None:
        """No model in GCS → still writes actuals/weather/generation/alerts."""
        monkeypatch.setattr(
            "jobs.scoring_job.load_model", lambda region, model_name: None
        )

        from jobs import scoring_job

        exit_code = scoring_job.run()
        # Non-model phases still succeed → exit 0.
        assert exit_code == 0

        # Actuals/weather/generation/alerts must still be present.
        for key in (
            "wattcast:actuals:ERCOT",
            "wattcast:weather:ERCOT",
            "wattcast:generation:ERCOT",
            "wattcast:alerts:ERCOT",
            "wattcast:meta:last_scored",
        ):
            assert key in fake_redis

        # Forecast key must NOT be present when the model is missing.
        assert "wattcast:forecast:ERCOT:1h" not in fake_redis

    def test_scoring_job_no_data_returns_failure(
        self,
        fake_redis,
        patch_single_region,
        monkeypatch,
    ) -> None:
        """Every region failing data fetch → exit code 1."""
        import jobs.phases as phases

        monkeypatch.setattr(phases, "fetch_region_data", lambda region: None)

        from jobs import scoring_job

        exit_code = scoring_job.run()
        assert exit_code == 1
        # last_scored still gets written with the failure summary.
        assert fake_redis["wattcast:meta:last_scored"]["regions_scored"] == 0
        assert "ERCOT" in fake_redis["wattcast:meta:last_scored"]["regions_failed"]
