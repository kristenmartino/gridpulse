"""Integration tests for jobs/training_job.py.

External I/O (EIA, weather, GCS) is faked. Training itself is monkeypatched
so the test runs fast and does not depend on xgboost / prophet / pmdarima
behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fake_redis(monkeypatch):
    store: dict[str, dict] = {}

    def _set(key: str, value, ttl: int = 86400) -> bool:
        store[key] = value
        return True

    import data.redis_client as rc

    monkeypatch.setattr(rc, "redis_set", _set)
    return store


@pytest.fixture
def synthetic_frames():
    ts = pd.date_range("2024-01-01", periods=30 * 24, freq="h", tz="UTC")
    n = len(ts)
    demand = pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": 40_000
            + 5000 * np.sin(2 * np.pi * np.arange(n) / 24)
            + np.random.normal(0, 200, n),
            "region": "ERCOT",
        }
    )
    weather = pd.DataFrame(
        {
            "timestamp": ts,
            "temperature_2m": 70.0,
            "apparent_temperature": 70.0,
            "relative_humidity_2m": 60.0,
            "dew_point_2m": 50.0,
            "wind_speed_10m": 8.0,
            "wind_speed_80m": 12.0,
            "wind_speed_120m": 15.0,
            "wind_direction_10m": 180.0,
            "shortwave_radiation": 100.0,
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
    return demand, weather


@pytest.fixture
def patch_data_sources(monkeypatch, synthetic_frames):
    demand_df, weather_df = synthetic_frames

    import data.eia_client as eia
    import data.weather_client as weather
    import jobs.phases as phases

    monkeypatch.setattr(eia, "fetch_demand", lambda region, **kw: demand_df.copy())
    monkeypatch.setattr(weather, "fetch_weather", lambda region, **kw: weather_df.copy())
    monkeypatch.setattr(phases, "_has_eia_key", lambda: True)


@pytest.fixture
def patch_single_region(monkeypatch):
    import jobs.phases as phases

    monkeypatch.setattr(phases, "ordered_regions", lambda *a, **kw: ["ERCOT"])


class TestTrainingJob:
    def test_training_job_persists_models_and_marks_meta(
        self,
        fake_redis,
        patch_data_sources,
        patch_single_region,
        monkeypatch,
    ) -> None:
        """Training job saves an XGBoost model, writes backtests, and marks meta."""
        # Fake each model trainer so we skip the heavy libs.
        import models.arima_model as arima_mod
        import models.prophet_model as prophet_mod
        import models.xgboost_model as xgb_mod

        monkeypatch.setattr(
            xgb_mod,
            "train_xgboost",
            lambda df, n_splits=3: {
                "model": {"booster": "fake"},
                "feature_importances": {"temperature_2m": 0.5},
                "cv_scores": [5.0, 5.5, 6.0],
            },
        )
        monkeypatch.setattr(
            prophet_mod, "train_prophet", lambda df: {"type": "prophet", "params": [1]}
        )
        monkeypatch.setattr(
            arima_mod, "train_arima", lambda df: {"type": "sarimax", "order": (1, 0, 1)}
        )

        # Capture save_model calls — substitute an in-memory implementation.
        saved_versions: dict[tuple[str, str], str] = {}

        def _save_model(
            region, model_name, model_obj, data_hash, train_rows, mape=None, extra=None
        ):
            version = f"v-{region}-{model_name}"
            saved_versions[(region, model_name)] = version
            return version

        monkeypatch.setattr("jobs.training_job.save_model", _save_model)

        # Stub backtests — the real path pulls callbacks into scope.
        import jobs.phases as phases

        monkeypatch.setattr(
            phases,
            "write_backtests",
            lambda data: phases.PhaseResult(
                region=data.region, ok=True, details={"horizons_written": [24, 168]}
            ),
        )

        from jobs import training_job

        exit_code = training_job.run()
        assert exit_code == 0

        # All three model types got persisted for ERCOT.
        assert ("ERCOT", "xgboost") in saved_versions
        assert ("ERCOT", "prophet") in saved_versions
        assert ("ERCOT", "arima") in saved_versions

        # last_trained meta is present and reflects the successful region.
        assert "wattcast:meta:last_trained" in fake_redis
        meta = fake_redis["wattcast:meta:last_trained"]
        assert meta["regions_trained"] == 1
        assert meta["mode"] == "training-job"

    def test_training_job_xgboost_failure_marks_region_failed(
        self,
        fake_redis,
        patch_data_sources,
        patch_single_region,
        monkeypatch,
    ) -> None:
        """If XGBoost save returns None, the region is marked failed."""
        import models.arima_model as arima_mod
        import models.prophet_model as prophet_mod
        import models.xgboost_model as xgb_mod

        monkeypatch.setattr(
            xgb_mod,
            "train_xgboost",
            lambda df, n_splits=3: {"model": "x", "cv_scores": []},
        )
        monkeypatch.setattr(prophet_mod, "train_prophet", lambda df: {"type": "prophet"})
        monkeypatch.setattr(arima_mod, "train_arima", lambda df: {"type": "sarimax"})

        # save_model returns None for xgboost (simulates GCS failure),
        # works for everything else.
        def _save_model(region, model_name, *args, **kwargs):
            if model_name == "xgboost":
                return None
            return f"v-{region}-{model_name}"

        monkeypatch.setattr("jobs.training_job.save_model", _save_model)

        import jobs.phases as phases

        monkeypatch.setattr(
            phases,
            "write_backtests",
            lambda data: phases.PhaseResult(region=data.region, ok=True),
        )

        from jobs import training_job

        exit_code = training_job.run()
        assert exit_code == 1
        # Region must show up in failed list when xgboost couldn't be saved.
        assert "ERCOT" in fake_redis["wattcast:meta:last_trained"]["regions_failed"]

    def test_training_job_no_data_records_failure(
        self,
        fake_redis,
        patch_single_region,
        monkeypatch,
    ) -> None:
        """fetch_region_data returning None flows through as a region failure."""
        import jobs.phases as phases

        monkeypatch.setattr(phases, "fetch_region_data", lambda region: None)

        from jobs import training_job

        exit_code = training_job.run()
        assert exit_code == 1
        assert "ERCOT" in fake_redis["wattcast:meta:last_trained"]["regions_failed"]
