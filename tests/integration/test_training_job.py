"""Integration tests for jobs/training_job.py.

External I/O (EIA, weather, GCS) is faked. Training itself is monkeypatched
so the test runs fast and does not depend on xgboost / prophet / pmdarima
behavior.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _stub_weather_normals(monkeypatch):
    """Keep the post-training normals refresh off the network.

    The file docstring says external I/O is faked, but only one test
    (``test_...refresh...``) ever patched this. The rest called the real
    ``refresh_weather_normals``, which requests a 10-year ERA5 archive window
    (~17 variables) per region from ``archive-api.open-meteo.com``. They
    survived only because Open-Meteo rate-limited us quickly; a runner that
    got a 200 would have downloaded the lot.

    Autouse so new tests are covered by default. A test that wants a
    different behaviour (e.g. asserting the failure path) still overrides it
    with its own ``monkeypatch.setattr`` — last patch wins.
    """
    monkeypatch.setattr("data.weather_normals.refresh_weather_normals", lambda regions: None)


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
            arima_mod,
            "train_arima",
            lambda df, **kwargs: {
                "type": "sarimax",
                "order": (1, 0, 1),
                "seasonal_order": (1, 1, 1, 24),
            },
        )

        # Capture save_model calls — substitute an in-memory implementation.
        saved_versions: dict[tuple[str, str], str] = {}

        def _save_model(
            region,
            model_name,
            model_obj,
            data_hash,
            train_rows,
            mape=None,
            extra=None,
            update_latest=True,
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
        assert "gridpulse:meta:last_trained" in fake_redis
        meta = fake_redis["gridpulse:meta:last_trained"]
        assert meta["regions_trained"] == 1
        assert meta["mode"] == "training-job"

    def test_weather_normal_refresh_is_best_effort_and_after_meta(
        self,
        fake_redis,
        patch_data_sources,
        patch_single_region,
        monkeypatch,
    ) -> None:
        """#283 Phase 1: a failing/slow weather-normal refresh must neither fail
        training nor block the last_trained pointer — the refresh runs AFTER the
        meta write and is best-effort."""
        import models.arima_model as arima_mod
        import models.prophet_model as prophet_mod
        import models.xgboost_model as xgb_mod

        monkeypatch.setattr(
            xgb_mod,
            "train_xgboost",
            lambda df, n_splits=3: {
                "model": {"booster": "fake"},
                "feature_importances": {"temperature_2m": 0.5},
                "cv_scores": [5.0],
            },
        )
        monkeypatch.setattr(prophet_mod, "train_prophet", lambda df: {"type": "prophet"})
        monkeypatch.setattr(arima_mod, "train_arima", lambda df, **kwargs: {"type": "sarimax"})
        monkeypatch.setattr("jobs.training_job.save_model", lambda *a, **k: "v1")

        import jobs.phases as phases

        monkeypatch.setattr(
            phases,
            "write_backtests",
            lambda data: phases.PhaseResult(region=data.region, ok=True, details={}),
        )

        def _boom(regions):
            raise RuntimeError("archive down")

        monkeypatch.setattr("data.weather_normals.refresh_weather_normals", _boom)

        from jobs import training_job

        # Training still succeeds despite the refresh blowing up...
        assert training_job.run() == 0
        # ...and the last_trained pointer was written (refresh runs AFTER it).
        assert "gridpulse:meta:last_trained" in fake_redis

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
        monkeypatch.setattr(arima_mod, "train_arima", lambda df, **kwargs: {"type": "sarimax"})

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
        assert "ERCOT" in fake_redis["gridpulse:meta:last_trained"]["regions_failed"]

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
        assert "ERCOT" in fake_redis["gridpulse:meta:last_trained"]["regions_failed"]


class TestServeGateWiring:
    """#326: the serve-path gate's verdict drives ``save_model(update_latest=…)``.

    A rejected candidate is still persisted (the forensic record the dive
    study depended on) but must never repoint ``latest.json``.
    """

    @pytest.fixture
    def region_data(self, synthetic_frames):
        import jobs.phases as phases

        demand_df, weather_df = synthetic_frames
        rd = phases.RegionData(region="ERCOT", demand_df=demand_df, weather_df=weather_df)
        rd.featured_df = demand_df.copy()  # content irrelevant — the gate is patched
        return rd

    def _wire(self, monkeypatch, verdict: dict) -> dict:
        import jobs.phases as phases
        import models.xgboost_model as xgb_mod

        monkeypatch.setattr(
            xgb_mod,
            "train_xgboost",
            lambda df, n_splits=3: {"model": {"booster": "fake"}, "cv_scores": [5.0]},
        )
        monkeypatch.setattr(phases, "serve_path_gate", lambda *a, **kw: verdict)

        calls: dict = {}

        def _save_model(
            region,
            model_name,
            model_obj,
            data_hash,
            train_rows,
            mape=None,
            extra=None,
            update_latest=True,
        ):
            calls.update(extra=extra, update_latest=update_latest)
            return "v-test"

        monkeypatch.setattr("jobs.training_job.save_model", _save_model)
        return calls

    def test_rejected_candidate_persists_but_never_repoints(self, monkeypatch, region_data):
        verdict = {"passed": False, "anchors": [{"ok": False, "trough_ratio": 0.3}]}
        calls = self._wire(monkeypatch, verdict)

        from jobs import training_job

        assert training_job._train_xgboost(region_data) == "v-test"
        assert calls["update_latest"] is False
        assert calls["extra"]["serve_gate"] == verdict

    def test_accepted_candidate_repoints_latest(self, monkeypatch, region_data):
        verdict = {"passed": True, "anchors": [{"ok": True}]}
        calls = self._wire(monkeypatch, verdict)

        from jobs import training_job

        assert training_job._train_xgboost(region_data) == "v-test"
        assert calls["update_latest"] is True
        assert calls["extra"]["serve_gate"] == verdict


class TestTrainingQualityGuard:
    """#309/#326 hygiene: the artifact guard runs on the training frame, so
    gross partials never become training targets, holdout ground truth, or
    the ADR-010 gate's reference/truth rows."""

    def test_trailing_artifact_never_reaches_training_targets(
        self,
        fake_redis,
        patch_single_region,
        monkeypatch,
        synthetic_frames,
    ) -> None:
        """A trailing near-zero partial (signal 1 — no forecast_mw needed,
        mirroring the GCS-fallback degradation path) must be absent from the
        frame handed to train_xgboost. This one assertion proves both the
        wiring and the ordering: a guard placed after feature engineering
        could not have removed the row."""
        demand_df, weather_df = synthetic_frames
        artifact_ts = demand_df["timestamp"].iloc[-1]
        demand_df = demand_df.copy()
        demand_df.loc[demand_df.index[-1], "demand_mw"] = 400.0  # ~1% of level

        import data.eia_client as eia
        import data.weather_client as weather
        import jobs.phases as phases

        monkeypatch.setattr(eia, "fetch_demand", lambda region, **kw: demand_df.copy())
        monkeypatch.setattr(weather, "fetch_weather", lambda region, **kw: weather_df.copy())
        monkeypatch.setattr(phases, "_has_eia_key", lambda: True)

        import models.arima_model as arima_mod
        import models.prophet_model as prophet_mod
        import models.xgboost_model as xgb_mod

        captured: dict = {}

        def _fake_train_xgboost(df, n_splits=3):
            captured["train_df"] = df.copy()
            return {"model": {"booster": "fake"}, "cv_scores": [5.0]}

        monkeypatch.setattr(xgb_mod, "train_xgboost", _fake_train_xgboost)
        monkeypatch.setattr(prophet_mod, "train_prophet", lambda df: {"type": "prophet"})
        monkeypatch.setattr(arima_mod, "train_arima", lambda df, **kwargs: {"type": "sarimax"})
        monkeypatch.setattr("jobs.training_job.save_model", lambda *a, **k: "v-guard-test")
        monkeypatch.setattr(
            phases,
            "write_backtests",
            lambda data: phases.PhaseResult(region=data.region, ok=True, details={}),
        )

        from jobs import training_job

        assert training_job.run() == 0  # the guard is never fatal

        train_df = captured["train_df"]
        train_ts = pd.to_datetime(train_df["timestamp"], utc=True)
        assert artifact_ts not in set(train_ts), (
            "the guard-coerced artifact hour reached the training frame"
        )
        assert float(train_df["demand_mw"].min()) > 1000.0, (
            "a partial-band demand value survived into training targets"
        )


class TestForceBypassesTheArimaOrderCache:
    """``--force`` is the only thing that re-runs the pmdarima order search.

    The cache has no invalidation — no TTL, no age check, no data-hash
    comparison — so once an order is stored it is reused forever. That is a
    deliberate cost decision, but before this change the docstring claimed the
    cache "is invalidated automatically if the data changes enough" and that
    "a force retrain bypasses it entirely". Neither was true: `force` gated
    only the data-hash resume, and nothing set it to True. These tests pin the
    behaviour the documentation now describes.
    """

    def test_force_skips_the_cached_order(self, monkeypatch):
        from jobs import training_job

        monkeypatch.setattr(
            training_job, "_read_cached_arima_order", lambda region: ((9, 9, 9), (9, 9, 9, 24))
        )
        seen = {}

        def fake_train_arima(df, cached_order=None, cached_seasonal_order=None, **kw):
            seen["order"] = cached_order
            raise RuntimeError("stop after the cache decision")

        import models.arima_model as am

        monkeypatch.setattr(am, "train_arima", fake_train_arima)
        rd = types.SimpleNamespace(region="ERCOT", featured_df=pd.DataFrame({"demand_mw": [1.0]}))

        training_job._train_arima(rd, force=True)

        assert seen["order"] is None, "force must not pass a cached order"

    def test_default_uses_the_cached_order(self, monkeypatch):
        """The saving is real and must survive — this is the money path."""
        from jobs import training_job

        monkeypatch.setattr(
            training_job, "_read_cached_arima_order", lambda region: ((2, 0, 1), (1, 1, 0, 24))
        )
        seen = {}

        def fake_train_arima(df, cached_order=None, cached_seasonal_order=None, **kw):
            seen["order"] = cached_order
            raise RuntimeError("stop after the cache decision")

        import models.arima_model as am

        monkeypatch.setattr(am, "train_arima", fake_train_arima)
        rd = types.SimpleNamespace(region="ERCOT", featured_df=pd.DataFrame({"demand_mw": [1.0]}))

        training_job._train_arima(rd)

        assert seen["order"] == (2, 0, 1)

    def test_holdout_mirrors_the_production_fit(self, monkeypatch):
        """If the holdout kept the cached order under --force it would score a
        different (p,d,q) than the model actually served."""
        from jobs import training_job

        monkeypatch.setattr(
            training_job, "_read_cached_arima_order", lambda region: ((5, 5, 5), (5, 5, 5, 24))
        )
        seen = {}

        def fake_train_arima(df, cached_order=None, cached_seasonal_order=None, **kw):
            seen["order"] = cached_order
            raise RuntimeError("stop after the cache decision")

        import models.arima_model as am

        monkeypatch.setattr(am, "train_arima", fake_train_arima)
        featured = pd.DataFrame({"demand_mw": [float(i) for i in range(1000)]})

        training_job._holdout_metrics_arima(featured, "ERCOT", force=True)

        assert seen["order"] is None


class TestForceReachesTheCli:
    """Before this change ``force`` had no caller that set it True — the
    escape hatch the docstring pointed at did not exist."""

    def test_force_flag_threads_to_run(self, monkeypatch):
        import jobs.__main__ as jobs_main

        got = {}
        monkeypatch.setitem(jobs_main._ENTRYPOINTS, "training", lambda **kw: got.update(kw) or 0)

        assert jobs_main.main(["training", "--force"]) == 0
        assert got == {"force": True}

    def test_default_is_not_forced(self, monkeypatch):
        import jobs.__main__ as jobs_main

        got = {}
        monkeypatch.setitem(jobs_main._ENTRYPOINTS, "training", lambda **kw: got.update(kw) or 0)

        assert jobs_main.main(["training"]) == 0
        assert got == {"force": False}

    def test_force_is_rejected_for_scoring_not_ignored(self, monkeypatch):
        """Silently dropping it would let an operator believe they forced a
        retrain when they had not.

        The entrypoint is stubbed AND asserted un-called: rejection has to
        happen before dispatch. Without the stub this test would invoke the
        real scoring job when the guard is removed — so it would still fail,
        but by hanging on live network work instead of asserting, which is a
        useless failure mode to leave in the suite.
        """
        import jobs.__main__ as jobs_main

        called = []
        monkeypatch.setitem(jobs_main._ENTRYPOINTS, "scoring", lambda **kw: called.append(kw) or 0)

        assert jobs_main.main(["scoring", "--force"]) == 2
        assert called == [], "scoring must not run at all when --force is rejected"

    def test_unknown_flag_is_rejected(self, monkeypatch):
        """A typo'd ``--forse`` must not silently run an ordinary retrain.

        Entrypoint stubbed and asserted un-called, for the same reason as
        above: if the guard is removed, an unstubbed test would launch the
        real training job and hang rather than fail.
        """
        import jobs.__main__ as jobs_main

        called = []
        monkeypatch.setitem(jobs_main._ENTRYPOINTS, "training", lambda **kw: called.append(kw) or 0)

        assert jobs_main.main(["training", "--forse"]) == 2
        assert called == []
