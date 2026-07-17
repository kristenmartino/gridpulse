"""Integration tests for jobs/scoring_job.py.

All external I/O is faked:
- EIA + Open-Meteo fetchers are monkeypatched to return synthetic DataFrames.
- ``data.redis_client.redis_set`` is replaced with an in-memory dict writer.
- ``models.persistence.load_model`` is monkeypatched to return a tiny fake model.

The tests assert the scoring job writes the expected gridpulse:* keys and
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
    demand_mw = 40_000 + 5000 * np.sin(2 * np.pi * np.arange(n) / 24) + np.random.normal(0, 200, n)
    demand = pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": demand_mw,
            "region": "ERCOT",
            # ``_parse_demand_records`` ALWAYS emits forecast_mw (EIA's DF
            # series), all-NaN when the BA published none — see
            # tests/unit/test_eia_client.py::…forecast_mw is None. Omitting it
            # here made the fixture claim a schema the real client never
            # returns, which would hide a consumer that needs it (#309).
            "forecast_mw": demand_mw + np.random.normal(0, 400, n),
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
            "shortwave_radiation": np.maximum(0, 500 * np.sin(2 * np.pi * (np.arange(n) - 6) / 24)),
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

    # Keep the alerts phase hermetic: the scoring job now fetches live
    # NOAA/NWS alerts, so stub the client to avoid real network calls in CI.
    import data.noaa_client as noaa

    monkeypatch.setattr(noaa, "fetch_alerts_for_region", lambda region, **kw: [])

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
        def predict(self, x):
            return np.full(len(x), 40_000.0)

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
            lambda model, x: np.full(len(x), 41_000.0),
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
            "gridpulse:actuals:ERCOT",
            "gridpulse:weather:ERCOT",
            "gridpulse:generation:ERCOT",
            "gridpulse:forecast:ERCOT:1h",
            "gridpulse:weather-correlation:ERCOT",
            "gridpulse:diagnostics:ERCOT",
            "gridpulse:alerts:ERCOT",
            "gridpulse:meta:last_scored",
        }
        missing = expected_keys - set(fake_redis.keys())
        assert not missing, f"Missing Redis keys: {missing}"

        # last_scored must record the successful region count.
        meta = fake_redis["gridpulse:meta:last_scored"]
        assert meta["regions_scored"] == 1
        assert meta["mode"] == "scoring-job"

    def test_real_score_region_attaches_and_publishes_gate_verdict(
        self,
        fake_redis,
        patch_data_sources,
        patch_single_region,
        monkeypatch,
    ) -> None:
        """#271 (adversarial-verify catch): exercise the REAL _score_region gate
        wiring end-to-end — _extract_holdout_metrics → gate_verdict_from_metrics →
        summary['gate'] → published gate_status map — not pre-baked verdict dicts.
        A loaded model with MAPE 5.0 (≤ 22 rollback) yields an acceptable verdict."""
        from models import persistence as mp

        fake_model = _fake_xgb_model()
        fake_meta = mp.ModelMetadata(
            region="ERCOT",
            model_name="xgboost",
            version="v-test",
            data_hash="h",
            trained_at="",
            train_rows=1,
            mape=5.0,  # → _extract_holdout_metrics {"mape": 5.0} → acceptable
            lib_versions={},
            extra={},
        )
        monkeypatch.setattr(
            "jobs.scoring_job.load_model",
            lambda region, model_name: (fake_model, fake_meta),
        )
        import models.xgboost_model as xgb_mod

        monkeypatch.setattr(xgb_mod, "predict_xgboost", lambda model, x: np.full(len(x), 41_000.0))
        import models.model_service as model_service

        monkeypatch.setattr(
            model_service,
            "get_forecasts",
            lambda region, df: {"ensemble": df["demand_mw"].values, "metrics": {}},
        )

        from jobs import scoring_job

        assert scoring_job.run() == 0

        gate = fake_redis["gridpulse:meta:gate_status"]
        assert gate["regions"]["ERCOT"] == {"acceptable": True, "best_mape": 5.0}

    def test_scoring_job_missing_model_still_writes_actuals(
        self,
        fake_redis,
        patch_data_sources,
        patch_single_region,
        monkeypatch,
    ) -> None:
        """No model in GCS → still writes actuals/weather/generation/alerts."""
        monkeypatch.setattr("jobs.scoring_job.load_model", lambda region, model_name: None)

        from jobs import scoring_job

        exit_code = scoring_job.run()
        # Non-model phases still succeed → exit 0.
        assert exit_code == 0

        # Actuals/weather/generation/alerts must still be present.
        for key in (
            "gridpulse:actuals:ERCOT",
            "gridpulse:weather:ERCOT",
            "gridpulse:generation:ERCOT",
            "gridpulse:alerts:ERCOT",
            "gridpulse:meta:last_scored",
        ):
            assert key in fake_redis

        # Forecast key must NOT be present when the model is missing.
        assert "gridpulse:forecast:ERCOT:1h" not in fake_redis

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
        assert fake_redis["gridpulse:meta:last_scored"]["regions_scored"] == 0
        assert "ERCOT" in fake_redis["gridpulse:meta:last_scored"]["regions_failed"]


class TestVintageCaptureIsWired:
    """#309 — the recorder must actually receive the frame the anchor is built
    from, and that frame must still carry ``forecast_mw``.

    Both halves fail *silently* if broken: a phase that never runs, or a frame
    that lost ``forecast_mw`` somewhere upstream, would leave the study quietly
    reading "no placeholders anywhere" — indistinguishable from a real finding
    of zero. That is the #131/#220 family, so it gets a test rather than trust.
    """

    def test_phase_receives_the_same_demand_frame_as_the_anchor(
        self, fake_redis, patch_data_sources, patch_single_region, monkeypatch
    ) -> None:
        from jobs import phases, scoring_job

        seen: dict[str, pd.DataFrame] = {}
        real = phases.write_vintage_records

        def _spy(region: str, demand_df):
            seen[region] = demand_df
            return real(region, demand_df)

        monkeypatch.setattr(phases, "write_vintage_records", _spy)
        scoring_job.run()

        assert "ERCOT" in seen, "write_vintage_records was never called by the scoring run"
        frame = seen["ERCOT"]
        assert "forecast_mw" in frame.columns, (
            "the demand frame reaching the vintage phase has no forecast_mw — the "
            "D == DF placeholder fingerprint would silently never fire in prod"
        )
        assert "demand_mw" in frame.columns

    def test_a_failing_capture_never_breaks_the_run(
        self, fake_redis, patch_data_sources, patch_single_region, monkeypatch
    ) -> None:
        """Capture is a measurement, not a critical path (the drift contract)."""
        from jobs import phases, scoring_job

        monkeypatch.setattr(
            phases,
            "write_vintage_records",
            lambda region, demand_df: (_ for _ in ()).throw(RuntimeError("redis exploded")),
        )
        with pytest.raises(RuntimeError):
            # Guard the guard: prove the injected failure is reachable at all,
            # so the assertion below can't pass because the phase was skipped.
            phases.write_vintage_records("ERCOT", None)

        # The run itself must still survive it via the phase's own try/except.
        monkeypatch.setattr(
            phases,
            "write_vintage_records",
            lambda region, demand_df: phases.PhaseResult(
                region=region, ok=False, error="redis exploded"
            ),
        )
        assert scoring_job.run() == 0


def _vintage_frame(d: float) -> pd.DataFrame:
    from datetime import UTC, datetime, timedelta

    hour = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    return pd.DataFrame({"timestamp": [hour], "demand_mw": [d], "forecast_mw": [7911.0]})


def _wire_vintage_redis(monkeypatch, store: dict, *, fail_reads: int = 0, configured: bool = True):
    """Route the vintage phase's strict-read/persist through an in-memory dict.

    ``fail_reads`` injects that many RedisReadErrors before reads succeed —
    the #313 anomaly, reproducible. Returns a counters dict.
    """
    import data.redis_client as rc

    calls = {"reads": 0, "writes": 0}

    def _strict(key):
        calls["reads"] += 1
        if calls["reads"] <= fail_reads:
            raise rc.RedisReadError("injected transient failure")
        return store.get(key)

    def _persist(key, value, ttl=86400):
        calls["writes"] += 1
        store[key] = value

    monkeypatch.setattr(rc, "redis_configured", lambda: configured)
    monkeypatch.setattr(rc, "redis_get_strict", _strict)
    monkeypatch.setattr(rc, "persist", _persist)
    monkeypatch.setattr("jobs.phases.time.sleep", lambda s: None)  # skip retry pause
    return calls


class TestVintageResetDefense:
    """#313 — prod re-pinned four regions' first-sight windows via unexplained
    nil reads (no error logged, no eviction, no TTL expiry, single execution).
    The trigger is unidentified; the defense must make the corruption
    impossible anyway: never rebuild history from an ambiguous read, and make
    every legitimate seed loud.
    """

    DATA = "gridpulse:vintage:AZPS"
    SEED = "gridpulse:vintage_seeded:AZPS"

    def test_two_ticks_capture_a_revision_end_to_end(self, monkeypatch):
        """The end-to-end shape on a recent frame: AZPS 1157 first seen,
        revised to 7815 four minutes later (the real prod case)."""
        from jobs import phases

        store: dict = {}
        _wire_vintage_redis(monkeypatch, store)

        assert phases.write_vintage_records("AZPS", _vintage_frame(1157.0)).ok
        assert store[self.DATA]["records"][0]["d"] == 1157.0
        assert self.SEED in store, "seeding must write the tombstone"

        assert phases.write_vintage_records("AZPS", _vintage_frame(7815.0)).ok
        row = store[self.DATA]["records"][0]
        assert row["d"] == 1157.0, "first_seen_d was overwritten — the study is dead"
        assert row["ld"] == 7815.0
        assert row["n"] == 1
        assert store[self.DATA]["mean_revision_pct"] == pytest.approx(85.2, abs=0.1)

    def test_window_absent_with_tombstone_refuses_to_repin(self, monkeypatch):
        """THE #313 test. Data key gone, tombstone alive → this is the anomaly,
        not a first run. Writing here re-pins 720 hours of first-sight history;
        the phase must refuse and fail loudly instead."""
        from jobs import phases

        store: dict = {self.SEED: {"last_write": "2026-07-16T11:00:00+00:00", "n_records": 719}}
        calls = _wire_vintage_redis(monkeypatch, store)

        result = phases.write_vintage_records("AZPS", _vintage_frame(8000.0))

        assert result.ok is False
        assert "refusing to re-pin" in (result.error or "")
        assert self.DATA not in store, "the defense wrote anyway — corruption shipped"
        assert calls["writes"] == 0

    def test_read_failure_never_writes(self, monkeypatch):
        """An infrastructure failure must fail the phase, not masquerade as an
        empty past. Both read attempts fail → no write of any kind."""
        from jobs import phases

        store: dict = {}
        calls = _wire_vintage_redis(monkeypatch, store, fail_reads=99)

        result = phases.write_vintage_records("AZPS", _vintage_frame(8000.0))

        assert result.ok is False
        assert "history read failed" in (result.error or "")
        assert calls["writes"] == 0
        assert store == {}

    def test_transient_read_failure_retries_and_preserves_first_seen(self, monkeypatch):
        """One failed read then success → the accumulated window survives."""
        from jobs import phases

        store: dict = {}
        _wire_vintage_redis(monkeypatch, store)
        assert phases.write_vintage_records("AZPS", _vintage_frame(1157.0)).ok

        _wire_vintage_redis(monkeypatch, store, fail_reads=1)
        assert phases.write_vintage_records("AZPS", _vintage_frame(7815.0)).ok
        assert store[self.DATA]["records"][0]["d"] == 1157.0

    def test_true_first_run_seeds_and_plants_tombstone(self, monkeypatch):
        """Data absent AND tombstone absent = genuine first run: proceed."""
        from jobs import phases

        store: dict = {}
        _wire_vintage_redis(monkeypatch, store)

        result = phases.write_vintage_records("AZPS", _vintage_frame(8000.0))

        assert result.ok is True
        assert self.DATA in store
        assert store[self.SEED]["n_records"] == 1

    def test_unusable_payload_refuses_to_overwrite(self, monkeypatch):
        """A readable-but-record-less payload is a failure, not a blank slate —
        this phase never writes one, so overwriting would destroy something."""
        from jobs import phases

        store: dict = {self.DATA: {"region": "AZPS"}}  # no records key
        calls = _wire_vintage_redis(monkeypatch, store)

        result = phases.write_vintage_records("AZPS", _vintage_frame(8000.0))

        assert result.ok is False
        assert store[self.DATA] == {"region": "AZPS"}, "unusable payload was clobbered"
        assert calls["writes"] == 0

    def test_unconfigured_redis_skips_quietly_without_reads(self, monkeypatch):
        """Dev/offline: nothing to protect, nowhere to write, no noise."""
        from jobs import phases

        calls = _wire_vintage_redis(monkeypatch, {}, configured=False)

        result = phases.write_vintage_records("AZPS", _vintage_frame(8000.0))

        assert result.ok is True
        assert result.details.get("skipped") == "redis_not_configured"
        assert calls["reads"] == 0

    def test_dropped_write_fails_the_phase(self, monkeypatch):
        """persist (#268) raising must surface as a failed phase — the silent
        redis_set bool this phase previously ignored."""
        import data.redis_client as rc
        from jobs import phases

        store: dict = {}
        _wire_vintage_redis(monkeypatch, store)

        def _exploding_persist(key, value, ttl=86400):
            raise rc.RedisWriteError("write dropped")

        monkeypatch.setattr(rc, "persist", _exploding_persist)

        result = phases.write_vintage_records("AZPS", _vintage_frame(8000.0))
        assert result.ok is False


class TestQualityGuardOrdering:
    """#309 PR 2 — the guard's ordering invariant, end to end.

    Vintage capture must see the RAW frame (it is the study of the artifacts);
    everything downstream — the actuals payload, drift, the anchor — must see
    the CLEANED frame. If the order ever flips, the vintage study silently
    loses its subject matter, which no unit test of either phase can notice.
    """

    def _run_with_partial_tail(self, monkeypatch, fake_redis):
        """Standard happy-path patches + an LDWP-style partial in the last row."""
        import data.eia_client as eia
        from jobs import phases, scoring_job

        seen: dict = {"vintage_demand": None, "order": []}

        real_vintage = phases.write_vintage_records

        def _vintage_spy(region, demand_df):
            seen["vintage_demand"] = demand_df["demand_mw"].tolist()
            seen["order"].append("vintage")
            return real_vintage(region, demand_df)

        real_guard = phases.apply_demand_quality_guard

        def _guard_spy(data):
            seen["order"].append("guard")
            return real_guard(data)

        monkeypatch.setattr(phases, "write_vintage_records", _vintage_spy)
        monkeypatch.setattr(phases, "apply_demand_quality_guard", _guard_spy)

        original_fetch = eia.fetch_demand

        def _partial_tail(region, **kwargs):
            df = original_fetch(region, **kwargs)
            df = df.copy()
            df.iloc[-1, df.columns.get_loc("demand_mw")] = 800.0  # ~2% of ~40k
            return df

        monkeypatch.setattr(eia, "fetch_demand", _partial_tail)
        scoring_job.run()
        return seen

    def test_vintage_sees_raw_payload_sees_cleaned(
        self, fake_redis, patch_data_sources, patch_single_region, monkeypatch
    ):
        seen = self._run_with_partial_tail(monkeypatch, fake_redis)

        # (a) ordering: vintage strictly before the guard
        assert seen["order"][:2] == ["vintage", "guard"], seen["order"]

        # (b) vintage captured the RAW partial
        assert seen["vintage_demand"] is not None
        assert seen["vintage_demand"][-1] == 800.0, "vintage saw a coerced frame — study corrupted"

        # (c) the actuals payload is CLEANED and DISCLOSES the exclusion
        payload = fake_redis["gridpulse:actuals:ERCOT"]
        assert np.isnan(payload["demand_mw"][-1]), "partial reached the tiles"
        exclusions = payload["artifact_excluded"]
        assert len(exclusions) == 1
        assert exclusions[0]["mw"] == 800.0
        assert exclusions[0]["reason"]

    def test_clean_tail_stamps_empty_disclosure(
        self, fake_redis, patch_data_sources, patch_single_region, monkeypatch
    ):
        """No artifacts → the field exists and is empty (never absent, so the
        web tier's .get() contract is uniform post-migration)."""
        from jobs import scoring_job

        scoring_job.run()
        payload = fake_redis["gridpulse:actuals:ERCOT"]
        assert payload["artifact_excluded"] == []


class TestScoringPartialFailureSemantics:
    """#267 — a run's exit code + freshness meta must reflect FORECAST outcomes,
    not 'any phase ran'. no_model (untrained) is expected, not a failure; a real
    forecast error is."""

    def _run(self, monkeypatch, outcomes):
        from jobs import phases, scoring_job

        regions = [o["region"] for o in outcomes]
        by_region = {o["region"]: o for o in outcomes}
        captured: dict = {}
        monkeypatch.setattr(phases, "ordered_regions", lambda default: regions)
        monkeypatch.setattr(scoring_job, "_score_region", lambda r: by_region[r])
        monkeypatch.setattr(
            phases, "write_meta", lambda key, extra=None: captured.setdefault(key, extra)
        )
        monkeypatch.setattr(scoring_job, "_check_runtime_headroom", lambda e: None)
        code = scoring_job.run()
        return code, captured.get("last_scored", {})

    def _fc(self, region, *, ok=False, error=None):
        ph = {"forecast": {"ok": ok} if ok else {"ok": False, "error": error}}
        return {"region": region, "ok": ok, "phases": ph}

    def test_all_no_model_exits_0_not_partial(self, monkeypatch):
        """Fresh deploy — every BA untrained. Nothing to retry; not a failure."""
        code, meta = self._run(
            monkeypatch, [self._fc("A", error="no_model"), self._fc("B", error="no_model")]
        )
        assert code == 0
        assert meta["partial_failure"] is False
        assert meta["regions_scored"] == 0
        assert meta["regions_errored"] == []

    def test_all_forecasts_errored_exits_1(self, monkeypatch):
        """Models exist but every forecast errored → total outage → retry."""
        code, meta = self._run(
            monkeypatch, [self._fc("A", error="boom"), self._fc("B", error="boom")]
        )
        assert code == 1
        assert meta["regions_errored"] == ["A", "B"]

    def test_partial_failure_alerts_but_exits_0(self, monkeypatch):
        """1 scored, 50 errored → below the floor with real errors → partial
        failure (visible + degrades health), but exits 0 (retry wouldn't help)."""
        outcomes = [self._fc("OK", ok=True)] + [self._fc(f"E{i}", error="boom") for i in range(50)]
        code, meta = self._run(monkeypatch, outcomes)
        assert code == 0
        assert meta["partial_failure"] is True
        assert meta["regions_scored"] == 1
        assert len(meta["regions_errored"]) == 50

    def test_forecast_ok_but_other_phase_failed_still_scored(self, monkeypatch):
        """The old bug in reverse: a good forecast counts even if alerts failed."""
        s = self._fc("OK", ok=True)
        s["phases"]["alerts"] = {"ok": False, "error": "noaa_down"}
        code, meta = self._run(monkeypatch, [s])
        assert code == 0
        assert meta["regions_scored"] == 1
        assert meta["partial_failure"] is False

    def test_gate_status_published_from_region_verdicts(self, monkeypatch):
        """#271 / P2-10: run() publishes gridpulse:meta:gate_status carrying each
        region's verdict. Regions with no metric signal (untrained) carry no
        verdict and are absent from the map — the web gate treats absent as
        warming (visible), so untrained BAs are never hidden."""
        from jobs import phases, scoring_job

        outcomes = [
            {
                "region": "PJM",
                "ok": True,
                "phases": {"forecast": {"ok": True}},
                "gate": {"acceptable": True, "best_mape": 3.2},
            },
            {
                "region": "CPLW",
                "ok": True,
                "phases": {"forecast": {"ok": True}},
                "gate": {"acceptable": False, "best_mape": 26.0},
            },
            {  # untrained — no "gate" key
                "region": "NEW",
                "ok": False,
                "phases": {"forecast": {"ok": False, "error": "no_model"}},
            },
        ]
        by_region = {o["region"]: o for o in outcomes}
        captured: dict = {}
        monkeypatch.setattr(
            phases, "ordered_regions", lambda default: [o["region"] for o in outcomes]
        )
        monkeypatch.setattr(scoring_job, "_score_region", lambda r: by_region[r])
        monkeypatch.setattr(
            phases, "write_meta", lambda key, extra=None: captured.setdefault(key, extra)
        )
        monkeypatch.setattr(scoring_job, "_check_runtime_headroom", lambda e: None)

        scoring_job.run()

        gate = captured.get("gate_status")
        assert gate is not None, "gate_status meta must be published"
        regions = gate["regions"]
        assert regions["PJM"] == {"acceptable": True, "best_mape": 3.2}
        assert regions["CPLW"] == {"acceptable": False, "best_mape": 26.0}
        assert "NEW" not in regions  # untrained → absent → warming/visible

    def test_degraded_run_with_no_verdicts_skips_gate_status_publish(self, monkeypatch):
        """#271 (adversarial-verify catch): a run where every region failed to
        load models produces NO verdicts. It must NOT publish an empty gate_status
        map — that would clobber the last-known good one on the same 24h key, and
        the web tier would read a present-but-empty map as 'every region warming
        -> visible', silently un-hiding rollback-grade BAs. Skip -> the prior map
        lives out its TTL."""
        from jobs import phases, scoring_job

        outcomes = [  # all no_model, none carries a "gate"
            {
                "region": "A",
                "ok": False,
                "phases": {"forecast": {"ok": False, "error": "no_model"}},
            },
            {
                "region": "B",
                "ok": False,
                "phases": {"forecast": {"ok": False, "error": "no_model"}},
            },
        ]
        by_region = {o["region"]: o for o in outcomes}
        captured: dict = {}
        monkeypatch.setattr(phases, "ordered_regions", lambda default: ["A", "B"])
        monkeypatch.setattr(scoring_job, "_score_region", lambda r: by_region[r])
        monkeypatch.setattr(
            phases, "write_meta", lambda key, extra=None: captured.setdefault(key, extra)
        )
        monkeypatch.setattr(scoring_job, "_check_runtime_headroom", lambda e: None)

        scoring_job.run()

        assert "last_scored" in captured  # run completed
        assert "gate_status" not in captured  # but did NOT clobber the gate map

    def test_partial_run_merges_verdicts_over_last_known(self, monkeypatch):
        """#271 (adversarial-verify catch): a run that re-scored only some regions
        merges over the last-known published map, PRESERVING a previously-hidden
        BA it didn't re-score this tick (rather than dropping it to visible)."""
        from jobs import phases, scoring_job

        outcomes = [
            {
                "region": "PJM",
                "ok": True,
                "phases": {"forecast": {"ok": True}},
                "gate": {"acceptable": True, "best_mape": 3.0},
            },
        ]
        by_region = {o["region"]: o for o in outcomes}
        captured: dict = {}
        monkeypatch.setattr(phases, "ordered_regions", lambda default: ["PJM"])
        monkeypatch.setattr(scoring_job, "_score_region", lambda r: by_region[r])
        monkeypatch.setattr(
            phases, "write_meta", lambda key, extra=None: captured.setdefault(key, extra)
        )
        monkeypatch.setattr(scoring_job, "_check_runtime_headroom", lambda e: None)
        # Last-known published map already hides CPLW (not re-scored this run).
        monkeypatch.setattr(
            "data.redis_client.redis_get",
            lambda key: {"regions": {"CPLW": {"acceptable": False, "best_mape": 30.0}}},
        )

        scoring_job.run()

        regions = captured["gate_status"]["regions"]
        assert regions["PJM"] == {"acceptable": True, "best_mape": 3.0}  # updated
        assert regions["CPLW"] == {"acceptable": False, "best_mape": 30.0}  # preserved


class TestSettledGradeDrift:
    """#304 endgame — the drift metric self-corrects as EIA revisions land.

    Two real ticks through write_drift_metrics: tick 1 scores a prediction
    against an LDWP-class partial (342% error on the books); tick 2's fetched
    frame carries the settled value for that hour, and the stored record must
    re-grade — the displayed aggregate collapses to the real error without
    any consumer change.
    """

    def _wire(self, monkeypatch, store: dict):
        import data.redis_client as rc

        # The window read is strict since #313's drift hardening.
        monkeypatch.setattr(rc, "redis_get_strict", lambda key: store.get(key))
        monkeypatch.setattr(rc, "redis_get", lambda key: store.get(key))
        monkeypatch.setattr(
            rc, "redis_set", lambda key, value, ttl=86400: store.__setitem__(key, value) or True
        )

    def _forecast(self, ts) -> dict:
        return {
            "region": "LDWP",
            "forecasts": [{"timestamp": ts.isoformat(), "ensemble": 4200.0}],
        }

    def _frame(self, hours: dict) -> pd.DataFrame:
        ts = sorted(hours)
        return pd.DataFrame({"timestamp": ts, "demand_mw": [hours[t] for t in ts]})

    def test_partial_scored_then_regraded_to_settled(self, monkeypatch):
        from datetime import UTC, datetime, timedelta

        from jobs import phases

        store: dict = {}
        self._wire(monkeypatch, store)
        h1 = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=2)
        h2 = h1 + timedelta(hours=1)

        # Tick 1: hour h1's actual arrives as a partial (950 vs true ~4840).
        res = phases.write_drift_metrics("LDWP", self._forecast(h1), self._frame({h1: 950.0}))
        assert res.ok
        block = store["gridpulse:drift:LDWP"]["models"]["ensemble"]
        assert block["records"][-1]["a"] == 950.0
        assert block["records"][-1]["e"] == pytest.approx(342.1, abs=0.5)

        # Tick 2: the fresh frame now carries h1's settled value.
        res = phases.write_drift_metrics(
            "LDWP", self._forecast(h2), self._frame({h1: 4840.0, h2: 4790.0})
        )
        assert res.ok
        block = store["gridpulse:drift:LDWP"]["models"]["ensemble"]
        rec_h1 = next(r for r in block["records"] if r["ts"].startswith(h1.isoformat()[:13]))
        assert rec_h1["a"] == 4840.0, "stored record was not re-graded against the settled value"
        assert rec_h1["e"] == pytest.approx(13.22, abs=0.05)
        # And the payload must never leak the ephemeral stats bag.
        assert "_regrade_stats" not in store["gridpulse:drift:LDWP"]

    def test_guard_excluded_hour_keeps_prior_value(self, monkeypatch):
        """If the fresh frame has NO value for a stored hour (guard-coerced
        NaN is dropped from the actuals map), the record must keep its prior
        actual — absence is unknown, never agreement."""
        from datetime import UTC, datetime, timedelta

        from jobs import phases

        store: dict = {}
        self._wire(monkeypatch, store)
        h1 = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=2)
        h2 = h1 + timedelta(hours=1)

        phases.write_drift_metrics("LDWP", self._forecast(h1), self._frame({h1: 950.0}))
        # Tick 2's frame carries h2 only — h1 became NaN (guard) and is absent.
        phases.write_drift_metrics("LDWP", self._forecast(h2), self._frame({h2: 4790.0}))
        block = store["gridpulse:drift:LDWP"]["models"]["ensemble"]
        rec_h1 = next(r for r in block["records"] if r["ts"].startswith(h1.isoformat()[:13]))
        assert rec_h1["a"] == 950.0


class TestDriftWindowStrictReads:
    """#313 defense-in-depth — the drift windows get the vintage treatment.

    Post-#318 these records carry re-graded history a fresh window cannot
    recompute, so a nil-read-during-outage must FAIL the phase, never rebuild.
    """

    def _wire_failing_reads(self, monkeypatch, store: dict):
        import data.redis_client as rc

        def _explode(key):
            raise rc.RedisReadError("injected outage")

        monkeypatch.setattr(rc, "redis_get_strict", _explode)
        monkeypatch.setattr(rc, "redis_get", lambda key: store.get(key))
        monkeypatch.setattr(
            rc, "redis_set", lambda key, value, ttl=86400: store.__setitem__(key, value) or True
        )
        monkeypatch.setattr("jobs.phases.time.sleep", lambda s: None)

    def _frame(self):
        from datetime import UTC, datetime, timedelta

        h = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
        return h, pd.DataFrame({"timestamp": [h], "demand_mw": [4000.0]})

    def test_drift_read_failure_fails_phase_without_writing(self, monkeypatch):
        from jobs import phases

        store: dict = {}
        self._wire_failing_reads(monkeypatch, store)
        h, frame = self._frame()
        forecast = {
            "region": "LDWP",
            "forecasts": [{"timestamp": h.isoformat(), "ensemble": 4100.0}],
        }

        result = phases.write_drift_metrics("LDWP", forecast, frame)

        assert result.ok is False
        assert "history read failed" in (result.error or "")
        assert "gridpulse:drift:LDWP" not in store, "window rebuilt during an outage"

    def test_horizon_drift_read_failure_fails_phase_without_writing(self, monkeypatch):
        from jobs import phases

        store: dict = {}
        self._wire_failing_reads(monkeypatch, store)
        h, frame = self._frame()

        result = phases.write_horizon_drift_metrics("LDWP", None, frame)

        assert result.ok is False
        assert "gridpulse:drift_horizon:LDWP" not in store

    def test_genuinely_absent_window_still_seeds(self, monkeypatch):
        """None from a healthy read = first run — the legitimate rebuild."""
        import data.redis_client as rc
        from jobs import phases

        store: dict = {}
        monkeypatch.setattr(rc, "redis_get_strict", lambda key: store.get(key))
        monkeypatch.setattr(rc, "redis_get", lambda key: store.get(key))
        monkeypatch.setattr(
            rc, "redis_set", lambda key, value, ttl=86400: store.__setitem__(key, value) or True
        )
        h, frame = self._frame()
        forecast = {
            "region": "LDWP",
            "forecasts": [{"timestamp": h.isoformat(), "ensemble": 4100.0}],
        }

        assert phases.write_drift_metrics("LDWP", forecast, frame).ok
        assert "gridpulse:drift:LDWP" in store
