"""Integration tests for jobs/scoring_job.py.

All external I/O is faked:
- EIA + Open-Meteo fetchers are monkeypatched to return synthetic DataFrames.
- ``data.redis_client.redis_set`` is replaced with an in-memory dict writer.
- ``models.persistence.load_model`` is monkeypatched to return a tiny fake model.

The tests assert the scoring job writes the expected gridpulse:* keys and
returns a success exit code.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

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

    def test_real_score_region_escalates_a_live_horizon_disagreement(
        self,
        fake_redis,
        patch_data_sources,
        patch_single_region,
        monkeypatch,
    ) -> None:
        """#349: exercise the REAL producer, not just the pure verdict function.

        The sibling of this test (above) proved the gate verdict is published.
        It could not prove the serve-path second opinion is *attached*, because
        the harness starts with no drift history — so `live_horizon` was None
        and the wiring was never executed. That is precisely how the
        `_observed_lead_hours` bug shipped: consumer covered, producer not.

        Here the region's drift window is pre-seeded with rollback-grade 24h
        numbers while the holdout says 5.0% (comfortably inside the gate's 22%
        bar). The published entry must carry both verdicts and the flag.
        """
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
        import models.xgboost_model as xgb_mod

        monkeypatch.setattr(xgb_mod, "predict_xgboost", lambda model, x: np.full(len(x), 41_000.0))
        import models.model_service as model_service

        monkeypatch.setattr(
            model_service,
            "get_forecasts",
            lambda region, df: {"ensemble": df["demand_mw"].values, "metrics": {}},
        )

        # Freeze the drift window at rollback grade. Pinned AFTER the drift
        # phase would rewrite it, so the seed is what the gate actually reads.
        from jobs import phases

        seeded = {
            "models": {
                "ensemble": {"24h": {"rolling_mape_7d": 12.215, "n_7d": 160}},
                "arima": {"24h": {"rolling_mape_7d": 16.942, "n_7d": 160}},
            }
        }

        def _seed_drift(*a, **kw):
            return phases.PhaseResult(region="ERCOT", ok=True)

        monkeypatch.setattr(phases, "write_horizon_drift_metrics", _seed_drift)

        # The fake_redis fixture only replaces redis_set, so a real redis_get
        # would return None and the wiring would go untested again. Serve the
        # seed for the drift key ONLY — every other read keeps its existing
        # behavior, so this test cannot perturb the other 36.
        import data.redis_client as rc

        real_get = rc.redis_get
        monkeypatch.setattr(
            rc,
            "redis_get",
            lambda key, *a, **kw: (
                seeded if key == "gridpulse:drift_horizon:ERCOT" else real_get(key, *a, **kw)
            ),
        )

        from jobs import scoring_job

        assert scoring_job.run() == 0

        entry = fake_redis["gridpulse:meta:gate_status"]["regions"]["ERCOT"]
        # The generous question still passes — nothing gets hidden.
        assert entry["acceptable"] is True
        assert entry["best_mape"] == 5.0
        # ...and the sharp question is now published beside it, disagreeing.
        assert entry["disagrees"] is True
        assert entry["live_horizon"]["grade"] == "rollback"
        assert entry["live_horizon"]["champion"] == "ensemble"
        assert entry["live_horizon"]["champion_mape"] == 12.215
        assert entry["live_horizon"]["horizon"] == "24h"

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

        def _spy(region: str, demand_df, data=None):
            seen[region] = demand_df
            return real(region, demand_df, data)

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
            lambda region, demand_df, data=None: (_ for _ in ()).throw(
                RuntimeError("redis exploded")
            ),
        )
        with pytest.raises(RuntimeError):
            # Guard the guard: prove the injected failure is reachable at all,
            # so the assertion below can't pass because the phase was skipped.
            phases.write_vintage_records("ERCOT", None)

        # The run itself must still survive it via the phase's own try/except.
        monkeypatch.setattr(
            phases,
            "write_vintage_records",
            lambda region, demand_df, data=None: phases.PhaseResult(
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

        def _vintage_spy(region, demand_df, data=None):
            seen["vintage_demand"] = demand_df["demand_mw"].tolist()
            seen["order"].append("vintage")
            return real_vintage(region, demand_df, data)

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
        monkeypatch.setattr(scoring_job, "_score_region", lambda r, deadline=None: by_region[r])
        monkeypatch.setattr(
            phases, "write_meta", lambda key, extra=None: captured.setdefault(key, extra)
        )
        monkeypatch.setattr(scoring_job, "_check_runtime_headroom", lambda e, rollup=None: None)
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
        monkeypatch.setattr(scoring_job, "_score_region", lambda r, deadline=None: by_region[r])
        monkeypatch.setattr(
            phases, "write_meta", lambda key, extra=None: captured.setdefault(key, extra)
        )
        monkeypatch.setattr(scoring_job, "_check_runtime_headroom", lambda e, rollup=None: None)

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
        monkeypatch.setattr(scoring_job, "_score_region", lambda r, deadline=None: by_region[r])
        monkeypatch.setattr(
            phases, "write_meta", lambda key, extra=None: captured.setdefault(key, extra)
        )
        monkeypatch.setattr(scoring_job, "_check_runtime_headroom", lambda e, rollup=None: None)

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
        monkeypatch.setattr(scoring_job, "_score_region", lambda r, deadline=None: by_region[r])
        monkeypatch.setattr(
            phases, "write_meta", lambda key, extra=None: captured.setdefault(key, extra)
        )
        monkeypatch.setattr(scoring_job, "_check_runtime_headroom", lambda e, rollup=None: None)
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


class TestAnchorConditioning:
    """ADR-009 — broken-feed regions anchor on their own day-ahead forecast.

    The study's verdict (docs/ANCHOR_CONDITIONING_STUDY.md): broken-class
    anchors average 58.2% wrong vs DF's 14.5%. Only ``broken`` conditions;
    churn/bulk measured AGAINST substitution; the fork never touches the
    real frame the tiles/drift/alerts read.
    """

    def _region_data(self, monkeypatch, *, revision_class, flag_on=True):
        import data.redis_client as rc
        from jobs import phases

        monkeypatch.setattr(
            "config.FEATURE_FLAGS",
            {**__import__("config").FEATURE_FLAGS, "anchor_conditioning": flag_on},
        )
        summary = (
            {"revision_class": revision_class, "mean_fresh_revision_pct": 60.0}
            if revision_class
            else None
        )
        monkeypatch.setattr(rc, "redis_get", lambda key: summary)

        from datetime import UTC, datetime, timedelta

        hours = [
            datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=h)
            for h in range(29, -1, -1)
        ]
        frame = pd.DataFrame(
            {
                "timestamp": hours,
                "demand_mw": [3300.0] * 28 + [900.0, 850.0],  # trailing partials
                "forecast_mw": [3400.0] * 30,
                "region": "LDWP",
            }
        )
        return phases.RegionData(region="LDWP", demand_df=frame, weather_df=pd.DataFrame())

    def test_broken_class_conditions_the_fork_only(self, monkeypatch):
        from jobs import phases

        data = self._region_data(monkeypatch, revision_class="broken")
        res = phases.condition_anchor_frame(data)

        assert res.ok and res.details.get("conditioned") == 3
        # the fork carries DF at the trailing hours
        assert list(data.conditioned_demand_df["demand_mw"].tail(3)) == [3400.0] * 3
        # THE INVARIANT: the real frame is untouched
        assert list(data.demand_df["demand_mw"].tail(2)) == [900.0, 850.0]
        # and the anchor_frame property prefers the fork
        assert data.anchor_frame is data.conditioned_demand_df

    def test_non_qualifying_classes_do_not_condition(self, monkeypatch):
        from jobs import phases

        for cls in ("churn", "bulk", "clean", "unknown", None):
            data = self._region_data(monkeypatch, revision_class=cls)
            res = phases.condition_anchor_frame(data)
            assert res.ok
            assert data.conditioned_demand_df is None, f"{cls} conditioned — study says no"
            assert data.anchor_frame is data.demand_df

    def test_flag_off_is_a_noop_even_for_broken(self, monkeypatch):
        from jobs import phases

        data = self._region_data(monkeypatch, revision_class="broken", flag_on=False)
        res = phases.condition_anchor_frame(data)
        assert res.ok and res.details.get("skipped") == "flag_off"
        assert data.conditioned_demand_df is None

    def test_missing_df_values_left_alone(self, monkeypatch):
        from jobs import phases

        data = self._region_data(monkeypatch, revision_class="broken")
        data.demand_df.loc[data.demand_df.index[-1], "forecast_mw"] = float("nan")
        res = phases.condition_anchor_frame(data)

        assert res.ok and res.details.get("conditioned") == 2
        # the NaN-DF hour keeps its (bad, real) value rather than a fabrication
        assert data.conditioned_demand_df["demand_mw"].iloc[-1] == 850.0

    def test_failure_never_fatal(self, monkeypatch):
        import data.redis_client as rc
        from jobs import phases

        monkeypatch.setattr(
            "config.FEATURE_FLAGS",
            {**__import__("config").FEATURE_FLAGS, "anchor_conditioning": True},
        )
        monkeypatch.setattr(
            rc, "redis_get", lambda key: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        data = phases.RegionData(region="LDWP", demand_df=pd.DataFrame(), weather_df=pd.DataFrame())
        res = phases.condition_anchor_frame(data)
        assert res.ok is False  # reported, not raised


class TestBenchmarkWiring:
    """E0 — the public benchmark phase, end to end.

    The benchmark is *published* evidence about accuracy, so the wiring
    matters as much as the arithmetic: it reads three keys written earlier
    in the same tick, and if it ever runs before them it silently scores
    nothing while still reporting success.
    """

    @pytest.fixture
    def readable_redis(self, fake_redis, monkeypatch):
        """The shared fixture fakes writes only; the benchmark phase READS
        three keys written earlier in the same tick, so reads must resolve
        against the same store or the phase correctly skips as 'no_vintage'
        and this class would test nothing."""
        import data.redis_client as rc

        # The vintage phase writes via persist / reads via redis_get_strict,
        # so route those through the same store (existing helper), then make
        # plain reads resolve against it too — the benchmark reads three keys
        # written earlier in the tick.
        _wire_vintage_redis(monkeypatch, fake_redis)
        monkeypatch.setattr(rc, "redis_get", lambda key: fake_redis.get(key))
        return fake_redis

    def _run(self, monkeypatch):
        from jobs import phases, scoring_job

        order: list[str] = []
        for name in (
            "write_vintage_records",
            "write_horizon_drift_metrics",
            "write_benchmark_metrics",
        ):
            real = getattr(phases, name)

            def _spy(*args, _real=real, _name=name, **kwargs):
                order.append(_name)
                return _real(*args, **kwargs)

            monkeypatch.setattr(phases, name, _spy)
        scoring_job.run()
        return order

    def test_benchmark_runs_after_the_keys_it_reads(
        self, fake_redis, patch_data_sources, patch_single_region, monkeypatch
    ):
        """Vintage (which also writes vintage_summary) and horizon drift must
        both precede the benchmark — it reads all three."""
        order = self._run(monkeypatch)
        assert "write_benchmark_metrics" in order, "benchmark phase never ran"
        assert order.index("write_vintage_records") < order.index("write_benchmark_metrics")
        assert order.index("write_horizon_drift_metrics") < order.index("write_benchmark_metrics")

    def test_phase_writes_a_publishable_payload(self, monkeypatch):
        """Phase-level: given the three keys it reads, the benchmark writes a
        payload. Driven directly rather than through scoring_job.run(), whose
        shared harness does not exercise the vintage phase."""
        import data.redis_client as rc
        from jobs import phases

        hour = "2026-07-01T00:00:00+00:00"
        reads = {
            "gridpulse:vintage:ERCOT": {
                "records": [
                    {"ts": hour, "d": 40000.0, "at": hour, "ld": 40000.0, "n": 1, "df": 41000.0}
                ]
            },
            "gridpulse:vintage_summary:ERCOT": {
                "revision_class": "clean",
                "mean_fresh_revision_pct": 0.4,
            },
            "gridpulse:drift_horizon:ERCOT": {
                "models": {
                    "ensemble": {"24h": {"records": [{"ts": hour, "p": 40500.0, "a": 40000.0}]}}
                }
            },
        }
        writes: dict = {}
        monkeypatch.setattr(rc, "redis_get", lambda key: reads.get(key))
        monkeypatch.setattr(
            rc, "redis_set", lambda key, value, ttl=86400: writes.__setitem__(key, value)
        )

        result = phases.write_benchmark_metrics("ERCOT")

        assert result.ok
        payload = writes.get("gridpulse:benchmark:ERCOT")
        assert payload is not None, "benchmark key not written"
        assert payload["region"] == "ERCOT"
        # One hour is far below the sample floor, so the honest answer is a
        # published refusal to call it — never a verdict on n=1.
        assert payload["scoreable"] is False
        assert payload["reason"], "excluded without a published reason"

    def test_benchmark_failure_never_fails_the_run(
        self, fake_redis, patch_data_sources, patch_single_region, monkeypatch
    ):
        """Published evidence is secondary to serving forecasts."""
        import models.benchmark as bench
        from jobs import scoring_job

        def _boom(*args, **kwargs):
            raise RuntimeError("benchmark exploded")

        monkeypatch.setattr(bench, "compute_benchmark_payload", _boom)
        assert scoring_job.run() == 0


class TestScoringSoftDeadline:
    """The run must record the work it did, even when it runs out of time.

    2026-08-04: both killed executions had already scored ~49-51 of 51 BAs,
    and per-BA Redis writes are incremental so that data landed — but
    `write_meta("last_scored")` and the fleet rollup sit AFTER the fan-out, so
    neither run recorded any of it. `last_scored` stayed pinned at 16:22 until
    20:01, past the 90-minute /health staleness threshold, for work that had
    actually been done.
    """

    def _run(self, monkeypatch, regions, *, scored, fraction=0.85, flag=True):
        """Run with a deadline that has ALREADY passed for all but `scored`.

        The first `scored` BAs run normally; every later pickup sees a deadline
        in the past and sheds. Deterministic — no thread timing involved.
        """
        import config
        from jobs import phases, scoring_job

        monkeypatch.setattr(config, "SCORING_SOFT_DEADLINE_FRACTION", fraction)
        monkeypatch.setattr(config, "SCORING_TASK_TIMEOUT_S", 1800)
        monkeypatch.setitem(config.FEATURE_FLAGS, "soft_deadline", flag)
        monkeypatch.setattr(phases, "ordered_regions", lambda default: regions)

        seen: list[str] = []
        real = scoring_job._score_region

        def fake_score(region, deadline=None):
            seen.append(region)
            # Force the shed decision rather than race a wall clock: once
            # `scored` BAs have been handled, hand the real function a deadline
            # that is already in the past.
            if deadline is not None and len(seen) > scored:
                deadline = time.monotonic() - 1
            if deadline is not None and time.monotonic() >= deadline:
                return real(region, deadline)  # exercises the real shed branch
            return {"region": region, "ok": True, "phases": {"forecast": {"ok": True}}}

        captured: dict = {}
        monkeypatch.setattr(scoring_job, "_score_region", fake_score)
        monkeypatch.setattr(
            phases, "write_meta", lambda key, extra=None: captured.setdefault(key, extra)
        )
        monkeypatch.setattr(scoring_job, "_check_runtime_headroom", lambda e, rollup=None: None)
        code = scoring_job.run()
        return code, captured.get("last_scored", {})

    def test_shed_run_still_writes_last_scored(self, monkeypatch):
        """The whole point: reaching the epilogue at all."""
        code, meta = self._run(monkeypatch, [f"BA{i}" for i in range(10)], scored=6)

        assert meta, "a shedding run must still write the freshness meta"
        assert meta["regions_scored"] == 6
        assert meta["deadline_hit"] is True
        assert len(meta["regions_deadline_skipped"]) == 4
        assert code == 0

    def test_shed_regions_are_not_counted_as_errors(self, monkeypatch):
        """A shed BA never ran and still holds a live forecast in Redis."""
        _, meta = self._run(monkeypatch, [f"BA{i}" for i in range(10)], scored=6)

        assert meta["regions_errored"] == []

    def test_deadline_starvation_below_the_floor_still_alerts(self, monkeypatch):
        """The hole that excluding shed BAs from `errored` would open.

        Excluding them is right — they did not fail. But if that were the whole
        change, a run scoring 3/50 purely by deadline would have `errored == []`
        so `partial_failure` would be False, and /health would read OK over 47
        stale BAs. Silent, and worse than the timeout it replaces.
        """
        code, meta = self._run(monkeypatch, [f"BA{i}" for i in range(50)], scored=3)

        assert meta["regions_scored"] == 3
        assert meta["partial_failure"] is True, "deadline starvation must be visible"
        assert code == 0  # a Scheduler retry re-enters the same degraded upstream

    def test_shedding_emits_its_own_event_not_partial_failure(self, monkeypatch):
        """Different runbooks: one points at the model path, one at runtime.

        A mild shed — 45 of 51 scored, above the `SCORING_MIN_OK_REGIONS` floor
        of 40 — is a runtime problem only. Firing `scoring_partial_failure`
        here would page someone toward the forecast path for a healthy one.
        """
        from jobs import scoring_job

        fake_log = MagicMock()
        monkeypatch.setattr(scoring_job, "log", fake_log)
        self._run(monkeypatch, [f"BA{i}" for i in range(51)], scored=45)

        errors = [c.args[0] for c in fake_log.error.call_args_list]
        assert "scoring_deadline_shed" in errors
        assert "scoring_partial_failure" not in errors

    def test_flag_off_is_byte_identical(self, monkeypatch):
        """Rollback path: no deadline consulted, nothing shed, no new events."""
        from jobs import scoring_job

        fake_log = MagicMock()
        monkeypatch.setattr(scoring_job, "log", fake_log)
        code, meta = self._run(monkeypatch, [f"BA{i}" for i in range(10)], scored=6, flag=False)

        assert meta["regions_scored"] == 10
        assert meta["deadline_hit"] is False
        assert meta["regions_deadline_skipped"] == []
        assert "scoring_deadline_shed" not in [c.args[0] for c in fake_log.error.call_args_list]
        assert code == 0

    def test_fraction_zero_disables_shedding(self, monkeypatch):
        """Env-var kill switch, no image rebuild."""
        _, meta = self._run(monkeypatch, [f"BA{i}" for i in range(10)], scored=6, fraction=0.0)

        assert meta["regions_scored"] == 10
        assert meta["deadline_hit"] is False

    def test_shed_region_does_no_network_work(self, monkeypatch):
        """Shedding is only cheap if it skips the fetch — that is the point."""
        from jobs import phases, scoring_job

        called: list[str] = []
        monkeypatch.setattr(phases, "fetch_region_data", lambda r: called.append(r))

        summary = scoring_job._score_region("ERCOT", deadline=time.monotonic() - 1)

        assert called == []
        assert summary["skipped"] == "deadline"
        assert summary["phases"]["forecast"]["error"] == "deadline_skipped"
        assert summary["elapsed_s"] == 0.0

    def test_deadline_uses_a_monotonic_clock(self, monkeypatch):
        """A container wall-clock step must not extend or collapse the budget."""
        import config
        from jobs import scoring_job

        monkeypatch.setattr(config, "SCORING_SOFT_DEADLINE_FRACTION", 0.85)
        monkeypatch.setattr(config, "SCORING_TASK_TIMEOUT_S", 1800)
        monkeypatch.setitem(config.FEATURE_FLAGS, "soft_deadline", True)

        before = time.monotonic()
        deadline = scoring_job._soft_deadline()

        # Derived from monotonic(), not time.time() — which is orders of
        # magnitude larger (a Unix epoch) and would be obvious here.
        assert deadline is not None
        assert before <= deadline <= time.monotonic() + 1800 * 0.85


class TestAnchorProvenanceReachesRedis:
    """#547 end-to-end: the anchor a forecast was seeded with survives the real
    scoring run, from the vintage phase's verdicts to the forecast payload the
    drift phases read on the next tick.

    Unit tests pin each hop; this pins that the hops are connected — the
    vintage phase runs before the forecast phase, its map reaches
    ``RegionData``, and the block lands top-level rather than on a row.
    """

    @staticmethod
    def _patch_models(monkeypatch) -> None:
        """The same fake-model wiring the core write test uses, so the forecast
        phase actually runs instead of short-circuiting on ``model_missing``."""
        import models.model_service as model_service
        import models.xgboost_model as xgb_mod
        from models import persistence as mp

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
            lambda region, model_name: (_fake_xgb_model(), fake_meta),
        )
        monkeypatch.setattr(xgb_mod, "predict_xgboost", lambda model, x: np.full(len(x), 41_000.0))
        monkeypatch.setattr(
            model_service,
            "get_forecasts",
            lambda region, df: {"ensemble": df["demand_mw"].values, "metrics": {}},
        )

    def test_the_forecast_payload_carries_a_complete_anchor_block(
        self, fake_redis, patch_data_sources, patch_single_region, monkeypatch
    ) -> None:
        from jobs import scoring_job

        self._patch_models(monkeypatch)
        assert scoring_job.run() == 0

        payload = fake_redis["gridpulse:forecast:ERCOT:1h"]
        anchor = payload.get("anchor")
        assert anchor is not None, "the forecast phase wrote no anchor block"
        assert set(anchor) == {
            "anchor_ts",
            "anchor_mw",
            "anchor_was_placeholder",
            "anchor_conditioned",
        }

        # The anchor is the hour BEFORE row 0 — the hour resolved from, not the
        # forecast start. Asserted against the payload's own first row so it
        # cannot drift from what was actually served.
        first_row = pd.Timestamp(payload["forecasts"][0]["timestamp"])
        assert pd.Timestamp(anchor["anchor_ts"]) == first_row - pd.Timedelta(hours=1)

        assert anchor["anchor_mw"] is not None and anchor["anchor_mw"] > 0
        assert anchor["anchor_conditioned"] is False, "ERCOT is not a broken-class feed"

        # This harness leaves Redis "unconfigured", so the vintage phase skips
        # and never hands a map over. The anchor must then read UNKNOWN — the
        # tri-state degrading correctly through a whole real run, which is the
        # case a construct-and-read unit test cannot reach.
        assert anchor["anchor_was_placeholder"] is None

    def test_the_anchor_is_not_on_any_forecast_row(
        self, fake_redis, patch_data_sources, patch_single_region, monkeypatch
    ) -> None:
        """A per-row anchor would be read as a model by the drift extractor —
        acquiring its own drift records, a Models-tab entry and a place in the
        rolling MAPE the visibility gate reads."""
        from jobs import scoring_job

        self._patch_models(monkeypatch)
        assert scoring_job.run() == 0

        rows = fake_redis["gridpulse:forecast:ERCOT:1h"]["forecasts"]
        offenders = {k for row in rows for k in row if "anchor" in k}
        assert not offenders, f"anchor keys leaked onto forecast rows: {offenders}"

    def test_the_vintage_phase_and_the_forecast_phase_share_one_region_data(
        self, fake_redis, patch_data_sources, patch_single_region, monkeypatch
    ) -> None:
        """The wiring the whole instrument rests on.

        The hand-off is in-memory, so it only works if the object the vintage
        phase stashes its verdicts on is the same object the forecast phase
        later reads. If the scoring job stopped passing ``region_data``, every
        anchor would silently record ``anchor_was_placeholder=None`` forever —
        the field would look present while measuring nothing, which is exactly
        the shape of #542.

        Asserted on object identity rather than on a populated map: whether any
        hour lands inside the vintage window depends on the fixture's dates,
        and this claim should not.
        """
        from jobs import phases, scoring_job

        self._patch_models(monkeypatch)
        seen: dict = {}
        real_vintage = phases.write_vintage_records
        real_forecast = phases.predict_and_write_forecast

        def _vintage_spy(region, demand_df, data=None):
            seen["vintage_data"] = data
            return real_vintage(region, demand_df, data)

        def _forecast_spy(data, *args, **kwargs):
            seen["forecast_data"] = data
            return real_forecast(data, *args, **kwargs)

        monkeypatch.setattr(phases, "write_vintage_records", _vintage_spy)
        monkeypatch.setattr(phases, "predict_and_write_forecast", _forecast_spy)
        assert scoring_job.run() == 0

        assert seen.get("vintage_data") is not None, (
            "the scoring job no longer hands RegionData to the vintage phase"
        )
        assert seen["vintage_data"] is seen["forecast_data"], (
            "the two phases hold different RegionData objects — the in-memory hand-off cannot work"
        )
