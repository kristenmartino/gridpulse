"""Unit tests for the forecast sub-phase timing helpers (#389 follow-up).

Covers ``jobs.phases.substep`` / ``jobs.phases.collect_substeps`` and the
``subtimings`` channel they feed through ``jobs.scoring_job._phase_rollup``.

The load-bearing property under test is the SEPARATION: sub-steps must never
reach ``summary["timings"]``, because ``_phase_rollup`` sums every key it finds
there and a sub-step summed alongside its own parent phase would double-count
against the 60.1% figure this instrumentation exists to refine.
"""

from __future__ import annotations

import threading
import time

from jobs.phases import collect_substeps, substep
from jobs.scoring_job import _phase_rollup


class TestSubstepRecording:
    def test_records_elapsed_under_its_name(self):
        with collect_substeps() as store, substep("alpha"):
            time.sleep(0.01)
        assert "alpha" in store
        assert store["alpha"] >= 0.01

    def test_accumulates_across_repeated_names(self):
        """A name used in a loop reports the TOTAL, not the last iteration.

        The per-model predict step relies on this only insofar as each model
        gets its own name; the horizon guard genuinely loops on one name.
        """
        with collect_substeps() as store:
            for _ in range(3):
                with substep("looped"):
                    time.sleep(0.005)
        assert store["looped"] >= 0.015

    def test_times_a_raising_substep(self):
        """A slow FAILING step must be as visible as a slow succeeding one —
        the same reason scoring_job.timed uses a finally."""
        with collect_substeps() as store:
            try:
                with substep("boom"):
                    time.sleep(0.01)
                    raise ValueError("expected")
            except ValueError:
                pass
        assert store["boom"] >= 0.01

    def test_noop_without_an_active_collector(self):
        """Every call site must be safe outside the scoring job — tests, the
        training job, the dev inline path."""
        with substep("orphan"):
            pass  # must not raise

    def test_nested_collectors_restore_the_outer_one(self):
        with collect_substeps() as outer:
            with substep("outer_step"):
                pass
            with collect_substeps() as inner, substep("inner_step"):
                pass
            with substep("after_nesting"):
                pass
        assert set(inner) == {"inner_step"}
        assert set(outer) == {"outer_step", "after_nesting"}


class TestThreadIsolation:
    def test_concurrent_collectors_do_not_interleave(self):
        """The scoring job runs PRECOMPUTE_MAX_WORKERS regions through one
        process, so a module-global dict would blend regions together."""
        seen: dict[str, dict] = {}
        barrier = threading.Barrier(4)

        def worker(name: str):
            with collect_substeps() as store:
                barrier.wait()  # force real overlap
                with substep(f"step_{name}"):
                    time.sleep(0.01)
            seen[name] = dict(store)

        threads = [threading.Thread(target=worker, args=(str(i),)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(seen) == 4
        for name, store in seen.items():
            assert list(store) == [f"step_{name}"], f"thread {name} saw another thread's steps"


class TestRollupSeparation:
    def _results(self):
        return [
            {
                "region": "CAISO",
                "timings": {"forecast": 60.0, "fetch": 13.0},
                "subtimings": {"predict_xgboost": 40.0, "future_frame": 15.0},
            },
            {
                "region": "SPA",
                "timings": {"forecast": 80.0, "fetch": 5.0},
                "subtimings": {"predict_xgboost": 70.0, "future_frame": 8.0},
            },
        ]

    def test_phase_rollup_ignores_subtimings(self):
        """Regression guard for the double-count hazard."""
        rollup = _phase_rollup(self._results())
        assert set(rollup) == {"forecast", "fetch"}
        assert rollup["forecast"]["total_s"] == 140.0

    def test_substep_rollup_folds_the_subtimings_key(self):
        rollup = _phase_rollup(self._results(), key="subtimings")
        assert set(rollup) == {"predict_xgboost", "future_frame"}
        assert rollup["predict_xgboost"]["total_s"] == 110.0
        assert rollup["predict_xgboost"]["max_s"] == 70.0
        assert rollup["predict_xgboost"]["slowest_region"] == "SPA"
        assert rollup["predict_xgboost"]["n"] == 2

    def test_substep_rollup_is_empty_when_nothing_recorded(self):
        results = [{"region": "X", "timings": {"forecast": 1.0}}]
        assert _phase_rollup(results, key="subtimings") == {}


class TestFetchSubsteps:
    """The `fetch` phase's own breakdown (#389).

    `fetch` is 13.0% of worker time and is the one phase whose total is
    dominated by an upstream we do not control. Sizing any weather-side change
    against it requires the EIA leg and the three Open-Meteo legs to be
    separable — otherwise EIA latency variance, which swung runs 660s→1800s on
    2026-08-04, swamps a ~1% signal.
    """

    def test_weather_legs_are_named_separately(self, monkeypatch):
        """One `fetch_weather` call records all three Open-Meteo legs."""
        import config as cfg
        import data.weather_client as wc

        monkeypatch.setitem(cfg.FEATURE_FLAGS, "nbm_weather", True)
        monkeypatch.setitem(cfg.FEATURE_FLAGS, "multipoint_weather", False)
        monkeypatch.setitem(cfg.FEATURE_FLAGS, "weather_archive_cache", False)

        import pandas as pd

        from config import WEATHER_VARIABLES

        def _frame():
            df = pd.DataFrame(
                {"timestamp": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")}
            )
            for v in WEATHER_VARIABLES:
                df[v] = 1.0
            return df

        cache = type("C", (), {"get": lambda *a, **k: None, "set": lambda *a, **k: None})()
        monkeypatch.setattr(wc, "get_cache", lambda: cache)
        monkeypatch.setattr(wc, "_fetch_forecast_endpoint", lambda *a, **k: _frame())
        monkeypatch.setattr(wc, "_fetch_archive_endpoint", lambda *a, **k: _frame())
        monkeypatch.setattr("data.gcs_store.write_parquet", lambda *a, **k: None)

        with collect_substeps() as store:
            wc.fetch_weather("ERCOT")

        assert set(store) == {"weather_forecast", "weather_nbm", "weather_archive"}

    def test_fetch_and_forecast_substeps_stay_in_separate_channels(self):
        """`fetch_subtimings` must not merge into `subtimings`.

        They are not comparable, and one flat dict would invite summing across
        two different parents — the same double-count trap the forecast
        sub-steps were deliberately kept out of ``timings`` to avoid.
        """
        results = [
            {
                "region": "ERCOT",
                "timings": {"fetch": 30.0, "forecast": 60.0},
                "subtimings": {"predict_xgboost": 40.0},
                "fetch_subtimings": {"weather_archive": 5.0, "eia_demand": 20.0},
            }
        ]
        assert set(_phase_rollup(results, key="subtimings")) == {"predict_xgboost"}
        fetch_roll = _phase_rollup(results, key="fetch_subtimings")
        assert set(fetch_roll) == {"weather_archive", "eia_demand"}
        assert fetch_roll["eia_demand"]["total_s"] == 20.0
        # and neither leaks into the phase-level rollup
        assert set(_phase_rollup(results)) == {"fetch", "forecast"}


class TestEiaPhaseSubsteps:
    """`generation` and `interchange` name their EIA call (#427).

    Those phases were 323.4s (10.8%) and 118.6s (4.0%) of worker time with no
    sub-steps at all, so nothing could say whether the cost was upstream
    latency or our own pivot/transform work. One named sub-step plus the phase
    total attributes the phase completely.
    """

    @staticmethod
    def _gen_frame():
        import pandas as pd

        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
                "fuel_type": ["NG", "WND", "SUN"],
                "generation_mw": [100.0, 50.0, 25.0],
            }
        )

    @staticmethod
    def _ix_frame():
        import pandas as pd

        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
                "to_ba": ["MISO", "PJM"],
                "interchange_mw": [-100.0, 50.0],
            }
        )

    def test_generation_names_its_eia_call(self, monkeypatch):
        import jobs.phases as ph

        monkeypatch.setattr(ph, "_has_eia_key", lambda: True)
        monkeypatch.setattr(
            "data.eia_client.fetch_generation_by_fuel", lambda *a, **k: self._gen_frame()
        )
        monkeypatch.setattr("data.redis_client.redis_set", lambda *a, **k: True)

        with collect_substeps() as store:
            ph.write_generation("ERCOT")

        assert "eia_generation" in store

    def test_interchange_names_its_eia_call(self, monkeypatch):
        import jobs.phases as ph

        monkeypatch.setattr(ph, "_has_eia_key", lambda: True)
        monkeypatch.setattr("data.eia_client.fetch_interchange", lambda *a, **k: self._ix_frame())
        monkeypatch.setattr("data.redis_client.redis_set", lambda *a, **k: True)

        with collect_substeps() as store:
            ph.write_interchange("ERCOT")

        assert "eia_interchange" in store

    def test_a_failing_eia_call_is_still_timed(self, monkeypatch):
        """A slow FAILING fetch is the one you most want to see — it paid the
        full retry budget and returned nothing. `substep` times in a finally."""
        import jobs.phases as ph

        monkeypatch.setattr(ph, "_has_eia_key", lambda: True)

        def _boom(*a, **k):
            raise RuntimeError("eia down")

        monkeypatch.setattr("data.eia_client.fetch_interchange", _boom)
        with collect_substeps() as store:
            res = ph.write_interchange("ERCOT")

        assert res.ok is False
        assert "eia_interchange" in store

    def test_each_phase_gets_its_own_collector(self):
        """The invariant that makes these readable: a phase's sub-steps belong
        to that phase alone, so each rolls up against its own total. A shared
        collector across phases would let `generation`'s legs be summed into
        `fetch`'s breakdown."""
        import jobs.scoring_job as sj

        results = [
            {
                "region": "ERCOT",
                "timings": {"fetch": 30.0, "generation": 20.0, "interchange": 5.0},
                "fetch_subtimings": {"eia_demand": 10.0, "weather_archive": 15.0},
                "generation_subtimings": {"eia_generation": 18.0},
                "interchange_subtimings": {"eia_interchange": 4.0},
            }
        ]
        assert set(sj._phase_rollup(results, key="fetch_subtimings")) == {
            "eia_demand",
            "weather_archive",
        }
        assert set(sj._phase_rollup(results, key="generation_subtimings")) == {"eia_generation"}
        assert set(sj._phase_rollup(results, key="interchange_subtimings")) == {"eia_interchange"}
        # and none of them leak into the phase-level rollup
        assert set(sj._phase_rollup(results)) == {"fetch", "generation", "interchange"}

    def test_substep_never_exceeds_its_phase_total(self):
        """The checkable property: `eia_generation` <= `phases.generation`.

        Pinned as an arithmetic contract so a future refactor that moves the
        fetch outside the timed phase shows up as a broken invariant rather
        than a quietly impossible number.
        """
        import jobs.scoring_job as sj

        results = [
            {
                "region": "R",
                "timings": {"generation": 20.0},
                "generation_subtimings": {"eia_generation": 18.0},
            }
        ]
        phase = sj._phase_rollup(results)["generation"]["total_s"]
        sub = sj._phase_rollup(results, key="generation_subtimings")["eia_generation"]["total_s"]
        assert sub <= phase
