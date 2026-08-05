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
