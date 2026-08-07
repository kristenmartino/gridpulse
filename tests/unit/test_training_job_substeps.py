"""Sub-phase timing for the training job.

The scoring job got this in #405 and it immediately overturned three wrong
guesses about where its time went — one of them off by 16x, because a
single-threaded laptop cannot observe GIL contention across the concurrent
workers. The training job had no equivalent, so every statement about its cost
(including the "96% of fold cost is the fit" number behind #433) is still a
LOCAL measurement awaiting production confirmation.

Two properties are pinned here:

* The collector wraps **every** exit path, including the early returns for
  no-data and failed feature engineering. A region that dies in `fetch` is
  exactly the one whose timing you most want — under an upstream slowdown it is
  the slowest in the fleet — and instrumenting only the happy path hides it.
  This is the same reasoning that made `_log_region_complete` fire on both
  scoring exits.
* Sub-steps stay in their own channel, never in a dict something else sums.
"""

from __future__ import annotations

from unittest.mock import patch

from jobs.scoring_job import _phase_rollup
from jobs.training_job import _train_region


class TestCollectorWrapsEveryExitPath:
    def test_no_data_early_return_still_records_fetch(self):
        """The region that failed its fetch is the one whose timing matters
        most. Recording only on success would lose exactly that case."""
        with patch("jobs.phases.fetch_region_data", return_value=None):
            summary = _train_region("ERCOT")

        assert summary["error"] == "no_data"
        assert "fetch" in summary["subtimings"], (
            "a region that died in fetch must still report how long fetch took"
        )
        assert summary["subtimings"]["fetch"] >= 0

    def test_feature_failure_records_fetch_and_features(self):
        with (
            patch("jobs.phases.fetch_region_data", return_value=object()),
            patch("jobs.phases.apply_demand_quality_guard"),
            patch("jobs.phases.engineer_region_features", return_value=None),
        ):
            summary = _train_region("ERCOT")

        assert summary["error"] == "feature_engineering_failed"
        assert {"fetch", "quality_guard", "features"} <= set(summary["subtimings"])


class TestChannelSeparation:
    def test_subtimings_are_not_summed_as_phases(self):
        """Same hazard the scoring job's sub-steps were kept out of `timings`
        to avoid: a sub-step summed alongside its own parent double-counts."""
        results = [
            {
                "region": "ERCOT",
                "subtimings": {"fit_xgboost": 40.0, "backtests": 120.0},
            },
            {
                "region": "CAISO",
                "subtimings": {"fit_xgboost": 30.0, "backtests": 100.0},
            },
        ]
        rollup = _phase_rollup(results, key="subtimings")
        assert set(rollup) == {"fit_xgboost", "backtests"}
        assert rollup["backtests"]["total_s"] == 220.0
        assert rollup["backtests"]["slowest_region"] == "ERCOT"
        # and nothing leaked into the default channel
        assert _phase_rollup(results) == {}

    def test_rollup_is_empty_when_a_region_crashed_before_timing(self):
        """`run()` appends a bare dict for a crashed region; the rollup must
        tolerate it rather than raising inside the epilogue."""
        results = [{"region": "SPA", "ok": False, "error": "boom"}]
        assert _phase_rollup(results, key="subtimings") == {}
