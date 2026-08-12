"""#512: the drift key must outlive the history it stores.

`gridpulse:drift:{region}` held a 30-day rolling window under the generic 24h
snapshot TTL. The drift phase returns *before* persisting when there is no
matchable actual hour, so a BA that could not grade for 24 consecutive hours
never refreshed the TTL and lost its entire window, silently restarting from one
record.

AZPS did exactly that: a 25.0h gap on 2026-08-04 (inside the #389 incident), then
54 records against 720 for every other BA a week later. Nothing logged it — the
job is stateless and cannot tell "history lost" from "new BA".
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from jobs import phases


class TestTheKeyOutlivesItsContents:
    def test_the_drift_ttl_exceeds_the_window_it_stores(self) -> None:
        """The relationship, not the literal.

        Asserting `DRIFT_REDIS_TTL == 2764800` would pass just as happily if the
        window later grew to 60 days and the TTL stayed put — which is the bug,
        restated. This pins the invariant that actually matters.
        """
        window_seconds = phases.DRIFT_WINDOW_HOURS * 3600
        assert window_seconds < phases.DRIFT_REDIS_TTL, (
            f"TTL {phases.DRIFT_REDIS_TTL}s does not cover a {phases.DRIFT_WINDOW_HOURS}h window"
        )

    def test_there_is_real_margin_beyond_the_window(self) -> None:
        """A TTL equal to the window expires the oldest record at the instant it
        would still be counted. The margin is what absorbs a slow tick."""
        assert phases.DRIFT_TTL_MARGIN_HOURS >= 24

    def test_it_is_not_the_generic_snapshot_ttl(self) -> None:
        """The regression, named.

        `REDIS_TTL` is right for a forecast — a snapshot SHOULD go stale. It was
        wrong for a 30-day history, and reverting to it is the whole of #512.
        """
        assert phases.DRIFT_REDIS_TTL != phases.REDIS_TTL
        assert phases.DRIFT_REDIS_TTL > phases.REDIS_TTL

    def test_a_ba_grading_twice_a_day_survives_the_gaps_it_actually_has(self) -> None:
        """The measured worst case, as a test.

        LDWP graded 2 records in 24h on 2026-08-11 and logged a 22h gap the day
        before. Under the old TTL a 25h gap wiped the window; this asserts the
        new one tolerates a gap far larger than any observed.
        """
        observed_worst_gap_hours = 25.0
        assert observed_worst_gap_hours * 3600 * 10 < phases.DRIFT_REDIS_TTL


class TestTheWindowDepthIsObservable:
    def _previous_forecast(self, ts: str) -> dict:
        return {
            "region": "AZPS",
            "forecasts": [{"timestamp": ts, "ensemble": 100.0, "xgboost": 101.0}],
        }

    def test_n_records_is_logged_on_every_write(self) -> None:
        """A 720 -> 1 collapse must be visible within the hour.

        Without this the only witness to AZPS's loss was the public API, six days
        later, by accident. The job is stateless, so it cannot detect the drop
        itself — but it can report the depth and let a human or an alert see it.
        """
        ts = pd.Timestamp("2026-06-01T00:00:00Z")
        demand = pd.DataFrame({"timestamp": [ts], "demand_mw": [100.0]})

        with (
            patch("data.redis_client.redis_set"),
            patch("jobs.phases._read_window_strict", return_value=None),
            patch("jobs.phases.log") as recorder,
        ):
            phases.write_drift_metrics(
                region="AZPS",
                previous_forecast=self._previous_forecast(ts.isoformat()),
                demand_df=demand,
            )

        calls = [c for c in recorder.info.call_args_list if c.args[0] == "drift_updated"]
        assert calls, (
            f"expected drift_updated; got {[c.args[0] for c in recorder.info.call_args_list]}"
        )
        assert "n_records" in calls[0].kwargs
        assert calls[0].kwargs["n_records"] >= 1

    def test_the_write_uses_the_drift_ttl_not_the_generic_one(self) -> None:
        """Pins the wiring, not just the constant.

        Both TTLs are in scope at the call site; the constant being correct is
        worth nothing if the write passes the other one.
        """
        ts = pd.Timestamp("2026-06-01T00:00:00Z")
        demand = pd.DataFrame({"timestamp": [ts], "demand_mw": [100.0]})

        with (
            patch("data.redis_client.redis_set") as redis_set,
            patch("jobs.phases._read_window_strict", return_value=None),
        ):
            phases.write_drift_metrics(
                region="AZPS",
                previous_forecast=self._previous_forecast(ts.isoformat()),
                demand_df=demand,
            )

        drift_writes = [c for c in redis_set.call_args_list if c.args[0] == "gridpulse:drift:AZPS"]
        assert drift_writes, f"no drift write; saw {[c.args[0] for c in redis_set.call_args_list]}"
        assert drift_writes[0].kwargs["ttl"] == phases.DRIFT_REDIS_TTL


class TestTheSkipPathIsStillTheSkipPath:
    def test_no_matchable_hour_still_writes_nothing(self) -> None:
        """#512 is fixed by TTL, deliberately NOT by writing on the skip path.

        Re-persisting an unchanged payload every skipped tick would refresh the
        TTL too, but it costs a write per tick per starved BA and hides the
        cadence problem rather than surviving it. If someone later "fixes" it
        that way instead, this test is where the trade-off has to be re-argued.
        """
        ts = pd.Timestamp("2026-06-01T00:00:00Z")
        # An actual that matches no forecast row → nothing to grade.
        demand = pd.DataFrame({"timestamp": [ts], "demand_mw": [100.0]})
        previous = {
            "region": "AZPS",
            "forecasts": [{"timestamp": "2026-07-01T00:00:00+00:00", "ensemble": 100.0}],
        }

        with (
            patch("data.redis_client.redis_set") as redis_set,
            patch("jobs.phases._read_window_strict", return_value=None),
        ):
            result = phases.write_drift_metrics(
                region="AZPS", previous_forecast=previous, demand_df=demand
            )

        assert result.ok
        assert result.details.get("skipped") == "no_matchable_actual_hour"
        assert not [c for c in redis_set.call_args_list if c.args[0] == "gridpulse:drift:AZPS"]


class TestNoBackfill:
    def test_a_restarted_window_reports_its_true_shallow_depth(self) -> None:
        """#512's acceptance says do not backfill, and this is why it is safe to.

        A window that restarted reports the depth it really has. Padding it to
        look 30 days deep would make the rolling MAPE — which the Models tab and
        the visibility gate read — describe a history that does not exist.
        """
        ts = pd.Timestamp("2026-06-01T00:00:00Z")
        demand = pd.DataFrame({"timestamp": [ts], "demand_mw": [100.0]})

        with (
            patch("data.redis_client.redis_set") as redis_set,
            patch("jobs.phases._read_window_strict", return_value=None),
        ):
            phases.write_drift_metrics(
                region="AZPS",
                previous_forecast={
                    "region": "AZPS",
                    "forecasts": [{"timestamp": ts.isoformat(), "ensemble": 100.0}],
                },
                demand_df=demand,
            )

        payload = next(
            c.args[1] for c in redis_set.call_args_list if c.args[0] == "gridpulse:drift:AZPS"
        )
        assert payload["models"]["ensemble"]["n_records"] == 1
        assert np.isfinite(payload["models"]["ensemble"]["n_records"])
