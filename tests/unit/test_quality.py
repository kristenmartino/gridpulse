"""Demand-reading plausibility guard (#225 promoted, #309 extended).

The matrix fixtures are the REAL readings measured against the EIA API on
2026-07-15/16 — every threshold traces to a named case, and the deliberate
non-fires are as load-bearing as the fires:

    EXCLUDE: LDWP 730/967/554 partials; AZPS 1959 (with real AND stuck prev);
             IID stuck at 339 for 6+ hours; TIDC 0
    KEEP:    BPAT's D==DF stub 8825 (a GOOD anchor — removing stubs measured
             WORSE, 6.55% -> 7.72%); BPAT's +20% high partial 10564 (the
             documented residual — no low-side signal can catch it and a
             high-side one would false-flag real spikes); PSCO at 118% of its
             own day-ahead; PNM normal.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from data.quality import (
    coerce_demand_artifacts,
    is_implausible_demand_artifact,
    is_real_positive,
)

NOW = datetime(2026, 7, 16, 19, 0, 0, tzinfo=UTC)


def _frame(rows: list[tuple[float | None, float | None]]) -> pd.DataFrame:
    """``[(demand_mw, forecast_mw)]`` oldest-first -> a demand frame."""
    n = len(rows)
    return pd.DataFrame(
        {
            "timestamp": [NOW - timedelta(hours=n - 1 - i) for i in range(n)],
            "demand_mw": [d for d, _ in rows],
            "forecast_mw": [f for _, f in rows],
        }
    )


def _steady(mw: float, hours: int = 30, df_mw: float | None = None):
    return [(mw, df_mw if df_mw is not None else mw * 1.02)] * hours


class TestRealCaseMatrix:
    """Every named prod case, both directions."""

    # -- EXCLUDE ------------------------------------------------------------

    def test_ldwp_730_partial_excluded(self):
        """LDWP: 3,464 -> 730 (-79%, 22% of median) — the screenshot case."""
        assert is_implausible_demand_artifact(
            730.0, [3300.0] * 23 + [3464.0], prev_mw=3464.0, day_ahead_mw=3515.0
        )

    def test_ldwp_967_and_554_partials_excluded(self):
        assert is_implausible_demand_artifact(967.0, [3300.0] * 24, prev_mw=4463.0)
        assert is_implausible_demand_artifact(554.0, [3300.0] * 24, prev_mw=3280.0)

    def test_azps_partial_with_real_prev_excluded(self):
        """AZPS 7815 -> 1959: step-collapse signal."""
        assert is_implausible_demand_artifact(
            1959.0, [7000.0] * 24, prev_mw=7815.0, day_ahead_mw=7446.0
        )

    def test_azps_stuck_partial_needs_the_day_ahead_signal(self):
        """AZPS frozen at 1959 for 46 min: prev == current, so the step signal
        is blind; 28% of median clears the 10% floor. Only D/DF < 0.5 catches
        it — the measured justification for signal 3."""
        stuck_history = [7000.0] * 20 + [1959.0] * 4
        assert not is_implausible_demand_artifact(1959.0, stuck_history, prev_mw=1959.0)
        assert is_implausible_demand_artifact(
            1959.0, stuck_history, prev_mw=1959.0, day_ahead_mw=7446.0
        )

    def test_iid_stuck_339_needs_the_day_ahead_signal(self):
        """IID at 339 for 6+ hours: 34% of its ~1000 median, no step. Signal 3
        (33% of day-ahead) is the only catch."""
        history = [1000.0] * 18 + [339.0] * 6
        assert not is_implausible_demand_artifact(339.0, history, prev_mw=339.0)
        assert is_implausible_demand_artifact(339.0, history, prev_mw=339.0, day_ahead_mw=1031.0)

    def test_tidc_zero_excluded(self):
        assert is_implausible_demand_artifact(0.0, [800.0] * 24)

    # -- KEEP (each non-fire is a measured decision) --------------------------

    def test_bpat_stub_kept(self):
        """D == DF placeholder: ratio exactly 1.0 — signal 3 must NOT fire.
        The stub is a good anchor (fleet mean ~2.7% error; removing it
        measured worse, 6.55% -> 7.72%, winning 9/12 BAs)."""
        assert not is_implausible_demand_artifact(
            8825.0, [8400.0] * 24, prev_mw=8669.0, day_ahead_mw=8825.0
        )

    def test_bpat_high_partial_kept_the_documented_residual(self):
        """+20% high partial: deliberately uncatchable — a high-side signal
        would false-flag genuine demand spikes."""
        assert not is_implausible_demand_artifact(
            10564.0, [8400.0] * 24, prev_mw=8825.0, day_ahead_mw=8825.0
        )

    def test_psco_running_high_of_its_day_ahead_kept(self):
        """PSCO legitimately runs 118-121% of its own day-ahead — the reason
        signal 3 is low-side only."""
        assert not is_implausible_demand_artifact(
            7405.0, [6500.0] * 24, prev_mw=7398.0, day_ahead_mw=6301.0
        )

    def test_pnm_normal_kept(self):
        assert not is_implausible_demand_artifact(
            2233.0, [2300.0] * 24, prev_mw=2392.0, day_ahead_mw=2207.0
        )

    def test_gradual_overnight_trough_kept(self):
        """The #225 carve-out: real troughs descend over many hours. DF is
        hour-matched (the BA forecasts the trough too), so ratio ~0.97."""
        descending = [3000 - i * 80 for i in range(24)]
        assert not is_implausible_demand_artifact(
            1160.0, descending, prev_mw=1240.0, day_ahead_mw=1200.0
        )

    def test_bad_day_ahead_alone_cannot_exclude_a_normal_reading(self):
        """PSEI-class: the BA's own DF runs ~47% high on average. A real
        reading near its median must survive even at <50% of a doubly-wrong
        DF — the below-median co-signal blocks the bare ratio."""
        assert not is_implausible_demand_artifact(
            2000.0, [2100.0] * 24, prev_mw=2050.0, day_ahead_mw=4100.0
        )

    def test_missing_day_ahead_degrades_to_the_original_pair(self):
        """GCS-fallback frames may lack forecast_mw — signal 3 skips, the
        #225 signals still work."""
        assert is_implausible_demand_artifact(730.0, [3300.0] * 24, prev_mw=3464.0)
        assert not is_implausible_demand_artifact(
            1959.0, [7000.0] * 20 + [1959.0] * 4, prev_mw=1959.0, day_ahead_mw=None
        )


class TestCoerceDemandArtifacts:
    def test_ldwp_tail_coerced_with_disclosure(self):
        frame = _frame(_steady(3300.0, 29) + [(730.0, 3515.0)])
        cleaned, exclusions = coerce_demand_artifacts(frame)

        assert np.isnan(cleaned["demand_mw"].iloc[-1])
        assert len(exclusions) == 1
        assert exclusions[0]["mw"] == 730.0
        assert "day-ahead" in exclusions[0]["reason"] or "drop" in exclusions[0]["reason"]

    def test_input_frame_never_mutated(self):
        """The vintage study reads the raw frame — mutation would corrupt it."""
        frame = _frame(_steady(3300.0, 29) + [(730.0, 3515.0)])
        original = frame["demand_mw"].copy()
        coerce_demand_artifacts(frame)
        pd.testing.assert_series_equal(frame["demand_mw"], original)

    def test_stuck_run_caught_row_by_row(self):
        """IID: several consecutive stuck partials in the trailing window —
        each judged against its own context, all coerced."""
        frame = _frame(_steady(1000.0, 26, df_mw=1020.0) + [(339.0, 1031.0)] * 4)
        cleaned, exclusions = coerce_demand_artifacts(frame)

        assert len(exclusions) == 4
        assert cleaned["demand_mw"].tail(4).isna().all()
        assert cleaned["demand_mw"].iloc[0] == 1000.0

    def test_settled_history_outside_window_never_judged(self):
        """A deep-history dip (a real event that settled) is out of scope —
        only the trailing hours are candidates."""
        rows = _steady(3300.0, 10) + [(300.0, 3300.0)] + _steady(3300.0, 19)
        cleaned, exclusions = coerce_demand_artifacts(_frame(rows))

        assert exclusions == []
        assert cleaned["demand_mw"].iloc[10] == 300.0

    def test_excluded_value_not_used_as_context_for_later_rows(self):
        """After coercing hour N, hour N+1 must be judged against real history,
        not against the artifact."""
        # 730 partial, then a normal 3400 reading: if 730 stayed as prev,
        # 3400 would look like a +366% spike vs prev (harmless) — but the
        # inverse trap is a SECOND partial at 700 that is only ~-4% vs the
        # first artifact and would evade the step signal if 730 remained.
        frame = _frame(_steady(3300.0, 28) + [(730.0, 3515.0), (700.0, 3515.0)])
        cleaned, exclusions = coerce_demand_artifacts(frame)

        assert len(exclusions) == 2
        assert cleaned["demand_mw"].tail(2).isna().all()

    def test_all_nan_tail_is_absence_not_artifact(self):
        frame = _frame(_steady(3300.0, 28) + [(None, 3515.0), (None, 3515.0)])
        _, exclusions = coerce_demand_artifacts(frame)
        assert exclusions == []

    def test_a_gap_does_not_stop_the_scan(self):
        """An absent hour is SKIPPED, not treated as the end of the window.

        The loop's ``continue`` could be changed to ``break`` with all 2,687
        unit tests green (mutation testing, docs/TEST_QUALITY.md). Every
        existing case had its NaNs at the very end of the frame, where
        skipping and stopping look identical — so the one arrangement that
        distinguishes them was never exercised.

        It is not a hypothetical arrangement. EIA publishes late: a missing
        hour followed by a partial for the *next* hour is ordinary behaviour
        for the broken-feed BAs this guard exists for (LADWP passes through
        multiple partials per tick). Under ``break`` the scan would stop at
        the gap and every artifact after it would reach the forecast anchor
        unflagged — the exact failure #309 shipped this guard to prevent, and
        silent, because the exclusion list would simply come back empty.
        """
        frame = _frame(_steady(3300.0, 28) + [(None, 3515.0), (730.0, 3515.0)])
        cleaned, exclusions = coerce_demand_artifacts(frame)

        assert len(exclusions) == 1, "the artifact AFTER the gap must still be caught"
        assert exclusions[0]["mw"] == 730.0
        assert bool(pd.isna(cleaned["demand_mw"].iloc[-1]))

    def test_degenerate_frames_safe(self):
        assert coerce_demand_artifacts(None) == (None, [])
        df, exc = coerce_demand_artifacts(pd.DataFrame())
        assert exc == []
        df2, exc2 = coerce_demand_artifacts(pd.DataFrame({"nope": [1]}))
        assert exc2 == []

    def test_missing_forecast_column_tolerated(self):
        frame = _frame(_steady(3300.0, 29) + [(730.0, None)]).drop(columns=["forecast_mw"])
        cleaned, exclusions = coerce_demand_artifacts(frame)
        # step-collapse still catches it without the day-ahead column
        assert len(exclusions) == 1

    def test_clean_series_untouched(self):
        frame = _frame(_steady(3300.0, 30))
        cleaned, exclusions = coerce_demand_artifacts(frame)
        assert exclusions == []
        pd.testing.assert_frame_equal(cleaned, frame)

    # ------------------------------------------------------------------
    # Below: the series-level behaviour mutation testing found unpinned
    # (docs/TEST_QUALITY.md — 40 confirmed survivors here, the largest
    # cluster in the codebase).
    #
    # The cases above all use inputs that TWO signals catch, so they cannot
    # tell whether the third is wired up, and they assert on the exclusion
    # COUNT rather than on what the exclusion says. Both gaps matter: the
    # day-ahead signal is the only one that catches a stuck partial, and the
    # payload is rendered verbatim on the region tiles, in the operating
    # summary, and on /api/v1/grid/summary.
    # ------------------------------------------------------------------

    def test_the_day_ahead_signal_survives_the_plumbing(self):
        """A stuck partial that ONLY signal 3 can catch.

        Every other coercion test here uses a reading that signals 1 or 2
        already catch, so the whole ``forecast_mw`` path — reading the column,
        indexing it per row, passing it through as ``day_ahead_mw`` — could be
        cut out entirely and the suite would stay green. That is the exact
        shape of the #309 finding this signal was added for: IID frozen at 339
        is above the near-zero floor and has no step to detect, and only the
        day-ahead ratio sees it.

        340 against a 1000 MW median is 34% — well above the 10% near-zero
        floor. The prior reading is 800, so the drop is 58% and the step
        signal (which needs >60%) stays silent. Only 340 < 50% of the BA's own
        1000 MW day-ahead forecast marks it.
        """
        rows = _steady(1000.0, 24, df_mw=1000.0) + [(800.0, 1000.0), (340.0, 1000.0)]

        _, exclusions = coerce_demand_artifacts(_frame(rows))
        assert len(exclusions) == 1, "the day-ahead signal must reach the series function"
        assert "day-ahead" in exclusions[0]["reason"]

        # Same reading, no day-ahead column: nothing else can see it.
        _, without = coerce_demand_artifacts(_frame(rows).drop(columns=["forecast_mw"]))
        assert without == [], "proves signals 1 and 2 are blind here, so signal 3 did the work"

    def test_the_disclosure_payload_is_exact(self):
        """``ts``/``mw``/``reason`` are rendered verbatim, so pin all three.

        The existing tests assert a count and a substring, which leaves the
        timestamp unasserted entirely — it could be dropped to ``None`` and
        nothing would notice, putting "as of None" on the region tile.

        The reading carries decimals on purpose. With a whole number the
        2-decimal rounding is invisible, and ``round(x, 2)`` could become
        ``round(x, 3)`` unnoticed — real frames are float64 and do carry
        fractional MW.
        """
        rows = _steady(1000.0, 24, df_mw=1000.0) + [(800.0, 1000.0), (340.456, 1000.0)]

        _, exclusions = coerce_demand_artifacts(_frame(rows))

        assert exclusions == [
            {
                "ts": NOW.isoformat(),
                "mw": 340.46,
                "reason": "34% of the BA's own day-ahead forecast",
            }
        ]

    def test_the_near_zero_reason_reports_the_share_of_the_median(self):
        """Signal 1's disclosure names the median it was measured against.

        Reached by keeping the reading above 50% of its day-ahead forecast, so
        the day-ahead branch does not claim the reason first.
        """
        rows = _steady(1000.0, 25, df_mw=1000.0) + [(50.0, 60.0)]

        _, exclusions = coerce_demand_artifacts(_frame(rows))

        assert exclusions[0]["reason"] == "near-zero vs 24h median (5%)"

    def test_the_step_collapse_reason_reports_the_drop_and_the_level(self):
        """Signal 2's disclosure carries two numbers, and both are computed.

        ``drop`` is ``1 - current/prev`` — a sign flip or a swapped operator
        renders a plausible-looking percentage that is simply wrong, on a
        surface whose entire job is explaining an exclusion to an operator.
        """
        rows = _steady(1000.0, 25, df_mw=1000.0) + [(300.0, 500.0)]

        _, exclusions = coerce_demand_artifacts(_frame(rows))

        assert exclusions[0]["reason"] == "70% single-hour drop to 30% of the daily median"

    def test_a_frame_without_timestamps_still_discloses(self):
        """The ``ts`` fallback is a real path, not dead defensive code.

        GCS-fallback frames are rebuilt from parquet and do not always carry a
        ``timestamp`` column, so the ``str(ts)`` branch runs in production. It
        was entirely unexercised — the exclusion still has to be well-formed
        for ``/api/v1/grid/summary`` to serialise it.
        """
        frame = _frame(_steady(3300.0, 29) + [(730.0, 3515.0)]).drop(columns=["timestamp"])

        _, exclusions = coerce_demand_artifacts(frame)

        assert len(exclusions) == 1
        assert exclusions[0]["ts"] == "None", "no timestamp, but the field is still a string"
        assert exclusions[0]["mw"] == 730.0

    def test_the_reason_thresholds_are_exclusive_at_the_edge(self):
        """Which reason an operator is shown is decided by strict ``<``.

        Both selection thresholds could be relaxed to ``<=`` with the suite
        green, which mislabels an exclusion at exactly the boundary — the
        reading would still be excluded, but the disclosure would name the
        wrong signal, sending whoever reads it to the wrong upstream cause.
        """
        # Exactly 50% of the day-ahead forecast: NOT the day-ahead reason.
        rows = _steady(6000.0, 25, df_mw=6000.0) + [(500.0, 1000.0)]
        _, at_day_ahead_edge = coerce_demand_artifacts(_frame(rows))
        assert at_day_ahead_edge[0]["reason"] == "near-zero vs 24h median (8%)"

        # Exactly 10% of the 24h median: NOT the near-zero reason.
        rows = _steady(6000.0, 25, df_mw=6000.0) + [(600.0, 1000.0)]
        _, at_near_zero_edge = coerce_demand_artifacts(_frame(rows))
        assert at_near_zero_edge[0]["reason"] == "90% single-hour drop to 10% of the daily median"

    def test_prev_is_the_newest_real_reading_not_an_older_one(self):
        """The step signal compares against the LAST real hour.

        ``history[-1]`` mutated to ``history[-2]`` or ``history[+1]`` reaches
        back to the 1000 MW plateau instead of the 500 MW hour immediately
        before, turning a legitimate 50% decline into an apparent 75% collapse
        and excluding a real reading.

        250 after 500 is a 50% step — under the 60% threshold, so nothing
        should fire. Measured against 1000 it would look like 75% and would.
        """
        rows = _steady(1000.0, 24, df_mw=1000.0) + [(500.0, 1000.0), (250.0, 400.0)]

        _, exclusions = coerce_demand_artifacts(_frame(rows))

        assert exclusions == [], "a 50% decline from the previous real hour is not a collapse"


class TestIsRealPositive:
    """Promoted verbatim from _callbacks_us_grid — behavior identical."""

    def test_accepts_finite_positive(self):
        assert is_real_positive(730.0) is True
        assert is_real_positive(np.float64(1.5)) is True

    def test_rejects_the_rest(self):
        for bad in (None, "3300", b"3300", 0.0, -5.0, float("nan"), float("inf")):
            assert is_real_positive(bad) is False
