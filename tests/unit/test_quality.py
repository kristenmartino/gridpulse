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


class TestIsRealPositive:
    """Promoted verbatim from _callbacks_us_grid — behavior identical."""

    def test_accepts_finite_positive(self):
        assert is_real_positive(730.0) is True
        assert is_real_positive(np.float64(1.5)) is True

    def test_rejects_the_rest(self):
        for bad in (None, "3300", b"3300", 0.0, -5.0, float("nan"), float("inf")):
            assert is_real_positive(bad) is False
