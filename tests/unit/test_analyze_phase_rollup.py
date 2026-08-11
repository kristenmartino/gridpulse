"""Unit tests for ``scripts/analyze_phase_rollup.py``.

The script exists because the same analysis was done by hand four times in one
day and the arithmetic was wrong three of them. A tool that automates a
calculation nobody checked is only useful if ITS arithmetic is pinned — so
these tests target the exact mistakes that were actually made:

* the denominator taken from a different tick,
* a sub-step reported larger than the phase containing it,
* and a verdict declared off an arm with one observation.

The real payloads below are the 2026-08-07 ticks, values verbatim.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "analyze_phase_rollup",
    Path(__file__).resolve().parents[2] / "scripts" / "analyze_phase_rollup.py",
)
mod = importlib.util.module_from_spec(_SPEC)
# Register BEFORE exec: @dataclass resolves annotations via
# sys.modules[cls.__module__], which is None for an unregistered spec-loaded
# module and fails at class-creation time.
sys.modules["analyze_phase_rollup"] = mod
_SPEC.loader.exec_module(mod)  # type: ignore[union-attr]


PHASES_1307 = {
    "actuals_weather": {"total_s": 4.2, "n": 51},
    "alerts": {"total_s": 13.4, "n": 51},
    "anchor_conditioning": {"total_s": 0.3, "n": 51},
    "benchmark": {"total_s": 12.1, "n": 51},
    "diagnostics": {"total_s": 0.8, "n": 51},
    "drift": {"total_s": 19.5, "n": 51},
    "drift_horizon": {"total_s": 37.7, "n": 51},
    "features": {"total_s": 26.5, "n": 51},
    "fetch": {"total_s": 609.7, "n": 51},
    "forecast": {"total_s": 1638.9, "n": 51},
    "generation": {"total_s": 323.4, "n": 51},
    "interchange": {"total_s": 118.6, "n": 51},
    "model_load": {"total_s": 158.4, "n": 51},
    "quality_guard": {"total_s": 0.3, "n": 51},
    "read_existing_forecast": {"total_s": 1.0, "n": 51},
    "vintage": {"total_s": 13.1, "n": 51},
    "weather_correlation": {"total_s": 3.8, "n": 51},
}

FETCH_SUBSTEPS_1207 = {
    "eia_demand": {"total_s": 175.3, "n": 51, "slowest_region": "CAISO", "max_s": 9.2},
    "weather_archive": {"total_s": 294.0, "n": 48, "slowest_region": "NWMT", "max_s": 13.6},
    "weather_forecast": {"total_s": 146.0, "n": 51, "slowest_region": "ERCOT", "max_s": 16.1},
    "weather_nbm": {"total_s": 74.2, "n": 48, "slowest_region": "IPCO", "max_s": 4.0},
}


def _tick(ts="2026-08-07T13:07:20Z", elapsed=393.32, phases=None, substeps=None):
    t = mod.Tick(timestamp=ts, elapsed_s=elapsed, phases=phases or {})
    t.substeps = substeps or {}
    return t


class TestWorkerTimeDenominator:
    """The mistake that published `forecast` at 91% when it was ~55-58%.

    The first draft of this test asserted **57.7%** against the 13:07 payload
    and failed at 54.97%. 57.7% is the *2026-08-05T20:06* tick (1636.3/2834.8);
    the 13:07 tick is 1638.9/2981.7 = 55.0%. Two ticks, near-identical
    numerators, different denominators — and the test written to stop
    cross-tick mixing was itself written by mixing ticks. Left documented
    because it is the clearest possible evidence that this class of error does
    not announce itself: both numbers look right.
    """

    def test_summed_from_this_ticks_phases(self):
        assert _tick(phases=PHASES_1307).worker_s == pytest.approx(2981.7, abs=0.1)

    def test_forecast_share_is_55_for_this_tick_not_91(self):
        t = _tick(phases=PHASES_1307)
        share = PHASES_1307["forecast"]["total_s"] / t.worker_s * 100
        assert share == pytest.approx(55.0, abs=0.1)
        # 1638.9/1800 would be 91% — the shape of the number actually published.
        assert share < 60

    def test_parallelism_corroborates_the_total(self):
        """An independent check on the denominator: summed/wall must land near
        the worker count. A wrong denominator shows up here as an absurd
        parallelism figure, which is how the 91% could have been caught."""
        t = _tick(phases=PHASES_1307)
        assert t.worker_s / t.elapsed_s == pytest.approx(7.58, abs=0.05)


class TestNeverMixesTicks:
    def test_a_tick_without_phases_has_no_denominator_to_borrow(self):
        """A payload carrying only sub-steps must not acquire a denominator
        from anywhere. Mixing a 12:07 sub-step block with a 13:07 phases block
        is precisely the error this guards."""
        t = _tick(ts="2026-08-07T12:07:58Z", substeps={"fetch_substeps": FETCH_SUBSTEPS_1207})
        assert t.worker_s == 0.0
        assert t.leg("fetch_substeps", "weather_archive") == 294.0


class TestInvariant:
    def test_substep_exceeding_its_phase_is_reported(self, capsys):
        t = _tick(
            phases={"generation": {"total_s": 20.0, "n": 51}},
            substeps={"generation_substeps": {"eia_generation": {"total_s": 25.0, "n": 51}}},
        )
        problems = mod.report_attribution(t)
        assert problems and "EXCEEDS" in problems[0]

    def test_normal_attribution_reports_the_remainder(self, capsys):
        t = _tick(
            phases={"generation": {"total_s": 20.0, "n": 51}},
            substeps={"generation_substeps": {"eia_generation": {"total_s": 18.0, "n": 51}}},
        )
        assert mod.report_attribution(t) == []
        assert "2.0s" in capsys.readouterr().out  # 20.0 - 18.0, our own work


class TestArchiveArms:
    """The miss arm grows at ONE observation per UTC day. A full day of logs
    looks like plenty of data and contains a single miss."""

    @staticmethod
    def _day(miss_val=294.0, hit_val=40.0, n_hits=23):
        ticks = [
            _tick(
                ts="2026-08-08T00:07:00Z",
                substeps={"fetch_substeps": {"weather_archive": {"total_s": miss_val, "n": 48}}},
            )
        ]
        ticks += [
            _tick(
                ts=f"2026-08-08T{h:02d}:07:00Z",
                substeps={"fetch_substeps": {"weather_archive": {"total_s": hit_val, "n": 48}}},
            )
            for h in range(1, 1 + n_hits)
        ]
        return ticks

    def test_one_day_is_inconclusive_however_many_hits(self, capsys):
        mod.report_archive_arms(self._day())
        out = capsys.readouterr().out
        assert "INCONCLUSIVE" in out
        assert "n=1" in out

    def test_three_days_of_misses_stops_being_inconclusive(self, capsys):
        ticks = self._day(n_hits=3) + self._day(n_hits=3) + self._day(n_hits=3)
        mod.report_archive_arms(ticks)
        out = capsys.readouterr().out
        assert "INCONCLUSIVE" not in out

    def test_missing_instrumentation_is_named_not_read_as_zero(self, capsys):
        """An absent field must never be reported as a small number — the
        20:06 tick had no `fetch_substeps` at all because it predated the
        deploy, and reading that as 'the leg is cheap' was the live risk."""
        mod.report_archive_arms([_tick(phases=PHASES_1307)])
        assert "instrumentation deployed" in capsys.readouterr().out


class TestParsing:
    def test_value_format_infers_channel_from_leg_names(self):
        line = (
            "2026-08-07T12:07:58.499666Z\t448.1\t"
            "eia_demand={'max_s': 9.2, 'n': 51, 'slowest_region': 'CAISO', 'total_s': 175.3};"
            "weather_archive={'max_s': 13.6, 'n': 48, 'slowest_region': 'NWMT', 'total_s': 294}"
        )
        (t,) = mod.parse(line)
        assert t.elapsed_s == pytest.approx(448.1)
        assert t.leg("fetch_substeps", "weather_archive") == 294.0
        assert t.hour == 12

    def test_json_format(self):
        payload = (
            '[{"timestamp": "2026-08-08T00:07:00Z", "jsonPayload": '
            '{"event": "scoring_phase_rollup", "elapsed_s": 400.0, '
            '"phases": {"fetch": {"total_s": 100.0, "n": 51}}}}]'
        )
        (t,) = mod.parse(payload)
        assert t.worker_s == 100.0
        assert t.hour == 0  # the miss arm

    def test_empty_input_is_not_a_crash(self):
        assert mod.parse("   ") == []


class TestOutlierConfound:
    """The check this script did NOT have, added after it missed a real event.

    The 2026-08-10T13:19 tick took 2.8x the median elapsed with
    `eia_generation` at 4531.5s — a live EIA degradation — and the only
    confound check keyed on `n != 51`, which that tick passed with all 51.

    Excluding such ticks is the CONSERVATIVE direction: 13:19's
    `weather_archive` was 6.4s, the LOWEST hit-arm value in the window, so
    pooling it makes the cache look better than it is.
    """

    @staticmethod
    def _tick_at(ts, elapsed, archive):
        return _tick(
            ts=ts,
            elapsed=elapsed,
            phases={"fetch": {"total_s": 400.0, "n": 51}},
            substeps={"fetch_substeps": {"weather_archive": {"total_s": archive, "n": 51}}},
        )

    def _window(self):
        """21 normal ticks plus the real 13:19 outlier."""
        ticks = [self._tick_at(f"2026-08-10T{h:02d}:07:00Z", 400.0, 16.0) for h in range(1, 13)]
        ticks += [self._tick_at(f"2026-08-10T{h:02d}:07:00Z", 400.0, 16.0) for h in range(14, 23)]
        ticks.append(self._tick_at("2026-08-10T13:19:30Z", 1132.75, 6.4))
        return ticks

    def test_the_1319_tick_is_flagged(self):
        import analyze_phase_rollup as m

        ticks = self._window()
        flagged = m.flag_outlier_ticks(ticks)
        assert len(flagged) == 1
        (outlier,) = [t for t in ticks if id(t) in flagged]
        assert outlier.timestamp.startswith("2026-08-10T13:19")

    def test_a_flagged_tick_is_kept_out_of_the_arms(self, capsys):
        """And its exclusion is announced — never silent."""
        import analyze_phase_rollup as m

        ticks = self._window()
        m.report_archive_arms(ticks, excluded=m.flag_outlier_ticks(ticks))
        out = capsys.readouterr().out
        assert "excluded as upstream events" in out
        # 6.4 was the minimum; excluding it must not appear in the arm range
        assert "6.4" not in out

    def test_too_few_ticks_to_have_a_median_flags_nothing(self):
        """Two ticks cannot establish what 'normal' is — guessing there would
        invent a confound rather than find one."""
        import analyze_phase_rollup as m

        pair = [
            self._tick_at("2026-08-10T01:07:00Z", 400.0, 16.0),
            self._tick_at("2026-08-10T02:07:00Z", 1200.0, 6.0),
        ]
        assert m.flag_outlier_ticks(pair) == set()

    def test_a_normal_window_flags_nothing(self):
        import analyze_phase_rollup as m

        ticks = [self._tick_at(f"2026-08-10T{h:02d}:07:00Z", 400.0 + h, 16.0) for h in range(1, 13)]
        assert m.flag_outlier_ticks(ticks) == set()
