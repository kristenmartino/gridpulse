"""#478: how the shadow evaluation weights BAs, and what it shows before deciding.

The arithmetic here is trivial. What is not trivial is *which* weighting the
headline uses, because #451's entire bias finding turned on that: its 12-BA cut
passed the ±2% constraint and its 51-BA cut failed it, from identical code. An
implicit weighting is a chosen conclusion.

Production coverage is uneven by construction — measured 2026-08-11, LDWP graded
0 records in 10 ticks while 46 of 51 BAs graded one per tick, because the #309
quality guard NaNs the broken-feed BAs' unreliable hours. Pooling records would
give those BAs no vote and the well-fed ones ten times the weight.
"""

from __future__ import annotations

import pytest

from models import shadow_eval as sw


def _ba(n: int, bias: float, mape: float = 5.0, days: float = 10.0) -> dict:
    stats = {"n": n, "bias_pct": bias, "mape": mape, "wape": mape}
    return {"n": n, "days": days, "served": dict(stats), "shadow": dict(stats)}


class TestPerBaWeighting:
    def test_each_ba_counts_once_regardless_of_record_count(self) -> None:
        """The whole point: a BA with 200 records does not outvote one with 2.

        Pooled, this fleet reads ≈ +0.06% (the big BA dominates). Per-BA it reads
        +5.0%, which is the honest statement that one of two BAs is badly biased.
        """
        per_region = {"BIG": _ba(200, 0.0), "SMALL": _ba(2, 10.0)}
        got = sw.fleet_stats(per_region, "served")
        assert got["n_bas"] == 2
        assert got["bias_pct"] == pytest.approx(5.0)
        assert got["n"] == 202  # total records still reported, just not as weight

    def test_it_disagrees_with_pooling_when_coverage_is_skewed(self) -> None:
        """Guard against the two collapsing into the same number by accident.

        If this ever stopped differing on skewed coverage, the per-BA figure
        would be decorative and the headline would silently be the pooled one.
        """
        per_region = {"BIG": _ba(200, 0.0), "SMALL": _ba(2, 10.0)}
        pooled_records = [{"actual": 100.0, "served_predicted": 100.0} for _ in range(200)] + [
            {"actual": 100.0, "served_predicted": 110.0} for _ in range(2)
        ]
        pooled = sw.arm_stats(pooled_records, "served")
        per_ba = sw.fleet_stats(per_region, "served")
        assert pooled["bias_pct"] < 1.0
        assert per_ba["bias_pct"] == pytest.approx(5.0)

    def test_a_ba_with_no_usable_pairs_is_excluded_not_counted_as_zero(self) -> None:
        """A BA that graded nothing has no bias — it must not vote 0.0%.

        Counting it as zero would pull the fleet figure toward the bound and make
        an unmeasured BA look like a well-behaved one. LDWP is exactly this case
        in production today.
        """
        per_region = {
            "OK": _ba(10, 4.0),
            "LDWP": {"n": 0, "days": 0.0, "served": None, "shadow": None},
        }
        got = sw.fleet_stats(per_region, "served")
        assert got["n_bas"] == 1
        assert got["bias_pct"] == pytest.approx(4.0)

    def test_no_usable_bas_returns_none_rather_than_a_number(self) -> None:
        assert sw.fleet_stats({"A": {"served": None}}, "served") is None
        assert sw.fleet_stats({}, "served") is None


class TestCoverageTable:
    def test_sparsest_first_because_the_tail_is_the_point(self) -> None:
        """A reader must meet the zero-coverage BAs before any average."""
        per_region = {"A": _ba(9, 1.0), "LDWP": _ba(0, 0.0), "B": _ba(4, 1.0)}
        rows = sw.coverage_rows(per_region)
        assert [r[0] for r in rows] == ["LDWP", "B", "A"]
        assert rows[0][1] == 0

    def test_every_region_appears(self) -> None:
        """Truncating the table in the report is a display choice; the data is not.

        `--json` consumers get the full list, so a silently dropped BA cannot
        hide in a summary.
        """
        per_region = {f"BA{i}": _ba(i, 0.0) for i in range(20)}
        assert len(sw.coverage_rows(per_region)) == 20


class TestThresholds:
    def test_the_bounds_are_451s_verbatim(self) -> None:
        """Re-chosen thresholds would make this a different experiment.

        #451 pre-registered |bias| <= 2.0% and MAPE regression <= 0.5 pts. If a
        future edit relaxes either to make a result pass, this test is where that
        has to be argued for.
        """
        assert sw.MAX_ABS_BIAS_PCT == 2.0
        assert sw.MAX_MAPE_REGRESSION_PTS == 0.5
        assert sw.MIN_DAYS_DEFAULT == 14


def _rec(ts: str, actual: float, served: float, shadow: float, lead: int | None = 1) -> dict:
    return {
        "timestamp": ts,
        "actual": actual,
        "served_predicted": served,
        "shadow_predicted": shadow,
        "lead_hours": lead,
    }


class TestQualityGate:
    """``compute_drift_payload`` filters before it averages; the shadow path did not.

    It reused the *grading* primitive (``build_records_from_actuals``) and neither
    filter, so the two paths graded identically and filtered differently. Measured
    on production 2026-08-18, served arm: per-BA bias +9.421% → +3.264% and pooled
    +3.656% → +2.412% once this gate is applied — almost all of it from 415 records
    whose known lead exceeded 1h, in a window whose name is "1-hour-ahead".

    **These tests pin the gate's behaviour, not a verdict.** The gate does NOT clear
    the ±2% bound: IID still reads +86.49% over 126 clean lead-1 records, against
    +1.65% from the drift path over the same window at a *longer* 24h lead. That
    residual is a defect in the shadow record stream, tracked separately — filtering
    is necessary here and is not sufficient.
    """

    def test_low_actual_artifacts_are_dropped(self) -> None:
        """The LDWP shape: a ~50 MW sentinel against a ~2500 MW median (#142)."""
        records = [_rec(f"2026-08-1{i}T00:00:00+00:00", 2500.0, 2550.0, 2540.0) for i in range(5)]
        records.append(_rec("2026-08-16T00:00:00+00:00", 50.0, 2550.0, 2540.0))
        kept, counts = sw.filter_records(records)
        assert counts["n_low_actual_dropped"] == 1
        assert all(r["actual"] > 100.0 for r in kept)

    def test_the_artifact_is_what_breaks_the_bias_bound(self) -> None:
        """Not a style fix: unfiltered this BA breaches ±2%, filtered it does not."""
        records = [_rec(f"2026-08-1{i}T00:00:00+00:00", 2500.0, 2525.0, 2525.0) for i in range(5)]
        records.append(_rec("2026-08-16T00:00:00+00:00", 50.0, 2525.0, 2525.0))

        unfiltered = sw.arm_stats(records, "served")
        assert abs(unfiltered["bias_pct"]) > sw.MAX_ABS_BIAS_PCT

        kept, _ = sw.filter_records(records)
        filtered = sw.arm_stats(kept, "served")
        assert abs(filtered["bias_pct"]) < sw.MAX_ABS_BIAS_PCT

    def test_known_high_lead_records_are_dropped(self) -> None:
        """Production carried leads out to 63h in a window labelled 1-hour-ahead."""
        records = [_rec(f"2026-08-1{i}T00:00:00+00:00", 1000.0, 1010.0, 1005.0) for i in range(4)]
        records.append(_rec("2026-08-15T00:00:00+00:00", 1000.0, 1400.0, 1400.0, lead=63))
        kept, counts = sw.filter_records(records)
        assert counts["n_lead_dropped"] == 1
        assert all(r.get("lead_hours") == 1 for r in kept)

    def test_unknown_lead_is_kept_matching_the_drift_rule(self) -> None:
        """``filter_by_lead`` keeps ``None`` deliberately; this must not diverge."""
        records = [_rec(f"2026-08-1{i}T00:00:00+00:00", 1000.0, 1010.0, 1005.0) for i in range(3)]
        records.append(_rec("2026-08-14T00:00:00+00:00", 1000.0, 1010.0, 1005.0, lead=None))
        kept, counts = sw.filter_records(records)
        assert counts["n_unknown_lead"] == 1
        assert len(kept) == 4

    def test_both_arms_keep_identical_hours(self) -> None:
        """The gate gets its safety from filtering the SHARED actual.

        A per-arm gate could keep different hours for each and quietly turn a
        weighting comparison into a coverage comparison.
        """
        records = [_rec(f"2026-08-1{i}T00:00:00+00:00", 2500.0, 2550.0, 9999.0) for i in range(5)]
        records.append(_rec("2026-08-16T00:00:00+00:00", 50.0, 2550.0, 9999.0))
        kept, _ = sw.filter_records(records)
        assert sw.arm_stats(kept, "served")["n"] == sw.arm_stats(kept, "shadow")["n"]

    def test_clean_windows_are_untouched(self) -> None:
        """The gate must not quietly shrink a healthy BA."""
        records = [_rec(f"2026-08-1{i}T00:00:00+00:00", 1000.0, 1010.0, 1005.0) for i in range(6)]
        kept, counts = sw.filter_records(records)
        assert len(kept) == 6
        assert counts["n_low_actual_dropped"] == 0
        assert counts["n_lead_dropped"] == 0

    def test_empty_input_is_not_an_error(self) -> None:
        kept, counts = sw.filter_records([])
        assert kept == []
        assert counts["n_in"] == 0
