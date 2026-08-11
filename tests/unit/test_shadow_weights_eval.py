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

import importlib.util
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location(
    "shadow_weights_eval",
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "shadow_weights_eval.py",
)
assert _spec and _spec.loader
sw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sw)


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
        pooled = sw._arm_stats(pooled_records, "served")
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
