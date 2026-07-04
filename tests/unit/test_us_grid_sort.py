"""Unit tests for US Grid card sorting (``_us_grid_card_sort_value``).

The US Grid Cards view offers a sort dropdown. "Region groups" (default)
keeps the geographic grouping; every other option flattens the grid and
ranks cards by the chosen key. These tests pin the ranking contract:

- demand / utilization / hourly-change rank high→low, name ranks A→Z
- BAs with missing data (warming pipeline), no prior hour, or unreliable
  capacity (import-dominated, peak×1.15 estimate) sink to the bottom
  rather than crowd the top.
"""

from __future__ import annotations

from components._callbacks_us_grid import _us_grid_card_sort_value as sort_value
from config import IS_IMPORT_DOMINATED, REGION_CAPACITY_MW


def _order(region_data: dict, sort: str) -> list[str]:
    return sorted(region_data, key=lambda c: sort_value(c, region_data[c], sort))


class TestUsGridSort:
    def test_demand_high_to_low(self):
        rd = {"A": {"current_mw": 1000}, "B": {"current_mw": 50000}, "C": {"current_mw": 20000}}
        assert _order(rd, "demand") == ["B", "C", "A"]

    def test_warming_ba_sinks_to_bottom_every_numeric_sort(self):
        rd = {"A": {"current_mw": 1000, "prev_mw": 900}, "WARM": {}}
        for sort in ("demand", "stress", "change"):
            assert _order(rd, sort)[-1] == "WARM", sort

    def test_change_ranks_biggest_movers_first(self):
        rd = {
            "FLAT": {"current_mw": 1000, "prev_mw": 1000},  # 0%
            "SPIKE": {"current_mw": 1200, "prev_mw": 1000},  # +20%
            "DROP": {"current_mw": 800, "prev_mw": 1000},  # -20%
        }
        ranked = _order(rd, "change")
        assert set(ranked[:2]) == {"SPIKE", "DROP"}  # both are 20% movers
        assert ranked[-1] == "FLAT"

    def test_no_prior_hour_sinks_in_change_sort(self):
        rd = {"MOVER": {"current_mw": 1200, "prev_mw": 1000}, "NEW": {"current_mw": 5000}}
        assert _order(rd, "change") == ["MOVER", "NEW"]

    def test_name_is_alphabetical(self):
        # Fake codes fall back to themselves (lowercased) for the sort key.
        rd = {"ZZZ": {"current_mw": 9}, "AAA": {"current_mw": 1}}
        assert _order(rd, "name") == ["AAA", "ZZZ"]

    def test_stress_excludes_import_dominated_to_bottom(self):
        imp = next(iter(IS_IMPORT_DOMINATED))
        normal = next(
            r
            for r in REGION_CAPACITY_MW
            if r not in IS_IMPORT_DOMINATED and REGION_CAPACITY_MW[r] > 0
        )
        rd = {
            # import-dominated BA at a *higher* raw ratio than the normal one
            imp: {"current_mw": REGION_CAPACITY_MW.get(imp, 1000) * 0.99},
            normal: {"current_mw": REGION_CAPACITY_MW[normal] * 0.50},
        }
        # Despite the higher ratio, the import-dominated BA sorts BELOW the
        # normal one — its capacity estimate is too noisy to rank on.
        assert _order(rd, "stress") == [normal, imp]
