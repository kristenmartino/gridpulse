"""Regression tests for the US Grid stress-display cap (V3.ζ follow-up).

User reported 2026-05-02: "Highest-Stress Region: CPLW · 1071%" and
"Lowest Reserve: -971%" on the deployed US Grid metrics bar. Same
class of bug for the per-region card's stress chip — CPLW would
display "1071%" on its own card.

Root cause: ``REGION_CAPACITY_MW["CPLW"] = 42`` (sourced from
EIA-860M Feb 2026, counts in-territory operating generators). CPLW
serves ~449 MW by importing nearly all its power from neighbors.
demand / capacity = 10.71× = 1071%. Math is correct, capacity figure
is meaningless for ranking purposes.

Fix:
- ``_STRESS_RELIABLE_CEILING = 2.0`` filters BAs with stress ratio
  above this from any ranking (their capacity figure is wrong).
- Metrics bar caps top-stress display at 100% and floors reserve at 0%.
- Region card swaps the stress chip for an "imports" qualitative
  chip when stress > ceiling.

Tests cover the user-reported scenario, edge cases at the threshold,
and the multi-BA case where one bad ratio doesn't affect others.
"""

from __future__ import annotations

from unittest.mock import patch


def _stress_chip(card_div):
    """Walk a region card and return the stress chip's className/text."""
    found = []

    def _walk(c):
        if hasattr(c, "className") and c.className:
            cls = c.className
            if isinstance(cls, str) and "gp-region-card__stress" in cls:
                text = getattr(c, "children", None)
                # Chip children may be ``[util-label span, value str]`` — the
                # test cares about the value, so pull the value string out.
                if isinstance(text, (list, tuple)):
                    text = next((x for x in reversed(text) if isinstance(x, str)), None)
                found.append((cls, text))
        children = getattr(c, "children", None)
        if isinstance(children, (list, tuple)):
            for ch in children:
                _walk(ch)
        elif children is not None and not isinstance(children, str):
            _walk(children)

    _walk(card_div)
    return found[0] if found else (None, None)


class TestMetricsBarStressCap:
    """The user-reported scenario — Top-Stress drove to 1071% and
    Lowest Reserve to -971% because CPLW's capacity is wrong for the
    ranking purpose."""

    def test_cplw_excluded_from_top_stress_ranking(self):
        """The 1071% BA must not win 'Highest-Stress'. The next-highest
        reliable BA wins instead."""
        from components.callbacks import _build_us_grid_metrics_items

        # CPLW: 449 / 42 = 10.71 (excluded). PJM: 70000 / 184202 ≈ 0.38 (highest reliable).
        region_data = {
            "CPLW": {"current_mw": 449.0, "today_mw": [449.0] * 24},
            "PJM": {"current_mw": 70000.0, "today_mw": [70000.0] * 24},
            "ERCOT": {"current_mw": 50000.0, "today_mw": [50000.0] * 24},
        }
        items = _build_us_grid_metrics_items(region_data)
        top = next(i for i in items if i["label"] == "Highest-Stress Region")
        # PJM @ 38% should win, not CPLW @ 1071%
        assert "PJM" in top["value"]
        assert "CPLW" not in top["value"]
        assert "1071" not in top["value"]
        assert "%" in top["value"]

    def test_lowest_reserve_floored_at_zero(self):
        """When the highest-stress BA is below 100%, reserve is the
        positive complement. Even if some BA edges over 100% it should
        floor at 0% — never the absurd -971% from the bug report."""
        from components.callbacks import _build_us_grid_metrics_items

        region_data = {
            "CPLW": {"current_mw": 449.0, "today_mw": [449.0] * 24},
            "PJM": {"current_mw": 70000.0, "today_mw": [70000.0] * 24},
        }
        items = _build_us_grid_metrics_items(region_data)
        reserve = next(i for i in items if i["label"] == "Lowest Reserve")
        assert "-" not in reserve["value"]
        assert reserve["value"] != "—"

    def test_displayed_stress_caps_at_100_percent(self):
        """Even within the reliable band (<200%), display caps at 100%
        so a tight-day 110% reading renders cleanly."""
        from components.callbacks import _build_us_grid_metrics_items

        # Inject a region with stress between 100% and ceiling
        with patch.dict(
            "components._callbacks_us_grid.REGION_CAPACITY_MW",
            {"PJM": 50000},  # 70000/50000 = 140% (reliable but over 100%)
            clear=False,
        ):
            region_data = {"PJM": {"current_mw": 70000.0, "today_mw": [70000.0] * 24}}
            items = _build_us_grid_metrics_items(region_data)
            top = next(i for i in items if i["label"] == "Highest-Stress Region")
            # Display should cap at 100% (no "140%" reading)
            assert "100" in top["value"]
            assert "140" not in top["value"]

    def test_all_unreliable_renders_placeholder(self):
        """If every populated region is import-dominated, there's no
        reliable stress ranking and the metric falls back to '—'.
        Using current_mw values chosen to push every ratio above the
        2.0 unreliable ceiling against the actual config capacities."""
        from components.callbacks import _build_us_grid_metrics_items

        region_data = {
            "CPLW": {"current_mw": 449.0, "today_mw": [449.0] * 24},  # 449/42 = 10.7×
            "HST": {"current_mw": 200.0, "today_mw": [200.0] * 24},  # 200/36 = 5.6×
            "GVL": {"current_mw": 2000.0, "today_mw": [2000.0] * 24},  # 2000/600 = 3.3×
        }
        items = _build_us_grid_metrics_items(region_data)
        top = next(i for i in items if i["label"] == "Highest-Stress Region")
        reserve = next(i for i in items if i["label"] == "Lowest Reserve")
        assert top["value"] == "—"
        assert reserve["value"] == "—"

    def test_total_demand_still_includes_unreliable_bas(self):
        """The Total Demand metric should NOT exclude import-dominated
        BAs — their demand reading is still real (just their capacity
        ratio isn't useful). Operators want the actual served-demand
        rollup, not a filtered view."""
        from components.callbacks import _build_us_grid_metrics_items

        region_data = {
            "CPLW": {"current_mw": 449.0, "today_mw": [449.0] * 24},
            "PJM": {"current_mw": 70000.0, "today_mw": [70000.0] * 24},
        }
        items = _build_us_grid_metrics_items(region_data)
        total = next(i for i in items if i["label"] == "Total Demand")
        # CPLW + PJM = 70449 MW = 70.4 GW
        assert total["value"] == "70.4"


class TestRegionCardStressChip:
    """The CPLW card itself would have shown '1071%' as a stress chip
    before the fix. After: shows a qualitative 'imports' chip with a
    hover tooltip explaining the data context."""

    def test_imports_chip_for_unreliable_capacity(self):
        from components.callbacks import _build_us_grid_region_card

        # CPLW with realistic served demand vs in-territory capacity
        with patch("components._callbacks_us_grid.REGION_NAMES", {"CPLW": "DEP-W (NC mtn)"}):
            card = _build_us_grid_region_card(
                "CPLW",
                {"current_mw": 449.0, "prev_mw": 445.0, "today_mw": [449.0] * 24},
            )
        cls, text = _stress_chip(card)
        assert cls is not None
        assert "imports" in cls
        assert text == "imports"

    def test_normal_chip_for_healthy_ba(self):
        """PJM has a sane capacity figure, so the chip should be a
        regular percentage."""
        from components.callbacks import _build_us_grid_region_card

        with patch("components._callbacks_us_grid.REGION_NAMES", {"PJM": "Mid-Atlantic (PJM)"}):
            card = _build_us_grid_region_card(
                "PJM",
                {"current_mw": 70000.0, "prev_mw": 69000.0, "today_mw": [70000.0] * 24},
            )
        cls, text = _stress_chip(card)
        assert cls is not None
        # 70000 / 184202 ≈ 38% → "low" tone
        assert "low" in cls
        assert "imports" not in cls
        assert text and "%" in text

    def test_card_chip_caps_at_100_percent_for_in_band_excess(self):
        """If a BA's stress is between 100% and the unreliable ceiling
        (200%) — e.g. tight-day reading — the chip caps the displayed
        number at 100% rather than showing 140%."""
        from components.callbacks import _build_us_grid_region_card

        with (
            patch.dict(
                "components._callbacks_us_grid.REGION_CAPACITY_MW",
                {"PJM": 50000},  # 140%
                clear=False,
            ),
            patch("components._callbacks_us_grid.REGION_NAMES", {"PJM": "PJM"}),
        ):
            card = _build_us_grid_region_card(
                "PJM",
                {"current_mw": 70000.0, "prev_mw": 69000.0, "today_mw": [70000.0] * 24},
            )
        cls, text = _stress_chip(card)
        assert text == "100%"


class TestStressCeilingThreshold:
    """The 200% ceiling is documented inline. These tests pin the
    boundary so the ceiling can't drift silently."""

    def test_ceiling_constant_value(self):
        from components.callbacks import _STRESS_RELIABLE_CEILING

        assert _STRESS_RELIABLE_CEILING == 2.0

    def test_just_below_ceiling_is_reliable(self):
        """199% (just below 200%) is treated as reliable — a region
        running this hot is highly stressed but the capacity figure
        is at least plausible."""
        from components.callbacks import _build_us_grid_metrics_items

        with patch.dict(
            "components._callbacks_us_grid.REGION_CAPACITY_MW", {"PJM": 35176}, clear=False
        ):  # 70000/35176 ≈ 199%
            region_data = {"PJM": {"current_mw": 70000.0, "today_mw": [70000.0] * 24}}
            items = _build_us_grid_metrics_items(region_data)
            top = next(i for i in items if i["label"] == "Highest-Stress Region")
            assert "PJM" in top["value"]  # included in ranking, not excluded

    def test_just_above_ceiling_is_unreliable(self):
        """201% triggers the unreliable-data filter."""
        from components.callbacks import _build_us_grid_metrics_items

        with patch.dict(
            "components._callbacks_us_grid.REGION_CAPACITY_MW", {"PJM": 34825}, clear=False
        ):  # 70000/34825 ≈ 201%
            region_data = {
                "PJM": {"current_mw": 70000.0, "today_mw": [70000.0] * 24},
                "ERCOT": {"current_mw": 50000.0, "today_mw": [50000.0] * 24},
            }
            items = _build_us_grid_metrics_items(region_data)
            top = next(i for i in items if i["label"] == "Highest-Stress Region")
            # PJM excluded; ERCOT (the next-highest reliable) wins
            assert "PJM" not in top["value"]
            assert "ERCOT" in top["value"]
