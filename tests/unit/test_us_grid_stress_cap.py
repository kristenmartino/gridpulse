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

    def test_national_utilization_renders_sane(self):
        """National Utilization (the 4th KPI) is a positive %, computed over
        the reliable-capacity BA set — import-dominated CPLW is excluded, so it
        can't produce the absurd -971% the old per-BA "Lowest Reserve" did."""
        from components.callbacks import _build_us_grid_metrics_items

        region_data = {
            "CPLW": {"current_mw": 449.0, "today_mw": [449.0] * 24},  # import-dominated → excluded
            "PJM": {"current_mw": 70000.0, "today_mw": [70000.0] * 24},  # 70000/184202 ≈ 38%
        }
        items = _build_us_grid_metrics_items(region_data)
        util = next(i for i in items if i["label"] == "National Utilization")
        assert "-" not in util["value"]
        assert util["value"] != "—"
        assert "%" in util["value"]

    def test_national_utilization_is_demand_over_capacity(self):
        """National Utilization = Σdemand ÷ Σcapacity over the reliable BA set."""
        from components.callbacks import _build_us_grid_metrics_items
        from config import REGION_CAPACITY_MW

        region_data = {
            "PJM": {"current_mw": 90000.0, "today_mw": [90000.0] * 24},
            "ERCOT": {"current_mw": 40000.0, "today_mw": [40000.0] * 24},
        }
        items = _build_us_grid_metrics_items(region_data)
        util = next(i for i in items if i["label"] == "National Utilization")
        expected = (90000 + 40000) / (REGION_CAPACITY_MW["PJM"] + REGION_CAPACITY_MW["ERCOT"]) * 100
        assert util["value"] == f"{expected:.0f}%"

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
        util = next(i for i in items if i["label"] == "National Utilization")
        # No reliable-capacity BA remains, so both the per-BA max and the
        # national aggregate fall back to the placeholder.
        assert top["value"] == "—"
        assert util["value"] == "—"

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


class TestDemandArtifactGuard:
    """#225: ``_is_implausible_demand_artifact`` — a near-zero glitch OR a
    single-hour collapse (>60% drop from the prior real reading, landing well
    below the day's median) is excluded from stress, while a gradual overnight
    trough and a return-from-spike are NOT flagged."""

    def test_near_zero_glitch_flagged(self):
        from components._callbacks_us_grid import _is_implausible_demand_artifact

        # 500 MW vs a ~6,000 MW day (< 10% of median).
        assert _is_implausible_demand_artifact(500.0, [6000.0] * 24, prev_mw=6100.0)

    def test_aps_single_step_collapse_flagged(self):
        """The reported case: APS 0.6 GW after 6.5 GW (−90.7%), day ~6 GW."""
        from components._callbacks_us_grid import _is_implausible_demand_artifact

        history = [6000.0] * 23 + [600.0]
        assert _is_implausible_demand_artifact(600.0, history, prev_mw=6500.0)

    def test_ladwp_moderate_collapse_flagged(self):
        """LADWP −68.6%: 0.9 GW after ~2.9 GW, day median ~3 GW. Above the 10%
        near-zero floor, so ONLY the step-collapse signal catches it."""
        from components._callbacks_us_grid import _is_implausible_demand_artifact

        history = [3000.0] * 23 + [900.0]
        assert _is_implausible_demand_artifact(900.0, history, prev_mw=2900.0)

    def test_gradual_overnight_trough_not_flagged(self):
        """A real trough descends over many hours — no single step halves the
        load — so a value at ~55% of median with a mild prior step is kept."""
        from components._callbacks_us_grid import _is_implausible_demand_artifact

        # median ~5,000; current 2,800 (56% of median) reached from 3,100 (a
        # 10% step, not a collapse).
        history = [6500, 6000, 5500, 5000, 4500, 4000, 3500, 3100, 2800] + [5000] * 15
        assert not _is_implausible_demand_artifact(2800.0, history, prev_mw=3100.0)

    def test_return_from_spike_not_flagged(self):
        """A sharp drop from an abnormal spike back to a normal level is not an
        artifact — the value lands at the day's median, so signal (2) requires
        it also be < 60% of median, which fails."""
        from components._callbacks_us_grid import _is_implausible_demand_artifact

        # prev spiked to 12,000; current returns to 6,000 = the day's median.
        history = [6000.0] * 23 + [12000.0]
        assert not _is_implausible_demand_artifact(6000.0, history, prev_mw=12000.0)

    def test_no_prev_falls_back_to_near_zero_only(self):
        """Without a prior reading, only the near-zero signal applies — a
        moderate low value is kept (can't confirm a collapse)."""
        from components._callbacks_us_grid import _is_implausible_demand_artifact

        assert not _is_implausible_demand_artifact(900.0, [3000.0] * 24)  # 30% of median, no prev
        assert _is_implausible_demand_artifact(200.0, [3000.0] * 24)  # < 10% of median


class TestPacwImportDominated:
    """#225: PACW (PacifiCorp West) nameplate ≈ served load, so it crowned the
    top-stress KPI at ~100%. It's now classified import-dominated."""

    def test_pacw_in_import_dominated_set(self):
        from config import IS_IMPORT_DOMINATED

        assert "PACW" in IS_IMPORT_DOMINATED

    def test_pacw_excluded_from_top_stress(self):
        from components.callbacks import _build_us_grid_metrics_items
        from config import REGION_CAPACITY_MW

        # PACW at ~99% of its nameplate vs a genuinely-lower PJM.
        region_data = {
            "PACW": {
                "current_mw": REGION_CAPACITY_MW["PACW"] * 0.99,
                "today_mw": [REGION_CAPACITY_MW["PACW"] * 0.95] * 24,
            },
            "PJM": {"current_mw": REGION_CAPACITY_MW["PJM"] * 0.60, "today_mw": [1.0] * 24},
        }
        items = _build_us_grid_metrics_items(region_data)
        top = next(i for i in items if i["label"] == "Highest-Stress Region")
        assert "PACW" not in top["value"]
        assert "PJM" in top["value"]
