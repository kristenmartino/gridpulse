"""#254 — peak-derived-capacity BAs are excluded from stress/utilization.

Background. Five BAs (SOCO, DUK, CPLE, PSCO, FMPP) carry
``REGION_CAPACITY_MW = 12-month peak × 1.15`` — a reserve-margin proxy, not a
measured EIA-860M nameplate — yet are NOT import-dominated, so before #254 they
flowed straight into ``national_utilization_pct`` and the Highest-Stress KPI.

Utilization = demand / (peak × 1.15) is **self-referential**: at its own
historical peak a BA reads exactly ~87% (1 / 1.15) and can never surface as
stressed above that, so the ratio is not a real stress signal. #254 excludes
these BAs the same way import-dominated ones are excluded, and the public API
labels their capacity ``capacity_source = "peak_estimate"`` instead of
"nameplate".

HST and CPLW are peak-derived too but are ALSO import-dominated (that's their
dominant story), so ``UNRELIABLE_CAPACITY`` is the union. SPA is the deliberate
exception — import-dominated but a *true* nameplate (federal dam fleet).
"""

from __future__ import annotations


def _find_text(node, target) -> bool:
    """Walk a Dash component tree for a text child containing ``target``."""
    children = getattr(node, "children", None)
    if children is None:
        return False
    if isinstance(children, str):
        return target in children
    if isinstance(children, (list, tuple)):
        return any(_find_text(c, target) for c in children)
    return _find_text(children, target)


class TestPeakDerivedSet:
    def test_set_is_a_frozenset_with_the_seven_peak_derived_bas(self):
        from config import PEAK_DERIVED_CAPACITY

        assert isinstance(PEAK_DERIVED_CAPACITY, frozenset)
        assert {
            "SOCO",
            "DUK",
            "CPLE",
            "PSCO",
            "FMPP",
            "HST",
            "CPLW",
        } == PEAK_DERIVED_CAPACITY

    def test_spa_is_not_peak_derived(self):
        """SPA is import-dominated but its 2,559 MW is a true federal-dam
        nameplate — it must NOT be tagged peak-derived."""
        from config import PEAK_DERIVED_CAPACITY

        assert "SPA" not in PEAK_DERIVED_CAPACITY

    def test_generation_rich_bas_are_not_peak_derived(self):
        from config import PEAK_DERIVED_CAPACITY

        for ba in ("PJM", "ERCOT", "CAISO", "MISO", "FPL"):
            assert ba not in PEAK_DERIVED_CAPACITY

    def test_all_peak_derived_bas_exist_in_capacity_dict(self):
        from config import PEAK_DERIVED_CAPACITY, REGION_CAPACITY_MW

        for ba in PEAK_DERIVED_CAPACITY:
            assert ba in REGION_CAPACITY_MW

    def test_unreliable_capacity_is_the_union(self):
        from config import (
            IS_IMPORT_DOMINATED,
            PEAK_DERIVED_CAPACITY,
            UNRELIABLE_CAPACITY,
        )

        assert UNRELIABLE_CAPACITY == IS_IMPORT_DOMINATED | PEAK_DERIVED_CAPACITY
        # SOCO: peak-derived only. CPLW: both. SPA: import-dominated only.
        assert "SOCO" in UNRELIABLE_CAPACITY
        assert "CPLW" in UNRELIABLE_CAPACITY
        assert "SPA" in UNRELIABLE_CAPACITY


class TestKpiExcludesPeakDerived:
    """The Highest-Stress KPI and National Utilization must both drop
    peak-derived BAs — otherwise a peak-derived BA near its own peak (~87%)
    would out-rank a genuinely stressed measured-plate BA, and its estimated
    capacity would pollute the national-utilization denominator."""

    def test_soco_never_wins_top_stress_despite_high_ratio(self):
        from components.callbacks import _build_us_grid_metrics_items

        # SOCO 50,000 / 54,980 ≈ 91% would beat PJM (~38%) for top stress if
        # it weren't excluded — but its capacity is peak-derived (#254).
        region_data = {
            "SOCO": {"current_mw": 50_000.0, "today_mw": [50_000.0] * 24},
            "PJM": {"current_mw": 70_000.0, "today_mw": [70_000.0] * 24},
        }
        items = _build_us_grid_metrics_items(region_data)
        top = next(i for i in items if i["label"] == "Highest-Stress Region")
        assert "PJM" in top["value"]
        assert "SOCO" not in top["value"]

    def test_national_utilization_excludes_peak_derived_from_denominator(self):
        from components.callbacks import _build_us_grid_metrics_items

        # Only PJM is a measured plate here → util = 70,000 / 184,202 ≈ 38%.
        # If SOCO leaked in it would read (120,000 / 239,182) ≈ 50%.
        region_data = {
            "SOCO": {"current_mw": 50_000.0, "today_mw": [50_000.0] * 24},
            "PJM": {"current_mw": 70_000.0, "today_mw": [70_000.0] * 24},
        }
        items = _build_us_grid_metrics_items(region_data)
        util = next(i for i in items if i["label"] == "National Utilization")
        assert util["value"] == "38%"


class TestCardEstimatedChip:
    """A peak-derived non-import BA renders the qualitative "est." chip, not a
    circular util % that reads like a measured-plate stress figure."""

    def test_soco_card_renders_est_chip_not_imports(self):
        from components.callbacks import _build_us_grid_region_card

        card = _build_us_grid_region_card(
            "SOCO",
            {"current_mw": 40_000.0, "today_mw": [40_000.0] * 24},
        )
        assert _find_text(card, "est."), "SOCO card should show the 'est.' chip"
        assert not _find_text(card, "imports"), "SOCO is not import-dominated"


class TestHoverSuffix:
    def test_suffix_distinguishes_import_peak_and_measured(self):
        from components._callbacks_us_grid import _capacity_hover_suffix

        assert _capacity_hover_suffix("SOCO") == " · est. cap"  # peak-derived
        assert _capacity_hover_suffix("CPLW") == " · imports"  # both → imports wins
        assert _capacity_hover_suffix("SPA") == " · imports"  # import-dominated
        assert _capacity_hover_suffix("PJM") == ""  # measured plate

    def test_choropleth_customdata_carries_est_cap_tag(self):
        from dash import dcc

        from components.callbacks import _build_us_grid_choropleth

        region_data = {
            "PJM": {"current_mw": 70_000.0},
            "SOCO": {"current_mw": 40_000.0},
        }
        body = _build_us_grid_choropleth(region_data)

        def find_graph(c):
            if isinstance(c, dcc.Graph):
                return c
            children = getattr(c, "children", None)
            if isinstance(children, (list, tuple)):
                for x in children:
                    g = find_graph(x)
                    if g is not None:
                        return g
            elif children is not None:
                return find_graph(children)
            return None

        graph = find_graph(body)
        rows = {row[0]: row for row in graph.figure.data[0].customdata}
        assert " · est. cap" in str(rows["SOCO"][3])
        assert str(rows["PJM"][3]) == ""
