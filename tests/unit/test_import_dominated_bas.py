"""V3.η — import-dominated BA tagging + capacity correction.

Background. EIA-860M reports nameplate capacity for *in-territory*
generators. For most BAs that's a fine proxy for the resource pool the
operator can draw on, but a handful of EIA-930 BAs serve far more
demand than they generate locally — they're effectively pure
distribution / wires utilities buying wholesale from neighbours.

CPLW (DEP-West, NC mountains) was the smoking gun: 42 MW of in-territory
generation vs ~1,261 MW peak demand (a 30× multiplier). On the dashboard
this surfaced as "Highest-Stress Region: CPLW · 1071%". PR #76 capped
the *display*; this V3.η work fixes the *underlying classification*:

1. Re-source capacity for the import-dominated BAs as `peak × 1.15`
   (12-month peak demand × standard reserve margin) — a realistic
   proxy for the resource pool, not a lower-bound count of in-territory
   generators.
2. Tag the structurally import-dominated BAs in
   ``IS_IMPORT_DOMINATED`` so the UI:
     - Excludes them from "highest stress" candidate ranking
     - Annotates their hover with a " · imports" suffix
     - Renders the "imports" qualitative chip on their region card
       even when their (corrected) ratio happens to be below the
       reliability ceiling on a given hour.
"""

from __future__ import annotations


class TestImportDominatedSet:
    """The set itself: who's in it, what shape, and why."""

    def test_set_is_a_frozenset(self):
        from config import IS_IMPORT_DOMINATED

        assert isinstance(IS_IMPORT_DOMINATED, frozenset)

    def test_known_import_dominated_bas_are_tagged(self):
        """The three BAs identified as structurally import-dependent
        (multiplier ≥ 2× in-territory generation vs 12-month peak)."""
        from config import IS_IMPORT_DOMINATED

        # CPLW: 30× multiplier (42 MW gen vs 1,261 MW peak)
        # HST: 4.08× multiplier (36 MW gen vs 147 MW peak)
        # SPA: federal hydro marketer — has no local generation in the
        #      vertically-integrated-utility sense; it's pure wires.
        assert "CPLW" in IS_IMPORT_DOMINATED
        assert "HST" in IS_IMPORT_DOMINATED
        assert "SPA" in IS_IMPORT_DOMINATED

    def test_normal_bas_not_tagged(self):
        """Sanity check — vertically integrated BAs with their own
        generation fleets must NOT be tagged. Tagging them would hide
        them from the highest-stress KPI (and they'd never contend for
        it again, hiding real grid stress events)."""
        from config import IS_IMPORT_DOMINATED

        for ba in ("PJM", "ERCOT", "CAISO", "MISO", "SPP", "SOCO", "DUK", "PSCO"):
            assert ba not in IS_IMPORT_DOMINATED, (
                f"{ba} is a generation-rich BA — must not be marked import-dominated"
            )

    def test_all_tagged_bas_exist_in_capacity_dict(self):
        """A tagged BA that isn't in REGION_CAPACITY_MW would silently
        render as a zero-capacity card. Belt-and-braces."""
        from config import IS_IMPORT_DOMINATED, REGION_CAPACITY_MW

        for ba in IS_IMPORT_DOMINATED:
            assert ba in REGION_CAPACITY_MW, (
                f"{ba} tagged as import-dominated but missing from REGION_CAPACITY_MW"
            )


class TestV3EtaCapacityCorrections:
    """The seven capacity bumps — peak × 1.15 for BAs whose previous
    EIA-860M figure was below the served peak."""

    def test_cplw_capacity_corrected(self):
        """CPLW was 42 MW (in-territory generators only) — produced the
        1,071% stress reading. Now: peak 1,261 × 1.15 ≈ 1,450 MW."""
        from config import REGION_CAPACITY_MW

        assert REGION_CAPACITY_MW["CPLW"] >= 1_400, (
            f"CPLW capacity {REGION_CAPACITY_MW['CPLW']} too low — "
            "the peak × 1.15 correction lifts it to ~1,450 MW"
        )

    def test_hst_capacity_corrected(self):
        """HST (Homestead, FL) was 36 MW. Peak 147 × 1.15 ≈ 169 MW."""
        from config import REGION_CAPACITY_MW

        assert REGION_CAPACITY_MW["HST"] >= 150

    def test_soco_capacity_corrected(self):
        """SOCO peak 47,809 × 1.15 ≈ 54,980 MW (was 46,000 MW)."""
        from config import REGION_CAPACITY_MW

        assert REGION_CAPACITY_MW["SOCO"] >= 54_000

    def test_duk_cple_psco_corrected(self):
        """The three other peak-bumped BAs from V3.η. Each was below its
        served peak under EIA-860M; corrected via peak × 1.15."""
        from config import REGION_CAPACITY_MW

        # DUK 22,186 × 1.15 ≈ 25,513
        assert REGION_CAPACITY_MW["DUK"] >= 25_000
        # CPLE 14,329 × 1.15 ≈ 16,478
        assert REGION_CAPACITY_MW["CPLE"] >= 16_000
        # PSCO 10,642 × 1.15 ≈ 12,238
        assert REGION_CAPACITY_MW["PSCO"] >= 12_000


class TestKpiSuppressesImportDominated:
    """The "Highest-Stress Region" KPI must drop import-dominated BAs
    from candidacy regardless of where their (corrected) ratio lands.
    The capacity is an estimate — even a benign 60% ratio against an
    estimated pool isn't comparable to a 60% ratio against measured
    plate at PJM."""

    def test_cplw_excluded_from_top_stress_even_when_ratio_is_normal(self):
        """With the V3.η correction CPLW's ratio sits in the normal
        band, but it must STILL be filtered out — the IS_IMPORT_DOMINATED
        tag is the authoritative signal, not the ratio threshold."""
        from components.callbacks import _build_us_grid_metrics_items

        # CPLW corrected capacity ≈ 1,450 MW. Demand 1,000 MW → 69% — a
        # mathematically reliable ratio. Without IS_IMPORT_DOMINATED
        # filter, CPLW would beat PJM (~38%) for top stress.
        region_data = {
            "CPLW": {"current_mw": 1_000.0, "today_mw": [1_000.0] * 24},
            "PJM": {"current_mw": 70_000.0, "today_mw": [70_000.0] * 24},
        }
        items = _build_us_grid_metrics_items(region_data)
        top = next(i for i in items if i["label"] == "Highest-Stress Region")
        assert "PJM" in top["value"]
        assert "CPLW" not in top["value"]

    def test_spa_excluded_even_with_low_ratio(self):
        """SPA (federal hydro marketer) capacity is real plate but
        served demand is wrong shape — exclude regardless of ratio."""
        from components.callbacks import _build_us_grid_metrics_items

        region_data = {
            "SPA": {"current_mw": 500.0, "today_mw": [500.0] * 24},
            "PJM": {"current_mw": 30_000.0, "today_mw": [30_000.0] * 24},
        }
        items = _build_us_grid_metrics_items(region_data)
        top = next(i for i in items if i["label"] == "Highest-Stress Region")
        assert "SPA" not in top["value"]


class TestCardImportsChip:
    """The region-card stress chip should render "imports" for every
    IS_IMPORT_DOMINATED BA, regardless of whether the ratio is above
    or below the reliability ceiling on the rendered hour."""

    def test_card_renders_imports_chip_for_low_ratio_cplw(self):
        """V3.η scenario: CPLW corrected capacity 1,450 MW. At 60%
        ratio (870 MW demand) the OLD code would have shown "60%" —
        misleading because the 1,450 is an estimate, not a plate.
        New code shows the qualitative "imports" chip."""
        from components.callbacks import _build_us_grid_region_card

        card = _build_us_grid_region_card(
            "CPLW",
            {"current_mw": 870.0, "today_mw": [870.0] * 24},
        )

        # Walk for the "imports" chip
        def find_text(node, target):
            from dash import html as _html  # noqa: F401

            children = getattr(node, "children", None)
            if children is None:
                return False
            if isinstance(children, str):
                return target in children
            if isinstance(children, (list, tuple)):
                for c in children:
                    if find_text(c, target):
                        return True
            else:
                return find_text(children, target)
            return False

        assert find_text(card, "imports"), (
            "CPLW card should render the 'imports' chip even when its "
            "ratio is in the reliable band — the IS_IMPORT_DOMINATED "
            "tag is the authoritative signal."
        )


class TestPolygonHoverImportTag:
    """The polygon view's hover should append " · imports" to the BA
    name for IS_IMPORT_DOMINATED entries — surfaces the structural
    classification in the most clicked drilldown surface."""

    def test_customdata_carries_import_tag(self):
        from components.callbacks import _build_us_grid_choropleth

        region_data = {
            "PJM": {"current_mw": 70_000.0},
            "CPLW": {"current_mw": 870.0},
        }
        body = _build_us_grid_choropleth(region_data)

        # Walk for the dcc.Graph
        def find_graph(c):
            from dash import dcc

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
        cd = graph.figure.data[0].customdata
        # Plotly stores customdata as nd.array of objects — each row
        # has [region, name, demand_gw, import_tag]
        rows = {row[0]: row for row in cd}
        assert " · imports" in str(rows["CPLW"][3]), (
            "CPLW customdata should carry the import tag suffix"
        )
        assert str(rows["PJM"][3]) == "", (
            "PJM customdata should have an empty import tag (renders invisibly)"
        )
