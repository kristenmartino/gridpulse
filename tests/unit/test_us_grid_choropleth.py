"""Unit tests for V3.β — real BA-polygon choropleth on the US Grid tab.

Coverage:
- ``assets/ba_polygons.geojson`` exists, parses, and covers every BA in
  ``REGION_COORDINATES``. The build-time guard catches drift if a future
  V3.ζ-style expansion adds a BA without a matching polygon.
- The polygon file size stays under the 500 KB target documented in
  NEXT_UP V3.β.
- ``_load_ba_polygons`` is cached (one filesystem read per process).
- ``_build_us_grid_choropleth`` returns an html.Div containing a
  ``us-grid-map`` ``dcc.Graph`` so the existing drilldown callback
  wires through unchanged.
- The figure carries a ``Choropleth`` trace with ``locations``,
  ``z`` (utilization %), ``customdata`` shaped as
  ``[[region, name, demand_gw], ...]``, and ``featureidkey`` set to
  ``properties.region``.
- Cold-state: every region missing ``current_mw`` → returns the
  warming placeholder.
- Drilldown handles both customdata shapes (string for scatter,
  list/tuple for choropleth) without breaking either path.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
GEOJSON_PATH = REPO_ROOT / "assets" / "ba_polygons.geojson"


class TestPolygonAsset:
    def test_asset_exists(self):
        assert GEOJSON_PATH.exists()

    def test_asset_under_size_budget(self):
        # NEXT_UP V3.β acceptance: <500 KB after simplification.
        size = GEOJSON_PATH.stat().st_size
        assert size < 500 * 1024, f"ba_polygons.geojson is {size:,} bytes (>500 KB)"

    def test_asset_parses_as_feature_collection(self):
        with open(GEOJSON_PATH) as f:
            gj = json.load(f)
        assert gj["type"] == "FeatureCollection"
        assert isinstance(gj["features"], list)
        assert len(gj["features"]) > 0

    def test_every_region_has_a_polygon(self):
        from config import REGION_COORDINATES

        with open(GEOJSON_PATH) as f:
            gj = json.load(f)
        coded = {f["properties"]["region"] for f in gj["features"]}
        missing = sorted(set(REGION_COORDINATES.keys()) - coded)
        assert not missing, f"Polygons missing for: {missing}"

    def test_no_unknown_codes_in_geojson(self):
        from config import REGION_COORDINATES

        with open(GEOJSON_PATH) as f:
            gj = json.load(f)
        coded = {f["properties"]["region"] for f in gj["features"]}
        unknown = sorted(coded - set(REGION_COORDINATES.keys()))
        assert not unknown, f"GeoJSON has codes that aren't in REGION_COORDINATES: {unknown}"

    def test_features_carry_geometry(self):
        with open(GEOJSON_PATH) as f:
            gj = json.load(f)
        for feat in gj["features"]:
            assert feat.get("geometry") is not None, (
                f"Feature {feat['properties'].get('region')} has no geometry"
            )
            assert feat["geometry"].get("type") in (
                "Polygon",
                "MultiPolygon",
            )


class TestPolygonLoader:
    def test_load_returns_feature_collection(self):
        from components.callbacks import _load_ba_polygons

        gj = _load_ba_polygons()
        assert gj is not None
        assert gj["type"] == "FeatureCollection"

    def test_load_is_cached(self):
        """Second call should be O(1) — no filesystem hit."""
        from components import callbacks as cb

        # Reset the cache to ensure a clean baseline
        cb._BA_POLYGONS_CACHE = None
        first = cb._load_ba_polygons()
        assert first is not None
        # Second call returns the cached object reference
        second = cb._load_ba_polygons()
        assert second is first


class TestChoroplethRender:
    def test_returns_html_div_with_us_grid_map(self):
        """Choropleth Div must carry ``us-grid-map`` Graph so the
        existing drilldown callback wires through without changes."""
        from components.callbacks import _build_us_grid_choropleth

        region_data = {
            "PJM": {"current_mw": 70_000.0},
            "ERCOT": {"current_mw": 50_000.0},
        }
        body = _build_us_grid_choropleth(region_data)
        # Walk for a Graph with id "us-grid-map"
        assert _find_graph_id(body) == "us-grid-map"

    def test_figure_uses_choropleth_trace(self):
        from components.callbacks import _build_us_grid_choropleth

        region_data = {
            "PJM": {"current_mw": 70_000.0},
            "ERCOT": {"current_mw": 50_000.0},
        }
        body = _build_us_grid_choropleth(region_data)
        graph = _find_graph(body)
        fig = graph.figure
        # Plotly stores trace types as strings on the trace dict
        assert fig.data[0].type == "choropleth"

    def test_locations_match_region_data_keys(self):
        from components.callbacks import _build_us_grid_choropleth

        region_data = {
            "PJM": {"current_mw": 70_000.0},
            "ERCOT": {"current_mw": 50_000.0},
            "MISO": {"current_mw": 65_000.0},
        }
        body = _build_us_grid_choropleth(region_data)
        graph = _find_graph(body)
        trace = graph.figure.data[0]
        assert set(trace.locations) == {"PJM", "ERCOT", "MISO"}

    def test_featureidkey_routes_through_properties_region(self):
        from components.callbacks import _build_us_grid_choropleth

        region_data = {"PJM": {"current_mw": 70_000.0}}
        body = _build_us_grid_choropleth(region_data)
        graph = _find_graph(body)
        assert graph.figure.data[0].featureidkey == "properties.region"

    def test_z_carries_utilization_percent(self):
        """Stress = current_mw / capacity * 100. With PJM 70k MW
        capacity 184,202 MW, z ≈ 38.0%."""
        from components.callbacks import _build_us_grid_choropleth

        region_data = {"PJM": {"current_mw": 70_000.0}}
        body = _build_us_grid_choropleth(region_data)
        graph = _find_graph(body)
        z_value = float(graph.figure.data[0].z[0])
        assert 35.0 < z_value < 42.0

    def test_customdata_shape_supports_drilldown_first_index(self):
        """``customdata[i][0]`` is the region code so the existing
        drilldown callback can pull it out generically."""
        from components.callbacks import _build_us_grid_choropleth

        region_data = {"PJM": {"current_mw": 70_000.0}}
        body = _build_us_grid_choropleth(region_data)
        graph = _find_graph(body)
        cd = graph.figure.data[0].customdata
        assert len(cd) == 1
        # Plotly may store as np.ndarray; just check first element.
        first_row = list(cd[0])
        assert first_row[0] == "PJM"

    def test_cold_state_returns_warming_placeholder(self):
        from components.callbacks import _build_us_grid_choropleth

        # No region has real current_mw → cold state
        region_data = {"PJM": {"current_mw": None}, "ERCOT": {}}
        body = _build_us_grid_choropleth(region_data)
        # The empty-state Div has class "gp-region-map--empty"
        assert "gp-region-map--empty" in (body.className or "")

    def test_falls_back_to_scatter_when_geojson_missing(self):
        from components import callbacks as cb

        # Force the loader to return None (asset corrupt / missing)
        with patch("components.callbacks._load_ba_polygons", return_value=None):
            body = cb._build_us_grid_choropleth({"PJM": {"current_mw": 70_000.0}})
        # Scatter helper returns a Div with the same gp-region-map class
        # but a Scattergeo trace.
        graph = _find_graph(body)
        assert graph is not None
        assert graph.figure.data[0].type == "scattergeo"


class TestPolygonCoverageCaption:
    """The Polygons view has visible dark-fill gaps where we don't yet
    ship a BA polygon (Idaho, parts of Montana / Wyoming, AK + HI
    insets). Without a caption a first-time viewer reads those as
    "broken." Caption converts them into honest "unmapped" indicators.
    """

    def test_caption_renders_in_polygon_body(self):
        from components.callbacks import _build_us_grid_choropleth

        body = _build_us_grid_choropleth(
            {
                "PJM": {"current_mw": 70_000.0},
                "ERCOT": {"current_mw": 50_000.0},
            }
        )

        def find_text(node, predicate):
            children = getattr(node, "children", None)
            if isinstance(children, str):
                return children if predicate(children) else None
            if isinstance(children, (list, tuple)):
                for c in children:
                    found = find_text(c, predicate)
                    if found is not None:
                        return found
            elif children is not None:
                return find_text(children, predicate)
            return None

        caption = find_text(body, lambda s: "balancing authorities mapped" in s)
        assert caption is not None, (
            "Polygon view body should contain a coverage caption that "
            "names how many BAs are mapped vs unmapped."
        )
        # The covered count comes from the populated dict — 2 in this test.
        assert "2 of " in caption

    def test_caption_absent_in_cold_state(self):
        """When all regions are warming, the choropleth returns an
        empty-state placeholder — caption shouldn't appear there
        (there's nothing mapped to count, and the empty-state copy
        already explains the situation)."""
        from components.callbacks import _build_us_grid_choropleth

        body = _build_us_grid_choropleth({"PJM": {"current_mw": None}})

        def has_text(node, target):
            children = getattr(node, "children", None)
            if isinstance(children, str):
                return target in children
            if isinstance(children, (list, tuple)):
                return any(has_text(c, target) for c in children)
            if children is not None:
                return has_text(children, target)
            return False

        assert not has_text(body, "balancing authorities mapped")


class TestPolygonWindingOrder:
    """Regression for the user-reported "all green" bug. Plotly's
    underlying renderer (D3-geo) treats a polygon's outer ring as the
    region boundary, but follows the **D3 convention** rather than RFC
    7946: outer rings are clockwise (negative shoelace area). When the
    upstream geojson had RFC-7946-style CCW outer rings, those polygons
    were rendered as "everything except this region" — they filled the
    entire viewport and visually merged into one giant green field.

    This test locks down winding order so a future asset replacement
    (or a manual edit) can't silently re-introduce the bug."""

    @staticmethod
    def _signed_area(ring):
        """Shoelace via trapezoid sum. Returns positive for CCW,
        negative for CW (standard mathematical convention with y-up)."""
        a = 0.0
        n = len(ring)
        for i in range(n):
            x1, y1 = ring[i][0], ring[i][1]
            x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
            a += (x2 - x1) * (y2 + y1)
        return -a / 2.0

    def test_all_outer_rings_clockwise(self):
        """Every outer ring must have negative signed area (CW). A CCW
        outer ring is the smoking gun for the all-green bug."""
        with open(GEOJSON_PATH) as f:
            gj = json.load(f)

        violations = []
        for feat in gj["features"]:
            rid = feat["properties"]["region"]
            for poly_idx, poly in enumerate(feat["geometry"]["coordinates"]):
                outer = poly[0]
                if self._signed_area(outer) > 0:
                    violations.append(f"{rid}[{poly_idx}]")

        assert not violations, f"Outer rings with CCW winding (would render inverted): {violations}"

    # Note: hole (inner ring) orientation is NOT asserted. Empirically
    # Plotly/D3-geo renders cutouts correctly regardless of hole winding
    # (likely uses an even-odd fill rule rather than strict ring-direction
    # interpretation). Holes are kept as-shipped by the upstream source.


class TestPolygonVisibility:
    """Regression tests for the user-reported "all green - no real lines"
    bug. Previously ``_MAP_BORDER_COLOR = "#1f1f23"`` was indistinguishable
    from ``_MAP_LAND_COLOR = "#111113"``, so adjacent BAs visually fused
    into one green blob and the user could only see Florida (where the
    BAs differ enough in stress to show distinct fills)."""

    def test_border_color_distinct_from_land_color(self):
        """Border must be perceptually different from the basemap fill —
        otherwise polygons have no visible edges against the unfilled
        background, and adjacent same-stress BAs merge visually."""
        from components.callbacks import _MAP_BORDER_COLOR, _MAP_LAND_COLOR

        # Same string would mean the bug is back.
        assert _MAP_BORDER_COLOR != _MAP_LAND_COLOR
        # _MAP_LAND_COLOR is a hex; _MAP_BORDER_COLOR is now an rgba()
        # with non-zero alpha. The simple string-inequality check above
        # catches the regression that motivated this test (both being
        # near-black hex). The alpha check below catches a different
        # regression (border accidentally set fully transparent).
        if _MAP_BORDER_COLOR.startswith("rgba"):
            # rgba(r, g, b, a) — alpha must be > 0
            alpha_str = _MAP_BORDER_COLOR.rsplit(",", 1)[1].rstrip(") ")
            assert float(alpha_str) > 0.0

    def test_colorscale_has_distinct_midrange_stops(self):
        """Most BAs operate at 30–70% utilization. With only [0.0, 0.7, 1.0]
        stops, 30% and 60% mapped to nearly identical greens. Need at
        least 4 stops so mid-range values are visually distinct."""
        from components.callbacks import _MAP_COLORSCALE

        assert len(_MAP_COLORSCALE) >= 4, (
            f"colorscale only has {len(_MAP_COLORSCALE)} stops — "
            "mid-range utilization values won't be visually distinct"
        )
        # Stops must be sorted, span [0.0, 1.0], and have unique colors.
        positions = [stop[0] for stop in _MAP_COLORSCALE]
        assert positions == sorted(positions)
        assert positions[0] == 0.0
        assert positions[-1] == 1.0
        colors = [stop[1] for stop in _MAP_COLORSCALE]
        assert len(set(colors)) == len(colors), "duplicate colors in colorscale"

    def test_choropleth_border_width_is_visible(self):
        """0.6 px renders as a hairline that disappears against same-shade
        neighbors. Bump to ≥ 0.8 px so polygon edges always read."""
        from components.callbacks import _build_us_grid_choropleth

        body = _build_us_grid_choropleth({"PJM": {"current_mw": 70_000.0}})
        graph = _find_graph(body)
        width = graph.figure.data[0].marker.line.width
        assert width >= 0.8, f"border width {width} too thin to read"


class TestDrilldownTolerance:
    """The drilldown callback should accept both customdata shapes:
    1-D string (scatter) and list/tuple (choropleth)."""

    def _drilldown_fn(self):
        """Locate the drilldown callback via the registered Dash app."""
        import dash
        import dash_bootstrap_components as dbc

        from components.callbacks import register_callbacks
        from components.layout import build_layout

        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            suppress_callback_exceptions=True,
        )
        app.layout = build_layout()
        register_callbacks(app)
        for _key, val in app.callback_map.items():
            fn = val.get("callback")
            if fn and getattr(fn, "__name__", "") == "drilldown_from_us_grid_map":
                return getattr(fn, "__wrapped__", fn)
        raise KeyError("drilldown_from_us_grid_map callback not registered")

    def test_scatter_customdata_string(self):
        """Scatter sends customdata as a string (the region code)."""
        fn = self._drilldown_fn()
        click_data = {"points": [{"customdata": "PJM"}]}
        region, tab = fn(click_data)
        assert region == "PJM"
        assert tab == "tab-outlook"

    def test_choropleth_customdata_list(self):
        """Choropleth sends customdata as a list whose [0] is the code."""
        fn = self._drilldown_fn()
        click_data = {"points": [{"customdata": ["PJM", "Mid-Atlantic (PJM)", 70.0]}]}
        region, tab = fn(click_data)
        assert region == "PJM"
        assert tab == "tab-outlook"

    def test_choropleth_customdata_tuple(self):
        fn = self._drilldown_fn()
        click_data = {"points": [{"customdata": ("ERCOT", "Texas (ERCOT)", 50.0)}]}
        region, _ = fn(click_data)
        assert region == "ERCOT"

    def test_no_click_returns_no_update(self):
        from dash import no_update

        fn = self._drilldown_fn()
        assert fn(None) == (no_update, no_update)
        assert fn({}) == (no_update, no_update)
        assert fn({"points": []}) == (no_update, no_update)


# ── helpers ──────────────────────────────────────────────────


def _find_graph(component):
    """Walk a Dash component tree and return the first dcc.Graph."""
    from dash import dcc

    if isinstance(component, dcc.Graph):
        return component
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for c in children:
            found = _find_graph(c)
            if found is not None:
                return found
    elif children is not None:
        return _find_graph(children)
    return None


def _find_graph_id(component) -> str | None:
    g = _find_graph(component)
    return g.id if g is not None else None
