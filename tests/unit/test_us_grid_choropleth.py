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
