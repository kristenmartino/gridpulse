"""Unit tests for the US Grid tab (V1.β).

Covers:
- Layout returns html.Div with the expected dynamic-content IDs
- Helper functions build the title, metrics-bar items, sparkline, and cards
- Region cards expose the pattern-matching ID shape that the drilldown
  callback listens on
- Empty / cold Redis state degrades gracefully (placeholder cards, "—"
  metrics, empty sparkline)
"""

from unittest.mock import patch

from dash import html


def _collect_ids(component, collected=None):
    """Walk a Dash layout tree and return every component id we find."""
    if collected is None:
        collected = set()
    cid = getattr(component, "id", None)
    if cid is not None:
        # pattern-matching IDs are dicts; record their type only
        collected.add(cid.get("type", str(cid)) if isinstance(cid, dict) else cid)
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            _collect_ids(child, collected)
    elif children is not None:
        _collect_ids(children, collected)
    return collected


class TestUsGridLayout:
    """Verify tab_us_grid.layout() exposes the dynamic-content IDs."""

    def test_layout_returns_div(self):
        from components.tab_us_grid import layout

        result = layout()
        assert isinstance(result, html.Div)

    def test_layout_has_required_ids(self):
        from components.tab_us_grid import layout

        ids = _collect_ids(layout())
        for rid in ("us-grid-title", "us-grid-metrics-bar", "us-grid-region-grid"):
            assert rid in ids, f"Missing component ID: {rid}"

    def test_layout_uses_section_stack(self):
        """Match the v2 page rhythm: gp-page > gp-section-stack."""
        from components.tab_us_grid import layout

        root = layout()
        assert root.className == "gp-page"
        inner = root.children[0]
        assert inner.className == "gp-section-stack"


class TestTabRegistration:
    """Tab is registered in config + visible in the layout strip."""

    def test_tab_id_in_config(self):
        from config import TAB_IDS, TAB_LABELS

        assert "tab-us-grid" in TAB_IDS
        assert TAB_LABELS["tab-us-grid"] == "US Grid"

    def test_visible_tabs_includes_us_grid(self):
        from components.layout import _VISIBLE_LABEL_OVERRIDES, _VISIBLE_TABS

        assert "tab-us-grid" in _VISIBLE_TABS
        assert _VISIBLE_LABEL_OVERRIDES["tab-us-grid"] == "US Grid"

    def test_us_grid_sits_between_overview_and_outlook(self):
        """Spec: 'New tab between Overview and Forecast.'"""
        from components.layout import build_layout

        layout = build_layout()
        # Find the dbc.Tabs container — it has id="dashboard-tabs"
        tabs_container = _find_by_id(layout, "dashboard-tabs")
        assert tabs_container is not None
        tab_ids = [child.tab_id for child in tabs_container.children]
        assert tab_ids.index("tab-us-grid") == tab_ids.index("tab-overview") + 1
        assert tab_ids.index("tab-us-grid") < tab_ids.index("tab-outlook")


class TestRegionDataCollection:
    """_collect_us_grid_region_data handles cold + warm Redis."""

    def test_empty_redis_returns_dict_per_region(self):
        from components.callbacks import _collect_us_grid_region_data
        from config import REGION_NAMES

        with patch("components.callbacks.redis_get", return_value=None):
            data = _collect_us_grid_region_data()

        assert set(data.keys()) == set(REGION_NAMES.keys())
        for region, region_data in data.items():
            assert region_data == {}, f"Expected empty placeholder for {region}"

    def test_populated_redis_yields_current_prev_today(self):
        from components.callbacks import _collect_us_grid_region_data

        synthetic = {"demand_mw": list(range(1000, 1050))}  # 50 entries
        with patch("components.callbacks.redis_get", return_value=synthetic):
            data = _collect_us_grid_region_data()

        sample = next(iter(data.values()))
        assert sample["current_mw"] == 1049
        assert sample["prev_mw"] == 1048
        assert sample["today_mw"] == list(range(1026, 1049 + 1))  # last 24
        assert len(sample["today_mw"]) == 24


class TestMetricsItems:
    """_build_us_grid_metrics_items returns the 4-up MetricsBar shape."""

    def test_empty_data_returns_dashes(self):
        from components.callbacks import _build_us_grid_metrics_items

        items = _build_us_grid_metrics_items({})
        assert len(items) == 4
        labels = [it["label"] for it in items]
        assert labels == [
            "Total Demand",
            "National Peak (24h)",
            "Highest-Stress Region",
            "Lowest Reserve",
        ]
        for item in items:
            assert item["value"] == "—"

    def test_populated_data_computes_rollups(self):
        from components.callbacks import _build_us_grid_metrics_items

        # Two regions, the second one is more stressed
        data = {
            "FPL": {"current_mw": 30000, "prev_mw": 29500, "today_mw": [29500, 30000]},
            "ERCOT": {"current_mw": 70000, "prev_mw": 69000, "today_mw": [69000, 70000]},
        }
        items = _build_us_grid_metrics_items(data)
        # Total demand
        assert items[0]["label"] == "Total Demand"
        assert items[0]["value"] == "100.0"
        assert items[0]["unit"] == "GW"
        assert items[0]["hero"] is True


class TestRegionCard:
    """_build_us_grid_region_card produces the pattern-matching ID + content."""

    def test_card_id_uses_pattern_match_shape(self):
        from components.callbacks import _build_us_grid_region_card

        card = _build_us_grid_region_card("ERCOT", {})
        assert isinstance(card.id, dict)
        assert card.id == {"type": "us-grid-region-card", "region": "ERCOT"}
        assert card.n_clicks == 0

    def test_empty_card_renders_placeholder(self):
        from components.callbacks import _build_us_grid_region_card

        card = _build_us_grid_region_card("FPL", {})
        assert "gp-region-card--empty" in card.className

    def test_populated_card_omits_empty_modifier(self):
        from components.callbacks import _build_us_grid_region_card

        card = _build_us_grid_region_card(
            "FPL",
            {"current_mw": 30000, "prev_mw": 29500, "today_mw": [29500, 30000]},
        )
        assert "gp-region-card--empty" not in card.className
        assert card.className == "gp-region-card"


class TestSparkline:
    """_build_us_grid_sparkline degrades cleanly on empty input."""

    def test_empty_values_returns_empty_marker(self):
        from components.callbacks import _build_us_grid_sparkline

        spark = _build_us_grid_sparkline([])
        assert "gp-region-card__sparkline--empty" in spark.className

    def test_single_value_too_short_for_polyline(self):
        from components.callbacks import _build_us_grid_sparkline

        spark = _build_us_grid_sparkline([100.0])
        assert "gp-region-card__sparkline--empty" in spark.className

    def test_two_or_more_values_renders_svg(self):
        from components.callbacks import _build_us_grid_sparkline

        spark = _build_us_grid_sparkline([100.0, 110.0, 105.0])
        assert spark.className == "gp-region-card__sparkline"


def _find_by_id(component, target_id):
    """Find the first descendant component whose .id matches target_id."""
    if getattr(component, "id", None) == target_id:
        return component
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            found = _find_by_id(child, target_id)
            if found is not None:
                return found
    elif children is not None:
        return _find_by_id(children, target_id)
    return None
