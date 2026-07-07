"""Unit tests for the Phase-2 jump-to-card wiring in the US Grid tab.

Covers:
- _build_us_grid_metrics_items(region_data, view) adds cell_id to the
  Highest-Stress Region KPI item when view="cards" and stress data is available
- _build_us_grid_metrics_items(region_data) defaults to view="cards" for
  backward compatibility
- build_metrics_bar renders clickable cells (with n_clicks=0, role=button,
  gp-metric-cell--clickable class) when a cell_id is present
- build_metrics_bar renders non-clickable cells when cell_id is absent
"""

from unittest.mock import patch

from dash import html

REGION_DATA = {
    "PJM": {"current_mw": 90000.0, "today_mw": [90000.0] * 24},
}


class TestJumpIdInCardsView:
    """Highest-Stress KPI includes cell_id when view='cards'."""

    @patch("components._callbacks_us_grid.REGION_CAPACITY_MW", {"PJM": 120000.0})
    @patch("components._callbacks_us_grid.REGION_NAMES", {"PJM": "PJM"})
    @patch("components._callbacks_us_grid.IS_IMPORT_DOMINATED", set())
    def test_jump_id_present_in_cards_view(self):
        """cell_id is set on Highest-Stress item when view='cards'."""
        from components._callbacks_us_grid import _build_us_grid_metrics_items

        items = _build_us_grid_metrics_items(REGION_DATA, view="cards")
        # Find the Highest-Stress Region item
        hs = next(i for i in items if i["label"] == "Highest-Stress Region")
        assert hs.get("cell_id") == {"type": "us-grid-kpi-jump", "region": "PJM"}

    @patch("components._callbacks_us_grid.REGION_CAPACITY_MW", {"PJM": 120000.0})
    @patch("components._callbacks_us_grid.REGION_NAMES", {"PJM": "PJM"})
    @patch("components._callbacks_us_grid.IS_IMPORT_DOMINATED", set())
    def test_no_jump_id_in_map_view(self):
        """cell_id is absent on Highest-Stress item when view='map'."""
        from components._callbacks_us_grid import _build_us_grid_metrics_items

        items = _build_us_grid_metrics_items(REGION_DATA, view="map")
        # Find the Highest-Stress Region item
        hs = next(i for i in items if i["label"] == "Highest-Stress Region")
        assert hs.get("cell_id") is None or "cell_id" not in hs

    @patch("components._callbacks_us_grid.REGION_CAPACITY_MW", {})
    @patch("components._callbacks_us_grid.REGION_NAMES", {})
    @patch("components._callbacks_us_grid.IS_IMPORT_DOMINATED", set())
    def test_no_jump_id_on_placeholder_path(self):
        """cell_id is absent on all items when region_data is empty."""
        from components._callbacks_us_grid import _build_us_grid_metrics_items

        items = _build_us_grid_metrics_items({})
        for item in items:
            assert item.get("cell_id") is None or "cell_id" not in item

    @patch("components._callbacks_us_grid.REGION_CAPACITY_MW", {"PJM": 120000.0})
    @patch("components._callbacks_us_grid.REGION_NAMES", {"PJM": "PJM"})
    @patch("components._callbacks_us_grid.IS_IMPORT_DOMINATED", set())
    def test_default_arity_still_cards(self):
        """_build_us_grid_metrics_items(REGION_DATA) defaults view to 'cards'."""
        from components._callbacks_us_grid import _build_us_grid_metrics_items

        # Call with one argument only (no view parameter)
        items = _build_us_grid_metrics_items(REGION_DATA)
        # Should behave the same as view="cards"
        hs = next(i for i in items if i["label"] == "Highest-Stress Region")
        assert hs.get("cell_id") == {"type": "us-grid-kpi-jump", "region": "PJM"}


class TestMetricsBarClickableCell:
    """build_metrics_bar renders clickable cells when cell_id is present."""

    def test_metrics_bar_renders_clickable_cell(self):
        """Cell with cell_id has n_clicks, role=button, clickable className."""
        from components.cards import build_metrics_bar

        items = [
            {
                "label": "X",
                "value": "1",
                "cell_id": {"type": "us-grid-kpi-jump", "region": "PJM"},
            }
        ]
        bar = build_metrics_bar(items)
        # bar.children is a list of cell divs
        assert isinstance(bar, html.Div)
        assert hasattr(bar, "children")
        cells = bar.children
        assert len(cells) > 0
        first_cell = cells[0]
        # Check id is set
        assert getattr(first_cell, "id", None) == {"type": "us-grid-kpi-jump", "region": "PJM"}
        # Check n_clicks
        assert getattr(first_cell, "n_clicks", None) == 0
        # Check role
        assert getattr(first_cell, "role", None) == "button"
        # Check className includes clickable
        assert "gp-metric-cell--clickable" in first_cell.className

    def test_metrics_bar_plain_item_not_clickable(self):
        """Cell without cell_id lacks id, n_clicks, and clickable className."""
        from components.cards import build_metrics_bar

        items = [{"label": "Y", "value": "2"}]
        bar = build_metrics_bar(items)
        cells = bar.children
        assert len(cells) > 0
        first_cell = cells[0]
        # Check id is not set
        assert getattr(first_cell, "id", None) is None
        # Check className is plain (no clickable)
        assert first_cell.className == "gp-metric-cell"
