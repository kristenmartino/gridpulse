"""
Unit tests for the Overview landing tab (#7).

Covers:
- Layout returns html.Div with all required component IDs
- Sparkline handles None and valid data
- Alerts returns valid count and breakdown
- Freshness handles None and valid JSON
- Config has tab-overview first in TAB_IDS
- All personas default to tab-overview
- Nav cards built correctly per persona
"""

import json

import pandas as pd
import plotly.graph_objects as go
from dash import html


class TestOverviewLayout:
    """Verify tab_overview.layout() has all required component IDs."""

    def test_layout_returns_div(self):
        from components.tab_overview import layout

        result = layout()
        assert isinstance(result, html.Div)

    def test_layout_has_required_ids(self):
        from components.tab_overview import layout

        result = layout()
        # Flatten all IDs from the layout tree
        ids = _collect_ids(result)
        required = [
            "overview-greeting",
            "overview-freshness-badges",
            "overview-last-updated",
            "overview-kpi-row",
            "overview-demand-sparkline",
            "overview-alerts-count",
            "overview-alerts-breakdown",
            "overview-nav-cards",
        ]
        for rid in required:
            assert rid in ids, f"Missing component ID: {rid}"


class TestOverviewConfig:
    """Verify config and persona changes for the overview tab."""

    def test_tab_overview_first_in_tab_ids(self):
        from config import TAB_IDS

        assert TAB_IDS[0] == "tab-overview"

    def test_tab_overview_in_tab_labels(self):
        from config import TAB_LABELS

        assert "tab-overview" in TAB_LABELS
        assert TAB_LABELS["tab-overview"] == "Overview"

    def test_all_personas_default_to_overview(self):
        from personas.config import PERSONAS

        for pid, persona in PERSONAS.items():
            assert persona.default_tab == "tab-overview", (
                f"Persona '{pid}' defaults to '{persona.default_tab}', expected 'tab-overview'"
            )

    def test_all_personas_have_overview_in_priority_tabs(self):
        from personas.config import PERSONAS

        for pid, persona in PERSONAS.items():
            assert "tab-overview" in persona.priority_tabs, (
                f"Persona '{pid}' missing 'tab-overview' in priority_tabs"
            )


class TestOverviewSparkline:
    """Test _build_overview_sparkline helper."""

    def test_sparkline_with_none(self):
        from components.callbacks import _build_overview_sparkline

        fig = _build_overview_sparkline(None, "FPL")
        assert isinstance(fig, go.Figure)

    def test_sparkline_with_empty_df(self):
        from components.callbacks import _build_overview_sparkline

        df = pd.DataFrame(columns=["timestamp", "demand_mw"])
        fig = _build_overview_sparkline(df, "FPL")
        assert isinstance(fig, go.Figure)

    def test_sparkline_with_valid_data(self):
        from components.callbacks import _build_overview_sparkline

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=48, freq="h"),
                "demand_mw": range(20000, 20048),
            }
        )
        fig = _build_overview_sparkline(df, "FPL")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        # Should show last 24 hours
        assert len(fig.data[0].y) == 24


class TestOverviewAlerts:
    """Test _build_overview_alerts helper."""

    def test_alerts_returns_tuple(self):
        from components.callbacks import _build_overview_alerts

        count, breakdown = _build_overview_alerts("FPL")
        assert isinstance(count, str)
        assert isinstance(breakdown, html.Div)

    def test_alerts_count_is_numeric(self):
        from components.callbacks import _build_overview_alerts

        count, _ = _build_overview_alerts("FPL")
        assert count.isdigit()


class TestOverviewFreshness:
    """Test _build_overview_freshness helper."""

    def test_freshness_with_none(self):
        from components.callbacks import _build_overview_freshness

        badges, updated = _build_overview_freshness(None)
        assert isinstance(badges, html.Div)
        assert "Last updated" in updated

    def test_freshness_with_valid_json(self):
        from components.callbacks import _build_overview_freshness

        data = json.dumps(
            {
                "demand": "fresh",
                "weather": "fresh",
                "alerts": "demo",
                "latest_data": "2024-06-01T12:00:00Z",
            }
        )
        badges, updated = _build_overview_freshness(data)
        assert isinstance(badges, html.Div)
        assert "Jun 01" in updated

    def test_freshness_with_degraded_status(self):
        from components.callbacks import _build_overview_freshness

        data = json.dumps({"demand": "stale", "weather": "error", "alerts": "fresh"})
        badges, updated = _build_overview_freshness(data)
        assert isinstance(badges, html.Div)
        assert "Last updated: --" in updated


class TestOverviewNav:
    """Test _build_overview_nav helper."""

    def test_nav_cards_for_grid_ops(self):
        from components.callbacks import _build_overview_nav

        result = _build_overview_nav("grid_ops")
        assert isinstance(result, html.Div)

    def test_nav_excludes_overview_tab(self):
        from components.callbacks import _build_overview_nav
        from personas.config import get_persona

        persona = get_persona("grid_ops")
        result = _build_overview_nav("grid_ops")
        # The nav should have cards for priority tabs minus overview
        expected_count = len([t for t in persona.priority_tabs if t != "tab-overview"])
        # Count the Col children in the Row inside the Div
        row = result.children
        assert len(row.children) == expected_count

    def test_nav_cards_for_all_personas(self):
        from components.callbacks import _build_overview_nav
        from personas.config import PERSONAS

        for pid in PERSONAS:
            result = _build_overview_nav(pid)
            assert isinstance(result, html.Div)


# ── Helpers ──────────────────────────────────────────────────


def _collect_ids(component, collected=None):
    """Recursively collect all component IDs from a Dash layout tree."""
    if collected is None:
        collected = set()

    if hasattr(component, "id") and component.id is not None:
        collected.add(component.id)

    if hasattr(component, "children"):
        children = component.children
        if isinstance(children, (list, tuple)):
            for child in children:
                _collect_ids(child, collected)
        elif children is not None:
            _collect_ids(children, collected)

    return collected
