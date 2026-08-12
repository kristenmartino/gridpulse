"""
Unit tests for the Overview tab (#7 reimagined).

Covers:
- Layout returns html.Div with all required component IDs
- Config has tab-overview first in TAB_IDS
- All personas default to tab-overview
"""

from dash import html


class TestOverviewLayout:
    """Verify tab_overview.layout() has all required component IDs."""

    def test_layout_returns_div(self):
        from components.tab_overview import layout

        result = layout()
        assert isinstance(result, html.Div)

    def test_layout_has_required_ids(self):
        """R2 v2-linear-stack layout exposes 5 dynamic IDs + the chart ID."""
        from components.tab_overview import layout

        result = layout()
        ids = _collect_ids(result)
        required = [
            "overview-title",
            "overview-metrics-bar",
            "overview-spotlight-chart",
            "overview-model-card",
            "overview-insight-card",
        ]
        for rid in required:
            assert rid in ids, f"Missing component ID: {rid}"

    def test_layout_no_legacy_ids(self):
        """IDs removed in R2 should not exist in the new linear-stack layout."""
        from components.tab_overview import layout

        result = layout()
        ids = _collect_ids(result)
        legacy_ids = [
            # Pre-R2 v1 IDs
            "overview-demand-sparkline",
            "overview-alerts-count",
            "overview-alerts-breakdown",
            "overview-nav-cards",
            "overview-kpi-row",
            "overview-freshness-badges",
            "overview-last-updated",
            # Cards removed by R2 (shell-redesign-v2.md)
            "overview-greeting",
            "overview-briefing",
            "overview-changes",
            "overview-data-health",
            "overview-insight-digest",
            "overview-news-feed",
        ]
        for legacy in legacy_ids:
            assert legacy not in ids, f"Legacy ID still present: {legacy}"


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


# ── Helpers ──────────────────────────────────────────────────


def _collect_ids(component, collected=None):
    """Recursively collect all component IDs from a Dash layout tree."""
    if collected is None:
        collected = set()

    if hasattr(component, "id") and component.id is not None:
        cid = component.id
        # Pattern-matching IDs are dicts; store the 'type' key as a string
        if isinstance(cid, dict):
            collected.add(cid.get("type", str(cid)))
        else:
            collected.add(cid)

    if hasattr(component, "children"):
        children = component.children
        if isinstance(children, (list, tuple)):
            for child in children:
                _collect_ids(child, collected)
        elif children is not None:
            _collect_ids(children, collected)

    return collected
