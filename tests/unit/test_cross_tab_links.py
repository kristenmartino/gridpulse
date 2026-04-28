"""
Unit tests for NEXD-11: Cross-Tab Contextual Links.

Covers:
- Insight.related_tab / related_tab_label fields
- Tab generators tag insights with correct related_tab
- build_insight_card() renders cross-tab link when related_tab is set
- build_insight_card() omits link when on same tab or related_tab is None
- Quick-nav cards have pattern-matching IDs and n_clicks
- Feature flag gating
"""

import numpy as np
import pandas as pd

from components.insights import (
    Insight,
    build_insight_card,
    generate_tab1_insights,
    generate_tab2_insights,
    generate_tab3_insights,
    generate_tab4_insights,
)

# ── Insight dataclass fields ────────────────────────────────────────


class TestInsightRelatedTab:
    def test_default_related_tab_is_none(self):
        ins = Insight(text="test", category="info", severity="info")
        assert ins.related_tab is None
        assert ins.related_tab_label is None

    def test_related_tab_set_explicitly(self):
        ins = Insight(
            text="test",
            category="info",
            severity="info",
            related_tab="tab-forecast",
            related_tab_label="Historical",
        )
        assert ins.related_tab == "tab-forecast"
        assert ins.related_tab_label == "Historical"


# ── Tab generators tag insights ─────────────────────────────────────


def _make_demand_df(n_hours=168):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC"),
            "demand_mw": np.linspace(20000, 25000, n_hours),
        }
    )


def _make_weather_df(n_hours=168):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC"),
            "temperature_2m": np.linspace(65, 95, n_hours),
        }
    )


class TestTab1InsightsTagging:
    def test_tab1_insights_have_related_tab(self):
        df = _make_demand_df()
        wdf = _make_weather_df()
        insights = generate_tab1_insights("grid_ops", "FPL", df, wdf)
        assert len(insights) > 0
        for ins in insights:
            assert ins.related_tab == "tab-forecast"
            assert ins.related_tab_label == "Historical"


class TestTab2InsightsTagging:
    def test_tab2_insights_have_related_tab(self):
        preds = np.linspace(20000, 25000, 168)
        ts = pd.date_range("2024-01-01", periods=168, freq="h", tz="UTC")
        insights = generate_tab2_insights("grid_ops", "FPL", preds, ts)
        assert len(insights) > 0
        for ins in insights:
            assert ins.related_tab == "tab-outlook"
            assert ins.related_tab_label == "Forecast"


class TestTab3InsightsTagging:
    def test_tab3_insights_have_related_tab(self):
        metrics = {
            "xgboost": {"mape": 3.5, "rmse": 500, "mae": 400, "r2": 0.96},
            "prophet": {"mape": 5.0, "rmse": 700, "mae": 600, "r2": 0.93},
        }
        actual = np.linspace(20000, 25000, 48)
        preds = actual * 1.02
        ts = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        insights = generate_tab3_insights(
            "data_scientist",
            "FPL",
            metrics,
            "xgboost",
            actual=actual,
            predictions=preds,
            timestamps=ts,
        )
        assert len(insights) > 0
        for ins in insights:
            assert ins.related_tab == "tab-backtest"
            assert ins.related_tab_label == "Validation"


class TestTab4InsightsTagging:
    def test_tab4_insights_have_related_tab(self):
        n = 168
        net_load = pd.Series(np.linspace(15000, 20000, n))
        demand = pd.Series(np.linspace(20000, 25000, n))
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        insights = generate_tab4_insights(
            "renewables",
            "FPL",
            net_load,
            demand,
            renewable_pct=25.0,
            pivot=None,
            timestamps=ts,
        )
        assert len(insights) > 0
        for ins in insights:
            assert ins.related_tab == "tab-generation"
            assert ins.related_tab_label == "Grid"


# ── build_insight_card cross-tab link rendering ─────────────────────


class TestBuildInsightCardLinks:
    def _make_insights(self, related_tab=None, related_tab_label=None):
        return [
            Insight(
                text="Peak demand reached 25,000 MW.",
                category="pattern",
                severity="info",
                persona_relevance=["grid_ops"],
                related_tab=related_tab,
                related_tab_label=related_tab_label,
            ),
        ]

    def test_card_has_link_when_related_tab_set(self):
        insights = self._make_insights("tab-forecast", "Historical")
        card = build_insight_card(insights, "grid_ops", "tab-overview")
        text = str(card)
        assert "Historical" in text
        assert "cross-tab-link" in text

    def test_card_no_link_when_related_tab_none(self):
        insights = self._make_insights(None, None)
        card = build_insight_card(insights, "grid_ops", "tab-overview")
        text = str(card)
        assert "cross-tab-link" not in text

    def test_card_no_link_when_same_tab(self):
        """Don't show link when we're already on the source tab."""
        insights = self._make_insights("tab-forecast", "Historical")
        card = build_insight_card(insights, "grid_ops", "tab-forecast")
        text = str(card)
        assert "cross-tab-link" not in text

    def test_card_empty_when_no_insights(self):
        card = build_insight_card([], "grid_ops", "tab-overview")
        assert card.children is None or card.children == []

    def test_link_element_has_pattern_matching_id(self):
        insights = self._make_insights("tab-outlook", "Forecast")
        card = build_insight_card(insights, "grid_ops", "tab-overview")
        link_span = _find_cross_tab_link(card)
        assert link_span is not None
        assert link_span.id == {"type": "cross-tab-link", "index": "tab-outlook"}
        assert link_span.n_clicks == 0


# ── Quick-nav cards (removed in R2 of shell-redesign-v2.md) ─────────
#
# The 4-card quick-nav row on Overview was deleted as part of the
# v2-style linear-stack rebuild — the tab strip is the canonical
# navigation. The assertions below pin the removal so they don't
# silently re-appear.


class TestQuickNavRemoved:
    def test_quick_nav_cards_absent_from_overview(self):
        from components.tab_overview import layout

        result = layout()
        ids = _collect_all_ids(result)
        quick_nav_ids = [i for i in ids if isinstance(i, dict) and i.get("type") == "quick-nav-btn"]
        assert quick_nav_ids == [], (
            "Quick-nav cards were removed in R2; tab strip handles navigation."
        )

    def test_quick_nav_cards_not_findable(self):
        from components.tab_overview import layout

        nav_cards = _find_quick_nav_cards(layout())
        assert nav_cards == []


# ── Feature flag ────────────────────────────────────────────────────


class TestFeatureFlag:
    def test_cross_tab_links_flag_exists(self):
        from config import FEATURE_FLAGS

        assert "cross_tab_links" in FEATURE_FLAGS

    def test_cross_tab_links_flag_enabled(self):
        from config import feature_enabled

        assert feature_enabled("cross_tab_links") is True


# ── Helpers ─────────────────────────────────────────────────────────


def _find_cross_tab_link(component):
    """Recursively find a Span with className='cross-tab-link'."""
    if hasattr(component, "className") and component.className == "cross-tab-link":
        return component
    if hasattr(component, "children"):
        children = component.children
        if isinstance(children, (list, tuple)):
            for child in children:
                result = _find_cross_tab_link(child)
                if result:
                    return result
        elif children is not None:
            return _find_cross_tab_link(children)
    return None


def _collect_all_ids(component, collected=None):
    """Recursively collect all component IDs (including dict IDs)."""
    if collected is None:
        collected = []
    if hasattr(component, "id") and component.id is not None:
        collected.append(component.id)
    if hasattr(component, "children"):
        children = component.children
        if isinstance(children, (list, tuple)):
            for child in children:
                _collect_all_ids(child, collected)
        elif children is not None:
            _collect_all_ids(children, collected)
    return collected


def _find_quick_nav_cards(component, cards=None):
    """Recursively find all divs with quick-nav-btn pattern-matching IDs."""
    if cards is None:
        cards = []
    if (
        hasattr(component, "id")
        and isinstance(component.id, dict)
        and component.id.get("type") == "quick-nav-btn"
    ):
        cards.append(component)
    if hasattr(component, "children"):
        children = component.children
        if isinstance(children, (list, tuple)):
            for child in children:
                _find_quick_nav_cards(child, cards)
        elif children is not None:
            _find_quick_nav_cards(children, cards)
    return cards
