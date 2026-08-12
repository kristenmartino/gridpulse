"""
Unit tests for the Overview tab (#7 reimagined).

Covers:
- Layout returns html.Div with all required component IDs
- AI briefing module (rule-based fallback)
- Spotlight chart returns per-persona figures
- Insight digest returns valid content
- Config has tab-overview first in TAB_IDS
- All personas default to tab-overview
"""

from unittest.mock import patch

import pandas as pd
import plotly.graph_objects as go
from dash import html

from components.accessibility import model_display_name


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


class TestOverviewBriefing:
    """Test the AI briefing module (rule-based path)."""

    def test_rule_based_briefing_returns_result(self):
        from data.ai_briefing import BriefingResult, generate_briefing

        result = generate_briefing("grid_ops", "FPL")
        assert isinstance(result, BriefingResult)
        assert result.source == "rule_based"
        assert len(result.summary) > 0

    def test_rule_based_briefing_with_data(self):
        from data.ai_briefing import generate_briefing

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=48, freq="h"),
                "demand_mw": range(20000, 20048),
            }
        )
        result = generate_briefing("grid_ops", "FPL", demand_df=df)
        assert len(result.summary) > 0
        assert isinstance(result.observations, list)

    def test_rule_based_briefing_all_personas(self):
        from data.ai_briefing import generate_briefing
        from personas.config import PERSONAS

        for pid in PERSONAS:
            result = generate_briefing(pid, "FPL")
            assert len(result.summary) > 0

    def test_extract_data_context(self):
        from data.ai_briefing import _extract_data_context

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=48, freq="h"),
                "demand_mw": range(20000, 20048),
            }
        )
        ctx = _extract_data_context("FPL", df, None)
        assert ctx["peak_mw"] is not None
        assert ctx["peak_mw"] > 0
        assert ctx["data_points"] == 48

    def test_extract_data_context_none(self):
        from data.ai_briefing import _extract_data_context

        ctx = _extract_data_context("FPL", None, None)
        assert ctx["peak_mw"] is None
        assert ctx["data_points"] == 0

    def test_briefing_result_has_timestamp(self):
        from data.ai_briefing import BriefingResult

        result = BriefingResult(summary="test")
        assert result.generated_at != ""
        assert result.source == "rule_based"

    def test_parse_claude_response(self):
        from data.ai_briefing import _parse_claude_response

        raw = "This is a summary.\n---\n- Point one\n- Point two\n- Point three"
        result = _parse_claude_response(raw)
        assert "summary" in result.summary.lower() or len(result.summary) > 0
        assert len(result.observations) == 3
        assert result.source == "claude"


class TestOverviewSpotlight:
    """Test persona-specific spotlight chart selection."""

    def test_sparkline_with_none(self):
        from components.callbacks import _build_overview_sparkline

        fig = _build_overview_sparkline(None, "FPL")
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
        assert len(fig.data[0].y) == 24

    def test_spotlight_renewables(self):
        from components.callbacks import _spotlight_renewables

        wdf = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=48, freq="h"),
                "wind_speed_80m": [12.0] * 48,
                "shortwave_radiation": [300.0] * 48,
            }
        )
        fig = _spotlight_renewables(wdf, "FPL")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_spotlight_trader(self):
        from components.callbacks import _spotlight_trader

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=48, freq="h"),
                "demand_mw": range(20000, 20048),
            }
        )
        fig = _spotlight_trader(df, "FPL")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_spotlight_model_accuracy(self):
        from components.callbacks import _spotlight_model_accuracy

        measured = {
            "prophet": {"mape": 2.8},
            "arima": {"mape": 3.5},
            "xgboost": {"mape": 2.1},
        }
        with patch("models.model_service.get_model_metrics", return_value=measured):
            fig = _spotlight_model_accuracy("FPL")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        # Should have 3 bars (prophet, arima, xgboost)
        assert len(fig.data[0].x) == 3
        assert list(fig.data[0].y) == [2.8, 3.5, 2.1]

    def test_spotlight_dispatch_grid_ops(self):
        from components.callbacks import _build_overview_spotlight

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=48, freq="h"),
                "demand_mw": range(20000, 20048),
            }
        )
        fig = _build_overview_spotlight("grid_ops", "FPL", df, None)
        assert isinstance(fig, go.Figure)

    def test_spotlight_dispatch_renewables(self):
        from components.callbacks import _build_overview_spotlight

        wdf = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=48, freq="h"),
                "wind_speed_80m": [12.0] * 48,
                "shortwave_radiation": [300.0] * 48,
            }
        )
        fig = _build_overview_spotlight("renewables", "FPL", None, wdf)
        assert isinstance(fig, go.Figure)


class TestSpotlightModelAccuracyHonesty:
    """The spotlight MAPE chart must never draw a number it did not measure.

    Regression cover for the #131-class bug this chart carried: a missing
    backtest produced ``4.5 + len(model_name) * 0.3``, so Prophet rendered
    6.6% and arima 6.0% purely from the letter count of the dict key, drawn
    with the same bar, the same ``%.1f%%`` label and the same
    ``%{y:.2f}% MAPE`` hover as a real measurement. Under ``REQUIRE_REDIS``
    a cold web tier must reach the warming state instead — the #149 rule.
    """

    @staticmethod
    def _accuracy_fig(metrics):
        from components.callbacks import _spotlight_model_accuracy

        with patch("models.model_service.get_model_metrics", return_value=metrics):
            return _spotlight_model_accuracy("FPL")

    def test_no_metrics_renders_warming_state_not_bars(self):
        """Cold read (``get_model_metrics`` → ``{}``) draws no bars at all."""
        fig = self._accuracy_fig({})
        assert not any(trace.type == "bar" for trace in fig.data)
        # Panel identity survives so the slot doesn't read as a broken chart.
        assert fig.layout.title.text == "Model MAPE Comparison"
        text = " ".join(a.text or "" for a in fig.layout.annotations)
        assert "not yet measured" in text.lower()

    def test_no_metrics_never_fabricates_from_name_length(self):
        """The old fallback's exact outputs must not appear anywhere."""
        fig = self._accuracy_fig({})
        # 4.5 + len("prophet")*0.3 = 6.6, len("arima")*0.3 → 6.0, "xgboost" → 6.6
        fabricated = {6.6, 6.0}
        for trace in fig.data:
            assert not (set(trace.y or ()) & fabricated)

    def test_partial_metrics_omits_unmeasured_model_and_discloses(self):
        """One measured model → one bar, plus a note naming what is missing."""
        fig = self._accuracy_fig({"xgboost": {"mape": 2.1}})
        bars = [t for t in fig.data if t.type == "bar"]
        assert len(bars) == 1
        # Canonical labels (#495) on both the bar and the omission note.
        assert list(bars[0].x) == [model_display_name("xgboost")] == ["XGBoost"]
        assert list(bars[0].y) == [2.1]
        note = " ".join(a.text or "" for a in fig.layout.annotations)
        assert "Prophet" in note and model_display_name("arima") in note

    def test_non_numeric_mape_is_treated_as_unmeasured(self):
        """``None``/NaN/garbage must drop the bar, not coerce to a value."""
        fig = self._accuracy_fig(
            {
                "prophet": {"mape": None},
                "arima": {"mape": float("nan")},
                "xgboost": {"mape": "n/a"},
            }
        )
        assert not any(trace.type == "bar" for trace in fig.data)

    def test_measured_values_are_passed_through_unmodified(self):
        fig = self._accuracy_fig(
            {
                "prophet": {"mape": 4.02},
                "arima": {"mape": 5.11},
                "xgboost": {"mape": 3.44},
            }
        )
        bars = [t for t in fig.data if t.type == "bar"]
        assert list(bars[0].y) == [4.02, 5.11, 3.44]
        # Nothing omitted → no disclosure note.
        assert not fig.layout.annotations


class TestOverviewDataHealth:
    """Test the data health badge builder."""

    def test_data_health_with_freshness(self):
        from components.callbacks import _build_overview_data_health

        freshness = {"demand": "fresh", "weather": "fresh", "alerts": "demo"}
        result = _build_overview_data_health(freshness)
        assert isinstance(result, html.Div)
        text = str(result)
        assert "DATA SOURCES" in text

    def test_data_health_with_none(self):
        from components.callbacks import _build_overview_data_health

        result = _build_overview_data_health(None)
        assert isinstance(result, html.Div)

    def test_data_health_shows_status(self):
        from components.callbacks import _build_overview_data_health

        freshness = {"demand": "stale", "weather": "fresh"}
        result = _build_overview_data_health(freshness)
        text = str(result)
        assert "STALE" in text or "LIVE" in text


class TestOverviewDigest:
    """Test cross-tab insight digest builder."""

    def test_digest_with_data(self):
        from components.callbacks import _build_overview_digest

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=168, freq="h"),
                "demand_mw": range(20000, 20168),
            }
        )
        result = _build_overview_digest("grid_ops", "FPL", df, None)
        assert isinstance(result, html.Div)

    def test_digest_with_no_data(self):
        from components.callbacks import _build_overview_digest

        result = _build_overview_digest("grid_ops", "FPL", None, None)
        assert isinstance(result, html.Div)

    def test_digest_for_all_personas(self):
        from components.callbacks import _build_overview_digest
        from personas.config import PERSONAS

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=168, freq="h"),
                "demand_mw": range(20000, 20168),
            }
        )
        for pid in PERSONAS:
            result = _build_overview_digest(pid, "FPL", df, None)
            assert isinstance(result, html.Div)


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
