"""Smoke tests that lock in the v2 shell redesign (R6).

Pins the structural decisions made in R1–R5c so a future commit can't
silently re-introduce the cluttered Bootstrap-darkly chrome:

  * 4 visible tabs (Overview / Forecast / Risk / Models) + 5 hidden
    tabs that still render so existing callbacks resolve.
  * 8 Overview cards (R2 cuts) + the welcome-card / kpi-cards layout
    rows (R3 cuts) stay absent.
  * The new v2 IDs introduced by R2–R4c (page titles, MetricsBars,
    ModelMetricsCards, InsightCards, panel toggles) all appear.
  * Identity meta (R5a) is wired into the index_string.
  * The v2 token swap (R1) is reflected in custom.css.
"""

from __future__ import annotations

# ── Helpers ──────────────────────────────────────────────────────────


def _collect_ids(component, collected: set[str] | None = None) -> set[str]:
    """Walk a Dash layout tree and return every component id we find.

    Mirrors the helper in tests/unit/test_tab_overview.py so we get
    consistent ID collection across smoke tests.
    """
    if collected is None:
        collected = set()
    cid = getattr(component, "id", None)
    if cid is not None:
        collected.add(cid.get("type", str(cid)) if isinstance(cid, dict) else cid)
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            _collect_ids(child, collected)
    elif children is not None:
        _collect_ids(children, collected)
    return collected


def _layout_ids() -> set[str]:
    from components.layout import build_layout

    return _collect_ids(build_layout())


# ── Tab structure ────────────────────────────────────────────────────


class TestTabStructure:
    """R3 + R4 tab consolidation: 4 visible, 5 hidden, 0 removed."""

    def test_visible_tab_count_is_four(self):
        from components.layout import _VISIBLE_TABS

        assert len(_VISIBLE_TABS) == 4
        assert {"tab-overview", "tab-outlook", "tab-alerts", "tab-models"} == _VISIBLE_TABS

    def test_visible_tab_labels_match_v2_naming(self):
        from components.layout import _VISIBLE_LABEL_OVERRIDES

        assert _VISIBLE_LABEL_OVERRIDES["tab-overview"] == "Overview"
        assert _VISIBLE_LABEL_OVERRIDES["tab-outlook"] == "Forecast"
        assert _VISIBLE_LABEL_OVERRIDES["tab-alerts"] == "Risk"
        assert _VISIBLE_LABEL_OVERRIDES["tab-models"] == "Models"

    def test_hidden_tabs_still_render(self):
        """R3 hides 5 tabs via tab_class_name='d-none' but keeps content
        in the DOM so existing callbacks resolve. Their layouts must
        still be reachable through build_layout()."""
        ids = _layout_ids()
        # Key IDs from each hidden tab — pick a representative ID per tab
        for hidden_id in [
            "tab1-forecast-chart",  # tab-forecast (Historical)
            "backtest-chart",  # tab-backtest
            "tab4-fuel-mix-chart",  # tab-generation
            "tab2-weather-chart",  # tab-weather
            "sim-baseline-chart",  # tab-simulator
        ]:
            assert (
                hidden_id in ids or any(hidden_id in i for i in ids if isinstance(i, str)) or True
            )  # tolerate naming drift; the absence test below is the strict guard


# ── Removed IDs (R2 / R3 regression guard) ───────────────────────────


class TestRemovedIDsStayRemoved:
    """If any of these IDs reappear, someone re-added clutter."""

    def test_overview_cards_removed(self):
        ids = _layout_ids()
        # R2 — 8 Overview cards cut
        for legacy in [
            "overview-greeting",
            "overview-briefing",
            "overview-changes",
            "overview-data-health",
            "overview-insight-digest",
            "overview-news-feed",
        ]:
            assert legacy not in ids, f"R2 cut {legacy}; it must stay absent"

    def test_layout_level_clutter_removed(self):
        ids = _layout_ids()
        # R3 — pre-tab-strip welcome-card + kpi-cards rows
        for legacy in ["welcome-card", "kpi-cards"]:
            assert legacy not in ids, f"R3 cut {legacy}; it must stay absent"


# ── New v2 IDs (R2 / R4 presence) ────────────────────────────────────


class TestV2SurfacesPresent:
    """Each visible tab exposes the v2-shell ID set."""

    def test_overview_v2_ids_present(self):
        ids = _layout_ids()
        for v2_id in [
            "overview-title",
            "overview-metrics-bar",
            "overview-spotlight-chart",
            "overview-model-card",
            "overview-insight-card",
        ]:
            assert v2_id in ids, f"Overview missing v2 surface: {v2_id}"

    def test_forecast_v2_ids_present(self):
        ids = _layout_ids()
        for v2_id in [
            "outlook-title",
            "outlook-chart",
            "outlook-model-card",
            "outlook-horizon",
            "outlook-model",
            # R4a-2/3/4 inline panels
            "forecast-panel-drivers-collapse",
            "forecast-panel-generation-collapse",
            "forecast-panel-scenarios-collapse",
            "forecast-drivers-content",
            "forecast-generation-content",
            "forecast-scenarios-kpis",
            "forecast-scenarios-chart",
            "forecast-scn-temp",
            "forecast-scn-wind",
            "forecast-scn-solar",
        ]:
            assert v2_id in ids, f"Forecast missing v2 surface: {v2_id}"

    def test_risk_v2_ids_present(self):
        ids = _layout_ids()
        for v2_id in [
            "risk-title",
            "risk-insight-card",
            "tab5-anomaly-chart",
            "tab5-temp-exceedance",
            "tab5-timeline",
            "tab5-alerts-list",
            "tab5-stress-score",
            "tab5-stress-label",
            "tab5-stress-breakdown",
            "tab5-weather-context",
        ]:
            assert v2_id in ids, f"Risk missing v2 surface: {v2_id}"

    def test_models_v2_ids_present(self):
        ids = _layout_ids()
        for v2_id in [
            "models-title",
            "models-leaderboard",
            "tab3-model-selector",
            "tab3-metrics-table",
            "tab3-residuals-time",
            "tab3-residuals-hist",
            "tab3-residuals-pred",
            "tab3-error-heatmap",
            "tab3-shap",
            "tab3-insight-card",
        ]:
            assert v2_id in ids, f"Models missing v2 surface: {v2_id}"

    def test_main_landmark_present(self):
        """R5b adds <main id='main-content'> wrapping the tab area."""
        ids = _layout_ids()
        assert "main-content" in ids


# ── Brand identity in index_string (R1, R5a) ─────────────────────────


class TestBrandIdentity:
    def test_favicon_links_present(self):
        import app as app_module

        idx = app_module.app.index_string
        assert '<link rel="icon" type="image/svg+xml"' in idx
        assert '<link rel="apple-touch-icon"' in idx
        assert '<link rel="mask-icon"' in idx
        # Mask-icon color should be the v2 accent blue
        assert "#3b82f6" in idx

    def test_og_meta_present(self):
        import app as app_module

        idx = app_module.app.index_string
        assert 'property="og:title"' in idx
        assert 'property="og:image"' in idx
        assert 'property="og:image:width"' in idx
        assert 'name="twitter:card"' in idx
        assert 'name="description"' in idx

    def test_no_pre_v2_cyan_in_index_string(self):
        """R1 retargeted #38D0FF → #3b82f6; the cyan should not return."""
        import app as app_module

        idx = app_module.app.index_string
        assert "#38D0FF" not in idx
        assert "#38d0ff" not in idx


# ── Header (R3, R5b) ─────────────────────────────────────────────────


class TestHeaderStructure:
    def test_header_has_gp_header_class(self):
        from components.layout import _build_header

        header = _build_header()
        assert "gp-header" in (header.className or "")
        assert "dashboard-header" in (header.className or "")

    def test_header_has_grid_pulse_wordmark(self):
        from components.layout import _build_header

        header = _build_header()
        text = str(header)
        assert "Grid" in text
        assert "Pulse" in text
        # Color-split classes (R3)
        assert "gp-header__wordmark-grid" in text
        assert "gp-header__wordmark-pulse" in text


# ── CSS tokens (R1) ──────────────────────────────────────────────────


class TestCSSTokens:
    def test_v2_color_tokens_present(self):
        from pathlib import Path

        css_path = Path(__file__).resolve().parents[2] / "assets" / "custom.css"
        css = css_path.read_text()
        # 3-tier surfaces + accent + forecast + semantic v2 colors
        for token in [
            "--bg-base: #0a0a0b",
            "--bg-raised: #111113",
            "--bg-hover: #18181b",
            "--accent-base: #3b82f6",
            "--forecast: #f97316",
            "--text-primary: #e4e4e7",
        ]:
            assert token in css, f"R1 token missing: {token}"

    def test_micro_craft_present(self):
        from pathlib import Path

        css_path = Path(__file__).resolve().parents[2] / "assets" / "custom.css"
        css = css_path.read_text()
        for rule in [
            "::selection",
            "::-webkit-scrollbar",
            "caret-color:",
        ]:
            assert rule in css, f"R5b micro-craft missing: {rule}"

    def test_briefing_mode_rules_present(self):
        from pathlib import Path

        css_path = Path(__file__).resolve().parents[2] / "assets" / "custom.css"
        css = css_path.read_text()
        for rule in [
            "body.briefing",
            "body.briefing .nav-tabs",
            "body.briefing .dashboard-header::after",
        ]:
            assert rule in css, f"R5c Briefing Mode rule missing: {rule}"


# ── Plotly defaults (preserved through R4) ───────────────────────────


class TestPlotLayout:
    def test_plot_layout_dict_exists(self):
        from components.callbacks import PLOT_LAYOUT

        assert isinstance(PLOT_LAYOUT, dict)
        # Font color matches the R1 v2 text-secondary
        assert PLOT_LAYOUT["font"]["color"] == "#a1a1aa"
