"""
Tests for callback closures registered inside register_callbacks(app).

These callbacks are defined as inner functions of register_callbacks() and
are only accessible via app.callback_map after registration. This module:

1. Creates a minimal Dash app
2. Calls register_callbacks(app) to wire up all callbacks
3. Looks up each callback function by its output key in app.callback_map
4. Invokes the function directly with controlled inputs

Covers ~10 callback closures across lines 1765-3112:
- update_forecast_chart (lines 1765-1823)
- update_tab1_kpis (lines 1843-1881)
- update_tab1_insights (lines 1898-1914)
- update_fallback_banner (lines 2868-2895)
- update_header_freshness (lines 2912-2941)
- restore_bookmark (lines 2968-2987)
- create_bookmark (lines 3001-3017)
- update_widget_confidence (lines 3030-3049)
- toggle_meeting_mode (lines 3072-3088)
"""

import json
from datetime import UTC, datetime

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from dash import html, no_update

# ---------------------------------------------------------------------------
# Fixture: register all callbacks, extract functions from app.callback_map
# ---------------------------------------------------------------------------


def _find_callback(app, name):
    """Find a callback function by its __name__ in app.callback_map."""
    for _key, val in app.callback_map.items():
        fn = val.get("callback")
        if fn and getattr(fn, "__name__", "") == name:
            return fn
    raise KeyError(f"Callback '{name}' not found in app.callback_map")


def _find_callback_by_output(app, output_fragment):
    """Find a callback function by a substring of its output key."""
    for key, val in app.callback_map.items():
        if output_fragment in key:
            fn = val.get("callback")
            if fn:
                return fn
    raise KeyError(f"No callback with output containing '{output_fragment}'")


@pytest.fixture(scope="module")
def registered_app():
    """Create a Dash app with all callbacks registered.

    Uses build_layout() to ensure every component ID exists in the DOM,
    which Dash requires for callback validation.
    """
    import dash_bootstrap_components as dbc

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
    )

    from components.layout import build_layout

    app.layout = build_layout()

    from components.callbacks import register_callbacks

    register_callbacks(app)
    return app


@pytest.fixture(scope="module")
def callbacks(registered_app):
    """Extract all unwrapped callback functions into a dict keyed by function name.

    Dash wraps registered callbacks with context-management code that
    expects internal kwargs (``outputs_list``).  The *original* user
    function is available via ``__wrapped__`` and can be called directly
    as a plain Python function.
    """
    fns = {}
    for _key, val in registered_app.callback_map.items():
        fn = val.get("callback")
        if fn and hasattr(fn, "__name__"):
            # Prefer the unwrapped original so we can call it without
            # Dash request context.
            raw = getattr(fn, "__wrapped__", fn)
            fns[fn.__name__] = raw
    return fns


# ---------------------------------------------------------------------------
# Shared test data helpers
# ---------------------------------------------------------------------------


def _demand_json(n=168, base_mw=30000):
    """Create realistic demand JSON for callback inputs."""
    start = datetime(2024, 6, 1, tzinfo=UTC)
    ts = pd.date_range(start, periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    demand = base_mw + 5000 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 300, n)
    df = pd.DataFrame({"timestamp": ts, "demand_mw": demand, "region": "FPL"})
    return df.to_json()


def _weather_json(n=168):
    """Create realistic weather JSON for callback inputs."""
    start = datetime(2024, 6, 1, tzinfo=UTC)
    ts = pd.date_range(start, periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "temperature_2m": 85 + rng.normal(0, 5, n),
            "wind_speed_80m": 12 + rng.normal(0, 3, n),
            "shortwave_radiation": np.maximum(0, 400 * np.sin(2 * np.pi * np.arange(n) / 24)),
            "relative_humidity_2m": 60 + rng.normal(0, 10, n),
            "cloud_cover": 50 + rng.normal(0, 15, n),
        }
    )
    return df.to_json()


def _freshness_json(**overrides):
    """Build a data-freshness-store JSON string."""
    data = {
        "demand": "fresh",
        "weather": "fresh",
        "alerts": "fresh",
        "timestamp": datetime.now(UTC).isoformat(),
        "latest_data": "2024-06-07T23:00:00Z",
    }
    data.update(overrides)
    return json.dumps(data)


# ===========================================================================
# 1. update_fallback_banner
# ===========================================================================


class TestUpdateFallbackBanner:
    """G2: API fallback banner callback (lines 2868-2895)."""

    def test_no_input_returns_no_update(self, callbacks):
        fn = callbacks["update_fallback_banner"]
        result = fn(None)
        assert result is no_update

    def test_empty_string_returns_no_update(self, callbacks):
        fn = callbacks["update_fallback_banner"]
        result = fn("")
        assert result is no_update

    def test_all_fresh_returns_empty_div(self, callbacks):
        fn = callbacks["update_fallback_banner"]
        result = fn(_freshness_json())
        # When all sources are fresh, should return an empty Div
        assert hasattr(result, "children") or result is not None

    def test_stale_source_shows_warning(self, callbacks):
        fn = callbacks["update_fallback_banner"]
        result = fn(_freshness_json(demand="stale"))
        # Should return a dbc.Alert with warning content
        import dash_bootstrap_components as dbc

        assert isinstance(result, dbc.Alert)

    def test_error_source_shows_danger(self, callbacks):
        fn = callbacks["update_fallback_banner"]
        result = fn(_freshness_json(demand="error"))
        import dash_bootstrap_components as dbc

        assert isinstance(result, dbc.Alert)
        assert result.color == "danger"

    def test_demo_source_shows_alert(self, callbacks):
        fn = callbacks["update_fallback_banner"]
        result = fn(_freshness_json(demand="demo", weather="demo", alerts="demo"))
        import dash_bootstrap_components as dbc

        assert isinstance(result, dbc.Alert)

    def test_mixed_statuses(self, callbacks):
        fn = callbacks["update_fallback_banner"]
        result = fn(_freshness_json(demand="stale", weather="error", alerts="fresh"))
        import dash_bootstrap_components as dbc

        assert isinstance(result, dbc.Alert)
        # error present => danger color
        assert result.color == "danger"


# ===========================================================================
# 2. update_header_freshness
# ===========================================================================


class TestUpdateHeaderFreshness:
    """G2: Header freshness badge callback (lines 2912-2941)."""

    def test_no_input_shows_loading(self, callbacks):
        fn = callbacks["update_header_freshness"]
        result = fn(None)
        assert isinstance(result, html.Span)

    def test_all_fresh_shows_live(self, callbacks):
        fn = callbacks["update_header_freshness"]
        result = fn(_freshness_json())
        assert isinstance(result, html.Span)
        # Check color indicates live
        style = result.style or {}
        assert style.get("color") == "#2BD67B"

    def test_all_demo_shows_demo_label(self, callbacks):
        fn = callbacks["update_header_freshness"]
        result = fn(_freshness_json(demand="demo", weather="demo", alerts="demo"))
        style = result.style or {}
        assert style.get("color") == "#A8B3C7"

    def test_any_error_shows_degraded(self, callbacks):
        fn = callbacks["update_header_freshness"]
        result = fn(_freshness_json(weather="error"))
        style = result.style or {}
        assert style.get("color") == "#FF5C7A"

    def test_stale_shows_partial(self, callbacks):
        fn = callbacks["update_header_freshness"]
        result = fn(_freshness_json(demand="stale"))
        style = result.style or {}
        assert style.get("color") == "#FFB84D"

    def test_latest_data_timestamp_rendered(self, callbacks):
        fn = callbacks["update_header_freshness"]
        result = fn(_freshness_json(latest_data="2024-06-07T18:00:00Z"))
        # The Span should contain nested children including data time
        assert isinstance(result, html.Span)
        # Children should be a list with sub-spans
        assert isinstance(result.children, list)
        assert len(result.children) >= 2

    def test_invalid_latest_data_handled(self, callbacks):
        fn = callbacks["update_header_freshness"]
        result = fn(_freshness_json(latest_data="not-a-date"))
        # Should not raise; returns Span with empty data time
        assert isinstance(result, html.Span)


# ===========================================================================
# 3. restore_bookmark
# ===========================================================================


class TestRestoreBookmark:
    """C2+NEXD-12: URL bookmark restore callback (15 outputs)."""

    def test_empty_search_returns_no_update(self, callbacks):
        fn = callbacks["restore_bookmark"]
        result = fn("")
        assert result == [no_update] * 15

    def test_none_search_returns_no_update(self, callbacks):
        fn = callbacks["restore_bookmark"]
        result = fn(None)
        assert result == [no_update] * 15

    def test_valid_region_parsed(self, callbacks):
        fn = callbacks["restore_bookmark"]
        result = fn("?region=FPL")
        assert result[0] == "FPL"
        assert result[1] is no_update
        assert result[2] is no_update

    def test_valid_persona_parsed(self, callbacks):
        fn = callbacks["restore_bookmark"]
        result = fn("?persona=grid_ops")
        assert result[0] is no_update
        assert result[1] == "grid_ops"

    def test_valid_tab_parsed(self, callbacks):
        fn = callbacks["restore_bookmark"]
        result = fn("?tab=tab-forecast")
        assert result[2] == "tab-forecast"

    def test_all_params_parsed(self, callbacks):
        fn = callbacks["restore_bookmark"]
        result = fn("?region=ERCOT&persona=trader&tab=tab-outlook")
        assert result[0] == "ERCOT"
        assert result[1] == "trader"
        assert result[2] == "tab-outlook"

    def test_invalid_region_ignored(self, callbacks):
        fn = callbacks["restore_bookmark"]
        result = fn("?region=INVALID_REGION")
        assert result[0] is no_update

    def test_invalid_persona_ignored(self, callbacks):
        fn = callbacks["restore_bookmark"]
        result = fn("?persona=nonexistent_persona")
        assert result[1] is no_update

    def test_invalid_tab_ignored(self, callbacks):
        fn = callbacks["restore_bookmark"]
        result = fn("?tab=tab-nonexistent")
        assert result[2] is no_update

    def test_filter_params_restored(self, callbacks):
        fn = callbacks["restore_bookmark"]
        result = fn("?region=FPL&f.tab1-timerange=168")
        # Index 3 = tab1-timerange (first filter output)
        assert result[3] == "168"

    def test_slider_params_restored(self, callbacks):
        fn = callbacks["restore_bookmark"]
        result = fn("?region=FPL&s.temp=5")
        # Index 10 = sim-temp (first slider output)
        assert result[10] == 5.0


# ===========================================================================
# 4. create_bookmark
# ===========================================================================


class TestCreateBookmark:
    """C2+NEXD-12: URL bookmark creation callback (full state)."""

    # 12 extra state values: 7 filters + 5 sim sliders (all None = no filter/slider state)
    _defaults = (None,) * 12

    def test_no_clicks_returns_no_update(self, callbacks):
        fn = callbacks["create_bookmark"]
        result = fn(0, "FPL", "grid_ops", "tab-forecast", *self._defaults)
        assert result == (no_update, no_update)

    def test_none_clicks_returns_no_update(self, callbacks):
        fn = callbacks["create_bookmark"]
        result = fn(None, "FPL", "grid_ops", "tab-forecast", *self._defaults)
        assert result == (no_update, no_update)

    def test_creates_bookmark_url(self, callbacks):
        fn = callbacks["create_bookmark"]
        search, toast = fn(1, "FPL", "grid_ops", "tab-forecast", *self._defaults)
        assert "region=FPL" in search
        assert "persona=grid_ops" in search
        assert "tab=tab-forecast" in search

    def test_creates_toast_component(self, callbacks):
        import dash_bootstrap_components as dbc

        fn = callbacks["create_bookmark"]
        search, toast = fn(1, "ERCOT", "trader", "tab-outlook", *self._defaults)
        assert isinstance(toast, dbc.Toast)
        assert toast.is_open is True

    def test_includes_filter_params(self, callbacks):
        fn = callbacks["create_bookmark"]
        # First filter state = tab1-timerange = "720"
        state_values = ("720",) + (None,) * 11
        search, toast = fn(1, "FPL", "grid_ops", "tab-forecast", *state_values)
        assert "f.tab1-timerange=720" in search

    def test_includes_slider_params(self, callbacks):
        fn = callbacks["create_bookmark"]
        # 7 filter Nones + sim-temp=5
        state_values = (None,) * 7 + (5,) + (None,) * 4
        search, toast = fn(1, "FPL", "grid_ops", "tab-simulator", *state_values)
        assert "s.temp=5" in search


# ===========================================================================
# 5. update_widget_confidence
# ===========================================================================


class TestUpdateWidgetConfidence:
    """A4+E3: Per-widget confidence badges callback (lines 3030-3049)."""

    def test_no_input_returns_empty(self, callbacks):
        fn = callbacks["update_widget_confidence"]
        result = fn(None)
        assert result == ""

    def test_empty_string_returns_empty(self, callbacks):
        fn = callbacks["update_widget_confidence"]
        result = fn("")
        assert result == ""

    def test_fresh_data_returns_badges(self, callbacks):
        fn = callbacks["update_widget_confidence"]
        result = fn(_freshness_json())
        # Should return children from widget_confidence_bar
        assert result is not None

    def test_stale_data_returns_badges(self, callbacks):
        fn = callbacks["update_widget_confidence"]
        result = fn(_freshness_json(demand="stale", timestamp=datetime.now(UTC).isoformat()))
        assert result is not None

    def test_no_timestamp_handles_gracefully(self, callbacks):
        fn = callbacks["update_widget_confidence"]
        data = json.dumps({"demand": "fresh", "weather": "fresh", "alerts": "fresh"})
        result = fn(data)
        assert result is not None


# ===========================================================================
# 6. toggle_meeting_mode
# ===========================================================================


class TestToggleMeetingMode:
    """C9: Briefing Mode (meeting-mode) toggle callback.

    R3 trimmed the welcome-card style output (the underlying div was
    deleted). Surviving outputs: meeting-mode-store, dashboard-header
    className, widget-confidence-bar style, fallback-banner style.
    """

    def test_toggle_on_from_false(self, callbacks):
        fn = callbacks["toggle_meeting_mode"]
        mode, header_cls, conf_style, banner_style = fn(1, "false")
        assert mode == "true"
        assert "meeting-mode" in header_cls
        assert conf_style == {"display": "none"}
        assert banner_style == {"display": "none"}

    def test_toggle_off_from_true(self, callbacks):
        fn = callbacks["toggle_meeting_mode"]
        mode, header_cls, conf_style, banner_style = fn(2, "true")
        assert mode == "false"
        # R3: header className includes both dashboard-header + gp-header
        assert "meeting-mode" not in header_cls
        assert "dashboard-header" in header_cls
        assert conf_style == {}
        assert banner_style == {}

    def test_toggle_on_from_none(self, callbacks):
        fn = callbacks["toggle_meeting_mode"]
        # None != "true" => is_meeting = True
        mode, header_cls, _conf_style, _banner_style = fn(1, None)
        assert mode == "true"
        assert "meeting-mode" in header_cls


# ===========================================================================
# 7. update_forecast_chart
# ===========================================================================


class TestUpdateForecastChart:
    """Historical demand chart callback (lines 1765-1823)."""

    def test_no_demand_returns_empty_figure(self, callbacks):
        fn = callbacks["update_forecast_chart"]
        result = fn(None, None, [], "168", [], "FPL")
        assert isinstance(result, go.Figure)
        # Empty figure should have the "No demand data loaded" annotation
        annotations = result.layout.annotations
        assert len(annotations) == 1
        assert "No demand data" in annotations[0].text

    def test_basic_demand_chart(self, callbacks):
        fn = callbacks["update_forecast_chart"]
        demand = _demand_json(n=200)
        result = fn(demand, None, [], "168", [], "FPL")
        assert isinstance(result, go.Figure)
        # Should have at least one trace (actual demand)
        assert len(result.data) >= 1
        assert result.data[0].name == "Actual Demand"

    def test_timerange_truncation(self, callbacks):
        fn = callbacks["update_forecast_chart"]
        demand = _demand_json(n=500)
        result = fn(demand, None, [], "168", [], "FPL")
        assert isinstance(result, go.Figure)
        # Should truncate to 168 hours
        assert len(result.data[0].x) == 168

    def test_weather_overlay_with_temp(self, callbacks):
        fn = callbacks["update_forecast_chart"]
        demand = _demand_json(n=168)
        weather = _weather_json(n=168)
        result = fn(demand, weather, ["temp"], "168", [], "FPL")
        assert isinstance(result, go.Figure)
        # Should have 2 traces: demand + temperature
        assert len(result.data) >= 2
        assert result.data[1].name == "Temperature (\u00b0F)"

    def test_weather_overlay_no_temp(self, callbacks):
        fn = callbacks["update_forecast_chart"]
        demand = _demand_json(n=168)
        weather = _weather_json(n=168)
        # overlay does not include 'temp'
        result = fn(demand, weather, ["wind"], "168", [], "FPL")
        assert isinstance(result, go.Figure)
        # Should have only demand trace
        assert len(result.data) == 1

    def test_chart_title_includes_region(self, callbacks):
        fn = callbacks["update_forecast_chart"]
        demand = _demand_json(n=100)
        result = fn(demand, None, [], "168", [], "ERCOT")
        assert "ERCOT" in result.layout.title.text

    def test_empty_demand_json_string(self, callbacks):
        fn = callbacks["update_forecast_chart"]
        result = fn("", None, [], "168", [], "FPL")
        assert isinstance(result, go.Figure)
        # Empty string is falsy, should return empty figure
        annotations = result.layout.annotations
        assert len(annotations) == 1


# ===========================================================================
# 8. update_tab1_kpis
# ===========================================================================


class TestUpdateTab1Kpis:
    """Tab 1 KPI cards callback (lines 1843-1881)."""

    def test_no_demand_returns_placeholders(self, callbacks):
        fn = callbacks["update_tab1_kpis"]
        result = fn(None, None, "FPL")
        assert len(result) == 7
        assert result[0] == "No data"

    def test_empty_demand_returns_placeholders(self, callbacks):
        fn = callbacks["update_tab1_kpis"]
        result = fn("", None, "FPL")
        assert result[0] == "No data"

    def test_valid_demand_returns_kpis(self, callbacks):
        fn = callbacks["update_tab1_kpis"]
        demand = _demand_json(n=168, base_mw=30000)
        result = fn(demand, None, "FPL")
        peak_str, peak_time, avg_str, min_str, min_time, data_str, days_span = result
        # Peak should be formatted with commas and "MW"
        assert "MW" in peak_str
        assert "," in peak_str  # thousands separator
        # Peak time should be UTC formatted
        assert "UTC" in peak_time
        # Average and min
        assert "MW" in avg_str
        assert "MW" in min_str
        assert "UTC" in min_time
        # Data count
        assert data_str == "168"
        # Days span
        assert isinstance(days_span, html.Span)

    def test_nan_demand_values_handled(self, callbacks):
        fn = callbacks["update_tab1_kpis"]
        # Create demand with all NaN demand_mw
        start = datetime(2024, 6, 1, tzinfo=UTC)
        ts = pd.date_range(start, periods=10, freq="h", tz="UTC")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "demand_mw": [float("nan")] * 10,
                "region": "FPL",
            }
        )
        result = fn(df.to_json(), None, "FPL")
        # All NaN => empty valid_data => placeholders
        assert result[0] == "No data"


# ===========================================================================
# 9. update_tab1_insights
# ===========================================================================


class TestUpdateTab1Insights:
    """Tab 1 insight card callback (lines 1898-1914)."""

    def test_no_demand_returns_empty_div(self, callbacks):
        fn = callbacks["update_tab1_insights"]
        result = fn(None, None, "168", "grid_ops", "FPL")
        assert isinstance(result, html.Div)
        # Empty Div has no children or empty children
        assert not result.children

    def test_with_demand_returns_insight_card(self, callbacks):
        fn = callbacks["update_tab1_insights"]
        demand = _demand_json(n=168)
        result = fn(demand, None, "168", "grid_ops", "FPL")
        # Should return a Div (the insight card)
        assert isinstance(result, html.Div)

    def test_with_weather_data(self, callbacks):
        fn = callbacks["update_tab1_insights"]
        demand = _demand_json(n=168)
        weather = _weather_json(n=168)
        result = fn(demand, weather, "168", "grid_ops", "FPL")
        assert isinstance(result, html.Div)

    def test_default_persona_used_when_none(self, callbacks):
        fn = callbacks["update_tab1_insights"]
        demand = _demand_json(n=168)
        result = fn(demand, None, "168", None, "FPL")
        # persona defaults to "grid_ops" when None
        assert isinstance(result, html.Div)

    def test_default_timerange_when_none(self, callbacks):
        fn = callbacks["update_tab1_insights"]
        demand = _demand_json(n=168)
        result = fn(demand, None, None, "grid_ops", "FPL")
        # timerange defaults to 168 when None
        assert isinstance(result, html.Div)

    def test_default_region_when_none(self, callbacks):
        fn = callbacks["update_tab1_insights"]
        demand = _demand_json(n=168)
        result = fn(demand, None, "168", "grid_ops", None)
        # region defaults to "FPL" when None
        assert isinstance(result, html.Div)


# Note: update_news_feed callback was removed — news moved into overview tab.
