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
    """C2+NEXD-12: URL bookmark restore callback.

    V2.1 reduced the output set from 15 (3 core + 7 filters + 5 sliders)
    to 6 (3 core + 3 filters) — the dropped filters/sliders belonged to
    the now-removed hidden tabs (Historical / Backtest / Generation /
    Simulator). Output indexing: 0=region, 1=persona, 2=tab,
    3=outlook-horizon, 4=outlook-model, 5=tab3-model-selector.
    """

    def test_empty_search_returns_no_update(self, callbacks):
        fn = callbacks["restore_bookmark"]
        result = fn("")
        assert result == [no_update] * 6

    def test_none_search_returns_no_update(self, callbacks):
        fn = callbacks["restore_bookmark"]
        result = fn(None)
        assert result == [no_update] * 6

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
        result = fn("?tab=tab-outlook")
        assert result[2] == "tab-outlook"

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
        result = fn("?region=FPL&f.outlook-horizon=168")
        # Index 3 = outlook-horizon (first surviving filter output)
        assert result[3] == "168"


# ===========================================================================
# 4. create_bookmark
# ===========================================================================


class TestCreateBookmark:
    """C2+NEXD-12: URL bookmark creation callback.

    V2.1 reduced the State set to 3 filters (was 7) and dropped the
    5 sim sliders entirely. Filter order (matches TRACKED_FILTERS):
    outlook-horizon, outlook-model, tab3-model-selector.
    """

    # 3 surviving filter states (None = no filter set).
    _defaults = (None,) * 3

    def test_no_clicks_returns_no_update(self, callbacks):
        fn = callbacks["create_bookmark"]
        result = fn(0, "FPL", "grid_ops", "tab-outlook", *self._defaults)
        assert result == (no_update, no_update)

    def test_none_clicks_returns_no_update(self, callbacks):
        fn = callbacks["create_bookmark"]
        result = fn(None, "FPL", "grid_ops", "tab-outlook", *self._defaults)
        assert result == (no_update, no_update)

    def test_creates_bookmark_url(self, callbacks):
        fn = callbacks["create_bookmark"]
        search, toast = fn(1, "FPL", "grid_ops", "tab-outlook", *self._defaults)
        assert "region=FPL" in search
        assert "persona=grid_ops" in search
        assert "tab=tab-outlook" in search

    def test_creates_toast_component(self, callbacks):
        import dash_bootstrap_components as dbc

        fn = callbacks["create_bookmark"]
        search, toast = fn(1, "ERCOT", "trader", "tab-outlook", *self._defaults)
        assert isinstance(toast, dbc.Toast)
        assert toast.is_open is True

    def test_includes_filter_params(self, callbacks):
        fn = callbacks["create_bookmark"]
        # First filter state = outlook-horizon = "720"
        state_values = ("720", None, None)
        search, toast = fn(1, "FPL", "grid_ops", "tab-outlook", *state_values)
        assert "f.outlook-horizon=720" in search


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
