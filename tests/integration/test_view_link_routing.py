"""Explicit URL view state wins over the hidden legacy persona/defaults."""

import pytest
from dash import Dash

from components.callbacks import register_callbacks
from components.layout import build_layout
from data.user_prefs import serialize_bookmark_params
from tests.integration.dash_driver import DashDriver


@pytest.fixture
def driver():
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = build_layout()
    register_callbacks(app)
    return DashDriver(app)


def test_generated_grid_link_survives_legacy_persona_and_saved_defaults(driver):
    query = serialize_bookmark_params(
        "FPL",
        "grid_ops",
        "tab-us-grid",
        {"us-grid-search": "Florida", "us-grid-view-toggle": "table"},
    )
    restored = driver.dispatch("dashboard-tabs.active_tab", {"url.search": query})
    assert restored.ok
    assert restored.value("dashboard-tabs", "active_tab") == "tab-us-grid"
    assert restored.value("us-grid-search", "value") == "Florida"
    legacy = driver.dispatch("dashboard-tabs.active_tab", {"persona-selector": "grid_ops"})
    assert legacy.prevented
    saved = driver.dispatch(
        "dashboard-tabs.active_tab",
        {"url.pathname": "/"},
        {
            "user-prefs-store": {"tab": "tab-overview", "persona": "grid_ops", "region": "FPL"},
            "url.search": query,
        },
    )
    assert saved.prevented


def test_history_restores_filled_then_cleared_grid_state(driver):
    filled = serialize_bookmark_params(
        "FPL", "grid_ops", "tab-us-grid", {"us-grid-search": "Florida", "us-grid-selected": "FPL"}
    )
    cleared = serialize_bookmark_params(
        "FPL", "grid_ops", "tab-us-grid", {"us-grid-search": "", "us-grid-selected": None}
    )
    for query, search, selection in (
        (filled, "Florida", "FPL"),
        (cleared, "", ""),
        (filled, "Florida", "FPL"),
    ):
        result = driver.dispatch("dashboard-tabs.active_tab", {"url.search": query})
        assert result.ok and result.value("dashboard-tabs", "active_tab") == "tab-us-grid"
        assert result.value("us-grid-search", "value") == search
        assert result.value("us-grid-selected", "data") == selection


def test_run_window_restores_independently_of_outlook_horizon(driver):
    query = serialize_bookmark_params(
        "FPL",
        "grid_ops",
        "tab-outlook",
        {
            "outlook-horizon": "168",
            "published-review-window": "24",
            "published-review-mode": "runs",
        },
    )
    restored = driver.dispatch("dashboard-tabs.active_tab", {"url.search": query})
    assert restored.ok
    assert restored.value("outlook-horizon", "value") == "168"
    assert restored.value("published-review-window", "value") == "24"
    assert restored.value("published-review-mode", "active_tab") == "runs"
    malformed = driver.dispatch(
        "dashboard-tabs.active_tab", {"url.search": "?tab=tab-outlook&f.published-review-window=25"}
    )
    assert malformed.ok and "published-review-window" not in malformed.payload["response"]
