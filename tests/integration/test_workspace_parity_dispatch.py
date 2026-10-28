"""Reviewed composition controls keep their real source and state behavior."""

from unittest.mock import patch

import pytest

from data.user_prefs import deserialize_bookmark_params, serialize_bookmark_params
from tests.integration import test_grid_forecast_review_dispatch as forecast_dispatch
from tests.unit.test_weather_explorer_review import cache_fixture


@pytest.fixture
def driver():
    return forecast_dispatch.driver.__wrapped__()


def test_hidden_history_retains_the_forecast_and_changes_no_summary_values(driver):
    cache, _, _ = cache_fixture()
    inputs = {
        "region-selector": "FPL",
        "outlook-model": "published",
        "outlook-horizon": "24",
        "dashboard-tabs.active_tab": "tab-outlook",
        "demand-store": "prior-region",
        "persona-selector": "grid_ops",
    }
    with (
        patch("components._callbacks_forecast.REQUIRE_REDIS", True),
        patch("components._callbacks_forecast.redis_get", side_effect=cache.get),
        patch("components._callbacks_forecast.redis_get_strict", side_effect=cache.get),
    ):
        shown = driver.dispatch(
            "outlook-chart.figure", inputs, {"weather-store": None, "outlook-show-history": True}
        )
        hidden = driver.dispatch(
            "outlook-chart.figure", inputs, {"weather-store": None, "outlook-show-history": False}
        )
    assert shown.ok and hidden.ok
    assert hidden.value("outlook-chart", "figure")["layout"]["yaxis"]["rangemode"] == "tozero"
    traces = hidden.value("outlook-chart", "figure")["data"]
    assert next(row for row in traces if row["name"] == "Actual")["visible"] is False
    assert (
        next(
            row for row in shown.value("outlook-chart", "figure")["data"] if row["name"] == "Actual"
        )["visible"]
        is True
    )
    assert hidden.text("outlook-peak") == shown.text("outlook-peak") == "1,023 MW"
    assert "Seasonal baseline" in traces[0]["name"]


def test_visible_ramp_uses_adjacent_selected_targets_and_qualifies_the_baseline(driver):
    cache, _, _ = cache_fixture()
    with patch("components._callbacks_forecast.redis_get", side_effect=cache.get):
        result = driver.dispatch(
            "forecast-ramp.children",
            {"region-selector": "FPL", "outlook-model": "published", "outlook-horizon": "24"},
        )
    assert result.ok
    assert "+1 MW/h" in result.text("forecast-ramp")
    assert "UTC" in result.text("forecast-ramp") and "not a dispatch instruction" in result.text(
        "forecast-ramp"
    )
    assert "no trained-model holdout" in result.text("forecast-curve-note")


def test_grid_summary_counts_only_reporting_authorities_and_valid_hour_pairs(driver):
    data = {"FPL": {"current_mw": 1000, "prev_mw": 900, "hourly_pair": True}, "PJM": {}}
    with patch("components._callbacks_us_grid._collect_us_grid_region_data", return_value=data):
        result = driver.dispatch(
            "us-grid-region-grid.children",
            {
                "dashboard-tabs.active_tab": "tab-us-grid",
                "us-grid-view-toggle": "table",
                "us-grid-sort": "change",
            },
        )
    text = result.text("us-grid-metrics-bar")
    assert "1 / 2" in text and "1.0 GW" in text
    assert "not simultaneous US demand" in text and "Adjacent valid reported hours" in text
    assert "2 of 2 tracked authorities" in result.text("us-grid-region-grid")


def test_forecast_history_and_context_controls_survive_a_view_link():
    query = serialize_bookmark_params(
        "FPL",
        "grid_ops",
        "tab-outlook",
        {"outlook-show-history": False, "forecast-context-tabs": "generation"},
    )
    filters = deserialize_bookmark_params(query)["filters"]
    assert filters["outlook-show-history"] is False
    assert filters["forecast-context-tabs"] == "generation"
