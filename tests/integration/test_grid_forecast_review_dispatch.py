"""Exercise actual Dash wiring for authority inspection and aligned weather."""

import json
from unittest.mock import patch

import dash_bootstrap_components as dbc
import pandas as pd
import pytest
from dash import Dash, dcc, html
from plotly.utils import PlotlyJSONEncoder

from components._callbacks_alerts import _alerts_tab_from_redis
from components._callbacks_forecast import (
    _add_forecast_horizon_divider,
    register_forecast_callbacks,
)
from components._callbacks_us_grid import register_us_grid_callbacks
from components.tab_demand_outlook import layout as forecast_layout
from components.tab_us_grid import layout as grid_layout
from tests.integration.dash_driver import DashDriver, component_text
from tests.unit.test_weather_explorer_review import cache_fixture


@pytest.fixture
def driver():
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div(
        [
            dbc.Select(id="region-selector", value="FPL"),
            dbc.Select(id="persona-selector", value="grid_ops"),
            dbc.Tabs(id="dashboard-tabs", active_tab="tab-outlook"),
            dcc.Interval(id="refresh-interval"),
            dcc.Store(id="weather-store"),
            dcc.Store(id="demand-store"),
            forecast_layout(),
            grid_layout(),
        ]
    )
    register_forecast_callbacks(app)
    register_us_grid_callbacks(app)
    return DashDriver(app)


def test_forecast_weather_follows_selected_utc_targets_not_future_tail(driver):
    cache, _, _ = cache_fixture()
    with patch("components._callbacks_forecast.redis_get", side_effect=cache.get):
        result = driver.dispatch(
            "forecast-drivers-content.children",
            {
                "forecast-panel-drivers-collapse.is_open": True,
                "weather-store": "wrong prior region store",
                "region-selector": "FPL",
                "outlook-horizon": "24",
                "outlook-model": "published",
            },
            {"dashboard-tabs.active_tab": "tab-outlook"},
        )
    assert result.ok
    text = result.text("forecast-drivers-content")
    assert "24/24 target hours" in text and "without explicit weather adjustment" in text
    assert "Wind at 80 m" in text and "2026" in text
    assert "215" in text and "2026-02-11T09:00" not in text


def test_forecast_region_change_reads_keyed_actuals_never_prior_store(driver):
    cache, _, _ = cache_fixture()
    with (
        patch("components._callbacks_forecast.redis_get", side_effect=cache.get),
        patch("components._callbacks_forecast.REQUIRE_REDIS", True),
        patch("components._callbacks_forecast.redis_get_strict", side_effect=cache.get),
    ):
        result = driver.dispatch(
            "outlook-chart.figure",
            {
                "region-selector": "FPL",
                "outlook-model": "published",
                "outlook-horizon": "24",
                "dashboard-tabs.active_tab": "tab-outlook",
                "demand-store": "corrupt-prior-authority",
                "persona-selector": "grid_ops",
            },
            {"weather-store": "corrupt-prior-authority"},
        )
    assert result.ok
    figure = result.value("outlook-chart", "figure")
    assert "Seasonal baseline" in figure["data"][0]["name"]
    values = figure["data"][0]["y"]
    if isinstance(values, dict):
        import base64

        import numpy as np

        values = np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
    assert values[0] == 1000
    assert result.text("outlook-peak") == "1,023 MW"


def test_grid_filter_and_explicit_review_dont_auto_navigate(driver):
    data = {
        "FPL": {
            "current_mw": 1000,
            "prev_mw": 900,
            "hourly_pair": True,
            "latest_at": "2026-02-01T01:00Z",
        },
        "PJM": {},
    }
    with patch("components._callbacks_us_grid._collect_us_grid_region_data", return_value=data):
        result = driver.dispatch(
            "us-grid-region-grid.children",
            {
                "dashboard-tabs.active_tab": "tab-us-grid",
                "us-grid-view-toggle": "table",
                "us-grid-sort": "change",
                "us-grid-search": "FPL",
                "us-grid-group": "all",
                "us-grid-selected": "FPL",
            },
        )
    assert result.ok and "FPL" in result.text("us-grid-region-grid")
    assert "PJM" not in result.text("us-grid-region-grid")
    assert "Review forecast" in result.text("us-grid-inspector")
    assert "region-selector" not in result.payload["response"]
    review = driver.dispatch(
        "region-selector.value",
        {"us-grid-review": 1},
        {"us-grid-selected": "FPL", "us-grid-data": data},
    )
    assert review.ok and review.value("region-selector", "value") == "FPL"
    assert review.value("dashboard-tabs", "active_tab") == "tab-outlook"


def test_map_geometry_coverage_uses_source_asset_and_local_topology(driver):
    with patch(
        "components._callbacks_us_grid._collect_us_grid_region_data",
        return_value={"FPL": {}, "PJM": {}},
    ):
        result = driver.dispatch(
            "us-grid-region-grid.children",
            {
                "dashboard-tabs.active_tab": "tab-us-grid",
                "us-grid-view-toggle": "map",
                "us-grid-sort": "name",
                "us-grid-group": "all",
            },
        )
    text = result.text("us-grid-region-grid")
    assert result.ok and "2/2 filtered authorities have territory" in text
    assert "/assets/geo/" in text
    assert "properties.region" in text
    assert "AGPL-3.0" in text


def test_extended_boundary_uses_target_membership_not_fixed_384():
    import plotly.graph_objects as go

    times = pd.date_range("2026-02-01", periods=720, freq="h", tz="UTC")
    fig = go.Figure()
    assert _add_forecast_horizon_divider(fig, times, 720, times[:366])
    assert fig.layout.shapes[0].x0 == times[366].isoformat()
    assert "Current weather-cache" in fig.layout.annotations[0].text
    unknown = go.Figure()
    _add_forecast_horizon_divider(unknown, times, 720)
    assert "Nominal" in unknown.layout.annotations[0].text


def test_clockless_current_falls_back_to_bounded_region_cache():
    from data.redis_client import redis_key

    cache = {
        redis_key("alerts:FPL"): {
            "scored_at": "2026-02-01T02:00Z",
            "weather_current": {"temperature_2m": 999},
            "temperature": {"timestamps": ["2026-02-15T00:00Z"], "values": [999]},
        },
        redis_key("weather:FPL"): {
            "timestamps": ["2026-02-01T01:00Z", "2026-02-15T00:00Z"],
            "temperature_2m": [55, 999],
        },
    }
    with patch("components._callbacks_alerts.redis_get", side_effect=cache.get):
        result = _alerts_tab_from_redis("FPL")
    text = component_text(json.loads(json.dumps(result[7], cls=PlotlyJSONEncoder)))
    assert "55°F" in text and "999" not in text and "2026" in text
    assert not result[5].data


def test_headline_observed_error_keeps_coverage_fraction_outside_disclosure(driver):
    cache, _, _ = cache_fixture()
    context = {
        "display_label": "Seasonal baseline",
        "display_status": "Published forecast",
        "error_label": "Observed error · ≥24h sampled lead",
        "error_value": "11.61%",
        "coverage_summary": "6 / 168 hourly targets scored",
    }
    with (
        patch("components._callbacks_forecast.redis_get", side_effect=cache.get),
        patch("components._callbacks_forecast.forecast_review_context", return_value=context),
        patch(
            "components._callbacks_forecast.build_forecast_review",
            return_value=html.Div("Detailed categories"),
        ),
    ):
        result = driver.dispatch(
            "outlook-model-card.children",
            {
                "region-selector": "FPL",
                "outlook-model": "published",
                "dashboard-tabs.active_tab": "tab-outlook",
                "outlook-horizon": "24",
            },
        )
    assert result.ok
    children = result.value("outlook-model-card", "children")["props"]["children"]
    assert children[0]["type"] == "Div" and children[1]["type"] == "Details"
    assert "11.61%" in component_text(children[0])
    assert "6 / 168 hourly targets scored" in component_text(children[0])


def test_forecast_explorer_link_caption_distinguishes_current_and_stored_sources():
    text = component_text(json.loads(json.dumps(forecast_layout(), cls=PlotlyJSONEncoder)))
    assert "Future reviews the current cached forecast" in text
    assert "selected stored published issue with later eligible actuals" in text
    assert "source selections are retained independently" in text


def test_production_cache_miss_preserves_completed_unavailable_guard(driver):
    cache, _, _ = cache_fixture()
    from data.redis_client import redis_key

    cache.pop(redis_key("forecast:FPL:1h"))
    with (
        patch("components._callbacks_forecast.redis_get", side_effect=cache.get),
        patch("components._callbacks_forecast.REQUIRE_REDIS", True),
        patch("components._callbacks_forecast.redis_get_strict", side_effect=cache.get),
        patch("components._callbacks_forecast._pipeline_alive", return_value=True),
        patch(
            "components._callbacks_forecast._scoring_pass_completed_since_actuals",
            return_value=True,
        ),
        patch(
            "components._callbacks_forecast._run_forecast_outlook",
            return_value={"error": "unavailable"},
        ) as guarded,
    ):
        result = driver.dispatch(
            "outlook-chart.figure",
            {
                "region-selector": "FPL",
                "outlook-model": "xgboost",
                "outlook-horizon": "24",
                "dashboard-tabs.active_tab": "tab-outlook",
                "demand-store": "prior-region",
                "persona-selector": "grid_ops",
            },
            {"weather-store": "prior-region"},
        )
    assert result.ok and guarded.call_count == 0
    assert "completed" in component_text(result.value("outlook-chart", "figure")).lower()
    assert "Loading data" not in component_text(result.value("outlook-chart", "figure"))


def test_generation_net_load_uses_region_keyed_actuals_not_prior_store(driver):
    cache, _, _ = cache_fixture()
    with (
        patch("components._callbacks_forecast.redis_get", side_effect=cache.get),
        patch("components._callbacks_forecast.REQUIRE_REDIS", True),
        patch(
            "components._callbacks_forecast._build_generation_panel",
            return_value=html.Div("Observed generation"),
        ) as render,
    ):
        result = driver.dispatch(
            "forecast-generation-content.children",
            {
                "forecast-panel-generation-collapse.is_open": True,
                "region-selector": "FPL",
                "demand-store": "prior-region",
            },
            {"dashboard-tabs.active_tab": "tab-outlook"},
        )
    assert result.ok
    region, actuals = render.call_args.args
    assert region == "FPL" and "prior-region" not in actuals and "2000.0" in actuals
