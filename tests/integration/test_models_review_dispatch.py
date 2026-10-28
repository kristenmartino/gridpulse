"""Exercise real Dash wiring: training choice must never recalculate diagnostics."""

from __future__ import annotations

from unittest.mock import patch

import dash_bootstrap_components as dbc
import pytest
from dash import Dash, dcc, html

from components._callbacks_models import register_models_callbacks
from components.tab_models import layout
from tests.integration.dash_driver import DashDriver
from tests.unit.test_models_review import diagnostic_payload


@pytest.fixture()
def driver() -> DashDriver:
    app = Dash(__name__)
    app.layout = html.Div(
        [
            dcc.Dropdown(id="region-selector", value="ERCOT"),
            dbc.Tabs(
                id="dashboard-tabs",
                active_tab="tab-models",
                children=[dbc.Tab(label="Models", tab_id="tab-models")],
            ),
            dcc.Store(id="demand-store"),
            layout(),
        ]
    )
    register_models_callbacks(app)
    return DashDriver(app)


def test_training_filter_dispatch_never_reads_diagnostics_or_gets_forecasts(
    driver: DashDriver,
) -> None:
    payload = {"region": "ERCOT", "model_metrics": {"xgboost": {"mape": 4.5}}}
    with (
        patch("components._callbacks_models.redis_get", return_value=payload) as read,
        patch(
            "models.model_service.get_forecasts", side_effect=AssertionError("must not recompute")
        ) as predict,
    ):
        result = driver.dispatch(
            "tab3-metrics-table.children",
            {
                "tab3-model-selector": ["xgboost"],
                "region-selector": "ERCOT",
                "dashboard-tabs.active_tab": "tab-models",
            },
        )
        assert result.ok and "4.50%" in result.text("tab3-metrics-table")
        assert "Prophet" not in result.text("tab3-metrics-table")
        assert read.call_count == 1 and "forecast:ERCOT:1h" in read.call_args.args[0]
        predict.assert_not_called()
    assert "tab3-model-selector" not in {
        dep["id"]
        for key, spec in driver._app.callback_map.items()
        if "tab3-residuals-time.figure" in key
        for dep in spec["inputs"]
    }


def test_fixed_cached_diagnostics_dispatch_and_exact_page(driver: DashDriver) -> None:
    with (
        patch(
            "components._callbacks_models.redis_get", return_value=diagnostic_payload(61)
        ) as read,
        patch(
            "models.model_service.get_forecasts", side_effect=AssertionError("must not recompute")
        ) as predict,
    ):
        result = driver.dispatch(
            "tab3-residuals-time.figure",
            {
                "demand-store": None,
                "dashboard-tabs.active_tab": "tab-models",
                "region-selector": "ERCOT",
            },
        )
        assert result.ok
        assert len(result.value("tab3-residuals-pred", "figure")["data"][0]["y"]) == 61
        assert "61 finite residual" in result.text("tab3-diagnostics-scope")
        data = result.value("models-diagnostics-data", "data")
        assert len(data["rows"]) == 61
        read.assert_called_once()
        predict.assert_not_called()
    exact = driver.dispatch(
        "tab3-diagnostics-evidence.children",
        {"models-diagnostics-data": data, "models-diagnostics-page": 2},
    )
    assert exact.ok and "51–61 of 61" in exact.text("tab3-diagnostics-evidence")
    assert data["rows"][-1]["time"] in exact.text("tab3-diagnostics-evidence")


@pytest.mark.parametrize("require_redis", [True, False])
def test_cold_diagnostics_never_falls_back_to_compute(
    driver: DashDriver, require_redis: bool
) -> None:
    with (
        patch("components._callbacks_models.redis_get", return_value=None) as read,
        patch("config.REQUIRE_REDIS", require_redis),
        patch(
            "models.model_service.get_forecasts", side_effect=AssertionError("forbidden")
        ) as predict,
    ):
        result = driver.dispatch(
            "tab3-residuals-time.figure",
            {"dashboard-tabs.active_tab": "tab-models", "region-selector": "ERCOT"},
        )
        assert result.ok and not result.value("tab3-residuals-time", "figure")["data"]
        assert result.value("models-diagnostics-data", "data")["rows"] == []
        scope = result.text("tab3-diagnostics-scope")
        assert "Horizon unavailable" in scope and "0 finite residual samples" in scope
        assert "no qualified UTC targets" in scope
        assert "timestamps omit offsets" not in scope
        read.assert_called_once()
        predict.assert_not_called()


def test_inactive_tab_does_not_read_cache(driver: DashDriver) -> None:
    with patch("components._callbacks_models.redis_get") as read:
        result = driver.dispatch(
            "tab3-residuals-time.figure", {"dashboard-tabs.active_tab": "tab-overview"}
        )
        assert result.prevented
        read.assert_not_called()


def test_native_sections_and_bases_are_wired(driver: DashDriver) -> None:
    assert not (driver.callback_output_ids() - driver.layout_ids())
    shown = driver.dispatch("models-backtest-section.style", {"models-section": "backtest"})
    assert shown.ok and shown.value("models-backtest-section", "style") == {}
    assert shown.value("models-observed-section", "style") == {"display": "none"}
    base = driver.dispatch("models-first-hour-panel.style", {"models-observed-basis": "first"})
    assert base.ok and base.value("models-first-hour-panel", "style") == {}
    assert base.value("models-drift-panel", "style") == {"display": "none"}


def test_region_cache_change_clamps_exact_table_page(driver: DashDriver) -> None:
    with patch(
        "components._callbacks_models.redis_get", return_value=diagnostic_payload(7)
    ) as read:
        result = driver.dispatch(
            "tab3-residuals-time.figure",
            {"dashboard-tabs.active_tab": "tab-models", "region-selector": "ERCOT"},
            state={"models-diagnostics-page": 17},
        )
        assert result.ok
        assert result.value("models-diagnostics-page", "value") == 1
        assert result.value("models-diagnostics-page", "options") == [
            {"label": "Rows 1–7", "value": 1}
        ]
        read.assert_called_once()


def test_foreign_region_payload_is_not_displayed(driver: DashDriver) -> None:
    with patch(
        "components._callbacks_models.redis_get", return_value=diagnostic_payload(7)
    ) as read:
        result = driver.dispatch(
            "tab3-residuals-time.figure",
            {"dashboard-tabs.active_tab": "tab-models", "region-selector": "CAISO"},
        )
        assert result.ok and result.value("models-diagnostics-data", "data")["rows"] == []
        assert not result.value("tab3-residuals-time", "figure")["data"]
        read.assert_called_once()


def test_observed_first_hour_uses_injected_cached_holdout_not_local_models(
    driver: DashDriver,
) -> None:
    def cache(key: str) -> dict | None:
        if "forecast:ERCOT:1h" in key:
            return {"region": "ERCOT", "model_metrics": {"xgboost": {"mape": 8.1}}}
        if key.endswith("drift:ERCOT"):
            return {
                "models": {"xgboost": {"n_7d": 28, "rolling_mape_7d": 3.2, "rolling_mape_30d": 3.1}}
            }
        return None

    with (
        patch("components._callbacks_models.redis_get", side_effect=cache) as read,
        patch("components._callbacks_models._read_vintage_summary", return_value=None),
        patch(
            "models.model_service.get_model_metrics",
            side_effect=AssertionError("no local resolver"),
        ) as local,
    ):
        result = driver.dispatch(
            "models-first-hour-panel.children",
            {
                "region-selector": "ERCOT",
                "dashboard-tabs.active_tab": "tab-models",
                "demand-store": None,
            },
        )
        assert result.ok and "8.10%" in result.text("models-first-hour-panel")
        assert "3.20%" in result.text("models-first-hour-panel")
        assert "n (7d)" in result.text("models-first-hour-panel")
        assert "n (30d)" in result.text("models-first-hour-panel")
        assert "Hourly-sampled error is unavailable" in result.text("models-drift-panel")
        assert read.call_count >= 4
        local.assert_not_called()


def test_investigation_rail_uses_current_region_source_clocks_and_real_categories(driver):
    from data.redis_client import redis_key

    forecast = {
        "region": "ERCOT",
        "served_series": "seasonal-naive",
        "forecasts": [{"timestamp": "2026-10-28T01:00Z", "predicted_demand_mw": 1000}],
    }
    sample = {
        "capture_started_at": "2026-10-25T20:38:00Z",
        "last_updated_at": "2026-10-28T01:25:00Z",
        "models": {
            "baseline": {
                "24h": {
                    "first_target_at": "2026-10-26T21:00:00Z",
                    "coverage_7d": {
                        "scored_target_hours": 6,
                        "possible_target_hours": 168,
                        "quality_guard_excluded_hours": 0,
                    },
                }
            }
        },
    }
    cache = {
        redis_key("forecast:ERCOT:1h"): forecast,
        redis_key("drift_hourly:ERCOT:summary"): sample,
    }
    with patch("components._callbacks_models.redis_get", side_effect=cache.get):
        current = driver.dispatch(
            "models-published-source.children",
            {"region-selector": "ERCOT", "dashboard-tabs.active_tab": "tab-models"},
        )
        empty = driver.dispatch(
            "models-published-source.children",
            {"region-selector": "CAISO", "dashboard-tabs.active_tab": "tab-models"},
        )
    assert current.ok and "Published forecast: Seasonal baseline" in current.text(
        "models-published-source"
    )
    assert "25 Oct 2026, 20:38 UTC" in current.text("models-investigation-context")
    assert "26 Oct 2026, 21:00 UTC" in current.text("models-investigation-context")
    coverage = current.text("models-coverage-context")
    assert "Coverage reference: Seasonal baseline" in coverage
    assert "Quality-guard excluded 0" in coverage
    assert "Low-actual exclusions Unavailable" in coverage
    assert "region=ERCOT" in str(current.value("models-forecast-evidence-link", "children"))
    assert empty.ok and "Published forecast: Unavailable" in empty.text("models-published-source")
    assert "25 Oct 2026" not in empty.text("models-investigation-context")
    assert "Seasonal baseline" not in empty.text("models-coverage-context")


def test_training_rail_navigation_changes_only_models_section(driver):
    for ident, expected in (
        ("models-review-observed", "observed"),
        ("models-review-backtest", "backtest"),
    ):
        result = driver.dispatch("models-section.value", {ident + ".n_clicks": 1})
        assert result.ok and result.value("models-section", "value") == expected
        assert set(result.payload["response"]) == {"models-section"}
