"""Forecast completed states agree across source, plot and KPI callbacks."""

from unittest.mock import patch

import pytest

from data.redis_client import RedisReadError, redis_key
from tests.integration import test_grid_forecast_review_dispatch as forecast_dispatch
from tests.integration.dash_driver import component_text
from tests.unit.test_weather_explorer_review import cache_fixture


@pytest.fixture
def driver():
    return forecast_dispatch.driver.__wrapped__()


def dispatch_pair(driver, region="FPL", model="published", hours=24):
    inputs = {
        "region-selector": region,
        "outlook-model": model,
        "outlook-horizon": str(hours),
        "dashboard-tabs.active_tab": "tab-outlook",
        "demand-store": "stale-prior-region-store",
        "persona-selector": "grid_ops",
    }
    source = driver.dispatch("outlook-model-card.children", inputs)
    chart = driver.dispatch(
        "outlook-chart.figure", inputs, {"weather-store": "stale-prior-region-store"}
    )
    assert source.ok and chart.ok
    return source.text("outlook-model-card"), chart


@pytest.mark.parametrize(
    "alive,completed,label,explanation",
    [
        (False, False, "No cached forecast available", "No cached forecast is available"),
        (False, True, "No cached forecast available", "No cached forecast is available"),
        (True, False, "Warming", "completed forecast update is not yet recorded"),
        (True, True, "Forecast unavailable", "latest completed data update"),
    ],
)
def test_empty_inputs_settle_without_loading_or_inline_compute(
    driver, alive, completed, label, explanation
):
    with (
        patch("components._callbacks_forecast.REQUIRE_REDIS", True),
        patch("components._callbacks_forecast.redis_get", return_value=None),
        patch("components._callbacks_forecast.redis_get_strict", return_value=None),
        patch("components._callbacks_forecast._pipeline_alive", return_value=alive),
        patch(
            "components._callbacks_forecast._scoring_pass_completed_since_actuals",
            return_value=completed,
        ),
        patch("components._callbacks_forecast._run_forecast_outlook") as compute,
    ):
        source, result = dispatch_pair(driver)
    figure = result.value("outlook-chart", "figure")
    text = component_text(figure).replace("<br>", " ")
    assert label in source and explanation in text
    assert not figure["data"] and figure["layout"]["xaxis"]["visible"] is False
    assert "Loading" not in source + text
    for ident in (
        "outlook-peak",
        "outlook-avg",
        "outlook-min",
        "outlook-range",
        "outlook-data-through",
    ):
        assert result.text(ident) == "—"
    compute.assert_not_called()


def test_cache_read_failure_is_error_not_absence_and_does_not_expose_diagnostics(driver):
    with (
        patch("components._callbacks_forecast.REQUIRE_REDIS", True),
        patch("components._callbacks_forecast.redis_get", return_value=None),
        patch(
            "components._callbacks_forecast.redis_get_strict",
            side_effect=RedisReadError("SECRET_HOST"),
        ),
    ):
        source, result = dispatch_pair(driver)
    text = source + component_text(result.value("outlook-chart", "figure"))
    assert "Could not load forecast" in source
    assert "Forecast could not be loaded" in text
    assert "No cached forecast available" not in text and "SECRET_HOST" not in text


def test_cached_forecast_survives_missing_inputs_and_roundtrip_to_empty_region(driver):
    cache, _, _ = cache_fixture()
    # No actuals or weather are needed to display genuine cached forecast values.
    cache.pop(redis_key("actuals:FPL"))
    cache.pop(redis_key("weather:FPL"))
    with (
        patch("components._callbacks_forecast.REQUIRE_REDIS", True),
        patch("components._callbacks_forecast.redis_get", side_effect=cache.get),
        patch("components._callbacks_forecast.redis_get_strict", side_effect=cache.get),
        patch("components._callbacks_forecast._pipeline_alive", return_value=False),
    ):
        for region in ("FPL", "CAISO", "FPL"):
            source, result = dispatch_pair(driver, region)
            if region == "FPL":
                assert "Seasonal baseline" in source and result.text("outlook-peak") == "1,023 MW"
                assert result.value("outlook-chart", "figure")["data"]
            else:
                assert "No cached forecast available" in source
                assert "Seasonal baseline" not in source
                assert not result.value("outlook-chart", "figure")["data"]
                assert result.text("outlook-peak") == "—"


@pytest.mark.parametrize("model,hours", [("prophet", 24), ("published", 168)])
def test_missing_selected_model_or_horizon_never_shows_cached_peak(driver, model, hours):
    cache, _, _ = cache_fixture()
    for row in cache[redis_key("forecast:FPL:1h")]["forecasts"]:
        row.pop("prophet", None)
    with (
        patch("components._callbacks_forecast.REQUIRE_REDIS", True),
        patch("components._callbacks_forecast.redis_get", side_effect=cache.get),
        patch("components._callbacks_forecast.redis_get_strict", side_effect=cache.get),
    ):
        source, result = dispatch_pair(driver, model=model, hours=hours)
    assert "Not shown for this selection" in source
    assert not result.value("outlook-chart", "figure")["data"]
    assert result.text("outlook-peak") == "—"


def test_horizon_guard_still_withholds_the_actual_cached_series(driver):
    cache, _, _ = cache_fixture()
    cache[redis_key("forecast:FPL:1h")]["horizon_guard"] = {
        "baseline": {"max_ok_horizon": 12, "reason": "flat"}
    }
    with (
        patch("components._callbacks_forecast.REQUIRE_REDIS", True),
        patch("components._callbacks_forecast.redis_get", side_effect=cache.get),
        patch("components._callbacks_forecast.redis_get_strict", side_effect=cache.get),
    ):
        source, result = dispatch_pair(driver)
    text = component_text(result.value("outlook-chart", "figure"))
    assert "Not shown for this selection" in source and "withheld" in text
    assert not result.value("outlook-chart", "figure")["data"]
    assert result.text("outlook-peak") == "—"
