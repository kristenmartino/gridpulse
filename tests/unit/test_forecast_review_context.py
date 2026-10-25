"""Published source, plotted values and accuracy evidence must describe one series."""

from unittest.mock import patch

import pandas as pd
import pytest

from components.forecast_context import forecast_review_context
from data.forecast_identity import resolve_forecast_identity, selection_values


def payload(hours=24, baseline=True):
    return {
        "primary_model": "xgboost",
        "served_series": "seasonal-naive" if baseline else "model",
        "scored_at": "2026-10-24T00:00:00Z",
        "model_metrics": {"xgboost": {"mape": 6.5}},
        "forecasts": [
            {
                "timestamp": ts.isoformat(),
                "predicted_demand_mw": 100 + i,
                "baseline": 100 + i,
                "xgboost": 200 + i,
                "ensemble": 300 + i,
            }
            for i, ts in enumerate(pd.date_range("2026-10-24", periods=hours, freq="h", tz="UTC"))
        ],
    }


def text(node):
    if isinstance(node, (str, int, float)):
        return str(node)
    children = getattr(node, "children", [])
    return " ".join(text(c) for c in (children if isinstance(children, list) else [children]))


def test_published_baseline_controls_overview_line_peak_and_evidence():
    from components._callbacks_overview import (
        _build_overview_hero_chart,
        _build_overview_insight,
        _build_overview_model_card,
    )

    cached = payload()
    actual = pd.DataFrame({"timestamp": ["2026-10-23T23:00:00Z"], "demand_mw": [99]})
    drift = {
        "models": {
            "baseline": {"24h": {"n_7d": 8, "rolling_mape_7d": 7.5}},
            "ensemble": {"24h": {"n_7d": 8, "rolling_mape_7d": 1.2}},
        }
    }
    with (
        patch("components.forecast_context.redis_get", return_value=drift),
        patch(
            "models.model_service.get_model_metrics", return_value={"ensemble": {"mape": 2.0}}
        ) as metrics,
        patch("components._callbacks_overview._provenance_note", return_value=[]),
    ):
        fig = _build_overview_hero_chart("FPL", actual, cached)
        summary = text(_build_overview_insight("FPL", actual, "grid_ops", cached=cached))
        panel = text(_build_overview_model_card("FPL", cached))
    line = next(t for t in fig.data if t.name == "Seasonal baseline (24h)")
    assert list(line.y[1:]) == list(range(100, 124))
    assert "123 MW" in summary and "323 MW" not in summary
    assert "Seasonal baseline" in panel and "7.50%" in panel and "1.20%" not in panel
    assert "Not applicable" in panel
    assert not any(t.fill == "toself" for t in fig.data)
    metrics.assert_not_called()


def test_named_model_comparison_keeps_published_identity_but_uses_own_errors():
    drift = {
        "models": {
            "baseline": {"24h": {"n_7d": 12, "rolling_mape_7d": 9}},
            "xgboost": {"24h": {"n_7d": 12, "rolling_mape_7d": 4}},
        }
    }
    with (
        patch("components.forecast_context.redis_get", return_value=drift),
        patch("models.model_service.get_model_metrics", return_value={"xgboost": {"mape": 6.5}}),
    ):
        context = forecast_review_context("FPL", payload(), "xgboost", 24)
    assert context["display_label"] == "XGBoost"
    assert context["display_status"] == "GridPulse model comparison"
    assert context["published_label"] == "Seasonal baseline"
    assert context["error_value"] == "4.00%"
    assert context["holdout_value"] == "6.50% MAPE"
    assert "multiple leads" in context["holdout_note"]


@pytest.mark.parametrize("hours", [168, 720])
def test_long_curve_never_inherits_one_hour_or_day_ahead_accuracy(hours):
    with (
        patch("components.forecast_context.redis_get") as redis,
        patch("models.model_service.get_model_metrics", return_value={"xgboost": {"mape": 6}}),
    ):
        context = forecast_review_context("FPL", payload(hours, baseline=False), "published", hours)
    assert context["error_value"] == "Not available"
    assert "shorter leads are not substituted" in context["error_note"]
    assert context["holdout_value"] == "6.50% MAPE"
    redis.assert_not_called()


@pytest.mark.parametrize(
    "block",
    [
        None,
        {},
        {"n_7d": 5, "n_records": 100, "rolling_mape_7d": 2},
        {"n_7d": 8, "rolling_mape_7d": None},
        {"n_7d": 8, "rolling_mape_7d": float("nan")},
        {"n_7d": 8, "rolling_mape_7d": -1},
        {"n_7d": float("inf"), "rolling_mape_7d": 2},
        {"n_7d": float("-inf"), "rolling_mape_7d": 2},
    ],
)
def test_unusable_matching_score_never_borrows_other_series_or_lead(block):
    drift = {
        "models": {
            "baseline": {"24h": block, "1h": {"n_7d": 20, "rolling_mape_7d": 1}},
            "ensemble": {"24h": {"n_7d": 20, "rolling_mape_7d": 1}},
        }
    }
    with patch("components.forecast_context.redis_get", return_value=drift):
        context = forecast_review_context("FPL", payload(), "published", 24)
    assert context["error_value"] == "Not available"


def test_unknown_primary_never_guesses_xgboost_or_uses_its_metrics():
    cached = payload(baseline=False)
    del cached["primary_model"]
    for row in cached["forecasts"]:
        del row["xgboost"]
    identity = resolve_forecast_identity(cached, "xgboost")
    assert identity.metric_key is None
    with patch("models.model_service.get_model_metrics") as metrics:
        context = forecast_review_context("FPL", cached, "xgboost", 24)
    assert "source unspecified" in context["display_label"]
    assert context["error_value"] == context["holdout_value"] == "Not available"
    metrics.assert_not_called()


def test_primary_substitute_uses_same_series_metrics_not_requested_model():
    cached = payload(baseline=False)
    cached["primary_model"] = "prophet"
    for row in cached["forecasts"]:
        del row["xgboost"]
    with (
        patch("components.forecast_context.redis_get", return_value=None),
        patch("models.model_service.get_model_metrics", return_value={"xgboost": {"mape": 1}}),
    ):
        context = forecast_review_context("FPL", cached, "xgboost", 24)
    assert context["display_label"] == "Prophet"
    assert "Requested XGBoost is unavailable" in context["source_note"]
    assert context["holdout_value"] == "Not available"


@pytest.mark.parametrize("defect", ["missing", "nan", "bad-time", "short", "guard"])
def test_invalid_selected_series_is_not_filled_with_another_model(defect):
    cached = payload()
    if defect == "missing":
        del cached["forecasts"][7]["xgboost"]
    elif defect == "nan":
        cached["forecasts"][7]["xgboost"] = float("nan")
    elif defect == "bad-time":
        cached["forecasts"][7]["timestamp"] = None
    elif defect == "short":
        cached["forecasts"].pop()
    else:
        cached["horizon_guard"] = {"xgboost": {"max_ok_horizon": 0}}
    identity = resolve_forecast_identity(cached, "xgboost")
    assert selection_values(cached, identity, 24) is None
    context = forecast_review_context("FPL", cached, "xgboost", 24)
    assert context["display_status"] == "Not shown at this horizon"
    assert context["error_value"] == "Not available"


@pytest.mark.parametrize(
    "cached, requested",
    [
        (None, "published"),
        ({}, "published"),
        ({"forecasts": [None]}, "published"),
        (payload(), "prophet"),
    ],
)
def test_unavailable_forecast_has_no_attributed_accuracy(cached, requested):
    context = forecast_review_context("FPL", cached, requested, 24)
    assert context["display_label"] == "Unavailable"
    assert context["error_value"] == context["holdout_value"] == "Not available"


def test_forecast_default_and_comparison_plot_distinct_real_columns():
    import components._callbacks_forecast as fc

    cached = payload()
    with (
        patch.object(fc, "redis_get", return_value=cached),
        patch.object(fc, "_add_confidence_bands", return_value={}),
        patch.object(fc, "_add_trailing_actuals"),
    ):
        default = fc._outlook_tab_from_redis("FPL", 24, "published", None, None, "grid_ops")
        comparison = fc._outlook_tab_from_redis("FPL", 24, "xgboost", None, None, "grid_ops")
    assert list(default[0].data[0].y) == list(range(100, 124))
    assert list(comparison[0].data[0].y) == list(range(200, 224))
    assert "Seasonal baseline" in default[0].layout.title.text
    assert "XGBoost" in comparison[0].layout.title.text
    assert default[2] != comparison[2]


def test_baseline_live_error_does_not_treat_all_history_as_window_count():
    with patch(
        "components.forecast_context.redis_get",
        return_value={"models": {"baseline": {"24h": {"n_records": 6, "rolling_mape_7d": 3}}}},
    ):
        assert (
            forecast_review_context("FPL", payload(), "published", 24)["error_value"]
            == "Not available"
        )


class CallbackCapture:
    def __init__(self):
        self.functions = {}
        self.inputs = {}

    def clientside_callback(self, *args, **kwargs):
        pass

    def callback(self, *args, **kwargs):
        def capture(fn):
            self.functions[fn.__name__] = fn
            self.inputs[fn.__name__] = args
            return fn

        return capture


def test_real_forecast_evidence_callback_tracks_horizon_and_never_falls_back():
    import components._callbacks_forecast as fc

    app = CallbackCapture()
    fc.register_forecast_callbacks(app)
    callback = app.functions["update_outlook_model_card"]
    inputs = app.inputs["update_outlook_model_card"][1]
    assert any(i.component_id == "outlook-horizon" for i in inputs)
    with (
        patch.object(fc, "redis_get", return_value=payload(168)),
        patch(
            "components.forecast_context.redis_get",
            return_value={"models": {"baseline": {"24h": {"n_7d": 6, "rolling_mape_7d": 7.5}}}},
        ),
    ):
        day = text(callback("FPL", "published", "tab-outlook", "24"))
        week = text(callback("FPL", "published", "tab-outlook", "168"))
    assert "7.50%" in day
    assert "7.50%" not in week and "Not available" in week
    from dash import no_update

    assert callback("FPL", "published", "tab-models", "24") is no_update


def test_overview_callback_reads_one_forecast_snapshot_for_every_surface():
    import components._callbacks_overview as ov

    app = CallbackCapture()
    ov.register_overview_callbacks(app)
    cached = payload()
    with (
        patch.object(ov, "redis_get", return_value=cached) as redis,
        patch.object(ov, "_build_overview_hero_chart") as chart,
        patch.object(ov, "_build_overview_model_card") as panel,
        patch.object(ov, "_build_overview_insight") as summary,
    ):
        app.functions["update_overview_tab"](None, "tab-overview", "grid_ops", None, "FPL", None)
    redis.assert_called_once_with("gridpulse:forecast:FPL:1h")
    assert chart.call_args.args[-1] is cached
    assert panel.call_args.args[-1] is cached
    assert summary.call_args.args[-1] is cached


def test_published_reader_loads_payload_when_not_supplied_and_never_reads_ensemble():
    from components._callbacks_shared import _read_published_forecast_from_redis

    with patch("components._callbacks_shared.redis_get", return_value=payload()) as redis:
        result = _read_published_forecast_from_redis("FPL")
    redis.assert_called_once_with("gridpulse:forecast:FPL:1h")
    assert list(result[1]) == list(range(100, 124))


@pytest.mark.parametrize(
    "cached",
    [
        None,
        {},
        {"forecasts": [{"predicted_demand_mw": 2}]},
        {"forecasts": [{"timestamp": "bad", "predicted_demand_mw": 2}] * 24},
    ],
)
def test_published_reader_rejects_unavailable_or_malformed_payload(cached):
    from components._callbacks_shared import _read_published_forecast_from_redis

    assert _read_published_forecast_from_redis("FPL", cached) is None


def test_published_baseline_does_not_inherit_trained_model_weather_caption():
    import components._callbacks_forecast as fc

    cached = payload(720)
    with (
        patch.object(fc, "redis_get", return_value=cached),
        patch.object(fc, "_add_trailing_actuals"),
        patch.object(fc, "_add_forecast_horizon_divider") as weather_boundary,
    ):
        result = fc._outlook_tab_from_redis("FPL", 720, "published", None, None, "grid_ops")
    assert result is not None
    assert "Seasonal baseline" in result[0].layout.title.text
    assert "Open-Meteo" not in result[0].layout.title.text
    assert "climatology" not in result[0].layout.title.text
    weather_boundary.assert_not_called()
