"""Overview inspection must preserve the published series and missing evidence."""

from unittest.mock import patch

import pandas as pd
import pytest
from dash import html

from components.forecast_context import forecast_review_context
from components.overview_review import (
    build_review_evidence,
    build_review_sidebar,
    build_review_summary,
)
from tests.unit.test_forecast_review_context import payload, text


def nodes(component):
    yield component
    children = getattr(component, "children", None)
    for child in children if isinstance(children, list) else [children]:
        if child is not None and not isinstance(child, (str, int, float)):
            yield from nodes(child)


@pytest.mark.parametrize(
    "coverage, expected",
    [
        (None, None),
        ({}, None),
        ({"scored_target_hours": 0, "possible_target_hours": 168}, (0, 168)),
        ({"scored_target_hours": 43, "possible_target_hours": 168}, (43, 168)),
        ({"scored_target_hours": 170, "possible_target_hours": 168}, None),
        ({"scored_target_hours": -1, "possible_target_hours": 168}, None),
        ({"scored_target_hours": True, "possible_target_hours": 168}, None),
        ({"scored_target_hours": 43, "possible_target_hours": "168"}, None),
    ],
)
def test_coverage_fraction_only_uses_valid_recorded_counts(coverage, expected):
    block = {"n_7d": 8, "rolling_mape_7d": 6.4}
    if coverage is not None:
        block["coverage_7d"] = coverage
    with patch(
        "components.forecast_context.redis_get",
        side_effect=lambda key: (
            {"models": {"baseline": {"24h": block}}} if "drift_horizon" in key else None
        ),
    ):
        context = forecast_review_context("FPL", payload(), "published", 24)
    panel = build_review_sidebar(context)
    bars = [n for n in nodes(panel) if getattr(n, "role", None) == "img"]
    if expected is None:
        assert context["coverage_scored"] is None
        assert context["coverage_possible"] is None
        assert not bars
        assert "No coverage fraction is inferred" in text(panel)
    else:
        assert (context["coverage_scored"], context["coverage_possible"]) == expected
        assert len(bars) == 1
        assert f"{expected[1] - expected[0]} not scored" in text(panel)
        assert "not when missing targets occurred" in text(panel)


def test_hourly_sampler_changes_score_and_coverage_together():
    sampled = {
        "schema_version": 1,
        "sampling_basis": "hourly_observation_before_target",
        "models": {"baseline": {"24h": {"n_7d": 14, "rolling_mape_7d": 5.2}}},
    }
    with patch(
        "components.forecast_context.redis_get",
        side_effect=lambda key: sampled if "drift_hourly" in key else None,
    ):
        context = forecast_review_context("FPL", payload(), "published", 24)
    assert context["error_value"] == "5.20%"
    assert "sampled ≥24h" in context["error_label"]
    assert (context["coverage_scored"], context["coverage_possible"]) == (14, 168)
    visible = text(build_review_evidence(context, payload()))
    assert "not the original publication lead" in visible
    assert "Legacy forecast-start-offset records are not included" in visible


def test_summary_uses_same_published_peak_and_never_model_comparison_column():
    cached = payload()
    with patch("components.forecast_context.redis_get", return_value=None):
        context = forecast_review_context("FPL", cached, "published", 24)
    visible = text(build_review_summary(context, cached))
    assert "123" in visible and "323" not in visible
    assert "Seasonal baseline" in visible
    assert "Not available" in visible


def test_target_table_excludes_bridge_and_exposes_gaps_without_filling():
    cached = payload()
    cached["forecasts"][3]["predicted_demand_mw"] = None
    cached["forecasts"][4]["predicted_demand_mw"] = float("nan")
    cached["forecasts"][5]["timestamp"] = ["malformed"]
    with patch("components.forecast_context.redis_get", return_value=None):
        context = forecast_review_context("FPL", cached, "published", 24)
    table = next(
        n for n in nodes(build_review_evidence(context, cached)) if isinstance(n, html.Tbody)
    )
    assert len(table.children) == 24
    assert "24 Oct 2026, 00:00 UTC" in text(table.children[0])
    assert "Unavailable" in text(table.children[3])
    assert "Unavailable" in text(table.children[4])
    assert "Timestamp unavailable" in text(table.children[5])
    assert "203" not in text(table.children[3])


def test_guard_withheld_targets_are_not_exposed_as_valid_values():
    cached = payload()
    cached["horizon_guard"] = {"baseline": {"max_ok_horizon": 0}}
    with patch("components.forecast_context.redis_get", return_value=None):
        context = forecast_review_context("FPL", cached, "published", 24)
    visible = text(build_review_evidence(context, cached))
    assert "targets are withheld" in visible
    assert "24 Oct 2026, 00:00 UTC" not in visible
    assert "Unavailable" in text(build_review_summary(context, cached))


def test_history_control_changes_clock_window_but_not_forecast_or_band_identity():
    from components._callbacks_overview import _build_overview_hero_chart

    # Gaps make sample-count slicing differ from a truthful 24h clock window.
    times = pd.date_range("2026-10-20", periods=100, freq="2h", tz="UTC")
    demand = pd.DataFrame({"timestamp": times, "demand_mw": 100})
    cached = payload()
    day = _build_overview_hero_chart("FPL", demand, cached, history_hours=24)
    week = _build_overview_hero_chart("FPL", demand, cached, history_hours=168)
    assert len(day.data[0].x) == 13
    assert len(week.data[0].x) > len(day.data[0].x)
    day_forecast = next(trace for trace in day.data if trace.name.endswith("(24h)"))
    week_forecast = next(trace for trace in week.data if trace.name.endswith("(24h)"))
    assert list(day_forecast.y[1:]) == list(week_forecast.y[1:]) == list(range(100, 124))
    assert len(day_forecast.x) == 25  # 24 true targets plus one continuity point
    assert pd.Timestamp(day_forecast.x[0]) == times[-1]
    peak = next(trace for trace in day.data if trace.name == "Published forecast peak")
    assert list(peak.y) == [123]
    boundary = next(shape for shape in day.layout.shapes if shape.type == "line")
    assert pd.Timestamp(boundary.x0) == pd.Timestamp(cached["forecasts"][0]["timestamp"])
    assert day.layout.uirevision != week.layout.uirevision
    assert not any(t.fill == "toself" for t in day.data)


def test_evidence_choices_are_native_disclosures_and_both_initially_reachable():
    cached = payload()
    with patch("components.forecast_context.redis_get", return_value=None):
        context = forecast_review_context("FPL", cached, "published", 24)
    panel = build_review_evidence(context, cached)
    choices = panel.children
    assert len(choices) == 2
    assert all(isinstance(choice, html.Details) for choice in choices)
    assert all(isinstance(choice.children[0], html.Summary) for choice in choices)
    assert [choice.children[0].children for choice in choices] == [
        "Scoring summary",
        "Forecast targets",
    ]
    assert all(not getattr(choice, "open", False) for choice in choices)
    # The value is the second column; mobile can hide the repeated series only.
    table = next(n for n in nodes(choices[1]) if isinstance(n, html.Table))
    assert [cell.children for cell in table.children[0].children.children] == [
        "Target hour · UTC",
        "Forecast · MW",
        "Published series",
    ]


def test_generation_context_stays_outside_wrapping_plot_legend():
    from components._callbacks_overview import _build_overview_hero_chart

    cached = payload()
    demand = pd.DataFrame({"timestamp": ["2026-10-23T23:00:00Z"], "demand_mw": [99]})
    figure = _build_overview_hero_chart("FPL", demand, cached)
    assert not any("forecast scored" in str(note.text) for note in figure.layout.annotations)
    assert figure.layout.legend.yanchor == "bottom"
    assert figure.layout.legend.y > 1
    with patch("components.forecast_context.redis_get", return_value=None):
        context = forecast_review_context("FPL", cached, "published", 24)
    assert "Generated: 2026-10-24 00:00 UTC" in text(build_review_summary(context, cached))


def test_stale_forecast_warning_is_visible_outside_compact_disclosures():
    from datetime import UTC, datetime, timedelta

    from components._callbacks_overview import _build_overview_insight
    from config import FRESHNESS_FRESH_MAX_AGE_HOURS

    cached = payload()
    cached["scored_at"] = (
        datetime.now(UTC) - timedelta(hours=FRESHNESS_FRESH_MAX_AGE_HOURS + 5)
    ).isoformat()
    demand = pd.DataFrame({"timestamp": ["2026-10-23T23:00:00Z"], "demand_mw": [99]})
    with patch("components._callbacks_overview._provenance_note", return_value=[]):
        insight = _build_overview_insight("FPL", demand, "grid_ops", cached=cached, compact=True)
    visible = " ".join(
        text(child) for child in insight.children if not isinstance(child, html.Details)
    )
    assert "Forecast last refreshed" in visible


def test_naive_target_clock_stays_unavailable_in_table_and_peak():
    cached = payload()
    cached["forecasts"][-1]["timestamp"] = "2026-10-24T23:00:00"
    with patch("components.forecast_context.redis_get", return_value=None):
        context = forecast_review_context("FPL", cached, "published", 24)
    table = next(
        n for n in nodes(build_review_evidence(context, cached)) if isinstance(n, html.Tbody)
    )
    assert table.children[-1].children[0].children == "Timestamp unavailable"
    assert table.children[-1].children[1].children == "123"
    peak = build_review_summary(context, cached).children[0].children[0]
    assert "123" in text(peak)
    assert "Target time unavailable" in text(peak)
    assert "23:00 UTC" not in text(peak)


def test_aware_offset_target_clock_normalizes_to_utc_in_table_and_peak():
    cached = payload()
    cached["forecasts"][-1]["timestamp"] = "2026-10-24T23:00:00-04:00"
    with patch("components.forecast_context.redis_get", return_value=None):
        context = forecast_review_context("FPL", cached, "published", 24)
    table = next(
        n for n in nodes(build_review_evidence(context, cached)) if isinstance(n, html.Tbody)
    )
    assert table.children[-1].children[0].children == "25 Oct 2026, 03:00 UTC"
    assert table.children[-1].children[1].children == "123"
    peak = build_review_summary(context, cached).children[0].children[0]
    assert "Sun 25 Oct, 03:00 UTC" in text(peak)
    assert "Target time unavailable" not in text(peak)
