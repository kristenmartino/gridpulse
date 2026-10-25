"""Coverage and clocks must not invent completeness, lead time, or source history."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from components.forecast_context import forecast_review_context
from data.forecast_evidence import baseline_selection, forecast_timing
from models.drift import DriftRecord, _horizon_rollup_block


def test_unknown_clocks_do_not_get_invented_or_misidentified():
    timing = forecast_timing({"scored_at": "2026-10-25T01:02:03", "forecasts": []}, 24)
    assert timing["generated_at"] is None
    assert timing["operator_issued_at"] is None
    assert timing["input_anchor_at"] is None
    selection = baseline_selection({"served_series": "seasonal-naive"})
    assert selection["reason"] is None and selection["selected_since"] is None


def test_coverage_and_score_exclude_future_and_left_boundary_targets():
    now = datetime(2026, 10, 25, 18, tzinfo=UTC)
    records = [
        DriftRecord((now + timedelta(hours=offset)).isoformat(), 110, 100, 10)
        for offset in [-168, -2, -1, 0, 1]
    ]
    block = _horizon_rollup_block(records, now.isoformat(), "24h", {})
    assert block["n_7d"] == 3
    assert block["coverage_7d"]["scored_target_hours"] == 3
    assert block["coverage_7d"]["unscored_target_hours"] == 165
    assert block["coverage_7d"]["scored_end"] == now.isoformat()
    assert block["rolling_mape_7d"] == 10


def test_complete_window_has_168_targets_not_169():
    now = datetime(2026, 10, 25, 18, tzinfo=UTC)
    records = [DriftRecord((now - timedelta(hours=i)).isoformat(), 100, 100, 0) for i in range(169)]
    coverage = _horizon_rollup_block(records, now.isoformat(), "24h", {})["coverage_7d"]
    assert coverage["scored_target_hours"] == coverage["possible_target_hours"] == 168
    assert coverage["unscored_target_hours"] == 0


def test_all_retained_records_are_not_a_seven_day_denominator():
    payload = {
        "primary_model": "xgboost",
        "forecasts": [
            {"timestamp": f"2026-10-25T{i:02d}:00:00+00:00", "predicted_demand_mw": 100}
            for i in range(24)
        ],
    }
    with patch(
        "components.forecast_context.redis_get",
        return_value={"models": {"xgboost": {"24h": {"n_records": 720, "rolling_mape_7d": 2}}}},
    ):
        context = forecast_review_context("FPL", payload, "published", 24)
    assert context["error_value"] == "Not available"
    assert "720" not in context["coverage_summary"]


def test_ui_uses_same_series_coverage_and_qualifies_horizon():
    now = datetime(2026, 10, 25, 18, tzinfo=UTC)
    records = [DriftRecord((now - timedelta(hours=i)).isoformat(), 110, 100, 10) for i in range(10)]
    block = _horizon_rollup_block(records, now.isoformat(), "24h", {})
    payload = {
        "primary_model": "xgboost",
        "served_series": "seasonal-naive",
        "served_reason": "recorded baseline decision",
        "scored_at": now.isoformat(),
        "forecasts": [
            {"timestamp": (now + timedelta(hours=i)).isoformat(), "predicted_demand_mw": 100}
            for i in range(24)
        ],
    }
    with patch(
        "components.forecast_context.redis_get",
        return_value={"models": {"baseline": {"24h": block}, "ensemble": {"24h": {"n_7d": 168}}}},
    ):
        context = forecast_review_context("FPL", payload, "published", 24)
    assert context["coverage_summary"] == "10 / 168 hourly targets scored"
    assert context["error_value"] == "10.00%"
    assert "not an exact publication-to-target lead" in context["error_note"]
    assert "recorded baseline decision" in context["baseline_note"]
    assert "first began is not recorded" in context["baseline_note"]
