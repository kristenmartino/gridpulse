"""Retained-hour eligibility; synthetic context around the measured 598/3535 pair."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from data.hourly_scoring_actuals import hourly_actuals
from data.quality import coerce_demand_artifacts
from models.hourly_scoring import compute_hourly_scoring, export_hourly_scoring

TARGET = datetime(2026, 10, 26, 21, tzinfo=UTC)


def demand(tail=8, value=598.0):
    stamps = [TARGET + timedelta(hours=i) for i in range(-30, tail)]
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "demand_mw": [value if t == TARGET else 3500.0 for t in stamps],
            "forecast_mw": [3535.0] * len(stamps),
        }
    )


def captured():
    sampled = TARGET - timedelta(hours=24, minutes=30)
    forecast = {
        "region": "PSCO",
        "primary_model": "xgboost",
        "scored_at": sampled.isoformat(),
        "forecasts": [{"timestamp": TARGET.isoformat(), "xgboost": 4600.0, "ensemble": 4500.0}],
    }
    return compute_hourly_scoring("PSCO", None, forecast, {}, now_iso=sampled.isoformat())


def first(state):
    return next(
        s
        for s in state["samples"]
        if s["horizon"] == "24h" and s["target_ts"] == TARGET.isoformat()
    )


def test_age_alone_cannot_clear_the_same_quality_flag():
    for tail in (2, 8, 100):
        frame = demand(tail)
        before = frame.copy(deep=True)
        actuals, exclusions = hourly_actuals(frame, now=TARGET + timedelta(hours=tail))
        assert TARGET.isoformat() not in actuals
        assert exclusions[TARGET.isoformat()]["mw"] == 598
        pd.testing.assert_frame_equal(frame, before)
    # The short forecast-input guard remains unchanged; the scoring consumer
    # must not use its age-dependent exemption as actual eligibility.
    old_guard, _ = coerce_demand_artifacts(demand(8))
    assert old_guard.loc[old_guard["timestamp"] == TARGET, "demand_mw"].iloc[0] == 598


def test_future_forecast_rows_do_not_push_a_target_out_of_quality_checks():
    frame = demand(80)
    frame.loc[frame["timestamp"] > TARGET + timedelta(hours=1), "demand_mw"] = float("nan")
    actuals, exclusions = hourly_actuals(frame, now=TARGET + timedelta(hours=2))
    assert TARGET.isoformat() in exclusions
    assert max(datetime.fromisoformat(s) for s in actuals) <= TARGET + timedelta(hours=2)


def test_previously_scored_flagged_actual_is_excluded_without_erasing_evidence():
    state = captured()
    accepted = compute_hourly_scoring(
        "PSCO",
        state,
        None,
        {TARGET.isoformat(): 598},
        now_iso=(TARGET + timedelta(hours=7)).isoformat(),
    )
    before = deepcopy(accepted)
    actuals, exclusions = hourly_actuals(demand(9), now=TARGET + timedelta(hours=9))
    excluded = compute_hourly_scoring(
        "PSCO",
        accepted,
        None,
        actuals,
        now_iso=(TARGET + timedelta(hours=9)).isoformat(),
        actual_exclusions=exclusions,
    )
    assert accepted == before
    old, new = first(accepted), first(excluded)
    for field in (
        "predictions",
        "sampled_at",
        "actual",
        "actual_first_value",
        "actual_first_observed_at",
    ):
        assert new[field] == old[field]
    assert new["actual_quality_exclusion"]["mw"] == 598
    block = excluded["models"]["xgboost"]["24h"]
    assert block["n_7d"] == 0 and block["rolling_mape_7d"] is None
    assert block["coverage_7d"]["quality_guard_excluded_hours"] == 1
    assert (
        sum(
            block["coverage_7d"][k]
            for k in (
                "scored_target_hours",
                "not_yet_tracked_hours",
                "missing_sample_hours",
                "series_unavailable_hours",
                "pending_actual_hours",
                "quality_guard_excluded_hours",
                "low_actual_excluded_hours",
            )
        )
        == 168
    )
    body = export_hourly_scoring(excluded)
    assert body["actual_quality_policy"] == "retained_window_guard_v1"
    assert body["models"]["xgboost"]["24h"]["coverage_7d"]["quality_guard_excluded_hours"] == 1
    assert "samples" not in body

    # A genuine value revision, unlike aging, can clear the flag. Keep the
    # first stored value and its clock, and retain the rejection evidence.
    actuals, exclusions = hourly_actuals(demand(10, value=3600), now=TARGET + timedelta(hours=10))
    revised = compute_hourly_scoring(
        "PSCO",
        excluded,
        None,
        actuals,
        now_iso=(TARGET + timedelta(hours=10)).isoformat(),
        actual_exclusions=exclusions,
    )
    point = first(revised)
    assert point["actual"] == 3600 and point["actual_first_value"] == 598
    assert point["actual_first_observed_at"] == old["actual_first_observed_at"]
    assert point["actual_quality_exclusion"] is None
    assert point["actual_last_quality_exclusion"]["mw"] == 598
    block = revised["models"]["xgboost"]["24h"]
    assert block["n_7d"] == 1
    assert block["rolling_mape_7d"] == pytest.approx(100 * abs(4600 - 3600) / 3600)


def test_explicit_exclusion_wins_over_a_conflicting_numeric_actual():
    exclusion = {TARGET.isoformat(): {"mw": 598.0, "reason": "flagged"}}
    state = compute_hourly_scoring(
        "PSCO",
        captured(),
        None,
        {TARGET.isoformat(): 598},
        now_iso=(TARGET + timedelta(hours=1)).isoformat(),
        actual_exclusions=exclusion,
    )
    assert first(state)["actual"] is None
    assert first(state)["actual_quality_exclusion"]
    # An absent new reading does not silently clear an existing exclusion.
    retry = compute_hourly_scoring(
        "PSCO", state, None, {}, now_iso=(TARGET + timedelta(hours=2)).isoformat()
    )
    assert first(retry)["actual_quality_exclusion"]


def test_phase_forwards_retained_hour_quality_flags_without_mutating_input():
    from jobs.phases import write_hourly_scoring_metrics

    frame = demand(8)
    before = frame.copy(deep=True)
    with (
        patch("data.hourly_scoring_actuals.datetime") as clock,
        patch("data.hourly_scoring_store.update_hourly_scoring", return_value=captured()) as update,
    ):
        clock.now.return_value = TARGET + timedelta(hours=8)
        result = write_hourly_scoring_metrics("PSCO", None, frame)
    assert result.ok
    assert TARGET.isoformat() not in update.call_args.args[2]
    assert update.call_args.kwargs["actual_exclusions"][TARGET.isoformat()]["mw"] == 598
    pd.testing.assert_frame_equal(frame, before)


def test_empty_actuals_still_allow_capture_and_naive_clock_is_rejected():
    assert hourly_actuals(pd.DataFrame()) == ({}, {})
    with pytest.raises(ValueError):
        hourly_actuals(demand(), now=TARGET.replace(tzinfo=None))


def test_upgrade_does_not_silently_validate_an_old_actual_absent_from_current_source():
    old = compute_hourly_scoring(
        "PSCO",
        captured(),
        None,
        {TARGET.isoformat(): 598},
        now_iso=(TARGET + timedelta(hours=7)).isoformat(),
    )
    old.pop("actual_quality_policy")
    state = compute_hourly_scoring(
        "PSCO",
        old,
        None,
        {},
        now_iso=(TARGET + timedelta(hours=8)).isoformat(),
    )
    assert first(state)["actual"] == 598
    assert first(state)["actual_quality_pending"]
    assert state["models"]["xgboost"]["24h"]["n_7d"] == 0
    assert state["models"]["xgboost"]["24h"]["coverage_7d"]["pending_actual_hours"] == 1
    revised = compute_hourly_scoring(
        "PSCO",
        state,
        None,
        {TARGET.isoformat(): 3600},
        now_iso=(TARGET + timedelta(hours=9)).isoformat(),
    )
    assert not first(revised)["actual_quality_pending"]
    assert first(revised)["actual_first_value"] == 598
    assert revised["models"]["xgboost"]["24h"]["n_7d"] == 1
