"""Sampling, provenance and coverage contracts, with explicitly synthetic values."""

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from redis.exceptions import WatchError

from models.hourly_scoring import BASIS, compute_hourly_scoring, export_hourly_scoring

T0 = datetime(2026, 10, 1, 0, 10, tzinfo=UTC)


def forecast(now=T0, origin=None, value=110.0):
    origin = origin or now.replace(minute=0, second=0, microsecond=0)
    return {
        "region": "FPL",
        "primary_model": "xgboost",
        "scored_at": (now - timedelta(hours=1)).isoformat(),
        "forecasts": [
            {
                "timestamp": (origin + timedelta(hours=i)).isoformat(),
                "predicted_demand_mw": value,
                "xgboost": value,
                "ensemble": value + 1,
            }
            for i in range(200)
        ],
    }


def partition(block):
    c = block["coverage_7d"]
    assert (
        sum(
            c[k]
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
        == c["possible_target_hours"]
        == 168
    )


def test_targets_follow_observation_clock_when_origin_stalls_or_jumps():
    state = None
    # The origin is frozen for 12h, then jumps. Every wall-clock slot still
    # has a prediction in a previously published 200-hour curve.
    for i in range(14):
        now = T0 + timedelta(hours=i)
        origin = T0.replace(minute=0) if i < 12 else now.replace(minute=0)
        state = compute_hourly_scoring(
            "FPL", state, forecast(now, origin), {}, now_iso=now.isoformat()
        )
    samples = [s for s in state["samples"] if s["horizon"] == "24h"]
    assert len(samples) == 14
    assert all(24 <= s["sample_lead_hours"] < 25 for s in samples)
    assert all(
        datetime.fromisoformat(b["target_ts"]) - datetime.fromisoformat(a["target_ts"])
        == timedelta(hours=1)
        for a, b in zip(samples, samples[1:], strict=False)
    )
    assert all(s["predictions"]["xgboost"] == 110 for s in samples)


def test_retry_freezes_first_observed_prediction_and_does_not_mutate_prior_state():
    original = compute_hourly_scoring("FPL", None, forecast(), {}, now_iso=T0.isoformat())
    before = deepcopy(original)
    retry = T0 + timedelta(minutes=20)
    state = compute_hourly_scoring(
        "FPL", original, forecast(retry, value=999), {}, now_iso=retry.isoformat()
    )
    assert original == before
    assert len(state["samples"]) == 3
    assert all(
        s["sampled_at"] == T0.isoformat() and s["predictions"]["xgboost"] == 110
        for s in state["samples"]
    )


def test_known_prediction_is_reconciled_even_when_actual_arrives_after_120_hours():
    state = compute_hourly_scoring("FPL", None, forecast(), {}, now_iso=T0.isoformat())
    sample = next(s for s in state["samples"] if s["horizon"] == "24h")
    target = datetime.fromisoformat(sample["target_ts"])
    later = target + timedelta(hours=130)
    state = compute_hourly_scoring(
        "FPL", state, None, {sample["target_ts"]: 100}, now_iso=later.isoformat()
    )
    recovered = next(
        s
        for s in state["samples"]
        if s["target_ts"] == sample["target_ts"] and s["horizon"] == "24h"
    )
    assert recovered["predictions"]["xgboost"] == 110
    assert recovered["actual"] == 100
    assert state["models"]["xgboost"]["24h"]["n_7d"] == 1
    assert state["models"]["xgboost"]["24h"]["rolling_mape_7d"] == 10
    partition(state["models"]["xgboost"]["24h"])


def test_actual_revisions_keep_forecast_and_first_observation_intact():
    state = compute_hourly_scoring("FPL", None, forecast(), {}, now_iso=T0.isoformat())
    sample = state["samples"][0]
    later = datetime.fromisoformat(sample["target_ts"]) + timedelta(hours=1)
    state = compute_hourly_scoring(
        "FPL", state, None, {sample["target_ts"]: 100}, now_iso=later.isoformat()
    )
    state = compute_hourly_scoring(
        "FPL",
        state,
        None,
        {sample["target_ts"]: 110},
        now_iso=(later + timedelta(hours=1)).isoformat(),
    )
    revised = next(
        s
        for s in state["samples"]
        if s["target_ts"] == sample["target_ts"] and s["horizon"] == sample["horizon"]
    )
    assert revised["actual_first_value"] == 100
    assert revised["actual"] == revised["predictions"]["xgboost"] == 110
    assert revised["sampled_at"] == sample["sampled_at"]


def test_startup_and_missing_sampling_slots_are_separate_from_waiting_for_actuals():
    state = compute_hourly_scoring("FPL", None, forecast(), {}, now_iso=T0.isoformat())
    assert state["models"]["xgboost"]["24h"]["coverage_7d"]["not_yet_tracked_hours"] == 168
    now = T0 + timedelta(hours=27)
    state = compute_hourly_scoring("FPL", state, None, {}, now_iso=now.isoformat())
    coverage = state["models"]["xgboost"]["24h"]["coverage_7d"]
    assert coverage["pending_actual_hours"] == 1
    assert coverage["missing_sample_hours"] == 2
    assert coverage["not_yet_tracked_hours"] == 165
    partition(state["models"]["xgboost"]["24h"])


def test_future_actuals_are_not_scored_and_unknown_generation_time_stays_unknown():
    payload = forecast()
    payload.pop("scored_at")
    state = compute_hourly_scoring("FPL", None, payload, {}, now_iso=T0.isoformat())
    target = state["samples"][0]["target_ts"]
    state = compute_hourly_scoring(
        "FPL", state, payload, {target: 100}, now_iso=(T0 + timedelta(minutes=1)).isoformat()
    )
    assert state["samples"][0]["actual"] is None
    assert state["samples"][0]["generated_at"] is None


def test_baseline_uses_published_column_and_withheld_model_is_not_sampled():
    payload = forecast()
    payload["served_series"] = "seasonal-naive"
    for row in payload["forecasts"]:
        row["predicted_demand_mw"] = 200
    payload["horizon_guard"] = {"ensemble": {"max_ok_horizon": 24}}
    state = compute_hourly_scoring("FPL", None, payload, {}, now_iso=T0.isoformat())
    for sample in state["samples"]:
        assert sample["predictions"]["baseline"] == 200
        assert sample["predictions"]["xgboost"] == 110
        assert "ensemble" not in sample["predictions"]


@pytest.mark.parametrize("bad", [None, "2026-10-01T00:10:00", "bad"])
def test_invalid_explicit_clock_is_not_invented(bad):
    if bad is None:
        # None means a real runtime UTC clock, so the malformed case is empty.
        bad = ""
    with pytest.raises(ValueError):
        compute_hourly_scoring("FPL", None, forecast(), {}, now_iso=bad)


def test_legacy_state_and_backwards_clocks_cannot_contaminate_new_history():
    with pytest.raises(ValueError):
        compute_hourly_scoring("FPL", {"models": {}}, forecast(), {}, now_iso=T0.isoformat())
    state = compute_hourly_scoring("FPL", None, forecast(), {}, now_iso=T0.isoformat())
    with pytest.raises(ValueError):
        compute_hourly_scoring(
            "FPL", state, forecast(), {}, now_iso=(T0 - timedelta(hours=1)).isoformat()
        )


def test_export_never_publishes_internal_sample_values_or_unknown_fields():
    state = compute_hourly_scoring("FPL", None, forecast(), {}, now_iso=T0.isoformat())
    state["private"] = "secret"
    state["models"]["xgboost"]["24h"]["coverage_7d"]["private"] = "secret"
    body = export_hourly_scoring(state)
    assert body["sampling_basis"] == BASIS
    assert "samples" not in body
    assert "secret" not in json.dumps(body)


def test_watch_conflict_keeps_a_concurrent_first_sample():
    from data.hourly_scoring_store import update_hourly_scoring

    # Runtime now is fixed only at the store boundary; the forecast value is
    # already captured by a different execution before this retry succeeds.
    now = T0
    prior = compute_hourly_scoring(
        "FPL", None, forecast(now, value=123), {}, now_iso=now.isoformat()
    )
    pipe = MagicMock()
    pipe.get.side_effect = [None, json.dumps(prior)]
    pipe.execute.side_effect = [WatchError("conflict"), [True]]
    client = MagicMock()
    client.pipeline.return_value.__enter__.return_value = pipe
    with (
        patch("data.hourly_scoring_store._get_redis", return_value=client),
        patch("data.hourly_scoring_store.datetime") as clock,
    ):
        clock.now.return_value = now + timedelta(minutes=1)
        state = update_hourly_scoring("FPL", forecast(now, value=999), {})
    assert all(s["predictions"]["xgboost"] == 123 for s in state["samples"])
    assert pipe.set.call_args.kwargs["ex"] == 11 * 86400
    ledger_write, summary_write = pipe.set.call_args_list[-2:]
    assert not ledger_write.args[0].endswith(":summary")
    assert summary_write.args[0] == ledger_write.args[0] + ":summary"
    assert "samples" in json.loads(ledger_write.args[1])
    assert "samples" not in json.loads(summary_write.args[1])


def test_phase_records_absence_of_actuals_and_reports_storage_failure():
    from jobs.phases import write_hourly_scoring_metrics

    with patch(
        "data.hourly_scoring_store.update_hourly_scoring", side_effect=RuntimeError("write failed")
    ) as update:
        result = write_hourly_scoring_metrics("FPL", forecast(), pd.DataFrame())
    assert not result.ok
    assert update.call_args.args[2] == {}


def test_legacy_resolved_target_cannot_be_resampled_with_a_later_prediction():
    from models.drift import compute_horizon_drift_payload

    origin = T0.replace(minute=0)
    payload = forecast(T0, origin)
    target = (origin + timedelta(hours=24)).isoformat()
    first = compute_horizon_drift_payload("FPL", None, payload, {}, now_iso=T0.isoformat())
    resolved = compute_horizon_drift_payload(
        "FPL", first, payload, {target: 100}, now_iso=(T0 + timedelta(hours=25)).isoformat()
    )
    later = forecast(T0 + timedelta(hours=26), origin, value=999)
    result = compute_horizon_drift_payload(
        "FPL", resolved, later, {target: 100}, now_iso=(T0 + timedelta(hours=26)).isoformat()
    )
    records = result["models"]["xgboost"]["24h"]["records"]
    assert next(r["p"] for r in records if r["ts"] == target) == 110
    assert not any(s["target_ts"] == target and s["horizon"] == "24h" for s in result["pending"])


def test_values_are_selected_by_target_timestamp_when_the_input_origin_is_old():
    origin = T0.replace(minute=0) - timedelta(hours=9)
    payload = forecast(T0, origin)
    for i, row in enumerate(payload["forecasts"]):
        row["predicted_demand_mw"] = row["xgboost"] = 100 + i
    state = compute_hourly_scoring("FPL", None, payload, {}, now_iso=T0.isoformat())
    point = next(s for s in state["samples"] if s["horizon"] == "24h")
    assert point["predictions"]["xgboost"] == 134
    assert point["forecast_start_at"] == origin.isoformat()


def test_a_full_week_has_168_scores_despite_repeated_and_jumping_origins():
    state = None
    for i in range(241):
        now = T0 + timedelta(hours=i)
        origin = T0.replace(minute=0) + timedelta(hours=(i // 14) * 14)
        actuals = {
            (now.replace(minute=0) - timedelta(hours=k)).isoformat(): 100.0 for k in range(168)
        }
        state = compute_hourly_scoring(
            "FPL", state, forecast(now, origin), actuals, now_iso=now.isoformat()
        )
    for label in ("24h", "48h", "72h"):
        block = state["models"]["xgboost"][label]
        assert block["n_7d"] == 168
        assert block["coverage_7d"]["missing_sample_hours"] == 0
        assert block["coverage_7d"]["not_yet_tracked_hours"] == 0
        partition(block)
    assert len(state["samples"]) <= 3 * (168 + 73)


def test_forecast_context_never_combines_clock_scores_with_legacy_offsets():
    from components.forecast_context import forecast_review_context

    legacy = {"models": {"xgboost": {"24h": {"n_7d": 100, "rolling_mape_7d": 1.0}}}}
    clock = {
        "schema_version": 1,
        "sampling_basis": BASIS,
        "models": {
            "xgboost": {
                "24h": {
                    "n_7d": 6,
                    "rolling_mape_7d": 10.0,
                    "future_sample_hours": 1,
                    "coverage_7d": {
                        "as_of": T0.isoformat(),
                        "not_yet_tracked_hours": 162,
                        "missing_sample_hours": 0,
                        "series_unavailable_hours": 0,
                        "pending_actual_hours": 0,
                        "low_actual_excluded_hours": 0,
                    },
                }
            }
        },
    }
    with patch(
        "components.forecast_context.redis_get",
        side_effect=lambda key: clock if "drift_hourly:" in key else legacy,
    ):
        context = forecast_review_context("FPL", forecast(), "published", 24)
    assert context["error_value"] == "10.00%"
    assert "sampled ≥24h" in context["error_label"]
    assert context["coverage_summary"] == "6 / 168 hourly targets scored"
    clock["models"]["xgboost"]["24h"].update(n_7d=0, rolling_mape_7d=None)
    with patch(
        "components.forecast_context.redis_get",
        side_effect=lambda key: clock if "drift_hourly:" in key else legacy,
    ):
        context = forecast_review_context("FPL", forecast(), "published", 24)
    assert context["error_value"] == "1.00%"
    assert "legacy" in context["error_label"]
    assert "two samples are not combined" in context["sampling_note"]


def test_model_view_keeps_legacy_scores_in_a_separate_expander():
    from components._callbacks_models import _build_horizon_drift_panel

    clock = compute_hourly_scoring("FPL", None, forecast(), {}, now_iso=T0.isoformat())
    legacy = {"models": {"xgboost": {"24h": {"n_7d": 100, "rolling_mape_7d": 1.0}}}}
    with patch(
        "components._callbacks_models.redis_get",
        side_effect=lambda key: clock if "drift_hourly:" in key else legacy,
    ):
        panel = _build_horizon_drift_panel("FPL")
    legacy_section = next(
        child
        for child in panel.children
        if child.__class__.__name__ == "Details"
        and "legacy" in str(child.children[0].children).lower()
    )
    assert "legacy" in legacy_section.children[0].children.lower()
    assert "separate sample" in legacy_section.children[0].children
    assert not getattr(legacy_section, "open", False)
    legacy_text = json.dumps(legacy_section, default=lambda item: item.to_plotly_json())
    sampled_text = json.dumps(
        [child for child in panel.children if child is not legacy_section],
        default=lambda item: item.to_plotly_json(),
    )
    assert "1.00%" in legacy_text and "1.00%" not in sampled_text
    assert "0 / 168 scored" in sampled_text


def test_bad_history_and_failed_transactions_are_not_reported_as_success():
    from data.hourly_scoring_store import update_hourly_scoring

    pipe = MagicMock()
    pipe.get.return_value = "not-json"
    client = MagicMock()
    client.pipeline.return_value.__enter__.return_value = pipe
    with (
        patch("data.hourly_scoring_store._get_redis", return_value=client),
        pytest.raises(ValueError),
    ):
        update_hourly_scoring("FPL", forecast(), {})
    pipe.set.assert_not_called()


def test_legacy_phase_captures_predictions_without_actuals_and_checks_write_result():
    from jobs.phases import write_horizon_drift_metrics

    payload = forecast(datetime.now(UTC))
    with (
        patch("jobs.phases._read_window_strict", return_value=None),
        patch("data.redis_client.redis_set", return_value=True) as write,
    ):
        result = write_horizon_drift_metrics("FPL", payload, pd.DataFrame())
    assert result.ok
    assert len(write.call_args.args[1]["pending"]) == 3
    with (
        patch("jobs.phases._read_window_strict", return_value=None),
        patch("data.redis_client.redis_set", return_value=False),
    ):
        failed = write_horizon_drift_metrics("FPL", payload, pd.DataFrame())
    assert not failed.ok
    assert "not persisted" in failed.error
