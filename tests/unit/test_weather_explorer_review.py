"""Real cache selection, retained quality, gaps and investigation state contracts."""

from copy import deepcopy
from unittest.mock import patch

import pandas as pd
import pytest

from components.explorer_state import clean_state, normalize_window, validate_start
from components.weather_explorer import figure
from components.weather_explorer_data import compare_issues, load_explorer_data, read_issues
from data.forecast_publications import prepare_publication, publication_summary
from data.redis_client import redis_key
from data.user_prefs import deserialize_bookmark_params, serialize_bookmark_params


def cache_fixture():
    stamps = pd.date_range("2026-02-01", periods=217, freq="h", tz="UTC")
    generated = stamps[192] + pd.Timedelta(minutes=2)
    future = pd.date_range(stamps[192], periods=30, freq="h", tz="UTC")
    forecast = {
        "region": "FPL",
        "scored_at": generated.isoformat(),
        "served_series": "seasonal-naive",
        "primary_model": "xgboost",
        "forecasts": [
            {
                "timestamp": t.isoformat(),
                "predicted_demand_mw": 1000 + i,
                "baseline": 1000 + i,
                "xgboost": 1200 + i,
                "prophet": 1300 + i,
                "arima": 1400 + i,
                "ensemble": 1500 + i,
            }
            for i, t in enumerate(future)
        ],
    }
    issue = prepare_publication(forecast)
    issue["observed_published_at"] = (generated + pd.Timedelta(seconds=2)).isoformat()
    weather = pd.date_range(stamps[0], periods=250, freq="h", tz="UTC")
    cache = {
        "actuals:FPL": {
            "timestamps": [t.isoformat() for t in stamps],
            "demand_mw": [2000.0] * len(stamps),
            "scored_at": stamps[-1].isoformat(),
        },
        "weather:FPL": {
            "timestamps": [t.isoformat() for t in weather],
            "temperature_2m": list(range(len(weather))),
            "wind_speed_80m": [2.0] * len(weather),
            "shortwave_radiation": [10.0] * len(weather),
            "scored_at": stamps[-1].isoformat(),
        },
        "forecast:FPL:1h": forecast,
        "publication_index:FPL": [publication_summary(issue)],
        f"publication:FPL:{issue['publication_id']}": issue,
    }
    cache = {redis_key(k): v for k, v in cache.items()}
    return cache, stamps[-1], issue


def test_exact_utc_join_and_source_identity_no_mutation():
    cache, now, issue = cache_fixture()
    original = deepcopy(cache)
    with patch("models.model_service.get_forecasts", side_effect=AssertionError("no model call")):
        data = load_explorer_data("FPL", reader=cache.get, now=now)
    assert len(data["past"]) == 217
    assert data["past"][-1]["temperature_f"] == 216
    assert len(data["future"]) == 5
    assert data["future"]["published"]["metric_key"] == "baseline"
    assert data["future"]["xgboost"]["rows"][0]["forecast_mw"] == 1200
    assert data["future"]["published"]["rows"][0]["generated_before_target"] is False
    rows = data["error"][issue["publication_id"]]
    assert rows[0]["timestamp"] == "2026-02-09T01:00:00+00:00"
    assert rows[0]["error_mw"] == 999
    assert data["quality_inputs"] == "legacy_without_operator_forecast"
    assert cache == original


def test_stamped_quality_exclusions_are_counted_and_never_compared():
    cache, now, issue = cache_fixture()
    actual = cache[redis_key("actuals:FPL")]
    actual["artifact_excluded"] = [
        {"ts": "2026-02-09T01:00:00Z", "mw": 2, "reason": "producer flag"}
    ]
    actual["demand_mw"][-24] = None
    data = load_explorer_data("FPL", reader=cache.get, now=now)
    assert "2026-02-09T01:00:00+00:00" in data["actual_exclusions"]
    assert all(
        r["timestamp"] != "2026-02-09T01:00:00+00:00"
        for r in data["error"][issue["publication_id"]]
    )


def test_retained_guard_detects_positive_history_artifact_and_operator_signal():
    cache, now, issue = cache_fixture()
    actual = cache[redis_key("actuals:FPL")]
    actual["demand_mw"][-20] = 2
    actual["forecast_mw"] = [2000.0] * len(actual["timestamps"])
    data = load_explorer_data("FPL", reader=cache.get, now=now)
    assert data["quality_inputs"] == "operator_forecast_provided"
    assert data["operator_forecast_finite_hours"] == 168
    assert data["actual_exclusions"]
    assert all(r["actual_mw"] != 2 for r in data["error"][issue["publication_id"]])


def test_missing_region_and_missing_receipt_never_relabel_or_load_forever():
    cache, now, issue = cache_fixture()
    cache.pop(redis_key(f"publication:FPL:{issue['publication_id']}"))
    data = load_explorer_data("FPL", reader=cache.get, now=now)
    assert data["issues"][0]["receipt_status"] == "unavailable"
    assert data["error"] == {}
    other = load_explorer_data("PJM", reader=cache.get, now=now)
    assert not other["past"] and not other["future"] and not other["issues"]


def test_lazy_receipts_expose_full_bounded_index():
    cache, _, issue = cache_fixture()
    index = []
    for i in range(12):
        x = publication_summary(issue)
        x["publication_id"] = f"{i:064x}"
        index.append(x)
    cache[redis_key("publication_index:FPL")] = index
    rows = read_issues("FPL", cache.get, (f"{11:064x}",))
    assert len(rows) == 12
    assert rows[5]["receipt_status"] == "not_requested"
    assert rows[11]["receipt_status"] == "unavailable"


def test_all_window_uses_elapsed_time_and_gaps_break_plot():
    state = clean_state({"region": "FPL", "mode": "error"}, "FPL")
    rows = [
        {"timestamp": t, "actual_mw": v}
        for t, v in [("2026-02-01T00:00:00+00:00", 10), ("2026-02-01T05:00:00+00:00", 20)]
    ]
    data = {"issues": [{"publication_id": "a"}], "error": {"a": rows}}
    state, selected = normalize_window(data, state)
    assert selected == rows
    plot = figure(rows, [("Actual", "actual_mw", "blue")], "test")
    assert list(plot.data[0].y) == [10, None, 20]
    assert plot.data[0].connectgaps is False


@pytest.mark.parametrize("invalid", ["", "bad", "2025-01-01", "2026-02-02T00:01"])
def test_bad_interactive_date_does_not_silently_clamp(invalid):
    rows = [{"timestamp": "2026-02-02T00:00:00+00:00"}]
    assert validate_start(invalid, rows)


def test_timestamp_offset_canonicalization_and_region_reset():
    raw = {
        "region": "FPL",
        "mode": "past",
        "windows": {
            "past": {
                "start": "2026-02-01T01:00:00-05:00",
                "hours": 24,
                "cursor": "2026-02-01T01:00:00-05:00",
            }
        },
    }
    state = clean_state(raw, "FPL")
    assert state["windows"]["past"]["start"] == "2026-02-01T06:00:00+00:00"
    assert clean_state(raw, "PJM")["windows"]["past"]["start"] == ""


def test_share_roundtrip_keeps_published_explorer_models_grid_and_sensitivity():
    state = clean_state(
        {
            "region": "FPL",
            "mode": "future",
            "source": "arima",
            "windows": {
                "future": {"start": "2026-02-01T02:00Z", "hours": 24, "cursor": "2026-02-01T05:00Z"}
            },
        },
        "FPL",
    )
    filters = {
        "outlook-model": "published",
        "explorer-state": state,
        "models-section": "backtest",
        "models-observed-basis": "first",
        "us-grid-view-toggle": "map",
        "us-grid-search": "Florida & sun",
        "us-grid-selected": "FPL",
        "meeting-mode-store": "true",
    }
    result = deserialize_bookmark_params(
        serialize_bookmark_params(
            "FPL", "grid_ops", "tab-alerts", filters, {"forecast-scn-temp": 3}
        )
    )
    assert result["filters"] == filters
    assert result["sim_sliders"] == {"forecast-scn-temp": 3}
    assert deserialize_bookmark_params("?f.outlook-model=invalid&forecast-scn-temp=nan") == {}


def test_run_comparison_strictly_after_both_issue_clocks():
    _, _, a = cache_fixture()
    b = deepcopy(a)
    b["timing"]["generated_at"] = "2026-02-09T04:02:00Z"
    b["observed_published_at"] = "2026-02-09T04:03:00Z"
    b["forecast"] = [{**r, "demand_mw": r["demand_mw"] + 12} for r in b["forecast"]]
    result = compare_issues(a, b)
    assert result[0]["timestamp"] == "2026-02-09T05:00:00+00:00"
    assert all(r["change_mw"] == 12 for r in result)


def test_empty_grid_filter_and_selection_are_explicit_roundtrip_states():
    filters = {"us-grid-search": "", "us-grid-selected": None, "us-grid-view-toggle": "table"}
    query = serialize_bookmark_params("FPL", "grid_ops", "tab-us-grid", filters)
    restored = deserialize_bookmark_params(query)
    assert restored["filters"]["us-grid-search"] == ""
    assert restored["filters"]["us-grid-selected"] == ""
    assert deserialize_bookmark_params("?region=&tab=&f.outlook-model=") == {}


def test_source_message_filters_share_without_unstable_selection_index():
    filters = {"conditions-message-search": "Wind", "conditions-message-severity": "warning"}
    restored = deserialize_bookmark_params(
        serialize_bookmark_params("FPL", "grid_ops", "tab-alerts", filters)
    )
    assert restored["filters"] == filters
    cleared = deserialize_bookmark_params(
        serialize_bookmark_params(
            "FPL",
            "grid_ops",
            "tab-alerts",
            {"conditions-message-search": "", "conditions-message-severity": "all"},
        )
    )
    assert cleared["filters"]["conditions-message-search"] == ""
    assert deserialize_bookmark_params("?f.conditions-message-severity=severe") == {}
