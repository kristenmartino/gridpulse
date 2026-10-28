"""Real Dash dispatch validates Explorer control/render and region wiring."""

from unittest.mock import patch

import dash_bootstrap_components as dbc
import pytest
from dash import Dash, dcc, html

from components._callbacks_explorer import register_explorer_callbacks
from components.explorer_state import clean_state
from components.tab_alerts import layout
from components.tab_demand_outlook import layout as forecast_layout
from components.weather_explorer_data import load_explorer_data
from tests.integration.dash_driver import DashDriver
from tests.unit.test_weather_explorer_review import cache_fixture


@pytest.fixture
def driver():
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div(
        [
            dbc.Select(id="region-selector", value="FPL"),
            dbc.Tabs(id="dashboard-tabs", active_tab="tab-alerts"),
            dcc.Interval(id="refresh-interval"),
            layout(),
            forecast_layout(),
        ]
    )
    register_explorer_callbacks(app)
    return DashDriver(app)


def test_published_actual_comparison_uses_one_genuine_issue_without_requiring_a_second(driver):
    cache, now, issue = cache_fixture()
    identity = issue["publication_id"]
    data = load_explorer_data("FPL", reader=cache.get, now=now, selected_ids=[identity])
    result = driver.dispatch(
        "published-review-content.children",
        {
            "explorer-data": data,
            "published-review-earlier": identity,
            "published-review-later": identity,
            "region-selector": "FPL",
            "published-review-mode.active_tab": "actuals",
        },
    )
    assert result.ok
    text = result.text("published-review-content")
    assert "Eligible actual matches" in text and "Comparison window · UTC" in text
    assert "official ≥24h observation-lead score" in text
    assert "Choose two different" not in text
    assert "Exact eligible matches" in text


def test_run_comparison_still_requires_distinct_immutable_issues(driver):
    cache, now, issue = cache_fixture()
    identity = issue["publication_id"]
    data = load_explorer_data("FPL", reader=cache.get, now=now, selected_ids=[identity])
    result = driver.dispatch(
        "published-review-content.children",
        {
            "explorer-data": data,
            "published-review-earlier": identity,
            "published-review-later": identity,
            "region-selector": "FPL",
            "published-review-mode.active_tab": "runs",
        },
    )
    assert result.ok
    assert "Choose two different published issues" in result.text("published-review-content")


def inputs(data, state, **changes):
    values = {
        "explorer-data": data,
        "explorer-state": state,
        "conditions-section": "explorer",
        "explorer-mode": state["mode"],
        "explorer-span": state["windows"][state["mode"]]["hours"],
        "explorer-variable": state["variable"],
        "explorer-source": state["source"],
        "explorer-issue": state["issue"],
        "explorer-start.value": state["windows"][state["mode"]]["start"],
        "region-selector": "FPL",
        "dashboard-tabs.active_tab": "tab-alerts",
    }
    values.update(changes)
    return values


def test_span_preserves_period_active_mode_noop_and_invalid_date_restores(driver):
    cache, now, _ = cache_fixture()
    data = load_explorer_data("FPL", reader=cache.get, now=now)
    state = clean_state(
        {
            "region": "FPL",
            "windows": {
                "past": {"start": "2026-02-03T00:00Z", "hours": 168, "cursor": "2026-02-03T03:00Z"}
            },
        },
        "FPL",
    )
    result = driver.dispatch(
        "explorer-span.options",
        inputs(data, state, **{"explorer-span": 24}),
        triggered=["explorer-span.value"],
    )
    assert result.ok
    state = result.value("explorer-state", "data")
    assert state["windows"]["past"] == {
        "start": "2026-02-03T00:00:00+00:00",
        "hours": 24,
        "cursor": "2026-02-03T03:00:00+00:00",
    }
    active = driver.dispatch(
        "explorer-span.options", inputs(data, state), triggered=["explorer-mode.value"]
    )
    assert active.ok and active.value("explorer-start", "value") == "2026-02-03T00:00"
    bad = driver.dispatch(
        "explorer-span.options",
        inputs(data, state, **{"explorer-start.value": ""}),
        triggered=["explorer-start.value"],
    )
    assert bad.ok and "last valid window" in bad.text("explorer-date-feedback")
    assert bad.value("explorer-start", "value") == "2026-02-03T00:00"
    assert bad.value("explorer-start", "invalid") is True


def test_cache_load_scoped_and_requested_missing_receipt_not_loading_forever(driver):
    cache, now, issue = cache_fixture()
    cache.pop(next(k for k in cache if k.endswith(issue["publication_id"])))
    data = load_explorer_data("FPL", reader=cache.get, now=now)
    state = clean_state({"region": "FPL", "mode": "error", "issue": issue["publication_id"]}, "FPL")
    result = driver.dispatch(
        "explorer-summary.children",
        {"explorer-data": data, "explorer-state": state, "region-selector": "FPL"},
    )
    assert result.ok and "receipt is unavailable" in result.text("explorer-summary")
    wrong = driver.dispatch(
        "explorer-summary.children",
        {"explorer-data": data, "explorer-state": state, "region-selector": "PJM"},
    )
    assert wrong.ok and "no cached target" in wrong.text("explorer-summary")
    assert not wrong.value("explorer-demand", "figure")["data"]


def test_mode_roundtrip_keeps_dates_source_and_future_pregeneration_marker(driver):
    cache, now, _ = cache_fixture()
    data = load_explorer_data("FPL", reader=cache.get, now=now)
    state = clean_state(
        {
            "region": "FPL",
            "source": "arima",
            "windows": {
                "past": {"start": "2026-02-03T00:00Z", "hours": 24},
                "future": {"start": "2026-02-09T00:00Z", "hours": 24},
            },
        },
        "FPL",
    )
    future = driver.dispatch(
        "explorer-span.options",
        inputs(data, state, **{"explorer-mode": "future"}),
        triggered=["explorer-mode.value"],
    )
    new = future.value("explorer-state", "data")
    assert (
        new["source"] == "arima" and new["windows"]["past"]["start"] == "2026-02-03T00:00:00+00:00"
    )
    result = driver.dispatch(
        "explorer-summary.children",
        {"explorer-data": data, "explorer-state": new, "region-selector": "FPL"},
    )
    assert result.ok and "before generation" in result.text("explorer-summary")
    assert "1,400" in result.text("explorer-exact") or "1400" in result.text("explorer-exact")


def test_loader_runs_only_cache_adapter_when_region_changes(driver):
    cache, now, _ = cache_fixture()
    data = load_explorer_data("FPL", reader=cache.get, now=now)
    state = clean_state(None, "FPL")
    with patch(
        "components._callbacks_explorer.load_explorer_data",
        return_value={
            "region": "PJM",
            "past": [],
            "future": {},
            "issues": [],
            "error": {},
            "sources": [],
        },
    ) as read:
        result = driver.dispatch(
            "explorer-span.options",
            inputs(data, state, **{"region-selector": "PJM"}),
            triggered=["region-selector.value"],
        )
        assert result.ok and result.value("explorer-data", "data")["region"] == "PJM"
        read.assert_called_once_with("PJM", selected_ids=())


def test_coalesced_date_blur_and_span_apply_both_intentions(driver):
    cache, now, _ = cache_fixture()
    data = load_explorer_data("FPL", reader=cache.get, now=now)
    state = clean_state(
        {"region": "FPL", "windows": {"past": {"start": "2026-02-03T00:00Z", "hours": 24}}}, "FPL"
    )
    result = driver.dispatch(
        "explorer-span.options",
        inputs(
            data,
            state,
            **{
                "explorer-start.value": "2026-02-04T00:00",
                "explorer-start.n_blur": 1,
                "explorer-span": 168,
            },
        ),
        triggered=["explorer-start.value", "explorer-start.n_blur", "explorer-span.value"],
    )
    assert result.ok
    updated = result.value("explorer-state", "data")
    assert updated["windows"]["past"]["start"] == "2026-02-04T00:00:00+00:00"
    assert updated["windows"]["past"]["hours"] == 168
    reverse = driver.dispatch(
        "explorer-span.options",
        inputs(data, updated, **{"explorer-start.value": "2026-02-02T00:00", "explorer-span": 24}),
        triggered=["explorer-start.n_blur", "explorer-span.value"],
    )
    assert reverse.ok
    assert reverse.value("explorer-state", "data")["windows"]["past"]["hours"] == 24
    assert reverse.value("explorer-start", "value") == "2026-02-02T00:00"


def test_raw_datetime_commit_invalid_does_not_hide_behind_stale_debounced_value(driver):
    cache, now, _ = cache_fixture()
    data = load_explorer_data("FPL", reader=cache.get, now=now)
    state = clean_state(
        {"region": "FPL", "windows": {"past": {"start": "2026-02-03T00:00Z", "hours": 24}}}, "FPL"
    )
    result = driver.dispatch(
        "explorer-span.options",
        inputs(
            data,
            state,
            **{
                "explorer-date-edit": {
                    "value": "",
                    "scope": {
                        "region": "FPL",
                        "mode": "past",
                        "window_start": "2026-02-03T00:00",
                        "source": "published",
                    },
                    "nonce": 1,
                }
            },
        ),
        triggered=["explorer-date-edit.data"],
    )
    assert result.ok and result.value("explorer-start", "invalid")
    assert result.value("explorer-start", "value") == "2026-02-03T00:00"
    assert "last valid window" in result.text("explorer-date-feedback")


def test_date_commit_owner_cannot_leak_old_mode_or_region(driver):
    cache, now, _ = cache_fixture()
    data = load_explorer_data("FPL", reader=cache.get, now=now)
    state = clean_state(
        {
            "region": "FPL",
            "windows": {
                "past": {"start": "2026-02-03T00:00Z", "hours": 24},
                "future": {"start": "2026-02-09T02:00Z", "hours": 24},
            },
        },
        "FPL",
    )
    edit = {
        "value": "2026-02-02T00:00",
        "scope": {
            "region": "FPL",
            "mode": "past",
            "window_start": "2026-02-03T00:00",
            "source": "published",
        },
        "nonce": 1,
    }
    result = driver.dispatch(
        "explorer-span.options",
        inputs(
            data,
            state,
            **{
                "explorer-mode": "future",
                "explorer-date-edit": edit,
                "explorer-start.value": "2026-02-02T00:00",
            },
        ),
        triggered=["explorer-date-edit.data", "explorer-start.n_blur", "explorer-mode.value"],
    )
    assert result.ok
    changed = result.value("explorer-state", "data")
    assert (
        changed["mode"] == "future"
        and changed["windows"]["future"]["start"] == "2026-02-09T02:00:00+00:00"
    )
    assert changed["windows"]["past"]["start"] == "2026-02-03T00:00:00+00:00"


def test_toolbar_scope_range_and_unit_follow_mode_and_current_region(driver):
    cache, now, _ = cache_fixture()
    data = load_explorer_data("FPL", reader=cache.get, now=now)
    state = clean_state({"region": "FPL", "variable": "temperature_f"}, "FPL")
    result = driver.dispatch(
        "explorer-series-scope.children",
        {"explorer-data": data, "explorer-state": state, "region-selector": "FPL"},
    )
    assert result.ok and "Historical model-weather context" in result.text("explorer-series-scope")
    assert result.text("explorer-weather-unit") == "°F"
    assert "Available past targets" in result.text("explorer-available-range")
    assert "UTC" in result.text("explorer-available-range")
    stale = driver.dispatch(
        "explorer-series-scope.children",
        {"explorer-data": data, "explorer-state": state, "region-selector": "CAISO"},
    )
    assert stale.ok and "No source targets" in stale.text("explorer-available-range")
    assert "2026" not in stale.text("explorer-available-range")


def test_guarded_future_range_is_stored_evidence_not_displayable_targets(driver):
    cache, now, _ = cache_fixture()
    data = load_explorer_data("FPL", reader=cache.get, now=now)
    state = clean_state({"region": "FPL", "mode": "future", "source": "published"}, "FPL")
    data["future"]["published"]["max_horizon"] = 12
    result = driver.dispatch(
        "explorer-available-range.children",
        {"explorer-data": data, "explorer-state": state, "region-selector": "FPL"},
    )
    assert result.ok
    note = result.text("explorer-available-range")
    assert "Stored future target range" in note and "24h horizon is withheld" in note
    assert "not displayed" in note and "Available future targets" not in note


@pytest.mark.parametrize(
    "hours,count,changed", [(None, 24, 1), ("168", 168, 7), ("720", 718, 29), ("bad", 24, 1)]
)
def test_revision_scope_metrics_plot_and_table_agree(driver, hours, count, changed):
    from tests.unit.test_publication_revision_window import issue_pair

    earlier, later = issue_pair()
    data = {"region": "FPL", "issues": [earlier, later]}
    result = driver.dispatch(
        "published-review-content.children",
        {
            "explorer-data": data,
            "published-review-earlier": "earlier",
            "published-review-later": "later",
            "region-selector": "FPL",
            "published-review-mode.active_tab": "runs",
            "published-review-window": hours,
        },
    )
    assert result.ok
    text = result.text("published-review-content")
    assert f"{changed} of {count}" in text
    assert "Peak demand change" in text and "Peak timing change" in text
    assert "+1,003 MW" in text and "repeating daily baseline" in text
    assert f"Exact shared future targets · {count}" in text
    if hours == "720":
        assert "Only 718 shared hourly targets available" in text
    nodes = result.value("published-review-content", "children")
    graph = next(node for node in nodes if node.get("type") == "Graph")
    traces = [
        trace for trace in graph["props"]["figure"]["data"] if trace["name"] != "Selected hour"
    ]
    assert len(traces) == 2
    assert all(len(trace["x"]) == count for trace in traces)
    assert all(len(trace["y"]) == count for trace in traces)
    assert traces[0]["x"] == traces[1]["x"]

    def table_bodies(node):
        if isinstance(node, list):
            return [body for child in node for body in table_bodies(child)]
        if isinstance(node, dict):
            if node.get("type") == "Tbody":
                return [node["props"]["children"]]
            return table_bodies(node.get("props", {}).get("children"))
        return []

    bodies = table_bodies(nodes)
    assert len(bodies) == 1 and len(bodies[0]) == count


def test_actual_mode_hides_window_control_and_does_not_trim_matches(driver):
    import pandas as pd

    from data.redis_client import redis_key

    cache, now, issue = cache_fixture()
    actual = cache[redis_key("actuals:FPL")]
    extra = pd.date_range(now + pd.Timedelta(hours=1), periods=6, freq="h")
    actual["timestamps"].extend(t.isoformat() for t in extra)
    actual["demand_mw"].extend([2000.0] * len(extra))
    now = extra[-1]
    identity = issue["publication_id"]
    data = load_explorer_data("FPL", reader=cache.get, now=now, selected_ids=[identity])
    values = {
        "explorer-data": data,
        "published-review-earlier": identity,
        "published-review-later": None,
        "region-selector": "FPL",
        "published-review-mode.active_tab": "actuals",
        "published-review-window": "24",
    }
    result = driver.dispatch("published-review-content.children", values)
    assert result.ok
    assert len(data["error"][identity]) > 24
    assert f"Exact eligible matches · {len(data['error'][identity])}" in result.text(
        "published-review-content"
    )
    assert "Peak demand change" not in result.text("published-review-content")
    visible = driver.dispatch(
        "published-review-window-control.style", {"published-review-mode.active_tab": "runs"}
    )
    hidden = driver.dispatch(
        "published-review-window-control.style", {"published-review-mode.active_tab": "actuals"}
    )
    assert visible.value("published-review-window-control", "style") == {}
    assert hidden.value("published-review-window-control", "style") == {"display": "none"}


def test_revision_small_change_does_not_round_to_zero(driver):
    from tests.unit.test_publication_revision_window import issue_pair

    a, b = issue_pair()
    a["forecast"] = a["forecast"][:4]
    b["forecast"] = [{**r, "demand_mw": r["demand_mw"] + 0.001} for r in a["forecast"]]
    result = driver.dispatch(
        "published-review-content.children",
        {
            "explorer-data": {"region": "FPL", "issues": [a, b]},
            "published-review-earlier": "earlier",
            "published-review-later": "later",
            "region-selector": "FPL",
            "published-review-mode.active_tab": "runs",
        },
    )
    assert result.ok
    assert "+<1 MW" in result.text("published-review-content")
    assert "2 of 2" in result.text("published-review-content")


@pytest.mark.parametrize(
    "case,expected",
    [
        ("empty", "Published issue history is not available"),
        ("region", "Loading this authority"),
        ("receipt", "receipt is unavailable"),
        ("loading", "Loading selected cached issues"),
        ("clock", "generation clock is unavailable"),
        ("order", "earlier issue must precede"),
        ("no_overlap", "no shared targets strictly after"),
    ],
)
def test_revision_unavailable_guards_never_emit_zero_metrics(driver, case, expected):
    from tests.unit.test_publication_revision_window import issue_pair

    a, b = issue_pair()
    data = {"region": "FPL", "issues": [a, b]}
    if case == "empty":
        data["issues"] = []
    if case == "region":
        data["region"] = "CAISO"
    if case == "receipt":
        b["receipt_status"] = "unavailable"
    if case == "loading":
        b["loaded"] = False
    if case == "clock":
        b["timing"]["generated_at"] = "invalid"
    if case == "order":
        a["timing"]["generated_at"] = b["timing"]["generated_at"]
    if case == "no_overlap":
        b["forecast"] = []
    result = driver.dispatch(
        "published-review-content.children",
        {
            "explorer-data": data,
            "published-review-earlier": "earlier",
            "published-review-later": "later",
            "region-selector": "FPL",
            "published-review-mode.active_tab": "runs",
        },
    )
    assert result.ok
    text = result.text("published-review-content")
    assert expected in text and "Peak demand change" not in text
