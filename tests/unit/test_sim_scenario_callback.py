"""Tests for the Scenario Simulator ``run_scenario`` callback closure.

Regression coverage for the "Scenario tab hangs on 'Loading…'" bug:
- active-tab gate returns `no_update` when the user is on another tab
- warming-state demand-store JSON returns "Warming up" across every KPI
  (previously fell through to ``np.max`` on an empty array and crashed)
- populated demand produces real numeric outputs and non-empty figures
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from dash import no_update


@pytest.fixture(scope="module")
def callbacks():
    """Register all callbacks and expose the unwrapped run_scenario fn."""
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
    )

    from components.layout import build_layout

    app.layout = build_layout()

    from components.callbacks import register_callbacks

    register_callbacks(app)

    fns = {}
    for _key, val in app.callback_map.items():
        fn = val.get("callback")
        if fn and hasattr(fn, "__name__"):
            fns[fn.__name__] = getattr(fn, "__wrapped__", fn)
    return fns


def _call_run_scenario(fn, *, demand_json, active_tab, duration=24):
    """Invoke run_scenario with a stable default slider/region set."""
    return fn(
        0,                 # run_clicks
        [0],               # preset_clicks (ALL wildcard — at least one preset button)
        demand_json,       # demand-store.data
        active_tab,        # dashboard-tabs.active_tab
        75,                # sim-temp
        10,                # sim-wind
        30,                # sim-cloud
        50,                # sim-humidity
        200,               # sim-solar
        duration,          # sim-duration (hours)
        "FPL",             # region-selector
    )


def _populated_demand_json(n=48, base_mw=28000):
    start = datetime(2024, 6, 1, tzinfo=UTC)
    ts = pd.date_range(start, periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    demand = base_mw + 3000 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 200, n)
    return pd.DataFrame({"timestamp": ts, "demand_mw": demand}).to_json()


def _warming_demand_json():
    """Exactly what load_data() emits under REQUIRE_REDIS when Redis misses."""
    return pd.DataFrame(columns=["timestamp", "demand_mw"]).to_json(date_format="iso")


class TestRunScenarioTabGate:
    def test_inactive_tab_returns_no_update(self, callbacks):
        fn = callbacks["run_scenario"]
        result = _call_run_scenario(
            fn, demand_json=_populated_demand_json(), active_tab="tab-outlook"
        )
        assert result == [no_update] * 11

    def test_none_active_tab_returns_no_update(self, callbacks):
        fn = callbacks["run_scenario"]
        result = _call_run_scenario(
            fn, demand_json=_populated_demand_json(), active_tab=None
        )
        assert result == [no_update] * 11


class TestRunScenarioWarmingState:
    def test_empty_demand_returns_warming(self, callbacks):
        """Regression: used to crash with ValueError on np.max(empty)."""
        fn = callbacks["run_scenario"]
        result = _call_run_scenario(
            fn, demand_json=_warming_demand_json(), active_tab="tab-simulator"
        )
        # 11 outputs: fig, demand, demand_pct, price, price_delta, reserve,
        # reserve_status, renewable, renewable_detail, price_fig, renewable_fig
        assert len(result) == 11
        assert result[1] == "Warming up"   # demand
        assert result[3] == "Warming up"   # price
        assert result[5] == "Warming up"   # reserve
        assert result[7] == "Warming up"   # renewable
        # Figures are plotly.graph_objs.Figure — assert they exist, no crash.
        assert isinstance(result[0], go.Figure)
        assert isinstance(result[9], go.Figure)
        assert isinstance(result[10], go.Figure)

    def test_none_demand_returns_warming(self, callbacks):
        """First page render before any store hydration."""
        fn = callbacks["run_scenario"]
        result = _call_run_scenario(fn, demand_json=None, active_tab="tab-simulator")
        assert result[1] == "Warming up"
        assert result[3] == "Warming up"

    def test_malformed_demand_returns_warming(self, callbacks):
        """Defensive: corrupt JSON shouldn't crash the callback."""
        fn = callbacks["run_scenario"]
        result = _call_run_scenario(
            fn, demand_json="not json", active_tab="tab-simulator"
        )
        assert result[1] == "Warming up"


class TestRunScenarioPopulated:
    def test_populated_demand_computes_scenario(self, callbacks):
        fn = callbacks["run_scenario"]
        mock_ctx = MagicMock()
        mock_ctx.triggered_id = "sim-run-btn"
        with patch("components.callbacks.ctx", mock_ctx):
            result = _call_run_scenario(
                fn,
                demand_json=_populated_demand_json(n=48),
                active_tab="tab-simulator",
            )
        # Demand delta should be a formatted MW string (e.g. "+123 MW")
        assert isinstance(result[1], str)
        assert "MW" in result[1]
        assert "Warming up" not in result[1]
        # Reserve margin is a percentage string
        assert "%" in result[5]
        # Forecast figure has data traces (baseline + scenario + delta)
        assert isinstance(result[0], go.Figure)
        assert len(result[0].data) >= 2
