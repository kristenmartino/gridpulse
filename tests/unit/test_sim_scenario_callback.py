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


def _call_run_scenario(
    fn,
    *,
    demand_json,
    active_tab,
    duration=24,
    temp=75,
    wind=10,
    cloud=50,
    humidity=60,
    solar_irr=500,
    region="FPL",
):
    """Invoke run_scenario with configurable slider values.

    Defaults mirror the baseline weather (75°F / 60% humidity / 50% cloud)
    so the scenario collapses to the baseline with zero delta.
    """
    return fn(
        0,  # run_clicks
        [0],  # preset_clicks (ALL wildcard — at least one preset button)
        demand_json,  # demand-store.data
        active_tab,  # dashboard-tabs.active_tab
        temp,  # sim-temp
        wind,  # sim-wind
        cloud,  # sim-cloud
        humidity,  # sim-humidity
        solar_irr,  # sim-solar
        duration,  # sim-duration (hours)
        region,  # region-selector
    )


def _extract_demand_delta_mw(result):
    """Pull the numeric MW delta out of the KPI string (e.g. '+1,234 MW')."""
    s = result[1].replace(",", "").replace(" MW", "").replace("+", "")
    return float(s)


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
        result = _call_run_scenario(fn, demand_json=_populated_demand_json(), active_tab=None)
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
        assert result[1] == "Warming up"  # demand
        assert result[3] == "Warming up"  # price
        assert result[5] == "Warming up"  # reserve
        assert result[7] == "Warming up"  # renewable
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
        result = _call_run_scenario(fn, demand_json="not json", active_tab="tab-simulator")
        assert result[1] == "Warming up"


class TestRunScenarioHeuristicSensitivity:
    """Regression coverage for the heuristic `/65` flattening bug.

    Before the fix, moving temperature from 75 → 120°F produced a
    demand delta of ~1% (buried in noise). These tests lock in that
    each slider now moves its target KPI by a materially visible
    amount.
    """

    def _base_kwargs(self):
        return dict(demand_json=_populated_demand_json(n=48), active_tab="tab-simulator")

    def test_hot_temperature_increases_demand(self, callbacks):
        fn = callbacks["run_scenario"]
        mock_ctx = MagicMock(triggered_id="sim-run-btn")
        with patch("components.callbacks.ctx", mock_ctx):
            baseline = _call_run_scenario(fn, temp=75, **self._base_kwargs())
            hot = _call_run_scenario(fn, temp=110, **self._base_kwargs())
        hot_delta = _extract_demand_delta_mw(hot)
        base_delta = _extract_demand_delta_mw(baseline)
        # A 35°F jump above baseline should lift demand by >5% of the
        # ~28k MW fixture — i.e. >1,400 MW above the baseline scenario.
        assert hot_delta - base_delta > 1400

    def test_cold_temperature_increases_demand(self, callbacks):
        fn = callbacks["run_scenario"]
        mock_ctx = MagicMock(triggered_id="sim-run-btn")
        with patch("components.callbacks.ctx", mock_ctx):
            baseline = _call_run_scenario(fn, temp=75, **self._base_kwargs())
            cold = _call_run_scenario(fn, temp=10, **self._base_kwargs())
        assert _extract_demand_delta_mw(cold) - _extract_demand_delta_mw(baseline) > 2000

    def test_humidity_amplifies_heat_demand(self, callbacks):
        fn = callbacks["run_scenario"]
        mock_ctx = MagicMock(triggered_id="sim-run-btn")
        with patch("components.callbacks.ctx", mock_ctx):
            dry = _call_run_scenario(fn, temp=100, humidity=20, **self._base_kwargs())
            humid = _call_run_scenario(fn, temp=100, humidity=95, **self._base_kwargs())
        assert _extract_demand_delta_mw(humid) > _extract_demand_delta_mw(dry)

    def test_solar_reduces_net_load_reserve(self, callbacks):
        """Higher solar irradiance → more renewable gen → higher reserve margin."""
        fn = callbacks["run_scenario"]
        mock_ctx = MagicMock(triggered_id="sim-run-btn")
        with patch("components.callbacks.ctx", mock_ctx):
            low_sun = _call_run_scenario(fn, solar_irr=50, **self._base_kwargs())
            high_sun = _call_run_scenario(fn, solar_irr=1000, **self._base_kwargs())
        low_reserve = float(low_sun[5].rstrip("%"))
        high_reserve = float(high_sun[5].rstrip("%"))
        assert high_reserve > low_reserve + 1.0  # at least 1 pp improvement


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
