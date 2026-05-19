"""Regression tests for the Forecast-tab scenario simulator heuristic.

A latent gap shipped where the ``_build_scenarios_panel`` heuristic
coupled ΔPeak to temperature only — wind and solar deltas moved the
renewable-share KPI but produced literally zero ΔPeak. That made the
"Stress-test demand against weather shifts" subhead misleading (the
Solar Eclipse and Calm Overcast scenario presets produced no demand
delta at all).

These tests lock the post-fix behavior: temperature is still the
dominant driver, but wind and solar now contribute small, defensible
demand-side terms. Full-fidelity physics lives in
``simulation/scenario_engine.py`` — these are heuristic-tier checks.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

from components._callbacks_overview import (
    _build_scenarios_panel,
    _scenario_demand_factor,
)


class TestScenarioDemandFactor:
    """Pure-function checks on the heuristic itself."""

    def test_zero_deltas_return_unit_factor(self):
        assert _scenario_demand_factor(0.0, 0.0, 0.0) == pytest.approx(1.0)

    def test_temp_delta_dominates(self):
        # ±2.5 % per 5 °F — existing coefficient, must stay stable.
        assert _scenario_demand_factor(5.0, 0.0, 0.0) == pytest.approx(1.025)
        assert _scenario_demand_factor(-5.0, 0.0, 0.0) == pytest.approx(0.975)
        assert _scenario_demand_factor(10.0, 0.0, 0.0) == pytest.approx(1.05)

    def test_solar_delta_moves_demand(self):
        # +200 W/m² → +3 % demand (sun load / AC). This is the headline
        # regression: pre-fix this was 1.000 (zero coupling).
        assert _scenario_demand_factor(0.0, 0.0, 200.0) == pytest.approx(1.03)
        assert _scenario_demand_factor(0.0, 0.0, -200.0) == pytest.approx(0.97)

    def test_wind_delta_moves_demand(self):
        # +10 mph → +0.5 % demand (wind chill → heating). Pre-fix was 1.000.
        assert _scenario_demand_factor(0.0, 10.0, 0.0) == pytest.approx(1.005)
        assert _scenario_demand_factor(0.0, -10.0, 0.0) == pytest.approx(0.995)

    def test_solar_effect_exceeds_wind_effect(self):
        # At slider maxima, solar (±200 → ±3 %) should swamp wind
        # (±10 → ±0.5 %). Captures the order-of-magnitude claim in
        # the docstring.
        solar_max = abs(_scenario_demand_factor(0.0, 0.0, 200.0) - 1.0)
        wind_max = abs(_scenario_demand_factor(0.0, 10.0, 0.0) - 1.0)
        assert solar_max > wind_max * 5  # solar effect ≥ 5× wind at slider maxima

    def test_temp_effect_dominates_at_realistic_combos(self):
        # +14 °F (Heat Dome) + max solar + max wind: temp should still
        # drive most of the delta. Locks the "temperature is dominant"
        # narrative.
        combined = _scenario_demand_factor(14.0, 10.0, 200.0)
        temp_only = _scenario_demand_factor(14.0, 0.0, 0.0)
        temp_share = (temp_only - 1.0) / (combined - 1.0)
        assert temp_share > 0.6  # temp explains >60 % of the combined delta

    def test_terms_combine_linearly(self):
        # f(a+b) = f(a) + f(b) - 1.0 (since baseline is 1.0). Locks the
        # linearity property the heuristic claims.
        t = _scenario_demand_factor(5.0, 0.0, 0.0) - 1.0
        w = _scenario_demand_factor(0.0, 5.0, 0.0) - 1.0
        s = _scenario_demand_factor(0.0, 0.0, 100.0) - 1.0
        combined = _scenario_demand_factor(5.0, 5.0, 100.0) - 1.0
        assert combined == pytest.approx(t + w + s)


class TestScenariosPanelEndToEnd:
    """End-to-end checks against ``_build_scenarios_panel``: assert the
    scenario forecast trace actually diverges from baseline when wind or
    solar move.
    """

    @staticmethod
    def _demand_json() -> str:
        # Synthetic 14-day hourly demand series; enough lookback for
        # the panel's ``get_forecasts`` call to produce a non-empty
        # ensemble (the panel needs 24+ hours of forecast).
        ts = pd.date_range("2026-05-01", periods=24 * 14, freq="h")
        # Sinusoidal daily cycle around 100k MW — mimics PJM scale.
        hours = np.arange(len(ts))
        demand = 100_000 + 15_000 * np.sin(2 * np.pi * hours / 24)
        df = pd.DataFrame({"timestamp": ts, "demand_mw": demand})
        return df.to_json(orient="records", date_format="iso")

    @staticmethod
    def _scenario_trace_y(figure) -> np.ndarray | None:
        """Extract the scenario forecast y-series from the panel figure."""
        for trace in figure.data:
            name = (trace.name or "").lower()
            if "scenario" in name:
                return np.asarray(trace.y, dtype=float)
        return None

    def test_solar_delta_diverges_from_baseline(self):
        demand_json = self._demand_json()
        _, fig_baseline = _build_scenarios_panel(0, 0, 0, "PJM", demand_json)
        _, fig_solar = _build_scenarios_panel(0, 0, 200, "PJM", demand_json)

        base_y = self._scenario_trace_y(fig_baseline)
        solar_y = self._scenario_trace_y(fig_solar)

        if base_y is None or solar_y is None:
            pytest.skip("ensemble forecast unavailable in this environment")

        # Solar +200 W/m² should lift the entire 24h trace by ~3 %.
        ratio = float(np.mean(solar_y / base_y))
        assert ratio == pytest.approx(1.03, rel=1e-3)

    def test_wind_delta_diverges_from_baseline(self):
        demand_json = self._demand_json()
        _, fig_baseline = _build_scenarios_panel(0, 0, 0, "PJM", demand_json)
        _, fig_wind = _build_scenarios_panel(0, 10, 0, "PJM", demand_json)

        base_y = self._scenario_trace_y(fig_baseline)
        wind_y = self._scenario_trace_y(fig_wind)

        if base_y is None or wind_y is None:
            pytest.skip("ensemble forecast unavailable in this environment")

        ratio = float(np.mean(wind_y / base_y))
        assert ratio == pytest.approx(1.005, rel=1e-3)

    def test_zero_deltas_produce_identical_traces(self):
        # Sanity check: baseline-vs-scenario should overlap perfectly
        # when all sliders are at 0. The user's screenshots showed
        # this happening for the wind/solar-only cases pre-fix —
        # post-fix that should ONLY happen at (0, 0, 0).
        demand_json = self._demand_json()
        _, fig = _build_scenarios_panel(0, 0, 0, "PJM", demand_json)

        base_y = self._scenario_trace_y(fig)
        if base_y is None:
            pytest.skip("ensemble forecast unavailable in this environment")

        # Find the "baseline" trace and confirm it matches scenario.
        baseline_trace = None
        for trace in fig.data:
            if "baseline" in (trace.name or "").lower():
                baseline_trace = np.asarray(trace.y, dtype=float)
                break
        if baseline_trace is None:
            pytest.skip("baseline trace not present in this layout")

        np.testing.assert_allclose(base_y, baseline_trace, rtol=1e-9)
