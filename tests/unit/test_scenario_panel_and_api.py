"""The scenario panel's tab-change bug, its source label, and the read API.

All three come from the same discovery: with the grid live in production the
panel rendered empty and the copy under it said "not a model re-forecast"
while a model re-forecast was exactly what was being served.
"""

import numpy as np
import pytest

import config
from simulation.scenario_grid import grid_axes


def _payload(hot_factor: float = 1.3) -> dict:
    temps, winds, solars = grid_axes()
    return {
        "region": "FPL",
        "horizon": 24,
        "generated_at": "2026-08-10T23:00:00+00:00",
        "axes": {"temp_f": list(temps), "wind_mph": list(winds), "solar_wm2": list(solars)},
        "factors": [
            [[[hot_factor if t > 0 else 1.0] * 24 for _ in solars] for _ in winds] for t in temps
        ],
    }


class TestPanelFiresOnTabChange:
    """`active_tab` was State, so a tab change never re-rendered the panel."""

    def test_active_tab_is_an_input_not_state(self):
        """The bug, pinned at the wiring rather than through a browser.

        As State, arriving at the Forecast tab with the panel already open —
        a bookmark, a reload, or opening the panel from another tab — left
        the KPI row blank until the user moved a slider, because none of the
        callback's inputs had changed. Observed live 2026-08-10: panel
        rendered, sliders responded, KPIs empty.
        """
        import inspect

        from components import _callbacks_forecast

        src = inspect.getsource(_callbacks_forecast)
        marker = src.index("forecast-scenarios-kpis")
        window = src[marker : marker + 1200]

        assert 'Input("dashboard-tabs", "active_tab")' in window
        assert 'State("dashboard-tabs", "active_tab")' not in window


class TestTheLabelTracksTheEngine:
    """The panel must not assert an engine it is not using."""

    def test_the_static_copy_no_longer_claims_the_heuristic(self):
        """`tab_demand_outlook` said "not a model re-forecast" unconditionally.

        Static copy cannot track a feature flag. It shipped that claim for a
        fortnight while #127 served real forecasts underneath it.
        """
        import inspect

        from components import tab_demand_outlook

        assert "not a model re-forecast" not in inspect.getsource(tab_demand_outlook)

    @pytest.mark.parametrize(
        ("flag", "expected"),
        [(True, "grid"), (False, "heuristic")],
        ids=["grid", "heuristic"],
    )
    def test_the_source_is_reported_for_whichever_engine_ran(self, monkeypatch, flag, expected):
        from components._callbacks_overview import _scenario_factors

        monkeypatch.setitem(config.FEATURE_FLAGS, "scenario_grid", flag)
        if flag:
            monkeypatch.setattr("data.redis_client.redis_get", lambda *_a, **_k: _payload())

        _, source = _scenario_factors("FPL", 20.0, 0.0, 0.0, 24)

        assert source == expected


class TestScenarioEndpoint:
    @pytest.fixture
    def client(self):
        # Same shape as tests/unit/test_web_tier_guard.py — api.py exposes a
        # Blueprint, and the Dash app mounts it.
        from flask import Flask

        from api import api_v1

        app = Flask(__name__)
        app.register_blueprint(api_v1)
        return app.test_client()

    @pytest.fixture(autouse=True)
    def _grid_on(self, monkeypatch):
        """The endpoint is gated on the same flag as the data it serves."""
        monkeypatch.setitem(config.FEATURE_FLAGS, "scenario_grid", True)

    def test_the_endpoint_is_absent_when_the_flag_is_off(self, client, monkeypatch):
        """With the flag off nothing writes a grid, so the endpoint could only
        ever answer "nothing here" — and a public surface that exists solely to
        say that is worse than one not published yet. 404, not 503: the
        resource does not exist rather than being temporarily cold."""
        monkeypatch.setitem(config.FEATURE_FLAGS, "scenario_grid", False)

        r = client.get("/api/v1/scenario/FPL")

        assert r.status_code == 404
        assert r.get_json()["error"] == "not_found"

    def test_a_warming_region_says_so_rather_than_500ing(self, client, monkeypatch):
        monkeypatch.setattr("api.redis_get", lambda *_a, **_k: None)

        r = client.get("/api/v1/scenario/FPL")

        assert r.status_code in (200, 503)
        assert "scenario grid" in r.get_json().get("detail", "").lower()

    def test_an_unknown_region_is_rejected(self, client):
        r = client.get("/api/v1/scenario/NOT_A_BA")

        assert r.status_code in (400, 404)

    def test_the_raw_grid_is_returned_without_deltas(self, client, monkeypatch):
        monkeypatch.setattr("api.redis_get", lambda *_a, **_k: _payload())

        body = client.get("/api/v1/scenario/FPL").get_json()

        assert body["region"] == "FPL"
        assert body["horizon"] == 24
        assert len(body["factors"]) == len(grid_axes()[0])

    def test_deltas_return_an_interpolated_curve(self, client, monkeypatch):
        """This is the half a direct Redis read cannot check.

        The endpoint goes through `interpolate_scenario_factors`, the same
        helper the web tier uses, so a green response exercises the serving
        path rather than only proving the payload exists.
        """
        monkeypatch.setattr("api.redis_get", lambda *_a, **_k: _payload(1.3))

        body = client.get("/api/v1/scenario/FPL?temp=20").get_json()

        assert body["deltas"] == {"temp_f": 20.0, "wind_mph": 0.0, "solar_wm2": 0.0}
        assert len(body["factors"]) == 24
        np.testing.assert_allclose(body["factors"], [1.3] * 24)

    def test_a_zero_position_returns_the_baseline(self, client, monkeypatch):
        monkeypatch.setattr("api.redis_get", lambda *_a, **_k: _payload())

        body = client.get("/api/v1/scenario/FPL?temp=0&wind=0&solar=0").get_json()

        np.testing.assert_allclose(body["factors"], [1.0] * 24)

    def test_non_numeric_deltas_are_refused(self, client, monkeypatch):
        monkeypatch.setattr("api.redis_get", lambda *_a, **_k: _payload())

        r = client.get("/api/v1/scenario/FPL?temp=hot")

        assert r.status_code == 400
        assert r.get_json()["error"] == "invalid_delta"
