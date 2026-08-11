"""The scenario grid's two integration seams (#127).

Web tier: read from Redis and interpolate, or fall back to the heuristic.
Scoring job: compute through the production inference path, fail open.
"""

import numpy as np
import pandas as pd
import pytest

import config
from components._callbacks_overview import _scenario_demand_factor, _scenario_factors
from simulation.scenario_grid import grid_axes

HOURS = 24


@pytest.fixture
def grid_on(monkeypatch):
    monkeypatch.setitem(config.FEATURE_FLAGS, "scenario_grid", True)


def _payload(factor: float = 1.4) -> dict:
    temps, winds, solars = grid_axes()
    return {
        "axes": {"temp_f": list(temps), "wind_mph": list(winds), "solar_wm2": list(solars)},
        "horizon": HOURS,
        "factors": [
            [[[factor if t > 0 else 1.0] * HOURS for _ in solars] for _ in winds] for t in temps
        ],
    }


class TestWebTierReadPath:
    def test_the_flag_off_serves_the_heuristic_untouched(self, monkeypatch):
        """#119's behaviour has to survive the flag being off — that is what
        makes this safe to ship dark."""
        monkeypatch.setitem(config.FEATURE_FLAGS, "scenario_grid", False)

        factors, source = _scenario_factors("PJM", 10.0, 0.0, 0.0, HOURS)

        assert source == "heuristic"
        np.testing.assert_allclose(factors, _scenario_demand_factor(10.0, 0.0, 0.0))

    def test_a_cold_cache_falls_back_rather_than_rendering_nothing(self, grid_on, monkeypatch):
        """Warming Redis, a shed region, a BA the job never reached — all the
        same to the user, and all better served by an approximate curve than
        by an empty panel."""
        monkeypatch.setattr("data.redis_client.redis_get", lambda *_a, **_k: None)

        factors, source = _scenario_factors("PJM", 10.0, 0.0, 0.0, HOURS)

        assert source == "heuristic"
        assert factors.shape == (HOURS,)

    def test_a_present_grid_is_used_in_preference_to_the_heuristic(self, grid_on, monkeypatch):
        monkeypatch.setattr("data.redis_client.redis_get", lambda *_a, **_k: _payload(1.4))

        factors, source = _scenario_factors("PJM", 20.0, 0.0, 0.0, HOURS)

        assert source == "grid"
        np.testing.assert_allclose(factors, np.full(HOURS, 1.4))
        # The heuristic would have said 1.10 for +20 °F. The whole point is
        # that these two disagree.
        assert not np.isclose(factors[0], _scenario_demand_factor(20.0, 0.0, 0.0))

    def test_a_malformed_payload_falls_back_instead_of_raising(self, grid_on, monkeypatch):
        monkeypatch.setattr("data.redis_client.redis_get", lambda *_a, **_k: {"axes": "nonsense"})

        factors, source = _scenario_factors("PJM", 10.0, 0.0, 0.0, HOURS)

        assert source == "heuristic"
        assert np.isfinite(factors).all()

    def test_a_redis_failure_falls_back_instead_of_raising(self, grid_on, monkeypatch):
        def boom(*_a, **_k):
            raise ConnectionError("redis down")

        monkeypatch.setattr("data.redis_client.redis_get", boom)

        _, source = _scenario_factors("PJM", 10.0, 0.0, 0.0, HOURS)

        assert source == "heuristic"

    def test_a_chart_longer_than_the_grid_holds_the_last_factor(self, grid_on, monkeypatch):
        """The grid is 24h; the chart is whatever the baseline happens to be.

        Running out mid-chart would draw a scenario line that silently
        rejoined the baseline partway across.
        """
        monkeypatch.setattr("data.redis_client.redis_get", lambda *_a, **_k: _payload(1.4))

        factors, source = _scenario_factors("PJM", 20.0, 0.0, 0.0, 48)

        assert source == "grid"
        assert factors.shape == (48,)
        np.testing.assert_allclose(factors, np.full(48, 1.4))

    def test_no_region_selected_is_not_an_error(self, grid_on):
        factors, source = _scenario_factors(None, 5.0, 0.0, 0.0, HOURS)

        assert source == "heuristic"
        assert factors.shape == (HOURS,)


class TestScoringJobWritePath:
    """``jobs.phases._write_scenario_grid`` — enrichment that must never cost
    a forecast that has already been written."""

    @staticmethod
    def _args(**over):
        n = 48
        future = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-08-01", periods=n, freq="h", tz="UTC"),
                "temperature_2m": np.linspace(70, 85, n),
                "wind_speed_80m": np.full(n, 8.0),
                "shortwave_radiation": np.full(n, 300.0),
                "hour_sin": np.zeros(n),
            }
        )
        args = {
            "region": "PJM",
            "featured": pd.DataFrame({"demand_mw": np.full(n, 1500.0)}),
            "future_df": future,
            "models": {"xgboost": object()},
            "baseline": np.full(n, 1500.0),
        }
        args.update(over)
        return args

    def test_the_flag_off_writes_nothing(self, monkeypatch):
        from jobs import phases

        monkeypatch.setitem(config.FEATURE_FLAGS, "scenario_grid", False)

        assert phases._write_scenario_grid(**self._args()) is False

    @pytest.mark.parametrize(
        "missing", [{"models": {}}, {"baseline": None}], ids=["no_model", "no_baseline"]
    )
    def test_a_region_without_a_model_or_baseline_is_skipped(self, grid_on, missing):
        from jobs import phases

        assert phases._write_scenario_grid(**self._args(**missing)) is False

    def test_a_failure_inside_the_grid_never_propagates(self, grid_on, monkeypatch):
        """The forecast is already in Redis when this runs. Nothing the grid
        can raise is worth turning a written forecast into a failed region
        (#268 -> #267)."""
        from jobs import phases

        def boom(*_a, **_k):
            raise RuntimeError("booster mismatch")

        monkeypatch.setattr(
            "data.feature_engineering.batched_recursive_autoregressive_forecast", boom
        )

        assert phases._write_scenario_grid(**self._args()) is False

    def test_a_successful_write_persists_under_the_region_key(self, grid_on, monkeypatch):
        from jobs import phases

        written = {}

        monkeypatch.setattr(
            "data.feature_engineering.batched_recursive_autoregressive_forecast",
            lambda model, seed, frames, predict_fn: [
                1500.0 + 20.0 * f["temperature_2m"].to_numpy() for f in frames
            ],
        )
        monkeypatch.setattr(
            "data.redis_client.persist",
            lambda key, payload, ttl=None: written.update({"key": key, "payload": payload}),
        )

        assert phases._write_scenario_grid(**self._args()) is True
        assert written["key"].endswith("scenario_grid:PJM")
        assert written["payload"]["region"] == "PJM"
        assert written["payload"]["horizon"] == config.SCENARIO_GRID_HORIZON_HOURS
        assert len(written["payload"]["factors"]) == len(grid_axes()[0])

    def test_the_grid_runs_the_same_recursive_path_as_the_baseline(self, grid_on, monkeypatch):
        """The correctness contract, asserted at the seam.

        A scenario from a plain vectorised predict divided by a baseline from
        the recursive path reports the gap between the *paths* as the response
        to *weather*. `scenario_engine._run_ensemble` is exactly such a second
        path, so this pins that the phase calls the production one and passes
        it the same seed frame.
        """
        from jobs import phases

        seen = []

        def spy(model, seed, frames, predict_fn):
            seen.append({"n_frames": len(frames), "rows": len(frames[0]), "seed": len(seed)})
            return [np.full(len(f), 1500.0) for f in frames]

        monkeypatch.setattr(
            "data.feature_engineering.batched_recursive_autoregressive_forecast", spy
        )
        monkeypatch.setattr("data.redis_client.persist", lambda *a, **k: None)

        phases._write_scenario_grid(**self._args())

        assert seen, "the production forecaster must be what computes the cells"
        # ONE batched call carrying all 80 computed cells (81 minus the origin,
        # which is defined rather than forecast). Cell-at-a-time issued 1,920
        # single-row predicts per region and cost 2.7x tick runtime (#462).
        assert len(seen) == 1
        assert seen[0]["n_frames"] == 81  # includes the origin parity cell (#472)
        assert seen[0]["rows"] == 24
        # Seeded from the same history the baseline was chained off, which is
        # what keeps scenario and baseline commensurable.
        assert seen[0]["seed"] > 0
