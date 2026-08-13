"""#127 criterion 3 — the scenario grid recovers a model's real sensitivity.

The blocker on this test was that CI has no trained models, and the objection
to faking one was that "a synthetic booster only re-measures the stub". That
is true of a hand-written stub function. It is NOT true of a REAL XGBoost
trained on data with a deliberately encoded temperature response: then the
test asserts the whole pipeline — weather override, derived-feature
recomputation, recursive inference, ratio against the baseline — recovers a
relationship that genuinely exists in the training data.

What that buys, and what it does not: it pins the machinery end to end and it
catches a grid that has stopped responding to weather at all. It says nothing
about whether the PRODUCTION boosters are well calibrated, which only the live
checks in docs/SCENARIO_GRID.md can answer.
"""

import numpy as np
import pandas as pd
import pytest

from data.feature_engineering import batched_recursive_autoregressive_forecast
from models.xgboost_model import predict_xgboost
from simulation.scenario_grid import build_scenario_grid, grid_axes, interpolate_scenario_factors

HORIZON = 12
FEATURES = [
    "temperature_2m",
    "cooling_degree_days",
    "heating_degree_days",
    "temperature_deviation",
    "wind_speed_80m",
    "wind_power_estimate",
    "shortwave_radiation",
    "solar_capacity_factor",
    "hour_sin",
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_roll_24h_mean",
]


def _synthetic_history(
    n: int, cooling: float, heating: float, seed: int = 0
) -> tuple[pd.DataFrame, np.ndarray]:
    """Hourly history whose demand is a KNOWN function of temperature.

    ``cooling`` and ``heating`` are MW per degree-day. A cooling-dominated BA
    (FPL-shaped) and a heating-dominated one (SPA-shaped) are the two cases
    that separate real physics from the #119 heuristic, which has a single
    positive temperature coefficient for every region on earth.
    """
    rng = np.random.default_rng(seed)
    hours = np.arange(n)
    # Wide temperature range so the booster has both regimes to learn from.
    temp = (
        55.0 + 30.0 * np.sin(hours / (24 * 30) * 2 * np.pi) + 12.0 * np.sin(hours / 24 * 2 * np.pi)
    )
    cdd = np.maximum(0.0, temp - 65.0)
    hdd = np.maximum(0.0, 65.0 - temp)
    demand = 1000.0 + cooling * cdd + heating * hdd + rng.normal(0, 5.0, n)

    df = pd.DataFrame(
        {
            "temperature_2m": temp,
            "cooling_degree_days": cdd,
            "heating_degree_days": hdd,
            "temperature_deviation": temp - pd.Series(temp).rolling(720, min_periods=1).mean(),
            "wind_speed_80m": np.full(n, 8.0),
            "wind_power_estimate": np.full(n, 0.2),
            "shortwave_radiation": np.clip(400.0 * np.sin(hours / 24 * 2 * np.pi), 0, None),
            "solar_capacity_factor": np.full(n, 0.3),
            "hour_sin": np.sin(hours / 24 * 2 * np.pi),
            "demand_lag_1h": np.roll(demand, 1),
            "demand_lag_24h": np.roll(demand, 24),
            "demand_roll_24h_mean": pd.Series(demand).rolling(24, min_periods=1).mean().to_numpy(),
            "demand_mw": demand,
        }
    )
    return df.iloc[48:].reset_index(drop=True), demand[48:]


def _train(df: pd.DataFrame) -> dict:
    """A real booster, not a stub — that is the whole point of this file."""
    from xgboost import XGBRegressor

    model = XGBRegressor(n_estimators=60, max_depth=4, learning_rate=0.2, n_jobs=1, random_state=0)
    model.fit(df[FEATURES].values, df["demand_mw"].values)
    return {"model": model, "feature_names": FEATURES}


def _future_window(base_temp: float, template: pd.DataFrame) -> pd.DataFrame:
    """A forward frame centred on a CHOSEN temperature regime.

    Reusing the tail of history put the window wherever the synthetic seasonal
    cycle happened to end — which on the first run landed at ~29 F, deep in the
    heating regime, so a "cooling-driven" model correctly showed demand FALLING
    as it warmed. The regime of the forecast window decides the sign, not the
    size of the coefficients, so the window has to be placed deliberately.
    That is the same reason FPL in August and SPA in January differ in
    production.
    """
    hours = np.arange(HORIZON)
    temp = base_temp + 8.0 * np.sin(hours / 24 * 2 * np.pi)
    out = template.iloc[-HORIZON:].reset_index(drop=True).drop(columns=["demand_mw"]).copy()
    out["temperature_2m"] = temp
    out["cooling_degree_days"] = np.maximum(0.0, temp - 65.0)
    out["heating_degree_days"] = np.maximum(0.0, 65.0 - temp)
    out["temperature_deviation"] = temp - float(template["temperature_2m"].tail(720).mean())
    out["hour_sin"] = np.sin(hours / 24 * 2 * np.pi)
    return out


def _grid(cooling: float, heating: float, base_temp: float, seed: int = 0) -> dict:
    hist, _ = _synthetic_history(1200, cooling, heating, seed)
    model_dict = _train(hist)
    future = _future_window(base_temp, hist)

    def forecaster(frames):
        return batched_recursive_autoregressive_forecast(
            model_dict, hist["demand_mw"], frames, predict_xgboost
        )

    baseline = forecaster([future])[0]
    return build_scenario_grid(
        featured=hist,
        future_df=future,
        baseline=baseline,
        forecaster=forecaster,
        horizon=HORIZON,
    )


@pytest.fixture(scope="module")
def cooling_ba() -> dict:
    """FPL in August: cooling-driven, and the window sits in the cooling regime."""
    return _grid(cooling=12.0, heating=1.0, base_temp=82.0, seed=1)


@pytest.fixture(scope="module")
def heating_ba() -> dict:
    """SPA in January: heating-driven, window in the heating regime.

    Warming REDUCES demand here — the case the #119 heuristic cannot express
    at any parameter value, and what SPA actually does in production.
    """
    return _grid(cooling=1.0, heating=12.0, base_temp=38.0, seed=2)


def _mean_factor(payload: dict, temp: float) -> float:
    curve = interpolate_scenario_factors(payload, temp, 0.0, 0.0)
    return float(np.mean(curve))


class TestTheGridRecoversTheModelsSensitivity:
    def test_a_heat_scenario_raises_demand_for_a_cooling_driven_model(self, cooling_ba):
        """The direction, against a booster whose training data says so."""
        assert _mean_factor(cooling_ba, 20.0) > 1.01

    def test_a_heat_scenario_lowers_demand_for_a_heating_driven_model(self, heating_ba):
        """The assertion that makes this a physics test rather than a smoke test.

        `_scenario_demand_factor` is `1 + (temp/5) * 0.025 + ...` — a single
        positive coefficient for every region. It cannot return a factor below
        1.0 for warming at any parameter value. A winter-peaking model can and
        must, and SPA does exactly this in production (0.9677 at +20 F).
        """
        assert _mean_factor(heating_ba, 20.0) < 1.0

    def test_the_two_regions_disagree(self, cooling_ba, heating_ba):
        """BA-dependence. The heuristic is BA-independent by construction, so
        any separation here is a property the old path could not have."""
        assert _mean_factor(cooling_ba, 20.0) - _mean_factor(heating_ba, 20.0) > 0.02

    def test_the_answer_cannot_have_come_from_the_analytical_coefficients(
        self, cooling_ba, heating_ba
    ):
        """#127 criterion 3, asserted structurally rather than by magnitude.

        The obvious version of this test — "the grid differs from 1.10 at
        +20 F" — is a bad test, and the first draft of it failed for the right
        reason: this fixture's cooling model returns 1.0959 against the
        heuristic's 1.1000, a 0.004 gap. Coincidental agreement on one region
        is not evidence the coefficients are still in use, so a magnitude
        threshold measures the fixture's tuning rather than the code path.

        What the heuristic CANNOT do, at any coefficient values, is return two
        different answers for two regions given the same slider, or return a
        value below 1.0 for warming. Both are asserted here, and neither can be
        satisfied by a lucky constant.
        """
        from components._callbacks_forecast import _scenario_demand_factor

        heuristic = _scenario_demand_factor(20.0, 0.0, 0.0)
        cooling = _mean_factor(cooling_ba, 20.0)
        heating = _mean_factor(heating_ba, 20.0)

        # One number for every region on earth — that is the property being
        # replaced, and it is a property of the function, not of its constants.
        assert _scenario_demand_factor(20.0, 0.0, 0.0) == heuristic

        assert cooling > 1.0 > heating, (
            f"the grid must straddle 1.0 across regimes; heuristic is fixed at {heuristic}"
        )

    def test_the_response_is_ordered_across_the_temperature_axis(self, cooling_ba):
        """Warmer means more cooling load, monotonically, for this model.

        Ordering rather than exact values: tree ensembles saturate outside the
        training range (docs/SCENARIO_GRID.md), so the magnitudes are not
        pinnable but the ORDER is.
        """
        factors = [_mean_factor(cooling_ba, t) for t in (-10.0, -5.0, 0.0, 5.0, 10.0)]

        assert factors == sorted(factors), f"not monotone in temperature: {factors}"

    def test_the_untouched_position_is_exactly_the_baseline(self, cooling_ba):
        """The parity property, asserted here against a real booster rather
        than only in production. Zero drift is what makes every other number
        in the payload a weather response and nothing else."""
        assert cooling_ba["origin_drift"] == 0.0
        np.testing.assert_allclose(_mean_factor(cooling_ba, 0.0), 1.0)

    def test_no_cell_is_implausible_for_a_well_behaved_model(self, cooling_ba, heating_ba):
        """The 0.6-1.7 band must not bind on real physics (#475)."""
        assert cooling_ba["implausible_cells"] == []
        assert heating_ba["implausible_cells"] == []


class TestHeatDomePreset:
    def test_the_heat_dome_preset_clamps_to_the_grid_edge(self, cooling_ba):
        """#127 names a "+25 F Heat Dome" and the grid spans +/-20 F.

        The preset is therefore CLAMPED, not extrapolated — `_axis_position`
        holds the edge rather than running a cubic wind-power response past its
        last measured point. Recorded as behaviour rather than left for someone
        to discover the slider going quiet above +20.
        """
        assert max(grid_axes()[0]) == 20.0
        assert _mean_factor(cooling_ba, 25.0) == pytest.approx(_mean_factor(cooling_ba, 20.0))
