"""Unit tests for the scenario grid (#127).

The grid replaces the simulator's analytical heuristic with real forecasts.
Two properties matter more than the arithmetic: the grid must span the slider
domain so no slider position extrapolates, and every cell must come from the
*same* forecaster as the baseline it is divided by.
"""

import numpy as np
import pandas as pd
import pytest

import config
from simulation.scenario_engine import apply_weather_deltas
from simulation.scenario_grid import (
    build_scenario_grid,
    grid_axes,
    interpolate_scenario_factors,
)

HORIZON = 6


def _future_frame(n: int = HORIZON) -> pd.DataFrame:
    """A forward frame with a real diurnal temperature curve."""
    hours = np.arange(n)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-08-01", periods=n, freq="h", tz="UTC"),
            "temperature_2m": 70.0 + 15.0 * np.sin(hours / 24 * 2 * np.pi),
            "wind_speed_80m": np.full(n, 8.0),
            "shortwave_radiation": np.full(n, 300.0),
            "hour_sin": np.sin(hours / 24 * 2 * np.pi),
        }
    )


class TestApplyWeatherDeltas:
    def test_the_input_frame_is_never_mutated(self):
        """ADR-007: the scenario engine copies, never mutates."""
        original = _future_frame()
        before = original.copy(deep=True)

        apply_weather_deltas(original, temp_delta_f=10.0)

        pd.testing.assert_frame_equal(original, before)

    def test_a_delta_shifts_the_curve_rather_than_flattening_it(self):
        """The distinction between this and ``simulate_scenario``.

        Setting `temperature_2m` to a constant — which is what an absolute
        override does — erases the day/night cycle that shapes the demand
        response. A relative shift has to preserve it.
        """
        base = _future_frame()

        shifted = apply_weather_deltas(base, temp_delta_f=10.0)

        np.testing.assert_allclose(
            shifted["temperature_2m"].to_numpy(),
            base["temperature_2m"].to_numpy() + 10.0,
        )
        assert shifted["temperature_2m"].std() == pytest.approx(base["temperature_2m"].std())

    def test_derived_features_follow_the_drivers(self):
        """The reason the engine exists: CDD/HDD must move when temperature does.

        A scenario that shifted `temperature_2m` and left `cooling_degree_days`
        at its baseline value would feed the model a contradiction — hot
        weather with no cooling load — and the forecast would barely respond.
        """
        base = _future_frame()
        base["cooling_degree_days"] = 0.0
        base["wind_power_estimate"] = 0.0
        base["solar_capacity_factor"] = 0.0

        hotter = apply_weather_deltas(base, temp_delta_f=20.0)
        windier = apply_weather_deltas(base, wind_delta_mph=10.0)
        sunnier = apply_weather_deltas(base, solar_delta_wm2=200.0)

        assert hotter["cooling_degree_days"].max() > 0.0
        assert windier["wind_power_estimate"].max() > 0.0
        assert sunnier["solar_capacity_factor"].max() > 0.0

    def test_negative_wind_and_solar_are_clipped_to_zero(self):
        """Neither is a physical state, and `compute_wind_power` cubes its input.

        The sliders reach -10 mph, and a BA becalmed at 4 mph is inside that
        range, so this is reachable from the UI rather than defensive.
        """
        calm = _future_frame()
        calm["wind_speed_80m"] = 4.0
        calm["shortwave_radiation"] = 50.0

        out = apply_weather_deltas(calm, wind_delta_mph=-10.0, solar_delta_wm2=-200.0)

        assert (out["wind_speed_80m"] >= 0.0).all()
        assert (out["shortwave_radiation"] >= 0.0).all()


class TestGridAxes:
    def test_the_axes_span_the_slider_domain(self):
        """Every reachable slider position must interpolate, never extrapolate.

        The slider bounds live in ``components/tab_demand_outlook.py``
        (`_scenario_slider("temp", ..., -20, 20)` and friends). If a slider is
        ever widened without widening the grid, positions past the end would
        silently clamp — a flat spot in the UI rather than an error.
        """
        temps, winds, solars = grid_axes()

        assert (min(temps), max(temps)) == (-20.0, 20.0)
        assert (min(winds), max(winds)) == (-10.0, 10.0)
        assert (min(solars), max(solars)) == (-200.0, 200.0)

    def test_temperature_is_the_finely_sampled_axis(self):
        """9x3x3, and the 9 is on temperature deliberately.

        CDD/HDD are piecewise-linear in temperature with a kink at 65 °F;
        wind and solar enter through smooth monotone transforms. Sampling all
        three axes equally would spend the budget where the curvature is not.
        """
        temps, winds, solars = grid_axes()

        assert len(temps) == 9
        assert len(winds) == len(solars) == 3
        assert len(temps) * len(winds) * len(solars) == 81

    def test_every_axis_contains_the_no_change_point(self):
        """Zero must be a grid point, not something interpolated toward.

        An untouched slider is the commonest state of the simulator; if 0.0
        were between grid points, the panel would open showing a scenario
        that already differs from the baseline.
        """
        for axis in grid_axes():
            assert 0.0 in axis


class TestBuildScenarioGrid:
    @staticmethod
    def _one(frame: pd.DataFrame) -> np.ndarray:
        """A stand-in with a known, monotone response to temperature."""
        return 1000.0 + 10.0 * frame["temperature_2m"].to_numpy()

    @classmethod
    def _temp_sensitive_forecaster(cls, frames: list[pd.DataFrame]) -> list[np.ndarray]:
        """Batched contract: a list in, a list out."""
        return [cls._one(f) for f in frames]

    def _grid(self, forecaster=None):
        future = _future_frame()
        fc = forecaster or self._temp_sensitive_forecaster
        baseline = self._one(future)
        return build_scenario_grid(
            featured=pd.DataFrame({"demand_mw": np.full(48, 1500.0)}),
            future_df=future,
            baseline=baseline,
            forecaster=fc,
            horizon=HORIZON,
        )

    def test_the_payload_is_shaped_like_the_axes(self):
        payload = self._grid()
        temps, winds, solars = grid_axes()

        assert len(payload["factors"]) == len(temps)
        assert all(len(w) == len(winds) for w in payload["factors"])
        assert all(len(s) == len(solars) for w in payload["factors"] for s in w)
        assert payload["horizon"] == HORIZON
        assert payload["axes"]["temp_f"] == list(temps)

    def test_the_origin_cell_is_exactly_one(self):
        """(0, 0, 0) is the baseline by definition, not by measurement.

        Re-running the forecaster there would spend a cell reproducing a row
        of 1.0s, and any drift it showed would be nondeterminism rather than
        physics — which the simulator would then render as a weather response
        to moving no slider at all.
        """
        payload = self._grid()
        temps, winds, solars = grid_axes()
        ti, wi, si = temps.index(0.0), winds.index(0.0), solars.index(0.0)

        assert payload["factors"][ti][wi][si] == [1.0] * HORIZON

    def test_a_warmer_scenario_reports_a_factor_above_one(self):
        """Direction, against a forecaster whose response direction is known."""
        payload = self._grid()
        temps, winds, solars = grid_axes()
        wi, si = winds.index(0.0), solars.index(0.0)

        hot = payload["factors"][temps.index(20.0)][wi][si]
        cold = payload["factors"][temps.index(-20.0)][wi][si]

        assert all(f > 1.0 for f in hot)
        assert all(f < 1.0 for f in cold)

    def test_the_grid_uses_the_forecaster_it_is_given(self):
        """The contract the whole module exists to enforce.

        A scenario computed through one inference path and divided by a
        baseline from another reports the difference between the *paths* as
        the response to *weather*. `scenario_engine._run_ensemble` is such a
        second path — a plain vectorised predict against production's
        recursive chaining — so this asserts the injected forecaster is the
        only thing consulted.
        """
        batches: list[int] = []

        def counting_forecaster(frames: list[pd.DataFrame]) -> list[np.ndarray]:
            batches.append(len(frames))
            return [self._one(f) for f in frames]

        self._grid(counting_forecaster)

        # ONE batched call carrying all 81 cells. The origin is forecast too
        # since #472 — it is the grid's parity check, and defining it as 1.0
        # hid exactly the path disagreement this module exists to prevent.
        assert batches == [81]

    def test_a_cell_that_forecasts_garbage_degrades_to_the_baseline(self):
        """One diverged cell must not take the panel down or dwarf the chart.

        A recursive forecast can dive (#296). Returning 1.0 for that cell
        renders it as "no change", which is wrong but bounded; propagating a
        NaN would blank the chart and a 40x ratio would flatten the baseline
        against the axis.
        """
        future = _future_frame()
        baseline = self._one(future)

        def diverging(frames: list[pd.DataFrame]) -> list[np.ndarray]:
            return [
                np.full(len(f), np.nan) if f["temperature_2m"].mean() > 80.0 else self._one(f)
                for f in frames
            ]

        payload = build_scenario_grid(
            featured=pd.DataFrame({"demand_mw": np.full(48, 1500.0)}),
            future_df=future,
            baseline=baseline,
            forecaster=diverging,
            horizon=HORIZON,
        )

        flat = np.asarray(payload["factors"], dtype=float)
        assert np.isfinite(flat).all()
        assert (flat > 0).all()

    @pytest.mark.parametrize(
        "baseline",
        [
            np.full(HORIZON, np.nan),
            np.zeros(HORIZON),
            np.full(HORIZON - 1, 1000.0),
        ],
        ids=["non_finite", "zeros", "too_short"],
    )
    def test_an_unusable_baseline_is_refused_not_divided_by(self, baseline):
        """Ratios are the payload, so the denominator is load-bearing.

        A zero baseline yields inf, a NaN baseline yields NaN, and a short one
        silently produces a shorter curve than the horizon promises. All three
        are better as an exception in the job than as a payload in Redis.
        """
        with pytest.raises(ValueError):
            build_scenario_grid(
                featured=pd.DataFrame({"demand_mw": np.full(48, 1500.0)}),
                future_df=_future_frame(),
                baseline=baseline,
                forecaster=self._temp_sensitive_forecaster,
                horizon=HORIZON,
            )


class TestInterpolation:
    @staticmethod
    def _payload() -> dict:
        """A grid whose factor is a known linear function of the deltas."""
        temps, winds, solars = grid_axes()
        factors = [
            [[[1.0 + t / 100.0 + w / 100.0 + s / 1000.0] * HORIZON for s in solars] for w in winds]
            for t in temps
        ]
        return {
            "axes": {"temp_f": list(temps), "wind_mph": list(winds), "solar_wm2": list(solars)},
            "horizon": HORIZON,
            "factors": factors,
        }

    def test_an_exact_grid_point_returns_that_cell(self):
        out = interpolate_scenario_factors(self._payload(), 10.0, 0.0, 0.0)

        np.testing.assert_allclose(out, np.full(HORIZON, 1.10))

    def test_a_midpoint_blends_its_neighbours(self):
        """+2.5 °F sits between the -0 and +5 grid points."""
        out = interpolate_scenario_factors(self._payload(), 2.5, 0.0, 0.0)

        np.testing.assert_allclose(out, np.full(HORIZON, 1.025))

    def test_all_three_axes_interpolate_together(self):
        out = interpolate_scenario_factors(self._payload(), 2.5, 5.0, 100.0)

        np.testing.assert_allclose(out, np.full(HORIZON, 1.0 + 0.025 + 0.05 + 0.1))

    def test_the_untouched_position_is_a_no_op(self):
        """All sliders at rest must return exactly 1.0, not 0.9998."""
        out = interpolate_scenario_factors(self._payload(), 0.0, 0.0, 0.0)

        np.testing.assert_allclose(out, np.ones(HORIZON))

    def test_out_of_range_clamps_instead_of_extrapolating(self):
        """The grid spans the sliders, so this means the UI outran the grid.

        Holding the edge value is wrong by a bounded amount; extrapolating a
        cubic wind-power response past its last measured point is not.
        """
        out = interpolate_scenario_factors(self._payload(), 500.0, 0.0, 0.0)

        np.testing.assert_allclose(out, np.full(HORIZON, 1.20))

    @pytest.mark.parametrize(
        "payload",
        [{}, {"axes": {}}, {"axes": {"temp_f": [0.0]}, "factors": []}, None],
        ids=["empty", "no_axes", "no_factors", "none"],
    )
    def test_a_malformed_payload_returns_none_rather_than_raising(self, payload):
        """The caller falls back to the heuristic; a raise would blank the tab."""
        assert interpolate_scenario_factors(payload, 5.0, 0.0, 0.0) is None


class TestFeatureFlag:
    def test_the_flag_is_registered(self):
        """Registration is the durable contract; the value is operational.

        Asserted rather than the on/off state because ``feature_enabled``
        fail-closes on unknown flags (#145): a typo or a dropped entry would
        silently disable the grid and fall back to the heuristic with nothing
        but a log line to say so. The value itself flips with operational
        decisions and pinning it would just make this test a changelog.

        Note the flag is NOT an env-var override — ``feature_enabled`` reads
        ``FEATURE_FLAGS`` directly, so flipping it is a code change and a
        redeploy, not a config change.
        """
        assert "scenario_grid" in config.FEATURE_FLAGS
        assert isinstance(config.FEATURE_FLAGS["scenario_grid"], bool)

    def test_the_horizon_matches_what_the_simulator_charts(self):
        """`_scenario_demand_factor` documents a 24h baseline, and the 24 is
        the entire reason this is affordable — 384 steps would not be."""
        assert config.SCENARIO_GRID_HORIZON_HOURS == 24


class TestOriginIsComputed:
    """The origin cell is the grid's own parity check (#472).

    It was defined as 1.0 to save one forecast in 81. That saved cell was the
    only thing that could have caught the two sides of every ratio coming from
    different inference paths — the exact failure this module exists to
    prevent. Defining it away made the check invisible, so it is computed now.
    """

    @staticmethod
    def _one(frame: pd.DataFrame) -> np.ndarray:
        return 1000.0 + 10.0 * frame["temperature_2m"].to_numpy()

    def _build(self, forecaster):
        future = _future_frame()
        return build_scenario_grid(
            featured=pd.DataFrame(
                {"demand_mw": np.full(48, 1500.0), "temperature_2m": np.full(48, 70.0)}
            ),
            future_df=future,
            baseline=self._one(future),
            forecaster=forecaster,
            horizon=HORIZON,
        )

    def test_every_cell_including_the_origin_is_forecast(self):
        calls: list[int] = []

        def counting(frames):
            calls.append(len(frames))
            return [self._one(f) for f in frames]

        self._build(counting)

        assert calls == [81], "81 cells, none of them assumed"

    def test_matching_paths_report_no_origin_drift(self):
        payload = self._build(lambda frames: [self._one(f) for f in frames])

        assert payload["origin_drift"] == 0.0

    def test_a_path_disagreement_shows_up_as_origin_drift(self):
        """A forecaster that disagrees with the baseline by a constant — which
        is what two different inference paths look like — is now visible in the
        payload instead of being hidden behind a hard-coded 1.0."""
        payload = self._build(lambda frames: [self._one(f) * 1.05 for f in frames])

        assert payload["origin_drift"] == pytest.approx(0.05, abs=1e-3)


class TestEnvelopeFlags:
    """Tree models do not extrapolate; the payload has to say where that starts."""

    @staticmethod
    def _one(frame: pd.DataFrame) -> np.ndarray:
        return 1000.0 + 10.0 * frame["temperature_2m"].to_numpy()

    def _build(self, observed_temps: np.ndarray):
        future = _future_frame()
        return build_scenario_grid(
            featured=pd.DataFrame(
                {
                    "demand_mw": np.full(len(observed_temps), 1500.0),
                    "temperature_2m": observed_temps,
                }
            ),
            future_df=future,
            baseline=self._one(future),
            forecaster=lambda frames: [self._one(f) for f in frames],
            horizon=HORIZON,
        )

    def test_a_narrow_history_marks_the_extremes_extrapolated(self):
        """SPA's shape, measured live: saturates above +5 F and then wanders.

        A BA whose history spans a narrow band cannot answer a +20 F question,
        and the flag is what lets the UI stop pretending it can.
        """
        payload = self._build(np.linspace(60.0, 80.0, 200))

        flags = payload["envelope"]["temp_f"]
        temps = payload["axes"]["temp_f"]

        assert flags[temps.index(20.0)] is False
        assert flags[temps.index(-20.0)] is False

    def test_a_wide_history_keeps_positions_in_envelope(self):
        payload = self._build(np.linspace(0.0, 140.0, 200))

        assert all(payload["envelope"]["temp_f"])

    def test_every_axis_is_flagged(self):
        payload = self._build(np.linspace(60.0, 80.0, 200))

        assert set(payload["envelope"]) == {"temp_f", "wind_mph", "solar_wm2"}
        for axis, positions in (
            ("temp_f", "temp_f"),
            ("wind_mph", "wind_mph"),
            ("solar_wm2", "solar_wm2"),
        ):
            assert len(payload["envelope"][axis]) == len(payload["axes"][positions])


class TestWindowDependentFeatures:
    """`temperature_deviation` is a 720h rolling mean, not a pointwise function.

    Recomputing it on the simulator's 24-row frame made a zero-delta scenario
    differ from the baseline despite identical weather — measured live at up to
    0.013 on FPL, non-zero on 5 of 6 BAs, on the first tick that computed the
    origin cell (#474).
    """

    @staticmethod
    def _frame(n: int = 24) -> pd.DataFrame:
        hours = np.arange(n)
        temp = 70.0 + 15.0 * np.sin(hours / 24 * 2 * np.pi)
        return pd.DataFrame(
            {
                "temperature_2m": temp,
                "hour_sin": np.sin(hours / 24 * 2 * np.pi),
                # As the production frame builder leaves it: computed against
                # 30 days of history, most of which a scenario never touches.
                "temperature_deviation": temp - 68.0,
                "wind_speed_80m": np.full(n, 8.0),
                "shortwave_radiation": np.full(n, 300.0),
            }
        )

    def test_a_zero_delta_scenario_is_identical_to_its_input(self):
        """The property the origin cell measures. Anything else means every
        ratio in the payload has a non-weather component."""
        base = self._frame()

        out = apply_weather_deltas(base, 0.0, 0.0, 0.0)

        pd.testing.assert_series_equal(
            out["temperature_deviation"], base["temperature_deviation"], check_names=False
        )

    def test_deviation_shifts_by_the_delta_rather_than_being_recomputed(self):
        """A 24h shift against a 720h reference moves the deviation by ~the
        full delta; recomputing on the slice would instead re-centre it on the
        slice's own mean and lose the anomaly entirely."""
        base = self._frame()

        out = apply_weather_deltas(base, 10.0, 0.0, 0.0)

        np.testing.assert_allclose(
            out["temperature_deviation"].to_numpy(),
            base["temperature_deviation"].to_numpy() + 10.0,
        )

    def test_recomputing_on_a_slice_disagrees_with_the_carried_value(self):
        """Characterises the bug so it cannot come back quietly.

        `window=720, min_periods=1` on a 24-row frame is an EXPANDING mean over
        those 24 hours, not a 30-day reference. The result disagrees materially
        with the value the production frame builder computed against real
        history — which is why a zero-delta scenario used to differ from its
        own baseline.
        """
        from data.feature_engineering import compute_temperature_deviation

        base = self._frame()
        sliced = compute_temperature_deviation(base["temperature_2m"])

        disagreement = float(
            np.max(np.abs(sliced.to_numpy() - base["temperature_deviation"].to_numpy()))
        )
        assert disagreement > 1.0, "a slice recomputation is not the same feature"

    def test_the_fix_does_not_reintroduce_the_slice_recomputation(self):
        """`apply_weather_deltas` must carry-and-shift, never recompute."""
        from data.feature_engineering import compute_temperature_deviation

        base = self._frame()
        out = apply_weather_deltas(base, 10.0, 0.0, 0.0)
        sliced = compute_temperature_deviation(base["temperature_2m"] + 10.0)

        assert not np.allclose(out["temperature_deviation"].to_numpy(), sliced.to_numpy())

    def test_pointwise_features_are_still_recomputed(self):
        """Only the window-dependent one is special-cased — CDD and friends
        must still follow their drivers."""
        base = self._frame()
        base["cooling_degree_days"] = 0.0

        out = apply_weather_deltas(base, 20.0, 0.0, 0.0)

        assert out["cooling_degree_days"].max() > 0.0


class TestImplausibleCells:
    """The clip band was 0.25-4.0 and silent. Both were wrong (#475)."""

    @staticmethod
    def _one(frame: pd.DataFrame) -> np.ndarray:
        return 1000.0 + 10.0 * frame["temperature_2m"].to_numpy()

    def _build(self, forecaster):
        future = _future_frame()
        return build_scenario_grid(
            featured=pd.DataFrame(
                {"demand_mw": np.full(48, 1500.0), "temperature_2m": np.linspace(0, 140, 48)}
            ),
            future_df=future,
            baseline=self._one(future),
            forecaster=forecaster,
            horizon=HORIZON,
        )

    def test_the_band_is_tight_enough_to_notice_a_wander(self):
        """0.25-4.0 could only catch a catastrophe. The realistic failure here
        is a cell that wanders outside the training envelope, not one that
        explodes — which is what SPA did, measured live."""
        from simulation.scenario_grid import _MAX_FACTOR, _MIN_FACTOR

        assert _MIN_FACTOR >= 0.5
        assert _MAX_FACTOR <= 2.0

    def test_the_band_does_not_bind_on_observed_physics(self):
        """Measured hourly factors span ~0.91 to ~1.20 across five BAs. A bound
        that clipped real physics would be worse than no bound."""
        from simulation.scenario_grid import _MAX_FACTOR, _MIN_FACTOR

        assert _MIN_FACTOR < 0.91
        assert _MAX_FACTOR > 1.20

    def test_a_diverged_cell_is_dropped_and_reported_not_clamped(self):
        """A clamped 0.25 is a diverged forecast wearing a plausible number.

        The simulator would render it as "demand drops 75%" with nothing to
        say otherwise, so an out-of-band cell is treated like a non-finite one:
        dropped to the baseline, counted, and named in the payload.
        """

        def diverging(frames):
            return [
                np.full(len(f), 50.0)  # ~0.05x the baseline — a dive
                if f["temperature_2m"].mean() > 80.0
                else self._one(f)
                for f in frames
            ]

        payload = self._build(diverging)

        assert payload["implausible_cells"], "a diverged cell must be reported"
        flat = np.asarray(payload["factors"], dtype=float)
        assert flat.min() >= 0.6, "nothing below the band survives into the payload"
        assert np.isfinite(flat).all()

    def test_a_healthy_grid_reports_no_implausible_cells(self):
        payload = self._build(lambda frames: [self._one(f) for f in frames])

        assert payload["implausible_cells"] == []
        assert payload["origin_drift"] == 0.0


class TestTheDeltasGoTheRightWayAndOnlyWhenAsked:
    """`apply_weather_deltas` is live code — the scoring job runs it 81 times
    per region, hourly, for 51 BAs (#458). These pin the parts of it that
    mutation testing found unasserted (#487).

    The gap that mattered: **inverting the solar sign passed the entire
    suite.** `test_derived_features_follow_the_drivers` asserts
    `solar_capacity_factor > 0` off a 300 W/m^2 baseline, so a +200 delta that
    silently became -200 still left 100 W/m^2 and a positive factor. A
    relationship assertion ("it went up") over a lenient fixture cannot see a
    sign flip — the same shape as the #426 finding, in code I wrote.
    """

    @staticmethod
    def _frame(n: int = 6, solar: float = 120.0, wind: float = 6.0) -> pd.DataFrame:
        hours = np.arange(n)
        return pd.DataFrame(
            {
                "temperature_2m": 70.0 + 5.0 * np.sin(hours / 24 * 2 * np.pi),
                "hour_sin": np.sin(hours / 24 * 2 * np.pi),
                "wind_speed_80m": np.full(n, wind),
                "shortwave_radiation": np.full(n, solar),
                "temperature_deviation": np.full(n, 2.0),
            }
        )

    def test_each_driver_moves_by_exactly_its_delta(self):
        """Values, not directions. A sign flip on any of the three survives an
        assertion that only checks the derived feature became positive."""
        base = self._frame()

        out = apply_weather_deltas(base, 10.0, 4.0, 50.0)

        np.testing.assert_allclose(out["temperature_2m"], base["temperature_2m"] + 10.0)
        np.testing.assert_allclose(out["wind_speed_80m"], base["wind_speed_80m"] + 4.0)
        np.testing.assert_allclose(out["shortwave_radiation"], base["shortwave_radiation"] + 50.0)

    def test_a_negative_delta_lowers_the_driver(self):
        """The other direction, on a fixture low enough that an inverted sign
        would clip to zero rather than landing on another plausible number."""
        base = self._frame(solar=120.0, wind=6.0)

        out = apply_weather_deltas(base, -10.0, -4.0, -50.0)

        np.testing.assert_allclose(out["temperature_2m"], base["temperature_2m"] - 10.0)
        np.testing.assert_allclose(out["wind_speed_80m"], np.full(len(base), 2.0))
        np.testing.assert_allclose(out["shortwave_radiation"], np.full(len(base), 70.0))

    def test_the_clip_floor_is_zero_not_one(self):
        """`clip(lower=0.0)`. At 1.0 a becalmed BA reports 1 mph of wind it does
        not have, and `compute_wind_power` cubes its input."""
        calm = self._frame(solar=10.0, wind=2.0)

        out = apply_weather_deltas(calm, 0.0, -10.0, -200.0)

        assert float(out["wind_speed_80m"].min()) == 0.0
        assert float(out["shortwave_radiation"].min()) == 0.0

    def test_a_zero_delta_leaves_its_driver_completely_alone(self):
        """The guard is `if delta and column in ...`, not `or`.

        With `or`, a zero delta still enters the branch — harmless on its own,
        but the same mutation makes a NON-zero delta enter the branch when the
        column is absent, and that raises. This pins the zero half; the next
        test pins the missing-column half.
        """
        base = self._frame()

        out = apply_weather_deltas(base, 0.0, 0.0, 0.0)

        for col in ("temperature_2m", "wind_speed_80m", "shortwave_radiation"):
            np.testing.assert_allclose(out[col], base[col])

    @pytest.mark.parametrize("missing", ["temperature_2m", "wind_speed_80m", "shortwave_radiation"])
    def test_a_missing_driver_column_is_skipped_rather_than_raising(self, missing):
        """Reachable, not defensive: `_build_future_feature_frame` fills what it
        can and a fetch failure can leave a driver absent. The grid runs after
        the forecast is already persisted, so raising here would turn a written
        forecast into a failed region (#268 -> #267)."""
        frame = self._frame().drop(columns=[missing])

        out = apply_weather_deltas(frame, 5.0, 5.0, 5.0)

        assert missing not in out.columns
        assert len(out) == len(frame)

    def test_temp_x_hour_needs_both_of_its_inputs(self):
        """`_recompute_derived_features` guards on temperature AND hour_sin.

        Relaxed to `or`, a frame carrying one but not the other raises inside
        the scoring job's forecast phase.
        """
        no_hour = self._frame().drop(columns=["hour_sin"])

        out = apply_weather_deltas(no_hour, 5.0, 0.0, 0.0)

        assert "temp_x_hour" not in out.columns

    def test_the_defaults_are_no_change(self):
        """Called with no deltas — the way a caller reads the signature — this
        must be an identity on the drivers. Every existing caller passes all
        three explicitly, so the published defaults were never executed."""
        base = self._frame()

        out = apply_weather_deltas(base)

        for col in ("temperature_2m", "wind_speed_80m", "shortwave_radiation"):
            np.testing.assert_allclose(out[col], base[col])
