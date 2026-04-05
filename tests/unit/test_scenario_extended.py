"""Extended unit tests for simulation/scenario_engine.py.

Covers paths not exercised by test_scenario.py:
- simulate_scenario() full pipeline with mocked models
- Feature mutation isolation (copy semantics)
- Override validation (invalid column rejection)
- Multiple simultaneous overrides
- _recompute_derived_features() for temperature deviation, temp_x_hour interaction
- compute_scenario_impact() delta and price calculations
- _run_ensemble() fallback when no models are available
- Edge cases: empty overrides, extreme temps, high wind cutout, boundary CDD/HDD
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from simulation.scenario_engine import (
    _recompute_derived_features,
    _run_ensemble,
    compute_scenario_impact,
    simulate_scenario,
)

# ---------------------------------------------------------------------------
# simulate_scenario() — full pipeline
# ---------------------------------------------------------------------------


class TestSimulateScenarioPipeline:
    """Full pipeline tests for simulate_scenario()."""

    def test_full_pipeline_returns_tuple_of_arrays(self, feature_df):
        """simulate_scenario returns (scenario_forecast, delta) as numpy arrays."""
        base = np.full(len(feature_df), 40000.0)
        mock_models = {}

        with patch(
            "simulation.scenario_engine._run_ensemble",
            return_value=np.full(len(feature_df), 42000.0),
        ):
            scenario, delta = simulate_scenario(
                feature_df,
                {"temperature_2m": 100.0},
                mock_models,
                base_forecast=base,
            )

        assert isinstance(scenario, np.ndarray)
        assert isinstance(delta, np.ndarray)
        assert len(scenario) == len(feature_df)
        assert len(delta) == len(feature_df)

    def test_delta_equals_scenario_minus_baseline(self, feature_df):
        """delta = scenario_forecast - base_forecast."""
        base = np.full(len(feature_df), 40000.0)
        scenario_values = np.full(len(feature_df), 45000.0)

        with patch(
            "simulation.scenario_engine._run_ensemble",
            return_value=scenario_values,
        ):
            scenario, delta = simulate_scenario(
                feature_df,
                {"temperature_2m": 105.0},
                {},
                base_forecast=base,
            )

        np.testing.assert_array_almost_equal(delta, scenario_values - base)

    def test_input_dataframe_not_mutated(self, feature_df):
        """Copy semantics: original feature_df must remain unchanged after simulation."""
        original_temp = feature_df["temperature_2m"].copy()
        original_wind = feature_df["wind_speed_80m"].copy()
        base = np.full(len(feature_df), 40000.0)

        with patch(
            "simulation.scenario_engine._run_ensemble",
            return_value=np.full(len(feature_df), 42000.0),
        ):
            simulate_scenario(
                feature_df,
                {"temperature_2m": 120.0, "wind_speed_80m": 0.0},
                {},
                base_forecast=base,
            )

        # Original DataFrame must be completely unchanged
        pd.testing.assert_series_equal(feature_df["temperature_2m"], original_temp)
        pd.testing.assert_series_equal(feature_df["wind_speed_80m"], original_wind)

    def test_invalid_override_column_raises_valueerror(self, feature_df):
        """Unknown override column raises ValueError with helpful message."""
        with pytest.raises(ValueError, match="Unknown weather column.*bogus_column"):
            simulate_scenario(feature_df, {"bogus_column": 99.0}, {})

    def test_multiple_overrides_applied_simultaneously(self, feature_df):
        """Multiple weather columns are overridden in a single call."""
        base = np.full(len(feature_df), 40000.0)
        overrides = {
            "temperature_2m": 110.0,
            "wind_speed_80m": 5.0,
            "shortwave_radiation": 900.0,
        }

        captured_features = {}

        def capturing_ensemble(features, models):
            captured_features["df"] = features.copy()
            return np.full(len(features), 43000.0)

        with patch(
            "simulation.scenario_engine._run_ensemble",
            side_effect=capturing_ensemble,
        ):
            simulate_scenario(feature_df, overrides, {}, base_forecast=base)

        # The second call to _run_ensemble gets the scenario features
        scenario_df = captured_features["df"]
        assert (scenario_df["temperature_2m"] == 110.0).all()
        assert (scenario_df["wind_speed_80m"] == 5.0).all()
        assert (scenario_df["shortwave_radiation"] == 900.0).all()

    def test_empty_overrides_returns_zero_delta(self, feature_df):
        """Empty overrides dict produces zero deltas (scenario == baseline)."""
        base = np.full(len(feature_df), 40000.0)

        with patch(
            "simulation.scenario_engine._run_ensemble",
            return_value=base.copy(),
        ):
            scenario, delta = simulate_scenario(
                feature_df,
                {},
                {},
                base_forecast=base,
            )

        # With identical ensemble output and no overrides, delta should be zero
        np.testing.assert_array_almost_equal(delta, np.zeros(len(feature_df)))

    def test_base_forecast_computed_when_none(self, feature_df):
        """When base_forecast is None, _run_ensemble is called twice (base + scenario)."""
        call_count = {"n": 0}

        def counting_ensemble(features, models):
            call_count["n"] += 1
            return np.full(len(features), 40000.0 + call_count["n"] * 1000)

        with patch(
            "simulation.scenario_engine._run_ensemble",
            side_effect=counting_ensemble,
        ):
            scenario, delta = simulate_scenario(
                feature_df,
                {"temperature_2m": 100.0},
                {},
                base_forecast=None,
            )

        # Should be called twice: once for base, once for scenario
        assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# _recompute_derived_features() — paths not covered by test_scenario.py
# ---------------------------------------------------------------------------


class TestRecomputeDerivedFeaturesExtended:
    """Additional derived feature recomputation tests."""

    def test_temperature_deviation_recomputed(self, feature_df):
        """temperature_deviation is recomputed after temp override."""
        df = feature_df.copy()
        # Set all temps to uniform 80F: deviation from rolling avg should be ~0
        df["temperature_2m"] = 80.0
        result = _recompute_derived_features(df)
        # With a constant temperature, deviation should be exactly zero
        # (since value == rolling mean for all rows)
        np.testing.assert_array_almost_equal(result["temperature_deviation"].values, 0.0, decimal=5)

    def test_temp_x_hour_interaction_recomputed(self, feature_df):
        """temp_x_hour interaction is recomputed when temperature changes."""
        df = feature_df.copy()
        original_interaction = df["temp_x_hour"].copy()
        df["temperature_2m"] = 100.0
        result = _recompute_derived_features(df)

        # Interaction should be 100 * hour_sin
        expected = 100.0 * result["hour_sin"]
        pd.testing.assert_series_equal(result["temp_x_hour"], expected, check_names=False)
        # And it should differ from the original (since temp changed)
        assert not np.allclose(result["temp_x_hour"].values, original_interaction.values)

    def test_cdd_at_exactly_65f_baseline(self, feature_df):
        """At exactly 65F (the baseline), both CDD and HDD should be zero."""
        df = feature_df.copy()
        df["temperature_2m"] = 65.0
        result = _recompute_derived_features(df)
        assert (result["cooling_degree_days"] == 0.0).all()
        assert (result["heating_degree_days"] == 0.0).all()

    def test_extreme_heat_cdd(self, feature_df):
        """Extreme heat (130F) produces CDD = 65."""
        df = feature_df.copy()
        df["temperature_2m"] = 130.0
        result = _recompute_derived_features(df)
        assert (result["cooling_degree_days"] == 65.0).all()
        assert (result["heating_degree_days"] == 0.0).all()

    def test_extreme_cold_hdd(self, feature_df):
        """Extreme cold (-20F) produces HDD = 85."""
        df = feature_df.copy()
        df["temperature_2m"] = -20.0
        result = _recompute_derived_features(df)
        assert (result["heating_degree_days"] == 85.0).all()
        assert (result["cooling_degree_days"] == 0.0).all()

    def test_high_wind_above_cutout_gives_zero_power(self, feature_df):
        """Wind above cutout speed (56 mph) should produce zero wind power."""
        df = feature_df.copy()
        df["wind_speed_80m"] = 60.0  # above 56 mph cutout
        result = _recompute_derived_features(df)
        assert (result["wind_power_estimate"] == 0.0).all()

    def test_moderate_wind_gives_nonzero_power(self, feature_df):
        """Moderate wind (20 mph) should produce positive wind power."""
        df = feature_df.copy()
        df["wind_speed_80m"] = 20.0
        result = _recompute_derived_features(df)
        assert (result["wind_power_estimate"] > 0.0).all()

    def test_high_radiation_solar_cf_capped_at_one(self, feature_df):
        """Radiation above 1000 W/m2 should produce solar CF capped at 1.0."""
        df = feature_df.copy()
        df["shortwave_radiation"] = 1500.0
        result = _recompute_derived_features(df)
        assert (result["solar_capacity_factor"] == 1.0).all()

    def test_moderate_radiation_solar_cf(self, feature_df):
        """500 W/m2 should produce solar CF = 0.5."""
        df = feature_df.copy()
        df["shortwave_radiation"] = 500.0
        result = _recompute_derived_features(df)
        np.testing.assert_array_almost_equal(result["solar_capacity_factor"].values, 0.5)

    def test_missing_columns_skipped_gracefully(self):
        """Recompute handles DataFrames missing some weather columns."""
        df = pd.DataFrame({"unrelated_col": [1, 2, 3]})
        result = _recompute_derived_features(df)
        # Should return without error and not add any derived columns
        assert "cooling_degree_days" not in result.columns
        assert "wind_power_estimate" not in result.columns
        assert "solar_capacity_factor" not in result.columns


# ---------------------------------------------------------------------------
# compute_scenario_impact() — delta and price impact
# ---------------------------------------------------------------------------


class TestComputeScenarioImpact:
    """Tests for compute_scenario_impact()."""

    def test_positive_delta_when_scenario_exceeds_baseline(self):
        """Scenario > baseline produces positive demand delta."""
        base = np.array([40000.0, 41000.0, 42000.0])
        scenario = np.array([45000.0, 46000.0, 47000.0])
        result = compute_scenario_impact(scenario, base, "ERCOT")

        assert (result["demand_delta_mw"] > 0).all()
        assert result["peak_delta_mw"] > 0

    def test_negative_delta_when_scenario_below_baseline(self):
        """Scenario < baseline produces negative demand delta."""
        base = np.array([40000.0, 41000.0, 42000.0])
        scenario = np.array([35000.0, 36000.0, 37000.0])
        result = compute_scenario_impact(scenario, base, "ERCOT")

        assert (result["demand_delta_mw"] < 0).all()
        assert result["peak_delta_mw"] < 0

    def test_percentage_delta_sign_correctness(self):
        """Demand delta percentage should be positive when scenario increases."""
        base = np.array([40000.0, 40000.0])
        scenario = np.array([44000.0, 44000.0])  # 10% increase
        result = compute_scenario_impact(scenario, base, "ERCOT")

        expected_pct = (scenario - base) / base * 100
        np.testing.assert_array_almost_equal(result["demand_delta_pct"], expected_pct)
        assert (result["demand_delta_pct"] > 0).all()

    def test_reserve_margin_decreases_with_higher_demand(self):
        """Higher demand should reduce reserve margin."""
        base = np.array([40000.0])
        scenario_low = np.array([40000.0])
        scenario_high = np.array([100000.0])

        result_low = compute_scenario_impact(scenario_low, base, "ERCOT")
        result_high = compute_scenario_impact(scenario_high, base, "ERCOT")

        assert result_high["min_reserve_margin_pct"] < result_low["min_reserve_margin_pct"]

    def test_unknown_region_uses_default_capacity(self):
        """Unknown region falls back to 100,000 MW default capacity."""
        base = np.array([50000.0])
        scenario = np.array([50000.0])
        result = compute_scenario_impact(scenario, base, "UNKNOWN_REGION")

        # capacity defaults to 100_000; reserve = (100000 - 50000)/100000 * 100 = 50%
        assert result["min_reserve_margin_pct"] == pytest.approx(50.0)

    def test_price_delta_positive_when_demand_increases(self):
        """Higher scenario demand should produce higher (or equal) price."""
        base = np.array([40000.0])
        scenario = np.array([120000.0])  # near ERCOT capacity
        result = compute_scenario_impact(scenario, base, "ERCOT")

        assert result["price_delta"] >= 0


# ---------------------------------------------------------------------------
# _run_ensemble() — fallback behavior
# ---------------------------------------------------------------------------


class TestRunEnsemble:
    """Tests for _run_ensemble() fallback paths."""

    def test_no_models_returns_zeros(self, feature_df):
        """When no models are available, _run_ensemble returns zeros."""
        result = _run_ensemble(feature_df, {})
        np.testing.assert_array_equal(result, np.zeros(len(feature_df)))

    def test_xgboost_failure_logged_and_skipped(self, feature_df):
        """XGBoost failure is caught and logged; other models still used."""
        broken_xgb_model = MagicMock()
        broken_xgb_model.predict.side_effect = RuntimeError("broken")

        models = {
            "xgboost_model": broken_xgb_model,
            "xgboost_feature_names": ["temperature_2m"],
        }

        # With XGBoost failing and no other models, should return zeros
        result = _run_ensemble(feature_df, models)
        np.testing.assert_array_equal(result, np.zeros(len(feature_df)))
