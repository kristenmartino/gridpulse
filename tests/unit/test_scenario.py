"""Unit tests for simulation/scenario_engine.py and simulation/presets.py."""

import pytest

from simulation.presets import PRESETS, get_preset, list_presets
from simulation.scenario_engine import (
    OVERRIDABLE_COLUMNS,
    _recompute_derived_features,
)


class TestPresets:
    """Historical extreme scenario presets."""

    def test_six_presets_exist(self):
        assert len(PRESETS) == 6

    def test_get_preset_by_name(self):
        preset = get_preset("winter_storm_uri")
        assert preset["name"] == "Winter Storm Uri"
        assert preset["region"] == "ERCOT"
        assert "temperature_2m" in preset["weather"]

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError, match="Unknown preset"):
            get_preset("nonexistent")

    def test_all_presets_have_required_fields(self):
        for key, preset in PRESETS.items():
            assert "name" in preset, f"Preset {key} missing 'name'"
            assert "description" in preset, f"Preset {key} missing 'description'"
            assert "date" in preset, f"Preset {key} missing 'date'"
            assert "region" in preset, f"Preset {key} missing 'region'"
            assert "weather" in preset, f"Preset {key} missing 'weather'"
            assert "temperature_2m" in preset["weather"], f"Preset {key} missing temperature"

    def test_list_presets(self):
        result = list_presets()
        assert len(result) == 6
        for item in result:
            assert "key" in item
            assert "name" in item

    def test_winter_storm_uri_extreme_cold(self):
        preset = get_preset("winter_storm_uri")
        assert preset["weather"]["temperature_2m"] < 20  # Very cold

    def test_hurricane_irma_high_wind(self):
        preset = get_preset("hurricane_irma")
        assert preset["weather"]["wind_speed_80m"] > 70  # Hurricane force

    def test_heat_dome_extreme_heat(self):
        preset = get_preset("summer_2023_heat_dome")
        assert preset["weather"]["temperature_2m"] > 100  # Extreme heat


class TestRecomputeDerivedFeatures:
    """Test that derived features update after weather override."""

    def test_cdd_updates_with_temperature(self, feature_df):
        """If we override temp to 100°F, CDD should = 35."""
        df = feature_df.copy()
        df["temperature_2m"] = 100.0
        result = _recompute_derived_features(df)
        assert (result["cooling_degree_days"] == 35.0).all()
        assert (result["heating_degree_days"] == 0.0).all()

    def test_hdd_updates_with_cold(self, feature_df):
        """If we override temp to 30°F, HDD should = 35."""
        df = feature_df.copy()
        df["temperature_2m"] = 30.0
        result = _recompute_derived_features(df)
        assert (result["heating_degree_days"] == 35.0).all()
        assert (result["cooling_degree_days"] == 0.0).all()

    def test_wind_power_updates(self, feature_df):
        """Zero wind → zero wind power."""
        df = feature_df.copy()
        df["wind_speed_80m"] = 0.0
        result = _recompute_derived_features(df)
        assert (result["wind_power_estimate"] == 0.0).all()

    def test_solar_updates(self, feature_df):
        """Zero radiation → zero solar CF."""
        df = feature_df.copy()
        df["shortwave_radiation"] = 0.0
        result = _recompute_derived_features(df)
        assert (result["solar_capacity_factor"] == 0.0).all()


class TestOverridableColumns:
    """Validate the set of columns that can be overridden."""

    def test_includes_all_weather_variables(self):
        from config import WEATHER_VARIABLES

        for var in WEATHER_VARIABLES:
            assert var in OVERRIDABLE_COLUMNS

    def test_includes_derived_energy_features(self):
        for col in [
            "cooling_degree_days",
            "heating_degree_days",
            "wind_power_estimate",
            "solar_capacity_factor",
        ]:
            assert col in OVERRIDABLE_COLUMNS
