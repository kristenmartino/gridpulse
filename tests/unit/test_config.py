"""Unit tests for config.py — validates all constants and lookups."""

import pytest

from config import (
    REGION_COORDINATES,
    REGION_CAPACITY_MW,
    STATE_TO_BA,
    BA_FOR_STATE,
    WEATHER_VARIABLES,
    TAB_IDS,
    TAB_LABELS,
    STALENESS_THRESHOLDS_SECONDS,
    CDD_HDD_BASELINE_F,
    WIND_CUTOUT_SPEED_MS,
    MPH_TO_MS,
)


class TestRegionConfig:
    def test_eight_regions(self):
        assert len(REGION_COORDINATES) == 8

    def test_all_regions_have_coordinates(self):
        for region, coords in REGION_COORDINATES.items():
            assert "lat" in coords, f"{region} missing lat"
            assert "lon" in coords, f"{region} missing lon"
            assert "name" in coords, f"{region} missing name"

    def test_fpl_is_nextera(self):
        assert "NextEra" in REGION_COORDINATES["FPL"]["name"]

    def test_all_regions_have_capacity(self):
        for region in REGION_COORDINATES:
            assert region in REGION_CAPACITY_MW, f"{region} missing capacity"
            assert REGION_CAPACITY_MW[region] > 0


class TestStateMapping:
    def test_all_regions_have_states(self):
        for region in REGION_COORDINATES:
            assert region in STATE_TO_BA, f"{region} missing state mapping"
            assert len(STATE_TO_BA[region]) > 0

    def test_reverse_lookup(self):
        assert "FL" in BA_FOR_STATE
        assert "FPL" in BA_FOR_STATE["FL"]

    def test_texas_in_multiple_bas(self):
        assert len(BA_FOR_STATE.get("TX", [])) >= 2


class TestWeatherVariables:
    def test_seventeen_variables(self):
        assert len(WEATHER_VARIABLES) == 17

    def test_key_variables_present(self):
        assert "temperature_2m" in WEATHER_VARIABLES
        assert "wind_speed_80m" in WEATHER_VARIABLES
        assert "shortwave_radiation" in WEATHER_VARIABLES


class TestTabs:
    def test_three_tabs(self):
        assert len(TAB_IDS) == 3
        assert len(TAB_LABELS) == 3

    def test_tab_ids_match_labels(self):
        for tab_id in TAB_IDS:
            assert tab_id in TAB_LABELS


class TestConstants:
    def test_cdd_baseline(self):
        assert CDD_HDD_BASELINE_F == 65.0

    def test_wind_cutout(self):
        assert WIND_CUTOUT_SPEED_MS == 25.0

    def test_mph_to_ms(self):
        assert MPH_TO_MS == pytest.approx(0.44704)

    def test_staleness_thresholds(self):
        assert STALENESS_THRESHOLDS_SECONDS["weather"] == 7200
        assert STALENESS_THRESHOLDS_SECONDS["generation"] == 300
