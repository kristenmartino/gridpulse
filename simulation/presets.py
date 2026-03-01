"""
Historical extreme scenario presets for the Scenario Simulator.

Per spec §Preset Historical Scenarios (Tab 6):
Each preset defines the weather conditions during a real historical event.
One-click replay lets users instantly see the grid impact.

Weather values are in Fahrenheit and mph per Open-Meteo request params.
"""

PRESETS: dict[str, dict] = {
    "winter_storm_uri": {
        "name": "Winter Storm Uri",
        "description": "February 2021 — Texas grid collapse. Temperatures below 0°F, "
        "widespread freezing, wind turbines iced over, natural gas "
        "infrastructure froze. 69 deaths, 4.5M without power.",
        "date": "2021-02-15",
        "region": "ERCOT",
        "weather": {
            "temperature_2m": 5.0,  # °F — extreme cold for Texas
            "apparent_temperature": -10.0,  # Wind chill
            "wind_speed_80m": 8.0,  # mph — low wind (ice on turbines)
            "wind_speed_120m": 10.0,
            "shortwave_radiation": 50.0,  # W/m² — overcast
            "cloud_cover": 95.0,  # %
            "relative_humidity_2m": 85.0,  # %
            "precipitation": 5.0,  # mm — freezing rain/sleet
            "snowfall": 10.0,  # cm
        },
    },
    "summer_2023_heat_dome": {
        "name": "Summer 2023 Heat Dome",
        "description": "July 2023 — Record-breaking heat across the Southwest. "
        "Phoenix hit 110°F+ for 31 consecutive days. "
        "Record electricity demand across CAISO, ERCOT, SPP.",
        "date": "2023-07-20",
        "region": "CAISO",
        "weather": {
            "temperature_2m": 112.0,  # °F — extreme heat
            "apparent_temperature": 118.0,  # Heat index
            "wind_speed_80m": 6.0,  # mph — stagnant air
            "wind_speed_120m": 8.0,
            "shortwave_radiation": 950.0,  # W/m² — intense sun
            "cloud_cover": 5.0,  # % — clear sky
            "relative_humidity_2m": 15.0,  # % — desert dry
        },
    },
    "polar_vortex_2019": {
        "name": "Polar Vortex 2019",
        "description": "January 2019 — Chicago hit -23°F, coldest in decades. "
        "MISO demand surged as heating systems maxed out. "
        "Rolling blackouts narrowly avoided.",
        "date": "2019-01-30",
        "region": "MISO",
        "weather": {
            "temperature_2m": -15.0,  # °F — extreme Midwest cold
            "apparent_temperature": -40.0,  # Wind chill advisory
            "wind_speed_80m": 25.0,  # mph — strong wind
            "wind_speed_120m": 30.0,
            "shortwave_radiation": 100.0,  # W/m² — winter sun
            "cloud_cover": 80.0,  # %
            "relative_humidity_2m": 70.0,  # %
            "snowfall": 15.0,  # cm
        },
    },
    "california_heat_wave_2022": {
        "name": "California Heat Wave 2022",
        "description": "September 2022 — All-time record demand in CAISO. "
        "Flex Alert issued asking consumers to reduce usage. "
        "Rolling blackout risk declared.",
        "date": "2022-09-06",
        "region": "CAISO",
        "weather": {
            "temperature_2m": 108.0,  # °F
            "apparent_temperature": 112.0,
            "wind_speed_80m": 5.0,  # mph — calm
            "wind_speed_120m": 7.0,
            "shortwave_radiation": 900.0,  # W/m²
            "cloud_cover": 10.0,  # %
            "relative_humidity_2m": 20.0,  # %
        },
    },
    "hurricane_irma": {
        "name": "Hurricane Irma",
        "description": "September 2017 — Category 4 hurricane struck Florida. "
        "FPL (NextEra's subsidiary) lost power to 4.4M customers. "
        "Largest power restoration in US history.",
        "date": "2017-09-10",
        "region": "FPL",
        "weather": {
            "temperature_2m": 82.0,  # °F — warm tropical
            "apparent_temperature": 90.0,
            "wind_speed_80m": 80.0,  # mph — hurricane force
            "wind_speed_120m": 95.0,
            "shortwave_radiation": 50.0,  # W/m² — heavy cloud/rain
            "cloud_cover": 100.0,  # % — total overcast
            "relative_humidity_2m": 98.0,  # % — saturated
            "precipitation": 100.0,  # mm — extreme rain
        },
    },
    "solar_eclipse_2024": {
        "name": "2024 Solar Eclipse",
        "description": "April 8, 2024 — Total solar eclipse crossed the US. "
        "Solar generation dropped to near-zero during totality. "
        "PJM and ERCOT saw rapid solar ramp-down and ramp-up.",
        "date": "2024-04-08",
        "region": "PJM",
        "weather": {
            "temperature_2m": 65.0,  # °F — mild spring day
            "apparent_temperature": 63.0,
            "wind_speed_80m": 12.0,  # mph — normal
            "wind_speed_120m": 15.0,
            "shortwave_radiation": 10.0,  # W/m² — near zero during totality
            "cloud_cover": 100.0,  # % — eclipse blocks sun
            "relative_humidity_2m": 50.0,  # %
        },
    },
}


def get_preset(name: str) -> dict:
    """
    Get a preset scenario by name.

    Args:
        name: Preset key (e.g., "winter_storm_uri").

    Returns:
        Preset dict with 'name', 'description', 'date', 'region', 'weather'.

    Raises:
        KeyError: If preset name not found.
    """
    if name not in PRESETS:
        raise KeyError(f"Unknown preset: '{name}'. Available: {list(PRESETS.keys())}")
    return PRESETS[name]


def list_presets() -> list[dict]:
    """
    List all available presets with summary info.

    Returns:
        List of dicts with 'key', 'name', 'date', 'region', 'description'.
    """
    return [
        {
            "key": key,
            "name": preset["name"],
            "date": preset["date"],
            "region": preset["region"],
            "description": preset["description"][:100] + "...",
        }
        for key, preset in PRESETS.items()
    ]
