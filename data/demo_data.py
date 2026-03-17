"""
Demo data generator for offline/demo mode.

Generates realistic synthetic demand, weather, generation, and alert data
when EIA/Open-Meteo/NOAA APIs are unavailable. Used for:
- Local development without API keys
- Portfolio demos
- CI/CD testing

All data mimics real patterns: daily/weekly seasonality, weather correlations,
and region-specific characteristics.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from config import (
    REGION_CAPACITY_MW,
    REGION_COORDINATES,
)


def generate_demo_demand(
    region: str = "FPL",
    days: int = 90,
) -> pd.DataFrame:
    """
    Generate realistic hourly demand data with daily/weekly seasonality.

    FPL (Florida) characteristics: summer peak from AC, ~30-45 GW range.
    """
    rng = np.random.default_rng(hash(region) % (2**32))
    n = days * 24
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=days)
    timestamps = pd.date_range(start, periods=n, freq="h", tz="UTC")

    hours = np.arange(n)
    day_of_year = timestamps.dayofyear.values

    # Region-specific base load
    capacity = REGION_CAPACITY_MW.get(region, 50000)
    base_load = capacity * 0.45

    # Daily seasonality (peak at 3-4 PM, trough at 4 AM)
    daily = capacity * 0.12 * np.sin(2 * np.pi * (hours % 24 - 6) / 24)

    # Seasonal (summer peak for southern regions, winter peak for northern)
    coords = REGION_COORDINATES.get(region, {"lat": 30})
    if coords["lat"] < 35:  # Southern: summer peak
        seasonal = capacity * 0.15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    else:  # Northern: winter + summer peaks (U-shape)
        seasonal = capacity * 0.10 * np.cos(2 * np.pi * (day_of_year - 200) / 365)

    # Weekly (lower on weekends)
    dow = timestamps.dayofweek.values
    weekly = np.where(dow >= 5, -capacity * 0.05, 0)

    noise = rng.normal(0, capacity * 0.02, n)

    demand = base_load + daily + seasonal + weekly + noise
    demand = np.maximum(demand, capacity * 0.15)

    forecast = demand + rng.normal(0, capacity * 0.015, n)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "demand_mw": np.round(demand, 1),
            "forecast_mw": np.round(forecast, 1),
            "region": region,
        }
    )


def generate_demo_weather(
    region: str = "FPL",
    days: int = 90,
) -> pd.DataFrame:
    """
    Generate realistic hourly weather data matching a region's climate.

    Florida (FPL): hot, humid, moderate wind, high solar.
    Texas (ERCOT): extreme heat summers, occasional cold snaps, high wind.
    """
    rng = np.random.default_rng(hash(region) % (2**32) + 1)
    n = days * 24
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=days)
    timestamps = pd.date_range(start, periods=n, freq="h", tz="UTC")

    hours = np.arange(n)
    day_of_year = timestamps.dayofyear.values

    coords = REGION_COORDINATES.get(region, {"lat": 30, "lon": -80})
    lat = coords["lat"]

    temp_base = 85 - (lat - 25) * 1.2
    temp_seasonal = 20 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    temp_daily = 12 * np.sin(2 * np.pi * (hours % 24 - 6) / 24)
    temp = temp_base + temp_seasonal + temp_daily + rng.normal(0, 4, n)

    if region in ("ERCOT", "SPP"):
        wind_base = 18
    elif region in ("CAISO", "ISONE"):
        wind_base = 12
    else:
        wind_base = 10
    wind_10m = np.abs(wind_base + rng.normal(0, 5, n))
    wind_80m = wind_10m * 1.4
    wind_120m = wind_10m * 1.6

    solar = np.maximum(0, 850 * np.sin(2 * np.pi * (hours % 24 - 6) / 24))
    solar *= 0.7 + 0.3 * rng.random(n)
    solar = np.maximum(solar, 0)

    humidity = np.clip(75 - 0.3 * (temp - 70) + rng.normal(0, 10, n), 10, 100)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature_2m": np.round(temp, 1),
            "apparent_temperature": np.round(temp + 3 * (humidity / 100) - 2, 1),
            "relative_humidity_2m": np.round(humidity, 1),
            "dew_point_2m": np.round(temp - 15 + rng.normal(0, 3, n), 1),
            "wind_speed_10m": np.round(wind_10m, 1),
            "wind_speed_80m": np.round(wind_80m, 1),
            "wind_speed_120m": np.round(wind_120m, 1),
            "wind_direction_10m": np.round(rng.uniform(0, 360, n), 0),
            "shortwave_radiation": np.round(solar, 1),
            "direct_normal_irradiance": np.round(solar * 0.7, 1),
            "diffuse_radiation": np.round(solar * 0.3, 1),
            "cloud_cover": np.round(np.clip(50 + rng.normal(0, 25, n), 0, 100), 0),
            "precipitation": np.round(np.maximum(0, rng.exponential(0.5, n)), 1),
            "snowfall": np.zeros(n),
            "surface_pressure": np.round(1013 + rng.normal(0, 5, n), 1),
            "soil_temperature_0cm": np.round(temp - 5, 1),
            "weather_code": rng.choice([0, 1, 2, 3, 45, 61, 80], n),
        }
    )


def generate_demo_generation(
    region: str = "FPL",
    days: int = 90,
) -> pd.DataFrame:
    """
    Generate realistic generation-by-fuel-type data.

    Fuel types: gas, nuclear, coal, wind, solar, hydro, other.
    """
    rng = np.random.default_rng(hash(region) % (2**32) + 2)
    n = days * 24
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=days)
    timestamps = pd.date_range(start, periods=n, freq="h", tz="UTC")
    hours = np.arange(n)

    capacity = REGION_CAPACITY_MW.get(region, 50000)

    shares = _get_fuel_shares(region)

    records = []
    for fuel, share in shares.items():
        base = capacity * share

        if fuel == "solar":
            gen = base * 2.5 * np.maximum(0, np.sin(2 * np.pi * (hours % 24 - 6) / 24))
            gen *= 0.7 + 0.3 * rng.random(n)
        elif fuel == "wind":
            gen = base * (0.8 + 0.4 * rng.random(n))
        elif fuel == "gas":
            daily_factor = 1 + 0.5 * np.sin(2 * np.pi * (hours % 24 - 6) / 24)
            gen = base * daily_factor + rng.normal(0, base * 0.1, n)
        else:
            gen = base + rng.normal(0, base * 0.05, n)

        gen = np.maximum(gen, 0)

        for i in range(n):
            records.append(
                {
                    "timestamp": timestamps[i],
                    "fuel_type": fuel,
                    "generation_mw": round(float(gen[i]), 1),
                    "region": region,
                }
            )

    return pd.DataFrame(records)


def generate_demo_alerts(region: str = "FPL") -> list[dict]:
    """Generate sample weather alerts for demo mode."""
    alerts = []

    # Generate 0-3 alerts based on region
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)

    if region in ("ERCOT", "CAISO", "FPL"):
        alerts.append(
            {
                "id": f"demo-alert-{region}-1",
                "event": "Heat Advisory",
                "headline": f"Heat Advisory for {region} region until 8 PM local time",
                "description": "Heat index values up to 105°F expected. Elevated demand likely.",
                "severity": "warning",
                "noaa_severity": "Moderate",
                "urgency": "Expected",
                "certainty": "Likely",
                "onset": (now - timedelta(hours=2)).isoformat(),
                "expires": (now + timedelta(hours=6)).isoformat(),
                "areas": [f"{region} service territory"],
                "states": [],
                "balancing_authorities": [region],
            }
        )

    if region in ("ERCOT", "SPP"):
        alerts.append(
            {
                "id": f"demo-alert-{region}-2",
                "event": "Wind Advisory",
                "headline": "Wind Advisory: gusts up to 45 mph",
                "description": "Strong winds may affect transmission lines and wind generation.",
                "severity": "info",
                "noaa_severity": "Minor",
                "urgency": "Expected",
                "certainty": "Possible",
                "onset": now.isoformat(),
                "expires": (now + timedelta(hours=12)).isoformat(),
                "areas": [f"{region} region"],
                "states": [],
                "balancing_authorities": [region],
            }
        )

    return alerts


def _get_fuel_shares(region: str) -> dict[str, float]:
    """Get typical fuel mix shares for a region."""
    shares = {
        "ERCOT": {
            "gas": 0.40,
            "wind": 0.25,
            "solar": 0.10,
            "coal": 0.10,
            "nuclear": 0.10,
            "hydro": 0.02,
            "other": 0.03,
        },
        "CAISO": {
            "gas": 0.35,
            "solar": 0.25,
            "wind": 0.10,
            "hydro": 0.10,
            "nuclear": 0.08,
            "coal": 0.02,
            "other": 0.10,
        },
        "PJM": {
            "gas": 0.35,
            "nuclear": 0.30,
            "coal": 0.15,
            "wind": 0.08,
            "solar": 0.05,
            "hydro": 0.03,
            "other": 0.04,
        },
        "MISO": {
            "gas": 0.30,
            "wind": 0.20,
            "coal": 0.20,
            "nuclear": 0.15,
            "solar": 0.05,
            "hydro": 0.05,
            "other": 0.05,
        },
        "NYISO": {
            "gas": 0.35,
            "nuclear": 0.25,
            "hydro": 0.20,
            "wind": 0.08,
            "solar": 0.05,
            "coal": 0.02,
            "other": 0.05,
        },
        "FPL": {
            "gas": 0.55,
            "nuclear": 0.20,
            "solar": 0.12,
            "coal": 0.03,
            "wind": 0.02,
            "hydro": 0.01,
            "other": 0.07,
        },
        "SPP": {
            "wind": 0.35,
            "gas": 0.30,
            "coal": 0.20,
            "solar": 0.05,
            "nuclear": 0.05,
            "hydro": 0.03,
            "other": 0.02,
        },
        "ISONE": {
            "gas": 0.45,
            "nuclear": 0.25,
            "hydro": 0.08,
            "wind": 0.08,
            "solar": 0.05,
            "coal": 0.02,
            "other": 0.07,
        },
    }
    return shares.get(region, shares["FPL"])
