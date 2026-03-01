"""
Shared test fixtures for the Energy Forecast Dashboard.

Generates realistic synthetic data for testing without API calls.
All fixtures produce data matching the exact schemas expected by
the data pipeline and models.
"""

import os

# Disable precomputation during tests — must be set before any app imports
os.environ["PRECOMPUTE_ENABLED"] = "false"

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_timestamps():
    """90 days of hourly UTC timestamps."""
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return pd.date_range(start, periods=90 * 24, freq="h", tz="UTC")


@pytest.fixture
def sample_demand_df(sample_timestamps):
    """
    Realistic hourly demand data for testing.

    Simulates daily/weekly seasonality + noise.
    Schema: [timestamp, demand_mw, forecast_mw, region]
    """
    n = len(sample_timestamps)
    hours = np.arange(n)

    # Daily seasonality (peak at 3pm, trough at 4am)
    daily = 5000 * np.sin(2 * np.pi * (hours - 6) / 24)

    # Weekly seasonality (lower on weekends)
    dow = sample_timestamps.dayofweek
    weekly = np.where(dow >= 5, -3000, 0)

    # Base load + seasonality + noise
    demand = 40000 + daily + weekly + np.random.normal(0, 500, n)
    demand = np.maximum(demand, 5000)  # Floor

    # Forecast with some error
    forecast = demand + np.random.normal(0, 1000, n)

    return pd.DataFrame(
        {
            "timestamp": sample_timestamps,
            "demand_mw": demand,
            "forecast_mw": forecast,
            "region": "ERCOT",
        }
    )


@pytest.fixture
def sample_weather_df(sample_timestamps):
    """
    Realistic hourly weather data for testing.

    All values in Fahrenheit / mph per Open-Meteo config.
    Schema: [timestamp] + all 17 WEATHER_VARIABLES
    """
    n = len(sample_timestamps)
    hours = np.arange(n)

    # Temperature: daily cycle around 75°F with seasonal drift
    day_of_year = sample_timestamps.dayofyear
    seasonal = 20 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Peak in summer
    daily_temp = 10 * np.sin(2 * np.pi * (hours - 6) / 24)
    temp = 75 + seasonal + daily_temp + np.random.normal(0, 3, n)

    # Solar: follows daylight pattern
    solar = np.maximum(0, 800 * np.sin(2 * np.pi * (hours % 24 - 6) / 24))
    solar = solar * (0.8 + 0.2 * np.random.random(n))  # Cloud variability

    return pd.DataFrame(
        {
            "timestamp": sample_timestamps,
            "temperature_2m": temp,
            "apparent_temperature": temp - 3 + np.random.normal(0, 1, n),
            "relative_humidity_2m": np.clip(60 + np.random.normal(0, 15, n), 10, 100),
            "dew_point_2m": temp - 15 + np.random.normal(0, 3, n),
            "wind_speed_10m": np.abs(10 + np.random.normal(0, 5, n)),
            "wind_speed_80m": np.abs(15 + np.random.normal(0, 6, n)),
            "wind_speed_120m": np.abs(18 + np.random.normal(0, 7, n)),
            "wind_direction_10m": np.random.uniform(0, 360, n),
            "shortwave_radiation": solar,
            "direct_normal_irradiance": solar * 0.7,
            "diffuse_radiation": solar * 0.3,
            "cloud_cover": np.clip(50 + np.random.normal(0, 25, n), 0, 100),
            "precipitation": np.maximum(0, np.random.exponential(0.5, n)),
            "snowfall": np.zeros(n),
            "surface_pressure": 1013 + np.random.normal(0, 5, n),
            "soil_temperature_0cm": temp - 5,
            "weather_code": np.random.choice([0, 1, 2, 3, 45, 61, 80], n),
        }
    )


@pytest.fixture
def merged_df(sample_demand_df, sample_weather_df):
    """Merged demand + weather DataFrame."""
    from data.preprocessing import merge_demand_weather

    return merge_demand_weather(sample_demand_df, sample_weather_df)


@pytest.fixture
def feature_df(merged_df):
    """Fully feature-engineered DataFrame ready for model training."""
    from data.feature_engineering import engineer_features

    return engineer_features(merged_df)


@pytest.fixture
def mock_eia_response():
    """Mocked EIA API JSON response."""
    return {
        "response": {
            "total": 3,
            "data": [
                {"period": "2024-01-01T00", "value": 40000, "respondent": "ERCOT", "type": "D"},
                {"period": "2024-01-01T01", "value": 39500, "respondent": "ERCOT", "type": "D"},
                {"period": "2024-01-01T00", "value": 41000, "respondent": "ERCOT", "type": "DF"},
            ],
        }
    }


@pytest.fixture
def mock_weather_response():
    """Mocked Open-Meteo API JSON response."""
    return {
        "hourly": {
            "time": ["2024-01-01T00:00", "2024-01-01T01:00", "2024-01-01T02:00"],
            "temperature_2m": [45.0, 44.0, 43.5],
            "apparent_temperature": [42.0, 41.0, 40.0],
            "relative_humidity_2m": [65.0, 67.0, 70.0],
            "dew_point_2m": [35.0, 34.0, 33.5],
            "wind_speed_10m": [8.0, 9.0, 7.5],
            "wind_speed_80m": [12.0, 13.5, 11.0],
            "wind_speed_120m": [15.0, 16.0, 14.0],
            "wind_direction_10m": [180.0, 185.0, 175.0],
            "shortwave_radiation": [0.0, 0.0, 0.0],
            "direct_normal_irradiance": [0.0, 0.0, 0.0],
            "diffuse_radiation": [0.0, 0.0, 0.0],
            "cloud_cover": [80.0, 85.0, 90.0],
            "precipitation": [0.0, 0.1, 0.0],
            "snowfall": [0.0, 0.0, 0.0],
            "surface_pressure": [1015.0, 1014.5, 1014.0],
            "soil_temperature_0cm": [40.0, 39.5, 39.0],
            "weather_code": [3, 3, 45],
        }
    }


@pytest.fixture
def mock_noaa_alerts_response():
    """Mocked NOAA alerts API response."""
    return {
        "features": [
            {
                "properties": {
                    "id": "urn:oid:2.49.0.1.840.0.alert-1",
                    "event": "Heat Advisory",
                    "headline": "Heat Advisory until 8PM CDT",
                    "description": "Dangerously hot conditions with heat index values up to 110.",
                    "severity": "Moderate",
                    "urgency": "Expected",
                    "certainty": "Likely",
                    "onset": "2024-07-15T12:00:00-05:00",
                    "expires": "2024-07-15T20:00:00-05:00",
                    "areaDesc": "Travis County; Williamson County",
                }
            },
            {
                "properties": {
                    "id": "urn:oid:2.49.0.1.840.0.alert-2",
                    "event": "Excessive Heat Warning",
                    "headline": "Excessive Heat Warning until 9PM CDT",
                    "description": "Heat index values up to 115 expected.",
                    "severity": "Extreme",
                    "urgency": "Immediate",
                    "certainty": "Observed",
                    "onset": "2024-07-15T10:00:00-05:00",
                    "expires": "2024-07-15T21:00:00-05:00",
                    "areaDesc": "Dallas County; Tarrant County",
                }
            },
        ]
    }


@pytest.fixture
def tmp_cache(tmp_path):
    """Temporary SQLite cache for testing."""
    from data.cache import Cache

    return Cache(db_path=str(tmp_path / "test_cache.db"), default_ttl=3600)
