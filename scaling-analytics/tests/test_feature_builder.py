"""Tests for the FeatureBuilder wrapper around v1 feature_engineering.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

HAS_PSYCOPG2 = pytest.importorskip("psycopg2", reason="psycopg2 not installed") is not None


@pytest.fixture
def sample_demand():
    """Realistic demand DataFrame matching v1 schema."""
    n = 500
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    hours = np.arange(n)
    demand = 40000 + 5000 * np.sin(2 * np.pi * (hours - 6) / 24) + np.random.normal(0, 500, n)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "demand_mw": demand,
            "forecast_mw": demand + np.random.normal(0, 1000, n),
            "region": "ERCOT",
        }
    )


@pytest.fixture
def sample_weather():
    """Realistic weather DataFrame matching v1 schema."""
    n = 500
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    hours = np.arange(n)
    temp = 75 + 10 * np.sin(2 * np.pi * (hours - 6) / 24) + np.random.normal(0, 3, n)
    solar = np.maximum(0, 800 * np.sin(2 * np.pi * (hours % 24 - 6) / 24))

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature_2m": temp,
            "apparent_temperature": temp - 3,
            "relative_humidity_2m": np.clip(60 + np.random.normal(0, 15, n), 10, 100),
            "dew_point_2m": temp - 15,
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
            "weather_code": np.random.choice([0, 1, 2, 3], n),
        }
    )


class TestFeatureBuilderDirect:
    """Test that FeatureBuilder correctly delegates to v1 code."""

    def test_v1_engineer_features_produces_43_features(self, sample_demand, sample_weather):
        """v1's engineer_features() produces the canonical 43 features."""
        from data.feature_engineering import engineer_features, get_feature_names
        from data.preprocessing import merge_demand_weather

        merged = merge_demand_weather(sample_demand, sample_weather)
        featured = engineer_features(merged)

        expected_names = get_feature_names()
        for name in expected_names:
            assert name in featured.columns, f"Missing feature: {name}"

    def test_get_feature_columns_returns_canonical_list(self):
        """FeatureBuilder.get_feature_columns() returns 43 feature names."""
        from src.processing.feature_builder import FeatureBuilder

        cols = FeatureBuilder.get_feature_columns()
        assert isinstance(cols, list)
        assert len(cols) > 40
        assert "cooling_degree_days" in cols
        assert "demand_lag_24h" in cols
        assert "temp_x_hour" in cols

    def test_v1_feature_names_match_builder(self):
        """FeatureBuilder delegates to v1's get_feature_names()."""
        from src.processing.feature_builder import FeatureBuilder

        from data.feature_engineering import get_feature_names

        assert FeatureBuilder.get_feature_columns() == get_feature_names()

    def test_v1_feature_names_count(self):
        """v1 get_feature_names() returns exactly 43 features."""
        from data.feature_engineering import get_feature_names

        names = get_feature_names()
        assert len(names) == 43
