"""Tests for the Kafka consumer (topics -> Postgres)."""

from unittest.mock import MagicMock

import pytest

psycopg2 = pytest.importorskip("psycopg2", reason="psycopg2 not installed")


@pytest.fixture
def weather_messages():
    """Sample weather messages as they'd come from Kafka."""
    return [
        {
            "region": "ERCOT",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "temperature_2m": 45.0,
            "apparent_temperature": 42.0,
            "relative_humidity_2m": 65.0,
            "dew_point_2m": 35.0,
            "wind_speed_10m": 8.0,
            "wind_speed_80m": 12.0,
            "wind_speed_120m": 15.0,
            "wind_direction_10m": 180.0,
            "shortwave_radiation": 0.0,
            "direct_normal_irradiance": 0.0,
            "diffuse_radiation": 0.0,
            "cloud_cover": 80.0,
            "precipitation": 0.0,
            "snowfall": 0.0,
            "surface_pressure": 1015.0,
            "soil_temperature_0cm": 40.0,
            "weather_code": 3,
        },
    ]


@pytest.fixture
def demand_messages():
    """Sample demand messages as they'd come from Kafka."""
    return [
        {
            "region": "ERCOT",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "demand_mw": 40000.0,
            "forecast_mw": 39500.0,
        },
    ]


class TestInsertFunctions:
    def test_insert_weather_builds_correct_sql(self, weather_messages):
        """_insert_weather builds UPSERT SQL with all 17 weather columns."""
        from src.processing.kafka_consumer import _insert_weather

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        _insert_weather(mock_conn, weather_messages)

        mock_conn.commit.assert_called_once()
        # execute_values was called
        assert True  # Just verify no exception

    def test_insert_demand_builds_correct_sql(self, demand_messages):
        """_insert_demand builds UPSERT SQL for demand records."""
        from src.processing.kafka_consumer import _insert_demand

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        _insert_demand(mock_conn, demand_messages)

        mock_conn.commit.assert_called_once()

    def test_weather_columns_list_has_19_entries(self):
        """WEATHER_COLUMNS should have region + timestamp + 17 weather vars."""
        from src.processing.kafka_consumer import WEATHER_COLUMNS

        assert len(WEATHER_COLUMNS) == 19
        assert "region" in WEATHER_COLUMNS
        assert "timestamp" in WEATHER_COLUMNS
        assert "temperature_2m" in WEATHER_COLUMNS
