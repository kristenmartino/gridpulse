"""Tests for Kafka producers (weather + grid demand)."""
import json
from unittest.mock import patch, MagicMock, call

import pandas as pd
import numpy as np
import pytest

pytest.importorskip("confluent_kafka", reason="confluent_kafka not installed")


@pytest.fixture
def mock_producer():
    """Mocked confluent_kafka.Producer."""
    producer = MagicMock()
    producer.produce = MagicMock()
    producer.flush = MagicMock()
    return producer


@pytest.fixture
def small_weather_df():
    """Small weather DataFrame for testing."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        "temperature_2m": [45.0, 46.0, 47.0],
        "wind_speed_80m": [12.0, 13.0, 11.0],
    })


@pytest.fixture
def small_demand_df():
    """Small demand DataFrame for testing."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        "demand_mw": [40000.0, 41000.0, 42000.0],
        "forecast_mw": [39500.0, 40500.0, 41500.0],
        "region": "ERCOT",
    })


class TestWeatherProducer:

    @patch("src.ingestion.weather_producer.Producer")
    @patch("src.ingestion.weather_producer.fetch_weather")
    def test_produces_to_correct_topic(self, mock_fetch, MockProducer, small_weather_df):
        """Weather messages go to the 'weather-raw' topic."""
        mock_fetch.return_value = small_weather_df
        producer_instance = MagicMock()
        MockProducer.return_value = producer_instance

        from src.config import KafkaConfig
        from src.ingestion.weather_producer import produce_weather

        config = KafkaConfig()
        # Only test with one region to keep it simple
        with patch("src.ingestion.weather_producer.GRID_REGIONS", ["FPL"]):
            produce_weather(config)

        # Check that produce was called with the weather topic
        assert producer_instance.produce.called
        first_call = producer_instance.produce.call_args_list[0]
        assert first_call.kwargs["topic"] == "weather-raw"

    @patch("src.ingestion.weather_producer.Producer")
    @patch("src.ingestion.weather_producer.fetch_weather")
    def test_message_key_is_region(self, mock_fetch, MockProducer, small_weather_df):
        """Message key is the region name encoded as bytes."""
        mock_fetch.return_value = small_weather_df
        producer_instance = MagicMock()
        MockProducer.return_value = producer_instance

        from src.config import KafkaConfig
        from src.ingestion.weather_producer import produce_weather

        with patch("src.ingestion.weather_producer.GRID_REGIONS", ["FPL"]):
            produce_weather(KafkaConfig())

        first_call = producer_instance.produce.call_args_list[0]
        assert first_call.kwargs["key"] == b"FPL"

    @patch("src.ingestion.weather_producer.Producer")
    @patch("src.ingestion.weather_producer.fetch_weather")
    def test_message_value_is_valid_json(self, mock_fetch, MockProducer, small_weather_df):
        """Each message value is valid JSON with region and timestamp."""
        mock_fetch.return_value = small_weather_df
        producer_instance = MagicMock()
        MockProducer.return_value = producer_instance

        from src.config import KafkaConfig
        from src.ingestion.weather_producer import produce_weather

        with patch("src.ingestion.weather_producer.GRID_REGIONS", ["FPL"]):
            produce_weather(KafkaConfig())

        first_call = producer_instance.produce.call_args_list[0]
        value = json.loads(first_call.kwargs["value"].decode("utf-8"))
        assert "region" in value
        assert "timestamp" in value
        assert value["region"] == "FPL"


class TestGridProducer:

    @patch("src.ingestion.grid_producer.Producer")
    @patch("src.ingestion.grid_producer.fetch_demand")
    def test_produces_to_correct_topic(self, mock_fetch, MockProducer, small_demand_df):
        """Demand messages go to the 'grid-demand-raw' topic."""
        mock_fetch.return_value = small_demand_df
        producer_instance = MagicMock()
        MockProducer.return_value = producer_instance

        from src.config import KafkaConfig
        from src.ingestion.grid_producer import produce_grid_demand

        with patch("src.ingestion.grid_producer.GRID_REGIONS", ["FPL"]):
            produce_grid_demand(KafkaConfig())

        assert producer_instance.produce.called
        first_call = producer_instance.produce.call_args_list[0]
        assert first_call.kwargs["topic"] == "grid-demand-raw"

    @patch("src.ingestion.grid_producer.Producer")
    @patch("src.ingestion.grid_producer.fetch_demand")
    def test_message_has_demand_fields(self, mock_fetch, MockProducer, small_demand_df):
        """Demand message contains demand_mw and forecast_mw."""
        mock_fetch.return_value = small_demand_df
        producer_instance = MagicMock()
        MockProducer.return_value = producer_instance

        from src.config import KafkaConfig
        from src.ingestion.grid_producer import produce_grid_demand

        with patch("src.ingestion.grid_producer.GRID_REGIONS", ["FPL"]):
            produce_grid_demand(KafkaConfig())

        first_call = producer_instance.produce.call_args_list[0]
        value = json.loads(first_call.kwargs["value"].decode("utf-8"))
        assert "demand_mw" in value
        assert "forecast_mw" in value
        assert isinstance(value["demand_mw"], float)
