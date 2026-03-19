"""Tests for v2 configuration and v1/v2 harmonization."""
import pytest


def test_grid_regions_is_list():
    """GRID_REGIONS should be a list of strings."""
    from src.config import GRID_REGIONS
    assert isinstance(GRID_REGIONS, list)
    assert all(isinstance(r, str) for r in GRID_REGIONS)


def test_grid_regions_has_eight_entries():
    """WattCast covers 8 balancing authorities."""
    from src.config import GRID_REGIONS
    assert len(GRID_REGIONS) == 8


def test_fpl_in_grid_regions():
    """FPL must be in GRID_REGIONS (not SOCO) for v1 compatibility."""
    from src.config import GRID_REGIONS
    assert "FPL" in GRID_REGIONS


def test_grid_regions_match_region_coordinates():
    """Every region in GRID_REGIONS must have coordinates in v1 config."""
    from src.config import GRID_REGIONS, REGION_COORDINATES
    for region in GRID_REGIONS:
        assert region in REGION_COORDINATES, (
            f"{region} is in GRID_REGIONS but not in REGION_COORDINATES"
        )


def test_redis_config_defaults():
    """RedisConfig defaults should be sensible."""
    from src.config import RedisConfig
    config = RedisConfig()
    assert config.host == "localhost"
    assert config.port == 6379
    assert config.forecast_ttl_seconds >= 3600


def test_kafka_config_defaults():
    """KafkaConfig defaults should be sensible."""
    from src.config import KafkaConfig
    config = KafkaConfig()
    assert config.weather_topic == "weather-raw"
    assert config.grid_topic == "grid-demand-raw"


def test_database_config_defaults():
    """DatabaseConfig should have a default PostgreSQL URL."""
    from src.config import DatabaseConfig
    config = DatabaseConfig()
    assert "postgresql" in config.url
    assert "wattcast" in config.url


def test_model_config_defaults():
    """ModelConfig should default to 43 features."""
    from src.config import ModelConfig
    config = ModelConfig()
    assert config.n_features == 43
    assert config.target_col == "demand_mw"


def test_forecast_granularities():
    """Three granularities must be defined."""
    from src.config import FORECAST_GRANULARITIES
    assert "15min" in FORECAST_GRANULARITIES
    assert "1h" in FORECAST_GRANULARITIES
    assert "1d" in FORECAST_GRANULARITIES


def test_scoring_interval_positive():
    """Scoring interval must be a positive integer."""
    from src.config import SCORING_INTERVAL_MINUTES
    assert isinstance(SCORING_INTERVAL_MINUTES, int)
    assert SCORING_INTERVAL_MINUTES > 0
