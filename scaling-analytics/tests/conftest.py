"""
Shared test fixtures for WattCast v2 tests.

Uses fakeredis for Redis mocking and in-memory data for Postgres mocking.
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import pytest

# Ensure v1 root is on sys.path FIRST (before scaling-analytics)
# so that bare `from config import ...` in v1 code resolves correctly
_V1_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _V1_ROOT not in sys.path:
    sys.path.insert(0, _V1_ROOT)

# Ensure scaling-analytics root is also on sys.path (for `from src.config import ...`)
_SA_ROOT = str(Path(__file__).resolve().parent.parent)
if _SA_ROOT not in sys.path:
    sys.path.insert(1, _SA_ROOT)

# Disable v1 precomputation during tests
os.environ["PRECOMPUTE_ENABLED"] = "false"


@pytest.fixture
def mock_redis():
    """Fake Redis instance for testing (no real Redis needed)."""
    import fakeredis
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def sample_forecast():
    """Sample forecast payload as stored in Redis."""
    return {
        "region": "ERCOT",
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "granularity": "1h",
        "forecasts": [
            {
                "timestamp": "2025-01-15T13:00:00",
                "predicted_demand_mw": 45000.0,
                "xgboost": 44800.0,
                "prophet": 45200.0,
                "arima": 45100.0,
                "upper_80": 46000.0,
                "lower_80": 44000.0,
                "price_usd_mwh": 52.30,
                "reserve_margin_pct": 18.5,
                "region": "ERCOT",
                "scored_at": "2025-01-15T12:00:00+00:00",
            },
            {
                "timestamp": "2025-01-15T14:00:00",
                "predicted_demand_mw": 46500.0,
                "xgboost": 46300.0,
                "prophet": 46700.0,
                "arima": 46400.0,
                "upper_80": 47500.0,
                "lower_80": 45500.0,
                "price_usd_mwh": 55.10,
                "reserve_margin_pct": 16.2,
                "region": "ERCOT",
                "scored_at": "2025-01-15T12:00:00+00:00",
            },
        ],
    }


@pytest.fixture
def sample_metadata():
    """Sample pipeline metadata as stored in Redis."""
    return {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "regions_scored": 8,
        "total_predictions": 768,
        "scoring_mode": "full (3-model ensemble)",
        "scoring_interval_min": 30,
    }


@pytest.fixture
def sample_backtest():
    """Sample backtest payload as stored in Redis."""
    return {
        "horizon": 24,
        "actual": [45000.0, 46000.0, 47000.0],
        "timestamps": ["2025-01-14T00:00:00", "2025-01-14T01:00:00", "2025-01-14T02:00:00"],
        "metrics": {
            "xgboost": {"mape": 3.13, "rmse": 1500, "mae": 1200, "r2": 0.95},
            "ensemble": {"mape": 2.90, "rmse": 1400, "mae": 1100, "r2": 0.96},
        },
        "predictions": {
            "xgboost": [45200.0, 46100.0, 47200.0],
            "ensemble": [45150.0, 46050.0, 47100.0],
        },
        "residuals": [-200.0, -100.0, 200.0],
        "error_by_hour": [
            {"hour": 0, "mean_abs_error": 200.0},
            {"hour": 1, "mean_abs_error": 100.0},
            {"hour": 2, "mean_abs_error": 200.0},
        ],
    }


@pytest.fixture
def sample_actuals():
    """Sample actuals payload as stored in Redis."""
    return {
        "region": "ERCOT",
        "timestamps": ["2025-01-14T00:00:00", "2025-01-14T01:00:00", "2025-01-14T02:00:00"],
        "demand_mw": [45000.0, 46000.0, 47000.0],
        "forecast_mw": [45100.0, 46100.0, 47100.0],
    }


@pytest.fixture
def sample_weather():
    """Sample weather payload as stored in Redis."""
    return {
        "region": "ERCOT",
        "timestamps": ["2025-01-14T00:00:00", "2025-01-14T01:00:00", "2025-01-14T02:00:00"],
        "temperature_2m": [72.0, 73.0, 74.0],
        "relative_humidity_2m": [55.0, 54.0, 53.0],
        "wind_speed_80m": [12.0, 13.0, 11.0],
        "cloud_cover": [30.0, 35.0, 25.0],
        "shortwave_radiation": [500.0, 550.0, 600.0],
        "surface_pressure": [1013.0, 1012.0, 1011.0],
    }


@pytest.fixture
def sample_weights():
    """Sample ensemble weights payload as stored in Redis."""
    return {
        "weights": {"xgboost": 0.50, "prophet": 0.30, "arima": 0.20},
        "metrics": {
            "xgboost": {"mape": 3.13, "rmse": 1500, "mae": 1200, "r2": 0.95},
            "prophet": {"mape": 4.20, "rmse": 1800, "mae": 1400, "r2": 0.92},
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_generation():
    """Sample generation mix payload as stored in Redis."""
    return {
        "region": "ERCOT",
        "timestamps": ["2025-01-14T00:00:00", "2025-01-14T01:00:00"],
        "nuclear": [5000.0, 5000.0],
        "gas": [20000.0, 21000.0],
        "wind": [8000.0, 7500.0],
        "solar": [3000.0, 3500.0],
        "coal": [2000.0, 2000.0],
        "hydro": [500.0, 500.0],
        "renewable_pct": [29.7, 29.5],
    }


@pytest.fixture
def sample_alerts():
    """Sample alerts payload as stored in Redis."""
    return {
        "region": "ERCOT",
        "alerts": [
            {"severity": "warning", "type": "demand_spike", "message": "Demand approaching peak."},
            {"severity": "critical", "type": "anomaly", "message": "Unusual demand pattern detected."},
        ],
        "stress_score": 45,
        "stress_label": "Elevated",
    }


@pytest.fixture
def sample_scenario_preset():
    """Sample scenario preset payload as stored in Redis."""
    return {
        "preset": {"key": "winter_storm_uri", "name": "Winter Storm Uri"},
        "region": "ERCOT",
        "baseline": [45000.0, 46000.0, 47000.0],
        "scenario": [52000.0, 53000.0, 55000.0],
        "delta_mw": [7000.0, 7000.0, 8000.0],
        "delta_pct": 15.94,
        "price_impact": {"base_avg": 52.0, "scenario_avg": 120.0, "delta": 68.0},
        "reserve_margin": {"min_pct": 3.2, "avg_pct": 8.1, "status": "CRITICAL"},
        "renewable_impact": {"wind_power_pct": 5.0, "solar_cf_pct": 10.0},
    }


@pytest.fixture
def sample_news():
    """Sample news payload as stored in Redis."""
    return {
        "articles": [
            {"title": "Energy Markets Update", "source": "Reuters",
             "published_at": datetime.now(timezone.utc).isoformat(),
             "description": "Latest developments in energy markets."},
        ],
    }


@pytest.fixture
def populated_redis(
    mock_redis, sample_forecast, sample_metadata, sample_backtest,
    sample_actuals, sample_weather, sample_weights, sample_generation,
    sample_alerts, sample_scenario_preset, sample_news,
):
    """Redis pre-loaded with all data types for testing."""
    prefix = "wattcast"
    ttl = 3600

    # Forecasts
    mock_redis.setex(
        f"{prefix}:forecast:ERCOT:1h", ttl, json.dumps(sample_forecast),
    )
    mock_redis.setex(
        f"{prefix}:forecast:ERCOT:15min", ttl,
        json.dumps({**sample_forecast, "granularity": "15min"}),
    )

    # Metadata
    mock_redis.set(f"{prefix}:meta:last_scored", json.dumps(sample_metadata))

    # Backtests
    mock_redis.setex(
        f"{prefix}:backtest:ERCOT:24", ttl, json.dumps(sample_backtest),
    )

    # Actuals
    mock_redis.setex(
        f"{prefix}:actuals:ERCOT", ttl, json.dumps(sample_actuals),
    )

    # Weather
    mock_redis.setex(
        f"{prefix}:weather:ERCOT", ttl, json.dumps(sample_weather),
    )

    # Weights
    mock_redis.setex(
        f"{prefix}:weights:ERCOT", ttl, json.dumps(sample_weights),
    )

    # Generation
    mock_redis.setex(
        f"{prefix}:generation:ERCOT", ttl, json.dumps(sample_generation),
    )

    # Alerts
    mock_redis.setex(
        f"{prefix}:alerts:ERCOT", ttl, json.dumps(sample_alerts),
    )

    # Scenario preset
    mock_redis.setex(
        f"{prefix}:scenario:ERCOT:winter_storm_uri", ttl,
        json.dumps(sample_scenario_preset),
    )

    # News
    mock_redis.setex(f"{prefix}:news", ttl, json.dumps(sample_news))

    return mock_redis
