"""
WattCast v2 — Centralized Configuration
All environment-driven settings in one place.
"""
import os
from dataclasses import dataclass, field


# ─── Pipeline Cadence ────────────────────────────────
# The pipeline runs every 30 minutes because that matches the upstream
# data cadence — EIA publishes hourly demand, Open-Meteo updates hourly.
# Scoring more frequently re-processes identical inputs with no new signal.
#
# Kafka is in the architecture because it supports tightening to 5-minute
# cycles without a rewrite — if we integrate a real-time data source
# (SCADA feed, sub-hourly ERCOT SCED pricing, streaming weather nowcast),
# only the Airflow schedule_interval changes. Below 1 minute, swap Airflow
# for Kafka Streams or Flink; ingestion and serving layers stay identical.
SCORING_INTERVAL_MINUTES = int(os.getenv("SCORING_INTERVAL_MINUTES", "60"))

# ─── Training Cadence ──────────────────────────────
# Models retrain daily — adding 24 new hours to a 365-day training window
# produces negligible weight changes. v1's MODEL_REFRESH_INTERVAL = 86400.
TRAINING_INTERVAL_HOURS = int(os.getenv("TRAINING_INTERVAL_HOURS", "24"))

# ─── Model Persistence ─────────────────────────────
# Trained models are persisted to disk via joblib and loaded for inference.
# This decouples training (daily, expensive) from scoring (hourly, cheap).
MODEL_ARTIFACT_DIR = os.getenv("MODEL_ARTIFACT_DIR", "models/artifacts")
MODEL_MAX_AGE_HOURS = int(os.getenv("MODEL_MAX_AGE_HOURS", "48"))
MODEL_KEEP_SNAPSHOTS = int(os.getenv("MODEL_KEEP_SNAPSHOTS", "3"))


@dataclass(frozen=True)
class KafkaConfig:
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    weather_topic: str = "weather-raw"
    grid_topic: str = "grid-demand-raw"
    consumer_group: str = "wattcast-pipeline"


@dataclass(frozen=True)
class RedisConfig:
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    # TTL must exceed scoring interval so data survives one failed run.
    # Default: 1 hour (2x the 30-min scoring cycle). If you tighten to
    # 5-min cycles, set this to 600 (10 min = 2x the interval).
    forecast_ttl_seconds: int = int(os.getenv(
        "REDIS_FORECAST_TTL",
        str(max(3600, SCORING_INTERVAL_MINUTES * 60 * 2))
    ))
    key_prefix: str = "wattcast"


@dataclass(frozen=True)
class DatabaseConfig:
    url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://wattcast:wattcast_dev@localhost:5432/wattcast"
    )


@dataclass(frozen=True)
class ModelConfig:
    artifact_path: str = os.getenv("MODEL_PATH", "models/wattcast_xgb_latest.joblib")
    n_features: int = 43
    target_col: str = "demand_mw"
    # At 5-min cycles, skip Prophet/ARIMA (too slow) and score XGBoost only.
    # At 30-min cycles, all three models have time to score.
    fast_mode: bool = SCORING_INTERVAL_MINUTES < 15


# ─── Grid Regions ──────────────────────────────────
# The 8 U.S. grid regions WattCast covers
GRID_REGIONS = [
    "PJM",       # Mid-Atlantic
    "ERCOT",     # Texas
    "CAISO",     # California
    "MISO",      # Midwest
    "SPP",       # Southwest Power Pool
    "NYISO",     # New York
    "ISONE",     # New England
    "FPL",       # Florida (FPL/NextEra)
]

# ─── Time Granularities ───────────────────────────
# Pre-compute forecasts at these intervals
FORECAST_GRANULARITIES = {
    "15min": 96,   # 96 intervals per day
    "1h": 24,      # 24 intervals per day
    "1d": 1,       # daily aggregate
}

# Total pre-computed values per scoring run:
# 8 regions × 96 intervals × 24h horizon = 768 forecasts (tiny — fits in Redis trivially)
