"""
WattCast v2 — Unified Configuration.

Bridges the v2 pipeline config (scaling-analytics/config.py) with v1
constants (project root config.py) so that all modules in src/ can
import from a single location.

Usage:
    from src.config import RedisConfig, GRID_REGIONS, REGION_COORDINATES
"""

import importlib.util
import os
import sys
from pathlib import Path

# ── Ensure v1 root is on sys.path ──────────────────────
# V1_ROOT env var overrides computed path (required inside Docker where
# the directory structure differs from local dev).
_V1_ROOT = os.getenv("V1_ROOT", str(Path(__file__).resolve().parent.parent.parent))
if _V1_ROOT not in sys.path:
    sys.path.insert(0, _V1_ROOT)

# ── Load v2 config (scaling-analytics/config.py) ───────
# We use spec_from_file_location to avoid name collision with v1's config.py
_v2_config_path = Path(__file__).resolve().parent.parent / "config.py"
_spec = importlib.util.spec_from_file_location("_v2_config", _v2_config_path)
_v2_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2_config)

# Re-export v2 dataclasses and constants
KafkaConfig = _v2_config.KafkaConfig
RedisConfig = _v2_config.RedisConfig
DatabaseConfig = _v2_config.DatabaseConfig
ModelConfig = _v2_config.ModelConfig
GRID_REGIONS = _v2_config.GRID_REGIONS
FORECAST_GRANULARITIES = _v2_config.FORECAST_GRANULARITIES
SCORING_INTERVAL_MINUTES = _v2_config.SCORING_INTERVAL_MINUTES
TRAINING_INTERVAL_HOURS = _v2_config.TRAINING_INTERVAL_HOURS
MODEL_ARTIFACT_DIR = _v2_config.MODEL_ARTIFACT_DIR
MODEL_MAX_AGE_HOURS = _v2_config.MODEL_MAX_AGE_HOURS
MODEL_KEEP_SNAPSHOTS = _v2_config.MODEL_KEEP_SNAPSHOTS

# ── Load v1 config (project root config.py) ────────────
_v1_config_path = Path(_V1_ROOT) / "config.py"
_spec_v1 = importlib.util.spec_from_file_location("_v1_config", _v1_config_path)
_v1_config = importlib.util.module_from_spec(_spec_v1)
_spec_v1.loader.exec_module(_v1_config)

# Re-export v1 constants needed by producers and feature builder
REGION_COORDINATES = _v1_config.REGION_COORDINATES
WEATHER_VARIABLES = _v1_config.WEATHER_VARIABLES
EIA_API_KEY = _v1_config.EIA_API_KEY
EIA_BASE_URL = _v1_config.EIA_BASE_URL
EIA_ENDPOINTS = _v1_config.EIA_ENDPOINTS
CACHE_TTL_SECONDS = _v1_config.CACHE_TTL_SECONDS
OPEN_METEO_BASE_URL = _v1_config.OPEN_METEO_BASE_URL

__all__ = [
    # v2
    "KafkaConfig",
    "RedisConfig",
    "DatabaseConfig",
    "ModelConfig",
    "GRID_REGIONS",
    "FORECAST_GRANULARITIES",
    "SCORING_INTERVAL_MINUTES",
    "TRAINING_INTERVAL_HOURS",
    "MODEL_ARTIFACT_DIR",
    "MODEL_MAX_AGE_HOURS",
    "MODEL_KEEP_SNAPSHOTS",
    # v1
    "REGION_COORDINATES",
    "WEATHER_VARIABLES",
    "EIA_API_KEY",
    "EIA_BASE_URL",
    "EIA_ENDPOINTS",
    "CACHE_TTL_SECONDS",
    "OPEN_METEO_BASE_URL",
]
