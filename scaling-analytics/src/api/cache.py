"""
Redis read helper for the FastAPI serving layer.


CRITICAL: This module NEVER imports from models/ or data/.
It reads pre-computed JSON from Redis. That's it.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import redis

from src.config import GRID_REGIONS, SCORING_INTERVAL_MINUTES, RedisConfig

logger = logging.getLogger(__name__)


class ForecastCache:
    """Read-only Redis cache interface for the serving layer."""

    def __init__(self, config: RedisConfig | None = None):
        self.config = config or RedisConfig()
        self.client = redis.Redis(
            host=self.config.host,
            port=self.config.port,
            decode_responses=True,
        )
        self._prefix = self.config.key_prefix

    # ── Helpers ────────────────────────────────────────

    def _get_json(self, key: str) -> dict | None:
        """Read a single key from Redis and parse JSON."""
        raw = self.client.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.error("Corrupt cache entry: %s", key)
            return None

    # ── Forecasts ──────────────────────────────────────

    def get_forecast(self, region: str, granularity: str = "15min") -> dict | None:
        """
        Read a single region's forecast from Redis.

        Returns parsed JSON dict or None if not found.
        """
        return self._get_json(f"{self._prefix}:forecast:{region}:{granularity}")

    def get_all_regions(self, granularity: str = "1h") -> list[dict]:
        """
        Read forecasts for all regions in a single Redis pipeline.

        Returns list of parsed dicts (empty list if none available).
        """
        pipe = self.client.pipeline()
        for region in GRID_REGIONS:
            pipe.get(f"{self._prefix}:forecast:{region}:{granularity}")
        results = pipe.execute()

        forecasts = []
        for raw in results:
            if raw is not None:
                try:
                    forecasts.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
        return forecasts

    # ── Backtests ──────────────────────────────────────

    def get_backtest(self, region: str, horizon: int, model: str | None = None) -> dict | None:
        """
        Read pre-computed backtest results for a region and horizon.

        If model is specified, filters metrics/predictions to that model.
        """
        data = self._get_json(f"{self._prefix}:backtest:forecast_exog:{region}:{horizon}")
        if data is None:
            data = self._get_json(f"{self._prefix}:backtest:{region}:{horizon}")
        if data is None or model is None:
            return data
        # Filter to a specific model
        filtered = {
            "horizon": data.get("horizon"),
            "actual": data.get("actual"),
            "timestamps": data.get("timestamps"),
            "metrics": {model: data.get("metrics", {}).get(model, {})},
            "predictions": {model: data.get("predictions", {}).get(model, [])},
            "residuals": data.get("residuals"),
            "error_by_hour": data.get("error_by_hour"),
        }
        return filtered

    def get_backtest_residuals(self, region: str, horizon: int) -> dict | None:
        """Return just the residual array from a backtest."""
        data = self._get_json(f"{self._prefix}:backtest:forecast_exog:{region}:{horizon}")
        if data is None:
            data = self._get_json(f"{self._prefix}:backtest:{region}:{horizon}")
        if data is None:
            return None
        return {
            "region": region,
            "horizon": horizon,
            "residuals": data.get("residuals", []),
            "timestamps": data.get("timestamps", []),
        }

    def get_backtest_error_by_hour(self, region: str, horizon: int) -> dict | None:
        """Return error-by-hour breakdown from a backtest."""
        data = self._get_json(f"{self._prefix}:backtest:forecast_exog:{region}:{horizon}")
        if data is None:
            data = self._get_json(f"{self._prefix}:backtest:{region}:{horizon}")
        if data is None:
            return None
        return {
            "region": region,
            "horizon": horizon,
            "error_by_hour": data.get("error_by_hour", []),
        }

    # ── Actuals ────────────────────────────────────────

    def get_actuals(self, region: str) -> dict | None:
        """Read historical demand actuals for a region."""
        return self._get_json(f"{self._prefix}:actuals:{region}")

    # ── Weather ────────────────────────────────────────

    def get_weather(self, region: str) -> dict | None:
        """Read latest weather data for a region."""
        return self._get_json(f"{self._prefix}:weather:{region}")

    # ── Ensemble weights ───────────────────────────────

    def get_ensemble_weights(self, region: str) -> dict | None:
        """Read ensemble weights and per-model metrics for a region."""
        return self._get_json(f"{self._prefix}:weights:{region}")

    # ── Scenarios ──────────────────────────────────────

    def get_scenario_preset(self, region: str, preset_key: str) -> dict | None:
        """Read a single pre-computed scenario preset."""
        return self._get_json(f"{self._prefix}:scenario:{region}:{preset_key}")

    def get_all_scenario_presets(self, region: str) -> list[dict]:
        """Read all pre-computed scenario presets for a region."""
        # We need the preset keys — scan for matching keys
        pattern = f"{self._prefix}:scenario:{region}:*"
        presets = []
        for key in self.client.scan_iter(match=pattern, count=100):
            raw = self.client.get(key)
            if raw is not None:
                try:
                    presets.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
        return presets

    # ── Generation ─────────────────────────────────────

    def get_generation(self, region: str) -> dict | None:
        """Read generation mix data for a region."""
        return self._get_json(f"{self._prefix}:generation:{region}")

    # ── Alerts ─────────────────────────────────────────

    def get_alerts(self, region: str) -> dict | None:
        """Read alerts and stress score for a region."""
        return self._get_json(f"{self._prefix}:alerts:{region}")

    # ── News ───────────────────────────────────────────

    def get_news(self) -> dict | None:
        """Read the global energy news feed."""
        return self._get_json(f"{self._prefix}:news")

    # ── Pipeline metadata ──────────────────────────────

    def get_pipeline_metadata(self) -> dict | None:
        """Read the last scoring run's metadata from Redis."""
        return self._get_json(f"{self._prefix}:meta:last_scored")

    def is_healthy(self) -> bool:
        """
        Check if the pipeline has run recently enough.

        'Recently enough' = within 2x the scoring interval.
        """
        meta = self.get_pipeline_metadata()
        if meta is None:
            return False
        try:
            scored_at = datetime.fromisoformat(meta["scored_at"])
            age_seconds = (datetime.now(UTC) - scored_at).total_seconds()
            max_age = SCORING_INTERVAL_MINUTES * 60 * 2
            return age_seconds < max_age
        except (KeyError, ValueError):
            return False

    # ── Lifecycle ──────────────────────────────────────

    def close(self):
        """Close the Redis connection."""
        self.client.close()
