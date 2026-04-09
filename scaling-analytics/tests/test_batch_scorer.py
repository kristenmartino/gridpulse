"""Tests for the BatchScorer."""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("psycopg2", reason="psycopg2 not installed")


@pytest.fixture
def sample_predictions():
    """Sample predictions DataFrame as produced by _train_and_score_region."""
    timestamps = pd.date_range("2024-01-15 12:00", periods=24, freq="h")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "region": "ERCOT",
            "predicted_demand_mw": np.random.uniform(35000, 50000, 24).round(2),
            "xgboost": np.random.uniform(35000, 50000, 24).round(2),
            "prophet": np.random.uniform(35000, 50000, 24).round(2),
            "arima": np.random.uniform(35000, 50000, 24).round(2),
            "upper_80": np.random.uniform(40000, 55000, 24).round(2),
            "lower_80": np.random.uniform(30000, 45000, 24).round(2),
            "price_usd_mwh": np.random.uniform(40, 80, 24).round(2),
            "reserve_margin_pct": np.random.uniform(10, 30, 24).round(2),
            "scored_at": "2024-01-15T12:00:00+00:00",
        }
    )


@pytest.fixture
def sample_result(sample_predictions):
    """Sample result dict as returned by _train_and_score_region."""
    return {
        "predictions_df": sample_predictions,
        "predictions": {
            "xgboost": sample_predictions["xgboost"].values,
            "ensemble": sample_predictions["predicted_demand_mw"].values,
        },
        "weights": {"xgboost": 1.0},
        "metrics": {"xgboost": {"mape": 3.13, "rmse": 1500, "mae": 1200, "r2": 0.95}},
        "backtests": {},
        "feature_cols": ["temperature_2m", "hour_sin", "demand_lag_1h"],
        "feature_hash": "abc123def456",
        "train_rows": 500,
    }


class TestCacheForecasts:
    def test_writes_all_granularity_keys(self, mock_redis, sample_result):
        """_cache_forecasts writes 15min, 1h, 1d, and latest keys."""
        from src.config import RedisConfig
        from src.processing.batch_scorer import BatchScorer

        config = RedisConfig()
        pipeline = mock_redis.pipeline()

        scorer = object.__new__(BatchScorer)
        scorer.redis_config = config
        scorer._cache_forecasts(
            pipeline,
            "ERCOT",
            sample_result,
            "2024-01-15T12:00:00+00:00",
            config.forecast_ttl_seconds,
            config.key_prefix,
        )
        pipeline.execute()

        assert mock_redis.exists("wattcast:forecast:ERCOT:latest")
        assert mock_redis.exists("wattcast:forecast:ERCOT:15min")
        assert mock_redis.exists("wattcast:forecast:ERCOT:1h")
        assert mock_redis.exists("wattcast:forecast:ERCOT:1d")

    def test_hourly_rollup_has_fewer_entries(self, mock_redis, sample_result):
        """Hourly rollup should have fewer entries than 15-min."""
        from src.config import RedisConfig
        from src.processing.batch_scorer import BatchScorer

        config = RedisConfig()
        pipeline = mock_redis.pipeline()

        scorer = object.__new__(BatchScorer)
        scorer.redis_config = config
        scorer._cache_forecasts(
            pipeline,
            "ERCOT",
            sample_result,
            "2024-01-15T12:00:00+00:00",
            config.forecast_ttl_seconds,
            config.key_prefix,
        )
        pipeline.execute()

        full = json.loads(mock_redis.get("wattcast:forecast:ERCOT:15min"))
        hourly = json.loads(mock_redis.get("wattcast:forecast:ERCOT:1h"))
        assert len(hourly["forecasts"]) <= len(full["forecasts"])

    def test_payload_has_required_fields(self, mock_redis, sample_result):
        """Each cached payload has region, scored_at, granularity, forecasts."""
        from src.config import RedisConfig
        from src.processing.batch_scorer import BatchScorer

        config = RedisConfig()
        pipeline = mock_redis.pipeline()

        scorer = object.__new__(BatchScorer)
        scorer.redis_config = config
        scorer._cache_forecasts(
            pipeline,
            "ERCOT",
            sample_result,
            "2024-01-15T12:00:00+00:00",
            config.forecast_ttl_seconds,
            config.key_prefix,
        )
        pipeline.execute()

        payload = json.loads(mock_redis.get("wattcast:forecast:ERCOT:1h"))
        assert payload["region"] == "ERCOT"
        assert "scored_at" in payload
        assert "granularity" in payload
        assert isinstance(payload["forecasts"], list)

    def test_forecast_payload_has_enriched_fields(self, mock_redis, sample_result):
        """Forecasts include per-model predictions, confidence bands, pricing."""
        from src.config import RedisConfig
        from src.processing.batch_scorer import BatchScorer

        config = RedisConfig()
        pipeline = mock_redis.pipeline()

        scorer = object.__new__(BatchScorer)
        scorer.redis_config = config
        scorer._cache_forecasts(
            pipeline,
            "ERCOT",
            sample_result,
            "2024-01-15T12:00:00+00:00",
            config.forecast_ttl_seconds,
            config.key_prefix,
        )
        pipeline.execute()

        payload = json.loads(mock_redis.get("wattcast:forecast:ERCOT:15min"))
        entry = payload["forecasts"][0]
        assert "xgboost" in entry
        assert "upper_80" in entry
        assert "lower_80" in entry
        assert "price_usd_mwh" in entry
        assert "reserve_margin_pct" in entry


class TestCacheWeights:
    def test_writes_weights_key(self, mock_redis, sample_result):
        """_cache_weights writes ensemble weights to Redis."""
        from src.config import RedisConfig
        from src.processing.batch_scorer import BatchScorer

        config = RedisConfig()
        pipeline = mock_redis.pipeline()
        scorer = object.__new__(BatchScorer)
        scorer.redis_config = config
        scorer._cache_weights(
            pipeline,
            "ERCOT",
            sample_result,
            "2024-01-15T12:00:00+00:00",
            config.forecast_ttl_seconds,
            config.key_prefix,
        )
        pipeline.execute()

        assert mock_redis.exists("wattcast:weights:ERCOT")
        data = json.loads(mock_redis.get("wattcast:weights:ERCOT"))
        assert "weights" in data
        assert data["weights"]["xgboost"] == 1.0


class TestCacheBacktests:
    def test_writes_backtest_keys(self, mock_redis, sample_result):
        """_cache_backtests writes backtest results per horizon."""
        from src.config import RedisConfig
        from src.processing.batch_scorer import BatchScorer

        sample_result["backtests"] = {
            24: {"horizon": 24, "metrics": {"xgboost": {"mape": 3.0}}, "actual": [1, 2, 3]},
        }
        config = RedisConfig()
        pipeline = mock_redis.pipeline()
        scorer = object.__new__(BatchScorer)
        scorer.redis_config = config
        scorer._cache_backtests(
            pipeline,
            "ERCOT",
            sample_result,
            config.forecast_ttl_seconds,
            config.key_prefix,
        )
        pipeline.execute()

        assert mock_redis.exists("wattcast:backtest:forecast_exog:ERCOT:24")


class TestBatchScorerStructure:
    def test_no_syntax_error(self):
        """batch_scorer.py parses without syntax error."""
        import ast

        from src.processing import batch_scorer

        source_path = batch_scorer.__file__
        with open(source_path) as f:
            source = f.read()
        tree = ast.parse(source)
        assert tree is not None

    def test_run_accepts_mode_parameter(self):
        """run() function accepts a mode parameter."""
        import inspect

        from src.processing.batch_scorer import run

        sig = inspect.signature(run)
        assert "mode" in sig.parameters
        assert sig.parameters["mode"].default == "inference"

    def test_score_all_regions_rejects_invalid_mode(self, mock_redis):
        """score_all_regions raises ValueError for invalid mode."""
        from src.processing.batch_scorer import BatchScorer

        scorer = object.__new__(BatchScorer)
        scorer.redis_config = type(
            "R", (), {"key_prefix": "wattcast", "forecast_ttl_seconds": 3600}
        )()
        scorer.redis_client = mock_redis
        scorer.model_config = type("M", (), {"fast_mode": False})()

        with pytest.raises(ValueError, match="Unknown mode"):
            scorer.score_all_regions(mode="invalid")
