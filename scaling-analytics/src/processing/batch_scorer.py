"""
Batch Scorer — The Heart of the Pre-Computation Architecture.

Supports three operating modes:
    mode="train"     — Train models, persist to disk, score, cache. (daily DAG)
    mode="inference"  — Load persisted models, score, cache. (hourly DAG)
    mode="backtest"   — Load persisted models, run backtests, cache. (daily DAG)

After this runs, the entire dashboard is served from Redis reads.
Zero computation at request time.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import psycopg2
import redis
from psycopg2.extras import execute_values

from data.demo_data import (
    generate_demo_alerts,
    generate_demo_generation,
)
from data.feature_engineering import get_feature_names
from models.ensemble import compute_ensemble_weights, ensemble_combine
from models.evaluation import (
    compute_all_metrics,
    compute_error_by_hour,
    compute_residuals,
)
from models.pricing import compute_reserve_margin, estimate_price_impact

# v1 model imports (available via sys.path set in src/__init__.py)
from models.xgboost_model import predict_xgboost, train_xgboost
from simulation.presets import get_preset, list_presets
from src.config import (
    GRID_REGIONS,
    MODEL_ARTIFACT_DIR,
    MODEL_KEEP_SNAPSHOTS,
    MODEL_MAX_AGE_HOURS,
    SCORING_INTERVAL_MINUTES,
    DatabaseConfig,
    ModelConfig,
    RedisConfig,
)
from src.processing.feature_builder import FeatureBuilder
from src.processing.model_store import ModelStore

logger = logging.getLogger(__name__)

# Horizons for which backtests are pre-computed
BACKTEST_HORIZONS = [24, 168, 720]
DEFAULT_BACKTEST_EXOG_MODE = "forecast_exog"


class BatchScorer:
    """
    Scores all regions and writes pre-computed forecasts to Redis + Postgres.

    After this runs, the API serves pre-computed results ONLY.
    """

    def __init__(
        self,
        redis_config: RedisConfig | None = None,
        db_config: DatabaseConfig | None = None,
        model_config: ModelConfig | None = None,
    ):
        self.redis_config = redis_config or RedisConfig()
        self.db_config = db_config or DatabaseConfig()
        self.model_config = model_config or ModelConfig()

        self.redis_client = redis.Redis(
            host=self.redis_config.host,
            port=self.redis_config.port,
            decode_responses=True,
        )
        self.conn = psycopg2.connect(self.db_config.url)
        self.feature_builder = FeatureBuilder(self.db_config)
        self.model_store = ModelStore(MODEL_ARTIFACT_DIR)

        logger.info(
            "BatchScorer initialized (fast_mode=%s, artifact_dir=%s)",
            self.model_config.fast_mode,
            MODEL_ARTIFACT_DIR,
        )

    def score_all_regions(self, mode: str = "inference"):
        """
        Main entry point.

        Args:
            mode: Operating mode.
                "train"     — Train models, persist, score, cache.
                "inference" — Load persisted models, score, cache.
                "backtest"  — Load models, run backtests, cache.
        """
        if mode not in ("train", "inference", "backtest"):
            raise ValueError(f"Unknown mode: {mode}. Use 'train', 'inference', or 'backtest'.")

        scored_at = datetime.now(UTC).isoformat()
        total_predictions = 0
        regions_scored = 0
        prefix = self.redis_config.key_prefix
        ttl = self.redis_config.forecast_ttl_seconds

        # Use staging prefix for atomic swap
        staging_prefix = f"{prefix}:staging"
        pipeline = self.redis_client.pipeline()

        for region in GRID_REGIONS:
            try:
                if mode == "backtest":
                    self._run_backtest_for_region(pipeline, region, ttl, staging_prefix)
                    regions_scored += 1
                    continue

                result = self._train_and_score_region(region, scored_at, mode=mode)
                if result is None:
                    continue

                # Write forecasts to staging keys (3 granularities)
                self._cache_forecasts(pipeline, region, result, scored_at, ttl, staging_prefix)

                # Write actuals + weather to staging
                self._cache_actuals(pipeline, region, ttl, staging_prefix)
                self._cache_weather(pipeline, region, ttl, staging_prefix)

                # Write ensemble weights
                self._cache_weights(pipeline, region, result, scored_at, ttl, staging_prefix)

                # Write generation mix (demo data for now)
                self._cache_generation(pipeline, region, ttl, staging_prefix)

                # Write alerts + stress score
                self._cache_alerts(pipeline, region, ttl, staging_prefix)

                # Write scenario presets
                self._cache_scenario_presets(pipeline, region, result, ttl, staging_prefix)

                if mode == "train":
                    # Write backtests during training mode
                    self._cache_backtests(pipeline, region, result, ttl, staging_prefix)

                # Write to Postgres
                self._store_forecasts(region, result["predictions_df"], scored_at)
                self._store_audit(region, result, scored_at)

                total_predictions += len(result["predictions_df"])
                regions_scored += 1
                logger.info(
                    "Scored %s (%s): %d predictions", region, mode, len(result["predictions_df"])
                )

            except Exception as e:
                logger.error("Failed to score %s: %s", region, e, exc_info=True)

        # Cache news (global, not per-region)
        if mode != "backtest":
            self._cache_news(pipeline, ttl, staging_prefix)

        # Execute staging writes
        pipeline.execute()

        # Atomic swap: rename staging keys to live keys
        self._atomic_swap(staging_prefix, prefix, ttl)

        # Update metadata on live prefix
        meta_payload = {
            "scored_at": scored_at,
            "regions_scored": regions_scored,
            "total_predictions": total_predictions,
            "scoring_mode": "fast (XGBoost only)"
            if self.model_config.fast_mode
            else "full (3-model ensemble)",
            "scoring_interval_min": SCORING_INTERVAL_MINUTES,
            "mode": mode,
        }

        # Add models_trained_at if available
        for region in GRID_REGIONS:
            age = self.model_store.model_age_hours(region)
            if age is not None:
                loaded = self.model_store.load_models(region)
                if loaded:
                    meta_payload["models_trained_at"] = loaded.get("trained_at", "")
                break

        self.redis_client.set(f"{prefix}:meta:last_scored", json.dumps(meta_payload))

        logger.info(
            "Batch scoring complete (%s): %d predictions across %d regions",
            mode,
            total_predictions,
            regions_scored,
        )

    def _atomic_swap(self, staging_prefix: str, live_prefix: str, ttl: int):
        """
        Atomically swap staging keys to live keys.

        Scans for all staging keys and renames them to the live prefix.
        If swap fails, extends existing live key TTLs to prevent 503.
        """
        try:
            # Find all staging keys
            staging_keys = []
            cursor = 0
            while True:
                cursor, keys = self.redis_client.scan(
                    cursor=cursor, match=f"{staging_prefix}:*", count=100
                )
                staging_keys.extend(keys)
                if cursor == 0:
                    break

            if not staging_keys:
                return

            # Rename staging -> live in a pipeline
            rename_pipe = self.redis_client.pipeline()
            for staging_key in staging_keys:
                live_key = staging_key.replace(staging_prefix, live_prefix, 1)
                rename_pipe.rename(staging_key, live_key)
                rename_pipe.expire(live_key, ttl)
            rename_pipe.execute()

            logger.info("Atomic swap: renamed %d keys", len(staging_keys))

        except Exception as e:
            logger.error("Atomic swap failed: %s — extending existing TTLs", e)
            self._extend_live_ttls(live_prefix, ttl)

    def _extend_live_ttls(self, prefix: str, ttl: int):
        """Extend TTL on all existing live keys to prevent 503."""
        try:
            cursor = 0
            extended = 0
            while True:
                cursor, keys = self.redis_client.scan(cursor=cursor, match=f"{prefix}:*", count=100)
                pipe = self.redis_client.pipeline()
                for key in keys:
                    if ":staging:" not in key and ":meta:" not in key:
                        pipe.expire(key, ttl)
                        extended += 1
                pipe.execute()
                if cursor == 0:
                    break
            logger.info("Extended TTL on %d existing keys", extended)
        except Exception as e:
            logger.error("Failed to extend TTLs: %s", e)

    # ── Core scoring ───────────────────────────────────

    def _train_and_score_region(
        self,
        region: str,
        scored_at: str,
        mode: str = "train",
    ) -> dict | None:
        """
        Build features and score for one region.

        In train mode: trains models from scratch and persists to disk.
        In inference mode: loads persisted models and predicts only.
        """
        features_df = self.feature_builder.build_features(region, horizon_hours=24)

        if features_df.empty:
            logger.warning("No features available for %s", region)
            return None

        feature_cols = get_feature_names()
        available_cols = [c for c in feature_cols if c in features_df.columns]
        if not available_cols:
            logger.warning("No feature columns available for %s", region)
            return None

        target_col = self.model_config.target_col
        has_target = target_col in features_df.columns

        # Split into training data (rows with target) and future (rows without)
        if has_target:
            train_df = features_df[features_df[target_col].notna()].copy()
            future_df = features_df[features_df[target_col].isna()].copy()
            if future_df.empty:
                future_df = features_df.tail(96).copy()
        else:
            train_df = features_df.copy()
            future_df = features_df.tail(96).copy()

        if train_df.empty or len(train_df) < 48:
            logger.warning("Insufficient training data for %s (%d rows)", region, len(train_df))
            return None

        # ── Get models (train or load) ─────────────
        if mode == "train":
            models, predictions, metrics_by_model, weights = self._train_models(
                region, train_df, future_df, target_col
            )
            if models is None:
                return None
        else:
            # Inference mode: load persisted models
            loaded = self.model_store.load_models(region)
            if loaded is None:
                logger.warning(
                    "No persisted models for %s — falling back to training",
                    region,
                )
                models, predictions, metrics_by_model, weights = self._train_models(
                    region, train_df, future_df, target_col
                )
                if models is None:
                    return None
            else:
                age = self.model_store.model_age_hours(region)
                if age is not None and age > MODEL_MAX_AGE_HOURS:
                    logger.warning(
                        "Models for %s are %.1fh old (max=%dh) — using anyway",
                        region,
                        age,
                        MODEL_MAX_AGE_HOURS,
                    )
                models, predictions, metrics_by_model, weights = self._predict_with_loaded(
                    region, loaded, future_df, train_df, target_col
                )

        ensemble_preds = predictions["ensemble"]

        # ── Confidence bands ──────────────────────────
        if len(predictions) > 2:
            model_arrays = [predictions[k] for k in predictions if k != "ensemble"]
            stacked = np.column_stack(model_arrays)
            spread = np.max(np.abs(stacked - ensemble_preds[:, None]), axis=1)
            upper_80 = np.round(ensemble_preds + spread, 2)
            lower_80 = np.round(np.maximum(ensemble_preds - spread, 0), 2)
        else:
            upper_80 = np.round(ensemble_preds * 1.03, 2)
            lower_80 = np.round(ensemble_preds * 0.97, 2)

        # ── Pricing ───────────────────────────────────
        from config import REGION_CAPACITY_MW

        capacity = REGION_CAPACITY_MW.get(region, 100_000)
        prices = estimate_price_impact(ensemble_preds, capacity)
        reserve_margins = compute_reserve_margin(ensemble_preds, region)

        # ── Build predictions DataFrame ───────────────
        timestamps = (
            future_df.index
            if hasattr(future_df.index, "to_pydatetime")
            else range(len(ensemble_preds))
        )
        predictions_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "region": region,
                "predicted_demand_mw": ensemble_preds,
                "xgboost": predictions.get("xgboost", ensemble_preds),
                "prophet": predictions.get("prophet", ensemble_preds),
                "arima": predictions.get("arima", ensemble_preds),
                "upper_80": upper_80,
                "lower_80": lower_80,
                "price_usd_mwh": np.round(np.atleast_1d(prices), 2),
                "reserve_margin_pct": np.round(np.atleast_1d(reserve_margins), 2),
                "scored_at": scored_at,
            }
        )

        # ── Backtests (only in train mode) ────────────
        backtests = {}
        if mode == "train" and has_target and len(train_df) > 200:
            for horizon in BACKTEST_HORIZONS:
                try:
                    bt = self._compute_backtest(train_df, feature_cols, target_col, horizon, region)
                    if bt is not None:
                        backtests[horizon] = bt
                except Exception as e:
                    logger.warning("Backtest %dh failed for %s: %s", horizon, region, e)

        # Feature hash for audit
        feature_hash = hashlib.md5(str(sorted(available_cols)).encode()).hexdigest()[:12]

        return {
            "predictions_df": predictions_df,
            "predictions": predictions,
            "weights": weights,
            "metrics": metrics_by_model,
            "backtests": backtests,
            "feature_cols": available_cols,
            "feature_hash": feature_hash,
            "train_rows": len(train_df),
            "models_trained_at": self.model_store.load_models(region).get("trained_at", "")
            if self.model_store.has_models(region)
            else None,
        }

    def _train_models(
        self,
        region: str,
        train_df,
        future_df,
        target_col: str,
    ) -> tuple[dict, dict, dict, dict]:
        """Train all models from scratch, persist, and predict."""
        models = {}
        predictions = {}
        metrics_by_model = {}

        # XGBoost (always runs)
        try:
            xgb_result = train_xgboost(train_df, target_col=target_col)
            models["xgboost"] = xgb_result
            xgb_preds = predict_xgboost(xgb_result, future_df)
            predictions["xgboost"] = np.round(xgb_preds, 2)
        except Exception as e:
            logger.error("XGBoost training failed for %s: %s", region, e)
            return None, None, None, None

        # Prophet + ARIMA (only in full mode)
        if not self.model_config.fast_mode:
            try:
                from models.prophet_model import predict_prophet, train_prophet

                prophet_model = train_prophet(train_df, target_col=target_col)
                prophet_result = predict_prophet(prophet_model, future_df, periods=len(future_df))
                models["prophet"] = prophet_model
                predictions["prophet"] = np.round(prophet_result["forecast"], 2)
            except Exception as e:
                logger.warning("Prophet training failed for %s: %s", region, e)

            try:
                from models.arima_model import predict_arima, train_arima

                arima_result = train_arima(train_df, target_col=target_col)
                arima_preds = predict_arima(arima_result, future_df, periods=len(future_df))
                models["arima"] = arima_result
                predictions["arima"] = np.round(arima_preds, 2)
            except Exception as e:
                logger.warning("ARIMA training failed for %s: %s", region, e)

        # Compute ensemble weights
        weights, metrics_by_model = self._compute_weights(models, predictions, train_df, target_col)
        predictions["ensemble"] = np.round(
            ensemble_combine(predictions, weights)
            if len(predictions) > 1
            else predictions["xgboost"],
            2,
        )

        # Persist models to disk
        feature_cols = get_feature_names()
        available_cols = [c for c in feature_cols if c in train_df.columns]
        feature_hash = hashlib.md5(str(sorted(available_cols)).encode()).hexdigest()[:12]
        self.model_store.save_models(
            region, models, weights, metrics_by_model, available_cols, feature_hash
        )
        self.model_store.cleanup_old(region, keep_n=MODEL_KEEP_SNAPSHOTS)

        return models, predictions, metrics_by_model, weights

    def _predict_with_loaded(
        self,
        region: str,
        loaded: dict,
        future_df,
        train_df,
        target_col: str,
    ) -> tuple[dict, dict, dict, dict]:
        """Predict using pre-loaded models (inference mode — no training)."""
        models = loaded["models"]
        weights = loaded["weights"]
        metrics_by_model = loaded.get("metrics", {})
        predictions = {}

        # XGBoost predict
        if "xgboost" in models:
            try:
                xgb_preds = predict_xgboost(models["xgboost"], future_df)
                predictions["xgboost"] = np.round(xgb_preds, 2)
            except Exception as e:
                logger.error("XGBoost inference failed for %s: %s", region, e)
                return None, None, None, None

        # Prophet predict (if available and not fast_mode)
        if "prophet" in models and not self.model_config.fast_mode:
            try:
                from models.prophet_model import predict_prophet

                prophet_result = predict_prophet(
                    models["prophet"], future_df, periods=len(future_df)
                )
                predictions["prophet"] = np.round(prophet_result["forecast"], 2)
            except Exception as e:
                logger.warning("Prophet inference failed for %s: %s", region, e)

        # ARIMA predict (if available and not fast_mode)
        if "arima" in models and not self.model_config.fast_mode:
            try:
                from models.arima_model import predict_arima

                arima_preds = predict_arima(models["arima"], future_df, periods=len(future_df))
                predictions["arima"] = np.round(arima_preds, 2)
            except Exception as e:
                logger.warning("ARIMA inference failed for %s: %s", region, e)

        if not predictions:
            return None, None, None, None

        # Ensemble combine
        if len(predictions) > 1:
            ensemble_preds = ensemble_combine(predictions, weights)
        else:
            ensemble_preds = list(predictions.values())[0]
            weights = {list(predictions.keys())[0]: 1.0}

        predictions["ensemble"] = np.round(ensemble_preds, 2)

        return models, predictions, metrics_by_model, weights

    def _compute_weights(
        self,
        models: dict,
        predictions: dict,
        train_df,
        target_col: str,
    ) -> tuple[dict, dict]:
        """Compute ensemble weights from validation metrics."""
        metrics_by_model = {}

        if len(predictions) > 1:
            val_size = min(168, len(train_df) // 4)
            val_actual = train_df[target_col].values[-val_size:]
            mape_scores = {}

            for model_name, model_data in models.items():
                try:
                    if model_name == "xgboost":
                        val_preds = predict_xgboost(model_data, train_df.tail(val_size))
                    else:
                        continue  # Prophet/ARIMA validation is harder inline
                    m = compute_all_metrics(val_actual, val_preds)
                    mape_scores[model_name] = m["mape"]
                    metrics_by_model[model_name] = m
                except Exception:
                    pass

            for model_name in ["prophet", "arima"]:
                if model_name in predictions and model_name not in mape_scores:
                    mape_scores[model_name] = 5.0 if model_name == "prophet" else 6.0

            weights = compute_ensemble_weights(mape_scores)
        else:
            weights = {"xgboost": 1.0}
            has_target = target_col in train_df.columns
            if has_target:
                val_size = min(168, len(train_df) // 4)
                val_actual = train_df[target_col].values[-val_size:]
                try:
                    val_preds = predict_xgboost(models["xgboost"], train_df.tail(val_size))
                    metrics_by_model["xgboost"] = compute_all_metrics(val_actual, val_preds)
                except Exception:
                    pass

        return weights, metrics_by_model

    def _run_backtest_for_region(self, pipeline, region: str, ttl: int, prefix: str):
        """Run backtests using persisted models for one region."""
        loaded = self.model_store.load_models(region)
        if loaded is None:
            logger.warning("No models for backtesting %s — skipping", region)
            return

        features_df = self.feature_builder.build_features(region, horizon_hours=24)
        if features_df.empty:
            return

        feature_cols = get_feature_names()
        target_col = self.model_config.target_col

        if target_col not in features_df.columns:
            return

        train_df = features_df[features_df[target_col].notna()].copy()
        if len(train_df) < 200:
            return

        for horizon in BACKTEST_HORIZONS:
            try:
                bt = self._compute_backtest(train_df, feature_cols, target_col, horizon, region)
                if bt is not None:
                    pipeline.setex(
                        f"{prefix}:backtest:{DEFAULT_BACKTEST_EXOG_MODE}:{region}:{horizon}",
                        ttl,
                        json.dumps(bt),
                    )
            except Exception as e:
                logger.warning("Backtest %dh failed for %s: %s", horizon, region, e)

    def _compute_backtest(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        horizon: int,
        region: str,
    ) -> dict | None:
        """Run a holdout backtest for one horizon."""
        if len(df) < horizon + 168:
            return None

        train_part = df.iloc[:-horizon]
        test_part = df.iloc[-horizon:]
        actual = test_part[target_col].values

        bt_metrics = {}
        bt_predictions = {}

        # XGBoost backtest
        try:
            xgb_result = train_xgboost(train_part, target_col=target_col)
            xgb_preds = predict_xgboost(xgb_result, test_part)
            bt_predictions["xgboost"] = np.round(xgb_preds, 2).tolist()
            bt_metrics["xgboost"] = compute_all_metrics(actual, xgb_preds)
        except Exception as e:
            logger.warning("Backtest XGBoost failed for %s/%dh: %s", region, horizon, e)
            return None

        # Prophet backtest (full mode only)
        if not self.model_config.fast_mode:
            try:
                from models.prophet_model import predict_prophet, train_prophet

                p_model = train_prophet(train_part, target_col=target_col)
                p_result = predict_prophet(p_model, test_part, periods=horizon)
                p_preds = p_result["forecast"][:horizon]
                bt_predictions["prophet"] = np.round(p_preds, 2).tolist()
                bt_metrics["prophet"] = compute_all_metrics(actual[: len(p_preds)], p_preds)
            except Exception as e:
                logger.warning("Backtest Prophet failed for %s/%dh: %s", region, horizon, e)

            try:
                from models.arima_model import predict_arima, train_arima

                a_result = train_arima(train_part, target_col=target_col)
                a_preds = predict_arima(a_result, test_part, periods=horizon)
                bt_predictions["arima"] = np.round(a_preds, 2).tolist()
                bt_metrics["arima"] = compute_all_metrics(actual[: len(a_preds)], a_preds)
            except Exception as e:
                logger.warning("Backtest ARIMA failed for %s/%dh: %s", region, horizon, e)

        # Ensemble backtest
        if len(bt_predictions) > 1:
            mape_scores = {k: v["mape"] for k, v in bt_metrics.items()}
            w = compute_ensemble_weights(mape_scores)
            forecasts_dict = {k: np.array(v) for k, v in bt_predictions.items()}
            ens = ensemble_combine(forecasts_dict, w)
            bt_predictions["ensemble"] = np.round(ens, 2).tolist()
            bt_metrics["ensemble"] = compute_all_metrics(actual[: len(ens)], ens)
        else:
            bt_predictions["ensemble"] = bt_predictions.get("xgboost", [])
            bt_metrics["ensemble"] = bt_metrics.get("xgboost", {})

        # Residuals and error-by-hour (based on ensemble)
        ens_preds = np.array(bt_predictions["ensemble"])
        actual_trimmed = actual[: len(ens_preds)]
        residuals = compute_residuals(actual_trimmed, ens_preds)
        timestamps = test_part.index[: len(ens_preds)]

        try:
            error_by_hour_df = compute_error_by_hour(timestamps, actual_trimmed, ens_preds)
            error_by_hour = error_by_hour_df.to_dict(orient="records")
        except Exception:
            error_by_hour = []

        return {
            "horizon": horizon,
            "exog_mode": DEFAULT_BACKTEST_EXOG_MODE,
            "exog_source": "climatology/naive baseline",
            "metrics": bt_metrics,
            "actual": actual_trimmed.tolist(),
            "predictions": bt_predictions,
            "residuals": residuals.tolist(),
            "error_by_hour": error_by_hour,
            "timestamps": [str(t) for t in timestamps],
        }

    # ── Redis cache writers ────────────────────────────

    def _cache_forecasts(self, pipeline, region, result, scored_at, ttl, prefix):
        """Write forecasts to Redis at 3 granularities with enriched payload."""
        predictions_df = result["predictions_df"]
        records = predictions_df.to_dict(orient="records")
        for r in records:
            r["timestamp"] = str(r["timestamp"])

        # Include metadata
        models_trained_at = result.get("models_trained_at", "")

        payload = json.dumps(
            {
                "region": region,
                "scored_at": scored_at,
                "models_trained_at": models_trained_at,
                "granularity": "15min",
                "forecasts": records,
            }
        )
        pipeline.setex(f"{prefix}:forecast:{region}:latest", ttl, payload)
        pipeline.setex(f"{prefix}:forecast:{region}:15min", ttl, payload)

        # Hourly rollup
        df_copy = predictions_df.copy()
        df_copy = df_copy.set_index("timestamp")
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        agg_dict = {c: "mean" for c in numeric_cols}
        for c in ["region", "scored_at"]:
            if c in df_copy.columns:
                agg_dict[c] = "first"
        hourly = df_copy.resample("1h").agg(agg_dict).reset_index()
        hourly_records = hourly.to_dict(orient="records")
        for r in hourly_records:
            r["timestamp"] = str(r["timestamp"])

        pipeline.setex(
            f"{prefix}:forecast:{region}:1h",
            ttl,
            json.dumps(
                {
                    "region": region,
                    "scored_at": scored_at,
                    "models_trained_at": models_trained_at,
                    "granularity": "1h",
                    "forecasts": hourly_records,
                }
            ),
        )

        # Daily rollup
        daily = df_copy.resample("1D").agg(agg_dict).reset_index()
        daily_records = daily.to_dict(orient="records")
        for r in daily_records:
            r["timestamp"] = str(r["timestamp"])

        pipeline.setex(
            f"{prefix}:forecast:{region}:1d",
            ttl,
            json.dumps(
                {
                    "region": region,
                    "scored_at": scored_at,
                    "models_trained_at": models_trained_at,
                    "granularity": "1d",
                    "forecasts": daily_records,
                }
            ),
        )

    def _cache_backtests(self, pipeline, region, result, ttl, prefix):
        """Write pre-computed backtest results to Redis."""
        for horizon, bt in result.get("backtests", {}).items():
            pipeline.setex(
                f"{prefix}:backtest:{DEFAULT_BACKTEST_EXOG_MODE}:{region}:{horizon}",
                ttl,
                json.dumps(bt),
            )

    def _cache_weights(self, pipeline, region, result, scored_at, ttl, prefix):
        """Write ensemble weights to Redis."""
        weights_payload = {
            "weights": result["weights"],
            "metrics": {k: v for k, v in result.get("metrics", {}).items()},
            "updated_at": scored_at,
        }
        pipeline.setex(
            f"{prefix}:weights:{region}",
            ttl,
            json.dumps(weights_payload),
        )

    def _cache_actuals(self, pipeline, region, ttl, prefix):
        """Write recent actuals from Postgres to Redis."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT timestamp, demand_mw, forecast_mw
                    FROM raw_demand
                    WHERE region = %s
                    ORDER BY timestamp DESC
                    LIMIT 2160
                    """,
                    (region,),
                )
                rows = cur.fetchall()

            if rows:
                rows.reverse()
                payload = {
                    "region": region,
                    "timestamps": [r[0].isoformat() for r in rows],
                    "demand_mw": [r[1] for r in rows],
                    "forecast_mw": [r[2] for r in rows],
                }
                pipeline.setex(
                    f"{prefix}:actuals:{region}",
                    ttl,
                    json.dumps(payload),
                )
        except Exception as e:
            logger.warning("Failed to cache actuals for %s: %s", region, e)

    def _cache_weather(self, pipeline, region, ttl, prefix):
        """Write recent weather from Postgres to Redis."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT timestamp, temperature_2m, apparent_temperature,
                           relative_humidity_2m, dew_point_2m,
                           wind_speed_10m, wind_speed_80m, wind_speed_120m,
                           wind_direction_10m, shortwave_radiation,
                           direct_normal_irradiance, diffuse_radiation,
                           cloud_cover, precipitation, snowfall,
                           surface_pressure, soil_temperature_0cm
                    FROM raw_weather
                    WHERE region = %s
                    ORDER BY timestamp DESC
                    LIMIT 2160
                    """,
                    (region,),
                )
                rows = cur.fetchall()
                cols = [desc[0] for desc in cur.description]

            if rows:
                rows.reverse()
                payload = {"region": region}
                payload["timestamps"] = [r[0].isoformat() for r in rows]
                for i, col in enumerate(cols[1:], start=1):
                    payload[col] = [r[i] for r in rows]
                pipeline.setex(
                    f"{prefix}:weather:{region}",
                    ttl,
                    json.dumps(payload),
                )
        except Exception as e:
            logger.warning("Failed to cache weather for %s: %s", region, e)

    def _cache_generation(self, pipeline, region, ttl, prefix):
        """Write generation mix to Redis (demo data until real EIA generation endpoint)."""
        try:
            gen_df = generate_demo_generation(region, days=30)
            pivoted = gen_df.pivot_table(
                index="timestamp",
                columns="fuel_type",
                values="generation_mw",
                aggfunc="sum",
            ).reset_index()

            payload = {"region": region, "timestamps": [str(t) for t in pivoted["timestamp"]]}
            for col in pivoted.columns:
                if col != "timestamp":
                    payload[col] = [round(v, 1) if pd.notna(v) else 0 for v in pivoted[col]]

            renewable_types = {"wind", "solar", "hydro"}
            total = np.zeros(len(pivoted))
            renewable = np.zeros(len(pivoted))
            for col in pivoted.columns:
                if col == "timestamp":
                    continue
                vals = np.array([v if pd.notna(v) else 0 for v in pivoted[col]])
                total += vals
                if col in renewable_types:
                    renewable += vals
            with np.errstate(divide="ignore", invalid="ignore"):
                pct = np.where(total > 0, np.round(renewable / total * 100, 1), 0)
            payload["renewable_pct"] = pct.tolist()

            pipeline.setex(f"{prefix}:generation:{region}", ttl, json.dumps(payload))
        except Exception as e:
            logger.warning("Failed to cache generation for %s: %s", region, e)

    def _cache_alerts(self, pipeline, region, ttl, prefix):
        """Write alerts and stress score to Redis."""
        try:
            alerts = generate_demo_alerts(region)
            n_crit = sum(1 for a in alerts if a.get("severity") == "critical")
            n_warn = sum(1 for a in alerts if a.get("severity") == "warning")
            stress_score = min(100, n_crit * 30 + n_warn * 15 + 20)
            if stress_score >= 60:
                stress_label = "Critical"
            elif stress_score >= 30:
                stress_label = "Elevated"
            else:
                stress_label = "Normal"

            payload = {
                "region": region,
                "alerts": alerts,
                "stress_score": stress_score,
                "stress_label": stress_label,
            }
            pipeline.setex(f"{prefix}:alerts:{region}", ttl, json.dumps(payload))
        except Exception as e:
            logger.warning("Failed to cache alerts for %s: %s", region, e)

    def _cache_scenario_presets(self, pipeline, region, result, ttl, prefix):
        """Pre-compute the 6 historical scenario presets."""
        ensemble_preds = result["predictions"]["ensemble"]
        from config import REGION_CAPACITY_MW

        for preset_info in list_presets():
            try:
                preset = get_preset(preset_info["key"])
                weather_overrides = preset.get("weather", {})

                temp = weather_overrides.get("temperature_2m", 75)
                cdd_delta = max(0, temp - 65) - max(0, 75 - 65)
                hdd_delta = max(0, 65 - temp) - max(0, 65 - 75)
                temp_factor = 1 + (cdd_delta * 0.02 + hdd_delta * 0.015) / 65

                wind_speed = weather_overrides.get("wind_speed_80m", 15)
                cloud = weather_overrides.get("cloud_cover", 50)
                solar = weather_overrides.get("shortwave_radiation", 500)

                scenario_demand = np.round(ensemble_preds * temp_factor, 2)
                delta_mw = np.round(scenario_demand - ensemble_preds, 2)

                capacity = REGION_CAPACITY_MW.get(region, 100_000)
                scenario_prices = estimate_price_impact(scenario_demand, capacity)
                base_prices = estimate_price_impact(ensemble_preds, capacity)
                scenario_reserve = compute_reserve_margin(scenario_demand, region)

                wind_power_pct = min(100, max(0, (wind_speed / 15) ** 3 * 30))
                solar_cf_pct = min(100, max(0, solar / 1000 * (1 - cloud / 100) * 100))

                payload = {
                    "preset": preset,
                    "region": region,
                    "baseline": ensemble_preds.tolist(),
                    "scenario": scenario_demand.tolist(),
                    "delta_mw": delta_mw.tolist(),
                    "delta_pct": float(
                        np.round(np.mean(delta_mw) / np.mean(ensemble_preds) * 100, 2)
                    ),
                    "price_impact": {
                        "base_avg": float(np.round(np.mean(np.atleast_1d(base_prices)), 2)),
                        "scenario_avg": float(np.round(np.mean(np.atleast_1d(scenario_prices)), 2)),
                        "delta": float(
                            np.round(
                                np.mean(np.atleast_1d(scenario_prices))
                                - np.mean(np.atleast_1d(base_prices)),
                                2,
                            )
                        ),
                    },
                    "reserve_margin": {
                        "min_pct": float(np.round(np.min(np.atleast_1d(scenario_reserve)), 2)),
                        "avg_pct": float(np.round(np.mean(np.atleast_1d(scenario_reserve)), 2)),
                        "status": "CRITICAL"
                        if np.min(np.atleast_1d(scenario_reserve)) < 5
                        else "Low"
                        if np.min(np.atleast_1d(scenario_reserve)) < 15
                        else "Adequate",
                    },
                    "renewable_impact": {
                        "wind_power_pct": round(wind_power_pct, 1),
                        "solar_cf_pct": round(solar_cf_pct, 1),
                    },
                }
                pipeline.setex(
                    f"{prefix}:scenario:{region}:{preset_info['key']}",
                    ttl,
                    json.dumps(payload),
                )
            except Exception as e:
                logger.warning(
                    "Failed to cache preset %s for %s: %s",
                    preset_info.get("key", "?"),
                    region,
                    e,
                )

    def _cache_news(self, pipeline, ttl, prefix):
        """Cache energy news (demo placeholder)."""
        try:
            from data.news_client import fetch_energy_news

            articles = fetch_energy_news(page_size=5)
        except Exception:
            articles = [
                {
                    "title": "Energy Markets Update",
                    "source": "Reuters",
                    "published_at": datetime.now(UTC).isoformat(),
                    "description": "Latest developments in energy markets.",
                },
            ]
        pipeline.setex(f"{prefix}:news", ttl, json.dumps({"articles": articles}))

    # ── Postgres writes ────────────────────────────────

    def _store_forecasts(self, region: str, predictions: pd.DataFrame, scored_at: str):
        """Write forecasts to Postgres for historical tracking."""
        rows = [
            (region, row["timestamp"], row["predicted_demand_mw"], scored_at)
            for _, row in predictions.iterrows()
        ]
        sql = """
            INSERT INTO forecasts (region, timestamp, predicted_demand_mw, scored_at)
            VALUES %s
            ON CONFLICT (region, timestamp, scored_at)
            DO UPDATE SET predicted_demand_mw = EXCLUDED.predicted_demand_mw
        """
        with self.conn.cursor() as cur:
            execute_values(cur, sql, rows)
        self.conn.commit()

    def _store_audit(self, region: str, result: dict, scored_at: str):
        """Write audit record to Postgres."""
        try:
            sql = """
                INSERT INTO audit_trail
                    (region, scored_at, demand_source, weather_source,
                     demand_rows, weather_rows, model_versions, ensemble_weights,
                     feature_count, feature_hash, mape, peak_forecast_mw, scoring_mode)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            preds = result["predictions_df"]["predicted_demand_mw"]
            with self.conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        region,
                        scored_at,
                        "postgres",
                        "postgres",
                        result["train_rows"],
                        result["train_rows"],
                        json.dumps(list(result["weights"].keys())),
                        json.dumps(result["weights"]),
                        len(result["feature_cols"]),
                        result["feature_hash"],
                        json.dumps(result.get("metrics", {})),
                        float(preds.max()) if len(preds) > 0 else 0,
                        "fast" if self.model_config.fast_mode else "full",
                    ),
                )
            self.conn.commit()
        except Exception as e:
            logger.warning("Failed to store audit for %s: %s", region, e)

    # ── Lifecycle ──────────────────────────────────────

    def close(self):
        """Close all connections."""
        self.redis_client.close()
        self.conn.close()
        self.feature_builder.close()


def run(mode: str = "inference"):
    """Entry point called by Airflow tasks."""
    scorer = BatchScorer()
    try:
        scorer.score_all_regions(mode=mode)
    finally:
        scorer.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    m = sys.argv[1] if len(sys.argv) > 1 else "inference"
    run(mode=m)
