"""
Training orchestrator for all forecasting models.

Manages the full training pipeline:
1. Train Prophet, SARIMAX, XGBoost on feature-engineered data
2. Evaluate each model on a held-out validation set
3. Compute ensemble weights (1/MAPE)
4. Serialize all models to disk

Designed to run on startup or on a scheduled interval (default: 24h).
"""

import os
import pickle  # noqa: S403 — restricted to trusted, integrity-checked files
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from config import MODEL_DIR, REGION_COORDINATES
from data.feature_engineering import compute_autoregressive_snapshot, engineer_exogenous_features
from models.arima_model import predict_arima, train_arima
from models.ensemble import compute_ensemble_weights
from models.evaluation import compute_all_metrics
from models.prophet_model import predict_prophet, train_prophet
from models.xgboost_model import predict_xgboost, train_xgboost

log = structlog.get_logger()


def train_all_models(
    df: pd.DataFrame,
    region: str,
    target_col: str = "demand_mw",
    validation_hours: int = 168,
) -> dict[str, Any]:
    """
    Train all models for a single region and compute ensemble weights.

    Args:
        df: Feature-engineered DataFrame (should cover TRAINING_WINDOW_DAYS).
        region: Balancing authority code.
        target_col: Column to forecast.
        validation_hours: Hours to hold out for validation (default: 7 days).

    Returns:
        Dict with trained models, metrics, and ensemble weights.
    """
    log.info("training_start", region=region, rows=len(df))

    # Train/validation split (temporal — no leakage)
    val_cutoff = len(df) - validation_hours
    train_df = df.iloc[:val_cutoff]
    val_raw = df.iloc[val_cutoff:].copy()
    val_df = engineer_exogenous_features(val_raw)

    if len(train_df) < 720:  # Need at least 30 days
        log.warning("insufficient_training_data", region=region, rows=len(train_df))

    y_val = val_df[target_col].values
    results = {}
    metrics = {}

    # --- Prophet ---
    try:
        prophet_model = train_prophet(train_df, target_col=target_col)
        prophet_pred = predict_prophet(prophet_model, val_df, periods=len(val_df))
        prophet_forecast = prophet_pred["forecast"][: len(y_val)]
        metrics["prophet"] = compute_all_metrics(y_val, prophet_forecast)
        results["prophet"] = {"model": prophet_model, "predictions": prophet_pred}
        log.info(
            "prophet_evaluation",
            region=region,
            **{f"prophet_{k}": round(v, 3) for k, v in metrics["prophet"].items()},
        )
    except Exception as e:
        log.error("prophet_training_failed", region=region, error=str(e))
        metrics["prophet"] = {
            "mape": float("inf"),
            "rmse": float("inf"),
            "mae": float("inf"),
            "r2": 0,
        }

    # --- ARIMA/SARIMAX ---
    try:
        arima_result = train_arima(train_df, target_col=target_col, auto_order=True)
        arima_forecast = predict_arima(arima_result, val_df, periods=len(val_df))
        arima_forecast = arima_forecast[: len(y_val)]
        metrics["arima"] = compute_all_metrics(y_val, arima_forecast)
        results["arima"] = {"model": arima_result, "predictions": arima_forecast}
        log.info(
            "arima_evaluation",
            region=region,
            **{f"arima_{k}": round(v, 3) for k, v in metrics["arima"].items()},
        )
    except Exception as e:
        log.error("arima_training_failed", region=region, error=str(e))
        metrics["arima"] = {
            "mape": float("inf"),
            "rmse": float("inf"),
            "mae": float("inf"),
            "r2": 0,
        }

    # --- XGBoost ---
    try:
        xgb_result = train_xgboost(train_df, target_col=target_col)
        demand_history = train_df[target_col].tolist()
        xgb_steps: list[float] = []
        for i in range(len(val_df)):
            row = val_df.iloc[[i]].copy()
            for col, val in compute_autoregressive_snapshot(demand_history).items():
                row[col] = val
            row = row.fillna(method="ffill").fillna(method="bfill").fillna(0)
            pred = float(predict_xgboost(xgb_result, row)[0])
            xgb_steps.append(pred)
            demand_history.append(pred)
        xgb_forecast = np.array(xgb_steps[: len(y_val)], dtype=float)
        metrics["xgboost"] = compute_all_metrics(y_val, xgb_forecast)
        results["xgboost"] = {"model": xgb_result, "predictions": xgb_forecast}
        log.info(
            "xgboost_evaluation",
            region=region,
            **{f"xgb_{k}": round(v, 3) for k, v in metrics["xgboost"].items()},
        )
    except Exception as e:
        log.error("xgboost_training_failed", region=region, error=str(e))
        metrics["xgboost"] = {
            "mape": float("inf"),
            "rmse": float("inf"),
            "mae": float("inf"),
            "r2": 0,
        }

    # --- Ensemble Weights ---
    mape_scores = {name: m["mape"] for name, m in metrics.items() if np.isfinite(m["mape"])}
    if mape_scores:
        weights = compute_ensemble_weights(mape_scores)
    else:
        weights = {name: 1.0 / len(metrics) for name in metrics}
        log.warning("ensemble_fallback_equal_weights", region=region)

    return {
        "region": region,
        "models": results,
        "metrics": metrics,
        "ensemble_weights": weights,
        "validation_actual": y_val,
    }


def save_models(training_result: dict[str, Any], output_dir: str | None = None) -> str:
    """
    Serialize all trained models to disk.

    Args:
        training_result: Output from train_all_models().
        output_dir: Directory for model files (default: MODEL_DIR).

    Returns:
        Path to the saved model file.
    """
    output_dir = output_dir or MODEL_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    region = training_result["region"]
    _validate_region(region)
    filepath = _safe_model_path(output_dir, region)

    # Store only serializable parts
    save_data = {
        "region": region,
        "ensemble_weights": training_result["ensemble_weights"],
        "metrics": training_result["metrics"],
    }

    # Save individual models
    for name, result in training_result["models"].items():
        if name == "xgboost":
            save_data[f"{name}_model"] = result["model"]["model"]
            save_data[f"{name}_feature_names"] = result["model"]["feature_names"]
        elif name == "arima" or name == "prophet":
            save_data[f"{name}_model"] = result["model"]

    with open(filepath, "wb") as f:
        pickle.dump(save_data, f)

    log.info("models_saved", region=region, filepath=filepath)
    return filepath


def load_models(region: str, model_dir: str | None = None) -> dict[str, Any]:
    """
    Load serialized models from disk.

    Args:
        region: Balancing authority code.
        model_dir: Directory containing model files.

    Returns:
        Dict with model objects and metadata.
    """
    _validate_region(region)
    model_dir = model_dir or MODEL_DIR
    filepath = _safe_model_path(model_dir, region)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No trained models for region {region} at {filepath}")

    with open(filepath, "rb") as f:
        data = pickle.load(f)  # noqa: S301 — region is validated against allowlist

    log.info("models_loaded", region=region, filepath=filepath)
    return data


def _validate_region(region: str) -> None:
    """Validate region is a known balancing authority code."""
    if not re.match(r"^[A-Z0-9]+$", region):
        raise ValueError(f"Invalid region format: {region}")
    if region not in REGION_COORDINATES:
        raise ValueError(f"Unknown region: {region}")


def _safe_model_path(base_dir: str, region: str) -> str:
    """Build a model file path, preventing path traversal."""
    base = Path(base_dir).resolve()
    target = (base / f"{region}_models.pkl").resolve()
    if not str(target).startswith(str(base) + os.sep) and target.parent != base:
        raise ValueError(f"Path traversal detected for region: {region}")
    return str(target)
