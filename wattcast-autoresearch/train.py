#!/usr/bin/env python3
"""
train.py — WattCast XGBoost Training Pipeline
===============================================
THIS IS THE EDITABLE ASSET. The agent modifies this file.

Trains an XGBoost model for energy demand forecasting across 8 U.S. grid regions.
Outputs metrics to results/metrics.json for the autoresearch loop to evaluate.

Metric: MAPE (Mean Absolute Percentage Error) — lower is better.
Baseline: 3.13%

Usage:
    python train.py
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
# CONSTANTS (agent: do not change these)
# ─────────────────────────────────────────────

METRIC = "mape"  # Optimization target
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
TARGET_COL = "demand_mw"

# ─────────────────────────────────────────────
# FEATURE CONFIGURATION
# The agent CAN and SHOULD modify this section.
# Experiment with different feature sets, add
# new derived features, remove noisy ones, etc.
# ─────────────────────────────────────────────

# Core weather features (pruned: removed snowfall, rain, precipitation, wind_direction,
# wind_gusts, diffuse_radiation, direct_radiation — near-zero importance)
WEATHER_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dewpoint_2m",
    "apparent_temperature",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
]

# Temporal features
TEMPORAL_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
]

# Lag features (demand)
DEMAND_LAG_FEATURES = [
    "demand_lag_1h",
    "demand_lag_2h",
    "demand_lag_3h",
    "demand_lag_6h",
    "demand_lag_12h",
    "demand_lag_24h",
    "demand_lag_48h",
    "demand_lag_168h",  # 1 week
]

# Temperature lag features
TEMP_LAG_FEATURES = [
    "temp_lag_1h",
    "temp_lag_2h",
    "temp_lag_3h",
    "temp_lag_6h",
    "temp_lag_12h",
    "temp_lag_24h",
]

# Rolling statistics
ROLLING_FEATURES = [
    "demand_rolling_mean_6h",
    "demand_rolling_std_6h",
    "demand_rolling_mean_12h",
    "demand_rolling_std_12h",
    "demand_rolling_mean_24h",
    "demand_rolling_std_24h",
    "demand_rolling_mean_48h",
    "demand_rolling_std_48h",
    "demand_rolling_mean_168h",
    "demand_rolling_std_168h",
]

# Interaction / derived features
INTERACTION_FEATURES = [
    "temp_squared",
    "temp_humidity_interaction",
    "cooling_degree_hours",
    "heating_degree_hours",
    "wind_chill_factor",
    "heat_index_proxy",
    "solar_is_up",
    "solar_net_radiation",
    "demand_momentum_short",
    "demand_momentum_long",
    "demand_ratio_24h",
    "demand_ratio_168h",
]


def get_feature_columns() -> list[str]:
    """
    Assemble the full feature list.
    Agent: modify this to experiment with feature selection.
    """
    features = (
        WEATHER_FEATURES
        + TEMPORAL_FEATURES
        + DEMAND_LAG_FEATURES
        + TEMP_LAG_FEATURES
        + ROLLING_FEATURES
        + INTERACTION_FEATURES
    )
    return features


# ─────────────────────────────────────────────
# CUSTOM FEATURE ENGINEERING
# Agent: add new derived features here.
# This function runs AFTER loading data and
# BEFORE feature selection.
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional derived features.
    Agent: this is your playground. Add new features here
    and include them in the feature lists above.
    """
    # Exp 1: demand momentum — captures ramping direction
    df["demand_momentum_short"] = df["demand_lag_1h"] - df["demand_lag_3h"]
    df["demand_momentum_long"] = df["demand_lag_1h"] - df["demand_lag_24h"]

    # Exp 19: same-hour yesterday ratio — captures daily deviation
    df["demand_ratio_24h"] = df["demand_lag_24h"] / df["demand_rolling_mean_24h"].clip(lower=1)

    # Exp 20: weekly deviation ratio — same-day-last-week vs weekly average
    df["demand_ratio_168h"] = df["demand_lag_168h"] / df["demand_rolling_mean_168h"].clip(lower=1)

    return df


# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# Agent: experiment with hyperparameters,
# try different model architectures, etc.
# ─────────────────────────────────────────────

def get_model_params() -> dict:
    """
    XGBoost hyperparameters.
    Agent: tune these aggressively.
    """
    return {
        "objective": "reg:squarederror",
        "eval_metric": "mape",
        "tree_method": "hist",
        "n_estimators": 6000,
        "max_depth": 8,
        "learning_rate": 0.015,
        "early_stopping_rounds": 100,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "colsample_bylevel": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.01,      # L1 regularization
        "reg_lambda": 0.5,      # L2 regularization
        "gamma": 0.05,          # min loss reduction for split
        "max_bin": 512,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }


# ─────────────────────────────────────────────
# PREPROCESSING
# Agent: experiment with scaling, outlier
# handling, missing value strategies, etc.
# ─────────────────────────────────────────────

def preprocess(train_df: pd.DataFrame, val_df: pd.DataFrame, features: list[str]):
    """
    Preprocess train and validation data.
    Agent: modify preprocessing strategy here.
    """
    # Currently: no scaling (XGBoost handles raw features well)
    # Agent could try: log transforms, outlier clipping, binning, etc.
    
    X_train = train_df[features].copy()
    y_train = train_df[TARGET_COL].copy()
    X_val = val_df[features].copy()
    y_val = val_df[TARGET_COL].copy()

    # Handle any remaining NaN
    X_train = X_train.fillna(X_train.median())
    X_val = X_val.fillna(X_train.median())

    return X_train, y_train, X_val, y_val


# ─────────────────────────────────────────────
# TRAINING
# Agent: experiment with training strategy,
# early stopping, ensemble methods, etc.
# ─────────────────────────────────────────────

def train_model(X_train, y_train, X_val, y_val, params: dict) -> xgb.XGBRegressor:
    """
    Train the XGBoost model.
    Agent: modify training strategy here.
    """
    model = xgb.XGBRegressor(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    if hasattr(model, "best_iteration"):
        print(f"  Best iteration: {model.best_iteration}")

    return model


# ─────────────────────────────────────────────
# EVALUATION
# Agent: do NOT modify this section.
# This ensures metrics are always computed
# the same way for fair comparison.
# ─────────────────────────────────────────────

def evaluate(model, X_val, y_val, val_df) -> dict:
    """
    Evaluate model and compute metrics.
    DO NOT MODIFY — must remain consistent across experiments.
    """
    preds = model.predict(X_val)
    
    # Clip predictions to reasonable range
    preds = np.clip(preds, 0, None)
    
    # Overall MAPE
    mape = np.mean(np.abs((y_val - preds) / y_val)) * 100
    
    # Overall RMSE
    rmse = np.sqrt(np.mean((y_val - preds) ** 2))
    
    # Overall MAE
    mae = np.mean(np.abs(y_val - preds))
    
    # Overall R²
    ss_res = np.sum((y_val - preds) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Per-region MAPE
    region_mape = {}
    for region in val_df["region"].unique():
        mask = val_df["region"] == region
        region_y = y_val[mask.values]
        region_p = preds[mask.values]
        region_mape[region] = round(
            np.mean(np.abs((region_y - region_p) / region_y)) * 100, 4
        )
    
    # Feature importance (top 20)
    feature_importance = {}
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        feature_names = X_val.columns.tolist()
        sorted_idx = np.argsort(fi)[::-1][:20]
        feature_importance = {
            feature_names[i]: round(float(fi[i]), 4) for i in sorted_idx
        }
    
    return {
        "mape": round(float(mape), 4),
        "rmse": round(float(rmse), 2),
        "mae": round(float(mae), 2),
        "r2": round(float(r2), 4),
        "region_mape": region_mape,
        "feature_importance": feature_importance,
        "n_features": X_val.shape[1],
        "n_train_samples": len(X_val),
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("⚡ WattCast Training Run")
    print("=" * 40)
    
    start_time = datetime.now()
    
    # Load data
    print("  Loading data...")
    if (DATA_DIR / "train.parquet").exists():
        train_df = pd.read_parquet(DATA_DIR / "train.parquet")
        val_df = pd.read_parquet(DATA_DIR / "val.parquet")
    else:
        train_df = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["timestamp"])
        val_df = pd.read_csv(DATA_DIR / "val.csv", parse_dates=["timestamp"])
    
    # Engineer additional features
    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df)
    
    # Get feature configuration
    features = get_feature_columns()
    
    # Validate features exist
    available = set(train_df.columns)
    features = [f for f in features if f in available]
    print(f"  Features: {len(features)}")
    
    # Preprocess
    X_train, y_train, X_val, y_val = preprocess(train_df, val_df, features)
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Val:   {len(X_val):,} samples")
    
    # Train
    print("  Training XGBoost...")
    params = get_model_params()
    model = train_model(X_train, y_train, X_val, y_val, params)
    
    # Evaluate
    print("  Evaluating...")
    metrics = evaluate(model, X_val, y_val, val_df)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["timestamp"] = datetime.now().isoformat()
    metrics["params"] = {k: v for k, v in params.items() if k != "random_state"}
    
    # Save metrics
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    
    # Append to history
    with open(RESULTS_DIR / "history.jsonl", "a") as f:
        f.write(json.dumps(metrics) + "\n")
    
    # Print results
    print(f"\n  {'─' * 36}")
    print(f"  MAPE:  {metrics['mape']:.4f}%  ← optimization target")
    print(f"  RMSE:  {metrics['rmse']:.2f} MW")
    print(f"  MAE:   {metrics['mae']:.2f} MW")
    print(f"  R²:    {metrics['r2']:.4f}")
    print(f"  Time:  {elapsed:.1f}s")
    print(f"  {'─' * 36}")
    print(f"  Region MAPE:")
    for region, m in sorted(metrics["region_mape"].items()):
        bar = "█" * int(m * 3)
        print(f"    {region:6s}  {m:6.3f}%  {bar}")
    print(f"  {'─' * 36}")
    print(f"  Top features:")
    for feat, imp in list(metrics["feature_importance"].items())[:10]:
        bar = "█" * int(imp * 100)
        print(f"    {feat:30s}  {imp:.4f}  {bar}")
    print(f"\n  ✅ Metrics saved to {RESULTS_DIR}/metrics.json\n")


if __name__ == "__main__":
    main()
