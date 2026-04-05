"""
XGBoost forecasting model with full feature set.

Per spec §Model 3:
- Captures non-linear weather-demand relationships
- Validated via TimeSeriesSplit (no data leakage)
- Provides SHAP feature importance for model explainability
"""

from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

log = structlog.get_logger()

# Features to exclude from model input
EXCLUDE_COLS = {"timestamp", "region", "data_quality", "forecast_mw", "demand_mw"}

# Default hyperparameters (tuned via autoresearch: 30 experiments, 16.4% MAPE improvement)
DEFAULT_PARAMS = {
    "n_estimators": 6000,
    "max_depth": 8,
    "learning_rate": 0.015,
    "early_stopping_rounds": 100,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "colsample_bylevel": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.01,
    "reg_lambda": 0.5,
    "gamma": 0.05,
    "max_bin": 512,
    "random_state": 42,
    "n_jobs": -1,
}


def train_xgboost(
    df: pd.DataFrame,
    target_col: str = "demand_mw",
    params: dict | None = None,
    n_splits: int = 5,
) -> dict[str, Any]:
    """
    Train an XGBoost model with TimeSeriesSplit cross-validation.

    Args:
        df: Feature-engineered DataFrame.
        target_col: Column to forecast.
        params: XGBoost hyperparameters (default: spec configuration).
        n_splits: Number of TimeSeriesSplit folds.

    Returns:
        Dict with 'model', 'feature_names', 'feature_importances', 'cv_scores'.
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    # Extract early_stopping_rounds — passed to fit(), not constructor
    early_stopping_rounds = params.pop("early_stopping_rounds", None)

    feature_cols = _get_feature_cols(df)
    X = df[feature_cols].values  # noqa: N806
    y = df[target_col].values

    log.info("xgboost_training", features=len(feature_cols), samples=len(X), n_splits=n_splits)

    # Cross-validation with TimeSeriesSplit (no data leakage)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # Verify no leakage: all train indices < all val indices
        assert train_idx.max() < val_idx.min(), "TimeSeriesSplit leakage detected"

        X_train, X_val = X[train_idx], X[val_idx]  # noqa: N806
        y_train, y_val = y[train_idx], y[val_idx]

        fold_model = XGBRegressor(**params)
        fit_kwargs: dict = {"eval_set": [(X_val, y_val)], "verbose": False}
        if early_stopping_rounds:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
        fold_model.fit(X_train, y_train, **fit_kwargs)

        y_pred = fold_model.predict(X_val)
        mape = _compute_mape(y_val, y_pred)
        cv_scores.append(mape)
        log.debug("xgboost_fold", fold=fold, mape=round(mape, 2))

    # Train final model on all data (no early stopping — no validation set)
    model = XGBRegressor(**params)
    model.fit(X, y, verbose=False)

    importances = model.feature_importances_

    log.info(
        "xgboost_trained",
        mean_cv_mape=round(np.mean(cv_scores), 2),
        std_cv_mape=round(np.std(cv_scores), 2),
        top_features=_top_features(feature_cols, importances, n=5),
    )

    return {
        "model": model,
        "feature_names": feature_cols,
        "feature_importances": dict(zip(feature_cols, importances.tolist(), strict=False)),
        "cv_scores": cv_scores,
    }


def predict_xgboost(
    model_dict: dict[str, Any],
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Generate predictions from a trained XGBoost model.

    Args:
        model_dict: Output from train_xgboost().
        df: DataFrame with feature columns matching training.

    Returns:
        Prediction array.
    """
    model = model_dict["model"]
    feature_cols = model_dict["feature_names"]

    # Ensure all expected features exist
    missing = set(feature_cols) - set(df.columns)
    if missing:
        log.warning("xgboost_missing_features", missing=list(missing))
        for col in missing:
            df = df.copy()
            df[col] = 0.0

    X = df[feature_cols].values  # noqa: N806
    predictions = model.predict(X)

    # Clamp negative predictions
    predictions = np.maximum(predictions, 0)

    return predictions


def compute_shap_values(
    model_dict: dict[str, Any],
    df: pd.DataFrame,
    max_samples: int = 500,
) -> dict[str, Any]:
    """
    Compute SHAP values for XGBoost model explainability.

    Args:
        model_dict: Output from train_xgboost().
        df: DataFrame with feature columns.
        max_samples: Max samples for SHAP computation (performance).

    Returns:
        Dict with 'shap_values' (array), 'feature_names', 'base_value'.
    """
    import shap

    model = model_dict["model"]
    feature_cols = model_dict["feature_names"]
    X = df[feature_cols].values  # noqa: N806

    if len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X = X[indices]  # noqa: N806

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    log.info("shap_computed", samples=len(X), features=len(feature_cols))

    return {
        "shap_values": shap_values,
        "feature_names": feature_cols,
        "base_value": float(explainer.expected_value),
    }


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Get numeric feature columns, excluding target and metadata."""
    return [col for col in df.select_dtypes(include=[np.number]).columns if col not in EXCLUDE_COLS]


def _compute_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error, excluding near-zero actuals."""
    mask = np.abs(actual) > 1e-6
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _top_features(names: list[str], importances: np.ndarray, n: int = 5) -> list[str]:
    """Return top N feature names by importance."""
    indices = np.argsort(importances)[::-1][:n]
    return [names[i] for i in indices]
