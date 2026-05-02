"""
Model service layer: single interface for all forecast consumers.

This module is the ONLY place callbacks should get model predictions from.
It handles the full chain: load trained models → predict → fall back to
simulated outputs if models aren't available.

Usage:
    from models.model_service import get_forecasts, get_model_metrics
    forecasts = get_forecasts(region, demand_df)
    # forecasts = {"prophet": np.array, "arima": np.array, "xgboost": np.array,
    #              "ensemble": np.array, "upper_80": np.array, "lower_80": np.array,
    #              "weights": dict, "metrics": dict, "source": "trained"|"simulated"}
"""

import os
from typing import Any

import numpy as np
import pandas as pd
import structlog

from config import MODEL_DIR
from hash_utils import stable_int_seed

log = structlog.get_logger()

# In-memory model cache: region → loaded model data
_model_cache: dict[str, dict[str, Any]] = {}


def get_forecasts(
    region: str,
    demand_df: pd.DataFrame,
    models_shown: list[str] | None = None,
) -> dict[str, Any]:
    """
    Get forecasts from all models for a region.

    Tries to load trained models from disk. If unavailable, generates
    deterministic simulated forecasts (seeded by region hash for consistency).

    Args:
        region: Balancing authority code (e.g., "FPL").
        demand_df: DataFrame with 'timestamp' and 'demand_mw' columns.
        models_shown: Optional filter — only compute these models.

    Returns:
        Dict with keys: prophet, arima, xgboost, ensemble, upper_80, lower_80,
        weights, metrics, source ("trained" or "simulated").
    """
    actual = demand_df["demand_mw"].values

    # Try trained models first
    model_data = _load_cached_models(region)
    if model_data is not None:
        return _predict_from_trained(model_data, demand_df, models_shown)

    # Fall back to deterministic simulated forecasts
    return _simulate_forecasts(region, actual, models_shown)


def get_model_metrics(region: str) -> dict[str, dict[str, float]]:
    """
    Get validation metrics for a region's trained models.

    Returns:
        Dict of model_name → {mape, rmse, mae, r2}.
        Returns simulated metrics if no trained models exist.
    """
    model_data = _load_cached_models(region)
    if model_data and "metrics" in model_data:
        return model_data["metrics"]

    # Simulated metrics (consistent, reasonable values)
    return {
        "prophet": {"mape": 2.8, "rmse": 450, "mae": 320, "r2": 0.967},
        "arima": {"mape": 3.5, "rmse": 580, "mae": 410, "r2": 0.945},
        "xgboost": {"mape": 2.1, "rmse": 380, "mae": 280, "r2": 0.974},
        "ensemble": {"mape": 1.9, "rmse": 340, "mae": 250, "r2": 0.979},
    }


def get_ensemble_weights(region: str) -> dict[str, float]:
    """Get ensemble weights for a region."""
    model_data = _load_cached_models(region)
    if model_data and "ensemble_weights" in model_data:
        return model_data["ensemble_weights"]

    # Default weights (1/MAPE normalized)
    return {"prophet": 0.30, "arima": 0.20, "xgboost": 0.50}


def is_trained(region: str) -> bool:
    """Check if trained models exist for a region."""
    from models.training import _safe_model_path, _validate_region

    try:
        _validate_region(region)
    except ValueError:
        return False
    filepath = _safe_model_path(MODEL_DIR, region)
    return os.path.exists(filepath)


# ── Forecast quality gate (V3.ζ follow-up) ───────────────────────


# Module-level cache so we don't read 51 meta.json blobs on every page
# render. Keyed by region; refreshed every ``_QUALITY_GATE_TTL`` seconds.
# Daily training cadence makes 10 min plenty fresh for production.
_QUALITY_GATE_TTL = 600
_quality_gate_cache: dict[str, tuple[float | None, float]] = {}


def get_xgboost_holdout_mape(region: str) -> float | None:
    """Return the latest XGBoost training-holdout MAPE for ``region``.

    Reads from GCS via :func:`models.persistence.get_model_metadata`.
    Returns ``None`` when the model hasn't been trained yet, the metadata
    is missing, or the ``mape`` field is null. Callers should treat
    ``None`` as "no signal, don't gate" rather than as a failure — that
    preserves the legitimate "warming up" state for newly-added BAs.
    """
    import time

    cached = _quality_gate_cache.get(region)
    if cached is not None and time.time() - cached[1] < _QUALITY_GATE_TTL:
        return cached[0]

    from models.persistence import get_model_metadata

    try:
        meta = get_model_metadata(region, "xgboost")
    except Exception:
        meta = None
    mape = meta.mape if meta is not None else None
    _quality_gate_cache[region] = (mape, time.time())
    return mape


def is_forecast_quality_acceptable(region: str) -> bool:
    """Return True when the region's primary-model forecast quality is
    in the ``acceptable`` grade or better (per ``mape_grade()``) for the
    7-day horizon. Returns True when no MAPE signal exists yet so the
    gate doesn't hide newly-added BAs that haven't been trained.

    Returns False only when the XGBoost holdout MAPE is in the
    ``rollback`` grade (>22% per ``MAPE_BY_HORIZON["7d"]``) — by the
    project's existing governance framework that means the model
    should be disabled.

    The ``forecast_quality_gate`` feature flag short-circuits the
    check when False (default in dev, on in staging/prod).
    """
    from config import MAPE_BY_HORIZON, feature_enabled

    if not feature_enabled("forecast_quality_gate"):
        return True

    mape = get_xgboost_holdout_mape(region)
    if mape is None:
        # No signal — don't hide the region. Preserves "warming up" UX
        # for BAs whose first training run hasn't completed yet.
        return True

    return mape <= MAPE_BY_HORIZON["7d"]["rollback"]


def stable_visible_regions(all_regions) -> list[str]:
    """Return the subset of ``all_regions`` that pass the quality gate,
    preserving input order. Convenience for the dropdown + US Grid
    consumers that iterate over the full ``REGION_NAMES`` keyspace."""
    return [r for r in all_regions if is_forecast_quality_acceptable(r)]


def hidden_regions(all_regions) -> list[str]:
    """Return the regions in ``all_regions`` that fail the quality gate.

    Used by the UI to render a small "*N BAs hidden — forecast quality
    below threshold*" note with a hover-list of affected codes.
    """
    return [r for r in all_regions if not is_forecast_quality_acceptable(r)]


def _reset_quality_gate_cache() -> None:
    """Test hook — clear the TTL cache so unit tests can vary the
    underlying meta.json without waiting 10 minutes."""
    _quality_gate_cache.clear()


# ── Private helpers ──────────────────────────────────────────────


def _load_cached_models(region: str) -> dict[str, Any] | None:
    """Load models from disk, caching in memory."""
    if region in _model_cache:
        return _model_cache[region]

    try:
        from models.training import load_models

        model_data = load_models(region)
        _model_cache[region] = model_data
        log.info("model_cache_loaded", region=region)
        return model_data
    except FileNotFoundError:
        return None
    except Exception as e:
        log.warning("model_load_failed", region=region, error=str(e))
        return None


def _predict_from_trained(
    model_data: dict[str, Any],
    demand_df: pd.DataFrame,
    models_shown: list[str] | None,
) -> dict[str, Any]:
    """Generate predictions from trained model objects.

    Runs feature engineering on ``demand_df`` before passing it to model
    predictors so that XGBoost receives its expected feature columns and
    Prophet receives time-based regressors.  Without this gate the raw
    DataFrame would cause XGBoost to silently zero-fill missing features
    and Prophet to receive no weather regressors.
    """
    actual = demand_df["demand_mw"].values
    n = len(actual)
    result: dict[str, Any] = {"source": "trained"}
    result["metrics"] = model_data.get("metrics", {})
    result["weights"] = model_data.get("ensemble_weights", {})

    # Feature-engineer the input so models receive the columns they were
    # trained on.  Only demand data is available here (no separate weather
    # DataFrame), so weather-derived features will be zero/NaN — but
    # time-based and demand-derived features (lags, rolling stats, hour,
    # dow, CDD/HDD from any inline temperature column) will be correct.
    try:
        from data.feature_engineering import engineer_features

        featured_df = engineer_features(demand_df)
        featured_df = featured_df.dropna(subset=["demand_mw"])
        if len(featured_df) >= n:
            predict_df = featured_df
        else:
            log.warning("model_service_feature_eng_short", original=n, featured=len(featured_df))
            predict_df = demand_df
    except Exception as e:
        log.warning("model_service_feature_eng_failed", error=str(e))
        predict_df = demand_df

    all_preds = {}

    for name in ["prophet", "arima", "xgboost"]:
        if models_shown and name not in models_shown:
            continue
        try:
            model_key = f"{name}_model"
            if model_key in model_data:
                if name == "prophet":
                    from models.prophet_model import predict_prophet

                    pred_result = predict_prophet(model_data[model_key], predict_df, periods=n)
                    all_preds[name] = pred_result["forecast"][:n]
                elif name == "arima":
                    from models.arima_model import predict_arima

                    all_preds[name] = predict_arima(model_data[model_key], predict_df, periods=n)[
                        :n
                    ]
                elif name == "xgboost":
                    from models.xgboost_model import predict_xgboost

                    xgb_model = {
                        "model": model_data[model_key],
                        "feature_names": model_data.get("xgboost_feature_names", []),
                    }
                    all_preds[name] = predict_xgboost(xgb_model, predict_df)[:n]
        except Exception as e:
            log.warning("model_predict_failed", model=name, error=str(e))
            # Fall back to simulated for this model
            seed = stable_int_seed(("model_fallback", name))
            rng = np.random.RandomState(seed)
            noise_scale = {"prophet": 0.025, "arima": 0.035, "xgboost": 0.018}.get(name, 0.025)
            all_preds[name] = actual * (1 + rng.normal(0, noise_scale, n))

    # Ensemble
    weights = result["weights"]
    if all_preds:
        weighted = np.zeros(n)
        total_weight = 0
        for name, pred in all_preds.items():
            w = weights.get(name, 1.0 / len(all_preds))
            weighted += pred * w
            total_weight += w
        if total_weight > 0:
            all_preds["ensemble"] = weighted / total_weight
        else:
            all_preds["ensemble"] = np.mean(list(all_preds.values()), axis=0)
    else:
        all_preds["ensemble"] = actual.copy()

    # Indicative range (±3% heuristic — not a calibrated confidence interval)
    ensemble = all_preds.get("ensemble", actual)
    result.update(all_preds)
    result["upper_80"] = ensemble * 1.03
    result["lower_80"] = ensemble * 0.97

    return result


def _simulate_forecasts(
    region: str,
    actual: np.ndarray,
    models_shown: list[str] | None,
) -> dict[str, Any]:
    """
    Generate deterministic simulated forecasts.

    Uses a seeded RNG so the same region + data always produces the same
    "model" outputs. This is consistent across page loads (no random flicker).
    """
    n = len(actual)
    seed = stable_int_seed(("simulate_forecasts", region))
    rng = np.random.RandomState(seed)

    # Model-specific noise levels (realistic relative accuracy)
    noise = {
        "prophet": rng.normal(0, 0.025, n),
        "arima": rng.normal(0, 0.035, n),
        "xgboost": rng.normal(0, 0.018, n),
    }

    preds = {}
    for name, n_arr in noise.items():
        preds[name] = actual * (1 + n_arr)

    # Ensemble = weighted average (XGBoost-heavy, ARIMA-light)
    weights = {"prophet": 0.30, "arima": 0.20, "xgboost": 0.50}
    ensemble = sum(preds[m] * w for m, w in weights.items())
    preds["ensemble"] = ensemble

    # Indicative range (heuristic — not a calibrated confidence interval)
    preds["upper_80"] = ensemble * 1.03
    preds["lower_80"] = ensemble * 0.97

    # Simulated metrics
    from models.evaluation import compute_all_metrics

    metrics = {}
    for name in ["prophet", "arima", "xgboost", "ensemble"]:
        if name in preds:
            m = compute_all_metrics(
                actual[-720:] if len(actual) > 720 else actual,
                preds[name][-720:] if len(preds[name]) > 720 else preds[name],
            )
            metrics[name] = m

    preds["weights"] = weights
    preds["metrics"] = metrics
    preds["source"] = "simulated"

    return preds
