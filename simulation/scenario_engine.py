"""
Scenario simulation engine for "What-If" weather analysis.

Per spec §How the Scenario Engine Works:
1. Copy feature matrix (NEVER mutate input)
2. Override specified weather columns with user values
3. Recompute ALL derived features (CDD, HDD, wind power, solar CF, etc.)
4. Re-run ensemble forecast
5. Compute deltas vs baseline
6. Estimate price impact

The scenario simulator sliders display mph for user familiarity;
all conversions happen internally.
"""

from typing import Any

import numpy as np
import pandas as pd
import structlog

from config import REGION_CAPACITY_MW, WEATHER_VARIABLES
from data.feature_engineering import (
    compute_cdd,
    compute_hdd,
    compute_solar_capacity_factor,
    compute_temp_hour_interaction,
    compute_temperature_deviation,
    compute_wind_power,
)
from models.pricing import estimate_price_impact

log = structlog.get_logger()

# Columns that can be overridden by sliders
OVERRIDABLE_COLUMNS = set(WEATHER_VARIABLES) | {
    "cooling_degree_days",
    "heating_degree_days",
    "wind_power_estimate",
    "solar_capacity_factor",
}


def simulate_scenario(
    features: pd.DataFrame,
    weather_overrides: dict[str, float],
    models: dict[str, Any],
    base_forecast: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a "What-If" scenario simulation.

    Replace weather features with user-specified values, recompute derived
    features, re-run the ensemble forecast, and return deltas.

    Args:
        features: Feature matrix (will be COPIED, not mutated).
        weather_overrides: Dict of {column_name: override_value}.
        models: Dict with trained model objects (from training.load_models).
        base_forecast: Optional pre-computed baseline. If None, computed
                       from unmodified features.

    Returns:
        (scenario_forecast, delta) where delta = scenario - baseline.

    Raises:
        ValueError: If an unknown weather column is specified.
    """
    # Validate override columns
    for col in weather_overrides:
        if col not in OVERRIDABLE_COLUMNS:
            raise ValueError(
                f"Unknown weather column: '{col}'. Valid columns: {sorted(OVERRIDABLE_COLUMNS)}"
            )

    # 1. Copy features (NEVER mutate input)
    scenario_features = features.copy()

    # 2. Compute baseline forecast if not provided
    if base_forecast is None:
        base_forecast = _run_ensemble(features, models)

    # 3. Apply weather overrides
    for col, value in weather_overrides.items():
        if col in scenario_features.columns:
            scenario_features[col] = value
            log.debug("scenario_override", column=col, value=value)

    # 4. Recompute ALL derived features
    scenario_features = _recompute_derived_features(scenario_features)

    # 5. Re-run ensemble forecast
    scenario_forecast = _run_ensemble(scenario_features, models)

    # 6. Compute deltas
    delta = scenario_forecast - base_forecast

    log.info(
        "scenario_simulated",
        overrides=list(weather_overrides.keys()),
        mean_delta=round(float(np.mean(delta)), 1),
        max_delta=round(float(np.max(delta)), 1),
    )

    return scenario_forecast, delta


def compute_scenario_impact(
    scenario_forecast: np.ndarray,
    base_forecast: np.ndarray,
    region: str,
) -> dict[str, Any]:
    """
    Compute full impact metrics for a scenario vs baseline.

    Returns:
        Dict with demand delta, price impact, reserve margin, etc.
    """
    delta = scenario_forecast - base_forecast
    capacity = REGION_CAPACITY_MW.get(region, 100_000)

    base_price = estimate_price_impact(base_forecast, capacity)
    scenario_price = estimate_price_impact(scenario_forecast, capacity)

    return {
        "demand_delta_mw": delta,
        "demand_delta_pct": (delta / base_forecast * 100) if base_forecast.any() else delta * 0,
        "peak_demand_mw": float(np.max(scenario_forecast)),
        "peak_delta_mw": float(np.max(delta)),
        "base_price": base_price,
        "scenario_price": scenario_price,
        "price_delta": scenario_price - base_price,
        "reserve_margin_pct": (capacity - scenario_forecast) / capacity * 100,
        "min_reserve_margin_pct": float(np.min((capacity - scenario_forecast) / capacity * 100)),
    }


def _recompute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute all derived features after weather overrides.

    This is critical: if temperature changes, CDD/HDD must update.
    If wind changes, wind power estimate must update.
    """
    if "temperature_2m" in df.columns:
        df["cooling_degree_days"] = compute_cdd(df["temperature_2m"])
        df["heating_degree_days"] = compute_hdd(df["temperature_2m"])
        df["temperature_deviation"] = compute_temperature_deviation(df["temperature_2m"])

    if "wind_speed_80m" in df.columns:
        df["wind_power_estimate"] = compute_wind_power(df["wind_speed_80m"])

    if "shortwave_radiation" in df.columns:
        df["solar_capacity_factor"] = compute_solar_capacity_factor(df["shortwave_radiation"])

    if "temperature_2m" in df.columns and "hour_sin" in df.columns:
        df["temp_x_hour"] = compute_temp_hour_interaction(df["temperature_2m"], df["hour_sin"])

    return df


def _run_ensemble(features: pd.DataFrame, models: dict[str, Any]) -> np.ndarray:
    """
    Run ensemble forecast using loaded models.

    Tries XGBoost first (fastest), falls back gracefully.
    """
    from models.xgboost_model import predict_xgboost

    forecasts = {}
    weights = models.get("ensemble_weights", {})

    # XGBoost
    if "xgboost_model" in models and "xgboost_feature_names" in models:
        try:
            xgb_dict = {
                "model": models["xgboost_model"],
                "feature_names": models["xgboost_feature_names"],
            }
            forecasts["xgboost"] = predict_xgboost(xgb_dict, features)
        except Exception as e:
            log.warning("scenario_xgboost_failed", error=str(e))

    # Prophet (if available and has predict method)
    if "prophet_model" in models:
        try:
            from models.prophet_model import predict_prophet

            pred = predict_prophet(models["prophet_model"], features, periods=len(features))
            forecasts["prophet"] = pred["forecast"][: len(features)]
        except Exception as e:
            log.warning("scenario_prophet_failed", error=str(e))

    if not forecasts:
        log.error("scenario_no_models_available")
        return np.zeros(len(features))

    # Weighted ensemble
    from models.ensemble import ensemble_combine

    return ensemble_combine(forecasts, weights)
