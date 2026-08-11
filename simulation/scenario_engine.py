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
from models.pricing import capacity_headroom_pct, estimate_price_impact

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


def apply_weather_deltas(
    features: pd.DataFrame,
    temp_delta_f: float = 0.0,
    wind_delta_mph: float = 0.0,
    solar_delta_wm2: float = 0.0,
) -> pd.DataFrame:
    """Offset the weather drivers by slider deltas and recompute what depends on them.

    ``simulate_scenario`` sets weather columns to *absolute* values, which is
    the right shape for a preset ("what if it were 105 °F"). The simulator's
    sliders are *relative* ("what if it were 10 °F warmer than forecast"), and
    a relative shift has to preserve the diurnal curve rather than flatten it
    — setting ``temperature_2m`` to a constant would erase the day/night cycle
    that drives the shape of the demand response.

    Units are the frame's own: ``temperature_2m`` is °F and ``wind_speed_80m``
    is mph (Open-Meteo is queried with ``temperature_unit=fahrenheit``, and
    ``compute_wind_power`` documents its input as mph), so slider values apply
    directly with no conversion.

    Args:
        features: Feature matrix. COPIED, never mutated — ADR-007.
        temp_delta_f: Temperature offset in °F.
        wind_delta_mph: Wind-speed offset in mph, clipped at zero (a negative
            wind speed is not a physical state, and ``compute_wind_power``
            cubes its input).
        solar_delta_wm2: Shortwave-radiation offset in W/m², clipped at zero
            for the same reason.

    Returns:
        A new frame with the drivers offset and every derived feature that
        depends on them recomputed.
    """
    scenario = features.copy()

    if temp_delta_f and "temperature_2m" in scenario.columns:
        scenario["temperature_2m"] = scenario["temperature_2m"] + temp_delta_f
    if wind_delta_mph and "wind_speed_80m" in scenario.columns:
        scenario["wind_speed_80m"] = (scenario["wind_speed_80m"] + wind_delta_mph).clip(lower=0.0)
    if solar_delta_wm2 and "shortwave_radiation" in scenario.columns:
        scenario["shortwave_radiation"] = (scenario["shortwave_radiation"] + solar_delta_wm2).clip(
            lower=0.0
        )

    # CDD/HDD, wind power, solar CF and temp x hour are POINTWISE functions of
    # their drivers, so recomputing them on a 24-row scenario frame gives the
    # same answer it would on the full frame.
    scenario = _recompute_derived_features(scenario)

    # `temperature_deviation` is NOT pointwise — it is a 720-hour rolling mean
    # (`compute_temperature_deviation`), and `_recompute_derived_features`
    # recomputed it against whatever slice it was handed. On the simulator's
    # 24-row frame `min_periods=1` made each row's reference a <=24h mean
    # instead of 30 days, so a scenario with IDENTICAL weather still got
    # different features from the baseline. Measured 2026-08-11 on the first
    # tick that computed the origin cell: up to 0.013 drift on FPL, non-zero
    # on 5 of 6 BAs.
    #
    # The correct adjustment is analytic rather than a recomputation. The
    # 30-day reference is dominated by history this scenario does not touch —
    # 24 shifted hours against a 720-hour window — so a uniform shift of d
    # moves the deviation by d and leaves everything else alone. At zero delta
    # that is exactly a no-op, which is what makes the origin cell a real
    # parity check rather than a measurement of this bug.
    if "temperature_deviation" in features.columns:
        base_dev = features["temperature_deviation"]
        scenario["temperature_deviation"] = base_dev + temp_delta_f

    return scenario


def compute_scenario_impact(
    scenario_forecast: np.ndarray,
    base_forecast: np.ndarray,
    region: str,
) -> dict[str, Any]:
    """
    Compute full impact metrics for a scenario vs baseline.

    Returns:
        Dict with demand delta, price impact, capacity headroom, etc.
        (headroom is nameplate-based, not a NERC reserve margin — see #243.)
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
        "headroom_pct": capacity_headroom_pct(scenario_forecast, capacity),
        "min_headroom_pct": float(np.min(capacity_headroom_pct(scenario_forecast, capacity))),
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
