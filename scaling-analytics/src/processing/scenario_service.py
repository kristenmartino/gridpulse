"""
On-demand scenario simulation for the POST /scenarios/simulate endpoint.

Handles custom weather scenarios that can't be pre-computed because
the combinatorial space (temperature × wind × cloud × humidity × radiation)
is infinite. The ~2s response time is acceptable because the user is in
exploration mode, not monitoring mode.

DESIGN RULE:
    This module is the ONLY place in the serving layer that runs ML inference.
    It's lazy-imported inside the POST handler to keep the read-only
    endpoints fully isolated from model imports.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import GRID_REGIONS

logger = logging.getLogger(__name__)


def simulate_custom_scenario(
    region: str,
    weather_overrides: dict[str, float],
    duration_hours: int = 24,
) -> dict:
    """
    Run a custom scenario simulation for a region with weather overrides.

    Args:
        region: Grid region code (e.g. "ERCOT").
        weather_overrides: Dict of weather variable overrides, e.g.
            {"temperature_2m": 105, "wind_speed_80m": 5, "cloud_cover": 90}.
        duration_hours: Forecast horizon in hours.

    Returns:
        Dict with baseline, scenario, delta, pricing, reserve_margin,
        and renewable_impact.
    """
    if region.upper() not in GRID_REGIONS:
        raise ValueError(f"Unknown region: {region}")

    region = region.upper()

    # Lazy imports — keep model code out of module-level scope
    from simulation.scenario_engine import simulate_scenario, compute_scenario_impact
    from models.xgboost_model import train_xgboost, predict_xgboost
    from models.pricing import estimate_price_impact, compute_reserve_margin
    from data.demo_data import generate_demo_demand, generate_demo_weather
    from data.preprocessing import merge_demand_weather, handle_missing_values
    from data.feature_engineering import engineer_features, get_feature_names
    from config import REGION_CAPACITY_MW

    # Build features from recent data (demo fallback)
    demand_df = generate_demo_demand(region, days=30)
    weather_df = generate_demo_weather(region, days=30)
    merged = merge_demand_weather(demand_df, weather_df)
    merged = handle_missing_values(merged)
    features_df = engineer_features(merged)

    feature_cols = get_feature_names()
    available_cols = [c for c in feature_cols if c in features_df.columns]
    target_col = "demand_mw"

    if target_col not in features_df.columns or len(features_df) < 48:
        raise RuntimeError(f"Insufficient data for scenario simulation in {region}")

    train_df = features_df[features_df[target_col].notna()].copy()
    future_df = features_df.tail(min(duration_hours * 4, 96)).copy()

    # Train XGBoost (fast — single model for on-demand scenarios)
    xgb_result = train_xgboost(train_df, target_col=target_col)
    baseline = predict_xgboost(xgb_result, future_df)

    # Apply weather overrides and re-score
    try:
        scenario_forecast, delta = simulate_scenario(
            features=future_df,
            weather_overrides=weather_overrides,
            models={"xgboost": xgb_result},
            base_forecast=baseline,
        )
    except Exception as e:
        logger.warning("simulate_scenario failed, using manual override: %s", e)
        scenario_forecast, delta = _manual_scenario(
            future_df, weather_overrides, xgb_result, baseline
        )

    # Compute impacts
    impact = compute_scenario_impact(scenario_forecast, baseline, region)

    capacity = REGION_CAPACITY_MW.get(region, 100_000)
    base_prices = estimate_price_impact(baseline, capacity)
    scenario_prices = estimate_price_impact(scenario_forecast, capacity)
    scenario_reserve = compute_reserve_margin(scenario_forecast, region)

    # Renewable impact estimates
    wind_speed = weather_overrides.get("wind_speed_80m", 15)
    cloud = weather_overrides.get("cloud_cover", 50)
    solar_rad = weather_overrides.get("shortwave_radiation", 500)
    wind_power_pct = min(100, max(0, (wind_speed / 15) ** 3 * 30))
    solar_cf_pct = min(100, max(0, solar_rad / 1000 * (1 - cloud / 100) * 100))

    return {
        "region": region,
        "duration_hours": duration_hours,
        "weather_overrides": weather_overrides,
        "baseline": np.round(baseline, 2).tolist(),
        "scenario": np.round(scenario_forecast, 2).tolist(),
        "delta_mw": np.round(delta, 2).tolist(),
        "delta_pct": float(np.round(
            np.mean(delta) / np.mean(baseline) * 100, 2
        )) if np.mean(baseline) != 0 else 0.0,
        "pricing": {
            "base_avg": float(np.round(np.mean(np.atleast_1d(base_prices)), 2)),
            "scenario_avg": float(np.round(np.mean(np.atleast_1d(scenario_prices)), 2)),
            "delta": float(np.round(
                np.mean(np.atleast_1d(scenario_prices))
                - np.mean(np.atleast_1d(base_prices)), 2
            )),
        },
        "reserve_margin": {
            "min_pct": float(np.round(np.min(np.atleast_1d(scenario_reserve)), 2)),
            "avg_pct": float(np.round(np.mean(np.atleast_1d(scenario_reserve)), 2)),
            "status": (
                "CRITICAL" if np.min(np.atleast_1d(scenario_reserve)) < 5
                else "Low" if np.min(np.atleast_1d(scenario_reserve)) < 15
                else "Adequate"
            ),
        },
        "renewable_impact": {
            "wind_power_pct": round(wind_power_pct, 1),
            "solar_cf_pct": round(solar_cf_pct, 1),
        },
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


def _manual_scenario(
    future_df: pd.DataFrame,
    weather_overrides: dict[str, float],
    xgb_result: dict,
    baseline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fallback: manually apply weather overrides if v1's simulate_scenario fails.

    Copies features, overrides weather columns, re-scores with XGBoost.
    """
    from models.xgboost_model import predict_xgboost
    from data.feature_engineering import engineer_features

    scenario_df = future_df.copy()

    # Override raw weather columns
    for col, val in weather_overrides.items():
        if col in scenario_df.columns:
            scenario_df[col] = val

    # Recompute derived features
    try:
        scenario_df = engineer_features(scenario_df)
    except Exception:
        pass  # Use overrides as-is if re-engineering fails

    scenario_preds = predict_xgboost(xgb_result, scenario_df)
    delta = scenario_preds - baseline
    return scenario_preds, delta
