"""
Forecast explainability: per-point feature attribution for hover tooltips (NEXD-13).

Provides human-readable labels for all model features and functions to extract
top drivers from SHAP values (XGBoost) or global feature importances (fallback).

Pure logic — no Dash or I/O dependencies.
"""

from __future__ import annotations

import numpy as np
import structlog

log = structlog.get_logger()

# ── Human-readable labels for model features ─────────────────────────

FEATURE_LABELS: dict[str, str] = {
    # Raw weather (17)
    "temperature_2m": "Temperature",
    "apparent_temperature": "Feels-Like Temp",
    "relative_humidity_2m": "Humidity",
    "dew_point_2m": "Dew Point",
    "wind_speed_10m": "Wind (10m)",
    "wind_speed_80m": "Wind Speed",
    "wind_speed_120m": "Wind (120m)",
    "wind_direction_10m": "Wind Direction",
    "shortwave_radiation": "Solar Radiation",
    "direct_normal_irradiance": "Direct Irradiance",
    "diffuse_radiation": "Diffuse Radiation",
    "cloud_cover": "Cloud Cover",
    "precipitation": "Precipitation",
    "snowfall": "Snowfall",
    "surface_pressure": "Pressure",
    "soil_temperature_0cm": "Soil Temp",
    "weather_code": "Weather Code",
    # Derived exogenous (9)
    "cooling_degree_days": "Cooling Demand",
    "heating_degree_days": "Heating Demand",
    "temperature_deviation": "Temp Anomaly",
    "wind_power_estimate": "Wind Power",
    "solar_capacity_factor": "Solar Factor",
    "hour_sin": "Time of Day (sin)",
    "hour_cos": "Time of Day (cos)",
    "dow_sin": "Day of Week (sin)",
    "dow_cos": "Day of Week (cos)",
    "is_holiday": "Holiday",
    "temp_x_hour": "Temp x Hour",
    # Autoregressive demand (20)
    "demand_lag_1h": "Demand (1h ago)",
    "demand_lag_3h": "Demand (3h ago)",
    "demand_lag_24h": "Demand (24h ago)",
    "demand_lag_168h": "Demand (7d ago)",
    "ramp_rate": "Ramp Rate",
    "demand_momentum_short": "Short Momentum",
    "demand_momentum_long": "Long Momentum",
    "demand_ratio_24h": "24h Demand Ratio",
    "demand_ratio_168h": "7d Demand Ratio",
    "demand_roll_24h_mean": "Avg Demand (24h)",
    "demand_roll_24h_std": "Demand Volatility (24h)",
    "demand_roll_24h_min": "Min Demand (24h)",
    "demand_roll_24h_max": "Max Demand (24h)",
    "demand_roll_72h_mean": "Avg Demand (3d)",
    "demand_roll_72h_std": "Demand Volatility (3d)",
    "demand_roll_72h_min": "Min Demand (3d)",
    "demand_roll_72h_max": "Max Demand (3d)",
    "demand_roll_168h_mean": "Avg Demand (7d)",
    "demand_roll_168h_std": "Demand Volatility (7d)",
    "demand_roll_168h_min": "Min Demand (7d)",
    "demand_roll_168h_max": "Max Demand (7d)",
    # Extra time features from _create_future_features
    "hour": "Hour of Day",
    "day_of_week": "Day of Week",
    "month": "Month",
    "day_of_year": "Day of Year",
    "is_weekend": "Weekend",
}


def _label(feature_name: str) -> str:
    """Look up human label for a feature, falling back to title-cased name."""
    return FEATURE_LABELS.get(feature_name, feature_name.replace("_", " ").title())


def get_top_drivers_shap(
    shap_values: np.ndarray,
    feature_names: list[str],
    index: int,
    top_n: int = 3,
) -> list[tuple[str, float]]:
    """Return top N (label, shap_value_mw) for a single forecast point.

    Args:
        shap_values: Shape (n_points, n_features).
        feature_names: Feature column names matching axis 1.
        index: Row index of the forecast point.
        top_n: Number of top drivers to return.

    Returns:
        List of (human_label, shap_mw) sorted by absolute impact descending.
    """
    row = shap_values[index]
    abs_vals = np.abs(row)
    top_indices = np.argsort(abs_vals)[::-1][:top_n]
    return [(_label(feature_names[i]), float(row[i])) for i in top_indices]


def get_top_drivers_global(
    model_dict: dict,
    top_n: int = 3,
) -> list[tuple[str, float]]:
    """Return top N (label, importance) from XGBoost feature importances.

    Args:
        model_dict: Output from train_xgboost() containing 'model' and 'feature_names'.
        top_n: Number of top features to return.

    Returns:
        List of (human_label, importance) sorted descending.
    """
    model = model_dict["model"]
    feature_names = model_dict["feature_names"]
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:top_n]
    return [(_label(feature_names[i]), float(importances[i])) for i in top_indices]


def format_driver_line(label: str, value: float, mode: str = "shap") -> str:
    """Format a single driver for tooltip display.

    Args:
        label: Human-readable feature name.
        value: SHAP value in MW (mode="shap") or importance score (mode="global").
        mode: "shap" for per-point MW values, "global" for importance ranking.

    Returns:
        Formatted string like "Temperature: +1,200 MW" or "Temperature (high)".
    """
    if mode == "shap":
        sign = "+" if value >= 0 else ""
        return f"{label}: {sign}{value:,.0f} MW"
    return f"{label} (high)" if value > 0 else label


def build_tooltip_strings(
    shap_values: np.ndarray | None,
    feature_names: list[str] | None,
    model_dict: dict | None,
    n_points: int,
    model_name: str,
    top_n: int = 3,
) -> list[str]:
    """Build a list of tooltip strings, one per forecast point.

    XGBoost with SHAP data → per-point drivers with MW values.
    Other models with model_dict → static global importance repeated.
    Fallback → empty strings (no extra tooltip content).

    Args:
        shap_values: SHAP values array (n_points, n_features) or None.
        feature_names: Feature names matching SHAP columns.
        model_dict: XGBoost model dict for global importance fallback.
        n_points: Number of forecast points.
        model_name: Model name string.
        top_n: Number of top drivers per point.

    Returns:
        List of n_points tooltip strings.
    """
    empty = [""] * n_points

    # XGBoost with per-point SHAP
    if (
        model_name == "xgboost"
        and shap_values is not None
        and feature_names is not None
        and len(shap_values) >= n_points
    ):
        tooltips = []
        for i in range(n_points):
            drivers = get_top_drivers_shap(shap_values, feature_names, i, top_n)
            lines = [format_driver_line(lbl, val, mode="shap") for lbl, val in drivers]
            tooltips.append("<br>".join(lines))
        log.info("tooltips_built", mode="shap", n_points=n_points)
        return tooltips

    # Global importance fallback (any model, if we have an XGBoost model_dict)
    if model_dict is not None:
        try:
            drivers = get_top_drivers_global(model_dict, top_n)
            lines = [format_driver_line(lbl, val, mode="global") for lbl, val in drivers]
            static_tip = "<br>".join(lines)
            log.info("tooltips_built", mode="global", n_points=n_points)
            return [static_tip] * n_points
        except Exception:
            log.debug("tooltip_global_fallback_failed")
            return empty

    return empty
