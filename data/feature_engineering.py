"""
Feature engineering for energy demand forecasting.

Computes 25+ derived features from raw demand and weather data.
All features are documented in project1-expanded-spec.md §Derived Features.

Key conventions:
- Temperature in Fahrenheit (CDD/HDD baseline = 65°F)
- Wind speed in mph (converted to m/s internally for power calculation)
- All features are numeric, no NaN in final output
- Backward-looking windows only (no future data leakage)
"""

import numpy as np
import pandas as pd
import structlog

from config import (
    CDD_HDD_BASELINE_F,
    MPH_TO_MS,
    WIND_CUTOUT_SPEED_MS,
    SOLAR_RATED_IRRADIANCE,
    AIR_DENSITY_KG_M3,
)

log = structlog.get_logger()

# US federal holidays
try:
    import holidays as holidays_lib
    US_HOLIDAYS = holidays_lib.US()
except ImportError:
    US_HOLIDAYS = {}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all derived features from a merged demand + weather DataFrame.

    Input must have columns: timestamp, demand_mw, and the 17 weather variables.
    Output adds 20+ new columns and drops rows with NaN.

    Args:
        df: Merged DataFrame from preprocessing.merge_demand_weather().

    Returns:
        Feature-engineered DataFrame ready for model training.
    """
    if df.empty:
        log.warning("feature_engineering_empty_input")
        return df

    df = df.copy()
    log.info("feature_engineering_start", input_rows=len(df))

    # --- Temperature-based features ---
    if "temperature_2m" in df.columns:
        df["cooling_degree_days"] = compute_cdd(df["temperature_2m"])
        df["heating_degree_days"] = compute_hdd(df["temperature_2m"])
        df["temperature_deviation"] = compute_temperature_deviation(df["temperature_2m"])

    # --- Wind power estimate ---
    if "wind_speed_80m" in df.columns:
        df["wind_power_estimate"] = compute_wind_power(df["wind_speed_80m"])

    # --- Solar capacity factor ---
    if "shortwave_radiation" in df.columns:
        df["solar_capacity_factor"] = compute_solar_capacity_factor(df["shortwave_radiation"])

    # --- Cyclical time features ---
    if "timestamp" in df.columns:
        df["hour_sin"], df["hour_cos"] = compute_cyclical_hour(df["timestamp"])
        df["dow_sin"], df["dow_cos"] = compute_cyclical_dow(df["timestamp"])
        df["is_holiday"] = compute_holiday_flag(df["timestamp"])

    # --- Lag features (demand) ---
    if "demand_mw" in df.columns:
        df["demand_lag_24h"] = compute_lag(df["demand_mw"], periods=24)
        df["demand_lag_168h"] = compute_lag(df["demand_mw"], periods=168)
        df["ramp_rate"] = compute_ramp_rate(df["demand_mw"])

    # --- Rolling statistics (demand, backward-looking) ---
    if "demand_mw" in df.columns:
        for window in [24, 72, 168]:
            prefix = f"demand_roll_{window}h"
            rolling = df["demand_mw"].rolling(window=window, min_periods=1)
            df[f"{prefix}_mean"] = rolling.mean()
            df[f"{prefix}_std"] = rolling.std()
            df[f"{prefix}_min"] = rolling.min()
            df[f"{prefix}_max"] = rolling.max()

    # --- Interaction terms ---
    if "temperature_2m" in df.columns and "hour_sin" in df.columns:
        df["temp_x_hour"] = compute_temp_hour_interaction(
            df["temperature_2m"], df["hour_sin"]
        )

    # --- Drop rows with NaN (from lag/rolling features at the start) ---
    initial_rows = len(df)
    df = df.dropna(subset=_get_feature_columns(df)).reset_index(drop=True)
    dropped = initial_rows - len(df)

    log.info(
        "feature_engineering_complete",
        output_rows=len(df),
        dropped_rows=dropped,
        feature_count=len(_get_feature_columns(df)),
    )

    return df


# ---------------------------------------------------------------------------
# Individual feature functions (public, used by scenario engine)
# ---------------------------------------------------------------------------


def compute_cdd(temperature_f: pd.Series) -> pd.Series:
    """
    Cooling Degree Days: max(0, temp - 65°F).

    Standard HVAC demand proxy. Higher CDD = more AC load.

    Args:
        temperature_f: Temperature in Fahrenheit.

    Returns:
        CDD series.
    """
    return np.maximum(0, temperature_f - CDD_HDD_BASELINE_F)


def compute_hdd(temperature_f: pd.Series) -> pd.Series:
    """
    Heating Degree Days: max(0, 65°F - temp).

    Standard winter heating demand proxy.

    Args:
        temperature_f: Temperature in Fahrenheit.

    Returns:
        HDD series.
    """
    return np.maximum(0, CDD_HDD_BASELINE_F - temperature_f)


def compute_temperature_deviation(temperature_f: pd.Series, window: int = 720) -> pd.Series:
    """
    Temperature deviation from 30-day (720-hour) rolling average.

    Unusual weather = unusual demand.

    Args:
        temperature_f: Temperature in Fahrenheit.
        window: Rolling window in hours (default: 720 = 30 days).

    Returns:
        Deviation series.
    """
    rolling_avg = temperature_f.rolling(window=window, min_periods=1).mean()
    return temperature_f - rolling_avg


def compute_wind_power(wind_speed_mph: pd.Series) -> pd.Series:
    """
    Simplified wind power estimate: 0.5 × ρ × A × v³.

    Converts mph → m/s internally. Applies cutout speed (25 m/s ≈ 56 mph).
    Above cutout, turbines shut down → power = 0.

    Args:
        wind_speed_mph: Wind speed in mph (from Open-Meteo).

    Returns:
        Normalized wind power estimate [0, 1].
    """
    # Convert mph to m/s
    v_ms = wind_speed_mph * MPH_TO_MS

    # Simplified power curve (normalized)
    # P = 0.5 * rho * A * v^3 — we normalize by rated conditions
    # Using v=12 m/s as rated speed (typical for modern turbines)
    rated_speed_ms = 12.0
    rated_power = 0.5 * AIR_DENSITY_KG_M3 * 1.0 * (rated_speed_ms ** 3)

    raw_power = 0.5 * AIR_DENSITY_KG_M3 * 1.0 * (v_ms ** 3)
    normalized = raw_power / rated_power

    # Apply cut-in (3 m/s) and cutout (25 m/s) speeds
    cut_in_ms = 3.0
    result = np.where(
        v_ms < cut_in_ms, 0.0,
        np.where(v_ms > WIND_CUTOUT_SPEED_MS, 0.0, np.minimum(normalized, 1.0))
    )

    return pd.Series(result, index=wind_speed_mph.index, dtype=float)


def compute_solar_capacity_factor(ghi: pd.Series) -> pd.Series:
    """
    Solar capacity factor: GHI / 1000 W/m², clipped to [0, 1].

    Solar panels rated at standard test conditions (1000 W/m²).

    Args:
        ghi: Global Horizontal Irradiance (shortwave_radiation) in W/m².

    Returns:
        Capacity factor [0, 1].
    """
    return np.clip(ghi / SOLAR_RATED_IRRADIANCE, 0.0, 1.0)


def compute_cyclical_hour(timestamps: pd.Series | pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """
    Cyclical sin/cos encoding for hour of day.

    hour=0 and hour=24 map to the same point on the unit circle.

    Returns:
        (hour_sin, hour_cos) tuple of Series.
    """
    # Handle both Series and DatetimeIndex
    if isinstance(timestamps, pd.DatetimeIndex):
        hour = timestamps.hour
        index = timestamps
    else:
        hour = timestamps.dt.hour
        index = timestamps.index
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    return (
        pd.Series(hour_sin, index=index, dtype=float),
        pd.Series(hour_cos, index=index, dtype=float),
    )


def compute_cyclical_dow(timestamps: pd.Series | pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """
    Cyclical sin/cos encoding for day of week.

    Monday=0, Sunday=6. Cyclical so Monday and next Monday are identical.

    Returns:
        (dow_sin, dow_cos) tuple of Series.
    """
    # Handle both Series and DatetimeIndex
    if isinstance(timestamps, pd.DatetimeIndex):
        dow = timestamps.dayofweek
        index = timestamps
    else:
        dow = timestamps.dt.dayofweek
        index = timestamps.index
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)
    return (
        pd.Series(dow_sin, index=index, dtype=float),
        pd.Series(dow_cos, index=index, dtype=float),
    )


def compute_holiday_flag(timestamps: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """
    Binary flag: 1 if US federal holiday, 0 otherwise.

    Uses the `holidays` library for accurate holiday detection.
    """
    # Handle both Series and DatetimeIndex
    if isinstance(timestamps, pd.DatetimeIndex):
        index = timestamps
        ts_iter = timestamps
    else:
        index = timestamps.index
        ts_iter = timestamps
    return pd.Series(
        [1.0 if ts.date() in US_HOLIDAYS else 0.0 for ts in ts_iter],
        index=index,
        dtype=float,
    )


def compute_lag(series: pd.Series, periods: int) -> pd.Series:
    """
    Compute lag feature (shift by N periods).

    Uses positive shift so lag_24 = value from 24 hours ago.
    No future data leakage — lagged values only look backward.
    """
    return series.shift(periods)


def compute_ramp_rate(demand: pd.Series) -> pd.Series:
    """
    Ramp rate: demand_t - demand_t-1.

    Critical for grid operations — high ramp rates require fast-responding
    generation (gas peakers).

    Known demand [100, 120, 110] → ramp [NaN, 20, -10].
    """
    return demand.diff()


def compute_temp_hour_interaction(temperature: pd.Series, hour_sin: pd.Series) -> pd.Series:
    """
    Temperature × Hour interaction term.

    Captures the pattern that AC peaks in the afternoon (high temp × high hour_sin)
    while heating peaks in the evening.
    """
    return temperature * hour_sin


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get list of numeric feature columns (excludes timestamp, metadata)."""
    exclude = {"timestamp", "region", "data_quality", "forecast_mw"}
    return [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in exclude
    ]


def get_feature_names() -> list[str]:
    """
    Return the canonical list of feature names produced by engineer_features().

    Used by models for consistent feature ordering.
    """
    return [
        # Raw weather
        "temperature_2m", "apparent_temperature", "relative_humidity_2m",
        "dew_point_2m", "wind_speed_10m", "wind_speed_80m", "wind_speed_120m",
        "wind_direction_10m", "shortwave_radiation", "direct_normal_irradiance",
        "diffuse_radiation", "cloud_cover", "precipitation", "snowfall",
        "surface_pressure", "soil_temperature_0cm", "weather_code",
        # Derived
        "cooling_degree_days", "heating_degree_days", "temperature_deviation",
        "wind_power_estimate", "solar_capacity_factor",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_holiday",
        "demand_lag_24h", "demand_lag_168h", "ramp_rate",
        "demand_roll_24h_mean", "demand_roll_24h_std",
        "demand_roll_24h_min", "demand_roll_24h_max",
        "demand_roll_72h_mean", "demand_roll_72h_std",
        "demand_roll_72h_min", "demand_roll_72h_max",
        "demand_roll_168h_mean", "demand_roll_168h_std",
        "demand_roll_168h_min", "demand_roll_168h_max",
        "temp_x_hour",
    ]
