"""
Prophet forecasting model with weather regressors.

Configured per spec §Model 1:
- Multiplicative seasonality (energy demand scales multiplicatively)
- Daily, weekly, yearly seasonality
- Weather regressors: temperature, apparent temp, wind, solar, CDD, HDD, holiday
- Rolling window training: 365 days → forecast 7 days
"""

from typing import Any

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger()

# Prophet is heavy — lazy import to speed up module load
_Prophet = None


def _get_prophet():
    global _Prophet
    if _Prophet is None:
        from prophet import Prophet

        _Prophet = Prophet
    return _Prophet


# Regressors added to Prophet — must exist in the feature DataFrame
PROPHET_REGRESSORS = [
    ("temperature_2m", "multiplicative"),
    ("apparent_temperature", "multiplicative"),
    ("wind_speed_80m", "additive"),
    ("shortwave_radiation", "additive"),
    ("cooling_degree_days", "multiplicative"),
    ("heating_degree_days", "multiplicative"),
    ("is_holiday", "multiplicative"),
]


def create_prophet_model() -> Any:
    """
    Create a configured Prophet model instance.

    Uses logistic growth with floor=0 to structurally prevent negative
    forecasts. Tight changepoint_prior_scale (0.001) prevents trend
    extrapolation from drifting away at long horizons (7-30 days).

    Returns:
        Unfitted Prophet model with regressors attached.
    """
    ProphetClass = _get_prophet()  # noqa: N806

    model = ProphetClass(
        growth="logistic",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.001,
        seasonality_mode="additive",
    )

    for regressor_name, _mode in PROPHET_REGRESSORS:
        # Use additive mode for all regressors to match the global setting.
        # Multiplicative regressors on a logistic growth trend can cause
        # erratic extrapolation beyond the training data range.
        model.add_regressor(regressor_name, mode="additive")

    log.debug("prophet_model_created", regressors=len(PROPHET_REGRESSORS))
    return model


def train_prophet(
    df: pd.DataFrame,
    target_col: str = "demand_mw",
) -> Any:
    """
    Train a Prophet model on the given DataFrame.

    Uses logistic growth: floor=0 prevents negative predictions,
    cap is set to 1.5x historical max demand to bound extrapolation.

    Args:
        df: Feature-engineered DataFrame with 'timestamp' and target column.
        target_col: Column to forecast (default: demand_mw).

    Returns:
        Fitted Prophet model (with _demand_cap attribute for predict).
    """
    model = create_prophet_model()

    # Prophet expects 'ds' and 'y' columns
    train_df = pd.DataFrame(
        {
            "ds": df["timestamp"].dt.tz_localize(None)
            if df["timestamp"].dt.tz
            else df["timestamp"],
            "y": df[target_col],
        }
    )

    # Logistic growth requires 'cap' and 'floor' columns
    demand_cap = float(df[target_col].max() * 1.5)
    train_df["cap"] = demand_cap
    train_df["floor"] = 0

    # Add regressor columns
    for regressor_name, _ in PROPHET_REGRESSORS:
        if regressor_name in df.columns:
            train_df[regressor_name] = df[regressor_name].values
        else:
            log.warning("prophet_missing_regressor", regressor=regressor_name)
            train_df[regressor_name] = 0.0

    model.fit(train_df)
    # Store cap for use in predict_prophet
    model._demand_cap = demand_cap
    log.info("prophet_trained", rows=len(train_df), demand_cap=round(demand_cap, 0))
    return model


def predict_prophet(
    model: Any,
    df: pd.DataFrame,
    periods: int = 168,
) -> dict[str, np.ndarray]:
    """
    Generate forecasts from a fitted Prophet model.

    Args:
        model: Fitted Prophet model.
        df: DataFrame with future timestamps and regressor values.
        periods: Number of hourly periods to forecast (default: 168 = 7 days).

    Returns:
        Dict with keys: 'forecast', 'lower_80', 'upper_80', 'lower_95', 'upper_95'
    """
    future = model.make_future_dataframe(periods=periods, freq="h")

    # Logistic growth requires cap and floor on future DataFrame
    demand_cap = getattr(model, "_demand_cap", 50000)
    future["cap"] = demand_cap
    future["floor"] = 0

    # Add regressor values to future DataFrame via timestamp join.
    # The positional approach (slicing by index) silently misaligns regressors
    # when df and future have different lengths or time ranges.  Joining on
    # the canonical 'ds' timestamp column guarantees each hour gets its own
    # regressor values, with forward-fill covering the forecast horizon.
    regressor_cols = [name for name, _ in PROPHET_REGRESSORS if name in df.columns]
    if regressor_cols:
        # Build a tz-naive timestamp key that matches Prophet's 'ds' column
        if "timestamp" in df.columns:
            ts = df["timestamp"].dt.tz_localize(None) if df["timestamp"].dt.tz else df["timestamp"]
        elif "ds" in df.columns:
            ts = pd.to_datetime(df["ds"]).dt.tz_localize(None) if pd.to_datetime(df["ds"]).dt.tz else pd.to_datetime(df["ds"])
        else:
            ts = df.index.tz_localize(None) if hasattr(df.index, "tz") and df.index.tz else df.index

        reg_df = pd.DataFrame({"ds": ts.values})
        for col in regressor_cols:
            reg_df[col] = df[col].values

        # Drop duplicates (keep last) so the merge is clean
        reg_df = reg_df.drop_duplicates(subset="ds", keep="last")

        future = future.merge(reg_df, on="ds", how="left")

        # Forward-fill covers the forecast horizon (beyond training data);
        # backward-fill handles any leading gaps; zero fills truly missing regressors.
        for col in regressor_cols:
            future[col] = future[col].ffill().bfill().fillna(0)

    # Fill any regressors not present in df at all
    for regressor_name, _ in PROPHET_REGRESSORS:
        if regressor_name not in future.columns:
            future[regressor_name] = 0.0

    forecast = model.predict(future)

    # Extract the forecast period (last N rows)
    fc = forecast.tail(periods)

    return {
        "forecast": fc["yhat"].values,
        "lower_80": fc["yhat_lower"].values,
        "upper_80": fc["yhat_upper"].values,
        "lower_95": fc["yhat_lower"].values * 0.95,  # Approximate 95% from 80%
        "upper_95": fc["yhat_upper"].values * 1.05,
        "timestamps": pd.to_datetime(fc["ds"]).values,
    }
