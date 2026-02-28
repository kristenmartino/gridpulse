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

    Returns:
        Unfitted Prophet model with regressors attached.
    """
    Prophet = _get_prophet()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_mode="multiplicative",
    )

    for regressor_name, mode in PROPHET_REGRESSORS:
        model.add_regressor(regressor_name, mode=mode)

    log.debug("prophet_model_created", regressors=len(PROPHET_REGRESSORS))
    return model


def train_prophet(
    df: pd.DataFrame,
    target_col: str = "demand_mw",
) -> Any:
    """
    Train a Prophet model on the given DataFrame.

    Args:
        df: Feature-engineered DataFrame with 'timestamp' and target column.
        target_col: Column to forecast (default: demand_mw).

    Returns:
        Fitted Prophet model.
    """
    model = create_prophet_model()

    # Prophet expects 'ds' and 'y' columns
    train_df = pd.DataFrame({
        "ds": df["timestamp"].dt.tz_localize(None) if df["timestamp"].dt.tz else df["timestamp"],
        "y": df[target_col],
    })

    # Add regressor columns
    for regressor_name, _ in PROPHET_REGRESSORS:
        if regressor_name in df.columns:
            train_df[regressor_name] = df[regressor_name].values
        else:
            log.warning("prophet_missing_regressor", regressor=regressor_name)
            train_df[regressor_name] = 0.0

    model.fit(train_df)
    log.info("prophet_trained", rows=len(train_df))
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

    # Add regressor values to future DataFrame
    for regressor_name, _ in PROPHET_REGRESSORS:
        if regressor_name in df.columns:
            # Use available values, forward-fill for forecast period
            available = df[regressor_name].values
            if len(available) >= len(future):
                future[regressor_name] = available[: len(future)]
            else:
                padded = np.concatenate([
                    available,
                    np.full(len(future) - len(available), available[-1] if len(available) > 0 else 0),
                ])
                future[regressor_name] = padded
        else:
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
