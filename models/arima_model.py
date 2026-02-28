"""
SARIMAX forecasting model with exogenous weather variables.

Classical statistical baseline per spec §Model 2:
- SARIMAX with exogenous weather features
- Auto-order selection via pmdarima
- Seasonal order (1,1,1,24) for daily seasonality
"""

from typing import Any

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger()

# Exogenous variables for SARIMAX
ARIMA_EXOG_COLS = [
    "temperature_2m",
    "wind_speed_80m",
    "shortwave_radiation",
    "cooling_degree_days",
    "heating_degree_days",
]

# Default order if auto_arima is too slow or fails
DEFAULT_ORDER = (2, 1, 2)
DEFAULT_SEASONAL_ORDER = (1, 1, 1, 24)


def train_arima(
    df: pd.DataFrame,
    target_col: str = "demand_mw",
    auto_order: bool = True,
    max_training_rows: int = 2160,
) -> dict[str, Any]:
    """
    Train a SARIMAX model on the given DataFrame.

    Args:
        df: Feature-engineered DataFrame.
        target_col: Column to forecast.
        auto_order: Whether to use pmdarima for auto order selection.
        max_training_rows: Limit training data to avoid slow fitting (default: 90 days).

    Returns:
        Dict with 'model' (fitted), 'order', 'seasonal_order', 'exog_cols'.
    """
    # Limit training data for performance (ARIMA is O(n³))
    train_df = df.tail(max_training_rows).copy()
    y = train_df[target_col].values

    exog = _get_exog(train_df)

    order = DEFAULT_ORDER
    seasonal_order = DEFAULT_SEASONAL_ORDER

    if auto_order:
        order, seasonal_order = _auto_select_order(y, exog)

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    log.info("arima_training", order=order, seasonal_order=seasonal_order, rows=len(y))

    try:
        model = SARIMAX(
            y,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=200)
        log.info("arima_trained", aic=round(fitted.aic, 1))
    except Exception as e:
        log.error("arima_training_failed", error=str(e), fallback="default_order")
        # Fallback to simpler model
        model = SARIMAX(
            y,
            exog=exog,
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=100)
        order = (1, 1, 1)
        seasonal_order = (0, 0, 0, 0)

    return {
        "model": fitted,
        "order": order,
        "seasonal_order": seasonal_order,
        "exog_cols": ARIMA_EXOG_COLS,
    }


def predict_arima(
    model_dict: dict[str, Any],
    future_exog: pd.DataFrame,
    periods: int = 168,
) -> np.ndarray:
    """
    Generate forecasts from a fitted SARIMAX model.

    Args:
        model_dict: Output from train_arima().
        future_exog: DataFrame with exogenous variable values for forecast period.
        periods: Number of hourly steps to forecast.

    Returns:
        Forecast array of length `periods`.
    """
    fitted = model_dict["model"]
    exog = _get_exog(future_exog, n_rows=periods)

    try:
        forecast = fitted.forecast(steps=periods, exog=exog)
        # Clamp negative forecasts to 0 (demand can't be negative)
        forecast = np.maximum(forecast, 0)
        log.debug("arima_forecast_generated", periods=periods)
        return forecast
    except Exception as e:
        log.error("arima_forecast_failed", error=str(e))
        return np.full(periods, np.nan)


def _get_exog(df: pd.DataFrame, n_rows: int | None = None) -> np.ndarray | None:
    """Extract exogenous variables matrix from DataFrame."""
    available = [c for c in ARIMA_EXOG_COLS if c in df.columns]
    if not available:
        return None

    exog = df[available].values
    if n_rows is not None and len(exog) > n_rows:
        exog = exog[:n_rows]
    elif n_rows is not None and len(exog) < n_rows:
        # Pad with last row repeated
        pad = np.tile(exog[-1:], (n_rows - len(exog), 1))
        exog = np.vstack([exog, pad])

    return exog


def _auto_select_order(
    y: np.ndarray,
    exog: np.ndarray | None,
) -> tuple[tuple, tuple]:
    """Use pmdarima auto_arima for order selection."""
    try:
        import pmdarima as pm

        # Use a subset for speed (auto_arima can be very slow)
        y_sub = y[-720:] if len(y) > 720 else y  # Last 30 days
        exog_sub = exog[-720:] if exog is not None and len(exog) > 720 else exog

        auto = pm.auto_arima(
            y_sub,
            exogenous=exog_sub,
            seasonal=True,
            m=24,
            max_p=3,
            max_q=3,
            max_P=2,
            max_Q=2,
            max_d=2,
            max_D=1,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            n_fits=30,
        )
        order = auto.order
        seasonal_order = auto.seasonal_order
        log.info("arima_auto_order", order=order, seasonal_order=seasonal_order)
        return order, seasonal_order

    except Exception as e:
        log.warning("arima_auto_order_failed", error=str(e), fallback="default")
        return DEFAULT_ORDER, DEFAULT_SEASONAL_ORDER
