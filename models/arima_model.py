"""
SARIMAX forecasting model with exogenous weather variables.

Classical statistical baseline per spec §Model 2:
- SARIMAX with exogenous weather features
- Auto-order selection via pmdarima
- Seasonal order with D=1 for daily seasonality (m=24)
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
# D=1 (seasonal differencing) is critical to prevent forecast drift
DEFAULT_ORDER = (2, 1, 2)
DEFAULT_SEASONAL_ORDER = (1, 1, 1, 24)

# Tail length (hours) kept with the pickled payload so SARIMAX can be
# reconstructed and its Kalman filter initialized at predict time. 10 full
# seasonal cycles at m=24 is more than enough for stable state initialization.
PICKLE_TAIL_ROWS = 240


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

    # Fill any NaN in exog to prevent SARIMAX fitting failures
    if exog is not None:
        nan_mask = np.isnan(exog)
        if nan_mask.any():
            log.warning("arima_exog_nan", nan_count=int(nan_mask.sum()))
            # Column-wise forward/backward fill, then zero fill
            for col_idx in range(exog.shape[1]):
                col = exog[:, col_idx]
                mask = np.isnan(col)
                if mask.any():
                    # Forward fill
                    for i in range(1, len(col)):
                        if np.isnan(col[i]):
                            col[i] = col[i - 1]
                    # Backward fill remaining
                    for i in range(len(col) - 2, -1, -1):
                        if np.isnan(col[i]):
                            col[i] = col[i + 1]
                    # Zero fill if still NaN (all-NaN column)
                    col[np.isnan(col)] = 0

    order = DEFAULT_ORDER
    seasonal_order = DEFAULT_SEASONAL_ORDER

    if auto_order:
        order, seasonal_order = _auto_select_order(y, exog)

    # Enforce seasonal differencing (D>=1) to prevent forecast drift.
    # Without D=1, ARIMA's integrated component causes predictions to
    # diverge from the daily cycle, especially beyond a few hours.
    if seasonal_order[1] == 0 and seasonal_order[3] >= 24:
        log.info(
            "arima_enforcing_seasonal_diff",
            original_seasonal=seasonal_order,
        )
        seasonal_order = (seasonal_order[0], 1, seasonal_order[2], seasonal_order[3])

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

        # Validate: check last 24h in-sample residuals for drift
        resid = fitted.resid[-24:]
        drift = np.mean(resid[-12:]) - np.mean(resid[:12])
        if abs(drift) > np.std(y) * 0.5:
            log.warning(
                "arima_drift_detected",
                drift=round(drift, 1),
                std=round(np.std(y), 1),
                fallback="default_seasonal_order",
            )
            # Re-fit with stronger default seasonal order
            model = SARIMAX(
                y,
                exog=exog,
                order=DEFAULT_ORDER,
                seasonal_order=DEFAULT_SEASONAL_ORDER,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False, maxiter=200)
            order = DEFAULT_ORDER
            seasonal_order = DEFAULT_SEASONAL_ORDER
            log.info("arima_retrained_default", aic=round(fitted.aic, 1))

    except Exception as e:
        log.error("arima_training_failed", error=str(e), fallback="default_order")
        # Fallback to simpler model — still keep D=1
        try:
            model = SARIMAX(
                y,
                exog=exog,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 24),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False, maxiter=100)
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, 24)
        except Exception as e2:
            log.error("arima_fallback_failed", error=str(e2), fallback="no_seasonal")
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

    # Lean payload: save only the fitted params and the tail of training data
    # needed to re-initialize SARIMAX at predict time. A full SARIMAXResults
    # object pickles at ~500 MB because it retains Kalman filter/smoother
    # state-covariance matrices; the lean form is a few kilobytes.
    tail_len = min(len(y), PICKLE_TAIL_ROWS)
    tail_y = np.asarray(y[-tail_len:], dtype=np.float32)
    tail_exog = np.asarray(exog[-tail_len:], dtype=np.float32) if exog is not None else None
    params = np.asarray(fitted.params, dtype=np.float64)
    log.info("arima_payload_lean", param_count=len(params), tail_rows=tail_len)

    return {
        "params": params,
        "order": order,
        "seasonal_order": seasonal_order,
        "exog_cols": ARIMA_EXOG_COLS,
        "tail_y": tail_y,
        "tail_exog": tail_exog,
    }


def predict_arima(
    model_dict: dict[str, Any],
    future_exog: pd.DataFrame,
    periods: int = 168,
) -> np.ndarray:
    """
    Generate forecasts from a SARIMAX model payload.

    Supports the lean payload produced by the current ``train_arima``
    (``params`` + ``tail_y`` / ``tail_exog``) as well as legacy payloads that
    stored a fitted ``SARIMAXResults`` under ``"model"`` for one roll-forward
    cycle of backward compatibility.

    Args:
        model_dict: Output from train_arima() (lean) or a legacy dict with
            a fitted ``SARIMAXResults`` under ``"model"``.
        future_exog: DataFrame with exogenous variable values for forecast period.
        periods: Number of hourly steps to forecast.

    Returns:
        Forecast array of length `periods`.
    """
    exog = _get_exog(future_exog, n_rows=periods)

    try:
        if "params" in model_dict:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            reconstructed = SARIMAX(
                model_dict["tail_y"],
                exog=model_dict.get("tail_exog"),
                order=model_dict["order"],
                seasonal_order=model_dict["seasonal_order"],
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = reconstructed.filter(model_dict["params"])
        else:
            fitted = model_dict["model"]

        forecast = fitted.forecast(steps=periods, exog=exog)
        # Clamp negative forecasts to 0 (demand can't be negative)
        forecast = np.maximum(forecast, 0)
        log.debug("arima_forecast_generated", periods=periods)
        return forecast
    except Exception as e:
        log.error("arima_forecast_failed", error=str(e))
        return np.full(periods, np.nan)


def _get_exog(df: pd.DataFrame, n_rows: int | None = None) -> np.ndarray | None:
    """Extract exogenous variables matrix from DataFrame.

    Fills NaN values to prevent SARIMAX forecast failures.
    """
    available = [c for c in ARIMA_EXOG_COLS if c in df.columns]
    if not available:
        return None

    exog = df[available].values.copy()

    # Fill NaN in exog — SARIMAX cannot handle NaN in exogenous variables
    if np.isnan(exog).any():
        for col_idx in range(exog.shape[1]):
            col = exog[:, col_idx]
            mask = np.isnan(col)
            if mask.any():
                # Forward fill, then backward fill, then zero
                for i in range(1, len(col)):
                    if np.isnan(col[i]):
                        col[i] = col[i - 1]
                for i in range(len(col) - 2, -1, -1):
                    if np.isnan(col[i]):
                        col[i] = col[i + 1]
                col[np.isnan(col)] = 0

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
    """Use pmdarima auto_arima for order selection.

    Uses 1440 rows (60 days) for at least 2.5 full seasonal cycles at m=24.
    Forces D=1 (seasonal differencing) to prevent forecast drift.
    """
    try:
        import pmdarima as pm

        # 30 days = 720 hours for order selection (D=1 is forced, so
        # auto_arima only needs to select p,d,q,P,Q — fewer cycles needed)
        subset_size = 720
        y_sub = y[-subset_size:] if len(y) > subset_size else y
        exog_sub = exog[-subset_size:] if exog is not None and len(exog) > subset_size else exog

        auto = pm.auto_arima(
            y_sub,
            exogenous=exog_sub,
            seasonal=True,
            m=24,
            max_p=2,
            max_q=2,
            max_P=1,
            max_Q=1,
            max_d=1,
            max_D=1,
            start_D=1,  # Start with seasonal differencing
            D=1,  # Force D=1 to prevent drift
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            n_fits=20,
        )
        order = auto.order
        seasonal_order = auto.seasonal_order
        log.info("arima_auto_order", order=order, seasonal_order=seasonal_order)
        return order, seasonal_order

    except Exception as e:
        log.warning("arima_auto_order_failed", error=str(e), fallback="default")
        return DEFAULT_ORDER, DEFAULT_SEASONAL_ORDER
