"""
Model evaluation metrics.

Per spec AC-4.1 through AC-4.6:
- MAPE: Mean Absolute Percentage Error (handles zero actuals)
- RMSE: Root Mean Squared Error
- MAE: Mean Absolute Error
- R²: Coefficient of determination
- Residual analysis helpers
"""

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger()


def compute_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.

    MAPE = mean(|actual - predicted| / |actual|) × 100

    Zero actuals are excluded to avoid division by zero (per AC-4.1).

    Args:
        actual: Array of actual values.
        predicted: Array of predicted values.

    Returns:
        MAPE as percentage (e.g., 3.5 means 3.5%).
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mask = np.abs(actual) > 1e-6
    if not mask.any():
        log.warning("mape_all_zeros")
        return float("inf")

    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    RMSE = sqrt(mean((actual - predicted)²))
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Error.

    MAE = mean(|actual - predicted|)
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def compute_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    R² (Coefficient of Determination).

    R² = 1 - SS_res / SS_tot

    Perfect forecast → R² = 1.0.
    Mean forecast → R² = 0.0.
    Worse than mean → R² < 0.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)

    if ss_tot < 1e-10:
        return 0.0

    return float(1 - ss_res / ss_tot)


def compute_all_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """
    Compute all four metrics in one call.

    Returns:
        Dict with keys: mape, rmse, mae, r2.
    """
    return {
        "mape": compute_mape(actual, predicted),
        "rmse": compute_rmse(actual, predicted),
        "mae": compute_mae(actual, predicted),
        "r2": compute_r2(actual, predicted),
    }


def compute_residuals(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    Compute residuals: actual - predicted.

    Per AC-4.5.
    """
    return np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)


def compute_error_by_hour(
    timestamps: pd.Series | pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> pd.DataFrame:
    """
    Aggregate absolute errors by hour of day.

    Used for Tab 3 heatmap (AC-4.6).

    Returns:
        DataFrame with columns: [hour, mean_abs_error, std_abs_error, count].
    """
    abs_errors = np.abs(np.asarray(actual) - np.asarray(predicted))
    # Handle both Series and DatetimeIndex
    hours = timestamps.hour if isinstance(timestamps, pd.DatetimeIndex) else timestamps.dt.hour
    df = pd.DataFrame(
        {
            "hour": hours,
            "abs_error": abs_errors,
        }
    )
    grouped = df.groupby("hour")["abs_error"].agg(["mean", "std", "count"])
    grouped = grouped.rename(columns={"mean": "mean_abs_error", "std": "std_abs_error"})
    return grouped.reset_index()


def empirical_error_quantiles(
    residuals: np.ndarray,
    lower_q: float = 0.10,
    upper_q: float = 0.90,
) -> dict[str, float]:
    """
    Estimate empirical forecast-error quantiles from residuals.

    Residuals are expected to be ``actual - predicted`` so interval bounds for
    predictions are derived as:
      lower = pred + q_lower
      upper = pred + q_upper
    """
    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"lower_error": 0.0, "upper_error": 0.0, "sample_size": 0}
    return {
        "lower_error": float(np.quantile(values, lower_q)),
        "upper_error": float(np.quantile(values, upper_q)),
        "sample_size": int(values.size),
    }


def apply_empirical_interval(
    predicted: np.ndarray,
    lower_error: float,
    upper_error: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply empirical residual quantiles around point forecasts."""
    pred = np.asarray(predicted, dtype=float)
    return pred + lower_error, pred + upper_error


def compute_interval_coverage(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Compute interval coverage rate in [0, 1]."""
    y = np.asarray(actual, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if y.size == 0:
        return 0.0
    inside = (y >= lo) & (y <= hi)
    return float(np.mean(inside))


def compute_interval_coverage_drift(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_coverage: float = 0.80,
    window_size: int = 168,
) -> dict[str, float]:
    """
    Monitor prediction-interval coverage drift over time.

    Returns both overall and recent-window drift (actual minus target).
    """
    y = np.asarray(actual, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    n = int(min(y.size, lo.size, hi.size))
    if n == 0:
        return {"overall_coverage": 0.0, "recent_coverage": 0.0, "drift": -target_coverage}
    y, lo, hi = y[:n], lo[:n], hi[:n]
    overall_cov = compute_interval_coverage(y, lo, hi)
    recent_n = max(1, min(window_size, n))
    recent_cov = compute_interval_coverage(y[-recent_n:], lo[-recent_n:], hi[-recent_n:])
    return {
        "overall_coverage": overall_cov,
        "recent_coverage": recent_cov,
        "drift": recent_cov - target_coverage,
    }
