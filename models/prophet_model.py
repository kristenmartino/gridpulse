"""
Prophet forecasting model with weather regressors.

Configured per spec §Model 1:
- Additive seasonality (``seasonality_mode='additive'``); weather regressors added as additive too
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


# Prophet needs at least ~2 full cycles to *identify* a 365-day seasonality.
# Fit on less than this (our default training window is ~90 days) the yearly
# Fourier terms are underdetermined and extrapolate to large spurious swings a
# few weeks past the training end — the dominant #281 driver: on DUK's 30-day
# horizon the yearly component alone accounted for a −11.8 GW phantom decline
# (78% of the total), dragging Prophet's forecast below 0 MW. The weather
# regressors (temperature / CDD / HDD) already carry the annual demand signal,
# so yearly seasonality is redundant *and* overfit until the history is long
# enough to identify it. Below this span it is disabled.
YEARLY_SEASONALITY_MIN_DAYS = 730


def create_prophet_model(yearly_seasonality: bool = False) -> Any:
    """
    Create a configured Prophet model instance.

    Uses logistic growth with a demand floor/cap on the *trend*. Tight
    changepoint_prior_scale (0.001) keeps the trend near-flat at long horizons.

    Args:
        yearly_seasonality: Whether to fit a 365-day seasonal component. Off by
            default — only meaningful with ≥ ``YEARLY_SEASONALITY_MIN_DAYS`` of
            history (see #281); ``train_prophet`` sets it from the training span.

    Returns:
        Unfitted Prophet model with regressors attached.
    """
    ProphetClass = _get_prophet()  # noqa: N806

    model = ProphetClass(
        growth="logistic",
        yearly_seasonality=yearly_seasonality,
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

    Logistic growth bounds the TREND to ``[floor=0, cap=1.5×max]``. Note this
    does NOT guarantee a non-negative forecast — the additive seasonality and
    regressor terms sit on top of the trend and can push the composite ``yhat``
    below zero (#281); ``predict_prophet`` clips explicitly for that. Yearly
    seasonality is enabled only when the training span justifies it (see
    ``YEARLY_SEASONALITY_MIN_DAYS``).

    Args:
        df: Feature-engineered DataFrame with 'timestamp' and target column.
        target_col: Column to forecast (default: demand_mw).

    Returns:
        Fitted Prophet model (with _demand_cap attribute for predict).
    """
    # Enable yearly seasonality only when the training window actually spans
    # enough of the annual cycle to identify it (#281). Our ~90-day window does
    # not, so this is off in production and the annual signal comes from the
    # weather regressors instead.
    ts = df["timestamp"]
    span_days = int((ts.max() - ts.min()) / pd.Timedelta(days=1)) if len(ts) else 0
    enable_yearly = span_days >= YEARLY_SEASONALITY_MIN_DAYS
    model = create_prophet_model(yearly_seasonality=enable_yearly)

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

    # Add regressor columns. Prophet's fit rejects any NaN in a regressor
    # ("Found NaN in column 'wind_speed_80m'", #176) — an archive-unstable
    # weather column (#164) carries NaN in recent rows. Sanitize the same
    # way predict_prophet already does (coerce -> ffill -> bfill -> 0) so a
    # gappy regressor degrades gracefully instead of dropping Prophet from
    # the ensemble. A clean numeric column is unchanged by this.
    for regressor_name, _ in PROPHET_REGRESSORS:
        if regressor_name in df.columns:
            train_df[regressor_name] = (
                pd.to_numeric(df[regressor_name], errors="coerce")
                .ffill()
                .bfill()
                .fillna(0.0)
                .to_numpy()
            )
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
    start_ts: Any | None = None,
) -> dict[str, np.ndarray]:
    """
    Generate forecasts from a fitted Prophet model.

    Args:
        model: Fitted Prophet model.
        df: DataFrame with future timestamps and regressor values. When
            ``start_ts`` is given this should span the gap + horizon so the
            gap hours get real (not forward-filled) regressor values.
        periods: Number of hourly periods to forecast (default: 168 = 7 days).
        start_ts: Optional explicit forecast-origin timestamp. Prophet's
            ``make_future_dataframe`` anchors on the model's training history
            end; when the caller needs the window to begin at a later
            ``start_ts`` (the scoring tick runs after daily training), we
            extend the future frame across the gap and return the
            ``periods``-long window starting at ``start_ts`` (#194).

    Returns:
        Dict with keys: 'forecast', 'lower_80', 'upper_80', 'timestamps'.
        ``timestamps`` begins at ``start_ts`` when it is supplied.

        ``lower_80``/``upper_80`` are Prophet's genuine 80% posterior
        prediction interval (``yhat_lower``/``yhat_upper`` at the model's
        default ``interval_width=0.80``). We deliberately do **not** emit a
        95% band: the prior ``yhat_lower*0.95`` / ``yhat_upper*1.05`` was an
        uncalibrated visual heuristic, not a real 95% interval, so it was
        removed (#150). Displayed forecast intervals use empirical residual
        quantiles via ``models.evaluation`` (see the Overview/Forecast tabs);
        a calibrated 95% would belong there, not as a scaled 80%.
    """
    # Anchor handling (#194): Prophet appends future rows after its training
    # history end. To make the returned window start at an explicit start_ts,
    # extend the future frame across the train_end→start_ts gap, then select
    # the periods rows at/after start_ts.
    total_periods = periods
    start_naive = None
    if start_ts is not None:
        hist_end = pd.Timestamp(model.history["ds"].max())
        start_naive = pd.Timestamp(start_ts)
        if start_naive.tz is not None:
            start_naive = start_naive.tz_localize(None)
        offset_hours = int(round((start_naive - hist_end) / pd.Timedelta(hours=1)))
        # +periods rows from start_ts, plus the gap to reach it. One extra
        # step of slack is harmless (sliced off by the head(periods) below).
        total_periods = max(offset_hours, 0) + periods

    future = model.make_future_dataframe(periods=total_periods, freq="h")

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
            ts = (
                pd.to_datetime(df["ds"]).dt.tz_localize(None)
                if pd.to_datetime(df["ds"]).dt.tz
                else pd.to_datetime(df["ds"])
            )
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

    # Extract the forecast window. With an explicit start_ts, take the first
    # ``periods`` rows at/after start_ts (its own label); otherwise the last
    # N rows (byte-identical to the pre-#194 behavior).
    if start_naive is not None:
        fc = forecast[pd.to_datetime(forecast["ds"]) >= start_naive].head(periods)
    else:
        fc = forecast.tail(periods)

    # Hard physical floor: demand is non-negative. Prophet's logistic ``floor=0``
    # bounds the TREND only, not the additive composite (trend + seasonality +
    # regressors), so ``yhat`` — and especially ``yhat_lower`` — can go negative
    # even with floor=0 set (#281). Clip explicitly; do not rely on Prophet's
    # floor. (The scoring job re-floors every model + the ensemble at its serve
    # layer; this covers the dev inline-compute path and Prophet's band too.)
    return {
        "forecast": np.maximum(fc["yhat"].values, 0.0),
        # Prophet's real 80% posterior interval (interval_width defaults to
        # 0.80). No fabricated 95% band — see the function docstring (#150).
        "lower_80": np.maximum(fc["yhat_lower"].values, 0.0),
        "upper_80": np.maximum(fc["yhat_upper"].values, 0.0),
        "timestamps": pd.to_datetime(fc["ds"]).values,
    }
