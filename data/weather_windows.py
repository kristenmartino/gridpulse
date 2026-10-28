"""Pure UTC selectors for feeds containing both historical and future weather.

Open-Meteo's combined frame extends beyond the current clock. Row position is
therefore never evidence that a sample is current or forecast-aligned. These
selectors retain the feed's units and gaps; they do not interpolate observations
or reconstruct the weather forecast that was available at an earlier issuance.
"""

from __future__ import annotations

from math import isfinite

import pandas as pd


def _utc(value: object) -> pd.Timestamp:
    """Interpret naive pipeline clocks as UTC, matching the weather feed schema."""
    clock = pd.Timestamp(value)
    if pd.isna(clock):
        raise ValueError("A valid weather-window clock is required")
    return clock.tz_localize("UTC") if clock.tzinfo is None else clock.tz_convert("UTC")


def _normalized(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(columns=["timestamp"])
    result = frame.copy()
    if "timestamp" not in result.columns:
        return result.iloc[:0].assign(timestamp=pd.Series(dtype="datetime64[ns, UTC]"))
    result["timestamp"] = pd.to_datetime(
        result["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    return result.loc[result["timestamp"].notna()].sort_values("timestamp", kind="stable")


def historical_weather_window(
    frame: pd.DataFrame | None, as_of: object, hours: float = 168
) -> pd.DataFrame:
    """Return existing samples in ``(as_of - hours, as_of]`` in UTC.

    Historical model estimates remain estimates; this temporal partition does
    not certify station observations. Missing hours stay missing.
    """
    if not isfinite(hours) or hours <= 0:
        raise ValueError("Weather history hours must be positive and finite")
    clock = _utc(as_of)
    result = _normalized(frame)
    return result.loc[
        (result["timestamp"] > clock - pd.Timedelta(hours=hours)) & (result["timestamp"] <= clock)
    ].copy()


def forecast_weather_window(frame: pd.DataFrame | None, start: object, end: object) -> pd.DataFrame:
    """Return existing weather samples in the inclusive selected target window.

    The caller supplies the demand forecast's actual target timestamps, not a
    nominal count of rows at the end of the weather feed.
    """
    first, last = _utc(start), _utc(end)
    if last < first:
        raise ValueError("Weather-window end must not precede start")
    result = _normalized(frame)
    return result.loc[result["timestamp"].between(first, last)].copy()


def latest_weather_at_or_before(frame: pd.DataFrame | None, as_of: object) -> pd.Series | None:
    """Return the latest timestamped sample at/before the explicit clock.

    Its age remains visible in ``timestamp``; callers must not imply freshness
    from selection alone. Future-only or unclocked feeds have no current sample.
    """
    clock = _utc(as_of)
    result = _normalized(frame)
    available = result.loc[result["timestamp"] <= clock]
    return None if available.empty else available.iloc[-1].copy()
