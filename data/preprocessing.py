"""
Data preprocessing: cleaning, alignment, and missing value handling.

Merges EIA demand data with Open-Meteo weather data on UTC timestamps.
Handles timezone alignment, gap detection, interpolation, and flagging.

Rules (from spec AC-1.7, AC-1.8):
- EIA and Open-Meteo timestamps aligned to UTC hourly
- Gaps < 6 hours: linear interpolation
- Gaps > 6 hours: flagged (not interpolated)
"""

import pandas as pd
import numpy as np
import structlog

log = structlog.get_logger()

MAX_INTERPOLATION_GAP_HOURS = 6


def merge_demand_weather(
    demand_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge EIA demand data with Open-Meteo weather data on timestamp.

    Both DataFrames must have a 'timestamp' column in UTC.

    Args:
        demand_df: From eia_client.fetch_demand(). Columns: [timestamp, demand_mw, ...]
        weather_df: From weather_client.fetch_weather(). Columns: [timestamp, temp_2m, ...]

    Returns:
        Merged DataFrame with demand + all weather columns, aligned hourly.
    """
    if demand_df.empty or weather_df.empty:
        log.warning("merge_empty_input", demand_rows=len(demand_df), weather_rows=len(weather_df))
        return pd.DataFrame()

    # Ensure both are UTC
    demand_df = _ensure_utc(demand_df, "demand")
    weather_df = _ensure_utc(weather_df, "weather")

    # Round to nearest hour for alignment
    demand_df["timestamp"] = demand_df["timestamp"].dt.floor("h")
    weather_df["timestamp"] = weather_df["timestamp"].dt.floor("h")

    # Drop duplicate timestamps (keep last)
    demand_df = demand_df.drop_duplicates(subset="timestamp", keep="last")
    weather_df = weather_df.drop_duplicates(subset="timestamp", keep="last")

    # Merge
    merged = demand_df.merge(weather_df, on="timestamp", how="left")

    log.info(
        "data_merged",
        demand_rows=len(demand_df),
        weather_rows=len(weather_df),
        merged_rows=len(merged),
        null_weather_rows=int(merged.drop(columns=["timestamp", "demand_mw"], errors="ignore").isna().any(axis=1).sum()),
    )

    return merged.sort_values("timestamp").reset_index(drop=True)


def handle_missing_values(
    df: pd.DataFrame,
    max_gap_hours: int = MAX_INTERPOLATION_GAP_HOURS,
) -> pd.DataFrame:
    """
    Handle missing values in the merged dataset.

    Rules:
    - Gaps < max_gap_hours: linear interpolation
    - Gaps >= max_gap_hours: flagged with 'data_gap' column, NOT interpolated
    - Adds 'data_quality' column: 'original', 'interpolated', or 'gap'

    Args:
        df: Merged demand + weather DataFrame.
        max_gap_hours: Maximum gap size to interpolate (default: 6 hours).

    Returns:
        DataFrame with missing values handled and quality flags added.
    """
    if df.empty:
        return df

    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Detect gaps in the demand column (primary signal)
    demand_col = "demand_mw" if "demand_mw" in df.columns else None

    # Initialize quality column
    df["data_quality"] = "original"

    if demand_col:
        # Find missing demand periods
        missing_mask = df[demand_col].isna()
        if missing_mask.any():
            # Identify gap groups (consecutive NaN runs)
            gap_groups = (~missing_mask).cumsum()
            gap_sizes = missing_mask.groupby(gap_groups).transform("sum")

            # Interpolate small gaps
            small_gap_mask = missing_mask & (gap_sizes < max_gap_hours)
            large_gap_mask = missing_mask & (gap_sizes >= max_gap_hours)

            if small_gap_mask.any():
                for col in numeric_cols:
                    df.loc[small_gap_mask, col] = df[col].interpolate(method="linear")
                df.loc[small_gap_mask, "data_quality"] = "interpolated"

            if large_gap_mask.any():
                df.loc[large_gap_mask, "data_quality"] = "gap"

            log.info(
                "missing_values_handled",
                total_missing=int(missing_mask.sum()),
                interpolated=int(small_gap_mask.sum()),
                flagged_gaps=int(large_gap_mask.sum()),
            )

    # Interpolate remaining NaN in weather columns (small gaps are common)
    weather_cols = [c for c in numeric_cols if c != demand_col and c in df.columns]
    for col in weather_cols:
        if df[col].isna().any():
            df[col] = df[col].interpolate(method="linear", limit=max_gap_hours)

    return df


def validate_dataframe(
    df: pd.DataFrame,
    context: str = "",
) -> dict[str, any]:
    """
    Validate a DataFrame and return quality metrics.

    Checks for NaN, negative demand, and value ranges.
    Does not modify the DataFrame — just reports.

    Returns:
        Dict with validation results.
    """
    report = {
        "context": context,
        "rows": len(df),
        "columns": len(df.columns),
        "null_counts": {},
        "issues": [],
    }

    if df.empty:
        report["issues"].append("DataFrame is empty")
        return report

    # Check nulls per column
    null_counts = df.isna().sum()
    report["null_counts"] = {col: int(count) for col, count in null_counts.items() if count > 0}

    # Check demand range
    if "demand_mw" in df.columns:
        demand = df["demand_mw"].dropna()
        if (demand < 0).any():
            report["issues"].append(f"Negative demand values: {int((demand < 0).sum())} rows")
        if (demand > 500_000).any():
            report["issues"].append(f"Demand > 500,000 MW: {int((demand > 500_000).sum())} rows")

    # Check temperature range (Fahrenheit)
    if "temperature_2m" in df.columns:
        temp = df["temperature_2m"].dropna()
        if (temp < -50).any() or (temp > 150).any():
            report["issues"].append(f"Temperature out of range [-50, 150°F]")

    # Check wind speed range (mph)
    for wind_col in ["wind_speed_10m", "wind_speed_80m", "wind_speed_120m"]:
        if wind_col in df.columns:
            wind = df[wind_col].dropna()
            if (wind < 0).any():
                report["issues"].append(f"Negative {wind_col} values")
            if (wind > 200).any():
                report["issues"].append(f"{wind_col} > 200 mph")

    if report["issues"]:
        log.warning("data_validation_issues", **report)
    else:
        log.debug("data_validation_passed", **report)

    return report


def _ensure_utc(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Ensure timestamp column is UTC-aware."""
    df = df.copy()
    if df["timestamp"].dt.tz is None:
        log.debug("adding_utc_timezone", source=source)
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    elif str(df["timestamp"].dt.tz) != "UTC":
        log.debug("converting_to_utc", source=source, from_tz=str(df["timestamp"].dt.tz))
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    return df
