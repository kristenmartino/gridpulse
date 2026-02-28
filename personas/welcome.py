"""
Dynamic welcome card generator.

Per spec §Persona-Specific Welcome Cards:
- Grid Ops: peak demand, temperature, active alerts
- Renewables: wind CF, solar CF, curtailment risk
- Trader: demand vs forecast %, watch items
- Data Sci: ensemble MAPE, top model, feature drift

Each welcome message incorporates real-time data from the data stores,
not placeholder text (AC-7.4).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone

from config import REGION_NAMES, REGION_CAPACITY_MW
from personas.config import get_persona


def generate_welcome_message(
    persona_id: str,
    region: str,
    demand_df: pd.DataFrame | None = None,
    weather_df: pd.DataFrame | None = None,
) -> str:
    """
    Generate a data-driven welcome message for a persona.

    Args:
        persona_id: Persona identifier.
        region: Current region code.
        demand_df: Current demand DataFrame (optional).
        weather_df: Current weather DataFrame (optional).

    Returns:
        Welcome message string with embedded data values.
    """
    persona = get_persona(persona_id)
    region_name = REGION_NAMES.get(region, region)
    now = datetime.now(timezone.utc)
    hour = now.hour

    # Time-of-day greeting
    if hour < 12:
        greeting = "Good morning"
    elif hour < 17:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"

    # Extract data points if available
    stats = _extract_data_stats(region, demand_df, weather_df)

    generators = {
        "grid_ops": _welcome_grid_ops,
        "renewables": _welcome_renewables,
        "trader": _welcome_trader,
        "data_scientist": _welcome_data_scientist,
    }

    generator = generators.get(persona_id, _welcome_grid_ops)
    return generator(greeting, persona.name, region_name, stats)


def _extract_data_stats(
    region: str,
    demand_df: pd.DataFrame | None,
    weather_df: pd.DataFrame | None,
) -> dict:
    """Extract key statistics from data stores."""
    stats = {
        "peak_mw": None,
        "peak_time": None,
        "current_temp": None,
        "max_temp": None,
        "mape": None,
        "demand_vs_forecast_pct": None,
        "wind_speed": None,
        "capacity": REGION_CAPACITY_MW.get(region, 50000),
    }

    if demand_df is not None and len(demand_df) > 0:
        demand_df = demand_df.copy()
        if "timestamp" in demand_df.columns:
            demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])

        # Today's data, or fall back to most recent 24 hours if no today data
        now = pd.Timestamp.now(tz="UTC")
        if "timestamp" in demand_df.columns:
            today = demand_df[demand_df["timestamp"].dt.date == now.date()]
            if len(today) == 0:
                # Fall back to most recent 24 hours of data
                today = demand_df.tail(24)
        else:
            today = demand_df.tail(24)

        if len(today) > 0:
            peak_idx = today["demand_mw"].idxmax()
            stats["peak_mw"] = today.loc[peak_idx, "demand_mw"]
            if "timestamp" in today.columns:
                stats["peak_time"] = today.loc[peak_idx, "timestamp"].strftime("%I:%M %p")

            # Demand vs forecast
            if "forecast_mw" in today.columns:
                actual_mean = today["demand_mw"].mean()
                forecast_mean = today["forecast_mw"].mean()
                if forecast_mean > 0:
                    stats["demand_vs_forecast_pct"] = (actual_mean - forecast_mean) / forecast_mean * 100

        # Rolling MAPE approximation
        if "forecast_mw" in demand_df.columns:
            recent = demand_df.tail(168)  # 7 days
            mask = recent["demand_mw"].abs() > 1
            if mask.sum() > 0:
                mape = (
                    (recent.loc[mask, "demand_mw"] - recent.loc[mask, "forecast_mw"]).abs()
                    / recent.loc[mask, "demand_mw"].abs()
                ).mean() * 100
                stats["mape"] = round(mape, 1)

    if weather_df is not None and len(weather_df) > 0:
        weather_df = weather_df.copy()
        if "timestamp" in weather_df.columns:
            weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])

        recent_weather = weather_df.tail(24)
        if "temperature_2m" in recent_weather.columns:
            stats["current_temp"] = recent_weather["temperature_2m"].iloc[-1]
            stats["max_temp"] = recent_weather["temperature_2m"].max()
        if "wind_speed_80m" in recent_weather.columns:
            stats["wind_speed"] = recent_weather["wind_speed_80m"].mean()

    return stats


def _welcome_grid_ops(greeting: str, name: str, region: str, stats: dict) -> str:
    """Grid Operations Manager welcome."""
    parts = [f"{greeting}, {name}. Here's your grid status for {region}."]

    if stats["peak_mw"] is not None:
        parts.append(
            f"Today's peak demand is forecast at {stats['peak_mw']:,.0f} MW"
            + (f" at {stats['peak_time']}" if stats["peak_time"] else "")
            + "."
        )
    if stats["current_temp"] is not None:
        parts.append(f"Current temperature: {stats['current_temp']:.0f}°F.")
    if stats["mape"] is not None:
        parts.append(f"7-day forecast accuracy: {stats['mape']:.1f}% MAPE.")

    return " ".join(parts)


def _welcome_renewables(greeting: str, name: str, region: str, stats: dict) -> str:
    """Renewables Portfolio Analyst welcome."""
    parts = [f"{greeting}, {name}. Your renewable generation outlook for {region} is ready."]

    if stats["wind_speed"] is not None:
        parts.append(f"Average wind speed: {stats['wind_speed']:.0f} mph.")
    if stats["max_temp"] is not None:
        parts.append(f"Peak temperature forecast: {stats['max_temp']:.0f}°F.")

    parts.append("Check the Generation Mix tab for detailed capacity factors.")
    return " ".join(parts)


def _welcome_trader(greeting: str, name: str, region: str, stats: dict) -> str:
    """Energy Trader welcome."""
    parts = [f"{greeting}, {name}. Market intelligence for {region} is updated."]

    if stats["demand_vs_forecast_pct"] is not None:
        direction = "above" if stats["demand_vs_forecast_pct"] > 0 else "below"
        parts.append(
            f"Demand is {abs(stats['demand_vs_forecast_pct']):.1f}% {direction} forecast."
        )
    if stats["peak_mw"] is not None and stats["capacity"]:
        utilization = stats["peak_mw"] / stats["capacity"] * 100
        parts.append(f"Peak utilization: {utilization:.0f}%.")

    return " ".join(parts)


def _welcome_data_scientist(greeting: str, name: str, region: str, stats: dict) -> str:
    """Data Scientist welcome."""
    parts = [f"Hey {name}. Model diagnostics for {region} are ready for review."]

    if stats["mape"] is not None:
        status = "on target" if stats["mape"] < 5 else "needs attention"
        parts.append(f"Ensemble MAPE: {stats['mape']:.1f}% ({status}).")

    parts.append("Check the Model Comparison tab for residual analysis and SHAP values.")
    return " ".join(parts)
