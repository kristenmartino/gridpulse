"""
Open-Meteo API client for weather data ingestion.

Fetches historical and forecast weather data for each balancing authority
centroid. No API key required for non-commercial use.

Key design decisions:
- Always request Fahrenheit/mph (CDD/HDD use 65°F baseline, sliders use mph)
- &past_days=92 seamlessly joins 3 months historical with 7-day forecast
- All 17 weather variables fetched in a single call per region

API docs: https://open-meteo.com/en/docs
"""

import pandas as pd
import requests
import structlog

from config import (
    CACHE_TTL_SECONDS,
    OPEN_METEO_BASE_URL,
    REGION_COORDINATES,
    WEATHER_VARIABLES,
)
from data.cache import get_cache

log = structlog.get_logger()

# Open-Meteo is generous with rate limits, but be respectful
REQUEST_TIMEOUT = 30


def fetch_weather(
    region: str,
    past_days: int = 92,
    forecast_days: int = 7,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical + forecast weather data for a balancing authority centroid.

    Uses Open-Meteo's &past_days parameter to seamlessly join historical
    data with the forecast in a single API call.

    Args:
        region: Balancing authority code (e.g., "ERCOT", "FPL").
        past_days: Number of historical days to include (default 92 = ~3 months).
        forecast_days: Number of forecast days (default 7).
        use_cache: Whether to check cache first.

    Returns:
        DataFrame with columns: [timestamp] + all 17 WEATHER_VARIABLES
    """
    if region not in REGION_COORDINATES:
        raise ValueError(f"Unknown region: {region}. Valid: {list(REGION_COORDINATES.keys())}")

    cache_key = f"weather_{region}_past{past_days}_fc{forecast_days}"
    cache = get_cache()

    if use_cache:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    coords = REGION_COORDINATES[region]
    log.info("weather_fetching", region=region, lat=coords["lat"], lon=coords["lon"])

    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": ",".join(WEATHER_VARIABLES),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
    }

    try:
        resp = requests.get(
            f"{OPEN_METEO_BASE_URL}/forecast",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        log.error("weather_request_failed", region=region, error=str(e))
        stale = cache.get(cache_key, allow_stale=True)
        if stale is not None:
            return stale
        return pd.DataFrame(columns=["timestamp"] + WEATHER_VARIABLES)

    df = _parse_weather_response(data)
    if df.empty:
        log.warning("weather_empty_response", region=region)
        stale = cache.get(cache_key, allow_stale=True)
        if stale is not None:
            return stale
        return df

    cache.set(cache_key, df, ttl=CACHE_TTL_SECONDS)
    log.info("weather_cached", region=region, rows=len(df))
    return df


def fetch_historical_weather(
    region: str,
    start_date: str,
    end_date: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical weather data for a specific date range.

    Uses Open-Meteo's archive endpoint for data back to 1940 (ERA5 reanalysis).

    Args:
        region: Balancing authority code.
        start_date: Start date "YYYY-MM-DD".
        end_date: End date "YYYY-MM-DD".
        use_cache: Whether to check cache first.

    Returns:
        DataFrame with columns: [timestamp] + all 17 WEATHER_VARIABLES
    """
    if region not in REGION_COORDINATES:
        raise ValueError(f"Unknown region: {region}")

    cache_key = f"weather_hist_{region}_{start_date}_{end_date}"
    cache = get_cache()

    if use_cache:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    coords = REGION_COORDINATES[region]
    log.info("weather_fetching_historical", region=region, start=start_date, end=end_date)

    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": ",".join(WEATHER_VARIABLES),
        "start_date": start_date,
        "end_date": end_date,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
    }

    # Historical API uses different base URL
    archive_url = "https://archive-api.open-meteo.com/v1/archive"
    try:
        resp = requests.get(
            archive_url,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        log.error("weather_historical_request_failed", region=region, error=str(e))
        stale = cache.get(cache_key, allow_stale=True)
        if stale is not None:
            return stale
        return pd.DataFrame(columns=["timestamp"] + WEATHER_VARIABLES)

    df = _parse_weather_response(data)
    # Historical data is stable — cache aggressively (24h)
    cache.set(cache_key, df, ttl=86400)
    log.info("weather_historical_cached", region=region, rows=len(df))
    return df


def _parse_weather_response(data: dict) -> pd.DataFrame:
    """
    Parse Open-Meteo JSON response into a DataFrame.

    Open-Meteo returns:
    {
        "hourly": {
            "time": ["2025-01-01T00:00", ...],
            "temperature_2m": [45.2, ...],
            ...
        }
    }
    """
    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame(columns=["timestamp"] + WEATHER_VARIABLES)

    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Ensure all expected columns exist (fill missing with NaN)
    for var in WEATHER_VARIABLES:
        if var not in df.columns:
            df[var] = None

    # Reorder columns
    cols = ["timestamp"] + [v for v in WEATHER_VARIABLES if v in df.columns]
    df = df[cols]

    return df.sort_values("timestamp").reset_index(drop=True)
