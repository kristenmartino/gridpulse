"""
Data ingestion, caching, preprocessing, and feature engineering.

Public API:
    from data import fetch_demand, fetch_weather, fetch_alerts_for_region
    from data import merge_demand_weather, engineer_features
"""

from data.cache import get_cache, Cache
from data.eia_client import fetch_demand, fetch_generation_by_fuel, fetch_interchange
from data.weather_client import fetch_weather, fetch_historical_weather
from data.noaa_client import fetch_alerts_for_region, fetch_all_alerts
from data.preprocessing import merge_demand_weather, handle_missing_values, validate_dataframe
from data.feature_engineering import engineer_features, get_feature_names

__all__ = [
    "get_cache", "Cache",
    "fetch_demand", "fetch_generation_by_fuel", "fetch_interchange",
    "fetch_weather", "fetch_historical_weather",
    "fetch_alerts_for_region", "fetch_all_alerts",
    "merge_demand_weather", "handle_missing_values", "validate_dataframe",
    "engineer_features", "get_feature_names",
]
