"""
Feature builder for the batch scoring pipeline.

Thin wrapper around the v1 feature_engineering.py (379 lines of tested code).
Reads raw data from Postgres, delegates to v1's engineer_features() for the
43-feature pipeline, and returns the feature matrix for model inference.
"""

from __future__ import annotations

import logging

import pandas as pd
import psycopg2

# v1 imports — resolved via sys.path set in src/__init__.py
from data.feature_engineering import engineer_features, get_feature_names
from data.preprocessing import handle_missing_values, merge_demand_weather
from src.config import DatabaseConfig

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Builds the 43-feature matrix from Postgres raw data."""

    def __init__(self, db_config: DatabaseConfig | None = None):
        self.db_config = db_config or DatabaseConfig()
        self.conn = psycopg2.connect(self.db_config.url)

    def build_features(self, region: str, horizon_hours: int = 24) -> pd.DataFrame:
        """
        Build feature matrix for a region from Postgres raw data.

        Steps:
          1. Query raw demand + weather from Postgres
          2. merge_demand_weather() (v1 preprocessing.py)
          3. handle_missing_values() (v1 preprocessing.py)
          4. engineer_features() (v1 feature_engineering.py — 43 features)

        Args:
            region: Balancing authority code (e.g., "ERCOT").
            horizon_hours: Forecast horizon in hours.

        Returns:
            Feature-engineered DataFrame ready for model.predict().
        """
        demand_df = self._query_demand(region)
        weather_df = self._query_weather(region)

        if demand_df.empty or weather_df.empty:
            logger.warning("No raw data for %s", region)
            return pd.DataFrame()

        merged = merge_demand_weather(demand_df, weather_df)
        cleaned = handle_missing_values(merged)
        featured = engineer_features(cleaned)

        logger.info(
            "Features built for %s: %d rows, %d features",
            region,
            len(featured),
            len(get_feature_names()),
        )
        return featured

    def _query_demand(self, region: str) -> pd.DataFrame:
        """Read raw demand data from Postgres raw_demand table."""
        query = """
            SELECT timestamp, demand_mw, forecast_mw, region
            FROM raw_demand
            WHERE region = %s
            ORDER BY timestamp DESC
            LIMIT 2160
        """
        df = pd.read_sql(query, self.conn, params=[region])
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)

    def _query_weather(self, region: str) -> pd.DataFrame:
        """Read raw weather data from Postgres raw_weather table."""
        query = """
            SELECT *
            FROM raw_weather
            WHERE region = %s
            ORDER BY timestamp DESC
            LIMIT 2160
        """
        df = pd.read_sql(query, self.conn, params=[region])
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)

    @staticmethod
    def get_feature_columns() -> list[str]:
        """Return canonical feature column names from v1."""
        return get_feature_names()

    def close(self):
        """Close the Postgres connection."""
        self.conn.close()
