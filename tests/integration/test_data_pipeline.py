"""
Integration tests for the full data pipeline.

Tests the chain: API response → parse → cache → merge → preprocess → features.
All external APIs are mocked — these tests run offline.
"""

from unittest.mock import MagicMock, patch

import numpy as np


class TestDataPipelineIntegration:
    """End-to-end data pipeline with mocked APIs."""

    def test_eia_parse_to_dataframe(self, mock_eia_response):
        """EIA JSON → DataFrame with correct schema."""
        from data.eia_client import _parse_demand_records

        records = mock_eia_response["response"]["data"]
        df = _parse_demand_records(records, "ERCOT")

        assert "timestamp" in df.columns
        assert "demand_mw" in df.columns
        assert "region" in df.columns
        assert len(df) > 0
        assert (df["demand_mw"] > 0).all()

    def test_weather_parse_to_dataframe(self, mock_weather_response):
        """Open-Meteo JSON → DataFrame with all 17 variables."""
        from data.weather_client import _parse_weather_response

        df = _parse_weather_response(mock_weather_response)

        assert "timestamp" in df.columns
        assert "temperature_2m" in df.columns
        assert "wind_speed_80m" in df.columns
        assert len(df) == 3

    def test_noaa_parse_alerts(self, mock_noaa_alerts_response):
        """NOAA JSON → sorted alert list."""
        from data.noaa_client import _fetch_state_alerts

        with patch("data.noaa_client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_noaa_alerts_response
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp

            alerts = _fetch_state_alerts("TX")

        assert len(alerts) == 2
        # Should have both a critical and a warning
        severities = {a.severity for a in alerts}
        assert "critical" in severities  # Extreme → critical
        assert "warning" in severities  # Moderate → warning

    def test_full_feature_pipeline(self, sample_demand_df, sample_weather_df):
        """demand + weather → merge → preprocess → features → model-ready."""
        from data.feature_engineering import engineer_features
        from data.preprocessing import handle_missing_values, merge_demand_weather

        # Merge
        merged = merge_demand_weather(sample_demand_df, sample_weather_df)
        assert len(merged) > 0

        # Handle missing values
        cleaned = handle_missing_values(merged)
        assert "data_quality" in cleaned.columns

        # Feature engineering
        features = engineer_features(cleaned)
        assert "cooling_degree_days" in features.columns
        assert "demand_lag_24h" in features.columns
        assert features.select_dtypes(include=[np.number]).isna().sum().sum() == 0

    def test_cache_roundtrip_dataframe(self, tmp_cache, sample_demand_df):
        """DataFrame → cache → retrieve → identical."""
        tmp_cache.set("test_df", sample_demand_df)
        retrieved = tmp_cache.get("test_df")

        assert len(retrieved) == len(sample_demand_df)
        assert list(retrieved.columns) == list(sample_demand_df.columns)

    def test_stale_fallback_flow(self, tmp_cache):
        """Simulate API down → serve stale cache."""
        tmp_cache.set("api_data", {"value": 42}, ttl=0)

        import time

        time.sleep(0.05)

        # Fresh get returns None
        assert tmp_cache.get("api_data") is None

        # Stale fallback returns old data
        stale = tmp_cache.get("api_data", allow_stale=True)
        assert stale == {"value": 42}
