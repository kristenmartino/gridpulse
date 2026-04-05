"""
Unit tests for data/weather_client.py.

Covers fetch_weather(), fetch_historical_weather(), and _parse_weather_response()
with full fallback chain verification. All HTTP, cache, and GCS calls are mocked.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from config import WEATHER_VARIABLES

# ---------------------------------------------------------------------------
# _parse_weather_response tests
# ---------------------------------------------------------------------------


class TestParseWeatherResponse:
    """Tests for _parse_weather_response — pure parsing, no I/O."""

    def test_full_response_all_17_vars(self, mock_weather_response):
        """Full Open-Meteo response with all 17 variables parses correctly."""
        from data.weather_client import _parse_weather_response

        df = _parse_weather_response(mock_weather_response)

        assert not df.empty
        assert len(df) == 3
        assert "timestamp" in df.columns
        for var in WEATHER_VARIABLES:
            assert var in df.columns, f"Missing expected column: {var}"
        assert df["temperature_2m"].tolist() == [45.0, 44.0, 43.5]

    def test_timestamps_are_utc(self, mock_weather_response):
        """Parsed timestamps are timezone-aware UTC."""
        from data.weather_client import _parse_weather_response

        df = _parse_weather_response(mock_weather_response)

        assert df["timestamp"].dt.tz is not None
        assert str(df["timestamp"].dt.tz) == "UTC"

    def test_timestamps_sorted_ascending(self, mock_weather_response):
        """Output rows are sorted by timestamp ascending."""
        from data.weather_client import _parse_weather_response

        # Reverse the time order in the input
        reversed_data = {
            "hourly": {k: list(reversed(v)) for k, v in mock_weather_response["hourly"].items()}
        }
        df = _parse_weather_response(reversed_data)

        assert df["timestamp"].is_monotonic_increasing

    def test_missing_hourly_key_returns_empty(self):
        """Response without 'hourly' key returns empty DataFrame with correct columns."""
        from data.weather_client import _parse_weather_response

        df = _parse_weather_response({"latitude": 31.0})

        assert df.empty
        assert "timestamp" in df.columns
        for var in WEATHER_VARIABLES:
            assert var in df.columns

    def test_empty_hourly_dict_returns_empty(self):
        """Response with empty 'hourly' dict returns empty DataFrame."""
        from data.weather_client import _parse_weather_response

        df = _parse_weather_response({"hourly": {}})

        assert df.empty

    def test_hourly_missing_time_key_returns_empty(self):
        """Response with 'hourly' but no 'time' key returns empty DataFrame."""
        from data.weather_client import _parse_weather_response

        df = _parse_weather_response({"hourly": {"temperature_2m": [45.0]}})

        assert df.empty

    def test_partial_vars_fills_missing_with_none(self):
        """Response with only some weather variables fills the rest with None."""
        from data.weather_client import _parse_weather_response

        partial_data = {
            "hourly": {
                "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
                "temperature_2m": [45.0, 44.0],
                "wind_speed_10m": [8.0, 9.0],
            }
        }
        df = _parse_weather_response(partial_data)

        assert len(df) == 2
        assert df["temperature_2m"].tolist() == [45.0, 44.0]
        assert df["wind_speed_10m"].tolist() == [8.0, 9.0]
        # Variables not in the response should be None/NaN
        assert df["cloud_cover"].isna().all()
        assert df["precipitation"].isna().all()

    def test_reset_index(self, mock_weather_response):
        """Output index is reset (0, 1, 2, ...) after sorting."""
        from data.weather_client import _parse_weather_response

        df = _parse_weather_response(mock_weather_response)

        assert list(df.index) == [0, 1, 2]


# ---------------------------------------------------------------------------
# fetch_weather tests
# ---------------------------------------------------------------------------


class TestFetchWeather:
    """Tests for fetch_weather — full fallback chain."""

    def test_invalid_region_raises_value_error(self):
        """Unknown region raises ValueError with helpful message."""
        from data.weather_client import fetch_weather

        with pytest.raises(ValueError, match="Unknown region"):
            fetch_weather("INVALID_REGION")

    @patch("data.weather_client.get_cache")
    def test_returns_cached_data_when_available(self, mock_get_cache):
        """Cache hit returns the cached DataFrame directly, no HTTP call."""
        from data.weather_client import fetch_weather

        cached_df = pd.DataFrame({"timestamp": [1], "temperature_2m": [70.0]})
        mock_cache = MagicMock()
        mock_cache.get.return_value = cached_df
        mock_get_cache.return_value = mock_cache

        result = fetch_weather("ERCOT", use_cache=True)

        mock_cache.get.assert_called_once()
        pd.testing.assert_frame_equal(result, cached_df)

    @patch("data.weather_client.get_cache")
    def test_skips_cache_when_disabled(self, mock_get_cache, mock_weather_response):
        """use_cache=False skips cache lookup, calls API directly."""
        from data.weather_client import fetch_weather

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_weather_response
        mock_resp.raise_for_status.return_value = None

        with (
            patch("data.weather_client.requests.get", return_value=mock_resp),
            patch("data.gcs_store.write_parquet"),
        ):
            result = fetch_weather("ERCOT", use_cache=False)

        # cache.get should NOT be called when use_cache=False
        mock_cache.get.assert_not_called()
        assert not result.empty

    @patch("data.weather_client.get_cache")
    def test_success_caches_result_and_writes_gcs(self, mock_get_cache, mock_weather_response):
        """Successful API call caches in SQLite and writes to GCS."""
        from data.weather_client import fetch_weather

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_weather_response
        mock_resp.raise_for_status.return_value = None

        with (
            patch("data.weather_client.requests.get", return_value=mock_resp),
            patch("data.gcs_store.write_parquet") as mock_write,
        ):
            result = fetch_weather("ERCOT")

        assert len(result) == 3
        mock_cache.set.assert_called_once()
        mock_write.assert_called_once()
        # Verify GCS write args: df, data_type, region
        write_args = mock_write.call_args
        assert write_args[0][1] == "weather"
        assert write_args[0][2] == "ERCOT"

    @patch("data.weather_client.get_cache")
    def test_api_timeout_falls_back_to_stale_cache(self, mock_get_cache):
        """API timeout returns stale cached data when available."""
        from data.weather_client import fetch_weather

        stale_df = pd.DataFrame({"timestamp": [1], "temperature_2m": [65.0]})
        mock_cache = MagicMock()
        # First get (fresh) returns None, stale get returns data
        mock_cache.get.side_effect = [None, stale_df]
        mock_get_cache.return_value = mock_cache

        with patch(
            "data.weather_client.requests.get",
            side_effect=requests.Timeout("Connection timed out"),
        ):
            result = fetch_weather("ERCOT")

        pd.testing.assert_frame_equal(result, stale_df)
        # Verify allow_stale=True was used in fallback
        assert mock_cache.get.call_count == 2
        assert mock_cache.get.call_args_list[1][1] == {"allow_stale": True}

    @patch("data.weather_client.get_cache")
    def test_api_error_no_stale_falls_back_to_gcs(self, mock_get_cache):
        """API failure + no stale cache reads from GCS."""
        from data.weather_client import fetch_weather

        gcs_df = pd.DataFrame(
            {"timestamp": pd.to_datetime(["2024-01-01"], utc=True), "temperature_2m": [55.0]}
        )
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # Both fresh and stale return None
        mock_get_cache.return_value = mock_cache

        with (
            patch(
                "data.weather_client.requests.get",
                side_effect=requests.ConnectionError("DNS failure"),
            ),
            patch("data.gcs_store.read_parquet", return_value=gcs_df) as mock_read,
        ):
            result = fetch_weather("ERCOT")

        mock_read.assert_called_once_with("weather", "ERCOT")
        pd.testing.assert_frame_equal(result, gcs_df)

    @patch("data.weather_client.get_cache")
    def test_api_error_no_stale_no_gcs_returns_empty(self, mock_get_cache):
        """API failure + no stale cache + no GCS returns empty DataFrame."""
        from data.weather_client import fetch_weather

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        with (
            patch(
                "data.weather_client.requests.get",
                side_effect=requests.ConnectionError("No network"),
            ),
            patch("data.gcs_store.read_parquet", return_value=None),
        ):
            result = fetch_weather("ERCOT")

        assert result.empty
        assert "timestamp" in result.columns
        for var in WEATHER_VARIABLES:
            assert var in result.columns

    @patch("data.weather_client.get_cache")
    def test_empty_api_response_falls_back_to_stale(self, mock_get_cache):
        """API returns empty hourly data -- falls back to stale cache."""
        from data.weather_client import fetch_weather

        stale_df = pd.DataFrame({"timestamp": [1], "temperature_2m": [72.0]})
        mock_cache = MagicMock()
        # First call (fresh) returns None, second call (stale) returns stale_df
        mock_cache.get.side_effect = [None, stale_df]
        mock_get_cache.return_value = mock_cache

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"hourly": {}}
        mock_resp.raise_for_status.return_value = None

        with patch("data.weather_client.requests.get", return_value=mock_resp):
            result = fetch_weather("ERCOT")

        pd.testing.assert_frame_equal(result, stale_df)

    @patch("data.weather_client.get_cache")
    def test_empty_response_no_stale_falls_back_to_gcs(self, mock_get_cache):
        """API returns empty + no stale cache reads from GCS."""
        from data.weather_client import fetch_weather

        gcs_df = pd.DataFrame(
            {"timestamp": pd.to_datetime(["2024-06-01"], utc=True), "temperature_2m": [90.0]}
        )
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"hourly": {}}
        mock_resp.raise_for_status.return_value = None

        with (
            patch("data.weather_client.requests.get", return_value=mock_resp),
            patch("data.gcs_store.read_parquet", return_value=gcs_df),
        ):
            result = fetch_weather("ERCOT")

        pd.testing.assert_frame_equal(result, gcs_df)

    @patch("data.weather_client.get_cache")
    def test_api_called_with_correct_params(self, mock_get_cache, mock_weather_response):
        """API request uses correct coordinates, variables, and units."""
        from data.weather_client import fetch_weather

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_weather_response
        mock_resp.raise_for_status.return_value = None

        with (
            patch("data.weather_client.requests.get", return_value=mock_resp) as mock_get,
            patch("data.gcs_store.write_parquet"),
        ):
            fetch_weather("ERCOT", past_days=30, forecast_days=3)

        _, kwargs = mock_get.call_args
        params = kwargs["params"]
        assert params["latitude"] == 31.0
        assert params["longitude"] == -97.0
        assert params["past_days"] == 30
        assert params["forecast_days"] == 3
        assert params["temperature_unit"] == "fahrenheit"
        assert params["wind_speed_unit"] == "mph"
        assert params["timezone"] == "UTC"
        assert "temperature_2m" in params["hourly"]

    @patch("data.weather_client.get_cache")
    def test_http_500_triggers_fallback_chain(self, mock_get_cache):
        """HTTP 500 error triggers the same fallback as timeout."""
        from data.weather_client import fetch_weather

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")

        with (
            patch("data.weather_client.requests.get", return_value=mock_resp),
            patch("data.gcs_store.read_parquet", return_value=None),
        ):
            result = fetch_weather("ERCOT")

        assert result.empty


# ---------------------------------------------------------------------------
# fetch_historical_weather tests
# ---------------------------------------------------------------------------


class TestFetchHistoricalWeather:
    """Tests for fetch_historical_weather — date-range archive endpoint."""

    def test_invalid_region_raises_value_error(self):
        """Unknown region raises ValueError."""
        from data.weather_client import fetch_historical_weather

        with pytest.raises(ValueError, match="Unknown region"):
            fetch_historical_weather("BOGUS", "2024-01-01", "2024-01-31")

    def test_invalid_start_date_format_raises(self):
        """Malformed start_date raises ValueError with helpful message."""
        from data.weather_client import fetch_historical_weather

        with pytest.raises(ValueError, match="Invalid start_date format"):
            fetch_historical_weather("ERCOT", "01-01-2024", "2024-01-31")

    def test_invalid_end_date_format_raises(self):
        """Malformed end_date raises ValueError."""
        from data.weather_client import fetch_historical_weather

        with pytest.raises(ValueError, match="Invalid end_date format"):
            fetch_historical_weather("ERCOT", "2024-01-01", "not-a-date")

    @patch("data.weather_client.get_cache")
    def test_success_returns_parsed_dataframe(self, mock_get_cache, mock_weather_response):
        """Successful historical fetch returns parsed DataFrame and caches it."""
        from data.weather_client import fetch_historical_weather

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_weather_response
        mock_resp.raise_for_status.return_value = None

        with patch("data.weather_client.requests.get", return_value=mock_resp) as mock_get:
            result = fetch_historical_weather("ERCOT", "2024-01-01", "2024-01-31")

        assert len(result) == 3
        # Verify archive URL was used
        call_url = mock_get.call_args[0][0]
        assert "archive-api.open-meteo.com" in call_url
        # Verify cached with 24h TTL
        mock_cache.set.assert_called_once()
        cache_ttl = mock_cache.set.call_args[1]["ttl"]
        assert cache_ttl == 86400

    @patch("data.weather_client.get_cache")
    def test_returns_cached_historical_data(self, mock_get_cache):
        """Cache hit skips API call and returns cached historical data."""
        from data.weather_client import fetch_historical_weather

        cached_df = pd.DataFrame({"timestamp": [1], "temperature_2m": [50.0]})
        mock_cache = MagicMock()
        mock_cache.get.return_value = cached_df
        mock_get_cache.return_value = mock_cache

        result = fetch_historical_weather("ERCOT", "2024-01-01", "2024-01-31")

        pd.testing.assert_frame_equal(result, cached_df)

    @patch("data.weather_client.get_cache")
    def test_api_error_falls_back_to_stale(self, mock_get_cache):
        """API failure returns stale cached historical data."""
        from data.weather_client import fetch_historical_weather

        stale_df = pd.DataFrame({"timestamp": [1], "temperature_2m": [48.0]})
        mock_cache = MagicMock()
        mock_cache.get.side_effect = [None, stale_df]
        mock_get_cache.return_value = mock_cache

        with patch(
            "data.weather_client.requests.get",
            side_effect=requests.Timeout("Timed out"),
        ):
            result = fetch_historical_weather("ERCOT", "2024-01-01", "2024-01-31")

        pd.testing.assert_frame_equal(result, stale_df)

    @patch("data.weather_client.get_cache")
    def test_api_error_no_stale_returns_empty(self, mock_get_cache):
        """API failure with no stale cache returns empty DataFrame."""
        from data.weather_client import fetch_historical_weather

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        with patch(
            "data.weather_client.requests.get",
            side_effect=requests.ConnectionError("No network"),
        ):
            result = fetch_historical_weather("ERCOT", "2024-01-01", "2024-01-31")

        assert result.empty
        assert "timestamp" in result.columns
        for var in WEATHER_VARIABLES:
            assert var in result.columns

    @patch("data.weather_client.get_cache")
    def test_none_date_raises_value_error(self, mock_get_cache):
        """Passing None as a date raises ValueError (not TypeError)."""
        from data.weather_client import fetch_historical_weather

        with pytest.raises(ValueError, match="Invalid start_date format"):
            fetch_historical_weather("ERCOT", None, "2024-01-31")
