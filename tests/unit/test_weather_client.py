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
    def test_calls_both_endpoints_with_correct_params(self, mock_get_cache, mock_weather_response):
        """fetch_weather now hits TWO endpoints (#161): /forecast for
        recent+future, archive for deep history. Verify each is called
        with the right params — coords/units shared, forecast uses a
        short past_days + forecast_days, archive uses start/end dates."""
        from data.weather_client import ARCHIVE_LAG_DAYS, fetch_weather

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

        # Two HTTP calls: forecast first, archive second.
        assert mock_get.call_count == 2
        forecast_call, archive_call = mock_get.call_args_list

        # Forecast endpoint: short past_days (overlap only), the requested
        # forecast horizon, shared coords/units.
        f_url = forecast_call.args[0] if forecast_call.args else forecast_call.kwargs.get("url", "")
        f_params = forecast_call.kwargs["params"]
        assert "/forecast" in f_url
        assert f_params["latitude"] == 31.0
        assert f_params["longitude"] == -97.0
        assert f_params["past_days"] == ARCHIVE_LAG_DAYS + 2
        assert f_params["forecast_days"] == 3
        assert f_params["temperature_unit"] == "fahrenheit"
        assert f_params["wind_speed_unit"] == "mph"
        assert "temperature_2m" in f_params["hourly"]

        # Archive endpoint: start/end dates instead of past_days, same
        # coords/units.
        a_url = archive_call.args[0] if archive_call.args else archive_call.kwargs.get("url", "")
        a_params = archive_call.kwargs["params"]
        assert "archive" in a_url
        assert "start_date" in a_params
        assert "end_date" in a_params
        assert "past_days" not in a_params
        assert a_params["latitude"] == 31.0
        assert a_params["temperature_unit"] == "fahrenheit"

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


# ---------------------------------------------------------------------------
# #161 (C): archive + forecast stitch
# ---------------------------------------------------------------------------


class TestStitchWeather:
    """Unit tests for _stitch_weather — boundary split + dedup."""

    @staticmethod
    def _df(start: str, n: int, temp: float) -> pd.DataFrame:
        ts = pd.date_range(start, periods=n, freq="h", tz="UTC")
        return pd.DataFrame({"timestamp": ts, "temperature_2m": [temp] * n})

    def test_archive_before_boundary_forecast_after(self):
        from data.weather_client import _stitch_weather

        boundary = pd.Timestamp("2026-05-20 23:00", tz="UTC")
        archive = self._df("2026-05-19 00:00", 48, temp=70.0)  # spans the boundary
        forecast = self._df("2026-05-20 00:00", 72, temp=99.0)  # spans + after

        out = _stitch_weather(archive, forecast, boundary)
        before = out[out["timestamp"] <= boundary]
        after = out[out["timestamp"] > boundary]
        # Everything <= boundary came from archive (70), everything after
        # from forecast (99).
        assert (before["temperature_2m"] == 70.0).all()
        assert (after["temperature_2m"] == 99.0).all()
        # No duplicate timestamps across the seam.
        assert out["timestamp"].is_unique
        assert out["timestamp"].is_monotonic_increasing

    def test_archive_none_returns_forecast_after_boundary(self):
        from data.weather_client import _stitch_weather

        boundary = pd.Timestamp("2026-05-20 23:00", tz="UTC")
        forecast = self._df("2026-05-21 00:00", 24, temp=88.0)
        out = _stitch_weather(None, forecast, boundary)
        assert len(out) == 24
        assert (out["temperature_2m"] == 88.0).all()

    def test_empty_inputs_return_empty_frame(self):
        from data.weather_client import _stitch_weather

        out = _stitch_weather(None, pd.DataFrame(), pd.Timestamp("2026-05-20", tz="UTC"))
        assert out.empty
        assert "timestamp" in out.columns


class TestFetchWeatherStitchOrchestration:
    """fetch_weather's archive-enrichment orchestration (#161)."""

    @patch("data.weather_client.get_cache")
    def test_archive_failure_degrades_to_forecast_only(self, mock_get_cache):
        """If the archive endpoint fails, keep the forecast result rather
        than dropping to the stale/GCS fallback — forecast is the data the
        model actually consumes."""
        from data.weather_client import fetch_weather

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        fc = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-05-25", periods=10, freq="h", tz="UTC"),
                "temperature_2m": [80.0] * 10,
            }
        )

        with (
            patch("data.weather_client._fetch_forecast_endpoint", return_value=fc),
            patch(
                "data.weather_client._fetch_archive_endpoint",
                side_effect=requests.ConnectionError("archive down"),
            ),
            patch("data.gcs_store.write_parquet"),
        ):
            result = fetch_weather("ERCOT", use_cache=False)

        # Forecast-only result, no exception, cached normally.
        assert not result.empty
        assert len(result) == 10
        mock_cache.set.assert_called_once()

    @patch("data.weather_client.get_cache")
    def test_combines_archive_history_with_forecast(self, mock_get_cache):
        """Happy path: deep archive history + recent/future forecast are
        stitched into one frame covering both."""
        from data.weather_client import fetch_weather

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        # Archive: 60 days ago up to ~now-5d. Forecast: ~now-7d into future.
        archive = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-03-01", periods=500, freq="h", tz="UTC"),
                "temperature_2m": [60.0] * 500,
            }
        )
        forecast = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-05-29", periods=200, freq="h", tz="UTC"),
                "temperature_2m": [95.0] * 200,
            }
        )

        with (
            patch("data.weather_client._fetch_forecast_endpoint", return_value=forecast),
            patch("data.weather_client._fetch_archive_endpoint", return_value=archive),
            patch("data.gcs_store.write_parquet"),
        ):
            result = fetch_weather("ERCOT", use_cache=False)

        # Result should contain BOTH the deep-history archive rows and the
        # recent/future forecast rows — far more than either alone could
        # provide under the degraded single-endpoint path.
        assert len(result) > 500
        assert result["timestamp"].is_unique
        # Has both temperature regimes (archive 60, forecast 95).
        temps = set(result["temperature_2m"].unique())
        assert 60.0 in temps and 95.0 in temps

    @patch("data.weather_client.get_cache")
    def test_forecast_failure_uses_fallback_chain(self, mock_get_cache):
        """If the FORECAST endpoint fails, the stale-cache fallback runs
        (archive is never reached) — preserves the original contract."""
        from data.weather_client import fetch_weather

        stale = pd.DataFrame({"timestamp": [1], "temperature_2m": [50.0]})
        mock_cache = MagicMock()
        mock_cache.get.side_effect = [None, stale]  # fresh miss, stale hit
        mock_get_cache.return_value = mock_cache

        with patch(
            "data.weather_client._fetch_forecast_endpoint",
            side_effect=requests.Timeout("forecast down"),
        ):
            result = fetch_weather("ERCOT")

        pd.testing.assert_frame_equal(result, stale)
        assert mock_cache.get.call_count == 2
        assert mock_cache.get.call_args_list[1][1] == {"allow_stale": True}
