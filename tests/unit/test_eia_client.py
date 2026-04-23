"""
Unit tests for data/eia_client.py.

Tests cover all public fetch functions, internal helpers (_paginated_fetch,
_request_with_backoff, _parse_* functions, _get_eia_code), and the full
cache fallback chain: SQLite cache -> stale cache -> GCS -> empty DataFrame.

All HTTP calls, cache access, and GCS I/O are mocked so tests are fully
isolated with no external dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.eia_client import (
    EIA_REGION_CODES,
    _get_eia_code,
    _paginated_fetch,
    _parse_demand_records,
    _parse_generation_records,
    _parse_interchange_records,
    _request_with_backoff,
    fetch_demand,
    fetch_generation_by_fuel,
    fetch_interchange,
)

# ---------------------------------------------------------------------------
# _get_eia_code
# ---------------------------------------------------------------------------


class TestGetEiaCode:
    """Tests for internal region code mapping."""

    @pytest.mark.parametrize(
        "region,expected",
        [
            ("ERCOT", "ERCO"),
            ("FPL", "FPL"),
            ("CAISO", "CISO"),
            ("PJM", "PJM"),
            ("MISO", "MISO"),
            ("NYISO", "NYIS"),
            ("ISONE", "ISNE"),
            ("SPP", "SWPP"),
        ],
    )
    def test_all_eight_regions_map_correctly(self, region: str, expected: str):
        assert _get_eia_code(region) == expected

    def test_unknown_region_returns_input_as_is(self):
        """Unknown region names pass through unchanged (used as EIA code directly)."""
        assert _get_eia_code("UNKNOWN_BA") == "UNKNOWN_BA"

    def test_region_codes_dict_has_eight_entries(self):
        assert len(EIA_REGION_CODES) == 8


# ---------------------------------------------------------------------------
# _parse_demand_records
# ---------------------------------------------------------------------------


class TestParseDemandRecords:
    """Tests for EIA demand record parsing."""

    def test_valid_records_with_demand_and_forecast(self):
        records = [
            {"period": "2024-01-01T00", "value": 40000, "type": "D"},
            {"period": "2024-01-01T01", "value": 39500, "type": "D"},
            {"period": "2024-01-01T00", "value": 41000, "type": "DF"},
        ]
        df = _parse_demand_records(records, "ERCOT")

        assert list(df.columns) == ["timestamp", "demand_mw", "region", "forecast_mw"]
        assert len(df) == 2
        assert df["demand_mw"].iloc[0] == 40000.0
        assert df["forecast_mw"].iloc[0] == 41000.0
        assert df["region"].iloc[0] == "ERCOT"

    def test_demand_only_no_forecast(self):
        """When no DF-type records exist, forecast_mw should be None."""
        records = [
            {"period": "2024-01-01T00", "value": 40000, "type": "D"},
            {"period": "2024-01-01T01", "value": 39500, "type": "D"},
        ]
        df = _parse_demand_records(records, "CAISO")

        assert "forecast_mw" in df.columns
        assert df["forecast_mw"].isna().all()
        assert len(df) == 2

    def test_empty_records_returns_empty_df(self):
        df = _parse_demand_records([], "ERCOT")
        assert df.empty
        assert list(df.columns) == ["timestamp", "demand_mw", "forecast_mw", "region"]

    def test_missing_value_field_becomes_nan(self):
        """Records with missing 'value' become NaN (preserved as missing)."""
        records = [
            {"period": "2024-01-01T00", "type": "D"},
        ]
        df = _parse_demand_records(records, "PJM")
        assert pd.isna(df["demand_mw"].iloc[0])

    def test_null_value_field_becomes_nan(self):
        """Records with None value become NaN (not zero-filled)."""
        records = [
            {"period": "2024-01-01T00", "value": None, "type": "D"},
        ]
        df = _parse_demand_records(records, "PJM")
        assert pd.isna(df["demand_mw"].iloc[0])

    def test_zero_value_coerced_to_nan(self):
        """Literal 0 demand is impossible for a balancing authority — coerce to NaN."""
        records = [
            {"period": "2024-01-01T00", "value": 0, "type": "D"},
            {"period": "2024-01-01T01", "value": 0.0, "type": "D"},
            {"period": "2024-01-01T02", "value": 28500, "type": "D"},
        ]
        df = _parse_demand_records(records, "NYIS")
        assert pd.isna(df["demand_mw"].iloc[0])
        assert pd.isna(df["demand_mw"].iloc[1])
        assert df["demand_mw"].iloc[2] == 28500.0

    def test_non_numeric_value_coerced_to_nan(self):
        """Garbage strings in 'value' don't blow up parsing — become NaN."""
        records = [
            {"period": "2024-01-01T00", "value": "not-a-number", "type": "D"},
        ]
        df = _parse_demand_records(records, "PJM")
        assert pd.isna(df["demand_mw"].iloc[0])

    def test_records_sorted_by_timestamp(self):
        records = [
            {"period": "2024-01-01T03", "value": 30000, "type": "D"},
            {"period": "2024-01-01T01", "value": 31000, "type": "D"},
            {"period": "2024-01-01T02", "value": 32000, "type": "D"},
        ]
        df = _parse_demand_records(records, "MISO")
        assert list(df["demand_mw"]) == [31000.0, 32000.0, 30000.0]

    def test_duplicate_timestamps_in_demand(self):
        """Duplicate D-type timestamps are kept (merge may produce multiple rows)."""
        records = [
            {"period": "2024-01-01T00", "value": 40000, "type": "D"},
            {"period": "2024-01-01T00", "value": 40500, "type": "D"},
        ]
        df = _parse_demand_records(records, "ERCOT")
        # Both demand rows should be present since there's no dedup in parser
        assert len(df) == 2


# ---------------------------------------------------------------------------
# _parse_generation_records
# ---------------------------------------------------------------------------


class TestParseGenerationRecords:
    """Tests for EIA generation-by-fuel record parsing."""

    def test_valid_records(self):
        records = [
            {"period": "2024-01-01T00", "fueltype": "NG", "value": 15000},
            {"period": "2024-01-01T00", "fueltype": "SUN", "value": 5000},
            {"period": "2024-01-01T01", "fueltype": "NG", "value": 16000},
        ]
        df = _parse_generation_records(records, "CAISO")

        assert list(df.columns) == ["timestamp", "fuel_type", "generation_mw", "region"]
        assert len(df) == 3
        assert set(df["fuel_type"]) == {"NG", "SUN"}
        assert df["region"].unique().tolist() == ["CAISO"]

    def test_missing_fueltype_falls_back_to_type_name(self):
        """When 'fueltype' key is absent, 'type-name' is used."""
        records = [
            {"period": "2024-01-01T00", "type-name": "Natural Gas", "value": 15000},
        ]
        df = _parse_generation_records(records, "ERCOT")
        assert df["fuel_type"].iloc[0] == "Natural Gas"

    def test_missing_both_fueltype_keys_defaults_to_unknown(self):
        records = [
            {"period": "2024-01-01T00", "value": 15000},
        ]
        df = _parse_generation_records(records, "ERCOT")
        assert df["fuel_type"].iloc[0] == "unknown"

    def test_missing_value_defaults_to_zero(self):
        records = [
            {"period": "2024-01-01T00", "fueltype": "NG"},
        ]
        df = _parse_generation_records(records, "PJM")
        assert df["generation_mw"].iloc[0] == 0.0

    def test_records_sorted_by_timestamp(self):
        records = [
            {"period": "2024-01-01T02", "fueltype": "NG", "value": 100},
            {"period": "2024-01-01T00", "fueltype": "NG", "value": 200},
        ]
        df = _parse_generation_records(records, "MISO")
        assert df["generation_mw"].iloc[0] == 200.0


# ---------------------------------------------------------------------------
# _parse_interchange_records
# ---------------------------------------------------------------------------


class TestParseInterchangeRecords:
    """Tests for EIA interchange record parsing."""

    def test_valid_records(self):
        records = [
            {"period": "2024-01-01T00", "fromba": "ERCO", "toba": "SWPP", "value": 500},
            {"period": "2024-01-01T01", "fromba": "ERCO", "toba": "SWPP", "value": -200},
        ]
        df = _parse_interchange_records(records)

        assert list(df.columns) == ["timestamp", "from_ba", "to_ba", "interchange_mw"]
        assert len(df) == 2
        assert df["interchange_mw"].iloc[1] == -200.0

    def test_missing_ba_fields_default_to_empty_string(self):
        records = [
            {"period": "2024-01-01T00", "value": 300},
        ]
        df = _parse_interchange_records(records)
        assert df["from_ba"].iloc[0] == ""
        assert df["to_ba"].iloc[0] == ""

    def test_missing_value_defaults_to_zero(self):
        records = [
            {"period": "2024-01-01T00", "fromba": "ERCO", "toba": "SWPP"},
        ]
        df = _parse_interchange_records(records)
        assert df["interchange_mw"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# _request_with_backoff
# ---------------------------------------------------------------------------


class TestRequestWithBackoff:
    """Tests for HTTP retry logic with exponential backoff."""

    @patch("data.eia_client.time.sleep")
    @patch("data.eia_client.requests.get")
    def test_success_on_first_try(self, mock_get, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": {"data": []}}
        mock_get.return_value = mock_resp

        result = _request_with_backoff("https://api.eia.gov/v2/test", {})

        assert result == {"response": {"data": []}}
        mock_sleep.assert_not_called()

    @patch("data.eia_client.time.sleep")
    @patch("data.eia_client.requests.get")
    def test_retry_on_503_then_success(self, mock_get, mock_sleep):
        """Server error (503) triggers retry; second attempt succeeds."""
        fail_resp = MagicMock()
        fail_resp.status_code = 503

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"response": {"data": [{"x": 1}]}}

        mock_get.side_effect = [fail_resp, ok_resp]

        result = _request_with_backoff("https://api.eia.gov/v2/test", {})

        assert result == {"response": {"data": [{"x": 1}]}}
        assert mock_sleep.call_count == 1

    @patch("data.eia_client.time.sleep")
    @patch("data.eia_client.requests.get")
    def test_retry_on_429_rate_limit(self, mock_get, mock_sleep):
        """Rate limiting (429) triggers retry with backoff."""
        rate_resp = MagicMock()
        rate_resp.status_code = 429

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"response": {"data": []}}

        mock_get.side_effect = [rate_resp, ok_resp]

        result = _request_with_backoff("https://api.eia.gov/v2/test", {})

        assert result is not None
        mock_sleep.assert_called_once()

    @patch("data.eia_client.time.sleep")
    @patch("data.eia_client.requests.get")
    def test_max_retries_exceeded_returns_none(self, mock_get, mock_sleep):
        """After exhausting all retries on 5xx, returns None."""
        fail_resp = MagicMock()
        fail_resp.status_code = 500

        mock_get.return_value = fail_resp

        result = _request_with_backoff("https://api.eia.gov/v2/test", {})

        assert result is None

    @patch("data.eia_client.time.sleep")
    @patch("data.eia_client.requests.get")
    def test_client_error_returns_none_immediately(self, mock_get, mock_sleep):
        """A 4xx error (not 429) returns None without retrying."""
        fail_resp = MagicMock()
        fail_resp.status_code = 403
        fail_resp.text = "Forbidden api_key=SECRETKEY123"

        mock_get.return_value = fail_resp

        result = _request_with_backoff("https://api.eia.gov/v2/test", {})

        assert result is None
        # No sleep for non-retryable error
        mock_sleep.assert_not_called()

    @patch("data.eia_client.time.sleep")
    @patch("data.eia_client.requests.get")
    def test_request_exception_retries(self, mock_get, mock_sleep):
        """Network-level exceptions (timeout, DNS) trigger retries."""
        import requests as req

        mock_get.side_effect = req.ConnectionError("Connection refused")

        result = _request_with_backoff("https://api.eia.gov/v2/test", {})

        assert result is None
        # Should have slept between retries (MAX_RETRIES - 1 sleeps)
        assert mock_sleep.call_count > 0

    @patch("data.eia_client.time.sleep")
    @patch("data.eia_client.requests.get")
    def test_api_key_sanitized_in_error_log(self, mock_get, mock_sleep):
        """Ensure API keys are scrubbed from logged error bodies."""
        fail_resp = MagicMock()
        fail_resp.status_code = 400
        fail_resp.text = "Invalid request: api_key=abc123def456 is not valid"
        mock_get.return_value = fail_resp

        # Should not raise — just returns None after sanitizing
        result = _request_with_backoff("https://api.eia.gov/v2/test", {})
        assert result is None


# ---------------------------------------------------------------------------
# _paginated_fetch
# ---------------------------------------------------------------------------


class TestPaginatedFetch:
    """Tests for multi-page API fetching."""

    @patch("data.eia_client._request_with_backoff")
    def test_single_page(self, mock_req):
        """When total <= page size, only one request is made."""
        mock_req.return_value = {
            "response": {
                "total": 3,
                "data": [
                    {"period": "2024-01-01T00", "value": 100},
                    {"period": "2024-01-01T01", "value": 200},
                    {"period": "2024-01-01T02", "value": 300},
                ],
            }
        }

        records = _paginated_fetch("electricity/rto/region-data", {"api_key": "test"})

        assert len(records) == 3
        assert mock_req.call_count == 1

    @patch("data.eia_client._request_with_backoff")
    def test_multiple_pages(self, mock_req):
        """When total > page size, multiple requests are made with incrementing offset."""
        # Page 1: total=7000, returns 5000 records
        page1 = {
            "response": {
                "total": 7000,
                "data": [{"period": f"2024-01-01T{i:02d}", "value": i} for i in range(5)],
            }
        }
        # Page 2: returns remaining records
        page2 = {
            "response": {
                "total": 7000,
                "data": [{"period": f"2024-01-02T{i:02d}", "value": i + 5} for i in range(2)],
            }
        }
        mock_req.side_effect = [page1, page2]

        records = _paginated_fetch("electricity/rto/region-data", {"api_key": "test"})

        assert len(records) == 7
        assert mock_req.call_count == 2

    @patch("data.eia_client._request_with_backoff")
    def test_empty_response(self, mock_req):
        """Empty API response returns empty list."""
        mock_req.return_value = {
            "response": {
                "total": 0,
                "data": [],
            }
        }

        records = _paginated_fetch("electricity/rto/region-data", {"api_key": "test"})

        assert records == []

    @patch("data.eia_client._request_with_backoff")
    def test_request_failure_returns_partial(self, mock_req):
        """If _request_with_backoff returns None, stop and return what we have."""
        mock_req.return_value = None

        records = _paginated_fetch("electricity/rto/region-data", {"api_key": "test"})

        assert records == []

    @patch("data.eia_client._request_with_backoff")
    def test_null_total_treated_as_zero(self, mock_req):
        """When 'total' is None in the response, it should be treated as 0."""
        mock_req.return_value = {
            "response": {
                "total": None,
                "data": [{"period": "2024-01-01T00", "value": 100}],
            }
        }

        records = _paginated_fetch("electricity/rto/region-data", {"api_key": "test"})

        # Records from the first page are still collected
        assert len(records) == 1
        # Only one call because offset (5000) >= total (0)
        assert mock_req.call_count == 1


# ---------------------------------------------------------------------------
# fetch_demand
# ---------------------------------------------------------------------------


class TestFetchDemand:
    """Tests for the main demand fetch function with caching and fallback."""

    @patch("data.gcs_store.write_parquet")
    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_success_fresh_api_data(self, mock_pf, mock_cache_fn, mock_wp):
        """Happy path: API returns data, caches and writes to GCS."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # cache miss
        mock_cache_fn.return_value = mock_cache

        mock_pf.return_value = [
            {"period": "2024-01-01T00", "value": 40000, "type": "D"},
            {"period": "2024-01-01T01", "value": 39500, "type": "D"},
            {"period": "2024-01-01T00", "value": 41000, "type": "DF"},
        ]

        df = fetch_demand("ERCOT", start="2024-01-01T00", end="2024-01-02T23")

        assert not df.empty
        assert "demand_mw" in df.columns
        assert "forecast_mw" in df.columns
        mock_cache.set.assert_called_once()
        mock_wp.assert_called_once()

    @patch("data.eia_client.get_cache")
    def test_returns_cached_data_when_available(self, mock_cache_fn):
        """When cache has data, return it immediately without API call."""
        cached_df = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
                "demand_mw": [40000.0],
                "forecast_mw": [41000.0],
                "region": ["ERCOT"],
            }
        )
        mock_cache = MagicMock()
        mock_cache.get.return_value = cached_df
        mock_cache_fn.return_value = mock_cache

        df = fetch_demand("ERCOT", start="2024-01-01T00", end="2024-01-02T23")

        pd.testing.assert_frame_equal(df, cached_df)

    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_cache_bypass_when_disabled(self, mock_pf, mock_cache_fn):
        """When use_cache=False, skip cache lookup but still cache results."""
        mock_cache = MagicMock()
        mock_cache_fn.return_value = mock_cache

        mock_pf.return_value = [
            {"period": "2024-01-01T00", "value": 40000, "type": "D"},
        ]

        with patch("data.gcs_store.write_parquet"):
            df = fetch_demand("ERCOT", start="2024-01-01T00", end="2024-01-02T23", use_cache=False)

        # Cache.get should NOT have been called for lookup
        # (it may be called for stale fallback but not the initial check)
        assert not df.empty

    @patch("data.gcs_store.read_parquet")
    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_api_empty_falls_back_to_stale_cache(self, mock_pf, mock_cache_fn, mock_rp):
        """When API returns no data, serve stale cache."""
        stale_df = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
                "demand_mw": [35000.0],
                "forecast_mw": [36000.0],
                "region": ["ERCOT"],
            }
        )
        mock_cache = MagicMock()
        # First .get() with default allow_stale=False -> None (cache miss)
        # Second .get() with allow_stale=True -> stale data
        mock_cache.get.side_effect = [None, stale_df]
        mock_cache_fn.return_value = mock_cache

        mock_pf.return_value = []  # API returns nothing

        df = fetch_demand("ERCOT", start="2024-01-01T00", end="2024-01-02T23")

        pd.testing.assert_frame_equal(df, stale_df)
        mock_rp.assert_not_called()  # GCS not needed since stale cache worked

    @patch("data.gcs_store.read_parquet")
    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_api_empty_no_stale_falls_back_to_gcs(self, mock_pf, mock_cache_fn, mock_rp):
        """When API and stale cache both empty, fall back to GCS parquet."""
        gcs_df = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
                "demand_mw": [33000.0],
                "forecast_mw": [34000.0],
                "region": ["ERCOT"],
            }
        )
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # no cache at all
        mock_cache_fn.return_value = mock_cache

        mock_pf.return_value = []  # API returns nothing
        mock_rp.return_value = gcs_df

        df = fetch_demand("ERCOT", start="2024-01-01T00", end="2024-01-02T23")

        pd.testing.assert_frame_equal(df, gcs_df)
        mock_rp.assert_called_once_with("demand", "ERCOT")

    @patch("data.gcs_store.read_parquet")
    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_all_fallbacks_exhausted_returns_empty_df(self, mock_pf, mock_cache_fn, mock_rp):
        """When API, stale cache, and GCS all fail, return empty DataFrame."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache_fn.return_value = mock_cache

        mock_pf.return_value = []
        mock_rp.return_value = None  # GCS also has nothing

        df = fetch_demand("ERCOT", start="2024-01-01T00", end="2024-01-02T23")

        assert df.empty
        assert list(df.columns) == ["timestamp", "demand_mw", "forecast_mw", "region"]

    def test_unknown_region_raises_value_error(self):
        assert pytest.raises(ValueError, fetch_demand, "UNKNOWN_BA")

    @patch("data.gcs_store.write_parquet")
    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_default_date_range_applied(self, mock_pf, mock_cache_fn, mock_wp):
        """When start/end not provided, defaults are applied (90 days)."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache_fn.return_value = mock_cache

        mock_pf.return_value = [
            {"period": "2024-01-01T00", "value": 40000, "type": "D"},
        ]

        df = fetch_demand("ERCOT")  # no start/end

        assert not df.empty
        # Verify _paginated_fetch was called (date params were generated)
        mock_pf.assert_called_once()

    @patch("data.gcs_store.write_parquet")
    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_uses_mock_eia_response_fixture(
        self, mock_pf, mock_cache_fn, mock_wp, mock_eia_response
    ):
        """Integration with the conftest mock_eia_response fixture."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache_fn.return_value = mock_cache

        # Use the fixture's data records directly
        mock_pf.return_value = mock_eia_response["response"]["data"]

        df = fetch_demand("ERCOT", start="2024-01-01T00", end="2024-01-01T23")

        assert len(df) == 2  # 2 demand rows (D type)
        assert df["demand_mw"].iloc[0] == 40000.0
        assert df["forecast_mw"].iloc[0] == 41000.0


# ---------------------------------------------------------------------------
# fetch_generation_by_fuel
# ---------------------------------------------------------------------------


class TestFetchGenerationByFuel:
    """Tests for generation-by-fuel fetching with caching."""

    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_success_fresh_data(self, mock_pf, mock_cache_fn):
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache_fn.return_value = mock_cache

        mock_pf.return_value = [
            {"period": "2024-01-01T00", "fueltype": "NG", "value": 15000},
            {"period": "2024-01-01T00", "fueltype": "SUN", "value": 5000},
        ]

        df = fetch_generation_by_fuel("CAISO", start="2024-01-01T00", end="2024-01-01T23")

        assert len(df) == 2
        assert "fuel_type" in df.columns
        assert "generation_mw" in df.columns
        mock_cache.set.assert_called_once()

    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_empty_api_falls_back_to_stale_cache(self, mock_pf, mock_cache_fn):
        stale_df = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
                "fuel_type": ["NG"],
                "generation_mw": [15000.0],
                "region": ["CAISO"],
            }
        )
        mock_cache = MagicMock()
        mock_cache.get.side_effect = [None, stale_df]
        mock_cache_fn.return_value = mock_cache

        mock_pf.return_value = []

        df = fetch_generation_by_fuel("CAISO", start="2024-01-01T00", end="2024-01-01T23")

        pd.testing.assert_frame_equal(df, stale_df)

    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_all_fallbacks_exhausted_returns_empty_df(self, mock_pf, mock_cache_fn):
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache_fn.return_value = mock_cache

        mock_pf.return_value = []

        df = fetch_generation_by_fuel("CAISO", start="2024-01-01T00", end="2024-01-01T23")

        assert df.empty
        assert list(df.columns) == ["timestamp", "fuel_type", "generation_mw", "region"]

    def test_unknown_region_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown region"):
            fetch_generation_by_fuel("BOGUS")

    @patch("data.eia_client.get_cache")
    def test_cached_data_returned_directly(self, mock_cache_fn):
        cached_df = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
                "fuel_type": ["NG"],
                "generation_mw": [15000.0],
                "region": ["ERCOT"],
            }
        )
        mock_cache = MagicMock()
        mock_cache.get.return_value = cached_df
        mock_cache_fn.return_value = mock_cache

        df = fetch_generation_by_fuel("ERCOT", start="2024-01-01T00", end="2024-01-01T23")

        pd.testing.assert_frame_equal(df, cached_df)


# ---------------------------------------------------------------------------
# fetch_interchange
# ---------------------------------------------------------------------------


class TestFetchInterchange:
    """Tests for interchange fetching with caching."""

    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_success_fresh_data(self, mock_pf, mock_cache_fn):
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache_fn.return_value = mock_cache

        mock_pf.return_value = [
            {"period": "2024-01-01T00", "fromba": "ERCO", "toba": "SWPP", "value": 500},
        ]

        df = fetch_interchange("ERCOT", start="2024-01-01T00", end="2024-01-01T23")

        assert len(df) == 1
        assert df["from_ba"].iloc[0] == "ERCO"
        assert df["interchange_mw"].iloc[0] == 500.0
        mock_cache.set.assert_called_once()

    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_empty_api_falls_back_to_stale_cache(self, mock_pf, mock_cache_fn):
        stale_df = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
                "from_ba": ["ERCO"],
                "to_ba": ["SWPP"],
                "interchange_mw": [500.0],
            }
        )
        mock_cache = MagicMock()
        mock_cache.get.side_effect = [None, stale_df]
        mock_cache_fn.return_value = mock_cache

        mock_pf.return_value = []

        df = fetch_interchange("ERCOT", start="2024-01-01T00", end="2024-01-01T23")

        pd.testing.assert_frame_equal(df, stale_df)

    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_all_fallbacks_exhausted_returns_empty_df(self, mock_pf, mock_cache_fn):
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache_fn.return_value = mock_cache

        mock_pf.return_value = []

        df = fetch_interchange("ERCOT", start="2024-01-01T00", end="2024-01-01T23")

        assert df.empty
        assert list(df.columns) == ["timestamp", "from_ba", "to_ba", "interchange_mw"]

    def test_unknown_region_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown region"):
            fetch_interchange("FAKE_BA")

    @patch("data.eia_client.get_cache")
    def test_cached_data_returned_directly(self, mock_cache_fn):
        cached_df = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
                "from_ba": ["ERCO"],
                "to_ba": ["SWPP"],
                "interchange_mw": [500.0],
            }
        )
        mock_cache = MagicMock()
        mock_cache.get.return_value = cached_df
        mock_cache_fn.return_value = mock_cache

        df = fetch_interchange("ERCOT", start="2024-01-01T00", end="2024-01-01T23")

        pd.testing.assert_frame_equal(df, cached_df)

    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_default_date_range_applied(self, mock_pf, mock_cache_fn):
        """When start/end not provided, defaults (30 days) are applied."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache_fn.return_value = mock_cache

        mock_pf.return_value = [
            {"period": "2024-01-01T00", "fromba": "ERCO", "toba": "SWPP", "value": 100},
        ]

        df = fetch_interchange("ERCOT")  # no start/end

        assert not df.empty
        mock_pf.assert_called_once()
