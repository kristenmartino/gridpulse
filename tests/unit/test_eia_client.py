"""
Unit tests for data/eia_client.py.

Tests cover all public fetch functions, internal helpers (_paginated_fetch,
_request_with_backoff, _parse_* functions, _get_eia_code), and the full
cache fallback chain: SQLite cache -> stale cache -> GCS -> empty DataFrame.

All HTTP calls, cache access, and GCS I/O are mocked so tests are fully
isolated with no external dependencies.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import requests

from data import eia_client
from data.eia_client import (
    _EMPTY_COLUMN_PROTOTYPES,
    EIA_CIRCUIT_PROBE_INTERVAL,
    EIA_CIRCUIT_TRIP_THRESHOLD,
    EIA_PAGE_SIZE,
    EIA_REGION_CODES,
    MAX_RETRIES,
    EIAIncompleteFetchError,
    _circuit_breaker,
    _EIACircuitBreaker,
    _get_eia_code,
    _paginated_fetch,
    _parse_demand_records,
    _parse_generation_records,
    _parse_interchange_records,
    _request_with_backoff,
    _typed_empty,
    fetch_demand,
    fetch_generation_by_fuel,
    fetch_interchange,
)


@pytest.fixture(autouse=True)
def _reset_eia_circuit_breaker():
    """Reset the module-level EIA circuit breaker around every test.

    The breaker carries process-local state; without this, a test that
    exhausts retries would leak failure counts into later tests.
    """
    _circuit_breaker.reset()
    yield
    _circuit_breaker.reset()


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

    def test_region_codes_dict_has_fifty_one_entries(self):
        assert len(EIA_REGION_CODES) == 51


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
        """When no DF-type records exist, forecast_mw is all-missing."""
        records = [
            {"period": "2024-01-01T00", "value": 40000, "type": "D"},
            {"period": "2024-01-01T01", "value": 39500, "type": "D"},
        ]
        df = _parse_demand_records(records, "CAISO")

        assert "forecast_mw" in df.columns
        assert df["forecast_mw"].isna().all()
        assert len(df) == 2

    def test_the_frame_is_numeric_whether_or_not_a_forecast_arrived(self):
        """Both branches must produce float64, not one float and one object.

        `isna().all()` above is true for a column of `None` *and* a column of
        `NaN`, so it cannot tell the two apart — and the difference matters.
        Assigning `None` builds an **object** column, and `np.isfinite` raises
        on an object array:

            TypeError: ufunc 'isfinite' not supported for the input types

        Everything downstream currently survives it, but only because five
        separate places each defend themselves — `pd.to_numeric(errors=
        "coerce")` in `data/quality.py` and `data/vintage.py`, a `try/except`
        in `models/benchmark.py`, an `is None` check in `jobs/phases.py`, and
        `np.asarray(..., dtype=float)` in the metric helpers. Drop any one of
        those and this shape crashes it.

        Mutation testing is how this surfaced: ~81 survivors across the
        codebase are precisely those defensive coercions, unpinned because
        nothing reachable was ever object-dtype **except** by way of the
        no-forecast branch here. Emitting the right dtype at the source is
        cheaper than pinning all of them.
        """
        with_forecast = _parse_demand_records(
            [
                {"period": "2024-01-01T00", "value": 40000, "type": "D"},
                {"period": "2024-01-01T00", "value": 41000, "type": "DF"},
            ],
            "ERCOT",
        )
        without_forecast = _parse_demand_records(
            [{"period": "2024-01-01T00", "value": 40000, "type": "D"}], "ERCOT"
        )

        for label, frame in (("with DF", with_forecast), ("without DF", without_forecast)):
            for col in ("demand_mw", "forecast_mw"):
                assert frame[col].dtype == "float64", f"{label}: {col} is {frame[col].dtype}"

        # The property those dtypes buy: numpy can be handed the column raw.
        assert not np.isfinite(np.asarray(without_forecast["forecast_mw"])).any()

    def test_empty_records_returns_empty_df(self):
        df = _parse_demand_records([], "ERCOT")
        assert df.empty
        assert list(df.columns) == ["timestamp", "demand_mw", "forecast_mw", "region"]
        # `.empty` and `list(df.columns)` are both true of an all-object frame,
        # so on their own they cannot see the dtype — the same blind spot that
        # let `isna().all()` miss the no-forecast branch in #434. Assert the
        # dtype explicitly or this test passes on the shape it exists to reject.
        assert df["demand_mw"].dtype == "float64"
        assert df["forecast_mw"].dtype == "float64"

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
# _typed_empty
# ---------------------------------------------------------------------------


#: ``(empty_cols, parser, sample_records)`` for each endpoint that owns an
#: ``empty_cols`` list, so the parity test below compares each typed-empty
#: frame against the frame its **own** parser builds from real records.
_ENDPOINT_CASES = [
    pytest.param(
        ["timestamp", "demand_mw", "forecast_mw", "region"],
        lambda recs: _parse_demand_records(recs, "ERCOT"),
        [
            {"period": "2024-01-01T00", "value": 40000, "type": "D"},
            {"period": "2024-01-01T00", "value": 41000, "type": "DF"},
        ],
        id="demand",
    ),
    pytest.param(
        ["timestamp", "fuel_type", "generation_mw", "region"],
        lambda recs: _parse_generation_records(recs, "ERCOT"),
        [{"period": "2024-01-01T00", "fueltype": "WND", "value": 5000}],
        id="generation",
    ),
    pytest.param(
        ["timestamp", "from_ba", "to_ba", "interchange_mw"],
        _parse_interchange_records,
        [{"period": "2024-01-01T00", "fromba": "ERCO", "toba": "SWPP", "value": 250}],
        id="interchange",
    ),
]


class TestTypedEmptyFrames:
    """The #174 fallback chain ends in a *typed* empty frame.

    ``pd.DataFrame(columns=[...])`` builds every column as object, and an
    object column of numbers is not a float column: ``np.isfinite`` raises on
    it. Because this frame is only returned when a fetch has failed **and**
    both the stale cache and the GCS parquet have missed, anything it breaks
    breaks during an outage — the worst time to be debugging a dtype.
    """

    @pytest.mark.parametrize("cols,parser,records", _ENDPOINT_CASES)
    def test_dtypes_match_the_real_parser(self, cols, parser, records):
        """Parity with real output, rather than hardcoded dtype strings.

        Asserting ``== "float64"`` and ``== "str"`` would pin *today's* pandas:
        3.0 gives ``str``/``datetime64[us, UTC]`` where 2.x gives
        ``object``/``[ns]``. Comparing against what the parser actually
        produces keeps the invariant that matters — the two agree — and lets
        both sides move together on a pandas bump.
        """
        real = parser(records)
        empty = _typed_empty(cols)
        assert list(empty.columns) == cols
        for col in cols:
            assert empty[col].dtype == real[col].dtype, (
                f"{col}: empty is {empty[col].dtype}, real parse is {real[col].dtype}"
            )

    @pytest.mark.parametrize("cols,parser,records", _ENDPOINT_CASES)
    def test_numeric_columns_survive_numpy(self, cols, parser, records):
        """The property the dtype buys: hand the column to numpy unguarded.

        This is the assertion that actually fails on the old construction —
        ``np.isfinite`` on an object array raises ``TypeError`` regardless of
        length, so a zero-row object frame crashes a consumer that a zero-row
        float frame passes straight through.
        """
        empty = _typed_empty(cols)
        for col in cols:
            if not str(col).endswith("_mw"):
                continue
            assert not np.isfinite(np.asarray(empty[col])).any()

    def test_still_empty_and_ordered(self):
        """Typing the frame must not change what callers already rely on."""
        cols = ["timestamp", "demand_mw", "forecast_mw", "region"]
        empty = _typed_empty(cols)
        assert empty.empty
        assert len(empty) == 0
        assert list(empty.columns) == cols

    def test_every_empty_cols_list_is_typed(self):
        """Guard the fail-open escape hatch in ``_typed_empty``.

        An unregistered column degrades to the old untyped frame rather than
        raising, because raising inside the outage fallback would convert a
        degraded fetch into a hard failure. That is the right runtime
        behaviour and the wrong thing to leave unwatched: without this test a
        new endpoint would silently reintroduce object dtype. Read the
        ``empty_cols=`` lists out of the source so adding one to the client
        without a prototype fails here.
        """
        source = Path(eia_client.__file__).read_text()
        declared = re.findall(r"empty_cols=\[([^\]]*)\]", source)
        assert declared, "no empty_cols= lists found — this guard has gone blind"
        for group in declared:
            for col in re.findall(r'"([^"]+)"', group):
                assert col in _EMPTY_COLUMN_PROTOTYPES, (
                    f"{col!r} is passed as an empty_cols entry but has no prototype in "
                    f"_EMPTY_COLUMN_PROTOTYPES, so _typed_empty falls back to an "
                    f"object-dtype frame for it. Add a prototype value."
                )

    def test_unknown_column_fails_open_instead_of_raising(self):
        """A missing prototype must not raise — this runs during an outage."""
        empty = _typed_empty(["timestamp", "not_a_real_column"])
        assert empty.empty
        assert list(empty.columns) == ["timestamp", "not_a_real_column"]


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

    def test_missing_value_preserved_as_nan(self):
        """P2-08 (#273): EIA nulls must never fabricate a 0 MW reading —
        the old null→0.0 coercion deflated renewable share and filled the
        fuel-mix pivot with fake zeros."""
        import numpy as np

        records = [
            {"period": "2024-01-01T00", "fueltype": "NG"},
            {"period": "2024-01-01T01", "fueltype": "NG", "value": None},
            {"period": "2024-01-01T02", "fueltype": "NG", "value": ""},
            {"period": "2024-01-01T03", "fueltype": "NG", "value": "not-a-number"},
        ]
        df = _parse_generation_records(records, "PJM")
        assert np.isnan(df["generation_mw"]).all()

    def test_true_zero_reading_is_preserved(self):
        """Unlike the demand parser, a literal 0 is legitimate here — a fuel
        type can genuinely produce nothing for an hour. No 0→NaN coercion."""
        records = [
            {"period": "2024-01-01T00", "fueltype": "SUN", "value": 0},
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

    def test_missing_value_preserved_as_nan(self):
        """P2-08 (#273): a null interchange reading is missing data, not a
        0 MW flow — preserving NaN makes the sparse-data dropna contract in
        jobs/phases.py work as documented (net_mw=None → UI renders "—")."""
        import numpy as np

        records = [
            {"period": "2024-01-01T00", "fromba": "ERCO", "toba": "SWPP"},
        ]
        df = _parse_interchange_records(records)
        assert np.isnan(df["interchange_mw"].iloc[0])

    def test_true_zero_flow_is_preserved(self):
        """A tie can genuinely sit at zero flow — no 0→NaN coercion here."""
        records = [
            {"period": "2024-01-01T00", "fromba": "ERCO", "toba": "SWPP", "value": 0},
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

    @patch("data.eia_client._request_with_backoff")
    def test_mid_pagination_failure_raises_incomplete(self, mock_req):
        """#269 / P2-06: a page hard-failing while ``total`` says rows remain is a
        truncation, not a clean end — raise so the caller falls back rather than
        caching the partial series."""
        # Page 1 succeeds with a full page (total says a second page is due),
        # then page 2 hard-fails (None).
        full_page = {
            "response": {
                "total": EIA_PAGE_SIZE + 100,
                "data": [
                    {"period": f"2024-01-01T{i:02d}", "value": i, "type": "D"}
                    for i in range(EIA_PAGE_SIZE)
                ],
            }
        }
        mock_req.side_effect = [full_page, None]

        with pytest.raises(EIAIncompleteFetchError):
            _paginated_fetch("electricity/rto/region-data", {"api_key": "test"})
        assert mock_req.call_count == 2

    @patch("data.eia_client._request_with_backoff")
    def test_first_page_failure_is_not_truncation(self, mock_req):
        """A first-page failure (nothing accumulated, total still 0) is the
        ordinary empty case — returns [] without raising EIAIncompleteFetchError."""
        mock_req.return_value = None

        records = _paginated_fetch("electricity/rto/region-data", {"api_key": "test"})

        assert records == []

    @patch("data.eia_client._request_with_backoff")
    def test_complete_multipage_fetch_does_not_raise(self, mock_req):
        """A cleanly completed multi-page fetch (every page a real 200, reaching
        the reported total) returns all rows and does NOT raise — the truncation
        gate is scoped to hard failures, not normal pagination."""
        page1 = {
            "response": {
                "total": EIA_PAGE_SIZE + 2,
                "data": [
                    {"period": f"2024-01-01T{i:02d}", "value": i, "type": "D"}
                    for i in range(EIA_PAGE_SIZE)
                ],
            }
        }
        page2 = {
            "response": {
                "total": EIA_PAGE_SIZE + 2,
                "data": [
                    {"period": "2024-02-01T00", "value": 1, "type": "D"},
                    {"period": "2024-02-01T01", "value": 2, "type": "D"},
                ],
            }
        }
        mock_req.side_effect = [page1, page2]

        records = _paginated_fetch("electricity/rto/region-data", {"api_key": "test"})

        assert len(records) == EIA_PAGE_SIZE + 2
        assert mock_req.call_count == 2

    @patch("data.eia_client._request_with_backoff")
    def test_empty_mid_page_is_not_flagged_as_truncation(self, mock_req):
        """Accepted-limitation pin (see _paginated_fetch comment): a valid 200
        with an *empty* data array mid-pagination is indistinguishable from EIA
        over-reporting ``total`` with data ending on a page boundary, so it is
        deliberately NOT raised as EIAIncompleteFetchError — the loop exits via
        ``not records`` and returns what it has. This test documents that chosen
        tradeoff so a future change doesn't 'fix' it and reintroduce false-positive
        fallbacks on the common over-counted-total quirk."""
        page1 = {
            "response": {
                "total": 3 * EIA_PAGE_SIZE,  # says two more pages are due
                "data": [
                    {"period": f"2024-01-01T{i:02d}", "value": i, "type": "D"}
                    for i in range(EIA_PAGE_SIZE)
                ],
            }
        }
        empty_page = {"response": {"total": 3 * EIA_PAGE_SIZE, "data": []}}
        mock_req.side_effect = [page1, empty_page]

        # Must NOT raise, and returns only the rows actually delivered.
        records = _paginated_fetch("electricity/rto/region-data", {"api_key": "test"})

        assert len(records) == EIA_PAGE_SIZE
        assert mock_req.call_count == 2


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

    @patch("data.gcs_store.write_parquet")
    @patch("data.gcs_store.read_parquet")
    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_truncated_fetch_falls_back_and_does_not_cache(
        self, mock_pf, mock_cache_fn, mock_rp, mock_wp
    ):
        """#269 / P2-06: a truncated multi-page fetch (EIAIncompleteFetchError) must
        serve last-known-good and never cache/persist the partial series over it."""
        stale_df = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
                "demand_mw": [35000.0],
                "forecast_mw": [36000.0],
                "region": ["ERCOT"],
            }
        )
        mock_cache = MagicMock()
        # fresh miss, then stale hit on allow_stale=True
        mock_cache.get.side_effect = [None, stale_df]
        mock_cache_fn.return_value = mock_cache
        mock_pf.side_effect = EIAIncompleteFetchError("truncated: 5000/7000 rows")

        df = fetch_demand("ERCOT", start="2024-01-01T00", end="2024-01-02T23")

        pd.testing.assert_frame_equal(df, stale_df)
        mock_cache.set.assert_not_called()  # partial must NOT poison the cache
        mock_wp.assert_not_called()  # ...nor overwrite the GCS last-known-good
        mock_rp.assert_not_called()  # stale cache satisfied the fallback

    @patch("data.gcs_store.write_parquet")
    @patch("data.gcs_store.read_parquet")
    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_df_only_200_parses_empty_falls_back_and_does_not_cache(
        self, mock_pf, mock_cache_fn, mock_rp, mock_wp
    ):
        """#270 / P2-07: a 200 carrying only day-ahead (DF) rows parses to a
        zero-row demand frame — it must fall back to last-known-good (here GCS)
        and never cache the empty frame that would blank the surface for 24h."""
        gcs_df = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
                "demand_mw": [33000.0],
                "forecast_mw": [34000.0],
                "region": ["ERCOT"],
            }
        )
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # no fresh, no stale
        mock_cache_fn.return_value = mock_cache
        mock_rp.return_value = gcs_df
        # A non-empty response with only DF rows: _parse_demand_records filters to
        # type == "D", so the parsed demand frame is empty.
        mock_pf.return_value = [
            {"period": "2024-01-01T00", "value": 41000, "type": "DF"},
            {"period": "2024-01-01T01", "value": 41200, "type": "DF"},
        ]

        df = fetch_demand("ERCOT", start="2024-01-01T00", end="2024-01-02T23")

        pd.testing.assert_frame_equal(df, gcs_df)
        mock_cache.set.assert_not_called()  # empty frame must NOT be cached
        mock_wp.assert_not_called()  # ...nor written to GCS
        mock_rp.assert_called_once_with("demand", "ERCOT")

    @patch("data.gcs_store.write_parquet")
    @patch("data.gcs_store.read_parquet")
    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_all_nan_demand_window_falls_back_and_does_not_cache(
        self, mock_pf, mock_cache_fn, mock_rp, mock_wp
    ):
        """#270 sibling: a 200 whose D observations are all null/zero parses to
        rows-present-but-all-NaN demand — real rows, no usable signal. That frame
        must NOT be cached or written over the GCS last-known-good even though
        ``df.empty`` is False (it has rows)."""
        gcs_df = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
                "demand_mw": [33000.0],
                "forecast_mw": [34000.0],
                "region": ["ERCOT"],
            }
        )
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # no fresh, no stale
        mock_cache_fn.return_value = mock_cache
        mock_rp.return_value = gcs_df
        # All D rows null/zero -> _parse_demand_records coerces every value to NaN,
        # yielding a non-empty frame whose demand_mw is entirely NaN.
        mock_pf.return_value = [
            {"period": "2024-01-01T00", "value": None, "type": "D"},
            {"period": "2024-01-01T01", "value": 0, "type": "D"},
        ]

        df = fetch_demand("ERCOT", start="2024-01-01T00", end="2024-01-02T23")

        pd.testing.assert_frame_equal(df, gcs_df)
        mock_cache.set.assert_not_called()  # all-NaN frame must NOT poison the cache
        mock_wp.assert_not_called()  # ...nor overwrite the GCS last-known-good
        mock_rp.assert_called_once_with("demand", "ERCOT")

    @patch("data.gcs_store.write_parquet")
    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_partially_nan_demand_window_is_still_cached(self, mock_pf, mock_cache_fn, mock_wp):
        """Guard against over-reach: a window with *some* real demand and some
        NaN is legitimate (short gaps interpolate downstream) — it is NOT all-NaN,
        so it must still be cached/persisted as usable data."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache_fn.return_value = mock_cache
        mock_pf.return_value = [
            {"period": "2024-01-01T00", "value": None, "type": "D"},  # gap
            {"period": "2024-01-01T01", "value": 41000, "type": "D"},  # real
        ]

        df = fetch_demand("ERCOT", start="2024-01-01T00", end="2024-01-02T23")

        assert not df.empty
        assert df["demand_mw"].notna().any()  # at least one real observation
        mock_cache.set.assert_called_once()  # usable -> cached
        mock_wp.assert_called_once()  # ...and persisted


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

    @patch("data.gcs_store.write_parquet")
    @patch("data.gcs_store.read_parquet")
    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_all_null_generation_window_falls_back_and_does_not_cache(
        self, mock_pf, mock_cache_fn, mock_rp, mock_wp
    ):
        """P2-08 (#273) verification HIGH: with nulls preserved as NaN, an
        all-null window must route to last-known-good like demand does —
        never cache a rows-present all-NaN frame for 24h or write it over
        the GCS last-known-good."""
        gcs_df = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
                "fuel_type": ["NG"],
                "generation_mw": [15000.0],
                "region": ["ERCOT"],
            }
        )
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # no fresh, no stale
        mock_cache_fn.return_value = mock_cache
        mock_rp.return_value = gcs_df
        mock_pf.return_value = [
            {"period": "2024-01-01T00", "fueltype": "NG", "value": None},
            {"period": "2024-01-01T01", "fueltype": "NG"},
        ]

        df = fetch_generation_by_fuel("ERCOT", start="2024-01-01T00", end="2024-01-01T23")

        pd.testing.assert_frame_equal(df, gcs_df)
        mock_cache.set.assert_not_called()
        mock_wp.assert_not_called()
        mock_rp.assert_called_once_with("generation", "ERCOT")

    @patch("data.gcs_store.write_parquet")
    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_all_zero_generation_window_is_still_usable(self, mock_pf, mock_cache_fn, mock_wp):
        """The all-NaN gate must never misfire on legitimate zeros: unlike
        demand there is no 0→NaN coercion, so a true all-zero window (e.g.
        solar overnight) parses to 0.0 and is cached/persisted normally."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache_fn.return_value = mock_cache
        mock_pf.return_value = [
            {"period": "2024-01-01T00", "fueltype": "SUN", "value": 0},
            {"period": "2024-01-01T01", "fueltype": "SUN", "value": 0},
        ]

        df = fetch_generation_by_fuel("ERCOT", start="2024-01-01T00", end="2024-01-01T23")

        assert (df["generation_mw"] == 0.0).all()
        mock_cache.set.assert_called_once()

    @patch("data.gcs_store.write_parquet")
    @patch("data.gcs_store.read_parquet")
    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_truncated_fetch_falls_back_and_does_not_cache(
        self, mock_pf, mock_cache_fn, mock_rp, mock_wp
    ):
        """#269 uniformity (#174 invariant): generation must share the demand
        no-poison guarantee — a truncated fetch serves last-known-good and never
        caches/persists the partial series."""
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
        mock_pf.side_effect = EIAIncompleteFetchError("truncated: 5000/12000 rows")

        df = fetch_generation_by_fuel("CAISO", start="2024-01-01T00", end="2024-01-01T23")

        pd.testing.assert_frame_equal(df, stale_df)
        mock_cache.set.assert_not_called()
        mock_wp.assert_not_called()
        mock_rp.assert_not_called()

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

    @patch("data.gcs_store.write_parquet")
    @patch("data.gcs_store.read_parquet")
    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_all_null_interchange_window_falls_back_and_does_not_cache(
        self, mock_pf, mock_cache_fn, mock_rp, mock_wp
    ):
        """P2-08 (#273) verification HIGH sibling: an all-null interchange
        window routes to last-known-good — it must not poison the cache,
        overwrite GCS, or blank the US Grid chip past a transient artifact."""
        gcs_df = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
                "from_ba": ["ERCO"],
                "to_ba": ["SWPP"],
                "interchange_mw": [500.0],
            }
        )
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache_fn.return_value = mock_cache
        mock_rp.return_value = gcs_df
        mock_pf.return_value = [
            {"period": "2024-01-01T00", "fromba": "ERCO", "toba": "SWPP", "value": None},
        ]

        df = fetch_interchange("ERCOT", start="2024-01-01T00", end="2024-01-01T23")

        pd.testing.assert_frame_equal(df, gcs_df)
        mock_cache.set.assert_not_called()
        mock_wp.assert_not_called()
        mock_rp.assert_called_once_with("interchange", "ERCOT")

    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_all_zero_interchange_window_is_still_usable(self, mock_pf, mock_cache_fn):
        """True zero flow is legitimate — the all-NaN gate must not misfire."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache_fn.return_value = mock_cache
        mock_pf.return_value = [
            {"period": "2024-01-01T00", "fromba": "ERCO", "toba": "SWPP", "value": 0},
        ]

        df = fetch_interchange("ERCOT", start="2024-01-01T00", end="2024-01-01T23")

        assert df["interchange_mw"].iloc[0] == 0.0
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

    @patch("data.gcs_store.write_parquet")
    @patch("data.gcs_store.read_parquet")
    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_truncated_fetch_falls_back_and_does_not_cache(
        self, mock_pf, mock_cache_fn, mock_rp, mock_wp
    ):
        """#269 uniformity (#174 invariant): interchange must share the demand
        no-poison guarantee — a truncated fetch serves last-known-good and never
        caches/persists the partial series."""
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
        mock_pf.side_effect = EIAIncompleteFetchError("truncated: 5000/20000 rows")

        df = fetch_interchange("ERCOT", start="2024-01-01T00", end="2024-01-01T23")

        pd.testing.assert_frame_equal(df, stale_df)
        mock_cache.set.assert_not_called()
        mock_wp.assert_not_called()
        mock_rp.assert_not_called()

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


# ---------------------------------------------------------------------------
# Circuit breaker (EIA outage resilience, #174)
# ---------------------------------------------------------------------------


class TestEIACircuitBreaker:
    """Unit tests for the EIA circuit-breaker state machine."""

    def test_starts_closed(self):
        cb = _EIACircuitBreaker(trip_threshold=3, probe_interval=30)
        assert cb.tripped is False
        assert cb.allow_request() is True

    def test_trips_after_threshold_consecutive_failures(self):
        cb = _EIACircuitBreaker(trip_threshold=3, probe_interval=30)
        cb.record_failure()
        cb.record_failure()
        assert cb.tripped is False  # below threshold
        cb.record_failure()
        assert cb.tripped is True  # 3rd consecutive trips it

    def test_success_resets_consecutive_failures(self):
        cb = _EIACircuitBreaker(trip_threshold=3, probe_interval=30)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # resets the counter
        cb.record_failure()
        cb.record_failure()
        assert cb.tripped is False  # only 2 consecutive since the reset

    def test_tripped_breaker_fails_fast(self):
        cb = _EIACircuitBreaker(trip_threshold=2, probe_interval=30)
        cb.record_failure()
        cb.record_failure()
        assert cb.tripped is True
        assert cb.allow_request() is False  # no probe yet -> fail fast

    def test_probe_allowed_every_interval(self):
        cb = _EIACircuitBreaker(trip_threshold=1, probe_interval=3)
        cb.record_failure()  # trips (threshold=1)
        assert cb.allow_request() is False  # 1 suppressed
        assert cb.allow_request() is False  # 2 suppressed
        assert cb.allow_request() is True  # 3rd -> probe permitted
        assert cb.allow_request() is False  # counter reset -> suppress again

    def test_success_after_trip_closes_breaker(self):
        cb = _EIACircuitBreaker(trip_threshold=1, probe_interval=30)
        cb.record_failure()
        assert cb.tripped is True
        cb.record_success()
        assert cb.tripped is False
        assert cb.allow_request() is True

    def test_reset_clears_state(self):
        cb = _EIACircuitBreaker(trip_threshold=1, probe_interval=30)
        cb.record_failure()
        assert cb.tripped is True
        cb.reset()
        assert cb.tripped is False
        assert cb.allow_request() is True


class TestRequestWithBackoffCircuitBreaker:
    """The breaker bounds EIA cost during a sustained outage."""

    @patch("data.eia_client.time.sleep")
    @patch("data.eia_client.requests.get")
    def test_trips_after_consecutive_exhausted_calls(self, mock_get, mock_sleep):
        fail = MagicMock()
        fail.status_code = 503
        mock_get.return_value = fail

        for _ in range(EIA_CIRCUIT_TRIP_THRESHOLD):
            assert _request_with_backoff("https://api.eia.gov/v2/test", {}) is None
        assert _circuit_breaker.tripped is True

    @patch("data.eia_client.time.sleep")
    @patch("data.eia_client.requests.get")
    def test_tripped_breaker_short_circuits_network(self, mock_get, mock_sleep):
        """Once tripped, calls fast-fail without further network attempts."""
        fail = MagicMock()
        fail.status_code = 503
        mock_get.return_value = fail

        for _ in range(EIA_CIRCUIT_TRIP_THRESHOLD):
            _request_with_backoff("https://api.eia.gov/v2/test", {})
        calls_after_trip = mock_get.call_count

        # Several suppressed calls (fewer than the probe interval) add no
        # network calls — proving the retry budget is short-circuited.
        for _ in range(EIA_CIRCUIT_PROBE_INTERVAL - 1):
            assert _request_with_backoff("https://api.eia.gov/v2/test", {}) is None
        assert mock_get.call_count == calls_after_trip

    @patch("data.eia_client.time.sleep")
    @patch("data.eia_client.requests.get")
    def test_probe_success_closes_breaker(self, mock_get, mock_sleep):
        fail = MagicMock()
        fail.status_code = 503
        ok = MagicMock()
        ok.status_code = 200
        ok.json.return_value = {"response": {"data": []}}
        # Enough failures to trip (full retry budget each), then OK for probes.
        mock_get.side_effect = [fail] * (EIA_CIRCUIT_TRIP_THRESHOLD * MAX_RETRIES) + [ok] * 3

        for _ in range(EIA_CIRCUIT_TRIP_THRESHOLD):
            _request_with_backoff("https://api.eia.gov/v2/test", {})
        assert _circuit_breaker.tripped is True

        # Force the next call to be a probe; it succeeds and closes the breaker.
        _circuit_breaker._suppressed_since_probe = EIA_CIRCUIT_PROBE_INTERVAL - 1
        result = _request_with_backoff("https://api.eia.gov/v2/test", {})
        assert result == {"response": {"data": []}}
        assert _circuit_breaker.tripped is False


class TestGenerationGcsFallback:
    """Generation falls back to GCS on an EIA outage and persists on success."""

    @patch("data.gcs_store.write_parquet")
    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_writes_to_gcs_on_success(self, mock_pf, mock_cache_fn, mock_write):
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache_fn.return_value = mock_cache
        mock_pf.return_value = [
            {"period": "2024-01-01T00", "fueltype": "NG", "value": 15000},
        ]

        df = fetch_generation_by_fuel("ERCOT", start="2024-01-01T00", end="2024-01-01T23")

        assert not df.empty
        mock_write.assert_called_once()
        assert mock_write.call_args.args[1] == "generation"

    @patch("data.gcs_store.read_parquet")
    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_falls_back_to_gcs_on_outage(self, mock_pf, mock_cache_fn, mock_read):
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # cold cache, no stale either
        mock_cache_fn.return_value = mock_cache
        mock_pf.return_value = []  # EIA outage -> no records

        gcs_df = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
                "fuel_type": ["NG"],
                "generation_mw": [15000.0],
                "region": ["ERCOT"],
            }
        )
        mock_read.return_value = gcs_df

        df = fetch_generation_by_fuel("ERCOT", start="2024-01-01T00", end="2024-01-01T23")

        pd.testing.assert_frame_equal(df, gcs_df)
        mock_read.assert_called_once_with("generation", "ERCOT")


class TestInterchangeGcsFallback:
    """Interchange falls back to GCS on an EIA outage and persists on success."""

    @patch("data.gcs_store.write_parquet")
    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_writes_to_gcs_on_success(self, mock_pf, mock_cache_fn, mock_write):
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache_fn.return_value = mock_cache
        mock_pf.return_value = [
            {"period": "2024-01-01T00", "fromba": "ERCO", "toba": "SWPP", "value": 500},
        ]

        df = fetch_interchange("ERCOT", start="2024-01-01T00", end="2024-01-01T23")

        assert not df.empty
        mock_write.assert_called_once()
        assert mock_write.call_args.args[1] == "interchange"

    @patch("data.gcs_store.read_parquet")
    @patch("data.eia_client.get_cache")
    @patch("data.eia_client._paginated_fetch")
    def test_falls_back_to_gcs_on_outage(self, mock_pf, mock_cache_fn, mock_read):
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache_fn.return_value = mock_cache
        mock_pf.return_value = []

        gcs_df = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
                "from_ba": ["ERCO"],
                "to_ba": ["SWPP"],
                "interchange_mw": [500.0],
            }
        )
        mock_read.return_value = gcs_df

        df = fetch_interchange("ERCOT", start="2024-01-01T00", end="2024-01-01T23")

        pd.testing.assert_frame_equal(df, gcs_df)
        mock_read.assert_called_once_with("interchange", "ERCOT")


# ---------------------------------------------------------------------------
# Success-latency instrumentation (2026-08-04 incident follow-up)
# ---------------------------------------------------------------------------


class TestLatencyStats:
    """``drain_latency_stats`` — the measurement the 30s read timeout never had.

    On 2026-08-04 a partial EIA degradation turned an ~800s scoring run into
    two SIGKILLs at the 1800s cap, and the cost unit was the hardcoded 30s read
    timeout. Nothing logged EIA latency, so there was no measured p99 to size a
    lower timeout against. These tests pin that the distribution is recorded and
    that draining it hands ownership to the caller exactly once.
    """

    @pytest.fixture(autouse=True)
    def _clear(self):
        from data.eia_client import drain_latency_stats

        drain_latency_stats()
        yield
        drain_latency_stats()

    def test_returns_none_when_nothing_recorded(self):
        from data.eia_client import drain_latency_stats

        # None, not an empty dict — the caller skips the log line entirely
        # rather than publishing percentiles over zero samples.
        assert drain_latency_stats() is None

    def test_percentiles_over_a_known_distribution(self):
        from data.eia_client import _record_latency, drain_latency_stats

        for ms in range(1, 101):  # 1..100 ms
            _record_latency(float(ms))

        stats = drain_latency_stats()

        assert stats["n"] == 100
        # Nearest-rank on n=100 puts p50 at the 50th sample, p99 at the 99th.
        assert stats["p50_ms"] == pytest.approx(50.0)
        assert stats["p95_ms"] == pytest.approx(95.0)
        assert stats["p99_ms"] == pytest.approx(99.0)
        assert stats["max_ms"] == pytest.approx(100.0)

    def test_unsorted_input_still_yields_ordered_percentiles(self):
        """Requests complete out of order across 4 worker threads."""
        from data.eia_client import _record_latency, drain_latency_stats

        for ms in [900.0, 10.0, 50.0, 5000.0, 20.0]:
            _record_latency(ms)

        stats = drain_latency_stats()

        assert stats["p50_ms"] <= stats["p95_ms"] <= stats["p99_ms"] <= stats["max_ms"]
        assert stats["max_ms"] == pytest.approx(5000.0)

    def test_single_sample_does_not_index_out_of_range(self):
        from data.eia_client import _record_latency, drain_latency_stats

        _record_latency(12.5)

        stats = drain_latency_stats()

        assert stats == {"n": 1, "p50_ms": 12.5, "p95_ms": 12.5, "p99_ms": 12.5, "max_ms": 12.5}

    def test_drain_clears_so_runs_do_not_accumulate(self):
        """Process-local and per-run, like the circuit breaker beside it."""
        from data.eia_client import _record_latency, drain_latency_stats

        _record_latency(1.0)
        assert drain_latency_stats()["n"] == 1
        assert drain_latency_stats() is None

    def test_successful_request_records_a_sample(self):
        from data.eia_client import _request_with_backoff, drain_latency_stats

        with patch("requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, content=b"{}")
            mock_get.return_value.json.return_value = {"response": {"data": []}}
            _request_with_backoff("https://api.eia.gov/v2/x", {})

        stats = drain_latency_stats()
        assert stats is not None and stats["n"] == 1

    def test_recording_from_many_threads_loses_nothing(self):
        """Records land from all 4 scoring workers at once.

        Deliberately NOT sold as a test of ``_latency_lock``: ``list.append``
        is atomic under CPython, so this passes with the lock removed. The
        lock's real job is making the drain's read-and-clear a single step,
        and that window is a few bytecodes wide — it cannot be hit reliably
        from a test, so no test here claims to pin it. Asserting something
        that holds with the guard broken is the pattern ``docs/TEST_QUALITY.md``
        calls out; this asserts only what it actually covers.
        """
        import threading

        from data.eia_client import _record_latency, drain_latency_stats

        def writer():
            for i in range(500):
                _record_latency(float(i))

        threads = [threading.Thread(target=writer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert drain_latency_stats()["n"] == 4000


# ---------------------------------------------------------------------------
# Per-call cost control (2026-08-04 EIA partial-degradation incident)
# ---------------------------------------------------------------------------


class _FakeClock:
    """Monotonic clock whose only advance is the sleeps the code asks for.

    Lets the budget arithmetic be exercised exactly, in zero wall time. A read
    timeout is also charged as elapsed time, because that is what a timed-out
    request actually costs and it is the whole subject here.
    """

    def __init__(self):
        self.now = 1000.0
        self.slept: list[float] = []

    def monotonic(self):
        return self.now

    def sleep(self, secs):
        self.slept.append(secs)
        self.now += secs

    def advance(self, secs):
        self.now += secs


class _FlakyTransport:
    """Deterministic partial degradation: every Nth call fails, rest succeed.

    Reproduces 2026-08-04's regime exactly (``fail_every=8`` is 12.5%) without
    randomness. A timeout also advances the fake clock by the read timeout it
    was handed, so a caller can measure what the failures actually cost.
    """

    def __init__(self, clock, fail_every=8, mode="timeout"):
        self.clock, self.fail_every, self.mode = clock, fail_every, mode
        self.calls = 0
        self.timeouts: list[float] = []

    def __call__(self, url, params=None, timeout=None):
        self.calls += 1
        self.timeouts.append(timeout)
        if self.calls % self.fail_every == 0:
            if self.mode == "timeout":
                # A read timeout costs its full read budget in wall time.
                self.clock.advance(timeout[1] if isinstance(timeout, tuple) else timeout)
                raise requests.Timeout("read timed out")
            return MagicMock(status_code=502, content=b"")
        resp = MagicMock(status_code=200, content=b"{}")
        resp.json.return_value = {"response": {"data": [], "total": 0}}
        return resp


@pytest.fixture
def clock(monkeypatch):
    c = _FakeClock()
    monkeypatch.setattr("data.eia_client.time", c)
    return c


class TestCallBudget:
    """One EIA call must cost a bounded amount of wall time.

    The breaker bounds a TOTAL outage. Nothing bounded a partial one: on
    2026-08-04 every call eventually succeeded, so the breaker stayed closed
    by design and each failing attempt still cost the hardcoded 30s read
    timeout. ~366 of them turned an ~800s run into two SIGKILLs at the 1800s
    task timeout.
    """

    def test_timeout_is_a_connect_read_tuple_not_a_scalar(self, clock):
        """A scalar applies the same value to connect AND read.

        Reverting to one number silently restores the expensive dead-connect
        case, and nothing else in the suite would notice.
        """
        from config import EIA_CONNECT_TIMEOUT_S, EIA_READ_TIMEOUT_S
        from data.eia_client import _request_with_backoff

        with patch("requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, content=b"{}")
            mock_get.return_value.json.return_value = {}
            _request_with_backoff("https://api.eia.gov/v2/x", {})

        timeout = mock_get.call_args.kwargs["timeout"]
        assert isinstance(timeout, tuple)
        assert timeout == (EIA_CONNECT_TIMEOUT_S, EIA_READ_TIMEOUT_S)

    def test_total_wall_time_stays_inside_the_budget(self, clock, monkeypatch):
        from config import EIA_CALL_BUDGET_S
        from data.eia_client import _request_with_backoff

        transport = _FlakyTransport(clock, fail_every=1)  # everything times out
        monkeypatch.setattr("requests.get", transport)
        start = clock.now

        assert _request_with_backoff("https://api.eia.gov/v2/x", {}) is None

        assert clock.now - start <= EIA_CALL_BUDGET_S

    def test_read_timeout_is_clamped_to_the_remaining_budget(self, clock, monkeypatch):
        """A later attempt must not be handed more read budget than is left.

        Backoff is jittered, so with the production budget whether the clamp
        bites is random — pinning it needs a case where it MUST. Zero jitter
        and a 20s budget give exact arithmetic: attempt 1 spends the full 12s
        read window, leaving 8s, so attempt 2 is handed 8s and not 12s.
        """
        import data.eia_client as ec
        from config import EIA_READ_TIMEOUT_S

        monkeypatch.setattr(ec, "EIA_CALL_BUDGET_S", 20.0)
        monkeypatch.setattr(ec.random, "uniform", lambda a, b: 0.0)
        transport = _FlakyTransport(clock, fail_every=1)
        monkeypatch.setattr("requests.get", transport)
        start = clock.now

        ec._request_with_backoff("https://api.eia.gov/v2/x", {})

        reads = [t[1] for t in transport.timeouts]
        assert reads[0] == pytest.approx(EIA_READ_TIMEOUT_S)
        assert reads[1] == pytest.approx(20.0 - EIA_READ_TIMEOUT_S)
        assert clock.now - start <= 20.0

    def test_backoff_is_jittered_and_capped(self, clock, monkeypatch):
        from config import EIA_MAX_BACKOFF_S
        from data.eia_client import _request_with_backoff

        monkeypatch.setattr("requests.get", _FlakyTransport(clock, fail_every=1, mode="502"))

        _request_with_backoff("https://api.eia.gov/v2/x", {})

        assert clock.slept, "expected at least one backoff sleep"
        for slept in clock.slept:
            assert 0 <= slept <= EIA_MAX_BACKOFF_S

    def test_budget_exhaustion_is_logged_not_silent(self, clock, monkeypatch):
        """The budget pins its own value rather than reading the production one.

        At the shipped 120s the retry ladder finishes first, so the budget
        never binds — which is the point of that value. This test is about the
        mechanism firing *when* the budget is the binding constraint, so it
        sets one that is. Reading the production default here is what made an
        earlier version of this test flip red purely because the default moved.
        """
        import data.eia_client as ec

        monkeypatch.setattr(ec, "EIA_CALL_BUDGET_S", 20.0)
        monkeypatch.setattr("requests.get", _FlakyTransport(clock, fail_every=1))
        fake_log = MagicMock()
        monkeypatch.setattr(ec, "log", fake_log)

        ec._request_with_backoff("https://api.eia.gov/v2/x", {})

        events = [c.args[0] for c in fake_log.warning.call_args_list]
        assert "eia_call_budget_exhausted" in events

    def test_a_4xx_still_returns_immediately_without_recording_failure(self, clock, monkeypatch):
        """Budget work must not disturb the #174 invariant: 4xx is our bug."""
        from data.eia_client import _circuit_breaker, _request_with_backoff

        resp = MagicMock(status_code=404, content=b"", text="nope")
        monkeypatch.setattr("requests.get", lambda *a, **k: resp)

        assert _request_with_backoff("https://api.eia.gov/v2/x", {}) is None
        assert _circuit_breaker._consecutive_failures == 0
        assert clock.slept == []


class TestPartialDegradationIsNotABreakerProblem:
    """Characterization tests — these encode a DECISION, not just behaviour.

    The instinct after 2026-08-04 is "make the breaker trip on a failure
    rate". That is the wrong instrument and these pin why. Zero data was lost
    that day: every call succeeded on retry, and a breaker tripping at 8-15%
    would have fail-fasted the remaining BAs onto stale-cache/GCS last-known
    data — trading fresh data we could actually get for runtime that the call
    budget recovers more cheaply. Do not "fix" these tests into tripping.
    """

    def test_interleaved_failures_do_not_trip_the_consecutive_breaker(self, clock, monkeypatch):
        from data.eia_client import _circuit_breaker, _request_with_backoff

        # 12.5% of attempts fail — 2026-08-04's regime. Each call retries past
        # its one bad attempt and succeeds, so no call exhausts its budget.
        monkeypatch.setattr("requests.get", _FlakyTransport(clock, fail_every=8))

        for _ in range(40):
            assert _request_with_backoff("https://api.eia.gov/v2/x", {}) is not None

        assert _circuit_breaker.tripped is False

    def test_partial_degradation_never_reaches_the_fallback_chain(self, clock, monkeypatch):
        """The observable signature of 2026-08-04, asserted.

        `eia_max_retries_exceeded` at ZERO while exceptions are high is what
        distinguishes partial degradation from an outage — and it is the
        discriminator the runbook now branches on.
        """
        import data.eia_client as ec

        monkeypatch.setattr("requests.get", _FlakyTransport(clock, fail_every=8))
        fake_log = MagicMock()
        monkeypatch.setattr(ec, "log", fake_log)

        for _ in range(40):
            ec._request_with_backoff("https://api.eia.gov/v2/x", {})

        errors = [c.args[0] for c in fake_log.error.call_args_list]
        assert "eia_request_exception" in errors  # degradation was real
        assert "eia_max_retries_exceeded" not in errors  # yet nothing was lost

    def test_a_true_outage_still_trips_the_breaker(self, clock, monkeypatch):
        """The #174 behaviour the budget work must not weaken."""
        from data.eia_client import _circuit_breaker, _request_with_backoff

        monkeypatch.setattr("requests.get", _FlakyTransport(clock, fail_every=1))

        for _ in range(3):
            _request_with_backoff("https://api.eia.gov/v2/x", {})

        assert _circuit_breaker.tripped is True
