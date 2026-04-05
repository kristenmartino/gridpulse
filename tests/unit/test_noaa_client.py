"""
Unit tests for data/noaa_client.py.

Covers fetch_alerts_for_region(), fetch_all_alerts(), _fetch_state_alerts(),
_parse_datetime(), _parse_areas(), and _alert_to_dict(). All HTTP calls and
cache access are mocked so tests are fully isolated with no external dependencies.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import requests

from data.noaa_client import (
    SEVERITY_MAP,
    WeatherAlert,
    _alert_to_dict,
    _fetch_state_alerts,
    _parse_areas,
    _parse_datetime,
    fetch_alerts_for_region,
    fetch_all_alerts,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_alert(**overrides) -> WeatherAlert:
    """Create a WeatherAlert with sensible defaults, allowing overrides."""
    defaults = {
        "id": "urn:oid:2.49.0.1.840.0.test-1",
        "event": "Heat Advisory",
        "headline": "Heat Advisory until 8PM CDT",
        "description": "Dangerously hot conditions.",
        "severity": "warning",
        "noaa_severity": "Moderate",
        "urgency": "Expected",
        "certainty": "Likely",
        "onset": datetime.fromisoformat("2024-07-15T12:00:00-05:00"),
        "expires": datetime.fromisoformat("2024-07-15T20:00:00-05:00"),
        "areas": ["Travis County", "Williamson County"],
        "states": ["TX"],
        "balancing_authorities": [],
    }
    defaults.update(overrides)
    return WeatherAlert(**defaults)


def _make_noaa_feature(**prop_overrides) -> dict:
    """Create a single NOAA GeoJSON feature with default properties."""
    props = {
        "id": "urn:oid:2.49.0.1.840.0.test-1",
        "event": "Heat Advisory",
        "headline": "Heat Advisory until 8PM CDT",
        "description": "Dangerously hot conditions.",
        "severity": "Moderate",
        "urgency": "Expected",
        "certainty": "Likely",
        "onset": "2024-07-15T12:00:00-05:00",
        "expires": "2024-07-15T20:00:00-05:00",
        "areaDesc": "Travis County; Williamson County",
    }
    props.update(prop_overrides)
    return {"properties": props}


# ---------------------------------------------------------------------------
# _parse_datetime tests
# ---------------------------------------------------------------------------


class TestParseDatetime:
    """Tests for _parse_datetime -- ISO string parsing."""

    def test_valid_iso_string_returns_datetime(self):
        """Standard ISO string parses to a datetime object."""
        result = _parse_datetime("2024-07-15T12:00:00-05:00")
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 7
        assert result.day == 15

    def test_none_returns_none(self):
        """None input returns None (covers line 170)."""
        assert _parse_datetime(None) is None

    def test_empty_string_returns_none(self):
        """Empty string returns None."""
        assert _parse_datetime("") is None

    def test_invalid_format_returns_none(self):
        """Unparseable string returns None (covers line 174)."""
        assert _parse_datetime("not-a-date") is None

    def test_garbage_string_returns_none(self):
        """Completely garbled input returns None."""
        assert _parse_datetime("abc123xyz") is None


# ---------------------------------------------------------------------------
# _parse_areas tests
# ---------------------------------------------------------------------------


class TestParseAreas:
    """Tests for _parse_areas -- semicolon-separated area parsing."""

    def test_standard_semicolon_separated(self):
        """Parses semicolon-delimited areas correctly."""
        result = _parse_areas("Travis County; Williamson County")
        assert result == ["Travis County", "Williamson County"]

    def test_empty_string_returns_empty_list(self):
        """Empty string returns empty list (covers line 180)."""
        assert _parse_areas("") == []

    def test_single_area(self):
        """Single area with no semicolons."""
        assert _parse_areas("Dallas County") == ["Dallas County"]

    def test_strips_whitespace(self):
        """Leading/trailing whitespace around areas is stripped."""
        result = _parse_areas("  Travis County ;  Williamson County  ")
        assert result == ["Travis County", "Williamson County"]

    def test_ignores_empty_segments(self):
        """Empty segments from extra semicolons are filtered out."""
        result = _parse_areas("Travis County;;; Williamson County;")
        assert result == ["Travis County", "Williamson County"]


# ---------------------------------------------------------------------------
# _alert_to_dict tests
# ---------------------------------------------------------------------------


class TestAlertToDict:
    """Tests for _alert_to_dict -- serialization (covers line 186+)."""

    def test_basic_serialization(self):
        """Alert with onset/expires serializes all fields correctly."""
        alert = _make_alert()
        d = _alert_to_dict(alert)

        assert d["id"] == alert.id
        assert d["event"] == "Heat Advisory"
        assert d["headline"] == "Heat Advisory until 8PM CDT"
        assert d["description"] == "Dangerously hot conditions."
        assert d["severity"] == "warning"
        assert d["noaa_severity"] == "Moderate"
        assert d["urgency"] == "Expected"
        assert d["certainty"] == "Likely"
        assert d["onset"] == alert.onset.isoformat()
        assert d["expires"] == alert.expires.isoformat()
        assert d["areas"] == ["Travis County", "Williamson County"]
        assert d["states"] == ["TX"]
        assert d["balancing_authorities"] == []

    def test_none_onset_expires(self):
        """Alert with None onset/expires serializes them as None."""
        alert = _make_alert(onset=None, expires=None)
        d = _alert_to_dict(alert)

        assert d["onset"] is None
        assert d["expires"] is None

    def test_roundtrip_reconstruction(self):
        """Dict from _alert_to_dict can reconstruct a WeatherAlert via **kwargs."""
        alert = _make_alert(onset=None, expires=None)
        d = _alert_to_dict(alert)
        reconstructed = WeatherAlert(**d)

        assert reconstructed.id == alert.id
        assert reconstructed.event == alert.event
        assert reconstructed.severity == alert.severity


# ---------------------------------------------------------------------------
# _fetch_state_alerts tests
# ---------------------------------------------------------------------------


class TestFetchStateAlerts:
    """Tests for _fetch_state_alerts -- HTTP fetch for a single state."""

    @patch("data.noaa_client.requests.get")
    def test_success_returns_parsed_alerts(self, mock_get, mock_noaa_alerts_response):
        """Successful response returns list of WeatherAlert objects."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_noaa_alerts_response
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        alerts = _fetch_state_alerts("TX")

        assert len(alerts) == 2
        assert all(isinstance(a, WeatherAlert) for a in alerts)
        # First alert from fixture is Moderate -> "warning"
        assert alerts[0].severity == "warning"
        assert alerts[0].event == "Heat Advisory"
        # Second alert is Extreme -> "critical"
        assert alerts[1].severity == "critical"
        assert alerts[1].event == "Excessive Heat Warning"

    @patch("data.noaa_client.requests.get")
    def test_request_exception_returns_empty(self, mock_get):
        """Network error returns empty list (covers lines 137-139)."""
        mock_get.side_effect = requests.ConnectionError("DNS failure")

        alerts = _fetch_state_alerts("TX")

        assert alerts == []

    @patch("data.noaa_client.requests.get")
    def test_timeout_returns_empty(self, mock_get):
        """Timeout returns empty list."""
        mock_get.side_effect = requests.Timeout("Connection timed out")

        alerts = _fetch_state_alerts("TX")

        assert alerts == []

    @patch("data.noaa_client.requests.get")
    def test_http_error_returns_empty(self, mock_get):
        """HTTP 500 returns empty list."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        mock_get.return_value = mock_resp

        alerts = _fetch_state_alerts("TX")

        assert alerts == []

    @patch("data.noaa_client.requests.get")
    def test_empty_features_returns_empty(self, mock_get):
        """Response with no features returns empty list."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"features": []}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        alerts = _fetch_state_alerts("TX")

        assert alerts == []

    @patch("data.noaa_client.requests.get")
    def test_missing_features_key_returns_empty(self, mock_get):
        """Response without 'features' key returns empty list."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        alerts = _fetch_state_alerts("TX")

        assert alerts == []

    @patch("data.noaa_client.requests.get")
    def test_description_truncated_to_500(self, mock_get):
        """Long descriptions are truncated to 500 characters."""
        long_desc = "A" * 1000
        feature = _make_noaa_feature(description=long_desc)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"features": [feature]}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        alerts = _fetch_state_alerts("TX")

        assert len(alerts[0].description) == 500

    @patch("data.noaa_client.requests.get")
    def test_missing_properties_uses_defaults(self, mock_get):
        """Feature with empty properties uses default values."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"features": [{"properties": {}}]}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        alerts = _fetch_state_alerts("CA")

        assert len(alerts) == 1
        alert = alerts[0]
        assert alert.id == ""
        assert alert.event == "Unknown"
        assert alert.headline == ""
        assert alert.noaa_severity == "Unknown"
        assert alert.severity == "info"  # Unknown maps to info
        assert alert.onset is None
        assert alert.expires is None
        assert alert.areas == []
        assert alert.states == ["CA"]

    @patch("data.noaa_client.requests.get")
    def test_all_severity_mappings(self, mock_get):
        """Each NOAA severity level maps to the correct dashboard severity."""
        for noaa_severity, expected_severity in SEVERITY_MAP.items():
            feature = _make_noaa_feature(severity=noaa_severity)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"features": [feature]}
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp

            alerts = _fetch_state_alerts("TX")

            assert alerts[0].severity == expected_severity
            assert alerts[0].noaa_severity == noaa_severity


# ---------------------------------------------------------------------------
# fetch_alerts_for_region tests
# ---------------------------------------------------------------------------


class TestFetchAlertsForRegion:
    """Tests for fetch_alerts_for_region -- full fetch + cache flow."""

    def test_unknown_region_raises_value_error(self):
        """Unknown region raises ValueError (covers line 77-78)."""
        with pytest.raises(ValueError, match="Unknown region"):
            fetch_alerts_for_region("BOGUS_REGION")

    @patch("data.noaa_client.get_cache")
    def test_returns_cached_alerts(self, mock_get_cache):
        """Cache hit returns deserialized WeatherAlert objects (covers line 86)."""
        cached_data = [
            {
                "id": "test-cached-1",
                "event": "Frost Advisory",
                "headline": "Frost Advisory",
                "description": "Frost expected.",
                "severity": "info",
                "noaa_severity": "Minor",
                "urgency": "Expected",
                "certainty": "Likely",
                "onset": None,
                "expires": None,
                "areas": ["County A"],
                "states": ["TX"],
                "balancing_authorities": ["ERCOT"],
            }
        ]
        mock_cache = MagicMock()
        mock_cache.get.return_value = cached_data
        mock_get_cache.return_value = mock_cache

        result = fetch_alerts_for_region("ERCOT", use_cache=True)

        assert len(result) == 1
        assert isinstance(result[0], WeatherAlert)
        assert result[0].id == "test-cached-1"
        assert result[0].event == "Frost Advisory"
        mock_cache.get.assert_called_once_with("noaa_alerts_ERCOT")

    @patch("data.noaa_client._fetch_state_alerts")
    @patch("data.noaa_client.get_cache")
    def test_cache_miss_fetches_from_api(self, mock_get_cache, mock_fetch):
        """Cache miss triggers API call and caches result (covers lines 80-112)."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        alert = _make_alert(id="api-alert-1", severity="critical")
        mock_fetch.return_value = [alert]

        result = fetch_alerts_for_region("ERCOT", use_cache=True)

        assert len(result) == 1
        assert result[0].id == "api-alert-1"
        assert result[0].balancing_authorities == ["ERCOT"]
        mock_cache.set.assert_called_once()
        # Verify cache key
        cache_key = mock_cache.set.call_args[0][0]
        assert cache_key == "noaa_alerts_ERCOT"

    @patch("data.noaa_client._fetch_state_alerts")
    @patch("data.noaa_client.get_cache")
    def test_skips_cache_when_disabled(self, mock_get_cache, mock_fetch):
        """use_cache=False skips cache lookup."""
        mock_cache = MagicMock()
        mock_get_cache.return_value = mock_cache

        mock_fetch.return_value = []

        fetch_alerts_for_region("ERCOT", use_cache=False)

        mock_cache.get.assert_not_called()

    @patch("data.noaa_client._fetch_state_alerts")
    @patch("data.noaa_client.get_cache")
    def test_results_sorted_by_severity(self, mock_get_cache, mock_fetch):
        """Alerts are sorted: critical first, then warning, then info."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        alerts = [
            _make_alert(id="info-1", severity="info"),
            _make_alert(id="critical-1", severity="critical"),
            _make_alert(id="warning-1", severity="warning"),
        ]
        mock_fetch.return_value = alerts

        result = fetch_alerts_for_region("ERCOT")

        assert result[0].severity == "critical"
        assert result[1].severity == "warning"
        assert result[2].severity == "info"

    @patch("data.noaa_client._fetch_state_alerts")
    @patch("data.noaa_client.get_cache")
    def test_deduplicates_alerts_across_states(self, mock_get_cache, mock_fetch):
        """Same alert ID from multiple states is only included once."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        # PJM covers many states; same alert could appear from multiple states
        duplicate_alert = _make_alert(id="dup-1", severity="warning")
        mock_fetch.return_value = [duplicate_alert]

        result = fetch_alerts_for_region("PJM")

        # PJM has 14 states, each returning the same alert -- should deduplicate to 1
        assert len(result) == 1
        assert result[0].id == "dup-1"

    @patch("data.noaa_client._fetch_state_alerts")
    @patch("data.noaa_client.get_cache")
    def test_empty_result_when_no_alerts(self, mock_get_cache, mock_fetch):
        """No alerts from any state returns empty list, still caches."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        mock_fetch.return_value = []

        result = fetch_alerts_for_region("FPL")

        assert result == []
        # Should still cache the empty result
        mock_cache.set.assert_called_once()

    @patch("data.noaa_client._fetch_state_alerts")
    @patch("data.noaa_client.get_cache")
    def test_cache_ttl_capped_at_1800(self, mock_get_cache, mock_fetch):
        """Cache TTL is capped at 1800 seconds (30 min)."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        mock_fetch.return_value = []

        fetch_alerts_for_region("ERCOT")

        _, kwargs = mock_cache.set.call_args
        assert kwargs["ttl"] <= 1800


# ---------------------------------------------------------------------------
# fetch_all_alerts tests
# ---------------------------------------------------------------------------


class TestFetchAllAlerts:
    """Tests for fetch_all_alerts -- iterates all regions (covers lines 122-125)."""

    @patch("data.noaa_client.fetch_alerts_for_region")
    def test_returns_dict_for_all_regions(self, mock_fetch_region):
        """Returns a dict with an entry for every region in STATE_TO_BA."""
        from config import STATE_TO_BA

        mock_fetch_region.return_value = []

        result = fetch_all_alerts(use_cache=True)

        assert isinstance(result, dict)
        for region in STATE_TO_BA:
            assert region in result
            assert result[region] == []
        assert mock_fetch_region.call_count == len(STATE_TO_BA)

    @patch("data.noaa_client.fetch_alerts_for_region")
    def test_passes_use_cache_flag(self, mock_fetch_region):
        """use_cache parameter is forwarded to each region fetch."""
        mock_fetch_region.return_value = []

        fetch_all_alerts(use_cache=False)

        for call in mock_fetch_region.call_args_list:
            assert call[1]["use_cache"] is False

    @patch("data.noaa_client.fetch_alerts_for_region")
    def test_returns_alerts_per_region(self, mock_fetch_region):
        """Each region gets its own list of alerts."""
        alert_ercot = _make_alert(id="ercot-1")

        def side_effect(region, use_cache=True):
            if region == "ERCOT":
                return [alert_ercot]
            return []

        mock_fetch_region.side_effect = side_effect

        result = fetch_all_alerts()

        assert len(result["ERCOT"]) == 1
        assert result["ERCOT"][0].id == "ercot-1"
        assert result["FPL"] == []
