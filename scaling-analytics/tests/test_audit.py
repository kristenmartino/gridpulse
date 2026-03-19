"""Tests for the audit module."""
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

import pytest


def _make_mock_conn(fetchone_return=None, description=None, side_effect=None):
    """Create a mock psycopg2 connection with proper cursor context manager."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    if side_effect:
        mock_cursor.execute.side_effect = side_effect

    mock_cursor.fetchone.return_value = fetchone_return
    mock_cursor.description = description

    # psycopg2 cursors use context managers: with conn.cursor() as cur:
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    return mock_conn


class TestReadLatestAudit:

    def test_returns_none_when_no_records(self):
        """read_latest_audit returns None when no audit records exist."""
        from src.processing.audit import read_latest_audit

        mock_conn = _make_mock_conn(fetchone_return=None)
        result = read_latest_audit(mock_conn, "ERCOT")
        assert result is None

    def test_returns_dict_when_record_exists(self):
        """read_latest_audit returns dict with audit fields."""
        from src.processing.audit import read_latest_audit

        now = datetime.now(timezone.utc)
        mock_conn = _make_mock_conn(
            fetchone_return=(
                "ERCOT", "2025-01-15T12:00:00+00:00", "postgres", "postgres",
                500, 500, '["xgboost"]', '{"xgboost": 1.0}',
                43, "abc123", '{"xgboost": {"mape": 3.13}}', 48000.0, "full", now,
            ),
            description=[
                ("region",), ("scored_at",), ("demand_source",), ("weather_source",),
                ("demand_rows",), ("weather_rows",), ("model_versions",), ("ensemble_weights",),
                ("feature_count",), ("feature_hash",), ("mape",), ("peak_forecast_mw",),
                ("scoring_mode",), ("created_at",),
            ],
        )

        result = read_latest_audit(mock_conn, "ERCOT")
        assert result is not None
        assert result["region"] == "ERCOT"
        assert result["demand_rows"] == 500
        assert result["feature_count"] == 43

    def test_handles_db_error_gracefully(self):
        """read_latest_audit returns None on DB error."""
        from src.processing.audit import read_latest_audit

        mock_conn = MagicMock()
        mock_conn.cursor.side_effect = Exception("DB connection lost")

        result = read_latest_audit(mock_conn, "ERCOT")
        assert result is None


class TestGetDataFreshness:

    def test_returns_list_of_sources(self):
        """get_data_freshness returns status for all monitored tables."""
        from src.processing.audit import get_data_freshness

        mock_conn = _make_mock_conn(fetchone_return=(None,))
        result = get_data_freshness(mock_conn)

        assert isinstance(result, list)
        assert len(result) == 4
        source_names = [s["source"] for s in result]
        assert "EIA Demand" in source_names
        assert "Open-Meteo Weather" in source_names
        assert "Forecast Pipeline" in source_names

    def test_handles_db_errors(self):
        """get_data_freshness marks sources as error on failure."""
        from src.processing.audit import get_data_freshness

        mock_conn = MagicMock()
        mock_conn.cursor.side_effect = Exception("DB gone")

        result = get_data_freshness(mock_conn)
        assert isinstance(result, list)
        for source in result:
            assert source["status"] == "error"
