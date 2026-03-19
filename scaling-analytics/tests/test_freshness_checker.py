"""Tests for the FreshnessChecker."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from src.processing.freshness_checker import FreshnessChecker


def _make_mock_conn(
    max_ts=None,
    freshness_ts=None,
    side_effect=None,
    row_count=100,
):
    """Create a mock psycopg2 connection for freshness checks."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    if side_effect:
        mock_cursor.execute.side_effect = side_effect
    else:
        call_count = [0]

        def execute_side_effect(sql, params=None):
            call_count[0] += 1
            # First call: SELECT MAX(timestamp) FROM raw_*
            # Second call: SELECT last_timestamp FROM data_freshness
            pass

        def fetchone_side_effect():
            c = call_count[0]
            if c == 1:
                return (max_ts,)
            elif c == 2:
                return (freshness_ts,) if freshness_ts is not None else None
            return (max_ts, row_count)

        mock_cursor.execute.side_effect = execute_side_effect
        mock_cursor.fetchone.side_effect = fetchone_side_effect

    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.commit = MagicMock()

    return mock_conn


class TestHasNewData:
    def test_returns_true_when_never_checked(self):
        """has_new_data returns True when data_freshness has no record."""
        now = datetime.now(UTC)
        conn = _make_mock_conn(max_ts=now, freshness_ts=None)
        checker = FreshnessChecker(conn)
        assert checker.has_new_data("demand") is True

    def test_returns_true_when_new_data(self):
        """has_new_data returns True when source has newer data."""
        now = datetime.now(UTC)
        old = now - timedelta(hours=2)
        conn = _make_mock_conn(max_ts=now, freshness_ts=old)
        checker = FreshnessChecker(conn)
        assert checker.has_new_data("demand") is True

    def test_returns_false_when_no_new_data(self):
        """has_new_data returns False when last_timestamp matches current max."""
        now = datetime.now(UTC)
        conn = _make_mock_conn(max_ts=now, freshness_ts=now)
        checker = FreshnessChecker(conn)
        assert checker.has_new_data("demand") is False

    def test_returns_false_when_no_data_at_all(self):
        """has_new_data returns False when source table is empty."""
        conn = _make_mock_conn(max_ts=None)
        checker = FreshnessChecker(conn)
        assert checker.has_new_data("demand") is False

    def test_returns_true_on_error(self):
        """has_new_data returns True on error (conservative — don't skip)."""
        conn = _make_mock_conn(side_effect=Exception("DB error"))
        checker = FreshnessChecker(conn)
        assert checker.has_new_data("demand") is True

    def test_returns_true_for_unknown_source(self):
        """has_new_data returns True for unknown source names."""
        conn = _make_mock_conn()
        checker = FreshnessChecker(conn)
        assert checker.has_new_data("unknown_source") is True


class TestShouldScore:
    def test_returns_true_when_demand_has_new_data(self):
        """should_score returns True when demand source has new data."""
        now = datetime.now(UTC)
        conn = _make_mock_conn(max_ts=now, freshness_ts=None)
        checker = FreshnessChecker(conn)
        assert checker.should_score() is True

    def test_returns_false_when_no_changes(self):
        """should_score returns False when neither source has new data."""
        now = datetime.now(UTC)
        conn = _make_mock_conn(max_ts=now, freshness_ts=now)
        checker = FreshnessChecker(conn)
        assert checker.should_score() is False


class TestRecordCheck:
    def test_records_without_error(self):
        """record_check does not raise on success."""
        now = datetime.now(UTC)
        conn = _make_mock_conn(max_ts=now)
        checker = FreshnessChecker(conn)
        checker.record_check("demand", now)  # Should not raise
        conn.commit.assert_called()

    def test_handles_error_gracefully(self):
        """record_check does not raise on DB error."""
        conn = _make_mock_conn(side_effect=Exception("DB error"))
        checker = FreshnessChecker(conn)
        checker.record_check("demand")  # Should not raise
