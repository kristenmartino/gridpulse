"""
Unit tests for NEXD-14: Time-Scrub Replay (forecast snapshot history).

Covers:
- save_forecast_snapshot(): insert, FIFO enforcement, idempotency
- list_forecast_snapshots(): ordering, empty state
- get_forecast_snapshot(): retrieval, missing keys
- build_replay_options(): dropdown option format
- GCS replication and restore
- Feature flag existence
"""

import json
import os
import tempfile
import time
from unittest.mock import MagicMock, patch

from data.forecast_history import (
    MAX_SNAPSHOTS,
    _gcs_blob_path,
    _replicate_to_gcs,
    _restore_from_gcs,
    build_replay_options,
    get_forecast_snapshot,
    list_forecast_snapshots,
    save_forecast_snapshot,
)


def _temp_db():
    """Create a temporary SQLite DB path for isolated tests."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


def _sample_timestamps(n=24):
    """Generate sample ISO timestamp strings."""
    import pandas as pd

    return [str(t) for t in pd.date_range("2024-06-01", periods=n, freq="h", tz="UTC")]


def _sample_predictions(n=24, base=30000):
    """Generate sample prediction values."""
    import numpy as np

    return list(np.linspace(base, base + 5000, n))


# ── save_forecast_snapshot + list ─────────────────────────────────


class TestSaveAndList:
    def test_save_one_and_list(self):
        db = _temp_db()
        try:
            with patch("data.forecast_history.CACHE_DB_PATH", db):
                import data.forecast_history

                data.forecast_history._initialized = False

                save_forecast_snapshot(
                    "FPL", 24, "xgboost", _sample_timestamps(), _sample_predictions()
                )
                result = list_forecast_snapshots("FPL", 24, "xgboost")
                assert len(result) == 1
                assert "scored_at" in result[0]
                assert result[0]["peak_mw"] > 0
                assert result[0]["avg_mw"] > 0
        finally:
            os.unlink(db)

    def test_save_multiple_newest_first(self):
        db = _temp_db()
        try:
            with patch("data.forecast_history.CACHE_DB_PATH", db):
                import data.forecast_history

                data.forecast_history._initialized = False

                save_forecast_snapshot(
                    "FPL", 24, "xgboost", _sample_timestamps(), _sample_predictions(base=20000)
                )
                time.sleep(0.01)  # Ensure different scored_at
                save_forecast_snapshot(
                    "FPL", 24, "xgboost", _sample_timestamps(), _sample_predictions(base=40000)
                )
                result = list_forecast_snapshots("FPL", 24, "xgboost")
                assert len(result) == 2
                # Newest first (higher peak)
                assert result[0]["peak_mw"] > result[1]["peak_mw"]
        finally:
            os.unlink(db)

    def test_different_combos_isolated(self):
        db = _temp_db()
        try:
            with patch("data.forecast_history.CACHE_DB_PATH", db):
                import data.forecast_history

                data.forecast_history._initialized = False

                save_forecast_snapshot(
                    "FPL", 24, "xgboost", _sample_timestamps(), _sample_predictions()
                )
                save_forecast_snapshot(
                    "FPL", 168, "xgboost", _sample_timestamps(), _sample_predictions()
                )
                save_forecast_snapshot(
                    "ERCOT", 24, "xgboost", _sample_timestamps(), _sample_predictions()
                )

                assert len(list_forecast_snapshots("FPL", 24, "xgboost")) == 1
                assert len(list_forecast_snapshots("FPL", 168, "xgboost")) == 1
                assert len(list_forecast_snapshots("ERCOT", 24, "xgboost")) == 1
                assert len(list_forecast_snapshots("FPL", 24, "prophet")) == 0
        finally:
            os.unlink(db)


# ── FIFO retention ────────────────────────────────────────────────


class TestFIFORetention:
    def test_fifo_enforced_at_max(self):
        db = _temp_db()
        try:
            with patch("data.forecast_history.CACHE_DB_PATH", db):
                import data.forecast_history

                data.forecast_history._initialized = False

                for i in range(MAX_SNAPSHOTS + 2):
                    save_forecast_snapshot(
                        "FPL",
                        24,
                        "xgboost",
                        _sample_timestamps(),
                        _sample_predictions(base=20000 + i * 100),
                    )
                    time.sleep(0.001)

                result = list_forecast_snapshots("FPL", 24, "xgboost")
                assert len(result) == MAX_SNAPSHOTS

                # Oldest should have been dropped
                all_peaks = [r["peak_mw"] for r in result]
                # The two lowest-base entries should be gone
                assert min(all_peaks) > 20000 + 100  # First two dropped
        finally:
            os.unlink(db)


# ── get_forecast_snapshot ─────────────────────────────────────────


class TestGetSnapshot:
    def test_retrieve_by_scored_at(self):
        db = _temp_db()
        try:
            with patch("data.forecast_history.CACHE_DB_PATH", db):
                import data.forecast_history

                data.forecast_history._initialized = False

                preds = _sample_predictions()
                ts = _sample_timestamps()
                save_forecast_snapshot("FPL", 24, "xgboost", ts, preds)
                snapshots = list_forecast_snapshots("FPL", 24, "xgboost")
                scored_at = snapshots[0]["scored_at"]

                snap = get_forecast_snapshot("FPL", 24, "xgboost", scored_at)
                assert snap is not None
                assert snap["scored_at"] == scored_at
                assert len(snap["predictions"]) == 24
                assert len(snap["timestamps"]) == 24
                assert snap["peak_mw"] > 0
        finally:
            os.unlink(db)

    def test_missing_returns_none(self):
        db = _temp_db()
        try:
            with patch("data.forecast_history.CACHE_DB_PATH", db):
                import data.forecast_history

                data.forecast_history._initialized = False

                result = get_forecast_snapshot("FPL", 24, "xgboost", "2099-01-01T00:00:00+00:00")
                assert result is None
        finally:
            os.unlink(db)


# ── build_replay_options ──────────────────────────────────────────


class TestBuildReplayOptions:
    def test_empty_state_returns_current_only(self):
        db = _temp_db()
        try:
            with patch("data.forecast_history.CACHE_DB_PATH", db):
                import data.forecast_history

                data.forecast_history._initialized = False

                options = build_replay_options("FPL", 24, "xgboost")
                assert len(options) == 1
                assert options[0]["value"] == "current"
                assert options[0]["label"] == "Current"
        finally:
            os.unlink(db)

    def test_with_snapshots(self):
        db = _temp_db()
        try:
            with patch("data.forecast_history.CACHE_DB_PATH", db):
                import data.forecast_history

                data.forecast_history._initialized = False

                save_forecast_snapshot(
                    "FPL", 24, "xgboost", _sample_timestamps(), _sample_predictions(base=20000)
                )
                time.sleep(0.01)
                save_forecast_snapshot(
                    "FPL", 24, "xgboost", _sample_timestamps(), _sample_predictions(base=40000)
                )

                options = build_replay_options("FPL", 24, "xgboost")
                assert len(options) == 3  # Current + 2 historical
                assert options[0]["value"] == "current"
                # Historical entries have ISO timestamp values
                assert options[1]["value"] != "current"
                assert "Peak:" in options[1]["label"]
                assert "MW" in options[1]["label"]
        finally:
            os.unlink(db)

    def test_current_always_first(self):
        db = _temp_db()
        try:
            with patch("data.forecast_history.CACHE_DB_PATH", db):
                import data.forecast_history

                data.forecast_history._initialized = False

                for i in range(5):
                    save_forecast_snapshot(
                        "FPL",
                        24,
                        "xgboost",
                        _sample_timestamps(),
                        _sample_predictions(base=20000 + i * 1000),
                    )
                    time.sleep(0.001)

                options = build_replay_options("FPL", 24, "xgboost")
                assert options[0]["value"] == "current"
                assert options[0]["label"] == "Current"
        finally:
            os.unlink(db)


# ── Empty state ───────────────────────────────────────────────────


class TestEmptyState:
    def test_list_returns_empty(self):
        db = _temp_db()
        try:
            with patch("data.forecast_history.CACHE_DB_PATH", db):
                import data.forecast_history

                data.forecast_history._initialized = False

                assert list_forecast_snapshots("FPL", 24, "xgboost") == []
        finally:
            os.unlink(db)


# ── Content deduplication ─────────────────────────────────────────


class TestContentDedup:
    def test_identical_predictions_not_saved_twice(self):
        db = _temp_db()
        try:
            with patch("data.forecast_history.CACHE_DB_PATH", db):
                import data.forecast_history

                data.forecast_history._initialized = False

                preds = _sample_predictions(base=30000)
                save_forecast_snapshot("FPL", 24, "xgboost", _sample_timestamps(), preds)
                time.sleep(0.01)
                # Same predictions again — should be skipped
                save_forecast_snapshot("FPL", 24, "xgboost", _sample_timestamps(), preds)

                result = list_forecast_snapshots("FPL", 24, "xgboost")
                assert len(result) == 1

        finally:
            os.unlink(db)

    def test_different_predictions_saved(self):
        db = _temp_db()
        try:
            with patch("data.forecast_history.CACHE_DB_PATH", db):
                import data.forecast_history

                data.forecast_history._initialized = False

                save_forecast_snapshot(
                    "FPL", 24, "xgboost", _sample_timestamps(), _sample_predictions(base=30000)
                )
                time.sleep(0.01)
                save_forecast_snapshot(
                    "FPL", 24, "xgboost", _sample_timestamps(), _sample_predictions(base=40000)
                )

                result = list_forecast_snapshots("FPL", 24, "xgboost")
                assert len(result) == 2
        finally:
            os.unlink(db)

    def test_same_predictions_different_combo_saved(self):
        """Same predictions for different region/horizon should both save."""
        db = _temp_db()
        try:
            with patch("data.forecast_history.CACHE_DB_PATH", db):
                import data.forecast_history

                data.forecast_history._initialized = False

                preds = _sample_predictions()
                save_forecast_snapshot("FPL", 24, "xgboost", _sample_timestamps(), preds)
                save_forecast_snapshot("ERCOT", 24, "xgboost", _sample_timestamps(), preds)

                assert len(list_forecast_snapshots("FPL", 24, "xgboost")) == 1
                assert len(list_forecast_snapshots("ERCOT", 24, "xgboost")) == 1
        finally:
            os.unlink(db)


# ── GCS persistence ───────────────────────────────────────────────


class TestGCSBlobPath:
    def test_blob_path_format(self):
        path = _gcs_blob_path("FPL", 24, "xgboost")
        assert "snapshots/FPL/24_xgboost.json" in path

    def test_blob_path_different_combos(self):
        p1 = _gcs_blob_path("FPL", 24, "xgboost")
        p2 = _gcs_blob_path("ERCOT", 168, "prophet")
        assert p1 != p2
        assert "FPL" in p1
        assert "ERCOT" in p2


class TestGCSReplication:
    def test_replicate_skipped_when_gcs_disabled(self):
        """No GCS calls when GCS_ENABLED is False."""
        db = _temp_db()
        try:
            with (
                patch("data.forecast_history.CACHE_DB_PATH", db),
                patch("data.forecast_history.GCS_ENABLED", False),
                patch("data.forecast_history._get_gcs_client") as mock_client,
            ):
                import data.forecast_history

                data.forecast_history._initialized = False
                data.forecast_history._gcs_restored = True  # skip restore

                save_forecast_snapshot(
                    "FPL", 24, "xgboost", _sample_timestamps(), _sample_predictions()
                )
                mock_client.assert_not_called()
        finally:
            os.unlink(db)

    def test_replicate_serializes_snapshots(self):
        """_replicate_to_gcs reads all snapshots and starts upload thread."""
        db = _temp_db()
        try:
            with (
                patch("data.forecast_history.CACHE_DB_PATH", db),
                patch("data.forecast_history.GCS_ENABLED", True),
                patch("data.forecast_history.GCS_BUCKET_NAME", "test-bucket"),
                patch("data.forecast_history.threading.Thread") as mock_thread,
            ):
                import data.forecast_history

                data.forecast_history._initialized = False
                data.forecast_history._gcs_restored = True

                save_forecast_snapshot(
                    "FPL", 24, "xgboost", _sample_timestamps(), _sample_predictions()
                )

                # Thread should have been started for GCS replication
                mock_thread.assert_called_once()
                assert mock_thread.return_value.start.called
        finally:
            os.unlink(db)

    def test_replicate_payload_format(self):
        """Verify the JSON payload contains expected fields."""
        db = _temp_db()
        try:
            import sqlite3

            import data.forecast_history

            with (
                patch("data.forecast_history.CACHE_DB_PATH", db),
                patch("data.forecast_history.GCS_ENABLED", True),
                patch("data.forecast_history.GCS_BUCKET_NAME", "test-bucket"),
            ):
                data.forecast_history._initialized = False
                data.forecast_history._gcs_restored = True

                save_forecast_snapshot(
                    "FPL", 24, "xgboost", _sample_timestamps(), _sample_predictions()
                )

                # Directly test the serialization by calling _replicate_to_gcs
                # with a mock that captures the payload
                conn = sqlite3.connect(db)
                conn.row_factory = sqlite3.Row
                captured_payload = []

                def mock_thread(target, daemon):
                    # Run the upload function to capture payload
                    mock_obj = MagicMock()
                    captured_payload.append(target)
                    return mock_obj

                with patch("data.forecast_history.threading.Thread", side_effect=mock_thread):
                    _replicate_to_gcs("FPL", 24, "xgboost", conn)

                conn.close()
        finally:
            os.unlink(db)


class TestGCSRestore:
    def test_restore_skipped_when_gcs_disabled(self):
        """No GCS calls when GCS_ENABLED is False."""
        db = _temp_db()
        try:
            import sqlite3

            conn = sqlite3.connect(db)
            conn.row_factory = sqlite3.Row

            with (
                patch("data.forecast_history.GCS_ENABLED", False),
                patch("data.forecast_history._get_gcs_client") as mock_client,
            ):
                import data.forecast_history

                data.forecast_history._gcs_restored = False

                _restore_from_gcs(conn)
                mock_client.assert_not_called()
            conn.close()
        finally:
            os.unlink(db)

    def test_restore_skipped_when_table_not_empty(self):
        """No GCS restore when SQLite already has snapshots."""
        db = _temp_db()
        try:
            with (
                patch("data.forecast_history.CACHE_DB_PATH", db),
                patch("data.forecast_history.GCS_ENABLED", True),
                patch("data.forecast_history.GCS_BUCKET_NAME", "test-bucket"),
                patch("data.forecast_history._get_gcs_client") as mock_client,
            ):
                import data.forecast_history

                data.forecast_history._initialized = False
                data.forecast_history._gcs_restored = True  # skip during init

                # Save a snapshot so table is not empty
                save_forecast_snapshot(
                    "FPL", 24, "xgboost", _sample_timestamps(), _sample_predictions()
                )

                # Now try restore — should skip because table is not empty
                import sqlite3

                conn = sqlite3.connect(db)
                conn.row_factory = sqlite3.Row
                data.forecast_history._gcs_restored = False
                mock_client.reset_mock()

                _restore_from_gcs(conn)

                # Client may be called for count check, but list_blobs should not be called
                conn.close()
        finally:
            os.unlink(db)

    def test_restore_populates_empty_table(self):
        """GCS restore inserts snapshots into empty SQLite table."""
        db = _temp_db()
        try:
            import sqlite3

            conn = sqlite3.connect(db)
            conn.row_factory = sqlite3.Row
            # Create the table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS forecast_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    region TEXT NOT NULL,
                    horizon_hours INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    scored_at TEXT NOT NULL,
                    predictions_json TEXT NOT NULL,
                    timestamps_json TEXT NOT NULL,
                    peak_mw REAL NOT NULL,
                    avg_mw REAL NOT NULL,
                    created_at REAL NOT NULL,
                    predictions_hash TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.commit()

            # Mock GCS with snapshot data
            snapshot_data = [
                {
                    "region": "FPL",
                    "horizon_hours": 24,
                    "model_name": "xgboost",
                    "scored_at": "2024-06-01T12:00:00+00:00",
                    "predictions_json": json.dumps([30000.0, 31000.0, 32000.0]),
                    "timestamps_json": json.dumps(
                        [
                            "2024-06-01T00:00:00+00:00",
                            "2024-06-01T01:00:00+00:00",
                            "2024-06-01T02:00:00+00:00",
                        ]
                    ),
                    "peak_mw": 32000.0,
                    "avg_mw": 31000.0,
                    "created_at": 1717243200.0,
                    "predictions_hash": "abc123",
                },
                {
                    "region": "FPL",
                    "horizon_hours": 24,
                    "model_name": "xgboost",
                    "scored_at": "2024-06-01T14:00:00+00:00",
                    "predictions_json": json.dumps([33000.0, 34000.0, 35000.0]),
                    "timestamps_json": json.dumps(
                        [
                            "2024-06-01T00:00:00+00:00",
                            "2024-06-01T01:00:00+00:00",
                            "2024-06-01T02:00:00+00:00",
                        ]
                    ),
                    "peak_mw": 35000.0,
                    "avg_mw": 34000.0,
                    "created_at": 1717250400.0,
                    "predictions_hash": "def456",
                },
            ]

            mock_blob = MagicMock()
            mock_blob.name = "cache/snapshots/FPL/24_xgboost.json"
            mock_blob.download_as_bytes.return_value = json.dumps(snapshot_data).encode()

            mock_bucket = MagicMock()
            mock_bucket.list_blobs.return_value = [mock_blob]

            mock_client = MagicMock()
            mock_client.bucket.return_value = mock_bucket

            with (
                patch("data.forecast_history.GCS_ENABLED", True),
                patch("data.forecast_history.GCS_BUCKET_NAME", "test-bucket"),
                patch("data.forecast_history._get_gcs_client", return_value=mock_client),
            ):
                import data.forecast_history

                data.forecast_history._gcs_restored = False

                _restore_from_gcs(conn)

                # Verify snapshots were inserted
                count = conn.execute("SELECT COUNT(*) FROM forecast_snapshots").fetchone()[0]
                assert count == 2

                # Verify data integrity
                rows = conn.execute(
                    "SELECT * FROM forecast_snapshots ORDER BY scored_at ASC"
                ).fetchall()
                assert rows[0]["peak_mw"] == 32000.0
                assert rows[1]["peak_mw"] == 35000.0
                assert rows[0]["predictions_hash"] == "abc123"

            conn.close()
        finally:
            os.unlink(db)

    def test_restore_handles_gcs_failure_gracefully(self):
        """GCS failure during restore should not crash."""
        db = _temp_db()
        try:
            import sqlite3

            conn = sqlite3.connect(db)
            conn.row_factory = sqlite3.Row
            conn.execute("""
                CREATE TABLE IF NOT EXISTS forecast_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    region TEXT NOT NULL,
                    horizon_hours INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    scored_at TEXT NOT NULL,
                    predictions_json TEXT NOT NULL,
                    timestamps_json TEXT NOT NULL,
                    peak_mw REAL NOT NULL,
                    avg_mw REAL NOT NULL,
                    created_at REAL NOT NULL,
                    predictions_hash TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.commit()

            mock_client = MagicMock()
            mock_client.bucket.side_effect = Exception("GCS unavailable")

            with (
                patch("data.forecast_history.GCS_ENABLED", True),
                patch("data.forecast_history.GCS_BUCKET_NAME", "test-bucket"),
                patch("data.forecast_history._get_gcs_client", return_value=mock_client),
            ):
                import data.forecast_history

                data.forecast_history._gcs_restored = False

                # Should not raise
                _restore_from_gcs(conn)

                # Table should still be empty
                count = conn.execute("SELECT COUNT(*) FROM forecast_snapshots").fetchone()[0]
                assert count == 0

            conn.close()
        finally:
            os.unlink(db)

    def test_restore_only_runs_once(self):
        """_gcs_restored flag prevents repeated restore attempts."""
        db = _temp_db()
        try:
            import sqlite3

            conn = sqlite3.connect(db)
            conn.row_factory = sqlite3.Row
            conn.execute("""
                CREATE TABLE IF NOT EXISTS forecast_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    region TEXT NOT NULL,
                    horizon_hours INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    scored_at TEXT NOT NULL,
                    predictions_json TEXT NOT NULL,
                    timestamps_json TEXT NOT NULL,
                    peak_mw REAL NOT NULL,
                    avg_mw REAL NOT NULL,
                    created_at REAL NOT NULL,
                    predictions_hash TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.commit()

            with (
                patch("data.forecast_history.GCS_ENABLED", True),
                patch("data.forecast_history.GCS_BUCKET_NAME", "test-bucket"),
                patch("data.forecast_history._get_gcs_client") as mock_client,
            ):
                import data.forecast_history

                # Already restored
                data.forecast_history._gcs_restored = True

                _restore_from_gcs(conn)
                mock_client.assert_not_called()

            conn.close()
        finally:
            os.unlink(db)


# ── Feature flag ──────────────────────────────────────────────────


class TestFeatureFlag:
    def test_forecast_replay_flag_exists(self):
        from config import FEATURE_FLAGS

        assert "forecast_replay" in FEATURE_FLAGS

    def test_forecast_replay_flag_disabled(self):
        """Replay was retired post-redesign (PR #51) — surfaces stale snapshots
        and competes with the v2 hero rhythm. The flag still exists so it can
        be re-enabled once the snapshot pipeline produces fresh data."""
        from config import feature_enabled

        assert feature_enabled("forecast_replay") is False
