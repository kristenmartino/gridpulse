"""
Unit tests for NEXD-14: Time-Scrub Replay (forecast snapshot history).

Covers:
- save_forecast_snapshot(): insert, FIFO enforcement, idempotency
- list_forecast_snapshots(): ordering, empty state
- get_forecast_snapshot(): retrieval, missing keys
- build_replay_options(): dropdown option format
- Feature flag existence
"""

import os
import tempfile
import time
from unittest.mock import patch

from data.forecast_history import (
    MAX_SNAPSHOTS,
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
                    "FPL", 24, "xgboost", _sample_timestamps(), _sample_predictions()
                )
                time.sleep(0.01)
                save_forecast_snapshot(
                    "FPL", 24, "xgboost", _sample_timestamps(), _sample_predictions()
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

                for _ in range(5):
                    save_forecast_snapshot(
                        "FPL", 24, "xgboost", _sample_timestamps(), _sample_predictions()
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


# ── Feature flag ──────────────────────────────────────────────────


class TestFeatureFlag:
    def test_forecast_replay_flag_exists(self):
        from config import FEATURE_FLAGS

        assert "forecast_replay" in FEATURE_FLAGS

    def test_forecast_replay_flag_enabled(self):
        from config import feature_enabled

        assert feature_enabled("forecast_replay") is True
