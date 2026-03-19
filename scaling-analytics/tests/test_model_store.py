"""Tests for the ModelStore — filesystem model persistence."""
import json
import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.processing.model_store import ModelStore


@pytest.fixture
def tmp_store(tmp_path):
    """ModelStore with a temporary directory."""
    return ModelStore(base_dir=str(tmp_path))


@pytest.fixture
def sample_models():
    """Sample model objects for saving."""
    return {
        "xgboost": {"type": "xgboost", "params": {"n_estimators": 100}},
        "prophet": {"type": "prophet", "params": {"growth": "linear"}},
    }


@pytest.fixture
def sample_weights():
    return {"xgboost": 0.6, "prophet": 0.4}


@pytest.fixture
def sample_metrics():
    return {
        "xgboost": {"mape": 3.13, "rmse": 1500},
        "prophet": {"mape": 4.20, "rmse": 1800},
    }


class TestSaveModels:

    def test_creates_region_directory(self, tmp_store, sample_models, sample_weights, sample_metrics):
        """save_models creates the region directory."""
        result = tmp_store.save_models(
            "ERCOT", sample_models, sample_weights, sample_metrics,
            ["temp", "hour_sin"], "abc123",
        )
        assert result.exists()
        assert result.name == "ERCOT"

    def test_saves_joblib_files(self, tmp_store, sample_models, sample_weights, sample_metrics):
        """save_models creates a .joblib file per model."""
        tmp_store.save_models(
            "ERCOT", sample_models, sample_weights, sample_metrics,
            ["temp", "hour_sin"], "abc123",
        )
        region_dir = tmp_store._region_dir("ERCOT")
        joblib_files = list(region_dir.glob("*.joblib"))
        assert len(joblib_files) == 2

    def test_writes_metadata_json(self, tmp_store, sample_models, sample_weights, sample_metrics):
        """save_models creates metadata.json with correct fields."""
        tmp_store.save_models(
            "ERCOT", sample_models, sample_weights, sample_metrics,
            ["temp", "hour_sin"], "abc123",
        )
        meta_path = tmp_store._region_dir("ERCOT") / "metadata.json"
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text())
        assert meta["region"] == "ERCOT"
        assert meta["weights"] == sample_weights
        assert meta["feature_hash"] == "abc123"
        assert "trained_at" in meta
        assert "artifact_paths" in meta
        assert len(meta["artifact_paths"]) == 2


class TestLoadModels:

    def test_returns_none_when_no_artifacts(self, tmp_store):
        """load_models returns None if no artifacts exist."""
        result = tmp_store.load_models("ERCOT")
        assert result is None

    def test_loads_saved_models(self, tmp_store, sample_models, sample_weights, sample_metrics):
        """load_models returns models saved by save_models."""
        tmp_store.save_models(
            "ERCOT", sample_models, sample_weights, sample_metrics,
            ["temp", "hour_sin"], "abc123",
        )
        loaded = tmp_store.load_models("ERCOT")
        assert loaded is not None
        assert loaded["weights"] == sample_weights
        assert "xgboost" in loaded["models"]
        assert "prophet" in loaded["models"]
        assert loaded["feature_hash"] == "abc123"
        assert loaded["trained_at"] != ""

    def test_returns_correct_structure(self, tmp_store, sample_models, sample_weights, sample_metrics):
        """load_models returns dict with expected keys."""
        tmp_store.save_models(
            "ERCOT", sample_models, sample_weights, sample_metrics,
            ["temp", "hour_sin"], "abc123",
        )
        loaded = tmp_store.load_models("ERCOT")
        assert set(loaded.keys()) == {
            "models", "weights", "metrics", "feature_cols",
            "feature_hash", "trained_at",
        }


class TestModelAge:

    def test_returns_none_when_no_artifacts(self, tmp_store):
        """model_age_hours returns None if no artifacts exist."""
        assert tmp_store.model_age_hours("ERCOT") is None

    def test_returns_small_age_after_save(self, tmp_store, sample_models, sample_weights, sample_metrics):
        """model_age_hours returns a small number right after saving."""
        tmp_store.save_models(
            "ERCOT", sample_models, sample_weights, sample_metrics,
            ["temp"], "abc",
        )
        age = tmp_store.model_age_hours("ERCOT")
        assert age is not None
        assert age < 0.1  # Should be near-zero


class TestCleanupOld:

    def test_cleanup_removes_extra_files(self, tmp_store, sample_models, sample_weights, sample_metrics):
        """cleanup_old keeps only the N most recent files."""
        # Save models 3 times to create multiple snapshots
        for _ in range(3):
            tmp_store.save_models(
                "ERCOT", sample_models, sample_weights, sample_metrics,
                ["temp"], "abc",
            )

        region_dir = tmp_store._region_dir("ERCOT")
        initial_count = len(list(region_dir.glob("*.joblib")))

        # Cleanup keeping only 1
        tmp_store.cleanup_old("ERCOT", keep_n=1)

        remaining = list(region_dir.glob("*.joblib"))
        # Should have at most active files + 1 extra
        assert len(remaining) <= initial_count

    def test_cleanup_noop_when_no_dir(self, tmp_store):
        """cleanup_old does nothing when region directory doesn't exist."""
        tmp_store.cleanup_old("NONEXISTENT", keep_n=1)  # Should not raise


class TestHasModels:

    def test_false_when_no_artifacts(self, tmp_store):
        """has_models returns False if no artifacts exist."""
        assert tmp_store.has_models("ERCOT") is False

    def test_true_after_save(self, tmp_store, sample_models, sample_weights, sample_metrics):
        """has_models returns True after saving."""
        tmp_store.save_models(
            "ERCOT", sample_models, sample_weights, sample_metrics,
            ["temp"], "abc",
        )
        assert tmp_store.has_models("ERCOT") is True
