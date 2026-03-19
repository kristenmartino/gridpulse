"""
Model Store — Filesystem-based model persistence via joblib.

Handles saving, loading, age checking, and cleanup of trained model artifacts.
Models are stored at models/artifacts/{region}/ with timestamped filenames
and a metadata.json sidecar for weights, metrics, and feature info.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)


class ModelStore:
    """
    Persists trained models to disk and loads them for inference.

    Directory structure:
        {base_dir}/{region}/
            {model}_{timestamp}.joblib   (e.g. xgboost_20260317_0200.joblib)
            metadata.json                (weights, metrics, trained_at, feature info)
    """

    def __init__(self, base_dir: str = "models/artifacts"):
        self.base_dir = Path(base_dir)

    def _region_dir(self, region: str) -> Path:
        return self.base_dir / region

    def _timestamp_str(self) -> str:
        return datetime.now(UTC).strftime("%Y%m%d_%H%M")

    def save_models(
        self,
        region: str,
        models: dict,
        weights: dict,
        metrics: dict,
        feature_cols: list[str],
        feature_hash: str,
    ) -> Path:
        """
        Save trained models + metadata to disk.

        Args:
            region: Grid region code (e.g. "ERCOT").
            models: Dict of model name -> trained model object.
            weights: Ensemble weights dict.
            metrics: Per-model metrics dict.
            feature_cols: Feature column names used for training.
            feature_hash: Hash of feature columns for audit.

        Returns:
            Path to the region directory where artifacts were saved.
        """
        region_dir = self._region_dir(region)
        region_dir.mkdir(parents=True, exist_ok=True)
        ts = self._timestamp_str()

        # Save each model as a joblib file
        artifact_paths = {}
        for model_name, model_obj in models.items():
            filename = f"{model_name}_{ts}.joblib"
            path = region_dir / filename
            joblib.dump(model_obj, path)
            artifact_paths[model_name] = filename
            logger.info("Saved %s model for %s: %s", model_name, region, path)

        # Write metadata sidecar
        metadata = {
            "region": region,
            "trained_at": datetime.now(UTC).isoformat(),
            "artifact_paths": artifact_paths,
            "weights": weights,
            "metrics": metrics,
            "feature_cols": feature_cols,
            "feature_hash": feature_hash,
        }
        meta_path = region_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, default=str))

        logger.info(
            "Saved %d models for %s (weights=%s)",
            len(models),
            region,
            weights,
        )
        return region_dir

    def load_models(self, region: str) -> dict | None:
        """
        Load the latest persisted models for a region.

        Returns:
            Dict with keys: models, weights, metrics, feature_cols,
            feature_hash, trained_at. Returns None if no artifacts exist.
        """
        region_dir = self._region_dir(region)
        meta_path = region_dir / "metadata.json"

        if not meta_path.exists():
            logger.warning("No model artifacts found for %s", region)
            return None

        try:
            metadata = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to read metadata for %s: %s", region, e)
            return None

        # Load each model
        models = {}
        for model_name, filename in metadata.get("artifact_paths", {}).items():
            path = region_dir / filename
            if not path.exists():
                logger.warning("Artifact missing for %s/%s: %s", region, model_name, path)
                continue
            try:
                models[model_name] = joblib.load(path)
            except Exception as e:
                logger.error("Failed to load %s/%s: %s", region, model_name, e)

        if not models:
            logger.warning("No valid model artifacts loaded for %s", region)
            return None

        return {
            "models": models,
            "weights": metadata.get("weights", {}),
            "metrics": metadata.get("metrics", {}),
            "feature_cols": metadata.get("feature_cols", []),
            "feature_hash": metadata.get("feature_hash", ""),
            "trained_at": metadata.get("trained_at", ""),
        }

    def model_age_hours(self, region: str) -> float | None:
        """
        How many hours since models were last trained.

        Returns None if no artifacts exist.
        """
        region_dir = self._region_dir(region)
        meta_path = region_dir / "metadata.json"

        if not meta_path.exists():
            return None

        try:
            metadata = json.loads(meta_path.read_text())
            trained_at = datetime.fromisoformat(metadata["trained_at"])
            if trained_at.tzinfo is None:
                trained_at = trained_at.replace(tzinfo=UTC)
            age = datetime.now(UTC) - trained_at
            return age.total_seconds() / 3600
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to compute model age for %s: %s", region, e)
            return None

    def cleanup_old(self, region: str, keep_n: int = 3):
        """
        Remove all but the N most recent model snapshots for a region.

        Keeps the latest metadata.json and its referenced artifacts.
        Removes orphaned .joblib files from older runs.
        """
        region_dir = self._region_dir(region)
        if not region_dir.exists():
            return

        # Find all joblib files, sorted by modification time (newest first)
        joblib_files = sorted(
            region_dir.glob("*.joblib"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Read current metadata to know which files are active
        meta_path = region_dir / "metadata.json"
        active_files = set()
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text())
                active_files = {region_dir / f for f in metadata.get("artifact_paths", {}).values()}
            except (json.JSONDecodeError, OSError):
                pass

        # Keep active files + the N most recent additional files
        keep_files = set(active_files)
        count = 0
        for f in joblib_files:
            if f in keep_files:
                continue
            if count < keep_n:
                keep_files.add(f)
                count += 1

        # Remove files not in keep set
        removed = 0
        for f in joblib_files:
            if f not in keep_files:
                f.unlink()
                removed += 1

        if removed > 0:
            logger.info("Cleaned up %d old artifacts for %s", removed, region)

    def has_models(self, region: str) -> bool:
        """Check if persisted models exist for a region."""
        meta_path = self._region_dir(region) / "metadata.json"
        return meta_path.exists()
