"""
GCS-backed model persistence for GridPulse.

Trained models are pickled and uploaded to
    gs://{bucket}/{prefix}/models/{region}/{model_name}/{version}.pkl
with a sibling ``.meta.json`` describing the data hash, training timestamp,
row count, MAPE, and library versions.

A single ``latest.json`` pointer at the top of the model tree maps
``{region: {model_name: version}}`` so the scoring job can load the newest
artifact without listing the bucket.

Design:
- Save flow: upload blob → upload meta → overwrite ``latest.json`` atomically
  (last-writer-wins is acceptable — training job runs once per day).
- Load flow: read ``latest.json`` → download blob (with local-disk cache)
  → unpickle. Local cache keyed by version so stale entries are ignored.
- Never raises during normal operation; callers must handle ``None`` returns.
- Completely optional: if GCS is disabled the module short-circuits to the
  in-memory caches that already exist in ``components.callbacks``.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import platform
import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from config import GCS_BUCKET_NAME, GCS_ENABLED, GCS_PATH_PREFIX

if TYPE_CHECKING:
    from google.cloud.storage import Client

log = structlog.get_logger()

# ── Constants ────────────────────────────────────────────────

MODELS_PREFIX = "models"
LATEST_POINTER = "latest.json"
LOCAL_MODEL_CACHE_DIR = os.getenv("MODEL_LOCAL_CACHE_DIR", "/tmp/gridpulse-models")

_client: Client | None = None
_client_lock = threading.Lock()
_latest_cache: dict[str, dict[str, str]] | None = None
_latest_cache_lock = threading.Lock()


# ── GCS client ───────────────────────────────────────────────


def _get_client() -> Client | None:
    """Return a lazy-initialized GCS client or ``None`` if GCS is disabled."""
    global _client
    if not GCS_ENABLED or not GCS_BUCKET_NAME:
        return None
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        try:
            from google.cloud.storage import Client as StorageClient

            _client = StorageClient()
            log.info("model_persistence_client_initialized", bucket=GCS_BUCKET_NAME)
        except Exception as e:
            log.warning("model_persistence_client_init_failed", error=str(e))
            return None
    return _client


def _model_prefix() -> str:
    return f"{GCS_PATH_PREFIX.rstrip('/')}/{MODELS_PREFIX}" if GCS_PATH_PREFIX else MODELS_PREFIX


def _blob_path(region: str, model_name: str, version: str, suffix: str) -> str:
    """Build the blob path for a model artifact (``.pkl`` or ``.meta.json``)."""
    return f"{_model_prefix()}/{region}/{model_name}/{version}{suffix}"


def _latest_path() -> str:
    return f"{_model_prefix()}/{LATEST_POINTER}"


# ── Metadata shape ───────────────────────────────────────────


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata persisted alongside each pickled model blob."""

    region: str
    model_name: str
    version: str
    data_hash: str
    trained_at: str
    train_rows: int
    mape: float | None
    lib_versions: dict[str, str]
    extra: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "model_name": self.model_name,
            "version": self.version,
            "data_hash": self.data_hash,
            "trained_at": self.trained_at,
            "train_rows": self.train_rows,
            "mape": self.mape,
            "lib_versions": self.lib_versions,
            "extra": self.extra,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ModelMetadata:
        return ModelMetadata(
            region=d["region"],
            model_name=d["model_name"],
            version=d["version"],
            data_hash=d.get("data_hash", ""),
            trained_at=d.get("trained_at", ""),
            train_rows=int(d.get("train_rows", 0)),
            mape=d.get("mape"),
            lib_versions=d.get("lib_versions", {}),
            extra=d.get("extra", {}),
        )


def _collect_lib_versions() -> dict[str, str]:
    """Capture the major library versions used during training.

    Stored in metadata so the scoring job can warn on a cross-major skew
    (e.g., xgboost trained on 2.x being loaded under 1.x).
    """
    versions = {
        "python": platform.python_version(),
    }
    for mod in ("pandas", "numpy", "xgboost", "prophet", "statsmodels", "sklearn"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            continue
    return versions


def _new_version() -> str:
    """Generate a timestamp-based version string (UTC, second resolution)."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# ── Local disk cache ─────────────────────────────────────────


def _local_cache_path(region: str, model_name: str, version: str) -> str:
    region_dir = os.path.join(LOCAL_MODEL_CACHE_DIR, region, model_name)
    os.makedirs(region_dir, exist_ok=True)
    return os.path.join(region_dir, f"{version}.pkl")


def _load_from_local(region: str, model_name: str, version: str) -> bytes | None:
    path = _local_cache_path(region, model_name, version)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception as e:
        log.debug("model_local_cache_read_failed", path=path, error=str(e))
        return None


def _store_in_local(region: str, model_name: str, version: str, data: bytes) -> None:
    path = _local_cache_path(region, model_name, version)
    try:
        with open(path, "wb") as f:
            f.write(data)
    except Exception as e:
        log.debug("model_local_cache_write_failed", path=path, error=str(e))


# ── latest.json ──────────────────────────────────────────────


def _read_latest(force: bool = False) -> dict[str, dict[str, str]]:
    """Read the ``latest.json`` pointer. Returns an empty dict on any failure."""
    global _latest_cache
    if _latest_cache is not None and not force:
        return _latest_cache
    client = _get_client()
    if client is None:
        _latest_cache = {}
        return _latest_cache
    try:
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(_latest_path())
        if not blob.exists():
            _latest_cache = {}
            return _latest_cache
        data = blob.download_as_text()
        parsed = json.loads(data)
        if not isinstance(parsed, dict):
            log.warning("model_latest_invalid_shape", path=_latest_path())
            _latest_cache = {}
            return _latest_cache
        with _latest_cache_lock:
            _latest_cache = parsed
        return parsed
    except Exception as e:
        log.warning("model_latest_read_failed", error=str(e))
        _latest_cache = {}
        return {}


def _write_latest(region: str, model_name: str, version: str) -> None:
    """Merge a single (region, model_name, version) entry into ``latest.json``.

    Last-writer-wins. The training job is the only writer, so contention is
    not a concern in normal operation.
    """
    client = _get_client()
    if client is None:
        return
    try:
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(_latest_path())
        current: dict[str, dict[str, str]] = {}
        if blob.exists():
            try:
                current = json.loads(blob.download_as_text())
                if not isinstance(current, dict):
                    current = {}
            except Exception as e:
                log.warning("model_latest_merge_read_failed", error=str(e))
                current = {}
        current.setdefault(region, {})[model_name] = version
        payload = json.dumps(current, indent=2, sort_keys=True)
        blob.upload_from_string(payload, content_type="application/json")
        with _latest_cache_lock:
            global _latest_cache
            _latest_cache = current
        log.info(
            "model_latest_updated",
            region=region,
            model=model_name,
            version=version,
        )
    except Exception as e:
        log.warning(
            "model_latest_write_failed",
            region=region,
            model=model_name,
            version=version,
            error=str(e),
        )


# ── Public API ───────────────────────────────────────────────


def save_model(
    region: str,
    model_name: str,
    model_obj: Any,
    data_hash: str,
    train_rows: int,
    mape: float | None = None,
    extra: dict[str, Any] | None = None,
) -> str | None:
    """Pickle ``model_obj`` and upload it plus metadata to GCS.

    Returns the assigned version string, or ``None`` if GCS is disabled or
    the upload failed. The ``latest.json`` pointer is updated on success.
    """
    client = _get_client()
    if client is None:
        log.info(
            "model_save_skipped_gcs_disabled",
            region=region,
            model=model_name,
        )
        return None

    version = _new_version()
    try:
        buf = io.BytesIO()
        pickle.dump(model_obj, buf, protocol=pickle.HIGHEST_PROTOCOL)
        model_bytes = buf.getvalue()
    except Exception as e:
        log.warning(
            "model_pickle_failed",
            region=region,
            model=model_name,
            error=str(e),
        )
        return None

    metadata = ModelMetadata(
        region=region,
        model_name=model_name,
        version=version,
        data_hash=data_hash,
        trained_at=datetime.now(UTC).isoformat(),
        train_rows=train_rows,
        mape=mape,
        lib_versions=_collect_lib_versions(),
        extra=extra or {},
    )

    try:
        bucket = client.bucket(GCS_BUCKET_NAME)
        model_blob = bucket.blob(_blob_path(region, model_name, version, ".pkl"))
        model_blob.upload_from_string(model_bytes, content_type="application/octet-stream")
        meta_blob = bucket.blob(_blob_path(region, model_name, version, ".meta.json"))
        meta_blob.upload_from_string(
            json.dumps(metadata.to_dict(), indent=2),
            content_type="application/json",
        )
    except Exception as e:
        log.warning(
            "model_upload_failed",
            region=region,
            model=model_name,
            version=version,
            error=str(e),
        )
        return None

    _store_in_local(region, model_name, version, model_bytes)
    _write_latest(region, model_name, version)

    log.info(
        "model_saved",
        region=region,
        model=model_name,
        version=version,
        size_bytes=len(model_bytes),
        mape=mape,
    )
    return version


def get_model_metadata(region: str, model_name: str) -> ModelMetadata | None:
    """Fetch metadata for the latest version of ``(region, model_name)``.

    Does not download the model pickle — cheap enough to call in scoring
    to detect stale artifacts.
    """
    client = _get_client()
    if client is None:
        return None
    latest = _read_latest()
    version = latest.get(region, {}).get(model_name)
    if not version:
        return None
    try:
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(_blob_path(region, model_name, version, ".meta.json"))
        if not blob.exists():
            return None
        return ModelMetadata.from_dict(json.loads(blob.download_as_text()))
    except Exception as e:
        log.warning(
            "model_metadata_read_failed",
            region=region,
            model=model_name,
            error=str(e),
        )
        return None


def load_model(region: str, model_name: str) -> tuple[Any, ModelMetadata] | None:
    """Load the latest saved model for ``(region, model_name)``.

    Uses a local disk cache under :data:`LOCAL_MODEL_CACHE_DIR` keyed by
    version so repeated loads on the same container don't re-download.

    Returns ``None`` if GCS is disabled, the model doesn't exist, or any
    step fails.
    """
    client = _get_client()
    if client is None:
        return None
    latest = _read_latest()
    version = latest.get(region, {}).get(model_name)
    if not version:
        log.debug(
            "model_load_no_pointer",
            region=region,
            model=model_name,
        )
        return None

    metadata = get_model_metadata(region, model_name)
    if metadata is None:
        log.warning(
            "model_load_metadata_missing",
            region=region,
            model=model_name,
            version=version,
        )
        return None

    _warn_if_lib_skew(metadata)

    cached_bytes = _load_from_local(region, model_name, version)
    if cached_bytes is not None:
        try:
            model_obj = pickle.loads(cached_bytes)
            log.debug(
                "model_loaded_from_local_cache",
                region=region,
                model=model_name,
                version=version,
            )
            return model_obj, metadata
        except Exception as e:
            log.warning(
                "model_local_unpickle_failed",
                region=region,
                model=model_name,
                version=version,
                error=str(e),
            )

    try:
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(_blob_path(region, model_name, version, ".pkl"))
        if not blob.exists():
            log.warning(
                "model_blob_missing",
                region=region,
                model=model_name,
                version=version,
            )
            return None
        data = blob.download_as_bytes()
    except Exception as e:
        log.warning(
            "model_download_failed",
            region=region,
            model=model_name,
            version=version,
            error=str(e),
        )
        return None

    try:
        model_obj = pickle.loads(data)
    except Exception as e:
        log.warning(
            "model_unpickle_failed",
            region=region,
            model=model_name,
            version=version,
            error=str(e),
        )
        return None

    _store_in_local(region, model_name, version, data)
    log.info(
        "model_loaded",
        region=region,
        model=model_name,
        version=version,
        size_bytes=len(data),
    )
    return model_obj, metadata


def invalidate_latest_cache() -> None:
    """Force the next :func:`load_model` / :func:`get_model_metadata` call to re-read ``latest.json``."""
    global _latest_cache
    with _latest_cache_lock:
        _latest_cache = None


def _warn_if_lib_skew(metadata: ModelMetadata) -> None:
    """Log a warning when a loaded model's major library versions differ from the current runtime."""
    runtime = _collect_lib_versions()
    skews: dict[str, tuple[str, str]] = {}
    for lib, trained_version in metadata.lib_versions.items():
        current = runtime.get(lib)
        if not current or not trained_version:
            continue
        trained_major = trained_version.split(".")[0]
        current_major = current.split(".")[0]
        if trained_major != current_major:
            skews[lib] = (trained_version, current)
    if skews:
        log.warning(
            "model_lib_version_skew",
            region=metadata.region,
            model=metadata.model_name,
            version=metadata.version,
            skews=skews,
        )


__all__ = [
    "ModelMetadata",
    "save_model",
    "load_model",
    "get_model_metadata",
    "invalidate_latest_cache",
    "LOCAL_MODEL_CACHE_DIR",
]


# ── Self-test CLI (manual) ───────────────────────────────────

if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) >= 3 and sys.argv[1] == "inspect":
        _r, _m = sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None
        if _m:
            meta = get_model_metadata(_r, _m)
            print(json.dumps(meta.to_dict() if meta else {}, indent=2))
        else:
            print(json.dumps(_read_latest(force=True).get(_r, {}), indent=2))
    else:
        print(json.dumps(_read_latest(force=True), indent=2))
