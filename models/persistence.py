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
import time
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
# `/tmp` is the POSIX dev/CI default; production overrides via
# MODEL_LOCAL_CACHE_DIR (e.g. `/app/trained_models/` in the Cloud Run image,
# wired in the Dockerfile so the cache survives across requests within a
# container instance).
LOCAL_MODEL_CACHE_DIR = os.getenv("MODEL_LOCAL_CACHE_DIR", "/tmp/gridpulse-models")

_client: Client | None = None
_client_lock = threading.Lock()
_latest_cache: dict[str, dict[str, str]] | None = None
_latest_cache_lock = threading.Lock()
#: Monotonic time the ``latest.json`` pointer cache was last populated. Before
#: #271 (P2-10) the cache was pinned for the whole process: the writer refreshes
#: its own copy after a persist, but the *web* tier never writes and never
#: invalidated, so a long-lived web process kept serving the model versions it
#: saw on first read and never reflected daily retraining. A TTL makes the web
#: tier re-read the pointer periodically (bounded — at most one GCS read per TTL,
#: not per render) so new versions surface without a redeploy.
_latest_cache_ts: float = 0.0
_LATEST_CACHE_TTL = 300.0
#: #272 (P2-11): whether the cached value came from a FAILED read. A failure
#: never overwrites a previously-good pointer (last-known-good is served
#: instead) and is re-probed on the short ``_LATEST_FAILURE_TTL`` — the old
#: behavior negative-cached the ``{}`` sentinel at the full TTL, making the
#: whole model store unloadable for minutes after one transient GCS blip.
_latest_cache_is_failure: bool = False
_LATEST_FAILURE_TTL = 30.0
_LATEST_READ_RETRIES = 2  # quick in-call retries before failure semantics engage
_LATEST_READ_RETRY_BACKOFF_S = 0.5


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


def _cache_latest_success(value: dict) -> dict:
    """Cache a VALID pointer read (incl. a legitimately-missing pointer's ``{}``)
    at the normal TTL."""
    global _latest_cache, _latest_cache_ts, _latest_cache_is_failure
    with _latest_cache_lock:
        _latest_cache = value
        _latest_cache_ts = time.monotonic()
        _latest_cache_is_failure = False
    return value


def _handle_latest_failure(reason: str) -> dict:
    """Failure semantics for a pointer read (#272 / P2-11): never let a blip
    erase a valid pointer.

    * A previously-read value (good pointer OR a legitimately-empty ``{}``)
      keeps being served — **last-known-good** — so a transient GCS outage
      mid-process never makes the whole model store unloadable. The pointer
      only changes on the daily training write, so stale-by-minutes is safe.
    * Only when there is NO prior value does the sentinel ``{}`` get cached,
      and then only for ``_LATEST_FAILURE_TTL`` (not the full success TTL), so
      a cold process re-probes within seconds instead of serving an empty
      model store for five minutes.
    """
    global _latest_cache, _latest_cache_ts, _latest_cache_is_failure
    with _latest_cache_lock:
        if _latest_cache is None:
            _latest_cache = {}
        else:
            log.warning("model_latest_serving_stale", reason=reason)
        _latest_cache_ts = time.monotonic()
        _latest_cache_is_failure = True  # short re-probe window either way
        return _latest_cache


def _read_latest(force: bool = False) -> dict[str, dict[str, str]]:
    """Read the ``latest.json`` pointer.

    The cache is honoured only within its TTL (unless ``force``): the normal
    ``_LATEST_CACHE_TTL`` after a successful read (#271 / P2-10 — a long-lived
    web process reflects daily retraining), or the short ``_LATEST_FAILURE_TTL``
    after a FAILED read (#272 / P2-11 — a blip re-probes in seconds). Failures
    retry briefly in-call and then fall back to the last-known-good pointer
    rather than negative-caching an empty store; see ``_handle_latest_failure``.
    """
    global _latest_cache, _latest_cache_ts
    ttl = _LATEST_FAILURE_TTL if _latest_cache_is_failure else _LATEST_CACHE_TTL
    fresh = (time.monotonic() - _latest_cache_ts) < ttl
    if _latest_cache is not None and fresh and not force:
        return _latest_cache
    client = _get_client()
    if client is None:
        if not GCS_ENABLED:
            # Dev / GCS-off: an empty store is the VALID answer, not a failure.
            return _cache_latest_success({})
        return _handle_latest_failure("client_unavailable")
    last_exc: Exception | None = None
    for attempt in range(1 + _LATEST_READ_RETRIES):
        try:
            bucket = client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(_latest_path())
            if not blob.exists():
                # Legitimately-missing pointer (fresh bucket) — a valid result.
                return _cache_latest_success({})
            parsed = json.loads(blob.download_as_text())
            if not isinstance(parsed, dict):
                # Corrupted pointer content — won't self-heal until the next
                # training write; failure semantics (keep last-known-good).
                log.warning("model_latest_invalid_shape", path=_latest_path())
                return _handle_latest_failure("invalid_shape")
            return _cache_latest_success(parsed)
        except Exception as e:
            last_exc = e
            if attempt < _LATEST_READ_RETRIES:
                time.sleep(_LATEST_READ_RETRY_BACKOFF_S * (attempt + 1))
    log.warning("model_latest_read_failed", error=str(last_exc))
    return _handle_latest_failure("read_failed")


_LATEST_WRITE_MAX_RETRIES = 5


def _write_latest(region: str, model_name: str, version: str) -> None:
    """Merge a single (region, model_name, version) entry into ``latest.json``.

    Uses GCS optimistic concurrency (``If-Generation-Match`` precondition)
    so concurrent training-job tasks can't silently drop each other's
    pointer updates. Earlier last-writer-wins implementation was unsafe
    under ``taskCount>1`` parallel training — two tasks reading the same
    snapshot and writing back would each preserve only their own entry,
    losing the sibling task's just-written pointer.

    Retries up to ``_LATEST_WRITE_MAX_RETRIES`` times on
    ``PreconditionFailed`` (412) — each retry re-reads the current
    blob so the merge picks up whatever the racing writer just wrote.
    If we still can't commit after the retry budget we log and bail;
    the lost write means one model version is orphaned in GCS but the
    pickle itself is still there and ``get_model_metadata`` falls back
    to the legacy meta path until the next training run reconciles.
    """
    client = _get_client()
    if client is None:
        return
    try:
        from google.api_core.exceptions import PreconditionFailed
    except Exception:  # pragma: no cover — google-cloud not installed in dev
        PreconditionFailed = Exception  # type: ignore[assignment, misc] # noqa: N806

    try:
        bucket = client.bucket(GCS_BUCKET_NAME)
        for attempt in range(_LATEST_WRITE_MAX_RETRIES):
            blob = bucket.blob(_latest_path())
            current: dict[str, dict[str, str]] = {}
            generation: int | None = None
            if blob.exists():
                # ``reload`` populates ``blob.generation`` — the generation
                # number is what makes the conditional write race-safe.
                blob.reload()
                generation = blob.generation
                try:
                    current = json.loads(blob.download_as_text())
                    if not isinstance(current, dict):
                        current = {}
                except Exception as e:
                    log.warning("model_latest_merge_read_failed", error=str(e))
                    current = {}
            current.setdefault(region, {})[model_name] = version
            payload = json.dumps(current, indent=2, sort_keys=True)
            try:
                # ``if_generation_match=0`` means "only if no version
                # exists yet" — the first writer wins; subsequent
                # writers see the generation and merge against it.
                blob.upload_from_string(
                    payload,
                    content_type="application/json",
                    if_generation_match=generation if generation is not None else 0,
                )
                break
            except PreconditionFailed:
                log.info(
                    "model_latest_race_retrying",
                    region=region,
                    model=model_name,
                    attempt=attempt + 1,
                    max=_LATEST_WRITE_MAX_RETRIES,
                )
                continue
        else:
            log.warning(
                "model_latest_race_exhausted",
                region=region,
                model=model_name,
                version=version,
                max=_LATEST_WRITE_MAX_RETRIES,
            )
            return
        with _latest_cache_lock:
            global _latest_cache, _latest_cache_ts, _latest_cache_is_failure
            _latest_cache = current
            _latest_cache_ts = time.monotonic()
            _latest_cache_is_failure = False
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
    update_latest: bool = True,
) -> str | None:
    """Pickle ``model_obj`` and upload it plus metadata to GCS.

    Returns the assigned version string, or ``None`` if GCS is disabled or
    the upload failed. The ``latest.json`` pointer is updated on success —
    unless ``update_latest`` is False (the #326 serve-path gate rejecting a
    candidate): the artifact is still persisted as the forensic record, but
    the pointer keeps serving the previously accepted version.
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
    if update_latest:
        _write_latest(region, model_name, version)

    log.info(
        "model_saved",
        region=region,
        model=model_name,
        version=version,
        size_bytes=len(model_bytes),
        mape=mape,
        latest_updated=update_latest,
    )
    return version


def write_extra_to_meta(
    region: str,
    model_name: str,
    version: str,
    key_updates: dict[str, Any],
) -> bool:
    """Merge ``key_updates`` into a saved model's ``extra`` dict in GCS.

    Round-trips the existing meta JSON, deep-merges the new keys on top of
    the existing ``extra`` (other top-level fields are preserved verbatim),
    and writes the result back to the same blob. Returns ``True`` on
    success, ``False`` if GCS is disabled / the blob is missing / the
    upload fails.

    Used by the training job to attach ensemble-level holdout metrics +
    weights to the xgboost meta after both per-model persistence and
    ensemble computation are done. Avoids a separate "ensemble meta"
    blob (which would need its own latest.json plumbing) at the cost
    of a single follow-up GCS write per region per training run.
    """
    client = _get_client()
    if client is None:
        log.info(
            "model_meta_extra_skipped_gcs_disabled",
            region=region,
            model=model_name,
        )
        return False
    try:
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(_blob_path(region, model_name, version, ".meta.json"))
        if not blob.exists():
            log.warning(
                "model_meta_extra_missing",
                region=region,
                model=model_name,
                version=version,
            )
            return False
        meta_dict = json.loads(blob.download_as_text())
        extra = dict(meta_dict.get("extra") or {})
        extra.update(key_updates)
        meta_dict["extra"] = extra
        blob.upload_from_string(
            json.dumps(meta_dict, indent=2),
            content_type="application/json",
        )
        log.info(
            "model_meta_extra_updated",
            region=region,
            model=model_name,
            version=version,
            keys=sorted(key_updates.keys()),
        )
        return True
    except Exception as e:
        log.warning(
            "model_meta_extra_failed",
            region=region,
            model=model_name,
            version=version,
            error=str(e),
        )
        return False


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
    global _latest_cache, _latest_cache_ts, _latest_cache_is_failure
    with _latest_cache_lock:
        _latest_cache = None
        _latest_cache_ts = 0.0
        _latest_cache_is_failure = False


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
# Uses ``print`` (not structlog) intentionally: this is a human-facing CLI
# inspector that emits raw JSON to stdout for piping to ``jq`` etc. Structured
# log output would corrupt the JSON stream consumers expect here.

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
