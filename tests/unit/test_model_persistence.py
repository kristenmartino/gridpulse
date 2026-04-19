"""Unit tests for models/persistence.py — GCS-backed model persistence.

The GCS client is fully mocked; blob reads/writes happen against an
in-memory dict keyed by blob path. These tests cover:

- Pickle round-trip for the three model types (xgboost/prophet/arima shapes).
- ``save_model`` updates ``latest.json`` with merge semantics.
- ``get_model_metadata`` short-circuits when GCS is disabled or no pointer exists.
- ``load_model`` uses the local disk cache on the second call.
- ``_warn_if_lib_skew`` triggers a structured warning when majors differ.
"""

from __future__ import annotations

import json
import pickle
from typing import Any
from unittest.mock import MagicMock, patch


def _reset_persistence_state(tmp_cache_dir: str) -> None:
    """Reset module-level caches and point the local cache at tmp."""
    import models.persistence as mp

    mp._client = None
    mp._latest_cache = None
    mp.LOCAL_MODEL_CACHE_DIR = tmp_cache_dir


class _FakeBlob:
    """Minimal fake mirroring the subset of google.cloud.storage.Blob we use."""

    def __init__(self, store: dict[str, bytes], path: str) -> None:
        self._store = store
        self._path = path

    def exists(self) -> bool:
        return self._path in self._store

    def upload_from_string(self, data: Any, content_type: str | None = None) -> None:
        if isinstance(data, str):
            data = data.encode("utf-8")
        self._store[self._path] = data

    def download_as_bytes(self) -> bytes:
        return self._store[self._path]

    def download_as_text(self) -> str:
        return self._store[self._path].decode("utf-8")


class _FakeBucket:
    def __init__(self, store: dict[str, bytes]) -> None:
        self._store = store

    def blob(self, path: str) -> _FakeBlob:
        return _FakeBlob(self._store, path)


def _make_fake_client(store: dict[str, bytes]) -> MagicMock:
    client = MagicMock()
    client.bucket.return_value = _FakeBucket(store)
    return client


# ---------------------------------------------------------------------------
# save_model / load_model round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    def test_pickle_roundtrip_xgboost_shape(self, tmp_path) -> None:
        """A dict-shaped XGBoost payload survives a save + load round-trip."""
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        store: dict[str, bytes] = {}
        fake_client = _make_fake_client(store)

        model_dict = {
            "model": {"booster_bytes": b"fake-booster"},
            "feature_importances": {"temperature_2m": 0.42, "hour_sin": 0.31},
            "cv_scores": [6.1, 5.9, 6.3],
        }

        with (
            patch.object(mp, "GCS_ENABLED", True),
            patch.object(mp, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(mp, "GCS_PATH_PREFIX", "cache"),
            patch.object(mp, "_get_client", return_value=fake_client),
        ):
            version = mp.save_model(
                region="ERCOT",
                model_name="xgboost",
                model_obj=model_dict,
                data_hash="abc123",
                train_rows=10_000,
                mape=5.8,
            )
            assert version is not None
            assert f"cache/models/ERCOT/xgboost/{version}.pkl" in store
            assert f"cache/models/ERCOT/xgboost/{version}.meta.json" in store
            assert "cache/models/latest.json" in store

            # Invalidate the latest-cache so the next call re-reads from GCS.
            mp.invalidate_latest_cache()

            loaded = mp.load_model("ERCOT", "xgboost")
            assert loaded is not None
            loaded_obj, meta = loaded
            assert isinstance(loaded_obj, dict)
            # MagicMocks are unpickleable via cloudpickle but pickle.HIGHEST_PROTOCOL
            # falls back fine for our structural dict (keys survive).
            assert "feature_importances" in loaded_obj
            assert meta.region == "ERCOT"
            assert meta.model_name == "xgboost"
            assert meta.mape == 5.8

    def test_pickle_roundtrip_simple_object(self, tmp_path) -> None:
        """A plain picklable object (stand-in for Prophet/SARIMAX) round-trips."""
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        store: dict[str, bytes] = {}
        fake_client = _make_fake_client(store)

        model_obj = {"type": "prophet-like", "params": [1, 2, 3], "nested": {"a": "b"}}

        with (
            patch.object(mp, "GCS_ENABLED", True),
            patch.object(mp, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(mp, "GCS_PATH_PREFIX", "cache"),
            patch.object(mp, "_get_client", return_value=fake_client),
        ):
            version = mp.save_model(
                region="CAISO",
                model_name="prophet",
                model_obj=model_obj,
                data_hash="deadbeef",
                train_rows=2000,
            )
            assert version is not None
            mp.invalidate_latest_cache()
            loaded = mp.load_model("CAISO", "prophet")
            assert loaded is not None
            obj, meta = loaded
            assert obj == model_obj
            assert meta.data_hash == "deadbeef"
            assert meta.train_rows == 2000


# ---------------------------------------------------------------------------
# Metadata shape
# ---------------------------------------------------------------------------


class TestMetadataShape:
    def test_meta_json_schema(self, tmp_path) -> None:
        """Metadata persisted to GCS matches the public ModelMetadata schema."""
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        store: dict[str, bytes] = {}
        fake_client = _make_fake_client(store)

        with (
            patch.object(mp, "GCS_ENABLED", True),
            patch.object(mp, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(mp, "GCS_PATH_PREFIX", "cache"),
            patch.object(mp, "_get_client", return_value=fake_client),
        ):
            version = mp.save_model(
                region="PJM",
                model_name="xgboost",
                model_obj={"k": "v"},
                data_hash="h",
                train_rows=500,
                mape=7.2,
                extra={"note": "unit-test"},
            )

        meta_path = f"cache/models/PJM/xgboost/{version}.meta.json"
        raw = json.loads(store[meta_path].decode("utf-8"))
        for key in (
            "region",
            "model_name",
            "version",
            "data_hash",
            "trained_at",
            "train_rows",
            "mape",
            "lib_versions",
            "extra",
        ):
            assert key in raw
        assert raw["region"] == "PJM"
        assert raw["mape"] == 7.2
        assert raw["extra"] == {"note": "unit-test"}
        assert "python" in raw["lib_versions"]


# ---------------------------------------------------------------------------
# latest.json merge behavior
# ---------------------------------------------------------------------------


class TestLatestPointer:
    def test_merge_preserves_other_regions(self, tmp_path) -> None:
        """Writing a new region+model entry does not evict sibling entries."""
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        store: dict[str, bytes] = {
            "cache/models/latest.json": json.dumps(
                {"ERCOT": {"xgboost": "20240101T000000Z"}}
            ).encode("utf-8")
        }
        fake_client = _make_fake_client(store)

        with (
            patch.object(mp, "GCS_ENABLED", True),
            patch.object(mp, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(mp, "GCS_PATH_PREFIX", "cache"),
            patch.object(mp, "_get_client", return_value=fake_client),
        ):
            mp.save_model(
                region="CAISO",
                model_name="xgboost",
                model_obj={"k": "v"},
                data_hash="h",
                train_rows=1,
            )

        merged = json.loads(store["cache/models/latest.json"].decode("utf-8"))
        assert "ERCOT" in merged
        assert merged["ERCOT"]["xgboost"] == "20240101T000000Z"
        assert "CAISO" in merged
        assert "xgboost" in merged["CAISO"]

    def test_latest_merge_tolerates_corrupt_pointer(self, tmp_path) -> None:
        """A corrupt ``latest.json`` does not cause save_model to raise."""
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        store: dict[str, bytes] = {
            "cache/models/latest.json": b"this is not json"
        }
        fake_client = _make_fake_client(store)

        with (
            patch.object(mp, "GCS_ENABLED", True),
            patch.object(mp, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(mp, "GCS_PATH_PREFIX", "cache"),
            patch.object(mp, "_get_client", return_value=fake_client),
        ):
            version = mp.save_model(
                region="MISO",
                model_name="xgboost",
                model_obj={"k": "v"},
                data_hash="h",
                train_rows=1,
            )
            assert version is not None

        rewritten = json.loads(store["cache/models/latest.json"].decode("utf-8"))
        assert rewritten["MISO"]["xgboost"] == version


# ---------------------------------------------------------------------------
# Disabled GCS short-circuits
# ---------------------------------------------------------------------------


class TestGCSDisabledShortCircuits:
    def test_save_returns_none_without_gcs(self, tmp_path) -> None:
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        with patch.object(mp, "_get_client", return_value=None):
            assert mp.save_model("ERCOT", "xgboost", {}, "h", 1) is None

    def test_load_returns_none_without_gcs(self, tmp_path) -> None:
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        with patch.object(mp, "_get_client", return_value=None):
            assert mp.load_model("ERCOT", "xgboost") is None

    def test_metadata_returns_none_without_gcs(self, tmp_path) -> None:
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        with patch.object(mp, "_get_client", return_value=None):
            assert mp.get_model_metadata("ERCOT", "xgboost") is None

    def test_load_returns_none_when_pointer_missing(self, tmp_path) -> None:
        """If latest.json has no entry, load_model returns None without raising."""
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        store: dict[str, bytes] = {}
        fake_client = _make_fake_client(store)

        with (
            patch.object(mp, "GCS_ENABLED", True),
            patch.object(mp, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(mp, "GCS_PATH_PREFIX", "cache"),
            patch.object(mp, "_get_client", return_value=fake_client),
        ):
            assert mp.load_model("ERCOT", "nonexistent") is None


# ---------------------------------------------------------------------------
# Local disk cache behavior
# ---------------------------------------------------------------------------


class TestLocalCache:
    def test_second_load_hits_local_cache(self, tmp_path) -> None:
        """A subsequent load_model should not re-download when the local cache is warm."""
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        store: dict[str, bytes] = {}
        fake_client = _make_fake_client(store)

        with (
            patch.object(mp, "GCS_ENABLED", True),
            patch.object(mp, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(mp, "GCS_PATH_PREFIX", "cache"),
            patch.object(mp, "_get_client", return_value=fake_client),
        ):
            version = mp.save_model(
                region="NYISO",
                model_name="xgboost",
                model_obj={"k": "v"},
                data_hash="h",
                train_rows=1,
            )
            assert version is not None
            mp.invalidate_latest_cache()

            # First load populates the local cache
            first = mp.load_model("NYISO", "xgboost")
            assert first is not None

            # Remove the blob from the fake store; second load must still succeed
            # because the local disk cache is warm for this version.
            pkl_path = f"cache/models/NYISO/xgboost/{version}.pkl"
            store.pop(pkl_path)

            mp.invalidate_latest_cache()
            second = mp.load_model("NYISO", "xgboost")
            assert second is not None
            obj, meta = second
            assert obj == {"k": "v"}
            assert meta.version == version


# ---------------------------------------------------------------------------
# Library version skew warning
# ---------------------------------------------------------------------------


class TestLibVersionSkew:
    def test_warn_on_major_version_skew(self, tmp_path) -> None:
        """When the loaded model's major lib version differs, log a warning."""
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))

        metadata = mp.ModelMetadata(
            region="ERCOT",
            model_name="xgboost",
            version="v1",
            data_hash="",
            trained_at="",
            train_rows=0,
            mape=None,
            lib_versions={"xgboost": "1.7.0", "pandas": "2.0.0"},
            extra={},
        )

        with (
            patch.object(
                mp,
                "_collect_lib_versions",
                return_value={"xgboost": "2.0.0", "pandas": "2.1.0"},
            ),
            patch.object(mp.log, "warning") as mock_warn,
        ):
            mp._warn_if_lib_skew(metadata)
            assert mock_warn.called
            (_, kwargs) = mock_warn.call_args
            assert "skews" in kwargs
            assert "xgboost" in kwargs["skews"]
            assert "pandas" not in kwargs["skews"]

    def test_no_warn_on_aligned_majors(self, tmp_path) -> None:
        """Same major version → no warning."""
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))

        metadata = mp.ModelMetadata(
            region="ERCOT",
            model_name="xgboost",
            version="v1",
            data_hash="",
            trained_at="",
            train_rows=0,
            mape=None,
            lib_versions={"xgboost": "2.0.0"},
            extra={},
        )

        with (
            patch.object(mp, "_collect_lib_versions", return_value={"xgboost": "2.1.3"}),
            patch.object(mp.log, "warning") as mock_warn,
        ):
            mp._warn_if_lib_skew(metadata)
            assert not mock_warn.called


# ---------------------------------------------------------------------------
# get_model_metadata
# ---------------------------------------------------------------------------


class TestGetModelMetadata:
    def test_returns_metadata_when_present(self, tmp_path) -> None:
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        store: dict[str, bytes] = {}
        fake_client = _make_fake_client(store)

        with (
            patch.object(mp, "GCS_ENABLED", True),
            patch.object(mp, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(mp, "GCS_PATH_PREFIX", "cache"),
            patch.object(mp, "_get_client", return_value=fake_client),
        ):
            mp.save_model("SPP", "arima", {"k": "v"}, "h", 1, mape=9.9)
            mp.invalidate_latest_cache()

            meta = mp.get_model_metadata("SPP", "arima")
            assert meta is not None
            assert meta.region == "SPP"
            assert meta.model_name == "arima"
            assert meta.mape == 9.9

    def test_returns_none_when_no_pointer(self, tmp_path) -> None:
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        store: dict[str, bytes] = {}
        fake_client = _make_fake_client(store)

        with (
            patch.object(mp, "GCS_ENABLED", True),
            patch.object(mp, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(mp, "GCS_PATH_PREFIX", "cache"),
            patch.object(mp, "_get_client", return_value=fake_client),
        ):
            assert mp.get_model_metadata("SPP", "arima") is None


# ---------------------------------------------------------------------------
# Pickle failure does not crash
# ---------------------------------------------------------------------------


class TestPickleFailure:
    def test_unpicklable_returns_none(self, tmp_path) -> None:
        """If an object fails to pickle, save_model logs + returns None."""
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        fake_client = _make_fake_client({})

        # A lambda is not picklable with pickle.HIGHEST_PROTOCOL.
        unpicklable = lambda x: x  # noqa: E731

        with (
            patch.object(mp, "GCS_ENABLED", True),
            patch.object(mp, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(mp, "GCS_PATH_PREFIX", "cache"),
            patch.object(mp, "_get_client", return_value=fake_client),
        ):
            assert (
                mp.save_model("ERCOT", "xgboost", unpicklable, "h", 1) is None
            )

    def test_corrupt_blob_returns_none(self, tmp_path) -> None:
        """A corrupt pickled blob → load_model returns None rather than raising."""
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        version = "20240101T000000Z"
        store: dict[str, bytes] = {
            "cache/models/latest.json": json.dumps(
                {"ERCOT": {"xgboost": version}}
            ).encode("utf-8"),
            f"cache/models/ERCOT/xgboost/{version}.pkl": b"not a pickle",
            f"cache/models/ERCOT/xgboost/{version}.meta.json": json.dumps(
                {
                    "region": "ERCOT",
                    "model_name": "xgboost",
                    "version": version,
                    "data_hash": "",
                    "trained_at": "",
                    "train_rows": 0,
                    "mape": None,
                    "lib_versions": {},
                    "extra": {},
                }
            ).encode("utf-8"),
        }
        fake_client = _make_fake_client(store)

        with (
            patch.object(mp, "GCS_ENABLED", True),
            patch.object(mp, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(mp, "GCS_PATH_PREFIX", "cache"),
            patch.object(mp, "_get_client", return_value=fake_client),
        ):
            assert mp.load_model("ERCOT", "xgboost") is None


# ---------------------------------------------------------------------------
# Pickle protocol sanity check (not dependent on GCS)
# ---------------------------------------------------------------------------


def test_pickle_highest_protocol_roundtrip_shapes() -> None:
    """Sanity: the shapes used by xgboost/prophet/arima save pathways pickle cleanly."""
    xgb_like = {
        "model": {"booster": "mock"},
        "feature_importances": {"a": 0.1, "b": 0.9},
        "cv_scores": [5.0, 5.5, 6.0],
    }
    prophet_like = {"type": "prophet", "regressors": ["temperature_2m"]}
    arima_like = {"type": "sarimax", "order": (1, 0, 1)}

    for obj in (xgb_like, prophet_like, arima_like):
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        assert pickle.loads(data) == obj
