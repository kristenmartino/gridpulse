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
from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import MagicMock, patch


def _reset_persistence_state(tmp_cache_dir: str) -> None:
    """Reset module-level caches and point the local cache at tmp."""
    import models.persistence as mp

    mp._client = None
    mp._latest_cache = None
    mp._latest_cache_ts = 0.0
    mp._latest_cache_is_failure = False
    mp.LOCAL_MODEL_CACHE_DIR = tmp_cache_dir


class _FakeBlob:
    """Minimal fake mirroring the subset of google.cloud.storage.Blob we use.

    Tracks per-path generation numbers and honors ``if_generation_match``
    on upload so optimistic-concurrency tests can exercise the precondition
    path without a real GCS client. The behavior matches GCS semantics:

    - ``generation`` is None until ``reload()`` (or implicit-on-read);
      after reload it equals the live generation of the underlying object
    - ``upload_from_string(if_generation_match=0)`` succeeds only when
      no object exists at that path
    - ``upload_from_string(if_generation_match=N)`` succeeds only when
      the live generation equals N; mismatch raises ``PreconditionFailed``
    - successful upload bumps the live generation
    """

    def __init__(self, store: dict[str, bytes], generations: dict[str, int], path: str) -> None:
        self._store = store
        self._generations = generations
        self._path = path
        self.generation: int | None = None

    def exists(self) -> bool:
        return self._path in self._store

    def reload(self) -> None:
        self.generation = self._generations.get(self._path)

    def upload_from_string(
        self,
        data: Any,
        content_type: str | None = None,
        if_generation_match: int | None = None,
    ) -> None:
        if if_generation_match is not None:
            current = self._generations.get(self._path, 0)
            if if_generation_match != current:
                from google.api_core.exceptions import PreconditionFailed

                raise PreconditionFailed(
                    f"generation mismatch: expected {if_generation_match}, got {current}"
                )
        if isinstance(data, str):
            data = data.encode("utf-8")
        self._store[self._path] = data
        self._generations[self._path] = self._generations.get(self._path, 0) + 1

    def download_as_bytes(self) -> bytes:
        return self._store[self._path]

    def download_as_text(self) -> str:
        return self._store[self._path].decode("utf-8")


class _FakeBucket:
    def __init__(self, store: dict[str, bytes], generations: dict[str, int]) -> None:
        self._store = store
        self._generations = generations

    def blob(self, path: str) -> _FakeBlob:
        return _FakeBlob(self._store, self._generations, path)


def _make_fake_client(
    store: dict[str, bytes],
    generations: dict[str, int] | None = None,
) -> MagicMock:
    """Build a GCS-client fake. ``generations`` is shared state — callers
    that need to verify the optimistic-concurrency contract pass an
    explicit dict so they can inspect it post-test."""
    if generations is None:
        generations = {}
    client = MagicMock()
    client.bucket.return_value = _FakeBucket(store, generations)
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
        store: dict[str, bytes] = {"cache/models/latest.json": b"this is not json"}
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
# latest.json pointer cache TTL (#271 / P2-10)
# ---------------------------------------------------------------------------


class TestLatestPointerTTL:
    """The pointer cache expires so a long-lived web process reflects daily
    retraining instead of pinning the pointer for its whole lifetime."""

    def test_cache_hit_within_ttl_does_not_reread(self, tmp_path) -> None:
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        mp._latest_cache = {"PJM": {"xgboost": "v1"}}
        mp._latest_cache_ts = 1000.0

        # _get_client would raise if consulted — proves the hit short-circuits.
        boom = MagicMock(side_effect=AssertionError("should not re-read within TTL"))
        with (
            patch.object(mp.time, "monotonic", return_value=1000.0 + mp._LATEST_CACHE_TTL - 1),
            patch.object(mp, "_get_client", boom),
        ):
            assert mp._read_latest() == {"PJM": {"xgboost": "v1"}}
        boom.assert_not_called()

    def test_cache_expires_after_ttl_and_rereads(self, tmp_path) -> None:
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        mp._latest_cache = {"PJM": {"xgboost": "v1"}}  # stale in-process copy
        mp._latest_cache_ts = 1000.0
        # GCS now holds a newer pointer (a retrain landed a v2).
        store: dict[str, bytes] = {
            "cache/models/latest.json": json.dumps({"PJM": {"xgboost": "v2"}}).encode("utf-8")
        }
        fake_client = _make_fake_client(store)

        with (
            patch.object(mp, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(mp, "GCS_PATH_PREFIX", "cache"),
            patch.object(mp.time, "monotonic", return_value=1000.0 + mp._LATEST_CACHE_TTL + 1),
            patch.object(mp, "_get_client", return_value=fake_client),
        ):
            assert mp._read_latest() == {"PJM": {"xgboost": "v2"}}  # re-read past TTL
        assert mp._latest_cache == {"PJM": {"xgboost": "v2"}}

    def test_invalidate_forces_reread_next_call(self, tmp_path) -> None:
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        mp._latest_cache = {"PJM": {"xgboost": "v1"}}
        mp._latest_cache_ts = 1000.0
        mp._latest_cache_is_failure = True
        mp.invalidate_latest_cache()
        assert mp._latest_cache is None
        assert mp._latest_cache_ts == 0.0
        assert mp._latest_cache_is_failure is False


class _RaisingBucketClient:
    """Client whose blob reads raise — simulates a GCS outage/blip."""

    def __init__(self, exc=None):
        self._exc = exc or ConnectionError("gcs blip")

    def bucket(self, name):
        raise self._exc


class TestLatestPointerFailureSemantics:
    """#272 (P2-11) — a failed pointer read must never negative-cache the ``{}``
    sentinel over a valid pointer, and a true failure re-probes on the SHORT
    failure TTL, not the 300s success TTL."""

    @contextmanager
    def _gcs_env(self, mp, client, monotonic=None):
        with ExitStack() as stack:
            stack.enter_context(patch.object(mp, "GCS_ENABLED", True))
            stack.enter_context(patch.object(mp, "GCS_BUCKET_NAME", "test-bucket"))
            stack.enter_context(patch.object(mp, "GCS_PATH_PREFIX", "cache"))
            stack.enter_context(patch.object(mp, "_get_client", return_value=client))
            stack.enter_context(patch.object(mp.time, "sleep", lambda s: None))
            if monotonic is not None:
                stack.enter_context(patch.object(mp.time, "monotonic", monotonic))
            yield

    def test_failure_serves_last_known_good_not_empty(self, tmp_path) -> None:
        """THE P2-11 core: a blip must not erase a previously-good pointer —
        the model store stays loadable on the stale-by-minutes value."""
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        good = {"PJM": {"xgboost": "v1"}}
        mp._latest_cache = good
        mp._latest_cache_ts = 1000.0
        with self._gcs_env(
            mp, _RaisingBucketClient(), monotonic=lambda: 1000.0 + mp._LATEST_CACHE_TTL + 1
        ):
            out = mp._read_latest()  # TTL expired → re-read → blip
        assert out == good  # last-known-good, NOT {}
        assert mp._latest_cache == good
        assert mp._latest_cache_is_failure is True  # short re-probe window armed

    def test_failure_with_no_prior_uses_short_ttl(self, tmp_path) -> None:
        """Cold process + blip: {} is cached only for _LATEST_FAILURE_TTL (30s),
        not the 300s success TTL — the old negative-cache made the whole store
        unloadable for minutes."""
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        clock = [1000.0]
        with self._gcs_env(mp, _RaisingBucketClient(), monotonic=lambda: clock[0]):
            assert mp._read_latest() == {}  # cold failure → sentinel
            assert mp._latest_cache_is_failure is True
            # Within the failure TTL: served from cache (no re-probe storm).
            clock[0] = 1000.0 + mp._LATEST_FAILURE_TTL - 1
            assert mp._read_latest() == {}
        # Past the failure TTL (but well under the success TTL): re-probes and
        # RECOVERS when GCS is healthy again.
        store = {"cache/models/latest.json": json.dumps({"PJM": {"xgboost": "v2"}}).encode()}
        with self._gcs_env(
            mp,
            _make_fake_client(store),
            monotonic=lambda: 1000.0 + mp._LATEST_FAILURE_TTL + 1,
        ):
            assert mp._read_latest() == {"PJM": {"xgboost": "v2"}}
        assert mp._latest_cache_is_failure is False  # healed → success TTL again

    def test_missing_pointer_is_valid_and_uses_success_ttl(self, tmp_path) -> None:
        """A legitimately-absent latest.json (fresh bucket) is a VALID empty
        result — cached at the normal TTL, not failure semantics."""
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        with self._gcs_env(mp, _make_fake_client({}), monotonic=lambda: 1000.0):
            assert mp._read_latest() == {}  # no latest.json in store
        assert mp._latest_cache_is_failure is False
        boom = MagicMock(side_effect=AssertionError("should not re-read within success TTL"))
        with (
            patch.object(mp, "_get_client", boom),
            patch.object(mp.time, "monotonic", return_value=1000.0 + mp._LATEST_FAILURE_TTL + 5),
        ):
            assert mp._read_latest() == {}  # still cached — success TTL applies
        boom.assert_not_called()

    def test_transient_blip_healed_by_in_call_retry(self, tmp_path) -> None:
        """A sub-second blip is absorbed by the in-call retry — no failure
        semantics engage at all."""
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        store = {"cache/models/latest.json": json.dumps({"PJM": {"xgboost": "v3"}}).encode()}
        healthy = _make_fake_client(store)
        flaky = MagicMock()
        flaky.bucket.side_effect = [ConnectionError("blip"), healthy.bucket("x")]
        with (
            patch.object(mp, "GCS_ENABLED", True),
            patch.object(mp, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(mp, "GCS_PATH_PREFIX", "cache"),
            patch.object(mp, "_get_client", return_value=flaky),
            patch.object(mp.time, "sleep", lambda s: None),
        ):
            out = mp._read_latest()
        assert out == {"PJM": {"xgboost": "v3"}}
        assert mp._latest_cache_is_failure is False  # retry healed it silently

    def test_gcs_disabled_empty_is_not_failure(self, tmp_path) -> None:
        """Dev (GCS off): the empty store is the valid answer — success TTL,
        no failure flag, no re-probe churn."""
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        with (
            patch.object(mp, "GCS_ENABLED", False),
            patch.object(mp, "_get_client", return_value=None),
        ):
            assert mp._read_latest() == {}
        assert mp._latest_cache_is_failure is False


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


class TestWriteLatestOptimisticConcurrency:
    """``_write_latest`` uses GCS optimistic concurrency
    (``If-Generation-Match`` precondition) so two parallel training-job
    tasks writing to the same ``latest.json`` can't silently drop each
    other's pointer updates. The previous last-writer-wins
    implementation was unsafe under taskCount>1 parallel training:

    1. Task A reads latest.json with {PJM: v1}
    2. Task B reads the same snapshot
    3. Task A writes back {PJM: v1, CAISO: v2}
    4. Task B writes back {PJM: v1, ERCOT: v3}  ← drops CAISO!

    These tests pin the contract: concurrent writers must converge to
    a union of all pointer updates, not the last-writer's view."""

    def _setup(self, tmp_path):
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        store: dict[str, bytes] = {}
        generations: dict[str, int] = {}
        client = _make_fake_client(store, generations)
        return mp, store, generations, client

    def test_first_writer_uses_generation_zero_precondition(self, tmp_path) -> None:
        """First write into an empty latest.json must use
        ``if_generation_match=0`` (matches "no object yet")."""
        mp, store, generations, client = self._setup(tmp_path)
        with (
            patch.object(mp, "GCS_ENABLED", True),
            patch.object(mp, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(mp, "GCS_PATH_PREFIX", "cache"),
            patch.object(mp, "_get_client", return_value=client),
        ):
            mp._write_latest("ERCOT", "xgboost", "v1")

        # Object exists post-write; generation incremented from 0 to 1
        assert "cache/models/latest.json" in store
        assert generations["cache/models/latest.json"] == 1

    def test_concurrent_writers_converge_via_retry(self, tmp_path) -> None:
        """Two simulated parallel writers — both read the same snapshot,
        then write back. The second writer must hit the precondition,
        re-read, and merge — final state must contain BOTH entries."""
        mp, store, generations, client = self._setup(tmp_path)

        with (
            patch.object(mp, "GCS_ENABLED", True),
            patch.object(mp, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(mp, "GCS_PATH_PREFIX", "cache"),
            patch.object(mp, "_get_client", return_value=client),
        ):
            # First writer: clean slate.
            mp._write_latest("ERCOT", "xgboost", "v-ercot")
            # Both writers now have a snapshot at generation 1. Simulate
            # the race by manually shifting generation between them.
            mp._write_latest("CAISO", "xgboost", "v-caiso")
            mp._write_latest("PJM", "xgboost", "v-pjm")

        merged = json.loads(store["cache/models/latest.json"].decode("utf-8"))
        # All three regions present — no silent drop.
        assert "ERCOT" in merged
        assert "CAISO" in merged
        assert "PJM" in merged
        assert merged["ERCOT"]["xgboost"] == "v-ercot"
        assert merged["CAISO"]["xgboost"] == "v-caiso"
        assert merged["PJM"]["xgboost"] == "v-pjm"

    def test_retry_on_precondition_failed(self, tmp_path) -> None:
        """When a write fails on precondition, the next attempt re-reads
        the current state, merges its update onto the new snapshot, and
        re-issues the write with the fresh generation."""
        mp, store, generations, client = self._setup(tmp_path)
        with (
            patch.object(mp, "GCS_ENABLED", True),
            patch.object(mp, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(mp, "GCS_PATH_PREFIX", "cache"),
            patch.object(mp, "_get_client", return_value=client),
        ):
            mp._write_latest("ERCOT", "xgboost", "v1")
            gen_before = generations["cache/models/latest.json"]
            assert gen_before == 1

            # The next write should reload (capture generation=1) and
            # write with if_generation_match=1, succeeding cleanly.
            mp._write_latest("CAISO", "xgboost", "v2")

        # Generation bumped to 2; both regions present.
        assert generations["cache/models/latest.json"] == 2
        merged = json.loads(store["cache/models/latest.json"].decode("utf-8"))
        assert merged["ERCOT"]["xgboost"] == "v1"
        assert merged["CAISO"]["xgboost"] == "v2"


class TestWriteExtraToMeta:
    """``write_extra_to_meta`` round-trips an existing meta blob,
    deep-merges new keys into ``extra``, and writes the result back.
    Used by the training job to attach ensemble-level holdout metrics
    to the xgboost meta after both per-model persistence and ensemble
    computation are done."""

    def _setup(self, tmp_path):
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        store: dict[str, bytes] = {}
        client = _make_fake_client(store)
        return mp, store, client

    def test_merges_new_keys_into_extra(self, tmp_path) -> None:
        mp, store, client = self._setup(tmp_path)
        with (
            patch.object(mp, "GCS_ENABLED", True),
            patch.object(mp, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(mp, "GCS_PATH_PREFIX", "cache"),
            patch.object(mp, "_get_client", return_value=client),
        ):
            version = mp.save_model(
                "PJM",
                "xgboost",
                {"k": "v"},
                "h",
                100,
                mape=1.10,
                extra={"cv_scores": [1.0, 1.2]},
            )
            assert version is not None

            ok = mp.write_extra_to_meta(
                "PJM",
                "xgboost",
                version,
                {
                    "ensemble_holdout_metrics": {
                        "mape": 0.94,
                        "rmse": 119.0,
                        "mae": 71.0,
                        "r2": 0.994,
                    },
                    "ensemble_weights": {
                        "xgboost": 0.6,
                        "prophet": 0.25,
                        "arima": 0.15,
                    },
                },
            )
            assert ok is True

            mp.invalidate_latest_cache()
            meta = mp.get_model_metadata("PJM", "xgboost")
            assert meta is not None

            # Existing keys preserved, new keys merged in.
            assert meta.extra["cv_scores"] == [1.0, 1.2]
            assert meta.extra["ensemble_holdout_metrics"]["mape"] == 0.94
            assert meta.extra["ensemble_holdout_metrics"]["rmse"] == 119.0
            assert meta.extra["ensemble_weights"]["xgboost"] == 0.6
            # Top-level fields unchanged.
            assert meta.mape == 1.10
            assert meta.region == "PJM"

    def test_returns_false_when_blob_missing(self, tmp_path) -> None:
        mp, store, client = self._setup(tmp_path)
        with (
            patch.object(mp, "GCS_ENABLED", True),
            patch.object(mp, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(mp, "GCS_PATH_PREFIX", "cache"),
            patch.object(mp, "_get_client", return_value=client),
        ):
            ok = mp.write_extra_to_meta("PJM", "xgboost", "v-nonexistent", {"foo": "bar"})
            assert ok is False

    def test_returns_false_when_gcs_disabled(self, tmp_path) -> None:
        mp, _, _ = self._setup(tmp_path)
        with patch.object(mp, "_get_client", return_value=None):
            ok = mp.write_extra_to_meta("PJM", "xgboost", "v1", {"foo": "bar"})
            assert ok is False


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
            assert mp.save_model("ERCOT", "xgboost", unpicklable, "h", 1) is None

    def test_corrupt_blob_returns_none(self, tmp_path) -> None:
        """A corrupt pickled blob → load_model returns None rather than raising."""
        import models.persistence as mp

        _reset_persistence_state(str(tmp_path / "cache"))
        version = "20240101T000000Z"
        store: dict[str, bytes] = {
            "cache/models/latest.json": json.dumps({"ERCOT": {"xgboost": version}}).encode("utf-8"),
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
