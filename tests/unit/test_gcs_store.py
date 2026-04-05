"""Unit tests for data/gcs_store.py — GCS Parquet persistence layer.

Tests cover lazy client initialization, blob path construction,
fire-and-forget writes, and synchronous reads with full error handling.
All GCS interactions are mocked — no real cloud calls.
"""

import io
import threading
from unittest.mock import MagicMock, patch

import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_client_singleton() -> None:
    """Reset the module-level _client singleton between tests."""
    import data.gcs_store as gcs

    gcs._client = None


def _sample_df() -> pd.DataFrame:
    """Small DataFrame for write/read round-trip tests."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "demand_mw": [40000.0, 39500.0, 41000.0],
        }
    )


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to Parquet bytes for mocking blob downloads."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _get_client()
# ---------------------------------------------------------------------------


class TestGetClient:
    """Tests for the lazy GCS client singleton."""

    def setup_method(self) -> None:
        _reset_client_singleton()

    def test_returns_none_when_gcs_disabled(self) -> None:
        """When GCS_ENABLED is False, _get_client returns None immediately."""
        import data.gcs_store as gcs

        with patch.object(gcs, "GCS_ENABLED", False):
            assert gcs._get_client() is None

    def test_returns_none_when_bucket_name_empty(self) -> None:
        """When GCS_BUCKET_NAME is empty, _get_client returns None."""
        import data.gcs_store as gcs

        with (
            patch.object(gcs, "GCS_ENABLED", True),
            patch.object(gcs, "GCS_BUCKET_NAME", ""),
        ):
            assert gcs._get_client() is None

    @patch("data.gcs_store.storage", create=True)
    def test_creates_client_on_first_call(self, _mock_storage: MagicMock) -> None:
        """First call with valid config creates a StorageClient singleton."""
        import data.gcs_store as gcs

        mock_client_instance = MagicMock()

        fake_module = MagicMock()
        fake_client = MagicMock()
        fake_module.Client.return_value = fake_client

        with (
            patch.object(gcs, "GCS_ENABLED", True),
            patch.object(gcs, "GCS_BUCKET_NAME", "my-bucket"),
            patch(
                "data.gcs_store.StorageClient",
                return_value=mock_client_instance,
                create=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "google": MagicMock(),
                    "google.cloud": MagicMock(),
                    "google.cloud.storage": fake_module,
                },
            ),
        ):
            result = gcs._get_client()
            assert result is fake_client
            fake_module.Client.assert_called_once()

    @patch("data.gcs_store.storage", create=True)
    def test_singleton_returns_same_client(self, _mock_storage: MagicMock) -> None:
        """Subsequent calls return the cached client, not a new one."""
        import data.gcs_store as gcs

        sentinel_client = MagicMock(name="sentinel_client")
        gcs._client = sentinel_client

        with (
            patch.object(gcs, "GCS_ENABLED", True),
            patch.object(gcs, "GCS_BUCKET_NAME", "my-bucket"),
        ):
            result = gcs._get_client()
            assert result is sentinel_client

    def test_returns_none_on_import_error(self) -> None:
        """If google-cloud-storage is missing, _get_client returns None."""
        import data.gcs_store as gcs

        with (
            patch.object(gcs, "GCS_ENABLED", True),
            patch.object(gcs, "GCS_BUCKET_NAME", "my-bucket"),
        ):
            # Force the lazy import inside the function to raise.
            import builtins

            real_import = builtins.__import__

            def _fail_gcs_import(name: str, *args, **kwargs):
                if "google.cloud.storage" in name:
                    raise ImportError("No module named 'google.cloud.storage'")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=_fail_gcs_import):
                result = gcs._get_client()
                assert result is None


# ---------------------------------------------------------------------------
# _blob_path()
# ---------------------------------------------------------------------------


class TestBlobPath:
    """Tests for GCS object path construction."""

    def test_demand_path(self) -> None:
        """Demand data follows the convention {prefix}/demand/{region}/latest.parquet."""
        import data.gcs_store as gcs

        with patch.object(gcs, "GCS_PATH_PREFIX", "cache"):
            result = gcs._blob_path("demand", "ERCOT")
            assert result == "cache/demand/ERCOT/latest.parquet"

    def test_weather_path(self) -> None:
        """Weather data follows the convention {prefix}/weather/{region}/latest.parquet."""
        import data.gcs_store as gcs

        with patch.object(gcs, "GCS_PATH_PREFIX", "cache"):
            result = gcs._blob_path("weather", "CAISO")
            assert result == "cache/weather/CAISO/latest.parquet"

    def test_custom_prefix(self) -> None:
        """A non-default GCS_PATH_PREFIX is reflected in the path."""
        import data.gcs_store as gcs

        with patch.object(gcs, "GCS_PATH_PREFIX", "v2/data"):
            result = gcs._blob_path("demand", "PJM")
            assert result == "v2/data/demand/PJM/latest.parquet"


# ---------------------------------------------------------------------------
# write_parquet()
# ---------------------------------------------------------------------------


class TestWriteParquet:
    """Tests for fire-and-forget Parquet writes to GCS."""

    def setup_method(self) -> None:
        _reset_client_singleton()

    def test_noop_when_gcs_disabled(self) -> None:
        """write_parquet returns immediately when GCS is disabled."""
        import data.gcs_store as gcs

        with patch.object(gcs, "GCS_ENABLED", False):
            # Should not raise and should not start any thread.
            gcs.write_parquet(_sample_df(), "demand", "ERCOT")

    def test_noop_when_df_empty(self) -> None:
        """write_parquet returns immediately for an empty DataFrame."""
        import data.gcs_store as gcs

        with patch.object(gcs, "GCS_ENABLED", True):
            gcs.write_parquet(pd.DataFrame(), "demand", "ERCOT")

    def test_success_uploads_parquet(self) -> None:
        """On the happy path, write_parquet serializes and uploads via GCS."""
        import data.gcs_store as gcs

        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        gcs._client = mock_client

        with (
            patch.object(gcs, "GCS_ENABLED", True),
            patch.object(gcs, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(gcs, "GCS_PATH_PREFIX", "cache"),
        ):
            df = _sample_df()
            gcs.write_parquet(df, "demand", "ERCOT")

            # The upload happens in a daemon thread; give it time to complete.
            import time

            time.sleep(0.5)

            mock_client.bucket.assert_called_with("test-bucket")
            mock_bucket.blob.assert_called_with("cache/demand/ERCOT/latest.parquet")
            mock_blob.upload_from_string.assert_called_once()
            uploaded_bytes = mock_blob.upload_from_string.call_args[0][0]
            assert isinstance(uploaded_bytes, bytes)
            assert len(uploaded_bytes) > 0

    def test_gcs_error_is_swallowed(self) -> None:
        """Upload failures are caught silently (fire-and-forget contract)."""
        import data.gcs_store as gcs

        mock_blob = MagicMock()
        mock_blob.upload_from_string.side_effect = Exception("GCS unavailable")
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        gcs._client = mock_client

        with (
            patch.object(gcs, "GCS_ENABLED", True),
            patch.object(gcs, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(gcs, "GCS_PATH_PREFIX", "cache"),
        ):
            # Must not raise.
            gcs.write_parquet(_sample_df(), "demand", "ERCOT")

            import time

            time.sleep(0.5)

            # The upload was attempted but failed silently.
            mock_blob.upload_from_string.assert_called_once()

    def test_write_spawns_daemon_thread(self) -> None:
        """write_parquet uses a daemon background thread for the upload."""
        import data.gcs_store as gcs

        gcs._client = MagicMock()
        started_threads: list[threading.Thread] = []
        original_thread_init = threading.Thread.__init__

        def _capture_thread(self_thread, *args, **kwargs):
            original_thread_init(self_thread, *args, **kwargs)
            started_threads.append(self_thread)

        with (
            patch.object(gcs, "GCS_ENABLED", True),
            patch.object(gcs, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(threading.Thread, "__init__", _capture_thread),
            patch.object(threading.Thread, "start", MagicMock()),
        ):
            gcs.write_parquet(_sample_df(), "demand", "ERCOT")
            assert len(started_threads) == 1
            assert started_threads[0].daemon is True

    def test_serialization_error_does_not_raise(self) -> None:
        """If Parquet serialization fails, write_parquet returns silently."""
        import data.gcs_store as gcs

        with (
            patch.object(gcs, "GCS_ENABLED", True),
            patch("pandas.DataFrame.to_parquet", side_effect=Exception("serialize boom")),
        ):
            # Must not raise.
            gcs.write_parquet(_sample_df(), "demand", "ERCOT")


# ---------------------------------------------------------------------------
# read_parquet()
# ---------------------------------------------------------------------------


class TestReadParquet:
    """Tests for synchronous Parquet reads from GCS."""

    def setup_method(self) -> None:
        _reset_client_singleton()

    def test_returns_none_when_gcs_disabled(self) -> None:
        """read_parquet returns None immediately when GCS is disabled."""
        import data.gcs_store as gcs

        with patch.object(gcs, "GCS_ENABLED", False):
            result = gcs.read_parquet("demand", "ERCOT")
            assert result is None

    def test_returns_none_when_client_unavailable(self) -> None:
        """read_parquet returns None when _get_client() yields None."""
        import data.gcs_store as gcs

        with (
            patch.object(gcs, "GCS_ENABLED", True),
            patch.object(gcs, "GCS_BUCKET_NAME", ""),
        ):
            result = gcs.read_parquet("demand", "ERCOT")
            assert result is None

    def test_returns_none_when_blob_not_found(self) -> None:
        """read_parquet returns None when the GCS blob does not exist."""
        import data.gcs_store as gcs

        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        gcs._client = mock_client

        with (
            patch.object(gcs, "GCS_ENABLED", True),
            patch.object(gcs, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(gcs, "GCS_PATH_PREFIX", "cache"),
        ):
            result = gcs.read_parquet("demand", "ERCOT")
            assert result is None
            mock_blob.exists.assert_called_once()

    def test_success_returns_dataframe(self) -> None:
        """On the happy path, read_parquet returns a DataFrame."""
        import data.gcs_store as gcs

        df_expected = _sample_df()
        parquet_data = _parquet_bytes(df_expected)

        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_bytes.return_value = parquet_data
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        gcs._client = mock_client

        with (
            patch.object(gcs, "GCS_ENABLED", True),
            patch.object(gcs, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(gcs, "GCS_PATH_PREFIX", "cache"),
        ):
            result = gcs.read_parquet("demand", "ERCOT")
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert list(result.columns) == ["timestamp", "demand_mw"]
            mock_client.bucket.assert_called_with("test-bucket")
            mock_bucket.blob.assert_called_with("cache/demand/ERCOT/latest.parquet")

    def test_returns_none_on_download_error(self) -> None:
        """read_parquet returns None if the GCS download raises."""
        import data.gcs_store as gcs

        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_bytes.side_effect = Exception("network error")
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        gcs._client = mock_client

        with (
            patch.object(gcs, "GCS_ENABLED", True),
            patch.object(gcs, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(gcs, "GCS_PATH_PREFIX", "cache"),
        ):
            result = gcs.read_parquet("demand", "ERCOT")
            assert result is None

    def test_returns_none_on_corrupt_parquet(self) -> None:
        """read_parquet returns None when the downloaded bytes are not valid Parquet."""
        import data.gcs_store as gcs

        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_bytes.return_value = b"this is not parquet"
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        gcs._client = mock_client

        with (
            patch.object(gcs, "GCS_ENABLED", True),
            patch.object(gcs, "GCS_BUCKET_NAME", "test-bucket"),
            patch.object(gcs, "GCS_PATH_PREFIX", "cache"),
        ):
            result = gcs.read_parquet("demand", "ERCOT")
            assert result is None
