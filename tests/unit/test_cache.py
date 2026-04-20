"""Unit tests for data/cache.py."""

import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from data.cache import Cache


class TestCacheBasicOperations:
    """Test get/set/delete/clear."""

    def test_set_and_get_string(self, tmp_cache):
        tmp_cache.set("key1", "hello")
        assert tmp_cache.get("key1") == "hello"

    def test_set_and_get_dict(self, tmp_cache):
        data = {"a": 1, "b": [2, 3]}
        tmp_cache.set("key2", data)
        assert tmp_cache.get("key2") == data

    def test_set_and_get_dataframe(self, tmp_cache):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
        tmp_cache.set("df1", df)
        result = tmp_cache.get("df1")
        # JSON serialization doesn't preserve exact dtypes (4.0 becomes 4 in JSON)
        pd.testing.assert_frame_equal(result, df, check_dtype=False)

    def test_get_missing_key_returns_none(self, tmp_cache):
        assert tmp_cache.get("nonexistent") is None

    def test_delete_key(self, tmp_cache):
        tmp_cache.set("del_me", "value")
        tmp_cache.delete("del_me")
        assert tmp_cache.get("del_me") is None

    def test_clear_all(self, tmp_cache):
        tmp_cache.set("a", "1")
        tmp_cache.set("b", "2")
        tmp_cache.clear()
        assert tmp_cache.get("a") is None
        assert tmp_cache.get("b") is None

    def test_overwrite_key(self, tmp_cache):
        tmp_cache.set("key", "old")
        tmp_cache.set("key", "new")
        assert tmp_cache.get("key") == "new"


class TestCacheTTL:
    """Test TTL expiration and stale fallback."""

    def test_expired_returns_none(self, tmp_cache):
        tmp_cache.set("exp", "value", ttl=0)
        time.sleep(0.05)
        assert tmp_cache.get("exp") is None

    def test_stale_fallback(self, tmp_cache):
        tmp_cache.set("stale", "old_value", ttl=0)
        time.sleep(0.05)
        result = tmp_cache.get("stale", allow_stale=True)
        assert result == "old_value"

    def test_is_stale(self, tmp_cache):
        tmp_cache.set("fresh", "value", ttl=3600)
        assert tmp_cache.is_stale("fresh") is False

    def test_is_stale_expired(self, tmp_cache):
        tmp_cache.set("old", "value", ttl=0)
        time.sleep(0.05)
        assert tmp_cache.is_stale("old") is True

    def test_is_stale_missing(self, tmp_cache):
        assert tmp_cache.is_stale("missing") is None

    def test_get_age_seconds(self, tmp_cache):
        tmp_cache.set("aged", "value")
        time.sleep(0.05)
        age = tmp_cache.get_age_seconds("aged")
        assert age is not None
        assert age >= 0.0

    def test_get_age_missing(self, tmp_cache):
        assert tmp_cache.get_age_seconds("missing") is None


class TestCacheInit:
    """Test cache initialization."""

    def test_creates_db_file(self, tmp_path):
        db_path = str(tmp_path / "new_cache.db")
        cache = Cache(db_path=db_path)
        cache.set("test", "value")
        assert cache.get("test") == "value"


class TestCacheConcurrency:
    """Regression tests for the SQLite writer-contention fix.

    The scoring job fans out region fetches via ThreadPoolExecutor; every
    fetch calls ``cache.set()`` on the same SQLite file. Before the
    ``threading.Lock`` + ``busy_timeout`` fix, this produced
    ``sqlite3.OperationalError: database is locked`` under load.
    """

    def test_concurrent_writes_do_not_raise(self, tmp_path):
        cache = Cache(db_path=str(tmp_path / "concurrent.db"), default_ttl=3600)

        # DataFrame payloads sized to approximate a real demand fetch
        # (hourly data for ~90 days).
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=2160, freq="h"),
                "demand_mw": range(2160),
            }
        )

        def write(i: int) -> bool:
            cache.set(f"region_{i}", df)
            return cache.get(f"region_{i}") is not None

        # 16 threads × 4 writes each matches the worst-case scoring-job fan-out
        # (8 regions × demand + weather + generation + diagnostics writes).
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(write, range(64)))

        assert all(results)
        # Every key should round-trip.
        for i in range(64):
            assert cache.get(f"region_{i}") is not None

    def test_concurrent_reads_during_writes(self, tmp_path):
        """Readers should not be blocked or fail while writers contend."""
        cache = Cache(db_path=str(tmp_path / "mixed.db"), default_ttl=3600)
        cache.set("seed", "initial")

        def writer(i: int) -> None:
            cache.set(f"w_{i}", {"i": i, "data": list(range(100))})

        def reader(_: int) -> str | None:
            return cache.get("seed")

        with ThreadPoolExecutor(max_workers=8) as pool:
            write_fut = [pool.submit(writer, i) for i in range(32)]
            read_fut = [pool.submit(reader, i) for i in range(32)]
            for f in write_fut:
                f.result()
            reads = [f.result() for f in read_fut]

        assert all(r == "initial" for r in reads)

    def test_get_cache_singleton_is_thread_safe(self, monkeypatch, tmp_path):
        """Concurrent first-callers of get_cache() must converge on one instance.

        Regression for the scoring-job ``database is locked`` failure: when
        multiple threads hit ``get_cache()`` simultaneously on a fresh
        container, each was constructing its own ``Cache`` and running
        ``_init_db`` (WAL pragma) concurrently, racing on the exclusive
        journal-mode transition lock.
        """
        import data.cache as cache_mod

        db_path = str(tmp_path / "singleton.db")
        monkeypatch.setattr(cache_mod, "CACHE_DB_PATH", db_path)
        monkeypatch.setattr(cache_mod, "_cache", None)

        instances: set[int] = set()
        lock = __import__("threading").Lock()

        def call_get_cache(_: int) -> int:
            c = cache_mod.get_cache()
            with lock:
                instances.add(id(c))
            return id(c)

        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(call_get_cache, range(32)))

        assert len(instances) == 1, f"expected 1 singleton, got {len(instances)}"
