"""Unit tests for data/cache.py."""

import time

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
