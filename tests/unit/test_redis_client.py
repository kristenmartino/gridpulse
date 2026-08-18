"""Unit tests for data/redis_client.py — dual-mode Redis client."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest


class TestRedisClientGracefulFallback:
    """Verify that redis_client returns None when Redis is unavailable."""

    def test_no_redis_host_returns_none(self):
        """Without REDIS_HOST, redis_get always returns None."""
        import data.redis_client as rc

        rc._redis_client = None
        rc._redis_last_attempt = 0.0

        with patch.dict("os.environ", {"REDIS_HOST": ""}, clear=False):
            result = rc.redis_get("gridpulse:actuals:FPL")
            assert result is None

    def test_redis_unavailable_returns_none(self):
        """When Redis host is set but unreachable, redis_get returns None."""
        import data.redis_client as rc

        rc._redis_client = None
        rc._redis_last_attempt = 0.0

        # Patch ``redis.Redis`` on the real module, NOT
        # ``data.redis_client.redis``. ``_get_client`` does a function-local
        # ``import redis``, which binds from ``sys.modules`` and ignores any
        # attribute set on ``data.redis_client`` — so the old patch target
        # never applied and this test made a real DNS lookup for the host
        # literally named "nonexistent" (~4.5s, and asserting nothing).
        with (
            patch.dict(
                "os.environ", {"REDIS_HOST": "nonexistent", "REDIS_PORT": "6379"}, clear=False
            ),
            patch("redis.Redis") as mock_redis_cls,
        ):
            mock_client = MagicMock()
            mock_client.ping.side_effect = ConnectionError("Connection refused")
            mock_redis_cls.return_value = mock_client
            rc._redis_last_attempt = 0.0
            result = rc.redis_get("gridpulse:actuals:FPL")
            assert result is None
            # The patch must actually be the thing that ran; without this the
            # test passes just as well against a real failed connection.
            mock_redis_cls.assert_called_once()

    def test_redis_available_returns_true(self):
        """redis_available() returns True when connected."""
        import data.redis_client as rc

        rc._redis_client = MagicMock()
        rc._redis_last_attempt = 0.0
        assert rc.redis_available() is True

    def test_redis_available_returns_false_no_host(self):
        """redis_available() returns False when no REDIS_HOST."""
        import data.redis_client as rc

        rc._redis_client = None
        rc._redis_last_attempt = 0.0
        assert rc.redis_available() is False


class TestRedisClientReads:
    """Verify redis_get correctly parses JSON from Redis."""

    def test_redis_get_parses_json(self):
        """redis_get returns parsed JSON dict."""
        import data.redis_client as rc

        mock_client = MagicMock()
        payload = {"region": "FPL", "demand_mw": [100, 200, 300]}
        mock_client.get.return_value = json.dumps(payload)
        rc._redis_client = mock_client
        rc._redis_last_attempt = 0.0

        result = rc.redis_get("gridpulse:actuals:FPL")
        assert result == payload
        mock_client.get.assert_called_once_with("gridpulse:actuals:FPL")

    def test_redis_get_returns_none_for_missing_key(self):
        """redis_get returns None when key doesn't exist."""
        import data.redis_client as rc

        mock_client = MagicMock()
        mock_client.get.return_value = None
        rc._redis_client = mock_client
        rc._redis_last_attempt = 0.0

        result = rc.redis_get("gridpulse:actuals:MISSING")
        assert result is None

    def test_redis_get_handles_corrupt_json(self):
        """redis_get returns None on invalid JSON without raising."""
        import data.redis_client as rc

        mock_client = MagicMock()
        mock_client.get.return_value = "not-valid-json{{"
        rc._redis_client = mock_client
        rc._redis_last_attempt = 0.0

        result = rc.redis_get("gridpulse:actuals:FPL")
        assert result is None

    def test_redis_get_handles_read_exception(self):
        """redis_get returns None on Redis read errors."""
        import data.redis_client as rc

        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("Connection lost")
        rc._redis_client = mock_client
        rc._redis_last_attempt = 0.0

        result = rc.redis_get("gridpulse:actuals:FPL")
        assert result is None


class TestRedisBacktestFormat:
    """Verify backtest data from Redis is in the expected format."""

    def test_backtest_has_expected_keys(self):
        """Backtest cached data has metrics, actual, predictions, timestamps."""
        import data.redis_client as rc

        backtest_payload = {
            "horizon": 24,
            "metrics": {
                "xgboost": {"mape": 3.45, "rmse": 1250.5, "mae": 950.25, "r2": 0.92},
            },
            "actual": [44500.0, 44600.0],
            "predictions": {
                "xgboost": [44520.5, 44620.3],
            },
            "timestamps": ["2024-01-15T14:00:00", "2024-01-15T15:00:00"],
        }

        mock_client = MagicMock()
        mock_client.get.return_value = json.dumps(backtest_payload)
        rc._redis_client = mock_client
        rc._redis_last_attempt = 0.0

        result = rc.redis_get("gridpulse:backtest:FPL:24")
        assert result is not None
        assert "metrics" in result
        assert "actual" in result
        assert "predictions" in result
        assert "timestamps" in result
        assert result["metrics"]["xgboost"]["mape"] == 3.45


class TestRedisKeyPrefix:
    """Verify the ``redis_key()`` helper composes the prefix correctly.

    The helper lives in ``data/redis_client.py`` and reads
    ``REDIS_KEY_PREFIX`` from ``config`` at every call. Tests that want
    to exercise an override re-import config after monkeypatching the
    env so the new value propagates.
    """

    def test_default_prefix_is_gridpulse(self):
        """The default prefix matches the product name."""
        import importlib

        import config
        import data.redis_client as rc

        # Clear any env-var override and re-import config so REDIS_KEY_PREFIX
        # picks up the actual default. Without this, a sibling test that
        # patched the env could leak into this one.
        with patch.dict("os.environ", {}, clear=False) as env:
            env.pop("REDIS_KEY_PREFIX", None)
            importlib.reload(config)
            importlib.reload(rc)
            assert rc.redis_key("actuals:FPL") == "gridpulse:actuals:FPL"
            assert rc.redis_key("forecast:ERCOT:1h") == "gridpulse:forecast:ERCOT:1h"

    def test_env_var_override_changes_prefix(self):
        """Setting ``REDIS_KEY_PREFIX`` flips the prefix on next import.

        Uses the historical ``wattcast`` value (issue #91) to verify the
        override path — pointing at any non-default namespace exercises
        the same code path. In production this is what you'd reach for
        when running an experimental scoring job that shouldn't clobber
        live keys.
        """
        import importlib

        import config
        import data.redis_client as rc

        with patch.dict("os.environ", {"REDIS_KEY_PREFIX": "wattcast"}, clear=False):
            importlib.reload(config)
            importlib.reload(rc)
            assert rc.redis_key("actuals:FPL") == "wattcast:actuals:FPL"
            assert rc.redis_key("backtest:forecast_exog:PJM:24") == (
                "wattcast:backtest:forecast_exog:PJM:24"
            )

        # Restore default so subsequent tests aren't tainted by the reload above.
        importlib.reload(config)
        importlib.reload(rc)

    def test_suffix_is_composed_verbatim(self):
        """No escaping, no validation — caller owns the suffix shape."""
        import data.redis_client as rc

        # Empty suffix (edge case)
        assert rc.redis_key("") == "gridpulse:"
        # Colons in the suffix pass through (this is how multi-part keys work)
        assert rc.redis_key("a:b:c") == "gridpulse:a:b:c"


class TestRedisReinit:
    """#268 (P2-03) — a failed connection self-heals instead of pinning Redis
    off for the whole process."""

    def test_failed_connection_retries_after_backoff(self):
        import sys

        import data.redis_client as rc

        rc._redis_client = None
        rc._redis_last_attempt = 0.0
        client = MagicMock()
        client.ping.side_effect = [ConnectionError("blip"), True]  # fail, then recover
        fake_redis = MagicMock()
        fake_redis.Redis.return_value = client
        # _get_redis does a local ``import redis``, so inject the mock into
        # sys.modules (a module-attr patch wouldn't intercept the import).
        with (
            patch.dict("os.environ", {"REDIS_HOST": "h", "REDIS_PORT": "6379"}, clear=False),
            patch.dict(sys.modules, {"redis": fake_redis}),
        ):
            assert rc._get_redis() is None  # 1st attempt fails
            assert rc._get_redis() is None  # within backoff → no re-probe
            assert client.ping.call_count == 1  # did NOT hammer the connection

            rc._redis_last_attempt = 0.0  # simulate the backoff window elapsing
            assert rc._get_redis() is client  # re-probes and recovers
            assert client.ping.call_count == 2
        rc._redis_client = None
        rc._redis_last_attempt = 0.0

    def test_healthy_client_is_cached_without_reprobe(self):
        import data.redis_client as rc

        cached = MagicMock()
        rc._redis_client = cached
        rc._redis_last_attempt = 0.0
        assert rc._get_redis() is cached
        rc._redis_client = None


class TestPersist:
    """persist() raises on a dropped write so scoring phases surface it (#268)."""

    def test_persist_raises_on_write_failure(self):
        import data.redis_client as rc

        with (
            patch("data.redis_client.redis_set", return_value=False),
            pytest.raises(rc.RedisWriteError),
        ):
            rc.persist("gridpulse:x", {"a": 1})

    def test_persist_succeeds_silently(self):
        import data.redis_client as rc

        with patch("data.redis_client.redis_set", return_value=True):
            assert rc.persist("gridpulse:x", {"a": 1}) is None


class TestRedisGetStrict:
    """#313 — absence and failure must be different answers.

    ``redis_get`` collapses "no client / command error / corrupt payload /
    key absent" into one ``None``; a stateful window phase that read that
    ``None`` as "no history" destructively re-pinned four regions' vintage
    first-sights in prod. The strict variant is the fix's foundation: it may
    return ``None`` ONLY when Redis affirmatively reports the key absent.
    """

    @pytest.fixture(autouse=True)
    def _restore_client_state(self):
        import data.redis_client as rc

        saved = (rc._redis_client, rc._redis_last_attempt)
        yield
        rc._redis_client, rc._redis_last_attempt = saved

    def test_parses_json_like_the_soft_variant(self):
        import data.redis_client as rc

        mock_client = MagicMock()
        payload = {"records": [{"ts": "2026-07-16T03:00:00+00:00", "d": 8825.0}]}
        mock_client.get.return_value = json.dumps(payload)
        rc._redis_client = mock_client

        assert rc.redis_get_strict("gridpulse:vintage:BPAT") == payload

    def test_affirmative_absence_returns_none(self):
        """The ONLY path allowed to return None."""
        import data.redis_client as rc

        mock_client = MagicMock()
        mock_client.get.return_value = None
        rc._redis_client = mock_client

        assert rc.redis_get_strict("gridpulse:vintage:NEW") is None

    def test_no_client_raises_instead_of_none(self):
        import data.redis_client as rc

        rc._redis_client = None
        rc._redis_last_attempt = time.monotonic()  # inside backoff → _get_redis None

        with pytest.raises(rc.RedisReadError):
            rc.redis_get_strict("gridpulse:vintage:BPAT")

    def test_command_error_raises(self):
        import data.redis_client as rc

        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("Connection lost")
        rc._redis_client = mock_client

        with pytest.raises(rc.RedisReadError, match="read failed"):
            rc.redis_get_strict("gridpulse:vintage:BPAT")

    def test_unparseable_payload_raises_not_none(self):
        """A value that exists but can't be parsed must never read as absent —
        the caller would overwrite whatever it was."""
        import data.redis_client as rc

        mock_client = MagicMock()
        mock_client.get.return_value = "not-valid-json{{"
        rc._redis_client = mock_client

        with pytest.raises(rc.RedisReadError, match="unparseable"):
            rc.redis_get_strict("gridpulse:vintage:BPAT")


class TestRedisConfigured:
    def test_true_when_host_set(self, monkeypatch):
        import data.redis_client as rc

        monkeypatch.setenv("REDIS_HOST", "10.0.0.5")
        assert rc.redis_configured() is True

    def test_false_when_unset(self, monkeypatch):
        import data.redis_client as rc

        monkeypatch.delenv("REDIS_HOST", raising=False)
        assert rc.redis_configured() is False


class TestInitRace:
    """#313 trigger — the cold-start thundering herd.

    51 scoring threads race ``_get_redis`` at container start. Pre-fix, the
    first thread set ``_redis_last_attempt`` and spent ~1s connecting over
    the VPC; every thread arriving in that window took the silent
    backoff-None — indistinguishable from "Redis is down", no log line —
    which is what destructively re-pinned four regions' vintage windows on
    2026-07-16 (caught live by the #314 tripwire: three failures within
    30ms at 09:00:52Z, CAISO succeeding 1.5s later). Post-fix, concurrent
    callers must WAIT for the in-flight connect and receive the client.
    """

    @pytest.fixture(autouse=True)
    def _reset_globals(self):
        import data.redis_client as rc

        saved = (rc._redis_client, rc._redis_last_attempt)
        rc._redis_client = None
        rc._redis_last_attempt = 0.0
        yield
        rc._redis_client, rc._redis_last_attempt = saved

    def test_concurrent_caller_waits_for_inflight_connect(self, monkeypatch):
        import sys
        import threading
        import types

        import data.redis_client as rc

        monkeypatch.setenv("REDIS_HOST", "10.0.0.5")

        connect_started = threading.Event()
        release_connect = threading.Event()

        class SlowRedis:
            def __init__(self, **kwargs):
                pass

            def ping(self):
                connect_started.set()
                assert release_connect.wait(5), "test deadlock"

        fake = types.ModuleType("redis")
        fake.Redis = SlowRedis
        monkeypatch.setitem(sys.modules, "redis", fake)

        results: dict = {}

        def winner():
            results["winner"] = rc._get_redis()

        def racer():
            assert connect_started.wait(5)
            # Arrives while the winner's connect is in flight — pre-fix this
            # returned the silent backoff-None.
            results["racer"] = rc._get_redis()

        t1 = threading.Thread(target=winner)
        t2 = threading.Thread(target=racer)
        t1.start()
        t2.start()
        assert connect_started.wait(5)
        release_connect.set()
        t1.join(5)
        t2.join(5)

        assert results["winner"] is not None
        assert results["racer"] is not None, (
            "concurrent caller got silent None during an in-flight connect — "
            "the #313 init race is back"
        )

    def test_backoff_after_genuine_failure_still_applies(self, monkeypatch):
        """The lock must not defeat #268's backoff: after a FAILED connect,
        immediate retries still get a fast None instead of hammering."""
        import sys
        import types

        import data.redis_client as rc

        monkeypatch.setenv("REDIS_HOST", "10.0.0.5")
        calls = {"n": 0}

        class FailingRedis:
            def __init__(self, **kwargs):
                pass

            def ping(self):
                calls["n"] += 1
                raise ConnectionError("down")

        fake = types.ModuleType("redis")
        fake.Redis = FailingRedis
        monkeypatch.setitem(sys.modules, "redis", fake)

        assert rc._get_redis() is None  # attempt 1: real connect failure
        assert rc._get_redis() is None  # inside backoff window
        assert calls["n"] == 1, "backoff did not suppress the re-probe"

    def test_successful_connect_clears_the_attempt_stamp(self, monkeypatch):
        import sys
        import types

        import data.redis_client as rc

        monkeypatch.setenv("REDIS_HOST", "10.0.0.5")

        class OkRedis:
            def __init__(self, **kwargs):
                pass

            def ping(self):
                return True

        fake = types.ModuleType("redis")
        fake.Redis = OkRedis
        monkeypatch.setitem(sys.modules, "redis", fake)

        assert rc._get_redis() is not None
        assert rc._redis_last_attempt == 0.0


class TestWriteFailureVisibility:
    """A dropped fail-soft write must be *visible*, even though it stays non-fatal.

    `redis_set` is deliberately fail-soft (#268/#313): the web tier degrades to
    the warming state rather than crashing, and the critical payloads use the
    fail-loud `persist()` twin instead. That design is not changed here.

    What was wrong is that a dropped write left no usable trace. All 15 job
    call sites ignore the returned False, and the only signal was a **stdlib
    logging** warning — which arrives in Cloud Logging as `textPayload` with no
    `jsonPayload.event`, so no log-based alert policy can match it. That is the
    same defect docs/monitoring/README.md records for the job logs, which sat
    inert until 2026-07-15.
    """

    @pytest.fixture(autouse=True)
    def _clear(self):
        import data.redis_client as rc

        rc.drain_write_failures()
        yield
        rc.drain_write_failures()

    def test_failed_write_emits_a_structlog_event_not_just_stdlib(self, monkeypatch):
        """`jsonPayload.event` is the only thing an alert policy can filter on."""
        import data.redis_client as rc

        class _Boom:
            def setex(self, *a, **kw):
                raise RuntimeError("redis down")

        monkeypatch.setattr(rc, "_get_redis", lambda: _Boom())
        fake_log = MagicMock()
        monkeypatch.setattr(rc, "_log", fake_log)

        assert rc.redis_set("gridpulse:alerts:ERCOT", {"a": 1}) is False

        call = next(c for c in fake_log.error.call_args_list if c.args[0] == "redis_write_failed")
        assert call.kwargs["key"] == "gridpulse:alerts:ERCOT"

    def test_failures_are_counted_and_grouped_by_kind(self, monkeypatch):
        import data.redis_client as rc

        class _Boom:
            def setex(self, *a, **kw):
                raise RuntimeError("redis down")

        monkeypatch.setattr(rc, "_get_redis", lambda: _Boom())
        monkeypatch.setattr(rc, "_log", MagicMock())

        for k in ("gridpulse:alerts:ERCOT", "gridpulse:alerts:PJM", "gridpulse:meta:last_scored"):
            rc.redis_set(k, {})

        stats = rc.drain_write_failures()
        assert stats["count"] == 3
        # Grouped by key kind — 51 BAs of raw keys would be unreadable in a log.
        assert stats["by_kind"] == {"alerts": 2, "meta": 1}

    def test_drain_clears_so_runs_do_not_accumulate(self, monkeypatch):
        import data.redis_client as rc

        rc._write_failures.append("gridpulse:drift:ERCOT")
        assert rc.drain_write_failures()["count"] == 1
        assert rc.drain_write_failures() is None

    def test_successful_write_records_nothing(self, monkeypatch):
        import data.redis_client as rc

        monkeypatch.setattr(rc, "_get_redis", lambda: MagicMock())
        assert rc.redis_set("gridpulse:alerts:ERCOT", {"a": 1}) is True
        assert rc.drain_write_failures() is None

    def test_absent_client_is_not_counted_as_a_failure(self, monkeypatch):
        """No REDIS_HOST is dev mode, not an outage — counting it would make
        every local run look like a production incident."""
        import data.redis_client as rc

        monkeypatch.setattr(rc, "_get_redis", lambda: None)
        assert rc.redis_set("gridpulse:alerts:ERCOT", {"a": 1}) is False
        assert rc.drain_write_failures() is None

    def test_fail_soft_contract_is_unchanged(self, monkeypatch):
        """The point of this change is visibility, NOT converting fail-soft to
        fail-loud — `persist()` is the fail-loud twin and stays the way the
        critical writes opt in."""
        import data.redis_client as rc

        class _Boom:
            def setex(self, *a, **kw):
                raise RuntimeError("redis down")

        monkeypatch.setattr(rc, "_get_redis", lambda: _Boom())
        monkeypatch.setattr(rc, "_log", MagicMock())

        assert rc.redis_set("gridpulse:alerts:ERCOT", {}) is False  # returns, never raises
        with pytest.raises(rc.RedisWriteError):
            rc.persist("gridpulse:forecast:ERCOT:1h", {})
