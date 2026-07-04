"""Unit tests for the #171 scoring-runtime headroom guardrail.

Covers ``jobs.scoring_job._check_runtime_headroom`` — the creep alarm that warns
when a completed run's elapsed_s approaches the Cloud Run task timeout for N
consecutive runs, before an outright timeout kills a tick.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class _FakeRedis:
    """Dict-backed redis_get/redis_set/redis_key so streak state persists across
    calls the way it would across job-run processes in prod."""

    def __init__(self):
        self.store: dict = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ttl=None):
        self.store[key] = value
        return True

    def key(self, suffix):
        return f"gridpulse:{suffix}"


def _run(elapsed, fake, monkeypatch, *, timeout=1000, frac=0.7, runs=3):
    """Invoke the guardrail once with a mocked log + the shared fake redis."""
    import config
    import jobs.scoring_job as sj

    monkeypatch.setattr(config, "SCORING_TASK_TIMEOUT_S", timeout)
    monkeypatch.setattr(config, "SCORING_RUNTIME_HEADROOM_FRACTION", frac)
    monkeypatch.setattr(config, "SCORING_RUNTIME_CREEP_RUNS", runs)
    fake_log = MagicMock()
    monkeypatch.setattr(sj, "log", fake_log)
    with (
        patch("data.redis_client.redis_get", side_effect=fake.get),
        patch("data.redis_client.redis_set", side_effect=fake.set),
        patch("data.redis_client.redis_key", side_effect=fake.key),
    ):
        sj._check_runtime_headroom(elapsed)
    return fake_log


def _events(fake_log) -> tuple[list[str], list[str]]:
    """(error-event-names, warning-event-names) emitted in one call."""
    errors = [c.args[0] for c in fake_log.error.call_args_list]
    warnings = [c.args[0] for c in fake_log.warning.call_args_list]
    return errors, warnings


class TestRuntimeHeadroomGuardrail:
    def test_alert_fires_after_n_consecutive_breaches(self, monkeypatch):
        # threshold = 0.7 * 1000 = 700s; 800s breaches. 3 consecutive -> alert.
        fake = _FakeRedis()
        for _ in range(2):
            log2 = _run(800.0, fake, monkeypatch)
            assert _events(log2)[0] == []  # no error yet
        log3 = _run(800.0, fake, monkeypatch)
        errors, _ = _events(log3)
        assert "scoring_runtime_creep" in errors
        assert fake.store["gridpulse:scoring_runtime_state"]["consecutive_breaches"] == 3

    def test_single_breach_warns_but_does_not_alert(self, monkeypatch):
        fake = _FakeRedis()
        log = _run(800.0, fake, monkeypatch)
        errors, warnings = _events(log)
        assert errors == []
        assert "scoring_runtime_headroom_low" in warnings

    def test_healthy_run_resets_the_streak(self, monkeypatch):
        fake = _FakeRedis()
        _run(800.0, fake, monkeypatch)
        _run(800.0, fake, monkeypatch)  # streak = 2
        assert fake.store["gridpulse:scoring_runtime_state"]["consecutive_breaches"] == 2
        _run(500.0, fake, monkeypatch)  # healthy -> reset
        assert fake.store["gridpulse:scoring_runtime_state"]["consecutive_breaches"] == 0
        # a subsequent breach starts the streak over, no immediate alert
        log = _run(800.0, fake, monkeypatch)
        assert _events(log)[0] == []

    def test_below_headroom_emits_nothing(self, monkeypatch):
        fake = _FakeRedis()
        log = _run(500.0, fake, monkeypatch)  # 50% of timeout
        errors, warnings = _events(log)
        assert errors == []
        assert "scoring_runtime_headroom_low" not in warnings
        assert fake.store["gridpulse:scoring_runtime_state"]["consecutive_breaches"] == 0

    def test_redis_failure_never_raises(self, monkeypatch):
        import jobs.scoring_job as sj

        fake_log = MagicMock()
        monkeypatch.setattr(sj, "log", fake_log)
        with patch("data.redis_client.redis_get", side_effect=RuntimeError("redis down")):
            sj._check_runtime_headroom(800.0)  # must not raise
        assert "scoring_runtime_headroom_check_failed" in [
            c.args[0] for c in fake_log.warning.call_args_list
        ]

    def test_zero_timeout_is_a_noop(self, monkeypatch):
        fake = _FakeRedis()
        log = _run(9999.0, fake, monkeypatch, timeout=0)
        assert _events(log) == ([], [])
        assert fake.store == {}  # never touched Redis

    def test_pct_and_threshold_recorded(self, monkeypatch):
        fake = _FakeRedis()
        _run(900.0, fake, monkeypatch, timeout=1000, frac=0.7)
        state = fake.store["gridpulse:scoring_runtime_state"]
        assert state["pct_of_timeout"] == pytest.approx(90.0)
        assert state["threshold_s"] == pytest.approx(700.0)
