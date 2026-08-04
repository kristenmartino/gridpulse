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


# ---------------------------------------------------------------------------
# #389 — per-phase runtime breakdown
# ---------------------------------------------------------------------------


class TestPhaseRollup:
    """``_phase_rollup`` turns per-region phase timings into the fleet answer
    to "where did the run go" — the number the creep runbook needs and that
    the job did not emit when the alert fired."""

    RESULTS = [
        {"region": "ERCOT", "timings": {"fetch": 30.0, "forecast": 5.0}},
        {"region": "MISO", "timings": {"fetch": 50.0, "forecast": 4.0, "alerts": 1.0}},
        {"region": "PJM", "timings": {"fetch": 20.0}},
    ]

    def test_totals_and_slowest_region(self):
        import jobs.scoring_job as sj

        rollup = sj._phase_rollup(self.RESULTS)

        assert rollup["fetch"]["total_s"] == pytest.approx(100.0)
        assert rollup["fetch"]["max_s"] == pytest.approx(50.0)
        assert rollup["fetch"]["slowest_region"] == "MISO"
        assert rollup["fetch"]["n"] == 3
        assert rollup["alerts"]["n"] == 1

    def test_sorted_by_total_descending(self):
        """The point of the rollup is ranking — the costliest phase leads."""
        import jobs.scoring_job as sj

        assert list(sj._phase_rollup(self.RESULTS)) == ["fetch", "forecast", "alerts"]

    def test_regions_without_timings_are_skipped(self):
        """A region that crashed before any phase ran has no ``timings`` key."""
        import jobs.scoring_job as sj

        rollup = sj._phase_rollup([{"region": "X", "ok": False, "error": "boom"}])
        assert rollup == {}

    def test_creep_alert_carries_the_breakdown(self, monkeypatch):
        """The alert log names the phases to go after, not just "slow"."""
        import config
        import jobs.scoring_job as sj

        monkeypatch.setattr(config, "SCORING_TASK_TIMEOUT_S", 1000)
        monkeypatch.setattr(config, "SCORING_RUNTIME_HEADROOM_FRACTION", 0.7)
        monkeypatch.setattr(config, "SCORING_RUNTIME_CREEP_RUNS", 1)
        fake, fake_log = _FakeRedis(), MagicMock()
        monkeypatch.setattr(sj, "log", fake_log)
        rollup = sj._phase_rollup(self.RESULTS)

        with (
            patch("data.redis_client.redis_get", side_effect=fake.get),
            patch("data.redis_client.redis_set", side_effect=fake.set),
            patch("data.redis_client.redis_key", side_effect=fake.key),
        ):
            sj._check_runtime_headroom(900.0, rollup)

        call = next(
            c for c in fake_log.error.call_args_list if c.args[0] == "scoring_runtime_creep"
        )
        top = call.kwargs["top_phases"]
        assert top[0][0] == "fetch"
        assert top[0][1]["total_s"] == pytest.approx(100.0)

    def test_rollup_is_optional(self, monkeypatch):
        """Called without a rollup (the pre-#389 signature) it must still work."""
        fake = _FakeRedis()
        log = _run(900.0, fake, monkeypatch, timeout=1000, runs=1)
        assert any(c.args[0] == "scoring_runtime_creep" for c in log.error.call_args_list)


class TestRegionCompleteLogging:
    """``_log_region_complete`` — the per-BA runtime line.

    ``_score_region`` computed ``elapsed_s`` and dropped it on the floor, so
    the fleet total was the only runtime number the job produced. The case
    that matters most is the one an obvious implementation misses: a BA whose
    fetch FAILS returns early, and under an upstream slowdown that BA's fetch
    is the slowest in the fleet — it paid the full retry budget and still got
    nothing. Logging only on the success path would hide exactly that.
    """

    def test_no_data_region_still_emits_its_timing(self, monkeypatch):
        import jobs.phases as phases
        import jobs.scoring_job as sj

        fake_log = MagicMock()
        monkeypatch.setattr(sj, "log", fake_log)
        # fetch_region_data returning None is the "upstream gave us nothing"
        # path — the early return that previously logged no timing at all.
        monkeypatch.setattr(phases, "fetch_region_data", lambda region: None)

        summary = sj._score_region("ERCOT")

        assert summary["ok"] is False
        assert summary["phases"]["fetch"] == {"ok": False, "error": "no_data"}
        call = next(
            c for c in fake_log.info.call_args_list if c.args[0] == "scoring_region_complete"
        )
        assert call.kwargs["region"] == "ERCOT"
        assert call.kwargs["ok"] is False
        # The fetch was still timed — that number is the whole point here.
        assert "fetch" in call.kwargs["timings"]

    def test_timings_are_ordered_slowest_first(self, monkeypatch):
        import jobs.scoring_job as sj

        fake_log = MagicMock()
        monkeypatch.setattr(sj, "log", fake_log)

        sj._log_region_complete(
            {
                "region": "MISO",
                "ok": True,
                "elapsed_s": 42.0,
                "timings": {"alerts": 1.0, "fetch": 30.0, "forecast": 5.0},
            }
        )

        call = fake_log.info.call_args
        assert list(call.kwargs["timings"]) == ["fetch", "forecast", "alerts"]

    def test_missing_timings_does_not_raise(self, monkeypatch):
        """A summary shaped by an older code path must not break the log line."""
        import jobs.scoring_job as sj

        monkeypatch.setattr(sj, "log", MagicMock())
        sj._log_region_complete({"region": "PJM"})  # no timings, no elapsed_s
