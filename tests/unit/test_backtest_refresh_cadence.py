"""Tests for the backtest refresh cadence gate.

The walk-forward backtest is the training job's largest single cost: 12 folds
per BA, and **every fold trains its own booster** (walk-forward requires it).
Measured, `train_xgboost(cross_validate=False)` is 11.26s per fold against
0.42s for the fold's recursive predict loop — training is 96% of fold cost,
~141s per BA, ~120 minutes across 51 BAs, ~29% of the job's task-seconds.

It ran daily only because the payloads carried a 24h TTL: the refresh interval
and the expiry were the same number, so skipping a day blanked the Models tab.

What these tests pin is the honesty of the replacement, not the saving:

* Anything unparseable counts as STALE, so the failure mode is a wasted
  recomputation rather than a number that silently never refreshes again.
* The TTL must OUTLIVE the refresh interval. If they are equal, a key expires
  exactly when it comes due and any hiccup empties the tab — which is the
  original bug, reintroduced.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from jobs.phases import _backtest_is_fresh


def _payload(age: timedelta | None, *, stamp: object = None) -> dict:
    if stamp is not None:
        return {"computed_at": stamp}
    if age is None:
        return {}
    return {"computed_at": (datetime.now(UTC) - age).isoformat()}


class TestFreshnessGate:
    def test_recent_payload_is_fresh(self):
        assert _backtest_is_fresh(_payload(timedelta(days=1)), 7) is True

    def test_payload_at_the_boundary_is_stale(self):
        assert _backtest_is_fresh(_payload(timedelta(days=7, seconds=30)), 7) is False

    def test_payload_just_inside_the_window_is_fresh(self):
        assert _backtest_is_fresh(_payload(timedelta(days=6, hours=23)), 7) is True


class TestEverythingUnparseableIsStale:
    """Fail toward doing the work. The alternative — treating an unreadable
    payload as fresh — is a number that never refreshes and no signal that it
    stopped."""

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            {},
            {"computed_at": None},
            {"computed_at": 1234567890},
            {"computed_at": "not-a-timestamp"},
            {"computed_at": ""},
            "not-a-dict",
            [],
        ],
    )
    def test_unparseable_is_stale(self, payload):
        assert _backtest_is_fresh(payload, 7) is False

    def test_payload_predating_the_field_is_stale(self):
        """Payloads written before `computed_at` existed must recompute once."""
        legacy = {"horizon": 168, "metrics": {"xgboost": {"mape": 4.1}}}
        assert _backtest_is_fresh(legacy, 7) is False

    def test_naive_timestamp_is_treated_as_utc(self):
        """A stamp without tzinfo must not raise on the subtraction."""
        naive = (datetime.now(UTC) - timedelta(days=1)).replace(tzinfo=None).isoformat()
        assert _backtest_is_fresh({"computed_at": naive}, 7) is True


class TestDisablingTheGate:
    @pytest.mark.parametrize("days", [0, -1])
    def test_non_positive_refresh_days_always_recomputes(self, days):
        """The escape hatch: BACKTEST_REFRESH_DAYS=0 restores every-run
        behaviour, which is what validating a change to the backtest needs."""
        assert _backtest_is_fresh(_payload(timedelta(seconds=1)), days) is False


class TestTtlOutlivesTheRefreshInterval:
    def test_ttl_exceeds_refresh_window(self):
        """THE ORIGINAL BUG. A 24h TTL on a daily refresh meant expiry and
        recompute coincided; one missed run blanked the Models tab. The TTL
        must be strictly longer, or the cadence change reintroduces it."""
        from config import BACKTEST_REFRESH_DAYS, BACKTEST_TTL_SECONDS

        assert BACKTEST_TTL_SECONDS > BACKTEST_REFRESH_DAYS * 24 * 3600, (
            "backtest TTL must outlive the refresh interval, or a key expires "
            "exactly when it comes due for refresh"
        )


class TestRecomputeIsObservable:
    """The alert signal (`docs/monitoring/backtest_recompute_alert.json`).

    The alert counts RECOMPUTES rather than skips on purpose. A metric built on
    ``job_backtest_fresh_skip`` looks like the obvious choice and is a trap:
    when the gate breaks, skips stop being emitted, the logs-based counter has
    no data, and a threshold-below condition never evaluates — the alert goes
    quiet at exactly the moment it should fire. Counting the thing that
    increases on failure removes that failure mode, but only if the recompute
    path actually emits it.
    """

    def test_recompute_path_emits_the_alert_event(self):
        import inspect

        from jobs import phases

        src = inspect.getsource(phases.write_backtests)
        assert '"job_backtest_recomputed"' in src, (
            "the alert in docs/monitoring/backtest_recompute_alert.json counts "
            "job_backtest_recomputed; without it the policy silently never fires"
        )

    def test_recompute_event_carries_the_diagnostic_field(self):
        """`previous_computed_at` is what separates the two root causes: null
        means the payload was missing (Redis flush, eviction, TTL expiry), a
        recent timestamp means the gate rejected a payload it should have
        accepted."""
        import inspect

        from jobs import phases

        assert "previous_computed_at" in inspect.getsource(phases.write_backtests)


class TestRecomputeCadenceGuard:
    """`check_backtest_recompute_cadence` — the frequency detection that Cloud
    Monitoring cannot express.

    A metric threshold over a multi-day window is the natural design and GCP
    rejects it: *"Alignment periods longer than 25h are not supported."* A
    <=25h window fails too — the job runs once daily, so consecutive recompute
    days land in adjacent windows and never the same one. Hence a marker in
    Redis compared in code.
    """

    def _patch(self, monkeypatch, previous, written):
        from jobs import phases

        monkeypatch.setattr(phases, "redis_get", lambda *a, **k: previous, raising=False)
        monkeypatch.setattr("data.redis_client.redis_get", lambda *a, **k: previous)
        monkeypatch.setattr("data.redis_client.redis_key", lambda k: k)
        monkeypatch.setattr(phases, "write_meta", lambda *a, **k: written.append(a[0]))

    def test_no_recompute_means_no_check_and_no_marker_write(self, monkeypatch):
        """A skip-day run must not touch the marker — refreshing it on a day
        nothing recomputed would push the next comparison a week out and
        silently disarm the guard."""
        from jobs import phases

        written: list = []
        self._patch(monkeypatch, None, written)
        assert phases.check_backtest_recompute_cadence(0) is False
        assert written == []

    def test_first_recompute_has_no_prior_and_does_not_alert(self, monkeypatch):
        """Correct, not a gap being hidden: there is genuinely nothing to
        compare against. The guard is armed from the SECOND recompute."""
        from jobs import phases

        written: list = []
        self._patch(monkeypatch, None, written)
        assert phases.check_backtest_recompute_cadence(51) is False
        assert written == ["last_backtest_recompute"]

    def test_recompute_too_soon_is_flagged(self, monkeypatch):
        from datetime import UTC, datetime, timedelta

        from jobs import phases

        written: list = []
        recent = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        self._patch(monkeypatch, {"updated_at": recent}, written)
        assert phases.check_backtest_recompute_cadence(51) is True

    def test_recompute_on_schedule_is_not_flagged(self, monkeypatch):
        from datetime import UTC, datetime, timedelta

        from jobs import phases

        written: list = []
        on_time = (datetime.now(UTC) - timedelta(days=7)).isoformat()
        self._patch(monkeypatch, {"updated_at": on_time}, written)
        assert phases.check_backtest_recompute_cadence(51) is False

    def test_one_day_of_scheduling_slack(self, monkeypatch):
        """The job runs at a fixed hour; a run starting minutes early must not
        alert. 6 days against a 7-day cadence is inside the slack."""
        from datetime import UTC, datetime, timedelta

        from jobs import phases

        written: list = []
        early = (datetime.now(UTC) - timedelta(days=6, hours=1)).isoformat()
        self._patch(monkeypatch, {"updated_at": early}, written)
        assert phases.check_backtest_recompute_cadence(51) is False

    def test_unparseable_marker_fails_toward_silence(self, monkeypatch):
        """Crying wolf on the first run after any format change would be worse
        than a missed detection — and the marker is still refreshed, so the
        guard self-heals on the next recompute."""
        from jobs import phases

        written: list = []
        self._patch(monkeypatch, {"updated_at": "not-a-timestamp"}, written)
        assert phases.check_backtest_recompute_cadence(51) is False
        assert written == ["last_backtest_recompute"]

    def test_marker_is_refreshed_even_when_flagged(self, monkeypatch):
        """A stuck marker would suppress every future detection."""
        from datetime import UTC, datetime, timedelta

        from jobs import phases

        written: list = []
        self._patch(
            monkeypatch,
            {"updated_at": (datetime.now(UTC) - timedelta(hours=6)).isoformat()},
            written,
        )
        assert phases.check_backtest_recompute_cadence(51) is True
        assert written == ["last_backtest_recompute"]
