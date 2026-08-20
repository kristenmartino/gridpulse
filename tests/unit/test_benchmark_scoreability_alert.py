"""The scorecard's population is watched, and the watcher matches the policy (#535).

The benchmark scorecard published **25 of 51** for roughly three weeks, with
five of the seven large ISOs missing from its fleet medians, while the scoring
job succeeded and exited 0 every hour. It was found by an unrelated scheduled
recheck. Nothing in CI or the job flagged it, because nothing was looking at the
population a published median is computed over.

Two things have to hold for the alarm to be worth anything, and both are pinned
here rather than assumed:

1. the rule fires on the situations it exists for, and stays quiet otherwise;
2. the *event names it emits* are the ones the Cloud Monitoring policy filters
   on. A renamed event is a policy that silently never fires again — the failure
   mode `backtest_recompute_alert.json` was written to avoid, applied here.
"""

from __future__ import annotations

import json
from pathlib import Path

from config import BENCHMARK_DF_GAP_WARN_HOURS
from models.benchmark import MAX_DF_GAP_HOURS, scoreability_alerts
from tests.unit.test_benchmark import _Rec

_ROOT = Path(__file__).resolve().parents[2]
#: One policy per event. The Cloud Monitoring API rejects a policy carrying
#: more than one log-matching condition, and the split is the better shape
#: anyway: the drop is an incident (the page is already wrong), the at-risk
#: is a lead (a BA is 1-3 points from falling out).
_POLICIES = sorted((_ROOT / "docs/monitoring").glob("benchmark_*_alert.json"))


def _payload(region, *, scoreable=True, cov=1.0, asissued=1.0, gap=0.0, stale=0.0):
    return {
        "region": region,
        "scoreable": scoreable,
        "df_coverage": cov,
        "df_asissued_coverage": asissued,
        "df_longest_gap_hours": gap,
        "df_stale_hours": stale,
    }


def _rollup(n_scoreable, excluded=()):
    return {
        "n_scoreable": n_scoreable,
        "n_excluded": len(excluded),
        "excluded": [{"region": r} for r in excluded],
    }


class TestTheDropAlarm:
    def test_the_535_incident_fires_it(self):
        """25 of 51, five large ISOs gone. The case the alarm exists for."""
        gone = ["ERCOT", "MISO", "NYISO", "ISONE", "SPP"]
        alerts = scoreability_alerts(_rollup(25, gone), [], min_scoreable=40)
        drop = [a for a in alerts if a["event"] == "benchmark_scoreability_drop"]
        assert len(drop) == 1
        assert drop[0]["n_scoreable"] == 25
        assert drop[0]["excluded_regions"] == sorted(gone), (
            "the alert must name WHICH BAs left — '26 excluded' and '26 excluded "
            "including five large ISOs' are different pages"
        )

    def test_a_healthy_fleet_is_silent(self):
        assert scoreability_alerts(_rollup(46), [], min_scoreable=40) == []

    def test_the_floor_is_exclusive_not_inclusive(self):
        """Exactly at the floor is acceptable; below it is not. Pinned because
        an off-by-one here is a page that alerts constantly or never."""
        assert scoreability_alerts(_rollup(40), [], min_scoreable=40) == []
        assert len(scoreability_alerts(_rollup(39), [], min_scoreable=40)) == 1


class TestTheEarlyWarning:
    def test_it_names_the_bas_that_were_next_to_fall(self):
        """SPP's shape, one warning band before it fell out. 130h of gap is
        past the 120h warn line and short of the 168h gate.

        The fixture used to be a coverage rate (TEC at 80.1%). That warning was
        retired in #587 — not because it never fired, but because it fired on
        EVERY tick for a healthy TEC once coverage stopped gating, and because
        its ordering against the gate was an arithmetic accident rather than a
        property anything could test.
        """
        payloads = [
            _payload("SPP", gap=130.0, stale=130.0, cov=0.72),
            _payload("BANC", gap=3.0, cov=0.993),
        ]
        at_risk = [
            a
            for a in scoreability_alerts(_rollup(45), payloads, gap_warn_hours=120.0)
            if a["event"] == "benchmark_df_gap_at_risk"
        ]
        assert [a["region"] for a in at_risk] == ["SPP"]
        assert at_risk[0]["gate_gap_hours"] == MAX_DF_GAP_HOURS
        assert at_risk[0]["df_longest_gap_hours"] == 130.0
        # Trailing beside longest: the reader must be able to tell "down now"
        # from "came back, hole still in the window" without opening a payload.
        assert at_risk[0]["df_stale_hours"] == 130.0
        assert at_risk[0]["df_asissued_coverage"] == 1.0
        assert "gate" not in at_risk[0], "a retired threshold must not be republished"

    def test_the_band_is_wider_than_a_payload_can_go_stale(self):
        """The sufficiency half, and it was nearly missed.

        `scoreability_alerts` is fed payloads read back from Redis, not the
        ones just computed, so what it sees can lag. If a payload could lag by
        more than the band width, a BA's observed gap could jump from under the
        warn line to over the gate with no tick in between and the warning
        would simply never fire.

        It cannot, because `REDIS_TTL` (24h) is narrower than the band (48h) —
        a payload stale enough to skip the band has expired instead, dropping
        the BA out of the fleet and into `benchmark_scoreability_drop`, which
        is louder. Two constants in two files with nothing connecting them:
        exactly the accidental ordering #587 exists to remove, so it is pinned
        rather than left to hold by luck.
        """
        from jobs.phases import REDIS_TTL

        band_hours = MAX_DF_GAP_HOURS - BENCHMARK_DF_GAP_WARN_HOURS
        assert band_hours > REDIS_TTL / 3600.0, (
            f"a benchmark payload can go {REDIS_TTL / 3600:.0f}h stale against a "
            f"{band_hours:.0f}h warning band — widen the band or shorten the TTL, "
            "or a dying feed can cross the gate unannounced"
        )

    def test_a_real_tick_sequence_passes_through_the_band(self):
        """The ordering claim against `_df_gaps` itself, not against synthetic
        payload dicts. Grow a hole one hour per tick the way a dead feed does,
        run every tick through the real scoreability path, and require that the
        BA is warned-and-still-scored on at least one of them before it is
        excluded on a later one."""
        from models.benchmark import scoreability

        def _recs(gap_hours):
            out = [
                _Rec(f"2026-07-{1 + h // 24:02d}T{h % 24:02d}:00:00Z", 900.0, 1000.0)
                for h in range(200)
            ]
            out += [
                _Rec(f"2026-07-{1 + h // 24:02d}T{h % 24:02d}:00:00Z", float("nan"), 1000.0)
                for h in range(200, 200 + gap_hours)
            ]
            return out

        warned_and_scored, excluded_at = [], None
        for gap in range(0, int(MAX_DF_GAP_HOURS) + 30):
            sc = scoreability(_recs(gap), "clean")
            payload = {"region": "X", **sc}
            alerts = scoreability_alerts(
                _rollup(46), [payload], gap_warn_hours=BENCHMARK_DF_GAP_WARN_HOURS
            )
            at_risk = [a for a in alerts if a["event"] == "benchmark_df_gap_at_risk"]
            if sc["scoreable"] and at_risk:
                warned_and_scored.append(gap)
            if not sc["scoreable"] and excluded_at is None:
                excluded_at = gap

        assert warned_and_scored, "no tick warned while the BA was still scored"
        assert excluded_at is not None, "the BA never crossed the gate"
        assert max(warned_and_scored) < excluded_at, (
            "every warning must land strictly before the exclusion"
        )
        assert excluded_at - min(warned_and_scored) >= 24, (
            "the lead time is under a day — too short to act on"
        )

    def test_the_warning_and_the_gate_are_the_same_measurement(self):
        """The whole of #587. The old warning was a coverage RATE and the gate
        a DURATION, so "warn before exclude" depended on the window length and
        two unrelated constants — and nothing tested it, because the two
        numbers were not comparable. Both are now hours of gap, so this one
        assertion IS the proof, for any window."""
        assert BENCHMARK_DF_GAP_WARN_HOURS < MAX_DF_GAP_HOURS

    def test_a_growing_gap_warns_while_the_page_is_still_right(self):
        """Ordering, behaviourally: as a hole grows there is a band where the
        BA is still scoreable AND warned. That band is what an on-call reader
        gets to act in, and it must not be empty."""
        warned_while_scoreable = [
            gap
            for gap in range(0, int(MAX_DF_GAP_HOURS) + 1, 6)
            if scoreability_alerts(
                _rollup(46),
                [_payload("X", gap=float(gap))],
                gap_warn_hours=BENCHMARK_DF_GAP_WARN_HOURS,
            )
        ]
        assert warned_while_scoreable, "no gap size warns before the gate — the band is empty"
        assert min(warned_while_scoreable) >= BENCHMARK_DF_GAP_WARN_HOURS
        assert max(warned_while_scoreable) <= MAX_DF_GAP_HOURS

    def test_it_warns_below_the_gate_not_at_it(self):
        """A BA that has already fallen out is a page that is already wrong —
        a warning at the gate would arrive too late to be one."""
        at_risk = scoreability_alerts(_rollup(46), [_payload("X", gap=130.0)], gap_warn_hours=120.0)
        assert len(at_risk) == 1, "130h is short of the 168h gate and must still warn"

    def test_already_excluded_bas_do_not_double_report(self):
        """They are in the drop alert's `excluded_regions`; re-reporting each one
        here would bury the early warning under the incident it precedes."""
        payloads = [_payload("SPP", scoreable=False, gap=391.0, cov=0.46)]
        assert not [
            a
            for a in scoreability_alerts(_rollup(46), payloads, gap_warn_hours=120.0)
            if a["event"] == "benchmark_df_gap_at_risk"
        ]

    def test_our_capture_rate_rides_along_but_never_triggers(self):
        """`df_asissued_coverage` is OUR number. Alerting on it would repeat
        #535's actual mistake — treating a collector gap as the BA's behaviour."""
        payloads = [_payload("X", cov=1.0, asissued=0.10, gap=0.0)]
        assert scoreability_alerts(_rollup(46), payloads, gap_warn_hours=120.0) == []

    def test_a_normal_whole_day_outage_does_not_warn(self):
        """Fitted to the fleet, 2026-08-20: the worst live BA's longest gap was
        52h (SPA), and whole-day 24-30h holes are routine. A warning that fires
        on those is the permanently-firing alert #587 retired."""
        for gap in (24.0, 30.0, 48.0, 52.0):
            assert not scoreability_alerts(
                _rollup(46),
                [_payload("X", gap=gap)],
                gap_warn_hours=BENCHMARK_DF_GAP_WARN_HOURS,
            ), f"{gap}h is a routine outage and must not warn"


class TestThePolicyMatchesTheCode:
    """A renamed event is a policy that silently never fires again."""

    def test_every_emitted_event_is_filtered_by_the_policy(self):
        emitted = {
            a["event"]
            for a in scoreability_alerts(
                _rollup(25, ["ERCOT"]),
                [_payload("CAISO", gap=130.0)],
                min_scoreable=40,
                gap_warn_hours=120.0,
            )
        }
        assert emitted == {"benchmark_scoreability_drop", "benchmark_df_gap_at_risk"}

        assert _POLICIES, "no benchmark alert policy files found"
        filters = " ".join(
            c["conditionMatchedLog"]["filter"]
            for path in _POLICIES
            for c in json.loads(path.read_text())["conditions"]
        )
        for event in emitted:
            assert f'jsonPayload.event="{event}"' in filters, (
                f"{event} is emitted but no committed benchmark policy matches it — "
                "the alert would never fire"
            )

    def test_each_policy_carries_exactly_one_log_condition(self):
        """The Cloud Monitoring API rejects more than one, and it does so at
        apply time — so a combined policy passes every local check and then
        cannot be deployed. Pinned because that is exactly how this shipped
        wrong the first time."""
        for path in _POLICIES:
            conds = json.loads(path.read_text())["conditions"]
            assert len(conds) == 1, (
                f"{path.name} has {len(conds)} conditions; "
                "'Alert policies with a log matching condition can only have a "
                "single condition' — split it"
            )

    def test_the_job_emits_them_at_error(self):
        """`log.error`, not `log.info`. Both the policy and any severity-based
        routing depend on it, and a scorecard quietly changing the fleet it
        describes is a public-correctness bug rather than a curiosity."""
        src = (_ROOT / "jobs/scoring_job.py").read_text()
        assert "scoreability_alerts(rollup, payloads)" in src, (
            "the scoring job must call the rule — it is the only caller, and "
            "without it the policy has nothing to match"
        )
        assert 'log.error(alert.pop("event"), **alert)' in src
