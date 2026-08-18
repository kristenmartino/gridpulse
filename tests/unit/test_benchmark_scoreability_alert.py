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

from models.benchmark import MIN_DF_COVERAGE, scoreability_alerts

_ROOT = Path(__file__).resolve().parents[2]
#: One policy per event. The Cloud Monitoring API rejects a policy carrying
#: more than one log-matching condition, and the split is the better shape
#: anyway: the drop is an incident (the page is already wrong), the at-risk
#: is a lead (a BA is 1-3 points from falling out).
_POLICIES = sorted((_ROOT / "docs/monitoring").glob("benchmark_*_alert.json"))


def _payload(region, *, scoreable=True, cov=1.0, asissued=1.0):
    return {
        "region": region,
        "scoreable": scoreable,
        "df_coverage": cov,
        "df_asissued_coverage": asissued,
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
        """CAISO 82.9% and PJM 81.0% on 2026-08-17 — still scoreable, 1-3 points
        above the gate, with nothing watching them."""
        payloads = [
            _payload("CAISO", cov=0.8289, asissued=0.7940),
            _payload("PJM", cov=0.8095, asissued=0.7810),
            _payload("BANC", cov=0.9930, asissued=0.9900),
        ]
        at_risk = [
            a
            for a in scoreability_alerts(_rollup(46), payloads, coverage_warn=0.85)
            if a["event"] == "benchmark_coverage_at_risk"
        ]
        assert [a["region"] for a in at_risk] == ["CAISO", "PJM"]
        assert all(a["gate"] == MIN_DF_COVERAGE for a in at_risk)

    def test_it_warns_above_the_gate_not_at_it(self):
        """The whole point. A BA that has already fallen out is a page that is
        already wrong — a warning at the gate would arrive too late to be one."""
        at_risk = scoreability_alerts(_rollup(46), [_payload("X", cov=0.83)], coverage_warn=0.85)
        assert len(at_risk) == 1, "0.83 is above the 0.80 gate and must still warn"

    def test_already_excluded_bas_do_not_double_report(self):
        """They are in the drop alert's `excluded_regions`; re-reporting each one
        here would bury the early warning under the incident it precedes."""
        payloads = [_payload("SPP", scoreable=False, cov=0.538, asissued=0.395)]
        assert not [
            a
            for a in scoreability_alerts(_rollup(46), payloads, coverage_warn=0.85)
            if a["event"] == "benchmark_coverage_at_risk"
        ]

    def test_our_capture_rate_rides_along_but_never_triggers(self):
        """`df_asissued_coverage` is OUR number. Alerting on it would repeat
        #535's actual mistake — treating a collector gap as the BA's behaviour."""
        payloads = [_payload("X", cov=1.0, asissued=0.10)]
        assert scoreability_alerts(_rollup(46), payloads, coverage_warn=0.85) == []


class TestThePolicyMatchesTheCode:
    """A renamed event is a policy that silently never fires again."""

    def test_every_emitted_event_is_filtered_by_the_policy(self):
        emitted = {
            a["event"]
            for a in scoreability_alerts(
                _rollup(25, ["ERCOT"]),
                [_payload("CAISO", cov=0.8289)],
                min_scoreable=40,
                coverage_warn=0.85,
            )
        }
        assert emitted == {"benchmark_scoreability_drop", "benchmark_coverage_at_risk"}

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
