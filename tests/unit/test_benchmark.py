"""The public forecast benchmark (models/benchmark.py).

This benchmark will be published and read adversarially by people whose job
is finding holes in accuracy claims. These tests pin the two traps that
would embarrass it, plus the exclusion contract:

1. **The stub trap** — EIA publishes the official forecast *as* the actual
   for not-yet-reported hours (``D == DF``). Scoring those hours credits the
   official forecast with a perfect prediction on hours it never made.
2. **The preliminary-actuals trap** — settled values only, never
   as-first-published (up to 70% wrong on high-revision feeds).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.benchmark import (
    CONSERVATIVE_LEAD,
    EXCLUDE_BROKEN_FEED,
    EXCLUDE_DF_COVERAGE,
    HEADLINE_LEAD,
    MIN_PAIRED_HOURS,
    OFFICIAL_DOCUMENTED_LEAD_H,
    compute_benchmark_payload,
    fleet_rollup,
    gridpulse_predictions,
    pair_hours,
    revised_df_from_frame,
    score_arm,
    scoreability,
    wape,
)


class _Rec:
    """Minimal stand-in for data.vintage.VintageRecord."""

    def __init__(self, ts, first_seen_df, last_d, was_placeholder=False):
        self.timestamp = ts
        self.first_seen_df = first_seen_df
        self.last_d = last_d
        self.was_placeholder = was_placeholder


def _records(n=300, *, df=1000.0, actual=1000.0, placeholder=False):
    return [
        _Rec(f"2026-07-{1 + i // 24:02d}T{i % 24:02d}:00:00Z", df, actual, placeholder)
        for i in range(n)
    ]


def _gp(records, value):
    return {r.timestamp: value for r in records}


class TestStubTrap:
    """The single most important behavior in the module."""

    def test_placeholder_hours_are_never_scored(self):
        """A stub hour has D == DF, so the official forecast looks perfect.
        Scoring it would flatter the incumbent with our own bug."""
        recs = _records(10, df=1000.0, actual=1000.0, placeholder=True)
        assert pair_hours(recs, _gp(recs, 900.0))[0] == []

    def test_official_cannot_score_perfect_via_stubs(self):
        """End-to-end: a BA whose hours are ALL stubs must not produce a
        flawless official score — it must fail the sample-size gate instead."""
        recs = _records(300, df=1000.0, actual=1000.0, placeholder=True)
        payload = compute_benchmark_payload(
            "TEST", recs, _horizon(recs, 950.0), revision_class="clean"
        )
        lead = payload["leads"][HEADLINE_LEAD]
        assert lead["scoreable"] is False
        assert lead["n"] == 0

    def test_mixed_stub_and_real_hours_scores_only_the_real_ones(self):
        real = _records(250, df=1100.0, actual=1000.0)
        stubs = [
            _Rec(f"2026-08-{1 + i // 24:02d}T{i % 24:02d}:00:00Z", 1000.0, 1000.0, True)
            for i in range(100)
        ]
        pairs, _drops = pair_hours(real + stubs, _gp(real + stubs, 1000.0))
        assert len(pairs) == 250
        assert all(p.official == 1100.0 for p in pairs)


class TestUnresolvedStubPredicate:
    """The sharper stub test, added after design review.

    ``was_placeholder`` only records what EIA said at FIRST sight. The
    condition that actually poisons a comparison is settled truth still
    equalling the day-ahead forecast: the official arm then scores 0% by
    construction, and our arm is graded against their forecast instead of
    reality. Both arms are poisoned, not just theirs.
    """

    def test_settled_equals_df_is_dropped_even_without_the_flag(self):
        recs = [_Rec("2026-07-01T00:00:00Z", 1000.0, 1000.0, was_placeholder=False)]
        pairs, drops = pair_hours(recs, _gp(recs, 900.0))
        assert pairs == []
        assert drops["unresolved_stub"] == 1
        assert drops["first_seen_placeholder"] == 0

    def test_corrected_stub_is_still_dropped_conservatively(self):
        """Flagged at first sight but later corrected — kept out for now;
        keeping it is a defensible refinement, so the count is published."""
        recs = [_Rec("2026-07-01T00:00:00Z", 1000.0, 1200.0, was_placeholder=True)]
        pairs, drops = pair_hours(recs, _gp(recs, 1150.0))
        assert pairs == []
        assert drops["first_seen_placeholder"] == 1

    def test_missing_df_counted_separately_from_stubs(self):
        """A missing DF reads as 'not a stub' on both predicates, so it must
        be checked independently or it silently becomes a scored hour."""
        recs = [_Rec("2026-07-01T00:00:00Z", float("nan"), 1000.0)]
        pairs, drops = pair_hours(recs, _gp(recs, 900.0))
        assert pairs == []
        assert drops["no_df"] == 1
        assert drops["unresolved_stub"] == 0

    def test_drop_counts_are_published_on_the_payload(self):
        """Exclusions are not neutral across BAs — a stub-heavy BA loses far
        more hours than a clean one, and a reader who cannot see that will
        assume the worst."""
        recs = _records(300, df=1100.0, actual=1000.0)
        payload = compute_benchmark_payload("TEST", recs, _horizon(recs, 1000.0), "clean")
        assert "excluded_hours" in payload["leads"][HEADLINE_LEAD]

    def test_lead_is_labelled_nominal_not_realized(self):
        """The forecast anchors on the last real demand hour, so EIA's
        publishing lag makes the true lead shorter than the label."""
        recs = _records(300, df=1100.0, actual=1000.0)
        payload = compute_benchmark_payload("TEST", recs, _horizon(recs, 1000.0), "clean")
        assert payload["leads"][HEADLINE_LEAD]["lead_basis"] == "nominal"


class TestTruthDiscipline:
    def test_both_arms_score_against_the_same_actual(self):
        """Neither side may be graded against its own yardstick."""
        recs = _records(5, df=1100.0, actual=1000.0)
        pairs, _drops = pair_hours(recs, _gp(recs, 900.0))
        assert all(p.actual == 1000.0 for p in pairs)
        official = score_arm(pairs, "official")
        gridpulse = score_arm(pairs, "gridpulse")
        assert official["mape"] == pytest.approx(10.0)
        assert gridpulse["mape"] == pytest.approx(10.0)

    def test_unsettled_or_nonpositive_actuals_are_dropped(self):
        recs = [
            _Rec("2026-07-01T00:00:00Z", 100.0, float("nan")),
            _Rec("2026-07-01T01:00:00Z", 100.0, 0.0),
            _Rec("2026-07-01T02:00:00Z", 100.0, -5.0),
            _Rec("2026-07-01T03:00:00Z", 100.0, 90.0),
        ]
        pairs, _drops = pair_hours(recs, _gp(recs, 95.0))
        assert len(pairs) == 1

    def test_hour_without_a_gridpulse_forecast_is_dropped(self):
        """No paired hour means no comparison — never a one-sided score.

        Publishing a 30-day official record beside a 1-day GridPulse one is
        the exact failure this benchmark exists to avoid. The records here
        must clear every EARLIER filter (non-stub, settled, DF present) so
        this asserts the gridpulse join specifically.
        """
        recs = _records(5, df=1100.0, actual=1000.0)
        pairs, drops = pair_hours(recs, {})
        assert pairs == []
        assert drops["no_gridpulse"] == 5, "hours dropped for the wrong reason"


class TestExclusions:
    def test_broken_feed_excluded_and_names_the_circularity(self):
        """ADR-009 seeds our anchor from the BA's own day-ahead forecast on
        broken feeds, so the comparison would be partly self-referential.
        The published reason must say so."""
        s = scoreability(_records(300), revision_class="broken")
        assert s["scoreable"] is False
        assert s["reason"] == EXCLUDE_BROKEN_FEED
        assert "self-referential" in s["reason_detail"]

    def test_sparse_day_ahead_coverage_excluded(self):
        recs = _records(200) + [
            _Rec(f"2026-09-{1 + i // 24:02d}T{i % 24:02d}:00:00Z", float("nan"), 1000.0)
            for i in range(200)
        ]
        s = scoreability(recs, revision_class="clean")
        assert s["scoreable"] is False
        assert s["reason"] == EXCLUDE_DF_COVERAGE
        assert s["df_coverage"] == pytest.approx(0.5)

    def test_scoreable_ba_reports_coverage_stats(self):
        s = scoreability(_records(300), revision_class="bulk")
        assert s["scoreable"] is True
        assert s["reason"] is None
        assert s["df_coverage"] == 1.0

    def test_excluded_ba_never_enters_the_fleet_aggregate(self):
        good = compute_benchmark_payload(
            "GOOD", _records(300, df=1100.0), _horizon(_records(300), 1000.0), "clean"
        )
        bad = compute_benchmark_payload("BAD", _records(300), None, "broken")
        roll = fleet_rollup([good, bad])
        assert roll["n_scoreable"] == 1
        assert roll["n_excluded"] == 1
        assert roll["excluded"][0]["region"] == "BAD"
        assert roll["fleet"]["n"] == 1


class TestMetrics:
    def test_wape_weights_by_magnitude_unlike_mape(self):
        """One tiny hour with a big relative miss blows up MAPE but barely
        moves WAPE — which is why small BAs need both."""
        actual = np.array([1000.0, 1000.0, 1.0])
        pred = np.array([1000.0, 1000.0, 2.0])
        mape = float(np.mean(np.abs(actual - pred) / actual * 100))
        assert mape > 30  # dominated by the tiny hour
        assert wape(actual, pred) < 0.1

    def test_mae_and_n_reported(self):
        recs = _records(5, df=1100.0, actual=1000.0)
        arm = score_arm(pair_hours(recs, _gp(recs, 1000.0))[0], "official")
        assert arm["mae"] == pytest.approx(100.0)
        assert arm["n"] == 5


class TestLeadsAndPayload:
    def test_both_headline_and_conservative_leads_are_reported(self):
        """Ours is a fixed 24h lead; theirs is 17-41h. The 48h variant gives
        them the lead advantage and must be published alongside."""
        recs = _records(300, df=1100.0, actual=1000.0)
        payload = compute_benchmark_payload(
            "TEST", recs, _horizon(recs, 1000.0, leads=("24h", "48h")), "clean"
        )
        assert set(payload["leads"]) == {"24h", "48h"}
        assert payload["leads"]["24h"]["scoreable"]
        assert payload["leads"]["48h"]["scoreable"]

    def test_winner_and_delta_orientation(self):
        """Positive delta = GridPulse more accurate. A sign flip here would
        invert the public scorecard."""
        recs = _records(300, df=1200.0, actual=1000.0)  # official off by 20%
        payload = compute_benchmark_payload("TEST", recs, _horizon(recs, 1050.0), "clean")
        lead = payload["leads"][HEADLINE_LEAD]
        assert lead["winner"] == "gridpulse"
        assert lead["delta_mape"] == pytest.approx(15.0)

    def test_thin_sample_refuses_a_verdict(self):
        recs = _records(MIN_PAIRED_HOURS - 1, df=1100.0)
        payload = compute_benchmark_payload("TEST", recs, _horizon(recs, 1000.0), "clean")
        assert payload["leads"][HEADLINE_LEAD]["scoreable"] is False

    def test_excluded_payload_still_carries_its_reason(self):
        """Excluded BAs must be publishable, not silently absent."""
        payload = compute_benchmark_payload("TEST", _records(300), None, "broken")
        assert payload["scoreable"] is False
        assert payload["reason"] == EXCLUDE_BROKEN_FEED
        assert payload["reason_detail"]


class TestFleetRollup:
    def test_ercot_is_isolated_from_the_aggregate(self):
        payloads = [
            compute_benchmark_payload(
                r, _records(300, df=1100.0), _horizon(_records(300), 1000.0), "clean"
            )
            for r in ("ERCOT", "PJM", "MISO")
        ]
        roll = fleet_rollup(payloads)
        assert roll["fleet"]["n"] == 2
        assert "ERCOT" in roll["isolated"]

    def test_fleet_uses_medians_not_means(self):
        """One PSEI-class outlier (official ~47%) must not drag the fleet
        figure — a mean would make us look better than we are."""
        payloads = []
        for region, off_mw in (("A", 1030.0), ("B", 1040.0), ("C", 1500.0)):
            recs = _records(300, df=off_mw, actual=1000.0)
            payloads.append(
                compute_benchmark_payload(region, recs, _horizon(recs, 1020.0), "clean")
            )
        roll = fleet_rollup(payloads)
        # medians of (3%, 4%, 50%) = 4%; the mean would be ~19%
        assert roll["fleet"]["median_official_mape"] == pytest.approx(4.0, abs=0.2)

    def test_spread_ratio_is_the_consistency_story(self):
        """The durable claim: our spread is narrow, theirs is 41x."""
        payloads = []
        for region, off in (("A", 1010.0), ("B", 1500.0)):
            recs = _records(300, df=off, actual=1000.0)
            payloads.append(
                compute_benchmark_payload(region, recs, _horizon(recs, 1020.0), "clean")
            )
        roll = fleet_rollup(payloads)
        assert (
            roll["fleet"]["official_spread"]["ratio"] > roll["fleet"]["gridpulse_spread"]["ratio"]
        )


class TestHorizonExtraction:
    def test_reads_the_drift_records_wire_format(self):
        """Must parse the real serialize_records shape ('ts'/'p'), not a
        guessed one — a silent mismatch would empty every comparison."""
        payload = {
            "models": {
                "ensemble": {
                    "24h": {"records": [{"ts": "2026-07-01T00:00:00Z", "p": 950.0, "a": 1000.0}]}
                }
            }
        }
        got = gridpulse_predictions(payload, "ensemble", "24h")
        assert got == {"2026-07-01T00:00:00Z": 950.0}

    def test_missing_payload_is_empty_not_an_error(self):
        assert gridpulse_predictions(None, "ensemble", "24h") == {}
        assert gridpulse_predictions({}, "ensemble", "24h") == {}


def _horizon(records, predicted, leads=("24h",)):
    """Build a drift_horizon-shaped payload for the given records."""
    return {
        "models": {
            "ensemble": {
                lead: {
                    "records": [{"ts": r.timestamp, "p": predicted, "a": r.last_d} for r in records]
                }
                for lead in leads
            }
        }
    }


class TestDualOfficialArm:
    """EIA revises the day-ahead forecast for some BAs (PSEI 26%, SOCO 24%;
    the big ISOs never — docs/BENCHMARK_PROVENANCE.md). Scoring only the
    as-issued value invites "you graded their stale number", so the
    conservative arm grades them on the revised value, which carries
    hindsight. We publish both."""

    def test_revised_arm_scored_on_the_same_hours_and_truth(self):
        """The conservative arm must not quietly get a different hour set or
        a different actual — that is how a head-to-head gets rigged."""
        recs = _records(300, df=1200.0, actual=1000.0)
        revised = {r.timestamp: 1100.0 for r in recs}
        pairs, _ = pair_hours(recs, _gp(recs, 1050.0), revised)
        assert len(pairs) == 300
        assert all(p.actual == 1000.0 for p in pairs)
        assert all(p.official == 1200.0 and p.official_revised == 1100.0 for p in pairs)

    def test_revised_falls_back_to_as_issued_when_absent(self):
        """7 of 10 sampled BAs never revise — they must score identically."""
        recs = _records(300, df=1200.0, actual=1000.0)
        pairs, _ = pair_hours(recs, _gp(recs, 1050.0), None)
        assert all(p.official_revised == p.official for p in pairs)

    def test_payload_reports_both_verdicts(self):
        """A reader must be able to see the result under either scoring."""
        recs = _records(300, df=1200.0, actual=1000.0)
        revised = {r.timestamp: 1100.0 for r in recs}
        payload = compute_benchmark_payload(
            "TEST", recs, _horizon(recs, 1050.0), "clean", revised_df_by_ts=revised
        )
        lead = payload["leads"][HEADLINE_LEAD]
        assert lead["official"]["mape"] == pytest.approx(20.0)
        assert lead["official_revised"]["mape"] == pytest.approx(10.0)
        assert lead["winner"] == "gridpulse"
        assert lead["winner_vs_revised"] == "gridpulse"
        assert lead["delta_mape_vs_revised"] == pytest.approx(5.0)

    def test_revised_arm_can_flip_the_verdict_and_that_is_visible(self):
        """If the revision is large enough to lose us the BA, the payload
        says so rather than hiding behind the as-issued number."""
        recs = _records(300, df=1300.0, actual=1000.0)
        revised = {r.timestamp: 1010.0 for r in recs}  # revision makes them great
        payload = compute_benchmark_payload(
            "TEST", recs, _horizon(recs, 1050.0), "clean", revised_df_by_ts=revised
        )
        lead = payload["leads"][HEADLINE_LEAD]
        assert lead["winner"] == "gridpulse"
        assert lead["winner_vs_revised"] == "official"

    def test_revised_df_read_from_the_live_frame(self):
        """forecast_mw on the fetched frame IS EIA's post-revision DF, so no
        new capture is needed."""
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-07-01T00:00Z", "2026-07-01T01:00Z"]),
                "forecast_mw": [900.0, float("nan")],
            }
        )
        got = revised_df_from_frame(frame)
        assert got["2026-07-01T00:00:00Z"] == 900.0
        assert len(got) == 1  # the NaN row is skipped, not zero-filled

    def test_missing_frame_is_empty_not_an_error(self):
        assert revised_df_from_frame(None) == {}
        assert revised_df_from_frame(pd.DataFrame()) == {}


class TestObservedLead:
    """Our nominal 24h record is a realized ~22.9h (the forecast anchors on
    the last real demand hour). The conservative label on the 48h arm is
    EARNED by measurement, never assumed."""

    def _payload(self, lead_h):
        recs = _records(300, df=1100.0, actual=1000.0)
        return compute_benchmark_payload(
            "TEST",
            recs,
            _horizon(recs, 1000.0, leads=("24h", "48h")),
            "clean",
            observed_lead_h=lead_h,
        )

    def test_observed_lead_replaces_the_nominal_label(self):
        payload = self._payload({"24h": 22.92, "48h": 46.92})
        head = payload["leads"][HEADLINE_LEAD]
        assert head["lead_basis"] == "observed"
        assert head["observed_lead_h"] == pytest.approx(22.92)

    def test_conservative_label_granted_when_lead_exceeds_their_maximum(self):
        payload = self._payload({"24h": 22.92, "48h": 46.92})
        arm = payload["leads"][CONSERVATIVE_LEAD]
        assert arm["conservative"] is True
        assert str(int(OFFICIAL_DOCUMENTED_LEAD_H[1])) in arm["conservative_basis"]

    def test_conservative_label_withheld_when_lead_falls_short(self):
        """If EIA's publishing lag ever grew, our 48h arm would stop
        exceeding their documented 41h maximum — and the claim must lapse
        automatically rather than persist as a stale assertion."""
        payload = self._payload({"24h": 16.0, "48h": 40.0})
        arm = payload["leads"][CONSERVATIVE_LEAD]
        assert arm["conservative"] is False
        assert "withheld" in arm["conservative_basis"]

    def test_no_observation_falls_back_to_nominal(self):
        payload = self._payload({})
        assert payload["leads"][HEADLINE_LEAD]["lead_basis"] == "nominal"
        assert payload["leads"][CONSERVATIVE_LEAD]["conservative"] is False
