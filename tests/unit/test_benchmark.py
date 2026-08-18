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

import inspect

import numpy as np
import pandas as pd
import pytest

from models.benchmark import (
    CONSERVATIVE_LEAD,
    EXCLUDE_BROKEN_FEED,
    EXCLUDE_DF_FEED_STOPPED,
    HEADLINE_LEAD,
    MIN_DF_COVERAGE,
    MIN_PAIRED_HOURS,
    OFFICIAL_DOCUMENTED_LEAD_H,
    _normalize_ts,
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

    def __init__(
        self, ts, first_seen_df, last_d, was_placeholder=False, captured_at=None, df_at=None
    ):
        self.timestamp = ts
        self.first_seen_df = first_seen_df
        self.last_d = last_d
        self.was_placeholder = was_placeholder
        # Default FRESH (captured at the target hour). `captured_at` is a
        # required field on the real VintageRecord, so only this stub can omit
        # it — and an hour that was never seen fresh has no business on the
        # as-issued arm (#358), which the stale tests below exercise.
        self.captured_at = captured_at if captured_at is not None else ts
        # `None` = "no separate DF observation date", which is what every
        # pre-#535 record carries and what makes `df_capture_lag_hours` fall
        # back to `captured_at`. Set it explicitly to model a DF that arrived on
        # a LATER tick than the hour itself — the #535 case.
        self.df_at = df_at


def _records(n=300, *, df=1000.0, actual=1000.0, placeholder=False):
    return [
        _Rec(f"2026-07-{1 + i // 24:02d}T{i % 24:02d}:00:00Z", df, actual, placeholder)
        for i in range(n)
    ]


def _gp(records, value):
    return {r.timestamp: value for r in records}


def _tail_records(n_normal=190, n_tail=20, *, actual=1000.0, normal_df=1010.0, tail_df=1500.0):
    """A feed whose day-ahead forecast is excellent most hours and
    catastrophic on a few — the shape that separates median from mean."""
    return [
        _Rec(
            f"2026-07-{1 + i // 24:02d}T{i % 24:02d}:00:00Z",
            normal_df if i < n_normal else tail_df,
            actual,
            False,
        )
        for i in range(n_normal + n_tail)
    ]


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

    def test_a_feed_that_stopped_publishing_is_excluded(self):
        """SPP's shape: complete, then nothing. 200 absent hours at the end of
        the window is well past MAX_DF_STALENESS_HOURS, so the feed is dead and
        every hour we could score predates the stop."""
        recs = _records(200) + [
            _Rec(f"2026-09-{1 + i // 24:02d}T{i % 24:02d}:00:00Z", float("nan"), 1000.0)
            for i in range(200)
        ]
        s = scoreability(recs, revision_class="clean")
        assert s["scoreable"] is False
        assert s["reason"] == EXCLUDE_DF_FEED_STOPPED
        assert s["df_coverage"] == pytest.approx(0.5)

    def test_a_low_coverage_ba_with_a_live_feed_is_scoreable(self):
        """TEC's shape, and the whole of #549: 50% coverage in whole-day blocks
        with the feed still publishing. The old rate gate excluded this as "too
        sparse to score fairly" — a claim about shape it had never measured and
        which is false for every BA it was ever applied to."""
        recs = []
        for day in range(20):
            absent = day % 2 == 0  # alternating 24h blackout blocks
            for hour in range(24):
                ts = f"2026-07-{1 + day:02d}T{hour:02d}:00:00Z"
                recs.append(_Rec(ts, float("nan") if absent else 900.0, 1000.0))
        s = scoreability(recs, revision_class="clean")
        assert s["df_coverage"] == pytest.approx(0.5), "half the hours, same as SPP"
        assert s["scoreable"] is True, "but the feed is alive, so it is scoreable"
        assert s["reason"] is None
        assert s["reason_detail"] is None

    def test_the_gate_is_liveness_not_the_coverage_threshold(self):
        """Pinned so a later reader cannot mistake #549 for the threshold
        raise the issue explicitly ruled out. MIN_DF_COVERAGE still exists —
        it is quoted in the sentences and anchors the warning band — but no
        code path compares against it to decide who is published."""
        assert MIN_DF_COVERAGE == 0.80
        gate = inspect.getsource(scoreability)
        assert "MIN_DF_COVERAGE" not in gate, (
            "coverage decides nothing since #549; if it gates again the "
            "exclusion sentence starts asserting a shape it has not measured"
        )

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

    def test_median_ape_published_beside_the_mean(self):
        """The offline reports characterise BAs by MEDIAN APE while the live
        payload's headline is a MEAN. Publishing both is what stops a reader
        carrying a figure between artifacts and silently comparing two
        different statistics."""
        recs = _records(5, df=1100.0, actual=1000.0)
        arm = score_arm(pair_hours(recs, _gp(recs, 1000.0))[0], "official")
        assert arm["median_ape"] == pytest.approx(10.0)
        assert arm["mape"] == pytest.approx(10.0)

    def test_median_ape_separates_from_the_mean_on_a_tail_heavy_feed(self):
        """A handful of catastrophic hours drag the mean and leave the median
        alone. The gap is information, not noise: it says this BA's error is
        tail-driven rather than pervasive."""
        recs = _tail_records()
        arm = score_arm(pair_hours(recs, _gp(recs, 1000.0))[0], "official")
        assert arm["median_ape"] == pytest.approx(1.0)
        assert arm["mape"] == pytest.approx(5.667, abs=0.01)


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

    def test_metric_dependent_verdicts_are_visible_not_hidden(self):
        """The winner is decided on mean MAPE. When the median disagrees —
        because the official forecast is excellent most hours and terrible on
        a few — the payload must SAY so, so a reader can see the call is
        metric-dependent instead of taking the headline on trust."""
        recs = _tail_records()
        payload = compute_benchmark_payload("TEST", recs, _horizon(recs, 1030.0), "clean")
        lead = payload["leads"][HEADLINE_LEAD]

        assert lead["official"]["median_ape"] < lead["official"]["mape"]  # tail-driven
        assert lead["winner"] == "gridpulse"  # on the mean, we win
        assert lead["delta_mape"] > 0
        assert lead["delta_median_ape"] < 0  # on the median, they do — published

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


class TestObservedLeadProducer:
    """The *producer* of the observed lead — ``jobs.phases._observed_lead_hours``.

    ``TestObservedLead`` above injects the dict and so never exercised this;
    both bugs it now pins survived that gap. The consumer being well-tested
    is what made them invisible: a producer that returns ``{}`` looks
    exactly like a BA with no forecast yet.
    """

    @staticmethod
    def _redis_payload(n_rows: int = 168, offset_min: int = -6):
        """A payload shaped like ``gridpulse:forecast:{region}:1h``.

        ``offset_min`` is row 0 minus ``scored_at`` — negative because the
        forecast anchors on the last settled demand hour, which is behind
        wall-clock by EIA's publishing lag.
        """
        scored_at = pd.Timestamp("2026-07-27T12:00:00+00:00")
        origin = scored_at + pd.Timedelta(minutes=offset_min)
        origin = origin.floor("h") if offset_min == 0 else origin
        return {
            "region": "TEST",
            "scored_at": scored_at.isoformat(),
            "granularity": "1h",
            "forecasts": [
                {
                    "timestamp": (origin + pd.Timedelta(hours=i)).isoformat(),
                    "predicted_demand_mw": 1000.0 + i,
                    "ensemble": 1000.0 + i,
                }
                for i in range(n_rows)
            ],
        }

    def test_reads_the_redis_payload_key(self):
        """The rows live under ``forecasts``; the API's ``forecast`` is a
        reshape. Reading the API name off the Redis payload returned no
        observation at all — the label silently stayed nominal fleet-wide."""
        from jobs import phases

        out = phases._observed_lead_hours(self._redis_payload())
        assert set(out) == {"24h", "48h"}, "no observation from the real payload shape"
        assert out["24h"] == pytest.approx(23.9, abs=0.01)
        assert out["48h"] == pytest.approx(47.9, abs=0.01)

    def test_measures_the_hour_the_benchmark_actually_scores(self):
        """The lead must describe the SAME target hour that
        ``snapshot_horizon_predictions`` snapshots and the benchmark later
        grades — row 0 + H. Measuring row index H−1 understated it by 1h."""
        from jobs import phases
        from models.drift import snapshot_horizon_predictions

        payload = self._redis_payload()
        made = pd.Timestamp(payload["scored_at"])
        observed = phases._observed_lead_hours(payload)

        snaps = {s["horizon"]: s["target_ts"] for s in snapshot_horizon_predictions(payload)}
        for label, lead_h in observed.items():
            target = pd.Timestamp(snaps[label])
            expected = (target - made).total_seconds() / 3600.0
            assert lead_h == pytest.approx(expected, abs=1e-6), (
                f"{label}: lead describes a different hour than the one scored"
            )

    def test_lead_hours_agree_with_the_drift_horizon_definition(self):
        """The local mapping is a copy of the drift module's; a divergence
        must fail here rather than degrade to a wrong lead at runtime."""
        from jobs import phases
        from models.drift import _HORIZON_HOURS

        for label, hours in phases._BENCHMARK_LEAD_HOURS.items():
            assert _HORIZON_HOURS[label] == hours

    def test_no_rows_or_no_scored_at_is_no_observation(self):
        """Absence must read as "not measured" (nominal basis), never as a
        lead of zero — a zero would withhold nothing, it would publish a
        wrong number."""
        from jobs import phases

        assert phases._observed_lead_hours(None) == {}
        assert phases._observed_lead_hours({"forecasts": []}) == {}
        assert phases._observed_lead_hours({"forecasts": [{"timestamp": "x"}]}) == {}
        payload = self._redis_payload()
        payload.pop("scored_at")
        assert phases._observed_lead_hours(payload) == {}

    def test_observed_lead_shrinks_when_eia_publishing_lag_grows(self):
        """The whole point of measuring per tick: if EIA fell further behind,
        the realized lead drops and the conservative label must lapse."""
        from jobs import phases

        normal = phases._observed_lead_hours(self._redis_payload(offset_min=-6))
        lagged = phases._observed_lead_hours(self._redis_payload(offset_min=-8 * 60))
        assert lagged["48h"] < normal["48h"]
        assert lagged["48h"] == pytest.approx(40.0, abs=0.01)
        assert lagged["48h"] < OFFICIAL_DOCUMENTED_LEAD_H[1]


def test_official_documented_lead_is_declared_exactly_once():
    """docs/BENCHMARK_METHODOLOGY.md §12.2 tells readers this constant lives
    in one place. It briefly did not — the provenance probe carried its own
    literal, so the bar the conservative label is judged against could have
    drifted from the engine's silently. Keep the doc's claim true."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    pattern = re.compile(r"^OFFICIAL_DOCUMENTED_LEAD_H\s*=", re.MULTILINE)
    declarations = [
        path.relative_to(root).as_posix()
        for folder in ("models", "scripts", "jobs", "components", "data")
        for path in (root / folder).rglob("*.py")
        if pattern.search(path.read_text())
    ]
    assert declarations == ["models/benchmark.py"], (
        f"expected a single declaration, found {declarations}"
    )


class TestStaleCapture:
    """#358 — a backfilled hour cannot supply an "as-issued" forecast.

    The official arm is documented throughout as *"the earliest day-ahead
    forecast we observed"*. For an hour first seen long after it passed — the
    first tick's ~700-hour backfill, or any reseed — `first_seen_df` is a
    POST-revision value. Scoring it on the as-issued arm silently collapses
    the as-issued/as-revised distinction the dual arm exists to draw.
    """

    def _stale(self, hours_late: float, n=300):
        """Records whose capture lag exceeds the freshness threshold."""
        from datetime import datetime, timedelta

        out = []
        for i in range(n):
            ts = f"2026-07-{1 + i // 24:02d}T{i % 24:02d}:00:00Z"
            cap = datetime.fromisoformat(ts.replace("Z", "+00:00")) + timedelta(hours=hours_late)
            out.append(_Rec(ts, 1000.0, 1100.0, captured_at=cap.isoformat()))
        return out

    def test_backfilled_hours_are_dropped_and_counted(self):
        from data.vintage import FRESH_CAPTURE_LAG_HOURS

        recs = self._stale(FRESH_CAPTURE_LAG_HOURS + 24)
        pairs, drops = pair_hours(recs, {_normalize_ts(r.timestamp): 1050.0 for r in recs})
        assert pairs == []
        assert drops["stale_capture"] == len(recs)

    def test_fresh_hours_survive(self):
        from data.vintage import FRESH_CAPTURE_LAG_HOURS

        recs = self._stale(FRESH_CAPTURE_LAG_HOURS)  # exactly at the boundary
        pairs, drops = pair_hours(recs, {_normalize_ts(r.timestamp): 1050.0 for r in recs})
        assert drops["stale_capture"] == 0
        assert len(pairs) == len(recs)

    def test_the_boundary_is_inclusive_not_exclusive(self):
        """At exactly FRESH_CAPTURE_LAG_HOURS the record is fresh; one hour
        later it is not. Pinned because an off-by-one here silently changes
        every published drop count."""
        from data.vintage import FRESH_CAPTURE_LAG_HOURS

        at = self._stale(FRESH_CAPTURE_LAG_HOURS, n=24)
        past = self._stale(FRESH_CAPTURE_LAG_HOURS + 1, n=24)
        gp = {_normalize_ts(r.timestamp): 1050.0 for r in at}
        assert pair_hours(at, gp)[1]["stale_capture"] == 0
        assert pair_hours(past, gp)[1]["stale_capture"] == 24

    def test_an_unmeasurable_lag_counts_as_stale(self):
        """A record whose capture time will not parse cannot be SHOWN fresh,
        and the official arm's whole claim is freshness — so it is dropped
        rather than waved through."""
        recs = [_Rec("2026-07-01T00:00:00Z", 1000.0, 1100.0, captured_at="not-a-timestamp")]
        pairs, drops = pair_hours(recs, {_normalize_ts(recs[0].timestamp): 1050.0})
        assert pairs == []
        assert drops["stale_capture"] == 1

    def test_the_filter_can_be_disabled_for_impact_measurement(self):
        """`exclude_stale_capture=False` reproduces the pre-#358 scoring, which
        is how the payload measures the direction this change moves our own
        number (methodology §14) rather than predicting it."""
        recs = self._stale(48)
        gp = {_normalize_ts(r.timestamp): 1050.0 for r in recs}
        pairs, drops = pair_hours(recs, gp, exclude_stale_capture=False)
        assert drops["stale_capture"] == 0
        assert len(pairs) == len(recs)

    def test_provenance_is_checked_before_the_stub_rules(self):
        """Evaluation order matters — the counts are disjoint, so an hour that
        is BOTH backfilled and a stub lands in exactly one bucket. Provenance
        wins: if the value was never as-issued, whether it later equalled the
        settled figure is moot."""
        recs = self._stale(48, n=24)
        for r in recs:
            r.last_d = r.first_seen_df  # also an unresolved stub
        pairs, drops = pair_hours(recs, {_normalize_ts(r.timestamp): 1050.0 for r in recs})
        assert drops["stale_capture"] == 24
        assert drops["unresolved_stub"] == 0

    def test_capture_lag_has_exactly_one_definition(self):
        """The OFFICIAL_DOCUMENTED_LEAD_H lesson: a second implementation of
        capture lag would drift from the classifier's. `models.benchmark`
        imports the one in `data.vintage` rather than recomputing it."""
        import ast
        from pathlib import Path

        src = Path("models/benchmark.py").read_text()
        assert "from data.vintage import" in src and "capture_lag_hours" in src
        tree = ast.parse(src)
        defined = [
            n.name
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and "capture_lag" in n.name
        ]
        assert defined == [], f"benchmark.py redefines capture lag: {defined}"


class TestStaleCaptureImpact:
    """§14 requires stating which direction a rule change moves our number.

    The payload measures it rather than predicting it, because the direction
    is NOT uniform: the issue records SOCO improving 1.84% → 1.76% under
    revisions and FMPP worsening 28.36% → 28.46%. A single fleet-level claim
    would be wrong for roughly half the fleet.
    """

    def _mixed(self, n_fresh=200, n_stale=100):
        from datetime import datetime, timedelta

        out = []
        for i in range(n_fresh + n_stale):
            ts = f"2026-07-{1 + i // 24:02d}T{i % 24:02d}:00:00Z"
            base = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            stale = i >= n_fresh
            # Stale hours carry a different official value AND a different
            # ERROR MAGNITUDE. A first pass used 1200 against a fresh 1000 with
            # last_d 1100 — both |error| = 100, so the arm's MAPE was
            # identical and the test asserted nothing. 1400 makes the stale
            # hours genuinely worse, which is what the exclusion must move.
            out.append(
                _Rec(
                    ts,
                    1400.0 if stale else 1000.0,
                    1100.0,
                    captured_at=(base + timedelta(hours=48 if stale else 0)).isoformat(),
                )
            )
        return out

    def test_impact_block_is_produced_with_real_numbers(self):
        """Added after mutation testing: forcing the helper to always return
        None passed the whole suite, because the original test exercised
        `pair_hours` and never asserted the block itself."""
        from models.benchmark import _stale_capture_impact

        recs = self._mixed()
        gp = {_normalize_ts(r.timestamp): 1050.0 for r in recs}
        fresh_pairs, drops = pair_hours(recs, gp, None)
        assert drops["stale_capture"] == 100

        impact = _stale_capture_impact(
            recs,
            gp,
            None,
            score_arm(fresh_pairs, "gridpulse"),
            score_arm(fresh_pairs, "official"),
        )
        assert impact is not None, "the block must be published when hours were excluded"
        assert impact["n_hours_excluded"] == 100
        # The stale hours carried a far worse official value (1400 vs a
        # settled 1100), so removing them must IMPROVE the official arm —
        # a positive shift under the documented sign convention.
        assert impact["official_mape_shift_pts"] > 0
        assert "positive means the exclusion improved" in impact["note"]

    def test_the_impact_sign_convention_is_documented_and_correct(self):
        """A flipped sign here would tell a reader the exclusion hurt us when
        it helped — on a page whose argument is that its numbers mean what
        they say."""
        from models.benchmark import _stale_capture_impact

        recs = self._mixed()
        gp = {_normalize_ts(r.timestamp): 1050.0 for r in recs}
        fresh_pairs, _ = pair_hours(recs, gp, None)
        all_pairs, _ = pair_hours(recs, gp, None, exclude_stale_capture=False)
        impact = _stale_capture_impact(
            recs, gp, None, score_arm(fresh_pairs, "gridpulse"), score_arm(fresh_pairs, "official")
        )
        expected = (
            score_arm(all_pairs, "official")["mape"] - score_arm(fresh_pairs, "official")["mape"]
        )
        assert impact["official_mape_shift_pts"] == pytest.approx(expected, abs=0.001)

    def test_excluding_stale_hours_changes_the_official_arm(self):
        """The whole point: those hours carried post-revision values on the
        as-issued arm, so removing them must move that arm's score."""
        recs = self._mixed()
        gp = {_normalize_ts(r.timestamp): 1050.0 for r in recs}
        all_pairs, _ = pair_hours(recs, gp, None, exclude_stale_capture=False)
        fresh_pairs, _ = pair_hours(recs, gp, None)
        assert (
            score_arm(all_pairs, "official")["mape"] != score_arm(fresh_pairs, "official")["mape"]
        )

    def test_no_impact_block_when_nothing_was_excluded(self):
        """A window that has rolled past its seed excludes nothing, and that
        is worth distinguishing from a measured zero."""
        from models.benchmark import _stale_capture_impact

        recs = [
            _Rec(f"2026-07-{1 + i // 24:02d}T{i % 24:02d}:00:00Z", 1000.0, 1100.0)
            for i in range(300)
        ]
        gp = {_normalize_ts(r.timestamp): 1050.0 for r in recs}
        pairs, _ = pair_hours(recs, gp, None)
        assert (
            _stale_capture_impact(
                recs, gp, None, score_arm(pairs, "gridpulse"), score_arm(pairs, "official")
            )
            is None
        )


class TestDeferredDayAheadCapture:
    """The #535 rail: filling a DF late must not buy it the as-issued arm.

    `data.vintage` now gives a missing `DF` a second look, because pinning it at
    first sight measured our collector and published the result as a fact about
    the BA. That fix is only safe while `pair_hours` grades freshness on the DF
    observation (`df_at`) rather than on the hour's own capture — otherwise the
    fix quietly puts post-revision values on the as-issued arm at scale, which
    is a worse defect than the one it repairs.
    """

    def test_late_filled_df_cannot_reach_the_asissued_arm(self):
        # Hour seen fresh; its DF only turned up 9h later.
        recs = [
            _Rec(
                "2026-07-01T00:00:00Z",
                900.0,
                1000.0,
                captured_at="2026-07-01T01:00:00Z",
                df_at="2026-07-01T09:00:00Z",
            )
        ]
        pairs, drops = pair_hours(recs, {_normalize_ts(recs[0].timestamp): 1050.0})
        assert pairs == [], "a DF observed 9h after the hour is not 'as issued'"
        assert drops["stale_capture"] == 1
        assert drops["no_df"] == 0, "the DF exists — it is its provenance that fails"

    def test_a_promptly_filled_df_still_scores(self):
        """The rail is a freshness test, not a ban on ever filling a DF.

        If it rejected every filled value the #535 fix would restore the
        exclusion labels and no scoreable hours, which is the failure mode worth
        distinguishing from success.
        """
        recs = [
            _Rec(
                "2026-07-01T00:00:00Z",
                900.0,
                1000.0,
                captured_at="2026-07-01T01:00:00Z",
                df_at="2026-07-01T02:00:00Z",
            )
        ]
        pairs, drops = pair_hours(recs, {_normalize_ts(recs[0].timestamp): 1050.0})
        assert len(pairs) == 1 and drops["stale_capture"] == 0

    def test_the_rail_reads_df_at_and_not_captured_at(self):
        """Pins the direction, which a symmetric fixture cannot.

        Here the HOUR is stale and the DF is fresh — impossible in production,
        and chosen precisely because the two clocks disagree: a `pair_hours`
        still grading on `captured_at` drops this row, and one grading on
        `df_at` keeps it. Nothing else in the suite separates them.
        """
        recs = [
            _Rec(
                "2026-07-01T00:00:00Z",
                900.0,
                1000.0,
                captured_at="2026-07-02T00:00:00Z",  # 24h — stale by the old rule
                df_at="2026-07-01T01:00:00Z",  # 1h — fresh by the new one
            )
        ]
        pairs, _ = pair_hours(recs, {_normalize_ts(recs[0].timestamp): 1050.0})
        assert len(pairs) == 1, "the as-issued claim is about the DF observation"


class TestFeedLivenessGate:
    """#549: the gate measures whether the feed is alive, not how often it fires.

    Every constant here was fitted to the fleet on 2026-08-18 rather than
    chosen: hours since the newest published DF were SPP 341, TEC 30, and every
    other BA at most 6.
    """

    @staticmethod
    def _rec(hour, *, df, d=1000.0):
        ts = f"2026-07-{1 + hour // 24:02d}T{hour % 24:02d}:00:00Z"
        return _Rec(ts, df, d, captured_at=ts, df_at=ts)

    def test_staleness_is_measured_from_the_newest_record_not_wall_clock(self):
        """Pure function: replaying an old window must report what that window
        saw, not how long ago the window itself was."""
        recs = [self._rec(h, df=900.0) for h in range(100)]
        out = scoreability(recs, "clean")
        assert out["df_stale_hours"] == 0.0
        assert out["scoreable"] is True

    def test_an_unsorted_caller_cannot_change_the_verdict(self):
        """The gate takes a max, not the last element. Callers sort today; a
        gate that silently depends on that is one refactor from reading a
        different number."""
        recs = [self._rec(h, df=900.0) for h in range(100)]
        assert scoreability(list(reversed(recs)), "clean") == scoreability(recs, "clean")

    def test_a_ba_with_no_day_ahead_forecast_at_all_is_excluded(self):
        """`stale_hours is None` is the strongest form of the condition, not an
        absence of evidence about it — it must not fall through as fresh."""
        out = scoreability([self._rec(h, df=float("nan")) for h in range(300)], "clean")
        assert out["scoreable"] is False
        assert out["reason"] == EXCLUDE_DF_FEED_STOPPED
        assert out["df_stale_hours"] is None
        assert "no day-ahead forecast for this BA at all" in out["reason_detail"]

    def test_an_unparseable_timestamp_is_skipped_not_defaulted(self):
        """One bad row must not decide whether a BA is published."""
        recs = [self._rec(h, df=900.0) for h in range(100)]
        recs.append(_Rec("not-a-timestamp", 900.0, 1000.0))
        out = scoreability(recs, "clean")
        assert out["scoreable"] is True
        assert out["df_stale_hours"] == 0.0


class TestAbsentHourBias:
    """The hazard the coverage rate was only ever a proxy for, measured (#549).

    Published; gates nothing. Promoting it needs a disqualifying magnitude and
    the fleet has not produced one to calibrate against.
    """

    @staticmethod
    def _rec(hour, *, df, d):
        ts = f"2026-07-{1 + hour // 24:02d}T{hour % 24:02d}:00:00Z"
        return _Rec(ts, df, d, captured_at=ts, df_at=ts)

    def test_no_absent_hours_means_no_bias_is_possible(self):
        """Distinct from unmeasurable, and must not read as a failure: full
        coverage is the case where the question cannot arise."""
        out = scoreability([self._rec(h, df=900.0, d=1000.0) for h in range(100)], "clean")
        assert out["absent_hours_bias_pct"] is None
        assert out["n_absent_hours"] == 0
        assert out["scoreable"] is True

    def test_too_few_absent_hours_is_reported_as_unknown_not_as_a_number(self):
        """PACE, NWMT and IPCO each showed a ~-20% apparent skew off 3-4 absent
        hours. That is which hours were missing, not a property of the BA."""
        recs = [self._rec(h, df=900.0, d=1000.0) for h in range(100)]
        recs += [self._rec(h, df=float("nan"), d=500.0) for h in range(100, 104)]
        assert scoreability(recs, "clean")["absent_hours_bias_pct"] is None

    def test_a_real_skew_is_measured_and_signed(self):
        recs = [self._rec(h, df=900.0, d=1000.0) for h in range(100)]
        recs += [self._rec(h, df=float("nan"), d=1200.0) for h in range(100, 140)]
        assert scoreability(recs, "clean")["absent_hours_bias_pct"] == pytest.approx(20.0)

    def test_one_bad_row_cannot_move_the_statistic(self):
        """Non-finite and non-positive D are dropped from BOTH sides."""
        recs = [self._rec(h, df=900.0, d=1000.0) for h in range(100)]
        recs += [self._rec(h, df=float("nan"), d=1000.0) for h in range(100, 140)]
        clean = scoreability(recs, "clean")["absent_hours_bias_pct"]
        recs.append(self._rec(200, df=float("nan"), d=float("nan")))
        recs.append(self._rec(201, df=float("nan"), d=-5.0))
        assert scoreability(recs, "clean")["absent_hours_bias_pct"] == clean


class TestTwoCoverages:
    """`df_coverage` describes the BA; `df_asissued_coverage` describes us (#535).

    One number used to answer both questions, and the answer it gave was
    published as the BA's. Twenty-six BAs were excluded on it — measured against
    EIA directly, exactly one of them (SPP) was genuinely below the threshold.
    """

    @staticmethod
    def _rec(hour, *, df, df_at):
        ts = f"2026-07-{1 + hour // 24:02d}T{hour % 24:02d}:00:00Z"
        return _Rec(ts, df, 1000.0, captured_at=ts, df_at=df_at)

    def test_a_ba_that_publishes_but_we_captured_late_is_still_scoreable(self):
        """The whole of #535 in one assertion."""
        recs = [
            self._rec(h, df=900.0, df_at=f"2026-07-{1 + h // 24:02d}T{h % 24:02d}:00:00Z")
            for h in range(60)
        ] + [
            # Published by EIA, captured by us a day late.
            self._rec(h, df=900.0, df_at="2026-08-01T00:00:00Z")
            for h in range(60, 100)
        ]
        out = scoreability(recs, "bulk")
        assert out["df_coverage"] == 1.0, "EIA published for every hour"
        assert out["df_asissued_coverage"] == 0.6, "we captured 60% of them in time"
        assert out["scoreable"] is True
        assert out["reason"] is None

    def test_spp_is_still_excluded_but_for_the_reason_that_is_true(self):
        """SPP stays out. Restoring it would be the regression, not the win —
        a bigger number is not the goal.

        The premise this test used to carry was wrong, and it is worth stating
        why (#549). It described SPP as a BA that "genuinely does not publish",
        which is what its 52.6% coverage looked like from the rate alone.
        Measured, SPP's absence is ONE contiguous 341-hour block — it published
        completely until 2026-08-04T06Z and then stopped, confirmed against EIA.
        The exclusion survives; the reason it survives on is the one that is
        actually true of it.
        """
        recs = [self._rec(h, df=900.0, df_at=None) for h in range(200)] + [
            self._rec(h, df=float("nan"), df_at=None) for h in range(200, 400)
        ]
        out = scoreability(recs, "bulk")
        assert out["df_coverage"] == 0.5
        assert out["scoreable"] is False
        assert out["reason"] == "df-feed-stopped"
        assert out["df_stale_hours"] > 168.0
        detail = out["reason_detail"]
        assert "stopped publishing" in detail
        assert "sparse" not in detail, "the false clause must not come back"

    def test_the_exclusion_reason_carries_this_bas_measured_numbers(self):
        """A reader who sees only a rule cannot tell 79% from 39%, and cannot
        tell a non-publishing BA from a capture gap at all — which is exactly
        how #535 stayed invisible for three weeks. Since #549 the sentence also
        carries the hour the feed stopped, which is checkable against EIA in a
        single query in a way that "sparse" never was."""
        recs = [self._rec(h, df=900.0, df_at=None) for h in range(156)] + [
            self._rec(h, df=float("nan"), df_at=None) for h in range(156, 400)
        ]
        out = scoreability(recs, "bulk")
        detail = out["reason_detail"]
        assert "39.0%" in detail, f"the BA's measured coverage is missing from: {detail}"
        assert "400 hours" in detail
        assert "as-issued" in detail
        assert out["df_last_published_at"] in detail, "the stop date must be named"

    def test_as_issued_coverage_never_gates(self):
        """It is our number, published for interpretation. A BA must not be
        excluded for our collector's behaviour — that is the bug."""
        recs = [self._rec(h, df=900.0, df_at="2026-09-01T00:00:00Z") for h in range(100)]
        out = scoreability(recs, "bulk")
        assert out["df_asissued_coverage"] == 0.0
        assert out["df_coverage"] == 1.0
        assert out["scoreable"] is True
