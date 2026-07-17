"""Demand-vintage capture (#309).

Fixtures are the REAL readings measured against the EIA API on 2026-07-15, so
these fail if the recorder would miss what actually happens in production:

    LDWP  01:00  first seen 1199 (26% of its own day-ahead), settles ~5200
    AZPS  01:00  first seen 1157 -> revised to 7815 in FOUR MINUTES
    IID          stuck at 339 for 6+ hours (~33% of day-ahead)
    BPAT  02:00  D == DF == 9008 exactly — a placeholder, not a measurement
    PNM   02:00  D == DF == 2153 exactly — placeholder, yet PNM revises 0.7%

The last two are why ``was_placeholder`` is recorded rather than acted on: the
placeholder is real, but it does NOT explain the revisions (corr 0.15), and
removing it makes the forecast worse (6.55% -> 7.72%). This module exists to
settle that with data instead of a fourth hypothesis.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from data.vintage import (
    VintageRecord,
    canonical_hour,
    classify_region,
    deserialize_records,
    serialize_records,
    summarize,
    update_vintage_records,
)

NOW = datetime(2026, 7, 16, 2, 0, 0, tzinfo=UTC)


def _frame(rows: list[tuple[int, float | None, float | None]]) -> pd.DataFrame:
    """``[(hours_ago, demand_mw, forecast_mw)]`` -> a demand frame."""
    return pd.DataFrame(
        {
            "timestamp": [NOW - timedelta(hours=h) for h, _, _ in rows],
            "demand_mw": [d for _, d, _ in rows],
            "forecast_mw": [f for _, _, f in rows],
        }
    )


def _hour(hours_ago: int) -> str:
    return (NOW - timedelta(hours=hours_ago)).isoformat()


class TestFirstSeenIsImmutable:
    """The one way this module can fail silently and destroy the study."""

    def test_azps_revision_preserves_the_anchors_seed(self):
        """AZPS 01:00 read 1157, then 7815 four minutes later. The record must
        keep 1157 — that is the number the anchor actually used."""
        records = update_vintage_records([], _frame([(1, 1157.0, 7911.0)]), now=NOW)
        assert records[0].first_seen_d == 1157.0

        later = update_vintage_records(records, _frame([(1, 7815.0, 7911.0)]), now=NOW)

        assert later[0].first_seen_d == 1157.0, "first_seen_d was overwritten by the revision"
        assert later[0].last_d == 7815.0
        assert later[0].n_updates == 1
        assert later[0].revision_pct == pytest.approx(85.2, abs=0.1)

    def test_repeated_revisions_accumulate(self):
        records = update_vintage_records([], _frame([(1, 1000.0, None)]), now=NOW)
        for value in (2000.0, 3000.0, 4000.0):
            records = update_vintage_records(records, _frame([(1, value, None)]), now=NOW)
        assert records[0].first_seen_d == 1000.0
        assert records[0].last_d == 4000.0
        assert records[0].n_updates == 3

    def test_unchanged_reading_does_not_count_as_a_revision(self):
        """IID sits at 339 for hours. Re-seeing the same value is not a
        revision — otherwise n_updates would just count scoring ticks."""
        records = update_vintage_records([], _frame([(1, 339.0, 1031.0)]), now=NOW)
        for _ in range(6):
            records = update_vintage_records(records, _frame([(1, 339.0, 1031.0)]), now=NOW)
        assert records[0].n_updates == 0
        assert records[0].last_d == 339.0

    def test_sub_epsilon_jitter_is_not_a_revision(self):
        records = update_vintage_records([], _frame([(1, 8000.0, None)]), now=NOW)
        records = update_vintage_records(records, _frame([(1, 8000.2, None)]), now=NOW)
        assert records[0].n_updates == 0
        assert records[0].last_d == 8000.0

    def test_captured_at_is_pinned_to_first_sight(self):
        records = update_vintage_records([], _frame([(1, 1157.0, None)]), now=NOW)
        first_at = records[0].captured_at
        later = update_vintage_records(
            records, _frame([(1, 7815.0, None)]), now=NOW + timedelta(hours=3)
        )
        assert later[0].captured_at == first_at


class TestPlaceholderFingerprint:
    """``D == DF`` exactly — recorded, deliberately not acted on."""

    def test_bpat_placeholder_detected(self):
        """BPAT 02:00: D == DF == 9008. Metered demand does not land on a
        day-ahead forecast to the megawatt."""
        records = update_vintage_records([], _frame([(0, 9008.0, 9008.0)]), now=NOW)
        assert records[0].was_placeholder is True

    def test_pnm_placeholder_detected_even_though_pnm_is_clean(self):
        """PNM's newest hour IS a placeholder (2153 == 2153) yet PNM revises
        0.7%. This pairing is exactly what refuted the placeholder hypothesis
        (corr 0.15) — the recorder must capture it, not editorialise."""
        records = update_vintage_records([], _frame([(0, 2153.0, 2153.0)]), now=NOW)
        assert records[0].was_placeholder is True

    def test_measured_reading_is_not_a_placeholder(self):
        """PJM: D=144597, DF=140940 — close, but measured."""
        records = update_vintage_records([], _frame([(0, 144597.0, 140940.0)]), now=NOW)
        assert records[0].was_placeholder is False

    def test_near_miss_is_not_a_placeholder(self):
        """One MW apart is a measurement that happens to be close. Only exact
        equality is the fingerprint — a tolerance would swallow real readings."""
        records = update_vintage_records([], _frame([(0, 9008.0, 9009.0)]), now=NOW)
        assert records[0].was_placeholder is False

    def test_absent_day_ahead_is_not_a_placeholder(self):
        records = update_vintage_records([], _frame([(0, 9008.0, None)]), now=NOW)
        assert np.isnan(records[0].first_seen_df)
        assert records[0].was_placeholder is False

    def test_placeholder_flag_survives_later_revision(self):
        """The seed was a forecast even after the real value lands — that fact
        is the join key for the whole study."""
        records = update_vintage_records([], _frame([(1, 9008.0, 9008.0)]), now=NOW)
        records = update_vintage_records(records, _frame([(1, 8700.0, 9008.0)]), now=NOW)
        assert records[0].was_placeholder is True
        assert records[0].last_d == 8700.0


class TestRevisionPct:
    def test_ldwp_partial_reading(self):
        """LDWP 01:00 came in at 1199 and settles ~5200."""
        rec = VintageRecord(
            timestamp=_hour(1),
            first_seen_d=1199.0,
            first_seen_df=4681.0,
            captured_at=_hour(1),
            last_d=5200.0,
            n_updates=1,
        )
        assert rec.revision_pct == pytest.approx(76.94, abs=0.01)

    def test_no_revision_is_zero_not_none(self):
        rec = VintageRecord(
            timestamp=_hour(1),
            first_seen_d=8000.0,
            first_seen_df=float("nan"),
            captured_at=_hour(1),
            last_d=8000.0,
        )
        assert rec.revision_pct == 0.0

    def test_unusable_settled_value_returns_none(self):
        """None means 'cannot judge' — never silently 0% (= agreement)."""
        for bad in (0.0, -5.0, float("nan")):
            rec = VintageRecord(
                timestamp=_hour(1),
                first_seen_d=8000.0,
                first_seen_df=float("nan"),
                captured_at=_hour(1),
                last_d=bad,
            )
            assert rec.revision_pct is None


class TestWindowing:
    def test_ages_out_beyond_the_window(self):
        old = update_vintage_records([], _frame([(24 * 40, 8000.0, None)]), now=NOW)
        assert old == []

    def test_keeps_records_inside_the_window(self):
        recs = update_vintage_records([], _frame([(24 * 29, 8000.0, None)]), now=NOW)
        assert len(recs) == 1

    def test_existing_records_are_trimmed_too(self):
        """Trimming must apply to the accumulated series, not just new rows —
        otherwise the payload grows without bound."""
        stale = [
            VintageRecord(
                timestamp=_hour(24 * 40),
                first_seen_d=8000.0,
                first_seen_df=float("nan"),
                captured_at=_hour(24 * 40),
                last_d=8000.0,
            )
        ]
        assert update_vintage_records(stale, _frame([(1, 8000.0, None)]), now=NOW) != stale
        assert len(update_vintage_records(stale, _frame([(1, 8000.0, None)]), now=NOW)) == 1

    def test_output_is_chronological(self):
        recs = update_vintage_records([], _frame([(3, 1.0, None), (1, 2.0, None)]), now=NOW)
        assert [r.timestamp for r in recs] == sorted(r.timestamp for r in recs)


class TestUnusableReadingsAreSkipped:
    """The anchor never sees these, so recording them would log a reading the
    forecast never used (``jobs/phases.py:209`` drops NaN; ``> 0`` mask at
    ``:741``)."""

    def test_skips_nan_zero_and_negative(self):
        frame = _frame([(1, float("nan"), 100.0), (2, 0.0, 100.0), (3, -50.0, 100.0)])
        assert update_vintage_records([], frame, now=NOW) == []

    def test_tidc_zero_is_skipped(self):
        """TIDC published D=0. eia_client coerces 0 -> NaN upstream; this is
        the belt-and-braces for a frame that reaches us unfiltered."""
        assert update_vintage_records([], _frame([(0, 0.0, 800.0)]), now=NOW) == []

    def test_a_gap_does_not_erase_an_existing_record(self):
        """EIA dropping an hour it already published must not lose first_seen."""
        recs = update_vintage_records([], _frame([(1, 1157.0, None)]), now=NOW)
        after = update_vintage_records(recs, _frame([(1, float("nan"), None)]), now=NOW)
        assert after[0].first_seen_d == 1157.0
        assert after[0].n_updates == 0


class TestDegenerateFrames:
    """Capture must never fail a scoring run."""

    def test_safe_inputs(self):
        assert update_vintage_records([], None, now=NOW) == []
        assert update_vintage_records([], pd.DataFrame(), now=NOW) == []
        assert update_vintage_records([], pd.DataFrame({"nope": [1]}), now=NOW) == []

    def test_missing_forecast_column_is_tolerated(self):
        frame = pd.DataFrame({"timestamp": [NOW], "demand_mw": [8000.0]})
        recs = update_vintage_records([], frame, now=NOW)
        assert len(recs) == 1
        assert np.isnan(recs[0].first_seen_df)

    def test_unparseable_timestamp_is_dropped_not_raised(self):
        frame = pd.DataFrame(
            {"timestamp": ["not-a-date"], "demand_mw": [8000.0], "forecast_mw": [1.0]}
        )
        assert update_vintage_records([], frame, now=NOW) == []


class TestRoundTrip:
    def test_serialize_deserialize_preserves_everything(self):
        recs = update_vintage_records([], _frame([(1, 1157.0, 7911.0)]), now=NOW)
        recs = update_vintage_records(recs, _frame([(1, 7815.0, 7911.0)]), now=NOW)
        back = deserialize_records(serialize_records(recs))
        assert back == recs

    def test_placeholder_equality_survives_the_round_trip(self):
        """d and df must round identically or the exact-equality fingerprint
        breaks in Redis but not in tests — the worst possible failure."""
        recs = update_vintage_records([], _frame([(0, 9008.126, 9008.126)]), now=NOW)
        assert deserialize_records(serialize_records(recs))[0].was_placeholder is True

    def test_nan_forecast_is_omitted_from_json_not_written_as_nan(self):
        """JSON has no NaN; a literal would make the payload unparseable."""
        recs = update_vintage_records([], _frame([(0, 8000.0, None)]), now=NOW)
        assert "df" not in serialize_records(recs)[0]
        assert np.isnan(deserialize_records(serialize_records(recs))[0].first_seen_df)

    def test_malformed_row_drops_out_without_poisoning_the_window(self):
        good = serialize_records(update_vintage_records([], _frame([(1, 8000.0, None)]), now=NOW))
        assert len(deserialize_records([*good, {"ts": "x"}, {"garbage": 1}])) == 1

    def test_deserialize_tolerates_empty(self):
        assert deserialize_records(None) == []
        assert deserialize_records([]) == []


class TestJoinsToDriftRecords:
    """``canonical_hour`` must agree with ``models.drift._normalize_ts``.

    The whole study is ``vintage.first_seen_d`` joined to the drift series by
    timestamp. ``data/vintage.py`` deliberately does NOT import the helper —
    ``models/`` already imports ``data/`` (``models/training.py:24``), so the
    import would invert the layering. This test is what makes that safe: if the
    two normalizations ever diverge, the join silently returns nothing and the
    study reads as "no revisions anywhere."
    """

    @pytest.mark.parametrize(
        "raw",
        [
            "2026-07-15T12:00:00+00:00",
            "2026-07-15T12:00:00Z",
            pd.Timestamp("2026-07-15T12:00:00", tz="UTC"),
            datetime(2026, 7, 15, 12, 0, tzinfo=UTC),
        ],
    )
    def test_matches_drift_normalization(self, raw):
        from models.drift import _normalize_ts

        mine = canonical_hour(raw)
        theirs = _normalize_ts(raw if isinstance(raw, str) else raw.isoformat())
        assert mine == theirs

    def test_naive_timestamps_are_treated_as_utc(self):
        assert canonical_hour("2026-07-15T12:00:00") == "2026-07-15T12:00:00+00:00"

    def test_unparseable_returns_none(self):
        assert canonical_hour("nope") is None
        assert canonical_hour(None) is None


class TestSummarize:
    def test_flat_scalars_for_structlog(self):
        recs = update_vintage_records([], _frame([(1, 1157.0, 7911.0)]), now=NOW)
        recs = update_vintage_records(recs, _frame([(1, 7815.0, 7911.0)]), now=NOW)
        fields = summarize(recs, region="AZPS")
        assert fields["region"] == "AZPS"
        assert fields["n_records"] == 1
        assert fields["n_revised"] == 1
        assert fields["mean_revision_pct"] == pytest.approx(85.2, abs=0.1)
        assert all(isinstance(v, str | int | float | bool) for v in fields.values())

    def test_counts_placeholders(self):
        frame = _frame([(0, 9008.0, 9008.0), (1, 8700.0, 8800.0), (2, 2153.0, 2153.0)])
        fields = summarize(update_vintage_records([], frame, now=NOW), region="BPAT")
        assert fields["n_placeholder"] == 2
        assert fields["newest_was_placeholder"] is True

    def test_empty_is_safe(self):
        fields = summarize([], region="X")
        assert fields["n_records"] == 0
        assert "mean_revision_pct" not in fields
        assert "newest_hour" not in fields


class TestClassifyRegion:
    """Revision-class heuristic v1 — each class is a named, measured BA.

    Precedence broken > bulk > churn > clean: LDWP is broken AND bulk, and the
    most consequential caveat must drive the user-facing copy.
    """

    def _rec(self, hours_ago: int, *, lag_h: float, first: float, last: float, n: int):
        target = NOW - timedelta(hours=hours_ago)
        return VintageRecord(
            timestamp=target.isoformat(),
            first_seen_d=first,
            first_seen_df=float("nan"),
            captured_at=(target + timedelta(hours=lag_h)).isoformat(),
            last_d=last,
            n_updates=n,
        )

    def _fresh(self, n_hours: int, *, revise_every: int = 1, rev_pct: float = 0.0):
        out = []
        for h in range(1, n_hours + 1):
            revised = rev_pct > 0 and h % revise_every == 0
            first = 4000.0 * (1 - rev_pct / 100.0) if revised else 4000.0
            out.append(self._rec(h, lag_h=1.0, first=first, last=4000.0, n=1 if revised else 0))
        return out

    def _backfilled(self, n_hours: int, *, n_revised: int = 0):
        out = []
        for h in range(48, 48 + n_hours):
            revised = h - 48 < n_revised
            out.append(
                self._rec(
                    h,
                    lag_h=200.0,
                    first=4000.0,
                    last=4100.0 if revised else 4000.0,
                    n=1 if revised else 0,
                )
            )
        return out

    def test_pnm_class_clean(self):
        verdict = classify_region(self._fresh(48) + self._backfilled(200))
        assert verdict["revision_class"] == "clean"
        assert verdict["n_fresh"] == 48

    def test_bpat_class_churn(self):
        """Every fresh hour revises ~15% — churn, not broken."""
        verdict = classify_region(self._fresh(48, rev_pct=15.0) + self._backfilled(200))
        assert verdict["revision_class"] == "churn"
        assert verdict["mean_fresh_revision_pct"] == pytest.approx(15.0, abs=0.1)

    def test_ldwp_class_broken_wins_over_bulk(self):
        """Fresh revisions at 70% AND deep-history rewrites: broken must win."""
        records = self._fresh(48, rev_pct=70.0) + self._backfilled(200, n_revised=25)
        verdict = classify_region(records)
        assert verdict["revision_class"] == "broken"
        assert verdict["n_backfilled_revised"] == 25

    def test_psco_class_bulk(self):
        """Fresh hours stable; the daily file rewrote 23 settled hours."""
        records = self._fresh(48) + self._backfilled(200, n_revised=23)
        verdict = classify_region(records)
        assert verdict["revision_class"] == "bulk"

    def test_insufficient_fresh_is_unknown(self):
        """A day-old deployment must hedge, whatever the backfill says."""
        records = self._fresh(10, rev_pct=70.0) + self._backfilled(600, n_revised=40)
        assert classify_region(records)["revision_class"] == "unknown"

    def test_ambiguous_middle_is_unknown(self):
        """Revised sometimes, moderately — neither clean nor churn: hedge."""
        verdict = classify_region(
            self._fresh(48, revise_every=4, rev_pct=8.0) + self._backfilled(200)
        )
        assert verdict["revision_class"] == "unknown"

    def test_evidence_fields_are_flat_scalars(self):
        verdict = classify_region(self._fresh(48, rev_pct=15.0))
        assert all(isinstance(v, str | int | float) for v in verdict.values())


class TestVintageSummaryKey:
    """The compact web-readable key the provenance callouts consume."""

    def test_summary_key_written_without_records(self, monkeypatch):
        import data.redis_client as rc
        from jobs import phases

        store: dict = {}
        monkeypatch.setattr(rc, "redis_configured", lambda: True)
        monkeypatch.setattr(rc, "redis_get_strict", lambda key: store.get(key))
        monkeypatch.setattr(
            rc, "persist", lambda key, value, ttl=86400: store.__setitem__(key, value)
        )

        hour = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
        frame = pd.DataFrame({"timestamp": [hour], "demand_mw": [1157.0], "forecast_mw": [7911.0]})
        assert phases.write_vintage_records("AZPS", frame).ok

        summary = store["gridpulse:vintage_summary:AZPS"]
        assert "records" not in summary, "the summary key must stay ~250B — no records array"
        assert summary["revision_class"] == "unknown"  # one fresh hour: hedge
        assert summary["n_records"] == 1
        assert "n_fresh" in summary
        # and the heavyweight key still carries the full records
        assert "records" in store["gridpulse:vintage:AZPS"]


class TestVintageGcsMirror:
    """Anchor-redesign PR A — the replay study's data access + flush durability.

    Best-effort by contract: the mirror must never affect the phase's
    ok-flag (unlike the persist-strict Redis writes)."""

    def _wire(self, monkeypatch, store: dict, mirror_calls: list):
        import data.gcs_store as gcs
        import data.redis_client as rc

        monkeypatch.setattr(rc, "redis_configured", lambda: True)
        monkeypatch.setattr(rc, "redis_get_strict", lambda key: store.get(key))
        monkeypatch.setattr(
            rc, "persist", lambda key, value, ttl=86400: store.__setitem__(key, value)
        )
        monkeypatch.setattr(
            gcs, "write_parquet", lambda df, dt, region: mirror_calls.append((df, dt, region))
        )

    def _frame(self):
        hour = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
        return pd.DataFrame({"timestamp": [hour], "demand_mw": [1157.0], "forecast_mw": [7911.0]})

    def test_mirror_receives_the_serialized_window(self, monkeypatch):
        from jobs import phases

        store: dict = {}
        calls: list = []
        self._wire(monkeypatch, store, calls)

        assert phases.write_vintage_records("AZPS", self._frame()).ok
        assert len(calls) == 1
        df, data_type, region = calls[0]
        assert data_type == "vintage" and region == "AZPS"
        assert list(df["d"]) == [1157.0]  # first_seen_d rides the mirror
        assert "ld" in df.columns and "at" in df.columns

    def test_mirror_failure_never_affects_the_phase(self, monkeypatch):
        import data.gcs_store as gcs
        from jobs import phases

        store: dict = {}
        self._wire(monkeypatch, store, [])
        monkeypatch.setattr(
            gcs,
            "write_parquet",
            lambda df, dt, region: (_ for _ in ()).throw(RuntimeError("gcs down")),
        )

        result = phases.write_vintage_records("AZPS", self._frame())
        assert result.ok is True, "a best-effort mirror failure must not fail capture"
        assert "gridpulse:vintage:AZPS" in store  # the Redis truth still landed
