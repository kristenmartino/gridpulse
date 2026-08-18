"""Unit tests for models/drift.py — #121 part 1.

Coverage targets:
- Pure-function arithmetic (absolute_pct_error, mape_over_records, rolling_mape)
- Edge cases (zero actual, NaN inputs, empty windows, malformed records)
- Payload construction (compute_drift_payload merges + trims + computes rolling)
- Forecast-row extraction (extract_one_hour_ahead_predictions tolerates
  Z suffix vs +00:00, missing rows, non-numeric fields)
- Record building (build_records_from_actuals picks most-recent matchable hour)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from models.drift import (
    DEFAULT_MAX_RECORDS,
    LOW_ACTUAL_FRACTION,
    WINDOW_7D_HOURS,
    WINDOW_30D_HOURS,
    DriftRecord,
    absolute_pct_error,
    build_records_from_actuals,
    compute_drift_payload,
    deserialize_records,
    extract_one_hour_ahead_predictions,
    filter_low_actuals,
    mape_over_records,
    merge_and_trim,
    rolling_mape,
    rolling_smape,
    serialize_records,
    smape_over_records,
    symmetric_pct_error,
)


def _ts(hours_ago: int, now: datetime | None = None) -> str:
    """Helper: ISO timestamp ``hours_ago`` before ``now`` (defaults to a fixed instant)."""
    base = now or datetime(2026, 5, 20, 15, 0, 0, tzinfo=UTC)
    return (base - timedelta(hours=hours_ago)).isoformat()


def _rec(hours_ago: int, error_pct: float, now: datetime | None = None) -> DriftRecord:
    """Helper: build a DriftRecord at ``hours_ago`` with the given % error."""
    return DriftRecord(
        timestamp=_ts(hours_ago, now=now),
        predicted=100_000.0,
        actual=100_000.0 / (1 + error_pct / 100.0) if error_pct != 0 else 100_000.0,
        abs_pct_error=error_pct,
    )


class TestAbsolutePctError:
    def test_basic(self):
        assert absolute_pct_error(110.0, 100.0) == pytest.approx(10.0)
        assert absolute_pct_error(90.0, 100.0) == pytest.approx(10.0)

    def test_zero_actual_returns_none(self):
        # EIA sentinel: actual=0 is "missing observation," not real demand.
        # We drop the record rather than report a degenerate error.
        assert absolute_pct_error(50.0, 0.0) is None

    def test_nan_inputs_return_none(self):
        assert absolute_pct_error(float("nan"), 100.0) is None
        assert absolute_pct_error(100.0, float("nan")) is None
        assert absolute_pct_error(float("inf"), 100.0) is None

    def test_perfect_prediction(self):
        assert absolute_pct_error(100.0, 100.0) == pytest.approx(0.0)


class TestMapeOverRecords:
    def test_empty_returns_none(self):
        # "no data yet" is distinguishable from "model is perfect" — both
        # would silently render as 0% if we returned 0.0 here.
        assert mape_over_records([]) is None

    def test_single_record(self):
        assert mape_over_records([_rec(1, 5.0)]) == pytest.approx(5.0)

    def test_mean_of_multiple(self):
        records = [_rec(1, 4.0), _rec(2, 8.0), _rec(3, 6.0)]
        assert mape_over_records(records) == pytest.approx(6.0)

    def test_skips_nonfinite_errors(self):
        # A bad row shouldn't poison the mean — production drift records
        # CAN end up with NaN if a stored record was corrupted by an old
        # bug. The aggregator must be tolerant.
        bad = DriftRecord(
            timestamp=_ts(1), predicted=100.0, actual=100.0, abs_pct_error=float("nan")
        )
        good = _rec(2, 6.0)
        assert mape_over_records([bad, good]) == pytest.approx(6.0)


class TestRollingMape:
    def test_empty_returns_none(self):
        assert rolling_mape([], WINDOW_7D_HOURS) is None

    def test_filters_outside_window(self):
        # now = 2026-05-20 15:00 UTC. Records at 1h, 24h, 200h ago.
        # 7d window (168h) should exclude only the 200h-ago record.
        now = datetime(2026, 5, 20, 15, 0, 0, tzinfo=UTC)
        recs = [_rec(1, 4.0, now=now), _rec(24, 8.0, now=now), _rec(200, 20.0, now=now)]
        result = rolling_mape(recs, WINDOW_7D_HOURS, now_iso=now.isoformat())
        # Only the 4% and 8% records should count.
        assert result == pytest.approx(6.0)

    def test_30d_window_includes_all_recent_data(self):
        now = datetime(2026, 5, 20, 15, 0, 0, tzinfo=UTC)
        recs = [_rec(h, 5.0, now=now) for h in range(0, 720, 24)]
        result = rolling_mape(recs, WINDOW_30D_HOURS, now_iso=now.isoformat())
        assert result == pytest.approx(5.0)

    def test_returns_none_when_window_excludes_everything(self):
        now = datetime(2026, 5, 20, 15, 0, 0, tzinfo=UTC)
        ancient = [_rec(2000, 10.0, now=now)]  # older than any reasonable window
        assert rolling_mape(ancient, WINDOW_7D_HOURS, now_iso=now.isoformat()) is None


class TestExtractOneHourAheadPredictions:
    def test_finds_matching_row(self):
        target = "2026-05-20T15:00:00+00:00"
        payload = {
            "forecasts": [
                {
                    "timestamp": target,
                    "predicted_demand_mw": 95234.5,
                    "xgboost": 95234.5,
                    "prophet": 96012.0,
                    "arima": 94800.3,
                    "ensemble": 95212.0,
                },
                {"timestamp": "2026-05-20T16:00:00+00:00", "xgboost": 96000},
            ]
        }
        out = extract_one_hour_ahead_predictions(payload, target)
        assert out == {
            "xgboost": pytest.approx(95234.5),
            "prophet": pytest.approx(96012.0),
            "arima": pytest.approx(94800.3),
            "ensemble": pytest.approx(95212.0),
        }

    def test_none_payload(self):
        assert extract_one_hour_ahead_predictions(None, "2026-05-20T15:00:00+00:00") == {}

    def test_empty_forecasts(self):
        assert extract_one_hour_ahead_predictions({"forecasts": []}, "x") == {}

    def test_no_matching_timestamp(self):
        payload = {"forecasts": [{"timestamp": "2026-05-20T16:00:00+00:00", "xgboost": 1}]}
        out = extract_one_hour_ahead_predictions(payload, "2026-05-20T15:00:00+00:00")
        assert out == {}

    def test_tolerates_z_suffix_vs_explicit_offset(self):
        # Production payloads use +00:00; some test fixtures use Z.
        # The extractor normalizes so they match.
        payload = {"forecasts": [{"timestamp": "2026-05-20T15:00:00Z", "xgboost": 100.0}]}
        out = extract_one_hour_ahead_predictions(payload, "2026-05-20T15:00:00+00:00")
        assert out == {"xgboost": pytest.approx(100.0)}

    def test_skips_non_numeric_and_nan(self):
        payload = {
            "forecasts": [
                {
                    "timestamp": "2026-05-20T15:00:00+00:00",
                    "xgboost": 100.0,
                    "prophet": float("nan"),
                    "arima": "broken",
                    "ensemble": 99.0,
                }
            ]
        }
        out = extract_one_hour_ahead_predictions(payload, "2026-05-20T15:00:00+00:00")
        assert "xgboost" in out and "ensemble" in out
        assert "prophet" not in out and "arima" not in out


class TestFilterByLead:
    """P2-19 (#273): the 1-hour-ahead window stops averaging in multi-hour leads."""

    def _rec(self, lead, err=5.0, hours_ago=1):
        return DriftRecord(
            timestamp=_ts(hours_ago),
            predicted=100_000.0,
            actual=100_000.0,
            abs_pct_error=err,
            lead_hours=lead,
        )

    def test_drops_known_leads_above_the_bar(self):
        from models.drift import filter_by_lead

        recs = [self._rec(1), self._rec(2), self._rec(6)]
        kept, dropped, unknown = filter_by_lead(recs, 1)
        assert [r.lead_hours for r in kept] == [1]
        assert (dropped, unknown) == (2, 0)

    def test_keeps_unknown_leads_and_counts_them(self):
        """Unknown lead is 97% likely to be lead 1 — dropping it costs history.

        This is the load-bearing decision in the filter: a record written
        before the field existed can never have its lead recovered, so
        excluding them would discard most of the window to remove a 3%
        contamination.
        """
        from models.drift import filter_by_lead

        recs = [self._rec(1), self._rec(None), self._rec(4)]
        kept, dropped, unknown = filter_by_lead(recs, 1)
        assert sorted(r.lead_hours or 0 for r in kept) == [0, 1]  # None + lead-1
        assert (dropped, unknown) == (1, 1)

    def test_no_op_when_disabled(self):
        from models.drift import filter_by_lead

        recs = [self._rec(1), self._rec(9)]
        kept, dropped, unknown = filter_by_lead(recs, None)
        assert len(kept) == 2 and (dropped, unknown) == (0, 0)

    def test_rolling_mape_excludes_the_contaminating_lead(self):
        """The whole point: a lead-6 record must not move the 1h headline."""
        from models.drift import rolling_mape

        clean = [self._rec(1, err=4.0, hours_ago=h) for h in range(1, 5)]
        dirty = clean + [self._rec(6, err=100.0, hours_ago=5)]
        unfiltered = rolling_mape(dirty, WINDOW_7D_HOURS, now_iso=_ts(0))
        filtered = rolling_mape(dirty, WINDOW_7D_HOURS, now_iso=_ts(0), max_lead=1)
        assert unfiltered == pytest.approx(23.2)  # the lead-6 record dominates
        assert filtered == pytest.approx(4.0)  # ...and is gone once filtered

    def test_payload_publishes_both_transition_counters(self):
        """n_lead_excluded_7d and n_lead_unknown_7d make the state observable."""
        recs = {
            "xgboost": DriftRecord(
                timestamp=_ts(1), predicted=100.0, actual=100.0, abs_pct_error=1.0, lead_hours=1
            )
        }
        existing = {
            "models": {
                "xgboost": {
                    "records": [
                        {"ts": _ts(2), "p": 100.0, "a": 100.0, "e": 1.0, "l": 5},
                        {"ts": _ts(3), "p": 100.0, "a": 100.0, "e": 1.0},  # legacy, no lead
                    ]
                }
            }
        }
        out = compute_drift_payload("PJM", existing, recs, now_iso=_ts(0))
        blk = out["models"]["xgboost"]
        assert blk["n_lead_excluded_7d"] == 1
        assert blk["n_lead_unknown_7d"] == 1
        # The dropped record stays in ``records`` — history is not rewritten.
        assert blk["n_records"] == 3
        assert len(blk["records"]) == 3
        # ...but it does not feed the mean.
        assert blk["n_7d"] == 2

    def test_horizon_drift_is_not_lead_filtered(self):
        """#227 records have a DESIGNED 24/48/72h horizon — filtering empties them."""
        import inspect

        from models import drift as mod

        src = inspect.getsource(mod._horizon_rollup_block)
        assert "filter_by_lead" not in src


class TestBuildRecordsFromActuals:
    def _previous_forecast(self) -> dict:
        return {
            "region": "PJM",
            "forecasts": [
                {
                    "timestamp": "2026-05-20T13:00:00+00:00",
                    "xgboost": 100_000.0,
                    "prophet": 102_000.0,
                },
                {
                    "timestamp": "2026-05-20T14:00:00+00:00",
                    "xgboost": 101_000.0,
                    "prophet": 103_000.0,
                },
            ],
        }

    def test_picks_most_recent_matchable_hour(self):
        # Two forecast hours, both have actuals available — we want the
        # MOST RECENT one (14:00), not the earlier (13:00). Hourly cadence
        # → one record per tick.
        actuals = {
            "2026-05-20T13:00:00+00:00": 101_500.0,
            "2026-05-20T14:00:00+00:00": 102_000.0,
        }
        recs = build_records_from_actuals(self._previous_forecast(), actuals)
        for model_name in ("xgboost", "prophet"):
            assert recs[model_name].timestamp == "2026-05-20T14:00:00+00:00"
            assert recs[model_name].actual == pytest.approx(102_000.0)

    def test_records_carry_the_lead_they_actually_had(self):
        """P2-19 (#273): the window is called 1-hour-ahead and isn't uniformly 1h.

        Both forecast hours are matchable here, so the kept record is the
        SECOND row — a lead of 2, pooled into a statistic labelled 1h. The
        field makes that visible instead of assumed.
        """
        actuals = {
            "2026-05-20T13:00:00+00:00": 101_500.0,
            "2026-05-20T14:00:00+00:00": 102_000.0,
        }
        recs = build_records_from_actuals(self._previous_forecast(), actuals)
        for model_name in ("xgboost", "prophet"):
            assert recs[model_name].lead_hours == 2

    def test_lead_is_one_when_the_first_row_is_the_match(self):
        actuals = {"2026-05-20T13:00:00+00:00": 101_500.0}
        recs = build_records_from_actuals(self._previous_forecast(), actuals)
        assert recs["xgboost"].lead_hours == 1

    def test_lead_is_none_rather_than_guessed_when_unknowable(self):
        """An unknown lead must stay unknown — never default to 1."""
        from models.drift import _lead_hours

        assert _lead_hours([], "2026-05-20T14:00:00+00:00") is None
        assert _lead_hours([{"timestamp": ""}], "2026-05-20T14:00:00+00:00") is None
        assert _lead_hours([{"timestamp": "not-a-date"}], "2026-05-20T14:00:00+00:00") is None
        # Target before the forecast even starts is not a lead of 0 or -1.
        assert (
            _lead_hours([{"timestamp": "2026-05-20T14:00:00+00:00"}], "2026-05-20T12:00:00+00:00")
            is None
        )

    def test_lead_survives_the_redis_round_trip(self):
        """And a record written before the field existed stays None."""
        from models.drift import DriftRecord, deserialize_records, serialize_records

        rec = DriftRecord(
            timestamp="2026-05-20T14:00:00+00:00",
            predicted=100.0,
            actual=101.0,
            abs_pct_error=1.0,
            lead_hours=3,
        )
        rows = serialize_records([rec])
        assert rows[0]["l"] == 3
        assert deserialize_records(rows)[0].lead_hours == 3

        legacy = [{"ts": "2026-05-20T14:00:00+00:00", "p": 100.0, "a": 101.0, "e": 1.0}]
        assert "l" not in legacy[0]
        assert deserialize_records(legacy)[0].lead_hours is None

        # A record with no known lead must not write a misleading key.
        unknown = DriftRecord(
            timestamp="2026-05-20T15:00:00+00:00", predicted=1.0, actual=1.0, abs_pct_error=0.0
        )
        assert "l" not in serialize_records([unknown])[0]

    def test_skips_hour_with_no_actual(self):
        # Only the older forecast hour has an actual. The newer one
        # (14:00) is still being awaited from EIA. Use what we have.
        actuals = {"2026-05-20T13:00:00+00:00": 101_500.0}
        recs = build_records_from_actuals(self._previous_forecast(), actuals)
        assert recs["xgboost"].timestamp == "2026-05-20T13:00:00+00:00"

    def test_no_previous_forecast_returns_empty(self):
        assert build_records_from_actuals(None, {"x": 1.0}) == {}

    def test_no_actuals_returns_empty(self):
        assert build_records_from_actuals(self._previous_forecast(), {}) == {}

    def test_zero_actual_filtered(self):
        # actual=0 makes the % error undefined → record skipped.
        actuals = {"2026-05-20T14:00:00+00:00": 0.0}
        assert build_records_from_actuals(self._previous_forecast(), actuals) == {}

    def test_computes_correct_pct_error(self):
        # predicted=101_000, actual=100_000 → 1% over.
        actuals = {"2026-05-20T14:00:00+00:00": 100_000.0}
        # Patch the forecast to only have one model + only the matching hour
        # for a clean numeric check.
        forecast = {"forecasts": [{"timestamp": "2026-05-20T14:00:00+00:00", "xgboost": 101_000.0}]}
        recs = build_records_from_actuals(forecast, actuals)
        assert recs["xgboost"].abs_pct_error == pytest.approx(1.0)


class TestMergeAndTrim:
    def test_appends_new_record(self):
        existing = [_rec(2, 5.0), _rec(1, 4.0)]
        new = _rec(0, 3.0)
        out = merge_and_trim(existing, new)
        assert len(out) == 3
        # Sorted oldest → newest
        assert out[0].timestamp < out[1].timestamp < out[2].timestamp

    def test_deduplicates_same_timestamp(self):
        # Re-scoring against the same actuals (e.g. backfill) shouldn't
        # double-count. The new record replaces the old.
        existing = [_rec(1, 10.0)]
        new = DriftRecord(
            timestamp=existing[0].timestamp,
            predicted=200.0,
            actual=210.0,
            abs_pct_error=4.76,
        )
        out = merge_and_trim(existing, new)
        assert len(out) == 1
        assert out[0].abs_pct_error == pytest.approx(4.76)

    def test_trims_to_max_records(self):
        # 25 records, max=20 → drop the 5 oldest.
        existing = [_rec(h, 5.0) for h in range(25, 0, -1)]
        new = _rec(0, 5.0)
        out = merge_and_trim(existing, new, max_records=20)
        assert len(out) == 20
        # Newest 20 retained.
        oldest_kept_h = max(h for h in range(25, 0, -1) if h < 20)
        assert any(_ts(oldest_kept_h) == r.timestamp for r in out)

    def test_none_new_record_returns_existing_sorted(self):
        existing = [_rec(2, 5.0), _rec(1, 4.0)]
        out = merge_and_trim(existing, None)
        assert [r.timestamp for r in out] == sorted(r.timestamp for r in existing)


class TestSerializeRoundTrip:
    def test_round_trip(self):
        original = [
            DriftRecord(
                timestamp="2026-05-20T15:00:00+00:00",
                predicted=95234.5,
                actual=98123.2,
                abs_pct_error=2.94,
            )
        ]
        serialized = serialize_records(original)
        # Compact short-key form.
        assert serialized[0]["ts"] == "2026-05-20T15:00:00+00:00"
        assert serialized[0]["p"] == pytest.approx(95234.5)
        assert serialized[0]["a"] == pytest.approx(98123.2)
        assert serialized[0]["e"] == pytest.approx(2.94)

        deserialized = deserialize_records(serialized)
        assert len(deserialized) == 1
        assert deserialized[0].abs_pct_error == pytest.approx(2.94)

    def test_deserialize_tolerates_malformed_rows(self):
        # One good row, one missing required key, one with wrong types.
        rows = [
            {
                "ts": "2026-05-20T15:00:00+00:00",
                "p": 100.0,
                "a": 102.0,
                "e": 2.0,
            },
            {"ts": "missing-others"},
            {"ts": "2026-05-20T16:00:00+00:00", "p": "not-a-number", "a": 0, "e": 0},
        ]
        out = deserialize_records(rows)
        # Only the good row survives.
        assert len(out) == 1
        assert out[0].abs_pct_error == pytest.approx(2.0)

    def test_deserialize_empty_or_none(self):
        assert deserialize_records(None) == []
        assert deserialize_records([]) == []


class TestComputeDriftPayload:
    def test_first_run_builds_per_model_entries(self):
        new_records = {
            "xgboost": _rec(0, 5.0),
            "prophet": _rec(0, 8.0),
        }
        now = datetime(2026, 5, 20, 15, 0, 0, tzinfo=UTC)
        payload = compute_drift_payload(
            "PJM",
            existing_payload=None,
            new_records=new_records,
            now_iso=now.isoformat(),
        )
        assert payload["region"] == "PJM"
        assert set(payload["models"].keys()) == {"xgboost", "prophet"}
        # n_records=1 means rolling MAPE is just that single record's error.
        assert payload["models"]["xgboost"]["n_records"] == 1
        assert payload["models"]["xgboost"]["rolling_mape_7d"] == pytest.approx(5.0)
        assert payload["models"]["prophet"]["rolling_mape_7d"] == pytest.approx(8.0)

    def test_merges_existing_records(self):
        now = datetime(2026, 5, 20, 15, 0, 0, tzinfo=UTC)
        # Same-scale actuals on purpose: PR-G9's region-relative low-actual
        # filter (compute_drift_payload now passes LOW_ACTUAL_FRACTION) would
        # legitimately drop a record whose actual sits below 10% of the
        # window median. The new _rec(0, 4.0) has an actual of ~96,154 MW, so
        # the existing record must be the same order of magnitude or it gets
        # filtered and the mean changes. (A toy 95 MW actual here *is* the
        # LDWP pathology — see test_ldwp_like_window_no_longer_explodes.)
        existing_payload = {
            "models": {
                "xgboost": {
                    "records": [
                        {
                            "ts": _ts(2, now=now),
                            "p": 100_000,
                            "a": 95_000,
                            "e": 5.26,  # ~5.26% error (stored, used directly)
                        }
                    ]
                }
            }
        }
        new_records = {"xgboost": _rec(0, 4.0, now=now)}
        payload = compute_drift_payload(
            "PJM",
            existing_payload=existing_payload,
            new_records=new_records,
            now_iso=now.isoformat(),
        )
        assert payload["models"]["xgboost"]["n_records"] == 2
        # rolling_mape_7d = mean(5.26, 4.0) ≈ 4.63 — both records survive the
        # low-actual filter because they're the same scale.
        assert payload["models"]["xgboost"]["rolling_mape_7d"] == pytest.approx(4.63, abs=0.01)
        assert payload["models"]["xgboost"]["n_low_actual_excluded_7d"] == 0

    def test_preserves_model_with_existing_but_no_new_record(self):
        # Prophet had records previously; this tick produced only an
        # xgboost record (maybe Prophet failed to load this tick).
        # Prophet's history should be preserved.
        now = datetime(2026, 5, 20, 15, 0, 0, tzinfo=UTC)
        existing = {
            "models": {
                "prophet": {"records": [{"ts": _ts(2, now=now), "p": 100, "a": 105, "e": 4.76}]}
            }
        }
        new_records = {"xgboost": _rec(0, 3.0, now=now)}
        payload = compute_drift_payload(
            "PJM",
            existing_payload=existing,
            new_records=new_records,
            now_iso=now.isoformat(),
        )
        assert "prophet" in payload["models"]
        assert payload["models"]["prophet"]["n_records"] == 1
        assert "xgboost" in payload["models"]

    def test_record_window_trimmed_to_max(self):
        now = datetime(2026, 5, 20, 15, 0, 0, tzinfo=UTC)
        # Existing has DEFAULT_MAX_RECORDS records already.
        existing_records = [
            {"ts": _ts(h, now=now), "p": 100, "a": 100, "e": 0.0}
            for h in range(DEFAULT_MAX_RECORDS, 0, -1)
        ]
        existing = {"models": {"xgboost": {"records": existing_records}}}
        new_records = {"xgboost": _rec(0, 1.0, now=now)}
        payload = compute_drift_payload(
            "PJM",
            existing_payload=existing,
            new_records=new_records,
            now_iso=now.isoformat(),
        )
        # New record added, oldest dropped → still exactly max_records.
        assert payload["models"]["xgboost"]["n_records"] == DEFAULT_MAX_RECORDS

    def test_includes_last_updated_at(self):
        now = datetime(2026, 5, 20, 15, 0, 0, tzinfo=UTC)
        payload = compute_drift_payload(
            "PJM",
            existing_payload=None,
            new_records={"xgboost": _rec(0, 1.0, now=now)},
            now_iso=now.isoformat(),
        )
        assert payload["last_updated_at"] == now.isoformat()


# ── PR-G9 / #142: robust drift stats for near-zero actuals ──────────────────


def _real_rec(
    hours_ago: int, predicted: float, actual: float, now: datetime | None = None
) -> DriftRecord:
    """DriftRecord from real (predicted, actual); sMAPE auto-computed."""
    err = absolute_pct_error(predicted, actual)
    return DriftRecord(
        timestamp=_ts(hours_ago, now=now),
        predicted=predicted,
        actual=actual,
        abs_pct_error=err if err is not None else float("nan"),
    )


def _window_payload(recs: list[DriftRecord]) -> dict:
    """Existing-payload shape carrying ``recs`` under a single 'ensemble' model."""
    return {"models": {"ensemble": {"records": serialize_records(recs)}}}


class TestSymmetricPctError:
    def test_basic_value(self):
        # 200 * |110-100| / (110+100) = 2000/210 = 9.5238...
        assert symmetric_pct_error(110.0, 100.0) == pytest.approx(9.5238, abs=1e-3)

    def test_perfect_prediction_is_zero(self):
        assert symmetric_pct_error(100.0, 100.0) == pytest.approx(0.0)

    def test_zero_actual_normal_prediction_is_bounded_200(self):
        # The whole point: actual=0 with a normal-scale prediction is a single
        # bounded 200% miss under sMAPE, NOT the None/undefined of plain MAPE.
        assert symmetric_pct_error(2500.0, 0.0) == pytest.approx(200.0)

    def test_both_zero_returns_none(self):
        # Degenerate: nothing predicted, nothing happened → no signal.
        assert symmetric_pct_error(0.0, 0.0) is None

    def test_near_zero_actual_normal_prediction_is_bounded(self):
        # actual=50 MW vs predicted=2500 MW (LDWP artifact scale):
        #   plain MAPE  = 2450/50*100        = 4900%   (explodes the mean)
        #   sMAPE       = 200*2450/2550       ≈ 192.16% (bounded by construction)
        assert absolute_pct_error(2500.0, 50.0) == pytest.approx(4900.0)
        smape = symmetric_pct_error(2500.0, 50.0)
        assert smape == pytest.approx(192.157, abs=1e-2)
        assert 0.0 <= smape <= 200.0

    def test_nan_or_inf_inputs_return_none(self):
        assert symmetric_pct_error(float("nan"), 100.0) is None
        assert symmetric_pct_error(100.0, float("nan")) is None
        assert symmetric_pct_error(float("inf"), 100.0) is None


class TestDriftRecordSmape:
    def test_auto_fills_smape_from_predicted_actual(self):
        # Bare construction (the common path, incl. legacy records + helpers)
        # gets a correct sMAPE without the caller passing one.
        rec = DriftRecord(
            timestamp="2026-05-20T15:00:00+00:00",
            predicted=110.0,
            actual=100.0,
            abs_pct_error=10.0,
        )
        assert rec.smape == pytest.approx(9.5238, abs=1e-3)

    def test_explicit_smape_is_preserved(self):
        rec = DriftRecord(
            timestamp="2026-05-20T15:00:00+00:00",
            predicted=110.0,
            actual=100.0,
            abs_pct_error=10.0,
            smape=42.0,
        )
        assert rec.smape == pytest.approx(42.0)


class TestFilterLowActuals:
    def test_drops_region_relative_outliers(self):
        # 6 normal (~2500) + 2 near-zero (50). median=2500, threshold=250.
        recs = [_real_rec(h, 2400.0, 2500.0) for h in range(6)]
        recs += [_real_rec(10, 2500.0, 50.0), _real_rec(11, 2500.0, 40.0)]
        kept, n_dropped = filter_low_actuals(recs)
        assert n_dropped == 2
        assert all(abs(r.actual) >= LOW_ACTUAL_FRACTION * 2500.0 for r in kept)

    def test_no_filter_when_fraction_zero(self):
        recs = [_real_rec(0, 2500.0, 50.0), _real_rec(1, 2400.0, 2500.0)]
        kept, n_dropped = filter_low_actuals(recs, min_fraction=0.0)
        assert n_dropped == 0
        assert len(kept) == 2

    def test_uniform_small_scale_region_not_decimated(self):
        # A genuinely small BA: every actual ~50 MW. Nothing is an outlier
        # relative to its own scale, so nothing is dropped — this is why the
        # threshold is fraction-of-median, not a universal MW floor.
        recs = [_real_rec(h, 48.0, 50.0) for h in range(10)]
        kept, n_dropped = filter_low_actuals(recs)
        assert n_dropped == 0
        assert len(kept) == 10

    def test_empty_window_is_noop(self):
        assert filter_low_actuals([]) == ([], 0)


class TestSmapeOverRecords:
    def test_empty_returns_none(self):
        assert smape_over_records([]) is None

    def test_mean_of_smape(self):
        recs = [_real_rec(0, 110.0, 100.0), _real_rec(1, 90.0, 100.0)]
        # sMAPE(110,100)=9.5238, sMAPE(90,100)=200*10/190=10.5263 → mean 10.025
        assert smape_over_records(recs) == pytest.approx(10.025, abs=1e-2)


class TestRollingSmape:
    def test_filters_window_and_low_actuals(self):
        now = datetime(2026, 5, 20, 15, 0, 0, tzinfo=UTC)
        recs = [
            _real_rec(1, 2400.0, 2500.0, now=now),  # in window, normal
            _real_rec(24, 2450.0, 2500.0, now=now),  # in window, normal
            _real_rec(30, 2500.0, 50.0, now=now),  # in window, near-zero → filtered
            _real_rec(200, 2400.0, 2500.0, now=now),  # outside 7d window
        ]
        result = rolling_smape(recs, WINDOW_7D_HOURS, now_iso=now.isoformat())
        # Only the two normal in-window records count.
        expected = smape_over_records([recs[0], recs[1]])
        assert result == pytest.approx(expected)
        assert result < 10.0  # plausible, not pinned near 200


class TestComputeDriftPayloadRobustness:
    def _ldwp_like(self, now: datetime, n_normal: int = 160, n_artifact: int = 8):
        """A 7d window: mostly normal ~2500 MW + a handful of ~50 MW artifacts."""
        recs = [_real_rec(h + n_artifact, 2400.0, 2500.0, now=now) for h in range(n_normal)]
        recs += [_real_rec(h, 2500.0, 50.0, now=now) for h in range(n_artifact)]
        return recs

    def test_ldwp_like_window_no_longer_explodes(self):
        now = datetime(2026, 5, 20, 15, 0, 0, tzinfo=UTC)
        recs = self._ldwp_like(now)

        # Sanity: the RAW (unfiltered) MAPE genuinely explodes — this is the
        # #142 symptom we are fixing, asserted so the test proves the before.
        raw_mape = mape_over_records(recs)
        assert raw_mape is not None and raw_mape > 100.0

        payload = compute_drift_payload(
            "LDWP",
            existing_payload=_window_payload(recs),
            new_records={},
            now_iso=now.isoformat(),
        )
        ens = payload["models"]["ensemble"]

        # Headline sMAPE and the (now-filtered) MAPE both land in a plausible
        # band for a region this size — not 200%+.
        assert ens["rolling_smape_7d"] is not None
        assert ens["rolling_smape_7d"] < 40.0
        assert ens["rolling_mape_7d"] is not None
        assert ens["rolling_mape_7d"] < 40.0
        # The 8 artifact hours were identified and excluded.
        assert ens["n_low_actual_excluded_7d"] == 8

    def test_normal_region_unchanged_by_filter(self):
        # FPL/MISO/SPP/NYISO/ISONE-style: every actual is the same scale, so
        # the filter is a no-op and the persisted MAPE equals the plain mean
        # (no regression), with sMAPE ≈ MAPE for well-behaved errors.
        now = datetime(2026, 5, 20, 15, 0, 0, tzinfo=UTC)
        recs = [_real_rec(h, 2400.0, 2500.0, now=now) for h in range(168)]
        payload = compute_drift_payload(
            "FPL",
            existing_payload=_window_payload(recs),
            new_records={},
            now_iso=now.isoformat(),
        )
        ens = payload["models"]["ensemble"]
        assert ens["n_low_actual_excluded_7d"] == 0
        assert ens["rolling_mape_7d"] == pytest.approx(mape_over_records(recs))
        # sMAPE within a couple points of MAPE when errors are small/normal.
        assert ens["rolling_smape_7d"] == pytest.approx(ens["rolling_mape_7d"], abs=1.0)


class TestSmapeSerialization:
    def test_round_trip_includes_smape(self):
        recs = [_real_rec(0, 2400.0, 2500.0)]
        serialized = serialize_records(recs)
        assert "s" in serialized[0]
        assert serialized[0]["s"] == pytest.approx(recs[0].smape, abs=1e-3)
        restored = deserialize_records(serialized)
        assert restored[0].smape == pytest.approx(recs[0].smape, abs=1e-3)

    def test_backward_compat_recomputes_missing_smape(self):
        # Pre-PR-G9 records have no 's' key → recomputed from p/a on load.
        rows = [{"ts": "2026-05-20T15:00:00+00:00", "p": 100.0, "a": 105.0, "e": 4.76}]
        restored = deserialize_records(rows)
        assert len(restored) == 1
        assert restored[0].smape == pytest.approx(symmetric_pct_error(100.0, 105.0), abs=1e-6)


class TestRegradeRecords:
    """Settled-grade drift (#304 endgame): stored records re-score against
    EIA's current view each tick.

    Fixtures use the real prod relationships the retired revision probe
    measured: LDWP records scored against partials (panel 147.91 vs settled
    53.22), AZPS 338.67 vs 15.64, PNM 2.06 vs 1.82.
    """

    def _rec(self, ts: str, predicted: float, actual: float) -> DriftRecord:
        return DriftRecord(
            timestamp=ts,
            predicted=predicted,
            actual=actual,
            abs_pct_error=abs(predicted - actual) / actual * 100.0,
        )

    def test_updates_actual_error_and_smape(self):
        """The LDWP case: prediction 4200 scored against partial 967 (334%
        error) re-grades against settled 4840 -> 13.2%."""
        from models.drift import regrade_records

        rec = self._rec("2026-07-17T05:00:00+00:00", 4200.0, 967.0)
        old_smape = rec.smape

        regraded, stats = regrade_records([rec], {"2026-07-17T05:00:00+00:00": 4840.0})

        r = regraded[0]
        assert r.actual == 4840.0
        assert r.abs_pct_error == pytest.approx(13.22, abs=0.01)
        assert np.isfinite(r.smape) and r.smape != old_smape, "sMAPE not refreshed"
        assert r.predicted == 4200.0 and r.timestamp == rec.timestamp
        assert stats["n_regraded"] == 1
        assert stats["mean_abs_shift_pct"] == pytest.approx(80.02, abs=0.05)

    def test_absent_hours_skipped_never_agreement(self):
        """A guard-excluded partial or fetch gap keeps the prior value —
        absence from the fresh frame is unknown, not confirmation."""
        from models.drift import regrade_records

        rec = self._rec("2026-07-17T05:00:00+00:00", 4200.0, 967.0)
        regraded, stats = regrade_records([rec], {"2026-07-17T06:00:00+00:00": 4840.0})

        assert regraded[0].actual == 967.0
        assert stats["n_regraded"] == 0

    def test_serializer_equal_values_do_not_churn(self):
        """No rebuild when the value is identical after 2dp rounding — every
        tick would otherwise rewrite every record from float noise."""
        from models.drift import regrade_records

        rec = self._rec("2026-07-17T05:00:00+00:00", 4200.0, 4840.0)
        regraded, stats = regrade_records([rec], {"2026-07-17T05:00:00+00:00": 4840.004})

        assert regraded[0] is rec
        assert stats["n_regraded"] == 0

    def test_unusable_new_values_skipped(self):
        from models.drift import regrade_records

        rec = self._rec("2026-07-17T05:00:00+00:00", 4200.0, 967.0)
        for bad in (0.0, -5.0, float("nan")):
            regraded, stats = regrade_records([rec], {"2026-07-17T05:00:00+00:00": bad})
            assert regraded[0].actual == 967.0
            assert stats["n_regraded"] == 0

    def test_z_suffix_timestamps_join(self):
        """Records serialized with Z-suffix history must still match the
        actuals map's +00:00 form (same normalization as the retired probe)."""
        from models.drift import regrade_records

        rec = self._rec("2026-07-17T05:00:00Z", 4200.0, 967.0)
        regraded, stats = regrade_records([rec], {"2026-07-17T05:00:00+00:00": 4840.0})
        assert stats["n_regraded"] == 1
        assert regraded[0].actual == 4840.0


class TestPayloadRegrade:
    """compute_drift_payload(actuals=...) — the rolling stats self-correct."""

    def _payload_with_partial_history(self):
        """A stored window where every record was scored against an LDWP-class
        partial: predictions ~4200, partial actuals ~950 -> ~342% each."""
        now = datetime.now(UTC)
        records = [
            DriftRecord(
                timestamp=(now - timedelta(hours=h)).isoformat(),
                predicted=4200.0,
                actual=950.0,
                abs_pct_error=abs(4200.0 - 950.0) / 950.0 * 100.0,
            )
            for h in range(1, 49)
        ]
        payload = {
            "region": "LDWP",
            "models": {"ensemble": {"records": serialize_records(records)}},
        }
        settled = {(now - timedelta(hours=h)).isoformat(): 4840.0 for h in range(1, 49)}
        return payload, settled, now

    def test_rolling_stats_land_on_the_settled_side(self):
        payload, settled, now = self._payload_with_partial_history()

        out = compute_drift_payload("LDWP", payload, {}, actuals=settled)

        block = out["models"]["ensemble"]
        assert block["rolling_mape_7d"] == pytest.approx(13.22, abs=0.05), (
            "regraded window should score prediction-vs-settled, not vs the partials"
        )
        stats = out["_regrade_stats"]
        assert stats["n_regraded"] == 48

    def test_regraded_records_exit_the_low_actual_exclusion(self):
        """A 950-actual record in a 4840-median window was low-actual-excluded;
        once re-graded to the real value it must re-enter the mean."""
        payload, settled, now = self._payload_with_partial_history()

        before = compute_drift_payload("LDWP", payload, {})
        after = compute_drift_payload("LDWP", payload, {}, actuals=settled)

        assert after["models"]["ensemble"]["n_7d"] >= before["models"]["ensemble"]["n_7d"]
        assert after["models"]["ensemble"]["n_low_actual_excluded_7d"] == 0

    def test_no_actuals_means_no_regrade(self):
        """Back-compat: callers without the actuals param get today's behavior."""
        payload, settled, now = self._payload_with_partial_history()
        out = compute_drift_payload("LDWP", payload, {})
        assert out["models"]["ensemble"]["rolling_mape_7d"] == pytest.approx(342.1, abs=0.5)
        assert out["_regrade_stats"] == {"n_regraded": 0}

    def test_horizon_payload_regrades_existing_records(self):
        from models.drift import compute_horizon_drift_payload

        now = datetime.now(UTC)
        ts = (now - timedelta(hours=2)).isoformat()
        record = DriftRecord(timestamp=ts, predicted=4200.0, actual=950.0, abs_pct_error=342.1)
        existing = {
            "models": {"ensemble": {"24h": {"records": serialize_records([record])}}},
            "pending": [],
        }
        out = compute_horizon_drift_payload(
            "LDWP", existing, None, {ts: 4840.0}, now_iso=now.isoformat()
        )
        block = out["models"]["ensemble"]["24h"]
        assert block["rolling_mape_7d"] == pytest.approx(13.22, abs=0.05)


class TestRegradePreservesLead:
    """#542: re-grading must not blank ``lead_hours``.

    The defect and its blast radius: ``regrade_records`` rebuilt a record
    without carrying the field, and ``filter_by_lead`` KEEPS unknown-lead
    records by design (they were assumed to be pre-field history). So every
    EIA revision quietly moved one more record past the P2-19 headline filter.
    Measured in production 2026-08-18T05:17Z, ensemble block: **2,275 of 2,880
    records (79%) unknown-lead** — IID 704/720 and SEC 682/720 against PJM
    443/720, the share tracking each BA's revision rate, which is what makes
    re-grading rather than pre-field history the only explanation.

    The sibling ``models.shadow_eval.regrade_records`` preserved the field
    deliberately from the day it shipped, and its test names this issue.
    """

    def _rec(self, ts: str, predicted: float, actual: float, lead: int | None) -> DriftRecord:
        return DriftRecord(
            timestamp=ts,
            predicted=predicted,
            actual=actual,
            abs_pct_error=abs(predicted - actual) / actual * 100.0,
            lead_hours=lead,
        )

    def test_lead_survives_the_rebuild(self):
        """The direct pin. A revision moves the actual; it cannot change how
        far ahead the prediction reached."""
        from models.drift import regrade_records

        rec = self._rec("2026-07-17T05:00:00+00:00", 4200.0, 967.0, 6)
        regraded, stats = regrade_records([rec], {"2026-07-17T05:00:00+00:00": 4840.0})

        assert stats["n_regraded"] == 1, "fixture must actually re-grade"
        assert regraded[0].actual == 4840.0, "the actual should move"
        assert regraded[0].lead_hours == 6, "...and the lead should not"

    def test_unknown_lead_stays_unknown(self):
        """The inverse: re-grading must not invent a lead either. A genuinely
        pre-field record has no recoverable lead and must keep saying so."""
        from models.drift import regrade_records

        rec = self._rec("2026-07-17T05:00:00+00:00", 4200.0, 967.0, None)
        regraded, _ = regrade_records([rec], {"2026-07-17T05:00:00+00:00": 4840.0})
        assert regraded[0].lead_hours is None

    def test_regraded_contaminating_lead_is_still_excluded(self):
        """The end-to-end case that would have caught this.

        Every fixture in ``TestPayloadRegrade`` is lead-less, so the whole
        re-grade path could blank the field without a single test noticing. A
        lead-6 record must stay out of the 1h headline *after* it re-grades,
        not just before.
        """
        now = datetime.now(UTC)
        ts_dirty = (now - timedelta(hours=2)).isoformat()
        ts_clean = (now - timedelta(hours=3)).isoformat()
        existing = {
            "models": {
                "ensemble": {
                    "records": [
                        # lead 6 with a huge error, scored against a partial
                        {"ts": ts_dirty, "p": 4200.0, "a": 950.0, "e": 342.1, "l": 6},
                        # lead 1, already settled — the honest half of the window
                        {"ts": ts_clean, "p": 4900.0, "a": 4840.0, "e": 1.24, "l": 1},
                    ]
                }
            }
        }
        # Both hours revise, so BOTH records go through the rebuild.
        settled = {ts_dirty: 4840.0, ts_clean: 4845.0}

        out = compute_drift_payload("PJM", existing, {}, actuals=settled, now_iso=now.isoformat())
        blk = out["models"]["ensemble"]

        assert out["_regrade_stats"]["n_regraded"] == 2, "fixture must re-grade both"
        assert blk["n_lead_excluded_7d"] == 1, "the lead-6 record must still be filtered"
        assert blk["n_lead_unknown_7d"] == 0, "re-grading must not manufacture unknowns"
        assert blk["n_7d"] == 1
        # 4900 vs 4845 = 1.135%. Had the lead-6 record leaked back in it would
        # have dragged this to ~7% (its own re-graded error is ~13.2%).
        assert blk["rolling_mape_7d"] == pytest.approx(1.135, abs=0.01)

    def test_horizon_records_stay_lead_free_across_regrade(self):
        """The no-op half, pinned rather than asserted.

        ``resolve_horizon_snapshots`` never sets a lead and
        ``_horizon_rollup_block`` never lead-filters — 24/48/72h records have a
        DESIGNED horizon. So this fix must not change the horizon path at all,
        which is what keeps ``benchmark.serve_grade`` off the blast radius.
        """
        from models.drift import compute_horizon_drift_payload

        now = datetime.now(UTC)
        ts = (now - timedelta(hours=2)).isoformat()
        record = DriftRecord(timestamp=ts, predicted=4200.0, actual=950.0, abs_pct_error=342.1)
        assert record.lead_hours is None
        existing = {
            "models": {"ensemble": {"24h": {"records": serialize_records([record])}}},
            "pending": [],
        }
        out = compute_horizon_drift_payload(
            "LGEE", existing, None, {ts: 4840.0}, now_iso=now.isoformat()
        )
        block = out["models"]["ensemble"]["24h"]
        assert block["rolling_mape_7d"] == pytest.approx(13.22, abs=0.05), "still re-grades"
        assert all("l" not in row for row in block["records"]), "and still carries no lead"
        assert "n_lead_excluded_7d" not in block, "horizon blocks do not lead-filter"


class TestAnchorProvenance:
    """#547: what each forecast was seeded with, carried onto its records.

    ``docs/BENCHMARK_METHODOLOGY.md`` limit 11 could only state its own
    materiality as unmeasured, because nothing recorded which forecasts
    anchored on EIA's own day-ahead value. These tests pin the two properties
    that make the instrument trustworthy: it survives every rewrite of a
    record, and an absent value never hardens into a claim.
    """

    ANCHOR = {
        "anchor_ts": "2026-08-18T03:00:00+00:00",
        "anchor_was_placeholder": True,
        "anchor_conditioned": False,
    }

    def _rec(self, **kw) -> DriftRecord:
        base = {
            "timestamp": "2026-08-18T06:00:00+00:00",
            "predicted": 4200.0,
            "actual": 4000.0,
            "abs_pct_error": 5.0,
        }
        return DriftRecord(**{**base, **kw})

    # ── the wire ────────────────────────────────────────────────────────────

    def test_round_trip_carries_all_three(self):
        from models.drift import deserialize_records, serialize_records

        out = deserialize_records(serialize_records([self._rec(**self.ANCHOR)]))[0]

        assert out.anchor_ts == self.ANCHOR["anchor_ts"]
        assert out.anchor_was_placeholder is True
        assert out.anchor_conditioned is False

    def test_absent_keys_deserialize_to_none_not_false(self):
        """A record written before this landed has an UNKNOWN anchor.

        ``False`` would be a claim that its anchor was metered — the tri-state
        convention ``served_series`` pins (#348) and ``lead_hours`` follows.
        """
        from models.drift import deserialize_records, serialize_records

        wire = serialize_records([self._rec()])
        assert not ({"at", "ap", "ac"} & set(wire[0])), "unknown must be omitted, not written"

        out = deserialize_records(wire)[0]
        assert out.anchor_ts is None
        assert out.anchor_was_placeholder is None
        assert out.anchor_conditioned is None

    def test_a_false_flag_is_written_and_read_back_as_false(self):
        """The other half: a CONFIRMED negative must survive as a negative.

        Omit-when-unknown is only safe if ``False`` is distinguishable from
        absent on the wire.
        """
        from models.drift import deserialize_records, serialize_records

        wire = serialize_records([self._rec(anchor_was_placeholder=False)])
        assert wire[0]["ap"] is False

        assert deserialize_records(wire)[0].anchor_was_placeholder is False

    def test_nan_shaped_values_read_as_unknown_not_as_true(self):
        """The parquet fork, and the reason the booleans get their own parser.

        Redis JSON omits an absent key; a parquet mirror materialises the
        column and hands back float ``NaN``. ``bool(float("nan"))`` is
        ``True`` and ``str(float("nan"))`` is the literal ``"nan"``, so a naive
        read turns "we never asked" into a positive claim. That asymmetry
        misread 713 of 719 BANC vintage records after #535, invisibly, because
        production reads Redis.

        The drift window is Redis-only today, so this guards the day someone
        mirrors it — which is the same day nobody re-checks this.
        """
        from models.drift import deserialize_records

        row = {
            "ts": "2026-08-18T06:00:00+00:00",
            "p": 4200.0,
            "a": 4000.0,
            "e": 5.0,
            "at": float("nan"),
            "ap": float("nan"),
            "ac": float("nan"),
        }
        out = deserialize_records([row])[0]

        assert out.anchor_ts is None, 'str(nan) is the literal "nan", which parses nowhere'
        assert out.anchor_was_placeholder is None, "bool(nan) is True — the silent flip"
        assert out.anchor_conditioned is None

    # ── the #542 shape: survives a rewrite, not just a construct-and-read ───

    def test_anchor_survives_regrading(self):
        """The #542 regression, with new fields.

        ``regrade_records`` rebuilt a record on every revision and omitted
        ``lead_hours``, blanking it on 2,275 of 2,880 records (79%) — tracking
        each BA's revision rate, and silent. The semantic rule that defect
        established decides these fields too: values DERIVED from the pair that
        moved are recomputed (sMAPE), properties of the OBSERVATION are carried
        (``lead_hours``, and the anchor — a revision to the actual cannot
        change what the forecast was seeded from).
        """
        from models.drift import regrade_records

        rec = self._rec(lead_hours=3, **self.ANCHOR)
        old_smape = rec.smape

        regraded, _ = regrade_records([rec], {"2026-08-18T06:00:00+00:00": 4100.0})
        out = regraded[0]

        assert out.actual == 4100.0, "the revision landed"
        assert np.isfinite(out.smape) and out.smape != old_smape, "sMAPE IS derived — recomputed"
        assert out.lead_hours == 3, "#542: carried, not blanked"
        assert out.anchor_ts == self.ANCHOR["anchor_ts"]
        assert out.anchor_was_placeholder is True
        assert out.anchor_conditioned is False

    def test_anchor_survives_regrade_then_serialization(self):
        """Both rewrite sites in one pass — the full production round trip.

        Each tick re-grades the window and then re-serialises it, so a field
        that survives one site and not the other still decays to unknown over
        time. Construct-and-read cannot see that.
        """
        from models.drift import deserialize_records, regrade_records, serialize_records

        records = [self._rec(lead_hours=1, **self.ANCHOR)]
        for actual in (4100.0, 4150.0, 4180.0):
            records, _ = regrade_records(records, {"2026-08-18T06:00:00+00:00": actual})
            records = deserialize_records(serialize_records(records))

        out = records[0]
        assert out.anchor_ts == self.ANCHOR["anchor_ts"]
        assert out.anchor_was_placeholder is True
        assert out.anchor_conditioned is False
        assert out.lead_hours == 1

    # ── producers ───────────────────────────────────────────────────────────

    def test_reads_the_anchor_block_off_a_forecast_payload(self):
        from models.drift import anchor_provenance

        out = anchor_provenance({"anchor": {**self.ANCHOR, "anchor_mw": 4123.45}})

        assert out == self.ANCHOR, "anchor_mw is deliberately not a record field"

    def test_a_payload_without_an_anchor_block_yields_unknown(self):
        from models.drift import anchor_provenance

        for payload in (None, {}, {"anchor": None}, {"anchor": "junk"}):
            assert anchor_provenance(payload) == {
                "anchor_ts": None,
                "anchor_was_placeholder": None,
                "anchor_conditioned": None,
            }

    def test_one_hour_ahead_records_carry_the_anchor(self):
        from models.drift import build_records_from_actuals

        payload = {
            "scored_at": "2026-08-18T04:05:00+00:00",
            "anchor": self.ANCHOR,
            "forecasts": [
                {"timestamp": "2026-08-18T04:00:00+00:00", "xgboost": 4200.0},
            ],
        }
        out = build_records_from_actuals(payload, {"2026-08-18T04:00:00+00:00": 4000.0})

        assert out["xgboost"].anchor_was_placeholder is True
        assert out["xgboost"].anchor_ts == self.ANCHOR["anchor_ts"]

    def test_the_horizon_anchor_rides_the_snapshot_across_resolution(self):
        """It MUST ride the pending snapshot, not be looked up at resolution.

        A 24h snapshot resolves a day after it is taken, by which point the
        forecast payload that produced it has been overwritten ~24 times. Same
        "cannot be recomputed" argument as ``lead_hours``.
        """
        from models.drift import resolve_horizon_snapshots, snapshot_horizon_predictions

        payload = {
            "scored_at": "2026-08-18T04:05:00+00:00",
            "anchor": self.ANCHOR,
            "forecasts": [
                {"timestamp": f"2026-08-18T{h:02d}:00:00+00:00", "xgboost": 4200.0}
                for h in range(4, 24)
            ]
            + [
                {"timestamp": f"2026-08-19T{h:02d}:00:00+00:00", "xgboost": 4200.0}
                for h in range(0, 8)
            ],
        }
        pending = snapshot_horizon_predictions(payload, horizons=("24h",))
        assert pending and pending[0]["anchor"] == self.ANCHOR, "carried onto the snapshot"

        resolved, _ = resolve_horizon_snapshots(pending, {pending[0]["target_ts"]: 4000.0})
        _, _, record = resolved[0]
        assert record.anchor_was_placeholder is True
        assert record.anchor_ts == self.ANCHOR["anchor_ts"]

    def test_a_snapshot_taken_before_the_field_existed_resolves_to_unknown(self):
        from models.drift import resolve_horizon_snapshots

        pending = [
            {
                "target_ts": "2026-08-19T04:00:00+00:00",
                "made_at": "2026-08-18T04:00:00+00:00",
                "horizon": "24h",
                "preds": {"xgboost": 4200.0},
            }
        ]
        resolved, _ = resolve_horizon_snapshots(pending, {"2026-08-19T04:00:00+00:00": 4000.0})

        assert resolved[0][2].anchor_was_placeholder is None

    # ── the accrual signal ──────────────────────────────────────────────────

    def test_counters_separate_unknown_from_metered(self):
        """The distinction the counter exists to make.

        Pooling ``None`` with ``False`` would report a fully-instrumented
        fleet the moment the field shipped.
        """
        from models.drift import count_anchor_provenance

        records = [
            self._rec(anchor_was_placeholder=True),
            self._rec(anchor_was_placeholder=True),
            self._rec(anchor_was_placeholder=False),
            self._rec(),
        ]
        assert count_anchor_provenance(records) == (2, 1)
