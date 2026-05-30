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
