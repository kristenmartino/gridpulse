"""#304: the EIA-revision probe for the Live Drift panel.

Context: BPAT's panel reports 11.7% live 1h-ahead MAPE, but replaying the real
production serving code against settled data measures 0.58% — a ~20x gap, and
the panel false-flags Rollback on four healthy models. The error is flat across
every lead (1h ~= 24h ~= 72h ~= 12%), which is the signature of the value being
scored *against* being wrong, not the prediction.

Each stored ``DriftRecord.actual`` is the PRELIMINARY value EIA had published
when that record was created. This probe diffs those against today's settled
values and rescores the same predictions, so one scoring tick answers whether
revisions explain the gap.
"""

from __future__ import annotations

from models.drift import DriftRecord, probe_actual_revisions, serialize_records


def _payload(records_by_model: dict[str, list[DriftRecord]]) -> dict:
    return {
        "region": "BPAT",
        "models": {m: {"records": serialize_records(r)} for m, r in records_by_model.items()},
    }


def _rec(ts: str, predicted: float, actual: float) -> DriftRecord:
    return DriftRecord(
        timestamp=ts,
        predicted=predicted,
        actual=actual,
        abs_pct_error=abs(predicted - actual) / actual * 100.0,
    )


class TestProbeActualRevisions:
    def test_revision_explains_the_gap(self):
        """The #304 hypothesis: predictions were accurate against SETTLED data,
        but were scored at tick time against preliminary actuals that EIA later
        revised ~10% — so stored_mape is inflated while settled_mape is tiny."""
        # Prediction ~= settled truth (8000); preliminary actual was 7200.
        records = [_rec(f"2026-07-14T{h:02d}:00:00+00:00", 8000.0, 7200.0) for h in range(5)]
        settled = {f"2026-07-14T{h:02d}:00:00+00:00": 8000.0 for h in range(5)}

        out = probe_actual_revisions(_payload({"xgboost": records}), settled)

        s = out["xgboost"]
        assert s["n_compared"] == 5
        assert s["n_revised"] == 5
        assert s["mean_abs_revision_pct"] == 10.0  # 7200 -> 8000 settled
        # The panel's number vs the honest one: this is the smoking gun.
        assert s["stored_mape"] > 11.0
        assert s["settled_mape"] == 0.0

    def test_no_revision_is_a_clean_bill(self):
        """When preliminary == settled, the probe exonerates revisions and the
        stored MAPE stands as real forecast error."""
        records = [_rec(f"2026-07-14T{h:02d}:00:00+00:00", 8100.0, 8000.0) for h in range(4)]
        settled = {f"2026-07-14T{h:02d}:00:00+00:00": 8000.0 for h in range(4)}

        s = probe_actual_revisions(_payload({"xgboost": records}), settled)["xgboost"]

        assert s["n_revised"] == 0
        assert s["mean_abs_revision_pct"] == 0.0
        assert s["stored_mape"] == s["settled_mape"] == 1.25

    def test_only_hours_present_in_both_are_compared(self):
        """Records whose hour has aged out of the fetched window are skipped —
        never counted as a zero-revision."""
        records = [_rec(f"2026-07-14T{h:02d}:00:00+00:00", 8000.0, 7200.0) for h in range(5)]
        settled = {"2026-07-14T00:00:00+00:00": 8000.0}  # only one hour still fetched

        s = probe_actual_revisions(_payload({"xgboost": records}), settled)["xgboost"]

        assert s["n_compared"] == 1

    def test_per_model_blocks(self):
        recs = {
            "xgboost": [_rec("2026-07-14T00:00:00+00:00", 8000.0, 7200.0)],
            "prophet": [_rec("2026-07-14T00:00:00+00:00", 7900.0, 7200.0)],
        }
        out = probe_actual_revisions(_payload(recs), {"2026-07-14T00:00:00+00:00": 8000.0})
        assert set(out) == {"xgboost", "prophet"}

    def test_timestamp_z_suffix_normalizes(self):
        """A settled key written with a Z suffix must still match a record
        stored with +00:00 (the drift path normalizes both)."""
        records = [_rec("2026-07-14T00:00:00+00:00", 8000.0, 7200.0)]
        out = probe_actual_revisions(
            _payload({"xgboost": records}), {"2026-07-14T00:00:00Z": 8000.0}
        )
        assert out["xgboost"]["n_compared"] == 1

    def test_degenerate_inputs_are_safe(self):
        assert probe_actual_revisions(None, {"x": 1.0}) == {}
        assert probe_actual_revisions({"models": {}}, {}) == {}
        assert probe_actual_revisions({"models": {"xgboost": "corrupt"}}, {"t": 1.0}) == {}

    def test_nonpositive_and_nonfinite_actuals_skipped(self):
        """A zero/negative settled value can't form a percentage — skip rather
        than divide by ~0 and report a fake giant revision."""
        records = [
            _rec("2026-07-14T00:00:00+00:00", 8000.0, 7200.0),
            _rec("2026-07-14T01:00:00+00:00", 8000.0, 7200.0),
        ]
        settled = {
            "2026-07-14T00:00:00+00:00": 0.0,  # unusable
            "2026-07-14T01:00:00+00:00": 8000.0,  # usable
        }
        s = probe_actual_revisions(_payload({"xgboost": records}), settled)["xgboost"]
        assert s["n_compared"] == 1
