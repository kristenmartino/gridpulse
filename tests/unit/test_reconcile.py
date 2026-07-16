"""Check A — does the Live Drift number match settled reality? (#304)

Fixtures are built from the REAL prod measurements taken 2026-07-15, so these
tests fail if the checker would miss the bugs that actually shipped:

    LDWP  70.7% mean revision   panel 147.91   settled  53.22
    AZPS  66.7% mean revision   panel 338.67   settled  15.64   <- 21x overstated
    PSCO  14.3% mean revision   panel   6.28   settled  13.26   <- UNDERSTATED
    BPAT  14.2% mean revision   panel  10.72   settled   8.67
    PNM    0.7% mean revision   panel   2.06   settled   1.82   <- clean

corr(revision, settled error) = 0.88 across all 51 BAs.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from models.drift import DriftRecord, serialize_records
from reconcile import (
    ReconcileFinding,
    check_drift_against_settled,
    recompute_settled_mape,
    settled_actuals_from_demand,
)

NOW = datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC)


def _ts(hours_ago: int) -> str:
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def _rec(hours_ago: int, predicted: float, preliminary_actual: float) -> DriftRecord:
    """A stored record: scored at tick time against the PRELIMINARY actual."""
    return DriftRecord(
        timestamp=_ts(hours_ago),
        predicted=predicted,
        actual=preliminary_actual,
        abs_pct_error=abs(predicted - preliminary_actual) / preliminary_actual * 100.0,
    )


def _payload(records: list[DriftRecord], *, rolling_mape_7d: float | None) -> dict:
    block: dict = {"records": serialize_records(records)}
    if rolling_mape_7d is not None:
        block["rolling_mape_7d"] = rolling_mape_7d
    return {"region": "TEST", "models": {"xgboost": block}}


class TestSettledActualsFromDemand:
    def test_builds_hour_map(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-07-15", periods=3, freq="h", tz="UTC"),
                "demand_mw": [8000.0, 8100.0, 8200.0],
            }
        )
        out = settled_actuals_from_demand(df)
        assert len(out) == 3
        assert set(out.values()) == {8000.0, 8100.0, 8200.0}

    def test_drops_unusable_rows(self):
        """Zero/NaN/negative can't be a percentage denominator — and a zero
        demand row is an EIA artifact, not a reading."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-07-15", periods=4, freq="h", tz="UTC"),
                "demand_mw": [8000.0, 0.0, float("nan"), -50.0],
            }
        )
        assert len(settled_actuals_from_demand(df)) == 1

    def test_degenerate_inputs_safe(self):
        assert settled_actuals_from_demand(None) == {}
        assert settled_actuals_from_demand(pd.DataFrame()) == {}
        assert settled_actuals_from_demand(pd.DataFrame({"nope": [1]})) == {}


class TestRecomputeSettledMape:
    def test_rescores_against_settled_not_preliminary(self):
        """The core move: same stored prediction, scored against settled truth.

        Prediction 8000 was scored at tick time against a preliminary 7200
        (=11.1% stored error). Settled truth is 8000 — the forecast was exact.
        """
        records = [_rec(h, predicted=8000.0, preliminary_actual=7200.0) for h in range(48)]
        settled = {_ts(h): 8000.0 for h in range(48)}

        mape, revision, n = recompute_settled_mape(records, settled, now=NOW)

        assert n == 48
        assert mape == pytest.approx(0.0, abs=1e-6)
        assert revision == pytest.approx(10.0, abs=1e-6)  # 7200 -> 8000

    def test_windows_to_7d_independently(self):
        """Records outside the 7d window must not enter the mean. Independently
        implemented — the n_records vs n_7d trap lives exactly here."""
        recent = [_rec(h, 8000.0, 8000.0) for h in range(24)]
        ancient = [_rec(24 * 20 + h, 5000.0, 8000.0) for h in range(24)]  # 20d ago
        settled = {_ts(h): 8000.0 for h in range(24)} | {
            _ts(24 * 20 + h): 8000.0 for h in range(24)
        }

        mape, _, n = recompute_settled_mape(recent + ancient, settled, now=NOW)

        assert n == 24  # ancient excluded
        assert mape == pytest.approx(0.0, abs=1e-6)

    def test_hours_missing_from_settled_are_skipped_not_counted(self):
        """GCS is written fire-and-forget and can lag Redis — an absent hour is
        unknown, never agreement."""
        records = [_rec(h, 8000.0, 7200.0) for h in range(48)]
        settled = {_ts(h): 8000.0 for h in range(10)}  # only 10 hours settled

        _, _, n = recompute_settled_mape(records, settled, now=NOW)
        assert n == 10

    def test_no_overlap_returns_none(self):
        records = [_rec(h, 8000.0, 8000.0) for h in range(48)]
        mape, revision, n = recompute_settled_mape(records, {}, now=NOW)
        assert (mape, revision, n) == (None, None, 0)

    def test_low_actual_outliers_dropped(self):
        """Mirrors the producer's region-relative rule so the two means compare
        like with like."""
        records = [_rec(h, 8000.0, 8000.0) for h in range(24)]
        records += [_rec(30 + i, 8000.0, 8000.0) for i in range(3)]
        settled = {_ts(h): 8000.0 for h in range(24)}
        settled |= {_ts(30 + i): 1.0 for i in range(3)}  # near-zero artifacts

        mape, _, n = recompute_settled_mape(records, settled, now=NOW)

        assert n == 24  # the three 1 MW hours dropped
        assert mape == pytest.approx(0.0, abs=1e-6)

    def test_rounding_tolerance(self):
        """serialize_records rounds p/a to 2dp — recomputation must be close,
        never exactly equal, and must not drift materially."""
        records = deserialize = [_rec(h, 8123.456, 8000.123) for h in range(48)]
        settled = {_ts(h): 8000.123 for h in range(48)}
        mape, _, _ = recompute_settled_mape(records, settled, now=NOW)
        assert mape == pytest.approx(1.5416, abs=0.01)
        assert deserialize is records  # keep ruff quiet about the alias


class TestCheckADivergence:
    """The prod cases this must catch, by name."""

    def test_azps_overstatement_fires(self):
        """AZPS: panel 338.67 vs settled 15.64 — 21x overstated."""
        records = [_rec(h, 8000.0, 2000.0) for h in range(48)]
        settled = {_ts(h): 8000.0 for h in range(48)}
        findings = check_drift_against_settled(
            "AZPS", _payload(records, rolling_mape_7d=338.67), settled, now=NOW
        )
        f = findings[0]
        assert f.ok is False
        assert "displayed_overstates" in f.reasons
        assert "high_eia_revision" in f.reasons
        assert f.settled_mape == pytest.approx(0.0, abs=1e-6)

    def test_psco_understatement_fires(self):
        """PSCO: panel 6.28 vs settled 13.26 — the panel makes the model look
        HEALTHIER than it is. The dangerous half."""
        # Predictions ~13.26% from settled truth.
        records = [_rec(h, 8000.0 * 1.1326, 8000.0) for h in range(48)]
        settled = {_ts(h): 8000.0 for h in range(48)}
        findings = check_drift_against_settled(
            "PSCO", _payload(records, rolling_mape_7d=6.28), settled, now=NOW
        )
        f = findings[0]
        assert f.ok is False
        assert "displayed_understates" in f.reasons
        assert f.settled_mape == pytest.approx(13.26, abs=0.05)
        assert f.divergence_pct_points == pytest.approx(6.98, abs=0.05)

    def test_pnm_clean_does_not_fire(self):
        """PNM: 0.7% revision, panel 2.06 vs settled 1.82 — must stay silent.
        Zero false positives on healthy BAs is the whole cry-wolf lesson."""
        records = [_rec(h, 8000.0 * 1.0182, 8000.0 * 1.0007) for h in range(48)]
        settled = {_ts(h): 8000.0 for h in range(48)}
        findings = check_drift_against_settled(
            "PNM", _payload(records, rolling_mape_7d=1.82), settled, now=NOW
        )
        f = findings[0]
        assert f.ok is True
        assert f.reasons == []

    def test_high_revision_fires_even_when_metric_agrees(self):
        """A2 is its own signal: revision is the upstream cause (corr 0.88) and
        must surface even if the displayed number happens to match."""
        records = [_rec(h, 8000.0, 6800.0) for h in range(48)]  # ~17.6% revision
        settled = {_ts(h): 8000.0 for h in range(48)}
        findings = check_drift_against_settled(
            "BPAT", _payload(records, rolling_mape_7d=0.0), settled, now=NOW
        )
        f = findings[0]
        assert f.ok is False
        assert f.reasons == ["high_eia_revision"]

    def test_insufficient_overlap_skips_rather_than_passes(self):
        """Declining to judge must be explicit — never a silent pass."""
        records = [_rec(h, 8000.0, 7200.0) for h in range(48)]
        settled = {_ts(h): 8000.0 for h in range(5)}
        f = check_drift_against_settled(
            "TEST", _payload(records, rolling_mape_7d=11.1), settled, now=NOW
        )[0]
        assert f.skipped == "insufficient_settled_overlap"
        assert f.ok is True

    def test_missing_displayed_metric_still_reports_revision(self):
        records = [_rec(h, 8000.0, 8000.0) for h in range(48)]
        settled = {_ts(h): 8000.0 for h in range(48)}
        f = check_drift_against_settled(
            "TEST", _payload(records, rolling_mape_7d=None), settled, now=NOW
        )[0]
        assert f.displayed_mape is None
        assert f.divergence_pct_points is None
        assert f.ok is True

    def test_per_model_findings(self):
        records = [_rec(h, 8000.0, 8000.0) for h in range(48)]
        settled = {_ts(h): 8000.0 for h in range(48)}
        payload = {
            "models": {
                m: {"records": serialize_records(records), "rolling_mape_7d": 0.0}
                for m in ("xgboost", "prophet", "arima", "ensemble")
            }
        }
        findings = check_drift_against_settled("TEST", payload, settled, now=NOW)
        assert {f.model for f in findings} == {"xgboost", "prophet", "arima", "ensemble"}

    def test_degenerate_payloads_safe(self):
        assert check_drift_against_settled("T", None, {}, now=NOW) == []
        assert check_drift_against_settled("T", {"models": "corrupt"}, {}, now=NOW) == []
        assert check_drift_against_settled("T", {"models": {"x": "bad"}}, {}, now=NOW) == []
        f = check_drift_against_settled("T", {"models": {"x": {}}}, {}, now=NOW)[0]
        assert f.skipped == "no_records"


class TestLogFields:
    def test_flat_scalars_for_structlog(self):
        f = ReconcileFinding(
            region="AZPS",
            model="xgboost",
            ok=False,
            displayed_mape=338.67,
            settled_mape=15.64,
            divergence_pct_points=323.03,
            mean_abs_revision_pct=66.66,
            n_compared=720,
            reasons=["displayed_overstates", "high_eia_revision"],
        )
        fields = f.as_log_fields()
        assert fields["region"] == "AZPS"
        assert fields["displayed_mape"] == 338.67
        assert fields["settled_mape"] == 15.64
        assert fields["reasons"] == "displayed_overstates,high_eia_revision"
        assert all(isinstance(v, str | int | float) for v in fields.values())

    def test_none_fields_omitted(self):
        fields = ReconcileFinding(
            region="T", model="m", ok=True, skipped="no_records"
        ).as_log_fields()
        assert "displayed_mape" not in fields
        assert fields["skipped"] == "no_records"


class TestIndependenceContract:
    """The checker must stay independent of the code it checks (#217 trap).

    reconcile.py's whole value rests on re-implementing windowing and
    aggregation rather than importing the producer's — a checker that reuses
    ``compute_drift_payload``/``_within_window``/``filter_low_actuals`` would
    agree with the panel *by construction*, hiding the very bug class it exists
    to catch, and **every other test here would still pass** (the numbers would
    match). Nothing but this test guards that property, so a well-meaning future
    "DRY up the duplication" refactor could silently re-couple them. This pins
    the allowed surface at the import boundary (cf. the ``_FORBID_V1`` guard in
    the #220 Models-tab tests).
    """

    #: The ONLY names reconcile.py may borrow from the serving path: leaf math +
    #: parsing. Aggregation/windowing/filtering are deliberately re-implemented.
    ALLOWED_FROM_DRIFT = {"_normalize_ts", "absolute_pct_error", "deserialize_records"}

    def _drift_imports(self) -> set[str]:
        import ast
        from pathlib import Path

        source = Path(__file__).resolve().parents[2] / "reconcile.py"
        tree = ast.parse(source.read_text(), filename=str(source))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "models.drift":
                names.update(alias.name for alias in node.names)
            # A bare ``import models.drift`` would let any attribute be reached
            # at call sites, bypassing this allowlist — forbid it outright.
            if isinstance(node, ast.Import):
                assert all(alias.name != "models.drift" for alias in node.names), (
                    "reconcile.py must not `import models.drift` wholesale — use named leaf imports"
                )
        return names

    def test_only_leaf_helpers_borrowed_from_producer(self):
        borrowed = self._drift_imports()
        extra = borrowed - self.ALLOWED_FROM_DRIFT
        assert not extra, (
            f"reconcile.py imports {sorted(extra)} from models.drift — the checker "
            "must re-implement aggregation/windowing/filtering itself, not reuse the "
            "code it checks (#217 circular-verdict trap)."
        )

    def test_aggregation_functions_never_imported(self):
        """Name the specific producer functions whose reuse would re-couple."""
        forbidden = {
            "compute_drift_payload",
            "_within_window",
            "filter_low_actuals",
        }
        assert not (self._drift_imports() & forbidden)
