"""Unit tests for the #227 horizon-matched drift pipeline in models/drift.py.

The 1-hour drift path is covered by test_drift.py / test_drift_panel.py; this
file targets the snapshot -> resolve -> grade pipeline that lets the multi-step
models (Prophet/SARIMAX) be judged at the horizon they're built for instead of
being condemned by a 1h metric.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from models.drift import (
    HORIZON_DRIFT_HORIZONS,
    _expire_pending,
    compute_horizon_drift_payload,
    resolve_horizon_snapshots,
    snapshot_horizon_predictions,
)

T0 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)


def _forecast(scored_at: datetime, hours: int = 80, start: datetime | None = None) -> dict:
    """A forecast payload whose first (hour-aligned) row is ``start`` (default
    ``scored_at``) — the origin horizons are measured from."""
    origin = start or scored_at
    rows = [
        {
            "timestamp": (origin + timedelta(hours=h)).isoformat(),
            "predicted_demand_mw": 40_000.0,
            "xgboost": 40_000.0,
            "prophet": 41_000.0,
            "arima": 39_000.0,
            "ensemble": 40_100.0,
        }
        for h in range(hours)
    ]
    return {"region": "ERCOT", "scored_at": scored_at.isoformat(), "forecasts": rows}


class TestSnapshotHorizonPredictions:
    def test_one_snapshot_per_horizon(self):
        snaps = snapshot_horizon_predictions(_forecast(T0))
        assert len(snaps) == len(HORIZON_DRIFT_HORIZONS)
        by_h = {s["horizon"]: s for s in snaps}
        # 24h snapshot targets scored_at + 24h, carries all 4 models.
        assert by_h["24h"]["target_ts"] == (T0 + timedelta(hours=24)).isoformat()
        assert set(by_h["24h"]["preds"]) == {"xgboost", "prophet", "arima", "ensemble"}
        assert by_h["72h"]["target_ts"] == (T0 + timedelta(hours=72)).isoformat()

    def test_missing_scored_at_returns_empty(self):
        assert snapshot_horizon_predictions({"forecasts": []}) == []
        assert snapshot_horizon_predictions(None) == []

    def test_short_forecast_skips_unreachable_horizons(self):
        # Only 30 forward rows -> 24h reachable, 48h/72h are not.
        snaps = snapshot_horizon_predictions(_forecast(T0, hours=30))
        assert {s["horizon"] for s in snaps} == {"24h"}

    def test_uses_hour_aligned_row_not_wallclock_scored_at(self):
        # Production ``scored_at`` is ``datetime.now()`` with sub-hour precision;
        # forecast rows are on the hour. Horizons must key off the first row or
        # they'd never match a row — a silent no-op. This fails on the naive
        # scored_at-based origin.
        fc = _forecast(T0, hours=80)  # rows on the hour from T0
        fc["scored_at"] = (T0 + timedelta(minutes=37, seconds=12)).isoformat()
        snaps = snapshot_horizon_predictions(fc)
        assert len(snaps) == 3  # NOT zero
        by_h = {s["horizon"]: s for s in snaps}
        assert by_h["24h"]["target_ts"] == (T0 + timedelta(hours=24)).isoformat()


class TestResolveHorizonSnapshots:
    def _snap(self, target, horizon="24h"):
        return {
            "target_ts": target.isoformat(),
            "made_at": (target - timedelta(hours=24)).isoformat(),
            "horizon": horizon,
            "preds": {"xgboost": 40_000.0, "prophet": 41_000.0},
        }

    def test_resolves_when_actual_known(self):
        target = T0 + timedelta(hours=24)
        pending = [self._snap(target)]
        resolved, still = resolve_horizon_snapshots(pending, {target.isoformat(): 40_200.0})
        assert still == []
        assert len(resolved) == 2  # one per model
        models = {m for m, _, _ in resolved}
        assert models == {"xgboost", "prophet"}
        # the record carries the horizon + a finite abs pct error
        _, horizon, rec = resolved[0]
        assert horizon == "24h"
        assert rec.actual == 40_200.0

    def test_keeps_pending_when_actual_absent(self):
        target = T0 + timedelta(hours=24)
        resolved, still = resolve_horizon_snapshots([self._snap(target)], {})
        assert resolved == []
        assert len(still) == 1

    def test_skips_zero_or_negative_actual(self):
        target = T0 + timedelta(hours=24)
        resolved, still = resolve_horizon_snapshots([self._snap(target)], {target.isoformat(): 0.0})
        assert resolved == []
        assert len(still) == 1  # not resolved, stays pending


class TestExpirePending:
    def test_drops_stale_keeps_fresh(self):
        stale = {"target_ts": (T0 - timedelta(hours=200)).isoformat(), "horizon": "24h"}
        fresh = {"target_ts": (T0 + timedelta(hours=24)).isoformat(), "horizon": "24h"}
        out = _expire_pending([stale, fresh], now_iso=T0.isoformat())
        assert out == [fresh]

    def test_drops_malformed(self):
        assert (
            _expire_pending([{"target_ts": "not-a-date", "horizon": "24h"}], T0.isoformat()) == []
        )


class TestComputeHorizonDriftPayload:
    def _actuals(self, now: datetime, value: float = 40_200.0) -> dict:
        return {
            (T0 + timedelta(hours=h)).isoformat(): value
            for h in range(0, 200)
            if T0 + timedelta(hours=h) <= now
        }

    def test_first_tick_snapshots_no_records(self):
        p = compute_horizon_drift_payload("ERCOT", None, _forecast(T0), {}, now_iso=T0.isoformat())
        assert len(p["pending"]) == 3
        assert p["models"] == {}
        assert p["horizons"] == list(HORIZON_DRIFT_HORIZONS)

    def test_resolution_produces_horizon_graded_record(self):
        # Tick 1 at T0 → snapshots. Tick 2 at T0+24 → the 24h snapshot resolves.
        p = compute_horizon_drift_payload("ERCOT", None, _forecast(T0), {}, now_iso=T0.isoformat())
        now2 = T0 + timedelta(hours=24)
        p = compute_horizon_drift_payload(
            "ERCOT", p, _forecast(now2), self._actuals(now2), now_iso=now2.isoformat()
        )
        xg = p["models"]["xgboost"]["24h"]
        pr = p["models"]["prophet"]["24h"]
        assert xg["n_records"] == 1
        # THE POINT OF #227: prophet's 24h error (~2%) grades against the 24h
        # band, not the 1h band — a competent day-ahead model isn't condemned.
        assert pr["rolling_mape_7d"] < 2.5
        assert pr["grade"] in ("excellent", "target")
        assert xg["grade"] == "excellent"

    def test_same_target_gets_both_horizons(self):
        # T0+48 is the 48h target of the T0 forecast AND the 24h target of the
        # T0+24 forecast — both must resolve into their own series. Feed ONLY
        # T0+48's actual so exactly those two snapshots mature.
        target48 = (T0 + timedelta(hours=48)).isoformat()
        p = compute_horizon_drift_payload("ERCOT", None, _forecast(T0), {}, now_iso=T0.isoformat())
        p = compute_horizon_drift_payload(
            "ERCOT",
            p,
            _forecast(T0 + timedelta(hours=24)),
            {},
            now_iso=(T0 + timedelta(hours=24)).isoformat(),
        )
        now3 = T0 + timedelta(hours=48)
        p = compute_horizon_drift_payload(
            "ERCOT", p, _forecast(now3), {target48: 40_200.0}, now_iso=now3.isoformat()
        )
        assert p["models"]["xgboost"]["48h"]["n_records"] == 1  # T0+48 as a 48h prediction
        assert p["models"]["xgboost"]["24h"]["n_records"] == 1  # T0+48 as a 24h prediction

    def test_dedup_on_retried_tick(self):
        # Re-running the same tick must not double-count the pending snapshots.
        p1 = compute_horizon_drift_payload("ERCOT", None, _forecast(T0), {}, now_iso=T0.isoformat())
        p2 = compute_horizon_drift_payload("ERCOT", p1, _forecast(T0), {}, now_iso=T0.isoformat())
        assert len(p2["pending"]) == len(p1["pending"]) == 3
