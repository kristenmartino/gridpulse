"""Unit tests for the #227 horizon-matched drift pipeline in models/drift.py.

The 1-hour drift path is covered by test_drift.py / test_drift_panel.py; this
file targets the snapshot -> resolve -> grade pipeline that lets the multi-step
models (Prophet/SARIMAX) be judged at the horizon they're built for instead of
being condemned by a 1h metric.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta

from models.drift import (
    HORIZON_DRIFT_HORIZONS,
    _expire_pending,
    compute_horizon_drift_payload,
    extract_one_hour_ahead_predictions,
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
        out, expired_by_horizon, malformed_by_horizon = _expire_pending(
            [stale, fresh], now_iso=T0.isoformat()
        )
        assert out == [fresh]
        # #537 channel 2: the stale drop is counted, keyed by its horizon.
        assert expired_by_horizon == {"24h": 1}
        assert malformed_by_horizon == {}

    def test_drops_malformed(self):
        out, expired_by_horizon, malformed_by_horizon = _expire_pending(
            [{"target_ts": "not-a-date", "horizon": "24h"}], T0.isoformat()
        )
        assert out == []
        # #537: the malformed exit is counted separately from channel 2 —
        # it is a different failure ("we could not even read this snapshot's
        # target hour") wearing the expiry path's clothes.
        assert expired_by_horizon == {}
        assert malformed_by_horizon == {"24h": 1}

    def test_malformed_without_horizon_bucketed_unknown(self):
        out, expired_by_horizon, malformed_by_horizon = _expire_pending(
            [{"target_ts": "not-a-date"}], T0.isoformat()
        )
        assert out == []
        assert malformed_by_horizon == {"unknown": 1}


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


class TestLossChannelDiagnostics:
    """#537: separate the two silent ways an hour drops out of n_7d.

    Channel 1 — the forecast origin repeats a (target_ts, horizon) already
    pending, so the ``seen`` dedup in ``compute_horizon_drift_payload``
    correctly drops it, but silently. Channel 2 — ``_expire_pending`` drops a
    snapshot whose actual never arrived before ``PENDING_STALE_HOURS``.
    Malformed ``target_ts`` used to exit on channel 2's silent path; it is now
    counted separately since it means something different (a snapshot we
    could not even read, not one that waited and lost).
    """

    def test_dedup_skip_is_counted_not_silent(self):
        # Re-running the identical tick reuses every (target_ts, horizon) key
        # — the exact channel-1 symptom (a repeated forecast origin).
        p1 = compute_horizon_drift_payload("ERCOT", None, _forecast(T0), {}, now_iso=T0.isoformat())
        p2 = compute_horizon_drift_payload("ERCOT", p1, _forecast(T0), {}, now_iso=T0.isoformat())
        assert len(p2["pending"]) == 3  # unchanged — existing coverage
        by_horizon = {
            e["horizon"]: e for e in p2["diag_events"] if e["logged_at"] == T0.isoformat()
        }
        assert by_horizon["24h"]["n_dedup_skipped"] == 1
        assert by_horizon["48h"]["n_dedup_skipped"] == 1
        assert by_horizon["72h"]["n_dedup_skipped"] == 1

    def test_expired_unresolved_is_counted(self):
        stale_snap = {
            "target_ts": (T0 - timedelta(hours=200)).isoformat(),
            "made_at": (T0 - timedelta(hours=224)).isoformat(),
            "horizon": "24h",
            "preds": {"xgboost": 40_000.0},
        }
        existing = {"pending": [stale_snap], "models": {}}
        p = compute_horizon_drift_payload("ERCOT", existing, None, {}, now_iso=T0.isoformat())
        assert p["pending"] == []
        events = [e for e in p["diag_events"] if e["horizon"] == "24h"]
        assert sum(e["n_expired_unresolved"] for e in events) == 1
        assert sum(e["n_malformed"] for e in events) == 0

    def test_malformed_target_ts_counted_separately_from_expired(self):
        malformed_snap = {"target_ts": "not-a-date", "horizon": "48h", "preds": {}}
        existing = {"pending": [malformed_snap], "models": {}}
        p = compute_horizon_drift_payload("ERCOT", existing, None, {}, now_iso=T0.isoformat())
        events = [e for e in p["diag_events"] if e["horizon"] == "48h"]
        assert sum(e["n_malformed"] for e in events) == 1
        assert sum(e["n_expired_unresolved"] for e in events) == 0

    def test_diagnostics_surface_in_published_model_blocks(self):
        # Seed a diag log directly, as if accumulated over prior ticks, plus
        # one existing model with all three horizons already present. The
        # next tick's rollup must echo the windowed 7d totals into every
        # model's per-horizon block, scoped to the RIGHT horizon only.
        existing = {
            "pending": [],
            "diag_events": [
                {
                    "logged_at": T0.isoformat(),
                    "horizon": "24h",
                    "n_dedup_skipped": 2,
                    "n_expired_unresolved": 3,
                    "n_malformed": 0,
                }
            ],
            "models": {
                "xgboost": {
                    "24h": {"records": []},
                    "48h": {"records": []},
                    "72h": {"records": []},
                }
            },
        }
        now2 = T0 + timedelta(hours=1)
        p = compute_horizon_drift_payload("ERCOT", existing, None, {}, now_iso=now2.isoformat())
        block_24h = p["models"]["xgboost"]["24h"]
        assert block_24h["n_dedup_skipped_7d"] == 2
        assert block_24h["n_expired_unresolved_7d"] == 3
        assert block_24h["n_malformed_7d"] == 0
        # A horizon the diag entry did NOT touch stays at zero — no bleed
        # across horizons.
        assert p["models"]["xgboost"]["48h"]["n_dedup_skipped_7d"] == 0

    def test_diagnostics_log_ages_out_past_7d(self):
        old_entry = {
            "logged_at": (T0 - timedelta(hours=200)).isoformat(),  # > 7d ago
            "horizon": "24h",
            "n_dedup_skipped": 9,
            "n_expired_unresolved": 0,
            "n_malformed": 0,
        }
        existing = {
            "pending": [],
            "diag_events": [old_entry],
            "models": {"xgboost": {"24h": {"records": []}, "48h": {}, "72h": {}}},
        }
        p = compute_horizon_drift_payload("ERCOT", existing, None, {}, now_iso=T0.isoformat())
        assert p["models"]["xgboost"]["24h"]["n_dedup_skipped_7d"] == 0
        assert old_entry not in p["diag_events"]

    def test_diagnostics_never_become_model_keys(self):
        """Pins the #537 no-leak constraint from the caller's-eye view: after
        two ticks that produce real dedup counts, none of the counter names
        appear as a key of ``models`` — they must never be readable as if
        they were a model."""
        p1 = compute_horizon_drift_payload("ERCOT", None, _forecast(T0), {}, now_iso=T0.isoformat())
        p2 = compute_horizon_drift_payload("ERCOT", p1, _forecast(T0), {}, now_iso=T0.isoformat())
        forbidden = {
            "n_dedup_skipped",
            "n_expired_unresolved",
            "n_malformed",
            "n_dedup_skipped_7d",
            "n_expired_unresolved_7d",
            "n_malformed_7d",
        }
        assert set(p2["models"].keys()).isdisjoint(forbidden)

    def test_extract_one_hour_ahead_would_treat_counter_key_as_model(self):
        """Documents the landmine #537 warns about:
        ``extract_one_hour_ahead_predictions`` treats EVERY numeric key on a
        forecast row as a model name. This is why the diag counters live only
        in ``drift_horizon:{region}``'s own structure and are never merged
        into a forecast payload's row dict — if they ever were, they would
        silently become a 5th "model" with its own drift records and a place
        in the published rolling MAPE."""
        row = {
            "timestamp": T0.isoformat(),
            "predicted_demand_mw": 40_000.0,
            "xgboost": 40_000.0,
            "n_dedup_skipped": 3.0,  # hypothetical accidental leak
        }
        preds = extract_one_hour_ahead_predictions({"forecasts": [row]}, T0.isoformat())
        assert "n_dedup_skipped" in preds  # confirms the landmine is real

    def test_compute_horizon_drift_payload_never_mutates_forecast_payload(self):
        """The actual safeguard against the landmine above: the diag counters
        are computed and stored ONLY in the returned drift_horizon payload —
        the caller's forecast_payload (which later feeds
        extract_one_hour_ahead_predictions again next tick) must come back
        byte-for-byte unchanged."""
        forecast = _forecast(T0)
        before = copy.deepcopy(forecast)
        compute_horizon_drift_payload("ERCOT", None, forecast, {}, now_iso=T0.isoformat())
        assert forecast == before
