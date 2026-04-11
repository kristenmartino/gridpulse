"""
Unit tests for data/session_diff.py (NEXD-8: "What Changed Since Last Time?").

Covers:
- SessionSnapshot creation, serialization, round-trip
- compute_snapshot() with various input combinations
- compute_diff() change detection rules and thresholds
- format_relative_time() human-friendly time strings
- _pct_change() edge cases
"""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from data.session_diff import (
    MAX_CHANGES,
    ChangeItem,
    SessionSnapshot,
    _pct_change,
    compute_diff,
    compute_snapshot,
    format_relative_time,
)

# ── SessionSnapshot ──────────────────────────────────────────────────


class TestSessionSnapshot:
    """Verify SessionSnapshot dataclass behaviour."""

    def test_defaults(self):
        snap = SessionSnapshot()
        assert snap.region == ""
        assert snap.peak_demand_mw is None
        assert snap.mape == {}
        assert snap.data_sources == {}

    def test_to_dict_serializable(self):
        snap = SessionSnapshot(region="FPL", peak_demand_mw=42000.0)
        d = snap.to_dict()
        assert d["region"] == "FPL"
        assert d["peak_demand_mw"] == 42000.0
        assert isinstance(d, dict)

    def test_from_dict_round_trip(self):
        snap = SessionSnapshot(
            region="ERCOT",
            persona="trader",
            peak_demand_mw=55000.0,
            mape={"xgboost": 3.2},
        )
        d = snap.to_dict()
        restored = SessionSnapshot.from_dict(d)
        assert restored.region == "ERCOT"
        assert restored.peak_demand_mw == 55000.0
        assert restored.mape == {"xgboost": 3.2}

    def test_from_dict_none(self):
        snap = SessionSnapshot.from_dict(None)
        assert snap.region == ""

    def test_from_dict_empty(self):
        snap = SessionSnapshot.from_dict({})
        assert snap.region == ""

    def test_from_dict_ignores_unknown_keys(self):
        snap = SessionSnapshot.from_dict({"region": "FPL", "unknown_field": 123})
        assert snap.region == "FPL"

    def test_none_optional_fields(self):
        snap = SessionSnapshot(alert_count=None, renewable_pct=None)
        d = snap.to_dict()
        assert d["alert_count"] is None
        assert d["renewable_pct"] is None


# ── ChangeItem ───────────────────────────────────────────────────────


class TestChangeItem:
    def test_to_dict(self):
        item = ChangeItem(
            category="demand",
            text="Peak demand up 12%",
            severity="notable",
            icon="\u2197",
            persona_relevance=["grid_ops"],
        )
        d = item.to_dict()
        assert d["category"] == "demand"
        assert d["severity"] == "notable"

    def test_from_dict(self):
        d = {"category": "alerts", "text": "2 new alerts", "severity": "warning", "icon": "\u26a0"}
        item = ChangeItem.from_dict(d)
        assert item.category == "alerts"
        assert item.severity == "warning"


# ── _pct_change ──────────────────────────────────────────────────────


class TestPctChange:
    def test_positive_change(self):
        assert _pct_change(100.0, 112.0) == pytest.approx(12.0)

    def test_negative_change(self):
        assert _pct_change(100.0, 90.0) == pytest.approx(-10.0)

    def test_none_old(self):
        assert _pct_change(None, 100.0) is None

    def test_none_new(self):
        assert _pct_change(100.0, None) is None

    def test_both_none(self):
        assert _pct_change(None, None) is None

    def test_zero_denominator(self):
        assert _pct_change(0.0, 50.0) is None

    def test_negative_base(self):
        """Percentage change uses abs(old) as denominator."""
        result = _pct_change(-100.0, -90.0)
        assert result == pytest.approx(10.0)


# ── format_relative_time ─────────────────────────────────────────────


class TestFormatRelativeTime:
    def test_just_now(self):
        now = datetime.now(UTC).isoformat()
        assert format_relative_time(now) == "just now"

    def test_minutes_ago(self):
        ts = (datetime.now(UTC) - timedelta(minutes=15)).isoformat()
        result = format_relative_time(ts)
        assert "m ago" in result

    def test_hours_ago(self):
        ts = (datetime.now(UTC) - timedelta(hours=3)).isoformat()
        result = format_relative_time(ts)
        assert "h ago" in result

    def test_yesterday(self):
        ts = (datetime.now(UTC) - timedelta(hours=30)).isoformat()
        result = format_relative_time(ts)
        assert result == "yesterday"

    def test_days_ago(self):
        ts = (datetime.now(UTC) - timedelta(days=5)).isoformat()
        result = format_relative_time(ts)
        assert "d ago" in result

    def test_invalid_string(self):
        assert format_relative_time("not-a-date") == ""

    def test_empty_string(self):
        assert format_relative_time("") == ""


# ── compute_snapshot ─────────────────────────────────────────────────


class TestComputeSnapshot:
    def test_with_valid_demand(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-04-01", periods=48, freq="h"),
                "demand_mw": [30000 + i * 100 for i in range(48)],
            }
        )
        snap = compute_snapshot("FPL", "grid_ops", demand_df=df)
        assert snap.region == "FPL"
        assert snap.persona == "grid_ops"
        assert snap.peak_demand_mw == 30000 + 47 * 100
        assert snap.avg_demand_24h is not None
        assert snap.timestamp != ""

    def test_with_none_demand(self):
        snap = compute_snapshot("FPL", "grid_ops", demand_df=None)
        assert snap.peak_demand_mw is None
        assert snap.avg_demand_24h is None

    def test_with_empty_demand(self):
        df = pd.DataFrame(columns=["timestamp", "demand_mw"])
        snap = compute_snapshot("FPL", "grid_ops", demand_df=df)
        assert snap.peak_demand_mw is None

    def test_with_audit_data(self):
        audit = {
            "peak_forecast_mw": 45000.0,
            "mape": {"prophet": 4.5, "xgboost": 3.2},
            "ensemble_weights": {"prophet": 0.4, "xgboost": 0.6},
            "model_versions": {"prophet": "v1", "xgboost": "v2"},
        }
        snap = compute_snapshot("ERCOT", "trader", audit_data=audit)
        assert snap.forecast_peak_mw == 45000.0
        assert snap.mape == {"prophet": 4.5, "xgboost": 3.2}
        assert snap.ensemble_weights == {"prophet": 0.4, "xgboost": 0.6}

    def test_audit_peak_zero_treated_as_none(self):
        audit = {"peak_forecast_mw": 0.0}
        snap = compute_snapshot("FPL", "grid_ops", audit_data=audit)
        assert snap.forecast_peak_mw is None

    def test_with_freshness_data(self):
        freshness = {
            "demand": "fresh",
            "weather": "stale",
            "alerts": "demo",
            "timestamp": "2026-04-10T12:00:00Z",
            "latest_data": "2026-04-10T11:00:00Z",
        }
        snap = compute_snapshot("FPL", "grid_ops", freshness_data=freshness)
        assert snap.data_sources == {"demand": "fresh", "weather": "stale", "alerts": "demo"}
        assert "timestamp" not in snap.data_sources
        assert "latest_data" not in snap.data_sources

    def test_with_none_audit(self):
        snap = compute_snapshot("FPL", "grid_ops", audit_data=None)
        assert snap.mape == {}
        assert snap.forecast_peak_mw is None

    def test_with_none_freshness(self):
        snap = compute_snapshot("FPL", "grid_ops", freshness_data=None)
        assert snap.data_sources == {}


# ── compute_diff ─────────────────────────────────────────────────────


class TestComputeDiff:
    def _make_snap(self, **overrides) -> SessionSnapshot:
        defaults = {
            "region": "FPL",
            "persona": "grid_ops",
            "timestamp": datetime.now(UTC).isoformat(),
            "peak_demand_mw": 40000.0,
            "avg_demand_24h": 35000.0,
            "forecast_peak_mw": 42000.0,
            "mape": {"xgboost": 5.0},
            "data_sources": {"demand": "fresh", "weather": "fresh"},
            "alert_count": 2,
            "renewable_pct": 15.0,
        }
        defaults.update(overrides)
        return SessionSnapshot(**defaults)

    def test_same_snapshots_returns_empty(self):
        snap = self._make_snap()
        changes = compute_diff(snap, snap, "grid_ops")
        assert changes == []

    def test_peak_demand_increase_above_threshold(self):
        prev = self._make_snap(peak_demand_mw=40000.0)
        curr = self._make_snap(peak_demand_mw=46000.0)  # +15%
        changes = compute_diff(prev, curr, "grid_ops")
        demand_changes = [c for c in changes if c.category == "demand" and "Peak" in c.text]
        assert len(demand_changes) == 1
        assert "up" in demand_changes[0].text
        assert demand_changes[0].severity == "notable"  # >10%

    def test_peak_demand_small_change_below_threshold(self):
        prev = self._make_snap(peak_demand_mw=40000.0)
        curr = self._make_snap(peak_demand_mw=41000.0)  # +2.5% < 5%
        changes = compute_diff(prev, curr, "grid_ops")
        demand_changes = [c for c in changes if "Peak demand" in c.text]
        assert len(demand_changes) == 0

    def test_peak_demand_moderate_change_info(self):
        prev = self._make_snap(peak_demand_mw=40000.0)
        curr = self._make_snap(peak_demand_mw=43000.0)  # +7.5%: >5% but <10%
        changes = compute_diff(prev, curr, "grid_ops")
        demand_changes = [c for c in changes if "Peak demand" in c.text]
        assert len(demand_changes) == 1
        assert demand_changes[0].severity == "info"

    def test_avg_demand_change(self):
        prev = self._make_snap(avg_demand_24h=35000.0)
        curr = self._make_snap(avg_demand_24h=38000.0)  # +8.6%
        changes = compute_diff(prev, curr, "grid_ops")
        avg_changes = [c for c in changes if "Average demand" in c.text]
        assert len(avg_changes) == 1
        assert "up" in avg_changes[0].text

    def test_forecast_peak_change(self):
        prev = self._make_snap(forecast_peak_mw=42000.0)
        curr = self._make_snap(forecast_peak_mw=47000.0)  # +11.9%
        changes = compute_diff(prev, curr, "grid_ops")
        forecast_changes = [c for c in changes if c.category == "forecast"]
        assert len(forecast_changes) == 1
        assert "upward" in forecast_changes[0].text

    def test_alert_count_increase(self):
        prev = self._make_snap(alert_count=1)
        curr = self._make_snap(alert_count=3)
        changes = compute_diff(prev, curr, "grid_ops")
        alert_changes = [c for c in changes if c.category == "alerts"]
        assert len(alert_changes) == 1
        assert "2 new" in alert_changes[0].text
        assert alert_changes[0].severity == "warning"

    def test_alerts_cleared(self):
        prev = self._make_snap(alert_count=3)
        curr = self._make_snap(alert_count=0)
        changes = compute_diff(prev, curr, "grid_ops")
        alert_changes = [c for c in changes if c.category == "alerts"]
        assert len(alert_changes) == 1
        assert "cleared" in alert_changes[0].text
        assert alert_changes[0].severity == "info"

    def test_alert_count_none_skipped(self):
        prev = self._make_snap(alert_count=None)
        curr = self._make_snap(alert_count=5)
        changes = compute_diff(prev, curr, "grid_ops")
        alert_changes = [c for c in changes if c.category == "alerts"]
        assert len(alert_changes) == 0

    def test_renewable_pct_change(self):
        prev = self._make_snap(renewable_pct=15.0)
        curr = self._make_snap(renewable_pct=22.0)  # +7pp
        changes = compute_diff(prev, curr, "renewables")
        gen_changes = [c for c in changes if c.category == "generation"]
        assert len(gen_changes) == 1
        assert "rose" in gen_changes[0].text

    def test_renewable_pct_small_change_skipped(self):
        prev = self._make_snap(renewable_pct=15.0)
        curr = self._make_snap(renewable_pct=16.5)  # +1.5pp < 3pp
        changes = compute_diff(prev, curr, "renewables")
        gen_changes = [c for c in changes if c.category == "generation"]
        assert len(gen_changes) == 0

    def test_mape_degradation_warning(self):
        prev = self._make_snap(mape={"xgboost": 3.0})
        curr = self._make_snap(mape={"xgboost": 6.0})  # +3pp
        changes = compute_diff(prev, curr, "data_scientist")
        model_changes = [c for c in changes if c.category == "models"]
        assert len(model_changes) == 1
        assert "degraded" in model_changes[0].text
        assert model_changes[0].severity == "warning"

    def test_mape_improvement_info(self):
        prev = self._make_snap(mape={"xgboost": 6.0})
        curr = self._make_snap(mape={"xgboost": 3.0})  # -3pp
        changes = compute_diff(prev, curr, "data_scientist")
        model_changes = [c for c in changes if c.category == "models"]
        assert len(model_changes) == 1
        assert "improved" in model_changes[0].text
        assert model_changes[0].severity == "info"

    def test_mape_small_change_skipped(self):
        prev = self._make_snap(mape={"xgboost": 5.0})
        curr = self._make_snap(mape={"xgboost": 5.5})  # +0.5pp < 2pp
        changes = compute_diff(prev, curr, "data_scientist")
        model_changes = [c for c in changes if c.category == "models"]
        assert len(model_changes) == 0

    def test_data_source_degradation(self):
        prev = self._make_snap(data_sources={"demand": "fresh", "weather": "fresh"})
        curr = self._make_snap(data_sources={"demand": "fresh", "weather": "stale"})
        changes = compute_diff(prev, curr, "grid_ops")
        data_changes = [c for c in changes if c.category == "data"]
        assert len(data_changes) == 1
        assert "Weather" in data_changes[0].text
        assert data_changes[0].severity == "warning"

    def test_data_source_recovery(self):
        prev = self._make_snap(data_sources={"demand": "stale"})
        curr = self._make_snap(data_sources={"demand": "fresh"})
        changes = compute_diff(prev, curr, "grid_ops")
        data_changes = [c for c in changes if c.category == "data"]
        assert len(data_changes) == 1
        assert data_changes[0].severity == "info"

    def test_persona_filtering_trader(self):
        """Trader should not see alerts (only grid_ops relevant)."""
        prev = self._make_snap(alert_count=0)
        curr = self._make_snap(alert_count=5)
        changes = compute_diff(prev, curr, "trader")
        alert_changes = [c for c in changes if c.category == "alerts"]
        assert len(alert_changes) == 0

    def test_persona_filtering_renewables(self):
        """Renewables persona should see generation changes."""
        prev = self._make_snap(renewable_pct=15.0)
        curr = self._make_snap(renewable_pct=25.0)
        changes = compute_diff(prev, curr, "renewables")
        gen_changes = [c for c in changes if c.category == "generation"]
        assert len(gen_changes) == 1

    def test_max_changes_capped(self):
        """Output should not exceed MAX_CHANGES items."""
        prev = self._make_snap(
            peak_demand_mw=10000.0,
            avg_demand_24h=8000.0,
            forecast_peak_mw=11000.0,
            mape={"prophet": 2.0, "xgboost": 2.0, "arima": 2.0},
            data_sources={"demand": "fresh", "weather": "fresh", "alerts": "fresh"},
            alert_count=0,
            renewable_pct=10.0,
        )
        curr = self._make_snap(
            peak_demand_mw=20000.0,
            avg_demand_24h=16000.0,
            forecast_peak_mw=22000.0,
            mape={"prophet": 12.0, "xgboost": 12.0, "arima": 12.0},
            data_sources={"demand": "stale", "weather": "error", "alerts": "demo"},
            alert_count=5,
            renewable_pct=30.0,
        )
        # grid_ops sees demand, forecast, alerts, data — many potential changes
        changes = compute_diff(prev, curr, "grid_ops")
        assert len(changes) <= MAX_CHANGES

    def test_sorted_by_severity(self):
        """Warning items should come before info items."""
        prev = self._make_snap(
            peak_demand_mw=40000.0,
            mape={"xgboost": 3.0},
            data_sources={"demand": "fresh"},
        )
        curr = self._make_snap(
            peak_demand_mw=46000.0,  # notable
            mape={"xgboost": 8.0},  # warning (degraded)
            data_sources={"demand": "stale"},  # warning
        )
        changes = compute_diff(prev, curr, "data_scientist")
        if len(changes) >= 2:
            severities = [c.severity for c in changes]
            # Warning items should appear before non-warning items
            warning_indices = [i for i, s in enumerate(severities) if s == "warning"]
            non_warning_indices = [i for i, s in enumerate(severities) if s != "warning"]
            if warning_indices and non_warning_indices:
                assert max(warning_indices) < min(non_warning_indices)
