"""Trust-signal measurement regression tests.

Pins the fixes for the 2026-07 critical-review P1 cluster
(docs/internal/CRITICAL_REVIEW_2026-07.md):

* P1-3 — freshness is MEASURED from each Redis payload's own ``scored_at``,
  never hardcoded ``'fresh'`` at render time;
* P1-4 — the Overview review surfaces the payload generation time and
  a visible stale warning instead of narrating an old forecast as current;
* P1-2 — an empirical interval calibrated on a substitute model's
  residuals discloses the calibration source in its label;
* P1-8 — an sMAPE value is labeled sMAPE, never "MAPE".
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd


def _iso_hours_ago(hours: float) -> str:
    return (datetime.now(UTC) - timedelta(hours=hours)).isoformat()


# ---------------------------------------------------------------------------
# P1-3: measured freshness
# ---------------------------------------------------------------------------


class TestMeasuredFreshness:
    def test_fresh_scored_at_measures_fresh(self):
        from components.callbacks import _freshness_from_payload

        now = pd.Timestamp.now(tz="UTC")
        assert _freshness_from_payload({"scored_at": _iso_hours_ago(0.5)}, now) == "fresh"

    def test_old_scored_at_measures_stale(self):
        from components.callbacks import _freshness_from_payload

        now = pd.Timestamp.now(tz="UTC")
        assert _freshness_from_payload({"scored_at": _iso_hours_ago(5)}, now) == "stale"

    def test_missing_stamp_returns_none(self):
        from components.callbacks import _freshness_from_payload

        now = pd.Timestamp.now(tz="UTC")
        assert _freshness_from_payload({}, now) is None
        assert _freshness_from_payload(None, now) is None

    def test_load_data_freshness_is_payload_driven(self):
        """Stale payloads must not render as fresh; alerts status follows alerts_source."""
        import components.callbacks as cbs

        ts = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("h"), periods=48, freq="h")
        payloads = {
            "gridpulse:actuals:ERCOT": {
                "region": "ERCOT",
                "scored_at": _iso_hours_ago(7),  # two+ missed scoring runs
                "timestamps": [t.isoformat() for t in ts],
                "demand_mw": [1000.0] * 48,
            },
            "gridpulse:weather:ERCOT": {
                "region": "ERCOT",
                "scored_at": _iso_hours_ago(7),
                "timestamps": [t.isoformat() for t in ts],
                "temperature_2m": [75.0] * 48,
            },
            "gridpulse:alerts:ERCOT": {
                "region": "ERCOT",
                "alerts_source": "unavailable",
                "alerts": [],
            },
        }
        with patch.object(cbs, "redis_get", side_effect=lambda k: payloads.get(k)):
            result = cbs._load_data_from_redis("ERCOT")

        assert result is not None
        freshness = json.loads(result[2])
        assert freshness["demand"] == "stale"
        assert freshness["weather"] == "stale"
        assert freshness["alerts"] == "unavailable"

    def test_load_data_fresh_payloads_measure_fresh(self):
        import components.callbacks as cbs

        ts = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("h"), periods=48, freq="h")
        payloads = {
            "gridpulse:actuals:ERCOT": {
                "region": "ERCOT",
                "scored_at": _iso_hours_ago(0.5),
                "timestamps": [t.isoformat() for t in ts],
                "demand_mw": [1000.0] * 48,
            },
            "gridpulse:weather:ERCOT": {
                "region": "ERCOT",
                "scored_at": _iso_hours_ago(0.5),
                "timestamps": [t.isoformat() for t in ts],
                "temperature_2m": [75.0] * 48,
            },
        }
        with patch.object(cbs, "redis_get", side_effect=lambda k: payloads.get(k)):
            result = cbs._load_data_from_redis("ERCOT")

        freshness = json.loads(result[2])
        assert freshness["demand"] == "fresh"
        assert freshness["weather"] == "fresh"
        # No alerts payload at all → warming, not fresh.
        assert freshness["alerts"] == "warming"
        # scored_at must not leak into the weather DataFrame columns.
        weather_df = pd.read_json(
            result[1] if hasattr(result[1], "read") else __import__("io").StringIO(result[1])
        )
        assert "scored_at" not in weather_df.columns


# ---------------------------------------------------------------------------
# P1-2: interval calibration-source disclosure
# ---------------------------------------------------------------------------


def _backtest_payload(n: int = 40) -> dict:
    rng = np.random.default_rng(7)
    actual = 1000 + rng.normal(0, 30, n)
    preds = actual + rng.normal(0, 20, n)
    return {
        "horizon": 24,
        "actual": actual.tolist(),
        "predictions": {"xgboost": preds.tolist()},
        "timestamps": [f"2026-06-{d:02d}T00:00:00" for d in range(1, 5) for _ in range(10)],
    }


class TestCalibrationSourceDisclosure:
    def test_substitute_residuals_are_attributed(self):
        import components._callbacks_shared as shared

        payload = _backtest_payload()
        with patch.object(shared, "redis_get", side_effect=lambda k: payload):
            residuals, calib = shared._collect_backtest_residuals("ERCOT", "ensemble", 24)
        assert residuals.size > 0
        assert calib == "xgboost"

    def test_exact_residuals_keep_model_name(self):
        import components._callbacks_shared as shared

        payload = _backtest_payload()
        with patch.object(shared, "redis_get", side_effect=lambda k: payload):
            residuals, calib = shared._collect_backtest_residuals("ERCOT", "xgboost", 24)
        assert calib == "xgboost"

    def test_interval_meta_carries_calibration_model(self):
        import components._callbacks_shared as shared

        payload = _backtest_payload()
        with patch.object(shared, "redis_get", side_effect=lambda k: payload):
            meta = shared._empirical_interval_from_backtests("ERCOT", "ensemble", 24)
        assert meta["available"] is True
        assert meta["calibration_model"] == "xgboost"


# ---------------------------------------------------------------------------
# P1-8: metric-name honesty
# ---------------------------------------------------------------------------


class TestMetricNameHonesty:
    def _drift_payload(self, field: str, value: float) -> dict:
        return {"models": {"ensemble": {"n_records": 100, field: value}}}

    def test_smape_labeled_smape(self):
        import components._callbacks_overview as ov

        with patch.object(
            ov, "redis_get", return_value=self._drift_payload("rolling_smape_7d", 13.2)
        ):
            value, source = ov._resolve_forecast_mape("LDWP")
        assert value == 13.2
        assert "sMAPE" in source
        assert "MAPE" not in source.replace("sMAPE", "")

    def test_mape_fallback_labeled_mape(self):
        import components._callbacks_overview as ov

        with patch.object(
            ov, "redis_get", return_value=self._drift_payload("rolling_mape_7d", 4.1)
        ):
            value, source = ov._resolve_forecast_mape("ERCOT")
        assert value == 4.1
        assert source == "live 7d MAPE"


# ---------------------------------------------------------------------------
# P1-4: generation time and measured staleness stay visible in Overview
# ---------------------------------------------------------------------------


class TestScoredAtSurfaced:
    def _demand_df(self):
        ts = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("h"), periods=48, freq="h")
        return pd.DataFrame({"timestamp": ts, "demand_mw": 1000.0})

    def _review(self, scored_at_iso: str) -> tuple[str, str]:
        from dash import html

        import components._callbacks_overview as ov
        from components.forecast_context import forecast_review_context
        from components.overview_review import build_review_summary
        from tests.unit.test_forecast_review_context import payload, text

        cached = payload()
        cached["scored_at"] = scored_at_iso
        forecast_ts = pd.date_range(
            start=pd.Timestamp.now(tz="UTC").floor("h") + pd.Timedelta(hours=1),
            periods=24,
            freq="h",
        )
        for row, stamp in zip(cached["forecasts"], forecast_ts, strict=True):
            row["timestamp"] = stamp.isoformat()
        with (
            patch("components.forecast_context.redis_get", return_value=None),
            patch.object(ov, "_provenance_note", return_value=[]),
        ):
            context = forecast_review_context("ERCOT", cached, "published", 24)
            summary = build_review_summary(context, cached)
            insight = ov._build_overview_insight(
                "ERCOT", self._demand_df(), "grid_ops", cached=cached, compact=True
            )

        def visible_copy(component) -> str:
            # Closed disclosures cannot satisfy a visible freshness requirement.
            return " ".join(
                text(child) for child in component.children if not isinstance(child, html.Details)
            )

        return visible_copy(summary), visible_copy(insight)

    def test_stale_forecast_is_marked(self):
        scored_at = _iso_hours_ago(9)
        summary, visible_insight = self._review(scored_at)
        expected_clock = datetime.fromisoformat(scored_at).strftime("%Y-%m-%d %H:%M")
        assert f"Generated: {expected_clock} UTC" in summary
        assert "Forecast last refreshed 9h ago." in visible_insight

    def test_fresh_forecast_shows_timestamp_without_stale_flag(self):
        scored_at = _iso_hours_ago(0.5)
        summary, visible_insight = self._review(scored_at)
        expected_clock = datetime.fromisoformat(scored_at).strftime("%Y-%m-%d %H:%M")
        assert f"Generated: {expected_clock} UTC" in summary
        assert "Forecast last refreshed" not in visible_insight
