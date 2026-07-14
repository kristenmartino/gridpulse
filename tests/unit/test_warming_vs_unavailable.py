"""#273 Batch B: P2-35 (warming-forever → honest unavailable) and P2-21
(per-window drift sample counts gate the "live 7d/30d" figures).

P2-35: "Pipeline is warming up — forecast will appear shortly" was rendered
indefinitely for regions whose forecast is PERSISTENTLY unavailable (models
never trained, forecast phase failing past the 24h Redis TTL) while the
pipeline was demonstrably alive. The forecast and alerts gates now escalate
to a distinct non-transient unavailable state when the pipeline is provably
writing (fresh actuals / an existing forecast payload).

P2-21: the Overview headline gated "live 7d sMAPE" on n_records — TOTAL
retained history, trimmed by count not age — so the figure could rest on a
handful of in-window observations while claiming week-scale stability. The
drift payload now emits per-window post-filter counts (n_7d/n_30d) and every
consumer gates each window's figure on its own count.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from models.drift import DriftRecord, compute_drift_payload

NOW = datetime(2026, 7, 14, 12, 0, 0, tzinfo=UTC)


def _rec(hours_ago: int, error_pct: float = 5.0, actual: float = 100_000.0) -> DriftRecord:
    ts = (NOW - timedelta(hours=hours_ago)).isoformat()
    return DriftRecord(
        timestamp=ts,
        predicted=actual * (1 + error_pct / 100.0),
        actual=actual,
        abs_pct_error=error_pct,
    )


def _actuals_payload(age_hours: float) -> dict:
    return {
        "region": "SEC",
        "scored_at": (datetime.now(UTC) - timedelta(hours=age_hours)).isoformat(),
        "timestamps": [],
        "demand_mw": [],
    }


class TestPipelineAlive:
    def test_fresh_actuals_is_alive(self):
        from components._callbacks_shared import _pipeline_alive

        with patch("data.redis_client.redis_get", return_value=_actuals_payload(1.0)):
            assert _pipeline_alive("SEC") is True

    def test_stale_actuals_is_not_alive(self):
        from components._callbacks_shared import _pipeline_alive

        with patch("data.redis_client.redis_get", return_value=_actuals_payload(8.0)):
            assert _pipeline_alive("SEC") is False

    def test_missing_payload_is_not_alive(self):
        from components._callbacks_shared import _pipeline_alive

        with patch("data.redis_client.redis_get", return_value=None):
            assert _pipeline_alive("SEC") is False

    def test_malformed_payload_fails_closed(self):
        from components._callbacks_shared import _pipeline_alive

        for bad in ("corrupt", {"scored_at": "not-a-time"}, {"scored_at": None}, []):
            with patch("data.redis_client.redis_get", return_value=bad):
                assert _pipeline_alive("SEC") is False

    def test_naive_timestamp_treated_as_utc(self):
        from components._callbacks_shared import _pipeline_alive

        naive = {"scored_at": datetime.now(UTC).replace(tzinfo=None).isoformat()}
        with patch("data.redis_client.redis_get", return_value=naive):
            assert _pipeline_alive("SEC") is True


class TestForecastGateEscalation:
    """The REQUIRE_REDIS gate in ``_run_forecast_outlook`` (P2-35)."""

    def _run(self, forecast_payload, alive):
        import components._callbacks_forecast as fc

        fc._PREDICTION_CACHE.clear()
        demand_df = pd.DataFrame(
            {"timestamp": pd.date_range("2026-07-01", periods=8, freq="h"), "demand_mw": 1.0}
        )
        with (
            patch.object(fc, "REQUIRE_REDIS", True),
            patch.object(fc, "redis_get", return_value=forecast_payload),
            patch.object(fc, "_pipeline_alive", return_value=alive),
            patch("data.cache.get_cache") as mock_get_cache,
        ):
            mock_cache = MagicMock()
            mock_cache.get.return_value = None
            mock_get_cache.return_value = mock_cache
            return fc._run_forecast_outlook(demand_df, demand_df, 24, "xgboost", "SEC")

    def test_existing_payload_that_cannot_serve_is_unavailable(self):
        """The forecast payload exists (scoring runs land) but this selection
        fell through the fast path — that's persistent, not warming."""
        result = self._run({"forecasts": [{"timestamp": "t"}]}, alive=False)
        assert result["error"] == "unavailable"

    def test_missing_payload_with_live_pipeline_is_unavailable(self):
        """Fresh actuals prove the pipeline writes this region, yet no
        forecast key exists — 'will appear shortly' would be a forever-lie."""
        result = self._run(None, alive=True)
        assert result["error"] == "unavailable"

    def test_missing_payload_with_cold_pipeline_stays_warming(self):
        """Nothing written at all: genuine warming (deploy/flush) — keep the
        softer transient copy."""
        result = self._run(None, alive=False)
        assert result["error"] == "warming"


class TestAlertsGateEscalation:
    """The alerts warming gate escalates the same way (P2-35)."""

    @pytest.fixture(scope="class")
    def callbacks(self):
        import dash
        import dash_bootstrap_components as dbc

        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            suppress_callback_exceptions=True,
        )
        from components.callbacks import register_callbacks
        from components.layout import build_layout

        app.layout = build_layout()
        register_callbacks(app)
        fns = {}
        for val in app.callback_map.values():
            fn = val.get("callback")
            if fn and hasattr(fn, "__name__"):
                fns[fn.__name__] = getattr(fn, "__wrapped__", fn)
        return fns

    def _render(self, callbacks, alive):
        import components._callbacks_alerts as al

        with (
            patch.object(al, "_alerts_tab_from_redis", return_value=None),
            patch.object(al, "REQUIRE_REDIS", True),
            patch("components._callbacks_shared._pipeline_alive", return_value=alive),
        ):
            return callbacks["update_alerts_tab"]("SEC", None, None, "tab-alerts")

    def test_live_pipeline_escalates_to_unavailable(self, callbacks):
        result = self._render(callbacks, alive=True)
        rendered = str(result[0])
        assert "Risk data unavailable" in rendered
        assert "won't resolve on its own" in rendered
        # The transient claim must be gone (CSS class names still say
        # "warming" — the card builder is shared — so check the copy).
        assert "warming up" not in rendered.lower()
        assert "next scoring run" not in rendered
        assert str(result[2].children) == "Unavailable"

    def test_cold_pipeline_keeps_warming(self, callbacks):
        result = self._render(callbacks, alive=False)
        rendered = str(result[0])
        assert "Risk data is warming up" in rendered
        assert str(result[2].children) == "Warming"


class TestDriftWindowCounts:
    """P2-21: compute_drift_payload emits per-window post-filter counts."""

    def test_counts_split_by_window_age(self):
        """5 recent records + 20 mid-age + old history: n_7d counts only the
        7d window, n_30d the 30d window, n_records the whole retained set."""
        existing = compute_drift_payload(
            "SEC",
            existing_payload=None,
            new_records={"ensemble": _rec(0)},
            now_iso=NOW.isoformat(),
        )
        # Manually build a merged history via successive payloads: 4 more
        # recent records, 20 mid-age (8-27 days), 10 ancient (>30d).
        records = (
            [_rec(h) for h in (1, 2, 3, 4)]
            + [_rec(24 * d) for d in range(8, 28)]
            + [_rec(24 * d) for d in range(31, 41)]
        )
        for r in records:
            existing = compute_drift_payload(
                "SEC",
                existing_payload=existing,
                new_records={"ensemble": r},
                now_iso=NOW.isoformat(),
            )
        block = existing["models"]["ensemble"]
        assert block["n_records"] == 35
        assert block["n_7d"] == 5
        assert block["n_30d"] == 25
        assert block["n_30d"] >= block["n_7d"]

    def test_low_actual_filter_reduces_window_counts(self):
        """The counts are POST-filter — the honest denominator of the mean."""
        payload = None
        # 24 normal records + 3 near-zero-actual artifacts, all inside 7d.
        recs = [_rec(h) for h in range(24)] + [
            _rec(30 + i, actual=1.0)
            for i in range(3)  # hours 30-32, still <7d
        ]
        for r in recs:
            payload = compute_drift_payload(
                "SEC",
                existing_payload=payload,
                new_records={"ensemble": r},
                now_iso=NOW.isoformat(),
            )
        block = payload["models"]["ensemble"]
        assert block["n_records"] == 27
        assert block["n_7d"] == 24  # 3 near-zero artifacts excluded
        assert block["n_low_actual_excluded_7d"] == 3


class TestOverviewWindowGating:
    """P2-21: _resolve_forecast_mape gates each window on ITS OWN count."""

    def _payload(self, **ens):
        return {"models": {"ensemble": {"records": [], **ens}}}

    def test_headline_7d_blocked_by_thin_window_despite_total_history(self):
        """The audit's lie: n_records=100 but only 3 in-window observations.
        The 7d figure must NOT be served; the 30d (n_30d=40) takes over."""
        import components._callbacks_overview as ov

        payload = self._payload(
            n_records=100,
            n_7d=3,
            n_30d=40,
            rolling_smape_7d=3.8,
            rolling_smape_30d=4.6,
        )
        with patch.object(ov, "redis_get", return_value=payload):
            value, source = ov._resolve_forecast_mape("SEC")
        assert value == 4.6
        assert source == "live 30d sMAPE"

    def test_both_windows_thin_falls_through(self):
        import components._callbacks_overview as ov

        payload = self._payload(
            n_records=100,
            n_7d=3,
            n_30d=9,
            rolling_smape_7d=3.8,
            rolling_smape_30d=4.6,
        )
        with (
            patch.object(ov, "redis_get", return_value=payload),
            patch("models.model_service.get_model_metrics", return_value={}),
        ):
            value, source = ov._resolve_forecast_mape("SEC")
        assert value is None

    def test_healthy_window_serves_7d(self):
        import components._callbacks_overview as ov

        payload = self._payload(n_records=100, n_7d=150, n_30d=600, rolling_smape_7d=3.8)
        with patch.object(ov, "redis_get", return_value=payload):
            value, source = ov._resolve_forecast_mape("SEC")
        assert value == 3.8
        assert source == "live 7d sMAPE"

    def test_legacy_payload_without_counts_keeps_old_gate(self):
        """Pre-#273 payloads (no n_7d/n_30d) must keep working on the old
        total-count gate for the one tick until the hourly rewrite."""
        import components._callbacks_overview as ov

        payload = self._payload(n_records=100, rolling_smape_7d=3.8)
        with patch.object(ov, "redis_get", return_value=payload):
            value, source = ov._resolve_forecast_mape("SEC")
        assert value == 3.8
        assert source == "live 7d sMAPE"


class TestApiExportsWindowCounts:
    def test_live_and_horizon_field_allowlists(self):
        from api import _EXPORTED_HORIZON_DRIFT_FIELDS, _EXPORTED_LIVE_DRIFT_FIELDS

        assert "n_7d" in _EXPORTED_LIVE_DRIFT_FIELDS
        assert "n_30d" in _EXPORTED_LIVE_DRIFT_FIELDS
        assert "n_7d" in _EXPORTED_HORIZON_DRIFT_FIELDS


class TestModelsTabWindowGating:
    """P2-21: the Models drift panel warms on the 7d in-window count."""

    def _drift_payload(self, n_records, n_7d=None):
        ens = {
            "n_records": n_records,
            "rolling_mape_7d": 4.0,
            "rolling_mape_30d": 4.2,
            "records": [],
        }
        if n_7d is not None:
            ens["n_7d"] = n_7d
        return {"models": {"ensemble": ens, "xgboost": dict(ens)}}

    def test_thin_window_warms_despite_total_history(self):
        import components._callbacks_models as cm

        with (
            patch.object(cm, "redis_get", return_value=self._drift_payload(100, n_7d=3)),
            patch("models.model_service.get_model_metrics", return_value={}),
        ):
            panel = str(cm._build_drift_panel("SEC"))
        assert "Warming" in panel

    def test_healthy_window_not_warming(self):
        import components._callbacks_models as cm

        with (
            patch.object(cm, "redis_get", return_value=self._drift_payload(100, n_7d=150)),
            patch("models.model_service.get_model_metrics", return_value={}),
        ):
            panel = str(cm._build_drift_panel("SEC"))
        assert "Warming" not in panel
