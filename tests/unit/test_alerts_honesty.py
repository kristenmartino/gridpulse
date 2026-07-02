"""Alert-feed honesty regression tests.

Pins the fix for the fabricated-alerts defect found by the 2026-07 critical
review (docs/internal/CRITICAL_REVIEW_2026-07.md, finding P0-1):

* the scoring job must never publish demo alerts outside demo mode
  (``write_alerts`` emits ``alerts_source="unavailable"`` with no alerts,
  ``stress_score=None``);
* when demo alerts ARE shown (demo mode / dev fallback), the UI must
  disclose them as demo data;
* a ``REQUIRE_REDIS`` deployment must render a warming state — never demo
  content — when the alerts payload is missing from Redis;
* absent model metrics must render as unavailable ("—"), never as perfect
  scores ("MAPE 0.0%").
"""

from __future__ import annotations

from unittest.mock import patch

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import pytest

import config
from jobs import phases

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _region_data(region: str = "FPL") -> phases.RegionData:
    ts = pd.date_range("2026-06-01", periods=200, freq="h", tz="UTC")
    demand = pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": 1000.0 + 100.0 * np.sin(np.arange(200) * 2 * np.pi / 24),
        }
    )
    weather = pd.DataFrame({"timestamp": ts, "temperature_2m": 75.0})
    return phases.RegionData(region=region, demand_df=demand, weather_df=weather)


def _collect_text(component) -> str:
    """Flatten a Dash component tree into a plain-text blob."""
    parts: list[str] = []

    def walk(node):
        if node is None:
            return
        if isinstance(node, str):
            parts.append(node)
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                walk(child)
            return
        walk(getattr(node, "children", None))

    walk(component)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# write_alerts (scoring job)
# ---------------------------------------------------------------------------


class TestWriteAlertsHonesty:
    def test_prod_mode_noaa_outage_writes_unavailable_payload(self, monkeypatch):
        """NOAA outage: no alerts, explicit unavailable source, no stress score —
        an outage must never be disguised as 'no active alerts' or demo content."""
        from data.noaa_client import NOAAAlertsUnavailableError

        monkeypatch.setattr(config, "USE_DEMO_DATA", False)
        captured: dict = {}

        def fake_set(key, payload, ttl=None):
            captured["key"] = key
            captured["payload"] = payload
            return True

        with (
            patch("data.redis_client.redis_set", side_effect=fake_set),
            patch(
                "data.noaa_client.fetch_alerts_for_region",
                side_effect=NOAAAlertsUnavailableError("total outage"),
            ),
        ):
            result = phases.write_alerts(_region_data("FPL"))

        assert result.ok
        payload = captured["payload"]
        assert payload["alerts"] == []
        assert payload["alerts_source"] == "unavailable"
        assert payload["stress_score"] is None
        assert payload["stress_label"] == "Unavailable"
        assert payload["alert_counts"] == {"critical": 0, "warning": 0, "info": 0}
        # The real (data-derived) sections must still be present.
        assert payload["anomaly"]["timestamps"]

    def test_prod_mode_never_calls_demo_generator(self, monkeypatch):
        """Structural guardrail: the demo generator is unreachable outside demo mode."""
        monkeypatch.setattr(config, "USE_DEMO_DATA", False)

        def boom(region):  # pragma: no cover - must not run
            raise AssertionError("generate_demo_alerts must not be called in prod")

        with (
            patch("data.redis_client.redis_set", return_value=True),
            patch("data.demo_data.generate_demo_alerts", side_effect=boom),
            patch("data.noaa_client.fetch_alerts_for_region", return_value=[]),
        ):
            result = phases.write_alerts(_region_data("ERCOT"))
        assert result.ok

    def test_demo_mode_labels_alerts_as_demo(self, monkeypatch):
        """Demo mode keeps the demo alerts but labels their provenance."""
        monkeypatch.setattr(config, "USE_DEMO_DATA", True)
        captured: dict = {}

        def fake_set(key, payload, ttl=None):
            captured["payload"] = payload
            return True

        with patch("data.redis_client.redis_set", side_effect=fake_set):
            result = phases.write_alerts(_region_data("FPL"))

        assert result.ok
        payload = captured["payload"]
        assert payload["alerts_source"] == "demo"
        assert payload["alerts"], "demo mode should produce demo alerts for FPL"
        assert payload["stress_score"] is not None


# ---------------------------------------------------------------------------
# _alerts_tab_from_redis (web tier rendering)
# ---------------------------------------------------------------------------


def _payload(alerts_source, alerts, stress, stress_label):
    return {
        "region": "FPL",
        "alerts": alerts,
        "alerts_source": alerts_source,
        "stress_score": stress,
        "stress_label": stress_label,
        "alert_counts": {
            "critical": 0,
            "warning": len([a for a in alerts if a.get("severity") == "warning"]),
            "info": 0,
        },
        "anomaly": {},
        "temperature": {},
    }


class TestAlertsTabRedisRendering:
    def test_unavailable_payload_renders_no_feed_state(self):
        from components import _callbacks_alerts as mod

        payload = _payload("unavailable", [], None, "Unavailable")
        with patch.object(mod, "redis_get", return_value=payload):
            result = mod._alerts_tab_from_redis("FPL")

        assert result is not None
        alert_cards, stress_str, stress_span, breakdown = result[0], result[1], result[2], result[3]
        text = _collect_text(alert_cards)
        assert "temporarily unavailable" in text
        assert stress_str == "—"
        assert _collect_text(stress_span) == "Unavailable"
        assert "No alert feed" in _collect_text(breakdown)

    def test_demo_payload_carries_disclosure(self):
        from components import _callbacks_alerts as mod

        alerts = [{"event": "Heat Advisory", "headline": "hot", "severity": "warning"}]
        payload = _payload("demo", alerts, 35, "Elevated")
        with patch.object(mod, "redis_get", return_value=payload):
            result = mod._alerts_tab_from_redis("FPL")

        text = _collect_text(result[0])
        assert "Demo data" in text
        assert "Heat Advisory" in text

    def test_legacy_payload_without_source_treated_as_demo(self):
        """Pre-fix payloads only ever carried demo content — disclose them."""
        from components import _callbacks_alerts as mod

        payload = _payload(
            "demo", [{"event": "X", "headline": "y", "severity": "info"}], 20, "Normal"
        )
        del payload["alerts_source"]
        with patch.object(mod, "redis_get", return_value=payload):
            result = mod._alerts_tab_from_redis("FPL")
        assert "Demo data" in _collect_text(result[0])


# ---------------------------------------------------------------------------
# update_alerts_tab warming gate (REQUIRE_REDIS)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registered_app():
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
    )
    from components.layout import build_layout

    app.layout = build_layout()

    from components.callbacks import register_callbacks

    register_callbacks(app)
    return app


@pytest.fixture(scope="module")
def callbacks(registered_app):
    fns = {}
    for _key, val in registered_app.callback_map.items():
        fn = val.get("callback")
        if fn and hasattr(fn, "__name__"):
            raw = getattr(fn, "__wrapped__", fn)
            fns[fn.__name__] = raw
    return fns


class TestAlertsWarmingGate:
    def test_redis_miss_under_require_redis_renders_warming(self, callbacks, monkeypatch):
        from components import _callbacks_alerts as mod

        monkeypatch.setattr(mod, "REQUIRE_REDIS", True)
        with patch.object(mod, "redis_get", return_value=None):
            result = callbacks["update_alerts_tab"]("ERCOT", None, None, "tab-alerts")

        text = _collect_text(result[0])
        assert "warming" in text.lower()
        assert "Heat Advisory" not in text
        assert result[1] == "—"

    def test_redis_miss_dev_mode_still_falls_back_with_disclosure(self, callbacks, monkeypatch):
        from components import _callbacks_alerts as mod

        monkeypatch.setattr(mod, "REQUIRE_REDIS", False)
        fake_alerts = [{"event": "Heat Advisory", "headline": "hot", "severity": "warning"}]
        with (
            patch.object(mod, "redis_get", return_value=None),
            patch("data.demo_data.generate_demo_alerts", return_value=fake_alerts),
        ):
            result = callbacks["update_alerts_tab"]("FPL", None, None, "tab-alerts")

        text = _collect_text(result[0])
        assert "Demo data" in text


# ---------------------------------------------------------------------------
# Absent metrics must not render as perfection (P1-7 slice in this PR)
# ---------------------------------------------------------------------------


class TestNoFabricatedPerfection:
    def test_model_card_missing_metrics_render_as_unavailable(self):
        from components._callbacks_overview import _build_overview_model_card

        with (
            patch("models.model_service.get_model_metrics", return_value={"xgboost": {}}),
            patch("models.model_service.is_trained", return_value=True),
        ):
            card = _build_overview_model_card("ERCOT")
        text = _collect_text(card)
        assert "0.0%" not in text
        assert "—" in text


# ---------------------------------------------------------------------------
# Live NOAA wiring (2026-07): real alerts flow end-to-end, honestly labeled
# ---------------------------------------------------------------------------


def _weather_alert(i: int, severity: str = "warning", expires_h: float = 6.0):
    from datetime import UTC, datetime, timedelta

    from data.noaa_client import WeatherAlert

    return WeatherAlert(
        id=f"urn:test-{i}",
        event="Heat Advisory",
        headline=f"Alert {i}",
        description="d",
        severity=severity,
        noaa_severity="Moderate",
        urgency="Expected",
        certainty="Likely",
        onset=None,
        expires=datetime.now(UTC) + timedelta(hours=expires_h),
        areas=["County A"],
        states=["TX"],
        balancing_authorities=["ERCOT"],
    )


class TestLiveNOAAWiring:
    def test_prod_mode_publishes_real_alerts_as_noaa(self, monkeypatch):
        monkeypatch.setattr(config, "USE_DEMO_DATA", False)
        captured: dict = {}

        def fake_set(key, payload, ttl=None):
            captured["payload"] = payload
            return True

        fetched = [_weather_alert(1, "critical"), _weather_alert(2, "warning")]
        with (
            patch("data.redis_client.redis_set", side_effect=fake_set),
            patch("data.noaa_client.fetch_alerts_for_region", return_value=fetched),
        ):
            result = phases.write_alerts(_region_data("ERCOT"))

        assert result.ok
        payload = captured["payload"]
        assert payload["alerts_source"] == "noaa"
        assert payload["alerts_total"] == 2
        assert payload["alert_counts"] == {"critical": 1, "warning": 1, "info": 0}
        assert payload["stress_score"] == 30 + 15 + 20
        assert payload["alerts"][0]["event"] == "Heat Advisory"

    def test_expired_alerts_are_filtered(self, monkeypatch):
        monkeypatch.setattr(config, "USE_DEMO_DATA", False)
        captured: dict = {}
        fetched = [_weather_alert(1, expires_h=-1.0), _weather_alert(2, expires_h=4.0)]
        with (
            patch(
                "data.redis_client.redis_set",
                side_effect=lambda k, p, ttl=None: captured.update(payload=p) or True,
            ),
            patch("data.noaa_client.fetch_alerts_for_region", return_value=fetched),
        ):
            phases.write_alerts(_region_data("ERCOT"))
        assert captured["payload"]["alerts_total"] == 1
        assert len(captured["payload"]["alerts"]) == 1

    def test_cache_reconstructed_alerts_do_not_degrade_to_unavailable(self, monkeypatch):
        """Regression: alerts returned from the client's cache path carry
        onset/expires as reconstructed objects, not raw strings. Before the
        _alert_from_dict fix, write_alerts crashed on ``.expires.isoformat()``
        / ``.tzinfo`` and silently degraded every cache-hit BA to
        ``alerts_source="unavailable"`` — which is exactly what surfaced on
        the Risk tab. This exercises the real round-trip."""
        from data.noaa_client import _alert_from_dict, _alert_to_dict

        monkeypatch.setattr(config, "USE_DEMO_DATA", False)
        # Simulate what fetch_alerts_for_region returns on a cache hit.
        cache_shaped = [_alert_from_dict(_alert_to_dict(_weather_alert(1, "warning")))]
        captured: dict = {}
        with (
            patch(
                "data.redis_client.redis_set",
                side_effect=lambda k, p, ttl=None: captured.update(payload=p) or True,
            ),
            patch("data.noaa_client.fetch_alerts_for_region", return_value=cache_shaped),
        ):
            result = phases.write_alerts(_region_data("ERCOT"))

        assert result.ok
        assert captured["payload"]["alerts_source"] == "noaa"
        assert captured["payload"]["alerts_total"] == 1
        assert captured["payload"]["alerts"][0]["event"] == "Heat Advisory"

    def test_cap_is_disclosed_not_silent(self, monkeypatch):
        monkeypatch.setattr(config, "USE_DEMO_DATA", False)
        captured: dict = {}
        fetched = [_weather_alert(i) for i in range(30)]
        with (
            patch(
                "data.redis_client.redis_set",
                side_effect=lambda k, p, ttl=None: captured.update(payload=p) or True,
            ),
            patch("data.noaa_client.fetch_alerts_for_region", return_value=fetched),
        ):
            phases.write_alerts(_region_data("ERCOT"))
        payload = captured["payload"]
        assert len(payload["alerts"]) == phases._ALERTS_PAYLOAD_CAP
        assert payload["alerts_total"] == 30
        # Counts reflect ALL live alerts, not just the capped card list.
        assert payload["alert_counts"]["warning"] == 30

    def test_noaa_render_has_attribution_and_no_disclosure(self):
        from components import _callbacks_alerts as mod

        payload = _payload(
            "noaa",
            [{"event": "Heat Advisory", "headline": "hot", "severity": "warning"}],
            35,
            "Elevated",
        )
        payload["alerts_total"] = 5
        with patch.object(mod, "redis_get", return_value=payload):
            result = mod._alerts_tab_from_redis("ERCOT")
        text = _collect_text(result[0])
        assert "NOAA/NWS" in text
        assert "showing 1 of 5" in text
        assert "Demo data" not in text

    def test_noaa_zero_alerts_render_names_the_live_feed(self):
        from components import _callbacks_alerts as mod

        payload = _payload("noaa", [], 20, "Normal")
        with patch.object(mod, "redis_get", return_value=payload):
            result = mod._alerts_tab_from_redis("ERCOT")
        text = _collect_text(result[0])
        assert "No active severe-weather alerts" in text
        assert "NOAA/NWS" in text
