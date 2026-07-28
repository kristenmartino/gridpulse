"""A served baseline must never render as a model forecast.

The scoring job substitutes a seasonal-naive series for regions whose model
measurably loses to it (``models.skill``). That series lands in
``predicted_demand_mw`` — the same key a model forecast uses — so every
surface that plots it has to check, or it silently attributes a naive
projection to a trained forecaster.

This is the last surface in that arc: the API already discloses
(``series_source: "seasonal-naive-baseline"``), the dashboard did not.
"""

from __future__ import annotations

from components._callbacks_forecast import (
    BASELINE_SERIES_LABEL,
    _served_model_for_payload,
    is_baseline_served,
)


def _payload(**over):
    payload = {
        "region": "SEC",
        "primary_model": "arima",
        "forecasts": [{"timestamp": "2026-07-28T00:00:00+00:00", "predicted_demand_mw": 300.0}],
    }
    payload.update(over)
    return payload


class TestDetection:
    def test_detects_a_substituted_payload(self):
        assert is_baseline_served(_payload(served_series="seasonal-naive")) is True

    def test_a_model_payload_is_not_baseline(self):
        assert is_baseline_served(_payload(served_series="model")) is False

    def test_a_legacy_payload_without_the_field_is_not_baseline(self):
        """Payloads written before the substitution shipped carry no
        ``served_series``. They must read as model forecasts, not baselines —
        the flag's absence means "not substituted", never "unknown"."""
        assert is_baseline_served(_payload()) is False

    def test_junk_is_not_baseline(self):
        assert is_baseline_served(None) is False
        assert is_baseline_served("seasonal-naive") is False


class TestAttribution:
    def test_a_substituted_region_resolves_to_no_model(self):
        """Naming any model here would attribute a naive projection to a
        trained forecaster on every label the tab renders."""
        served = _served_model_for_payload(_payload(served_series="seasonal-naive"), "xgboost")
        assert served == BASELINE_SERIES_LABEL
        assert "xgboost" not in served
        assert "arima" not in served

    def test_the_label_is_not_a_model_name(self):
        """It flows into `.upper()`-style labels, so it must read as a
        description rather than something a user could mistake for a model
        in the dropdown."""
        assert BASELINE_SERIES_LABEL not in {"xgboost", "prophet", "arima", "ensemble"}
        assert "baseline" in BASELINE_SERIES_LABEL

    def test_an_explicitly_requested_model_still_resolves_to_itself(self):
        """A payload can carry per-model rows alongside a substituted
        headline series — the substitution replaces what is SERVED, and the
        models are kept as evidence. Asking for one of them by name gets it."""
        payload = _payload(
            served_series="seasonal-naive",
            forecasts=[{"timestamp": "t", "predicted_demand_mw": 300.0, "prophet": 310.0}],
        )
        assert _served_model_for_payload(payload, "prophet") == "prophet"

    def test_normal_payloads_are_unaffected(self):
        """Flag-dark today: every non-substituted payload must resolve
        exactly as it did before."""
        assert _served_model_for_payload(_payload(), "xgboost") == "arima"
        payload = _payload(forecasts=[{"timestamp": "t", "xgboost": 1.0}])
        assert _served_model_for_payload(payload, "xgboost") == "xgboost"


class TestHeadlineFollowsTheLabel:
    """The bug this class exists for: the API published `series_source:
    "seasonal-naive-baseline"` while `demand_mw` carried the ENSEMBLE's
    numbers, because the row builder preferred the `ensemble` column and the
    substitution writes `predicted_demand_mw` / `baseline`.

    A false label is worse than no substitution — a consumer trusts it.
    """

    @staticmethod
    def _redis(payload):
        def _get(key):
            return payload if "forecast:" in key else None

        return _get

    @staticmethod
    def _payload(**over):
        p = {
            "region": "SEC",
            "scored_at": "2026-07-28T12:00:00+00:00",
            "primary_model": "arima",
            "forecasts": [
                {
                    "timestamp": "2026-07-28T12:00:00+00:00",
                    "predicted_demand_mw": 300.0,
                    "baseline": 300.0,
                    "ensemble": 241.3,
                    "arima": 249.2,
                    "xgboost": 261.6,
                }
            ],
        }
        p.update(over)
        return p

    def test_substituted_region_publishes_the_baseline_numbers(self, monkeypatch):
        import api as api_module

        api_module._memo.clear()
        monkeypatch.setattr(
            api_module, "redis_get", self._redis(self._payload(served_series="seasonal-naive"))
        )
        from flask import Flask

        app = Flask(__name__)
        app.register_blueprint(api_module.api_v1)
        body = app.test_client().get("/api/v1/forecast/SEC?horizon=1").get_json()

        assert body["series_source"] == "seasonal-naive-baseline"
        assert body["forecast"][0]["demand_mw"] == 300.0, "label says baseline, numbers must be too"
        assert body["forecast"][0]["demand_mw"] != 241.3, (
            "served the ensemble under a baseline label"
        )
        # the models stay visible as the evidence for the substitution
        assert body["forecast"][0]["by_model"]["ensemble"] == 241.3

    def test_normal_region_still_publishes_the_ensemble(self, monkeypatch):
        import api as api_module

        api_module._memo.clear()
        monkeypatch.setattr(api_module, "redis_get", self._redis(self._payload()))
        from flask import Flask

        app = Flask(__name__)
        app.register_blueprint(api_module.api_v1)
        body = app.test_client().get("/api/v1/forecast/SEC?horizon=1").get_json()

        assert body["series_source"] == "ensemble"
        assert body["forecast"][0]["demand_mw"] == 241.3
