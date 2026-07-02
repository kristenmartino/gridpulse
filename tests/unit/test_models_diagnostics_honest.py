"""Models-tab diagnostics honesty regression tests.

Pins the interim fix for the 2026-07 critical-review finding P2-32 / issue
#166: in production the scoring job's ``write_diagnostics`` called the
strict-gated ``get_forecasts`` (which returns ``{"source": "unavailable"}``)
and the old ``diag.get("ensemble", actuals)`` default substituted actual
demand as the "prediction", writing identically-zero residuals that the
Models tab rendered as a perfect model.

After the fix:
* no real forecast ⇒ ``write_diagnostics`` writes an explicit
  ``diagnostics_source="unavailable"`` marker with NO fabricated residuals;
* an absent trained model ⇒ ``feature_importance`` is None, never the
  hardcoded ``[10, 9, 8, …]`` placeholder;
* the Models-tab renderer shows an honest empty state for the four residual
  charts (keeping the real metrics table + real SHAP).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from jobs import phases


def _region_data(region: str = "ERCOT") -> phases.RegionData:
    ts = pd.date_range("2026-06-01", periods=200, freq="h", tz="UTC")
    demand = pd.DataFrame(
        {"timestamp": ts, "demand_mw": 20000.0 + 3000.0 * np.sin(np.arange(200) * 2 * np.pi / 24)}
    )
    weather = pd.DataFrame({"timestamp": ts, "temperature_2m": 75.0})
    return phases.RegionData(region=region, demand_df=demand, weather_df=weather)


_REAL_XGB = {"feature_importances": {"demand_lag_1h": 0.5, "temperature_2m": 0.2, "hour_sin": 0.1}}


class TestWriteDiagnosticsHonesty:
    def test_unavailable_forecast_writes_marker_no_residuals(self):
        captured: dict = {}
        with (
            patch(
                "data.redis_client.redis_set",
                side_effect=lambda k, p, ttl=None: captured.update(key=k, payload=p) or True,
            ),
            patch(
                "models.model_service.get_forecasts",
                return_value={"source": "unavailable"},
            ),
        ):
            result = phases.write_diagnostics(_region_data(), _REAL_XGB)

        assert result.ok
        payload = captured["payload"]
        assert payload["diagnostics_source"] == "unavailable"
        # No fabricated residual series at all.
        assert "residuals" not in payload
        assert "ensemble" not in payload
        # Real SHAP is preserved (it comes from the trained model, not the forecast).
        assert payload["feature_importance"]["names"][0] == "demand_lag_1h"

    def test_no_model_yields_none_feature_importance_not_placeholder(self):
        captured: dict = {}
        with (
            patch(
                "data.redis_client.redis_set",
                side_effect=lambda k, p, ttl=None: captured.update(payload=p) or True,
            ),
            patch(
                "models.model_service.get_forecasts",
                return_value={"source": "unavailable"},
            ),
        ):
            phases.write_diagnostics(_region_data(), None)

        # Must be None, never the hardcoded [10, 9, 8, …] placeholder.
        assert captured["payload"]["feature_importance"] is None

    def test_real_forecast_writes_residuals(self):
        captured: dict = {}
        data = _region_data()
        # A genuine (non-actual) forecast → non-zero residuals.
        pred = data.demand_df["demand_mw"].values * 1.02
        with (
            patch(
                "data.redis_client.redis_set",
                side_effect=lambda k, p, ttl=None: captured.update(payload=p) or True,
            ),
            patch(
                "models.model_service.get_forecasts",
                return_value={"source": "trained", "ensemble": pred, "metrics": {}},
            ),
        ):
            result = phases.write_diagnostics(data, _REAL_XGB)

        assert result.ok
        payload = captured["payload"]
        assert payload["diagnostics_source"] == "trained"
        assert len(payload["residuals"]) == len(pred)
        assert not np.allclose(payload["residuals"], 0.0)  # genuinely non-zero


class TestModelsTabHonestRender:
    def _collect_text(self, fig) -> str:
        # empty-figure carries its message as an annotation.
        try:
            return " ".join(a.text or "" for a in fig.layout.annotations)
        except Exception:  # pragma: no cover
            return ""

    def test_unavailable_payload_renders_empty_residual_charts(self):
        from components import _callbacks_models as mod

        payload = {
            "region": "ERCOT",
            "diagnostics_source": "unavailable",
            "feature_importance": {"names": ["demand_lag_1h"], "values": [0.5]},
        }
        with (
            patch.object(mod, "redis_get", return_value=payload),
            patch("models.model_service.get_model_metrics", return_value={}),
        ):
            result = mod._models_tab_from_redis(
                "ERCOT", ["prophet", "arima", "xgboost", "ensemble"]
            )

        assert result is not None
        table, f_time, f_hist, f_pred, f_heat, f_shap = result
        # The four residual charts must show the honest unavailable message,
        # not a flat-zero fabrication.
        for fig in (f_time, f_hist, f_pred, f_heat):
            assert "unavailable" in self._collect_text(fig).lower()
        # Real SHAP still renders (real importances present).
        assert (
            self._collect_text(f_shap) == ""
            or "unavailable" not in self._collect_text(f_shap).lower()
        )

    def test_shap_fig_empty_when_no_importances(self):
        from components._callbacks_models import _build_shap_fig

        fig = _build_shap_fig({}, ["xgboost", "ensemble"], "ERCOT:x")
        text = " ".join(a.text or "" for a in fig.layout.annotations)
        assert "unavailable" in text.lower()

    def test_shap_fig_renders_with_real_importances(self):
        from components._callbacks_models import _build_shap_fig

        fig = _build_shap_fig(
            {"names": ["demand_lag_1h", "temperature_2m"], "values": [0.5, 0.2]},
            ["xgboost"],
            "ERCOT:x",
        )
        # A real bar trace, no unavailable annotation.
        assert len(fig.data) == 1
        assert list(fig.data[0].y) == ["temperature_2m", "demand_lag_1h"]
