"""Models-tab diagnostics honesty regression tests.

History (P2-32 / #166 / #220): the original ``write_diagnostics`` substituted
actual demand as the "prediction" and wrote identically-zero residuals; the
#166 interim fix wrote an honest ``unavailable`` marker — but sourced from the
legacy v1 ``get_forecasts``, which is strict-gated in production and NEVER
produces a series on the job container, leaving the Models tab's four residual
panels permanently empty in prod (#220).

Current contract (pinned here):
* residuals come from the Redis walk-forward BACKTEST payload
  (``backtest:{exog_mode}:{region}:{horizon}``, nightly training job), 24h
  horizon preferred, with provenance (``residual_source``) on the payload;
* no backtest yet ⇒ honest ``unavailable`` marker with reason
  ``no_backtest_yet`` and NO fabricated residuals;
* an absent trained model ⇒ ``feature_importance`` is None, never a placeholder;
* the renderer shows the honest empty state (with the TRUE self-heal copy) for
  the four residual charts, keeps the real metrics table + SHAP, and — when
  residuals exist — captions all four charts with backtest provenance.
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


def _backtest_payload(n=120, offset=400.0, models=("xgboost",), n_timestamps=None):
    ts = pd.date_range("2026-06-20", periods=n, freq="h", tz="UTC")
    actual = 20000.0 + 3000.0 * np.sin(np.arange(n) * 2 * np.pi / 24)
    n_ts = n if n_timestamps is None else n_timestamps
    return {
        "horizon": 24,
        "actual": actual.tolist(),
        # dict-insertion order preserved: a non-xgboost model listed first
        # exercises the xgboost-preference branch.
        "predictions": {m: (actual - offset * (i + 1)).tolist() for i, m in enumerate(models)},
        "timestamps": [t.isoformat() for t in ts[:n_ts]],
        "metrics": {"xgboost": {"mape": 2.0, "rmse": 450.0, "mae": 400.0, "r2": 0.97}},
    }


def _fake_backtest_redis(payloads_by_horizon: dict[int, dict]):
    def _get(key: str):
        for h, p in payloads_by_horizon.items():
            if key.endswith(f":{h}"):
                return p
        return None

    return _get


#: Structural pin for the #220 root cause: write_diagnostics must NEVER touch
#: the strict-gated web-tier v1 forecast path again — a "fallback" re-adding it
#: would ship green in dev (where get_forecasts simulates) and re-break prod.
_FORBID_V1 = patch(
    "models.model_service.get_forecasts",
    side_effect=AssertionError(
        "web-tier v1 get_forecasts must not be called from write_diagnostics (#220)"
    ),
)


class TestWriteDiagnosticsFromBacktests:
    def test_no_backtest_writes_honest_unavailable_marker(self):
        """Fresh deploy, pre-first-training-run: unavailable marker with the
        TRUE reason, no fabricated residuals, real SHAP preserved."""
        captured: dict = {}
        with (
            _FORBID_V1,
            patch("data.redis_client.redis_get", return_value=None),
            patch(
                "data.redis_client.redis_set",
                side_effect=lambda k, p, ttl=None: captured.update(key=k, payload=p) or True,
            ),
        ):
            result = phases.write_diagnostics(_region_data(), _REAL_XGB)

        assert result.ok
        payload = captured["payload"]
        assert payload["diagnostics_source"] == "unavailable"
        assert payload["reason"] == "no_backtest_yet"
        assert "residuals" not in payload
        assert "predicted" not in payload
        assert payload["feature_importance"]["names"][0] == "demand_lag_1h"

    def test_backtest_present_writes_real_residuals_with_provenance(self):
        """#220 fix: residuals come from the walk-forward backtest payload —
        genuinely non-zero, provenance names the horizon + model."""
        captured: dict = {}
        with (
            _FORBID_V1,
            patch(
                "data.redis_client.redis_get",
                side_effect=_fake_backtest_redis({24: _backtest_payload()}),
            ),
            patch(
                "data.redis_client.redis_set",
                side_effect=lambda k, p, ttl=None: captured.update(payload=p) or True,
            ),
        ):
            result = phases.write_diagnostics(_region_data(), _REAL_XGB)

        assert result.ok
        payload = captured["payload"]
        assert payload["diagnostics_source"] == "backtest"
        assert payload["residual_source"] == {
            "kind": "walk_forward_backtest",
            "horizon": 24,
            "model": "xgboost",
            "exog_mode": phases.DEFAULT_BACKTEST_EXOG_MODE,
        }
        assert len(payload["residuals"]) == 120
        assert np.allclose(payload["residuals"], 400.0)  # actual - pred = +400, not zeros
        assert "predicted" in payload  # canonical name — not mislabeled "ensemble"
        assert "ensemble" not in payload
        assert len(payload["hourly_error"]["hours"]) == 24
        assert payload["metrics"]["xgboost"]["mape"] == 2.0

    def test_prefers_24h_horizon_over_deeper(self):
        """Day-ahead residuals are the operational standard — 24h wins when
        multiple horizons exist."""
        captured: dict = {}
        deep = _backtest_payload(n=168, offset=900.0)
        with (
            _FORBID_V1,
            patch(
                "data.redis_client.redis_get",
                side_effect=_fake_backtest_redis({24: _backtest_payload(), 168: deep}),
            ),
            patch(
                "data.redis_client.redis_set",
                side_effect=lambda k, p, ttl=None: captured.update(payload=p) or True,
            ),
        ):
            phases.write_diagnostics(_region_data(), _REAL_XGB)

        assert captured["payload"]["residual_source"]["horizon"] == 24

    def test_falls_back_to_deeper_horizon_when_24h_missing(self):
        captured: dict = {}
        with (
            _FORBID_V1,
            patch(
                "data.redis_client.redis_get",
                side_effect=_fake_backtest_redis({168: _backtest_payload(n=168)}),
            ),
            patch(
                "data.redis_client.redis_set",
                side_effect=lambda k, p, ttl=None: captured.update(payload=p) or True,
            ),
        ):
            phases.write_diagnostics(_region_data(), _REAL_XGB)

        assert captured["payload"]["diagnostics_source"] == "backtest"
        assert captured["payload"]["residual_source"]["horizon"] == 168

    def test_short_backtest_loses_to_deeper_horizon(self):
        """The ≥24-aligned-rows gate: a degenerate 10-row 24h payload must not
        win over a full 168h payload."""
        captured: dict = {}
        with (
            _FORBID_V1,
            patch(
                "data.redis_client.redis_get",
                side_effect=_fake_backtest_redis(
                    {24: _backtest_payload(n=10), 168: _backtest_payload(n=168)}
                ),
            ),
            patch(
                "data.redis_client.redis_set",
                side_effect=lambda k, p, ttl=None: captured.update(payload=p) or True,
            ),
        ):
            phases.write_diagnostics(_region_data(), _REAL_XGB)

        assert captured["payload"]["residual_source"]["horizon"] == 168

    def test_only_short_backtest_writes_unavailable(self):
        captured: dict = {}
        with (
            _FORBID_V1,
            patch(
                "data.redis_client.redis_get",
                side_effect=_fake_backtest_redis({24: _backtest_payload(n=10)}),
            ),
            patch(
                "data.redis_client.redis_set",
                side_effect=lambda k, p, ttl=None: captured.update(payload=p) or True,
            ),
        ):
            phases.write_diagnostics(_region_data(), _REAL_XGB)

        assert captured["payload"]["diagnostics_source"] == "unavailable"
        assert captured["payload"]["reason"] == "no_backtest_yet"

    def test_xgboost_preferred_in_multi_model_payload(self):
        """predictions listing prophet FIRST must still calibrate on xgboost —
        and the residuals must match the xgboost series, not prophet's."""
        captured: dict = {}
        with (
            _FORBID_V1,
            patch(
                "data.redis_client.redis_get",
                side_effect=_fake_backtest_redis(
                    # prophet offset=400, xgboost offset=800 (models order i+1)
                    {24: _backtest_payload(models=("prophet", "xgboost"))}
                ),
            ),
            patch(
                "data.redis_client.redis_set",
                side_effect=lambda k, p, ttl=None: captured.update(payload=p) or True,
            ),
        ):
            phases.write_diagnostics(_region_data(), _REAL_XGB)

        payload = captured["payload"]
        assert payload["residual_source"]["model"] == "xgboost"
        assert np.allclose(payload["residuals"], 800.0)  # xgboost's offset, not prophet's 400

    def test_misaligned_timestamps_payload_is_skipped(self):
        """Verify catch: a payload whose timestamps don't cover the series is
        UNUSABLE — it loses to the next horizon rather than being written with
        empty timestamps (which crashed the renderer) or index-synthesized
        hour-of-day (fabricated attribution)."""
        captured: dict = {}
        bad_24 = _backtest_payload(n=120, n_timestamps=5)  # timestamps too short
        with (
            _FORBID_V1,
            patch(
                "data.redis_client.redis_get",
                side_effect=_fake_backtest_redis({24: bad_24, 168: _backtest_payload(n=168)}),
            ),
            patch(
                "data.redis_client.redis_set",
                side_effect=lambda k, p, ttl=None: captured.update(payload=p) or True,
            ),
        ):
            phases.write_diagnostics(_region_data(), _REAL_XGB)

        payload = captured["payload"]
        assert payload["residual_source"]["horizon"] == 168  # bad 24h skipped
        # And the written payload is always fully aligned.
        assert len(payload["timestamps"]) == len(payload["residuals"])

    def test_no_model_yields_none_feature_importance_not_placeholder(self):
        captured: dict = {}
        with (
            _FORBID_V1,
            patch("data.redis_client.redis_get", return_value=None),
            patch(
                "data.redis_client.redis_set",
                side_effect=lambda k, p, ttl=None: captured.update(payload=p) or True,
            ),
        ):
            phases.write_diagnostics(_region_data(), None)

        # Must be None, never the hardcoded [10, 9, 8, …] placeholder.
        assert captured["payload"]["feature_importance"] is None


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
            "reason": "no_backtest_yet",
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
        for fig in (f_time, f_hist, f_pred, f_heat):
            text = self._collect_text(fig).lower()
            assert "unavailable" in text
            # #220: the copy states the TRUE self-heal condition (nightly
            # training backtests), not the old false promise about scoring.
            assert "training job" in text
            assert "scoring job" not in text
        # Real SHAP still renders (real importances present).
        assert (
            self._collect_text(f_shap) == ""
            or "unavailable" not in self._collect_text(f_shap).lower()
        )

    def test_backtest_payload_renders_residuals_with_provenance(self):
        """The populated path (#220): four real charts, each carrying the
        walk-forward-backtest provenance caption."""
        from components import _callbacks_models as mod

        n = 120
        ts = pd.date_range("2026-06-20", periods=n, freq="h", tz="UTC")
        payload = {
            "region": "ERCOT",
            "diagnostics_source": "backtest",
            "residual_source": {
                "kind": "walk_forward_backtest",
                "horizon": 24,
                "model": "xgboost",
                "exog_mode": "forecast_exog",
            },
            "timestamps": [t.isoformat() for t in ts],
            "actual": [20000.0] * n,
            "predicted": [19600.0] * n,
            "residuals": [400.0] * n,
            "hourly_error": {"hours": list(range(24)), "values": [400.0] * 24},
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
        _, f_time, f_hist, f_pred, f_heat, _ = result
        for fig in (f_time, f_hist, f_pred, f_heat):
            assert len(fig.data) >= 1  # a real trace, not the empty placeholder
            text = self._collect_text(fig)
            assert "24h walk-forward backtest residuals" in text
            assert "XGBOOST" in text
        # residuals-vs-predicted reads the canonical "predicted" series.
        assert float(np.asarray(f_pred.data[0].x)[0]) == 19600.0

    def test_legacy_ensemble_field_still_renders(self):
        """Back-compat: a pre-#220 payload (predictions under "ensemble", no
        residual_source) renders the charts — without a provenance caption."""
        from components import _callbacks_models as mod

        n = 48
        ts = pd.date_range("2026-06-20", periods=n, freq="h", tz="UTC")
        payload = {
            "region": "ERCOT",
            "diagnostics_source": "trained",
            "timestamps": [t.isoformat() for t in ts],
            "actual": [20000.0] * n,
            "ensemble": [19700.0] * n,
            "residuals": [300.0] * n,
            "hourly_error": {"hours": list(range(24)), "values": [300.0] * 24},
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
        _, f_time, _, f_pred, _, _ = result
        assert len(f_time.data) >= 1
        assert float(np.asarray(f_pred.data[0].x)[0]) == 19700.0  # legacy field read
        assert "walk-forward" not in self._collect_text(f_time)  # no false provenance

    def test_mismatched_timestamps_render_degrades_honestly(self):
        """Reader belt-and-braces (verify catch): a malformed payload with
        non-empty residuals but mismatched timestamps must show the honest
        unavailable state — never crash the whole Models-tab callback in
        lttb_downsample."""
        from components import _callbacks_models as mod

        payload = {
            "region": "ERCOT",
            "diagnostics_source": "backtest",
            "residual_source": {"kind": "walk_forward_backtest", "horizon": 24, "model": "xgboost"},
            "timestamps": [],  # malformed: residuals present, timestamps missing
            "actual": [20000.0] * 48,
            "predicted": [19600.0] * 48,
            "residuals": [400.0] * 48,
            "hourly_error": {"hours": list(range(24)), "values": [400.0] * 24},
            "feature_importance": {"names": ["demand_lag_1h"], "values": [0.5]},
        }
        with (
            patch.object(mod, "redis_get", return_value=payload),
            patch("models.model_service.get_model_metrics", return_value={}),
        ):
            result = mod._models_tab_from_redis(
                "ERCOT", ["prophet", "arima", "xgboost", "ensemble"]
            )

        assert result is not None  # no exception — degraded, not dead
        _, f_time, _, _, _, _ = result
        assert "unavailable" in self._collect_text(f_time).lower()

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
