"""#283 Phase 3b — lead-time-resolved (widening) P10–P90 forecast band.

`_widening_interval_from_backtests` anchors empirical error quantiles at the
24h/168h/720h backtest lead times; `_add_confidence_bands` interpolates them
across the chart's lead axis with monotone widening and a non-negative floor.
Falls back to the flat empirical estimator, then the heuristic envelope.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import plotly.graph_objects as go

import components._callbacks_shared as shared
from components._callbacks_forecast import _add_confidence_bands
from components._callbacks_shared import _widening_interval_from_backtests


def _fake_backtest_store(spreads: dict[int, float], region="DUK"):
    """Build a fake redis_get returning backtest payloads whose residual spread
    (uniform ±spread) differs per horizon. Residual count == horizon."""
    rng = np.random.default_rng(7)

    def _get(key: str):
        for h, spread in spreads.items():
            if key.endswith(f":{region}:{h}"):
                n = max(h, 48)
                actual = 40_000 + rng.normal(0, 1, n)
                # symmetric uniform-ish residuals with the requested spread
                resid = spread * np.linspace(-1, 1, n)
                preds = actual - resid
                return {
                    "actual": actual.tolist(),
                    "predictions": {"xgboost": preds.tolist()},
                    "timestamps": ["2026-01-01T00:00:00"] * n,
                }
        return None

    return _get


class TestWideningIntervalFromBacktests:
    def setup_method(self):
        shared._BACKTEST_CACHE.clear()

    def test_three_horizons_yield_sorted_widening_anchors(self):
        with patch(
            "components._callbacks_shared.redis_get",
            side_effect=_fake_backtest_store({24: 300.0, 168: 900.0, 720: 2500.0}),
        ):
            out = _widening_interval_from_backtests("DUK", "xgboost")
        assert out["available"] is True
        hs = [a["horizon"] for a in out["anchors"]]
        assert hs == [24, 168, 720]
        uppers = [a["upper_error"] for a in out["anchors"]]
        assert uppers[0] < uppers[1] < uppers[2]  # spread grows with lead
        assert out["calibration_model"] == "xgboost"

    def test_single_horizon_is_unavailable(self):
        """<2 anchors → unavailable → caller falls back to the flat estimator."""
        with patch(
            "components._callbacks_shared.redis_get",
            side_effect=_fake_backtest_store({168: 900.0}),
        ):
            out = _widening_interval_from_backtests("DUK", "xgboost")
        assert out == {"available": False}

    def test_substitute_calibration_disclosed(self):
        """Backtests carry only XGBoost residuals; a prophet request must name
        xgboost as the calibration model (P1-2 disclosure contract)."""
        with patch(
            "components._callbacks_shared.redis_get",
            side_effect=_fake_backtest_store({24: 300.0, 720: 2500.0}),
        ):
            out = _widening_interval_from_backtests("DUK", "prophet")
        assert out["available"] is True
        assert out["calibration_model"] == "xgboost"


def _widening(anchors):
    return {
        "available": True,
        "anchors": anchors,
        "target_coverage": 0.80,
        "calibration_model": "xgboost",
    }


class TestAddConfidenceBandsWidening:
    def _run(self, widening, preds=None, horizon=720):
        preds = preds if preds is not None else np.full(horizon, 15_000.0)
        ts = np.arange(horizon)
        fig = go.Figure()
        with patch(
            "components._callbacks_forecast._widening_interval_from_backtests",
            return_value=widening,
        ):
            meta = _add_confidence_bands(fig, ts, preds, horizon, region="DUK")
        upper = np.asarray(fig.data[0].y, dtype=float)
        lower = np.asarray(fig.data[1].y, dtype=float)
        return meta, upper, lower, fig

    def test_band_widens_monotonically(self):
        anchors = [
            {"horizon": 24, "lower_error": -300.0, "upper_error": 300.0, "sample_size": 24},
            {"horizon": 168, "lower_error": -900.0, "upper_error": 900.0, "sample_size": 168},
            {"horizon": 720, "lower_error": -2500.0, "upper_error": 2500.0, "sample_size": 720},
        ]
        meta, upper, lower, fig = self._run(_widening(anchors))
        assert meta["method"] == "empirical_widening"
        width = upper - lower
        assert np.all(np.diff(width) >= -1e-9)  # monotone non-shrinking
        assert width[-1] > width[0] * 3  # genuinely widens toward the tail
        # P50 = the forecast line; band trace labeled as the P10–P90 fan
        assert any("P10–P90" in (tr.name or "") for tr in fig.data)

    def test_out_of_order_anchors_still_monotone(self):
        """A 720h backtest can sample narrower than 168h (single-origin noise);
        cummax/cummin must keep the band from shrinking with lead."""
        anchors = [
            {"horizon": 24, "lower_error": -300.0, "upper_error": 300.0, "sample_size": 24},
            {"horizon": 168, "lower_error": -1200.0, "upper_error": 1200.0, "sample_size": 168},
            {"horizon": 720, "lower_error": -800.0, "upper_error": 800.0, "sample_size": 720},
        ]
        _, upper, lower, _ = self._run(_widening(anchors))
        width = upper - lower
        assert np.all(np.diff(width) >= -1e-9)
        assert width[-1] >= 2 * 1200.0 - 1e-9  # held at the widest anchor, not shrunk

    def test_lower_band_floored_at_zero(self):
        """Deep-tail P10 offset larger than a small forecast → floor at 0, never
        a negative-demand band edge (#282 consistency)."""
        anchors = [
            {"horizon": 24, "lower_error": -300.0, "upper_error": 300.0, "sample_size": 24},
            {"horizon": 720, "lower_error": -5000.0, "upper_error": 5000.0, "sample_size": 720},
        ]
        _, _, lower, _ = self._run(_widening(anchors), preds=np.full(720, 2000.0))
        assert (lower >= 0.0).all()
        assert lower[-1] == 0.0  # the floor actually engaged in the deep tail

    def test_short_view_degrades_to_flat_near_anchor(self):
        """On a 24h view every lead ≤ the first anchor → constant band (np.interp
        holds the end value) — no artificial widening on short horizons."""
        anchors = [
            {"horizon": 24, "lower_error": -300.0, "upper_error": 300.0, "sample_size": 24},
            {"horizon": 720, "lower_error": -2500.0, "upper_error": 2500.0, "sample_size": 720},
        ]
        _, upper, lower, _ = self._run(_widening(anchors), horizon=24)
        width = upper - lower
        assert np.allclose(width, 600.0)

    def test_unavailable_falls_back_to_flat_empirical(self):
        """Widening unavailable → the pre-3b flat empirical path (regression pin
        of the fallback chain)."""
        fig = go.Figure()
        with (
            patch(
                "components._callbacks_forecast._widening_interval_from_backtests",
                return_value={"available": False},
            ),
            patch(
                "components._callbacks_forecast._empirical_interval_from_backtests",
                return_value={
                    "available": True,
                    "lower_error": -500.0,
                    "upper_error": 500.0,
                    "sample_size": 720,
                    "target_coverage": 0.80,
                    "calibration_window_hours": 720,
                    "calibration_model": "xgboost",
                },
            ),
        ):
            meta = _add_confidence_bands(
                fig, np.arange(720), np.full(720, 15_000.0), 720, region="DUK"
            )
        assert meta["method"] == "empirical"
        upper = np.asarray(fig.data[0].y, dtype=float)
        lower = np.asarray(fig.data[1].y, dtype=float)
        assert np.allclose(upper - lower, 1000.0)  # flat, as before Phase 3b
