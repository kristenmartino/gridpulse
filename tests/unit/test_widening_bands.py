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
from components._callbacks_forecast import _add_confidence_bands, _interval_caption
from components._callbacks_shared import _widening_interval_from_backtests


def _fake_backtest_store(spreads: dict[int, float], region="DUK", holdout=None, counts=None):
    """Build a fake redis_get returning per-horizon backtest payloads whose
    residual spread (uniform ±spread) differs per horizon. Matches ONLY the
    exog-mode key form so residuals aren't double-counted across the two key
    variants the collector probes. Optionally serves a horizon-agnostic
    ``holdout:{region}`` payload (which the widening estimator must ignore)."""
    rng = np.random.default_rng(7)

    def _payload(spread, n, model="xgboost"):
        actual = 40_000 + rng.normal(0, 1, n)
        resid = spread * np.linspace(-1, 1, n)  # symmetric ±spread
        return {
            "actual": actual.tolist(),
            "predictions": {model: (actual - resid).tolist()},
            "timestamps": ["2026-01-01T00:00:00"] * n,
        }

    def _get(key: str):
        if key.endswith(f"holdout:{region}"):
            return holdout
        if ":forecast_exog:" not in key:
            return None  # legacy key form — empty, prevents double-count
        for h, spread in spreads.items():
            if key.endswith(f":{region}:{h}"):
                n = (counts or {}).get(h, max(h, 48))
                return _payload(spread, n)
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
        assert [a["horizon"] for a in out["anchors"]] == [24, 168, 720]
        # Anchors pinned at the pool's EFFECTIVE lead (~H/2), not at H — a
        # horizon-H backtest pools leads 1..H, so its quantiles measure roughly
        # the mid-window error.
        assert [a["effective_lead"] for a in out["anchors"]] == [12, 84, 360]
        # sample sizes match the single (exog-key) payload — a double-count
        # across the two key forms would double these.
        assert [a["sample_size"] for a in out["anchors"]] == [48, 168, 720]
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

    def test_thin_horizon_anchor_dropped(self):
        """A horizon whose payload has too few residuals (< max(24, H/2)) drops
        that anchor rather than producing a noisy pin."""
        with patch(
            "components._callbacks_shared.redis_get",
            side_effect=_fake_backtest_store(
                {24: 300.0, 168: 900.0, 720: 2500.0}, counts={720: 100}
            ),
        ):
            out = _widening_interval_from_backtests("DUK", "xgboost")
        assert out["available"] is True
        assert [a["horizon"] for a in out["anchors"]] == [24, 168]  # 720 dropped

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

    def test_horizon_agnostic_holdout_excluded(self):
        """THE Phase-3b verification HIGH: the 168h training-holdout pool is
        horizon-AGNOSTIC — feeding it into every anchor (as the flat estimator's
        exact-beats-substitute rule would) collapses the per-horizon spread into
        a window-size artifact (a fake fan) and drops the 720h anchor. The
        widening estimator must IGNORE it: per-horizon xgboost substitutes win,
        all three anchors survive, and the substitution is disclosed."""
        holdout = {
            # exact ensemble residuals, tiny spread — poisonous if pooled
            "actual": (40_000 + np.zeros(168)).tolist(),
            "predictions": {"ensemble": (40_000 + np.linspace(-50, 50, 168)).tolist()},
        }
        with patch(
            "components._callbacks_shared.redis_get",
            side_effect=_fake_backtest_store({24: 300.0, 168: 900.0, 720: 2500.0}, holdout=holdout),
        ):
            out = _widening_interval_from_backtests("DUK", "ensemble")
        assert out["available"] is True
        assert [a["horizon"] for a in out["anchors"]] == [24, 168, 720]  # 720 survives
        uppers = [a["upper_error"] for a in out["anchors"]]
        assert uppers[2] > uppers[0] * 3  # genuine per-horizon spread, not ~±50 noise
        assert out["calibration_model"] == "xgboost"  # substitution disclosed

    def test_flat_estimator_still_uses_holdout(self):
        """Regression pin: the FLAT estimator keeps its documented trade-off
        (exact-model holdout beats per-horizon substitutes)."""
        holdout = {
            "actual": (40_000 + np.zeros(168)).tolist(),
            "predictions": {"ensemble": (40_000 + np.linspace(-50, 50, 168)).tolist()},
        }
        with patch(
            "components._callbacks_shared.redis_get",
            side_effect=_fake_backtest_store({168: 900.0}, holdout=holdout),
        ):
            out = shared._empirical_interval_from_backtests("DUK", "ensemble", 168)
        assert out["available"] is True
        assert out["calibration_model"] == "ensemble"  # exact holdout won
        assert abs(out["upper_error"]) <= 60  # calibrated on the ±50 holdout pool


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

    def test_effective_lead_pins_shift_the_widening_earlier(self):
        """Anchors carrying effective_lead interpolate on it (not the raw
        horizon): with pins at 12/360, the mid-chart band is wider than a
        naive 24/720-pinned interp would be."""
        anchors = [
            {
                "horizon": 24,
                "effective_lead": 12,
                "lower_error": -300.0,
                "upper_error": 300.0,
                "sample_size": 24,
            },
            {
                "horizon": 720,
                "effective_lead": 360,
                "lower_error": -2500.0,
                "upper_error": 2500.0,
                "sample_size": 720,
            },
        ]
        _, upper, lower, _ = self._run(_widening(anchors))
        width = upper - lower
        # At lead 360 the full 720-anchor width is already reached (pin at 360,
        # not 720) and holds flat beyond.
        assert np.isclose(width[359], 5000.0)
        assert np.allclose(width[360:], 5000.0)

    def test_substitute_calibration_disclosed_in_band_name(self):
        """P1-2: a prophet view calibrated on xgboost residuals must say so in
        the legend."""
        anchors = [
            {"horizon": 24, "lower_error": -300.0, "upper_error": 300.0, "sample_size": 24},
            {"horizon": 720, "lower_error": -2500.0, "upper_error": 2500.0, "sample_size": 720},
        ]
        w = _widening(anchors)  # calibration_model == "xgboost"
        preds = np.full(720, 15_000.0)
        fig = go.Figure()
        with patch(
            "components._callbacks_forecast._widening_interval_from_backtests",
            return_value=w,
        ):
            _add_confidence_bands(
                fig, np.arange(720), preds, 720, region="DUK", model_name="prophet"
            )
        assert any("(xgboost-calibrated)" in (tr.name or "") for tr in fig.data)

    def test_widening_preferred_over_flat_when_both_available(self):
        """Precedence pin: the lead-resolved band wins when both estimators
        could serve."""
        anchors = [
            {"horizon": 24, "lower_error": -300.0, "upper_error": 300.0, "sample_size": 24},
            {"horizon": 720, "lower_error": -2500.0, "upper_error": 2500.0, "sample_size": 720},
        ]
        fig = go.Figure()
        with (
            patch(
                "components._callbacks_forecast._widening_interval_from_backtests",
                return_value=_widening(anchors),
            ),
            patch(
                "components._callbacks_forecast._empirical_interval_from_backtests",
            ) as flat,
        ):
            meta = _add_confidence_bands(
                fig, np.arange(720), np.full(720, 15_000.0), 720, region="DUK"
            )
        assert meta["method"] == "empirical_widening"
        flat.assert_not_called()  # never consulted when widening is available

    def test_degenerate_negative_upper_error_keeps_edges_ordered(self):
        """A systematically over-forecasting model can have q90 < 0. After the
        lower floor, the band must stay ordered (upper ≥ lower ≥ 0) — never an
        inverted fill."""
        anchors = [
            {"horizon": 24, "lower_error": -900.0, "upper_error": -200.0, "sample_size": 24},
            {"horizon": 720, "lower_error": -6000.0, "upper_error": -500.0, "sample_size": 720},
        ]
        _, upper, lower, _ = self._run(_widening(anchors), preds=np.full(720, 800.0))
        assert (lower >= 0.0).all()
        assert (upper >= lower).all()  # clamped, no inversion

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


class TestIntervalCaption:
    """Both chart paths share _interval_caption — a typo'd method string used to
    silently drop the subtitle (Phase-3b verification)."""

    def test_widening_caption(self):
        meta = {
            "method": "empirical_widening",
            "anchors": [
                {"horizon": 24, "lower_error": -1.0, "upper_error": 1.0, "sample_size": 24},
                {"horizon": 720, "lower_error": -2.0, "upper_error": 2.0, "sample_size": 720},
            ],
            "calibration_model": "xgboost",
        }
        cap = _interval_caption(meta, "prophet")
        assert "P10–P90 empirical outcome range" in cap
        assert "widens with lead time" in cap
        assert "24h/720h" in cap
        assert "xgboost-calibrated" in cap  # substitute disclosed
        # exact-model calibration → no note
        assert "calibrated" not in _interval_caption(
            {**meta, "calibration_model": "prophet"}, "prophet"
        )

    def test_flat_caption(self):
        meta = {
            "method": "empirical",
            "calibration_window_hours": 840,
            "calibration_model": "xgboost",
        }
        cap = _interval_caption(meta, "xgboost")
        assert "80% empirical prediction interval" in cap
        assert "840h" in cap

    def test_heuristic_caption_empty(self):
        assert _interval_caption({"method": "heuristic"}, "xgboost") == ""
