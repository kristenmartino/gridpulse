"""Forecast time-alignment regression tests (#194, Workstream C — PR 1 foundation).

The scoring job runs hourly but loads daily-trained Prophet/SARIMAX pickles,
whose forecast origin is the (frozen) training end. Before #194 the predict
functions returned a window anchored at train_end while the scoring job wrote
those values at scoring-tick timestamps — a mislabel that grows to ~23h before
each retrain.

These tests pin the model-layer foundation: an explicit ``start_ts`` makes each
predict function return the ``periods``-long window whose first timestamp equals
``start_ts`` (resolving the off-by-one), while ``start_ts=None`` stays
byte-identical to the pre-#194 behavior. The ``phases.py`` integration
(gap-spanning exog + write-by-timestamp) is a separate increment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.arima_model import predict_arima

# ---------------------------------------------------------------------------
# ARIMA — pure, no statsmodels fit needed (we monkeypatch .forecast)
# ---------------------------------------------------------------------------


class _FakeFitted:
    """Returns a ramp so a value's identity reveals which step it came from."""

    def forecast(self, steps, exog=None):
        return np.arange(steps, dtype=float)


class TestPredictArimaAnchor:
    def _payload(self, train_end):
        return {
            "model": _FakeFitted(),  # legacy branch → uses .forecast directly
            "order": (1, 0, 1),
            "seasonal_order": (0, 0, 0, 0),
            "tail_y": np.zeros(10),
            "tail_exog": None,
            "train_end": train_end,
        }

    def _future_exog(self, n):
        # No ARIMA_EXOG_COLS present → _get_exog returns None; shape only matters
        return pd.DataFrame({"timestamp": pd.date_range("2026-06-10", periods=n, freq="h")})

    def test_start_ts_returns_window_starting_at_start_ts(self):
        train_end = pd.Timestamp("2026-06-10 04:00", tz="UTC")
        start_ts = train_end + pd.Timedelta(hours=20)  # 20h gap
        out = predict_arima(
            self._payload(str(train_end)), self._future_exog(200), periods=24, start_ts=start_ts
        )

        assert isinstance(out, dict)
        assert out["timestamps"][0] == start_ts
        assert len(out["forecast"]) == 24
        # gap = offset_hours - 1 = 19 → forecast window is steps 19..42 of the ramp.
        assert out["forecast"][0] == 19.0
        assert out["forecast"][-1] == 42.0

    def test_start_ts_none_is_legacy_array(self):
        out = predict_arima(self._payload("2026-06-10 04:00"), self._future_exog(24), periods=24)
        assert isinstance(out, np.ndarray)
        assert out.shape == (24,)
        assert out[0] == 0.0  # ramp step 0 == train_end+1h (today's behavior)

    def test_zero_gap_matches_legacy_values(self):
        train_end = pd.Timestamp("2026-06-10 04:00", tz="UTC")
        start_ts = train_end + pd.Timedelta(hours=1)  # zero gap
        anchored = predict_arima(
            self._payload(str(train_end)), self._future_exog(24), periods=24, start_ts=start_ts
        )
        legacy = predict_arima(self._payload(str(train_end)), self._future_exog(24), periods=24)
        assert np.array_equal(anchored["forecast"], legacy)

    def test_legacy_pickle_without_train_end_falls_back_to_gap_zero(self):
        payload = self._payload("2026-06-10 04:00")
        del payload["train_end"]
        start_ts = pd.Timestamp("2026-06-11 00:00", tz="UTC")
        out = predict_arima(payload, self._future_exog(24), periods=24, start_ts=start_ts)
        # No anchor → gap 0 → today's (pre-fix) window, but still dict-shaped
        # and labeled from start_ts (best-effort for one retrain cycle).
        assert out["timestamps"][0] == start_ts
        assert out["forecast"][0] == 0.0


# ---------------------------------------------------------------------------
# Prophet — requires the real library; skip cleanly if unavailable
# ---------------------------------------------------------------------------

prophet = pytest.importorskip("prophet", reason="prophet not installed")


def _fit_tiny_prophet():
    from prophet import Prophet

    ds = pd.date_range("2026-05-01", periods=240, freq="h")
    y = 1000 + 100 * np.sin(np.arange(240) * 2 * np.pi / 24)
    m = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=False)
    m.fit(pd.DataFrame({"ds": ds, "y": y}))
    return m, ds[-1]


class TestPredictProphetAnchor:
    def test_start_ts_window_starts_at_start_ts(self):
        from models.prophet_model import predict_prophet

        model, hist_end = _fit_tiny_prophet()
        start_ts = pd.Timestamp(hist_end) + pd.Timedelta(hours=15)
        df = pd.DataFrame({"timestamp": pd.date_range(hist_end, periods=200, freq="h")})
        out = predict_prophet(model, df, periods=24, start_ts=start_ts)

        assert len(out["forecast"]) == 24
        ts0 = pd.Timestamp(out["timestamps"][0])
        assert ts0 == pd.Timestamp(start_ts).tz_localize(None)

    def test_start_ts_none_is_legacy_tail(self):
        from models.prophet_model import predict_prophet

        model, hist_end = _fit_tiny_prophet()
        df = pd.DataFrame({"timestamp": pd.date_range(hist_end, periods=24, freq="h")})
        out = predict_prophet(model, df, periods=24)
        # First forecast row is the hour right after training end.
        assert pd.Timestamp(out["timestamps"][0]) == pd.Timestamp(hist_end) + pd.Timedelta(hours=1)
