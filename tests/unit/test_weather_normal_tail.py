"""#283 Phase 2 — `_overlay_weather_normal_tail` wiring.

The tail-injection swaps the recent-28d climatology weather for the
`(day_of_year, hour)` weather-normal past the Open-Meteo boundary, keeping the
autoregressive demand features on the recent window. Behind the
`weather_normal_tail` flag with a graceful recent-28d fallback.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from jobs.phases import _overlay_weather_normal_tail


def _future(horizon=48, start="2026-08-01"):
    ts = pd.date_range(start, periods=horizon, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
            "temperature_2m": np.full(horizon, 70.0),  # recent-28d fill (uniform)
            "cooling_degree_days": np.full(horizon, 5.0),
            "direct_normal_irradiance": np.full(horizon, 100.0),  # a solar companion
            "relative_humidity_2m": np.full(horizon, 55.0),  # a NORMAL col
            "temp_x_hour": np.zeros(horizon),
            "temperature_deviation": np.zeros(horizon),
            "demand_lag_24h": np.full(horizon, 15000.0),  # autoregressive — MUST NOT change
        }
    )


def _featured(n=200):
    ts = pd.date_range("2026-05-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {"timestamp": ts, "region": ["DUK"] * n, "temperature_2m": np.full(n, 68.0)}
    )


def _normal(temp=90.0, cdd=25.0, dni=800.0, cols=None):
    """A full-year (doy,hour) normal. ``cols`` overrides which feature columns the
    normal carries (default: temp, cdd, dni — a solar family member — so tests can
    assert the whole weather family, not just temperature, is injected)."""
    base = {"temperature_2m": temp, "cooling_degree_days": cdd, "direct_normal_irradiance": dni}
    if cols is not None:
        base = {k: v for k, v in base.items() if k in cols}
    rows = [{"doy": doy, "hour": hour, **base} for doy in range(1, 367) for hour in range(24)]
    return pd.DataFrame(rows)


class TestWeatherNormalTail:
    def test_flag_off_is_noop(self):
        fut = _future()
        with patch("config.feature_enabled", return_value=False):
            out = _overlay_weather_normal_tail(fut, _featured(), None, len(fut))
        pd.testing.assert_frame_equal(out, fut)  # byte-identical to recent-28d path

    def test_no_artifact_is_noop(self):
        fut = _future()
        with (
            patch("config.feature_enabled", return_value=True),
            patch("data.weather_normals.load_weather_normal_cached", return_value=None),
        ):
            out = _overlay_weather_normal_tail(fut, _featured(), None, len(fut))
        pd.testing.assert_frame_equal(out, fut)  # flag on but not backfilled → fallback

    def test_injects_full_weather_family_and_keeps_autoregressive(self):
        fut = _future()
        with (
            patch("config.feature_enabled", return_value=True),
            patch("data.weather_normals.load_weather_normal_cached", return_value=_normal()),
        ):
            # weather_df=None → no Open-Meteo coverage → every hour is "tail".
            out = _overlay_weather_normal_tail(fut, _featured(), None, len(fut))
        # The WHOLE weather family comes from the normal — not just temperature.
        # A split (GHI-family from normal, DNI/diffuse from recent-28d) is the
        # #283 Phase-2-verification bug this pins against.
        assert (out["temperature_2m"] == 90.0).all()
        assert (out["cooling_degree_days"] == 25.0).all()
        assert (out["direct_normal_irradiance"] == 800.0).all()  # solar companion injected
        # ...autoregressive demand feature untouched (anchors current load)...
        assert (out["demand_lag_24h"] == 15000.0).all()
        # ...temperature_deviation recomputed from the injected normal temps (not 0)...
        assert out["temperature_deviation"].abs().sum() > 0
        # ...and temp_x_hour recomputed = normal_temp × hour_sin (#5).
        expected_txh = 90.0 * fut["hour_sin"].to_numpy()
        assert np.allclose(out["temp_x_hour"].to_numpy(), expected_txh)

    def test_absent_normal_col_keeps_recent_value(self):
        """A NORMAL_FEATURE_COL that the artifact happens not to carry must retain
        its recent-28d tail value (get() → NaN → masked out), not be zeroed."""
        fut = _future()
        # Normal carries temp + cdd but NOT relative_humidity_2m.
        with (
            patch("config.feature_enabled", return_value=True),
            patch(
                "data.weather_normals.load_weather_normal_cached",
                return_value=_normal(cols=["temperature_2m", "cooling_degree_days"]),
            ),
        ):
            out = _overlay_weather_normal_tail(fut, _featured(), None, len(fut))
        assert (out["relative_humidity_2m"] == 55.0).all()  # unchanged recent-28d value

    def test_seam_blend_no_jump_covered_untouched_and_decays_to_normal(self):
        """#283 Phase 3: covered hours stay exact; the tail does NOT jump to the
        raw normal at the boundary (the current -20°F anomaly persists, decaying)
        and reverts toward the normal deep in the tail."""
        fut = _future(horizon=400)
        from data.feature_engineering import compute_temp_hour_interaction

        fut["temp_x_hour"] = compute_temp_hour_interaction(
            fut["temperature_2m"], fut["hour_sin"]
        ).to_numpy()
        # Open-Meteo covers the first 100 hours (real temp 70); normal is 90 →
        # boundary anomaly = -20°F.
        wx = pd.DataFrame({"timestamp": fut["timestamp"].iloc[:100]})
        with (
            patch("config.feature_enabled", return_value=True),
            patch("data.weather_normals.load_weather_normal_cached", return_value=_normal()),
        ):
            out = _overlay_weather_normal_tail(fut, _featured(), wx, len(fut))
        temp = out["temperature_2m"].to_numpy()
        # covered hours untouched (real 70) + covered-hour derived byte-identical
        assert (temp[:100] == 70.0).all()
        assert np.allclose(out["temp_x_hour"].to_numpy()[:100], fut["temp_x_hour"].to_numpy()[:100])
        # NO JUMP at the seam: first tail hour continues from ~70, not a leap to 90
        assert abs(temp[100] - temp[99]) < 3.0
        assert temp[100] < 75.0  # blended toward the current regime, not the 90 normal
        # ...decays back toward the normal deep in the tail
        assert temp[-1] > 85.0  # ~90 recovered by hour 400
        assert temp[-1] > temp[100]  # rises across the tail

    def test_seam_blend_skipped_without_coverage(self):
        """weather_df=None → no boundary anomaly to persist → the tail is the pure
        normal (blend is a no-op), exercised by the injection test above too."""
        fut = _future(horizon=48)
        with (
            patch("config.feature_enabled", return_value=True),
            patch("data.weather_normals.load_weather_normal_cached", return_value=_normal()),
        ):
            out = _overlay_weather_normal_tail(fut, _featured(), None, len(fut))
        assert (out["temperature_2m"] == 90.0).all()  # pure normal, no anomaly to blend

    def test_all_covered_is_noop(self):
        fut = _future(horizon=24)
        wx = pd.DataFrame({"timestamp": fut["timestamp"]})  # covers everything
        with (
            patch("config.feature_enabled", return_value=True),
            patch("data.weather_normals.load_weather_normal_cached", return_value=_normal()),
        ):
            out = _overlay_weather_normal_tail(fut, _featured(), wx, len(fut))
        assert (out["temperature_2m"] == 70.0).all()  # nothing to fill → unchanged
