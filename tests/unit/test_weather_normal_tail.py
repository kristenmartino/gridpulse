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

    def test_convex_blend_keeps_derived_in_bounds(self):
        """Phase-3a verification catch (HIGH): the blend is a CONVEX combination, so
        a bounded/convex derived feature can't be driven out of range. An ADDITIVE
        anomaly would go negative here — covered real CDD 0 (cool) with a high
        boundary-day normal CDD but a low tail-day normal."""
        ts = pd.date_range("2026-08-01", periods=48, freq="h", tz="UTC")
        fut = pd.DataFrame(
            {
                "timestamp": ts,
                "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
                "temperature_2m": np.full(48, 60.0),
                "cooling_degree_days": np.full(48, 0.0),  # covered real CDD = 0
            }
        )
        doy_a, doy_b = ts[0].dayofyear, ts[24].dayofyear
        rows = [
            {
                "doy": d,
                "hour": h,
                "temperature_2m": 90.0,
                # high normal on the boundary day, low on the tail day → additive
                # `tail + (0 − 25)·decay` would be ≈ −24
                "cooling_degree_days": 25.0 if d == doy_a else (1.0 if d == doy_b else 10.0),
            }
            for d in range(1, 367)
            for h in range(24)
        ]
        wx = pd.DataFrame({"timestamp": ts[:24]})  # covers the boundary day
        with (
            patch("config.feature_enabled", return_value=True),
            patch(
                "data.weather_normals.load_weather_normal_cached", return_value=pd.DataFrame(rows)
            ),
        ):
            out = _overlay_weather_normal_tail(fut, _featured(), wx, len(fut))
        assert (out["cooling_degree_days"] >= -1e-9).all()  # never physically-impossible

    def test_circular_and_categorical_features_not_blended(self):
        """wind_direction_10m (circular) and weather_code (categorical) can't be
        linearly blended (a blend of 10° and 350° → ~180°, wrong) — they stay at
        the injected normal."""
        ts = pd.date_range("2026-08-01", periods=48, freq="h", tz="UTC")
        fut = pd.DataFrame(
            {
                "timestamp": ts,
                "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
                "temperature_2m": np.full(48, 70.0),
                "wind_direction_10m": np.full(48, 10.0),  # covered real = 10°
                "weather_code": np.full(48, 1.0),
            }
        )
        rows = [
            {
                "doy": d,
                "hour": h,
                "temperature_2m": 90.0,
                "wind_direction_10m": 350.0,
                "weather_code": 3.0,
            }
            for d in range(1, 367)
            for h in range(24)
        ]
        wx = pd.DataFrame({"timestamp": ts[:24]})
        with (
            patch("config.feature_enabled", return_value=True),
            patch(
                "data.weather_normals.load_weather_normal_cached", return_value=pd.DataFrame(rows)
            ),
        ):
            out = _overlay_weather_normal_tail(fut, _featured(), wx, len(fut))
        assert (out["wind_direction_10m"].iloc[24:] == 350.0).all()  # injected normal, unblended
        assert (out["weather_code"].iloc[24:] == 3.0).all()

    def test_diurnal_anomaly_persisted_per_hour(self):
        """Per-hour-of-day remap: a diurnal real profile (warm afternoons, cool
        nights) persists into the near tail — a flat shift can't reproduce this."""
        ts = pd.date_range("2026-08-01", periods=48, freq="h", tz="UTC")
        real_temp = 72.0 + 12.0 * np.sin(2 * np.pi * (ts.hour - 9) / 24)  # diurnal
        fut = pd.DataFrame(
            {
                "timestamp": ts,
                "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
                "temperature_2m": real_temp.to_numpy(),
            }
        )
        wx = pd.DataFrame({"timestamp": ts[:24]})  # covered day carries the diurnal profile
        with (
            patch("config.feature_enabled", return_value=True),
            patch(
                "data.weather_normals.load_weather_normal_cached", return_value=_normal(temp=72.0)
            ),
        ):
            out = _overlay_weather_normal_tail(fut, _featured(), wx, len(fut))
        near = out.iloc[24:48]  # first tail day, all 24 hours (decay ≈ 1)
        aft = near[near["timestamp"].dt.hour.isin([14, 15, 16, 17])]["temperature_2m"].mean()
        night = near[near["timestamp"].dt.hour.isin([2, 3, 4, 5])]["temperature_2m"].mean()
        assert aft > night + 5  # diurnal shape carried through, not a flat 72 normal

    def test_decay_magnitude_matches_tau(self):
        """~1/e of the anomaly survives 120h (τ) past the boundary."""
        fut = _future(horizon=200)
        wx = pd.DataFrame({"timestamp": fut["timestamp"].iloc[:24]})  # boundary at idx 23
        with (
            patch("config.feature_enabled", return_value=True),
            patch("data.weather_normals.load_weather_normal_cached", return_value=_normal()),
        ):
            out = _overlay_weather_normal_tail(fut, _featured(), wx, len(fut))
        # convex: temp = 90 − 20·w; at 120h past the boundary w = 1/e → 90 − 20/e ≈ 82.6
        temp_at_tau = out["temperature_2m"].to_numpy()[23 + 120]
        assert abs(temp_at_tau - (90.0 - 20.0 / np.e)) < 0.5

    def test_last_covered_day_with_gaps_does_not_crash(self):
        """Non-contiguous Open-Meteo coverage in the final 24h: hours-of-day missing
        from the last covered day simply get no persistence (stay at the normal)."""
        ts = pd.date_range("2026-08-01", periods=48, freq="h", tz="UTC")
        fut = _future(horizon=48)
        # cover 0-23 EXCEPT 11-14 (an interior gap in the last covered day)
        keep = [i for i in range(24) if i not in (11, 12, 13, 14)]
        wx = pd.DataFrame({"timestamp": ts[keep]})
        with (
            patch("config.feature_enabled", return_value=True),
            patch("data.weather_normals.load_weather_normal_cached", return_value=_normal()),
        ):
            out = _overlay_weather_normal_tail(fut, _featured(), wx, len(fut))
        assert len(out) == 48  # no crash
        assert (out["temperature_2m"].iloc[keep] == 70.0).all()  # covered rows exact
