"""Cooling-response features — the pack the error analysis pointed at.

`docs/ERROR_ANALYSIS.md` measured the hottest temperature quintile carrying a
mean **34.7%** of our forecast error against **11.9%** for the coldest,
monotone in 7 of 8 BAs. The existing representation of cooling load is one
linear ``cooling_degree_days`` against a fixed 65°F baseline.

These tests pin the three things it cannot express — accumulation, convexity,
humidity — plus the two invariants that make the pack safe to ship: it is
built only from weather (so a day-ahead forecast can have it) and it is
strictly backward-looking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.feature_engineering import (
    COOLING_ACCUM_WINDOWS_H,
    compute_cdd_accumulation,
    compute_heat_index,
    engineer_features,
)

COOLING_FEATURES = ("cdd_accum_24h", "cdd_accum_72h", "cdd_squared", "heat_index", "cdd_x_humidity")


def _frame(n: int = 900, temp: float = 85.0, rh: float = 60.0) -> pd.DataFrame:
    ts = pd.date_range("2026-06-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": rng.random(n) * 1000 + 5000,
            "forecast_mw": rng.random(n) * 1000 + 5000,
            "temperature_2m": np.full(n, temp) + rng.random(n),
            "relative_humidity_2m": np.full(n, rh),
            "wind_speed_80m": rng.random(n) * 20,
            "shortwave_radiation": rng.random(n) * 500,
            "cloud_cover": rng.random(n) * 100,
        }
    )


class TestHeatIndex:
    @pytest.mark.parametrize(
        "temp,rh,expected",
        [(95.0, 70.0, 122.6), (95.0, 20.0, 91.5), (105.0, 60.0, 148.9)],
    )
    def test_matches_the_nws_regression(self, temp, rh, expected):
        """Spot values from the NWS heat-index chart.

        95°F at 70% RH is 'danger' territory while the same dry-bulb at 20% is
        below the reading itself — which is precisely the load difference the
        model currently has to infer from two separate columns.
        """
        got = compute_heat_index(pd.Series([temp]), pd.Series([rh])).iloc[0]
        assert got == pytest.approx(expected, abs=0.5)

    def test_below_80f_uses_the_simple_form_not_the_hot_polynomial(self):
        """The Rothfusz regression is fitted for hot, humid air; extrapolating
        it down to cool hours produces nonsense, so the NWS simple form applies
        there. At 70°F the index should stay near the temperature."""
        got = compute_heat_index(pd.Series([70.0]), pd.Series([50.0])).iloc[0]
        assert 65 < got < 75

    def test_humidity_raises_it_monotonically_when_hot(self):
        idx = compute_heat_index(pd.Series([95.0] * 4), pd.Series([20.0, 40.0, 60.0, 80.0]))
        assert list(idx) == sorted(idx)

    def test_out_of_range_humidity_is_clipped_not_propagated(self):
        """A bad upstream RH must not produce a NaN column that drops rows from
        the whole training frame."""
        got = compute_heat_index(pd.Series([95.0, 95.0]), pd.Series([-5.0, 150.0]))
        assert np.all(np.isfinite(got))


class TestAccumulation:
    def test_is_a_trailing_mean_never_a_centred_one(self):
        """Leakage check. The value at t must depend only on t and earlier —
        a centred window would import future weather into a feature the model
        treats as known."""
        cdd = pd.Series([0.0, 0.0, 0.0, 100.0, 0.0])
        acc = compute_cdd_accumulation(cdd, 3)
        assert acc.iloc[2] == 0.0, "the spike at index 3 must not reach backwards"
        assert acc.iloc[3] > 0

    def test_early_rows_survive_rather_than_becoming_nan(self):
        """min_periods=1 — a NaN here would drop the frame's first rows for
        every model, not just this feature."""
        acc = compute_cdd_accumulation(pd.Series([5.0, 10.0, 15.0]), 72)
        assert np.all(np.isfinite(acc))

    def test_a_heat_wave_accumulates_while_a_single_hot_hour_does_not(self):
        """The mechanism, stated as a test: the third consecutive hot day and
        an isolated hot hour look identical to point-in-time CDD."""
        sustained = compute_cdd_accumulation(pd.Series([20.0] * 24), 24).iloc[-1]
        isolated = compute_cdd_accumulation(pd.Series([0.0] * 23 + [20.0]), 24).iloc[-1]
        assert sustained > isolated * 5

    def test_windows_cover_both_timescales(self):
        assert 24 in COOLING_ACCUM_WINDOWS_H, "overnight carry-over"
        assert 72 in COOLING_ACCUM_WINDOWS_H, "multi-day heat wave"


class TestFlagGating:
    def test_features_are_absent_when_the_flag_is_off(self, monkeypatch):
        """Default OFF. The error analysis justified the experiment, not the
        shipping — the study decides."""
        import config

        monkeypatch.setitem(config.FEATURE_FLAGS, "cooling_response_features", False)
        out = engineer_features(_frame())
        for col in COOLING_FEATURES:
            assert col not in out.columns

    def test_features_appear_when_the_flag_is_on(self, monkeypatch):
        import config

        monkeypatch.setitem(config.FEATURE_FLAGS, "cooling_response_features", True)
        out = engineer_features(_frame())
        for col in COOLING_FEATURES:
            assert col in out.columns
            assert np.all(np.isfinite(out[col])), f"{col} must not introduce NaN"

    def test_the_pack_is_weather_only_so_a_day_ahead_forecast_can_have_it(self, monkeypatch):
        """The invariant that makes this shippable.

        Every feature here derives from temperature and humidity — both
        available from the weather forecast at issue time. If any of them
        touched demand it would be unusable at a 24h horizon, which is the
        trap `scripts/error_analysis.py::make_day_ahead_safe` exists to
        handle for the autoregressive features.
        """
        import config

        monkeypatch.setitem(config.FEATURE_FLAGS, "cooling_response_features", True)
        base = _frame()
        out_a = engineer_features(base)
        # Same weather, completely different demand — the pack must not move.
        shifted = base.copy()
        shifted["demand_mw"] = shifted["demand_mw"] * 3.0 + 10_000
        out_b = engineer_features(shifted)
        for col in COOLING_FEATURES:
            assert np.allclose(out_a[col].to_numpy(), out_b[col].to_numpy()), (
                f"{col} responds to demand — it cannot be known at issue time"
            )
