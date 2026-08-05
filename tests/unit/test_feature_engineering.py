"""Unit tests for data/feature_engineering.py."""

import numpy as np
import pandas as pd
import pytest

from data.feature_engineering import (
    AUTOREGRESSIVE_DEMAND_FEATURES,
    add_autoregressive_demand_features,
    compute_autoregressive_snapshot,
    compute_cdd,
    compute_cyclical_dow,
    compute_cyclical_hour,
    compute_demand_momentum,
    compute_demand_ratio,
    compute_hdd,
    compute_holiday_flag,
    compute_lag,
    compute_ramp_rate,
    compute_solar_capacity_factor,
    compute_wind_power,
    engineer_features,
    get_feature_names,
)


class TestCDD:
    """Cooling Degree Days: max(0, temp - 65°F)."""

    def test_above_baseline(self):
        result = compute_cdd(pd.Series([85.0]))
        assert result.iloc[0] == 20.0

    def test_below_baseline(self):
        result = compute_cdd(pd.Series([50.0]))
        assert result.iloc[0] == 0.0

    def test_at_baseline(self):
        result = compute_cdd(pd.Series([65.0]))
        assert result.iloc[0] == 0.0

    def test_vectorized(self):
        temps = pd.Series([50, 65, 85, 100])
        result = compute_cdd(temps)
        np.testing.assert_array_equal(result.values, [0, 0, 20, 35])


class TestHDD:
    """Heating Degree Days: max(0, 65°F - temp)."""

    def test_below_baseline(self):
        result = compute_hdd(pd.Series([30.0]))
        assert result.iloc[0] == 35.0

    def test_above_baseline(self):
        result = compute_hdd(pd.Series([85.0]))
        assert result.iloc[0] == 0.0

    def test_cdd_hdd_complementary(self):
        """CDD and HDD should never both be positive for the same temp."""
        temp = pd.Series([30, 50, 65, 80, 100])
        cdd = compute_cdd(temp)
        hdd = compute_hdd(temp)
        assert ((cdd > 0) & (hdd > 0)).sum() == 0


class TestWindPower:
    """Wind power estimate with cut-in/cutout."""

    def test_zero_wind(self):
        result = compute_wind_power(pd.Series([0.0]))
        assert result.iloc[0] == 0.0

    def test_below_cut_in(self):
        """Below ~6.7 mph (3 m/s), turbines don't spin."""
        result = compute_wind_power(pd.Series([5.0]))
        assert result.iloc[0] == 0.0

    def test_above_cutout(self):
        """Above 56 mph (25 m/s), turbines shut down."""
        result = compute_wind_power(pd.Series([60.0]))
        assert result.iloc[0] == 0.0

    def test_normal_wind(self):
        """Normal operating wind should produce power in (0, 1]."""
        result = compute_wind_power(pd.Series([25.0]))
        assert 0.0 < result.iloc[0] <= 1.0

    def test_rated_wind(self):
        """~27 mph ≈ 12 m/s rated speed → CF near 1.0."""
        result = compute_wind_power(pd.Series([27.0]))
        assert result.iloc[0] > 0.5

    # ------------------------------------------------------------------
    # The tests above assert RELATIONSHIPS — "is zero", "in (0, 1]",
    # "> 0.5" — which stay true under most ways of getting the physics
    # wrong. Mutation testing found 20 survivors here: the rated speed, the
    # air-density term, the normalisation divide, the cut-in constant and
    # both threshold comparisons could all be changed with the suite green
    # (docs/TEST_QUALITY.md).
    #
    # The curve is fully determined, so assert the curve.
    # ------------------------------------------------------------------

    #: Speeds are specified in m/s (the units the physics is written in) and
    #: converted, so the fixtures stay readable if MPH_TO_MS ever changes.
    @staticmethod
    def _mph(ms: float) -> float:
        from config import MPH_TO_MS

        return ms / MPH_TO_MS

    def test_output_follows_the_cube_law_normalised_at_rated_speed(self):
        """P ∝ v³, normalised to exactly 1.0 at the 12 m/s rated speed.

        Half the rated speed is one eighth the power — that single ratio pins
        the exponent, the rated constant and the normalisation divide at once.
        A multiply instead of a divide, or a rated speed of 13, moves every
        one of these numbers.
        """
        speeds = pd.Series([self._mph(v) for v in (6.0, 9.0, 12.0)])

        power = compute_wind_power(speeds)

        assert power.iloc[0] == pytest.approx((6.0 / 12.0) ** 3)  # 0.125
        assert power.iloc[1] == pytest.approx((9.0 / 12.0) ** 3)  # 0.421875
        assert power.iloc[2] == pytest.approx(1.0), "rated speed is the normalisation point"

    def test_cut_in_and_cutout_are_both_inclusive(self):
        """A turbine AT cut-in is spinning, and AT cutout has not yet tripped.

        The comparisons are `v < cut_in` → 0 and `v > cutout` → 0, so both
        endpoints produce power. Relaxing either to `<=` / `>=` silently
        zeroes a boundary hour, and the existing tests use 5 mph and 60 mph —
        far enough outside that either spelling passes.
        """
        at_cut_in = compute_wind_power(pd.Series([self._mph(3.0)]))
        assert at_cut_in.iloc[0] == pytest.approx((3.0 / 12.0) ** 3), "3 m/s generates"

        just_below = compute_wind_power(pd.Series([self._mph(2.999)]))
        assert just_below.iloc[0] == 0.0

        at_cutout = compute_wind_power(pd.Series([self._mph(25.0)]))
        assert at_cutout.iloc[0] == 1.0, "25 m/s has not tripped yet"

        just_above = compute_wind_power(pd.Series([self._mph(25.001)]))
        assert just_above.iloc[0] == 0.0

    def test_output_is_capped_at_one_and_keeps_the_input_index(self):
        """Between rated and cutout the cube law would exceed 1.0, so it is
        clipped — and the result must stay aligned with its input, or every
        downstream `df[col] = compute_wind_power(...)` silently misaligns.
        """
        speeds = pd.Series([self._mph(20.0)], index=[7])

        power = compute_wind_power(speeds)

        assert power.iloc[0] == 1.0
        assert list(power.index) == [7], "a reset index would scramble the join"


class TestSolarCapacityFactor:
    """Solar CF = GHI / 1000, clipped [0, 1]."""

    def test_nighttime(self):
        result = compute_solar_capacity_factor(pd.Series([0.0]))
        assert result.iloc[0] == 0.0

    def test_standard_conditions(self):
        result = compute_solar_capacity_factor(pd.Series([1000.0]))
        assert result.iloc[0] == 1.0

    def test_exceeds_rating(self):
        """CF should be clipped at 1.0 even if GHI > 1000."""
        result = compute_solar_capacity_factor(pd.Series([1200.0]))
        assert result.iloc[0] == 1.0

    def test_partial_sun(self):
        result = compute_solar_capacity_factor(pd.Series([500.0]))
        assert result.iloc[0] == pytest.approx(0.5)


class TestCyclicalEncoding:
    """Cyclical sin/cos for hour and day of week."""

    def test_hour_0_and_24_equal(self):
        ts = pd.Series(pd.to_datetime(["2024-01-01 00:00", "2024-01-02 00:00"]))
        sin_vals, cos_vals = compute_cyclical_hour(ts)
        assert sin_vals.iloc[0] == pytest.approx(sin_vals.iloc[1])

    def test_hour_sin_range(self):
        ts = pd.date_range("2024-01-01", periods=24, freq="h")
        sin_vals, cos_vals = compute_cyclical_hour(ts)
        assert sin_vals.min() >= -1.0
        assert sin_vals.max() <= 1.0

    def test_dow_monday_equals_next_monday(self):
        ts = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-08"]))  # Both Monday
        sin_vals, cos_vals = compute_cyclical_dow(ts)
        assert sin_vals.iloc[0] == pytest.approx(sin_vals.iloc[1])

    # ------------------------------------------------------------------
    # 44 survivors across these two functions (docs/TEST_QUALITY.md) — the
    # largest pair in the codebase. The reason is visible above: every
    # assertion compares the output to ITSELF or to a range.
    # `test_hour_0_and_24_equal` compares two midnights, which agree under
    # ANY period, so `/24` could become `/25`. `test_hour_sin_range` holds
    # for every sine of anything.
    #
    # The encoding is exact at the cardinal points, so assert it there.
    # ------------------------------------------------------------------

    def test_hour_encoding_is_exact_at_the_cardinal_hours(self):
        """A quarter turn per six hours: 00→(0,1), 06→(1,0), 12→(0,−1), 18→(−1,0).

        These four points fix the period (24), the 2π factor and the
        sin/cos assignment simultaneously. A period of 25, a 3π factor or a
        swapped divide all move at least one of them.
        """
        ts = pd.Series(pd.to_datetime([f"2026-01-01T{h:02d}:00" for h in (0, 6, 12, 18)]))

        sin_vals, cos_vals = compute_cyclical_hour(ts)

        assert sin_vals.tolist() == pytest.approx([0.0, 1.0, 0.0, -1.0], abs=1e-12)
        assert cos_vals.tolist() == pytest.approx([1.0, 0.0, -1.0, 0.0], abs=1e-12)

    def test_hour_encoding_lies_on_the_unit_circle(self):
        """sin² + cos² == 1 for every hour — the property that makes this an
        *encoding* rather than two unrelated columns. Scaling either component
        breaks it."""
        ts = pd.date_range("2026-01-01", periods=48, freq="h")

        sin_vals, cos_vals = compute_cyclical_hour(ts)

        assert np.allclose(sin_vals**2 + cos_vals**2, 1.0)

    def test_dow_encoding_is_exact_and_has_a_seven_day_period(self):
        """Monday is the origin, (0, 1), and the week closes on itself.

        A period of 8 leaves Monday at (0, 1) — which is why the existing
        Monday-to-Monday test cannot see it — but moves every other day.
        """
        # 2026-01-05 is a Monday.
        ts = pd.Series(pd.to_datetime([f"2026-01-{5 + d:02d}" for d in range(7)]))

        sin_vals, cos_vals = compute_cyclical_dow(ts)

        assert sin_vals.iloc[0] == pytest.approx(0.0, abs=1e-12)
        assert cos_vals.iloc[0] == pytest.approx(1.0, abs=1e-12)
        # Tuesday: one seventh of a turn, not one eighth.
        assert sin_vals.iloc[1] == pytest.approx(np.sin(2 * np.pi / 7))
        assert cos_vals.iloc[1] == pytest.approx(np.cos(2 * np.pi / 7))
        assert np.allclose(sin_vals**2 + cos_vals**2, 1.0)

    def test_both_encoders_preserve_the_index_of_a_series(self):
        """The results are assigned straight onto the frame, so a dropped or
        reset index misaligns every row silently rather than raising."""
        ts = pd.Series(pd.to_datetime(["2026-03-01T05:00", "2026-03-01T06:00"]), index=[11, 12])

        for fn in (compute_cyclical_hour, compute_cyclical_dow):
            sin_vals, cos_vals = fn(ts)
            assert list(sin_vals.index) == [11, 12], fn.__name__
            assert list(cos_vals.index) == [11, 12], fn.__name__
            assert sin_vals.dtype == float and cos_vals.dtype == float, fn.__name__

    def test_both_encoders_accept_a_datetimeindex(self):
        """The *other* branch, which nothing reached.

        Both functions take a ``Series`` or a ``DatetimeIndex`` and read the
        clock field differently in each. Every test here passed a Series, so
        the ``DatetimeIndex`` half was unexercised — its field lookup and its
        index assignment could both be nulled with the suite green.

        The distinction is not academic: a Series carries its own index
        through the arithmetic, so the explicit ``index=`` only matters on
        this path.
        """
        idx = pd.date_range("2026-01-05T00:00", periods=4, freq="h")  # a Monday

        hour_sin, hour_cos = compute_cyclical_hour(idx)
        dow_sin, dow_cos = compute_cyclical_dow(idx)

        for series in (hour_sin, hour_cos, dow_sin, dow_cos):
            assert series.index.equals(idx), "the timestamps themselves are the index"

        # Midnight, and the first hour of the day, on the hour encoder.
        assert hour_sin.iloc[0] == pytest.approx(0.0, abs=1e-12)
        assert hour_cos.iloc[0] == pytest.approx(1.0, abs=1e-12)
        assert hour_sin.iloc[1] == pytest.approx(np.sin(2 * np.pi / 24))
        # All four hours fall on the same Monday.
        assert dow_sin.tolist() == pytest.approx([0.0] * 4, abs=1e-12)
        assert dow_cos.tolist() == pytest.approx([1.0] * 4, abs=1e-12)


class TestHolidayFlag:
    """US federal holiday detection."""

    def test_christmas(self):
        ts = pd.Series(pd.to_datetime(["2024-12-25"]))
        result = compute_holiday_flag(ts)
        assert result.iloc[0] == 1.0

    def test_regular_day(self):
        ts = pd.Series(pd.to_datetime(["2024-03-15"]))
        result = compute_holiday_flag(ts)
        assert result.iloc[0] == 0.0


class TestLagAndRampRate:
    """Lag features and ramp rate."""

    def test_lag_24h(self):
        series = pd.Series(range(48))
        result = compute_lag(series, 24)
        assert pd.isna(result.iloc[0])
        assert result.iloc[24] == 0

    def test_ramp_rate(self):
        demand = pd.Series([100, 120, 110])
        result = compute_ramp_rate(demand)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == 20
        assert result.iloc[2] == -10


class TestDemandMomentum:
    """Demand momentum: recent_lag - older_lag."""

    def test_positive_momentum(self):
        recent = pd.Series([120.0])
        older = pd.Series([100.0])
        result = compute_demand_momentum(recent, older)
        assert result.iloc[0] == 20.0

    def test_negative_momentum(self):
        recent = pd.Series([80.0])
        older = pd.Series([100.0])
        result = compute_demand_momentum(recent, older)
        assert result.iloc[0] == -20.0

    def test_zero_momentum(self):
        recent = pd.Series([100.0])
        older = pd.Series([100.0])
        result = compute_demand_momentum(recent, older)
        assert result.iloc[0] == 0.0


class TestDemandRatio:
    """Demand ratio: lag / rolling_mean (clipped)."""

    def test_above_average(self):
        lag = pd.Series([120.0])
        mean = pd.Series([100.0])
        result = compute_demand_ratio(lag, mean)
        assert result.iloc[0] == pytest.approx(1.2)

    def test_below_average(self):
        lag = pd.Series([80.0])
        mean = pd.Series([100.0])
        result = compute_demand_ratio(lag, mean)
        assert result.iloc[0] == pytest.approx(0.8)

    def test_clips_near_zero_mean(self):
        """Rolling mean near zero should be clipped to 1.0 to avoid inf."""
        lag = pd.Series([100.0])
        mean = pd.Series([0.5])
        result = compute_demand_ratio(lag, mean)
        assert result.iloc[0] == pytest.approx(100.0)

    def test_vectorized(self):
        lag = pd.Series([50, 100, 200])
        mean = pd.Series([100, 100, 100])
        result = compute_demand_ratio(lag, mean)
        np.testing.assert_array_almost_equal(result.values, [0.5, 1.0, 2.0])


class TestEngineerFeatures:
    """Full feature engineering pipeline."""

    def test_adds_expected_columns(self, merged_df):
        result = engineer_features(merged_df)
        assert "cooling_degree_days" in result.columns
        assert "heating_degree_days" in result.columns
        assert "wind_power_estimate" in result.columns
        assert "solar_capacity_factor" in result.columns
        assert "hour_sin" in result.columns
        assert "demand_lag_24h" in result.columns
        assert "demand_roll_24h_mean" in result.columns

    def test_adds_autoresearch_features(self, merged_df):
        result = engineer_features(merged_df)
        assert "demand_lag_1h" in result.columns
        assert "demand_lag_3h" in result.columns
        assert "demand_momentum_short" in result.columns
        assert "demand_momentum_long" in result.columns
        assert "demand_ratio_24h" in result.columns
        assert "demand_ratio_168h" in result.columns

    def test_no_nan_in_output(self, merged_df):
        result = engineer_features(merged_df)
        feature_cols = [
            c for c in result.select_dtypes(include=[np.number]).columns if c not in {"forecast_mw"}
        ]
        # After dropna, there should be no NaN
        assert result[feature_cols].isna().sum().sum() == 0

    def test_output_has_fewer_rows_due_to_lag(self, merged_df):
        """First 168 rows lost to lag/rolling features."""
        result = engineer_features(merged_df)
        assert len(result) < len(merged_df)
        assert len(result) > len(merged_df) - 200  # Reasonable loss

    def test_empty_input(self):
        result = engineer_features(pd.DataFrame())
        assert result.empty

    # ── #161 regression: a sparse exogenous weather column must NOT
    #    collapse the feature-row set below the model threshold ──

    def test_fully_nan_weather_column_does_not_collapse_rows(self, merged_df):
        """The 2026-05-29 P0: Open-Meteo served ``soil_temperature_0cm``
        non-null for only ~100 of ~2100 rows. The old
        ``dropna(subset=<all features>)`` collapsed every region below
        the 168-row threshold → forecasts down nationwide.

        With one weather column entirely NaN, the row count must stay
        healthy (≈ input − warm-up), not collapse — that column is
        imputed, not used to drop rows.
        """
        df = merged_df.copy()
        df["soil_temperature_0cm"] = np.nan  # simulate the outage

        result = engineer_features(df)

        # Should keep ~all rows minus the ~168h autoregressive warm-up,
        # NOT collapse to near-zero. Generous lower bound well above the
        # 168-row model threshold.
        assert len(result) > len(merged_df) - 250, (
            f"sparse weather column collapsed rows to {len(result)} "
            f"(input {len(merged_df)}) — #161 regression"
        )
        assert len(result) >= 168

    def test_partially_sparse_weather_is_imputed(self, merged_df):
        """A weather column missing its older 80% (the real incident
        shape — recent data present, history dropped) must be imputed,
        leaving no residual NaN and a healthy row count."""
        df = merged_df.copy()
        cutoff = int(len(df) * 0.8)
        df.loc[df.index[:cutoff], "temperature_2m"] = np.nan

        result = engineer_features(df)

        feature_cols = [
            c for c in result.select_dtypes(include=[np.number]).columns if c not in {"forecast_mw"}
        ]
        assert result[feature_cols].isna().sum().sum() == 0  # imputed, no residual NaN
        assert len(result) >= 168

    def test_autoregressive_warmup_still_dropped(self, merged_df):
        """The fix must NOT stop dropping the legitimate lag/rolling
        warm-up prefix — demand-derived features remain the drop subset."""
        result = engineer_features(merged_df)
        # demand_lag_168h is NaN for the first 168 rows; none should survive
        assert result["demand_lag_168h"].isna().sum() == 0
        assert len(result) < len(merged_df)  # warm-up was dropped


class TestGetFeatureNames:
    """Feature name listing."""

    def test_returns_list(self):
        names = get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 20

    def test_includes_key_features(self):
        names = get_feature_names()
        assert "cooling_degree_days" in names
        assert "wind_power_estimate" in names
        assert "hour_sin" in names
        assert "demand_lag_24h" in names
        assert "demand_lag_1h" in names
        assert "demand_momentum_short" in names
        assert "demand_ratio_24h" in names
        assert "demand_ratio_168h" in names


# ────────────────────────────────────────────────────────────────────────
# PR-D (2026-05-20) — autoregressive features must not leak target row
# ────────────────────────────────────────────────────────────────────────


def _synthetic_demand_df(n: int = 240, seed: int = 42) -> pd.DataFrame:
    """Build a smooth synthetic demand series for autoregressive tests."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    hours = np.arange(n)
    demand = (
        20_000
        + 5_000 * np.sin(2 * np.pi * hours / 24)
        + 1_000 * np.sin(2 * np.pi * hours / (24 * 7))
        + rng.normal(0, 500, size=n)
    )
    return pd.DataFrame({"timestamp": timestamps, "demand_mw": demand})


class TestAutoregressiveSnapshotValues:
    """Absolute values, not parity.

    The class below pins the snapshot against
    ``add_autoregressive_demand_features`` — a strong invariant, but a
    *relative* one: both paths could drift the same way and it would still
    pass. That is why 21 mutants survived here (docs/TEST_QUALITY.md),
    including window sizes shifting by one hour and the rolling standard
    deviation switching from sample to population.

    These features seed the recursive forecast, so a wrong window is a wrong
    starting point for every hour that follows.

    The history is ``1..200``, strictly increasing, so a 24-hour window and a
    25-hour window can never coincide.
    """

    HISTORY = [float(i) for i in range(1, 201)]

    def test_lags_and_derived_differences(self):
        snap = compute_autoregressive_snapshot(self.HISTORY)

        assert snap["demand_lag_1h"] == 200.0
        assert snap["demand_lag_3h"] == 198.0
        assert snap["demand_lag_24h"] == 177.0
        assert snap["demand_lag_168h"] == 33.0
        assert snap["ramp_rate"] == 1.0  # lag_1 − lag_2
        assert snap["demand_momentum_short"] == 2.0  # lag_1 − lag_3
        assert snap["demand_momentum_long"] == 23.0  # lag_1 − lag_24

    def test_rolling_windows_are_exactly_24_72_and_168_hours(self):
        """Each window's min is its oldest hour — off by one and it moves.

        A 24-hour window over 1..200 spans 177..200; a 25-hour window starts
        at 176. On smooth synthetic demand the two can round to the same
        value, which is how the parity test misses this.
        """
        snap = compute_autoregressive_snapshot(self.HISTORY)

        assert (snap["demand_roll_24h_min"], snap["demand_roll_24h_max"]) == (177.0, 200.0)
        assert (snap["demand_roll_72h_min"], snap["demand_roll_72h_max"]) == (129.0, 200.0)
        assert (snap["demand_roll_168h_min"], snap["demand_roll_168h_max"]) == (33.0, 200.0)
        assert snap["demand_roll_24h_mean"] == 188.5
        assert snap["demand_roll_72h_mean"] == 164.5
        assert snap["demand_roll_168h_mean"] == 116.5

    def test_rolling_window_sizes_also_hold_for_the_maxima(self):
        """A *descending* history, because the ascending one hides this.

        On ``1..200`` the maximum of any trailing window is the newest hour,
        so a 24-hour window and a 25-hour window both report 200 and the size
        is invisible — the minima above pin the size, the maxima do not.
        Reversing the series swaps which end is extreme, and each window's
        maximum becomes its oldest hour.

        Recorded because the first version of this class asserted the maxima
        on the ascending fixture and three window mutations survived it. A
        fixture can satisfy an assertion without exercising it.
        """
        descending = [float(200 - i) for i in range(200)]  # 200 … 1

        snap = compute_autoregressive_snapshot(descending)

        assert snap["demand_roll_24h_max"] == 24.0
        assert snap["demand_roll_72h_max"] == 72.0
        assert snap["demand_roll_168h_max"] == 168.0
        assert snap["demand_roll_24h_min"] == 1.0, "and the minimum is now the newest hour"

    def test_rolling_std_is_the_sample_deviation(self):
        """``ddof=1``. The population form gives 6.922 over the same window,
        and nothing distinguished them — a systematic understatement of
        volatility feeding a model that uses it to size uncertainty."""
        snap = compute_autoregressive_snapshot(self.HISTORY)

        assert snap["demand_roll_24h_std"] == pytest.approx(
            float(np.std(np.arange(177, 201, dtype=float), ddof=1))
        )
        assert snap["demand_roll_24h_std"] == pytest.approx(7.0710678118654755)

    def test_ratios_divide_the_lag_by_its_window_mean(self):
        snap = compute_autoregressive_snapshot(self.HISTORY)

        assert snap["demand_ratio_24h"] == pytest.approx(177.0 / 188.5)
        assert snap["demand_ratio_168h"] == pytest.approx(33.0 / 116.5)

    def test_a_single_observation_has_zero_deviation_not_nan(self):
        """``len(arr) > 1`` guards ``ddof=1``, which is undefined for one
        sample. Relaxed to ``>= 1`` numpy returns NaN with a warning, and a
        NaN feature propagates into the forecast rather than failing."""
        snap = compute_autoregressive_snapshot([42.0])

        assert snap["demand_roll_24h_std"] == 0.0
        assert snap["demand_lag_1h"] == 42.0

    def test_an_empty_history_yields_nan_not_a_crash(self):
        snap = compute_autoregressive_snapshot([])

        assert np.isnan(snap["demand_lag_1h"])
        assert np.isnan(snap["demand_roll_24h_mean"])


class TestAutoregressiveFeaturesNoTargetLeakage:
    """Regression for PR-D: training-time autoregressive features must not
    include the current row's ``demand_mw`` value.

    Before PR-D, ``ramp_rate[i] = demand[i] - demand[i-1]`` and
    ``demand_roll_24h_mean[i]`` included ``demand[i]`` in its window —
    direct target leakage. The XGBoost model trained on this then saw
    feature definitions change at inference time (where
    ``compute_autoregressive_snapshot`` computes them backward-only),
    creating a train/serve distribution shift. These tests guard against
    that regression by perturbing only the LAST row's target and
    asserting every autoregressive feature at that row is unchanged.
    """

    def test_last_row_autoregressive_features_invariant_to_target_perturbation(self):
        """Modify only the last row's demand by 50 GW (massive perturbation).

        Every autoregressive feature at the last row should be IDENTICAL
        between the two dataframes — they depend only on rows 0..n-2.
        """
        base = _synthetic_demand_df(n=240)
        perturbed = base.copy()
        last = len(perturbed) - 1
        # Perturb by ~+250% — if any feature at row `last` changes,
        # leakage is real and the diff will be obvious.
        perturbed.loc[last, "demand_mw"] = float(base.loc[last, "demand_mw"]) + 50_000.0

        base_feats = add_autoregressive_demand_features(base)
        pert_feats = add_autoregressive_demand_features(perturbed)

        for col in AUTOREGRESSIVE_DEMAND_FEATURES:
            base_val = base_feats[col].iloc[last]
            pert_val = pert_feats[col].iloc[last]
            # Use NaN-safe comparison so we don't fail when row 0/1 of
            # ramp_rate are both NaN (they should be NaN in both frames).
            if pd.isna(base_val) and pd.isna(pert_val):
                continue
            assert base_val == pytest.approx(pert_val, rel=1e-9), (
                f"Feature {col!r} at last row differs: base={base_val} "
                f"perturbed={pert_val} — last-row target leaked into feature"
            )

    def test_training_features_match_inference_snapshot_row_by_row(self):
        """For each row i of training features, compare against
        ``compute_autoregressive_snapshot(demand[:i])`` (i.e. inference
        snapshot using only prior demand). They must match exactly.

        This pins the train/inference parity that PR-D establishes —
        any future drift between the two code paths will fail this test.
        """
        df = _synthetic_demand_df(n=240)
        feats = add_autoregressive_demand_features(df)
        demand = df["demand_mw"].tolist()

        # Spot-check rows that should have full windows: 168 (just past
        # the longest lag) and 200 (well past). Random row in between.
        for i in (168, 200, 239):
            snapshot = compute_autoregressive_snapshot(demand[:i])
            for col in AUTOREGRESSIVE_DEMAND_FEATURES:
                training_val = feats[col].iloc[i]
                inference_val = snapshot[col]
                if pd.isna(training_val) and pd.isna(inference_val):
                    continue
                # roll_*_std uses ddof=1 in snapshot vs pandas default (also
                # ddof=1) — should agree. Allow tiny float-roundoff slack.
                assert training_val == pytest.approx(inference_val, rel=1e-6, abs=1e-6), (
                    f"Train/inference mismatch at row {i}, col {col!r}: "
                    f"training={training_val} inference={inference_val}"
                )

    def test_ramp_rate_first_two_rows_are_nan(self):
        """ramp_rate[i] = demand[i-1] - demand[i-2]. The first row (i=0)
        has no prior, the second row (i=1) has only one prior. Both
        should be NaN; row 2 onward should be finite."""
        df = _synthetic_demand_df(n=10)
        feats = add_autoregressive_demand_features(df)
        assert pd.isna(feats["ramp_rate"].iloc[0])
        assert pd.isna(feats["ramp_rate"].iloc[1])
        # Row 2: ramp_rate = demand[1] - demand[0]
        expected = float(df["demand_mw"].iloc[1] - df["demand_mw"].iloc[0])
        assert feats["ramp_rate"].iloc[2] == pytest.approx(expected)

    def test_demand_roll_24h_min_excludes_current_row(self):
        """If the current row's demand is the lowest in the last 25 hours,
        the previous (leaky) behavior would surface that value as the
        min. The shifted behavior surfaces the lowest of the PRIOR 24
        rows instead.

        Construct a series where the last row's demand is unambiguously
        the lowest, and verify ``demand_roll_24h_min`` at the last row
        does NOT equal the last row's demand.
        """
        n = 50
        timestamps = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
        # All rows except last = 20,000; last row = 5,000 (clearly the min)
        demand = np.full(n, 20_000.0)
        demand[-1] = 5_000.0
        df = pd.DataFrame({"timestamp": timestamps, "demand_mw": demand})

        feats = add_autoregressive_demand_features(df)
        last_min = float(feats["demand_roll_24h_min"].iloc[-1])

        # Last-row min must NOT be the current row's demand (5000).
        # Must be the minimum of the PRIOR 24 rows (all 20,000).
        assert last_min == pytest.approx(20_000.0), (
            f"demand_roll_24h_min[last] = {last_min} — leaked current row "
            "(should be 20,000, the min of the prior 24 rows)"
        )

    def test_demand_roll_24h_mean_uses_only_prior_rows(self):
        """Build a series where the last row breaks the constant pattern.
        The mean of the trailing 24-hour window EXCLUDING the current
        row should not be moved by the perturbation."""
        n = 50
        timestamps = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
        demand = np.full(n, 20_000.0)
        df_base = pd.DataFrame({"timestamp": timestamps, "demand_mw": demand})

        df_pert = df_base.copy()
        df_pert.loc[n - 1, "demand_mw"] = 99_999.0  # massive last-row change

        base_feats = add_autoregressive_demand_features(df_base)
        pert_feats = add_autoregressive_demand_features(df_pert)

        # The roll_24h_mean at the LAST row should be identical in both —
        # neither depends on demand[last].
        assert base_feats["demand_roll_24h_mean"].iloc[-1] == pytest.approx(
            pert_feats["demand_roll_24h_mean"].iloc[-1]
        )
        # Sanity: the mean equals the prior-24-row constant (20,000).
        assert base_feats["demand_roll_24h_mean"].iloc[-1] == pytest.approx(20_000.0)
