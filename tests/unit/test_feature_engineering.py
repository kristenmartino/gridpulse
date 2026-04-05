"""Unit tests for data/feature_engineering.py."""

import numpy as np
import pandas as pd
import pytest

from data.feature_engineering import (
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
