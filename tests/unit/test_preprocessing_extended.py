"""Extended unit tests for data/preprocessing.py — targeting uncovered lines.

Covers:
- Line 135: weather column interpolation for NaN values in handle_missing_values
- Line 175: demand > 500,000 MW validation in validate_dataframe
- Lines 186-190: wind speed validation (negative, >200 mph) in validate_dataframe
- Lines 204-205: _ensure_utc tz-naive localization
- Lines 207-208: _ensure_utc non-UTC timezone conversion
"""

import numpy as np
import pandas as pd

from data.preprocessing import (
    _ensure_utc,
    handle_missing_values,
    merge_demand_weather,
    validate_dataframe,
)


class TestWeatherColumnInterpolation:
    """Tests for weather NaN interpolation (line 135)."""

    def test_weather_nans_interpolated_when_demand_present(self):
        """Weather columns with NaN values are interpolated up to max_gap_hours."""
        ts = pd.date_range("2024-06-01", periods=10, freq="h", tz="UTC")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "demand_mw": np.arange(100, 200, 10, dtype=float),
                "temperature_2m": [70.0, np.nan, np.nan, 76.0, 78.0, 80.0, 82.0, 84.0, 86.0, 88.0],
                "wind_speed_10m": [10.0, 12.0, np.nan, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0],
            }
        )
        result = handle_missing_values(df, max_gap_hours=6)
        # Weather NaNs should be filled via linear interpolation
        assert result["temperature_2m"].isna().sum() == 0
        assert result["wind_speed_10m"].isna().sum() == 0

    def test_weather_nans_not_interpolated_beyond_limit(self):
        """Weather NaN runs longer than max_gap_hours are only partially filled."""
        ts = pd.date_range("2024-06-01", periods=12, freq="h", tz="UTC")
        temp_vals = [70.0] + [np.nan] * 10 + [90.0]
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "demand_mw": np.arange(100, 220, 10, dtype=float),
                "temperature_2m": temp_vals,
            }
        )
        result = handle_missing_values(df, max_gap_hours=3)
        # interpolate(limit=3) fills at most 3 consecutive NaN from each side
        # so some NaN remain in the middle
        remaining_nans = result["temperature_2m"].isna().sum()
        assert remaining_nans > 0, "Long weather gaps should not be fully filled"

    def test_weather_only_nans_no_demand_column(self):
        """Weather interpolation still runs when demand column is absent."""
        ts = pd.date_range("2024-06-01", periods=10, freq="h", tz="UTC")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "temperature_2m": [60.0, np.nan, np.nan, 72.0, 75.0, 78.0, 80.0, 82.0, 84.0, 86.0],
            }
        )
        result = handle_missing_values(df, max_gap_hours=6)
        assert result["temperature_2m"].isna().sum() == 0


class TestValidateDemandExceedsMax:
    """Tests for demand > 500,000 MW validation (line 175)."""

    def test_demand_exceeds_500k(self):
        """Demand values above 500,000 MW are reported as issues."""
        ts = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "demand_mw": [40000.0, 500001.0, 600000.0, 35000.0, 42000.0],
            }
        )
        report = validate_dataframe(df, context="high_demand_test")
        assert any("500,000 MW" in issue for issue in report["issues"])
        # Should report exactly 2 rows exceeding threshold
        matching = [i for i in report["issues"] if "500,000" in i]
        assert len(matching) == 1
        assert "2 rows" in matching[0]


class TestValidateWindSpeed:
    """Tests for wind speed validation (lines 186-190)."""

    def test_negative_wind_speed_10m(self):
        """Negative wind_speed_10m values are flagged."""
        ts = pd.date_range("2024-01-01", periods=4, freq="h")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "wind_speed_10m": [10.0, -5.0, 12.0, 8.0],
            }
        )
        report = validate_dataframe(df)
        assert any("Negative wind_speed_10m" in i for i in report["issues"])

    def test_wind_speed_exceeds_200_mph(self):
        """Wind speed > 200 mph is flagged."""
        ts = pd.date_range("2024-01-01", periods=4, freq="h")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "wind_speed_80m": [15.0, 210.0, 18.0, 20.0],
            }
        )
        report = validate_dataframe(df)
        assert any("wind_speed_80m > 200 mph" in i for i in report["issues"])

    def test_wind_speed_120m_both_issues(self):
        """Both negative and >200 mph issues reported for wind_speed_120m."""
        ts = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "wind_speed_120m": [20.0, -3.0, 250.0, 18.0, 22.0],
            }
        )
        report = validate_dataframe(df)
        assert any("Negative wind_speed_120m" in i for i in report["issues"])
        assert any("wind_speed_120m > 200 mph" in i for i in report["issues"])

    def test_valid_wind_no_issues(self):
        """Wind values within normal range produce no issues."""
        ts = pd.date_range("2024-01-01", periods=4, freq="h")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "wind_speed_10m": [5.0, 10.0, 15.0, 20.0],
                "wind_speed_80m": [10.0, 15.0, 20.0, 25.0],
                "wind_speed_120m": [12.0, 18.0, 22.0, 28.0],
            }
        )
        report = validate_dataframe(df)
        wind_issues = [i for i in report["issues"] if "wind" in i.lower()]
        assert len(wind_issues) == 0


class TestEnsureUtc:
    """Tests for _ensure_utc timezone handling (lines 204-205, 207-208)."""

    def test_naive_timestamps_localized_to_utc(self):
        """Timezone-naive timestamps are localized to UTC (lines 204-205)."""
        ts = pd.date_range("2024-07-01", periods=5, freq="h")  # tz-naive
        df = pd.DataFrame({"timestamp": ts, "value": range(5)})
        assert df["timestamp"].dt.tz is None

        result = _ensure_utc(df, source="test_naive")
        assert str(result["timestamp"].dt.tz) == "UTC"
        # Values should be identical (just tz-aware now)
        assert result["timestamp"].iloc[0].hour == 0

    def test_non_utc_timestamps_converted_to_utc(self):
        """Non-UTC timestamps are converted to UTC (lines 207-208)."""
        ts = pd.date_range("2024-07-01", periods=5, freq="h", tz="US/Eastern")
        df = pd.DataFrame({"timestamp": ts, "value": range(5)})
        assert str(df["timestamp"].dt.tz) != "UTC"

        result = _ensure_utc(df, source="test_eastern")
        assert str(result["timestamp"].dt.tz) == "UTC"
        # US/Eastern is UTC-4 in July (EDT), so midnight ET => 04:00 UTC
        assert result["timestamp"].iloc[0].hour == 4

    def test_merge_with_naive_timestamps(self):
        """merge_demand_weather correctly handles tz-naive inputs via _ensure_utc."""
        ts_naive = pd.date_range("2024-03-15", periods=6, freq="h")  # no tz
        demand = pd.DataFrame(
            {
                "timestamp": ts_naive,
                "demand_mw": [30000.0, 31000.0, 32000.0, 33000.0, 34000.0, 35000.0],
            }
        )
        weather = pd.DataFrame(
            {
                "timestamp": ts_naive,
                "temperature_2m": [55.0, 56.0, 58.0, 62.0, 65.0, 63.0],
            }
        )
        result = merge_demand_weather(demand, weather)
        assert not result.empty
        assert str(result["timestamp"].dt.tz) == "UTC"
        assert len(result) == 6

    def test_merge_with_non_utc_timezone(self):
        """merge_demand_weather converts non-UTC timezone inputs to UTC."""
        ts_cst = pd.date_range("2024-01-10", periods=4, freq="h", tz="US/Central")
        demand = pd.DataFrame(
            {
                "timestamp": ts_cst,
                "demand_mw": [40000.0, 41000.0, 42000.0, 43000.0],
            }
        )
        weather = pd.DataFrame(
            {
                "timestamp": ts_cst,
                "temperature_2m": [30.0, 29.0, 28.0, 27.0],
            }
        )
        result = merge_demand_weather(demand, weather)
        assert not result.empty
        assert str(result["timestamp"].dt.tz) == "UTC"
        # US/Central in January is UTC-6, so 00:00 CST => 06:00 UTC
        assert result["timestamp"].iloc[0].hour == 6
