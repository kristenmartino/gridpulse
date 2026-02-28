"""Unit tests for data/preprocessing.py."""

import numpy as np
import pandas as pd
import pytest
from datetime import timezone

from data.preprocessing import (
    merge_demand_weather,
    handle_missing_values,
    validate_dataframe,
)


class TestMergeDemandWeather:
    """Test demand + weather merge on UTC timestamps."""

    def test_merge_basic(self, sample_demand_df, sample_weather_df):
        result = merge_demand_weather(sample_demand_df, sample_weather_df)
        assert "demand_mw" in result.columns
        assert "temperature_2m" in result.columns
        assert len(result) > 0

    def test_merge_preserves_demand_rows(self, sample_demand_df, sample_weather_df):
        result = merge_demand_weather(sample_demand_df, sample_weather_df)
        # Left join — at most as many rows as demand
        assert len(result) <= len(sample_demand_df)

    def test_merge_empty_demand(self, sample_weather_df):
        empty = pd.DataFrame(columns=["timestamp", "demand_mw", "region"])
        result = merge_demand_weather(empty, sample_weather_df)
        assert result.empty

    def test_merge_empty_weather(self, sample_demand_df):
        empty = pd.DataFrame(columns=["timestamp", "temperature_2m"])
        result = merge_demand_weather(sample_demand_df, empty)
        assert result.empty

    def test_timestamps_are_utc(self, sample_demand_df, sample_weather_df):
        result = merge_demand_weather(sample_demand_df, sample_weather_df)
        assert str(result["timestamp"].dt.tz) == "UTC"


class TestHandleMissingValues:
    """Test gap interpolation and flagging."""

    def test_small_gap_interpolated(self):
        ts = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        df = pd.DataFrame({
            "timestamp": ts,
            "demand_mw": [100, 110, np.nan, np.nan, 140, 150, 160, 170, 180, 190],
        })
        result = handle_missing_values(df, max_gap_hours=6)
        assert result["demand_mw"].isna().sum() == 0
        assert (result["data_quality"] == "interpolated").sum() == 2

    def test_large_gap_flagged(self):
        ts = pd.date_range("2024-01-01", periods=15, freq="h", tz="UTC")
        values = [100] + [np.nan] * 8 + [200] + [210, 220, 230, 240, 250]
        df = pd.DataFrame({"timestamp": ts, "demand_mw": values})
        result = handle_missing_values(df, max_gap_hours=6)
        assert (result["data_quality"] == "gap").sum() > 0

    def test_no_gaps(self, sample_demand_df):
        result = handle_missing_values(sample_demand_df)
        assert (result["data_quality"] == "original").sum() == len(result)

    def test_empty_df(self):
        result = handle_missing_values(pd.DataFrame())
        assert result.empty


class TestValidateDataframe:
    """Test data validation reporting."""

    def test_valid_data(self, sample_demand_df):
        report = validate_dataframe(sample_demand_df, context="test")
        assert report["rows"] > 0
        assert len(report["issues"]) == 0

    def test_negative_demand(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "demand_mw": [100, -50, 200],
        })
        report = validate_dataframe(df)
        assert any("Negative demand" in i for i in report["issues"])

    def test_extreme_temperature(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "temperature_2m": [75, 200, 80],
        })
        report = validate_dataframe(df)
        assert any("Temperature out of range" in i for i in report["issues"])

    def test_empty_df(self):
        report = validate_dataframe(pd.DataFrame())
        assert any("empty" in i for i in report["issues"])
