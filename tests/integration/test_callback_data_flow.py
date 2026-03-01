"""
Integration tests for callback data flow.

These tests verify the ACTUAL data path through callbacks:
  demo_data → .to_json() → dcc.Store → pd.read_json(StringIO()) → merge → chart

Bug history:
  - Bug #1: pd.read_json(string) fails on pandas 2.x (needs StringIO wrapper)
  - Bug #2: demand/weather timestamps had microsecond mismatch → 0-row merge
"""

import io

import numpy as np
import pandas as pd
import pytest

from config import REGION_COORDINATES
from data.demo_data import (
    generate_demo_demand,
    generate_demo_generation,
    generate_demo_weather,
)

# ── Bug #1: JSON serialization roundtrip ──────────────────────────


class TestJSONRoundtrip:
    """Verify DataFrames survive the dcc.Store serialize/deserialize cycle.

    dcc.Store uses .to_json() on write, pd.read_json(StringIO()) on read.
    Pandas 2.x treats raw strings as file paths — StringIO is mandatory.
    """

    def test_demand_roundtrip(self):
        """Demand DataFrame survives JSON serialize → deserialize."""
        df = generate_demo_demand("FPL", days=2)
        json_str = df.to_json(date_format="iso")

        restored = pd.read_json(io.StringIO(json_str))
        assert len(restored) == len(df)
        assert set(restored.columns) == set(df.columns)

    def test_weather_roundtrip(self):
        """Weather DataFrame survives JSON serialize → deserialize."""
        df = generate_demo_weather("FPL", days=2)
        json_str = df.to_json(date_format="iso")

        restored = pd.read_json(io.StringIO(json_str))
        assert len(restored) == len(df)
        assert "temperature_2m" in restored.columns
        assert "wind_speed_80m" in restored.columns

    def test_generation_roundtrip(self):
        """Generation DataFrame survives JSON serialize → deserialize."""
        df = generate_demo_generation("FPL", days=2)
        json_str = df.to_json(date_format="iso")

        restored = pd.read_json(io.StringIO(json_str))
        assert len(restored) == len(df)

    def test_roundtrip_preserves_values(self):
        """Numeric values are not corrupted by the roundtrip."""
        df = generate_demo_demand("FPL", days=2)
        json_str = df.to_json(date_format="iso")
        restored = pd.read_json(io.StringIO(json_str))

        np.testing.assert_allclose(
            restored["demand_mw"].values,
            df["demand_mw"].values,
            rtol=1e-6,
        )

    def test_roundtrip_timestamps_parse(self):
        """Timestamps deserialize to proper datetime after roundtrip."""
        df = generate_demo_demand("FPL", days=2)
        json_str = df.to_json(date_format="iso")
        restored = pd.read_json(io.StringIO(json_str))

        restored["timestamp"] = pd.to_datetime(restored["timestamp"])
        assert pd.api.types.is_datetime64_any_dtype(restored["timestamp"])

    @pytest.mark.parametrize("region", list(REGION_COORDINATES.keys()))
    def test_all_regions_roundtrip(self, region):
        """Every region's demo data survives the roundtrip."""
        demand = generate_demo_demand(region, days=2)
        weather = generate_demo_weather(region, days=2)

        d_json = demand.to_json(date_format="iso")
        w_json = weather.to_json(date_format="iso")

        d2 = pd.read_json(io.StringIO(d_json))
        w2 = pd.read_json(io.StringIO(w_json))

        assert len(d2) == len(demand)
        assert len(w2) == len(weather)


# ── Bug #2: Timestamp alignment ───────────────────────────────────


class TestTimestampAlignment:
    """Verify demand and weather timestamps are identical.

    Both generators must produce the exact same timestamp series so
    that merge(..., on='timestamp', how='inner') doesn't drop rows.
    """

    def test_demand_weather_timestamps_match(self):
        """Demand and weather timestamps are byte-identical."""
        demand = generate_demo_demand("FPL", days=7)
        weather = generate_demo_weather("FPL", days=7)

        pd.testing.assert_index_equal(
            pd.DatetimeIndex(demand["timestamp"]),
            pd.DatetimeIndex(weather["timestamp"]),
        )

    def test_timestamps_are_whole_hours(self):
        """All timestamps are on the hour (no stray minutes/seconds)."""
        demand = generate_demo_demand("FPL", days=2)
        ts = pd.to_datetime(demand["timestamp"])

        assert (ts.dt.minute == 0).all(), "Timestamps have non-zero minutes"
        assert (ts.dt.second == 0).all(), "Timestamps have non-zero seconds"
        assert (ts.dt.microsecond == 0).all(), "Timestamps have microseconds"

    def test_merge_produces_expected_rows(self):
        """Inner merge of demand + weather keeps all rows (no drops)."""
        days = 7
        demand = generate_demo_demand("FPL", days=days)
        weather = generate_demo_weather("FPL", days=days)

        merged = demand.merge(weather, on="timestamp", how="inner")
        expected = days * 24
        assert len(merged) == expected, (
            f"Merge produced {len(merged)} rows, expected {expected}. "
            f"Timestamp mismatch between demand and weather."
        )

    def test_merge_after_json_roundtrip(self):
        """Merge works after the full serialize → deserialize cycle.

        This is the exact path taken by Tab 2 (Weather Correlation):
          load_data → dcc.Store → update_weather_tab → merge
        """
        days = 7
        demand = generate_demo_demand("FPL", days=days)
        weather = generate_demo_weather("FPL", days=days)

        # Simulate dcc.Store roundtrip
        d_json = demand.to_json(date_format="iso")
        w_json = weather.to_json(date_format="iso")

        d2 = pd.read_json(io.StringIO(d_json))
        w2 = pd.read_json(io.StringIO(w_json))

        d2["timestamp"] = pd.to_datetime(d2["timestamp"])
        w2["timestamp"] = pd.to_datetime(w2["timestamp"])

        merged = d2.merge(w2, on="timestamp", how="inner")
        assert len(merged) == days * 24

    @pytest.mark.parametrize("region", list(REGION_COORDINATES.keys()))
    def test_all_regions_merge(self, region):
        """Every region's demand + weather merge without row drops."""
        demand = generate_demo_demand(region, days=2)
        weather = generate_demo_weather(region, days=2)

        merged = demand.merge(weather, on="timestamp", how="inner")
        assert len(merged) == 48, f"Region {region}: merge got {len(merged)} rows"


# ── Callback contract tests ───────────────────────────────────────


class TestCallbackContracts:
    """Test the data shape contracts that callbacks depend on.

    These don't import Dash or run callbacks — they verify that the
    data arriving at callback inputs has the columns/types callbacks expect.
    """

    def test_demand_has_required_columns(self):
        """Demand data has all columns referenced by callbacks."""
        df = generate_demo_demand("FPL", days=7)
        required = {"timestamp", "demand_mw", "forecast_mw", "region"}
        assert required.issubset(set(df.columns))

    def test_weather_has_required_columns(self):
        """Weather data has all columns referenced by callbacks."""
        df = generate_demo_weather("FPL", days=7)
        required = {
            "timestamp",
            "temperature_2m",
            "wind_speed_80m",
            "shortwave_radiation",
            "relative_humidity_2m",
            "cloud_cover",
            "surface_pressure",
        }
        assert required.issubset(set(df.columns))

    def test_generation_has_required_columns(self):
        """Generation data has all columns for Tab 4."""
        df = generate_demo_generation("FPL", days=7)
        required = {"timestamp", "fuel_type", "generation_mw", "region"}
        assert required.issubset(set(df.columns))

    def test_generation_has_all_fuel_types(self):
        """All 7 fuel types present in generation data."""
        df = generate_demo_generation("FPL", days=7)
        expected_fuels = {"gas", "nuclear", "coal", "wind", "solar", "hydro", "other"}
        assert expected_fuels == set(df["fuel_type"].unique())

    def test_correlation_columns_exist_in_merged(self):
        """Tab 2 heatmap expects these columns after merge."""
        demand = generate_demo_demand("FPL", days=7)
        weather = generate_demo_weather("FPL", days=7)
        merged = demand.merge(weather, on="timestamp", how="inner")

        corr_cols = [
            "demand_mw",
            "temperature_2m",
            "wind_speed_80m",
            "shortwave_radiation",
            "relative_humidity_2m",
            "cloud_cover",
            "surface_pressure",
        ]
        for col in corr_cols:
            assert col in merged.columns, f"Missing correlation column: {col}"

    def test_merged_demand_is_numeric(self):
        """demand_mw is numeric after merge (not object/string)."""
        demand = generate_demo_demand("FPL", days=2)
        weather = generate_demo_weather("FPL", days=2)
        merged = demand.merge(weather, on="timestamp", how="inner")

        assert pd.api.types.is_numeric_dtype(merged["demand_mw"])
        assert pd.api.types.is_numeric_dtype(merged["temperature_2m"])
