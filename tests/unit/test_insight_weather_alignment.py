"""Historical insights pair UTC hours, never positions in a future weather tail."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from components.insights import _extract_historical_stats


def _demand():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-10-27", periods=24, freq="h", tz="UTC"),
            "demand_mw": np.arange(24, dtype=float) + 1000,
        }
    )


def test_future_tail_cannot_replace_true_hourly_weather_matches():
    demand = _demand()
    weather = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-10-27", periods=48, freq="h", tz="UTC"),
            "temperature_2m": [*range(24), *range(200, 176, -1)],
        }
    ).sample(frac=1, random_state=4)
    before_d, before_w = demand.copy(deep=True), weather.copy(deep=True)
    result = _extract_historical_stats(demand, weather, 24)
    assert result["max_temp"] == 23
    assert result["temp_at_peak_demand"] == 23
    assert result["temp_demand_correlation"] == pytest.approx(1.0)
    assert_frame_equal(demand, before_d)
    assert_frame_equal(weather, before_w)


def test_unmatched_hours_do_not_create_positional_correlations_or_nearest_peak_weather():
    demand = _demand()
    # The first 12 points have genuine matches. Later weather and a missing
    # peak hour cannot be paired merely because the arrays have the same size.
    weather = pd.DataFrame(
        {
            "timestamp": list(demand.timestamp[:12])
            + list(pd.date_range("2026-10-28", periods=12, freq="h", tz="UTC")),
            "temperature_2m": [*range(12), *range(500, 512)],
        }
    )
    result = _extract_historical_stats(demand, weather, 24)
    assert result["temp_demand_correlation"] == pytest.approx(1.0)
    assert result["max_temp"] == 11
    assert result["temp_at_peak_demand"] is None
    weather = pd.DataFrame({"timestamp": demand.timestamp[:-1], "temperature_2m": 75.0})
    result = _extract_historical_stats(demand, weather, 24)
    assert result["temp_demand_correlation"] is None  # Constant series has no Pearson coefficient.
    assert result["temp_at_peak_demand"] is None  # No silent nearest-hour substitution.


def test_no_matching_or_unclocked_weather_stays_unavailable():
    demand = _demand()
    for weather in (
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-11-01", periods=24, freq="h", tz="UTC"),
                "temperature_2m": 90.0,
            }
        ),
        pd.DataFrame({"temperature_2m": [90.0] * 24}),
    ):
        result = _extract_historical_stats(demand, weather, 24)
        assert result["max_temp"] is None
        assert result["temp_demand_correlation"] is None
        assert result["temp_at_peak_demand"] is None
