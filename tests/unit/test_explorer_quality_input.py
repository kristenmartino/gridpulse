"""The explorer needs the same paired quality inputs as the hourly scorer."""

from unittest.mock import patch

import pandas as pd
import pytest

from data.hourly_scoring_actuals import hourly_actuals
from jobs import phases


@pytest.mark.parametrize("has_operator_values", [True, False])
def test_actuals_cache_preserves_optional_paired_operator_values(has_operator_values):
    stamps = pd.date_range("2026-10-20", periods=48, freq="h", tz="UTC")
    frame = pd.DataFrame({"timestamp": stamps, "demand_mw": [1000.0] * 40 + [400.0] * 8})
    if has_operator_values:
        frame["forecast_mw"] = 1000.0
    weather = pd.DataFrame({"timestamp": stamps, "temperature_2m": 75.0})
    before = frame.copy(deep=True)
    captured = {}
    with (
        patch(
            "data.redis_client.persist", side_effect=lambda k, v, **kw: captured.update({k: v})
        ) as write,
        patch.object(phases, "_temperature_percentiles", return_value=None),
    ):
        result = phases.write_actuals_and_weather(phases.RegionData("FPL", frame, weather))
    assert result.ok
    assert write.call_count == 2
    pd.testing.assert_frame_equal(frame, before)
    payload = next(v for k, v in captured.items() if "actuals:" in k)
    assert len(payload["timestamps"]) == len(payload["demand_mw"]) == 48
    if not has_operator_values:
        assert "forecast_mw" not in payload
        return
    assert payload["forecast_mw"] == [1000.0] * 48
    cached_frame = pd.DataFrame(
        {
            "timestamp": payload["timestamps"],
            "demand_mw": payload["demand_mw"],
            "forecast_mw": payload["forecast_mw"],
        }
    )
    now = stamps[-1].to_pydatetime()
    eligible, excluded = hourly_actuals(cached_frame, now=now)
    target = stamps[-1].isoformat()
    assert target not in eligible and target in excluded
    assert "day-ahead" in excluded[target]["reason"]
    # The old cache shape cannot trigger this signal: current=previous=400,
    # above the near-zero floor, with no single-step collapse at this target.
    partial, _ = hourly_actuals(cached_frame.drop(columns="forecast_mw"), now=now)
    assert partial[target] == 400.0
