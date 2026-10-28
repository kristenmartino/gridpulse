"""Mixed weather feeds must never expose future tails as historical context."""

from datetime import UTC, datetime
from unittest.mock import patch

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from data.weather_windows import (
    forecast_weather_window,
    historical_weather_window,
    latest_weather_at_or_before,
)


def _mixed():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-10-20", "2026-11-13", freq="h", tz="UTC"),
            "temperature_2m": range(24 * 24 + 1),
            "wind_speed_80m": 12.0,
        }
    )


def test_history_selects_clock_not_future_tail_and_preserves_input():
    frame = _mixed().sample(frac=1, random_state=12)
    original = frame.copy(deep=True)
    selected = historical_weather_window(frame, "2026-10-28T00:00Z", hours=168)
    assert len(selected) == 168
    assert selected.timestamp.iloc[0] == pd.Timestamp("2026-10-21T01:00Z")
    assert selected.timestamp.iloc[-1] == pd.Timestamp("2026-10-28T00:00Z")
    assert selected.temperature_2m.iloc[-1] == 192
    assert_frame_equal(frame, original)


def test_target_window_uses_actual_endpoints_and_retains_gaps():
    frame = _mixed()
    frame = frame.loc[frame.timestamp != pd.Timestamp("2026-10-28T02:00Z")]
    selected = forecast_weather_window(frame, "2026-10-28T01:00Z", "2026-10-28T03:00Z")
    assert selected.timestamp.tolist() == [
        pd.Timestamp("2026-10-28T01:00Z"),
        pd.Timestamp("2026-10-28T03:00Z"),
    ]


def test_mixed_offsets_naive_clocks_invalid_rows_and_last_sample():
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-10-28T00:00:00", "2026-10-27T21:00:00-04:00", "bad", None],
            "temperature_2m": [70, 71, 99, 99],
        }
    )
    last = latest_weather_at_or_before(frame, "2026-10-28T00:30:00Z")
    assert last.timestamp == pd.Timestamp("2026-10-28T00:00Z")
    assert last.temperature_2m == 70
    # Offset target clocks normalize to UTC and keep the inclusive endpoint.
    selected = forecast_weather_window(frame, "2026-10-27T20:00-04:00", "2026-10-28T01:00")
    assert selected.temperature_2m.tolist() == [70, 71]


@pytest.mark.parametrize("frame", [None, pd.DataFrame(), pd.DataFrame({"temperature_2m": [70]})])
def test_missing_or_clockless_feed_is_unavailable(frame):
    assert historical_weather_window(frame, "2026-10-28").empty
    assert forecast_weather_window(frame, "2026-10-28", "2026-10-29").empty
    assert latest_weather_at_or_before(frame, "2026-10-28") is None


def test_future_only_feed_has_no_current_sample():
    assert latest_weather_at_or_before(_mixed(), "2026-10-19") is None


@pytest.mark.parametrize("hours", [0, -1, float("inf"), float("nan")])
def test_invalid_history_span_is_rejected(hours):
    with pytest.raises(ValueError):
        historical_weather_window(_mixed(), "2026-10-28", hours)


def test_invalid_clocks_or_reversed_target_range_are_rejected():
    with pytest.raises(ValueError):
        historical_weather_window(_mixed(), None)
    with pytest.raises(ValueError):
        forecast_weather_window(_mixed(), "2026-10-29", "2026-10-28")


@pytest.mark.parametrize("future_only", [False, True])
def test_scoring_payload_excludes_future_current_and_temperature(monkeypatch, future_only):
    import config
    from jobs import phases

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 28, 0, 30, tzinfo=UTC)

    frame = _mixed()
    if future_only:
        frame = frame.loc[frame.timestamp > pd.Timestamp("2026-10-28T00:30Z")]
    demand = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-10-20", periods=192, freq="h", tz="UTC"),
            "demand_mw": 1000.0,
        }
    )
    captured = {}
    monkeypatch.setattr(config, "USE_DEMO_DATA", False)
    with (
        patch.object(phases, "datetime", Clock),
        patch.object(phases, "_live_noaa_alerts", return_value=([], 0, 0, 0, 0)),
        patch(
            "data.redis_client.redis_set",
            side_effect=lambda key, value, **kw: captured.update(value),
        ),
    ):
        result = phases.write_alerts(phases.RegionData("FPL", demand, frame))
    assert result.ok
    assert captured["scored_at"] == "2026-10-28T00:30:00+00:00"
    if future_only:
        assert "weather_current" not in captured
        assert "temperature" not in captured
    else:
        assert captured["weather_current"]["timestamp"] == "2026-10-28T00:00:00+00:00"
        assert captured["weather_current"]["temperature_2m"] == 192
        assert captured["weather_current"]["as_of"] == captured["scored_at"]
        assert len(captured["temperature"]["timestamps"]) == 168
        assert max(pd.to_datetime(captured["temperature"]["timestamps"], utc=True)) <= Clock.now()
