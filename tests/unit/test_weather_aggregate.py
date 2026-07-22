"""Per-point weather aggregation (ADR-012, data/weather_aggregate.py).

The special cases here are the subtle part of multi-point weather: a
wrong circular mean points the wind backwards, an averaged WMO code
invents weather that isn't happening, and a null-biased mean would drag
values toward zero — a silent coverage collapse of the #161 flavor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import WEATHER_VARIABLES
from data.weather_aggregate import TimestampGridMismatchError, aggregate_weather

TS = pd.to_datetime(["2026-07-01T00:00Z", "2026-07-01T01:00Z"])


def _frames(per_point: dict[str, list[list[float]]], n_hours: int = 2):
    """per_point maps var -> [[h0,h1] for each point] → K frames."""
    k = len(next(iter(per_point.values())))
    out = []
    for i in range(k):
        row: dict[str, object] = {"timestamp": TS[:n_hours]}
        for v in WEATHER_VARIABLES:
            row[v] = per_point.get(v, [[0.0] * n_hours] * k)[i]
        out.append(pd.DataFrame(row))
    return out


class TestSpecialCaseVariables:
    def test_wind_direction_uses_circular_mean(self):
        """350° and 10° average to ~0° (north), never the arithmetic 180°
        (south) — an inverted wind vector would be worse than no data."""
        frames = _frames({"wind_direction_10m": [[350.0, 350.0], [10.0, 10.0]]})
        agg = aggregate_weather(frames)
        d = float(agg["wind_direction_10m"].iloc[0])
        assert min(d, 360 - d) < 1.0, f"expected ~0°, got {d}"

    def test_weather_code_uses_mode_not_mean(self):
        """WMO codes are ordinal: clear(0), clear(0), thunderstorm(95) →
        0, never the mean ~32 ("drizzle" — weather nobody is having)."""
        frames = _frames({"weather_code": [[0.0, 0.0], [0.0, 0.0], [95.0, 95.0]]})
        agg = aggregate_weather(frames)
        assert float(agg["weather_code"].iloc[0]) == 0.0

    def test_ordinary_variable_uses_plain_mean(self):
        frames = _frames({"temperature_2m": [[60.0, 60.0], [80.0, 80.0]]})
        agg = aggregate_weather(frames)
        assert float(agg["temperature_2m"].iloc[0]) == pytest.approx(70.0)


class TestNullPolicy:
    def test_null_point_is_dropped_not_counted_as_zero(self):
        """THE null-safety lock. The study script used nansum(w*arr), where
        a null point contributes 0: [10, 20, nan] → 10.0. Production must
        renormalize: → 15.0. A cell over water or outside CONUS must not
        drag the aggregate toward zero (#161-flavored silent collapse)."""
        frames = _frames({"temperature_2m": [[10.0, 10.0], [20.0, 20.0], [np.nan, np.nan]]})
        agg = aggregate_weather(frames)
        assert float(agg["temperature_2m"].iloc[0]) == pytest.approx(15.0)

    def test_all_null_hour_is_nan_not_zero(self):
        """All points null → NaN, which engineer_features imputes exactly
        as it does a single-point null today. Zero would be a fake reading."""
        frames = _frames({"temperature_2m": [[np.nan, 5.0], [np.nan, 15.0]]})
        agg = aggregate_weather(frames)
        assert np.isnan(float(agg["temperature_2m"].iloc[0]))
        assert float(agg["temperature_2m"].iloc[1]) == pytest.approx(10.0)

    def test_circular_and_mode_are_null_safe(self):
        frames = _frames(
            {
                "wind_direction_10m": [[np.nan, np.nan], [90.0, 90.0]],
                "weather_code": [[np.nan, np.nan], [61.0, 61.0]],
            }
        )
        agg = aggregate_weather(frames)
        assert float(agg["wind_direction_10m"].iloc[0]) == pytest.approx(90.0, abs=1e-6)
        assert float(agg["weather_code"].iloc[0]) == 61.0


class TestGuards:
    def test_mismatched_timestamp_grids_raise(self):
        """Misaligned frames must raise so the caller fails open to
        single-point rather than averaging different hours together."""
        a, b = _frames({"temperature_2m": [[1.0, 2.0], [3.0, 4.0]]})
        b = b.copy()
        b["timestamp"] = pd.to_datetime(["2026-07-02T00:00Z", "2026-07-02T01:00Z"])
        with pytest.raises(TimestampGridMismatchError):
            aggregate_weather([a, b])

    def test_single_frame_passes_through(self):
        frames = _frames({"temperature_2m": [[42.0, 43.0]]})
        agg = aggregate_weather(frames)
        assert agg["temperature_2m"].tolist() == [42.0, 43.0]

    def test_schema_is_preserved(self):
        """The aggregate must carry the same 17 raw columns
        engineer_features reads — a dropped column is a silent feature loss."""
        frames = _frames({"temperature_2m": [[1.0, 2.0], [3.0, 4.0]]})
        agg = aggregate_weather(frames)
        for var in WEATHER_VARIABLES:
            assert var in agg.columns, f"aggregation dropped {var}"

    def test_aggregation_actually_runs(self):
        """Anti-passthrough: perturbing ONE point by Δ moves the aggregate
        by exactly Δ/K — proves this isn't returning frames[0]."""
        base = aggregate_weather(_frames({"temperature_2m": [[10.0, 10.0], [10.0, 10.0]]}))
        moved = aggregate_weather(_frames({"temperature_2m": [[10.0, 10.0], [30.0, 30.0]]}))
        delta = float(moved["temperature_2m"].iloc[0]) - float(base["temperature_2m"].iloc[0])
        assert delta == pytest.approx(10.0)  # Δ=20 over K=2
