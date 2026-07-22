"""Pure-function checks for the multi-point weather study's geometry and
aggregation — the parts that must be correct BEFORE the expensive fetch
(a bad circular mean or a leaky polygon would silently poison the verdict).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.multipoint_weather_study import (
    _counties_in_polygon,
    aggregate_weather,
    select_points,
)


def _square(x0, y0, x1, y1):
    """A GeoJSON MultiPolygon ring (lon/lat) for an axis-aligned box."""
    return [[[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]]


class TestPointInPolygon:
    def test_inside_and_outside(self):
        geom = {"type": "MultiPolygon", "coordinates": _square(-100, 30, -90, 40)}
        lon = np.array([-95.0, -80.0, -99.9])
        lat = np.array([35.0, 35.0, 39.9])
        mask = _counties_in_polygon(geom, lon, lat)
        assert mask.tolist() == [True, False, True]

    def test_hole_is_excluded(self):
        # Outer 0..10 box with an inner 4..6 hole.
        geom = {
            "type": "MultiPolygon",
            "coordinates": [
                [
                    [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],  # exterior
                    [[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]],  # hole
                ]
            ],
        }
        lon = np.array([1.0, 5.0])  # outside hole, inside hole
        lat = np.array([1.0, 5.0])
        assert _counties_in_polygon(geom, lon, lat).tolist() == [True, False]

    def test_degenerate_ring_is_skipped_not_fatal(self):
        geom = {"type": "MultiPolygon", "coordinates": [[[[0, 0]]]]}  # 1-vertex
        mask = _counties_in_polygon(geom, np.array([0.0]), np.array([0.0]))
        assert mask.tolist() == [False]  # no valid ring → nothing inside, no raise


class TestAggregation:
    def _frames(self, per_point: dict[str, list[float]]):
        """Build K single-hour frames; per_point maps var -> [v_p0, v_p1, ...]."""
        from config import WEATHER_VARIABLES

        k = len(next(iter(per_point.values())))
        ts = pd.to_datetime(["2026-07-01T00:00Z"])
        frames = []
        for i in range(k):
            row = {"timestamp": ts}
            for v in WEATHER_VARIABLES:
                row[v] = [per_point.get(v, [0.0] * k)[i]]
            frames.append(pd.DataFrame(row))
        return frames

    def test_wind_direction_is_circular_not_arithmetic(self):
        """350° and 10° average to 0°, never the arithmetic 180°."""
        frames = self._frames({"wind_direction_10m": [350.0, 10.0]})
        agg = aggregate_weather(frames, np.array([1.0, 1.0]))
        d = float(agg["wind_direction_10m"].iloc[0])
        assert min(d, 360 - d) < 1.0, f"expected ~0°, got {d}"

    def test_weather_code_is_weighted_mode_not_mean(self):
        """Ordinal WMO codes: clear(0) at 0.7 vs thunderstorm(95) at 0.3 →
        the code 0, never the meaningless mean ~28."""
        frames = self._frames({"weather_code": [0.0, 95.0]})
        agg = aggregate_weather(frames, np.array([0.7, 0.3]))
        assert float(agg["weather_code"].iloc[0]) == 0.0

    def test_temperature_is_weighted_mean(self):
        frames = self._frames({"temperature_2m": [60.0, 80.0]})
        agg = aggregate_weather(frames, np.array([0.75, 0.25]))
        assert abs(float(agg["temperature_2m"].iloc[0]) - 65.0) < 1e-9


class TestSelectPoints:
    def _geojson(self, ba, coordinates):
        return {
            "features": [
                {
                    "properties": {"region": ba},
                    "geometry": {"type": "MultiPolygon", "coordinates": coordinates},
                }
            ]
        }

    def test_grid_snap_sums_colocated_populations(self):
        """Three counties within one 0.25° cell collapse to one weighted
        point carrying their summed population."""
        # ERCOT centroid is inside this box; put 3 counties ~same cell + 3 far.
        gj = self._geojson("ERCOT", _square(-99, 29, -95, 33))
        counties = pd.DataFrame(
            {
                "GEOID": [f"{i:05d}" for i in range(6)],
                "lat": [31.00, 31.02, 31.03, 30.0, 30.5, 32.5],
                "lon": [-97.00, -97.02, -97.03, -96.0, -96.5, -98.5],
                "pop": [100, 100, 100, 10, 10, 10],
            }
        )
        pts = select_points("ERCOT", gj, counties)
        assert not bool(pts["is_single"].iloc[0])
        # The 3 co-located high-pop counties → one cell with pop 300, the
        # top weight by a wide margin.
        assert pts["weight"].max() > 0.7

    def test_fewer_than_three_counties_falls_back_to_single(self):
        gj = self._geojson("ERCOT", _square(-99, 29, -95, 33))
        counties = pd.DataFrame({"GEOID": ["1"], "lat": [31.0], "lon": [-97.0], "pop": [100]})
        pts = select_points("ERCOT", gj, counties)
        assert bool(pts["is_single"].iloc[0]) and len(pts) == 1
