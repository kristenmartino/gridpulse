"""The BA-polygon rebuild path (``scripts/build_ba_polygons.py``).

The shipped asset was produced ad hoc and described only in prose, so when
its source turned out to be AGPL-3.0 rather than MIT (#357) there was no
repeatable way to swap it. These tests pin the two things that decide whether
a rebuild is trustworthy:

1. **Naming.** Sources disagree — we say ``CAISO``, EIA-930 says ``CISO``,
   electricitymaps said ``US-CAL-CISO``. Matching one spelling is exactly how
   a rebuild silently drops regions and nobody notices until the map has
   holes.
2. **Coverage is a gate, not a statistic.** A source covering 44 of 51 is not
   usable, and the operator has to be told *which* are missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_ba_polygons import build  # noqa: E402

from config import REGION_COORDINATES  # noqa: E402

_SQUARE = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}


def _source(tmp_path, codes, field="NAME"):
    fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {field: c}, "geometry": _SQUARE} for c in codes
        ],
    }
    p = tmp_path / "src.geojson"
    p.write_text(json.dumps(fc))
    return p


class TestNaming:
    def test_matches_our_own_codes(self, tmp_path):
        src = _source(tmp_path, ["CAISO", "ERCOT", "PJM"])
        _fc, rep = build(src, "NAME")
        assert rep["matched"] == 3

    def test_matches_eia_respondent_codes(self, tmp_path):
        """EIA-930 calls it CISO, we call it CAISO. A source keyed by the
        respondent code must still resolve, or the swap loses the BA."""
        src = _source(tmp_path, ["CISO", "ERCO", "ISNE", "NYIS", "SWPP"])
        _fc, rep = build(src, "NAME")
        assert rep["matched"] == 5, f"missing: {rep['missing'][:8]}"

    def test_matches_prefixed_zone_names(self, tmp_path):
        """The electricitymaps convention — US-CAL-CISO. Suffix matching is
        what let the original filter work; keep it."""
        src = _source(tmp_path, ["US-CAL-CISO", "US-TEX-ERCO", "US-MIDA-PJM"])
        _fc, rep = build(src, "NAME")
        assert rep["matched"] == 3, f"missing: {rep['missing'][:8]}"

    def test_properties_are_normalised_not_passed_through(self, tmp_path):
        """The asset's schema is {region, name}. Passing source properties
        through would leak whatever the upstream carried — including fields
        that might themselves be licence-encumbered."""
        fc_in = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"NAME": "ERCOT", "OBJECTID": 7, "SHAPE_Area": 1.2},
                    "geometry": _SQUARE,
                }
            ],
        }
        p = tmp_path / "s.geojson"
        p.write_text(json.dumps(fc_in))
        fc, _rep = build(p, "NAME")
        assert set(fc["features"][0]["properties"]) == {"region", "name"}


class TestCoverageIsAGate:
    def test_missing_regions_are_named_not_counted(self, tmp_path):
        """ "44 of 51" is useless to whoever has to fix it."""
        src = _source(tmp_path, ["ERCOT", "PJM"])
        _fc, rep = build(src, "NAME")
        assert rep["matched"] == 2
        assert len(rep["missing"]) == len(REGION_COORDINATES) - 2
        assert "CAISO" in rep["missing"]

    def test_a_complete_source_reports_no_gaps(self, tmp_path):
        src = _source(tmp_path, sorted(REGION_COORDINATES))
        _fc, rep = build(src, "NAME")
        assert rep["matched"] == len(REGION_COORDINATES)
        assert rep["missing"] == []

    def test_features_without_geometry_are_flagged(self, tmp_path):
        """A feature can match by name and still be useless. The choropleth
        test requires geometry, so catch it at build time rather than in CI."""
        fc_in = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {"NAME": "ERCOT"}, "geometry": None}],
        }
        p = tmp_path / "s.geojson"
        p.write_text(json.dumps(fc_in))
        _fc, rep = build(p, "NAME")
        assert rep["no_geometry"] == ["ERCOT"]

    def test_unmatched_source_codes_are_surfaced(self, tmp_path):
        """A world file is mostly non-US. Showing what did NOT match is how
        an operator spots a wrong --code-field rather than concluding the
        source is bad."""
        src = _source(tmp_path, ["ERCOT", "FR", "DE-LU", "JP-TK"])
        _fc, rep = build(src, "NAME")
        assert rep["matched"] == 1
        assert {"FR", "DE-LU", "JP-TK"} <= set(rep["unmatched_source_codes"])

    def test_duplicate_claims_keep_the_first(self, tmp_path):
        """Two features claiming one BA must not make the output depend on
        dict ordering — the asset has to be reproducible."""
        src = _source(tmp_path, ["ERCOT", "ERCO"])
        fc, rep = build(src, "NAME")
        assert rep["matched"] == 1
        assert len(fc["features"]) == 1


class TestRoundTrip:
    def test_the_shipped_asset_rebuilds_from_itself(self):
        """The strongest available check without a replacement source: the
        script reproduces the current asset, all 51, inside budget."""
        asset = REPO_ROOT / "assets" / "ba_polygons.geojson"
        fc, rep = build(asset, "region")
        assert rep["matched"] == len(REGION_COORDINATES)
        assert rep["missing"] == []
        assert rep["no_geometry"] == []
        assert len(json.dumps(fc, separators=(",", ":")).encode()) < 500 * 1024
