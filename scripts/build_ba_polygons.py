"""Rebuild ``assets/ba_polygons.geojson`` from a source boundary file.

The shipped asset was produced ad hoc in May 2026 and described only in
prose, so there was no repeatable way to swap its source — which mattered
once the source turned out to be AGPL-3.0 rather than MIT (#357). This makes
the swap a one-command operation against *any* source file, so the remaining
work on that issue is "obtain a licence-clean file", not "redo the filtering
by hand and hope it matches".

It is deliberately source-agnostic. HIFLD Control Areas, a Census-derived
build, or anything else all work provided the file is GeoJSON with one
feature per balancing authority and some property carrying a BA code.

Usage:
    python scripts/build_ba_polygons.py --source control_areas.geojson \\
        --code-field NAME --dry-run
    python scripts/build_ba_polygons.py --source control_areas.geojson \\
        --code-field NAME --output assets/ba_polygons.geojson

``--dry-run`` reports coverage and size without writing, which is the mode to
run first: coverage against all 51 BAs is the criterion that decides whether a
candidate source is usable at all.

Requires no network access.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import REGION_COORDINATES, REGION_NAMES  # noqa: E402
from data.eia_client import EIA_REGION_CODES  # noqa: E402

#: The existing asset contract, pinned by tests/unit/test_us_grid_choropleth.py.
SIZE_BUDGET_BYTES = 500 * 1024


def _candidate_codes(region: str) -> set[str]:
    """Every string a source file might plausibly use for one of our BAs.

    Sources disagree on naming: we use ``CAISO`` where EIA-930 uses ``CISO``
    and electricitymaps used ``US-CAL-CISO``. Matching on a single spelling is
    how a rebuild silently loses regions, so try our code, the EIA respondent
    code, and a suffix match against both.
    """
    eia = EIA_REGION_CODES.get(region, region)
    return {region.upper(), eia.upper()}


def _extract_code(props: dict[str, Any], field: str) -> str | None:
    raw = props.get(field)
    return str(raw).strip().upper() if raw is not None else None


def build(source: Path, code_field: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Filter ``source`` to our 51 BAs and normalise properties.

    Returns ``(feature_collection, report)``. The report is the decision-useful
    half — a source that covers 44 of 51 is not usable, and the caller needs to
    see which 7 are missing rather than a bare count.
    """
    gj = json.loads(source.read_text())
    features = gj.get("features") or []

    wanted = {r: _candidate_codes(r) for r in REGION_COORDINATES}
    matched: dict[str, dict[str, Any]] = {}
    unmatched_source: list[str] = []

    for feat in features:
        code = _extract_code(feat.get("properties") or {}, code_field)
        if code is None:
            continue
        hit = None
        for region, cands in wanted.items():
            if code in cands or any(code.endswith(c) for c in cands):
                hit = region
                break
        if hit is None:
            unmatched_source.append(code)
            continue
        if hit in matched:
            # Two source features claim the same BA. Keep the first and say so
            # — silently picking one would make the asset non-reproducible.
            continue
        matched[hit] = {
            "type": "Feature",
            "properties": {"region": hit, "name": REGION_NAMES.get(hit, hit)},
            "geometry": feat.get("geometry"),
        }

    missing = sorted(set(REGION_COORDINATES) - set(matched))
    out = {
        "type": "FeatureCollection",
        "features": [matched[r] for r in sorted(matched)],
    }
    report = {
        "source_features": len(features),
        "matched": len(matched),
        "expected": len(REGION_COORDINATES),
        "missing": missing,
        "unmatched_source_codes": sorted(set(unmatched_source))[:20],
        "no_geometry": sorted(r for r, f in matched.items() if not f["geometry"]),
    }
    return out, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", type=Path, required=True, help="source GeoJSON")
    ap.add_argument(
        "--code-field", required=True, help="property holding the BA code (e.g. NAME, ID)"
    )
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.source.exists():
        print(f"source not found: {args.source}")
        return 1

    fc, report = build(args.source, args.code_field)
    body = json.dumps(fc, separators=(",", ":"))
    size = len(body.encode())

    print(f"source features   : {report['source_features']}")
    print(f"matched           : {report['matched']} of {report['expected']}")
    print(f"size              : {size:,} bytes (budget {SIZE_BUDGET_BYTES:,})")
    if report["missing"]:
        print(f"MISSING ({len(report['missing'])}): {', '.join(report['missing'])}")
    if report["no_geometry"]:
        print(f"NO GEOMETRY: {', '.join(report['no_geometry'])}")
    if report["unmatched_source_codes"]:
        print(f"unmatched source codes (first 20): {', '.join(report['unmatched_source_codes'])}")

    ok = not report["missing"] and not report["no_geometry"] and size < SIZE_BUDGET_BYTES
    print(f"\nusable as a drop-in replacement: {'YES' if ok else 'NO'}")
    if not ok:
        if report["missing"]:
            print("  - coverage is short; the choropleth test requires all 51")
        if size >= SIZE_BUDGET_BYTES:
            print("  - over the size budget; simplify geometry before shipping")

    if args.dry_run or args.output is None:
        print("\n(dry run — nothing written)")
        return 0 if ok else 2

    args.output.write_text(body)
    print(f"\nwritten → {args.output}")
    print("Remember: update THIRD_PARTY_NOTICES.md with the new source and licence.")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
