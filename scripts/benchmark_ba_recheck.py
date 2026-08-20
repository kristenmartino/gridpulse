"""Why is this balancing authority excluded from the benchmark, and is it improving?

`/api/v1/benchmark` publishes a one-word reason per excluded BA —
`df-coverage`, `insufficient-paired-hours`, `broken-feed`. That is the right
amount of detail for a public page and not enough to act on: it says which gate
failed, not which of the five drop buckets consumed the hours, nor whether the
gap is closing on its own.

This answers both, from the GCS vintage mirror (no Redis, no cluster access),
by running the BA's records through the same `models.benchmark.pair_hours` the
published payload uses. It reports the drop breakdown, and — given a saved
baseline — the delta and **the elapsed time between the two measurements**.

The elapsed time is not decoration. A delta over an unknown interval cannot be
read: identical numbers mean "stalled" after a day and "you measured twice in
five minutes" after five minutes, and nothing in the numbers distinguishes
them. This script was written after exactly that mistake — a MISO baseline was
compared against a re-measure with no recorded interval, and the flat result was
briefly read as a stall. Baselines therefore carry their own timestamp, and the
verdict below refuses to interpret an interval shorter than `--min-interval`.

The mirror's own age is reported for the same reason: the freshest possible
answer is still bounded by when the scoring job last wrote.

Usage:
    python scripts/benchmark_ba_recheck.py MISO
    python scripts/benchmark_ba_recheck.py MISO --save-baseline /tmp/miso.json
    python scripts/benchmark_ba_recheck.py MISO --baseline /tmp/miso.json

Requires GCS_BUCKET_NAME (or --bucket) and ADC credentials.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from data.vintage import (  # noqa: E402
    FRESH_CAPTURE_LAG_HOURS,
    deserialize_records,
    df_capture_lag_hours,
)
from models.benchmark import MIN_PAIRED_HOURS, pair_hours  # noqa: E402

#: Hours of the trailing edge to inspect. A thin edge is ambiguous on its own —
#: recent hours may simply not have had their DF observed yet — which is the
#: whole reason this script compares across time rather than judging one sample.
TRAILING_HOURS = 48

#: Below this, a baseline comparison is refused rather than guessed at.
DEFAULT_MIN_INTERVAL_HOURS = 6.0


def _load(region: str, bucket_name: str) -> tuple[list, datetime]:
    """Return (records, mirror_last_updated) for one BA."""
    from google.cloud import storage

    blob = storage.Client().bucket(bucket_name).get_blob(f"cache/vintage/{region}/latest.parquet")
    if blob is None:
        raise SystemExit(f"no vintage mirror for {region} in gs://{bucket_name}")
    raw = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
    records = sorted(deserialize_records(raw.to_dict("records")), key=lambda r: r.timestamp)
    return records, blob.updated


def measure(region: str, bucket_name: str) -> dict:
    """Run the BA through the real pairing path and summarise why hours drop."""
    records, mirror_updated = _load(region, bucket_name)
    _, drops = pair_hours(records, {})

    # `no_gridpulse` counts hours that survived every official-arm gate and fell
    # out only on the forecast join, which is absent here — so it IS the
    # would-pair count, and the number the threshold is applied to.
    paired = int(drops.get("no_gridpulse", 0))

    edge = records[-TRAILING_HOURS:]
    lags = [df_capture_lag_hours(r) for r in edge]
    edge_usable = sum(
        1
        for r, lag in zip(edge, lags, strict=True)
        if not r.was_placeholder and lag is not None and lag <= FRESH_CAPTURE_LAG_HOURS
    )

    timestamps = [r.timestamp for r in records]
    return {
        "region": region,
        "measured_at": datetime.now(UTC).isoformat(),
        "mirror_updated": mirror_updated.isoformat(),
        "records": len(records),
        "paired": paired,
        "threshold": MIN_PAIRED_HOURS,
        "short_by": max(0, MIN_PAIRED_HOURS - paired),
        "edge_usable": edge_usable,
        "edge_window": TRAILING_HOURS,
        # `no_gridpulse` is excluded: it is not a defect, it is the
        # would-pair count reported as `paired` above.
        "drops": {k: int(v) for k, v in sorted(drops.items()) if v and k != "no_gridpulse"},
        "window_start": min(timestamps) if timestamps else None,
        "window_end": max(timestamps) if timestamps else None,
    }


def _print(now: dict, base: dict | None, min_interval: float) -> None:
    age_h = (
        datetime.now(UTC) - datetime.fromisoformat(now["mirror_updated"])
    ).total_seconds() / 3600
    print(f"{now['region']}: {now['records']} records   threshold={now['threshold']}")
    print(f"  window        {now['window_start']} -> {now['window_end']}")
    print(f"  mirror written {now['mirror_updated']} ({age_h:.1f}h ago)")
    print()

    if base is None:
        print(f"  {'metric':30s} {'now':>7s}")
        print(f"  {'paired (vs threshold)':30s} {now['paired']:7d}")
        for k, v in now["drops"].items():
            print(f"  {'  drop: ' + k:30s} {v:7d}")
        print(f"  {f'usable in last {TRAILING_HOURS}h':30s} {now['edge_usable']:7d}")
        print()
        if now["paired"] >= now["threshold"]:
            print("  Clears the paired-hours threshold.")
        else:
            print(f"  Short by {now['short_by']} hours. Save a baseline and re-run later:")
            print("     --save-baseline /tmp/ba.json   then   --baseline /tmp/ba.json")
        return

    elapsed = (
        datetime.fromisoformat(now["measured_at"]) - datetime.fromisoformat(base["measured_at"])
    ).total_seconds() / 3600
    print(f"  baseline taken {base['measured_at']}  ({elapsed:.1f}h ago)")
    print()
    print(f"  {'metric':30s} {'baseline':>9s} {'now':>7s} {'delta':>7s}")
    rows = [
        ("paired (vs threshold)", "paired"),
        (f"usable in last {TRAILING_HOURS}h", "edge_usable"),
    ]
    for label, key in rows:
        b, n = base.get(key, 0), now.get(key, 0)
        print(f"  {label:30s} {b:9d} {n:7d} {n - b:+7d}")
    for k in sorted(set(base.get("drops", {})) | set(now["drops"])):
        b, n = base.get("drops", {}).get(k, 0), now["drops"].get(k, 0)
        print(f"  {'  drop: ' + k:30s} {b:9d} {n:7d} {n - b:+7d}")
    print()

    if elapsed < min_interval:
        print(f"  INTERVAL TOO SHORT ({elapsed:.1f}h < {min_interval}h) — no verdict.")
        print("  Identical numbers here mean the window has barely turned over, not")
        print("  that the BA has stalled. Re-run later against the same baseline.")
        return
    if now["paired"] >= now["threshold"]:
        print(f"  CROSSED — {now['paired']} >= {now['threshold']}. Expect it to score now.")
    elif now["edge_usable"] > base.get("edge_usable", 0):
        print("  IMPROVING — the trailing edge is filling in, so the thin edge was")
        print(f"  capture lag. Still {now['short_by']} short; keep waiting.")
    else:
        print(f"  STALLED over {elapsed:.1f}h — the trailing edge did not fill in.")
        print("  This is capture loss rather than lag: waiting on window turnover")
        print("  will not fix it, and the BA deserves its own issue.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("region", help="BA code, e.g. MISO")
    ap.add_argument("--bucket", default=os.getenv("GCS_BUCKET_NAME", ""))
    ap.add_argument("--baseline", type=Path, help="compare against this saved measurement")
    ap.add_argument(
        "--save-baseline", type=Path, help="write this measurement for later comparison"
    )
    ap.add_argument("--min-interval", type=float, default=DEFAULT_MIN_INTERVAL_HOURS)
    args = ap.parse_args()

    if not args.bucket:
        raise SystemExit("set GCS_BUCKET_NAME or pass --bucket")

    now = measure(args.region, args.bucket)
    base = json.loads(args.baseline.read_text()) if args.baseline else None
    if base and base.get("region") != now["region"]:
        raise SystemExit(f"baseline is for {base.get('region')}, not {now['region']}")

    _print(now, base, args.min_interval)

    if args.save_baseline:
        args.save_baseline.write_text(json.dumps(now, indent=2, default=str))
        print(f"\n  baseline written -> {args.save_baseline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
