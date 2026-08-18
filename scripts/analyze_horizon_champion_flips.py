#!/usr/bin/env python3
"""Do per-horizon model champions flip across independent weekly windows?

Session B's fleet baseline swept ONE live window per horizon and found the
champion constant across 24/48/72h on 28 of 51 BAs. At 1h, the champion
already flips between the 7d and 30d rolling windows on 18/51 BAs — but
those windows are nested (~79% shared data), so that's a lower bound, not
an estimate. This answers the same question on the 24/48/72h horizon-drift
records dumped by ``scripts/dump_drift_horizon.py``, using four
NON-overlapping 168h windows sliced from each region's own most recent
records — a real estimate, not a lower bound.

Usage:
    python scripts/analyze_horizon_champion_flips.py --dump scratch/drift_horizon_dump_*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODELS = ("xgboost", "prophet", "arima", "ensemble")
HORIZONS = ("24h", "48h", "72h")
WINDOW_HOURS = 168
N_WINDOWS = 4
MIN_RECORDS_PER_WINDOW = 20


def _wape(records: list[dict]) -> float | None:
    num = 0.0
    den = 0.0
    for r in records:
        p = r.get("p")
        a = r.get("a")
        if p is None or a is None:
            continue
        num += abs(p - a)
        den += abs(a)
    if den <= 0:
        return None
    return num / den * 100.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="path to the drift_horizon_dump_*.json file")
    args = ap.parse_args()

    with open(args.dump) as f:
        dump = json.load(f)
    regions: dict[str, dict] = dump["regions"]

    # results[horizon] = {"flips": int, "eligible": int, "windows_champion": list}
    results = {h: {"flips": 0, "eligible": 0, "insufficient": 0} for h in HORIZONS}
    detail_flips: dict[str, list[str]] = {h: [] for h in HORIZONS}

    for region, payload in regions.items():
        models_block = payload.get("models") or {}
        for horizon in HORIZONS:
            # Anchor windows on this region's own latest record timestamp,
            # per horizon (models may have slightly different coverage).
            all_ts: list[datetime] = []
            per_model_records: dict[str, list[dict]] = {}
            for model in MODELS:
                recs = ((models_block.get(model) or {}).get(horizon) or {}).get("records") or []
                per_model_records[model] = recs
                for r in recs:
                    ts = r.get("ts")
                    if ts:
                        all_ts.append(datetime.fromisoformat(ts))
            if not all_ts:
                continue
            latest = max(all_ts)

            window_champions: list[str] = []
            for w in range(N_WINDOWS):
                w_end = latest - timedelta(hours=WINDOW_HOURS * w)
                w_start = w_end - timedelta(hours=WINDOW_HOURS)
                model_wape: dict[str, float] = {}
                for model in MODELS:
                    in_window = [
                        r
                        for r in per_model_records[model]
                        if r.get("ts") and w_start < datetime.fromisoformat(r["ts"]) <= w_end
                    ]
                    if len(in_window) < MIN_RECORDS_PER_WINDOW:
                        continue
                    wape = _wape(in_window)
                    if wape is not None:
                        model_wape[model] = wape
                if len(model_wape) < len(MODELS):
                    continue
                champion = min(model_wape, key=model_wape.get)
                window_champions.append(champion)

            if len(window_champions) < 2:
                results[horizon]["insufficient"] += 1
                continue
            results[horizon]["eligible"] += 1
            if len(set(window_champions)) > 1:
                results[horizon]["flips"] += 1
                detail_flips[horizon].append(f"{region}: {window_champions}")

    print("=" * 72)
    print("PER-HORIZON MODEL CHAMPION FLIP RATE — 4 non-overlapping 168h windows")
    print("=" * 72)
    for horizon in HORIZONS:
        r = results[horizon]
        elig = r["eligible"]
        flips = r["flips"]
        rate = (flips / elig * 100.0) if elig else float("nan")
        print(
            f"  {horizon:4}  flips {flips:2}/{elig:2} BAs ({rate:5.1f}%)   "
            f"insufficient data: {r['insufficient']}"
        )
    print()
    print("Reference: 1h nested-window (7d vs 30d, ~79% overlap) flip rate = 18/51 (35.3%)")
    print("           — a lower bound, not comparable in kind to the non-overlapping rates above.")
    print()
    for horizon in HORIZONS:
        if detail_flips[horizon]:
            print(f"-- {horizon} flips --")
            for line in detail_flips[horizon]:
                print(f"   {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
