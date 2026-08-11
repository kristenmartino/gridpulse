#!/usr/bin/env python3
"""#478 — evaluate the shadow weighting against the served one, on production forecasts.

#451 answered the WAPE half of the smoothed-weights question with a replay of
persisted vintages and **could not** answer the bias half: the replay's *control*
arm over-forecasts by ~6%, and a harness whose control fails a constraint cannot
certify the treatment against it. This reads the arms the scoring job records
side by side on real production forecasts, where the bias is production's.

Reads ``gridpulse:shadow_weights:{region}``, which carries a bounded window of
per-tick records, each pairing one settled hour's actual with BOTH arms'
predictions — graded by the same `models.drift` primitives that grade the served
ensemble, so neither arm gets a coverage advantage.

## What this refuses to do

* **It reports control bias first.** If the served arm also breaches ±2% across
  the fleet, that is a finding about the fleet, not about smoothing, and this
  experiment is again unable to decide. Printed before any comparison so the
  reading order cannot be reversed after the fact.
* **It refuses to decide early.** #478 asks for >=14 days; fewer prints the
  coverage and stops.
* **It does not re-run the WAPE comparison.** That half is settled
  (docs/WEIGHTS_AB_STUDY.md) and re-running it here would invite quietly
  preferring whichever cut reads better.

Usage:
    python scripts/shadow_weights_eval.py
    python scripts/shadow_weights_eval.py --min-days 14 --json /tmp/shadow.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

os.environ.setdefault("ENVIRONMENT", "production")

import numpy as np  # noqa: E402

MIN_DAYS_DEFAULT = 14
#: #451's pre-registered thresholds, reused verbatim rather than re-chosen.
MAX_ABS_BIAS_PCT = 2.0
MAX_MAPE_REGRESSION_PTS = 0.5


def _load(region: str) -> dict[str, Any] | None:
    from data.redis_client import redis_get, redis_key

    payload = redis_get(redis_key(f"shadow_weights:{region}"))
    return payload if isinstance(payload, dict) else None


def _arm_stats(records: list[dict], arm: str) -> dict[str, float] | None:
    """Bias and MAPE for one arm over graded records."""
    pairs = [
        (float(r["actual"]), float(r[f"{arm}_predicted"]))
        for r in records
        if r.get("actual") and r.get(f"{arm}_predicted") is not None
    ]
    pairs = [(a, p) for a, p in pairs if np.isfinite(a) and np.isfinite(p) and a > 0]
    if not pairs:
        return None
    actual = np.array([a for a, _ in pairs])
    pred = np.array([p for _, p in pairs])
    return {
        "n": len(pairs),
        # Signed: the constraint is about systematic DIRECTION. MAPE's asymmetry
        # biases optimisation toward under-forecasting, which is the expensive
        # direction for a grid, so the sign is the point.
        "bias_pct": float(np.mean((pred - actual) / actual) * 100.0),
        "mape": float(np.mean(np.abs(pred - actual) / actual) * 100.0),
        "wape": float(np.sum(np.abs(pred - actual)) / np.sum(actual) * 100.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-days", type=int, default=MIN_DAYS_DEFAULT)
    ap.add_argument("--json", default=None, help="write the full result here")
    args = ap.parse_args()

    from config import REGION_COORDINATES

    per_region: dict[str, dict] = {}
    all_records: list[dict] = []
    spans: list[float] = []
    for region in sorted(REGION_COORDINATES):
        payload = _load(region)
        if not payload:
            continue
        records = [r for r in (payload.get("records") or []) if r.get("timestamp")]
        if not records:
            continue
        ts = sorted(datetime.fromisoformat(r["timestamp"]) for r in records)
        days = (ts[-1] - ts[0]).total_seconds() / 86400.0
        spans.append(days)
        per_region[region] = {
            "days": round(days, 2),
            "n": len(records),
            "served": _arm_stats(records, "served"),
            "shadow": _arm_stats(records, "shadow"),
            "shadow_arm": payload.get("shadow_arm"),
        }
        all_records.extend(records)

    if not per_region:
        print("No shadow-weight records in Redis yet. The scoring job writes them")
        print("once both arms have a usable MAPE for every ensemble member.")
        return 0

    span = float(np.median(spans)) if spans else 0.0
    print(f"regions with records: {len(per_region)}   median span: {span:.1f} days")
    print(f"total graded hours:   {len(all_records)}\n")

    served = _arm_stats(all_records, "served")
    shadow = _arm_stats(all_records, "shadow")
    if not served or not shadow:
        print("Records exist but one arm has no usable pairs — cannot compare.")
        return 0

    # ── Control first. Not a formality: this is the check #451 failed. ──
    print("=" * 68)
    print("CONTROL ARM FIRST (#478 acceptance) — is this harness able to decide?")
    print("=" * 68)
    print(
        f"  served bias {served['bias_pct']:+.3f}%   MAPE {served['mape']:.3f}%   n={served['n']}"
    )
    breached = [
        r
        for r, v in per_region.items()
        if v["served"] and abs(v["served"]["bias_pct"]) > MAX_ABS_BIAS_PCT
    ]
    print(
        f"  BAs whose SERVED arm breaches ±{MAX_ABS_BIAS_PCT}%: {len(breached)}/{len(per_region)}"
    )
    if abs(served["bias_pct"]) > MAX_ABS_BIAS_PCT:
        print()
        print("  STOP. The served arm itself breaches the bias constraint on production")
        print("  forecasts. That is a finding about the fleet, not about the weighting,")
        print("  and this experiment cannot certify the treatment against a bound its")
        print("  own control fails — the same wall #451 hit, for a different reason.")
        print("  Report the fleet bias; do not report a weighting verdict.")
        if args.json:
            with open(args.json, "w") as f:
                json.dump(
                    {"per_region": per_region, "verdict": "control_fails_constraint"}, f, indent=2
                )
        return 0
    print("  Control is within the bound — the constraint is measurable here.\n")

    if span < args.min_days:
        print(f"Only {span:.1f} days of records; #478 asks for >={args.min_days}. Not deciding.")
        return 0

    # ── Now, and only now, the comparison ──
    from models.rolling_eval import satisficing_check

    print("=" * 68)
    print("TREATMENT vs CONTROL")
    print("=" * 68)
    for label, v in (("served ", served), ("shadow ", shadow)):
        print(f"  {label} bias {v['bias_pct']:+.3f}%  MAPE {v['mape']:.3f}%  WAPE {v['wape']:.3f}%")
    sat = satisficing_check(
        treatment_bias_pct=shadow["bias_pct"],
        control_mape=served["mape"],
        treatment_mape=shadow["mape"],
        max_abs_bias_pct=MAX_ABS_BIAS_PCT,
        max_mape_regression_pts=MAX_MAPE_REGRESSION_PTS,
    )
    print(f"\n  satisficing: {sat}")

    # Per-BA sign consistency — descriptive, not a second decision rule.
    wins = sum(
        1
        for v in per_region.values()
        if v["served"] and v["shadow"] and v["shadow"]["wape"] < v["served"]["wape"]
    )
    print(f"  shadow beats served on WAPE in {wins}/{len(per_region)} BAs")
    print(
        "\n  NOTE: the WAPE half is already settled by docs/WEIGHTS_AB_STUDY.md."
        "\n  This run exists to decide the CONSTRAINTS, not to re-open the effect."
    )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {"per_region": per_region, "served": served, "shadow": shadow, "satisficing": sat},
                f,
                indent=2,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
