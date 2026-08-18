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

# The pure statistics live in ``models/shadow_eval.py``. This module sets
# ENVIRONMENT at import time to reach production Redis, and nothing importable
# by a test may carry that side effect with it — doing so leaked the var into
# the whole pytest session and flipped unrelated tests into the strict
# production gate.
from models.shadow_eval import (  # noqa: E402
    MAX_ABS_BIAS_PCT,
    MAX_MAPE_REGRESSION_PTS,
    MIN_DAYS_DEFAULT,
    arm_stats,
    coverage_rows,
    filter_records,
    fleet_stats,
)


def _load(region: str) -> dict[str, Any] | None:
    from data.redis_client import redis_get, redis_key

    payload = redis_get(redis_key(f"shadow_weights:{region}"))
    return payload if isinstance(payload, dict) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-days", type=int, default=MIN_DAYS_DEFAULT)
    ap.add_argument("--json", default=None, help="write the full result here")
    args = ap.parse_args()

    from config import REGION_COORDINATES

    per_region: dict[str, dict] = {}
    all_records: list[dict] = []
    spans: list[float] = []
    dropped_total = {"n_lead_dropped": 0, "n_low_actual_dropped": 0, "n_unknown_lead": 0}
    for region in sorted(REGION_COORDINATES):
        payload = _load(region)
        if not payload:
            continue
        records = [r for r in (payload.get("records") or []) if r.get("timestamp")]
        if not records:
            continue
        # ONE gate, before either arm is scored, on the shared ``actual`` — so
        # both arms keep identical hours and the comparison stays paired.
        # Without this the fleet bias is whatever LDWP's ~50 MW reporting
        # artifacts say it is (see ``filter_records``).
        records, filt = filter_records(records)
        if not records:
            continue
        for k in dropped_total:
            dropped_total[k] += filt[k]
        ts = sorted(datetime.fromisoformat(r["timestamp"]) for r in records)
        days = (ts[-1] - ts[0]).total_seconds() / 86400.0
        spans.append(days)
        per_region[region] = {
            "days": round(days, 2),
            "n": len(records),
            "n_dropped": filt["n_in"] - filt["n_kept"],
            "served": arm_stats(records, "served"),
            "shadow": arm_stats(records, "shadow"),
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

    served_pooled = arm_stats(all_records, "served")
    shadow_pooled = arm_stats(all_records, "shadow")
    served = fleet_stats(per_region, "served")
    shadow = fleet_stats(per_region, "shadow")
    if not served or not shadow or not served_pooled or not shadow_pooled:
        print("Records exist but one arm has no usable pairs — cannot compare.")
        return 0

    # ── Coverage BEFORE any statistic, because it decides how to read one. ──
    print("=" * 68)
    print("COVERAGE — who is actually in this average?")
    print("=" * 68)
    rows = coverage_rows(per_region)
    zero = [r for r, n, _ in rows if n == 0]
    for r, n, d in rows[:8]:
        print(f"    {r:6} {n:5} records   {d:5.1f} days")
    if len(rows) > 8:
        print(f"    … {len(rows) - 8} more, up to {rows[-1][1]} records ({rows[-1][0]})")
    print(
        f"\n  {len(per_region)} BAs, {rows[0][1]}–{rows[-1][1]} records each. "
        f"{len(zero)} contributed nothing: {', '.join(zero) if zero else '—'}"
    )
    print(
        "  Coverage is uneven BY CONSTRUCTION: the #309 quality guard NaNs the\n"
        "  broken-feed BAs' unreliable hours, so they grade fewer records. Both\n"
        "  arms are graded by the same code on the same hours, so the PAIRED\n"
        "  comparison holds — but a pooled average would not be one BA, one vote."
    )
    # Never let the gate be silent: a filter nobody can see is how the
    # unfiltered version of this statistic survived in the first place.
    print(
        f"\n  QUALITY GATE dropped {dropped_total['n_low_actual_dropped']} low-actual "
        f"and {dropped_total['n_lead_dropped']} known-lead>1 records "
        f"({dropped_total['n_unknown_lead']} unknown-lead kept). Same gate the drift\n"
        "  path applies before it averages; the shadow path omitted it, which is\n"
        "  what let LDWP's ~50 MW reporting artifacts set the fleet bias (#142)."
    )
    worst_dropped = sorted(
        ((v.get("n_dropped", 0), r) for r, v in per_region.items()), reverse=True
    )[:5]
    if worst_dropped and worst_dropped[0][0] > 0:
        shown = ", ".join(f"{r} {n}" for n, r in worst_dropped if n > 0)
        print(f"  Most-filtered BAs: {shown}")

    # ── Control first. Not a formality: this is the check #451 failed. ──
    print()
    print("=" * 68)
    print("CONTROL ARM FIRST (#478 acceptance) — is this harness able to decide?")
    print("=" * 68)
    print(
        f"  per-BA (headline)  bias {served['bias_pct']:+.3f}%   "
        f"MAPE {served['mape']:.3f}%   {served['n_bas']} BAs"
    )
    print(
        f"  pooled records     bias {served_pooled['bias_pct']:+.3f}%   "
        f"MAPE {served_pooled['mape']:.3f}%   n={served_pooled['n']}"
    )
    breached = [
        r
        for r, v in per_region.items()
        if v["served"] and abs(v["served"]["bias_pct"]) > MAX_ABS_BIAS_PCT
    ]
    print(
        f"  BAs whose SERVED arm breaches ±{MAX_ABS_BIAS_PCT}%: {len(breached)}/{len(per_region)}"
    )
    if breached:
        print(f"    {', '.join(sorted(breached)[:12])}")
    # Either weighting breaching is enough to stop. An unmeasurable constraint
    # counts as failed (EVALUATION_POLICY.md), and a bound that holds under one
    # weighting and fails under another is not measured — it is chosen.
    if (
        abs(served["bias_pct"]) > MAX_ABS_BIAS_PCT
        or abs(served_pooled["bias_pct"]) > MAX_ABS_BIAS_PCT
    ):
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
    for label, v in (
        ("served  per-BA", served),
        ("shadow  per-BA", shadow),
        ("served  pooled", served_pooled),
        ("shadow  pooled", shadow_pooled),
    ):
        print(
            f"  {label}  bias {v['bias_pct']:+.3f}%  MAPE {v['mape']:.3f}%  WAPE {v['wape']:.3f}%"
        )
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
                {
                    "per_region": per_region,
                    "coverage": coverage_rows(per_region),
                    "served_per_ba": served,
                    "shadow_per_ba": shadow,
                    "served_pooled": served_pooled,
                    "shadow_pooled": shadow_pooled,
                    "satisficing": sat,
                },
                f,
                indent=2,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
