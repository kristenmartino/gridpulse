"""#559: read the seed-shadow record stream and report what it can support.

Mirrors ``scripts/shadow_weights_eval.py`` — same decision order, same
primitives from ``models/shadow_eval.py``, which is arm-agnostic by string key
(``served_predicted`` / ``shadow_predicted``).

**This script is built not to overclaim.** The offline replay
(``docs/POSITIONAL_LAG_SEED_STUDY.md``) was inconclusive at both horizons, and
the shadow accrues evidence only when a gap occurs, which is rare: at the
observed rate a decisive accuracy verdict is 1.2-6.6 years away. So it prints
its own minimum detectable effect and the implied wait alongside every
comparison, and refuses to emit a verdict before both the coverage bar and the
control-arm constraint are met.

What it *can* answer today, and what the shadow was built for:
  * does the temporal path run clean against real production frames
  * what does the second recursion actually cost
  * does live divergence match the 2.1-2.7% of demand the replay predicted
  * is the gate still deciding (the rotating audit must stay at zero)

Redis is a private Memorystore address, so this runs in-VPC — inline it into a
Cloud Run job rather than expecting it to work from a laptop.

Usage:
    ENVIRONMENT=production python scripts/seed_shadow_eval.py [--min-days 14] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("ENVIRONMENT", "production")

import numpy as np  # noqa: E402

import config  # noqa: E402
from models.rolling_eval import satisficing_check  # noqa: E402
from models.shadow_eval import (  # noqa: E402
    MAX_ABS_BIAS_PCT,
    MAX_MAPE_REGRESSION_PTS,
    MIN_DAYS_DEFAULT,
    arm_stats,
    coverage_rows,
    filter_records,
    fleet_stats,
)

ARMS = ("served", "shadow")


def _load(region: str) -> dict | None:
    from data.redis_client import redis_get, redis_key

    payload = redis_get(redis_key(f"seed_shadow:{region}"))
    return payload if isinstance(payload, dict) else None


def _mde_note(deltas: list[float]) -> str:
    """The honest footer: how far this sample is from being able to decide."""
    if len(deltas) < 2:
        return "  power: too few paired observations to estimate."
    arr = np.asarray(deltas, dtype=float)
    mean, stderr = float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr)))
    mde = 2 * stderr
    line = f"  power: n={len(arr)}, mean {mean:+.4f}, stderr {stderr:.4f}, MDE {mde:.4f}"
    if abs(mean) >= mde:
        return line + "  -> the observed effect clears its own MDE."
    if mean == 0:
        return line + "  -> no observed difference."
    need = len(arr) * (mde / abs(mean)) ** 2
    return line + f"\n  power: would need ~{need:.0f} paired observations to resolve this effect."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-days", type=int, default=MIN_DAYS_DEFAULT)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    per_region: dict[str, dict] = {}
    gate_counts = {"diverges": 0, "identical": 0, "audited": 0, "computed": 0}
    audit_alarms: list[str] = []
    divergences: list[float] = []
    all_records: list[dict] = []

    for region in sorted(config.REGION_COORDINATES):
        payload = _load(region)
        if payload is None:
            continue
        gate_counts[payload.get("gate", "identical")] = (
            gate_counts.get(payload.get("gate", "identical"), 0) + 1
        )
        gate_counts["audited"] += bool(payload.get("audited"))
        gate_counts["computed"] += bool(payload.get("computed"))
        div = payload.get("divergence_pct")
        if div is not None:
            divergences.append(float(div))
            # A nonzero divergence on an audited region is a statement about
            # the GATE, not about the seed: it means BAs the gate skipped were
            # real observations that went unrecorded.
            if payload.get("audited") and float(div) != 0.0:
                audit_alarms.append(f"{region} ({div:.6f}%)")
        kept, counts = filter_records(list(payload.get("records") or []))
        if kept:
            per_region[region] = {"records": kept, "counts": counts}
            all_records.extend(kept)

    if not per_region:
        print("no seed-shadow records yet — nothing to report.")
        print("check for `seed_shadow_written` in the scoring-job logs before assuming a bug:")
        print("a fleet with no gaps writes payloads with computed=false and no records.")
        return 0

    print("=" * 72)
    print("SEED SHADOW (#559) — positional (served) vs temporal (shadow)")
    print("=" * 72)

    print("\nGate")
    print(f"  regions reporting     : {len(per_region)}")
    print(f"  gate=diverges         : {gate_counts.get('diverges', 0)}")
    print(f"  gate=identical        : {gate_counts.get('identical', 0)}")
    print(f"  second arm computed   : {gate_counts['computed']}")
    print(f"  audited (should be identical): {gate_counts['audited']}")
    if audit_alarms:
        print(f"  !! AUDIT DIVERGED on {len(audit_alarms)}: {', '.join(audit_alarms)}")
        print("     The gate is skipping BAs whose arms DO differ. Those are lost")
        print("     observations, and the gate must be fixed before trusting any")
        print("     comparison below.")
    else:
        print("  audit divergence      : 0 (gate is deciding correctly)")

    if divergences:
        arr = np.asarray(divergences)
        print("\nDivergence between arms (share of demand)")
        print(f"  mean {arr.mean():.3f}%   max {arr.max():.3f}%   n={len(arr)}")
        print("  offline replay predicted 2.1-2.7% mean — a large gap here means")
        print("  production frames differ from the mirrors the replay used.")

    print("\nCoverage (sparsest first)")
    for region, n, days in coverage_rows(per_region):
        print(f"  {region:6} {n:5} records   {days:5.1f} days")
    span = max((d for _, _, d in coverage_rows(per_region)), default=0.0)

    # Control arm first, exactly as #478's script does: a harness whose control
    # fails a constraint cannot certify a treatment against it.
    served = arm_stats(all_records, "served")
    shadow = arm_stats(all_records, "shadow")
    print("\nPooled")
    for name, st in (("served (positional)", served), ("shadow (temporal)", shadow)):
        if st:
            print(
                f"  {name:22} n={st['n']:5}  bias {st['bias_pct']:+7.3f}%  "
                f"MAPE {st['mape']:6.3f}  WAPE {st['wape']:6.3f}"
            )
    for arm in ARMS:
        fs = fleet_stats(per_region, arm)
        if fs:
            print(
                f"  fleet {arm:16} BAs={fs['n_bas']:3}  bias {fs['bias_pct']:+7.3f}%  "
                f"MAPE {fs['mape']:6.3f}  WAPE {fs['wape']:6.3f}"
            )

    verdict: dict = {"verdict": "insufficient"}
    if served and abs(served["bias_pct"]) > MAX_ABS_BIAS_PCT:
        print(
            f"\nSTOP: the CONTROL arm's bias {served['bias_pct']:+.3f}% exceeds "
            f"±{MAX_ABS_BIAS_PCT}%. A harness whose control fails a constraint "
            "cannot certify the treatment against it."
        )
        verdict = {"verdict": "control_fails_constraint", "served": served}
    elif span < args.min_days:
        print(f"\nCoverage is {span:.1f} days against a {args.min_days}-day bar — not yet.")
        verdict = {"verdict": "insufficient_coverage", "days": span}
    elif served and shadow:
        sat = satisficing_check(
            treatment_bias_pct=shadow["bias_pct"],
            control_mape=served["mape"],
            treatment_mape=shadow["mape"],
            max_abs_bias_pct=MAX_ABS_BIAS_PCT,
            max_mape_regression_pts=MAX_MAPE_REGRESSION_PTS,
        )
        print(f"\nSatisficing: passed={sat['passed']} {sat['failures']}")
        verdict = {"verdict": "scored", "served": served, "shadow": shadow, "satisficing": sat}

    # The footer that keeps this honest, printed whatever the verdict.
    per_ba = [
        (arm_stats(v["records"], "served"), arm_stats(v["records"], "shadow"))
        for v in per_region.values()
    ]
    deltas = [c["wape"] - t["wape"] for c, t in per_ba if c and t]
    print("\n" + _mde_note(deltas))
    print(
        "  NOTE: this shadow cannot settle the accuracy question on any useful\n"
        "  timescale — gaps are rare, so paired observations accrue at roughly\n"
        "  2-7 per week fleet-wide. Its job is pre-rollout safety. The accuracy\n"
        "  question is answered by synthetic gap injection, offline."
    )

    if args.json:
        print(json.dumps({**verdict, "gate": gate_counts, "audit_alarms": audit_alarms}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
