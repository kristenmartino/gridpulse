"""#559: where do the losing windows sit?

Pre-registered in ``docs/POSITIONAL_LAG_LOSING_QUARTER_PREREGISTRATION.md``,
committed before this file existed.

**EXPLORATORY.** This re-cuts a dataset that already produced a verdict, so any
pattern is a hypothesis for a future confirmatory study on fresh data — never
grounds for shipping a rule. Bins, flag criteria and the multiplicity disclosure
are all fixed by the pre-registration and are not tunable here.

Usage:
    PYTHONPATH=. python scripts/positional_lag_losing_quarter.py --dir /path/to/artifacts
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

# Fixed by the pre-registration.
GAP_LEN_BINS = [("1", (1, 1)), ("2-3", (2, 3)), ("13-24", (13, 24))]
LEAD_BINS = [("1-24", (1, 24)), ("25-72", (25, 72)), ("73-168", (73, 168))]
HOUR_BINS = [("0-5", (0, 5)), ("6-11", (6, 11)), ("12-17", (12, 17)), ("18-23", (18, 23))]
MIN_N = 30
BOOTSTRAP = 10_000


def _boot_ci(x: np.ndarray, seed: int = 559) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(BOOTSTRAP, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _cells(df: pd.DataFrame, col: str, bins) -> list[dict]:
    out = []
    for label, (lo, hi) in bins:
        sub = df[(df[col] >= lo) & (df[col] <= hi)]
        if sub.empty:
            out.append({"bin": label, "n": 0})
            continue
        d = sub["delta"].to_numpy()
        ci_lo, ci_hi = _boot_ci(d)
        out.append(
            {
                "bin": label,
                "n": len(d),
                "mean": float(d.mean()),
                "median": float(np.median(d)),
                "win_rate": float((d > 0).mean()),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                # All three pre-registered conditions, together.
                "flagged": bool(len(d) >= MIN_N and (d > 0).mean() < 0.5 and ci_hi < 0),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    print("=" * 78)
    print("EXPLORATORY — a re-cut of data that already produced a verdict.")
    print("Any pattern here is a hypothesis for a confirmatory study on FRESH data.")
    print("=" * 78)

    examined = 0
    flagged: list[str] = []
    for stratum in ("A", "B"):
        df = pd.read_json(f"{args.dir}/rerun_{stratum}.json")
        df["delta"] = df["wape_control"] - df["wape_treatment"]
        print(
            f"\n{'=' * 78}\nSTRATUM {stratum} — n={len(df)}, "
            f"overall mean Δ {df.delta.mean():+.4f}, win rate {(df.delta > 0).mean():.3f}"
        )

        for col, bins, name in (
            ("gap_len", GAP_LEN_BINS, "H1  gap length (hours)"),
            ("gap_lead_h", LEAD_BINS, "H2  gap lead — hours before the origin"),
            ("gap_hour_utc", HOUR_BINS, "H3  gap hour UTC  [POST-HOC]"),
        ):
            print(f"\n  {name}")
            print(
                f"    {'bin':>8} {'n':>5} {'mean Δ':>9} {'median':>9} {'win':>7} "
                f"{'95% CI':>20}  flag"
            )
            for c in _cells(df, col, bins):
                if not c["n"]:
                    print(f"    {c['bin']:>8} {0:5}   (empty)")
                    continue
                examined += 1
                if c["flagged"]:
                    flagged.append(f"{stratum}/{col}/{c['bin']}")
                ci = f"[{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}]"
                print(
                    f"    {c['bin']:>8} {c['n']:5} {c['mean']:+9.4f} {c['median']:+9.4f} "
                    f"{c['win_rate']:7.3f} {ci:>20}  {'FLAG' if c['flagged'] else ''}"
                )

    # The pre-specified PSCO look, kept separate and labelled.
    print(f"\n{'=' * 78}\nPSCO — pre-specified, POST-HOC (suggested by a result already seen)")
    a = pd.read_json(f"{args.dir}/rerun_A.json")
    a["delta"] = a["wape_control"] - a["wape_treatment"]
    p = a[a.region == "PSCO"]
    rest = a[a.region != "PSCO"]
    print(
        f"    PSCO      n={len(p):3}  mean Δ {p.delta.mean():+.4f}  win {(p.delta > 0).mean():.3f}"
    )
    print(
        f"    stratum A n={len(rest):3}  mean Δ {rest.delta.mean():+.4f}  "
        f"win {(rest.delta > 0).mean():.3f}   (A without PSCO)"
    )

    print(f"\n{'=' * 78}\nMULTIPLICITY: {examined} cells examined, {len(flagged)} flagged.")
    print(f"  At a 5% level, ~{0.05 * examined:.1f} false flags are expected by chance alone.")
    if flagged:
        print(f"  Flagged: {', '.join(flagged)}")
        print("  An ISOLATED flag is consistent with chance. Only a coherent, monotone")
        print("  pattern across adjacent bins of one covariate is worth carrying forward.")
    else:
        print("  Nothing flagged — the losing windows are diffuse, and there is no")
        print("  subgroup to carve out. Per the pre-registration, that settles it.")


if __name__ == "__main__":
    main()
