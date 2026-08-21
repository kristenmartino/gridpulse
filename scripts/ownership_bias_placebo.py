"""Phase 2 of the ownership study: signed day-ahead bias, per balancing authority.

Implements ``docs/studies/OWNERSHIP_FORECAST_BIAS.md`` §4 and §9 against the
GCS vintage mirror. Reads no Redis and needs no cluster access.

Two things this script is deliberately built to do:

**It computes; nothing here reports a number it did not compute** (§8.1). The
control in ``--control`` reproduces a figure the published ``/benchmark``
payload already carries for the same BA, and refuses to continue if it
cannot — a script that cannot reproduce a known-good number is not trusted
with an unknown one.

**It stays blind to ownership.** No ownership classification is imported,
loaded or referenced. Per-BA rows come out; grouping happens downstream,
after these numbers exist. That ordering is the pre-registration's, and
keeping the two apart in code is what makes it real rather than stated.

Usage::

    python scripts/ownership_bias_placebo.py --control PJM
    python scripts/ownership_bias_placebo.py --out study_bias.json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.vintage import deserialize_records  # noqa: E402
from models.benchmark import _normalize_ts, pair_hours  # noqa: E402

BUCKET = "nextera-portfolio-energy-cache"

#: RTOs and ISOs are markets, not utilities, and have no ownership class
#: (§5). Excluded here rather than downstream so no aggregate can include
#: them by accident.
NOT_UTILITIES = frozenset({"CAISO", "ERCOT", "ISONE", "MISO", "NYISO", "PJM", "SPP"})


def load_records(region: str, bucket: str = BUCKET) -> list:
    from google.cloud import storage

    blob = storage.Client().bucket(bucket).get_blob(f"cache/vintage/{region}/latest.parquet")
    if blob is None:
        return []
    raw = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
    return sorted(deserialize_records(raw.to_dict("records")), key=lambda r: r.timestamp)


def _official_arm_pairs(records: list) -> list:
    """Hours where the OPERATOR's arm is fair, ignoring our availability (§9.2).

    Reuses ``pair_hours`` by handing it a GridPulse series that is present for
    every hour the official arm survived, so the `no_gridpulse` rule cannot
    fire. Every other fairness rule applies untouched — this does not
    reimplement the gate, it neutralises exactly one term of it.
    """
    # Keyed exactly as `pair_hours` looks them up, or the drop it is meant to
    # neutralise fires anyway and the arm silently reports zero hours.
    stand_in = {_normalize_ts(r.timestamp): 1.0 for r in records}
    pairs, drops = pair_hours(records, stand_in, exclude_stale_capture=True)
    assert drops["no_gridpulse"] == 0, (
        f"stand-in failed to neutralise the availability drop "
        f"({drops['no_gridpulse']} hours still dropped)"
    )
    return pairs


def signed_bias(pairs: list) -> dict:
    """Signed percentage error of the operator's as-issued forecast (§4).

    Positive means the operator forecast MORE demand than materialised.
    """
    if not pairs:
        return {"n": 0}
    actual = np.array([p.actual for p in pairs], dtype=float)
    official = np.array([p.official for p in pairs], dtype=float)
    ok = np.isfinite(actual) & np.isfinite(official) & (actual > 0)
    actual, official = actual[ok], official[ok]
    if actual.size == 0:
        return {"n": 0}
    pct = (official - actual) / actual * 100.0
    return {
        "n": int(actual.size),
        "bias_pct_mean": round(float(np.mean(pct)), 3),
        "bias_pct_median": round(float(np.median(pct)), 3),
        "share_over_forecast": round(float(np.mean(pct > 0)), 4),
        # Magnitude, for cross-reference with the published payload only.
        "mape": round(float(np.mean(np.abs(pct))), 3),
    }


def measure(region: str) -> dict:
    records = load_records(region)
    if not records:
        return {"region": region, "error": "no vintage mirror"}
    row: dict = {"region": region}
    row["operator_arm"] = signed_bias(_official_arm_pairs(records))
    bench_pairs, drops = pair_hours(records, {}, exclude_stale_capture=True)
    row["benchmark_arm"] = signed_bias(bench_pairs)
    row["benchmark_arm_note"] = "no_gridpulse applied; comparable to /benchmark"
    row["drops"] = {k: int(v) for k, v in drops.items()}
    return row


def run_control(region: str) -> bool:
    """Reproduce published values before trusting unpublished ones.

    The obvious control — reproduce the published official MAPE — is not
    available here. That figure is scored over the benchmark pairing, which
    includes the ``no_gridpulse`` drop, and GridPulse's per-hour predictions
    live in Redis rather than in the vintage mirror this script reads.
    Computing a near-match over a *different* hour set and calling it a
    reproduction would be precisely the adjacent-signal verification this
    project keeps getting caught by.

    So the control checks what IS exactly reproducible from the mirror: the
    per-reason drop counts the payload publishes. They come from the same
    ``pair_hours`` gate, so agreement validates the pairing path itself.
    Every reason except ``no_gridpulse`` must match.
    """
    import urllib.request

    url = "https://gridpulse.kristenmartino.ai/api/v1/benchmark"
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
        payload = json.load(resp)
    published = None
    for r in payload.get("regions", []):
        if r.get("region") == region:
            published = ((r.get("leads") or {}).get("24h") or {}).get("excluded_hours")
    if not published:
        print(f"CONTROL FAILED: /benchmark publishes no drop counts for {region}")
        return False

    records = load_records(region)
    if not records:
        print(f"CONTROL FAILED: no vintage mirror for {region}")
        return False
    _, mine = pair_hours(records, {}, exclude_stale_capture=True)

    # Mirror and payload are written by different ticks, so edge hours drift
    # by a few counts. A pairing BUG would not be a few.
    reasons = [k for k in published if k != "no_gridpulse"]
    worst = 0.0
    print(f"CONTROL {region}: per-reason drop counts, mirror vs published")
    for reason in sorted(reasons):
        pub, got = int(published[reason]), int(mine.get(reason, -1))
        denom = max(pub, got, 1)
        rel = abs(pub - got) / denom
        worst = max(worst, rel)
        flag = "" if rel <= 0.10 else "   <-- MISMATCH"
        print(f"   {reason:24s} published={pub:5d}  computed={got:5d}{flag}")
    ok = worst <= 0.10
    print(f"CONTROL {'PASS' if ok else 'FAILED'}: worst relative gap {worst:.1%}")
    if not ok:
        print("  The pairing path disagrees with production. Not proceeding.")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--control", default="PJM", help="BA used for the reproduce-a-known-number check"
    )
    ap.add_argument("--out", type=Path, help="write per-BA rows as JSON")
    ap.add_argument("--skip-control", action="store_true", help="not for study runs")
    args = ap.parse_args()

    if not args.skip_control and not run_control(args.control):
        return 1
    print()

    from config import REGION_NAMES

    regions = [r for r in sorted(REGION_NAMES) if r not in NOT_UTILITIES]
    print(f"measuring {len(regions)} BAs (RTO/ISOs excluded per §5)\n")
    rows = []
    for region in regions:
        row = measure(region)
        rows.append(row)
        op = row.get("operator_arm", {})
        if op.get("n"):
            print(
                f"  {region:6s} n={op['n']:5d}  signed mean={op['bias_pct_mean']:+7.2f}%  "
                f"median={op['bias_pct_median']:+7.2f}%  over={op['share_over_forecast']:.0%}"
            )
        else:
            print(f"  {region:6s} {row.get('error', 'no scoreable hours')}")

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    print("\nNo ownership grouping performed. That happens downstream, per §8.3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
