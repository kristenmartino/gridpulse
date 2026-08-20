"""Attribute the ensemble-24h drift-window shortfall to its loss channels.

Evidence for [`docs/DRIFT_COVERAGE_CHANNELS.md`](../docs/DRIFT_COVERAGE_CHANNELS.md)
(#537). Answers: of the hours missing from each BA's rolling 7-day horizon-drift
window, how many were never proposed as a snapshot, and how many were proposed
and never resolved?

Two stages, because production Redis is a private Memorystore:

1. ``--dump`` runs INSIDE the VPC (inlined into a Cloud Run job execution, see
   the invocation note in the doc) and uploads every ``drift_horizon:{BA}``
   payload to GCS.
2. ``--analyze`` runs locally against that dump plus the GCS vintage mirror.

Classification is exact, not inferential. For a target hour ``T`` in
``W_full = [t_ref-119h, t_ref-1h]`` — a window chosen to sit entirely inside
``PENDING_STALE_HOURS``, so nothing in it can have been expired — exactly one of:

* **RESOLVED** — ``T`` is in the ensemble/24h ``records``
* **PENDING**  — ``T`` is in ``pending`` with ``horizon == "24h"`` (channel C)
* **ABSENT**   — neither, so no snapshot was ever taken (channels A and B)

ABSENT splits by the origin-skip test. A 24h snapshot for ``T`` is taken by the
tick whose resolved origin is ``T-24h``, which requires some tick to see hour
``T-25h`` as its newest published hour. If EIA published ``T-25h`` and ``T-24h``
in the same tick, no tick ever did, the origin jumped past it, and the target was
never proposed:

* **A** — ``captured_at(T-25h)`` and ``captured_at(T-24h)`` fall in the same tick
* **B** — they do not, so the origin was held by something else
  (``last_featured_ts``, the #559 row deletion)

Controls run before any per-BA result is printed, and the run refuses to report
if the cross-instrument check fails.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

BUCKET = "nextera-portfolio-energy-cache"
DUMP_BLOB = "probe/drift_channel_split.json"
WINDOW_HOURS = 119
CONTROLS = ("TAL", "CPLW", "FMPP", "GCPD")


# ── stage 1: in-VPC dump ──────────────────────────────────────────────────
def dump() -> None:
    """Read every drift_horizon payload from Redis and upload it to GCS."""
    from google.cloud import storage

    from config import REGION_COORDINATES
    from data.redis_client import redis_get, redis_key

    out: dict[str, Any] = {}
    for region in sorted(REGION_COORDINATES):
        payload = redis_get(redis_key(f"drift_horizon:{region}"))
        if not isinstance(payload, dict):
            out[region] = {"missing": True}
            continue
        blocks = {}
        for name, by_horizon in (payload.get("models") or {}).items():
            block = (by_horizon or {}).get("24h") if isinstance(by_horizon, dict) else None
            if isinstance(block, dict):
                blocks[name] = {
                    k: block.get(k)
                    for k in (
                        "n_7d",
                        "n_30d",
                        "n_records",
                        "n_low_actual_excluded_7d",
                        "rolling_mape_7d",
                        "records",
                    )
                }
        out[region] = {
            "last_updated_at": payload.get("last_updated_at"),
            "blocks": blocks,
            "pending": [
                {"t": s.get("target_ts"), "h": s.get("horizon")}
                for s in (payload.get("pending") or [])
            ],
        }

    body = json.dumps(out)
    storage.Client().bucket(BUCKET).blob(DUMP_BLOB).upload_from_string(body)
    print(f"UPLOADED gs://{BUCKET}/{DUMP_BLOB} bytes={len(body)}")


# ── stage 2: local analysis ───────────────────────────────────────────────
def _norm(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(minute=0, second=0, microsecond=0)


def _load_vintage(region: str) -> tuple[str, dict[datetime, tuple[datetime, float]]]:
    import pandas as pd
    from google.cloud import storage

    blob = storage.Client().bucket(BUCKET).blob(f"cache/vintage/{region}/latest.parquet")
    frame = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
    ts = pd.to_datetime(frame["ts"], utc=True)
    at = pd.to_datetime(frame["at"], utc=True)
    lag = (at - ts).dt.total_seconds() / 3600.0
    return region, {
        t.to_pydatetime(): (a.to_pydatetime(), float(g))
        for t, a, g in zip(ts, at, lag, strict=True)
    }


def analyze(dump_path: str) -> int:
    from models.drift import (
        PENDING_STALE_HOURS,
        _within_window,
        deserialize_records,
        filter_low_actuals,
    )

    with open(dump_path) as fh:
        raw = json.load(fh)
    raw.pop("_live_1h", None)
    regions = sorted(r for r in raw if not raw[r].get("missing"))
    with ThreadPoolExecutor(max_workers=10) as pool:
        vintage = dict(pool.map(_load_vintage, regions))

    # ── controls, evaluated before any per-BA result is read ──────────────
    def reproduce(region: str, hours: int) -> tuple[int, int]:
        block = raw[region]["blocks"]["ensemble"]
        recs = deserialize_records(block["records"])
        kept, n_low = filter_low_actuals(
            _within_window(recs, hours, now_iso=raw[region]["last_updated_at"])
        )
        return len(kept), n_low

    c3 = [r for r in regions if reproduce(r, 168)[0] != raw[r]["blocks"]["ensemble"]["n_7d"]]
    c3b = [
        r
        for r in regions
        if reproduce(r, 168)[1] != raw[r]["blocks"]["ensemble"]["n_low_actual_excluded_7d"]
    ]
    c4 = [r for r in regions if reproduce(r, 167)[0] != raw[r]["blocks"]["ensemble"]["n_7d"]]

    print("CONTROLS")
    print(f"  C3  n_7d reproduced from records            {len(regions) - len(c3)}/{len(regions)}")
    print(f"  C3b n_low_actual_excluded_7d reproduced     {len(regions) - len(c3b)}/{len(regions)}")
    print(
        f"  C4  deliberately-wrong 167h window DISAGREES {len(c4)}/{len(regions)}"
        "   (must be non-trivial, else C3 is vacuous)"
    )
    if c3 or c3b:
        print(f"\nABORT: harness does not reproduce production. {c3[:5]} {c3b[:5]}")
        return 1
    if len(c4) < len(regions) // 2:
        print("\nABORT: the wrong-window control did not disagree; C3 proves nothing.")
        return 1

    # ── expiry bound over the full 168h window ────────────────────────────
    eligible = []
    for region in regions:
        t_ref = _norm(raw[region]["last_updated_at"])
        for k in range(168, 0, -1):
            hour = t_ref - timedelta(hours=k)
            entry = vintage[region].get(hour)
            if entry is None or entry[1] > PENDING_STALE_HOURS:
                eligible.append((region, str(hour)[:16]))
    print(
        f"\nEXPIRY BOUND: hours in the 168h window whose actual never published or"
        f" published later than {PENDING_STALE_HOURS}h: {len(eligible)} of {168 * len(regions)}"
    )
    for row in eligible:
        print("   ", row)

    # ── per-hour attribution ──────────────────────────────────────────────
    rows = []
    skip_test = {"ABS": [0, 0], "RES": [0, 0]}
    for region in regions:
        block = raw[region]["blocks"]["ensemble"]
        t_ref = _norm(raw[region]["last_updated_at"])
        resolved_hours = {_norm(r.timestamp) for r in deserialize_records(block["records"])}
        pending = {_norm(p["t"]) for p in raw[region]["pending"] if p["h"] == "24h"}
        window = [t_ref - timedelta(hours=k) for k in range(WINDOW_HOURS, 0, -1)]

        a = b = c = unknown = 0
        for hour in window:
            prev = vintage[region].get(hour - timedelta(hours=25))
            orig = vintage[region].get(hour - timedelta(hours=24))
            same = (
                None
                if prev is None or orig is None
                else _norm(prev[0].isoformat()) == _norm(orig[0].isoformat())
            )
            if hour in resolved_hours:
                if same is not None:
                    skip_test["RES"][0 if same else 1] += 1
                continue
            if hour in pending:
                c += 1
                continue
            if same is None:
                unknown += 1
            else:
                skip_test["ABS"][0 if same else 1] += 1
                if same:
                    a += 1
                else:
                    b += 1

        assert (
            a + b + c + unknown + len([h for h in window if h in resolved_hours]) == WINDOW_HOURS
        ), f"partition identity failed for {region}"
        rows.append(
            {
                "region": region,
                "n_7d": block["n_7d"],
                "resolved": WINDOW_HOURS - a - b - c - unknown,
                "A_skip": a,
                "B_freeze": b,
                "C_unresolved": c,
                "unknown": unknown,
            }
        )

    print(f"  C2  partition identity                      {len(rows)}/{len(rows)}")
    print("\nORIGIN-SKIP TEST (ABSENT should show the same-tick signature, RESOLVED must not)")
    for key, label in (("ABS", "ABSENT  "), ("RES", "RESOLVED")):
        same, diff = skip_test[key]
        total = same + diff
        print(f"  {label} same-tick {same:>5} / {total:<5} = {100 * same / total:6.2f}%")

    print("\n  C1  never-frozen controls, channel C:", end=" ")
    print(", ".join(f"{r['region']}={r['C_unresolved']}" for r in rows if r["region"] in CONTROLS))

    rows.sort(key=lambda r: r["n_7d"])
    print("\n" + "=" * 74)
    print(f"PER-BA CHANNEL ATTRIBUTION — ensemble 24h, W_full = {WINDOW_HOURS} h")
    print("=" * 74)
    print(
        f"{'BA':6} {'n_7d':>5} {'resolved':>8} | {'A skip':>6} {'B freeze':>8} {'C unres':>7} {'?':>3}"
    )
    for r in rows:
        print(
            f"{r['region']:6} {r['n_7d']:>5} {r['resolved']:>8} | {r['A_skip']:>6} "
            f"{r['B_freeze']:>8} {r['C_unresolved']:>7} {r['unknown']:>3}"
        )
    total = {k: sum(r[k] for r in rows) for k in ("A_skip", "B_freeze", "C_unresolved", "unknown")}
    short = sum(total.values())
    print("-" * 74)
    print(
        f"{'FLEET':6} {sum(r['n_7d'] for r in rows):>5} {sum(r['resolved'] for r in rows):>8} | "
        f"{total['A_skip']:>6} {total['B_freeze']:>8} {total['C_unresolved']:>7} "
        f"{total['unknown']:>3}"
    )
    for key, label in (
        ("A_skip", "A origin skip     "),
        ("B_freeze", "B origin freeze   "),
        ("C_unresolved", "C unresolved actual"),
    ):
        print(f"   {label} {total[key]:>4}  {100 * total[key] / short:5.1f}% of shortfall")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", action="store_true", help="run inside the VPC; upload to GCS")
    parser.add_argument("--analyze", metavar="PATH", help="local dump file to analyse")
    args = parser.parse_args()
    if args.dump:
        dump()
        return 0
    if args.analyze:
        return analyze(args.analyze)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
