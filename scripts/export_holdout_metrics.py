#!/usr/bin/env python3
"""Export per-BA holdout metrics from the GCS model store into a table.

Reads the SAME per-region 168-hour holdout metrics the daily training job
computes and persists to each model's GCS ``meta.json`` (and that the live
Models tab reads from Redis via ``model_metrics``). It does **not** train
anything and does **not** fetch EIA/weather data — it only reads metadata,
so it runs in minutes and covers every trained BA.

Prefer this over ``scripts/backtest_all.py`` when refreshing the accuracy
numbers in ``docs/BACKTEST_RESULTS.md`` / ``docs/CANONICAL_FACTS.md``: the
numbers here are current, de-leaked (post-PR #135), and span all 51 BAs —
not a 90-day-fetch / 21-day-holdout recompute on a different methodology.

Prerequisites on the machine that runs it:
  * ``google-cloud-storage`` installed and credentials available
    (``gcloud auth application-default login`` or a service-account key via
    ``GOOGLE_APPLICATION_CREDENTIALS``).
  * GCS configured for the repo — easiest is ``ENVIRONMENT=production`` so
    ``config`` resolves the real bucket; or set ``GCS_ENABLED=true`` +
    ``GCS_BUCKET_NAME`` + ``GCS_PATH_PREFIX`` explicitly.

Outputs:
  * Markdown table (one row per BA) — written to ``--out-md`` if given,
    always echoed to stdout.
  * CSV with the full ``{mape, rmse, mae, r2}`` per model — ``--out-csv``.
  * A per-BA MAPE **distribution** (min/median/mean/p90/max + worst BAs).
    Deliberately NOT a single "across-51" figure — accuracy is per-BA, and
    one pooled number hides the tail (e.g. LDWP/AZPS/SPA).

Usage:
    python scripts/export_holdout_metrics.py
    python scripts/export_holdout_metrics.py \
        --out-md docs/_holdout_table.md --out-csv holdout_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Any

# Repo root on path so ``config`` / ``models`` import when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE_MODELS = ("xgboost", "prophet", "arima")
METRIC_FIELDS = ("mape", "rmse", "mae", "r2")


def _metrics_from_meta(meta: Any) -> dict[str, Any] | None:
    """Return the full holdout-metric dict for one model, or ``None``.

    Prefers ``extra['holdout_metrics']`` (the full ``{mape,rmse,mae,r2}``);
    falls back to the top-level ``mape`` for older metas that predate the
    richer payload.
    """
    if meta is None:
        return None
    extra = getattr(meta, "extra", None) or {}
    hm = extra.get("holdout_metrics")
    if isinstance(hm, dict) and hm:
        return {k: hm.get(k) for k in METRIC_FIELDS}
    if getattr(meta, "mape", None) is not None:
        return {"mape": meta.mape, "rmse": None, "mae": None, "r2": None}
    return None


def _ensemble_from_xgb_meta(meta: Any) -> dict[str, Any] | None:
    """Pull the ensemble holdout metrics, stashed on the xgboost meta extra."""
    if meta is None:
        return None
    extra = getattr(meta, "extra", None) or {}
    em = extra.get("ensemble_holdout_metrics")
    if isinstance(em, dict) and em:
        return {k: em.get(k) for k in METRIC_FIELDS}
    return None


def _pct(sorted_vals: list[float], q: float) -> float | None:
    """Linear-interpolated percentile of an already-sorted list (numpy-free)."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _fmt(v: Any, suffix: str = "%") -> str:
    """Format a metric value, or em-dash when missing."""
    return f"{v:.2f}{suffix}" if isinstance(v, int | float) else "—"


def _headline_mape(row: dict[str, Any]) -> float | None:
    """Per-BA MAPE for the distribution: ensemble, else XGBoost, else best base."""
    ens = row.get("ensemble") or {}
    if isinstance(ens.get("mape"), int | float):
        return ens["mape"]
    xgb = row["models"].get("xgboost") or {}
    if isinstance(xgb.get("mape"), int | float):
        return xgb["mape"]
    cand = [
        m["mape"] for m in row["models"].values() if m and isinstance(m.get("mape"), int | float)
    ]
    return min(cand) if cand else None


def collect(regions: list[str]) -> list[dict[str, Any]]:
    """Read the latest per-model holdout metadata for each region from GCS."""
    from models.persistence import get_model_metadata, invalidate_latest_cache

    invalidate_latest_cache()  # force a fresh latest.json read
    rows: list[dict[str, Any]] = []
    for region in regions:
        per_model: dict[str, dict[str, Any] | None] = {}
        xgb_meta: Any = None
        provenance: Any = None
        for model in BASE_MODELS:
            meta = get_model_metadata(region, model)
            per_model[model] = _metrics_from_meta(meta)
            if model == "xgboost":
                xgb_meta = meta
            if meta is not None and provenance is None:
                provenance = meta
        provenance = xgb_meta or provenance
        rows.append(
            {
                "region": region,
                "models": per_model,
                "ensemble": _ensemble_from_xgb_meta(xgb_meta),
                "train_rows": getattr(provenance, "train_rows", None),
                "trained_at": getattr(provenance, "trained_at", None),
                "version": getattr(provenance, "version", None),
            }
        )
    return rows


def to_markdown(rows: list[dict[str, Any]], names: dict[str, str]) -> str:
    """Render the one-row-per-BA markdown table."""
    lines = [
        "| BA | Region | XGBoost | Prophet | ARIMA | Ensemble | Best base | "
        "Train rows | Trained (UTC) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: x["region"]):
        base = {k: (r["models"].get(k) or {}).get("mape") for k in BASE_MODELS}
        finite = {k: v for k, v in base.items() if isinstance(v, int | float)}
        best = min(finite, key=lambda k: finite[k]) if finite else None
        ens = (r["ensemble"] or {}).get("mape")
        trained = (r["trained_at"] or "")[:10] or "—"
        rows_n = r["train_rows"] if r["train_rows"] is not None else "—"
        lines.append(
            f"| {r['region']} | {names.get(r['region'], '')} | "
            f"{_fmt(base['xgboost'])} | {_fmt(base['prophet'])} | "
            f"{_fmt(base['arima'])} | {_fmt(ens)} | {best or '—'} | "
            f"{rows_n} | {trained} |"
        )
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict[str, Any]], names: dict[str, str], path: str) -> None:
    """Write one row per (region, model) including the ensemble row."""
    fieldnames = [
        "region",
        "name",
        "model",
        *METRIC_FIELDS,
        "train_rows",
        "trained_at",
        "version",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["region"]):
            entries = list(r["models"].items())
            if r["ensemble"]:
                entries.append(("ensemble", r["ensemble"]))
            for model, m in entries:
                if not m:
                    continue
                w.writerow(
                    {
                        "region": r["region"],
                        "name": names.get(r["region"], ""),
                        "model": model,
                        **{fld: m.get(fld) for fld in METRIC_FIELDS},
                        "train_rows": r["train_rows"],
                        "trained_at": r["trained_at"],
                        "version": r["version"],
                    }
                )


def print_summary(rows: list[dict[str, Any]]) -> None:
    """Print the per-BA MAPE distribution — never a single pooled number."""
    headline = [(r["region"], _headline_mape(r)) for r in rows]
    have = [(reg, v) for reg, v in headline if isinstance(v, int | float)]
    missing = sorted(reg for reg, v in headline if v is None)

    print(f"\nBAs with holdout metrics: {len(have)} / {len(rows)}")
    if missing:
        print(f"Missing (no GCS metadata): {', '.join(missing)}")
    if not have:
        print(
            "\nNo metrics found. Check that GCS is enabled "
            "(ENVIRONMENT=production or GCS_ENABLED=true) and that credentials "
            "can read the model bucket (gcloud auth application-default login)."
        )
        return

    vals = sorted(v for _, v in have)
    print("\nPer-BA MAPE distribution (ensemble where available, else XGBoost):")
    print(f"  n      = {len(vals)}")
    print(f"  min    = {vals[0]:.2f}%")
    print(f"  median = {statistics.median(vals):.2f}%")
    print(f"  mean   = {statistics.mean(vals):.2f}%")
    print(f"  p90    = {_pct(vals, 0.90):.2f}%")
    print(f"  max    = {vals[-1]:.2f}%")
    print("  (A DISTRIBUTION, not a single 'across-51' accuracy number.)")

    worst = sorted(have, key=lambda x: x[1], reverse=True)[:5]
    print("\nWorst 5 BAs by MAPE:")
    for reg, v in worst:
        print(f"  {reg:6s} {v:.2f}%")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regions",
        default="all",
        help="'all' (default) or a comma-separated list, e.g. ERCOT,FPL,PJM",
    )
    parser.add_argument("--out-md", default=None, help="Write the markdown table here.")
    parser.add_argument("--out-csv", default=None, help="Write the full CSV here.")
    args = parser.parse_args(argv)

    from config import GCS_ENABLED, REGION_COORDINATES, REGION_NAMES

    if not GCS_ENABLED:
        print(
            "GCS is disabled for this environment. Set ENVIRONMENT=production "
            "(or GCS_ENABLED=true + GCS_BUCKET_NAME) and authenticate with "
            "`gcloud auth application-default login`, then re-run.",
            file=sys.stderr,
        )
        return 2

    if args.regions.strip().lower() == "all":
        regions = sorted(REGION_COORDINATES.keys())
    else:
        regions = [r.strip().upper() for r in args.regions.split(",") if r.strip()]
        unknown = [r for r in regions if r not in REGION_COORDINATES]
        if unknown:
            print(f"Unknown region(s): {', '.join(unknown)}", file=sys.stderr)
            return 1

    rows = collect(regions)
    md = to_markdown(rows, REGION_NAMES)
    print(md)

    if args.out_md:
        Path(args.out_md).write_text(md)
        print(f"Wrote markdown table -> {args.out_md}")
    if args.out_csv:
        write_csv(rows, REGION_NAMES, args.out_csv)
        print(f"Wrote CSV -> {args.out_csv}")

    print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
