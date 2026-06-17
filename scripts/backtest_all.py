#!/usr/bin/env python3
"""Run the holdout backtest across many BAs and tabulate the results.

FALLBACK / reproducibility tool. Prefer ``scripts/export_holdout_metrics.py``
when you just need current numbers: this script RE-TRAINS Prophet + SARIMAX +
XGBoost per region on freshly fetched EIA + weather data, so:

  * it is SLOW — roughly 10-20 min per BA; all 51 is multiple hours;
  * it uses a DIFFERENT methodology than production — a ~90-day fetch /
    21-day holdout vs the training job's rolling 168-hour holdout — and it
    scores against TODAY's data, not the Feb-2026 reference snapshot.

Use it only when you want a self-contained, reproducible offline backtest
artifact AND the machine has: the ML deps installed (xgboost, prophet,
pmdarima, statsmodels, scikit-learn), ``EIA_API_KEY`` set, and network access
to api.eia.gov + Open-Meteo.

Outputs match ``export_holdout_metrics.py`` (shared renderers) so either
source drops cleanly into the docs:
  * Markdown table (one row per BA) — ``--out-md`` (always echoed to stdout);
  * CSV with full ``{mape, rmse, mae, r2}`` per model — ``--out-csv``;
  * a per-BA MAPE distribution summary (never a single pooled number).

Usage:
    python scripts/backtest_all.py --regions ERCOT,FPL,PJM     # a few BAs
    python scripts/backtest_all.py --regions all --limit 3 -q  # quick smoke
    python scripts/backtest_all.py --regions all --holdout-days 21 \
        --out-md docs/_backtest_all.md --out-csv backtest_all.csv
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))  # sibling scripts (backtest, exporter)
sys.path.insert(0, str(_SCRIPTS_DIR.parent))  # repo root for config / data / models

import backtest as bt_mod  # noqa: E402  (path set above)
from export_holdout_metrics import (  # noqa: E402  (shared renderers)
    BASE_MODELS,
    METRIC_FIELDS,
    print_summary,
    to_markdown,
    write_csv,
)


def _empty_row(region: str, error: str) -> dict[str, Any]:
    return {
        "region": region,
        "models": {m: None for m in BASE_MODELS},
        "ensemble": None,
        "train_rows": None,
        "trained_at": None,
        "version": None,
        "error": error,
    }


def run_region(region: str, holdout_days: int, quiet: bool) -> dict[str, Any]:
    """Backtest one region and shape it like ``export_holdout_metrics`` rows."""
    sink = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    try:
        with sink:
            results = bt_mod.run_backtest(region, holdout_days)
    except Exception as e:  # network / data / fit failure — keep going
        return _empty_row(region, str(e))

    if not results or "error" in results:
        return _empty_row(region, (results or {}).get("error", "no results"))

    def metrics(name: str) -> dict[str, Any] | None:
        block = results.get(name) or {}
        m = block.get("metrics")
        return {k: m.get(k) for k in METRIC_FIELDS} if isinstance(m, dict) else None

    return {
        "region": region,
        "models": {m: metrics(m) for m in BASE_MODELS},
        "ensemble": metrics("ensemble"),
        "train_rows": None,
        "trained_at": None,
        "version": f"backtest-{holdout_days}d",
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regions",
        default="all",
        help="'all' (default) or a comma-separated list, e.g. ERCOT,FPL,PJM",
    )
    parser.add_argument("--holdout-days", type=int, default=21, help="Holdout length (default 21).")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N regions.")
    parser.add_argument("--out-md", default=None, help="Write the markdown table here.")
    parser.add_argument("--out-csv", default=None, help="Write the full CSV here.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress per-run backtest logs.")
    args = parser.parse_args(argv)

    from config import REGION_COORDINATES, REGION_NAMES

    if args.regions.strip().lower() == "all":
        regions = sorted(REGION_COORDINATES.keys())
    else:
        regions = [r.strip().upper() for r in args.regions.split(",") if r.strip()]
        unknown = [r for r in regions if r not in REGION_COORDINATES]
        if unknown:
            print(f"Unknown region(s): {', '.join(unknown)}", file=sys.stderr)
            return 1
    if args.limit is not None:
        regions = regions[: args.limit]

    print(
        f"Backtesting {len(regions)} BA(s) at a {args.holdout_days}-day holdout. "
        "This RE-TRAINS 3 models per BA on fresh data — expect ~10-20 min each.",
        file=sys.stderr,
    )

    rows: list[dict[str, Any]] = []
    for i, region in enumerate(regions, start=1):
        print(f"[{i}/{len(regions)}] {region} ...", file=sys.stderr, flush=True)
        row = run_region(region, args.holdout_days, args.quiet)
        if row.get("error"):
            print(f"    skipped: {row['error']}", file=sys.stderr)
        rows.append(row)

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
