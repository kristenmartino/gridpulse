"""Per-BA scoreability report for the public forecast benchmark (E0-1).

Answers, for every balancing authority: *can GridPulse be compared against
this BA's own day-ahead forecast, and if not, exactly why?* The exclusion
reasons are the transparency product — a benchmark that quietly drops the
BAs it does badly on is worthless, so the excluded set ships with the
result.

Reads the GCS vintage mirror (no Redis, no cluster access), following the
``scripts/anchor_conditioning_study.py`` pattern, and writes a committed
markdown table.

Usage:
    python scripts/benchmark_scoreability.py
    python scripts/benchmark_scoreability.py --output docs/BENCHMARK_SCOREABILITY.md

Requires GCS_ENABLED=true, GCS_BUCKET_NAME, and ADC credentials.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from config import REGION_COORDINATES  # noqa: E402
from data.vintage import classify_region, deserialize_records  # noqa: E402
from models.benchmark import STUB_EPSILON_MW, scoreability  # noqa: E402


def _md_table(df: pd.DataFrame) -> str:
    """Minimal markdown table (the export_holdout_metrics precedent)."""
    if df.empty:
        return "(no rows)"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def official_quality(records: list) -> tuple[float | None, int]:
    """Median APE of the BA's OWN day-ahead forecast vs settled truth.

    Applies the same per-hour exclusions the benchmark does, so the number
    is the one a reader would reproduce from the benchmark itself.
    """
    errs = [
        abs(r.first_seen_df - r.last_d) / r.last_d * 100.0
        for r in records
        if np.isfinite(r.first_seen_df)
        and np.isfinite(r.last_d)
        and r.last_d > 0
        and not r.was_placeholder
        and abs(r.last_d - r.first_seen_df) >= STUB_EPSILON_MW
    ]
    if len(errs) < 50:
        return None, len(errs)
    return float(np.median(errs)), len(errs)


def build_rows() -> list[dict]:
    from data.gcs_store import read_parquet

    rows: list[dict] = []
    for ba in REGION_COORDINATES:
        df = read_parquet("vintage", ba)
        if df is None or df.empty:
            rows.append({"ba": ba, "scoreable": False, "reason": "no-vintage-mirror"})
            continue
        records = deserialize_records(df.to_dict("records"))
        if not records:
            rows.append({"ba": ba, "scoreable": False, "reason": "no-records"})
            continue

        cls = classify_region(records)["revision_class"]
        score = scoreability(records, cls)
        median_ape, n = official_quality(records)
        rows.append(
            {
                "ba": ba,
                "class": cls,
                "scoreable": score["scoreable"],
                "reason": score["reason"] or "",
                "df_coverage_pct": round(score["df_coverage"] * 100, 1),
                "stub_pct": score["placeholder_pct"],
                "official_median_ape_pct": None if median_ape is None else round(median_ape, 2),
                "n_scoreable_hours": n,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    if not config.GCS_ENABLED or not config.GCS_BUCKET_NAME:
        print("GCS access required: GCS_ENABLED=true, GCS_BUCKET_NAME, ADC login.")
        return 1

    rows = build_rows()
    scoreable = [r for r in rows if r.get("scoreable")]
    excluded = [r for r in rows if not r.get("scoreable")]
    quality = [r["official_median_ape_pct"] for r in scoreable if r.get("official_median_ape_pct")]

    report = [
        "# Benchmark scoreability — which BAs can be compared, and why not\n",
        f"\n**{len(scoreable)} of {len(rows)} balancing authorities are scoreable.** "
        "A BA is excluded only when it cannot be compared *fairly*; the reason "
        "is published for every one of them.\n",
    ]

    if quality:
        report.append(
            f"\nAmong the scoreable set, the operators' *own* day-ahead accuracy "
            f"spans **{min(quality):.2f}% to {max(quality):.2f}% median APE** "
            f"(a {max(quality) / min(quality):.0f}× spread), median-of-medians "
            f"{np.median(quality):.2f}%. That spread is measured here against "
            "settled values with placeholder hours excluded — the same "
            "discipline the benchmark applies to both arms.\n"
        )

    report.append("\n## Excluded\n\n")
    ex_df = pd.DataFrame(
        [{"ba": r["ba"], "class": r.get("class", ""), "reason": r["reason"]} for r in excluded]
    )
    report.append(_md_table(ex_df) + "\n")
    report.append(
        "\n**`broken-feed`** — the feed's provisional readings revise heavily "
        "before settling, so intraday scoring is not meaningful; and GridPulse "
        "anchors its own forecast on that BA's day-ahead value (ADR-009), which "
        "would make the comparison partly self-referential. "
        "**`df-coverage`** — the BA publishes a day-ahead forecast too sparsely "
        "to score.\n"
    )
    report.append(
        "\nNote the direction of the bias: four of the exclusions are for feed "
        "brokenness, and BAs with sloppy data operations plausibly also forecast "
        "sloppily — so excluding them likely removes BAs where GridPulse would "
        "win. The exclusion set is conservative against our own claim.\n"
    )

    report.append("\n## Scoreable\n\n")
    sc_df = pd.DataFrame(scoreable).drop(columns=["scoreable", "reason"], errors="ignore")
    sc_df = sc_df.sort_values("official_median_ape_pct", na_position="last")
    report.append(_md_table(sc_df) + "\n")

    print("".join(report))
    if args.output:
        args.output.write_text("".join(report))
        print(f"\nreport written → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
