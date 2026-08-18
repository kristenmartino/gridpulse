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
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from config import REGION_COORDINATES  # noqa: E402
from data.vintage import (  # noqa: E402
    FRESH_CAPTURE_LAG_HOURS,
    classify_region,
    deserialize_records,
    df_capture_lag_hours,
)
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
    is the one a reader would reproduce from the benchmark itself. That
    includes the ``stale_capture`` rule (#358/#392, measured on ``df_at`` since
    #535): a DF observed after the hour settled is a post-revision value, and
    grading the operator on it here while the benchmark refuses to would make
    this table quietly kinder to them than the scorecard is.
    """
    errs = [
        abs(r.first_seen_df - r.last_d) / r.last_d * 100.0
        for r in records
        if np.isfinite(r.first_seen_df)
        and np.isfinite(r.last_d)
        and r.last_d > 0
        and not r.was_placeholder
        and abs(r.last_d - r.first_seen_df) >= STUB_EPSILON_MW
        and (lag := df_capture_lag_hours(r)) is not None
        and lag <= FRESH_CAPTURE_LAG_HOURS
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
                # Our capture rate, beside the BA's publication rate. Publishing
                # only the first is how #535 stayed invisible: a reader could
                # not tell a BA that does not publish from one we failed to
                # record in time.
                "df_asissued_pct": round(score["df_asissued_coverage"] * 100, 1),
                "stub_pct": score["placeholder_pct"],
                "official_median_ape_pct": None if median_ape is None else round(median_ape, 2),
                "n_scoreable_hours": n,
                # The gate itself since #549, published for scored BAs too so
                # a reader sees the margin rather than trusting the verdict.
                "df_stale_hours": score["df_stale_hours"],
                "absent_bias_pct": score["absent_hours_bias_pct"],
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

    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    report = [
        "# Benchmark scoreability — which BAs can be compared, and why not\n",
        # NOT a standing claim. This file is generated on demand and was, for
        # three weeks, a 2026-07-27 snapshot asserting "44 of 51" in the present
        # tense while the live payload served 25 — the #535 drift. The count is
        # a measurement with a date on it; the API is the authority.
        f"\n*Generated {stamp} from the GCS vintage mirror. A dated snapshot, not "
        "a standing figure — the current count is whatever "
        "[`/api/v1/benchmark`](https://gridpulse.kristenmartino.ai/api/v1/benchmark) "
        "reports as `n_scoreable`, computed by the same "
        "`models.benchmark.scoreability` this script calls. Where the two "
        "disagree, this file is the stale one.*\n",
        f"\n**As measured on {stamp}: {len(scoreable)} of {len(rows)} balancing "
        "authorities clear the scoreability gate.** A BA is excluded only when "
        "it cannot be compared *fairly*; the reason is published for every one "
        "of them.\n",
        # The gate is not the whole rule, and the difference is not staleness.
        # `scoreability()` answers "can this BA be compared at all"; the payload
        # then drops any BA left with fewer than MIN_PAIRED_HOURS comparable
        # hours, which is a per-lead question this script does not evaluate. On
        # 2026-08-18 that is exactly one BA (MISO, 175 paired hours), so the
        # gate reads 46 and `n_scoreable` reads 45. A reader who finds the two
        # numbers and no explanation would reasonably assume one is wrong.
        "\nThis is the **gate** count. The live payload additionally requires "
        "at least `MIN_PAIRED_HOURS` comparable hours per lead, so its "
        "`n_scoreable` can be lower — a BA that publishes a day-ahead forecast "
        "but has too thin a paired sample is reported as "
        "`insufficient-paired-hours`, which is a different fact from "
        "`df-feed-stopped` and is published as such.\n",
        "\n`df_coverage_pct` is the **BA's** publication rate — the share of "
        "hours EIA carried a day-ahead forecast for. `df_asissued_pct` is "
        "**ours**: the share we observed early enough to score as-issued. "
        "Before #535 these were one number, and the second was being "
        "published as the first.\n",
        "\n**Neither one gates (#549).** `df_stale_hours` does — hours since "
        "the BA's most recent published day-ahead forecast, against a 168h "
        "ceiling. A rate cannot tell a BA that half-publishes from one that "
        "published completely and then stopped, and no BA in this fleet is "
        "diffusely sparse: every one with any absence has 92–100% of those "
        "hours inside runs of ≥3h.\n",
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
        "**`df-feed-stopped`** — the BA has stopped publishing a day-ahead "
        "forecast, so every hour we could score predates the stop and the row "
        "would describe a different slice of the window than every other row.\n"
    )
    n_broken = sum(1 for r in excluded if r["reason"] == "broken-feed")
    report.append(
        f"\nNote the direction of the bias: {n_broken} of the exclusions are for "
        "feed brokenness, and BAs with sloppy data operations plausibly also "
        "forecast sloppily — so excluding them likely removes BAs where "
        "GridPulse would win. The exclusion set is conservative against our own "
        "claim.\n"
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
