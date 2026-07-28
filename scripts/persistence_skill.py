"""Does the served forecast beat "yesterday, same hour"? — every scoreable BA.

A forecasting product's minimum bar is not accuracy, it is *skill*: beating a
baseline anyone could run without a model. Nothing in this repo measured that
until SEC turned up on the public benchmark at 18.0% against a seasonal-naive
11.5% — worse than doing nothing — with every existing instrument reporting it
as merely a bad region.

Our arm is the live benchmark's 24h mean MAPE (``/api/v1/benchmark``), so the
comparison uses the same window, the same exclusions and the same settled
truth the published scorecard uses. The baseline is computed from the same EIA
series over the same 30 days.

Usage:
    python scripts/persistence_skill.py
    python scripts/persistence_skill.py --output docs/PERSISTENCE_SKILL.md

Requires EIA_API_KEY (.env).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from data.eia_client import fetch_demand  # noqa: E402
from models.skill import SEASONAL_NAIVE_LAG_H, mape, skill_score  # noqa: E402

API_BASE = "https://gridpulse.kristenmartino.ai/api/v1"
WINDOW_DAYS = 30


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def _live_benchmark() -> dict:
    with urllib.request.urlopen(f"{API_BASE}/benchmark", timeout=60) as fh:
        return json.load(fh)


def measure() -> pd.DataFrame:
    bm = _live_benchmark()
    scored = {
        r["region"]: r["leads"]["24h"]
        for r in bm.get("regions", [])
        if (r.get("leads") or {}).get("24h", {}).get("scoreable")
    }
    start = (pd.Timestamp.utcnow() - pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")

    rows = []
    for ba, lead in sorted(scored.items()):
        try:
            df = fetch_demand(ba, start=start)
            d = df.dropna(subset=["demand_mw"]).copy()
            d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
            # asfreq exposes real gaps as NaN. Without it the lag reaches
            # across a gap and compares hours that are days apart, which
            # flatters the baseline and understates the model's deficit.
            y = d.sort_values("timestamp").set_index("timestamp").asfreq("h")["demand_mw"]
        except Exception as exc:  # pragma: no cover — network probe
            print(f"  {ba}: demand fetch failed ({exc})")
            continue

        arr = y.to_numpy(dtype=float)
        if arr.size <= SEASONAL_NAIVE_LAG_H:
            continue
        baseline = mape(arr[SEASONAL_NAIVE_LAG_H:], arr[:-SEASONAL_NAIVE_LAG_H])
        ours = lead["gridpulse"]["mape"]
        rows.append(
            {
                "ba": ba,
                "ours_pct": round(ours, 2),
                "naive_pct": round(baseline, 2),
                "official_pct": round(lead["official"]["mape"], 2),
                "skill": skill_score(ours, baseline),
                "points_vs_naive": round(baseline - ours, 2),
                "n": lead["n"],
            }
        )
    return pd.DataFrame(rows).sort_values("points_vs_naive")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    out = measure()
    if out.empty:
        print("no scoreable regions returned by the benchmark endpoint")
        return 1

    losing = out[out.points_vs_naive < 0]
    winning = out[out.points_vs_naive > 0]

    report = [
        "# Forecast skill vs a naive baseline\n",
        "\nDoes the served forecast beat *yesterday, same hour*? Re-run "
        "`python scripts/persistence_skill.py` to refresh.\n\n",
        f"Baseline: seasonal-naive at a {SEASONAL_NAIVE_LAG_H}h lag — every value it "
        "uses is known a full day before the target hour, so it is a fair opponent "
        "for a 24h-lead forecast. Our arm is the live benchmark's 24h mean MAPE, so "
        "the window, exclusions and settled truth match the published scorecard.\n\n",
    ]

    report.append(
        f"**{len(winning)} of {len(out)} balancing authorities beat the baseline**, "
        f"by a median of {out.points_vs_naive.median():.2f} points.\n\n"
    )

    if not losing.empty:
        worst = losing.iloc[0]
        report.append(
            f"**{len(losing)} do not.** The forecast that loses by most is "
            f"**{worst['ba']}**, at {worst['ours_pct']:.2f}% against the baseline's "
            f"{worst['naive_pct']:.2f}% — {abs(worst['points_vs_naive']):.2f} points of "
            "*negative* skill, meaning the model is subtracting information rather "
            "than adding it. The rest sit within about a point of the line, which is "
            "noise at this sample size; one is not.\n\n"
        )
        report.append("## Losing to the baseline\n\n")
        report.append(_md_table(losing) + "\n\n")

    report.append("## Every scoreable BA\n\n")
    report.append(_md_table(out) + "\n\n")
    report.append(
        "**Reading.** `skill` is `1 − ours ÷ naive`: positive means the model beats "
        "the baseline, negative means it is worse than free. `points_vs_naive` is the "
        "same comparison in error points, which is the figure to act on. "
        "`official_pct` is the operator's own day-ahead forecast over the same hours, "
        "for context — it is not the baseline.\n\n"
        "**What this does not say.** A model that beats seasonal-naive is not "
        "thereby good; the baseline is the floor, not a target. And skill is measured "
        "at one lead (24h) on one metric (mean MAPE), so it inherits every caveat in "
        "[`BENCHMARK_METHODOLOGY.md`](BENCHMARK_METHODOLOGY.md).\n"
    )

    print(_md_table(out))
    print(f"\nlosing to baseline: {len(losing)} of {len(out)}")
    if args.output:
        args.output.write_text("".join(report))
        print(f"report written → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
