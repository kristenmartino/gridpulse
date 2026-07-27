"""Provenance probe for the public forecast benchmark (E0 gate).

The benchmark engine (PR #340) shipped deliberately unpublished, because
design review found two claims the code could not support. This probe
answers both, reproducibly, so the methodology can cite measurements
rather than assumptions.

## Gate 1 — does EIA revise the day-ahead forecast after we bank it?

The vintage window only admits an hour once EIA publishes a metered ``D``,
so ``first_seen_df`` is the day-ahead value *re-read 0–3h after the target
hour*, not at day-ahead time. If EIA revises DF in between, the benchmark
could be scoring a value that carries hindsight — or, conversely, a stale
one a reader would call unfair.

Measured by comparing every banked ``first_seen_df`` against EIA's
*current* DF for the same hour, and then scoring the official arm **both
ways** against settled truth. If the two scorings agree, the choice does
not matter and the benchmark is robust to it.

**What this probe cannot see:** a revision that happened *before* our
first capture. Detecting that requires capturing DF for hours that have no
``D`` yet — a separate instrument, not built here. So the honest phrasing
downstream is always *"the earliest day-ahead forecast we observed,"*
never *"their day-ahead forecast."*

## Gate 2 — what lead do we actually forecast at?

``_resolve_forecast_start`` anchors row 0 at the last *real* demand hour,
so with EIA's publishing lag a "24h" record is not 24h ahead. Measured
from live forecast payloads (``scored_at`` vs each row's timestamp).

This also tests, rather than assumes, the claim that our nominal-48h arm
gives the operators more lead than their own documented maximum (41h) —
the basis for publishing it as the conservative comparison.

Usage:
    python scripts/benchmark_provenance_probe.py
    python scripts/benchmark_provenance_probe.py --output docs/BENCHMARK_PROVENANCE.md

Requires EIA_API_KEY, GCS_ENABLED=true, GCS_BUCKET_NAME, and ADC.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import config  # noqa: E402
from data.eia_client import _get_eia_code  # noqa: E402
from data.vintage import deserialize_records  # noqa: E402

# Imported, never re-declared: a second literal here could drift from the
# engine's and quietly grade the conservative label against a different bar.
from models.benchmark import OFFICIAL_DOCUMENTED_LEAD_H, STUB_EPSILON_MW  # noqa: E402

EIA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
API_BASE = "https://gridpulse.kristenmartino.ai/api/v1"

#: A banked and a current DF value are "the same" within this many MW.
DF_REVISION_EPSILON_MW = 0.5

#: Revision measurement needs a spread of feed behaviours, not the whole
#: fleet — each BA costs one EIA call over a 30-day window.
REVISION_SAMPLE = (
    "PJM",
    "MISO",
    "ERCOT",
    "CAISO",
    "SOCO",
    "PSEI",
    "FMPP",
    "GVL",
    "SPP",
    "NYISO",
)
#: Lead is read from live payloads (cheap), so sample more widely.
LEAD_SAMPLE = REVISION_SAMPLE + ("ISONE", "BPAT", "AVA", "PGE", "DUK")


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def _current_df(region: str, start: str, end: str) -> dict[str, float]:
    """EIA's CURRENT day-ahead forecast per hour — i.e. post-revision."""
    params = {
        "api_key": os.environ["EIA_API_KEY"],
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": _get_eia_code(region),
        "facets[type][]": "DF",
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }
    resp = requests.get(EIA_URL, params=params, timeout=120)
    resp.raise_for_status()
    return {
        row["period"][:13]: float(row["value"])
        for row in resp.json()["response"]["data"]
        if row["value"] is not None
    }


def probe_df_revision() -> pd.DataFrame:
    """Per BA: how often EIA revises DF, and whether it changes the verdict."""
    from data.gcs_store import read_parquet

    rows = []
    for region in REVISION_SAMPLE:
        mirror = read_parquet("vintage", region)
        if mirror is None or mirror.empty:
            continue
        records = deserialize_records(mirror.to_dict("records"))
        if not records:
            continue

        hours = [str(r.timestamp)[:13] for r in records]
        current = _current_df(region, min(hours), max(hours))

        n_compared = n_revised = 0
        max_delta_pct = 0.0
        banked_errs: list[float] = []
        revised_errs: list[float] = []

        for r in records:
            key = str(r.timestamp)[:13]
            if not np.isfinite(r.first_seen_df) or key not in current:
                continue
            n_compared += 1
            delta = abs(r.first_seen_df - current[key])
            if delta >= DF_REVISION_EPSILON_MW:
                n_revised += 1
                max_delta_pct = max(max_delta_pct, delta / max(abs(current[key]), 1.0) * 100.0)

            # Score both ways under the benchmark's own exclusions, so the
            # numbers are the ones a reader reproduces from the engine.
            if not (np.isfinite(r.last_d) and r.last_d > 0):
                continue
            if r.was_placeholder or abs(r.last_d - r.first_seen_df) < STUB_EPSILON_MW:
                continue
            banked_errs.append(abs(r.first_seen_df - r.last_d) / r.last_d * 100.0)
            revised_errs.append(abs(current[key] - r.last_d) / r.last_d * 100.0)

        if n_compared < 50 or len(banked_errs) < 50:
            continue
        as_banked = float(np.median(banked_errs))
        as_revised = float(np.median(revised_errs))
        rows.append(
            {
                "ba": region,
                "n_compared": n_compared,
                "revised_pct": round(n_revised / n_compared * 100, 1),
                "max_revision_pct": round(max_delta_pct, 2),
                "official_as_issued_pct": round(as_banked, 2),
                "official_as_revised_pct": round(as_revised, 2),
                "median_ape_shift_pts": round(as_banked - as_revised, 2),
            }
        )
    return pd.DataFrame(rows)


def probe_realized_lead() -> pd.DataFrame:
    """Per BA: the lead our nominal 24h / 48h records actually carry."""
    rows = []
    for region in LEAD_SAMPLE:
        try:
            with urllib.request.urlopen(
                f"{API_BASE}/forecast/{region}?horizon=48", timeout=30
            ) as fh:
                payload = json.load(fh)
            scored_at = pd.Timestamp(payload["scored_at"])
            forecast = payload["forecast"]
            # Measured from row 0 + H, exactly how
            # ``models.drift.snapshot_horizon_predictions`` picks the hour the
            # benchmark later grades. The first cut read ``rows[H-1]``, which
            # is the hour BEFORE that one — understating every lead by 1h.
            origin = pd.Timestamp(forecast[0]["timestamp"])
            lead_24 = (origin + pd.Timedelta(hours=24) - scored_at).total_seconds() / 3600
            lead_48 = (origin + pd.Timedelta(hours=48) - scored_at).total_seconds() / 3600
        except Exception as exc:  # pragma: no cover — network probe
            print(f"  {region}: lead probe failed ({exc})")
            continue
        rows.append(
            {
                "ba": region,
                "nominal_24h_realized_h": round(lead_24, 2),
                "nominal_48h_realized_h": round(lead_48, 2),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    if "EIA_API_KEY" not in os.environ:
        print("EIA_API_KEY required (.env)")
        return 1
    if not config.GCS_ENABLED or not config.GCS_BUCKET_NAME:
        print("GCS access required: GCS_ENABLED=true, GCS_BUCKET_NAME, ADC login.")
        return 1

    report: list[str] = [
        "# Benchmark provenance — what we actually measured\n",
        "\nThe two questions the benchmark engine could not answer about "
        "itself, measured. Re-run "
        "`python scripts/benchmark_provenance_probe.py` to refresh.\n",
    ]

    print("== gate 1: DF revision ==")
    revision = probe_df_revision()
    print(_md_table(revision))
    report.append("\n## Gate 1 — does EIA revise the day-ahead forecast?\n\n")
    report.append(
        "`first_seen_df` is the day-ahead value re-read 0–3h *after* the "
        "target hour (the vintage window only admits an hour once EIA "
        "publishes a metered `D`). If EIA revises DF in between, the scoring "
        "choice matters. Below: how often it revises, and the official arm "
        "scored **both ways** against settled truth, under the benchmark's "
        "own exclusions.\n\n"
    )
    report.append(_md_table(revision) + "\n")

    if not revision.empty:
        worst = revision.loc[revision["median_ape_shift_pts"].abs().idxmax()]
        n_zero = int((revision["revised_pct"] == 0).sum())
        report.append(
            f"\n**Reading.** Revision is real but uneven — {n_zero} of "
            f"{len(revision)} sampled BAs never revise at all. The largest "
            f"movement in any operator's own **median APE** is "
            f"**{abs(worst['median_ape_shift_pts']):.2f} points** "
            f"({worst['ba']}: {worst['official_as_issued_pct']:.2f}% "
            f"as-issued vs {worst['official_as_revised_pct']:.2f}% "
            "as-revised).\n\n**What this does NOT establish.** Both columns "
            "are medians, and the benchmark decides every verdict on *mean* "
            "MAPE — which this probe does not measure. A fat-tailed feed can "
            "move a mean far more than a median, so nothing here bounds a "
            "head-to-head result. Whether a revision changes a verdict is "
            "decided per BA and published as `winner_vs_revised` beside "
            "`winner`. That is why the benchmark publishes **both** official "
            "arms — as-issued as the fair comparison, as-revised as the "
            "conservative one, since a forecast revised after the target hour "
            "carries hindsight — rather than asserting the choice is "
            "immaterial.\n"
        )

    report.append(
        "\n**Limit of this probe.** It cannot see a revision that happened "
        "*before* our first capture. That would need DF captured for hours "
        "with no `D` yet — a separate instrument, not built. So the phrasing "
        "everywhere is *the earliest day-ahead forecast we observed*, never "
        "*their day-ahead forecast*.\n"
    )

    print("\n== gate 2: realized lead ==")
    lead = probe_realized_lead()
    print(_md_table(lead))
    report.append("\n## Gate 2 — what lead do our forecasts actually carry?\n\n")
    report.append(
        "The forecast anchors on the last *real* demand hour, so EIA's "
        "publishing lag makes a nominal 24h record shorter than 24h. "
        "Measured from live payloads (`scored_at` vs row timestamp).\n\n"
    )
    report.append(_md_table(lead) + "\n")

    if not lead.empty:
        lo24 = lead["nominal_24h_realized_h"].min()
        hi24 = lead["nominal_24h_realized_h"].max()
        lo48 = lead["nominal_48h_realized_h"].min()
        conservative_ok = lo48 > OFFICIAL_DOCUMENTED_LEAD_H[1]
        report.append(
            f"\n**Reading.** A nominal-24h record is a realized "
            f"**{lo24:.2f}–{hi24:.2f}h** lead — shorter than its label, and "
            "sitting *inside* the operators' documented "
            f"{OFFICIAL_DOCUMENTED_LEAD_H[0]:.0f}–"
            f"{OFFICIAL_DOCUMENTED_LEAD_H[1]:.0f}h day-ahead window rather "
            "than beyond it, so on a typical hour they had at least as much "
            "lead as we did. No *N hours ahead* claim should be published "
            "without this caveat.\n\n"
            f"The nominal-48h arm carries a minimum realized "
            f"**{lo48:.2f}h** — "
            + (
                f"which exceeds their documented maximum of "
                f"{OFFICIAL_DOCUMENTED_LEAD_H[1]:.0f}h, so publishing it as the "
                "*conservative* comparison is supported by measurement rather "
                "than assumed.\n"
                if conservative_ok
                else f"which does **not** exceed their documented maximum of "
                f"{OFFICIAL_DOCUMENTED_LEAD_H[1]:.0f}h — the conservative label "
                "must be withheld.\n"
            )
        )
        print(f"\nconservative-lead claim supported: {conservative_ok}")

    if args.output:
        args.output.write_text("".join(report))
        print(f"\nreport written → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
