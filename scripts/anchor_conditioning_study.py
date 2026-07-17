"""Anchor-conditioning study — should the forecast anchor on DF instead of D? (#309)

The go/no-go gate for anchor-redesign PR C, in the mold of
``phase0_weather_normal_backtest.py`` (a binary verdict, segmented tallies)
wearing ``export_holdout_metrics.py``'s GCS-read + markdown body.

## The question, per revision class

Production seeds ``demand_lag_1h`` (and SARIMAX's Kalman origin) from EIA's
newest reading. The vintage instrument (#312) recorded, per hour:

* ``first_seen_d``   — what production actually anchored on
* ``first_seen_df``  — the BA's own day-ahead forecast (the substitution
  candidate; already on every frame, excluded from features since ever)
* ``last_d``         — EIA's settled truth

So the replay #309 called impossible reduces to arithmetic: for every
fresh-captured hour, which candidate was closer to settled truth?

## Tiers

**Tier 1 (default)** — the anchor-error proxy, every mirrored BA, every
fresh-captured hour: |candidate − settled| / settled for the three arms
(production / DF-substitution / oracle bound), win-loss + magnitudes
segmented by ``revision_class`` (recomputed locally via ``classify_region``
— self-contained, no Redis). ``was_placeholder`` hours are ties by
construction (the stub IS the DF) and are counted separately so neither arm
claims them.

**Tier 2 (--tier2)** — end-to-end model replay on representative BAs: prod
pickles via ``load_model`` (same model both arms — differences isolate to
the anchor), the real ``engineer_features`` + ``recursive_autoregressive_
forecast`` path, two histories (as-seen vs conditioned at the anchor hour),
scored 1–24h forward against settled. Documented approximation: the tick-T
view is reconstructed as settled history + the first-seen value at the
anchor hour — exact for the dominant ``demand_lag_1h`` mechanism.

## Verdict rule (per class)

CONDITION when the DF arm beats production on BOTH win-rate (>60% of
non-placeholder hours) AND mean error (by ≥2 points) for that class;
SKIP otherwise. ``clean`` is reported but never conditioned (policy);
``bulk`` is expected to fail its gate (PSCO runs 118-121% of its own DF) —
if it *passes*, that is a finding worth a second look, not an auto-ship.

Usage:
    python scripts/anchor_conditioning_study.py                # tier 1, all BAs
    python scripts/anchor_conditioning_study.py --tier2        # + model replay
    python scripts/anchor_conditioning_study.py --output docs/ANCHOR_CONDITIONING_STUDY.md

Requires GCS_ENABLED=true, GCS_BUCKET_NAME, and ADC credentials
(``gcloud auth application-default login``) — the export_holdout_metrics
pattern. Reads the vintage mirror written by anchor-redesign PR A.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from data.vintage import (  # noqa: E402
    VintageRecord,
    classify_region,
    deserialize_records,
)

#: Tier-2 representative BAs — one per measured class + the counterexample.
TIER2_REGIONS = ("LDWP", "IID", "BPAT", "PSCO", "PNM")
#: Verdict thresholds (see module docstring).
WIN_RATE_GATE = 0.60
MEAN_MARGIN_GATE_PCT = 2.0


def _md_table(df: pd.DataFrame) -> str:
    """Minimal markdown table — the export_holdout_metrics precedent
    (pandas.to_markdown needs tabulate, which is not a dependency)."""
    if df.empty:
        return "(no rows)"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def _load_vintage(region: str) -> list[VintageRecord]:
    """Read one region's mirrored vintage window from GCS."""
    from data.gcs_store import read_parquet

    df = read_parquet("vintage", region)
    if df is None or df.empty:
        return []
    return deserialize_records(df.to_dict("records"))


def _fresh_records(records: list[VintageRecord]) -> list[VintageRecord]:
    """Fresh-captured records (capture lag ≤ 3h) with usable truth + DF."""
    from data.vintage import FRESH_CAPTURE_LAG_HOURS, _capture_lag_hours

    out = []
    for r in records:
        lag = _capture_lag_hours(r)
        if lag is None or lag > FRESH_CAPTURE_LAG_HOURS:
            continue
        if not (np.isfinite(r.last_d) and r.last_d > 0):
            continue
        out.append(r)
    return out


def tier1(regions: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    """The anchor-error proxy. Returns (per-class table, per-class verdicts)."""
    by_class: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"prod": [], "df": [], "oracle": [], "ties": []}
    )
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "placeholder": 0, "df_missing": 0, "df_wins": 0}
    )

    for region in regions:
        records = _load_vintage(region)
        if not records:
            continue
        cls = classify_region(records)["revision_class"]
        for r in _fresh_records(records):
            c = counts[cls]
            if not np.isfinite(r.first_seen_df):
                c["df_missing"] += 1
                continue
            prod_err = abs(r.first_seen_d - r.last_d) / r.last_d * 100.0
            df_err = abs(r.first_seen_df - r.last_d) / r.last_d * 100.0
            c["n"] += 1
            if r.was_placeholder:
                # The stub IS the DF — a tie by construction; neither arm
                # may claim it.
                c["placeholder"] += 1
                by_class[cls]["ties"].append(prod_err)
                continue
            by_class[cls]["prod"].append(prod_err)
            by_class[cls]["df"].append(df_err)
            by_class[cls]["oracle"].append(0.0)
            if df_err < prod_err:
                c["df_wins"] += 1

    rows = []
    verdicts: dict[str, str] = {}
    for cls in ("broken", "churn", "bulk", "clean", "unknown"):
        prod = by_class[cls]["prod"]
        dfe = by_class[cls]["df"]
        c = counts[cls]
        contested = len(prod)
        if contested == 0:
            verdicts[cls] = "INSUFFICIENT DATA"
            rows.append({"class": cls, "n_hours": c["n"], "verdict": verdicts[cls]})
            continue
        win_rate = c["df_wins"] / contested
        margin = float(np.mean(prod)) - float(np.mean(dfe))
        passes = win_rate > WIN_RATE_GATE and margin >= MEAN_MARGIN_GATE_PCT
        if cls in ("clean", "unknown"):
            verdicts[cls] = "NEVER CONDITIONED (policy)"
        elif cls == "bulk" and passes:
            verdicts[cls] = "PASSES GATE — unexpected, review before shipping"
        else:
            verdicts[cls] = "CONDITION" if passes else "SKIP"
        rows.append(
            {
                "class": cls,
                "n_hours": c["n"],
                "placeholder_ties": c["placeholder"],
                "prod_mean_err_pct": round(float(np.mean(prod)), 2),
                "df_mean_err_pct": round(float(np.mean(dfe)), 2),
                "prod_median_err_pct": round(float(np.median(prod)), 2),
                "df_median_err_pct": round(float(np.median(dfe)), 2),
                "df_win_rate": round(win_rate, 3),
                "mean_margin_pct": round(margin, 2),
                "verdict": verdicts[cls],
            }
        )
    return pd.DataFrame(rows), verdicts


def tier2(regions: tuple[str, ...], horizon: int = 24) -> pd.DataFrame:
    """End-to-end model replay on representative BAs.

    For each fresh-captured hour H with ≥``horizon`` settled hours after it:
    build the settled history through H, swap H's value per arm
    (as-seen ``first_seen_d`` vs conditioned ``first_seen_df``), engineer
    features, run the production recursive path with the region's prod
    XGBoost pickle, and score 1..horizon ahead against settled truth.
    """
    from data.feature_engineering import (
        engineer_features,
        recursive_autoregressive_forecast,
    )
    from data.preprocessing import merge_demand_weather
    from data.weather_client import fetch_weather
    from jobs.phases import _build_future_feature_frame
    from models.persistence import load_model
    from models.xgboost_model import predict_xgboost

    rows = []
    for region in regions:
        records = _load_vintage(region)
        fresh = _fresh_records(records)
        if not fresh:
            continue
        loaded = load_model(region, "xgboost")
        if loaded is None:
            print(f"  {region}: no prod pickle — skipped")
            continue
        model, _meta = loaded
        settled = {r.timestamp: r.last_d for r in records}
        weather_df = fetch_weather(region)

        cls = classify_region(records)["revision_class"]
        arm_errors: dict[str, list[float]] = {"as_seen": [], "conditioned": []}
        hours = sorted(settled)
        for r in fresh:
            if not np.isfinite(r.first_seen_df):
                continue
            idx = hours.index(r.timestamp) if r.timestamp in hours else -1
            if idx < 48 or idx + horizon >= len(hours):
                continue
            history_hours = hours[: idx + 1]
            truth = [settled[h] for h in hours[idx + 1 : idx + 1 + horizon]]
            for arm, anchor_value in (
                ("as_seen", r.first_seen_d),
                ("conditioned", r.first_seen_df),
            ):
                demand = pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(history_hours, utc=True),
                        "demand_mw": [settled[h] for h in history_hours[:-1]] + [anchor_value],
                        "region": region,
                    }
                )
                merged = merge_demand_weather(demand, weather_df)
                featured = engineer_features(merged).dropna(subset=["demand_mw"])
                if len(featured) < 48:
                    continue
                future = _build_future_feature_frame(featured, horizon, weather_df=weather_df)
                preds = recursive_autoregressive_forecast(
                    model,
                    featured["demand_mw"].tolist(),
                    future.iloc[:horizon],
                    lambda m, x: predict_xgboost(m, x),
                )
                n = min(len(preds), len(truth))
                errs = [
                    abs(preds[i] - truth[i]) / truth[i] * 100.0 for i in range(n) if truth[i] > 0
                ]
                if errs:
                    arm_errors[arm].append(float(np.mean(errs)))
        if arm_errors["as_seen"] and arm_errors["conditioned"]:
            rows.append(
                {
                    "region": region,
                    "class": cls,
                    "n_replays": len(arm_errors["as_seen"]),
                    "as_seen_mape": round(float(np.mean(arm_errors["as_seen"])), 2),
                    "conditioned_mape": round(float(np.mean(arm_errors["conditioned"])), 2),
                    "delta": round(
                        float(np.mean(arm_errors["as_seen"]))
                        - float(np.mean(arm_errors["conditioned"])),
                        2,
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tier2", action="store_true", help="run the model replay too")
    ap.add_argument("--output", type=Path, default=None, help="write markdown report")
    args = ap.parse_args()

    if not config.GCS_ENABLED or not config.GCS_BUCKET_NAME:
        print(
            "GCS access required: set GCS_ENABLED=true and GCS_BUCKET_NAME, and\n"
            "authenticate via `gcloud auth application-default login`."
        )
        return 1

    regions = list(config.REGION_COORDINATES)
    print(f"Tier 1 — anchor-error proxy over {len(regions)} BAs' vintage mirrors…")
    table, verdicts = tier1(regions)
    print(_md_table(table))
    print()
    for cls, v in verdicts.items():
        print(f"  {cls:8s} → {v}")

    report = ["# Anchor-conditioning study (#309)\n", "## Tier 1 — anchor-error proxy\n"]
    report.append(_md_table(table))
    report.append("\n\n## Verdicts\n")
    report.extend(f"- **{cls}** → {v}\n" for cls, v in verdicts.items())

    if args.tier2:
        print(f"\nTier 2 — model replay on {TIER2_REGIONS}…")
        t2 = tier2(TIER2_REGIONS)
        print(_md_table(t2))
        report.append("\n## Tier 2 — end-to-end model replay\n")
        report.append(_md_table(t2))

    if args.output:
        args.output.write_text("".join(report))
        print(f"\nreport written → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
