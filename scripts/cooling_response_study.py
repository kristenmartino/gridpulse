"""Does the cooling-response feature pack actually help? (post-ERROR_ANALYSIS)

`docs/ERROR_ANALYSIS.md` sized the target: the hottest temperature quintile
carries a mean **34.7%** of our forecast error against **11.9%** for the
coldest, monotone in 7 of 8 BAs. That justifies an experiment. It does not
justify shipping anything, which is what this decides.

## Arms

Identical in every respect except `config.FEATURE_FLAGS`:

* **control** — `cooling_response_features` off. One linear
  ``cooling_degree_days`` against a fixed 65°F baseline.
* **treatment** — on. Adds CDD accumulation (24h/72h), CDD², the NWS heat
  index, and a CDD×humidity interaction.

## Verdict

Routed through `models/rolling_eval.py`, per `docs/EVALUATION_POLICY.md`:
optimise **WAPE** over 6 rolling windows, and a win is vetoed unless the
satisficing constraints hold (|bias| ≤ 2%, published MAPE not regressed by
more than 0.5 pts). `verdict()` may refuse to decide, which is a valid and
common outcome.

Both arms forecast **day-ahead** (`make_day_ahead_safe`) with known future
weather, so the comparison isolates the features rather than the horizon.

Usage:
    python -m scripts.cooling_response_study --regions MISO,ISONE --windows 6
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from scripts.arima_order_exog_study import ARCHIVE_LAG_DAYS, _archive_weather
from scripts.error_analysis import (
    HISTORY_DAYS,
    HOLDOUT_H,
    MIN_TRAIN_H,
    _eia_with_forecast,
    _fit_predict_xgb,
    make_day_ahead_safe,
)

#: The BAs the error analysis identified: 82% of the fleet's addressable gap,
#: and every one of them shows the hot-quintile concentration.
DEFAULT_REGIONS = ["MISO", "ERCOT", "ISONE", "NYISO", "PJM", "TVA", "FPL", "SOCO"]


def _features(df: pd.DataFrame, *, cooling: bool) -> pd.DataFrame:
    """Engineer with the flag forced, then make the frame day-ahead honest."""
    import config
    from data.feature_engineering import engineer_features

    before = config.FEATURE_FLAGS.get("cooling_response_features")
    config.FEATURE_FLAGS["cooling_response_features"] = cooling
    try:
        feats = engineer_features(df)
    finally:
        config.FEATURE_FLAGS["cooling_response_features"] = before
    return make_day_ahead_safe(feats).dropna(subset=["demand_mw"]).reset_index(drop=True)


def study_region(region: str, api_key: str, *, n_windows: int) -> dict[str, Any] | None:
    from config import REGION_COORDINATES
    from models.evaluation import compute_mape
    from models.rolling_eval import (
        bias_pct,
        paired_deltas,
        rolling_origin_splits,
        satisficing_check,
        verdict,
        wape,
    )

    coords = REGION_COORDINATES.get(region)
    if not coords:
        return None
    end = datetime.now(UTC) - timedelta(days=ARCHIVE_LAG_DAYS)
    start = end - timedelta(days=HISTORY_DAYS + (n_windows * HOLDOUT_H) // 24)

    demand = _eia_with_forecast(region, start, end, api_key)
    if demand.empty:
        return {"region": region, "skipped": "no EIA data"}
    weather = _archive_weather(coords["lat"], coords["lon"], start, end)
    raw = demand.merge(weather, on="timestamp", how="inner").sort_values("timestamp")

    arms = {"control": _features(raw, cooling=False), "treatment": _features(raw, cooling=True)}
    # Both arms must be scored on identical rows, or the paired test is not
    # paired. The flag adds columns, never drops rows, so this should hold —
    # asserted rather than assumed.
    if len(arms["control"]) != len(arms["treatment"]):
        return {"region": region, "skipped": "arms have different row counts"}

    splits = rolling_origin_splits(
        len(arms["control"]), n_windows=n_windows, holdout_h=HOLDOUT_H, min_train_h=MIN_TRAIN_H
    )
    if not splits:
        return {"region": region, "skipped": f"no windows from {len(arms['control'])} rows"}

    per_window: list[dict] = []
    for train_sl, test_sl in splits:
        w: dict[str, Any] = {}
        for arm, feats in arms.items():
            train, test = feats.iloc[train_sl], feats.iloc[test_sl]
            pred = _fit_predict_xgb(train, test)
            if pred is None:
                w[arm] = None
                continue
            actual = test["demand_mw"].to_numpy(dtype=float)
            w[arm] = {
                "wape": round(wape(actual, pred), 4),
                "mape": round(float(compute_mape(actual, pred)), 4),
                "bias_pct": round(bias_pct(actual, pred), 4),
            }
        if w.get("control") and w.get("treatment"):
            w["delta_wape"] = round(w["control"]["wape"] - w["treatment"]["wape"], 4)
        per_window.append(w)

    scored = [w for w in per_window if w.get("delta_wape") is not None]
    if not scored:
        return {"region": region, "skipped": "no window scored both arms"}

    deltas = paired_deltas(
        [w["control"]["wape"] for w in scored], [w["treatment"]["wape"] for w in scored]
    )
    v = verdict(deltas)

    def _mean(arm: str, key: str) -> float:
        return round(float(np.mean([w[arm][key] for w in scored])), 4)

    sat = satisficing_check(
        treatment_bias_pct=_mean("treatment", "bias_pct"),
        control_mape=_mean("control", "mape"),
        treatment_mape=_mean("treatment", "mape"),
    )
    return {
        "region": region,
        "n_windows_scored": len(scored),
        "verdict": v,
        "satisficing": sat,
        "ship": bool(v["decisive"] and v["winner"] == "treatment" and sat["passed"]),
        "control_mean": {k: _mean("control", k) for k in ("wape", "mape", "bias_pct")},
        "treatment_mean": {k: _mean("treatment", k) for k in ("wape", "mape", "bias_pct")},
        "windows": per_window,
    }


def main() -> int:
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default=",".join(DEFAULT_REGIONS))
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--out", default="docs/COOLING_RESPONSE_STUDY.json")
    args = ap.parse_args()

    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        print("EIA_API_KEY not set")
        return 2

    results = []
    for region in [r.strip() for r in args.regions.split(",") if r.strip()]:
        print(f"[{region}] ...", flush=True)
        try:
            res = study_region(region, api_key, n_windows=args.windows)
        except Exception as e:
            res = {"region": region, "error": str(e)[:200]}
        if not res:
            continue
        results.append(res)
        if "verdict" in res:
            print(
                f"    wape {res['control_mean']['wape']:.3f} -> "
                f"{res['treatment_mean']['wape']:.3f}  ship={res['ship']}  "
                f"{res['verdict']['reason']}",
                flush=True,
            )
        else:
            print(f"    {res.get('skipped') or res.get('error')}", flush=True)
        with open(args.out, "w") as fh:
            json.dump({"regions": results}, fh, indent=2)

    scored = [r for r in results if "verdict" in r]
    print("\n== SUMMARY ==")
    print(
        json.dumps(
            {
                "n_regions": len(scored),
                "n_ship": sum(1 for r in scored if r["ship"]),
                "n_decisive_treatment": sum(
                    1
                    for r in scored
                    if r["verdict"]["decisive"] and r["verdict"]["winner"] == "treatment"
                ),
                "n_decisive_control": sum(
                    1
                    for r in scored
                    if r["verdict"]["decisive"] and r["verdict"]["winner"] == "control"
                ),
                "n_inconclusive": sum(1 for r in scored if not r["verdict"]["decisive"]),
                "n_vetoed": sum(
                    1
                    for r in scored
                    if r["verdict"]["decisive"]
                    and r["verdict"]["winner"] == "treatment"
                    and not r["satisficing"]["passed"]
                ),
                "mean_delta_wape_pts": round(
                    float(np.mean([r["verdict"]["mean"] for r in scored])), 4
                )
                if scored
                else None,
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
