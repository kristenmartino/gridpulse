"""Bottom-up (11 zones) vs top-down (one BA) forecasting — NYISO (#375 follow-up).

`NYISO_ZONAL_PROBE.md` found that zonal weather diversity predicts our BA-level
error and survives a temperature control (+0.51 pts in the cool band, +0.80 in
the hot, +0.06 in the mid — the mid showing nothing is what made it credible).
It recommended exactly one next experiment, and this is it:

> Bottom-up (11 zonal models with zonal weather, summed) vs top-down (one
> BA-level model), on NYISO alone, through the existing rolling-eval harness.
> If zonal weather diversity really costs us, bottom-up should recover the
> 0.5–0.8 pts the control isolated.

## The target, and why it is not EIA's demand

The obvious target is EIA-930's NYISO `D`, since that is what the product
forecasts. Measured first: **the sum of NYISO's own 11 zones differs from EIA's
`D` by 2.70% WAPE hour-by-hour** (means agree — ratio 1.0003 — but the hourly
ratio ranges 0.94 to 1.07). Our top-down error on NYISO is ~4–5%, so that
definitional gap is more than half the error budget.

Scoring bottom-up against `D` would therefore charge it a ~2.7% floor it has
no way to control, and the experiment would measure a data-definition mismatch
rather than the forecasting question. **Both arms target the zone sum**, which
is internally consistent and isolates bottom-up vs top-down.

*(The gap is a real finding in its own right: adopting NYISO zonal data in
production would change what is being forecast, and the published benchmark —
scored against EIA settled `D` — would move for definitional reasons rather
than accuracy ones.)*

## Arms

Same windows, same scored hours, same target, both day-ahead honest.

* **top-down (control)** — one model on the total, BA-level weather. What
  production does.
* **bottom-up (treatment)** — 11 models, one per zone, each trained on that
  zone's load with **that zone's own weather**; predictions summed. Giving the
  treatment zonal weather is the point: with BA-level weather it would be a
  strawman, the mistake the #230 study made by sampling 14 horizons.

## Verdict

`models/rolling_eval.py` per `docs/EVALUATION_POLICY.md` — optimise WAPE,
satisficing constraints veto a win, `verdict()` may refuse.

Usage:
    python -m scripts.nyiso_bottom_up_study --months 4 --windows 6
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from scripts.arima_order_exog_study import ARCHIVE_LAG_DAYS, _archive_weather
from scripts.error_analysis import MIN_TRAIN_H, _fit_predict_xgb, make_day_ahead_safe
from scripts.nyiso_zonal_probe import ZONE_COORDS, fetch_zonal_load

HOLDOUT_H = 168

#: The BA-level weather point the top-down arm uses — production's own NYISO
#: coordinate, so the control is what production actually runs.
from config import REGION_COORDINATES  # noqa: E402

BA_COORDS = REGION_COORDINATES["NYISO"]


def _featurise(load: pd.Series, weather: pd.DataFrame) -> pd.DataFrame:
    """Production feature engineering for one load series + its weather."""
    from data.feature_engineering import engineer_features

    df = (
        load.rename("demand_mw")
        .to_frame()
        .reset_index()
        .rename(columns={"index": "timestamp"})
        .merge(weather, on="timestamp", how="inner")
        .sort_values("timestamp")
    )
    feats = engineer_features(df)
    return make_day_ahead_safe(feats).dropna(subset=["demand_mw"]).reset_index(drop=True)


def study(months: int, windows: int) -> dict[str, Any]:
    from models.evaluation import compute_mape
    from models.rolling_eval import (
        bias_pct,
        paired_deltas,
        rolling_origin_splits,
        satisficing_check,
        verdict,
        wape,
    )

    end = datetime.now(UTC) - timedelta(days=ARCHIVE_LAG_DAYS)
    start = end - timedelta(days=months * 31)

    print("  fetching zonal load ...", flush=True)
    zl = fetch_zonal_load(months)
    if zl.empty:
        return {"skipped": "no zonal load"}
    zones = [z for z in ZONE_COORDS if z in zl.columns]
    total = zl[zones].sum(axis=1)

    print("  fetching weather (1 BA point + 11 zone points) ...", flush=True)
    ba_weather = _archive_weather(BA_COORDS["lat"], BA_COORDS["lon"], start, end)
    zone_weather = {z: _archive_weather(*ZONE_COORDS[z], start, end) for z in zones}

    # Feature frames: one for the total (BA weather), one per zone (own weather).
    top = _featurise(total, ba_weather)
    per_zone = {z: _featurise(zl[z], zone_weather[z]) for z in zones}
    # ABLATION arm: zonal LOAD decomposition but BA-level weather for every
    # zone. Bottom-up gets two new things at once — 11 load histories and 11
    # weather points — and the probe's hypothesis was specifically about
    # weather diversity. Without this arm a win could not be attributed, and
    # the recommendation differs: zonal weather means 12 archive calls per BA,
    # zonal load alone is just the ISO feed.
    per_zone_ba_wx = {z: _featurise(zl[z], ba_weather) for z in zones}

    # Every arm must be scored on identical rows, so align on the shortest.
    n = min(
        [len(top)] + [len(f) for f in per_zone.values()] + [len(f) for f in per_zone_ba_wx.values()]
    )
    top = top.iloc[-n:].reset_index(drop=True)
    per_zone = {z: f.iloc[-n:].reset_index(drop=True) for z, f in per_zone.items()}
    per_zone_ba_wx = {z: f.iloc[-n:].reset_index(drop=True) for z, f in per_zone_ba_wx.items()}
    print(f"  aligned rows: {n}   zones: {len(zones)}", flush=True)

    splits = rolling_origin_splits(
        n, n_windows=windows, holdout_h=HOLDOUT_H, min_train_h=MIN_TRAIN_H
    )
    if not splits:
        return {"skipped": f"no windows from {n} rows"}

    per_window: list[dict] = []
    for w_i, (train_sl, test_sl) in enumerate(splits):
        actual = top["demand_mw"].to_numpy(dtype=float)[test_sl]
        w: dict[str, Any] = {"window": w_i}

        td = _fit_predict_xgb(top.iloc[train_sl], top.iloc[test_sl])

        def _sum_zones(
            frames: dict, *, n=len(actual), tr=train_sl, te=test_sl
        ) -> np.ndarray | None:
            # Loop variables bound as defaults: the closure is invoked inside
            # this iteration, so late binding is harmless today, but a later
            # edit that defers the call would silently score the wrong window.
            out = np.zeros(n, dtype=float)
            for z in zones:
                f = frames[z]
                p = _fit_predict_xgb(f.iloc[tr], f.iloc[te])
                if p is None:
                    return None
                out += p
            return out

        bu = _sum_zones(per_zone)
        bu_bawx = _sum_zones(per_zone_ba_wx)

        for name, pred in (("top_down", td), ("bottom_up", bu), ("bottom_up_ba_weather", bu_bawx)):
            if pred is None or not np.all(np.isfinite(pred)):
                w[name] = None
                continue
            w[name] = {
                "wape": round(wape(actual, pred), 4),
                "mape": round(float(compute_mape(actual, pred)), 4),
                "bias_pct": round(bias_pct(actual, pred), 4),
            }
        if w.get("top_down") and w.get("bottom_up"):
            w["delta_wape"] = round(w["top_down"]["wape"] - w["bottom_up"]["wape"], 4)
        if w.get("top_down") and w.get("bottom_up_ba_weather"):
            # How much of the win survives WITHOUT zonal weather?
            w["delta_wape_ba_weather"] = round(
                w["top_down"]["wape"] - w["bottom_up_ba_weather"]["wape"], 4
            )
        per_window.append(w)
        print(
            f"    window {w_i}: {json.dumps({k: v for k, v in w.items() if k != 'window'})[:150]}",
            flush=True,
        )

    scored = [w for w in per_window if w.get("delta_wape") is not None]
    if not scored:
        return {"skipped": "no window scored both arms", "windows": per_window}

    v = verdict(
        paired_deltas(
            [w["top_down"]["wape"] for w in scored], [w["bottom_up"]["wape"] for w in scored]
        )
    )

    def _mean(arm: str, key: str) -> float:
        return round(float(np.mean([w[arm][key] for w in scored])), 4)

    def _ablation(rows: list[dict]) -> dict:
        """Attribution: zonal LOAD decomposition alone, BA weather for all zones."""
        ok = [w for w in rows if w.get("delta_wape_ba_weather") is not None]
        if not ok:
            return {"skipped": "no window scored the ablation arm"}
        full = float(np.mean([w["delta_wape"] for w in ok]))
        partial = float(np.mean([w["delta_wape_ba_weather"] for w in ok]))
        return {
            "n_windows": len(ok),
            "wape": round(float(np.mean([w["bottom_up_ba_weather"]["wape"] for w in ok])), 4),
            "gain_vs_topdown_pts": round(partial, 4),
            "full_gain_pts": round(full, 4),
            # >0.5 means most of the benefit is load decomposition, not weather.
            "share_of_gain_from_load_decomposition": (round(partial / full, 3) if full else None),
            "verdict": verdict(
                paired_deltas(
                    [w["top_down"]["wape"] for w in ok],
                    [w["bottom_up_ba_weather"]["wape"] for w in ok],
                )
            ),
        }

    sat = satisficing_check(
        treatment_bias_pct=_mean("bottom_up", "bias_pct"),
        control_mape=_mean("top_down", "mape"),
        treatment_mape=_mean("bottom_up", "mape"),
    )
    return {
        "target": "sum of NYISO zones (not EIA D — see module docstring)",
        "n_zones": len(zones),
        "n_windows_scored": len(scored),
        "verdict": v,
        "mde_pts": round(2 * v["stderr"], 4) if v.get("stderr") else None,
        "detectable": (
            bool(abs(v["mean"]) >= 2 * v["stderr"]) if v.get("stderr") and v.get("mean") else None
        ),
        "satisficing": sat,
        "ship": bool(v["decisive"] and v["winner"] == "treatment" and sat["passed"]),
        "top_down_mean": {k: _mean("top_down", k) for k in ("wape", "mape", "bias_pct")},
        "bottom_up_mean": {k: _mean("bottom_up", k) for k in ("wape", "mape", "bias_pct")},
        "ablation_bottom_up_ba_weather": _ablation(scored),
        "windows": per_window,
    }


def main() -> int:
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=4)
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--out", default="docs/NYISO_BOTTOM_UP_STUDY.json")
    args = ap.parse_args()

    if not os.environ.get("EIA_API_KEY"):
        print("EIA_API_KEY not set")
        return 2

    res = study(args.months, args.windows)
    print("\n== RESULT ==")
    print(json.dumps({k: v for k, v in res.items() if k != "windows"}, indent=2))
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
