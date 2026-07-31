"""Direct multi-horizon vs recursive forecasting (#230).

#230's evidence: after #195/#209 made the holdout honest (recursive), the
168h-ahead MAPE roughly doubled — median best-base 2.30% → 4.12%, XGBoost
2.32% → 4.32%. That gap is **error accumulation**: recursive forecasting feeds
the model's own predictions forward across 168 steps, so a small step-1 error
compounds into a large step-168 one.

The issue asks for a prototype on 3–4 BAs measured against the recursive
baseline before any rewrite is committed. This is that prototype.

A separate reason to run it: `docs/ERROR_ANALYSIS.md` reconstructed a *direct*
day-ahead model that beat production's live numbers on several BAs (PJM 2.72
vs 4.10, MISO 2.85 vs 3.41). That was confounded by perfect weather, so it
could not settle anything — but it was large enough that dismissing #230 on
"model work is not where the error is" was not defensible.

## Arms

Identical training data, identical features, identical weather. Only the
horizon strategy differs.

* **recursive (control)** — production's own
  `data.feature_engineering.recursive_autoregressive_forecast`, the single
  source of truth for both production scoring and holdout evaluation. Chains
  its own predictions forward, recomputing autoregressive features each step
  from the growing prediction history.
* **direct (treatment)** — one model trained on
  ``(features known at origin, horizon h) -> demand at origin+h``, with the
  autoregressive block frozen at the origin and ``horizon_h`` as a feature.
  Nothing is fed forward, so nothing compounds.

Both arms are seeded from the same real history and see the same future
weather, so the comparison isolates the strategy rather than the inputs.

## Verdict

Through `models/rolling_eval.py` per `docs/EVALUATION_POLICY.md` — optimise
WAPE, satisficing constraints veto a win, and `verdict()` may refuse.

Usage:
    python -m scripts.direct_multihorizon_study --regions PJM,MISO,ERCOT,ISONE
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from scripts.arima_order_exog_study import ARCHIVE_LAG_DAYS, _archive_weather
from scripts.error_analysis import MIN_TRAIN_H, _eia_with_forecast

#: The horizon #230 is about. The recursive penalty is a function of how many
#: steps get chained, so a 24h test would measure almost none of it.
FORECAST_H = 168

#: Origins are strided rather than taken at every hour: the direct arm needs
#: one training row per (origin, horizon) pair, so an unstrided build over a
#: ~2500-row window would be ~420k rows for a prototype. At stride 12 with all
#: 168 horizons it is ~31k — dense in horizon, which is what matters here.
ORIGIN_STRIDE_H = 12

#: EVERY horizon 1..168 is trained on, not a sampled subset.
#:
#: A first pass sampled 14 horizons and let `horizon_h` interpolate. That
#: handicaps the treatment arm for an implementation reason rather than a
#: strategic one: gradient-boosted trees do not interpolate, they split, so a
#: horizon never seen in training is served by whichever bucket it falls into.
#: Publishing a loss measured that way would have blamed the strategy for a
#: choice I made. Density is bought back by striding origins instead.
TRAIN_HORIZONS = tuple(range(1, FORECAST_H + 1))

#: History needed before an origin can produce a full autoregressive snapshot.
MIN_SEED_H = 168

DEFAULT_REGIONS = ["PJM", "MISO", "ERCOT", "ISONE"]


def _non_autoregressive_cols(feats: pd.DataFrame) -> list[str]:
    """Feature columns that are known at issue time for any future hour.

    Weather and calendar: knowable from the forecast. Everything derived from
    demand is not, and is supplied instead by the origin snapshot.
    """
    from data.feature_engineering import compute_autoregressive_snapshot

    ar = set(compute_autoregressive_snapshot([1.0] * 200))
    drop = ar | {"timestamp", "demand_mw", "forecast_mw", "region", "data_quality"}
    return [c for c in feats.columns if c not in drop]


def _build_direct_training(feats: pd.DataFrame, train_end: int) -> pd.DataFrame:
    """Reshape history into ``(origin features, horizon) -> demand@origin+h``.

    The autoregressive block is computed **once per origin** and reused for
    every horizon, which is precisely the difference from recursion: at
    horizon 168 the direct arm still uses real demand up to the origin, while
    the recursive arm is 168 predictions deep into its own output.
    """
    from data.feature_engineering import compute_autoregressive_snapshot

    exog_cols = _non_autoregressive_cols(feats)
    demand = feats["demand_mw"].to_numpy(dtype=float)
    rows: list[dict] = []
    for origin in range(MIN_SEED_H, train_end, ORIGIN_STRIDE_H):
        history = [float(v) for v in demand[:origin] if v > 0]
        if len(history) < MIN_SEED_H:
            continue
        snap = compute_autoregressive_snapshot(history)
        for h in TRAIN_HORIZONS:
            target = origin + h - 1
            if target >= train_end:
                continue
            row = {c: feats[c].iloc[target] for c in exog_cols}
            row.update(snap)
            row["horizon_h"] = h
            row["target"] = demand[target]
            rows.append(row)
    return pd.DataFrame(rows)


def _direct_forecast(model_dict: Any, feats: pd.DataFrame, origin: int, horizon: int) -> np.ndarray:
    """Predict every step from the origin snapshot — nothing fed forward."""
    from data.feature_engineering import compute_autoregressive_snapshot
    from models.xgboost_model import predict_xgboost

    exog_cols = _non_autoregressive_cols(feats)
    demand = feats["demand_mw"].to_numpy(dtype=float)
    snap = compute_autoregressive_snapshot([float(v) for v in demand[:origin] if v > 0])
    rows = []
    for h in range(1, horizon + 1):
        target = origin + h - 1
        row = {c: feats[c].iloc[target] for c in exog_cols}
        row.update(snap)
        row["horizon_h"] = h
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["demand_mw"] = np.nan  # predict_xgboost drops the target column
    return np.asarray(predict_xgboost(model_dict, frame), dtype=float)


def study_region(region: str, api_key: str, *, n_windows: int) -> dict[str, Any] | None:
    from config import REGION_COORDINATES
    from data.feature_engineering import engineer_features, recursive_autoregressive_forecast
    from models.evaluation import compute_mape
    from models.rolling_eval import (
        bias_pct,
        paired_deltas,
        rolling_origin_splits,
        satisficing_check,
        verdict,
        wape,
    )
    from models.xgboost_model import predict_xgboost, train_xgboost

    coords = REGION_COORDINATES.get(region)
    if not coords:
        return None
    end = datetime.now(UTC) - timedelta(days=ARCHIVE_LAG_DAYS)
    start = end - timedelta(days=200 + (n_windows * FORECAST_H) // 24)

    demand = _eia_with_forecast(region, start, end, api_key)
    if demand.empty:
        return {"region": region, "skipped": "no EIA data"}
    weather = _archive_weather(coords["lat"], coords["lon"], start, end)
    raw = demand.merge(weather, on="timestamp", how="inner").sort_values("timestamp")
    feats = engineer_features(raw).dropna(subset=["demand_mw"]).reset_index(drop=True)

    splits = rolling_origin_splits(
        len(feats), n_windows=n_windows, holdout_h=FORECAST_H, min_train_h=MIN_TRAIN_H
    )
    if not splits:
        return {"region": region, "skipped": f"no windows from {len(feats)} rows"}

    per_window: list[dict] = []
    for train_sl, test_sl in splits:
        origin = test_sl.start
        actual = feats["demand_mw"].to_numpy(dtype=float)[test_sl]
        w: dict[str, Any] = {}

        # -- control: production's own recursive protocol -------------------
        try:
            rec_model = train_xgboost(feats.iloc[train_sl])
            seed = feats["demand_mw"].to_numpy(dtype=float)[:origin]
            future = feats.iloc[test_sl].copy()
            rec = recursive_autoregressive_forecast(rec_model, seed, future, predict_xgboost)
        except Exception as e:
            w["recursive_error"] = str(e)[:150]
            rec = None

        # -- treatment: direct, horizon as a feature ------------------------
        try:
            direct_train = _build_direct_training(feats, origin)
            if direct_train.empty:
                raise ValueError("no direct training rows")
            fit_frame = direct_train.rename(columns={"target": "demand_mw"})
            dir_model = train_xgboost(fit_frame)
            dir_pred = _direct_forecast(dir_model, feats, origin, FORECAST_H)
        except Exception as e:
            w["direct_error"] = str(e)[:150]
            dir_pred = None

        for name, pred in (("recursive", rec), ("direct", dir_pred)):
            if pred is None or not np.all(np.isfinite(pred)):
                w[name] = None
                continue
            w[name] = {
                "wape": round(wape(actual, pred), 4),
                "mape": round(float(compute_mape(actual, pred)), 4),
                "bias_pct": round(bias_pct(actual, pred), 4),
            }
        if w.get("recursive") and w.get("direct"):
            w["delta_wape"] = round(w["recursive"]["wape"] - w["direct"]["wape"], 4)
        per_window.append(w)

    scored = [w for w in per_window if w.get("delta_wape") is not None]
    if not scored:
        return {"region": region, "skipped": "no window scored both arms", "windows": per_window}

    v = verdict(
        paired_deltas(
            [w["recursive"]["wape"] for w in scored], [w["direct"]["wape"] for w in scored]
        )
    )

    def _mean(arm: str, key: str) -> float:
        return round(float(np.mean([w[arm][key] for w in scored])), 4)

    sat = satisficing_check(
        treatment_bias_pct=_mean("direct", "bias_pct"),
        control_mape=_mean("recursive", "mape"),
        treatment_mape=_mean("direct", "mape"),
    )
    return {
        "region": region,
        "horizon_h": FORECAST_H,
        "n_windows_scored": len(scored),
        "verdict": v,
        "satisficing": sat,
        "ship": bool(v["decisive"] and v["winner"] == "treatment" and sat["passed"]),
        "recursive_mean": {k: _mean("recursive", k) for k in ("wape", "mape", "bias_pct")},
        "direct_mean": {k: _mean("direct", k) for k in ("wape", "mape", "bias_pct")},
        "windows": per_window,
    }


def main() -> int:
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default=",".join(DEFAULT_REGIONS))
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--out", default="docs/DIRECT_MULTIHORIZON_STUDY.json")
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
                f"    {FORECAST_H}h WAPE  recursive {res['recursive_mean']['wape']:.3f}"
                f" -> direct {res['direct_mean']['wape']:.3f}   ship={res['ship']}\n"
                f"    {res['verdict']['reason']}",
                flush=True,
            )
        else:
            print(f"    {res.get('skipped') or res.get('error')}", flush=True)
        with open(args.out, "w") as fh:
            json.dump({"regions": results}, fh, indent=2)

    scored = [r for r in results if "verdict" in r]
    if scored:
        print("\n== SUMMARY ==")
        print(
            json.dumps(
                {
                    "horizon_h": FORECAST_H,
                    "n_regions": len(scored),
                    "n_ship": sum(1 for r in scored if r["ship"]),
                    "n_decisive_direct": sum(
                        1
                        for r in scored
                        if r["verdict"]["decisive"] and r["verdict"]["winner"] == "treatment"
                    ),
                    "n_decisive_recursive": sum(
                        1
                        for r in scored
                        if r["verdict"]["decisive"] and r["verdict"]["winner"] == "control"
                    ),
                    "n_inconclusive": sum(1 for r in scored if not r["verdict"]["decisive"]),
                    "mean_delta_wape_pts": round(
                        float(np.mean([r["verdict"]["mean"] for r in scored])), 4
                    ),
                },
                indent=2,
            )
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
