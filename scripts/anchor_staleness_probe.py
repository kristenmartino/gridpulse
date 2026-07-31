"""What would a real-time anchor be worth? (ISO feed evaluation)

The case for ingesting ISO real-time feeds rested on an assumption I made and
did not check: that EIA-930's publishing lag leaves our forecast anchored on
stale demand, and that 5-minute ISO data would fix it.

**Measured first, and the premise is weak.** As of 2026-07-31 the EIA
publication lag is **1.7h for all 51 BAs** — one schedule, no variation — and
trailing stub hours (where EIA republishes its own forecast as the actual)
have a **median of 0**, with only ERCOT showing any at 6h. Effective anchor
staleness is therefore ~1.7h almost everywhere, 7.7h at worst.

*(A first pass reported a 19.7h median. That was a bug in the probe: `D` and
`DF` cover different ranges — `DF` extends into the future — so every
forecast-only future hour was counted as a stale actual. Corrected above.)*

So the remaining question is not how stale the anchor is, but whether that
staleness costs anything. This measures the **ceiling**: forecast accuracy as
a function of anchor age, holding the scored hours fixed.

## Method

For each (BA, window) one direct multi-horizon model is trained — the #230
formulation, which makes this cheap: the same model answers every staleness
level, since staleness is just a shift in `horizon_h`.

For staleness `s`, the autoregressive snapshot is taken at `origin - s` and
the scored hours stay `[origin, origin + 24)`, so horizons run `s+1 .. s+24`.
`s = 0` is a perfect real-time anchor — the best an ISO feed could deliver.
The gap between `s = 0` and `s = 2` bounds what real-time ingestion could buy
at today's lag.

Usage:
    python -m scripts.anchor_staleness_probe --regions PJM,MISO,ERCOT
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from scripts.arima_order_exog_study import ARCHIVE_LAG_DAYS, _archive_weather
from scripts.direct_multihorizon_study import (
    MIN_SEED_H,
    ORIGIN_STRIDE_H,
    STUDY_CV_SPLITS,
    _non_autoregressive_cols,
)
from scripts.error_analysis import MIN_TRAIN_H, _eia_with_forecast

#: The operating horizon. Staleness matters most relative to horizon length,
#: so a 24h test is the fairest place to look for an effect — at 168h a couple
#: of hours of anchor age is negligible by construction.
SCORE_H = 24

#: Anchor ages tested, in hours. 0 = perfect real-time (the ISO-feed ideal),
#: 2 ~ today's EIA lag, and the rest bracket it to show the shape of the curve
#: rather than a single contrast.
STALENESS_H = (0, 1, 2, 4, 8, 16, 24)

#: Training must cover the longest horizon any staleness level reaches.
MAX_H = SCORE_H + max(STALENESS_H)

DEFAULT_REGIONS = ["PJM", "MISO", "ERCOT", "CAISO", "ISONE"]


def _build_training(feats: pd.DataFrame, train_end: int) -> pd.DataFrame:
    from data.feature_engineering import compute_autoregressive_snapshot

    exog = _non_autoregressive_cols(feats)
    demand = feats["demand_mw"].to_numpy(dtype=float)
    rows: list[dict] = []
    for origin in range(MIN_SEED_H, train_end, ORIGIN_STRIDE_H):
        history = [float(v) for v in demand[:origin] if v > 0]
        if len(history) < MIN_SEED_H:
            continue
        snap = compute_autoregressive_snapshot(history)
        for h in range(1, MAX_H + 1):
            target = origin + h - 1
            if target >= train_end:
                continue
            row = {c: feats[c].iloc[target] for c in exog}
            row.update(snap)
            row["horizon_h"] = h
            row["demand_mw"] = demand[target]
            rows.append(row)
    return pd.DataFrame(rows)


def _forecast_at_staleness(
    model: Any, feats: pd.DataFrame, origin: int, stale_h: int
) -> np.ndarray:
    """Snapshot taken `stale_h` hours before the origin; scored hours unchanged."""
    from data.feature_engineering import compute_autoregressive_snapshot
    from models.xgboost_model import predict_xgboost

    exog = _non_autoregressive_cols(feats)
    demand = feats["demand_mw"].to_numpy(dtype=float)
    snap = compute_autoregressive_snapshot([float(v) for v in demand[: origin - stale_h] if v > 0])
    rows = []
    for step in range(SCORE_H):
        target = origin + step
        row = {c: feats[c].iloc[target] for c in exog}
        row.update(snap)
        # The forecast is issued `stale_h` earlier, so the target sits further out.
        row["horizon_h"] = stale_h + step + 1
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["demand_mw"] = np.nan
    return np.asarray(predict_xgboost(model, frame), dtype=float)


def probe_region(region: str, api_key: str, *, n_windows: int) -> dict[str, Any] | None:
    from config import REGION_COORDINATES
    from data.feature_engineering import engineer_features
    from models.rolling_eval import rolling_origin_splits, wape
    from models.xgboost_model import train_xgboost

    coords = REGION_COORDINATES.get(region)
    if not coords:
        return None
    end = datetime.now(UTC) - timedelta(days=ARCHIVE_LAG_DAYS)
    start = end - timedelta(days=160)

    demand = _eia_with_forecast(region, start, end, api_key)
    if demand.empty:
        return {"region": region, "skipped": "no EIA data"}
    weather = _archive_weather(coords["lat"], coords["lon"], start, end)
    raw = demand.merge(weather, on="timestamp", how="inner").sort_values("timestamp")
    feats = engineer_features(raw).dropna(subset=["demand_mw"]).reset_index(drop=True)

    splits = rolling_origin_splits(
        len(feats), n_windows=n_windows, holdout_h=SCORE_H, min_train_h=MIN_TRAIN_H
    )
    if not splits:
        return {"region": region, "skipped": "no windows"}

    by_stale: dict[int, list[float]] = {s: [] for s in STALENESS_H}
    # train_sl is unused: training runs to `origin` via _build_training, which
    # rebuilds the (origin, horizon) frame itself.
    for _train_sl, test_sl in splits:
        origin = test_sl.start
        if origin - max(STALENESS_H) < MIN_SEED_H:
            continue
        if origin + SCORE_H > len(feats):
            continue
        try:
            train = _build_training(feats, origin)
            if train.empty:
                continue
            model = train_xgboost(train, n_splits=STUDY_CV_SPLITS)
        except Exception as e:
            print(f"      train failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
            continue
        actual = feats["demand_mw"].to_numpy(dtype=float)[origin : origin + SCORE_H]
        for s in STALENESS_H:
            try:
                pred = _forecast_at_staleness(model, feats, origin, s)
                if np.all(np.isfinite(pred)):
                    by_stale[s].append(wape(actual, pred))
            except Exception:
                continue

    scored = {s: v for s, v in by_stale.items() if v}
    if 0 not in scored:
        return {"region": region, "skipped": "no baseline window scored"}
    base = float(np.mean(scored[0]))
    return {
        "region": region,
        "n_windows": len(scored[0]),
        "wape_by_staleness_h": {str(s): round(float(np.mean(v)), 4) for s, v in scored.items()},
        # Cost of anchor age, in WAPE points relative to a perfect real-time anchor.
        "cost_vs_realtime_pts": {
            str(s): round(float(np.mean(v)) - base, 4) for s, v in scored.items()
        },
    }


def main() -> int:
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default=",".join(DEFAULT_REGIONS))
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--out", default="docs/ANCHOR_STALENESS_PROBE.json")
    args = ap.parse_args()

    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        print("EIA_API_KEY not set")
        return 2

    results = []
    for region in [r.strip() for r in args.regions.split(",") if r.strip()]:
        print(f"[{region}] ...", flush=True)
        try:
            res = probe_region(region, api_key, n_windows=args.windows)
        except Exception as e:
            res = {"region": region, "error": str(e)[:200]}
        if not res:
            continue
        results.append(res)
        if "cost_vs_realtime_pts" in res:
            c = res["cost_vs_realtime_pts"]
            print(
                f"    WAPE at s=0 {res['wape_by_staleness_h']['0']:.3f}  "
                f"cost @2h {c.get('2', float('nan')):+.3f}  @8h {c.get('8', float('nan')):+.3f}  "
                f"@24h {c.get('24', float('nan')):+.3f}",
                flush=True,
            )
        else:
            print(f"    {res.get('skipped') or res.get('error')}", flush=True)
        with open(args.out, "w") as fh:
            json.dump({"regions": results}, fh, indent=2)

    scored = [r for r in results if "cost_vs_realtime_pts" in r]
    if scored:
        print("\n== SUMMARY: cost of anchor age vs a perfect real-time anchor ==")
        for s in STALENESS_H:
            vals = [r["cost_vs_realtime_pts"].get(str(s)) for r in scored]
            vals = [v for v in vals if v is not None]
            if vals:
                print(
                    f"  staleness {s:>2}h  mean {np.mean(vals):+.3f} pts   median {np.median(vals):+.3f}"
                )
        print("\n  Today's EIA lag is ~1.7h, so the 2h row bounds what real-time ISO")
        print("  ingestion could buy on the anchor.")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
