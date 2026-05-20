"""Extended holdout MAPE check — does recursive prediction degrade past hour 168?

The training holdout in ``training.py`` uses ``validation_hours=168`` (one
week), so empirically we only have MAPE numbers for recursive-feature
inference within that window. PR-E proposes extending the recursive
zone to 384 hours (16 days, aligned with Open-Meteo's forecast horizon).
Before shipping, we want to confirm the holdout MAPE for hours 169-384
isn't materially worse than for hours 1-168.

Procedure:

1. Fetch EIA demand + Open-Meteo weather for the supplied region.
2. Run engineer_features → take the last 384 rows as the val window.
3. Train XGBoost on the prior rows (post-PR-D feature definitions, so
   no leakage).
4. Run recursive prediction for all 384 val hours using
   ``compute_autoregressive_snapshot`` — exactly what PR-E would do.
5. Compute per-window MAPE: 0-24, 25-72, 73-168, 169-384.
6. Report whether 169-384 is materially worse than 1-168.

Decision criterion: if MAPE(169-384) / MAPE(1-168) < 1.5, ship PR-E with
the 384h cap. If the ratio is higher, fall back to the 168h cap.

Run with::

    PYTHONPATH=. .venv/bin/python scripts/audit/extended_holdout_check.py FPL
    PYTHONPATH=. .venv/bin/python scripts/audit/extended_holdout_check.py ERCOT PJM CAISO

A single region typically takes ~5-10 minutes (XGBoost CV + recursive
prediction loop).
"""

from __future__ import annotations

import sys
import time

import numpy as np

# Load environment variables from .env BEFORE importing config
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from data.eia_client import fetch_demand  # noqa: E402
from data.feature_engineering import (  # noqa: E402
    compute_autoregressive_snapshot,
    engineer_features,
)
from data.preprocessing import merge_demand_weather  # noqa: E402
from data.weather_client import fetch_weather  # noqa: E402
from models.xgboost_model import predict_xgboost, train_xgboost  # noqa: E402

VAL_HOURS = 384  # 16 days = aligned with Open-Meteo forecast horizon
WINDOWS = [(0, 24), (24, 72), (72, 168), (168, 384)]


def run_one(region: str) -> dict | None:
    """Train + recursive holdout for one region. Returns per-window MAPE."""
    print(f"\n{'=' * 72}")
    print(f"  Extended holdout check — {region}")
    print(f"{'=' * 72}")

    t0 = time.time()
    print(f"  Fetching EIA demand for {region}…")
    demand_df = fetch_demand(region)
    if demand_df is None or demand_df.empty:
        print(f"  ✗ No demand data for {region}")
        return None

    print(f"  Fetching weather for {region}…")
    weather_df = fetch_weather(region)
    if weather_df is None or weather_df.empty:
        print(f"  ✗ No weather data for {region}")
        return None

    print("  Merging + engineering features…")
    merged = merge_demand_weather(demand_df, weather_df)
    featured = engineer_features(merged).dropna(subset=["demand_mw"]).reset_index(drop=True)
    if len(featured) < VAL_HOURS + 720:  # Need ~30 days train minimum
        print(f"  ✗ Insufficient rows ({len(featured)}); need {VAL_HOURS + 720}+")
        return None

    print(f"  Total feature-engineered rows: {len(featured)}")
    train_df = featured.iloc[:-VAL_HOURS].reset_index(drop=True)
    val_df = featured.iloc[-VAL_HOURS:].reset_index(drop=True)
    print(f"  Train: {len(train_df)} rows · Val: {len(val_df)} rows ({VAL_HOURS}h)")

    print("  Training XGBoost (n_splits=3 for speed)…")
    t_train = time.time()
    xgb_result = train_xgboost(train_df, n_splits=3)
    print(f"  ✓ Trained in {time.time() - t_train:.1f}s")

    print(f"  Running recursive prediction for {VAL_HOURS} hours…")
    t_pred = time.time()
    demand_history = train_df["demand_mw"].tolist()
    y_val = val_df["demand_mw"].values
    preds = []
    for i in range(len(val_df)):
        row = val_df.iloc[[i]].copy()
        for col, val in compute_autoregressive_snapshot(demand_history).items():
            row[col] = val
        row = row.ffill().bfill().fillna(0)
        pred = float(predict_xgboost(xgb_result, row)[0])
        preds.append(pred)
        demand_history.append(pred)
    preds = np.array(preds, dtype=float)
    print(f"  ✓ Predicted {len(preds)} rows in {time.time() - t_pred:.1f}s")

    # Per-window MAPE
    results = {"region": region, "windows": {}}
    print()
    print("  Per-window holdout MAPE (recursive prediction, post-PR-D features):")
    for lo, hi in WINDOWS:
        if hi > len(preds):
            continue
        y_slice = y_val[lo:hi]
        p_slice = preds[lo:hi]
        mask = np.abs(y_slice) > 1e-6
        if not mask.any():
            continue
        mape = float(np.mean(np.abs((y_slice[mask] - p_slice[mask]) / y_slice[mask])) * 100)
        results["windows"][f"{lo}-{hi}h"] = mape
        print(f"    hours {lo:3d}-{hi:3d}: MAPE = {mape:6.2f}%")

    # Decision: ratio of 168-384 vs 72-168
    mape_long = results["windows"].get("168-384h", 0)
    mape_full_short = results["windows"].get("72-168h", 0)
    if mape_full_short > 0 and mape_long > 0:
        ratio = mape_long / mape_full_short
        print()
        print(f"  Ratio MAPE(168-384h) / MAPE(72-168h) = {ratio:.2f}")
        if ratio < 1.5:
            print("  ✓ DECISION: ship PR-E with 384h cap (168-384 degradation < 1.5×)")
        elif ratio < 2.0:
            print("  ⚠ DECISION: borderline — consider 168h cap for safety")
        else:
            print(
                f"  ✗ DECISION: fall back to 168h cap "
                f"(168-384 degradation {ratio:.2f}× is too high)"
            )
        results["ratio_168_384_vs_72_168"] = ratio

    print(f"  Total time: {time.time() - t0:.1f}s")
    return results


def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: extended_holdout_check.py REGION [REGION ...]")
        return 1

    all_results = []
    for region in argv:
        try:
            r = run_one(region)
            if r:
                all_results.append(r)
        except Exception as e:
            print(f"  ✗ Failed for {region}: {e}")
            import traceback

            traceback.print_exc()

    if not all_results:
        print("\nNo successful runs.")
        return 1

    print(f"\n{'=' * 72}")
    print("  SUMMARY")
    print(f"{'=' * 72}")
    print(
        f"  {'Region':10s} {'0-24h':>10s} {'25-72h':>10s} {'73-168h':>10s} {'169-384h':>10s} {'ratio':>8s}"
    )
    for r in all_results:
        w = r.get("windows", {})
        ratio = r.get("ratio_168_384_vs_72_168", 0)
        # Keys are formatted as ``"{lo}-{hi}h"`` per ``WINDOWS``;
        # lookups must include the ``h`` suffix.
        print(
            f"  {r['region']:10s}"
            f" {w.get('0-24h', 0):>9.2f}%"
            f" {w.get('24-72h', 0):>9.2f}%"
            f" {w.get('72-168h', 0):>9.2f}%"
            f" {w.get('168-384h', 0):>9.2f}%"
            f" {ratio:>7.2f}x"
        )

    # Overall recommendation
    ratios = [r.get("ratio_168_384_vs_72_168", 0) for r in all_results]
    ratios = [r for r in ratios if r > 0]
    if ratios:
        max_ratio = max(ratios)
        print()
        if max_ratio < 1.5:
            print(f"  ✓ OVERALL: SHIP PR-E with 384h cap (max ratio = {max_ratio:.2f}x)")
        elif max_ratio < 2.0:
            print(f"  ⚠ OVERALL: borderline — consider 168h cap (max ratio = {max_ratio:.2f}x)")
        else:
            print(f"  ✗ OVERALL: fall back to 168h cap (max ratio = {max_ratio:.2f}x)")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
