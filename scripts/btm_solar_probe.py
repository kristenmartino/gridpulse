"""Is the hot-hour error behind-the-meter solar? A diagnostic, not a feature.

`COOLING_RESPONSE_STUDY.md` established what the hot-hour error is *not*: with
perfect future weather, better temperature features moved nothing. The
surviving hypothesis was rooftop PV — hot afternoons are peak-irradiance
afternoons, PV suppresses *net* load exactly at cooling peak, and installed
capacity grows quarterly.

**The naive form of that hypothesis is already weak, and this says so before
measuring anything.** The model is trained on EIA-930 `D`, which is metered
grid load and therefore already net of BTM generation, and it is given
``solar_capacity_factor``, raw ``shortwave_radiation``,
``direct_normal_irradiance``, ``diffuse_radiation`` and ``cloud_cover``. It
can learn the average suppression. The hypothesis only survives if there is
**residual** structure after the model has had that chance.

So this probes residuals rather than building features — the lesson from the
cooling pack, which was built first and measured second, and failed.

## Falsifiable predictions

If unmodelled BTM solar drives hot-hour error, then:

1. **Sign.** Signed error runs **positive** (over-forecast) on high-irradiance
   hours — we predict load that PV then removes.
2. **Not just temperature.** That holds *within* a temperature bucket, since
   irradiance and temperature are strongly correlated and temperature is
   already well represented.
3. **Dose-response.** The effect is larger in high-penetration BAs (CAISO,
   FPL) than in low ones (MISO, ISONE, NYISO).

Any of these failing is evidence against. All three are reported whatever they
say.

Usage:
    python -m scripts.btm_solar_probe --windows 6
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

#: Chosen to span BTM penetration, which is what prediction 3 needs. CAISO is
#: the US extreme; FPL is high; MISO/ISONE/NYISO/ERCOT are comparatively low.
#: A probe run only on the addressable-8 could not test dose-response at all.
DEFAULT_REGIONS = ["CAISO", "FPL", "SOCO", "TVA", "MISO", "ISONE", "NYISO", "ERCOT"]

#: Rough rooftop/small-scale PV penetration, ordinal only. Used to rank BAs for
#: the dose-response check, never as a regressor — the exact figures are an
#: EIA-861M question this probe deliberately does not depend on.
BTM_RANK = {
    "CAISO": 5,
    "FPL": 4,
    "SOCO": 3,
    "TVA": 2,
    "ERCOT": 2,
    "NYISO": 2,
    "ISONE": 3,
    "MISO": 1,
}

#: Daylight only. Night hours cannot carry a PV signal and would dilute every
#: bucket with zeros.
MIN_DAY_IRRADIANCE = 50.0


def probe_region(region: str, api_key: str, *, n_windows: int) -> dict[str, Any] | None:
    from config import REGION_COORDINATES
    from data.feature_engineering import engineer_features
    from models.rolling_eval import rolling_origin_splits

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

    feats = make_day_ahead_safe(engineer_features(raw))
    feats = feats.dropna(subset=["demand_mw"]).reset_index(drop=True)
    splits = rolling_origin_splits(
        len(feats), n_windows=n_windows, holdout_h=HOLDOUT_H, min_train_h=MIN_TRAIN_H
    )
    if not splits:
        return {"region": region, "skipped": "no windows"}

    rows: list[dict] = []
    for w_i, (train_sl, test_sl) in enumerate(splits):
        train, test = feats.iloc[train_sl], feats.iloc[test_sl]
        pred = _fit_predict_xgb(train, test)
        if pred is None:
            continue
        actual = test["demand_mw"].to_numpy(dtype=float)
        ok = actual != 0
        rows.append(
            pd.DataFrame(
                {
                    # Positive = we forecast MORE than materialised, which is
                    # the direction unmodelled PV suppression would produce.
                    "signed_pct": (pred[ok] - actual[ok]) / actual[ok] * 100,
                    "irradiance": test["shortwave_radiation"].to_numpy(dtype=float)[ok],
                    "temp_f": test["temperature_2m"].to_numpy(dtype=float)[ok],
                    "window": w_i,
                }
            )
        )
    if not rows:
        return {"region": region, "skipped": "no window produced a forecast"}

    h = pd.concat(rows, ignore_index=True)
    day = h[h["irradiance"] > MIN_DAY_IRRADIANCE].copy()
    if len(day) < 200:
        return {"region": region, "skipped": f"only {len(day)} daylight hours"}

    day["irr_q"] = pd.qcut(day["irradiance"].rank(method="first"), 5, labels=list("12345"))
    day["temp_q"] = pd.qcut(day["temp_f"].rank(method="first"), 3, labels=["cool", "mid", "hot"])

    by_irr = day.groupby("irr_q", observed=True)["signed_pct"].mean().round(3)
    # Prediction 2: the same gradient must survive INSIDE a temperature bucket,
    # otherwise it is just temperature wearing an irradiance costume.
    within = day.groupby(["temp_q", "irr_q"], observed=True)["signed_pct"].mean().round(3).unstack()
    hot_band = within.loc["hot"] if "hot" in within.index else None

    return {
        "region": region,
        "btm_rank": BTM_RANK.get(region),
        "n_daylight_hours": int(len(day)),
        "mean_signed_pct_all_day": round(float(day["signed_pct"].mean()), 3),
        "signed_pct_by_irradiance_quintile": {str(k): float(v) for k, v in by_irr.items()},
        # The headline: top-quintile irradiance minus bottom, in points of
        # signed error. Positive supports the hypothesis.
        "irradiance_gradient_pts": round(float(by_irr.iloc[-1] - by_irr.iloc[0]), 3),
        "signed_pct_by_irradiance_within_hot": (
            {str(k): float(v) for k, v in hot_band.items()} if hot_band is not None else None
        ),
        "gradient_within_hot_pts": (
            round(float(hot_band.iloc[-1] - hot_band.iloc[0]), 3) if hot_band is not None else None
        ),
        # Prediction implied by capacity growth: later windows over-forecast
        # more. Weak within one season, reported rather than relied on.
        "signed_pct_by_window": {
            str(int(k)): round(float(v), 3)
            for k, v in day.groupby("window")["signed_pct"].mean().items()
        },
    }


def main() -> int:
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default=",".join(DEFAULT_REGIONS))
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--out", default="docs/BTM_SOLAR_PROBE.json")
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
        if "irradiance_gradient_pts" in res:
            print(
                f"    gradient {res['irradiance_gradient_pts']:+.3f} pts "
                f"(within-hot {res['gradient_within_hot_pts']:+.3f})  "
                f"mean signed {res['mean_signed_pct_all_day']:+.3f}%",
                flush=True,
            )
        else:
            print(f"    {res.get('skipped') or res.get('error')}", flush=True)
        with open(args.out, "w") as fh:
            json.dump({"regions": results}, fh, indent=2)

    scored = [r for r in results if "irradiance_gradient_pts" in r]
    if scored:
        grads = [r["irradiance_gradient_pts"] for r in scored]
        ranks = [r["btm_rank"] for r in scored]
        corr = float(np.corrcoef(ranks, grads)[0, 1]) if len(scored) > 2 else float("nan")
        print("\n== SUMMARY ==")
        print(
            json.dumps(
                {
                    "n_regions": len(scored),
                    "n_positive_gradient": sum(1 for g in grads if g > 0),
                    "mean_gradient_pts": round(float(np.mean(grads)), 3),
                    # nan-safe: the within-hot control is unmeasurable where
                    # hot AND low-irradiance hours barely exist, which is a
                    # real property of the data, not an error. Averaging with
                    # np.mean would turn one missing BA into a NaN headline.
                    "mean_gradient_within_hot_pts": round(
                        float(
                            np.mean(
                                [
                                    r["gradient_within_hot_pts"]
                                    for r in scored
                                    if r["gradient_within_hot_pts"] is not None
                                    and np.isfinite(r["gradient_within_hot_pts"])
                                ]
                            )
                        ),
                        3,
                    ),
                    "n_within_hot_measurable": sum(
                        1
                        for r in scored
                        if r["gradient_within_hot_pts"] is not None
                        and np.isfinite(r["gradient_within_hot_pts"])
                    ),
                    # Prediction 3. Positive and sizeable would be real support;
                    # near zero or negative kills the dose-response claim.
                    "corr_btm_rank_vs_gradient": None if np.isnan(corr) else round(corr, 3),
                },
                indent=2,
            )
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
