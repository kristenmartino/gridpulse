"""Error analysis: where does our forecast error actually come from?

The project has spent months building measurement apparatus and then choosing
work from an issue queue. This is the step that was missing — take the errors,
bucket them by cause, and size each bucket, so the next change is chosen by
evidence about where the error is rather than by which issue is next.

Three arms per hour, which is what makes the buckets meaningful:

* **operator** — the BA's own day-ahead forecast (EIA-930 ``forecast_mw``).
  A human-level baseline in the Ng sense: what a competent incumbent with
  local knowledge actually achieves.
* **naive** — seasonal-naive, lag-24. The trivial floor (``models/skill.py``).
* **ours** — reconstructed locally over a rolling origin.

The decomposition that matters is **shared vs idiosyncratic**: an hour both we
and the operator miss is largely inherent (weather surprise, event, bad
settlement), and no model change recovers it. An hour they get right and we do
not is addressable. Those two need different work, and until they are sized
separately the total error number cannot tell you which you have.

Everything is measured in **MW of error**, not percent. A 20% miss on a 300 MW
co-op and a 2% miss on PJM are 60 MW and ~2,000 MW; the percentage view has
been driving attention toward the former.

Usage:
    python -m scripts.error_analysis --regions MISO,ERCOT,ISONE --windows 6
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from scripts.arima_order_exog_study import (
    ARCHIVE_LAG_DAYS,
    _archive_weather,
)

warnings.filterwarnings("ignore")

HOLDOUT_H = 168
MIN_TRAIN_H = 1200
HISTORY_DAYS = 120
SEASONAL_LAG_H = 24

#: The 8 BAs carrying 82% of the fleet's *addressable* gap — where we are
#: worse than the operator, ranked by MW rather than by percentage. Derived
#: from /api/v1/benchmark; see docs/ERROR_ANALYSIS.md for the derivation.
DEFAULT_REGIONS = ["MISO", "ERCOT", "ISONE", "NYISO", "PJM", "TVA", "FPL", "SOCO"]

#: An hour counts as "missed" by an arm when its absolute percentage error
#: exceeds this. Used only to classify hours into shared/idiosyncratic buckets,
#: never to score anything.
MISS_PCT = 5.0


def _eia_with_forecast(region: str, start: datetime, end: datetime, api_key: str) -> pd.DataFrame:
    """Settled demand plus the operator's own day-ahead forecast.

    ``_eia_demand`` drops ``forecast_mw``; this analysis needs it, because the
    operator arm is the human-level baseline the whole decomposition rests on.
    """

    from data.eia_client import _get_eia_code
    from scripts.arima_order_exog_study import EIA_URL, _get_with_retry

    rows: list[dict] = []
    for series in ("D", "DF"):
        offset = 0
        while True:
            r = _get_with_retry(
                EIA_URL,
                params={
                    "api_key": api_key,
                    "frequency": "hourly",
                    "data[0]": "value",
                    "facets[respondent][]": _get_eia_code(region),
                    "facets[type][]": series,
                    "start": start.strftime("%Y-%m-%dT%H"),
                    "end": end.strftime("%Y-%m-%dT%H"),
                    "sort[0][column]": "period",
                    "sort[0][direction]": "asc",
                    "offset": offset,
                    "length": 5000,
                },
                timeout=90,
            )
            batch = r.json().get("response", {}).get("data", [])
            for b in batch:
                rows.append({"period": b["period"], "type": series, "value": b["value"]})
            if len(batch) < 5000:
                break
            offset += 5000
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["period"], utc=True, format="mixed")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    wide = df.pivot_table(index="timestamp", columns="type", values="value", aggfunc="last")
    wide = wide.rename(columns={"D": "demand_mw", "DF": "forecast_mw"}).reset_index()
    for col in ("demand_mw", "forecast_mw"):
        if col not in wide.columns:
            wide[col] = np.nan
    return wide.sort_values("timestamp")


#: Autoregressive features a day-ahead forecast cannot have. `demand_lag_1h`
#: and `_3h` fall inside the horizon outright; the rolling windows end at the
#: target hour, so they too contain demand the issuer has not seen.
SUB_HORIZON_LAGS = ("demand_lag_1h", "demand_lag_3h")


def make_day_ahead_safe(feats: pd.DataFrame, horizon_h: int = 24) -> pd.DataFrame:
    """Restrict autoregressive features to what is known at issue time.

    Without this the "ours" arm is not a day-ahead forecast at all. The first
    run scored WAPE **0.90%** against the operator's 2.70% on MISO — three
    times better than production's own live number for that BA, which is the
    tell. `demand_lag_1h` was the top feature: the model was reading demand
    from one hour before the target while the operator forecast a full day
    out. Comparing those two is meaningless.

    Sub-horizon lags are dropped; rolling windows are shifted by ``horizon_h``
    so they end at issue time rather than at the target hour. Production
    solves the same problem differently (recursive prediction through
    `_build_future_feature_frame`, plus ADR-009 anchor conditioning); this is
    the direct-multi-horizon equivalent, and it is a *different* forecaster
    from the served one — stated as a limit in the write-up.
    """
    out = feats.copy()
    out = out.drop(columns=[c for c in SUB_HORIZON_LAGS if c in out.columns])
    for col in out.columns:
        if col.startswith("demand_roll_"):
            out[col] = out[col].shift(horizon_h)
    return out


def _fit_predict_xgb(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray | None:
    """Our arm, via the production XGBoost trainer.

    XGBoost is ADR-005's primary single-model forecaster, used here as a
    tractable stand-in for the served ensemble — a limitation, stated in the
    write-up and validated against live drift rather than assumed.
    """
    from models.xgboost_model import predict_xgboost, train_xgboost

    # `predict_xgboost` takes the whole dict `train_xgboost` returns (it needs
    # `feature_names` to align columns), not the raw booster inside it.
    try:
        model = train_xgboost(train)
        pred = np.asarray(predict_xgboost(model, test), dtype=float)
        return pred if np.all(np.isfinite(pred)) else None
    except Exception as e:
        # Loud, not silent: a swallowed failure here reads as "no windows"
        # and quietly shrinks the study.
        print(f"      xgb failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
        return None


def _seasonal_naive(full: pd.DataFrame, test_idx: pd.Index) -> np.ndarray:
    """Lag-24 on the full frame, so the first holdout hours have a lag."""
    return full["demand_mw"].shift(SEASONAL_LAG_H).loc[test_idx].to_numpy(dtype=float)


def analyse_region(region: str, api_key: str, *, n_windows: int) -> dict[str, Any] | None:
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
    df = demand.merge(weather, on="timestamp", how="inner").sort_values("timestamp")

    feats = engineer_features(df)
    # `engineer_features` already carries forecast_mw through, and
    # `xgboost_model.EXCLUDE_COLS` already drops it from the feature set.
    # Merging it back created forecast_mw_x/_y, which SLIPPED PAST that
    # exclusion and made the operator's own forecast the model's top feature —
    # leaking the arm we are measuring against into the arm being measured.
    # Caught from the training log (`top_features=['forecast_mw_x', ...]`).
    assert "forecast_mw" in feats.columns, "operator arm requires forecast_mw"
    # Make our arm an actual DAY-AHEAD forecast before anything is scored.
    feats = make_day_ahead_safe(feats)
    feats = feats.dropna(subset=["demand_mw"]).reset_index(drop=True)

    splits = rolling_origin_splits(
        len(feats), n_windows=n_windows, holdout_h=HOLDOUT_H, min_train_h=MIN_TRAIN_H
    )
    if not splits:
        return {"region": region, "skipped": f"no windows from {len(feats)} rows"}

    hours: list[dict] = []
    for train_sl, test_sl in splits:
        train, test = feats.iloc[train_sl], feats.iloc[test_sl]
        ours = _fit_predict_xgb(train, test)
        if ours is None:
            continue
        naive = _seasonal_naive(feats, test.index)
        actual = test["demand_mw"].to_numpy(dtype=float)
        operator = test["forecast_mw"].to_numpy(dtype=float)
        ts = pd.to_datetime(test["timestamp"])
        prev = feats["demand_mw"].shift(1).loc[test.index].to_numpy(dtype=float)
        for i in range(len(actual)):
            hours.append(
                {
                    "ts": ts.iloc[i],
                    "actual": actual[i],
                    "ours": ours[i],
                    "operator": operator[i],
                    "naive": naive[i],
                    "temp_f": float(test["temperature_2m"].iloc[i])
                    if "temperature_2m" in test.columns
                    else np.nan,
                    "ramp_mw": abs(actual[i] - prev[i]) if np.isfinite(prev[i]) else np.nan,
                    "is_holiday": int(test["is_holiday"].iloc[i])
                    if "is_holiday" in test.columns
                    else 0,
                }
            )
    if not hours:
        return {"region": region, "skipped": "no window produced a forecast"}
    return {"region": region, "n_windows": len(splits), "hours": hours}


def bucket(hours: list[dict]) -> dict[str, Any]:
    """Size each error bucket in MW, not percent."""
    h = pd.DataFrame(hours)
    h = h[np.isfinite(h["actual"]) & np.isfinite(h["ours"])]
    h["our_err"] = (h["ours"] - h["actual"]).abs()
    h["op_err"] = (h["operator"] - h["actual"]).abs()
    h["naive_err"] = (h["naive"] - h["actual"]).abs()
    h["our_pct"] = h["our_err"] / h["actual"].abs() * 100
    h["op_pct"] = h["op_err"] / h["actual"].abs() * 100

    total = float(h["our_err"].sum())
    paired = h[np.isfinite(h["op_err"])]

    def share(mask: pd.Series) -> dict:
        sel = paired[mask]
        return {
            "hours": int(len(sel)),
            "hours_pct": round(len(sel) / len(paired) * 100, 1) if len(paired) else None,
            "our_mw_err": round(float(sel["our_err"].sum())),
            "share_of_our_error_pct": round(
                float(sel["our_err"].sum()) / float(paired["our_err"].sum()) * 100, 1
            )
            if len(paired)
            else None,
            "our_mean_err_mw": round(float(sel["our_err"].mean())) if len(sel) else None,
            "operator_mean_err_mw": round(float(sel["op_err"].mean())) if len(sel) else None,
        }

    both = (paired["our_pct"] > MISS_PCT) & (paired["op_pct"] > MISS_PCT)
    ours_only = (paired["our_pct"] > MISS_PCT) & (paired["op_pct"] <= MISS_PCT)
    theirs_only = (paired["our_pct"] <= MISS_PCT) & (paired["op_pct"] > MISS_PCT)
    neither = (paired["our_pct"] <= MISS_PCT) & (paired["op_pct"] <= MISS_PCT)

    by_hour = (h.assign(hr=h["ts"].dt.hour).groupby("hr")["our_err"].sum() / total * 100).round(2)
    ramp_q = pd.qcut(h["ramp_mw"].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    by_ramp = (h.groupby(ramp_q, observed=True)["our_err"].sum() / total * 100).round(1)
    temp_q = pd.qcut(
        h["temp_f"].rank(method="first"), 5, labels=["cold", "cool", "mild", "warm", "hot"]
    )
    by_temp = (h.groupby(temp_q, observed=True)["our_err"].sum() / total * 100).round(1)
    weekend = h["ts"].dt.dayofweek >= 5

    return {
        "n_hours": int(len(h)),
        "n_paired_with_operator": int(len(paired)),
        "our_total_mw_err": round(total),
        "our_wape_pct": round(float(h["our_err"].sum() / h["actual"].abs().sum() * 100), 3),
        "operator_wape_pct": round(
            float(paired["op_err"].sum() / paired["actual"].abs().sum() * 100), 3
        )
        if len(paired)
        else None,
        "naive_wape_pct": round(float(h["naive_err"].sum() / h["actual"].abs().sum() * 100), 3),
        "shared_vs_idiosyncratic": {
            "both_missed": share(both),
            "only_we_missed": share(ours_only),
            "only_operator_missed": share(theirs_only),
            "neither_missed": share(neither),
        },
        "share_by_hour_of_day_pct": by_hour.to_dict(),
        "share_by_ramp_quartile_pct": by_ramp.to_dict(),
        "share_by_temp_quintile_pct": by_temp.to_dict(),
        "weekend_share_pct": round(float(h[weekend]["our_err"].sum()) / total * 100, 1),
        "weekend_hours_pct": round(float(weekend.mean()) * 100, 1),
        "holiday_share_pct": round(
            float(h[h["is_holiday"] == 1]["our_err"].sum()) / total * 100, 2
        ),
        "holiday_hours_pct": round(float((h["is_holiday"] == 1).mean()) * 100, 2),
    }


def main() -> int:
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default=",".join(DEFAULT_REGIONS))
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--out", default="docs/ERROR_ANALYSIS.json")
    args = ap.parse_args()

    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        print("EIA_API_KEY not set")
        return 2

    results = []
    for region in [r.strip() for r in args.regions.split(",") if r.strip()]:
        print(f"[{region}] ...", flush=True)
        try:
            res = analyse_region(region, api_key, n_windows=args.windows)
        except Exception as e:
            res = {"region": region, "error": str(e)[:200]}
        if not res:
            continue
        if "hours" in res:
            res = {"region": region, "n_windows": res["n_windows"], **bucket(res["hours"])}
        results.append(res)
        print("   ", json.dumps({k: v for k, v in res.items() if k != "region"})[:180])
        with open(args.out, "w") as fh:
            json.dump({"regions": results}, fh, indent=2)

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
