"""ARIMA order selection: univariate search vs exog-aware search (#297).

`_auto_select_order` passed `exogenous=` to `pm.auto_arima`. pmdarima 2.x
calls that parameter `X`, and `auto_arima` takes `**fit_args` — so the kwarg
was accepted, swallowed, and ignored. The stepwise (p,q,P,Q) search ran on a
**univariate** view of demand while the final `SARIMAX(..., exog=exog)` fit
used all five weather regressors. Every selected order was chosen for a model
we do not fit.

The one-word fix is obvious. What is *not* obvious is whether it helps — a
search that ignores regressors tends to absorb weather-driven structure into
higher AR/MA terms, which is wrong but not necessarily worse out-of-sample.
The issue's own acceptance says to measure before merging, so this measures.

## Arms

Both arms fit the SAME final model (`SARIMAX` with exog), because that half
was never broken. Only the ORDER differs:

* **A — control** (pre-#297): order selected with no `X` — a univariate search.
* **B — fixed**: order selected with `X=exog` — the search sees the regressors.

## Protocol

Per BA: settled EIA demand + ERA5 archive weather over a window ending ≥5
days ago (the archive's publication lag; see the local-re-measure note in
docs/BACKTEST_RESULTS.md). Features are engineered through the production
`engineer_features`. The last `HOLDOUT_H` hours are held out; both arms fit
on the same training slice and forecast the holdout with **known** future
exog, which isolates the order choice from weather-forecast error.

Reported per BA: selected orders, search wall-time, and holdout sMAPE for
each arm. The verdict table is committed with the run so the decision is not
re-litigated from memory.

Usage:
    python -m scripts.arima_order_exog_study --regions FPL,PJM,MISO
    python -m scripts.arima_order_exog_study --all
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
EIA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

#: Days of history to pull. auto_arima's search slice is 504 rows and
#: train_arima caps at 2160; 120 days leaves room for both plus the holdout.
HISTORY_DAYS = 120
#: The ERA5 archive publishes on a ~5-day lag. End the window well clear of
#: it so every hour scored is settled on both sides.
ARCHIVE_LAG_DAYS = 7
#: One week of hourly holdout — long enough that a seasonal order matters.
HOLDOUT_H = 168

#: The BAs where ARIMA is currently the 24h champion (live drift, 2026-07-29)
#: plus the major ISOs, where the published benchmark says we lose. If the
#: fix helps anywhere it has to help here. NOTE these are the project's own
#: codes (`config.REGION_COORDINATES`), not EIA respondent codes — ERCOT/CAISO,
#: not ERCO/CISO. A code that misses simply produces no row, so a typo reads
#: as a smaller study rather than an error.
DEFAULT_REGIONS = [
    "FPL",
    "IPCO",
    "PACE",
    "PSCO",
    "LDWP",
    "WALC",
    "ERCOT",
    "MISO",
    "PJM",
    "CAISO",
]


def _eia_demand(region: str, start: datetime, end: datetime, api_key: str) -> pd.DataFrame:
    # Our code is not always EIA's respondent code (ERCOT→ERCO, CAISO→CISO).
    # Reuse the client's own mapping rather than re-deriving it — passing the
    # internal code straight through returns an empty frame, which reads as
    # "insufficient history" rather than as the lookup error it is.
    from data.eia_client import _get_eia_code

    respondent = _get_eia_code(region)
    rows: list[dict] = []
    offset = 0
    while True:
        r = requests.get(
            EIA_URL,
            params={
                "api_key": api_key,
                "frequency": "hourly",
                "data[0]": "value",
                "facets[respondent][]": respondent,
                "facets[type][]": "D",
                "start": start.strftime("%Y-%m-%dT%H"),
                "end": end.strftime("%Y-%m-%dT%H"),
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "offset": offset,
                "length": 5000,
            },
            timeout=90,
        )
        r.raise_for_status()
        batch = r.json().get("response", {}).get("data", [])
        rows.extend(batch)
        if len(batch) < 5000:
            break
        offset += 5000
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["period"], utc=True, format="mixed")
    df["demand_mw"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["timestamp", "demand_mw"]].dropna().sort_values("timestamp")


def _archive_weather(lat: float, lon: float, start: datetime, end: datetime) -> pd.DataFrame:
    r = requests.get(
        ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "hourly": ",".join(
                [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "dew_point_2m",
                    "apparent_temperature",
                    "precipitation",
                    "cloud_cover",
                    "wind_speed_10m",
                    "wind_speed_100m",
                    "wind_direction_10m",
                    "shortwave_radiation",
                    "direct_radiation",
                    "diffuse_radiation",
                    "surface_pressure",
                ]
            ),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "UTC",
        },
        timeout=120,
    )
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame(h)
    df["timestamp"] = pd.to_datetime(df.pop("time"), utc=True)
    # The feature builder expects the serving names.
    df = df.rename(columns={"wind_speed_100m": "wind_speed_80m"})
    return df


def _smape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Symmetric MAPE, the study metric used across this repo's arms."""
    denom = (np.abs(actual) + np.abs(pred)) / 2.0
    ok = denom > 0
    return float(np.mean(np.abs(actual[ok] - pred[ok]) / denom[ok]) * 100)


def _select_order(y: np.ndarray, exog: np.ndarray | None, *, use_exog: bool):
    """Run the real pmdarima search, with or without the regressors.

    Arm A reproduces the pre-#297 behaviour exactly: the kwarg was swallowed,
    so the search simply never saw `X`. Passing nothing is not an
    approximation of that bug — it *is* that bug.
    """
    import pmdarima as pm

    t0 = time.perf_counter()
    auto = pm.auto_arima(
        y,
        X=exog if use_exog else None,
        seasonal=True,
        m=24,
        max_p=2,
        max_q=2,
        max_P=1,
        max_Q=1,
        max_d=0,
        d=0,
        max_D=1,
        D=1,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        n_fits=20,
    )
    return auto.order, auto.seasonal_order, time.perf_counter() - t0


def _fit_and_score(
    y_train: np.ndarray,
    x_train: np.ndarray | None,
    y_test: np.ndarray,
    x_test: np.ndarray | None,
    order,
    seasonal_order,
) -> float | None:
    """Both arms fit WITH exog — only the order under test differs."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    try:
        fit = SARIMAX(
            y_train,
            exog=x_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        pred = np.asarray(fit.forecast(steps=len(y_test), exog=x_test), dtype=float)
        if not np.all(np.isfinite(pred)):
            return None
        return _smape(y_test, pred)
    except Exception:
        return None


def study_region(region: str, api_key: str) -> dict[str, Any] | None:
    from config import REGION_COORDINATES
    from data.feature_engineering import engineer_features
    from models.arima_model import _get_exog

    coords = REGION_COORDINATES.get(region)
    if not coords:
        return None

    end = datetime.now(UTC) - timedelta(days=ARCHIVE_LAG_DAYS)
    start = end - timedelta(days=HISTORY_DAYS)

    demand = _eia_demand(region, start, end, api_key)
    if demand.empty or len(demand) < HOLDOUT_H * 3:
        return {"region": region, "skipped": "insufficient demand history"}
    weather = _archive_weather(coords["lat"], coords["lon"], start, end)

    df = demand.merge(weather, on="timestamp", how="inner").sort_values("timestamp")
    feats = engineer_features(df)
    feats = feats.dropna(subset=["demand_mw"]).reset_index(drop=True)
    if len(feats) < HOLDOUT_H * 3:
        return {"region": region, "skipped": "insufficient engineered rows"}

    train, test = feats.iloc[:-HOLDOUT_H], feats.iloc[-HOLDOUT_H:]
    y_train = train["demand_mw"].to_numpy(dtype=float)
    y_test = test["demand_mw"].to_numpy(dtype=float)
    x_train = _get_exog(train)
    x_test = _get_exog(test)

    # The search slice production uses (last 504 rows), so the orders this
    # study compares are the orders production would actually select.
    sub = 504
    y_sub = y_train[-sub:] if len(y_train) > sub else y_train
    x_sub = x_train[-sub:] if x_train is not None and len(x_train) > sub else x_train

    out: dict[str, Any] = {"region": region, "n_train": len(y_train), "n_test": len(y_test)}
    for arm, use_exog in (("A_univariate", False), ("B_exog_aware", True)):
        try:
            order, seasonal, secs = _select_order(y_sub, x_sub, use_exog=use_exog)
        except Exception as e:  # pragma: no cover - study script
            out[arm] = {"error": str(e)[:120]}
            continue
        smape = _fit_and_score(y_train, x_train, y_test, x_test, order, seasonal)
        out[arm] = {
            "order": list(order),
            "seasonal_order": list(seasonal),
            "search_s": round(secs, 1),
            "smape": None if smape is None else round(smape, 3),
        }
    a, b = out.get("A_univariate", {}), out.get("B_exog_aware", {})
    if a.get("smape") is not None and b.get("smape") is not None:
        out["delta_smape_pts"] = round(a["smape"] - b["smape"], 3)  # + = fix is better
        out["order_changed"] = (
            a["order"] != b["order"] or a["seasonal_order"] != b["seasonal_order"]
        )
    return out


def main() -> int:
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default=",".join(DEFAULT_REGIONS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="docs/ARIMA_ORDER_EXOG_STUDY.json")
    args = ap.parse_args()

    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        print("EIA_API_KEY not set")
        return 2

    if args.all:
        from config import REGION_COORDINATES

        regions = sorted(REGION_COORDINATES)
    else:
        regions = [r.strip() for r in args.regions.split(",") if r.strip()]

    results = []
    for r in regions:
        print(f"[{r}] ...", flush=True)
        try:
            res = study_region(r, api_key)
        except Exception as e:
            res = {"region": r, "error": str(e)[:160]}
        if res:
            results.append(res)
            print("   ", json.dumps({k: v for k, v in res.items() if k != "region"})[:200])

    scored = [r for r in results if r.get("delta_smape_pts") is not None]
    summary = {
        "n_regions": len(results),
        "n_scored": len(scored),
        "n_order_changed": sum(1 for r in scored if r.get("order_changed")),
        "n_fix_better": sum(1 for r in scored if r["delta_smape_pts"] > 0),
        "n_fix_worse": sum(1 for r in scored if r["delta_smape_pts"] < 0),
        "median_delta_pts": (
            round(float(np.median([r["delta_smape_pts"] for r in scored])), 3) if scored else None
        ),
        "mean_delta_pts": (
            round(float(np.mean([r["delta_smape_pts"] for r in scored])), 3) if scored else None
        ),
        "median_search_s_A": (
            round(float(np.median([r["A_univariate"]["search_s"] for r in scored])), 1)
            if scored
            else None
        ),
        "median_search_s_B": (
            round(float(np.median([r["B_exog_aware"]["search_s"] for r in scored])), 1)
            if scored
            else None
        ),
    }
    payload = {"summary": summary, "regions": results}
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\n== SUMMARY ==")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
