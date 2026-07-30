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
#: A window with less training data than this is dropped rather than scored.
MIN_TRAIN_H = 1200

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


def _get_with_retry(url: str, params: dict, *, timeout: int, tries: int = 4):
    """GET with backoff. A fleet run makes 102 upstream calls; a single
    throttled one must fail loudly after retrying, never silently reduce the
    study to a smaller one (the lesson from the ERCO/CISO code typo, which
    read as 'insufficient history' rather than as an error)."""
    last: Exception | None = None
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001 - retried, then re-raised
            last = e
            if attempt < tries - 1:
                time.sleep(2**attempt * 3)
    raise RuntimeError(f"upstream failed after {tries} tries: {last}")


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
        r = _get_with_retry(
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
    r = _get_with_retry(
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
) -> dict | None:
    """Both arms fit WITH exog — only the order under test differs.

    Returns every metric, but only ``wape`` decides: see
    ``models.rolling_eval`` for why the optimising metric is not MAPE.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    from models.evaluation import compute_mape
    from models.rolling_eval import bias_pct, wape

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
        return {
            "wape": round(wape(y_test, pred), 3),
            "mape": round(float(compute_mape(y_test, pred)), 3),
            "smape": round(_smape(y_test, pred), 3),
            "bias_pct": round(bias_pct(y_test, pred), 3),
        }
    except Exception:
        return None


def study_region(region: str, api_key: str, *, n_windows: int = 8) -> dict[str, Any] | None:
    """Both arms over a ROLLING ORIGIN, decided by the harness.

    One 168h window is what reversed CAISO's sign between two adjacent days
    (`docs/ARIMA_ORDER_EXOG_STUDY.md`). Every window re-runs the order search
    on that window's own training slice, which is what production does on a
    cold cache — the order is not carried between windows.
    """
    from config import REGION_COORDINATES
    from data.feature_engineering import engineer_features
    from models.arima_model import _get_exog
    from models.rolling_eval import (
        DECISION_METRIC,
        paired_deltas,
        rolling_origin_splits,
        satisficing_check,
        verdict,
    )

    coords = REGION_COORDINATES.get(region)
    if not coords:
        return None

    end = datetime.now(UTC) - timedelta(days=ARCHIVE_LAG_DAYS)
    # Rolling windows need history for every holdout plus a training slice.
    span = HISTORY_DAYS + (n_windows * HOLDOUT_H) // 24
    start = end - timedelta(days=span)

    demand = _eia_demand(region, start, end, api_key)
    if demand.empty:
        return {"region": region, "skipped": "no demand history"}
    weather = _archive_weather(coords["lat"], coords["lon"], start, end)

    df = demand.merge(weather, on="timestamp", how="inner").sort_values("timestamp")
    feats = engineer_features(df).dropna(subset=["demand_mw"]).reset_index(drop=True)

    splits = rolling_origin_splits(
        len(feats), n_windows=n_windows, holdout_h=HOLDOUT_H, min_train_h=MIN_TRAIN_H
    )
    if not splits:
        return {"region": region, "skipped": f"no windows from {len(feats)} rows"}

    per_window: list[dict] = []
    for train_sl, test_sl in splits:
        train, test = feats.iloc[train_sl], feats.iloc[test_sl]
        y_train = train["demand_mw"].to_numpy(dtype=float)
        y_test = test["demand_mw"].to_numpy(dtype=float)
        x_train, x_test = _get_exog(train), _get_exog(test)

        # The search slice production uses (last 504 rows of THIS window).
        sub = 504
        y_sub = y_train[-sub:] if len(y_train) > sub else y_train
        x_sub = x_train[-sub:] if x_train is not None and len(x_train) > sub else x_train

        w: dict[str, Any] = {"holdout_start": str(test["timestamp"].iloc[0])}
        for arm, use_exog in (("A_univariate", False), ("B_exog_aware", True)):
            try:
                order, seasonal, secs = _select_order(y_sub, x_sub, use_exog=use_exog)
            except Exception as e:  # pragma: no cover - study script
                w[arm] = {"error": str(e)[:120]}
                continue
            scored = _fit_and_score(y_train, x_train, y_test, x_test, order, seasonal)
            w[arm] = {
                "order": list(order),
                "seasonal_order": list(seasonal),
                "search_s": round(secs, 1),
                **(scored or {}),
            }
        a, b = w.get("A_univariate", {}), w.get("B_exog_aware", {})
        if a.get(DECISION_METRIC) is not None and b.get(DECISION_METRIC) is not None:
            w["delta_pts"] = round(a[DECISION_METRIC] - b[DECISION_METRIC], 3)
            w["order_changed"] = a.get("order") != b.get("order") or a.get(
                "seasonal_order"
            ) != b.get("seasonal_order")
        per_window.append(w)

    scored_windows = [w for w in per_window if w.get("delta_pts") is not None]
    deltas = paired_deltas(
        [w["A_univariate"][DECISION_METRIC] for w in scored_windows],
        [w["B_exog_aware"][DECISION_METRIC] for w in scored_windows],
    )
    v = verdict(deltas)

    def _avg(arm: str, key: str) -> float | None:
        vals = [w[arm][key] for w in scored_windows if w[arm].get(key) is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    sat = satisficing_check(
        treatment_bias_pct=_avg("B_exog_aware", "bias_pct"),
        control_mape=_avg("A_univariate", "mape"),
        treatment_mape=_avg("B_exog_aware", "mape"),
    )
    return {
        "region": region,
        "decision_metric": DECISION_METRIC,
        "n_windows_requested": n_windows,
        "n_windows_scored": len(scored_windows),
        "n_order_changed": sum(1 for w in scored_windows if w.get("order_changed")),
        "verdict": v,
        "satisficing": sat,
        # A win requires BOTH: the optimising metric decisive AND every
        # constraint held. Either alone is not a result.
        "ship": bool(v["decisive"] and v["winner"] == "treatment" and sat["passed"]),
        "control_mean": {k: _avg("A_univariate", k) for k in ("wape", "mape", "smape", "bias_pct")},
        "treatment_mean": {
            k: _avg("B_exog_aware", k) for k in ("wape", "mape", "smape", "bias_pct")
        },
        "windows": per_window,
    }


def main() -> int:
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default=",".join(DEFAULT_REGIONS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument(
        "--windows",
        type=int,
        default=8,
        help="rolling-origin windows per BA (1 reproduces the old single-window study)",
    )
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
            res = study_region(r, api_key, n_windows=args.windows)
        except Exception as e:
            res = {"region": r, "error": str(e)[:160]}
        if res:
            results.append(res)
            print("   ", json.dumps({k: v for k, v in res.items() if k != "region"})[:200])
            # Checkpoint every region — a fleet run is ~90 minutes and a crash
            # at region 48 should cost the last region, not all of them.
            with open(args.out, "w") as fh:
                json.dump({"summary": {"partial": True}, "regions": results}, fh, indent=2)

    scored = [r for r in results if isinstance(r.get("verdict"), dict) and r["verdict"]["n"]]
    ship = [r for r in scored if r["ship"]]
    decisive = [r for r in scored if r["verdict"]["decisive"]]
    summary = {
        "decision_metric": "wape",
        "n_regions": len(results),
        "n_scored": len(scored),
        # The headline the old single-window study could not produce: how many
        # BAs the harness is willing to call at all.
        "n_decisive": len(decisive),
        "n_inconclusive": len(scored) - len(decisive),
        "n_ship": len(ship),
        "n_vetoed_by_satisficing": sum(
            1
            for r in scored
            if r["verdict"]["decisive"]
            and r["verdict"]["winner"] == "treatment"
            and not r["satisficing"]["passed"]
        ),
        "median_windows_scored": (
            int(np.median([r["n_windows_scored"] for r in scored])) if scored else None
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
