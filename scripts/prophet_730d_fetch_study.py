"""#231 follow-up: does extending the fetch to re-enable Prophet's yearly_seasonality help?

Companion to ``scripts/xgboost_730d_fetch_study.py`` (real, cheap win, 21/28
BAs) and ``scripts/arima_730d_fetch_study.py`` (no win — systematic
under-forecast bias in 5/6 BAs). This is the model #231 was actually about:
``models/prophet_model.py`` auto-enables ``yearly_seasonality`` once a
training span clears ``YEARLY_SEASONALITY_MIN_DAYS = 730`` (``train_prophet``
computes ``span_days`` from the training data itself — nothing to force by
hand). Below that, #281 found the yearly Fourier term extrapolates a
spurious −11.8 GW phantom decline; the weather regressors already carry the
annual signal in the meantime.

**A fetch-window arithmetic trap, caught before running anything:** a naive
730-day fetch would NOT have cleared the gate in this backtest. Feature
engineering drops a 7-day warmup, and each of the 8 rolling-origin windows
carves back another week of holdout — the smallest (oldest) window's
training span off a 730-day fetch works out to just 667 days, still under
the gate. Every window in the "treatment" arm would have silently kept
``yearly_seasonality`` OFF, making it identical to the 90-day baseline and
guaranteeing a fake "no difference" verdict. This script fetches 800 days
instead — comfortably clearing the gate (737d) even for the smallest window
— as a STUDY-DESIGN artifact to exercise the flag across all 8 windows, not
a claim that production should fetch 800 days (production only needs to
clear 730 once, since it fits once per day, not across 8 backtest windows).
Every window logs whether ``yearly_seasonality`` actually activated, so this
doesn't rely on the arithmetic being right after the fact.

**What this study does NOT test.** Every holdout window here uses REAL
historical weather for the test period (`engineer_features` on real EIA +
ERA5 archive data), matching how `models/training.py`'s own validation
split works. That means this backtest answers "does yearly_seasonality
help when Prophet gets enough real history" — the literal #231 proposal —
but it does NOT exercise the separate redundancy risk flagged in
`prophet_model.py`'s own docstring: weather regressors already carry the
annual signal, and #283's weather-normal-tail overlay (serving-time only,
days 17-30, climatology-derived weather standing in for the real forecast)
could double-count with yearly_seasonality specifically in that degraded-
weather regime. Testing THAT needs a design that simulates climatology-
derived exog for the tail, matching the real serve path — this script
doesn't attempt it.

Same evaluation contract as the other two studies: rolling origin (8
windows), WAPE optimizing metric, MAPE-regression/bias satisficing
constraints, ``verdict()`` from ``models/rolling_eval.py`` allowed to come
back inconclusive. No train/holdout MAE overfitting-gap check (same
limitation as the SARIMAX script — Prophet's fit object isn't cheaply
probed for in-sample residuals here); the rolling-origin holdout is the
primary defense against overfitting in this framework regardless.

No writes to Redis, GCS, or ``latest.json`` — reads real EIA/Open-Meteo
data and fits local, throwaway Prophet models.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

warnings.filterwarnings("ignore")

from data.eia_client import fetch_demand  # noqa: E402
from data.feature_engineering import engineer_features  # noqa: E402
from data.preprocessing import merge_demand_weather  # noqa: E402
from data.weather_client import fetch_historical_weather  # noqa: E402
from models.evaluation import compute_mape  # noqa: E402
from models.prophet_model import (  # noqa: E402
    YEARLY_SEASONALITY_MIN_DAYS,
    predict_prophet,
    train_prophet,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402
from _study_guards import assert_arms_differ  # noqa: E402

from models.rolling_eval import (  # noqa: E402
    bias_pct,
    rolling_origin_splits,
    satisficing_check,
    verdict,
    wape,
)

log = structlog.get_logger()

# Same 6-BA sample as the XGBoost/SARIMAX studies, for direct comparability.
SAMPLE_BAS = {
    "founding8_full_history": ["CAISO", "ERCOT"],
    "v1_alpha": ["TVA", "DUK"],
    "v3_zeta_short_history": ["SCEG", "CPLW"],
}

# 800d, not 730d -- see module docstring for why a literal 730-day fetch
# would silently never clear the yearly_seasonality gate in this backtest.
FETCH_WINDOWS_DAYS = {"baseline_90d": 90, "treatment_800d": 800}

TRAINING_JOB_TIMEOUT_S = 18000  # deploy-prod.yml gridpulse-training-job --task-timeout
N_WINDOWS = 8  # docs/EVALUATION_POLICY.md default
HOLDOUT_H = 168  # matches models/training.py validation_hours default

#: Report cumulative-to-lead, not pooled over the whole holdout. An earlier
#: version of this study reported ONE number averaged across all 168 hours --
#: but 144 of those (86%) sit beyond day-ahead while models/benchmark.py sets
#: HEADLINE_LEAD = "24h". Re-reporting the XGBoost study per lead removed EVERY
#: satisficing veto (SCEG 6/8 and ERCOT 7/8 bias failures at 168h -> 0 at 24h),
#: because the systematic under-forecast was recursive drift over days 2-7, not
#: a property of the training window. The same collapse may be hiding the same
#: thing here -- especially for Prophet, whose #281 failure mode is explicitly
#: long-horizon extrapolation.
LEADS = (24, 48, 168)
MIN_TRAIN_H = 720  # matches train_all_models' "need at least 30 days" floor


def _build_dataset(region: str, days: int, end: pd.Timestamp) -> tuple[pd.DataFrame | None, float]:
    """Fetch + merge + feature-engineer one (region, window) arm. Returns (df, fetch_seconds)."""
    start = (end - pd.Timedelta(days=days)).strftime("%Y-%m-%dT00")
    end_s = end.strftime("%Y-%m-%dT23")
    t0 = time.time()
    demand = fetch_demand(region, start=start, end=end_s, use_cache=True)
    if demand.empty:
        return None, time.time() - t0
    weather = fetch_historical_weather(region, start[:10], end_s[:10], use_cache=True)
    fetch_s = time.time() - t0
    if weather.empty:
        return None, fetch_s
    merged = merge_demand_weather(demand, weather)
    if merged.empty:
        return None, fetch_s
    return engineer_features(merged), fetch_s


def _span_days(df: pd.DataFrame) -> int:
    """Same formula train_prophet uses internally to gate yearly_seasonality."""
    ts = df["timestamp"]
    return int((ts.max() - ts.min()) / pd.Timedelta(days=1)) if len(ts) else 0


def _rolling_score(df: pd.DataFrame) -> dict:
    """Train/predict/score across rolling-origin windows."""
    splits = rolling_origin_splits(
        len(df), n_windows=N_WINDOWS, holdout_h=HOLDOUT_H, min_train_h=MIN_TRAIN_H
    )
    out = {f"{k}_{L}": [] for k in ("wape", "mape", "bias") for L in LEADS}
    out["yearly_on"] = []
    out["train_s"] = []
    for train_slice, test_slice in splits:
        train_df = df.iloc[train_slice]
        test_df = df.iloc[test_slice]
        span = _span_days(train_df)
        yearly_on = span >= YEARLY_SEASONALITY_MIN_DAYS
        t0 = time.time()
        try:
            model = train_prophet(train_df)
            pred = predict_prophet(model, test_df, periods=len(test_df))
        except Exception as e:  # noqa: BLE001
            log.warning("window_train_failed", error=str(e))
            continue
        out["train_s"].append(time.time() - t0)
        out["yearly_on"].append(yearly_on)
        y_test = test_df["demand_mw"].to_numpy()
        pred_test = pred["forecast"]
        for lead in LEADS:
            k = min(lead, len(y_test))
            out[f"wape_{lead}"].append(wape(y_test[:k], pred_test[:k]))
            out[f"mape_{lead}"].append(compute_mape(y_test[:k], pred_test[:k]))
            out[f"bias_{lead}"].append(bias_pct(y_test[:k], pred_test[:k]))
    return out


def run(region: str, end: pd.Timestamp) -> dict | None:
    print(f"\n== {region} ==")
    datasets, fetch_times = {}, {}
    for label, days in FETCH_WINDOWS_DAYS.items():
        df, fetch_s = _build_dataset(region, days, end)
        fetch_times[label] = fetch_s
        if df is None or len(df) < MIN_TRAIN_H + HOLDOUT_H:
            rows = 0 if df is None else len(df)
            print(f"  {label}: insufficient data ({rows} rows post-feature-eng) — skipping region")
            return None
        datasets[label] = df
        print(f"  {label}: {len(df)} rows, span {_span_days(df)}d, fetch {fetch_s:.1f}s")

    t0 = time.time()
    control = _rolling_score(datasets["baseline_90d"])
    control_wall = time.time() - t0
    t0 = time.time()
    treatment = _rolling_score(datasets["treatment_800d"])
    treatment_wall = time.time() - t0

    n = min(len(control["wape_168"]), len(treatment["wape_168"]))
    if n == 0:
        print("  no comparable windows — skipping")
        return None

    yearly_on_count = sum(treatment["yearly_on"][:n])
    if yearly_on_count < n:
        print(
            f"  WARNING: yearly_seasonality only activated in {yearly_on_count}/{n} "
            f"treatment windows — fetch buffer may be too thin for this region's data availability"
        )
    else:
        print(f"  yearly_seasonality confirmed ON in all {n} treatment windows")

    # If the treatment is byte-identical to the control the arms are a no-op
    # (e.g. a fetch window that never actually clears a feature's gate) and
    # every verdict below would be meaningless.
    assert_arms_differ(
        "wape_168",
        {
            "control": np.array(control["wape_168"][:n]),
            "treatment": np.array(treatment["wape_168"][:n]),
        },
    )

    for lead in LEADS:
        deltas = np.array(control[f"wape_{lead}"][:n]) - np.array(treatment[f"wape_{lead}"][:n])
        v = verdict(deltas)
        sat = satisficing_check(
            treatment_bias_pct=float(np.mean(treatment[f"bias_{lead}"][:n])),
            control_mape=float(np.mean(control[f"mape_{lead}"][:n])),
            treatment_mape=float(np.mean(treatment[f"mape_{lead}"][:n])),
        )
        tag = "  <-- benchmark HEADLINE_LEAD" if lead == 24 else ""
        print(
            f"  [{lead:>3d}h] ctl {np.mean(control[f'wape_{lead}'][:n]):.2f}% -> "
            f"trt {np.mean(treatment[f'wape_{lead}'][:n]):.2f}%  "
            f"bias {np.mean(treatment[f'bias_{lead}'][:n]):+.2f}%  "
            f"{'PASS' if sat['passed'] else 'SAT-FAIL'}  |  {v['reason']}{tag}"
        )
    print(f"  windows compared: {n}")
    print(
        f"  wall-clock: 90d={control_wall:.0f}s ({n} windows), 800d={treatment_wall:.0f}s ({n} windows)"
    )

    return {
        "region": region,
        "n_windows": n,
        "yearly_on_count": yearly_on_count,
        "fetch_times": fetch_times,
        "control_wall": control_wall,
        "treatment_wall": treatment_wall,
    }


def main():
    # 5-day lag: avoids the recent-data-incompleteness / forecast-endpoint-throttle
    # trap noted for local re-measurement (memory: gridpulse-local-remeasure).
    end = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=5)
    regions = (
        sys.argv[1].split(",")
        if len(sys.argv) > 1
        else [r for tier in SAMPLE_BAS.values() for r in tier]
    )
    print(f"Regions: {regions} (excludes the other 45 BAs - see SAMPLE_BAS tiers for rationale)")
    print("Fetch windows: baseline=90d, treatment=800d (see module docstring for why not 730d)")

    results = []
    for region in regions:
        try:
            r = run(region, end)
        except Exception as e:  # noqa: BLE001
            print(f"{region}: ERROR {e}")
            continue
        if r:
            results.append(r)

    print("\n" + "=" * 70)
    print("Per-lead results are reported per BA above; 24h is the benchmark headline lead.")

    fetch_90 = [r["fetch_times"]["baseline_90d"] for r in results]
    fetch_800 = [r["fetch_times"]["treatment_800d"] for r in results]
    train_90 = [r["control_wall"] for r in results]
    train_800 = [r["treatment_wall"] for r in results]
    if fetch_90 and fetch_800:
        avg_f90, avg_f800 = float(np.mean(fetch_90)), float(np.mean(fetch_800))
        avg_t90, avg_t800 = float(np.mean(train_90)), float(np.mean(train_800))
        total_800_51 = (avg_f800 + avg_t800) * 51
        print(
            f"\nPer-BA avg: fetch {avg_f90:.1f}s->{avg_f800:.1f}s, "
            f"train (8 windows) {avg_t90:.0f}s->{avg_t800:.0f}s"
        )
        print(
            f"Extrapolated x51 BAs, 8-window backtest cost (NOT production's single-fit-per-day "
            f"cost): {total_800_51:.0f}s vs training job's {TRAINING_JOB_TIMEOUT_S}s budget "
            f"({total_800_51 / TRAINING_JOB_TIMEOUT_S:.1%})"
        )
    if not results:
        print("No regions produced comparable results.")


if __name__ == "__main__":
    main()
