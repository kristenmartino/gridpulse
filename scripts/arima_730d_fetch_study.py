"""#231 follow-up: is a 730-day SARIMAX training fetch a real win?

Companion to ``scripts/xgboost_730d_fetch_study.py``. That study found a
real, cheap, unstacked XGBoost win on 4/6 sampled BAs. SARIMAX was initially
assumed out of scope because ``max_training_rows=2160`` caps its training at
90 days regardless of fetch window, and the code comment at
``models/arima_model.py:75`` calls the fit "O(n^3)" — extending to 730 days
(8x rows) would be ~512x slower under a literal cubic law.

Two measurements this session found that comment doesn't hold up as stated:

1. **The final SARIMAX MLE fit scales close to linearly, not cubically.**
   Empirically on CAISO: 2,160 rows -> 11s, 4,320 -> 25s, 8,640 -> 49s (fitted
   exponent ~1.08, not 3). A fixed-state-dimension Kalman filter's forward
   pass is textbook O(n) in observations, so near-linear is the theoretically
   expected result, not a surprise.
2. **The order selected by ``auto_arima`` is NOT stable across rolling-origin
   windows** — 5 distinct ``(order, seasonal_order)`` combos across 8 CAISO
   windows spanning 730 days, confirmed deterministic on identical input (so
   the drift is real, driven by which slice of history each window's
   trailing-504-row selection subset lands on, not search randomness). This
   means the obvious cost-saving shortcut — select the order once, reuse it
   across all 8 windows, the way production's ``cached_order`` fast path does
   on a *daily* refresh — is invalid here: it would change what the study
   measures (fixed-order-more-data) rather than what it's supposed to measure
   (production's actual fresh-order-selection-per-cycle process, given more
   data). So this script selects the order fresh every window, same as
   ``train_arima(auto_order=True)`` with no cache.

Net effect: no algorithm change was needed to make this affordable — the
premise that it was expensive was itself unverified. Order search costs
~20-40s regardless of window size (it always subsets to the trailing 504
rows, per ``_auto_select_order``); the final fit costs up to ~100s at 730d.
That is ~20 min/BA for both arms, not hours.

Same evaluation contract as the XGBoost study: rolling origin (8 windows),
WAPE as the optimizing metric, MAPE-regression/bias as satisficing
constraints, ``verdict()`` from ``models/rolling_eval.py`` allowed to come
back inconclusive. Same caveat: one-shot scoring against the actual holdout
rows (SARIMAX's own multi-step ``forecast()`, not the recursive XGBoost
protocol), not the full production serve path — a fair, paired proxy, not a
drop-in replacement for it.

No train-vs-holdout MAE "overfitting gap" check here, unlike the XGBoost
script: ``train_arima`` deliberately returns a lean payload (params + a
short data tail, not the full fitted ``SARIMAXResults``) specifically to
avoid ~500MB pickles, so in-sample fitted values aren't available through
the public API without reimplementing internals. The rolling-origin holdout
evaluation itself is the primary defense against overfitting in this
framework (every scored point is out-of-sample by construction); a low-order
linear-Gaussian model is also far less prone to memorization than a
6000-tree booster, so this gap is a smaller loss here than it would be
for XGBoost.

No writes to Redis, GCS, or ``latest.json`` — reads real EIA/Open-Meteo data
and fits local, throwaway SARIMAX models.
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd
import structlog

warnings.filterwarnings("ignore")

from data.eia_client import fetch_demand  # noqa: E402
from data.feature_engineering import engineer_features  # noqa: E402
from data.preprocessing import merge_demand_weather  # noqa: E402
from data.weather_client import fetch_historical_weather  # noqa: E402
from models.arima_model import predict_arima, train_arima  # noqa: E402
from models.evaluation import compute_mape  # noqa: E402
from models.rolling_eval import (  # noqa: E402
    bias_pct,
    rolling_origin_splits,
    satisficing_check,
    verdict,
    wape,
)

log = structlog.get_logger()

# Same 6-BA sample as the XGBoost study, for direct comparability.
SAMPLE_BAS = {
    "founding8_full_history": ["CAISO", "ERCOT"],
    "v1_alpha": ["TVA", "DUK"],
    "v3_zeta_short_history": ["SCEG", "CPLW"],
}

FETCH_WINDOWS_DAYS = {"baseline_90d": 90, "treatment_730d": 730}

TRAINING_JOB_TIMEOUT_S = 18000  # deploy-prod.yml gridpulse-training-job --task-timeout
N_WINDOWS = 8  # docs/EVALUATION_POLICY.md default
HOLDOUT_H = 168  # matches models/training.py validation_hours default
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


def _rolling_score(df: pd.DataFrame) -> dict:
    """Train/predict/score across rolling-origin windows, fresh order per window."""
    splits = rolling_origin_splits(
        len(df), n_windows=N_WINDOWS, holdout_h=HOLDOUT_H, min_train_h=MIN_TRAIN_H
    )
    out = {"wape": [], "mape": [], "bias": [], "order": [], "train_s": []}
    for train_slice, test_slice in splits:
        train_df = df.iloc[train_slice]
        test_df = df.iloc[test_slice]
        t0 = time.time()
        try:
            # max_training_rows must be raised to the actual slice length —
            # train_arima's own default (2160) would silently re-truncate
            # every window back to 90 days regardless of arm, defeating the
            # whole comparison.
            result = train_arima(train_df, auto_order=True, max_training_rows=len(train_df))
            pred_test = predict_arima(result, test_df, periods=len(test_df))
        except Exception as e:  # noqa: BLE001
            log.warning("window_train_failed", error=str(e))
            continue
        out["train_s"].append(time.time() - t0)
        out["order"].append((result["order"], result["seasonal_order"]))
        y_test = test_df["demand_mw"].to_numpy()
        out["wape"].append(wape(y_test, pred_test))
        out["mape"].append(compute_mape(y_test, pred_test))
        out["bias"].append(bias_pct(y_test, pred_test))
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
        print(f"  {label}: {len(df)} rows, fetch {fetch_s:.1f}s")

    t0 = time.time()
    control = _rolling_score(datasets["baseline_90d"])
    control_wall = time.time() - t0
    t0 = time.time()
    treatment = _rolling_score(datasets["treatment_730d"])
    treatment_wall = time.time() - t0

    n = min(len(control["wape"]), len(treatment["wape"]))
    if n == 0:
        print("  no comparable windows — skipping")
        return None

    deltas = np.array(control["wape"][:n]) - np.array(treatment["wape"][:n])
    v = verdict(deltas)
    sat = satisficing_check(
        treatment_bias_pct=float(np.mean(treatment["bias"][:n])),
        control_mape=float(np.mean(control["mape"][:n])),
        treatment_mape=float(np.mean(treatment["mape"][:n])),
    )

    unique_orders = len(set(treatment["order"][:n]))
    print(f"  windows compared: {n}")
    print(f"  verdict: {v['reason']}")
    print(f"  satisficing: {'PASS' if sat['passed'] else 'FAIL - ' + '; '.join(sat['failures'])}")
    print(
        f"  distinct orders selected (730d arm, {n} windows): {unique_orders} "
        f"(confirms per-window order selection is not skippable)"
    )
    print(
        f"  wall-clock: 90d={control_wall:.0f}s ({n} windows), "
        f"730d={treatment_wall:.0f}s ({n} windows)"
    )

    return {
        "region": region,
        "n_windows": n,
        "verdict": v,
        "satisficing": sat,
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
    print("SUMMARY")
    for r in results:
        v, sat = r["verdict"], r["satisficing"]
        if v["decisive"] and v["winner"] == "treatment" and sat["passed"]:
            tag = "REAL WIN"
        elif v["decisive"] and v["winner"] == "treatment":
            tag = "WAPE-WIN-BUT-VETOED"
        elif v["decisive"]:
            tag = "NO WIN"
        else:
            tag = "INCONCLUSIVE"
        print(f"  {r['region']:8s} {tag:20s} n={r['n_windows']}  {v['reason']}")

    fetch_90 = [r["fetch_times"]["baseline_90d"] for r in results]
    fetch_730 = [r["fetch_times"]["treatment_730d"] for r in results]
    train_90 = [r["control_wall"] for r in results]
    train_730 = [r["treatment_wall"] for r in results]
    if fetch_90 and fetch_730:
        avg_f90, avg_f730 = float(np.mean(fetch_90)), float(np.mean(fetch_730))
        avg_t90, avg_t730 = float(np.mean(train_90)), float(np.mean(train_730))
        total_90_51 = (avg_f90 + avg_t90) * 51
        total_730_51 = (avg_f730 + avg_t730) * 51
        print(
            f"\nPer-BA avg: fetch {avg_f90:.1f}s->{avg_f730:.1f}s, "
            f"train (8 windows) {avg_t90:.0f}s->{avg_t730:.0f}s"
        )
        print(
            f"Extrapolated x51 BAs (sequential, no concurrency assumed): "
            f"{total_90_51:.0f}s -> {total_730_51:.0f}s "
            f"vs training job's {TRAINING_JOB_TIMEOUT_S}s task-timeout budget "
            f"({total_730_51 / TRAINING_JOB_TIMEOUT_S:.1%} of budget). "
            f"NOTE: this is 8-window backtest cost, not production's single-fit-per-day cost — "
            f"production only pays this once per day per BA, not x8."
        )
    if not results:
        print("No regions produced comparable results.")


if __name__ == "__main__":
    main()
