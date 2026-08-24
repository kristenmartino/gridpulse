"""#231 follow-up: is a 730-day XGBoost training fetch a real win?

Issue #231 proposed extending the training fetch to 365 days (Prophet-scoped)
to re-enable Prophet's ``yearly_seasonality``. Investigation found that fix
doesn't hold up as scoped: the real threshold is 730 days
(``models/prophet_model.py:YEARLY_SEASONALITY_MIN_DAYS``), most BAs don't
have 730 days of demand history yet regardless of fetch window, and #283
already evaluated "re-enable yearly seasonality on more data" and chose a
different fix (the weather-normal tail overlay) instead.

The one candidate with a clean, unstacked possible win is XGBoost: none of
its feature windows (max 168h lag/rolling, 30d ``temperature_deviation``)
depend on more than 90 days, but more history means more distinct
summer/winter/shoulder examples to train on. This script measures whether
that's actually true, or whether it's overfitting / not worth the added
fetch cost — per ``docs/EVALUATION_POLICY.md`` (rolling origin, WAPE as the
optimizing metric, MAPE-regression and bias as satisficing constraints,
``verdict()`` allowed to come back inconclusive) rather than a naive
before/after read of one metric.

**Proxy, not the production pipeline** — like ``phase0_weather_normal_backtest.py``,
this is a go/no-go measurement, not a drop-in replacement for the real
recursive-forecast serve path (ADR-013's lesson: a scenario/production
comparison across two different inference paths reports the gap between the
paths, not the thing being tested). Both arms here score one-shot
predictions on held-out rows with real (not recursively-forecast) lag
features, so absolute error is understated versus production — but the
understatement applies equally to both arms, so the paired WAPE delta stays
a fair comparison.

Two questions, both required:
1. Is the WAPE win real (rolling-origin verdict + satisficing constraints),
   not noise from one window? Does a train-vs-holdout MAE gap widen for the
   730d arm (the overfitting signature) even where WAPE improves?
2. What does the extra fetch actually cost in wall-clock, measured (not
   assumed) against the training job's real 18000s (5h) task-timeout budget?

No writes to Redis, GCS, or ``latest.json`` — this only reads real EIA/
Open-Meteo data and trains local, throwaway XGBoost models.
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
from models.evaluation import compute_mape  # noqa: E402
from models.rolling_eval import (  # noqa: E402
    bias_pct,
    rolling_origin_splits,
    satisficing_check,
    verdict,
    wape,
)
from models.xgboost_model import predict_xgboost, train_xgboost  # noqa: E402

log = structlog.get_logger()

# Coverage tiers give the overfitting control for free: if only the
# full-history tier improves and the short-history tier doesn't, that's
# expected (more data helping), not a confound.
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
    """Train/predict/score across rolling-origin windows. Newest window first."""
    splits = rolling_origin_splits(
        len(df), n_windows=N_WINDOWS, holdout_h=HOLDOUT_H, min_train_h=MIN_TRAIN_H
    )
    out = {"wape": [], "mape": [], "bias": [], "train_mae": [], "holdout_mae": []}
    for train_slice, test_slice in splits:
        train_df = df.iloc[train_slice]
        test_df = df.iloc[test_slice]
        try:
            result = train_xgboost(train_df, cross_validate=False)
            pred_train = predict_xgboost(result, train_df)
            pred_test = predict_xgboost(result, test_df)
        except Exception as e:  # noqa: BLE001
            log.warning("window_train_failed", error=str(e))
            continue
        y_train = train_df["demand_mw"].to_numpy()
        y_test = test_df["demand_mw"].to_numpy()
        out["wape"].append(wape(y_test, pred_test))
        out["mape"].append(compute_mape(y_test, pred_test))
        out["bias"].append(bias_pct(y_test, pred_test))
        out["train_mae"].append(float(np.mean(np.abs(y_train - pred_train))))
        out["holdout_mae"].append(float(np.mean(np.abs(y_test - pred_test))))
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

    control = _rolling_score(datasets["baseline_90d"])
    treatment = _rolling_score(datasets["treatment_730d"])

    n = min(len(control["wape"]), len(treatment["wape"]))
    if n == 0:
        print("  no comparable windows — skipping")
        return None

    # Positive delta = treatment (730d) better, per rolling_eval's convention.
    deltas = np.array(control["wape"][:n]) - np.array(treatment["wape"][:n])
    v = verdict(deltas)
    sat = satisficing_check(
        treatment_bias_pct=float(np.mean(treatment["bias"][:n])),
        control_mape=float(np.mean(control["mape"][:n])),
        treatment_mape=float(np.mean(treatment["mape"][:n])),
    )

    control_gap = float(np.mean(control["holdout_mae"][:n]) - np.mean(control["train_mae"][:n]))
    treat_gap = float(np.mean(treatment["holdout_mae"][:n]) - np.mean(treatment["train_mae"][:n]))
    widened = treat_gap > control_gap

    print(f"  windows compared: {n}")
    print(f"  verdict: {v['reason']}")
    print(f"  satisficing: {'PASS' if sat['passed'] else 'FAIL - ' + '; '.join(sat['failures'])}")
    print(
        f"  train/holdout MAE gap: 90d={control_gap:.1f} MW, 730d={treat_gap:.1f} MW"
        f" ({'WIDENED - possible overfitting' if widened else 'not widened'})"
    )

    return {
        "region": region,
        "n_windows": n,
        "verdict": v,
        "satisficing": sat,
        "gap_widened": widened,
        "fetch_times": fetch_times,
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
        if v["decisive"] and v["winner"] == "treatment" and sat["passed"] and not r["gap_widened"]:
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
    if fetch_90 and fetch_730:
        avg_90, avg_730 = float(np.mean(fetch_90)), float(np.mean(fetch_730))
        est_90_51, est_730_51 = avg_90 * 51, avg_730 * 51
        print(
            f"\nFetch cost: avg {avg_90:.1f}s (90d) -> {avg_730:.1f}s (730d) per BA "
            f"({avg_730 / max(avg_90, 0.01):.1f}x)."
        )
        print(
            f"Extrapolated x51 BAs (sequential, no concurrency assumed): "
            f"{est_90_51:.0f}s -> {est_730_51:.0f}s "
            f"vs training job's {TRAINING_JOB_TIMEOUT_S}s task-timeout budget "
            f"({est_730_51 / TRAINING_JOB_TIMEOUT_S:.1%} of budget)."
        )
    if not results:
        print("No regions produced comparable results.")


if __name__ == "__main__":
    main()
