"""#231 follow-up, CORRECTED: does a 730-day XGBoost training window beat 90 days?

Supersedes ``scripts/xgboost_730d_fetch_study.py``. That first pass reported
21/28 BAs as decisive wins; two methodology defects made that number
untrustworthy, and both are fixed here.

**Defect 1 — the control arm was a strawman.** ``rolling_origin_splits``
returns ``slice(0, test_start)``: train on *everything* before the holdout.
Fed a 90-day dataset, the newest window trained on ~83 days but the oldest
trained on ~35 — so 6 of 7 windows compared a full-history treatment against
a control that was starved of data, not one trained the way production
trains. The measured gap conflated "more history helps" with "the control
was crippled." Here BOTH arms use a FIXED trailing window (90d vs 730d) and
the identical test windows, so training-window length is the only variable.

**Defect 2 — the scoring regime was wrong.** The first pass scored one-shot
predictions against holdout rows carrying REAL ``demand_lag_1h`` values.
``demand_lag_1h`` was the top feature in every fit. Production forecasts
recursively — feeding the model its own predictions as lags, up to 720h out
— and ADR-010 exists precisely because the non-recursive holdout is blind to
how a model behaves in that regime (~27% of persisted LDWP vintages dive in
the serve path while the holdout looks fine). This version scores through
``recursive_autoregressive_forecast``, the shared protocol
``models/training.py`` itself uses, so the number is commensurable with
production's own holdout.

**Dropped: the train/holdout MAE gap check.** With 6,000 trees at depth 8 and
no early stopping on the final fit (``models/xgboost_model.py`` — "Train
final model on all data (no early stopping — no validation set)"), the 90d
arm memorizes its training set: measured train_MAE of **0.1 MW** against
~30,000 MW demand on CAISO, versus 17.6 MW for the 730d arm. The two arms
sit in different memorization regimes, so their gaps were never comparable
quantities. Decomposing that CAISO gap showed the shrink came from holdout
error falling (−144 MW), not train error rising (+17.5) — so the finding
survived, but the instrument was invalid and is not reused. The rolling
holdout is the overfitting defense here.

Same evaluation contract as before and as the Prophet/SARIMAX companions:
rolling origin, WAPE optimizing, bias + MAPE-regression satisficing,
``verdict()`` from ``models/rolling_eval.py`` free to return inconclusive.

Both arms are built from ONE fetched dataset per BA, so the test windows are
byte-identical between arms and no difference in feature-engineering warmup
or data availability can leak into the comparison.

No writes to Redis, GCS, or ``latest.json``.
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd
import structlog

warnings.filterwarnings("ignore")

from data.eia_client import fetch_demand  # noqa: E402
from data.feature_engineering import (  # noqa: E402
    engineer_exogenous_features,
    engineer_features,
    recursive_autoregressive_forecast,
)
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

SAMPLE_BAS = ["CAISO", "ERCOT", "TVA", "DUK", "SCEG", "CPLW"]

CONTROL_DAYS = 90  # production's current effective window (data/eia_client.py default)
TREATMENT_DAYS = 730
CONTROL_ROWS = CONTROL_DAYS * 24
TREATMENT_ROWS = TREATMENT_DAYS * 24

N_WINDOWS = 8  # docs/EVALUATION_POLICY.md default
HOLDOUT_H = 168  # matches models/training.py validation_hours default

# One fetch must cover: the treatment train window + every window's holdout
# carve-back + feature-engineering's 7-day warmup drop. 820d leaves ~27d of
# margin so a BA with minor gaps still yields the full 8 windows.
FETCH_DAYS = 820
TRAINING_JOB_TIMEOUT_S = 18000  # deploy-prod.yml gridpulse-training-job --task-timeout


def _build_dataset(region: str, end: pd.Timestamp) -> tuple[pd.DataFrame | None, float]:
    """Fetch + merge + feature-engineer ONE dataset serving both arms."""
    start = (end - pd.Timedelta(days=FETCH_DAYS)).strftime("%Y-%m-%dT00")
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


def _paired_splits(n_rows: int) -> list[tuple[slice, slice, slice]]:
    """(control_train, treatment_train, test) with a FIXED trailing train window each.

    Window placement is delegated to the tested ``rolling_origin_splits``;
    ``min_train_h=TREATMENT_ROWS`` makes it drop any window lacking a full
    730-day history, so both arms are defined on exactly the same test
    windows and the pairing in ``paired_deltas`` stays honest.
    """
    splits = rolling_origin_splits(
        n_rows, n_windows=N_WINDOWS, holdout_h=HOLDOUT_H, min_train_h=TREATMENT_ROWS
    )
    out = []
    for _train_slice, test_slice in splits:
        test_start = test_slice.start
        if test_start - TREATMENT_ROWS < 0:
            continue
        out.append(
            (
                slice(test_start - CONTROL_ROWS, test_start),
                slice(test_start - TREATMENT_ROWS, test_start),
                test_slice,
            )
        )
    return out


def _score(df: pd.DataFrame, train_slice: slice, test_slice: slice) -> tuple | None:
    """Fit one arm on one window and score it the way production scores.

    Mirrors ``models/training.py``: exogenous-only future frame, then the
    shared recursive protocol chaining the model's own predictions as lags.
    ``cross_validate=False`` skips the CV boosters whose only output is
    ``cv_scores`` (unused here); the final fitted model is identical.
    """
    train_df = df.iloc[train_slice]
    test_df = engineer_exogenous_features(df.iloc[test_slice].copy())
    try:
        result = train_xgboost(train_df, cross_validate=False)
        y_test = test_df["demand_mw"].to_numpy()
        forecast = recursive_autoregressive_forecast(
            result,
            train_df["demand_mw"].tolist(),
            test_df,
            predict_xgboost,
            seed_timestamps=train_df.get("timestamp"),
        )[: len(y_test)]
    except Exception as e:  # noqa: BLE001
        log.warning("window_failed", error=str(e))
        return None
    return wape(y_test, forecast), compute_mape(y_test, forecast), bias_pct(y_test, forecast)


def run(region: str, end: pd.Timestamp) -> dict | None:
    print(f"\n== {region} ==")
    df, fetch_s = _build_dataset(region, end)
    if df is None:
        print("  no data — skipping")
        return None
    splits = _paired_splits(len(df))
    print(
        f"  {len(df)} rows ({len(df) / 24:.0f}d), fetch {fetch_s:.1f}s, {len(splits)} paired windows"
    )
    if not splits:
        print(f"  insufficient history for a {TREATMENT_DAYS}d train window — skipping")
        return None

    c = {"wape": [], "mape": [], "bias": []}
    t = {"wape": [], "mape": [], "bias": []}
    t0 = time.time()
    for c_train, t_train, test in splits:
        cs, ts = _score(df, c_train, test), _score(df, t_train, test)
        if cs is None or ts is None:
            continue  # drop the PAIR, never one arm alone
        for k, v in zip(("wape", "mape", "bias"), cs, strict=True):
            c[k].append(v)
        for k, v in zip(("wape", "mape", "bias"), ts, strict=True):
            t[k].append(v)
    wall = time.time() - t0

    n = len(c["wape"])
    if n == 0:
        print("  no scored windows — skipping")
        return None

    deltas = np.array(c["wape"]) - np.array(t["wape"])  # +ve = treatment better
    v = verdict(deltas)
    sat = satisficing_check(
        treatment_bias_pct=float(np.mean(t["bias"])),
        control_mape=float(np.mean(c["mape"])),
        treatment_mape=float(np.mean(t["mape"])),
    )
    print(f"  mean WAPE: 90d={np.mean(c['wape']):.2f}%  730d={np.mean(t['wape']):.2f}%")
    print(f"  mean MAPE: 90d={np.mean(c['mape']):.2f}%  730d={np.mean(t['mape']):.2f}%")
    print(f"  mean bias: 90d={np.mean(c['bias']):+.2f}%  730d={np.mean(t['bias']):+.2f}%")
    print(f"  verdict: {v['reason']}")
    print(f"  satisficing: {'PASS' if sat['passed'] else 'FAIL - ' + '; '.join(sat['failures'])}")
    print(f"  wall-clock: {wall:.0f}s for {n} paired windows")
    return {"region": region, "n": n, "verdict": v, "satisficing": sat, "wall": wall}


def main():
    # 5-day lag: avoids recent-data incompleteness / forecast-endpoint throttling.
    end = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=5)
    regions = sys.argv[1].split(",") if len(sys.argv) > 1 else SAMPLE_BAS
    print(
        f"CORRECTED study — fixed trailing windows ({CONTROL_DAYS}d vs {TREATMENT_DAYS}d), "
        f"recursive scoring via models/training.py's protocol"
    )
    print(f"Regions: {regions} (excludes the other {51 - len(regions)} BAs)")

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
            tag = "CONTROL WINS"
        else:
            tag = "INCONCLUSIVE"
        print(f"  {r['region']:8s} {tag:20s} n={r['n']}  {v['reason']}")
    if not results:
        print("  no regions produced comparable results.")


if __name__ == "__main__":
    main()
