"""Phase 1: does hierarchical reconciliation improve PER-BA forecast accuracy?

GridPulse forecasts 51 BAs independently. Nothing enforces that they sum
coherently to regional or national totals, and reconciliation (MinT and
friends) exploits that structure to improve the base forecasts themselves —
not merely to make them add up.

**Why per-BA accuracy is the question.** We serve per-BA forecasts. A method
that only sharpens aggregates would be worthless here. Primary sources are
encouraging on exactly this point: Brégère & Huard (EDF/INRIA, 1,545 UK
households, 136-node hierarchy) report bottom-level MSE improvements of 8.0%
(GAM base), 18.8% (Random Forest), 14.7% (autoregressive) — comparable to or
larger than their total-consumption gains. Wickramasuriya et al. (2019, the
original MinT paper) corroborate for MinT specifically, though on tourism
rather than energy. Caveats carried into this study: those are MSE (the most
flattering scale — ~13% MSE is ~6.7% RMSE); Brégère & Huard used OLS
projection rather than MinT-shrink; and the review literature reports **no
quantitative effect size** for any published MinT-on-electricity application.
So this measures our own hierarchy rather than inheriting a number.

**The summation premise was verified before this script was written**
(2026-08-21): sum of the 51 BA demands vs EIA's published US48 over 343
hours came to −0.75% (range −0.94% to −0.62%, zero hours beyond ±2%). The
BA sum runs slightly *below* US48, which is uncovered load rather than
double-counting; the tight spread rules out erratic reporting misalignment.
Summing these series is therefore physically meaningful. Note that US48 is
not used as a node here — with aggregates defined as sums of our own BAs the
hierarchy is coherent by construction; the check established that the
construction corresponds to something real.

**Scope: one complete sub-hierarchy at a time, never an arbitrary BA sample.**
Reconciling a scattered subset (say CAISO + DUK + CPLW) is meaningless —
they sum to no real aggregate, so coherence would be enforced against a
constraint corresponding to nothing. Every run here takes a full
``config.REGION_GROUPS`` member: Northeast (3) to validate the implementation,
Central (7) for first real signal, Southeast (16) / West (25) after.

**Methods compared** (all against the unreconciled base):
* ``base``       — the raw per-BA forecasts, the control
* ``bottom_up``  — aggregate from the bottom; coherent but cannot improve the
  bottom level at all (it *is* the bottom level). Included because
  Wickramasuriya et al. found it the *least* accurate approach, actively
  worsening most levels — a useful sanity floor.
* ``ols``        — orthogonal projection onto the coherent subspace (the
  Brégère & Huard family)
* ``mint_shrink``— MinT with a shrinkage covariance estimator

**No lookahead in the covariance.** ``W`` is estimated only from residuals of
windows strictly *earlier* than the one being scored. A W fitted on the
scored window would let the reconciler see the answers it is graded on —
the same defect #404 found in the ensemble weights (fitted in-sample until
2026-08-05, making every ensemble figure optimistic).

Base forecasts are generated locally through the production recursive path
(``recursive_autoregressive_forecast`` + exogenous-only future frame), the
same protocol ``models/training.py`` uses. Verdicts route through
``models/rolling_eval.py`` per ``docs/EVALUATION_POLICY.md``.

No writes to Redis, GCS, or ``latest.json``.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402
from _study_guards import (  # noqa: E402
    assert_arms_differ,
    assert_frames_aligned,
    assert_plausible,
)

from config import REGION_GROUPS, WEATHER_VARIABLES  # noqa: E402
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

FETCH_DAYS = 400  # 90d train + 12 windows x 168h holdout + warmup, with margin
TRAIN_ROWS = 90 * 24  # production's effective window (data/eia_client.py:325)
N_WINDOWS = 12
STRIDE_H = 336  # 14d between origins -> ~154d span, several seasons
HOLDOUT_H = 168
MIN_COV_WINDOWS = 3  # windows of prior residuals before W is estimable

#: Report cumulative-to-lead. The first pass pooled all 168 holdout hours into
#: one number, but 144 of those (86%) sit beyond day-ahead while
#: ``models/benchmark.py`` sets ``HEADLINE_LEAD = "24h"``. That mattered here
#: more than anywhere: §7's null rested on the aggregate model scoring 24.14%
#: against 3.35% for sum-of-parts, and the CAUSE was recursive drift — one
#: 168-step aggregate trajectory drifts freely while 16 independent ones partly
#: cancel. At 24h that mechanism is far weaker, so the aggregate model should be
#: competitive and MinT should finally have information to work with. If the
#: null survives at 24h it is a real property of this hierarchy; if it does not,
#: the earlier conclusion was another horizon artifact.
LEADS = (24, 48, 168)


def _dataset(region: str, end: pd.Timestamp) -> pd.DataFrame | None:
    start = (end - pd.Timedelta(days=FETCH_DAYS)).strftime("%Y-%m-%dT00")
    end_s = end.strftime("%Y-%m-%dT23")
    demand = fetch_demand(region, start=start, end=end_s, use_cache=True)
    if demand.empty:
        return None
    weather = fetch_historical_weather(region, start[:10], end_s[:10], use_cache=True)
    if weather.empty:
        return None
    merged = merge_demand_weather(demand, weather)
    if merged.empty:
        return None
    return engineer_features(merged)


def _base_forecast(df: pd.DataFrame, test: slice) -> tuple[np.ndarray, np.ndarray] | None:
    """One BA's base forecast for one window, via production's recursive path."""
    tr = df.iloc[max(0, test.start - TRAIN_ROWS) : test.start]
    if len(tr) < TRAIN_ROWS // 2:
        return None
    test_df = engineer_exogenous_features(df.iloc[test].copy())
    y = test_df["demand_mw"].to_numpy()
    try:
        res = train_xgboost(tr, cross_validate=False)
        fc = recursive_autoregressive_forecast(
            res,
            tr["demand_mw"].tolist(),
            test_df,
            predict_xgboost,
            seed_timestamps=tr.get("timestamp"),
        )[: len(y)]
    except Exception as e:  # noqa: BLE001
        log.warning("base_forecast_failed", error=str(e))
        return None
    return y, np.asarray(fc, dtype=float)


def _summing_matrix(m: int) -> np.ndarray:
    """S for a 2-level hierarchy: total on top, m bottom series. Shape (m+1, m)."""
    return np.vstack([np.ones((1, m)), np.eye(m)])


def _aggregate_dataset(data: dict[str, pd.DataFrame], bas: list[str]) -> pd.DataFrame | None:
    """Build an INDEPENDENTLY-FORECASTABLE series for the regional total.

    **This function exists because of a bug the Northeast validation run
    caught.** The first version of this study formed the aggregate forecast as
    ``F.sum(axis=0)`` — the sum of the per-BA forecasts. That is already
    coherent by construction, so every reconciliation method was a mathematical
    no-op and all four arms returned byte-identical WAPE.

    Reconciliation exploits *disagreement* between forecasts produced
    independently at different levels. Deriving the top by summing the bottom
    IS bottom-up; there is then nothing left to reconcile. So the total needs
    its own model, fit on the aggregate series, free to disagree with the sum
    of the parts.

    Demand aggregates by summation (verified against EIA's US48: −0.75%,
    no double-counting). Weather does not — it is intensive, not extensive —
    so weather columns are combined as a **load-weighted mean** across BAs,
    weighting each BA's conditions by its share of regional demand. An
    unweighted mean would let CPLW (~42 MW) pull the regional temperature as
    hard as PJM.
    """
    idx = None
    for ba in bas:
        ts = pd.DatetimeIndex(data[ba]["timestamp"])
        idx = ts if idx is None else idx.intersection(ts)
    if idx is None or len(idx) < TRAIN_ROWS + HOLDOUT_H:
        return None

    aligned = {ba: data[ba].set_index("timestamp").loc[idx] for ba in bas}
    demand = pd.DataFrame({ba: aligned[ba]["demand_mw"] for ba in bas})
    total = demand.sum(axis=1)
    share = demand.div(total.replace(0, np.nan), axis=0).fillna(1.0 / len(bas))

    # Only the 17 RAW weather variables get aggregated. An earlier version took
    # every numeric column of the FEATURED frame, which load-weighted
    # ``demand_lag_1h`` / ``demand_roll_*`` too — the lag of a sum is not the
    # weighted sum of lags. Everything derived is re-computed from the
    # aggregate series by ``engineer_features`` below.
    wx_cols = [c for c in WEATHER_VARIABLES if c in aligned[bas[0]].columns]
    agg = {"timestamp": idx, "demand_mw": total.to_numpy()}
    for c in wx_cols:
        stack = np.vstack([aligned[ba][c].to_numpy(dtype=float) for ba in bas])
        agg[c] = np.nansum(stack * share.to_numpy().T, axis=0)
    featured = engineer_features(pd.DataFrame(agg))
    if featured.empty:
        return None

    # **Per-BA weather, kept UNaveraged.** The first attempt gave the aggregate
    # model only spatially-averaged weather and it scored 11.31% WAPE against
    # 4.18% for simply summing the parts — 2.7x worse, so MinT correctly
    # ignored it and OLS was destroyed trusting it. Averaging temperature from
    # Maine to Virginia describes no real weather-demand relationship: each
    # BA's load responds to ITS OWN weather. These columns hand the aggregate
    # model the spatial signal the mean destroyed.
    # ``engineer_exogenous_features`` only ever ADDS columns, so these survive
    # into the predict-time frame and the feature sets stay aligned.
    keep = ["temperature_2m", "cooling_degree_days", "heating_degree_days"]
    extra = pd.DataFrame({"timestamp": idx})
    for ba in bas:
        for c in keep:
            if c in aligned[ba].columns:
                extra[f"{ba}__{c}"] = aligned[ba][c].to_numpy(dtype=float)
    return featured.merge(extra, on="timestamp", how="left")


def _shrink_cov(resid: np.ndarray) -> np.ndarray:
    """Shrinkage covariance (Schäfer–Strimmer style) toward a diagonal target.

    ``resid`` is (n_obs, n_series). Shrinking matters here because the number
    of residual observations available is small relative to n_series once the
    hierarchy widens, and the raw sample covariance is then ill-conditioned —
    which is precisely when MinT's inverse blows up.
    """
    n, k = resid.shape
    r = resid - resid.mean(axis=0, keepdims=True)
    S = (r.T @ r) / max(n - 1, 1)  # noqa: N806
    target = np.diag(np.diag(S))
    d = np.sqrt(np.diag(S))
    d[d == 0] = 1.0
    R = S / np.outer(d, d)  # noqa: N806
    off = R[~np.eye(k, dtype=bool)]
    denom = float((off**2).sum())
    lam = 1.0 if denom == 0 else float(np.clip(k * (k - 1) / 2 / max(n, 1) / denom, 0.0, 1.0))
    return lam * target + (1 - lam) * S


def _reconcile(
    yhat: np.ndarray,
    S: np.ndarray,  # noqa: N803 — standard hierarchical-forecasting notation
    W: np.ndarray | None,  # noqa: N803
    method: str,
) -> np.ndarray:
    """Return reconciled BOTTOM-level forecasts. ``yhat`` is (m+1,) [total, bottom...]."""
    m = S.shape[1]
    if method == "base":
        return yhat[1:]
    if method == "bottom_up":
        return yhat[1:]  # bottom is unchanged by construction; the floor case
    if method == "ols":
        Winv = np.eye(m + 1)  # noqa: N806
    else:
        if W is None:
            return yhat[1:]
        Winv = np.linalg.pinv(W)  # noqa: N806
    A = S.T @ Winv @ S  # noqa: N806
    P = np.linalg.pinv(A) @ S.T @ Winv  # noqa: N806
    return (S @ (P @ yhat))[1:]


def run(group: str, end: pd.Timestamp) -> None:
    bas = REGION_GROUPS[group]
    print(f"\n=== {group} — {len(bas)} BAs: {bas} ===")
    data = {}
    for ba in bas:
        d = _dataset(ba, end)
        if d is None:
            print(f"  {ba}: no data — ABORT (hierarchy must be complete)")
            return
        data[ba] = d

    agg = _aggregate_dataset(data, bas)
    if agg is None:
        print("  could not build the aggregate series — ABORT")
        return
    print(f"  aggregate series: {len(agg)} rows, mean {agg['demand_mw'].mean():,.0f} MW")

    # ALIGN EVERY FRAME TO A COMMON HOURLY INDEX BEFORE SLICING.
    # Without this the study is silently invalid: `test` is a POSITIONAL slice
    # applied to frames of different lengths and start times, so row i means a
    # different calendar hour in each. Measured on Northeast before the fix, at
    # slice.start=5404: ISONE 2026-03-02 04:00, NYISO 2026-03-03 00:00, AGG
    # 2026-03-10 00:00 — the aggregate forecast was being scored against BA
    # actuals from a DIFFERENT WEEK. `_aggregate_dataset` intersects BA
    # timestamps and then `engineer_features` drops a further 168-row warmup
    # from that intersection, so AGG starts 7 days after the BA frames; NYISO's
    # 3 gaps shift it 20h from ISONE/PJM. That fabricated ~10-24% "incoherence"
    # and a 13.25% aggregate error where the true figure is 3.65%.
    common = None
    for ba in bas:
        ts = pd.DatetimeIndex(data[ba]["timestamp"])
        common = ts if common is None else common.intersection(ts)
    common = common.intersection(pd.DatetimeIndex(agg["timestamp"])).sort_values()
    data = {ba: data[ba].set_index("timestamp").loc[common].reset_index() for ba in bas}
    agg = agg.set_index("timestamp").loc[common].reset_index()
    # The guard, not a bare length check: equal lengths with OFFSET start times
    # is exactly the shape that got through here before (ISONE 2026-03-02 04:00
    # vs AGG 2026-03-10 00:00 at the same positional index).
    n = assert_frames_aligned({**data, "AGG": agg})
    print(f"  aligned {len(bas) + 1} frames to {n} common hours")
    splits = rolling_origin_splits(
        n, n_windows=N_WINDOWS, holdout_h=HOLDOUT_H, stride_h=STRIDE_H, min_train_h=TRAIN_ROWS
    )
    print(
        f"  {n} usable rows | {len(splits)} windows | "
        f"months {sorted({data[bas[0]]['timestamp'].iloc[t.start].strftime('%b') for _, t in splits})}"
    )
    if len(splits) < MIN_COV_WINDOWS + 1:
        print("  too few windows to estimate W without lookahead — ABORT")
        return

    m = len(bas)
    S = _summing_matrix(m)  # noqa: N806
    METHODS = ["base", "bottom_up", "ols", "mint_shrink"]  # noqa: N806
    scored: dict[str, list] = {k: [] for k in METHODS}
    resid_hist: list[np.ndarray] = []  # per-window (m+1,) residuals, oldest first
    incoherence: list[float] = []  # |agg forecast - sum of parts|, as % of truth
    agg_wape: dict[int, list] = {L: [] for L in LEADS}  # independent agg model
    bu_wape: dict[int, list] = {L: [] for L in LEADS}  # sum-of-parts at the top
    t0 = time.time()

    for _tr, test in reversed(splits):  # oldest first: W sees only the past
        per_ba = {}
        for ba in bas:
            got = _base_forecast(data[ba], test)
            if got is None:
                break
            per_ba[ba] = got
        if len(per_ba) < m:
            continue
        got_agg = _base_forecast(agg, test)
        if got_agg is None:
            continue
        Y = np.vstack([per_ba[b][0] for b in bas])  # noqa: N806  # (m, H) actuals
        F = np.vstack([per_ba[b][1] for b in bas])  # noqa: N806  # (m, H) base forecasts
        # Truth aggregates by summation. The FORECAST of the total comes from
        # its own independent model — that disagreement is what reconciliation
        # has to work with (see _aggregate_dataset).
        Yfull = np.vstack([Y.sum(axis=0, keepdims=True), Y])  # noqa: N806
        Ffull = np.vstack([got_agg[1][None, :], F])  # noqa: N806
        incoh = float(np.mean(np.abs(Ffull[0] - F.sum(axis=0)) / np.maximum(Yfull[0], 1e-9)) * 100)
        incoherence.append(incoh)
        # THE distinguishing diagnostic: is the aggregate model any good?
        # Aggregation should make forecasting EASIER (idiosyncratic BA noise
        # partially cancels), so a competent aggregate model should beat the
        # bottom-level mean WAPE. If it is worse, reconciliation has nothing
        # informative to reconcile toward and any null result says more about
        # the aggregate model than about the method.
        for lead in LEADS:
            k = min(lead, Yfull.shape[1])
            agg_wape[lead].append(wape(Yfull[0][:k], Ffull[0][:k]))
            bu_wape[lead].append(wape(Yfull[0][:k], F.sum(axis=0)[:k]))

        W = None  # noqa: N806
        if len(resid_hist) >= MIN_COV_WINDOWS:
            W = _shrink_cov(np.vstack(resid_hist))  # noqa: N806
        for meth in METHODS:
            rec = np.column_stack(
                [_reconcile(Ffull[:, h], S, W, meth) for h in range(Ffull.shape[1])]
            )  # (m, H)
            scored[meth].append(
                {
                    f"{stat}_{lead}": [
                        fn(Y[i][: min(lead, Y.shape[1])], rec[i][: min(lead, Y.shape[1])])
                        for i in range(m)
                    ]
                    for lead in LEADS
                    for stat, fn in (("wape", wape), ("mape", compute_mape), ("bias", bias_pct))
                }
            )
        resid_hist.append((Ffull - Yfull).T)  # (H, m+1), strictly past by loop order

    nw = len(scored["base"])
    print(
        f"  {time.time() - t0:.0f}s | {nw} scored windows "
        f"(W estimable from window {MIN_COV_WINDOWS + 1} on)"
    )
    if nw == 0:
        print("  nothing scored")
        return
    incoh_mean = float(np.mean(incoherence))
    print(f"  base incoherence |agg - sum(parts)|: {incoh_mean:.2f}% of truth")
    # Two forecasts of the same quantity from the same data do not disagree by
    # tens of percent. A reading like that was reported as a property of the
    # hierarchy three times before it was recognised as a misalignment bug.
    assert_plausible("base incoherence %", incoh_mean, 0.0, 10.0)

    sizes = {ba: float(data[ba]["demand_mw"].mean()) for ba in bas}
    order = sorted(bas, key=lambda b: sizes[b])
    half = max(1, len(order) // 2)
    small_idx = [bas.index(b) for b in order[:half]]

    for lead in LEADS:
        tag = "  <-- benchmark HEADLINE_LEAD" if lead == 24 else ""
        a, b = np.mean(agg_wape[lead]), np.mean(bu_wape[lead])
        print(f"\n  === cumulative to {lead}h{tag} ===")
        print(
            f"  TOP-level: independent agg model {a:.2f}%  vs  sum-of-parts {b:.2f}%"
            f"  ({'agg competitive' if a <= b * 1.25 else 'agg carries little info'})"
        )
        base_w = np.array([w for r in scored["base"] for w in r[f"wape_{lead}"]])
        assert_arms_differ(
            f"wape_{lead}",
            {m_: np.array([x for r in scored[m_] for x in r[f"wape_{lead}"]]) for m_ in METHODS},
        )
        print(f"  {'method':13s} {'BA-mean':>8s} {'bias':>7s} {'small-BA':>9s} | vs base")
        for meth in METHODS:
            arr = scored[meth]
            w = np.array([x for r in arr for x in r[f"wape_{lead}"]])
            bi = np.array([x for r in arr for x in r[f"bias_{lead}"]])
            sm = np.array([r[f"wape_{lead}"][i] for r in arr for i in small_idx])
            line = (
                f"  {meth:13s} {np.nanmean(w):>7.2f}% {np.nanmean(bi):>+6.2f}% "
                f"{np.nanmean(sm):>8.2f}%"
            )
            if meth == "base":
                print(line + " | (control)")
                continue
            v = verdict(base_w - w)
            sat = satisficing_check(
                treatment_bias_pct=float(np.nanmean(bi)),
                control_mape=float(
                    np.nanmean([x for r in scored["base"] for x in r[f"mape_{lead}"]])
                ),
                treatment_mape=float(np.nanmean([x for r in arr for x in r[f"mape_{lead}"]])),
            )
            tagf = "" if sat["passed"] else "  SAT-FAIL"
            print(line + f" | {np.nanmean(base_w - w):+.3f} — {v['reason']}{tagf}")


def main():
    end = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=5)
    groups = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Northeast"]
    print("Phase 1 — hierarchical reconciliation, per-BA (bottom-level) accuracy")
    print(f"Base: XGBoost via production recursive path | control window {TRAIN_ROWS // 24}d")
    for g in groups:
        if g not in REGION_GROUPS:
            print(f"unknown group {g}; have {list(REGION_GROUPS)}")
            continue
        run(g, end)


if __name__ == "__main__":
    main()
