"""#231 follow-up v3: how much history, and weighted how? (XGBoost)

Supersedes the fixed-window studies. Three things changed, each because the
previous design was shown to be blind to something.

**1. Window length and recency weighting are ONE knob, so sweep the knob.**
A hard 90-day training window is already a weighting scheme — weight 1 inside,
weight 0 outside, a step function. Exponential decay is its smooth
generalisation: ``w = 0.5 ** (age_days / half_life)``. Half-life ~30d behaves
like today's short window; half-life -> infinity is a flat multi-year window.
Sweeping half-life therefore subsumes the window-length question instead of
running it as a separate experiment, and needs only ONE fetch.

It also resolves a tension truncation cannot. A hard cutoff forces last
winter to be either fully in the fit or fully out. Decay lets it contribute
its seasonal SHAPE while discounting its stale LEVEL — which is the split the
evidence points at, since the failure mode measured across all three model
families was a systematic under-forecast (a level error), not a shape error.

**2. Test windows must span seasons.** The previous study's 8 windows were
consecutive: 56 total days, all of them Jun-Aug. It asked "does more history
help in July," never "does more history help across a seasonal turn" — the
only regime where the seasonality argument claims a benefit, and the one
#283's Phase 0 found matters (weather-normal beat recent-28d ~10:2 at
seasonal turns and was "a wash mid-season"). A null result in mid-summer is
what that prior predicts, so the earlier null was uninformative about
seasonality. ``rolling_origin_splits`` takes ``stride_h``; at 720h the same
harness spreads 12 windows across ~330 days.

**3. The seasonal segmentation is PRE-DECLARED, because averaging hides the
effect.** If the benefit really is "large at turns, a wash mid-season," then
an aggregate verdict over seasonally-spread windows averages a real effect
into nothing. ``TURN_MONTHS``/``PEAK_MONTHS`` below are fixed before any
result is seen, and every arm is reported on all three segments (all / turn /
peak). Reading only the segment that happens to look good afterwards would be
the same error in a new costume, so all three are always printed.

**Scoring matches production**: ``recursive_autoregressive_forecast`` with an
exogenous-only future frame, as ``models/training.py`` does. Scoring one-shot
against holdout rows carrying real ``demand_lag_1h`` measures a regime
production never runs in (ADR-010), and 419 of 420 fits in the earlier sweep
put an autoregressive lag on top of the feature-importance list.

**Weighting is XGBoost-only.** Verified: ``XGBRegressor.fit`` accepts
``sample_weight``; ``Prophet.fit(self, df, **kwargs)`` has no weight
parameter; statsmodels' SARIMAX MLE has none either. For those two the
equivalent knob is hard window length, which this same harness measures via
the ``HARD_`` arms.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402
from _study_guards import assert_fixed_window  # noqa: E402

from models.xgboost_model import (  # noqa: E402
    DEFAULT_PARAMS,
    _get_feature_cols,
    predict_xgboost,
    train_xgboost,
)

log = structlog.get_logger()

SAMPLE_BAS = ["CAISO", "DUK"]  # pilot; widen once the harness is validated

FETCH_DAYS = 1500  # ~4.1y — measured ~100% EIA coverage on all 6 sampled BAs
N_WINDOWS = 12
STRIDE_H = 720  # 30d between window origins -> ~330d of seasonal spread
HOLDOUT_H = 168

#: Arms. ``None`` half-life = flat (unweighted). ``hard_days`` truncates to a
#: rectangular window instead, which is what Prophet/SARIMAX are limited to.
ARMS = [
    ("control_hard_90d", {"hard_days": 90}),
    ("hard_365d", {"hard_days": 365}),
    ("hard_730d", {"hard_days": 730}),
    ("decay_hl_30d", {"half_life": 30}),
    ("decay_hl_90d", {"half_life": 90}),
    ("decay_hl_180d", {"half_life": 180}),
    ("decay_hl_365d", {"half_life": 365}),
    ("decay_hl_730d", {"half_life": 730}),
    ("flat_all", {}),
]
CONTROL_ARM = "control_hard_90d"

# Pre-declared BEFORE any result is seen. Shoulder/ramp months are where a
# seasonal-coverage benefit should appear if it exists at all.
TURN_MONTHS = {3, 4, 5, 9, 10, 11}
PEAK_MONTHS = {6, 7, 8, 12, 1, 2}

#: Report accuracy CUMULATIVE TO each lead, not pooled over the whole holdout.
#: An earlier version reported one number pooled across all 168 hours -- but 144
#: of those 168 (86%) are beyond day-ahead, so the headline was dominated by
#: days 2-7 while ``models/benchmark.py`` HEADLINES ON 24h. Error is not flat
#: across the horizon either (CAISO step-wise: -0.75% at h1, -6.68% at h24,
#: +1.79% at h168), so a pooled mean can hide an effect that exists at one lead
#: and not another. 24h is the benchmark's headline lead, 48h its conservative
#: one, 168h the full holdout for continuity with the earlier runs.
LEADS = (24, 48, 168)

#: Rows whose weight falls below this contribute nothing measurable; dropping
#: them makes short half-lives cheap without changing the fit.
WEIGHT_EPS = 1e-3


def _build_dataset(region: str, end: pd.Timestamp) -> pd.DataFrame | None:
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


def _fit(df: pd.DataFrame, weights: np.ndarray | None) -> dict:
    """Mirror of ``train_xgboost``'s FINAL fit, plus ``sample_weight``.

    ``train_xgboost`` has no weight parameter, so the final-fit path is
    reproduced here rather than modifying production code for a study. It is
    a copy and can therefore drift; ``_selfcheck`` asserts the unweighted case
    is prediction-identical to ``train_xgboost`` before any arm is scored, so
    drift fails loudly instead of silently changing the control.
    """
    from xgboost import XGBRegressor

    feature_cols = _get_feature_cols(df)
    params = DEFAULT_PARAMS.copy()
    params.pop("early_stopping_rounds", None)  # production's final fit drops it too
    model = XGBRegressor(**params)
    model.fit(df[feature_cols].values, df["demand_mw"].values, sample_weight=weights, verbose=False)
    return {"model": model, "feature_names": feature_cols, "cv_scores": []}


def _selfcheck(df: pd.DataFrame) -> None:
    """Unweighted ``_fit`` must equal ``train_xgboost``, or the control is not the control."""
    sl = df.iloc[-2160:]
    a = predict_xgboost(_fit(sl, None), sl)
    b = predict_xgboost(train_xgboost(sl, cross_validate=False), sl)
    if not np.allclose(a, b, rtol=1e-9, atol=1e-6):
        raise SystemExit(f"_fit diverged from train_xgboost (max |d|={np.max(np.abs(a - b)):.6g})")
    print("  selfcheck: _fit reproduces train_xgboost exactly (unweighted)")


def _arm_frame(
    df: pd.DataFrame, end_idx: int, spec: dict
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Training rows + sample weights for one arm, ending at ``end_idx``."""
    if "hard_days" in spec:
        rows = spec["hard_days"] * 24
        return df.iloc[max(0, end_idx - rows) : end_idx], None
    train = df.iloc[:end_idx]
    if "half_life" not in spec:
        return train, None
    age_d = (train["timestamp"].iloc[-1] - train["timestamp"]).dt.total_seconds().to_numpy() / 86400
    w = 0.5 ** (age_d / spec["half_life"])
    keep = w >= WEIGHT_EPS
    return train.iloc[keep], w[keep]


def run(region: str, end: pd.Timestamp) -> dict | None:
    print(f"\n== {region} ==")
    df = _build_dataset(region, end)
    if df is None:
        print("  no data")
        return None
    _selfcheck(df)

    splits = rolling_origin_splits(
        len(df),
        n_windows=N_WINDOWS,
        holdout_h=HOLDOUT_H,
        stride_h=STRIDE_H,
        min_train_h=90 * 24,
    )
    # Every hard_* arm must be a FIXED trailing window, not an expanding one.
    for aname, spec in ARMS:
        if "hard_days" in spec:
            rows = spec["hard_days"] * 24
            sl = [_arm_frame(df, t.start, spec) for _, t in splits]
            assert_fixed_window(aname, [slice(0, len(f)) for f, _w in sl], rows)

    ts = df["timestamp"]
    months = [ts.iloc[t.start].month for _, t in splits]
    print(
        f"  {len(df)} rows ({len(df) / 24:.0f}d) | {len(splits)} windows | "
        f"months {sorted({ts.iloc[t.start].strftime('%b') for _, t in splits})}"
    )

    scores = {
        name: {f"{k}_{L}": [] for k in ("wape", "mape", "bias") for L in LEADS} for name, _ in ARMS
    }
    t0 = time.time()
    for _train_slice, test in splits:
        test_df = engineer_exogenous_features(df.iloc[test].copy())
        y = test_df["demand_mw"].to_numpy()
        for name, spec in ARMS:
            tr, w = _arm_frame(df, test.start, spec)
            try:
                res = _fit(tr, w)
                fc = recursive_autoregressive_forecast(
                    res,
                    tr["demand_mw"].tolist(),
                    test_df,
                    predict_xgboost,
                    seed_timestamps=tr.get("timestamp"),
                )[: len(y)]
            except Exception as e:  # noqa: BLE001
                log.warning("arm_failed", arm=name, error=str(e))
                for k in scores[name]:
                    scores[name][k].append(np.nan)
                continue
            for lead in LEADS:
                n = min(lead, len(y))
                scores[name][f"wape_{lead}"].append(wape(y[:n], fc[:n]))
                scores[name][f"mape_{lead}"].append(compute_mape(y[:n], fc[:n]))
                scores[name][f"bias_{lead}"].append(bias_pct(y[:n], fc[:n]))
    print(f"  {time.time() - t0:.0f}s for {len(splits)} windows x {len(ARMS)} arms")

    seg = {
        "all": list(range(len(months))),
        "turn": [i for i, m in enumerate(months) if m in TURN_MONTHS],
        "peak": [i for i, m in enumerate(months) if m in PEAK_MONTHS],
    }
    ctl = scores[CONTROL_ARM]
    for lead in LEADS:
        tag = " <-- benchmark HEADLINE_LEAD" if lead == 24 else ""
        print(f"\n  === cumulative to {lead}h{tag} ===")
        print(f"  {'arm':18s} {'WAPE':>7s} {'bias':>7s} | {'all':>20s} {'turn':>10s} {'peak':>10s}")
        for name, _ in ARMS:
            s_ = scores[name]
            row = (
                f"  {name:18s} {np.nanmean(s_[f'wape_{lead}']):>6.2f}% "
                f"{np.nanmean(s_[f'bias_{lead}']):>+6.2f}% |"
            )
            if name == CONTROL_ARM:
                print(row + f" {'(control)':>20s}")
                continue
            cell = ""
            for lbl in ("all", "turn", "peak"):
                idx = seg[lbl]
                d = np.array([ctl[f"wape_{lead}"][i] - s_[f"wape_{lead}"][i] for i in idx])
                v = verdict(d)
                mark = (
                    "WIN"
                    if (v["decisive"] and v["winner"] == "treatment")
                    else "LOSS"
                    if v["decisive"]
                    else "~"
                )
                cell += f" {np.nanmean(d):>+6.2f}{mark:<4s}"
            sat = satisficing_check(
                treatment_bias_pct=float(np.nanmean(s_[f"bias_{lead}"])),
                control_mape=float(np.nanmean(ctl[f"mape_{lead}"])),
                treatment_mape=float(np.nanmean(s_[f"mape_{lead}"])),
            )
            print(row + cell + ("" if sat["passed"] else "  SAT-FAIL"))
    return None


def main():
    end = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=5)
    regions = sys.argv[1].split(",") if len(sys.argv) > 1 else SAMPLE_BAS
    print("v3 — half-life sweep (window length and recency weighting are one knob)")
    print(f"Seasonal spread: {N_WINDOWS} windows @ {STRIDE_H}h stride")
    print(f"Segments pre-declared: turn={sorted(TURN_MONTHS)} peak={sorted(PEAK_MONTHS)}")
    print(f"Regions: {regions}")
    for r in regions:
        try:
            run(r, end)
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"{r}: ERROR {e}")


if __name__ == "__main__":
    main()
