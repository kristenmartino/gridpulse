"""Does skipping degenerate Prophet regressors cost accuracy, and does it stop the cold-front blow-up?

Companion to the ``prophet_skip_degenerate_regressors`` flag in ``config.py``.
On 2026-10-17 ERCOT's summer-trained Prophet forecast 244-251 GW against ~42 GW
actual: its training window held 1 of 1,996 hours with any HDD, so Prophet
standardized HDD by a std of 0.0095 and the first cold night was extrapolated.

Two questions, answered separately:

1. **Replay.** Train both arms on the 1,996 hours before 2026-10-16 04:00 UTC
   (the history end of the production 2026-10-16 model) and forecast through
   2026-10-18 12:00. Exog is ERA5 archive weather, not the forecast weather the
   serve path used, so this replays the model, not the full serve path.
2. **Regression check.** Rolling origin, 8 x 168h windows, WAPE deciding with
   the MAPE/bias satisficing constraints (``models/rolling_eval.py``). Each
   window trains on a FIXED trailing 1,996 rows, matching production, so the
   summer-only windows that produce a degenerate HDD exist in the study at all.

Arms differ only by the flag. Every window logs whether the treatment actually
skipped a regressor; a region where no window skips anything is reported as
"no change" rather than scored, since its arms are identical by construction.

No writes to Redis, GCS, or ``latest.json`` — reads real EIA/Open-Meteo data
and fits local, throwaway Prophet models.
"""

import sys
import time
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import structlog

warnings.filterwarnings("ignore")

from data.eia_client import fetch_demand  # noqa: E402
from data.feature_engineering import engineer_features  # noqa: E402
from data.preprocessing import merge_demand_weather  # noqa: E402
from data.weather_client import fetch_historical_weather  # noqa: E402
from models.evaluation import compute_mape  # noqa: E402
from models.prophet_model import predict_prophet, train_prophet  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402
from _study_guards import assert_arms_differ, assert_fixed_window  # noqa: E402

from models.rolling_eval import (  # noqa: E402
    bias_pct,
    rolling_origin_splits,
    satisficing_check,
    verdict,
    wape,
)

log = structlog.get_logger()

FLAG = "prophet_skip_degenerate_regressors"
TRAIN_ROWS = 1996  # production train_rows (GCS .meta.json, 2026-10)
N_WINDOWS = 8  # docs/EVALUATION_POLICY.md default
HOLDOUT_H = 168
LEADS = (24, 48, 168)

#: Regions whose 2026-10-21 production Prophet had HDD std < 0.5 (FPL, IID, TEC,
#: FMPP) plus ERCOT, whose 2026-10-16 model failed, plus two controls whose HDD
#: varies (PJM, BPAT) and should show no change.
DEFAULT_REGIONS = ["ERCOT", "FPL", "IID", "TEC", "FMPP", "PJM", "BPAT"]

REPLAY_REGION = "ERCOT"
REPLAY_ORIGIN = pd.Timestamp("2026-10-16T04:00", tz="UTC")
REPLAY_END = pd.Timestamp("2026-10-18T12:00", tz="UTC")


def _build_dataset(region: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    """Fetch + merge + feature-engineer real data for [start, end]."""
    demand = fetch_demand(
        region, start=start.strftime("%Y-%m-%dT00"), end=end.strftime("%Y-%m-%dT23"), use_cache=True
    )
    if demand.empty:
        return None
    weather = fetch_historical_weather(
        region, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), use_cache=True
    )
    if weather.empty:
        return None
    merged = merge_demand_weather(demand, weather)
    return None if merged.empty else engineer_features(merged).reset_index(drop=True)


def _fit_predict(
    train_df: pd.DataFrame, test_df: pd.DataFrame, flag: bool
) -> tuple[np.ndarray, bool]:
    """Train one arm and forecast ``test_df``. Returns (forecast, hdd_attached)."""
    with patch.dict("config.FEATURE_FLAGS", {FLAG: flag}):
        model = train_prophet(train_df)
    pred = predict_prophet(model, test_df, periods=len(test_df))
    return pred["forecast"], "heating_degree_days" in model.extra_regressors


def replay() -> None:
    """Question 1: the 2026-10-16 ERCOT model on the 10-17 cold front."""
    print(f"\n== Replay: {REPLAY_REGION}, origin {REPLAY_ORIGIN:%Y-%m-%d %H:%M} UTC ==")
    df = _build_dataset(
        REPLAY_REGION, REPLAY_ORIGIN - pd.Timedelta(days=95), REPLAY_END + pd.Timedelta(days=1)
    )
    if df is None:
        print("  no data")
        return
    train_df = df[df.timestamp < REPLAY_ORIGIN].tail(TRAIN_ROWS)
    test_df = df[(df.timestamp >= REPLAY_ORIGIN) & (df.timestamp <= REPLAY_END)]
    assert_fixed_window("replay", [slice(0, len(train_df))], TRAIN_ROWS)
    y = test_df["demand_mw"].to_numpy()
    print(
        f"  train {train_df.timestamp.min():%m-%d %H:%M} -> {train_df.timestamp.max():%m-%d %H:%M}, "
        f"HDD hours>0: {int((train_df.heating_degree_days > 0).sum())}, "
        f"test hours {len(test_df)}, actual max {y.max():,.0f} MW"
    )
    for flag in (False, True):
        fc, hdd_on = _fit_predict(train_df, test_df, flag)
        print(
            f"  flag={flag!s:5}  HDD attached={hdd_on!s:5}  MAPE {compute_mape(y, fc):6.2f}%  "
            f"max forecast {fc.max():,.0f} MW  max fc/actual {np.max(fc / y):.2f}"
        )


def _rolling(region: str, df: pd.DataFrame) -> dict:
    splits = rolling_origin_splits(
        len(df), n_windows=N_WINDOWS, holdout_h=HOLDOUT_H, min_train_h=TRAIN_ROWS
    )
    # Fixed trailing window, not the expanding slice rolling_origin_splits returns.
    splits = [(slice(te.start - TRAIN_ROWS, te.start), te) for _, te in splits]
    assert_fixed_window(region, [tr for tr, _ in splits], TRAIN_ROWS)
    out: dict = {"skipped": []}
    for arm in ("control", "treatment"):
        for k in ("wape", "mape", "bias"):
            for lead in LEADS:
                out[f"{arm}_{k}_{lead}"] = []
    for tr, te in splits:
        train_df, test_df = df.iloc[tr], df.iloc[te]
        y = test_df["demand_mw"].to_numpy()
        fc_c, _ = _fit_predict(train_df, test_df, False)
        fc_t, hdd_on = _fit_predict(train_df, test_df, True)
        out["skipped"].append(not hdd_on)
        for arm, fc in (("control", fc_c), ("treatment", fc_t)):
            for lead in LEADS:
                out[f"{arm}_wape_{lead}"].append(wape(y[:lead], fc[:lead]))
                out[f"{arm}_mape_{lead}"].append(compute_mape(y[:lead], fc[:lead]))
                out[f"{arm}_bias_{lead}"].append(bias_pct(y[:lead], fc[:lead]))
    return out


def regression_check(regions: list[str], end: pd.Timestamp) -> None:
    """Question 2: does skipping cost accuracy in ordinary weeks?"""
    for region in regions:
        print(f"\n== {region} ==")
        t0 = time.time()
        df = _build_dataset(region, end - pd.Timedelta(days=90 + 7 * N_WINDOWS + 7), end)
        if df is None or len(df) < TRAIN_ROWS + HOLDOUT_H:
            print("  insufficient data")
            continue
        r = _rolling(region, df)
        n = len(r["skipped"])
        n_skip = sum(r["skipped"])
        print(f"  windows {n}, treatment skipped HDD in {n_skip}/{n}  ({time.time() - t0:.0f}s)")
        if n_skip == 0:
            print("  no change: arms identical by construction")
            continue
        assert_arms_differ(
            "wape_168",
            {
                "control": np.array(r["control_wape_168"]),
                "treatment": np.array(r["treatment_wape_168"]),
            },
        )
        for lead in LEADS:
            deltas = np.array(r[f"control_wape_{lead}"]) - np.array(r[f"treatment_wape_{lead}"])
            v = verdict(deltas)
            sat = satisficing_check(
                treatment_bias_pct=float(np.mean(r[f"treatment_bias_{lead}"])),
                control_mape=float(np.mean(r[f"control_mape_{lead}"])),
                treatment_mape=float(np.mean(r[f"treatment_mape_{lead}"])),
            )
            print(
                f"  [{lead:>3d}h] WAPE ctl {np.mean(r[f'control_wape_{lead}']):.2f}% -> "
                f"trt {np.mean(r[f'treatment_wape_{lead}']):.2f}%  "
                f"MAPE ctl {np.mean(r[f'control_mape_{lead}']):.2f}% -> "
                f"trt {np.mean(r[f'treatment_mape_{lead}']):.2f}%  "
                f"bias ctl {np.mean(r[f'control_bias_{lead}']):+.2f}% -> "
                f"trt {np.mean(r[f'treatment_bias_{lead}']):+.2f}%  "
                f"{'PASS' if sat['passed'] else 'SAT-FAIL'}  |  {v['reason']}"
            )


def main() -> None:
    replay()
    # 5-day lag: recent EIA hours are still settling.
    end = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=5)
    regions = sys.argv[1].split(",") if len(sys.argv) > 1 else DEFAULT_REGIONS
    regression_check(regions, end)


if __name__ == "__main__":
    main()
