"""#559 — does the positional AR seed change forecast VALUES, and does it cost accuracy?

The demand frame is a complete hourly grid on 50 of 51 BAs (PR #578), so
``shift(24)`` in ``add_autoregressive_demand_features`` is temporally exact and
the training features are correct. The defect is one layer down: ``dropna``
deletes rows whose lag source was null, punching real holes into ``featured``,
and ``jobs/phases.py`` seeds the recursion with ``featured["demand_mw"].tolist()``
— a list that ``compute_autoregressive_snapshot`` then indexes POSITIONALLY.
``history[-168]`` is 168 surviving rows back, not 168 hours back.

Two arms, one difference. Same archived vintage, same weather, same origins,
same future frame; only the history indexing moves:

  control    ``recursive_autoregressive_forecast``  (production, positional)
  treatment  the same loop with the history keyed by TIMESTAMP

This is a replay, not a retrain: GCS holds daily vintages, so each origin loads
the model that was live at that moment (#451's lesson).

Reports two things, deliberately:

  1. **Divergence** — |treatment − control| as a share of demand. Needs no truth,
     so it answers "does this change values at all" even where the accuracy test
     is underpowered.
  2. **Accuracy** — paired deltas through ``models/rolling_eval.py``. Pooled
     across BAs is the well-powered unit; per-BA verdicts ship with their MDE.

Ungapped BAs are a null control: the two arms must be BYTE-IDENTICAL there. A
difference is a bug in this harness, not a finding.

Usage:
    PYTHONPATH=. python scripts/positional_lag_value_study.py [--horizon 168]
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

BUCKET = "nextera-portfolio-energy-cache"
MODEL_PREFIX = "cache/models"
MIRROR = os.environ.get("GP_MIRROR_DIR", "")

# BAs carrying a corrupt origin, measured 2026-08-18; MISO/PJM are null controls.
AFFECTED = ["TIDC", "PSCO", "IID", "NWMT", "SPA", "NEVP", "LGEE"]
CONTROLS = ["MISO", "PJM"]

AR_WINDOWS = (24, 72, 168)


# ── history, indexed two ways ────────────────────────────────────────


def _temporal_snapshot(hist: dict[pd.Timestamp, float], now: pd.Timestamp) -> dict[str, float]:
    """``compute_autoregressive_snapshot``, resolved by hour instead of position.

    Same key set and same arithmetic; the only change is that ``lag_k`` asks for
    the hour ``now - k`` and returns NaN when that hour is absent, rather than
    counting back k surviving entries. Rolling windows cover the hours actually
    present in ``[now - window, now - 1h]`` — the temporal reading of the
    training side's ``min_periods=1``.
    """

    def lag(k: int) -> float:
        return hist.get(now - pd.Timedelta(hours=k), np.nan)

    def roll(window: int, op: str) -> float:
        vals = [
            hist[now - pd.Timedelta(hours=i)]
            for i in range(1, window + 1)
            if (now - pd.Timedelta(hours=i)) in hist
        ]
        if not vals:
            return np.nan
        arr = np.asarray(vals, dtype=float)
        if op == "mean":
            return float(np.mean(arr))
        if op == "std":
            return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        if op == "min":
            return float(np.min(arr))
        return float(np.max(arr))

    lag_1, lag_2, lag_3 = lag(1), lag(2), lag(3)
    lag_24, lag_168 = lag(24), lag(168)
    roll_24_mean, roll_168_mean = roll(24, "mean"), roll(168, "mean")

    def diff(a: float, b: float) -> float:
        return a - b if not np.isnan(a) and not np.isnan(b) else np.nan

    def ratio(a: float, b: float) -> float:
        return a / max(b, 1.0) if not np.isnan(a) and not np.isnan(b) else np.nan

    out = {
        "demand_lag_1h": lag_1,
        "demand_lag_3h": lag_3,
        "demand_lag_24h": lag_24,
        "demand_lag_168h": lag_168,
        "ramp_rate": diff(lag_1, lag_2),
        "demand_momentum_short": diff(lag_1, lag_3),
        "demand_momentum_long": diff(lag_1, lag_24),
        "demand_ratio_24h": ratio(lag_24, roll_24_mean),
        "demand_ratio_168h": ratio(lag_168, roll_168_mean),
    }
    for w in AR_WINDOWS:
        for op in ("mean", "std", "min", "max"):
            out[f"demand_roll_{w}h_{op}"] = roll(w, op)
    return out


def temporal_recursive_forecast(model, seed: pd.Series, future_df: pd.DataFrame, predict_fn):
    """``recursive_autoregressive_forecast`` with a timestamp-keyed history.

    ``seed`` is indexed by timestamp. Mirrors the production loop step for step,
    including the #129 zero/NaN filter, so any delta between the arms is the
    indexing and nothing else.
    """
    hist: dict[pd.Timestamp, float] = {
        ts: float(v) for ts, v in seed.items() if v is not None and not pd.isna(v) and v > 0
    }
    stamps = pd.to_datetime(future_df["timestamp"]).tolist()
    cols = list(future_df.columns)
    ar_keys = [k for k in _temporal_snapshot(hist, stamps[0]) if k in cols]
    to_cast = {k: "float64" for k in ar_keys if future_df[k].dtype != np.float64}
    if to_cast:
        future_df = future_df.astype(to_cast)
    positions = [cols.index(k) for k in ar_keys]

    preds: list[float] = []
    for i, now in enumerate(stamps):
        row = future_df.iloc[[i]].copy()
        snap = _temporal_snapshot(hist, now)
        if positions:
            row.iloc[0, positions] = [snap[k] for k in ar_keys]
        row = row.fillna(0)
        pred = float(predict_fn(model, row)[0])
        preds.append(pred)
        hist[now] = pred
    return np.asarray(preds, dtype=float)


# ── data + vintages ──────────────────────────────────────────────────


def _client():
    from google.cloud import storage

    return storage.Client()


def load_mirror(kind: str, region: str) -> pd.DataFrame:
    if MIRROR:
        return pd.read_parquet(f"{MIRROR}/{kind}/{region}/latest.parquet")
    blob = _client().bucket(BUCKET).blob(f"cache/{kind}/{region}/latest.parquet")
    return pd.read_parquet(io.BytesIO(blob.download_as_bytes()))


def vintage_index(client, region: str, model_name: str = "xgboost") -> list[tuple]:
    """(trained_at, version) for every persisted vintage, oldest first."""
    out = []
    prefix = f"{MODEL_PREFIX}/{region}/{model_name}/"
    for b in client.list_blobs(BUCKET, prefix=prefix):
        if not b.name.endswith(".meta.json"):
            continue
        version = b.name.rsplit("/", 1)[-1].removesuffix(".meta.json")
        if version == "latest":
            continue
        try:
            meta = json.loads(b.download_as_bytes())
            out.append((pd.Timestamp(meta["trained_at"]), version))
        except Exception:
            continue
    return sorted(out)


def load_vintage(client, region: str, version: str, model_name: str = "xgboost"):
    blob = client.bucket(BUCKET).blob(f"{MODEL_PREFIX}/{region}/{model_name}/{version}.pkl")
    return pickle.loads(blob.download_as_bytes())


# ── the run ──────────────────────────────────────────────────────────


def corrupt_origins(seed_ts: pd.Series) -> list[int]:
    """Seed positions where a positional lag lands on the wrong hour."""
    n = len(seed_ts)
    bad: set[int] = set()
    for k in (24, 168):
        j = np.arange(k, n)
        origin = seed_ts.values[j - 1] + np.timedelta64(1, "h")
        bad.update(j[seed_ts.values[j - k] != origin - np.timedelta64(k, "h")].tolist())
    return sorted(bad)


def run_region(client, region: str, horizon: int, vintages: list[tuple]) -> list[dict]:
    from data.feature_engineering import engineer_features, recursive_autoregressive_forecast
    from data.preprocessing import merge_demand_weather
    from jobs.phases import _build_future_feature_frame
    from models.xgboost_model import predict_xgboost

    demand = load_mirror("demand", region)
    weather = load_mirror("weather", region)
    featured = engineer_features(merge_demand_weather(demand, weather))
    ts = pd.to_datetime(featured["timestamp"]).reset_index(drop=True)
    vals = featured["demand_mw"].to_numpy()
    keep = np.where(~pd.isna(vals) & (vals > 0))[0]
    seed_ts = ts.iloc[keep].reset_index(drop=True)

    truth = demand.dropna(subset=["demand_mw"]).set_index(
        pd.to_datetime(demand.dropna(subset=["demand_mw"])["timestamp"])
    )["demand_mw"]
    last = ts.iloc[-1]

    cand = corrupt_origins(seed_ts)
    if not cand:
        # Ungapped BA: no corrupt origin exists by construction. Sample evenly
        # so the null control still runs — the arms must agree exactly here.
        cand = list(range(200, len(seed_ts), horizon))
    picked, last_t = [], None
    for j in cand:
        origin = seed_ts.iloc[j - 1] + pd.Timedelta(hours=1)
        if origin + pd.Timedelta(hours=horizon - 1) > last:
            continue
        if last_t is not None and (origin - last_t) < pd.Timedelta(hours=horizon):
            continue
        picked.append(origin)
        last_t = origin

    rows = []
    for origin in picked:
        elig = [v for t, v in vintages if t < origin]
        if not elig:
            continue
        try:
            model = load_vintage(client, region, elig[-1])
        except Exception:
            continue
        past = featured[ts < origin]
        if len(past) < 200:
            continue
        future = _build_future_feature_frame(
            past, horizon, weather_df=weather, start_ts=origin
        ).iloc[:horizon]
        if len(future) < horizon:
            continue

        ctl = recursive_autoregressive_forecast(
            model, past["demand_mw"].tolist(), future.copy(), predict_xgboost
        )
        past_seed = past.set_index(pd.to_datetime(past["timestamp"]))["demand_mw"]
        trt = temporal_recursive_forecast(model, past_seed, future.copy(), predict_xgboost)

        idx = pd.date_range(origin, periods=horizon, freq="h")
        act = truth.reindex(idx).to_numpy(dtype=float)
        ok = np.isfinite(act) & np.isfinite(ctl) & np.isfinite(trt)
        if ok.sum() < horizon * 0.8:
            continue
        rows.append(
            {
                "region": region,
                "origin": origin,
                "version": elig[-1],
                "n": int(ok.sum()),
                "divergence_pct": float(
                    100 * np.abs(trt[ok] - ctl[ok]).sum() / np.abs(act[ok]).sum()
                ),
                "wape_control": float(
                    100 * np.abs(act[ok] - ctl[ok]).sum() / np.abs(act[ok]).sum()
                ),
                "wape_treatment": float(
                    100 * np.abs(act[ok] - trt[ok]).sum() / np.abs(act[ok]).sum()
                ),
                "mape_control": float(100 * np.mean(np.abs((act[ok] - ctl[ok]) / act[ok]))),
                "mape_treatment": float(100 * np.mean(np.abs((act[ok] - trt[ok]) / act[ok]))),
                "bias_treatment": float(100 * (trt[ok] - act[ok]).sum() / np.abs(act[ok]).sum()),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=168)
    ap.add_argument("--regions", default=",".join(AFFECTED + CONTROLS))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from models.rolling_eval import paired_deltas, satisficing_check, verdict

    client = _client()
    all_rows: list[dict] = []
    for region in args.regions.split(","):
        vints = vintage_index(client, region)
        rows = run_region(client, region, args.horizon, vints)
        all_rows.extend(rows)
        if rows:
            d = pd.DataFrame(rows)
            print(
                f"{region:5} n={len(rows):3}  divergence={d.divergence_pct.mean():6.3f}%  "
                f"WAPE ctl={d.wape_control.mean():6.3f} trt={d.wape_treatment.mean():6.3f}"
            )
        else:
            print(f"{region:5} n=0   (no scoreable corrupt origin at horizon {args.horizon})")

    if not all_rows:
        print("\nno scored origins")
        return

    df = pd.DataFrame(all_rows)
    if args.out:
        df.to_json(args.out, orient="records", date_format="iso", indent=2)

    print("\n── null control (must be exactly 0 divergence) ──")
    for r in CONTROLS:
        sub = df[df.region == r]
        if len(sub):
            print(f"  {r:5} max divergence {sub.divergence_pct.max():.9f}%")

    aff = df[df.region.isin(AFFECTED)]
    print(f"\n── pooled over affected BAs (n={len(aff)}) ──")
    if not len(aff):
        print("  no affected-BA origins scored")
        return
    dl = paired_deltas(aff.wape_control.to_numpy(), aff.wape_treatment.to_numpy())
    v = verdict(dl)
    if v["mean"] is None:
        print(f"  {v['reason']}")
        return
    print(
        f"  mean Δ WAPE {v['mean']:+.4f}  median {v['median']:+.4f}  "
        f"stderr {v['stderr']:.4f}  MDE {2 * v['stderr']:.4f}"
    )
    print(f"  decisive={v['decisive']}  winner={v['winner']}  — {v['reason']}")
    sat = satisficing_check(
        treatment_bias_pct=float(aff.bias_treatment.mean()),
        control_mape=float(aff.mape_control.mean()),
        treatment_mape=float(aff.mape_treatment.mean()),
    )
    print(f"  satisficing: passed={sat['passed']} {sat['failures']}")
    print(
        f"  mean divergence {aff.divergence_pct.mean():.3f}% of demand, "
        f"max {aff.divergence_pct.max():.3f}%"
    )


if __name__ == "__main__":
    main()
