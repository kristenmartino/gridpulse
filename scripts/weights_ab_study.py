#!/usr/bin/env python3
"""#451 — pre-registered A/B: do smoothed holdout MAPEs make better ensemble weights?

The question P2-17 (#273) deliberately left undecided. `docs/HOLDOUT_STABILITY_STUDY.md`
established that the holdout estimator flaps (median 12% run-to-run, p90 43%) and that
ADR-004 weights move 0.220 L1/day — but stability is not the objective. **Accuracy is**,
and nothing measured so far answers whether smoothed weights *forecast* better.

## Why this needs no training runs

#451 assumed ~3h of training per arm. It does not need any, because the arms differ
**only in the weight vector** — the per-model forecasts are identical across arms. So
the expensive part is paid once:

1. GCS holds every daily model vintage back to 2026-04-19 (~3,500 for these 12 BAs).
2. For each origin, load the vintage that was *live at that moment* and predict the
   168h forward window through the production serve path — a **replay**, not a retrain.
3. Score each arm's blend of those same predictions.

That also makes it the serve regime rather than the holdout regime, which is what
production actually does with these weights.

## Pre-registered design (issue #451, quoted, not re-chosen here)

* **Arms:** `raw` (control, today) vs `ewma_0.3` vs `ewma_0.5`
* **Optimising metric:** WAPE. **Reported:** MAPE.
* **Windows:** 8 rolling origins x 168h, non-overlapping
* **Satisficing:** `|bias| <= 2.0%`, MAPE regression <= 0.5 pts; unmeasurable == failed
* **Decisive:** >=4 windows, `|mean| >= 2*stderr`, >=75% sign consistency, mean/median agree
* **Ship criterion:** a decisive WAPE win with both constraints satisfied. Anything
  else — including inconclusive — ships nothing.

Every verdict routes through `models/rolling_eval.py` per CLAUDE.md.

## Known limitations, stated before the result

* **Weather regressors are partly imputed on deep history.** The ERA5 archive endpoint
  lacks `wind_speed_80m/120m` and `soil_temperature_0cm` (documented in
  `data/weather_client`), so replayed windows carry imputed values where production's
  forward run had real forecast wind. This is identical across arms, so the *paired*
  comparison holds; the absolute error levels are not production's.
* **Rolling origins share training data**, so the t-statistic is a decision rule, not a
  significance claim — `rolling_eval.verdict` says so itself.

Usage:
    python scripts/weights_ab_study.py --out docs/WEIGHTS_AB_STUDY.md
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import sys
import time
from typing import Any

os.environ.setdefault("ENVIRONMENT", "production")
os.environ.setdefault("PRECOMPUTE_ENABLED", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BUCKET = "nextera-portfolio-energy-cache"
MODEL_PREFIX = "cache/models"

#: The 12 BAs of docs/HOLDOUT_STABILITY_STUDY.md — deliberately the same set, so the
#: churn measurement and this accuracy measurement describe one population.
BAS = ["SPA", "IID", "AZPS", "SEC", "HST", "PSCO", "ERCOT", "PJM", "MISO", "FPL", "CAISO", "BPAT"]
MODELS = ("xgboost", "prophet", "arima")

WINDOW_HOURS = 168
N_WINDOWS = 8
ALPHAS = (0.3, 0.5)


def ewma(values: list[float], alpha: float) -> float | None:
    """Exponentially weighted mean, most recent observation weighted `alpha`."""
    vals = [float(v) for v in values if v is not None and np.isfinite(v) and v > 0]
    if not vals:
        return None
    out = vals[0]
    for v in vals[1:]:
        out = alpha * v + (1 - alpha) * out
    return float(out)


def load_mape_history(client: Any) -> dict[str, list[dict]]:
    """Per-(BA, model) holdout-MAPE history from the GCS metas. No pickles."""
    from google.cloud import storage  # noqa: F401  (client already constructed)

    bucket = client.bucket(BUCKET)
    hist: dict[str, list[dict]] = {}
    for ba in BAS:
        for m in MODELS:
            rows = []
            for b in client.list_blobs(bucket, prefix=f"{MODEL_PREFIX}/{ba}/{m}/"):
                if not b.name.endswith(".meta.json"):
                    continue
                version = b.name.rsplit("/", 1)[-1].replace(".meta.json", "")
                if version == "latest":
                    continue
                try:
                    meta = json.loads(b.download_as_bytes())
                except Exception:
                    continue
                rows.append(
                    {
                        "version": version,
                        "trained_at": meta.get("trained_at"),
                        "mape": meta.get("mape"),
                    }
                )
            rows = [r for r in rows if r["trained_at"]]
            rows.sort(key=lambda r: r["trained_at"])
            hist[f"{ba}|{m}"] = rows
    return hist


def arm_weights(
    history: dict[str, list[dict]],
    ba: str,
    origin: pd.Timestamp,
    members: list[str],
    arm: str,
) -> tuple[dict[str, float], str]:
    """The weights each arm would have used at `origin`, for `members`.

    Only vintages trained **strictly before** the origin are visible, which is the
    whole point — an arm that could see the window it is scored on would be the
    in-sample reasoning `ledger-23` was fixed for.
    """
    from models.ensemble import resolve_ensemble_weights

    scores: dict[str, float | None] = {}
    for m in members:
        past = [
            r
            for r in history.get(f"{ba}|{m}", [])
            if pd.Timestamp(r["trained_at"]) < origin and r["mape"] is not None
        ]
        if not past:
            scores[m] = None
        elif arm == "raw":
            scores[m] = float(past[-1]["mape"])
        else:
            scores[m] = ewma([r["mape"] for r in past], float(arm.split("_")[1]))
    return resolve_ensemble_weights(members, scores)


def replay_region(ba: str, origins: list[pd.Timestamp], history, client) -> list[dict]:
    """Per-model forward predictions for every origin of one BA. One fetch, N replays."""
    from data.eia_client import fetch_demand
    from data.feature_engineering import engineer_features, recursive_autoregressive_forecast
    from data.preprocessing import merge_demand_weather
    from data.weather_client import fetch_weather
    from models.arima_model import predict_arima
    from models.prophet_model import predict_prophet
    from models.xgboost_model import predict_xgboost

    bucket = client.bucket(BUCKET)
    try:
        featured = engineer_features(merge_demand_weather(fetch_demand(ba), fetch_weather(ba, 92)))
    except Exception as e:  # pragma: no cover — one BA's upstream, not the study's
        print(f"  {ba}: fetch/feature failed: {e}")
        return []
    featured["timestamp"] = pd.to_datetime(featured["timestamp"], utc=True)

    out = []
    for origin in origins:
        end = origin + pd.Timedelta(hours=WINDOW_HOURS)
        past = featured[featured["timestamp"] < origin]
        window = featured[(featured["timestamp"] >= origin) & (featured["timestamp"] < end)]
        if len(window) < WINDOW_HOURS // 2 or len(past) < 336:
            print(
                f"  {ba} {origin.date()}: insufficient rows (win {len(window)}, past {len(past)})"
            )
            continue
        y = np.asarray(window["demand_mw"].values, dtype=float)
        preds: dict[str, np.ndarray] = {}
        for model_name in MODELS:
            eligible = [
                r
                for r in history.get(f"{ba}|{model_name}", [])
                if pd.Timestamp(r["trained_at"]) < origin
            ]
            if not eligible:
                continue
            version = eligible[-1]["version"]
            try:
                raw = bucket.blob(
                    f"{MODEL_PREFIX}/{ba}/{model_name}/{version}.pkl"
                ).download_as_bytes()
                model = pickle.load(io.BytesIO(raw))
                if model_name == "xgboost":
                    p = recursive_autoregressive_forecast(
                        model, past["demand_mw"].tolist(), window, predict_xgboost
                    )
                elif model_name == "prophet":
                    p = predict_prophet(model, window, periods=len(window))["forecast"]
                else:
                    r = predict_arima(model, window, periods=len(window))
                    p = r["forecast"] if isinstance(r, dict) else r
                p = np.asarray(p, dtype=float)[: len(y)]
            except Exception as e:
                print(f"  {ba} {origin.date()} {model_name}: {type(e).__name__}: {e}")
                continue
            if len(p) == len(y) and np.isfinite(p).all() and (p > 0).any():
                preds[model_name] = p
        if preds:
            out.append({"region": ba, "origin": origin, "actual": y, "preds": preds})
            print(f"  {ba} {origin.date()}: {len(preds)} models, {len(y)}h")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write the markdown report here")
    ap.add_argument("--end", default=None, help="last window end, ISO (default: today 00Z)")
    ap.add_argument("--bas", default=None, help="comma-separated override of the BA set")
    ap.add_argument(
        "--cache",
        default="/tmp/ab451_cache.pkl",
        help="replay cache. The replay is the expensive half (~15 min) and is "
        "independent of how the arms are scored, so every robustness re-cut reuses it.",
    )
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and re-replay")
    args = ap.parse_args()

    from google.cloud import storage

    from models.evaluation import compute_all_metrics
    from models.rolling_eval import bias_pct, paired_deltas, satisficing_check, verdict, wape

    global BAS
    if args.bas:
        BAS = [b.strip().upper() for b in args.bas.split(",") if b.strip()]

    end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.utcnow().normalize()
    origins = sorted(end - pd.Timedelta(hours=WINDOW_HOURS * (i + 1)) for i in range(N_WINDOWS))
    print(f"origins: {origins[0].date()} → {origins[-1].date()} ({len(origins)} x {WINDOW_HOURS}h)")

    t0 = time.time()
    if not args.refresh and os.path.exists(args.cache):
        with open(args.cache, "rb") as f:
            blob = pickle.load(f)
        history, cases = blob["history"], blob["cases"]
        print(f"loaded {len(cases)} cases from {args.cache} ({time.time() - t0:.0f}s)")
    else:
        client = storage.Client()
        print("reading MAPE history from GCS metas…")
        history = load_mape_history(client)
        print(f"  {sum(len(v) for v in history.values())} vintages ({time.time() - t0:.0f}s)")
        cases = []
        for ba in BAS:
            print(f"replaying {ba}…")
            cases.extend(replay_region(ba, origins, history, client))
        with open(args.cache, "wb") as f:
            pickle.dump({"history": history, "cases": cases}, f)
        print(f"\n{len(cases)} (BA, window) cases replayed in {time.time() - t0:.0f}s")
    if not cases:
        print("nothing replayed — cannot decide")
        return 1

    arms = ["raw"] + [f"ewma_{a}" for a in ALPHAS]
    per_case: list[dict] = []
    for c in cases:
        members = sorted(c["preds"])
        row = {"region": c["region"], "origin": c["origin"], "members": members}
        for arm in arms:
            w, rule = arm_weights(history, c["region"], c["origin"], members, arm)
            blend = np.zeros_like(c["actual"], dtype=float)
            for m in members:
                blend += w.get(m, 0.0) * c["preds"][m]
            row[arm] = {
                "weights": {k: round(v, 4) for k, v in w.items()},
                "blend": blend,
                "rule": rule,
                "wape": wape(c["actual"], blend),
                "mape": compute_all_metrics(c["actual"], blend)["mape"],
                "bias": bias_pct(c["actual"], blend),
            }
        per_case.append(row)

    # Window-level aggregate: mean across BAs, so each origin contributes one paired
    # observation — the unit `verdict()` is built for.
    by_origin: dict[Any, list[dict]] = {}
    for r in per_case:
        by_origin.setdefault(r["origin"], []).append(r)
    origins_sorted = sorted(by_origin)

    def series(arm: str, metric: str) -> list[float]:
        return [float(np.mean([r[arm][metric] for r in by_origin[o]])) for o in origins_sorted]

    report: dict[str, Any] = {
        "origins": [str(o) for o in origins_sorted],
        "n_cases": len(per_case),
        "bas": sorted({r["region"] for r in per_case}),
        "arms": {},
    }
    control_wape = series("raw", "wape")
    control_mape = series("raw", "mape")
    print(f"\ncontrol (raw) WAPE by window: {[round(x, 3) for x in control_wape]}")

    for arm in arms[1:]:
        t_wape, t_mape, t_bias = series(arm, "wape"), series(arm, "mape"), series(arm, "bias")
        deltas = paired_deltas(control_wape, t_wape)
        v = verdict(deltas)
        # Signed mean bias, not mean |bias| — the constraint is about systematic
        # direction (MAPE's asymmetry biases toward under-forecasting demand),
        # and averaging absolute values would hide a well-centred arm.
        sat = satisficing_check(
            treatment_bias_pct=float(np.mean(t_bias)),
            control_mape=float(np.mean(control_mape)),
            treatment_mape=float(np.mean(t_mape)),
        )
        report["arms"][arm] = {
            "wape_by_window": [round(x, 4) for x in t_wape],
            "delta_wape": [round(x, 4) for x in deltas],
            "verdict": v,
            "satisficing": sat,
            "mean_mape_control": round(float(np.mean(control_mape)), 3),
            "mean_mape_treatment": round(float(np.mean(t_mape)), 3),
            "mean_bias": round(float(np.mean(t_bias)), 3),
            "mean_abs_bias": round(float(np.mean(np.abs(t_bias))), 3),
        }
        ships = bool(v["decisive"] and v["winner"] == "treatment" and sat.get("passed"))
        report["arms"][arm]["ships"] = ships
        print(f"\n=== {arm} vs raw ===")
        print(f"  deltas (WAPE, + = smoothing better): {[round(x, 3) for x in deltas]}")
        print(f"  verdict: {v}")
        print(f"  satisficing: {sat}")
        print(f"  SHIPS: {ships}")

    # ── sensitivity + confound, run every time ──────────────────────────
    # Neither is a pre-registered criterion and neither may overturn the verdict.
    # They are published because a decisive result whose fragility is unstated is
    # a number nobody can re-check.
    print("\n=== sensitivity: leave-one-out (NOT a ship criterion — disclosure) ===")
    for arm in arms[1:]:

        def refit(drop_ba=None, drop_origin=None, _arm=arm):
            out = []
            for o in origins_sorted:
                if o == drop_origin:
                    continue
                sel = [r for r in by_origin[o] if r["region"] != drop_ba]
                if sel:
                    out.append(
                        float(np.mean([r["raw"]["wape"] for r in sel]))
                        - float(np.mean([r[_arm]["wape"] for r in sel]))
                    )
            return np.asarray(out)

        broke_ba = [
            ba
            for ba in sorted({r["region"] for r in per_case})
            if not ((vv := verdict(refit(drop_ba=ba)))["decisive"] and vv["winner"] == "treatment")
        ]
        broke_win = [
            str(o)[:10]
            for o in origins_sorted
            if not (
                (vv := verdict(refit(drop_origin=o)))["decisive"] and vv["winner"] == "treatment"
            )
        ]
        report["arms"][arm]["fragile_to_dropping_ba"] = broke_ba
        report["arms"][arm]["fragile_to_dropping_window"] = broke_win
        print(f"  {arm}: breaks when dropping BA {broke_ba or 'none'}")
        print(f"  {arm}: breaks when dropping window {broke_win or 'none'}")

    # Confound: smoothing compresses the spread between models' MAPEs, which could
    # simply flatten the blend. If a lower ADR-004 exponent on RAW MAPEs reproduced
    # the gain, the finding would be about the exponent, not about smoothing.
    print("\n=== confound: is the gain just a less concentrated blend? ===")

    def hhi(weight_dicts: list[dict]) -> float:
        return float(np.mean([sum(v * v for v in w.values()) for w in weight_dicts]))

    def score_exponent(k: float) -> tuple[np.ndarray, list[dict]]:
        per_origin: dict[Any, list] = {}
        for c in cases:
            members = sorted(c["preds"])
            raw_mape = {
                m: (lambda past: float(past[-1]["mape"]) if past else None)(
                    [
                        r
                        for r in history.get(f"{c['region']}|{m}", [])
                        if pd.Timestamp(r["trained_at"]) < c["origin"] and r["mape"] is not None
                    ]
                )
                for m in members
            }
            usable = {m: v for m, v in raw_mape.items() if v and np.isfinite(v) and v > 0}
            if len(usable) != len(members):
                w = {m: 1 / len(members) for m in members}
            else:
                inv = {m: (1.0 / v) ** k for m, v in usable.items()}
                tot = sum(inv.values())
                w = {m: v / tot for m, v in inv.items()}
            blend = np.zeros_like(c["actual"], dtype=float)
            for m in members:
                blend += w[m] * c["preds"][m]
            per_origin.setdefault(c["origin"], []).append((wape(c["actual"], blend), w))
        os_ = sorted(per_origin)
        return (
            np.asarray([np.mean([x[0] for x in per_origin[o]]) for o in os_]),
            [w for o in os_ for _, w in per_origin[o]],
        )

    base_series, base_w = score_exponent(3.0)
    report["confound"] = {"control_hhi": round(hhi(base_w), 4), "exponents": {}}
    print(f"  control raw/k=3: WAPE {base_series.mean():.3f}  concentration(HHI) {hhi(base_w):.3f}")
    for arm in arms[1:]:
        w_arm = [r[arm]["weights"] for r in per_case]
        print(f"  {arm}: concentration(HHI) {hhi(w_arm):.3f}")
        report["arms"][arm]["hhi"] = round(hhi(w_arm), 4)
    for k in (2.5, 2.0, 1.5, 1.0):
        s_k, w_k = score_exponent(k)
        v_k = verdict(base_series - s_k)
        report["confound"]["exponents"][str(k)] = {
            "wape": round(float(s_k.mean()), 4),
            "hhi": round(hhi(w_k), 4),
            "delta_vs_k3": round(float((base_series - s_k).mean()), 4),
            "winner": v_k["winner"],
        }
        print(
            f"  raw/k={k}: WAPE {s_k.mean():.3f}  HHI {hhi(w_k):.3f}  "
            f"delta {(base_series - s_k).mean():+.3f}  winner={v_k['winner']}"
        )

    with open("/tmp/ab451_per_case.pkl", "wb") as f:
        pickle.dump({"per_case": per_case, "cases": cases}, f)
    with open("/tmp/ab451_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nwrote /tmp/ab451_report.json")
    if args.out:
        print(f"(markdown for {args.out} is written by the caller from this JSON)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
