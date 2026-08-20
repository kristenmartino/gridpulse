"""#559: settle the seed question by injecting gaps instead of waiting for them.

Pre-registered in ``docs/POSITIONAL_LAG_INJECTION_PREREGISTRATION.md``. Read that
first — the hypothesis, run-length distribution, strata, n, confirmation criteria
and stopping rule were all fixed before this file existed, and the git history
shows the ordering.

The observational study could not settle accuracy and cannot be made to: gaps are
rare, so a verdict is 1.2-6.6 years out. Injection removes the rate limit, and
the trade is unusually cheap here because the defect is deterministic —
``dropna`` deletes the row whose lag source was null and the snapshot then
indexes by position, so an injected null hour and an EIA-dropped one cause the
same deletion and the same index shift.

Two arms, one difference, on identical injected data:

  control    positional seed (what production serves)
  treatment  temporal seed (``temporal_ar_seed``)

Scored against the TRUE demand, which we still have because we injected the gap
rather than losing the data.

Usage:
    PYTHONPATH=. python scripts/positional_lag_injection_study.py \
        --stratum A --out /tmp/inj_A.json
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

# Frozen in the pre-registration: the empirical run-length distribution measured
# 2026-08-20 across all 51 mirrors over 90 days. Not a convenient shape.
GAP_RUNS: list[int] = [1] * 25 + [2] * 2 + [3] + [13] + [16] + [24]

STRATUM_A = ["LGEE", "PSCO", "TIDC", "IID", "NWMT", "NEVP", "SPA"]
STRATUM_B = [
    "MISO",
    "PJM",
    "ERCOT",
    "CAISO",
    "SPP",
    "DUK",
    "ISONE",
    "NYISO",
    "FPL",
    "TVA",
    "BPAT",
    "PACE",
]

HORIZON = 48
LOOKBACK = 168
WARMUP = LOOKBACK + 24

#: Criterion 4 of the re-run pre-registration: the NaN-lag rate must be
#: RE-MEASURED here, on this run's scored windows, not inherited from #615's own
#: reporting. These are the four point lags that land in the row and are then
#: zero-filled by ``row.fillna(0)`` — the #129 poison the absent-hour policy
#: exists to remove.
POINT_LAG_KEYS = ("demand_lag_1h", "demand_lag_3h", "demand_lag_24h", "demand_lag_168h")

#: Per-call flags recorded by the instrumented snapshot, cleared per arm.
_SNAPSHOT_FLAGS: list[tuple[bool, bool]] = []


def install_nan_lag_probe() -> None:
    """Wrap the temporal snapshot so every call records whether it NaN'd.

    Pure observation: the wrapper returns the callee's dict unchanged, so the
    treatment arm is byte-identical to an uninstrumented run. The caller asserts
    the expected call count afterwards — a probe that silently failed to
    intercept would otherwise report 0.00% for the wrong reason, which is
    exactly the failure mode criterion 4 exists to rule out.
    """
    import data.feature_engineering as fe

    if getattr(fe.compute_temporal_autoregressive_snapshot, "_nan_lag_probe", False):
        return
    inner = fe.compute_temporal_autoregressive_snapshot

    def wrapped(history, now):
        snap = inner(history, now)
        _SNAPSHOT_FLAGS.append(
            (
                any(np.isnan(snap[k]) for k in POINT_LAG_KEYS),
                any(isinstance(v, float) and np.isnan(v) for v in snap.values()),
            )
        )
        return snap

    wrapped._nan_lag_probe = True
    fe.compute_temporal_autoregressive_snapshot = wrapped


def _client():
    from google.cloud import storage

    return storage.Client()


def nan_lag_probe_selftest() -> tuple[bool, bool]:
    """Positive and negative controls for the criterion-4 probe, run before results.

    Criterion 4 is satisfied by a **zero**, and a probe that silently flagged
    nothing would produce the same zero. The call-count assertion in
    ``run_region`` proves the wrapper intercepted; it does not prove the wrapper
    can ever report non-zero. So before any window is scored, drive the wrapped
    snapshot through two histories with known answers:

    * **designed to disagree** — one observed hour, then a query 200 hours later.
      Both imputation regimes must fail there (no neighbour within 6, no same
      clock hour within 7 days), so the point lags are genuinely NaN and the
      probe MUST flag it.
    * **designed to agree** — a dense contiguous history, where nothing is
      absent and the probe MUST NOT flag it.

    Returns:
        ``(flagged_on_nan, flagged_on_dense)`` — criterion 4's measurement is
        only readable when this is ``(True, False)``.
    """
    import data.feature_engineering as fe

    base = pd.Timestamp("2026-01-01T00:00:00Z")

    _SNAPSHOT_FLAGS.clear()
    sparse = fe.HourIndexedHistory.build([base], [1000.0], extra_hours=400)
    fe.compute_temporal_autoregressive_snapshot(sparse, base + pd.Timedelta(hours=200))
    flagged_on_nan = bool(_SNAPSHOT_FLAGS and _SNAPSHOT_FLAGS[-1][0])

    _SNAPSHOT_FLAGS.clear()
    stamps = [base + pd.Timedelta(hours=h) for h in range(400)]
    dense = fe.HourIndexedHistory.build(stamps, [1000.0 + h for h in range(400)])
    fe.compute_temporal_autoregressive_snapshot(dense, base + pd.Timedelta(hours=399))
    flagged_on_dense = bool(_SNAPSHOT_FLAGS and _SNAPSHOT_FLAGS[-1][0])

    _SNAPSHOT_FLAGS.clear()
    return flagged_on_nan, flagged_on_dense


def load_mirror(kind: str, region: str) -> pd.DataFrame:
    if MIRROR:
        return pd.read_parquet(f"{MIRROR}/{kind}/{region}/latest.parquet")
    blob = _client().bucket(BUCKET).blob(f"cache/{kind}/{region}/latest.parquet")
    return pd.read_parquet(io.BytesIO(blob.download_as_bytes()))


def vintage_index(client, region: str, model_name: str = "xgboost") -> list[tuple]:
    out = []
    for b in client.list_blobs(BUCKET, prefix=f"{MODEL_PREFIX}/{region}/{model_name}/"):
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


def inject_gap(demand: pd.DataFrame, origin: pd.Timestamp, rng) -> tuple[pd.DataFrame, dict]:
    """NaN a run of hours inside the origin's 168h lookback.

    Values on present rows, which is the shape EIA actually produces (7 absent
    rows fleet-wide against 78 null ones), so this reproduces the dominant
    mechanism rather than the convenient one.
    """
    out = demand.copy()
    ts = pd.to_datetime(out["timestamp"])
    length = int(rng.choice(GAP_RUNS))
    lo = origin - pd.Timedelta(hours=LOOKBACK)
    hi = origin - pd.Timedelta(hours=length + 1)
    eligible = ts[(ts >= lo) & (ts <= hi)]
    if eligible.empty:
        return out, {}
    start = pd.Timestamp(rng.choice(eligible.to_numpy()))
    mask = (ts >= start) & (ts < start + pd.Timedelta(hours=length))
    out.loc[mask, "demand_mw"] = np.nan
    return out, {
        "gap_len": length,
        "gap_start": start.isoformat(),
        "gap_hour_utc": int(start.hour),
        "gap_lead_h": int((origin - start).total_seconds() // 3600),
    }


def run_region(client, region: str, vintages: list[tuple], *, seed: int) -> list[dict]:
    from config import FEATURE_FLAGS
    from data.feature_engineering import engineer_features, recursive_autoregressive_forecast
    from data.preprocessing import merge_demand_weather
    from jobs.phases import _build_future_feature_frame
    from models.xgboost_model import predict_xgboost

    demand = load_mirror("demand", region)
    weather = load_mirror("weather", region)
    ts_all = pd.to_datetime(demand["timestamp"]).reset_index(drop=True)
    truth = demand.dropna(subset=["demand_mw"]).copy()
    truth = truth.set_index(pd.to_datetime(truth["timestamp"]))["demand_mw"]

    rows: list[dict] = []
    n_windows = (len(ts_all) - WARMUP - HORIZON) // HORIZON
    for w in range(max(0, n_windows)):
        rng = np.random.default_rng(seed + w)
        origin = ts_all.iloc[WARMUP + w * HORIZON]
        elig = [v for t, v in vintages if t < origin]
        if not elig:
            continue
        try:
            model = load_vintage(client, region, elig[-1])
        except Exception:
            continue

        injected, meta = inject_gap(demand, origin, rng)
        if not meta:
            continue

        arms: dict[str, np.ndarray] = {}
        nan_lag_steps = nan_any_steps = 0
        ok = True
        for arm, temporal in (("control", False), ("treatment", True)):
            feat = engineer_features(merge_demand_weather(injected, weather))
            fts = pd.to_datetime(feat["timestamp"])
            past = feat[fts < origin]
            if len(past) < LOOKBACK + 24:
                ok = False
                break
            future = _build_future_feature_frame(
                past, HORIZON, weather_df=weather, start_ts=origin
            ).iloc[:HORIZON]
            if len(future) < HORIZON:
                ok = False
                break
            FEATURE_FLAGS["temporal_ar_seed"] = temporal
            _SNAPSHOT_FLAGS.clear()
            try:
                arms[arm] = recursive_autoregressive_forecast(
                    model,
                    past["demand_mw"].tolist(),
                    future.copy(),
                    predict_xgboost,
                    seed_timestamps=past["timestamp"],
                )
            finally:
                FEATURE_FLAGS["temporal_ar_seed"] = False
            if temporal:
                # The recursion probes the snapshot once to resolve column
                # positions, then calls it once per step: 1 + HORIZON. Anything
                # else means the probe did not intercept the arm it claims to
                # measure, and a 0.00% rate from a defeated probe would be
                # indistinguishable from a real one. Fail loudly instead.
                if len(_SNAPSHOT_FLAGS) != HORIZON + 1:
                    raise RuntimeError(
                        f"nan-lag probe intercepted {len(_SNAPSHOT_FLAGS)} calls, "
                        f"expected {HORIZON + 1} — instrumentation is not measuring the arm"
                    )
                steps = _SNAPSHOT_FLAGS[1:]
                nan_lag_steps = int(sum(1 for point, _ in steps if point))
                nan_any_steps = int(sum(1 for _, anyk in steps if anyk))
        if not ok or len(arms) != 2:
            continue

        idx = pd.date_range(origin, periods=HORIZON, freq="h")
        act = truth.reindex(idx).to_numpy(dtype=float)
        c, t = arms["control"], arms["treatment"]
        good = np.isfinite(act) & np.isfinite(c) & np.isfinite(t)
        if good.sum() < HORIZON * 0.8:
            continue
        denom = float(np.abs(act[good]).sum())
        rows.append(
            {
                "region": region,
                "origin": origin.isoformat(),
                "version": elig[-1],
                "n": int(good.sum()),
                **meta,
                "identical": bool(np.array_equal(c[good], t[good])),
                "nan_lag_steps": nan_lag_steps,
                "nan_any_steps": nan_any_steps,
                "horizon_steps": HORIZON,
                "wape_control": float(100 * np.abs(act[good] - c[good]).sum() / denom),
                "wape_treatment": float(100 * np.abs(act[good] - t[good]).sum() / denom),
                "mape_control": float(100 * np.mean(np.abs((act[good] - c[good]) / act[good]))),
                "mape_treatment": float(100 * np.mean(np.abs((act[good] - t[good]) / act[good]))),
                "bias_control": float(100 * (c[good] - act[good]).sum() / denom),
                "bias_treatment": float(100 * (t[good] - act[good]).sum() / denom),
                "divergence_pct": float(100 * np.abs(t[good] - c[good]).sum() / denom),
            }
        )
    return rows


def _seed_is_contiguous(seed_ts: pd.Series, origin: pd.Timestamp, span: int = LOOKBACK) -> bool:
    """Whether positional indexing lands on the hours it intends to.

    Every lag and rolling window reaches at most ``span`` entries back and the
    recursion appends its own predictions contiguously, so the arms are
    byte-identical exactly when the last ``span`` seed entries are contiguous
    hours ending at ``origin - 1h``.
    """
    if len(seed_ts) < span:
        return False
    tail = seed_ts.iloc[-span:]
    return bool(
        tail.iloc[-1] == origin - pd.Timedelta(hours=1)
        and (tail.iloc[-1] - tail.iloc[0]) == pd.Timedelta(hours=span - 1)
    )


def null_control(client, region: str, vintages: list[tuple]) -> dict | None:
    """A GAP-FREE origin: the arms must be EXACTLY equal, or the harness is wrong.

    "No injected gap" is not the same as "no gap" — stratum A BAs carry real
    ones, which is the whole reason they are stratum A. So the origin is chosen
    by the contiguity predicate rather than assumed, and a BA with no clean
    origin in its window is skipped and said so, not silently passed.
    """
    from config import FEATURE_FLAGS
    from data.feature_engineering import engineer_features, recursive_autoregressive_forecast
    from data.preprocessing import merge_demand_weather
    from jobs.phases import _build_future_feature_frame
    from models.xgboost_model import predict_xgboost

    demand = load_mirror("demand", region)
    weather = load_mirror("weather", region)
    feat = engineer_features(merge_demand_weather(demand, weather))
    fts = pd.to_datetime(feat["timestamp"])

    # Walk back for an origin whose seed tail is genuinely contiguous.
    origin = None
    past = None
    for back in range(HORIZON, len(fts) - WARMUP, 12):
        cand = fts.iloc[-1] - pd.Timedelta(hours=back) + pd.Timedelta(hours=1)
        cand_past = feat[fts < cand]
        if _seed_is_contiguous(pd.to_datetime(cand_past["timestamp"]).reset_index(drop=True), cand):
            origin, past = cand, cand_past
            break
    if origin is None:
        return {"region": region, "max_abs_diff": None, "exact": None, "skipped": "no_clean_origin"}

    elig = [v for t, v in vintages if t < origin]
    if not elig:
        return None
    model = load_vintage(client, region, elig[-1])
    future = _build_future_feature_frame(past, HORIZON, weather_df=weather, start_ts=origin)
    future = future.iloc[:HORIZON]
    out = {}
    for arm, temporal in (("control", False), ("treatment", True)):
        FEATURE_FLAGS["temporal_ar_seed"] = temporal
        try:
            out[arm] = recursive_autoregressive_forecast(
                model,
                past["demand_mw"].tolist(),
                future.copy(),
                predict_xgboost,
                seed_timestamps=past["timestamp"],
            )
        finally:
            FEATURE_FLAGS["temporal_ar_seed"] = False
    diff = float(np.max(np.abs(out["treatment"] - out["control"])))
    return {"region": region, "max_abs_diff": diff, "exact": diff == 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stratum", choices=["A", "B"], required=True)
    ap.add_argument("--seed", type=int, default=559)
    ap.add_argument("--out", default="")
    ap.add_argument("--regions", default="")
    args = ap.parse_args()

    from models.rolling_eval import paired_deltas, satisficing_check, verdict

    regions = (
        args.regions.split(",")
        if args.regions
        else (STRATUM_A if args.stratum == "A" else STRATUM_B)
    )
    client = _client()
    install_nan_lag_probe()

    pos, neg = nan_lag_probe_selftest()
    print(f"-- nan-lag probe controls: flags a real NaN={pos}, flags a dense history={neg} --")
    if not pos or neg:
        print("   PROBE CONTROLS FAILED — criterion 4 cannot be measured. Run void.")
        return

    all_rows: list[dict] = []
    for i, region in enumerate(regions):
        vints = vintage_index(client, region)
        rows = run_region(client, region, vints, seed=args.seed + 1000 * i)
        all_rows.extend(rows)
        if rows:
            d = pd.DataFrame(rows)
            nan_rate = 100 * d.nan_lag_steps.sum() / d.horizon_steps.sum()
            print(
                f"{region:6} n={len(rows):3}  div={d.divergence_pct.mean():6.3f}%  "
                f"WAPE ctl={d.wape_control.mean():6.3f} trt={d.wape_treatment.mean():6.3f}  "
                f"nan-lag {int(d.nan_lag_steps.sum())}/{int(d.horizon_steps.sum())} "
                f"({nan_rate:.2f}%)",
                flush=True,
            )
        else:
            print(f"{region:6} n=0", flush=True)

    if not all_rows:
        print("no scored windows")
        return

    df = pd.DataFrame(all_rows)
    if args.out:
        df.to_json(args.out, orient="records", indent=2)

    # Criterion 2, checked before any comparison is reported.
    print("\n-- null control (no injected gap; arms must be EXACTLY equal) --")
    nulls = [null_control(client, r, vintage_index(client, r)) for r in regions[:3]]
    for nc in nulls:
        if nc and nc.get("skipped"):
            print(f"   {nc['region']:6} SKIPPED — {nc['skipped']}")
        elif nc:
            print(f"   {nc['region']:6} max|diff| {nc['max_abs_diff']:.10f}  exact={nc['exact']}")
    checked = [nc for nc in nulls if nc and nc.get("exact") is not None]
    if not checked:
        print("   NO NULL CONTROL COULD BE RUN — treat the comparison as unverified.")
    if any(not nc["exact"] for nc in checked):
        print("   NULL CONTROL FAILED — the harness is measuring something else. Run void.")
        return

    # Criterion 4, re-measured on THIS run's scored windows.
    tot_steps = int(df.horizon_steps.sum())
    tot_nan = int(df.nan_lag_steps.sum())
    tot_any = int(df.nan_any_steps.sum())
    print("\n-- criterion 4: NaN-lag rate on scored windows (treatment arm) --")
    print(
        f"   point lags NaN: {tot_nan}/{tot_steps} steps "
        f"({100 * tot_nan / tot_steps:.4f}%)  [criterion 4 requires 0.0000%]"
    )
    print(
        f"   any snapshot key NaN: {tot_any}/{tot_steps} steps ({100 * tot_any / tot_steps:.4f}%)"
    )
    worst = (
        df.groupby("region")[["nan_lag_steps", "horizon_steps"]]
        .sum()
        .assign(rate=lambda x: 100 * x.nan_lag_steps / x.horizon_steps)
        .sort_values("rate", ascending=False)
    )
    print(f"   worst BA: {worst.index[0]} at {worst['rate'].iloc[0]:.4f}%")

    n_identical = int(df.identical.sum())
    print(f"\n-- injected windows where the arms still matched: {n_identical}/{len(df)} --")
    print("   (a gap outside every lag's reach changes nothing; expected, not a fault)")

    print(f"\n-- stratum {args.stratum}: pooled paired windows (n={len(df)}) --")
    dl = paired_deltas(df.wape_control.to_numpy(), df.wape_treatment.to_numpy())
    v = verdict(dl)
    if v["mean"] is not None:
        print(
            f"   mean Δ WAPE {v['mean']:+.4f}  median {v['median']:+.4f}  "
            f"stderr {v['stderr']:.4f}  MDE {2 * v['stderr']:.4f}"
        )
        print(f"   sign consistency {v['sign_consistency']:.3f}  n={v['n']}")
    print(f"   decisive={v['decisive']}  winner={v['winner']}")
    print(f"   reason: {v['reason']}")

    sat = satisficing_check(
        treatment_bias_pct=float(df.bias_treatment.mean()),
        control_mape=float(df.mape_control.mean()),
        treatment_mape=float(df.mape_treatment.mean()),
    )
    print(
        f"   control bias {df.bias_control.mean():+.3f}%  treatment bias {df.bias_treatment.mean():+.3f}%"
    )
    print(f"   satisficing: passed={sat['passed']} {sat['failures']}")
    print(f"   divergence {df.divergence_pct.mean():.3f}% mean, {df.divergence_pct.max():.3f}% max")


if __name__ == "__main__":
    main()
