"""Forecast-dive diagnosis — why does a clean frame produce partial-band output?

On 2026-07-18 the live LDWP XGBoost forecast (scored 02:11Z) launched sanely
off the conditioned anchor (4,005 MW) then dove to 1,302 MW by 07Z — deep in
EIA's partial band, ~50% under the settled overnight trough — with every
input verified clean: settled history (guard-log proof), sane mirrored
weather (LA July night, 68-75F, zero NaNs), gap-free 90-day EIA view, and a
conditioned day-ahead anchor. Live 7d sMAPE 24% vs holdout 8.8%, so the
defect is chronic and lives in the model + serving recursion path.

The close-read narrowed the mechanism space (see the plan / PR body):
train/serve AR semantics are parity-clean, the recursion overwrites all 21
AR features, and the published holdout NEVER exercises the serve path
(different frame, different weather, freshly retrained model). This script
is the ablation ladder that names the mechanism — one dimension per rung:

    0. Reproduce the 02:11Z tick from mirrors + the deployed pickle.
    1. Teacher-forced 1h-ahead sweep on settled featured rows (the
       holdout's world). Low overnight => model miscalibration.
    2. Frame ablation over one past window: holdout-style featured slice
       vs serve-style ``_build_future_feature_frame``. A split names an
       exogenous frame column; both diving => recursion-compounding.
    3. Pickle vintages (0715/0716/0717) on the failing config.
    4. Anchor arms (conditioned trailing-3 / settled / single-hour) on the
       same past window — quantifies the seed's contribution.
    5. Per-step forensics: the exact rows the recursion fed the model,
       diffed against settled featured rows, plus one-group-at-a-time
       perturbation at the first diving step.
    6. SHAP on the dive rows (only reached if 5 is ambiguous).

Everything reads GCS mirrors via ADC (the anchor-study pattern): vintage
mirror for settled truth + first-seen DF, weather mirror for the frame the
job consumed, pickles fetched BY VERSION BLOB (``load_model`` only follows
``latest.json``). No live API calls.

Documented approximations: the weather mirror is the newest tick's stitch
(hour-scale vintage drift vs 02:11Z); conditioning uses the vintage
``first_seen_df`` where prod used the fetch-time ``forecast_mw`` (these
differ by at most a late DF revision); past-window "forecast" weather is
the stitched near-analysis, not the as-of vintage — fine, because weather
VALUES were already ruled out and rung 2 tests frame MECHANICS.

Usage:
    python scripts/forecast_dive_diagnosis.py                    # full ladder
    python scripts/forecast_dive_diagnosis.py --output docs/FORECAST_DIVE_DIAGNOSIS.md

Requires GCS_ENABLED=true, GCS_BUCKET_NAME, and ADC credentials.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402

#: The live curve this study must reproduce — read from the deployed
#: dashboard's 7d XGBOOST trace (forecast scored 2026-07-18 02:11 UTC),
#: hours 02Z..01Z. Provenance: Plotly bdata decode, session 2026-07-17.
PROD_CURVE_0718 = [
    4005,
    2911,
    2081,
    1614,
    1315,
    1302,
    1644,
    1741,
    1608,
    1651,
    1745,
    1796,
    1853,
    2004,
    2194,
    2393,
    2618,
    2857,
    3037,
    3158,
    3281,
    3361,
    3353,
    3403,
]
TICK = pd.Timestamp("2026-07-18T02:00:00Z")  #: forecast_start of that tick

#: A prediction "dives" when its first-24h trough undercuts the reference
#: trough by more than this fraction (settled LDWP troughs never move 25%
#: night-over-night; the observed dive is ~50%).
DIVE_TROUGH_FRACTION = 0.75


def _md_table(df: pd.DataFrame) -> str:
    """Minimal markdown table (the export_holdout_metrics precedent)."""
    if df.empty:
        return "(no rows)"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


# ── GCS plumbing ─────────────────────────────────────────────


def _load_vintage(region: str) -> list:
    from data.gcs_store import read_parquet
    from data.vintage import deserialize_records

    df = read_parquet("vintage", region)
    if df is None or df.empty:
        return []
    return deserialize_records(df.to_dict("records"))


def _load_weather(region: str) -> pd.DataFrame:
    from data.gcs_store import read_parquet

    df = read_parquet("weather", region)
    if df is None or df.empty:
        raise SystemExit(f"{region}: no weather mirror in GCS")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _list_versions(region: str, model_name: str) -> list[str]:
    """All persisted versions for (region, model), oldest first."""
    from models.persistence import _blob_path, _get_client

    client = _get_client()
    prefix = _blob_path(region, model_name, "", "")
    versions = sorted(
        b.name.rsplit("/", 1)[-1].removesuffix(".pkl")
        for b in client.list_blobs(config.GCS_BUCKET_NAME, prefix=prefix)
        if b.name.endswith(".pkl")
    )
    return versions


def _load_pickle_version(region: str, model_name: str, version: str) -> dict:
    """Fetch one model blob by exact version — ``load_model`` only follows
    ``latest.json``, and this study needs every vintage."""
    from models.persistence import _blob_path, _get_client

    client = _get_client()
    bucket = client.bucket(config.GCS_BUCKET_NAME)
    blob = bucket.blob(_blob_path(region, model_name, version, ".pkl"))
    return pickle.loads(blob.download_as_bytes())


# ── frame construction ───────────────────────────────────────


def _settled_series(records: list) -> dict[pd.Timestamp, float]:
    return {
        pd.Timestamp(r.timestamp): r.last_d
        for r in records
        if np.isfinite(r.last_d) and r.last_d > 0
    }


def _df_series(records: list) -> dict[pd.Timestamp, float]:
    return {
        pd.Timestamp(r.timestamp): r.first_seen_df for r in records if np.isfinite(r.first_seen_df)
    }


def _history_frame(
    region: str,
    settled: dict[pd.Timestamp, float],
    end_ts: pd.Timestamp,
    tail_override: dict[pd.Timestamp, float] | None = None,
) -> pd.DataFrame:
    """Settled demand through ``end_ts`` with an optional tail substitution
    (the conditioning arm: trailing hours -> day-ahead DF)."""
    hours = sorted(h for h in settled if h <= end_ts)
    values = [(tail_override or {}).get(h, settled[h]) for h in hours]
    return pd.DataFrame(
        {"timestamp": pd.to_datetime(hours, utc=True), "demand_mw": values, "region": region}
    )


def _featured(demand: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    from data.feature_engineering import engineer_features
    from data.preprocessing import merge_demand_weather

    merged = merge_demand_weather(demand, weather)
    return engineer_features(merged).dropna(subset=["demand_mw"]).reset_index(drop=True)


def _capture_recursion(
    model: dict, featured: pd.DataFrame, future_df: pd.DataFrame, horizon: int
) -> tuple[np.ndarray, list[pd.DataFrame]]:
    """The production recursive call, with every single-row frame the model
    actually saw captured via the predict_fn seam."""
    from data.feature_engineering import recursive_autoregressive_forecast
    from models.xgboost_model import predict_xgboost

    rows: list[pd.DataFrame] = []

    def _predict(m: dict, x: pd.DataFrame) -> np.ndarray:
        if len(x) == 1:
            rows.append(x.copy())
        return predict_xgboost(m, x)

    preds = recursive_autoregressive_forecast(
        model, featured["demand_mw"].tolist(), future_df.iloc[:horizon], _predict
    )
    return np.asarray(preds, dtype=float), rows


def _gain_importance(model: dict) -> dict[str, float]:
    booster = model["model"].get_booster()
    raw = booster.get_score(importance_type="gain")
    names = model["feature_names"]
    # Booster features are f0..fN in feature_names order (trained on .values).
    return {names[int(k[1:])]: v for k, v in raw.items() if k.startswith("f")}


def _curve_stats(preds: np.ndarray, reference: list[float] | np.ndarray) -> dict:
    ref = np.asarray(reference, dtype=float)
    n = min(len(preds), len(ref), 24)
    p = preds[:n]
    return {
        "launch": round(float(p[0])),
        "trough": round(float(p.min())),
        "trough_ref": round(float(ref[:n].min())),
        "level_ratio": round(float(p.mean() / ref[:n].mean()), 3),
        "dives": bool(p.min() < DIVE_TROUGH_FRACTION * ref[:n].min()),
    }


# ── rungs ────────────────────────────────────────────────────


def rung0_reproduce(region: str, model: dict, records: list, weather: pd.DataFrame) -> dict:
    """Rebuild the 02:11Z tick: settled history + conditioned trailing-3
    anchor + mirror weather through the real serve chain."""
    from jobs.phases import _build_future_feature_frame

    settled = _settled_series(records)
    dfv = _df_series(records)
    anchor_end = TICK - pd.Timedelta(hours=1)
    tail = {h: dfv[h] for h in [anchor_end - pd.Timedelta(hours=i) for i in range(3)] if h in dfv}
    demand = _history_frame(region, settled, anchor_end, tail_override=tail)
    featured = _featured(demand, weather)
    future = _build_future_feature_frame(featured, 168, weather_df=weather, start_ts=TICK)
    preds, rows = _capture_recursion(model, featured, future, 168)
    stats = _curve_stats(preds, PROD_CURVE_0718)
    stats["prod_trough"] = min(PROD_CURVE_0718)
    stats["max_dev_vs_prod_pct"] = round(
        float(
            np.max(
                np.abs(preds[: len(PROD_CURVE_0718)] - np.asarray(PROD_CURVE_0718))
                / np.asarray(PROD_CURVE_0718)
            )
        )
        * 100.0,
        1,
    )
    stats["preds_24h"] = [round(float(v)) for v in preds[:24]]
    stats["_rows"] = rows
    stats["_featured"] = featured
    return stats


def rung1_teacher_forced(model: dict, featured: pd.DataFrame, hours: int = 72) -> dict:
    """Direct 1h-ahead predictions on settled featured rows — real AR
    features, observed weather: the holdout's world. If THESE are low the
    model itself is miscalibrated."""
    from models.xgboost_model import predict_xgboost

    tail = featured.tail(hours)
    preds = predict_xgboost(model, tail)
    actual = tail["demand_mw"].to_numpy(dtype=float)
    ape = np.abs(preds - actual) / actual * 100.0
    ts = pd.to_datetime(tail["timestamp"]).dt.hour  # UTC
    overnight = (ts >= 5) & (ts <= 15)  # LA night/morning
    return {
        "n": len(tail),
        "mape_all": round(float(ape.mean()), 2),
        "mape_overnight": round(float(ape[overnight].mean()), 2),
        "mape_day": round(float(ape[~overnight].mean()), 2),
        "worst_pct": round(float(ape.max()), 1),
        "bias_mw": round(float((preds - actual).mean())),
    }


def rung2_frame_ablation(
    region: str,
    model: dict,
    records: list,
    weather: pd.DataFrame,
    start: pd.Timestamp,
    horizon: int = 48,
) -> dict:
    """Same pickle, same settled seed, recursion over the same window twice:
    (a) holdout-style featured slice vs (b) serve-style built frame."""
    from jobs.phases import _build_future_feature_frame

    settled = _settled_series(records)
    truth_hours = [start + pd.Timedelta(hours=i) for i in range(1, horizon + 1)]
    truth = np.asarray([settled.get(h, np.nan) for h in truth_hours], dtype=float)

    full = _featured(_history_frame(region, settled, truth_hours[-1]), weather)
    full_ts = pd.to_datetime(full["timestamp"], utc=True)
    seed = full[full_ts <= start].reset_index(drop=True)
    holdout_slice = full[full_ts > start].head(horizon).reset_index(drop=True)

    preds_a, rows_a = _capture_recursion(model, seed, holdout_slice, horizon)
    built = _build_future_feature_frame(
        seed, horizon, weather_df=weather, start_ts=start + pd.Timedelta(hours=1)
    )
    preds_b, rows_b = _capture_recursion(model, seed, built, horizon)

    ok = np.isfinite(truth) & (truth > 0)
    out = {
        "start": str(start),
        "holdout_frame": _curve_stats(preds_a[ok], truth[ok]),
        "built_frame": _curve_stats(preds_b[ok], truth[ok]),
        "holdout_mape": round(float(np.mean(np.abs(preds_a[ok] - truth[ok]) / truth[ok] * 100)), 2),
        "built_mape": round(float(np.mean(np.abs(preds_b[ok] - truth[ok]) / truth[ok] * 100)), 2),
        "_rows_a": rows_a,
        "_rows_b": rows_b,
        "_seed": seed,
        "_truth": truth,
    }

    # Column diff (exogenous only — the recursion owns the AR set), ranked
    # by |mean delta| weighted by the model's gain importance.
    from data.feature_engineering import AUTOREGRESSIVE_DEMAND_FEATURES

    gain = _gain_importance(model)
    diffs = []
    n = min(len(rows_a), len(rows_b), 24)
    for col in model["feature_names"]:
        if col in AUTOREGRESSIVE_DEMAND_FEATURES:
            continue
        va = np.array([float(rows_a[i][col].iloc[0]) for i in range(n) if col in rows_a[i]])
        vb = np.array([float(rows_b[i][col].iloc[0]) for i in range(n) if col in rows_b[i]])
        if len(va) != n or len(vb) != n:
            continue
        d = float(np.mean(np.abs(va - vb)))
        scale = float(np.nanstd(full[col])) if col in full else 1.0
        diffs.append(
            {
                "col": col,
                "mean_abs_delta": round(d, 3),
                "delta_in_sd": round(d / scale, 3) if scale > 0 else np.inf,
                "gain_weight": round(gain.get(col, 0.0), 1),
                "score": round((d / scale if scale > 0 else 0) * gain.get(col, 0.0), 1),
            }
        )
    out["column_diff_top"] = sorted(diffs, key=lambda r: -r["score"])[:12]
    return out


def rung3_vintages(
    region: str, records: list, weather: pd.DataFrame, start: pd.Timestamp, horizon: int = 48
) -> list[dict]:
    """The failing configuration (serve-style frame) across every persisted
    pickle vintage."""
    from jobs.phases import _build_future_feature_frame

    settled = _settled_series(records)
    truth_hours = [start + pd.Timedelta(hours=i) for i in range(1, horizon + 1)]
    truth = np.asarray([settled.get(h, np.nan) for h in truth_hours], dtype=float)
    full = _featured(_history_frame(region, settled, truth_hours[-1]), weather)
    seed = full[pd.to_datetime(full["timestamp"], utc=True) <= start].reset_index(drop=True)
    built = _build_future_feature_frame(
        seed, horizon, weather_df=weather, start_ts=start + pd.Timedelta(hours=1)
    )

    rows = []
    for version in _list_versions(region, "xgboost"):
        model = _load_pickle_version(region, "xgboost", version)
        preds, _ = _capture_recursion(model, seed, built, horizon)
        ok = np.isfinite(truth) & (truth > 0)
        stats = _curve_stats(preds[ok], truth[ok])
        rows.append(
            {
                "version": version,
                "launch": stats["launch"],
                "trough": stats["trough"],
                "trough_truth": stats["trough_ref"],
                "level_ratio": stats["level_ratio"],
                "mape": round(float(np.mean(np.abs(preds[ok] - truth[ok]) / truth[ok] * 100)), 2),
                "dives": stats["dives"],
            }
        )
    return rows


def rung4_anchor_arms(
    region: str,
    model: dict,
    records: list,
    weather: pd.DataFrame,
    start: pd.Timestamp,
    horizon: int = 48,
) -> list[dict]:
    """Seed-tail arms on the same past window: conditioned trailing-3 vs
    settled truth vs single-hour first-seen."""
    from jobs.phases import _build_future_feature_frame

    settled = _settled_series(records)
    dfv = _df_series(records)
    first_seen = {pd.Timestamp(r.timestamp): r.first_seen_d for r in records}
    truth_hours = [start + pd.Timedelta(hours=i) for i in range(1, horizon + 1)]
    truth = np.asarray([settled.get(h, np.nan) for h in truth_hours], dtype=float)
    tail3 = [start - pd.Timedelta(hours=i) for i in range(3)]

    arms = {
        "settled": {},
        "conditioned_3h": {h: dfv[h] for h in tail3 if h in dfv},
        "first_seen_1h": {start: first_seen.get(start, settled.get(start))},
    }
    out = []
    for arm, tail in arms.items():
        demand = _history_frame(region, settled, start, tail_override=tail)
        featured = _featured(demand, weather)
        built = _build_future_feature_frame(
            featured, horizon, weather_df=weather, start_ts=start + pd.Timedelta(hours=1)
        )
        preds, _ = _capture_recursion(model, featured, built, horizon)
        ok = np.isfinite(truth) & (truth > 0)
        stats = _curve_stats(preds[ok], truth[ok])
        out.append(
            {
                "arm": arm,
                "launch": stats["launch"],
                "trough": stats["trough"],
                "level_ratio": stats["level_ratio"],
                "mape": round(float(np.mean(np.abs(preds[ok] - truth[ok]) / truth[ok] * 100)), 2),
            }
        )
    return out


def rung5_forensics(
    model: dict,
    dive_rows: list[pd.DataFrame],
    reference_featured: pd.DataFrame,
    steps: int = 6,
) -> dict:
    """The exact rows the model saw, diffed against the settled featured row
    for the same timestamp, plus one-group-at-a-time perturbation at the
    first diving step."""
    from data.feature_engineering import AUTOREGRESSIVE_DEMAND_FEATURES
    from models.xgboost_model import predict_xgboost

    ref = reference_featured.copy()
    ref["timestamp"] = pd.to_datetime(ref["timestamp"], utc=True)
    ref = ref.set_index("timestamp")

    ar_cols = [c for c in AUTOREGRESSIVE_DEMAND_FEATURES if c in model["feature_names"]]
    exo_cols = [c for c in model["feature_names"] if c not in ar_cols and c in ref.columns]

    per_step = []
    perturbation = None
    for i, row in enumerate(dive_rows[:steps]):
        ts = pd.to_datetime(row["timestamp"].iloc[0], utc=True) if "timestamp" in row else None
        base_pred = float(predict_xgboost(model, row)[0])
        entry = {"step": i + 1, "ts": str(ts), "pred": round(base_pred)}
        if ts is not None and ts in ref.index:
            ref_row = ref.loc[ts]
            entry["actual"] = round(float(ref_row["demand_mw"]))
            # one-group-at-a-time swap toward the settled featured row
            swaps = {}
            for label, cols in (("ar_features", ar_cols), ("exogenous", exo_cols)):
                mod = row.copy()
                for c in cols:
                    mod[c] = float(ref_row[c])
                swaps[label] = round(float(predict_xgboost(model, mod)[0]))
            entry["pred_with_settled_ar"] = swaps["ar_features"]
            entry["pred_with_settled_exo"] = swaps["exogenous"]
            if perturbation is None and entry.get("actual") and base_pred < 0.8 * entry["actual"]:
                # first diving step: per-feature attribution of the gap
                deltas = []
                for c in ar_cols + exo_cols:
                    mod = row.copy()
                    mod[c] = float(ref_row[c])
                    deltas.append(
                        {
                            "feature": c,
                            "row_value": round(float(row[c].iloc[0]), 2),
                            "settled_value": round(float(ref_row[c]), 2),
                            "delta_pred": round(float(predict_xgboost(model, mod)[0]) - base_pred),
                        }
                    )
                perturbation = {
                    "step": i + 1,
                    "ts": str(ts),
                    "base_pred": round(base_pred),
                    "actual": entry["actual"],
                    "top_movers": sorted(deltas, key=lambda d: -abs(d["delta_pred"]))[:12],
                }
        per_step.append(entry)
    return {"per_step": per_step, "first_dive_perturbation": perturbation}


def rung0b_tick_vintage_sweep(
    region: str, records: list, weather: pd.DataFrame, versions: list[str]
) -> list[dict]:
    """The exact Jul-18 tick frame across pinned vintages — which pickle
    reproduces the live 1,302 trough? Adds a first-seen-history arm for the
    pickle that served it (falsifies the clean-frame proof if needed)."""
    from jobs.phases import _build_future_feature_frame

    settled = _settled_series(records)
    dfv = _df_series(records)
    anchor_end = TICK - pd.Timedelta(hours=1)
    tail = {h: dfv[h] for h in [anchor_end - pd.Timedelta(hours=i) for i in range(3)] if h in dfv}
    featured = _featured(_history_frame(region, settled, anchor_end, tail_override=tail), weather)
    future = _build_future_feature_frame(featured, 48, weather_df=weather, start_ts=TICK)

    out = []
    for version in versions:
        model = _load_pickle_version(region, "xgboost", version)
        preds, _ = _capture_recursion(model, featured, future, 48)
        stats = _curve_stats(preds, PROD_CURVE_0718)
        out.append(
            {
                "version": version,
                "arm": "settled+cond3",
                "launch": stats["launch"],
                "trough_24h": stats["trough"],
                "prod_trough": min(PROD_CURVE_0718),
                "max_dev_vs_prod_pct": round(
                    float(
                        np.max(
                            np.abs(
                                preds[: len(PROD_CURVE_0718)]
                                - np.asarray(PROD_CURVE_0718, dtype=float)
                            )
                            / np.asarray(PROD_CURVE_0718, dtype=float)
                        )
                    )
                    * 100.0,
                    1,
                ),
            }
        )
    return out


def rung3b_meta_correlation(region: str, records: list, versions: list[str]) -> pd.DataFrame:
    """Per-vintage meta (holdout MAPE the training job SAW) joined with a
    tail-contamination estimate: mirror hours first-captured partial-band
    within the 8h before that vintage's training timestamp."""
    import json

    from models.persistence import _blob_path, _get_client

    client = _get_client()
    bucket = client.bucket(config.GCS_BUCKET_NAME)

    partial_hours = []  # (captured_at, hour_ts) for partial-band first sights
    for r in records:
        if (
            np.isfinite(r.first_seen_d)
            and np.isfinite(r.last_d)
            and r.last_d > 0
            and r.first_seen_d < 0.6 * r.last_d
        ):
            partial_hours.append(pd.Timestamp(r.captured_at))

    rows = []
    for version in versions:
        blob = bucket.blob(_blob_path(region, "xgboost", version, ".meta.json"))
        try:
            meta = json.loads(blob.download_as_bytes())
        except Exception:
            continue
        trained = pd.Timestamp(meta["trained_at"])
        recent_partials = sum(
            1 for c in partial_hours if trained - pd.Timedelta(hours=8) <= c <= trained
        )
        rows.append(
            {
                "version": version,
                "holdout_mape": round(float(meta.get("mape", np.nan)), 2),
                "train_rows": meta.get("train_rows"),
                "partial_first_sights_8h_before_train": recent_partials if partial_hours else None,
            }
        )
    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--region", default="LDWP")
    ap.add_argument("--control", default="PNM")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--past-start",
        default=str(TICK - pd.Timedelta(hours=49)),
        help="anchor for the truth-scored rungs (default: 49h before the tick)",
    )
    args = ap.parse_args()

    if not config.GCS_ENABLED or not config.GCS_BUCKET_NAME:
        print("GCS access required: GCS_ENABLED=true, GCS_BUCKET_NAME, ADC login.")
        return 1

    past_start = pd.Timestamp(args.past_start)
    report: list[str] = ["# Forecast-dive diagnosis\n"]

    for region, is_control in ((args.region, False), (args.control, True)):
        tag = f"{region}{' (control)' if is_control else ''}"
        print(f"\n══ {tag} ══")
        records = _load_vintage(region)
        if not records:
            print("  no vintage mirror — skipped")
            continue
        weather = _load_weather(region)
        versions = _list_versions(region, "xgboost")
        model = _load_pickle_version(region, "xgboost", versions[-1])
        report.append(f"\n## {tag} — pickle {versions[-1]}\n")

        print("rung 0 — reproduce the tick…")
        r0 = rung0_reproduce(region, model, records, weather)
        pub0 = {k: v for k, v in r0.items() if not k.startswith("_") and k != "preds_24h"}
        print(f"  {pub0}")
        print(f"  first 24h: {r0['preds_24h']}")
        report.append(f"**Rung 0 — reproduction**: `{pub0}`\n")
        report.append(f"first 24h: `{r0['preds_24h']}`\n")

        print("rung 1 — teacher-forced 1h-ahead…")
        r1 = rung1_teacher_forced(model, r0["_featured"])
        print(f"  {r1}")
        report.append(f"\n**Rung 1 — teacher-forced 1h-ahead (settled rows)**: `{r1}`\n")

        print(f"rung 2 — frame ablation @ {past_start}…")
        r2 = rung2_frame_ablation(region, model, records, weather, past_start)
        pub2 = {k: v for k, v in r2.items() if not k.startswith("_") and k != "column_diff_top"}
        print(f"  {pub2}")
        report.append(f"\n**Rung 2 — frame ablation @ {past_start}**: `{pub2}`\n")
        cd = pd.DataFrame(r2["column_diff_top"])
        if not cd.empty:
            print(_md_table(cd))
            report.append("\nColumn diff (exogenous, importance-weighted):\n\n")
            report.append(_md_table(cd) + "\n")

        if is_control:
            continue  # the control only sanity-checks the harness

        print("rung 3 — pickle vintages…")
        r3 = pd.DataFrame(rung3_vintages(region, records, weather, past_start))
        print(_md_table(r3))
        report.append(f"\n**Rung 3 — vintages**:\n\n{_md_table(r3)}\n")

        print("rung 0b — tick frame × recent vintages…")
        r0b = pd.DataFrame(rung0b_tick_vintage_sweep(region, records, weather, versions[-6:]))
        print(_md_table(r0b))
        report.append(
            f"\n**Rung 0b — the Jul-18 tick frame across vintages**:\n\n{_md_table(r0b)}\n"
        )

        print("rung 3b — vintage metas × tail contamination…")
        r3b = rung3b_meta_correlation(region, records, versions)
        r3b = r3b.merge(
            r3[["version", "dives", "mape"]].rename(columns={"mape": "past_window_mape"}),
            on="version",
            how="left",
        )
        print(_md_table(r3b))
        report.append(f"\n**Rung 3b — metas × contamination × dive**:\n\n{_md_table(r3b)}\n")

        print("rung 4 — anchor arms…")
        r4 = pd.DataFrame(rung4_anchor_arms(region, model, records, weather, past_start))
        print(_md_table(r4))
        report.append(f"\n**Rung 4 — anchor arms**:\n\n{_md_table(r4)}\n")

        print("rung 5 — per-step forensics on the failing config…")
        dive_rows = r2["_rows_b"] if r2["built_frame"]["dives"] else r2["_rows_a"]
        full_featured = _featured(
            _history_frame(
                region,
                _settled_series(records),
                past_start + pd.Timedelta(hours=49),
            ),
            weather,
        )
        r5 = rung5_forensics(model, dive_rows, full_featured)
        for e in r5["per_step"]:
            print(f"  {e}")
        report.append("\n**Rung 5 — per-step forensics**:\n")
        report.extend(f"- `{e}`\n" for e in r5["per_step"])
        if r5["first_dive_perturbation"]:
            p = r5["first_dive_perturbation"]
            print(
                f"  first diving step {p['step']} ({p['ts']}): pred {p['base_pred']} vs actual {p['actual']}"
            )
            movers = pd.DataFrame(p["top_movers"])
            print(_md_table(movers))
            report.append(
                f"\nFirst diving step {p['step']} ({p['ts']}): pred {p['base_pred']} vs "
                f"actual {p['actual']} — per-feature settled-swap movers:\n\n"
            )
            report.append(_md_table(movers) + "\n")

    if args.output:
        args.output.write_text("".join(report))
        print(f"\nreport written → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
