"""Weather-model A/B study — NBM/HRRR arms vs the serving endpoint (#332).

The data-source research ranked "higher-resolution NOAA weather via the
existing vendor" the top accuracy candidate, with one honest gap: nothing
quantifies the demand-MAPE gain. This study closes the gap the project's
way (the anchor-study / dive-diagnosis mold): replay the REAL serve path
with each weather arm and let a committed verdict table decide the switch.

## Arms

* **A — control**: Open-Meteo ``best_match`` (what production serves).
* **B — seamless**: ``ncep_gfs_seamless`` (GFS + HRRR blend, full vars).
* **C — NBM composite**: ``ncep_nbm_conus`` where NBM has the variable,
  arm-A values filling the gaps (measured in rung 0 — the API serves
  NBM 80m wind despite doc claims, but radiation is all-null).

History (seed + engineered lags, ERA5 archive weather) is IDENTICAL
across arms — the study isolates FUTURE-hour weather quality, the only
thing an endpoint switch changes.

## Honest-vintage mechanics

Future weather per arm comes from Open-Meteo's **Previous Runs API**
(``{var}_previous_dayN``: the forecast issued 24·N h before valid time).
For an anchor at hour H and target T, the study uses lead
``L = ceil((T-H)/24h)`` so the issue time ``T - 24L <= H`` — never a
forecast production could not have had. Truth is the ERA5 archive;
anchors end 13 days ago so every scored hour clears the 5-day archive
lag.

## Rungs

0. Variable audit per arm (coverage + unit sanity) — authoritative over
   docs; derives arm C's fill list empirically.
1. Weather-vs-truth at leads 1/3/5/7 for temperature, apparent
   temperature, radiation. STOP-gate: if B and C are within noise of A
   everywhere, the demand replay cannot find what the weather doesn't
   contain — verdict SKIP.
2. Demand replay: per (BA, anchor, arm) — settled EIA history, the
   CURRENT prod pickle, the real ``_build_future_feature_frame`` +
   recursive serve chain, scored 1-168h vs settled. Paired deltas
   (same pickle, only weather differs) in 1-24h / 25-72h / 73-168h
   buckets. This measures the exact deployment reality: swap first,
   pickles retrain into the new source via the daily cadence.
3. Verdict per arm: ADOPT iff mean paired improvement >= 0.3 sMAPE pts
   across the sample OR >= 1.0 pt on >= 2 BAs, AND no BA worsens by
   > 0.5 pts. PNM (clean control) must sit near zero delta or the
   harness is suspect.

Usage:
    python scripts/weather_model_ab_study.py                 # all rungs
    python scripts/weather_model_ab_study.py --output docs/WEATHER_MODEL_AB.md

Requires: EIA_API_KEY in .env; GCS_ENABLED=true + ADC for prod pickles.
All API pulls cache to the scratch dir (parquet) — reruns are free.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import config  # noqa: E402
from config import REGION_COORDINATES, WEATHER_VARIABLES  # noqa: E402

#: Error-mode-spanning sample: large single-point (ERCOT/MISO/PJM), BTM
#: solar (CAISO), desert tail (AZPS), hydro north (BPAT), small muni
#: tail (SEC), clean control (PNM).
SAMPLE_BAS = ("ERCOT", "MISO", "PJM", "CAISO", "AZPS", "BPAT", "SEC", "PNM")

ARMS: dict[str, str] = {
    "A_best_match": "best_match",
    "B_gfs_seamless": "ncep_gfs_seamless",
    "C_nbm": "ncep_nbm_conus",
}

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
EIA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

LEADS = (1, 2, 3, 4, 5, 6, 7)
TIER1_LEADS = (1, 3, 5, 7)
TIER1_VARS = ("temperature_2m", "apparent_temperature", "shortwave_radiation")

HORIZON_H = 168
ANCHOR_STEP_DAYS = 3
WINDOW_START_DAYS_AGO = 45
#: Anchors end 13 days back: +168h of truth then still clears the ERA5
#: archive's ~5-day lag.
WINDOW_END_DAYS_AGO = 13

#: Verdict gates (see module docstring).
ADOPT_MEAN_SMAPE_PTS = 0.3
ADOPT_BIG_WIN_PTS = 1.0
ADOPT_BIG_WIN_BAS = 2
VETO_WORSE_PTS = 0.5
#: Tier-1 stop-gate: an arm is "within noise" when its RMSE improves on
#: arm A by less than this relative fraction for every (var, lead, BA).
TIER1_NOISE_REL = 0.02

CACHE_DIR = Path(
    os.environ.get(
        "AB_CACHE_DIR",
        "/private/tmp/claude-501/-Users-rootk-nextera-portfolio-energy-forecast-"
        "energy-forecast-final/2be75a4a-f193-4d22-af21-a755a09218ad/scratchpad/weather_ab_cache",
    )
)


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def _cached(name: str, fetch) -> pd.DataFrame:
    """Parquet-cache every pull — reruns are free, the study is resumable."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = fetch()
    df.to_parquet(path)
    time.sleep(1.0)  # stay polite far under the free-tier rate limits
    return df


# ── data pulls ───────────────────────────────────────────────


def fetch_previous_runs(region: str, model: str, start: str, end: str) -> pd.DataFrame:
    """All 17 vars × leads 1..7 from the Previous Runs API, one frame."""
    coords = REGION_COORDINATES[region]
    hourly = [f"{v}_previous_day{lead}" for v in WEATHER_VARIABLES for lead in LEADS]
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": ",".join(hourly),
        "start_date": start,
        "end_date": end,
        "models": model,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
    }
    resp = requests.get(PREVIOUS_RUNS_URL, params=params, timeout=120)
    resp.raise_for_status()
    h = resp.json()["hourly"]
    df = pd.DataFrame(h)
    df["timestamp"] = pd.to_datetime(df.pop("time"), utc=True)
    return df


def fetch_archive_truth(region: str, start: str, end: str) -> pd.DataFrame:
    """ERA5 archive — both the weather truth and the history-side frame."""
    coords = REGION_COORDINATES[region]
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": ",".join(WEATHER_VARIABLES),
        "start_date": start,
        "end_date": end,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=120)
    resp.raise_for_status()
    h = resp.json()["hourly"]
    df = pd.DataFrame(h)
    df["timestamp"] = pd.to_datetime(df.pop("time"), utc=True)
    return df


def fetch_settled_demand(region: str, start: str, end: str) -> pd.DataFrame:
    """Settled EIA demand + day-ahead (schema parity with the prod frame)."""
    from data.eia_client import _get_eia_code

    rows: list[dict] = []
    for typ in ("D", "DF"):
        params = {
            "api_key": os.environ["EIA_API_KEY"],
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": _get_eia_code(region),
            "facets[type][]": typ,
            "start": f"{start}T00",
            "end": f"{end}T23",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": 5000,
        }
        resp = requests.get(EIA_URL, params=params, timeout=120)
        resp.raise_for_status()
        for r in resp.json()["response"]["data"]:
            rows.append({"period": r["period"], "type": typ, "value": r["value"]})
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["period"], utc=True, format="%Y-%m-%dT%H")
    wide = df.pivot_table(index="timestamp", columns="type", values="value", aggfunc="first")
    out = pd.DataFrame(
        {
            "timestamp": wide.index,
            "demand_mw": pd.to_numeric(wide.get("D"), errors="coerce"),
            "forecast_mw": pd.to_numeric(wide.get("DF"), errors="coerce"),
        }
    ).reset_index(drop=True)
    out["region"] = region
    return out


# ── arm frames ───────────────────────────────────────────────


def asof_future_frame(
    prev_runs: pd.DataFrame,
    fill_from: pd.DataFrame | None,
    anchor: pd.Timestamp,
    horizon: int = HORIZON_H,
) -> pd.DataFrame:
    """The lead-faithful future weather frame for one anchor.

    For target hour T, lead ``L = ceil((T - anchor)/24h)`` — issue time
    ``T - 24L <= anchor``, never a forecast production couldn't have had.
    ``fill_from`` (arm A's frame) patches variables the arm never carries
    (arm C's radiation) — per-value, only where the arm is null.
    """
    pr = prev_runs.set_index("timestamp")
    fill = fill_from.set_index("timestamp") if fill_from is not None else None
    rows = []
    for i in range(1, horizon + 1):
        ts = anchor + pd.Timedelta(hours=i)
        lead = min(7, max(1, math.ceil(i / 24)))
        if ts not in pr.index:
            continue
        row: dict = {"timestamp": ts}
        for v in WEATHER_VARIABLES:
            val = pr.at[ts, f"{v}_previous_day{lead}"]
            if pd.isna(val) and fill is not None and ts in fill.index:
                val = fill.at[ts, f"{v}_previous_day{lead}"]
            row[v] = val
        rows.append(row)
    return pd.DataFrame(rows)


# ── rung 0: variable audit ───────────────────────────────────


def rung0_audit(report: list[str]) -> dict[str, list[str]]:
    """Empirical per-arm coverage (authoritative over docs). Returns the
    all-null variable list per arm — arm C's fill set."""
    audit_bas = SAMPLE_BAS[:3]
    start = (pd.Timestamp.utcnow() - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    end = (pd.Timestamp.utcnow() - pd.Timedelta(days=18)).strftime("%Y-%m-%d")
    gaps: dict[str, list[str]] = {}
    rows = []
    for arm, model in ARMS.items():
        missing_all: set[str] | None = None
        for ba in audit_bas:
            df = _cached(
                f"audit_{arm}_{ba}", lambda b=ba, m=model: fetch_previous_runs(b, m, start, end)
            )
            missing = {v for v in WEATHER_VARIABLES if df[f"{v}_previous_day1"].isna().all()}
            missing_all = missing if missing_all is None else (missing_all & missing)
            # unit sanity on the variables that exist
            t = df["temperature_2m_previous_day1"].dropna()
            assert t.empty or (t.min() > -40 and t.max() < 135), f"{arm}/{ba}: temp not in °F"
        gaps[arm] = sorted(missing_all or set())
        rows.append(
            {
                "arm": arm,
                "model": model,
                "vars_missing_everywhere": ", ".join(gaps[arm]) or "(none)",
                "n_missing": len(gaps[arm]),
            }
        )
    table = pd.DataFrame(rows)
    print(_md_table(table))
    report.append("\n## Rung 0 — variable audit (empirical, 3 BAs)\n\n" + _md_table(table) + "\n")

    # Variables the Previous Runs PRODUCT lacks in every arm (measured:
    # soil_temperature_0cm) are an API-wide gap, not a model gap — they
    # fall back to the builder's (hour,dow) climatology IDENTICALLY in
    # every arm, so the comparison stays fair. Only arm-specific gaps
    # (C's radiation etc.) get arm-A fill.
    api_wide = set(gaps["A_best_match"]) & set(gaps["B_gfs_seamless"]) & set(gaps["C_nbm"])
    assert set(gaps["A_best_match"]) <= api_wide, "control has arm-specific gaps"
    assert set(gaps["B_gfs_seamless"]) <= api_wide, "seamless has arm-specific gaps"
    gaps = {arm: sorted(set(g) - api_wide) for arm, g in gaps.items()}
    note = (
        f"\nAPI-wide gaps (all arms → identical climatology fallback, fair): "
        f"`{sorted(api_wide)}`; arm-specific fill sets: "
        + ", ".join(f"{a}: `{g or '(none)'}`" for a, g in gaps.items())
        + "\n"
    )
    print(note.strip())
    report.append(note)
    return gaps


# ── tier 1: weather vs truth at lead ─────────────────────────


def tier1_weather_accuracy(start: str, end: str, report: list[str]) -> tuple[pd.DataFrame, bool]:
    """RMSE per (arm, var, lead) vs ERA5 truth, pooled across sample BAs.
    Returns (table, any_arm_beats_noise)."""
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    truths: dict[str, pd.DataFrame] = {}
    for ba in SAMPLE_BAS:
        truths[ba] = _cached(f"truth_{ba}", lambda b=ba: fetch_archive_truth(b, start, end))
        for arm, model in ARMS.items():
            frames[(arm, ba)] = _cached(
                f"prev_{arm}_{ba}", lambda b=ba, m=model: fetch_previous_runs(b, m, start, end)
            )

    rows = []
    for arm in ARMS:
        for v in TIER1_VARS:
            for lead in TIER1_LEADS:
                errs: list[np.ndarray] = []
                for ba in SAMPLE_BAS:
                    t = truths[ba].set_index("timestamp")[v]
                    p = frames[(arm, ba)].set_index("timestamp")[f"{v}_previous_day{lead}"]
                    joined = pd.concat([t.rename("t"), p.rename("p")], axis=1).dropna()
                    if len(joined):
                        errs.append((joined["p"] - joined["t"]).to_numpy())
                if not errs:
                    continue
                all_err = np.concatenate(errs)
                rows.append(
                    {
                        "arm": arm,
                        "var": v,
                        "lead_d": lead,
                        "rmse": round(float(np.sqrt(np.mean(all_err**2))), 3),
                        "bias": round(float(np.mean(all_err)), 3),
                        "n": len(all_err),
                    }
                )
    table = pd.DataFrame(rows)
    print(_md_table(table))
    report.append("\n## Tier 1 — weather RMSE vs ERA5 truth (pooled)\n\n" + _md_table(table) + "\n")

    # STOP-gate: does any candidate beat the control beyond noise anywhere?
    base = table[table["arm"] == "A_best_match"].set_index(["var", "lead_d"])["rmse"]
    beats = False
    for arm in ("B_gfs_seamless", "C_nbm"):
        sub = table[table["arm"] == arm].set_index(["var", "lead_d"])["rmse"]
        for key in sub.index:
            if (
                key in base.index
                and base[key] > 0
                and (base[key] - sub[key]) / base[key] > TIER1_NOISE_REL
            ):
                beats = True
    verdict = "proceed to tier 2" if beats else "STOP — no arm beats control beyond noise"
    print(f"tier-1 gate: {verdict}")
    report.append(f"\n**Tier-1 gate:** {verdict}\n")
    return table, beats


# ── tier 2: demand replay ────────────────────────────────────


def tier2_demand_replay(
    anchors: list[pd.Timestamp], start: str, end: str, gaps: dict[str, list[str]], report: list[str]
) -> pd.DataFrame:
    from data.feature_engineering import engineer_features
    from data.preprocessing import merge_demand_weather
    from jobs.phases import (
        _build_future_feature_frame,
        _predict_xgboost_with_recursive_autoregressive,
    )
    from scripts.forecast_dive_diagnosis import _list_versions, _load_pickle_version

    rows = []
    for ba in SAMPLE_BAS:
        demand = _cached(f"demand_{ba}", lambda b=ba: fetch_settled_demand(b, start, end))
        truth_weather = _cached(f"truth_{ba}", lambda b=ba: fetch_archive_truth(b, start, end))
        versions = _list_versions(ba, "xgboost")
        if not versions:
            print(f"  {ba}: no prod pickle — skipped")
            continue
        model = _load_pickle_version(ba, "xgboost", versions[-1])
        prev = {
            arm: _cached(
                f"prev_{arm}_{ba}",
                lambda b=ba, m=model_name: fetch_previous_runs(b, m, start, end),
            )
            for arm, model_name in ARMS.items()
        }
        settled = demand.set_index("timestamp")["demand_mw"]

        for anchor in anchors:
            hist = demand[demand["timestamp"] <= anchor]
            if len(hist) < 720:
                continue
            featured = (
                engineer_features(merge_demand_weather(hist, truth_weather))
                .dropna(subset=["demand_mw"])
                .reset_index(drop=True)
            )
            if len(featured) < 400:
                continue
            truth_ts = [anchor + pd.Timedelta(hours=i) for i in range(1, HORIZON_H + 1)]
            truth = np.asarray([settled.get(t, np.nan) for t in truth_ts], dtype=float)

            for arm in ARMS:
                fill = prev["A_best_match"] if gaps.get(arm) else None
                future_weather = asof_future_frame(prev[arm], fill, anchor)
                if len(future_weather) < HORIZON_H * 0.9:
                    continue
                future = _build_future_feature_frame(
                    featured,
                    HORIZON_H,
                    weather_df=future_weather,
                    start_ts=anchor + pd.Timedelta(hours=1),
                )
                preds = np.asarray(
                    _predict_xgboost_with_recursive_autoregressive(
                        model, featured, future, HORIZON_H
                    ),
                    dtype=float,
                )
                for lo, hi, bucket in ((0, 24, "1-24h"), (24, 72, "25-72h"), (72, 168, "73-168h")):
                    t = truth[lo:hi]
                    p = preds[lo:hi]
                    ok = np.isfinite(t) & (t > 0) & np.isfinite(p)
                    if ok.sum() < (hi - lo) * 0.8:
                        continue
                    smape = float(
                        np.mean(2 * np.abs(p[ok] - t[ok]) / (np.abs(p[ok]) + np.abs(t[ok]))) * 100.0
                    )
                    rows.append(
                        {
                            "ba": ba,
                            "anchor": str(anchor)[:10],
                            "arm": arm,
                            "bucket": bucket,
                            "smape": round(smape, 3),
                        }
                    )
        print(f"  {ba}: replayed")
    return pd.DataFrame(rows)


def verdicts_from_replay(replay: pd.DataFrame, report: list[str]) -> None:
    wide = replay.pivot_table(
        index=["ba", "anchor", "bucket"], columns="arm", values="smape"
    ).dropna()
    for arm in ("B_gfs_seamless", "C_nbm"):
        if arm not in wide.columns:
            continue
        delta = wide["A_best_match"] - wide[arm]  # + = candidate better
        by_ba = delta.groupby("ba").mean().round(3)
        by_bucket = delta.groupby("bucket").mean().round(3)
        mean_delta = float(delta.mean())
        big_wins = int((by_ba >= ADOPT_BIG_WIN_PTS).sum())
        worst = float(by_ba.min())
        passes = (
            mean_delta >= ADOPT_MEAN_SMAPE_PTS or big_wins >= ADOPT_BIG_WIN_BAS
        ) and worst > -VETO_WORSE_PTS
        verdict = "ADOPT" if passes else "SKIP"
        summary = pd.DataFrame({"ba": by_ba.index, "mean_delta_pts": by_ba.values})
        print(f"\n=== {arm}: mean Δ {mean_delta:+.3f} pts | verdict {verdict} ===")
        print(_md_table(summary))
        print("by bucket:", by_bucket.to_dict())
        report.append(
            f"\n## Tier 2 — {arm} paired deltas (positive = beats control)\n\n"
            f"Mean Δ **{mean_delta:+.3f}** sMAPE pts over {len(delta)} paired "
            f"(BA, anchor, bucket) cells; big wins (≥{ADOPT_BIG_WIN_PTS} pt): "
            f"{big_wins} BA(s); worst BA {worst:+.3f}.\n\n"
            + _md_table(summary)
            + f"\n\nBy bucket: `{by_bucket.to_dict()}`\n\n**Verdict: {verdict}**\n"
        )
        pnm = by_ba.get("PNM")
        if pnm is not None and abs(pnm) > 1.0:
            report.append(
                f"\n⚠ control-sanity flag: PNM delta {pnm:+.3f} exceeds ±1.0 — "
                "inspect before trusting the verdict.\n"
            )


# ── main ─────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--skip-tier2", action="store_true")
    args = ap.parse_args()

    if "EIA_API_KEY" not in os.environ:
        print("EIA_API_KEY required (.env)")
        return 1
    if not config.GCS_ENABLED or not config.GCS_BUCKET_NAME:
        print("GCS access required for prod pickles (GCS_ENABLED, GCS_BUCKET_NAME, ADC).")
        return 1

    now = pd.Timestamp.utcnow().floor("h").tz_localize(None)
    start = (now - pd.Timedelta(days=WINDOW_START_DAYS_AGO + 30)).strftime("%Y-%m-%d")
    end = (now - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    anchor_lo = now - pd.Timedelta(days=WINDOW_START_DAYS_AGO)
    anchor_hi = now - pd.Timedelta(days=WINDOW_END_DAYS_AGO)
    anchors = list(
        pd.date_range(
            anchor_lo.normalize() + pd.Timedelta(hours=6),
            anchor_hi,
            freq=f"{ANCHOR_STEP_DAYS}D",
            tz="UTC",
        )
    )
    print(
        f"window {start}..{end} | {len(anchors)} anchors × {len(SAMPLE_BAS)} BAs × {len(ARMS)} arms"
    )

    report: list[str] = ["# Weather-model A/B study\n"]
    report.append(
        f"\nSample: {', '.join(SAMPLE_BAS)} · window {start}..{end} · "
        f"{len(anchors)} anchors/BA · arms: "
        + ", ".join(f"{a} ({m})" for a, m in ARMS.items())
        + "\n"
    )

    print("\n== rung 0: variable audit ==")
    gaps = rung0_audit(report)

    print("\n== tier 1: weather vs truth ==")
    prev_start = (anchor_lo - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    _table, beats = tier1_weather_accuracy(prev_start, end, report)

    if not beats:
        print("\nVERDICT: SKIP (tier-1 stop-gate)")
        report.append(
            "\n## Verdict\n\nSKIP — no candidate arm beat the control beyond noise in tier 1.\n"
        )
    elif args.skip_tier2:
        print("tier 2 skipped by flag")
    else:
        print("\n== tier 2: demand replay ==")
        replay = tier2_demand_replay(anchors, start, end, gaps, report)
        if replay.empty:
            report.append("\n## Verdict\n\nINCONCLUSIVE — no replay rows survived.\n")
        else:
            verdicts_from_replay(replay, report)

    if args.output:
        args.output.write_text("".join(report))
        print(f"\nreport written → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
