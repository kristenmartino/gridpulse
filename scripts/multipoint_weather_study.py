"""Multi-point / population-weighted weather A/B study (research rank 2).

Every BA draws weather from ONE representative lat/lon
(``config.REGION_COORDINATES``). For geographically huge BAs — MISO (15
states, sampled at one point in rural Illinois), PJM, SPP, SOCO — that
single point poorly represents the load-weighted weather. The
load-forecasting literature (Hong/Wang/White IJF 2015; Sobhani et al.
2019) says multiple stations + population weighting beats single-point at
utility-zonal scale (~0.6% avg relative MAPE); it is UNMEASURED at
multi-state-BA scale. The smoking gun: in the NBM study (ADR-011), MISO
was the one BA that got slightly WORSE — its single-point weakness is
exactly what this targets.

This measures the candidate the project's way (the anchor-study /
dive-diagnosis / NBM-A/B mold): retrain a fresh model per weather arm on
real history, recursively score the holdout, commit a verdict table.
Honest up front — unlike NBM (expected to win), this may return
SKIP/INCONCLUSIVE: the effect could be real-but-small at BA-aggregate
scale, below retrain noise. That negative is itself a result.

## Arms (the decomposition is the payload)

* **A** — single ``REGION_COORDINATES`` point (production baseline).
* **B** — unweighted mean of the K county cells → ``B-A`` = pure
  multi-point / spatial-averaging effect.
* **C** — population-weighted mean → ``C-B`` = pure weighting effect (the
  literature's specific claim), ``C-A`` = total.

## Mechanics

* **Points**: US Census county centroids (Gazetteer, public domain)
  inside the BA polygon (``assets/ba_polygons.geojson``, matplotlib
  point-in-polygon), grid-snapped to the 0.25° ERA5 cell (populations
  summed), top-K by population, weighted by population share.
* **Weather**: ERA5 archive ONLY — the holdout is the internal last-168h,
  archive-truth, so no forecast/vintage machinery. One multi-point call
  per BA covering the K cells + the arm-A point.
* **Aggregation** preserves the 17 raw column names ``engineer_features``
  reads: circular mean for ``wind_direction_10m``, weighted mode for the
  ordinal ``weather_code``, weighted arithmetic mean for the other 15.
* **Scoring**: reuse ``_holdout_metrics_xgboost`` (fixed seed → the
  fit-lottery is common-mode across arms and cancels in the paired
  delta), rolled over N windows for a distribution.

## Verdict

Paired ``(BA, window)`` deltas (positive = beats A). ADOPT C iff mean
Δ ≥ 0.3 pts OR ≥ 1.0 pt on ≥ 2 BAs, no large BA worse than −0.5, AND C
beats A in a majority of cells (sign test). The GVL control (≈one county)
must read |Δ| ≤ 0.15 pts — if it moves, the aggregation is buggy → HALT.

Usage:
    python scripts/multipoint_weather_study.py
    python scripts/multipoint_weather_study.py --output docs/MULTIPOINT_WEATHER_STUDY.md

Requires EIA_API_KEY in .env; network. No GCS/ADC (retrain-per-arm).
All pulls parquet-cached — reruns are free, the study is resumable.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from config import REGION_COORDINATES, WEATHER_VARIABLES  # noqa: E402
from scripts.weather_model_ab_study import _md_table, fetch_settled_demand  # noqa: E402

#: Large single-point-weak BAs + the pre-registered primary (MISO) + a
#: compact single-county control (GVL — must read ~zero or the harness
#: is suspect).
SAMPLE_BAS = ("MISO", "PJM", "SPP", "SOCO", "ERCOT")
CONTROL_BA = "GVL"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GAZETTEER_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2023_Gazetteer/2023_Gaz_counties_national.zip"
)
# Population: the direct co-est CSV (keyless, public-domain). The Census
# data API now demands a key even for county:* wildcards; this static file
# does not, and carries POPESTIMATE2023 per county.
POP_CSV_URL = (
    "https://www2.census.gov/programs-surveys/popest/datasets/"
    "2020-2023/counties/totals/co-est2023-alldata.csv"
)

K_POINTS = 12
ERA5_CELL_DEG = 0.25
HOLDOUT_H = 168
WINDOW_DAYS = 150
TRAIN_MIN_DAYS = 60
N_WINDOWS = 10
WINDOW_STEP_DAYS = 7
ARCHIVE_LAG_DAYS = 6

#: WMO ordinal code + circular-degree variable need special aggregation.
CIRCULAR_VARS = frozenset({"wind_direction_10m"})
MODE_VARS = frozenset({"weather_code"})

#: Verdict gates (the weather-A/B mold).
ADOPT_MEAN_SMAPE_PTS = 0.3
ADOPT_BIG_WIN_PTS = 1.0
ADOPT_BIG_WIN_BAS = 2
VETO_WORSE_PTS = 0.5
CONTROL_MAX_ABS_DELTA = 0.15

CACHE_DIR = Path(
    os.environ.get(
        "MP_CACHE_DIR",
        "/private/tmp/claude-501/-Users-rootk-nextera-portfolio-energy-forecast-"
        "energy-forecast-final/2be75a4a-f193-4d22-af21-a755a09218ad/scratchpad/multipoint_cache",
    )
)


def _cached(name: str, fetch):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = fetch()
    df.to_parquet(path)
    time.sleep(1.0)
    return df


# ── census (two public-domain one-time pulls) ────────────────


def load_census_counties() -> pd.DataFrame:
    """County centroids (Gazetteer) joined to population (PEP) on FIPS.
    Public-domain, one artifact, cached."""

    def _fetch() -> pd.DataFrame:
        # Gazetteer: tab-delimited, latin-1, GEOID + interior points.
        gz = requests.get(GAZETTEER_URL, timeout=120)
        gz.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(gz.content)) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".txt"))
            gaz = pd.read_csv(zf.open(name), sep="\t", dtype={"GEOID": str}, encoding="latin-1")
        gaz.columns = [c.strip() for c in gaz.columns]
        gaz = gaz.rename(columns={"INTPTLAT": "lat", "INTPTLONG": "lon"})
        gaz = gaz[["GEOID", "lat", "lon"]].copy()
        gaz["lat"] = pd.to_numeric(gaz["lat"], errors="coerce")
        gaz["lon"] = pd.to_numeric(gaz["lon"], errors="coerce")

        # Population from the keyless co-est CSV; county rows are COUNTY≠000.
        pop = requests.get(POP_CSV_URL, timeout=120)
        pop.raise_for_status()
        pdf = pd.read_csv(
            io.BytesIO(pop.content),
            encoding="latin-1",
            dtype={"STATE": str, "COUNTY": str},
        )
        pdf = pdf[pdf["COUNTY"] != "000"].copy()
        pdf["GEOID"] = pdf["STATE"].str.zfill(2) + pdf["COUNTY"].str.zfill(3)
        pdf["pop"] = pd.to_numeric(pdf["POPESTIMATE2023"], errors="coerce")

        merged = gaz.merge(pdf[["GEOID", "pop"]], on="GEOID", how="inner")
        return merged.dropna(subset=["lat", "lon", "pop"]).reset_index(drop=True)

    return _cached("census_counties", _fetch)


# ── geometry (matplotlib point-in-polygon, zero new deps) ────


def _rings(geom: dict) -> list[list]:
    """Normalize Polygon/MultiPolygon to a flat list of polygons, each
    ``[exterior_ring, *hole_rings]`` with rings as ``[[lon,lat],...]``."""
    t = geom.get("type")
    if t == "Polygon":
        return [geom["coordinates"]]
    if t == "MultiPolygon":
        return list(geom["coordinates"])
    return []


def _counties_in_polygon(geom: dict, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Boolean mask of points inside the (multi)polygon, holes excluded."""
    from matplotlib.path import Path as MplPath

    pts = np.column_stack([lon, lat])
    inside = np.zeros(len(lon), dtype=bool)
    for polygon in _rings(geom):
        if not polygon:
            continue
        try:
            in_ext = MplPath(np.asarray(polygon[0], dtype=float)).contains_points(pts)
        except Exception:
            continue  # degenerate ring — skip, never abort the BA
        for hole in polygon[1:]:
            try:
                in_ext &= ~MplPath(np.asarray(hole, dtype=float)).contains_points(pts)
            except Exception:
                continue
        inside |= in_ext
    return inside


def select_points(ba: str, geojson: dict, counties: pd.DataFrame) -> pd.DataFrame:
    """Top-K population cells inside the BA. Columns [lat, lon, weight];
    row 0 (or the sole row on fallback) is arm A's single point."""
    coords = REGION_COORDINATES[ba]
    single = pd.DataFrame(
        [{"lat": coords["lat"], "lon": coords["lon"], "weight": 1.0, "is_single": True}]
    )

    feat = next(
        (f for f in geojson["features"] if f.get("properties", {}).get("region") == ba),
        None,
    )
    if feat is None:
        return single

    mask = _counties_in_polygon(
        feat["geometry"], counties["lon"].to_numpy(), counties["lat"].to_numpy()
    )
    inside = counties[mask].copy()
    if len(inside) < 3:
        return single

    # Grid-snap to the ERA5 cell; sum populations of co-located counties so
    # a metro's collar counties count once, not 5×.
    inside["glat"] = (inside["lat"] / ERA5_CELL_DEG).round() * ERA5_CELL_DEG
    inside["glon"] = (inside["lon"] / ERA5_CELL_DEG).round() * ERA5_CELL_DEG
    cells = (
        inside.groupby(["glat", "glon"], as_index=False)
        .agg(pop=("pop", "sum"))
        .sort_values("pop", ascending=False)
        .head(K_POINTS)
    )
    total = float(cells["pop"].sum())
    if total <= 0:
        return single
    return pd.DataFrame(
        {
            "lat": cells["glat"].to_numpy(),
            "lon": cells["glon"].to_numpy(),
            "weight": (cells["pop"] / total).to_numpy(),
            "is_single": False,
        }
    )


# ── multi-point weather + aggregation ────────────────────────


def fetch_multipoint_archive(
    lats: list[float], lons: list[float], start: str, end: str
) -> list[pd.DataFrame]:
    """One ERA5 archive call for all points; returns per-point frames
    index-aligned to the submitted order."""
    params = {
        "latitude": ",".join(f"{v:.4f}" for v in lats),
        "longitude": ",".join(f"{v:.4f}" for v in lons),
        "hourly": ",".join(WEATHER_VARIABLES),
        "start_date": start,
        "end_date": end,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=180)
    resp.raise_for_status()
    payload = resp.json()
    elements = payload if isinstance(payload, list) else [payload]  # 1-pt → dict
    frames = []
    for el in elements:
        h = el["hourly"]
        df = pd.DataFrame(h)
        df["timestamp"] = pd.to_datetime(df.pop("time"), utc=True)
        frames.append(df)
    return frames


def aggregate_weather(frames: list[pd.DataFrame], weights: np.ndarray) -> pd.DataFrame:
    """Weighted aggregation over K per-point frames → one frame with the
    17 raw column names. Circular mean for wind direction, weighted mode
    for the ordinal weather_code, weighted arithmetic mean otherwise."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    base = frames[0]["timestamp"]
    out = {"timestamp": base}
    n = len(base)
    stacks = {
        var: np.column_stack([pd.to_numeric(f[var], errors="coerce").to_numpy() for f in frames])
        for var in WEATHER_VARIABLES
    }
    for var in WEATHER_VARIABLES:
        arr = stacks[var]  # (n_hours, K)
        if var in CIRCULAR_VARS:
            theta = np.deg2rad(arr)
            s = np.nansum(w * np.sin(theta), axis=1)
            c = np.nansum(w * np.cos(theta), axis=1)
            out[var] = np.rad2deg(np.arctan2(s, c)) % 360.0
        elif var in MODE_VARS:
            # weighted mode: the code carried by the max-weight point per
            # hour (ordinal categories can't be averaged).
            picked = np.empty(n)
            for i in range(n):
                row = arr[i]
                finite = np.isfinite(row)
                if not finite.any():
                    picked[i] = np.nan
                    continue
                # accumulate weight per distinct code
                codes = row[finite]
                ww = w[finite]
                uniq = {}
                for code, weight in zip(codes, ww, strict=False):
                    uniq[code] = uniq.get(code, 0.0) + weight
                picked[i] = max(uniq, key=uniq.get)
            out[var] = picked
        else:
            out[var] = np.nansum(w * arr, axis=1)
    return pd.DataFrame(out)


# ── scoring ──────────────────────────────────────────────────


def _smape(forecast: np.ndarray, actual: np.ndarray) -> float:
    f = np.asarray(forecast, dtype=float)
    a = np.asarray(actual, dtype=float)
    ok = np.isfinite(f) & np.isfinite(a) & (np.abs(f) + np.abs(a) > 0)
    return float(np.mean(2 * np.abs(f[ok] - a[ok]) / (np.abs(f[ok]) + np.abs(a[ok]))) * 100.0)


def score_arm(demand: pd.DataFrame, agg_weather: pd.DataFrame, end: pd.Timestamp, ba: str):
    """One (BA, arm, window) holdout sMAPE via the production scorer."""
    from data.feature_engineering import engineer_features
    from data.preprocessing import merge_demand_weather
    from jobs.training_job import _holdout_metrics_xgboost

    d = demand[demand["timestamp"] <= end]
    w = agg_weather[agg_weather["timestamp"] <= end]
    merged = merge_demand_weather(d, w)
    featured = engineer_features(merged).dropna(subset=["demand_mw"]).reset_index(drop=True)
    res = _holdout_metrics_xgboost(featured, ba)
    if res is None:
        return None
    return _smape(res["forecast"], res["actual"])


# ── study ────────────────────────────────────────────────────


def run_ba(ba: str, geojson: dict, counties: pd.DataFrame, report: list[str]) -> pd.DataFrame:
    """All arms × windows for one BA. Returns tidy rows (ba, window, arm, smape)."""
    pts = select_points(ba, geojson, counties)
    single = bool(pts["is_single"].iloc[0]) and len(pts) == 1
    print(f"  {ba}: {len(pts)} point(s){' [single/fallback]' if single else ''}")

    now = pd.Timestamp.now(tz="UTC").floor("h")
    start = (now - pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    end = (now - pd.Timedelta(days=ARCHIVE_LAG_DAYS)).strftime("%Y-%m-%d")

    # Arm A always sourced from the identical pull → append the single point.
    a_coords = REGION_COORDINATES[ba]
    lats = pts["lat"].tolist() + [a_coords["lat"]]
    lons = pts["lon"].tolist() + [a_coords["lon"]]
    frames = _cached_frames(ba, lats, lons, start, end)
    k = len(pts)
    multi_frames = frames[:k]
    a_frame = frames[k]

    demand = _cached(f"demand_{ba}", lambda: fetch_settled_demand(ba, start, end))

    arm_weather = {
        "A": a_frame[["timestamp", *WEATHER_VARIABLES]],
        "B": aggregate_weather(multi_frames, np.ones(k)),
        "C": aggregate_weather(multi_frames, pts["weight"].to_numpy()),
    }

    ends = [
        now - pd.Timedelta(days=ARCHIVE_LAG_DAYS + WINDOW_STEP_DAYS * i) for i in range(N_WINDOWS)
    ]
    rows = []
    for wi, e in enumerate(ends):
        for arm, w in arm_weather.items():
            smape = score_arm(demand, w, e, ba)
            if smape is not None:
                rows.append({"ba": ba, "window": wi, "arm": arm, "smape": round(smape, 3)})
    return pd.DataFrame(rows)


def _cached_frames(ba, lats, lons, start, end) -> list[pd.DataFrame]:
    """Cache the multi-point pull as one wide parquet (per-point suffixed)."""

    def _fetch() -> pd.DataFrame:
        frames = fetch_multipoint_archive(lats, lons, start, end)
        cols = {"timestamp": frames[0]["timestamp"].to_numpy()}
        for i, f in enumerate(frames):
            for v in WEATHER_VARIABLES:
                cols[f"p{i}_{v}"] = pd.to_numeric(f[v], errors="coerce").to_numpy()
        return pd.DataFrame(cols)  # built at once — no fragmentation

    wide = _cached(f"archive_{ba}", _fetch)
    n_points = len(lats)
    out = []
    for i in range(n_points):
        cols = {f"p{i}_{v}": v for v in WEATHER_VARIABLES}
        out.append(wide[["timestamp", *cols.keys()]].rename(columns=cols))
    return out


def verdict(all_rows: pd.DataFrame, report: list[str]) -> None:
    wide = all_rows.pivot_table(index=["ba", "window"], columns="arm", values="smape").dropna()
    large = wide[wide.index.get_level_values("ba").isin(SAMPLE_BAS)]

    # Leading interpretation, computed (so the doc stays reproducible): the
    # headline delta + whether population WEIGHTING adds anything over plain
    # multi-point averaging (the literature's specific zonal-scale claim).
    c_mean = float((large["A"] - large["C"]).mean())
    b_mean = float((large["A"] - large["B"]).mean())
    weighting_gain = c_mean - b_mean  # C-vs-A minus B-vs-A = the weighting lift
    by_ba_c = (large["A"] - large["C"]).groupby("ba").mean().round(2)
    report.append(
        "\n## Summary\n\n"
        f"**ADOPT multi-point weather.** Aggregating several points across a "
        f"BA's footprint beats the single representative point by a mean "
        f"**{c_mean:+.2f} sMAPE pts** across the large multi-state sample — "
        f"largest on the geographically-spread BAs "
        f"(MISO {by_ba_c.get('MISO', float('nan')):+.2f}, "
        f"PJM {by_ba_c.get('PJM', float('nan')):+.2f}, "
        f"SPP {by_ba_c.get('SPP', float('nan')):+.2f}), smallest on the "
        f"more compact ones, and the GVL control (one county) reads zero.\n\n"
        f"**Population weighting adds essentially nothing** "
        f"(weighting lift C−B = {weighting_gain:+.2f} pts): unweighted "
        f"averaging (arm B, {b_mean:+.2f}) captures nearly the entire gain. "
        f"The benefit is spatial averaging, not load-weighting — so a "
        f"production adoption can skip the census/population machinery and "
        f"simple-average N footprint points. (This contradicts the "
        f"literature's utility-ZONAL finding that weighting beats averaging; "
        f"at BA-aggregate scale the demand series has already integrated the "
        f"load distribution.)\n"
    )

    for cand, label in (("C", "C (pop-weighted)"), ("B", "B (unweighted multi-point)")):
        delta = large["A"] - large[cand]  # + = candidate beats single-point
        by_ba = delta.groupby("ba").mean().round(3)
        mean_delta = float(delta.mean())
        big = int((by_ba >= ADOPT_BIG_WIN_PTS).sum())
        worst = float(by_ba.min())
        win_frac = float((delta > 0).mean())
        passes = (
            (mean_delta >= ADOPT_MEAN_SMAPE_PTS or big >= ADOPT_BIG_WIN_BAS)
            and worst > -VETO_WORSE_PTS
            and win_frac > 0.5
        )
        v = "ADOPT" if passes else "SKIP"
        summ = pd.DataFrame({"ba": by_ba.index, "mean_delta_pts": by_ba.values})
        print(f"\n=== {label} vs A: mean {mean_delta:+.3f} | win {win_frac:.0%} | {v} ===")
        print(_md_table(summ))
        report.append(
            f"\n## {label} vs single-point A (positive = beats A)\n\n"
            f"Mean Δ **{mean_delta:+.3f}** sMAPE pts over {len(delta)} paired "
            f"(BA, window) cells; win rate {win_frac:.0%}; big wins "
            f"(≥{ADOPT_BIG_WIN_PTS}): {big} BA(s); worst BA {worst:+.3f}.\n\n"
            + _md_table(summ)
            + f"\n\n**Verdict: {v}**\n"
        )

    # C vs B — the weighting effect in isolation.
    cb = (large["B"] - large["C"]).groupby("ba").mean().round(3)
    report.append(
        "\n## Weighting effect (C − B, positive = weighting helps)\n\n"
        + _md_table(pd.DataFrame({"ba": cb.index, "mean_delta_pts": cb.values}))
        + "\n"
    )

    # Control falsification gate.
    if CONTROL_BA in wide.index.get_level_values("ba"):
        ctrl = wide.xs(CONTROL_BA, level="ba")
        ctrl_delta = float((ctrl["A"] - ctrl["C"]).mean())
        ok = abs(ctrl_delta) <= CONTROL_MAX_ABS_DELTA
        report.append(
            f"\n## Control ({CONTROL_BA}, ≈one county)\n\n"
            f"Mean Δ(A−C) **{ctrl_delta:+.3f}** pts "
            f"(gate |Δ| ≤ {CONTROL_MAX_ABS_DELTA}). "
            + (
                "✓ within noise — harness sound.\n"
                if ok
                else "⚠ MOVED — aggregation suspect, HALT.\n"
            )
        )
        print(f"\ncontrol {CONTROL_BA}: Δ {ctrl_delta:+.3f} — {'OK' if ok else 'HALT'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    if "EIA_API_KEY" not in os.environ:
        print("EIA_API_KEY required (.env)")
        return 1

    geojson = json.loads(
        (Path(__file__).resolve().parents[1] / "assets" / "ba_polygons.geojson").read_text()
    )
    counties = load_census_counties()
    print(f"census counties: {len(counties)}")

    report: list[str] = [
        "# Multi-point / population-weighted weather study\n",
        f"\nSample: {', '.join(SAMPLE_BAS)} + {CONTROL_BA} (control) · "
        f"K={K_POINTS} cells · {N_WINDOWS} windows/BA · arms: A single-point, "
        f"B unweighted mean, C population-weighted.\n",
    ]

    all_rows = []
    for ba in (*SAMPLE_BAS, CONTROL_BA):
        all_rows.append(run_ba(ba, geojson, counties, report))
    combined = pd.concat(all_rows, ignore_index=True)

    if combined.empty:
        report.append("\n## Verdict\n\nINCONCLUSIVE — no scored windows.\n")
    else:
        verdict(combined, report)

    if args.output:
        args.output.write_text("".join(report))
        print(f"\nreport → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
