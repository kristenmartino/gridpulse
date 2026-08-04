"""Does zonal structure predict our BA-level error? (NYISO)

`ISO_REALTIME_FEEDS.md` killed the anchor case for ISO ingestion — EIA's lag
is 1.7h and a 2h stale anchor costs +0.014 WAPE pts. What survived was a
*different* hypothesis, stated there but untested:

> EIA-930 gives one number per BA; NYISO publishes eleven zones. Sub-BA
> structure is information the model has never had, and a BA's load is a sum
> of zones with different weather and different mixes.

This probes that before anything is built, which is the pattern that has been
working: the cooling pack was built first and measured second, and failed.

## Falsifiable predictions

If zonal structure carries information our BA-level model is missing:

1. **Weather diversity.** Our absolute residual is larger in hours when
   temperature is more spread out across the 11 zones — a single BA-level
   response cannot be right for all of them at once.
2. **Load mix.** Our signed residual tracks the zonal composition of load
   (e.g. the downstate share), because a shifting mix re-weights which zone's
   weather should dominate.
3. **Mix instability.** Hours where the mix departs furthest from its recent
   norm are hours we do worst.

Failing all three kills the hypothesis for ~7 BAs' worth of integration work
before any connector is written.

## Note on what this does NOT test

Whether an 11-model bottom-up forecast beats a top-down one. That is the
expensive experiment this probe exists to justify or avoid.

Usage:
    python -m scripts.nyiso_zonal_probe --months 4
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from scripts.arima_order_exog_study import ARCHIVE_LAG_DAYS, _archive_weather, _get_with_retry
from scripts.error_analysis import (
    MIN_TRAIN_H,
    _eia_with_forecast,
    _fit_predict_xgb,
    make_day_ahead_safe,
)

NYISO_PAL_ZIP = "http://mis.nyiso.com/public/csv/pal/{ym}01pal_csv.zip"

#: Approximate load centres for the 11 NYISO zones. Coarse by design — the
#: probe asks whether zonal weather DIVERSITY carries signal, which needs the
#: zones to be geographically distinct, not precisely centroided. If the probe
#: succeeds, real centroids become part of the follow-up work.
ZONE_COORDS: dict[str, tuple[float, float]] = {
    "CAPITL": (42.65, -73.75),  # Albany
    "CENTRL": (43.05, -76.15),  # Syracuse
    "DUNWOD": (40.94, -73.86),  # Yonkers
    "GENESE": (43.16, -77.61),  # Rochester
    "HUD VL": (41.70, -73.92),  # Poughkeepsie
    "LONGIL": (40.79, -73.14),  # Long Island
    "MHK VL": (43.10, -75.23),  # Utica
    "MILLWD": (41.20, -73.83),  # Millwood
    "N.Y.C.": (40.71, -74.01),  # New York City
    "NORTH": (44.70, -73.45),  # Plattsburgh
    "WEST": (42.89, -78.88),  # Buffalo
}

#: NYISO's 11 zones aggregated into 5 by geographic contiguity, to mirror
#: CAISO's large contiguous utility territories. Fixed in
#: docs/SUPERZONE_PREREGISTRATION.md before the run; not to be re-grouped to
#: fit a result.
SUPER_ZONES: dict[str, tuple[str, ...]] = {
    "WEST": ("WEST", "GENESE"),
    "CENTRAL": ("CENTRL", "MHK VL"),
    "NORTH_CAPITAL": ("NORTH", "CAPITL"),
    "LOWER_HUDSON": ("HUD VL", "MILLWD", "DUNWOD"),
    "METRO": ("N.Y.C.", "LONGIL"),
}

#: Super-zone weather point = mean of its members' coordinates, so a
#: super-zone gets exactly one weather input, as CAISO's TAC areas do.
SUPER_ZONE_COORDS: dict[str, tuple[float, float]] = {
    name: (
        sum(ZONE_COORDS[z][0] for z in members) / len(members),
        sum(ZONE_COORDS[z][1] for z in members) / len(members),
    )
    for name, members in SUPER_ZONES.items()
}


def fetch_superzone_load(months: int, end: datetime | None = None) -> pd.DataFrame:
    """The same NYISO load, summed into the 5 pre-registered super-zones.

    Identical source and hours as `fetch_zonal_load` — only the column
    grouping differs, so a comparison between them isolates zone count rather
    than data availability.
    """
    zl = fetch_zonal_load(months, end)
    if zl.empty:
        return zl
    out = pd.DataFrame(index=zl.index)
    for name, members in SUPER_ZONES.items():
        present = [z for z in members if z in zl.columns]
        if len(present) != len(members):
            # Partial membership would silently change what the super-zone
            # means; better to drop it than to sum a different thing.
            continue
        out[name] = zl[list(present)].sum(axis=1)
    return out.dropna(how="any")


#: Zones treated as "downstate" for the load-mix test — the metropolitan load
#: pocket whose weather differs most from the upstate zones.
DOWNSTATE = ("N.Y.C.", "LONGIL", "DUNWOD", "MILLWD", "HUD VL")

HOLDOUT_H = 168


def fetch_zonal_load(months: int, end: datetime | None = None) -> pd.DataFrame:
    """Hourly zonal load from NYISO's monthly archives (5-min data, averaged).

    ``end`` targets a window other than "now minus the archive lag" — needed
    for the winter run, which asks whether the bottom-up effect is a summer
    artifact.
    """
    end = end or (datetime.now(UTC) - timedelta(days=ARCHIVE_LAG_DAYS))
    frames: list[pd.DataFrame] = []
    ym = end.replace(day=1)
    for _ in range(months):
        url = NYISO_PAL_ZIP.format(ym=ym.strftime("%Y%m"))
        try:
            r = _get_with_retry(url, params={}, timeout=120)
        except Exception as e:
            print(f"    {ym:%Y-%m}: unavailable ({str(e)[:60]})", flush=True)
            ym = (ym - timedelta(days=1)).replace(day=1)
            continue
        z = zipfile.ZipFile(io.BytesIO(r.content))
        for name in z.namelist():
            if name.endswith(".csv"):
                frames.append(pd.read_csv(z.open(name)))
        print(f"    {ym:%Y-%m}: {len(z.namelist())} daily files", flush=True)
        ym = (ym - timedelta(days=1)).replace(day=1)
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["Time Stamp"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["ts", "Load"])
    # NYISO stamps local time with a DST column; localize to Eastern and convert.
    df["timestamp"] = (
        df["ts"]
        .dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
        .dt.tz_convert("UTC")
    )
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.floor("h")
    wide = df.pivot_table(index="timestamp", columns="Name", values="Load", aggfunc="mean")
    return wide.dropna(how="any").sort_index()


def zonal_weather(months: int) -> pd.DataFrame:
    """Temperature at each zone centre, one archive call per zone."""
    end = datetime.now(UTC) - timedelta(days=ARCHIVE_LAG_DAYS)
    start = end - timedelta(days=months * 31)
    cols = {}
    for zone, (lat, lon) in ZONE_COORDS.items():
        w = _archive_weather(lat, lon, start, end)
        cols[zone] = w.set_index("timestamp")["temperature_2m"]
        print(f"    weather {zone}: {len(w)} hours", flush=True)
    return pd.DataFrame(cols).dropna(how="any")


def probe(months: int, windows: int) -> dict[str, Any]:
    import os

    from config import REGION_COORDINATES
    from data.feature_engineering import engineer_features
    from models.rolling_eval import rolling_origin_splits

    api_key = os.environ["EIA_API_KEY"]
    print("  fetching NYISO zonal load ...", flush=True)
    zl = fetch_zonal_load(months)
    if zl.empty:
        return {"skipped": "no zonal load"}
    print("  fetching zonal weather ...", flush=True)
    zw = zonal_weather(months)

    # --- our BA-level forecast, exactly as ERROR_ANALYSIS builds it ---------
    end = datetime.now(UTC) - timedelta(days=ARCHIVE_LAG_DAYS)
    start = end - timedelta(days=months * 31)
    coords = REGION_COORDINATES["NYISO"]
    demand = _eia_with_forecast("NYISO", start, end, api_key)
    weather = _archive_weather(coords["lat"], coords["lon"], start, end)
    raw = demand.merge(weather, on="timestamp", how="inner").sort_values("timestamp")
    feats = (
        make_day_ahead_safe(engineer_features(raw))
        .dropna(subset=["demand_mw"])
        .reset_index(drop=True)
    )

    splits = rolling_origin_splits(
        len(feats), n_windows=windows, holdout_h=HOLDOUT_H, min_train_h=MIN_TRAIN_H
    )
    if not splits:
        return {"skipped": f"no windows from {len(feats)} rows"}

    rows: list[pd.DataFrame] = []
    for train_sl, test_sl in splits:
        pred = _fit_predict_xgb(feats.iloc[train_sl], feats.iloc[test_sl])
        if pred is None:
            continue
        test = feats.iloc[test_sl]
        actual = test["demand_mw"].to_numpy(dtype=float)
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(test["timestamp"]).to_numpy(),
                    "resid_pct": (pred - actual) / np.where(actual == 0, np.nan, actual) * 100,
                }
            )
        )
    if not rows:
        return {"skipped": "no window produced a forecast"}
    res = pd.concat(rows, ignore_index=True).set_index("timestamp")

    # --- join zonal structure onto our residuals ---------------------------
    zones = [z for z in ZONE_COORDS if z in zl.columns]
    j = res.join(zl[zones], how="inner").join(
        zw[[z for z in zones if z in zw.columns]].add_suffix("_t"), how="inner"
    )
    j = j.dropna()
    if len(j) < 200:
        return {"skipped": f"only {len(j)} joined hours"}

    temps = j[[f"{z}_t" for z in zones if f"{z}_t" in j.columns]]
    loads = j[zones]
    total = loads.sum(axis=1)

    j["temp_spread"] = temps.max(axis=1) - temps.min(axis=1)
    j["downstate_share"] = loads[[z for z in DOWNSTATE if z in loads.columns]].sum(axis=1) / total
    j["mix_departure"] = (
        j["downstate_share"] - j["downstate_share"].rolling(168, min_periods=24).mean()
    ).abs()
    j["abs_resid"] = j["resid_pct"].abs()
    j = j.dropna()

    def corr(a: str, b: str) -> float:
        return float(np.corrcoef(j[a], j[b])[0, 1])

    def by_quintile(col: str, target: str) -> dict[str, float]:
        q = pd.qcut(j[col].rank(method="first"), 5, labels=list("12345"))
        return {
            str(k): round(float(v), 3)
            for k, v in j.groupby(q, observed=True)[target].mean().items()
        }

    # --- the control that decides prediction 1 -----------------------------
    # Temperature spread rises with fronts and with season, and error rises in
    # extreme temperatures anyway. If the diversity effect is real it must
    # survive INSIDE a temperature band — otherwise it is temperature wearing
    # a diversity costume, which is exactly how the BTM probe's prediction 2
    # turned out to be non-diagnostic.
    j["temp_level"] = temps.mean(axis=1)
    j["temp_band"] = pd.qcut(j["temp_level"].rank(method="first"), 3, labels=["cool", "mid", "hot"])
    j["spread_half"] = pd.qcut(
        j["temp_spread"].rank(method="first"), 2, labels=["low_spread", "high_spread"]
    )
    within = (
        j.groupby(["temp_band", "spread_half"], observed=True)["abs_resid"]
        .mean()
        .round(3)
        .unstack()
    )
    within_effect = {
        str(band): round(float(row["high_spread"] - row["low_spread"]), 3)
        for band, row in within.iterrows()
        if "high_spread" in row and "low_spread" in row
    }
    q5 = j[j["temp_spread"] >= j["temp_spread"].quantile(0.8)]
    return {
        "n_hours": int(len(j)),
        "n_zones": len(zones),
        "corr_tempspread_vs_templevel": round(corr("temp_spread", "temp_level"), 4),
        "within_temperature_band_spread_effect_pts": within_effect,
        "top_spread_quintile": {
            "n_hours": int(len(q5)),
            "share_of_total_abs_error_pct": round(
                float(q5["abs_resid"].sum() / j["abs_resid"].sum() * 100), 1
            ),
            "share_of_hours_pct": round(float(len(q5) / len(j) * 100), 1),
        },
        "mean_abs_resid_pct": round(float(j["abs_resid"].mean()), 3),
        "prediction_1_weather_diversity": {
            "corr_tempspread_vs_absresid": round(corr("temp_spread", "abs_resid"), 4),
            "abs_resid_by_tempspread_quintile": by_quintile("temp_spread", "abs_resid"),
            "temp_spread_range_f": [
                round(float(j["temp_spread"].min()), 1),
                round(float(j["temp_spread"].max()), 1),
            ],
        },
        "prediction_2_load_mix": {
            "corr_downstateshare_vs_signedresid": round(corr("downstate_share", "resid_pct"), 4),
            "signed_resid_by_downstate_quintile": by_quintile("downstate_share", "resid_pct"),
        },
        "prediction_3_mix_instability": {
            "corr_mixdeparture_vs_absresid": round(corr("mix_departure", "abs_resid"), 4),
            "abs_resid_by_mixdeparture_quintile": by_quintile("mix_departure", "abs_resid"),
        },
    }


def main() -> int:
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=4)
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--out", default="docs/NYISO_ZONAL_PROBE.json")
    args = ap.parse_args()

    if not os.environ.get("EIA_API_KEY"):
        print("EIA_API_KEY not set")
        return 2

    res = probe(args.months, args.windows)
    print("\n== RESULT ==")
    print(json.dumps(res, indent=2))
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
