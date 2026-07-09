"""#283 Phase 0 go/no-go (weather-error proxy).

Demand is ~monotonic in weather (CDD/HDD), so the days-17-30 demand-forecast
error is dominated by how wrong the tail WEATHER estimate is. This measures,
against what ACTUALLY happened, whether a (day_of_year, hour) weather-normal
beats the current recent-28d (hour, dow) climatology at estimating days-17-30
weather — segmented by whether the tail straddles a seasonal turn.

If the normal is NOT a better tail-weather estimate, the whole approach is a
wash → stop (recent-28d workaround is good enough). If it is clearly better
(esp. at seasonal turns), proceed to the full demand backtest.
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from data.feature_engineering import compute_cdd, compute_hdd  # noqa: E402
from data.weather_client import _fetch_archive_endpoint  # noqa: E402

OM_H, TAIL_END_H = 384, 720  # days 17-30 = hours 384..720
NORMAL_YEARS = 3  # smaller archive requests to stay under the free-tier burst limit


def _fetch_retry(region, s, e, tries=5):
    """Archive fetch with backoff on 429 (free-tier burst limit)."""
    delay = 3.0
    for i in range(tries):
        try:
            return _fetch_archive_endpoint(region, s, e)
        except Exception as ex:  # noqa: BLE001
            if "429" in str(ex) and i < tries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    return None


def _arch(region, start, end):
    time.sleep(1.5)  # gentle throttle between archive calls
    df = _fetch_retry(region, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if df is None or df.empty or "temperature_2m" not in df:
        return None
    df = df.dropna(subset=["temperature_2m"]).copy()
    df["cdd"] = compute_cdd(df["temperature_2m"])
    df["hdd"] = compute_hdd(df["temperature_2m"])
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["doy"] = df["timestamp"].dt.dayofyear
    return df


def evaluate(region, origin_str):
    origin = pd.Timestamp(origin_str, tz="UTC")
    tail_start = origin + pd.Timedelta(hours=OM_H)
    tail_end = origin + pd.Timedelta(hours=TAIL_END_H)

    actual = _arch(region, tail_start - pd.Timedelta(days=1), tail_end + pd.Timedelta(days=1))
    if actual is None:
        return None
    actual = actual[(actual["timestamp"] >= tail_start) & (actual["timestamp"] < tail_end)]
    if len(actual) < 24 * 10:
        return None

    recent = _arch(region, origin - pd.Timedelta(days=28), origin)
    hist = _arch(
        region, origin - pd.Timedelta(days=365 * NORMAL_YEARS), origin - pd.Timedelta(days=5)
    )
    if recent is None or hist is None or len(hist) < 24 * 200:
        return None

    def _err(col):
        a = actual[col].to_numpy()
        # recent-28d: (hour, dow) mean
        rmean = recent.groupby(["hour", "dow"])[col].mean()
        pr = np.array(
            [rmean.get((h, d), np.nan) for h, d in zip(actual["hour"], actual["dow"], strict=False)]
        )
        # weather-normal: (doy, hour) median over prior years (robust to extremes)
        nmean = hist.groupby(["doy", "hour"])[col].median()
        pn = np.array(
            [
                nmean.get((doy, h), np.nan)
                for doy, h in zip(actual["doy"], actual["hour"], strict=False)
            ]
        )
        m = ~np.isnan(pr) & ~np.isnan(pn)
        return (
            float(np.mean(np.abs(pr[m] - a[m]))),
            float(np.mean(np.abs(pn[m] - a[m]))),
            float(a.mean()),
        )

    # Straddle flag: origin month vs tail-midpoint month differ, or a shoulder month
    tail_mid = origin + pd.Timedelta(hours=(OM_H + TAIL_END_H) // 2)
    straddles = origin.month != tail_mid.month
    return {
        "region": region,
        "origin": origin_str,
        "tail": f"{tail_start.date()}..{tail_end.date()}",
        "straddles_turn": straddles,
        "temp": _err("temperature_2m"),
        "cdd": _err("cdd"),
        "hdd": _err("hdd"),
    }


REGIONS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["DUK"]
# Origins spanning ALL seasons; tail = origin +16..30d. Ramp months are the
# phase-lag stress cases; mid-summer/mid-winter are the near-null cases. Winter
# origins exercise HDD (winter-peaking BAs).
ORIGINS = [
    "2024-01-05",  # tail late-Jan       (WINTER, HDD)
    "2024-04-10",  # tail late-Apr..May  (SPRING RAMP)
    "2024-06-15",  # tail Jul            (mid-summer, near-null)
    "2024-08-01",  # tail mid-late Aug   (summer peak/shoulder)
    "2024-09-20",  # tail Oct            (FALL RAMP)
    "2024-11-20",  # tail Dec            (fall→winter ramp, HDD)
]

overall = {"recent": 0, "normal": 0, "ties": 0}
by_straddle = {True: {"recent": 0, "normal": 0}, False: {"recent": 0, "normal": 0}}
for REGION in REGIONS:
    print(f"\n== #283 Phase 0 weather-error backtest: {REGION} ==")
    print(
        f"{'origin':<12} {'straddle':<9} {'metric':<5} {'recent28d':>10} {'normal':>10} {'winner':>8} {'act_mean':>9}"
    )
    for o in ORIGINS:
        try:
            r = evaluate(REGION, o)
        except Exception as e:  # noqa: BLE001
            print(f"{o:<12} ERROR {str(e)[:50]}")
            continue
        if r is None:
            print(f"{o:<12} (insufficient data)")
            continue
        mo = pd.Timestamp(o, tz="UTC").month
        for metric in ("temp", "cdd", "hdd"):
            rec, nor, am = r[metric]
            w = "normal" if nor < rec else "recent"
            relevant = (metric == "cdd" and 4 <= mo <= 9) or (
                metric == "hdd" and (mo <= 3 or mo >= 11)
            )
            if relevant:
                # tie if within 5% of each other
                if abs(nor - rec) <= 0.05 * max(rec, 1e-6):
                    overall["ties"] += 1
                else:
                    overall[w] += 1
                    by_straddle[r["straddles_turn"]][w] += 1
            star = "*" if relevant else " "
            print(
                f"{o:<12} {str(r['straddles_turn']):<9} {metric:<5} {rec:>10.2f} {nor:>10.2f} {w:>8}{star} {am:>8.1f}"
            )

print("\n" + "=" * 60)
print(
    f"SEASON-RELEVANT metric wins:  normal={overall['normal']}  recent-28d={overall['recent']}  ties={overall['ties']}"
)
print(
    f"  at seasonal TURNS (straddle=True): normal={by_straddle[True]['normal']} recent={by_straddle[True]['recent']}"
)
print(
    f"  mid-season       (straddle=False): normal={by_straddle[False]['normal']} recent={by_straddle[False]['recent']}"
)
