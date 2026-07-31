"""CAISO zonal (TAC-area) load, for the bottom-up generalisation test.

The NYISO bottom-up result (`NYISO_BOTTOM_UP_STUDY.md`, +0.729 WAPE pts)
needed testing on another ISO before anyone builds on it. PJM and ISO-NE were
the obvious candidates and **both are gated**: PJM's Data Miner 2 API and the
ISO-NE web services API each return HTTP 401 without a registered key, and
every open ISO-NE static report path tried returned 404 (verified 2026-07-31).
Registration is free for both but requires creating accounts.

CAISO's OASIS is open, so it is the available generalisation test. It is a
genuinely different case rather than a convenience substitute: **five** TAC
areas against NYISO's eleven, western rather than northeastern geography, and
one very unusual zone (MWD is pumping load, not population load).

OASIS caps a request at roughly a month, so history is fetched in chunks.
``SLD_FCST`` with ``market_run_id=ACTUAL`` gives hourly integrated actual load
per TAC area; the file also carries every other WECC balancing authority, so
the CAISO-proper areas are selected explicitly.
"""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime, timedelta

import pandas as pd

from scripts.arima_order_exog_study import ARCHIVE_LAG_DAYS, _get_with_retry

OASIS = "http://oasis.caiso.com/oasisapi/SingleZip"

#: CAISO-proper TAC areas. The OASIS response also lists ~30 other WECC BAs
#: (AVA, BPAT, PACE, …) that are not part of CAISO's own load, so this is an
#: explicit allow-list rather than "everything in the file".
#:
#: Coordinates are approximate load centres, same standard as the NYISO probe.
CAISO_ZONE_COORDS: dict[str, tuple[float, float]] = {
    "PGE-TAC": (37.90, -121.50),  # PG&E — northern/central CA
    "SCE-TAC": (34.05, -117.80),  # Southern California Edison — inland basin
    "SDGE-TAC": (32.72, -117.16),  # San Diego Gas & Electric
    "VEA-TAC": (36.21, -115.98),  # Valley Electric — Pahrump NV
    "MWD-TAC": (33.90, -117.30),  # Metropolitan Water District — pumping load
}

#: Days per OASIS request. The endpoint rejects much longer spans.
CHUNK_DAYS = 30


def fetch_caiso_zonal_load(months: int) -> pd.DataFrame:
    """Hourly actual load per CAISO TAC area, chunked over OASIS."""
    end = datetime.now(UTC) - timedelta(days=ARCHIVE_LAG_DAYS)
    start = end - timedelta(days=months * 31)

    frames: list[pd.DataFrame] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + timedelta(days=CHUNK_DAYS), end)
        params = {
            "queryname": "SLD_FCST",
            "market_run_id": "ACTUAL",
            "startdatetime": f"{cursor:%Y%m%d}T07:00-0000",
            "enddatetime": f"{stop:%Y%m%d}T07:00-0000",
            "version": "1",
            "resultformat": "6",
        }
        try:
            r = _get_with_retry(OASIS, params=params, timeout=120)
        except Exception as e:
            print(f"    CAISO {cursor:%Y-%m-%d}: unavailable ({str(e)[:60]})", flush=True)
            cursor = stop
            continue
        if r.content[:2] != b"PK":
            print(f"    CAISO {cursor:%Y-%m-%d}: not a zip ({len(r.content)}B)", flush=True)
            cursor = stop
            continue
        z = zipfile.ZipFile(io.BytesIO(r.content))
        for name in z.namelist():
            if name.endswith(".csv"):
                frames.append(pd.read_csv(z.open(name)))
        print(f"    CAISO {cursor:%Y-%m-%d} → {stop:%Y-%m-%d}: ok", flush=True)
        cursor = stop

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df[df["TAC_AREA_NAME"].isin(CAISO_ZONE_COORDS)]
    if df.empty:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["INTERVALSTARTTIME_GMT"], utc=True, format="mixed")
    df["MW"] = pd.to_numeric(df["MW"], errors="coerce")
    wide = df.pivot_table(index="timestamp", columns="TAC_AREA_NAME", values="MW", aggfunc="mean")
    # Same rule as the NYISO source: an hour missing any zone cannot be summed,
    # so drop it rather than silently under-counting the total.
    return wide.dropna(how="any").sort_index()
