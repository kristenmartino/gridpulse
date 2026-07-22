"""Aggregate per-point weather frames into one representative frame (ADR-012).

A BA's demand responds to weather across its whole footprint, but the
fetch historically sampled ONE point. The multi-point study
(``docs/MULTIPOINT_WEATHER_STUDY.md``) measured aggregating ~12 footprint
points at **+1.14 sMAPE pts** — and measured population *weighting* as
adding nothing (C−B ≈ 0), so this is a plain, unweighted aggregation.

Kept out of ``data/weather_client.py`` deliberately: the per-variable
special cases below are the subtle part and deserve isolated tests, while
the client stays focused on fetch/fallback safety.

Three rules over ``config.WEATHER_VARIABLES``:

* ``wind_direction_10m`` — **circular mean**. An arithmetic mean of 350°
  and 10° is 180°, i.e. due south when the true answer is due north.
* ``weather_code`` — **mode**. WMO codes are ordinal categories
  (3=overcast, 61=rain, 71=snow, 95=thunderstorm); mean(3, 95) = 49 =
  "drizzle" is nonsense.
* everything else — **renormalizing nanmean** (see the null note below).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from config import WEATHER_VARIABLES

#: Degrees on a circle — an arithmetic mean is physically wrong.
CIRCULAR_VARS: frozenset[str] = frozenset({"wind_direction_10m"})
#: Ordinal WMO categories — average is meaningless, take the mode.
MODE_VARS: frozenset[str] = frozenset({"weather_code"})


class TimestampGridMismatchError(ValueError):
    """Per-point frames disagree on their hour grid — the caller must fail
    open to single-point rather than aggregate misaligned rows."""


def aggregate_weather(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Collapse K per-point frames into one ``[timestamp] + 17 vars`` frame.

    ``frames`` must share an identical timestamp grid (they come from one
    multi-point Open-Meteo call, so they do); a mismatch raises
    :class:`TimestampGridMismatchError` so the caller can fail open.

    **Null policy — the one place this deliberately differs from the study
    script.** The study aggregated with ``nansum(w * arr)``, where a null
    point contributes 0 rather than being dropped: ``[10, 20, nan]``
    averages to 10.0, not 15.0. That never bit offline (census centroids
    are over land, ERA5-only), but in production a grid-snapped cell can
    land over water or outside CONUS (NBM is CONUS-only), and a cell that
    starts nulling would drag every aggregated value toward zero — a
    silent coverage collapse of exactly the #161 flavor. So this uses a
    **renormalizing nanmean**: null points are dropped and the remainder
    averaged. An all-null hour yields NaN, which ``engineer_features``
    imputes exactly as it does a single-point null today.
    """
    if not frames:
        raise TimestampGridMismatchError("no frames to aggregate")
    if len(frames) == 1:
        return frames[0].copy()

    base_ts = pd.to_datetime(frames[0]["timestamp"], utc=True)
    for f in frames[1:]:
        ts = pd.to_datetime(f["timestamp"], utc=True)
        if len(ts) != len(base_ts) or not ts.equals(base_ts):
            raise TimestampGridMismatchError(
                f"per-point timestamp grids differ ({len(ts)} vs {len(base_ts)} rows)"
            )

    out: dict[str, object] = {"timestamp": base_ts}
    for var in WEATHER_VARIABLES:
        if var not in frames[0].columns:
            continue
        # (n_hours, K)
        arr = np.column_stack(
            [pd.to_numeric(f[var], errors="coerce").to_numpy(dtype=float) for f in frames]
        )
        # An all-null hour is an EXPECTED condition (the 3 vars ERA5 lacks
        # on deep history, or every cell outside a model's domain) and
        # yields NaN by design — numpy's "Mean of empty slice" warning
        # would be noise in the job logs, so it is suppressed here rather
        # than left to obscure real warnings.
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if var in CIRCULAR_VARS:
                theta = np.deg2rad(arr)
                s = np.nanmean(np.sin(theta), axis=1)
                c = np.nanmean(np.cos(theta), axis=1)
                out[var] = np.rad2deg(np.arctan2(s, c)) % 360.0
            elif var in MODE_VARS:
                out[var] = _row_mode(arr)
            else:
                out[var] = np.nanmean(arr, axis=1)
    return pd.DataFrame(out)


def _row_mode(arr: np.ndarray) -> np.ndarray:
    """Most-common finite value per row; NaN when a row is all-null."""
    picked = np.full(arr.shape[0], np.nan)
    for i in range(arr.shape[0]):
        row = arr[i]
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            continue
        values, counts = np.unique(finite, return_counts=True)
        picked[i] = values[np.argmax(counts)]
    return picked
