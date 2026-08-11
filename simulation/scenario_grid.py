"""Precomputed what-if grid for the scenario simulator (#127).

The simulator shipped with an analytical heuristic — ``±2.5 % per 5 °F`` and
two smaller terms — because running the real physics per slider drag would
have put model inference in the web-tier request path, which
``CLAUDE.md``'s I/O guardrail forbids. This module is the other half of the
deal: the scoring job evaluates real forecasts across a fixed grid of weather
deltas, and the web tier interpolates between grid points. Redis stays the
only thing the web tier reads, and slider latency is unchanged.

**The delta is only meaningful if both sides come from the same inference
path.** ``scenario_engine._run_ensemble`` calls ``predict_xgboost`` directly,
a plain vectorised predict; production scoring runs
``recursive_autoregressive_forecast``, which chains each hour's prediction
into the next hour's lag features and is documented as "the single source of
truth for both production scoring and holdout evaluation". Those two paths
disagree on the same weather by construction, so a scenario computed through
one and a baseline computed through the other would report the difference
between the *paths* as if it were the response to *weather*. This module
therefore takes the forecaster as a parameter and the caller passes the
production one — the same correction #437 made to the backtest and #444 made
to ensemble weighting.

What the engine contributes is the part that is genuinely its own: copying
the frame, offsetting the drivers, and recomputing every derived feature that
depends on them (CDD/HDD, wind power, solar capacity factor, temp×hour).
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import structlog

from config import (
    SCENARIO_GRID_SOLAR_DELTAS_WM2,
    SCENARIO_GRID_TEMP_DELTAS_F,
    SCENARIO_GRID_WIND_DELTAS_MPH,
)
from simulation.scenario_engine import apply_weather_deltas

log = structlog.get_logger()

# A factor outside this band is not a demand response — it is a diverged
# recursive forecast (#296). The band was 0.25-4.0, which is four times wider
# in both directions than anything observed: across FPL/ERCOT/SPA/NWMT/ISONE
# the measured HOURLY factors at the slider extremes span ~0.91 to ~1.20
# (2026-08-11). A bound that loose can only catch a catastrophe, and this
# feature's realistic failure is a cell that wanders rather than explodes —
# which is what SPA did outside its training envelope.
#
# 0.6-1.7 is roughly 3x the observed excursion, so it should never bind on
# real physics, and it is tight enough to notice a wander.
_MIN_FACTOR, _MAX_FACTOR = 0.6, 1.7

# The origin cell should come back at exactly 1.0. Anything above this is
# a path disagreement worth an operator's attention rather than float noise.
_ORIGIN_DRIFT_WARN = 0.001


def grid_axes() -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """The three delta axes, in the order the payload is indexed."""
    return (
        SCENARIO_GRID_TEMP_DELTAS_F,
        SCENARIO_GRID_WIND_DELTAS_MPH,
        SCENARIO_GRID_SOLAR_DELTAS_WM2,
    )


def build_scenario_grid(
    featured: pd.DataFrame,
    future_df: pd.DataFrame,
    baseline: np.ndarray,
    forecaster: Callable[[list[pd.DataFrame]], list[np.ndarray]],
    horizon: int,
) -> dict[str, Any]:
    """Evaluate the forecaster across the delta grid and return factor curves.

    Args:
        featured: Historical featured frame — the seed for recursive lags.
            Passed through to ``forecaster`` unchanged.
        future_df: Forward feature frame the baseline was produced from.
        baseline: The baseline forecast over ``horizon`` hours, from the same
            ``forecaster``. Passing a baseline from a different path is the
            one mistake this module exists to prevent.
        forecaster: ``(list[future_frame]) -> list[predictions]``, BATCHED —
            one call for every scenario, not one call per scenario. The caller
            binds the production recursive path; tests bind a stub. Batched
            because cell-at-a-time issued 1,920 single-row predicts per region
            and cost 2.7x tick runtime (#462).
        horizon: Hours to evaluate. 24 for the simulator.

    Returns:
        Payload dict with ``axes``, ``horizon`` and ``factors`` — a nested
        list indexed ``[temp][wind][solar]``, each entry an ``horizon``-length
        list of ratios against the baseline. Ratios rather than absolute MW so
        the web tier can apply them to whichever baseline it reads, which may
        be a tick older than the grid.
    """
    temps, winds, solars = grid_axes()
    base = np.asarray(baseline, dtype=float)[:horizon]

    if base.size < horizon or not np.isfinite(base).all() or (base <= 0).any():
        raise ValueError("baseline must be finite, positive and at least `horizon` long")

    # Build every scenario frame first, then forecast them in ONE batched call.
    # Cell-at-a-time cost 1,920 single-row predicts per region and 2.7x tick
    # runtime (#462); the variants differ only in weather, so their step-i rows
    # travel through the model together.
    coords: list[tuple[float, float, float]] = []
    frames: list[pd.DataFrame] = []
    for t in temps:
        for w in winds:
            for s in solars:
                coords.append((t, w, s))
                frames.append(apply_weather_deltas(future_df, t, w, s))

    results = forecaster(frames) if frames else []
    if len(results) != len(frames):
        raise ValueError(f"forecaster returned {len(results)} curves for {len(frames)} scenarios")

    by_coord: dict[tuple[float, float, float], list[float]] = {}
    implausible: list[list[float]] = []
    for coord, preds_raw in zip(coords, results, strict=True):
        preds = np.asarray(preds_raw, dtype=float)[:horizon]
        if preds.size < horizon or not np.isfinite(preds).all():
            log.warning(
                "scenario_grid_cell_failed",
                temp=coord[0],
                wind=coord[1],
                solar=coord[2],
                n=int(preds.size),
            )
            by_coord[coord] = [1.0] * horizon
            continue
        ratio = preds / base
        # Do NOT clamp silently. A clamped 0.25 is not a measurement — it is a
        # diverged forecast wearing a plausible number, and the simulator would
        # render it as "demand drops 75%" with nothing to say otherwise. Treat
        # an out-of-band cell exactly like a non-finite one: drop it to the
        # baseline, count it, and say so.
        if float(np.min(ratio)) < _MIN_FACTOR or float(np.max(ratio)) > _MAX_FACTOR:
            log.warning(
                "scenario_grid_cell_implausible",
                temp=coord[0],
                wind=coord[1],
                solar=coord[2],
                lo=round(float(np.min(ratio)), 4),
                hi=round(float(np.max(ratio)), 4),
            )
            implausible.append(list(coord))
            by_coord[coord] = [1.0] * horizon
            continue
        by_coord[coord] = [round(float(v), 5) for v in ratio]

    ones = [1.0] * horizon
    factors: list[list[list[list[float]]]] = [
        [[by_coord.get((t, w, s), ones) for s in solars] for w in winds] for t in temps
    ]

    # The origin is now COMPUTED rather than defined as 1.0, and its deviation
    # from 1.0 is the grid's own parity check. A zero-delta scenario runs the
    # same weather through the same forecaster as the baseline, so anything
    # other than ~1.0 means the two sides of every ratio in this payload came
    # from different places — the exact failure this module was built to
    # prevent, and one that defining the origin away made invisible.
    origin = by_coord.get((0.0, 0.0, 0.0), ones)
    origin_drift = float(np.max(np.abs(np.asarray(origin, dtype=float) - 1.0)))
    # Log the HEALTHY path too, not only the breach. Warning-only meant a drift
    # of 0.0 left no trace, so "it held at zero all week" could not be
    # established from logs at all — the only way to check was polling the API
    # per region per tick, by hand. That makes the strongest evidence the check
    # is working the *absence* of a warning across 51 BAs, which is
    # indistinguishable from the grid never being computed.
    #
    # Same correction #478 made for shadow weights three commits earlier, and
    # the "configured and inert" pattern this project keeps re-finding
    # (docs/monitoring/README.md). A parity check nobody can confirm is running
    # is a parity check nobody should trust.
    if origin_drift > _ORIGIN_DRIFT_WARN:
        log.warning("scenario_grid_origin_drift", drift=round(origin_drift, 5))
    else:
        log.info("scenario_grid_origin_ok", drift=round(origin_drift, 5), cells=len(coords))

    return {
        "axes": {"temp_f": list(temps), "wind_mph": list(winds), "solar_wm2": list(solars)},
        "horizon": horizon,
        "factors": factors,
        "origin_drift": round(origin_drift, 5),
        "implausible_cells": implausible,
        "envelope": _envelope(featured, future_df, temps, winds, solars),
    }


def _envelope(
    featured: pd.DataFrame,
    future_df: pd.DataFrame,
    temps: tuple[float, ...],
    winds: tuple[float, ...],
    solars: tuple[float, ...],
) -> dict[str, list[bool]]:
    """Which axis positions keep their driver inside the model's observed range.

    XGBoost is a tree ensemble and does not extrapolate. Once a shifted driver
    leaves the range the booster was trained on, every split routes the same
    way and the response stops depending on the input — measured 2026-08-11:
    FPL's first-hour factor is 1.0264 at BOTH +10 and +20 F, and 0.979 at both
    -10 and -20 (Florida has no cold training data). SPA saturates above +5 and
    then wanders, which is worse than flat: outside the training envelope the
    prediction is unconstrained rather than merely constant.

    The grid is faithfully reporting what the model says. The model has nothing
    to say out there, and the UI needs to be able to show that rather than
    render an extrapolation as physics.

    Compares each shifted forecast series against the observed range in
    ``featured``, which is this BA's own history. A position is in-envelope
    only if EVERY hour of the shifted series stays inside it — a partially
    covered position is still asking the model to extrapolate for some hours.

    Wind and solar are checked the same way. Note the check is per-axis, not
    per-cell: a cell can be in-envelope on all three axes and still sit in a
    sparse CORNER the booster never saw jointly. That is a real limitation of
    this flag, not something it detects.
    """

    def flags(column: str, deltas: tuple[float, ...]) -> list[bool]:
        if column not in featured.columns or column not in future_df.columns:
            return [True] * len(deltas)
        observed = pd.to_numeric(featured[column], errors="coerce").dropna()
        future = pd.to_numeric(future_df[column], errors="coerce").dropna()
        if observed.empty or future.empty:
            return [True] * len(deltas)
        lo, hi = float(observed.min()), float(observed.max())
        out = []
        for d in deltas:
            shifted = future + d
            out.append(bool((shifted >= lo).all() and (shifted <= hi).all()))
        return out

    return {
        "temp_f": flags("temperature_2m", temps),
        "wind_mph": flags("wind_speed_80m", winds),
        "solar_wm2": flags("shortwave_radiation", solars),
    }


def _axis_position(axis: list[float] | tuple[float, ...], value: float) -> tuple[int, int, float]:
    """Locate ``value`` on ``axis``, returning ``(lo_idx, hi_idx, weight)``.

    ``weight`` is how far along the ``lo -> hi`` interval the value sits.
    Values outside the axis clamp to its ends rather than extrapolating: the
    grid is built to span the slider domain, so an out-of-range value means a
    UI change outran the grid, and holding the edge is the honest answer.
    """
    n = len(axis)
    if n == 1:
        return 0, 0, 0.0
    if value <= axis[0]:
        return 0, 0, 0.0
    if value >= axis[-1]:
        return n - 1, n - 1, 0.0

    hi = int(np.searchsorted(np.asarray(axis, dtype=float), value, side="left"))
    lo = max(hi - 1, 0)
    span = axis[hi] - axis[lo]
    weight = 0.0 if span == 0 else (value - axis[lo]) / span
    return lo, hi, float(weight)


def interpolate_scenario_factors(
    payload: dict[str, Any],
    temp_delta_f: float,
    wind_delta_mph: float,
    solar_delta_wm2: float,
) -> np.ndarray | None:
    """Trilinearly interpolate the stored grid at an arbitrary slider position.

    Returns an hourly factor curve, or ``None`` if the payload is unusable —
    the caller falls back to the heuristic rather than rendering nothing.
    """
    try:
        axes = payload["axes"]
        factors = payload["factors"]
        temps = axes["temp_f"]
        winds = axes["wind_mph"]
        solars = axes["solar_wm2"]
    except (KeyError, TypeError):
        log.warning("scenario_grid_payload_malformed")
        return None

    ti, tj, tw = _axis_position(temps, float(temp_delta_f))
    wi, wj, ww = _axis_position(winds, float(wind_delta_mph))
    si, sj, sw = _axis_position(solars, float(solar_delta_wm2))

    try:
        corners = [
            (np.asarray(factors[a][b][c], dtype=float), fa * fb * fc)
            for a, fa in ((ti, 1.0 - tw), (tj, tw))
            for b, fb in ((wi, 1.0 - ww), (wj, ww))
            for c, fc in ((si, 1.0 - sw), (sj, sw))
            if fa * fb * fc > 0.0
        ]
    except (IndexError, TypeError, ValueError):
        log.warning("scenario_grid_index_failed")
        return None

    if not corners:
        return None

    length = min(c.size for c, _ in corners)
    if length == 0:
        return None

    total = sum(weight for _, weight in corners)
    blended = sum(curve[:length] * weight for curve, weight in corners) / total
    return blended if np.isfinite(blended).all() else None
