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

# Ratios are clipped before they are stored. A scenario factor outside this
# band is not a demand response, it is a diverged recursive forecast — the
# #296 failure mode — and clipping keeps one bad grid cell from rendering a
# simulator chart that dwarfs the baseline.
_MIN_FACTOR, _MAX_FACTOR = 0.25, 4.0


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
    forecaster: Callable[[pd.DataFrame], np.ndarray],
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
        forecaster: ``(future_frame) -> predictions``. The caller binds the
            production recursive path; tests bind a stub.
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

    factors: list[list[list[list[float]]]] = []
    for t in temps:
        wind_rows: list[list[list[float]]] = []
        for w in winds:
            solar_rows: list[list[float]] = []
            for s in solars:
                if t == 0.0 and w == 0.0 and s == 0.0:
                    # The origin is the baseline by definition. Re-running it
                    # would spend a grid cell to reproduce a row of 1.0s, and
                    # any drift it showed would be nondeterminism, not physics.
                    solar_rows.append([1.0] * horizon)
                    continue

                scenario_future = apply_weather_deltas(future_df, t, w, s)
                preds = np.asarray(forecaster(scenario_future), dtype=float)[:horizon]

                if preds.size < horizon or not np.isfinite(preds).all():
                    log.warning(
                        "scenario_grid_cell_failed", temp=t, wind=w, solar=s, n=int(preds.size)
                    )
                    solar_rows.append([1.0] * horizon)
                    continue

                ratio = np.clip(preds / base, _MIN_FACTOR, _MAX_FACTOR)
                solar_rows.append([round(float(v), 5) for v in ratio])
            wind_rows.append(solar_rows)
        factors.append(wind_rows)

    return {
        "axes": {"temp_f": list(temps), "wind_mph": list(winds), "solar_wm2": list(solars)},
        "horizon": horizon,
        "factors": factors,
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
