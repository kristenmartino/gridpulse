"""Forecast skill against a naive baseline.

A forecasting system that cannot beat "yesterday, same hour" is not adding
information, and until now nothing in this codebase measured that. The gap
was not academic: SEC has been served a three-model ensemble scoring 18.0%
against a seasonal-naive baseline's 11.5% — actively worse than doing
nothing — and every existing instrument reported it as merely *bad* rather
than *worse than free*.

Skill score is the standard framing::

    skill = 1 - error_model / error_baseline

Positive means the model beats the baseline; zero means it matches it;
negative means it is subtracting value. Reported here in error *points* as
well, because a point difference is what a reader can act on.

The baseline is deliberately the dumbest defensible one. Seasonal-naive at a
24-hour lag is available to anyone with the demand feed, needs no model, no
weather, and no training — so beating it is the minimum bar a forecasting
product clears, not a target.
"""

from __future__ import annotations

import numpy as np

#: Lag for the seasonal-naive baseline, in hours. 24 rather than 1: every
#: value it uses is known a full day before the target hour, so it is a fair
#: opponent for a 24h-lead forecast. A 1-hour-lag baseline would be strictly
#: better informed than the model it judges.
SEASONAL_NAIVE_LAG_H = 24


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error over hours with a positive actual."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    ok = np.isfinite(a) & np.isfinite(p) & (a > 0)
    if not ok.any():
        return float("nan")
    return float(np.mean(np.abs(a[ok] - p[ok]) / a[ok]) * 100.0)


def seasonal_naive_mape(series: np.ndarray, lag_h: int = SEASONAL_NAIVE_LAG_H) -> float:
    """MAPE of "the value ``lag_h`` hours ago" over an hourly series.

    ``series`` must be hourly and gap-aligned — pass a reindexed frame, not a
    compacted one, or the lag silently reaches across a gap and flatters the
    baseline by comparing adjacent readings that are days apart.
    """
    y = np.asarray(series, dtype=float)
    if y.size <= lag_h:
        return float("nan")
    return mape(y[lag_h:], y[:-lag_h])


def skill_score(model_mape: float, baseline_mape: float) -> float | None:
    """``1 - model/baseline``, or None when the baseline carries no signal.

    Returns None rather than 0.0 for a non-finite or non-positive baseline:
    "no comparison was possible" and "the model exactly matched the baseline"
    are different states, and collapsing them would let a missing measurement
    read as a neutral result.
    """
    if not np.isfinite(model_mape) or not np.isfinite(baseline_mape) or baseline_mape <= 0:
        return None
    return round(1.0 - (model_mape / baseline_mape), 4)


def skill_payload(
    model_mape: float, series: np.ndarray, *, lag_h: int = SEASONAL_NAIVE_LAG_H
) -> dict:
    """Per-region skill block for ``gridpulse:skill:{region}``.

    ``beats_baseline`` is the field worth acting on. It is explicitly None
    when either side is missing, so a consumer cannot mistake an unmeasured
    region for a passing one — the failure mode that let SEC serve a
    worse-than-nothing forecast without anything noticing.
    """
    baseline = seasonal_naive_mape(series, lag_h)
    skill = skill_score(model_mape, baseline)
    return {
        "model_mape": None if not np.isfinite(model_mape) else round(float(model_mape), 3),
        "baseline_mape": None if not np.isfinite(baseline) else round(baseline, 3),
        "baseline": f"seasonal-naive lag {lag_h}h",
        "skill": skill,
        "points_vs_baseline": (None if skill is None else round(baseline - model_mape, 3)),
        "beats_baseline": None if skill is None else bool(skill > 0),
        "n_hours": int(np.isfinite(np.asarray(series, dtype=float)).sum()),
    }
