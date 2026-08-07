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

``skill_payload`` is the single definition of the published skill block. It
is nested under ``skill`` inside ``gridpulse:forecast:{region}:1h``, written
by ``jobs.phases._baseline_substitution`` on the ticks where a region is
substituted onto the baseline, and passed through verbatim by
``/api/v1/forecast``. There is no standalone ``gridpulse:skill:{region}``
key — an earlier draft of this module claimed one, and the continuous
per-tick publication it anticipated was never built.
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
    model_mape: float,
    series: np.ndarray,
    *,
    lag_h: int = SEASONAL_NAIVE_LAG_H,
    window_days: int | None = None,
) -> dict:
    """The per-region skill block, nested under ``skill`` in the forecast payload.

    ``beats_baseline`` is the field worth acting on. It is explicitly None
    when either side is missing, so a consumer cannot mistake an unmeasured
    region for a passing one — the failure mode that let SEC serve a
    worse-than-nothing forecast without anything noticing.

    Every derived field nulls together, and that is what makes the block safe
    to hand to ``should_serve_baseline``. A non-finite ``points_vs_baseline``
    would pass that policy's ``points > -MIN_POINTS`` test — NaN compares
    False against everything — and fall through to *substitute*, which is the
    one direction the policy is written to never take on a missing
    measurement. None short-circuits it at "skill not measurable" instead.

    ``window_days`` records how much history the caller measured over, for a
    reader who cannot otherwise tell a 7-day number from a 30-day one. It is
    disclosure, not policy — ``n_hours`` is what gates substitution.

    The served block carries one field this function does not set:
    ``decision``, the published reason string, which the caller attaches from
    ``should_serve_baseline`` after the fact because that policy reads this
    block as its input.
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
        "window_days": window_days,
    }


# ── serving a baseline where the model has negative skill ────

#: A model must lose to the baseline by at least this many error points
#: before its forecast is replaced. Set well clear of the noise floor: on the
#: 2026-07-27 fleet, eight regions sat within ~1 point of the line and one —
#: SEC — was ~4 points down at every measured lead. Substituting on a
#: fractional deficit would churn served forecasts for no measurable gain.
BASELINE_SUBSTITUTION_MIN_POINTS = 2.0

#: Minimum measured hours before a skill number may change what a region is
#: served.
BASELINE_SUBSTITUTION_MIN_HOURS = 24 * 7


def should_serve_baseline(skill_block: dict | None) -> tuple[bool, str]:
    """Should this region be served the baseline instead of its model?

    Returns ``(decision, reason)`` — the reason is published, because a region
    whose forecast silently changed source is exactly what a user should not
    have to discover for themselves.

    Deliberately asymmetric. Absence of a measurement is never a reason to
    substitute: an unmeasured region keeps its model, so a broken skill
    pipeline degrades to today's behaviour rather than swapping all 51
    regions onto a naive forecast.
    """
    if not isinstance(skill_block, dict):
        return False, "no skill measurement"
    points = skill_block.get("points_vs_baseline")
    hours = skill_block.get("n_hours") or 0
    if points is None:
        return False, "skill not measurable"
    if hours < BASELINE_SUBSTITUTION_MIN_HOURS:
        return False, f"only {hours}h measured, need {BASELINE_SUBSTITUTION_MIN_HOURS}h"
    if points > -BASELINE_SUBSTITUTION_MIN_POINTS:
        return False, f"model within {BASELINE_SUBSTITUTION_MIN_POINTS} pts of baseline"
    return True, (
        f"model loses to seasonal-naive by {abs(points):.2f} error points "
        f"over {hours}h — serving the baseline instead"
    )


def seasonal_naive_forecast(
    history: np.ndarray, horizon_h: int, *, lag_h: int = SEASONAL_NAIVE_LAG_H
) -> np.ndarray:
    """Project the baseline forward: each future hour takes the same clock
    hour from the most recent FULLY OBSERVED day.

    Lead 1–24 reads yesterday, 25–48 reads two days ago, and so on — exactly
    the predictor the skill score measures, so what gets served is what was
    measured. Recursing on the baseline's own output instead would serve
    something no measurement covers.

    ``history`` is the observed series ending at the forecast origin, oldest
    first. Returns an empty array when history is too short to project even
    one day — the caller must then keep the model.

    The index is derived, not guessed. For lead ``h`` the target is
    ``origin + h``; the source is ``target - lag_h*k`` where ``k`` is the
    smallest integer putting the source at or before the origin, i.e.
    ``k = ceil(h / lag_h)``. So the source index is
    ``(N-1) + h - lag_h*k`` — which always lands inside the LAST observed
    day, meaning every day of the horizon repeats that same day.

    An earlier cut used ``N - lag_h*k + (h-1) % lag_h``, which walks one
    extra day back per block: lead 25 read 72h before its target instead of
    48h, and lead 49 read 96h instead of 72h. It matched what the skill
    score measures only for the first day, so the served series drifted
    further from the measured predictor the longer the horizon ran. The unit
    test encoded the same wrong arithmetic and passed.
    """
    y = np.asarray(history, dtype=float)
    if y.size < lag_h:
        return np.empty(0, dtype=float)
    out = np.empty(horizon_h, dtype=float)
    last = y.size - 1
    for h in range(1, horizon_h + 1):
        k = -(-h // lag_h)  # ceil division
        idx = last + h - lag_h * k
        # A gap in the most recent day must not disable the whole projection:
        # step back further whole days for that hour only. Every candidate is
        # still the same clock hour and still a real observation.
        while idx >= 0 and not np.isfinite(y[idx]):
            idx -= lag_h
        if idx < 0 or idx >= y.size:
            return np.empty(0, dtype=float)
        out[h - 1] = y[idx]
    return out
