"""Demand-reading plausibility — is EIA's number a measurement or an artifact?

Promoted from ``components/_callbacks_us_grid.py`` (#225), where it guarded only
the US-Grid stress surfaces and ``/api/v1/grid/summary``. The 2026-07 vintage
study (#309/#312) showed the same artifacts reaching two unguarded consumers:

* **The forecast anchor** — ``demand_lag_1h`` and 20 sibling autoregressive
  features seed from the newest reading (``feature_engineering.py:220``), so a
  partial poisons every recursive forecast hour.
* **The region-page tiles** — LADWP rendered NOW 730 MW / trend −80.6% /
  "demand is 78.1% below average" from a partial that settled to ~3,034 an
  hour later.

## What EIA actually publishes (measured 2026-07-15/16, Form EIA-930 grounded)

The newest hour's ``D`` value cycles **stub → partial → settled**: the row is
born mid-hour carrying the BA's own day-ahead forecast (``D == DF`` exactly),
a first estimate replaces it ~2-6 min after the hour closes, and the same-day
deadline value lands by ~+50 min. Form EIA-930 *mandates* this: same-day files
are due within 60 minutes and respondents are told to "submit their best
estimates on schedule and correct the data with a resubmission within 3 days."
Partials are required behavior, not a bug — the guard's job is to keep the
gross ones out of anchors and KPIs, with disclosure.

## Signal design — every threshold traces to a measured case

Three independent signals (any fires ⇒ artifact):

1. **Near-zero glitch** — below 10% of the trailing-24h median (#142's
   region-relative rule; catches TIDC-class zeros that survive upstream
   coercion in aggregate lists).
2. **Single-step collapse** — a >60% one-hour drop that ALSO lands below 60%
   of the median (both required, so a return-from-spike is not flagged).
   Catches LDWP 3,464→730 (−79%, 22% of median) and the #225 "APS −90.7%".
3. **Day-ahead ratio** (new, #309) — below 50% of the BA's own day-ahead
   forecast (``DF``), **low side only**. Load-bearing for stuck partials that
   evade signals 1-2: IID frozen at 339 for 6+ hours (34% of median — above
   signal 1; no step — signal 2 blind; 33% of DF ⇒ caught), AZPS frozen at
   1959 (26% of DF). Deliberate non-fires, measured: the ``D == DF``
   placeholder stub gives ratio exactly 1.0 (and the stub is a GOOD anchor —
   fleet mean ~2.7% error; removing it measured WORSE, 6.55%→7.72%); PSCO
   legitimately runs 118-121% of its own day-ahead, which is why there is no
   high-side test — a symmetric band would false-flag real demand. BPAT's
   +20% high partial is therefore deliberately uncatchable: the accepted,
   documented residual.

## Scope rule for series coercion

``coerce_demand_artifacts`` judges only the **trailing hours** of a frame.
Measured: gross artifacts live only at the tail — EIA revisions settle older
hours in every fresh fetch — while bulk-revision regions' same-day values are
provisional-but-plausible (14-40% off, revised at the next-morning daily file)
and must NOT be excluded; that is provenance-callout territory, not artifact
territory.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import (
    DEMAND_ARTIFACT_DAY_AHEAD_FRACTION,
    DEMAND_ARTIFACT_NEAR_ZERO_FRACTION,
    DEMAND_ARTIFACT_STEP_DROP_FRACTION,
    DEMAND_ARTIFACT_STEP_LOW_FRACTION,
)

#: How many trailing hours of a demand frame are candidates for exclusion.
#: Measured settle behavior: stub→partial→settled completes within ~1h for
#: most BAs; the worst observed partial (LDWP) held ~44 min, and AZPS held a
#: partial for ~46 min. Six hours covers every observed case with margin
#: while structurally exempting the settled bulk of the series.
DEMAND_ARTIFACT_TRAILING_HOURS = 6


def is_real_positive(value: Any) -> bool:
    """Strict guard for downstream arithmetic: True only when ``value`` is
    a finite (non-NaN, non-inf) strictly positive number.

    Rejects strings outright (no silent coercion) and normalizes numpy
    bool returns to Python ``bool`` so callers can use ``is True / is False``.
    """
    if value is None or isinstance(value, (str, bytes)):
        return False
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(f) and f > 0)


def is_implausible_demand_artifact(
    current_mw: float,
    today_mw: list,
    prev_mw: float | None = None,
    day_ahead_mw: float | None = None,
) -> bool:
    """Check if a BA's demand reading is an implausible artifact.

    See the module docstring for the three signals and the measured cases
    behind every threshold. ``day_ahead_mw`` (EIA's ``DF`` for the same hour)
    enables signal 3; when absent or non-finite that signal is skipped — the
    guard degrades to the original #225 pair.

    Args:
        current_mw: The reading under judgment, MW.
        today_mw: Trailing ~24h demand list (may contain NaN / zero).
        prev_mw: The previous real reading, MW — enables the step-collapse
            signal. Optional; when absent only signals (1) and (3) apply.
        day_ahead_mw: The BA's own day-ahead forecast for the same hour.

    Returns:
        True if ``current_mw`` should be excluded from anchors and KPIs.
    """
    if not is_real_positive(current_mw):
        return True  # Already filtered out by is_real_positive, but be explicit

    if not today_mw:
        return False  # No history available; assume current is real

    positive_history = [float(v) for v in today_mw if is_real_positive(v)]
    if not positive_history:
        return False  # All history is suspect; can't establish scale, assume current is real

    median_24h = float(np.median(positive_history))
    if median_24h <= 0:
        return False  # Degenerate history; can't establish scale, assume current is real

    # (3) day-ahead ratio, PAIRED with a below-normal-scale co-signal (the
    # same design as signal 2): some BAs' own day-ahead forecasts are badly
    # high (PSEI's mean DF error is 47%), so a bare ratio would false-flag a
    # real deep trough under a bad forecast. A stuck partial is far below the
    # median too; a real reading under a bad DF usually is not.
    if (
        is_real_positive(day_ahead_mw)
        and current_mw < DEMAND_ARTIFACT_DAY_AHEAD_FRACTION * float(day_ahead_mw)
        and current_mw < DEMAND_ARTIFACT_STEP_LOW_FRACTION * median_24h
    ):
        return True

    # (1) near-zero glitch
    if current_mw < DEMAND_ARTIFACT_NEAR_ZERO_FRACTION * median_24h:
        return True
    # (2) single-step collapse (needs a prior real reading)
    return bool(
        is_real_positive(prev_mw)
        and current_mw < DEMAND_ARTIFACT_STEP_DROP_FRACTION * float(prev_mw)
        and current_mw < DEMAND_ARTIFACT_STEP_LOW_FRACTION * median_24h
    )


def coerce_demand_artifacts(
    demand_df: pd.DataFrame | None,
    *,
    trailing_hours: int = DEMAND_ARTIFACT_TRAILING_HOURS,
) -> tuple[pd.DataFrame | None, list[dict[str, Any]]]:
    """NaN-coerce implausible readings in a demand frame's trailing hours.

    Returns ``(cleaned_frame, exclusions)`` where ``cleaned_frame`` is a copy
    (the input is never mutated — the vintage study must keep seeing the raw
    values) and ``exclusions`` is ``[{"ts", "mw", "reason"}]`` for disclosure
    on the tiles, the operating summary, and ``/grid/summary``.

    Only the trailing ``trailing_hours`` rows are judged (module docstring:
    artifacts live at the tail). Each candidate row is judged against its OWN
    trailing-24h median and its own prior real reading, so a run of stuck
    partials is caught row by row. Rows already NaN are skipped — they are
    absences, not artifacts. Degrades gracefully when ``forecast_mw`` is
    absent (GCS-fallback frames carry it, but never assume).
    """
    if demand_df is None or len(demand_df) == 0 or "demand_mw" not in demand_df.columns:
        return demand_df, []

    cleaned = demand_df.copy()
    values = pd.to_numeric(cleaned["demand_mw"], errors="coerce")
    day_ahead = (
        pd.to_numeric(cleaned["forecast_mw"], errors="coerce")
        if "forecast_mw" in cleaned.columns
        else None
    )

    n = len(cleaned)
    start = max(0, n - trailing_hours)
    exclusions: list[dict[str, Any]] = []

    for i in range(start, n):
        current = values.iloc[i]
        if not is_real_positive(current):
            continue  # an absence, not an artifact

        history = [v for v in values.iloc[max(0, i - 24) : i].tolist() if is_real_positive(v)]
        prev = history[-1] if history else None
        df_mw = float(day_ahead.iloc[i]) if day_ahead is not None else None

        if not is_implausible_demand_artifact(
            float(current), history, prev_mw=prev, day_ahead_mw=df_mw
        ):
            continue

        # Name the reason for the disclosure surfaces — mirror the signal order.
        median_24h = float(np.median(history)) if history else float("nan")
        if (
            df_mw is not None
            and is_real_positive(df_mw)
            and float(current) < (DEMAND_ARTIFACT_DAY_AHEAD_FRACTION * df_mw)
        ):
            reason = f"{float(current) / df_mw:.0%} of the BA's own day-ahead forecast"
        elif np.isfinite(median_24h) and float(current) < (
            DEMAND_ARTIFACT_NEAR_ZERO_FRACTION * median_24h
        ):
            reason = f"near-zero vs 24h median ({float(current) / median_24h:.0%})"
        else:
            drop = (1 - float(current) / float(prev)) if prev else float("nan")
            reason = f"{drop:.0%} single-hour drop to {float(current) / median_24h:.0%} of the daily median"

        ts = cleaned["timestamp"].iloc[i] if "timestamp" in cleaned.columns else None
        exclusions.append(
            {
                "ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "mw": round(float(current), 2),
                "reason": reason,
            }
        )
        cleaned.iloc[i, cleaned.columns.get_loc("demand_mw")] = np.nan
        values.iloc[i] = np.nan  # later rows must not use an excluded value as prev/history

    return cleaned, exclusions
