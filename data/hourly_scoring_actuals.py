"""Apply the existing quality signals to every retained scoring target.

The forecast-input guard deliberately examines a short tail. Scoring must
also check older retained targets: aging out of that tail is not a revision.
This function never changes the caller's forecast-input frame or thresholds.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from data.quality import coerce_demand_artifacts
from models.hourly_scoring import WINDOW_HOURS


def hourly_actuals(
    demand_df: pd.DataFrame | None, *, now: datetime | None = None
) -> tuple[dict[str, float], dict[str, dict]]:
    if demand_df is None or demand_df.empty:
        return {}, {}
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("Scoring quality checks require an aware clock")
    cutoff = now - timedelta(hours=WINDOW_HOURS)
    frame = demand_df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    # Keep the preceding day's context and omit future DF-only rows. The
    # number of eligible rows determines the guard span, not the frame tail.
    frame = frame.loc[
        (frame["timestamp"] > cutoff - timedelta(hours=24)) & (frame["timestamp"] <= now)
    ].sort_values("timestamp")
    candidates = int((frame["timestamp"] > cutoff).sum())
    cleaned, exclusions = coerce_demand_artifacts(frame, trailing_hours=candidates)
    actuals = {}
    for row in cleaned.itertuples():
        value = row.demand_mw
        if (
            row.timestamp > cutoff
            and not isinstance(value, (bool, np.bool_))
            and pd.notna(value)
            and np.isfinite(value)
            and value > 0
        ):
            actuals[row.timestamp.isoformat()] = float(value)
    return actuals, {e["ts"]: {"mw": e["mw"], "reason": e["reason"]} for e in exclusions}
