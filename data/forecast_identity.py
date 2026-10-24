"""Resolve the plotted series and its published counterpart without guessing identity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "arima": "SARIMAX",
    "prophet": "Prophet",
    "ensemble": "Ensemble",
    "baseline": "Seasonal baseline",
}


@dataclass(frozen=True)
class ForecastIdentity:
    column: str
    metric_key: str | None
    label: str
    published_label: str
    status: str
    note: str


def resolve_forecast_identity(
    payload: dict | None, requested: str = "published"
) -> ForecastIdentity | None:
    """Never attach another model's name or errors to the selected values.

    The scoring job writes its published default to predicted_demand_mw;
    primary_model identifies the model unless served_series marks a baseline.
    Explicit model columns remain comparisons when a different series is default.
    """
    if not isinstance(payload, dict):
        return None
    rows = payload.get("forecasts") or []
    if not rows or not isinstance(rows[0], dict):
        return None
    baseline = payload.get("served_series") == "seasonal-naive"
    primary = "baseline" if baseline else payload.get("primary_model")
    primary = primary if isinstance(primary, str) and primary in MODEL_LABELS else None
    published_label = MODEL_LABELS.get(primary, "Published forecast (source unspecified)")
    first = rows[0]
    if requested == "published" or (requested == "xgboost" and requested not in first):
        if "predicted_demand_mw" not in first:
            return None
        column, metric_key = "predicted_demand_mw", primary
        label, status = published_label, "Published default"
    elif requested in MODEL_LABELS and requested in first:
        column, metric_key = requested, requested
        label = MODEL_LABELS[requested]
        status = "Published default" if requested == primary else "Model comparison"
    else:
        return None
    if metric_key == "baseline":
        note = "Repeats the last observed day's demand by clock hour; this is not a trained model."
    elif status == "Model comparison":
        note = f"This chart and its peak statistics show {label} for comparison. The published default is {published_label}."
    elif metric_key is None:
        note = "The published values do not identify their source. No model-specific error is attributed to them."
    else:
        note = f"This chart and its peak statistics show the published {label} forecast."
    if requested not in ("published", metric_key) and column == "predicted_demand_mw":
        note = (
            f"Requested {MODEL_LABELS.get(requested, requested)} is unavailable; showing {label}. "
            + note
        )
    return ForecastIdentity(column, metric_key, label, published_label, status, note)


def selection_values(
    payload: dict, identity: ForecastIdentity, hours: int
) -> tuple[list, list[float]] | None:
    """Return only complete, finite values from the selected column and allowed lead."""
    rows = (payload.get("forecasts") or [])[:hours]
    if len(rows) < hours:
        return None
    guard_map = payload.get("horizon_guard")
    guard = guard_map.get(identity.metric_key) if isinstance(guard_map, dict) else None
    max_ok = guard.get("max_ok_horizon") if isinstance(guard, dict) else None
    if isinstance(max_ok, (int, float)) and isfinite(max_ok) and hours > max_ok:
        return None
    try:
        values = [float(row[identity.column]) for row in rows]
        timestamps = [row["timestamp"] for row in rows]
        for ts in timestamps:
            datetime.fromisoformat(ts)
        if not all(isfinite(v) for v in values):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return timestamps, values
