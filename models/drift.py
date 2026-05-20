"""Model drift measurement against live actuals.

Holdout MAPE — the basis for the inverse-MAPE ensemble weights — is
computed once per day during the training job. Between trainings,
individual models can drift relative to live actuals while the
ensemble weights silently treat them as if they hadn't. The PJM
walkthrough on 2026-05-19 surfaced exactly this symptom: four
forecasts spanning a 47 GW range at the same horizon, indicating
at least one model was performing materially worse on live actuals
than its frozen training-time MAPE suggested.

This module computes a continuous **1-hour-ahead** drift signal:

* At each hourly scoring tick the previous tick's forecast for the
  *current* hour has a knowable actual. We compute the per-model
  absolute percent error for that single point.
* Records accumulate in Redis under ``gridpulse:drift:{region}`` as
  a rolling window per model (default 30 days = 720 hourly points).
* Rolling 7-day and 30-day MAPE are computed from the records and
  surfaced alongside them so downstream UI / alerting can read
  either the headline number or the underlying series.

Longer-horizon drift (e.g. 24h-ahead) would require *snapshotting*
predictions at scoring time and re-evaluating them N hours later.
That's part 2 work; this module ships the 1-hour-ahead path first
because the existing forecast key already contains the prediction
we need — no extra storage.

The scope deliberately stops short of the ensemble-weight update
loop. That's a separate decision the next part will resolve:
incorporate live MAPE into the weights, or surface a stale-weights
warning when holdout-vs-live diverges past a threshold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

# Default rolling-window depth. 30 days × 24 hours = 720 hourly records
# per model. Each record is ~80 bytes of JSON → ~57 KB per model × 4
# models = ~230 KB per region × 51 regions ≈ 12 MB total in Redis.
# Well within Memorystore capacity.
DEFAULT_MAX_RECORDS = 720
WINDOW_7D_HOURS = 24 * 7
WINDOW_30D_HOURS = 24 * 30


@dataclass(frozen=True)
class DriftRecord:
    """One (predicted, actual) observation for a single (model, region, hour).

    Stored under ``gridpulse:drift:{region}.models.{model}.records``. Frozen
    so accidental mutation can't corrupt the rolling window.
    """

    timestamp: str  # ISO-8601 UTC, the *target* hour the prediction was for
    predicted: float
    actual: float
    abs_pct_error: float  # |actual - predicted| / |actual| * 100, NaN-safe


def absolute_pct_error(predicted: float, actual: float) -> float | None:
    """MAPE-component for one observation, returns ``None`` when undefined.

    Returns ``None`` when:
    - actual is zero (division by zero — happens when EIA publishes a
      sentinel zero for a missing observation; we'd rather drop the
      record than report a degenerate error)
    - either value is NaN / non-finite
    """
    if not (np.isfinite(predicted) and np.isfinite(actual)):
        return None
    if actual == 0:
        return None
    return abs(actual - predicted) / abs(actual) * 100.0


def mape_over_records(records: list[DriftRecord]) -> float | None:
    """Mean Absolute Percent Error across a list of drift records.

    Records carry their own pre-computed ``abs_pct_error`` so this is a
    simple mean. Returns ``None`` on empty input rather than 0.0 so
    callers can distinguish "no data yet" from "model is perfect".
    """
    if not records:
        return None
    errors = [r.abs_pct_error for r in records if np.isfinite(r.abs_pct_error)]
    if not errors:
        return None
    return float(np.mean(errors))


def rolling_mape(
    records: list[DriftRecord],
    window_hours: int,
    *,
    now_iso: str | None = None,
) -> float | None:
    """MAPE over the most recent ``window_hours`` of records.

    Records older than ``window_hours`` ago (as measured from ``now_iso``,
    defaulting to ``datetime.now(UTC)``) are excluded. Returns ``None``
    when no records fall within the window.
    """
    if not records:
        return None
    now = datetime.fromisoformat(now_iso) if now_iso is not None else datetime.now(UTC)
    cutoff = now - timedelta(hours=window_hours)
    in_window = [r for r in records if datetime.fromisoformat(r.timestamp) >= cutoff]
    return mape_over_records(in_window)


def extract_one_hour_ahead_predictions(
    previous_forecast: dict[str, Any] | None,
    target_timestamp_iso: str,
) -> dict[str, float]:
    """Pull each model's 1-hour-ahead prediction for ``target_timestamp_iso``.

    The previous-tick forecast in Redis has the shape::

        {
            "region": ...,
            "scored_at": ...,
            "forecasts": [
                {"timestamp": "2026-05-20T15:00:00+00:00",
                 "predicted_demand_mw": 95234.5,
                 "xgboost": 95234.5,
                 "prophet": 96012.0,
                 "arima": 94800.3,
                 "ensemble": 95212.0},
                ...
            ],
            ...
        }

    We locate the row whose ``timestamp`` matches ``target_timestamp_iso``
    (the hour we now have an actual for) and return a mapping of
    ``model_name -> predicted_mw`` for every model that produced a
    finite value. Returns an empty dict when no matching row exists or
    the payload is missing/malformed.
    """
    if not previous_forecast:
        return {}
    rows = previous_forecast.get("forecasts") or []
    target = _normalize_ts(target_timestamp_iso)
    for row in rows:
        if _normalize_ts(row.get("timestamp", "")) != target:
            continue
        preds: dict[str, float] = {}
        for key, val in row.items():
            if key in ("timestamp", "predicted_demand_mw"):
                continue
            # Skip non-numeric keys (e.g. a future metadata field). A
            # row contains exactly the model names + the two shared
            # keys above.
            try:
                f = float(val)
            except (TypeError, ValueError):
                continue
            if np.isfinite(f):
                preds[key] = f
        return preds
    return {}


def _normalize_ts(ts: str) -> str:
    """Normalize ISO-8601 timestamps for comparison.

    Tolerates ``Z`` suffix vs explicit ``+00:00``, and stray whitespace.
    Returns the input unchanged on parse failure so the comparison
    upstream just fails to match (rather than crashing).
    """
    if not ts:
        return ts
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return ts.strip()


def build_records_from_actuals(
    previous_forecast: dict[str, Any] | None,
    actuals: dict[str, float],
) -> dict[str, DriftRecord]:
    """Pair the previous tick's predictions with the now-known actuals.

    Args:
        previous_forecast: The Redis payload at ``gridpulse:forecast:{region}:1h``
            *before* this tick overwrites it. ``None`` for first-run regions.
        actuals: Mapping of ``timestamp_iso -> actual_mw`` covering hours
            the actual for which is now available (typically the just-fetched
            recent EIA window).

    Returns:
        Mapping of ``model_name -> DriftRecord`` for the most recent hour
        we have both a prediction and an actual for. Empty dict when no
        such hour exists (e.g. first-ever scoring tick, or the previous
        forecast's earliest hour is already in the future).

    The "most recent matchable hour" choice deliberately avoids creating
    N records for N matchable hours: at hourly cadence each tick should
    add exactly one observation per model. Backfilling more would happen
    on first-deploy and could spike record counts oddly.
    """
    if not previous_forecast or not actuals:
        return {}

    # Pick the most recent target timestamp that's in both the previous
    # forecast's row set AND the actuals mapping. Iterate actuals in
    # reverse-chronological order so we get "most recent" cheaply.
    rows = previous_forecast.get("forecasts") or []
    fc_ts = {_normalize_ts(r.get("timestamp", "")): r for r in rows}
    ordered_actuals = sorted(actuals.items(), reverse=True)
    for ts_iso, actual_mw in ordered_actuals:
        ts_norm = _normalize_ts(ts_iso)
        if ts_norm not in fc_ts:
            continue
        if not np.isfinite(actual_mw):
            continue
        preds = extract_one_hour_ahead_predictions(previous_forecast, ts_norm)
        out: dict[str, DriftRecord] = {}
        for model_name, predicted_mw in preds.items():
            err = absolute_pct_error(predicted_mw, actual_mw)
            if err is None:
                continue
            out[model_name] = DriftRecord(
                timestamp=ts_norm,
                predicted=predicted_mw,
                actual=float(actual_mw),
                abs_pct_error=err,
            )
        return out
    return {}


def merge_and_trim(
    existing_records: list[DriftRecord],
    new_record: DriftRecord | None,
    *,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> list[DriftRecord]:
    """Append ``new_record`` to ``existing_records`` and enforce window size.

    De-duplicates by timestamp — if the same target hour is already in
    the existing records (which happens when a scoring tick re-runs
    against the same actuals, e.g. backfill / replay), the new record
    replaces the old one. Otherwise appended and the oldest record
    drops out when the window exceeds ``max_records``.

    Returns a chronologically-sorted list, oldest first.
    """
    if new_record is None:
        return sorted(existing_records, key=lambda r: r.timestamp)

    by_ts: dict[str, DriftRecord] = {r.timestamp: r for r in existing_records}
    by_ts[new_record.timestamp] = new_record
    ordered = sorted(by_ts.values(), key=lambda r: r.timestamp)
    if len(ordered) > max_records:
        ordered = ordered[-max_records:]
    return ordered


def serialize_records(records: list[DriftRecord]) -> list[dict[str, Any]]:
    """Compact dict form for Redis JSON. Keys deliberately short."""
    return [
        {
            "ts": r.timestamp,
            "p": round(r.predicted, 2),
            "a": round(r.actual, 2),
            "e": round(r.abs_pct_error, 4),
        }
        for r in records
    ]


def deserialize_records(rows: list[dict[str, Any]] | None) -> list[DriftRecord]:
    """Inverse of ``serialize_records``. Tolerant of missing rows."""
    if not rows:
        return []
    out: list[DriftRecord] = []
    for row in rows:
        try:
            out.append(
                DriftRecord(
                    timestamp=_normalize_ts(row["ts"]),
                    predicted=float(row["p"]),
                    actual=float(row["a"]),
                    abs_pct_error=float(row["e"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            # A malformed historical row shouldn't poison the whole
            # window. Drop it and continue.
            continue
    return out


def compute_drift_payload(
    region: str,
    existing_payload: dict[str, Any] | None,
    new_records: dict[str, DriftRecord],
    *,
    max_records: int = DEFAULT_MAX_RECORDS,
    now_iso: str | None = None,
) -> dict[str, Any]:
    """Build the Redis payload for ``gridpulse:drift:{region}``.

    Merges ``new_records`` (one per model from the just-processed tick)
    into the existing per-model rolling windows, computes 7d and 30d
    rolling MAPE per model, and returns the full payload ready for
    ``redis_set``.

    The ``models`` sub-mapping uses the same model-name vocabulary as
    the forecast payload (``xgboost`` / ``prophet`` / ``arima`` /
    ``ensemble``). Models without records yet are omitted; the UI
    handles "no data yet" by simply not showing a drift indicator
    for that model.
    """
    existing_models = (existing_payload or {}).get("models") or {}

    # Union of models we have history for + models we have a new record for.
    all_model_names = set(existing_models.keys()) | set(new_records.keys())

    models_out: dict[str, Any] = {}
    for model_name in sorted(all_model_names):
        prior = deserialize_records(existing_models.get(model_name, {}).get("records"))
        new_rec = new_records.get(model_name)
        merged = merge_and_trim(prior, new_rec, max_records=max_records)

        models_out[model_name] = {
            "rolling_mape_7d": rolling_mape(merged, WINDOW_7D_HOURS, now_iso=now_iso),
            "rolling_mape_30d": rolling_mape(merged, WINDOW_30D_HOURS, now_iso=now_iso),
            "n_records": len(merged),
            "records": serialize_records(merged),
        }

    return {
        "region": region,
        "last_updated_at": now_iso or datetime.now(UTC).isoformat(),
        "models": models_out,
    }


# Convenience helper kept here so ``jobs/phases.py`` can stay thin and
# focused on Redis IO rather than re-importing dataclass machinery.
def record_to_dict(r: DriftRecord) -> dict[str, Any]:
    """Public-friendly dict (longer keys than ``serialize_records``)."""
    return asdict(r)
