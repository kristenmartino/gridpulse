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

Longer-horizon drift (24h / 48h / 72h-ahead) *snapshots* predictions at
scoring time and re-evaluates them N hours later against the now-known
actual — the "part 2" path, added for #227 (see
``compute_horizon_drift_payload`` and ``gridpulse:drift_horizon:{region}``).
It grades each horizon against its OWN ``MAPE_BY_HORIZON`` band, so a model
built for day-ahead isn't condemned by the 1-hour number below.

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

# Region-relative low-actual filter (#142 / PR-G9). A drift record whose
# actual demand is below this fraction of the rolling-window *median* actual
# is treated as a data-quality artifact, not model error, and excluded from
# the rolling aggregates. Scale-relative on purpose: a 50 MW record next to a
# 2.5 GW median is an EIA reporting glitch for a large BA, but 50 MW is a
# legitimate load for a tiny one — a universal MW floor can't tell them apart,
# a fraction-of-median can. 0.10 = "ignore actuals under 10% of typical."
LOW_ACTUAL_FRACTION = 0.10


@dataclass(frozen=True)
class DriftRecord:
    """One (predicted, actual) observation for a single (model, region, hour).

    Stored under ``gridpulse:drift:{region}.models.{model}.records``. Frozen
    so accidental mutation can't corrupt the rolling window.

    ``smape`` is the symmetric MAPE component (bounded [0, 200] per record),
    the robust companion to ``abs_pct_error`` that does not explode when the
    actual is near zero. It is auto-derived from ``predicted``/``actual`` when
    not supplied (default sentinel ``NaN``), so historical records that
    predate the field — and bare ``DriftRecord(ts, p, a, e)`` constructions —
    get a correct value without any caller change.
    """

    timestamp: str  # ISO-8601 UTC, the *target* hour the prediction was for
    predicted: float
    actual: float
    abs_pct_error: float  # |actual - predicted| / |actual| * 100, NaN-safe
    smape: float = float("nan")  # 200*|a-p|/(|a|+|p|); auto-filled below

    def __post_init__(self) -> None:
        # Frozen dataclass: assign through object.__setattr__. Only fill when
        # the caller didn't provide a finite sMAPE (the common path for legacy
        # records and the test helpers).
        if not np.isfinite(self.smape):
            s = symmetric_pct_error(self.predicted, self.actual)
            object.__setattr__(self, "smape", s if s is not None else float("nan"))


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


def symmetric_pct_error(predicted: float, actual: float) -> float | None:
    """Symmetric MAPE component for one observation, in percent.

    ``sMAPE = 200 * |actual - predicted| / (|actual| + |predicted|)``, bounded
    to ``[0, 200]`` per record. Unlike :func:`absolute_pct_error` it does not
    divide by the actual alone, so a near-zero actual against a normal-scale
    prediction yields ~200% (a flagged-but-bounded miss) instead of a
    thousands-of-percent spike that dominates the rolling mean. This is the
    headline drift metric surfaced to the UI (#142 / PR-G9).

    Returns ``None`` when:
    - either value is NaN / non-finite
    - both values are zero (denominator zero — degenerate, no signal)
    """
    if not (np.isfinite(predicted) and np.isfinite(actual)):
        return None
    denom = abs(actual) + abs(predicted)
    if denom == 0:
        return None
    return 200.0 * abs(actual - predicted) / denom


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


def smape_over_records(records: list[DriftRecord]) -> float | None:
    """Mean symmetric MAPE across a list of drift records.

    Mirror of :func:`mape_over_records` over the bounded ``smape`` component.
    Returns ``None`` on empty input or when every record's sMAPE is
    non-finite, so "no data yet" stays distinguishable from "perfect".
    """
    if not records:
        return None
    vals = [r.smape for r in records if np.isfinite(r.smape)]
    if not vals:
        return None
    return float(np.mean(vals))


def _within_window(
    records: list[DriftRecord],
    window_hours: int,
    *,
    now_iso: str | None = None,
) -> list[DriftRecord]:
    """Records whose timestamp is within ``window_hours`` of ``now_iso``.

    ``now_iso`` defaults to ``datetime.now(UTC)``. Returns ``[]`` for empty
    input or when nothing falls inside the window.
    """
    if not records:
        return []
    now = datetime.fromisoformat(now_iso) if now_iso is not None else datetime.now(UTC)
    cutoff = now - timedelta(hours=window_hours)
    return [r for r in records if datetime.fromisoformat(r.timestamp) >= cutoff]


def filter_low_actuals(
    records: list[DriftRecord],
    min_fraction: float = LOW_ACTUAL_FRACTION,
) -> tuple[list[DriftRecord], int]:
    """Drop records whose ``|actual|`` is a near-zero outlier for the window.

    The threshold is ``min_fraction × median(|actual|)`` over the supplied
    records — region-relative, so it adapts to each balancing authority's own
    scale rather than imposing a universal MW floor (see #142: LDWP's sporadic
    ~50 MW reporting artifacts vs its ~2.5 GW median). Returns
    ``(kept_records, n_dropped)``.

    No-ops (returns everything, ``n_dropped=0``) when ``min_fraction <= 0``,
    the window is empty, or the median is non-positive — i.e. it never removes
    data unless there is a well-defined positive scale to be an outlier
    *relative to*.
    """
    if not records or min_fraction <= 0:
        return list(records), 0
    actuals = [abs(r.actual) for r in records if np.isfinite(r.actual)]
    if not actuals:
        return list(records), 0
    median = float(np.median(actuals))
    if median <= 0:
        return list(records), 0
    threshold = min_fraction * median
    kept = [r for r in records if np.isfinite(r.actual) and abs(r.actual) >= threshold]
    return kept, len(records) - len(kept)


def rolling_mape(
    records: list[DriftRecord],
    window_hours: int,
    *,
    now_iso: str | None = None,
    min_actual_fraction: float = 0.0,
) -> float | None:
    """MAPE over the most recent ``window_hours`` of records.

    Records older than ``window_hours`` ago (as measured from ``now_iso``,
    defaulting to ``datetime.now(UTC)``) are excluded. Returns ``None``
    when no records fall within the window.

    ``min_actual_fraction`` (default ``0.0`` = off, preserving the raw
    diagnostic behaviour) optionally applies the region-relative low-actual
    filter before averaging — the scoring job passes ``LOW_ACTUAL_FRACTION``
    so the persisted headline MAPE is robust to near-zero-actual artifacts.
    """
    in_window = _within_window(records, window_hours, now_iso=now_iso)
    kept, _ = filter_low_actuals(in_window, min_actual_fraction)
    return mape_over_records(kept)


def rolling_smape(
    records: list[DriftRecord],
    window_hours: int,
    *,
    now_iso: str | None = None,
    min_actual_fraction: float = LOW_ACTUAL_FRACTION,
) -> float | None:
    """Symmetric MAPE over the most recent ``window_hours`` of records.

    The bounded, robust counterpart to :func:`rolling_mape`. Applies the
    low-actual filter by default (belt-and-suspenders alongside sMAPE's own
    boundedness) so a handful of near-zero hours can neither explode the mean
    nor each pin it at ~200%.
    """
    in_window = _within_window(records, window_hours, now_iso=now_iso)
    kept, _ = filter_low_actuals(in_window, min_actual_fraction)
    return smape_over_records(kept)


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
    """Compact dict form for Redis JSON. Keys deliberately short.

    ``s`` (sMAPE) is written when finite and omitted otherwise — a missing
    ``s`` is recomputed from ``p``/``a`` on deserialize, keeping old payloads
    readable and the JSON free of non-finite values.
    """
    return [
        {
            "ts": r.timestamp,
            "p": round(r.predicted, 2),
            "a": round(r.actual, 2),
            "e": round(r.abs_pct_error, 4),
            **({"s": round(r.smape, 4)} if np.isfinite(r.smape) else {}),
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
            s_raw = row.get("s")
            out.append(
                DriftRecord(
                    timestamp=_normalize_ts(row["ts"]),
                    predicted=float(row["p"]),
                    actual=float(row["a"]),
                    abs_pct_error=float(row["e"]),
                    # Missing/None ``s`` (pre-PR-G9 records) → NaN → the
                    # dataclass recomputes sMAPE from p/a on construction.
                    smape=float(s_raw) if s_raw is not None else float("nan"),
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

        # Per-window sample counts (P2-21/#273): the number of records that
        # actually feed each rolling mean, post low-actual filter. ``n_records``
        # is the TOTAL retained history (trimmed by count, not age — it can be
        # dominated by records far older than either window), so consumers
        # gating "is this 7d figure statistically defensible?" must use
        # ``n_7d``, never ``n_records``. n_low_excl_7d stays as the
        # transparency signal for how many the filter dropped.
        kept_7d, n_low_excl_7d = filter_low_actuals(
            _within_window(merged, WINDOW_7D_HOURS, now_iso=now_iso)
        )
        kept_30d, _ = filter_low_actuals(_within_window(merged, WINDOW_30D_HOURS, now_iso=now_iso))

        models_out[model_name] = {
            # sMAPE is the headline drift metric (bounded, near-zero-robust).
            "rolling_smape_7d": rolling_smape(merged, WINDOW_7D_HOURS, now_iso=now_iso),
            "rolling_smape_30d": rolling_smape(merged, WINDOW_30D_HOURS, now_iso=now_iso),
            # MAPE kept for diagnostics / holdout comparison, now filtered with
            # the same region-relative rule so a stray near-zero hour can no
            # longer pin it at 200%+ (#142).
            "rolling_mape_7d": rolling_mape(
                merged, WINDOW_7D_HOURS, now_iso=now_iso, min_actual_fraction=LOW_ACTUAL_FRACTION
            ),
            "rolling_mape_30d": rolling_mape(
                merged, WINDOW_30D_HOURS, now_iso=now_iso, min_actual_fraction=LOW_ACTUAL_FRACTION
            ),
            "n_records": len(merged),
            "n_7d": len(kept_7d),
            "n_30d": len(kept_30d),
            "n_low_actual_excluded_7d": n_low_excl_7d,
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


# ── #227: horizon-matched drift ──────────────────────────────────────────────
# The 1-hour-ahead signal above structurally penalizes Prophet/SARIMAX, which
# have no last-value anchor and only earn their keep at multi-day horizons. This
# is the "part 2" the module docstring defers: snapshot each model's forward
# forecast at scoring time, re-score it against the now-known actual N hours
# later, and grade each horizon against ITS OWN ``MAPE_BY_HORIZON`` band — so a
# competent day-ahead model is no longer condemned by a 1h number it was never
# built to win.
HORIZON_DRIFT_HORIZONS: tuple[str, ...] = ("24h", "48h", "72h")
_HORIZON_HOURS: dict[str, int] = {"24h": 24, "48h": 48, "72h": 72}
# A snapshot whose target hour is older than the longest horizon + this slack and
# still hasn't resolved (BA went dark, actual never published) is dropped so the
# pending buffer can't grow unbounded.
PENDING_STALE_HOURS = 72 + 48


def snapshot_horizon_predictions(
    forecast_payload: dict[str, Any] | None,
    horizons: tuple[str, ...] = HORIZON_DRIFT_HORIZONS,
) -> list[dict[str, Any]]:
    """Snapshot a forecast's per-model prediction at each horizon.

    A forecast made at ``scored_at`` predicts the hour ``scored_at + H`` at
    horizon ``H``. Returns one pending-snapshot dict per horizon that has a
    matching forecast row::

        {"target_ts": iso, "made_at": iso, "horizon": "24h",
         "preds": {"xgboost": mw, "prophet": mw, "arima": mw, "ensemble": mw}}

    Empty list when the payload is missing/malformed or carries no ``scored_at``.
    """
    if not forecast_payload:
        return []
    rows = forecast_payload.get("forecasts") or []
    if not rows:
        return []
    # Horizon is measured from the forecast's FIRST row (hour-aligned), NOT the
    # wall-clock ``scored_at``: ``scored_at`` is ``datetime.now()`` with sub-hour
    # precision (e.g. 14:37:22) and would never exact-match an on-the-hour
    # forecast row, silently producing zero snapshots.
    origin_iso = _normalize_ts(rows[0].get("timestamp", ""))
    made_at = _normalize_ts(forecast_payload.get("scored_at", "")) or origin_iso
    try:
        origin = datetime.fromisoformat(origin_iso)
    except (ValueError, TypeError):
        return []
    out: list[dict[str, Any]] = []
    for horizon in horizons:
        hours = _HORIZON_HOURS.get(horizon)
        if hours is None:
            continue
        target = _normalize_ts((origin + timedelta(hours=hours)).isoformat())
        preds = extract_one_hour_ahead_predictions(forecast_payload, target)
        if preds:
            out.append(
                {"target_ts": target, "made_at": made_at, "horizon": horizon, "preds": preds}
            )
    return out


def resolve_horizon_snapshots(
    pending: list[dict[str, Any]],
    actuals: dict[str, float],
) -> tuple[list[tuple[str, str, DriftRecord]], list[dict[str, Any]]]:
    """Resolve pending snapshots whose target hour now has a finite actual.

    Returns ``(resolved, still_pending)`` where ``resolved`` is a list of
    ``(model_name, horizon, DriftRecord)`` triples ready to append to the
    per-(model, horizon) rolling windows.
    """
    lookup = {_normalize_ts(k): v for k, v in actuals.items()}
    resolved: list[tuple[str, str, DriftRecord]] = []
    still: list[dict[str, Any]] = []
    for snap in pending:
        target = _normalize_ts(snap.get("target_ts", ""))
        actual = lookup.get(target)
        if actual is None or not np.isfinite(actual) or actual <= 0:
            still.append(snap)
            continue
        horizon = snap.get("horizon", "")
        for model_name, predicted in (snap.get("preds") or {}).items():
            err = absolute_pct_error(predicted, actual)
            if err is None:
                continue
            resolved.append(
                (
                    model_name,
                    horizon,
                    DriftRecord(
                        timestamp=target,
                        predicted=float(predicted),
                        actual=float(actual),
                        abs_pct_error=err,
                    ),
                )
            )
    return resolved, still


def _expire_pending(pending: list[dict[str, Any]], now_iso: str | None) -> list[dict[str, Any]]:
    """Drop snapshots whose target hour is older than ``PENDING_STALE_HOURS`` and
    still unresolved — the never-published-actual tail that would otherwise leak."""
    try:
        now = datetime.fromisoformat(now_iso) if now_iso else datetime.now(UTC)
    except (ValueError, TypeError):
        now = datetime.now(UTC)
    cutoff = now - timedelta(hours=PENDING_STALE_HOURS)
    out: list[dict[str, Any]] = []
    for snap in pending:
        try:
            target = datetime.fromisoformat(_normalize_ts(snap.get("target_ts", "")))
        except (ValueError, TypeError):
            continue  # malformed → drop
        if target >= cutoff:
            out.append(snap)
    return out


def _horizon_rollup_block(
    merged: list[DriftRecord], now_iso: str | None, grade_horizon: str
) -> dict[str, Any]:
    """One (model, horizon) rollup block. Mirrors ``compute_drift_payload``'s
    per-model block (windowed sMAPE/MAPE + the region-relative low-actual filter)
    and adds a horizon-matched ``grade`` from ``config.mape_grade`` — the point of
    #227: judge each horizon against its OWN band, not the 1h band."""
    from config import mape_grade

    kept_7d, n_low_excl_7d = filter_low_actuals(
        _within_window(merged, WINDOW_7D_HOURS, now_iso=now_iso)
    )
    kept_30d, _ = filter_low_actuals(_within_window(merged, WINDOW_30D_HOURS, now_iso=now_iso))
    mape_7d = rolling_mape(
        merged, WINDOW_7D_HOURS, now_iso=now_iso, min_actual_fraction=LOW_ACTUAL_FRACTION
    )
    return {
        "rolling_smape_7d": rolling_smape(merged, WINDOW_7D_HOURS, now_iso=now_iso),
        "rolling_smape_30d": rolling_smape(merged, WINDOW_30D_HOURS, now_iso=now_iso),
        "rolling_mape_7d": mape_7d,
        "rolling_mape_30d": rolling_mape(
            merged, WINDOW_30D_HOURS, now_iso=now_iso, min_actual_fraction=LOW_ACTUAL_FRACTION
        ),
        "n_records": len(merged),
        # P2-21 (#273): per-window post-filter sample counts — the honest
        # denominators behind the 7d/30d means (n_records is total history).
        "n_7d": len(kept_7d),
        "n_30d": len(kept_30d),
        "n_low_actual_excluded_7d": n_low_excl_7d,
        "grade": mape_grade(mape_7d, horizon=grade_horizon) if mape_7d is not None else None,
        "records": serialize_records(merged),
    }


def compute_horizon_drift_payload(
    region: str,
    existing_payload: dict[str, Any] | None,
    forecast_payload: dict[str, Any] | None,
    actuals: dict[str, float],
    *,
    horizons: tuple[str, ...] = HORIZON_DRIFT_HORIZONS,
    max_records: int = DEFAULT_MAX_RECORDS,
    now_iso: str | None = None,
) -> dict[str, Any]:
    """Build the ``gridpulse:drift_horizon:{region}`` payload for one tick.

    Pipeline: (1) resolve pending snapshots whose target hour now has an actual,
    (2) snapshot the current forecast's +H predictions into the pending buffer,
    (3) expire the never-resolved tail, (4) roll up each (model, horizon) rolling
    window with a horizon-matched grade. Shape::

        {"region", "last_updated_at", "horizons": [...],
         "pending": [<snapshot>, ...],
         "models": {"xgboost": {"24h": <block>, "48h": ..., "72h": ...}, ...}}
    """
    existing = existing_payload or {}
    pending = list(existing.get("pending") or [])
    existing_models = existing.get("models") or {}

    # 1. Resolve pending against the now-known actuals.
    resolved, pending = resolve_horizon_snapshots(pending, actuals)

    # 2. Snapshot the current forecast's per-horizon predictions (dedup on
    #    (target_ts, horizon) so a retried tick can't double-count).
    seen = {(s.get("target_ts"), s.get("horizon")) for s in pending}
    for snap in snapshot_horizon_predictions(forecast_payload, horizons):
        key = (snap["target_ts"], snap["horizon"])
        if key not in seen:
            pending.append(snap)
            seen.add(key)

    # 3. Expire the never-resolved tail.
    pending = _expire_pending(pending, now_iso)

    # 4. Roll up per (model, horizon).
    resolved_by_key: dict[tuple[str, str], list[DriftRecord]] = {}
    for model_name, horizon, record in resolved:
        resolved_by_key.setdefault((model_name, horizon), []).append(record)

    all_models = set(existing_models) | {m for m, _, _ in resolved}
    models_out: dict[str, Any] = {}
    for model_name in sorted(all_models):
        existing_hz = existing_models.get(model_name) or {}
        block_by_horizon: dict[str, Any] = {}
        for horizon in horizons:
            prior = deserialize_records((existing_hz.get(horizon) or {}).get("records"))
            merged = prior
            for record in resolved_by_key.get((model_name, horizon), []):
                merged = merge_and_trim(merged, record, max_records=max_records)
            block_by_horizon[horizon] = _horizon_rollup_block(merged, now_iso, horizon)
        models_out[model_name] = block_by_horizon

    return {
        "region": region,
        "last_updated_at": now_iso or datetime.now(UTC).isoformat(),
        "horizons": list(horizons),
        "pending": pending,
        "models": models_out,
    }
