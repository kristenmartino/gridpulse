"""Hourly observations of cached forecasts, independent of their moving origin.

These samples are separate from legacy forecast-start-offset diagnostics. A
sample taken at 10:05 selects target 11:00 + H, so its observation-to-target
lead is H..H+1 hours. This is not the forecast's original publication lead.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import isfinite

from data.forecast_evidence import utc_timestamp
from data.forecast_identity import MODEL_LABELS, resolve_forecast_identity
from models.drift import DriftRecord, absolute_pct_error, filter_low_actuals, mape_over_records

HORIZONS = (24, 48, 72)
WINDOW_HOURS = 168
BASIS = "hourly_observation_before_target"
SERIES = tuple(MODEL_LABELS)


def _time(value: object) -> datetime | None:
    normalized = utc_timestamp(value)
    return datetime.fromisoformat(normalized) if normalized else None


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _ceiling_hour(value: datetime) -> datetime:
    floor = value.replace(minute=0, second=0, microsecond=0)
    return floor if value == floor else floor + timedelta(hours=1)


def _sample(payload: dict | None, observed: datetime, horizon: int) -> dict:
    target = _ceiling_hour(observed) + timedelta(hours=horizon)
    payload = payload if isinstance(payload, dict) else {}
    generated = _time(payload.get("scored_at"))
    rows = payload.get("forecasts") or []
    identity = resolve_forecast_identity(payload)
    selected = next(
        ((i, r) for i, r in enumerate(rows) if _time(r.get("timestamp")) == target), None
    )
    preds = {}
    reason = None
    # A future generation clock contradicts an observation taken now. A missing
    # generation clock stays unknown; observing the values is still possible.
    if generated and generated > observed:
        reason = "future_generation_clock"
    elif selected is None:
        reason = "target_not_in_cached_forecast" if rows else "forecast_unavailable"
    else:
        index, row = selected
        guards = payload.get("horizon_guard") or {}
        for name in SERIES:
            column = identity.column if identity and name == identity.metric_key else name
            value = row.get(column)
            limit = (guards.get(name) or {}).get("max_ok_horizon")
            if _finite(limit) and index + 1 > limit:
                continue
            if _finite(value):
                preds[name] = float(value)
        if not preds:
            reason = "predictions_unavailable_or_withheld"
    return {
        "target_ts": target.isoformat(),
        "horizon": f"{horizon}h",
        "sampled_at": observed.isoformat(),
        "sample_lead_hours": (target - observed).total_seconds() / 3600,
        "generated_at": generated.isoformat() if generated else None,
        "forecast_start_at": utc_timestamp(rows[0].get("timestamp")) if rows else None,
        "publication_id": payload.get("publication_id"),
        "published_source": identity.metric_key if identity else None,
        "predictions": preds,
        "unavailable_reason": reason,
        "actual": None,
        "actual_first_observed_at": None,
        "actual_latest_observed_at": None,
    }


def _rollup(samples: list[dict], model: str, horizon: int, now: datetime, started: datetime):
    label = f"{horizon}h"
    first_target = _ceiling_hour(started) + timedelta(hours=horizon)
    end = now.replace(minute=0, second=0, microsecond=0)
    targets = [end - timedelta(hours=i) for i in range(WINDOW_HOURS)]
    by_target = {_time(s["target_ts"]): s for s in samples if s["horizon"] == label}
    records = []
    for target in targets:
        sample = by_target.get(target)
        if not sample:
            continue
        predicted = sample["predictions"].get(model)
        actual = sample.get("actual")
        if _finite(predicted) and _finite(actual) and actual > 0:
            error = absolute_pct_error(predicted, actual)
            if error is not None:
                records.append(DriftRecord(target.isoformat(), predicted, actual, error))
    kept, _ = filter_low_actuals(records)
    scored = {r.timestamp for r in kept}
    counts = dict.fromkeys(
        (
            "scored_target_hours",
            "not_yet_tracked_hours",
            "missing_sample_hours",
            "series_unavailable_hours",
            "pending_actual_hours",
            "low_actual_excluded_hours",
        ),
        0,
    )
    for target in targets:
        sample = by_target.get(target)
        if target < first_target:
            status = "not_yet_tracked_hours"
        elif sample is None:
            status = "missing_sample_hours"
        elif model not in sample["predictions"]:
            status = "series_unavailable_hours"
        elif sample.get("actual") is None:
            status = "pending_actual_hours"
        elif target.isoformat() in scored:
            status = "scored_target_hours"
        else:
            status = "low_actual_excluded_hours"
        counts[status] += 1
    future = [s for t, s in by_target.items() if t > now and model in s["predictions"]]
    leads = [by_target[_time(r.timestamp)]["sample_lead_hours"] for r in kept]
    return {
        "rolling_mape_7d": mape_over_records(kept),
        "n_7d": len(kept),
        "first_target_at": first_target.isoformat(),
        "future_sample_hours": len(future),
        "sample_lead_min_hours": min(leads) if leads else None,
        "sample_lead_max_hours": max(leads) if leads else None,
        "coverage_7d": {
            "as_of": now.isoformat(),
            "window_start": (now - timedelta(hours=WINDOW_HOURS)).isoformat(),
            "window_end": now.isoformat(),
            "possible_target_hours": WINDOW_HOURS,
            **counts,
        },
    }


def compute_hourly_scoring(
    region: str,
    previous: dict | None,
    forecast: dict | None,
    actuals: dict[str, float],
    *,
    now_iso: str | None = None,
) -> dict:
    """Take one immutable hourly sample and reconcile every retained actual.

    First sample wins on retries. Predictions are never synthesized for a past
    sampling slot. Unresolved samples remain while their target is inside the
    seven-day reporting window, including actuals arriving after 120 hours.
    """
    now = _time(now_iso) if now_iso is not None else datetime.now(UTC)
    if now is None:
        raise ValueError("Hourly scoring requires an aware observation clock")
    old = previous if isinstance(previous, dict) else {}
    if previous is not None and (
        not isinstance(previous, dict)
        or old.get("schema_version") != 1
        or old.get("sampling_basis") != BASIS
        or old.get("region") != region
    ):
        raise ValueError("Refusing to mix hourly observations with another scoring basis")
    prior_clock = _time(old.get("last_updated_at"))
    if prior_clock and now < prior_clock:
        raise ValueError("Hourly scoring observation clock moved backwards")
    started = _time(old.get("capture_started_at")) or now
    if old and (not prior_clock or not _time(old.get("capture_started_at"))):
        raise ValueError("Hourly scoring history has an invalid collection clock")
    if old and not isinstance(old.get("samples"), list):
        raise ValueError("Hourly scoring history is missing its sample ledger")
    cutoff = now - timedelta(hours=WINDOW_HOURS)
    samples = {}
    for item in old.get("samples", []):
        target = _time(item.get("target_ts"))
        sampled_at = _time(item.get("sampled_at"))
        if (
            not target
            or not sampled_at
            or item.get("horizon") not in ("24h", "48h", "72h")
            or target != _ceiling_hour(sampled_at) + timedelta(hours=int(item["horizon"][:-1]))
            or not isinstance(item.get("predictions"), dict)
        ):
            raise ValueError("Hourly scoring history contains a malformed sample")
        if target and target > cutoff:
            # Copy each record and its values: computing a new state cannot
            # mutate a cached prior snapshot or a test's source fixture.
            sample = dict(item, predictions=dict(item["predictions"]))
            samples[(sample["target_ts"], sample["horizon"])] = sample
    for horizon in HORIZONS:
        sample = _sample(forecast, now, horizon)
        samples.setdefault((sample["target_ts"], sample["horizon"]), sample)
    observed_actuals = {
        stamp: float(value)
        for key, value in actuals.items()
        if (stamp := utc_timestamp(key)) and _finite(value) and value > 0
    }
    for sample in samples.values():
        # Future rows in an upstream frame are not completed actuals.
        if _time(sample["target_ts"]) > now:
            continue
        value = observed_actuals.get(sample["target_ts"])
        if value is not None:
            if sample.get("actual") is None:
                sample["actual_first_observed_at"] = now.isoformat()
                sample["actual_first_value"] = value
            sample["actual"] = value
            sample["actual_latest_observed_at"] = now.isoformat()
    ordered = sorted(samples.values(), key=lambda s: (s["target_ts"], s["horizon"]))
    return {
        "schema_version": 1,
        "region": region,
        "sampling_basis": BASIS,
        "capture_started_at": started.isoformat(),
        "last_updated_at": now.isoformat(),
        "horizons": [f"{h}h" for h in HORIZONS],
        "samples": ordered,
        "models": {
            model: {f"{h}h": _rollup(ordered, model, h, now, started) for h in HORIZONS}
            for model in SERIES
        },
    }


def export_hourly_scoring(payload: dict) -> dict:
    """Explicit public contract; sample values and internal clocks stay private."""
    fields = (
        "rolling_mape_7d",
        "n_7d",
        "first_target_at",
        "future_sample_hours",
        "sample_lead_min_hours",
        "sample_lead_max_hours",
    )
    coverage_fields = (
        "as_of",
        "window_start",
        "window_end",
        "possible_target_hours",
        "scored_target_hours",
        "not_yet_tracked_hours",
        "missing_sample_hours",
        "series_unavailable_hours",
        "pending_actual_hours",
        "low_actual_excluded_hours",
    )
    return {
        **{
            k: payload.get(k)
            for k in (
                "schema_version",
                "region",
                "sampling_basis",
                "capture_started_at",
                "last_updated_at",
                "horizons",
            )
        },
        "models": {
            model: {
                f"{h}h": {
                    **{k: block.get(k) for k in fields},
                    "coverage_7d": {
                        k: block.get("coverage_7d", {}).get(k) for k in coverage_fields
                    },
                }
                for h in HORIZONS
                if isinstance((block := by_horizon.get(f"{h}h")), dict)
            }
            for model, by_horizon in payload.get("models", {}).items()
            if model in SERIES and isinstance(by_horizon, dict)
        },
        "notes": [
            "Separate hourly observations: H..H+1 hours from sampling to target, not original publication lead.",
            "First observation for a target and horizon wins; no retrospective forecast reconstruction.",
            "Not-yet-tracked hours, missing samples, unavailable series, pending actuals and exclusions partition coverage.",
            "Actuals are latest observed values and may be revised. Legacy offset diagnostics remain separate.",
        ],
    }
