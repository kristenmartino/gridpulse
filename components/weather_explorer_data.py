"""Read-only, region-keyed datasets for demand/weather and immutable-issue review.

No model, upstream fetch, cache writer or archive reconstruction is called here.
Weather is contextual; the cache cannot reconstruct weather as issued.
"""

from __future__ import annotations

from datetime import UTC, datetime
from math import isfinite

import pandas as pd

from config import REGION_NAMES
from data.forecast_identity import resolve_forecast_identity, selection_values
from data.forecast_publications import MAX_ISSUES, export_publication
from data.hourly_scoring_actuals import hourly_actuals
from data.redis_client import redis_get, redis_key
from data.weather_windows import historical_weather_window

WEATHER_FIELDS = {
    "temperature_f": ("temperature_2m", "Temperature", "°F"),
    "wind_mph": ("wind_speed_80m", "Wind at 80 m", "mph"),
    "solar_wm2": ("shortwave_radiation", "Solar radiation", "W/m²"),
}


def utc(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = pd.Timestamp(value)
        if pd.isna(result):
            return None
        return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")
    except (ValueError, TypeError, OverflowError):
        return None


def iso(value):
    stamp = utc(value)
    return stamp.isoformat() if stamp is not None else None


def number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _weather_frame(payload):
    if not isinstance(payload, dict):
        return pd.DataFrame(columns=["timestamp"])
    stamps = payload.get("timestamps") or []
    result = {"timestamp": stamps}
    for output, (source, _, _) in WEATHER_FIELDS.items():
        values = payload.get(source)
        result[output] = (
            values
            if isinstance(values, list) and len(values) == len(stamps)
            else [None] * len(stamps)
        )
    frame = pd.DataFrame(result)
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    return (
        frame.dropna(subset=["timestamp"])
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
    )


def _actual_rows(payload, now):
    if not isinstance(payload, dict):
        return []
    stamps, values = payload.get("timestamps") or [], payload.get("demand_mw") or []
    excluded = {
        iso(v.get("ts") if isinstance(v, dict) else v) for v in payload.get("artifact_excluded", [])
    }
    out = {}
    for stamp, value in zip(stamps, values, strict=False):
        clock, val = utc(stamp), number(value)
        if clock is not None and clock <= now:
            out[iso(clock)] = {
                "timestamp": iso(clock),
                "demand_mw": val
                if val is not None and val > 0 and iso(clock) not in excluded
                else None,
            }
    return sorted(out.values(), key=lambda r: r["timestamp"])


def _weather_values(row):
    return {key: number(row.get(key)) if row is not None else None for key in WEATHER_FIELDS}


def read_issues(region, reader=redis_get, selected_ids=()):
    """Expose the whole bounded index; read only selected and four recent receipts."""
    index = reader(redis_key(f"publication_index:{region}"))
    if not isinstance(index, list):
        return []
    entries = []
    for item in index[:MAX_ISSUES]:
        key = item.get("publication_id") if isinstance(item, dict) else None
        if (
            isinstance(key, str)
            and len(key) == 64
            and all(c in "0123456789abcdef" for c in key)
            and utc(item.get("generated_at")) is not None
        ):
            entries.append(item)
    wanted = set(selected_ids) | {x["publication_id"] for x in entries[:4]}
    issues = []
    for item in entries:
        key = item["publication_id"]
        summary = {
            "publication_id": key,
            "region": region,
            "series_label": item.get("series_label")
            or item.get("series_source")
            or "Published forecast",
            "timing": {
                k: item.get(k)
                for k in ("generated_at", "forecast_start_at", "target_end_at", "target_hours")
            },
            "observed_published_at": item.get("observed_published_at"),
            "forecast": [],
            "loaded": False,
            "receipt_status": "not_requested",
        }
        if key in wanted:
            summary["receipt_status"] = "unavailable"
            record = reader(redis_key(f"publication:{region}:{key}"))
            if (
                isinstance(record, dict)
                and record.get("publication_id") == key
                and record.get("region") == region
            ):
                summary = {
                    **export_publication(record),
                    "loaded": True,
                    "receipt_status": "available",
                }
        issues.append(summary)
    return sorted(issues, key=lambda r: r["timing"]["generated_at"], reverse=True)


def load_explorer_data(region, *, reader=None, now=None, selected_ids=()):
    """Only cache reads; a new authority can never inherit the prior store's rows."""
    reader = reader or redis_get
    now = utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if region not in REGION_NAMES:
        return {
            "region": region,
            "past": [],
            "future": {},
            "error": {},
            "issues": [],
            "status": "Unknown authority",
        }
    actual = reader(redis_key(f"actuals:{region}")) or {}
    weather = reader(redis_key(f"weather:{region}")) or {}
    forecast = reader(redis_key(f"forecast:{region}:1h")) or {}
    actual_rows = _actual_rows(actual, now)
    weather_frame = _weather_frame(weather)
    weather_lookup = {iso(r["timestamp"]): r for r in weather_frame.to_dict("records")}
    past = []
    if actual_rows:
        as_of = utc(actual_rows[-1]["timestamp"])
        hours = (as_of - utc(actual_rows[0]["timestamp"])).total_seconds() / 3600 + 1
        historical = historical_weather_window(weather_frame, as_of, max(1, hours))
        history_lookup = {iso(r["timestamp"]): r for r in historical.to_dict("records")}
        past = [{**r, **_weather_values(history_lookup.get(r["timestamp"]))} for r in actual_rows]
    future, sources = {}, []
    raw_rows = forecast.get("forecasts") or []
    for requested in ("published", "prophet", "arima", "xgboost", "ensemble"):
        identity = resolve_forecast_identity(forecast, requested)
        if identity is None or (requested != "published" and identity.metric_key != requested):
            continue
        guard = (
            (forecast.get("horizon_guard") or {}).get(identity.metric_key, {})
            if isinstance(forecast.get("horizon_guard", {}), dict)
            else {}
        )
        maximum = number(guard.get("max_ok_horizon")) if isinstance(guard, dict) else None
        count = min(168, len(raw_rows), int(maximum) if maximum is not None else 168)
        selected = selection_values(forecast, identity, count) if count > 0 else None
        if selected is None:
            continue
        generated = utc(
            (forecast.get("timing") or {}).get("generated_at") or forecast.get("scored_at")
        )
        rows = []
        for stamp, value in zip(*selected, strict=True):
            clock = utc(stamp)
            if clock is None:
                continue
            key = iso(clock)
            rows.append(
                {
                    "timestamp": key,
                    "forecast_mw": value,
                    "generated_before_target": generated is not None and clock > generated,
                    **_weather_values(weather_lookup.get(key)),
                }
            )
        if rows:
            future[requested] = {
                "rows": sorted(rows, key=lambda r: r["timestamp"]),
                "label": identity.label,
                "status": identity.status,
                "metric_key": identity.metric_key,
                "note": identity.note,
                "generated_at": iso(generated),
                "max_horizon": int(maximum) if maximum is not None else None,
                "guard_reason": guard.get("reason") if isinstance(guard, dict) else None,
            }
            sources.append(
                {
                    "value": requested,
                    "label": identity.label
                    + (" · published" if requested == "published" else " · comparison"),
                }
            )
    issues = read_issues(region, reader, selected_ids)
    actual_frame = pd.DataFrame(actual_rows, columns=["timestamp", "demand_mw"])
    operator_values = actual.get("forecast_mw")
    operator_timestamps = actual.get("timestamps") or []
    has_operator_forecast = isinstance(operator_values, list) and len(operator_values) == len(
        operator_timestamps
    )
    if has_operator_forecast:
        operator_by_time = {
            iso(t): number(v) for t, v in zip(operator_timestamps, operator_values, strict=True)
        }
        actual_frame["forecast_mw"] = [operator_by_time.get(t) for t in actual_frame["timestamp"]]
    actual_lookup, actual_exclusions = hourly_actuals(actual_frame, now=now.to_pydatetime())
    for exclusion in actual.get("artifact_excluded", []):
        if not isinstance(exclusion, dict):
            continue
        stamp = utc(exclusion.get("ts"))
        if stamp is not None and now - pd.Timedelta(hours=168) < stamp <= now:
            actual_exclusions[stamp.isoformat()] = {
                "mw": exclusion.get("mw"),
                "reason": exclusion.get("reason"),
            }
    errors = {}
    for issue in issues:
        generated = utc(issue["timing"].get("generated_at"))
        published = utc(issue.get("observed_published_at"))
        boundary = max(v for v in (generated, published) if v is not None)
        rows = []
        for row in issue.get("forecast", []):
            stamp, value = utc(row.get("timestamp")), number(row.get("demand_mw"))
            key = iso(stamp)
            if stamp is None or stamp <= boundary or value is None or key not in actual_lookup:
                continue
            observed = actual_lookup[key]
            rows.append(
                {
                    "timestamp": key,
                    "forecast_mw": value,
                    "actual_mw": observed,
                    "error_mw": observed - value,
                    **_weather_values(weather_lookup.get(key)),
                }
            )
        if issue.get("loaded"):
            errors[issue["publication_id"]] = sorted(rows, key=lambda r: r["timestamp"])
    return {
        "region": region,
        "loaded_at": datetime.now(UTC).isoformat(),
        "past": past,
        "future": future,
        "sources": sources,
        "issues": issues,
        "error": errors,
        "actuals_at": actual.get("scored_at"),
        "weather_at": weather.get("scored_at"),
        "weather_latest_at": iso(weather_frame["timestamp"].max())
        if not weather_frame.empty
        else None,
        "actual_latest_at": actual_rows[-1]["timestamp"] if actual_rows else None,
        "actual_exclusions": actual_exclusions,
        "error_window_hours": 168,
        "quality_inputs": "operator_forecast_provided"
        if has_operator_forecast
        else "legacy_without_operator_forecast",
        "operator_forecast_finite_hours": sum(
            1
            for t, v in operator_by_time.items()
            if v is not None
            and utc(t) is not None
            and now - pd.Timedelta(hours=168) < utc(t) <= now
        )
        if has_operator_forecast
        else 0,
        "status": "Available" if past or future else "Cached demand/weather data unavailable",
        "notes": [
            "Historical weather is regional model/reanalysis/backfill context, not station measurements.",
            "Values can contain provider revisions/outliers. Quality-excluded or non-positive demand is not compared.",
            "Weather is matched by UTC target; the original weather input/issue vintage is not archived.",
        ],
    }


def compare_issues(earlier, later):
    """Compare genuine shared targets strictly after both recorded issue clocks."""
    boundaries = [utc(x.get("timing", {}).get("generated_at")) for x in (earlier, later)] + [
        utc(x.get("observed_published_at")) for x in (earlier, later)
    ]
    clocks = [x for x in boundaries if x is not None]
    if not clocks:
        return []
    late = {iso(r.get("timestamp")): number(r.get("demand_mw")) for r in later.get("forecast", [])}
    rows = []
    for item in earlier.get("forecast", []):
        stamp, value = utc(item.get("timestamp")), number(item.get("demand_mw"))
        other = late.get(iso(stamp))
        if stamp is not None and stamp > max(clocks) and value is not None and other is not None:
            rows.append(
                {
                    "timestamp": iso(stamp),
                    "earlier_mw": value,
                    "later_mw": other,
                    "change_mw": other - value,
                }
            )
    return sorted(rows, key=lambda r: r["timestamp"])


COMPARISON_WINDOWS = {"24": 24, "168": 168, "720": 720}


def comparison_window(rows: list[dict], hours: object = "24") -> tuple[list[dict], int]:
    """Select an elapsed shared-target window; gaps never extend its end.

    Args:
        rows: Eligible common targets produced by compare_issues.
        hours: Allowlisted duration control; malformed values use 24 hours.

    Returns:
        Selected targets and requested elapsed duration in hours.
    """
    duration = COMPARISON_WINDOWS.get(hours, 24) if isinstance(hours, str) else 24
    if not rows:
        return [], duration
    ordered = sorted(rows, key=lambda r: r["timestamp"])
    start = utc(ordered[0]["timestamp"])
    end = start + pd.Timedelta(hours=duration)
    return [r for r in ordered if start <= utc(r["timestamp"]) < end], duration


def revision_summary(rows: list[dict]) -> dict | None:
    """Describe revisions on the same eligible targets, choosing earliest ties.

    Args:
        rows: The selected finite common-target comparison rows.

    Returns:
        Same-window peak, timing, revision and count metrics, or None if empty.
    """
    if not rows:
        return None
    ordered = sorted(rows, key=lambda r: r["timestamp"])
    earlier_peak = max(r["earlier_mw"] for r in ordered)
    later_peak = max(r["later_mw"] for r in ordered)
    earlier_ties = [r for r in ordered if r["earlier_mw"] == earlier_peak]
    later_ties = [r for r in ordered if r["later_mw"] == later_peak]
    largest = max(ordered, key=lambda r: abs(r["change_mw"]))
    return {
        "earlier_peak_mw": earlier_peak,
        "later_peak_mw": later_peak,
        "peak_change_mw": later_peak - earlier_peak,
        "earlier_peak_target": earlier_ties[0]["timestamp"],
        "later_peak_target": later_ties[0]["timestamp"],
        "peak_shift_hours": (
            utc(later_ties[0]["timestamp"]) - utc(earlier_ties[0]["timestamp"])
        ).total_seconds()
        / 3600,
        "earlier_peak_ties": len(earlier_ties),
        "later_peak_ties": len(later_ties),
        "largest": largest,
        "largest_ties": sum(abs(r["change_mw"]) == abs(largest["change_mw"]) for r in ordered),
        "changed": sum(r["change_mw"] != 0 for r in ordered),
        "count": len(ordered),
    }
