"""Validated, shareable explorer controls. No I/O or model execution."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd

from components.weather_explorer_data import WEATHER_FIELDS, iso, utc

MODES = ("past", "future", "error")
SPANS = {"past": (24, 168), "future": (24, 168), "error": (6, 12, 24, 168, "all")}
SOURCES = ("published", "prophet", "arima", "xgboost", "ensemble")


def clean_state(raw, region):
    raw = raw if isinstance(raw, dict) and raw.get("region", region) == region else {}
    mode = raw.get("mode") if raw.get("mode") in MODES else "past"
    windows = {}
    for name in MODES:
        old = (
            (raw.get("windows") or {}).get(name, {}) if isinstance(raw.get("windows"), dict) else {}
        )
        old = old if isinstance(old, dict) else {}
        span = old.get("hours")
        windows[name] = {
            "start": iso(old.get("start")) or "",
            "hours": span
            if span in SPANS[name]
            else (168 if name == "past" else "all" if name == "error" else 24),
            "cursor": iso(old.get("cursor")) or "",
        }
    return {
        "region": region,
        "mode": mode,
        "section": raw.get("section")
        if raw.get("section") in ("explorer", "messages")
        else "explorer",
        "variable": raw.get("variable")
        if isinstance(raw.get("variable"), str) and raw.get("variable") in WEATHER_FIELDS
        else "temperature_f",
        "source": raw.get("source") if raw.get("source") in SOURCES else "published",
        "issue": str(raw.get("issue") or "")[:64],
        "windows": windows,
    }


def rows_for(data, state):
    mode = state["mode"]
    if mode == "past":
        return data.get("past") or []
    if mode == "future":
        return (data.get("future", {}).get(state["source"]) or {}).get("rows") or []
    return (data.get("error") or {}).get(state["issue"]) or []


def default_start(rows, mode, span, generated=None):
    if not rows:
        return ""
    if span == "all":
        return rows[0]["timestamp"]
    if mode == "future":
        clock = utc(generated)
        return next(
            (r["timestamp"] for r in rows if clock is None or utc(r["timestamp"]) > clock),
            rows[0]["timestamp"],
        )
    return max(
        utc(rows[0]["timestamp"]), utc(rows[-1]["timestamp"]) - pd.Timedelta(hours=span - 1)
    ).isoformat()


def normalize_window(data, state):
    """Clamp only hydrated state; interactive invalid dates use validate_start."""
    result = deepcopy(state)
    issues = data.get("issues") or []
    valid_ids = {x.get("publication_id") for x in issues}
    if result["issue"] not in valid_ids:
        result["issue"] = next(
            (x["publication_id"] for x in issues if data.get("error", {}).get(x["publication_id"])),
            next(iter([x["publication_id"] for x in issues]), ""),
        )
    rows = rows_for(data, result)
    window = result["windows"][result["mode"]]
    if not rows:
        return result, []
    generated = (data.get("future", {}).get(result["source"]) or {}).get("generated_at")
    start = utc(window["start"]) or utc(
        default_start(rows, result["mode"], window["hours"], generated)
    )
    start = min(max(start, utc(rows[0]["timestamp"])), utc(rows[-1]["timestamp"])).floor("h")
    window["start"] = start.isoformat()
    selected = [
        r
        for r in rows
        if start
        <= utc(r["timestamp"])
        < (
            utc(rows[-1]["timestamp"]) + pd.Timedelta(hours=1)
            if window["hours"] == "all"
            else start + pd.Timedelta(hours=window["hours"])
        )
    ]
    if not selected:
        window["start"] = rows[0]["timestamp"]
        selected = rows[:1]
    if window["cursor"] not in {r["timestamp"] for r in selected}:
        window["cursor"] = selected[0]["timestamp"]
    return result, selected


def validate_start(value, rows):
    """A bad edit never silently selects another investigation period."""
    clock = utc(value)
    if clock is None:
        return "Enter a complete UTC date and hour."
    if clock != clock.floor("h"):
        return "Use a whole UTC hour, with minutes set to 00."
    if not rows:
        return "This source has no available target hours."
    if not utc(rows[0]["timestamp"]) <= clock <= utc(rows[-1]["timestamp"]):
        return f"Choose an hour from {rows[0]['timestamp']} to {rows[-1]['timestamp']}."
    return None
