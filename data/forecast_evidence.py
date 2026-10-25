"""Public, explicitly qualified provenance for an existing forecast snapshot."""

from __future__ import annotations

from datetime import UTC, datetime


def utc_timestamp(value: object) -> str | None:
    """Unknown/naive clocks stay unknown; never silently assign a timezone."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(UTC).isoformat()
    except ValueError:
        return None


def forecast_timing(payload: dict | None, hours: int) -> dict:
    """Scoring completion is generation time, not an operator issue clock."""
    payload = payload if isinstance(payload, dict) else {}
    rows = (payload.get("forecasts") or [])[:hours]
    anchor = payload.get("anchor")
    anchor = anchor if isinstance(anchor, dict) else {}
    targets = [utc_timestamp(r.get("timestamp")) for r in rows if isinstance(r, dict)]
    return {
        "generated_at": utc_timestamp(payload.get("scored_at")),
        "forecast_start_at": targets[0] if targets else None,
        "target_end_at": targets[-1] if targets else None,
        "target_hours": len(rows),
        "timezone": "UTC",
        "interval_convention": "hour_ending",
        "input_anchor_at": utc_timestamp(anchor.get("anchor_ts")),
        "input_anchor_was_placeholder": anchor.get("anchor_was_placeholder"),
        "input_anchor_conditioned": anchor.get("anchor_conditioned"),
        "operator_issued_at": None,
        "publication_id": payload.get("publication_id"),
    }


def baseline_selection(payload: dict | None) -> dict | None:
    """Disclose a recorded selection reason, without inventing an onset time."""
    if not isinstance(payload, dict) or payload.get("served_series") != "seasonal-naive":
        return None
    skill = payload.get("skill")
    skill = skill if isinstance(skill, dict) else {}
    return {
        "reason": payload.get("served_reason") or None,
        "decision_recorded_at": utc_timestamp(payload.get("scored_at")),
        "selected_since": None,
        "compared_series": skill.get("compared_series"),
        "comparison_basis": skill.get("comparison_basis"),
    }


_COVERAGE_FIELDS = (
    "as_of",
    "window_start",
    "window_end",
    "possible_target_hours",
    "scored_target_hours",
    "unscored_target_hours",
    "scored_start",
    "scored_end",
    "low_actual_excluded",
    "duplicate_records",
)


def public_coverage(block: dict) -> dict | None:
    """Nested allow-list: future cache/debug fields must not become public."""
    coverage = block.get("coverage_7d")
    if not isinstance(coverage, dict):
        return None
    return {key: coverage[key] for key in _COVERAGE_FIELDS if key in coverage}
