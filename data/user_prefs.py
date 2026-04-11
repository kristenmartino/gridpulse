"""
Smart Defaults (Backlog NEXD-9: "Smart Defaults That Learn").

Persists the user's preferred region, persona, active tab, and per-tab filter
values in localStorage.  On return visits the dashboard silently restores these
instead of falling back to hardcoded defaults.

Pure logic — no Dash or I/O dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import structlog

log = structlog.get_logger()

# ── Tracked filter IDs and their defaults ────────────────────────────

TRACKED_FILTERS: list[str] = [
    "tab1-timerange",
    "outlook-horizon",
    "outlook-model",
    "backtest-horizon",
    "gen-date-range",
    "tab3-model-selector",
    "sim-duration",
]

FILTER_DEFAULTS: dict[str, str | list[str]] = {
    "tab1-timerange": "168",
    "outlook-horizon": "168",
    "outlook-model": "xgboost",
    "backtest-horizon": "24",
    "gen-date-range": "168",
    "tab3-model-selector": ["prophet", "arima", "xgboost", "ensemble"],
    "sim-duration": 24,
}

# ── Valid option sets for per-filter validation ──────────────────────

_FILTER_OPTIONS: dict[str, set[str] | None] = {
    "tab1-timerange": {"24", "168", "720", "2160"},
    "outlook-horizon": {"24", "168", "720"},
    "outlook-model": {"xgboost", "prophet", "arima", "ensemble"},
    "backtest-horizon": {"24", "168", "720"},
    "gen-date-range": {"24", "168", "720", "2160"},
    "tab3-model-selector": {"prophet", "arima", "xgboost", "ensemble"},
    "sim-duration": None,  # numeric, validated separately
}


@dataclass
class UserPrefs:
    """Persisted user preferences for dashboard state."""

    region: str = "FPL"
    persona: str = "grid_ops"
    tab: str = "tab-overview"
    filters: dict[str, str | list[str] | int] = field(default_factory=dict)
    updated_at: str = ""

    def to_dict(self) -> dict:
        """Serialize for JSON/localStorage storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> UserPrefs:
        """Reconstruct from a JSON-round-tripped dict."""
        if not d or not isinstance(d, dict):
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


def _validate_filter_value(
    filter_id: str, value: str | list[str] | int | None
) -> str | list[str] | int | None:
    """Validate a single filter value against known options.

    Returns the value if valid, or None if invalid.
    """
    options = _FILTER_OPTIONS.get(filter_id)
    if options is None:
        # Numeric filter (sim-duration) — accept int-like values
        if filter_id == "sim-duration":
            try:
                v = int(value) if value is not None else None
                if v in (24, 48, 72, 168):
                    return v
            except (TypeError, ValueError):
                pass
            return None
        return None

    if isinstance(value, list):
        # Checklist — keep only valid items
        return [v for v in value if str(v) in options] or None

    if isinstance(value, str) and value in options:
        return value

    return None


def validate_prefs(prefs_data: dict | None) -> UserPrefs:
    """Parse and sanitize stored preferences against known valid values.

    Invalid or missing values fall back to defaults. Unknown filter IDs
    are silently dropped.
    """
    from config import REGION_NAMES, TAB_IDS
    from personas.config import PERSONAS

    prefs = UserPrefs.from_dict(prefs_data)

    # Validate globals
    if prefs.region not in REGION_NAMES:
        prefs.region = "FPL"
    if prefs.persona not in PERSONAS:
        prefs.persona = "grid_ops"
    if prefs.tab not in TAB_IDS:
        prefs.tab = "tab-overview"

    # Validate per-filter values
    clean_filters: dict[str, str | list[str] | int] = {}
    for fid in TRACKED_FILTERS:
        raw = prefs.filters.get(fid)
        if raw is not None:
            valid = _validate_filter_value(fid, raw)
            if valid is not None:
                clean_filters[fid] = valid
    prefs.filters = clean_filters

    return prefs
