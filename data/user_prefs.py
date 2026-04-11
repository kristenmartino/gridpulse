"""
Smart Defaults & Scenario Bookmarks (NEXD-9 + NEXD-12).

Persists the user's preferred region, persona, active tab, and per-tab filter
values in localStorage.  On return visits the dashboard silently restores these
instead of falling back to hardcoded defaults.

NEXD-12 adds serialize/deserialize for full dashboard state in URL bookmarks,
including per-tab filters and scenario simulator slider values.

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

# ── Scenario simulator slider IDs, defaults, ranges ──────────────────

SIM_SLIDERS: list[str] = [
    "sim-temp",
    "sim-wind",
    "sim-cloud",
    "sim-humidity",
    "sim-solar",
]

SIM_SLIDER_DEFAULTS: dict[str, float] = {
    "sim-temp": 75,
    "sim-wind": 15,
    "sim-cloud": 50,
    "sim-humidity": 60,
    "sim-solar": 500,
}

SIM_SLIDER_RANGES: dict[str, tuple[float, float]] = {
    "sim-temp": (-10, 120),
    "sim-wind": (0, 80),
    "sim-cloud": (0, 100),
    "sim-humidity": (0, 100),
    "sim-solar": (0, 1000),
}

# Default sim-duration (tracked as a filter, not a slider)
SIM_DURATION_DEFAULT: int = 24


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


# ── NEXD-12: Bookmark serialization / deserialization ─────────────────

# Checklist filter IDs whose values are lists (comma-separated in URL)
_CHECKLIST_FILTERS: set[str] = {"tab3-model-selector"}


def serialize_bookmark_params(
    region: str,
    persona: str,
    tab: str,
    filters: dict[str, str | list[str] | int],
    sim_sliders: dict[str, float] | None = None,
) -> str:
    """Serialize full dashboard state into a URL query string.

    Args:
        region: Balancing authority code.
        persona: Persona ID.
        tab: Active tab ID.
        filters: Dict of tracked filter ID to current value.
        sim_sliders: Dict of sim slider ID to current numeric value.

    Returns:
        URL query string with leading "?".
    """
    from urllib.parse import urlencode

    params: dict[str, str] = {
        "region": region or "FPL",
        "persona": persona or "grid_ops",
        "tab": tab or "tab-overview",
    }

    for fid in TRACKED_FILTERS:
        val = filters.get(fid)
        if val is None:
            continue
        if isinstance(val, list):
            params[f"f.{fid}"] = ",".join(str(v) for v in val)
        else:
            params[f"f.{fid}"] = str(val)

    if sim_sliders:
        for sid in SIM_SLIDERS:
            val = sim_sliders.get(sid)
            if val is not None:
                short_key = sid.replace("sim-", "s.")
                params[short_key] = str(val)

    return f"?{urlencode(params)}"


def deserialize_bookmark_params(search: str) -> dict:
    """Parse URL query string into validated dashboard state.

    Args:
        search: URL search string (e.g. "?region=FPL&f.tab1-timerange=168").

    Returns:
        Dict with optional keys: region, persona, tab, filters, sim_sliders.
        Missing or invalid values are omitted so callers can use no_update.
    """
    from urllib.parse import parse_qs

    if not search:
        return {}

    params = parse_qs(search.lstrip("?"))
    result: dict = {}

    from config import REGION_NAMES, TAB_IDS
    from personas.config import PERSONAS

    region = params.get("region", [None])[0]
    if region and region in REGION_NAMES:
        result["region"] = region

    persona = params.get("persona", [None])[0]
    if persona and persona in PERSONAS:
        result["persona"] = persona

    tab = params.get("tab", [None])[0]
    if tab and tab in TAB_IDS:
        result["tab"] = tab

    # Tracked filters (f. prefix)
    filters: dict[str, str | list[str] | int] = {}
    for fid in TRACKED_FILTERS:
        raw = params.get(f"f.{fid}", [None])[0]
        if raw is None:
            continue
        if fid in _CHECKLIST_FILTERS:
            val: str | list[str] | int = raw.split(",")
        else:
            val = raw
        validated = _validate_filter_value(fid, val)
        if validated is not None:
            filters[fid] = validated
    if filters:
        result["filters"] = filters

    # Sim sliders (s. prefix)
    sim_sliders: dict[str, float] = {}
    for sid in SIM_SLIDERS:
        short_key = sid.replace("sim-", "s.")
        raw_slider = params.get(short_key, [None])[0]
        if raw_slider is None:
            continue
        try:
            num = float(raw_slider)
            min_v, max_v = SIM_SLIDER_RANGES[sid]
            if min_v <= num <= max_v:
                sim_sliders[sid] = num
            else:
                log.warning("bookmark_slider_out_of_range", slider=sid, value=num)
        except (TypeError, ValueError):
            log.warning("bookmark_slider_invalid", slider=sid, raw=raw_slider)
    if sim_sliders:
        result["sim_sliders"] = sim_sliders

    return result
