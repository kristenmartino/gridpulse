"""
NOAA/NWS API client for severe weather alerts.

Fetches active weather alerts and maps them to balancing authorities
using the STATE_TO_BA lookup. Alerts are classified by severity:
    Extreme, Severe → Critical
    Moderate → Warning
    Minor → Info

API docs: https://www.weather.gov/documentation/services-web-api
"""

from dataclasses import dataclass
from datetime import datetime

import requests
import structlog

from config import STATE_TO_BA, CACHE_TTL_SECONDS, NOAA_BASE_URL
from data.cache import get_cache

log = structlog.get_logger()

REQUEST_TIMEOUT = 15

# NOAA requires a User-Agent header
HEADERS = {
    "User-Agent": "(NextEra Energy Dashboard, contact@example.com)",
    "Accept": "application/geo+json",
}

# Severity mapping: NOAA severity → dashboard severity
SEVERITY_MAP = {
    "Extreme": "critical",
    "Severe": "critical",
    "Moderate": "warning",
    "Minor": "info",
    "Unknown": "info",
}


@dataclass
class WeatherAlert:
    """Parsed weather alert from NOAA."""

    id: str
    event: str
    headline: str
    description: str
    severity: str  # "critical", "warning", "info"
    noaa_severity: str  # Original NOAA severity
    urgency: str
    certainty: str
    onset: datetime | None
    expires: datetime | None
    areas: list[str]  # Affected areas/counties
    states: list[str]  # Affected state codes
    balancing_authorities: list[str]  # Mapped BAs


def fetch_alerts_for_region(
    region: str,
    use_cache: bool = True,
) -> list[WeatherAlert]:
    """
    Fetch active NOAA weather alerts for a balancing authority.

    Queries all states mapped to the given BA and deduplicates alerts.

    Args:
        region: Balancing authority code (e.g., "ERCOT", "FPL").
        use_cache: Whether to check cache first.

    Returns:
        List of WeatherAlert objects sorted by severity (critical first).
    """
    if region not in STATE_TO_BA:
        raise ValueError(f"Unknown region: {region}. Valid: {list(STATE_TO_BA.keys())}")

    cache_key = f"noaa_alerts_{region}"
    cache = get_cache()

    if use_cache:
        cached = cache.get(cache_key)
        if cached is not None:
            return [WeatherAlert(**a) for a in cached]

    states = STATE_TO_BA[region]
    log.info("noaa_fetching_alerts", region=region, states=states)

    all_alerts: dict[str, WeatherAlert] = {}

    for state in states:
        alerts = _fetch_state_alerts(state)
        for alert in alerts:
            if alert.id not in all_alerts:
                alert.balancing_authorities = [region]
                all_alerts[alert.id] = alert

    result = sorted(
        all_alerts.values(),
        key=lambda a: {"critical": 0, "warning": 1, "info": 2}.get(a.severity, 3),
    )

    # Cache as list of dicts for serialization
    cache.set(
        cache_key,
        [_alert_to_dict(a) for a in result],
        ttl=min(CACHE_TTL_SECONDS, 1800),  # Alerts refresh every 30 min max
    )
    log.info("noaa_alerts_cached", region=region, count=len(result))
    return result


def fetch_all_alerts(use_cache: bool = True) -> dict[str, list[WeatherAlert]]:
    """
    Fetch alerts for all 8 balancing authorities.

    Returns:
        Dict mapping region code → list of WeatherAlert.
    """
    result = {}
    for region in STATE_TO_BA:
        result[region] = fetch_alerts_for_region(region, use_cache=use_cache)
    return result


def _fetch_state_alerts(state: str) -> list[WeatherAlert]:
    """Fetch active alerts for a single state."""
    url = f"{NOAA_BASE_URL}/alerts/active"
    params = {"area": state}

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        log.warning("noaa_request_failed", state=state, error=str(e))
        return []

    features = data.get("features", [])
    alerts = []

    for feature in features:
        props = feature.get("properties", {})
        alert = WeatherAlert(
            id=props.get("id", ""),
            event=props.get("event", "Unknown"),
            headline=props.get("headline", ""),
            description=props.get("description", "")[:500],  # Truncate long descriptions
            severity=SEVERITY_MAP.get(props.get("severity", "Unknown"), "info"),
            noaa_severity=props.get("severity", "Unknown"),
            urgency=props.get("urgency", "Unknown"),
            certainty=props.get("certainty", "Unknown"),
            onset=_parse_datetime(props.get("onset")),
            expires=_parse_datetime(props.get("expires")),
            areas=_parse_areas(props.get("areaDesc", "")),
            states=[state],
            balancing_authorities=[],
        )
        alerts.append(alert)

    log.debug("noaa_state_alerts", state=state, count=len(alerts))
    return alerts


def _parse_datetime(dt_str: str | None) -> datetime | None:
    """Parse ISO datetime string from NOAA."""
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str)
    except (ValueError, TypeError):
        return None


def _parse_areas(area_desc: str) -> list[str]:
    """Parse semicolon-separated area descriptions."""
    if not area_desc:
        return []
    return [a.strip() for a in area_desc.split(";") if a.strip()]


def _alert_to_dict(alert: WeatherAlert) -> dict:
    """Convert a WeatherAlert to a JSON-serializable dict."""
    return {
        "id": alert.id,
        "event": alert.event,
        "headline": alert.headline,
        "description": alert.description,
        "severity": alert.severity,
        "noaa_severity": alert.noaa_severity,
        "urgency": alert.urgency,
        "certainty": alert.certainty,
        "onset": alert.onset.isoformat() if alert.onset else None,
        "expires": alert.expires.isoformat() if alert.expires else None,
        "areas": alert.areas,
        "states": alert.states,
        "balancing_authorities": alert.balancing_authorities,
    }
