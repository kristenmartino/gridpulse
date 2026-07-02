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

from config import CACHE_TTL_SECONDS, NOAA_BASE_URL, STATE_TO_BA
from data.cache import get_cache

log = structlog.get_logger()

REQUEST_TIMEOUT = 10

# NOAA requires a User-Agent header
HEADERS = {
    "User-Agent": "GridPulse (kristen@kristenmartino.ai)",
    "Accept": "application/geo+json",
}


class NOAAAlertsUnavailableError(RuntimeError):
    """Raised when NO state fetch for a region succeeded and no cached data
    (fresh or stale) exists. Callers must degrade to an explicit
    "unavailable" state — a NOAA outage must never be indistinguishable
    from "zero active alerts" (2026-07 critical-review loophole class)."""


class _NOAACircuitBreaker:
    """Process-local fail-fast guard for sustained NOAA outages.

    Same policy as ``data.eia_client._EIACircuitBreaker`` (see the
    2026-06-04 EIA-outage rule in CLAUDE.md), minimally reimplemented here
    until #185 unifies client fallback machinery: after ``threshold``
    consecutive hard failures, subsequent calls skip HTTP entirely (a
    scoring run fans out to ~49 unique states — unbounded timeouts would
    threaten the job ceiling, #171), with a probe allowed every
    ``probe_interval`` suppressed calls to detect recovery. Per-process
    state; resets on any success and on every fresh job run.
    """

    def __init__(self, threshold: int = 5, probe_interval: int = 20) -> None:
        self._threshold = threshold
        self._probe_interval = probe_interval
        self._consecutive_failures = 0
        self._suppressed_since_probe = 0

    def allow_request(self) -> bool:
        if self._consecutive_failures < self._threshold:
            return True
        self._suppressed_since_probe += 1
        if self._suppressed_since_probe >= self._probe_interval:
            self._suppressed_since_probe = 0
            return True
        return False

    def record_success(self) -> None:
        if self._consecutive_failures >= self._threshold:
            log.info("noaa_circuit_recovered")
        self._consecutive_failures = 0
        self._suppressed_since_probe = 0

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures == self._threshold:
            log.warning("noaa_circuit_tripped", threshold=self._threshold)


_breaker = _NOAACircuitBreaker()

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
            return [_alert_from_dict(a) for a in cached]

    states = STATE_TO_BA[region]
    log.info("noaa_fetching_alerts", region=region, states=states)

    all_alerts: dict[str, WeatherAlert] = {}
    failed_states: list[str] = []

    for state in states:
        alerts = _fetch_state_alerts(state)
        if alerts is None:
            failed_states.append(state)
            continue
        for alert in alerts:
            if alert.id not in all_alerts:
                alert.balancing_authorities = [region]
                all_alerts[alert.id] = alert

    if failed_states and len(failed_states) == len(states):
        # Total failure — an outage must be distinguishable from "no active
        # alerts". Serve stale real data if we have it, else raise.
        stale = cache.get(cache_key, allow_stale=True)
        if stale is not None:
            log.warning("noaa_serving_stale_alerts", region=region, count=len(stale))
            return [_alert_from_dict(a) for a in stale]
        raise NOAAAlertsUnavailableError(
            f"All {len(states)} state fetches failed for {region} and no cached alerts exist"
        )
    if failed_states:
        log.warning("noaa_partial_state_failure", region=region, failed_states=failed_states)

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
    Fetch alerts for every balancing authority in ``STATE_TO_BA`` (all 51
    BAs in ``config.REGION_COORDINATES`` carry a state mapping).

    Returns:
        Dict mapping region code → list of WeatherAlert.
    """
    result = {}
    for region in STATE_TO_BA:
        result[region] = fetch_alerts_for_region(region, use_cache=use_cache)
    return result


def _fetch_state_alerts(state: str) -> list[WeatherAlert] | None:
    """Fetch active alerts for a single state.

    Returns ``None`` on fetch failure (outage), ``[]`` on a genuine
    zero-alert response — callers must not conflate the two. State-level
    results are cached (states are shared across BAs: 51 regions fan out to
    115 state lookups but only ~49 unique states, and the scoring job runs
    hourly), and a stale state cache is served on failure before giving up.
    """
    cache = get_cache()
    state_key = f"noaa_state_{state}"
    cached = cache.get(state_key)
    if cached is not None:
        return [_alert_from_dict(a) for a in cached]

    if not _breaker.allow_request():
        stale = cache.get(state_key, allow_stale=True)
        if stale is not None:
            return [_alert_from_dict(a) for a in stale]
        return None

    url = f"{NOAA_BASE_URL}/alerts/active"
    params = {"area": state}

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        _breaker.record_success()
    except requests.RequestException as e:
        _breaker.record_failure()
        log.warning("noaa_request_failed", state=state, error=str(e))
        stale = cache.get(state_key, allow_stale=True)
        if stale is not None:
            log.warning("noaa_serving_stale_state", state=state, count=len(stale))
            return [_alert_from_dict(a) for a in stale]
        return None

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

    cache.set(
        state_key,
        [_alert_to_dict(a) for a in alerts],
        ttl=min(CACHE_TTL_SECONDS, 1800),
    )
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


def _alert_from_dict(d: dict) -> WeatherAlert:
    """Rebuild a WeatherAlert from a cached dict, inverting ``_alert_to_dict``.

    Critically, ``onset``/``expires`` are re-parsed back to ``datetime`` so
    the ``WeatherAlert`` contract (``expires: datetime | None``) holds on the
    cache-hit path too. Reconstructing with the raw ISO strings (as a bare
    ``WeatherAlert(**d)`` would) leaves them as ``str`` and crashes any
    consumer that treats them as datetimes — which silently degraded the
    scoring job's alerts phase to "unavailable" on every cache hit.
    """
    fields = dict(d)
    fields["onset"] = _parse_datetime(fields.get("onset"))
    fields["expires"] = _parse_datetime(fields.get("expires"))
    return WeatherAlert(**fields)
