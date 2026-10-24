"""
Error handling, loading states, and empty state components.

Provides:
- Safe callback wrapper (catches exceptions, logs, returns error UI)
- Loading spinner components
- Empty state components (no data, API error, stale data)
- Data freshness badge generator
"""

import functools
import traceback
from datetime import UTC, datetime

import dash_bootstrap_components as dbc
import structlog
from dash import html

log = structlog.get_logger()


# ── Safe Callback Decorator ───────────────────────────────────


def safe_callback(*fallback_outputs):
    """
    Decorator that wraps Dash callbacks in try/except.

    On exception: logs the error with structlog, returns fallback outputs.

    Usage:
        @app.callback(Output("chart", "figure"), Input("store", "data"))
        @safe_callback(empty_figure("Error loading chart"))
        def update_chart(data):
            ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log.error(
                    "callback_error",
                    callback=func.__name__,
                    error=str(e),
                    traceback=traceback.format_exc()[-500:],
                )
                if len(fallback_outputs) == 1:
                    return fallback_outputs[0]
                return fallback_outputs

        return wrapper

    return decorator


# ── Loading Components ────────────────────────────────────────


def loading_spinner(message: str = "Loading data...") -> html.Div:
    """
    Full-width loading spinner with message.

    Used as initial content for tab panels while data loads.
    """
    return html.Div(
        [
            dbc.Spinner(
                color="danger",
                spinner_style={"width": "2rem", "height": "2rem"},
            ),
            html.P(
                message,
                style={
                    "color": "#A8B3C7",
                    "marginTop": "12px",
                    "fontSize": "0.9rem",
                },
            ),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",
            "justifyContent": "center",
            "padding": "60px 0",
        },
    )


def loading_overlay(component_id: str) -> dbc.Spinner:
    """
    Wrap a component in a dbc.Spinner for loading states.

    The spinner shows while callbacks update the component.
    """
    return dbc.Spinner(
        id=f"{component_id}-loading",
        color="danger",
        spinner_style={"width": "1.5rem", "height": "1.5rem"},
        delay_show=300,  # Only show if loading takes >300ms
    )


# ── Empty State Components ────────────────────────────────────


def empty_state(
    title: str = "No data available",
    message: str = "No data is available for this selection.",
    icon: str = "📊",
) -> html.Div:
    """
    Empty state placeholder when no data is loaded.
    """
    return html.Div(
        [
            html.Div(icon, style={"fontSize": "2.5rem", "marginBottom": "12px"}),
            html.H5(title, style={"color": "#DDE6F2", "marginBottom": "6px"}),
            html.P(message, style={"color": "#A8B3C7", "fontSize": "0.85rem"}),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",
            "justifyContent": "center",
            "padding": "60px 0",
            "textAlign": "center",
        },
    )


def error_state(
    title: str = "Could not load this view",
    message: str = "Refresh the page to retry, or select another region.",
    error_detail: str | None = None,
) -> html.Div:
    """
    Error state shown when a callback fails.
    """
    children = [
        html.Div("⚠️", style={"fontSize": "2.5rem", "marginBottom": "12px"}),
        html.H5(title, style={"color": "#FF5C7A", "marginBottom": "6px"}),
        html.P(message, style={"color": "#A8B3C7", "fontSize": "0.85rem"}),
    ]
    if error_detail:
        log.warning("error_state_detail", detail=error_detail)
    return html.Div(
        children,
        style={
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",
            "justifyContent": "center",
            "padding": "60px 0",
            "textAlign": "center",
        },
    )


def api_error_state(api_name: str = "API", cached_available: bool = False) -> html.Div:
    """
    Show a source failure; disclose cached data only when the caller confirms it.
    """
    return html.Div(
        [
            html.Div("🔄", style={"fontSize": "1.5rem", "marginBottom": "8px"}),
            html.P(
                (
                    f"{api_name} is unavailable. Showing cached data."
                    if cached_available
                    else f"{api_name} is unavailable. Try again later."
                ),
                style={"color": "#ffb74d", "fontSize": "0.8rem", "margin": 0},
            ),
        ],
        style={
            "background": "rgba(255, 183, 77, 0.1)",
            "borderRadius": "6px",
            "padding": "10px 16px",
            "marginBottom": "8px",
            "textAlign": "center",
        },
    )


# ── Data Freshness Badge ─────────────────────────────────────


def freshness_badge(
    last_updated: datetime | None,
    stale_threshold_seconds: int = 7200,
    expired_threshold_seconds: int = 21600,
) -> html.Span:
    """
    Generate a freshness badge based on data age.

    Args:
        last_updated: When the data was last fetched.
        stale_threshold_seconds: Age (s) at which data is considered stale (default: 2h).
        expired_threshold_seconds: Age (s) at which data is considered expired (default: 6h).

    Returns:
        Colored badge: "fresh" (green), "stale" (yellow), "expired" (red).
    """
    if last_updated is None:
        return html.Span("No data", className="freshness-badge expired")

    now = datetime.now(UTC)
    age_seconds = (now - last_updated).total_seconds()

    if age_seconds < stale_threshold_seconds:
        minutes = int(age_seconds // 60)
        label = f"Updated {minutes}m ago" if minutes > 0 else "Just updated"
        css_class = "fresh"
    elif age_seconds < expired_threshold_seconds:
        hours = int(age_seconds // 3600)
        label = f"Stale ({hours}h ago)"
        css_class = "stale"
    else:
        hours = int(age_seconds // 3600)
        label = f"Expired ({hours}h ago)"
        css_class = "expired"

    return html.Span(label, className=f"freshness-badge {css_class}")


def format_last_updated(dt: datetime | None) -> str:
    """Format a datetime for display in the header."""
    if dt is None:
        return "Never"
    now = datetime.now(UTC)
    delta = now - dt
    if delta.total_seconds() < 60:
        return "Just now"
    elif delta.total_seconds() < 3600:
        return f"{int(delta.total_seconds() // 60)}m ago"
    elif delta.total_seconds() < 86400:
        return f"{int(delta.total_seconds() // 3600)}h ago"
    else:
        return dt.strftime("%b %d %H:%M UTC")


# ── Data Confidence Badge (E3) ───────────────────────────────

# Confidence levels map data source status to a user-visible reliability tier
CONFIDENCE_LEVELS = {
    "high": {
        "icon": "check-circle",
        "label": "Fresh",
        "color": "#34d399",
        "description": "Source status is fresh. This does not assess data accuracy.",
    },
    "medium": {
        "icon": "alert-circle",
        "label": "Stale or incomplete",
        "color": "#fbbf24",
        "description": "Data may be stale or partially unavailable",
    },
    "low": {
        "icon": "alert-triangle",
        "label": "Unavailable",
        "color": "#f87171",
        "description": "Data could not be loaded. Check the view for any available fallback data.",
    },
    "demo": {
        "icon": "flask",
        "label": "Demo",
        "color": "#a1a1aa",
        "description": "Synthetic data for demonstration.",
    },
    # "warming" is surfaced in Redis-only deployments when the scheduled
    # scoring / training jobs haven't produced results yet (e.g. first
    # deploy, Redis eviction, or a scheduler outage).
    "warming": {
        "icon": "clock",
        "label": "Not available",
        "color": "#60a5fa",
        "description": "No published data is available for this selection.",
    },
}


def data_confidence_level(
    source_status: str,
    age_seconds: float | None = None,
    stale_threshold: int = 7200,
) -> str:
    """
    Determine confidence level from source status and data age.

    Args:
        source_status: "fresh", "stale", "warming", "demo", or "error".
        age_seconds: How old the data is (None = unknown).
        stale_threshold: Seconds before fresh data becomes medium confidence.

    Returns:
        Confidence level: "high", "medium", "low", "warming", or "demo".
    """
    if source_status == "demo":
        return "demo"
    if source_status == "warming":
        return "warming"
    if source_status == "error":
        return "low"
    if source_status == "stale":
        return "medium"
    # Fresh — but check age
    if age_seconds is not None and age_seconds > stale_threshold:
        return "medium"
    return "high"


def warming_state(
    title: str = "Data unavailable",
    message: str = (
        "No published data is available for this selection. "
        "Try another region or check again later."
    ),
) -> html.Div:
    """Render a calm degraded-state placeholder for Redis-only deployments.

    Used when the primary data source has no published results. The status
    alone does not establish a cause or a recovery time.
    """
    from components.icons import icon as _icon

    return html.Div(
        [
            html.Div(
                _icon("clock", size="xl", className="gp-warming-icon"),
                style={"marginBottom": "12px"},
            ),
            html.H5(title, style={"color": "var(--text-secondary)", "marginBottom": "6px"}),
            html.P(message, style={"color": "var(--text-tertiary)", "fontSize": "0.85rem"}),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",
            "justifyContent": "center",
            "padding": "60px 0",
            "textAlign": "center",
            "background": "rgba(122, 168, 255, 0.04)",
            "borderRadius": "6px",
        },
    )


def is_warming(freshness: dict[str, str] | None) -> bool:
    """Return ``True`` when the primary data source is in the warming state."""
    if not freshness:
        return False
    # Treat any primary source being "warming" as a warming page. Alerts
    # alone warming (without demand/weather warming) doesn't dominate the UI.
    return any(freshness.get(source) == "warming" for source in ("demand", "weather"))


def confidence_badge(
    widget_name: str,
    confidence: str,
    age_text: str = "",
) -> html.Div:
    """
    Generate a compact confidence badge for a widget (E3).

    Displayed in the corner of each widget showing data reliability.

    Args:
        widget_name: Display name (e.g., "Demand", "Weather", "Pricing").
        confidence: "high", "medium", "low", or "demo".
        age_text: Optional age string (e.g., "4m ago").

    Returns:
        Colored badge element: "🟢 Demand · 4m ago"
    """
    from components.icons import icon as _icon

    level = CONFIDENCE_LEVELS.get(confidence, CONFIDENCE_LEVELS["low"])
    parts = [
        _icon(
            level["icon"],
            size="xs",
            className="gp-confidence-badge__icon",
        ),
        html.Span(
            widget_name,
            style={
                "fontWeight": "600",
                "color": level["color"],
                "marginRight": "4px",
            },
        ),
    ]
    if age_text:
        parts.append(
            html.Span(
                f"· {age_text}",
                style={"color": "#A8B3C7", "fontSize": "0.7rem"},
            )
        )

    return html.Div(
        parts,
        title=level["description"],
        style={
            "display": "inline-flex",
            "alignItems": "center",
            "fontSize": "0.7rem",
            "padding": "2px 8px",
            "borderRadius": "4px",
            "background": f"{level['color']}15",
            "marginRight": "8px",
        },
    )


def widget_confidence_bar(
    freshness: dict[str, str],
    age_seconds: float | None = None,
) -> html.Div:
    """
    Generate a row of confidence badges for all data sources.

    Args:
        freshness: Dict of source -> status (from data-freshness-store).
        age_seconds: Data age in seconds.

    Returns:
        Horizontal bar of confidence badges (E3 + A4 combined).
    """

    age_text = ""
    if age_seconds is not None:
        if age_seconds < 60:
            age_text = "just now"
        elif age_seconds < 3600:
            age_text = f"{int(age_seconds // 60)}m ago"
        else:
            age_text = f"{int(age_seconds // 3600)}h ago"

    source_labels = {
        "demand": "Demand",
        "weather": "Weather",
        "alerts": "Alerts",
    }

    badges = []
    # Allow-list, not iterate-all: the freshness dict has grown metadata keys
    # (artifact_excluded #315, latest_data) that are not sources and were
    # leaking into the header as "Artifact_Excluded · just now" chips. Mirrors
    # the explicit source sets in update_fallback_banner / update_header_freshness.
    for source in ("demand", "weather", "alerts"):
        if source not in freshness:
            continue
        label = source_labels[source]
        conf = data_confidence_level(freshness[source], age_seconds)
        badges.append(confidence_badge(label, conf, age_text))

    return html.Div(
        badges,
        id="widget-confidence-bar",
        style={
            "display": "flex",
            "flexWrap": "wrap",
            "gap": "4px",
            "padding": "6px 0",
            "marginBottom": "8px",
        },
    )
