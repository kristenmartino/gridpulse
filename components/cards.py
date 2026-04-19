"""
Reusable card components: KPI cards, welcome cards, alert cards.

All card builders return Dash HTML/Bootstrap components.
"""

import dash_bootstrap_components as dbc
from dash import html


def build_kpi_card(
    label: str,
    value: str,
    delta: str | None = None,
    delta_direction: str = "neutral",
) -> dbc.Col:
    """
    Build a single KPI card.

    Args:
        label: Metric name (e.g., "Peak Demand").
        value: Formatted value (e.g., "28,450 MW").
        delta: Optional delta string (e.g., "↑6% vs yesterday").
        delta_direction: "positive", "negative", or "neutral".
    """
    delta_el = html.P(delta, className=f"kpi-delta {delta_direction}") if delta else None

    return dbc.Col(
        html.Div(
            [
                html.P(label, className="kpi-label"),
                html.H3(value, className="kpi-value"),
                delta_el,
            ],
            className="kpi-card",
        ),
        xs=6,
        sm=6,
        md=3,
        lg=3,
    )


def build_kpi_row(kpis: list[dict]) -> dbc.Row:
    """
    Build a row of KPI cards.

    Args:
        kpis: List of dicts with keys: label, value, delta (optional), direction (optional).
    """
    cards = []
    for kpi in kpis:
        cards.append(
            build_kpi_card(
                label=kpi.get("label", ""),
                value=kpi.get("value", "No data"),
                delta=kpi.get("delta"),
                delta_direction=kpi.get("direction", "neutral"),
            )
        )
    return dbc.Row(cards, className="kpi-row g-2")


def build_welcome_card(
    title: str,
    message: str,
    avatar: str = "👋",
    color: str = "#38D0FF",
) -> html.Div:
    """
    Build a persona-specific welcome card.

    Args:
        title: Card title (e.g., "Grid Operations Dashboard").
        message: Contextual greeting with data highlights.
        avatar: Emoji avatar.
        color: Left border color.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.Span(avatar, style={"fontSize": "1.3rem", "marginRight": "8px"}),
                    html.Span(title, className="welcome-title"),
                ],
                style={"display": "flex", "alignItems": "center"},
            ),
            html.P(message, className="welcome-message"),
        ],
        className="welcome-card",
        style={"borderLeftColor": color},
    )


def build_alert_card(
    event: str,
    headline: str,
    severity: str = "info",
    expires: str | None = None,
) -> html.Div:
    """
    Build a weather alert card.

    Args:
        event: Alert type (e.g., "Heat Advisory").
        headline: Short description.
        severity: "critical", "warning", or "info".
        expires: Optional expiry time string.
    """
    icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(severity, "ℹ️")

    return html.Div(
        [
            html.Div(
                [
                    html.Span(f"{icon} ", style={"fontSize": "1rem"}),
                    html.Strong(event, style={"color": "#ffffff"}),
                ],
            ),
            html.P(
                headline, style={"margin": "4px 0 0 0", "fontSize": "0.8rem", "color": "#A8B3C7"}
            ),
            html.Small(f"Expires: {expires}", style={"color": "#A8B3C7"}) if expires else None,
        ],
        className=f"alert-card {severity}",
    )


def build_chart_container(
    chart_id: str,
    title: str,
    height: str = "400px",
    freshness: str | None = None,
) -> html.Div:
    """
    Wrap a Plotly Graph in a styled container with title and optional freshness badge.
    """
    from dash import dcc

    header_items = [html.Span(title, className="chart-title")]
    if freshness:
        # Map freshness token → badge CSS class. "warming" is a distinct
        # degraded-but-expected state surfaced in Redis-only deployments.
        badge_class = {
            "fresh": "fresh",
            "stale": "stale",
            "warming": "warming",
            "demo": "stale",
        }.get(freshness, "expired")
        header_items.append(
            html.Span(
                freshness, className=f"freshness-badge {badge_class}", style={"marginLeft": "8px"}
            )
        )

    return html.Div(
        [
            html.Div(
                header_items,
                style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
            ),
            dcc.Loading(
                dcc.Graph(
                    id=chart_id,
                    style={"height": height},
                    config={"displayModeBar": True, "responsive": True},
                ),
                type="circle",
                color="#38D0FF",
            ),
        ],
        className="chart-container",
    )


def build_news_card(
    title: str,
    source: str,
    published_at: str,
    url: str,
    description: str | None = None,
) -> html.Div:
    """
    Build a compact news card for the horizontal ribbon.

    Args:
        title: Article headline.
        source: News source name.
        published_at: Publication timestamp.
        url: Link to full article.
        description: Unused (kept for API compatibility).
    """
    from datetime import datetime

    try:
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        time_str = dt.strftime("%b %d, %H:%M")
    except (ValueError, AttributeError):
        time_str = published_at[:16] if published_at else ""

    return html.A(
        html.Div(
            [
                html.Div(title, className="news-title"),
                html.Div(
                    f"{source} · {time_str}",
                    className="news-meta",
                ),
            ],
        ),
        href=url,
        target="_blank",
        rel="noopener noreferrer",
        className="news-ribbon-card",
    )


def build_news_feed(articles: list[dict]) -> html.Div:
    """
    Build an auto-scrolling news ticker with compact article cards.

    Args:
        articles: List of article dicts from news_client.
    """
    if not articles:
        return html.Div(
            html.P(
                "No news available",
                style={"color": "#A8B3C7", "textAlign": "center", "padding": "20px"},
            ),
            className="news-ribbon",
        )

    cards = [
        build_news_card(
            title=article.get("title", ""),
            source=article.get("source", ""),
            published_at=article.get("published_at", ""),
            url=article.get("url", "#"),
            description=article.get("description"),
        )
        for article in articles[:10]
    ]

    # Duplicate cards for seamless looping
    return html.Div(
        [
            html.Div("Grid Signals", className="news-ribbon-header"),
            html.Div(
                html.Div(cards + cards, className="news-ticker-track"),
                className="news-ticker-viewport",
            ),
        ],
        className="news-ribbon",
    )
