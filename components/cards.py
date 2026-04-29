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
    color: str = "#3b82f6",
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
    from components.icons import icon

    icon_name = {
        "critical": "alert-triangle",
        "warning": "alert-circle",
        "info": "info",
    }.get(severity, "info")

    return html.Div(
        [
            html.Div(
                [
                    icon(
                        icon_name,
                        size="sm",
                        className=f"alert-card__icon alert-card__icon--{severity}",
                    ),
                    html.Strong(event, className="alert-card__title"),
                ],
                className="alert-card__header",
            ),
            html.P(headline, className="alert-card__body"),
            html.Small(f"Expires: {expires}", className="alert-card__expires") if expires else None,
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
                color="#3b82f6",
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


def section_header(title: str, subtitle: str = "") -> html.Div:
    """Standardized section title used across tabs.

    Renders a quiet divider-label pair (Notion/Linear style) via the
    shared ``.section-header`` CSS rule. Prefer this over inline-styled
    section headers in tab layouts.
    """
    children: list = [html.Span(title, className="section-header__title")]
    if subtitle:
        children.append(html.Span(subtitle, className="section-header__subtitle"))
    return html.Div(children, className="section-header")


# ── v2-style components (R2 of shell-redesign-v2.md) ─────────────────
#
# These mirror the gridpulse-v2 dashboard rhythm: max-w-5xl content width,
# space-y-8 between sections, p-5 card padding, borders (not shadows) for
# separation, Geist-style 10px eyebrow / 13px body / 2xl hero typography.


def build_page_title(
    title: str,
    subtitle: str | None = None,
    freshness_chip: html.Span | None = None,
) -> html.Div:
    """Page-level title block: h1 + 1-line subtitle + optional freshness chip.

    Mirrors gridpulse-v2 dashboard/page.tsx:148-155.

    Args:
        title: Region name or page title (e.g., "Florida Power & Light").
        subtitle: 1-line descriptive context.
        freshness_chip: Optional small chip rendered to the right of the title.
    """
    title_row: list = [html.H1(title, className="gp-page-title__heading")]
    if freshness_chip is not None:
        title_row.append(freshness_chip)
    children: list = [html.Div(title_row, className="gp-page-title__row")]
    if subtitle:
        children.append(html.P(subtitle, className="gp-page-title__subtitle"))
    return html.Div(children, className="gp-page-title")


def build_metrics_bar(items: list[dict]) -> html.Div:
    """5-up KPI bar with vertical dividers (gridpulse-v2 MetricsBar).

    Each item: ``{"label": str, "value": str, "unit": str | None,
                  "tone": "primary" | "secondary" | "positive" | "negative" | None,
                  "hero": bool}``.

    Mirrors gridpulse-v2 components/MetricsBar.tsx:34. Up to 5 cells.
    Uses ``.tabular`` on values for aligned numerics.
    """
    cells = []
    for item in items:
        tone = item.get("tone", "primary")
        hero = item.get("hero", False)
        unit = item.get("unit")
        value_classes = ["gp-metric-value", "tabular"]
        if hero:
            value_classes.append("gp-metric-value--hero")
        if tone in ("positive", "negative"):
            value_classes.append(f"gp-metric-value--{tone}")
        elif tone == "secondary":
            value_classes.append("gp-metric-value--secondary")
        cell_children: list = [
            html.Div(item.get("label", ""), className="gp-metric-label"),
            html.Div(
                [
                    html.Span(item.get("value", "—"), className=" ".join(value_classes)),
                    html.Span(unit, className="gp-metric-unit") if unit else None,
                ],
                className="gp-metric-value-row",
            ),
        ]
        cells.append(html.Div(cell_children, className="gp-metric-cell"))
    return html.Div(cells, className="gp-metrics-bar")


def build_model_metrics_card(
    model_name: str,
    metrics: dict[str, str],
    badge: str | None = None,
) -> html.Div:
    """Horizontal model-performance bar (top/bottom borders, no card chrome).

    Mirrors gridpulse-v2 components/ModelMetricsCard.tsx:22.

    Args:
        model_name: Display name (e.g., "Ensemble").
        metrics: Ordered dict of metric_label → formatted_value (e.g.,
            ``{"MAPE": "1.9%", "RMSE": "340 MW", "MAE": "250 MW", "R²": "0.979"}``).
            Insertion order is preserved — render order matches.
        badge: Optional small badge text rendered next to model name
            (e.g., "v3.2" or "trained").
    """
    left: list = [
        html.Span("Model", className="gp-model-card__eyebrow"),
        html.Span(model_name, className="gp-model-card__name"),
    ]
    if badge:
        left.append(html.Span(badge, className="gp-model-card__badge"))

    metric_cells = [
        html.Div(
            [
                html.Span(label, className="gp-model-card__metric-label"),
                html.Span(value, className="gp-model-card__metric-value tabular"),
            ],
            className="gp-model-card__metric",
        )
        for label, value in metrics.items()
    ]

    return html.Div(
        [
            html.Div(left, className="gp-model-card__left"),
            html.Div(metric_cells, className="gp-model-card__metrics"),
        ],
        className="gp-model-card",
    )


def build_insight_card(
    eyebrow: str,
    body: list | str,
) -> html.Div:
    """Narrative summary block: small eyebrow caption + relaxed-leading body.

    Mirrors gridpulse-v2 components/InsightCard.tsx:43.

    Args:
        eyebrow: Short uppercase label (e.g., "Summary", "Outlook").
        body: Paragraph text or a list of inline children (Spans for
            semantic colored deltas, plain text strings for the rest).
    """
    paragraph_children = body if isinstance(body, list) else [body]
    return html.Div(
        [
            html.Div(eyebrow, className="gp-insight-card__eyebrow"),
            html.P(paragraph_children, className="gp-insight-card__body"),
        ],
        className="gp-insight-card",
    )


def build_page_footer(
    sources: list[str] | None = None,
    note: str | None = None,
) -> html.Div:
    """Small attribution footer at the bottom of the linear stack."""
    sources = sources or ["EIA", "Open-Meteo", "NOAA"]
    parts: list = [html.Span(" · ".join(sources), className="gp-footer__sources")]
    if note:
        parts.append(html.Span(note, className="gp-footer__note"))
    return html.Div(parts, className="gp-footer")
