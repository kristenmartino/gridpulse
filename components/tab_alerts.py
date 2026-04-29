"""Tab: Risk (formerly Extreme Events / Tab 5).

R4b of shell-redesign-v2.md. Restructures the prior 5-section sprawl
(stress KPI card + alerts panel + weather context + anomaly + temp
exceedance + history timeline) into the v2 linear-stack rhythm:
title → 3-up Risk metrics → severity timeline → hero anomaly chart →
secondary chart pair → InsightCard → footer.

All existing component IDs (tab5-stress-score, tab5-stress-label,
tab5-stress-breakdown, tab5-alerts-list, tab5-weather-context,
tab5-anomaly-chart, tab5-temp-exceedance, tab5-timeline) are
preserved so the existing 8-output callback works untouched. The new
visual structure wraps the IDs in v2-styled containers + adds two
new slots (risk-title, risk-insight-card) for callback population.
"""

from dash import dcc, html

from components.cards import build_page_footer

_GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}


def _risk_metrics_bar() -> html.Div:
    """3-up Risk MetricsBar: Stress Score (hero) / Stress Label / Component breakdown.

    Reuses the existing tab5-stress-* IDs as inner spans so the existing
    Risk callback fills these without needing new outputs.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.Div("Grid Stress", className="gp-metric-label"),
                    html.Div(
                        [
                            html.Span(
                                id="tab5-stress-score",
                                children="—",
                                className="gp-metric-value gp-metric-value--hero tabular",
                            ),
                        ],
                        className="gp-metric-value-row",
                    ),
                    html.Div(
                        id="tab5-stress-label",
                        children="Normal",
                        className="gp-metric-sub",
                    ),
                ],
                className="gp-metric-cell",
            ),
            html.Div(
                [
                    html.Div("Components", className="gp-metric-label"),
                    html.Div(
                        id="tab5-stress-breakdown",
                        className="gp-metric-stress-breakdown",
                    ),
                ],
                className="gp-metric-cell gp-metric-cell--wide",
            ),
        ],
        className="gp-metrics-bar gp-metrics-bar--risk",
    )


def _alerts_card() -> html.Div:
    """Severity timeline. Existing alerts list rendered with v2 panel chrome
    + border-left severity emphasis (handled by .alert-card CSS already)."""
    return html.Div(
        [
            html.Div(
                [
                    html.Span("Active Weather Alerts", className="gp-panel__eyebrow"),
                    html.Span(
                        "Severity-ordered, most recent first",
                        className="gp-panel__hint",
                    ),
                ],
                className="gp-panel__header",
            ),
            html.Div(
                id="tab5-alerts-list",
                className="gp-severity-timeline",
                style={"maxHeight": "320px", "overflowY": "auto"},
            ),
        ],
        className="gp-panel",
    )


def _weather_strip() -> html.Div:
    """Current weather conditions strip. Existing callback populates
    tab5-weather-context with a freshness-ready snapshot."""
    return html.Div(
        [
            html.Div("Current Conditions", className="gp-control-eyebrow"),
            html.Div(id="tab5-weather-context", className="gp-weather-strip"),
        ],
        className="gp-control",
    )


def _hero_anomaly_chart() -> html.Div:
    return html.Div(
        dcc.Loading(
            dcc.Graph(
                id="tab5-anomaly-chart",
                style={"height": "320px"},
                config=_GRAPH_CONFIG,
            ),
            type="circle",
            color="#3b82f6",
        ),
        className="gp-chart-card",
    )


def _secondary_chart_grid() -> html.Div:
    """2-up grid: temperature exceedance + historical events timeline."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div("Temperature Exceedance", className="gp-control-eyebrow"),
                    dcc.Loading(
                        dcc.Graph(
                            id="tab5-temp-exceedance",
                            style={"height": "260px"},
                            config=_GRAPH_CONFIG,
                        ),
                        type="circle",
                        color="#3b82f6",
                    ),
                ],
                className="gp-chart-card",
            ),
            html.Div(
                [
                    html.Div("Historical Events", className="gp-control-eyebrow"),
                    dcc.Loading(
                        dcc.Graph(
                            id="tab5-timeline",
                            style={"height": "260px"},
                            config=_GRAPH_CONFIG,
                        ),
                        type="circle",
                        color="#3b82f6",
                    ),
                ],
                className="gp-chart-card",
            ),
        ],
        className="gp-risk-secondary",
    )


def layout() -> html.Div:
    """Build the v2 linear-stack Risk tab layout."""
    return html.Div(
        [
            html.Div(
                [
                    # 1. Title block (callback fills risk-title)
                    html.Div(id="risk-title"),
                    # 2. Risk MetricsBar (Stress Score hero + Components breakdown)
                    _risk_metrics_bar(),
                    # 3. Active alerts severity timeline
                    _alerts_card(),
                    # 4. Current weather conditions strip
                    _weather_strip(),
                    # 5. Hero anomaly detection chart
                    _hero_anomaly_chart(),
                    # 6. Secondary 2-up: temperature exceedance + timeline
                    _secondary_chart_grid(),
                    # 7. InsightCard (callback-filled)
                    html.Div(id="risk-insight-card", className="gp-insight-card-slot"),
                    # 8. Footer
                    build_page_footer(),
                ],
                className="gp-section-stack",
            ),
        ],
        className="gp-page",
    )
