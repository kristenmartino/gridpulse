"""
Tab 0: Overview — mission-control screen for the GridPulse platform.

Sections:
A. Greeting + AI Executive Briefing (persona-specific)
B. Data Health (per-source freshness badges)
C. Quick Navigation — pathways into Forecast, Risk, and Scenarios
D. Spotlight Chart (persona-relevant metric) + Insight Digest (cross-tab)
E. Grid Signals / Energy News Feed
"""

import dash_bootstrap_components as dbc
from dash import dcc, html


def layout() -> html.Div:
    """Build Overview (mission-control) layout."""
    return html.Div(
        [
            # ── Section A: Greeting + AI Executive Briefing ───────
            dbc.Row(
                dbc.Col(
                    html.Div(id="overview-greeting"),
                    md=12,
                ),
                className="g-2 mb-2",
            ),
            html.Div(
                id="overview-briefing",
                children=_skeleton_briefing(),
                className="briefing-card",
            ),
            # ── Section A.5: What Changed Since Last Visit (NEXD-8) ──
            html.Div(id="overview-changes", className="mt-2"),
            # ── Section B: Data Health ────────────────────────────
            html.Div(id="overview-data-health", className="mt-2"),
            # ── Section C: Quick Navigation ───────────────────────
            _quick_nav(),
            # ── Section D: Spotlight Chart + Insight Digest ───────
            _section_header("Spotlight", "Persona-relevant metric at a glance"),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            dcc.Loading(
                                dcc.Graph(
                                    id="overview-spotlight-chart",
                                    style={"height": "280px"},
                                    config={
                                        "displayModeBar": False,
                                        "responsive": True,
                                    },
                                ),
                                type="circle",
                                color="#38D0FF",
                            ),
                            className="chart-container",
                        ),
                        md=8,
                    ),
                    dbc.Col(
                        html.Div(
                            id="overview-insight-digest",
                            className="insight-digest",
                        ),
                        md=4,
                    ),
                ],
                className="g-2",
            ),
            # ── Section E: Grid Signals ───────────────────────────
            _section_header("Grid Signals", "Energy news and market context"),
            html.Div(id="overview-news-feed"),
        ]
    )


def _section_header(title: str, subtitle: str) -> html.Div:
    """Render a lightweight section header for Overview content areas."""
    return html.Div(
        [
            html.Span(
                title,
                style={
                    "color": "#F7FAFC",
                    "fontSize": "0.85rem",
                    "fontWeight": "600",
                    "marginRight": "8px",
                },
            ),
            html.Span(
                subtitle,
                style={
                    "color": "#A8B3C7",
                    "fontSize": "0.75rem",
                },
            ),
        ],
        style={
            "padding": "10px 0 4px 0",
            "borderBottom": "1px solid #263556",
            "marginTop": "12px",
            "marginBottom": "8px",
        },
    )


def _quick_nav() -> html.Div:
    """Render quick-navigation cards linking to Forecast, Risk, and Scenarios."""
    nav_items = [
        {
            "label": "Forecast",
            "tab_id": "tab-outlook",
            "icon": "trending_up",
            "desc": "Demand predictions & confidence",
            "color": "#38D0FF",
        },
        {
            "label": "Risk",
            "tab_id": "tab-alerts",
            "icon": "warning",
            "desc": "Alerts & extreme conditions",
            "color": "#FF5C7A",
        },
        {
            "label": "Grid",
            "tab_id": "tab-generation",
            "icon": "bolt",
            "desc": "Generation mix & net load",
            "color": "#2DE2C4",
        },
        {
            "label": "Scenarios",
            "tab_id": "tab-simulator",
            "icon": "tune",
            "desc": "What-if analysis & presets",
            "color": "#4A7BFF",
        },
    ]
    cards = []
    for item in nav_items:
        cards.append(
            dbc.Col(
                html.Div(
                    [
                        html.Div(
                            item["label"],
                            style={
                                "color": item["color"],
                                "fontWeight": "600",
                                "fontSize": "0.85rem",
                            },
                        ),
                        html.Div(
                            item["desc"],
                            style={
                                "color": "#A8B3C7",
                                "fontSize": "0.72rem",
                                "marginTop": "2px",
                            },
                        ),
                    ],
                    className="overview-quick-nav-card",
                    style={
                        "background": "#11182D",
                        "border": "1px solid #263556",
                        "borderTop": f"2px solid {item['color']}",
                        "borderRadius": "6px",
                        "padding": "10px 14px",
                        "cursor": "default",
                    },
                ),
                md=3,
                sm=6,
            )
        )
    return dbc.Row(cards, className="g-2 mt-2")


def _skeleton_briefing() -> html.Div:
    """Pulsing skeleton placeholder while briefing loads."""
    return html.Div(
        [
            html.Div(className="skeleton-line skeleton-line-long"),
            html.Div(className="skeleton-line skeleton-line-long"),
            html.Div(className="skeleton-line skeleton-line-medium"),
        ],
        className="skeleton-pulse",
    )
