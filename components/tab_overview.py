"""
Tab 0: Overview — Landing page with system-at-a-glance.

Shows persona greeting, top KPIs, demand sparkline, alert summary,
data freshness status, and quick-nav cards to priority tabs.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html


def layout() -> html.Div:
    """Build Tab 0 (Overview) layout."""
    return html.Div(
        [
            # Row 1: Persona greeting + data status
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(id="overview-greeting", className="welcome-card"),
                        md=8,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.P("DATA STATUS", className="kpi-label"),
                                html.Div(id="overview-freshness-badges"),
                                html.P(
                                    id="overview-last-updated",
                                    children="Last updated: --",
                                    style={
                                        "fontSize": "0.75rem",
                                        "color": "#8a8fa8",
                                        "marginTop": "8px",
                                    },
                                ),
                            ],
                            className="kpi-card",
                            style={"minHeight": "120px"},
                        ),
                        md=4,
                    ),
                ],
                className="g-2 mb-2",
            ),
            # Row 2: Persona KPIs
            html.Div(id="overview-kpi-row"),
            # Row 3: Demand sparkline + alerts summary
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.Span(
                                    "Demand (Last 24h)",
                                    style={
                                        "fontSize": "0.85rem",
                                        "fontWeight": "bold",
                                        "color": "#e0e0e0",
                                    },
                                ),
                                dcc.Loading(
                                    dcc.Graph(
                                        id="overview-demand-sparkline",
                                        style={"height": "200px"},
                                        config={
                                            "displayModeBar": False,
                                            "responsive": True,
                                        },
                                    ),
                                    type="circle",
                                    color="#00d4aa",
                                ),
                            ],
                            className="chart-container",
                        ),
                        md=7,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.P("ACTIVE ALERTS", className="kpi-label"),
                                html.H2(
                                    id="overview-alerts-count",
                                    children="Loading...",
                                    className="kpi-value",
                                ),
                                html.Div(id="overview-alerts-breakdown"),
                            ],
                            className="kpi-card",
                            style={"minHeight": "200px"},
                        ),
                        md=5,
                    ),
                ],
                className="g-2 mt-1",
            ),
            # Row 4: Quick navigation cards
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(id="overview-nav-cards"),
                        md=12,
                    ),
                ],
                className="g-2 mt-1",
            ),
        ]
    )
