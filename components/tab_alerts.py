"""
Tab 5: Extreme Events & Alerts (Maria's primary view).

Components:
- Active NOAA weather alerts panel
- Demand anomaly detection chart
- Temperature exceedance forecast
- Stress indicator (combined metric)
- Historical extreme events timeline
"""

import dash_bootstrap_components as dbc
from dash import html

from components.cards import build_chart_container


def layout() -> html.Div:
    """Build Tab 5 layout."""
    return html.Div(
        [
            # Top: stress indicator + active alerts
            dbc.Row(
                [
                    dbc.Col(
                        [
                            # Stress indicator gauge
                            html.Div(
                                [
                                    html.P("GRID STRESS INDICATOR", className="kpi-label"),
                                    html.H2(
                                        id="tab5-stress-score",
                                        children="Loading...",
                                        className="kpi-value",
                                    ),
                                    html.P(
                                        id="tab5-stress-label",
                                        children="Normal",
                                        className="kpi-delta neutral",
                                    ),
                                    html.Hr(style={"borderColor": "#0f3460"}),
                                    html.P(
                                        "Components:",
                                        style={
                                            "fontSize": "0.75rem",
                                            "color": "#8a8fa8",
                                            "margin": "4px 0",
                                        },
                                    ),
                                    html.Div(id="tab5-stress-breakdown"),
                                ],
                                className="kpi-card",
                                style={"minHeight": "200px"},
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Span("Active Weather Alerts", className="chart-title"),
                                    html.Div(
                                        id="tab5-alerts-list",
                                        style={"maxHeight": "260px", "overflowY": "auto"},
                                    ),
                                ],
                                className="chart-container",
                                style={"minHeight": "200px"},
                            ),
                        ],
                        md=9,
                    ),
                ],
                className="g-2",
            ),
            # Middle: anomaly detection + temperature exceedance
            dbc.Row(
                [
                    dbc.Col(
                        build_chart_container(
                            "tab5-anomaly-chart", "Demand Anomaly Detection", height="300px"
                        ),
                        md=7,
                    ),
                    dbc.Col(
                        build_chart_container(
                            "tab5-temp-exceedance",
                            "Temperature Exceedance Forecast",
                            height="300px",
                        ),
                        md=5,
                    ),
                ],
                className="g-2 mt-1",
            ),
            # Bottom: historical timeline
            dbc.Row(
                [
                    dbc.Col(
                        build_chart_container(
                            "tab5-timeline", "Historical Extreme Events", height="280px"
                        ),
                        md=12,
                    ),
                ],
                className="g-2 mt-1",
            ),
        ]
    )
