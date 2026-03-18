"""
Tab 1: Historical Demand.

Shows actual historical demand data with weather overlay option.
"""

import dash_bootstrap_components as dbc
from dash import html

from components.cards import build_chart_container


def layout() -> html.Div:
    """Build Tab 1 (Historical Demand) layout."""
    return html.Div(
        [
            # Controls row
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Checklist(
                                id="tab1-weather-overlay",
                                options=[
                                    {"label": " Weather Overlay (Temperature)", "value": "temp"}
                                ],
                                value=[],
                                inline=True,
                                className="mt-1",
                                style={"fontSize": "0.85rem"},
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.Label(
                                "Time Range:",
                                className="fw-bold me-2",
                                style={"fontSize": "0.85rem"},
                            ),
                            dbc.RadioItems(
                                id="tab1-timerange",
                                options=[
                                    {"label": "24h", "value": "24"},
                                    {"label": "7 days", "value": "168"},
                                    {"label": "30 days", "value": "720"},
                                    {"label": "90 days", "value": "2160"},
                                ],
                                value="168",
                                inline=True,
                                className="mt-1",
                                style={"fontSize": "0.85rem"},
                            ),
                        ],
                        md=6,
                    ),
                ],
                className="mb-2",
            ),
            # Hidden model toggle (keep for callback compatibility)
            html.Div(
                dbc.Checklist(
                    id="tab1-model-toggle",
                    options=[],
                    value=[],
                ),
                style={"display": "none"},
            ),
            # Main chart
            build_chart_container("tab1-forecast-chart", "Historical Demand", height="420px"),
            # AI insight card
            html.Div(id="tab1-insight-card"),
            # Bottom row: KPI cards
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.P("PEAK DEMAND", className="kpi-label"),
                                    html.H4(
                                        id="tab1-peak-value", children="— MW", className="kpi-value"
                                    ),
                                    html.P(
                                        id="tab1-peak-time",
                                        children="",
                                        className="kpi-delta neutral",
                                    ),
                                ],
                                className="kpi-card",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.P("AVERAGE DEMAND", className="kpi-label"),
                                    html.H4(
                                        id="tab1-mape-value", children="— MW", className="kpi-value"
                                    ),
                                    html.P("Over selected period", className="kpi-delta neutral"),
                                ],
                                className="kpi-card",
                            ),
                        ],
                        md=6,
                    ),
                ],
                className="g-2 mt-1",
            ),
            # Hidden elements for callback compatibility
            html.Div(
                [
                    html.Div(id="tab1-reserve-value"),
                    html.Div(id="tab1-reserve-status"),
                    html.Div(id="tab1-alerts-count"),
                    html.Div(id="tab1-alerts-summary"),
                ],
                style={"display": "none"},
            ),
        ]
    )
