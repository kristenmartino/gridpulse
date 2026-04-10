"""
Tab 1: Historical Demand.

Shows actual historical demand data with weather overlay option.
"""

import dash_bootstrap_components as dbc
from dash import html

from components.cards import build_chart_container


def _section_header(title: str, subtitle: str) -> html.Div:
    """Render a lightweight section header."""
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


def layout() -> html.Div:
    """Build Tab 1 (Historical Demand) layout."""
    return html.Div(
        [
            # ── Controls ────────────────────────────────────
            _section_header("Controls", "Time range and overlay options"),
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
            # ── Key Metrics ──────────────────────────────
            _section_header("Key Metrics", "Demand statistics for selected period"),
            # Bottom row: KPI cards
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.P("PEAK DEMAND", className="kpi-label"),
                                    html.H4(
                                        id="tab1-peak-value",
                                        children="Loading...",
                                        className="kpi-value",
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
                                        id="tab1-mape-value",
                                        children="Loading...",
                                        className="kpi-value",
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
